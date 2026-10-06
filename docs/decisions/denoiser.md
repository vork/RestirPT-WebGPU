# Denoiser (M5.5): A-SVGF-lite design

Status: **design for M5.5** (branch `m55-denoiser`, from `m5-temporal` at 6831032). The implementation follows this
note; deviations found while implementing are appended to the Changelog at the end (append-only numbering).

Sources: PLAN §3 step 7, §5 M5.5, §6 (Denoiser rows), §1.10, T16 of §7.4; plan-review (`docs/research/plan-review.json`,
"Move A-SVGF-lite to a new M5.5": acceptance criteria); `docs/research/restir-practice-dynamics.md` (SVGF / A-SVGF
constants); `restir-temporal-api.md` (TD3, TD12 q′ rule, TD19/TD20 resets, D-1…D-4, TD-I1, §2.8 tState);
`platform-lanes.md` Q1–Q4; `data-formats.md` (compact, precision-appropriate formats).

Precedence as everywhere: PLAN > gap-* > verify > critique > reports > this note's own choices. Every choice this note
makes on its own is marked **(own)** with its reason.

---

## 0. Decisions at a glance

| # | Decision | Why |
|---|---|---|
| DN1 | **A-SVGF-lite** = demodulation by the primary albedo → temporal accumulation of colour and luminance moments (bilinear reprojection, TD12 predicate) → variance estimate (temporal, or 7×7 spatial for short histories) → N à-trous iterations (variance-guided, edge-stopping on depth, normal, luminance) → remodulation + the length-1 term. History α is driven by a **ReSTIR temporal gradient** (DN6). | PLAN §5 M5.5; SVGF (Schied 2017) and A-SVGF (Schied 2018) as summarised in restir-practice-dynamics.md. |
| DN2 | **Input** = the per-frame (1-frame) estimate, never the progressive mean: ReSTIR `rsFrame` (L1 + estimate) and `rsL1`; PT: the 1-spp frame sample (the PT's accumulation is switched off while the denoiser is on). Only `rsFrame − rsL1` is demodulated and filtered; L1 (emitters / env seen directly) is added back unfiltered. Background pixels (no primary hit) pass through. | plan-review acceptance: "denoised 1-frame ReSTIR"; PLAN §1.7 background rule. |
| DN3 | `accumulate` **|** `denoise` (PLAN §3 step 7): with the denoiser active the displayed colour is the denoised frame. ReSTIR's finalize keeps its own progressive mean running unseen (toggling back is seamless); the PT's progressive mean restarts. | PLAN §3. |
| DN4 | **Off in every validation path.** The denoiser exists only inside `Renderer` (the interactive app). Validation runners never build a `Renderer`; a page-global registry of live denoisers is checked before/after every harness readback and written into `meta.json` (`t16.denoiser`); the gates assert `'none'` (T16). The app forces it off in **ReSTIR-unbiased** (the validation-mode preset). | PLAN §7.4 T16; plan-review acceptance. |
| DN5 | **Reprojection** = TD12's q′ geometry, made deterministic and bilinear: back-project the current jittered primary hit x₁ with `prevCam` (`frame_project − 0.5`, pixel centres at integers); the 2×2 taps around it are valid iff a previous hit exists, `dot(n, n′) ≥ 0.5` and `|‖x₁ − o_{t−1}‖ − z′| ≤ 0.1·z′` (TD12's predicate); fallback = the 3×3 ring around the rounded position (valid taps averaged); none ⇒ disoccluded. | "consistent with q′"; bilinear instead of stochastic rounding because the denoiser history is a filtered image, not a sample. |
| DN6 | **α from the ReSTIR temporal gradient**: per pixel, the same path evaluated in frame t and frame t−1, read from the temporal stage's tState (forward shift of X_p: F_t(Y_p)·J_p·W_p vs the stored F_p·W_p; inverse shift of X_c: π_c vs π_p(X_c)); 8×8 tile sums, a 3×3-tile window, λ = \|ΣΔ\|/ΣM. λ is **gated by the change bits**: λ := 0 on frames with `TF_LIGHTS_SAME ∧ TF_ENV_SAME` (static lighting). | A-SVGF's gradient, without re-tracing: ReSTIR already re-evaluates last frame's sample under the new lighting (T1/T2) and the current sample under the old lighting (T4). |
| DN7 | **History control**: λ′ = clamp((λ − λ₀)/(λ₁ − λ₀), 0, 1); n ← 1 + (1 − λ′)·n_reproj; α = max(α_min, 1/n, λ′). Full resets: ReSTIR temporal resets (`!histValid`: config hash, scene, resize, env-map swap, explicit reset, freeze seed/frame/history), denoiser setting changes; in modes without a ReSTIR gradient (PT, temporal off) every accumulation restart (`FRAME_RESET_HISTORY`: light / env edits). | A-SVGF α′ = (1 − λ)α + λ, plus a history-length cut so that recovery after a detected change does not wait on α_min. |
| DN8 | **Formats** (§2): rgba16float for demodulated colour, à-trous ping-pong and moments (stored as EMA mean and **standard deviation**, not the raw second moment), rg32uint guides (f32 camera distance + oct 2×16 shading normal), rg32float tile sums. | data-formats.md; the user's format rule (no fp16 for unit vectors). |
| DN9 | **Idempotent frame**: every denoiser pass is a pure function of its inputs (ping-pong histories, parity flips only on advanced frames). A paused (held) frame re-runs the denoiser with the same parity and reproduces the last output bit for bit; the HUD timing re-runs the frame's denoiser passes in a **separate timing submit** with timestamp writes right after the frame's submit (Q3). | TD20 (held frames), Q3 (no timestamp writes on or around ReSTIR passes). |
| DN10 | **Code-hash isolation** (own): the denoiser's WGSL lives in `src/core/render/denoise/shaders/` (composed with the shared sources), and no file of the validation runners' TS closures or of `src/core/shaders/` changes. | The M4 ReSTIR and M5 chain cache keys hash every WGSL file under `src/core/shaders` plus the runners' TS closures; touching them would force hours of re-rendered references for no change in what they render. |

---

## 1. Pass graph

Per **advanced** frame, encoded by `Renderer.encode` right after the frame's ReSTIR finalize (or the PT pass), in the
same encoder and submit:

```
ReSTIR temporal frames only (gradient available, §4):
  dn_gradient        8×8 wg, per pixel → workgroup sums        tState (arena ro), res[w] plane 0 (ro), dnMom[prev].a
                                                               input, L1, dnHist[prev], dnAlb[prev] (colour family, DN-11)
                                                               → dnGradTile (rgba32float, W/8 × H/8): (ΣΔ_f, ΣM_f, ΣΔ_i, ΣM_i)
                                                               → dnGradTile2 (rgba32float): (Σcur, Σold, Σcur², N)
  dn_grad_filter     8×8 wg over tiles                         dnGradTile, dnGradTile2 → dnLambda (r32float, W/8 × H/8): λ of a
                                                               3×3-tile window (inverse family: (2·invRadius + 1)², DN-10)
PT only:
  copy colour → dnInput                                        (the PT writes its 1-spp sample into the colour target)
every mode:
  dn_temporal        8×8 wg                                    G-buffer (albedo, ns, pos, flags), input (rsFrame | dnInput),
                                                               rsL1 (| zero), dnGeo[prev], dnHist[prev], dnMom[prev], dnLambda,
                                                               res[final] plane 0 (ReSTIR)
                                                               → dnAtrous[0] (colour, temporal variance), dnHist[cur] (integrated
                                                                 colour, overwritten by à-trous 0's output), dnMom[cur], dnGeo[cur]
  dn_variance        8×8 wg                                    dnAtrous[0], dnMom[cur], dnGeo[cur], dnAlb[cur], dnL1[cur], dnTaa[prev]
                                                               → dnAtrous[1] (n < 4: 7×7 bilateral spatial variance; else copy),
                                                               dnLumG (3×3 prefiltered luminance guide of à-trous 0, DN-13),
                                                               dnTap (ā, DN-9 guide luminance: the per-tap data, DN-14)
  dn_atrous i        8×8 wg, i = 0 … N−1, step 2^i             dnAtrous[src], dnGeo[cur] → dnAtrous[dst]
                       i = 0 also writes dnHist[cur] (SVGF feedback of the first iteration)
                       i = N−1 remodulates: ā·filtered + L̄1 → dnOut (pass-through for background; Changelog DN-1/2/6)
  dn_resolve         8×8 wg                                    dnOut, dnTaa[prev], gbuf (motion) → dnTaa[cur], the colour target
                                                               (temporal resolve of the output, Changelog DN-6)
```

N = 0 is allowed (temporal accumulation only: `dn_variance` then writes the colour target). Default N = 4 (Changelog DN-15; SVGF uses 5);
the panel offers 0–6.

A **held frame** (paused, TD20; `RestirFramePass.encodeHold` re-displays the last ReSTIR estimate) runs the same passes
with the same parity: identical inputs ⇒ identical outputs (DN9). The PT colour copy is skipped on held frames.

---

## 2. Buffers and formats (960×540: 518 400 px)

| Name | Format | B/px | 540p | Content | Why this format |
|---|---|---|---|---|---|
| `dnHist[2]` | rgba16float | 2 × 8 | 8.3 MB | demodulated colour history (rgb; a unused) | fp16 keeps 11 significant bits (rel. 4.9·10⁻⁴): far below the Monte-Carlo noise and the 8-bit display after the view transform. Range 6.5·10⁴: the demodulated input is clamped to it (a biased display path; only firefly pixels reach it). |
| `dnMom[2]` | rgba16float | 2 × 8 | 8.3 MB | (μ_l, σ_l, n, FW): EMA mean and **standard deviation** of the demodulated luminance, history length n, lum(F·W) of the final reservoir (gradient input of the next frame) | The raw second moment would square the dynamic range (fp16 overflows at l > 256) and lose the variance to cancellation (m₂ − m₁² with 2⁻¹¹ relative error leaves a floor of 4.9·10⁻⁴·μ²). The West/Welford EMA update `δ = l − μ; μ += αδ; σ² = (1 − α)(σ² + αδ²)` runs in f32 registers and stores σ, which has the range and precision of the luminance. n ≤ 2048 is exact in fp16 (we cap at 64). FW is a radiance-like estimate (relative precision suffices; a clamp at 6.5·10⁴ only weakens the gradient of a firefly). |
| `dnAlb[2]` | rgba32float | 2 × 16 | 16.6 MB | accumulated demodulation factor ā, n_a (Changelog DN-1, DN-2) | a′ ∈ [0.02, 1.04]. A 1/n mean: its increments (down to 1/1024 of the value) are below binary16 resolution, and Metal's round-toward-zero f16 stores bias it (DN-7). |
| `dnGeo[2]` | rg32uint | 2 × 8 | 8.3 MB | x: f32 distance to the camera; y: oct 2×16 snorm shading normal | Distance in **f32**: the à-trous depth weight compares neighbour differences against the local depth gradient, which on grazing planes is ~10⁻³ of the distance (fp16 would quantise it). Normal **oct 2×16** (< 0.004° worst case with WGSL's round-to-nearest `pack2x16snorm`; data-formats.md §B3 quotes 0.0025° for an optimised encoder): never fp16 for unit vectors. One guide for the reprojection predicate (prev) and the à-trous edge stops (cur). |
| `dnAtrous[2]` | rgba16float | 2 × 8 | 8.3 MB | à-trous ping-pong: demodulated colour + variance | Colour as `dnHist`. The variance only steers the luminance edge weight; it saturates at 6.5·10⁴ (σ = 256: such pixels are then filtered by geometry only, which is the desired behaviour for extreme noise). |
| `dnInput` (PT only) | colour format | 16 | 8.3 MB | copy of the PT 1-spp sample | The PT writes its sample into the colour target, which the denoiser overwrites. |
| `dnL1[2]` | rgba32float | 2 × 16 | 16.6 MB | accumulated L1 (Changelog DN-2) | a 1/n mean of emitter / env radiance seen directly: f32 as `dnAlb` (DN-7). |
| `dnOut` | rgba16float | 8 | 4.1 MB | remodulated output of the last à-trous (DN-6) | display radiance, as `dnHist`. |
| `dnTaa[2]` | rgba32float | 2 × 16 | 16.6 MB | resolved output history (rgb, n_t ≤ 1024) (DN-6) | a 1/n_t mean: f32 as `dnAlb` (DN-7). |
| `dnTap` | rgba32uint | 16 | 8.3 MB | everything an à-trous tap reads besides its colour (DN-14, DN-15): x, y = the `dnGeo` texel (f32 distance, oct 2×16 normal); z, w = binary16 (ā.r, ā.g), (ā.b, DN-9 guide luminance or −1) | read by every tap of every iteration: one fetch. ā ∈ [0.02, 1.04] in binary16 is 2⁻¹² relative (≤ 2.5·10⁻⁴ absolute, 0.5 % of σ_a = 0.05); the guide luminance only enters a difference against σ_l·σ ≫ its binary16 step. The remodulation still reads the f32 `dnAlb`. |
| `dnLumG` | r32float | 4 | 2.1 MB | prefiltered luminance guide of à-trous 0 (DN-13) | written and read within the frame; f32 because it feeds a difference against σ_l·σ. |
| `dnGradTile` | rgba32float | 16 per tile | 130 KB | (ΣΔ_f, ΣM_f, ΣΔ_i, ΣM_i) over an 8×8 tile (DN-2) | Sums of up to 64 radiance-like values: f32 (tiny texture). |
| `dnGradTile2` | rgba32float | 16 per tile | 130 KB | (Σcur, Σold, Σcur², N) colour family over an 8×8 tile (DN-11) | as `dnGradTile`. |
| `dnLambda` | r32float | 4 per tile | 33 KB | λ of the 3×3-tile window | tiny. |
| params | uniform 96 B | – | – | settings, flags, tile sizes, arena offsets | – |

Total ≈ 98 MB at 540p (≈ 90 MB without the PT input copy; DN-7 made three 1/n accumulators rgba32float). The M1 G-buffer (80 B/px f32, `renderer.ts`) is read once
per pixel by `dn_temporal` (albedo, ns, pos, flags); data-formats.md P4 ("G-buffer 80 → 40 B in M5.5") is **not**
done here **(own)**: the G-buffer layout is shared with the M1–M3 debug views and the primary pass, and the denoiser
reads it once per pixel, so its cost to the denoiser is one 80-byte read; the à-trous iterations read only the compact
`dnGeo`.

---

## 3. Demodulation, remodulation

- a′ = max(albedo, 0.02) per channel, or 1 when max(albedo) < 0.02 (black or purely glossy V1 materials: no
  demodulation instead of a ×50 gain). albedo = `material_albedo` of the primary G-buffer (V1: diffuse colour; V2:
  base colour × texture × COLOR_0).
- c = clamp((L_frame − L1)/a′, 0, 65504) per channel (ReSTIR: L_frame = `rsFrame`, L1 = `rsL1`; PT: L1 = 0).
- Output = ā·filtered + L1 with ā the temporally accumulated a′ (Changelog DN-1; = a′ inside a surface, so the round
  trip is exact up to fp16 rounding there).

## 4. Reprojection (DN5)

For pixel q with a primary hit: x₁ = G-buffer `pos` (the jittered hit of this frame, the same ray as `rs_primary`),
sp = `frame_project(x₁, prevCam) − 0.5`, z = ‖x₁ − o_{t−1}‖.
- Bilinear taps c ∈ ⌊sp⌋ + {0,1}², weights w_c = (1 − |sp − c|) per axis. A tap is **valid** iff inside the image,
  the previous frame had a hit there (`dnGeo[prev].x > 0`), `dot(n_s, n′_s) ≥ 0.5`, `|z − z′| ≤ 0.1·z′`.
- History = Σ w_c·H_c / Σ w_c over valid taps; if Σ w_c < 10⁻³, the 3×3 ring around round(sp) (equal weights over
  valid taps); none ⇒ disoccluded (n_reproj = 0).
- Reprojected quantities: colour, μ, σ² (+ the between-tap spread Σw(μ_c − μ̄)², so blending is exact for the mixture),
  n (weighted, then floored), FW is **not** reprojected (it is a per-pixel value of the previous frame, read at q′ by the
  gradient pass).
- Background pixels are never destinations; a background previous tap is invalid (z′ = 0).
- **Static camera** (Changelog DN-1): identity, the pixel's own history whenever the previous frame had a hit there.

TD12 uses n^g; the denoiser uses n^s for both the predicate and the edge stops (one guide) **(own)**: identical on the
flat-shaded validation scenes, and the 60° threshold makes the difference immaterial on smooth meshes.

## 5. Temporal gradient → λ (DN6)

Per pixel q (ReSTIR temporal frame with `TF_HIST_VALID`, contribution MIS, q has a hit and `TS_QVALID`):

| Pair | When | a (frame t) | b (frame t−1) |
|---|---|---|---|
| forward: X_p | fwd code `SC_OK` | wp/c_p = lum F_t(Y_p)·W_p·J_p (tState TS3.z / TS2.y) | FW_{t−1}(q′) = lum(F·W) of the final reservoir of t−1 at q′ = `dnMom[prev].a` at q′ (TS2.x) |
| | fwd code `SC_O0_LIGHT`, `SC_OCCLUDED`, `SC_ZERO` | 0 (the path contributes nothing now: light removed, newly shadowed, zero) | same |
| inverse: X_c | `TS_SEL_C ∧ TS_INV_DONE ∧ ¬TS_EMPTY_OUT`, inv code `SC_OK`, executed spatial rounds ≤ 1 | w̃_c = lum F_c·W_c (TS3.y) | w̃_c·π_p(X_c)/π_c with π_p(X_c) = TS4.y, π_c = lum F of `res[w][q]` |
| | inv code `SC_O0_LIGHT`, `SC_OCCLUDED`, `SC_ZERO` | w̃_c | 0 (X_c could not exist at t−1: light added, newly lit) |

Other codes (O0_MISS/LOBE/TECH/SUPPORT, O1–O3, J_INVALID, …) are geometric (camera motion, footprint) and give no pair.
a and b are radiance-like one-sample estimates of the same pixel integral under the lighting of t and t−1 with the
**same path**; for samples distributed ∝ p̂_{t−1} (history samples), E[a − b]/E[b] = ΔL/L of the lighting change on the
support of p̂_{t−1}; the inverse pairs add the light that is new at t (support of p̂_t only).

- Δ_q = Σ(a − b), M_q = Σ max(a, b) over the pixel's pairs; if M_q > 10⁴ both are scaled to M_q = 10⁴ (a single
  firefly must not own a tile).
- `dn_gradient` reduces Δ, M over 8×8 tiles in workgroup memory; `dn_grad_filter`: λ_tile = |Σ_{3×3 tiles} Δ| /
  Σ_{3×3 tiles} M (24 × 24 px window, ≈ 576 pixels), 0 if ΣM ≤ 10⁻⁸ (no pairs).
- `dn_temporal` reads λ bilinearly between tile centres.
- **Gate (change bits):** λ := 0 when `RsTemporal.flags` has `TF_LIGHTS_SAME ∧ TF_ENV_SAME`: with static lights and
  env the radiance field is static and the pair differences are shift geometry (camera motion, jitter), not lighting.
  An option `gradientOnCamera` (default off) also uses λ on camera-only frames (view-dependent glossy changes) **(own)**.
- res[w] holds the temporal output of frame t only while at most one spatial round ran (round 0 reads res[w], writes
  res[1 − w]); with more rounds the inverse pairs are skipped (all temporal app modes use ≤ 1 round).
- Talbot MIS (`temporalMis: 'talbot'`) stores MIS-weighted w̃: the gradient is off there (λ = 0) **(own)**.

## 6. History, α and resets (DN7)

- λ′ = clamp((λ − λ₀)/(λ₁ − λ₀), 0, 1), defaults λ₀ = 0.03, λ₁ = 0.15 (tuned by the recovery gate; a change of ≥ 15 %
  of the local radiance discards the history).
- n = min(1 + (1 − λ′)·n_reproj, 64); α = max(α_min, 1/n, λ′); α_min = 0.2 (SVGF) for colour and moments.
- colour_t = (1 − α)·colour_reproj + α·c; moments by the EMA of §2 with the same α (n = 1 ⇒ μ = l, σ = 0).
- Variance for filtering: σ² when n ≥ 4, else the 7×7 bilateral spatial estimate of `dn_variance` (SVGF: weights of
  the depth/normal edge stops, the luminance variance of the integrated colour's luminance over the window).
- **Full reset** (n_reproj := 0 everywhere) on: the first frame of a denoiser (or after a reallocation: resize, scene
  load), a change of denoiser settings that changes the meaning of the history (α_min, demodulation), the input mode
  (PT ↔ ReSTIR), and
  - ReSTIR with temporal reuse: every frame whose `RestirAdvance.histValid` is false (config hash, scene, resize,
    env-map swap, explicit "reset temporal history", freeze seed / frame / history, temporal re-enabled). Light / env
    edits and animation do **not** reset (they are the gradient's job; the app restarts only the accumulation).
  - PT and ReSTIR without temporal reuse: every `FRAME_RESET_HISTORY` frame (any accumulation restart, i.e. every light
    / env edit, as the PT's own accumulation does).

## 7. Edge-stopping functions and the à-trous kernel

Iteration i, step s = 2^i, 5×5 B3-spline kernel h = (1/16, 1/4, 3/8, 1/4, 1/16)² (SVGF):
- depth: w_z = exp(−|z_p − z_q| / (σ_z·|∇z_p|·‖p − q‖ + 10⁻⁴·z_p)), σ_z = 1; ∇z_p from one-sided differences of the
  camera distance (the smaller of the two per axis, so a silhouette does not inflate it).
- normal: w_n = max(0, n_p·n_q)^σ_n, σ_n = 128.
- luminance: w_l = exp(−|l_p − l_q| / (σ_l·√(g₃ₓ₃(Var_p)) + 10⁻⁶)), σ_l = 4, g₃ₓ₃ = 3×3 Gaussian of the variance.
  Off while the colour history is younger than 4 frames (DN-12). In iteration 0, l is the 3×3 geometry-weighted
  prefiltered luminance (DN-13); with a converged output (static ≥ 8 frames) l is the demodulated previous output (DN-9).
- albedo: exp(−|ā_p − ā_q|₁ / (3σ_a)), σ_a = 0.05 (DN-5).
- background taps and taps outside the image have weight 0.
- colour = Σ h w c_q / Σ h w; variance = Σ h² w² Var_q / (Σ h w)² (SVGF variance propagation).

## 8. Interaction with the app modes

| Mode | Denoiser | Default | Gradient |
|---|---|---|---|
| PT (reference path tracer, interactive) | allowed | off | none (resets on accumulation restarts) |
| ReSTIR-interactive | allowed | **on** | yes (temporal, contribution MIS, 1 round) |
| ReSTIR-2022-criteria, initial only | allowed | off | yes when temporal is on |
| Offline | allowed | off | none (temporal off) |
| ReSTIR-unbiased | **forced off** | off | – |
| albedo (M1 placeholder) | off | off | – |

The toggle state is kept per mode (switching modes restores that mode's state). Debug views keep working: the
denoiser runs whenever it is on, also while a debug view is shown, so the history stays continuous. The stage tap
"denoised" (DBG_TAP_DENOISED) is not used by any view; the denoiser views below are untapped.

## 9. Debug views (PLAN §6 "Denoiser: variance; history; α")

| Id | Key | Kind | Content |
|---|---|---|---|
| 520 | `dn.variance` | scalar (log) | variance used by the first à-trous iteration (temporal or spatial) |
| 521 | `dn.history` | scalar | history length n after the update |
| 522 | `dn.alpha` | scalar | α of the colour blend |
| 523 | `dn.lambda` | scalar | gradient λ (after the change-bit gate, before the ramp) |
| 524 | `dn.demod` | vec3 | demodulated input c |
| 525 | `dn.integrated` | vec3 | temporally integrated colour (before à-trous) |
| 526 | `dn.reproj` | code | 0 background, 1 bilinear (all taps), 2 bilinear (partial), 3 ring fallback, 4 disoccluded, 5 reset |
| 527 | `dn.pairs` | code | gradient pairs of the pixel: bit 0 forward, bit 1 inverse |
| 528 | `dn.resolveN` | scalar | history length n_t of the output resolve (DN-6) |
| 530 + i | `dn.level[i]` | vec3 | output of à-trous iteration i (demodulated), i = 0 … 5 |
| 540 | `dn.demodFactor` | vec3 | a′ of this frame |
| 541 | `dn.albedoAccum` | vec3 | ā (accumulated a′; remodulates the output) |
| 542 | `dn.demodCheck` | scalar | lum(c·a′)/lum(L − L1): 1 wherever c is not clamped (the albedo is this sample's) |

## 10. Timing (HUD) and Q3

No pass of the frame carries `timestampWrites` (Q3). Every 30th advanced frame (and on demand), right after the
frame's `queue.submit`, the renderer encodes the frame's denoiser passes again into a separate command buffer with a
timestamp pair per pass and submits it; DN9 makes the re-run write the identical values. The HUD shows the per-pass and
total GPU ms averaged over the last ≥ 32 measurements. The gate measures the same way at 960×540 (the Chrome flag
`--enable-webgpu-developer-features` removes the 100 µs timestamp quantisation; the total is also measured as one span
from the first pass's begin to the last pass's end).

## 11. Validation (T16) and the M5.5 gate

**T16 (every gate).** `src/core/render/denoise/registry.ts` counts live `Denoiser` objects on the page. `harness.ts`
wraps every validation entry point (PT batches, ReSTIR batches, ReSTIR chains): it fails the run if a denoiser is live
before or after it, and writes `t16.denoiser` (`'none'` or the live instances) into the run's `meta.json` (PT
references gain a `t16` block). gate-m4 `t16Problems` and gate-m5 `t16ChainProblems` already require `'none'` for the
ReSTIR side; they (and gate-m3a/b/c) now also require it for our PT runs when the field is present (pre-M5.5 caches
have no field and were rendered by code that had no denoiser). The renderer test asserts that ReSTIR-unbiased forces
the denoiser off.

**`npm run validate -- --milestone M5.5`** (`validation/harness/gate-m55.ts`):
1. Gate 0: typecheck; CPU unit tests (`tests/denoise/*`: reference implementations of every formula above, the mode
   table, the T16 helpers); the Chrome suite `validation/gpu-tests/denoiser.gpu.test.ts` (each pass against an f64 CPU
   reference on synthetic inputs, held-frame idempotence, the timing re-run, no timestamp writes in the frame);
   the M5.5 app smoke (toggle, views finite, mode defaults, unbiased forced off, held frames, screenshots).
2. **FLIP** on (i) `cornell_i_512`, (v) `v_glossy_v1_512`, (vii) `vii_textured_512`, ix-d `ixs_d_camera_256`:
   ReSTIR-interactive (the interactive preset, c_cap 5, boost 3, RR), validation textures and Woop (the PT reference's
   scene), i.i.d. jitter, 4 seeds. Raw = `rsFrame` (1-frame ReSTIR, no accumulation), denoised = the colour target,
   from the **same** run. Static scenes: frames 16, 32, 48, 63 after a reset; ix-d: test frames 16, 24, …, 64 of the
   package's fly path (frame-override PT references). Reference: our PT, **65 536 spp** (16 batches × 4096), cached in
   `validation/out/m55/ptrefs` keyed by the package hash, the PT code hash (gate-m4 `codeHashes().pt`), spp, batches,
   seed and frame; rendered in GPU-lock chunks of ≤ 12 min. Metric: mean **LDR-ꟻLIP** (flip-evaluator, the version is
   recorded) of the display-encoded images (Standard view: exposure 1, clamp, sRGB OETF: the app's default display),
   averaged over seeds and frames; **pass iff mean FLIP(raw)/mean FLIP(denoised) ≥ 2** per scene. HDR-ꟻLIP is reported.
3. **Recovery** after the ix-e steps on `ixs_e_addremove_256` (C added, A × 2, B removed) and `ixs_e_half_256`
   (A × 0.5): each package state is held for 24 frames (32 before the first step), 8 seeds; per-frame 8×8-tile means of
   the denoised (and raw) luminance are read back. Mask of a step = tiles whose PT reference (4096 spp, before / after
   state) changes by ≥ 10 % of max(tile, 0.05·image mean). m(t) = mean over the mask and seeds; r(t) = (m(t) − m_before)
   / (m_after − m_before) with m_before = mean of the 8 frames before the step and m_after = mean of frames step + 16 …
   step + 23 (the denoiser's own converged levels, so its steady-state blur bias does not enter). Frames to recover =
   the smallest k (the step frame is k = 1) with |1 − r| ≤ 0.05 from frame step + k − 1 through step + 15.
   **Pass iff k ≤ 8 for every step.** The raw recovery and the steady-state difference to the PT reference are reported.
4. **Timing**: 960×540, `cornell_i_512` and `vii_textured_512` re-rendered at 960×540, default settings (N = 4),
   ≥ 64 timing submits after 32 warm-up frames, M5 Pro, Chrome with `--enable-webgpu-developer-features`.
   **Pass iff the mean total ≤ 3 ms** on both.
5. T16 on every PT reference used (`t16.denoiser === 'none'`).
6. **Edge temporal stability** (Changelog DN-6) on (i) and (vii), frames 40–63, display luminance, edge pixels from the
   PT reference: static camera, denoised with i.i.d. and with R2 jitter vs the denoiser-off progressive mean (pass iff
   the edge std and the edge Δ/frame are ≤ the mean's); slow pan 0.5 mm/frame, denoised vs the raw 1-frame output
   (pass iff the edge |I_t − 2I_{t−1} + I_{t−2}| is ≤ the raw output's). Reported, not gating: pixel-centre primaries
   (`--jitter none`) with the denoiser, the alternative to the resolve: their edge std and LDR-FLIP against the
   box-filtered reference next to the jittered + resolved output's.

Writes `validation/out/m55-gate-<time>/summary.json` and `summary.md`. Never writes outside this worktree's
`validation/out`.

---

## Changelog

(append-only)

- **DN-1 (implementation, first FLIP trial on (i)).** With the TD12 predicate on every frame and remodulation by the
  current frame's albedo, silhouettes came out stair-stepped and speckled: each frame's single jittered sample decides
  which surface (and albedo) a silhouette pixel shows, so its history failed the depth/normal test whenever the
  previous sample hit the other surface, and remodulation point-sampled the albedo. Two changes:
  1. **Static camera ⇒ identity reprojection** (`FRAME_CAMERA_MOVED` clear): the pixel's own history is used whenever
     the previous frame had a hit there. With static geometry (PLAN §0) the pixel footprint is identical, only the
     jitter moves the sample, exactly as for the progressive mean. With camera motion the §4 rule (TD12) applies.
  2. **Accumulated demodulation factor ā** (`dnAlb`, rgba16float): a′ is blended with the colour's α and reprojection
     weights, and the last à-trous pass remodulates with ā instead of the current a′. Inside a surface ā = a′ exactly;
     at silhouettes and on textures it is the jitter-averaged (box-filtered) factor, so edges and texture detail are
     anti-aliased like the reference. L1 (directly seen emitters and env) stays per frame.
  The CPU reference and the GPU suite cover both (U-DN-3 runs the identity path).
- **DN-2 (first gate run: (vii) FLIP ×1.19, "C added" recovered in 10 frames).**
  1. *ā and L1 get their own history length* n_a (`dnAlb.a`), independent of the gradient (albedo does not change with
     the lighting): α_a = 1/n_a with a static camera (the jitter's progressive mean: anti-aliased texture detail),
     max(1/n_a, 0.1) in motion. L1 (directly seen emitters, env) is accumulated the same way (`dnL1[2]`, rgba16float)
     with α = max(α_a, λ′); the output remodulates as ā·filtered + L̄1.
  2. *Gradient families.* The inverse pairs exist only on s = c pixels (≈ 25 % of the pixels once c_p = 5), so adding
     them to the forward sums diluted an added light's change. Each family is now summed apart (tile texture
     rgba32float: Δ_f, M_f, Δ_i, M_i), the inverse pairs weighted by 1/P(s = c) = (w̃_c + w̃_p)/w̃_c (both in tState), and
     λ = max(|ΣΔ_f|/ΣM_f, |ΣΔ_i|/ΣM_i): forward pairs see changes on the support of p̂_{t−1} (removed / changed light),
     selection-weighted inverse pairs on the support of p̂_t (added light).
- **DN-3 (demodulation with F0).** a′ = max(albedo, 0.02) + 0.04: a dielectric's white specular term divided by the
  small channels of a coloured albedo (the blue squares of (vii)'s floor) inflated them, and filtering bled them into
  neighbours of another albedo. Adding the dielectric F0 makes c ≈ E across albedo edges; the round trip stays exact.
- **DN-4 (variance for the edge stops).** SVGF steers the luminance stop with the variance of the input samples, which
  does not shrink as the history converges, so a converged history was over-blurred ((vii): temporal accumulation alone
  beat 5 à-trous iterations). The stop now uses min(1, K·α/(2 − α)) × the sample variance (an EMA of weight α keeps
  α/(2 − α) of the variance of independent samples; ReSTIR's temporal reuse correlates successive frames, K = 3 frames
  per independent sample, tuned on (i), (v), (vii), ix-d; K = 0 restores SVGF).
- **DN-5 (albedo edge stop).** w_l also multiplies exp(−|Δā|₁/(3σ_a)), σ_a = 0.05: the demodulated signal is not smooth
  across albedo edges where the specular part differs (DN-3 is an approximation), so the filter must not mix them.
- **DN-6 (temporal resolve of the output; user report "fizzy edges", "R2 wobble").** Reproduced with the eval harness
  (`denoise_eval.py stability`: per-pixel temporal std and Δ/frame of the display luminance on edge pixels (≥ 25 %
  luminance step to a 4-neighbour in the PT reference, dilated 1 px) and interior pixels, static camera, frames 40–63;
  slow pan 0.5 mm/frame: mean |I_t − 2I_{t−1} + I_{t−2}|, which cancels constant-velocity motion). Root cause: the
  displayed image was each frame's à-trous output, and that follows the frame's jittered primary hits: at a silhouette
  the jittered sample picks one surface per frame, the guides and the α_min = 0.2 share of new samples follow it, so
  edge pixels flipped from frame to frame (R2: a coherent shift of every edge and texel, the "wobble"). The progressive
  mean the user compared with averages the jitter away. Not the cause: the albedo source. The M1 G-buffer albedo is the
  same sample as `rs_primary`'s (same `pcg3d(runSeed, seedIndex, pixel)` jitter, LOD 0 textures); the harness checks
  ≤ 1 primId mismatch in 262 144 pixels between the M1 V-buffer and `rsVbuf` (an exact edge tie; barycentrics equal up to FMA-contraction differences, ≤ 7e-4), and
  view 542 shows lum(c·a′)/lum(L − L1) = 1. Pixel-centre rays (no jitter) would also be stable but alias every edge
  and texel against the box-filtered reference; a resolve keeps the anti-aliasing. Fix: `dn_resolve` (after the last
  à-trous, which now writes `dnOut`, rgba16float) accumulates the output (`dnTaa[2]`, rgba16float: rgb, n_t):
  - static camera and no lighting change in the last 8 frames: identity for every pixel, hit or background (the
    footprint is the same, as for the progressive mean), α_t = 1/n_t, n_t ≤ 1024;
  - camera motion or a lighting change within 8 frames: the pixel centre reprojected by the G-buffer motion vector
    (bilinear, no geometric test: a silhouette pixel's jittered hit flips surfaces, a depth / normal test would reject
    half its history every frame), n_t ≤ 8, variance clipping against the current 3×3 neighbourhood in YCoCg (γ = 1);
  - λ′ cuts n_t like the colour history, α_t ≥ λ′; reset / disocclusion of the colour history do not reset it.
  `resolve: false` restores the per-frame output. View 528 shows n_t. The colour history keeps the TD12 rule (q′
  consistency); only the display resolve uses the motion vector.
  Measured (static camera, display luminance, edge std / Δ per frame):
  (i): progressive mean 0.0037 / 0.0011, denoiser before DN-6 0.030 / 0.027, with the resolve 0.0013 / 0.0004 (R2 0.0009 / 0.0004);
  (vii): 0.0021 / 0.0007, 0.0051 / 0.0034, 0.0007 / 0.0001 (R2 0.0005 / 0.0001).
  The gate gains the edge-stability step (§11).
- **DN-7 (systematic darkening; coordinator: "a correctness bug in the display path").** After DN-6 the (vii) output
  was 2–4 % darker than the reference (raw: +0.1 %). Ablation by stage (vii / (i), frame 63, mean over lit pixels):
  temporal only −0.5 % / −0.2 %, + the resolve −1.5 % / −1.4 %, growing linearly with the accumulation length
  (−0.8 % → −1.5 % over 64 frames at a constant per-frame input bias of −0.5 %). Cause: Metal converts f32 to f16 on
  rgba16float stores **toward zero**, so every store loses ≈ ulp/2 (2⁻¹² relative); an EMA of weight α settles at
  ≈ −ulp/(2α) (−0.1 % at α_min = 0.2), but a 1/n mean drifts by ≈ −(n/2)·ulp/2 (−0.8 % at n = 64). Not the cause:
  ā vs a′, texture LOD, sRGB: albedo, a′ and ā are linear values of the same sample (LOD 0 in both pipelines; the
  ā·c̄ − mean(a′c) covariance term is ≥ 0 for the dielectric demodulation, i.e. could only brighten). Fixes:
  (1) the 1/n accumulators (ā and n_a: `dnAlb`, L̄1: `dnL1`, the resolve: `dnTaa`) are **rgba32float** (their
  increments, down to 1/1024 of the value, are below binary16 resolution anyway); (2) every remaining rgba16float store
  goes through `dn_rn16`, an exact round-half-to-even on the f32 bits, so the hardware conversion is exact. Result
  (frame 63): temporal only +0.03 % / +0.03 %, + resolve −0.13 % / −0.12 %, full pipeline −0.6 % / −0.35 %: the
  remainder is the à-trous luminance edge stop (σ_l → ∞: −0.15 % / +0.33 %), the known bias of value-dependent
  (robust) weights on right-skewed Monte-Carlo noise; it trades against edge preservation and is left as is.
  Memory: +16.6 MB at 540p (three rgba16float pairs → rgba32float).
- **DN-8 (recovery of a partially detected change).** λ′ cut the stored history length (up to 64 for the colour, 1024
  for the resolve), but α ≥ α_min means lengths beyond 1/α_min (colour) or 8 (resolve in the dynamic window) do not
  change α: a partial cut (λ′ = 0.6 of n = 64 → 26) left α at α_min, and the stale share decayed at 0.8 per frame
  ("C added": r = 0.76 on the step frame, 9 frames to 95 %). λ′ now cuts the effective lengths min(n, 1/α_min) and
  min(n_t, 8).
- **DN-9 (luminance guide from the converged output).** With static camera and lighting (no change for 8 frames,
  `DNF_GUIDE`) and n_t ≥ 8 at both pixels, the à-trous luminance stop compares the demodulated previous resolved output
  lum((T̄ − L̄1)/ā) instead of this frame's noisy values. Weights that depend on the values being filtered pull the
  mean toward the mode of right-skewed Monte-Carlo noise (the remainder of DN-7). Frame 16 / 32 / 48 / 63 bias on (vii):
  −0.68 / −0.66 / −0.65 / −0.62 % without, −0.57 / −0.47 / −0.44 / −0.40 % with; (i): −0.38 … −0.35 % → −0.28 … −0.09 %.
  FLIP ratio cost: (vii) 2.31 → 2.27, (i) 3.34 → 3.18 (the guide is smoother, so the stop is slightly weaker). On by
  default (`guide`): the bias is what the user sees on a still image; both scenes stay well above the ×2 criterion.
- **DN-10 (inverse family window).** The inverse pairs exist only on s = c pixels whose canonical sample picked the
  changed light, so a 3×3-tile window often holds only a few. Their window is now (2·invRadius + 1)² tiles (default 3:
  7×7 tiles); the forward family keeps 3×3. Recovery after "C added" did not change (12 frames at radius 3 and 6, r on
  the step frame 0.737 / 0.731): the inverse family was not the limiting factor (DN-12).
- **DN-11 (colour family).** A third, sample-independent gradient family per 8×8 tile: the mean of this frame's raw
  estimate L − L1 against the remodulated history at q′ (dnHist[prev]·ā[prev]), over the 3×3-tile window,
  λ_c = max(0, |m_cur − m_old| − 3·se)/max(m_cur, m_old) with se the standard error of m_cur, N ≥ 16 pairs; λ = max of the
  three families. It sees changes that the reservoir pairs sample sparsely (an added light's indirect share) and a
  history that still lags. Its noise allowance makes it insensitive to changes below ≈ 3 se, so it did not move the
  "C added" recovery either; it is kept as a safety net for large changes (U-DN-4 checks it against the CPU reference).
- **DN-12 (no luminance stop on young histories; "C added" in 12 frames).** After DN-10/11 the C-added output was still
  26 % low on the step frame (r = 0.74) and climbed for 12 frames; disabling the luminance stop for young histories
  (below) removes the deficit, which locates the cause: the luminance stop on a 1–3 sample history. Its variance is the 7×7 spatial
  estimate, the samples of the new light are rare and bright, and w_l rejects exactly those taps, so the filter
  returns the dark mode. The stop now applies only once the colour history holds n ≥ `lumMinN` = 4 frames; younger
  pixels are filtered by geometry and albedo only (blurrier for ≤ 3 frames, but unbiased). Recovery (4 seeds): C added
  12 → 8 frames (r = 0.98, 0.97, 0.96, 0.90, 0.92, …), A × 2 3 → 4, B removed 1 → 4 (r = 0.88 for 3 frames: the wider
  blur mixes masked tiles with unchanged neighbours until n reaches 4). FLIP unchanged ((i) 3.11, (vii) 2.25,
  (v) 2.93, ix-d 2.05).
- **DN-13 (prefiltered luminance guide).** In à-trous iteration 0 both sides of the luminance stop use a 3×3 binomial,
  geometry-weighted prefilter of the integrated luminance (`dnLumG`, r32float, written by `dn_variance`) rather than
  each tap's own value. This lowers the variance of the compared values without changing the weights' dependence on
  geometry. Later iterations filter already smoothed values and keep their own. Bias (frame mean of 16 / 32 / 48 / 63):
  (i) −0.05 → −0.02 %, (vii) −0.42 → −0.40 %, (v) −0.55 → −0.48 %, ix-d −0.24 → −0.10 %; FLIP and recovery frames unchanged
  (C added r at k = 8: 0.952 → 0.957). Cost: one 3×3 pass inside `dn_variance`.
- **DN-14 (à-trous bandwidth; the full gate after DN-13 failed timing).** 960×540, N = 5: (i) 3.59 ms, (vii) 8.81 ms
  (10-01: 1.72 / 2.74 ms). Each à-trous tap had come to read 68 B: colour 8, guide 8, ā 16 (rgba32float since DN-7),
  and with the DN-9 guide active (the static timing scene) the previous output 16 and L̄1 16 to compute
  lum((T̄ − L̄1)/ā) per tap and iteration, plus the DN-13 guide. `dn_variance` now writes the per-pixel tap data once
  (`dnTap`, rgba16float: ā and the guide luminance, −1 where the guide does not apply), and the DN-13 texture is read
  only in iteration 0: 24 B per tap again (28 B in iteration 0). Result: (i) 1.84 ms, (vii) 2.86 ms (à-trous
  0.48 / 0.39 / 0.36 / 0.36 / 0.38 ms on (vii)); FLIP, bias and recovery identical to DN-13 to the printed digits. The
  CPU reference rounds ā to fp16 in the albedo stop.
- **DN-15 (the full gate after DN-14: (vii) 3.16 ms; default N = 4).** (vii) fills the 960×540 frame; (i) leaves the
  sides of the 16:9 frame as background, which the passes skip, hence its lower cost. Since 10-01 each à-trous
  iteration on (vii) has cost about the same (0.36–0.52 ms; 5 iterations ≈ 2.1 ms). What grew is `dn_variance`
  (0.04 → 0.18 ms: the DN-13 prefilter, 0.08 ms of it, and the tap texture), `dn_gradient` (+0.05 ms, DN-11) and
  `dn_temporal`. Tried on the à-trous, none measurable beyond run-to-run noise (±0.05 ms):
  - geometry and tap data merged into one rgba32uint fetch (`dnTap`, kept: one fetch per tap);
  - a skip of the colour fetch for taps with a geometric weight < 10⁻⁶ (kept: exact to 10⁻⁵ of the sum; the CPU
    reference mirrors it);
  - all edge stops in one exp2 (log2 of the depth and normal weights, kept);
  - the B3 weights without a dynamically indexed array.
  The cost is per-tap bandwidth (≈ 24 B per tap from the texture caches), not ALU. The **default iteration count is now
  4**, measured against 5 (4 seeds; the gate's settings):
  - FLIP ratio: (i) 3.11 → 3.27, (vii) 2.25 → 2.31, (v) 2.93 → 2.97, ix-d 2.05 → 2.08 (3 iterations: 3.34, 2.36,
    2.91, 2.08);
  - recovery: C added 8 → 7 frames, A × 2 4 → 4, B removed 4 → 3 (the step-frame dip of DN-12 is 0.94 instead of
    0.88).
  With DN-4's history-aware variance, the fifth level (step 16, a 61-pixel footprint) mostly blurs a converged
  signal. It also costs 0.36 ms on (vii). The panel still offers 0–6.
