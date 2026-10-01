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
                                                               → dnGradTile (rg32float, W/8 × H/8): (ΣΔ, ΣM)
  dn_grad_filter     8×8 wg over tiles                         dnGradTile → dnLambda (r32float, W/8 × H/8): λ of a 3×3-tile window
PT only:
  copy colour → dnInput                                        (the PT writes its 1-spp sample into the colour target)
every mode:
  dn_temporal        8×8 wg                                    G-buffer (albedo, ns, pos, flags), input (rsFrame | dnInput),
                                                               rsL1 (| zero), dnGeo[prev], dnHist[prev], dnMom[prev], dnLambda,
                                                               res[final] plane 0 (ReSTIR)
                                                               → dnAtrous[0] (colour, temporal variance), dnHist[cur] (integrated
                                                                 colour, overwritten by à-trous 0's output), dnMom[cur], dnGeo[cur]
  dn_variance        8×8 wg                                    dnAtrous[0], dnMom[cur], dnGeo[cur] → dnAtrous[1] (n < 4: 7×7
                                                               bilateral spatial variance; else copy)
  dn_atrous i        8×8 wg, i = 0 … N−1, step 2^i             dnAtrous[src], dnGeo[cur] → dnAtrous[dst]
                       i = 0 also writes dnHist[cur] (SVGF feedback of the first iteration)
                       i = N−1 remodulates: colour = filtered·a′ + L1 → the colour target (pass-through for background)
```

N = 0 is allowed (temporal accumulation only: `dn_variance` then writes the colour target). Default N = 5 (SVGF);
the panel offers 0–6.

A **held frame** (paused, TD20; `RestirFramePass.encodeHold` re-displays the last ReSTIR estimate) runs the same passes
with the same parity: identical inputs ⇒ identical outputs (DN9). The PT colour copy is skipped on held frames.

---

## 2. Buffers and formats (960×540: 518 400 px)

| Name | Format | B/px | 540p | Content | Why this format |
|---|---|---|---|---|---|
| `dnHist[2]` | rgba16float | 2 × 8 | 8.3 MB | demodulated colour history (rgb; a unused) | fp16 keeps 11 significant bits (rel. 4.9·10⁻⁴): far below the Monte-Carlo noise and the 8-bit display after the view transform. Range 6.5·10⁴: the demodulated input is clamped to it (a biased display path; only firefly pixels reach it). |
| `dnMom[2]` | rgba16float | 2 × 8 | 8.3 MB | (μ_l, σ_l, n, FW): EMA mean and **standard deviation** of the demodulated luminance, history length n, lum(F·W) of the final reservoir (gradient input of the next frame) | The raw second moment would square the dynamic range (fp16 overflows at l > 256) and lose the variance to cancellation (m₂ − m₁² with 2⁻¹¹ relative error leaves a floor of 4.9·10⁻⁴·μ²). The West/Welford EMA update `δ = l − μ; μ += αδ; σ² = (1 − α)(σ² + αδ²)` runs in f32 registers and stores σ, which has the range and precision of the luminance. n ≤ 2048 is exact in fp16 (we cap at 64). FW is a radiance-like estimate (relative precision suffices; a clamp at 6.5·10⁴ only weakens the gradient of a firefly). |
| `dnGeo[2]` | rg32uint | 2 × 8 | 8.3 MB | x: f32 distance to the camera; y: oct 2×16 snorm shading normal | Distance in **f32**: the à-trous depth weight compares neighbour differences against the local depth gradient, which on grazing planes is ~10⁻³ of the distance (fp16 would quantise it). Normal **oct 2×16** (< 0.004° worst case with WGSL's round-to-nearest `pack2x16snorm`; data-formats.md §B3 quotes 0.0025° for an optimised encoder): never fp16 for unit vectors. One guide for the reprojection predicate (prev) and the à-trous edge stops (cur). |
| `dnAtrous[2]` | rgba16float | 2 × 8 | 8.3 MB | à-trous ping-pong: demodulated colour + variance | Colour as `dnHist`. The variance only steers the luminance edge weight; it saturates at 6.5·10⁴ (σ = 256: such pixels are then filtered by geometry only, which is the desired behaviour for extreme noise). |
| `dnInput` (PT only) | colour format | 16 | 8.3 MB | copy of the PT 1-spp sample | The PT writes its sample into the colour target, which the denoiser overwrites. |
| `dnGradTile` | rg32float | 8 per tile | 65 KB | (ΣΔ, ΣM) over an 8×8 tile | Sums of up to 64 radiance-like values: f32 (tiny texture). |
| `dnLambda` | r32float | 4 per tile | 33 KB | λ of the 3×3-tile window | tiny. |
| params | uniform 64 B | – | – | settings, flags, tile sizes, arena offsets | – |

Total ≈ 42 MB at 540p (≈ 34 MB without the PT input copy). The M1 G-buffer (80 B/px f32, `renderer.ts`) is read once
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
- Output = a′·filtered + L1. The round trip is exact up to fp16 rounding of `filtered`.

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
| 530 + i | `dn.level[i]` | vec3 | output of à-trous iteration i (demodulated), i = 0 … 5 |

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
4. **Timing**: 960×540, `cornell_i_512` and `vii_textured_512` re-rendered at 960×540, default settings (N = 5),
   ≥ 64 timing submits after 32 warm-up frames, M5 Pro, Chrome with `--enable-webgpu-developer-features`.
   **Pass iff the mean total ≤ 3 ms** on both.
5. T16 on every PT reference used (`t16.denoiser === 'none'`).

Writes `validation/out/m55-gate-<time>/summary.json` and `summary.md`. Never writes outside this worktree's
`validation/out`.

---

## Changelog

(append-only)
