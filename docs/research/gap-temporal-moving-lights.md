# Gap fill: exact temporal reuse in ReSTIR PT under moving lights and a moving, jittered camera

> **Where this file lives.** The orchestrator asked for this report at
> `/private/tmp/claude-502/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-ea73-491f-b7b7-6d0aab1ffce1/scratchpad/research/gap-temporal-moving-lights.md`.
> This session ran in **plan mode**, where the only writable file is this plan file, so the report is here. Copy it verbatim to the intended path.
> Nothing was cloned or written anywhere else. Sources were streamed to stdout (`curl … | cat -n`) or read from the local page PNGs.
>
> **Tags.** **[SRC]** means a source says it, with the location. **[INFERENCE]** means my own derivation or recommendation. **[UNVERIFIED]** means plausible but not confirmed.
> Code locations are `repo/file:line` at the snapshot read on 2026-09-28:
> - `DQLin/ReSTIR_PT` master, called **Falcor22** below;
> - `EvanLuo42/ReSTIR-PT-Enhanced` HEAD, called **Evan**;
> - `bevyengine/bevy` main, commit `a173a349` (2026-09-28), called **Bevy**;
> - `NVIDIA-RTX/RTXDI-Library` main, called **RTXDI**. RTXDI is proprietary. I describe its semantics only and copied no code.
>
> The other reports are cited by short name, e.g. "restir-practice §3.4". They all live under `/Users/mark.boss/.claude/plans/do-a-deep-dive-shimmering-ritchie-agent-*.md`.

---

## 0. Decisions (read this first)

1. **Temporal GRIS form: contribution MIS (RTXDI-style π/π_sum) with confidence weights.**
   - All previous-frame target values come from **one** deterministic previous-frame evaluator `E_{t−1}`.
   - An inverse shift is run **only when the canonical sample is selected**. This happens with probability `P_c ≈ c_c/(c_c+c_p) ≈ 5–30 %`.
   - When the temporal sample is selected, its previous-frame target is read from storage. That is exact by a storage invariant (I1, §3.4).
   - Cap convention: `c_p = min(c_cap, c_prev)` with `c_cap = 20` and `c_c = 1`, so `c_out = c_c + c_p`. This equals Enhanced's `min(c_Cap, c_temp)+1` and Falcor's `min(20·M_cur, M_prev)` when `M_cur = 1`.
   - Generalized Talbot (= pairwise with |R| = 1, M = 2) using the same evaluator stays as a cross-check mode. It is also unbiased but always needs the inverse shift (§3).
2. **Falcor's temporal Talbot is biased under light changes.** It mixes a stored exact `p̂_{t−1}` (temporal self-term, `TemporalReuse.cs.slang:192`) with a recomputed `p̂_prev` that uses the previous camera but **current** lights and no suffix update (`:206-208`).
   - The partition of unity fails (§3.5).
   - Closed-form prediction for a light-intensity step of ratio `r`: relative bias `B(r) = 1/(1+c_p) − r/(r+c_p)`. That is −4.3 % for r = 2 and +2.3 % for r = 0.5 at c_p = 20.
   - Evan's temporal MIS has **swapped arguments** (`Temporal/TemporalReuse.cs.slang:69-75`). It is biased whenever `p̂_t(Y_p) ≠ p̂_←p(Y_p)`, with the opposite sign to Falcor (§2.3).
3. **Previous-frame state needed (static triangles, non-occluding analytic lights):**
   - previous camera position and view-projection;
   - the **stored jittered previous V-buffer**;
   - previous light buffer, plus cur↔prev light-index maps;
   - previous light-selection pmf, plus the alias table only if suffix NEE is replayed through it.

   **No previous TLAS/BVH and no previous light tiles** (proofs in §4.2 and §4.3). This resolves critique X3 in favour of restir-practice.
4. **Replaying RIS-NEE over per-frame light tiles is not a temporal shift.** Falcor's `traceTemporalUpdate` re-draws the light through the RNG (`PathTracer.slang:1753-1754`, `:1807`). Under Enhanced's RIS-NEE over per-frame tiles (ENH §6.1, 32/B² candidates at every bounce, txt L503-509, L599-603), that re-draw is not reproducible. It is not a function of the stored sample, so it is not a bijection (GRIS Def. 4.2).
   - Using it anyway is biased. A two-light counterexample gives **+64 %** on the refreshed suffix (§5.3).
   - A fresh plain (non-RIS) re-draw is also biased when some lights are occluded: it darkens by the visible PSS fraction.
5. **Recommended exact refresh: explicit light vertices plus a small suffix cache.**
   - Every NEE-terminated path stores its light vertex as `(lightIdx, u, v)`, whether that vertex is the rc vertex or deep in the suffix.
   - Store a suffix cache `{x_{d−1} hit, ω_o, β_s}`, or `{ω_{d−1}, t_occ, p2}` for BSDF-hit endings.
   - Refreshing is then exact and deterministic (a light-local shift with a known Jacobian), costs ≤ 1 shadow ray and 0 closest-hit rays, and works with Enhanced's RIS-NEE and light tiles unchanged.
   - Only samples attached to a **moved** light need rays. Intensity, colour, cone, add and remove need an analytic rescale or nothing.
   - Minimal-storage alternatives: plain alias NEE on fixed counter-based RNG dimensions in the suffix (R2), or "virtual tiles" (R3).
   - Cheapest unbiased fallback (E2): on light-change frames, give deep-suffix paths zero temporal MIS weight.
6. **Light-vertex Jacobian.**
   - Area measure: `J_M = A_t/A_{t−1}`, which is 1 for rigid motion.
   - In the Enhanced/Falcor **PSS** formulation (reservoir stores `F = ωf/p`), with uniform-area or delta light sampling: `J_PSS = P_sel^t(ℓ_t)/P_sel^{t−1}(ℓ_{t−1})`.
   - Omitting the pmf ratio biases every temporal sample whenever the power pmf changes, including intensity edits and add/remove of *other* lights (§6).
7. **Cost of exact mode on the M5 Pro software BVH** (§8; inputs from webgpu-platform §4.1: 192 Mrays/s incoherent closest-hit, 610 coherent shadow).
   - Temporal pass: **≈ 2.7 ms at 960×540 and ≈ 4.7 ms at 1280×720** (contribution MIS), versus ≈ 4.8 and 8.6 ms for Talbot.
   - Light-change frames add ≈ **0.5 / 0.9 ms** with the cache, versus ≈ 3.0 / 5.4 ms for a naive full suffix re-trace. Exact mode is affordable.
8. **Tests (§9).**
   - Per-frame ensemble tests for scenes ix-a…d plus a new ix-e (add, remove and intensity steps).
   - T6(b) as a stored-vs-recomputed identity check. T6(c) as provenance tags plus a cross-evaluator check.
   - Planted controls with predicted signs:
     - N1-mixed (current lights, Falcor-style): `B(r)` sign on intensity steps, and −95 % on the first frame of an added light;
     - N1-consistent: darkening only, at support violations;
     - N2, omit the pmf Jacobian: the brightened light darkens and the others brighten;
     - N3, stale suffix: lag;
     - N4, fresh RIS re-draw: brightening in the two-light test.

---

## 1. Setup and notation

- **Frame state** `S_t = (C_t, J_t, V_t, L_t, P_t, G)`:
  - `C_t`: camera (position `o_t`, view-projection `VP_t`);
  - `J_t`: per-pixel subpixel jitter;
  - `V_t`: jittered V-buffer, i.e. primary hit `x_1^t(q)` stored as (instanceId, primId, bary);
  - `L_t`: light buffer (type, transform `M_t(ℓ)`, size, radiometry, spot parameters, existence);
  - `P_t`: power-based light-selection pmf `P_sel^t(ℓ)` (alias table + pmf);
  - `G`: static triangle geometry/BVH and materials.
- **Pixel domain** `Ω_{t,q}`: PSS paths starting at the frame-t jittered primary hit `x_1^t(q)`, with `x_0 = o_t`.
  - [INFERENCE, critique X1] Each frame's jitter defines its own domain. Per-frame unbiasedness holds for that domain, and the ensemble over jitter equals the BOX-1.0 pixel integral that Cycles computes.
- **PSS integrand** (ENH S-§1, S-Eq.3; Falcor22 `PathReservoir.F`):
  - `F_t(ū) = ω_t(x̄) f_t(x̄) / p_t(x̄)` in RGB, with target `p̂_t = lum(F_t)`;
  - RR is removed from the PSS (ENH §6.2.4);
  - `ω_t` is the NEE/BSDF path-MIS weight: `ω1 = M_B p1/(M_B p1 + p2)` for area lights (ENH S-§5), and `ω1 = 1` for delta lights (gris-verify C1).
- **Temporal correspondence.** `q' = q'(q)` is chosen from G-buffers and a sample-independent RNG only (course notes Tip 4.1).
- **Temporal shift** `T: D(T) ⊂ Ω_{t−1,q'} → Ω_{t,q}` is the hybrid shift:
  - replay the prefix from `x_1^t(q)`;
  - reconnect at the rc vertex `x_k`;
  - copy the suffix.

  Light vertices are handled light-locally (§6).
- **Reservoir** `r = (X, W_X, c, F)`, where `F = F_{frame}(X)` in the reservoir's own pixel domain (Falcor22 and ENH S-Alg.1).
- **Path classes** used for the refresh (§5). These are structural and preserved by all shifts. `k` is the rc index and `d` the light vertex index, as in `x̄=[x_0 … x_d]`.

| Class | Definition | Suffix after `x_k` |
|---|---|---|
| **L** | rc = NEE light vertex (forced reconnection, ENH §6.2.3), `k = d` | none |
| **N1** | `k = d−1`, NEE at the rc vertex | one NEE segment `x_k → x_d` |
| **B1** | `k = d−1`, BSDF-sampled hit of an emitter at `x_d` | one BSDF segment |
| **D-NEE** | `k < d−1`, ends with NEE at `x_{d−1}` | deep: ≥ 1 BSDF bounce, then NEE |
| **D-BSDF** | `k < d−1`, ends with a BSDF-hit emitter | deep |
| **E** | emitter itself is rc (escaped vertex, Falcor22) or env map | treated like B1 |
| **R** | no rc vertex (full random replay) | whole path replayed, J = 1 |

---

## 2. What the sources say and do (evidence)

### 2.1 Papers and course notes

- **GRIS App. B, "Temporal Reuse"** (page image `pages/gris/p-23.png`, p.75:23) [SRC]:
  - temporal reuse is unbiased with proper MIS (their generalized Talbot or pairwise), but it needs a temporal shift mapping;
  - the MIS weights require bijective shifts between the prior and current frames, so fully unbiased reuse must evaluate paths in both frames;
  - this is tricky in dynamic scenes, and biased approximations (e.g. ignoring visibility) are acceptable when the bias is imperceptible.
- **GRIS §4.3, Eq. 16–18, 21** (p.75:6–7, verified on `p-06.png` / `p-07.png`) [SRC]:
  - resampling weights `w_i` are "arbitrary non-negative random variables" with `w_i > 0` iff `X_i ∈ D(T_i)` and `p̂(Y_i) > 0`;
  - this can be relaxed to allow `w_i = 0` where `c_i(Y_i) = 0` or `W_i = 0` ("Relaxing constraints");
  - the general UCW is Eq. 18, `W_Y = c_s(Y) · W_s |∂T_s/∂X_s| · Σ_j w_j / w_s`;
  - unbiasedness requires only the contribution-MIS partition Eq. 17, `Σ_{i: y ∈ T_i(supp X_i)} c_i(y) = 1`;
  - Eq. 21 is the special case `w_i = m_i p̂ W_i J_i`.

  This is the lemma behind §3.
- **Course notes (Wyman et al. 2023, updated 4 Mar 2024)**, text extracted in memory from the PDF [SRC]:
  - §4.3 (p.19): temporal MIS "require[s] evaluating last frame's p̂", which may need visibility queries in the previous scene and hence a stored previous acceleration structure.
  - §7.1.4 (p.41–42):
    - the costly term is the previous-frame density of the current candidate;
    - suggested shortcuts are the current BVH, assuming zero, or last frame's data with unchanged visibility;
    - replacing that density by an over-estimate darkens, an exact value adds no bias, and an under-estimate brightens;
    - assuming zero over-counts and brightens.
  - Footnote 9 (p.33): (triangle id, barycentrics) follows animated triangles automatically, but a temporal shift is needed to update the cached radiance `L`. One way is to reuse the world-space direction to find `y_{k+1}` and **reuse the RNG seed recorded on `x_{k+1}`** to generate the rest of the path. That is random replay of the suffix, including its light sampling.
  - Footnote 8 (p.33): for an NEE-sampled light vertex, the reconnection uses the light pdf converted to solid angle and lobe pdf 1.
- **Enhanced** (`restirpt_enhanced_2026.txt`) [SRC]:
  - L127-129: temporal confidence after resampling is `min(c_Cap, c_temp) + 1`;
  - L503-509: optional RIS over NEE with Wyman–Panteleev light tiles, 128 tiles × 1024 lights per frame, each 8×8 screen tile drawing one tile "to generate candidates per bounce";
  - L599-603: 32 candidates at the primary hit, `32/B²` (clamped ≥ 1) at bounce B;
  - L525-530 (§6.2.3): NEE light vertices are forced to be the rc vertex if no earlier rc exists, because "replayed random numbers usually select the same light";
  - L797-799: for unbiased converged images, accumulate with temporal reuse off.
  - **The paper says nothing about dynamic lights or its temporal MIS form.** The enhanced-paper report §5.4 and the critique confirm this.

### 2.2 Falcor22 (the 2022 reference)

- **`TemporalReuse.cs.slang`:**
  - `:67-78`: previous shading at the history pixel is built from `temporalVbuffer` with `computeRayPinholePrevFrame`;
  - `:126`: `prevPixel = pixel + mv·frameDim + U[0,1)²` (stochastic rounding);
  - `:139`: cap `M_prev ← min(20·M_cur, M_prev)`.
- **Talbot, `:146-226`:**
  - forward shift of the temporal sample: `shiftAndMergeReservoir(…, doTemporalUpdateForDynamicScene, …)` at `:174-175`;
  - temporal self-term: **stored** `toScalar(temporalReservoir.F)/dstJacobian·M` at `:192`;
  - canonical's inverse term: `computeShiftedIntegrand(…, evalVisibility=true, usePrev=true)` at `:206-208`, **without** the temporal-update flag.

  `Shift.slang:36-38` confirms the defaults. `usePrev` only swaps the camera position (`Shift.slang:101`, `:126`).
- **"Constant" contribution-MIS branch (`:230-296`):**
  - for a selected canonical it computes the inverse-shift pdf (`:261-263`);
  - for a selected temporal it uses the stored `F/J` (`:282-284`);
  - quirk: it tests `kSpatialReSTIRMISKind` inside the temporal pass (`:271`, `:280`, `:286`) [SRC].
  - Hybrid shifts force Talbot anyway (reference-code §1.7.3).
- **`traceTemporalUpdate` (`PathTracer.slang:1730-1841`):**
  - *NEE-at-rc branch* (`:1737-1777`): builds `SampleGenerator(rcRandomSeed)`, calls `generateLightSample`, casts a shadow ray, then rewrites `rcVertexWi = ls.dir`, irradiance `= Li·pdf` (i.e. `Le`), the light type and `lightPdf`.
  - *Deep branch* (`:1779-1840`): reuses the stored `rcVertexWi` (`:1797`), replays the suffix with `SampleGenerator(rcRandomSeed)` (`:1807`) through `handleHit`, which performs NEE with the replayed stream, and rewrites irradiance, `lightPdf` and type (`:1837-1839`).
  - It is called from the hybrid shift (`Shift.slang:180-188`), except when the rc vertex is the escaped emitter, and from the reconnection shift (`:69-72`).
  - It is off by default and enabled automatically for animated scenes (reference-code §1.7.4).
- **Net effect** [INFERENCE, consistent with gris-verify C4]:
  - the forward shift is evaluated in the current scene with a refreshed suffix;
  - the inverse shift uses the previous camera and V-buffer but current lights and a stale (current-frame) canonical suffix;
  - the self-term is exact-previous.

  It is **exact for camera-only motion and biased for any light change** (§3.5).

### 2.3 Evan (unofficial Enhanced implementation, BSD-3)

- **Pass structure (`Temporal/TemporalShift.cs.slang`):**
  - history pixel from primary or dual motion plus a compatibility test (`:29-71`);
  - pairs needing replay are enqueued (`:141-150`);
  - both directions are evaluated with `evaluateHybridShift(…, Current→Previous)` and `(…, Previous→Current)` (`:171-178`).
- **MIS weights (`Temporal/TemporalReuse.cs.slang`):**
  - cap `cHistory = min(max(c,1), cCap)` (`:36-37`);
  - canonical `mCurrent = pairwise(c_c, p̂_c(X_c), c_h, p̂_prev(T⁻¹X_c), J)` (`:54-60`). This is correct Talbot.
  - temporal `mHistory = pairwise(c_h, p̂_t(Y_p), c_c, p̂_{t−1}(X_p), 1/J)` (`:69-75`), which evaluates to `c_h p̂_t(Y_p)/(c_h p̂_t(Y_p) + c_c p̂_←p(Y_p))`;
  - the correct Talbot weight is `c_h p̂_←p/(c_h p̂_←p + c_c p̂_t)`. **Arguments are swapped** (`GRIS.slang:28-34` defines `pairwiseMISWeight`). This is the same bug reference-code §2.3 found in spatial reuse.
  - With an intensity step `r` and c_p = 20: Σm − 1 = r/(r+20) + 20r/(20r+1) − 1, i.e. **+6.7 % at r = 2** and −6.7 % at r = 0.5 [INFERENCE].
  - It is also biased for camera-only motion wherever the shift changes the target.
- **Previous-frame handling:**
  - Analytic NEE endpoints use `evalAnalyticNee(src, prefixPos)` (`Common/HybridShift.slang:129-158`; `Common/Lights.slang:334-389`), which reads the **current** `gScene.getLight(index)`, whatever `srcTime/destTime` is.
  - Area-light points come back from the **stored world position** (`Lights.slang:372-374`), so a moving analytic area light does not carry its samples.
  - Triangles and emissive triangles are mapped between frames with instance transforms (`SurfaceLoad.slang:480-530`, `loadPreviousShading` at `:315`, `mapWorldDirBetweenFrames` at `Common/SurfaceTypes.slang:80`, used at `HybridShift.slang:325`).
  - The suffix is `src.suffixThroughput` and is never refreshed (`HybridShift.slang:25-31`).
- **Do not port its temporal logic.**

### 2.4 Bevy Solari (ReSTIR DI/GI in WESL, not PT)

- **Light-id translation (`restir.wesl:104-113`):** the temporal reservoir's light id is translated through `previous_frame_light_id_translations`. If the light is `LIGHT_NOT_PRESENT_THIS_FRAME`, the temporal reservoir is dropped. Confidence is capped at `:115`.
- **MIS evaluation (`merge_reservoirs`, `:179-281`):**
  - both light samples are resolved with the **current** `light_sources` (`:193`, `:208`);
  - the canonical-at-previous-surface term traces `trace_visibility_previous_frame` (`:257`);
  - the previous camera position comes from `previous_view` (`:36-39`).
- [INFERENCE] Bevy recomputes both previous-domain terms with the *same* approximate light state ("consistent-approximate"). Magnitude errors therefore do not bias, only support errors do (§3.5).
- It keeps a previous TLAS because Bevy supports moving meshes.

### 2.5 RTXDI ReSTIR PT (semantics only; proprietary)

- **Temporal cap** (`PT/TemporalResampling.hlsli:215-218`): `min(M, maxM, reducedMaxHistory)`.
- **`BiasCorrection`** (`:327-360`):
  - always inverse-shifts the **selected** sample into the previous surface;
  - forms `temporalP = lum(target)·J`, sets `pi = selectedPrev ? temporalP : pi`, and accumulates `piSum += temporalP·M_prev`;
  - finalizes `W` with `pi/(piSum·p̂)` (`:429-438`).

  This is contribution MIS with the previous term **recomputed** in both cases, which is consistent.
- **Hybrid shift, NEE-light rc** (`PT/HybridShift.hlsli`):
  - the light index is translated prev→cur, and an invalid mapping invalidates the sample (`:230-262`);
  - light info is loaded with `isPrevFrame` (`:272`);
  - `J = src light solid-angle pdf / dst pdf` (`:305-313`).
- Mid-path rc vertices cache world position and radiance, and the suffix is not refreshed (restir-practice §2.6) [INFERENCE].

---

## 3. (a) The temporal GRIS step

### 3.1 Lemma (GRIS with arbitrary resampling weights)

[SRC GRIS Eq. 10, 16–18, 21; the proof is a restatement.]

**Setup.**
- Candidates `X_i` have UCWs `W_i`, bijective shifts `T_i` and Jacobians `J_i`.
- `c_i(y)` are **deterministic** functions satisfying Eq. 17 (they sum to 1 over the `i` that can produce `y`).
- `w_i ≥ 0` are any random variables with `w_i > 0` whenever `c_i(Y_i) h(Y_i) W_i ≠ 0`.

**Procedure.** Select `s` with probability `w_s/Σw`. Set `W_Y = c_s(Y) W_s J_s Σ_j w_j / w_s`.

**Claim.** For every integrable test function `h`, `E[h(Y) W_Y] = ∫_{∪_i T_i(supp X_i)} h`. If the shading value is an estimate `ĥ_s` with `E[ĥ_i | X_i, W_i] = h(Y_i)`, then `E[ĥ_s W_Y] = ∫ h`.

**Proof.**
1. Conditioned on everything except the selection, `E_s[h(Y_s) W_Y] = Σ_i (w_i/Σw)·c_i(Y_i)h(Y_i)W_iJ_i·Σw/w_i = Σ_i c_i(Y_i)h(Y_i)W_iJ_i`.
2. By the UCW property and the change of variables, `E[c_i(T_iX_i)h(T_iX_i)W_iJ_i] = ∫_{T_i(supp X_i ∩ D(T_i))} c_i h`.
3. Summing over `i` and applying Eq. 17 gives the claim. ∎

**Consequences.**
- The resampling weights `w_i` only affect variance. They may use stale radiance, approximate targets or random estimates.
- Unbiasedness rests entirely on the `c_i` being one deterministic partition of unity whose support matches the true producibility sets.

### 3.2 The three candidate forms (M = 2, canonical `c`, temporal `p`)

Define:
- `p̂_t(y) = lum F_t(y)` for `y ∈ Ω_{t,q}`;
- `π_p(y) = lum F_{t−1}(T^{-1}y) · |∂T^{-1}/∂y|` for `y ∈ I(T)`, and 0 otherwise. This is GRIS Eq. 35, "p̂ from p";
- `C = c_c + c_p`.

1. **Generalized Talbot** (GRIS Eq. 36 with confidences, Falcor22). Resampling MIS `m = c` gives
   `m_c(y) = c_c p̂_t(y)/(c_c p̂_t(y) + c_p π_p(y))` and `m_p(y) = c_p π_p(y)/(c_c p̂_t(y) + c_p π_p(y))`.
   - `w_c = m_c(X_c) p̂_t(X_c) W_c`, `w_p = m_p(Y_p) p̂_t(Y_p) W_p J_p`, `W_Y = (w_c+w_p)/p̂_t(Y)`.
   - Needs `π_p(X_c)` (an inverse shift) **always**, and `π_p(Y_p) = lum F_{t−1}(X_p)/J_p`.
2. **Pairwise with |R| = 1** (GRIS Eq. 37/38; course notes Eq. 7.8).
   - Non-defensive with M = 2 is identical to Talbot [INFERENCE, algebra].
   - Defensive is `m_c = c_c/C + (c_p/C)·Talbot_c` and `m_p = (c_p/C)·Talbot_p`. It has the same cost as Talbot and differs by `c_c/C = 1/21`.
3. **Contribution MIS (π/π_sum)**:
   - constant resampling weights `m_i = c_i/C`, i.e. `w̃_c = c_c p̂_t(X_c)W_c` and `w̃_p = c_p p̂_t(Y_p)W_pJ_p`;
   - contribution MIS `ĉ_i(y) = c_i π_i(y)/(c_c π_c(y) + c_p π_p(y))`, with `π_c = p̂_t`;
   - Eq. 21 then gives
     `W_Y = [π_s(Y)/(c_c π_c(Y) + c_p π_p(Y))] · (w̃_c + w̃_p)/p̂_t(Y)`;
   - needs `π_p` **only at the selected sample**:
     - `s = c`: inverse shift of `X_c`;
     - `s = p`: `π_p(Y_p) = lum F_{t−1}(X_p)/J_p`, which comes from storage or a recompute.

### 3.3 Which quantity is stored and which is recomputed

| Quantity | Meaning | Source in the recommended design | Exact iff |
|---|---|---|---|
| `F_c = F_t(X_c)` | canonical integrand | stored by initial sampling at frame t | always |
| `W_c`, `c_c = 1` | canonical UCW, confidence | stored | always |
| `F_p^{st} = F_{t−1}(X_p)` | temporal integrand in `q'` at frame t−1 | stored in the previous reservoir | **invariant I1** holds at t−1 |
| `W_p`, `c_prev` | temporal UCW and confidence | stored; `c_p = min(20, c_prev)` | always |
| `Y_p = T(X_p)`, `J_p`, `F_t(Y_p)` | forward shift | recomputed under `S_t`, with the suffix refreshed if its light state changed (§5) | refresh is a deterministic shift |
| `π_p(X_c)` | inverse shift of the canonical into `q'` | recomputed by `E_{t−1}` (prev camera, prev V-buffer, prev lights and pmf, current BVH, suffix under `L_{t−1}`) | `E_{t−1}` uses exact `S_{t−1}` |
| `π_p(Y_p)` | "p̂ from p" at the shifted temporal sample | `lum F_p^{st}/J_p` (stored) **or** recomputed by `E_{t−1}(T^{-1}Y_p)` | stored equals recomputed (T6(b)) |

**Invariant I1.** Every reservoir written at the end of frame t holds `F = F_t(X)` exactly, i.e. with suffix radiance valid under `S_t`, in its own pixel domain.
- It holds for initial samples.
- The temporal step preserves it if the forward shift refreshes the suffix and writes the refreshed `F` and caches into the output.
- Spatial reuse at frame t preserves it trivially, because lights are fixed within a frame.
- Induction then gives I1 for all frames.

### 3.4 Theorem (exact temporal step)

**Assume:**
- (A1) `T` is a bijection `D(T) → I(T)` whose inverse is the same code run with the frame roles swapped (§6, §7);
- (A2) `E_{t−1}` is deterministic and uses exact `S_{t−1}`;
- (A3) I1 holds at t−1;
- (A4) the canonical covers `supp p̂_t` (the initial path tree has positive density wherever `F_t > 0`, GRIS Def. 5.2).

**Claim.** Both Talbot (3.2-1) and contribution MIS (3.2-3), with `π_p(Y_p)` taken from storage, are unbiased.

**Proof.**
1. By (A3), `F_p^{st} = F_{t−1}(X_p)`. By (A1), `X_p = T^{-1}(Y_p)` and `|∂T^{-1}/∂y|(Y_p) = 1/J_p`. So the stored route and the recomputed route give the **same function** `π_p(·)` at every point where it is used.
2. **Producibility.** The temporal input can produce `y` iff `T^{-1}(y) ∈ supp X_p`. By induction with (A4) at t−1, `supp X_p = supp p̂_{t−1,q'}`. Hence `y` is temporally producible iff `π_p(y) > 0`. The canonical can produce `y` iff `p̂_t(y) > 0`.
3. **Talbot.** `m_c(y) + m_p(y) = 1` wherever the denominator is positive.
   - Where only the canonical produces `y`: `π_p = 0`, so `m_c = 1`.
   - Where only the temporal produces `y`: `p̂_t(y) = 0`, so `F_t(y) = 0` and the point contributes nothing.

   Eq. 20 therefore holds on `supp p̂_t`, and with `c = m` the lemma applies.
4. **Contribution MIS.** The same argument applies with `ĉ_i` in place of `m_i`. Eq. 17 holds, and the constant resampling weights satisfy the positivity condition, since `w̃_i > 0` iff `p̂_t(Y_i)W_iJ_i > 0`. The lemma applies. ∎

### 3.5 What goes wrong with approximate previous-frame evaluation

Let `π̃_p` be an approximate previous-domain evaluator (e.g. current lights, or a current-frame canonical suffix).

- **(i) Consistent approximation.** The same `π̃_p(·)` is used at every evaluation point. This means recomputing at `Y_p` too, as RTXDI and Bevy do.
  - Eq. 17 holds for every `y` producible by the temporal input, whatever the magnitude errors.
  - Where `π̃_p(y) > 0` but `π_p(y) = 0` (support violation), only the canonical produces `y` while `ĉ_c(y) < 1`.
  - Bias of the temporal output:
    `E − I = −∫_A F_t(y) · c_p π̃_p(y)/(c_c p̂_t(y) + c_p π̃_p(y)) dy ≤ 0`, with `A = {y : π_p(y) = 0 < π̃_p(y)}`.
  - Result: **darkening only**, and the sign propagates through spatial reuse, which is linear in the UCWs.
  - Where `π̃_p = 0 < π_p`: harmless (`ĉ_c = 1`).
- **(ii) Mixed.** Stored exact at `Y_p`, approximate at `X_c`. This is Falcor22, and Evan for analytic lights.
  - `m_c(y) + m_p(y) = a/(a+b̃) + b/(a+b)`, with `a = c_c p̂_t`, `b = c_p π_p`, `b̃ = c_p π̃_p`.
  - The sum is below 1 when `π̃ > π` (darkening) and above 1 when `π̃ < π` (brightening). This is the course-notes §7.1.4 sign rule. Magnitude errors therefore bias.
  - **Intensity step `r`** (only `Le(ℓ)` changes by `r`, one light): `π̃_p = r π_p` exactly. With `ρ = p̂_t/π_p ≈ r` (near-identity shift):
    `B(r) = 1/(1+c_p) − r/(r+c_p)`.

    | r | B(r) at c_p = 20 |
    |---|---|
    | 10 | −28.6 % |
    | 2 | −4.3 % |
    | 0.5 | +2.3 % |
    | 0.1 | +4.3 % |

    The bias then decays by roughly `c_p/(c_p+c_c) = 20/21` per frame, a half-life of about 14 frames, because the next temporal inputs carry the biased UCW [INFERENCE: near-identity static approximation].
- **(iii) Zeroing a class.** `π_p := 0` on a deterministic class of paths (e.g. deep suffixes on frames where lights changed), with `w_p := 0` for the temporal sample of that class.
  - Eq. 17 holds with `ĉ_c = 1` on that class, so the step is **unbiased**. It only loses temporal reuse there.
  - The predicate must be a function of `y` and of scene-level flags, never of sample values (course notes §4.4).
- **(iv) EvanLuo swapped arguments.** See §2.3. The sign is opposite to (ii).

### 3.6 Decision and full formulas

**Choice: contribution MIS in the Eq. 18/21 form, with exact `E_{t−1}`.**

Rationale:
1. Inverse shifts only happen when the canonical is selected (`P_c` of pixels) rather than always. The inverse shift is the most expensive part: replay into `q'`, a visibility ray and possibly a suffix refresh under `L_{t−1}`.
2. Unbiasedness hinges on one deterministic function `π_p`. The form tolerates approximate or random resampling weights (option E3, §5.4) and class zeroing (E2).
3. Production precedent: RTXDI.
4. In the static limit (`π_p ≈ p̂_t`, `J ≈ 1`) the weights coincide with Talbot's, so variance is the same.

Caveats and verification:
- GRIS §5.6: constant resampling weights forfeit the convergence-to-p̂ guarantee when inputs are non-canonical. Monitor with the harness efficiency ratio `ER`.
- Keep **Talbot-exact** (inverse shift always, stored self-term) as a debug mode. Both must pass §9.

**Per pixel `q`, frame `t`:**
```
c_c = 1;  X_c, W_c, F_c from initial sampling (F_c = F_t(X_c))
q'  = temporalPixel(q)                      // G-buffer tests + sample-independent RNG only
if (!valid(q')) { out = (X_c, W_c, c = 1, F_c); return }
(X_p, W_p, c_prev, F_p^st) = prevReservoir[q'];  c_p = min(20, c_prev)
translate light indices of X_p with prevToCur; if any is −1: Y_p undefined, w̃_p = 0

// forward shift under S_t (with suffix refresh, §5)
(Y_p, J_p, F_t(Y_p)) = T(X_p)               // J_p = Eq.2 reconnection Jacobian × light Jacobian J_P (§6)
w̃_c = c_c · lum(F_c) · W_c
w̃_p = c_p · lum(F_t(Y_p)) · W_p · J_p       // 0 if undefined or occluded
s   = select(w̃_c, w̃_p)                      // resampling RNG, never the replay RNG

if (s == p) { πc = lum(F_t(Y_p));  πp = lum(F_p^st) / J_p;   Y = Y_p; F = F_t(Y_p) }
else        { (Fprev, Jinv) = T^{-1}(X_c)  under S_{t-1}   // prev V-buffer x1', prev camera, prev lights and pmf,
                                                             // cur→prev light map (−1 ⇒ Fprev = 0), suffix under L_{t-1}
              πc = lum(F_c);  πp = lum(Fprev) · Jinv;  Y = X_c; F = F_c }
W_Y = (s == c ? πc : πp) / (c_c·πc + c_p·πp) · (w̃_c + w̃_p) / πc
out[q] = (Y, W_Y, c = c_c + c_p, F, refreshed caches, light indices in frame-t numbering)
```

- **Degenerate case.** If `w̃_c + w̃_p = 0`, output an empty reservoir with `W = 0` and `c = c_c + c_p` (GRIS p.75:6).
- **Talbot-exact debug variant.** Always compute `πp(X_c)` and use `m_c, m_p` from §3.2-1:
  `w_c = m_c(X_c)·lum(F_c)·W_c`, `w_p = m_p(Y_p)·lum F_t(Y_p)·W_p·J_p`, `W_Y = (w_c+w_p)/lum F(Y)`.
- **Robust variant.** In validation builds recompute `πp(Y_p)` with `E_{t−1}(T^{-1}(Y_p))` as well, and assert equality with the stored route. This is T6(b), and it catches I1 violations.

### 3.7 Confidence convention (one convention everywhere)

- Temporal: `c_p = min(c_cap, c_prev)` with `c_cap = 20`, and `c_out = c_c + c_p`.
  - Identical to ENH L129 (`min(c_Cap, c_temp)+1`).
  - Identical to Falcor22 `:139` `min(20·M_cur, M_prev)` when `M_cur = 1`.
- Spatial: `c_out = c_center + Σ_{valid j} c_j`, uncapped. The next temporal step caps it.
- Values are integers ≤ 21·(1+N) = 84 for N = 3, so they fit in ENH's 8-bit `M`.
  - Keep f32 in validation builds.
  - Any fractional cap (the duplication map) is biased (ENH §5) and is excluded from validation (critique X8).
- The capped `c_p` must be used both in the MIS weights and in `c_out`, as Falcor22 does. Mixing capped and uncapped values breaks nothing formally, but the weights must use one consistent value.
- Optional unbiased responsiveness: lower `c_cap` on frames where lights changed, globally or by a G-buffer distance to the changed light. This is sample-independent (course notes §4.4).

---

## 4. (b) Previous-frame state for the inverse shift

### 4.1 Exact list

| # | State | Why it is needed | Size | Notes |
|---|---|---|---|---|
| 1 | Previous camera position `o_{t−1}` (and `VP_{t−1}` for back-projection) | View direction at `x_1'`: `ω_o = normalize(o_{t−1} − x_1')`. Compute it from positions, **not** from a pixel-centre ray (Falcor's `computeRayPinholePrevFrame` assumes no jitter). Also the previous primary footprint `R_pri^{t−1}(q') = ‖o_{t−1}−x_1'‖²·4π/⟨n_{x_1'},ω_o⟩` for Enhanced Eq. 5 invertibility checks on the inverse path | uniform | RTXDI likewise uses the previous camera position for the previous-footprint test (enhanced-paper §3.10) |
| 2 | **Stored jittered previous V-buffer** (inst, prim, bary) per pixel | Defines `Ω_{t−1,q'}`. Replay starts at `x_1'` | 12–16 B/px (7.4–14.7 MB at 720p) | Recomputing the hit from `o_{t−1}`, the previous jitter and the static BVH is an alternative: 1 coherent ray per inverse shift. Previous depth and normal for validity tests are derivable from it plus static geometry |
| 3 | Previous light buffer `L_{t−1}` (transform, size, radiance/power/colour, spot angle and blend, type, one-sidedness, flags) | Light-vertex position `Φ_{t−1}(ℓ,u,v)`, `Le`, spot profile, BSDF-hit intersection of lights in suffixes, `A_{t−1}` | N × ~64 B | Double-buffer and swap |
| 4 | `curToPrev[]`, `prevToCur[]` (−1 = none) | Translate stored light indices: forward uses prev→cur, inverse uses cur→prev | 2N × 4 B | Built on the CPU from stable application IDs (as RTXDI `PrepareLights`, Bevy translation buffer) |
| 5 | Previous selection pmf `P_sel^{t−1}(ℓ)` | `p1^{t−1}` in `F_{t−1}` (the `1/p1` factor and `ω1`/`ω2`), and the light Jacobian `J_P` | N × 4 B | The alias *structure* (prob/alias arrays) is needed **only** if suffix NEE is replayed through it (option R2/R3) |
| 6 | Frame configuration: `M_B` (RIS candidates per bounce), max path length, footprint constant `c`, `α_min` | `ω1 = M_B p1/(M_B p1+p2)`; invertibility predicates | constants | If any change, reset history, or treat the old values as part of `S_{t−1}` |
| 7 | `lightsChanged` flags per light: moved / radiometric / added / removed | Decide the refresh work (§5.4). They never enter MIS except through the deterministic E2 rule | N bits | Computed on the CPU by diffing `L_t` against `L_{t−1}` |

### 4.2 The current BVH suffices (proof)

[INFERENCE, under the stated assumptions]
- Visibility `V(a,b)` depends only on the occluder set. Analytic lights never occlude (Cycles pass-through lights, critique X2), and triangles are static, so `V_{t−1} ≡ V_t` as functions of the endpoints.
- The inverse shift needs `V` between previous-domain endpoints. Those are `x_1'` and the replayed prefix vertices (the same triangle set), `x_k` (static), and `Φ_{t−1}(ℓ,u,v)`. All of them are computed from item 2, item 3 and `G`, never from a previous BVH.
- Prefix replay and suffix BSDF bounces hit the same triangles in both frames, because BSDF sampling does not depend on lights.
- Hence the current BVH evaluates `E_{t−1}` exactly.

**Caveat 1: BSDF rays that hit analytic lights** (classes B1, D-BSDF, E) must be intersected against the **previous** light set in `E_{t−1}`: a linear loop over `L_{t−1}`, or a small per-frame light BVH kept for one frame.

**Caveat 2: phase 2.** Once meshes or emissive instances move, a previous TLAS or previous instance transforms become necessary (webgpu-platform §4.5, Bevy `trace_visibility_previous_frame`, Evan `loadPreviousShading`).

### 4.3 Previous light tiles are not needed (proof)

[INFERENCE, from ENH S-§5 as transcribed in enhanced-paper §5.2]
- Enhanced keeps the PSS "single NEE sample with plain pdf `p1`" and treats RIS-NEE as a black box: the NEE sample's UCW is `W^{RIS}·p1`, which is folded into the path's `W`.
- So the target `p̂ = lum F`, with `F = ω1 f/p1` and `ω1 = M_B p1/(M_B p1 + p2)`, depends on RIS only through the deterministic `M_B` and the plain pdf `p1 = P_sel·(1/A)`.
  - A tile entry's *marginal* density is `p1`, because tiles are presampled from the power distribution.
  - Tile contents enter only `W^{RIS}`, which is a realized number stored in `W_p` and never re-evaluated.
- The inverse evaluator computes `p̂_{t−1}(T^{-1}X_c)`, a deterministic function of the path and `S_{t−1}`. No tile content appears.
- This holds **provided no light sampling is replayed**. Every NEE light vertex whose value must be re-evaluated is stored explicitly (§5.2 R1). With R2/R3 the replay uses the alias table (item 5) or counter-based "virtual tiles", not the frame's tile buffer.

### 4.4 Not needed

- Previous materials and textures (static).
- Previous duplication map (biased; off in validation).
- Previous motion vectors.
- Previous RR probabilities (RR affects only the initial `W`, ENH §6.2.4).
- Previous jitter offsets (the V-buffer carries the domain).

---

## 5. (c) Keeping `rcVertexRadiance` valid when lights change

### 5.1 Which reservoir quantities depend on lights

[INFERENCE, from the hybrid-shift factorization, gris-math §6.8, reference-code §1.5.5]

- **Light-independent:** the prefix and the reconnection factors (BSDFs, pdfs, visibility between surface points), and the suffix BSDF-bounce vertices and throughput `β_s = ∏_{j=k+1}^{d−2} f_j|cos_j|/p_j`. These are identical in both frames for static geometry, because suffix directions come from BSDF sampling or are copied, never from lights.
- **Light-dependent, and only at the path's end:**
  - NEE: `Le`, light position, one-sidedness, spot profile, `V(x_{d−1}, y_d)`, `p1` and `ω1`;
  - BSDF-hit emitter: which light the final ray hits, `Le`, `p1` of the hit point, `ω2`.
- The reconnection vertex choice `k` is light-independent. Enhanced Eq. 5 uses BSDF pdfs, geometry and the primary footprint only; forced NEE reconnection depends only on "no earlier rc". So the class of a path never changes under light edits.

### 5.2 Is re-tracing a suffix that contains RIS-NEE over per-frame tiles a valid deterministic temporal shift? **No.**

1. **No map.** In Enhanced's PSS, an NEE vertex is parameterized by hypothetical single-sample coordinates `U` under the plain sampler (ENH S-§5). The RIS's own random numbers are **not** coordinates of the path: the tile index for the 8×8 screen tile, the 1024 tile entries drawn from the frame's global RNG, the candidate slots and the selection number.
   - Replaying `rcRandomSeed` at frame t through RIS on frame-t tiles, as `traceTemporalUpdate` does through `generateLightSample`/`handleHit` (`PathTracer.slang:1753-1754`, `:1807`), yields a light point that depends on the frame-t tile buffer and on dimensions not stored in `X_p`.
   - So `Y_p` is not a function of `X_p`, and GRIS Def. 4.2 fails.
   - The inverse evaluation `π_p(X_c)` would need frame t−1's tile buffer **and** the canonical's RIS numbers, which are not stored.
2. **It is biased, with an explicit counterexample** [INFERENCE, exact arithmetic]. Take a fixed suffix vertex and two visible point lights:
   - unoccluded contributions `f_A = 1`, `f_B = 9`; equal power, so `p_A = p_B = ½`;
   - `F(ℓ) = f_ℓ/p_ℓ`, with true integral `Σf = 10`;
   - a steady-state reservoir sample over the NEE dimensions (∝ target) picks ℓ with `P(ℓ) = f_ℓ/Σf = (0.1, 0.9)` and UCW `W(ℓ) = p_ℓ/P(ℓ) = (5, 0.556)`.

   | Refresh draw | Expected result | Error |
   |---|---|---|
   | fresh RIS (target f, M→∞) | `Σ_old P W · Σ_new P_RIS(new) F(new) = 1 · (0.1·2 + 0.9·18) = 16.4` | **+64 %** |
   | fresh plain (uniform PSS) | `1 · 10 = 10` | exact here, but see below |

   The plain draw is exact here only because every NEE direction contributes. With occlusion, `E[W | prefix] = φ/p_prefix`, where `φ` is the PSS fraction of NEE dimensions with non-zero target, so a plain re-draw **darkens by `φ`**.
3. **"Suffix estimate as part of a random UCW."** This works only if the *whole* pipeline lives in the NEE-marginalized space:
   - `W` excludes `W^{RIS}p1`, and the random estimate `f·W^{RIS}` becomes part of the integrand;
   - **all** MIS (temporal *and* spatial) is contribution MIS with deterministic proxies `π` whose supports are contained in the true supports, e.g. a hashed single-sample NEE evaluation. By the §3.1 lemma, random resampling weights are fine;
   - Talbot/pairwise resampling MIS with random `p̂` is invalid (the `m_i` must be deterministic functions).
   - This is a redesign of Enhanced's reservoirs and spatial pass, so **not recommended**.

### 5.3 Reproducible alternatives (each is a valid deterministic shift)

- **R1: explicit light vertex (recommended).**
  - Store `(lightIdx, u, v)` for every NEE-terminated path (classes L, N1, D-NEE), wherever the NEE vertex is.
  - Refreshing re-evaluates the **same light point** under the target frame. This is a light-local shift (§6) with Jacobian `J_P`.
  - RIS-NEE with per-frame shared tiles stays exactly as in Enhanced: the RIS randomness lives in `W`, as for forced reconnection. Nothing is replayed.
  - Spatially the light point is copied with the suffix, which is identical to caching radiance.
- **R2: plain alias NEE on fixed counter-based RNG dimensions**, for NEE at suffix vertices `v > k`. RIS-NEE stays allowed at `v ≤ k`, where the light vertex is explicit.
  - The full suffix replay is then PSS random replay (Jacobian 1).
  - The path-MIS weight uses `M_v = 1` at those vertices. Since `M_v` is a deterministic function of the path prefix, the NEE/BSDF partition stays exact.
  - Downside: under pmf changes the replayed `u_sel` may jump to another light. Alias tables are not monotone, so one intensity edit reshuffles many lights. That adds variance, not bias.
- **R3: virtual tiles.**
  - RIS candidates at bounce B are `alias_sample(hash(seed, B, j))` for `j < M_B`, plus a fixed selection dimension.
  - Replay in the extended PSS (all RIS dimensions) is deterministic with Jacobian 1, and the extended-PSS integrand integrates to the right value because the RIS estimator is unbiased.
  - Costs `M_B` alias samples plus target evaluations per replayed NEE, and gives up the shared-tile cache coherence (irrelevant for tens of user lights).

### 5.4 The four options (plus two variants): bias status and cost

Costs are per refreshed sample and per evaluated direction (forward under `S_t`, inverse under `S_{t−1}`). The ms figures are in §8.

| Option | Mechanism | Bias status | Rays per refresh |
|---|---|---|---|
| **(i) Suffix re-trace on light-change frames** (Falcor22 `traceTemporalUpdate`, course notes fn 9) | Re-trace the suffix from `x_k` along the stored `ω_k` with the suffix seed | **Unbiased iff the suffix is reproducible** (R1/R2/R3) **and** done in both directions: forward always, inverse for the canonical whenever `π_p(X_c)` is needed. Falcor does only the forward direction, which is biased (§3.5-ii). With RIS over per-frame tiles it is **biased** (§5.2) | Deep suffix: `ℓ` closest-hit + 1 shadow (NEE end) or `ℓ+1` closest-hit (BSDF end), with `ℓ` = suffix BSDF bounces |
| **(i′) Recommended: R1 + suffix cache** | Cache `x_{d−1}` (hit), `ω_o` at `x_{d−1}`, `β_s` (NEE end), or `ω_{d−1}`, `t_occ`, `p2^{all}` (BSDF end). Re-evaluate the end term only, and only if its light changed | **Unbiased** (exact `F`) | Moved light: 1 shadow ray (NEE end), 0 BVH rays (BSDF end: light loop with `t < t_occ`). Radiometric-only change: 0 rays (analytic rescale). Unchanged: 0 |
| **(ii) Light-local re-evaluation of a single NEE segment** (class N1; class L inside the reconnection) | Stored `(lightIdx,u,v)`; recompute `y = Φ_s(ℓ,u,v)`, `Le_s`, `V`, `p1_s`, `ω1_s` | **Unbiased**; a special case of (i′). RTXDI does this for its NEE-light rc (§2.5) | 1 shadow ray if the light moved (class L: shared with the reconnection ray), else 0 |
| **(iii) ReSTIR GI-style periodic validation** (Ouyang 2021 §4.3: every 6 frames, same random numbers, clear on mismatch) | Periodic re-trace plus reset | **Biased**: stale between validations, and the reset is sample-dependent (course notes §4.4). A variant that *overwrites* `F` with the re-traced value instead of clearing keeps `W` a valid UCW (it doesn't depend on the target, §3.1) but violates I1 between validations, so it is still biased through the MIS | Cost of (i) amortized over N frames (1/N of pixels per frame) |
| **(iv) Accept staleness** | Keep cached `L` | **Biased.** Shading uses the wrong integrand for about the sample lifetime (≈ `c_cap` frames, longer via spatial propagation), plus the §3.5-ii MIS bias. This is the "lag" of ReSTIR GI | 0 |
| **(v) E2: class zeroing** (§3.5-iii) | On frames where any light changed, set `π_p := 0` and `w̃_p := 0` for classes D-NEE and D-BSDF | **Unbiased**; loses temporal reuse for deep-suffix paths during light motion | 0 for class D; N1/B1/L as in (ii) |
| **(vi) E3: selected-only refresh** (contribution MIS, §3.1) | Resample with a stale-suffix weight `w̃_p = c_p · lum(prefix_t(Y_p)) · lum(L_{t−1}) · W_p J_p`, which is positive wherever `F_t(Y_p)` can be. Refresh the suffix only if `s = p`; use Eq. 18: `W_Y = ĉ_s(Y)W_sJ_s·Σw̃/w̃_s` | **Unbiased** | One refresh per pixel instead of `1 + P_c`. Optional; worthwhile only without the suffix cache |

**Decision.**
- Implement **(i′) R1 + suffix cache**; it contains (ii).
- Keep **E2** as a low-memory fallback, and E3 as an optional optimization.
- Never use (iii) or (iv) in validation.
- Never replay RIS over per-frame tiles.

### 5.5 Exact refresh formulas (frame selector `s ∈ {t−1, t}`)

- **End-term for NEE at vertex `x` with incoming direction `ω_o`:**
  - `y = Φ_s(ℓ_s,u,v)`, `ω = normalize(y − x)`, `d = ‖y − x‖`;
  - area light: `N_s = f(x;ω_o,ω)|n_x·ω| · V(x,y) · Le_s(y,−ω) · ω1_s / p1σ_s`, with
    - `p1σ_s = (P_sel^s(ℓ_s)/A_s(ℓ_s)) · d²/|n_ℓ^s·ω|`;
    - `ω1_s = M_B p1σ_s/(M_B p1σ_s + p2(x;ω_o→ω))`;
    - `Le_s = 0` on the back side (one-sided rect or disk);
  - point/spot light: `N_s = f(x;ω_o,ω)|n_x·ω| · V · I_s(−ω)/d² · 1/P_sel^s(ℓ_s)`, with `ω1 = 1`;
  - spot intensity: `I_s = (Φ/4π)·smoothstep((cos α − cos(s/2))/((1−cos(s/2))β))`, with `α` measured against the frame-s axis (validation-harness §7.2).
- **Class D-NEE:**
  - `L_k^{(s)} = β_s ⊙ N_s(x_{d−1}, ω_o)`, with all inputs cached or stored;
  - `F_s(y) = T_prefix ⊙ (f_{k−1}/p^y_{k−1}) ⊙ (f_k(ω_in^y, ω_k)/p^y_k) ⊙ L_k^{(s)}`;
  - Jacobian: Eq. 2 × `J_P`.
- **Class N1:** as the reconnection factor with the NEE end-term evaluated at `x_k`, with `ω_o = −ω'_{k−1}` of the *offset* path. Jacobian: Eq. 2 without `p_k` (the NEE pdf does not depend on `ω_in`) × `J_P`.
- **Class L:** `F_s = T_prefix ⊙ N_s(y_{d−1}, ω_o^y)`, with `J = J_P`.
- **Class B1, D-BSDF:**
  - cast the final ray from `x_{k}` or `x_{d−1}` along the stored `ω` against the analytic lights of frame s, over `t < t_occ`. `t_occ` is the static nearest-triangle distance, cached once;
  - emitter term `Le_s · ω2_s`, with `ω2_s = p2^{all}/(p2^{all} + M_B p1σ_s(hit))`;
  - the direction is copied (PSS replay), so `J` is unchanged: Eq. 2 including the `p_k` ratio (gris-verify C3).
- **Radiometric-only change** (moved = false): `N_t = N_{t−1} · (Le_t/Le_{t−1}) · (ω1_t/ω1_{t−1}) · (p1_{t−1}/p1_t)`. No rays.
- **Quantization rule.** If `(u,v)` or barycentrics are stored as unorm16, quantize **before** computing the initial `F`, so every evaluation sees the same point. Validation builds keep f32.

### 5.6 Storage

- **Enhanced 64 B reservoir** (S-Alg.1, enhanced-paper §6.2): `W`, `F`, `initRandomSeed`, `rcVertexRandomSeed`, `pathFlags` with 8-bit `M`, rc (inst, prim, bary unorm16), `rcVertexWi` (oct), `rcVertexRadiance`, and `rcVertexCachedValues`.
- **Add two `vec4<u32>` planes (32 B), giving 96 B.** SoA planes are read only on light-change frames or when the sample is selected (webgpu-platform §5.5 AoS/SoA measurement).

| Plane E | Plane F |
|---|---|
| `x_{d−1}` inst, prim, bary (12 B), `ω_o` or `ω_{d−1}` oct (4 B) | `β_s` (f32×3, or f16×3 in performance builds) + one of `{lightIdx, uv}` or `{t_occ, p2^{all}}` |

- For class N1/L the light `(idx, uv)` goes in plane F; for class L, `x_{d−1}` is not needed.
- `pathFlags` gains a 3-bit class field. `rcVertexRadiance` keeps its role on static-light frames and is overwritten by each refresh.
- **Memory at 1280×720:** 3 buffers × 96 B = 265 MB, versus 177 MB at 64 B. At 960×540: 149 MB.
- **Filling the cache during initial sampling costs nothing extra.** Falcor's post-rc throughput `thp` restarts at 1 after the rc vertex (reference-code §1.5.2 step 8), so at an NEE or emitter candidate `β_s = thp`, `x_{d−1}` is the current hit, and `(ℓ,u,v)` is the NEE sample.

---

## 6. (d) Forward shift of light vertices and its Jacobian

- **Parameterizations** (fixed per light type, used identically in both frames) [INFERENCE]:
  - Rect `a×b`: `Φ_s(u,v) = c_s + (u−½)a_s ê_u^s + (v−½)b_s ê_v^s`, with `n_s = R_s n_0` and `A = ab`.
  - Disk/ellipse: `Φ_s(u,v) = c_s + ρ(u,v)` using a fixed area-uniform map (e.g. concentric), scaled by `(r_x, r_y)`, with `A = π r_x r_y`.
  - Point/spot with radius 0: `Φ_s = p_s` (delta). The orientation `R_s` enters only through `I_s(ω)`.
  - Emissive triangles (static): (inst, prim, bary), i.e. the identity.
- **Area-measure Jacobian of `x_d ↦ y_d`:** `J_M = |∂Φ_t/∂Φ_{t−1}| = A_t/A_{t−1}` for any affine rescale.
  - It equals **1 for rigid motion** (translation and rotation).
  - Rect: `(a_t b_t)/(a_{t−1} b_{t−1})`. Ellipse: `(r_x r_y)_t/(r_x r_y)_{t−1}`.
  - Delta lights: 1 (point to point, discrete measure).
- **PSS Jacobian for the NEE dimensions** (the Enhanced/Falcor convention, reservoir `F = ωf/p`) [INFERENCE, change of variables `ū_x → x_d → y_d → ū_y`]:
  `J_P = [p_A^t(y_d)/p_A^{t−1}(x_d)] · J_M`.
  - With uniform-area sampling `p_A = P_sel/A`, this gives `J_P = P_sel^t(ℓ_t)/P_sel^{t−1}(ℓ_{t−1})`: the area terms cancel.
  - Delta lights: `J_P = P_sel^t/P_sel^{t−1}`.
  - Spatially (same frame) `J_P = 1`, matching enhanced-paper §5.4 and restir-practice §3.5.
  - Consistency check: `F_t(y) J_P = ω_t f_t(ȳ) J_M/p^{t−1}(x̄)`, which equals the path-space GRIS weight (gris-verify §2.2(e)).
  - With solid-angle light sampling (spherical rectangles), `p_A` depends on `y_{d−1}` and must be evaluated at the offset vertex (gris-verify C2). Use uniform-area sampling in v1.
- **Full temporal Jacobian:**
  `J = J_prefix (=1, replay) × J_rc (Enhanced Eq. 2 at x_k, unchanged, static geometry) × J_suffix (=1, copied or replayed) × J_P (NEE-terminated paths only)`.
  - For class L, Eq. 2 with `k = d` and area sampling contributes `p_A^y/p_A^x`, which the light-local map extends to `J_P`.
  - For a BSDF-hit analytic light used as the rc vertex (class E): `J = [p^y_{k−1}(ω')G(y_{k−1}→Φ_t(u,v))]/[p^x_{k−1}(ω)G(x_{k−1}→x_k)] · J_M`, with the cosine at the frame-t light normal.
- **RTXDI reconciliation** [UNVERIFIED equivalence]. RTXDI uses `J = pdf_σ,src/pdf_σ,dst` for the NEE-light rc (`HybridShift.hlsli:305-313`) in its mixed-measure target. For a fixed light point that is the solid-angle reconnection factor `G_dst/G_src`. The difference from `J_P` is absorbed by its target definition (reference-code §2.2.2). Verify by T4/T5 on our own code, not by porting.
- **Failure vs zero contribution.**
  - Back-facing on a one-sided light, outside the spot cone, or occluded: the shift is defined with `F = 0`. This is fine for MIS, since `π = 0` either way.
  - An unmapped light index (removed or added): the shift is **undefined** (`w = 0`, `π = 0`).
  - Keep the two cases symmetric in the forward and inverse code.

---

## 7. (e) Lights that are added or removed (and other edits)

| Event at frame t | Forward shift (temporal `X_p` → frame t) | Inverse `π_p(X_c)` | Bias (exact design) |
|---|---|---|---|
| **Light ℓ removed** (`prevToCur[ℓ] = −1`) | Explicit vertex on ℓ (classes L, N1, D-NEE): **undefined**, `w̃_p = 0`, matching Bevy `:109-110` and RTXDI `:251-254`. BSDF-terminated: re-intersection misses ℓ, so contribution 0 (defined). R2 replay: `u_sel` maps through the frame-t table to another light (defined) | Canonical never involves ℓ | None. `P_sel` of all other lights changes, so `J_P ≠ 1` for every explicit NEE sample |
| **Light ℓ added** (`curToPrev[ℓ] = −1`) | No temporal sample can involve ℓ | Canonical on ℓ (explicit): **undefined**, so `π_p = 0` and `ĉ_c = 1` (exact support). Canonical BSDF-terminated suffix: ℓ absent from `L_{t−1}`, ray re-intersected accordingly | None |
| Intensity/colour/spot-angle change | Radiometric rescale, no rays; `J_P` from the new pmf | Evaluated with `L_{t−1}` values | None |
| Move/rotate/scale | 1 shadow ray per explicit NEE sample on that light; `J_M` for scale; light loop for BSDF ends | Same, under `L_{t−1}` | None |
| Type change or shape-topology change | Treat as remove + add (new stable ID) | — | None |

**Implementation.**
- Stable application IDs per light. Each frame, compact the GPU array, rebuild the pmf and alias table deterministically (stable ID order, identical float operations), and build both maps.
- Skip the rebuild when nothing changed, so that `P_sel^t ≡ P_sel^{t−1}` bitwise and `J_P = 1` exactly.
- Reservoirs store the light **index of the frame they were written in**. Translate on read.

---

## 8. (f) Per-pixel counts and cost

### 8.1 Counts per temporal candidate

`R` is the expected number of replay closest-hit rays per shift, i.e. `Σ_k P(k)(k−2)` including class L with `k = d`.

| Item | Forward shift | Inverse shift | Refresh (light-change frames, per direction) |
|---|---|---|---|
| Talbot (any frame) | `R` ch + 1 vis | **always**: `R` ch + 1 vis | both directions |
| Contribution MIS (any frame) | `R` ch + 1 vis | **only if s = c** (prob. `P_c`) | forward always, plus inverse if `s = c` |
| Class L | — | — | 0 extra (the reconnection visibility ray is the refresh) |
| Class N1 (R1) | — | — | 1 shadow if its light moved, else 0 |
| Class B1 / D-BSDF (cache) | — | — | 0 BVH rays (light loop vs `t_occ`); without `t_occ`, 1 ch |
| Class D-NEE (cache) | — | — | 1 shadow if its light moved |
| Class D, no cache (R1 + BSDF replay, or R2) | — | — | `ℓ` ch + 1 shadow (NEE end) or `ℓ+1` ch (BSDF end) |
| Radiometric-only change | — | — | 0 rays |

Camera-only motion needs **no** refresh. The cached suffix is exact because geometry and lights are static and the suffix does not depend on the camera.

### 8.2 Conversion to ms

`t = N_px · (n_ch/R_ch + n_vis/R_vis)`, with inputs from webgpu-platform §4.1 [MEASURED there]:
- `R_ch ≈ 192 Mrays/s` (incoherent closest-hit secondary);
- `R_vis` for incoherent short any-hit rays is unmeasured (critique Gap 5). I use 300 Mrays/s as the central value, bracketed by 192 and 610 (coherent shadow).

| Rays per pixel → ms | closest-hit @192 | vis @300 (range 610–192) |
|---|---|---|
| 960×540 (518,400 px) | 2.70 ms | 1.73 ms (0.85–2.70) |
| 1280×720 (921,600 px) | 4.80 ms | 3.07 ms (1.51–4.80) |

**Workload A, typical diffuse-dominant** [INFERENCE]:
- temporal hit rate `h = 0.95`, `R = 0.3`, `P_c = 0.1`;
- class mix: L 0.25, N1 0.25, B1 0.10, D-NEE 0.30, D-BSDF 0.10;
- deep-suffix extra bounces `ℓ̄ = 1.3`;
- fraction of samples attached to the moved light `φ_mv = 0.5`.

| Scheme | rays/px (ch, vis) | 960×540 ms | 1280×720 ms |
|---|---|---|---|
| **Base temporal (every frame), Talbot** | 0.57, 1.90 | **4.8** (3.2–6.7) | **8.6** (5.6–11.9) |
| **Base temporal, contribution MIS** | 0.31, 1.05 | **2.7** (1.7–3.7) | **4.7** (3.1–6.5) |
| + light-change, naive full suffix re-trace, Talbot (both dirs) | +1.37, +1.05 | +5.5 (4.6–6.5) | +9.8 (8.2–11.6) |
| + light-change, naive full re-trace, contribution MIS | +0.75, +0.57 | +3.0 (2.5–3.6) | +5.4 (4.5–6.4) |
| + light-change, **R1 + cache, moved-only**, Talbot | 0, +0.52 | +0.9 (0.4–1.4) | +1.6 (0.8–2.5) |
| + light-change, **R1 + cache, moved-only, contribution MIS** | 0, +0.29 | **+0.5** (0.2–0.8) | **+0.9** (0.4–1.4) |
| + intensity/colour/add/remove only (R1 + cache) | 0, 0 | ≈ 0 (ALU only) | ≈ 0 |

**Workload B, glossy worst case** (`h = 0.9`, `R = 1.0`, `P_c = 0.3`):
- base contribution MIS: 1.17 ch + 1.17 vis, giving **5.2 ms** (4.2–6.3) at 540p and **9.2 ms** (7.4–11.2) at 720p;
- base Talbot: 1.8 + 1.8, giving 8.0 ms (6.4–9.7) and 14.2 ms (11.4–17.3).

**Not included:**
- reservoir traffic, about 0.3 ms (540p) to 0.75 ms (720p) for roughly 1.5 × 64 B read+write, scaled from the measured 1.1 ms per 64 B r+w at 1080p. Add about 0.15–0.3 ms for planes E/F on light-change frames;
- BSDF/ALU;
- indirect-dispatch overhead (tens of µs per queue).

These figures fit the webgpu-platform §5.7 temporal budget (3–6 ms at 540p, 6–10 ms at 720p) and should be replaced by the Gap-5 microbenchmarks.

### 8.3 WebGPU pass layout for the temporal step [INFERENCE]

- **T1 (classify):**
  - per pixel: temporal-pixel selection, light-index translation, `c_p`;
  - inline reconnection-only forward shifts (rc = x2, or class L with `d = 2`);
  - append pairs needing forward replay to queue Q_f (subgroup-ballot append, webgpu-platform §5.6).
- **T2:** `dispatchWorkgroupsIndirect(Q_f)`, forward replay + reconnection + refresh under `S_t`.
- **T3 (resample):**
  - `w̃_c`, `w̃_p`, select `s`;
  - if `s = p`, finalize with the stored `π_p`;
  - if `s = c`, append to Q_i.
- **T4:** `dispatchWorkgroupsIndirect(Q_i)`, inverse replay + reconnection + refresh under `S_{t−1}`, finalize `W_Y`.

Rules:
- Keep exactly one BVH-traversal call site per pipeline; loop over "jobs" instead of duplicating calls (the C2-Renderer compile-time lesson, reference-code §2.4).
- Frame-selector uniforms choose `L_t`/`L_{t−1}`, the pmf, the camera and the V-buffer in the shared shift code (the RTXDI `isPrevFrame` pattern).
- Storage budget: `lightsCur`, `lightsPrev`, maps and pmfs go into one "scene records" buffer (webgpu-platform §5.2, ≤ 10 storage buffers).

---

## 9. (g) Tests

### 9.1 Per-frame ensemble tests

Harness defaults from validation-harness §3.6/§3.8:
- R ≥ 256 independent runs (1024 for sign-off) of the same scripted sequence;
- frame-t ensemble mean vs our PT mean at the identical frame state, tile δ = 1% (TOST), plus drift regression over frames;
- Cycles keyframes every 8th frame with K = 8 seeds, BOX 1.0, per validation-harness §7;
- jitter on, uniform per pixel per frame from a stream independent of the replay RNG.

Masks come from the PT reference at frames t−1 and t:
- `M_new`: `L_t > 0` and `L_{t−1} = 0` (newly lit, or light added);
- `M_gone`: newly dark;
- `M_edge`: shadow or cone edges;
- `M_steady`: everything else;
- `M_disocc` (camera scenes).

| Scene | Script (≥ 64 frames) | What it isolates | Test frames |
|---|---|---|---|
| **ix-a** Cornell + point light | Linear path at about 3 px/frame of shadow-edge motion. Passes behind a box at frames 20–30 | Explicit class L/N1 light-local shift, previous visibility with the current BVH, `M_new` from unshadowing | 1, 10, 25 (occluder edge), 40 (first frame after stop), 40 + 2·c_cap |
| **ix-b** rect area light | Translate + rotate 5°/frame (one-sided sweep); **area ×1.5 step at frame 16**; **intensity ×2 step at frame 24** | `J_M` (scale), `J_P` and `Le` (intensity), one-sidedness support | 15, 16, 17, 24, 25, 40 |
| **ix-c** spot light | Rotating 2°/frame; blend 0 (hard edge) and 0.3 | Cone-edge support violations (`M_new` = leading edge) | 5, 20, 40 |
| **ix-d** fly camera, static lights | Scripted path plus one recorded WASD path; also **ix-d0**: static camera with jitter only | Previous V-buffer, previous camera, R_pri^{t−1}, disocclusion. **No refresh ever runs**, which checks the cache | every 8th |
| **ix-e** (new) two lights, P 1:1 | Light C added at 8, intensity ×2 on A at 14, B removed at 20 | Light maps, pmf Jacobian `J_P`, add/remove supports | 8, 9, 14, 15, 20, 21 |

**Pass criteria:**
- All exact modes pass every tested frame at tile δ = 1% and global δ = 0.2%: contribution MIS (default), Talbot-exact, E2 fallback, E3.
- Drift slope ≈ 0.
- The fraction of failing tiles per frame follows Binomial(m, α′).

### 9.2 T6 extensions

- **T6(a) partition of unity.** Random `(c_c, c_p ∈ {0…20}, p̂_t, π_p)` including zeros and a `c_cap`-clamped value.
  - Talbot: `|m_c + m_p − 1| < 1e-6`.
  - Contribution MIS: `ĉ_c + ĉ_p = 1` where both are producible, and `ĉ_c = 1` where `π_p = 0`.
  - E2: `ĉ_c ≡ 1` on class D on light-change frames.
- **T6(b) stored vs recomputed.** In a debug build, for every pixel with a valid temporal candidate:
  - compute `a = lum F_p^{st}` (stored) and `b = lum F_{t−1}(T^{-1}(T(X_p)))` (fresh inverse evaluation);
  - compare IDs of `T^{-1}(T(X_p))` with `X_p`: rc (inst, prim, bary quantized), light (idx, uv), seed and class;
  - require `|a−b| ≤ 1e-3·max(a,b)` and ID equality on all pixels. Report counts; the target is 0, with rare float-threshold cases logged.
  - Run it in all ix scenes. With the planted N1 it **must fail exactly on light-dependent pixels**. With N3 (no refresh) it fails on `M_steady` after a move (I1 broken).
- **T6(c) previous-state provenance and cross-evaluator.**
  - Every evaluator call writes `(camGen, vbufGen, lightGen, pmfGen)` to a debug buffer. Assert `== t−1` for inverse evaluations and `== t` for forward ones.
  - Cross-evaluator: for 10^5 random canonical samples, reconstruct `T^{-1}(X_c)` as explicit vertices and evaluate it with the plain PT's "path contribution from vertices" routine under `S_{t−1}`. Require equality with `π_p(X_c)/Jinv` within 1e-4.

### 9.3 New unit tests

1. **Light maps.** Random add/remove/reorder sequences give `curToPrev[prevToCur[j]] = j` for surviving lights, and −1 exactly for added or removed ones.
2. **Light-local Jacobian.**
   - Finite-difference: map a small `(u,v)` triangle from the previous to the current light; the area ratio should equal `A_t/A_{t−1}` within 1e-5.
   - PSS change of variables (T5 style): `E_{x∼prev sampler}[h(T x)·J_P·1{x∈D}] = E_{y∼cur sampler}[h(y)·1{y∈I}]` with two lights and an intensity change, 10^8 samples, z-test.
3. **Refresh idempotence.** On a frame with `lightsChanged = false`, running the refresh changes `F` by ≤ 1e-6 relative. This validates the suffix cache and `β_s`.
4. **Refresh vs full re-trace.** On a light-change frame, the cached refresh (R1 + cache) equals a full BSDF-dimension replay + NEE to the stored light point within 1e-4 (tests `β_s`, `x_{d−1}`, `ω_o`).
5. **Identity.** With jitter off and a static camera and lights, `T` is the identity, `J = 1`, and `F_t(Y_p) = F_p^{st}` (critique X1: only with jitter off).
6. **Class invariance.** The class and `k` of `T(X_p)` equal those of `X_p` for 10^7 shifts, including across light edits.

### 9.4 Planted negative controls with predicted outcomes

| ID | Plant | Expected sign and magnitude | Where |
|---|---|---|---|
| **N1-mixed** | Inverse evaluator uses `L_t`, pmf_t and the current-frame canonical suffix; the temporal self-term stays stored (**Falcor22 behaviour**) | ix-b/ix-e intensity step r = 2: global relative bias on the stepped light's contribution ≈ **B(2) = −4.3 %** at the step frame, decaying ≈ ×20/21 per frame (half-life ≈ 14). r = 0.5 gives **+2.3 %** (ix-e variant). Moving lights: **darkening where the light approaches, brightening where it recedes**, plus support darkening in `M_new`. ix-e add at frame 8: **≈ −95 %** of C's contribution (`1/(1+20)`), recovering as `1−(20/21)^{n+1}` | ix-a/b/c/e; **inert in ix-d** (no light change), which is a sanity check |
| **N1-consistent** | Same approximate evaluator, but `π_p(Y_p)` is also recomputed with it (Bevy/RTXDI style) | Intensity step: **no bias** (magnitude only). Moving lights: **darkening only**, confined to `M_new` (unshadowing band, spot leading edge, one-sided rect sweep). Add-light: −95 % as above (support) | Tests the §3.5-i theory: sign must be ≤ 0 in every tile |
| **N2** | Omit `J_P` (use `J = 1` for light vertices) | ix-e frame 14: P_A 0.5→0.667 and P_B 0.5→0.333. Temporal contributions on A are scaled by 0.75 (**darkening of A's light**); on B by 1.5 (**brightening**). Also at frames 8 and 20 (renormalization) | Only frames where the pmf changes |
| **N3** | No refresh (stale suffix, option iv) | Lag: after a move, an indirect-light bias (sign follows the change: the old brighter region stays too bright, the new one too dark), persisting beyond the stop frame and decaying slower than ×20/21 per frame because of spatial propagation. T6(b) fails | ix-a/ix-b frame 40 onward |
| **N4** | Refresh replays NEE through **fresh per-frame RIS tiles** (Falcor-style re-draw under Enhanced tiles) | ix-e-like two-light set-up with equal power and distance ratio 3 (`f_B/f_A = 9`): class D-NEE suffix contribution **brightens**. The M→∞ limit is +64 % (§5.2); finite `M_B` (8 at B = 2) gives less. With heavy occlusion the sign can flip (plain-redraw darkening). Only on light-change frames | Isolate by forcing all paths into class D (roughness settings) |
| **N5** | Inverse shift uses a pixel-centre (un-jittered) previous primary ray instead of the stored jittered V-buffer | Contribution MIS: darkening at silhouettes and edges (support). Talbot-stored: mixed sign on glossy primaries | ix-d0, ix-d |
| **N6** | Inverse shift uses the current camera position for `ω_o` at `x_1'` | Magnitude errors on glossy primaries; contribution MIS mostly unbiased except at glossy-lobe support edges; Talbot-stored mixed sign | ix-d with glossy floor |

Detection: planted biases of ≥ 2% over masked regions are well above the harness MDB at R = 256 (validation-harness §3.4, §3.9). Each control must be detected with the predicted sign. This also tests the statistics code.

---

## 10. Risks and open questions

1. **Incoherent any-hit throughput is unmeasured.** All ms numbers bracket it between 192 and 610 Mrays/s. Run Gap-5 microbenchmark (a).
2. **Pass-through light semantics (critique Gap 1).** The B1/D-BSDF refresh assumes the final BSDF ray's emitter term depends only on the first analytic light hit before `t_occ`. If Cycles adds emission from **every** light passed through, or counts transparent bounces toward a limit that moving lights can change, the path-tree candidate definition and the refresh must follow [UNVERIFIED].
3. **Variance of contribution MIS vs Talbot under strong light motion.** Not measured. The GRIS §5.6 caveat applies. Compare `ER(t)` for both on ix-a/b/c.
4. **Float asymmetry between pipelines.** The stored `F` is produced by the spatial pass and the recomputed one by the temporal pass (FMA/relaxed math, webgpu-platform §8.1). Values agree to ~1e-6, but discrete invertibility decisions near thresholds can flip. T6(b) and T3 must count these and keep them rare.
5. **Suffix cache memory**, +32 B per reservoir (265 MB for 3 buffers at 720p). If it is too much, fall back to E2 (zero cost, loses deep-suffix history during light drags) or to R1 + BSDF replay (§8: about +1 ms at 540p).
6. **Emissive-mesh lights that move** (a GLB instance) break the static-geometry assumption. They need a previous TLAS (phase 2).
7. **ENH S-§5 PSS details**, i.e. the exact form of `ω1` with RIS and `W = W^{RIS}·p1`, are taken from enhanced-paper §5.2's transcription of the supplement, not re-read here [UNVERIFIED re-check].
8. **The `B(r)` closed form** assumes a near-identity shift and a single light. Its sign is exact under (§3.5-ii); its magnitude with jitter and spatial reuse should be treated as ±30% [INFERENCE].

---

## 11. Sources

- **GRIS 2022:** page images `scratchpad/pages/gris/p-06.png`, `p-07.png`, `p-23.png` (Eq. 10–22, §4.3, App. B).
- **Enhanced 2026:** `scratchpad/restirpt_enhanced_2026.txt`, lines 118–132, 495–535, 586–606, 792–802.
- **Course notes:** https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf (fetched with a browser user-agent; text extracted in memory). §4.3 p.19, §7.1.4 p.41–42, footnotes 8–9 p.33.
- **Falcor22 (DQLin/ReSTIR_PT master, BSD-3):**
  - https://raw.githubusercontent.com/DQLin/ReSTIR_PT/master/Source/RenderPasses/ReSTIRPTPass/TemporalReuse.cs.slang (L55–313)
  - https://raw.githubusercontent.com/DQLin/ReSTIR_PT/master/Source/RenderPasses/ReSTIRPTPass/Shift.slang (L25–200, L570–660)
  - https://raw.githubusercontent.com/DQLin/ReSTIR_PT/master/Source/RenderPasses/ReSTIRPTPass/PathTracer.slang (L1722–1845)
- **Evan (EvanLuo42/ReSTIR-PT-Enhanced HEAD):** `Source/RenderPasses/ReSTIRPTPass/Temporal/{TemporalShift,TemporalReuse,TemporalReplay}.cs.slang`, `Common/{GRIS,HybridShift,Lights,SurfaceLoad}.slang`, `Common/SurfaceTypes.slang:80`.
- **Bevy:** https://raw.githubusercontent.com/bevyengine/bevy/main/crates/bevy_solari/src/realtime/restir.wesl (commit a173a349, 2026-09-28), L36–39, L75–118, L179–300.
- **RTXDI-Library (proprietary; semantics only):** `Include/Rtxdi/PT/TemporalResampling.hlsli` L215–218, L327–360, L425–445; `Include/Rtxdi/PT/HybridShift.hlsli` L230–313.
- **Prior reports (plans dir):**
  - restir-practice `…a668fe13fb61070e8.md` §0–3, §5.3–5.4
  - gris-math `…a7f08dec9968f0324.md` §6.8–6.9, §7.4
  - gris-verify `…a147516e5d2fb9558.md` C1–C4
  - reference-code `…a3b89657b22d15f34.md` §1.4–1.7, §2.2–2.3
  - enhanced-paper `…a91ab1d861819b276.md` §4–5, §7
  - validation-harness `…a26c6df2c3f5e4af0.md` §2.2, §3.4–3.9, T1–T16, §7–9
  - webgpu-platform `…a64ceeae6903be301.md` §0, §4.1, §4.5, §5.2–5.7, §8
  - critique `…a40267d04641507fa.md` X1–X3, X7–X8, Gap 2
