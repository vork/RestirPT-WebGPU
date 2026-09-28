# ReSTIR in Practice: Temporal Reuse Under Motion, Robustness, and Display

Research report for the WebGPU ReSTIR PT project (topic: broader ReSTIR literature, dynamic scenes, robustness, denoising/display)

> Intended location: `/private/tmp/claude-502/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-ea73-491f-b7b7-6d0aab1ffce1/scratchpad/research/restir-practice-dynamics.md`. This session ran in plan mode, which allowed writing only to this plan file, so the report lives here. Copy it to the intended path when plan mode ends.

Conventions:
- **[INFERENCE]** marks my own derivation or recommendation, as opposed to what a source says.
- **[UNVERIFIED]** marks a statement I could not confirm against a primary source during this session.
- Citations give paper, section, equation, page, file path and line numbers. Local files are:
  - `ENH` = `scratchpad/restirpt_enhanced_2026.txt` (Lin, Kettunen, Wyman 2026) and page images `scratchpad/pages/enhanced/p-XX.png`
  - `GRIS` = `scratchpad/gris_sig22.txt` / `scratchpad/pages/gris/p-XX.png` (Lin et al. 2022). The GRIS text extraction is badly garbled, so I cite GRIS by printed page (75:N) and equation number, which I checked against the page PNGs.
- Following the session's copyright rules, I paraphrase sources and do not reproduce long verbatim passages or proprietary code. Equations are restated in my own notation.

---

## 0. Executive summary (read this first)

1. **ReSTIR's per-frame estimator is unbiased under motion only if the temporal GRIS step uses a valid temporal shift and valid MIS weights.** Valid MIS weights need the *previous frame's* target function evaluated on the canonical (current) sample after inverse-shifting it into the previous frame. That evaluation uses the previous camera, previous primary hit, previous light state, previous geometry/visibility, and the previous frame's cached suffix radiance. The GRIS paper says fully unbiased temporal reuse must evaluate paths in both frames and calls this tricky in dynamic scenes (GRIS App. B, p. 75:23). The course notes say access to previous frames' target functions is required to remove all bias (§4.2, p. 17) and discuss previous-frame visibility and BVH (§4.3, p. 19; §7.1.4, p. 41–42).
2. **Sign of the bias when current-frame data stands in for previous-frame data** (course notes §7.1.4). Suppose the prev-frame density of the current sample, \(p_j(Y_i)\), is replaced by \(\tilde p_j\). Overestimating it darkens the result, an exact value adds no bias, and underestimating it brightens. Assuming \(p_j(Y_i)=0\) ("could not have come from last frame") overcounts and brightens. My derivation in §4.4 [INFERENCE] adds that the damage depends on which MIS form you use:
   - **Talbot / generalized-balance resampling MIS.** Mixing an *exact stored* \(\hat p_{prev}\) (for the temporal sample) with an *approximate recomputed* \(\hat p_{prev}\) (for the canonical sample) breaks the partition of unity, so both magnitude and support errors bias the result.
   - **Contribution MIS with constant (confidence-proportional) resampling weights.** This is the Bitterli 2020 / RTXDI form. The same recomputed \(\tilde p_{prev}\) appears in both numerator and denominator for the selected sample, so only *support* errors (\(\tilde p_{prev}>0\) where the previous reservoir could not have produced the path) cause bias. Magnitude errors only cost variance.
3. **Our project needs static geometry, a moving camera, and moving/added/removed/re-intensified lights. That makes exact unbiased temporal reuse affordable [INFERENCE]:**
   - The BVH is static, so previous-frame visibility equals current-frame visibility for any fixed segment. No previous BVH is needed.
   - Keep lights as analytic primitives in a separate small light buffer, outside the triangle BVH, with **current and previous** copies plus a prev↔cur index map. This follows RTXDI's layout, `RAB_TranslateLightIndex`, and `PrepareLights.hlsl` L185–204.
   - Store light-sample vertices in **light-local coordinates** (light id + (u,v)), never world space. A moved light then carries its samples along. The shift Jacobian is 1 in PSS, or the area ratio in area measure.
   - Evaluating \(\hat p_{prev}\) means running the same shading code with the prev camera position, the prev V-buffer primary hit, and the prev light buffer and light-selection tables.
   - The one expensive part is **cached suffix radiance** at the reconnection vertex (ReSTIR GI's \(L_o\), ReSTIR PT's `rcVertexRadiance`). When lights change it is stale in the forward shift and wrong for the inverse shift. It must be re-traced from the stored reconnection direction and seed (course notes footnote 9, p. 33; original code `traceTemporalUpdate`, `PathTracer.slang` L1730–1835). Otherwise you get lag/bias, which ReSTIR GI handles with periodic sample validation (§4.3, every 6 frames).
4. **Confidence (M) capping is mandatory.** Without it, reuse converges to a wrong image (GRIS Fig. 2 and Fig. 10b, §6.4). Defaults across sources:

   | Source | Cap |
   |---|---|
   | GRIS / ReSTIR PT, Enhanced | 20 |
   | ReSTIR DI | 20× the current M |
   | ReSTIR GI | 30 temporal, 500 spatial |
   | RTXDI PT (non-DLSS-RR) | 8 |
   | Course notes | 5–30, start at 20 |

   Enhanced adds a duplication-map adaptive cap (ENH §5), which it states is biased. Confidence resets are unbiased only if they depend on G-buffer changes, never on sample values (course notes §4.4, p. 20).
5. **Disocclusion** is the main quality failure under camera motion. The toolbox:
   - Reset confidence and boost spatial reuse (ReSTIR GI §5; RTXDI `numDisocclusionBoostSamples=8`).
   - Search a 3×3 neighbourhood around the reprojected pixel, plus a zero-motion fallback (RTXDI PT `TemporalResampling.hlsli` L118–159).
   - Dual motion vectors (Zeng et al. 2021, adopted by ENH §6.4).
   - Newer work: backup samples and forward splatting (Reservoir Splatting 2025), multi-layer splatting (2026), stochastic pairwise MIS (2026).
6. **Display.**
   - Every source pairs ReSTIR with a denoiser: SVGF/A-SVGF, NRD ReLAX (designed for RTXDI signals), or DLSS-RR.
   - SVGF / A-SVGF are practical in WebGPU compute.
   - OIDN via `oidn-web` (TF.js/WebGPU) suits still frames.
   - For comparisons against Cycles, run a **progressive accumulation mode** with temporal reuse *off* (spatial-only, independent seeds each frame) and all biased heuristics off. GRIS §6.5 and ENH §7.4 both say this converges faster than accumulating with temporal reuse.
   - To validate *temporal* unbiasedness under motion, average one frame of a deterministic animation over N independent runs. ENH Fig. 14–15 used 1024 runs.

---

## 1. Math you need for temporal reuse (GRIS core, restated)

### 1.1 Unbiased contribution weights (UCW) and the RIS/GRIS estimator

- A real random variable \(W_X\) is a UCW for sample \(X\) if \(\mathbb E[f(X)W_X]=\int_{\mathrm{supp}X} f\) for all integrable \(f\). This is equivalent to \(\mathbb E[W_X\mid X]=1/p_X(X)\) (GRIS App. A, Thm A.1, Eq. 55–56, p. 75:22).
- **GRIS resampling weight** (GRIS Eq. 19, p. 75:7; ENH Eq. 1, p. 13:3). Inputs are \(X_i\in\Omega_i\) with UCW \(W_{X_i}\) and shift \(T_i:\Omega_i\to\Omega\):
  \[w_i = m_i(T_i(X_i))\;\hat p(T_i(X_i))\;W_{X_i}\;\Big|\tfrac{\partial T_i}{\partial X_i}\Big|\quad (X_i\in\mathcal D(T_i)),\qquad w_i=0\ \text{otherwise.}\]
- Select \(Y=T_s(X_s)\) with probability \(w_s/\sum_j w_j\). The output UCW (GRIS Eq. 22) is
  \[W_Y=\frac{1}{\hat p(Y)}\sum_{j=1}^M w_j .\]
- **Resampling-MIS constraint** (GRIS Eq. 20). For every \(y\) in supp \(Y\): \(\sum_{i:\,y\in T_i(\mathrm{supp}X_i)} m_i(y)=1\), with \(m_i\ge0\).
- **Contribution-MIS form** (GRIS Eq. 17–18 and Eq. 21; course notes Eq. 7.1–7.2, §7.1.2 p. 38–39). Resampling weights \(m_i\) may then be arbitrary non-negative functions, for example constants. Unbiasedness is carried by contribution MIS weights \(c_i\) satisfying \(\sum_{i:\,y\in T_i(\mathrm{supp}X_i)}c_i(y)=1\):
  \[W_Y=\frac{c_s(Y)}{m_s(Y)}\,\frac{1}{\hat p(Y)}\sum_j w_j .\]
  The GRIS authors note that constant resampling weights forfeit the convergence-to-\(\hat p\) guarantees unless all inputs are canonical, while contribution MIS still removes bias (GRIS §5.6 "Constant resampling weights", p. 75:10).
- **Canonical sample** (GRIS Def. 5.2, p. 75:9; course notes Def. 5.3.1). An input is canonical if it lives in \(\Omega\), uses the identity shift, has \(\hat p_i=\hat p\), and covers supp \(\hat p\). ReSTIR relies on one canonical sample per pass to cover the support (ENH §2.1, `ENH` L105–109).

### 1.2 MIS families (with confidence weights \(c_i\))

- **"\(\hat p\) from i"** (GRIS Eq. 35; course notes Eq. 5.9):
  \[\hat p_{\leftarrow i}(y)=\hat p_i\!\big(T_i^{-1}(y)\big)\Big|\tfrac{\partial T_i^{-1}}{\partial y}\Big|\ \text{if } y\in T_i(\mathrm{supp}X_i),\ \text{else }0.\]
  For the sample that actually came from \(i\) (\(y=T_i(x)\)) this equals \(\hat p_i(x)/|\partial T_i/\partial x|\). This is the "pHatFrom_opt" shortcut in course notes Alg. 5, lines 6–7.
- **Generalized Talbot / balance heuristic with confidences** (GRIS Eq. 36; course notes Eq. 5.11; GRIS §6.2 multiplies each \(\hat p\) term by the reservoir's \(M_r\)):
  \[m_i(y)=\frac{c_i\,\hat p_{\leftarrow i}(y)}{\sum_j c_j\,\hat p_{\leftarrow j}(y)} .\]
  Cost is \(O(M^2)\) shifts.
- **Generalized pairwise MIS.** Canonical set \(R\), \(|R|=1\) in ReSTIR (GRIS Eq. 37 uniform, Eq. 38 defensive). The confidence-weighted defensive form is course notes Eq. 7.8 / Alg. 7, with \(c_{tot}=\sum_j c_j\) and canonical index \(\kappa\):
  \[m_i(y)=\frac{c_i}{c_{tot}}\cdot\frac{(c_{tot}-c_\kappa)\hat p_{\leftarrow i}(y)}{(c_{tot}-c_\kappa)\hat p_{\leftarrow i}(y)+c_\kappa\hat p(y)}\ (i\ne\kappa),\]
  \[m_\kappa(y)=\frac{c_\kappa}{c_{tot}}+\sum_{i\ne\kappa}\frac{c_i}{c_{tot}}\cdot\frac{c_\kappa\hat p(y)}{(c_{tot}-c_\kappa)\hat p_{\leftarrow i}(y)+c_\kappa\hat p(y)} .\]
  Cost is \(O(M)\): two shifts per neighbour.
- **Temporal reuse with one temporal neighbour** [INFERENCE, direct algebra]. Here \(M=2\), \(|R|=1\), so *non-defensive* pairwise equals generalized Talbot. The original ReSTIR PT code uses Talbot for temporal and defensive pairwise for spatial (GRIS §8.3 "Resampling MIS", p. 75:15; `ReSTIRPTPass.h` L105–106: `spatialMisKind = Pairwise`, `temporalMisKind = Talbot`).
- **ReSTIR DI "unbiased combine"** (Bitterli 2020, Alg. 6, §4.3–4.4). \(W=\frac{1}{\hat p_q(y)}\cdot\frac{1}{Z}\cdot w_{sum}\) with \(Z=\sum_{i:\hat p_{q_i}(y)>0}M_i\). The MIS version uses \(m(x_z)=p_z(x_z)/\sum_i p_i(x_z)\) (Eq. 22). Any proxy that is zero wherever the true density is zero may stand in for the unknown density; with visibility reuse the proxy must also be zeroed where the sample is occluded at \(q_i\) (§4.4).

### 1.3 Confidence weights and reservoirs

- A reservoir is \((X, W_X, c)\). The confidence \(c\) (historically \(M\)) is the relative MIS weight of the reservoir's domain. The sum of inputs' \(c\) is an upper bound on effective sample count, and the confidences must be capped (course notes §4.4, p. 19–20, Alg. 3; GRIS §6.2, p. 75:11; ENH §2.2, `ENH` L111–119).
- **Merge rule** (GRIS §6.2). Reservoir merging is weighted GRIS with the \(\hat p\) terms multiplied by each reservoir's \(M_r\). The output confidence is \(M_{r_m}=\min(M_c,\sum_j M_{r_j})\).
- Variants in the wild:

  | Source | Cap rule |
  |---|---|
  | ENH §2.2 (`ENH` L127–130) | Cap the *temporal input*: output \(=\min(c_{Cap},c_{temp})+1\) |
  | ReSTIR DI §5 | Cap the previous frame's M at 20× the current reservoir's M |
  | RTXDI PT (`TemporalResampling.hlsli` L217) | `prevSample.M = min(prevSample.M, min(maxM, reducedMaxTemporalHistory))`, then add the current M |
  | Reservoir Splatting Eq. 17 | \(c_q=\min(c_q+\sum_{q'}b_{q'}c_{q'},c_{cap})\), where \(b\) = bilinear weights of the back-projected pixel |

### 1.4 ReSTIR as chained GRIS (GRIS §6.3, p. 75:11)

Per frame \(t\):
1. Initial candidates \(X_i^t\) with UCW.
2. Temporal GRIS between the previous frame's final sample \(Y_i^{t-1}\) and \(X_i^t\). Correspondence comes from motion vectors.
3. One or more spatial GRIS passes.
4. Shade with \(f_i(Y_i^t)W_{Y_i^t}\).

The reservoir forwarded to the next frame is the post-spatial one, with its spatial confidence. ReSTIR DI Alg. 5 does the same. ReSTIR GI differs: its spatial pass reads only temporal reservoirs, to avoid compounding bias (ReSTIR GI §4.3, Fig. 10).

A useful interpretation from GRIS §6.4 (boxed, p. 75:12): in a still scene, ReSTIR is an unbiased non-Markovian chain whose state is one path per pixel, and averaging frames converges.

### 1.5 Shift mappings and Jacobians that appear in temporal reuse

- **Definition** (course notes §5.1.1, Def. 5.1.1, p. 22–23). A shift is a deterministic bijection from a subset \(\mathcal D(T)\) of the source domain to a subset of the target domain. The inverse must map back, and undefined shifts must be undefined symmetrically. Failing invertibility is "a common source of bias" (Reservoir Splatting, footnote 1).
- **Reconnection Jacobian in solid angle** (GRIS Eq. 52, p. 75:15; ReSTIR GI Eq. 11):
  \[\Big|\tfrac{\partial\omega^y_i}{\partial\omega^x_i}\Big|=\frac{\cos\theta^y_2}{\cos\theta^x_2}\cdot\frac{\|x_{i+1}-x_i\|^2}{\|x_{i+1}-y_i\|^2},\]
  where \(\theta_2^\bullet\) is the angle at the shared vertex \(x_{i+1}\) between its normal and the connecting direction. RTXDI implements it as `RTXDI_CalculateJacobian` (`Utils/Math.hlsli` L153–168), caching the base path's half as `partialJacobian`.
- **Random replay in PSS.** The Jacobian is 1 (GRIS §8.1, p. 75:15). Solid-angle ↔ PSS conversion is GRIS Eq. 53–54.
- **Hybrid shift, PSS Jacobian** (ENH Eq. 2, p. 13:4, transcribed from the page image):
  \[\Big|\tfrac{\partial T}{\partial\bar u}\Big|=\frac{p^y_{k-1}(\omega'_{k-1})\,G(y_{k-1}\to x_k)\,p^y_k(\omega_k)}{p^x_{k-1}(\omega_{k-1})\,G(x_{k-1}\to x_k)\,p^x_k(\omega_k)} .\]
  Here \(p_k^x(\omega_k)=p(\omega_k\mid x_k,-\omega_{k-1})\), \(p_k(\cdot)\to1\) for \(k=d\), and \(G(x\to y)=\cos\theta/\|x-y\|^2\) at the receiver. The Jacobian is 1 without reconnection. The \(p_k\) ratio exists because the suffix's first direction \(\omega_k\) is preserved in world space, not its random numbers.
- **Reconnection criteria.**
  - Original (course notes Eq. 6.6–6.11): distance \(\ge d_{min}\) and lobe roughness \(\ge\alpha_{min}\) on \(x_{k-1},x_k\) and on \(y_{k-1}\). Invertibility also requires that no earlier vertex of the offset path would have qualified.
  - Enhanced (ENH Eq. 5, §4, `ENH` L331–342): a dual footprint test with \(\kappa=0.02\), plus a single-vertex roughness threshold at \(x_{k-1}\) (ENH §4.2, \(\alpha=0.2\) in §7.2).
  - RTXDI PT defaults (`Source/ReSTIRPT.cpp`): footprint mode, `minConnectionFootprint=0.02`, `minPdfRoughness=0.1`, `roughnessThreshold=0.1`.
- **Rigid motion of an attached vertex.** Area Jacobian is 1. For non-rigid deformation it is the ratio of the triangle's current and previous area (Reservoir Splatting §4.1.3, after Eq. 16).

---

## 2. What each source contributes (implementer's digest)

### 2.1 ReSTIR DI (Bitterli, Wyman, Pharr, Shirley, Lefohn, Jarosz 2020)

- **Pipeline** (Alg. 5, §3.2): initial RIS with M candidates, then a visibility test of the chosen sample with its reservoir zeroed if occluded ("visibility reuse"), then temporal combine with the back-projected previous pixel, then n spatial passes of k neighbours, then shading.
- **Temporal reuse** (§3.2 "Temporal Reuse"; §5 "Neighbor selection"). Motion vectors project the current pixel into the previous frame, and the reservoir there is reused. The history then implicitly includes all previous frames.
- **Biased versus unbiased.**
  - The naive combine with 1/M weights is biased because neighbouring pixels have different targets, and it darkens near geometric discontinuities (§3, §4.2).
  - The unbiased variant (Alg. 6) counts only neighbours whose \(\hat p\) at the selected sample is positive. With visibility reuse that check needs an extra shadow ray per neighbour (§4.4).
  - The biased variant adds heuristic neighbour rejection: 10% depth, 25° normal (§5).
- **Parameters** (§5):
  - Initial candidates: M = 32, power-based triangle sampling; 25% environment candidates if an environment map is present.
  - Spatial reuse: k = 5 (k = 3 unbiased), 30 px radius, low-discrepancy neighbour sampling.
  - Spatial passes: n = 2 (n = 1 unbiased).
  - N = 1 shaded samples (unbiased) and N = 4 (biased).
  - Previous-frame M clamped to ≤ 20× the current M. This bounds temporal influence and prevents unbounded growth (§5 "Reservoir storage and temporal weighting").
- **Limitations** (§8.1). Quality degrades near disocclusions, lighting discontinuities, high geometric complexity, and fast-moving lights.

### 2.2 ReSTIR GI (Ouyang, Liu, Kettunen, Pharr, Pantaleoni, HPG 2021)

- **Sample representation** (§4, Fig. 3): visible point and normal, sample point and normal, outgoing radiance \(L_o\) at the sample point (RGB), and random numbers.
- **Target function** (Eq. 9): \(\hat p=L_o(x_s,-\omega_i)f(\omega_o,\omega_i)\langle\cos\theta_i\rangle\). A simpler Eq. 10, \(\hat p=L_o\), also works well for spatial reuse.
- **Three buffers:** initial, temporal, and spatial reservoirs. Double-buffered, which adds one frame of indirect lag (§5).
- **Temporal resampling** (Alg. 3) reads the reprojected temporal reservoir and updates it with the new sample. **If reprojection fails, both temporal and spatial reservoirs are reset** (§5).
- **Spatial resampling** (Alg. 4):
  - Uses the solid-angle Jacobian (Eq. 11).
  - Similarity test: normals within 25°, normalized depths within 0.05.
  - Biased mode skips the visibility ray. Unbiased mode traces it and counts only neighbours that could have produced the sample (lines 16–20).
  - Adaptive radius starts at 10% of resolution. `maxIterations` is 9 when the spatial M is below half the maximum, else 3.
- **Dynamic-scene bias** (§4.3).
  - Temporal reuse is biased if the stored \(L_o\) becomes stale. ReSTIR tends to hold bright samples for many frames, which causes lag.
  - Mitigation, inspired by A-SVGF: every few frames (default 6) re-trace all reservoir samples with the *same random numbers*, compare radiance within a tolerance, and clear the reservoir on mismatch.
  - Occluders that newly block the visible-point → sample-point segment are handled by a shadow ray during validation (Fig. 11).
- **Clamps:** M ≤ 30 temporal and ≤ 500 spatial (§5).
- **Known bias:** Lambertian assumption at the sample point.
- **Multi-bounce:** 25% of pixels, via tile-level Russian roulette.

### 2.3 GRIS / ReSTIR PT (Lin, Kettunen, Bitterli, Pantaleoni, Yuksel, Wyman 2022)

- **Convergence requirements** (§1, p. 75:2): correct MIS, bounded \(f/\hat p\), resampling-weight variance tending to 0, enough canonical samples, and a reasonable M-cap for temporal reuse.
- **M-cap** (§6.4, Fig. 9–10, p. 75:16–17).
  - Without a cap, relative weights of new samples decay exponentially and the result converges to a wrong image.
  - With a cap, the temporal sample's relative weight is at most \(M_c/(M_c+1)\), akin to an exponential moving average.
  - In Fig. 9a, large caps reduce error at first and then raise it, while cap 20 stays low. Measured in a static scene.
- **Real-time settings** (§8.3, p. 75:15): one path tree per pixel; one spatial pass of 3 neighbours within 20 px; \(M_c=20\); defensive pairwise for spatial and generalized Talbot for temporal; \(|R|=1\).
- **Offline mode** (§6.5): temporal reuse off. Each iteration runs spatial GRIS (32 initial path trees, 3 spatial iterations of 6 neighbours within 10 px, low-discrepancy) and the independent frames are averaged. This converges faster than accumulating with temporal reuse (Fig. 9b).
- **Correctness notes (App. B, p. 75:23).**
  - *Visibility:* ReSTIR PT always includes visibility in \(\hat p\). Dropping it makes samples non-canonical and breaks the convergence argument.
  - *Temporal reuse:* unbiased only with proper MIS and a temporal shift mapping, because the MIS weights must bijectively shift paths between the prior and current frames. Fully unbiased temporal reuse must evaluate paths in both frames, which is tricky in dynamic environments. Biased approximations such as ReSTIR GI's are acceptable if their bias is imperceptible.
- **Reservoir** (§8.2): 88 bytes, holding reconnection vertex, seed, W, and M.

### 2.4 Course notes: "A Gentle Introduction to ReSTIR" (Wyman et al., SIGGRAPH 2023)

Source: `https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf`, updated 4 Mar 2024. Implementation-relevant items, with page numbers from the notes:

- **Spatiotemporal reuse (§4.2, p. 16–17).**
  - Tip 4.1: never choose spatial neighbours based on the samples stored in them; that biases.
  - Temporal reuse needs advanced MIS and access to previous frames' target functions to remove all bias.
  - Natural order per frame: initial → temporal → spatial → shade.
- **Temporal reuse for DI (§4.3, p. 19).**
  - Needs careful motion-vector tracking.
  - Proper MIS needs last frame's \(\hat p\), possibly including visibility against the *previous* scene, which means storing the previous acceleration structure.
  - Tip 4.4 gives the validation order: initial candidates first, then spatial, then temporal *without motion*. "Validated" means the average of many still frames converges to path-traced ground truth.
- **Confidence (§4.4, p. 19–20).**
  - Summing confidences is an upper bound that overestimates exponentially, so cap (typically 5–30; start at 20).
  - Pixels entering the screen get \(c\) reset to 0. Resetting on occlusion/disocclusion is often sensible.
  - Resets are allowed **only if based on G-buffer (changes)**. Resets that depend on sample details bias.
- **Domains change when objects move (§4.5, p. 21, footnote 8).** One can move a path to the next frame by attaching its vertices to moving objects: keep triangle index and UV in both frames. This is a shift mapping between frames.
- **Tip 5.1 (p. 22).** Implement the reconnection shift first. Validate that the shift to the same pixel is the identity with Jacobian 1.
- **Balance heuristic across domains (§5.3, Eq. 5.8–5.11, Alg. 5, p. 24–26).** Correctness first, performance later (Tip 5.2).
- **ReSTIR PT reservoir (§6.6, Alg. 6, footnote 9, p. 33–34).**
  - Stored: reconnection vertex as triangle id + barycentrics, two RNG seeds, incident direction and radiance at the reconnection vertex, and lobe indices.
  - Footnote 9: if the scene changes, (triangle id, barycentrics) automatically follows the animated triangle. But a temporal shift is needed to update the cached radiance. One option re-uses the world-space direction \(\omega_k\) to find \(y_{k+1}\) and replays the second RNG seed for the rest of the path.
- **Neighbour rejection and contribution MIS (§7.1.1–7.1.2, p. 37–39).**
  - Neighbour rejection behaves like approximate MIS, similar to Veach's cutoff heuristic.
  - Heuristics are unbiased only if they do not look at individual samples or weights.
  - Contribution MIS gives O(M) unbiasedness, but excess noise if domains differ and no convergence of the distribution.
- **Pairwise MIS (§7.1.3, Eq. 7.3–7.8, Alg. 7, p. 39–41).**
- **Biased MIS weights (§7.1.4, p. 41–42).**
  - The previous-frame evaluation of the current sample is the expensive term: it needs the prior BVH.
  - Informed shortcuts: use the current BVH as a stand-in, assume the term is zero, or recompute with last frame's data but assume unchanged visibility.
  - Bias sign rule: overestimate darkens, underestimate brightens. Assuming zero brightens (overcounting).
- **Accelerating the hybrid shift (§7.2.3, p. 45–46).**
  - Split into a replay kernel (path tracing only) and a reconnection kernel (BSDF re-evaluation and visibility).
  - Stream-compact replay work.
  - Each gave roughly 40% shader-time reduction in Veach Ajar.
- **Advice (§9, p. 48–49):**
  - Start with a ground-truth path tracer in the same code.
  - Start with basic RIS.
  - Spatial reuse is easier to debug; add temporal next.
  - Don't adopt RTXDI's many options (checkerboarding, permutation, boiling suppression) early; many are not meant to be unbiased.
  - A reservoir is a distribution at a point, not over a voxel.
  - Reuse visibility very carefully: always include visibility in targets and MIS until validated.

### 2.5 Later work that ReSTIR PT Enhanced builds on or cites

| Work | What it does | Relevance to us |
|---|---|---|
| **Conditional RIS & ReSTIR** (Kettunen, Lin, Ramamoorthi, Bashford-Rogers, Wyman, SIGGRAPH Asia 2023) | GRIS in conditional path spaces; reuses path *suffixes* while keeping prefixes independent; a final-gather ReSTIR PT prototype reduces blotchy correlation. Author page says it is more expensive than baseline ReSTIR PT. Code: `github.com/NVlabs/conditional-restir-prototype`. ENH §5 cites it as a partial, expensive fix for correlation. | Optional later; not needed for correctness |
| **Decorrelating ReSTIR via MCMC mutations** (Sawhney et al., TOG 2024) | Interleaves MCMC mutations with resampling to counter correlation/impoverishment; unbiased; one mutation per reservoir sample per frame helps. ENH §5: expensive and only reduces some correlation types. | Optional |
| **Area ReSTIR** (Zhang, Lin, Kettunen, Yuksel, Wyman, TOG/SIGGRAPH 2024) | Reservoirs integrate the 4D pixel/lens ray space; subpixel-tracking temporal reuse (1×1 *fractional* reservoir around the back-projected pixel-centre hit, filled from the 2×2 prior pixels; "fast" vs "robust" variants), per Reservoir Splatting §3.5. ENH §8 suggests merging with Area ReSTIR in future. | Antialiasing/DoF; later |
| **Reservoir Splatting** (Liu, Lin, Kettunen, Wyman, Ramamoorthi, SIGGRAPH 2025) | Forward-reprojects (splats) prior-frame paths so object-space primary hits are exactly preserved; GRIS with scatter; backup back-projected samples; motion blur; DoF. Code: `github.com/Jebbly/Reservoir-Splatting`. Details in §4.10. | Strong candidate for camera-motion robustness |
| **Multi-Layer Reservoir Splatting** (Hong, Zhang, Lin, Kettunen, Wyman, Yuksel, SIGGRAPH 2026) | Multiple screen-space layers; splatting shifts samples between layers so previously occluded samples can be reused after disocclusion; depth ranges cut ray queries; tracks only active domains (Utah project page; I did not read the PDF). | Disocclusion; advanced |
| **Stochastic Pairwise MIS** (Hedstrom, Kettunen, Lin, Wyman, Li, CGF/EG 2026) | Unbiased reuse from many spatial neighbours in real time, focusing on pixels with contributing samples; targets revealed regions during motion (abstract). | Disocclusion/spatial; later |
| **Compatibility-Guided Neighbor Selection** (Junkins, Kettunen, Lin, Ramamoorthi, Wyman, HPG 2026 Best Paper) | Chooses spatial neighbours by pixel compatibility: 6–29% lower SMAPE and 22–49% lower temporal covariance (abstract). Code: `github.com/orion-junkins/ReSTIR-CGNS`. Unbiasedness not stated on the page [UNVERIFIED]; per course notes Tip 4.1 it is safe only if selection ignores reservoir samples. | Later |
| **Enhancing Spatiotemporal Resampling with a Novel MIS Weight** (Pan et al., CGF/EG 2024) | New MIS weight for mixing domains that keeps convergence as the non-canonical share grows; applied to temporal resampling to reduce noise from scene changes or jitter (abstract only). | Possible temporal MIS alternative [UNVERIFIED details] |
| **ReSTIR PT Enhanced** (Lin, Kettunen, Wyman, I3D/PACM CGIT 2026 Best Paper) | Paired spatial reuse, footprint reconnection criteria, duplication-map adaptive \(c_{Cap}\) (biased), unified DI+GI reservoirs, low-level optimizations, vector (RGB) resampling weights for colour noise, dual motion vectors for disocclusion. | Our base algorithm |
| Other 2025–2026 NVIDIA RTR work (listed on the lab page, not read) | ReSTIR PG (path guiding), ReSTIR BDPT (caustics), Gradient-Domain ReSTIR PT, Real-Time LOD with ReSTIR, ReSTIR-sampled shadow maps. | Out of scope for the temporal design |

### 2.6 Production and reference code (what implementations actually do)

**NVIDIA RTXDI-Library** (`github.com/NVIDIA-RTX/RTXDI-Library`, 2025–2026) now ships a **ReSTIR PT** implementation (`Include/Rtxdi/PT/*`) with duplication map, decorrelation, footprint reconnection, and boiling filter. Its licence is proprietary (`LicenseRef-NvidiaProprietary`). **Read it for understanding only; do not copy code** [INFERENCE: licensing caution]. Observations:

- **Temporal pass** (`PT/TemporalResampling.hlsli`):
  - Neighbour search: 9 taps in a fixed 3×3 pattern around the reprojected pixel, plus an optional *fallback* tap at zero motion with permutation sampling for disocclusions (L49–93, L118–159).
  - Validity: normal-dot, relative depth, and material similarity (L95–116).
  - Duplication-based history reduction (L405–415, detail in §4.7).
  - Age-based rejection: reject the temporal neighbour if the sample age exceeds `maxReservoirAge` (L211–227).
  - Forward shift = hybrid shift with `isBasePathInPrevFrame=true` (L248–257, L290–325).
  - "BiasCorrection" inverse-shifts the *selected* sample into the previous pixel with `isPrevFrame=true` and visibility. It forms \(\pi\) and \(\pi_{sum}=\hat p\,c_{cur}+\hat p_{temporal}\,c_{prev}\) and finalizes \(W\) with \(\pi/(\pi_{sum}\hat p)\) (L327–360, L429–438). This is the **contribution-MIS form** from §1.1.
  - If no temporal neighbour was found, it sets a "boost spatial samples" flag for the spatial pass (L439–443).
- **Light remapping inside the shift** (`PT/HybridShift.hlsli`):
  - NEE-sampled light vertices are stored as (light index, uv), flagged by `radiance.x = INF` (L113–118; `Reservoir.hlsli` `RTXDI_GetSampledLightData`).
  - When the base path is from the previous frame, the light index is translated prev→cur. An invalid mapping (light removed) zeroes the target (L230–262).
  - Light info is loaded with the `isPrevFrame` flag, and the light surface point is rebuilt from uv (L265–287): the **light-local representation**.
  - Mid-path reconnection vertices are stored as **world-space position** (`translatedWorldPosition`) plus normal and **cached radiance** (`Reservoir.hlsli` struct). Consequences:
    1. Mid-path reconnection vertices do **not** follow moving geometry in RTXDI PT.
    2. Suffix radiance is reused as-is across frames [INFERENCE from the struct and absence of a suffix re-trace in the temporal pass].
- **Visibility during temporal bias correction** in PT uses `RAB_GetConservativeVisibility` from the previous surface against the *current* BVH (`EvaluateReconnection`, L234–246, called with `prevSurface`).
  - RTXDI's DI temporal pass instead calls `RAB_GetTemporalConservativeVisibility`. Its header says that ray should ideally use the previous-frame BVH, and falling back to the current BVH gives more bias (`DI/TemporalResampling.hlsli` L20–26, L192–200).
  - The sample app implements it with `PrevSceneBVH` when `enablePrevTLAS`, else the current BVH, and documents the fallback as transient bias (`Samples/FullSample/.../RAB_VisibilityTest.hlsli`).
- **Light buffer layout.**
  - Two light groups (odd/even frames) are kept so temporal resampling can read the previous frame's lights for correct normalization, and a light-index mapping buffer is provided (`Doc/Integration.md` L48–78).
  - Mapping construction: for each light that existed last frame, write prev→cur and cur→prev indices +1, with 0 meaning invalid (`Samples/FullSample/Shaders/PrepareLights.hlsl` L185–204).
  - `RAB_TranslateLightIndex` returns −1 for lights missing in the other frame (`RAB_Buffers.hlsli` L123–135).
- **DI temporal pass** (`DI/TemporalResampling.hlsli`, whole file):
  - History limit is `maxHistoryLength * curSample.M` (L42).
  - It translates the prev light index (killing the reservoir if the light is gone, L125–148).
  - It evaluates the previous sample at the current surface with current light info (L152–162).
  - For bias correction it loads the **previous-frame light info** for the selected sample (translated cur→prev) and evaluates it at the **previous-frame surface** with the same uv. Optionally it adds temporal visibility (L173–206).
- **ReSTIR PT defaults** (`Source/ReSTIRPT.cpp`):

  | Parameter | Default (non-DLSS-RR) | DLSS-RR |
  |---|---|---|
  | `numInitialSamples` | 1 | |
  | `maxBounceDepth` | 3 | |
  | `maxRcVertexLength` | 5 | |
  | temporal `depthThreshold` / `normalThreshold` | 0.1 / 0.6 | |
  | `maxHistoryLength` | 8 | 20 |
  | `maxReservoirAge` | 30 | |
  | age-based rejection | on | off |
  | duplication-based reduction | off | on, strength 0.5 |
  | `enableFallbackSampling` | true | |
  | spatial samples | 1 | |
  | spatial radius | 32 | |
  | `numDisocclusionBoostSamples` | 8 | |
  | boiling filter | on, strength 0.2 | off (uses firefly replacement) |
  | decorrelation | none | Stagnancy, factor 0.4 |

**Original ReSTIR PT code** (`github.com/DQLin/ReSTIR_PT`, Falcor, 2022):
- **Temporal pass** (`Source/RenderPasses/ReSTIRPTPass/TemporalReuse.cs.slang` L113–299):
  - Previous pixel = pixel + motionVector·frameDim + a random [0,1) offset (stochastic rounding, L119–127).
  - It reads the previous V-buffer `temporalVbuffer` and builds previous shading data with the *previous* camera ray (`computeRayPinholePrevFrame`, L67–78).
  - History is capped at `gTemporalHistoryLength × currentM` (L139).
  - Talbot MIS (L146–226):
    - For the temporal sample, the self term is `temporalReservoir.F / dstJacobian * temporalReservoir.M`, i.e. the *stored previous-frame* integrand divided by the shift Jacobian (the "pHatFrom_opt" shortcut).
    - For the canonical sample, the temporal term comes from `computeShiftedIntegrand(..., usePrev=true)`, which uses the previous camera position (`Shift.slang` L101, L126).
  - `kTemporalUpdateForDynamicScene` (default false, `ReSTIRPTPass.h` L109) enables `traceTemporalUpdate`. That function re-samples the NEE light with the stored reconnection seed, or re-traces the suffix from the reconnection vertex along the stored `rcVertexWi` with the stored seed, refreshing `rcVertexIrradiance`, `lightPdf`, and light type (`PathTracer.slang` L1730–1835; `Shift.slang` L69–72, L180–187).
  - The reconnection vertex is a Falcor `HitInfo` (instance + primitive + barycentrics) reloaded with current scene data (`loadShadingDataWithPrevVertexPosition`), so it follows animated triangles.
  - [INFERENCE] Apart from the camera, the previous-frame evaluation uses current scene data. That is exact for camera-only motion in a static scene and approximate otherwise.
- **Defaults** (`ReSTIRPTPass.h` L105–139): spatial Pairwise, temporal Talbot, 3 spatial neighbours, radius 20, temporal history 20, reprojection enabled.

**ENH supplemental** (`lin2026restirptenhanced_supplemental.pdf`, §8, Alg. 1). The 64-byte reservoir holds:
- `W`, `F` (RGB integrand; \(\hat p=\)lum(F))
- `initRandomSeed`, `rcVertexRandomSeed`
- `pathFlags`, with M packed as 8 bits
- `rcVertexInstanceID`, `rcVertexPrimitiveIndex`
- `rcVertexBarycentrics` (2×unorm16)
- `rcVertexWi` (octahedral 2×unorm16)
- `rcVertexRadiance` (float3)
- `rcVertexCachedValues` (float2: product of the base path's Jacobian terms, and the NEE light pdf)

So Enhanced keeps the **object-attached** reconnection vertex (instance + primitive + barycentrics). The supplemental does not discuss dynamic lights.

---

## 3. Dynamic scenes: precise formulation for our renderer

### 3.1 Setup and notation

- Frame \(t\), current pixel \(q\), matched previous pixel \(q'\).
- Scene state \(S_t=(C_t,G_t,L_t,P_t)\): camera, geometry and instance transforms, lights (transforms, intensities, existence), and light-selection distribution.
- Path space of pixel \(q\) at frame \(t\): \(\Omega_{t,q}\). Paths are \(\bar x=[x_0,x_1,\dots,x_d]\), with \(x_0\) the camera and \(x_1\) the primary hit through \(q\) (jittered or centre).
- Contribution \(f_t(\bar x)\) (RGB). It includes the NEE/BSDF technique MIS weight \(\omega_t\), which depends on \(P_t\) and \(L_t\) (ENH §2.3 "Jacobian"; course notes Eq. 6.13–6.14). Work in PSS: \(F(\bar u)=\omega_t f/p_t\).
- Target \(\hat p_{t,q}=\mathrm{lum}(f_t)\) restricted to \(\Omega_{t,q}\), including visibility.
- Temporal shift \(T:\Omega_{t-1,q'}\to\Omega_{t,q}\) (partial bijection).

**Temporal GRIS with generalized Talbot and confidences** [INFERENCE: direct instantiation of GRIS Eq. 19/22/36 with confidence weights]:

- Inputs:
  - canonical \(X_c\in\Omega_{t,q}\), with \(W_c\) and \(c_c\) (usually 1), identity shift;
  - temporal \(X_p\in\Omega_{t-1,q'}\), with \(W_p\) and \(c_p=\min(c(r_{t-1,q'}),c_{cap})\), shifted to \(Y_p=T(X_p)\).
- Proxy densities:
  \[\hat p_{\leftarrow c}(y)=\hat p_{t,q}(y),\qquad \hat p_{\leftarrow p}(y)=\hat p_{t-1,q'}\big(T^{-1}(y)\big)\,\big|\partial T^{-1}/\partial y\big| .\]
- MIS weights:
  \[m_c(y)=\frac{c_c\hat p_{t,q}(y)}{c_c\hat p_{t,q}(y)+c_p\hat p_{\leftarrow p}(y)},\qquad m_p(y)=\frac{c_p\hat p_{\leftarrow p}(y)}{c_c\hat p_{t,q}(y)+c_p\hat p_{\leftarrow p}(y)} .\]
- Resampling weights and output:
  \[w_c=m_c(X_c)\,\hat p_{t,q}(X_c)\,W_c,\qquad w_p=m_p(Y_p)\,\hat p_{t,q}(Y_p)\,W_p\,|\partial T/\partial X_p|,\qquad W_Y=\frac{w_c+w_p}{\hat p_{t,q}(Y)},\quad c=c_c+c_p .\]
- Where each piece comes from:
  - \(m_p(Y_p)\) needs \(\hat p_{\leftarrow p}(Y_p)=\hat p_{t-1,q'}(X_p)/|\partial T/\partial X_p|\). The numerator is **stored** in the previous reservoir as \(F\), computed in frame \(t-1\) with frame-\(t-1\) data, so it is exact.
  - \(m_c(X_c)\) needs \(\hat p_{\leftarrow p}(X_c)\). This is the **inverse shift of the fresh canonical path into frame \(t-1\)**, evaluated under \(S_{t-1}\). It is the only place previous-frame state is required.

**Equivalent contribution-MIS form** (RTXDI style; verified against GRIS Eq. 21) [INFERENCE]:
\[\tilde w_c=c_c\hat p_{t,q}(X_c)W_c,\qquad \tilde w_p=c_p\hat p_{t,q}(Y_p)W_p|\partial T/\partial X_p| .\]
Select \(Y=Y_s\) with probability \(\propto\tilde w\), then
\[W_Y=\frac{\pi_s(Y)}{c_c\hat p_{t,q}(Y)+c_p\tilde p_{\leftarrow p}(Y)}\cdot\frac{\tilde w_c+\tilde w_p}{\hat p_{t,q}(Y)},\qquad \pi_c=\hat p_{t,q},\ \pi_p=\tilde p_{\leftarrow p}.\]
Only **one** inverse shift is needed, of the *selected* sample. When \(s=p\), \(T^{-1}(Y)=X_p\) and the value is recomputed rather than read from storage, which keeps it consistent. Cost is the same as Talbot: one forward shift plus one inverse shift per pixel.

### 3.2 The temporal shift \(T\): what is copied, what is recomputed

Adapted from the hybrid shift. Components:

1. **Camera and primary hit.** \(y_0=\) current camera; \(y_1=\) current pixel's primary hit (current V-buffer). With the standard back-projection gather, the primary hit is *not* preserved between frames. It is re-derived from the current pixel. The prefix is replayed from \(y_1\) with the stored seed (ENH §2.3; `TemporalReuse.cs.slang`).
   - Area ReSTIR approximately preserves the primary hit (fractional reservoir). Reservoir Splatting preserves it exactly (§3.10).
2. **Prefix (random replay).** Replay the stored seed from \(y_1\) until \(y_{k-1}\). The PSS Jacobian is 1.
   - Russian roulette must not be replayed. ENH §6.2.4 and supplemental §6 remove RR from replay and fold survival probabilities into the *initial* sample's PSS pdf, \(p(\bar u)=\prod_i q_i(\bar u)\). That prevents "survived path shifts to killed path" failures.
3. **Reconnection vertex \(x_k\).** Must be stored in a representation that makes \(T\) a sensible bijection *across frames*:
   - **Object-attached:** instance + primitive + barycentrics (ENH supplemental Alg. 1; course notes footnote 8/9). Follows moving objects. Rigid-motion area Jacobian is 1.
   - **Light-local for analytic area lights hit by BSDF rays** [INFERENCE]: (light id, u, v) with \(x_k=\mathcal T^{light}_t(u,v)\). Follows the moving light. Area Jacobian \(=A_t/A_{t-1}\), which is 1 for rigid motion without rescale.
   - **World-space** (RTXDI PT): still a valid shift, since it is deterministic and invertible. But when the thing it lay on moves, the shifted path usually has zero contribution. Unbiased, just wasteful [INFERENCE]. The inverse shift must use the *same* representation.
4. **NEE light vertex.** Store (light id, uv or PSS random numbers). Translate the id each frame; if it maps to −1 the shift is undefined (RTXDI `HybridShift.hlsli` L230–262). Rebuild the point from uv on the frame-\(t\) light.
   - ENH §6.2.3 *forces* reconnection at NEE light vertices, so replay never re-runs light selection. With per-frame light tiles, re-running RIS light selection would not be reproducible, so forcing reconnection is also a correctness convenience [INFERENCE].
5. **Suffix after \(x_k\).** Reuse the cached incident radiance \(L_i(x_k,\omega_k)\) and re-evaluate only the BSDF at \(x_k\) for the new incoming direction (ENH §2.3 "Implementation"; GRIS §7.4). This is exact only if the suffix's contribution did not change between frames. Otherwise perform the **temporal suffix update**: trace from \(x_k\) along stored \(\omega_k\) (world space, or object-local for moving objects) to \(y_{k+1}\), then replay the second seed (course notes footnote 9; `traceTemporalUpdate`).
   - The \(p^y_k(\omega_k)/p^x_k(\omega_k)\) factor in ENH Eq. 2 already accounts for preserving \(\omega_k\) in world space.

### 3.3 The inverse shift \(T^{-1}\) for MIS: which previous-frame state is required

To evaluate \(\hat p_{t-1,q'}(T^{-1}(Y))\) exactly [INFERENCE: enumerated from the path-contribution definition]:

| Quantity | Needed for | Our static-geometry project |
|---|---|---|
| Previous camera position \(x_0'\) | View direction at the previous primary hit (BSDF eval at \(x_1'\)) | Keep `camPrev` uniform. The original code uses `camera.getPosition(usePrev)`. |
| Previous primary hit \(x_1'\) of pixel \(q'\) | Start of replay | Keep the previous V-buffer (tri id + barycentrics), as the original code does with `temporalVbuffer`. |
| Previous geometry / BVH for replayed vertices and visibility | Prefix hits, reconnection visibility, suffix | **Identical** (static BVH), so no extra memory |
| Previous light buffer (transforms, sizes, intensity, colour, spot cone, existence) | NEE vertex position via (id,uv), emission \(L_e\), BSDF-hit emitter intersection, light pdf in \(\omega_t\) | Keep `lightsPrev[]`, `lightMapCurToPrev[]`, `lightMapPrevToCur[]` |
| Previous light-selection distribution \(P_{t-1}\) (power CDF / alias table) | Technique MIS weight \(\omega_t\) (NEE vs BSDF) | Keep previous CDF, or recompute from `lightsPrev` |
| Previous suffix radiance at \(x_k\) (for the canonical path) | Suffix factor of \(\hat p_{t-1}\) | Needs a suffix re-trace with previous lights if lights changed this frame; otherwise identical |
| Previous reconnection-vertex position (if on moving geometry or light) | Reconnection geometry term and Jacobian | Light-local: \(\mathcal T^{light}_{t-1}(u,v)\) |

Once the light buffer is double-buffered, the inverse shift is the ordinary shift code run with a "frame selector" choosing buffers [INFERENCE]. RTXDI's `isPrevFrame` flag and `RAB_LoadLightInfo(index, prevFrame)` follow this pattern.

### 3.4 Consequences of evaluating the previous-frame target with current-frame data

**Course notes §7.1.4 (p. 41–42):**
- The costly term is the prior-frame density of the current candidate. It needs last frame's data, including the prior BVH.
- Informed approximations include using the current BVH, assuming the term is zero, or using last frame's data without re-testing visibility.
- Overestimating that density darkens, exact adds no bias, and underestimating brightens; errors can go either way in different image regions.
- Assuming zero is not true in general, especially in static scenes, and overcounts.

**GRIS App. B (p. 75:23):** full unbiasedness needs evaluating paths in both frames. Biased approximations, such as ReSTIR GI ignoring visibility, trade imperceptible bias for speed.

**RTXDI comments** (`DI/TemporalResampling.hlsli` L20–26; sample `RAB_VisibilityTest.hlsli`): the previous BVH should ideally be used; the current BVH gives more, transient bias.

**My analysis** [INFERENCE: derived from GRIS Eq. 17/18/20 and Bitterli 2020 §4.4]. Let \(\tilde p(y)\) be whatever you compute for \(\hat p_{\leftarrow p}(y)\).

- **Talbot / resampling-MIS form** (§3.1 first block). \(m_p\) at \(Y_p\) uses the *stored exact* value, while \(m_c\) at \(X_c\) uses \(\tilde p\). The functions \(m_c(\cdot)\) and \(m_p(\cdot)\) then sum to 1 only where \(\tilde p=\hat p_{\leftarrow p}\):
  \[m_c(y)+m_p(y)=\frac{c_c\hat p(y)}{c_c\hat p(y)+c_p\tilde p(y)}+\frac{c_p\hat p_{\leftarrow p}(y)}{c_c\hat p(y)+c_p\hat p_{\leftarrow p}(y)}\]
  This is below 1 when \(\tilde p>\hat p_{\leftarrow p}\) (darkening) and above 1 when \(\tilde p<\hat p_{\leftarrow p}\) (brightening). This matches the course-notes rule. **Magnitude errors matter.** Examples: using current light intensity, current suffix radiance, or the current light pdf for the canonical sample's previous-frame evaluation.
- **Fix without extra previous state:** evaluate \(\tilde p\) **consistently** for both weights. Recompute \(\hat p_{\leftarrow p}(Y_p)\) with the same approximate evaluator instead of reading the stored value, or use the contribution-MIS form. The contribution-MIS constraint (GRIS Eq. 17) then reads
  \[\sum_{i:\,y\in T_i(\mathrm{supp}X_i)}\frac{c_i\tilde p_i(y)}{\sum_j c_j\tilde p_j(y)}=1\]
  with \(\tilde p_c=\hat p_{t,q}\). It holds whenever \(\tilde p_p(y)>0\Rightarrow y\in T(\mathrm{supp}X_p)\) (support containment). Magnitude errors then only change variance.
  - Remaining bias comes from **support violations**: \(\tilde p_p(y)>0\) for paths the previous reservoir could not have produced, which means the canonical weight is below 1 and the result darkens.
  - Pure-zero errors (\(\tilde p_p=0\) where the previous frame *could* produce \(y\)) are harmless for bias: \(c_c=1\) and the temporal sample's contribution MIS is 0. They cost variance.
- **Support in terms of the previous frame's target.** The support that matters is that of the target actually used by frame \(t-1\)'s resampling, since RIS outputs share their target's support (course notes Tip 3.11). If frame \(t-1\) used "conservative" visibility, reuse the *same* visibility function evaluated in frame \(t-1\)'s scene [INFERENCE].
- **Case table for our renderer** (static geometry; contribution-MIS form unless noted) [INFERENCE]:

| Change between \(t-1\) and \(t\) | If \(\tilde p\) uses current data | Exact fix |
|---|---|---|
| Camera only (lights static) | Must still use previous camera and previous V-buffer, otherwise it is not the previous domain. With both, current scene data *is* previous data: **exact**. | Previous camera + previous V-buffer (cheap) |
| Light translated/rotated | NEE vertex rebuilt on the *current* light: \(\tilde p>0\) for light points that were not on the previous light → darkening (support). Visibility to the moved light differs. | Rebuild with `lightsPrev`; static BVH gives exact previous visibility |
| Light intensity/colour change | Magnitude only; unbiased in contribution-MIS form, biased in Talbot-with-stored-values | `lightsPrev` intensities |
| Light added at \(t\) | Paths ending on the new light get \(\tilde p>0\) though the previous frame could not produce them → one-frame darkening of the new light's contribution | cur→prev map returns −1, so \(\tilde p=0\) |
| Light removed at \(t\) | Previous samples on it: forward shift undefined (prev→cur = −1), \(w_p=0\). Canonical never lands on it. **No bias.** | Same |
| Suffix contribution changed (light moved in view of a suffix vertex) | Cached current \(L_i\) used for the previous evaluation: magnitude error (and support error if the suffix was dark at \(t-1\)) → small darkening | Re-trace suffix with `lightsPrev` (extra cost) |
| Moving occluders/receivers (future) | Current BVH says visible where the previous frame was occluded → darkening band at moving shadow edges every frame (support) | Previous BVH/TLAS (RTXDI `PrevSceneBVH`), or accept transient bias |

### 3.5 Lights: representation and lifecycle (WebGPU design) [mostly INFERENCE; mechanisms cited]

- **Light buffer.** `struct Light { type; transform (3×4); size; radiance/intensity; spot params; flags }`.
  - Per frame, write `lightsCur` and keep `lightsPrev` (swap).
  - Maintain stable application IDs and build `curToPrev[i]` and `prevToCur[j]` maps (−1 = none), following RTXDI's mapping buffer (`PrepareLights.hlsl` L185–204).
  - Also keep previous and current light-selection CDFs or alias tables (power-based, as ReSTIR DI §5 / RTXDI PDF texture).
- **Sample encoding.**
  - Area light: (lightId, u, v). Pack \((u,v)\) as 2×unorm16 in the barycentric slot of the 64-byte reservoir, with a flag in `instanceID` meaning "analytic light".
  - Point and spot lights (delta): lightId only.
  - Environment: direction (octahedral) or env texel uv, as in RTXDI's IS env light uv convention (`RAB_LightInfo.hlsli` comment).
- **Delta lights.** They cannot be hit by BSDF rays, so they are NEE-only and the technique MIS weight is 1. Shifting the NEE vertex copies a fixed point. In PSS the Jacobian is 1, since light selection and the (degenerate) position come from replayed or stored numbers [INFERENCE; worth a unit test]. RTXDI computes the NEE-light reconnection Jacobian as the ratio of solid-angle light pdfs (`HybridShift.hlsli` L305–313). For a point light the "solid-angle pdf" convention must be chosen consistently; validate against Cycles [UNVERIFIED].
- **Area lights hit by BSDF rays.** Intersect analytic light shapes separately from the BVH. Emission is MIS-weighted with the NEE pdf from the *frame's* light tables.
  - Whether a BSDF path should *terminate* at an area light or pass through it (Cycles lights are not ordinary geometry) must match the Blender reference [UNVERIFIED; flag to the validation workstream].
- **Lifecycle.**
  - Added light: cur→prev = −1, so it contributes nothing to \(\hat p_{prev}\).
  - Removed light: prev→cur = −1, so the forward shift is undefined.
  - Intensity change: handled by evaluating each frame with its own buffer.
  - Interactive dragging: fine, since the per-frame transforms are simply different. For *validation* sequences, drive lights from keyframes so runs are reproducible.

### 3.6 Stale cached radiance: the real dynamic-lighting cost

- **ReSTIR GI (§4.3).**
  - Stale \(L_o\) biases temporal reuse and produces lag.
  - Fix: periodic sample validation every N frames (6 by default). Re-trace with the *same random numbers*, clear the reservoir if radiance moved beyond a tolerance, and also re-test the visible→sample segment for new occluders.
  - Reusing the initial-sampling pass means no extra cost on validation frames (§5).
- **ReSTIR PT reference code.** Optional `traceTemporalUpdate` refreshes the reconnection-vertex radiance by re-sampling NEE or re-tracing the suffix with the stored seed (§2.6). Off by default.
- **Recommended policy** [INFERENCE]:
  1. Keep the invariant: *a reservoir stored for frame \(t\) holds suffix radiance valid under \(S_t\).*
  2. Maintain a per-frame flag `lightsChanged(t) = any light transform/intensity/existence differs between t−1 and t`. For finer granularity, use a per-light "changed" bit and the set of lights a suffix touched. Keep that optional; a global flag is simpler and correct.
  3. If `!lightsChanged(t)` (camera-only or static frame), reuse cached radiance. This is **exact** for static geometry.
  4. If `lightsChanged(t)`:
     - Forward shift: re-trace the temporal sample's suffix under \(S_t\).
     - Inverse shift, exact mode: re-trace the canonical sample's suffix under \(S_{t-1}\).
     - Inverse shift, fast mode: skip it and accept the small darkening from the §3.4 table, using contribution MIS so only support errors bias.
  5. Spatial reuse in frame \(t\) then only sees frame-\(t\)-valid radiance.
- **Cost** [INFERENCE]. Worst case per pixel is roughly 2 prefix replays + 2 suffix re-traces + 2 reconnection visibility rays per temporal pass, only in frames where lights change. With software BVH traversal in WebGPU this dominates frame time during light drags.
  - Mitigations: stream-compact the replay/re-trace work (ENH §6.2.2; course notes §7.2.3, appendable queues + `dispatchWorkgroupsIndirect`).
  - Alternatively, ReSTIR GI-style amortized validation (every N frames, or only on light-change frames) in "interactive mode".

### 3.7 Confidence capping and history control

- **Values.** See the executive-summary table. ENH §7 and GRIS §8.3 use \(c_{Cap}=20\). GRIS Fig. 9a shows cap 5 and 20 behaving well, 100 drifting upward, and no cap diverging.
- **Why cap** (GRIS §6.4, p. 75:16–17; ENH §2.2 `ENH` L127–130). A cap avoids slow sample turnover in dynamic scenes and unbounded correlation build-up. The temporal relative weight is at most \(c_{cap}/(c_{cap}+1)\).
- **Frame-rate awareness** [INFERENCE]. A cap is "frames of history". NRD recommends `maxAccumulatedFrameNum = accumulationPeriodInSeconds * FPS` for denoisers (NRD README). An analogous \(c_{cap}(\text{fps})\) keeps history duration constant. Keep \(c_{cap}\) fixed in validation.
- **Adaptive caps, unbiased vs biased** (course notes §4.4 rule):
  - **Unbiased:** caps or resets driven by G-buffer or scene-level information independent of the stored samples. Examples: no predecessor or G-buffer mismatch → \(c_p=0\); "a light moved this frame" → lower \(c_{cap}\) globally or near the light [INFERENCE: independent of sample values].
  - **Biased** (depend on sample values or seeds):
    - ENH duplication map. After each frame, count reservoirs in a 17×17 window sharing the pixel's seed and divide by 288 to get \(D\in[0,1]\). Look up \(D\) at the back-projected pixel. Then \(c_{Cap}=\mathrm{lerp}(c^{Default}_{Cap},c^{min}_{Cap},D^\alpha)\) with \(c^{min}=1\), \(\alpha=0.1\), default 20 (ENH §5, `ENH` L452–470, page image p-09).
    - ENH explicitly says this violates MIS partition of unity and is biased: 3.25% mean absolute relative bias in Kitchen (Fig. 5); unbiased if disabled (§7.4).
    - RTXDI parameterization: \(\alpha=0.1\cdot2^{6(1-s)-3}\) with strength \(s=0.5\) giving \(\alpha=0.1\); `reduced = max(1, int(lerp(maxHistory, 1, D^α)))`. The map is built with 16×16 workgroups and a 32×32 LDS tile (`PT/DuplicationMap.hlsli` L109–125). In WGSL that is 4 KiB of `var<workgroup>` u32, within the 16 KiB default.
    - Age-based rejection (RTXDI). Also sample-dependent [INFERENCE].
    - RTXDI boiling filter. Averages resampling weight over a warp and clears reservoirs with \(w_i>\beta\bar w\), \(\beta=-9+10/s\), \(s\in(0,1]\) (ENH §7.2, `ENH` L746–752). Causes energy loss (ENH Fig. 9).
    - ReSTIR GI sample validation. Clears reservoirs based on their sample's radiance change: sample-dependent, but it removes a larger staleness bias [INFERENCE].

### 3.8 Temporal neighbour selection, motion vectors, reprojection

Static geometry + moving camera [INFERENCE: standard derivation].

- **Primary hit.** Store a V-buffer per frame (tri id, barycentrics, or world position) plus linear depth and normals. Swap each frame. World position \(x_1\) of pixel \(q\) comes from the current V-buffer.
- **Back-projection.** \(\mathbf s'=\mathrm{Proj}_{t-1}\mathrm{View}_{t-1}\,[x_1;1]\), then NDC → pixel, giving \(q'_{f}\). Motion vector = \(q'_f-q_c\) (pixel units). With subpixel jitter, use unjittered matrices for the motion vector, or include the jitter consistently.
- **Rounding.**
  - Original code: \(q'=\lfloor q+mv+\xi\rfloor\), \(\xi\sim U[0,1)^2\) (stochastic rounding).
  - RTXDI: `round(q + mv)`, plus a random ±0.5 jitter when permutation sampling is off.
- **Validity** (G-buffer only, so no bias):
  - normal: \(\langle n,n'\rangle\ge\tau_n\);
  - depth: relative difference of expected previous linear depth (current depth + motion z) vs stored previous depth \(\le\tau_z\);
  - optionally material similarity (RTXDI).
  - Thresholds: ReSTIR DI 25° / 10% depth; ReSTIR GI 25° / 0.05; RTXDI PT `normalThreshold 0.6` (a dot-product threshold) and `depthThreshold 0.1`.
- **Neighbourhood search.** If the centre tap fails, try other taps: RTXDI PT tries a fixed 3×3 ring; RTXDI DI tries 8 random offsets of about ±2 px (`int((rand−0.5)·4)`, doubled under checkerboarding). Then optionally the zero-motion fallback.
  - Unbiased as long as the search decides on G-buffer data, not reservoir contents [course notes Tip 4.1].
  - RTXDI PT also skips taps whose reservoir is empty (M = 0). That depends on history existence, not sample value; probably benign [INFERENCE/UNVERIFIED].
- **Permutation sampling** (RTXDI `ReservoirAddressing.hlsli` L50–59). A per-frame random XOR shuffle of the reprojected position within 4×4 blocks. It decorrelates temporal chains at the cost of reusing from a slightly different pixel, and is unbiased (sample-independent choice) [INFERENCE].
- **Frame-boundary pixels.** No predecessor: set \(c_p=0\) (course notes §4.4).

### 3.9 Disocclusion handling

- **Why it hurts.** Newly visible surfaces have no history, so quality drops to 1-spp path tracing (Reservoir Splatting §1). ENH §6.4 calls it severe noise.
- **Options, cheap to expensive:**
  1. **Reset and boost spatial reuse.** ReSTIR GI raises spatial `maxIterations` to 9 when spatial M is low; RTXDI PT sets a flag when no temporal neighbour exists and uses `numDisocclusionBoostSamples = 8` spatial samples. Unbiased (G-buffer-driven).
  2. **Zero-motion fallback tap** plus permutation (RTXDI PT). Often finds the background surface when only the camera moved slightly.
  3. **Dual motion vectors** (Zeng et al. 2021, Eq. 6), used by ENH §6.4 for ReSTIR temporal resampling (supplemental Fig. 6: FLIP 0.259 → 0.231 in Veach Ajar):
     - For disoccluded pixel \(x_i\): back-project to the previous frame, landing on the occluder at \(y\). Forward-project that occluder point to the current frame, giving \(z\). Use \(x^O_{i-1}=y+(x_i-z)\) (screen space).
     - This assumes the occluder–background relative offset is preserved (transcribed from the paper text extraction; notation approximate).
     - ENH notes ReSTIR PT avoids the "copy-paste" artefacts of colour-reusing methods because it resamples paths. Some correlated noise remains, which decorrelation reduces.
     - Unbiased as a neighbour-choice heuristic, since it is G-buffer based [INFERENCE].
     - With static geometry the occluder motion is pure camera parallax, so both projections use camera matrices only.
  4. **Backup samples + splatting** (Reservoir Splatting §4.2), **multi-layer splatting** (2026), **stochastic pairwise MIS** (2026). Research-grade; defer.

### 3.10 Reservoir splatting (forward reprojection) as an alternative temporal gather

From Reservoir Splatting §4 (text extraction; Eq. 14–16 structure only, [UNVERIFIED exact form]).

- **Idea.** Scatter each previous reservoir's path into the current frame by keeping its **object-space primary hit** \(x_1\) (\(y_1=M(t)M(t-1)^{-1}x_1\); identity for static scenes). Connect to the current camera, test visibility of \(y_0\to y_1\) (occluded ⇒ undefined), then hybrid-shift the suffix.
- **GRIS with scatter.** Conceptually every previous sample is an input to every pixel, but with a box pixel filter only the pixel it lands in has non-zero target. The balance heuristic therefore reduces to two terms:
  \[m_i(Y_i)=\frac{c_i\,\hat p^{prev}(X_i)|\partial Y_i/\partial X_i|^{-1}}{c^*\hat p(Y_i)+c_i\,\hat p^{prev}(X_i)|\partial Y_i/\partial X_i|^{-1}}\ \ (\text{Eq. 11}),\qquad m_{N+1}(Y^*)=\frac{c^*\hat p(Y^*)}{c^*\hat p(Y^*)+c_i\,\hat p^{prev}(T^{-1}(Y^*))|\partial T^{-1}/\partial Y^*|}\ \ (\text{Eq. 13}).\]
  The \(c_i\) in Eq. 13 belongs to the previous pixel hit by the *reverse splat* of the canonical sample (zero if occluded or off-screen).
  - **Note that the paper again uses \(\hat p^{prev}\): the previous-frame target.**
- **Jacobian.** Subpixel parameterization \(\times\) \(|\partial y_1/\partial x_1|\) (1 for rigid) \(\times\) hybrid-shift Jacobian. Here \(|\partial x_1/\partial s|=\|x_1-x_0\|^2\cos^3\theta_c/\cos\theta_n\) (pinhole projection density).
- **Confidence** (Eq. 17). Bilinearly interpolate the confidences of the 2×2 previous pixels around the back-projected pixel centre, plus the backup's confidence, then cap. Using only contributing splats would bias.
- **Cost and behaviour** (§7). 0.6–0.9 valid splats per pixel on average, max about 5. Atomics are needed for concurrent splats. Holes appear under zoom or forward motion, which backup samples fill. It inherits forward-motion-vector issues with mirrors.
- **WebGPU feasibility** [INFERENCE].
  - WGSL has 32-bit `atomicAdd` / `atomicMax` on storage buffers, so implement bounded per-pixel splat lists: `atomicAdd` a counter, K ≤ 8 slots, overflow dropped. Dropping on overflow is a sample-dependent decision, so keep K large or make drops independent of sample values.
  - Then a gather pass performs streaming RIS over the list plus the canonical sample.
  - 64-bit `atomic<vec2u>` min/max had an intent-to-ship around Sept 2026 [UNVERIFIED availability]. Do not depend on it.

### 3.11 Recommended temporal pipeline for this project

Assuming static geometry, dynamic camera and lights [INFERENCE].

Per frame \(t\) (compute passes; buffers double-buffered where marked *):

1. **Scene update.** Write `camCur` and `lightsCur`, keep `camPrev` and `lightsPrev*`. Build `curToPrev` / `prevToCur` maps and light CDFs (cur, prev). Set `lightsChanged`.
2. **Primary visibility.** Write V-buffer*, depth*, normals*, motion vectors.
3. **Initial sampling.** One path tree per pixel with NEE (RIS over light candidates if many lights). Unified DI+GI as in ENH §6.1. Store canonical reservoir \(c=1\), \(F\), \(W\), seeds, reconnection data (object-attached or light-local), cached suffix radiance.
4. **Temporal neighbour.** Back-project, G-buffer validate, 3×3 search, optional dual MV and fallback. Set \(c_p=\min(c,c_{cap})\), or 0 if none found.
5. **Temporal shifts** (split into replay and resample kernels, course notes §7.2.3):
   - Forward: \(Y_p=T(X_p)\) under \(S_t\), with suffix re-trace if `lightsChanged`.
   - Inverse: \(T^{-1}\) of the canonical (Talbot), or of the selected sample (contribution MIS), under \(S_{t-1}\) = (`camPrev`, V-buffer_prev[q'], `lightsPrev`, prev CDF, static BVH, suffix re-trace with `lightsPrev` if `lightsChanged` and exact mode).
6. **Temporal GRIS** (§3.1). Store \(F\) (RGB under \(S_t\)), \(\hat p\), \(W\), \(c\).
7. **Spatial reuse.** 3 neighbours, defensive pairwise MIS; paired reuse (ENH §3) optional. Vector resampling weights for shading (ENH §6.3).
8. **Shade** \(F(Y)W_Y\), or \(\sum w_i\) vectorized. Store the final reservoirs as next frame's temporal input.
9. **Denoise or accumulate** (§5).

**Modes:**
- **Validation (unbiased):** exact previous evaluation, suffix updates on, no duplication map / boiling / age rejection, fixed \(c_{cap}=20\).
- **Interactive:** contribution-MIS temporal, cached radiance with suffix updates only on light-change frames (or amortized), duplication-map cap (ENH), dual MVs, disocclusion boost, denoiser on.

**Unit tests** [INFERENCE, extending course notes Tip 5.1]:
1. Static camera and lights, no jitter: \(T\) is the identity, \(|J|=1\), and \(\hat p_{t,q}(Y_p)\) equals the stored \(\hat p\) (up to float error).
2. \(T^{-1}(T(x))=x\): same seed and reconnection vertex. The recomputed \(\hat p_{prev}\) equals the stored value when the previous evaluation is exact.
3. Light translation: the (id,uv) sample lands on the moved light, with PSS Jacobian 1.
4. Removed light: the forward shift is undefined.
5. Converged mean of many still frames equals a path-traced reference, for spatial-only then temporal-only modes (course notes Tip 4.4).

---

## 4. Robustness beyond dynamics (brief, implementation-relevant)

- **Correlation sources** (ENH §5): low-probability, high-energy fireflies from initial sampling, and imperfect shifts that turn samples into fireflies. They spread spatially and persist temporally.
- **Remedies:**
  - Adaptive \(c_{Cap}\) via duplication map (biased).
  - CRIS final gather and MCMC mutations (unbiased, expensive).
  - RTXDI decorrelation at final shading: with some probability fall back to the preserved unresampled initial reservoir, plus firefly replacement (`ReSTIRPTParameters.h`, `RTXDI_PTDecorrelationParameters`; default "Stagnancy" only with DLSS-RR). Changes the shading estimator, so bias status is [UNVERIFIED].
- **Colour noise** (ENH §6.3). Resampling uses scalar \(\hat p=|F|\). Accumulating vector weights \(\mathbf w_i=m_i(Y_i)F(Y_i)W_{X_i}|\partial T_i/\partial X_i|\) and shading with \(\sum\mathbf w_i\) decouples resampling from shading at no extra cost (`ENH` L539–555). This is the marginalization over the index choice; GRIS Eq. 46–47 discusses the offline analogue.
- **Paired spatial reuse, reconnection footprint thresholds, unified DI/GI.** See ENH §3, §4, §6.1. These are covered by other workstreams; they do not change the temporal math.

---

## 5. Denoising and display

### 5.1 What the literature pairs with ReSTIR

- **ReSTIR DI (Fig. 2).** Better samples help existing denoisers (OptiX shown). "Filter PDFs rather than colours" (§8).
- **ReSTIR GI.** Used with a denoiser; references SVGF / A-SVGF (§1, §4.3).
- **ENH §2.** Real-time path tracing is practical thanks to SVGF, NVIDIA NRD, and DLSS Ray Reconstruction. Correlation artefacts make ReSTIR hard to denoise (§5).
- **CRIS.** Reduced correlation improves modern denoisers' behaviour on ReSTIR PT (abstract).
- **NRD** (README, `github.com/NVIDIA-RTX/NRD`).
  - RELAX was designed for RTXDI signals and works well with them. It is à-trous-based and tracks second moments (HDR range \([0,250]\) recommended).
  - REBLUR uses normalized hit distance.
  - Required inputs: motion vectors, normal+roughness, view Z, hit distance.
  - **Materials must be demodulated** before denoising.
  - An optional history-confidence input in \([0,1]\) comes from gradient-based change detection.
  - Advice on filtering `hitT` through the BRDF lobe applies when using advanced sampling.
- **RTXDI** keeps a `t_PrevRestirLuminance` buffer in its sample (`RAB_Buffers.hlsli` L32), apparently for temporal-gradient / confidence computation [UNVERIFIED purpose].

### 5.2 Feasible in WebGPU (Chrome/Metal on M5 Pro) [INFERENCE unless cited]

- **SVGF** (Schied et al. HPG 2017).
  - Temporal accumulation of colour and first/second luminance moments with back-projection. Variance estimate, then a 5-level edge-avoiding à-trous filter with depth/normal/luminance-variance stopping functions.
  - A-SVGF's variant uses a 3×3 box kernel, giving a 49×49 effective footprint over 5 levels (A-SVGF §4).
  - Typical constants: α ≈ 0.2, σ_z = 1, σ_n = 128, σ_l = 4 [UNVERIFIED: from memory of the SVGF paper].
  - Pure compute, cheap, many WebGPU/WGSL precedents: `DavidPeicho/loupiote` (includes SVGF), `theMagicalKarp/wgsl-raytrace` (à-trous), James Randall's WebGPU path tracer (temporal + spatial).
- **A-SVGF** (Schied, Peters, Dachsbacher, PACM CGIT 2018) replaces fixed α with a per-pixel adaptive factor from temporal gradients:
  - One gradient sample per 3×3 stratum by **forward-projecting** a previous surface sample *with its random seed* and re-shading it in the current frame.
  - Temporal gradient \(\delta=f_i(G_{i-1,j},\xi_{i-1,j})-f_{i-1}(G_{i-1,j},\xi_{i-1,j})\).
  - Normalizer \(\Delta=\max(f_i(\cdot),f_{i-1}(\cdot))\) (Eq. 13). Both are reconstructed with the joint-bilateral à-trous weights (Eq. 12).
  - \(\lambda=\min(1,|\hat\delta|/\hat\Delta)\) (Eq. 14), \(\alpha_i=(1-\lambda)\alpha+\lambda\) (Eq. 15), take the max over 3×3, then EMA \(\hat c_i=\alpha c_i+(1-\alpha)\hat c_{i-1}\) (Eq. 1).
  - **Synergy [INFERENCE]:** ReSTIR's forward temporal shift already re-evaluates the previous sample under the current scene (same seed, same reconnection vertex). \(\hat p_{t,q}(T(X_p))\) versus the stored \(\hat p_{t-1,q'}(X_p)\), with Jacobian awareness, is a free temporal-gradient sample for driving the denoiser's α or history confidence, especially when lights move. It must drive only the *denoiser*, not ReSTIR confidences, or it becomes a sample-dependent reset (§3.7).
- **ReLAX-like.** A WGSL port of NRD ReLAX is heavy, and NRD's licence terms should be checked [UNVERIFIED]. SVGF / A-SVGF captures most of the value for a first version.
- **Demodulation.** Denoise irradiance-like signals (diffuse radiance / albedo, specular separately) and remodulate after (NRD requirement). Needs albedo and roughness from the G-buffer.
- **Neural (still frames).** `oidn-web` (`github.com/pissang/oidn-web`, npm `oidn-web`) runs the OIDN U-Net with TF.js on the WebGPU backend and requires WebGPU. `DennisSmolek/Denoiser` is similar. Good for accumulated or static screenshots; not a real-time 1080p per-frame denoiser [UNVERIFIED performance].
- **Platform limits** (W3C WebGPU spec defaults; request higher from the adapter):

  | Limit | Default | Implication |
  |---|---|---|
  | `maxStorageBufferBindingSize` | 128 MiB | A 64-byte reservoir at 1920×1080 is 132,710,400 B ≈ 126.6 MiB: just fits one binding. |
  | `maxBufferSize` | 256 MiB | |
  | `maxComputeWorkgroupStorageSize` | 16 KiB | |
  | `maxComputeInvocationsPerWorkgroup` | 256 | |
  | `maxStorageBuffersPerShaderStage` | 8 | Resampling kernels binding V-buffers, reservoirs, light buffers, maps, BVH, materials will exceed 8: request a higher adapter limit or pack buffers. |
  | `maxStorageTexturesPerShaderStage` | 4 | |
  | `maxBindGroups` | 4 | |

  Subgroups shipped in Chrome 134 (`enable subgroups;`).

### 5.3 Progressive accumulation mode (for convergence checks against Cycles)

- **When.** Camera, lights, scene, resolution and all render settings unchanged since the last reset. Reset \(n\leftarrow0\) on any change. Implement a dirty flag bumped by the UI, camera controller, and light editor.
- **What to accumulate.** \(A_n=A_{n-1}+(x_n-A_{n-1})/n\) in fp32 per channel. Precision is fine up to about \(10^5\)–\(10^6\) frames; optionally keep fp32 sum + count or Kahan compensation [INFERENCE].
- **Which estimator:**
  - **Preferred: temporal reuse OFF, spatial reuse ON, independent seeds per frame.** Each frame is an unbiased, independent estimate, so the mean converges at the Monte Carlo rate. GRIS §6.5 proposes this for offline rendering and shows it converges faster than accumulating temporal ReSTIR (Fig. 9b). ENH §7.4 says the same when a noise-free unbiased image is wanted. GRIS's offline configuration: 32 initial path trees, 3 spatial iterations × 6 neighbours within 10 px.
  - **Also useful: temporal reuse ON with \(c_{cap}\).** The average of frames still converges in a static scene (GRIS §6.4 box and Fig. 9b), but more slowly because of inter-frame correlation. Negative test: **with no cap it converges to the wrong answer** (GRIS Fig. 2, Fig. 10b).
  - Disable all biased features (duplication-map cap, boiling filter, age rejection, firefly clamps, denoiser). Keep a **plain path tracer** mode in the same code for A/B (course notes §9).
- **Pixel filter.** Match the reference's pixel filter. Uniform subpixel jitter per frame equals a 1-px box filter; set the Cycles filter accordingly [UNVERIFIED Cycles settings; validation workstream].

### 5.4 Validating temporal unbiasedness under motion

- **ENH methodology** (Fig. 14–15, §7.4). Run the *same camera animation* 1024 times with different seeds, capture the same frame during motion, and average across runs. Plot MSE vs iteration and FLIP vs cumulative time.
  - Their biased default shows systematic error in the averaged image. The unbiased (no-decorrelation) variant does not.
- **Our version** [INFERENCE]:
  1. Keyframe camera and lights deterministically.
  2. For frame \(t^*\) of the animation, render N independent runs (fresh seeds, full history from frame 0).
  3. Average frame \(t^*\) across runs.
  4. Compare with a Cycles render of that exact frame state (same camera and light transforms at \(t^*\)).
  5. Report bias maps \((\bar I-I_{ref})/I_{ref}\) (ENH Fig. 5f) and mean \(|bias|/ref\).
  - This directly tests the temporal shift, the previous-frame MIS evaluation, and light remapping. A wrong \(\hat p_{prev}\) shows up as darkening or brightening bands near moving lights, per §3.4.
- **Metrics.**
  - MAPE (GRIS footnote 13): \(\mathrm{mean}\big(|I-I_{gt}|/(0.01\cdot\mathrm{mean}(I_{gt})+I_{gt})\big)\).
  - HDR-FLIP (ENH §7).
  - MSE / relMSE.
  - RMAE (ReSTIR DI §6).

---

## 6. Parameter cheat-sheet (sources)

| Parameter | Value(s) | Source |
|---|---|---|
| Temporal confidence cap | 20 | GRIS §8.3; ENH §7; course notes §4.4 (5–30) |
| Temporal cap (ReSTIR DI) | 20× current M | Bitterli 2020 §5 |
| Temporal / spatial M clamp (ReSTIR GI) | 30 / 500 | Ouyang 2021 §5 |
| RTXDI PT `maxHistoryLength` | 8 (20 with DLSS-RR) | `Source/ReSTIRPT.cpp` |
| Duplication map window / normalizer | 17×17 / 288 | ENH §5 |
| Adaptive cap: \(c^{min}\), \(\alpha\) | 1, 0.1 | ENH §5 |
| Spatial neighbours / radius | 3 / 30 px (ENH); 3 / 20 px (GRIS real-time); 5 / 30 px (DI biased); 1 / 32 (RTXDI PT) | ENH §7; GRIS §8.3; DI §5; RTXDI |
| Paired-reuse Gaussian \(\sigma\) | \(\sqrt{8/(9\pi)}\,R\) (16.0 for R = 30) | ENH §7 (`ENH` L591–596) |
| Neighbour validity | 25° / 10% depth (DI); 25° / 0.05 (GI); dot ≥ 0.6 / 0.1 (RTXDI PT) | as cited |
| Reconnection footprint \(\kappa\) | 0.02 (optimum 0.005–0.04) | ENH §4, §7.2; supplemental §10 |
| Single-vertex roughness threshold | 0.2 (ENH), 0.1 (RTXDI) | ENH §7.2; `ReSTIRPT.cpp` |
| Reservoir size | 64 B (ENH); 88 B (original) | ENH supplemental §8; GRIS §8.2 |
| ReSTIR GI validation period | 6 frames | Ouyang 2021 §5 |
| Disocclusion spatial boost | 8 samples | RTXDI PT defaults |
| RTXDI `maxReservoirAge` | 30 | RTXDI PT defaults |
| Boiling filter strength | 0.2; \(\beta=-9+10/s\) | RTXDI; ENH §7.2 |

---

## 7. Risks and open questions

1. **Cached suffix radiance under moving lights.** The exact fix doubles or triples path-tracing cost in light-change frames. The mode split (exact vs interactive) needs profiling on M5 Pro with software BVH traversal.
2. **Delta-light Jacobian and target conventions in PSS.** Point and spot NEE reconnection: RTXDI uses a solid-angle-pdf ratio. Pin down the convention with unit tests and Cycles comparisons [UNVERIFIED].
3. **Area-light hit semantics.** Whether BSDF rays terminate at or pass through lights, one- vs two-sided emission, and spread angle must match Cycles [UNVERIFIED; for the Blender workstream]. Blender point/spot "radius" and "soft falloff" defaults also differ from ideal delta lights [UNVERIFIED].
4. **Russian roulette in PSS.** ENH's trick (RR only alters the initial pdf; no RR in replay) must be implemented exactly, or temporal shifts will fail or bias.
5. **Light tiles / RIS-NEE vs replay.** Per-frame random light tiles are not replayable. Force reconnection at NEE vertices (ENH §6.2.3) and treat RIS-NEE as a black box in PSS (ENH supplemental §5).
6. **Duplication-map, age rejection, boiling filter** are biased and must be togglable. Keep validation mode free of them.
7. **Licensing.** RTXDI-Library is `LicenseRef-NvidiaProprietary`, so no code copying. The original ReSTIR_PT and NRD licences are [UNVERIFIED]; implement clean-room.
8. **WebGPU limits.** Storage-buffer count and binding sizes (§5.2) likely need raised adapter limits. Race-free splatting needs 32-bit atomics with bounded lists.
9. **Pan et al. 2024 temporal MIS weight and CGNS 2026.** Not studied in detail; they may lower noise in dynamic scenes [UNVERIFIED].
10. **Moving geometry (future).** Needs previous instance transforms, per-instance motion vectors, and ideally a previous BVH for exact previous visibility. Otherwise accept transient darkening (RTXDI's documented fallback).

---

## 8. Sources

Local:
- `scratchpad/restirpt_enhanced_2026.txt` + `pages/enhanced/p-03, p-04, p-09.png` (ENH §2.2 L111–133, §2.3 L134–205, §5 L420–476, §6 L477–564, §7 L565–603, L746–799)
- `scratchpad/pages/gris/p-02, p-07 … p-18, p-22, p-23.png` (GRIS §1, §4.3–4.4, §5.4–5.7, §6.1–6.5, §7.1–7.6, §8.1–8.3, §9.1–9.2, App. A, App. B)

Web (all accessed 2026-09-28):
- Course notes: https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf (index https://intro-to-restir.cwyman.org/)
- ReSTIR DI: https://cs.dartmouth.edu/~wjarosz/publications/bitterli20spatiotemporal.pdf
- ReSTIR GI: https://d1qx31qr3h6wln.cloudfront.net/publications/ReSTIR%20GI.pdf
- Reservoir Splatting: https://research.nvidia.com/labs/rtr/publication/liu2025splatting/liu2025splatting_paper.pdf ; code https://github.com/Jebbly/Reservoir-Splatting
- ENH supplemental: https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/lin2026restirptenhanced_supplemental.pdf ; project https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/
- CRIS: https://dqlin.xyz/pubs/2023-sa-CRIS/ ; code https://github.com/NVlabs/conditional-restir-prototype
- Area ReSTIR: https://dqlin.xyz/pubs/2024-sig-AREA/ ; https://research.nvidia.com/labs/rtr/publication/zhang2024area/
- Decorrelating ReSTIR (MCMC): https://arxiv.org/abs/2211.00166 ; https://research.nvidia.com/labs/rtr/publication/sawhney2022decorrelating/
- Multi-Layer Reservoir Splatting (2026): https://graphics.cs.utah.edu/research/projects/multi-layer-restir/ ; https://research.nvidia.com/labs/rtr/publication/hong2026multilayer/
- Stochastic Pairwise MIS (2026): https://research.nvidia.com/labs/rtr/publication/hedstrom2026stochastic/hedstrom2026stochastic.pdf
- Compatibility-Guided Neighbor Selection (2026): https://research.nvidia.com/labs/rtr/publication/junkins2026compatibility/
- Pan et al. 2024 MIS weight: https://onlinelibrary.wiley.com/doi/10.1111/cgf.15049
- NVIDIA RTR publication list: https://research.nvidia.com/labs/rtr/publication/
- RTXDI-Library (PT/DI/Utils headers, `Source/ReSTIRPT.cpp`): https://github.com/NVIDIA-RTX/RTXDI-Library
- RTXDI sample and docs (`Doc/Integration.md`, `RAB_VisibilityTest.hlsli`, `RAB_LightInfo.hlsli`, `RAB_Buffers.hlsli`, `PrepareLights.hlsl`): https://github.com/NVIDIA-RTX/RTXDI
- Original ReSTIR PT code (`TemporalReuse.cs.slang`, `Shift.slang`, `PathTracer.slang`, `PathReservoir.slang`, `ReSTIRPTPass.h/.cpp`): https://github.com/DQLin/ReSTIR_PT
- A-SVGF: https://cg.ivd.kit.edu/publications/2018/adaptive_temporal_filtering/adaptive_temporal_filtering.pdf
- Dual motion vectors: https://sites.cs.ucsb.edu/~lingqi/publications/paper_trmv.pdf
- NRD: https://github.com/NVIDIA-RTX/NRD
- oidn-web: https://github.com/pissang/oidn-web
- WebGPU spec limits: https://www.w3.org/TR/webgpu/
- Chrome 134 subgroups: https://developer.chrome.com/blog/new-in-webgpu-134
- WGSL `atomic<vec2u>` intent to ship: http://www.mail-archive.com/blink-dev@chromium.org/msg17518.html
- WebGPU path tracers with SVGF: https://github.com/DavidPeicho/loupiote ; https://github.com/theMagicalKarp/wgsl-raytrace
