# Validation methodology and automated test harness for a WebGPU ReSTIR PT renderer (vs. Blender Cycles, plus internal unbiasedness)

Report for implementation engineers. Topic: how to show that (1) our WebGPU path tracer (PT) matches Blender Cycles 5.1.2, and (2) our ReSTIR PT matches our PT in expectation, i.e. it is unbiased. The report also covers the property and unit tests, the macOS automation, image IO and the test scene suite.

> NOTE ON LOCATION: This session ran in plan mode (read-only). The only writable file was this plan file, so the report is here and NOT at the requested path `/private/tmp/claude-502/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-ea73-491f-b7b7-6d0aab1ffce1/scratchpad/research/validation-harness.md`. Nothing was cloned or extracted to disk. Package tarballs were inspected by streaming (`curl | tar -t` / `tar -xO`). Upstream sources were read through `curl` to stdout or WebFetch.

Tags: **[SOURCE]** marks what a paper or code says, with its location. **[VERIFIED]** marks what I checked on this Mac on 2026-09-28. **[INFERENCE]** marks my own reasoning. **[UNVERIFIED]** marks plausible claims that nobody has confirmed yet, each with a suggested check.

---

## 0. TL;DR (decisions)

1. **Two-stage acceptance, never skip stage 1.**
   (A) Our *plain PT* must match *Cycles* in expectation on every scene, within a declared equivalence tolerance. This is a statistical equivalence test (TOST), not only "no significant difference".
   (B) Our *ReSTIR PT, in its unbiased preset*, must match *our plain PT* in expectation. Both run in the same engine, so the integrand is identical and the tolerance can be much tighter.
   Biased features are excluded from (B). They get their own bias-budget metrics instead. These include the Enhanced paper's duplication-map c_cap reduction (§5, explicitly biased), any clamping, and boiling filters.
2. **Estimate uncertainty from replicates of the statistic itself, never from per-pixel variances summed across pixels.** This matters once spatial reuse is on, because it correlates neighboring pixels.
   - Cycles: K ≥ 16 independent **seeds** (not `sample_offset` subsets of one sequence).
   - Our PT: batch means, accumulated in Float64 on the CPU.
   - ReSTIR: R independent **runs** (ensemble). The Enhanced paper averages 1024 runs (Fig. 14/15, p.17). Long single-run batch means are allowed only when batch length ≫ τ_int ≈ 2·c_cap+1 frames.
3. **Harness:**
   - **Primary:** Playwright 1.63 drives the *installed* Google Chrome 154 (`channel:'chrome'`, new headless, `--enable-unsafe-webgpu`). It loads a `/harness.html` page served by the Vite dev server. Float32 images go back by POSTing raw bytes to a Vite middleware that writes PFM/EXR files.
   - **Fallback 1:** the same, headed.
   - **Fallback 2 / fast shader-unit-test lane:** Node 26 + npm `webgpu@0.6.1` (Dawn's `dawn.node`, ships a `darwin-universal` prebuilt) running the *same* TS core and WGSL.
   - One TS core takes an injected `GPUDevice` and contains no DOM code.
4. **Analysis:**
   - Python 3.13 in a `uv` venv: numpy, OpenImageIO, scipy, matplotlib, `flip-evaluator` 1.7. flip-evaluator has **no cp314 wheel**, and system python is 3.14, so the venv must be 3.13.
   - Blender's bundled Python 3.13.9 already has numpy 2.3.4 and OpenImageIO 3.1.7 (EXR + PFM), so it can read/convert EXRs with zero installs.
   - There is no JS port of FLIP on npm.
5. **Cycles must be de-biased and de-defaulted.** In Blender 5.1.2 the defaults are biased or mismatched for this purpose:
   - `sample_clamp_indirect=10`, `blur_glossy=1.0`, `light_sampling_threshold=0.01`
   - Glossy/Principled `distribution='MULTI_GGX'`
   - Blackman-Harris 1.5 px filter
   - adaptive sampling and OIDN denoising on
   - gray world 0.0509
   - `use_soft_falloff=True`, `shadow_terminator_geometry_offset=0.1`
   - camera `clip_start=0.1`, `sensor_fit='AUTO'`
   - view transform AgX
   
   All of these must be overridden (full list in §7.1).
6. **Build Blender scenes from *our* parsed scene (a "scene bridge"), not from Blender's glTF/USD importers.** This separates renderer correctness from importer differences. Importer and loader fidelity gets its own geometry-level tests; pxr 0.25.8 is available inside Blender's Python for USD.
7. Use a **restricted material model for exact equivalence**:
   - Lambert (Diffuse BSDF, roughness 0)
   - single-scatter GGX (Glossy BSDF, `distribution='GGX'`)
   - constant-weight lobe mixtures (Mix Shader)
   - Emission

   Principled v2 layering cannot be matched without re-implementing Cycles' energy-compensation tables.

---

## 1. Environment facts checked on this Mac

| Item | Finding | Tag |
|---|---|---|
| OS / HW | macOS 26.6.2 (25G83), Apple M5 Pro, 16-core GPU, "Metal 4" | [VERIFIED] `sw_vers`, `system_profiler` |
| Node | v26.3.1 at `/opt/homebrew/bin/node`; npm, npx present | [VERIFIED] |
| Deno / Bun | not installed | [VERIFIED] |
| Python (system) | 3.14.6 (Homebrew). No numpy or OIIO. `uv` 0.11.2 at `~/.local/bin/uv` with managed CPython 3.13.12, 3.12.13, 3.11.15 already installed. python3.10 also present | [VERIFIED] |
| Toolchain | Xcode + Apple clang 21, cmake at `/opt/homebrew/bin/cmake` (enough to build FLIP C++ if ever needed) | [VERIFIED] |
| Chrome | Google Chrome 154.0.8037.58 in /Applications | [VERIFIED] |
| Playwright cache | `~/Library/Caches/ms-playwright`: chromium-1208, chromium-1243, chromium_headless_shell-1208/-1243, ffmpeg-1011 (Playwright used before on this machine) | [VERIFIED] |
| npm registry (2026-09-28) | `webgpu` 0.6.1, `playwright`/`@playwright/test` 1.63.0, `puppeteer` 25.12.0, `@webgpu/types` 0.1.74, `vitest` 5.0.2, `@vitest/browser` + `@vitest/browser-playwright` 5.0.2, `vite` 8.3.1, `tsx` 4.23.15, `@gltf-transform/core` 4.5.1, `three` 0.186.1, `parse-exr` 1.0.2 ("port of Three.js EXR loader, no three dependency") | [VERIFIED] `npm view` |
| `webgpu@0.6.1` contents | `dist/darwin-universal/dawn.node` (21.2 MB), linux-x64/arm64, win32-x64/arm64; `"type":"module"`; deps `@webgpu/types`, `debug`; `postinstall` removes `com.apple.quarantine` xattr on macOS; `index.js` loads `dist/${platform}-${isMac?'universal':arch}/dawn.node` and exports `{create, globals}`; README usage `Object.assign(globalThis, globals); const navigator={gpu:create([])}`; options `backend=metal`, `adapter=<name>`, `enable-dawn-features=...` | [VERIFIED] tarball streamed |
| Blender | 5.1.2 (hash ec6e62d40fa9, built 2026-05-19); bundled Python 3.13.9 at `/Applications/Blender.app/Contents/Resources/5.1/python/bin/python3.13` | [VERIFIED] |
| Blender Python modules | numpy 2.3.4, OpenImageIO 3.1.7.0 (formats incl. `openexr`, `pnm` → `pfm`), pxr (USD 0.25.8), MaterialX 1.39.4, requests; **no** OpenEXR/Imath/PIL/scipy/cv2; pip 25.2, venv + ensurepip importable | [VERIFIED] |
| Cycles devices | `('Apple M5 Pro','CPU')`, `('Apple M5 Pro (GPU - 16 cores)','METAL')` | [VERIFIED] |
| PyPI | `flip-evaluator` 1.7 wheels: cp38–cp313 (macOS arm64 included), **no cp314**. `openimageio` 3.1.17.0, `scipy` 1.18.1, `matplotlib` 3.11.2 all have cp313 macOS arm64 wheels. `OpenEXR` 3.5.1 has cp313 macOS wheels | [VERIFIED] PyPI JSON |

Not yet verified, and the first smoke tests to run:
- (a) `dawn.node` loads under Node 26 and returns a Metal adapter. N-API is ABI-stable, so this should work [INFERENCE].
- (b) `navigator.gpu.requestAdapter()` returns non-null in *headless* Chrome 154 on macOS. A third-party page reports new-headless Chrome on macOS supports WebGPU/Metal with only `--enable-unsafe-webgpu` (agent-browser.dev/webgpu) [UNVERIFIED on this machine].

---

## 2. What the sources say that constrains validation

### 2.1 Unbiased contribution weights (GRIS)

- **[SOURCE] GRIS Def. 4.1, Eq. 8 (p.5, `pages/gris/p-05.png`).** An unbiased contribution weight W ∈ ℝ for a random variable X ∈ Ω is any real random variable with
  `E[f(X)·W] = ∫_{supp(X)} f(x) dx`, for every integrable f.
- **[SOURCE] Eq. 9 (p.6) + Thm A.1 (p.22).** This is equivalent to `E[W | X] = 1/p_X(X)`.
- **[SOURCE] Eq. 3 (p.4).** With W_Y from Eq. 2 and supp f ⊂ supp Y: `∫_Ω f(x)dx = E[f(Y) W_Y]`.

**Test implication [INFERENCE].** The indicator choice f = 1_B gives the "W-histogram identity":

`E[W_Y · 1{Y ∈ B}] = |B ∩ supp Y|` for every measurable region B.

This is testable directly on toy domains (T7 in §4). It validates Def. 4.1 for all test functions at once. The identity holds in the renderer's PSS as well, but there the PSS coordinates of shifted paths are awkward to recover, so use it on toys.

### 2.2 GRIS resampling weights, partition of unity and coverage

- **[SOURCE] Eq. 7 / 19 (p.5, p.7):**
  `w_i = m_i(T_i(X_i)) · p̂(T_i(X_i)) · W_i · |∂T_i/∂X_i|` if X_i ∈ D(T_i), else 0.
  The Enhanced paper restates this as Eq. 1 (p.3): `w_i = m_i(Y_i) p̂(Y_i) W_{X_i} |∂T_i/∂X_i|`.
- **[SOURCE] Eq. 22 (p.7):** `W_Y = (1/p̂(Y)) Σ_{j=1}^{M} w_j`. The division by M is absent because the m_i already carry it (footnote 6).
- **[SOURCE] Eq. 20 (p.7), resampling-MIS partition of unity:**
  `Σ_{i: y ∈ T_i(supp X_i)} m_i(y) = 1` for all y ∈ supp Y, with m_i ≥ 0.
  Eq. 17 is the same condition for contribution MIS c_i. Both sums run *only over the techniques that can produce y with positive PDF*.
- **[SOURCE] Eq. 15 (p.6):** `supp Y = supp p̂ ∩ ∪_i T_i(supp X_i)`.
- **[SOURCE] Def. 5.2 (p.9), canonical sample:** X_i is canonical if Ω_i = Ω, T_i = identity, p̂_i = p̂ and supp p̂ ⊂ supp X_i.
  Enhanced §2.1 (p.3): "Unbiased integration of f using Y is typically guaranteed by including a canonical sample that completely covers the support of p̂."
- **[SOURCE] GRIS Appendix B, "On Visibility" (p.23).** If p̂ ignores visibility, samples may have p̂>0 where they are never generated, so coverage breaks. "ReSTIR PT always considers visibility between vertices."
- **[SOURCE] GRIS Appendix B, "Temporal Reuse" (p.23).** "Temporal sample reuse is unbiased with proper MIS weights … but a temporal shift mapping is needed … fully unbiased temporal reuse must evaluate paths in both the current and prior frames." Biased approximations exist, e.g. ignoring visibility as in ReSTIR GI.

**Test implication [INFERENCE].** With moving lights or geometry, the temporal MIS weights must evaluate the prior-frame target p̂_{t−1} with **prior-frame scene state** (light transforms, BVH). An implementation that evaluates p̂_{t−1} on the current scene will pass static tests and fail the dynamic tests in §8 (ix). The harness has to include dynamic scenes for exactly this reason.

### 2.3 MIS families actually used, and the exact weights in the reference code

**[SOURCE] GRIS Eq. 35 (p.9):**
`p̂_{←i}(y) = p̂_i(T_i^{-1}(y)) · |∂T_i^{-1}/∂y|` if y ∈ T_i(supp X_i), else 0.

**[SOURCE] Generalized Talbot MIS, Eq. 36:**
`m_i(y) = p̂_{←i}(y) / Σ_j p̂_{←j}(y)`

**[SOURCE] Generalized pairwise MIS, Eq. 37 (p.9)** (|R| canonical samples, M inputs):
- for i ∈ R: `m_i(y) = (1/(M−|R|)) Σ_{j∉R} p̂(y) / (|R| p̂(y) + (M−|R|) p̂_{←j}(y))`
- for i ∉ R: `m_i(y) = p̂_{←i}(y) / (|R| p̂(y) + (M−|R|) p̂_{←i}(y))`

**[SOURCE] Defensive pairwise, Eq. 38 (p.10):**
- for i ∈ R: `m_i(y) = 1/M + (1/M) Σ_{j∉R} p̂(y) / (|R| p̂(y) + (M−|R|) p̂_{←j}(y))`
- for i ∉ R: `m_i(y) = ((M−|R|)/M) · p̂_{←i}(y) / (|R| p̂(y) + (M−|R|) p̂_{←i}(y))`

**[SOURCE] §8.3 (p.15).** ReSTIR PT uses defensive pairwise for spatial reuse, plus generalized Talbot for temporal reuse in real time, both with |R| = 1.

**[SOURCE] §6.2 (p.11), confidence weights.** "Reservoir merging generalizes to weighted GRIS, with proper MIS weights, simply by multiplying the p̂ and p̂← in the MIS formulas by the corresponding reservoir's M_r."

**[SOURCE] Enhanced §2.2 (p.3), temporal confidence cap.** The confidence after temporal resampling is `min(c_cap, c_temp) + 1`. Spatial: "M − 1 spatial neighbors are randomly drawn from a disk with a set radius (often 30 pixels). Neighbors whose G-Buffer attributes (e.g., normal, depth) differ too much are rejected."

**[SOURCE] Reference implementation** `DQLin/ReSTIR_PT`, `Source/RenderPasses/ReSTIRPTPass/SpatialReuse.cs.slang`, "PAIRWISE RMIS" block (lines 338–407 after CR→LF normalization, fetched 2026-09-28). With k = `neighborCount`, c_c = central `M`, c_j = neighbor `M`:

```
canonicalWeight = 1
for each valid neighbor j:
    // prefix_approxPdf = p̂_j(T_{c→j}(x_c)) · |J_{c→j}| = p̂_{←j}(x_c)
    canonicalWeight += 1
    if prefix_approxPdf > 0:
        canonicalWeight -= c_j·p̂_{←j}(x_c) / (c_j·p̂_{←j}(x_c) + c_c·p̂_c(x_c)/k)
    // neighbor sample y_j = T_{j→c}(x_j);
    // p̂_j(x_j)/dstJacobian = p̂_{←j}(y_j)
    neighborWeight = c_j·p̂_{←j}(y_j) / (c_j·p̂_{←j}(y_j) + c_c·p̂_c(y_j)/k)
    merge neighbor with w = p̂_c(y_j)·|J|·W_j·neighborWeight
merge canonical with w = p̂_c(x_c)·W_c·canonicalWeight
finalizeGRIS():  W = Σw / p̂(Y)
W /= (validNeighborCount + 1)
```

`PathReservoir.slang` lines ~278–457: p̂ = `toScalar(F)` = dot(F, (0.299, 0.587, 0.114)). `finalizeGRIS` computes `weight/p̂`, and `mergeWithResamplingMIS` uses `w = toScalar(F)·Jacobian·W·misWeight`.

**[INFERENCE] Effective weights and proof of partition of unity.** Let k_v be the number of valid neighbors. Then

- `m_c(y) = [1 + Σ_j (c_c p̂_c(y)/k) / (c_j p̂_{←j}(y) + c_c p̂_c(y)/k)] / (k_v+1)`
- `m_j(y) = [c_j p̂_{←j}(y) / (c_j p̂_{←j}(y) + c_c p̂_c(y)/k)] / (k_v+1)`

Each pair (canonical term j, neighbor j) sums to exactly 1, whatever constant sits inside. So Σ m = (1 + k_v)/(k_v+1) = 1. With c_c = c_j = 1 and k = M−1 this is exactly GRIS Eq. 38 with |R| = 1. This is the formula the property tests in §4 T6 check.

### 2.4 Shift mappings: bijectivity and Jacobians

**[SOURCE] Def. 4.2 (p.6).** A shift mapping T_i is a bijection from a subset D(T_i) ⊂ Ω_i onto its image. §7.1 "Ensuring bijectivity" (p.12) adds:
- "Any successful shift must be invertible: if x̄ shifts to ȳ, an inverse shift must exist to map ȳ back to x̄."
- "Removing paths from a map's domain may cause noise and waste computation, but neglecting bijectivity introduces significant bias."

**[SOURCE] GRIS §8.1 Eq. 52 (p.15), reconnection in solid-angle measure:**
`|∂ω_i^y/∂ω_i^x| = |cos θ_2^y / cos θ_2^x| · ‖x_{i+1} − x_i‖² / ‖x_{i+1} − y_i‖²`
Here θ_2^• is the angle between ω_i^• and the surface normal at x_{i+1} = y_{i+1}.

**[SOURCE] Eq. 53, random replay (solid angle):** `|∂ω^y/∂ω^x| = p_{ω_i^x}(x_{i+1}) / p_{ω_i^y}(y_{i+1})`.

**[SOURCE] Eq. 54, PSS:**
`|∂ū_i^y/∂ū_i^x| = (p_{ω_i^y}(y_{i+1}) / p_{ω_i^x}(x_{i+1})) · |∂ω_i^y/∂ω_i^x|`
"The Jacobian for random replay in the PSS parametrization is always 1". The full-path Jacobian is the product over vertices.

**[SOURCE] Enhanced §2.3 Eq. 2 (p.4), PSS Jacobian of the hybrid shift:**
`|∂T/∂ū| = [p^y_{k−1}(ω'_{k−1}) · G(y_{k−1}→x_k) · p^y_k(ω_k)] / [p^x_{k−1}(ω_{k−1}) · G(x_{k−1}→x_k) · p^x_k(ω_k)]`
- p^x_k(ω_k) ≡ p(ω_k | x_k, −ω_{k−1}) is the solid-angle PDF.
- p^x_k is replaced by 1 when k = d (last vertex).
- G(x→y) = cos θ / ‖x−y‖² is the single-sided geometry term, with θ at the far vertex (here x_k).
- The Jacobian is 1 when no reconnection happens.

**[SOURCE] Enhanced §2.3, choosing the reconnection vertex.** The first pair (x_{k−1}, x_k) with min(α_{x_{k−1}}, α_{x_k}) ≥ α_min and ‖x_k − x_{k−1}‖ ≥ d_min. "Shift invertibility is checked when connecting from y_{k−1} to x_k."

**[SOURCE] Enhanced §4 (p.6–7).** Eq. 4a/4b split |∂T/∂ū| into an area-density ratio and a direction-density ratio. Eq. 5, the dual ray-footprint threshold:
`min((p^x_{k−1}(ω_{k−1}) G(x_{k−1}→x_k))^{-1}, (p^x_k(ω_k) G(x_k→x_{k−1}))^{-1}) ≥ (c/100) · ‖x0−x1‖² / (⟨n_{x1}, x1x0⟩/(4π))`, with c = 0.02.
There is also a single-vertex threshold α_{x_{k−1}} ≥ α_min. Eq. 6a/6b bound both ratios to within ε, giving (1−ε)² < |∂T/∂ū| < (1+ε)².

**Test implications [INFERENCE].**
1. Reconnection criteria (GRIS §7.5 or Enhanced Eq. 5) affect *variance only*, provided the shift stays a bijection. The inverse shift evaluates the criterion on the *offset* path, so it may pick a different vertex; the forward shift must then return "undefined". This asymmetry is the most likely source of hidden bias. §4 T3 tests it exhaustively.
2. Enhanced Eq. 6 predicts |J| ≈ 1 when the thresholds are satisfied. A histogram of log|J| over accepted reconnections is a cheap sanity dashboard: mass should concentrate near 0, and large tails are suspicious.
3. Jacobian reciprocity `|J_{k→j}(x̄)| · |J_{j→k}(T_{k→j}(x̄))| = 1` must hold to float precision (T4).

### 2.5 Convergence versus unbiasedness (and why time averages can mislead)

- **[SOURCE] GRIS §6.4 (p.11–12) and footnote 11.** Without an M-cap, "the relative weights of new samples exponentially approach zero, causing convergence to the wrong result" (Fig. 2). Fig. 10(b) (p.16) shows a Cornell box converging to a wrong image after 10^4 iterations with no M-cap. With an M-cap, the temporal sample's weight is ≈ M_c/(M_c+1). Footnote 11 conjectures a correlation bound `b_k = (M_c/(M_c+1))^k`.
- **[SOURCE] Boxed statement (p.12).** ReSTIR "produces an unbiased, explorative non-Markovian chain … Averaging images of this chain converges in a still scene."
- **[SOURCE] §5.7 (p.10), Fig. 8 (p.16).** Convergence of the *reuse-window* estimate needs |R| to grow faster than √M. Reuse over an ever-larger window with |R| = 1 loses the guarantee.
- **[SOURCE] §6.5 (p.12), offline mode.** "Rendering independent frames with spatial-only GRIS … averaging independently sampled frames converges." Eq. 47: `Ĩ_i = (1/T) Σ_t f_i(Y_i^t) W_{Y_i^t}`. The README of the official code agrees: "disable temporal reuse, use 32 candidate samples per pixel, and set the number of spatial reuse rounds, spatial neighbors, and spatial reuse radius to 3, 6, 10".
- **[SOURCE] Enhanced §7.4 (p.14–16).** "When a noise-free unbiased image is desired, accumulating it with temporal reuse disabled is typically more efficient [Lin et al. 2022]."

**Test implication [INFERENCE].** Per-frame unbiasedness means E over *independent runs* of frame t's estimate equals the true value, even without an M-cap. It does **not** mean the *time average of one run* converges quickly: correlation can stay near 1. So:
- the **ensemble test** (independent runs) is the test for *bias*;
- the **time-average test** (single run, batch means) is a test of *convergence/ergodicity*.

A failing time-average test with a passing ensemble test points at M-cap or correlation, not at a biased estimator.

Under the footnote-11 correlation model ρ_k = ρ^k with ρ = M_c/(M_c+1), the integrated autocorrelation time is
`τ_int = 1 + 2Σ_{k≥1} ρ^k = (1+ρ)/(1−ρ) = 2M_c + 1`,
which is 41 frames for M_c = 20 [INFERENCE; measure it in practice].

### 2.6 Which Enhanced features are unbiased (include in stage B) and which are biased (exclude)

| Feature (Enhanced) | Bias status | Source |
|---|---|---|
| §3 Paired spatial reuse (reciprocal neighbor pairing via reuse textures; flip/mirror/transpose/offset per frame) | Unbiased [INFERENCE: neighbor choice independent of samples; MIS per pair as in §2.3] | p.4–6 |
| §4 Dual footprint + single-vertex roughness thresholds | Unbiased if the shift stays bijective (variance-only change) | p.6–8 |
| §5 Duplication maps → adaptive c_cap = lerp(c_cap^Default, c_cap^min, s^γ) with 17×17 seed-duplication score s = count/288, c_cap^min = 1, γ = 0.1 | **Biased**: "the partition of unity of MIS weights m_i is violated, introducing a small bias"; Kitchen "average absolute relative bias (mean(\|bias\|/ref.)) is 3.25%" | p.8–10, Fig. 5 |
| §6.1 Unified DI + GI in one reservoir (NEE RIS with 32 light-tile candidates, visibility only for selected light) | Unbiased [INFERENCE] | p.10 |
| §6.2.1 Code micro-optimizations; reservoir 88 B → 64 B with "lossy compression to selected quantities" | Lossy compression can bias if it touches W or p̂ inputs [INFERENCE]: test with compression on and off | p.10 |
| §6.2.2 Stream compaction for replay | Unbiased (scheduling only) | p.10–11 |
| §6.2.3 Forced NEE light reconnection during replay | Unbiased if the inverse is consistent [INFERENCE] | p.11 |
| §6.2.4 Russian roulette only in initial sampling ("external roulette … modifying the sampling PDF in PSS"), removed from replay | Unbiased as described; details in their supplemental [UNVERIFIED] | p.11 |
| §6.3 Color-noise reduction: vector weights `w_i = m_i(Y_i) F(Y_i) W_{X_i} \|∂T/∂X_i\|`, shade with Σ_i w_i | Unbiased (marginalizes over the selected index) [INFERENCE consistent with the text] | p.11 |
| §6.4 Dual motion vectors | Unbiased (neighbor choice only) | p.11 |
| RTXDI boiling filter (compared in Fig. 9) | **Biased** ("substantial energy loss") | p.14 |

### 2.7 Metrics used by the papers

- **GRIS §9.1 (p.16–17).** "Variance" on log-log axes versus sample count or time, computed on "grayscale images" (Figs. 7–9, 11).
- **GRIS §9.3 footnote 13 (p.18), MAPE:**
  `MAPE(I, Ĩ_gt) = mean( |I − Ĩ_gt| / (0.01·mean(Ĩ_gt) + Ĩ_gt) )`, for Ĩ_gt a grayscale ground truth. Rationale: "This L1 metric is more resistant to occasional fireflies in sample-reuse algorithms."
- **Enhanced §7 (p.11–12).** HDR-FLIP [Andersson et al. 2021] is the headline metric, at 1920×1080, with c_cap = 20 and 3 spatial neighbors in a 30-px radius.
- **Enhanced Fig. 14 (p.17).** Maps of `((avg of 1024 runs) − ref)/ref` reveal bias.
- **Enhanced Fig. 15 (p.17).** MSE vs iteration and FLIP vs cumulative time: "we run the same camera animation 1024 times with different random seeds for sampling, capture the same view during camera motion, and average the results across runs." FLIP "is less sensitive to fireflies and correlation artifacts, but still sensitive to regional brightness differences."
- **Enhanced Fig. 5 (p.9).** Mean absolute relative bias `mean(|bias|/ref)`.

---

## 3. Statistical methodology

### 3.1 Notation

- Pixel p. Channel: luminance Y = 0.2126R + 0.7152G + 0.0722B for tests, plus per-channel RGB for color bugs. ReSTIR's own p̂ uses Rec.601 weights (0.299, 0.587, 0.114) in the reference code. That has no effect on tests.
- True value μ_p = pixel integral (with box filter).
- **Our estimator.** B independent batches b = 1..B, each producing a batch mean X̄_{p,b} (the mean of n_b spp or frames).
  - Point estimate: `X̄_p = (1/B) Σ_b X̄_{p,b}`.
  - Standard error: `SE_X,p = s_B,p/√B`, with `s_B,p² = (1/(B−1)) Σ_b (X̄_{p,b} − X̄_p)²`.
  - df ν_X = B − 1.
  - Batches must be *independent* (different seeds or independent runs).
- **Cycles reference.** K seeds k = 1..K, each an image R_{p,k} with S spp. `R̄_p = (1/K)ΣR_{p,k}`, `SE_R,p = s_K,p/√K`, ν_R = K−1.
  - Different seeds give independent randomizations of Cycles' tabulated Sobol pattern [INFERENCE]. Keep `scrambling_distance=1.0` and `auto_scrambling_distance=False` so pixels are not correlated.
  - Cycles' `use_sample_subset`/`sample_offset` render disjoint pieces of *one* randomized sequence. Those pieces are not i.i.d. batches, so **do not** use them for variance estimation [INFERENCE].
  - Cycles does not output a per-pixel variance pass (none among `use_pass_*` or `view_layer.cycles` props, [VERIFIED] list in §7.1). Seeds are the only clean route.

### 3.2 Two-sample tests (per pixel, per tile, global)

For any aggregate A (pixel, tile of P pixels, or whole image), compute the aggregate *per batch/seed/run*, then take the mean and SE of those replicate aggregates. Do not combine per-pixel SEs.

Welch statistic:
```
Δ_A = X̄_A − R̄_A
SE_Δ = sqrt(SE_X,A² + SE_R,A²)
t_A  = Δ_A / SE_Δ
ν_A  = (SE_X² + SE_R²)² / (SE_X⁴/ν_X + SE_R⁴/ν_R)      (Welch–Satterthwaite)
```
Under H0 (both unbiased, same integrand), t_A ~ Student-t(ν_A). Use relative forms `Δ_A/R̄_A` for reporting.

**Multiple testing.** For m tiles at family-wise α, use the Šidák per-test level `α' = 1 − (1−α)^{1/m}` with threshold `t* = t_{ν}^{-1}(1 − α'/2)`. Examples at α = 0.01 with large ν:
- m = 1024 tiles → z* ≈ 4.42
- m = 262,144 pixels → z* ≈ 5.5

This matches the Šidák form in Mitsuba's `chi2.py` and pbrt-v4's `bsdfs_test.cpp` ([SOURCE], §4 T8). For visual maps, also use Benjamini–Hochberg FDR q = 0.01 to highlight regions.

**Distribution-level checks.** These catch diffuse small bias that no single tile shows.
1. Reduced chi-square over aggregates: `χ²_red = (1/m) Σ_A t_A²`. Under H0 its expectation is ν/(ν−2), e.g. 1.154 for ν = 15, and its SD ≈ sqrt(2/m) for large ν. Require χ²_red < ν/(ν−2)·(1 + 4·sqrt(2/m)).
2. Mean of the t_A. Under H0 it is ≈ 0 with SE 1/√m, so it detects a global sign bias.
3. KS or Anderson–Darling of {t_A} against t(ν), plus a QQ plot in the report.

### 3.3 Equivalence testing (the actual pass/fail rule)

"No significant difference" is not evidence of correctness: underpowered tests always pass. Declare **equivalence within relative tolerance δ** by TOST:

`pass ⇔ |Δ_A/R̄_A| + t_{ν}^{-1}(1−α)·SE_Δ/R̄_A < δ`

That is, the (1−2α) confidence interval of the relative difference must lie inside [−δ, δ]. Report the **minimum detectable bias** (MDB) of each test: `MDB ≈ (z_{1−α/2}+z_{1−β})·SE_Δ/R̄`.

Suggested δ, to be tuned by the A/A tests in §3.9:

| Comparison | Global δ | Tile (32×32) δ | Why |
|---|---|---|---|
| Our PT vs Cycles | 0.5% | 2% | Model mismatch floor: float precision, ray-offset epsilons, texture filtering, fp32 sRGB decode |
| Our ReSTIR vs our PT (same engine) | 0.2% | 1% | Only estimator differs |
| Per-bounce difference images (§8) | 1% of the full image mean | 3% | Differences of noisy images |

### 3.4 Sample-size planning (how many Cycles spp)

For per-sample coefficient of variation c̄ (per-sample std / mean, pooled over the aggregate) and P independent pixels in the aggregate, the relative SE of the mean with N spp is `c̄/√(P·N)`.

To detect relative bias β with two-sided α = 0.001 (z = 3.29) and 90% power (z = 1.28), when our estimator has far more samples than the reference (GPU is cheap):

`N_ref ≥ ((3.29 + 1.28) · c̄ / β)² / P = (4.57 c̄/β)² / P`

| Aggregate | β | c̄ = 2 (diffuse GI) | c̄ = 5 (glossy/caustic) |
|---|---|---|---|
| Global, 512² (P = 262,144) | 0.5% | 13 spp | 80 spp |
| Tile 32² (P = 1024) | 1% | 816 spp | 5,100 spp |
| Tile 16² (P = 256) | 2% | 816 spp | 5,100 spp |
| Pixel (P = 1) | 10% | 8,350 spp | 52,200 spp |
| Pixel (P = 1) | 5% | 33,400 spp | 209,000 spp |

**[INFERENCE] Caveats.**
- Cycles' QMC often beats this 1/N MC bound, so treat the table as conservative.
- Measure c̄ with a 64-spp pilot per scene.
- For ReSTIR with spatial reuse, the effective P is smaller than P because pixels are correlated. Rely on run-replicates.

Practical default: Cycles K = 16 seeds × S = 4096 spp (65,536 spp total) at 512×512, and our PT at ≥ 262,144 spp (B = 64 batches of 4096). Per-pixel tests then only catch gross (≥ 5–10%) errors. Rely on tile, global and distribution tests for subtle bias.

### 3.5 Convergence diagnostics

**relMSE (common definition, also in the literature):**
`relMSE(I, R) = (1/P) Σ_p (I_p − R_p)² / (R_p² + ε)`, ε = 0.01.
Normalize images so mean(R) = 1 before applying ε; otherwise ε depends on exposure [INFERENCE convention].

**Noise-floor correction.** E[(X̄_N − R̄)²] = σ²/N + b² + Var(R̄). The reference noise floor makes the curve plateau even for an unbiased estimator. Use:
`relMSE_corr(N) = (1/P) Σ_p [(X̄_{N,p} − R̄_p)² − SE_R,p²] / (R̄_p² + ε)`

**Slope test.** Fit log relMSE_corr = a + s·log N over N = 2^k.
- Unbiased MC gives s ≈ −1 (accept s ∈ [−1.10, −0.90] when P is large).
- Bias shows as flattening (s → 0 at large N).
- Cycles' own convergence can be steeper than −1 (RQMC). Don't apply the −1 test to Cycles.

**N·MSE plot (more sensitive than the slope).** `N·MSE_corr(N) = σ² + N·b²`. It is flat for unbiased estimators and rises linearly with slope b² when biased.

**Bias-to-noise ratio.** `BNR(N) = |Δ̄|/SE_Δ` on the global or tile aggregate. Under H0 E[BNR²] ≈ 1 at every N; under bias it grows ∝ √N. Plotting BNR vs N shows emerging bias clearly, and it is what Enhanced Fig. 15 shows qualitatively as flattening curves.

**RMS relative bias** (noise-corrected, one number per scene):
`RMSRB = sqrt(max(0, (1/P) Σ_p [(X̄_p − R̄_p)² − SE_X,p² − SE_R,p²] / R̄_p²))` over pixels with R̄_p above a floor.

**MAPE** (GRIS footnote 13, see §2.7) and **HDR-FLIP** (Enhanced) are reported for comparability with the papers. As pass criteria they are only used relative to the noise floor (§3.9), because both are dominated by residual noise.

**Error ratio (efficiency).** `ER(t) = relMSE_ReSTIR(t) / relMSE_PT(t)` at equal wall time t (or equal frame count). This is a performance metric, not a correctness metric.

### 3.6 Correlated ReSTIR frames: three valid designs

1. **Ensemble / independent runs (preferred; Enhanced Fig. 14/15 methodology).**
   - Run the exact same frame sequence R times with different seeds. Reset reservoirs, and use the same camera and light script.
   - Frame t across runs gives R i.i.d. samples.
   - Per-pixel mean and SE across runs; tile aggregates per run, then across runs.
   - Works for static and animated scenes alike, and captures spatial correlation automatically.
   - Cost: R × t frames.
   - **Ensemble mode in the engine [design recommendation]:** render E independent chains in one dispatch.
     - E × W × H reservoirs; the RNG seed includes the ensemble index.
     - Spatial neighbors are clamped inside the member's viewport; temporal reprojection happens within the member.
     - The per-pixel Welford accumulation (mean, M2) across members runs on the GPU in f32, and each frame's (sum, sumsq) is flushed to f64 on the CPU.
     - Example: E = 64 members at 256×256 ≈ 4.2 M pixels/frame, about 2× the pixel count of a 1080p frame, so 16 dispatch-batches reach R = 1024.
     - Memory: at Enhanced's 64 B/reservoir, 4.2 M × 64 B × 2 (ping-pong) ≈ 537 MB. That is over the WebGPU default `maxStorageBufferBindingSize` (128 MiB) and `maxBufferSize` (256 MiB), so request adapter limits or split buffers [INFERENCE].
2. **Offline spatial-only mode (independent frames).** Temporal reuse is off, so frames are independent given independent seeds (GRIS §6.5). Standard CLT: treat each frame as a batch.
3. **Single long run + batch means (static scenes only).**
   - Accumulate T frames and split them into B ≈ 20–30 contiguous batches of length L ≥ 10·τ_int.
   - `SE = s_batch/√B`, using the t(B−1) distribution.
   - Estimate τ_int per tile from the tile-mean time series, with Sokal's automatic window: the smallest W with W ≥ 5·τ̂_int(W), where `τ̂_int(W) = 1 + 2Σ_{k=1}^{W} ρ̂_k`.
   - With M_c = 20 (τ_int ≈ 41 [INFERENCE]), L ≥ 410 and T ≥ ~10,000 frames for B = 24. Effective sample size ESS = T/τ_int ≈ 244 per pixel for T = 10k.
   - A failure here with an ensemble pass means non-convergence of the chain (GRIS Fig. 2/10b), not bias.

### 3.7 Heavy tails

Path-tracing sample distributions (caustics, glossy-diffuse-glossy) are heavy-tailed. Per-pixel CLT approximations are poor and per-pixel sample variances are noisy.
- Test aggregates (tiles, global), not pixels.
- Use B, K ≥ 16 replicates.
- Optionally report **median-of-means** CIs, which have robust sub-Gaussian concentration.
- **Never** use trimmed or winsorized means or clamped samples in bias tests; they are biased estimators.
- Report the *max single-sample contribution* per scene as a firefly diagnostic.

### 3.8 Temporal sequences (moving light or camera)

- Per frame f: ensemble mean of ReSTIR vs our-PT mean for frame f (and Cycles for keyframes).
- Aggregate across frames: the series {Δ_f/SE_f} should be i.i.d. N(0,1). Regress Δ_f on f to detect drift. The fraction of failing tiles per frame should follow Binomial(m, α').
- Disocclusion regions (camera motion) and regions near moving lights deserve their own tile masks in the report.

### 3.9 Calibrating the harness itself (mandatory)

- **A/A tests.**
  - Cycles seeds 1–8 vs seeds 9–16 through the full pipeline must pass, with χ²_red, the KS test and the tile false-positive rate at their nominal levels.
  - The same for our PT, with two disjoint seed sets.
  - This validates the statistics code and gives the **noise-floor** FLIP, MAPE and relMSE: pass if `metric(ours, cycles) ≤ 1.2 × metric(cyclesA, cyclesB)` at matched sample counts [INFERENCE rule].
- **Planted-bias positive controls.**
  - Scale one light by 1.01. Drop 1% of paths at bounce 2. Use a wrong spot blend (0.15 → 0.16). Omit the Jacobian in the spatial shift (ReSTIR).
  - Confirm each is detected at the claimed MDB. A harness that cannot see a planted 1% error cannot certify 0.5%.

---

## 4. Unit and property tests specific to ReSTIR PT (and its PT substrate)

All GPU tests are compute kernels that `#include` the production WGSL modules (rng, bvh, bsdf, lights, path, shift, reservoir, mis) and write results to storage buffers for readback (see §5). Tolerances assume f32 with Metal fast math on (see §5.6).

**T1. RNG determinism and quality.**
- Counter-based RNG, e.g. PCG-hash of (seed, pixel, frame, bounce, dim).
- Test that identical inputs give bitwise-identical outputs across two dispatches and across the Chrome and dawn.node lanes.
- 1D and 2D χ² uniformity at 10^7 draws, 100 or 32×32 bins, Šidák α = 0.01.
- Cross-pixel and cross-dimension Pearson correlation |r| < 4/√N.
- Frame-to-frame stream independence.
- Recommend a **fixed per-bounce dimension layout** (e.g. dims 0–1 BSDF direction, 2 lobe select, 3 RR, 4–5 light position, 6 light select). Random replay (GRIS §7.2 "Random replay copies the base path's random numbers") then reproduces the base path's decisions regardless of branch history.

**T2. Random-replay determinism.**
- Generate a path from seed s in pixel k, storing hit IDs, barycentrics, lobe indices and the reconnection index.
- Replay with the same seed in the same pixel using the *shift kernel's* replay code: IDs, lobes and indices must be equal, and positions equal within 1e-5 × scene scale.
- Failures here usually come from different code paths (e.g. initial-sampling kernel vs replay kernel) compiled with different fast-math reassociation. See §5.6.
- Compare reconnection vertices by (instance, primitive, barycentrics), not by float positions.

**T3. Shift invertibility (bijectivity).**
- For random (k, j) pixel pairs at distances 1–30 px and random seeds (10^7 trials per scene, on the §8 scenes): compute ȳ = T_{k→j}(x̄). If defined, compute x̄' = T_{j→k}(ȳ).
- Require: the inverse is defined whenever the forward is (rate of violations = 0), and x̄' == x̄ (same IDs, lobe sequence, reconnection index, and replayed random numbers).
- Also require the converse census: `#{forward defined}` vs `#{inverse defined from j}` must satisfy the change-of-variables identity (T5).
- Include the hybrid-shift reconnection criteria (GRIS §7.4–7.5 or Enhanced Eq. 5), forced NEE reconnection (Enhanced §6.2.3), and the temporal shift (previous frame → current frame).

**T4. Jacobian reciprocity.** `|J_{k→j}(x̄)| · |J_{j→k}(ȳ)| = 1` within 1e-4 relative (f32), for every successful round trip in T3. Report a histogram of log|J| (it should concentrate near 0 per Enhanced Eq. 6).

**T5. Jacobian correctness: change-of-variables identities** [INFERENCE, exact math].
For a bijection T: D(T) ⊂ U_k → I(T) = D(T^{-1}) ⊂ U_j between PSS hypercubes, and any bounded h:

`E_{ū∼U}[ h(T(ū)) · |∂T/∂ū| · 1{ū ∈ D(T)} ] = E_{ū'∼U}[ h(ū') · 1{ū' ∈ D(T^{-1})} ]`

- h ≡ 1: E[|J| · 1{fwd ok}] from pixel k must equal Pr[inverse ok] from pixel j.
- h = F_j/(1+F_j), a bounded path-contribution proxy, tests the Jacobian where it matters.
- Run 10^7–10^8 samples per side and use a two-sample z test.
- Finite-difference cross-check for the geometric factor: perturb ω_{k−1} at x_{k−1} over a tiny cone, intersect the surface at x_k, and measure the solid angle the hit patch subtends from y_{k−1}. The ratio of solid angles must match GRIS Eq. 52 within 1e-3.
- The pdf ratios in Enhanced Eq. 2 are validated indirectly by T8's χ² tests: they prove p_k(ω) is the true density of the sampler.

**T6. Resampling-MIS partition of unity and cross-formula consistency.**
- (a) Algebraic check. For random configurations (k ∈ {1..8} neighbors, random c_c and c_j including c_cap-clamped values, some neighbors invalid, some p̂ = 0), evaluate the implementation's m_c(y) and all m_j(y) *for the same y*. Require |Σ m − 1| < 1e-5 (pairwise per §2.3; Talbot for temporal).
- (b) **Consistency between the two places p̂_{←j} is computed** (catches real bugs). For x_j from neighbor j and y = T_{j→c}(x_j):
  - a = p̂_j(x_j)/|J_{j→c}(x_j)| (what the neighbor-weight code uses)
  - b = p̂_j(T_{c→j}(y))·|J_{c→j}(y)| (what the canonical-weight code uses)
  - Require |a−b| ≤ 1e-3·max(a, b).
  - This fails on stale or prior-frame p̂, mismatched Jacobians, a wrong neighbor's primary hit, or asymmetric reconnection criteria.
- (c) For temporal reuse in dynamic scenes, check that the prior-frame p̂ is evaluated with prior-frame scene state (GRIS App. B).

**T7. GRIS toy integrals** (tests the reservoir/MIS WGSL library in isolation).
- Domain [0,1], K "pixels" with integrands `f_i(x) = 1 + a·sin(2π(x+φ_i))`, so ∫f_i = 1. Add variants with partial support (f_i = 0 on sub-intervals) and non-uniform canonical sampling.
- Shifts:
  - translation `T_{i→j}(x) = (x + φ_j − φ_i) mod 1` (|J| = 1)
  - power warp `T(x) = x^γ` (|J| = γ x^{γ−1})
  - maps with undefined regions
- Run initial RIS (M candidates), temporal chains with c_cap, and pairwise spatial reuse with k neighbors and multiple rounds, all using the production WGSL code.
- Assert `E[f_i(Y)W_Y] = 1` (z-test, 10^8 trials) and the **W-histogram identity** `E[W_Y 1{Y∈B}] = |B ∩ supp p̂_i|` for 64 bins.
- Negative controls: constant MIS m = 1/M with non-canonical inputs, and no c_cap. These should reproduce GRIS Fig. 7's non-convergence or bias patterns.

**T8. BSDF sampling and evaluation (Mitsuba 3 / pbrt-v4 methodology).**
- [SOURCE] Mitsuba 3 `chi2.py` (`ChiSquareTest`):
  - `res=101` (odd on purpose), `ires=4` trapezoid sub-steps, `sample_count=1000000`
  - spherical domain mapped to [φ, −cosθ]
  - expected-frequency pooling below 5
  - checks: histogram sum ≤ 1.1, pdf sum ≤ 1.1, pdf ≥ 0, no samples in zero-pdf cells
  - `run(significance_level=0.01, test_count=1)` with Šidák `1−(1−α)^{1/test_count}`
- [SOURCE] pbrt-v4 `src/pbrt/bsdfs_test.cpp`: CHI2_THETA_RES 80, CHI2_PHI_RES 160, CHI2_SAMPLECOUNT 1e6, CHI2_MINFREQ 5, CHI2_SLEVEL 0.01, CHI2_RUNS 5, adaptive-Simpson integration of the pdf per bin, Šidák correction. `TestEnergyConservation` averages f·|cos|/pdf over 16,384 samples and requires < 1.01.
- Our tests, per lobe (Lambert, GGX VNDF), per θ_i ∈ {0°, 30°, 60°, 80°, 89°} and roughness r ∈ {0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0} (α = r²):
  - (a) χ² sampling vs `pdf()`
  - (b) ∫pdf dω ≤ 1 (VNDF loses mass below the horizon)
  - (c) weight consistency: the sampler's returned `f·cos/pdf` equals `eval()·cos/pdf()` within 1e-4
  - (d) directional albedo via E[f cos/pdf] vs an independent quadrature of ∫f cos dω (1e-3), and albedo ≤ 1 + 1e-3
  - (e) reciprocity f(ωi, ωo) = f(ωo, ωi)
  - (f) lobe-selection probabilities sum to 1, and the mixture pdf equals Σ_lobes p_sel·p_lobe
- Match Cycles' GGX exactly. [SOURCE] Cycles `kernel/closure/bsdf_microfacet.h` (main): eval returns `F · D/(4 cos_NI) / (1 + Λ_O + Λ_I)` (height-correlated Smith; the returned value includes the cosine), pdf = `D/(4 cos_NI)/(1+Λ_I)` (VNDF), and α = roughness² [INFERENCE from the Blender node convention].

**T9. Light-sampling pdf consistency.**
Per light type (rect area, emissive triangle mesh; point and spot are delta lights):
- (a) χ² of sampled directions vs `pdf_solid_angle = p_A(y)·‖x−y‖²/|cosθ_y|`.
- (b) E over light samples of `L_e cos θ_x / p_ω` vs the analytic unoccluded irradiance (rect: Lambert polygon formula, §8 C0e).
- (c) **NEE/BSDF MIS partition.** For every path ending on an area or mesh light, w_NEE + w_BSDF = 1 when computed by the NEE-time code and the BSDF-hit-time code (catches light-selection pdf mismatches, e.g. power- or tree-based selection evaluated differently at hit time).
- (d) Three-estimator test: NEE-only, BSDF-only and MIS renders of direct light must agree in expectation.
- (e) Light-selection probabilities sum to 1 at random shading points.

**T10. Integrator furnace tests.**
- Closed enclosure where every surface has emission L_e and diffuse albedo ρ:
  `L = L_e(1 − ρ^{D})/(1 − ρ)` for paths with at most D edges; `L_e/(1−ρ)` without a depth limit [INFERENCE, exact].
- With Cycles bounce limit b, D = b + 2 (see §7.3). Render with b ∈ {0, 1, 3, 7} in our PT, our ReSTIR and Cycles.
- Also a "BSDF furnace": a convex object inside a uniformly emitting sphere of radiance 1, max 0 bounces. Pixel value = directional albedo at the view angle. This compares our BSDF with Cycles' per viewing angle without any light-unit ambiguity.

**T11. Russian roulette and technique-toggle invariance.** Renders with RR on and off, NEE on and off (area lights only), and ReSTIR components on and off (initial-RIS-only; + temporal; + spatial; + both) must agree in expectation (§9 stage B ladder).

**T12. BVH and intersection.**
- 10^6 random rays vs brute force: identical closest primitive ID (ties within 1e-6·t) and identical any-hit results.
- Watertightness: rays from inside closed meshes (sphere, Stanford bunny) must never escape. Use a watertight ray/triangle test (Woop et al. 2013).
- Self-intersection: spawned rays with our offset scheme must not re-hit the origin triangle at grazing angles.

**T13. Camera, jitter and reprojection.**
- Primary-ray jitter is uniform over the pixel (box filter, equal to Cycles `BOX`, width 1.0).
- FOV, aspect and orientation registration against Cycles (§8 C0a).
- Motion-vector reprojection of world points matches analytic projection with the previous camera within 0.01 px.

**T14. Paired spatial reuse textures (Enhanced §3).**
- Involution: pair(pair(p)) = p for all p after each per-frame flip, mirror, transpose or offset.
- Offset distribution: mean ≈ 0, std ≈ σ, KS against N(0, σ²) per axis. Target σ = sqrt(8/(9π))·r, i.e. σ = 16.0 for r = 30 (Enhanced §7, p.12).
- n_σ from Eq. 3: `n_σ = ⌊σ²/2 + 1.46σ^{-1} + 1.76σ^{-2} + 0.656σ^{-3} + 0.5⌋`.

**T15. NaN/Inf guards.** WGSL has no isNan builtin, and Metal compiles with fast math by default in Chrome (see §5.6), so `x != x` may be folded away.
- Implement guards on bits: `(bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u`.
- Assert the count of non-finite outputs is 0 in every image test.
- Assert no negative radiance.

**T16. Config assertions.** The "unbiased preset" must have:
- duplication-map c_cap adaptation off, no clamping, no boiling filter
- the same max path length as the PT reference
- compression of reservoir fields that feed W, p̂ or the Jacobians disabled or proven lossless

The harness stores the full config hash in each result's `meta.json`.

---

## 5. Automation on macOS

### 5.1 Options assessed

| Option | Facts | Verdict |
|---|---|---|
| **Playwright + installed Chrome 154** | Playwright 1.63 current. Playwright docs: branded Chrome/Edge "switched to a new headless mode implementation that is closer to a regular headed mode". The `chromium` channel opts into new headless; the default Playwright headless uses the separate `chromium-headless-shell`. Playwright's default switches include no GPU/ANGLE/SwiftShader/WebGPU flags (from `chromiumSwitches.ts`: `--disable-background-timer-throttling`, `--disable-renderer-backgrounding`, `--force-color-profile=srgb`, …). WebGPU needs a secure context: `http://localhost` qualifies, `about:blank` does not. | **Primary.** Same browser, Dawn and Tint as the product. [UNVERIFIED on this Mac: headless WebGPU adapter availability. Run the smoke test first.] |
| Puppeteer 25.12 | Similar. Reported to inject `--use-angle=swiftshader-webgl` on some configs, which needs `ignoreDefaultArgs` (third-party blog). Chrome's own guide covers only Linux flags. | Equivalent alternative; no reason to prefer it. |
| **Node 26 + `webgpu@0.6.1` (dawn.node)** | darwin-universal prebuilt; compute and render-to-texture plus readback; no DOM, canvas or image decoding. The README itself suggests Puppeteer for testing web pages. The Dawn/Tint revision differs from Chrome 154's. | **Fallback + fast lane** for shader unit tests and batch rendering. [UNVERIFIED: loads under Node 26.] |
| Deno | Not installed. WebGPU is still behind `--unstable-webgpu` (Deno docs). Built on wgpu with the **naga** WGSL frontend, not Tint, so validation and codegen differ from Chrome. | Not recommended. |

### 5.2 Architecture to share the same WGSL everywhere

- `src/core/**` (TS, **no DOM**): `Renderer` takes an injected `GPUDevice` and pre-decoded scene data (typed arrays + RGBA8/RGBA32F texture arrays).
- WGSL sources are imported as strings. `import pt from './pt.wgsl?raw'` works under Vite, Vitest and Vitest browser mode. For plain Node scripts use `vite-node`/`tsx` with a raw loader, or pre-bundle.
- A tiny runtime WGSL composer (`#include`, `#define` constants) implemented in TS, identical in browser and Node. No build-time-only transforms.
- `src/app/**` (DOM): canvas presentation, WASD/mouse, file loading (GLB via `@gltf-transform/core`, which works in Node and browser; USD per the USD research), `createImageBitmap` texture decode.
- `validation/harness/harness.html` + `harness.ts`: loads `src/core` and exposes `window.__harness` (`loadScene(url)`, `configure(json)`, `renderBatches({mode, sppPerBatch, batches, ensemble, frames, seed})`, `upload(name, Float32Array)`).
- `validation/gpu-tests/*.wgsl`: test kernels that `#include` production modules. `*.gpu.test.ts`: Vitest tests with a device factory. Two Vitest projects:
  - `node-dawn`: environment node, device from `webgpu`
  - `chrome`: `@vitest/browser` + `@vitest/browser-playwright` provider, launch options `{channel:'chrome', args:['--enable-unsafe-webgpu']}`

  [UNVERIFIED: exact Vitest 5 browser-provider config keys; check the Vitest 5 docs.]

### 5.3 Primary lane: Playwright driving installed Chrome (sketch; untested)

```ts
// validation/harness/run-chrome.ts   (run with: npx tsx validation/harness/run-chrome.ts <scene> <config>)
import { chromium } from 'playwright';
const browser = await chromium.launch({
  channel: 'chrome',              // /Applications/Google Chrome.app (154)
  headless: true,                 // branded Chrome => new headless
  args: [
    '--enable-unsafe-webgpu',
    // optional, dev-only: enables GPUShaderModuleDescriptor.strictMath (Metal/D3D) — see 5.6
    '--enable-webgpu-developer-features',   // [UNVERIFIED switch name; chrome://flags/#enable-webgpu-developer-features]
  ],
});
const page = await browser.newPage();
page.on('console', m => console.log('[page]', m.type(), m.text()));
page.on('pageerror', e => { console.error(e); process.exitCode = 1; });
await page.goto('http://localhost:5173/validation/harness/harness.html');   // secure context
const adapterInfo = await page.evaluate(async () => {
  const a = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  return a ? { vendor: a.info.vendor, arch: a.info.architecture, desc: a.info.description,
               features: [...a.features], maxBuf: a.limits.maxBufferSize,
               maxSSBO: a.limits.maxStorageBufferBindingSize } : null;
});
if (!adapterInfo) throw new Error('No WebGPU adapter in headless Chrome -> use headed fallback');
await page.evaluate(cfg => (window as any).__harness.run(cfg), config); // page POSTs batches itself
await browser.close();
```

**Data egress, recommended.** Add a Vite dev-server middleware that streams raw POST bodies to disk. There is no size limit, no JSON or base64 overhead, and it is independent of CDP:

```ts
// vite.config.ts (plugin excerpt)
configureServer(server) {
  server.middlewares.use('/__harness/upload', (req, res) => {
    const u = new URL(req.url ?? '/', 'http://localhost');
    const name = path.basename(u.searchParams.get('name') ?? 'out.bin'); // sanitize!
    const ws = fs.createWriteStream(path.join(OUT_DIR, name));
    req.pipe(ws);
    ws.on('finish', () => { res.statusCode = 204; res.end(); });
  });
}
```

The page side: `await fetch('/__harness/upload?name=pt_b012.f32', {method:'POST', body: f32.buffer})`, plus a small JSON `meta` post. The Node runner converts `.f32` + meta into PFM or EXR, or the page writes PFM bytes directly (§6.1).

Alternatives:
- `page.route(...)` + `request.postDataBuffer()`. Large binary bodies through CDP interception are [UNVERIFIED] reliable.
- Download events: Blob plus `<a download>`, then `page.waitForEvent('download')` and `download.saveAs()`.
- Base64 chunks through `page.evaluate`: about 1.33× size and slow, but fine for ≤ 16 MB.

**Driving frames.** Do not depend on `requestAnimationFrame` in headless. The harness loops `render → submit → await device.queue.onSubmittedWorkDone()`. Readback uses `copyBufferToBuffer` into a `MAP_READ` buffer, then `mapAsync(GPUMapMode.READ)`, `getMappedRange().slice(0)` and `unmap`. (Texture→buffer copies need `bytesPerRow` to be a multiple of 256; storage-buffer accumulators avoid this.)

**Smoke test to run first.** Launch as above and print `adapterInfo`. If it is null, retry with `headless:false` (Fallback 1). If that also fails, use the dawn.node lane.

### 5.4 Fallback lane: Node + dawn.node (sketch; untested)

```ts
import { create, globals } from 'webgpu';        // ESM package
Object.assign(globalThis, globals);             // GPUBufferUsage, GPUMapMode, ...
const gpu = create(['backend=metal']);           // other options: 'adapter=<name>', 'enable-dawn-features=...'
const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
const device = await adapter!.requestDevice({
  requiredLimits: { maxStorageBufferBindingSize: adapter!.limits.maxStorageBufferBindingSize,
                    maxBufferSize: adapter!.limits.maxBufferSize },
});
// ... const r = new Renderer(device, sceneData); await r.renderBatches(...)
// Lifetime caveat (README): drop references to the object returned by create() or node won't exit.
```

Texture decoding in Node: pre-decode textures into a raw sidecar in the scene-bridge export. The same scene package then loads without image codecs in Node, and the browser uses the same raw path in harness mode.

### 5.5 Running Blender headless from the harness

`/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup -noaudio --python validation/blender/render_reference.py -- <scene.json> <config.json> <outdir>`

Blender runs outside the browser, so the Node orchestrator spawns it. References are cached by `sha256(scene package + reference config + Blender version string)`.

### 5.6 GPU/driver concerns that affect validation

- **Metal fast math.** [SOURCE] Chrome blog "What's New in WebGPU (Chrome 131)":
  - a developer-only `strictMath` boolean on `GPUShaderModuleDescriptor`, "currently supported on Metal and Direct3D"
  - behind `chrome://flags/#enable-webgpu-developer-features`
  - when disabled, the compiler "may optimize by ignoring NaN/Infinity possibilities, treating -0 as +0, replacing division with reciprocal multiplication, and rearranging operations"

  Consequences:
  - (a) NaN guards need bit tests (T15).
  - (b) Replay determinism across pipelines is not guaranteed bitwise (T2), so compare discrete IDs.
  - (c) A/B-run the unit tests with `strictMath: true` in the harness to separate precision issues from logic bugs.
- **Command-buffer duration.** Keep each `queue.submit` short, at most tens of ms: split accumulation into many submits. Long Metal command buffers risk GPU timeouts and device loss [INFERENCE]. `--disable-gpu-watchdog` exists as a Chrome switch, but its effect on Metal is [UNVERIFIED].
- **Accumulation precision.** Accumulate each batch (≤ 4096 spp) in f32 on the GPU, as a per-pixel sum plus sum of squares if wanted. Read back per batch and accumulate across batches in `Float64Array` on the CPU. Batch means then come for free.
- **Limits.** Request `maxStorageBufferBindingSize` and `maxBufferSize` from the adapter for ensemble mode (§3.6).
- **Determinism.** Avoid float atomics (WGSL has none; CAS loops are order-dependent). Otherwise two runs with the same seed can differ, which breaks T1-style whole-frame determinism tests.

---

## 6. Image IO and the comparison tool

### 6.1 Writing float images from JS/TS

**PFM (recommended for the harness).**
- Header: ASCII `"PF\n{W} {H}\n-1.0\n"` (negative scale = little-endian).
- Body: float32 RGB rows ordered **bottom-to-top**.
- 15 lines of TS. OIIO reads it ([VERIFIED] OIIO's `pnm` plugin lists `pfm`), numpy can read it with `np.fromfile`, and most HDR viewers open it.

**Minimal OpenEXR writer** (for a "download EXR" button in the app and for EXR artifacts), uncompressed scanline:
1. Magic `76 2f 31 01`.
2. Version bytes `02 00 00 00`.
3. Header attributes, each `name\0 type\0 int32 size value`:
   - `channels` (chlist, **sorted names** `B`,`G`,`R`: name\0, int32 pixel_type 2 = FLOAT, uint8 pLinear 0, 3 reserved bytes, int32 xSampling 1, int32 ySampling 1; list terminated by \0)
   - `compression` (compression, 1 byte 0 = NONE)
   - `dataWindow` and `displayWindow` (box2i: xMin, yMin, xMax, yMax)
   - `lineOrder` (1 byte 0 = INCREASING_Y)
   - `pixelAspectRatio` (float 1)
   - `screenWindowCenter` (v2f 0,0)
   - `screenWindowWidth` (float 1)
   - terminated by `\0`
4. Offset table: H × uint64 absolute offsets (one scanline per block).
5. Blocks: int32 y, int32 byteCount, then per channel in sorted order all W float32 values of that row.

[INFERENCE from the OpenEXR file-layout spec. Validate by reading back with OIIO in CI.]

`three@0.186` also ships an `EXRExporter` (examples/jsm) with ZIP compression via fflate; its current signature is [UNVERIFIED].

### 6.2 Reading EXRs

- **Blender Python (zero install, [VERIFIED] modules):** `/Applications/Blender.app/Contents/Resources/5.1/python/bin/python3.13 -c "import OpenImageIO as oiio; a = oiio.ImageBuf('x.exr').get_pixels(oiio.FLOAT)"` gives an H×W×C numpy array, top row first. Good for conversions and quick checks. **Do not pip-install into the app bundle.**
- **Analysis venv (recommended):**
  ```
  uv venv --python 3.13 validation/.venv
  uv pip install --python validation/.venv numpy scipy matplotlib openimageio flip-evaluator
  ```
  - All five have cp313 macOS-arm64 wheels ([VERIFIED] on PyPI). flip-evaluator 1.7 has **no cp314 wheel**, so the system Python 3.14 is unsuitable.
  - uv already has CPython 3.13.12 installed.
  - A `python3.13 -m venv --system-site-packages` created from Blender's interpreter to reuse its numpy and OIIO is possible (venv + ensurepip import fine) but [UNVERIFIED] for relocatability. Prefer uv.
- **Node:** `parse-exr` 1.0.2 (a port of three's EXRLoader with no three dependency) if TS-side reading is ever needed, e.g. to show Cycles references in the web app.

### 6.3 FLIP

- **Python.** [SOURCE] NVlabs README/api_example: `import flip_evaluator as flip; flipErrorMap, meanFLIPError, parameters = flip.evaluate(ref, test, "HDR")`. It accepts `.exr`/`.png` paths or numpy arrays, and `flip.load(path)` exists. Parameter keys: `ppd`, `startExposure`, `stopExposure`, `numExposures`, `tonemapper`.
- **CLI:** `flip -r reference.exr -t test.exr`. The C++ tool is buildable with the local cmake if needed.
- **JS port:** none found. npm names checked: `flip-js` is an unrelated package, and `@nvlabs/flip` and `nv-flip` do not exist. The NVlabs repo offers Python, PyTorch, C++ and CUDA only. A WGSL port for in-app display is feasible later but must be validated against `flip-evaluator`.
- **Usage rule.** HDR-FLIP is a perceptual *report* metric. Pass/fail uses FLIP only relative to the A/A noise floor (§3.9), because FLIP of two unconverged images mostly measures noise.

### 6.4 `validation/tools/compare.py` design

**Inputs:**
- `--ours DIR`: batch images `batch_###.pfm` plus `meta.json` (seed, spp/batch, config hash)
- `--ref DIR`: Cycles `seed_###.exr` plus `manifest.json`, or another "ours" directory for stage B
- `--test test.json`: thresholds, tile size, masks, channel set

**Pipeline:**
1. Load everything as float64 and check dimensions and orientation (the registration test must have passed).
2. Per replicate, compute luminance (and RGB), tile means (16² and 32²) and the global mean.
3. Means, SEs, Welch t and ν per pixel, tile and global.
4. Šidák thresholds, BH-FDR masks, χ²_red, mean t, KS/AD vs t(ν).
5. TOST equivalence at global and tile level; MDB report.
6. relMSE (raw and noise-corrected), MAPE (GRIS footnote 13), RMSRB, HDR-FLIP of converged means, and noise-floor versions of each (A/A).
7. Convergence mode (`--curve`): prefix-mean images at N = 2^k batches, then relMSE_corr(N), N·MSE, BNR(N), slope fits.

**Outputs:**
- `report.json`: every metric, verdicts per gate, config hashes, Blender and Chrome versions
- PNGs:
  - side-by-side tonemapped images (identical exposure)
  - signed relative difference with a diverging colormap clipped at ±δ
  - |t| map clipped at 6 with Šidák contour
  - tile t heatmap
  - FLIP map
  - z histogram vs t(ν) and QQ plot
  - convergence plots
- `index.html`: a static report linking all of the above

A later step can publish it as a private claude.ai Artifact for the team if wanted. [UNVERIFIED as a workflow; nothing was published in this session.]

Exit code: non-zero if any gate fails, so the orchestrator can run it in CI.

---

## 7. Cycles reference generation (Blender 5.1.2)

### 7.1 Required overrides (defaults found by headless introspection vs required values)

| Setting | Blender 5.1.2 default ([VERIFIED]) | Required | Reason |
|---|---|---|---|
| `scene.render.engine` | BLENDER_EEVEE | CYCLES | |
| Device | cycles.device CPU | prefs `compute_device_type='METAL'`, `get_devices()`, enable METAL device; `cycles.device='GPU'` | Speed. CPU also valid for spot checks |
| `cycles.samples` | 4096 | per test | |
| `use_adaptive_sampling` | **True** (threshold 0.01) | **False** | Keep fixed spp and clean replicate statistics. The stopping rule depends on samples [INFERENCE: tiny bias] |
| `use_denoising` (scene and view layer) | **True** (OIDN) | **False** (both) | Biased |
| `sample_clamp_direct` / `_indirect` | 0 / **10.0** | 0 / 0 | Clamping is biased |
| `blur_glossy` (Filter Glossy) | **1.0** | 0 | Biased |
| `light_sampling_threshold` | **0.01** | 0 | Probabilistic light-sample termination (RR-like, unbiased in principle) adds variance; 0 disables |
| `caustics_reflective/refractive` | True | True | Disabling would bias |
| `use_guiding` | False | False | Keep simple (CPU only anyway) |
| `max_bounces`, `diffuse_bounces`, `glossy_bounces`, `transmission_bounces` | 12 / 4 / 4 / 12 | all = b (test-specified) | So only the total count limits paths (§7.3) |
| `volume_bounces`, `transparent_max_bounces` | 0 / 8 | 0 / 8 (no alpha in early tests) | |
| `pixel_filter_type`, `filter_width` | **BLACKMAN_HARRIS, 1.5** | **BOX, 1.0** | Match uniform in-pixel jitter (hard_min 0.01 [VERIFIED]) |
| `sampling_pattern` | TABULATED_SOBOL (options: AUTOMATIC, TABULATED_SOBOL, BLUE_NOISE [VERIFIED]) | TABULATED_SOBOL | Blue noise deliberately correlates neighboring pixels |
| `scrambling_distance`, `auto_scrambling_distance` | 1.0, False | 1.0, False | Pixel independence |
| `seed`, `use_animated_seed` | 0, False | seed = replicate id; for animations use a unique seed per (frame, replicate) | |
| `film_exposure` | 1.0 | 1.0 | |
| `render.film_transparent` | False | False | |
| View transform | AgX | Standard, look None, exposure 0, gamma 1 | EXR is scene-linear; set anyway to be safe |
| Working color space (`bpy.data.colorspace.working_space`) | Linear Rec.709 (options Rec.709, Rec.2020, ACEScg) | Linear Rec.709 | Match our linear sRGB/Rec.709 primaries |
| Output | — | `image_settings.file_format='OPEN_EXR'`, `color_depth='32'`, `exr_codec='ZIP'` (lossless), `color_mode='RGB'` | Default EXR settings seen: RGBA, 32, ZIP, `color_management='FOLLOW_SCENE'` |
| World | **gray (0.0509) Background, strength 1** in the factory scene | black (strength 0), or start from `read_factory_settings(use_empty=True)` and create a world with strength 0 | Otherwise the environment adds light |
| Factory scene objects | Cube, Light, Camera | delete / start empty | |
| Camera | `sensor_fit='AUTO'`, sensor 36×24, lens 50, `clip_start=0.1`, DOF off | `sensor_fit='VERTICAL'` + `angle_y` = our fovY (or FIT per our convention); `clip_start` = ours (e.g. 1e-4); DOF off; shift 0 | |
| Motion blur | `render.use_motion_blur` False | False | |
| Object `shadow_terminator_geometry_offset` | **0.1** | 0 | Alters shadow-ray origins on smooth-shaded meshes |
| Object `shadow_terminator_shading_offset`, `_normal_offset` | 0, 0 | 0 | |
| Light `use_soft_falloff` (point/spot) | **True** | False | Non-physical near-field falloff (only matters with radius > 0) |
| Light `shadow_soft_size` (radius) | 0.0 for `bpy.data.lights.new` | 0 for delta lights | |
| Light `normalize` | True | True (see §7.2) | |
| Light `exposure` | 0 | 0 | |
| Area `spread` | π | π (Lambertian) | |
| Glossy BSDF `distribution` | **MULTI_GGX** | **GGX** | Our single-scatter GGX |
| Principled `distribution` | **MULTI_GGX** | avoid Principled in exact tests | Layering and energy compensation |
| Diffuse BSDF `Roughness` | 0.0 | 0.0 | Lambert (Oren–Nayar if > 0) |
| `render.use_persistent_data` | False | True | Reuse the BVH across seed renders |

### 7.2 Light and emission units (from Cycles source, main branch)

[SOURCE: `intern/cycles/blender/light.cpp`]
`strength = light_color * energy * exp2f(exposure)`, `normalize = !(mode & LA_UNNORMALIZED)`, spot `angle = spot_size`, `smooth = spot_blend`, point radius = `radius`, `is_sphere = !(mode & LA_USE_SOFT_FALLOFF)`.

[SOURCE: `intern/cycles/scene/light.cpp`]
- `PointLight::area() = 4π r²`, returning **4** when r = 0.
- `PointLight::copy_to_kernel`: `invarea = normalize ? 1/area : 1`, `eval_fac = invarea · (1/π)`.
- Spot: `cos_half_spot_angle = cos(angle/2)`, `spot_smooth = 1/((1 − cos_half_spot_angle)·smooth)`.
- Area: `invarea = 1/area` (when normalize).

[SOURCE: `kernel/light/spot.h`]
- Attenuation = `smoothstepf((ray.z − cos_half_spot_angle) · spot_smooth)`, with ray.z the cosine to the spot axis in light space.
- For r = 0, the kernel-side `invarea = 1`.

[SOURCE: `kernel/light/area.h`]
- `eval_fac = M_1_PI_F · invarea`, times a spread attenuation.
- One-sided: `if (dot(ray->D, Ng) >= 0) return false`.

Derived conventions [INFERENCE; verify with the calibration scenes C0b–C0e in §8]:
- **Point light, radius 0:** eval_fac = (1/4)(1/π), so radiant intensity I = P/(4π) W/sr. Irradiance at distance d, incidence θ: `E = P cosθ / (4π d²)`.
- **Spot light:** `I(α) = P/(4π) · S(α)`, with `S(α) = smoothstep((cos α − cos(s/2)) / ((1 − cos(s/2))·β))` and `smoothstep(t) = 0 (t ≤ 0), 1 (t ≥ 1), t²(3 − 2t)`. Here s = spot_size, β = spot_blend, and α is the angle to the axis (Blender lights point along local −Z [UNVERIFIED direction]). No extra cosine factor, and the cone does not concentrate power (the energy description says it is the power "if it wasn't limited by the spot angle").
- **Area light, normalize = True, spread = π:** `L_e = P / (π · A)`, one-sided. Whether spread = π makes the attenuation exactly 1 is [UNVERIFIED].
- **Emission shader on meshes:** radiance `L_e = Strength · Color` [INFERENCE: the standard Cycles convention; checked by C0b]. Mesh emitters emit from both sides unless explicitly masked [UNVERIFIED].

### 7.3 Bounce semantics

[SOURCE: `kernel/integrator/path_state.h` `path_state_next`]
After each surface BSDF sample the bounce counter increments. When `bounce >= kernel max_bounce`, the flag `PATH_RAY_TERMINATE_AFTER_TRANSPARENT` is set. [SOURCE: `shade_surface.h` `integrate_surface_terminate`] The next surface hit then *adds its emission (MIS-weighted) and terminates* before NEE, and RR divides throughput by the continuation probability.

The Blender docs say "0 bounces = direct lighting only". So [INFERENCE] kernel `max_bounce = b + 1` (probably set in `scene/integrator.cpp`; [UNVERIFIED +1]):
- NEE happens at x1…x_{b+1};
- BSDF continuation from x_{b+1} only collects emission.

Total path length in edges is D = b + 2, matching the Enhanced notation x̄ = [x0 … x_d] with d = b + 2.

**Our PT and ReSTIR must use the same truncation**: at the last scattering vertex, do NEE plus BSDF-sampled emission with MIS, and no further NEE. The furnace test T10 / C0f verifies the mapping numerically: `L = L_e(1−ρ^{b+2})/(1−ρ)`.

### 7.4 Material model for exact equivalence

Use only:
- Diffuse BSDF (roughness 0)
- Glossy BSDF with `distribution='GGX'` (height-correlated Smith, VNDF, α = r²; Fresnel = color tint [UNVERIFIED: the 5.x Glossy node has no Fresnel term; check with the BSDF furnace])
- Mix Shader with constant or texture-driven factor (an exact linear lobe mixture)
- Emission, combined via Add Shader

Our "validation material" is `f = (1−m)·ρ_d/π + m·GGX(α, F = c_s)`. If the production material is richer (glTF metallic-roughness with Schlick Fresnel), keep a **validation-compatible mode**, or accept a looser tolerance for those scenes and mark them "model-approximate". Principled v2 (dielectric layering with albedo scaling, F82 metals, `energy_scale`) is not reproduced.

Textures: Cycles image nodes default to `Linear` interpolation with no mipmapping [INFERENCE: Cycles does not use ray-differential mip filtering for image textures by default]. Our validation mode must sample **LOD 0 bilinear** (`textureSampleLevel(t, s, uv, 0)`) with matching wrap modes. sRGB decode must happen before filtering in both (8-bit sRGB → linear).

### 7.5 Scene bridge (strongly recommended)

The web app exports its parsed scene as `scene.json` + `.bin`:
- world-space or instanced triangles with **explicit per-vertex normals** and UVs
- material parameters of the validation model
- raw-decoded textures
- lights in Blender units (W, spot size and blend, rect size)
- camera (position, orientation, fovY, aspect, clip)
- per-frame transforms for animations, recorded from the app's scripted or WASD-recorded path

`validation/blender/build_scene.py`:
- starts from `bpy.ops.wm.read_factory_settings(use_empty=True)`
- creates meshes with custom split normals and flat shading when normals are faceted
- sets node materials per §7.4
- sets lights with `use_soft_falloff=False`
- sets the camera with `sensor_fit='VERTICAL'`, `angle_y`
- applies our Y-up → Blender Z-up conversion exactly once (glTF is Y-up; Blender is Z-up)

For animations, set `obj.matrix_world` directly in the per-frame render loop. Do not rely on keyframe interpolation (Bezier by default).

Importer fidelity is tested separately (§8 vii/viii):
- our GLB loader vs Blender's glTF importer, via geometry dumps (world-space vertex positions, normals and transforms, compared with tolerances)
- our USD loader vs `pxr.Usd` inside Blender's Python ([VERIFIED] pxr 0.25.8): `UsdGeom.Mesh` points and `faceVertexIndices`, `ComputeLocalToWorldTransform`, `UsdLux` Rect/Sphere/Shaping parameters, `UsdPreviewSurface` inputs

Light-unit conversion importers:
- Blender's `usd_import` has `light_intensity_scale=1.0`, `apply_unit_conversion_scale=True` ([VERIFIED] option names).
- Blender's glTF importer has `export_import_convert_lighting_mode='SPEC'` (physical, lm/W).

Our loaders must document and test their own conversions against these.

### 7.6 Render loop sketch (untested)

```python
# validation/blender/render_reference.py  (run: Blender -b --factory-startup -noaudio --python this.py -- scene.json ref.json outdir)
import bpy, sys, json, os
argv = sys.argv[sys.argv.index('--') + 1:]
scene_path, cfg_path, out = argv
cfg = json.load(open(cfg_path))
bpy.ops.wm.read_factory_settings(use_empty=True)
import build_scene; build_scene.build(scene_path)          # bridge (7.5)
sc = bpy.context.scene; sc.render.engine = 'CYCLES'
p = bpy.context.preferences.addons['cycles'].preferences
p.compute_device_type = 'METAL'; p.get_devices()
for d in p.devices: d.use = (d.type == 'METAL')
c = sc.cycles; c.device = 'GPU'
c.samples = cfg['spp']; c.use_adaptive_sampling = False; c.use_denoising = False
sc.view_layers[0].cycles.use_denoising = False
c.sample_clamp_direct = 0; c.sample_clamp_indirect = 0; c.blur_glossy = 0; c.light_sampling_threshold = 0
b = cfg['max_bounces']
c.max_bounces = c.diffuse_bounces = c.glossy_bounces = c.transmission_bounces = b
c.pixel_filter_type = 'BOX'; c.filter_width = 1.0
c.sampling_pattern = 'TABULATED_SOBOL'; c.scrambling_distance = 1.0; c.auto_scrambling_distance = False
sc.render.use_persistent_data = True
sc.view_settings.view_transform = 'Standard'
im = sc.render.image_settings; im.file_format = 'OPEN_EXR'; im.color_depth = '32'; im.exr_codec = 'ZIP'; im.color_mode = 'RGB'
sc.render.resolution_x, sc.render.resolution_y = cfg['res']; sc.render.resolution_percentage = 100
for frame in cfg.get('frames', [None]):
    if frame is not None: build_scene.apply_frame(frame)     # set matrix_world for lights/camera
    for s in cfg['seeds']:
        c.seed = s if frame is None else hash((frame, s)) & 0x7fffffff
        sc.render.filepath = os.path.join(out, f"f{frame or 0:04d}_seed{s:03d}.exr")
        bpy.ops.render.render(write_still=True)
json.dump({'blender': bpy.app.version_string, 'settings': cfg}, open(os.path.join(out, 'manifest.json'), 'w'))
```

Render time per image on the M5 Pro Metal backend is [UNVERIFIED]. Benchmark one Cornell box at 512², 4096 spp, and size the seed count K from the §3.4 table.

---

## 8. Test scene suite (incremental; each isolates features)

Defaults unless stated:
- 512×512
- Cycles K = 16 seeds × 4096 spp
- our PT ≥ 64 batches × 4096 spp
- ReSTIR ensemble R ≥ 256 (1024 for the final sign-off)
- tiles 32²
- gates per §3.3 and §9

Bounce sweep b ∈ {0, 1, 3, 7}, with per-bounce difference images D_b = I_b − I_{b−1} for localization.

### Calibration scenes (analytic plus Cycles; run first)

| ID | Scene | Validates | Pass criteria |
|---|---|---|---|
| C0a | Black background; small emissive squares at 4 corners and the center, at known world positions | Camera FOV/aspect, orientation (y-flip), handedness, Y-up↔Z-up conversion, pixel-center convention | Centroid of each square within 0.1 px of Cycles; no flips |
| C0b | Camera facing a large emissive quad, Strength·Color = (1, 0.5, 0.25) | Emission units, color pipeline, EXR linearity | Pixel = L_e within 1e-4 (both engines) |
| C0c | Diffuse plane (ρ = 0.5), point light P = 100 W at height h = 1, b = 0 | Point-light units and falloff | `L_o(x) = ρ P h / (4π² d³)` (d = distance light–x) within 0.2% per tile, in both engines |
| C0d | Same with a spot light (s = 60°, β ∈ {0, 0.15, 0.5}), axis tilted 20° | Spot profile | `L_o = (ρ/π)·(P/4π)·S(α)·cosθ/d²` (§7.2) within 0.5% per tile away from the hard edge; edge band compared statistically only |
| C0e | Diffuse plane under a one-sided rect light a×b, power P | Area-light units, one-sidedness, orientation | `L_o = (ρ/π)·E`, `E = (L_e/2) Σ_i acos(û_i·û_{i+1}) · n·normalize(û_i × û_{i+1})` with `û_i = normalize(v_i − x)` and consistent winding (light fully above the horizon), `L_e = P/(π a b)`; within 0.3% per tile |
| C0f | Closed enclosure (cube or sphere), all surfaces emissive L_e and diffuse ρ = 0.5; b ∈ {0, 1, 3, 7} | Bounce semantics, emission/NEE/MIS double counting | Pixel = `L_e(1 − ρ^{b+2})/(1−ρ)` within 0.1% global; also for ReSTIR |
| C0g | BSDF furnace: sphere of GGX roughness r ∈ {0.05, 0.2, 0.5, 1.0} (and Lambert) inside a uniformly emissive enclosing sphere (L = 1), b = 0 | Per-angle directional albedo equality with Cycles | Tile-wise equivalence δ = 1%; Lambert pixel = ρ within 0.1% |

### Validation scenes

| ID | Scene | What it validates | Stage A pass (PT vs Cycles) | Stage B pass (ReSTIR vs PT) |
|---|---|---|---|---|
| (i) | Diffuse Cornell box (Lambert walls, colored), single one-sided rect area light near the ceiling; b ∈ {0, 1, 3, 7} | Core integrator, NEE+MIS, area-light sampling, reconnection shift on rough surfaces (GRIS §9.1 uses this family) | Global δ = 0.5%, tile δ = 2%, χ²_red within bound, no Šidák tile failures, relMSE/MAPE/FLIP ≤ 1.2× A/A floor | Global δ = 0.2%, tile δ = 1% for: initial-RIS-only, +spatial (offline mode), +temporal (static, ensemble), full. Time-average test with M_c = 20 must also pass |
| (ii) | Cornell box lit only by a point light (radius 0) | Delta lights: NEE-only paths, no BSDF-hit MIS, reconnection to a fixed light vertex, forced NEE reconnection (Enhanced §6.2.3) | Same as (i) | Same as (i) + T3/T4 subset on light-ending paths |
| (iii) | Spot light (s = 45°, β ∈ {0, 0.3}) grazing a wall + floor + box | Spot profile inside GI, hard cone edges (strong spatial discontinuities for spatial reuse) | Same; edge tiles use t-tests only (no TOST) | Same + disocclusion/edge tile mask report |
| (iv) | Emissive mesh light (subdivided sphere or a textured emissive plane) + a small area light | Mesh-light sampling pdf (area→solid angle), two-sidedness, emission MIS on BSDF hits, many-triangle light selection | Same | Same |
| (v) | Glossy GGX spheres r ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.8} + metallic walls r = 0.3 variant (Enhanced Fig. 12) | Hybrid shift: postponed reconnection, random-replay Jacobian = 1 (PSS), roughness and footprint thresholds (Enhanced Eq. 5), lobe-specific connectability (GRIS §7.6) | Tile δ = 2% (3% for r ≤ 0.1) | Tile δ = 1%; plus T3–T6 over 10^7 shifts in this scene with zero invertibility violations |
| (vi) | Mirror (r = 0.02 and a delta mirror if supported) + diffuse receiver + area light (caustic LS+DE paths). Point-light caustics through a mirror are absent in both unidirectional engines by construction | Near-specular handling, delta-lobe bookkeeping, caustics via BSDF hits, high-variance paths | Longer runs (K = 32); tile δ = 3% outside caustic, caustic region by t-test only | Ensemble R = 1024; caustic mask reported separately |
| (vii) | Textured GLB (Sponza subset or a Khronos sample model) exported through the scene bridge | Texture sampling (sRGB decode, LOD 0 bilinear, wrap), UVs, vertex normals, instancing | δ tile = 2% | Same as (i) |
| (vii-L) | Loader fidelity: our GLB parse vs Blender glTF import | Geometry and transforms | max vertex error < 1e-5·scene scale; identical triangle counts per mesh | — |
| (viii) | USD scene (small usda/usdc with Xform hierarchy, instancing, UsdPreviewSurface, UsdLux Rect/Sphere(point)/Shaping(spot)) via the bridge | USD loader → identical render | Same as (vii) | Same |
| (viii-L) | Loader fidelity: our USD parse vs `pxr` dump in Blender Python | Geometry, xforms, light parameter conversion | As (vii-L); light params equal after documented conversion | — |
| (ix-a) | Static scene (i) with a **moving point light** (linear path), F = 32 frames | Temporal reuse with changing lights; prior-frame p̂ evaluation (GRIS App. B) | Cycles per keyframe (e.g. every 8th frame), K = 8 seeds; our PT every frame | Per frame: ensemble mean (R ≥ 256) vs PT, tile δ = 1%; drift regression slope ≈ 0 |
| (ix-b) | Moving **area light** (translate + rotate) | Light-vertex temporal shift (light-local parametrization recommended [INFERENCE]) | as (ix-a) | as (ix-a) |
| (ix-c) | Rotating **spot** | Cone-edge motion | as (ix-a) | as (ix-a) |
| (ix-d) | **Moving camera** (scripted fly path, including one recorded WASD path), static lights | Reprojection, dual motion vectors, disocclusions (Enhanced §6.4), temporal shift across camera motion | as (ix-a) | as (ix-a) + disocclusion mask report |
| (x) | Many lights: 64 point + 8 rect + 1 emissive mesh | Light selection pdfs (power/tree/light tiles, Enhanced §6.1) | Same as (i) | Same |
| (xi) | Contact geometry: thin gaps, corners, box-on-floor | Distance thresholds, short-segment Jacobian singularities (GRIS §7.5), self-intersection offsets | Tile δ = 3% at contacts | Same as (i) |

Each biased feature (§2.6) gets a separate **bias-budget run** on scenes (i), (v) and (ix-d), mirroring Enhanced Fig. 5/14: the ensemble mean (R = 1024) against our PT gives RMSRB and mean(|bias|/ref) maps. They are reported against a budget (e.g. ≤ 5% MARB), never mixed into the unbiasedness gates.

---

## 9. Acceptance logic (gates)

- **Gate 0: unit and property tests (§4 T1–T16).** All pass, with statistical tests at family-wise α = 0.01 (Šidák). T3 must show zero invertibility violations; T6(a) |Σm − 1| < 1e-5; T6(b) mismatch < 1e-3.
- **Gate 1: harness calibration (§3.9).**
  - A/A tests at nominal false-positive rates.
  - Planted-bias controls detected, with MDB ≤ δ for every gate that claims δ.
- **Gate 2 (Stage A): our PT ≡ Cycles.**
  - Calibration scenes C0a–C0g first, then (i)–(viii), (x), (xi).
  - Per scene: TOST equivalence global and tiles, χ²_red, mean-t, and KS checks within bounds, and metrics ≤ 1.2× the A/A floor.
  - Per-bounce D_b images also pass, to localize failures.
  - Scenes marked "model-approximate" (materials outside §7.4) pass with the looser δ and are listed as such.
- **Gate 3 (Stage B): our ReSTIR (unbiased preset, T16 asserted) ≡ our PT.** Ladder, in order (stop at the first failure):
  1. initial RIS over the path tree only (no reuse)
  2. \+ spatial only (offline mode, independent frames)
  3. \+ temporal only (static, ensemble)
  4. full spatiotemporal (static, ensemble)
  5. full spatiotemporal time-average with M_c = 20 (convergence check; §2.5)
  6. dynamic sequences (ix-a…d) per frame (ensemble)
  7. every unbiased Enhanced feature toggled individually on the full configuration (§2.6 list)

  ReSTIR is also compared directly to Cycles with Gate-2 tolerances as a transitivity sanity check.
- **Gate 4: convergence behavior (offline and spatial-only).** Slope of relMSE_corr ∈ [−1.10, −0.90], N·MSE flat within noise, BNR(N) ~ O(1). Temporal mode: no divergence of the time average over 10^4 frames with M_c = 20. No-M-cap runs are expected to fail, as a documented negative control (GRIS Fig. 10b).
- **Gate 5: biased modes.** Excluded from Gates 3–4. Reported with bias budgets (RMSRB, MARB maps, FLIP), plus a check that the config switch actually restores Gate-3 behavior when turned off.

Why this ordering is valid [INFERENCE]: Stage B compares estimators of one and the same integrand in one engine. Any Stage-B failure is therefore an estimator bug (shift, Jacobian, MIS, W, reservoir), not a scene or material mismatch. Stage A separately proves that the integrand equals Cycles'. Together they give ReSTIR ≡ Cycles.

---

## 10. Risks and open questions

**Risks**
1. **Headless WebGPU on macOS** is unverified on this machine (Chrome 154). Mitigations: headed fallback, dawn.node lane.
2. **dawn.node vs Chrome divergence.** Different Dawn/Tint revisions give different WGSL feature sets and validation, and possibly different codegen. The browser lane is the authority for sign-off; the Node lane is for fast iteration.
3. **Metal fast math** can remove NaN checks and break bitwise replay determinism. It needs bit-level NaN tests and ID-based replay comparisons. `strictMath` is dev-only.
4. **Cycles unit and semantics drift.** Formulas in §7.2–7.3 come from Cycles' *main* branch fetched on 2026-09-28, not the exact 5.1.2 tag. Calibration scenes C0b–C0f must confirm them empirically before other scenes are trusted. Uncertain details: kernel `max_bounce = b+1`, area `spread = π` attenuation, Glossy-node Fresnel, mesh-emitter sidedness.
5. **Material model mismatch** (Principled/glTF PBR vs Cycles Principled v2) makes exact equality impossible outside the restricted model.
6. **Texture filtering.** Cycles uses no mipmaps; the real-time path uses mips. A separate validation sampling mode is needed.
7. **Smooth normals, normal maps and shadow terminator.** Cycles modifies shading at terminators (`shadow_terminator_*`) and may adjust invalid reflections with normal maps [UNVERIFIED]. Expect larger tolerances in normal-mapped scenes.
8. **Spatial-reuse correlation.** Tile SEs must come from run-replicates. Naive per-pixel SE aggregation under-reports uncertainty and produces false failures.
9. **Heavy tails** (caustics, scene vi) need very large R, K and N. Some scenes can only be validated at the tile or global level.
10. **Temporal reuse in dynamic scenes** can be subtly biased if prior-frame state is not kept (GRIS App. B). This may force architectural decisions: keeping previous-frame light transforms and previous BVH/TLAS, or accepting documented bias.
11. **Enhanced §6.2.1 lossy reservoir compression** could inject bias if applied to W, p̂ or Jacobian inputs. Needs an on/off A/B within Gate 3.
12. **Compute cost of Cycles references** (K × S × frames) on the M5 Pro is unmeasured. Cache refs by content hash.

**Open questions**
1. What is the production material model? This decides whether a validation-compatible mode is needed, and how many scenes are "exact" vs "model-approximate".
2. Does the engine support perfect-specular (delta) BSDFs and dielectrics? Scene (vi) changes accordingly.
3. Should the app support an environment light? This affects furnace-test convenience and Cycles world settings.
4. Is RR in initial sampling (Enhanced §6.2.4) wanted from day one? Its PSS pdf modification is described only in their supplemental, which is not in the extracted text.
5. What final sign-off resolution and R: 1024 runs at 512² (Enhanced used 1024 runs), or cheaper?
6. Should the HTML validation report be published as a private claude.ai Artifact for sharing, or kept as local files?

---

## 11. Sources

Papers (local copies):
- Lin, Kettunen, Bitterli, Pantaleoni, Yuksel, Wyman, "Generalized Resampled Importance Sampling: Foundations of ReSTIR", ACM TOG 41(4) Article 75, SIGGRAPH 2022. `…/scratchpad/gris_sig22.pdf`, pages `…/scratchpad/pages/gris/p-04…p-18, p-22, p-23.png`. Equations 1–54 and Thms A.1–A.4 as cited.
- Lin, Kettunen, Wyman, "ReSTIR PT Enhanced: Algorithmic Advances for Faster and More Robust ReSTIR Path Tracing", Proc. ACM CGIT 9(1) Article 13, May 2026. `…/scratchpad/restirpt_enhanced_2026.pdf/.txt` (text lines ~420–476, 565–600, 750–835), pages `…/pages/enhanced/p-03…p-07, p-11, p-17.png`.

Code (read via curl or GitHub API to stdout, 2026-09-28):
- DQLin/ReSTIR_PT, `Source/RenderPasses/ReSTIRPTPass/PathReservoir.slang` (merge/finalize ~L278–457) and `SpatialReuse.cs.slang` (pairwise block ~L338–407 after CR normalization). Repo README (offline settings).
- Blender/Cycles (GitHub mirror, main): `intern/cycles/blender/light.cpp`, `intern/cycles/scene/light.cpp`, `intern/cycles/kernel/light/spot.h`, `intern/cycles/kernel/light/area.h`, `intern/cycles/kernel/integrator/path_state.h` (L112–200), `intern/cycles/kernel/integrator/shade_surface.h` (L659–673), `intern/cycles/kernel/closure/bsdf_microfacet.h` (L782–791, L880–925, L1235–1250).
- Mitsuba 3 `src/python/python/chi2.py`; pbrt-v4 `src/pbrt/bsdfs_test.cpp`.
- NVlabs/flip README and `src/python/api_example.py`, `src/python/README.md`.
- microsoft/playwright `packages/playwright-core/src/server/chromium/chromiumSwitches.ts`.

Web:
- https://playwright.dev/docs/browsers
- https://developer.chrome.com/docs/automation-and-testing/headless-chrome-shell
- https://developer.chrome.com/blog/supercharge-web-ai-testing
- https://agent-browser.dev/webgpu
- https://developer.chrome.com/blog/new-in-webgpu-131
- https://github.com/gpuweb/gpuweb/issues/2270
- https://docs.deno.com/runtime/desktop/webgpu/
- https://github.com/dawn-gpu/node-webgpu
- https://github.com/NVlabs/flip

Registries: npm (`npm view`, tarball `webgpu-0.6.1.tgz` streamed); PyPI JSON for flip-evaluator, OpenEXR, openimageio, numpy, scipy, matplotlib.

Local introspection (read-only): Blender 5.1.2 `--background --factory-startup --python-expr` (Cycles and light properties, shader node enums, colorspace, camera/world defaults, USD/glTF importer options, Cycles devices). Blender Python module checks. `sw_vers`, `system_profiler`, Chrome `--version`, `uv python list`.
