# ReSTIR PT Enhanced (Lin, Kettunen, Wyman 2026): implementation deep dive for a WebGPU/WGSL renderer

> **Where this report lives and why.** The orchestrator asked for this report at
> `/private/tmp/claude-502/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-ea73-491f-b7b7-6d0aab1ffce1/scratchpad/research/enhanced-paper.md`.
> This agent ran in **plan mode**, where the only writable file is this plan file, so the report is here instead.
> Nothing was written to the scratchpad or the project directory. No repos were cloned. The supplemental PDF (196 MB) and the GitHub sources
> were streamed over HTTP to stdout (`curl | gs -sDEVICE=txtwrite -o -`, raw.githubusercontent, the GitHub tree API) and never saved.
>
> **Tags.** Anything the paper, the supplement or code does not say explicitly is marked **[INFERENCE]**. Anything uncertain,
> inconsistent in the sources, or poorly transcribed is marked **[UNVERIFIED]**.
>
> **Citation keys.**
> - **P-§x / P-Eq.n / P-p.N** = main paper (19 pp). Local copy: `scratchpad/restirpt_enhanced_2026.pdf`. Text: `scratchpad/restirpt_enhanced_2026.txt`. Page images: `scratchpad/pages/enhanced/p-NN.png`.
> - **S-§x / S-Eq.n** = supplemental document (13 pp): https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/lin2026restirptenhanced_supplemental.pdf
> - **G-§x / G-Eq.n** = GRIS / ReSTIR PT 2022 (Lin et al.). Local: `scratchpad/gris_sig22.pdf`, `scratchpad/pages/gris/p-NN.png`.
> - **Code22** = the 2022 reference code https://github.com/DQLin/ReSTIR_PT (Falcor/Slang).
> - **RTXDI** = the NVIDIA RTXDI SDK 3.x ReSTIR PT implementation. Samples: https://github.com/NVIDIA-RTX/RTXDI. Runtime: https://github.com/NVIDIA-RTX/RTXDI-Library. **The runtime is proprietary-licensed; see §10.**

---

## 0. TL;DR

ReSTIR PT Enhanced keeps the 2022 ReSTIR PT architecture: PSS hybrid shift (random replay then reconnection), pairwise/Talbot GRIS MIS, and one reservoir per pixel. On top of that it adds the following:

1. **Paired (reciprocal) spatial reuse (P-§3).** Precomputed, tileable "pairing textures" give every pixel exactly one partner per texture, and pairing is symmetric (A↔B).
   - Pairwise MIS for A reusing from B needs the same two shifts, T_{B→A}(X_B) and T_{A→B}(X_A), as B reusing from A. Each pixel therefore shifts **only its own path** into each partner's domain in a **pre-pass**. A second pass then does the resampling.
   - This halves shift evaluations: 2N becomes N per pixel for N neighbors.
   - Textures come from 2×2 random-permutation shuffles. They give Gaussian-distributed offsets with σ = sqrt(8/(9π))·R, so σ = 16 px matches a 30 px disk.
   - Each frame, every texture gets a random flip, mirror, transpose and offset.
   - Measured result: spatial reuse is 1.63× faster, with slightly lower FLIP.
2. **Scene-independent reconnection criteria (P-§4, S-§2–4).** The 2022 rules were `min(α_{k−1},α_k) ≥ α_min` plus `‖x_k−x_{k−1}‖ ≥ d_min`. They are replaced by the following:
   - A **dual ray-footprint threshold**:
     `min( 1/(p_{k−1}(ω_{k−1})·G(x_{k−1}→x_k)), 1/(p_k(ω_k)·G(x_k→x_{k−1})) ) ≥ (c/100)·‖x0−x1‖² / (⟨n_{x1}, x̂1x0⟩/(4π))`, with **c = 0.02**.
   - A **single-vertex roughness test** α_{x_{k−1}} ≥ α_min = 0.2. For black-box BSDFs there is a PDF proxy.
3. **Duplication maps (P-§5).** After each frame, count how many reservoirs in the 17×17 window share the pixel's (initial) random seed, and set D = count/288. The temporal confidence cap becomes `c_Cap = lerp(c_Cap^Default = 20, c_Cap^min = 1, D^α)` with **α = 0.1**. This reduces correlation artifacts but is **biased**: about 3.25% mean absolute relative bias in the hard Kitchen scene. Disabling it restores unbiasedness.
4. **Unified DI + GI (P-§6.1, S-§5).**
   - An NEE ray is also traced from the primary hit x1, so length-2 (direct) paths join the same path tree and reservoir. The separate ReSTIR DI pass and its reservoirs go away.
   - NEE uses **RIS over light-tile candidates**: 128 tiles × 1024 lights, one tile per 8×8 screen tile, 32/B² candidates at bounce B.
   - The PSS is kept "single NEE sample". The RIS effect goes into the UCW (W = W^RIS·p1), and the NEE MIS weight becomes M·p1/(M·p1+p2).
5. **Engineering changes.**
   - Branch-to-select micro-optimisations.
   - **Stream compaction** of pixel–neighbor pairs that need random replay.
   - **Forced reconnection to NEE light vertices**, so replay never has to sample lights.
   - **Russian roulette at initial sampling only**: the PSS source PDF becomes ∏q_i and replay is deterministic.
   - **Reservoir compression** from 88 B to 64 B.
   - **Vector-valued (RGB) resampling-weight accumulation** for shading, which reduces colour noise "for free".
   - **Dual motion vectors** (Zeng et al. 2021) for disocclusions.
6. **Net result.** Frame time drops from 35.73 ms to 13.04 ms (2.74×) with only the cost reductions, or to 15.53 ms (2.30×) with every quality feature. Per-figure speed-ups range from 2.08× to 3.05×. Memory drops from 431 MB to 265 MB at 1080p. Measured on an RTX 5880 Ada at 1920×1080 in Falcor.
7. **Code.** There is **no official code release** for the paper; the project page lists only the PDF, supplement and video. **RTXDI SDK 3.x** (March 2026 onward) ships a ReSTIR PT with footprint reconnection, duplication-map history reduction, forced NEE reconnection and 64 B packed reservoirs.
   - As far as I can tell it does **not** implement paired spatial reuse **[UNVERIFIED: based on grepping SpatialResampling.hlsli]**.
   - Its runtime is under an NVIDIA proprietary license. Use it to understand the semantics; do not copy code from it.

---

## 1. Background as the paper restates it (P-§2, P-p.2–4; S-§1)

### 1.1 RIS / GRIS (P-§2.1, P-Eq.1)

- RIS draws M i.i.d. candidates X_i ~ p and picks Y ∝ `w_i = (1/M)·p̂(X_i)/p(X_i)`. Then `W_Y = (1/p̂(Y))·Σ_{i=1}^{M} w_i` estimates 1/p_Y(Y), and `f(Y)·W_Y` is an unbiased estimator of ∫f.
- GRIS (P-Eq.1): candidates X_i come from domains Ω_i and are mapped by bijective shifts `T_i: Ω_i → Ω` to `Y_i = T_i(X_i)`. The resampling weights are
  `w_i = m_i(Y_i) · p̂(Y_i) · W_{X_i} · |∂T_i/∂X_i|`.
  - W_{X_i} is an unbiased contribution weight (UCW) with E[W_{X_i} | X_i] = 1/p_{X_i}(X_i).
  - m_i are resampling MIS weights that must form a partition of unity over the target domain.
  - Unbiasedness needs a *canonical* sample that covers supp p̂.
- A reservoir is a tuple (X, W_X, c). The confidence weight c used to be called "M".

### 1.2 ReSTIR (P-§2.2)

- **Initial:** per-pixel RIS with streaming weighted reservoir sampling (Chao 1982), reservoir c = 1.
- **Temporal GRIS:** reproject along the shading point's motion. The temporal reservoir's confidence is capped, so the new confidence is `min(c_Cap, c_temp) + 1` with c_Cap = 20 by default.
- **Spatial GRIS:** M−1 neighbors drawn from a disk, "often 30 pixels". Neighbors whose G-buffer normal/depth differ too much are rejected. The output confidence is the sum of all accepted confidences.
- **Shading** is `f(X)·W_X`. The reservoirs then become next frame's temporal input.

### 1.3 ReSTIR PT recap (P-§2.3, S-§1)

**Path notation.** x̄ = [x0 (camera), x1 (primary hit), x2, …, x_d (light)].
- ReSTIR GI always reconnects at x2, which fails if x1 or x2 is specular.
- The **hybrid shift** random-replays (reuses the base path's random numbers) until the first vertex pair (x_{k−1}, x_k) that meets the reconnection criteria, then reconnects y_{k−1} → x_k.
- The reconnection vertex is **precomputed on the base path** during initial sampling, so replay length is known up front. This is GPU-friendly because base-path vertices never need to be stored.
- Invertibility is checked when connecting y_{k−1} → x_k.

**Technique index.** Paths carry a technique index: last vertex NEE-sampled vs BSDF-sampled. The integrand is split with path MIS weights `ω_n(x̄) = p_n(x̄)/(p_1(x̄)+p_2(x̄))`.

**Lobe-indexed path space (S-§1, S-Eq.1–2).** Only one BSDF lobe is evaluated per vertex. Each path carries a lobe sequence `ℓ̄ = (ℓ_i)_{i=1}^{d−1}` with `ℓ_i ∈ {0,…,N_lobe}`; an NEE sample at the last vertex is encoded by index 0.

- S-Eq.1: `I = Σ_{d=1}^{∞} Σ_{ℓ̄∈L_d} ∫_{Ω_d} ω_{n(ℓ̄)}(x̄) · f_{ℓ̄}(x̄) dx̄`, with n(ℓ̄) ∈ {1,2} choosing NEE or BSDF from ℓ_{d−1}.
- S-Eq.2: `f_{ℓ̄}(x̄) = [∏_{i=0}^{d−1} f_{ℓ_i}(x_{i−1}→x_i→x_{i+1}) · G(x_i↔x_{i+1}) · V(x_i↔x_{i+1})] · L_e(x_d→x_{d−1})`.
  - f_{ℓ_i} is the chosen lobe at x_i for i ≥ 1. For i = 0 the factor is the sensor response W_e(x1→x0).
  - At the NEE vertex all lobes are evaluated (ℓ = 0 means "all lobes").
  - Transcription from garbled text; the structure is standard.

**Primary sample space (PSS; S-Eq.3).**
- `I = Σ_d Σ_{t=1}^{2} ∫_{U_{d,t}} F_{d,t}(ū) dū`, with `F(ū) = ω_{n(ℓ̄)}(X_{d,t}(ū)) · f_{ℓ̄}(X_{d,t}(ū)) / p_t(X_{d,t}(ū))`.
- X_{d,t} warps random numbers into a lobe-indexed path.
- The PSS source PDF is p(ū) = 1 for path-tracer samples.
- P-footnote 4: geometry and NDF terms cancel in ω·f/p_t, which is why the PSS target is well behaved.

**Hybrid-shift PSS Jacobian (P-Eq.2; lobe version S-Eq.4).**
```
|∂T/∂ū| = [ p^y_{k−1}(ω'_{k−1}, ℓ_{k−1}) · G(y_{k−1}→x_k) · p^y_k(ω_k, ℓ_k) ]
        / [ p^x_{k−1}(ω_{k−1},  ℓ_{k−1}) · G(x_{k−1}→x_k) · p^x_k(ω_k, ℓ_k) ]
```
- `p^x_k(ω_k, ℓ_k) ≡ p(ω_k, ℓ_k | x_k, −ω_{k−1})` is the joint PDF of sampling direction ω_k = (x_{k+1}−x_k)/‖·‖ and lobe ℓ_k at x_k, given outgoing −ω_{k−1}.
- ω'_{k−1} is the reconnection direction y_{k−1}→x_k.
- `p^•_k(ω_k)` is replaced by 1 when k = d (x_k is the light vertex).
- The Jacobian is 1 when there is no reconnection: random replay is the identity in PSS.
- **G(a→b) = |cos θ_b| / ‖a−b‖², with θ_b measured at the receiving vertex b.** This is the solid-angle-to-area conversion at b.
  - P-p.4 says "θ … angle between y_{k−1}x_k and y_{k−1}'s normal", which reads like a wording slip **[INFERENCE]**. G-Eq.52 and RTXDI use the cosine at the reconnection vertex.
  - P-p.4 also defines p^x_k as "the PDF of sampling −ω_{k−1}". S-§1 gives the correct definition, which is the PDF of ω_k.

**Reservoir contents in ReSTIR PT (P-§2.3 "Implementation").** Random seed(s), the reconnection vertex x_k, and the incident radiance and direction at x_k. In 2022 ReSTIR PT skipped DI paths (d = 2) and relied on a separate ReSTIR DI.

---

## 2. Contribution 1: reciprocal neighbor selection, "paired spatial reuse" (P-§3, P-p.4–6, P-Figs.2,3,6,7,8)

### 2.1 Core observation (P-§3, P-Fig.2)

With pairwise MIS (G-§5.6, Code22), each spatial neighbor j of the canonical pixel c costs **two** shifts:
- T_{j→c}(X_j) to evaluate the candidate in c's domain;
- T_{c→j}(X_c) to evaluate c's sample under j's target for the MIS weights.

When B reuses from A it needs exactly the same pair. So if pixels are *paired* (A's neighbor is B and B's neighbor is A), one pair of shifts serves both pixels. That saves 50% of the shift work (P-§3 ¶2).

### 2.2 Pairwise MIS under pairing: which quantities are exchanged

The paper does not restate the MIS formulas. The version below is transcribed from Code22 `SpatialReuse.cs.slang` (ReSTIRMISKind::Pairwise). The paper states that it keeps the 2022 architecture (P-§2.3 last ¶) **[INFERENCE that Enhanced keeps this MIS]**.

**Setup.** Canonical pixel c. Valid paired neighbors j = 1..k (at most N, one per pairing texture). Post-temporal reservoirs (X_•, W_•, c_•, F_•(X_•)), with p̂ = luminance(F).

**Shift outputs.**
- `Y_j = T_{j→c}(X_j)`, giving F_c(Y_j) (RGB) and J_{j→c}. **Computed by pixel j.**
- `Z_j = T_{c→j}(X_c)`, giving F_j(Z_j) (RGB) and J_{c→j}. **Computed by pixel c.**

**MIS weights** (the division by (k+1) is done once at the end in Code22):
```
m_j(Y_j) = 1/(k+1) · c_j·p̂_j(X_j) / ( c_j·p̂_j(X_j) + (c_c/k)·p̂_c(Y_j)·J_{j→c} )
m_c(X_c) = 1/(k+1) · [ 1 + Σ_j (c_c/k)·p̂_c(X_c) / ( (c_c/k)·p̂_c(X_c) + c_j·p̂_j(Z_j)·J_{c→j} ) ]
```
With c_• = 1 these reduce to the "defensive pairwise MIS" of G-Eq.38 with |R| = 1 and M = k+1.

**Resampling weights.** `w_j = m_j · p̂_c(Y_j) · W_j · J_{j→c}` and `w_c = m_c · p̂_c(X_c) · W_c`. Select Y ∝ w, then `W_Y = Σw / p̂_c(Y)` and `c_out = c_c + Σ_j c_j`.

**Symmetry.** Pixel j, reusing from c through the same texture, needs F_j(Z_j)·J_{c→j} (its candidate) and F_c(Y_j)·J_{j→c} (for its canonical term). That is exactly the same pair. **[INFERENCE, derived]**

**Per-slot storage.** Only the product p̂_•(·)·J (or F·J as RGB) enters MIS and w, so a slot can store `F·J` as RGB plus J separately. J is needed on its own to recover p̂_c(Y) for W_Y, or store F and J. **[INFERENCE]**

**Rejection tests.** Code22 uses `dot(n_c, n_j) ≥ 0.5 && |d_c − d_j| < 0.1·d_c`, where d is camera distance. The depth test is **asymmetric**.
- Correctness does not require symmetry: each pixel's MIS uses only its own accepted set.
- A symmetric test such as `|d_c−d_j| < 0.1·max(d_c,d_j)` avoids computing shifts that only one side uses. **[INFERENCE]**

### 2.3 Pairing ("reuse") texture generation (P-§3.1, P-Eq.3, P-Fig.3)

**Target σ and size.**
- Target per-axis standard deviation σ ≥ 0.8 px of the coordinate delta. P-footnote 1: the smallest σ is 0.8, from one shuffle.
- Choose an **even** square size W, e.g. 254×254.
- P-footnote 2: at W ≥ 256 you need 16-bit channels, because deltas reach ±128 and beyond. With W = 254, |Δ| ≤ 127 fits in signed 8 bits per axis.

**Initialisation (P-Fig.3 left).** Consecutive "link indices", with each index occupying **two horizontally adjacent pixels**:
```
row 0: 1 1 2 2 …
row 1: 3 3 4 4 …
```
So link index L = (y·W + x) >> 1, and there are N/2 links for N = W² pixels.

**Shuffle (P-Fig.3 right).**
- Tile the image into 2×2 blocks and apply an independent uniformly random permutation to the 4 entries of each block.
- Repeat n_σ times. **Every other shuffle offsets the block grid diagonally by (1,1)**, wrapping around the edges (torus).

**Repeat count, P-Eq.3.** Verified by ASCII-rasterising the equation from the PDF at 600 dpi: the bracket is a floor, and all operators are "+".
```
n_σ = ⌊ σ²/2 + 1.46σ⁻¹ + 1.76σ⁻² + 0.656σ⁻³ + 0.5 ⌋ ≈ ⌊ σ²/2 + 0.5 ⌋
```

**Why it works.** Each walk has per-axis std ≈ σ/√2. The two copies of a link walk nearly independently, so the pair delta has std sqrt(σ²/2 + σ²/2) = σ. The negative powers are a fit correction for small σ.

**[UNVERIFIED / INCONSISTENT].** As printed, Eq.3 gives n_σ(0.8) = ⌊6.68⌋ = 6, not 1. The function also has a minimum of about 3.5 near σ ≈ 1.6, so it can never return 1 or 2. That contradicts footnote 1 (σ = 0.8 comes from one shuffle).

My own check:
- One 2×2 shuffle of the adjacent pair gives a per-axis second moment E[dx²] = 4/6, so σ ≈ 0.816. This agrees with the footnote.
- With alternating grids a single walk's variance grows about 1 per step (0.25, 1.25, 2.25 for n = 1, 2, 3), so σ² ≈ 2n and n ≈ σ²/2 for large σ.

**Practical recommendation.** For the default σ = 16, n = ⌊128 + 0.09 + 0.007 + 0.0002 + 0.5⌋ = **128**; the correction terms do not matter there. For small σ, measure the achieved σ empirically after n shuffles and build a small lookup table instead of using the printed fit. **[INFERENCE]**

**Pairing (P-§3.1).**
- After shuffling, the two pixels that carry the same link index are partners.
- Links near the edges can be long. To make the texture **tileable**, wrap each delta: subtract W if the delta is greater than W/2, add W if it is less than −W/2.
- Pack (Δx, Δy) into a 16-bit two-channel texture. P-§7.2: "(Δx,Δy) ∈ [−127,127]², two-channel tileable images".
- Wrapping keeps the negation exact (d_ba = −d_ab), so the *tiled* texture stays self-inverting everywhere on screen. **[INFERENCE, verified by reasoning]**

**Finding links in CUDA (P-§3.1 "Finding links").**
- Build an index table of size N/2 × 2.
- Pass 1: every pixel writes its location to (linkIndex, 0). This is a deliberate write race; only half succeed.
- Pass 2: every pixel whose location is not at (linkIndex, 0) writes to (linkIndex, 1).
- For WebGPU: WGSL/WebGPU make no promise about which value wins a plain-store race. Generate the textures **on the CPU (TypeScript) once at load** (about 254²·128/4 ≈ 2M block permutations, well under 100 ms) and upload them as `rg8sint`. **[INFERENCE]**

**Pseudo-code [INFERENCE: reconstructed from P-§3.1 / P-Fig.3]:**
```
genPairing(W even, sigma):
  n = round(sigma^2/2)                  // or P-Eq.3; = 128 for sigma = 16
  L[y][x] = (y*W + x) >> 1
  for s in 0..n-1:
    o = s & 1                           // diagonal offset every other pass
    for by, bx in 0..W/2-1:
      P = [(2bx+o, 2by+o), (2bx+1+o, 2by+o), (2bx+o, 2by+1+o), (2bx+1+o, 2by+1+o)] mod W
      permute values L[P[0..3]] with a uniformly random permutation of 4
  pos[l] = [] ; for each pixel p: pos[L[p]].push(p)
  for each link l: (a, b) = pos[l]; d = wrap(b - a); delta[a] = d; delta[b] = -d
  wrap(v) = v >  W/2 ? v - W : (v < -W/2 ? v + W : v)   // per component
```

**Sanity test (P-Fig.6).** Swapping every pixel with its partner once gives a Gaussian-"blurred" image; swapping twice returns the original exactly. This makes a good unit test.

### 2.4 Using the textures (P-§3.2)

- For N neighbors, preload N textures of **different sizes** (e.g. **254, 230, 210** for N = 3). This avoids correlation from repeats.
- P-footnote 3: the size heuristic "minimizes near-periods within 3840 pixels", i.e. cases where a multiple of one width is close to a multiple of another.
- Because each texture is self-inverting, a static texture would always pair the same pixels. So **each frame, randomly flip, mirror, transpose and offset each texture**. "Empirically, this completely solves the problem."

**Transform algebra [INFERENCE].**
- Let M be one of the 8 dihedral transforms (integer orthogonal) and o the offset.
- Texture coordinate: `q = (M·p + o) mod W` (use a positive modulo).
- Partner: `p' = p + Mᵀ·d(q)`.
- Then q' = (q + d) mod W and d(q') = −d(q), so p'' = p. Reciprocity survives any such transform.
- A partner that falls off-screen is lost symmetrically.

### 2.5 Choosing σ (P-§7, P-p.12, P-Fig.7)

- An isotropic 2D Gaussian with per-axis std σ has mean radial distance σ·sqrt(π/2).
- A uniform disk of radius R has mean radial distance 2R/3.
- Matching the two gives **σ = sqrt(8/(9π))·R ≈ 0.5319·R**, so R = 30 gives **σ = 16.0**.
- P-Fig.8 pairs: R = 5, 10, 20, 30 correspond to σ = 2.7, 5.3, 10.6, 16.0.
- The Gaussian concentrates samples near the centre (P-Fig.7b histogram), which raises the chance that neighbors are compatible. That is the proposed reason FLIP also drops.

### 2.6 Pass structure, synchronisation and cost (P-§7 ¶3, P-§6.2.2)

**What the paper says.** "Our paired spatial reuse does not fully halve costs, due to overhead from splitting spatial reuse in two: a **pre-pass shifts each path to all paired neighbors**, and a **second pass performs resampling**."

**Implied data flow [INFERENCE].**
- **Pass S1 (shift pre-pass).** One thread per pixel p, or per (p, texture t). For each t:
  - find the partner q_t;
  - run the validity test;
  - if valid, compute T_{p→q_t}(X_p): replay the prefix if needed, reconnect, evaluate F_{q_t}(·) including the visibility ray, and compute J.
  - Write the result to `slot[t][p]`.
  - Inputs: the post-temporal reservoir buffer (read-only), the G-buffer, the pairing textures, and per-frame transform uniforms.
- **Pass S2 (resample).** For each pixel c:
  - read its own slots (for canonical MIS), each partner's slot t (the candidate T_{q→c}(X_q)), the partner reservoirs (W, c, p̂, and the fields to copy if selected), and its own reservoir;
  - compute the MIS above, select, and write the output reservoir (next frame's temporal input) plus the shading colour (§6.6).
- **Synchronisation.** Only the dispatch boundary between S1 and S2. There are **no atomics and no workgroup cooperation**, and S2 does no ray tracing.
- **Memory** is N slots per pixel. For example `vec4<f32>(F·J rgb, J)` is 16 B per slot, so 3 slots is 48 B/px, about 100 MB at 1080p. Halve that with f16 packing if J is clamped. **[INFERENCE]**

**Replay compaction (P-§6.2.2)** layers on top.
- Only some (pixel, neighbor) pairs need random replay: those where the path's reconnection vertex is beyond x2 or absent.
- Parallelise over pixel–neighbor pairs and **stream-compact** the ones that need replay. This reduces warp divergence and active warps.
- In WebGPU **[INFERENCE]**: a classification pass does `atomicAdd` on a storage-buffer counter and appends (pixel, slot) to a list. A tiny pass writes `dispatchWorkgroupsIndirect` arguments. The replay pass writes the replay state (Code22 `ReconnectionData`: rcPrevHit 16 B, rcPrevWo 12 B, pathThroughput 12 B) per slot. S1 then only reconnects.
- In the baseline this state is 2 entries per neighbor (both directions) in a 256 B/pixel `PixelReconnectionData`; pairing needs **one** per slot. **[INFERENCE]**

**Measured cost.**

| Measurement | Before | After | Speed-up |
|---|---|---|---|
| P-Fig.8, Opera House, σ = 2.7 | 6.24 ms (FLIP 0.329) | 3.72 ms (FLIP 0.313) | |
| P-Fig.8, Opera House, σ = 5.3 | 6.13 ms (0.294) | 3.99 ms (0.289) | |
| P-Fig.8, Opera House, σ = 10.6 | 6.24 ms (0.285) | 3.77 ms (0.283) | |
| P-Fig.8, Opera House, σ = 16.0 | 6.21 ms (0.289) | 3.79 ms (0.287) | **1.63× average** (P-§7.2) |
| Table 1 spatial pass, 4-scene average | 6.99 ms | 4.11 ms | 1.70× |
| Veach Ajar (S-Table 1) | 4.50 | 2.88 | |
| Carousel (S-Table 1) | 8.81 | 4.81 | |
| Opera House (S-Table 1) | 9.10 | 5.11 | |
| Bathroom (S-Table 1) | 5.55 | 3.63 | |

In the Table 1 row, initial sampling rose from 11.79 to 12.56 ms and temporal from 2.66 to 2.80 ms. That is unexplained; possibly measurement noise.

### 2.7 Relation to prior work (P-§3.3)

Bekaert et al. 2002 split 16×16 blocks into 16-pixel N-rooks groups with all-to-all reuse. That also avoids extra MIS work but causes structured artifacts; Code22 has this as the "BPR/NRooks" mode. The overlaid, differently sized Gaussian-permuted textures have no block structure. The asterisk in P-§1 says this applies to other reuse algorithms (ReSTIR GI, path reuse) too.

---

## 3. Contribution 2: scene-independent (footprint) reconnection criteria (P-§4, S-§2–4, S-§10–11)

### 3.1 The 2022 rule it replaces

**Reconnection vertex.** x_k is the first pair (x_{k−1}, x_k) with:
- **roughness** `min(α_{x_{k−1}}, α_{x_k}) ≥ α_min`, using the roughness of the *sampled lobe*;
- **distance** `‖x_k − x_{k−1}‖ ≥ d_min`.

**Values.**
- G-§8.3 uses a per-lobe GGX roughness threshold of 0.2 and d_min of 1–5% of scene size.
- In Code22 α is Falcor's **`linearRoughness`**, the perceptual roughness. GGX α = linearRoughness², so 0.2 means GGX α ≈ 0.04. Default `specularRoughnessThreshold = 0.2`, `nearFieldDistance = 0.1` world units.
- Note that glTF `roughnessFactor` and Blender Principled Roughness use the same perceptual convention.
- The Enhanced paper's comparisons use d_min = 2% of the shortest scene dimension (P-§7.2).

**Problem.** Suitable distance and roughness are interdependent and need per-scene tuning. P-Fig.10: thresholds that are good near are bad far. P-Fig.11: San Miguel caustics fail from near-delta distant highlights.

### 3.2 What a good reconnection means (P-§4, P-Eq.4)

- The ideal shift preserves the target: `p̂_j(T(ū))·|∂T/∂ū| ≈ p̂_i(ū)` (Wyman et al. 2023 course).
- Assuming p̂_j(T(ū)) ≈ p̂_i(ū) (G and NDF cancel in the PSS integrand), the goal is **|∂T/∂ū| ≈ 1**.
- Factor P-Eq.2 into:
  - **(4a)** `[p^y_{k−1}(ω'_{k−1})·G(y_{k−1}→x_k)] / [p^x_{k−1}(ω_{k−1})·G(x_{k−1}→x_k)]`: the ratio of the *area densities of x_k* when traced from y_{k−1} vs x_{k−1};
  - **(4b)** `p^y_k(ω_k) / p^x_k(ω_k)`: the ratio of the solid-angle densities of ω_k with outgoing direction y_{k−1}x_k vs x_{k−1}x_k.
- Require each relative change to be below ε (P-Eq.6a, 6b). Then `(1−ε)² < |∂T/∂ū| < (1+ε)²`.

### 3.3 The new criterion (P-Eq.5; S-Eq.5 is the same written as (max(…))⁻¹)

```
min( (p^x_{k−1}(ω_{k−1}) · G(x_{k−1}→x_k))⁻¹ ,  (p^x_k(ω_k) · G(x_k→x_{k−1}))⁻¹ )
      ≥  (c/100) · ‖x0 − x1‖² / ( ⟨n_{x1}, x̂1x0⟩ / (4π) )
```

**Ray footprint (first term).** `(p^x_{k−1}(ω_{k−1})·G(x_{k−1}→x_k))⁻¹` is the reciprocal area density of x_k when traced from x_{k−1}, i.e. the area per sample at x_k.
- G(x_{k−1}→x_k) = |cos θ_{x_k}| / t², where t = ‖x_k − x_{k−1}‖ and the cosine is at x_k.
- p_{k−1} is the solid-angle BSDF sampling PDF at x_{k−1} for ω_{k−1}.

**Inverse ray footprint (second term).** `(p^x_k(ω_k)·G(x_k→x_{k−1}))⁻¹` is the area per sample at x_{k−1} of a *reverse* ray sampled at x_k.
- G(x_k→x_{k−1}) = |cos θ_{x_{k−1}}| / t², with the cosine at x_{k−1}.
- p_k(ω_k) is the BSDF sampling PDF at x_k of the *next* direction, given incoming from x_{k−1}.

**Right-hand side.** `R_pri² = ‖x0−x1‖² · 4π / |cos θ_{x1}|` is the "primary ray footprint" of Müller et al. 2021 (NRC's "spread at primary vertex", which assumes a 1/(4π) primary PDF), scaled by c/100.
- **c = 0.02**, so c/100 = 0.0002. P-§4 and P-§7.2 call it "near-optimal across a wide range of scenes".
- A larger c is more conservative: reconnection happens later.

**Single-vertex roughness.** Additionally **α_{x_{k−1}} ≥ α_min**, using ReSTIR PT's default **α_min = 0.2** (P-§7.2). There is *no* roughness test at x_k.
- This guards cases where parallax, curvature or very low roughness make the footprint bounds unreliable (P-§4.2).
- It also handles **environment-light reconnection**, where Δx_k is unbounded and the angular density can change sharply.

**Special cases.**
- P-footnote 6: for **diffuse or emissive x_k**, reconnection does not change p^x_k(ω_k), so **the inverse footprint test is skipped**. For an emissive light vertex there is no ω_k.
- Delta lobes have PDF = ∞, so the footprint is 0 and they are never reconnectable. RTXDI returns 0 for `IsDelta()`.

### 3.4 Derivation (S-§2, S-Eqs.6–25)

Summarised with renamed constants; the garbled subscripts were resolved by structure.

**1. Assumed bounds on neighbor displacement.**
- Primary hits of neighboring pixels satisfy `|x1 − y1| < c1·R^x̄_pri`, with `R^x̄_pri = sqrt(‖x0−x1‖² / (⟨n_{x1}, x̂1x0⟩/(4π)))` (S-Eq.6).
- Before reconnection, random replay yields corresponding vertices on the same locally flat surface, with `|x_k − y_k| < c2'·|x1 − y1|` (S-Eq.7).

**2. Area density.** ρ^x̄(x_k) ≡ p^x_{k−1}(ω_{k−1})·G(x_{k−1}↔x_k) (S-Eq.10); 1/ρ is the **ray footprint**. Tracing from y_{k−1} gives ρ^ȳ(x_k) = p^y_{k−1}(ω'_{k−1})·G(y_{k−1}↔x_k) (S-Eq.11).

**3. Empirical assumption (S-Eq.13, P-Fig.4).** Random replay preserves area density: ρ^x̄(x_k) ≈ ρ^ȳ(T_replay(x_k)). P-Fig.4 shows that replay with the same random numbers produces a near-exact spatial *translation* of the secondary-hit populations. So (4a) is controlled by how much a single density function varies under a small spatial shift (S-Eq.14).

**4. Footprint smoothness assumption (S-Eqs.15–16).** For an area density p_area, if ‖a−b‖ < sqrt(c_s / p_area(a)) then |p_area(a) − p_area(b)| / p_area(a) < ε. In words: density is roughly constant within a footprint-sized disk.

**5. Resulting sufficient condition (S-Eqs.17–19).**
- `c̃1·R^ȳ_pri < sqrt(c_s/ρ^ȳ(x_k))`, i.e. `1/ρ^ȳ(x_k) > (c̃1²/c_s)·(R^ȳ_pri)²`.
- By invertibility the same must hold for x̄: `(p^x_{k−1}(ω_{k−1})·G(x_{k−1}→x_k))⁻¹ > (c̃1²/c_s)·(R^x̄_pri)²`.
- The constants collapse into c.

**6. Inverse footprint (S-Eqs.20–25).**
- Assume BSDF sampling-PDF reciprocity, `p(ω_k | x_k, −ω_{k−1}) ≈ p(−ω_{k−1} | x_k, ω_k)`. P-footnote 5: this is exact for NDF sampling of microfacets and approximate for VNDF (Heitz 2018).
- Then (p_k(ω_k)·G(x_k→x_{k−1}))⁻¹ is the footprint at x_{k−1} of a reverse-traced ray. The area density is p(x_{k−1} | x_k, ω_k) = p_σ(−ω_{k−1} | x_k, ω_k)·G(x_k→x_{k−1}) (S-Eq.22; transcription reordered for dimensional consistency **[UNVERIFIED exact typography]**).
- Apply the same smoothness argument to |y_{k−1} − x_{k−1}|, with G(x_k→x_{k−1}) ≈ G(x_k→y_{k−1}) for distant reconnections.
- Result: `(p^x_k(ω_k)·G(x_k→x_{k−1}))⁻¹ > (c1²/c2)·(R^x̄_pri)²` (S-Eq.25).
- The same c is reused for both, which gives P-Eq.5.

### 3.5 Lobe handling: marginal vs conditional PDF (S-§3, S-Fig.1)

- 2022 uses lobe-specific connectability and keeps the lobe index through reconnection (S-Eq.4).
- Ideally the footprint would use the lobe-conditional PDFs p(ω|ℓ). In practice the paper uses the **marginal PDF over all lobes**, which is already computed for MIS.
- Effect: on diffuse+specular mixtures (plastic) the specular PDF is diluted by the diffuse lobe, so reconnection happens early. This is minor: S-Fig.1, Veach Ajar variant, gives marginal FLIP 0.171 vs conditional 0.169 at c = 0.02.

### 3.6 Black-box / PDF-based roughness proxy (S-§4, S-Eq.26, S-Fig.2)

- For materials without roughness or lobes (e.g. neural materials, Zeltner et al. 2024), the footprint tests work unchanged with the BSDF sampling PDF.
- The roughness test becomes, **as printed** (confirmed by ASCII-rasterising the PDF at 600 dpi; the exponent is a superscript "2"):
  **`1 / (p^x_{k−1}(ω_{k−1}))² ≥ α_min`**, "using the same α_min tuned for the original roughness threshold".
- **[UNVERIFIED: likely scale mismatch]** Rough calibration using the GGX reflection-PDF peak ≈ D(n)/4 = 1/(4πα_ggx²):
  - The printed form with α_min = 0.2 accepts only p ≤ 2.24, i.e. perceptual roughness ≳ 0.43.
  - The form `1/sqrt(p) ≥ α_min` accepts p ≤ 25, i.e. perceptual roughness ≳ 0.24, which is close to linearRoughness 0.2.
  - **RTXDI implements `pdf ≤ 1/minPdfRoughness²`, i.e. 1/sqrt(p) ≥ minPdfRoughness**, with default minPdfRoughness = **0.1** (about perceptual 0.17), and applies it in footprint mode even for parametric materials.
- Recommendation: for parametric glTF materials, use the sampled lobe's perceptual roughness ≥ 0.2. If using a PDF proxy, use the 1/sqrt(p) form and calibrate. **[INFERENCE]**

### 3.7 Alternatives the authors rejected (S-§11)

- **Path-footprint threshold alone** (Müller 2021 / Bekaert 2003) ignores the reconnection vertex's material. It lets the rc vertex sit on a delta surface (glass egg), producing noisy caustics.
  - S-Fig.11: path footprint FLIP 0.173 at c = 0.02 and 0.219 at c = 2; ours 0.159. No c matches ours.
- **Jacobian bounding** (S-Eq.27): `(p^x_{k−1}(ω_{k−1})·G(x_{k−1}↔x_k)·p^x_k(ω_k))⁻¹ ≥ (c/100)·‖x0−x1‖² / (⟨n_{x1}, x̂1x0⟩/(8π²))`, i.e. the Eq.5 RHS multiplied by 2π.
  - It is analogous to VPL clamping (Davidovič 2010; Hašan 2009) and rejects glossy-to-glossy connections too aggressively.
  - S-Fig.12 (Kitchen): FLIP 0.237 at c = 0.02, 0.234 at c = 0.0002; ours 0.229.

### 3.8 Parameter sweep (S-§10, S-Fig.8–10)

- c was swept from 0.005 to 0.64 over six scenes. Per-scene optimum: Kitchen 0.04, Crown 0.01, Bathroom 0.02, Veach Ajar 0.02, Burger Restaurant 0.01, Bistro Exterior 0.005. The average is 0.0175, hence c = 0.02; the optimal range is 0.005–0.04.
- Diffuse scenes are insensitive to c (Burger Restaurant: 0.005 and 0.02 look alike). Glossy scenes are sensitive (Kitchen: 0.02 clearly beats 0.005).

### 3.9 Numeric intuition (my computation, **[INFERENCE]**)

- Threshold = 0.0002·4π·d²/cos θ1 ≈ 0.00251·d²/cos θ1.
- Primary hit at d = 5 m, cos = 1: threshold ≈ 0.063 m².
- From a Lambertian x_{k−1} (p ≈ 0.32) to x_k at normal incidence: 3.1·t² ≥ 0.063, so reconnection is allowed once t ≳ 0.14 m (about 2.8% of the view distance).
- A glossy lobe with p ≈ 10 needs t ≳ 0.8 m.
- The threshold therefore scales automatically with view distance (P-Fig.10) and with roughness (P-Figs.11,12). Fig.12 shows metallic walls at roughness 0.3 get larger reconnection distances.

**Caveat [INFERENCE].** R_pri uses 4π, not the pixel's solid angle, so the criterion ignores FOV and resolution. Spatial neighbors are chosen in *pixels* (σ = 16 px), so the world-space neighbor displacement does depend on FOV and resolution. c was tuned at 1080p with the authors' scene FOVs. A web app with a user FOV slider and arbitrary canvas sizes may want to scale c by (pixel solid angle / reference). This is untested.

### 3.10 Invertibility with the new criteria [INFERENCE, consistent with the RTXDI code comments]

The rc vertex is a property of the base path. For T to be bijective, the offset path ȳ must:
- **fail** the criteria at every replayed pair (y_{i−1}, y_i) with i < k, otherwise the inverse shift would reconnect earlier;
- **pass** at (y_{k−1}, x_k), otherwise the inverse would not reconnect there.

Consequences:
- Re-evaluate the dual-footprint and roughness tests **on the offset path**. The inverse footprint at x_k needs p^y_k(ω_k) with the new incoming direction.
- Use the **offset pixel's own** R_pri. RTXDI recomputes it from the resampling surface; for temporal shifts into the previous frame it uses the **previous** camera position.
- RTXDI **jitters** the thresholds (minConnectionFootprint ~ N(μ, (0.2μ)²), minPdfRoughness ~ N(μ, (0.01μ)²)) using the **replayed** random stream, so base and offset draw the same threshold.
- Code22 similarly stores `initRandomSeed` "for recovering the random distance threshold for hybrid shift".

### 3.11 Consolidated per-vertex test (pseudo-code) [INFERENCE: assembled from P-§4, S-§2–4, footnotes 5–6, RTXDI]

```
thr = (c/100) * dot(x1-x0, x1-x0) * 4*PI / abs(dot(n_x1, normalize(x0-x1)))    // per pixel (per domain)

// x_k reached by BSDF sampling from x_{k-1}, k >= 2, while rcVertex not yet found
t2      = |x_k - x_{k-1}|^2
rayFP   = isDelta(lobe_{k-1}) ? 0 : t2 / (pdf_{k-1}(w_{k-1}) * |cos theta_{x_k}|)
roughOK = roughness(lobe_{k-1}) >= alpha_min                  // or 1/sqrt(pdf_{k-1}) >= alpha_min (black box)
if x_k is emitter / env hit (path ends): reconnect_here = roughOK && rayFP >= thr   // env: rayFP = +inf
else:
  // after sampling w_k at x_k (marginal pdf over lobes)
  invFP = (x_k diffuse-only) ? +inf
        : isDelta(lobe_k) ? 0 : t2 / (pdf_k(w_k) * |cos theta_{x_{k-1}}|)
  reconnect_here = roughOK && min(rayFP, invFP) >= thr
// NEE-sampled light vertex x_d: if no rcVertex yet -> forced reconnection (P-§6.2.3)
```
The paper does not spell out the exact order of the tests. RTXDI keeps "last vertex rough", "last vertex far" (ray footprint) and "current vertex rough-for-connection" (inverse footprint) as separate flags.

---

## 4. Contribution 3: duplication maps for decorrelation (P-§5, P-Figs.5,9,14,15)

### 4.1 Motivation

Correlation comes from low-probability, high-energy initial fireflies, and from imperfect shifts that "shift samples into fireflies". Spatiotemporal reuse spreads these into blobs and streaks that survive many frames. Denoisers then treat them as signal. Prior fixes are more expensive and narrower: Conditional ReSTIR (Kettunen 2023) keeps prefixes independent, and MCMC mutations (Sawhney 2024) address sample impoverishment.

### 4.2 Algorithm (P-§5 list)

1. **Build the map after each frame.** For each pixel, count reservoirs in the surrounding **17×17** window whose random seed equals its own. Divide by **288** (= 17² − 1, i.e. excluding the centre) to get **D ∈ [0,1]**.
   - "Share its sample" means shifted copies of the same initial candidate, detected by comparing the seeds already stored for random replay.
   - **[INFERENCE]** Use `initRandomSeed`, which follows the sample through every shift. Treat empty reservoirs as "no ID" (RTXDI writes 0 and skips 0).
2. **Look up D during temporal resampling** at the temporal reservoir used, i.e. the current pixel back-projected into the previous frame.
3. **Adapt the cap:**
   `c_Cap = lerp(c_Cap^Default, c_Cap^min, D^α)`, with **c_Cap^Default = 20**, **c_Cap^min = 1** and **α = 0.1** (P-§5 last ¶).
   - α = 1 is a linear reduction; α → 0 reduces faster.
   - Worked values (my computation): one duplicate (D = 1/288) gives D^0.1 = 0.568 and c_Cap ≈ 9.2. At D = 10/288, c_Cap ≈ 6.4; at D = 0.2, c_Cap ≈ 3.8. The reduction is very aggressive.
   - After temporal resampling the confidence is min(c_Cap, c_temp) + 1.

### 4.3 Bias (P-§5 end, P-§7.4)

- c becomes sample-dependent, so the MIS weights no longer form a partition of unity and the method is biased. The bias only appears where the cap is reduced ("trades correlation for bias"). It shows as energy loss, larger on glossy surfaces.
- Kitchen: mean(|bias|/ref) = **3.25%** (P-Fig.5f). Frame time 12.9 → 13.6 ms and FLIP 0.297 → 0.295.
- Watercolor with an upward-moving camera: 1024-run averages show systematic error (P-Fig.14h).
- P-Fig.15 convergence (1024 independent runs of the same animation):
  - Biased and unbiased variants nearly overlap for MSE and FLIP within < 0.1 s of accumulated time.
  - The biased one starts with *lower* MSE because it stops outliers spreading, then plateaus as bias dominates. The plateau is stronger in Kitchen than in Watercolor.
- Authors' recommendation: use the biased variant for real time. For an unbiased converged image, accumulate with **temporal reuse disabled** (per Lin 2022).

### 4.4 Comparison with the RTXDI boiling filter (P-§7.2, P-Fig.9, Tower Bridge)

- The boiling filter computes the average resampling weight w̄ over each warp and clears reservoirs with w_j > a·w̄, where a = −9 + 10/s and strength s ∈ (0,1].
- FLIP, where higher means more darkening bias:

| Method | Weak | Medium | Strong |
|---|---|---|---|
| Duplication map | α = 0.5: 0.372 | α = 0.1: 0.405 | α = 0.02: 0.451 |
| RTXDI boiling filter | s = 0.05: 0.402 | s = 0.2: 0.499 | s = 0.8: 0.576 |

- Unfiltered FLIP is 0.352. The figure labels the α glyph so that it looks like "c".

### 4.5 GPU implementation

**Paper.** The whole decorrelation, colour-noise and disocclusion package costs about +1 ms on average (Table 1: 14.51 → 15.53).

**RTXDI's version** (a useful reference implementation, `PT/DuplicationMap.hlsli`):
- 16×16 thread groups load a 32×32 tile of sample IDs into group-shared memory (each thread loads 2×2).
- Count equal non-zero IDs in the 17×17 window, excluding self.
- Store count/255 in an unorm channel.
- In temporal resampling: `impoverishment = saturate(dupCount/288)`, `t = pow(impoverishment, 0.1·2^{6(1−s)−3})` (s = 0.5 gives exponent 0.1, i.e. the paper's α), and `cap = max(1, int(lerp(maxHistory, 1, t)))`.
- RTXDI defaults: maxHistoryLength = 8 without DLSS-RR and 20 with it. Duplication reduction is **on only with DLSS-RR**.

**WGSL mapping [INFERENCE].** `@workgroup_size(16,16)` with `var<workgroup> tile: array<u32, 1024>` (4 KB, well under WebGPU's 16 KB default). One pass per frame; output to an `r8unorm` or `r16float` texture that the next frame's temporal pass reads.

**Seed uniqueness.** Seeds must be unique per pixel per frame; a 32-bit hash of (pixel, frame) is fine, and collisions are about 289/2³² per pixel per frame.

**Integer M.** Enhanced packs **M as an 8-bit integer** (S-Alg.1) while lerp gives fractional caps. **[UNVERIFIED]** how the paper rounds; RTXDI truncates with `int()`.

---

## 5. Unifying direct and indirect illumination (P-§6.1, S-§5, S-Fig.4)

### 5.1 Path tree and selection

**2022.** The path tree spawns NEE rays from the BSDF-sampled vertices x2..x_n. Initial RIS selects one path with d ≥ 3; DI came from a separate ReSTIR DI.

**Enhanced.**
- Also trace an **NEE ray from x1** while building the path tree.
- The single initial RIS may now pick a **length-2** path, either NEE at x1 or a BSDF sample from x1 that hits an emitter. **[INFERENCE]** The BSDF-hit case follows from the path tree already containing emitter hits, since MIS pairs NEE and BSDF.
- The selected path is drawn from the full path space and stored in the **single** reservoir. This removes the ReSTIR DI pass and its storage.

**Quality benefit ("somewhat surprisingly").** DI now gets the hybrid shift and pairwise MIS, which helps glossy highlights.
- S-Fig.4: separate ReSTIR DI 16.0 ms / FLIP 0.390 vs unified 13.4 ms / FLIP 0.332.
- Using light tiles for NEE also improved lighting at later bounces.

### 5.2 NEE with RIS and light tiles (P-§6.1 last ¶, P-§7 ¶4)

**Light RIS.**
- Optional RIS for NEE, used when a scene has many lights.
- As in ReSTIR DI, the target excludes visibility; only the selected light is shadow-tested.

**Light tiles** (Wyman & Panteleev 2021):
- Each frame, presample **128 tiles × 1024 lights**.
- Each **8×8** screen tile picks a light tile and draws candidates from it at every bounce. **[UNVERIFIED]** whether the tile is re-picked per bounce.

**Candidate counts.** **32** at the primary hit (bounce B = 1), and **32/B²** at bounce B, clamped to at least 1. That gives 32, 8, 3.6, 2, 1.28… **[UNVERIFIED rounding]**.

**PSS treatment (S-§5).** The RIS is a black box. The PSS stays defined as "one NEE sample with a known PDF p1" (the light-sampling source PDF).
- **Path MIS weight** (multi-sample balance heuristic): `ω1(x̄) = M·p1(x̄)/(M·p1(x̄) + p2(x̄))` and `ω2(x̄) = p2(x̄)/(M·p1(x̄) + p2(x̄))`. p2 is the BSDF-sampling PDF and M the candidate count at that bounce. This is a valid partition of unity, so it stays unbiased; it heuristically assumes RIS behaves like evaluating all M lights.
- **PSS integrand for an NEE path:** `ω1(x̄)·f(x̄)/p1(x̄)`.
- **UCW of the chosen NEE sample:** `W_{X(U)} = W^{RIS}_{X(U)} · p1(X(U))` instead of 1. Here U are the hypothetical single-sample random numbers that would produce that light sample, and W^{RIS} is the area-measure path-space UCW from RIS. The p1 factor is the Jacobian between parameterisations; plain NEE gives W^{RIS} = 1/p1, which cancels to 1.
- **Combined with RR (§6.5) [INFERENCE]:** initial UCW = W^{RIS}·p1 / ∏_i q_i.

### 5.3 Forced NEE-light reconnection (P-§6.2.3)

- Replaying an NEE-terminated path would force the replay to re-run light sampling, which is expensive, especially with RIS.
- Since replayed random numbers usually pick the same light anyway (e.g. power-based sampling), **the NEE light vertex is forced to be the reconnection vertex if no earlier one exists**.
- Any variance increase is absorbed by the path MIS weights, which already downweight these cases (the preceding surface is usually glossy).
- RTXDI does the same: "we don't allow random replay to the NEE sample (NEE sample is always reconnected if the path is not reconnected before)". It stores the light identity/UV in the reservoir's radiance slot and flags it with x = INF.

### 5.4 Implications for point, spot, area and environment lights [INFERENCE unless noted]

**Point and spot lights (delta position).**
- Only NEE can reach them, so ω1 = 1 and the NEE/BSDF MIS is not needed for them.
- The light vertex is a fixed point. Reconnection from any y_{d−1} is the ReSTIR DI shift, and the light part of the Jacobian is 1 when the selection probability does not depend on the shading point. For power-based or light-tile selection it doesn't; for "importance by distance" selection it does, so use the ratio of selection PMFs.
- The spot cone/profile goes into L_e(x_d→x_{d−1}), evaluated at the offset path's direction.

**Area lights (emissive triangles, quads, spheres).**
- With **area-measure** sampling (uniform over the surface after power-based selection), the light-vertex Jacobian is 1.
- With **solid-angle** sampling (spherical rectangle/triangle/cone), the area density depends on the shading point. The reconnection Jacobian then includes `p^y_σ(x_d)·G(y_{d−1}→x_d) / (p^x_σ(x_d)·G(x_{d−1}→x_d))`. RTXDI records a `partialJacobian` equal to the light's solid-angle PDF for this.
- BSDF hits on area lights take part in MIS via ω2, and the reconnection criteria apply to them as to any BSDF-sampled final vertex (rayFP test, no inverse test).

**Environment map.** A vertex at infinity means Δx is unbounded and the ray footprint is infinite. **Only the x_{k−1} roughness test guards it** (P-§4.2). Store a direction, not a position.

**Moving lights and objects.**
- Store the reconnection vertex as (instanceID, primitiveIndex, barycentrics), as S-Alg.1 does. The rc vertex then follows animated geometry.
- For light vertices store (light index, uv); positions are re-derived each frame from current transforms.
- **Cached suffix radiance** (`rcVertexRadiance`) goes stale when lights move. Code22 has a `temporalUpdateForDynamicScene` path that re-traces the temporal sample.
- Temporal *inverse* shifts (current → previous domain, for MIS) need the **previous frame's** geometry, lights and camera. RTXDI computes the previous primary footprint from the previous camera position.
- For a WebGPU BVH, keep the previous TLAS/instance transforms and previous light buffer for one frame.

---

## 6. Other optimisations and quality fixes (P-§6.2–6.4, S-§6–8)

### 6.1 Low-level code optimisation (P-§6.2.1)

Replace most branches (reservoir-update logic, heterogeneous path cases) with conditional moves and simplify arithmetic, without changing results.
- Profile, Opera House in NSight: SM warp occupancy 22.4% → 31.1%, active threads per warp 15.3 → 19.9, warp latency 347k → 241k cycles, "without changing the sampler behavior" (P-§7.1).
- Table 1: 35.73 → 32.98 ms.

**WGSL [INFERENCE]:** use `select()`, avoid early `return` inside reservoir-merge loops, and keep a uniform loop trip count over the N neighbors. Apple GPUs also execute 32-wide SIMD groups.

### 6.2 Reservoir compression (P-§6.2.1, S-§8, S-Alg.1)

The baseline is 88 B (Code22 `PathReservoir`):
- M (f32), weight (f32), pathFlags (u32), rcRandomSeed (u32), F (f32×3), lightPdf (f32), cachedJacobian (f32×3), initRandomSeed (u32);
- rcVertexHit {instanceID, primitiveIndex, bary f32×2} (16 B), rcVertexWi (f32×3), rcVertexIrradiance (f32×3).

Enhanced is **64 B** (S-Alg.1 verbatim fields):

| Field | Storage | Note |
|---|---|---|
| `float W` | 4 B | UCW |
| `float3 F` | 12 B | integrand; **p̂ = luminance(F)** |
| `uint initRandomSeed` | 4 B | seed of initial path (also the duplication-map ID [INFERENCE]) |
| `uint rcVertexRandomSeed` | 4 B | seed to continue after the rc vertex |
| `uint pathFlags` | 4 B | **M packed as an 8-bit integer** inside; path length, rc length, NEE flag, light type, delta/transmission/specular bits (Code22 bit layout) |
| `uint rcVertexInstanceID` | 4 B | |
| `uint rcVertexPrimitiveIndex` | 4 B | |
| `uint rcVertexBarycentrics` | 4 B | 2×16-bit unorm |
| `uint rcVertexWi` | 4 B | octahedral 2×16-bit unorm |
| `float3 rcVertexRadiance` | 12 B | incident radiance at rc vertex |
| `float2 rcVertexCachedValues` | 8 B | (a) **product** of the base path's Jacobian terms p(ω_k), G(x_{k−1},x_k), p(ω_{k+1}), formerly float3; (b) NEE light PDF, unchanged, for path MIS |

Total 64 B.

**Memory.**
- Double buffering: 2×(88 + 16 ReSTIR DI) = 208 B/px, 431 MB at 1080p, becomes 2×64 = 128 B/px, 265 MB (P-§7.1 "Storage").
- Lower bandwidth in the neighbor-heavy reuse passes gives better resolution scaling: 1080p→4K frame time ×4.03 vs ×4.57 for the baseline (S-§9, S-Table 2).

**WGSL [INFERENCE].**
- `array<vec4<u32>, 4·numPixels>` (4×vec4u per reservoir) with `bitcast<f32>`, `pack2x16unorm`/`unpack2x16unorm` and octahedral helpers.
- 1920×1080×64 B = 132.7 MB, which is just under WebGPU's **default** `maxStorageBufferBindingSize` of 128 MiB (134.2 MB). Higher resolutions must request a larger limit from the adapter.
- RTXDI also uses 4×uint4 = 64 B, but a different layout: world position + normal instead of instance/prim/bary, plus age, rcWiPdf, partialJacobian and randomIndex.

### 6.3 Stream compaction for random replay (P-§6.2.2)

Covered in §2.6. Table 1: 29.75 → 26.81 ms. Spatial 9.73 → 6.99 ms and temporal 3.40 → 2.66 ms. Initial sampling rose slightly (11.44 → 11.79 ms), which is not explained.

### 6.4 Forced NEE reconnection (P-§6.2.3)

Covered in §5.3. Table 1: 32.98 → 29.75 ms. Spatial 13.06 → 9.73 ms and temporal 4.16 → 3.40 ms.

### 6.5 Russian roulette at initial sampling only (P-§6.2.4, S-§6, S-Fig.3)

**Problem.** Normally RR adds PSS dimensions. Replaying RR decisions in a shift can map a surviving path to a killed one, which is a shift failure.

**Fix.**
- Define the PSS **without** RR. At initial sampling, apply an "external" roulette whose only effect is to change the initial sample's PSS source PDF from p(ū) = 1 to **p(ū) = ∏_{i=1}^{d−1} q_i(ū)**, where q_i are the per-bounce survival probabilities. The UCW becomes 1/∏q_i.
- During replay, survival is deterministic, i.e. no RR.
- The paper does not give the q_i formula. **[UNVERIFIED]**; a throughput-luminance-based q is typical.

**Effect.**
- Table 1: 25.02 → 16.52 ms, the largest single win. Initial sampling 12.56 → 5.21 ms.
- NSight (Opera House): occupancy 34.9%, 20.6 threads/warp, warp latency 82k cycles.
- S-Fig.3, Veach Ajar:

| Method | Time | FLIP |
|---|---|---|
| PT | 6.1 ms | 0.914 |
| PT + RR | 4.8 ms | 0.942 |
| ReSTIR PT | 12.1 ms | 0.207 |
| ReSTIR PT + RR | 10.2 ms | 0.222 |

- Code22 default is `useRussianRoulette = false`.

### 6.6 Colour-noise reduction (P-§6.3, S-Fig.5)

**Why colour noise appears.** p̂ is scalar (luminance of F) while F is RGB, so chroma is not importance-sampled.

**Fix.** Rao-Blackwellise the random index choice. Since p̂ = |F|, the vector F(Y_i) is already computed during spatial reuse. Accumulate vector-valued resampling weights
**`w_i = m_i(Y_i) · F(Y_i) · W_{X_i} · |∂T/∂X_i|`**
and shade with **`Σ_{i=1}^{M} w_i`** instead of F(Y)·W_Y.
- This is unbiased: E_i[F(Y_i)·Σw/p̂(Y_i)] = Σ_i m_i·F(Y_i)·W_i·J_i.
- Scalar weights still drive the selection and the stored reservoir; vector weights are used only for the displayed colour.
- Spatial neighbors carry uncorrelated chroma noise (P-footnote 7), so the sum averages it out.
- Unlike Wyman & Panteleev 2021, this adds no extra computation.
- S-Fig.5, Zero Day, M = 4 (3 spatial neighbors): FLIP 0.376 → 0.284 at the same 17.0 ms.
- Code22 already has the pattern in its BPR path (`addPathReuseSample`: F += F_in·J·W·mis).

### 6.7 Disocclusion noise: dual motion vectors (P-§6.4, S-Fig.6)

**Technique** (Zeng et al. 2021). When a pixel is disoccluded (its standard back-projection lands on the occluder), reproject it using an alternative MV that assumes the disoccluded surface moves *consistently with the occluder*.

**[INFERENCE, my reading of Zeng 2021]:** prev = x − mv_occluder(x′), where x′ is the standard reprojection. This lands on background next to the occluder's previous position.

**Why it is safe here.** Prior radiance-caching methods get "copy-paste" artifacts from this. ReSTIR PT does not: unbiased path resampling (shift + MIS) prevents pattern cloning. The remaining mild correlation is handled by §4.

**Result.** S-Fig.6, Veach Ajar: 8.8 ms / FLIP 0.259 → 9.1 ms / FLIP 0.231.

**RTXDI extra.** RTXDI instead (also) boosts spatial samples on disocclusion (`numDisocclusionBoostSamples = 8`). That is not in the paper.

---

## 7. Full per-frame pipeline of ReSTIR PT Enhanced

The paper gives the pass *categories* (Table 1: initial sampling, temporal reuse, spatial reuse, "ReSTIR DI and others") and the split of spatial reuse into a pre-pass and a resampling pass. The exact granularity below is **[INFERENCE]** assembled from P-§2–6, S-§5–8 and the Code22 pass list (GeneratePaths, TracePass, TemporalPathRetrace, TemporalReuse, SpatialPathRetrace, SpatialReuse).

| # | Pass | Inputs | Outputs | Notes |
|---|---|---|---|---|
| 0 | Per-frame setup (CPU + small GPU) | light list and emissive triangles; frame index; pairing textures | light-tile buffer (128×1024 light indices + PDFs); per-texture transform (flip/mirror/transpose/offset); previous/current camera; previous transforms | pairing textures are static, generated at load |
| 1 | V/G-buffer + motion vectors (+ dual MVs) | scene, BVH, camera | primary hit (instance/prim/bary), normals, depth, material ID; MV; dual MV for disocclusions | Code22 uses a VBuffer |
| 2 | Initial sampling (path tree) | G-buffer, light tiles, seed | reservoir_cur (64 B): one selected path of length 2..n; its rc vertex from the new criteria (§3.11) | NEE at **x1** and every vertex; RIS NEE (32/B²); RR (initial only); streaming RIS with p̂ = lum(F); UCW includes W^{RIS}·p1/∏q |
| 3a | Temporal: back-project + classify | reservoir_prev, MV/dual MV, dup map_prev, G-buffers (cur, prev) | temporal pairs; compacted replay list (atomic append) | c_Cap = lerp(20, 1, D^0.1) from dup map_prev at the prev pixel |
| 3b | Temporal replay (indirect dispatch) | compacted list | per-pair replay state | only pairs whose rc vertex is beyond x2 or absent |
| 3c | Temporal resample | shifts prev→cur and cur→prev (the latter in the prev-frame domain) | reservoir_temporal | 2022 used generalized Talbot MIS for temporal; confidence = min(c_Cap, c_prev) + 1 |
| 4a | Spatial: partner lookup + classify | reservoir_temporal, G-buffer, N pairing textures + transforms | per-(pixel, slot) validity; compacted replay list | N = 3, σ = 16 |
| 4b | Spatial replay (indirect) | list | replay state per slot | |
| 4c | Spatial shift pre-pass | reservoir_temporal, replay state | slot[t][p] = shift of p's own path into partner q_t: F_{q_t}(·)·J, J | one shift per (pixel, slot) instead of two |
| 4d | Spatial resample | own slots, partners' slots, partner reservoirs | reservoir_out (= next frame's reservoir_prev); shading colour = Σ_i w_i (RGB) | pairwise MIS (§2.2); c_out = Σc |
| 5 | Duplication map | per-pixel initRandomSeed of reservoir_out | D map (17×17 count / 288) | used by the next frame's 3a |
| 6 | Output | colour | denoiser (NRD / DLSS-RR in the paper's context) → tonemap | Enhanced removes the ReSTIR DI pass |

In Table 1, "ReSTIR DI and others" drops from 5.05 ms to **1.00 ms** after unification (the ReSTIR DI pass is gone), then rises to 1.83 ms with all improvements. The paper does not break down what "others" contains.

---

## 8. Results (P-§7, S-§9–12)

### 8.1 Setup

- NVIDIA RTX 5880 Ada and an AMD Threadripper PRO 3975WX.
- Falcor, starting from the public 2022 ReSTIR PT code.
- 1920×1080, 1 spp (one path tree per pixel).
- Metric: HDR-FLIP (Andersson et al. 2021), plus MSE for convergence.
- Defaults: c_Cap = 20; 3 spatial neighbors in a 30 px radius (random) or 3 pairing textures with σ = 16; α_min = 0.2; c = 0.02; decorrelation α = 0.1 and c_Cap^min = 1; NEE 32 candidates at x1.
- Max path length and RR q are not stated **[UNVERIFIED]**. Code22 defaults to maxSurfaceBounces = 9.

**Scenes:** Spaceship, Kitchen, Veach Ajar (plus a metallic-wall variant), Carousel, Opera House, Bathroom, Tower Bridge, San Miguel, Watercolor, Zero Day, Crown, Burger Restaurant, Bistro Exterior.

### 8.2 Table 1: cumulative optimisation ladder (ms, average of Veach Ajar, Carousel, Opera House, Bathroom)

| Method (cumulative) | Total | Initial | Temporal | Spatial | ReSTIR DI + others |
|---|---|---|---|---|---|
| Baseline (Lin 2022 code) | 35.73 | 10.59 | 5.30 | 14.79 | 5.05 |
| + Code micro-opt (§6.2.1) | 32.98 | 10.70 | 4.16 | 13.06 | 5.07 |
| + Forced NEE reconnect (§6.2.3) | 29.75 | 11.44 | 3.40 | 9.73 | 5.19 |
| + Replay compaction (§6.2.2) | 26.81 | 11.79 | 2.66 | 6.99 | 5.37 |
| + Paired spatial reuse (§3) | 25.02 | 12.56 | 2.80 | 4.11 | 5.55 |
| + Russian roulette (§6.2.4) | 16.52 | 5.21 | 2.24 | 3.83 | 5.24 |
| + Unify DI & GI (§6.1) | **13.04** | 6.47 | 2.14 | 3.43 | 1.00 |
| + New thresholds (§4), *adds cost* | 14.51 | 6.68 | 2.92 | 3.60 | 1.31 |
| + All improvements (§5, 6.3, 6.4) | **15.53** | 6.77 | 3.20 | 3.73 | 1.83 |

- Cost reductions alone: 2.74×. Everything: 2.30×. The quality features add 19% over the fastest variant.
- The new thresholds mostly add temporal and initial cost. This is plausibly because they enable more replay or reconnection work and extra PDF evaluations **[INFERENCE]**.

### 8.3 Per-scene frame times (S-Table 1)

| Scene | Baseline | Code micro-opt | Forced NEE | Compaction | Paired | RR | Unify | New thresholds | All |
|---|---|---|---|---|---|---|---|---|---|
| Veach Ajar | 21.99 | 19.34 | 15.75 | 13.02 | 12.10 | 10.20 | 7.64 | 8.74 | 9.89 |
| Carousel | 46.76 | 43.82 | 40.55 | 36.51 | 32.85 | 20.16 | 17.47 | 19.43 | 20.70 |
| Opera House | 49.29 | 46.87 | 43.61 | 40.30 | 38.63 | 22.14 | 16.69 | 18.67 | 19.54 |
| Bathroom | 24.87 | 21.90 | 19.10 | 17.41 | 16.49 | 13.59 | 10.35 | 11.20 | 11.99 |

Per-pass numbers are in S-Table 1 if needed. For example, the Carousel initial pass goes 17.74 → 6.29 ms with RR, then 8.98 ms after unification.

### 8.4 Resolution scaling (S-Table 2, frame time in ms)

| Method | Resolution | Veach Ajar | Carousel | Opera House | Bathroom | Avg. scaling |
|---|---|---|---|---|---|---|
| Baseline | 960×540 | 5.75 | 14.28 | 13.34 | 6.53 | |
| Baseline | 1920×1080 | 21.99 | 46.76 | 49.29 | 24.87 | 3.65× |
| Baseline | 3840×2160 | 97.66 | 208.63 | 234.12 | 114.63 | 4.57× |
| Ours | 960×540 | 2.70 | 6.66 | 5.26 | 3.16 | |
| Ours | 1920×1080 | 15.53* | 20.70 | 19.54 | 11.99 | 3.57× |
| Ours | 3840×2160 | 62.34 | 77.99 | 82.10 | 50.13 | 4.03× |

\* **[UNVERIFIED / likely typo]** 15.53 is the 4-scene average from Table 1; the Veach Ajar value there is 9.89. The stated 3.57× matches 9.89, but the stated 4.03× matches 15.53, so one of the numbers is inconsistent.

### 8.5 Headline image comparisons (P-Fig.1, P-Fig.13, S-Fig.13)

| Scene | Baseline | Ours (no decorrelation) | Ours (default) | Speed-up |
|---|---|---|---|---|
| Spaceship | 37.1 ms / FLIP 0.321 | 12.1 ms / 0.256 | 12.6 ms / 0.263 | 2.94× |
| Watercolor | 30.1 / 0.171 | 13.9 / 0.166 | 14.5 / 0.166 | 2.08× |
| Zero Day | 47.1 / 0.439 | 20.3 / 0.407 | 21.4 / 0.414 | 2.20× |
| Crown | 50.0 / 0.483 | 16.0 / 0.436 | 16.4 / 0.444 | 3.05× |

FLIP rises slightly with decorrelation because FLIP is insensitive to correlation but sensitive to brightness bias.

### 8.6 Threshold ablations

- Kitchen near/far (P-Fig.10): near 0.223 vs 0.223; far 0.296 (baseline) vs 0.291 (ours).
- San Miguel caustics (P-Fig.11): 0.339 → 0.322.
- Veach Ajar (P-Fig.12): diffuse 0.185 vs 0.185; metallic walls (roughness 0.3) 0.271 → 0.252.

**Other ablations:** paired reuse (§2.6), decorrelation (§4.4), RR, unification, colour noise and dual MV (§5–6).

---

## 9. Prior works it depends on

| Work | What Enhanced uses |
|---|---|
| **Lin et al. 2022 (GRIS / ReSTIR PT)** | Everything structural: GRIS theory, PSS hybrid shift and Jacobian (P-Eq.2), lobe-indexed paths, pairwise/Talbot MIS with confidence weights, reservoir design, c_Cap = 20 and 3 neighbors defaults, public code as the starting point |
| Bitterli et al. 2020 (ReSTIR DI) | Reservoir/RIS framework; light RIS target without visibility and only the chosen light shadow-tested (P-§6.1); the separate DI pass is *removed* |
| Wyman & Panteleev 2021 (HPG, "Rearchitecting…"; RTXDI) | **Presampled light tiles** (128×1024, 8×8 screen tiles); RTXDI **boiling filter** as the decorrelation baseline; contrasted colour-noise technique |
| Kettunen et al. 2023 (Conditional ReSTIR) | Cited, with Lin 2022, for marginalising over the random index → **§6.6 colour-noise technique**; also cited as an expensive decorrelation alternative (not used) |
| Sawhney et al. 2024 (MCMC mutations) | Cited as an alternative decorrelation; **not used** |
| Zhang et al. 2024 (Area ReSTIR) | Future work only (smooth transition to an enhanced Area ReSTIR) |
| Wyman et al. 2023 (SIGGRAPH course) | Ideal-shift criterion p̂_j(T(ū))·\|∂T/∂ū\| ≈ p̂_i(ū) that motivates §4 |
| Kettunen et al. 2015 (G-PT) | Classic "all three vertices rough" reconnection rule |
| Müller et al. 2021 (NRC) | **Primary ray footprint** R_pri (1/(4π) spread) in P-Eq.5; the path-footprint criterion is compared and rejected |
| Bekaert et al. 2002 / 2003 | N-rooks path reuse (contrasted with pairing textures); footprint density estimation (related) |
| Zeng et al. 2021 | **Dual motion vectors** for temporal reprojection on disocclusion |
| Heitz 2018 | VNDF sampling; reciprocity is only approximate for VNDF (footnote 5) |
| Chao 1982; Talbot 2005 | WRS, RIS |
| Andersson et al. 2021 | HDR-FLIP metric |
| Zeltner et al. 2024 | Neural material demo for the PDF-based threshold (S-Fig.2) |
| Davidovič 2010; Hašan 2009 | Analogy for the rejected Jacobian-bounding threshold |
| Benty et al. 2020 | Falcor framework |

**Code availability.**
- **No official code for the Enhanced paper.** The project page lists paper, supplemental and video (`lin2026restirptenhanced.mp4`) only.
- **Baseline 2022 code:** https://github.com/DQLin/ReSTIR_PT (Falcor 5 / Slang; key files under `Source/RenderPasses/ReSTIRPTPass/`: `PathReservoir.slang`, `SpatialReuse.cs.slang`, `TemporalReuse.cs.slang`, `Shift.slang`, `PathTracer.slang`, `Params.slang`). A good MIS/shift reference. Its license was not checked in this pass **[UNVERIFIED]**.
- **NVIDIA RTXDI 3.x** (commit "ReSTIR PT" on 2026-03-10; "3.1 Update" on 2026-09-03):
  - Docs: `Doc/RestirPT.md` in https://github.com/NVIDIA-RTX/RTXDI. Runtime: https://github.com/NVIDIA-RTX/RTXDI-Library `Include/Rtxdi/PT/*.hlsli`.
  - Implements: footprint reconnection mode (default) with Gaussian-jittered thresholds; PDF-roughness; forced NEE reconnection; duplication map (17×17, /288) with history reduction; 64 B packed reservoirs; decorrelation for DLSS-RR; optional compatibility-guided neighbor selection (Junkins et al.).
  - **Paired spatial reuse not found [UNVERIFIED]**.
  - Defaults (`Source/ReSTIRPT.cpp`): minConnectionFootprint = 0.02 (the threshold is R_pri²·0.02² = 0.0004·R_pri², i.e. **paper-equivalent c ≈ 0.04**, twice the paper's 0.0002); minPdfRoughness = 0.1; maxBounceDepth = 3; maxRcVertexLength = 5; temporal maxHistoryLength 8 (20 with DLSS-RR); spatial samplingRadius 32, numSpatialSamples 1, disocclusion boost 8.
  - **License:** "NVIDIA RTX SDKs LICENSE" / `LicenseRef-NvidiaProprietary`. It restricts copying and derivative works outside SDK terms. **Treat it as a semantic reference only for an independent WebGPU implementation** (not legal advice).
- NVIDIA RTX Kit 2026.2 is reported to add ReSTIR PT to RTXDI (TechPowerUp). Related repos: NVlabs/conditional-restir-prototype, guiqi134/Area-ReSTIR.

---

## 10. Diff: ReSTIR PT (2022) vs ReSTIR PT Enhanced (2026)

| Component | 2022 ReSTIR PT (paper + Code22) | Enhanced (2026) |
|---|---|---|
| Spatial neighbor choice | 3 random neighbors from a precomputed disk-offset sequence; G-§8.3 says 20 px radius (Code22 `mSpatialReuseRadius = 20`); Enhanced cites 30 px as ReSTIR PT's default | 3 **reciprocal pairing textures** (254/230/210), Gaussian offsets σ = 16 px (≈ R = 30), random flip/mirror/transpose/offset per frame |
| Shifts per pixel in spatial reuse | 2N (neighbor→center + center→neighbor for pairwise MIS) | **N** (own path → each partner); results shared by both partners |
| Spatial passes | SpatialPathRetrace (replay for both directions, 256 B/px rcData) + SpatialReuse | Replay over compacted pairs → shift pre-pass → resampling pass |
| Resampling MIS | Defensive pairwise (spatial), generalized Talbot (temporal); confidence-weighted | Same family **[INFERENCE: not restated]**; pairing changes only who computes the terms |
| Reconnection criteria | min(α_{k−1}, α_k) ≥ 0.2 (perceptual linearRoughness, sampled lobe) **and** ‖x_k − x_{k−1}‖ ≥ d_min (world units; 1–5% scene size; code 0.1) | **Dual footprint** min(rayFP, invFP) ≥ (c/100)·R_pri², c = 0.02, **plus** single-vertex α_{x_{k−1}} ≥ 0.2; marginal PDFs; PDF proxy for black-box BSDFs; inverse test skipped for diffuse/emissive x_k |
| Scene dependence | Needs per-scene d_min / α tuning | Scale-free and adaptive to view distance and roughness (c range 0.005–0.04 optimal) |
| NEE-terminated replay | May need light sampling inside replay | **Forced reconnection** to the NEE light vertex if no earlier rc vertex |
| Replay scheduling | Per-pixel threads with warp divergence | **Stream-compacted** pixel–neighbor pair lists |
| Russian roulette | Off by default | On at initial sampling only; PSS source PDF ∏q_i; replay deterministic |
| Direct illumination | Separate ReSTIR DI pass (16 B reservoirs); PT reservoirs only for d ≥ 3 | **Unified**: NEE from x1, d = 2 paths in the same reservoir, DI gets hybrid shift and pairwise MIS |
| NEE sampling | 1 light sample (power-based emissive sampler) | Optional **RIS NEE** via light tiles (128×1024, 8×8 screen tiles, 32/B² candidates); MIS ω1 = M·p1/(M·p1+p2); UCW = W^RIS·p1 |
| Reservoir size | 88 B (+16 B DI), float M | **64 B**: 8-bit M, 16-bit unorm barycentrics, octahedral ω_i, product of Jacobian terms |
| Memory @1080p (double-buffered) | 431 MB | 265 MB |
| Temporal confidence cap | Fixed c_Cap = 20 | **Adaptive** lerp(20, 1, D^0.1) from the duplication map (biased; can be disabled) |
| Shading estimator | F(Y)·W_Y | **Σ_i w_i with RGB weights** (colour-noise reduction) |
| Disocclusion | Standard MV reprojection; history lost | **Dual motion vectors** for reprojection |
| Code style | Branchy | Branches → conditional moves; occupancy 22% → 31–35% |
| Bias | Unbiased | Default slightly **biased** (duplication maps); unbiased with them off |
| Frame time (4-scene average, RTX 5880 Ada, 1080p) | 35.73 ms | 13.04 ms (cost-only) / 15.53 ms (all features) |

---

## 11. Implications for this WebGPU project [INFERENCE throughout]

**Ordering of adoption.**
1. Build the unbiased core first: unified DI/GI path tree, hybrid shift with the footprint criteria, pairwise MIS, paired spatial reuse, RR at initial sampling only, and RGB shading weights.
2. Validate it against Cycles.
3. Only then enable duplication-map decorrelation, and treat it as a *visual* option.
   - Cycles-validation mode must run with decorrelation off.
   - Unbiasedness checks should compare (a) the average of many independent runs of the same static view against the reference (the paper uses 1024 runs, Fig.15), and (b) spatial-only accumulation, which the paper recommends for unbiased convergence.

**Paired reuse is especially attractive without hardware RT.** In a compute-shader BVH on Metal, each shift costs at least one visibility ray plus possible replay rays, so halving the shift count halves the most expensive part of spatial reuse. Build the pairing textures on the CPU and upload them as `rg8sint`; pass the per-frame transforms as uniforms.

**Compaction needs** `atomicAdd` on storage buffers plus `dispatchWorkgroupsIndirect`. Both are core WebGPU. Reset the counters with `clearBuffer` each frame.

**WebGPU limits to plan for:**
- `maxStorageBufferBindingSize`: 64 B/px reservoirs sit exactly at the 128 MiB default at 1080p; request the adapter maximum.
- `maxStorageBuffersPerShaderStage`: default 8; request more if needed.
- `maxComputeWorkgroupStorageSize`: 16 KB default is enough for the 32×32 duplication tile.
- `shader-f16` is available on Apple and is useful for compact slot storage.

**Materials.** glTF roughness is perceptual, like Falcor's linearRoughness, so α_min = 0.2 transfers directly. For the Principled/Blender comparison, make sure the WGSL BSDF matches Cycles' GGX multiscatter vs single-scatter choice (not covered by this paper).

**Lights.** Use area-measure sampling for emissive meshes and quad/disk lights (Jacobian 1). Point and spot lights are pure NEE. With few user-placed lights, light tiles are unnecessary: sample all lights with RIS, or use power-based selection.

**Dynamic scenes (moving lights and camera).** Keep one frame of history: previous camera, TLAS/instance transforms and light buffer. This is needed for temporal inverse shifts and the previous-frame primary footprint.

---

## 12. Open questions and inconsistencies found

1. **P-Eq.3 (n_σ fit)** as printed cannot yield n = 1 at σ = 0.8 (it gives 6) and is non-monotonic. Use n = round(σ²/2), i.e. 128 for σ = 16, or an empirical lookup table.
2. **S-Eq.26 PDF-roughness proxy.** The printed 1/p² ≥ α_min looks mis-scaled against "same α_min" and against RTXDI's 1/sqrt(p) ≥ minPdfRoughness.
3. **RTXDI's default footprint constant** is twice the paper's (0.0004 vs 0.0002 × R_pri²) and adds jitter.
4. **S-Table 2**: the Veach Ajar 1080p "Ours" value (15.53) looks like a typo for 9.89.
5. **P-p.4 wording.** The definitions of p^x_k and of the cosine in G look slipped; use S-§1 and G-Eq.52 semantics.
6. **Unspecified by the paper:**
   - RR survival formula;
   - max path length;
   - how fractional c_Cap maps to the 8-bit integer M;
   - temporal MIS type in Enhanced;
   - whether the light tile is re-picked per bounce;
   - rounding of 32/B²;
   - exact G-buffer rejection thresholds;
   - how the pre-pass stores shift results (layout);
   - the exact replay-compaction partitioning.
7. **FOV/resolution dependence** of the footprint threshold is not discussed (R_pri uses 4π, not the pixel solid angle).
8. **Seed for the duplication test.** The paper says "random seed"; the reservoir has two seeds. initRandomSeed is the natural ID (RTXDI uses `randomSeed`).

---

## 13. Sources

- Paper: https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/lin2026restirptenhanced.pdf (local `scratchpad/restirpt_enhanced_2026.pdf`, text `…/restirpt_enhanced_2026.txt`; Sec.3 text lines 210–273, Sec.4 lines 274–419, Sec.5 lines 420–476, Sec.6 lines 477–564, Sec.7 lines 565–799; page PNGs `scratchpad/pages/enhanced/p-01..p-19.png`)
- Supplemental: https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/lin2026restirptenhanced_supplemental.pdf (streamed, not saved)
- Project page: https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/ ; DOI https://doi.org/10.1145/3804494
- GRIS 2022: https://d1qx31qr3h6wln.cloudfront.net/publications/sig22_GRIS.pdf (local `scratchpad/gris_sig22.pdf`, pages 9–15 consulted)
- Code22: https://github.com/DQLin/ReSTIR_PT (PathReservoir.slang, SpatialReuse.cs.slang, Params.slang, ReSTIRPTPass.h, PathTracer.slang, BxDF.slang)
- RTXDI: https://github.com/NVIDIA-RTX/RTXDI (Doc/RestirPT.md); https://github.com/NVIDIA-RTX/RTXDI-Library (Include/Rtxdi/PT/PathReconnectibility.hlsli, DuplicationMap.hlsli, TemporalResampling.hlsli, SpatialResampling.hlsli, Reservoir.hlsli, ReSTIRPTParameters.h; Source/ReSTIRPT.cpp; LICENSE.txt)
- TechPowerUp news on ReSTIR PT Enhanced / RTX Kit: https://www.techpowerup.com/348358/nvidia-develops-2-3x-faster-real-time-path-tracing-with-better-image-quality
