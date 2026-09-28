# GRIS / ReSTIR PT (Lin, Kettunen, Bitterli, Pantaleoni, Yuksel, Wyman — SIGGRAPH 2022): complete mathematical deep dive for a WebGPU/WGSL implementation

> **Location note.** The orchestrator asked for this report at
> `/private/tmp/claude-502/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-ea73-491f-b7b7-6d0aab1ffce1/scratchpad/research/gris-math.md`.
> This session runs in **plan mode**, where the only writable file is this plan file. The content below is the complete report. Copy it to the intended path verbatim.

---

## 0. Sources, provenance, conventions

| Tag | Source | How it was read |
|---|---|---|
| **[P]** | Main paper, *Generalized Resampled Importance Sampling: Foundations of ReSTIR*, ACM TOG 41(4) Art. 75, 2022. Local copy `scratchpad/gris_sig22.pdf` (23 pages). | All 23 page PNGs (`scratchpad/pages/gris/p-01.png … p-23.png`) were read visually for the equations. Line numbers refer to `scratchpad/gris_sig22.txt`, which is garbled for math but usable for prose. |
| **[S]** | Supplemental, *Supplementary Document: GRIS* (12 pp.), `https://graphics.cs.utah.edu/research/projects/gris/GRIS_supplemental.pdf`. | Fetched read-only. Text was extracted with ghostscript `txtwrite` XML plus surrogate-pair repair, with no files written. Display equations came out partly scrambled, but prose, the struct listing and most equations are reliable. Anything uncertain is flagged. |
| **[C]** | Authors' reference code, `github.com/DQLin/ReSTIR_PT` (Falcor 4.4 / Slang), HEAD commit `8d12332228eb64bc234e27c6f7e0913a926285ab` (2025-07-20). Files under `Source/RenderPasses/ReSTIRPTPass/`. | Fetched with `curl` to stdout. The `file:line` references below are for that commit. |

Tags in the text:
- **[P §x, p.75:N, Eq. n]**: what the paper says. Page `75:N` is PDF page N.
- **[S §S.x, Eq. S.n]**: what the supplemental says.
- **[C file:line]**: what the reference code does.
- **[INFERENCE]**: my own derivation or interpretation, not stated by a source.
- **[UNVERIFIED]**: uncertain transcription or claim. Check before relying on it.

Notation follows Table 1 of the paper [P p.75:3]:

| Symbol | Meaning |
|---|---|
| `x, y` | a generic input to a function |
| `x̄, x_i` | a path, and vertex `i` on the path |
| `Ω_i` | domain from which sample `X_i` is drawn |
| `Ω` | domain of integration of `f` |
| `X_i` | input sample for RIS, often a sequence `(X_i)_{i=1}^M` |
| `Y` | sample selected by RIS (`Y = X_s` in the simple case, `Y = T_s(X_s)` in general) |
| `M, N` | number of input and output samples of RIS |
| `p_X(·)` | density of random variable `X` |
| `p̂(·)` | unnormalized target function (we want `Y ∝ p̂`) |
| `p̄(·)` | normalized target, `p̄ = p̂/‖p̂‖₁` (`‖·‖` without subscript means the 1-norm, footnote 2) |
| `f(·)` | integrand (e.g. path contribution function) |
| `g_i(·)` | contribution function for `X_i ∈ Ω_i` to integrate `f` in `Ω` |
| `W_i` | unbiased contribution weights, which estimate reciprocal PDFs |
| `w_i` | resampling weights; RIS selects index `i` with probability `w_i/Σ_j w_j` |
| `c_i` | contribution MIS weights (the MIS weights of prior work) |
| `m_i` | the new *resampling* MIS weights |
| `T_i(·)` | shift mapping, `Ω_i → Ω` |
| `|∂T_i/∂x|` | Jacobian determinant of `T_i` |
| `p̂_{←i}(·)` | "`p̂` from `i`": `p̂_i` pulled into `Ω` through the shift |
| `C` | various constants (bounds) |
| `R, |R|` | set of canonical sample indices, and its size |

---

## 1. Background: RIS, unbiased contribution weights, streaming reservoirs, and why naive reuse fails

### 1.1 Basic RIS with i.i.d. candidates [P §3.1, p.75:4, Eq. 1–3]

Inputs are i.i.d. `X_1..X_M ∈ Ω` with a known PDF `p`. Choose index `s` with `Pr[s = i] = w_i / Σ_{j=1}^M w_j` and output `Y = X_s`. Prior work uses `w_i = p̂(X_i)/p(X_i)`. Because the weights are relative, the paper folds in constants:

```
(1)   w_i = (1/M) · p̂(X_i) · W_i ,      W_i = 1 / p(X_i)
```

The PDF `p_Y` of the selected sample is intractable. Its **unbiased contribution weight** can be used in place of `1/p_Y(Y)` [P Eq. 2; = Bitterli 2020 Eq. 12]:

```
(2)   W_Y = (1 / p̂(Y)) · Σ_{i=1}^M w_i
```

If `p_Y > 0` wherever `f > 0` (i.e. `supp f ⊂ supp Y`):

```
(3)   ∫_Ω f(x) dx = E[ f(Y) · W_Y ]
```

Under suitable constraints `p_Y → p̄`, and `Var[f(Y)W_Y]` tends to the variance you would get if `Y` had PDF exactly `p̄`. So `p̂ ∝ f` gives an asymptotically zero-variance estimator [P p.75:5 top].

### 1.2 Differently distributed candidates: "resampling MIS" [P §3.2, Eq. 4–6]

If each `X_i` has its own PDF `p_i`, you need a partition of unity `m_i ≥ 0` with

```
(4)   Σ_{i=1}^M m_i(x) = 1     for all x in supp p̂
```

Talbot [2005] proposes the balance-heuristic analogue:

```
(5)   m_i(x) = p_i(x) / Σ_{j=1}^M p_j(x)
```

and the `1/M` in Eq. 1 is replaced by `m_i`:

```
(6)   w_i = m_i(X_i) · p̂(X_i) · W_i ,     W_i = 1 / p_i(X_i)
```

Eq. 3 still holds with `W_Y` from Eq. 2, provided at least one `p_i` covers each `x ∈ supp p̂`.

### 1.3 Unbiased contribution weights as a first-class concept [P §4.2, Def. 4.1, Eq. 8–9; Thm A.1, Eq. 55–56]

**Definition 4.1.** An unbiased contribution weight `W ∈ ℝ` for a random variable `X ∈ Ω` is any real random variable with

```
(8)   E[ f(X) · W ] = ∫_{supp(X)} f(x) dx      for every integrable f: Ω → ℝ
```

`f(X)W` generalizes `f(X)/p(X)`. If `p` is tractable you can use `W = 1/p(X)`. The integral is naturally restricted to `supp(X)`. Equivalently (Theorem A.1, proof in [S §S.5.1, Eq. S.78–S.80]):

```
(9)   E[ W | X ] = 1 / p_X(X)
```

So "any unbiased estimator of the reciprocal marginal PDF is an unbiased contribution weight, and vice versa." Other examples are the conditional PDFs of continuous MIS [West et al. 2020] [P p.75:6].

**Degenerate case** [P p.75:6 right]. If all `w_i = 0`, no sample is selected and the contribution is zero. Formally, output a null sample `Y_∅` with `p̂(Y_∅) = f(Y_∅) = 0`. `W_{Y∅}` is irrelevant and can be set to 0. *(Implementation: an empty reservoir has `W = 0` and must evaluate to zero radiance.)*

### 1.4 Streaming reservoir sampling [P §2.1, p.75:3]

Chao [1982]: a reservoir stores the selected sample, the stream length `M`, and the running `Σ w_i`. Each new element `X_i` replaces the selected sample with probability `w_i / Σ_{j≤i} w_j`. Combined with resampling, this gives single-pass, constant-memory RIS. ReSTIR [Bitterli 2020] chains reservoir resampling across pixels and frames. It alternately generates new independent samples per reservoir (per pixel) and reuses samples between similar reservoirs, i.e. between domains.

Code equivalent [C PathReservoir.slang:290-310, `add`]:

```
M += 1; w = lum(F_in)/p; if (w==0||nan) return; weight += w;
if (u * weight <= w) { F = F_in; /* select */ }
```

and `merge` [C PathReservoir.slang:320-353] with `w = lum(F_shifted) · J · M_in · W_in · misWeight`. That is the old constant-`1/M` ReSTIR convention, finalized by `W = Σw/(p̂·M)` (`finalizeRIS`, :435-441). `mergeWithResamplingMIS` (:355-388) omits `M_in`, uses the GRIS `m_i`, and finalizes with `W = Σw/p̂` (`finalizeGRIS`, :443-450).

### 1.5 Why naive cross-pixel reuse is biased or fails to converge

What the sources say:
- **[P §3.3, p.75:5]** Talbot's theory assumes independent samples in a *shared* domain `Ω`. ReSTIR stretches both assumptions and "may not retain any theoretical convergence guarantees". Seemingly innocuous changes can make correlated reuse converge to a wrong result.
- **[P Fig. 2, p.75:2]** A two-pixel toy where ReSTIR without an M-cap is *unbiased* but converges to the *wrong* result. Each iteration adds only two new independent samples. The other reused samples are duplicates, so correlation, not bias, breaks convergence. The rendered analogue is Fig. 10b.
- **[P §4.3, footnote 3]** Mapping samples between domains changes densities by the transformation law. With `y = T(x)`: `p_Y(y) = p_X(x) · |∂T/∂x|^{-1}`. Omitting `|∂T/∂x|` is therefore wrong.
- **[P §7.1, p.75:12]** A shift must be invertible. "Removing paths from a map's domain may cause noise and waste computation, but neglecting bijectivity introduces significant bias."
- **[P §5.6 "Constant resampling weights", p.75:10]** `m_i(y) = 1/M` satisfies Eq. 20 only when every realizable sample could have been sampled from *all* domains with positive PDF. That is generally false, except when all inputs are canonical. Otherwise convergence to `p̄` and the variance results are lost. Bias can be removed with proper contribution MIS [Bitterli 2020].
- **[P App. B "On Visibility", p.75:23]** A target `p̂_i` without visibility creates paths with positive `p̂_i` that are never sampled (they are occluded). Then `supp p̂_i ⊄ supp X_i`, the sample becomes non-canonical, and `supp p̂ ⊂ supp Y_M` (the premise of Thm A.2) fails. ReSTIR PT always includes visibility between vertices.
- **[P footnote 12, p.75:14]** Ouyang et al. [2021] (ReSTIR GI) list bias sources such as assuming Lambertian scattering at reconnection vertices. GRIS removes all of these.

**[INFERENCE] Concrete failure modes to avoid in the WGSL implementation:**
1. Reusing a neighbor's path with the identity map, i.e. adding its stored radiance without re-evaluating the throughput from *this* pixel's primary hit. This integrates the wrong integrand.
2. Reconnecting without the Jacobian (Eq. 52/54). This gives a density mismatch and systematic energy error, worst at geometric discontinuities and near-field reconnections.
3. `1/M` weights when domains differ (occlusion, different BSDF support, back-facing reconnection). Paths reachable from only some pixels get the wrong weight, which typically darkens or brightens edges.
4. Non-invertible shifts, e.g. the hybrid shift where the offset path would pick a different reconnection vertex than the base path. This double-counts or drops paths.
5. Evaluating the target of a shifted sample with a stale suffix (moving lights) while pretending it is the current-frame integrand. This is biased toward the old frame.

---

## 2. GRIS theory [P §4, p.75:5–8]

### 2.1 Overview, and the one-line algorithm [P §4.1, Eq. 7; Fig. 3]

Inputs `X_i ∈ Ω_i` may be correlated and come from different domains. Each is paired with an unbiased contribution weight `W_i ∈ ℝ` that replaces `1/p_i(X_i)`. The inputs may themselves be previous resampling outputs: their `p_i` is intractable but `W_i` is tractable (Eq. 2). Samples are mapped into `f`'s domain by shift maps `T_i : Ω_i → Ω`, and the resampling weights become

```
(7)   w_i = m_i(T_i(X_i)) · p̂(T_i(X_i)) · W_i · |∂T_i/∂X_i|
```

The output is `Y = T_s(X_s)`. `W_Y` is again given by Eq. 2.

**Transcription of Fig. 3 [P p.75:4]**

| RIS step | Talbot et al. 2005 (identically distributed) | Talbot 2005 (differently distributed) | **GRIS** (correlations & different source domains) |
|---|---|---|---|
| (1) generate `M` candidates `(X_1..X_M)` | same domain: `X_i ∈ Ω`, same PDF `p` | same domain: `X_i ∈ Ω`, different PDFs `p_i` | arbitrary domains: `X_i ∈ Ω_i`; intractable `p_i` are OK |
| (2) unbiased contribution weights `W_i` | `W_i = 1/p(X_i)` | `W_i = 1/p_i(X_i)` | `W_i` must unbiasedly estimate `1/p_i(X_i)` |
| (3) resampling weights `w_i` | `w_i = (1/M) p̂(X_i) W_i` | `w_i = m_i(X_i) p̂(X_i) W_i` | `w_i = m_i(T_i(X_i)) p̂(T_i(X_i)) W_i |∂T_i/∂X_i|` |
| (4) select `s ∝ w_i`, output `Y` in `Ω` | `Y = X_s` | `Y = X_s` | `Y = T_s(X_s)` |

"The unbiased estimate for the integral of `f` is `f(Y)W_Y` in all cases, with `W_Y` defined in Eq. 2."

### 2.2 Unbiased integration with arbitrary weights [P §4.2, Eq. 10–12]

Assign each `X_i` a **contribution function** `g_i : Ω_i → ℝ` that is evaluated when `s = i`. With `p_s(i) = w_i / Σ_j w_j` (footnote 4: requires `w_i > 0` whenever `g_i(X_i) ≠ 0`; lifted later):

```
(10)  E[ g_s(X_s) W_s / p_s(s) ] = E[ Σ_{i=1}^M g_i(X_i) (p_s(i)/p_s(i)) W_i ]
                                  = Σ_{i=1}^M ∫_{supp(X_i)} g_i(x_i) dx_i
```

Special case: all `X_i` in `Ω` with support `S`, all `w_i > 0` on `S`, and `g_i = f/M`:

```
(11)  E[ (1/M) f(Y) (Σ_j w_j / w_s) W_s ] = ∫_{supp(Y)} f(x) dx
(12)  W_Y = (1/M) · (Σ_{j=1}^M w_j / w_s) · W_s
```

### 2.3 Shift mappings and the unbiasedness conditions [P §4.3, Def. 4.2, Eq. 13–18]

**Definition 4.2.** A shift mapping `T_i` from `Ω_i` to `Ω` is a **bijective function from a subset `D(T_i) ⊂ Ω_i` onto its image `I(T_i) ⊂ Ω`.** Bijectivity is required only on a sub-domain, and the shift may return "undefined".

Choose contribution functions

```
(13)  g_i(x) = c_i(y_i) · f(y_i) · |∂T_i/∂x| ,    y_i := T_i(x)
```

Here the `c_i : Ω → ℝ` are **contribution MIS weights** with `Σ_{i=1}^M c_i(y) = 1` for `y ∈ Ω`. In principle this gives

```
(14)  Σ_{i=1}^M ∫_{Ω_i} g_i(x) dx = ∫_Ω f(x) dx
```

but Eq. 13 is undefined for `x ∉ D(T_i)`. Set `g_i(x) = 0` there, and fix the `c_i` to compensate.

Assumption on `w_i`: `w_i > 0` **iff** `X_i ∈ D(T_i)` and `p̂(Y_i) > 0`. Under this assumption each possible `Y` lies in `supp p̂` and is sampleable as `Y = T_i(X_i)` by one or more `X_i` with positive PDF, and vice versa (footnote 5: if `supp X_i` is larger than `D(T_i)`, set `T_i(supp X_i) := T_i(supp X_i ∩ D(T_i))`):

```
(15)  supp Y = supp p̂ ∩ ⋃_{i=1}^M T_i(supp X_i)
```

Later, `supp p̂ ⊂ supp Y` is assumed, which implies `supp Y = supp p̂`. Substituting `g_i` into Eq. 10:

```
(16)  E[ c_s(Y) f(Y) |∂T_s/∂X_s| (Σ_j w_j / w_s) W_s ] = ∫_{supp(Y)} f(x) dx
```

This holds if the contribution MIS weights satisfy, **for all `y ∈ supp Y`**,

```
(17)  Σ_{i=1..M : y ∈ T_i(supp X_i)}  c_i(y) = 1
```

"Every realizable `y`, possibly from multiple `Ω_i`, must be covered exactly once in total" (Fig. 4). Only `c_i ≥ 0` allows chaining multiple GRIS passes. The resulting weight is

```
(18)  W_Y = c_s(Y) · ( W_s |∂T_s/∂X_s| ) · [ Σ_{j=1}^M w_j / w_s ]
```

`f(Y)W_Y` estimates the integral of `f` over `supp Y`, and `W_Y` is an unbiased estimate of `1/p_Y(Y)`.

> **Boxed [P p.75:7]:** "This specifies when generalized RIS can integrate an arbitrary function `f`: when the supports of random variates `X_i` (mapped to `Ω` via `T_i`) together cover the support of `f`."

This is automatically true if one sampling domain, say `Ω_1`, is `f`'s domain, uses the identity shift `T_1(x) = x`, and generates `X_1` with a sampler for `f` (`p(x_1) > 0` wherever `f(x_1) > 0`). Such samples are called **canonical**. Since `p_{X_1}` is known, `W_1 = 1/p_{X_1}(X_1)` can be used.

**Relaxing constraints ★ [P p.75:7 bottom].** The condition "`w_i > 0` when `p̂(Y_i) > 0`" can be relaxed to also allow `w_i = 0` when `c_i(Y_i) = 0` or `W_i = 0` (the expectation is unchanged). Eq. 17 must then be guaranteed explicitly on `supp p̂ ∩ ⋃_i T_i(supp X_i)` for Eq. 15 to hold. Proofs are in [S §S.7.1, Eq. S.107–S.127]. Using the `w_i` of §2.4 removes the need for these constraints.

### 2.4 Resampling weights with resampling MIS, and the final `W_Y` [P §4.4, Eq. 19–25]

```
(19)  w_i = { m_i(T_i(X_i)) · p̂(T_i(X_i)) · W_i · |∂T_i/∂X_i| ,   if X_i ∈ D(T_i)
            { 0 ,                                                  otherwise
```

- `w_i` are normalized into probabilities, so they must be non-negative. Hence `m_i ≥ 0` and `W_i ≥ 0` [S §S.7.2, Eq. S.128–S.131].
- `w_i = 0` outside `supp p̂`.
- Requirement on `m_i`, for all `y ∈ supp Y`:

```
(20)  Σ_{i=1..M : y ∈ T_i(supp X_i)}  m_i(y) = 1 ,     m_i ≥ 0
```

- Unbiasedness additionally requires **`m_i(y) > 0` wherever `c_i(y) ≠ 0`**, so the `m_i` do not invalidate the `c_i` partition [S §S.7.3, Eq. S.132–S.138].

Substituting Eq. 19 into Eq. 18:

```
(21)  W_Y = [ c_s(Y) / m_s(Y) ] · (1/p̂(Y)) · Σ_{j=1}^M w_j
```

The ratio `c_s/m_s` adds variance even if `Σ w_j` were constant. The paper shows **`c_i = m_i` is ideal**, giving the GRIS unbiased contribution weight used from then on:

```
(22)  W_Y = (1/p̂(Y)) · Σ_{j=1}^M w_j
```

Footnote 6: this skips the `1/M` often seen in RIS/ReSTIR formulas, because the `w_j` already contain it through `m_i`. Choosing `m_i = 1/M` recovers the prior formulas.

**Convergence statement (leads to Thm A.2).** Consider resampling results `Y_M` with `supp p̂ ⊂ supp Y_M` as `M` grows. If

```
(23)  Var[ Σ_{i=1}^M w_{M,i} ]  →  0     (M → ∞)
```

then `p̄(Y)/p_Y(Y) → 1` in the mean-square sense. Footnote 7: `E[ |p̄(Y_M)/p_Y(Y_M) − 1|² ] → 0`. Also

```
(24)  E[ Σ_i w_i ] = E[ p̂(Y) W_Y ] = ∫_{supp Y} p̂(y) dy = ‖p̂‖
(25)  W_Y = (1/p̂(Y)) Σ_j w_j ≈ ‖p̂‖/p̂(Y) = 1/p̄(Y)       for large M      [boxed]
```

Eq. 23 is strong: convergence may not be pointwise, but the probability of any given error goes to 0 and every subset of `Ω` asymptotically gets the right ratio of samples.

**[INFERENCE] Checklist for "our GRIS pass is unbiased":**
- (a) Every `T_i` is a bijection on its declared domain. The code returns 0 (undefined) whenever a path falls outside `D(T_i)`, and the *inverse* test uses the same criteria.
- (b) `|∂T_i/∂X_i|` is correct in the measure you work in.
- (c) The `m_i` satisfy Eq. 20 *pointwise*. They must be evaluated with the **same shifts** (in the reverse direction) that define the domains.
- (d) `m_i > 0` wherever `y` is reachable from `i`.
- (e) At least one canonical sample covers `supp p̂`, which needs visibility in `p̂`.
- (f) `W_i ≥ 0`.
- (g) The target of the *output* domain is used, i.e. the target in the current frame and pixel.

---

## 3. Resampling MIS weights in full [P §5.5–5.6, Eq. 35–40; S §S.1, Eq. S.1–S.58; Thm A.4]

### 3.1 Canonical samples [P §5.5, Def. 5.2]

**Definition 5.2.** Input `X_i ∈ Ω_i` is canonical if its domain is `Ω`, it uses the identity shift `T_i(x) = x`, uses `p̂_i = p̂`, and covers `supp p̂` (i.e. `supp p̂ ⊂ supp X_i`).

> **Boxed [P p.75:9]:** "`R` = set of canonical indices, `|R|` = their number. If canonical sample count increases sufficiently as the total input count increases, the MIS weights of §5.6 guarantee asymptotic convergence of `p_Y` to `p̄` when resampling from multiple domains."

Motivation: resampling gives up access to the PDFs of the other strategies. Robust MIS weights are therefore built from the non-negative unnormalized targets `p̂_i` as proxies for `p_{X_i}`. Samples that directly target `p̂` in `Ω` cover `supp p̂` as often as convergence needs.

### 3.2 "`p̂` from `i`" [P Eq. 35]

```
(35)  p̂_{←i}(y) = { p̂_i( T_i^{-1}(y) ) · |∂T_i^{-1}/∂Y_i| ,   if y ∈ T_i(supp X_i)
                  { 0 ,                                    otherwise
```

Evaluate the proxy PDF `p̂_i` at the pre-image `x = T_i^{-1}(y)` in the original domain `Ω_i`, times the Jacobian of the inverse shift. This is the density of `y` "as seen from domain `i`". For a canonical `i`, `p̂_{←i} = p̂`.

**[INFERENCE] Practical evaluation.** There are two cases.
- **The sample came from `i`** (`y = T_i(x_i)`): then `p̂_{←i}(y) = p̂_i(x_i) / |∂T_i/∂x_i|`. No extra shift is needed; you have `p̂_i(x_i)` stored and `J` from the forward shift. The code does this [C SpatialReuse.cs.slang:306, :395; TemporalReuse.cs.slang:192]: `lum(neighborReservoir.F) / dstJacobian * M_i`.
- **The sample came from elsewhere** (`y` in the central domain): you must *shift `y` into `Ω_i`* with the shift from `Ω` to `Ω_i`, i.e. the inverse of `T_i`. Then `p̂_{←i}(y) = p̂_i(T_i^{-1}(y)) · |∂T_i^{-1}/∂y|`. The code calls `computeShiftedIntegrand(dst = neighbor, src = central)` and multiplies the target by the returned Jacobian [C SpatialReuse.cs.slang:320-323, :372-375]. This costs **one extra shift, and one or more rays**, per pair.

### 3.3 Generalized Talbot MIS [P Eq. 36; derivation S §S.1.1, Eq. S.1–S.7]

```
(36)  m_i(y) = p̂_{←i}(y) / Σ_{j=1}^M p̂_{←j}(y)
```

This is the balance heuristic over the possible sources of `Y`. Talbot's form is recovered for independent samples in one domain (`Ω_i = Ω`, `T_i = id`) with exact PDFs `p_k` in place of `p̂_{←k}`.

Derivation sketch [S]: require `w_i ≤ C̃_i`. From `w_i = m_i(Y_i) p̂(Y_i) W_i |∂Y_i/∂X_i|` and `p̂_i(X_i) W_i ≤ C_i`, you get `m_i(Y_i) ≤ (C̃_i/C_i) · p̂_i(X_i) |∂Y_i/∂X_i|^{-1} / p̂(Y_i) = (C̃_i/C_i) · p̂_{←i}(Y_i)/p̂_{←j}(Y_i)` for any canonical `j` (S.5–S.6). Symmetrizing the denominator over all `j` and choosing `C̃_i = C_i` gives S.7 = Eq. 36.

**Cost:** evaluating `m_i` for every candidate needs `p̂_{←j}(T_i(X_i))` for all `j`. That is **O(M²) shifts**.

### 3.4 Generalized pairwise MIS (uniform and defensive) [P Eq. 37–38; S §S.1.2, Eq. S.8–S.23]

This generalizes Bitterli's [2021] pairwise MIS, originally for one canonical sample and domain (`|R| = 1`, `Ω_i = Ω`, `T_i = id`), to arbitrary domains. MIS is applied only to *pairs* of target functions: each involves `p̂` and one `p̂_{←i}` (non-canonical `i`), or an average over pairs `(p̂, p̂_{←j})` (canonical `i`). **Cost: O(M·|R|)** instead of O(M²).

**General family [S Eq. S.13–S.14]** (parameters `α > 0`, `0 ≤ x ≤ 1`, independent of `i`):

```
(S.13) m_i(y) = (1/|R|) · ( 1 − x + (x/(M−|R|)) · Σ_{j∉R} p̂(y) / ( p̂(y) + α·p̂_{←j}(y) ) )   if i ∈ R
(S.14) m_i(y) = (x/(M−|R|)) · α·p̂_{←i}(y) / ( p̂(y) + α·p̂_{←i}(y) )                        if i ∉ R
```

- `x = 0` means always pick a canonical sample (`m_i = 1/|R|` for `i ∈ R`, 0 otherwise).
- `x = 1` is the "fundamental" scheme parametrized by `α` (S.15–S.16).
- **Uniform case:** requiring `m_1 = … = m_M = 1/M` when all `X_i` are i.i.d. with `p̂_i = p̂` gives `α = M/|R| − 1` (S.17–S.19).

**Uniform pairwise [P Eq. 37 = S.20–S.21]:**

```
(37)  m_i(y) = { (1/(M−|R|)) · Σ_{j∉R}  p̂(y) / ( |R|·p̂(y) + (M−|R|)·p̂_{←j}(y) ) ,   if i ∈ R
               {  p̂_{←i}(y) / ( |R|·p̂(y) + (M−|R|)·p̂_{←i}(y) ) ,                     if i ∉ R
```

**Defensive pairwise [P Eq. 38 = S.22–S.23].** This keeps `α = M/|R| − 1` and chooses `x = (M−|R|)/M`, so canonical weights are never below `(1−x)/|R|` and non-canonical weights never exceed `x/(M−|R|)`. The two bounds are equal: `1/M`.

```
(38)  m_i(y) = { 1/M + (1/M) · Σ_{j∉R}  p̂(y) / ( |R|·p̂(y) + (M−|R|)·p̂_{←j}(y) ) ,          if i ∈ R
               { ((M−|R|)/M) · p̂_{←i}(y) / ( |R|·p̂(y) + (M−|R|)·p̂_{←i}(y) ) ,             if i ∉ R
```

"Slightly less efficient but often more robust." **ReSTIR PT uses Eq. 38 for spatial reuse** [P §8.3].

**[INFERENCE] Specialization used by ReSTIR PT (`|R| = 1`, `k = M−1` neighbors), with confidence weights.** Multiply `p̂` by `M_c` and `p̂_{←i}` by `M_i` (§3.6). Then:

```
m_c(y)  = (1/(k+1)) · [ 1 + Σ_{i=1}^k  (M_c·p̂_c(y)/k) / ( M_i·p̂_{←i}(y) + M_c·p̂_c(y)/k ) ]      (canonical, evaluated at y = X_c)
m_i(y)  = (1/(k+1)) ·        M_i·p̂_{←i}(y)   / ( M_i·p̂_{←i}(y) + M_c·p̂_c(y)/k )              (neighbor i, evaluated at y = T_i(X_i))
```

This is exactly what the code computes [C SpatialReuse.cs.slang:377-379 (canonical accumulation), :395-396 (neighbor weight), :408 (final division by `validNeighborCount+1`)].

**Partition-of-unity check [INFERENCE].** For any fixed `y`:

```
m_c + Σ m_i = (1 + Σ_i [c/(n_i+c) + n_i/(n_i+c)])/(k+1) = 1
```

with `c = M_c p̂_c/k` and `n_i = M_i p̂_{←i}`. It holds for any positive constant in place of `k` inside `c`. The code uses `neighborCount` inside and `validNeighborCount+1` outside, which is still a valid partition when some neighbors are rejected, provided rejected neighbors are skipped consistently in both roles (they are).

### 3.5 Constant weights [P §5.6]

`m_i(y) = 1/M` is valid only if every realizable sample could come from every domain, or if all inputs are canonical, e.g. inputs produced by GRIS that itself used at least one canonical sample and a proper MIS scheme. Otherwise convergence guarantees are lost (footnote 8: that condition alone does not guarantee finite variance). Bias can still be removed with contribution MIS [Bitterli 2020]. The code keeps `Constant / ConstantBinary / ConstantBiased` variants [C Params.slang:89-99; SpatialReuse.cs.slang:415-531]. For the hybrid shift the code *forces* Pairwise (spatial) and Talbot (temporal) [C ReSTIRPTPass.cpp:688-694].

### 3.6 Confidence weights (M-counts) [P §6.2]

A reservoir `r` stores a path `X_r`, its weight `W_r`, and a count `M_r`. A merge of `r_1` and `r_2` resamples `X_{r_m}` from `X_{r1}` and `X_{r2}` "as if" from the concatenated inputs, so `M_{r_m} = M_{r1} + M_{r2}`. The paper argues this sample-count reading is too strict. A merge is RIS from canonical `X_{r1}` and `X_{r2}` with **resampling MIS weights `m_{r_i}(y) = M_{r_i}/(M_{r1}+M_{r2})`**, i.e. the `M_r` are relative **confidence weights**. ReSTIR even caps `M_r` to `M_c`, which invalidates the count interpretation.

> "Reservoir merging generalizes to weighted GRIS with proper MIS weights simply by **multiplying the `p̂` and `p̂_←` in the MIS formulas by the corresponding reservoir's `M_r`**. The result is stored in `X_{r_m}`, and `M_{r_m} = min(M_c, Σ_j M_{r_j})`."

**[C] divergence to note.** The code does not cap the output `M`. It caps the *incoming temporal history* relative to the current reservoir: `temporalReservoir.M = min(gTemporalHistoryLength · currentM, temporalReservoir.M)` with default history length 20 [C TemporalReuse.cs.slang:139; ReSTIRPTPass.h:139]. Spatial reuse sums `M` without a cap [C PathReservoir.slang:360 via `mergeWithResamplingMIS`].

### 3.7 Tractable-PDF variants [P §5.6 "Tractable PDFs"; S §S.1.4, Eq. S.39–S.42]

If all inputs have known PDFs `p_i`, use `p_{←i}(y) = p_i(T_i^{-1}(y)) |∂T_i^{-1}/∂y|` (S.39) in place of `p̂_{←i}`. Talbot then becomes `m_i = p_{←i}/Σ_j p_{←j}` (S.40). In pairwise MIS, replace the `p̂(y)` terms by the PDF `p_c` of a *fixed canonical importance sampler* `c` that is reasonable for `p̂` (`p̂ ≤ C_c p_c`) (S.41–S.42):

```
m_i(y) = (1/(M−|R|)) Σ_{j∉R} p_c(y) / ( |R| p_c(y) + (M−|R|) p_{←j}(y) )   (i ∈ R)
m_i(y) = p_{←i}(y) / ( |R| p_c(y) + (M−|R|) p_{←i}(y) )                   (i ∉ R)
```

### 3.8 Resampling-weight bounds [P Eq. 39–40; Thm A.4, Eq. 67; S §S.1.3]

If `m_i` is given by Eq. 36, 37 or 38 and `X_i` is reasonably distributed for `p̂_i` (`p̂_i(X_i) W_i ≤ C_i`):

```
(39)  w_i ≤ C_i / |R|
```

The bound on `p̂_i(X_i)W_i` is equivalent to a bounded relative error of `W_i` from its ideal `1/p̂_i(X_i)`. For independent samples from a reasonable IS strategy (`p̂_i ≤ C_i p_i`), `p̂_i(X_i)W_i = p̂_i(X_i)/p_i(X_i) ≤ C_i`. Then

```
(40)  p̂(Y) W_Y = Σ_{i=1}^M w_i ≤ Σ_{i=1}^M C_i/|R|
```

so Eq. 39 applies to `Y` too. **Inductively, chaining GRIS from independent samples keeps a finite worst-case weight sum and worst-case contribution `f(Y)W_Y`, provided `f/p̂` is bounded.** This is not guaranteed with the `1/M` weights of earlier ReSTIR, which may not partition unity and ignore the singularities of Eq. 33. Detailed bounds [S]:
- Talbot: S.24–S.28.
- Pairwise with `α ≤ M/|R|−1`: canonical `w_i ≤ C_i/|R|` (S.29–S.32), non-canonical `w_i ≤ α x C_i/(M−|R|) ≤ C_i/|R|` (S.33–S.38).
- Tractable variant: `w_i ≤ (1/|R|) max_{j∈R} C_j` (S.58).

---

## 4. Convergence and variance analysis [P §5, p.75:8–11; App. A; S §S.2, §S.5]

### 4.1 Reasonable distributions [P Def. 5.1, Eq. 26–27]

A PDF `p` is *reasonable* for a non-negative `f` if there is `C_f` with

```
(26)  f(x) ≤ C_f · p(x)    ∀x
(27)  f(X) W_X ≤ C_f       with probability 1   (random variate X with unbiased contribution weight W_X)
```

### 4.2 Asymptotic variance [P §5.2, Eq. 28; Thm A.3]

If `p̂` is reasonable for `f` and `p_Y → p̄` (Thm A.2), then with `0 ≤ f ≤ C_f p̂`:

```
(28)  Var[ f(Y) W_Y ]  →  Var[ f(X)/p̄(X) ] ,    X ~ p̄
```

> **Boxed:** "if `p̂` is not proportional to `f`, increasing the input sample count eventually leads to diminishing returns; further variance reduction requires choosing a `p̂` better matching `f`."

### 4.3 Finite-M variance [P §5.3, Theorem 1, Eq. 29–32]

**Theorem 1.** With the assumptions of Thm A.3:

```
(29)  Var[ f(Y) W_Y ] ≤ Var[ f(X)/p̂(X) ] + b ,     X ~ p̄
(30)  b = C_f² · sqrt( Var[Σ_i w_i] ) · ( ‖p̂‖ + 2·sqrt( Var[Σ_i w_i] ) )
```

(proof [S §S.5.4]). The law of total variance gives

```
(31)  Var[ f(Y)W_Y ] = Var[ f(Y)/p_Y(Y) ] + E[ f(Y)² · Var[ W_Y | Y ] ]
```

For `f = C_f p̂` exactly (using Eq. 22):

```
(32)  Var[ f(Y)W_Y ] = Var[ (f(Y)/p̂(Y)) Σ_i w_i ] = C_f² · Var[ Σ_i w_i ]
```

> **Boxed [p.75:9]:** "Independent of `p̂`'s proportionality to `f`, our analysis shows the importance of reducing `Var[Σ_i w_i]` given finite samples. In practice, we can minimize variance by making `w_i` more uniform. In particular, preventing singularities in `w_i` avoids unbounded variance."

### 4.4 Avoiding singularities [P §5.4, Eq. 33–34]

```
(33)  f(Y)W_Y = (f(Y)/p̂(Y)) Σ_{i=1}^M w_i
(34)  Σ_i w_i = Σ_{i=1..M : X_i ∈ D(T_i)} m_i(T_i(X_i)) · p̂(T_i(X_i)) W_i · |∂T_i/∂X_i|
```

Four ways uniformity can break:
1. Some `y` reachable via only finitely many `T_i`, even in the limit.
2. `m_i ≫ 1/M`.
3. Unbounded `p̂(T_i(X_i)) W_i`.
4. Unbounded Jacobians.

Remedies:
- For (1): add samples from all source domains as `M` grows.
- For (4): **shrink the shift domain to cut off extreme Jacobians while keeping bijectivity**. The code has an optional symmetric rejection `max(J, 1/J) > 1 + 10` [C Shift.slang:556-564; Params.slang:146-147, off by default].

The paper solves all four at once with canonical samples plus the robust MIS families.

### 4.5 Guaranteeing convergence [P §5.7, Eq. 41–43; S §S.2, Eq. S.59–S.68]

**Independent `(X_i, W_i)` pairs.** With Thm A.4, `w_i ≤ C_i/|R|`. By Popoviciu's inequality:

```
(41)  Σ_i Var[w_i] ≤ Σ_i (1/4) · C_i² / |R|²
```

With a uniform bound `C_i ≤ C`, `Var[Σ w_i] ≤ C² · M/(4|R|²)`. This goes to 0 if `|R|/√M → ∞` (e.g. `|R| ≈ c·M^0.5001`). More practically, keep `|R|/M ≥ γ > 0`, which gives `O(1/M)` variance convergence.

**Dependent samples.** If `|R|/M ≥ γ`, `w_i ≤ C/|R|`, and the correlation `ρ_{i,i+k} ≤ b_k` with `b_k → 0`:

```
(42)  Var[Σ_i w_i] = Σ_i Var[w_i] + 2 Σ_{i=1}^M Σ_{k=1}^{M−i} Cov(w_i, w_{i+k})
(43)  Σ_i Σ_k Cov(w_i, w_{i+k}) ≤ (C²/(4γ²)) · ( (1/M) Σ_{k=1}^M b_k )  →  0
```

[S §S.2]: if `Σ_k b_k < ∞`, `|R| ≥ c·M^0.5001` suffices. More generally `|R| ≥ c_M · M^0.5 · sqrt(Σ_{i≤M} b_i)` with `c_M → ∞` (S.64–S.68).

### 4.6 Theorems in Appendix A [P p.75:22–23]

- **Thm A.1 (Eq. 55–56):** `E[f(X)W] = ∫_{supp X} f` for all integrable `f` ⟺ `E[W|X] = 1/p_X(X)`.
- **Thm A.2 (Asymptotic sample distribution).** For each `M ≥ M_0`, take a sequence `(X_i ∈ Ω_i)_{i=1}^M` and resample `Y_M = T_{s_M}(X_{s_M})` with weights `w_{M,i}` from Eq. 19. Assume `supp p̂ ⊂ supp Y_M` for `M ≥ M_0` (Eq. 57) and `Var[Σ_i w_{M,i}] → 0` (Eq. 58). Then:
  1. `p_Y → p̄` in probability: `Pr[|p_Y(Y) − p̄(Y)| > ε] → 0` (Eq. 59).
  2. `E[|p̄(Y)/p_Y(Y) − 1|²] → 0` (Eq. 60).
  3. `∫_Ω |p_Y(y) − p̄(y)| dy → 0` (Eq. 61).
  4. Pointwise convergence except on a null set.
  5. Every subset of `Ω` asymptotically gets the correct ratio of samples.
  
  The proof [S §S.5.2] actually uses the weaker `Var[p̂(Y_M) W_{Y_M}] → 0` (S.82), via Chebyshev and the law of total variance (S.87–S.89).
- **Thm A.3 (Asymptotic variance).** Add `f ≥ 0` and `f ≤ C_f p̂` (Eq. 62). Then `Y` covers the supports of `p̂` and `f`, and:
  - `E[|f(Y)W_Y − f(Y)/p̄(Y)|^p] → 0` for `p = 1, 2` (Eq. 63), and in probability (Eq. 64).
  - `Var[f(Y)W_Y] → Var[f(X)/p̄(X)]` (Eq. 65).
  - If `p̂ ∝ f`, `Var[f(Y)W_Y] → 0` (Eq. 66).
- **Thm A.4 (Resampling-weight bounds).** Weights from Eq. 19, MIS from 36, 37 or 38, `|R| ≥ 1`, `p̂_i(X_i)W_i ≤ C_i`. Then `w_i ≤ C_i/|R|` (Eq. 67).

### 4.7 ReSTIR as a chain; M-capping [P §6.4]; offline variant [P §6.5]

- **M-cap is critical.** Without it, the relative weight of new samples decays exponentially and the result converges to a wrong answer (Fig. 2, Fig. 10b). With a cap, the relative weight of the temporally reused sample is at most about `M_c/(M_c+1)`. Footnote 11: this gives `b_k = (M_c/(M_c+1))^k → 0`, but an exact proof is hard.
- Even if the input constraint holds, the ReSTIR output does not converge per frame, because the spatial input count is fixed (`M ↛ ∞`). Instead ReSTIR explores path space so that its *average over frames* converges in a still scene. Hypothetically with `p̂ = f_i`, `Y = Y_i^s` for random `s`: its PDF approaches `f_i/‖f_i‖` and `f_i(Y)W_Y = (f_i(Y)/p̂(Y)) Σ_t (1/T) f_i(Y_i^t) W_{Y_i^t}`.
  > **Boxed:** "with *state* defined as one path `X_i` for each pixel, ReSTIR produces an unbiased, explorative non-Markovian chain whose PDF approximates `f` better with more input samples. Averaging images of this chain converges in a still scene. In real-time, we display single states of the unbiased chain, updated in time by sampling, shifting and resampling paths."
- **Offline ReSTIR [§6.5].** Temporal reuse correlates frames and slows progressive convergence (Fig. 9b). The proposal is to render independent frames with *spatial-only* GRIS and average them. This is a two-pass method: one or more rounds of cross-pixel reuse, then averaging. Convergence of the mean follows from unbiasedness plus bounded contributions (Eq. 40). For vector-valued `f_i` with grayscale `p̂_i = |f_i|`, literal evaluation gives

  ```
  (46)  f_i(Y)W_Y = (f_i(Y)/|f_i(Y)|) · Σ_{t=1}^T (1/T) |f_i(Y_i^t)| W_{Y_i^t}
  ```
  
  which has color noise, so they recommend the explicit mean

  ```
  (47)  Ĩ_i = (1/T) Σ_{t=1}^T f_i(Y_i^t) W_{Y_i^t}
  ```
  
- **Measured results [§9.1].**
  - Fig. 7 (7×7 window, `|R|/M` fixed): Talbot and pairwise both give linear, asymptotically zero-variance curves; constant MIS does not. Talbot has lower per-sample error, but "pairwise MIS can achieve similar variance 6–7× cheaper".
  - Fig. 8: `|R| = 1` with a growing window (`|R| < O(√M)`) first lowers variance, then raises it. Convergence is guaranteed only if `|R| > O(√M)`.
  - Fig. 9a/10: M-cap 5 / 20 / 100 / none; `M_c = 20` is consistently good and no cap diverges.
  - Fig. 9b: the offline variant converges faster than averaging the real-time version (`M_c = 20`), which still empirically converges.

---

## 5. Path-space formulation [P §6.1, §7.6, §8.1; S §S.3]

### 5.1 Per-pixel integrals [P Eq. 44–45]

```
(44)  I_i = ∫_Ω h_i(x̄) f(x̄) dx̄        (pixel i, filter h_i, all sensor-to-light paths Ω, contribution f)
(45)  I_i = ∫_{Ω_i} f(x̄) dx̄           (only paths directly contributing to pixel i; box filter)
```

"Sharing paths between `Ω_i` and `Ω_j` is impossible without path modification", hence shift maps.

Per pixel: domain `Ω_i`, integrand `f_i` (`f` restricted to `Ω_i`, footnote 10: `D(f_i) = Ω_i`, `f_i = f` there), and target `p̂_i`. The target "could be e.g. grayscale path contribution functions `|f_i|` or still cheaper approximations with bounded relative error." Each pixel has a canonical sampler that is reasonable for `p̂_i`, either direct importance sampling or RIS over several reasonably sampled candidates [P §6.1].

**[C] target function:** `p̂ = dot(F, (0.299, 0.587, 0.114))` [C PathReservoir.slang:278-281]. These are Rec.601 luma weights, applied to the cached integrand `F`. **In the implementation, the domain `Ω_i` is "all paths starting at pixel i's primary hit"**. The primary hit comes from a V-buffer with a pinhole camera through the pixel center [C PathTracer.slang:200-214; SpatialReuse.cs.slang:80-93], and the paper uses no antialiasing [P §9.3]. So `x_0` and `x_1` are fixed per pixel, and the reused and shifted part of a path starts at `x_1`.

### 5.2 NEE + BSDF MIS inside the integrand; lobe-extended path space [P §7.6, Eq. 49–51]

A path tracer generating `N` strategies per length `d`:

```
(49)  I = Σ_{d=1}^∞ Σ_{n=1}^N ∫_{Ω_d} ω_n(x̄) f(x̄) dx̄ ,     ω_n(x̄) = p_n(x̄)/(p_1(x̄)+p_2(x̄))   (N=2, balance)
```

Here `n = 1` is NEE for the last vertex and `n = 2` is BSDF sampling. `p_1` and `p_2` are the NEE and BSDF path PDFs, summed over all BSDF lobes. For lobe-specific shifts, paths `x̄ = (x_0..x_d)` are paired with lobe indices `ℓ = (ℓ_1..ℓ_{d−1})`, forming the extended space `Ω̃_d` of pairs `(x̄, ℓ)`. Each `ℓ_j ∈ {1..N_lobe}`, or `ℓ_{d−1} = 𝒩` if the path ends with NEE. The partial contribution `f_ℓ`:
- For fully BSDF-sampled paths, it evaluates only lobe `ℓ_j` at each `x_j`.
- If `x_d` is NEE-sampled (`ℓ_{d−1} = 𝒩`), the BSDF at `x_{d−1}` is evaluated with **all** lobes.

```
(50)  I = Σ_{d=1}^∞ Σ_{ℓ ∈ L_d} ∫_{Ω_d} ω_{n(ℓ)}(x̄) f_ℓ(x̄) dx̄ ,      n(ℓ) ∈ {1,2}
(51)  I = ∫_{Ω̃} ω_{n(ℓ)}(x̄) f_ℓ(x̄) dx̃          (x̃ = (x̄, ℓ) over all lengths d)
```

"This formulation allows use of shift mappings that reason about the BSDF lobes, which is not possible in vertex-based path spaces."

**[INFERENCE] Consequences:**
- The **integrand that GRIS resamples is `f̃(x̃) = ω_{n(ℓ)}(x̄) · f_ℓ(x̄)`**. The NEE/BSDF MIS weight is *part of the integrand*. A shifted path must therefore **re-evaluate `ω`** with the offset path's PDFs. The code does this [C Shift.slang:518-536].
- The "path sample" stored in a reservoir is a (sub)path of a specific length and termination technique: NEE-terminated, or BSDF-sampled emitter or env hit.
- The union over lengths makes the domains *disjoint by length and technique*. Shifts preserve length and technique, since the replay terminates "when the path types match" [C PathTracer.slang:1027-1030, :1152-1155]. So a sample of length `d` can only be produced from domains' samples of length `d` with the same technique.

### 5.3 Primary sample space (what the code actually does) [P §8.1 "PSS"; S §S.3, Eq. S.69–S.77]

```
(S.69) I = E[F(Ū)] = ∫_U F(ū) dū ,       Ū = (U_1, U_2, ...) ∈ [0,1)^∞
(S.72) I = Σ_d Σ_n ∫_{Ω_d} ω_{d,n}(x) f(x) dx
(S.73) I = E[ Σ_d Σ_n ω_{d,n}(X_{d,n}) f(X_{d,n}) / p_{d,n}(X_{d,n}) ]       (ω_{d,n}(x) = 0 wherever p_{d,n}(x) = 0)
(S.77) F(ū) = Σ_{d=1}^∞ Σ_{n=1}^N ω_{d,n}(x_{d,n}(ū)) · f(x_{d,n}(ū)) / p_{d,n}(x_{d,n}(ū))
```

"Our prototypes use primary-sample space for easier implementation. Results are identical, but interpretations change: the integrand is `f(x(ū))/p_X(x(ū))` over `U`, and the PDF of the primary samples is `p_U = 1`." [P p.75:15; txt lines 1177-1197]

**[INFERENCE, verified algebraically] The PSS and path-space GRIS estimates are identical.**
- In PSS, a canonical candidate has `W = 1`, and `p̂^{PSS}(u) = lum(f(y)/p(y))`.
- The PSS Jacobian is `J_U = (p(y)/p(x)) · J_path` (Eq. 54 below).
- Resampling weight: `lum(f(y)/p(y)) · (p(y)/p(x)) J_path · 1 = lum(f(y)) J_path / p(x)`, the same as path space.
- `p̂_{←i}/p̂` ratios in the MIS weights: the common `1/p(y)` cancels.
- Output: `F(u)W_u = (f(y)/p(y)) · Σw / lum(f(y)/p(y)) = f(y) Σw / lum(f(y))`, the same as path space.

Choose PSS in WGSL: it is what a path tracer naturally accumulates (`throughput · Le · misWeight`).

### 5.4 Initial candidates = the path tree [C PathBuilder.slang:57-180; PathTracer.slang:1034-1120, :1263-1353; TracePass.cs.slang:55-73]

- One base path per pixel generates, at each vertex, an NEE candidate and (at the next vertex) a BSDF-sampled emitter-hit candidate. Each carries its MIS weight, and each is a separate term of Eq. S.77.
- Each is **streamed into the reservoir** with `w = lum(pathWeight) / rrPdf` (`russianRoulettePdf`, 1 when RR is off, which is the default [C ReSTIRPTPass.h:88]). `pathWeight = throughput(x_1..) · Le · ω` is the PSS integrand of that subpath.
- `PathBuilder.finalize` sets `M = 1` [:45-55], and `finalizeRIS` sets `W = Σ_k w_k / (p̂(F_sel)·M)` [PathReservoir.slang:435-441].

[INFERENCE] This is RIS over *disjoint* domains (different lengths or techniques), so `m = 1` for each candidate. The result `F_sel · W = F_sel · Σ_k lum(F_k)/lum(F_sel)` is unbiased for the **sum** of all subpath contributions. That is why `W` is *not* divided by the number of subpaths.

- **Multiple candidate path trees** (offline `S = 32`): trees are merged with `mergeInSamplePixel` (`w = Σw_tree`, `M += 1`) and finalized with `W = Σw/(p̂·M)` [C PathTracer.slang:1947-1960+]. That is standard i.i.d. RIS with `1/M`.
- **Excluded subpaths (direct lighting is handled by ReSTIR DI):**
  - NEE at the primary hit, skipped when `kDisableDirectIllumination && isPrimaryHit` (default `disableDirectIllumination = true` [C ReSTIRPTPass.h:96]).
  - BSDF-sampled emitter hits at `x_2`: `if (path.length == 1 && !transmission && !delta) computeEmissive = false` [C PathTracer.slang:1025]. Also `pathLength ≥ 1` in `addEscapeVertex/addNeeVertex`.
  - So ReSTIR PT only carries paths with **≥ 2 scattering vertices** (`x_1`, `x_2`, …, then the light). Direct light seen through delta/transmissive first bounces is accumulated separately as `LDeltaDirect` and added to the output [C PathTracer.slang:1074, :1867].

---

## 6. Shift mappings [P §7, §8.1]

### 6.1 Generic shift and design principles [P §7.1, Eq. 48]

```
(48)  T([x_0, x_1, x_2, x_3, ...]) = [y_0, y_1, y_2, y_3, ...]
```

The base path `x̄` is from pixel `k`, the offset path `ȳ` is in pixel `j`. `y_0` is on the sensor and `y_1` comes from tracing through pixel `j`. The freedom is in choosing `y_2, …` so that `f_k(T_k(x̄)) ≈ f_j(x̄)` and `|∂T_k/∂x̄| ≈ 1`.
- **Local decisions:** build `y` vertex by vertex. `y_{i+1}` is decided from `x_{i−1}, x_i, x_{i+1}` and `y_{i−1}, y_i`. For example, if `x_i`, `x_{i+1}` and `y_i` are rough, connect with `y_{i+1} = x_{i+1}`.
- **Ensuring bijectivity:** local sequential construction can halt, e.g. half-vector copy can turn a refraction into total internal reflection, but not the reverse. The shift may return **"undefined"**, and that path is then not in the shift's domain. **Every successful shift must be invertible.** Symmetric shift maps are preferred: if `T_{k→j}(x̄) = ȳ` then `T_{j→k}(ȳ) = x̄`.

### 6.2 Building blocks [P §7.2]

- **Vertex copy (reconnection):** if `x_i`, `x_{i+1}` and `y_i` are all on rough materials, set `y_{i+1} = x_{i+1}` [Lehtinen 2013]. Later vertices are copied too. Good for diffuse and rough materials.
- **Half-vector copy:** copy the base half-vector (in local tangent space) and re-trace `y_{i+1}` in the reflection or refraction direction [Kettunen 2015]. For near-specular vertices.
- **Direction copy:** copy the outgoing world-space direction. Often used with environment maps.
- **Random replay:** reuse the base path's random numbers to trace `y_{i+1}` with the base path's sampling method. It approximates half-vector copy, direction copy, or reconnecting to an area light for NEE.
- **Manifold exploration:** find the next connectable vertex and use manifold walks through specular chains. High quality, high cost.

Pitfalls: not closely approximating ideal reflections on high gloss, reconnecting through occlusion, drastic changes in segment length, switching objects or materials, or diverging after reflection or refraction.

### 6.3 Full shift [P §7.3]

Kettunen et al. [2015] test sequentially: if `x_i`, `x_{i+1}` and `y_i` are all "sufficiently rough", reconnect, otherwise half-vector copy and repeat at the next vertex. Hua et al. [2019] show equivalent results by replacing half-vector copy with random replay, which is more efficient on the GPU and slightly more general. The paper adopts that with improvements.

### 6.4 The two real-time shifts [P §7.4]

1. **Reconnection shift:** always sets `y_2 = x_2` (reconnect at the first indirect vertex). This works well for mostly diffuse scenes. ReSTIR GI implicitly uses this choice "but trades correctness for performance". It needs only the reconnection vertex stored plus re-evaluation of the path contribution.
2. **Hybrid shift** (random replay + reconnection, improved from Hua et al. 2019): postpone reconnection using random replay while the connectability conditions fail.
   - **Key implementation idea:** the base path precomputes **the first vertex `x_i` whose `x_i` and `x_{i+1}` satisfy the connectability condition. Reconnection must happen at this vertex, or it does not happen.** Only `x_{i+1}` is stored, not the full path.
   - Bijectivity then *forces* agreement: "when building `y`, if we find it disagrees on the earliest possible reconnection vertex, the shift must return 'undefined' as it would not be invertible."

### 6.5 Connectability conditions [P §7.5]

- **Distance condition.** Reconnect to `x_{i+1}` only if `‖x_{i+1} − x_i‖ ≥ d_max`. "By symmetry, the offset path must fulfill `‖x_{i+1} − y_i‖ ≥ d_max` for `y_{i+1}` to become `x_{i+1}`." This avoids the geometry-term singularity for short segments, e.g. in corners. The idea resembles Manzi et al. [2014], but postpones via random replay instead of a manifold walk.
- **Lobe-specific roughness.** Kettunen's test uses roughness of `x_i`, `x_{i+1}`, `y_i` above a threshold, which is ambiguous for multi-lobe BSDFs such as a diffuse base plus clear coat. Instead, examine **the roughness of the lobe `ℓ_j` chosen to sample `x_{j+1}`**.
  - NEE-sampled vertices count as rough if **at least one** of their BRDF lobes is sufficiently rough.
  - **All light vertices are rough.**
  - If all three vertices `x_j`, `x_{j+1}`, `y_j` pass the roughness and distance conditions, reconnect. Otherwise sample `y_{j+1}` via random replay.
  - With random replay the offset path usually picks the same lobe index as the base. A reconnection copies the lobe index from the base vertex [§7.6].
- **Thresholds [§8.3]:**
  - A per-lobe **GGX roughness threshold of 0.2** "generally works well".
  - The distance threshold scales with the scene: **1%–5% of scene size** in their tests.
  - The optimum depends on camera distance to the region of interest and on glossiness. Automatic setting is future work.

**[C] implementation details of the conditions:**
- `specularRoughnessThreshold = 0.2` and `nearFieldDistance = 0.1` (absolute world units) [C Params.slang:150-152].
- Both conditions are enabled by default: `localStrategyType = RoughnessCondition | DistanceCondition` [C Params.slang:145].
- The rough-vertex test with separate lobes (`separatePathBSDF = true`, the default [C ReSTIRPTPass.h:113]):
  - `hasRoughComponent(sd, t) = (pDiffuseReflection > 0) || linearRoughness > t` [C Falcor BxDF.slang:918-921], used for the *current* vertex, where any rough lobe exists.
  - `classifyAsRough(sd, t) = linearRoughness > t` [:913-916].
  - For the *sampled* bounce: `!isSpecularBounce || linearRoughness > t` [C PathTracer.slang:955-956, :1405].
- `canConnect` at vertex `path.length` (0 = primary hit):
  ```
  length ≥ 1 && length < rcVertexLength
  && !(DistanceCondition && !isFarField)
  && !(RoughnessCondition && !(isCurrentVertexRough && isLastVertexRough))
  ```
  [C PathTracer.slang:1167-1169]. `isFarField = ‖x_cur − x_prev‖ ≥ nearFieldDistance` [:1159].
- After sampling the scatter at a candidate rc vertex, if the *sampled lobe* is not connectable (`!seenAsConnectible`), the rc choice is revoked (`rcVertexLength` reset) [C :1402-1425].
- `[UNVERIFIED]` In Falcor, `linearRoughness` is the perceptual roughness and GGX `α = linearRoughness²`. Confirm before mapping the 0.2 threshold to glTF `roughnessFactor`, which is also perceptual, so it maps 1:1 if the assumption holds.

### 6.6 Jacobian determinants [P §8.1, Eq. 52–54]

"We assume base and offset paths up to vertex `i` are fixed; probability densities below are … conditional to earlier path state." Here `ω_i^x` is the unit vector from `x_i` to `x_{i+1}`, `ū_i^x` the random numbers producing it, and `y` is analogous.

**Solid angle (reconnection)** [e.g. Kettunen 2015]:

```
(52)  |∂ω_i^y / ∂ω_i^x| = |cos θ_2^y / cos θ_2^x| · ‖x_{i+1} − x_i‖² / ‖x_{i+1} − y_i‖²
```

`θ_2^•` is the angle between `ω_i^•` and the **geometric** surface normal at `x_{i+1} = y_{i+1}`. The subscript "2" is as printed and refers to the reconnection vertex, which is `x_2` in the plain reconnection shift.

**Random replay (solid-angle view):**

```
(53)  |∂ω_i^y/∂ω_i^x| = |∂ω_i^y/∂ū_i^y| · |∂ū_i^y/∂ū_i^x| · |∂ū_i^x/∂ω_i^x| = p_{ω_i^x}(x_{i+1}) / p_{ω_i^y}(y_{i+1})
```

This is the ratio of the solid-angle sampling PDFs of the next vertices, given the paths up to vertex `i`. For local shift decisions, **the full-path Jacobian is the product of the per-vertex Jacobians**.

**PSS.** The Jacobian of random replay is always 1. Solid-angle Jacobians convert to PSS by dividing by the right-hand side of Eq. 53:

```
(54)  |∂ū_i^y/∂ū_i^x| = |∂ū_i^y/∂ω_i^y| · |∂ω_i^y/∂ω_i^x| · |∂ω_i^x/∂ū_i^x| = ( p_{ω_i^y}(y_{i+1}) / p_{ω_i^x}(x_{i+1}) ) · |∂ω_i^y/∂ω_i^x|
```

**Mixing PSS and path-space shifts ★.** Samplers often use more random numbers than the path has dimensions, so path-space ↔ PSS Jacobians do not exist. Bitterli et al. [2017] pad paths with unused dimensions to make a bijection. Assuming that theoretical bijection exists allows mixing path-space shifts and random replay.

**[INFERENCE + C] The composite PSS Jacobian actually used for a reconnection at vertex `k`.**
- Base path: `x_{k−1} → x_k → x_{k+1}`. Offset path: `y_{k−1} → x_k → x_{k+1}`.
- `y_{k−1}` is the pixel's primary hit `y_1` for the pure reconnection shift (`k = 2`), or the last random-replayed vertex for the hybrid shift.
- Suffix `x_{k+1}, …` is unchanged. Random-replayed prefix vertices contribute 1.

```
J = |∂ū^y/∂ū^x|
  = [ p_ω(y_{k−1} → x_k) / p_ω(x_{k−1} → x_k) ]                                   (BSDF pdf at the vertex before the rc vertex)
  × [ (|cos θ_{x_k}^y| / ‖x_k − y_{k−1}‖²) / (|cos θ_{x_k}^x| / ‖x_k − x_{k−1}‖²) ]   (Eq. 52: geometry at rc vertex)
  × [ p_ω(x_k → x_{k+1} | from y_{k−1}) / p_ω(x_k → x_{k+1} | from x_{k−1}) ]        (BSDF pdf at rc vertex; only if x_{k+1} was BSDF-sampled at x_k)
```

Code [C Shift.slang:450-470, :477-482, :495-504, :538-542]:
- `Jacobian = (shifted_cos/shifted_dist2) · (orig_dist2/orig_cos)`, with `cos = |dot(rcVertex.faceN, dir)|`.
- Then `*= dstPDF1/srcPDF1`, then `*= dstRcVertexScatterPdf/srcRcVertexScatterPdf` when the rc vertex is not final and not an emitter.
- When `x_{k+1}` was NEE-sampled, the direction PDF is the light's PDF, which is unchanged because `x_k` and the light point are shared. There is no BSDF-PDF ratio: `isRcVertexNEE` gives `dstPDF2 = lightPdf`.
- When `x_{k+1}` was a BSDF emitter hit at the final vertex, the PDF ratio is applied inside the MIS branch [C :525-535].
- PDFs are lobe-restricted when `separatePathBSDF` is on: `allowedSampledTypes = specular ? 0xC : 0x3` [C PathTracer.slang:37-41]. They use the *single-lobe* sampling PDF (`result.pdfSingle`) for the Jacobian and the all-lobe PDF for MIS [C PathTracer.slang:309-340].

The **source-side factors are cached in the reservoir** so they need not be recomputed, because `x_{k−1}` of a hybrid base path is not otherwise available. `cachedJacobian.x` = source BSDF PDF at `x_{k−1}` toward `x_k`; `.y` = source BSDF PDF at `x_k` toward `x_{k+1}`; `.z` = `|cos θ_{x_k}| / ‖x_k − x_{k−1}‖²` [C PathTracer.slang:333-338, :1175-1185; Shift.slang:461-500]. In Algorithm 1 this is `rcVertexCachedValues (float4)`: "various partial terms for evaluating the Jacobian at reconnection. Light sampling PDF for the MIS weight is also stored here." When the offset path is shifted *again* (chained reuse), the cache is replaced by the destination values (`dstCachedJacobian`) [C Shift.slang:568-569].

**[INFERENCE] Area-measure view.** With per-pixel domains of paths starting at the fixed primary hit, and vertex-area measure for `x_2, x_3, …`, the pure reconnection shift is the *identity* on `(x_2, x_3, …)`. So its area-measure Jacobian is 1, and Eq. 52 is purely the solid-angle-to-area conversion at `x_1`. This is a good sanity check: resampling weights computed in area measure (`p̂ = lum(f)` with geometry terms, `W` in area measure) and in PSS must agree numerically on a test scene.

### 6.7 When a shift fails (returns 0 / "undefined")

Stated or implied by the paper:
- Non-invertible cases, e.g. disagreement on the earliest reconnection vertex [§7.4], or TIR in half-vector copy [§7.1].
- Connectability failure at the fixed rc vertex: roughness or distance [§7.5].
- Visibility failure: `p̂ = 0`, since `p̂` includes visibility [App. B].

Implemented in [C Shift.slang] and [C PathTracer.slang]:
- `srcReservoir.weight == 0` → 0 [Shift.slang:55, :95, :121].
- **Delta lobe before or after the rc vertex** → 0 [:438-442], since delta vertices cannot be reconnected.
- `isJacobianInvalid(J)`: `J ≤ 0`, NaN or Inf → 0 [:224-227, :471, :484, :544].
- Near-field at the offset: `‖x_k − y_{k−1}‖ < nearFieldDistance` in hybrid mode → 0 [:454-458]. This is the symmetric distance test.
- `dstF1 == 0` or `dstF2 == 0` (connection point behind the surface, or the BSDF lobe not supported) → 0 [:511-512].
- Occluded segment `y_{k−1} ↔ x_k` (one shadow ray, `tMax·0.999`) → 0 [:546-552, :632-654].
- Hybrid replay invertibility [PathTracer.slang:1194-1261], checked vertex by vertex while replaying `y_1…y_{k−1}` for the base path's `rcVertexLength = k`:
  - If an offset vertex before `k−1` *would* satisfy the connectability condition (so the offset path would reconnect earlier), the shift is **non-invertible** → invalidate. `invalidateAndTerminateReplayPath` sets `rcPrevWo = 0`, which later makes `Shift.slang:167` return 0.
  - At `y_{k−1}`, if the offset vertex cannot produce a rough bounce → invalidate. Otherwise terminate the replay and store `(rcPrevHit, rcPrevWo, Tp)` for reconnection.
  - If the replay terminates via a miss with a mismatched length → 0 [PathTracer.slang:1669-1673].
- Optional symmetric Jacobian-magnitude rejection: `max(J, 1/J) > 11` → 0 (off by default) [Shift.slang:556-564].

### 6.8 Suffix reuse, and NEE vs. BSDF-sampled light at or after the rc vertex [C Shift.slang:383-572; PathBuilder.slang; PathTracer.slang]

**Stored data for the suffix.**
- `rcVertexWi`: world-space direction from `x_k` to `x_{k+1}`, or toward the light.
- `rcVertexIrradiance` (named `rcVertexRadiance` in Algorithm 1): the PSS-weighted **incident radiance at `x_k` along `rcVertexWi`**. It is the product of `f/p` for vertices after `x_k` times the emitted radiance times the MIS weight at the last vertex, and **excludes `x_k`'s own BSDF**. It works because `prefixThp` absorbs throughput up to and including `x_k`'s scatter, and `thp` restarts at 1 after that [C PathState.slang:114-115; PathTracer.slang:417-421].
- The suffix is **reused unchanged** (no rays), except for the MIS-weight fix-up at the final vertex when `x_k` is also the last scattering vertex.

`pathFlags` encode `pathLength` (the index of the last scattering vertex, excluding the light vertex), `rcVertexLength` (the index `k`), `lastVertexNEE`, the light type, and delta, transmission and specular bits for the bounce before and after the rc vertex [C PathReservoir.slang:21-140]. With vertex index 0 = primary hit, the cases in [C Shift.slang:434-536] are:

| Case | Condition | Shifted integrand `F_dst` (PSS) | Jacobian factors |
|---|---|---|---|
| rc vertex strictly inside the path | `k < pathLength` | `(f(y_{k−1}→x_k)/p_{y}) · (f_{x_k}(ω_in^y, rcWi)/p_{x_k}^y(rcWi)) · L_stored` | geometry × `p_{y_{k−1}}/p_{x_{k−1}}` × `p_{x_k}^y/p_{x_k}^x` |
| rc vertex is final, **NEE** to light | `k == pathLength && lastVertexNEE` | `(f/p)_{y_{k−1}} · f_{x_k}(·,rcWi)/lightPdf · Le · ω_NEE^dst` with `ω_NEE^dst = lightPdf/(lightPdf + p_{BSDF,x_k}^{dst,all}(rcWi))` | geometry × `p_{y_{k−1}}/p_{x_{k−1}}` (light PDF unchanged) |
| rc vertex is final, **BSDF-sampled emitter hit** | `k == pathLength && !lastVertexNEE` | `(f/p)_{y_{k−1}} · (f_{x_k}/p_{x_k}^{dst}) · Le · ω_BSDF^dst` with `ω_BSDF^dst = p_{x_k}^{dst,all}/(p_{x_k}^{dst,all} + lightPdf)` | geometry × `p_{y_{k−1}}` ratio × `p_{x_k}^{dst}/p_{x_k}^{src}` |
| rc vertex **is the emitter** (escaped vertex used as rc; hybrid only; `k = pathLength+1`) | `k == pathLength+1 && !lastVertexNEE` | `(f/p)_{y_{k−1}} · Le · ω_BSDF^dst` with `ω_BSDF^dst = p_{y_{k−1}}^{all}(→x_k)/(p^{all} + lightPdf)` | geometry at emitter × `p_{y_{k−1}}/p_{x_{k−1}}` |
| rc "vertex" is the **environment map** (no hit, env light, escaped) | `!rcVertexHit.valid && lightType == EnvMap && k == pathLength+1` | direction copy: `(f/p)_{y_{k−1}}(rcWi) · Le · ω_BSDF^dst`; visibility along the direction | `p_{y_{k−1}}(rcWi)/p_{x_{k−1}}(rcWi)` (no geometry term) |
| **no rc vertex** (hybrid: no connectable pair found) | `rcVertexLength > pathLength+1` | the whole path is random-replayed; `F_dst = Tp` = replayed contribution [C PathTracer.slang:1686-1689] | 1 |

For stored values at the rc vertex, `rcVertexIrradiance` has the last-vertex MIS weight **divided out** at insertion:
- Emitter hit: `/misWeight` [C PathBuilder.slang:103-107].
- NEE: `· lightPdf / misWeight`, i.e. it stores `Le` [C PathBuilder.slang:168-172].

The MIS weight is then recomputed with the offset path's BSDF PDF after the shift [C Shift.slang:518-536].

**Hybrid composition** [C Shift.slang:116-200]: `F_dst = Tp · F_rc`.
- `Tp` is the random-replay prefix throughput `y_1 … y_{k−1}` (PSS, Jacobian 1).
- `F_rc` is the reconnection evaluated from `y_{k−1}` (loaded from the stored `rcPrevHit` and `rcPrevWo`) to `x_k`, as in the table above.
- Only the reconnection factor contributes to `J`.
- If `k == 1`, i.e. the rc vertex is `x_2` and the prefix is only the primary hit, then `Tp = 1`. The primary vertex must itself be classified rough, else `Tp = 0` [:135-153].
- Random replay is traced in a separate "retrace" pass *before* the reuse pass. It stores `ReconnectionData{rcPrevHit (16B), rcPrevWo (12B), pathThroughput (12B)}` per (pixel, neighbor, direction) [C SpatialPathRetrace.cs.slang:131-163; PathReservoir.slang:142-174]. There are 2 per neighbor: `data[2i]` for central→neighbor and `data[2i+1]` for neighbor→central. `PixelReconnectionData` is padded to 256 B in real time (6 entries) and 512 B offline (12 entries).

**`[UNVERIFIED]` light-PDF caveat in the reference code.** In the "emitter is rc vertex" case, the MIS weight uses `srcReservoir.lightPdf`. That is the solid-angle NEE PDF computed from the **base** path's previous vertex `x_{k−1}` [C PathTracer.slang:1056, :1115], but the offset's previous vertex `y_{k−1}` differs. For triangle area lights the solid-angle light PDF depends on that vertex (`p_A · d²/cos`), so the recomputed MIS weights of the two techniques no longer form an exact partition of unity for offset paths. The code comment at [C PathReservoir.slang:231] acknowledges the light PDF "might change after shift". **Recommendation:** store the light-selection PDF in *area* measure (`p_A`, including the light-selection probability) and convert with the offset geometry. Alternatively, recompute the light PDF at `y_{k−1}`. For the NEE-at-final-rc-vertex case the light is sampled from `x_k`, which is shared, so it is exact.

### 6.9 Dynamic scenes: temporal re-evaluation of the suffix [C Shift.slang:69-72, :180-188; PathTracer.slang:1730-1840; P App. B]

[P App. B] Temporal reuse is unbiased with proper MIS weights, *but a temporal shift mapping is needed*. Evaluating GRIS MIS weights requires bijectively shifting paths between the prior and current frames, i.e. evaluating paths in *both* frames. This is "tricky in dynamic environments". Biased approximations, such as neglecting visibility [Ouyang 2021], give imperceptible bias for performance. Lin et al. [2021] account for temporal changes explicitly.

[C] Option `temporalUpdateForDynamicScene` (default off [C ReSTIRPTPass.h:109]): before shifting a temporal sample, `traceTemporalUpdate` **re-traces the suffix from the rc vertex in the current scene** by random replay with the stored `rcRandomSeed`:
- If the rc vertex is the NEE vertex, it re-samples the light with the same random numbers and re-traces visibility.
- Otherwise it re-traces the whole suffix.
- It overwrites `rcVertexIrradiance`, `rcVertexWi`, `lightPdf` and the light type.

**This matters for moving lights.** Without it, `L_stored` is last frame's radiance.

**[INFERENCE]** For an exact GRIS temporal step in a dynamic scene, the forward shift (prev → current) should evaluate `p̂` in the **current** scene, and the reverse shift for the MIS weight (current → prev) should evaluate `p̂_prev` in the **previous** scene. The latter needs the previous frame's geometry and lights, i.e. last frame's BVH and light buffer. The code evaluates the reverse shift with the previous camera (`usePrev`) and previous V-buffer, but with current-frame BVH and materials [C TemporalReuse.cs.slang:206-209]. That is exact only for static geometry and lights. Budget for keeping one frame of history for light transforms (cheap) and, if feasible, for moving-object transforms. A two-level BVH with per-instance transforms makes "evaluate in frame t−1" feasible by using the previous instance matrices.

---

## 7. The full ReSTIR PT algorithm

### 7.1 As stated in the paper [P §6.3, p.75:11], transcribed

Let `Y_i^{t−1}` be a resampled (or sampled) path for pixel `i` on frame `t−1`, stored for later reuse with its unbiased contribution weight `W_{Y_i^{t−1}}`. For each frame `t`:

1. **(Initial candidates)** Generate an independent sample `X_i^t` for each pixel `i` and evaluate its contribution weight `W_{X_i^t}`.
2. **(Temporal reuse)** Use GRIS to select `Z_i` by resampling between last frame's sample `Y_i^{t−1}` and the new sample `X_i^t`. Pixel correspondences may be identified via motion vectors.
3. **(Spatial reuse)** Each pixel selects numerous random spatial neighbors `j`, and selects `Y_i^t` by resampling between `Z_i` and neighbor samples `Z_j` via GRIS. This step may be executed multiple times with the assignment `Z_i := Y_i^t`.
4. Estimate the pixel integral, `I_i^t ≈ f_i(Y_i^t) W_{Y_i^t}`.

"ReSTIR typically stores a reservoir for each pixel `i`. The new samples `X_i^t` are treated as reservoirs with `M_r = 1`, and are merged with the reservoir storing `Y_i^{t−1}`, accounting for the confidence weights. Spatial resampling works akin to a stochastic convolution … The last sample `Y_i^t` is stored in that pixel's reservoir, and its confidence weight from the spatial reuse passes is used in the next frame's temporal resampling step."

### 7.2 Implementation choices [P §8, §8.3]

- Built on Falcor. ReSTIR PT is implemented as chained GRIS passes per §6.3.
- Two shifts are implemented: the hybrid shift with lobe-specific improvements, and the simpler reconnection shift (always at the first indirect vertex).
- "Like many path tracers, ours only evaluates the sampled BSDF lobe for BSDF-sampled vertices and evaluates all lobes for NEE-sampled vertices." Lobe selections are extra path parameters (§7.6), and the sampled lobe's roughness decides between reconnection and replay.
- Full surface-to-surface light transport only. Volumes need volumetric shifts (future work).
- Two prototypes, real-time and offline, neither performance-optimized. ReSTIR PT primarily addresses **indirect** light. **ReSTIR DI** [Bitterli 2020] handles direct lighting.
- **Resampling MIS:** defensive pairwise (Eq. 38) for spatial reuse. The real-time variant also uses generalized Talbot (Eq. 36) for temporal reuse. In both, **`|R| = 1`**. Offline convergence comes from rendering multiple independent frames.

### 7.3 Per-frame pass structure in the reference code [C ReSTIRPTPass.cpp:676-783]

For each of `samplesPerPixel` independent ReSTIR chains (default 1), with a separate temporal reservoir buffer per chain:
1. `GeneratePaths` (once per frame): primary hits from the V-buffer.
2. `TracePass`: trace `candidateSamples` path trees per pixel (1 real-time, 32 offline). Stream the subpaths into a reservoir, `finalizeRIS`, and write `outputReservoirs`.
3. If temporal reuse is enabled and a previous frame exists:
   - if hybrid, a `TemporalPathRetrace` pass does the random-replay prefixes for both directions and writes `ReconnectionData`;
   - then `TemporalReuse` (Talbot) runs.
4. For `r` in `numSpatialRounds` (1 real-time, 3 offline): if hybrid, `SpatialPathRetrace`; then `SpatialReuse` (pairwise). The output goes to `temporalReservoirs`, which ping-pong with `outputReservoirs`.
5. Copy reservoirs and the V-buffer for the next frame when needed.

Final pixel color [C SpatialReuse.cs.slang:535-549]:

```
color = F_sel · W  (+ directLighting from ReSTIR DI) (+ LDeltaDirect)
```

accumulated as `/samplesPerPixel`.

### 7.4 Temporal reuse in detail [C TemporalReuse.cs.slang:85-313]

- `prevPixel = pixel` by default. Reprojection is optional: `pixel + mv·frameDim + rand2`, which randomizes the rounding.
- The previous reservoir is fetched, and the M-cap is applied: `M_prev ← min(20·M_cur, M_prev)`.
- **Talbot with 2 candidates** (current `c`, previous `p`) and confidence weights:

```
Candidate c (y = X_c, J = 1):
   p̂_c(y)·M_c ,   p̂_{←p}(y) = p̂_prev(T_{c→p}(y)) · |∂T_{c→p}/∂y|      (shift current → previous: usePrev camera, prev V-buffer)
   m_c = M_c p̂_c(y) / ( M_c p̂_c(y) + M_p p̂_{←p}(y) )
Candidate p (y = T_{p→c}(X_p), J = |∂y/∂X_p|):
   p̂_{←p}(y) = p̂_prev(X_p)/J ,   m_p = M_p p̂_{←p}(y) / ( M_c p̂_c(y) + M_p p̂_{←p}(y) )
w_c = m_c · p̂_c(X_c) · W_c ;   w_p = m_p · p̂_c(y) · J · W_p ;   select ∝ w ;   W = (w_c + w_p)/p̂_c(Y) ;   M = M_c + M_p
```

This costs **2 shifts**, each with one visibility ray, plus hybrid prefix replays.

### 7.5 Spatial reuse in detail [C SpatialReuse.cs.slang:151-411]

- Neighbors: `k` = 3 (real-time) or 6 (offline).
  - They are drawn from a precomputed table of 8192 offsets. The offsets form an R2 low-discrepancy sequence (plastic constant 1.3247179572447) inside a unit disk, stored as RG8Snorm [C ReSTIRPTPass.cpp:804-826].
  - Each offset is scaled by `gGatherRadius` (20 px real-time, 10 px offline). The start index per pixel is random.
- Neighbor rejection (`isValidGeometry`, on by default [C ReSTIRPTPass.h:181]): `dot(N_c, N_n) ≥ 0.5` and `|d_c − d_n| < 0.1·d_c` (camera distance) [C SpatialReuse.cs.slang:71-78]. `[INFERENCE]` This is unbiased because the choice depends only on primary-hit geometry, not on the samples, and rejected neighbors are skipped in every MIS term.
- Defensive pairwise MIS (§3.4). Per valid neighbor `i` there are **two shifts**: central → neighbor for the canonical weight's pair term, and neighbor → central for the candidate itself. That gives **2k shifts** per pixel per round, each with one shadow ray plus hybrid replays.
- The canonical sample is merged last with `J = 1` and `m_c`. Then `W = Σw/p̂(Y)/(k+1)`.
- Talbot spatial (optional) needs `(k+1)·k` shifts.

### 7.6 Reservoir contents

**Paper, Algorithm 1 [S §S.4, p.5], "Content of the reservoir struct (88 bytes)":**

```
1  struct Reservoir
2      float  M;                     // Confidence weight (for e.g., M-capping).
3      float  W;                     // Unbiased contribution weight.
4      float3 F;                     // Cached integrand value of the sample.
5      uint   pathFlags;             // Path length, technique type, reconn. vertex id, etc.
6      uint   initRandomSeed;        // Random state at primary hit x̄_1.
       // Information about the reconnection vertex (rc):
7      uint   rcVertexRandomSeed;    // Random state at reconn. vertex.
8      uint   rcVertexInstanceID;    // Hit point information:
9      uint   rcVertexPrimitiveIndex;
10     float2 rcVertexBarycentrics;
11     float3 rcVertexWi;            // Direction to next vertex of base path.
12     float3 rcVertexRadiance;      // Incident radiance from next vertex.
13     float4 rcVertexCachedValues;  // Various partial terms for evaluating the Jacobian at
                                     // reconnection. Light sampling PDF for the MIS weight is also stored here.
```

Byte count: 4+4+12+4+4+4+4+4+8+12+12+16 = **88 B**.

[S §S.4] "most of the storage is used for enabling a reconnection to the base path's vertex: reconnection requires evaluating offset path's visibility to the reconnection vertex and the BSDF towards base path's next vertex … our reservoir data structure is unoptimized and highly compressible; real-time use would allow lossy compression for increased performance, but our prototype implementation does not do it." [P §8.2]: "Our reservoirs consume 88 bytes per path while supporting our hybrid shift. Beyond storing contribution weights `W_r` and confidence weights `M_r`, we store information needed for our shift map: the path's chosen reconnection vertex and a seed for random replay."

**Code struct** [C PathReservoir.slang:223-237], same content, commented "88/128 B":

```
M (float, fractional allowed), weight (w_sum during RIS, W after), pathFlags,
rcRandomSeed, F (float3), lightPdf, cachedJacobian (float3: srcPdf@x_{k-1}, srcPdf@x_k, |cos|/d² at x_k),
initRandomSeed, rcVertexHit {instanceID, primitiveIndex, float2 barycentrics}, rcVertexWi, rcVertexIrradiance.
```

`pathFlags` bit layout [C PathReservoir.slang:21-140]:

| Bits | Field |
|---|---|
| 0–3 | `pathLength` (≤ 15) |
| 4–7 | `rcVertexLength` (15 = none) |
| 8 / 9 | delta before / after the rc vertex |
| 10 / 11 | transmission before / after |
| 16 | `lastVertexNEE` |
| 18–19 | light type (e.g. Emissive, EnvMap, Analytic) |
| 26 / 27 | specular bounce before / after (lobe class) |

`kMaximumPathLength = 15` [C StaticParams.slang:41].

**RNG requirement [INFERENCE from C].** Random replay regenerates the offset path from a stored **32-bit seed**: `SampleGenerator(uint seed)` with `initRandomSeed = sg.getCurrentSeed()` at the primary hit, before the first scatter [C PathTracer.slang:1393-1397; :250]. Replay consumes random numbers *identically* to the base path. When NEE is not needed at a vertex during replay, `skipLightSampleRandomNumbers` still consumes the same count [C :1354-1366], and RR consumes its number even during replay [C :1378-1383].

**WGSL design rule:** use a counter- or hash-based RNG with a ≤ 32-bit state (e.g. PCG32 state, or hash(seed, bounce, dim)). Consume a **fixed number of dimensions per bounce** regardless of branch outcomes. Deriving each bounce's numbers from `hash(seed, bounceIndex, dimension)` is the most robust way to keep base and offset in lockstep.

---

## 8. Pseudo-code

The paper contains no algorithm listing other than the §6.3 step list (transcribed in §7.1) and supplemental Algorithm 1 (§7.6). Below is a **[INFERENCE] reconstruction** of the reference implementation, WGSL-oriented. It is condensed but faithful to [C].

```text
// ---------- Pass A: path tree + initial RIS (per pixel) ----------
r = emptyReservoir();                            // M=0, wsum=0
for c in 0..S-1:                                 // S = candidate path trees (1 real-time, 32 offline)
  t = emptyReservoir(); seed = rngSeed(pixel, frame, c)
  x1 = primaryHit(pixel)                         // V-buffer, pixel center, pinhole
  prefixThp = 1; thp = 1; rcLen = NONE; builder = {}
  t.initSeed = seed_at_x1_before_first_scatter
  for len = 0 .. maxBounces:                     // len 0 = x1
    sd = shade(x_len)
    if len>=1 and emitterHit and not (len==1 and diffuse-reflection-from-x1):   // BSDF-sampled emitter hit, path length len-1
       mis = pdfBsdfAll/(pdfBsdfAll + lightPdf(prev->x_len))
       streamCandidate(t, F = prefixThp*thp*Le*mis, suffix = (len-1==rcLen ? Le : thp*Le*mis), NEE=false, len-1)
    if hybrid and canConnect(len, x_prev, x_len):   // first rough/rough + far-field pair
       rcLen = len; store rcHit, cache srcPdf(x_prev->x_len), |cos|/d^2, flags(before)
    if len>=1 (x1 NEE is ReSTIR DI's job):
       ls = sampleLight(x_len); mis = ls.pdf/(ls.pdf + pdfBsdfAll(ls.dir)); V = shadow(ls)
       if V: streamCandidate(t, F = prefixThp*thp*f(ls.dir)*Le/ls.pdf*mis, suffix = (len==rcLen ? Le : thp*f*Le/ls.pdf*mis), NEE=true, len)
    else consumeLightRandomNumbers()
    (dir, f/p, lobe) = sampleBsdfOneLobe(sd)      // single-lobe pdf for Jacobian, all-lobe pdf for MIS
    if len == rcLen: cache srcPdf at x_k toward next; record flags(after); rcWi = dir;
                     if sampled lobe not rough: revoke rc (rcLen = NONE)
    thp *= f/p; if len <= rcLen: prefixThp *= thp; thp = 1
  t.M = 1;  // finalize tree
  r.mergeIID(t)                                   // w = t.wsum, M += 1, pick ∝ w
r.W = r.wsum / (lum(r.F) * r.M)                   // RIS with 1/M over i.i.d. trees
store r

streamCandidate(t, F, suffix, nee, len):          // Chao reservoir over the tree's subpaths
  w = lum(F); t.wsum += w; t.M += 1
  if u()*t.wsum <= w: t.F=F; t.flags=(len, rcLen, nee, lightType); t.rcIrr = suffix (MIS of last vertex removed if len==rcLen);
                      t.rcWi = (len==rcLen ? dirToLight : builder.rcWi); t.rcHit=builder.rcHit; t.cache=builder.cache; t.lightPdf=...

// ---------- Shift (reconnection part), returns (F_dst, J) ----------
reconnect(dstPrevVertex y, srcReservoir s):
  if deltaBefore(s) or deltaAfter(s): return 0
  xk = load(s.rcHit, viewFrom=y)
  d = xk.p - y.p; dist2 = dot(d,d); if hybrid and sqrt(dist2) < dNear: return 0
  G_dst = |dot(xk.Ng, d/|d|)| / dist2 ;  J = G_dst / s.cache.G_src
  p1 = pdfBsdfLobe(y, dir=d/|d|, lobe=s.lobeBefore); J *= p1 / s.cache.p1_src
  f1 = bsdfCosLobe(y, d/|d|, lobe)
  if s is NEE-final at xk:    F2 = bsdfCosAll(xk, wo=-d, wi=s.rcWi)/s.lightPdf * s.Le * lightPdf/(lightPdf+pdfBsdfAll(xk,-d,s.rcWi))
  elif s final BSDF-hit at xk: p2 = pdfBsdfLobe(xk,-d,s.rcWi); F2 = bsdfCosLobe/p2 * s.Le * pdfAll/(pdfAll+s.lightPdf); J *= p2 / s.cache.p2_src
  elif xk is emitter:          F2 = s.Le * p1All/(p1All + lightPdfAt(y))          // recompute light pdf at y (see caveat)
  else:                        p2 = pdfBsdfLobe(xk,-d,s.rcWi); F2 = bsdfCosLobe/p2 * s.rcIrr; J *= p2 / s.cache.p2_src
  if f1==0 or F2==0 or !finite(J) or J<=0: return 0
  if occluded(y, xk): return 0
  return (f1/p1 * F2, J)

hybridShift(dstPixel, s):                         // random replay y1..y_{k-1}, then reconnect
  if s.rcLen == NONE: return (replayWholePath(dstPixel, s.initSeed, s.flags), 1)
  (y_{k-1}, Tp, ok) = replayPrefix(dstPixel, s.initSeed, s.rcLen)   // must NOT find an earlier connectable pair, must be rough at y_{k-1}
  if !ok: return 0
  (Frc, J) = reconnect(y_{k-1}, s);  return (Tp * Frc, J)

// ---------- Temporal (Talbot, 2 candidates) ----------   see §7.4 formulas
// ---------- Spatial (defensive pairwise, |R|=1, k neighbors) ----------
out = empty; mc_acc = 1
for i in neighbors (valid only):
   (Fi_c, Jc) = shift(central sample -> domain i)          // for canonical pair term
   n = M_i * lum(Fi_c) * Jc ;  c = M_c * lum(F_c) / k
   mc_acc += (n>0) ? c/(n+c) : 1
   (Fc_i, Ji) = shift(neighbor sample i -> central)
   m_i = (Fc_i!=0) ? (M_i*lum(F_i)/Ji) / (M_i*lum(F_i)/Ji + M_c*lum(Fc_i)/k) : 0
   out.stream(w = lum(Fc_i)*Ji*W_i*m_i, sample = shifted i with F = Fc_i), out.M += M_i
out.stream(w = lum(F_c)*1*W_c*mc_acc, sample = central), out.M += M_c
out.W = out.wsum / lum(out.F) / (numValid + 1)
pixel += out.F * out.W  (+ ReSTIR DI direct)
```

---

## 9. Implementation-relevant numbers

### 9.1 Parameters

| Parameter | Real-time | Offline | Source |
|---|---|---|---|
| Initial candidate path trees per pixel | 1 | 32 (RIS picks one path) | [P §8.3], [S §S.6] |
| Spatial reuse passes | 1 | 3 | [P §8.3] |
| Spatial neighbors (excluding self) | 3 random | 6, via low-discrepancy sequence | [P §8.3] |
| Spatial radius | 20 px | 10 px | [P §8.3] |
| Temporal reuse | on, Talbot MIS | **off** (independent frames averaged) | [P §6.5, §8.3] |
| M-cap | `M_c = 20` ("prior frame confidence ≤ 20× new samples") | n/a | [P §8.3]; code: `min(20·M_cur, M_prev)` [C TemporalReuse:139] |
| Spatial MIS | defensive pairwise (Eq. 38), `|R| = 1` | same | [P §8.3] |
| Roughness threshold | per-lobe GGX 0.2 | same | [P §8.3]; `specularRoughnessThreshold = 0.2` [C Params.slang:150] |
| Distance threshold | 1%–5% of scene size | same | [P §8.3]; code default `nearFieldDistance = 0.1` (absolute) [C Params.slang:152] |
| Max path length (scenes) | 5–11 (Fig. 15); 10 (Fig. 1) | same | [P] |
| Code defaults | `maxSurfaceBounces = 9`, NEE on, MIS balance, RR off, `separatePathBSDF = true`, hybrid shift, feature rejection on | | [C ReSTIRPTPass.h:77-113, :127-140] |
| Neighbor offset table | 8192 R2 points in a disk | | [C ReSTIRPTPass.cpp:202, :804-826] |

**Parameter exploration [S §S.6].**
- Efficiency model for dense reuse of `K` pixels with `S` candidates each, path length `L`, and resampling ray cost `u`:
  ```
  (S.106)  #rays gained / #rays computed = K·S·L / ( S·L + u·(K−1) )
  ```
  - With `S = 1` this is bounded by `L/u`.
  - With `S → ∞` it is bounded by `K`.
  - In practice large `K` or `S` eventually hurt, because path-space similarity falls with distance.
- Dense optimum: `S = 32`, `K = 49` (7×7).
- Sparse random reuse is near-optimal at `S = 32`, 2–3 rounds, 6–10 neighbors, 5–10 px radius. The chosen set is 10 px, 3 rounds, 6 neighbors.
- Random reuse has similar MSE to dense reuse but is visually closer to the reference (60 s Kitchen: MSE 2.45e-7 vs 2.37e-7, MAPE 0.0571 vs 0.0502).
- Real-time uses a 20 px radius, 1 candidate, 1 pass, 3 neighbors, because chaining over frames builds up correlation.

### 9.2 Timings and quality (RTX 3090, 1920×1080, no AA, no denoising) [P §9]

- **Fig. 1** (paths up to length 10): ReSTIR PT at 80 ms per frame. MAPE for PT / ReSTIR GI / ReSTIR PT:
  - Carousel: 1.63 / 0.45 / 0.39.
  - Opera House: 1.28 / 0.39 / 0.33.
  - That is about 16% lower MAPE than ReSTIR GI.
- **Fig. 6c** (glossy Cornell box, 1 spp): reconnection shift **5 ms**, hybrid **12 ms**.
- **Fig. 12** (lobe handling):
  - All lobes: MAPE 0.316, 51.7 ms, average pre-connection path length 2.3.
  - Random single lobe: MAPE 0.297, 38.7 ms, average pre-connection length 1.3.
- **Fig. 13:** glass bunny at equal time 25 ms. San Miguel indirect-only at 60 ms. The two shifts excel on different caustic types.
- **Fig. 14** (Kitchen, 33 ms equal time): MAPE PT 0.958, BPR 0.898, ReSTIR PT (reconnection) 0.325. BPR uses 16-pixel N-rooks tiles.
- **Fig. 15 [UNVERIFIED: read from a 130-dpi figure].** "MAPE / ms", all methods with ReSTIR DI except the first column:

  | Scene (max len) | PT (direct+indirect) | PT + ReSTIR DI | Reconnection + DI | Hybrid + DI |
  |---|---|---|---|---|
  | BistroInterior (11) | 1.0079 / 79.8 | 0.4808 / 78.3 | 0.3276 / 70.2 | 0.2999 / 65.9 |
  | Kitchen (5) | 1.1409 / 36.6 | 0.9576 / 33.9 | 0.3248 / 33.2 | 0.3933 / 31.4 |
  | VeachAjar (9) | 1.1472 / 43.3 | 1.1817 / 41.4 | 0.3192 / 45.3 | 0.2974 / 38.8 |
  | Zeroday (5) | 0.9145 / 75.3 | 0.8618 / 70.9 | 0.5332 / 68.1 | 0.5765 / 68.1 |
  | SanMiguel (5) | 0.9404 / 74.2 | 0.7725 / 75.2 | 0.5847 / 70.6 | 0.5451 / 55.3 |

  Equal-time spp shown in the insets: PT 5–9 spp, reconnection 2–3 spp, hybrid 1 spp.
  - ReSTIR PT achieves **24%–75% lower MAPE** than path tracing, both with ReSTIR DI.
  - Hybrid helps glossy and refractive surfaces (wine glasses, mirror, metal) and noise at geometric edges.
  - Reconnection wins on rough surfaces, where cheaper samples are better, and with color noise.
  - Images were captured during camera motion.
- **Offline (5–640 s, Fig. 16–18).** Reaching the same MAPE, BPR is up to 2.8× faster than PT, ReSTIR PT (reconnection) up to 14.4× faster, and ReSTIR PT (hybrid) up to 10× faster.
  - 5 s VeachAjar MAPE: PT 0.1587, BPR 0.1432, ReSTIR GI 0.1308, reconnection 0.0925, hybrid 0.0706.
  - ZeroDay: 0.2320 / 0.2511 / 0.4657 / 0.1951 / 0.2098.
  - The hybrid advantage partly vanishes offline, because temporal reuse is off, and thread divergence costs time. At equal sample count (Fig. 18) hybrid is generally better except for distant-highlight caustics.

**Error metric [P footnote 13, p.75:18; `UNVERIFIED` exact placement of tildes]:**

```
MAPE(I, I_gt) = mean( |I − Ĩ_gt| / (0.01·mean(Ĩ_gt) + Ĩ_gt) ) ,   Ĩ_gt = grayscale ground truth
```

"L1 metric … more resistant to occasional fireflies." It is computed on HDR images.

### 9.3 Memory

- **88 B per reservoir** [P §8.2].
- **[INFERENCE]** At 1920×1080 that is 2,073,600 × 88 B ≈ 182.5 MB per reservoir buffer. You need at least 3 buffers (current/output, temporal history, spatial ping-pong), about 550 MB.
- The hybrid shift adds `ReconnectionData` at 40 B × 2 × k per pixel (256 B padded for k = 3), about 530 MB at 1080p. The reference code stores full per-neighbor retrace results in a separate pass.
- At 1280×720 everything is about 2.25× smaller.
- For WebGPU:
  - Plan to compress the reservoir. Candidates: `F` as RGB9E5 or f16×3, directions octahedral 2×16, barycentrics unorm16×2, `M` and `W` as f32, seeds u32, cached terms f16.
  - Consider fusing the retrace into the reuse pass instead of storing `ReconnectionData`.
  - Check `maxStorageBufferBindingSize` and `maxBufferSize` on the target adapter `[UNVERIFIED]`: the WebGPU defaults are 128 MiB and 256 MiB, and higher limits must be requested explicitly.

---

## 10. Validation hooks for the Blender/Cycles ground truth [INFERENCE unless tagged]

1. **Match what ReSTIR PT integrates.** The GRIS pass only produces paths with ≥ 2 scattering vertices. Direct light at `x_1` must come from a separate estimator (ReSTIR DI or plain NEE+MIS), plus delta and transmission direct light (`LDeltaDirect`) [C]. For validation, first compare "PT (direct + indirect)" against Cycles. Then compare ReSTIR PT indirect + DI, with the same bounce limits. Cycles "max bounces" counts differently from `pathLength`: verify on a Cornell box.
2. **Pixel footprint.** The paper and code use a **fixed primary hit at the pixel center** (no AA) [P §9.3; C V-buffer]. A Cycles reference with the default 1.5 px Blackman-Harris filter will differ at edges and texture detail. Either render Cycles with a very small filter width, or add primary-ray jitter to both (then the shift domains start at `x_0` and `x_1` varies per frame, which is still fine since `Ω_i` is per-frame).
3. **Unbiasedness tests** that follow directly from the theory:
   - (a) The spatial-only "offline" mode averaged over N independent frames must converge to the PT reference [P §6.5; Fig. 9b].
   - (b) Swapping pairwise ↔ Talbot MIS must not change the mean [P Fig. 7].
   - (c) Setting the distance threshold very large (the hybrid degenerates to pure random replay, Jacobian 1) or tiny must not change the mean.
   - (d) The reconnection shift and hybrid shift must converge to the same mean [P Fig. 11].
   - (e) With `m_i = 1/M` constant weights and occluders, expect **bias**. This is a useful negative test that the MIS machinery matters.
   - (f) With the M-cap removed, expect Fig. 10b-style wrong convergence in a static scene.
   - Use the MAPE of §9.2 and also a signed mean-error image, since bias shows up as a nonzero mean over many frames.
4. **Target choice does not affect unbiasedness.** Luminance with Rec.601 weights (code) or Rec.709 both work. Only variance changes [P §4.2–4.4].
5. **Materials.** Keep validation scenes to what both renderers model identically: Lambert + GGX (Principled BSDF subset), emissive triangles, point, spot and area lights. The lobe-specific shift needs a clean per-lobe decomposition that matches the WGSL BSDF, and Cycles' Principled BSDF layering differs.

---

## 11. Risks and pitfalls

1. **Bijectivity of the hybrid shift.** Replay must use the same connectability predicate, the same lobe choice, and the same random-number consumption as base-path generation. Any asymmetry (e.g. using `x_i` roughness but `y_i` shading normal, or different float epsilons) gives silent bias. Unit-test `T_{j→k}(T_{k→j}(x̄)) = x̄` on the GPU.
2. **Light-PDF-at-previous-vertex caveat** (§6.8) for the emitter-as-rc-vertex case. Recompute it, or store it in area measure.
3. **Dynamic scenes:** stale suffix radiance after moving lights. Use suffix re-tracing (`traceTemporalUpdate`), or accept bias. Temporal MIS in a moving scene needs previous-frame geometry and lights to be exact.
4. **Memory and bandwidth.** 88 B reservoirs plus retrace data at 1080p exceed WebGPU default binding limits. Plan compression and tiling.
5. **Thread divergence.** The hybrid replay has variable length, so compute-shader BVH traversal makes this worse than on RTX. The paper notes divergence already offsets the hybrid's benefits [P §9.3].
6. **Color noise.** A grayscale target samples brightness only [P §10.1].
7. **Undersampling** of tiny geometry or sharp caustics causes streaks and splotches [P §10.1].
8. **Delta vertices** before or after the rc vertex are never reconnectable [C Shift.slang:442]. Paths through mirrors or glass rely on replay (hybrid) or fail (reconnection shift).
9. **Normals.** Jacobian cosines use the **geometric** normal (`faceN`) [P Eq. 52; C]. Using shading normals breaks the Jacobian.

---

## 12. Open questions

1. Exact rendering of Eq. 52's `θ_2` subscript and the MAPE tilde placement (footnote 13) should be confirmed at higher DPI.
2. Whether Falcor's `linearRoughness` equals glTF perceptual roughness.
3. Whether the direct-lighting pass also adds primary-hit emission, which the code excludes from path `L` when direct illumination is disabled. `[UNVERIFIED]`
4. Whether the reverse temporal shift should use previous-frame BVH and lights for exactness in our dynamic-light use case, and whether the cost is acceptable.
5. What the right WebGPU reservoir compression is, and whether to fuse the retrace pass into the reuse pass to avoid storing `ReconnectionData`.
