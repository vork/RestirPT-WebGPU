# Adversarial verification: "ReSTIR PT Enhanced" research report (topic `enhanced-paper`)

**Report under review:** `/Users/mark.boss/.claude/plans/do-a-deep-dive-shimmering-ritchie-agent-a91ab1d861819b276.md` (872 lines).

**Where these notes live.** The orchestrator asked for `scratchpad/research/enhanced-paper.verify.md`. This verifier also ran in **plan mode**, where the only writable file is this plan file, so the notes are here instead. Nothing was written to the scratchpad or the project directory.

**Primary sources checked.**
- Main paper page images `scratchpad/pages/enhanced/p-01..p-18.png` (all read) and text `scratchpad/restirpt_enhanced_2026.txt`.
- Supplemental PDF, streamed with `curl | gs -sDEVICE=txtwrite -o - -` (not saved). The harness kept a copy of that tool output at `~/.claude/projects/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-.../tool-results/bkusnhdet.txt`.
- GRIS 2022 page images `scratchpad/pages/gris/p-10.png` (Eq.38) and `p-15.png` (Eq.52 and §8.3).
- Code22 (github.com/DQLin/ReSTIR_PT, HEAD) via raw.githubusercontent:
  - `SpatialReuse.cs.slang`, `Shift.slang`, `PathReservoir.slang`, `PathTracer.slang`, `Params.slang`, `ReSTIRPTPass.h`, `TemporalReuse.cs.slang`
  - Falcor `StandardMaterial.slang`, `ShadingData.slang`
- RTXDI-Library (HEAD):
  - `PT/PathReconnectibility.hlsli`, `PT/InitialSamplingPathTracerContext.hlsli`, `PT/HybridShiftPathTracerContext.hlsli`, `PT/HybridShift.hlsli`, `PT/TemporalResampling.hlsli`, `PT/DuplicationMap.hlsli`, `PT/Reservoir.hlsli`, `PT/SpatialNeighborSelection.hlsli`
  - `Utils/Math.hlsli`, `Source/ReSTIRPT.cpp`, `LICENSE.txt`
- WebGPU spec (w3.org/TR/webgpu) for limits and storage-texture formats.
- Pure-Python simulation (stdin, no files) of the pairing-texture shuffle, used to test P-Eq.3.

**Confidence tags.** HIGH means checked directly against the source. MED means derived or strongly indicated. LOW means plausible.

---

## 0. Verdict

The report is **high quality and mostly accurate**. I checked every table number against P-Table 1, S-Table 1/2, the figure captions and S-Fig.13, and all match.

Equations P-Eq.1, 2, 4a/b, 5 and 6a/b match the page images. So do S-Eq.4, 5, 6, 7, 10–19, 25, 26 and 27, the constants (c = 0.02, α_min = 0.2, 17×17 window, /288, α = 0.1, c_Cap^min = 1, 128×1024 light tiles, 8×8 screen tiles, 32/B²) and S-Alg.1.

The measure conventions are right:
- G has its cosine at the receiving vertex.
- Footprints are in area measure.
- W^RIS is area-measure and p1 is the conversion Jacobian.

The report also correctly spots three inconsistencies in the sources:
- P-Eq.3 cannot give 1 at σ = 0.8.
- S-Table 2 has an inconsistent Veach Ajar number.
- The P-p.4 wording puts the cosine of G at the wrong vertex.

**Problems found.**
- **Two design statements could make an implementation silently biased** (§1, C3 and C4). One misleading claim about asymmetric neighbor validity in the paired pre-pass. One missing requirement to update the cached Jacobian of a selected shifted sample.
- **Several bias-relevant implementation details are missing** (§2): geometric vs shading normal in the Jacobian, the index range of the RR product, and lobe-joint vs marginal PDFs in the Jacobian.
- **P-Eq.3 is misdiagnosed.** It is almost certainly a sign typo (−1.76σ⁻²), and the report's fallback `round(σ²/2)` is wrong for small σ (C1).
- A handful of minor attribution and API issues.

---

## 1. Corrections (errors or misleading statements in the report)

### C1. P-Eq.3 (n_σ fit): a probable sign typo, not an unusable fit. The report's fallback is wrong for small σ.

**Severity:** minor for the default σ = 16; wrong for σ < ~1.3. **Confidence:** MED-HIGH that it is a typo.

**What the source prints** (p-05.png; text line 239): `n_σ = ⌊σ²/2 + 1.46σ⁻¹ + 1.76σ⁻² + 0.656σ⁻³ + 0.5⌋ ≈ ⌊σ²/2 + 0.5⌋`, all "+". The report transcribes this correctly and correctly notes that it gives 6 at σ = 0.8 and has a minimum of about 3.54 near σ ≈ 1.6.

**What the report recommends.** `n = round(σ²/2)` "or an empirical lookup table". round(σ²/2) returns **0** for the one-shuffle σ = 0.814 and 2 for σ = 1.78, where the true n is 1 and 2. It is fine only for σ ≳ 1.3.

**Independent check.** I simulated the exact procedure from P-§3.1 and P-Fig.3:
- consecutive link indices over horizontal pixel pairs;
- 2×2 block random permutations;
- a (1,1) grid offset on odd passes;
- torus wrap;
- per-axis RMS of the wrapped pair delta.

| n shuffles | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 20 | 50 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| measured σ (px) | 0.814 | 1.778 | 2.284 | 2.709 | 3.069 | 3.382 | 3.670 | 3.942 | 4.198 | 4.434 | 4.651 | 4.862 | 6.278 | 9.997 | 15.984 |
| printed Eq.3 (floor) | 6 | 3 | 4 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 20 | 50 | 128 |
| Eq.3 with **−1.76σ⁻²** (floor) | **1** | **2** | **3** | **4** | **5** | **6** | **7** | **8** | **9** | **10** | **11** | **12** | **20** | **50** | **128** |

The pre-floor value with −1.76 is n + (0.46…0.60) for every n, so it is a fit of n + 0.5 followed by a floor. The single-walk variance progression the report derives (0.25, 1.25, 2.25) is correct.

**Corrected formula** [INFERENCE: probable typo in the source]:
```
n_σ = ⌊ σ²/2 + 1.46·σ⁻¹ − 1.76·σ⁻² + 0.656·σ⁻³ + 0.5 ⌋
```
For σ = 16: ⌊128 + 0.0913 − 0.0069 + 0.0002 + 0.5⌋ = 128. This is the same as the report's value, so the default is unaffected. Measured σ(128) = 15.98.

---

### C2. Pairwise-MIS "transcription from Code22" uses c_c/k. Code22 uses c_c/N with N = the configured neighbor count.

**Severity:** minor; not a bias issue. **Confidence:** HIGH.

**Evidence.**
- `SpatialReuse.cs.slang:379` and `:396` divide `centralReservoir.M * p̂` by `neighborCount`.
- `:182` sets `neighborCount = getNeighborCount()`, and `:102` returns `gNeighborCount` (3, including invalid or off-screen slots).
- `:408` divides the final W by `(validNeighborCount + 1)`.

**Corrected transcription.** k is the number of valid neighbors and N the number of neighbor slots:
```
m_j(Y_j) = 1/(k+1) · c_j·p̂_j(X_j) / ( c_j·p̂_j(X_j) + (c_c/N)·p̂_c(Y_j)·J_{j→c} )
m_c(X_c) = 1/(k+1) · [ 1 + Σ_{j valid} (c_c/N)·p̂_c(X_c) / ( (c_c/N)·p̂_c(X_c) + c_j·p̂_j(Z_j)·J_{c→j} ) ]
```

**Independent derivation.** For any positive constant a in place of c_c/N, and any y in c's domain:
```
m_c(y) + Σ_j m_j(y) = (1/(k+1)) · [ 1 + Σ_j (a p̂_c(y) + c_j p̂_←j(y)) / (a p̂_c(y) + c_j p̂_←j(y)) ] = 1
```
Here p̂_←j(y) = p̂_j(T_{c→j}(y))·|J_{c→j}| and p̂_j(X_j)/J_{j→c} = p̂_←j(Y_j). Both variants are therefore valid partitions of unity. With c ≡ 1 they reduce to G-Eq.38 with |R| = 1 and M = k+1 (checked on gris p-10.png).

---

### C3. The paired pre-pass with an asymmetric validity test is biased. "Correctness does not require symmetry" is wrong for the S1/S2 design as written.

**Severity:** major (silent darkening bias). **Confidence:** HIGH for the math; MED that an implementer would read S1 this way.

**What the report says.**
- §2.2: "Code22 uses `dot(n_c,n_j) ≥ 0.5 && |d_c − d_j| < 0.1·d_c` … asymmetric. Correctness does not require symmetry: each pixel's MIS uses only its own accepted set."
- §2.6 S1: "find the partner q_t; run the validity test; if valid, compute T_{p→q_t}(X_p)".

**Why that fails.** Suppose pixel p runs *its own* test in S1. Take a pair where accept(c, j) is true but accept(j, c) is false:
- Pixel j never writes T_{j→c}(X_j) into its slot.
- Pixel c still counts j in k and in m_c(·), but finds no candidate, so w_j = 0.

The estimator's expectation then becomes
```
E[F(Y)·W_Y] = ∫ f_c(y) · [ m_c(y) + Σ_{j ∈ S_c, slot computed} m_j(y) ] dy  <  ∫ f_c(y) dy
```
The missing m_j mass is lost. This is energy loss that grows with depth discontinuities.

**Correct rule.**
- Use a **symmetric** acceptance predicate A(p,q) = A(q,p), and evaluate the identical predicate in S1 and S2. Example: `dot(n_p,n_q) ≥ 0.5 && |d_p−d_q| < 0.1·max(d_p,d_q)`.
- Alternatively, compute the S1 shift whenever A(p,q) || A(q,p).
- In S2, never treat "slot not computed" as "shift failed". A genuine shift failure (T undefined, visibility 0, invertibility failure) correctly contributes p̂ = 0 on *both* sides, which keeps the partition of unity.

---

### C4. A selected shifted sample must carry the offset path's cached Jacobian product. The slot and resampling-pass design omits this.

**Severity:** major (silent bias in later frames). **Confidence:** HIGH.

**What the sources say.**
- **S-Alg.1** stores `rcVertexCachedValues.x` = product of the base path's Jacobian factors. In P-Eq.2 indexing that is p^x_{k−1}(ω_{k−1})·G(x_{k−1}→x_k)·p^x_k(ω_k) (with p_k := 1 for the NEE or light case, see C9). This product is the **denominator** of the next shift that uses this reservoir as its base path.
- **Code22** overwrites the cached terms with the *offset* path's values during a shift (`Shift.slang:425` and `:569`: `srcReservoir.cachedJacobian = dstCachedJacobian`). The resampling merge then copies them from the shifted temp reservoir (`Shift.slang:620–623`, `PathReservoir.slang:372`).

**What the report omits.** §2.6 S2 says "read … the partner reservoirs (W, c, p̂, and the fields to copy if selected)". The slot holds only `F·J, J`. If S2 copies the partner's reservoir wholesale, the cached product is the partner-domain one, so every later temporal or spatial Jacobian is wrong. That is bias, not just noise.

**Correct rule** [INFERENCE, consistent with Code22]: when Y_j = T_{j→c}(X_j) is selected, set
```
cached_{Y_j} = p^y_{k−1}(ω'_{k−1}) · G(y_{k−1}→x_k) · p^y_k(ω_k)  ( = J_{j→c} · cached_{X_j}  when J was computed as num/den of exactly these factors )
F_out = F_c(Y_j)
```
- Keep every other offset-independent field from X_j's reservoir: rc hit, rcVertexWi, rcVertexRadiance, seeds, flags, and the NEE light PDF when light sampling is area-measure.
- Store cached_{Y_j} in the slot, or derive it as J × cached_{X_j}.
- For solid-angle light sampling, also update the light PDF.

---

### C5. P-§6.2.1 NSight numbers are attributed to §6.2.1 alone

**Severity:** minor. **Confidence:** HIGH (p-12.png).

P-§7.1 says the optimizations "in Section 6.2.1–6.2.3" (micro-optimisation plus compaction plus forced NEE) take occupancy from 22.4% to 31.1%, threads per warp from 15.3 to 19.9, and latency from 347k to 241k. The report's §6.1 lists them under "Low-level code optimisation (P-§6.2.1)" next to the 35.73 → 32.98 ms Table 1 row, which belongs to §6.2.1 alone.

---

### C6. "'ReSTIR DI and others' drops from 5.05 ms to 1.00 ms after unification" (report §7)

**Severity:** minor. **Confidence:** HIGH.

The row before "Unify DI & GI" is "+Russian roulette" at **5.24** ms (P-Table 1; S-Table 1 average). 5.05 ms is the baseline.

---

### C7. Pairing-texture storage wording

**Severity:** minor. **Confidence:** HIGH.

P-§3.1 (p-06.png) says "we pack these x and y directional deltas into a **16-bit luminance texture**", i.e. one 16-bit channel holding two 8-bit deltas. P-§7.2 says "two-channel tilable images". The report merges these into "16-bit two-channel texture", which is neither. `rg8sint` (2×8-bit), as the report recommends, is equivalent and correct for W ≤ 254.

---

### C8. The cosine in G: the supplement repeats the paper's wording

**Severity:** minor; the report's conclusion is correct. **Confidence:** HIGH.

The report calls the P-p.4 phrase ("θ is the angle between y_{k−1}x_k and **y_{k−1}'s** normal") a one-off slip. **S-§1** (after S-Eq.4) repeats the same wording: "the angle made by y_{k−1}x_k and the normal at y_{k−1}". So it is systematic in the text. The math still requires the cosine at the *receiving* vertex (x_k):
- **Derivation.** The PSS Jacobian is the ratio of area densities of x_k, and p_σ(ω)·|cos θ_{x_k}|/d² is the area density at x_k. The factor p^x_k·G(x_k→x_{k+1}) cancels because x_{k+1} is shared.
- **GRIS Eq.52** (gris p-15.png): "θ^•_2 the angle between ω^•_i and the **geometric surface normal at x_{i+1} = y_{i+1}**".
- **Code22** `Shift.slang:452/268`: `shifted_cosine = abs(dot(rcVertexSd.faceN, -dstConnectionV))`.
- **RTXDI** `RTXDI_CalculateRayFootprint(n_{x_k}, viewDir_{x_k}, t, pdf)` for the ray footprint, and `n_{x_{k−1}}` for the inverse footprint.

**Corrected definition:**
```
G(a→b) = |⟨n^geo_b, (a−b)/‖a−b‖⟩| / ‖a−b‖²
```

---

### C9. S-Alg.1 Jacobian-factor indices are copied without mapping them to P-Eq.2

**Severity:** minor. **Confidence:** HIGH (supplement text).

S-Alg.1 writes the factors as "p(ω_k), G(x_{k−1},x_k), p(ω_{k+1})", indexing a direction by the vertex it arrives at. In P-Eq.2 and S-Eq.4 notation these are **p^x_{k−1}(ω_{k−1}), G(x_{k−1}→x_k), p^x_k(ω_k)**. The report's §6.2 table copies S-Alg.1 verbatim, which can mislead an implementer reading §1.3's notation.

---

### C10. WebGPU format caveat for the duplication map output

**Severity:** minor (API). **Confidence:** HIGH (w3.org/TR/webgpu §25.19 and the format table).

- The report suggests writing the duplication map from a compute shader to "an `r8unorm` or `r16float` texture". Neither is a core `STORAGE_BINDING` format. Both need the optional feature `"texture-formats-tier1"`.
- **Core alternatives:** an `r32uint` or `r32float` storage texture, or a `u32` storage buffer.
- If you use 8-bit storage, clamp the count. The maximum possible count is 288 > 255, and RTXDI clamps at `RTXDI_PT_DUPMAP_MAX_COUNT 255u` (DuplicationMap.hlsli:119).
- The other limits quoted are correct: `maxStorageBufferBindingSize` 128 MiB default, `maxComputeWorkgroupStorageSize` 16384, `maxStorageBuffersPerShaderStage` 8, and 1080p × 64 B = 132,710,400 B < 134,217,728 B.

---

### C11. Unverified items the report left open, now resolved

**Confidence:** HIGH.

| Open item | Resolution |
|---|---|
| Code22 license (report §9, "not checked") | **BSD-3-Clause**, per the GitHub license API (`LICENSE.md`) |
| RR survival formula (report §6.5, "[UNVERIFIED]") | The paper does not give q_i. Code22 uses `prob_kill = max(0, 1 − luminance(throughput))`, i.e. **q_i = min(1, luminance(throughput_i))** (`PathTracer.slang:919–931`); Code22 has RR off by default. Enhanced's q is still unstated. |
| RTXDI jitter (report §3.10) | Confirmed relative: `Z·stddev·mean + mean` (`Utils/Math.hlsli:93–101`). minConnectionFootprint 0.02 with σ_rel 0.2, and minPdfRoughness 0.1 with σ_rel 0.01 (`ReSTIRPT.cpp:56–59`). |
| RTXDI footprint threshold | `R_pri² × 0.02²` (`PathReconnectibility.hlsli:37, 52`), so the "paper-equivalent c ≈ 0.04 (2× the paper)" is **correct** |
| RTXDI duplication cap | `max(1,(int)lerp(maxHistory, 1, pow(saturate(dup/288), 0.1·2^{6(1−s)−3})))` (`TemporalResampling.hlsli:405–414`); **correct** as reported |

---

## 2. Omissions (bias-relevant items missing from the report)

### O1. Use the geometric normal in the reconnection Jacobian

**Confidence:** HIGH.

- GRIS Eq.52 and Code22 (`faceN`) use the geometric normal at x_k for the cosine ratio.
- glTF and USD assets usually carry interpolated shading normals, often with normal maps. A shading-normal cosine makes |∂T/∂ū| inexact, which is bias.
- For the footprint tests, which are heuristics, either normal works, but the same one must be used in base-path evaluation and in the offset-path invertibility re-evaluation.

### O2. The Jacobian must use the lobe-joint PDF p(ω,ℓ). The marginal PDF is only for the footprint test and the NEE/BSDF path MIS.

**Confidence:** HIGH.

- S-Eq.4 uses the joint p(ω_k, ℓ_k | x_k, −ω_{k−1}), which includes the lobe-selection probability.
- Code22 `Shift.slang:478`: `dstPDF1 = evalPdfBSDF(sd, dir, dstPDF1All, allowedSampledTypes1)`. The lobe-restricted value goes into J; `dstPDF1All` (marginal) is used only in `evalMIS`. The integrand is also lobe-restricted (`evalBSDFCosine(..., allowedSampledTypes)`).
- The report's §3.5 correctly says the *footprint* uses marginal PDFs. It never warns that the *Jacobian* must not. With mixture sampling, a marginal-PDF Jacobian is not the PSS Jacobian of the lobe-indexed mapping, because a direction can be produced by several lobes.

### O3. Index range of the RR PSS source PDF

**Confidence:** MED [INFERENCE].

- S-§6 writes p(ū) = ∏_{i=1}^{d−1} q_i(ū) generically.
- In a path-tree tracer, NEE at x_i is normally done *before* the RR test at x_i. Then an NEE candidate ending at x_d (sampled from x_{d−1}) survived only q_1..q_{d−2}. A BSDF-sampled emitter hit at x_d survived q_1..q_{d−1}.
- The product must contain **exactly** the RR tests the candidate had to survive. An off-by-one gives a bright or dark bias per technique.
- Combined UCW for an NEE candidate with RIS-NEE: `W = W^RIS(x_d)·p1(x_d) / ∏_{RR tests survived} q_i`.
- The q_i may depend on the prefix throughput (it is a function of ū). Replay must not apply RR.

### O4. Invertibility re-evaluation needs a defined p_k(ω_k) for NEE-terminated paths

**Confidence:** MED.

- For an NEE path whose rc candidate is x_{d−1}, the "next direction" ω_{d−1} is light-sampled, so the Jacobian does not change it (p_k := 1). The integrand still changes through the path-MIS weight.
- RTXDI evaluates the inverse footprint at x_{d−1} with the **BSDF PDF of the NEE direction** (`InitialSamplingPathTracerContext.hlsli`, RecordNeeLightSample: `brs.outPdf = scatterPdf`).
- Whatever definition you choose, the initial-sampling decision and the offset-path invertibility check must use the **same** one. Otherwise the shift is not bijective, which is bias.
- The report's §3.11 pseudo-code does not specify this case.

### O5. RTXDI behaviours to avoid copying blindly, especially for Cycles validation

**Confidence:** HIGH.

1. **Firefly filter.** `fireflyReplacementFilterStrength = 0.7` (`ReSTIRPT.cpp:44`) and the boiling filter (`boilingFilterStrength = 0.2`, `:96`) are biased. Disable them for validation.
2. **Emitter hits.** For BSDF-sampled emitter hits, RTXDI picks the rc vertex from `IsLastVertexFar()` alone, with **no roughness test at x_{k−1}** (`RecordEmissiveLightSample`). The paper applies the single-vertex roughness test (P-§4.2). The report's pseudo-code follows the paper, which is correct; just note the divergence.
3. **Footprint constant.** RTXDI defaults to 2× the paper's value (see C11).

### O6. Code22's optional Jacobian-rejection heuristic

**Confidence:** HIGH.

- `rejectShiftBasedOnJacobian` (default **false**) rejects a shift when `max(J, 1/J) > 1 + 10` (`Params.slang`; `Shift.slang:556–563`).
- It is symmetric in J ↔ 1/J. Applied identically in both shift directions it just restricts the shift domain, so it stays unbiased. If you use it, keep it symmetric.

### O7. S-Eq.22 is printed with the relation reversed; say so explicitly

**Confidence:** MED-HIGH (text extraction).

- The supplement prints `p(−ω_{k−1}|x_k,ω_k) = p(x_{k−1}|x_k,ω_k)·G(x_k→x_{k−1})`. That is dimensionally wrong: a solid-angle density cannot equal an area density times G.
- The correct relation, which the report uses, is `p_A(x_{k−1}|x_k,ω_k) = p_σ(−ω_{k−1}|x_k,ω_k)·G(x_k→x_{k−1})`.
- The report says "reordered for dimensional consistency"; it should state plainly that the printed equation is a typo.

### O8. "Diffuse x_k skips the inverse test" must mean the whole-BSDF marginal PDF is independent of ω_{k−1}

**Confidence:** MED.

- P-footnote 6 skips the inverse test for diffuse x_k.
- A diffuse *lobe* inside a multi-lobe material whose lobe-selection probability depends on the view direction (Fresnel or albedo weighting) does **not** qualify, because the marginal PDF used in the test changes.
- The report's pseudo-code says "diffuse-only", which is correct. Keep that restriction.

### O9. Reservoir compression makes the base path slightly inconsistent

**Confidence:** LOW-MED.

- With 16-bit unorm barycentrics and octahedral ω_i, the decoded rc vertex differs slightly from the one used to compute the stored F and cached Jacobian product.
- The paper reports no quality impact, but for strict unbiasedness tests against Cycles, either recompute F and the cached product from the decoded values, or store full precision in validation builds.

---

## 3. Items verified as correct (evidence)

**Main paper.**
- **P-Eq.1** (p-03): `w_i = m_i(Y_i)·p̂(Y_i)·W_{X_i}·|∂T_i/∂X_i|`.
- **§2.2** statements: c_Cap gives `min(c_Cap, c_temp)+1`; spatial reuse uses a 30 px disk and sums confidences.
- **P-Eq.2** (p-04), with p^x_k := 1 for k = d, and J = 1 without reconnection.
- **P-Eq.4a/4b, 5, 6a/6b** (p-06, p-07), including the RHS `(c/100)·‖x0−x1‖² / (⟨n_{x1}, x̂1x0⟩/(4π))`, c = 0.02 and c/100 = 0.0002.
- Single-vertex roughness `α_{x_{k−1}} ≥ α_min` (p-07, p-14; α = 0.2).
- Footnotes 1–7: σ ≥ 0.8 from one shuffle; W = 256 needs 16 bits; near-periods within 3840 px; geometry and NDF cancel; NDF reciprocity is exact and VNDF approximate; diffuse or emissive x_k skips the inverse test; chroma noise is uncorrelated.
- **§3.2 sizes** 254, 230, 210, and the random flip/mirror/transpose/offset per frame.
- **§5 algorithm:** 17×17, /288, lerp(c^Default, c^min, D^α), c^min = 1, α = 0.1, 3.25% bias.
- **§6.1:** NEE from x1, light tiles 128×1024, 8×8 screen tiles.
- **§6.3:** vector weight `w_i = m_i F(Y_i) W_{X_i}|∂T/∂X_i|`, shading = Σ w_i.
  - I re-derived its unbiasedness: E_i[F(Y_i)Σw/p̂(Y_i)] = Σ_i w_i F(Y_i)/p̂(Y_i) = Σ_i m_i F(Y_i) W_i J_i. The report is **correct**.
  - Divide the RGB sum by (k+1) too when using Code22-style unnormalised pairwise weights.
- **§7:** σ = sqrt(8/(9π))R ≈ 0.5319R, so 30 → 15.96; 32 light candidates at the primary hit and 32/B² thereafter.
- **All numbers** in Table 1, Fig.5, Fig.8–12, Fig.13 and Fig.9.

**Supplement.**
- S-Eq.1–4 and the lobe-indexed definitions.
- S-Eq.5 in (max(·))⁻¹ form.
- S-Eq.6–19 and 25; renaming c2 to c_s for the smoothness constant is fine.
- S-Eq.26 is printed as `1/(p^x_{k−1}(ω_{k−1}))² ≥ α_min`, and the text says "reciprocal squared solid angle PDF". The report's scale-mismatch analysis is arithmetically correct: the GGX peak PDF is 1/(4πα_ggx²), giving thresholds of about 0.43, 0.24 and 0.17 perceptual.
- S-Eq.27 (/(8π²), i.e. Eq.5 × 2π).
- S-§5: ω1 = M p1/(M p1 + p2) and W = W^RIS·p1.
- S-§6: p(ū) = ∏ q_i.
- S-Alg.1 field list totalling 64 B.
- S-Table 1/2 and the report's inconsistency analysis of S-Table 2: 3.57× matches 9.89 and 4.03× matches 15.53. I recomputed both.
- S-Fig.1–6, 8–13.

**GRIS.**
- Eq.38 defensive pairwise MIS; the report's c ≡ 1 reduction is correct.
- Eq.52 cosine at x_{i+1} (geometric normal).
- §8.3: 20 px radius, M_c = 20, GGX roughness 0.2, 1–5% scene size, pairwise for spatial and Talbot for temporal.

**Code22.**
- Defaults: specularRoughnessThreshold 0.2, nearFieldDistance 0.1, maxSurfaceBounces 9, useRussianRoulette false, radius 20, 3 neighbors, history 20, temporal Talbot, power emissive sampler.
- `isValidGeometry` as quoted (`SpatialReuse.cs.slang:77`).
- `alpha = linearRoughness²` (`StandardMaterial.slang:840`).
- p̂ = `dot(F, (0.299, 0.587, 0.114))` (Rec.601 luma, `PathReservoir.slang:280`). The report says "luminance"; any fixed scalarisation is fine if used consistently.

**Project page.** Only the PDF, supplement and mp4 are listed; there is no code.

---

## 4. Independent re-derivations (for implementers)

**(a) Hybrid-shift PSS Jacobian.** x̄ is the base path and ȳ = T(x̄). The prefix y_1..y_{k−1} is replayed with the same ū, and x_k onward is copied.
```
|∂ū^y/∂ū^x| = [p_A^y(x_k | y_{k−1}) · p_A^y(x_{k+1} | x_k, y_{k−1})] / [p_A^x(x_k | x_{k−1}) · p_A^x(x_{k+1} | x_k, x_{k−1})]
p_A(x_k | a) = p_σ(ω_{a→x_k}) · |cos θ^geo_{x_k}(a)| / ‖x_k − a‖²
p_A(x_{k+1} | x_k, a) = p_σ(ω_k | x_k, −ω_{a→x_k}) · |cos θ_{x_{k+1}}| / ‖x_{k+1}−x_k‖²
```
The second G is identical for both paths and cancels, which yields P-Eq.2 with the cosine at x_k.

Special cases:
- **Light vertex** (k = d): drop the p_k factor.
- **Area-sampled NEE vertex:** p_σ·G = p_area, which is shading-point independent, so J = 1 (with an independent light-selection PMF).
- **Environment vertex:** J = p^y_{k−1}(ω)/p^x_{k−1}(ω), with the direction fixed.

**(b) Partition of unity of the confidence-weighted pairwise MIS.** See C2.

**(c) Bias from the asymmetric paired pre-pass.** See C3.

**(d) Footprint threshold in code form.** Matches the report's §3.11 and RTXDI:
```
thr   = (c/100) · ‖x1−x0‖² · 4π / |⟨n_{x1}, normalize(x0−x1)⟩|
rayFP = ‖x_k−x_{k−1}‖² / (p_σ(ω_{k−1}) · |cos θ_{x_k}|)
invFP = ‖x_k−x_{k−1}‖² / (p_σ(ω_k | x_k, −ω_{k−1}) · |cos θ_{x_{k−1}}|)
```
- Delta lobes give footprint 0.
- Diffuse-only or emissive x_k skips invFP.
- The environment gives rayFP = ∞, so only the roughness test at x_{k−1} guards it.

**(e) Duplication-cap numbers.** For D = 1/288, 10/288 and 0.2, c_Cap = 9.22, 6.42 and 3.83. I recomputed these; they match the report.

---

## 5. Summary list for the orchestrator

| # | Item | Severity | Confidence |
|---|---|---|---|
| C3 | Asymmetric validity in paired pre-pass → bias; require a symmetric predicate or compute on A(p,q) \|\| A(q,p) | major | HIGH (math) |
| C4 | Selected shifted sample must carry the offset cached-Jacobian product (J·cached) | major | HIGH |
| C1 | Eq.3 probable typo: −1.76σ⁻²; round(σ²/2) is wrong for σ < 1.3 | minor | MED-HIGH |
| C2 | Code22 pairwise uses c_c/N, not c_c/k (both unbiased) | minor | HIGH |
| C5 | NSight numbers belong to §6.2.1–6.2.3 combined | minor | HIGH |
| C6 | DI+others drops 5.24 → 1.00, not 5.05 → 1.00 | minor | HIGH |
| C7 | Paper says "16-bit luminance texture" (§3.1) vs "two-channel" (§7.2) | minor | HIGH |
| C8 | Supplement repeats the "normal at y_{k−1}" wording; the math needs the geometric normal at x_k | minor | HIGH |
| C9 | S-Alg.1 factor indices ≠ Eq.2 indices | minor | HIGH |
| C10 | r8unorm and r16float storage textures need texture-formats-tier1; clamp the count at 255 | minor | HIGH |
| C11 | Code22 is BSD-3; Code22 RR q = min(1, lum(thp)) | minor | HIGH |
| O1–O9 | geometric normal; lobe-joint PDF in J; RR product index range; NEE p_k consistency; RTXDI biased filters; Jacobian-rejection symmetry; S-Eq.22 typo; the "diffuse" definition; compression consistency | — | see above |
