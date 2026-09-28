# Adversarial verification of the "gris-math" report

**Target report:** `/Users/mark.boss/.claude/plans/do-a-deep-dive-shimmering-ritchie-agent-a7f08dec9968f0324.md` (GRIS / ReSTIR PT math deep dive).

> **Location note.** The orchestrator asked for these notes at
> `/private/tmp/claude-502/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-ea73-491f-b7b7-6d0aab1ffce1/scratchpad/research/gris-math.verify.md`.
> This session runs in **plan mode**, where the only writable file is this plan file. Copy it to the intended path verbatim.

---

## 0. Verdict

The report is **largely accurate**, and its paper transcription is unusually faithful. Every numbered equation I re-read on the page PNGs matches. Every reference-code `file:line` citation I spot-checked (about 70 of them) matches commit `8d12332228eb64bc234e27c6f7e0913a926285ab` of `DQLin/ReSTIR_PT`.

I found **no wrong sign, exponent, or measure in the core GRIS equations**. That covers Eqs. 1–67, the Jacobians of Eqs. 52–54, and the pairwise and Talbot MIS formulas. The problems are elsewhere:

- **One issue that would silently bias this project's renderer (MAJOR).** The report's NEE MIS formulas, both in the shift table and in the pseudo-code, omit the **delta / analytic-light exception**. The user needs point and spot lights. Implemented as written, those lights would lose energy every time an NEE path is shifted. See C1.
- **One recommendation that is wrong under a common design choice.** The report says to store the emitter-as-reconnection-vertex light PDF in area measure. That is not sufficient if light selection depends on the shading point, e.g. a light tree. See C2.
- **One latent reference-code bug** the report describes but does not flag: the Jacobian's BSDF-PDF ratio for a BSDF-sampled final vertex is applied only inside the MIS branch. See C3.
- **Minor inaccuracies:** C4 to C9.
- **Two open questions** in the report can be closed from source: Q2 and Q3 in §4.
- **Validation omissions that would make the Cycles reference itself biased:** Cycles 5.1.2 defaults for clamp, glossy blur, light threshold, per-type bounce caps and the denoiser. See O1.

Confidence labels: **[HIGH]** = verified directly in the PDF page image or source line. **[MED]** = derived or inferred with good support. **[LOW]** = plausible but not fully verified.

---

## 1. What was checked, and how

| Source | Method |
|---|---|
| GRIS paper, pages 1–23 | Read page PNGs p-01 … p-23 visually, in particular pp. 2–12, 13–18, 20–23. |
| GRIS supplement | Fetched by WebFetch; the harness cached the PDF at `~/.claude/projects/.../tool-results/webfetch-1790606535310-2v74cz.pdf`. Text extracted with `gs -sDEVICE=txtwrite -o -` to stdout only. Glyphs are garbled; the prose of Algorithm 1, S.82 and S.106 is legible. |
| Reference code | `raw.githubusercontent.com/DQLin/ReSTIR_PT/8d12332228eb64bc234e27c6f7e0913a926285ab/...` fetched to stdout. Files read: `SpatialReuse.cs.slang`, `TemporalReuse.cs.slang`, `Shift.slang`, `PathReservoir.slang`, `PathBuilder.slang`, `PathTracer.slang` (lines 1–440, 940–1440, 1640–1990), `PathState.slang`, `Params.slang`, `ReSTIRPTPass.h`, `ReSTIRPTPass.cpp` (670–830), `StaticParams.slang`, `TracePass.cs.slang`, Falcor `Rendering/Materials/BxDF.slang`, `ScreenSpaceReSTIRPass/FinalShading.cs.slang`. |
| Blender 5.1.2 | Read-only RNA introspection via `--background --factory-startup --python-expr`. Nothing was saved. |

---

## 2. Confirmed items (no action needed)

### 2.1 Paper equations [HIGH]

- **Eqs. 1–3** (p.75:4). Eq. 3 in the paper is printed with a typo, `∫_Ω f(x) dy`, which is harmless.
- **Eqs. 4–6** (p.75:5).
- **Eq. 7.**
- **Footnote 3:** `p_Y(y) = p_X(x)|∂T/∂x|^{-1}`.
- **Eqs. 8–9**, and Thm A.1 (Eqs. 55–56).
- **Eqs. 10–12.**
- **Def. 4.2.**
- **Eqs. 13–18.**
- **The boxed statement** on p.75:7.
- **"Relaxing constraints."**
- **Eqs. 19–25**, including footnote 6 on the missing `1/M`.
- **Eqs. 26–34.**
- **Def. 5.2 and the boxed** `R` / `|R|` definition.
- **Eqs. 35–38.**
- **Eqs. 39–43**, including footnote 9.
- **Eqs. 44–47**, including footnote 11: `b_k = (M_c/(M_c+1))^k`.
- **Eqs. 48–54.**
- **Thms A.2–A.4** (Eqs. 57–67).
- **App. B.**
- **The MAPE definition, footnote 13** (p.75:18): `MAPE(I, I_gt) = mean(|I − Ĩ_gt| / (0.01·mean(Ĩ_gt) + Ĩ_gt))`, with `Ĩ_gt` the grayscale ground truth. The tilde placement is as the report gives it; it is legible at 130 dpi.
- **Eq. 52 subscript.** It is printed as `θ_2^•`, defined as "the angle between `ω_i^•` and the geometric surface normal at `x_{i+1} = y_{i+1}`". This closes the report's open question 1.
- **§8.3 parameters:**
  - offline: 32 path trees, 3 spatial passes, 6 neighbours, 10 px radius, low-discrepancy sequence;
  - real-time: 1 tree, 1 pass, 3 random neighbours, 20 px radius, `M_c = 20`;
  - per-lobe GGX roughness threshold 0.2;
  - distance threshold 1–5% of scene size;
  - defensive pairwise MIS for spatial reuse, Talbot for temporal reuse, `|R| = 1`.
- **All numbers in Figs. 1, 6, 7, 12, 14, 15, 16 and 17–18.** Fig. 15 is legible at 130 dpi, so its `[UNVERIFIED]` tag can be dropped.

### 2.2 Independent re-derivations [HIGH]

**(a) Eqs. 37 and 38 follow from the S.13/S.14 family.** Set `α = M/|R| − 1`.

- `x = 1` gives Eq. 37 exactly.
- `x = (M−|R|)/M` gives Eq. 38 exactly. The canonical term reduces to `(1/|R|)(|R|/M) = 1/M`, and the pair term to `(1/M)·|R|p̂/(|R|p̂ + (M−|R|)p̂_{←j})·(1/|R|)`.

**(b) Partition of unity at a fixed `y`.**

- Eq. 37: `Σ = Σ_{j∉R} (|R|p̂/(M−|R|) + p̂_{←j})/D_j = 1`, with `D_j = |R|p̂ + (M−|R|)p̂_{←j}`.
- Eq. 38: `Σ = |R|/M + (1/M) Σ_j (|R|p̂ + (M−|R|)p̂_{←j})/D_j = |R|/M + (M−|R|)/M = 1`.

**(c) Confidence-weighted pairwise MIS as coded** (`SpatialReuse.cs.slang:343-409`). Substitute `p̂ → M_c p̂_c` and `p̂_{←i} → M_i p̂_{←i}` into Eq. 38 with `|R| = 1` and `M = k+1`:

```
m_c(y) = (1/(V+1)) [ 1 + Σ_{valid i} c/(n_i + c) ]
m_i(y) = (1/(V+1)) · n_i/(n_i + c)
c   = M_c · p̂_c(y) / k           // k = nominal neighborCount (:379, :396)
n_i = M_i · p̂_{←i}(y)
V   = validNeighborCount          // (:408)
```

This sums to 1 for any positive constant in place of `k`. The report's claim is correct.

**(d) Composite PSS Jacobian for reconnection at code index `k`.** The map `(ω_{k−1}^x, ω_k^x) → (ω_{k−1}^y, ω_k^y)` is block-triangular: `ω_k^y = ω_k^x` because `x_{k+1}` is copied. So the solid-angle determinant is only the geometric factor of Eq. 52. Converting each dimension to PSS with Eq. 54 gives

```
J = [ p^y_ω(y_{k−1}→x_k) / p^x_ω(x_{k−1}→x_k) ]
  · [ (|cos θ^y_{x_k}| / ‖x_k − y_{k−1}‖²) / (|cos θ^x_{x_k}| / ‖x_k − x_{k−1}‖²) ]
  · [ p^y_ω(x_k→x_{k+1}) / p^x_ω(x_k→x_{k+1}) ]     // only if x_{k+1} was BSDF-sampled at x_k
```

- The `p_ω` terms are the single-lobe (lobe-class) PDFs, including the lobe-selection probability. `pdfSingle *= pLobe` at `BxDF.slang:957-1003`.
- In area measure the reconnection is the identity on `x_k, x_{k+1}, …`. The PSS factor above is therefore exactly `p_area(ȳ)/p_area(x̄)`, which is consistent with the Eq. 54 conversion.

This matches the report's §6.6 and `Shift.slang:450-542`.

**(e) PSS and path-space GRIS give identical estimates** (report §5.3). The ratio `p̂_{←i}/p̂_c` in PSS is `lum f_i(x)/(J_path · lum f_c(y))`, the same as in path space, because the common `1/p_c(y)` cancels. Resampling weights and `f(Y)W_Y` also agree.

**(f) Temporal Talbot with confidence weights** (report §7.4) matches `TemporalReuse.cs.slang:146-225`.
- The M-cap `min(20·M_cur, M_prev)` is at `:139`.
- The reverse shift uses the previous V-buffer and the previous camera position (`usePrev`, `:206-207`), but the **current** scene.

**(g) Path tree** (report §5.4).
- `add()`: `w = lum(F)/rrPdf`, `M += 1` (`PathReservoir.slang:290-310`).
- `PathBuilder.finalize` sets `M = 1` (`:45-55`).
- `finalizeRIS` computes `W = Σw/(p̂·M)`.
- `mergeInSamplePixel` uses `w = tree w_sum` (`:392-425`), then `finalizeRIS` divides by `S`.

This is two-level RIS: disjoint candidates with `m = 1` inside a tree, and `m = 1/S` across i.i.d. trees. It is unbiased.

**(h) Excluded subpaths.**
- `computeEmissive = false` when `length == 1 && !transmission && !delta` (`PathTracer.slang:1025`).
- NEE is skipped at the primary hit when DI is disabled (`:1266`).
- `addEscapeVertex`/`addNeeVertex` require `pathLength ≥ 1` (`PathBuilder.slang:63, :123`).
- `LDeltaDirect` is set at `PathTracer.slang:1074` and added at `:1867`.

---

## 3. Corrections

### C1 [MAJOR for this project, HIGH confidence]: NEE and BSDF MIS must be skipped for delta (analytic) lights

**What the report says.** It gives the shifted NEE-final integrand with `ω_NEE^dst = lightPdf/(lightPdf + p_{BSDF,x_k}^{dst,all}(rcWi))` (§6.8 table, row 2). The Pass A and `reconnect` pseudo-code (report lines 943 and 968) use `mis = ls.pdf/(ls.pdf + pdfBsdfAll)` unconditionally.

**What the code does.** It applies the balance heuristic **only if the light type is not `Analytic`**:

- `Shift.slang:525-535`: `if (isRcVertexFinal && kUseMIS) { if (lightType != Analytic) { misWeight = evalMIS(...) ... } }`.
- `PathTracer.slang:1301`: `if (kUseMIS && ls.lightType != (uint)LightSampleType::Analytic)`.
- `PathTracer.slang:1322-1325`: for analytic lights `ls.pdf == 0` is replaced by the light-selection probability, so `lightPdf` is **not** a solid-angle density for these lights.

**Why it matters.** Point and spot lights are Dirac deltas, so BSDF sampling can never hit them and their MIS weight must be exactly 1.
- Following the report's formula, with the selection probability as `lightPdf`, gives `ω < 1` on every shifted NEE path to a point or spot light.
- The partition of unity over techniques is lost, which produces a systematic darkening bias.
- It shows up exactly in the reconnection case, so it would look like a "ReSTIR-only" bias against Cycles.

**Corrected formula.** Let `L` be the light the NEE sample hit, `p_sel` its selection probability, and `p_ω^L` its solid-angle density at the shading point (zero or undefined for delta lights).

```
ω_NEE(ȳ) = 1                                        if L is a delta light (point, spot, directional)
         = p_ω^L / (p_ω^L + p_BSDF,all(x_k; wo = −ω_{k−1}^y, wi = rcWi))   otherwise

ω_BSDF(emitter hit) = p_BSDF,all / (p_BSDF,all + p_ω^L)    // only for intersectable emitters
```

Equivalently, store a per-sample flag "light is delta". Do not rely on `lightPdf == 0`.

The same exception applies to the "rc vertex is final, BSDF hit" and "emitter is rc vertex" rows. Those rows only arise for intersectable emitters, so they are unaffected when point and spot lights are non-intersectable.

**Note for the project [INFERENCE].** In Falcor, analytic *area* lights (rect, disc, sphere) are also `Analytic`: NEE only, never hit by BSDF rays, MIS weight 1. Cycles area lights **are** hit by BSDF rays and use MIS (`Light.use_multiple_importance` / `cycles` light setting, *[LOW]*, not introspected). To match Cycles, implement area lights as intersectable emitters with correct `p_ω^L` in both techniques. Alternatively make them NEE-only in both renderers.

### C2 [MODERATE, MED–HIGH confidence]: the fix for the emitter-as-rc light PDF is incomplete

**What the report says.** Its `[UNVERIFIED]` caveat (§6.8, around line 784) correctly states the problem: the stored `lightPdf` is the solid-angle NEE PDF seen from the **base** previous vertex `x_{k−1}`. I **verified** this:
- `PathTracer.slang:1056` computes `lightPdf = selProb · emissiveSampler.evalPdf(path.origin, path.normal, upperHemisphere, hit)`.
- `PathBuilder.slang:191` stores it (`markEscapeVertexAsRcVertex`).
- `Shift.slang:520` reuses it for the offset path.

So the reference code is biased in this case. The `[UNVERIFIED]` tag can be dropped: **CONFIRMED**.

**What is wrong with the recommendation.** The report recommends storing `p_A` "including the light-selection probability" and converting it with the offset geometry. That is only correct when the light-selection probability does not depend on the shading point, e.g. uniform or power-proportional selection. The reference default is `EmissiveLightSamplerType::Power` (`ReSTIRPTPass.h:101`), for which it works. It is wrong for:
- a light BVH or light tree, since Falcor's `evalPdf` takes `origin` and `normal`;
- Cycles-like light trees (`scene.cycles.use_light_tree = True` by default in 5.1.2);
- hemisphere-restricted sampling.

**Corrected rule.** Evaluate the offset's light PDF **at the offset's previous vertex**:

```
p_ω^L(y_{k−1} → x_k) = P_sel(L | y_{k−1}, n_{y_{k−1}}) · p_A(x_k | L) · ‖x_k − y_{k−1}‖² / |cos θ_L(x_k)|
```

Store `p_A` alone only if `P_sel` is position-independent. Otherwise call the same `evalLightPdf(y_{k−1}, n, lightHit)` routine that NEE uses at `y_{k−1}`.

### C3 [MODERATE for a port, HIGH confidence]: the BSDF-PDF ratio at a BSDF-sampled final rc vertex is conditional on MIS in the code

**What the code does.** For `isRcVertexFinal && !lastVertexNEE`, the factor `J *= dstRcVertexScatterPdf / srcRcVertexScatterPdf` is applied **only inside** `if (isRcVertexFinal && kUseMIS) { if (lightType != Analytic) {...} }` (`Shift.slang:525-535`). The unconditional branch at `:539` excludes final vertices.

**Why it matters.** With MIS disabled, e.g. a "BSDF-only vs Cycles" debugging run, the Jacobian lacks the PSS factor `p^y(x_k→x_{k+1})/p^x(x_k→x_{k+1})`, which introduces bias. The report's pseudo-code applies it correctly and unconditionally (report line 969), but its prose attributes it to the MIS branch without flagging this.

**Correct rule.** Apply the ratio whenever `x_{k+1}` was **BSDF-sampled** at `x_k`, whether or not MIS is on. Skip it only for NEE-sampled `x_{k+1}` (the light sample is independent of the incoming direction) and when `x_k` is itself the emitter.

### C4 [MINOR, HIGH]: `traceTemporalUpdate` does not overwrite `rcVertexWi` in the non-NEE branch

The report (§6.9) says it "overwrites `rcVertexIrradiance`, `rcVertexWi`, `lightPdf` and the light type". In fact:

- **NEE branch** (`PathTracer.slang:1737-1777`): the light is re-sampled with `rcRandomSeed`, and `rcVertexWi = ls.dir` is set.
- **Non-NEE branch** (`:1779-1839`): it **copies the stored direction** at `x_k` (`scatterDir = rcVertexWi`), random-replays the rest of the suffix with `rcRandomSeed`, and overwrites only `rcVertexIrradiance`, `lightPdf` and the light type (`:1837-1839`). The direction at `x_k` is kept. This is direction copy, with PSS Jacobian `p^y/p^x`, which is already in `J`.

Two further facts:
- In the hybrid shift the update is skipped when the rc vertex is the escaped emitter (`Shift.slang:180`, `!isRcVertexEscapedVertex`). A moving or emissive-changing emitter used as rc vertex keeps stale `Le`.
- Even with the update enabled, the Talbot reverse-shift term uses the **current** scene (`TemporalReuse.cs.slang:206`). The forward term uses the previous frame's `lum(F)` (`:192`), evaluated in the **previous** scene. Temporal MIS therefore stays inexact under motion, which confirms the report's inference.

### C5 [MINOR, HIGH]: Thm A.2 item 4 is overstated

The report writes "Pointwise convergence except on a null set." The paper (p.75:23) says: "(4) *in the set in which* `p_Y(y)` *converges*, it converges to `p̄(y)` (except for a possible set of zero measure)." It does **not** claim that `p_Y` converges pointwise.

### C6 [MINOR, MED]: S.82 is a "generalization", not a "weaker" condition

The report says the proof "actually uses the weaker `Var[p̂(Y_M)W_{Y_M}] → 0` (S.82)". The supplement calls S.82 "a generalization of Equation 58". It applies to arbitrary non-negative unbiased contribution weights `W_{Y_M}`. For GRIS with Eq. 22, `p̂(Y)W_Y = Σ_i w_i` identically, so the two conditions are **the same**, not weaker.

### C7 [MINOR, HIGH]: checklist item (d) is not a requirement

The report's §2.4 item (d), "`m_i > 0` wherever `y` is reachable from `i`", is not needed. The paper only requires:
- `m_i ≥ 0`;
- Eq. 20, the sum over the domains that can generate `y` equals 1;
- `m_i > 0` wherever `c_i ≠ 0`, which is automatic for `c_i = m_i`.

Pairwise and Talbot weights can legitimately be 0 at reachable `y`, e.g. when `p̂_{←i}(y) = 0`. **Correct item (d):** "`m_i > 0` wherever `c_i(y) ≠ 0`; with `c_i = m_i` this is automatic."

### C8 [MINOR, HIGH]: pseudo-code condition for dropping BSDF emitter hits at `x_2`

The report's line 937 writes `not (len==1 and diffuse-reflection-from-x1)`. The code (`PathTracer.slang:1025`) is `path.length == 1 && !path.isTransmission() && !path.isDelta()`. That drops **any non-delta, non-transmission** bounce from `x_1`, glossy reflection included.

The Pass A pseudo-code also omits the non-hybrid assignment `rcLen = 1` (`PathTracer.slang:1171`, `!useHybridShift && path.length == 1`).

### C9 [MINOR, HIGH]: `linearRoughness` is verified to be perceptual roughness

`BxDF.slang:840` has `float alpha = sd.linearRoughness * sd.linearRoughness;`, so GGX `α = linearRoughness²`. Also, `α < kMinGGXAlpha` becomes 0 (a delta lobe) at `:843-845`.

So the 0.2 threshold is on **perceptual** roughness (α = 0.04). That maps 1:1 to glTF `roughnessFactor` and to Blender Principled "Roughness", both of which are perceptual with α = r². Drop the `[UNVERIFIED]` in the report's §6.5 and open question 2.

Related minor point: the reference code's "lobe" is really a **lobe class**. `getAllowedBSDFFlags` returns 0x3 (diffuse reflection + transmission) or 0xC (specular reflection + transmission), and `pdfSingle` in `evalPdf` sums over the class (`BxDF.slang:1025-1050`). The paper's lobe-extended space (Eq. 50) is per lobe. For opaque Lambert + GGX these coincide.

---

## 4. Open questions in the report that can now be closed

- **Q1** (Eq. 52 subscript, MAPE tildes): as printed; see §2.1.
- **Q2** (`linearRoughness` = perceptual?): **yes**; see C9.
- **Q3** (who adds primary-hit emission?): ReSTIR DI's resolve does. `ScreenSpaceReSTIRPass/FinalShading.cs.slang:78-80` has `emission = sd.emissive; color += diffuse + specular + emission;`.
  - The ReSTIR DI resolve evaluates only the `DiffuseReflection` and `SpecularReflection` lobes (`:69-73`). This is why transmission and delta direct light go through `LDeltaDirect`.
  - The reference renders **nothing** for primary misses. The environment background is commented out (`FinalShading.cs.slang:94-99`), and `TracePass.cs.slang:52-53` returns early on an invalid V-buffer hit.
  - For Cycles comparisons, the WGSL renderer must add camera-visible environment and emission itself.

---

## 5. Important omissions

1. **O1 (validation, HIGH). Cycles 5.1.2 defaults that bias the ground truth.** The report's §10 mentions only the pixel filter. Values from headless RNA introspection of `bpy.context.scene.cycles`:
   - `sample_clamp_indirect = 10.0`, "higher values will be scaled down … at the cost of accuracy". This is **biased**; set it to 0. `sample_clamp_direct = 0.0` is already fine.
   - `blur_glossy = 1.0` (Filter Glossy), which is biased; set it to 0.
   - `light_sampling_threshold = 0.01`. This skips low-contribution light samples, which is biased; set it to 0.
   - `use_denoising = True`; set it to False.
   - `diffuse_bounces = 4`, `glossy_bounces = 4`, `transmission_bounces = 12`, `max_bounces = 12`, `transparent_max_bounces = 8`. Per-type caps below the total change which paths exist. Set every per-type cap to `max_bounces`, and match the WGSL path-length limit.
   - `pixel_filter_type = 'BLACKMAN_HARRIS'`, `filter_width = 1.5`. The hard minimum is 0.01 px; use 0.01 (or `'BOX'` at 0.01) to approximate pixel-centre primary hits.
   - `use_adaptive_sampling = True`. Prefer False for references.
   - `use_light_tree = True`. This is unbiased, but relevant to C2 if mirrored.
   - `caustics_reflective = caustics_refractive = True`. Keep them on.
   - **Bounce-count mapping** [LOW, not verified in Cycles source]: in Cycles `max_bounces = 0` means direct lighting only. Verify the mapping to the WGSL "max path length" with a Cornell box sweep.
2. **O2 (HIGH). Russian roulette is not handled by the shifts.** `PathTracer.slang:1377` has the comment `// TODO: check if this is handled correctly in ReSTIR PT`. During replay the RR random number is consumed but no survival probability enters `J` (`:1378-1383`). The candidate weight divides by `rrPdf` (`PathReservoir.add`), but a shifted path's survival probability is never re-evaluated. **Recommendation:** keep RR off, which is the reference default. Otherwise treat RR as a sampling decision whose probability ratio (offset/base) enters the PSS Jacobian for every replayed vertex.
3. **O3 (MED). Uninitialized cache components.** In `computeShiftedIntegrandReconnection`, `dstCachedJacobian` is declared uninitialized (`Shift.slang:387`). Its `.y` is not written in the escaped-emitter case, and `.y`/`.z` are not written in the env branch. It is then copied into the reservoir (`:424-425`, `:568-569`). This is harmless in the reference, because those fields are unused in those cases, but in WGSL initialize the cache to 0 or 1 to avoid NaN propagation.
4. **O4 (MED). App. B on ReSTIR DI.** ReSTIR DI's spatial reuse uses an **unoccluded** target `p̂^{−V}`: "Without full coverage of `p̂^{−V}`'s domain, intermediate distributions never converge. The design still ensures coverage of `f_i`'s domain, allowing final estimators to remain unbiased." This matters because ReSTIR PT delegates direct light to ReSTIR DI.
   - App. B also warns that "conflicting motion vectors" may need careful map definitions to keep temporal shifts bijective.
   - The report covers only the visibility or canonical part of App. B.
5. **O5 (MED). Hybrid primary-vertex roughness test.** When `rcVertexLength == 1`, the non-specular branch calls `hasRoughComponent(dstPrimarySd, 1.f)` with threshold **1.0** (`Shift.slang:146`, and at `PathTracer.slang:1232`). That reduces to "has a diffuse lobe". The specular branch uses `linearRoughness > 0.2`.
   - A port that uses 0.2 in both branches changes the shift domain. That is unbiased only if it is applied identically in the forward and reverse shifts and in base-path rc selection.
   - These predicates must be **bit-identical** across base-path construction, replay and shift.
6. **O6 (LOW). Neighbour offsets can truncate to `(0,0)`.** `pixel + int2(offset·radius)` truncates (`SpatialReuse.cs.slang:138`), so about 0.3% of offsets pick the central pixel as a "neighbour". This stays unbiased, since GRIS allows duplicate or correlated candidates, but it wastes shifts. A WGSL port may reject `(0,0)`. Rejection depends only on the offset, not on the samples, so it is still unbiased.

---

## 6. Corrected formula set for the WGSL port

These are consolidated corrections to report §6.8 and §8.

```
// ω for the final vertex of an offset path (balance heuristic)
omegaNEE(y) = isDeltaLight ? 1 : pL / (pL + pBsdfAll(x_k; wo_y, wi))
omegaBSDF(y) = pBsdfAll / (pBsdfAll + pL_at_prev)     // pL_at_prev is evaluated at the offset's previous vertex (C2)

// Jacobian for reconnection at x_k from y_{k−1} (PSS)
J = (pLobe(y_{k−1}→x_k) / cache.pLobeSrcPrev)
  · (|dot(Ng(x_k), dir_y)| / d_y²) / cache.Gsrc
  · (x_{k+1} BSDF-sampled at x_k ? pLobe(x_k; wo_y→wi) / cache.pLobeSrcRc : 1)   // unconditional w.r.t. MIS (C3)
// invalid if J ≤ 0, non-finite, delta lobe before/after x_k, d_y < d_min (hybrid), or segment occluded
```

---

## 7. Items spot-checked and found correct

Keep these as written in the report.

**Paper claims:**
- Fig. 2's toy example; M-capping (§6.4).
- Canonical samples (Def. 5.2).
- Defensive pairwise MIS for spatial reuse and Talbot for temporal reuse.
- Offline mode drops temporal reuse.
- The boxed takeaways.

**Reference-code struct and bit layout:**
- Struct layout: 88 B. `lightPdf` plus the `float3 cachedJacobian` in code equals `float4 rcVertexCachedValues` in Algorithm 1.
- `pathFlags` bits 0–3, 4–7, 8/9, 10/11, 16, 18–19, 26/27.
- `kMaximumPathLength = 15`.

**Reference-code parameters and defaults:**
- Neighbour table: 8192 R2 points in a disk, RG8Snorm (`ReSTIRPTPass.cpp:202, 804-826`).
- Feature rejection: `dot ≥ 0.5`, `|Δd| < 0.1·d_c`.
- Defaults: `maxSurfaceBounces = 9`, RR off, `separatePathBSDF = true`, hybrid shift, `temporalHistoryLength = 20`, `nearFieldDistance = 0.1`, `specularRoughnessThreshold = 0.2`.
- Optional Jacobian rejection: `max(J, 1/J) > 1 + 10`, off by default.
- Forced MIS kinds for hybrid (`ReSTIRPTPass.cpp:688-694`).
- Pass order (`:676-783`).
- Pairwise needs `2k` shifts per pixel; spatial Talbot needs `k(k+1)`; temporal Talbot needs 2.

**Reference-code behaviour:**
- Hybrid invertibility logic (`PathTracer.slang:1194-1261, 1402-1425`).
- Distance test on the offset (`Shift.slang:454-458`).
- Stored irradiance with the last-vertex MIS weight divided out (`PathBuilder.slang:103-107, 168-172`).
- Direction copy for the environment map (`Shift.slang:403-432`).

**Numbers:**
- Memory: 88 B × 1920·1080 = 182.5 MB per buffer.
- WebGPU default limits: 128 MiB `maxStorageBufferBindingSize` and 256 MiB `maxBufferSize`.
- `ReconnectionData` padding: 256 B / 512 B.
