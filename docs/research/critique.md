# Completeness critique of the WebGPU ReSTIR PT research reports

> **Location note.** The orchestrator asked for this critique at
> `/private/tmp/claude-502/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-ea73-491f-b7b7-6d0aab1ffce1/scratchpad/research/critic.md`.
> This session runs in **plan mode**. The only writable file is this plan file, so the critique is here. Copy it verbatim to the intended path. Nothing else was written.
>
> **Reports reviewed.** All read in full. Paths are under `/Users/mark.boss/.claude/plans/`.
>
> | Topic | Report | Verification | Verdict |
> |---|---|---|---|
> | gris-math | `do-a-deep-dive-shimmering-ritchie-agent-a7f08dec9968f0324.md` | `…-a147516e5d2fb9558.md` | minor_issues |
> | enhanced-paper | `…-a91ab1d861819b276.md` | `…-acb243dbea3ce2a29.md` | minor_issues |
> | reference-code | `…-a3b89657b22d15f34.md` | none | – |
> | restir-practice-dynamics | `…-a668fe13fb61070e8.md` | none | – |
> | webgpu-platform | `…-a64ceeae6903be301.md` | none | – |
> | scene-io | `…-a597df4a1dd101994.md` | none | – |
> | cycles-conventions | `…-a2d6cc1a462b20a80.md` | `…-ac3ddf11ffb75ef90.md` | minor_issues |
> | validation-harness | `…-a26c6df2c3f5e4af0.md` | none | – |
>
> **Tags.** [INFERENCE] = my reasoning. [UNVERIFIED] = not checked against a primary source. Section references like "gris-math §6.8" point into the reports above. "verify Cx" means the correction numbered Cx in that report's verification.

---

## 0. Bottom line

The reports are individually strong, and three have been adversarially verified. Together they cover:
- the GRIS maths: Eqs. 1–67, MIS families, Jacobians, PSS equivalence;
- the 2022 reference code, down to line numbers;
- the Enhanced paper's contributions and its constants;
- WebGPU limits on the actual M5 Pro, plus a measured BVH2 throughput;
- the Cycles light, BSDF and integrator conventions, verified against the 5.1.2 kernel sources;
- a statistically sound validation methodology.

**They do not yet let an engineer build the renderer without making unguided, bias-critical design decisions.** Almost all of the residual risk sits at the *seams* between reports, where each report deferred to another:

1. **Lights inside ReSTIR PT.** Point, spot and area lights, plus emissive triangles, in the unified DI+GI path tree with RIS-NEE and forced NEE reconnection.
   - The per-light-type integrand, MIS, UCW, reservoir encoding and shift Jacobian are scattered across four reports and partly contradictory.
   - Cycles' "lights are transparent, non-occluding pass-through emitters" behaviour (cycles-conventions §2.8, verified) is not reconciled with any ReSTIR path representation.
2. **Temporal reuse with moving lights.** The MIS form, the required previous-frame state, and the suffix-radiance refresh are all unresolved.
   - Falcor's suffix refresh replays light sampling. Enhanced's per-frame light tiles / RIS-NEE are not reproducible under replay, and no report notices this.
3. **The BSDF / lobe model.** The reports disagree on whether Cycles' Principled BSDF can be matched. None defines the lobe-indexed path space (lobe set, selection probabilities, per-lobe roughness, joint vs. marginal pdf) for the BSDF that will actually be implemented.
4. **The exact reconnection predicate and invertibility checks** under the Enhanced footprint criteria. This is the #1 source of silent bias, and it is only available as an [INFERENCE] sketch.
5. **Feasibility.** The per-pass costs of the shift, replay and suffix-refresh passes on a software BVH are unmeasured, and the per-pass binding and memory budget has not been tallied.

Sections 2–3 give the contradictions and the ranked gaps, with self-contained research instructions.

---

## 1. Per-report completeness notes (brief)

- **gris-math.** Excellent transcription. Its verification found one MAJOR issue: delta lights need NEE MIS weight 1 (verify C1). Also the light pdf at the offset vertex (C2) and the Jacobian pdf ratio being conditional on MIS in the code (C3).
  - Its validation section (§10.2) is wrong about the pixel filter (see contradiction X1).
  - Its §10.1 assumes the 2022 DI/GI split, which the project is dropping.
- **enhanced-paper.** Thorough. Its verification found two MAJOR issues:
  - C3: an asymmetric neighbour-validity test in the paired pre-pass gives biased darkening.
  - C4: the cached Jacobian product must be updated when a shifted sample is selected.
  - The report says little about light types (§5.4 is [INFERENCE]) and nothing firm about the temporal MIS in Enhanced. The paper itself is silent on both; I grepped `restirpt_enhanced_2026.txt` and found no temporal-MIS or light-type discussion.
- **reference-code.** Very useful file/line map. Its two findings deserve promotion to design rules:
  - EvanLuo42 has a probable swapped-argument MIS bug (§2.3), so do not copy its MIS.
  - RTXDI does not re-evaluate the BSDF at the rc vertex (§2.2.3), which is biased for glossy vertices.
  - Its 64-B layout (§4.2) differs from Enhanced S-Alg.1 and from the webgpu-platform layout.
- **restir-practice-dynamics.** The best treatment of temporal bias. Its §3.4 sign analysis is original and correct [INFERENCE-checked algebra].
  - It assumes static geometry. That is fine for this project *only if* area lights are analytic and non-occluding (X2).
  - It does not notice that suffix re-trace is incompatible with RIS-NEE light tiles (Gap 2).
- **webgpu-platform.** Measured limits and throughput are the most reliable platform facts; they supersede the [UNVERIFIED] limit claims elsewhere (X10).
  - It proposes area lights as emissive BLAS geometry, which contradicts Cycles (X2).
  - It says S-Alg.1 was "not read".
  - Its pass-time budget is [INFERENCE] extrapolated from plain PT.
- **scene-io.** Solid unit contract (Blender-equivalent units, glTF lights ÷683, emission ×1) and a solid light model.
  - It contradicts validation-harness on how the Blender reference is built (X4).
  - It maps a USD SphereLight to a true sphere, which contradicts cycles verify C6 (X12).
  - It recommends `rgba8unorm-srgb`, which contradicts Cycles' decode-after-filter behaviour (X6).
  - Its USD choice (LightUSD rc4) still needs an API spike (G6 in §4).
- **cycles-conventions.** Accurate. Verified corrections:
  - C1: camera `angle_x` is not the render's horizontal FOV;
  - C6: the unnormalised USD sun, and a USD SphereLight becomes a soft-falloff disk;
  - C7: set `sampling_pattern` explicitly.
  - Its §2.8 (pass-through lights) is the most consequential fact for ReSTIR PT, and no ReSTIR report has absorbed it.
- **validation-harness.** Methodologically excellent: TOST, ensembles, planted-bias controls, gate ladder.
  - Test T14 hard-codes the likely-typo form of Eq. 3 (X9).
  - It restricts materials more than cycles-conventions says is necessary (X5).
  - Its "scene bridge" conflicts with scene-io's "import-the-same-file" (X4).

---

## 2. Contradictions between reports

Each item says who says what, which side is better supported, and what I recommend.

**X1. Pixel filter / primary-ray jitter.**
- gris-math §10.2 and gris-verify O1 recommend rendering Cycles with a tiny filter ("use 0.01 (or 'BOX' at 0.01)") to mimic the reference code's pixel-centre V-buffer.
- cycles-conventions §5.4, confirmed by cycles-verify: **for BOX, Blender forces `filter_width = 1.0`** (`blender/sync.cpp:587-589`). "BOX at 0.01" is impossible.
- scene-io §8.4 and validation-harness §7.1/T13 use BOX 1.0 with uniform in-pixel jitter.
- **Resolution:** the WGSL renderer must jitter primary rays uniformly per frame (fresh V-buffer per frame) and compare against Cycles BOX. [INFERENCE] ReSTIR stays unbiased per frame, because each frame's pixel domain Ω_i is defined by that frame's jittered x1.
- **Consequence:** the "temporal shift is the identity in a static scene" unit test (restir-practice §3.11 test 1) holds only with jitter off.

**X2. How area lights are represented.**
- webgpu-platform §4.5: area lights "should be real emissive geometry (triangles in a BLAS …)", moving via the TLAS.
- cycles-conventions §2.8 (verified): Blender area, point and spot lights are **one-sided, non-occluding and invisible to shadow rays**. A camera or BSDF ray that hits one adds its MIS-weighted emission and **continues past it**, costing a *transparent* bounce (`transparent_max_bounces` = 8). They are camera-invisible by default.
- scene-io §2.3/§3.7 says the same: a rect light is not an emissive quad.
- restir-practice §3.5 keeps lights analytic, outside the BVH, and flags hit semantics as [UNVERIFIED].
- gris-verify C1 note: "implement area lights as intersectable emitters with MIS … or NEE-only in both".
- **Resolution:** analytic lights live in a separate light buffer and never occlude; emissive triangles are opaque and two-sided. The ReSTIR PT consequences are open (Gap 1).

**X3. Whether a previous-frame BVH/TLAS is needed.**
- webgpu-platform §4.5/§5.2 and enhanced-paper §5.4/§11: keep the previous TLAS for temporal inverse shifts.
- restir-practice §0.3/§3.3: static geometry needs no previous BVH, only the previous camera, V-buffer and light buffer.
- **Resolution:** this follows from X2. With static triangles and non-occluding analytic lights, the previous BVH equals the current one. A previous TLAS is only needed once geometry moves (phase 2).

**X4. How the Blender reference scene is built.**
- scene-io §0.6/§8.5 re-imports the exact asset file with Blender's stock importer and adds the JSON lights, camera and world.
- validation-harness §0.6/§7.5 builds the Blender scene from *our* parsed scene (a "scene bridge"), never through Blender's importers, and tests importer fidelity separately.
- **Resolution [INFERENCE]:** use the bridge for Stage A (renderer correctness), because it removes importer differences such as MULTI_GGX defaults, TANGENT handling and USD soft-falloff. Keep the import path only as a loader-fidelity test.

**X5. Which material model can match Cycles.**
- validation-harness §0.7/§7.4: "Principled v2 layering cannot be matched"; restrict to Diffuse + Glossy(GGX) + Mix + Emission.
- cycles-conventions §4.4–4.7 (verified, LUT test vectors re-computed): Principled with `distribution='GGX'` (Tier 1) **is exactly matchable** with a port of the 4096-float `table_ggx_gen_schlick_ior_s`. MULTI_GGX (Tier 2) needs `ggx_E`/`ggx_Eavg`.
- scene-io §3.3: "either".
- The Blender glTF importer leaves Principled at **MULTI_GGX** (cycles §4.8).
- **Resolution:** cycles-conventions is better supported. Material-model scope is part of Gap 3.

**X6. Order of sRGB decode and texture filtering.**
- scene-io §3.2/§7.4, webgpu-platform §3.2 and validation-harness §7.4 recommend `rgba8unorm-srgb` (hardware decode before filtering), claiming this matches Cycles.
- cycles-conventions §4.10 (verified, `kernel/svm/image.h:36-38`): Cycles **interpolates 8-bit sRGB bytes first and decodes afterwards**. To match exactly, use `rgba8unorm` and decode in the shader after bilinear filtering.
- **Resolution:** cycles-conventions is right. The bias is small but systematic at texture edges.

**X7. Temporal MIS form.**
- gris-math §7.4 and reference-code §1.7.3 transcribe Falcor's generalized Talbot. It uses the *stored* previous-frame p̂ (`lum(F)`) for the temporal sample, and a *recomputed* p̂_prev (previous camera, **current** scene) for the canonical sample.
- restir-practice §3.4 [INFERENCE, algebra checked by me] shows that this mix breaks the partition of unity whenever the recomputation is inexact, e.g. with moving lights. It recommends either contribution-MIS (RTXDI π/π_sum) or recomputing both terms consistently.
- enhanced-paper: the temporal MIS in Enhanced is "not restated".
- No report makes a final choice (Gap 2).

**X8. Confidence-cap convention and how M is stored.**

| Source | Cap rule |
|---|---|
| GRIS §6.2 | Cap the **output**: M_rm = min(M_c, ΣM) |
| Falcor | Cap the incoming history relative to the current M: min(20·M_cur, M_prev); spatial sums uncapped |
| Enhanced §2.2 | min(c_Cap, c_temp) + 1 |
| RTXDI | min(maxM, …) |

Storage of M also differs: Enhanced packs M as an **8-bit integer** in `pathFlags`, Falcor uses a fractional f32, and webgpu-platform and reference-code propose f16.
- All caps are unbiased if they are sample-independent.
- The fractional duplication-map cap cannot be stored exactly in 8 bits; RTXDI truncates.
- **Pick one convention.** Keep M as f32 or f16 in validation builds.

**X9. Paired-reuse n_σ formula.**
- validation-harness T14 and EvanLuo42 use Eq. 3 as printed (+1.76σ⁻²).
- enhanced-verify C1 simulated the shuffle and showed the printed form is almost certainly a sign typo: **−1.76σ⁻²** reproduces n = 1…12 exactly.
- enhanced-paper's fallback `round(σ²/2)` is wrong for σ < 1.3.
- Harmless at σ = 16 (n = 128), but T14 must use the corrected form or a measured table.

**X10. WebGPU limits.**
- restir-practice §5.2: "request a higher adapter limit" for more than 8 storage buffers.
- reference-code §3.2: "Apple adapters expose much more [UNVERIFIED]".
- scene-io §7.1: ~4 GiB "[UNVERIFIED for M5 Pro]".
- webgpu-platform §2.1 **measured** the M5 Pro adapter:

| Limit | Adapter maximum |
|---|---|
| `maxStorageBuffersPerShaderStage` | **10 (hard)** |
| `maxStorageBufferBindingSize` | 4 GiB − 4 |
| `maxComputeWorkgroupStorageSize` | 32 KiB |
| `maxSampledTexturesPerShaderStage` | 48 |
| `maxStorageTexturesPerShaderStage` | 8 |
| `maxBindGroups` | 4 |

- The measured values supersede the others. EvanLuo42's ~28 KB duplication-map LDS fits only after raising the workgroup-storage limit to 32 KiB.
- enhanced-verify C10 says r8unorm/r16float storage textures need `texture-formats-tier1`. webgpu-platform measured that tier1 and tier2 are available on this adapter.

**X11. Reservoir layout (all 64 B, all different).**
- Enhanced S-Alg.1 (enhanced-paper §6.2):
  - float2 cached values: the *product* of the Jacobian factors, plus the NEE light pdf;
  - 8-bit M;
  - f32×3 F and f32×3 radiance;
  - `rcVertexRandomSeed` kept.
- webgpu-platform §5.5 (it says S-Alg.1 was not read):
  - three f32 cached Jacobian terms;
  - F and L in f16, with an overflow-to-inf risk it measured;
  - f16 M.
- reference-code §4.2: cached terms as f16×3, RGB9E5 radiance, and the proposal to drop `rcRandomSeed` with a counter-based RNG.
- Unresolved:
  - whether one product float suffices, since J needs numerator and denominator over the *same* factor set; for NEE-final and escaped cases p_k := 1;
  - whether lossy F/L encodings are acceptable in validation. validation-harness T16 says no.

**X12. USD SphereLight.**
- scene-io §5.4 maps a UsdLux SphereLight with r > 0 to a true sphere light.
- cycles-verify C6: Blender imports it as a POINT light with `use_soft_falloff = True` (the default), i.e. an **oriented disk**, not a sphere.
- For parity, either force `use_soft_falloff=False` in Blender or implement the disk. The JSON-lights contract sidesteps this for validation.

**X13. Reconnection roughness rule.** It differs by source, and the choice must be bit-identical between base path and replay.

| Source | Rule |
|---|---|
| GRIS / Falcor | Two-vertex min(α_{k−1}, α_k) ≥ 0.2 (perceptual), plus absolute distance 0.1 |
| Falcor, k = 1 | `hasRoughComponent(sd, 1.0)`, i.e. "has a diffuse lobe" (gris-verify O5) |
| Enhanced | Single vertex α_{x_{k−1}} ≥ 0.2, plus dual footprint c/100 = 2e-4 |
| RTXDI | pdf-proxy `pdf ≤ 1/0.1²`; footprint c² = 4e-4 with Gaussian jitter; no roughness test for emitter hits |
| EvanLuo42 | α = 1 for diffuse, 0 for delta |

**X14. Light-vertex Jacobian convention.**
- enhanced-paper §5.4, restir-practice §3.5 and EvanLuo42: J = 1 for an area-measure-sampled or delta NEE light vertex, with position-independent selection.
- RTXDI stores the light's solid-angle pdf as a `partialJacobian` and uses J = pdf_src/pdf_dst (reference-code §2.2.5).
- Both are consistent only under matching target and integrand conventions. restir-practice §7.2 leaves this [UNVERIFIED]. Folded into Gap 1.

**X15. DI/GI split.**
- gris-math §5.4/§10.1 and reference-code §1.9 (2022): NEE at x1 and emitter hits at x2 are excluded and handled by ReSTIR DI + `LDeltaDirect`.
- Enhanced §6.1: unified.
- Every report that makes a recommendation picks unified, so this is resolved, but gris-math's validation plan must be updated.
- Still unstated: whether length-1 paths (camera sees an emitter) are added outside the reservoir. See Gap 1.

**X16. Pairwise MIS normalisation variants.** All are valid partitions of unity except EvanLuo's, but they are different estimators:
- Falcor: c_c/N inside, 1/(valid+1) outside (enhanced-verify C2);
- enhanced-paper transcription: c_c/k;
- course notes, Eq. 7.8, cited in restir-practice §1.2: c_i/c_tot weighting;
- EvanLuo42: swapped arguments, not a valid partition.

Choose one that can be computed from the shared paired-slot data (`F·J`, `J`, plus the partner's `c` and `p̂`).

---

## 3. The five most important gaps (ranked)

### Gap 1 (highest): Lights inside ReSTIR PT (key `light-integration-restir-pt`)

**Why it matters.**
- The user explicitly requires point, spot and area lights. The GLB/USD loaders will also produce emissive triangles.
- No report gives one consistent, exact set of formulas for each light type inside the Enhanced unified DI+GI path tree with RIS-NEE and forced NEE reconnection:
  - the PSS integrand, the NEE and BSDF MIS weights, and the RIS-NEE UCW for a candidate list mixing delta and area measures;
  - the reservoir encoding;
  - the rc-case table: rc = NEE light vertex, rc = x_{d−1} with an NEE or emitter-hit suffix, rc = emitter hit;
  - the Jacobian and failure rules.
- A bias of exactly this kind has already been found once (gris-verify C1: delta-light MIS). Two more are known: the offset-vertex light pdf (C2) and the X14 Jacobian convention.
- Cycles' pass-through, non-occluding analytic lights (X2) have no ReSTIR PT path representation yet. It is unknown whether a BSDF path continues past a light, how path length is counted, whether a pass-through light can be the rc vertex, and whether visibility rays ignore lights.

### Gap 2: Exact temporal reuse under moving lights and a moving camera (key `temporal-moving-lights`)

**Why it matters.**
- Moving lights are an explicit user requirement.
- GRIS App. B requires previous-frame evaluation. The MIS form is contested (X7), the previous-frame state is contested (X3), and stale cached suffix radiance is a known bias.
- New issue found here: Falcor's `traceTemporalUpdate` replays light sampling with `rcRandomSeed`. Under Enhanced's per-frame light tiles and RIS-NEE, that replay is **not reproducible**, so the "suffix update" is no longer a deterministic shift. This is untreated in every report.
- The cost of exact mode on a software BVH is unknown.

### Gap 3: The BSDF / lobe model that matches Cycles and supports lobe-indexed hybrid shifts (key `bsdf-lobe-model`)

**Why it matters.**
- The reports contradict each other on what can match Cycles (X5, X6).
- ReSTIR PT needs a lobe-indexed path space: the joint pdf p(ω,ℓ) in J (enhanced-verify O2), marginal pdfs for MIS and footprints, per-lobe roughness for connectability, and lobe classes (gris-verify C9).
- Cycles' Principled has V-dependent layering weights and albedo-scaled sample weights (cycles-verify omission 1), and is non-reciprocal. The Enhanced "diffuse x_k skips the inverse footprint test" rule is invalid for it (enhanced-verify O8).
- glTF imports as MULTI_GGX Principled.
- Without one spec, the shift code, the rc predicate and the Cycles parity work will each assume a different BSDF.

### Gap 4: The exact reconnection predicate and invertibility checks under the Enhanced criteria (key `rc-predicate-invertibility`)

**Why it matters.**
- Non-bijective shifts are the most likely hidden bias.
- The Enhanced paper gives the criteria but not the test order or the edge cases. enhanced-paper §3.11 is an [INFERENCE] sketch. Missing:
  - the NEE-final p_k definition (enhanced-verify O4);
  - the k = 1 primary-vertex rule (gris-verify O5);
  - emitter and pass-through-light vertices;
  - the interaction with forced NEE reconnection;
  - the offset pixel's and the previous frame's R_pri;
  - threshold jitter;
  - the lobe-revocation rule.
- Paired reuse adds its own bias traps: the symmetric neighbour predicate (enhanced-verify C3) and the cached-Jacobian update on selection (C4).

### Gap 5: WebGPU feasibility of the shift, replay and refresh passes, and the binding/memory budget (key `webgpu-pass-cost-budget`)

**Why it matters.**
- Only plain PT throughput was measured: ~190–210 Mrays/s incoherent, 77 ms for 3 bounces + NEE at 1080p.
- These are unmeasured:
  - incoherent reconnection-visibility rays;
  - compacted prefix replays;
  - suffix re-traces for moving lights;
  - compile time of shift kernels (C2-Renderer's inlining blow-up);
  - the ≤ 10 storage-buffer layout per pass;
  - the total memory, with previous-frame buffers and ensemble mode.
- The platform report's 45–65 ms at 720p is extrapolated. The "real-time" goal is at risk, and so is the pass architecture (fused vs separate retrace passes).

(Structured research instructions for each gap are in the orchestrator output. They are reproduced in §5 below so this file is self-contained.)

---

## 4. Lower-priority gaps (not in the top 5)

- **G6. USD in the browser.** Spike LightUSD `1.0.0-rc4`: check that the `getLight()` / `getMeshCopy()` fields exist, that the wasm loads in Chrome 154 without COOP/COEP, and that composition and PointInstancer work. Fallbacks: three.js `USDLoader`, and offline Blender conversion. It can block the "load USD" requirement but has fallbacks.
- **G7. Environment lighting.** No decision on HDRI. There is no Blender world ↔ WGSL equirect orientation mapping, and no environment rc-vertex handling (direction copy exists in Falcor). The user did not ask for it; defer.
- **G8. Transmission / dielectrics / alpha.** No validated spec: Cycles transparent BSDF plus transparent bounce count, glass with delta lobes and replay. Defer until the core is validated.
- **G9. Cycles reference cost.** Seconds per 4096 spp at 512² on Metal are unmeasured. This drives how many K seeds × frames the dynamic ensemble tests (ix) can afford.
- **G10. Primary emission and ω for camera rays.** Camera-visible emitters and lights (`visible_camera`) must be added outside the reservoir with MIS weight 1. Partly covered by gris-verify Q3; restate it for unified mode.
- **G11. Fly-camera UX.** WASD + mouse + Q/E, pointer lock, frame-rate-independent motion. Trivial and not researched; nothing blocks it.
- **G12. RR survival formula.** Enhanced does not state it. Code22 uses q = min(1, lum(thp)); Cycles uses q = min(√max_c|β|, 1). For validation keep RR off, or use initial-only RR with an exact ∏q over the RR tests actually survived (enhanced-verify O3).

---

## 5. Research instructions for the top 5 gaps

### Gap 1: `light-integration-restir-pt`

**Read first.** All report paths are under `/Users/mark.boss/.claude/plans/`.

| Report | Sections |
|---|---|
| gris-math `…-a7f08dec9968f0324.md` | §5.2–5.4, §6.6–6.8, §8 |
| its verification `…-a147516e5d2fb9558.md` | C1–C3, §6 |
| enhanced-paper `…-a91ab1d861819b276.md` | §1.3, §5.1–5.4, §6.5 |
| its verification `…-acb243dbea3ce2a29.md` | C4, O3, O4 |
| reference-code `…-a3b89657b22d15f34.md` | §1.5, §1.6, §2.2.1, §2.2.5, §2.3 (EvanLuo "evalStoredEmissivePdf", unified DI/GI) |
| restir-practice `…-a668fe13fb61070e8.md` | §3.2, §3.5 |
| scene-io `…-a597df4a1dd101994.md` | §2, §3.7, §7.5–7.7 |
| cycles-conventions `…-a2d6cc1a462b20a80.md` | §2 (esp. §2.7–2.9) |
| its verification `…-ac3ddf11ffb75ef90.md` | C5, C10, omission 8 |
| webgpu-platform `…-a64ceeae6903be301.md` | §4.5 |

**Primary sources (read-only).**
- Enhanced paper page PNGs `scratchpad/pages/enhanced/p-10.png`, `p-11.png` (§6.1, §6.2.3).
- Supplemental §5: `https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/lin2026restirptenhanced_supplemental.pdf`.
- Cycles 5.1.2 kernel shipped in the app at `/Applications/Blender.app/Contents/Resources/5.1/scripts/addons_core/cycles/source/kernel/`: `integrator/shade_light.h`, `integrator/shade_surface.h`, `integrator/intersect_closest.h`, `light/light.h`, `light/area.h`, `light/point.h`, `light/spot.h`, `light/sample.h`, `integrator/path_state.h`.
- `github.com/EvanLuo42/ReSTIR-PT-Enhanced` (BSD-3): `Common/HybridShift.slang`, `Path/PathSampling.slang`, the light presampling pass.
- `github.com/NVIDIA-RTX/RTXDI-Library` `Include/Rtxdi/PT/HybridShift.hlsli`, `Reservoir.hlsli`. **Proprietary: describe semantics, never copy code.**

**Assume:**
- static opaque triangle geometry, which may include emissive triangles (two-sided, opaque);
- analytic point, spot, rect and disk lights in a separate buffer (non-occluding, one-sided area, camera-invisible by default), which move;
- Enhanced unified DI+GI with NEE at x1, optional RIS-NEE over light candidates, and forced NEE reconnection;
- PSS formulation; Blender units per scene-io §1.

**Deliver** exact formulas and pseudo-code, each tagged SOURCE or INFERENCE:

1. **Cycles pass-through semantics** for a BSDF or camera ray that hits an analytic light. Settle from the kernel:
   - whether emission is added with the forward MIS weight and the ray continues;
   - whether `bounce` or only `transparent_bounce` changes;
   - whether a later NEE at the next real vertex is affected;
   - whether shadow and NEE rays ignore all analytic lights.

   Then define the ReSTIR PT path representation:
   - the emitter-hit candidate is a separate path-tree candidate;
   - the continuing path's vertices are unchanged;
   - path-length accounting and the pathFlags encoding;
   - whether a pass-through light hit may be an rc vertex (recommend yes/no, with a bijectivity argument);
   - whether replay must reproduce light pass-throughs;
   - whether reconnection visibility ignores lights.

   Compare with "mode A" (analytic lights NEE-only; Blender `use_multiple_importance_sampling=False`) and recommend a default.
2. **Per light type:** point r=0, spot r=0, rect/disk area with spread 180°, two-sided textured emissive triangle, optionally sun with angle 0. Give:
   - the sampling routine;
   - the pdf in the measure used in PSS (selection pmf × area / solid-angle / discrete);
   - Le or I, including the spot smoothstep and one-sidedness (Cycles formulas);
   - the NEE MIS weight and BSDF-hit MIS weight: balance heuristic, the Enhanced ω1 = M·p1/(M·p1 + p2), and delta → 1. Show that Cycles' power heuristic changes only variance;
   - the RIS-NEE UCW W = W^RIS·p1 when candidates mix delta and area lights. Define the product measure so that p̂ and source pdf are consistent per candidate type.
3. **Reservoir encoding.** (lightId, u, v) or (instanceId, primId, bary); isDelta; lightType. What `rcVertexRadiance` holds in each case (Le with the MIS divided out, or not). Which pdfs are cached (area vs solid angle), and confirm that selection pmfs are shading-point independent under a power alias table and light tiles.
4. **The shifted integrand F(y) and PSS Jacobian** for:
   - (a) rc = NEE light vertex (forced, k = d);
   - (b) rc = x_{d−1} with an NEE suffix;
   - (c) rc = x_{d−1} with a BSDF-hit-emitter suffix;
   - (d) rc = the emitter-hit vertex;
   - (e) rc earlier with a light-terminated suffix.

   Include:
   - light-pdf re-evaluation at y_{k−1} (gris-verify C2);
   - the spot cone and area one-sidedness at the new direction;
   - failure vs. zero-contribution semantics;
   - reconciliation of J = 1 (area measure) with RTXDI's solid-angle-pdf ratio.
5. **Length-1 paths.** Camera sees an emitter or a visible light: handled outside the reservoir? Check Enhanced text and EvanLuo.
6. **Unit tests:**
   - NEE+BSDF MIS partition per type;
   - shift/Jacobian reciprocity on light-ending paths;
   - analytic scenes C0c–C0e from validation-harness §8.

### Gap 2: `temporal-moving-lights`

**Read first.**

| Report | Sections |
|---|---|
| restir-practice `/Users/mark.boss/.claude/plans/do-a-deep-dive-shimmering-ritchie-agent-a668fe13fb61070e8.md` | §0, §1.2–1.5, all of §3, §5.3–5.4 |
| gris-math `…-a7f08dec9968f0324.md` | §6.9, §7.4 |
| its verification `…-a147516e5d2fb9558.md` | C4 |
| reference-code `…-a3b89657b22d15f34.md` | §1.4, §1.7, §2.2.6–2.2.8, §2.3 |
| enhanced-paper `…-a91ab1d861819b276.md` | §4, §5.2–5.4, §7 |
| validation-harness `…-a26c6df2c3f5e4af0.md` | §2.2, §3.6, §3.8, T6(c), scenes (ix-a…d) |
| webgpu-platform `…-a64ceeae6903be301.md` | §4.1, §4.5, §8.5 |

**Primary sources (read-only).**
- `github.com/DQLin/ReSTIR_PT`: `Source/RenderPasses/ReSTIRPTPass/TemporalReuse.cs.slang`; `PathTracer.slang` `traceTemporalUpdate` (~L1730–1841); `Shift.slang` L69–72, L180–188.
- EvanLuo42 `Temporal/TemporalShift.cs.slang`, `TemporalReuse.cs.slang`, `mapWorldDirBetweenFrames`, `loadPreviousShading`.
- Bevy `crates/bevy_solari/src/realtime/restir.wesl`: previous-frame light-id translation.
- RTXDI `PT/TemporalResampling.hlsli`, `PT/HybridShift.hlsli`: semantics only.
- Wyman et al. 2023 course notes §4.3, §7.1.4, footnote 9: `https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf`.
- GRIS App. B: `scratchpad/pages/gris/p-23.png`.

**Assume:** static triangles; analytic non-occluding lights that translate, rotate, change intensity, or are added and removed; a moving camera with per-frame uniform subpixel jitter (see X1 in `/private/tmp/…/scratchpad/research/critic.md`, or in the plan file `…-a40267d04641507fa.md`).

**Deliver:**

- **(a) The temporal GRIS step.** Choose among generalized Talbot, contribution-MIS π/π_sum, and pairwise with |R| = 1.
  - Give full formulas with confidence weights and one cap convention.
  - Prove the partition of unity *given which quantities are stored vs recomputed* (restir-practice §3.4).
- **(b) Previous-frame state for the inverse shift.** List it exactly: previous camera position and previous jittered V-buffer, previous light buffer plus cur↔prev id maps, previous selection pmf / alias table, and previous light tiles (or not). Show that the current BVH suffices.
- **(c) Keeping `rcVertexRadiance` valid under light motion.** Compare:
  - (i) suffix re-trace on light-change frames;
  - (ii) light-local (id, uv) re-evaluation when the suffix is a single NEE segment;
  - (iii) ReSTIR-GI-style periodic validation;
  - (iv) accepting staleness.

  For each, state bias status and cost. **Decide explicitly whether re-tracing a suffix that contains RIS-NEE over per-frame light tiles is a valid deterministic temporal shift.** If it is not, specify a reproducible suffix NEE scheme: for example, plain power-alias NEE with fixed counter-based RNG dimensions in the suffix, or treating the suffix estimate as part of a random UCW, with justification.
- **(d) Forward shift of a light vertex** via frame-t light transforms. Jacobian: 1 for rigid motion, the area ratio when scaled.
- **(e) Added and removed lights.**
- **(f) Per-pixel cost.** Count replays, visibility rays and suffix traces per case, and convert to ms at 960×540 and 1280×720 using the webgpu-platform §4.1 Mrays/s figures.
- **(g) Tests.** Ensemble-per-frame tests for scenes (ix-a…d), T6(b)/(c), and a planted "use current-frame lights in the inverse shift" negative control with the expected bias sign.

### Gap 3: `bsdf-lobe-model`

**Read first.**

| Report | Sections |
|---|---|
| cycles-conventions `/Users/mark.boss/.claude/plans/do-a-deep-dive-shimmering-ritchie-agent-a2d6cc1a462b20a80.md` | all of §4 |
| its verification `…-ac3ddf11ffb75ef90.md` | C2, C4, omissions 1–5 |
| validation-harness `…-a26c6df2c3f5e4af0.md` | §7.4, T8 |
| scene-io `…-a597df4a1dd101994.md` | §3.2–3.3, §4.3 |
| gris-math `…-a7f08dec9968f0324.md` | §5.2, §6.5–6.6 |
| its verification `…-a147516e5d2fb9558.md` | C9 (lobe classes 0x3/0xC; linearRoughness is perceptual) |
| enhanced-paper `…-a91ab1d861819b276.md` | §3.5–3.6 |
| its verification `…-acb243dbea3ce2a29.md` | O1, O2, O8 |
| reference-code `…-a3b89657b22d15f34.md` | §1.5.2 steps 8–9, §2.3 (lobe packing) |

**Primary sources (read-only).**
- Cycles kernel in the app bundle (`/Applications/Blender.app/Contents/Resources/5.1/scripts/addons_core/cycles/source/kernel/`): `closure/bsdf_microfacet.h`, `closure/bsdf_diffuse.h`, `closure/alloc.h`, `svm/closure.h`, `integrator/surface_shader.h`, `closure/bsdf.h`.
- Falcor `Source/Falcor/Rendering/Materials/BxDF.slang` (lobe probabilities, `pdfSingle`, allowed-lobe masks, `classifyAsRough`/`hasRoughComponent`) from `github.com/DQLin/ReSTIR_PT`.
- EvanLuo42 per-vertex lobe packing and `hasExactSingleLobe`.

**Deliver** one normative WGSL BSDF spec in two variants:
- **V1 "validation":** Diffuse (Lambert) + Glossy GGX (single scatter, F ≡ 1) + constant Mix + Emission.
- **V2 "Cycles Principled Tier 1":** `distribution='GGX'`, LUT-based dielectric layering, F82 metal. Say whether Tier 2 (MULTI_GGX) is required so that GLBs imported by Blender's defaults match.

For each variant specify:
- the lobes;
- per-lobe `eval·cos` and the VNDF sampling pdf;
- lobe weights and selection probabilities as a function of V. Either reproduce Cycles' albedo-scaled `sample_weight`, or argue that any positive choice is unbiased and only changes MIS noise;
- the marginal pdf and the joint pdf p(ω, ℓ);
- which lobe index or class is stored per vertex for random replay and reconnection;
- the per-lobe roughness used by the reconnection predicate (perceptual α_min = 0.2);
- the singular / delta threshold (α_x·α_y ≤ 2e-10);
- the precise meaning of "diffuse-only x_k" for Enhanced footnote 6.

Also specify:
- geometric vs shading normals: Jacobian cosines use Ng; Cycles rejects samples with Ng·L ≤ 0 but does not in eval; non-partition MIS under smooth shading; bump_shadowing; recommend flat shading first;
- texture handling: sRGB decode **after** bilinear filtering via `rgba8unorm` (X6), LOD 0, no mips in validation mode;
- the glTF metallic-roughness → V2 mapping (Specular IOR Level = 0.5·specularFactor, IOR, B/G channels);
- a unit-test list (χ², weight consistency, furnace vs Cycles LUT test vectors).

### Gap 4: `rc-predicate-invertibility`

**Read first.**

| Report | Sections |
|---|---|
| enhanced-paper `/Users/mark.boss/.claude/plans/do-a-deep-dive-shimmering-ritchie-agent-a91ab1d861819b276.md` | §1.3, all of §3 (esp. §3.3, §3.9–3.11), §2.2–2.6, §5.3 |
| its verification `…-acb243dbea3ce2a29.md` | C3, C4, C8, O1, O4, O6, O8 |
| gris-math `…-a7f08dec9968f0324.md` | §6.3–6.8 |
| its verification `…-a147516e5d2fb9558.md` | O5, C8 |
| reference-code `…-a3b89657b22d15f34.md` | §1.4, §1.5.2 (steps 3, 5, 9), §1.6.3, §2.2.4–2.2.5, §2.3 |
| validation-harness `…-a26c6df2c3f5e4af0.md` | T2–T5 |

**Primary sources (read-only).**
- Enhanced paper page PNGs `scratchpad/pages/enhanced/p-06.png`, `p-07.png`, `p-08.png`, and supplemental §2–4.
- EvanLuo42 `Common/EnhancedPolicy.slang` (~L117–149), `Common/HybridShift.slang`, `Path/PathSampling.slang`, `Spatial/SpatialShift.cs.slang`, `Temporal/TemporalShift.cs.slang` (BSD-3).
- Falcor `PathTracer.slang` L1157–1261 and L1402–1437, `Shift.slang` L116–200 and L434–572.
- RTXDI `PT/PathReconnectibility.hlsli`, `PT/InitialSamplingPathTracerContext.hlsli` (`RecordNeeLightSample`, `RecordEmissiveLightSample`), `PT/HybridShiftPathTracerContext.hlsli` (`ValidateInvertibilityCondition`, `checkPreRcInverseGeoTerm`). Semantics only.

**Deliver:**

1. **One normative predicate,** used verbatim by base-path construction and by every replay (spatial and temporal), e.g. `rcTest(prev, cur, nextSampleInfo, domain)`. Give a case table by vertex kind: BSDF-sampled surface vertex, NEE light vertex, BSDF-hit emissive triangle, pass-through analytic light (per Gap 1), and environment. For each case state:
   - exact rayFP and invFP formulas;
   - which pdf is used (marginal vs lobe-joint), which normal (geometric vs shading), and which vertex carries the cosine;
   - delta-lobe handling;
   - the k = 1 (rc = x2) primary-vertex rule;
   - the single-vertex roughness rule;
   - the NEE-final p_k definition (enhanced-verify O4);
   - the lobe-revocation rule after sampling at a candidate rc vertex;
   - forced NEE reconnection when no earlier rc exists.
2. **Offset-path checks:**
   - no earlier replayed pair passes;
   - (y_{k−1}, x_k) passes, including invFP recomputed with p^y_k;
   - R_pri from the destination pixel's own jittered primary hit, and the *previous* camera for temporal inverse shifts;
   - whether thresholds must be jittered (RTXDI) or may be fixed.
3. **Paired-reuse rules:**
   - a symmetric neighbour-acceptance predicate A(p,q) = A(q,p) (enhanced-verify C3);
   - the cached-Jacobian-product update when a shifted sample is selected (C4);
   - one chosen pairwise-MIS formula (X16).
4. **Pseudo-code and an exhaustive GPU test.** T3/T4 with a zero-violation target, and a list of the most likely asymmetry bugs, such as FMA or relaxed-math differences between kernels. Compare IDs, not floats.

### Gap 5: `webgpu-pass-cost-budget`

**Read first.**

| Report | Sections |
|---|---|
| webgpu-platform `/Users/mark.boss/.claude/plans/do-a-deep-dive-shimmering-ritchie-agent-a64ceeae6903be301.md` | all, esp. §0, §2.1, §4.1–4.3, §5.1–5.7, §8.5 |
| reference-code `…-a3b89657b22d15f34.md` | §1.11 item 7, §3, §4.1–4.4, §2.4 (C2-Renderer compile-time lesson) |
| enhanced-paper `…-a91ab1d861819b276.md` | §2.6, §6.2–6.3, §7, §8.2–8.4 |
| restir-practice `…-a668fe13fb61070e8.md` | §3.6 (cost), §3.11 |
| validation-harness `…-a26c6df2c3f5e4af0.md` | §3.6 (ensemble memory) |

**Method.** Reuse the platform report's method (§1): in-page WebGPU code run in the Claude desktop built-in browser pane on the same M5 Pro, with Khronos Sponza fetched from raw.githubusercontent.com. Write no files outside the scratchpad, and do not touch the user's Chrome profile.

**Measure at 1280×720, and at 960×540 if time allows:**

- (a) Incoherent segment-visibility (any-hit) rays between the primary hit of pixel p and a secondary-hit point of a pixel 1–30 px away. These stand in for reconnection rays.
- (b) Compacted prefix random-replay: 1–3 closest-hit bounces with BSDF sampling, over ~30% of pixel×neighbour pairs, using subgroup-ballot append plus `dispatchWorkgroupsIndirect`.
- (c) A mock shift kernel:
  - load a 64-B reservoir (4× `vec4<u32>`);
  - rebuild the rc vertex from (inst, prim, bary);
  - 2 GGX + Lambert evals;
  - 1 visibility ray;
  - write a 16-B slot.
- (d) A mock suffix re-trace of 1–2 bounces + NEE (the moving-light case).
- (e) Cold `createComputePipelineAsync` compile time for a 60–120 KB WGSL module containing the traversal, BSDF, replay and shift, with exactly one traversal call site. Compare with two call sites.

**Deliver:**
- a per-pass ms table for the Enhanced pass graph: initial, temporal (forward + inverse shift, replays), spatial pre-pass with N = 3, resample, duplication map, resolve. Mark measured vs inferred;
- a per-pass binding table showing ≤ 10 storage buffers and ≤ 4 bind groups, with the merged-buffer scheme;
- a memory table for 720p and 1080p: reservoirs × 3, pairing slots, replay state, previous-frame V-buffer and lights, light tiles, duplication map, accumulation, ensemble mode;
- a go/no-go recommendation for interactive rates: internal resolution, max bounces, neighbour count, whether the retrace must be fused or separate, and whether CWBVH is required.
