# Reference Implementations of ReSTIR PT: a code-level deep dive for a WebGPU/WGSL port

> Report for implementation engineers. Topic: the actual shader code of public ReSTIR PT implementations,
> what each one computes, and how to port it to WGSL (Chrome, Metal backend, Apple M5 Pro, compute-shader BVH).
>
> **Where this file lives:** the orchestrator asked for `.../scratchpad/research/reference-code.md`. This session
> runs in **plan mode**, where the only writable file is this plan file. The report was therefore written here.
> No repositories were cloned to disk. All sources were read read-only, streamed to stdout through
> `raw.githubusercontent.com` and the GitHub REST tree API. One accidental write happened: a 14-byte file
> `scratchpad/.dummy_never` was created by a mis-typed `curl -o` probe. It is harmless and was not deleted.
>
> **Conventions.**
> - `file:line` refers to the upstream file at the commit that was read. Snapshot dates are in §0.
> - **[INFERENCE]** marks my own reasoning or derivation, as opposed to what the code or paper states.
> - **[UNVERIFIED]** marks a statement I could not confirm.
> - Vertex notation: `x0` is the camera, `x1` the primary hit, `x2` the secondary hit, and so on.
> - Falcor's `path.length` counts differently: it is 0 at `x1`, 1 at `x2`, so `path.length = L` means vertex `x_{L+1}`.
> - PSS means primary sample space. UCW means unbiased contribution weight (`W`). `p̂` is the scalar target function.

---

## 0. Executive summary

1. **The canonical code is `DQLin/ReSTIR_PT`, `Source/RenderPasses/ReSTIRPTPass/`.** It is Falcor 4.4 and Slang, with a BSD-3 repo license.
   - It implements GRIS in PSS. Each reservoir stores the selected path's PSS integrand `F = f/p` as an RGB value.
   - The target function is `p̂ = luma(F)`, with weights (0.299, 0.587, 0.114).
   - The reservoir is 88 bytes.
   - A per-pixel "reconnection data" buffer (256 B/pixel) caches **prefix random-replay results** computed in separate "retrace" passes. Those passes run *before* the temporal and spatial resampling passes.
   - Default shift is **Hybrid**: random replay of the prefix, then a reconnection at the first vertex pair that is both rough and far apart.
   - Temporal MIS is **Talbot** GRIS MIS. Spatial MIS is **pairwise** (defensive, GRIS Eq. 38) with confidence weights `M`.
   - Defaults: 1 candidate, 3 spatial neighbors in a 20-pixel radius, 1 spatial round, M-cap = 20·M_current.
   - Direct lighting is **not** computed by ReSTIR PT. It comes from a separate ReSTIR DI pass (`ScreenSpaceReSTIRPass`).
2. **Jacobian.** The code computes exactly the PSS Jacobian of the Enhanced paper, Eq. (2):
   `J = [p^y_{k-1}(ω') G(y_{k-1}→x_k) p^y_k(ω_k)] / [p^x_{k-1}(ω) G(x_{k-1}→x_k) p^x_k(ω_k)]`.
   - The three source terms are cached in the reservoir as `cachedJacobian = (p^x_{k-1}, p^x_k, G^x)`, so the base path's `x_{k-1}` never has to be stored.
   - All pdfs are **single-lobe-group** pdfs, because the path space is lobe-extended (GRIS §7.6).
3. **NVIDIA RTXDI 3.0/3.1** (2025–2026) ships a production ReSTIR PT: `NVIDIA-RTX/RTXDI-Library/Include/Rtxdi/PT/*`. It includes several Enhanced-paper features:
   - footprint reconnection criteria;
   - a duplication map with history reduction;
   - forced NEE-light reconnection;
   - compatibility-guided neighbor selection (Junkins et al.);
   - DLSS-RR decorrelation.

   It differs structurally from Falcor:
   - The target is in a *mixed PSS / solid-angle measure*.
   - The reconnection vertex is stored as position + normal + cached radiance. There is no hit or material reference, so the rc-vertex BSDF is **not** re-evaluated.
   - MIS is "MIS-like" `pi/piSum` rather than pairwise.
   - **License:** the NVIDIA RTX SDK license is proprietary and prohibits distributing derivative source. WGSL ships as source to the browser, so **do not port its code; read it only for ideas.**
4. **`EvanLuo42/ReSTIR-PT-Enhanced`** (Falcor 7/8, BSD-3 Falcor license, August 2026) is an **unofficial but faithful** implementation of the 2026 Enhanced paper. It matches the paper's constants exactly:
   - 3 pairing textures of 254/230/210 px with σ=16;
   - footprint `c/100 = 2e-4`;
   - duplication map 17×17 → /288, `c_Cap = lerp(20, 1, D^0.1)`.

   It also implements replay stream compaction via atomic queues and indirect dispatch, dual motion vectors, "vector weights" color estimation, and unified DI/GI with presampled light tiles. **Its pass structure is the best template for a WebGPU port.** However, its neighbor and history pairwise MIS weight appears to have swapped target arguments (§2.3, [INFERENCE]). Do not copy its MIS formulas.
5. **`Domenicobrz/C2-Renderer`** is a real **WebGPU/WGSL ReSTIR PT** (MIT, TypeScript/Svelte). It is progressive/offline oriented: static camera, no primary G-buffer, and random replay starts from a stored `firstVertexSeed`. It supports pairwise, GBH and biased-1/M MIS. Its WGSL-specific lessons are directly applicable:
   - Tint/Metal compile-time explosion when the replay function has multiple call sites.
   - Tiled dispatch to avoid GPU timeouts.
   - Uniform-buffer array layout.
   - Storage-buffer count limits.
6. **Other public code.**
   - `TomClabault/HIPRT-Path-Tracer` has ReSTIR PT plus an "Enhanced fused single-reservoir DI-GI" mode, but it is GPL-3.0.
   - `HummaWhite/Vulkan-ReSTIR-PT` is self-described as incomplete.
   - `elite-sheep/gradient-restir` (Falcor, EG 2026) is gradient-domain ReSTIR PT.
   - `NVIDIA-RTX/RTXPT` has ReSTIR DI/GI through RTXDI but **no ReSTIR PT**.
7. **WGSL porting.** The hard parts are:
   - no hardware ray tracing, so BVH traversal must be a single call site and pipelines must be split to control code size;
   - no generics, overloading, default arguments, `out` parameters, recursion or `isnan`;
   - a different struct layout, where `vec3` aligns to 16, so reservoirs must be hand-packed into `array<vec4<u32>>`;
   - no bindless textures, so use texture arrays or atlases;
   - HLSL `select` argument order is reversed relative to WGSL;
   - no 64-bit integers;
   - `textureSample` is not allowed in compute shaders;
   - read-write storage textures only for r32 formats in core;
   - the default `maxStorageBufferBindingSize` is 128 MiB, which a 1080p 88-B reservoir buffer (≈182 MB) exceeds.

   Subgroups (Chrome ≥134) and `shader-f16` are available.

Snapshot dates (GitHub `pushed_at`): `DQLin/ReSTIR_PT` master 2025-07-20; `NVIDIA-RTX/RTXDI` main 2026-09-17; `NVIDIA-RTX/RTXDI-Library` main 2026-09-03 (v3.1.0 per ChangeLog); `EvanLuo42/ReSTIR-PT-Enhanced` master 2026-08-21; `Domenicobrz/C2-Renderer` master 2025-07-09; `NVIDIA-RTX/RTXPT` main 2026-03-19.

---

## 1. Falcor ReSTIR PT (Lin et al. 2022): `DQLin/ReSTIR_PT`

### 1.1 File map (`Source/RenderPasses/ReSTIRPTPass/`)

| File | Role |
|---|---|
| `ReSTIRPTPass.h/.cpp` (90 KB) | Host pass: static params → defines, resources, per-frame pass scheduling, UI |
| `Params.slang` | Host/device shared `RestirPathTracerParams`, enums (`ShiftMapping`, `ReSTIRMISKind`, `LocalStrategy`, …), reservoir addressing |
| `StaticParams.slang` | Compile-time constants from defines (`kCandidateSamples`, `kShiftStrategy`, `kMaximumPathLength = 15`, `kSeparatePathBSDF`, …) |
| `PathReservoir.slang` | `ReSTIRPathFlags`, `ReconnectionData`, `PixelReconnectionData`, `TriMeshHitInfo`, `PathReservoir` + RIS/GRIS merge ops |
| `PathState.slang` | Live path state, `PathFlags` bitfield |
| `PathBuilder.slang` | Streams path-tree candidates into the reservoir while tracing (`addNeeVertex`, `addEscapeVertex`, `markEscapeVertexAsRcVertex`) |
| `PathTracer.slang` (86 KB) | Path tracer: NEE/MIS, lobe-separated BSDF sampling, rc-vertex selection, random-replay variants, temporal suffix update |
| `Shift.slang` | Shift mappings (reconnection, random replay, hybrid), Jacobians, `shiftAndMergeReservoir`, visibility |
| `GeneratePaths.cs.slang` | Writes background color for pixels with no primary hit |
| `TracePass.cs.slang` | Initial candidate generation (1 thread/pixel, 16×16) |
| `TemporalPathRetrace.cs.slang` / `SpatialPathRetrace.cs.slang` | Precompute hybrid-shift prefix replays → `reconnectionDataBuffer` |
| `TemporalReuse.cs.slang` / `SpatialReuse.cs.slang` | GRIS resampling (Talbot / Pairwise / Constant variants), output color |
| `ComputePathReuseMISWeights.cs.slang` + `Data/16RooksPattern256.txt` | Baseline "Bekaert-style path reuse" (`BPR`), not ReSTIR |
| `LoadShadingData.slang` | Vertex/material loading, including `loadShadingDataWithPrevVertexPosition` (view dir from an arbitrary previous vertex) |
| `Source/Mogwai/Data/ReSTIRPT.py`, `ReSTIRPTDemo.py` | Render graph: `VBufferRT` (center sample, Mip0) → `ReSTIRPTPass` + `ScreenSpaceReSTIRPass` (ReSTIR DI) → Accumulate → ToneMapper |

Falcor utilities that matter:
- `Utils/Sampling/TinyUniformSampleGenerator.slang` (LCG);
- `Utils/Math/HashUtils.slang` (Jenkins, TEA);
- `Rendering/Materials/MaterialShading.slang` and `BxDF.slang` (`FalcorBSDF`: lobe probabilities, `pdfSingle`, `evalPdfAll`, `classifyAsRough`, `hasRoughComponent`);
- `Utils/Geometry/GeometryHelpers.slang` (`computeRayOrigin`);
- `Scene/HitInfo.slang` (packed hit).

### 1.2 Frame graph and pass sequence

External inputs, from `ReSTIRPT.py`:
- `vbuffer`: packed primary hit from `VBufferRT`. The sample pattern is **Center**, so there is no sub-pixel jitter.
- `motionVectors`: RG32F, screen-UV delta from current to previous.
- `directLighting`: the RGBA32F output of `ScreenSpaceReSTIRPass`. It contains primary emission plus ReSTIR-DI diffuse and specular reflection. See `ScreenSpaceReSTIRPass/FinalShading.cs.slang:78-80`.

`ReSTIRPTPass::execute` (`ReSTIRPTPass.cpp:676-783`), for `samplesPerPixel = 1` and default settings:

| # | Pass (file) | Condition | Reads | Writes |
|---|---|---|---|---|
| 1 | `generatePaths` (`GeneratePaths.cs.slang`) | `restir_i==0` | vbuffer, env map | `outputColor` for background pixels |
| 2 | `tracePass` (`TracePass.cs.slang` → `PathTracer.slang`) | always | vbuffer, scene/BVH, light samplers | `outputReservoirs[p]` (candidate reservoir with W finalized), `outputColor[p] = LDeltaDirect/spp` |
| 3a | `PathRetracePass(temporal)` (`TemporalPathRetrace.cs.slang`) | temporal on, hybrid, history exists | outputReservoirs, temporalReservoirs, vbuffer, temporalVBuffer, motionVectors | `reconnectionDataBuffer[p].data[0]` (current→prev prefix replay), `.data[1]` (prev→current) |
| 3b | `PathReusePass(temporal)` (`TemporalReuse.cs.slang`) | temporal on, history exists (`mReservoirFrameCount>0`) | same + reconnectionData | `outputReservoirs[p]` (in place); adds `(F·W + directLighting)/spp` to `outputColor` only if spatial is off |
| 4a | `PathRetracePass(spatial, round r)` | spatial on, hybrid | reservoirs, vbuffer | `reconnectionDataBuffer[p].data[2i]` (center→neighbor i), `.data[2i+1]` (neighbor i→center) |
| 4b | `PathReusePass(spatial, round r)` (`SpatialReuse.cs.slang`) | spatial on | "input" reservoirs, reconnectionData, neighborOffsets | "output" reservoirs; on the last round, `outputColor += (F·W + directLighting)/spp` |
| 5 | copies | temporal on | – | `temporalReservoirs ← outputReservoirs` if `numSpatialRounds` is even or spatial is off; `temporalVBuffer ← vbuffer` |

**Buffer ping-pong** (`ReSTIRPTPass.cpp:1665,1675,1756,1763`):
- In spatial round `r`, the pass reads `outputReservoirs` when `r` is even and `temporalReservoirs` when `r` is odd. It writes the other one.
- With one round, the spatial result lands directly in `temporalReservoirs`, which is next frame's history. That is why the copy happens only when the round count is even.
- The temporal pass reads the previous frame from `temporalReservoirs` and writes `outputReservoirs` in place.
- For `spp > 1` there are `spp` independent temporal chains (`mpTemporalReservoirs[restir_i]`), and the whole pipeline loops over `restir_i`.

**Reservoir addressing** (`Params.slang:157-168`, `getReservoirOffset`): 16×16 screen tiles in scanline order, Morton order inside each tile (`interleave_16bit(pixel) & 0xFF`). The buffer holds `screenTiles.x*screenTiles.y*256` elements, padded to whole tiles. The purpose is cache locality for neighbor fetches.

**Retrace/reuse RNG coupling.** Each retrace pass and its matching reuse pass seed the neighbor-selection RNG identically:
- `TinyUniformSampleGenerator(pixel, (kCandidateSamples+1+gNumSpatialRounds)*seed + kCandidateSamples + 1 + gSpatialRoundId)`;
- `startIndex = sampleNext1D(sg)*8192`.

As a result they visit identical neighbors (`SpatialPathRetrace.cs.slang:115,129` vs `SpatialReuse.cs.slang:153,183`). **A port must preserve this invariant** or store the neighbor list explicitly. Storing the list is better.

### 1.3 Data structures and byte layouts

#### `PathReservoir` (`PathReservoir.slang:224-236`): 88 B (`ReSTIRPTPass.cpp:1211` `baseReservoirSize = 88`)

HLSL StructuredBuffer packing: `float3` is 12 B with 4-B alignment.

| Offset | Field | Type | Meaning |
|---|---|---|---|
| 0 | `M` | f32 | Confidence weight. Float, because the temporal history can be fractional (`min(20·M_cur, M_prev)`) |
| 4 | `weight` | f32 | **Overloaded**: `w_sum` during RIS/merge; the UCW `W` after `finalizeRIS`/`finalizeGRIS` |
| 8 | `pathFlags` | i32 | `ReSTIRPathFlags` bitfield (below) |
| 12 | `rcRandomSeed` | u32 | LCG state at the rc vertex, used to replay the **suffix** (temporal update) |
| 16 | `F` | f32×3 | PSS integrand `f(x̄)/p(x̄)·ω_t` of the selected path **in this pixel's domain**. Shading = `F·W` |
| 28 | `lightPdf` | f32 | Solid-angle light pdf for the path's final light vertex. Used to re-evaluate NEE/BSDF MIS after a shift |
| 32 | `cachedJacobian` | f32×3 | `(p^x_{k-1}, p^x_k, G^x_k)`: source pdf at `x_{k-1}` toward `x_k`, source scatter pdf at the rc vertex, `|cos θ_k|/d²` |
| 44 | `initRandomSeed` | u32 | LCG state **just before the BSDF sample at x1**. Seeds prefix random replay |
| 48 | `rcVertexHit` | `TriMeshHitInfo` 16 B | `instanceID` (0xffffffff = none/env), `primitiveIndex`, `float2 barycentrics` (full precision; a lossy unorm2x16 variant is commented out) |
| 64 | `rcVertexWi[0]` | f32×3 | Direction **leaving the rc vertex** toward the next vertex or light (Falcor calls it "incident"). For an env rc vertex: the env direction |
| 76 | `rcVertexIrradiance[0]` | f32×3 | Cached radiance arriving at `x_k` along `rcVertexWi`, **excluding `f` at `x_k`**. Exact semantics per case in §1.5.5 |

With `BPR=1` (baseline path reuse only) the size is 128 B, adding a second `rcVertexWi/Irradiance`, `rcLightPdf`, and `rcVertexBSDFLightSamplingIrradiance`.

#### `ReSTIRPathFlags` bit map (`PathReservoir.slang:21-139`)

| Bits | Field |
|---|---|
| 0–3 | `pathLength`: Falcor `path.length` of the **last scattering vertex**. For NEE paths it is the length where NEE was taken. For a BSDF-hit emitter it is (emitter vertex length − 1). Max 15 |
| 4–7 | `rcVertexLength`: `path.length` of the rc vertex. `15` (`kMaximumPathLength`) = no rc vertex |
| 8 / 9 | delta event before rc (the scatter at `x_{k-1}` that produced `x_k`) / after rc (the scatter at `x_k`) |
| 10 / 11 | transmission event before / after rc |
| 16 | `lastVertexNEE` (path ends with an NEE-sampled light vertex) |
| 18–19 | `lightType`: 0 = EnvMap, 1 = Emissive (triangles), 2 = Analytic (point/spot/directional/area analytic) |
| 26 / 27 | "specular bounce" (lobe group) before / after rc. This is the **lobe index** of GRIS §7.6: 0 = diffuse group, 1 = specular group |
| 30–31 | `rcLightType` (BPR only) |

Derived predicates, from `Shift.slang:434-436`:
- `isRcVertexFinal = pathLength == rcVertexLength`: the rc vertex is the last scattering vertex. It is followed by NEE or by a BSDF hit on a light.
- `isRcVertexEscapedVertex = pathLength+1 == rcVertexLength && !lastVertexNEE`: **the light vertex itself** (an emitter hit by BSDF sampling) is the rc vertex.
- `isRcVertexNEE = isRcVertexFinal && lastVertexNEE`.

#### `ReconnectionData` / `PixelReconnectionData` (`PathReservoir.slang:142-174`)

- `ReconnectionData = { HitInfo rcPrevHit (16 B), float3 rcPrevWo (12 B), float3 pathThroughput (12 B) }`, 40 B. It is the **result of replaying a prefix into a destination pixel**:
  - `rcPrevHit`: the hit of `y_{k-1}`;
  - `rcPrevWo`: `sd.V` at `y_{k-1}`, i.e. the direction toward `y_{k-2}`;
  - `pathThroughput`: the product of `f·cos/p` over `y_1…y_{k-1}`.
- `PixelReconnectionData = ReconnectionData[RCDATA_PATH_NUM] + float4 pad[RCDATA_PAD_SIZE]`:
  - real-time: 6 entries + 16 B = **256 B/pixel** (supports ≤3 spatial neighbors × 2 directions);
  - offline (`neighborCount>3`): 12 entries + 32 B = **512 B/pixel**. Selected in `ReSTIRPTPass.cpp:1157,1862-1863`.
- Slot use:
  - temporal: `[0]` = current reservoir replayed into the prev pixel, `[1]` = temporal reservoir replayed into the current pixel;
  - spatial neighbor `i`: `[2i]` = center→neighbor, `[2i+1]` = neighbor→center.

#### Other state

- `PathState` (`PathState.slang:64-111`) holds the usual throughput and origin, plus:
  - `prefixThp` (throughput up to and including the rc-vertex scatter) vs `thp` (after it);
  - `rcVertexPathTreeIrradiance` (BPR);
  - `LDeltaDirect`: direct light via delta or transmission primary scatter, which ReSTIR DI cannot handle;
  - `russianRoulettePdf`, `prevScatterPdf`, `rcPrevVertexHit/Wo`;
  - replay controls: `enableRandomReplay`, `randomReplayIsNEE`, `randomReplayIsEscaped`, `randomReplayLength`, `isReplayForHybridShift`, `isLastVertexClassifiedAsRough`.
- `PathFlags` (16-bit): `active`, `hit`, `transmission`, `specular`, `delta`, `volume`, `insideDielectricVolume`, `lightSampledUpper/Lower`, `diffusePrimaryHit`, `freePath`, `specularBounce`, `specularPrimaryHit`.
- `PathBuilder` (`PathBuilder.slang:24-43`) holds:
  - the candidate rc-vertex bookkeeping (`rcVertexHit`, `rcVertexWi`, `cachedRandomSeed`, `rcVertexLength`, `pathFlags`, `cachedJacobian`);
  - **its own** `TinyUniformSampleGenerator`, seeded by `jenkinsHash(pathSg.next())`, which is used only for reservoir accept/reject decisions.
- `RestirPathTracerParams` (`Params.slang:133-153`): `useFixedSeed`, `seed`, `fixedSeed`, `lodBias`, `frameDim`, `screenTiles`, `frameCount`, `localStrategyType`, `rejectShiftBasedOnJacobian`, `jacobianRejectionThreshold`, `specularRoughnessThreshold`, `nearFieldDistance`, 8 B of padding.
- Falcor `PackedHitInfo` (`Scene/HitInfo.slang`, `HitInfo.cpp`):
  - hit type in the top bits of word 0 (`None=0`), then instance and primitive indices packed into 32 or 64 bits;
  - barycentrics as unorm2x16;
  - texture format RG32Uint or RGBA32Uint depending on the scene;
  - validity test `packed.x != 0`.

### 1.4 RNG and seeding: the heart of random replay

- **Path RNG** = `TinyUniformSampleGenerator` = a 32-bit LCG (`LCG.slang`): `state = 1664525*state + 1013904223`.
  - `sampleNext1D = (next() >> 8) * 2^-24` (`SampleGeneratorInterface.slang`).
  - Pixel seeding: `seed = blockCipherTEA(interleave_32bit(pixel), sampleNumber).x` (16 TEA rounds).
  - **Replay seeding** is `SampleGenerator(uint seed)`, which **sets the LCG state directly** (`TinyUniformSampleGenerator.slang:22-25`). `getCurrentSeed()` returns the raw state. Storing one `u32` is enough to resume the exact stream.
- **Resampling RNG is separate.** `PathBuilder.sg` makes RIS accept/reject decisions during tracing, so those decisions never perturb the replayable path stream. The reuse passes use their own `TinyUniformSampleGenerator`. **Port rule: never draw resampling randoms from the replayable path stream.**
- Per-frame seed (`ReSTIRPTPass.cpp:1535`): `seed = seedOffset + spp*frameCount (+restir_i)`.
- Path stream per candidate: `SampleGenerator(pixel, (itersPerPass+1+numSpatialRounds)*seed + sampleIdx)` (`PathTracer.slang:220`).
- Temporal resampling uses sample number `… + kCandidateSamples`. Spatial round `r` uses `… + kCandidateSamples + 1 + r`.
- `initRandomSeed` is captured at `x1` **after** NEE and RR at x1 would have happened, and **immediately before** the x1 BSDF sample (`PathTracer.slang:1393-1397`). With `disableDirectIllumination=true` there is no NEE at x1, so nothing is consumed there anyway.
- **Random-number consumption must be identical in base and replay.** During replay:
  - NEE at intermediate vertices is skipped through `skipLightSampleRandomNumbers` (`PathTracer.slang:796-817,1354-1366`). It consumes the light-type selection number plus the sampler's count: env 2, power sampler 1+1+2 (`EmissivePowerSampler.slang:91-96`), analytic 1+2 for rect/sphere/disc/distant (`PathTracer.slang:670-687`).
  - Russian roulette consumes one number and does not terminate (`PathTracer.slang:1378-1388`).
  - Comment at `PathTracer.slang:620` ("when random number replay, this can fails with light BVH …") suggests the LightBVH sampler breaks this contract. The default emissive sampler is `Power`.

### 1.5 Base path tracer and initial candidates

#### 1.5.1 Control flow (`TracePass.cs.slang:19-43`, `PathTracer.slang`)

```
generatePath: path.hit = vbuffer[pixel]; path.sg = TEA-seeded LCG; pathBuilder.init(sg); reservoir.init()
while active:
  if hit: handleHit(path); if terminated break; nextVertex(path)   // closest-hit trace; length++
  else:   handleMiss(path)
pathBuilder.finalize(reservoir)       // M = 1
writeOutput: finalizeRIS (W = w_sum/(p̂·M)), store reservoir, outputColor = LDeltaDirect/spp
```

Pixels without a primary hit are skipped (`TracePass.cs.slang:52-53`). Multiple candidates loop `sampleIdx` and are merged with `mergeInSamplePixel` (w = that tree's `w_sum`, `M += 1`, `PathTracer.slang:1947-1987`).

#### 1.5.2 `handleHit` (`PathTracer.slang:964-1471`), in order

1. Load shading data. Apply homogeneous-volume absorption and nested-dielectric false-hit rejection (`handleNestedDielectrics`, up to 16 rejected hits).
2. **Emission.**
   - `computeEmissive` excludes `x2` emission when the x1 scatter was neither delta nor transmission: `path.length==1 && !transmission && !delta → false` (`:1025`), because ReSTIR DI covers it. Primary emission comes from ReSTIR DI's output.
   - MIS weight: `evalMIS(1, path.pdf, 1, lightPdf)`, where `path.pdf` is the **full mixture pdf** (`pdfAll`) of the previous BSDF sample, and `lightPdf = P(select emissive) · emissiveSampler.evalPdf(prevOrigin, prevNormal, upperHemisphere, hit)` (solid angle).
   - Candidate streamed via `addEscapeVertex(pathLength = length-1, dir, Lr, postfix = thp·Le·mis, …)`.
3. **Escaped emitter as rc vertex** (hybrid only, `:1093-1118`). If the escape candidate was *selected*, `path.length ≥ 2`, and no earlier rc vertex exists, the emitter can become the rc vertex. The conditions are the distance and last-vertex-rough tests. `markEscapeVertexAsRcVertex` stores:
   - `rcVertexLength = length`, the emitter hit, `rcVertexIrradiance = Le`;
   - `cachedJacobian.x = prevScatterPdf`, `.z = |n_f·V|/d²`;
   - `lightPdf`, flags.
4. Terminate if the bounce budget is exhausted (`hasFinishedSurfaceBounces`) or a replay-for-escape reaches its target.
5. **rc-vertex test for the current vertex `x_{L+1}`**, where L = `path.length` (`:1157-1193`). For `L ≥ 1` and `L < current rcVertexLength`:
   ```
   isFarField  = |x_{L+1} - x_L| >= nearFieldDistance          (0.1 scene units)
   curRough    = kSeparatePathBSDF ? hasRoughComponent(sd, thr)   // pDiffuseReflection>0 || roughness>thr
                                   : roughness > thr             (thr = specularRoughnessThreshold = 0.2)
   canConnect  = !(Distance && !isFarField || Roughness && !(curRough && lastVertexRough))
   is_rcVertex = hybrid ? canConnect : (L == 1)                   // pure reconnection always uses x2
   ```
   If it is the rc vertex:
   - `pathBuilder.rcVertexHit = hit`, `cachedJacobian.x = prevScatterPdf` (single-lobe pdf of the scatter at `x_L`);
   - before-rc delta, transmission and specular flags from the scatter that produced this vertex;
   - `cachedJacobian.z = |dot(faceN, V)| / d²`;
   - `rcVertexLength = L`.
6. **NEE** (skipped at x1 when `disableDirectIllumination`). If `L == rcVertexLength`, `cachedRandomSeed = sg.state` is captured **before** light sampling (`:1272-1275`). Then:
   - `generateLightSample`: pick a light type uniformly among the enabled types, then sample it, with hemisphere rejection by shading and face normals.
   - MIS: `evalMIS(1, ls.pdf, 1, evalPdfBSDF(sd, ls.dir))`, where the BSDF pdf is over **all lobes**. **Analytic lights get mis = 1** and a pdf placeholder equal to the selection probability.
   - `Lr = f_all(sd, ls.dir)·cos · Li·mis · thpTotal`. After a shadow ray, the candidate is `addNeeVertex(pathLength = L, …, postfix = (is_rcVertex ? 1 : f·cos)·Li·thp)`.
   - In replay mode, NEE runs only at the terminating vertex; otherwise the randoms are skipped.
7. Russian roulette (**off by default**): `prob = max(0, 1 - luma(thpTotal))`. The throughput is **not divided**; instead `russianRoulettePdf *= 1-prob`, which is used as the RIS source pdf.
8. **BSDF sample** (`generateScatterRay`, `:298-429`), with `kSeparatePathBSDF=true`:
   - `sampleBSDF(..., useDeterministicBSDF=false)` returns `weight = f_lobe·cos/pdf_lobe_with_selection`, `pdf = pdfAll`, `pdfSingle = p_select·pdf_lobe`;
   - `result.pdf = pdfSingle`, `path.pdf = pdfAll` (kept for the next hit's MIS), `prevScatterPdf = pdfSingle`;
   - if `L == rcVertexLength`: `cachedJacobian.y = pdfSingle`, and the after-rc delta, transmission and specular flags are set;
   - "reflection" with roughness > thr counts as a diffuse bounce;
   - `recordPrefixThp()` while `L ≤ rcVertexLength`: the prefix throughput absorbs the rc-vertex scatter too, so `thp` restarts at 1 **after** the rc vertex.
9. **Lobe-specific veto** (`:1403-1425`). After sampling at a would-be rc vertex, `seenAsConnectible = !specularBounce || roughness > thr`.
   - If the sampled lobe is specular-group and the material is glossy, the vertex **stops being the rc vertex** (`rcVertexLength = 15`). Note that NEE at this vertex was already streamed with rc = this vertex. This is consistent with GRIS §7.6: NEE vertices are rough if any lobe is rough.
10. After the BSDF sample at the rc vertex: `cachedRandomSeed = sg.state` (`:1428-1431`), which is the suffix replay seed. `rcVertexWi[0] = path.dir` (`:1434-1437`).

`handleMiss` (`:1476-1575`): env radiance with MIS `evalMIS(1, pdfAll, 1, P(env)·envPdf(dir))`, streamed through `addEscapeVertex(pathLength = length)`. It can mark the **env map as rc vertex** (hit invalid, `rcVertexWi = dir`, `rcVertexIrradiance = Le`, `cachedJacobian.z = 1`) if `length ≥ 1`, `length+1 < rcVertexLength`, and the last vertex is rough.

#### 1.5.3 RIS streaming math (`PathReservoir.slang:290-310`, `PathBuilder.slang:117-180`, `:436-450`)

For each light-reaching candidate `j` of the path tree (NEE at vertex v, or a BSDF-hit emitter or env map):

```
F_j    = contribution in PSS (f/p·ω_t), RGB       // what the path tracer would add to L
p̂_j    = luma(F_j) = dot(F_j, (0.299, 0.587, 0.114))
w_j    = p̂_j / π_j          (π_j = product of RR survival probabilities; 1 if RR off)
M += 1; w_sum += w_j; if u_resample * w_sum <= w_j: select j (copy flags, F, rc data, seeds)
finalize: M = 1                 // one "path tree" = one sample
finalizeRIS: W = w_sum / (p̂(y)·M)
```

- **[INFERENCE]** This is RIS over a set of disjoint domains: different path lengths and techniques, each with PSS source pdf 1. The estimator `F_y·W` is an unbiased estimator of `Σ_j F_j`, the path tracer's radiance estimate.
- With `N` candidate trees, `w_sum` sums over all trees and `M = N`.
- The luma weights are Rec.601 (0.299/0.587/0.114), not Rec.709.

#### 1.5.4 What the reservoir must remember to re-create the path elsewhere

- `initRandomSeed`: replays the prefix from any destination `x1'`.
- `pathFlags`: path length, rc index, NEE vs BSDF ending, lobe groups around rc, light type.
- `rcVertexHit`: the rc vertex is **re-loaded from the triangle and barycentrics** in the current scene. Materials and textures are therefore re-evaluated and moving instances are handled.
- `rcVertexWi`, `rcVertexIrradiance`, `lightPdf`: the suffix is cached, so it is never retraced except in the temporal update.
- `cachedJacobian`: source-path densities, so `x_{k-1}` of the source path is never needed.
- `rcRandomSeed`: replays the suffix for dynamic-scene updates.

#### 1.5.5 Exact semantics of `rcVertexIrradiance` (non-BPR), with L_k = path.length of the rc vertex

| Case | Stored `rcVertexWi` | Stored `rcVertexIrradiance` |
|---|---|---|
| Path continues beyond `x_k`, ends by NEE at `x_v`, v>k | dir sampled at `x_k` | `f(x_v)cos·Le·mis/p_light · Π_{k<j<v}(f cos/p)` (suffix product, excludes f at x_k) |
| Continues beyond `x_k`, ends by BSDF hit on emitter/env | dir sampled at `x_k` | `Π_{k<j}(f cos/p) · Le · mis` |
| NEE **at** `x_k` (`isRcVertexNEE`) | light direction | `Le` (= `Li·p_light/mis`, `PathBuilder.slang:168-172`); `lightPdf` stored separately |
| BSDF hit on emitter/env directly after `x_k` (final, BSDF) | scattered dir | `Le` (divided by mis, `PathBuilder.slang:103-107`) |
| Emitter itself is rc (`isRcVertexEscapedVertex`) | 0 | `Le` of emitter (`markEscapeVertexAsRcVertex`) |
| Env map is rc (hit invalid) | env direction | `Le(dir)` |

### 1.6 Shift mappings (`Shift.slang`)

`computeShiftedIntegrand_` (`:50-87`) dispatches on the compile-time `kShiftStrategy`: 0 Reconnection, 1 RandomReplay, 2 Hybrid (default). It returns `F(y)` in the destination domain and writes `dstJacobian` (= |∂y/∂x| in PSS); **0 means the shift failed**.

#### 1.6.1 Reconnection-only (`kShiftStrategy=0`)

`computeShiftedIntegrandReconnection(useHybridShift=false)`. The rc vertex is always `x2` (`rcVertexLength=1`). The source quantities are recomputed from the source pixel's primary hit (the neighbor's vbuffer), so nothing is cached. This is ReSTIR GI's shift with a correct Jacobian and MIS.

#### 1.6.2 Random replay only (`kShiftStrategy=1`)

`traceRandomReplayPath` (`PathTracer.slang:1578-1619`) re-runs the full path tracer from the destination `x1'`, starting at `initRandomSeed`:
- `enableRandomReplay=true`;
- light sampling only at the base path's final vertex, with the same technique (NEE or escape) and the same length;
- Jacobian = 1 (PSS), `F(y) = path.L`.

#### 1.6.3 Hybrid shift (default)

The work is split across two passes.

**(a) Retrace pass** (`traceHybridShiftRays` → `traceRandomReplayPathHybridSimple`, `PathTracer.slang:1621-1692`). It replays the prefix from destination `y1 = x1'` using `initRandomSeed`, with `isReplayForHybridShift=true` and `rcVertexLength = src rcVertexLength`, and stops at `y_{k-1}`. It records `rcPrevHit = y_{k-1}`, `rcPrevWo`, and `Tp = getCurrentThp()`. If there is no rc vertex, it returns the full `path.L`. Along the way it enforces **invertibility** (`PathTracer.slang:1194-1261`):
- At every intermediate `y_j` (1 ≤ j ≤ k−2): if `(y_{j-1}, y_j)` would pass the connectability test, the offset path would have picked an earlier rc vertex. The shift is non-invertible, so `invalidateAndTerminateReplayPath` (`L=0`, `rcPrevWo=0`).
  - With separate lobes the exact test is (`:1202-1217`): an NEE-terminating y_j that is rough while the last vertex is rough → invalid; `y_{k-1}` with an acceptable previous vertex → invalid; then overridden to "valid" when the distance condition fails (near field).
  - Also (`:1403-1415`): after sampling at `y_j`, if the sampled lobe makes it "seen as connectible", the previous vertex is rough, and the segment is far → invalid.
- At `y_{k-1}` (L == rcVertexLength−1): it must be able to produce the copied lobe group toward `x_k`:
  - if the base's before-rc bounce was specular-group, require `roughness > thr`;
  - otherwise require a rough component (`hasRoughComponent(sd, 1.0)`, effectively "has a diffuse lobe").
  - Otherwise the shift fails. On success the path terminates here and saves `rcPrevVertexHit/Wo`.
- Termination-by-type must match. An escaped base path must also escape at the same length, and an NEE base path must do NEE at the same length. A path-length mismatch → 0 (`:1669-1673`).
- **Emitter-as-rc during replay:** if the replay itself finds a valid rc-able emitter hit, it is non-invertible → `L=0` (`:1104-1107`).

**(b) Reuse pass** (`computeShiftedIntegrandHybrid`, `Shift.slang:116-200`):

```
if rcVertexLength == 1:  Tp = 1; dstRcPrev = dst primary (x1')
     + roughness test on x1' for the copied lobe group, else Tp = 0
else: (dstRcPrevHit, dstRcPrevWo, Tp) = rcData from the retrace pass
if Tp>0 and rc exists (rcVertexLength <= pathLength or escaped-vertex rc):
   if rcPrevWo == 0: fail
   load y_{k-1} shading (V = rcPrevWo)
   [optional temporal update of cached suffix, §1.7.4]
   rcTp = computeShiftedIntegrandReconnection(dst=y_{k-1}, src=x_{k-1} (only used if rc==x2),
                                              useHybridShift=true, useCachedJacobian = rcVertexLength>1)
   J = reconnection Jacobian
return Tp * rcTp        // if no rc vertex: Tp is already the full replayed radiance, J = 1
```

**(c) The reconnection itself** (`computeShiftedIntegrandReconnection`, `Shift.slang:383-572`). With `x_k` the rc vertex (re-loaded with its view direction toward `y_{k-1}`), `ω' = normalize(x_k − y_{k-1})`, and `ω_k = rcVertexWi`:

```
if src has delta before or after rc: fail                       // (:438-442)
if Distance condition && hybrid: require |x_k - y_{k-1}| >= nearFieldDistance   (:454-458)
G^y    = |dot(n_f(x_k), ω')| / |x_k - y_{k-1}|^2                 // face normal at x_k
J      = G^y / cachedJacobian.z                (or G^y·d_x²/|cos_x| if not cached)
p1^y   = pdfSingle(y_{k-1}, ω'; lobes of src before-rc group); p1^yAll = mixture pdf
J     *= p1^y / cachedJacobian.x              (or evalPdf(x_{k-1}) if not cached)
f1     = f_lobe(y_{k-1}, ω')·cos
if not escaped-vertex rc:
   p2^y = pdfSingle(x_k: -ω' → ω_k; after-rc lobes or ALL lobes if rc is NEE-final); p2^yAll
   dstPDF2 = isRcVertexNEE ? lightPdf : p2^y
   f2 = f(x_k: -ω' → ω_k)·cos       (same lobe set)
if f1==0 or f2==0: fail                                           // "connection point behind surface"
F_y    = (f1/p1^y) · (f2/dstPDF2) · rcVertexIrradiance
if escaped-vertex rc:  F_y *= balance(p1^yAll, lightPdf)          // stored lightPdf (see caveat)
if rc is final and useMIS and lightType != Analytic:
    mis = isNEE ? balance(lightPdf, p2^yAll) : balance(p2^yAll, lightPdf);  F_y *= mis
    if !isNEE: J *= p2^y / cachedJacobian.y
if rc not final and not escaped: J *= p2^y / cachedJacobian.y
if J invalid (<=0, NaN, Inf): fail
visibility y_{k-1} → x_k (tMax·0.999)
optional: if max(J,1/J) > 1 + jacobianRejectionThreshold: fail (unbiased rejection)
update reservoir.cachedJacobian = (p1^y, p2^y, G^y)   // the sample now lives in the dst domain
```

Env map as rc vertex (hit invalid, `Shift.slang:403-432`, only with MIS, env light type, escaped):
- `F_y = f1/p1^y · balance(p1^yAll, lightPdf) · Le`;
- `J = p1^y / p1^x`, with no geometry term.

This equals **Eq. (2) of the Enhanced paper** exactly (p^x_k replaced by 1 when k = d). The Enhanced paper also states that the Jacobian is 1 without reconnection.

#### 1.6.4 Reverse shifts for MIS

There is no special code. The *same* `computeShiftedIntegrand` is called with src/dst swapped:
- **temporal:** `computeShiftedIntegrand(dst = temporal pixel (prev vbuffer, usePrev=true → prev camera position for the replay ray), src = central, reservoir = candidate)` using `rcData[0]`;
- **spatial:** `dst = neighbor`, `src = central`, using `rcData[2i]`.

The prefix replays for both directions were precomputed in the retrace passes.

#### 1.6.5 Failure handling

- A failed shift returns `F=0` and `J=0`.
- `merge*` still adds `M_i` to `M` before the `w==0` early-out (`PathReservoir.slang:323-327`, `358-362`). A failed neighbor therefore still counts toward confidence.
- NaN or Inf results are zeroed. After resampling, negative, NaN or Inf `W` is set to 0 (`SpatialReuse.cs.slang:534-535`, `TemporalReuse.cs.slang:298`).

### 1.7 Temporal reuse (`TemporalReuse.cs.slang`, `TemporalPathRetrace.cs.slang`)

#### 1.7.1 Reprojection (`:119-133`)

`prevPixel = int2(pixel + mv*frameDim + u)`, where `u ∈ [0,1)²` from the resampling RNG. Truncation turns this into **stochastic rounding** of the reprojected position.
- If the reprojected pixel is off-screen, or the **previous-frame vbuffer** (`temporalVBuffer`) has no hit there, there is no temporal reuse.
- **There is no normal or depth similarity test**. Dissimilar surfaces are handled by shift failure or low `p̂`.
- The previous-frame shading at the history pixel is loaded from the **previous vbuffer hit** with the previous camera ray direction, evaluated with the **current** scene data (`getPixelTemporalShadingData`, `:67-78`).

#### 1.7.2 M-capping (`:139`)

`M_prev ← min(temporalHistoryLength · M_cur, M_prev)`, with `temporalHistoryLength = 20`. `useMaxHistory=false` sets the cap to 1e30.

#### 1.7.3 Talbot GRIS MIS for 2 inputs (`:146-226`; forced when hybrid)

With `c` the current candidate (canonical, identity shift), `t` the history, `y_c` the current sample, `y_t = T_{t→c}(x_t)`, `J_t = |∂y_t/∂x_t|`, `J_{c→t}` the Jacobian of shifting `y_c` into the previous pixel, and `p̂_c(·)`, `p̂_t(·)` = luma of the integrand in the respective domain:

```
m_c = M_c p̂_c(y_c) / ( M_c p̂_c(y_c) + M_t p̂_t(T_{c→t}(y_c)) J_{c→t} )
m_t = (M_t p̂_t(x_t)/J_t) / ( M_c p̂_c(y_t) + M_t p̂_t(x_t)/J_t )     // p̂_t(x_t) = luma(temporalReservoir.F), stored last frame
w_c = m_c p̂_c(y_c) W_c                      w_t = m_t p̂_c(y_t) J_t W_t
select ∝ w;  M = M_c + M_t;  W = (w_c + w_t) / p̂_c(y_sel)                    (finalizeGRIS)
```

This is GRIS Eq. (36) (generalized Talbot MIS) with confidence weights. Code mapping:
- `p_self`/`p_sum` at `:184-209`;
- `mergeReservoirWithResamplingMIS` computes `w = luma(F)·J·W·m` (`PathReservoir.slang:356-388`);
- `finalizeGRIS` computes `W = w_sum/luma(F)`.

The **other temporal MIS kinds** apply only to non-hybrid shifts:
- `Constant` (`:230-296`): select with `w = luma(F)·J·M·W`, then compute a balance-heuristic contribution MIS weight for the chosen sample only (the ReSTIR DI "pi/piSum" style);
- `ConstantBinary`: 1/|Z|;
- `ConstantBiased`: 1/M;
- `noResamplingForTemporalReuse`: merge the temporal reservoir without shifting (biased).

#### 1.7.4 Scene changes

- `setScene` enables `rejectShiftBasedOnJacobian` and `temporalUpdateForDynamicScene` automatically when the scene is animated (`ReSTIRPTPass.cpp:649-652`).
- **`traceTemporalUpdate`** (`PathTracer.slang:1730-1841`) refreshes the *cached suffix* of the temporal sample in the current frame before shifting:
  - If the rc vertex is NEE-final: redo light sampling at `x_k` with `SampleGenerator(rcRandomSeed)`, which yields the same light and uv under a static sampler, plus a shadow ray. Rewrite `rcVertexWi`, `rcVertexIrradiance = Li·pdf`, `lightType`, `lightPdf`.
  - Otherwise: re-trace from `x_k` along the stored `rcVertexWi` with the suffix stream `rcRandomSeed` (replay mode). Rewrite `rcVertexIrradiance`, `lightPdf`, `lightType`.
  - **This is how moving lights and objects avoid stale cached radiance.**
- History is dropped (`mReservoirFrameCount=0`) on option changes. There is no explicit handling of instance motion for the source path: the prev-vbuffer triangle is evaluated with **current** transforms. **[INFERENCE]** This makes temporal reuse in dynamic scenes an approximation. The history's `W` was computed against the previous frame's integrand.

#### 1.7.5 Output

`outputReservoirs[p] = result`. If this is the last round (spatial off): `outputColor += (F·W + directLighting)/spp`.

### 1.8 Spatial reuse (`SpatialReuse.cs.slang`, `SpatialPathRetrace.cs.slang`)

**Neighbor pattern (`Default`).**
- `neighborOffsets` is an 8192-entry `RG8Snorm` 1D texture. It holds the R2 low-discrepancy sequence (plastic constant `1/1.3247179572447`), rejection-sampled to the unit disk and scaled by 254 (`ReSTIRPTPass.cpp:804-826`).
- Neighbor `i` = `pixel + int2(offset[(startIndex+i) & 8191] · gatherRadius)`, with `startIndex` random per pixel, round and frame.
- Defaults: `gNeighborCount = 3`, radius 20 px, 1 round.
- `SmallWindow` pattern: all pixels in a (2r+1)² window excluding self, `r=2`.

**Neighbor rejection** (`featureBasedRejection = true`, `:71-78`): `dot(N_c, N_n) ≥ 0.5` and `|d_c − d_n| < 0.1·d_c`, where d is the distance to the camera. Off-screen or no-hit neighbors are skipped.

**Pairwise MIS** (`:340-411`), the default and forced for hybrid. With `k = gNeighborCount` (the *requested* count), `V` the set of valid neighbors, `v = |V|`, and for neighbor `i`: `y_i = T_{i→c}(x_i)`, `J_i = |∂y_i/∂x_i|`, and `T_{c→i}(y_c)` with Jacobian `J_{c→i}`:

```
m_i = 1/(v+1) · [M_i p̂_i(x_i)/J_i] / ( [M_i p̂_i(x_i)/J_i] + M_c p̂_c(y_i)/k )
m_c = 1/(v+1) · [ 1 + Σ_{i∈V} ( 1 - M_i p̂_i(T_{c→i}(y_c)) J_{c→i} / ( M_i p̂_i(T_{c→i}(y_c)) J_{c→i} + M_c p̂_c(y_c)/k ) ) ]
w_i = m_i p̂_c(y_i) J_i W_i ;  w_c = m_c p̂_c(y_c) W_c
M = M_c + Σ_{i∈V} M_i ;  W = Σw / p̂_c(y_sel)
```

- The code accumulates `canonicalWeight` and `neighborWeight` **without** the `1/(v+1)` factor and divides `W` by `(v+1)` after `finalizeGRIS` (`:405-409`). This is equivalent.
- This is GRIS Eq. (38), the defensive pairwise MIS with |R|=1, generalized with confidences. **[INFERENCE]** The weights still sum to 1 for any constant used in place of `k`, so using `k` rather than `v` inside does not bias.
- Per neighbor, the shifts needed are `T_{i→c}(x_i)` (via `shiftAndMergeReservoir(..., forceMerge=true)`, which writes the shifted `F` and `J`) and `T_{c→i}(y_c)` (via `computeShiftedIntegrand` into the neighbor). That is **2 shifts per neighbor**, with prefixes precomputed in `SpatialPathRetrace`.

**Talbot spatial** (`:260-336`) is O((k+1)²) shifts and is not supported with hybrid (no retrace data). **Constant spatial** (`:415-531`) selects and then computes a single-sample contribution MIS.

**Output.** `temporalReservoirs[p] = result` (becomes next frame's history). On the last round: `outputColor += (F·W + directLighting)/spp`. There is also optional NRD demodulation output.

### 1.9 Final image composition (default settings)

`outputColor = LDeltaDirect (trace pass) + F·W (last reuse pass) + directLighting (ReSTIR DI incl. primary emission)`, all divided by spp. Background pixels come from `GeneratePaths`. An `AccumulatePass` optionally averages frames.

### 1.10 Parameter defaults (`ReSTIRPTPass.h:74-190`, `Params.slang:133-153`, `.cpp:202, 649-652, 688-694`)

| Parameter | Default | Notes / UI range |
|---|---|---|
| samplesPerPixel | 1 | 1–64 (independent ReSTIR chains) |
| candidateSamples | 1 | 1–64 |
| maxSurfaceBounces | 9 | diffuse/specular/transmission default to the same; clamp `kMaxBounces=14`; path length ≤ 15 (4-bit field) |
| useBSDFSampling / useNEE / useMIS | true / true / true | MIS heuristic: Balance (PowerTwo/PowerExp available) |
| useRussianRoulette | **false** | |
| useAlphaTest | true | |
| emissiveSampler | Power | Uniform / LightBVH available (LightBVH suspected replay-unsafe) |
| separatePathBSDF ("Use Sampled BSDFs") | true | lobe-extended path space |
| useDeterministicBSDF | true | |
| disableDirectIllumination | **true** | DI from ReSTIR DI |
| shiftStrategy | Hybrid | forces spatial=Pairwise, temporal=Talbot |
| localStrategyType | Roughness \| Distance (=3) | |
| specularRoughnessThreshold | 0.2 | linear roughness |
| nearFieldDistance | 0.1 (scene units) | "TODO: make adaptive" |
| rejectShiftBasedOnJacobian / threshold | false (true if animated) / 10 | reject if max(J,1/J) > 11 |
| temporalUpdateForDynamicScene | false (true if animated) | |
| enableTemporalReuse / enableSpatialReuse | true / true | |
| temporalHistoryLength (M-cap factor) | 20 | 0–100; `useMaxHistory` true |
| enableTemporalReprojection | true (member init) | `Init()` sets false. [UNVERIFIED] which applies at runtime |
| featureBasedRejection | true | normal ≥ 0.5, depth within 10% |
| spatialNeighborCount / spatialReuseRadius / numSpatialRounds | 3 / 20 px / 1 | 0–6 / 0–100 / 1–5 |
| spatialReusePattern / smallWindowRadius | Default / 2 | |
| kNeighborOffsetCount | 8192 | |
| useDirectLighting | true | adds the `directLighting` input |
| Offline preset (README) | temporal off, 32 candidates, 3 rounds, 6 neighbors, radius 10 | |

### 1.11 Quirks, approximations and potential issues (all [INFERENCE] unless noted)

1. **Escaped-emitter rc vertex MIS uses a stale light pdf.**
   - Code: `Shift.slang:518-522` uses `srcReservoir.lightPdf`, the solid-angle light pdf measured from the *source* `x_{k-1}`, to MIS-weight the shifted path from `y_{k-1}`.
   - The solid-angle pdf changes with `y_{k-1}`, so `F(y)` is not the destination's true MIS-weighted integrand and the NEE/BSDF partition of unity breaks slightly.
   - EvanLuo42's implementation re-evaluates it (`evalStoredEmissivePdf` in `HybridShift.slang:291-300`).
   - **Port: recompute.**
2. **`useMIS=false` path.** When the rc vertex is final with a BSDF-sampled emitter hit and MIS is off, the `p2^y/p2^x` factor is not applied to `J` (`Shift.slang:525-542`). The default has MIS on.
3. **Temporal-only configuration.** If temporal reprojection fails and spatial reuse is off, `TemporalReuse` returns before adding `F·W` to `outputColor` (`:129,133`), so the pixel loses its indirect light that frame. The default has spatial reuse on.
4. **Dynamic geometry.** Source-domain quantities for temporal reuse are evaluated with current transforms. The temporal update only refreshes the suffix.
5. **Stochastic rounding** of the reprojected pixel, and no depth or normal test in the temporal pass.
6. **Direct lighting is split out.** ReSTIR DI does not handle transmission or delta primaries, so those go into `LDeltaDirect`, which is plain path tracing. For Cycles validation, the DI and PT pieces must be summed exactly. The Enhanced paper's unified DI/GI (§6.1) removes this complexity.
7. **Memory at 1920×1080** (2,073,600 px): reservoir 88 B → ≈182 MB per buffer ×2, plus reconnection data 256 B → ≈531 MB. That is ≈900 MB. This is significant in a browser and needs raised binding limits (§3.3).

### 1.12 WGSL pseudocode (Falcor semantics, condensed)

```wgsl
// ---- reservoir (logical); see §4.2 for packed storage ----
struct Reservoir {
  M: f32, W: f32,            // W is w_sum while streaming
  flags: u32, rcSeed: u32, initSeed: u32,
  F: vec3f, lightPdf: f32,
  cachedJ: vec3f,            // (p1_src, p2_src, G_src)
  rcInst: u32, rcPrim: u32, rcBary: vec2f,
  rcWi: vec3f, rcL: vec3f,
};

fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }

// streaming RIS inside the path tracer (resampling RNG is SEPARATE from path RNG)
fn risAdd(r: ptr<function, Reservoir>, cand: Reservoir, rrPdf: f32, u: f32) -> bool {
  let w = luma(cand.F) / rrPdf;
  if (!(w > 0.0)) { return false; }            // also rejects NaN
  (*r).W += w;
  if (u * (*r).W <= w) { let wsum = (*r).W; *r = cand; (*r).W = wsum; return true; }
  return false;
}
fn finalizeRIS(r: ptr<function, Reservoir>) {
  let ph = luma((*r).F);
  (*r).W = select(0.0, (*r).W / (ph * (*r).M), ph > 0.0 && (*r).M > 0.0);
}

// GRIS merge of a shifted neighbor (m = MIS weight, J = |dy/dx| PSS)
fn grisMerge(dst: ptr<function, Reservoir>, shifted: Reservoir, Fy: vec3f, J: f32,
             Wsrc: f32, Msrc: f32, m: f32, u: f32, wsum: ptr<function, f32>) -> bool {
  (*dst).M += Msrc;                             // counts even if the shift failed
  let w = luma(Fy) * J * Wsrc * m;
  if (!(w > 0.0)) { return false; }
  *wsum += w;
  if (u * *wsum <= w) { let M = (*dst).M; *dst = shifted; (*dst).F = Fy; (*dst).M = M; return true; }
  return false;
}
// finalizeGRIS: W = wsum / luma(F_selected)
```

---

## 2. Other public implementations

### 2.1 Survey

| Repo | Lang / API | License | Status | ReSTIR PT specifics |
|---|---|---|---|---|
| `DQLin/ReSTIR_PT` | Slang / Falcor 4.4 / DXR 1.1 inline | BSD-3 (repo `LICENSE.md`) | reference, 2022 | §1 |
| `NVIDIA-RTX/RTXDI` + `RTXDI-Library` 3.1 | HLSL / Donut / DXR or VK | **NVIDIA RTX SDK license (proprietary)** | production | Hybrid shift, footprint criteria, duplication map, forced NEE reconnect, CGNS neighbor selection, DLSS-RR decorrelation |
| `EvanLuo42/ReSTIR-PT-Enhanced` | Slang / Falcor (7/8-era API) | Falcor BSD-3 | unofficial Enhanced implementation, Aug 2026 | Pairing textures, dual footprint, duplication map, replay compaction queues, dual MV, vector weights, unified DI/GI, presampled light tiles |
| `Domenicobrz/C2-Renderer` | **WGSL / WebGPU**, TS/Svelte | **MIT** | working, progressive | Full-path replay from `firstVertexSeed`; pairwise / GBH / biased MIS; tiled dispatch |
| `TomClabault/HIPRT-Path-Tracer` | HIP / HIPRT | **GPL-3.0** | active 2026 | ReSTIR PT, "ReSTIR PT Enhanced fused single reservoir DI-GI", many MIS variants incl. stochastic pairwise (Hedstrom 2026). Reservoir comment says rc vertex = x1/G-buffer, i.e. reconnection-style [INFERENCE from `Reservoir.h`] |
| `HummaWhite/Vulkan-ReSTIR-PT` | C++ / Vulkan | – | README: "Not all correct"; hybrid only, MIS unchecked | – |
| `elite-sheep/gradient-restir` | Slang / Falcor | Falcor | EG 2026 paper code | Gradient-domain ReSTIR PT; `shiftMappingType` 0/1/2 (reconnection/replay/hybrid) |
| `Shmaug/UnityReSTIR` | HLSL / Unity | – | small | "ReSTIR PT implemented in Unity" (not inspected) |
| `NVIDIA-RTX/RTXPT` | HLSL | NVIDIA | production | ReSTIR DI + ReSTIR GI through RTXDI; **no ReSTIR PT** (tree has only `Rtxpt/RTXDI/*`) |

### 2.2 RTXDI 3.x ReSTIR PT (`RTXDI-Library/Include/Rtxdi/PT/*`, docs `RTXDI/Doc/RestirPT.md`, `ShaderAPI-RestirPT.md`)

**License caution.** `LicenseRef-NvidiaProprietary` / "NVIDIA RTX SDKs LICENSE" prohibits creating or distributing derivative works of the SDK except as object code in an application. WGSL is delivered as source. **Use RTXDI as a design reference only.** Below I describe its algorithm in my own words.

#### 2.2.1 Reservoir (`Reservoir.hlsli`)

The packed form is 64 B: 4×uint4. Logical fields:
- `translatedWorldPosition` (rc vertex position) and `worldNormal` (snorm 3×16);
- `radiance`: the cached suffix radiance, *including* the BSDF at the rc vertex for the original incoming direction;
- `weightSum` (w_sum → UCW) and `M` (f16);
- `age` (5 bits) and `auxFlag`;
- `rcWiPdf`, the BSDF pdf at the rc vertex, used for the invertibility footprint test;
- `partialJacobian`, stored as `d²/|cos|`, the inverse G;
- `rcVertexLength`, `pathLength`, `randomIndex` (8 bits each);
- `randomSeed`;
- `targetFunction` (float3).

**NEE-light rc vertex.** `radiance.x = +Inf` marks it; `radiance.yz` then hold the packed light index and uv (`RTXDI_ConnectsToNeeLight`). **The light is re-sampled from its ID and uv at shift time**, so moving lights are handled without re-tracing.

Buffer layout: 4×4 pixel tiles (`ComputePTReservoirAddress`). Slots ping-pong, plus a preserved initial-sample slot.

#### 2.2.2 Target function

Mixed PSS / solid-angle measure. From the comment in `InitialSamplingPathTracerContext.hlsli:351-358`:
- the bounce `x_{k-2}→x_{k-1}→x_k` is in solid-angle measure, so only `f` is used, not `f/p`;
- all other bounces are in PSS (`f/p`);
- Russian roulette and the pre-rc BSDF pdf are excluded from `p̂`.

Consequently the reconnection Jacobian is purely geometric (`newCos/newD² · storedD²/storedCos`). The pdf ratios of Eq. (2) are absorbed into the target. **[INFERENCE]** This is equivalent to Falcor's PSS formulation up to how `p̂` is defined. `p̂` is a free choice in GRIS, but the integrand used for shading must stay the true integrand.

#### 2.2.3 No BSDF re-evaluation at the rc vertex

`radiance` bakes in `f(x_k; ω_in_orig → ω_k)·cos/p`. The shift multiplies by the BSDF at `y_{k-1}` toward `x_k` (`RAB_GetPTSampleTargetPdfForSurface`) but **does not** re-evaluate `f` at `x_k` for the new incoming direction. The comment in `HybridShift.hlsli:218-220` explicitly says they assume the rc-vertex pdf unchanged "(we don't store the material reference anyway)".

**[INFERENCE]** This is exact for Lambertian rc vertices and approximate (biased) for glossy ones. The footprint criteria are designed to make it small. **For a Cycles-validated renderer, prefer Falcor's approach:** store the triangle and barycentrics and re-evaluate `f` at `x_k`.

#### 2.2.4 Reconnection criteria (`PathReconnectibility.hlsli`, `InitialSamplingPathTracerContext.hlsli:60-121`)

- `rayFootprint = 1/(G·pdf)`, with `G = |cos|/d²` at the hit (0 if the lobe is delta).
- "far" = forward footprint of the ray arriving at the vertex > threshold.
- "rough for connection" = inverse footprint `1/(G_inv·pdf_at)` > threshold, using the previous vertex normal.
- "last vertex rough" = `outPdf ≤ pdfThreshold` and not delta.
- `footprintThreshold = primaryFootprint · c²`, with `primaryFootprint = |x1 − cam|²·4π/|n·v|`.
- `c ~ GaussRand(0.02, rel σ 0.2)` and `pdfThreshold = 1/α²` with `α ~ GaussRand(0.1, rel σ 0.01)`. The jitter is drawn from the **replay** RNG (4 randoms), so it is consistent between base and shift.
- **Difference from the paper [INFERENCE].** The Enhanced paper, Eq. (5), uses threshold `(c/100)·primaryFootprint` with c = 0.02, i.e. 2e-4. RTXDI's `c²` gives 4e-4. `RTXDI_GetRngForShading` advances the RNG index by +4 to skip these jitter draws.
- A fixed-threshold mode (roughness 0.1, distance 0) is also available.

#### 2.2.5 Hybrid shift (`HybridShift.hlsli`)

- `NeedToRunRandomReplayPathTracer` = `rcVertexLength > 2`. RTXDI counts from the camera: x1 = 1, so rcVertexLength 2 means x2.
- Replay reuses the application's own `RAB_PathTrace` in a "hybrid" context (`HybridShiftPathTracerContext.hlsli`). That context:
  - disables NEE and RR;
  - checks that no earlier vertex pair becomes connectible, otherwise the shift is invalid;
  - stops at `rcVertexLength−1` or `pathLength`.
- Invertibility, footprint mode (`ValidateInvertibilityCondition`):
  - forward footprint `1/(G·pdf_{y_{k-1}})` > threshold;
  - `pdf_{y_{k-1}} ≤ pdfThreshold` for mid-path reconnections;
  - inverse footprint with the *stored* `rcWiPdf` > threshold;
  - a deferred check that the pre-rc vertex is *not* connectible (`checkPreRcInverseGeoTerm`).
- **Forced NEE-light reconnection:**
  - if no earlier rc vertex exists, the NEE-sampled light vertex is always the rc vertex (never replay light sampling), matching Enhanced §6.2.3;
  - the partial Jacobian stores the light's solid-angle pdf, and `J = pdf_src/pdf_dst`;
  - MIS for NEE is re-evaluated through `RAB_GetMISWeightForNEE`.

#### 2.2.6 Resampling MIS: "MIS-like" normalization, not pairwise

Temporal (`TemporalResampling.hlsli`):
- Combine current and shifted temporal with `w = luma(p̂)·W·M·J`.
- Then compute `pi/piSum`, where `piSum = p̂_c(y)·M_c + Σ p̂_{←j}(y)·J·M_j` over the **selected** sample only, via an inverse shift (`BiasCorrection`).
- `W = w_sum·pi/(p̂(y)·piSum)`.

This is GRIS generalized balance heuristic "contribution MIS" (1-sample). It costs one inverse shift per neighbor, like pairwise. Spatial (`SpatialResampling.hlsli`) uses the same scheme.

Neighbor search:
- temporal: reprojection, then 9 ring samples of radius 1, then a fallback (zero motion), with normal/depth/material similarity;
- spatial: disk radius 32 with rejection, or CGNS.

History and boosting:
- `maxHistoryLength` 8 (20 with DLSS-RR), `maxReservoirAge` 30 (age-based rejection);
- disocclusion boost: 8 spatial samples when the temporal neighbor was not found.

#### 2.2.7 Duplication map (`DuplicationMap.hlsli` + `TemporalResampling.hlsli:405-415`)

- Sample ID = `randomSeed` (0 if empty).
- A 16×16 group loads a 32×32 LDS tile and counts equal IDs in a 17×17 window, excluding self. It stores `saturate(count/255)`.
- In the temporal pass: `impoverishment = saturate(dupCount/288)`, `power = 0.1·2^(6(1−strength)−3)`. With the default strength 0.5 → 0.1, which is the paper's α=0.1.
- `Mcap = max(1, lerp(maxHistory, 1, impoverishment^power))`, matching Enhanced §5.
- Smoothed stagnancy (age/40, 5×5 à-trous on the reprojected previous frame, then an EMA) drives DLSS-RR decorrelation.

#### 2.2.8 RNG (`RandomSamplerState.hlsli`)

Counter-based: `(seed, index)` → murmur3 → a float in [0,1) from 23 mantissa bits. Replay needs only `(seed, index)`, and skipping N draws is `index += N`. **This is ideal for WGSL.**

#### 2.2.9 Sample pass order (`ReSTIRPTRenderPasses.cpp`)

GenerateInitialSamples → [SpatialNeighborSelection] → Temporal → Spatial (or Spatial → Temporal) → [DuplicationMap, SmoothedDuplicationMap] → FinalShading.

### 2.3 `EvanLuo42/ReSTIR-PT-Enhanced`: the closest template for the Enhanced paper

This is unofficial. Its features were checked against the paper text and equations.

**Pass list** (`ReSTIRPTPass.cpp:681-994`):
1. ResetFrame (atomic counters)
2. PresampleLights (128 tiles × 1024 lights, Enhanced §6.1)
3. BuildPairing (per slot)
4. BuildDualMotion
5. InitialSample
6. TemporalShift → BuildDispatchArgs → **TemporalReplay (indirect)** → TemporalReuse
7. SpatialShift (per slot) → BuildDispatchArgs → **SpatialReplay (indirect)** → SpatialResample (writes the history buffer and output color); or Resolve if spatial is off
8. DuplicationMap
9. Copy the vbuffer and depth to history

**Reciprocal pairing (Enhanced §3, Eq. 3).**
- Host code generates 3 tileable pairing textures, 254/230/210 px.
- Link indices `i/2` are shuffled with `n_σ = ⌊σ²/2 + 1.46/σ + 1.76/σ² + 0.656/σ³ + 0.5⌋` rounds of 2×2 block shuffles, alternating a diagonal offset. With σ = 16, the partner deltas are wrapped to ±size/2.
- Each frame, every set gets a random flip, mirror, transpose and offset.
- The shader resolves the partner and **verifies reciprocity** (`partner(partner(p)) == p`, `Pairing.slang:45-67`).
- **Only the lower-index pixel of a pair computes both shifts** (A→B, B→A) and stores them in `spatialShifts[slot·N + pixel]` (`SpatialShift.cs.slang:36-115`). This halves the spatial shift work, as the paper claims.

**Pairwise MIS over pairs** (`SpatialResample.cs.slang`, `GRIS.slang:28-34`).
- `pairwise(cI, pI, cJ, pJshifted, J) = cI·pI / (cI·pI + cJ·pJshifted·J)`.
- Canonical `m_c = (1/n) Σ_pairs pairwise(...)`, with 1 per failed pair. Neighbor `m_n = pairwise(cN, p̂_toward, cCur, p̂_n(x_n), 1/J)/n`.
- Confidence `c = max(confidence, 1)`.
- **[INFERENCE]** This variant is *not* the defensive form. The canonical term lacks the `1/(k+1)` "self" share that Falcor has.

**Probable bug in this repo's neighbor and history MIS weight [INFERENCE].**
- The shift record `toward = spatialShifts[slot·N + partner]` holds the partner's sample shifted *into the current pixel*. So `toward.target = p̂_c(y_n)`, and `neighbor.sample.target = p̂_n(x_n)`.
- The neighbor weight is computed as `pairwise(cN, p̂_c(y_n), cCur, p̂_n(x_n), 1/J)` = `cN·p̂_c(y_n) / (cN·p̂_c(y_n) + cCur·p̂_n(x_n)/J)` (`SpatialResample.cs.slang:115-121`). The history weight in `TemporalReuse.cs.slang:69-75` has the same structure.
- The correct pairwise weight (GRIS Eq. 37/38 structure; Falcor `SpatialReuse.cs.slang:395-396`) is `cN·p̂_←n(y_n) / (cN·p̂_←n(y_n) + cCur·p̂_c(y_n))`, with `p̂_←n(y_n) = p̂_n(x_n)/J_{n→c}`. That equals `pairwise(cN, p̂_n(x_n), cCur, p̂_c(y_n), J_{n→c})`.
- In other words, `pI` and `pJShifted` appear swapped. The canonical term is correct.
- With equal targets the two forms coincide. Otherwise the per-pair weights no longer sum to 1 (example: `p̂_c(y) ≪ p̂_←n(y)` makes both weights → 0, so energy is lost).
- **Use this repo for its structure (passes, pairing, queues, footprint gate, duplication map) and re-derive every MIS formula yourself.**

**Dual footprint gate** (`EnhancedPolicy.slang:117-149`), matching the paper's Eq. (5):
- `min(1/(p_from·G_fwd), 1/(p_at·G_inv)) ≥ footprintScale·primaryFootprint`, with `footprintScale = 2e-4` = c/100 and `primaryFootprint = d²/(|cos|/4π)`.
- The inverse footprint is skipped for diffuse `x_k` (paper footnote 6).
- Plus a single-vertex roughness `α_{x_{k-1}} ≥ minReconnectionRoughness = 0.2`, where α = 1 for diffuse, 0 for delta, and the material α for glossy (`sampledLobeAlpha`).

**Lobe-consistent replay.** Per-vertex sampled lobes are packed 8 bits × 8 vertices. Replay fails if a sampled lobe differs from the stored one (`HybridShift.slang:62-64`). A reconnection requires `hasExactSingleLobe` at `y_{k-1}` and `x_k`. **[INFERENCE]** This is stricter than Falcor's two-group model and matches GRIS §7.6 with actual lobe indices.

**Reconnection Jacobian.**
- Stores `reconnectDensity = p_from·G_fwd·p_at`. `J = newDensity/oldDensity` (`HybridShift.slang:33-37, 364-366`), which is Eq. (2).
- For NEE endpoint reconnections, `J = 1`: the contribution is re-evaluated with the light's area→solid-angle pdf at the new prefix.
- For BSDF-hit emissive endpoints, the **light pdf is recomputed at the shifted prefix** (`evalStoredEmissivePdf`).
- `mapWorldDirBetweenFrames` and `loadPreviousShading` handle `GeometryTime::Previous` for animated instances.

**Duplication map** (`Frame/DuplicationMap.cs.slang`, `EnhancedPolicy.slang:239-246`).
- Birth identity = (seed.xy, candidateID, techniqueKey+valid bit).
- Radius 8 → 17×17 window, normalized by 288.
- `c_Cap = lerp(cDefault=20, cMin=1, D^0.1)`, which is exactly Enhanced §5 steps 1–3.
- The LDS hash-bucket implementation uses about 28 KB of workgroup memory, which is **above the WebGPU default 16 KB**.

**Dual motion vectors (Enhanced §6.4 / Zeng et al. 2021)** (`BuildDualMotion.cs.slang`).
- Unproject the previous frame's depth at the same pixel with `prevInvViewProj`, then reproject with the current viewProj.
- The resulting camera-only motion serves as the fallback history pixel when the primary reprojection fails the compatibility test.

**Vector weights (Enhanced §6.3).** The pixel color is `Σ_i m_i F(y_i) W_i J_i` over all candidates, while the stored reservoir is still chosen by scalar GRIS (`SpatialResample.cs.slang:82-139`).

**Replay compaction (Enhanced §6.2.2).**
- The shift passes do reconnection-only shifts inline.
- Pairs needing replay (`rcVertex > 2` or none) are appended to a queue with `InterlockedAdd`.
- `BuildDispatchArgs` writes indirect arguments, and the replay pass processes only those pairs.

**Unified DI/GI (Enhanced §6.1).**
- `TechniqueKind::{CameraHit, NEE}` with a path-length key.
- NEE RIS candidates per bounce: `max(1, 32/b²)`. Visibility is excluded from the target and tested only for the selected light.

**Defaults** (`Params.slang:77-117`): `maxPathLength 6`, `pairingSlotCount 3`, `cDefault 20`, `cMin 1`, `duplicationExponent 0.1`, `footprintScale 2e-4`, `minReconnectionRoughness 0.2`, `spatialNormalThreshold 0.9`, depth thresholds 0.1, `specularRoughnessThreshold 0.25`, RR start at bounce 2 with minimum survival 0.05, max bounces 8.

### 2.4 C2-Renderer: a WebGPU/WGSL ReSTIR PT (MIT)

Files: `src/lib/shaders/integrators/ReSTIR-PT/*`, docs in `src/docs/integrators/ReSTIR-PT/*`.

**Architecture.** One WGSL compute shader (`ReSTIRPTShader.ts`) is invoked repeatedly. `passInfo` selects the mode:
- the initial-candidate pass, run `RESTIR_INITIAL_CANDIDATES` times and merged through `combineReservoirs`;
- the temporal pass, same pixel only, so the camera is static;
- spatial passes.

Buffers are `array<Reservoir>` in/out. The dispatch is **tiled** (a `tile` uniform), to keep each submission short and avoid GPU watchdog resets. Workgroup size is 8×8.

**No G-buffer primary.**
- Every replay starts from the camera, using the pixel's `firstVertexSeed` to reproduce the same primary ray, and then switches to the path seed after skipping the camera randoms (`resampleLogic.ts:6-18`, `firstVertexSeed.md`).
- The canonical pixel's `firstVertexSeed` is kept through spatial rounds so every shift lands on the same `x1`.
- DOF therefore breaks the method (`about DOF.md`).

**Reservoir (`reservoir.ts`).**
- `PathInfo {F, firstVertexSeed, seed, bounceCount, flags, reconnectionBounce, jacobian: vec2f, reconnectionBarycentrics, reconnectionRadiance, radianceDirection, reconnectionTriangleIndex}`.
- The Jacobian is stored as a pair `(pdf-part, G-part)`, and `J = (Y.j.x/X.j.x)·|Y.j.y/X.j.y|`. For env-map rc vertices, `G = 1` because the infinite distances cancel (`envmapJacobian.md`).
- `p̂ = length(F)`.
- Minimum reconnection segment length 0.15 (`isSegmentTooShortForReconnection`).

**MIS.** Standard GBH, O(M²) replays; pairwise (defensive, with confidence `c`); and biased 1/M.

**WGSL lessons** (explicit in code comments):
- **Compile-time blowup.** Having `randomReplay(...)` at multiple call sites made the Tint → driver compile explode, because the function is inlined each time. Precomputing all replays in one loop with a single call site cut compile time by 65% (`gbhVariants.ts:265-272`). **Port rule: keep exactly one call site each for BVH traversal and path tracing per pipeline.**
- **Fixed random consumption for light sampling** allows NEE skipping (`randoms.md`). BSDF sampling may vary because it always runs in replay.
- **Limits:** `maxStorageBuffersPerShaderStage = 8` and 4 bind groups. Randoms live in a uniform `array<vec4f, 50>`, because uniform arrays need a 16-B stride.
- Texture arrays by size (128/512/1024) stand in for bindless.
- **Debugging methodology** (`debuggingReSTIR.md`):
  - start with 1 spatial candidate, no temporal, 1 initial candidate, and the standard GBH;
  - log `m_i`, `w_i`, `|F(y)|`, `W` per candidate;
  - check that `Σ m_i ≈ 1`;
  - look for invertibility violations, such as a replay escaping at a different bounce than the base path.

### 2.5 Enhanced-paper features × implementations

| Feature (Enhanced paper §) | Falcor 2022 | RTXDI 3.1 | EvanLuo42 | C2 | HIPRT |
|---|---|---|---|---|---|
| Hybrid shift, lobe-aware | yes (2 lobe groups) | yes (roughness or footprint) | yes (exact lobes) | yes | reconnection-style [INFERENCE] |
| Reciprocal / paired spatial reuse (§3, Eq. 3) | no (pairwise, 2 shifts/neighbor) | no (CGNS instead) | **yes** (3 textures, σ=16) | no | stochastic pairwise MIS |
| Dual footprint + single-vertex roughness (§4, Eq. 5) | no (d ≥ 0.1, roughness > 0.2) | yes (c², Gaussian jitter, pdf threshold) | **yes** (c/100, α ≥ 0.2) | no (d ≥ 0.15) | no |
| Duplication map → c_Cap (§5) | no (fixed ×20 cap) | yes (optional, DLSS-RR mode) | **yes** | no | no |
| Unified DI + GI (§6.1) | no (separate ReSTIR DI) | no (DI separate; PT is indirect) | **yes** | yes (single integrator) | "fused single reservoir DI-GI" |
| Presampled light tiles (§6.1) | no | RTXDI DI has them | yes | no | ReGIR instead |
| Replay stream compaction (§6.2.2) | no (retrace passes, 256 B/px) | NvReorderThread (SER) | **yes** (atomic queue + indirect) | no | ? |
| Forced NEE-light reconnection (§6.2.3) | no | **yes** | yes | no | ? |
| RR only at initial sampling (§6.2.4) | yes, implicitly (RR as source pdf, replay skips RR) | yes | yes | ? | ? |
| Vector-weight color (§6.3) | no | no | yes | no | ? |
| Dual motion vectors (§6.4) | no | no (fallback sample instead) | yes | n/a | ? |
| Reservoir size | 88 B | 64 B packed | ~144 B [INFERENCE] | large | ? |

---

## 3. Slang/HLSL idioms that break in WGSL, with WGSL equivalents

### 3.1 Language-level

| Slang/HLSL idiom (where used) | WGSL issue | Recommended WGSL equivalent |
|---|---|---|
| Interfaces and generics (`ISampleGenerator`, `IBxDF`, `SceneRayQuery<kUseAlphaTest>`, `sampleNext1D<S>`) | No generics or interfaces | Monomorphize by hand. One RNG type. BSDF as a `switch` on material type. Alpha-test variant through an `override` constant |
| Member functions, `[mutating]` (`PathReservoir.add/merge`, `ReSTIRPathFlags.insert*`) | No methods | Free functions taking `ptr<function, T>` |
| Default arguments (`evalPdfBSDF(sd, L, allowedFlags=-1, allowDelta=false)`) and overloading (two `evalPdfBSDF`) | Neither exists | Distinct function names; pass every argument explicitly |
| `out` / `inout` parameters | No `out` | Return a struct, or `ptr<function, T>`. Storage pointers as parameters need the `unrestricted_pointer_parameters` language feature (check `navigator.gpu.wgslLanguageFeatures`) |
| Preprocessor `#if BPR`, `#define`-driven static params (`StaticParams.slang`) | No preprocessor | `override` constants (pipeline-overridable) for scalars and branches. Structural variants via TS string templating, as C2 does. `override` cannot size function or private arrays |
| Recursion | Forbidden | Falcor needs none. Keep loops; the BVH uses an explicit stack |
| Replay called from many places (Talbot loops, pairwise) | Inlining explodes compile time on Tint→MSL (C2 measured it) | One call site per pipeline. Split into separate shift, replay and resample pipelines (EvanLuo/Enhanced structure) |
| `asuint`/`asfloat`/`asint` (`PackedPathState`, `computeRayOrigin`) | Different names | `bitcast<u32>(x)`, `bitcast<f32>(u)`, `bitcast<i32>(v)` |
| `f32tof16`/`f16tof32` | – | `pack2x16float` / `unpack2x16float`; `pack2x16unorm` / `pack2x16snorm` for barycentrics and normals |
| `isnan`/`isinf`/`isfinite` (everywhere) | **No builtins**, and `x != x` may be folded under fast-math | Bit test: `(bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u` → Inf or NaN. Treat `!(w > 0.0)` as reject, which is also false for NaN |
| HLSL `select(c, a, b)` / ternary `c ? a : b` | WGSL `select(f, t, cond)` has **reversed order** | Write `select(falseVal, trueVal, cond)` carefully |
| `lerp`, `frac`, `rsqrt`, `rcp`, `mul(M, v)`, `countbits`, `firstbithigh`, `reversebits` | Renamed | `mix`, `fract`, `inverseSqrt`, `1.0/x`, `M * v` (watch row/column-major), `countOneBits`, `firstLeadingBit`, `reverseBits` |
| `uint16_t`, `float16_t` (`PathState`) | No 16-bit integers; f16 needs `enable f16;` and the `shader-f16` feature | Pack into u32; f16 only for storage |
| 64-bit integers (Falcor `SplitMix64`/`Xoshiro` options; `mAccumulatedRayCount`) | **No i64/u64** in WGSL | 32-bit LCG or PCG, or a counter-based hash (§4.3). Host counters in JS |
| `bool` in structured buffers | `bool` is not host-shareable | `u32` flags |
| `uint(-1)` default flag masks | `u32(-1)` is a const-eval error | `0xffffffffu` |
| Static `groupshared` arrays (`DuplicationMap`) | `var<workgroup>`; size ≤ `maxComputeWorkgroupStorageSize` (default **16384 B**) | Request a higher limit (Apple adapters expose more; query), or restructure. RTXDI's 32×32×4 B = 4 KB tile fits. EvanLuo's ~28 KB hash does not fit the default |
| `GroupMemoryBarrierWithGroupSync` | `workgroupBarrier()` must be in **uniform control flow** (uniformity analysis) | No early `return` before barriers; guard work with `if` after the barrier |
| `InterlockedAdd/Exchange` | Atomics only on `atomic<u32/i32>` in storage or workgroup; no float atomics | `atomicAdd(&counters[i], 1u)`; compaction through atomics + `dispatchWorkgroupsIndirect` |
| Wave intrinsics (boiling filter, reductions) | `subgroups` feature (**shipped in Chrome 134**) | `enable subgroups;` `subgroupAdd`, `subgroupBallot`. Apple subgroup size is 32 [UNVERIFIED for M5] |
| `NvReorderThread` (SER) | None | Stream compaction by category (Enhanced §6.2.2) |
| `[unroll]`, `[loop]` attributes | None | Rely on the compiler; keep loop bounds constant where possible |

### 3.2 Resources and memory layout

| HLSL | WGSL issue | Recommendation |
|---|---|---|
| `StructuredBuffer<PathReservoir>` with `float3` members (88 B, 4-B aligned) | WGSL storage layout: `vec3<f32>` has align 16, size 12. Struct size rounds up to the max alignment. Mixed scalars/vec3 get padded | Store as `array<vec4<u32>>` (or `array<u32>`) and write explicit `pack`/`unpack` functions. Target 64 B = 4×`vec4<u32>` as in RTXDI and the Enhanced paper (§4.2) |
| `ByteAddressBuffer.Load2(addr)` | – | `array<u32>` indexing (`addr/4`) |
| `Texture1D<float2>` neighbor offsets (RG8Snorm) | `texture_1d` is fine but a buffer is simpler | `array<vec2<f32>>` or packed `u32` storage buffer |
| `Texture2D<PackedHitInfo>` (RG32Uint/RGBA32Uint vbuffer) | Fine | `texture_2d<u32>` + `textureLoad`, or a storage buffer |
| `RWTexture2D<float4> outputColor += …` (read-modify-write) | Core read-write storage textures are **only r32float/r32uint/r32sint**. `texture-formats-tier1/2` (Chrome 142) extends this | Accumulate in a storage buffer `array<vec4f>`; blit to the swapchain in a render pass |
| Bindless material textures (`gScene.materialResources`) | No bindless in shipped WebGPU (proposal stage; `texture_and_sampler_let` in Chrome 146 as groundwork) | Texture arrays grouped by size (C2) or a single atlas with per-material UV rect. `textureSampleLevel` with a computed LOD (ray cones) |
| `textureSample` / derivatives in compute | `textureSample` is fragment-only; no `dpdx` in compute | `textureSampleLevel` or `textureSampleGrad` with explicit LOD |
| cbuffer arrays with 4-B stride | Uniform arrays need a 16-B element stride | `array<vec4f, N>` in uniforms, or move to storage |
| Parameter blocks, many resources per pass | Default `maxBindGroups = 4`, `maxStorageBuffersPerShaderStage = 8`, `maxStorageBufferBindingSize = 128 MiB`, `maxBufferSize = 256 MiB` | Request the adapter's limits at device creation (Apple adapters expose much more [UNVERIFIED exact numbers; query `adapter.limits`]). Merge small buffers |
| `executeIndirect` | – | `dispatchWorkgroupsIndirect(buf, offset)` (core) |

### 3.3 Platform notes for Chrome on Apple Metal

- **Compile time.** WGSL is compiled by Tint to MSL, then by the Metal compiler.
  - Keep pipelines small and use `createComputePipelineAsync`.
  - Avoid duplicated inlining of the traversal or path-tracing loop.
  - Pre-warm pipelines at load time.
- **Watchdog.** Very long single dispatches can trigger GPU device loss; C2 tiles its dispatch for this reason. Budget each dispatch to stay well under 1 s. Listen to `device.lost`. [UNVERIFIED: exact macOS/Chrome timeout.]
- **Memory.** 1080p requires raising the binding size limits:

  | Buffer | Size |
  |---|---|
  | Falcor-style reservoir, 88 B | 182 MB per buffer |
  | 64 B reservoir | 133 MB per buffer |
  | Reconnection data, 256 B | 531 MB |

  Recommendations: the Enhanced 64-B reservoir; pair-based shift records instead of 256 B/pixel; optionally render ReSTIR at a lower internal resolution (at 1280×720, 64 B → 59 MB).
- **Timestamp queries** (`timestamp-query` feature) for per-pass profiling. Chrome quantizes them unless flags are enabled.

---

## 4. Recommended WGSL architecture for this project [INFERENCE: design, derived from §1–§3]

### 4.1 Pass list per frame (Enhanced-style, single unified DI+GI reservoir)

| # | Compute pipeline | Notes |
|---|---|---|
| 0 | TLAS/instance transform update (moving lights and objects) | Instance-level BVH refit or rebuild in JS/compute; keep previous transforms for GeometryTime::Previous |
| 1 | Primary visibility (compute BVH or raster V-buffer) | Store `(instanceId, primId, bary)`, depth, normal. Fixed per-pixel primary ray per frame (no jitter, or jitter with the domain defined by that frame's V-buffer) |
| 2 | Motion vectors (+ dual MV) | From current and previous transforms and camera |
| 3 | Initial sampling (path tree + streaming RIS, unified DI with NEE at x1) | One path-tracing call site. RR as external source pdf. Store the reservoir |
| 4 | Temporal shift (reconnection-only inline, enqueue replay pairs) | Write shift records for both directions |
| 5 | Temporal replay (indirect over the compacted queue) | Single call site of the replay loop |
| 6 | Temporal resample (Talbot 2-input or pairwise with the canonical) | `c_Cap` from the duplication map (optional) |
| 7 | Spatial shift (reciprocal pairs, 1–3 slots) + replay (indirect) | Pair textures are precomputed on the host with Eq. (3), plus a random flip/transpose/offset each frame |
| 8 | Spatial resample + shade (vector weights optional) | Writes the history reservoir and the HDR accumulation buffer |
| 9 | Duplication map (optional; **off for bias validation**) | 17×17, /288, fits 16 KB if done like RTXDI (uint IDs, 32×32 tile) |
| 10 | Accumulate / tonemap / present | Progressive accumulation mode for Cycles comparison |

### 4.2 Reservoir packing proposal (64 B = 4 × `vec4<u32>`)

This keeps Falcor semantics, so the rc-vertex BSDF can be re-evaluated for unbiasedness.

```
v0: F.rgb (f32×3)            | W (f32)
v1: rcInst|rcPrim packed? -> rcPrim (u32) | rcInst (u32) | rcBary (unorm2x16) | M (f16) + age/flags (u16)
v2: rcWi (oct-encoded 2x16 snorm -> u32) | rcL.rgb as RGB9E5 or 3×f16 (2 u32) | lightPdf (f32)
v3: cachedJ.x, .y, .z as f16 (2 u32 incl. pad) | initSeed (u32) | pathFlags (u32: pathLen 4b, rcLen 4b, delta/trans/lobe bits, lightType, NEE)
```

- Suffix-replay seed: with a counter-based RNG (§4.3), the suffix stream is `(initSeed, dimension offset of vertex k)`, so `rcRandomSeed` becomes derivable and need not be stored.
- For NEE-final rc vertices, also keep `lightId` + `lightUV` (can reuse the rcL slot, RTXDI-style) to support **moving lights** without the temporal re-trace.

### 4.3 RNG design for replay robustness

- Use a counter-based hash RNG `rand(seed, dim)` (PCG or murmur3, as RTXDI does) instead of Falcor's stateful LCG.
- **Reserve a fixed dimension block per vertex:** `dim = base + vertex*K + j`, for example K = 16. BSDF sampling, NEE, RR, and lobe selection each use fixed sub-offsets.
- Replay can then skip NEE or RR without knowing how many numbers they would have consumed, which removes the class of bugs in §1.4.
- Keep a separate resampling RNG stream `rand(pixelHash ^ frame ^ passId, i)`.
- The spatial-shift and resample passes must derive neighbor choices from the same seed, or read an explicit pair buffer (preferred).

### 4.4 Shift kernel skeleton (hybrid, lobe-aware, Falcor-exact Jacobian)

```wgsl
// Returns F(y) in dst domain and J = |dy/dx| (PSS); J == 0 => failure.
fn shiftHybrid(src: Reservoir, dstPrimary: SurfaceHit, dstPixel: vec2u) -> ShiftOut {
  var o: ShiftOut;
  let k = rcLen(src.flags); let d = pathLen(src.flags);
  var prefix: ReplayState = replayPrefix(src.initSeed, dstPrimary, k, src.flags); // stops at y_{k-1}; checks
  if (prefix.failed) { return o; }                        // invertibility & lobe checks inside
  if (!hasRc(src.flags)) { o.F = prefix.L; o.J = 1.0; return o; }   // pure replay
  let xk = loadSurface(src.rcInst, src.rcPrim, unpackBary(src), prefix.pos);    // V toward y_{k-1}
  let w1 = normalize(xk.pos - prefix.pos);
  if (distance(xk.pos, prefix.pos) < dMin && useDistance) { return o; } // or dual-footprint gate (Eq.5)
  let G  = abs(dot(xk.faceN, -w1)) / dot(xk.pos - prefix.pos, xk.pos - prefix.pos);
  let e1 = evalLobe(prefix.sd, w1, lobeBefore(src.flags));      // f, pdfSingle, pdfAll
  var J  = (e1.pdfSingle / src.cachedJ.x) * (G / src.cachedJ.z);
  var F  = e1.f / e1.pdfSingle;
  if (!isEscapedRc(src.flags)) {
    let e2 = evalLobe(xk.sd /*V=-w1*/, src.rcWi, select(lobeAfter(src.flags), ALL_LOBES, isNEEFinal(src.flags)));
    let p2 = select(e2.pdfSingle, src.lightPdf, isNEEFinal(src.flags));
    F *= e2.f / p2;
    if (!isNEEFinal(src.flags)) { J *= e2.pdfSingle / src.cachedJ.y; }
    if (isFinal(src.flags) && !isDeltaLight(src)) {
      F *= select(balance(e2.pdfAll, src.lightPdf), balance(src.lightPdf, e2.pdfAll), isNEEFinal(src.flags));
    }
  } else {
    F *= balance(e1.pdfAll, lightPdfFrom(prefix.pos, xk));   // RECOMPUTE (fixes Falcor quirk §1.11-1)
  }
  if (!(J > 0.0) || !isFiniteF(J) || !visible(prefix, xk)) { return o; }
  o.F = prefix.thp * F * src.rcL; o.J = J;
  o.newCachedJ = vec3f(e1.pdfSingle, /*e2.pdfSingle or 1*/ 1.0, G); // write back when selected
  return o;
}
```

---

## 5. Implications for Cycles-based validation

1. **Validate the base path tracer first.** Use Falcor's `PathSamplingMode::PathTracing` equivalent, matching Cycles light-path settings:
   - max bounces; Falcor counts surface bounces, and its `pathLength` excludes the light vertex;
   - no caustic filtering, no clamping;
   - the same MIS behavior for area lights. Point and spot lights are delta: NEE only, MIS weight 1.
2. **Then test the initial candidates without reuse** (`F·W` with N candidates). This must match the path tracer in expectation.
3. **Then run spatial-only GRIS with independent frames, averaged** (GRIS §6.5 "ReSTIR for offline rendering"; Falcor README offline preset: 32 candidates, 3 rounds, 6 neighbors, radius 10). This is provably unbiased and converges.
4. **Then test temporal reuse with a static scene and camera.** Averaging the chain's frames converges (GRIS §6 box: "unbiased explorative non-Markovian chain").
5. **Disable known-biased options for these tests:**
   - duplication-map `c_Cap` reduction (the paper reports 3.25% average absolute relative bias, Enhanced §5, Fig. 5);
   - `ConstantBiased` MIS and `noResamplingForTemporalReuse`;
   - RTXDI-style "no rc BSDF re-evaluation";
   - DLSS-RR firefly replacement.

   Jacobian-based shift rejection and footprint or distance gates are unbiased when applied symmetrically (they only restrict the shift domain).
6. **Use a per-candidate debug view** as C2 does: `m_i`, `w_i`, `|F(y)|`, `W`. Check `Σ m_i = 1` numerically for a probe pixel, by evaluating all techniques' `m` for the same `y`.
7. **Color space.** Falcor uses Rec.601 luma for `p̂`. Any positive `p̂` is valid, but keep it consistent across all passes.

---

## 6. Sources

Code (read at the snapshots listed in §0):
- https://github.com/DQLin/ReSTIR_PT: `Source/RenderPasses/ReSTIRPTPass/{Params,StaticParams,PathReservoir,PathState,PathBuilder,PathTracer,Shift,LoadShadingData}.slang`, `{TracePass,GeneratePaths,TemporalPathRetrace,TemporalReuse,SpatialPathRetrace,SpatialReuse,ReflectTypes}.cs.slang`, `ReSTIRPTPass.{h,cpp}`, `Source/Mogwai/Data/ReSTIRPT*.py`, `README.md`, `LICENSE.md`; Falcor utilities `Utils/Sampling/{TinyUniformSampleGenerator,SampleGenerator,SampleGeneratorInterface}.slang`, `Utils/Sampling/Pseudorandom/LCG.slang`, `Utils/Math/HashUtils.slang`, `Rendering/Materials/{MaterialShading,BxDF,IBxDF}.slang`, `Utils/Geometry/GeometryHelpers.slang`, `Scene/HitInfo.{slang,cpp}`, `Rendering/Lights/{EmissiveLightSampler,EmissivePowerSampler}.slang`, `RenderPasses/ScreenSpaceReSTIRPass/FinalShading.cs.slang`
- https://github.com/NVIDIA-RTX/RTXDI: `ChangeLog.md`, `Doc/RestirPT.md`, `Doc/ShaderAPI-RestirPT.md`, `Samples/FullSample/Source/RenderPasses/LightingPasses/ReSTIRPTRenderPasses.cpp`, `LICENSE.txt`
- https://github.com/NVIDIA-RTX/RTXDI-Library: `Include/Rtxdi/PT/{Reservoir,PathReconnectibility,PathTracerContext,InitialSamplingPathTracerContext,HybridShiftPathTracerContext,HybridShift,TemporalResampling,SpatialResampling,DuplicationMap,PathTracerRandomContext}.hlsli`, `ReSTIRPTParameters.h`, `Include/Rtxdi/Utils/{RandomSamplerState,Math}.hlsli`, `Source/ReSTIRPT.cpp`, `LICENSE.txt`
- https://github.com/EvanLuo42/ReSTIR-PT-Enhanced: `Source/RenderPasses/ReSTIRPTPass/{Params.slang, ReSTIRPTPass.cpp, Common/{Types,GRIS,Pairing,EnhancedPolicy,HybridShift}.slang, Path/PathSampling.slang, Spatial/{BuildPairing,SpatialShift,SpatialResample}.cs.slang, Temporal/{TemporalShift,TemporalReuse}.cs.slang, Frame/{DuplicationMap,BuildDualMotion}.cs.slang}`, `README.md`, `LICENSE.md`
- https://github.com/Domenicobrz/C2-Renderer: `src/lib/shaders/integrators/ReSTIR-PT/{ReSTIRPTShader,reservoir,reservoirFunctions,restirRandomPart,resampleLogic,gbhVariants,pathConstruction}.ts`, `src/docs/integrators/ReSTIR-PT/*.md`, `README.md`, `LICENSE`
- https://github.com/TomClabault/HIPRT-Path-Tracer (`README.md`, `src/Device/includes/ReSTIR/PT/Reservoir.h`, `src/HostDeviceCommon/ReSTIR/ReSTIRPTSettings.h`), https://github.com/HummaWhite/Vulkan-ReSTIR-PT (README), https://github.com/elite-sheep/gradient-restir (README), https://github.com/NVIDIA-RTX/RTXPT (tree + README)

Papers (local copies in the scratchpad): Lin, Kettunen, Wyman 2026, *ReSTIR PT Enhanced*, Eqs. (2), (3), (5), §§3–6, pp. 13:4–13:11. Lin et al. 2022, *GRIS*, Eqs. (36)–(38), §§5.6, 6.5, 7.3–7.6, 8, pp. 75:9–75:14.

WebGPU status: https://developer.chrome.com/blog/new-in-webgpu-134 (subgroups shipped), https://developer.chrome.com/blog/new-in-webgpu-146 (`texture_and_sampler_let`), https://developer.chrome.com/blog/new-in-webgpu-142 (texture-formats tiers), https://github.com/gpuweb/gpuweb/blob/main/proposals/bindless.md, https://github.com/gpuweb/gpuweb/discussions/4651 (rgba32float read-write not in core).
