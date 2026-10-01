# M8 research: hardware RT in WebGPU, tinybvh, Gigi, and the Randall path-tracer post

Target: Chrome 154, Apple M5 Pro, Dawn/Metal. Research date 2026-10-01. Read-only: nothing in the repo was edited and no GPU jobs were run.
Tags: **[SRC]** = stated by the cited source; **[CODE]** = read in the cited source file; **[REPO]** = this repo's docs and measurements; **[INF]** = my inference.

## 0. TL;DR

1. **The James Randall post uses no hardware ray tracing.** It is a software BVH2 traversed in WGSL compute shaders, with Möller–Trumbore triangle tests and a CPU-built SAH tree. It runs in stock WebGPU with no flags and no optional features. The only hardware-RT text in the post is commentary on RTX cores. The claim that "we can use hardware BVH" is a misreading.
2. **No hardware ray tracing or ray queries are available to us in Chrome 154 on Metal.** None of these exist: shipped, behind a flag, in an origin trial, or Dawn-native-only. Dawn `main` (2026-10-01) has no ray-tracing feature or WGSL extension. The gpuweb issue is open at "Milestone 4+", and there is no proposal document. A Chrome WebGPU engineer wrote in Oct 2025 that "Dawn has nothing".
3. **Gigi does not use hardware RT on WebGPU either.** It emulates DXR in compute shaders over a tinybvh **BVH_GPU** (BVH2, Aila–Laine) tree. The tree is built **offline** in the native DX12 viewer with `BuildHQ` (SBVH) and shipped as `.bvh` files. Its HLSL traversal (from tinybvh `traverse_bvh2.cl`) goes through Slang to WGSL. This is essentially what we already have.
4. **tinybvh** (MIT, v1.9.0) offers `BVH`, `BVH_GPU`, `BVH4_GPU`, `BVH8_CWBVH`, `BVH4_CPU`/`BVH8_CPU`, `MBVH<M>` and double-precision variants.
   - Builders: binned SAH, SBVH, presplitting and a Bittner optimizer.
   - It now uses an **SAH-optimal dynamic-programming collapse** to the 8-wide tree.
   - It has WASM/Emscripten support, but there is **no WGSL kernel**: GPU kernels exist only in GLSL, HLSL and OpenCL.
   - It can serve as our builder, provided we **discard its triangle payload and re-emit our own Woop `trisW` records keyed by its `primIdx`**.
5. **For M8, keep CWBVH as planned, built by tinybvh WASM in single-threaded mode, with our existing WGSL traversal.** Expect the plan's ~1.2× per frame. Further gains from a better build are plausible but unmeasured.

---

## 1. "Building a Real-Time Path Tracer in WebGPU" (James Randall)

Source: https://www.jamesdrandall.com/posts/building-a-real-time-path-tracer-in-webgpu/ (dated March 2026). Repo: https://github.com/JamesRandall/webgpu-doom-pathtracer (MIT, created 2026-01, last push 2026-03-08).

| Question | Answer |
|---|---|
| Hardware RT / ray query extension? | **No.** The intro says "No hardware ray tracing cores, no ML denoisers, no engine" [SRC]. The only hardware-RT passage is commentary: RT cores achieve "maybe 10× the throughput of our software compute shader" [SRC]. |
| Chrome flag, Dawn experimental feature, proposal? | None. The renderer calls `adapter.requestDevice()` with no `requiredFeatures` or limits [CODE `src/path-tracer.ts`]. The README says "Requires a browser with WebGPU support (Chrome 113+, Edge 113+)" [SRC, repo README]. |
| Backend | Whatever the browser uses. The author reports "On my Mac I get 60fps with these defaults" [SRC], so Dawn/Metal is used in practice [INF]. |
| BVH | Built on the CPU in TS: SAH with 12 buckets, ≤ 4 triangles per leaf, C_trav = 1, C_isect = 2 [CODE `src/bvh/builder.ts`]. The 32 B node is `min,leftOrFirst,max,rightOrCount` with the high bit marking a leaf. Traversal uses an explicit `array<u32,32>` stack and visits the near child first [SRC]. |
| Triangle test | Möller–Trumbore [SRC]. |
| Scene / numbers | About 3,000 triangles (Doom E1M1 and a dungeon) at 1280×800 [SRC]. The "330M rays/sec" figure is the assumed throughput in a back-of-envelope table (1 spp ≈ 12 ms ≈ 80 fps), not a benchmark [SRC]. It is not comparable with our Sponza numbers (262k triangles, ~190 Mrays/s incoherent, BVH2) [REPO platform §4.1]. |
| Architecture | One path-trace compute pass (a megakernel), then temporal accumulation, then spatial denoise, then blit [SRC]. No wavefront, no ray sorting [SRC: not mentioned]. |
| Other | PCG hash RNG; RR after bounce 2; cosine sampling. Triangle data is split into hot data (positions) and cold data (attributes). A separate distance-culled BVH is precomputed per dungeon tile (~130 BVHs) for the torch light [SRC]. |

---

## 2. tinybvh (https://github.com/jbikker/tinybvh)

**Facts.** MIT license. Version 1.9.0 [CODE `tiny_bvh.h` l.108–110]. About 1.2k stars; pushed 2026-10-01 (GitHub API). Header-only C++, with C++17 for threading [SRC README].

**Layouts** [SRC README; CODE `tiny_bvh_base.h` l.674–681]:

| Layout | What it is | Relevance to us |
|---|---|---|
| `BVH` | 32 B Wald node; the build layout | Intermediate |
| `BVH_GPU` | 64 B Aila–Laine node holding both child boxes | **Our current BVH2 layout** (layout.ts says so) |
| `BVH4_GPU` | Compact 4-wide; "may be faster for GPU ray tracing" | Not measured on M5 [INF] |
| `BVH8_CWBVH` | 80 B Ylitie et al. 2017 node with 8-bit quantized child boxes | **The M8 target** |
| `BVH4_CPU`, `BVH8_CPU` | AVX2/NEON CPU traversal layouts. README: "BVH4_CPU is now faster" | CPU only; useless for WGSL |
| `MBVH<M>`, `*_Double`, `BVH_Verbose` | Wide/intermediate, double precision, optimizer | Builder internals |

**Builders** [SRC README; CODE `BVHBuildSettings`]:
- Reference full-sweep SAH; binned SAH (scalar, AVX, NEON).
- SBVH spatial splits with "unsplitting" (`useSpatialSplits`, i.e. `BuildHQ`).
- Presplitting; the Bittner tree-rotation optimizer (`postOptimize`, `Optimize()`).
- TLAS/BLAS; refit; Save/Load.

**How CWBVH is built** [CODE `tiny_bvh_base.h` l.5686–5701, 4838ff, 5760ff]:
1. Build a BVH2, then `Compact`, `CombineLeafs(3)` and `SplitLeafs(3)` (a CWBVH leaf slot holds ≤ 3 triangles).
2. Collapse to 8-wide by **bottom-up dynamic programming over SAH cost**. This is the paper's optimal collapse; our gap-perf port used a greedy largest-area collapse.
3. Assign octant-ordered slots greedily and quantize with floor/ceil.

The triangle payload is controlled by `CWBVH_COMPRESSED_TRIS`, which is **`#define`d unconditionally** with the comment "doesn't seem to help on GPU?" [CODE `tiny_bvh.h` ~l.206]:
- **Compressed** (the default): a Baldwin–Weber 3×vec4 transform plus a 4th vec4 holding `triIdx` in `.w`.
- **Uncompressed:** `(v2−v0, v1−v0, v0|triIdx)`.

**GPU kernels.** GLSL (`external/vulkanrt/shaders/tiny.comp`, `tiny4.comp`, `tiny8.comp`, `tinyrq.comp`), HLSL (`external/dx12rt/*.hlsl`) and OpenCL (`kernels/traverse_bvh2.cl`, `traverse_bvh4.cl`, `traverse_cwbvh.cl`, `traverse_tlas.cl`, `wavefront*.cl`). **No WGSL and no WebGPU sample** (repo tree via the GitHub API).
- `tiny8.comp` uses a **shared-memory, SoA, 8-entry node-group stack**, with the comment "Measured peak: 6 at 450k primitives" [CODE].
- `tiny8.comp` also has a `QUNPACK_BITTRICK` path for unpacking the quantized bytes.

**WASM / threads.**
- README: "Support for WASM / Emscripten" [SRC].
- Threaded builds can be compiled out with `NO_THREADED_BUILDS`. Alternatively, `TINYBVH_NO_BUILTIN_POOL` with null `BVHContext` hooks makes every spawn and parallel-for run serially. Builds under `MT_BUILD_THRESHOLD = 50000` triangles are single-threaded anyway [CODE `tiny_bvh.h` l.125, 203; `tiny_bvh_base.h` l.588–633].
- SIMD under WASM goes through the **x86 path** when `__wasm_simd128__` is defined. That path needs `__SSE4_2__`/`__AVX__`, plus `__AVX2__`+`__FMA__` for the AVX builder [CODE `tiny_bvh.h` l.270–300]. Emscripten provides these through `-msimd128 -msse4.2 -mavx -mavx2 -mfma` (https://emscripten.org/docs/porting/simd.html).
- tinybvh also accepts `__wasm_relaxed_simd__`, which is non-deterministic by design. Avoid it [INF].

**Watertight.** `WATERTIGHT_TRITEST` (Woop) exists only in the **CPU** triangle test [CODE l.6308]. The GPU kernels use Baldwin–Weber or Möller–Trumbore.

**Quality and performance numbers published:**
- README: "up to _5 billion rays_ per second in Crytek's Sponza … using the basic binary BVH format" on an RTX 5080 laptop GPU, without RT hardware [SRC].
- The `cpu_vs_gpu.png` chart shows about 3,000 (nearest-hit) and about 4,100 (any-hit) MRays/s on the GPU (RTX 5080 / Ryzen AI 9 HX 370) [SRC image].
- The repo's DX12/Vulkan benchmarks print each software layout "as % of DXR" [CODE `external/dx12rt/program.cpp` l.665–670], but **no per-layout GPU table is published**.
- The advanced manual is qualitative only: CWBVH is "a fast layout", "relatively costly" to obtain and "mostly useful for static geometry" (https://jacco.ompf2.com/2025/01/25/tinybvh-manual-advanced/).
- The Ylitie et al. 2017 paper, linked from the README, reports 1.9–2.1× on NVIDIA [REPO platform §4.2].

---

## 3. Gigi (https://github.com/electronicarts/gigi)

- **Release:** WebGPU code generation shipped in v1.0.0 (2025-06-17). Its release notes thank tinybvh, "used by Gigi WebGPU for super fast ray tracing". The viewer gained an "option to save vertex information as BVH. Saved in tinybvh format BVH_GPU" (https://github.com/electronicarts/gigi/releases/tag/v1.0.0).
- **Builder (CPU, offline, native):** `GigiViewerDX12/BVH.cpp` calls `tinybvh::BVH_GPU bvh; bvh.BuildHQ(...)` (SBVH) and writes `.vertices.bvh`, `.nodes.bvh`, `.triindices.bvh` and `.combined.bvh` [CODE]. There is **no WASM**: the browser loads these precomputed files (the demo references `.combined.bvh` etc.: https://electronicarts.github.io/gigi/Demos/tinybvh/index.html).
- **Traversal:**
  - `GigiCompilerLib/Backends/WebGPU/templates/Module/BVH.hlsli` is "adapted from tinybvh … traverse_bvh2.cl". It is BVH2 with a 32-entry stack, a `primIdx` indirection to raw vertices, and Möller–Trumbore [CODE].
  - `GigiRayTracingEmulation.hlsli` emulates DXR `TraceRay`/`RayQuery`: raygen runs as compute, and any-hit, closest-hit and miss are called from inside the traversal loop [CODE].
  - HLSL becomes WGSL via **Slang** (Gigi `UserDocumentation/WebGPU.pdf`). The demo's generated WGSL has `RayVsMesh_BVH_*` and `IsOccluded_BVH_*` with `array<u32, 32>` stacks, and `requiredFeatures: []` [CODE demo `tinybvh_Module.js`, `index.html`].
- **Hardware RT:** none. Gigi's own doc says WebGPU "Is missing features like raytracing". It calls BVH emulation "essentially as fast as built in raytracing APIs" (WebGPU.pdf p.2) [SRC]. That speed claim is Gigi's own and comes with no benchmark [INF].
- **Takeaway:** Gigi validates the BVH_GPU + WGSL approach we already use. It does not use CWBVH or BVH4, so it offers nothing new for M8 beyond "SBVH-built BVH2" [INF].

---

## 4. Hardware ray tracing in Chrome 154 WebGPU on Metal

| Channel | Status | Evidence |
|---|---|---|
| Shipped in WebGPU spec / Chrome | **Not available** | The Chrome "What's New in WebGPU" index (113–154) has no ray tracing, ray query or acceleration structure (https://developer.chrome.com/docs/web-platform/webgpu/news). The gpuweb `proposals/` directory has no ray-tracing doc; it does list `bindless.md` (https://github.com/gpuweb/gpuweb/tree/main/proposals) |
| Behind a flag (`chrome://flags`, `--enable-unsafe-webgpu`, `--enable-dawn-features`) | **Not available** | Dawn `main` `src/dawn/dawn.json` "feature name" enum (fetched 2026-10-01) has no ray, acceleration-structure or ray-query entry. The experimental entries are subgroup matrix, timestamp-inside-passes and the sampling resource table. Tint `wgsl.def` `enum extension` has no ray extension, and the WGSL language features (`feature_status.cc`) have none either (https://github.com/google/dawn) |
| Origin trial | **Not available** | No intent or trial found. The Chrome WebGPU news index shows none |
| Dawn-native only (e.g. node bindings) | **Not available in upstream Dawn** | Same `dawn.json` evidence. kainino0x (Chrome WebGPU) wrote on 2025-10-20 that a flag could only exist once an implementation is in the browser, and "wgpu already has some support … Dawn has nothing" (https://github.com/gpuweb/gpuweb/issues/535) |
| Proposal | **Discussion only** | gpuweb #535 "Ray Tracing extension" is open, Milestone 4+, labels `large`, `api`, `wgsl`. Kangz (2024-05-20): "to be useful ray-tracing needs bindless resources". Earlier, kainino0x (2023-11-28) estimated "no less than 1-2 years" (https://github.com/gpuweb/gpuweb/issues/535) |
| Third-party forks | **Not usable** | maierfelix/dawn-ray-tracing covers **Vulkan and D3D12 only**, needs a custom Chromium build, and was last pushed 2020-09 (https://github.com/maierfelix/dawn-ray-tracing, https://github.com/maierfelix/chromium-ray-tracing). WebRTX is a **software** polyfill (https://github.com/codedhead/webrtx). wgpu has `EXPERIMENTAL_RAY_QUERY`, including Metal acceleration structures (wgpu CHANGELOG, PR #8071: https://github.com/gfx-rs/wgpu/blob/trunk/CHANGELOG.md), but that is native Rust and not exposed in any browser |

**Verdict:** hardware ray tracing and ray queries are **not possible on our platform, even behind a flag**. The M5's RT hardware is reachable only through native Metal, outside the browser [INF].

**If it ever shipped** [INF, for planning only]:
- **Headless Playwright Chrome:** works only if headless Chrome exposes the feature on the same Metal adapter. We would need an adapter-feature probe that fails loudly.
- **Woop watertight requirement:** cannot be enforced. The DXR spec relies on a top-left rule for watertightness (https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html), but our canonical-global-vertex-edge-order Woop test is a stronger, bit-specified contract. HW intersection results would differ from T12's brute-force oracle at edges and ties. T12 would have to become tolerance-based, or the software Woop path kept for validation lanes.
- **Determinism:** DXR states "There is no defined order of execution of any hit shaders" (same URL). Visibility stays deterministic only if the MASK test is a pure function of (primId, uv), which ours is. Closest-hit ties could resolve differently from the software path.
- **≤ 10 storage buffers:** an acceleration structure would replace the nodes and tris bindings. wgpu notes Metal shares one buffer argument table across buffers and acceleration structures (CHANGELOG #9709).
- **MASK any-hit:** this needs non-opaque geometry plus a candidate loop in a ray query (DXR `CANDIDATE_NON_OPAQUE_TRIANGLE`; Gigi emulates the same model). It is feasible but costs the HW fast path on alpha-tested triangles.

---

## 5. Software BVH: which layout, and tinybvh as our builder

### 5.1 Expected speed-ups for our ray mix

Measured in this repo (gap-perf §2.7, CWBVH port with greedy collapse, binned-SAH BVH2, leaf ≤ 3, Möller–Trumbore) [REPO]:
- Any-hit segment rays (reconnection visibility, shift, NEE suffix): **1.32–1.57×**.
- Closest-hit bounce rays: **1.05–1.21×**.
- Whole frame: **1.19–1.22×**.
- Hits match BVH2 for 99.99% of rays.

The plan's ~1.2× is this measurement, not an extrapolation.

| Candidate | Expected effect on M5 | Basis |
|---|---|---|
| **CWBVH, tinybvh DP collapse** instead of greedy | Small extra gain, most on incoherent closest-hit rays; plausibly a few % per frame | [INF]. Ylitie's results rely on SAH-optimal collapse, and gap-perf §2.7 lists it as "not tried" |
| **SBVH (`BuildHQ`)** feeding CWBVH | Gains on long, thin, axis-skewed triangles (Sponza has many); unmeasured here | [INF]. tinybvh/Gigi choose it for quality. **Not for Woop lanes; see 5.2** |
| Bittner `Optimize()` | Lower SAH, small traversal gain; slow build | [INF] |
| Compressed tris (Baldwin–Weber) | **Reject.** Not watertight, and tinybvh's own comment doubts the GPU gain | [CODE] |
| tinybvh's 8-entry shared-memory group stack | Probably neutral or negative. On M5 a private stack beat a workgroup stack by 3–6% for BVH2 | [REPO platform §4.1] + [INF] |
| `BVH4_GPU` | Unmeasured. Not worth a third variant, since CWBVH is already measured and wider | [INF] |
| Hardware RT | Not available | §4 |

**Best layout for secondary, shadow and reconnection rays:** CWBVH, built by tinybvh, MT for production and Woop for validation. The realistic frame estimate stays **~1.2×**. Upside to roughly 1.25–1.3× from the DP collapse plus SBVH on non-watertight lanes is **[INF], unmeasured**. The 2× in the Ylitie paper should not be expected on M5 unified memory [REPO gap-perf §2.7].

### 5.2 Using tinybvh WASM as the builder while keeping Woop and a stable primId

**Feasible** [INF unless tagged].

1. **Build.**
   - Compile `tiny_bvh.h` with Emscripten, using `-DNO_THREADED_BUILDS` (or `TINYBVH_NO_BUILTIN_POOL` with null hooks). No SharedArrayBuffer and no COOP/COEP needed.
   - Start with `-DTINYBVH_NO_SIMD` (scalar reference builder) for determinism and simplicity. Later, try `-msimd128 -msse4.2 -mavx -mavx2 -mfma` [CODE + Emscripten docs], measuring build time and SAH cost.
   - Never enable relaxed-SIMD.
   - Run it inside the existing BVH Worker (`src/core/bvh/build-in-worker.ts`).
2. **primId.** tinybvh's `primIdx` is the index into the triangle array we pass in [CODE `Fragment.primIdx`; CWBVH writes `triIdx` into each triangle record]. Pass our flattened, stable-primId triangle order (non-finite and degenerate triangles already dropped) and primId is preserved.
3. **Watertight.**
   - **Do not use tinybvh's triangle payload.** For each CWBVH triangle slot, read `triIdx` from the record's `.w` (record 3 if compressed, record 2 if not). Re-emit our existing `trisW` record `[v0|vid0][v1|vid1][v2|vid2]` plus the primId tail [REPO `src/core/bvh/layout.ts`], **bit-copied from the original vertex buffer**.
   - Never reconstruct vertices as `v0 + e1`: the float add is not bit-identical to the shared vertex, which breaks the canonical-edge watertight guarantee [INF].
   - Rewrite each node's `baseTri` word from tinybvh's vec4 units (stride 4 with compressed tris, 3 without) to our triangle stride. Alternatively, patch the unconditional `#define CWBVH_COMPRESSED_TRIS` out of the vendored header.
4. **SBVH and watertightness.** `ClipFrag` computes edge/plane intersections in float without ulp padding [CODE `tiny_bvh_base.h` l.6463ff]. Clipped child boxes may therefore be slightly non-conservative, which risks rare missed hits [INF]. **Use the plain binned builder for T12, glass and contact lanes.** Allow SBVH only for the production MT variant, and only if a brute-force agreement test (T12-style, 10⁶ rays) passes on it.
5. **Determinism.**
   - Wasm float arithmetic is deterministic except for NaN bit patterns [INF from the WebAssembly design: https://github.com/WebAssembly/design/blob/main/Nondeterminism.md].
   - The scalar, single-threaded build is reproducible per input. Pin the tinybvh commit and add a golden hash of the node and triangle buffers per test scene.
   - To make BVH2 and CWBVH return identical hits on exact-`t` ties, tie-break on the smaller primId in the closest-hit update. That removes the 0.01% coplanar differences gap-perf saw [INF].
6. **Validation counters.** The §10.2 port silently drops pushes on stack overflow. Before CWBVH can enter `validate --all`, add the probe-buffer stack-overflow and iteration-cap counters (plan §1.3: a non-zero count fails validation). Depth is not a concern: one group is pushed per level, and tinybvh measured a peak of 6 at 450k primitives.
7. **MASK cutout.** Unchanged: call `alpha_pass(primId,u,v)` in the CWBVH leaf-triangle loop, as `traverse.wgsl` already does for BVH2.
8. **Bindings.** CWBVH uses the same two buffers (nodes plus tris/trisW) as BVH2, so the ≤ 9/10 storage-buffer plan is unchanged [REPO gap-perf §4].
9. **Fallback.** Keep the verified JS encoder from gap-perf §10.2 and swap only the greedy collapse for a DP collapse. That avoids an Emscripten toolchain dependency at the cost of SBVH and the optimizer.

---

## 6. Other techniques from the path-tracer post that are relevant to us

| Technique in post | Relevance |
|---|---|
| Megakernel PT pass + temporal pass + à-trous pass | Matches our "megakernel per stage" design [REPO PLAN §1.8]. There is **no wavefront and no ray sorting** in the post, so it offers no evidence on those. gap-perf found neighbour distance does not buy coherence for reconnection rays [REPO §2.1]. Sorting would cost an extra pass and storage; do not prototype without a measured hypothesis [INF] |
| À-trous, 5 passes (5×5, step doubling to a 32 px radius), normal and depth weights; the colour weight was removed because it biased toward dark values [SRC] | Our M5.5 A-SVGF-lite (demodulation, moments, temporal-gradient α) supersedes it [REPO PLAN M5.5]. Their colour-weight finding is a known luminance-variance pitfall, which A-SVGF's variance-guided weights address [INF] |
| Temporal accumulation with reprojection, depth-based disocclusion, YCoCg 3×3 neighbourhood clamp [SRC] | Relevant to the M8 "better upscaler" item: a YCoCg variance or clip TAA/upscaler at the 540p→canvas blit [INF] |
| Hot/cold triangle split (positions vs attributes) [SRC] | Already done: `tris`/`trisW` vs `prims`/`verts` [REPO gap-perf §1.2] |
| Per-tile, distance-culled BVHs [SRC] | **Do not adopt.** It drops geometry from GI and occlusion and is biased; it conflicts with unbiased validation [INF] |
| PCG hash RNG, RR after bounce 2 [SRC] | Nothing new |

---

## 7. Recommendation for M8

**Adopt:**
- **CWBVH as the production traversal, built by tinybvh compiled to WASM** in single-threaded mode (`NO_THREADED_BUILDS`, scalar first) inside the existing BVH Worker. Use the `BVH8_CWBVH` node buffer with its SAH-optimal DP collapse. **Re-emit our own Woop `trisW` and primId tail**, keyed by tinybvh `triIdx` and bit-copied from the source vertices. Remap `baseTri`.
- Keep our WGSL CWBVH traversal (gap-perf §10.2), adding the overflow and iteration-cap probe counters, `alpha_pass`, and primId tie-breaking.
- Pin the tinybvh commit and add golden buffer hashes. Re-run T12 (brute-force agreement, watertight) on the CWBVH variant.
- Budget **~1.2× per frame**, as the plan already does.

**Prototype (measure; adopt only if it wins):**
1. tinybvh WASM build time and SAH cost vs our TS builder (428 ms / 262k triangles), scalar vs SIMD128-AVX2.
2. DP collapse vs greedy collapse on the gap-perf §2.7 pass set.
3. SBVH (`BuildHQ`) and `Optimize()` for the **MT production variant only**, gated by a brute-force hit-agreement test; never for Woop lanes unless clipped boxes are padded.
4. Optionally, the `QUNPACK_BITTRICK` unpack from `tiny8.comp`.

Skip `BVH4_GPU` and compressed triangles.

**Not possible on our platform:**
- Hardware ray tracing or ray queries in Chrome 154 WebGPU on Metal: not shipped, not behind any flag, no origin trial, nothing in upstream Dawn, and only an open discussion issue in gpuweb.
- Gigi's and the Randall post's "ray tracing" are both software BVH2 in compute, so they are not a route to hardware RT.
- Revisit only if gpuweb ships bindless and a ray-query proposal appears in `gpuweb/proposals`.
