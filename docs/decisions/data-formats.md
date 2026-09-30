# GPU data formats: audit and quantization plan

Status: **proposal** (design only; nothing in the repo has been changed). Date 2026-09-30.
Scope: `main` (M0–M3c, `/Users/mark.boss/Dev/WebGPURestirPT`) and `m4-restir` (M4, `…/WebGPURestirPT-m4`).
Binding constraints: PLAN §1.2, §1.3, §1.6, §1.9, §2 rule 14 (quantize lossy fields before computing F),
`docs/decisions/scene-bridge.md` (anything the bridge cannot represent exactly is a hard error).

All numbers below were **measured on the CPU** (no GPU jobs) with scratch scripts that import the repo's own loader,
flatten, package reader and SAH builder: Sponza (`validation/assets/downloaded/sponza`), all 62 packages in
`validation/scenes` (both branches), and the 29 M3c packages in `validation/out/m3c/scenes`. The scripts are in
the scratchpad next to this file (`dataformats/*.mts`, `enc.mjs`). Angles are worst cases over 2·10⁶ random unit
vectors followed by local hill climbing, measured in f64.

---

## 0. Decisions at a glance

1. **One lossy step, in the loader. Every GPU encoding after it is a lossless recoding.**
   A new pure function `quantizeScene(SceneData) → SceneData` runs right after every loader (glTF flatten, USD, the
   validation scene kit). It snaps positions, UVs, normals, tangents and COLOR_0 to their lattices and writes the
   *dequantized* f32 values back into `SceneGeometry`. Everything downstream sees exactly those f32 values: the BVH
   builder, lights (emissive areas), the CPU tracer and picking, the scene-package export (so Blender gets them), and
   the GPU packer. The packer re-encodes them and **throws** if any value does not round-trip bit-exactly. This is
   rule 14 applied to geometry. Validation exactness is kept by construction.
2. **Positions: one global power-of-two lattice, 3 × 21 bits in 8 bytes ("P21").** This replaces both the
   per-object bounding-box UNORM16 proposal and per-mesh origins. The step is `s = 2^k` with
   `k = max(⌈log2(E/(2²¹−2))⌉, ⌈log2(A/(2²⁴−1))⌉)`, where E is the largest scene extent and A the largest
   |world coordinate|. Sponza: **s = 15.3 µm, worst error 13.2 µm**, 0 degenerate, 0 flipped, **0 cracks**. The
   per-object proposal gives 0.45 mm steps and opens 72 of 72 cross-mesh seams (up to 0.32 mm). Decoding is exact
   integer arithmetic times a power of two, so it is bit-exact on CPU and GPU even under fast-math/FMA. It also
   **removes an existing inexactness**: today 37% of Sponza's recentred coordinates, and those of 19 of 87 validation
   packages (including `xi_contact`), are rounded when recentred to f32, so the GPU and Blender differ by up to half
   an ulp. On the lattice that count is 0.
3. **Normals: octahedral 2 × 16 snorm (4 B); worst error 0.0025° (precise encoder) / 0.0037° (round-to-nearest).**
   RGB10A2 is 0.079–0.097°, so oct is **~30× better at the same size**. Flat faces get a new `TRI_FLAT` flag and use ng
   (as Cycles does), so they need no stored normal.
4. **Tangents: octahedral 2 × 15 snorm + bitangent sign bit (4 B); worst error 0.0049° / 0.0074°.**
   RGB10A2 is ~0.08–0.1°. Tangents are the one attribute that cannot be exported: Blender recomputes MikkTSpace from
   the exported mesh. See E-7.
5. **UVs: 2 × 16 bits on a per-material dyadic lattice `uv = (q + base)·2^k` (4 B), with a per-material "wide"
   fallback to 2 × f32 when the worst error exceeds τ texels.** τ is measured on the largest bound texture after
   KHR_texture_transform; the default is 1/8 texel. Sponza UVs reach ±32 (u ∈ [−27.8, 29.7]): **1,476 vertices lie
   outside [−8, 8]**, and fp16 would err by up to 16 texels on 1024² textures. See §B5.
6. **Re-weld after flatten.** MikkTSpace forces unwelding, which inflates Sponza from 192,493 unique vertices to
   786,783 (×4.09). Re-welding identical (p, n, uv, t, color) tuples is lossless and is the **largest single memory
   win** (−75%).
7. **Env texture: pick the smallest lossless format per map (rgb9e5 → rgba16float → rgba32float).**
   All three validation HDRIs and every M3c env are lossless in rgba16float; the HDRIs are also lossless in rgb9e5.
   RGBE sources are always lossless in rgb9e5 for exponents 112…144. Only the synthetic C0s sun (value 10 000 needs
   10 mantissa bits) falls back to f16. Validation adopts this only after a hardware-filtering equality test (E-3).
8. **Everything radiometric, pdf-valued, Jacobian-valued or accumulated stays f32**, as do the V-buffer barycentrics
   (§C). The M4 validation reservoir (160 B) is unchanged. The M8 96 B production layout reuses the encodings
   defined here: dyadic 2 × 16 barycentrics and light uv, oct 2 × 16 rcWi/sfxDir. "oct16" must mean 16 bits **per
   component**: 2 × 8 is 0.64° (§B12).
9. **Every Cycles reference whose geometry moves must be re-rendered once.** That is 45/62 packages in
   `validation/scenes` and 19/29 M3c packages, with displacement ≤ 29 µm. 27 packages are bit-identical (dyadic
   coordinates) and keep their references. No current validation package is smooth-shaded, so the normal format
   changes no reference. Schedule: after the M4 merge, so the ladder runs once (§D).

---

## A. Inventory

Sponza = Khronos glTF Sponza as our loader produces it: 262,266 triangles, **786,783 vertices after flatten**
(102/103 primitives unwelded for MikkTSpace), 25 materials, 69 textures (68 × 1024², 1 × 4²), extent 29.77 m.
540p = 960 × 540 = 518,400 px (the interactive default, PLAN §1.10). MB = 10⁶ B.
"Tier" is the implementation step in §D (P0–P5), or "keep".

### A.1 Scene data (resolution-independent)

| # | Item | Branch | File (layout / creation) | Current element | B/elem | Sponza size | Recommendation | Tier |
|---|---|---|---|---|---|---|---|---|
| 1 | Vertex positions | both | `render/scene-gpu.ts` `packVertices`; `shaders/scene/scene-data.wgsl` `SceneVertex.p` | vec3f (recentred) | 12 | 9.44 MB | P21 global lattice, 8 B | P1 |
| 2 | Shading normals | both | same (`SceneVertex.n`) | vec3f | 12 | 9.44 MB | oct 2×16 snorm, 4 B; `TRI_FLAT` → ng | P1 |
| 3 | Tangents | both | `scene-gpu.ts` (`tangents` buffer, uploaded but **not bound** before M7) | vec4f | 16 | 12.59 MB | don't upload until M7; then oct 2×15 + sign, 4 B | P0 / M7 |
| 4 | UV0 | both | `SceneVertex.uvx/uvy` | 2 × f32 | 8 | 6.29 MB | 2 × 16 per-material dyadic lattice, 4 B (+ wide fallback) | P1 |
| 5 | COLOR_0 | both | `SceneVertex.color` (always present, 1 when absent) | vec4f | 16 | 12.59 MB (all 1s) | optional section; rgba8 (unorm8 source) or rgba16 exact products, 4/8 B | P1 |
| – | **Vertex record total** | | `VERTEX_BYTES = 48` (+16 tangents) | | 64 | **50.36 MB** | **16 B (+4 tangent) × 192,493 re-welded = 3.08 / 3.85 MB** | P1 |
| 6 | Indices + material + triFlags | both | `packTris` → `sceneTris: array<vec4u>` | (i0,i1,i2, mat \| flags≪24) | 16 | 4.20 MB | keep; add `TRI_FLAT` (bit 3) | keep |
| 7 | BVH2 nodes | both | `bvh/layout.ts`, `shaders/bvh/traverse.wgsl` | 4 × vec4f (two child AABBs + refs) | 64 | 131,360 nodes = 8.41 MB | keep f32 (BVH2); CWBVH 80 B/8-wide in M8 (3.46 MB) | keep / M8 |
| 8 | BVH tris, MT | both | `layout.ts` `tris` | (v0\|primId, e1, e2) vec4f × 3 | 48 | 12.59 MB | lattice-coded 32 B, shared MT/Woop | P3 |
| 9 | BVH tris, Woop | both | `layout.ts` `trisW` + primId tail | 3 × (v, vid) + ¼ vec4 | 52 | 13.64 MB | same 32 B (vid replaced by lattice-key order) | P3 |
| 10 | Materials | both | `MATERIAL_LAYOUT` (336 B), `MaterialGpu` | f32 + TexSlot × 7 | 336 | 8.4 KB | keep; add per-material UV lattice params in the pad words | keep (+P1 fields) |
| 11 | Light records (cur/prev) | both | `render/lights-gpu.ts` `LIGHT_REC_WORDS = 28` in `records` | f32/u32 | 112 | ≤ kB | keep | keep |
| 12 | Light alias table | both | `render/alias.ts` `packAliasEntries` | (q, alias) u32 × 2 | 8 / entry | tiny (scene lights only) | keep (optional pack `alias≪16 \| q` when n ≤ 2¹⁶) | P4 |
| 13 | Realized pmf, cur↔prev maps | both | `lights-gpu.ts` | f32 / u32 | 4 | tiny | keep (rule 15, J_P ≡ 1 bitwise) | keep |
| 14 | Emissive-triangle entries | both | `emissive-tris.ts` → `records` | (primId, area f32) | 8 | scene-dependent (8 B per emissive tri) | keep; areas from lattice positions (automatic) | keep |
| 15 | primId → emissive entry map | both | `lights-gpu.ts` `primMap` | u32 per primId | 4 | 1.05 MB | keep | keep |
| 16 | BSDF LUTs | both | `render/luts/lut-layout.ts` (in `records`) | f32 | 4 | 9,248 floats = 37 KB | keep (Cycles tables verbatim) | keep |
| 17 | Env row/col alias | M3c+ | `scene/env/env-importance.ts` | `alias≪16 \| q` u32 | 4 | 1k env: 2.1 MB; 4k (cap 4096): 33.6 MB | keep (already packed) | keep |
| 18 | Env pdfUV | M3c+ | same | f32 | 4 | 1k: 2.1 MB; 4k: 33.6 MB | keep (ENV-U3 realized-pdf identity) | keep |
| 19 | Env texture | M3c+ | `render/env-gpu.ts` | rgba32float | 16 | 1k: 8.4 MB; 4k: 134 MB; 8k: 537 MB | lossless-only rgb9e5 → rgba16float → rgba32float | P2 |
| 20 | Material textures (validation) | both | `render/textures-gpu.ts` | rgba8unorm, `texture_2d_array` per (w,h), 1 mip | 4 / texel | 68 × 4 MiB = 285 MB | keep (plan §1.6: never resample) | keep |
| 21 | Material textures (interactive) | both | same | rgba8unorm, 256–2048 buckets + mips | 4 / texel | ≈ 380 MB | BC7/ASTC is an open question (E-11) | P5 |

### A.2 Screen-space buffers, main branch (PT)

| # | Item | File | Current | B/px | 540p | Recommendation | Tier |
|---|---|---|---|---|---|---|---|
| 22 | G-buffer | `render/renderer.ts` `GBUF_TEXEL_BYTES = 80`, `passes/gbuffer.wgsl` | ng, thr, ns, viewZ, pos, matId, albedo, flags, motion (f32), pad | 80 | 41.5 MB | debug-only today; in M5.5 drop `pos` and pad, oct32 ng/ns → 40 B | P4 |
| 23 | Motion vectors | inside #22 | vec2f px | 8 | (in #22) | keep f32 (see B11) | keep |
| 24 | V-buffer | `renderer.ts` `vbuffer` | rgba32uint (primId, bits u, bits v, matId) | 16 | 8.3 MB | keep f32 barycentrics | keep |
| 25 | Accumulation (interactive, PtFrame) | `renderer.ts` `accum`, `pt-kernel.ts` `pt-frame-accum` | vec4f sum | 16 | 8.3 MB each | keep f32 | keep |
| 26 | Batch accumulator | `render/batch-accumulator.ts` | vec4f per batch + host f64 | 16 | 4.2 MB at 512² | keep f32 + f64 host | keep |
| 27 | Debug AOV buffer + probe | `render/debug-views.ts` (`aovStride 16`, header 8,256 B), `probe.ts` | vec4f | 16 | 8.3 MB | keep | keep |
| 28 | debugOut texture | `debug-views.ts` | rgba32float | 16 | 8.3 MB | keep | keep |
| 29 | Counters, uniforms, timestamps, overlay VB | various | small | – | < 1 MB | keep | keep |

### A.3 ReSTIR (m4-restir), interactive atlas = 540p (E = 1, NS = 3) and validation atlas 1024² (E = 16 × 256², NS = 6)

| # | Item | File | Current | B/px | 540p | 1024² val. | Recommendation | Tier |
|---|---|---|---|---|---|---|---|---|
| 30 | Reservoirs `resA` + `resB` | `render/restir/resources.ts`, `layout.ts` (`RES_BYTES = 160`), `shaders/restir/reservoir.wgsl` | 10 × vec4u planes | 2 × 160 | 165.9 MB | 336 MB | keep 160 B validation layout; M8 96 B production (§B12) | keep / M8 |
| 31 | Suffix cache | inside #30 (planes P6–P8) | ids + f32 | (48 of 160) | – | – | M8: 32 B (§B12) | M8 |
| 32 | Shift arena | `layout.ts` `arenaBytes = 256 + 24·P·NS` | slots (FJ f32×3, Jword), codes u32, items u32 | 24·NS | 37.3 MB (NS=3) | 151 MB (NS=6) | keep FJ/J f32; code word already packs f16 margin | keep |
| 33 | Indirect args | `resources.ts` `rs-args` | u32 | 64 B | – | – | keep | keep |
| 34 | `rsVbuf` | `resources.ts` | rgba32uint (primId, u, v, camDist) | 16 | 8.3 MB | 16.8 MB | keep | keep |
| 35 | `rsGeo` | `resources.ts` | rgba32float (ng, thr) | 16 | 8.3 MB | 16.8 MB | M8: rg32uint (oct32 ng + f32 thr) if every consumer decodes | P4 |
| 36 | `rsL1`, `rsShade`, `rsFrame` | `resources.ts` | rgba32float | 3 × 16 | 24.9 MB | 50.3 MB | keep f32 (unclamped radiance > 65504) | keep |
| 37 | Pairing maps | `resources.ts` `rs-pairtex`, `render/restir/pairing.ts` | rg8sint 256² × 8 | – | 1.05 MB | 1.05 MB | keep (already minimal) | keep |
| 38 | Mask texture | `resources.ts` | r8uint | 1 | – | – | keep | keep |
| 39 | `ensStats`, `ensPixel` | `resources.ts`, `ensemble.ts`, `batch-runner.ts` | f32 sums (+ f64 host) | 32 (ensPixel) | – | 33.6 MB (E·256²·32) | keep f32 | keep |
| 40 | Candidate dump (tests only) | `layout.ts` `dumpBytes` | 32 × 48 u32 + count | 6,148 | – | – | keep (test fixtures are tiny) | keep |
| 41 | rs-frame-accum, counters, probe accum | `restir/kernel.ts`, `batch-runner.ts` | vec4f | 16 | 8.3 MB | – | keep | keep |
| 42 | `RestirParams`, `RsDispatch` ring | `resources.ts` `createUniforms` | uniform | 128 B / 512 × 256 B | – | – | keep | keep |

---

## B. Per-item analysis and recommended formats

### B0. The architecture that keeps validation exact

```
glTF / USD / scene-kit ──► flatten ──► quantizeScene() ──► SceneData (dequantized f32, on-lattice)
                                             │                    │
                                             │                    ├─► exportScenePackage (v2) ──► Blender/Cycles
                                             │                    ├─► SAH BVH (Worker)  ──► bvh nodes/tris
                                             │                    ├─► LightsGpu (emissive areas), CPU trace, picking
                                             │                    └─► packVerticesQ() ──► lossless recode, throws if not exact
                                             └─ quant params (k, per-material UV lattices) stored in SceneData and package v2
```

- **Idempotency.** The step is chosen from the data, and snapping can move the extent across a power-of-two
  threshold, so re-deriving k from a package is not safe. Package **v2** stores `quant` (posLog2, per-material
  `{ku, kv, baseU, baseV, wide}`, normal/tangent/color encodings). `readScenePackage` **asserts** that every value
  is on its stored lattice (hard error, as the bridge contract requires). `quantizeScene(quantizeScene(x))` must
  equal `quantizeScene(x)` bit for bit (U-Q1).
- **Order inside `quantizeScene`** (it matters): (1) positions → (2) drop triangles that became degenerate (logged;
  primIds stay dense, flatten.ts rule) → (3) recompute flat normals from the *snapped* positions and set `TRI_FLAT`
  → (4) oct-snap smooth normals → (5) MikkTSpace on the snapped positions/normals/UVs, so Blender's own MikkTSpace
  sees the same inputs → (6) snap tangents → (7) snap UVs and COLOR_0 → (8) re-weld identical tuples.
- **Render origin.** `computeRenderOrigin` snaps O to the lattice: `O = round(centre/s)·s`. Then
  `recentrePositions` (f64 subtract, f32 store) is exact, because |n − n_O| ≤ 2²¹ < 2²⁴.
- **Two decode classes.** (a) *Dyadic* fields (positions, UVs) decode as `f32(int) × 2^k`. The integer→float
  conversion is exact below 2²⁴, and multiplying by a power of two is exact, so CPU = GPU bit for bit, even if Metal
  fast-math reassociates or contracts to FMA (the products are exact). (b) *Unit vectors and colours* decode through
  a correctly rounded product `f32(q) × f32(1/(2^b−1))`. The CPU mirror (`Math.fround(q * Math.fround(c))`) is bit
  identical. It differs from `f32(q/255)` in 126/256 codes, but 255 → exactly 1.0, so α ≡ 1 survives. The
  normalisation afterwards (`normalize`, WGSL ≤ a few ulp) and the oct fold under fast-math may differ from the
  exported f32 value by a few ulp (≤ 4·10⁻⁷ rad). That is the same class as the existing interpolation-order and FMA
  differences, and 10³× below the quantization error. Do **not** use `unpack4x8unorm`/`unpack2x16unorm` for exported
  fields: WGSL leaves their division precision to the implementation.
- **Lossless mode.** `quantizeScene({ mode: 'lossless' })` is the identity, and the packer then selects the f32
  vertex format (today's 48 B record) through a compile-time define `VERTEX_FORMAT`. It serves three purposes:
  scenes that exceed the precision floor (§B1), and the loader-fidelity and E2E stock-import gates (vii-L, viii-L,
  E2E-GLB/USD in M7), where Blender imports the *original* asset rather than our package.

### B1. Vertex positions: global 21-bit lattice (P21)

**Encoding.** n = round(p_world / s) is an i32 per axis with s = 2^k. The vertex stores the offset
`n − n_min ∈ [0, 2²¹)` for x, y, z in 63 bits of a `vec2u`, with 1 spare bit. A per-scene header holds
`posBase = n_min − n_O` (i32 × 3) and `posScale = 2^k`.

```wgsl
fn vq_pos(w: vec2u) -> vec3f {                       // ~13 ALU, 0 extra loads (header is uniform/cached)
  let x = w.x & 0x1FFFFFu;
  let y = (w.x >> 21u) | ((w.y & 0x3FFu) << 11u);
  let z = (w.y >> 10u) & 0x1FFFFFu;
  return vec3f(vec3i(vec3u(x, y, z)) + sceneQ.posBase) * sceneQ.posScale;   // exact: |int| ≤ 2^22, × 2^k
}
```

**Choice of k.** `k = max(⌈log2(E / (2²¹ − 2))⌉, ⌈log2(A / (2²⁴ − 1))⌉)`.
- The E term keeps every offset inside 21 bits.
- The A term keeps `n·s` exactly representable as an f32 **world** coordinate, which is what the package exports
  and Blender stores. The lattice is therefore never finer than f32 at the farthest vertex, which is also Cycles'
  own precision.
- Precision floor: if `s > s_max` (default 2⁻¹⁰ m ≈ 0.98 mm, i.e. E > 2 km), the scene falls back to lossless f32
  (§B0).

**Measured.**

| Scene | E | k | step | max displacement | degenerate / flipped tris | ng rotated > 1° |
|---|---|---|---|---|---|---|
| Sponza | 29.77 m | −16 | 15.3 µm | 7.6 µm/axis, 13.2 µm Euclid. | 0 / 0 | 12 slivers (max 1.16°) |
| Sponza at k = −13 | | | 122 µm | 105 µm | 0 / 0 | 5,126 (max 10.4°) |
| Sponza at k = −11 | | | 0.49 mm | 0.42 mm | 13 / 2 | 41,364 (max 30°) |
| Sponza at k = −10 | | | 0.98 mm | 0.84 mm | 125 / 8 | 80,844 (max 90°) |
| Cornell family (`cornell_i`, ii, ix-*, xiv, iv, g8) | 0.555 m | −21 | 0.48 µm | ≤ 0.36 µm | 0 | – |
| `xi_contact_512` | 2.5 m | −19 | 1.9 µm | 1.1 µm | 0 | – |
| `c0a_far_512` (A = 1003 m binds) | 5.3 m | −14 | 61 µm | 29 µm | 0 | – |
| All 91 packages | | | | ≤ 29.2 µm | 0 | – |

The Sponza SAH BVH built on lattice positions is equivalent to the f32 one: 131,361 vs 131,360 nodes, SAH cost
76.70423 vs 76.70421. MT edge components that are not exact in f32: 680 today, 0 on the lattice.

**Watertightness and contacts.** Rounding is a deterministic function of the input value. So bit-identical input
vertices, whether shared within a mesh or duplicated across meshes, map to bit-identical lattice points, and the
decode is exact. The Woop canonical-edge argument therefore holds across mesh boundaries. Measured:
- **Global lattice:** 0 cracks by construction.
- **Near-coincident, not identical vertices** (distinct f32 bits, closer than one step) can round apart by at most
  one step. Sponza has exactly **1** such pair (0.95 µm → 15.3 µm). `xi_contact`, the Cornell scenes and the glossy
  spheres have **0**, because their contacts use identical coordinates.
- One step (≤ 15 µm at Sponza) is below the Wächter–Binder offset at the same coordinates: 256 ulp ≈ 0.24 mm at
  10 m, and a float offset of 15.3 µm below 1/32 m.

**Alternatives evaluated.**

| Scheme | bytes | Sponza step / max err | cracks (72 cross-mesh shared positions) | notes |
|---|---|---|---|---|
| **P21 global lattice (recommended)** | 8 | 15.3 µm / 13.2 µm | **0** | no per-mesh data, no extent limit per mesh |
| User proposal: RGBA16 UNORM in per-object bbox, scale in model matrix | 8 | 0.45 mm / 0.23 mm per axis | **72 / 72 open, max gap 0.32 mm** | lattices differ per mesh. The model-matrix multiply produces non-lattice f32 values, so no mesh edge is bit-identical to its neighbour. Needs instancing (TLAS/BLAS), which the static flattened BVH (PLAN §1.3) does not have |
| Global step + per-mesh integer origin + u16 offsets (4th u16 = mesh id) | 8 | step forced to 2⁻¹¹ (0.49 mm) by the 29.8 m mesh; 0.42 mm | 0 | 13 degenerate + 2 flipped triangles on Sponza; a mesh > 65,535 steps needs cluster splitting (triangles longer than a cluster cannot be split) or f32 |
| Same, with a 2¹⁶-step brick table (lo 16 bit + brick id) | 8 | could reach 24-bit precision | 0 | brick count grows with surface area (Sponza at 3.8 µm: ≫ 65,536 bricks): rejected |
| f32 (today) | 12 | – | 0 | recentring rounds 37% of Sponza components; 19/87 packages |

With P21 no mesh-extent fallback is needed: the only limit is scene-wide (E ≤ 2²¹·s_max). The fallback is
therefore scene-level lossless f32, not cluster splitting.

**Validation exactness: safe.** Blender receives `n·s`, the exact f32 of what the GPU decodes. It also makes the
bridge's `1 − v` UV flip and the recentring exact (§B5).
**Saving:** 12 → 8 B per vertex.
**Unpack:** ~13 ALU per vertex, 3 vertices per hit.
**Consumers to switch to an accessor `scene_vertex_pos(i)`:** `scene-data.wgsl` (`scene_surface`),
`lights/emissive.wgsl` (5 direct `.p` reads), `passes/restir/debug.wgsl` (m4).

### B2. Indices, triMaterial, triFlags

`sceneTris` is one aligned `vec4u` per triangle: 3 × u32 indices, plus a 24-bit material and 8 flag bits. Keep it.
- 21-bit index packing would save 4 B/triangle but caps the scene at 2²¹ vertices (PLAN allows 2²⁴ triangles), and
  one 16 B load is already optimal.
- Add `TRI_FLAT = 8` (bit 3). The re-weld (§0 item 6) shrinks the vertex count, not the index size.

### B3. Normals

| Encoding (32 bit unless noted) | worst error | mean |
|---|---|---|
| xyz snorm10 (RGB10A2 read as signed) | 0.0969° | 0.0423° |
| xyz snorm10, best-of-8 rounding | 0.0792° | 0.0310° |
| xyz unorm10 (RGB10A2 as proposed; 0 not representable) | 0.0968° | 0.0423° |
| xyz fp16 (48 bit) | 0.0229° | – |
| oct 2×8 (16 bit) | 0.957° (precise 0.638°) | 0.337° |
| oct 2×12 (24 bit) | 0.0593° | 0.0209° |
| **oct 2×16 snorm** | **0.00370°** (64.5 µrad) | 0.00131° |
| **oct 2×16, precise (best of 4)** | **0.00247°** (43.0 µrad) | 0.00122° |

The user's reasoning is confirmed. fp16 xyz spends 48 bits on 0.023°. An unorm/snorm cube encoding wastes codes
off the sphere, and the octahedral map puts every code on the sphere.

**Recommendation:** oct 2 × 16 snorm with the precise encoder, in word `z` of the vertex record.
- Decode (~15 ALU): `e = vec2f(vec2i(sign-extended q)) * f32(1/32767)`, then the Stubbe fold
  (`t = max(−v.z, 0); v.xy += select(t, −t, v.xy ≥ 0)`), then `normalize`.
- Axis-aligned normals are exact.
- **Flat faces:** `TRI_FLAT` makes `scene_surface` return `ns = ng` without fetching normals. This matches Cycles
  (Ng on flat faces) *exactly*. Today we instead use a stored face normal computed in f64 by flatten, which differs
  from the GPU's `normalize(cross)` by an ulp. It is a prerequisite for oct normals: without it, flat faces would
  pick up a 0.0025° ns ≠ ng deviation.
- Validation: all 91 current packages are flat-shaded (`flatShaded: true`), so **no reference changes** because of
  normals. Smooth packages (M7 gate 3.8) export the decoded f32 normals through the bit-exact `custom_normal` path
  in `build_scene.py` (lines 183–199).
- **Saving:** 12 → 4 B (0 B for flat-only vertices).

### B4. Tangents (M7; currently computed, uploaded, never bound)

| Encoding (32 bit) | worst | mean |
|---|---|---|
| RGB10A2: xyz 10 bit + A = sign | 0.079–0.097° | 0.031–0.042° |
| **oct 2×15 snorm + sign bit (+1 spare)** | **0.00739°** | 0.00261° |
| oct 2×15, precise | **0.00494°** | 0.00244° |

**Recommendation:** oct 2 × 15 + sign, in its own optional 4 B section. It is present only when some material has a
normal map. On Sponza that is every primitive.
- Decode: two sign-extending shifts (`i32(w ≪ 17) ≫ 17`, `i32(w ≪ 2) ≫ 17`), a product, the oct fold, normalize, and
  the sign from bit 31. About 18 ALU.
- A tangent-angle variant (θ relative to a deterministic ONB of the decoded normal, 16 bit + sign) reaches a similar
  ~0.005° in 17 bits. It is only worth it if another field needs the other 15 bits. Not recommended now.
- **Stop uploading** the 16 B/vertex tangent buffer until M7 binds it (P0): 12.6 MB for Sponza today.
- Exactness: see E-7. Tangents cannot be exported; Cycles recomputes them.

### B5. UVs

**Encoding.** Per material, per axis: `uv = (q + base) · 2^k`, with q ∈ [0, 65535] and k the smallest exponent such
that (max − min)/2^k ≤ 65534. `base` (i32) and k (i8) live in the `MaterialGpu` pad words. The decode is exact
(dyadic), so CPU = GPU, and Blender gets the same UVs. A vertex referenced by two materials with different lattices
is duplicated by the GPU packer. That is internal only: primIds and the export are unaffected.

```wgsl
fn vq_uv(w: u32, m: MaterialGpu) -> vec2f {        // ~8 ALU; wide groups: one extra 8 B load
  if ((m.flags & MAT_UV_WIDE) != 0u) { return uvWide[w]; }
  return vec2f(vec2i(vec2u(w & 0xFFFFu, w >> 16u)) + m.uvBase) * m.uvScale;
}
```

**Precision check.** For texture ref r (W × H, transform T = [a b c; d e f]) the worst texel error is
`max(W(|a|·s_u/2 + |b|·s_v/2), H(|d|·s_u/2 + |e|·s_v/2))`. The maximum over all refs of the material must be ≤ τ.
- Default τ = 1/8 texel. That is 4× stricter than the user's criterion "step ≤ 1 texel of a 4096² texture", and 32×
  looser than the 1/256-texel bilinear weight precision of the sampler.
- Materials that fail become **wide** (2 × f32 in a side table; the vertex word holds the index).
- τ = 0 ("lossless-only") is used for vii-L / E2E-GLB.

**Measured on Sponza** (post-flatten vertices; textures 1024²):

| Scheme | worst error | out of range | notes |
|---|---|---|---|
| fp16 | **16 texels** (u ≈ 30: fp16 ulp 2⁻⁵) | – | confirms the user's objection |
| User: RG16 UNORM over [−8, 8] | 0.125 texel (1024²), 0.5 texel (4096²) | **1,476 vertices (source: 4,785 of 192,496 = 2.5%)** | not dyadic (16/65535) → FMA-sensitive decode; a dyadic variant (step 2⁻¹², q·2⁻¹² − 8) fixes that |
| Per-material dyadic 16 bit | 0.5 texel (material 5: u ∈ [−27.8, 29.7]) | none | materials 5, 6, 7 fail τ = 1/8 → wide (50,520 post-flatten vertices, ≈ 12k after re-weld; ≈ 0.1 MB) |
| Per-primitive dyadic 16 bit | τ = 1/2: 100%, 1/8: 97.6%, 1/16: 97.5%, 1/32: 81.9% of source vertices pass | none | needs a per-triangle group id (tri.w has no free bits → a group table indirection); optional refinement |

For a [0, 1] UV range, the per-material lattice gives 2⁻¹⁶ steps: 1/32 texel worst case at 4096². That is 16× better
than the fixed [−8, 8] range.

**Validation packages:** the only textured packages (`vii_textured`, `xii_alpha_foliage`) are already on their lattice
(0 UVs change). Worst errors are 0.0037 and 0.0020 texel.
**Bonus:** `build_scene.py` computes `v_b = 1 − v` in float32. On a dyadic lattice that is exact; for arbitrary f32 UVs
below 0.5 it is not.
**Saving:** 8 → 4 B.

### B6. COLOR_0

Today every scene pays 16 B/vertex, even without COLOR_0 (all ones: 12.6 MB on Sponza). Recommendations:
- Make it an optional section, present only if the scene has COLOR_0.
- Store rgba8 (4 B) when every source accessor is unorm8, which is lossless. Otherwise store rgba16 (8 B,
  quantization error ≤ 7.6·10⁻⁶). α enters the MASK test (α ≥ cutoff), so the snapped α is exported with it.
- Decode with the exact product `f32(q) * f32(1/255)` (§B0), not `unpack4x8unorm`.

### B7. BVH nodes and triangles

- **Nodes:** keep f32 AABBs in BVH2 (64 B/node, 8.41 MB). M8's CWBVH quantizes child boxes conservatively (lo floor,
  hi ceil). That changes traversal order only, never which triangle is hit, apart from exact-t ties (gap-perf §2.7:
  99.99% identical), so it is safe for validation. The builder must run on the lattice positions (B0) and does.
- **Triangles (P3):** one 32 B record for both MT and Woop: `vec4u × 2 = [P21 v0 | P21 v1] [P21 v2 | primId, spare]`.
  - MT edges are computed in-shader and are exact on the lattice (0 inexact vs 680 today).
  - The Woop canonical edge order compares the 63-bit lattice keys instead of welded vids. Identical positions give
    identical keys, which is exactly what vid encoded, so the tail and the vid words disappear.
  - Saves 12.59/13.64 → 8.39 MB (−33/−38%) and removes the MT↔Woop re-upload.
  - Cost: ≈ 40 extra ALU per triangle test for 16 fewer bytes. **Perf-gated** (E-10).
  - Do it with CWBVH in M8, or earlier if a traversal microbenchmark shows ≥ 5%.

### B8. Materials, lights, alias tables, LUTs, env tables

- All are small, or already integer-packed (env alias `alias≪16 | q`; the alias thresholds are u16 by design,
  rule 15).
- The f32 fields carry Cycles-exact values (emission, spot smoothstep, LUT texels) or realized pmfs that must be
  bitwise stable across frames (J_P ≡ 1). **Keep them all.**
- Optional (P4): pack the light alias entry to one u32 when n ≤ 2¹⁶. That saves 4 B/entry, which only matters with
  many emissive triangles.

### B9. Env texture

**Measured** (texels as decoded, RGBE OIIO-exact, `scene/env/hdr.ts`):

| Env | max | rgb9e5 lossless | rgba16float lossless |
|---|---|---|---|
| `studio_small_09_1k` (xiii) | 524 | yes | yes |
| `overcast_soil_puresky_1k` (xiv, C0r) | 5.56 | yes | yes |
| `kloofendal_48d_partly_cloudy_puresky_1k` (xiv) | 61,180 | yes | yes |
| C0q constant | 1 | yes | yes |
| C0s single-texel sun | 10,000 | **no** (1 texel: 10,000 needs 10 mantissa bits) | yes |

In theory, an RGBE texel `byte · 2^(e−136)` is exactly representable in rgb9e5 and in f16 whenever
112 ≤ e ≤ 144, because the shared exponent is identical and there are ≥ 8 mantissa bits.

**Rule.** Upload each map in the smallest format in which **every texel round-trips bit-exactly**: rgb9e5ufloat
(4 B), else rgba16float (8 B), else rgba32float (16 B). Env maps are therefore never quantized, the `env.exr` export
and the ENV-U9 hash are unchanged, and E2E-HDR stays exact.
- **Saving:** 4× at 4k (134 → 34 MB) and 8k (537 → 134 MB).
- The sampling code is unchanged (`textureSampleLevel`, filterable).
- **Condition for validation mode:** a GPU test must show that hardware bilinear filtering of f16/9e5 texels equals
  filtering of the same texels stored as f32 (E-3). Until it passes, validation keeps rgba32float and only
  interactive mode uses the compact format.

### B10. Material textures

Validation must stay rgba8unorm, never resampled (PLAN §1.6; the bridge exports lossless PNGs). Interactive mode
dominates memory (Sponza ≈ 380 MB with mips). Block compression is the only large lever. It stays exact only with
bit-exact decoders (BC7, ASTC LDR) plus exporting the decoded texels. This is an open question for M8 (E-11), outside
this audit's first steps.

### B11. Screen-space buffers (main)

- **V-buffer:** keep the f32 barycentric bits. `vertex_from_ids` must be bit-identical to `scene_surface`
  (restir-api §2.3), and T3 round trips depend on it. It is only 8.3 MB at 540p.
- **G-buffer (80 B):** consumed only by debug views today (`debug/gbuffer-views.wgsl`). When M5.5 defines its guides,
  drop `pos` (reconstructible from the V-buffer) and `pad`, store ng/ns as oct 2 × 16 (0.0025°), and keep viewZ and
  thr f32. Result: 40 B/px, saving 20.7 MB at 540p (P4).
- **Motion vectors:** keep f32. f16 pixel motion has 0.5 px ulp for |mv| ≥ 512 px. That is harmless for ReSTIR
  unbiasedness (the shift decides) but not for the denoiser. The saving (2 MB) does not justify it.
- **Accumulation / debug:** keep (§C).

### B12. ReSTIR buffers and the M8 96 B layout

The M4 validation layout (160 B, f32) is normative (restir-api §2.2, PLAN §1.9) and stays as is.

For the M8 production layout (64 B core + 32 B suffix), the encodings in this document apply to **identifier fields
only**. Rule 14 is honoured by quantizing these at the moment the path tree selects them, before F, the Jacobian
denominators or the suffix are computed:

| Field | Validation | M8 production | Error | Note |
|---|---|---|---|---|
| rc/suffix/endpoint barycentrics (u, v) | 2 × f32 bits | 2 × 16 dyadic (`q·2⁻¹⁶`) | ≤ 2⁻¹⁷ of the edge (76 µm on a 10 m triangle) | exact decode; snap before F (rule 14) |
| light-local (u, v) | 2 × f32 bits | 2 × 16 dyadic | ≤ 2⁻¹⁷ of the light's size | same |
| env NEE (i, j, du16, dv16) | already lossless | unchanged | 0 | PLAN §1.9 |
| rcWi / sfxDir (BSDF_ENV escape ω, ω_o) | f32 × 3 | **oct 2 × 16** (32 bit) | 0.0025° | "oct16" must mean 16 bits per component: oct 2×8 = 0.64°, i.e. ≈ 2 texels of a 1k env (0.35°/texel) and a sizeable fraction of a glossy lobe |
| c | f32 | u8 saturating iff c_max ≤ 255 (PLAN §1.9) | – | offline needs f32 |
| primId, flags, seed, endpointId | u32 | u32 | 0 | |
| W, F, jDen, aux, rcRad, betaS, sfxP2, sfxT | f32 | **f32** | – | radiometric / pdf values (§C) |

The byte budget is tight, and this is an open question (E-9):
- Core: W 4 + F 12 + seed 8 + flags/c 4 + rc 8 + rcWi 4 + jDen 4 + aux 4 + rcRad 12 = 60 B.
- Suffix: ids 8 + sfxDir 4 + sfxT 4 + betaS 12 + sfxP2 4 = 32 B.
- The endpoint triple (≥ 8 B with dyadic barycentrics) and endpointId still have to fit into the remaining 4 B, or be
  derivable.

Shift arena: FJ and J stay f32. J spans many decades; a 16-bit log or f16 form would break the MIS partition checks
`mis.sumM = 0` and `mis.lumL = 0`. The code word already packs an f16 margin.
`rsGeo` → `rg32uint` (oct32 ng + f32 thr) is optional in M8. It is allowed only if every consumer, including the pair
acceptance A(p, q), reads the decoded ng.

---

## C. What must stay as it is (and why)

| Item | Why |
|---|---|
| All accumulators (`accum`, `pt-frame-accum`, `rs-frame-accum`, batch accumulator, `ensStats`, `ensPixel`) | Estimator sums. f16 saturates at 65,504 and stops accumulating after ≈ 2¹¹ unit samples (11-bit significand). f32 with bounded batches (relative error ≤ N·2⁻²⁴, e.g. 1.5·10⁻⁵ at N = 256) plus the f64 host sum keeps the Stage A/B statistics clean |
| Reservoir W, F, wSum, jDen, aux, rcRad, betaS, sfxP2, sfxT, kMargin (and c in validation) | Rule 14 quantizes the path *inputs*, not the integrand. Rounding F or W changes the integrand (∫F_q ≠ ∫F): a bias bounded by the rounding error that never averages out. c is f32 in validation by rule 13. T3 round trips need bit-identical recomputation |
| Shift-arena FJ, J; `jDen` | Jacobians span > 10 decades; MIS partition-of-unity checks must be 0 to f32 rounding |
| Realized pmf, pdfUV, alias thresholds | Rule 15: the pdf *is* the stored integers evaluated in f64 and stored as f32; J_P ≡ 1 bitwise needs identical bits |
| Light records, BSDF LUTs, materials | Cycles-exact constants; negligible size |
| V-buffer / `rsVbuf` barycentrics, `rsVbuf.camDist`, `rsGeo.thr` | Bit-identical surface reconstruction contract (restir-api §2.3); thr feeds the rc predicate |
| `rsL1`, `rsShade`, `rsFrame`, `debugOut` | Unclamped radiance. Small bright emitters exceed f16's 65,504, and the plan forbids clamping. 9e5 cannot be a storage texture |
| BVH2 node boxes | f32 boxes over lattice points are exact. Compression belongs to CWBVH (conservative) |
| Validation material textures (rgba8unorm, 1 mip) | PLAN §1.6; the bridge exports the exact PNG bytes |
| Debug AOVs, probe records, candidate dump | Diagnostics must show exact values; test-only sizes |
| Env tables | Already integer-packed; pdfUV must be f32 (ENV-U3) |
| Indices (u32 × 3) | 2²⁴-triangle scenes need > 2²¹ vertices; one aligned 16 B load |

---

## D. Prioritized implementation plan

The steps are ordered to change Cycles references **once**. Because M4 is mid-gate in the worktree, the
geometry-changing step (P1) should land **after the M4 merge**, on a branch `data-formats`, and be followed by one
full re-validation. P0 and P2 do not change any package and can land any time.

### P0: foundations, no render change (CPU only)

- **Files.**
  - New `src/core/scene/quantize.ts`: lattice choice, the snapping functions, CPU decode mirrors (`Math.fround`), the
    oct encoder (precise), re-weld, and `assertQuantized`.
  - New `src/core/gpu/vertex-format.ts`: the TS mirror of the WGSL decode.
  - `src/core/render/scene-gpu.ts`: stop uploading `tangents` until M7.
- **Tests (new, `tests/scene/quantize.test.ts`).**
  - **U-Q1 positions:** decode(encode(p)) equals the lattice point bit for bit; |p − p_q| ≤ s/2 per axis;
    idempotency; k selection including the A bound (`c0a_far` → k = −14) and the 2 km floor → lossless fallback.
  - **U-Q2 exactness:** recentred values and MT edges exact; world export exact (|n| < 2²⁴); run over Sponza and all
    91 packages (0 inexact; 872,829 → 0 on Sponza).
  - **U-Q4 normals:** worst angle over 10⁷ random plus adversarial directions ≤ 0.0025° precise / ≤ 0.0037° plain;
    axis vectors exact; idempotent re-encode; TS decode ≡ WGSL formula (the GPU check comes in P1).
  - **U-Q5 tangents:** ≤ 0.0050° precise; sign preserved.
  - **U-Q6 UVs:** dyadic decode exact; τ bound per texture ref including KHR_texture_transform (TextureTransformTest
    matrices); wide fallback triggers on Sponza materials 5/6/7; `1 − v` exact on the lattice.
  - **U-Q7 colour:** unorm8 source lossless; 255 → 1.0; CPU product ≡ GPU product.
  - **U-Q3 watertightness (CPU part):**
    - bit-identical inputs give bit-identical outputs;
    - near-pair growth ≤ 1 step (Sponza: 1 pair);
    - degenerate-after-snap triangles are dropped and logged, with primIds dense;
    - re-weld is lossless (every corner's tuple is preserved).
  - Re-weld count: Sponza 786,783 → 192,493.

### P1: geometry lattice and compact vertex record (changes references)

**Files.**
- `scene/types.ts`: `SceneData.quant`; `SceneGeometry` unchanged in meaning (the values are now on-lattice); `TRI_FLAT`.
- `scene/flatten.ts`: TRI_FLAT, flat normals after snapping, MikkTSpace after snapping (B0 order), re-weld.
- `scene/gltf-loader.ts`, `scene/usd/usd-scene.ts` (+ worker paths): call `quantizeScene`.
- `scene/scene-package.ts`: **v2**, with the `quant` block written and asserted on read. Keep the v1 reader for old
  packages, re-quantizing them with an explicit warning.
- `validation/blender/build_scene.py`: accept v2 and assert on-lattice positions (`p/2^k` integral) as a
  bridge-contract check. The `1 − v` flip is now exact.
- `render/frame-uniforms.ts` `computeRenderOrigin` (snap O), `app/app.ts`, `validation/harness/batch-run.ts`.
- `render/scene-gpu.ts`:
  - `packVerticesQ` / `VERTEX_BYTES`;
  - the vertex arena with its sections (base 16 B, optional tangent 4 B, optional colour 4/8 B, `uvWide`) and header
    (`posBase`, `posScale`);
  - `MATERIAL_LAYOUT` UV lattice fields;
  - `VERTEX_FORMAT` define (Q / F32).
- `shaders/scene/scene-data.wgsl`: `vq_*` decoders, `scene_vertex_pos`, TRI_FLAT, `scene_uv0`, `scene_color0`,
  `scene_surface`.
- `shaders/lights/emissive.wgsl`; m4: `shaders/passes/restir/debug.wgsl`. The `restir/rc.wgsl` `vertex_from_ids`
  bit-identity is unaffected because it calls `scene_surface`.
- `validation/scenes/scene-kit.ts` and `make-*.ts` (both branches, incl. `make-m4.ts`): regenerate the packages.
- `tests/render/scene-gpu.test.ts`: WGSL ↔ TS layout.

**Tests.**
- **T12 re-run on lattice geometry** (`validation/gpu-tests/bvh.gpu.test.ts`): 10⁶-ray brute force, Woop,
  self-intersection, zero overflow.
- **T12-Q:** 10⁶ rays aimed at shared edges and vertices of `xi_contact` and at the Sponza cross-mesh seams:
  0 leaks with Woop.
- GPU decode equality: positions and UVs bit-exact vs the TS mirror; normals/colours ≤ 4 ulp.
- `scene_surface` vs the CPU reference on 10⁵ random (primId, u, v).
- Loader and package tests (`tests/scene/*.test.ts`): round-trip v2, reject off-lattice packages.
- Sponza load: vertex count and bytes asserted.

**Gates to re-run.**
- Gate 0 (all).
- Regenerate packages, then **re-render the 64 Cycles references whose geometry changed** (45 in `validation/scenes`,
  19 M3c). The 27 bit-identical packages keep theirs: C0b, C0c, C0e, C0f × 4, C0m, C0n, C0o, C0p × 4, spot C0d × 2,
  vi-B mirror, C0q openbox/quad × 4, C0s × 6.
- Then the M2 (24), M3a (175), M3b (96) and M3c (173) gates.
- The M4 gates (Gate 0 T2–T7/T3-*, Gate 3 rungs 3.1–3.2) on the regenerated `make-m4` fixtures.
- **Q-EQ:** a Stage-A-style two-sample test of lossless vs quantized renders of Sponza and `xi_contact`. It is a
  diagnostic, not an exactness gate; it detects visible fidelity loss.

**Expected result (Sponza).**
- Vertex data 50.4 → 3.85 MB (−92%).
- Scene storage without textures 76.6 → 30.1 MB, so a much larger share of the scene stays cache-resident.
- Shading fetch per hit 160 → 64 B (208 → 76 B with M7 tangents).

### P2: env texture lossless compaction (no reference change)

- **Files:** `render/env-gpu.ts` (format selection + upload), `scene/env/*` (the exactness check at load),
  `env-debug.ts`. The texel arrays exported to `env.exr` are unchanged.
- **Tests:** round-trip exactness for every env (the table in B9); C0s → f16; ENV-U9 hash unchanged; **ENV-F:** GPU
  filtered lookups, f16/9e5 vs f32 texture on 10⁶ random uv (incl. seams and poles), max |Δ|/L ≤ 2⁻²⁰, else
  validation keeps rgba32float.
- **Gates:** ENV-U1…U9, the M3c gate on our side only (the references are unchanged).

### P3: 32 B lattice-coded BVH triangles (perf-gated; with M8 CWBVH at the latest)

- **Files:** `bvh/layout.ts`, `bvh/sah-builder.ts`, `bvh/cpu-trace.ts`, `shaders/bvh/traverse.wgsl`,
  `shaders/geom/intersect.wgsl` (canonical order by lattice key), `scene-gpu.ts` (no more MT/Woop re-upload).
- **Tests:** T12 (both intersectors), T12-Q, the CPU/GPU hit-equality sweep, and the traversal microbenchmark from
  the gap-perf harness (Sponza, coherent and incoherent).
- **Gates:** Gate 0, plus one Stage A smoke per milestone. The references do not change, because the geometry is
  identical.

### P4: screen-space and production layouts (M5.5 / M8)

- G-buffer 80 → 40 B (M5.5 guides), optional `rsGeo` rg32uint, light alias single-word packing.
- M8 96 B reservoir with the encodings of §B12:
  - Gate 3 δ plus T3 round trips on the quantized fields;
  - oct 2 × 16 rcWi for BSDF_ENV;
  - a planted-bias negative control: oct 2 × 8 rcWi must be detected.

### P5: interactive texture compression (open question, M8)

BC7 or ASTC for interactive mode only, if the memory budget demands it (E-11).

**Fit with the plan.**
- P1 is independent of the M8 production reservoir. It changes geometry *inputs* (identical f32 values everywhere)
  and no reservoir field.
- M8 reuses exactly two encodings from this document: dyadic 2 × 16 (barycentrics and light uv) and oct 2 × 16
  (directions).
- CWBVH should adopt the P3 triangle record, so tinybvh's layout is not an independent third format.
- "f16 path state" (M8) is register-level and outside this audit. The same rule applies to it: nothing that feeds
  F, pdfs or Jacobians may be rounded after the fact.

---

## E. Risks and open questions

1. **Reference re-render cost.** 64 Cycles references must be re-rendered once, followed by the full M2–M4 ladder.
   *Mitigation:* land P1 after the M4 merge, in one batch.
2. **Fast-math decode drift for unit vectors and colours.** Few-ulp differences between the exported f32 value and
   the GPU's `normalize` (≤ 4·10⁻⁷ rad). Positions and UVs are immune (dyadic). *Accepted:* same class as the
   existing FMA/interpolation-order differences; documented in `cycles-deviations.md`.
3. **Hardware filtering precision of f16/9e5 textures** may be lower than for f32, even with identical texels (P2
   ENV-F test). If ENV-F fails, validation keeps rgba32float for the env.
4. **Slivers.** After snapping, 12 Sponza triangles rotate their ng by > 1° (max 1.16°). They are exported identically,
   so exactness holds, but they are visible as faceting on grazing slivers. At k = −16 this is negligible; it grows
   quickly for coarser steps (table B1). This is why s_max is 1 mm and not coarser.
5. **Precision floor for huge scenes.** Scenes larger than 2 km fall back to lossless f32, which means two vertex
   formats. The lossless path is exercised by the loader-fidelity and E2E gates, so it stays tested.
6. **Near-coincident, not identical, vertices** can separate by up to one step (Sponza: 1 pair, 0.95 → 15.3 µm).
   That is below the Wächter–Binder offsets at the same coordinates. Authoring tools that snap contacts to identical
   coordinates (as all our validation scenes do) are unaffected.
7. **Tangents are not exportable.** Blender recomputes MikkTSpace from the exported mesh. We feed MikkTSpace the
   snapped data (B0 order), so the only deviations are the MikkTSpace implementation difference (Rust port vs
   Blender C++, pre-existing, ~10⁻⁶ rad) plus the oct 2 × 15 error (≤ 0.0049°). The M7 vii-N / Gate 3.8 decides.
   Fallback: tangent section in f32 (12 B + sign).
8. **Idempotency and old packages.** Package v1 has no stored lattice. Reading v1 re-quantizes with a warning;
   references built from v1 are invalid once P1 lands.
9. **M8 96 B budget.** With the fields of §B12 the core is already ≈ 60 B; where the endpoint triple/endpointId go is
   open. Quantized rcWi and barycentrics have no density w.r.t. the continuous measure (a many-to-one snap), which is a
   second-order bias. The plan's Gate 3 δ is the arbiter; the fallback is oct 2 × 24 in 48 bits (≈ 10⁻⁵°).
10. **P3 trade-off unmeasured.** It saves 16 B per triangle test and adds ≈ 40 ALU. On M5 Pro incoherent traversal
    this is probably a small win, but it must be measured, not assumed.
11. **Interactive texture memory** (≈ 380 MB for Sponza) dwarfs all geometry savings. BC7 and ASTC LDR decoding is
    bit-exact by spec, so exporting decoded texels would keep exactness. BC1–3 decoding is not. Normal-map BC5 z
    reconstruction differs from Cycles' stored z. This is an M8 decision.
12. **UV τ is a fidelity choice, not a validation one.** Dequantized UVs are exported, so validation is exact for any
    τ. The user should confirm the default: 1/8 texel (Sponza: 3 of 25 materials wide), versus the user's own
    "1 texel at 4096²" (none wide).
13. **E2E-GLB / loader fidelity in lossless mode** means those gates do not exercise the quantized path. Q-EQ covers
    the quantized path statistically. Alternative: run E2E quantized with a documented µm-level deviation. A
    decision is needed before M7.
14. **Future instancing** (not in v1: static geometry): a per-BLAS local lattice combined with instance transforms
    cannot keep cross-instance seams bit-exact. That would reintroduce exactly the cracks of the per-object proposal.
    Keep the flattened global lattice for validation scenes.
15. **Units.** The lattice is in metres after unit conversion. USD `metersPerUnit` must be applied *before*
    `quantizeScene`, otherwise k is chosen in the wrong unit (a cm-authored stage would get a 100× too fine grid,
    still correct but with 7 bits of range lost).
