# M8: performance (PLAN §5 M8, §1.10; gap-perf; m8-hwrt-tinybvh-gigi.md)

Branch `m8-perf` from main eda22be (M7 merge). Every speed change in this milestone is landed separately and followed by
its correctness evidence (bitwise identity where the change must not alter results, Gate-0 suites, Stage-B units where
results legitimately change, plants where a new approximation could hide bias). Validation configurations keep their
results bitwise unless a section below says otherwise.

## 1. Method

**Driver.** `validation/harness/run-perf.ts` (Node: Vite + headless Chrome 154 with `--enable-webgpu-developer-features`,
one GPU-lock hold per job) → `validation/harness/perf-run.ts` (page). It drives the interactive `Renderer` in the app's
configuration: ReSTIR-interactive (temporal, N = 3 paired spatial slots, boost 3, c_cap 5, RR at initial sampling,
σ 16 pairing, RIS-NEE M 32, dual MVs, duplication map), light mode B, Möller–Trumbore, validation textures (the app
default), denoiser on (4 à-trous iterations), maxBounces 3 (4 for glass). Per job:
- **frame**: blocks of 8 frames encoded and submitted back to back (one submit per frame, as the app), wall time / 8;
  ≥ 32 timed frames after 32 warm-up frames. This is the reported frame time.
- **latency**: single frames, awaited, minus an empty submit's round trip.
- **passes**: the same frame encoded through a proxy `GPUCommandEncoder` that closes the command buffer before every
  compute pass; each command buffer is submitted alone and awaited, minus the empty-submit round trip (the
  one-submit-per-pass method of the M6 / M7 perf tests; **Q3: no timestampWrites on or around ReSTIR passes**). Σpasses
  exceeds the pipelined frame by the per-submit overhead (~0.1–0.15 ms × ~25 passes).
- **denoiser**: the denoiser's own timing re-run (separate submits with timestamp writes, DN9; allowed by Q3).

Scenes: (i) Cornell `cornell_i_512` (rect light), **Sponza** (glTF, 262 k triangles, 25 normal-mapped materials,
MASK foliage) under `kloofendal_48d_partly_cloudy_puresky_1k.hdr` with one 2000 W point light, camera along the long
axis at 1/5 height (the M5.5 smoke setup), `m6_crossings_B_256` (point / spot / rect / disk / sun / emissive + 3
crossing lights), `m7_nm_smooth_256` (normal maps on smooth geometry). Moving lights: light 0 moves sinusoidally along
x every frame. Run: `npx tsx validation/harness/run-perf.ts --suite baseline --frames 64`.

## 2. Baseline (eda22be; Chrome 154, M5 Pro; run `baseline-eda22be`)

Frame time (ms, pipelined mean of 64 frames) against the PLAN §5 M8 targets:

| Scene | 540p N=3 (≤ 35) | 540p N=1 + moving lights (≤ 32) | 720p N=3 (≤ 51) |
|---|---|---|---|
| Cornell (i) | 16.2 | 15.0 | 28.9 |
| m6_crossings | 25.6 | 23.2 | 44.7 |
| m7_nm_smooth | 27.4 | 24.9 | 48.3 |
| **Sponza + HDRI** | **88.4** | **82.7** | **155.7** |

Per pass (ms, one submit per pass, mean of 16 frames; 540p N=3):

| Pass | Cornell | Sponza | crossings | nm_smooth |
|---|---|---|---|---|
| primary (M1 G-buffer for the denoiser) | 0.47 | 1.79 | 0.48 | 0.56 |
| rs_primary | 0.23 | 1.58 | 0.28 | 0.29 |
| rs_initial | 6.81 | **56.31** | 13.54 | 14.23 |
| rs_t_classify | 1.01 | 4.83 | 1.40 | 1.63 |
| rs_t_forward / select / inverse | 1.58 | 3.76 | 2.18 | 2.25 |
| rs_pair_accept | 1.17 | 1.44 | 1.21 | 1.20 |
| rs_spatial_replay | 0.21 | 0.56 | 0.64 | 0.51 |
| rs_spatial_shift | 2.61 | **14.49** | 3.91 | 4.68 |
| rs_spatial_resample | 1.30 | 1.33 | 1.33 | 1.31 |
| rs_dupmap | 0.84 | 1.27 | 0.83 | 0.84 |
| denoiser (timestamps) | 1.59 | 2.99 | 1.67 | 1.67 |
| replay fraction f_r | 0.009 | 0.005 | 0.047 | 0.028 |

At 720p Sponza's denoiser takes 5.26 ms; Cornell 2.73 ms.

**Where Sponza's time goes** (540p N=3, run `variants-sponza-eda22be`, frame / rs_initial ms):

| Variant | Frame | rs_initial |
|---|---|---|
| as shipped | 88.3 | 56.3 |
| no material textures | 70.0 | 39.8 |
| no alpha cutouts | 87.6 | 55.8 |
| no env | 80.4 | 48.9 |
| no textures, alpha, env | 63.1 | 33.8 |
| RIS-NEE off | 79.2 | 46.6 |
| light mode A | 81.9 | 49.6 |
| maxBounces 1 / 2 | 67.4 / 80.3 | 36.0 / 48.3 |
| Woop | 86.8 | 54.1 |
| interactive textures (mips) | 87.6 | 55.5 |

Isolated traversal throughput on Sponza (T12 perf kernels, 1080p ray counts, BVH2): primary 428–575 Mrays/s, hemisphere
secondary 214, shadow any-hit 811, random incoherent 192. rs_initial at maxBounces 1 traces ≈ 5 rays per pixel
(≈ 2.6 M rays ≈ 13 ms at 200 Mrays/s) but takes 36 ms: traversal is about a third of the path tree. Mode B costs
6.5 ms on a scene without a single area light (its crossing loop finds nothing), and textures 16.6 ms: the path tree
is limited by per-thread state (occupancy) and texture / material work as much as by traversal.

**Measurement hygiene.** Absolute frame times drift between sessions on this machine (Cornell 540p: 16.2 ms at the
baseline, 17.3 ms two hours later with identical code). Every before / after below is therefore an A/B in ONE session
(interleaved jobs; every M8 interactive option can be switched off: `RendererOptions.bvhKind`, `restirKernel`,
`DenoiserSettings.atrousTile`), and the final table (§14) runs the M8 defaults and the pre-M8-equivalent configuration
interleaved in the same session (gate-m8 `--part perf`).

## 3. CWBVH (P-2)

**Builder.** `src/core/bvh/cwbvh.ts`: the binned-SAH BVH2 rebuilt with leaves of ≤ 3 triangles, collapsed to 8-wide with
the SAH-optimal dynamic programme of Ylitie et al. 2017 (the collapse tinybvh's `BVH8_CWBVH` uses; c_node 1, c_prim 0.3:
C(n, 1) = min(leaf, internal), C(n, i) = min(C(n, i − 1), dist(n, i)), bottom-up, a subtree becomes a leaf slot only if its
≤ 3 triangles are contiguous in leaf order), octant-ordered slot assignment, tinybvh's 80 B node layout. The node's child
boxes are quantized conservatively (floor / ceil in exact f64; asserted). **Triangles are the BVH2's own records**
(MT `tris`, Woop `trisW` + primId tail) bit-copied into CWBVH order, so both intersectors, the canonical Woop edge order
and the stable primId are unchanged. Built in the existing BVH Worker (`buildBvhInWorker(…, { cwbvh: true })`).

**tinybvh.** `which emcc` → not installed, so tinybvh was not compiled to WASM (the brief: report rather than install a
toolchain). What tinybvh would have contributed (the DP collapse, octant slots, the node layout) is in the TS port above;
the triangle payload would have been discarded anyway (m8-hwrt-tinybvh-gigi.md §5.2). Not ported: SBVH (`BuildHQ`, not
watertight-safe), the Bittner optimizer. To build tinybvh later: emscripten (`brew install emscripten`), single-threaded
(`-DNO_THREADED_BUILDS`), scalar first; then compare SAH cost / build time against `buildCwbvh`.

**Traversal.** `shaders/bvh/traverse-cwbvh.wgsl`, included by `traverse.wgsl` under the composer define `BVH_CWBVH` (absent
⇒ the BVH2 text, U-M7-BITS): tinybvh's node-group / triangle-group loop with a 16-entry group stack (max depth measured 6
on procedural scenes; overflow and iteration-cap flags as BVH2), four children per vector op in the `q·(2^e·rd) + (p − o)·rd`
form with a slack S = 8ε(|O| + 255|A|) per axis: every computed slab distance is within 4.6ε(|O| + 255|A|) of the exact
one, so no child whose exact quantized box the ray overlaps is culled (the plain tinybvh form is not conservative under
cancellation). Same triangle loop body as BVH2 (MT / Woop, skip prims, `alpha_pass`, the B-shadow plant).
- Lesson: the node words are arbitrary bit patterns; reading them through `array<vec4f>` + bitcast lost NaN payloads on
  Metal (garbage child indices, a GPU hang). CWBVH nodes are bound as `array<vec4u>`.
- The first robust version dequantized every child box to f32 and reused `bvh_slab`: 2× slower than BVH2 on primary and
  shadow rays. The greedy collapse of gap-perf §10.2 gave 4.2 children per node (leaf slots are never merged); the DP
  collapse gives 6.3 (Sponza: 43 304 → 25 161 nodes, SAH 16.6 → 14.8).

**Evidence (T12, `validation/gpu-tests/bvh.gpu.test.ts`, every check on BVH2 and CWBVH, MT and Woop):** 10⁶ random +
edge-aimed rays per scene vs the f64 reference: unexplained 0 on procedural / Sponza / both lattice variants; ray-by-ray
BVH2 vs CWBVH on the GPU (Sponza): 3 094 different closest primIds of which 3 093 are exact-t ties (same t bits; the
remaining one is a classified precision case), 0 t / u bit differences on equal prims, 0 any-hit differences; watertight
icosphere (Woop) 0 escapes; offset spawns 0 self-hits; `visible()` 0 mismatches; alpha hook 0; T12-Q seams 0 cracks (Woop).
CPU: `tests/bvh/cwbvh.test.ts` (structure, record bit copies, a JS mirror of the traversal = brute force).

**Speed.** Isolated kernels (Sponza, 1080p ray counts, Mrays/s, BVH2 → CWBVH, MT): primary 586 → 440, hemisphere
secondary 214 → 211, shadow any-hit 811 → 546, random incoherent 193 → 212. In the frame (540p N=3, same session):
Sponza 90.7 → 83.0 ms (rs_initial 59.3 → 50.4), m6_crossings 26.1 → 27.3 ms. The CWBVH's gain is in the register-limited
megakernels (fewer node fetches per ray), not in isolated traversal; it loses on small scenes. Hence `bvhKind: 'auto'`
in the app: CWBVH from 65 536 triangles (`CWBVH_AUTO_MIN_TRIS`). Every validation path keeps BVH2.

## 4. U-M8-BITS (the bitwise-identity rig)

`validation/gpu-tests/m8-bits.ts` / `m8-bits.gpu.test.ts`: per frame of short chains, hashes of the final reservoirs,
the frame's mean image, the finalize counters and the arena counters; cases: the shipped interactive configuration
(every M6 feature, light mode B, RR, boost, dupmap) on the all-lights fixture (point / spot / rect / disk / sun / emissive
/ env, roughness-0 mirror) with a moving light and camera, full-m6 chains and offline-m6 (Mode B), the interactive
configuration on the normal-mapped smooth package and the glass package, full-m6 on the alpha-foliage package; PT
batches on the same scenes (U-M8-PTBITS). Goldens recorded after the CWBVH text was added (no `BVH_CWBVH` key in any
validation define set: U-M7-BITS still shows the composed text of every validation pipeline equal to the M6 build) and
before any other shader change. Every later bitwise claim in this document is checked against them.

## 5. P-4: the Mode-A text while no rect / disk light exists

In light modes B / A′ the path tree, the shifts and the refresh carry the crossing code (BSDF rays crossing analytic
area lights). Without a rect or disk light it never fires, but costs 6.7 ms on Sponza (rs_initial 56.3 → 49.6 with
light mode A): per-thread state. **U-M8-MODEB:** without area lights, light modes B and A give bitwise equal ReSTIR
chains (interactive and full-m6) and PT images (the mode flags only touch analytic area lights: `mis_w1` tests
`ls.analytic && LP_MODE_A`, `isDelta` first). `RestirKernelOptions.modeBNeedsAreaLights` (on only in
`RestirKernel.interactive`, i.e. the app; validation callers compile the requested text) compiles `RS_MODE_B = 0` while
neither the pending light list nor the current / previous committed light slot holds a rect / disk light; re-evaluated
at frame boundaries (`Renderer.encodeRestir` → `syncLightModeVariant`), a switch is a new pipeline variant (the PT beauty
shows while it compiles, as for any variant) and resets the temporal history. glTF scenes have no area lights (KHR
lights_punctual), so the app's typical case is the Mode-A text. Measured on top of the CWBVH: rs_initial 50.4 → 49.7 ms
(the two savings overlap: both relieve the same register limit).

## 6. Experiments not adopted

| Experiment | Result | Decision |
|---|---|---|
| P-5: `visible(x_{B−1}, x_B)` traced once per path-tree vertex (NEE (b) rc, tree pair, (c-ana) crossings test the same segment) | bitwise equal (U-M8-BITS); rs_initial 49.7 → 49.4 ms, crossings 13.7 → 13.7 | reverted (no measurable gain) |
| P-8: RIS-NEE reuses the last point / spot / sun candidate's ratio (bitwise) | Sponza +0.4 ms, crossings −0.1 ms | reverted |
| Strided-lattice tiled à-trous (one residue class per group, all levels) | step 1 −35 %, steps 4 / 8 2.5–4.7× slower | replaced by P-6 |
| Contiguous tile at steps 2 / 4 | step 2 −0.1 ms on Sponza, +0.03…0.09 ms on Cornell; step 4 2.5× slower (20 KB tile) | step 1 only |
| CWBVH greedy collapse; per-child f32 dequantization | 4.2 children per node; 2× slower | DP collapse + slack form |
| RIS-NEE M = 8 / 1 / off instead of 32 (Sponza) | 76.2 / 73.3 / 72.9 ms vs 82.7 | not changed: M is an estimator setting (env importance quality), the user's choice |
| maxBounces 2 (PLAN §1.10 interactive default) instead of the app's 3 | Sponza 80.3 vs 88.3 ms (baseline) | not changed: a visible default (open item) |

## 7. P-6: tiled step-1 à-trous

`dn_atrous_tile` (`denoise/dn-filter.wgsl`, `DN_TILE_STEP`): the step-1 level reads the 5 × 5 taps of its 8 × 8 group
from a 12 × 12 workgroup-memory tile (tap + colour + luminance-guide texels) instead of 24 texture reads per pixel;
arithmetic, order and skip rules are dn_atrous'. Steps 2, 4, 8 stay on dn_atrous (§6). **Evidence:** U-DN-3c (the tiled
level is bit-identical to dn_atrous: a held frame re-encoded with dn_atrous at that level reproduces hist / atrous /
colour bit for bit), U-DN-3b (N = 4 against the f64 reference). Step-1 level: Sponza 540p 0.69 → 0.42 ms, 720p
1.21 → 0.78 ms; Cornell 540p 0.27 → 0.23 ms. `DenoiserSettings.atrousTile` (default on).

## 8. P-7: plane-major (SoA) reservoirs

PLAN §1.9's production layout asks for SoA planes. `RestirKernelOptions.resLayout: 'soa'` (the interactive kernel;
composer define `RS_RES_SOA`, absent ⇒ the M7 text): plane p of record i at p·n + i. Only the accessors change
(`reservoir.wgsl`, the path tree's three direct words, the debug source, the denoiser's plane-0 stride);
`readReservoirs()` returns AoS words in both layouts. **Evidence:** U-M8-BITS (SoA) = the AoS goldens on all six cases.
Same-session A/B (540p N3): Sponza 85.1 / 85.5 → 84.5 / 84.5 ms, Cornell 17.35 → 16.97 ms.

## 9. Dynamic resolution (app)

`src/app/dynres.ts`: levels 1, 0.875, 0.75, 0.625, 0.5 of the preset's height (multiples of 8); the frame's GPU busy time
from `queue.onSubmittedWorkDone()` (done − max(submit, previous done); no timestamps, Q3); EMA (α 0.15); down when the EMA
exceeds target · 1.08, up when the time predicted at the next level (∝ pixels) stays below target · 0.9; one decision per
30 frames, the 3 frames after a change ignored. A change resizes the internal targets (ReSTIR and denoiser histories
reset once, as any resize). Panel: "dynamic res" (default off) and "target ms" (33); HUD: GPU ms and the scale.
`tests/app/dynres.test.ts`: holds under the target, steps down monotonically to the first level under the target with
±3 % noise (no oscillation), steps back up when the load drops, guards. With the frame cost ∝ pixels, Sponza (≈ 82 ms at
540p) is predicted to settle at 0.625 × 540p (600 × 336); measured there: §14.

## 10. Upscaler (app)

Present's `bicubic` filter (the app default): Catmull-Rom over display-space texels (each texel tonemapped as before),
clamped to the range of the 2 × 2 nearest texels (FSR-style anti-ringing: no halos around HDR edges or fireflies).
Nearest / bilinear stay selectable. A temporal upscaler was not added: the denoiser's output resolve already
accumulates temporally at the internal resolution; a reconstruction across resolutions (TAAU / FSR2-class) is an open item.

## 11. Env compact formats in validation (`ENV_COMPACT_IN_VALIDATION = true`)

Since C-10 every renderer env lookup is `envTexel`'s explicit f32 bilinear over `textureLoad` (exact texel values in
every format), so the compact texture gives the f32 results bit for bit. **Evidence:** ENV-U7c (envRadiance and
envBackground, compact vs rgba32float, 10⁵ uv incl. seam and poles, 0 mismatches: the three validation HDRIs (rgb9e5),
the C0s sun (rgba16float), an RGBE map (rgb9e5), a constant env (stays rgba32float)); `tests/render/env-compact.test.ts`
(no shader samples `texEnv` through the hardware filter); U-M8-PTBITS unchanged. ENV-F keeps measuring the hardware
filter (report only); ENV-U7 tests the bilinear on a forced rgba32float texture (its memory assertion assumed f32).
1k HDRI 8 → 2 MiB, 4k 128 → 32 MiB.

## 12. Items of the PLAN not adopted (deviations)

- **D-M8-1, the 96 B production reservoir.** Not reachable with the fields the M5 / M6 features need without lossy
  radiometric encodings, which data-formats.md §C forbids (W, F, jDen, aux, rcRad, βs, sfxP2, sfxT stay f32). Words per
  record in production: W 1, F 3, seed 2, flags 1, c 1, rc 3 (an env-NEE rc vertex `(ENV, (i<<16)|j, (du16<<16)|dv16)` is
  64 lossless bits besides its entry, whose numbering is per frame), jDen 1, rcWi 1 (oct 2×16), aux 1, rcRad 3, end 3,
  suffix ids 2 (prim + 2×16 dyadic), sfxFlags 1, sfxDir 1 (oct), sfxT 1, βs 3, sfxP2 1 = 30 words (120 B) even with every
  quantization of data-formats §B12 and with the diagnostics plane (nCand, selId, kMargin, endpointId), lobeHist and the
  chunk Σw dropped (none is read by a production pass). The quantizations (dyadic barycentrics / light uv, oct16
  directions) also have to be applied inside the path tree before F, the Jacobian denominators and the suffix are
  computed (rule 14), plus Gate 3 δ, T3 round trips and the oct 2×8 plant — for an estimated 1–2 ms (reservoir traffic)
  of an 82 ms Sponza frame. Done instead: the SoA plane order of the same production layout (P-7, bitwise). Open: a
  lossless 144 B variant (drop the diagnostics plane) and the quantized 120 B variant, if the reservoir traffic ever
  dominates.
- **D-M8-2, f16 path state.** No path-state field qualifies: throughput, radiance, pdfs and Jacobians are radiometric /
  pdf-valued (§C: f32), unit vectors and UVs are never f16 (the user's rule; data-formats §0), ids and flags are already
  packed integers. The register pressure the item targeted is addressed by variants that remove dead code (P-4).
- **D-M8-3, branch-to-select.** Applied where it is a design choice of new code (the CWBVH child test is a branch-free
  four-wide select over the node's children); no edit of the validation traversal / BSDF text (it would change every
  validation pipeline's composed text, and the measured kernels are limited by per-thread state, not branches).
- **D-M8-4, 40 B G-buffer.** Not done: the M1 G-buffer pass is 0.5 ms (Cornell) – 1.8 ms (Sponza, traversal-bound) at 540p;
  halving its writes is worth ≈ 0.1–0.3 ms and changes the denoiser's inputs (oct normals), i.e. a re-run of the M5.5
  evaluation. Open item.
- **D-M8-5, tinybvh WASM.** emscripten is not installed (§3): the DP collapse is ported to TS instead.
- **D-M8-6, the PLAN performance targets on Sponza + HDRI** (§14). Missed by ≈ 2.3× (540p) / 2.9× (720p); met on Cornell,
  m6_crossings and m7_nm_smooth. The path tree (rs_initial, ≈ 50 ms on Sponza) is limited by per-thread state and texture /
  material / RIS work (§2, §6), not by traversal alone; the remaining levers are structural (a wavefront split of the
  path tree, a cheaper RIS target) or quality settings (RIS M, bounces, internal resolution — the dynamic-resolution
  controller holds 33 ms at ≈ 0.625 × 540p).

## 13. Gate M8

`npm run validate -- --milestone M8 [--part core|stageB|perf]` (`validation/harness/gate-m8.ts`):
- **core:** typecheck; cpu lane (U-M7-BITS with the M8 code: every validation pipeline composes to the M6 text);
  python tests; Chrome: m8-bits (U-M8-BITS AoS + SoA, PTBITS, MODEB, P-4 switching), bvh (T12 incl. CWBVH and the hit
  equivalence), denoiser (U-DN-3b / 3c), env / env-format / env-sampling (ENV-U7c), normal-map; the M5.5 and M7 app smokes.
- **stageB:** CWBVH end to end: ReSTIR offline-m6 on the CWBVH (Woop) vs our PT on BVH2, δ 0.2 % / 1 %, on
  m7_nm_smooth_256 (Mode A), m7_nm_smooth_B_256 (Mode B), m7_nm_env_256 (HDRI); gate-m7's sequential units (pilot sizing,
  one disjoint-seed re-run per failed unit); T16 also requires `meta.config.bvh = 'cwbvh'`.
- **perf:** the targets, M8 vs the pre-M8-equivalent configuration, same session (recorded, not gating).

`npm run validate -- --all` runs M0 … M8 in order, each in its own process (PLAN §5 M8 exit).

## Changelog

- **P-1** (perf driver): `validation/harness/run-perf.ts` / `perf-run.ts`; baseline recorded (§2).
- **P-2** CWBVH (§3): builder, traversal, T12 on both structures, `bvhKind` (validation BVH2, app 'auto').
- **P-3** U-M8-BITS rig and goldens (§4).
- **P-4** Mode-A text without area lights in the interactive kernel (§5), U-M8-MODEB.
- **P-5** segment-visibility memo: tried, reverted (§6).
- **P-6** tiled step-1 à-trous (§7), U-DN-3b / 3c.
- **P-7** SoA reservoir planes in the interactive kernel (§8).
- **P-8** RIS delta-candidate memo: tried, reverted (§6).
- **P-9** dynamic resolution and the bicubic upscale (§9, §10).
- **P-10** `ENV_COMPACT_IN_VALIDATION = true` (§11), ENV-U7c.
- **P-11** gate M8 and `validate --all` (§13).
- **D-M8-1 … D-M8-6** deviations (§12).
