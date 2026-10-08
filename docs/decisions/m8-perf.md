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

## Changelog

- **P-1** (perf driver): `validation/harness/run-perf.ts` / `perf-run.ts`; baseline recorded (§2).
