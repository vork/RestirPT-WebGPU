# Validation scenes, sizing and coverage (plan §7.2, §7.3; M3a Gate 2)

`npm run validate -- --milestone M3a` runs `validation/harness/gate-m3a.ts`, which covers:
- **Gate 0.** Unit tests (cpu lane, python stats) and the Chrome GPU tests: T1, T13b (`rng`), T8 (`bsdf`), T9
  (`lights`), T10/C0f, U11, the C0 analytic probes, offline split-dispatch equality and the plant plumbing (`pt`),
  `primary` and `emission`. It also runs **T13 camera registration**: the PT kernel renders C0a at 512², 640×360,
  360×640 and far/recentred, plus C0b, and `marker_check.py` checks the result.
- **Gate 1, M3a items.** Planted biases in *our* PT are compared against Cycles (i) using `compare.py --calibrate
  --planted`, with ten half-size repeats. Each plant must be detected in ≥ 9/10 repeats, and the Cycles A/A control must
  pass in ≥ 9/10. Separately, a full 16-vs-16 comparison must not pass. The plants are:
  - every emitter ×1.01, via the `PtParams.emitScale` debug uniform;
  - "drop 1% of the paths at vertex 2" without compensation, via `dropProb`/`dropBounce`.

  Gate 1 also runs an A/A test on two of our PT seed sets (1001 vs 2002).
- **Gate 2, Stage A.** Every scene below is compared as our PT batches against cached Cycles seeds. The per-scene
  settings are in `M3A_G2` in `gate-m3a.ts`. Analytic expectations are checked on **both** renderers with
  `validation/tools/analytic_check.py`:
  - C0f: the closed-form constant;
  - C0c, C0e, C0m: a supersampled f64 image.

## Scenes (generators: `validation/scenes/make-m3a.ts`, `make-cornell-i.ts`, `make-spot-c0d.ts`, `validation/blender/calib_scenes.py`)

All scenes are flat shaded and Mode A: analytic lights are NEE-only and never camera-visible. No scene has an env. Our
PT uses seed 7. A failed unit is re-run once, with our seed 100007 and Cycles seeds 100…, as compare.py `--rerun-of`.

| Unit | Content | b | Res | Cycles spp×K | PT spp×B |
|---|---|---|---|---|---|
| C0a | emissive squares (camera registration) | 0 | 512² | 256×16 | 256×16 |
| C0b | emission units | 0 | 512² | 16×16 | 16×16 |
| C0c | point light over a Lambert floor (analytic) | 0 | 256² | 1024×16 | 2048×16 |
| C0d | spot, 60°, blend 0.15, tilted | 0 | 512² | 1024×16 | 1024×16 |
| C0e | one-sided rect tilted 55° (black region behind; analytic) | 0 | 256² | 1024×16 | 4096×16 |
| C0f ×4 | closed emissive Lambert cube, b ∈ {0, 1, 3, 7} (analytic L_e(1−ρ^{b+2})/(1−ρ)) | b | 256² | 1024×16 | 2048×16 |
| C0g | BSDF furnace: V1 GGX r 0.05/0.2/0.5/1.0 + Lambert inside an L = 1 emissive sphere | 0 | 512×256 | 1024×16 | 2048×16 |
| C0l | sun at 40° elevation + shadow box | 0 | 256² | 1024×16 | 2048×16 |
| C0m | disk light (analytic, 512-gon) | 0 | 256² | 1024×16 | 4096×16 |
| C0n | rect lights with spread 30° and 90° | 0 | 256² | 1024×16 | 8192×16 |
| (i) | Cornell, rect light | 3 | 512² | 1024×16 | 1024×16 |
| (ii) | Cornell, point light | 3 | 512² | 1024×16 | 1024×16 |
| (iii) | spot grazing a wall + floor + box | 3 | 512² | 4096×16 | 4096×16 |
| (iv) | Cornell + emissive icosphere + small rect | 3 | 512² | 1024×16 | 1024×16 |
| (v) sharp | V1 GGX r 0.05/0.1, lit by emissive quads, **tile δ 3%** (plan) | 3 | 512² | 1024×16 | 1024×16 |
| (v) | V1 GGX r 0.2/0.3/0.5/0.8, rect lights | 3 | 512² | 1024×16 | 1024×16 |
| (v) V2 | Principled: metallic 0/1 × r 0.2/0.5/0.8 × specular level 0.5/0.8 (tint at 0.8), IOR 1.5 | 3 | 512² | 1024×16 | 1024×16 |
| (vii) | textures: sRGB repeat + KHR_texture_transform, clamp, mirror, nearest, metal/rough, emissive texture | 3 | 512² | 1024×16 | 1024×16 |
| (viii) | Cornell USD through our USD loader → package | 3 | 512² | 1024×16 | 1024×16 |
| (x) | 64 point + 8 rect + 4 disk (spread 60°) + 4 spot + sun + emissive mesh, exposure ≠ 0 on 3 lights | 2 | 512² | 4096×16 | 4096×16 |
| (xi) | contact: wall corner, box in the corner and on the floor, 2 mm slab gap, 3 mm wall panel | 3 | 512² | 1024×16 | 1024×16 |
| (xii) | alpha MASK leaf cards (α ∈ {0, 1}) over a floor | 3 | 512² | 1024×16 | 1024×16 |
| ix-a…g | keyframes 0, 8, …, 48 of 48 @ 24 fps (see below) | 3 | 256² | 2048×8 (ix-c 4096×16) | 2048×8 (ix-c 4096×16) |

The ix sequences are made with `src/core/scene/animation.ts` and exported as resolved `frames[]` with CONSTANT keys:
- ix-a: moving point light;
- ix-b: moving and tilting rect light;
- ix-c: sweeping spot light;
- ix-d: moving camera;
- ix-e: add, remove and intensity steps; a light that is absent has power 0 through a step key;
- ix-f: camera teleport by a step key at frame 24, an FOV ramp from 30° to 60°, and a moving point light with a
  power ramp;
- ix-g: rotating sun over an open courtyard.

The ix sequences use K = B = 8 per frame, as the task allowed. `min_replicates` is 8, and the change is recorded in
each report's `notes`. ix-c needed K = 16.

**Suite FWER.** n_units counts every scene, every ix frame, the two plants and the A/A test, times {Y, R, G, B}.

**Recorded deviations.**
- (v) sharp uses tile δ 3%, which is plan §5 M3a (v). compare.py records it in `notes.delta_loosened`.
- Every unit uses the numerical floor `num_eps = 1e-4` (`docs/decisions/cycles-deviations.md` D2), recorded in
  `notes.numeric_floor`.

No global δ is loosened, and no aggregate is enlarged.

## Coverage: light types and editor properties → Stage-A scenes

Stage B (ReSTIR vs PT) arrives with M4.

| Feature | Stage-A scenes (M3a) |
|---|---|
| point | C0c, (ii), (x), (xi), ix-a, ix-e, ix-f |
| spot (size, blend) | C0d, (iii), (x), ix-c, ix-e |
| rect | C0e, (i), (iv), (v), (v) V2, (vii), (viii), (x), (xi), (xii), ix-b, ix-d, ix-e, ix-f |
| disk | C0m, (x) |
| sun | C0l, (x), ix-g |
| spread < 180° | C0n (30°, 90°), (x) (disks at 60°) |
| emissive triangles (MIS) | C0a, C0b, C0f, C0g, (iv), (v) sharp, (vii) (textured emission), (x) |
| exposure ≠ 0 | (x) |
| colour | every scene: static colour. An animated colour cannot be represented in `frames` and is a hard error |
| position / rotation tracks | ix-a, ix-b, ix-c, ix-f, ix-g (camera: ix-d, ix-f) |
| power tracks (step / linear) | ix-e, ix-f |
| visibleToCamera | C0o, **deferred to M3b** (Mode B only) |
