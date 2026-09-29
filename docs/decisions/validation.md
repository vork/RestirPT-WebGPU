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

---

# M3c: environment lighting (plan §5 M3c, §7.2 env scenes, §7.4 M3c)

`npm run validate -- --milestone M3c` runs `validation/harness/gate-m3c.ts`:
- **Gate 0.** Typecheck, cpu lane (`tests/env/env-importance.test.ts`: ENV-U3 CPU part, ENV-U5, ENV-U8, env as a light),
  python stats tests, scene-package determinism (`make-m3c.ts` twice, byte-identical), Chrome GPU tests
  (`validation/gpu-tests/env-sampling.gpu.test.ts`: ENV-U3 χ² + realized-pdf identity, ENV-U4, ENV-U6; regressions
  env/lights/pt), `budget.json` env rows (`validation/tools/budget_env.py`) and the app smoke
  (`validation/harness/m3c-app-smoke.ts`: env panel, Worker tables, env debug views, splat χ²).
- **Gate 2, Stage A** (our PT ≡ Cycles, δ 0.5% global / 2% per 32² tile, Y/R/G/B, suite FWER) on the scenes below,
  Mode A. Analytic expectations (`validation/tools/env-expected.ts`: f64 ray casting + midpoint quadrature of the
  8-bit-fraction bilinear map, 16×16 subsamples per pixel) are checked on **both** renderers with
  `analytic_check.py --expected-image`.
- **Env plants** (compare.py `--calibrate --planted`, ≥ 9/10 half-size repeats, Cycles A/A control ≥ 9/10, and the full
  comparison must fail) and **negative controls** (must pass Stage A), plus an A/A of two of our seed sets on C0r.

Scenes are generated (not committed) into `validation/out/m3c/scenes/` by `validation/scenes/make-m3c.ts`, because the
HDRI scenes embed the downloaded Poly Haven texels (`validation/assets/fetch_hdris.ts`, pinned SHA-256). Cycles
references are cached in `validation/out/m3c/refs/`. `--light-mode B` and `--glass` write the Mode-B and glass variants
(after the M3b merge).

| Unit | Content | b | Res | Cycles spp×K | PT spp×B | Tier |
|---|---|---|---|---|---|---|
| C0q lambert ×3 (+ `_bg`) | Lambert sphere ρ 0.8, constant L = 1 (64×32 texture; `_bg`: Blender constant Background) | 0/1/3 | 256² | 1024×16 | 1024×16 | tight |
| C0q quad (+ `_bg`) | Lambert quad seen from above | 1 | 256² | 1024×16 | 1024×16 | tight |
| C0q ggx02 / ggx05 (+ `_bg`) | V1 GGX sphere F ≡ 1, α 0.2 / 0.5 (L·E_ss by quadrature) | 1 | 256² | 1024×16 | 1024×16 | tight |
| C0q openbox (+ `_bg`) | ρ = 1 open box (no closed form) | 13 | 256² | 1024×16 | 1024×16 | tight |
| C0r irradiance / mirror | overcast_soil_puresky γ 0.6: Lambert sphere, mirror sphere (V1 r = 0) | 1 | 256² | 1024×16 | 1024×16 | tight |
| C0s 45 / seam / top, NEE | 512×256 map, texel 1e4 at 45° / on the u seam / in the top row; Lambert plane ρ 0.5 | 1 | 256² | 1024×16 | 1024×16 | tight |
| C0s …, NONE | same, env NEE off (Cycles sampling_method NONE) | 1 | 256² | 16384×16 | 16384×16 | tight |
| (xiii) | V1 GGX r 0/0.05/0.15/0.19/0.21/0.3/0.5 + Principled gold r 0.25, studio_small_09 γ 0.3 | 3 | 512×256 | 1024×32 | 1024×32 | heavy-tail |
| (xiv) overcast b 1/3/7, + rect b 3 | Cornell without ceiling, overcast_soil_puresky γ 0.9 (+ the 4 W rect light) | 1/3/7 | 512² | 1024×16 | 1024×16 | tight |
| (xiv) kloofendal, + rect | same under kloofendal_48d_partly_cloudy_puresky, sun turned to shine in | 3 | 512² | 1024×32 | 1024×32 | heavy-tail |

Extra units: the two Cycles C0q world variants against each other (texture + NEE vs constant Background, BSDF-only;
the `_bg` references use seeds 200.. so the two are independent).

**Sizing.** The C0s NONE units use 16× the spp of the NEE units: at 4096 spp the pilot had a tile MDB of 2.0% and a
replicate multiplier of 2.7 (BSDF-only hits the 1.06e-4 sr texel with p ≈ 2.4e-5). Heavy-tail units double K and B
(plan §7.2). No δ is loosened and no aggregate is enlarged.
