# Validation scenes, sizing and coverage (plan §7.2, §7.3; M3a and M3b Gate 2)

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
| visibleToCamera | deferred to M3b (Mode B only): covered by C0o, see the M3b section below |

# M3b: glass and Mode B (plan §5 M3b, §7.2; gap-glass §7)

`npm run validate -- --milestone M3b` runs `validation/harness/gate-m3b.ts`:
- **Gate 0.** typecheck, cpu lane (`tests/material/glass-ref.test.ts`: U-G1…U-G10 CPU parts, the §3 albedo tables,
  spurious-region integrals, Tier-2 multipliers), python stats tests, package freshness of `make-m3b.ts`, and the
  Chrome GPU suites: `glass.gpu.test.ts` (U-G1…U-G10: Fresnel, eval/pdf vectors, full-sphere χ² against f64 expected
  counts of the valid-only pdf with the B-spur / η×1.01 / r×1.04 controls, albedo, weight/support consistency, q
  vectors, delta, backfacing, BTDF non-reciprocity, Λ(−c) = Λ(c)), `pt-glass.gpu.test.ts` (C0h/G1 closed forms, the
  C0i η² detector, U9 A ≡ B ≡ A′ on a delta-free scene with A′ bit-identical to A, the mirror positive control
  A < B with A′ ≡ B, U10 pass-through replay), and the M3a suites `bsdf`, `pt`, `lights` as regression.
- **Gate 1, M3b items.** The gap-glass §7 plants in *our* PT vs Cycles, ten half-size repeats each (detected ≥ 9/10,
  Cycles A/A control ≥ 9/10, full 16-vs-16 must fail): B-η (1/η² BTDF; on C0i, since the scaling cancels across the two
  interfaces of a slab), B-tint (C for √C), B-side (η not inverted on backfaces), an un-compensated P_R = 0.5 (all on
  G1 Principled, C 0.5, N 3), B-shadow (glass does not occlude; on C0k). B-spur is the Gate-0 χ² control. Our A/A on G1.
- **Gate 2, Stage A.** TOST δ 0.5% global / 2% per 32² tile, Y/R/G/B, suite FWER, one disjoint-seed re-run. Closed
  forms are checked on **both** renderers (`analytic_check.py` kinds `glass-slab`, `glass-shadow`).

Cycles references render at ≥ 4096 spp (cycles-deviations.md D4). Our PT uses seed 7 (re-run 100007).

| Unit | Content | b | Res | Mode | Cycles spp×K | PT spp×B |
|---|---|---|---|---|---|---|
| C0h | smooth Glass-node slab over an L_e = 1 plane (closed form (1−F)²ΣF^{2j}) | 4 | 256² | A | 4096×16 | 2048×16 |
| G1 Principled | smooth slab in an L = 1 furnace, C 0.5 (closed form) | 3 | 256² | A | 4096×16 | 2048×16 |
| G1 Glass node | same, colour 0.8 (closed form) | 2 | 256² | A | 4096×16 | 2048×16 |
| C0i / G2 | emitter inside a smooth glass box (η² detector, closed form) | 1 | 256² | A | 4096×16 | 2048×16 |
| C0j | rough glass spheres (Glass node r 0.1/0.25/0.5/1, Principled t 1 coloured) in an L = 1 furnace | 8 | 512×256 | A | 4096×16 | 2048×16 |
| C0k | glass shadow: floor inside the cube's shadow is exactly 0 | 0 | 256² | A | 4096×16 | 2048×16 |
| G3 | rough slabs r 0.1/0.25/0.5/1, reflection only | 0 | 512×256 | A | 4096×16 | 2048×16 |
| G4 | the same slabs, N 3 (inside η = 1/ior, TIR) | 3 | 512×256 | A | 4096×16 | 2048×16 |
| G5 | point light over smooth / rough slab / rough pane | 3 | 256² | A | 10240×16 | 10240×16 |
| G5b | rect light over the panes, **approximate tier** (D5) | 3 | 256² | B | 4096×16 | 2048×16 |
| G6 | caustic through a smooth glass icosphere | 4 | 256² | B | 4096×16 | 8192×16 |
| G6-neg | the same in Mode A (no caustic) | 4 | 256² | A | 4096×16 | 2048×16 |
| G7 | Principled t 0.25/0.5 × m 0/0.25, coloured C, tint, level | 4 | 512² | A | 4096×16 | 1024×16 |
| G7 furnace | the same materials in an L = 1 furnace | 0 | 512×256 | A | 4096×16 | 2048×16 |
| G8 | Cornell + rough glass icosphere (r 0.3) + smooth cube | 8 | 512² | A | 10240×16 | 10240×16 |
| G9 | bubbles: Principled IOR 0.75, smooth and r 0.25 | 6 | 256² | A | 16384×16 | 16384×16 |
| G10 | coloured Glass node + Refraction node | 6 | 256² | A | 20480×16 | 20480×16 |
| (vi) A | glass sphere + roughness-0 mirror + glossy sphere, rect + disk (Cycles MIS off) | 6 | 512² | A | 4096×16 | 1024×16 |
| (vi) B | the same, Cycles MIS on | 6 | 512² | B | 4096×16 | 1024×16 |
| (vi-B) | rect light in a roughness-0 mirror floor (plan §1.4 positive control) | 2 | 256² | B | 4096×16 | 8192×16 |
| C0o | camera-visible rect + disk (spread 120°) + a hidden rect | 2 | 256² | B | 4096×16 | 2048×16 |

**Sizing.** The pilot (`validation/out/m3b-gate-20260929-025921`, Cycles 1024×16) measured the replicate multiplier
the worst 32² tile needs (plan §7.3): G5 5.67, G8 6.99, G9 9.39, G10 11.34; the per-sample variances of the two
renderers were within 0.8–1.05× of each other, so both sides were scaled, leaving ×1.3–1.4 margin. No aggregate is
enlarged and no δ is loosened. (vi-B) got a larger light (1.0 × 0.7) and 4096/8192 spp after the first pilot.

**Recorded deviations.** D4 (Cycles spp ≥ 4096), D5 (G5b approximate tier). Every unit uses `num_eps = 1e-4` (D2).

**Coverage additions (M3b).**

| Feature | Stage-A scenes (M3b) |
|---|---|
| Glass node (GGX), smooth / rough | C0h, G1, C0i, C0j, C0k, G3, G4, G5, G5b, G6, G8, G10, (vi) |
| Refraction node | G10 |
| Principled transmission (+ metallic, coloured C, tint, level) | G1, G7, G9 |
| η_side < 1 from the front (IOR < 1) | G9 |
| delta glass chains, caustics (Mode B) | G6, (vi) B |
| roughness-0 mirror, light seen in the mirror | (vi), (vi-B) |
| Mode B (analytic lights hittable, MIS) | G5b, G6, (vi) B, (vi-B), C0o |
| Mode A′ | GPU U9 / mirror control (A′ ≡ B), app smoke |
| visibleToCamera | C0o (M3a deferral closed) |

**Result.** `npm run validate -- --milestone M3b` passed 96/96 in 5382 s (`validation/out/m3b-gate-20260929-044220`):
all 21 Stage-A units pass on the first seed set (no re-run), closed forms agree on both renderers (C0h, G1 ×2, C0i
within 0.001%; C0k 5545 shadow pixels exactly 0 in both), the five glass plants are detected 10/10 with the Cycles
A/A control 10/10 (B-η −55.6%, P_R 0.5 +2.5%, B-tint −26.7%, B-side +0.64%, B-shadow +27.3%), and our A/A passes.
G9 reports a replicate multiplier of 1.079 for its worst tile (tile MDB 1.20% < δ_tile 2%); it passes, and the next
budget change should raise G9 to ≥ 18432 spp per side. The app smoke (`validation/harness/m3b-app-smoke.ts`) passes
11/11 (`validation/out/m3b-app-smoke-20260929-044348`).

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
| (xiv) overcast b 1/3/7, + rect b 3 | Cornell without ceiling, overcast_soil_puresky γ 0.9 (+ the 4 W rect light; b = 1: strength 1.3, tint (1, 0.9, 0.75)) | 1/3/7 | 512² | 1024×16 | 1024×16 | tight |
| (xiv) kloofendal, + rect | same under kloofendal_48d_partly_cloudy_puresky, sun turned to shine in | 3 | 512² | 1024×32 | 1024×32 | heavy-tail |

Extra units: the two Cycles C0q world variants against each other (texture + NEE vs constant Background, BSDF-only;
the `_bg` references use seeds 200.. so the two are independent).

**Sizing.** The C0s NONE units use 16× the spp of the NEE units: at 4096 spp the pilot had a tile MDB of 2.0% and a
replicate multiplier of 2.7 (BSDF-only hits the 1.06e-4 sr texel with p ≈ 2.4e-5). Heavy-tail units double K and B
(plan §7.2). No δ is loosened and no aggregate is enlarged.

**Findings while closing M3c.**
- **Path RNG seed was 32 bits** (our defect, fixed): C0s seam with env NEE off was −0.107% / −0.084% vs the f64
  quadrature (z ≈ −3.6 on both runs) while Cycles and our NEE variants matched. A GPU probe of the same BSDF-only
  estimator reproduced −0.13% (z −9) for the seam texel and −0.07% for a texel at +X, but was unbiased when the BSDF
  dims came from a 96-bit-input hash. With a 32-bit `initSeed` all paths are one of 2³² dimension vectors: a fixed
  quadrature, whose error for a single-sample σ_rel ≈ 36 is ≈ 0.05%. `initSeed` is now 64 bits (pcg3d(...).xy) and the
  slot hash is pcg4d (math.md#rng-layout); the jitter keeps initSeed.x, so camera rays are unchanged.
- **Cycles light tree** biases env + analytic-light scenes by ~7e-5 (cycles-deviations.md D6); the (xiv) rect-light
  references use the light distribution.

## Coverage: environment features → Stage-A scenes (M3c)

| Feature | Stage-A scenes |
|---|---|
| NEE_ENV + BSDF_ENV MIS (env alone, P(env) = 1) | C0q (texture), C0r, C0s NEE, (xiii), (xiv) without light |
| env NEE off (sampling_method NONE) | C0s NONE (×3), C0q `_bg` (Cycles side BSDF-only) |
| P(env) proxy + clamp with other emitters | (xiv) rect variants; app smoke (point light) |
| seam / pole-wrap rows | C0s seam, C0s top; ENV-U4 |
| rotation γ | C0r (0.6), (xiii) (0.3), (xiv) (0.9 / sun-aligned) |
| delta lobe after env (ω2 = 1) | C0r mirror, (xiii) r = 0 |
| V2 metal under an HDRI | (xiii) |
| strength / tint | (xiv) overcast b = 1 (strength 1.3, tint (1, 0.9, 0.75)); strength ×1.0075 plant; ENV-U8 (pmf) |
| Mode B / glass with env | deferred to the post-M3b merge (`make-m3c.ts --light-mode B`, `--glass`) |

---

# M4: ReSTIR PT core, Gate 3 Stage B (restir-api.md §6.3/§6.4, PLAN §5 M4, §7.1, §7.3, §7.4 M4)

`npm run validate -- --milestone M4` runs `validation/harness/gate-m4.ts`:
- **Gate 0.** Typecheck; cpu lane (`tests/restir/*`: U-RES-1 layout, rc-dual / T3-D CPU part, MIS / pairing / queue /
  npz, `gate-m4-config.test.ts`, plus every M1–M3 test); python stats tests; `make-m4.ts` determinism (twice,
  byte-identical); the Chrome GPU suites `restir-initial` (WP-A), `restir-shift` (WP-B), `restir-spatial` (WP-C),
  `restir-debug` (WP-D) and the M3 regressions `pt`, `bsdf`, `lights`, `env-sampling`, `glass`, `pt-glass` (the
  `length1.wgsl` / env-sample refactors are bit-identical); `budget.json` M4 rows; the M4 app smoke
  (`validation/harness/m4-app-smoke.ts`).
- **Gate 3, Stage B (our ReSTIR ≡ our PT).** compare.py stage B: TOST α = 0.01, δ 0.2% global / 1% per 32² tile,
  Y/R/G/B, Šidák tiles, χ²_red, mean-t and KS/AD at the suite FWER α_u = 1 − 0.99^{1/n_units} (n_units = 280),
  `num_eps` 1e-4 (two f32 implementations), `min_replicates` 16 (heavy-tail 32). Per scene the **ladder** runs rung 3.1
  (preset `initial`: S = 1, no RR) → 3.1b (`initial-rr`: RR from bounce 1, D11) → 3.2 (`offline`: S = 32, 3 rounds ×
  6 slots, R = 10 px, RR off); it stops at the first rung that fails after its re-run and the later rungs are "not run".
  A failed unit is re-run once on disjoint seeds on both sides (compare.py `--rerun-of`). Mode A everywhere; env NEE
  per package; maxBounces = the package's on both sides.
- **Special units.** The **ensemble unit**: C0q(d) rung 3.2 in ensemble mode (E = 16 members of 256² in one atlas,
  `ensemble.npz` rows r = t·E + m) against the same PT reference. The **2022-criteria unit**: (i) with preset
  `criteria2022`.
- **Planted controls** (compare.py `--calibrate --planted`: ≥ 9/10 half-size repeats detected, the PT A/A control
  ≥ 9/10, and the full comparison must fail): omitted spatial Jacobian (`RSF_PLANT_NO_J`, rung 3.2) on (i) and (v) V1;
  marginal pdfs in J (`RSF_PLANT_MARGINAL_J`, the U-11 negative control) on (v) V2. The synthetic **W × 1.003** plant
  (compare.py stage-B default) runs with `--calibrate` on a (i) rung-3.2 ReSTIR run, whose A/A re-splits must also
  pass at the nominal rate. **A/A**: two ReSTIR seed sets (5002 vs 5003) on (i) rung 3.2 must pass Stage B.
  Calibration sides that are split into halves or compared at equal size are rendered at 4× the unit's size (the A/A
  pair; the rendered plants' PT references, seed 4201): halves of a set sized to SE_Δ ≈ target are 1.3–1.6× above the
  target and the half-size A/A control would then fail TOST at the worst of 256 tiles. The planted runs stay 1×.
- **T15/T16 per run** (restir-batch-run.ts errors + gate-m4.ts `t16Problems`): NaN/Inf = 0, negatives = 0, BVH
  overflow / itercap = 0, `RSC_CAND_NONFINITE = SHIFT_NONFINITE = PENDING_LEFT = SLOT_MISMATCH = W_NONFINITE = 0`,
  every queue overflow = 0, `BASE_JDEN_INVALID` ≤ 1e-5 of the candidates, no submit above 200 ms; unbiased preset
  (plants only where named), internal scale 1, no denoiser / upscaler, readback of the linear radiance, jitter
  iid-per-run, maxBounces = the PT reference's, same scene bytes / resolution / env NEE, rung 3.2 executed all 3
  spatial rounds.

**Stage-B runs** are sequential independent frames (D14): a ReSTIR batch is the mean of F frames `t ∈ [b·F, (b+1)·F)`
(`run-batches.ts --kernel restir --preset <rung> --spp F`), packed into submits by `BatchAccumulator.runBatchUnits`
(target 50 ms, learned ms per costHint; the chunking probe sets the tree chunk / row band, Changelog E2). The PT side
is `--kernel pt` (RR off), seed 4001 (re-run 104001); ReSTIR seed 4002 (re-run 104002).

**PT references are cached** in `validation/out/m4/ptrefs/`, keyed by package bytes + spp × B + seed + a hash of the
PT's TS import closure (`batch-run.ts`) and WGSL include closure (`passes/pt.wgsl`); one reference serves all rungs of
a scene, so re-runs with unchanged PT code render only the ReSTIR side. Pilots are cached in `validation/out/m4/pilots/`.

**Sizing** (PLAN §7.3; Changelog E7). Per scene a pilot of the PT (128 spp × B) and of each rung (128 frames, 3.2: 8
frames, × B; B = 16, heavy-tail 32) gives the per-sample SD of every 32² tile and of the global mean of Y/R/G/B. The
PT sample count (shared by the three rungs) and each rung's frame count minimise GPU time subject to
`SE_Δ ≤ δ·D/(t_{0.99,B−1} + z_{1−0.005/m})` on every aggregate (D = max(R̄, 0.05·R̄_image), the dark-tile rule), then
× 1.25 and rounded per batch to m·2^k, with floors of 256 spp (PT), 128 frames (3.1/3.1b) and 8 frames (3.2) per
batch so every batch mean averages enough samples (the smooth env scenes otherwise size down to 1 frame per batch). A side needing more than 60 min would switch that unit to 64² tiles (recorded
as `aggregate_enlarged` in report.json); δ is never loosened. `gate-m4 --pilot-only --write-budget` writes the sizes
to `budget.json` `m4_entries`.

**Sizing (full gate, `budget.json` `m4_entries`).** No unit needed 64² tiles; the largest sides are (x) many lights
(PT 57344 spp × 16 ≈ 27 min; rungs 3.1/3.1b 81920 frames × 16 ≈ 57/55 min), (iii) spot (PT 57344 spp × 16 ≈ 17 min)
and the 3.2 rungs of (v) sharp / (v) V2 (≈ 15–16 min). The smooth env scenes sit at the per-batch floors.

**Result.** `npm run validate -- --milestone M4` on a54ac2d (WP-B final 6e02a93; units after 00:55 CEST ran on 0ab5fa4,
which changes only the interactive `RestirFramePass.encode`, not the batch path) passed 220/221 in 44 882 s
(`validation/out/m4-gate-20260929-190611`):
- Gate 0: every suite green (restir-initial 24/24, restir-shift 23/23 with LOGIC = 0 / FP-BOUNDARY = 0 in every T3 run
  and the t3_rare none-tri bins at ≥ 10⁷, restir-spatial 14/14, restir-debug 8/8, M3 regressions); the M4 app smoke
  failed 29/31 (interactive spatial reuse lost under timestamp writes, WP-C C8) and passes 31/31 when re-run on 0ab5fa4
  (`validation/out/m4-gate-20260929-190611-app-smoke-rerun`).
- Gate 3: 63/63 ladder units pass on the first seed set (no re-run): |Δ_Y| ≤ 0.0048%, worst tile ≤ 0.33%, tile MDB
  max ≤ 0.58%; the ensemble unit (C0q(d) 3.2, E = 16) and the 2022-criteria unit ((i)) pass.
- Plants detected 10/10 with the PT A/A control 10/10 and the full comparison failing: omitted J on (i) (Δ_Y +0.174%)
  and (v) V1 (Δ_Y +0.007%, 63 Šidák-rejected tiles, worst tile +9.3%), marginal-pdf J on (v) V2 (13 Šidák tiles);
  synthetic W × 1.003 10/10 with A/A re-splits 20/20; ReSTIR A/A (5002 vs 5003) passes.
- Production counters summed over every Stage-B run: RSC_CAND/SHIFT/W_NONFINITE, PENDING_LEFT, SLOT_MISMATCH,
  BASE_JDEN_INVALID, BVH overflow/itercap, queue overflow, finalize NaN/Inf and negatives are all 0
  (1.9·10¹¹ accepted pairs, replay fraction 2.7%).

### M4 gate provenance (merge into main, 2026-09-30)

- **What was gated.** The full gate `m4-gate-20260929-190611` passed 220/221. The one failure, m4-app-smoke, passes 31/31 after C8. The gate ran on a54ac2d; Stage B ran on 0ab5fa4.
- **Commits merged afterwards:**
  - m4-appfix: regression test only.
  - m4-t3fault (defeba3): the Metal Q2 workaround, which keeps the NEE state compact in `shift_hybrid`.
- **Bit comparison of d456fac vs the merged ad759b4** (seed 777, 2 × 8 frames):
  - `initial`: bit-identical on (i), (iv) and (x).
  - `offline`, where spatial shifts run: identical sample selection. Pixel values differ only by f32 rounding: at most 3.9e-7 relative, image means within 1.2e-10 relative. That is 4 orders of magnitude below the Stage-B δ (0.2% global).
- **Conclusion:** the M4 Stage-B results stand for the merged code without a re-run.

# M5: temporal reuse and dynamics, Gate 3 rungs 3.3–3.6 (restir-temporal-api.md §6, PLAN §5 M5, §7.1, §7.3, §7.4 M5)

`npm run validate -- --milestone M5 [--part core|static]` runs `validation/harness/gate-m5.ts` (owner T-E):
- **Gate 0.** Typecheck; cpu lane (`tests/restir/*` incl. `gate-m5-config.test.ts`, tmis, tqueue, refresh-ref,
  light-maps, config-hash, frame-state); python tests (`validation/tools/tests`, incl. `test_dynamic.py` for
  dynamic.py / dyn_masks.py / plant_sign.py); `make-m5.ts` determinism (twice, byte-identical, and the committed
  packages equal the generator's output); the Chrome GPU suites `restir-tframe`, `restir-temporal`, `restir-refresh`,
  `restir-debug` and the M4/M3 regressions (one GPU-lock hold per suite); `budget.json` M5 rows; the M5 app smoke.
- **Scenes** (`validation/scenes/make-m5.ts`; env packages in `validation/out/m5/scenes`): the static subset `m5s_*`
  (the M4 packages of (i), (iii), (v) V1, (vi) A, (x), (xii), C0q(d), (xiv) overcast + rect with render 256², every other
  byte identical; TD28); 16 sequences `ixs_*_256` with dense frames and a `sequence` block (TD27, §6.2); the U8 scenes
  `u8_c0{c,d,e}_*_b{0,1}` (TD30). (xiii) and the kloof scenes are reported, not gating (Q2).
- **Chains** (TD25, restir-temporal-api.md Changelog E-1). E = 16 chains per 256² atlas; a chain is reset at t = 0 and
  follows the package's frames; chain id = memberBase + member; one chain run serves every test frame of its unit. At a
  test frame the atlas `rsFrame` is copied and reduced on the host (f64) into `f<t>/ensemble.npz` (rows = chains, tiles
  16/32/64, global, mask regions, per-pixel moments). Submits never span a frame boundary (TD26). GPU-lock chunks are
  whole batches (`--batch-offset`), merged by `dynamic.py merge-npz`.
- **Rungs.** 3.3 (`temporal`) and 3.4 (`full`) on the 8 static scenes at t ∈ {1, 24} of 25-frame chains, stage B
  (0.2 % global, 1 % per 32² tile); 3.5 = the per-chain mean of frames 32…287 on (i), (v) V1, (xiv) overcast + rect
  (Q9, stage B); per scene the ladder stops at the first failing rung. 3.6: 18 units (13 sequences + Talbot and E2 on
  ixs_b / ixs_e (Q11) + boost 3 on ixs_d (Q7)), stage `dyn` per test frame (0.2 % global, 2 % per 64² tile, 3 % per mask
  region) plus `dynamic.py sequence`: the drift of Δ_f over the test frames (OLS slope, sandwich variance across the
  shared chains, gate |z| ≤ z_{1−α_u/2}, Changelog E-2) and the failing-tile count per frame vs Binomial(m, 0.01) at α_u.
- **PT references** (our PT, RR off, seed 7001 / re-run 107001) at every test frame (`--frames t`, the resolved frame
  state of `resolvePackageFrame`) and one base-state reference per static package (E-8); cached in
  `validation/out/m5/ptrefs` keyed by package bytes, frame, size, seed and the PT code closure; sizes frozen per PT code
  (`validation/out/m5/ptsize`).
- **Masks** (`dyn_masks.py`, §6.4; E-3): partition M_disocc (the chain runner's `--mode disocc`: pixel-centre V-buffers
  at t−1 and t through the production `temporal_pixel`), M_new, M_gone, M_edge, M_steady from 4×4-block means of fixed
  mask references (512 spp × 4, seed 7301) at t and t−1; plants add M_sil and the dominance regions M_light:<name>
  (single-emitter PT renders of derived packages, E-7). Regions < 256 px are dropped.
- **Sizing** (PLAN §7.3; §6.4). Pilots: PT 128 spp × 16 per (package, frame), 64 chains per unit (seed 7011 / 7012).
  Per test frame and aggregate (global, gate tiles, mask regions; Y/R/G/B) u = per-sample variance / (D·T)², T =
  δ/(t_{0.99,15} + z_{1−0.005/m}); the PT samples of a frame are shared by every chain variant of the package (static:
  3.3/3.4/3.5; dyn: base/Talbot/E2/boost); the joint allocation minimises GPU time, × 1.25, R ≥ 256 (3.5: 64, E-4) in
  multiples of 16, PT spp per batch niceCeil ≥ 256, B = 16. Unit cap 30 min per side: the tile aggregate is enlarged
  one step (32² → 64², 64² → 128²; masks unchanged); a unit whose global aggregate alone exceeds the cap gets 90 min once
  (Q4), beyond that it is `infeasible` and goes to the coordinator. Plan > 14 h ⇒ `--part core` + `--part static` (Q3).
- **Plants** (§6.5, TD29; plant_sign.py): 21 rendered plants at 1× chains against 4× PT references (seed 7201), chains
  cut after the last predicted frame; a plant passes iff detected (half of the planted chains vs a PT half fails in
  ≥ 9/10 repeats, PT A/A ≥ 9/10, E-5; full comparison not `pass`) AND every evaluable prediction holds (one-sided z ≥ 3
  on its region; "only" predictions: no opposite-sign region/tile with z ≥ 4; empty regions are "not evaluable", E-6).
  Synthetic W × 1.003 + calibrate A/A re-splits (`dynamic.py calibrate`) on the A/A run; **A/A**: m5s_cornell_i 3.4 at
  t = 24, seeds 7502 vs 7503, 4× size, must pass.
- **U8 ladder**: the six u8 scenes through 3.1 / 3.1b / 3.2 (M4 sequential harness) and 3.3 / 3.4 (chains); all pass.
- **T15/T16 per chain run** (restir-chain-run.ts + `t16ChainProblems`): NaN/Inf/negatives 0, every RSC error counter
  incl. `RSC_T_NONFINITE` / `RSC_T_PENDING_LEFT` 0, q0–q2 overflow 0, no submit over 200 ms, identical reset patterns in
  every batch; **history valid on every frame except t = 0 and the package's `resetFrames`** (ixs_k: the map swap at
  20) — "any config change resets history" and nothing else does; temporal units on every frame; preset of the rung,
  RR off, cCap 20, jitter iid, Mode A, spatial rounds executed = rounds; maxBounces / scene bytes / resolution / env NEE /
  frame = the PT reference's.
