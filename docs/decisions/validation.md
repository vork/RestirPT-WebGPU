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
- **Plants** (§6.5, TD29; plant_sign.py): 20 rendered plants (U8-4 is deferred to M6 with U8-7, 8, 10 by B-9 / Q6 and listed as such in the summary) at 1× chains against 4× PT references (seed 7201), chains
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

**Pilot sizing (2026-10-01, `budget.json` `m5_entries`, harness 908aa2b+).** 49 sized units = 7.0 h of chains + PT
references (static 3.3–3.5 + U8 3.3/3.4: 2.6 h of chains; rung 3.6: 2.8 h), + Gate 0 ≈ 1.5 h, plants ≈ 1.9 h, A/A ≈ 3.2 h
(2 × 4 × the m5s_cornell_i 3.4 chains), U8 M4 rungs ≈ 0.5 h ⇒ 14.1 h > 14 h: two parts (Q3), `--part core` ≈ 9.6 h and
`--part static` ≈ 3.3 h. Tile enlargements by the 30-min cap: m5s_spot_grazing, glass_mirror_A, many_lights (32² → 64²);
ixs_c_spot_b03, ixs_e_addremove (+ Talbot, E2), ixs_f_combined, ixs_i_envradio (64² → 128²). Q4 raises (global alone
> 30 min): ixs_c_spot_b03 (43 min, R 25 488) and ixs_f_combined (31 min, R 14 432). No unit is infeasible.

**A/A sizing (coordinator decision, Changelog E-21).** The A/A pair (seeds 7502 / 7503, m5s_cornell_i 3.4 at t = 24) ran
at 4× the core-part sizing of its unit: the core run measured 90 ms per chain, so 32² tiles would have taken 42 min
(> 30-min cap) and the unit was sized at 64² tiles, R 2 272 → A/A 9 088 chains per seed. The static part, measuring
55 ms per chain, gated the same unit at 32² tiles with R 27 760. The tile decision depends on measured timing. The A/A
was kept as run: it passed (Δ_Y +0.003 %) and the synthetic W × 1.003 plant was detected 10/10 at the same power, which
is what the A/A calibrates; mirroring the static sizing (111 k chains per seed, ≈ 3.4 h) was not spent.

**M5 gate result (final, 2026-10-06; harness through 42c8dbe, kernel code e871ea6 + the C-11 N1 fix a8045d6).**
Runs (all under `validation/out/`, copied to the main checkout): core `m5-gate-20261001-081949`; static
`m5-gate-20261001-162035` + re-run `-191515` (cornell_i, spot_grazing, u8_c0c_point_b1); 128² re-evaluation
`-203321` (`--reuse-chains`, aggregation fix only); plants `-20261002-051049` (fresh seeds), `-074225` (N1 trio, U8-3,
A/A); ixs_i after E-20 `-20261006-180639` (unit), `-180833` (no-jp-env), `-180946` (U8-2t, moved to Gate 0); env-γ_t
third seed set `-181922`; T3-2 addendum `m5-core-addendum/`.

| Part | Result |
|---|---|
| Gate 0 | green: restir-tframe, restir-temporal 47/47, restir-refresh, restir-debug, restir-initial, restir-shift 25/25 (one 2.5 h hold in this run; E-16 splits it per T3 variant from now on), restir-spatial, M3 regressions; typecheck, cpu 435, python, make-m5 determinism, U-TR-1, M5 app smoke; T3-2 rare bins (translate / add-remove + intensity / moving lights + env) 5.6–7.1·10⁸ trials each, LOGIC = FP = PLATFORM = 0; U8 plant activity (U8-2t, E-22) |
| Static 3.3–3.5 + U8 | 77/77: 3.3/3.4 × 8 scenes × t ∈ {1, 24} (32), 3.5 × 3, U8 chains 24, U8 M4 rungs 18; \|Δ_Y\| ≤ 0.023 %, worst tile ≤ 0.31 % |
| Rung 3.6 | 18/18 (13 sequences + Talbot / E2 on ixs_b and ixs_e + boost on ixs_d), every test frame + drift / failing-tile statistics; \|Δ_Y\| ≤ 0.063 %, worst tile ≤ 0.39 %, \|drift z\| ≤ 2.2; every counter 0 |
| Plants | 18 of 19 gating plants pass with their predictions; env-no-rot-vis detect-only (resolution-limited, C-12); U8-2t Gate 0 (E-22); U8-4/7/8/10 deferred to M6 |
| Calibration | A/A (7502 vs 7503, m5s_cornell_i 3.4, 4×) pass, Δ_Y +0.003 %; synthetic W × 1.003 detected 10/10, control 10/10, A/A re-splits ok |

Plants (Δ = ReSTIR − PT relative, Y; every row detected 10/10 with the PT A/A control ≥ 9/10 and the full comparison
failing; "fresh seeds" = rendered on disjoint seeds after the prediction was revised, E-18 / E-23):

| Plant | Scene, frames | Prediction | Measured | Status |
|---|---|---|---|---|
| N1-mixed | ixs_e 8, 14, 15 | M_light:C −, M_light:A − | −70.4 %, −19.9 %, −18.9 % | pass (re-rendered after the C-11 fix) |
| N1-mixed r 0.5 | ixs_e_half 14, 15 | M_light:A + | +61.6 %, +58.5 % | pass (C-11 fix) |
| N1-consistent | ixs_a 10, 25 | M_new −, M_up − (only −) | M_up −38 % / −82 %, M_new −88 % (f25; empty at f10) | pass (C-11 fix) |
| N2 | ixs_e 14, 15 | M_light:A −, M_light:B + | −16.0 / −15.2 %, +44.9 / +42.7 % | pass |
| N3 | ixs_a 40 | M_down +, M_gone + | M_down +3.1 % (z 8.1); M_gone empty | revised after measurement (C-11), confirmed on fresh seeds |
| N4 | ixs_n4 8, 16 | global − | −9.3 %, −10.6 % | revised (C-11), confirmed on fresh seeds |
| N5 | ixs_d0 32 / ixs_d 16 | M_sil + / M_edge + | +13.7 % / +8.8 % | revised (B-12), confirmed on fresh seeds |
| N6 | ixs_d_glossy 16, 32 | detect (sign not predicted, Q5) | −0.14 %, −0.22 % | pass |
| N7 | ixs_e 14, 15 | M_light:B − | −5.0 %, −4.7 % | pass |
| skip rotation refresh | ixs_h 10, 25 | detect (expected M_down +, M_up −; informational) | M_down +0.22 %, +0.20 % (z 2.1, 1.6) | detect-only, resolution-limited (C-12) |
| E_{t−1} with γ_t | ixs_h 10, 25 | M_new, M_up, M_down, global − | M_up −61 / −64 %, M_down −25 / −20 %, global −9.7 / −15.3 % | revised (B-12, C-12), confirmed on a third seed set |
| omit J_P on env | ixs_i 16, 17 | M_light:env − | −1.9 %, −1.8 % | revised (C-11), confirmed on fresh seeds (after E-20) |
| c_p + 1 | m5s_cornell_i 3.4 t 24 | global − | −46.7 % | pass |
| W × 1.003 | A/A run | detected | 10/10 | pass |
| U8-1 ω1 < 1 for delta | u8_c0c_point_b1 | global + | +0.87 % | revised (B-12), confirmed on fresh seeds |
| U8-3 no p_k ratio | m5s_cornell_i 3.4 t 24 | detect | −1.1 % | pass (scene moved by B-12: inert without deep paths) |
| U8-4 J = t_x²/t_y² | — | — | — | deferred to M6 (B-9) |
| U8-6 one-sided ignored | u8_c0e_rect_b1 | detect (sign reported) | −0.68 % | revised (B-12), confirmed on fresh seeds |
| U8-9 FAILED dropped from k | u8_c0e_rect_b1 | global + | +1.83 % | revised (D-5), confirmed on fresh seeds |
| U8-2t stale aux | — | Gate-0 activity test | emissive + env mean +0.006 %, reservoirs differ; rect-only bitwise identical | active, bias below δ (E-22) |
| U8-5t spot axis of t−1 | ixs_c_spot_b03 20 | detect | −2.95 % | pass |
| A/A | m5s_cornell_i 3.4 ×4 | pass | Δ_Y +0.003 % | pass |

Harness fixes found by the gate (restir-temporal-api.md Changelog): E-15 128² aggregates from ensemble.npz; E-16 one
GPU-lock hold per T3 variant (≤ 24 min, the documented exception); E-17 chain npz uploads in 8 MB parts (Playwright CDP
string limit) and regex-escaped `-t` patterns (two T3-2 cases had silently run no test); E-18 / E-23 fresh disjoint seeds
for predictions revised after measurement; E-19 / C-11 the N1-mixed inverse refresh (no PENDING_LEFT exception); E-20
the ixs_i rect back at the ceiling (the E-13 placement made the scene heavy-tailed); E-21 PT runs at ≤ 32 spp per
dispatch (hard cap) and the A/A sizing rationale; E-22 U8-2t as a Gate-0 activity test.

# M6: Enhanced features, Mode-B ReSTIR, glass and alpha in ReSTIR, Gate 3 rungs 3.7 / 3.9 / 3.10 / 3.11 and Gate 5 (restir-m6-api.md §4–§5, PLAN §5 M6, §7.1, §7.4 M6)

`npm run validate -- --milestone M6 --part core|r37|r39|r310|r311|gate5|plants` runs `validation/harness/gate-m6.ts`
(seven required parts; the harness of each is the M4 sequential harness or gate-m5's chain machinery):
- **core (Gate 0).** Typecheck; cpu lane (U-WGSL-BITS, T14, the plant-m6 predictions, layout / inline budget, gate
  configs, every M1–M5 test); python tests (incl. dup_bias); `make-m6.ts` determinism; the Chrome suites restir-m6
  (U-M5-BITS, compile smoke, U4-tiles, U1-M, U10-B, U9-R, U-RR-M6, U-DMV-1, functional smokes), the T3-M6 variants of
  restir-shift (t3_cases_256 + RIS-NEE, t3_modeb_256, t3_modeb_rare_256, t3_glass_256, t3_glass_pane_256,
  t3_alpha_256; one ≤ 24-min hold each, E-16), the Mode-B T3-2 cases of restir-temporal (24 000 pairs at 256², every
  ana / deep bin ≥ 10⁶; M6-8, M6-13), and every M4 / M5 Gate-0 suite and T3-2 rare-bin hold as regressions; the M6 app
  smoke (views 471–479, Mode B default, feature toggles).
- **Rungs (Stage B, δ 0.2 % global / 1 % per 32² tile; dyn 2 % per 64² tile, 3 % per mask).** 3.7 toggles one feature
  per unit on the M4 / M5 rung it extends against the cached M4 / M5 PT references (MD14): pairing σ 16, RR, RIS-NEE,
  all features (`offline-m6`), chains `full-m6` on m5s_cornell_i and dual MVs on ixs_d_camera. 3.9 glass and 3.10 alpha
  run `offline-m6` (new PT references, seed 6001; (vi) A and (xii) reuse M4's); 3.11 runs `offline-m6` in light mode B,
  `full-m6` Mode-B chains and ixs_b_area in Mode B through the rung-3.6 statistics. One disjoint-seed re-run per failed
  unit; seeds per MD17 / M6-7.
- **Gate 5** (MD15): chains of the interactive configuration with the duplication map on, at cCap 5 and 20, on
  m5s_cornell_i t = 24, m5s_glossy_v1 t = 24 and ixs_d_camera (every test frame); dup_bias.py: mean noise-debiased 16²
  tile bias b̂_t ≤ 3.25 % and the 99 % bound of the global |bias| ≤ 3.25 %, noise floor ≤ 1 %; the dupmap-off twin of
  each must pass Stage B / dyn.
- **Plants** (§5.3, predictions derived before measurement in `tests/restir/plant-m6.test.ts`): U8-4, U8-7, U8-8,
  U8-10 (the M5 deferrals) by the TD29 rule (detected ≥ 9/10 half-size repeats vs the 4× PT, PT A/A ≥ 9/10, full
  comparison not `pass`, predicted sign z ≥ 3 on the predicted region); the synthetic W × 1.003 + calibrate re-splits
  and a ReSTIR A/A (6502 / 6503, 4×) on (i) `offline-m6`.

**Sizing** (`budget.json` `m6_entries`, 53 rows from the runs r37, r39, r310, r311 and gate5 listed in `m6_runs`,
written by `--write-budget`, 305a358). The rows equal the PT spp × B, ReSTIR frames × B / chains R and tile size of
every executed unit in the final logs (checked unit by unit). The largest sides: (x) RIS-NEE 3.7 (ReSTIR 768 fr × 16,
36 min, in 4 lock chunks), G9 bubble (41 min), G7 Principled mix (25 min), G8 (22 min, 64² tiles), the full-m6 /
Mode-B chain units (17–19 min).

**M6 gate result (2026-10-08; final code = M6-10 delta-incoming fix + M6-11 dual-MV c_p ≤ 1; harness through 23087c7).**
Every rung, Gate-5 and plant run below is on the merged fixes (ReSTIR code hash fca1aeda68c6, PT bf362071108b, i.e. the
PT closure is unchanged, MD2). Runs (under `validation/out/`): r37 `m6-gate-r37-20261007-062218`, r39
`m6-gate-r39-20261007-082614`, r310 `m6-gate-r310-20261007-104930`, r311 `m6-gate-r311-20261007-105855`, gate5
`m6-gate-gate5-20261007-115614` recomputed by `m6-gate-gate5-20261007-154618` (M6-12; in the WebGPURestirPT-g5
worktree, with `validation/out/gate5-recompute/gate5-recompute.json`), plants
`m6-gate-plants-20261007-153724`, core `m6-gate-core-20261007-172751` (31/40; started 2026-10-07 17:27 on the
production code of cd279de, before 2fe00ec) + a targeted re-check of its 9 failing steps on 23087c7
(`m6-recheck-app-smoke` and the vitest logs of the re-check).

| Part | Result |
|---|---|
| Gate 0 (core) | 31/40 in the full core run; the 9 failures re-run individually at the gate setting on 23087c7, all pass (table below). Green in the full run: typecheck, cpu 499, python 74, make-m6 determinism, restir-m6, T3-M6 t3_cases_256 + RIS-NEE / t3_modeb_256 / t3_modeb_rare_256 / t3_alpha_256 (LOGIC 0), every M4 T3 variant, restir-initial / spatial / debug / tframe (U-M4-BITS) / temporal / refresh, the M3 suites, U8-2t activity with RIS on |
| 3.7 features | 112/112 steps; 20 units + the dual-MV sequence statistics: \|Δ_Y\| ≤ 0.0010 % on the sequential units (MDB ≤ 0.015 %), chains full-m6 −0.0038 / +0.0002 %, dual MV on ixs_d_camera f8–f64 ≤ +0.020 % (f40 +0.0078 %, MDB 0.051 %), worst tile ≤ 0.31 %, no re-run |
| 3.9 glass | 108/108; 21 units (C0h–k, G1 ×2, G3–G10, G5b (B), G6 (B), G6-neg, G7 furnace, (vi) A, (vi) B, (vi-B) (B), (xiv) glass): \|Δ_Y\| ≤ 0.0076 %, worst tile ≤ 0.29 %, no re-run |
| 3.10 alpha | 13/13; (xii) −0.0008 %, x10_foliage −0.0000 %, worst tile ≤ 0.11 % |
| 3.11 Mode B | 73/73; (i) B, m6_crossings_B, C0o, (xiv) overcast + rect B: \|Δ_Y\| ≤ 0.0067 %; Mode-B chains on m6_crossings_B f1 / f24 +0.023 / +0.006 %; ixs_b_area Mode B (6 test frames + sequence) \|Δ_Y\| ≤ 0.036 %, worst tile ≤ 0.26 % |
| Gate 5 | 6/6 configurations pass (M6-12 recompute): mean b̂_t 0.22–0.89 % against the 3.25 % budget, 99 % global bound ≤ 0.30 %, noise floor 0.04–0.21 %; every dupmap-off twin passes (\|Δ_Y\| ≤ 0.02 %) |
| Plants | 23/23: U8-4 M_foot +4.36 % (z 674, global +2.30 %), U8-7 global −1.06 % (z −325), U8-8 global +26.5 % (z 3721; predicted +26 %, M6-3), U8-10 global +0.79 % / M_hl +0.92 %; each detected 10/10 with the control 10/10; ReSTIR A/A Δ_Y +0.0004 %; W × 1.003 detected 10/10, A/A re-splits ok |

Core failures and their re-check (each step run alone with the gate's command and environment on 23087c7):

| Failed step (core run) | Cause | Fix | Re-check on 23087c7 |
|---|---|---|---|
| T3-2/M6 Mode B: camera translate, moving + rotating crossing lights, add / remove + intensity | the 24 000-pair holds hit the 120 s Chrome test timeout (no test result) | 2fe00ec `--testTimeout 900000`; M6-13 makes the Mode-B rule stricter (no empty crossing bin is skipped) | pass, LOGIC 0, FP 0; smallest bin c-ana/k>2 rtOk **1 233 254** (camera translate), 1 602 779 (moving lights), 1 303 123 (add / remove) |
| T3-2 M5 rare bins: translate, add / remove + intensity, moving lights + env | the M5 every-bin rule iterated over the six M6 crossing bins, structurally empty in light mode A | M6-13 (mode-A cases assert the crossing bins empty) | pass; minima c-tri/k>2 1 377 989, a-delta/k>2 1 427 754, c-tri/k>2 1 656 006 (bit-identical to the failing run) |
| T3-M6 t3_glass_256, t3_glass_pane_256 | grKm1 0.38 M, sideFlip 0.27 M < 10⁶: the glass counters stalled once the common case bins closed | M6-14 keep mask (selection only; scenes and shift unchanged) | pass; grKm1 5.38 M (grK 21.0 M), sideFlip 9.27 M; T3-1 LOGIC 0, FP 0, T3-D LOGIC 0, U-11 J 0 bad |
| M6 app smoke 5/7 | a ReSTIR frame encoded with an uncompiled pipeline variant at the RIS-NEE toggle | 0bf99a6 / 2a558ba (encodeRestir waits for the prepared variant) | 15/15 |

The full core part was not repeated on 23087c7; the 31 steps that passed were not re-run. Between the core run's code
(cd279de) and 23087c7 no shader changed: `src/` changed only in the app (interactive default light mode B, the variant
preparation in `renderer.ts` / `restir/kernel.ts`); everything else is harness, test and tool code (2fe00ec, M6-12's
dup_bias.py, M6-13's T3-2 fixtures, M6-14's keep mask in the T3 test kernel, where xKeep = 0 is the old behaviour).

**Failures that led to the fixes (pre-fix code, ReSTIR bad2cc978b59).**
- Rung 3.7 (`m6-gate-r37-20261006-210627`): 110/111, `ixs_d_camera_256@3.7-dualmv-f40` failed M_disocc and tile TOST on
  the run and its disjoint-seed re-run (−0.90 % / +1.85 %): a heavy tail from dual-MV picks reused at c_p = 20, not a
  bias ⇒ **M6-11** (`c_p ≤ 1` for a dual pick); f40 passes on the final code (+0.0078 %).
- Rung 3.9 (`m6-gate-r39-20261007-013338`, stopped during the 21st unit, no summary): 17 units passed and C0h Δ_Y
  +0.120 % (MDB 0.015 %), G1 Glass node +0.074 %, G1 Principled +0.052 % failed, all positive and concentrated at grazing incidence; reproduced at rung 3.1 with M4's own preset ⇒ not an M6
  feature: the path tree took the incoming direction after a delta event from positions ⇒ **M6-10**; fresh-seed
  verification 12/12 and the final run 21/21.
- Gate 5 (`m6-gate-gate5-20261007-115614`): dup_bias.py read the per-chain sums as means (≈ 25 000 % "bias") ⇒ **M6-12**
  (tool fix; the verdicts were recomputed from the same chains).

**Changes to earlier results.** M6-10 changes ReSTIR output wherever a delta lobe is sampled (smooth glass, roughness-0
mirrors / specular), including the M4 / M5 validation presets; MD1's "M4 / M5 results stay valid by construction" holds
only for delta-free scenes. The M4 / M5 units with a delta lobe were re-run on 23087c7 (next subsection). The M4 / M5
results on delta-free scenes, U-M4-BITS on (i) and every cached PT reference are unaffected.

### M4 / M5 units re-verified after M6-10 (2026-10-08, 23087c7)

**Selection** (from the packages' materials, not the scene names). A lobe is delta when its roughness is ≤ 0.00376
(`BSDF_ROUGHNESS_SQ_THRESH` 2e-10 on α²) and the lobe exists (V1: mix · glossy > 0; Principled: metallic > 0 or a
dielectric specular with IOR ≠ 1; glass / refraction nodes), or a refraction with |η − 1| < 1e-4. No package uses a
roughness texture. Among the M4 / M5 packages, delta lobes occur in exactly four: (vi) `vi_glass_mirror_A_512` (Principled metal
mirror r 0 + smooth Glass node), C0r `c0r_mirror_256` (V1 glossy r 0), (xiii) `xiii_spheres_512x256` (V1 glossy r 0
sphere) and M5's `m5s_glass_mirror_A` (the (vi) materials at 256²). Every other M4 scene ((i)–(v), (x)–(xii), C0q(d),
C0r irradiance, (xiv) ×6), every M5 static, sequence and U8 package and every M4 / M5 plant / A/A / ensemble scene is
delta-free (smallest roughness 0.05, (v) V1 sharp). (xiii) is report-only in M5 and was not run there. The fixture box
with the roughness-0 mirror is a Gate-0 GPU-test fixture: its U-M4-BITS / U-M5-BITS goldens were re-recorded by M6-10.

**Runs.** `npm run validate -- --milestone M4 --only vi_glass_mirror_A_512,c0r_mirror_256,xiii_spheres_512x256`
(`m4-gate-20261008-024722`, 29/29) and `npm run validate -- --milestone M5 --part static --only m5s_glass_mirror_A`
(`m5-gate-20261008-025320`, 19/19), on the normal seeds (M4 PT 4001 / ReSTIR 4002; M5 PT 7001 / chains 7002) and the
cached PT references (PT closure bf362071108b, unchanged). Old values: M4 from `m4-gate-20260930-162356` (C0r and
(xiii) also from the M6-branch subset `m4-gate-20261006-185706`, identical to 3 digits), M5 from
`m5-gate-20261001-162035`. The M4 pilots were re-measured, so (vi) and (xiii) 3.1 ran at slightly different frame
counts; the C0r, (xiii) 3.1b / 3.2 and M5 units ran at the old sizes and seeds, i.e. paired with the old runs.

| Unit | Δ_Y new (old) | MDB_Y new (old) | worst tile new (old) | size new (old) | Verdict |
|---|---|---|---|---|---|
| (vi) A 3.1 | −0.0028 % (−0.0028 %) | 0.0094 % (0.0085 %) | −0.29 % (−0.33 %) | 448 (512) fr × 16 | pass (pass) |
| (vi) A 3.1b | −0.0032 % (−0.0031 %) | 0.0091 % (0.0089 %) | −0.29 % (−0.33 %) | 448 (512) fr × 16 | pass (pass) |
| (vi) A 3.2 | −0.0028 % (−0.0031 %) | 0.0067 % (0.0074 %) | −0.14 % (−0.16 %) | 32 (40) fr × 16 | pass (pass) |
| C0r mirror 3.1 | −0.0007 % (−0.0007 %) | 0.0013 % (0.0013 %) | +0.01 % (+0.01 %) | 128 fr × 16 | pass (pass), identical |
| C0r mirror 3.1b | −0.0007 % (−0.0007 %) | 0.0013 % (0.0013 %) | +0.01 % (+0.01 %) | 128 fr × 16 | pass (pass), identical |
| C0r mirror 3.2 | −0.0009 % (−0.0009 %) | 0.0058 % (0.0058 %) | −0.03 % (−0.03 %) | 8 fr × 16 | pass (pass), identical |
| (xiii) 3.1 | −0.0014 % (−0.0013 %) | 0.0099 % (0.0101 %) | −0.20 % (−0.17 %) | 224 (256) fr × 32 | pass (pass) |
| (xiii) 3.1b | −0.0015 % (−0.0015 %) | 0.0100 % (0.0100 %) | −0.17 % (−0.17 %) | 256 fr × 32 | pass (pass) |
| (xiii) 3.2 | +0.0005 % (+0.0005 %) | 0.0102 % (0.0102 %) | +0.29 % (+0.29 %) | 28 fr × 32 | pass (pass) |
| m5s_glass_mirror_A 3.3 t 1 | +0.0037 % (+0.0035 %) | 0.024 % (0.024 %) | −0.03 % (−0.03 %) | R 2592, 64² | pass (pass) |
| m5s_glass_mirror_A 3.3 t 24 | −0.0002 % (−0.0002 %) | 0.034 % (0.034 %) | −0.10 % (−0.10 %) | R 2592, 64² | pass (pass) |
| m5s_glass_mirror_A 3.4 t 1 | +0.0211 % (+0.0209 %) | 0.041 % (0.041 %) | +0.11 % (+0.11 %) | R 928, 64² | pass (pass) |
| m5s_glass_mirror_A 3.4 t 24 | −0.0142 % (−0.0146 %) | 0.057 % (0.057 %) | −0.14 % (−0.14 %) | R 928, 64² | pass (pass) |

All 13 units pass with no re-run (multiplier 1 except (vi) 3.1 / 3.1b at 1.012 / 1.008) and every T16 counter 0. C0r is
unchanged to all printed digits (the mirror sphere reflects straight into the environment: no surface vertex follows
the delta event, so the incoming direction M6-10 changes is never formed). On the paired units the shifts are ≤ 0.0005
percentage points, consistent with M6-10 mattering mainly near grazing incidence on smooth glass (the C0h / G1
mechanism), which these scenes barely sample. (vi) A also passed rung 3.9 (`offline-m6`) against the same M4 PT reference (−0.0045 %). The M4
and M5 Stage-B results therefore stand for the M6 code.

**Performance (540p interactive, report only).** m6_crossings: 24.0 ms (before M6) → 28.8 ms (all M6 features) →
31.0 ms (+ Mode B); Cornell (i): ~16–17 → 20.7 → 21.3 ms. RIS-NEE ≈ +3.5 ms in rs_initial, Mode-B
crossings ≈ +1.7 ms, the duplication-map pass ≈ 1.0 ms.

**Interactive default** (305a358, MD9 after rung 3.11): light mode B, Gaussian pairing σ 16, RIS-NEE, dual MVs, the
duplication map, cCap 5. (Superseded for the app by perf2 D1 / D3: the app's ReSTIR-interactive runs RR after bounce 2
with the duplication map off; the `interactive` preset, which validation uses, is unchanged. See
[perf2 app defaults](#perf2-app-defaults-d1--d3).) The silhouette resolve fixes of the denoiser (denoiser.md DN-16 / DN-17) were merged on main
(cf8d358) after this branch forked; they are not part of the M6 runs.

**Open items.**
- Mode-B T3-2 margin: the rarest bin (c-ana/k>2, camera translate) reached 1.23·10⁶ round trips against the 10⁶
  minimum at 24 000 pairs (M6-13 projected ≈ 1.2·10⁶). A small change in the case mix would fail the bin rule; raise the
  pair count before tightening anything else.
- `validation/harness/denoise-run.ts` still rejects non-A light modes with "the interactive ReSTIR is Mode A only"
  (harness-only, stale since M6-9); the file headers of `src/core/render/renderer.ts` and `restir/kernel.ts` still say
  "Mode A only (D1)".

# M7: normal maps, smooth shading in ReSTIR, USD completion, loader fidelity, E2E (m7-api.md §6, PLAN §5 M7, §7.1 rung 3.8, §7.2)

`npm run validate -- --milestone M7 --part core|loader|stageA|e2e|r38|plants` runs `validation/harness/gate-m7.ts`:
- **core (Gate 0).** Typecheck; cpu lane (U-M7-BITS: 390 composed pipelines = the b5b0f5d text for scenes without normal
  maps; U-M7-ARENA: committed packages load and pack to the b5b0f5d bytes; tangents; plant-m7 predictions; gate config;
  every earlier test); python tests; make-m7 / make-m7-e2e determinism; Chrome: `normal-map` (U-NM-1..3), T3-M7
  (t3_smooth_256, t3_nm_256; LOGIC 0, every bin and the nmRc / smoothRc counters ≥ 10⁶), the M6 / M5 / M4 Gate-0 suites
  and holds as regressions, the perf probe (recorded), the M7 app smoke.
- **loader.** (viii-L) 7 USD files vs pxr, (vii-L) 9 glTF files vs Blender's stock importer, U-TAN-B on 5 packages.
- **stageA.** (vii-N) and E2E-HDR: our PT vs Cycles 5.2.2, TOST δ 0.5 % global / 2 % per 32² tile, Cycles 4096 spp × 16
  seeds (D4), ours 4096 × 16; flat normal maps and E2E-HDR tight, smooth scenes model-approximate (m7-api N6: TOST at
  the same δ gates, the Δ = 0 rejection checks are reported); a PT A/A.
- **e2e.** E2E-GLB (8 assets × FLAT / NORMALS) and E2E-USD (5 files), Stage A, model-approximate.
- **r38.** Rung 3.8, Stage B (δ 0.2 % / 1 %), pilot-sized as gate-m6.
- **plants.** Normal Map sign / strength in Stage A and Stage B, the shading-normal Jacobian plant, W × 1.003, ReSTIR A/A.

**Runs** (under `validation/out/`; PT code 41af3209…, ReSTIR 38f4b9e7…): loader `m7-gate-loader-20261008-045402`,
stageA `m7-gate-stageA-20261008-045417` (+ `-063001` for the low-poly scene), e2e `m7-gate-e2e-20261008-051440`
(+ `-054935` / `-055349` for e2e_usd_instancing after M7-10, `-101235` after M7-12), r38 `m7-gate-r38-20261008-054651`
(+ `-063257`), plants `m7-gate-plants-20261008-060715` (+ `-100741`: the revised B-SM-J), core
`m7-gate-core-20261008-062943` (killed with the session at its M4 restir-shift step; every step before passed) and its
re-run (see Gate 0). Code changes after the first E2E / Stage-A runs touched only `usd-scene.ts` / `usda-scan.ts`
(USD meshes without normals, compat instances): the other 20 E2E packages and every make-m7 package are byte-identical
before / after, so their units stand.

**Loader (6/6).** (viii-L) 7/7 USD files (cornell, m7_textured, m7_instancing incl. 16 PointInstancer instances and
bindings inside prototypes, the Y-up cm / Z-up m hand files, spike_hand, spike_blender): every draw's triangle count and
bbox, PreviewSurface constants and texture bindings, lights and cameras equal pxr. (vii-L) 9/9 glTF files (incl. Sponza
262 266 triangles / 69 textures and MetalRoughSpheres 1 040 213 triangles). U-TAN-B: 0 sign mismatches on every
normal-mapped corner; max angle 2.9e-4° (m7_nm_flat), 3.6e-3° (m7_nm_smooth), 3.5e-3° (m7_nm_env), 6.6e-4°
(NormalTangentMirror), 8.3e-5° (USD textured).

**Stage A (vii-N, E2E-HDR).**

| Unit | Tier | Δ_Y | MDB_Y | worst tile | Verdict |
|---|---|---|---|---|---|
| m7_nm_flat_256 | tight | +0.0006 % | 0.006 % | +0.03 % | pass |
| m7_smooth_256 | model-approx. | −0.0005 % | 0.013 % | −0.11 % | pass (also tight) |
| m7_smooth_lowpoly_256 | model-approx. | +0.0018 % | 0.008 % | −0.04 % | pass (also tight) |
| m7_nm_smooth_256 | model-approx. | −0.0014 % | 0.008 % | +0.07 % | pass (also tight) |
| m7_nm_smooth_B_256 (Mode B) | model-approx. | −0.0019 % | 0.007 % | +0.06 % | pass (also tight) |
| m7_nm_env_256 (overcast) | model-approx. | −0.0047 % | 0.002 % | −0.04 % | pass (also tight) |
| m7_xivlite_hdr_256 (original .hdr in Blender) | tight | +0.0021 % | 0.004 % | +0.03 % | pass |
| m7_xivlite_exr_256 (original .exr in Blender) | tight | +0.0021 % | 0.004 % | +0.03 % | pass |
| PT A/A (7301 vs 7302, m7_nm_flat) | tight | +0.0010 % | 0.006 % | −0.04 % | pass |

No unit needed a re-run, every replicate multiplier is 1, and no smooth unit failed even an informational rejection
check: the non-partition region (math §29) is below resolution in these scenes. E2E-HDR: the bridge recorded
`original: true`, the format and `blender_equals_ours: true` (Blender's decode of the Poly Haven file equals ours texel
for texel); the .hdr and .exr units give identical numbers.

**E2E (Stage A, model-approximate; 21/21 after M7-10 / M7-12).**

| Unit | Δ_Y | MDB_Y | worst tile | informational | Verdict |
|---|---|---|---|---|---|
| glb cornell_point_spot flat / normals | −0.0022 / −0.0010 % | 0.021 % | −0.28 % | – | pass / pass |
| glb metalrough_spheres flat / normals | +0.072 / +0.074 % | 0.010 / 0.012 % | +0.51 / +0.52 % | 17 / 20 rejection checks | pass / pass |
| glb texture_transform flat / normals | +0.0004 % | 0.002 % | +0.01 % | – | pass / pass |
| glb normal_tangent_mirror flat / normals | −0.0005 / −0.0002 % | 0.001 % | −0.01 % | – | pass / pass |
| glb alpha_mask flat / normals | −0.0014 / −0.0003 % | 0.004 % | −0.05 % | – | pass / pass |
| glb emissive_strength flat / normals | +0.0006 / +0.0011 % | 0.003 % | +0.02 % | – | pass / pass |
| glb transmission flat / normals | +0.0075 / +0.0081 % | 0.004 % | +0.09 % | 6 / 6 | pass / pass |
| glb ior_grid flat / normals | +0.0009 / +0.0012 % | 0.002 % | +0.06 % | – | pass / pass |
| usd cornell | +0.0004 % | 0.007 % | +0.07 % | – | pass |
| usd hand_yup / hand_zup | +0.0007 / +0.0013 % | 0.015 % | −0.29 % | – | pass / pass |
| usd textured | −0.0052 % | 0.005 % | −0.03 % | 3 (numeric_tiles) | pass |
| usd instancing (first run; after M7-10) | −0.0019 %; −0.12 % | 0.003 % | +1.47 % Y, R / B > 2 %; −2.40 % | 19 | fail; fail (re-runs confirmed) |
| usd instancing after M7-12 | +0.0004 % | 0.003 % | +0.02 % | – | pass (first run) |

- MetalRoughSpheres: a +0.07 % systematic (TOST passes; Šidák / χ² / mean-t / KS reject Δ = 0) confined to the sphere
  columns 3–4 under the small Mode-A rect light (sphere roughness down to 0): the D3 situation (near-specular lobes and
  Mode-A analytic NEE; heavy-tailed replicates). Identical at 1024 and 4096 Cycles spp (not D4). Reported, not hidden.
- TransmissionTest: +0.008 % (tiles at the glass), below the TOST δ by a factor 60; informational only.
- e2e_usd_instancing (PointInstancer pyramids / gems without authored normals, non-uniform instance scales) failed
  twice before passing: (1) LightUSD synthesises normals for meshes without authored ones (M7-10: Blender's automatic
  corner-angle-weighted normals instead, decided from the scan); (2) Cycles interpolates an instanced mesh's normals in
  object space (M7-12: compat mode keeps unnormalised `M^-T·n` for instance draws). The decisive diagnostic: Cycles
  rendering our package agreed with our renderer to 0.02 % per tile while Blender's stock import did not; transforms,
  per-vertex normals and materials were dumped equal on both Blender scenes, and realising the instances changed
  nothing. Flat shading (+0.92 %, tiles +14 %) and normals-as-directions (tiles +4 %) were measured and rejected.
  The final run used the gate seeds unchanged (no seed was re-drawn).

**Rung 3.8 (Stage B; 10/10 pass, no re-run; multiplier 1 except the low-poly unit, 1.028).**

| Unit | PT | ReSTIR | Δ_Y | MDB_Y | worst tile |
|---|---|---|---|---|---|
| m7_smooth_256 initial | 1792 × 16 | 1280 fr × 16 | −0.0081 % | 0.020 % | −0.15 % |
| m7_nm_smooth_256 initial | 1536 × 16 | 1280 fr × 16 | −0.0031 % | 0.017 % | −0.08 % |
| m7_smooth_256 offline-m6 | 1792 × 16 | 48 fr × 16 | −0.0047 % | 0.015 % | −0.21 % |
| m7_nm_flat_256 offline-m6 | 640 × 16 | 40 fr × 16 | −0.0009 % | 0.022 % | −0.11 % |
| m7_nm_smooth_256 offline-m6 | 1536 × 16 | 56 fr × 16 | −0.0051 % | 0.018 % | −0.16 % |
| m7_nm_env_256 offline-m6 | 256 × 16 | 14 fr × 16 | −0.0057 % | 0.030 % | +0.13 % |
| m7_nm_smooth_B_256 offline-m6, Mode B | 1280 × 16 | 48 fr × 16 | −0.0054 % | 0.023 % | −0.09 % |
| m7_nm_smooth_256 full-m6 chains t = 1 | 896 × 16 | chains, 64² | −0.0023 % | 0.027 % | −0.06 % |
| m7_nm_smooth_256 full-m6 chains t = 24 | 896 × 16 | chains, 64² | −0.0001 % | 0.053 % | −0.05 % |
| m7_smooth_lowpoly_256 offline-m6 | 896 × 16 | 32 fr × 16 | −0.0044 % | 0.019 % | −0.21 % (mult 1.028) |

**Plants** (predictions in `validation/scenes/m7-plants.ts`, pinned by `tests/scene/plant-m7.test.ts`, derived before
any planted render; detection = half-size repeats failing the gate, control = the reference side's A/A).

| Plant | Stage | Scene / region | Predicted | Measured | Detection | Verdict |
|---|---|---|---|---|---|---|
| A-NM-sign (bitangent sign ignored) | A (PT vs Cycles) | m7_nm_flat / M_P2 | − (−55.9 % direct) | −44.98 %, z −9333 (global −5.1 %) | 10/10 (10/10) | pass |
| A-NM-strength (glTF-style strength) | A | m7_nm_flat / M_P3 | − (−32.6 %) | −21.33 %, z −2967 (global −2.0 %) | 10/10 (10/10) | pass |
| B-NM-sign | B (ReSTIR vs 4× PT) | m7_nm_flat / M_P2 | − | −38.19 %, z −1489 (global −3.1 %) | 10/10 (10/10) | pass |
| B-NM-strength | B | m7_nm_flat / M_P3 | − | −16.73 %, z −408 (global −0.45 %) | 10/10 (10/10) | pass |
| B-SM-J (first scene) | B | m7_smooth_256 / global | detect | −0.000 %, z −0.01 | 0/10 (10/10) | **not detected** (M7-11) |
| B-SM-J (revised, fresh seeds 7805 / 7905) | B | m7_smooth_lowpoly_256 / global | detect | +0.023 %, z +3.55 | 10/10 (10/10) | pass |
| W × 1.003 (synthetic) | B | m7_nm_smooth offline-m6 | detect | – | 10/10 (A/A re-splits ok) | pass |
| ReSTIR A/A (7502 vs 7503, 224 fr × 16) | B | m7_nm_smooth | pass | +0.0002 % (MDB 0.011 %) | – | pass |

The measured magnitudes are smaller than the direct-light predictions (indirect light and the unplanted panel borders
inside the 15 %-inset masks dilute them); the signs hold everywhere with |z| ≥ 400.

**Gate 0 (core).** First run (`m7-gate-core-20261008-062943`, until the session that owned it ended): typecheck,
cpu lane (57 files / 517 tests incl. U-M7-BITS, U-M7-ARENA, gate-m7-config), python tests, make-m7 (8 packages) and
make-m7-e2e (21) determinism, `normal-map` (U-NM-1 quantised / lossless: 43 107 hits each, |N − N_f64| ≤ 2.2e-7,
866 mirrored, 10 826 backfacing, 0 fallbacks; U-NM-2: 2²⁰ (V, L) pairs, 302 840 bump-rejected, 0 mismatches in eval /
query / NEE f / sampling, glossy unchanged; U-NM-3: 0.44117 vs 0.44151 at s = 1 (z −2.4), 0.48486 vs 0.48492 at s = 0.5
(z −0.5)), T3-M7 t3_smooth_256 and t3_nm_256 (LOGIC 0, every bin and the nmRc / smoothRc counters ≥ 10⁶; one 24-min
hold each), the M6 regressions (restir-m6, the three Mode-B T3-2 cases, all six T3-M6 variants) and restir-initial all
passed. The full core part was relaunched (`--part core`, log `m7-core.log`) and stopped on 2026-10-08 at 13:35 by
the user's decision, after 15/15 steps had passed again (typecheck through the three Mode-B T3-2 cases; it stopped in
the first T3-M6 variant). Not run on the M7 code: the T3-M6 variants in that relaunch (they passed in the first run
above), the M4/M5 regression suites (restir-shift T3 variants, restir-spatial/-debug/-tframe/-temporal/-refresh, the
M3 suites, the T3-2 rare bins), the perf probe and the in-gate app smoke (21/21 in development). Rationale for
accepting: U-M7-BITS and U-M7-ARENA show that scenes without normal maps compose the b5b0f5d WGSL and pack the b5b0f5d
bytes, which is what those regression suites exercise. Run `--part core` before relying on M7 for anything those
suites cover.

**Performance (540p interactive Mode B, all M6 features; report only).** m7_nm_smooth: 27.2 ms with normal maps vs
26.5 ms with its normal textures removed (+0.7 ms; rs_initial +0.4 ms); Cornell (i) 16.6 ms (no NORMAL_MAP: the M6
pipelines). The M7 app smoke (Sponza 262 k triangles with 25 normal-mapped materials at 540p) renders PT and
ReSTIR-interactive finite, views 320–327 written.

**Open items.**
- MetalRoughSpheres' +0.07 % D3-type systematic: a Mode-B variant (light MIS on) of the E2E unit would separate the
  Mode-A near-specular NEE tails from a model difference.
- B-SM-J was not detectable on finely tessellated meshes (M7-11); the low-poly scene is the plant's home.

# M8: performance (docs/decisions/m8-perf.md; PLAN §5 M8)

`npm run validate -- --milestone M8 --part core|stageB|perf` runs `validation/harness/gate-m8.ts`; `npm run validate --
--all` runs every milestone gate M0 … M8 in order, each in its own process (PLAN §5 M8 exit).

**What M8 changes for validation.** Every validation path keeps BVH2, the AoS reservoir planes, the requested light-mode
text and the texture-path à-trous: the CWBVH, P-4, the SoA planes and the tiled à-trous are interactive-only variants
behind composer defines / options whose keys are absent in validation (U-M7-BITS: every validation pipeline still
composes to the M6 text). The one validation-path change is `ENV_COMPACT_IN_VALIDATION = true` (the env texture in the
smallest exact format), bitwise neutral by ENV-U7c. The interactive variants are bitwise the pre-M8 results
(U-M8-BITS AoS / SoA, U-M8-MODEB, U-DN-3c); the CWBVH is hit-equivalent to BVH2 (T12) and is checked end to end in
Stage B (part stageB).

**Code hashes.** The PT closure hash changed (traverse.wgsl gained the `BVH_CWBVH` include, scene-gpu / the BVH worker
the CWBVH option, env-gpu the format rule) although the validation outputs are bitwise unchanged; cached PT references
keyed by an older hash (M4–M7) are re-rendered by `--all` (no hash aliasing was introduced).

**M8 gate result (2026-10-08, code 2f393dd = 76ee705 + the ABBA perf ordering; PT code 7ca99daa6ec4, ReSTIR fced7bf07351).**
Runs under `validation/out/`: core `m8-gate-core-20261008-133740`, stageB `m8-gate-stageB-20261008-134328`, perf
`m8-gate-perf-20261008-135504` (ABBA; the first perf run `m8-gate-perf-20261008-134944` measured M8 before pre-M8 and is
superseded).

| Part | Result |
|---|---|
| core (Gate 0) | 12/12: typecheck, cpu lane (U-M7-BITS: every validation pipeline = the M6 text; cwbvh encoder; dynres; env-compact static check; every earlier CPU test), python tests, m8-bits (U-M8-BITS AoS and SoA = the pre-M8 goldens on 6 cases, U-M8-PTBITS 5, U-M8-MODEB 3, P-4 switching), bvh (T12 on BVH2 and CWBVH, MT and Woop: unexplained 0, hit equivalence, watertight 0, self-hits 0, seams 0), denoiser (incl. U-DN-3b / 3c), env (ENV-U7c 0 mismatches on 6 envs), env-format (report), env-sampling, normal-map, M5.5 app smoke 35/35, M7 app smoke |
| stageB (CWBVH end to end, δ 0.2 % / 1 %) | 3/3, no re-run, multiplier 1: m7_nm_smooth_256 offline-m6 Mode A Δ_Y −0.0057 % (MDB 0.016 %), worst tile −0.14 %; m7_nm_smooth_B_256 Mode B −0.0054 % (0.023 %), −0.09 %; m7_nm_env_256 −0.0057 % (0.030 %), +0.13 %; T16 incl. `config.bvh = cwbvh` |
| perf (recorded) | m8-perf.md §14: targets met on Cornell, m6_crossings (540p; 720p 52.1 ms in a slow session, 44.7 ms at the baseline), m7_nm_smooth; Sponza + HDRI missed (78.7 ms at 540p N=3, −12 % vs pre-M8); dyn-res 0.625: 31.9 ms |

The CWBVH units use the M7 rung-3.8 seeds and sizes on the same packages, so they are close to M7's BVH2 values
(−0.0051 / −0.0054 / −0.0057 %): the hit-equivalent structure reproduces the estimator, as T12 predicts.

**`validate --all`.** Launched on 1b37d25 (run `validate-all-20261008-140328`, started 2026-10-08 16:03 CEST); it re-renders the
M4–M7 PT references whose cache keys hold an older PT code hash and re-runs every Stage-A / Stage-B ladder (≈ 40 GPU
hours by budget.json). Its summary: `validation/out/validate-all-20261008-140328/summary.json`. The M0 / M1 lane steps
run their own milestone's GPU suites only (M0: smoke.gpu.test.ts in node-dawn and Chrome under the GPU lock; M1: bvh /
env / textures / primary): the full lanes now contain every later suite, which need their own gates' env, timeouts and
lock holds. Result: stopped on 2026-10-08 by the user's decision after M0–M2 and part of M3 (168 steps passed; M1 10/10,
M2 24/24; M0 8/9: the only failure was the Chrome-version pin, Chrome having auto-updated to 155, now relaxed to
major ≥ 154, platform-lanes.md). The rest was not run on the M8 code; the validation path's bitwise identity to eda22be
(U-M8-BITS, U-M7-BITS) and the CWBVH Stage-B units above stand in for it. Run `validate --all` before relying on M8
for anything those gates cover.

**Open items.** D-M8-1 (96 B / 120 B quantized layout), D-M8-4 (40 B G-buffer), Sponza's targets (D-M8-6: a wavefront split
of the path tree, a cheaper RIS target), rs_primary reusing the M1 V-buffer (≈ 1.4 ms on Sponza), tinybvh WASM once
emscripten is available, the app's maxBounces default (3) vs PLAN §1.10 (2).

# perf2 app defaults (D1 / D3)

Interactive app defaults since perf2: Russian roulette after bounce 2 (`rrMinBounces 2`, D1) and the duplication map off
(`dupmap false`, D3), applied through `INTERACTIVE_APP_DEFAULTS` (renderer.ts). Test rigs keep `INTERACTIVE_PINNED`, so
every golden is unchanged. With the map off, ReSTIR-interactive is unbiased by default.

- **Equal quality (WP-Q, perf2-plan.md §5):** D1 (rr2) passes on Cornell and Sponza (Sponza relMSE×ms −7.2 %). D3 fails the
  denoised Cornell checks (FLIP +1.9–5.1 %, temporal std +3.8–6.8 %) and passes Sponza; adopted by the user's decision for
  unbiasedness.
- **Stage B** (`npm run validate -- --milestone M6 --part decisions`, run `m6-gate-decisions-20261009-150044`): **11/11
  PASS** at unchanged δ, interactive c_cap 5, map off, rrMinBounces 2:
  m6_crossings_B_256 Mode B t=24 Δ_Y +0.0053 % (MDB 0.060 %); m5s_cornell_i t=24 −0.0044 % (MDB 0.013 %);
  ixs_d_camera_256 dyn f8…f64 all pass (|Δ_Y| ≤ 0.021 %, MDB ≤ 0.049 %), sequence drift test pass.
- **Timing** (same-session ABBA, 540p N3, 2 blocks, after the run-perf fix 4b760d6): new vs old defaults Sponza
  85.8 → 79.2 ms (−7.6 %), Cornell −0.9 % (noise). Separately: rr2 Sponza −8.3 %, Cornell 0; map off Sponza −0.8 %,
  Cornell −6.1 % (rs_dupmap −0.7 ms).
- **Harness bug found on the way (4b760d6):** `run-perf --abba FLAGS` without `--kernel-flags` ran the base with the release
  set and the variant without any released flag; the first app-defaults ABBA therefore read +23 %. Any `--abba` run made
  after flags were released and without `--kernel-flags` must be re-measured; the wave-1 overall run used
  `--kernel-flags ""` and is unaffected.
- **App smokes:** M5 87/87, M5.5 35/35, M6 15/15 (expects the new defaults).
