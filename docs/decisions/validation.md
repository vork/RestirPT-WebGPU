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
