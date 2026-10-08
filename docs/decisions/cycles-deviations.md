# Cycles reference: deviations from plan §7.5 and documented Cycles behaviour (M3a, M3b, M3c)

Gate 2 (Stage A, our PT ≡ Cycles) found the items below while closing M3a. Each has its evidence and its effect on
the gate. None of them loosens δ.

## D1. `sampling_pattern`: SOBOL_BURLEY, not TABULATED_SOBOL (reference-setting fix)

**Symptom.** In scene (iii), the spot-grazing scene, the dim indirect-only regions failed Stage A. The worst tile was
−2.2% to −3.7% (Šidák-rejected on both the first run and the confirmatory re-run on disjoint seeds), and there were
radial streaks of ±1…3% centred on the spot light.

**Localisation.**
- The failure is in the per-bounce difference. At b = 0 the two renderers agree everywhere (|Δ| ≤ 1e-5 per quadrant).
  At b = 1 the streaks appear, so the cause is the first indirect bounce.
- An A/A split of each renderer (seeds 0–7 vs 8–15 for Cycles, seed 7 vs 99 for ours) is clean. So the pattern does not
  depend on the seed, and it is present in one renderer only.
- Variant scenes: without the wall the streaks vanish, and without the box they remain. The streaks need the wall
  hotspot next to the light, a small, very bright area seen at grazing angles from the floor.
- An **f64 quadrature** was built for the no-box variant: floor (ρ 0.6) + wall (ρ 0.7/0.6/0.5), spot 150 W, 70°, blend
  0.25, 0.12 m from the wall. The indirect term is ρ_f/π ∫_wall L_wall cos cos / r² dA on a 3 mm wall grid. Results in
  3° angular bins around the light (luminance, indirect only, 16 replicates each):

  | bin | ours vs truth | Cycles TABULATED_SOBOL vs truth | Cycles SOBOL_BURLEY vs truth |
  |---|---|---|---|
  | 23° | −0.02 ± 0.18% | **+1.62 ± 0.10%** | −0.13 ± 0.11% |
  | 50° | +0.24 ± 0.31% | **−1.96 ± 0.27%** | +0.07 ± 0.24% |
  | 59° | −0.03 ± 0.49% | **−3.56 ± 0.21%** | −0.66 ± 0.24% |
  | 71° | +0.47 ± 0.42% | **−3.50 ± 0.24%** | +0.03 ± 0.29% |
  | 86° | −1.02 ± 0.31% | **+2.73 ± 0.33%** | +0.36 ± 0.28% |

  Over all 24 bins: ours stays within about 2 SE of the truth. TABULATED_SOBOL is off by up to 17 SE. SOBOL_BURLEY
  matches.

**Cause.** TABULATED_SOBOL takes its sample values from precomputed Owen-scrambled Sobol tables that every pixel and
every seed share. With `scrambling_distance = 1` a pixel's hash (which includes the seed) only chooses and shuffles
table entries; there is no per-pixel continuous randomisation. The estimator is therefore a fixed quadrature rule: its
error for a small bright feature in sample space is deterministic, the same for every seed, and varies smoothly from
pixel to pixel. That is where the streaks come from. SOBOL_BURLEY computes Owen-scrambled Sobol on the fly with a
per-pixel, per-seed hash, which makes it an unbiased randomised QMC. Replicate statistics (plan §7.3) require that.

**Fix.** `validation/blender/cycles_settings.py` now sets `sampling_pattern = "SOBOL_BURLEY"` in the `_sampling_rows`
table, around line 166. It is a debug enum item, so `apply_settings` enables `preferences.experimental.use_cycles_debug`
and `view.show_developer_ui` first. Nothing is saved, because renders use `--factory-startup`. The plan §7.5 bullet is
updated. Every Cycles reference cache key changes with the script hash, so the M2 references re-render on their next
use.

## D2. Numerical-equivalence floor for near-deterministic tiles (`num_eps = 1e-4`)

**Symptom.** Two scenes failed only the rejection checks (χ²_red, mean-t, Šidák), never TOST:
- C0l, the sun on a floor: direct light from a delta light is deterministic per pixel, the tile SE is about 1e-7, and
  Δ_Y was +2.5e-6 globally with a worst tile of 1.7e-5 (4.3e-5 on the re-run).
- C0g, the BSDF furnace at b = 0: tile SE about 5e-6, Δ_Y +1.2e-5, and ≤ 4e-5 on the r = 1 GGX sphere. Our three
  estimators (MIS, NEE-only, BSDF-only) agree with each other to about 1e-5.

**Evidence that this is arithmetic, not a model mismatch.** In C0l, per-pixel values differ by a median of 2.7e-6
(relative) in the lit region. That is a few f32 ULPs: the two implementations order the products differently (ρ/π·E·cos,
light normalisation, f32 accumulation of 1024–2048 samples per batch). Every such tile is 50–500× below δ.

**Rule.** This is `stats.py` `GateSpec.num_eps`, set per unit through test.json `"num_eps"`. A tile whose
SE_Δ ≤ num_eps·denominator leaves the Δ = 0 rejection family (Šidák, χ²_red, mean-t, KS/AD). Instead it must **match**:
|Δ| ≤ 2·num_eps·denominator + z_Šidák·SE_Δ (the tile's own noise is allowed for; z is the two-sided normal quantile at the Šidák level of the tile family), which is the gating check `numeric_tiles`. This extends the existing zero-variance rule,
which requires a match to 1e-6. TOST is unchanged. A real bias above 2e-4 in such a tile is still caught, and
`tests/test_stats.py::test_num_eps_floor` checks both directions. Every report records it in `notes.numeric_floor`.

**M2 consequence.** After the D1 switch, the M2 Gate 1-lite A/A on C0d (a b = 0 spot, tile SE about 1e-7) raised
χ²_red false alarms in about 1% of the re-splits: 2/80 at 20 splits, and 8/400 and 4/400 at 100 splits. TABULATED_SOBOL
gave 0/400. These tiles sit at f32 resolution, so validate.ts M2 now passes the same `num_eps = 1e-4`. With it, the A/A
is ok in 20/20 splits (and in 100/100 on two other re-split seeds), and the synthetic plants (×1.0075, 32² +3%) and the
rendered plant (blend 0.16) are still detected 10/10. `npm run validate -- --milestone M2` passes 24/24.

## D3. Mode-A analytic lights and near-specular lobes (scene design, not a Cycles defect)

The (v) glossy sweep with r ≤ 0.1 lit by **analytic** rect lights in Mode A (NEE only, as in Cycles with light MIS
off) had tile MDB up to 17% at 16×1024 spp. Both renderers are unbiased, but a light-area sample landing in a lobe with
α = 0.0025 is rare, so the tiles are too noisy to test. The r ≤ 0.1 scene therefore uses the same two rectangles as
**emissive quads** with the same front radiance. Their highlights are then sampled by BSDF MIS in both renderers.
Mode-A analytic NEE with glossy lobes stays covered by the r ≥ 0.2 scene and the V2 sweep. Mode B (light MIS on) is
M3b.

## D4. Sobol-Burley with next_pow2(spp) ≤ 2048: a seed-independent error on multi-bounce rough glass (M3b)

**Symptom.** In the M3b pilot (`validation/out/m3b-gate-20260929-025921`, Cycles 1024 spp × 16 seeds), two units passed
TOST but failed the rejection checks, on both the first run (seeds 0…15) and the re-run (seeds 100…115):
- **G4** (rough Glass-node slabs, N 3, L = 1 furnace): Δ_Y +0.004% globally, mean-t 8.4. Our PT was higher than Cycles
  by +2…4e-4 (relative) in the r = 0.5 and r = 1 tiles.
- **(vi) Mode B** (smooth glass sphere, roughness-0 mirror, rect + disk lights, 512²): χ²_red 3.35, KS/AD p ≈ 0, and two
  Šidák tiles. The worst tile was only 0.51%, but the tile-t maps of the two runs correlate at 0.63 (structured
  bands on the wall and floor, not noise).

**Localisation (G4).**
- An infinite rough slab (r 0.5, N 3) in the same furnace, seen at 25°/45°/60° through a 2° FOV: ours, Cycles and an
  f64 CPU Monte Carlo of the slab (independent random numbers) agree to ≤ 6e-5 (≈ 2 SE).
- The same slab seen by the G4 camera: ours is +3.2e-4 above Cycles at 2048 spp × 16 seeds (t ≈ 6 on every 64-px strip).
  At 4096 spp × 16 the two agree (|Δ| ≤ 7e-5, |t| ≤ 1.9). Resolution, aspect ratio and slab extent do not matter.
- A Cycles spp sweep of that scene with the **same** 16 seeds (200…215), luminance image mean vs ours (8192 spp × 16):

  | Cycles spp | next_pow2 | Δ vs ours | t | per-seed image means |
  |---|---|---|---|---|
  | 1024 | 1024 | −2.99e-4 | −11.1 | 0.608491 … 0.608724 |
  | 1536 | 2048 | −3.31e-4 | −18.4 | 0.608514 … 0.608610 |
  | 2048 | 2048 | −3.27e-4 | −18.6 | 0.608514 … 0.608598 |
  | 2560 | 4096 | −4.5e-6 | −0.3 | 0.608710 … 0.608810 |
  | 3072 | 4096 | −7.6e-6 | −0.5 | 0.608711 … 0.608798 |
  | 4096 | 4096 | +4.3e-6 | +0.3 | 0.608737 … 0.608786 |
  | 8192 | 8192 | +9.5e-6 | +0.6 | 0.608734 … 0.608790 |

  Every one of the 16 per-seed means at ≤ 2048 spp lies below every per-seed mean at ≥ 2560 spp. Cycles **CPU** at
  1024 spp gives the same images as Metal (mean |Δ| 6e-7 relative, identical statistics), so it is not the GPU backend.
- A Cycles-vs-Cycles A/A across seed sets at the same spp passes (G4 seeds 0…15 vs 100…115: global t −0.55, χ²_red
  0.76). The error is therefore common to all seeds: replicate statistics cannot see it, only a comparison with an
  independent estimator can.

**Localisation ((vi) Mode B).** At 256² with Cycles 4096 spp, variants without the mirror, without the spheres and
without the disk light all pass (χ²_red 0.99 / 1.46 / 1.61 over 64 tiles, max |t| ≤ 4.9), and so does the full scene:
ours (8192 × 16) vs Cycles 4096 × 16 has χ²_red 1.42, max |t| 3.0. The same full scene against Cycles **1024** × 16
fails (χ²_red 4.12, max |t| 7.1), and Cycles 1024 vs Cycles 4096 fails by itself (χ²_red 3.83, max |t| 6.6). Our Mode A′
and Mode B agree on this scene (tile χ²_red 0.04 with the same seed).

**Cause (as far as localised).** The step sits exactly where next_pow2(spp) goes from 2048 to 4096 (spp 2048 is
biased, 2560 is not). That is the Sobol-Burley `sobol_index_mask` (next_pow2(spp) − 1, reversed), which selects how
many Owen-shuffled Sobol indices a pixel's padded 4D sample sets use (`kernel/sample/sobol_burley.h`,
`pattern.h::blue_noise_indexing`). Each dimension set's scramble seed is the pixel hash XOR a fixed per-dimension
constant (`seed ^= hash_hp_uint(dimension_set)`) passed through the Laine–Karras-style `reversed_bit_owen`, so the
relative shuffle between the padded dimension sets is not independently randomised per pixel. We read this as the
same class of defect as D1: at ≤ 2^11 indices the pairing of dimension sets behaves like a shared quadrature rule for
high-dimensional integrands (three rough-glass bounces; caustic chains), and its error does not average out over
pixels or seeds. We have not isolated it further inside Cycles.

**Rule.** M3b Gate-2 references render at ≥ 4096 spp (`gate-m3b.ts` default `cyclesSpp: 4096`; the larger budgets of
G5, G8, G9 and G10 also have next_pow2 ≥ 16384). δ is unchanged. The M2/M3a scenes passed at 1024 spp (their
integrands are low-dimensional or diffuse), so their budgets are unchanged.

## D5. Rough transmission NEE of an MIS light: the approximate tier (M3b)

gap-glass §5.3 (table at line 363): for an MIS-enabled light reached by NEE through a rough transmission lobe, Cycles
weights the spurious region (f_T > 0 there, but the sampler cannot produce the direction) by its power heuristic,
w_C < 1, while our valid-only p2 gives ω1 = 1 there. Our expectation exceeds Cycles' by (1 − w_C)·f_spur, and only in
that region. G5b (rect light over smooth / rough slabs and a rough pane, Mode B) therefore runs in the documented
**approximate** tier (report.json `"tier": "approximate"`); δ is not loosened and nothing is clamped. Measured once
(gap-glass risk 1): in the pilot (Cycles 1024 × 16, ours 2048 × 16) G5b also met every tight rule, Δ_Y +0.017% globally,
worst tile 0.46%, no Šidák tile, so the excess is below the resolution of that run.
## D6. `use_light_tree`: off for env + analytic-light scenes (M3c, reference-setting fix)

**Symptom.** (xiv) Cornell open to the sky under kloofendal_48d_partly_cloudy_puresky **with** the interior 4 W rect
light (Mode A, b = 3, 32 × 1024 spp both sides) failed only the rejection checks (mean-t, KS/AD on Y/R/G/B), on the
first run and on the confirmatory re-run with disjoint seeds: Δ_Y = +0.0071% and +0.0069% (ours brighter), global MDB
0.005%. TOST passed (δ is 0.5%). The same scene without the rect light passed (Δ_Y +0.0013%), and so did the overcast
variant with the rect light (+0.0024%, not significant).

**Localisation.** A diagnostic Cycles render of the same package with `scene.cycles.use_light_tree = False` (seeds
300–331, otherwise identical settings):
- ours vs Cycles (light tree off): **pass**, Δ_Y +0.0007% (MDB 0.005%);
- Cycles (light tree off) vs Cycles (light tree on): Δ_Y +0.0072%, MDB 0.003%, mean-t/KS rejected.

So the shift is between Cycles' two light-selection methods, not between the renderers: with a world light and an
analytic light, Cycles' light tree is about 7·10⁻⁵ darker than its light distribution. The rect-only contribution
(scene with rect − scene without) differs by about +0.018% in the tree reference. The light tree splits between the
distant (world) and local subtrees per shading point. A plausible cause is that the selection pdf used in the world's
MIS weights (`light_tree_pdf`) does not exactly match the realized selection, so ω1 + ω2 ≠ 1 by a tiny amount. This was
not traced further in the kernel. The plan's pin ("use_light_tree True — only noise") does not hold at this resolution.

**Fix.** A scene package may set `"cycles": {"use_light_tree": false}` (scene-bridge.md). `cycles_settings.py` takes
the value from the cfg (default True, pinned as before), and `build_scene.py` accepts only this one override.
`make-m3c.ts` sets it on the (xiv) rect-light variants, the only env + analytic-light scenes. All other references keep
the light tree. With a single emitter (every other env scene) the tree selects that emitter with probability 1, so it
cannot bias there.

## D7. Metal kernel binary archives off, kernel optimization OFF, Blender pinned to 5.2.2 (reference-setting fix)

**Symptom.** Blender aborted intermittently during a reference render (4 crashes on 2026-09-29: 00:18, 05:41, 08:40,
08:53; macOS reports `~/Library/Logs/DiagnosticReports/Blender-2026-09-29-*.ips`). The main thread was always inside
`ccl::Session::wait()` (mid-render), and the crashing thread was one of 13 `ccl::ShaderCache::compile_thread_func` →
`MetalKernelPipeline::compile()` threads. There were two signatures:
- `-[NSURL initFileURLWithPath:]` raised on a nil path, which led to SIGABRT (3 crashes);
- `BUG IN CLIENT OF LIBMALLOC: memory corruption of free block` inside the AGX compiler (1 crash).

Each abort skipped `atexit`, which left `/tmp/restirpt-gpu.lock` behind and stalled every GPU job. Independently of
this fix, the lock now recovers by itself: every job writes a holder file `<tag>-<pid>` into it, and waiters reclaim a
lock whose holder files all name dead pids (`validation/harness/gpu-lock.ts`, `validation/blender/gpu_lock.py`).

**Cause.** In `intern/cycles/util/path.cpp`, `path_cache_get()` lazily assigns a file-static `std::string`
(`cached_xdg_cache_path`) with no synchronisation. `MetalKernelPipeline::compile()` (`device/metal/kernel.mm`) calls it
for every pipeline that uses a binary archive, from up to `maximumConcurrentCompilationTaskCount − 1` threads at once:
- concurrent assignment corrupts the heap;
- a torn read yields a path that is not valid UTF-8, so `@(path.c_str())` is nil and `fileURLWithPath:nil` throws at
  archive serialisation.

The code is identical in v5.1.2, v5.2.2 and main.

**Fix** (`cycles_settings.enable_metal`, read back into manifest.json):
- `CYCLES_METAL_DISABLE_BINARY_ARCHIVES=1` makes `should_use_binary_archive()` false, so no worker thread reaches
  `path_cache_get`. It is set via `os.environ` before the first render; `getenv` is read per pipeline compile.
- `CYCLES_METAL_PROFILING` and `CYCLES_METAL_DEBUG` must stay unset: they reach the same function from device init.
- `kernel_optimization_level = 'OFF'` drops the per-scene specialised kernels, so the whole render uses one kernel
  set with no mid-render switch.
- The Blender version is asserted to be 5.2.2, the reference version since 2026-09-29 by the user's decision. The
  reference cache key already includes the version string.

**Effect on the gates.** None on the expectation; both changes only affect how kernels are compiled. All references
re-render under 5.2.2 anyway.

## D8. Glass lobe-selection weight `sw_G` follows Cycles 5.1.2, not 5.2.2 (noise-only)

**Change in 5.2.** Cycles 5.2 changed how `bsdf_microfacet_estimate_albedo` estimates T for glass-type closures (the
Principled transmission lobe and the Glass node).
- 5.1.2 used the smooth-surface value `(1−F_gs(μ;η))·tint`, with TIR giving 0.
- 5.2 uses the LUT form `(1−mix(f0,1,lut3(r,μ,z)))·tint`, with no TIR zero.

**Where it matters.** The estimate feeds only closure picking and the one-sample normalisation `sum_pdf/sum_sample_weight`,
so Cycles' **expectation is unchanged** and only its variance moves (blender-5.2-migration.md §C3).

**What we do.** Our lobe-selection pmf q(ℓ|V) (math.md §12 "Sample weight") keeps the 5.1.2 form:
- every pdf in our code is the realized sampler density of our own q, so q never has to equal Cycles';
- our NEE/BSDF MIS already uses the balance form, not Cycles' power heuristic.

**Effect on the gates.** None on the expectation. D5's documented approximate-tier excess on G5b depends on Cycles'
normalisation and may shift slightly; G5b stays in the approximate tier.

## D9. Tangents: our MikkTSpace on the package mesh vs Blender's (M7, measured equal)

Blender computes MikkTSpace itself (`Mesh.calc_tangents`) on the package mesh; we run the Rust MikkTSpace port on the
same corners (whole mesh, corner normals, UV map, w negated for the bridge's v flip; m7-api.md N4). Different
implementations, same algorithm: U-TAN-B (gate-m7 loader part) measures sign agreement on every normal-mapped corner
and ≤ 0.0036° max angle on the M7 packages (gate: p99.9 ≤ 0.01°, max ≤ 0.05°; oct-15 storage adds ≤ 0.0049°). No
effect on the expectation beyond that.

## D10. Smooth shading and normal maps: Cycles' NEE/BSDF MIS is not a partition (M7, model-approximate tier)

Where `Ng·L ≤ 0 < N·L` (smooth shading, gap-bsdf §8.2) or where the bump-shadowing term rejects a direction a glossy
sample still reaches (math.md §29), only one technique contributes and its MIS weight is < 1, so the expectation depends
on the MIS heuristic (Cycles power, ours balance), the lobe pmf and the light pmf. We implement Cycles' rules exactly
(no Ng test in eval, the bump term in eval + diffuse sampling, `use_bump_map_correction` off) but not its heuristic, so
smooth-shaded Stage-A scenes run in the **model-approximate tier** (m7-api.md N6: TOST at the unchanged δ gates, the
Δ = 0 rejection checks are reported). Flat geometry with normal maps is a partition (math.md §29) and stays tight.
Stage B is unaffected (ReSTIR and our PT share F, ω, p).

## D11. E2E stock imports: importer conventions (M7, model-approximate tier)

E2E-GLB / E2E-USD render Blender's own importer output. Measured Blender 5.2.2 conventions our loader reproduces only in
the E2E (Blender-compatible) mode: USD SphereLight energy π·i with `normalize` kept (Cycles: an un-normalised point at
radius 0 is 4× a normalised one), DistantLight ×4, the exporter-only `specular` input ignored (m7-api.md §3.4, M7-6 /
M7-7), and — a Cycles convention rather than an importer one — an INSTANCED mesh's normals are interpolated in object
space and then transformed, while single-user meshes get per-vertex normalised world normals; under non-uniform
instance scale the two differ by up to ~3 % on low-poly smooth prototypes (M7-12; compat mode keeps unnormalised
`M^-T·n` for instance draws). glTF import (SPEC lighting, FLAT / NORMALS) needs no compat rule. Stock units are model-approximate: TOST at the
unchanged δ gates, so an unmodelled importer detail larger than δ (e.g. KHR_materials_volume) still fails the unit; a
sub-δ systematic is reported, not hidden (validation.md M7).

## Deferred

- **C0o visibleToCamera** (M3a): Mode B only. In Mode A Cycles 5.1.2 does not show camera-visible area lights when no
  light has MIS, so `exportScenePackage` and `build_scene.py` refuse them. **Covered in M3b** by
  `c0o_visible_camera_B_256` (Mode B).
