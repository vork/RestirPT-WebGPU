# M7: normal maps, smooth shading in ReSTIR, USD completion, loader fidelity and the stock-import E2E tests

Status: **normative for M7** (branch `m7-usd`, based on main b5b0f5d = M0–M6, every gate green). This contract extends
`restir-api.md` (M4), `restir-temporal-api.md` (M5), `restir-m6-api.md` (M6), `usd.md`, `scene-bridge.md` and
`data-formats.md`; every rule there stays binding unless an item below amends it and names it. Precedence (PLAN §4.2):
PLAN §0–§2 > gap-* (gap-bsdf §8 for normals, gap-rc for the predicate) > *-verify > critique > original reports
(`gris-math.md`, `enhanced-paper.md` are the authority for the Jacobian) > this contract's own choices.

Normative math: `docs/math.md` §29 [M7 addition] (shading normals: the Normal Map node, the bump-shadowing term, where
N, Ns and Ng enter the BSDF, the shift Jacobian and the reconnection predicate).

Contents: §0 decisions · §1 normal maps and smooth shading · §2 bitwise identity · §3 USD, loader fidelity, E2E ·
§4 scenes · §5 plants · §6 gate · §7 debug views · §8 performance · Changelog.

---

## 0. Decisions at a glance

| # | Decision | Why |
|---|---|---|
| N1 | **Normal maps are a composer define.** `NORMAL_MAP` is set by `SceneGpu.defines()` iff the scene has a material with a `normalTexture` (`sceneHasNormalMaps`); only then does the vertex arena carry the tangent section and do `SurfaceHit`, `MatEval`, `BsdfCtx` gain `nsm`. Without it the composed WGSL of every pipeline is the M6 text (U-M7-BITS) and the packed scene bytes are the M6 bytes (U-M7-ARENA). | Every M4–M6 Stage-B result, golden and cached PT reference stays valid by construction (as M6's MD1). |
| N2 | **Cycles' Normal Map node, literally** (svm/tex_coord.h): `c = 2(rgb − ½)`, `c.xy ·= s`, `c.z = mix(1, c.z, saturate(s))`, `N = safe_normalize(c.x·T + c.y·(sgn·nU × T) + c.z·nU)` with the UN-normalised interpolated tangent T, sign and normal nU (Ng on flat faces, before the backface flip); `N` is inverted for backfacing hits; zero / non-finite ⇒ the unmapped normal. | PLAN §5 M7 "Cycles node math"; gap-bsdf §8.4. |
| N3 | **Bump-shadowing term as Cycles** (closure/bsdf.h `bump_shadowing_term`, `use_bump_map_correction` off in the references): reject `(Ns·L)(Ns·N)(N·L) < 0` in EVAL for every closure (the NEE all-lobe f, `bsdf_eval`, `bsdf_query.f_all`) and in SAMPLING for the diffuse lobe only; glossy / glass samples keep their weight. | Cycles-exact; gap-bsdf §8.4. U-NM-2 checks the four places. |
| N4 | **Tangents with Blender semantics, generated on package read.** Packages carry no tangents (Blender recomputes MikkTSpace); `withSceneTangents` runs MikkTSpace (Rust port) over the whole mesh in primId order from the corner normals (stored normals; f32 face normals on TRI_FLAT faces of flat packages) and glTF-convention UVs, w negated (the bridge's v flip), splitting vertices whose corners disagree. Stored as oct 2 × 15 + present bit + sign bit (one u32, Q format) or 4 × f32 (lossless). | data-formats.md §B4 (0.0049° worst snap); scene-bridge.md "tangents are never exported"; U-TAN-B measures ≤ 0.0036° vs Blender. |
| N5 | **Smooth shading in ReSTIR needs no new rule.** The shift Jacobian and the reconnection footprints keep the GEOMETRIC normal at the receiving vertex (gris-math §9, gap-bsdf §8.1, math §18–§19); the BSDF at a reconnection vertex is evaluated with the vertex's own N / Ns (reconstructed from ids, `vertex_from_ids` flips `nsm` with `ns`); the sampler-support indicator (math §14) keeps p̂ = 0 outside the canonical sampler's support. Proven by T3-M7 (round trips on smooth / normal-mapped reconnections, f64 dual with interpolated normals) and rung 3.8. | PLAN §5 M7 "Smooth shading in ReSTIR". |
| N6 | **The model-approximate tier** (Stage A only): TOST at the UNCHANGED δ (0.5 % global, 2 % per 32² tile) and `min_replicates` gate; the Δ = 0 rejection checks (Šidák tiles, χ²_red, mean-t, KS/AD, the numeric-floor tile rule) are reported but not gating (`checks` in test.json ⇒ `notes.non_gating_checks`). Used for smooth-shaded Stage-A scenes (Cycles' NEE/BSDF MIS is not a partition where `Ng·L ≤ 0 < N·L`, gap-bsdf §8.2) and for the stock-import E2E units (importer conventions). δ is never loosened; Stage B is always tight. | PLAN §5 M7 "(vii-N) in the model-approximate tier"; gap-bsdf §8.2; validation-harness.md (Gate 2 tiers: "model-approximate … pass with the looser δ"; here stricter: δ unchanged, only the Δ = 0 checks are informational); validation/tools/README.md "Tension to be aware of". |
| N7 | **USD completion.** UsdUVTexture networks are rebuilt from the root-layer scan with Blender 5.2's importer semantics (§3.1); PointInstancer transforms from the authored arrays (unnormalised half quaternions, as pxr); material bindings inside instance prototypes from the scan. | PLAN §5 M7 "Full USD per usd.md". |
| N8 | **Stock-import E2E**: the Cycles reference is Blender's OWN importer (`import_scene.gltf` SPEC FLAT / NORMALS, `wm.usd_import`) on the original asset (`scene.json "stock"`), with only the §7.5 sampling rows applied; our side loads the same file with our loader (USD: Blender-compatible mode, §3.4). | PLAN §7.2 E2E-GLB / E2E-USD. |
| N9 | **E2E-HDR**: `scene.json env.original` names the downloaded Poly Haven file; `build_scene.py` loads it with Blender's decoder (not our env.exr) and records that Blender's pixels equal ours. | PLAN §5 M7 E2E-HDR. |
| N10 | **Plants** with predictions derived before measurement (§5): Normal Map sign / strength (Stage A and B), shading normal in the shift Jacobian (Stage B, detect), W × 1.003. | TD29 / E-18 rules. |
| N11 | **Seeds.** Stage A: our PT 7007 / 107007, Cycles 0..15 / 100..115; Stage B: PT 7001 / 107001, ReSTIR 7002 / 107002, pilots 7011 / 7012, A/A 7502 / 7503, plants 7101 + i (4× PT 7201 + i), Stage-A plants 7401 + i, PT A/A 7301 / 7302. | Disjoint from M3–M6. |

---

## 1. Normal maps and smooth shading

### 1.1 Where the normals enter (math.md §29)
- `scene_surface` (scene-data.wgsl) returns `ns` = the closure normal N (Normal Map output for a normal-mapped material,
  else the interpolated smooth normal, else Ng on flat faces) and, under NORMAL_MAP, `nsm` = the unmapped normal Ns.
- `material_eval` copies `nsm` into `MatEval`; `bsdf_ctx` flips it with `ns` (two-sided flip to V's side).
- Every lobe evaluates and samples around N; the sampler's below-surface tests use Ng (unchanged, math §14).

### 1.2 Tangent data (vertex-format.ts)
- Q format: one u32 per vertex after the colour section: bits 0–14 oct x, 15–29 oct y (snorm15), bit 30 present,
  bit 31 bitangent sign (1 ⇒ −1). F32: 4 words per vertex after the records. `verifyArena` round-trips the section.
- A vertex without the present bit (only corners of materials without a normal map, which never evaluate the node)
  decodes to T = 0, sign 0; a degenerate MikkTSpace corner gives T = 0 as in Blender, where the node then reduces to
  `N = normalize(c.z·nU)` on both sides.
- Cost: +4 B per vertex (Q), only in normal-mapped scenes.

### 1.3 BSDF (bsdf.wgsl)
- `bsdf_bump_ok(c, L) = !((Ns·L)(Ns·N)(N·L) < 0)`; `LobeEvals.bump`; `f_d` and the all-lobe f are zeroed when it fails;
  a diffuse SAMPLE failing it is rejected (no weight). Specular / glass lobes are untouched (N3).
- Without normal maps Ns = N, so the term is identically 1 (and compiled out).

### 1.4 ReSTIR
- `vertex_from_ids` (rc.wgsl) reconstructs ns and nsm from ids and flips both to the incoming side.
- Shift Jacobian, footprints F_k / I_k and the primary threshold use Ng (N5). `RS_PLANT_SMOOTH_J` (validation only)
  swaps Ng for Ns in the shift's receiving geometry term to prove the rule matters (§5.2).

---

## 2. Bitwise identity for scenes without normal maps

- **U-M7-BITS** (`tests/scene/m7-wgsl-bits.test.ts`): 390 composed pipelines (PT batch / frame / probe / glass plants,
  primary, emission, env-debug, every ReSTIR pass under the M6 variant sets, debug views; both vertex formats, with and
  without textures) hash to the text recorded on b5b0f5d (comments / whitespace removed); `NORMAL_MAP: false` ≡ absent.
- **U-M7-ARENA** (`tests/scene/m7-arena-bits.test.ts`): five committed packages load (`readScenePackage` incl.
  `withSceneTangents`) and pack (`packVertexArena`) to the bytes of b5b0f5d (digests recorded with `git archive b5b0f5d
  src`).
- Together: every M4 / M5 / M6 result, golden and cached reference stays valid without re-running.

---

## 3. USD, loader fidelity, E2E

### 3.1 UsdUVTexture (usd-textures.ts)
Measured Blender 5.2.2 `wm.usd_import` behaviour, adopted for our loader (the only reference we can render against):
one Image Texture per UsdUVTexture, Linear; colour space from `sourceColorSpace` (auto ⇒ sRGB for colour inputs);
ONE extension mode = `wrapS` (`repeat` when unauthored; `wrapT` ignored); a connected single channel through
Separate Color; `normal` → Normal Map node (tangent space, strength 1; scale / bias ignored); the connected input's
constant ignored. Fixed MaterialData slots: diffuse(+opacity of the same texture) → baseColor; roughness + metallic
→ metallicRoughness (the file itself if G / B of one texture, else a synthesised RGBA8 texel-for-texel pair);
normal → normalTexture; emissive → emissiveTexture. Anything else is warned and the constant used.

### 3.2 Loader fidelity
- **(viii-L)** `m7-loader-fidelity.ts` vs `validation/tools/usd_pxr_dump.py` (Blender's bundled pxr): per draw (instance
  proxies, PointInstancer instances) triangle count and canonical-frame bbox (1e-5 · extent), per material the
  PreviewSurface constants and texture bindings, per light the converted LightData vs `convertUsdLight` on pxr's fields
  and world matrix, per camera FOV and matrix.
- **(vii-L)** vs `validation/blender/import_dump.py` (Blender's stock glTF importer): total triangles (non-zero area),
  triangles per material, bbox, Principled inputs, lights. Name rules: Blender suffixes `.001`, names unnamed
  materials `Material_<i>`, calls unbound faces `<none>` (ours `__default`); lights matched by type and position.
- **U-TAN-B** `m7-tangent-check.ts` vs `validation/blender/tangent_dump.py` (Blender's `calc_tangents` on the package
  mesh): sign equal on every normal-mapped corner, angle p99.9 ≤ 0.01°, max ≤ 0.05°.

### 3.3 E2E-HDR
`make-m7.ts` writes `m7_xivlite_{hdr,exr}_256` with `env.original = {file, sha256, format}`; `build_scene.py
build_env_original` loads that file in Blender, compares Blender's pixels with OIIO and with our decode, and records
`original`, `format`, `blender_equals_ours`; the gate requires all three.

### 3.4 Stock-import E2E (make-m7-e2e.ts, build_scene.py `build_from_stock`, cycles_settings.py `stock_light_rows`)
- 21 units: 8 glTF assets × {FLAT, NORMALS} (Cornell point + spot, MetalRoughSpheresNoTextures, TextureTransformTest,
  NormalTangentMirrorTest, AlphaBlendModeTest, EmissiveStrengthTest, TransmissionTest, IORTestGrid) and 5 USD files
  (Blender Cornell, hand files Y-up cm / Z-up m, textured, instancing). Assets without lights get an `origin: 'added'`
  rect (built from the package on both sides) and the overcast HDRI; imported lights are kept as the importer made them.
- Blender keeps the importer's energy / colour / normalize / temperature / exposure / size (asserted in the manifest)
  and gets only: point / spot radius 0, no soft falloff, sun angle 0, the light mode's MIS, not camera-visible.
- **Blender-compatible USD mode** (`usdToScene(..., {blenderCompat})`), measured on 5.2.2: SphereLight → energy π·i,
  `normalize` kept, so at radius 0 power = (normalize ? 1 : 4)·π·i (Cycles renders an un-normalised point 4× a
  normalised one at r = 0); DistantLight ×4 always; the Blender-exporter-only `specular` input is NOT read (Blender's
  importer keeps Specular IOR Level 0.5). Default (non-compat) mode keeps usd.md's rules (Blender round trip).
- Node-side PNG decode: palette / interlaced PNGs fall back to sharp in make-m7-e2e (the app decodes in the browser).

---

## 4. Scenes (make-m7.ts, m7-kit.ts, m7-fixtures.ts)

| Package | Content | Stage A tier | Stage B |
|---|---|---|---|
| m7_smooth_256 | smooth spheres (dielectric, gold), torus, OPEN wavy sheet; rect + point | model-approximate | 3.8 initial, offline-m6 (first B-SM-J: not detected, M7-11) |
| m7_smooth_lowpoly_256 | the same with low-poly meshes (Ns up to ~40° from Ng; M7-11) | model-approximate | 3.8 offline-m6; B-SM-J plant |
| m7_nm_flat_256 | flat geometry + normal maps (tiles floor, bumps wall, waves wall, bumps box), panels P1–P3 | tight | 3.8 offline-m6; NM plants |
| m7_nm_smooth_256 | smooth + normal-mapped spheres / torus / open sheet | model-approximate | 3.8 initial, offline-m6, chains full-m6; A/A |
| m7_nm_smooth_B_256 | the same, light mode B | model-approximate | 3.8 offline-m6 Mode B |
| m7_nm_env_256 | normal-mapped spheres on a tiles ground, overcast HDRI | model-approximate | 3.8 offline-m6 |
| m7_xivlite_{hdr,exr}_256 | (xiv)-lite under the ORIGINAL Poly Haven file | tight | — |
| 21 × e2e_* | §3.4 | model-approximate | — |

T3 fixtures `t3_smooth_256`, `t3_nm_256` (m7-fixtures.ts) for T3-M7.

---

## 5. Plants (validation/scenes/m7-plants.ts; tests/scene/plant-m7.test.ts pins the signs)

### 5.1 Stage A (our PT with NM_PLANT vs Cycles; plant_sign.py on the panel masks, 15 % inset)
- **A-NM-sign** (`NM_PLANT 1`, bitangent sign ignored): P2 (mirrored u) tilts 35° DOWN instead of up ⇒ predicted
  **M_P2 −** (direct irradiance 5.836 → 2.573, −55.9 %).
- **A-NM-strength** (`NM_PLANT 2`, glTF-style strength: c.z not mixed): P3 30.0° → 40.9° down ⇒ **M_P3 −** (−32.6 %).

### 5.2 Stage B (ReSTIR with the plant in every pass but the path tree vs a 4× PT)
- **B-NM-sign** (`m7NmSign`), **B-NM-strength** (`m7NmStrength`): the same normals in shifts / replay / refresh; the
  same sign on the same panel (shifted contributions through the panel use the planted normal).
- **B-SM-J** (`RS_PLANT_SMOOTH_J`, `m7SmoothJ`; on `m7_smooth_lowpoly_256`, M7-11): Ns instead of Ng in the receiving geometry term of the shift; ratio
  cos_s/cos_g ≠ 1 wherever Ns ≠ Ng, sign varies ⇒ **detect** (global).
- W × 1.003 (synthetic) + calibrate A/A re-splits; a ReSTIR A/A.

---

## 6. Gate (`npm run validate -- --milestone M7 [--part …]`, validation/harness/gate-m7.ts)

### 6.1 Parts
- **core**: typecheck, cpu lane (U-M7-BITS, U-M7-ARENA, tangents, plant-m7, gate-m7-config + every earlier CPU test),
  python tests, make-m7 / make-m7-e2e determinism; Chrome: `normal-map` (U-NM-1 decode vs the f64 node incl. mirrored
  UVs, backfaces, both formats; U-NM-2 bump term in the four places; U-NM-3 tilted-normal furnace closed form
  `ρ(1 + cos θ')/2`), T3-M7 (`t3_smooth_256`, `t3_nm_256`: LOGIC 0, every bin and the nmRc / smoothRc counters ≥ 10⁶),
  the M6 / M5 / M4 Gate-0 suites as regressions, the perf probe (recorded), the M7 app smoke.
- **loader**: (viii-L) 7 USD files, (vii-L) 9 glTF files (incl. Sponza), U-TAN-B on 5 packages.
- **stageA**: §4 Stage-A column + a PT A/A; Cycles 4096 spp × 16 (D4), ours 4096 × 16, confirmatory re-run on disjoint seeds.
- **e2e**: the 21 E2E units, Stage A model-approximate.
- **r38**: rung 3.8 Stage B units of §4 (pilot sizing as gate-m6, PLAN §7.3 × 1.25) + chains.
- **plants**: §5.

### 6.2 U-TAN-B — see §3.2.

### 6.3 Sizing
Stage A fixed (pilots at 1024 × 8 showed MDB_global ≤ 0.05 % and tile MDB well below the δ/6.4 target); compare.py
reports `replicate_multiplier_needed` per unit. Stage B as gate-m6 (`sizeScene`).

### 6.4 T16
Every PT / ReSTIR run records `scene.normalMaps` in meta.json; the gate requires it to equal the package's; the M6
T16 (light mode, settings, plants named only where named) applies to the Stage-B runs; the PT plant runs record
`config.plant.nm`.

---

## 7. Debug views 320–327 (shading-debug.ts / passes/shading-debug.wgsl)
320 closure normal N, 321 unmapped Ns, 322 tangent T, 323 bitangent sign·(Ns × T), 324 normal-map decode c, 325
angle(N, Ns) [°], 326 angle(Ns, Ng) [°], 327 bitangent sign code. A separate pass (compiled on first selection) at
primary hits; without normal maps 322 is 0 and 324 is (0, 0, 1).

## 8. Performance
540p (960 × 540) interactive, Mode B, all M6 features (m7-perf.gpu.test.ts, median per pass): m7_nm_smooth 27.2 ms with
normal maps vs 26.5 ms with the same scene's normal textures removed (+0.7 ms, +2.6 %; rs_initial +0.4 ms); Cornell (i)
16.6 ms (no NORMAL_MAP: the M6 pipelines).

---

## Changelog

(append-only; amendments made while implementing)

- **M7-1 (sphere caps, found by U-TAN-B).** m7-kit's smooth sphere caps had reversed winding and degenerate polar UVs:
  Blender and our MikkTSpace disagreed in sign on 528 / 864 cap corners. Caps now use planar UVs with the correct
  winding; 0 mismatches. Scene change before any Stage-A measurement.
- **M7-2 (P3 redesign before measurement).** The first P3 (50° toward +B at s 0.5) sat at the panel's irradiance
  maximum (predicted Δ +0.1 %, below detection). Re-derived BEFORE any planted render: 60° toward −B at s 0.5
  (monotone region), −32.6 %.
- **M7-3 (PointInstancer orientation).** LightUSD normalises `quath` orientations, pxr does not (an unnormalised
  quaternion scales the instance); (viii-L) showed 3e-5 bbox errors. Transforms are now rebuilt from the authored
  arrays exactly as pxr.
- **M7-4 (bindings inside instance prototypes).** LightUSD reports no material for meshes inside instanceable
  prototypes (`__default`); the binding and the PreviewSurface constants are now read from the root-layer scan.
- **M7-5 ((vii-L) comparator rules).** Zero-area triangles excluded on both sides; Blender's material naming
  (`.001`, `Material_<i>`, `<none>`) and light-object naming mapped as §3.2.
- **M7-6 (Blender-compatible USD lights, measured).** The first compat rule (π·i for every sphere light) was −34 % on
  `hand_yup`; Cycles renders an un-normalised point at radius 0 4× a normalised one (0.2084 vs 0.0521), hence
  (normalize ? 1 : 4)·π·i (§3.4).
- **M7-7 (Blender-compatible USD: `specular` ignored).** e2e_usd_cornell had side walls −6…−9 % (red wall G/B −30 %):
  the Blender exporter writes `specular = 0`, which our default mode maps to specularFactor 0 (Blender round trip),
  but Blender's importer ignores it (Specular IOR Level 0.5, verified in bpy). Compat mode ignores it; the unit then
  passes (Δ_Y +0.007 %, 1024 × 8 pilot).
- **M7-8 (palette PNG in the E2E generator).** TransmissionTest's palette PNGs were not decoded by io/png.ts (Node
  subset) ⇒ textures dropped ⇒ +41.8 %. make-m7-e2e falls back to sharp; the app's browser decoder was never
  affected.
- **M7-9 (Stage-A sizes).** Pilots ran Cycles at 1024 spp; the gate uses ≥ 4096 spp (cycles-deviations D4). The
  sub-δ systematic of MetalRoughSpheres (+0.07 % at 1024 spp, TOST pass, rejection checks fail) is reported in the
  model-approximate tier, not hidden; see validation.md M7.
- **M7-10 (USD meshes without authored normals).** The gate's first E2E run failed `e2e_usd_instancing` (tile [3,1]
  Y +1.47 %, R / B tiles beyond 2 % on the PointInstancer pyramids and gems; confirmed on the disjoint re-run seeds):
  our loader averaged unit face normals UNWEIGHTED in world space, while Blender 5.2's importer leaves such faces smooth
  with its automatic normals (corner-angle-weighted, per mesh, verified in bpy: Pyr base corners (±0.6555, ±0.6555,
  −0.3752)). The loader now computes Blender's normals on the mesh in its own space and transforms them like authored
  normals (non-uniformly scaled instances keep Blender's weighting). LightUSD rc4 synthesises normals of its own for such
  meshes (Pyr vertex 0: (−0.707, −0.707, 0)), so "not authored" is decided from the root-layer scan (faceVertexCounts
  without normals / primvars:normals). Affects only USD meshes without normals (in the repo: m7_instancing's prototypes,
  spike_hand.usda); the other E2E USD packages are byte-identical; the unit was re-run from scratch on the gate seeds.
- **M7-11 (B-SM-J revised).** The planned B-SM-J on `m7_smooth_256` was **not detected** (0/10 half-size repeats, full
  comparison pass, global Δ −0.000 %, z −0.01; plants run `m7-gate-plants-…`): its meshes are finely tessellated
  (sphere 24 × 48, torus 40 × 20, sheet 24²), so Ns and Ng at reconnection vertices differ by ≲ 4° and cos_s/cos_g ≈ 1.
  The prediction ("detect") was wrong for that scene, not the rule. Revised BEFORE any run on the new scene: a low-poly
  copy `m7_smooth_lowpoly_256` (spheres 4 × 6 / 4 × 7, torus 8 × 5, sheet 4²: Ns up to ~40° from Ng), measured on fresh
  disjoint seeds (ReSTIR 7801 + i, 4× PT 7901 + i, E-18). The scene also joins Stage A (model-approximate) and rung 3.8
  (offline-m6), so the plant's unplanted twin is itself shown unbiased.
