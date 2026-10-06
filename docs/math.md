# Normative formula reference (math.md)

This is the one place the renderer's formulas are written down. WGSL modules cite it as
`// math.md#<anchor>`, for example `// math.md#jacobian`. Every section starts with an explicit
`<a id="…"></a>` anchor. Anchor names are stable: renaming one breaks the WGSL citations.

**Precedence** (PLAN §4.2). PLAN §0–§2 > gap-* specs and `gap-glass` > `*-verify` > `critique` >
original reports. The PLAN §4.2 list of superseded statements applies. Where two sources disagree,
this document follows the higher-ranked one and records the conflict in
[Open inconsistencies](#open-inconsistencies). Nothing is resolved silently.

**Citation keys.** `plan` = `docs/PLAN.md`. The research files are in `docs/research/`:
- `gap-rc` = gap-rc-predicate.md
- `gap-light` = gap-light-integration.md
- `gap-temporal` = gap-temporal-moving-lights.md
- `gap-bsdf` = gap-bsdf-lobe-model.md
- `glass` = gap-glass.md
- `env` = gap-env-hdri.md
- `env-verify` = gap-env-hdri.verify.json
- `gris` / `gris-verify` = gris-math.md / gris-math.verify.md
- `enh` / `enh-verify` = enhanced-paper.md / enhanced-paper.verify.md
- `cyc` / `cyc-verify` = cycles-conventions.md / cycles-conventions.verify.md
- `val` = validation-harness.md
- `io` = scene-io.md
- `review Rn` = plan-review.txt finding Rn

`K/` is the Cycles kernel source of the pinned reference Blender: 5.2.2 since 2026-09-29 (formulas re-audited unchanged from 5.1.2, docs/decisions/blender-5.2-migration.md; line numbers cited below are mostly 5.1.2).

**Notation used throughout.**
- `·` is a scalar product or multiplication. `⊙` is component-wise RGB multiplication.
- `lum(c) = 0.2126 c.r + 0.7152 c.g + 0.0722 c.b` (Rec.709). This is the ReSTIR target
  scalarisation and the test channel. Any fixed scalarisation is unbiased as long as every pass uses
  the same one (enh-verify §3). One WGSL function `lum()` is used everywhere.
- `avg(c) = (c.r + c.g + c.b)/3` is Cycles' `average()`, used only where Cycles uses it (sample
  weights, P_R, env importance).
- `saturate(x) = clamp(x, 0, 1)`.
- `smoothstep01(f) = 0` if f ≤ 0, `1` if f ≥ 1, otherwise `3f² − 2f³`.
- Vectors are unit length unless stated. `n^g` is the geometric (face) normal. `N` / `Ns` is the
  shading normal used by a closure. On flat faces (`TRI_FLAT`, set by quantizeScene when every corner normal is the face
  normal) `Ns ≡ n^g` bit for bit, as Cycles uses Ng on flat faces (data-formats.md §B3).
- **BSDF direction convention (Cycles).**
  - `V` points from the shading point toward the previous (camera-side) vertex.
  - `L` points toward the next vertex or the light.
  - "eval" or `f_cos` means `f_s(V,L)·|N·L|`, i.e. it includes the cosine (gap-bsdf §0).
- `FLT_MAX = bitcast<f32>(0x7f7fffffu)`. IEEE Inf is never computed (plan §1.7).

---

<a id="indices"></a>
## 1. Index conventions

Source: gap-rc §1 (copied verbatim, per review R14), plan §1.7.

| Concept | This project / Enhanced paper | Falcor 2022 (`DQLin/ReSTIR_PT`) | RTXDI | EvanLuo42 |
|---|---|---|---|---|
| Camera | x₀ | – | – | – |
| Primary hit | x₁ | `path.length = 0` | bounce depth 1 | surface vertex 1 |
| rc (reconnection) vertex | x_k, k ≥ 2 | `rcVertexLength = k−1` (1 ⇒ x₂) | `rcVertexLength = k` | `reconnectionVertex = k` |
| Last vertex | x_d (light, emitter or env) | `pathLength = d−1` (index of the last scattering vertex) | `pathLength = d` | `getPathLength` |
| Needs replay | **k > 2 ∨ k = ∅** | `rcVertexLength > 1` | `rcVertexLength > 2` | `reconnectionVertex > 2` or invalid |

Definitions:
- **Path.** x̄ = [x₀, x₁, …, x_d].
  - x₀ is the camera position (pinhole).
  - x₁ … x_{d−1} are scattering vertices on triangles.
  - x_d is the light vertex: a point on an emissive triangle or an analytic area light, a delta
    light position, the sun direction, or an env direction `(ENV, ω)`.
- **d** is the index of the light vertex, which is also the number of segments.
  - The reservoir holds **d ≥ 2** only. d = 1 (the camera sees an emitter, a camera-visible
    analytic light or the env) is added outside the reservoir with weight 1
    ([path-tree](#path-tree)).
- **k** ∈ {2, …, d} ∪ {∅}.
  - k = ∅ means no reconnection: full random replay, with J = 1.
  - k = 2 means rc at x₂, with x_{k−1} = x₁ and no replay pass.
- **B** is the vertex index of x_{d−1}, the last scattering vertex. It is the argument of M(B)
  ([mis](#mis)).
- **ℓ_i** is the lobe code used at x_i, for 1 ≤ i ≤ d−1 ([bsdf-v2](#lobe-codes)). ℓ_{d−1} = NEE
  means the outgoing direction at x_{d−1} was NEE-sampled, and the BSDF there is evaluated with
  all lobes.
- **ω_i** = (x_{i+1} − x_i)/‖x_{i+1} − x_i‖ is the direction leaving x_i.
- **Base and offset.** The base path is x̄ and the offset (shifted) path is ȳ. Replayed vertices
  are y₁ … y_{k−1}, and the copied suffix is x_k … x_d. The reconnection direction is
  ω' = normalize(x_k − y_{k−1}).
- **pdfs.**
  - `p_i(ω, ℓ)` is the joint pdf at x_i of sampling direction ω **and** lobe ℓ, given incoming
    −ω_{i−1}.
  - `p̄_i(ω) = Σ_ℓ p_i(ω, ℓ)` is the marginal. See [pdf-conventions](#pdf-conventions).
- **G(a→b) = |⟨n^g_b, (a−b)/‖a−b‖⟩| / ‖a−b‖².** The cosine is taken at the **receiving**
  vertex b, with the geometric normal and an absolute value (enh-verify C8, gris Eq. 52).
  - The paper and supplement wording "normal at y_{k−1}" is a slip; see open item 21.

---

<a id="bounces"></a>
## 2. Bounce semantics (Cycles `max_bounces`)

Sources: plan §1.2, cyc §5.1, cyc-verify §4 item 7, glass §2.5, gap-light §2.2.

- Blender `max_bounces = N` sets `kintegrator.max_bounce = N + 1` (`scene/integrator.cpp:192-193`).
  - The camera hit x₁ arrives with bounce 0.
  - NEE happens at x₁ … x_{N+1}.
  - Emission (BSDF hit or escape) is collected at x₁ … x_{N+2}.
- **So there are at most N+1 scattering vertices and d ≤ N + 2.** N = 0 is direct lighting only
  (d = 2).
- Every scattering event counts as one bounce. That includes each glass interface (reflection or
  transmission, rough or singular) and each mirror.
- The per-type caps (diffuse, glossy, transmission) are all set equal to N in the reference, so
  only the total counts. `volume = 0`.
- **Consume no bounce, no RNG dimension and no RR:**
  - alpha-MASK cutouts, which are a traversal-level geometric cutout;
  - Mode-B pass-throughs of analytic lights, which are "transparent" in Cycles.
- Cycles counts light pass-throughs as transparent bounces. The reference therefore pins
  `transparent_max_bounces = 1024`, and our renderer applies no cap (gap-light §2.4).
- The path-length field is 4 bits with d ≤ 15, so N ≤ 13 (gap-light §2.2).
- RR is **not** part of Cycles-parity semantics. The reference pins
  `min_light_bounces`/RR as it chooses, since RR is unbiased. Our RR rules are in [rr](#rr).

---

<a id="raster"></a>
## 3. Pixels, raster, camera ray, jitter

Sources: plan §1.2, cyc §5.4 and §6, cyc-verify C1, review §1 raster rule.

**Pixel ↔ Cycles raster.**
- Image pixel (c, r) has row r = 0 at the **top** of the saved image. It covers Cycles raster
  `[c, c+1] × [H−1−r, H−r]`. Cycles raster y = 0 is the **bottom** of the viewplane.
- Blender forces the BOX filter width to 1.0, so a Cycles pixel is the uniform average over that
  raster square (cyc §5.4).

**Camera ray** (pinhole, no DOF, no shift; the camera looks down its local −Z with +Y up in both
glTF and Blender).
```
(u, v) ∈ [0,1)²  subpixel jitter
t_y   = tan(vfov/2)
t_x   = t_y · W/H
d_cam = ( (2(c+u)/W − 1)·t_x ,  (2(H−1−r+v)/H − 1)·t_y ,  −1 )
d     = normalize(R_cam · d_cam)          R_cam = camera-to-world rotation (glTF canonical frame)
o     = camera position (render-internal recentred frame)
```
- Blender side: `sensor_fit = 'VERTICAL'` and `angle_y = vfov`. **Never** read `Camera.angle_x`
  as the render's horizontal FOV (cyc-verify C1). At 16:9 with vfov = 40°, the true hfov is
  65.81°.
- Clip planes are 1e-4 and 1e5, clipped against planes (`K/camera/camera.h:160-170`). This
  matters only for degenerate scenes.

**Jitter.**
- Every frame draws fresh jitter and builds a fresh V-buffer. Each frame's jitter defines its own
  pixel domain Ω_{t,p} (gap-temporal §1). The ensemble over jitter equals Cycles' BOX-1.0 pixel
  integral.
- **Validation:** `(u, v) = u01²(hash(runSeed, t, p, JITTER))`, i.i.d. per run and per frame. It
  is a separate stream from the path stream and the resampling stream (plan §1.2).
- **Interactive:** Halton or R2 with a per-run Cranley–Patterson rotation is allowed.
- The **jittered** primary hit defines `thr` ([rc-predicate](#rc-predicate)) and the temporal
  domain. The previous frame's jittered V-buffer is stored ([temporal](#temporal)).

**Render-internal recentring.**
- `p_int = p_world − O`, with one offset O applied identically to triangles, camera, lights and
  tracks. Directions and env mapping are unaffected.
- **Geometry precision (data-formats.md §B1).** Positions lie on one global lattice `p = n·2^k` (|n| < 2^24,
  k = max(⌈log2(E/(2²¹−2))⌉, ⌈log2(A/(2²⁴−1))⌉), E = scene extent, A = max |coordinate|; Sponza k = −16 = 15.3 µm, the
  validation scenes ≤ 61 µm steps, worst displacement 29 µm). O is snapped to the same lattice, so `p_int` is exact in
  f32, as are all triangle edge vectors; the GPU decodes `(n − n_O)·2^k` bit-identically to the CPU value. Shading
  normals are oct 2 × 16 snorm (≤ 0.0025°), UVs per-material dyadic 2 × 16 (≤ 1/8 texel, else f32).
- `scene.json` stays un-recentred, in the glTF canonical frame: right-handed, +Y up, metres
  (plan §1.2).
- **Axis conversion to Blender.** `C = R_x(+90°)`, i.e. `(x, y, z)_glTF → (x, −z, y)_Blender`.
  `build_scene.py` applies it exactly once. Object matrices convert as `M_b = C·M_g`, and
  object-local axes are unchanged (cyc-verify §4 item 11).

---

<a id="units-lights"></a>
## 4. Light units and emission (Cycles-exact)

Sources: plan §1.2 and §1.4, cyc §2–§3, cyc-verify C5/C6 and §4, gap-light §3.2–§3.6.

**Common definitions.**
- Blender power: `Φ = color ⊙ energy · 2^exposure`. Temperature is off, and the light's default
  node tree (Emission (1,1,1), strength 1) is kept. Validation lights have `normalize = True` and
  are **unscaled**; scale would distort the spot cone and the area size.
- Light-local axes, as world vectors: `a_u = X_obj`, `a_v = Y_obj`, and emission axis
  `a_L = −Z_obj`. The direction toward the sun is `s_L = +Z_obj`.

| Type | Emission (radiometric) | Visibility | Source |
|---|---|---|---|
| Point, r = 0 | Radiant intensity `I = Φ/(4π)` W/sr. `eval_fac = 1/(4π)`, and Cycles' pseudo-pdf is t² (not a density) | never hit by any ray | cyc §2.2(a); `K/light/point.h:65-77, 138-141` |
| Spot, r = 0 | `I(ω) = Φ/(4π) · S(cosθ')` (see below). **Not** cone-normalised: on-axis I equals a point light of the same Φ | never hit | cyc §2.3; `K/light/spot.h:15-31` |
| Rect/square | `L_e = Φ/(π A) · spread(θ)`, `A = len_u·len_v`. **One-sided**: `L_e(z→x) = 0` unless `dot(x − z, a_L) > 0` | Mode B: hit by BSDF rays (pass-through). Camera only if `visibleToCamera` | cyc §2.4; `K/light/area.h:106-121, 313, 337-341, 405-408` |
| Disk | as rect, with `A = (π/4)·len_u·len_v` (size = diameter) | as rect | cyc §2.4 |
| Sun, angle 0 | Irradiance on a surface perpendicular to `s_L`: `E = Φ` (W/m²). pdf = 1, eval_fac = 1 | never hit | cyc §2.5, cyc-verify §4.5 |
| Emissive triangle | `L_e = strength·color` (glTF: `emissiveFactor ⊙ sRGB⁻¹(tex(uv)) · emissiveStrength`). **Two-sided**, no π, opaque, occluding | always hittable | cyc §3; `K/closure/emissive.h:43-60` |

**Spot profile** (cos-domain smoothstep; cyc §2.3, gap-light §3.3).
```
cosθ' = dot( normalize(x − c), a_L )                 x = shading point, c = light position
cosH  = cos(spot_size/2)                              spot_size = FULL cone angle, 1°…180°
S(t)  = smoothstep01( (t − cosH) / ((1 − cosH)·spot_blend) )
spot_blend == 0  ⇒  S(t) = (t > cosH) ? 1 : 0        // guard: Cycles computes (t−cosH)·∞
```
- The blend region is `cosθ' ∈ [cosH, cosH + (1 − cosH)·spot_blend]`.
- S is always evaluated at the **evaluating** shading point (the offset point in shifts), never
  cached (gap-light §3.3).

**Area-light spread** (`area_light_spread_attenuation`, `K/light/area.h:106-118`; cyc §2.4;
cyc-verify §4.4).
```
a   = spread/2,   0 < spread ≤ π                     (spread = 0 is a non-goal)
θ   = angle between the emitted direction (z → x) and the light normal a_L
spread == π  ⇒  spread(θ) ≡ 1                        (Cycles: tan_half = FLT_MAX, factor → 1)
else:  N_s  = (a > 0.05) ? 1/(tan a − a) : 3/a³      (Taylor branch, scene/light.cpp:1351-1355)
       spread(θ) = max( (tan a − tan θ)·N_s , 0 )
```
Power is preserved for any spread: `A·2π·∫₀^a L(θ) cosθ sinθ dθ = Φ` (cyc-verify §4.4).

**glTF `KHR_lights_punctual` → Blender** (importer `blender/imp/light.py`, default
`export_import_convert_lighting_mode='SPEC'`; cyc §2.9, cyc-verify §4 item 10 and omission 9).
```
point/spot:  Φ = cd · 4π / 683  [W]    ⇒  I = cd/683 W/sr
sun:         E = lux / 683  [W/m²]
spot_size  = 2·outerConeAngle
spot_blend = 1 − innerConeAngle/outerConeAngle     (angle ratio, NOT the glTF falloff curve)
'spot' present but a cone field missing → glTF defaults outer = π/4, inner = 0 ⇒ 90°, blend 1
glTF emissive: taken 1:1 (L_e = emissiveFactor ⊙ sRGB⁻¹(texture) · emissiveStrength)
shadow_soft_size = 0 (new-light default) ⇒ delta lights
```

**USD lights, v1 rules** (plan §1.2; cyc §4.9; cyc-verify C6). The Blender importer sets
`energy = intensity·π`, or `intensity·4` for DistantLight, times `2^exposure`. `inputs:normalize`
has USD fallback **false**. Let `i` be the USD intensity and `e` the exposure.
```
SphereLight → point, r := 0, radiant intensity preserved:
    normalize = true           : I = i·2^e/4                 (= energy/(4π))
    normalize = false, r > 0   : I = i·2^e·π r²              (soft-falloff disk radiance i·2^e times its area)
    normalize = false, r = 0 / treatAsPoint : I = i·2^e      [derived, cyc §4.9; not listed in plan]
SphereLight + ShapingAPI → spot: spot_size = 2·coneAngle, spot_blend = coneSoftness, same I rule
DistantLight → sun, angle := 0, E = 4·i·2^e                  (Blender-author quirk)
Rect/DiskLight (M2 minimal UsdSceneSource) [derived, cyc §4.9]:
    normalize = false : L_e = i·2^e                          (eval_fac = 1/π, no area term)
    normalize = true  : L_e = i·2^e/A                        (Φ = i·π·2^e)
DiskLight: size = 2·radius;  RectLight: size = width, size_y = height
Simplified lights (r or angle forced to 0) get a UI badge.
```

---

<a id="env-mapping"></a>
## 5. Environment mapping (Cycles-identical)

Sources: plan §1.4b and §1.2, env §1.1 and §2.3, env-verify (checked against `projection.h` and
`svm/image.h`).

```
d : unit glTF-world direction;  γ : Blender Mapping-node rotation Z (vector_type POINT), written unchanged by the bridge
cγ = cos γ, sγ = sin γ
b  = R_z(γ)·C·d = ( cγ·d.x + sγ·d.z ,  sγ·d.x − cγ·d.z ,  d.y )           C = R_x(+90°), used only inside envUV
u  = (atan2(b.y, b.x) − π) / (−2π)          = 0.5 − atan2(b.y,b.x)/(2π)
v  = (acos(clamp(b.z, −1, 1)) − π) / (−π)   = 1 − acos(b.z)/π
```
- **Orientation.**
  - u = 0.5 looks toward +X_glTF, u = 0.25 toward −Z_glTF, u = 0.75 toward +Z_glTF, and
    u ∈ {0, 1} toward −X_glTF.
  - v = 1 is the zenith (+Y) and v = 0 the nadir. The image is not mirrored.
  - The visible environment rotates by −γ about +Y.
- **Inverse `envDir(u, v)`.**
  ```
  φ = −2πu + π,  θ = −πv + π,  b = (sinθ cosφ, sinθ sinφ, cosθ)
  r = R_z(−γ)·b = ( cγ b.x + sγ b.y ,  −sγ b.x + cγ b.y ,  b.z )
  d = C⁻¹·r = ( r.x, r.z, −r.y )
  ```
- **Texture and sampler.**
  - Format `rgba32float`, rows **bottom-up** (row 0 = nadir, as in the Blender ImBuf), 1 mip.
  - Sampler `repeat/repeat`, `linear/linear`, lookup `textureSampleLevel(texEnv, sEnv, (u,v), 0)`.
  - This reproduces Cycles' `EXTENSION_REPEAT` quirk: within half a texel of a pole the lookup
    blends the top and bottom rows. Do not "fix" it.
  - **[M5 addition, restir-temporal-api.md C-10]** The lookup is an explicit f32 bilinear instead of the
    hardware sampler: x = fma(uv, (W, H), −½), i₀ = ⌊x⌋ wrapped (repeat/repeat, as above), t = x − ⌊x⌋,
    four `textureLoad`s and fma lerps; `envUV` / `envDir` use explicit fma and constant reciprocals of 2π, π.
    The hardware filter quantizes t (8 bit on Apple GPUs, the same root cause as platform-lanes Q4), which made
    ulp-level `envUV` differences between pipelines jumps of up to 2·10⁻³ in L_env on high-frequency maps, so
    the same stored path had different F in the path tree, the refresh and the shifts. The explicit lookup is
    continuous in uv, exact in f32 for every texel format, and bit-stable across pipelines. (Cycles on Metal
    keeps the quantized hardware weights: a ≤ 2⁻⁹·Δtexel per-lookup difference with zero mean over a footprint.)
- **Radiance.** `L_env(ω) = strength · tint ⊙ texel(envUV(ω))`. This one function is
  `envRadiance(uv, scale)`.
  - NEE-env samples evaluate `envRadiance(uv_sampled)` directly, with no direction round trip.
  - Escapes, camera misses and refreshes evaluate `envRadiance(envUV(ω))`.
- **Loading.**
  - RGBE decodes as `byte·2^(e−136)` for e ≠ 0 and 0 for e = 0, computed in f64 and stored as f32
    (bit-identical to OIIO). three.js HDRLoader is never used; it is +0.39%.
  - EXR is parsed by EXRLoader, which emits bottom-up rows. Validation accepts lossless codecs only.
  - **Negative texels:** rejected in validation, clamped to 0 in interactive mode (env-verify).
  - NaN/Inf: rejected in validation, replaced by 0 with a warning in interactive mode.
- The camera-miss term `visibleToCamera · L_env(d)` is added outside the reservoir with weight 1
  ([path-tree](#path-tree)).

---

<a id="env-sampling"></a>
## 6. Environment importance sampling (weights, alias, realized pdf, pole cap, cell clamp)

Sources: plan §1.4b and §2 rule 15, env §2.4, env-verify corrections 1, 2 and 10.

**Grid.**
- `W_m` = the largest power of two ≤ min(W, cap), with **W_m ≥ 4**. `H_m = W_m/2`.
  - cap = 4096 in validation, 2048 interactively.
- Row i is bottom-up in v: `v ∈ [i/H_m, (i+1)/H_m)`. Column j covers `u ∈ [j/W_m, (j+1)/W_m)`.
  `log2 W_m ≤ 16`.

**Cell weights** (Worker, f64, fixed loop order, deterministic; strength and tint are **excluded**,
so they never change the table).
```
a[r][c]  = (|R| + |G| + |B|)/3                               of the raw texels
B[r][c]  = Σ_{dr,dc ∈ {−1,0,1}} k(dr)·k(dc)·a[r+dr][c+dc],   k = (1/8, 6/8, 1/8)
           u periodic; v periodic too (row −1 ≡ row H−1: the Cycles repeat quirk)
B̂_ij    = mean of B over the texel block of cell (i,j)      (W a multiple of W_m;
           otherwise an area-weighted overlap of texel footprints)
s̄_i     = (H_m/π)·( cos(π i/H_m) − cos(π (i+1)/H_m) )      (row mean of sinθ = sin(πv))
w_ij     = s̄_i · B̂_ij
floors:    w_ij ← max(w_ij, 2⁻¹⁰·mean_all(w))
           R_i = Σ_j w_ij ;  w_ij ← max(w_ij, 2⁻¹⁰·R_i/W_m) ;  recompute R_i
           R_i ← max(R_i, 2⁻¹⁰·mean_i(R))                    (marginal floor)
```
- B is the exact integral of the bilinear reconstruction over a texel footprint. The floors make
  every cell's realized probability positive, so NEE alone covers the sphere outside the pole cap.

**Two-level integer alias** (Vose; one table per level; n = 2^m entries; the row table has
n = H_m, and each row's column table has n = W_m).
```
p_k    = w_k·n / Σw                                              (f64)
small bucket s:  q_s = clamp(round(p_s·65536), 0, 65535),  alias_s = l,
                 and the large bucket l is debited by exactly (65536 − q_s)/65536
leftover buckets: alias = self, q = 65535
entry  = (alias << 16) | q                                       (u32)
realized:  P(k) = [ Σ_b ( [b == k]·q_b + [alias_b == k]·(65536 − q_b) ) ] / (65536·n)     (exact rational, f64)
pdfUV[i·W_m + j] = P_row(i) · P_col(j | i) · W_m · H_m           (stored as f32)
```
**pdfUV always comes from the stored integers, never from the target weights** (plan §2 rule 15).

**Sampling** (h0, h1, h2 = the vertex's NEE u32 hash dims; [rng-layout](#rng-layout)).
```
i0 = h0 >> (32 − log2 H_m);  e = rowAlias[i0];         i = ((h0 & 0xFFFF) < (e & 0xFFFF)) ? i0 : (e >> 16)
j0 = h1 >> (32 − log2 W_m);  e = colAlias[i·W_m + j0];  j = ((h1 & 0xFFFF) < (e & 0xFFFF)) ? j0 : (e >> 16)
du16 = h2 >> 16;  dv16 = h2 & 0xFFFF
u = (j + (du16 + 0.5)/65536) / W_m ;   v = (i + (dv16 + 0.5)/65536) / H_m
CELL CLAMP (env-verify 1): u = min(u, prevFloat((j+1)/W_m)),  v = min(v, prevFloat((i+1)/H_m))
                           prevFloat(x) = bitcast<f32>(bitcast<u32>(x) − 1u)
```
- The bucket (top bits) and the threshold (low 16 bits) of one hash are independent, because
  log2 n ≤ 16. W_m ≥ 4 avoids a shift by 32, which WGSL takes mod 32.
- The in-cell lattice has spacing 2⁻¹⁶ of a cell. With the clamp,
  `p1Env(uv(i,j,du,dv)) ≡ pmf[ENV]·pdfUV[i,j]/(2π² s)`, always from the sampled cell.

**Solid-angle pdf (given "env chosen") and pole cap.**
```
pdf_σ(uv; s) = pdfUV[i(uv), j(uv)] / (2π² · s),     i = min(u32(v·H_m), H_m−1),  j = min(u32(u·W_m), W_m−1)
s = ‖b.xy‖ = sqrt(b.x² + b.y²)          when a DIRECTION is available (BSDF escapes, camera misses, refresh of BSDF ends)
s = sin(π·min(v, 1 − v))                 for NEE lattice samples (uv from the sampler)
POLE CAP (both techniques):  s < 1e-6  ⇒  pdf_σ := 0
```
- An NEE sample with s < 1e-6 is rejected: F = 0, same test. The cap region (about 6·10⁻¹² sr)
  is covered by BSDF sampling only, where ω2 = 1. The partition stays exact (env-verify 2).
- `dω = 2π² sinθ du dv`, and pdf_σ integrates to 1 ± 1e-5 (ENV-U5).
- Residual inexactness: the `envUV(envDir(uv))` round trip at cell edges and near the poles gives
  a partition error on a set of measure ≈ 10⁻⁷. It is accepted and bounded by ENV-U6.

**One function each** (plan rule 15): `envUV`, `envDir`, `envRadiance`, `envSample`, `envPdfSA`,
`p1Env(uv) = pmf[ENV]·pdf_σ(uv)`, and `visibleInf(a, ω)`. `p1Env` is used by NEE, BSDF-escape MIS,
RIS-NEE, the shifts and refresh.

**Rebuild policy.** The tables are built in a Worker and rebuilt only when the map or the
importance resolution changes. Either change is a config change that resets history
([light-changes](#light-changes)).

---

<a id="light-selection"></a>
## 7. Light selection (power alias, realized pmf, P(env) clamp)

Sources: plan §1.4 and §1.4b, gap-light §3.9, env §2.5, io §7.6, env-verify omission 2.

**Entries.** Every analytic light, every emissive triangle with `emission_sampling ≠ NONE`, and
`ENV_ID` (if an env exists) are entries of **one** global alias table.
- They are added in stable-id order.
- Triangles with `emission_sampling = NONE` are excluded: they are BSDF-only, with forward MIS
  weight 1 (glass §7.1).

**Power proxies Φ̃** (io §7.6, env §2.5; variance-only choices, except that the SAME pmf must be
used for sampling and for every pdf evaluation).
```
point:              4π·lum(I)                                        (= lum Φ)
spot:               lum(I)·2π·[ (1 − c_full) + (c_full − c_outer)/2 ],  c_outer = cos(spot_size/2),
                    c_full = c_outer + (1 − c_outer)·spot_blend
rect/disk:          lum(L_e)·π·A · (spread factor mean if spread < π)
emissive triangle:  lum(avg L_e)·A·π·2                               (two-sided)
sun:                lum(E)·π·R_s²
env:                Φ_env = 4π²·R_s²·L̄_env,
                    L̄_env = strength·(1/4π)·Σ_{r,c} lum(tint ⊙ texel_rc)·ΔΩ_r,
                    ΔΩ_r = (2π/W)·( cos(π r/H) − cos(π (r+1)/H) )     (rows bottom-up)
R_s = bounding-sphere radius of the static triangles (fixed per scene load)
```

**Normalisation and the P(env) clamp** (f64, stable-id order).
```
if env is the only emitter:   P(env) = 1
else if env exists:           P(env) = clamp( Φ_env / ΣΦ̃ , 0.1, 0.9 );
                              P(L)   = (1 − P(env)) · Φ̃_L / Σ_{L' ≠ env} Φ̃_{L'}
else:                         P(L)   = Φ̃_L / ΣΦ̃
```

**Alias table and realized pmf.**
- The global table stores **u16 integer thresholds**, like the env tables. The pmf used anywhere
  (NEE q, p1, ω1, ω2, RIS ratios, J_P, refresh) is the **realized** probability computed from the
  stored integers with the [env-sampling](#env-sampling) formula. It is never the target
  `P(L)` (plan §1.4).
- Bucket selection must be exactly uniform. See open item 5 for the proposed power-of-two padding.

**Determinism and invariants.**
- The table and pmf are rebuilt **only when lights change** (power, colour, exposure, size, spot
  angles, add, remove, env strength or tint, env-NEE toggle). Rebuilds are deterministic: stable-id
  order and identical float operations. So `pmf_t ≡ pmf_{t−1}` bitwise when nothing changed, and
  J_P = 1 exactly.
- Rigid motion changes neither Φ̃ nor A, so it never rebuilds. **Env rotation never changes the
  pmf.**
- The pmf never depends on the shading point, the pixel or the path (no light tree, no
  tile-conditional pmf).

**Light tiles** (RIS-NEE, M6; gap-light §3.9).
- 128 tiles × 1024 entries are filled with i.i.d. draws from **this frame's** alias table. Each
  8×8 screen tile picks one light tile.
- Only the **marginal** `pmf[L]·p_A(z|L)` may appear in q, p1, ω or J, never the tile-conditional
  multiplicity.

**Per-light position sampling** (area-uniform in light-local (u, v); plan §1.4). Samplers that
depend on the shading point, such as the spherical rectangle, are **forbidden**, because they
break J = 1.
```
rect:      z = c + (u − ½)·len_u·a_u + (v − ½)·len_v·a_v                  p_A = 1/A
disk:      (δx, δy) = concentric(u, v)  (Shirley–Chiu; gap-bsdf §11 sample_uniform_disk)
           z = c + ½len_u·δx·a_u + ½len_v·δy·a_v                          p_A = 1/A
triangle:  bary = (1 − √u₁,  √u₁(1 − u₂),  √u₁·u₂)  over (v0, v1, v2)       p_A = 1/A_tri
point/spot: z = c (no positional dims consumed for position, but the dims stay reserved)
sun:        direction s_L
env:        (i, j, du16, dv16) per [env-sampling]
```
- Quantised (u, v) or barycentrics (production layout) are quantised **before** F is computed
  (plan rule 14).
- **Same-triangle skip** (Cycles `shade_surface.h:345-351`): if an emissive-triangle NEE sample
  lies on the shading triangle itself, skip it when the sample direction is in the Ng hemisphere
  (gap-bsdf §8.1 item 4).

---

<a id="measure"></a>
## 8. NEE measure μ, source density q, light term Λ, RIS-NEE

Sources: plan §2 rule 2, gap-light §3.1 and §3.8, env §3.1, review R7.

**Product measure.**
```
μ = Σ_{L ∈ area ∪ tri} δ_L ⊗ A_L   +   Σ_{L ∈ delta ∪ sun} δ_(L, c_L)   +   δ_ENV ⊗ σ_env
```
- A_L is the area measure on the light surface.
- Delta lights and the sun are unit atoms.
- σ_env is solid angle on S², parameterised env-locally by (u, v).

At a shading vertex x = x_{d−1}, with outgoing ω_o (toward the previous vertex), light sample
Y = (L, z), `r = ‖z − x‖`, `ω_L = (z − x)/r`, and `cosθ_z = n_z·(x − z)/r`:
```
direct integrand   g(L, z) = f_cos(ω_o, ω_L) · V(x, z) · Λ_L(z; x)          (∫ g dμ = Cycles' direct light at x)

Λ_L(z; x) = L_e(z→x)·|cosθ_z| / r²        area light (one-sided, spread), emissive triangle (two-sided)
          = I_L(ω_{c→x}) / r²              point, spot
          = E_L                            sun   (r and cos irrelevant; z is the direction s_L)
          = L_env(uv)                      env   (no r², no cos)

q(L, z)   = P(L) / A_L                     area light, emissive triangle (area-uniform)
          = P(L)                           point, spot, sun
          = pmf[ENV] · pdf_σ(uv)           env   (already solid angle)
P(L) = realized pmf ([light-selection])
```

**Single-sample NEE PSS integrand.**
```
F_NEE = T ⊙ ω1 · f_cos(ω_o, ω_L) ⊙ Λ_L(z; x) · V(x, z) / q(L, z)
```
T is the prefix throughput, with RR excluded ([rr](#rr)).

**Solid-angle light pdf p1** (used **only** inside ω1 and ω2, [mis](#mis)).
```
p1 = q · r² / |cosθ_z|      area light, emissive triangle      (0 if cosθ_z = 0)
p1 = q                      env
p1 : undefined for delta lights and the sun — they use the isDelta flag, never p1
```

**RIS-NEE** (Enhanced S-§5, run in μ).
```
candidates  X_i = (L_i, z_i) ~ q,  i = 1..M = M(B)       (i.i.d. marginally; tile draws may correlate, still unbiased)
target      p̂_NEE(X) = lum( f_cos(ω_o, ω_L) ⊙ Λ_L(z; x) )     (visibility excluded)
ratio       r_i = p̂_NEE(X_i) / q(X_i)                      BOTH in μ (area: Λ with P/A; delta: I/r² with P; env: L_env with pmf·pdf_σ)
selection   Y ∝ r_i  (resampling RNG stream, never the path stream)
W^RIS       = (1/p̂_NEE(Y)) · (1/M) Σ_i r_i                 (UCW in μ)
W_NEE       = W^RIS · q(Y) = (1/M)·Σ_i r_i / r_Y             ("W^RIS·p1" of Enhanced: its p1 is q in μ, NOT the solid-angle p1)
```
- The NEE candidate enters the path-tree stream with `F_NEE` from above and source weight
  `W_NEE / ∏_{RR tests survived} q_RR` ([path-tree](#path-tree), [rr](#rr)).
- Plain NEE is M = 1, which gives W_NEE = 1.
- **Trap (bias):** forming r_i with p̂ in one measure and q in another. Examples are
  `lum(f·L_e)` over `P/A`, or `lum(f·I)` without `1/r²`. Test U3 plants exactly this.
  - **[M6 addition, restir-m6-api.md §5.3]** Precisely: a *consistent* target in any measure is unbiased — RIS only needs
    the same ratio r = p̂/q in the selection and in the UCW (p̂ > 0 wherever the integrand is). The bias comes from a UCW
    whose measure differs from q's: `W_NEE = W^RIS·p1_σ` instead of `W^RIS·q` multiplies the estimator by p1_σ/q =
    r²/|cos θ_z| for area / triangle picks (U8-8 plant; tests/restir/plant-m6.test.ts shows both statements in f64).
- **[M6 addition, restir-m6-api.md MD4]** Realisation in ReSTIR: RIS-NEE runs only at x₁ (B = 1) with M = 32. A per-frame
  pass draws `128 tiles × 1024` i.i.d. alias entries (env entry included) per ensemble member from hashes of
  `(runSeed ⊕ member·φ, t, tile·1024 + slot, STREAM_LIGHT_TILE)`; each 8×8 member-local screen tile picks one tile. A
  candidate j uses slot `h.x & 1023` of `h = pcg4d(seed, j, STREAM_RIS_NEE)` and the per-entry light-local words of
  `nee_draw` from `h.yzw`; same-triangle samples have p̂ = 0. `q` in r and in W_NEE is the **marginal** q (pmf[L]·…),
  never the tile frequency (plan rule 2; the U8-10 plant puts the tile multiplicity/1024 into ω1 instead). The selected
  endpoint then runs the unchanged NEE code (F, k*, visibility).
- **Never** replay RIS-NEE over per-frame tiles in a shift or refresh. It is not a shift and is
  +64% biased (plan rule 10; gap-temporal §5.2).

---

<a id="mis"></a>
## 9. NEE/BSDF MIS (ω1, ω2), M(B), delta and Mode rules

Sources: plan §2 rules 1–2 and §1.4 Modes, gap-light §3.7, env §0.6, glass §5.2, gris-verify C1,
review R7.

```
ω1 = M(B)·p1 / (M(B)·p1 + p2)          NEE-sampled endpoint
ω2 = p2 / (M(B)·p1 + p2)               BSDF-sampled endpoint (emissive triangle, Mode-B analytic area light, env escape)
B  = vertex index of x_{d−1}
p2 = p̄_{d−1}(ω) = marginal BSDF pdf at x_{d−1} of the endpoint direction, over NON-delta lobes,
     VALID-ONLY ([pdf-conventions]); delta lobes' sample weights stay in the lobe-pmf normaliser
p1 = [measure] solid-angle light pdf seen from x_{d−1}, including the realized selection pmf
```

**M(B)** is **one** deterministic function, shared by NEE-time, BSDF-hit-time, replay, shift and
refresh code (gap-light N11):
```
M(B) = 32   if B = 1 and RIS-NEE is enabled (light tiles at x₁, M6)
     = 1    otherwise
if RIS-NEE is ever extended to deeper bounces: M(B) = max(1, ⌊32/B²⌋)      (Enhanced rounding unverified)
```
M(B) is part of the frame config: a change resets history.

**Hard rules** (decided by flags, **never** by `pdf == 0` arithmetic).
1. **ω1 ≡ 1** for delta lights (point, spot, sun). BSDF sampling can never hit them.
2. **ω1 ≡ 1** for analytic lights in **Mode A** (NEE-only) and **Mode A′**.
   - The A′ case is derived here, not stated in the plan. In A′ only rays leaving a singular vertex
     can hit analytic lights, and singular vertices do no NEE. NEE and BSDF hits therefore never
     overlap on the same path, so each technique has weight 1.
3. **ω2 = 1 after a delta lobe.** This is Cycles' `PATH_RAY_MIS_SKIP`. A singular vertex does no
   NEE.
4. **Camera rays:** weight 1 for everything the camera ray sees (length-1 terms).
5. **Env: always MIS**, in modes A, A′ and B, because escaping BSDF rays always see it.
   `p1 = p1Env(uv)`.
6. **Emissive triangles:** always two-technique MIS, in every mode. `emission_sampling = NONE`
   triangles are BSDF-only with ω2 := 1.
7. **Spurious rough-refraction directions** (valid-only `p2 = 0`) give ω1 = 1 exactly
   ([glass](#glass)).

**Modes** (plan §1.4):

| Mode | Analytic area lights | ω1 (analytic) | ω2 (analytic) | Blender side |
|---|---|---|---|---|
| A (validation start) | NEE only, never hit by BSDF rays | 1 | – | per-light MIS = False |
| A′ | hittable only by rays leaving a **singular** vertex | 1 | 1 (after a delta lobe) | – (same expectation as B) |
| B (product default after Gate 3.11) | hit by BSDF rays, pass-through, one candidate per crossing | M p1/(M p1 + p2) | p2/(M p1 + p2) | MIS True, `transparent_max_bounces = 1024` |

- **Partition of unity.** For every continuous light point reachable by both techniques,
  ω1 + ω2 = 1. Any heuristic with that property changes only variance. Cycles' power heuristic
  gives the same expectation (gap-light §3.7).
- **We use the balance form with M(B) everywhere.** The power heuristic is only a diagnostic
  negative control.
- **A ≡ B holds only** when no light-ending path has a singular lobe at x_{d−1} (glass §4.3). The
  mirror/glass positive control must show A < B. The U9 check "A ≡ B" runs only on scenes without
  singular lobes.
- **Pass-throughs.** An emitter hit behind a Mode-B area-light crossing uses the unchanged p2 of
  x_{d−1}, measured from the unchanged ray origin. Crossings change neither the MIS origin nor the
  pdf (gap-light §1.3).
- **In shifts**, p1 and p2 are recomputed at the **offset path's own** x_{d−1} (y_{d−1} in cases
  (a), (d), (e), (f)). p2 changes in cases (b) and (c) because V at x_{d−1} changes
  ([jacobian](#jacobian)).
- **[M6 addition, restir-m6-api.md MD5]** In ReSTIR, M(B) = risM (32) at B = 1 with RIS-NEE and 1 otherwise. It is
  realised as `p2/M` (ω1 = p1/(p1 + p2/M) = M p1/(M p1 + p2), exact for M a power of two) in every MIS evaluation whose
  x_{d−1} can be x₁ (path tree NEE / BSDF ends at B = 1, shift cases (a)/(f), (d)/(d-ana), (e) at d = 2, replay ∅ at
  b = 1); the PT's `mis_M` is not touched (the PT has no RIS-NEE).
- **[M6 addition, restir-m6-api.md MD6]** **Mode-B crossings in ReSTIR.** After the BSDF continuation at x_B, every
  rect / disk light crossed front-facing by the continuation ray before its hit (or on an escape) is a candidate
  `(d = B + 1, BSDF_ANALYTIC, entry)` with its own direction ω_c to the stored crossing point, its own BSDF query and
  `ω2 = p2(ω_c)/M(B) / (p1 + p2(ω_c)/M(B))` (ω2 = 1 after a delta lobe; Mode A′: only after delta lobes). Crossings
  consume no bounce, no RNG dimension, no RR test.

---

<a id="bsdf-v1"></a>
## 10. BSDF V1 (validation): Lambert + GGX glossy (F ≡ 1) + constant Mix + Emission

Sources: plan §1.5, gap-bsdf §3 and §5.

**Blender recipe.** Diffuse BSDF (Roughness 0), Glossy BSDF `distribution='GGX'` (default is
MULTI_GGX), Mix Shader Fac f, Add Shader + Emission. Also `use_bump_map_correction = False` and
`emission_sampling = 'FRONT_BACK'`.

**Closures.**
```
f    = saturate(Fac)
w_D  = max((1 − f)·ρ, 0)            allocated iff |avg(w_D)| ≥ 1e-5
w_G  = max(f·k, 0)                  allocated iff |avg(w_G)| ≥ 1e-5,   α = saturate(r)²   (r = perceptual Roughness)
L_e  = Strength·Color               two-sided, no π, not a lobe
```

**GGX building blocks** (isotropic; `K/closure/bsdf_microfacet.h`; gap-bsdf §3.2).
```
a2        = α_x·α_y = α² = r⁴
D(c)      = a2 / ( π·((1 − c²) + a2·c²)² ),           c² = min(cosNH², 1)
Λ(c)      = ½·( sqrt(1 + a2·max(1/c² − 1, 0)) − 1 ),  c := max(|c|, 1e-7)   (|·| fix: glass U-G10)
G1(V)     = 1/(1 + Λ(N·V))
G2        = 1/(1 + Λ(N·V) + Λ(N·L))                   (height-correlated Smith)
H         = normalize(V + L)
eval_S    = F(V·H) · D(N·H) · G2 / (4·N·V)            (= f·cos; F ≡ 1 in V1 times w_G)
p_VNDF(L) = D(N·H) · G1(V) / (4·N·V) = D / (4·N·V·(1 + Λ(N·V)))
valid     : N·V > 0,  a2 > 2e-10,  N·L ≥ 0   (eval tests N only, never Ng)
```
- **VNDF sampler** (Heitz 2018, Cycles `:189-220`):
  1. `V_s = normalize(α V.x, α V.y, V.z)`.
  2. Build T1 and T2. If `lensq > 1e-7`, `T1 = (−V_s.y, V_s.x, 0)/√lensq` and `T2 = V_s × T1`;
     otherwise T1 = x̂, T2 = ŷ.
  3. `t = concentric(u)`, then `t.y = mix(√(1 − t.x²), t.y, ½(1 + V_s.z))`.
  4. `H_s = t.x T1 + t.y T2 + √(1 − |t|²) V_s`.
  5. `H = normalize(α H_s.x, α H_s.y, max(0, H_s.z))`.
  6. `L = 2(V·H)H − V`.
  7. **Reject** if `Ng·L < 0` or `N·L < 0`.
- **Lambert.** `eval_D = max(N·L, 0)/π` and `p_D = max(N·L, 0)/π`. Cosine-hemisphere sampling
  about N uses the concentric map. **Reject unless Ng·L > 0 (strict).**

**Lobe pmf** (Cycles plain sample weights; V1 GGX has no albedo scaling; V-independent).
```
q(D) = |avg(w_D)| / (|avg(w_D)| + |avg(w_G)|),   q(S) = 1 − q(D)
```

**pdfs.**
```
p(L, D | V) = q(D)·max(N·L,0)/π
p(L, S | V) = q(S)·p_VNDF(L|V)
p̄(L | V)   = p(L,D|V) + p(L,S|V)
```

**Delta.** `isDelta ⇔ !(α_x·α_y > 2e-10)`, i.e. r ≤ 0.0037606.
- Sampling gives `L = 2(N·V)N − V`, rejected if `N·V ≤ 0`, `Ng·L < 0` or `N·L < 0`.
- Throughput factor: `w_G/q(S)`.
- It contributes 0 to NEE eval and 0 to p̄, but its q stays in the normaliser.

**Reconnection properties.**
- Perceptual roughness: `r(D) = 1`; `r(S) = r`, or 0 if delta.
- `diffuseOnly ⇔ hasD ∧ ¬hasS`.
- V1 is reciprocal: `f(V,L)/(N·L) = f(L,V)/(N·V)`.

---

<a id="bsdf-v2"></a>
## 11. BSDF V2: Cycles Principled Tier 1 (`distribution='GGX'`)

Sources: plan §1.5, gap-bsdf §3.4–3.5 and §6, cyc §4.4–4.5, cyc-verify C2/C4 and omission 1.

**Input clamps.**
```
C  = max(Base, 0)        (unclamped above)       Cc = min(C, 1)
r  = saturate(Roughness), α = r²                 m  = saturate(Metallic)
η  = max(IOR, 1e-5)      L_s = max(Specular IOR Level, 0)     T_s = max(Specular Tint, 0)
t  = saturate(Transmission Weight)   (glass, [glass]; 0 in the no-glass subset)
weight = mix_weight · alpha = 1      (validation: alpha = 1, coat/sheen/SSS/thin film/aniso/diffuse roughness = 0)
```

**Closure order** (`K/svm/closure.h:169-509`). Each closure is allocated only if
`|avg(closure weight)| ≥ 1e-5`, and this cutoff is part of the BSDF definition.
1. **Metal** (if m > 1e-5).
   - `w_M = m·weight`, F82-tint with `F0_M = Cc`, `f82 = min(T_s, 1)`, `B_M = B(Cc, f82)`.
   - Then `weight *= (1 − m)`, even if the metal closure was not allocated.
2. **Glass** (if t > 1e-5): `w_G = t·weight` ([glass](#glass)). Then `weight *= (1 − t)`. There is
   no albedo layering after glass.
3. **IOR level.** `f0 = F0(η)`. If `L_s ≠ 0.5`: `f0 *= 2L_s`, `η' = ior_from_F0(f0)`, and
   `η' = 1/η'` if η < 1. Otherwise `η' = η`.
4. **Dielectric specular** (if η' ≠ 1).
   - `w_S = weight`, generalized Schlick with `f0_S = saturate(f0·T_s)`, `f90 = 1`,
     `exponent = −η'`.
   - Then **layering:**
   ```
   E_S(μ) = mix(f0_S, 1, S_ior(r, μ, z_S))        per channel,  z_S = sqrt(|(η'−1)/(η'+1)|),  μ = dot(V, Ns)
   Λ_S(μ) = saturate( 1 − max_c E_S,c(μ) )          (safe_divide: 0 where weight_c = 0)
   weight ← weight · Λ_S(μ)
   ```
5. **Diffuse (Lambert):** `w_D = C·weight = C·(1−t)(1−m)·Λ_S(μ)`. It uses the **unclamped** C and
   normal N.
6. **Emission:** `L_e = EmissionColor·EmissionStrength·(weight at emission = alpha = 1)`,
   two-sided.

**Fresnel models** (`bsdf_util.h`).
```
F0(η)          = ((η−1)/(η+1))²
ior_from_F0(f) = (1 + √f)/(1 − √f),  f clamped to [0, 0.99]
F_diel(c, η)   : g = η² − (1 − c²); g ≤ 0 → 1 (TIR); ci = |c|; ct = −√g/η;
                 r_s = (ci + η ct)/(ci − η ct);  r_p = (ct + η ci)/(η ci − ct);  F = (r_s² + r_p²)/2
F_gs(c; η, f0) = mix(f0, 1, saturate((F_diel(c,η) − F0(η))/(1 − F0(η))))        (exponent < 0 mode)
                 = F_diel(c,η) exactly for 1 < η ≤ 2+√3 with f0 = F0(η) (cyc-verify C4); port the saturate literally
F82(c; F0, B)  = saturate( F0 + (1−F0)·s⁵ − B·c·s⁶ ),  s = saturate(1 − c)
B(F0, tint)    = 0 if tint ≡ (1,1,1); else (F0 + (1−F0)·f⁵)·(7/f⁶)·(1 − tint),  f = 6/7
```

**Lobe class S** is metal ⊕ dielectric specular. They merge exactly: same Ns, same α, same VNDF
(gap-bsdf §6.4).
```
f_S·cos(V,L) = [ w_M·F82(V·H; Cc, B_M) + w_S·F_gs(V·H; η', f0_S) ] · D(N·H)·G2 / (4·N·V)
p_S(L|V)     = p_VNDF(L|V)
f_D·cos(V,L) = w_D(V)·max(N·L,0)/π,        p_D = max(N·L,0)/π
```

**Albedo-scaled sample weights** (Cycles-exact `sample_weight`, noise-only but normative for
Cycles-identical MIS; gap-bsdf §6.6).
```
sw_M(V) = m · avg( mix(Cc, 1, S5(r, μ_V)) )            S5 = lut3(r, μ, 0.5, table_ggx_gen_schlick_s)   (B ignored, Cycles TODO)
sw_S(V) = (1−t)(1−m) · avg( E_S(μ_V) )
sw_G(V) = [glass]
sw_D(V) = |avg(C)| · (1−t)(1−m) · Λ_S(μ_V)
guard:  an allocated lobe with sw = 0 gets sw = 1e-12 (bias guard; measure-zero in Cycles)
q(D|V) = sw_D/Σ,  q(S|V) = (sw_M + sw_S)/Σ,  q(G|V) = sw_G/Σ,   Σ = sw_D + sw_M + sw_S + sw_G
```
- p(L, ℓ | V) = q(ℓ|V)·p_ℓ(L|V), and p̄ = Σ over non-delta, valid lobes.
- **q depends on V**, so the joint pdf of D changes under reconnection, and V2 is
  **non-reciprocal**. Always evaluate with V toward the camera-side predecessor. At a reconnection
  vertex, `V = normalize(y_{k−1} − x_k)`.

**LUT reader** (exact port of `util/lookup_table.h`, from a read-only **storage buffer** of f32,
never from a filtered texture; gap-bsdf §3.5).
```
lut1(x, off, n):   x' = saturate(x)·(n−1); i = min(trunc(x'), n−1); j = min(i+1, n−1); t = x' − i
                   if t == 0 return d[off+i];  return (1−t)·d[off+i] + t·d[off+j]
lut2(x, y, off, nx, ny):  nests lut1 over rows y (layout d[off + y·nx + x])
lut3(x, y, z, off, n):    nests lut2 over slices z (layout d[off + z·n² + y·n + x]), axes z (outer) → y → x (inner)
x = rough = sqrt(sqrt(α_x α_y)) (= r),   y = μ = dot(V, Ns),   z per table
```

| Table (v5.1.2 `shader.tables`) | Floats | Σ over table | Offset | Use |
|---|---|---|---|---|
| `table_ggx_gen_schlick_ior_s` | 4096 (16³) | 184.247126 | 0 | V2 layering and sample weight, z = sqrt\|(η−1)/(η+1)\| |
| `table_ggx_gen_schlick_s` | 4096 | 764.768581 | 4096 | metal sample weight (z = 0.5) |
| `table_ggx_E` | 1024 (32²) | 849.736018 | 8192 | Tier 2 (stretch) |
| `table_ggx_Eavg` | 32 | 25.879163 | 9216 | Tier 2 |
| `table_ggx_glass_{E,Eavg,inv_E,inv_Eavg}` | 4096/256/4096/256 | 3758.664856 / 235.910746 / 3477.751955 / 216.090299 | 9248 / 13344 / 13600 / 17696 | Tier-2 glass (stretch) |

**Test vectors** (gap-bsdf §13):
- `S_ior(0.5, 0.4, IOR 1.5) = 0.0639`
- `S5(0.5, 0.1) = 0.1718`
- `ggx_E(0.5, 1) = 0.91528`
- `Eavg(1) = 0.40914`
- q(S) for C = 0.8, m = 0, r = 0.5, IOR 1.5: 0.0502 at μ = 1, 0.2422 at μ = 0.1.

**Delta** (as V1).
- S-delta throughput is `(w_M·F82(N·V) + w_S·F_gs(N·V))/q(S)`.
- Cycles singular eval is `F·1e6` and pdf is `lobe_prob·1e6`, so the sample weight is `F/lobe_prob`
  (cyc-verify C2).

<a id="lobe-codes"></a>
**Lobe codes** (plan §1.5; 3 bits + 1 delta bit per stored vertex; see open item 2).
```
D = 0   Lambert
S = 1   GGX reflection class (metal ⊕ dielectric specular; V1 glossy)
G_R = 2 glass closure, reflection sub-event (N·L ≥ 0 side)
G_T = 3 glass closure, transmission sub-event (N·L < 0 side)
NEE = 4 outgoing direction NEE-sampled (all lobes evaluated)
NONE = 5 light / env vertex (no outgoing direction)
6, 7 reserved
delta bit: set iff the event is singular ([bsdf-v1] / [glass]); never inferred from pdf == 0
```
- Per-lobe perceptual roughness for R_k:
  - `r(D) = 1`;
  - `r(S) = r(G_R) = r(G_T) = r`, or 0 if that event is delta;
  - `r(NEE) = max over allocated non-delta lobes`, unused by the predicate;
  - `r(NONE) = +∞`.
- `diffuseOnly(x) ⇔ hasD ∧ ¬hasS ∧ ¬hasG`. It is never true with glass.

---

<a id="glass"></a>
## 12. Glass (lobe class G: Walter/Cycles GGX glass closure)

Sources: plan §1.5 and §2 rule 12, glass §2 and §5, review R9 and R10.

**Sources of G closures.**
- **Principled** (t > 1e-5):
  - `w_G = t(1−m)`, isotropic α = r² (shared Roughness);
  - generalized Schlick with `f0 = saturate(F0(ior)·T_s)`, `f90 = 1`, `exponent = −ior`;
  - `reflection_tint = 1`, **`transmission_tint = √Cc`** (per interface);
  - it uses the node IOR, **not** η'.
- **Glass node:** weight `mix_weight`; `reflection_tint = transmission_tint = max(Color, 0)`;
  `f0 = F0(ior)`.
- **Refraction node:** weight `Color·mix_weight`; Fresnel NONE, i.e. R = 0 and
  `T = (F_diel == 1) ? 0 : 1`. There is no reflection; TIR kills the path.
- Validation forces `distribution = 'GGX'` on Principled, Glass (default MULTI_GGX) and
  Refraction (default BECKMANN).

**Side-dependent η** (stateless, recomputed at every evaluation from the evaluating V; never
stored; no medium stack; no nested-dielectric logic):
```
Ng       = normalize(cross(v1 − v0, v2 − v0)), flipped for negative-determinant transforms (winding identical to Blender)
back     = dot(Ng, V) < 0     ⇒  Ng ← −Ng,  N ← −N
η_side   = back ? 1/ior : ior
```

**Fresnel at the microfacet** (`cHI = H·V`).
```
F    = mix(f0, 1, saturate( (F_diel(cHI, η_side) − F0(η_side)) / (1 − F0(η_side)) ))
R    = F · reflection_tint ,     T = (1 − F) · transmission_tint
P_R  = avg(R) / avg(R + T)        (TIR: F = 1 ⇒ T = 0 ⇒ P_R = 1, always reflect)
Refraction node: R = 0, T = (F_diel == 1) ? 0 : 1, P_R = 0 unless TIR
```
With white T_s, F = F_diel exactly for 1/3.73 < η_side < 3.73 (glass §2.2).

**eval·cos and pdf** (Cycles `bsdf_microfacet.h:586-675`; cNI = N·V, cNO = N·L).
```
return 0 if cNI ≤ 0, or a2 ≤ 2e-10 (singular: no eval)
isT    = cNO < 0
H      = isT ? −(η_side·L + V) : (V + L);   invLen = 1/|H|;   H ← H·invLen      (not re-oriented)
common = D(N·H)/cNI · ( isT ? (η_side·invLen)²·|cHI·(H·L)|  :  1/4 )
         (refraction half-vector Jacobian: (η·invLen)²·|H·L| = η²|H·L|/(V·H + η L·H)²)
pdf_C  = common · (isT ? 1 − P_R : P_R) / (1 + Λ(cNI))                            ("Cycles eval pdf")
f_G·cos = w_G · (isT ? T : R) · common / (1 + Λ(cNI) + Λ(cNO))                      [× Tier-2 multiplier]
```
- **Joint pdfs** (valid-only, [pdf-conventions](#pdf-conventions)).
  ```
  p(L, G_R | V) = q(G|V) · D(H_r)·G1(V)/(4·N·V) · P_R(H_r)                         H_r = normalize(V + L)
  p(L, G_T | V) = q(G|V) · D(H_t)·G1(V)·|V·H_t|/(N·V) · η²|H_t·L|/(V·H_t + η L·H_t)² · (1 − P_R(H_t)) · 1_valid
                  H_t = −normalize(η_side·L + V)
  ```
  G_R and G_T are disjoint by side, so the marginal sums them.
- **Validity (sampler support)** ("spurious region": Cycles' eval lacks this check,
  `bsdf_microfacet.h:621-623` TODO).
  ```
  G_R valid ⇔ Ng·L ≥ 0 (and N·L ≥ 0)
  G_T valid ⇔ Ng·L < 0 ∧ N·L < 0 ∧ Hn·V > 0 ∧ Hn·L < 0 ∧ no TIR at Hn ∧ |η_side − 1| ≥ 1e-4
             Hn = H oriented so that N·Hn ≥ 0
  p_G(L|V) = pdf_C(L|V) · 1[valid]           ("valid-only" = true sampler density)
  ```
- **Sampling** (u = (u_h1, u_h2, u_rt); the class pick used u_lobe).
  ```
  H = singular ? N : VNDF(V, α, (u_h1, u_h2));  (R,T) = fresnel(H·V);  refract = (u_rt ≥ P_R)
  L = refract ? (cHI/η + cT)·H − V/η  :  2cHI·H − V          (cT = signed cos from F_diel, cT < 0)
  reject if (Ng·L < 0) ≠ refract  or  (N·L < 0) ≠ refract
  weight (non-singular) = w_G·(R or T)·common/(1+Λ_I+Λ_O) / (q(G)·pdf) = w_G·(R or T)/(q(G)·(P_R or 1−P_R)) · (1+Λ_I)/(1+Λ_I+Λ_O)
  ```
- **Delta rules** (per sub-event).
  - `G_R` is delta iff `α² ≤ 2e-10`.
  - `G_T` is delta iff `α² ≤ 2e-10 ∨ |η_side − 1| < 1e-4` (`bsdf_microfacet.h:771`).
  - Singular throughput: `w_G·R/(P_R·q(G))` or `w_G·T/((1−P_R)·q(G))`.
  - Singular glass does no NEE, and the next emitter hit has MIS weight 1.
- **No 1/η² radiance scaling anywhere**, and no η² factor in any Jacobian. η enters only through
  the evaluated pdfs (glass §1 item 3). A smooth dielectric has albedo exactly 1 on both sides.
- **Sample weight** (glass §2.6; noise-only).
  ```
  sw_G = |avg(w_G)| · avg(R_est + T_est),   μ = N·V (flipped N),  z = sqrt(|(η_side−1)/(η_side+1)|)
  R_est = mix(f0, 1, lut3(r, μ, z, table_ggx_gen_schlick_ior_s)) ⊙ reflection_tint
  T_est = (1 − F_gs(μ; η_side)) ⊙ transmission_tint              (smooth-surface estimate at N)
  Refraction node: sw = |avg(Color·mix_weight)| (no albedo scaling)
  ```
- **Usage rules.**
  - NEE uses the Cycles eval **including** the spurious part, with no support indicator.
  - Every pdf used for MIS, footprints, Jacobians and χ² tests is the valid-only density.
  - Shifted BSDF-sampled segments carry `1_supp` ([support-indicator](#support-indicator)).
- **Visibility.** Glass occludes every visibility query ([visibility](#visibility)). Transmitted
  rays and shadow rays are offset along `sign(Ng·L)·Ng`.
- **Replay through glass** (R/T flips, TIR, a different class under the offset's own V) is the PSS
  identity with J = 1. TIR never makes a hybrid shift undefined.
- **Reconnection.** v1: **G_T never passes the predicate** (plan rule 5). G_R may pass, with
  r ≥ 0.2.
- `ggx_lambda` must use `max(|cosN|, 1e-7)`, because cNO < 0 for refraction (test U-G10).

---

<a id="pdf-conventions"></a>
## 13. pdf conventions (joint vs marginal; never mix)

Sources: plan §1.5, gap-bsdf §4.4 and §7.1, glass §5.2, enh-verify O2.

| Consumer | Quantity |
|---|---|
| Path throughput at a BSDF-sampled vertex (initial sampling, replay) | `f_ℓ·cos / p(L, ℓ | V)`, **joint**, valid-only by construction |
| NEE contribution at x_{d−1} | `f(V,L)·cos` over **all** lobes (Cycles eval, glass spurious part included, no support indicator) |
| NEE/BSDF MIS (p2 in ω1, ω2) | **marginal** `p̄(L|V) = Σ_{ℓ non-delta} q(ℓ|V)·p_ℓ(L|V)`, valid-only |
| Enhanced footprints F_k, I_k (p̄) | **marginal**, valid-only |
| PSS reconnection Jacobian factors (Eq. 2), jDen | **joint** `p(ω, ℓ)` with the **copied** ℓ, valid-only |
| χ² tests of samplers | valid-only density (Cycles' eval pdf integrates to > 1 in spurious cases) |
| Target p̂ | `lum(F)` with the lobe-extended F |

- `p(ω, ℓ | V) = q(ℓ|V)·p_ℓ(ω|V)`, and it includes the lobe-selection probability.
- Mixing conventions biases the result, e.g. a marginal Jacobian with a joint throughput (U-11
  negative control).
- **Any strictly positive q is unbiased.** Only noise and the MIS weights change (gap-bsdf §4.3).
- A delta lobe has probability mass, not density. `p = 0` in p̄ and in the joint slot, and
  throughput is `F/q`. The flag decides.

---

<a id="support-indicator"></a>
## 14. Sampler-support indicator

Sources: plan §2 rule 7, gap-bsdf §8.3, glass §5.2, review R5.

When a shift evaluates a **BSDF-sampled** segment, multiply both f and the joint pdf by
`1_supp(ℓ, V, L)` from `bsdf_sample_support()`. It is computed after the two-sided N/Ng flip for
the **evaluating** V.
```
D   : Ng·L > 0                                   (and N·L > 0 via eval)
S   : Ns·V > 0 ∧ Ng·L ≥ 0 ∧ Ns·L ≥ 0
G_R : Ns·V > 0 ∧ Ng·L ≥ 0 ∧ Ns·L ≥ 0
G_T : Ns·V > 0 ∧ Ng·L < 0 ∧ Ns·L < 0 ∧ Hn·V > 0 ∧ Hn·L < 0 ∧ no TIR ∧ |η_side−1| ≥ 1e-4
      Hn = −normalize(η_side·L + V), oriented so Ns·Hn > 0
NEE : true (no indicator: Cycles' NEE has no Ng test)
```
It applies to:
- ω' at y_{k−1} with the copied ℓ_{k−1};
- ω_k at x_k under the new `V^y = normalize(y_{k−1} − x_k)`, when ℓ_k is a BSDF lobe;
- the copied direction in cases (c), (d) and (e), including the env direction at y_{d−1}.

With flat shading and no glass the indicator is redundant but harmless. It is what keeps
p̂(ȳ) = 0 outside the canonical sampler's support (GRIS Def. 5.2).

---

<a id="rng-layout"></a>
## 15. RNG layout (counter-based, fixed per-vertex slots)

Sources: plan §1.7, glass §5.6, env §4 "RNG layout", val T1, gap-rc §4. The concrete slot numbers
are pinned here; see open items 4 and 5.

```
u32 h(seed, i) = hash(seed, i)                     (PCG-style integer hash; bitwise identical in every lane)
f32 u01(h)     = f32(h >> 8) · 2⁻²⁴                ∈ [0, 1)
path stream:   slot value = pcg4d(initSeed.x, initSeed.y, vertex·D + slot, PATH).x,   D = 16,  vertex = index b of x_b (b ≥ 1)
initSeed       = pcg3d(runSeed ⊕ member·φ, frame t, pixel p).xy    64 bits, unique per pixel per frame (dupmap identity)
jitter seed    = initSeed.x                                           (the jitter stream hashes 32 bits, never a path input)
```
- **initSeed must be ≥ 64 bits** (M3c finding). With a 32-bit initSeed every path of every pixel, sample and frame is
  one of only 2³² dimension vectors, so the estimator's expectation is a fixed 2³²-point quadrature rather than the
  integral. Its error is ≈ σ_rel/√2³² for a single-sample relative standard deviation σ_rel. That is invisible for
  smooth integrands but reached 0.05–0.13% of the pixel value for BSDF-only sampling of a 1.06·10⁻⁴ sr texel of 10⁴
  (σ_rel ≈ 36; C0s NONE, env §5.2). The error was reproducible across run seeds and changed sign with the 32-bit hash
  used (a 96-bit-input hash of the same draws was unbiased). T1 checks that colliding 32-bit words still give different
  path dims.

| Slot | Name | Used by | Replayed? |
|---|---|---|---|
| 0 | `u_lobe` | lobe-class pick at x_b | **yes** |
| 1 | `u_h1` | VNDF / cosine sample, dim 1 | **yes** |
| 2 | `u_h2` | VNDF / cosine sample, dim 2 | **yes** |
| 3 | `u_rt` | glass R/T decision (consumed at every vertex, even non-glass) | **yes** |
| 4 | `u_sel` | NEE light-alias bucket (top bits) + threshold (low 16 bits) | never (NEE is never replayed) |
| 5 | `h_l0` | light (u,v).u / triangle u₁ / env row hash h0 | never |
| 6 | `h_l1` | light (u,v).v / triangle u₂ / env column hash h1 | never |
| 7 | `h_l2` | env in-cell offsets h2 = (du16 << 16) \| dv16 | never |
| 8 | `u_sel2` | reserved: second alias hash if the padded light table has > 2¹⁶ entries (open item 5) | never |
| 9 | `u_rr` | Russian roulette at x_b (initial sampling only) | never |
| 10–13 | `u_cross[0..3]` | reserved: alpha-BLEND crossings (non-goal in v1; MASK consumes **no** dims; Mode-B crossings consume none) | – |
| 14 | `u_tau` | optional τ jitter of the rc threshold (off by default; gap-rc §6.4). It must come from the path seed | yes (deterministic in the path) |
| 15 | – | reserved | – |

- All four BSDF dims are consumed at **every** bounce, including delta, TIR and non-glass
  vertices. Replay therefore never shifts the stream (glass §5.6).
- The lobe pick is `u_lobe` against the cumulative q(D), q(S), q(G). The direction uses
  (u_h1, u_h2). The R/T decision uses `u_rt ≥ P_R`.
- **Separate streams** (never mixed with the path stream):
  - **Jitter:** `h(runSeed, t, p, JITTER)` ([raster](#raster)).
  - **Resampling:** `h(runSeed, t, p, passId, RESAMPLE, counter)`. It drives streaming-RIS
    selections, RIS-NEE selection, temporal s and spatial selection.
  - **Light tiles:** `h(runSeed, t, tileId, TILES, j)`.
  - **Temporal q′ choice:** sample-independent, from `h(runSeed, t, p, TEMPORAL_PICK)`.
  - **Pairing transforms:** per frame and round, `h(runSeed, t, round, member, PAIRING)`.
- **Ensemble mode:** the member id is part of every seed.

**[M4 addition, restir-api.md]** Concrete formulas of the separate streams (docs/decisions/restir-api.md §5):
```
key          = pcg3d(runSeed ⊕ member·φ, t, localIdx).xy        (= initSeed of tree 0; jitter uses key.x)
tree seed    = s = 0: key;  s ≥ 1: pcg4d(key.x, key.y, s, STREAM_TREE).xy          (offline S trees)
resampling   = u01(pcg4d(key.x, key.y ⊕ passId·φ, counter, STREAM_RESAMPLE).x)   key of the RESAMPLING pixel
               counter: initial (tree<<20)|(B<<12)|slotInVertex (0 NEE, 1 BSDF end, 2… Mode-B crossings);
                        spatial 0 = canonical, 1+s = slot s
pairing      = pcg4d(runSeed ⊕ member·φ, t, (round<<8)|slot, STREAM_PAIRING): dihedral code = .x & 7, offset = (.y, .z) mod W_s
```
The counters depend only on (tree, vertex, candidate kind), so splitting trees across dispatches is bitwise neutral.

**[M5 addition, restir-temporal-api.md]** Temporal streams and chain seeds (docs/decisions/restir-temporal-api.md §5):
```
temporal pick   = pcg4d(runSeed ⊕ member·φ, t, localIdx, STREAM_TEMPORAL_PICK = 0x2c1b3c6d)
                  ξ = (u01(.x), u01(.y)) stochastic rounding of the back-projected position; ring rotation = .z & 7
temporal select = u01(pcg4d(key.x, key.y ⊕ RS_PASS_TEMPORAL·φ, counter, STREAM_RESAMPLE)), key of the DESTINATION
                  pixel; counter 0 = canonical, 1 = temporal (streamed in that order, contribution MIS and Talbot alike)
```
- q′ never reads the path streams or reservoir contents, so it is sample-independent in the sense of
  [temporal](#temporal).
- **Validation chains:** the chain id is the member id (`memberBase + atlas member`); `t` is the animation frame counted
  from the chain's reset (t = 0 has no history); every chain of a run shares `runSeed` and the scene script. A chain is
  a deterministic function of (runSeed, chain id, frames 0…t, scene states 0…t). Interactive: t = the app's
  `seedIndex`; freezing the seed resets history every frame, because identical canonical seeds are not independent
  candidates.

---

<a id="path-tree"></a>
## 16. Path tree, candidates, techniques, streaming RIS, length-1 terms

Sources: plan §1.7 and §3 pass 2, gap-light §2.1 and §6, gap-rc §4, env §3.1, gris §5.4,
gris-verify §2.2(g).

**Techniques.** Codes: `NEE = 0`, `BSDF_TRI = 1`, `BSDF_ANALYTIC = 2` (Mode B), `BSDF_ENV = 3`.
Endpoint type: `tri 0, point 1, spot 2, rect 3, disk 4, sun 5, (6 reserved), ENV 7`.

**Candidates** at each scattering vertex x_B (B = 1 … maxBounces+1), all streamed with
**m = 1**, because the domains are disjoint by (d, technique, endpoint label):
1. **NEE** (not at a delta-only vertex). `d = B + 1`, technique NEE, endpoint `(L, z)` from
   RIS-NEE or plain NEE ([measure](#measure)). This includes NEE_ENV.
2. **RR** at x_B (initial sampling only; [rr](#rr)).
3. **BSDF sample** at x_B, with lobe ℓ_B and direction ω_B (replayable dims). Then trace the
   closest **triangle**.
   - **Mode B:** each front-facing analytic area light L crossed at `t_L < t_hit` (or
     `t_L < FLT_MAX` on a miss) is a candidate `(d = B+1, BSDF_ANALYTIC, lightId = L)`. A crossing
     creates no vertex and consumes no RNG. This also happens on rays that escape.
   - A hit on an emissive triangle gives a candidate `(d = B+1, BSDF_TRI, (primId, bary))`.
   - A miss gives a candidate `(d = B+1, BSDF_ENV, world ω_B)`, and the path ends.
   - Otherwise the path continues with `x_{B+1} = hit`.

**Candidate integrands** (T = the product of `f_ℓ cos/p(ω,ℓ)` over x₁ … x_{d−1}'s BSDF-sampled
factors, RR excluded).
```
NEE:            F = T ⊙ ω1·f_cos,all(x_{d−1}; ω_o, ω_L) ⊙ Λ_L / q · V                  ([measure])
BSDF_TRI:       F = T ⊙ L_e(z→x_{d−1}) · ω2                (T includes the last BSDF factor)
BSDF_ANALYTIC:  F = T ⊙ L_e(z→x_{d−1}) · ω2 · 1[front-facing crossing before the first triangle]
BSDF_ENV:       F = T ⊙ L_env(envUV(ω)) · ω2               (ω2 = 1 after a delta lobe)
```

**Streaming RIS** (Chao; the resampling RNG; one tree per pixel):
```
for each candidate i:  w_i = lum(F_i) · W_src,i        W_src = W_NEE/∏q_RR (NEE) or 1/∏q_RR (BSDF endings)
                       Σw += w_i;  select i with probability w_i/Σw
finalize:              W = Σw / lum(F_Y)     (tree M = 1);  c = 1;  W = 0 if Σw = 0 (empty reservoir: zero radiance)
```
- **Offline mode** uses S = 32 i.i.d. trees. This is two-level RIS: stream all candidates with
  weights `(1/S)·w_i`, then `W = Σ_all w / lum(F_Y)`. Σw persists across split dispatches. The
  result is bitwise equal to a single dispatch.
  - **[M4 addition, restir-api.md]** The common factor 1/S is applied once at finalisation, `W = Σ_all w_i / (S·lum(F_Y))`,
    so the persisted Σw is the plain sum of `lum(F_i)·W_src,i`.
- **k\*** is decided **per candidate** and streamed with it (deferred k*, no lobe revocation;
  [rc-predicate](#rc-predicate)).
- **[M4 addition, restir-api.md]** **Same-formula directions.** Every direction between two stored vertices is
  `normalize(pos(b) − pos(a))` with positions rebuilt from ids (`vertex_from_ids`), in the base path, in replay and in
  every shift: incoming directions V, the direction whose p̄ / joint pdf enters the predicate, jDen and jNum, and the
  stored `rcWi` of a continuing x_k. Sampled bits are used only to trace the ray and as the escape direction of a
  `BSDF_ENV` ending. Base-path throughput factors may use the sampler's `weight` (a few-ulp difference in F is not a
  bias: F is a target and a fixed function of the path).
- **[M4 addition, restir-api.md]** **Base NEE endpoint.** The NEE candidate stores its endpoint as the sampler's own light-local
  coordinates (alias entry, `u01(h_l0)`, `u01(h_l1)`; for emissive triangles these are the `(u₁, u₂)` of the
  area-uniform map; env `(i<<16)|j, h2`) and evaluates it with the same per-entry function as the PT, so every later
  evaluation (shift cases (a), (b), (f); refresh) reproduces z, Λ, q and p1 bit-identically.
- **Suffix cache** for the selected candidate (gap-temporal §5.5, plan §1.9):
  - x_{d−1} hit, ω_o at x_{d−1}, and β_s = the post-rc throughput;
  - plus `(lightIdx, uv)` for an NEE end, or `(t_occ, p2)` for a BSDF end;
  - an escape sets a flag with t_occ = FLT_MAX.
  - **[M4 addition, restir-api.md]** Pinned: NEE end `β_s = ∏_{j=k+1}^{d−2} f_j cos/p_j`, direction = ω_o at x_{d−1};
    BSDF end `β_s = ∏_{j=k+1}^{d−1}` (includes x_{d−1}'s factor), direction = ω_{d−1}, `t_occ` = t of the final ray to
    the first triangle, `p2` = marginal pdf of ω_{d−1} at x_{d−1}. The cache is meaningful only for `k ≤ d−1`
    (x_{d−1} on the copied suffix) and is copied verbatim by spatial selection.

**Length-1 terms (outside the reservoir, weight 1).**
```
pixel = Σ_{L analytic, visibleToCamera, crossed front-facing before x₁ (or before FLT_MAX on a miss)} L_e,L(z_L → x₀)
      + L_e(x₁ → x₀)                                   (emissive triangle at x₁)
      + [miss] visibleToCamera · L_env(d_cam)          (also through alpha cutouts and camera-visible analytic lights, all modes)
      + reservoir estimate (d ≥ 2)                      (F·W, or the RGB paired-MIS sum, [paired-mis])
```
- The V-buffer tests triangles only.
- **Background pixels** (no primary hit) have no reservoir, W = 0 and c = 0. They are never an
  accepted spatial partner or a temporal source, and the denoiser passes them through.

---

<a id="reservoir-fields"></a>
## 17. Reservoir fields (validation layout, uncompressed f32)

Sources: plan §1.9, gap-light §4, env §3.2, gap-temporal §5.6, glass §5.5. The binary word order
belongs to `restir/reservoir.wgsl`. This section defines the semantics.

| Field | Type | Meaning |
|---|---|---|
| `W` | f32 | UCW in PSS |
| `F` | f32×3 | PSS integrand of the stored path **in this reservoir's own pixel domain and frame**; p̂ = lum(F) (invariant I1, [temporal](#temporal)) |
| `initSeed` | u32×2 | path-stream seed for prefix replay; also the dupmap identity. **[M4 addition, restir-api.md]** 64 bits (two words), per the M3c finding in [rng-layout](#rng-layout) |
| `endpointId` | u32 | stable lightId, global emissive-triangle id, or `ENV_ID` (the repurposed `rcVertexRandomSeed`). **[M4 addition, restir-api.md]** M4 values: analytic light = its alias entry index in the reservoir's frame (renumbered on temporal selection), emissive triangle = `primId`, env = `0xFFFFFFFE` |
| `flags.d` | 4 bit | light-vertex index, 2 … 15 |
| `flags.k` | 4 bit | rc index 2 … d; **0 = ∅** |
| `flags.technique` | 2 bit | NEE, BSDF_TRI, BSDF_ANALYTIC, BSDF_ENV |
| `flags.endpointType` | 3 bit | tri, point, spot, rect, disk, sun, –, ENV (= 7) |
| `flags.isDelta` | 1 bit | the endpoint light is delta (point, spot, sun) |
| `flags.ℓ_{k−1}`, `flags.δ_{k−1}` | 3 + 1 bit | lobe that sampled ω_{k−1} at x_{k−1}, and its delta bit |
| `flags.ℓ_k`, `flags.δ_k` | 3 + 1 bit | lobe that sampled ω_k at x_k (NEE / NONE for light vertices), and its delta bit |
| `flags.mode` | 2 bit | A / A′ / B at creation (an assertion only: a mode change is a config reset) |
| `flags.forced` | 1 bit | k = d reached by forcing (NEE with no passing pair) |
| `c` | f32 | confidence ([confidence](#confidence)) |
| rc vertex | 3 words | one of: `(primId, bary.u f32, bary.v f32)`; `(LIGHT\|lightId, u f32, v f32)`; env NEE `(ENV, (i<<16)\|j, (du16<<16)\|dv16)` (lossless, env-local); BSDF_ENV `(ENV_RC sentinel, –, –)` |
| `rcWi` | f32×3 | ω_k at x_k (exact bits of the sampled direction); for BSDF_ENV with k = d, the **world** escape direction ω |
| `rcRadiance` | f32×3 | per case, [jacobian](#jacobian) table: (a)/(d)/(e)/(f) none, recomputed; (b) Λ_x with MIS and 1/q divided out; (c) L_e with ω2 divided out; (deep) full suffix with MIS included |
| `jDen` | f32 | the base path's own joint-pdf denominator product (Eq. 2); **replaced on selection of a shifted sample** |
| `aux` | f32 | case (b)/(c) p1 measured at x_{d−1}. Valid **within a frame only**, and only because x_{d−1} is the copied rc vertex |
| suffix cache | ~12 words | `x_{d−1}` (primId, bary f32×2), `ω_o` at x_{d−1} (f32×3), `β_s` (f32×3), then either `(lightIdx, u, v)` (NEE end) or `(t_occ, p2)` (BSDF end; escape ⇒ t_occ = FLT_MAX and an escape flag) |

- **[M4 addition, restir-api.md]** Encodings (word layout in restir-api.md §2.2–2.3): surface vertex `(primId, bits(bary.u),
  bits(bary.v))`; NEE light vertex `(0x80000000|entry, bits(u), bits(v))` with `(u, v) = (u01(h_l0), u01(h_l1))` for
  area lights and emissive triangles and `(0, 0)` for delta lights and the sun; env NEE `(0x80000000|envEntry,
  (i<<16)|j, h2)`; BSDF_ENV `(0xFFFFFFF1, 0, 0)` with ω in rcWi; none `0xFFFFFFFF`. The endpoint triple is stored for
  every path in addition to the rc triple. **Empty** reservoir ⇔ `d = 0` (W = 0, F = 0, c = 1 on a hit pixel);
  **background** ⇔ bg flag, d = 0, c = 0.
- **[M5 addition, restir-temporal-api.md]** Suffix flag `SFX_DELTA_END` (word 27 bit 3): the final BSDF event of a BSDF-ended path was a delta lobe
  (ω2 = 1). The refresh of deep BSDF ends needs it; `lobeHist` covers only x₁…x₈ while d ≤ 15.
- **[M6 addition, restir-m6-api.md §2.4, MD6–MD8]** `BSDF_ANALYTIC` (Mode B / A′ crossings): endpoint and (k = d) rc words
  `(RC_TAG_CROSS | entry, bits(x), bits(y))` with planar light-local `xy ∈ [−1, 1]²` (`z = c + x·halfU·a_u + y·halfV·a_v`;
  disks x² + y² ≤ 1), so a moved light carries its crossing rigidly and no inverse concentric map is needed. Case (c-ana):
  rcWi = ω_c, rcRadiance = L_e, aux = p1; deep: rcRadiance = β_s ⊙ (f/p)(ω_c) ⊙ ω2 L_e. Suffix cache of a crossing:
  `sfxDir = ω_c`, `sfxT` = the continuation's hit distance, flag `SFX_CROSS` (word 27 bit 4). The end term of cases (c-ana),
  deep, ∅ and of the refresh is ONE function: the ray (x, ω) re-intersected with the light (front-facing, inside, t > 0),
  L_e toward x and p1 at the crossing (a miss is a defined zero).
- Class is derivable and **not stored** ([light-changes](#light-changes)):
  - L: k = d ∧ NEE
  - N1: k = d−1 ∧ NEE
  - B1: k = d−1 ∧ BSDF_*
  - E: k = d ∧ BSDF_*
  - D-NEE / D-BSDF: k ≤ d−2
  - R: k = ∅
- Per-pixel planes outside the reservoir:
  - `thr_t` and `thr_{t−1}` (f32);
  - the V-buffer, current and previous (primId, bary f32×2);
  - background flag, depth, Ng.
- **Production (M8):** 64 B core + 32 B suffix cache. c is 8-bit and saturating only if
  c_max ≤ 255. Quantise before computing F and jDen.

---

<a id="jacobian"></a>
## 18. PSS Jacobian: Eq. 2, the light and env cases, temporal J_P

Sources: plan §2 rules 3–4 and 12, gap-light §5, gap-rc §3.5, env §3.3–3.4, gap-temporal §6,
enh-verify §4(a), gris-verify §6, review R3.

**Enhanced Eq. 2 (joint pdfs, lobe ℓ copied).**
```
J = |∂T/∂ū| = [ p^y_{k−1}(ω', ℓ_{k−1}) · G(y_{k−1}→x_k) · p^y_k(ω_k, ℓ_k) ] / jDen
jDen        =   p^x_{k−1}(ω_{k−1}, ℓ_{k−1}) · G(x_{k−1}→x_k) · p^x_k(ω_k, ℓ_k)        (stored with the base path)

ω'          = normalize(x_k − y_{k−1})
p^y_k       evaluated at x_k with V^y_k = normalize(y_{k−1} − x_k) (and η_side, q(·|V) of that V)
p_k := 1    if ω_k is NEE-sampled (ℓ_k = NEE) or x_k is a light/env vertex (k = d)
G(a→b)      = |n^g_b·(a−b)/‖a−b‖| / ‖a−b‖²       (receiving vertex, geometric normal)
k = ∅       ⇒ J = 1 (random replay is the identity in PSS)
```
- The p_k ratio applies **whenever ω_k was BSDF-sampled, whether or not MIS is on** (gris-verify
  C3; this is a Falcor bug).
- Joint pdfs are always valid-only. The support indicator multiplies both f and p
  ([support-indicator](#support-indicator)).
- **Never** apply an η² factor, and never a `t_x²/t_y²` factor for delta lights or the sun.
- Guards: `J` must be finite and > 0, otherwise the shift is undefined. Optionally, a symmetric
  `max(J, 1/J) > 11` rejection (enh-verify O6), used identically in both directions.

**Case table** (canonical letters of this document; see open item 1 for the other reports'
letters). Tp is the replayed prefix throughput Π over y₁ … y_{k−2}, with Tp = 1 for k = 2.

| Case | Definition | Spatial J | F(ȳ) (offset integrand) | Notes |
|---|---|---|---|---|
| **(a)** | forced NEE rc at the light vertex, k = d (class L) | **1** (area-uniform light-local (u,v) and a shading-point-independent pmf). A solid-angle sampler would give `p_A(z|y)/p_A(z|x)`: **forbidden** | `Tp ⊙ [f cos/p]_{y_{d−2}→y_{d−1}} ⊙ f_cos,all(y_{d−1}; ω_o^y, ω_y) ⊙ Λ_L(z; y_{d−1})/q · ω1^y · V(y_{d−1}, z)` | p1^y and p2^y are recomputed at y_{d−1} (C2). S and one-sidedness are evaluated at y_{d−1}. No criterion check on the forced pair |
| **(b)** | rc = x_{d−1}, NEE suffix, k = d−1 (class N1) | `p^y_{d−2}(ω',ℓ_{d−2})·G(y_{d−2}→x_{d−1}) / jDen` (p_k := 1) | `Tp ⊙ f_{ℓ}cos/p^y(ω') ⊙ f_cos,all(x_{d−1}; −ω' → ω_{d−1}) ⊙ rcRadiance(=Λ_x)/q · ω1^y` | **p2 changes** (new V at x_{d−1}); p1 unchanged (aux or recompute). ω_{d−1} is light-sampled |
| **(c)** | rc = x_{d−1}, BSDF-hit emitter / Mode-B light / env-escape suffix, k = d−1 (class B1) | full Eq. 2: `p^y_{d−2}(ω')·G·p^y_{d−1}(ω_{d−1},ℓ_{d−1}) / jDen` | `Tp ⊙ f_{ℓ_{d−2}}cos/p^y(ω') ⊙ f_{ℓ_{d−1}}(x_{d−1}; −ω'→ω_{d−1})cos/p^y(ω_{d−1},ℓ_{d−1}) ⊙ rcRadiance(=L_e or L_env) · ω2^y` | `ω2^y = p2^y/(M(B)·p1 + p2^y)`, where p1 = aux (triangle / light) or `p1Env(envUV(ω_{d−1}))` (env). Direction copy |
| **(d)** | rc = the BSDF-hit emitter vertex z (triangle, or Mode-B analytic area light), k = d (class E) | `p^y_{d−1}(ω',ℓ_{d−1})·G(y_{d−1}→z) / jDen`; G uses n_L or the triangle n_g; **no pmf factor** | `Tp ⊙ f_{ℓ_{d−1}}(y_{d−1}; ω_o^y→ω')cos/p^y(ω',ℓ_{d−1}) ⊙ L_e(z→y_{d−1}) · ω2^y · V(y_{d−1}, z)` | `p1^y = P(L)·p_A(z)·r_y²/|cosθ_z^y|` recomputed at y_{d−1} (C2). The area light is 0 unless `dot(y − z, a_L) > 0`. Point/spot/sun never reach this case |
| **(e)** | rc = BSDF_ENV escape, k = d, direction copy | `p^y_{d−1}(ω, ℓ_{d−1}) / jDen`, with `jDen = p^x_{d−1}(ω, ℓ_{d−1})`; **G ≡ 1, p_k := 1, no pmf, no J_P** | `Tp ⊙ 1_supp·f_ℓ(y_{d−1}; ω_o^y, ω)|cos|/p^y(ω,ℓ) ⊙ L_env(envUV(ω)) · ω2^y · V_∞(y_{d−1}, ω)` | ω is the **world** direction stored in rcWi. `ω2^y = p2^y(ω)/(M(B)·p1Env(envUV(ω)) + p2^y(ω))` |
| **(f)** | forced NEE rc on the env, k = d | **1** spatially (the env sampler is shading-point independent) | `Tp ⊙ … ⊙ f_cos,all(y_{d−1}; ω_o^y, ω) ⊙ L_env(uv)/q · ω1^y · V_∞(y_{d−1}, ω)`, `ω = envDir(uv)` | env-local (i,j,du,dv) copied; `p1 = pmf[ENV]·pdf_σ(uv)` unchanged; p2^y recomputed |
| **(deep)** | rc = surface x_k with k ≤ d−2 (classes D-NEE, D-BSDF) | full Eq. 2 at k | `Tp ⊙ f_{ℓ_{k−1}}cos/p^y(ω') ⊙ f_{ℓ_k}(x_k; −ω'→ω_k)cos/p^y(ω_k,ℓ_k) ⊙ rcRadiance` | the suffix (MIS included) is unchanged within a frame |
| **(∅)** | no rc | 1 | the replayed path's own F | must end at the same d with the same technique ([offset-checks](#offset-checks) O3) |

All offset integrands also carry the reconnection visibility `V(y_{k−1}, x_k)` (surface rc) or
`V`/`V_∞` to the endpoint, and the support indicators.

**Temporal factor J_P** (gap-temporal §6, env §3.4, plan rule 3).
```
J_temporal = J_rc (Eq. 2 / case table, frame roles as in the shift) × J_P × J_M

J_P = pmf_t(L) / pmf_{t−1}(L)      for EVERY NEE-terminated path (classes L, N1, D-NEE; cases (a), (b), (f), deep-NEE),
                                   wherever the light vertex sits relative to k; light index L translated with prevToCur.
                                   Area-uniform (u,v): A_t/A_{t−1} cancels (it enters only F through q = pmf/A).
                                   Env: J_P = pmf_t(ENV)/pmf_{t−1}(ENV) (the pdf_uv table is shared; rotation-invariant).
J_P = 1 (not applied)              for BSDF-hit endings (BSDF_TRI, BSDF_ANALYTIC, BSDF_ENV). Never folded into jDen.
J_M = A_t/A_{t−1}                  only for case (d) on a RESIZED Mode-B analytic area light (1 for rigid motion);
                                   cosine taken at the frame-t light normal.
inverse shift (t → t−1): the reciprocal factors, J_P⁻¹ = pmf_{t−1}(L)/pmf_t(L).
```
- Delta lights and the sun: J = 1 spatially, and J = J_P temporally.
- **[M5 addition, restir-temporal-api.md]** **Entries, storage and application of J_P.**
  - The stored endpoint entry is an alias index of the record's own frame. Translation between t−1 and t covers every
    entry kind: analytic lights through the id maps of the frame-t slot (`prevToCur` forward, `curToPrev` inverse),
    emissive triangles by `e − nA_from + nA_to` (triangles are static), the env to the target frame's env entry.
  - `J_P = pmf_to(e_to)/pmf_from(e_from)` with the **realized** pmf of the translated entries. A missing target entry
    or `pmf_to(e_to) ≤ 0` makes the shift **undefined** (both directions, the same predicate).
  - J_P is kept apart from J_rc: the resampling weight and π use `J_p = J_rc·J_P`; the write-back stores
    `jDen ← J_rc·jDen_src`. J_P never applies to ∅ records (BSDF ends only).
  - Mode A has no J_M (case (d) with an analytic light is Mode B); a resized area light changes F through q = pmf/A,
    not the Jacobian.

**Write-back when a shifted sample is selected** (plan rule 4; gap-rc §7.4; enh-verify C4).
```
F    ← F_c(Y)            = slot(F·J)/J      (the integrand in the destination domain)
jDen ← offset's own product = J_rc · jDen_src   (J_P and J_M are excluded)
aux  ← case (b)/(c): p1 at x_{d−1} (unchanged spatially; recomputed from frame-t lights temporally)
temporal only: light indices renumbered to frame t; refreshed rcRadiance and end-term cache
keep: seeds, d, k, technique, rc ids, ℓ_{k−1}, ℓ_k, rcWi, endpoint (L, u, v)
```

---

<a id="rc-predicate"></a>
## 19. Reconnection predicate k*

Sources: plan §2 rule 5, gap-rc §0, §3 and §6.3, glass §5.4, env §3.3, review R4 and R6.

**Pair predicate** for the pair (a, b) = (x_{k−1}, x_k), k ≥ 2, with events e_a (leaving a) and
e_b (leaving b).
```
P_k = D_k ∧ R_k ∧ F_k ∧ I_k
Δ = pos(b) − pos(a),  t² = Δ·Δ,  t = √t²,  cos_b = |n^g_b·Δ|/t,  cos_a = |n^g_a·Δ|/t

D_k : e_a not delta; and, if the path continues from b by BSDF sampling, e_b not delta
R_k : r(ℓ_{k−1}) ≥ α_min = 0.2   (PERCEPTUAL roughness; GGX α = r²; D counts as 1; delta 0)
      v1: ℓ ∈ {G_T} never passes (as ℓ_{k−1} or ℓ_k)
F_k : kind(b) == ENV  ? pass (rayFP = +∞)
                      : rayFP = t² / (p̄_a(ω_{a→b}) · cos_b) ≥ thr
I_k : skipped if kind(b) ∈ {LIGHT, ENV} or diffuseOnly(b)
      else  e_b not delta ∧ invFP = t² / (p̄_b(ω_b | from a) · cos_a) ≥ thr
guards: t² > 1e-12, p̄ > 0 and finite, cos > 0 — a failed guard FAILS the pair (except ENV's rayFP)
```
- **Env endpoint** (`kind ENV`): F passes, I is skipped, and **R at x_{k−1} is mandatory**. It is
  the only guard.
- **Threshold** (per domain, stored per pixel per frame as f32 bits; plan rule 5):
  ```
  thr = τ · R²_pri,   τ = c/100 = 2·10⁻⁴  (runtime uniform, never an override)
  R²_pri = ‖x₁ − x₀‖² · 4π / max( |⟨n^g_{x₁}, normalize(x₀ − x₁)⟩| , 1e-6 )    of the DESTINATION domain
  ```
  Here x₁ is the destination's **jittered** primary hit and x₀ its camera. The temporal inverse
  shift uses the previous camera and the previous V-buffer.
- **p̄** is the marginal, all-lobe, valid-only BSDF pdf, conditioned on the path's own incoming
  direction. It is re-**evaluated** (never the sampler's return value) from positions
  reconstructed from ids.
- **Events** (gap-rc §3.2):

| Event | Lobe | α for R | p̄ used |
|---|---|---|---|
| EV_BSDF | sampled ℓ | r(ℓ) | p̄_v(ω_sampled) |
| EV_NEE (last scattering vertex of an NEE candidate) | NEE | never x_{k−1} | p̄_v(ω_NEE) = the p2 of that NEE sample |
| EV_RECONNECT (y_{k−1} on an offset) | **copied** ℓ_{k−1} | r(y_{k−1}, ℓ_{k−1}) | p̄^y_{k−1}(ω') |
| EV_RECONNECT_NEE (y_{d−1} of a forced-NEE offset) | NEE | – | p̄^y_{d−1}(normalize(x_d − y_{d−1})) (env: the env direction) |

- **Pair list per candidate** (gap-rc §3.4):
  - pairs 2 … d−2 are shared prefix pairs (EV_BSDF, EV_BSDF);
  - pair d−1 is (EV_BSDF, EV_NEE) for an NEE candidate, or (EV_BSDF, EV_BSDF of the final ray)
    for BSDF endings;
  - pair d is the terminal pair (x_{d−1}, x_d) of kind LIGHT or ENV, **BSDF endings only**. The
    NEE light vertex is never tested.
- **k\*:**
  ```
  k* = min{ k in the pair list : P_k }
  if none:  k* = d   if technique == NEE (forced, incl. NEE_ENV)     — pair d is not tested
            k* = ∅   otherwise (full replay; must end at the same d with the same technique)
  ```
- **No lobe revocation.** An NEE candidate's k* never depends on the BSDF sample drawn later at the
  same vertex.
- **k\* is evaluated in the destination frame's light state.** EV_NEE's p̄, EV_RECONNECT_NEE and
  Mode-B terminal pairs use Φ_t for forward shifts and Φ_{t−1} for inverse shifts. If k or the
  class would differ, the shift is undefined in both directions. The refresh never relabels k or
  the class (review R4).
- **Same-formula principle** (gap-rc §8.3):
  - one `rcPairTest`, `primaryThreshold`, `vertexFromIds` and `bsdfPdfMarginal`, all in `rc.wgsl`;
  - integer compares on bit patterns (`geqPos`, `posFinite`);
  - comparison operator **≥** everywhere;
  - `margin = min over sub-tests of log2(value/thr)` or `(α − α_min)`, for the violation classes.
- Test scenes avoid r = 0.2 exactly (use 0.19 and 0.21).

---

<a id="offset-checks"></a>
## 20. Offset-path checks O0–O3

Sources: plan §2 rule 6, gap-rc §6.1–6.2 and §9.1, env §3.3, review R6.

Given a base x̄ with rc index k and destination domain D (primary y₁, `thr_D`), the shift is
**defined** iff all of the following hold. All checks go through the same `rcPairTest` with `thr_D`.
- **O0 (structure and feasibility).**
  - The replay from y₁ with x̄'s initSeed reaches y_{k−1} without a miss or termination.
  - The copied-lobe joint pdfs are > 0: `p^y_{k−1}(ω', ℓ_{k−1}) > 0`, and for a continuing x_k,
    `p^y_k(ω_k, ℓ_k | −ω') > 0`. Both include the support indicator.
  - For k = ∅: the same d and the same technique (BSDF_ENV must escape again at the same d).
  - Every light index maps into the destination frame.
- **O1 (no earlier pair passes).** For every 2 ≤ j ≤ k−1, P_j(ȳ; thr_D) is false.
  - Pairs j ≤ k−2 use the replayed EV_BSDF events.
  - The **last pre-rc pair** (y_{k−2}, y_{k−1}) uses **EV_RECONNECT** at y_{k−1}: copied ℓ_{k−1},
    ω' toward x_k, and p̄^y_{k−1}(ω'). **Never** use the direction the replay would sample at
    y_{k−1}.
  - Forced NEE uses EV_RECONNECT_NEE toward the shared light vertex.
- **O2 (the rc pair passes).** `P_k(y_{k−1}, EV_RECONNECT, x_k, e_{x_k}; thr_D)` is true, with
  recomputed p̄^y_{k−1}(ω') and p̄^y_k(ω_k | from y_{k−1}):
  - EV_BSDF with the stored ω_k (continuing), or EV_NEE (case b);
  - I is skipped for LIGHT, ENV or diffuseOnly;
  - for case (e) this reduces to D ∧ R at y_{k−1}.
  - **Skipped only for forced NEE** (cases a and f).
- **O3 (k = ∅).** For every 2 ≤ j ≤ d, P_j(ȳ) is false, **including the terminal emitter or env
  pair**.

**Claim** (gap-rc §6.1): O0–O3 ⇔ k*(ȳ; D) = k. Hence T_{D→S}∘T_{S→D} = id, and the domains match.

**Results.** Undefined and zero both give `w = 0` and `p̂_← = 0`. The distinction matters only for
bijectivity; "undefined" must be decided by identical code in T and T⁻¹.
- **Undefined:** O0–O3 fail; a light is missing in the destination frame; J is non-finite or ≤ 0.
- **Defined with F = 0:**
  - an occluded segment, including an occluded `visibleInf`;
  - a back-facing area light;
  - spot S = 0;
  - L_e = 0;
  - f = 0;
  - Mode-B full replay that no longer crosses L.

- **[M4 addition, restir-api.md]** In the paired slots, "defined with F = 0" and "undefined" share the J word `FAILED`
  (both give w = 0 and p̂_← = 0, [paired-mis](#paired-mis)); a separate per-slot code keeps the distinction
  (`SC_O0_*`, `SC_O1`, `SC_O2`, `SC_O3`, `SC_J_INVALID` = undefined; `SC_OCCLUDED`, `SC_ZERO` = defined zero). The
  order of evaluation puts every undefined decision before any zero decision, so the code is identical in T and T⁻¹.

**Violation classes** (gap-rc §10.3):
- **LOGIC:** |margin| ≥ 2⁻¹⁶ (for roughness, |α − α_min| ≥ 1e-6), any signature mismatch not
  explained by a margin flip, any T3-3 asymmetry, or any disagreement with the T3-D f64 dual.
  Target **0** in ≥ 10⁷ round trips per case.
- **FP-BOUNDARY:** |margin| < 2⁻¹⁶, or edge-hit divergence. Target ≤ 1e-5 per round trip, and
  exactly 0 in a single-pipeline or `strictMath` build.
- **VIS:** 0. **J:** 0.

---

<a id="visibility"></a>
## 21. Visibility

Sources: plan §1.3, §1.4b and rule 15, gap-light §2.5–2.6, review R11, env-verify 5.

One function `visible(a, b)` serves every segment:
- NEE shadow rays;
- reconnection, forward and inverse;
- refresh;
- the T3 re-test of the base's own rc segment.

One function `visibleInf(a, ω)` serves directions (sun, env NEE, env rc, env refresh).
```
occluders   : triangles only; alpha-MASK cutout applied (α = baseColorFactor.a·tex.a·COLOR_0.a vs alphaCutoff,
              bilinear at LOD 0); glass OCCLUDES; emissive triangles occlude; analytic lights NEVER occlude
ω           = normalize(b − a)
offsets     : Wächter–Binder per endpoint, toward the side the segment arrives from:
              a' = offset(a, sign(n^g_a·ω)·n^g_a),   b' = offset(b, sign(n^g_b·(−ω))·n^g_b)
t_max       : ‖b' − a'‖ − ε  (one shared ε)
exclusion   : both endpoint primitives (an emissive-triangle endpoint is its own primitive)
visibleInf  : offset a toward ω, t_max = FLT_MAX (never +Inf), a's primitive excluded
```
- **Consistency rule.** Lights are non-occluding for **all** rays, BSDF and shadow alike. Mixing
  (BSDF rays stop at lights while shadow rays pass) breaks the MIS partition (gap-light §2.6).
- The primary V-buffer is the first **triangle**. Camera-visible analytic lights are summed
  separately ([path-tree](#path-tree)).
- Counters for BVH stack overflow and the iteration cap (1 << 16) must be 0 in any validation run.
- The current BVH is exact for E_{t−1}, because geometry is static and lights never occlude
  (gap-temporal §4.2).

---

<a id="paired-mis"></a>
## 22. Paired spatial reuse: acceptance, MIS, W_Y, c_out, RGB shading

Sources: plan §2 rule 8 and §3 pass 5, gap-rc §7, enh §2, enh-verify C2–C4, review R8 and R13.

**Acceptance** (once per pair, canonical order, G-buffer only; `pair_accept` pass).
```
A(p, q) = partner(p) == q ∧ partner(q) == p ∧ sameMember(p, q) ∧ A0(G[min(p,q)], G[max(p,q)])
A0(a,b) = a.hit ∧ b.hit ∧ dot(a.n^g, b.n^g) ≥ 0.5 ∧ |a.z − b.z| ≤ 0.1·min(a.z, b.z)      (z = camera distance, frame t)
```
- A never depends on reservoir contents. The shift pass (S2) never re-evaluates A; it reads the
  slot status.
- **[M4 addition, restir-api.md]** `z` is the camera distance ‖x₁ − x₀‖ of the jittered primary hit, stored once per pixel as f32 bits
  (`rsVbuf.w`); `n^g` is oriented toward the camera. "Once per pair" is literal: the thread of the smaller pixel index
  evaluates `A0(G[min], G[max])` and writes the acceptance of both slots.
- Background pixels are never accepted, and cross-member partners are NOT_ACCEPTED.

**Slot status** (in the J word, compared as integers):
- `VALID`: J finite and > 0.
- `FAILED`: J bits == 0. The pair is accepted, but the shift is undefined or p̂ = 0.
- `NOT_ACCEPTED`: bits == 0xFFFFFFFF.

FAILED partners **are in** S_c (they count in k and contribute 0). NOT_ACCEPTED partners are not.

**Shift pass (S1)**, per (pixel p, slot t), reading only the immutable post-temporal `resIn`:
```
slot[t][p] = A(p,q) ? ( X_p non-empty ∧ shift defined ? (F_q(T_{p→q}(X_p))·J_{p→q}  RGB,  J) : FAILED ) : NOT_ACCEPTED
```

**MIS** (defensive pairwise, GRIS Eq. 38 with |R| = 1 and confidences; canonical pixel c).
```
S_c = { partners j with status ∈ {VALID, FAILED} },   k = |S_c|,   a_c = c_c / k     (k ≥ 1; if k = 0 keep the canonical as is)
G′_j = F_j(Z_j)·J_{c→j}       own slot (c's sample shifted into j),     Z_j = T_{c→j}(X_c)          (gap-rc: H_j)
G_j  = F_c(Y_j)·J_{j→c}       partner's slot (j's sample shifted into c), Y_j = T_{j→c}(X_j);  J_j = its J
p̂_c(X_c) = lum(F_c),   p̂_j(X_j) = lum(F_j)  (from resIn)

m_j = [ c_j·p̂_j(X_j) / ( c_j·p̂_j(X_j) + a_c·lum(G_j) ) ] / (k+1)                          (:= 0 if FAILED or denominator 0)
m_c = [ 1 + Σ_{j∈S_c} a_c·p̂_c(X_c) / ( a_c·p̂_c(X_c) + c_j·lum(G′_j) ) ] / (k+1)          (pair term := 1 if denominator 0)
w_j = m_j · lum(G_j) · W_j ,        w_c = m_c · p̂_c(X_c) · W_c
select Y ∝ w (resampling RNG);   p̂_c(Y_j) = lum(G_j)/J_j
W_Y  = Σ w / p̂_c(Y)
c_out = c_c + Σ_{j∈S_c} c_j                   (FAILED partners still count; uncapped)
```
- **Partition of unity:** m_c(y) + Σ_j m_j(y) = 1 for any positive constant in place of k
  (enh-verify C2; Falcor uses c_c/N, which is also unbiased).
- **Why this is J-free:** `p̂_{←j}(Y_j) = p̂_j(X_j)/J_{j→c}` and `p̂_c(Y_j) = lum(G_j)/J_{j→c}`, so
  J cancels between numerator and denominator.
- **RGB shading** (Enhanced §6.3):
  ```
  L = m_c·F_c(X_c)·W_c + Σ_{j∈S_c} m_j·G_j·W_j
  ```
  m already contains 1/(k+1). Gate-0 check: `lum(L) = Σ w` to 1e-6 relative. Without a spatial
  pass, shade with F(Y)·W.
- **Write-back** on selecting Y_j: see [jacobian](#jacobian).
- Reservoirs are ping-ponged: S1 and S2 read `resIn` and S2 writes `resOut`.
- **Disocclusion boost:** extra pairing textures only as reciprocal pairs, i.e.
  `A(p,q) ∧ (dis(p) ∨ dis(q))`, decided once in `pair_accept`.
- Do **not** copy EvanLuo42's MIS (its arguments are swapped) or its shift (no O1/O2).

<a id="pairing-textures"></a>
**Pairing textures** (Enhanced §3; enh-verify C1; plan M6).
- **Generation:**
  - link index `L(x, y) = (y·W + x) >> 1`;
  - n_σ shuffles of random 2×2 block permutations, where every odd pass offsets the grid by (1, 1)
    on a torus;
  - partner delta `d = wrap(b − a)`, with d(b) = −d(a), and
    `wrap(v) = v > W/2 ? v − W : (v < −W/2 ? v + W : v)`.
- **n_σ** ([dupmap](#dupmap)). σ = 16 gives n = 128.
- **Sizes:** 254, 230 and 210 px, `rg8sint` (|Δ| ≤ 127). They are generated on the CPU.
- **Per frame, round and member:** a dihedral transform M (8 choices) and an offset o.
  ```
  q = (M·p + o) mod W,   partner(p) = p + Mᵀ·d(q)
  ```
  Reciprocity partner(partner(p)) = p survives any transform. Off-screen partners are lost
  symmetrically.
- σ = sqrt(8/(9π))·R ≈ 0.5319·R, so R = 30 px gives σ = 16.0.
- The M4 uniform-disk involution maps are replaced by these in M6 without kernel changes.
- **[M4 addition, restir-api.md]** **M4 uniform-disk involution maps**: same texture format and transform. Each layer is a W_s-torus
  involution built on the CPU by greedy random matching: texels in random order; an unmatched texel a draws up to 64
  integer offsets d uniformly from `0 < |d|² ≤ R²` and is matched with the first unmatched `b = (a + d) mod W_s`
  (`d(b) = −d(a)`); leftovers keep d = 0 ("no partner", NOT_ACCEPTED). R = 30 px interactive, 10 px offline. Offsets
  are only approximately uniform (late matches are constrained); this changes variance only.

---

<a id="temporal"></a>
## 23. Temporal step: contribution MIS with exact E_{t−1}

Sources: plan §2 rule 9 and §3 pass 4, gap-temporal §3 and §4, review R1, R8 and R15.

**Lemma** (GRIS Eq. 17/18 with arbitrary resampling weights). Resampling weights w̃ only affect
variance. Unbiasedness requires only that the contribution weights ĉ_i be one deterministic
partition of unity whose support matches the true producibility sets.

**Definitions.**
```
c_c = 1,  c_p = min(c_cap = 20, c_prev)                            ([confidence])
q′  = temporalPixel(q): a function ONLY of the current/previous G-buffers, (dual) motion vectors and a
      sample-independent RNG (back-projection rounding, 3×3 search order/acceptance, dual-MV choice);
      never of reservoir contents. No q′ ⇒ no temporal candidate (c_p = 0).
T   : temporal hybrid shift (p′, t−1) → (q, t);  J_p = J_rc · J_P [· J_M]      ([jacobian])
π_c(y) = p̂_t(y) = lum F_t(y)
π_p(y) = lum F_{t−1}(T⁻¹ y) · |∂T⁻¹/∂y|     (0 if y ∉ image of T)
```

**Pseudo-code** (passes T1–T4).
```
T1 temporal_classify (per pixel q):
   X_c, W_c, F_c ← initial reservoir (F_c = F_t(X_c))
   if !valid(q′):  out = (X_c, W_c, c = 1, F_c); done
   (X_p, W_p, c_prev, F_p^st) ← reservoir_prev[q′]          // READ-ONLY (I1 relies on it)
   c_p = min(20, c_prev)
   translate every stored light index of X_p (endpoint + suffix-cache lightIdx) with prevToCur; any −1 ⇒ Y_p undefined
   inline reconnection-only forward shifts (k = 2, or class L with d = 2); else append to Q_f
T2 temporal_forward (indirect over Q_f):
   (Y_p, J_p, F_t(Y_p)) = T(X_p) under S_t, with the end term refreshed ([light-changes]); scratch only
T3 temporal_select:
   w̃_c = c_c · lum(F_c) · W_c
   w̃_p = c_p · lum(F_t(Y_p)) · W_p · J_p            (0 if undefined, occluded, or no q′)
   if w̃_c + w̃_p == 0:  out = empty (W = 0, c = c_c + c_p); done
   s = select(w̃_c, w̃_p)                             (resampling RNG)
   if s == p:  π_c = lum F_t(Y_p);   π_p = lum(F_p^st) / J_p;   Y = Y_p;  finalize
   else:       append q to Q_i
T4 temporal_inverse (indirect over Q_i), under E_{t−1}:
   (F_prev, J_inv) = T⁻¹(X_c): replay from the previous jittered V-buffer hit of q′ with the previous camera,
                     thr_{t−1}[q′], lightsPrev, pmf_{t−1}, curToPrev (−1 ⇒ F_prev = 0), the end term under L_{t−1}
   π_c = lum F_c;   π_p = lum(F_prev) · J_inv;   Y = X_c
finalize:
   W_Y   = π_s / (c_c·π_c + c_p·π_p) · (w̃_c + w̃_p) / π_c        (π_c = p̂_t(Y))
   c_out = c_c + c_p
   out[q] = (Y, W_Y, c_out, F_t(Y), refreshed caches, light indices in frame-t numbering, jDen per [jacobian] write-back)
```

**E_{t−1} contents** (exact previous-frame evaluator):
- previous camera position and view-projection;
- the **stored jittered previous V-buffer**;
- thr_{t−1};
- lightsPrev, including the env record `(cγ, sγ, scale, pmf[ENV])_{t−1}`, and pmf_{t−1};
- the `curToPrev` / `prevToCur` maps;
- the previous ray-cone LOD plane (interactive; LOD 0 in exact modes);
- in Mode B, the crossing loop over lightsPrev;
- the frame config: M(B), τ, α_min, max length, BSDF tier, env map id, importance resolution, env
  NEE.

**Any config change resets history.**
- The current BVH suffices; no previous TLAS or previous light tiles are needed (gap-temporal
  §4.2–4.3).

**Invariant I1.** Every reservoir written at the end of frame t holds `F = F_t(X)` exactly: suffix
radiance valid under S_t, in its own pixel domain.
- Initial samples satisfy it.
- The temporal step keeps it by writing the refreshed F and caches.
- Spatial reuse keeps it trivially.
- **reservoir_prev is read-only during the whole frame. The suffix refresh writes scratch only.**

**Variants.**
- **Talbot-exact** (debug cross-check): always compute π_p(X_c).
  `m_c = c_c p̂_t/(c_c p̂_t + c_p π_p)` and `m_p = c_p π_p/(c_c p̂_t + c_p π_p)`, with
  `w_c = m_c(X_c)·lum F_c·W_c`, `w_p = m_p(Y_p)·lum F_t(Y_p)·W_p·J_p` and
  `W_Y = (w_c + w_p)/lum F(Y)`.
- **Robust (validation, T6(b)):** also recompute π_p(Y_p) with E_{t−1}(T⁻¹Y_p) and assert that it
  equals the stored route to within 1e-3.

**Forbidden.** Falcor's mixed evaluator (stored self-term + current-lights inverse term) is biased:
`B(r) = 1/(1+c_p) − r/(r+c_p)`, which is −4.3% at r = 2 and c_p = 20. EvanLuo's swapped temporal
MIS is also forbidden.

**[M5 addition, restir-temporal-api.md]** **Pinned details of the temporal step** (restir-temporal-api.md §3.3, §3.6):
- **q′ rule.** Rebuild x₁(q) from the current V-buffer ids, project it with the previous camera to the continuous
  position s′, round stochastically `c₀ = ⌊s′ + ξ⌋`; accept the first tap c of {c₀, then the 8-ring around c₀ in a fixed
  order rotated by the pick hash} with: previous V-buffer hit at c, same member, `dot(n^g, n^g′) ≥ 0.5` (both toward
  their own camera), `|‖x₁ − o_{t−1}‖ − z′| ≤ 0.1·z′` (z′ = stored camera distance at c). No tap ⇒ no temporal candidate.
- **No q′** (or no history): the canonical record is left **bitwise unchanged** (c = 1), not rewritten through the
  formula (which would round W).
- **A valid q′ whose record is empty, or whose shift is undefined, still counts:** c_p = min(20, c_prev), w̃_p = 0, and
  π_p(X_c) is still evaluated when s = c. The partition of unity is over producibility sets, not over realized samples.
- **Selection:** streaming RIS with the canonical first (counter 0) and the temporal candidate second (counter 1).
- **In place:** the output is written into the canonical's own buffer at its own pixel (s = p after selection, s = c
  after the inverse); the previous-frame buffer is read-only until the last temporal read of the frame (I1).
- **Degenerate:** w̃_c + w̃_p = 0 ⇒ empty record with c = c_c + c_p; a non-finite W_Y ⇒ empty record, counted (must be 0).
- **Variants.** Talbot-exact selects after the inverse evaluation (all valid-q′ pixels); "recompute" uses
  π_p(Y_p) = lum F_{t−1}(T⁻¹Y_p)·|∂T⁻¹/∂y| instead of the stored route (unbiased with the exact E_{t−1}, consistent case
  (i) of gap-temporal §3.5 otherwise); "robust" computes both, uses the stored route and asserts agreement (T6(b)).
- The W-scale plant (`wScale`) multiplies the temporal W_Y as well as the spatial W.
- **Write-back of Y_p:** all fields of X_p, then `F ← F_t(Y_p)`, `jDen ← J_rc·jDen`, NEE entry words and `endpointId`
  renumbered to frame t, deep `rcRad` and the (b)/(c) cache (`rcRad`, `aux`) from the frame-t refresh, `W`, `c`.

---

<a id="light-changes"></a>
## 24. Light and env changes (taxonomy)

Sources: plan §2 rule 10, gap-temporal §5 and §7, env §3.4, env-verify omission 1, review R2.

**Refresh scope is per frame.** If `pmf_t ≠ pmf_{t−1}` bitwise, **every** light-terminated
reservoir gets its end term re-evaluated analytically, with no rays. That covers NEE ends (1/q and
ω1) and BSDF-hit ends (ω2 through p1 of the hit emitter), including emissive triangles in Mode A.
- **Shadow rays** are needed only for samples whose **own** light moved, rotated or was resized:
  ≤ 1 ray each via the suffix cache.
- **End-term refresh formulas** (frame selector s ∈ {t−1, t}; gap-temporal §5.5).
  ```
  NEE end at x with ω_o:   y = Φ_s(L, u, v), ω = normalize(y − x), r = ‖y − x‖
     area:   N_s = f(x; ω_o, ω)|n_x·ω| · V(x,y) · L_e,s(y, −ω) · ω1_s / p1σ_s,
             p1σ_s = (pmf_s(L)/A_s(L)) · r²/|n_L^s·ω|,   ω1_s = M(B)·p1σ_s/(M(B)·p1σ_s + p2(x; ω_o→ω)),  L_e,s = 0 on the back side
     point/spot: N_s = f(x;ω_o,ω)|n_x·ω| · V · I_s(−ω)/r² / pmf_s(L),   ω1 = 1
     env:    ω = envDir(uv, cs_s),  N_s = f(x;ω_o,ω)|n_x·ω|·V_∞(x,ω)·L_env,s(uv)·ω1_s / (pmf_s(ENV)·pdf_σ(uv))
  class D-NEE:  L_k^(s) = β_s ⊙ N_s(x_{d−1}, ω_o)
  class B1 / D-BSDF (BSDF end, direction ω_{d−1}):  re-intersect analytic lights of frame s for t < t_occ
                (escape: env of frame s, L = envRadiance(envUV(ω, cs_s))); emitter term L_e,s·ω2_s,
                ω2_s = p2/(p2 + M(B)·p1σ_s(hit))
  radiometric-only (light not moved, N_{t−1}, L_e,t−1, ω1_{t−1} > 0):
                N_t = N_{t−1}·(L_e,t/L_e,t−1)·(ω1_t/ω1_{t−1})·(q_{t−1}/q_t)
  ```
- A cached case-(b)/(c) `aux p1` is **never** reused across frames.

| Event (CPU diff of cur vs prev) | Classification | Forward (S_t) | Inverse (S_{t−1}) | Rays |
|---|---|---|---|---|
| Light moved / rotated / resized | moved | explicit NEE vertex rebuilt `Φ_t(L,u,v)`; Λ, p1, ω1, V re-evaluated; J_P (and J_M only for case (d)) | same with Φ_{t−1} | 1 shadow per explicit NEE sample on that light; BSDF ends: light loop with t < t_occ (0 BVH rays) |
| Intensity / colour / exposure / spot angles / blend | radiometric | analytic rescale; pmf rebuilt ⇒ J_P ≠ 1 for **all** NEE-terminated samples; end terms of all light-terminated samples | with L_{t−1}, pmf_{t−1} | 0 |
| Light removed (`prevToCur = −1`) | remove | explicit vertex on L: **undefined** (w̃_p = 0); BSDF-terminated: re-intersection misses L (defined, 0) | canonical never involves L | 0 |
| Light added (`curToPrev = −1`) | add | no temporal sample involves it | canonical on L: **undefined** ⇒ π_p = 0 (exact support) | 0 |
| Type or shape-topology change | remove + add with a new stable id | – | – | – |
| **Env rotation γ** | env **moved** | NEE_ENV: `ω_t = envDir(uv, cs_t)`, new V_∞, L_env(uv) and p1 unchanged. BSDF_ENV and escapes: world ω; `L = envRadiance(envUV(ω, cs_t))`, ω2_t with `p1Env(envUV(ω, cs_t))` | same with cs_{t−1} | 1 ray to FLT_MAX per NEE_ENV sample; 0 for BSDF escapes |
| **Env strength / tint** | radiometric | L scaled; pmf rebuilt ⇒ J_P on every NEE-terminated sample (all lights) | scale_{t−1}, pmf_{t−1} | 0 |
| Env map swap, importance-resolution change, Env-NEE toggle | **config change** | history reset | – | – |
| Env added / removed | add / remove with `ENV_ID` | removed: NEE_ENV undefined, BSDF escapes defined with L = 0 | added: canonical NEE_ENV π_p = 0 | 0 |
| visibleToCamera toggled | length-1 only | – | – | 0 |
| Camera only | – | no refresh (the cached suffix is exact) | previous camera in E_{t−1} | 0 |

- The alias table is rebuilt only when the light set or a power-relevant property changes, so the
  pmf stays bitwise stable otherwise.
- **Never** replay RIS-NEE over per-frame light tiles.

**[M5 addition, restir-temporal-api.md]** **Refresh responsibilities, change classes and the visibility rule** (restir-temporal-api.md §3.5):
- **Two places evaluate end terms under the destination frame's state:** the shift itself (classes L, E, B1 incl. env
  escapes, ∅ through replay, and the prefix-dependent factors of N1: f at x_k with the offset's V and ω1 with the
  offset's p2), and the separate refresh pass (prefix-independent parts only): the deep suffix radiance
  `β_s ⊙ N_s` or `β_s ⊙ ω2_s·L_s` (classes D-NEE, D-BSDF), the N1 end visibility V(x_k, Φ_s), the translated entry and
  J_P, and the (b)/(c) cache values (Λ_s or L_s and p1_s at x_{d−1}) for the write-back. The refresh writes scratch
  only and runs on every frame whose light, pmf or env state changed ("refresh scope per frame").
- **Change classes** (CPU diff of the light records, stored per light): MOVED ⇔ a record word that enters the light
  point or direction Φ differs (point/spot: position; sun: direction; rect/disk: position, axes, half sizes, normal);
  RADIO ⇔ any other word differs (emission, spot cone/blend, spread, area, flags, the spot axis). Env: MOVED ⇔ rotation
  words differ; RADIO ⇔ strength or tint differ.
- **Visibility rule.** A shadow ray (N1 end, D-NEE end) is traced iff the endpoint's own light MOVED between t−1 and t
  (env: rotated). Otherwise V = 1 exactly: every record the refresh serves has p̂ > 0 in its own frame (it was
  selected), geometry is static, and an unmoved light keeps Φ(L, u, v).
- **Deep refresh formulas** use the stored suffix cache and the path tree's own functions and grouping:
  `x = scene_surface(sfx ids, −sfxDir)`, `V = sfxDir`; NEE end `rad = β_s·(ω1/q)·f_all·Λ` with ω1 from
  `bsdf_query(m, V, ω_L, LOBE_NEE).p_marg`; BSDF end `rad = β_s·(ω2·L)`, ω2 = 1 after a delta end (`SFX_DELTA_END`), else
  from p1 of the frame's pmf and the cached p2. With an unchanged state the result equals the stored `rcRad` to 1e-6.
- **E2 (class zeroing, unbiased fallback):** on refresh frames, deep-class temporal samples are undefined (w̃_p = 0)
  and deep canonicals get π_p := 0; every other class stays exact.
- **Forbidden:** a per-light refresh mask (refreshing only samples whose own light changed). Any pmf change alters
  1/q and ω1/ω2 of every light-terminated sample (plan-review R2; planted control N7).

---

<a id="rr"></a>
## 25. Russian roulette

Sources: plan §2 rule 11, enh §6.5, enh-verify O3, review R14.

- RR exists **only at initial sampling**. The PSS is defined without RR, and RR changes only the
  initial sample's source pdf: `p(ū) = ∏ q_i`. **Replay never applies RR.**
- RR test i happens at x_i, **after** NEE at x_i and before the BSDF continuation from x_i.
  ```
  NEE candidate ending at x_d (sampled from x_{d−1}):          survived tests i = 1 … d−2
  BSDF-hit candidate at x_d (BSDF_TRI, BSDF_ANALYTIC, BSDF_ENV): survived tests i = 1 … d−1
  W_src = W_NEE / ∏_{i=1}^{d−2} q_i     or     1 / ∏_{i=1}^{d−1} q_i
  ```
- q_i ∈ (0, 1] is any deterministic function of the prefix (the path stream up to x_i), drawn with
  slot `u_rr` ([rng-layout](#rng-layout)). The plan does not fix the formula; see open item 34.
  Recommended: Cycles' form `q_i = min( sqrt(max_c |β_c|), 1 )` on the RR-free prefix throughput
  β, with no RR while i ≤ minBounces.
- Gate ladder: 3.1b tests RR alone. RR stays off in rungs 3.2–3.6 and is toggled in 3.7.
- **[M4 addition, restir-api.md]** Pinned for ReSTIR (resolves open item 34 for the ReSTIR paths): exactly the reference PT's rule,
  `q_B = min(sqrt(max_c β_c), 1)` on the RR-free prefix throughput, tested at x_B for `B > rrMinBounces` (rung 3.1b uses
  rrMinBounces = 1), drawn from slot `u_rr` of the path seed.

---

<a id="confidence"></a>
## 26. Confidence weights

Sources: plan §2 rule 13 and §1.9, gap-temporal §3.7, review R15.
- **Validation keeps c in f32.** Initial c = 1.
- **Temporal:** `c_p = min(20, c_prev)`, `c_out = c_c + c_p = 1 + c_p`. The same capped c_p is used
  in the weights and in c_out.
- **Spatial:** `c_out = c_c + Σ_{j∈S_c} c_j`, **uncapped**. The next temporal step caps it.
  FAILED partners count; NOT_ACCEPTED partners do not.
- **Offline:** temporal is off, and there are 3 rounds × 6 paired partners with fresh maps each
  round, so c ≤ (1+6)³ = 343 (f32).
- **Production 8-bit c:** allowed only when c_max ≤ 255, saturating with `min(255, ·)`.
- Any fractional or sample-dependent cap (the duplication map) is biased and off in every
  unbiasedness gate.
- **[M5 addition, restir-temporal-api.md]** No q′ (or no history) keeps the canonical c = 1. An empty temporal output keeps c = c_c + c_p. c_prev is the
  previous frame's final (post-spatial, uncapped) c; the cap is applied once, in the temporal step.

---

<a id="dupmap"></a>
## 27. Duplication map (biased) and the corrected n_σ

Sources: plan §2 rule 13 and §3 pass 6, enh §4, enh-verify C1 and C10.

```
D(p)  = #{ q ∈ 17×17 window around p, q ≠ p : initSeed(q) == initSeed(p) ≠ 0 } / 288           (288 = 17² − 1)
c_Cap = lerp(c_Cap^Default = 20,  c_Cap^min = 1,  D^α),   α = 0.1        = 20 − 19·D^0.1
c_p   = min(c_Cap, c_prev)          (D looked up at the temporal source pixel q′ in the previous frame's map)
```
- Worked values: D = 1/288 → c_Cap = 9.22; D = 10/288 → 6.42; D = 0.2 → 3.83.
- **Biased.** c becomes sample-dependent, so the MIS weights no longer form a partition of unity.
  It is off in all unbiasedness gates. The Gate 5 bias budget is measured against Enhanced's 3.25%.
- Output is an `r32uint` storage texture holding the count. Empty reservoirs have id 0 and are
  skipped.

**Corrected shuffle count for the pairing textures** (the paper prints +1.76; a sign typo,
enh-verify C1):
```
n_σ = ⌊ σ²/2 + 1.46·σ⁻¹ − 1.76·σ⁻² + 0.656·σ⁻³ + 0.5 ⌋          n_σ(16) = 128,  n_σ(0.814) = 1
```
See [pairing-textures](#pairing-textures).

**[M6 addition, restir-m6-api.md MD3, MD10]** Realisations. Gaussian pairing maps: n_σ shuffles of 2×2 blocks on a W-torus
(link L = (yW + x) ≫ 1; every odd pass offset by (1, 1)), the partner delta stored explicitly with d(b) = −d(a); layer
sizes [254, 230, 210, 246, 238, 222], deterministic PCG32 per (layer, σ, W); T14 measures per-axis σ within 3 % of 16
and KS < 0.01. Duplication map: `rs_dupmap` after the frame's final reservoirs counts, per atlas pixel, the pixels q ≠ p
of the 17×17 window (same member, inside the tile) with the same non-empty 64-bit seed; the next frame's T1 caps
`c_p = min(c_Cap, c_prev)` with `c_Cap = cCap − (cCap − 1)·(count/288)^0.1` (f32, not truncated). Biased; off in every
unbiasedness unit; Gate 5 bounds its bias (dup_bias.py).

---

<a id="validation-stats"></a>
## 28. Validation statistics (compare.py)

Sources: plan §7.3, val §3.

**Replicates.**
- Uncertainty comes from **replicates of the statistic**: Cycles K seeds, PT batches B, ReSTIR runs
  R. Never sum per-pixel variances.
- For an aggregate A (pixel, tile, mask region or image), compute A per replicate, then the mean
  and SE of those:
  ```
  X̄_A = mean_b X_{A,b},   SE_X = s_X/√B,  ν_X = B − 1        (ours)
  R̄_A = mean_k R_{A,k},   SE_R = s_R/√K,  ν_R = K − 1        (reference)
  Δ_A = X̄_A − R̄_A,   SE_Δ = sqrt(SE_X² + SE_R²),   t_A = Δ_A/SE_Δ
  ν_A = (SE_X² + SE_R²)² / ( SE_X⁴/ν_X + SE_R⁴/ν_R )            (Welch–Satterthwaite)
  ```

**TOST** (the pass rule; α = 0.01 per side).
```
pass ⇔ |Δ_A/R̄_A| + t_{1−α, ν_A} · SE_Δ/R̄_A < δ
δ:  PT vs Cycles        0.5% global, 2% per tile
    ReSTIR vs PT        0.2% global, 1% per tile
    dynamic             0.2% global, 2% per 64² tile, 3% per mask region
MDB ≈ (z_{1−α/2} + z_{1−β}) · SE_Δ/R̄        (reported per test)
```

**Sizing rule** (choose spp, K, R and B before running; take SE from a ~5% pilot).
```
SE_Δ/R̄ ≤ δ / ( t_{1−α, ν} + z_{1 − 0.005/m} )          m = number of tiles  (≈ δ/6.4 for m = 256)
```
If this is unaffordable, enlarge the aggregates and record that in `report.json`. **Never loosen δ
silently.**

**Multiplicity.**
```
Šidák per-tile level:  α′ = 1 − (1 − α_u)^{1/m},   threshold t* = t_ν⁻¹(1 − α′/2)
Suite FWER:            α_u = 1 − 0.99^{1/n_units},   n_units = scenes × configs × gates × {Y, R, G, B}
```
- A failed unit is re-run once on disjoint seeds.
- BH-FDR at q = 0.01 is used for visual maps only.

**Distribution checks.**
- `χ²_red = (1/m) Σ_A t_A²`, required to be `< ν/(ν−2)·(1 + 4·sqrt(2/m))`.
- mean-t, whose SE is 1/√m.
- KS/AD of {t_A} against t(ν).

**Dark tiles and zeros.**
- If `R̄_tile < 0.05·R̄_image`, use an absolute margin `δ·0.05·R̄_image` instead of a relative one.
- Zero-variance tiles must match to 1e-6.
- Guard every 0/0.

**Metrics.**
```
relMSE_corr(N) = (1/P) Σ_p [ (X̄_{N,p} − R̄_p)² − SE_R,p² ] / (R̄_p² + ε),   ε = 0.01 after normalising mean(R) = 1
N·MSE_corr(N) = σ² + N·b²          (flat iff unbiased)
BNR(N) = |Δ̄|/SE_Δ                  (E[BNR²] ≈ 1 under H0; grows ∝ √N under bias)
Gate-4 slope: fit log relMSE_corr = a + s·log N,  s ∈ [−1.1, −0.9]   (no-cap run = negative control)
MAPE(I, I_gt) = mean( |I − Ĩ_gt| / (0.01·mean(Ĩ_gt) + Ĩ_gt) )         (Ĩ = greyscale)
```
HDR-FLIP is reported alongside.

**Calibration.**
- A/A tests on disjoint splits of existing replicates, with ≥ 20 re-splits.
- δ-scale plants must fail in ≥ 9/10 repeats:
  - Stage A: light ×1.0075, and one 32² region +3%;
  - Stage B: W ×1.003, and c_p + 1 in the MIS denominator.

**Heavy tails.** Test aggregates, not pixels, with ≥ 16 replicates. Median-of-means is optional.
**Never** clamp, trim or winsorise to make a scene pass.

---

<a id="open-inconsistencies"></a>
## Open inconsistencies

Conflicts found between PLAN.md and the reports, or between reports. Each has a proposed
resolution, and none has been applied silently to code. Items resolved by the plan are listed so
that implementers do not follow the superseded text.

1. **Reconnection-case letters differ between documents.**
   - gap-light uses (a) forced NEE, (b) rc = x_{d−1} with NEE suffix, (c) rc = x_{d−1} with
     BSDF-emitter suffix, (d) emitter as rc, (e) deep rc k ≤ d−2, (n) no rc.
   - gap-rc uses (a) continuing surface, (b) NEE light vertex, (c) BSDF-hit triangle, (d) Mode-B
     light, (e) env escape, (f) env NEE, (g) NEE-final x_{d−1} as rc, (h) k = 2.
   - PLAN rule 3 uses (a), (d), (e) = env escape and (f) = env NEE.
   - **Resolution:** this document keeps the PLAN letters (a), (d), (e) and (f), fills (b) and (c)
     from gap-light, and calls gap-light's (e) "(deep)".
   - Mapping: gap-light (e) → (deep); gap-rc (b) → (a); gap-rc (c)/(d) → (d); gap-rc (g) → (b);
     gap-rc (a) → (deep) or (c); gap-rc (h) is "k = 2", not a case.
   - WGSL and tests should use these letters.
2. **Lobe codes.**
   - gap-bsdf: 2 bits `{D=0, S=1, NEE=2, NONE=3}`.
   - glass: 3 bits `{D=0, S=1, G=2, TRANSP=3, NEE=4, NONE=5}` plus side bits.
   - PLAN §1.5 and review R9: `{D, S, G_R, G_T, NEE, NONE}` plus a delta bit.
   - **Resolution:** the PLAN set with `D=0, S=1, G_R=2, G_T=3, NEE=4, NONE=5` ([bsdf-v2](#lobe-codes)).
     TRANSP is dropped, because MASK consumes no event in v1. The side bits are redundant with
     G_R/G_T.
   - The WGSL constants printed in gap-bsdf §11 and glass §9 must be renumbered.
3. **Glass side flip at a reconnection vertex: zero or undefined?**
   - glass §6.3 says a side flip at x_k makes `f·1_supp = 0`, i.e. a zero contribution.
   - review R9(d) says that with a copied G_R/G_T code the joint pdf is 0, so the shift is
     undefined.
   - Both give w = 0, but the classification must be identical in T and T⁻¹.
   - **Resolution:** treat it as an **O0 failure** (copied-lobe joint pdf = 0 ⇒ undefined),
     evaluated identically in both directions.
   - In v1 only G_R can reach reconnection, because G_T never passes the predicate.
4. **BSDF RNG dimensions.**
   - gap-bsdf §7.3 and §11: 3 dims per bounce `(u_sel, u1, u2)`.
   - val T1: a different example layout.
   - glass §5.6 and PLAN §1.7: 4 dims `(u_lobe, u_h1, u_h2, u_rt)`.
   - **Resolution:** 4 dims (PLAN). The gap-bsdf `bsdf_sample(u: vec3f)` signature becomes
     `vec4f`. Slot numbers are pinned in [rng-layout](#rng-layout).
5. **Global light alias: bucket selection is unspecified.**
   - PLAN §1.4 requires u16 thresholds and the realized pmf for every alias table.
   - Only the env tables define exact bucket selection (power-of-two n, top bits of a hash).
   - The global table has arbitrary N, possibly > 2¹⁶ emissive triangles. `floor(u·N)` is not
     exactly uniform, so the "realized pmf" would be wrong at the 2⁻³² level, and one hash cannot
     hold > 16 bucket bits plus 16 threshold bits.
   - **Resolution (proposed):** pad the table to n = 2^⌈log2 N⌉ with zero-weight entries that
     always alias to real entries. The bucket is the top log2 n bits of `u_sel`. The threshold is
     the low 16 bits of `u_sel` if log2 n ≤ 16, otherwise the low 16 bits of `u_sel2` (slot 8).
     The realized pmf uses the same formula as the env tables.
6. **MIS heuristic.**
   - gap-bsdf §11 usage notes and gap-light §1 describe Cycles' power heuristic.
   - PLAN rule 2 and gap-light §3.7 fix the balance form with M(B).
   - **Resolution:** the balance form with M(B) everywhere in our renderer. The power heuristic
     appears only as an env negative control; the expectation is identical.
7. **Mode A vs Mode B.**
   - gap-light §0 and §2.7 recommend Mode A as the v1 default and claim unconditional identical
     expectation.
   - PLAN: Mode A is the validation start and B the product default after Gate 3.11. A ≡ B holds
     only without singular lobes (glass §4.3). The unconditional claim is superseded (PLAN §4.2).
8. **M(B) rounding.**
   - Enhanced gives `32/B²` without rounding.
   - PLAN: 32 at B = 1 with RIS-NEE, else 1.
   - **Resolution:** as the PLAN states. If deeper RIS-NEE is ever enabled, use `max(1, ⌊32/B²⌋)`.
     M(B) is in the frame config.
9. **Class invariance under light edits.**
   - gap-temporal §5.1 and test §9.3-6 claim that k and the class never change when lights move.
   - Under the adopted gap-rc predicate, EV_NEE's p̄ and EV_RECONNECT_NEE depend on Φ(L,u,v), so
     L ↔ N1 can flip.
   - PLAN rule 5 (review R4): k* is evaluated in the destination frame's light state, and a class
     change makes the shift undefined in both directions.
   - **Action:** replace test §9.3-6 with "T defined ⇒ k and class preserved; report the
     class-change rate."
10. **Refresh scope.**
    - gap-temporal §0.5 and §5.4 key the refresh per moved light, and treat intensity edits as a
      per-light rescale.
    - PLAN rule 10 (review R2): the scope is per frame. Any pmf change re-evaluates every
      light-terminated end term. gap-temporal's per-light mask is superseded (planted control N7).
11. **Pair-acceptance thresholds.**
    - gap-rc: n·n ≥ 0.5 and |Δz| ≤ 0.1·min(z).
    - enh-verify C3: 0.1·max(z).
    - EvanLuo: n·n ≥ 0.9.
    - The PLAN gives no values. All are symmetric and variance-only.
    - **Resolution:** gap-rc's values (a gap-* spec outranks the verify).
12. **Env pdf and pole cap.**
    - gap-env §2.4 uses `s = sin(π·min(v,1−v))` for every lookup and `t_max = +∞`.
    - env-verify and PLAN §1.4b: `s = ‖b.xy‖` for directions, the sine form only for NEE lattice
      samples, and FLT_MAX. The PLAN wins. The cell clamp (env-verify 1) is also adopted.
13. **Env tiers.** gap-env §5.4: tight ≤ 12 EV. PLAN §7.2: tight ≤ 18 EV with no sun or caustic
    chains. The PLAN wins, and (xiii) with `studio_small_09` (18 EV) is tight.
14. **Negative env texels.** gap-env §2.1 keeps negatives, because Cycles renders them. PLAN and
    env-verify: reject in validation, clamp to 0 interactively, since p̂ must be ≥ 0. The PLAN wins.
15. **Env-NEE toggle.** gap-env lists the UI toggle but not its temporal class. PLAN rule 10: a
    config change that resets history.
16. **Reservoir flag layout.**
    - gap-light §2.2 and §4.1 give a 64 B layout with 2-bit lobes and `k: 0 = none`.
    - glass §5.5 moves the lobes to 3 bits at bits 22–27.
    - PLAN §1.9 validation layout: uncompressed, 3+1-bit lobes, mode bits.
    - **Resolution:** [reservoir-fields](#reservoir-fields) for validation. The M8 production
      packing must be re-derived; gap-light §4.1 is not normative.
17. **Pairwise MIS normaliser.** Code22 and enh §2.2 use `c_c/N` (nominal slots); gap-rc and the
    PLAN use `c_c/|S_c|`. Both are unbiased. The PLAN wins.
18. **n_σ sign.** The paper and val print `+1.76σ⁻²`. enh-verify and the PLAN use `−1.76σ⁻²`. The
    PLAN wins. The value at σ = 16 is 128 either way.
19. **Footprint constant.** RTXDI uses `0.02²` = 4·10⁻⁴·R²_pri. The paper and the PLAN use
    τ = 2·10⁻⁴. The PLAN wins.
20. **Roughness units.**
    - EvanLuo compares GGX α ≥ 0.2, i.e. r ≥ 0.447.
    - gap-env's scene lists said "α ∈ {…, 0.2}".
    - PLAN: perceptual r ≥ 0.2, and scenes avoid r = 0.2 exactly. The PLAN wins.
21. **G cosine vertex.** The Enhanced paper p.4 and supplement §1 say "normal at y_{k−1}". GRIS
    Eq. 52, Falcor and enh-verify C8 put it at the receiving vertex x_k. The receiving vertex is
    adopted.
22. **Black-box PDF roughness proxy (S-Eq. 26)** is printed as `1/p² ≥ α_min`, which is a likely
    scale error. It is not used: all v1 materials are parametric.
23. **gap-bsdf r(NEE) = max over lobes** exists but is never consumed. EV_NEE is never x_{k−1} in
    the gap-rc pair list. Harmless; keep it only for debug views.
24. **Glass G_T reconnection.** glass §6.2 allows rough-glass T events as x_{k−1} and x_k. PLAN
    rule 5 says v1 G_T never passes; rough-transmission reconnection is a stretch goal. The PLAN
    wins.
25. **Shadow-ray t_max.** gap-light §2.5 uses `(1−ε)·dist`, gap-rc uses `dist`, and gap-env uses
    `+∞`. PLAN §1.3 (review R11): the distance between the offset points with one shared ε, and
    FLT_MAX for directions. The PLAN wins.
26. **USD SphereLight with normalize = false and r = 0 (treatAsPoint).** PLAN §1.2 lists only
    normalize true and false with r > 0. cyc §4.9 implies `I = i·2^e`. That value is added in
    [units-lights](#units-lights) as derived; confirm it with the E2E-USD test.
27. **USD Rect/Disk radiance.** PLAN §1.2 does not state it. The values `L_e = i·2^e` (unnormalised)
    and `i·2^e/A` (normalised) are derived from cyc §4.9. Confirm them in M2's minimal
    UsdSceneSource.
28. **Suffix-cache class field.** gap-temporal §5.6 adds a 3-bit class field. The class is
    derivable from (d, k, technique), so it is not stored ([reservoir-fields](#reservoir-fields)).
29. **Alpha/crossing RNG dims.** PLAN §1.7 lists "alpha/crossing dimensions per crossing", while
    PLAN §1.3 says MASK consumes no RNG, and BLEND is a non-goal. **Resolution:** slots 10–13 are
    reserved and unused in v1.
30. **The temporal E_{t−1} list in PLAN rule 9** does not mention the Mode-B crossing loop over
    lightsPrev or M(B), τ and α_min explicitly. Review R15 requires them. "Frame config" is read to
    include them ([temporal](#temporal)).
31. **Plan rule 13 groups n_σ with the duplication map.** n_σ belongs to the pairing textures.
    Both are kept under [dupmap](#dupmap) with a cross-link. This is a documentation grouping, not
    a maths conflict.
32. **TOST α vs suite FWER.**
    - PLAN §7.3 fixes TOST at α = 0.01 per side and separately a suite FWER
      `α_u = 1 − 0.99^{1/n_units}` with one re-run.
    - It does not say which tests α_u governs.
    - **Proposed reading:** α_u is the family level for the difference-type tests inside a unit
      (Šidák tiles, χ²_red, mean-t, KS). TOST keeps α = 0.01 per side, and the sizing rule makes it
      pass under H0.
    - To be confirmed when compare.py is written.
33. **Cycles camera example.** cyc §6's "`angle_x` = 57.265° at 16:9" is misleading (cyc-verify
    C1). Use VERTICAL fit with `angle_y` only.
34. **RR survival probability.**
    - Neither the PLAN nor Enhanced specifies q_i.
    - Code22 uses `min(1, lum(β))`; Cycles uses `min(sqrt(max_c|β_c|), 1)`.
    - Any deterministic prefix function is unbiased. **Proposed:** Cycles' form
      ([rr](#rr)). Pin it before M6.
35. **Background-pixel thr.** gap-env §3.6 says "thr undefined" for misses. Since misses are never
    accepted as partners or temporal sources, thr is never read there. Write thr = 0 and flag it,
    so no NaN enters the planes.
36. **[M4 addition, restir-api.md]** **initSeed width in the reservoir.** §17 listed `initSeed` as u32; §15 (M3c) requires 64 bits.
    **Resolution:** two words in the reservoir (restir-api.md D2).
37. **[M4 addition, restir-api.md]** **Triangle NEE endpoints.** PLAN §1.9 lists `(primId, bary)` for triangle rc vertices. For an
    NEE-sampled triangle the reservoir stores the sampler's `(u₁, u₂)` with the alias entry instead, so the endpoint is
    re-evaluated by the PT's own sampler function; BSDF-hit triangles keep `(primId, bary)`. Both are lossless in
    validation; the encodings differ by technique (restir-api.md D4).
38. **[M4 addition, restir-api.md]** **Shift status vs. "undefined/zero".** math §22 FAILED = "undefined or p̂ = 0"; gap-light §5.6
    separates undefined and zero. Resolution: one J word (FAILED) plus a detailed code word (see §20 addition).
39. **[M4 addition, restir-api.md]** **Glass and alpha scenes in M4.** PLAN §5 M4 exit lists (i)–(vi) and (xii); §7.1 puts rungs 3.9
    (glass) and 3.10 (alpha) in M6. Resolution: M4 gates (vi) Mode A and (xii) at rungs 3.1–3.2; M6's rungs add Mode B,
    the glass scene set, rough-glass reconnection and cutouts on shift segments (restir-api.md D13, §6.5).
40. **[M4 addition, restir-api.md]** **Ensemble tile size.** PLAN §3 fixes 256² member tiles; the Stage-B scenes are mostly 512².
    Resolution: the member tile is the scene's render resolution, `E·W·H ≤ 2²²`; the M4 Stage-B rungs use sequential
    independent frames, which is valid with temporal reuse off (restir-api.md D14, D15).
41. **[M4 addition, restir-api.md]** **Single-engine build.** gap-rc §8.3 recommends it for FP-BOUNDARY = 0; PLAN §5 M4 exit only requires
    FP-BOUNDARY ≤ 1e-5. Deferred unless the measured rate exceeds the limit (restir-api.md D21).
42. **[M4 addition, restir-api.md]** **Case (b)/(c) cached end terms.** PLAN §1.9 stores `rcRadiance` and `aux` for cases (b)/(c).
    M4 shifts re-evaluate these end terms at the copied x_{d−1} from the stored endpoint (bit-identical within a frame)
    and use the stored values only as a debug cross-check (restir-api.md D6).
43. **[M5 addition, restir-temporal-api.md]** **Scope of the inverse refresh.** PLAN §3 step 3 refreshes "canonicals under S_{t−1}"; plan-review offers
    "selected canonicals only". Resolution: the refresh pass runs over the canonicals whose π_p is needed (queue Q_i,
    after selection); the values are exact either way, only the work differs.
44. **[M5 addition, restir-temporal-api.md]** **Where the N1 end term is refreshed.** plan-review WGPU-2 puts N1 end terms in the shift. The prefix-dependent
    factors stay in the shift; the prefix-independent visibility V(x_k, Φ_s) is traced by the refresh pass and handed to
    the shift, so the shift needs no "moved" logic.
45. **[M5 addition, restir-temporal-api.md]** **N4 needs RIS-NEE light tiles** (M6). M5 plants a synthetic fresh RIS re-draw over 8 alias candidates drawn with
    frame-t random numbers (test only).
46. **[M5 addition, restir-temporal-api.md]** **U8 plants 7, 8, 10** need Mode B or RIS-NEE tiles and move to M6; plant 5 is replaced by its temporal analogue
    (spot axis of frame t−1).
47. **[M5 addition, restir-temporal-api.md]** **Dynamic scene scripts.** The M3a ix-a…g packages store 7 test frames and model light removal as power 0;
    gap-temporal §9.1 asks for dense sequences with true add/remove. M5 adds new `ixs_*` sequence packages; the M3a
    packages stay as Stage-A regressions.
48. **[M5 addition, restir-temporal-api.md]** **δ per rung.** Static temporal rungs 3.3–3.5 use the Stage-B δ (0.2% global, 1% per 32² tile); dynamic 3.6
    uses the PLAN §7.3 dynamic δ (0.2% global, 2% per 64² tile, 3% per mask region).
49. **[M5 addition, restir-temporal-api.md]** **Rung 3.5 "time-average"** is defined as the per-chain mean of frames 32…287 with chains as replicates (valid
    because each frame is unbiased); it tests long-run convergence, while 3.3/3.4 test individual frames.
50. **[M5 addition, restir-temporal-api.md]** **Motion vectors.** PLAN §3 step 1 lists motion vectors; the temporal step back-projects positions rebuilt from
    ids with the previous camera, so no motion-vector texture is needed in M5; dual MVs move to M6.
51. **[M5 addition, restir-temporal-api.md]** **Talbot and E2 as gating modes.** gap-temporal §9.1 requires every exact mode to pass; M5 gates Talbot and E2 on
    two light-change sequences each and does not implement E3.
52. **[M5 addition, restir-temporal-api.md]** **One light commit per frame.** The M3a app flips the light slot on every staged edit (env parameter edits
    flip it again), so `prev` could be two frames old. M5 stages edits and commits once per rendered frame.
