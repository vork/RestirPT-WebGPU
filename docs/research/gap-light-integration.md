# Gap-fill: lights inside ReSTIR PT (Enhanced unified DI+GI) — point, spot, area, emissive triangles, sun

> **Location note.** The orchestrator asked for this report at
> `/private/tmp/claude-502/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-ea73-491f-b7b7-6d0aab1ffce1/scratchpad/research/gap-light-integration-restir-pt.md`.
> This session ran in **plan mode**, where the only writable file is this plan file, so the report is here. Copy it verbatim to the intended path. Nothing else was written, nothing was cloned. Blender was run once headless for read-only RNA introspection (nothing saved).
>
> **Tags.** **[SOURCE]** = stated by a cited source (file:line, page, section). **[INFERENCE]** = my derivation or recommendation. **[UNVERIFIED]** = could not confirm; check before relying on it.
>
> **Sources read for this report (primary):**
> - Cycles 5.1.2 kernel shipped in the app: `/Applications/Blender.app/Contents/Resources/5.1/scripts/addons_core/cycles/source/kernel/` — `integrator/shade_light.h`, `integrator/intersect_closest.h`, `integrator/intersect_shadow.h`, `integrator/path_state.h`, `integrator/shade_surface.h`, `integrator/surface_shader.h`, `light/light.h`, `light/sample.h`, `light/area.h`, `light/point.h`, `light/spot.h`, `light/distribution.h`, `light/triangle.h`, `light/common.h`. Below, `K/` abbreviates that kernel directory.
> - Blender 5.1.2 headless RNA introspection (this session): `scene.cycles.transparent_max_bounces` default **8**, hard range **[0, 1024]**; new AREA light: `cycles.use_multiple_importance_sampling = True`, `cycles.max_bounces = 1024`, `spread = π`, `shape = 'SQUARE'`, `size = 0.25`, `energy = 10`, `normalize = True`; new light *object*: `visible_camera = False`, `visible_diffuse = visible_glossy = visible_shadow = True`; new POINT light: `shadow_soft_size = 0.0`, `use_soft_falloff = True`, MIS True.
> - Enhanced paper page images `scratchpad/pages/enhanced/p-03.png`, `p-04.png` (§2.3, Eq. 2), `p-10.png` (§6.1), `p-11.png` (§6.2.3, §6.2.4, §6.3).
> - Enhanced supplemental text (harness-cached extraction of the official PDF): `/Users/mark.boss/.claude/projects/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-ea73-491f-b7b7-6d0aab1ffce1/tool-results/bkusnhdet.txt`, §1 (lines 12–93), **§5 (lines 326–348)**, §6 (lines 349–381), §8 Alg. 1 (lines 393–413).
> - `github.com/EvanLuo42/ReSTIR-PT-Enhanced` (BSD-3), `Source/RenderPasses/ReSTIRPTPass/`: `Common/HybridShift.slang`, `Common/Lights.slang`, `Path/PathSampling.slang` (read through WebFetch; line numbers are approximate as reported by the fetch tool **[UNVERIFIED line numbers]**).
> - `github.com/NVIDIA-RTX/RTXDI-Library` `Include/Rtxdi/PT/HybridShift.hlsli` — **proprietary; semantics described in my own words only, no code copied.**
> - Existing reports (all under `/Users/mark.boss/.claude/plans/do-a-deep-dive-shimmering-ritchie-agent-`): gris-math `a7f08dec9968f0324` (+verify `a147516e5d2fb9558`), enhanced-paper `a91ab1d861819b276` (+verify `acb243dbea3ce2a29`), reference-code `a3b89657b22d15f34`, restir-practice `a668fe13fb61070e8`, scene-io `a597df4a1dd101994`, cycles-conventions `a2d6cc1a462b20a80` (+verify `ac3ddf11ffb75ef90`), webgpu-platform `a64ceeae6903be301`, validation-harness `a26c6df2c3f5e4af0`, critique `a40267d04641507fa`.

---

## 0. TL;DR — decisions

1. **Cycles semantics, settled from the 5.1.2 kernel [SOURCE].** A camera or BSDF ray that hits an analytic light (area; point/spot only if radius > 0) adds `emission × MIS` and **the same ray continues** (`tmin` advanced past the hit, origin unchanged). Only `transparent_bounce` is incremented (path terminated when it reaches `transparent_max_bounces`, default 8). `bounce`, the RNG offset, the path flags, `mis_ray_pdf` and the MIS origin are all untouched, and no Russian roulette happens at the light. MIS weight: **1 for camera rays** (`PATH_RAY_MIS_SKIP`), **power heuristic** `p_bsdf²/(p_bsdf² + p_light²)` for BSDF rays. **Shadow rays never test analytic lights** (they are not BVH primitives). Point/spot with r = 0 and sun with angle 0 are never hit by anything and get NEE weight exactly 1. Emissive triangles are ordinary opaque, two-sided, occluding geometry.
2. **Path space [INFERENCE].** Use **triangle-only visibility `V_tri`** for every segment (camera, BSDF, shadow, reconnection). Analytic lights are pure emitters that never occlude. A BSDF ray from `x_{d−1}` then produces up to `1 + N_cross` light-ending candidates of the *same* length `d`: one per front-facing analytic light it crosses before the first triangle (Mode B only) plus the emissive-triangle hit. Each is a separate path-tree candidate. The continuing path's vertices are unchanged, and light crossings create no vertex and consume no bounce and no random number.
3. **Two modes; recommended default = Mode A [INFERENCE].**
   - **Mode A (default):** analytic lights are NEE-only. Blender side: `light.cycles.use_multiple_importance_sampling = False` and `visible_camera = False`. ω_NEE = 1 for every analytic light. There are no crossings, no transparent-bounce accounting, and no rc case (d) or replay case for analytic lights. Emissive triangles keep full NEE/BSDF MIS.
   - **Mode B (optional, later):** area lights hittable by BSDF rays with pass-through. Blender: MIS on, `transparent_max_bounces = 1024` (the RNA hard max) so the Cycles cap is negligible. Requires light-labelled candidates `(d, BSDF_ANALYTIC, lightId)`.
   - Both modes have **identical expectation** (proof in §2.7), so Mode B is a pure variance option and "Mode A ≡ Mode B" is itself a unit test.
4. **Per-light formulas in one measure [INFERENCE, Cycles constants SOURCE].** Put every light sample in a single product measure μ: area measure on area lights and emissive triangles, a unit atom on delta lights. Then:
   - the source density is `q = P(L)/A_L` for area/triangle lights and `q = P(L)` for delta lights;
   - the light term is `Λ = L_e|cosθ_z|/r²`, `I(ω)/r²` or `E_sun` respectively;
   - the NEE PSS integrand is `F = T · ω1 · f_cos · Λ · V / q`.

   RIS-NEE over mixed delta/area candidate lists is exact if each candidate's ratio `r_i = p̂_i/q_i` is formed in μ. The Enhanced UCW `W = W^RIS·p1` equals `(1/M)Σ r_i / r_sel`, which is measure-free.
5. **MIS [SOURCE Enhanced S-§5 + INFERENCE].** `ω1 = M p1/(M p1 + p2)` and `ω2 = p2/(M p1 + p2)`. `p1 = q · r²/|cosθ_z|` is in solid angle at `x_{d−1}`, `p2` is the **marginal** BSDF pdf, and `M = M(index of x_{d−1})`. **Delta lights and Mode-A analytic lights: ω1 ≡ 1, hard-coded by flag, never by `p2 = 0` arithmetic.** Cycles' power heuristic is also a partition of unity, so it changes only variance.
6. **Selection pmfs.** Use a global per-frame power alias table. Light tiles draw i.i.d. from it, so every candidate's *marginal* pmf is `pmf[L]`, independent of the shading point and the pixel. Always use the marginal, never the tile-conditional multiplicity.
7. **Reservoir [INFERENCE on S-Alg.1].** 64 B. Repurpose `rcVertexRandomSeed` as `endpointId`, which a counter-based RNG makes possible. Light samples are stored **light-locally**: `(lightId, u, v)` for analytic lights, `(instanceId, primId, bary)` for triangles. Everything light-dependent at the offset shading point (Λ, spot `S`, one-sidedness, `p1`, `ω`) is **recomputed**, not cached. `rcVertexRadiance` holds the MIS-and-pdf-free `Λ`/`L_e` in cases (b) and (c) and the full suffix (MIS included) in case (e).
8. **Jacobians [INFERENCE from Enhanced Eq. 2 + S-§5].** With area-uniform light sampling and a frame-global pmf, the light-vertex PSS Jacobian is **1** (spatial) or `pmf_t(L)/pmf_{t−1}(L)` (temporal). With solid-angle light sampling it is `p_A(z|y)/p_A(z|x)`. Delta lights and the sun: **1, never `t_x²/t_y²`**. RTXDI's "solid-angle-pdf ratio" is the same Jacobian written in solid-angle measure (§5.7).
9. **Length-1 paths are outside the reservoir.** The Enhanced reservoir holds `d ≥ 2` (§6.1: "shorter direct lighting path (d = 2)"). EvanLuo42 outputs `primaryEmission` separately. Pixel = `Σ_{visible lights crossed} L_e` (weight 1) + `L_e(x1)` (weight 1) + `F·W`.
10. **Bias traps catalogued (§9).** Known: C1, C2, C3, X14. New:
    - mixed occluding semantics (BSDF rays stop at lights while shadow rays ignore them);
    - Cycles point-light pseudo-pdf `t²` leaking into a ratio;
    - tile-conditional pdfs;
    - a stale pmf across frames;
    - dropping failed neighbours from the MIS normaliser;
    - the Cycles transparent cap in Mode B;
    - unlabelled multi-crossing candidates;
    - a V-buffer that treats lights as occluders;
    - measure-inconsistent RIS ratios.

---

## 1. Cycles 5.1.2: what happens when a ray hits an analytic light

### 1.1 Which rays can hit which emitters [SOURCE]

| Emitter | Camera ray | BSDF (indirect) ray | Shadow / NEE ray | NEE MIS weight |
|---|---|---|---|---|
| Point, radius 0 | never: `point_light_intersect` returns false for `radius == 0` (`K/light/point.h:138-141`) | never | never tested | **1**: the light shader lacks `SHADER_USE_MIS`, so the BSDF pdf is forced to 0 (`K/integrator/surface_shader.h:377-381`) and `power_heuristic(p_nee, 0) = 1` |
| Spot, radius 0 | never (`spot_light_intersect` → `point_light_intersect`, `K/light/spot.h:206-216`) | never | never | **1** |
| Point/spot r > 0 | only if not `SHADER_EXCLUDE_CAMERA` (object `visible_camera`) (`K/light/light.h:278-282`) | only if `SHADER_USE_MIS` (`light.h:283-286`) | never | power heuristic |
| Area (rect/square/disk/ellipse) | only if `visible_camera` (default **False**, introspected) | only if `SHADER_USE_MIS` (default MIS **True**; also requires spread > 0, cycles-verify §5); one-sided `dot(D, Ng) ≥ 0 → no hit` (`K/light/area.h:405-408`) | never | power heuristic, or 1 if MIS off |
| Sun, angle 0 | never; distant lights are skipped for the main path in `lights_intersect_impl` (`light.h:347-354`) | never | ray to ∞ against geometry only | 1 |
| Emissive triangle | opaque surface, always | opaque surface, always | **occludes** | power heuristic (`light_sample_mis_weight_forward_surface`, `K/light/sample.h:464-506`) |

Lights are intersected in a separate loop over all lamps, `lights_intersect` (`K/light/light.h:258-421`, commented "Lights intersection for the main path"). It is called only from `integrator_intersect_closest` when `use_light_mis` is set (`K/integrator/intersect_closest.h:425-433`). The shadow-ray kernel `integrator_intersect_shadow` (`K/integrator/intersect_shadow.h:149-183`) calls only `scene_intersect_shadow` / `scene_intersect_shadow_all`, i.e. the geometry BVH, never `lights_intersect`. **Shadow rays therefore ignore all analytic lights**, and so do NEE shadow rays toward any light. (The shadow-linking special kernel is out of scope.)

### 1.2 The MIS weight applied to the emission [SOURCE]

`integrate_light_forward` (`K/integrator/shade_light.h:19-71`):
- `eval = light_sample_shader_eval_forward(...) · light_eval.eval_fac` (`:55-57`). `light_sample_shader_eval_forward` multiplies the emission shader by `klight->strength` (`K/light/sample.h:97-101`). `eval_fac` is the per-type normalisation, e.g. `(1/π)·invarea` for area lights (`K/light/area.h:313`).
- `mis_weight = light_sample_mis_weight_forward_lamp(kg, state, path_flag, isect.prim, light_eval.pdf, ray_P)` (`shade_light.h:63-64`), defined at `K/light/sample.h:508-543`:
  - `if (path_flag & PATH_RAY_MIS_SKIP) return 1`. Camera rays start with `PATH_RAY_CAMERA | PATH_RAY_MIS_SKIP` and `mis_ray_pdf = 0` (`K/integrator/path_state.h:59-61`). A singular (delta) BSDF bounce also sets `PATH_RAY_MIS_SKIP` (`path_state.h:216`). A regular bounce clears it (`:147`).
  - Otherwise `pdf = light_eval.pdf × selection pdf`. The selection pdf is the light tree's, or the flat constant `distribution_pdf_lights` (`K/light/distribution.h:55-58`, the same value for every lamp).
  - `power_heuristic(mis_ray_pdf, pdf)` (`sample.h:322-337`).
- `mis_ray_pdf` and `mis_origin_n` are written only after a non-transparent BSDF bounce (`K/integrator/shade_surface.h:578-582`). They are the one-sample *mixture* pdf of the **last real scattering vertex**. A light pass-through does not change them.
- `light_eval.pdf` is the solid-angle pdf of Cycles' own NEE sampler, evaluated from `ray_P`, the **origin of the ray**, i.e. the last scattering vertex:
  - rect: spherical-rectangle `1/S` (`area.h:19-102, :445-462`);
  - disk: `4/(π len_u len_v) · t²/cos` (`area.h:302, :321-323`);
  - soft-falloff disk / sphere point lights: `point.h:154-179`.

So **camera ray → weight 1; BSDF ray → power heuristic between the previous vertex's BSDF mixture pdf and the light's NEE pdf (selection × solid angle) seen from that same vertex.**

### 1.3 Continuation, counters, RR, RNG [SOURCE]

`integrator_shade_light_forward` (`shade_light.h:74-104`):
- `ray.tmin = intersection_t_offset(isect.t)` (`:36`). The ray **origin and direction are unchanged**, so the next `intersect_closest` continues the *same* ray past the light.
- `transparent_bounce += 1` (`:88-89`). If `transparent_bounce >= transparent_max_bounce` the **path is terminated** (`:91-95`). The emission of that last light hit was already added. Otherwise it continues with `INTERSECT_CLOSEST` (`:97-99`).
- `path_state_next` is **not** called, so `bounce` is not incremented, the path flags (`PATH_RAY_MIS_SKIP`, `TERMINATE_AFTER_TRANSPARENT`, …) are unchanged, and `rng_offset` is **not** advanced (`path_state.h:110-235`, where only `path_state_next` advances `rng_offset`).
- Russian roulette is evaluated only for **surface** hits (`integrator_intersect_terminate`, `intersect_closest.h:29-87`, called at `:261` for non-lamp hits). A lamp hit goes straight to `SHADE_LIGHT_FORWARD` (`:253-255`), so no RR at the light.
- The code comment at `shade_light.h:82-87` says counting the pass as a transparent bounce is a workaround against infinite loops, and that lights are "interpreted as transparent surfaces".
- `light_eval.eval_fac == 0` (e.g. outside a spread cone) → `return` before shading, but the transparent bounce is still counted, because `integrate_light_forward` returns into `integrator_shade_light_forward`, which increments. Back-facing area lights are never intersected at all (`area.h:405-408`), so they cost no transparent bounce.

**Effect on the next real vertex [SOURCE + INFERENCE].** NEE at the next scattering vertex is unaffected: same RNG dimensions, same flags. Its shadow ray ignores lights. The NEE shadow state copies `transparent_bounce` (`shade_surface.h:249-250`), which only limits the number of *transparent-surface* shadow hits (`intersect_shadow.h:55-62`). That is irrelevant for opaque scenes. After a pass-through, the MIS weight of an emissive triangle hit behind the light uses the unchanged `mis_ray_pdf`, and `sd->ray_length = isect.t` measured from the unchanged origin (`sample.h:464-506`). It is exactly the weight it would get with no light in between.

### 1.4 NEE side [SOURCE]

- Area NEE is one-sided: `if (dot(ls->P − P, ls->Ng) > 0) return false` (`area.h:337-341`). `light_pdf_area_to_solid_angle` returns 0 for `cos ≤ 0` (`K/light/common.h:62-70`).
- Point r = 0: `ls->P = co`, `pdf = invarea(=1) · t²/1 = t²` (`point.h:65-77`); `eval_fac = 1/(4π)`. Per-sample contribution `(strength/(4π)) · f_cos / t² / P_sel`, with MIS 1.
- Spot r = 0: same, times `smoothstepf((local.z − cos_half_spot_angle)·spot_smooth)`, with `local = normalize(itfm · (−ls.D))` and `local.z` negated (`spot.h:15-31, :122-142`).
- NEE contribution: `bsdf_eval × light_shader_eval × eval_fac / ls.pdf × mis_weight`, with `mis_weight = power_heuristic(ls.pdf, bsdf_pdf)` (`shade_surface.h:402-404`, `sample.h:339-357`). `ls.pdf` already includes the selection pdf (`light.h:249`).

### 1.5 What this means for the integral Cycles computes [INFERENCE]

Let `V_tri(a,b)` be binary visibility against scene triangles only. Cycles' estimator (MIS on, no cap reached) is an unbiased estimator of

```
I = Σ_{d≥1} ∫ f(x̄) dx̄ ,   f(x̄) = W_e(x1→x0) · Π_{i=1}^{d−1} f_s(x_{i−1}→x_i→x_{i+1}) · Π_{i=0}^{d−1} G(x_i↔x_{i+1}) V_tri(x_i↔x_{i+1}) · L_e(x_d→x_{d−1})
```

Here `x_1..x_{d−1}` lie on triangles, and `x_d` lies on an emissive triangle (two-sided `L_e`), on an analytic area light (one-sided `L_e`), or is a delta light (`G·L_e` replaced by `I(ω)/r²` with the counting measure, see §3).

The point is that **`V_tri` ignores analytic lights on every segment**, in both the NEE and the BSDF technique, so the MIS weights of the two techniques are defined on the same support and form a partition of unity.

The only deviation from this integral is the **transparent cap**: a path whose rays (camera and BSDF segments; NEE rays excluded) cumulatively cross `transparent_max_bounces` lights is truncated after the crossing that reaches the cap. At the default of 8 this is a real, if tiny, truncation. At 1024 it is negligible.

---

## 2. ReSTIR PT path representation for analytic lights

### 2.1 Path-tree candidates with pass-through lights [INFERENCE]

**Indexing.** Use the Enhanced convention: `x0` camera, `x1` primary hit, …, `x_d` emitter. `d` is the index of the light vertex (the number of segments). The scattering vertices are `x1..x_{d−1}`, and the reservoir holds `d ≥ 2` (§6).

At each scattering vertex `x_B` (B = 1 … B_max) the path tree emits:

1. **One NEE candidate** `(d = B+1, tech = NEE, endpoint = sampled light sample Y)`, with Y from RIS-NEE over `M(B)` candidates (§3.8).
2. **A BSDF continuation** `ω_B` (lobe `ℓ_B`). Trace `hit = closest triangle along (x_B, ω_B)`. Then:
   - **Mode B only:** for every analytic area light `L` whose plane is crossed front-facing at `t_L < hit.t` (or `t_L < ∞` on a miss), emit a candidate `(d = B+1, tech = BSDF_ANALYTIC, endpoint = (L, u_L, v_L))`.
   - If `hit` is on an emissive triangle, emit a candidate `(d = B+1, tech = BSDF_TRI, endpoint = hit)`.
   - The continuing path is `x_{B+1} = hit` (unchanged by any crossing).

Every candidate is streamed into the tree reservoir with `m = 1`, because the domains are disjoint by `(d, tech, endpoint label)`, exactly as in Falcor's path tree (gris-math §5.4). The tree reservoir is finalised with `M = 1`.

**Why a label per light and not per crossing index.**
- A single BSDF sample `ū` generates several light-ending paths of the same length.
- If the domain were only `(d, BSDF)`, the PSS integrand would be the *sum* over all crossings. A stored sample would then not be a single path, and the rc vertex could not be a light point.
- Splitting the technique index by `lightId` makes each domain `Ω_{d,BSDF,L}` contain single paths.
- Its PSS integrand is well defined:
  `F_{d,L}(ū) = ω2 · f/p · 1[ray(x_{d−1}, ω_{d−1}(ū)) crosses L front-facing before the first triangle] · L_e`.
- This is a legal refinement of GRIS Eq. 51's technique index (gris-math §5.2), and the canonical path tracer covers every `Ω_{d,BSDF,L}`.
- Planar rect and disk lights are crossed at most once per ray.
- Sphere lights (r > 0, `use_soft_falloff = False`) can be crossed twice by one ray. Cycles would then add emission at the exit point as well: the sphere pdf branch does not test the side [SOURCE `point.h:154-179`; the physical interpretation is my INFERENCE]. **Do not support r > 0 spheres in Mode B**, or label by `(L, crossingIndex)`.

### 2.2 Path-length accounting and `pathFlags` [INFERENCE]

- A crossing is **not a vertex**, consumes **no bounce** and **no random number**, and triggers **no RR** (mirrors §1.3). `d` counts scattering vertices plus 1.
- Blender `max_bounces = N` allows at most `N+1` scattering vertices (cycles-conventions §1 item 7, verified), so `d ≤ N+2`. A 4-bit `d` field (≤ 15) supports `N ≤ 13`.
- Proposed `pathFlags` (u32):

| Bits | Field | Notes |
|---|---|---|
| 0–7 | `M` confidence (u8) | S-Alg.1 packs M in 8 bits. Validation builds keep an f32 copy elsewhere (critique X8) |
| 8–11 | `d` (2…15) | index of the light vertex |
| 12–15 | `k` rc index (0 = none, else 2…d) | k ≥ 2, since x1 is fixed by the pixel |
| 16–17 | technique: 0 NEE, 1 BSDF_TRI, 2 BSDF_ANALYTIC (Mode B), 3 reserved (env) | |
| 18–20 | endpoint type: 0 tri, 1 point, 2 spot, 3 rect, 4 disk/ellipse, 5 sun, 6 (reserved: sphere), 7 (env) | |
| 21 | `isDelta` | explicit bit; never infer delta from `pdf == 0` (gris-verify C1) |
| 22–23 | lobe ℓ_{k−1} (lobe that sampled ω_{k−1}; 3 = "light-sampled / NEE") | |
| 24–25 | lobe ℓ_k (lobe that sampled ω_k at the rc vertex; 3 = NEE all-lobes or n/a) | |
| 26–31 | reserved (frame parity for the light-id map, RR bookkeeping) | |

### 2.3 May a pass-through light hit be the reconnection vertex? [INFERENCE]

**Yes for the light-ending candidate itself (case (d) in §5). No for the continuing path**, whose vertices never include the light point.

**Bijectivity argument.** Let `x̄ = (x_0 … x_{d−1}, z)` with `z = (L, u, v)` in `Ω_{d,BSDF,L}` and rc index `k = d`. Define `T(x̄) = ȳ = (y_0 … y_{d−1}, z)`, where `y_1..y_{d−1}` come from random replay and `z` is copied in light-local coordinates. The shift is well defined and invertible because:
1. crossings of *other* lights along any segment change neither vertices, random numbers nor counters, so the replayed prefix is a function of `(ū, y_1)` alone;
2. the rc rule ("first pair that meets the criteria, else none") is a deterministic function of the path, evaluated with identical code on both sides;
3. the label `(d, BSDF_ANALYTIC, L)` is preserved;
4. `z` is copied exactly, so `T⁻¹(ȳ)` copies it back.

`T` is undefined exactly when the offset violates the rc rule (an earlier pair qualifies, or `(y_{d−1}, z)` fails the criterion), and those tests are symmetric.

**Consequences of the definition.** `ȳ` is a valid path of `Ω_{d,BSDF,L}` in the destination pixel iff:
- `V_tri(y_{d−1}, z) = 1`;
- `L` is front-facing from `y_{d−1}`: `dot(y_{d−1} − z, n_L) > 0`;
- `p_{ℓ}^y(ω') > 0`.

Otherwise the shift is defined with zero contribution.

**Criteria at a light vertex** (GRIS §7.5, "all light vertices are rough", via gris-math §6.5; Enhanced footnote 6, "for diffuse or emissive x_k … the inverse ray footprint test is skipped", `restirpt_enhanced_2026.txt:431`):
- the roughness test is applied only at `x_{k−1}` (`α_{x_{d−1}}` of lobe ℓ_{d−1} ≥ α_min);
- the ray footprint `‖z − x_{d−1}‖² / (p_{d−1}(ω_{d−1})·|cos θ_z|) ≥ thr` uses the geometric normal of the emitter (`n_L`, or the triangle `n_g`).

### 2.4 Must replay reproduce pass-throughs? [INFERENCE]

- **Vertex sequence, RNG, counters:** no action. Crossings are invisible to them by construction (§2.2). This holds only if the replay kernel does not count transparent bounces or apply a cap. See the cap discussion below.
- **Full replay of a BSDF_ANALYTIC candidate with no rc vertex (`k = 0`):**
  - Replay reaches `y_{d−1}`, samples ω with the same random numbers and lobe, traces the first triangle, then intersects **only light L**.
  - `F = T_y · L_e · ω2^y` if L is crossed front-facing before the triangle, else `F = 0`. This is defined with zero contribution: random replay is the identity in PSS.
  - Invertibility: if `(y_{d−1}, z')` would qualify as an rc pair → **undefined**. This is the same rule as Falcor's "emitter-as-rc during replay → non-invertible" (reference-code §1.6.3).
- **NEE candidates are never replayed.** Enhanced §6.2.3 forces reconnection to the NEE light vertex (p-11: "we force reconnection to such vertices if no earlier reconnection is found"). No shift ever re-runs light selection. This matters because RIS-NEE over per-frame light tiles is not reproducible.
- **Cycles cap.** If you insist on emulating `transparent_max_bounces = 8`, every candidate would need its *cumulative* crossing count over all its BSDF segments and the camera segment. Reconnection changes one segment, which could change that count, and the count would have to enter F. **Recommendation:** never emulate the cap. Set Blender `transparent_max_bounces = 1024` (the RNA hard max) for Mode-B references, and use Mode A by default, where no crossings happen.

### 2.5 Visibility for reconnection and NEE ignores analytic lights [INFERENCE, required]

All visibility queries test **triangles only**: shadow rays, reconnection rays `y_{k−1} → x_k`, primary V-buffer rays, and temporal refresh rays. Emissive triangles remain occluders.
- For a triangle endpoint, trace to the point with `tmax = (1−ε)·dist` and exclude the endpoint primitive (Cycles uses `ray.self.light_prim`, `K/light/sample.h:292-295`).
- **V-buffer consequence:** the primary hit `x1` is the first *triangle*. Camera-visible analytic lights are handled by a separate crossing loop in the resolve pass (§6).

### 2.6 A trap: occluding for BSDF rays but not for shadow rays [INFERENCE]

webgpu-platform §4.5 proposed putting area lights as emissive triangles in the BVH, which makes them occluding for BSDF rays. If shadow rays still ignored them (to match Cycles), then:
- a path `x_{d−1} → surface behind the light` is reachable by NEE (`V = 1`) but not by BSDF sampling (blocked);
- yet `ω_NEE = p1/(p1+p2) < 1` assumes both techniques can sample it.

The MIS partition of unity then fails and the image darkens behind lights. Either make lights non-occluding for **all** rays (Cycles; this report) or occluding for **all** rays (a different scene: Blender emissive quads, not area lights). Never mix. This resolves critique X2 in favour of cycles-conventions §2.8.

### 2.7 Mode A vs Mode B [INFERENCE]

| | **Mode A (NEE-only analytic lights)** | **Mode B (pass-through, MIS)** |
|---|---|---|
| Blender reference | `light.cycles.use_multiple_importance_sampling = False` on every analytic light; `visible_camera = False` | MIS True (default); `scene.cycles.transparent_max_bounces = 1024` |
| Cycles behaviour | BSDF rays skip the light (`light.h:283-286`); NEE weight 1 (`surface_shader.h:377-381`); no transparent bounces | pass-through, power heuristic, transparent cap (negligible at 1024) |
| Our BSDF rays | never test analytic lights | crossing loop over area lights per BSDF segment, O(#area lights) |
| ω_NEE for analytic lights | 1 | `M p1/(M p1 + p2)` |
| Candidate techniques | NEE, BSDF_TRI | NEE, BSDF_TRI, BSDF_ANALYTIC(L) |
| rc cases needing analytic-light code | (a), (b) (NEE endings only) | (a)–(d) |
| Replay | never touches analytic lights | must evaluate crossings of L on the final segment (§2.4) |
| Temporal refresh when lights move | NEE endpoint only | NEE endpoint + crossings of BSDF segments |
| Variance | NEE-only on glossy receivers of large area lights is noisy (RIS-NEE + reuse mitigates) | lower on glossy |
| Matches Falcor / EvanLuo42 | yes: analytic lights are NEE-only there (EvanLuo42 `PathSampling.slang`: "only emissive triangles and environment are hit via BSDF rays", WebFetch) | no public reference |

**Same expectation (proof).**
- Mode A: `I_A = Σ ∫ 1·f` over light-ending paths, because NEE alone covers every analytic light point with `pmf[L]·p_A > 0`.
- Mode B: `I_B = Σ ∫ (ω1+ω2)·f = Σ ∫ f`, because `ω1+ω2 = 1` pointwise (§3.7).
- The integrands are identical: both use `V_tri` and the same `L_e`. So `I_A = I_B` exactly, given the same camera visibility and no cap.
- Camera visibility enters both identically: weight 1, outside the reservoir.

**Recommendation.** Default **Mode A** for v1 and all validation gates. Add Mode B later as a quality option, and gate it with the invariance test "our Mode A ≡ our Mode B" (U9 in §8) before comparing against Cycles-MIS-on references. Emissive triangles (glTF/USD emissive meshes) always use two-technique MIS in both modes.

---

## 3. Per-light-type sampling, emission, pdfs and MIS

### 3.1 Notation and the unified measure μ [INFERENCE]

- Shading vertex `x = x_{d−1}`, with geometric normal `n_g`, shading normal `n_s`, and outgoing direction `ω_o` (toward the previous vertex).
- Light sample `Y = (L, z)`; `r = ‖z − x‖`, `ω_L = (z − x)/r`.
- `f_cos(ω_o, ω) = f_s(ω_o, ω)·|n_s·ω|`. This is Cycles' "eval" convention, which includes the cosine (cycles-conventions §0).
- `P(L)` = per-frame light-selection pmf (power alias table; §3.9).
- `Φ_L = color · energy · 2^exposure` (Cycles `strength`, `blender/light.cpp:80`, cycles-conventions §2.1). The Blender unit contract of scene-io §1 applies (glTF lights ÷683; emission ×1).
- Light frame (Cycles `scene/light.cpp:1211-1215` via scene-io §2.1): emission axis `a_L = −Z_obj`, `axis_u = X_obj`, `axis_v = Y_obj`, centre `c`. Keep light objects **unscaled** (the JSON contract has no scale).

**Measure.** Let
```
μ = Σ_{L ∈ area ∪ tri} δ_L ⊗ A_L   +   Σ_{L ∈ delta} δ_{(L, c_L)}
```
Area measure on each area light or emissive triangle; one unit atom per point/spot position and per sun direction. With respect to μ, the direct-light integrand at `x` is
```
g(L,z) = f_cos(ω_o, ω_L) · V_tri(x,z) · Λ_L(z; x)
Λ_L(z; x) = L_e(z→x) · |cosθ_z| / r²     (area light, emissive triangle; cosθ_z = n_z·(x−z)/r)
          = I_L(ω_{c→x}) / r²              (point, spot)
          = E_L                             (sun: z is the direction s_L; r, cos irrelevant)
```
and `∫ g dμ` is exactly Cycles' direct radiance at `x` (one bounce of §1.5). The **source density** of a single-sample NEE in μ is
```
q(L,z) = P(L) · p_A(z | L)   (area/tri; p_A = 1/A_L for area-uniform sampling)
       = P(L)                 (delta)
```
The solid-angle NEE pdf (continuous lights only) is `p1 = q · r²/|cosθ_z|`.

### 3.2 Point light, r = 0 (delta position)

- **Sampling [INFERENCE]:** pick L by pmf; `z = c_L`. No positional random numbers, but consume the fixed per-bounce light dimensions anyway (validation-harness T1 layout).
- **Emission [SOURCE Cycles]:** `I = Φ/(4π)` W/sr: `eval_fac = invarea·(1/π)` with `area = 4` for r = 0 (cycles-conventions §2.1–2.2, verified; `K/light/point.h:65-77`, where `pdf = t²`).
- **Source density:** `q = P(L)` (atom). **Λ = I/r².**
- **NEE PSS integrand:** `F = T · f_cos(ω_o, ω_L) · (Φ/(4π r²)) · V / P(L)`, with **ω1 ≡ 1**.
- **BSDF hit:** impossible. There is no ω2 term and no BSDF_ANALYTIC candidate.
- **Cycles cross-check:** `(strength/(4π))/(t²·P_sel) · f_cos · mis(=1)` (`shade_surface.h:402-404`). Identical.

### 3.3 Spot light, r = 0 (delta position)

- As for the point light, with **I(ω) = Φ/(4π) · S(cosθ')**:
  ```
  cosθ' = dot( normalize(x − c), a_L )                  // a_L = −Z_obj; with unscaled lights this equals Cycles' spot_light_to_local z
  cosH  = cos(spot_size/2)
  S(t)  = smoothstep01( (t − cosH) / ((1 − cosH)·spot_blend) ),  smoothstep01(f) = 0 (f≤0), 1 (f≥1), 3f²−2f³
  spot_blend == 0 → S(t) = (t > cosH) ? 1 : 0         // guard: Cycles computes (t−cosH)·∞ (spot.h:28-31)
  ```
  [SOURCE `K/light/spot.h:15-31`; `spot_smooth = 1/((1−cosH)·blend)` per cycles-conventions §2.3, verified.] Power is **not** renormalised by the cone.
- `Λ = I(ω_{c→x})/r²`, `q = P(L)`, ω1 ≡ 1.
- **At an offset shading point `y`, S must be evaluated with `normalize(y − c)`** (§5).

### 3.4 Rect / square / disk / ellipse area light, spread = 180°

- **Emission [SOURCE]:** `L_e = Φ/(π·A)`, with `A = len_u·len_v` (rect/square) or `(π/4)·len_u·len_v` (disk/ellipse); `size` is the full edge length or diameter (`area.h:313`; cycles-conventions §2.4). One-sided: `L_e(z→x) = 0` unless `dot(x − z, a_L) > 0` (`area.h:337-341, :405-408`). At spread = π the spread factor is 1 (cycles-verify §4.4).
- **Sampling (recommended) [INFERENCE]:** area-uniform in light-local `(u,v) ∈ [0,1)²`:
  - rect: `z = c + (u−½)len_u·a_u + (v−½)len_v·a_v`;
  - disk: concentric map of `(u,v)` to the unit disk `(δx, δy)`, then `z = c + ½len_u·δx·a_u + ½len_v·δy·a_v`.

  Then `p_A = 1/A` and `q = P(L)/A`. Because `(u,v)` are both the PSS coordinates and the light-local surface parameters, **copying `(L,u,v)` is the identity in PSS** (§5.7).
- **Alternative:** spherical-rectangle solid-angle sampling (Cycles `area.h:19-102`, Ureña) is lower-variance near the light. Store the surface point `(u,v)` (not the sampler's random numbers). The light-vertex Jacobian then becomes `p_A(z|y)/p_A(z|x)` with `p_A(z|•) = P(L)·p_σ(z|•)·|cosθ_z^•|/r_•²` (§5.1). Use this only after v1 is validated.
- `Λ = L_e·|cosθ_z|/r²` (with `cosθ_z > 0` required by one-sidedness). Therefore
  ```
  F_NEE = T · ω1 · f_cos · L_e · |cosθ_z| · A /(r² · P(L)) · V ,   p1 = P(L)·r²/(A·|cosθ_z|)
  ω1 = Mode A ? 1 : M p1/(M p1 + p2)
  ```
- **BSDF hit (Mode B):** `F = T_prefix · (f_ℓ cos/p(ω,ℓ)) · L_e(z→x) · ω2`, with `ω2 = p2/(M p1(z|x) + p2)` and `p2` = marginal BSDF pdf at `x` for ω.

### 3.5 Emissive triangle (two-sided, textured)

- **Emission [SOURCE]:** `L_e = strength·color` (textured: `emissiveFactor · decode(tex(uv(z))) · emissiveStrength`), **two-sided** (`|Ng·wi| > 0`, cycles-conventions §3; `K/closure/emissive.h` via that report), opaque and occluding.
- **Sampling [INFERENCE]:** `P(tri)` from the power alias table (flux pre-integration: scene-io §3.7), uniform on the triangle: `b = (1−√u₁, √u₁(1−u₂), √u₁u₂)`, `p_A = 1/A_t`, `q = P(tri)/A_t`. (Cycles switches to spherical-triangle solid-angle sampling for large or near triangles, `K/light/triangle.h:63-109`. Only variance differs.)
- `Λ = L_e(uv)·|cosθ_z|/r²`. `p1 = q·r²/|cosθ_z|`.
- NEE: `ω1 = M p1/(M p1 + p2)`. BSDF hit: `ω2 = p2/(M p1 + p2)`. This holds in **both** modes.
- The texture enters `L_e` only. `P(tri)` may use an approximate (averaged) flux; that is variance only, provided the same pmf is used for sampling and for evaluating `p1`.

### 3.6 Sun, angle = 0 (delta direction)

- **Emission [SOURCE]:** irradiance `E = Φ` (W/m²) on a surface perpendicular to the sun. Direction to the sun `s_L = +Z_obj` (cycles-conventions §2.5; verified). Angle 0: `pdf = 1`, `eval_fac = 1` (cycles-verify §4.5).
- `Λ = E`, `q = P(L)`. `F = T · f_cos(ω_o, s_L) · E · V_tri(x, ∞ along s_L) / P(L)`, with ω1 ≡ 1.
- **Reconnection:** the "vertex" is the direction; copying it is J = 1. No geometry term, no footprint test.

### 3.7 NEE/BSDF MIS weights, and why Cycles' power heuristic only changes variance

**Balance heuristic (single sample each) [SOURCE GRIS Eq. 49; Enhanced p-04]:** `ω1 = p1/(p1+p2)`, `ω2 = p2/(p1+p2)`, with `p1, p2` in solid angle at `x_{d−1}`. **p2 is the marginal (all-lobe mixture) BSDF pdf** (GRIS §7.6: "summed over all BSDF lobes"). The integrand of a BSDF path with lobe ℓ is `ω2·f_ℓ/p(ω,ℓ)`, where `p(ω,ℓ)` is lobe-joint (enhanced-verify O2).

**Enhanced multi-sample weights with RIS-NEE [SOURCE S-§5, `bkusnhdet.txt:326-333`]:** "for an NEE path x̄, ω1(x̄) = M p1(x̄)/(M p1(x̄) + p2(x̄)) … ω2 is defined similarly". `M` is the candidate count at that bounce, and `p1` is "the light sampling source PDF". So:
```
ω1 = M(B)·p1 / (M(B)·p1 + p2) ,   ω2 = p2 / (M(B)·p1 + p2) ,   B = index of x_{d−1}
```
[INFERENCE] `M(B)` must be the same deterministic function in the NEE code, the BSDF-hit code, and every shift. Offset paths preserve vertex indices, so `M` is unchanged by shifts. Enhanced uses 32 at B = 1 and `32/B²` thereafter (P-§6.1, §7). The rounding is **[UNVERIFIED]**; pick one, e.g. `max(1, floor(32/B²))`, and use it everywhere. EvanLuo42 does exactly this with a stored `risM`: BSDF-hit MIS is `evalBalanceMIS(1, pBsdf, risM, pLight)`, and the NEE endpoint MIS is `evalBalanceMIS(risM, pNew, 1, pBsdf)` [SOURCE `Common/HybridShift.slang`, WebFetch].

**Delta and Mode-A lights: ω1 ≡ 1 [SOURCE Cycles `surface_shader.h:377-381`; gris-verify C1].** Implement it with the `isDelta` / `modeA` flag. `P(L)` is a probability mass, not a density. Plugging it into `M p1/(M p1 + p2)` next to a nonzero BSDF density `p2` mixes units and gives ω1 < 1, which is the C1 bias.

**Partition of unity.**
- For every continuous light point `z` reachable by both techniques, `ω1 + ω2 = (M p1 + p2)/(M p1 + p2) = 1`.
- Where `p2 = 0` (delta BSDF lobe, Mode A, delta light), `ω1 = 1`.
- Where `p1 = 0` (zero-power light), `ω2 = 1`, and the path still carries `L_e = 0` in that case.

**Power heuristic changes only variance [INFERENCE, standard].**
- For any weights with `Σ_n ω_n(x̄) = 1` wherever `f > 0` and `ω_n = 0` wherever `p_n = 0`: `E[Σ_n ω_n f/p_n] = Σ_n ∫_{p_n>0} ω_n f = ∫ f`.
- Cycles' `ω_n = p_n²/Σ p_j²` (`K/sample/mis.h` via cycles-verify) satisfies both conditions, as does the balance / Enhanced M-weighted family.
- So Cycles' image expectation equals ours even though per-sample weights differ. Compare **means only**.
- Caveat (cycles-verify omission 3): under smooth shading Cycles' eval/sampler asymmetry breaks the partition slightly. Validate flat-shaded first.

### 3.8 RIS-NEE with a candidate list mixing delta and area lights

**What Enhanced says [SOURCE S-§5, `bkusnhdet.txt:334-348`]:**
- "We treat the RIS process as a black box and stick to our original primary sample space definition which assumes only one NEE sample is produced with a known PDF."
- The integrand stays `ω1(x̄)f(x̄)/p1(x̄)`.
- "The effect of the RIS is incorporated into the UCW. Instead of 1 … the UCW of the selected NEE sample becomes `W = W^RIS_{X(U)} · p1(X(U))`." Here `U` are the hypothetical single-sample random numbers, and `W^RIS` is "the path-space UCW produced by the RIS process, assuming area measure like in ReSTIR DI".
- `p1` "is also the Jacobian that converts between the parameterizations"; for plain NEE `W^RIS = 1/p1`, which cancels to 1.

**Consistent product measure [INFERENCE].** Run the RIS in μ (§3.1): candidates `X_i = (L_i, z_i) ~ q` (i = 1..M), target `p̂(X) = lum(f_cos · Λ_L(z;x))` (visibility excluded, as in Enhanced §6.1), and
```
r_i   = p̂(X_i) / q(X_i)                         // BOTH in μ: area-measure Λ with P/A,  or  I/r² with P
W^RIS = (1/p̂(Y)) · (1/M) Σ_i r_i                 // UCW in μ
W_NEE = W^RIS · q(Y) = (1/M) Σ_i r_i / r_Y       // the Enhanced "W^RIS·p1" — measure-free
```
Why this is right:
- Changing the measure of an area candidate (for example to solid angle) multiplies `p̂` and `q` by the same `r²/|cosθ_z|`, so `r_i` is invariant.
- A delta candidate has no continuous measure, so its `r_i` must be `(lum(f_cos·I/r²))/P(L)`.
- **The trap:** forming `r` with `p̂` in one measure and `q` in another. Examples:
  - `p̂ = lum(f_cos·L_e)` (solid-angle radiance, no `|cosθ_z|/r²`) with `q = P/A` (area);
  - `p̂ = lum(f_cos·I)` (no `1/r²`) with `q = P`.

  Either one reweights light types against each other and **biases**. The unbiasedness condition is `E[f(Y)W_Y] = ∫g dμ`, and it requires `p̂/q` to be the ratio of the *same* density's target and source.

**Cross-check with EvanLuo42 [SOURCE, WebFetch of `Common/Lights.slang`]:**
- `sampleEmissiveFromTiles` uses candidate weights `w = luminance(f · Le/saPdf)` with `saPdf = selectionPdf × evalTrianglePdf` (solid angle). That is `r_i` in solid-angle form for emissive triangles only.
- It returns `risUcw = weightSum/(n · w_selected)`, which is exactly `W_NEE` above, and `risM = n`.
- The NEE candidate is streamed with weight `tree.pathUcw · light.risUcw` (`Path/PathSampling.slang`), where `pathUcw` already contains `1/∏q_RR`.
- Analytic lights are sampled separately (`sampleAnalyticLight`: uniform `1/lightCount` selection, Falcor sampler, `risM = 1`) and are NEE-only.

**Pseudo-code:**
```
fn sampleNeeRIS(x: Surface, wo: vec3f, B: u32, tileBase: u32, rng: ptr<function,Rng>) -> NeeResult {
  let M = neeCandidateCount(B);                  // deterministic in B
  var sumR = 0.0; var sel: LightCand; var rSel = 0.0;
  for (var i = 0u; i < M; i++) {
    let e   = lightTile[tileBase + u32(next(rng) * 1024.0)];   // i.i.d. draws from the frame's alias table
    let uv  = vec2f(next(rng), next(rng));                     // area-uniform (ignored for delta)
    let c   = lightCandidate(e.lightId, uv, x.p);              // z, n_z, isDelta, type, Λ (one-sided, spot S), q (P/A or P)
    let ph  = luminance(bsdfEvalAllCos(x, wo, c.dir) * c.Lambda);  // p̂ in μ (no V)
    let r   = select(0.0, ph / c.q, c.q > 0.0);
    sumR += r;
    if (next(rng) * sumR <= r && r > 0.0) { sel = c; rSel = r; }   // resampling RNG: NOT the replayable path stream
  }
  if (rSel == 0.0) { return NeeResult(false); }
  let Wfac = (sumR / f32(M)) / rSel;              // = W^RIS · q(Y)
  return NeeResult(true, sel, Wfac, M);
}
```
The resampling decision must draw from a separate RNG (reference-code §1.4).

**The candidate is then**
```
F_NEE  = T · ω1 · bsdfEvalAllCos(x, wo, sel.dir) · sel.Lambda / sel.q · V_tri(x, sel.z)
ω1     = (sel.isDelta || (modeA && sel.isAnalytic)) ? 1 : M·p1/(M·p1 + p2),
         p1 = sel.q · r²/|cosθ_z|,  p2 = bsdfPdfMarginal(x, wo, sel.dir)
stream(w = lum(F_NEE) · Wfac / ∏_{RR tests survived} q_i)
```
The RR survival product must contain exactly the tests survived: for NEE from `x_{d−1}`, the tests at `x_1..x_{d−2}` if RR is evaluated after NEE (enhanced-verify O3).

**Unbiasedness with correlated candidates (tiles) [INFERENCE from GRIS Eq. 10].** Candidates in a pixel share a light tile and are correlated, but each `X_i` has marginal density `q` (tile entries are i.i.d. alias draws, and the slot is uniform). `1/q(X_i)` is therefore a valid UCW for each candidate, and GRIS Eq. 10 holds for correlated inputs. So `F·W_NEE` is unbiased for `∫ω1 g dμ`.

### 3.9 Selection pmfs are shading-point independent (alias table, light tiles)

- **Power alias table [INFERENCE]:** `P(L) = pmf[L]` is a per-frame global constant array, independent of `x`, the pixel and the path. Build it on the CPU from the power proxies of scene-io §7.6. Rigid light motion changes neither power nor area, so no rebuild is needed.
- **Light tiles [SOURCE Enhanced §6.1 p-10: "each frame precomputes 128 tiles with 1024 lights, from which each 8×8 screen tile draws a light tile"] + [INFERENCE]:**
  - Tiles are filled with i.i.d. alias draws, so a candidate's **marginal** distribution is `pmf[L]·p_A(z|L)` for every pixel.
  - The **conditional** distribution given a screen tile's light tile does depend on the pixel's screen tile: it equals the multiplicity of L in that tile divided by 1024.
  - **Only the marginal may be used** for `q`, `p1`, ω and J. Offset pixels may belong to other screen tiles, and the BSDF-hit MIS must be evaluable anywhere without knowing tile contents.
  - Tiles must be generated from the **same frame's** alias table that is used to evaluate `pmf`.
- **Not independent:** a light tree or BVH, or Cycles' `use_light_tree`. Then `P(L|y)` must be evaluated at the offset vertex (gris-verify C2, corrected rule).

---

## 4. Reservoir encoding

### 4.1 Layout (64 B, 16 × u32) [INFERENCE, adapted from S-Alg.1 `bkusnhdet.txt:401-413`]

| Word | Field | Content |
|---|---|---|
| 0 | `W` f32 | UCW in PSS |
| 1–3 | `F` f32×3 | PSS integrand of the stored path in this pixel's domain; `p̂ = lum(F)` |
| 4 | `initSeed` u32 | counter-RNG key for prefix replay |
| 5 | `endpointId` u32 | **repurposed `rcVertexRandomSeed`**: analytic `lightId` or global emissive-triangle index. With a counter-based RNG keyed by `(initSeed, vertexIndex, dim)` the suffix seed is derivable (reference-code §4.2 proposes dropping `rcRandomSeed`) |
| 6 | `pathFlags` u32 | §2.2 |
| 7 | `rcA` u32 | `instanceId`, or `0x80000000 \| lightId` when the rc vertex is on an analytic light |
| 8 | `rcB` u32 | `primId` (triangle); unused for analytic lights |
| 9 | `rcBary` u32 | unorm16×2: triangle barycentrics, or light-local `(u,v)` |
| 10 | `rcWi` u32 | oct unorm16×2 direction ω_k; **or** light `(u,v)` in case (b) |
| 11–13 | `rcRadiance` f32×3 | per case, §4.2 |
| 14 | `cachedJden` f32 | Jacobian denominator product of the **stored** path, §5 |
| 15 | `aux` f32/u32 | case (c): `p1` at `x_{d−1}` (solid angle); case (e) with NEE ending: packed light `(u,v)`; otherwise unused |

**Validation build (critique X11, enhanced-verify O9):** 96 B with f32 barycentrics, f32 `(u,v)`, f32×3 direction and f32 `M`. Alternatively, quantise *before* computing `F` and `cachedJden`, so the stored path is exactly the quantised one.

### 4.2 Field contents per reconnection case [INFERENCE]

Cases (§5): **(a)** rc = NEE light vertex (k = d, forced); **(b)** rc = x_{d−1}, NEE suffix; **(c)** rc = x_{d−1}, BSDF-hit-emitter suffix; **(d)** rc = BSDF-hit emitter vertex (k = d); **(e)** k ≤ d−2; **(n)** no rc (BSDF endpoints only).

| Case | rcA/rcB/rcBary | rcWi | rcRadiance | cachedJden | aux | endpointId |
|---|---|---|---|---|---|---|
| (a) area/tri | light: `LIGHT\|L`, `(u,v)`; tri: inst, prim, bary | unused | unused. **Recompute** `L_e`/`Λ` at y (moving lights, spot, one-sided). Optional cache of `L_e` | area-uniform: unused (J = pmf ratio, §5.1); solid-angle sampler: `p_A^x(z) = P·p_σ^x·|cosθ_z^x|/r_x²` | unused | L / tri index |
| (a) delta | `LIGHT\|L` | unused | unused (recompute `I(ω_y)/r_y²`) | unused (J = 1) | unused | L |
| (b) | x_{d−1} (inst, prim, bary) | light `(u,v)` (area/tri; for delta and sun the direction is derivable from L) | `Λ_x = Λ_L(z; x_{d−1})`: **MIS and 1/q divided out** | `p^x_{d−2}(ω_{d−2}, ℓ_{d−2}) · G(x_{d−2}→x_{d−1})` (p_k := 1, §5.2) | optional `p1^x` (recomputable) | L / tri index |
| (c) | x_{d−1} | ω_{d−1} | `L_e(z→x_{d−1})`: **ω2 divided out** | `p^x_{d−2}(ω_{d−2},ℓ_{d−2}) · G(x_{d−2}→x_{d−1}) · p^x_{d−1}(ω_{d−1}, ℓ_{d−1})` | `p1^x = P·p_A·r²/\|cosθ_z\|` from x_{d−1} (unchanged in spatial shifts) | L (Mode B) / tri index (temporal refresh) |
| (d) | emitter point: `LIGHT\|L`, `(u,v)`, or tri inst/prim/bary | unused | optional `L_e` cache (recompute at y for one-sidedness and texture lookups) | `p^x_{d−1}(ω_{d−1}, ℓ_{d−1}) · G(x_{d−1}→z)` | unused (**p1 recomputed at y_{d−1}**, C2) | L / tri |
| (e) | x_k | ω_k | full suffix `Π_{k<j<d}(f cos/p)·Λ-or-L_e·ω_end`: **MIS included, not divided out** (unchanged by shifts) | full Eq. 2 product at k | NEE ending: light `(u,v)` for temporal refresh | L / tri |
| (n) | unused | unused | unused | unused | unused | L (Mode B label) |

**Differences from Falcor (reference-code §1.5.5, gris-math §6.8).** Falcor stores `L_e` (MIS divided out) for cases (a)–(d) plus a *solid-angle* `lightPdf` measured from the base `x_{d−1}`. Case (d) with that stale pdf is the confirmed C2 bias (gris-verify C2). This design recomputes every quantity that depends on the offset shading point.

### 4.3 Which pdfs to cache [INFERENCE]

- **Never cache a solid-angle light pdf measured at a vertex the shift replaces** (cases (a) and (d): the shading point becomes `y_{d−1}`). Recompute `p1^y = P(L)·p_A(z|L)·r_y²/|cosθ_z^y|`, or `P(L)` for delta.
- A cache is allowed when the measuring vertex is copied (cases (b) and (c): `x_{d−1}` is the rc vertex), **within a frame**.
- Across frames (temporal forward or inverse shift), recompute from the frame's own light buffer and pmf. Power, colour, size or existence may have changed.
- Area-measure quantities `P(L)/A_L` are shading-point independent. They are cheap to recompute from `lightId`, so caching them buys little.
- The Jacobian denominator (`cachedJden`) **must** be cached (the base `x_{k−1}` is not stored). It **must be replaced by the offset's product when a shifted sample is selected** (enhanced-verify C4).

---

## 5. Shifted PSS integrand and Jacobian, per reconnection case

**Common definitions [SOURCE Enhanced Eq. 2, p-04; S-Eq. 4].**
```
|∂T/∂ū| = [ p^y_{k−1}(ω'_{k−1}, ℓ_{k−1}) · G(y_{k−1}→x_k) · p^y_k(ω_k, ℓ_k) ] / [ p^x_{k−1}(ω_{k−1}, ℓ_{k−1}) · G(x_{k−1}→x_k) · p^x_k(ω_k, ℓ_k) ]
p^•_k(ω_k,ℓ_k) := 1 when k = d ;   J = 1 without reconnection (random replay is the identity in PSS)
G(a→b) = |⟨n^geo_b, (a−b)/‖a−b‖⟩| / ‖a−b‖²   (cosine at the RECEIVING vertex b; enhanced-verify C8)
```
- `p(ω, ℓ)` is the lobe-joint pdf (selection × lobe pdf) (enhanced-verify O2).
- `Tp` = replayed prefix throughput `Π_{i=1}^{k−2} f_{ℓ_i}cos/p(ω_i,ℓ_i)` over `y_1..y_{k−2}` (`Tp = 1` if `k = 2`). All vertex indices are Enhanced camera-based.
- `ω' = normalize(x_k − y_{k−1})`.
- **[INFERENCE]** When `ω_{k−1}` is light-sampled (NEE, case (a)), `p_{k−1}` in Eq. 2 is the light sampler's solid-angle pdf. Then `p_{k−1}·G(·→z) = p_A(z)`, the light's area density.

**Prefix replay and the invertibility checks are identical for every case** (enhanced-paper §3.10–3.11, gris-math §6.7):
- no earlier replayed pair qualifies;
- the lobe exists at `y_{k−1}`;
- the pair `(y_{k−1}, x_k)` meets the criteria, except for forced NEE.

The cases below add the light-specific parts.

### 5.1 Case (a): rc = NEE light vertex (forced, k = d)

- Base: `x_{d−1} →NEE→ z`.
- Offset: `y_{d−1}` (replayed; or the destination primary hit if d = 2, which is the ReSTIR DI shift) → `z`. The light sample is copied: `(L, u, v)` or `(tri, bary)`.
```
ω_y   = normalize(z − y_{d−1})          (sun: s_L)
F(ȳ)  = Tp · [f_{ℓ_{d−2}} cos/p]_{y_{d−2}→y_{d−1}} (inside Tp when k=d) · f_cos,all(y_{d−1}; ω_o^y → ω_y) · Λ_L(z; y_{d−1}) / q · ω1^y · V_tri(y_{d−1}, z)
ω1^y  = isDelta||modeA ? 1 : M(d−1)·p1^y/(M(d−1)·p1^y + p2^y)
p1^y  = q · r_y²/|cosθ_z^y|            (RE-EVALUATED at y_{d−1}; C2)
p2^y  = bsdfPdfMarginal(y_{d−1}; ω_o^y → ω_y)
Λ_L(z; y) : area → L_e·|cosθ_z^y|/r_y² with one-sidedness dot(y − z, a_L) > 0 else 0;
            tri  → L_e(bary)·|cosθ_z^y|/r_y² (two-sided);
            point→ Φ/(4π r_y²);  spot → Φ/(4π r_y²)·S(dot(normalize(y − c), a_L));  sun → E
```
NEE evaluates **all lobes** at `y_{d−1}` (GRIS §7.6). This matches EvanLuo42's NEE endpoint: `contribution = prefixThp · fNew · mNew · suffixThroughput / pNew`, with `pNew = lightSolidAnglePdfFromArea(storedAreaDensity, prefix.position, …)`, i.e. the stored area density converted at the new prefix, and `jacobian = 1` [SOURCE `Common/HybridShift.slang`, WebFetch].

**Jacobian:**
```
area-uniform light sampling, same frame (spatial):  J = 1
same, temporal (frame t−1 → t):                    J = pmf_t(L)/pmf_{t−1}(L)   (area-uniform (u,v): A cancels; §5.7)
solid-angle light sampler (e.g. spherical rect):    J = p_A(z|y_{d−1}) / p_A(z|x_{d−1}) = [p_σ^y(z)·|cosθ_z^y|/r_y²] / cachedJden   (× pmf ratio if temporal)
delta (point/spot), sun:                            J = 1   (× pmf ratio if temporal)
```
**Undefined:**
- replay invertibility failure (an earlier pair qualifies in ȳ);
- the replay escapes or terminates before `y_{d−1}`;
- `L` not present in the destination frame (`curToPrev`/`prevToCur` map = −1; restir-practice §3.5);
- no NEE possible at `y_{d−1}` (no evaluable lobe).

The forced connection itself has **no criterion check**, so it never fails the rc test.

**Defined, zero contribution:**
- `V_tri = 0`;
- area light seen from behind;
- spot `S = 0`;
- `f_cos = 0` (below the hemisphere).

### 5.2 Case (b): rc = x_{d−1}, NEE suffix (k = d−1)

- Base: `x_{d−2} → x_{d−1} →NEE→ z`.
- Offset: `y_{d−2} → x_{d−1}` (reconnect), then `x_{d−1} → z` is **copied**: same point, same direction `ω_{d−1}`.
```
F(ȳ) = Tp · f_{ℓ_{d−2}}(y_{d−2}; ω_o^y → ω') cos / p^y(ω', ℓ_{d−2})
          · f_cos,all(x_{d−1}; −ω' → ω_{d−1}) · Λ_x / q · ω1^y · V_tri(y_{d−2}, x_{d−1})
ω1^y = isDelta||modeA ? 1 : M(d−1)·p1/(M(d−1)·p1 + p2^y)
p1   = q·r_x²/|cosθ_z| from x_{d−1}         (unchanged: cached `aux` or recomputed)
p2^y = bsdfPdfMarginal(x_{d−1}; −ω' → ω_{d−1})   (CHANGES: new incoming direction)
J    = [p^y(ω', ℓ_{d−2})·G(y_{d−2}→x_{d−1})] / cachedJden      (p_k := 1: ω_{d−1} is light-sampled and the
                                                               light sample from the fixed x_{d−1} is unchanged)
```
- `V_tri(x_{d−1}, z) = 1` is known from the base path in a spatial shift.
- In a temporal forward shift where L moved, rebuild `z = Φ_t(L,u,v)`, recompute `Λ`, `p1` and `ω1`, and re-trace `V_tri(x_{d−1}, z)`. That is one extra ray and no light re-sampling (restir-practice §3.6).
- **Undefined:** replay or rc-criterion failure at `(y_{d−2}, x_{d−1})` (including delta lobes, footprint, roughness); L removed.
- **Zero:** `V_tri(y_{d−2}, x_{d−1}) = 0`; `f = 0` at either vertex; temporal `V_tri(x_{d−1}, z_t) = 0`; spot or one-sided zeros after the light moved.

### 5.3 Case (c): rc = x_{d−1}, BSDF-hit-emitter suffix (k = d−1)

- Base: `x_{d−2} → x_{d−1} →BSDF(ℓ_{d−1}, ω_{d−1})→ z` (emissive triangle, or analytic area light in Mode B).
- Offset: reconnect to `x_{d−1}`, then **direction copy** of `ω_{d−1}`, so `z` is unchanged in the same frame.
```
F(ȳ) = Tp · f_{ℓ_{d−2}}(y_{d−2}; ω_o^y→ω') cos / p^y(ω', ℓ_{d−2})
          · f_{ℓ_{d−1}}(x_{d−1}; −ω' → ω_{d−1}) cos / p^y(ω_{d−1}, ℓ_{d−1})
          · L_e(z→x_{d−1}) · ω2^y · V_tri(y_{d−2}, x_{d−1})
ω2^y = p2^y / (M(d−1)·p1 + p2^y) ,   p2^y = bsdfPdfMarginal(x_{d−1}; −ω' → ω_{d−1}) ,   p1 = aux (unchanged)
J    = [p^y(ω',ℓ_{d−2})·G(y_{d−2}→x_{d−1})·p^y(ω_{d−1},ℓ_{d−1})] / cachedJden        (full Eq. 2)
```
- The factor `p^y(ω_{d−1},ℓ_{d−1})/p^x(…)` is applied **whenever ω_{d−1} was BSDF-sampled, independently of whether MIS is on** (gris-verify C3). Falcor applies it only inside the MIS branch (`Shift.slang:525-535`), which is biased with MIS off.
- **Undefined:** replay or rc failure; lobe `ℓ_{d−1}` unavailable at `x_{d−1}` for the new incoming direction; delta lobe at `ℓ_{d−1}`; Mode B L removed.
- **Zero:** `V_tri = 0`; `f = 0`; `p^y(ω_{d−1},ℓ_{d−1}) = 0`.
- Temporal refresh (Mode B, L moved): re-trace `(x_{d−1}, ω_{d−1})` for crossings of L. F changes and may become 0.

### 5.4 Case (d): rc = the BSDF-hit emitter vertex (k = d)

- Base: `x_{d−1} →BSDF(ℓ_{d−1}, ω_{d−1})→ z`.
- Offset: `y_{d−1}` (replayed) → `z` (copied, light-local or `(inst, prim, bary)`).
```
ω'   = normalize(z − y_{d−1})
F(ȳ) = Tp · f_{ℓ_{d−1}}(y_{d−1}; ω_o^y → ω') cos / p^y(ω', ℓ_{d−1}) · L_e(z→y_{d−1}) · ω2^y · V_tri(y_{d−1}, z)
ω2^y = p2^y / (M(d−1)·p1^y + p2^y) ,  p2^y = bsdfPdfMarginal(y_{d−1}; ω_o^y → ω') ,
p1^y = P(L)·p_A(z)·r_y²/|cosθ_z^y|        (RE-EVALUATED at y_{d−1}: gris-verify C2; EvanLuo42 evalStoredEmissivePdf)
J    = [p^y(ω', ℓ_{d−1})·G(y_{d−1}→z)] / cachedJden,  G uses n_L (area light) or the triangle n_g
```
- `L_e(z→y)`: area light → 0 unless `dot(y−z, a_L) > 0`; triangle → two-sided texture lookup at `bary`.
- **Visibility:** for a triangle endpoint, the first triangle hit from `y_{d−1}` toward `z` must be `z`'s triangle (trace to `(1−ε)r`, excluding that primitive). For an area light, `V_tri` only: other analytic lights never block.
- **Undefined:**
  - replay invertibility failure;
  - `(y_{d−1}, z)` fails the rc criterion: roughness at `y_{d−1}` for `ℓ_{d−1}`, `rayFP = r_y²/(p_{d−1}(ω')|cosθ_z^y|) ≥ thr` with the offset pixel's own R_pri (enhanced-paper §3.10); no inverse-footprint test (Enhanced footnote 6);
  - delta lobe;
  - Mode B L removed.
- **Zero:** `V_tri = 0`, back-facing area light, `f = 0`, `p^y(ω', ℓ) = 0`.
- **Point/spot r = 0 and sun never reach this case** (they cannot be hit).

### 5.5 Case (e): rc earlier (k ≤ d−2), light-terminated suffix

Generic hybrid reconnection (gris-math §6.8 row 1). **Nothing light-specific changes in a spatial shift.** The suffix vertices `x_{k+1} … x_{d−1}` and `z` are copied. The incoming direction at `x_{d−1}` comes from `x_{d−2}`, which is `x_k` or later and therefore fixed. So `ω_end` (NEE or BSDF MIS at `x_{d−1}`) and `p1` are unchanged, and the cached `rcRadiance` keeps its MIS weight.
```
F(ȳ) = Tp · f_{ℓ_{k−1}}(y_{k−1}; ω_o^y→ω')cos/p^y(ω',ℓ_{k−1}) · f_{ℓ_k}(x_k; −ω' → ω_k)cos/p^y(ω_k,ℓ_k) · rcRadiance · V_tri(y_{k−1}, x_k)
J    = full Eq. 2
```
**Temporal refresh when analytic lights moved [INFERENCE].** Geometry is static and lights are non-occluding, so `x_{k+1} … x_{d−1}` are **identical** across frames (same rays, same BVH). Only the terminal factor changes: the NEE point `z_t = Φ_t(L,u,v)` from `aux`, `Λ`, `p1`, ω, `V_tri(x_{d−1}, z_t)`, plus crossings in Mode B. Evaluating `f(x_{d−1})` toward the new `z_t` needs `x_{d−1}`, so re-trace the suffix by direction copy of `ω_k` plus suffix replay (BSDF sampling only, no light sampling), then connect to the stored light point.

This removes the dependency on replaying light selection that critique Gap 2 flagged for Falcor's `traceTemporalUpdate`, which is unusable under per-frame tiles.

### 5.6 "Shift undefined" vs "defined with zero contribution" [INFERENCE from GRIS Def. 4.2, Eq. 19–20]

- **Both** produce `w_i = 0` and `p̂_{←i}(y) = 0` in every MIS denominator. Numerically they are the same `(F = 0, J = 0)` return.
- **The distinction matters for bijectivity only.** "Undefined" conditions define `D(T)`, and they must be decided by **exactly the same predicate** in `T` and `T⁻¹` (same code, same thresholds, same normal type, same R_pri definition). "Zero" conditions are physical zeros of `f` and need no symmetry.
- **Undefined:**
  - an earlier qualifying pair in the offset replay;
  - the offset replay ends (miss / termination) before `y_{k−1}`;
  - lobe unavailable or delta at `ℓ_{k−1}` / `ℓ_k`;
  - `(y_{k−1}, x_k)` fails the (non-forced) criteria;
  - in full replay, the offset finds a qualifying emitter rc;
  - technique or length mismatch;
  - endpoint light id missing in the destination frame;
  - non-finite or ≤ 0 J (numerical guard);
  - optional symmetric `max(J, 1/J)` rejection (enhanced-verify O6).
- **Zero:**
  - `V_tri = 0` on any re-traced segment;
  - area light back side;
  - spot outside the cone (`S = 0`);
  - `L_e = 0`;
  - `f = 0` (below the hemisphere);
  - `p^y(ω, ℓ) = 0`;
  - in Mode B full replay, L not crossed.
- **Never drop a neighbour from the pairwise normaliser because its shift failed.** "Valid neighbour" must depend on G-buffer data only (course notes Tip 4.1, restir-practice §2.4). Falcor still adds `M_i` for a failed shift (reference-code §1.6.5).

### 5.7 Reconciling J = 1 (area measure) with RTXDI's solid-angle pdf ratio (critique X14) [INFERENCE]

For a light vertex copied as a point `z`, three bookkeeping conventions are consistent **iff the target is expressed in the matching measure**. The product `p̂(y)·J·W` is invariant:

| Measure of the light vertex in the reservoir domain | target `p̂(ȳ)` (last factor) | `J` for copying `z` | initial `W` of an NEE sample |
|---|---|---|---|
| Area (path space) | `lum(f_cos · L_e · |cosθ_z|/r²)` | **1** | `1/(P(L)·p_A(z))` (× prefix) |
| Solid angle at the shading vertex (`ω_{d−1}` is the variable) | `lum(f_cos · L_e)` | `G_y/G_x` (Eq. 52) = `p_σ^x(z)/p_σ^y(z)` **if** `p_A(z)` is shading-point independent | `1/(P(L)·p_σ^x(z))` |
| PSS (this report; Falcor; Enhanced) | `lum(f_cos · L_e · |cosθ_z|/(r² · P(L) p_A(z|·)))` | `p_A(z|y)/p_A(z|x)` (= **1** for area-uniform sampling) | `1` (× RIS/RR factors) |

**RTXDI [SOURCE reference-code §2.2.5; semantics only].** It stores the NEE light's solid-angle pdf as `partialJacobian` and uses a ratio of solid-angle pdfs. That is row 2 when the ratio is `pdf_src/pdf_dst`. My own WebFetch reading of `HybridShift.hlsli` returned *both* the orientation "old/new" and a target that divides by the new solid-angle pdf. That combination belongs to none of the three consistent rows, so the exact RTXDI convention is **[UNVERIFIED]**. **Do not port it; use the PSS row.**

**Delta lights.** There is no continuous measure for the light vertex, so J = 1 in every convention. Cycles' pseudo-pdf `t²` for point lights (`point.h:75-76`) is **not** a density. Using it in `J = pdf_src/pdf_dst` would give `t_x²/t_y²`, which is a bias.

**Temporal (moving, rescaled or reweighted lights) in PSS with area-uniform `(u,v)`.**
- The light dimensions are `(u_sel, u, v)`.
- Copying `(L, u, v)` is the identity on `(u, v)`, and maps `u_sel` from L's alias bucket at t−1 to L's bucket at t.
- So `J = pmf_t(L)/pmf_{t−1}(L)`.
- The area `A_t` (e.g. a resized light) enters only `F` through `q = pmf/A_t`, and cancels in J because `(u,v)` is area-uniform at both times.
- Rigid motion with unchanged power gives J = 1.

### 5.8 Selection of a shifted sample: what to write back [SOURCE enhanced-verify C4 + INFERENCE]

When `ȳ = T(x̄)` is selected:
- `F ← F(ȳ)`;
- `cachedJden ← ` the offset's own product: `p^y_{k−1}·G(y_{k−1}→x_k)·p^y_k`, or `p_A^y(z)` for case (a) with a solid-angle sampler;
- case (c): `aux ← p1` (unchanged, since `x_{d−1}` is the same);
- keep `(L, u, v)`, rc, flags and seeds.

Nothing light-dependent at the shading point is cached, so no light pdf needs updating.

---

## 6. Length-1 paths (camera sees an emitter or a camera-visible light)

- **Enhanced [SOURCE p-03, p-10].**
  - The path-space integral is written `Σ_{d=1}^∞ ∫ f`, "starting from primary hit x1".
  - §6.1: NEE is traced from `x1`, "giving the (unified) initial resampling a chance to select a shorter direct lighting path (d = 2)".
  - No length-1 (`d = 1`) resampling is described. **[INFERENCE]** The reservoir holds `d ≥ 2`, and `d = 1` is added outside it.
- **EvanLuo42 [SOURCE `Path/PathSampling.slang`, WebFetch].** `primaryEmission = surface.properties.emission` is output separately and "not added to reservoir". Analytic lights are never hit by rays there.
- **Falcor 2022 [SOURCE gris-verify Q3].** ReSTIR DI's final shading adds `sd.emissive`.
- **Cycles [SOURCE §1.2].** Camera rays carry `PATH_RAY_MIS_SKIP`, so the weight is 1. Camera-visible analytic lights are passed through.
- **Rule [INFERENCE]:**
  ```
  pixel = Σ_{L ∈ analytic, visibleToCamera, crossed front-facing before x1} L_e,L(z_L → x0)   // weight 1, pass-through
        + L_e(x1 → x0)                                                              // emissive triangle at x1, weight 1
        + reservoirEstimate(d ≥ 2)                                                  // F·W or Σ RGB w_i (Enhanced §6.3)
  ```
  - The V-buffer must ignore analytic lights.
  - Point/spot r = 0 are never camera-visible.
  - This is unbiased because the `d = 1` sub-integral has exactly one technique.
  - With per-frame primary jitter (critique X1) it is a per-frame estimate like the rest.

---

## 7. WGSL-oriented pseudo-code (condensed)

```wgsl
// ---------- light helpers (frame-selectable: CUR or PREV buffers) ----------
struct LightPoint { valid: bool, isDelta: bool, isInf: bool, isAnalytic: bool, type: u32,
                    pos: vec3f, n: vec3f, dirInf: vec3f, q: f32 /* P/A or P */ };
fn lightPointFromLocal(L: u32, uv: vec2f, fr: u32) -> LightPoint;       // rect/disk/point/spot/sun
fn triLightPoint(inst: u32, prim: u32, bary: vec2f, fr: u32) -> LightPoint; // q = pmf[tri]/A_tri
fn lightTerm(lp: LightPoint, x: vec3f, fr: u32) -> vec3f {                // Λ in μ
  if (lp.isInf) { return sunE(lp, fr); }
  let d = x - lp.pos; let r2 = dot(d,d); let w = d * inverseSqrt(r2);
  switch lp.type {
    case LT_POINT: { return I0(lp,fr) / r2; }
    case LT_SPOT:  { return I0(lp,fr) * spotS(dot(w, axis(lp,fr)), lp, fr) / r2; }
    case LT_RECT, LT_DISK: { let c = dot(lp.n, w); return select(vec3f(0), Le(lp,fr) * c / r2, c > 0.0); } // one-sided
    default /*tri*/: { return LeTex(lp, fr) * abs(dot(lp.n, w)) / r2; }                                     // two-sided
  }
}
fn p1Sigma(lp: LightPoint, x: vec3f) -> f32 {   // single-sample NEE pdf, solid angle; 0 for delta
  if (lp.isDelta) { return 0.0; }
  let d = x - lp.pos; let r2 = dot(d,d); let c = abs(dot(lp.n, d)) * inverseSqrt(r2);
  return select(0.0, lp.q * r2 / c, c > 0.0);
}
fn omegaNEE(lp: LightPoint, M: f32, p1: f32, p2: f32) -> f32 {
  if (lp.isDelta || (MODE_A && lp.isAnalytic)) { return 1.0; }
  let a = M * p1; return a / (a + p2);
}
fn omegaBSDF(M: f32, p1: f32, p2: f32) -> f32 { return p2 / (M * p1 + p2); }

// ---------- path tree at vertex x_B (B >= 1) ----------
// NEE (all lobes)
let nee = sampleNeeRIS(sd, wo, B, tileBase, &resRng);               // §3.8
if (nee.ok) {
  let lp = nee.sel; let wi = dirTo(lp, sd.p);
  let fc = bsdfEvalAllCos(sd, wo, wi);
  let om = omegaNEE(lp, f32(nee.M), p1Sigma(lp, sd.p), bsdfPdfMarginal(sd, wo, wi));
  var F = T * fc * lightTerm(lp, sd.p, CUR) / lp.q * om;
  if (any(F > vec3f(0)) && visibleTri(sd, lp)) {
    stream(cand(NEE, d = B+1, endpoint = lp, rc = rcForNee(tree, B)), F, nee.Wfac / rrProdBefore(B));
  }
}
// RR (initial only) here, then BSDF sample (fixed RNG dims)
let bs = sampleLobe(sd, wo, dimsBSDF(B));                          // ω, ℓ, pJoint = P(ℓ)p_ℓ(ω), pMarg
T *= bs.fcosLobe / bs.pJoint;
let hit = traceClosestTri(sd.p, bs.w);
if (MODE_B) {
  for (var L = 0u; L < numAreaLights; L++) {                       // pass-through, no vertex, no RNG
    let c = intersectFrontFacing(L, sd.p, bs.w, hit.t);
    if (c.hit) {
      let lp = lightPointFromLocal(L, c.uv, CUR);
      let F = T * Le(lp, CUR) * omegaBSDF(M(B), p1Sigma(lp, sd.p), bs.pMarg);
      stream(cand(BSDF_ANALYTIC, d = B+1, endpoint = lp, rc = rcForBsdfEnd(tree, B, lp)), F, 1.0 / rrProd(B));
    }
  }
}
if (hit.valid && hit.emissive) {
  let lp = triLightPoint(hit.inst, hit.prim, hit.bary, CUR);
  let F = T * LeTex(lp, CUR) * omegaBSDF(M(B), p1Sigma(lp, sd.p), bs.pMarg);
  stream(cand(BSDF_TRI, d = B+1, endpoint = lp, rc = rcForBsdfEnd(tree, B, lp)), F, 1.0 / rrProd(B));
}

// ---------- reconnection (after hybrid replay gave yPrev = y_{k-1}, Tp, and replay invertibility passed) ----------
fn reconnect(yPrev: Surface, woY: vec3f, r: Reservoir, Tp: vec3f, fr: u32) -> Shift {
  let k = rcIndex(r); let d = pathLen(r);
  if (k == d && tech(r) == NEE) {                                   // case (a)
    let lp = endpointFromRc(r, fr); if (!lp.valid) { return UNDEFINED; }
    let wi = dirTo(lp, yPrev.p);
    let om = omegaNEE(lp, M(d-1), p1Sigma(lp, yPrev.p), bsdfPdfMarginal(yPrev, woY, wi));
    let F = Tp * bsdfEvalAllCos(yPrev, woY, wi) * lightTerm(lp, yPrev.p, fr) / lp.q * om;
    if (all(F == vec3f(0)) || !visibleTri(yPrev, lp)) { return ZERO; }
    return Shift(F, jacLightVertex(lp, yPrev, r, fr) /* 1 | pmf ratio | pA(z|y)/cachedJden */);
  }
  if (k == d) {                                                      // case (d): BSDF-hit emitter is rc
    let lp = endpointFromRc(r, fr); if (!lp.valid) { return UNDEFINED; }
    let w = dirTo(lp, yPrev.p);
    if (!rcCriterionPasses(yPrev, lobeKm1(r), w, lp)) { return UNDEFINED; }   // roughness@y, rayFP (no invFP)
    let pj = bsdfPdfLobe(yPrev, woY, w, lobeKm1(r)); let fl = bsdfEvalLobeCos(yPrev, woY, w, lobeKm1(r));
    let om = omegaBSDF(M(d-1), p1Sigma(lp, yPrev.p), bsdfPdfMarginal(yPrev, woY, w));   // C2: p1 at y
    let F = Tp * fl / pj * emittedToward(lp, yPrev.p, fr) * om;                          // one-sided / two-sided
    if (all(F == vec3f(0)) || !visibleToEndpoint(yPrev, lp)) { return ZERO; }
    return Shift(F, pj * geomG(yPrev.p, lp.pos, lp.n) / r.cachedJden);
  }
  // cases (b), (c), (e): rc = scattering vertex x_k
  let xk = loadSurfaceFromRc(r, fr); let wp = normalize(xk.p - yPrev.p);
  if (!rcCriterionPasses(yPrev, lobeKm1(r), wp, xk)) { return UNDEFINED; }
  let p1y = bsdfPdfLobe(yPrev, woY, wp, lobeKm1(r)); let f1 = bsdfEvalLobeCos(yPrev, woY, wp, lobeKm1(r));
  var J = p1y * geomG(yPrev.p, xk.p, xk.ng) / r.cachedJden;
  var F: vec3f;
  if (k == d - 1 && tech(r) == NEE) {                                                  // (b)
    let lp = endpointFromAux(r, fr); let wi = dirTo(lp, xk.p);
    let om = omegaNEE(lp, M(d-1), p1Sigma(lp, xk.p), bsdfPdfMarginal(xk, -wp, wi));  // p2 changes
    F = Tp * f1 / p1y * bsdfEvalAllCos(xk, -wp, wi) * r.rcRadiance /* Λ_x */ / lp.q * om;
  } else if (k == d - 1) {                                                             // (c)
    let wk = r.rcWi; let p2y = bsdfPdfLobe(xk, -wp, wk, lobeK(r));
    let om = omegaBSDF(M(d-1), r.aux /* p1 from x_k */, bsdfPdfMarginal(xk, -wp, wk));
    F = Tp * f1 / p1y * bsdfEvalLobeCos(xk, -wp, wk, lobeK(r)) / p2y * r.rcRadiance /* L_e */ * om;
    J *= p2y;                                                                          // unconditional (C3)
  } else {                                                                             // (e)
    let wk = r.rcWi; let p2y = bsdfPdfLobe(xk, -wp, wk, lobeK(r));
    F = Tp * f1 / p1y * bsdfEvalLobeCos(xk, -wp, wk, lobeK(r)) / p2y * r.rcRadiance;  // MIS inside
    J *= p2y;
  }
  if (!finitePositive(J)) { return UNDEFINED; }
  if (all(F == vec3f(0)) || !visibleTri(yPrev, xk)) { return ZERO; }
  return Shift(F, J);
}
```
In (b), the NEE point is `endpointFromAux` (lightId + `(u,v)` stored in `rcWi`/`endpointId`). In (b) and (c) the ratio `p2y/(…)` enters J only for BSDF-sampled `ω_k`, and `cachedJden` includes `p^x_k` exactly when `J` includes `p^y_k`.

---

## 8. Unit tests and validation scenes

Numbering continues validation-harness §4 (T1–T16). These are additions or specialisations.

**U1 — NEE+BSDF MIS partition per light type (extends T9c).**
- Random configurations: `x`, `n`, `ω_o`, BSDF ∈ {Lambert, GGX r ∈ {0.05, 0.2, 0.5, 1}, 50/50 mix}; lights rect, disk, emissive triangle (Mode B also analytic area); `M ∈ {1, 2, 8, 32}`.
- Sample `z` via NEE, compute `ω1` with the *NEE-time* code. Construct the same `z` via the BSDF-hit code path by tracing toward `z`, compute `ω2` with the *hit-time* code.
- Assert `|ω1 + ω2 − 1| < 1e-6`.
- Delta lights and Mode-A analytic lights: assert `ω1 == 1` exactly, and that the BSDF crossing loop never reports them.
- Negative control: compute ω1 for a point light via `M·P/(M·P + p2)`. The test must catch it (C1).

**U2 — Three-estimator direct light per type (extends T9d).** NEE-only, BSDF-only (continuous lights) and MIS renders of C0c–C0e and an emissive-triangle variant agree with each other and with the analytic value to within 3σ.

**U3 — RIS-NEE mixed candidate list.**
- Scene: 1 point + 1 spot + 1 rect + 1 textured emissive triangle + 1 sun over a Lambert plane.
- `E[F·W_NEE]` with `M ∈ {1, 4, 32}` equals the analytic sum.
- **Planted bias:** form `r_i` with `p̂` in solid angle and `q = P/A` (area). This must fail TOST at δ = 0.2 %.

**U4 — Light-tile marginal.** Over ≥ 10⁴ frames, a χ² test of candidate `(L, cell of z)` counts against `pmf[L]·p_A` (T9a style), per screen tile. Also assert that the code path for `p1` never reads tile contents.

**U5 — Light-ending shift invertibility and Jacobian reciprocity (T3/T4 restricted to d ≥ 2 light-ending paths).**
- For each case (a)–(e), (n) and each light type:
  - `T⁻¹(T(x̄)) == x̄`, comparing `(lightId, u, v)` / `(inst, prim, bary)`, lobes and k;
  - `J_{k→j}·J_{j→k} = 1 ± 1e-4`.
- Self-shift (same pixel, same frame): `F` unchanged, `J == 1` (restir-practice §3.11 test 1; jitter off).
- Temporal self-shift with a light that moved rigidly but whose power is unchanged: `J == 1`.
- With power doubled: `J == pmf_t/pmf_{t−1}`.

**U6 — `p̂_←` two-way consistency on light-ending paths (T6b).**
- `a = p̂_j(x_j)/J_{j→c}` vs `b = p̂_j(T_{c→j}(y))·J_{c→j}`.
- Planted controls: stale base light pdf in case (d) (C2); spot `S` or one-sidedness evaluated at the *base* direction; `p2` not updated in case (b)/(c); `J` without the `p^y_k` factor when MIS is off (C3).

**U7 — Change-of-variables identity (T5)** restricted to `Ω_{d,NEE}`, `Ω_{d,BSDF_TRI}` and `Ω_{d,BSDF_ANALYTIC,L}` separately.

**U8 — Planted-bias ladder on C0c–C0e (Stage B).** Each must be *detected* (gate fails) with the minimum detectable bias below the gate δ:
1. ω1 < 1 for delta (C1);
2. stale light pdf (C2);
3. the `p^y_k/p^x_k` ratio applied only inside the MIS branch (C3), run with MIS off;
4. `J = t_x²/t_y²` for point lights;
5. spot `S` at the base direction;
6. one-sidedness ignored at the offset;
7. BSDF rays stopped by area lights while shadow rays ignore them (§2.6);
8. RIS ratio in mixed measures (U3);
9. failed neighbours removed from the normaliser;
10. tile-conditional pdf used in MIS.

**U9 — Mode A ≡ Mode B.** Our PT and our ReSTIR (unbiased preset) in Mode A and Mode B agree in expectation on scene (i) (Cornell box, rect light) and on a crossing scene (below), with no cap.

**U10 — Pass-through replay determinism (T2 extension).** A rect light is placed between `x_1` and a wall, facing the wall, and another light faces away.
- Base generation and full replay enumerate identical candidate labels and counts.
- The vertex sequence and RNG counters are identical with and without the lights present.
- Back-facing lights produce no candidates.

**U11 — Primary emission outside the reservoir.**
- (i) Camera facing an emissive quad (C0b): the pixel equals `L_e` exactly, and the reservoir never holds `d = 1`.
- (ii) Camera looking through a `visible_camera = True` rect light at a lit wall: pixel = `L_e,light + wall radiance`, compared against Cycles with `visible_camera = True`.

**U12 — Cycles transparent cap sanity.** A stack of 10 facing area lights, rendered in Cycles with cap 8 vs 1024. Record the difference, and assert that all Mode-B references use 1024.

**Scenes C0c–C0e (validation-harness §8) under ReSTIR.** Both `b = 0` (d = 2, pure DI: forced rc at the light, case (a)) and `b = 1` with a diffuse side wall (cases (b), (c), (e)). Run the Stage-B ladder (initial-RIS only → spatial-only offline → temporal static ensemble → full).

| ID | Analytic value (validation-harness §8, [INFERENCE] there) | Light-integration features exercised |
|---|---|---|
| C0c point P = 100 W, h = 1, ρ = 0.5 | `L_o = ρ P h/(4π² d³)` | delta NEE, ω1 = 1, J = 1 at the light vertex, `I/r²` recomputed at the offset |
| C0d spot s = 60°, β ∈ {0, 0.15, 0.5}, tilt 20° | `L_o = (ρ/π)(P/4π) S(α) cosθ/d²` | `S` at the offset direction, β = 0 hard-step guard, cone-edge discontinuity in spatial reuse |
| C0e rect a×b, power P, plus a variant with part of the receiver behind the light plane and a Mode-B variant | `L_o = (ρ/π)E`, `E = (L_e/2)Σ_i acos(û_i·û_{i+1})·n·normalize(û_i×û_{i+1})`, `L_e = P/(πab)` | one-sidedness at the offset, area-uniform J = 1, Mode A/B, crossing candidates |
| C0e-tri (new) | same with a two-sided emissive quad, `L_e = S` | two-sided, texture lookup at bary, BSDF_TRI rc case (d), C2 |

Gates follow validation-harness §9: global δ = 0.2 %, tile δ = 1 % for Stage B; δ = 0.2–0.5 % vs analytic.

---

## 9. Bias-trap checklist for lights (known + new)

| # | Trap | Status / fix |
|---|---|---|
| C1 | Delta-light NEE MIS computed with a pmf in a density formula (< 1) | gris-verify C1. Fix: `isDelta` flag → ω1 = 1 |
| C2 | Solid-angle light pdf measured at the base `x_{d−1}` reused at `y_{d−1}` (cases a, d) | gris-verify C2; confirmed in Falcor `Shift.slang:520`. Fix: recompute `p1^y`. EvanLuo42 does (`evalStoredEmissivePdf`) |
| C3 | `p^y_k/p^x_k` applied only when MIS is on | gris-verify C3. Fix: apply whenever ω_k is BSDF-sampled |
| X14 | J = 1 vs pdf ratio | §5.7: consistent only with matching target measure. PSS row recommended |
| N1 | Area lights occlude BSDF rays but not shadow rays | §2.6: all-or-nothing; non-occluding for Cycles |
| N2 | Cycles point-light pseudo-pdf `t²` used in any ratio | J = 1 for delta; ω1 = 1 |
| N3 | Tile-conditional light pdf in MIS or J | use `pmf[L]` (§3.9) |
| N4 | Previous-frame pmf / area / intensity used in the current-frame F, or cached pdfs across frames | recompute per frame; temporal J = `pmf_t/pmf_{t−1}` |
| N5 | Neighbour dropped from the pairwise normaliser when its shift fails | G-buffer-only validity (§5.6) |
| N6 | Cycles `transparent_max_bounces = 8` truncating Mode-B references | set 1024 or use Mode A |
| N7 | Multiple light crossings folded into one unlabelled candidate while storing one endpoint | label by lightId (§2.1) |
| N8 | V-buffer or shadow rays test analytic lights | triangles only; camera-visible lights in the resolve pass |
| N9 | RIS ratio `p̂/q` formed in mixed measures | §3.8 |
| N10 | Spot cone from a scaled light object (Cycles uses the inverse object transform, `spot.h:15-24`) | keep lights unscaled (JSON contract) |
| N11 | `M(B)` differs between NEE-time, hit-time and shift code | one deterministic function |
| N12 | Quantised `(u,v)`/bary decoded differently from the path used to compute `F`/`cachedJden` | quantise before evaluating, or f32 in validation builds (enhanced-verify O9) |

---

## 10. Open questions / [UNVERIFIED]

1. The exact RTXDI NEE-reconnection convention: orientation of the pdf ratio, and whether its target divides by the new pdf (§5.7). Irrelevant if not ported.
2. Enhanced's `32/B²` rounding and whether the light tile is re-picked per bounce (enhanced-paper §5.2). Only consistency matters.
3. EvanLuo42 line numbers (from WebFetch summaries), and whether its BSDF-hit MIS uses the lobe or the marginal BSDF pdf (`bs.pdf`).
4. Whether Cycles' area-light `SHADER_USE_MIS` needs `spread > 0` (cycles-verify §5 says yes; host `scene/light.cpp` was not re-read here). At spread = 180° it is on.
5. The Mode-B variance benefit on the WebGPU budget (the crossing loop is O(#area lights) per BSDF ray) is unmeasured.
6. Sphere or soft-disk point lights (r > 0) are deliberately excluded. Supporting them needs cone sampling, J with `p_A(z|y)/p_A(z|x)`, double-crossing labels in Mode B, and Cycles' soft-falloff disk semantics (cycles-verify C6).

---

## 11. Source list

- Cycles 5.1.2 kernel (local, read-only): `K/integrator/shade_light.h:19-104`, `K/integrator/intersect_closest.h:29-87, 251-255, 425-433`, `K/integrator/intersect_shadow.h:31-183`, `K/integrator/path_state.h:59-61, 110-235`, `K/integrator/shade_surface.h:249-250, 402-404, 578-582, 784-796`, `K/integrator/surface_shader.h:270-306, 377-381`, `K/light/light.h:25-30, 107-251, 258-421, 454-477`, `K/light/sample.h:24-104, 258-296, 322-357, 464-543`, `K/light/area.h:19-121, 250-326, 328-372, 390-462`, `K/light/point.h:18-80, 134-179`, `K/light/spot.h:15-31, 43-145, 206-249`, `K/light/distribution.h:45-58`, `K/light/triangle.h:50-120`, `K/light/common.h:62-70`.
- Blender 5.1.2 RNA introspection (this session): `transparent_max_bounces` default 8, range 0–1024; light and object defaults as listed in the header.
- Lin, Kettunen, Wyman 2026, ReSTIR PT Enhanced: `scratchpad/pages/enhanced/p-03.png` (§2.2–2.3), `p-04.png` (Eq. 2), `p-10.png` (§6.1), `p-11.png` (§6.2.3–6.3); text `scratchpad/restirpt_enhanced_2026.txt:431` (footnote 6), `:498` (d = 2).
- Enhanced supplemental (official PDF, harness-cached text `…/tool-results/bkusnhdet.txt`): §1 Eq. 1–4, **§5 RIS-based NEE (lines 326–348)**, §6 RR (349–381), §8 Alg. 1 (393–413).
- Lin et al. 2022 GRIS: via gris-math §5.2 (Eq. 49–51), §6.5 ("all light vertices are rough"), §6.6 (Eq. 52–54), §6.8.
- `github.com/EvanLuo42/ReSTIR-PT-Enhanced` (BSD-3), `Source/RenderPasses/ReSTIRPTPass/{Common/HybridShift.slang, Common/Lights.slang, Path/PathSampling.slang}`, tree listing via the GitHub API.
- `github.com/NVIDIA-RTX/RTXDI-Library` `Include/Rtxdi/PT/HybridShift.hlsli` (proprietary; semantics only).
- Project reports listed in the header (gris-math, enhanced-paper, reference-code, restir-practice, scene-io, cycles-conventions, webgpu-platform, validation-harness, critique, plus verifications).
