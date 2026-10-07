# ReSTIR PT Enhanced features and Mode-B ReSTIR: implementation contract for M6

Status: **normative for M6** (branch `m6-enhanced`, based on main f5c23fd = M0–M5.5, every gate green). This contract
**extends** `docs/decisions/restir-api.md` (M4) and `docs/decisions/restir-temporal-api.md` (M5); every rule there stays
binding unless an MD item below amends it and names the item. Metal quirks Q1–Q4 (`platform-lanes.md`) apply to every
new pass. Precedence (PLAN §4.2): PLAN §0–§2 > gap-* (gap-light for RIS-NEE / Mode B, gap-glass for glass, gap-rc for the
predicate, gap-temporal for dynamics) > *-verify > critique > original reports > this contract's own choices.

Normative math: `docs/math.md` §8 (RIS-NEE), §9 (M(B), Modes), §16 (path tree, crossings), §18 (case (d) on analytic
lights), §22 (pairing textures), §25 (RR), §26–§27 (confidence, duplication map, n_σ). Sections marked **[M6 addition]**
in math.md were added with this contract (appendix A).

Contents: §0 decisions · §1 feature designs · §2 data contracts · §3 pass graph · §4 tests (Gate 0) · §5 Gate 3/5 and
plants · §6 risks · §7 open questions · appendix A · Changelog.

---

## 0. Decisions at a glance

| # | Decision | Why |
|---|---|---|
| MD1 | **Every M6 feature is a pipeline variant or a pass of its own.** RIS-NEE (`RS_RIS_NEE`), Mode B / A′ crossings (`RS_MODE_B`), dual motion vectors (`RS_DUAL_MV`) and the duplication-map cap (`RS_DUPMAP`) are composer defines derived from the settings / light mode; the light tiles and the duplication map are new passes (`rs_light_tiles`, `rs_dupmap`). With every feature off, the composed WGSL of every M5 pass is the M5 text (comments and whitespace aside). The pairing maps are TS data (texture contents + `pairTexSize`), no shader change. | The M4/M5 Stage-B results and every cached PT / ReSTIR reference stay valid by construction (U-WGSL-BITS, U-M4-BITS, U-M5-BITS); PLAN §1.8 already makes structural features variant axes ("others compile lazily on toggle"); restir-api D1 already made Mode B a compiled-out define (`RS_MODE_B = 0`). B-8 lesson (plants on the default path changed Metal code) avoided. |
| MD2 | **The PT closure is untouched.** No file in the WGSL include closure of `passes/pt.wgsl` or the TS closure of `batch-run.ts` changes; M(B) enters the ReSTIR code by scaling p2 (MD5), not through `mis_M`. | Every cached PT reference (M4 `ptrefs`, M5 `ptrefs`, keyed by those closures) is reused without re-rendering (coordinator instruction); U-WGSL-BITS hashes `pt:batch` / `pt:frame`. |
| MD3 | **Pairing maps (M6):** Gaussian reciprocal maps by n_σ 2×2-block shuffles on a W-torus (link L = (yW + x) ≫ 1, every odd pass offset by (1, 1)), partner delta `d = wrap(b − a)`, `d(b) = −d(a)` stored explicitly, corrected `n_σ = ⌊σ²/2 + 1.46/σ − 1.76/σ² + 0.656/σ³ + 0.5⌋` (= 128 at σ = 16). Layer sizes `[254, 230, 210, 246, 238, 222]` (the plan's 254/230/210 first; layers 3–5 serve 6-slot / boost configurations). Deterministic PCG32 per (layer, σ, W); generated once per (σ, sizes) on the CPU (cached), uploaded `rg8sint` into the M4 texture. Setting `pairing: 'disk' \| 'gauss'`, `pairSigma` (16). The M4 disk maps stay the default of the M4/M5 validation presets and of `criteria2022` (PLAN §3: the 2022 baseline uses uniform-disk maps). | PLAN §5 M6, math §22 / §27, enh-verify C1; "replace the M4 maps without kernel changes" (D9). Every texel has a partner (W² even, links pair all texels). |
| MD4 | **RIS-NEE at x₁ only.** A per-frame pass draws `128 tiles × 1024` alias entries per ensemble member (i.i.d. `alias_sample` with hashes of `(runSeed ⊕ member·φ, t, tile·1024 + slot, STREAM_LIGHT_TILE)`; env entry included automatically). Each 8×8 member-local screen tile picks tile `hash(runSeed ⊕ member·φ, t, screenTile, STREAM_LIGHT_TILE) & 127`. At B = 1 the path tree draws M = 32 candidates: slot `h.x & 1023` uniform, light-local words from the same hash (the per-entry rule of `nee_draw`), target `p̂ = lum(f_all ⊙ Λ)` (visibility excluded, same-triangle samples have p̂ = 0), ratio `r = p̂/q` in μ, selection on the resampling stream, `W_NEE = (1/M)Σr/r_Y` (0 if no candidate). The selected endpoint then runs the unchanged NEE code (F, k*, visibility, fields). Shifts, replay and refresh never draw from tiles (PLAN rule 10). | PLAN §2 rule 2 (M(B) = 32 at B = 1, never tile-conditional pmfs), §5 M6 ("RIS-NEE light tiles at x₁", "env entries in the tiles"); math §8; gap-light §3.8–3.9; Enhanced §6.1 (128 × 1024, 8×8). Tiles per member so ensemble / chain replicates stay independent. |
| MD5 | **M(B) in ReSTIR = p2 scaled by 1/M(B).** `ω1 = M p1/(M p1 + p2) = p1/(p1 + p2/M)`; every ReSTIR MIS call whose x_{d−1} index can be 1 (path tree NEE and BSDF ends at B = 1; shift cases (a)/(f), (d) / (d-ana), (e) at d = 2; replay ∅ at b = 1) passes `rs_p2m(p2, B) = p2·(1/M)` (B = 1, RIS on) under `#if RS_RIS_NEE`; M = 32 is a power of two, so the scaling is exact and ω equals `M p1/(M p1 + p2)` bit for bit. Cases (b)/(c)/deep and the refresh have x_{d−1} with index ≥ 2 (M = 1) and are unchanged. `misM` in the config hash = 32 with RIS on. | One function M(B) (math §9, gap-light N11) without touching `lights/measure.wgsl` (MD2). |
| MD6 | **Mode-B ReSTIR (`BSDF_ANALYTIC`).** After the BSDF continuation at x_B (hit or escape), every rect / disk light crossed front-facing by the ray `(x_B, ω_out)` (D3 direction to the next vertex, or the sampled ω on an escape) before the continuation's distance is a candidate `(d = B + 1, BSDF_ANALYTIC, entry)` (Mode A′: only after a delta lobe, ω2 = 1). The candidate's own geometry: crossing point → light-local planar coordinates `xy ∈ [−1, 1]²` (`z = c + x·halfU·a_u + y·halfV·a_v`), `z_ids = z(xy)`, its own direction `ω_c = normalize(z_ids − x_B)`, its own BSDF query (`f/p` at ω_c; sampler weight for a delta lobe), `ω2 = mis_w2(analytic_area_p1(x_B, entry, z_ids), p2(ω_c)/M(B))`, `L_e = area_radiance(r, −ω_c)`. Its own k*: the tree pairs j < B are shared; pair B is re-tested with the candidate's own event at ω_c (FP boundary only); then the terminal pair `(x_B, z | RCK_LIGHT)`. Cases: **(d-ana)** k = d: rc = `RC_TAG_CROSS \| entry, bits(x), bits(y)`, `jDen = p(ω_c)·G(x_B→z_ids)` (n_L), visibility `visible(x_B, z_ids)` (B-5); **(c-ana)** k = d − 1: `rcWi = ω_c`, `jDen = p_{d−2}·G·p(ω_c)`, `rcRad = L_e`, `aux = p1`; **deep**: `rcRad = betaPost ⊙ (f/p)(ω_c) ⊙ ω2 L_e`; **∅**: full replay. End term of (c)/deep/∅ = `cross_end(x, ω, entry)`: the ray `(x, ω)` re-intersected with the light (one function for base, shift, replay and refresh). Crossings consume no bounce, no RNG dimension, no RR. RIS counter `(s<<20) \| (B<<12) \| (2 + entry)`. | PLAN §1.4 Modes, §5 M6 ("BSDF_ANALYTIC candidates, replay crossings, rc case (d), crossings on rays escaping to the env"), rule 3 (d); gap-light §2.1–§2.4, §5.4; math §16, §18. The per-candidate direction and BSDF query make the base F the shift's own formula (B-2 / D3); planar light-local coordinates need no inverse of the concentric map. |
| MD7 | **Mode-B shifts.** (d-ana): `z = z(xy)` of the destination frame's light record, `ω' = normalize(z − y_{d−1})`, O1, O2 `rcPairTest(y, eY, z as RCK_LIGHT)`, O0 + support, `F = Tp ⊙ (f/p)(ω')·ω2·L_e` with p1 recomputed at y_{d−1}, `jNum = p_joint·G(y→z)`, visibility `visible(y, z)`, one-sided back side ⇒ ZERO. (c-ana): the stored `ω_k = rcWi`, end term `cross_end(x_k, rcWi, entry)` of frame fs, end visibility = the base's spatially (copied x_k, ω_k, static geometry), from the refresh temporally (`endOcc`, as N1). ∅-ana: replay re-samples y_{d−1}, the candidate is `cross` of the SAME entry on the replayed continuation ray (else `SC_O0_TECH`), its own event at ω_c enters the last O3 pair. No J_P for crossings (BSDF-hit ending, PLAN rule 3). | PLAN rule 3 (d), rules 5–7; restir-api §3.7; gap-light §2.4, §5.4. |
| MD8 | **Mode-B temporal (R16).** The refresh translates `RC_TAG_CROSS` entries (TD7 maps; missing ⇒ undefined), evaluates D-BSDF crossing ends (`β_s ⊙ ω2 L_e` of `cross_end(x_{d−1}, sfxDir, entry)` under fs, a shadow ray only when the light MOVED) and B1-ana ends (`L_e`, `p1`, `SXS_VIS`, ray iff moved); `tsrc_load` renumbers `end.x` (and `rc.x` for k = d) and sets `endOcc` for B1-ana; the T3 write-back renumbers crossing entries and `endpointId`. The suffix cache of a crossing candidate stores `sfxDir = ω_c`, `sfxT` = the continuation's hit distance, flag `SFX_CROSS = 16`. | restir-temporal-api R16 ("Mode B temporal (crossing loop over lightsPrev)"); gap-temporal §5.5; TD7–TD10. |
| MD9 | **Light mode in the kernel.** `RestirKernelOptions.lightMode` accepts 'A', 'B', 'A′' (D1 lifted); `RestirParams.lightMode` = 0 / 2 / 1; `RS_MODE_B = 1` for B and A′. `RestirKernel.setLightMode(m)` recompiles the variant and resets history (config hash). The app's default light mode becomes **B** (PLAN §1.4 "B (product default once Gate 3.11 passes)") in the commit after rung 3.11 passes. | PLAN §1.4, §5 M6 exit. |
| MD10 | **Duplication map (biased).** `rs_dupmap` (after the frame's final reservoirs, 16×16 workgroups, a 32×32 shared tile of 64-bit seeds) writes per atlas pixel the count of pixels q ≠ p in the 17×17 window (same member, inside the tile) with `seed(q) == seed(p)`, both non-empty, into an arena region. The next frame's T1 (`RS_DUPMAP`) caps `c_p = min(c_Cap, c_prev)` with `c_Cap = cCap − (cCap − 1)·D^α`, `D = count(q′)/288`, α = 0.1 (math §27; f32, not truncated). Off in every unbiasedness unit (T16); on in the interactive preset (Enhanced: biased variant for real time). | PLAN §2 rule 13, §3 pass 6, §5 M6; enh §4; math §27. |
| MD11 | **Dual motion vectors (interactive only, unbiased).** `RS_DUAL_MV`: when every tap of TD12 fails (disocclusion), the previous-frame hit at the standard reprojection c₀ (the occluder at t−1) is projected with the current camera to s; the dual position is `p − (s − c₀)` (the occluder's motion applied to the pixel), and its centre + ring taps are tested with the unchanged validity rule. G-buffer and pick stream only (PLAN rule 9: sample-independent ⇒ unbiased). Counter `RSC_T_DUAL = 29`. **Amended (M6-11, DMV-1):** a dual q′ carries `c_p = min(DMV_C_CAP = 1, c_prev)` instead of `min(cCap, c_prev)` (tState flag `TS_DUAL_PICK`). | TD22, enh §6.7 (Zeng et al. 2021); M6-11. |
| MD12 | **Russian roulette.** Unchanged (D11): q_B = min(√max β, 1) at x_B, B > rrMinBounces, after NEE, before the continuation; NEE candidates carry 1/∏_{i ≤ d−2} q_i, BSDF endings (tri, env, **analytic crossings**) 1/∏_{i ≤ d−1} q_i; RIS-NEE multiplies `W_NEE`; replay / shifts / refresh never read `u_rr`. Verified by U-RR-M6 and toggled in rung 3.7. | PLAN rule 11, math §25. |
| MD13 | **Presets.** New: `offline-m6` = offline + gauss σ 16 + RIS-NEE (the rung 3.9 / 3.10 / 3.11 configuration); `full-m6` = full + gauss σ 16 + RIS-NEE (chains). `interactive` gains gauss σ 16, RIS-NEE, dual MV and the duplication map (cCap stays 5, TD-I1). Validation presets of M4/M5 (`initial`, `initial-rr`, `offline`, `criteria2022`, `temporal`, `full`) are unchanged. | Coordinator rule (no M4/M5 default changes); PLAN §3 mode list. |
| MD14 | **Rung 3.7 = one unbiased feature toggled per unit** on top of the M4/M5 rung it extends (sequential `offline` for spatial features, chains for temporal ones), reusing the cached PT references (same package, spp, B, seed 4001 / 7001). Glass (3.9), alpha (3.10) and Mode B (3.11) units run `offline-m6` (plus Mode-B chains), PT references new (seed 6001). | PLAN §7.1 ladder; restir-api D13 / §6.5. |
| MD15 | **Gate 5** (duplication map): chains with `dupmap` on vs the PT; per 16² tile the noise-debiased relative bias `b̂_t = sqrt(max(Δ_t² − SE_t², 0))/R̄_t`; budget: mean over tiles of b̂_t ≤ 3.25 % and the 99 % upper bound of the global relative |bias| ≤ 3.25 %, with the noise floor (mean SE_t/R̄_t) ≤ 1 % so the budget is resolvable; the same chains with `dupmap` off must pass Stage B (the config is unbiased, so the measured bias is the duplication map's). | PLAN §7.1 Gate 5, §5 M6 exit; enh §4.3 (3.25 % mean |bias|/ref, Kitchen). |
| MD16 | **Plants (Gate 3, predicted signs, TD29 rules):** U8-4 `J = t_x²/t_y²` (point lights, case (a), spatial), U8-7 BSDF rays stopped by crossed area lights (Mode B, path tree only), U8-8 RIS UCW in mixed measures (`W = W^RIS_A·p1_σ`), U8-10 tile-conditional pmf in ω1 (§5.3). Revisit of U8-2t (§5.3). | TD30 / R15 / B-9 deferrals; gap-light U8. |
| MD17 | **Seeds.** PT references of new units 6001 (re-run 106001); ReSTIR 6002 (106002); A/A 6502 / 6503; plants 6101 + i (4× PT 6201); pilots PT 6011, ReSTIR 6012; revised predictions on fresh seeds 6801 + i (E-18 rule). Units that reuse M4 / M5 PT references keep 4001 / 7001 on the PT side. | E6 / E-18 conventions. |

---

## 1. Feature designs

### 1.1 Pairing textures (σ = 16)

Generator `pairing.ts generateGaussLayer(W, σ, layer)`:
```
n   = nSigma(σ)                                   (corrected Eq. 3; 128 at σ = 16)
L[y][x] = (y·W + x) >> 1                           (two horizontally adjacent texels per link)
for s in 0 … n−1:  o = s & 1                       (odd passes: block grid offset by (1, 1), torus)
   for each 2×2 block {(2bx+o, 2by+o), (2bx+1+o, 2by+o), (2bx+o, 2by+1+o), (2bx+1+o, 2by+1+o)} mod W:
       apply a uniformly random permutation of 4 (PCG32 stream of (layer, σ, W))
pos[l] = the two texels of link l;  d = wrap(b − a) per axis (wrap(v) = v > W/2 ? v − W : v < −W/2 ? v + W : v)
delta[a] = d;  delta[b] = −d                       (explicit negation: |d| ≤ W/2 ≤ 127 fits rg8sint)
```
- Transform, partner, acceptance, reciprocity, slot layout: unchanged (§2.8 of restir-api). `RestirParams.pairTexSize`
  = the layer sizes of the active maps.
- Settings: `pairing: 'gauss'`, `pairSigma: 16`. Sizes `GAUSS_PAIR_SIZES = [254, 230, 210, 246, 238, 222]`.
- Debug views (PLAN §6 M6): view 471 + s `pair.offset[s]` (vec3: hue = atan2(dy, dx), value = |d|/(3σ)), 477
  `pair.recip` (code: 0 = partner(partner(p)) = p and A symmetric, 1 = no partner / off-tile, 2 = broken (never)).

### 1.2 Russian roulette

Unchanged (MD12). Rung 3.7 toggles it on top of rung 3.2 (`offline` + RR, rrMinBounces 1) — math §25: "RR stays off in
rungs 3.2–3.6 and is toggled in 3.7".

### 1.3 RIS-NEE light tiles at x₁

- `rs_light_tiles` (file `passes/restir/light-tiles.wgsl`, G1 empty, G2 = arena rw): dispatch `(16, 128, E)` × 64
  threads; tile entry `(m, tile, slot)` at `tilesBase + (m·128 + tile)·1024 + slot` = `alias_sample(lightsParams.cur,
  h.x, h.y)` (LIGHT_NONE when the slot has no entries), `h = pcg4d(runSeed ⊕ (memberBase + m)·φ, t, tile·1024 + slot,
  STREAM_LIGHT_TILE)`.
- Path tree (`#if RS_RIS_NEE`, B = 1 only, after `nee_draw` and before `nee_eval`): `ris_nee_select` (module
  `restir/ris-nee.wgsl`) runs the M candidates and replaces `ep`; the candidate's source weight becomes
  `rrInv·W_NEE` (exact multiply by 1 never happens: the code is absent without RIS).
- Candidate hashes `pcg4d(seed.x, seed.y, j, STREAM_RIS_NEE)` of the tree seed; the selection draw
  `rs_rand(key, RS_PASS_RIS_NEE = 2, (s << 20) | j)`.
- Marginal of a candidate: `pmf[L]·p_A(z|L)` (i.i.d. tile entries, uniform slot), so `q` is the marginal (rule 2);
  correlation between the 32 candidates of a pixel (same tile) does not bias RIS (GRIS Eq. 10, math §8).

### 1.4 Mode-B ReSTIR

Path tree (`#if RS_MODE_B`, after the continuation trace, before the BSDF endings), per crossed light (MD6):
```
crossAll = LP_CROSS_ALL;  crossDelta = LP_CROSS_DELTA && bs.is_delta           (A′)
ray = (x_B, ω_out, tMax = hit ? |x_{B+1} − x_B| : FLT_MAX)
for each rect / disk light i with a front-facing crossing at t ∈ (0, tMax), inside:
   xy, z_ids = z(xy), ω_c = normalize(z_ids − x_B)
   delta lobe : fac = bs.weight, ω2 = 1, ec = rc_event_bsdf(.., delta)          (k* ∈ {treeRc < B, ∅})
   else       : qc = bsdf_query(m, V, ω_c, ℓ); fac = rs_path_weight(qc, bs.weight, false)
                ω2 = crossAll ? mis_w2(analytic_area_p1(x_B, i, z_ids), p2m(qc.p_marg, B), B) : 1
   F = betaB ⊙ fac ⊙ ω2·L_e(z_ids → x_B)
   k*: treeRc < B (shared) | pair B with ec (re-test when B ≥ 2 and treeRc ∈ {0, B}) | terminal (x_B, z | LIGHT) | ∅
   (c-ana) rc = x_B, rcWi = ω_c, jDen = prevPJoint·G(x_{B−1}→x_B)·p(ω_c), rcRad = L_e, aux = p1, visibility = (tree pair B
           segment) — B-5
   (d-ana) rc = (RC_TAG_CROSS | i, xy), jDen = p(ω_c)·G(x_B → z_ids), visibility visible(x_B, z_ids)
   deep    rcRad = betaPostB ⊙ fac ⊙ ω2 L_e
   end words (RC_TAG_CROSS | i, xy); endpointId = i; endpointType = LT_RECT / LT_DISK
   suffix: sfx = x_B ids, sfxDir = ω_c, sfxT = continuation hit t (FLT_MAX on escape), flags BSDF_END | CROSS | valid
           | escape?, betaS = betaPostB ⊙ fac (k ≤ B−1; the x_{d−1} factor included, as for triangle / env ends) else 1,
           sfxP2 = p_marg(ω_c)
   stream: w = lum(F)·rrInv (·W_NEE never: a BSDF ending), counter (s<<20)|(B<<12)|(2 + i)
```
End term `cross_end(x, ω, entry, slot)`: the ray `(x, ω)` re-intersected with light `entry` of `slot`
(front-facing, inside, t > 0; no distance test spatially), `L_e` toward x and `p1` at the crossing point; a miss of the
re-intersection (an FP boundary at the light's edge) gives 0 (defined zero).

### 1.5 Glass and alpha in ReSTIR (rungs 3.9, 3.10)

No new code path (D13): glass is replayed through `bsdf_sample`, `G_T` never passes the predicate, rough `G_R` may be
`x_{k−1}` and `x_k` (gap-glass §6.2), side flips at x_k are re-evaluated (backfacing from the evaluating V), MASK cutouts
live in traversal and therefore in every visibility test of a reconnection segment. M6 makes them **gating**: T3 on
`t3_glass_256` (smooth + rough r 0.3 glass) with counters for pairs with ℓ_{k−1} = G_R, ℓ_k = G_R and side flips, and on
a new `t3_alpha_256` (cutout cards between the walls, counters for rc segments that cross a card's rectangle), LOGIC = 0;
Stage B on the glass scenes C0h–C0k, G1–G10, (vi) A/B, (vi-B), (xiv) glass (rung 3.9) and on (xii) + a large-foliage
scene (rung 3.10).

### 1.6 Duplication map, dual MVs

MD10, MD11. The duplication-map count also feeds view 478 `dup.count` and 479 `dup.cap` (written by `rs_debug_views`
from the arena region).

---

## 2. Data contracts

### 2.1 Constants (`restir/m6-types.wgsl`, mirrored in `layout.ts RS_M6_CONSTS`; included only by M6 code)

```
RSF_RIS_NEE 8192 · RSF_DUPMAP 16384 · RSF_DUAL_MV 32768 · RSF_PLANT_U8_RIS_MIXED 65536 (U8-8) ·
RSF_PLANT_U8_TILE_PMF 131072 (U8-10) · RSF_PLANT_U8_CROSS_OCC 262144 (U8-7)            (RSF_PLANT_U8_T2 = 1024 exists: U8-4)
RS_PASS_RIS_NEE 2 · RS_PASS_LIGHT_TILES 3 · RS_PASS_DUPMAP 4
STREAM_LIGHT_TILE 0x3c6ef372 · STREAM_RIS_NEE 0x1b873593
RS_TILES 128 · RS_TILE_SIZE 1024 · RS_SCREEN_TILE 8 · RS_RIS_M_MAX 32
SFX_CROSS 16 (word 27 bit 4)
RSC_T_DUAL 29 (header word 29: q′ found by the dual MV)
LP_CROSS_ALL 4, LP_CROSS_DELTA 8 (existing, crossings.wgsl)
```

### 2.2 RestirParams (B.3 amended: words 120–127 were `pad4`, `pad5`)

| Offset | Field | Meaning |
|---|---|---|
| 120 | `m6Base` | words[] index of the M6 arena region (0xFFFFFFFF: none) |
| 124 | `risM` | RIS-NEE candidates at x₁ (32; power of two) |

The WGSL struct keeps the names `pad4`, `pad5` (MD1: identical M5 text); `restir/m6-types.wgsl` reads them through
`rs_m6_base()` / `rs_ris_m()`.

### 2.3 Arena extension (appended after the M5 region; `layout.ts arenaM6`)

```
m6Base = 6·P·NS_alloc + (temporal ? 36·P : 0)          (words[] index, after the 64-word header)
  dup    P words         count of the 17×17 duplicates of the frame's final reservoirs (written by rs_dupmap)   [dupmap]
  tiles  E·128·1024 u32  alias entries of the frame's light tiles (written by rs_light_tiles)                [RIS-NEE]
arenaBytes += 4·(dupmap ? P : 0) + 4·(risNee ? E·131072 : 0)
```
No new storage binding in any pass (≤ 9 of 10).

### 2.4 Reservoir encodings

- `RS_TECH_BSDF_ANALYTIC` (2) records: end / rc triple `(RC_TAG_CROSS | entry, bits(x), bits(y))` with planar
  light-local `xy ∈ [−1, 1]²`; flags endpointType `LT_RECT` / `LT_DISK`, isDelta 0; endpointId = the alias entry
  (renumbered by the temporal write-back as analytic NEE entries, D5).
- Suffix flag `SFX_CROSS` marks an analytic crossing end (sfxDir = ω_c of the candidate).

### 2.5 Settings (`presets.ts RestirSettings`, appended)

```ts
pairing: 'disk' | 'gauss';  pairSigma: number;          // MD3 (default 'disk', 16)
risNee: boolean;  risM: number;                          // MD4 (default false, 32)
dualMv: boolean;  dupmap: boolean;                        // MD11, MD10 (default false)
plant.u8RisMixed?, plant.u8TilePmf?, plant.u8CrossOcc?    // U8-8, U8-10, U8-7 (U8-4 = plant.u8T2, existing)
```
`RestirKernelOptions.lightMode: 'A' | 'B' | 'A′'` (MD9). Defines: `RS_RIS_NEE = risNee`, `RS_MODE_B = lightMode ≠ A`,
`RS_DUAL_MV = dualMv && temporal`, `RS_DUPMAP = dupmap && temporal`; the pipeline cache key includes them.

---

## 3. Pass graph (additions to restir-temporal-api §4.1)

```
frame t: rs_primary → [rs_light_tiles (RIS)] → rs_initial × chunks → temporal (T1 reads dup[q′] under RS_DUPMAP)
         → spatial rounds → [rs_dupmap (dupmap; reads res[final], writes dup)] → rs_finalize (+ ensemble / debug)
```
`rs_light_tiles` and `rs_dupmap` are work units of their own (labels start with their kind, E1). Bindings:
`rs_light_tiles` G2 = [arena rw] (SB 1 + 0 + 1 + 1 = 3); `rs_dupmap` G2 = [resIn ro, arena rw] (SB 4).

---

## 4. Tests and Gate 0 (registry; `npm run validate -- --milestone M6`)

| ID | Where | Criterion |
|---|---|---|
| U-WGSL-BITS | tests/restir/wgsl-bits.test.ts | features off ⇒ composed WGSL of every M5 pass and the PT = the f5c23fd text |
| U-M4-BITS | restir-tframe (existing) | M4 rung outputs bitwise the pre-M5 goldens (interactive cases pinned to the M6-off settings) |
| U-M5-BITS | restir-m6.gpu.test.ts | temporal chains (static, light + camera + env motion, interactive-M5, Talbot): per-frame hashes = f5c23fd |
| T14 | tests/restir/pairing-gauss.test.ts | n_σ(16) = 128, n_σ(0.814) = 1; every layer an involution on the torus (d(a+d) = −d), no unmatched texel, |d| ≤ 127; per-axis σ of the deltas within 3 % of σ; normality (KS of d_x/σ vs N(0,1)) < 0.02; after all 8 dihedral codes × random offsets on 960×540 / 1024²: partner(partner(p)) = p |
| T3-3/M6 | restir-spatial (extended) | with the Gaussian maps: A(p,q) = A(q,p) bitwise, acceptedness equal on both sides, RSC_SLOT_MISMATCH = 0, partner(partner(p)) = p |
| U-PAIR-VIEW | restir-debug | views 471–477 write the expected hue / length / reciprocity codes at known pixels |
| U4-tiles | restir-m6 | χ² of tile entries vs the realized pmf (per member and pooled), ≥ 10⁶ draws; tiles differ across members and frames; identical across chunkings |
| U3-RIS | restir-m6 | mixed point + spot + rect + tri + sun + env scene, 64²: E[F·W] of the initial NEE d = 2 candidates (RIS M ∈ {1, 4, 32}) equals the PT's direct light (z ≤ 4); the mixed-measure UCW plant (U8-8) fails |
| U1-M | restir-m6 | ω1 (NEE-time, M = 32) + ω2 (hit-time, BSDF_TRI / BSDF_ENV / crossing, M = 32) = 1 to 1e-6 on random configurations; delta / Mode-A analytic ω1 = 1 exactly |
| U-RR-M6 | restir-m6 | RR + RIS on: W = Σw/lum F with the RR factors of §3.10 and W_NEE (candidate dump) to 1e-6 |
| U-RIS-1/M6 | restir-initial (variant) | RIS off ⇒ U-RIS-1 unchanged; RIS on ⇒ only d = 2 NEE candidates differ from the PT tree (masked compare) |
| U10-B | restir-m6 | pass-through replay: base path tree and a ∅ replay enumerate identical crossing labels / counts; vertex sequence identical with and without the lights; back-facing lights produce no candidate |
| U9-R | restir-m6 | ReSTIR initial (3.1) Mode A ≡ Mode B ≡ Mode A′ on a delta-free scene (means, z ≤ 4); mirror positive control A < B, A′ ≡ B |
| T3-M6 | restir-shift (variants) | T3-0 / T3-1 / T4 / T3-D LOGIC = 0, FP-BOUNDARY ≤ 1e-5: `t3_cases_256` with RIS on (every bin ≥ 10⁶); `t3_modeb_256` (Mode B; new bins d-ana, c-ana, none-ana ≥ 10⁶, deep-bsdf incl. crossings); `t3_glass_256` gating (G_R at x_{k−1} / x_k and side-flip counters ≥ 10⁶); `t3_alpha_256` (rc segments crossing a cutout card ≥ 10⁶) |
| T3-2/M6 | restir-temporal (variant) | Mode-B round trips under light motion (T3-2 harness, crossing entries renumbered): LOGIC = 0 on d-ana / c-ana / none-ana / deep bins ≥ 10⁶ |
| U-DUP-1 | restir-m6 | GPU duplication counts ≡ an f64 CPU count on a synthetic seed image (member borders, empty, background); cap formula vs CPU |
| U-DMV-1 | restir-m6 | dual MV: q′ is a function of the G-buffers + pick stream only (two runs with different reservoir contents give identical q′); disoccluded pixels behind a moving occluder get a valid q′ more often than without; every dual pick carries `TS_DUAL_PICK` and c_p ≤ 1 (M6-11), no flag with dual MVs off |
| U8-2t (Gate 0) | restir-temporal "U8 plant activity" | re-run with RIS on (E-22 revisit, §5.3) |
| T16 / T15 | every run | §5.5 |

GPU suites of Gate 0: restir-m6, restir-shift (M6 variants, one lock hold per variant ≤ 24 min, E-16), restir-spatial,
restir-temporal (M6 variant), restir-initial, restir-tframe (U-M4-BITS), restir-debug, restir-refresh, M3 regressions
pt / bsdf / lights / env-sampling / glass / pt-glass; cpu lane; python tests; package determinism of `make-m6.ts`; the M6
app smoke (views 471–479, Mode B default, interactive presets compile).

---

## 5. Gate 3, Gate 5 and plants

### 5.1 Units

Sequential units (compare.py stage B: TOST δ 0.2 % global / 1 % per 32² tile, Y/R/G/B, suite FWER, pilot sizing as M4
E7, one disjoint-seed re-run):

| Rung | Units (preset + toggle, package) | PT reference |
|---|---|---|
| 3.7 pairing | offline + gauss σ 16: (i), (v) V1 | cached M4 (4001) |
| 3.7 RR | offline + RR (rrMinBounces 1): (i), (iv) | cached M4 |
| 3.7 RIS-NEE | offline + RIS: (x) many lights, (iv) emissive mesh, (xiv) overcast + rect (env in the tiles), C0r irradiance (env only) | cached M4 |
| 3.7 all | offline-m6 + RR: (v) V2 | cached M4 |
| 3.7 temporal | chains `full-m6` (σ 16 + RIS) on m5s_cornell_i t ∈ {1, 24}; dual MV: `full` + dualMv on ixs_d_camera (its M5 test frames) | cached M5 (7001) |
| 3.9 glass | offline-m6: C0h, C0i, C0j, C0k, G1 ×2, G3, G4, G5, G5b (B), G6 (B), G6-neg, G7, G7 furnace, G8, G9, G10, (vi) A, (vi) B, (vi-B) (B), (xiv) glass | new (6001); (vi) A cached |
| 3.10 alpha | offline-m6: (xii), x10_foliage_256 (new large-foliage scene) | (xii) cached; new |
| 3.11 Mode B | offline-m6 Mode B: (i) B, m6_crossings_B_256 (new), C0o, (xiv) overcast + rect B; chains `full-m6` Mode B on m6_crossings_B t ∈ {1, 24}; dynamic ixs_b_area in Mode B (rung 3.6 statistics) | new |

Mode-B glass units of 3.9 ((vi) B, (vi-B), G5b, G6) also count for 3.11. Per scene the ladder stops at the first
failing rung (3.7 → 3.9 → 3.10 → 3.11 are independent features; a failure is investigated, never re-classified).

### 5.2 Gate 5

MD15 on chains of the `interactive` configuration made deterministic for validation (jitter iid, RR on, σ 16, RIS, dual
MV, boost 3, cCap 5 **and** cCap 20) with `dupmap` on, at m5s_cornell_i t = 24, m5s_glossy_v1 t = 24 and ixs_d_camera
(its last test frame); the dupmap-off twin of each must pass Stage B. Reported: mean b̂_t, global relative bias with its
99 % interval, the noise floor, per configuration; budget 3.25 %.

### 5.3 Plants (detected with ≥ 9/10 half-size repeats, PT A/A control ≥ 9/10, full comparison failing, AND the
predicted sign on the predicted region; revised predictions are re-rendered on fresh seeds, E-18)

| Plant | Flag | Unit | Prediction | Derivation |
|---|---|---|---|---|
| U8-4 J = t_x²/t_y² | RSF_PLANT_U8_T2 | u8_c0c_point_b0, offline (rung 3.2) | M_foot (pixels within 24 px of the light's foot point) **+**; global sign reported | Pairwise MIS toy (single point light, d = 2, F ∝ cos/t³): the planted J_{j→c} = ρ = t_j²/t_c² (reciprocal) changes the canonical estimate by the factor [1 + ρ(A + B)/(ρA + B)]/2 with B = A·ρ^{−1.5}: > 1 for ρ > 1. Pixels near the foot have partners farther away (ρ > 1 for every partner) ⇒ brighter; far pixels see ρ symmetric around 1 ⇒ second-order darkening (toy −1.2 % at ±20 % spread) — sign not predicted there. Confirmed by `tests/restir/plant-m6.test.ts` (f64 toy with the production MIS twins). |
| U8-7 lights occlude BSDF rays | RSF_PLANT_U8_CROSS_OCC | m6_crossings_B_256, Mode B, initial (rung 3.1) | global **−**; M_behind (pixels whose PT radiance drops ≥ 5 % when the lights occlude, from a mask render) **−** | The base path tree terminates a BSDF ray at its first crossed light, while NEE / shifts keep triangle-only visibility: every light-ending path with a segment through a light is lost from initial sampling ⇒ energy loss; ω-weights of NEE still assume both techniques (gap-light §2.6). |
| U8-8 RIS UCW in mixed measures | RSF_PLANT_U8_RIS_MIXED | u8_c0e_rect_b0, initial + RIS (rung 3.1) | global **+** (r²/|cosθ_z| > 1 at every receiver of that scene) | `W_NEE ← W^RIS_A·p1_σ` instead of `W^RIS_A·q`: the factor p1/q = r²/|cosθ_z| for area / triangle candidates (env: 1, delta: unchanged). The prediction is derived from the scene's geometry (light 1 m above the receiver plane: r²/cos ≥ 1 everywhere); `plant-m6.test.ts` evaluates the factor on the package. The gap-light §3.8 wording "p̂ in one measure, q in another" alone is NOT biased (any target is unbiased when the same ratio forms the selection and the UCW; CPU toy in plant-m6.test.ts); math §8 is amended. |
| U8-10 tile-conditional pmf in MIS | RSF_PLANT_U8_TILE_PMF | m6_tiles_glossy_256 (4096 small emissive triangles over a GGX r 0.3 floor), initial + RIS | global **+**; M_hl (glossy highlight of the emitter, top 10 % PT radiance) **+** | ω1 of the RIS-NEE candidate uses p1 with the tile multiplicity/1024 instead of pmf[L]; the BSDF-hit ω2 keeps the marginal. A selected light is present in its tile, so its conditional frequency is size-biased upward (≥ 1/1024 vs pmf ≈ 2.4e-4) ⇒ ω1 inflated ⇒ ω1 + ω2 > 1 ⇒ over-count, strongest where p2 is large (highlights). |
| U8-2t stale aux (revisit, Gate 0) | TP_U8_STALE_AUX | restir-temporal activity test, RIS on | active, bias below δ (unchanged) | RIS-NEE exists only at x₁; aux is consumed only in cases (b)/(c), whose x_{d−1} has index ≥ 2 (M(B) = 1), so the tiles do not change the path the plant perturbs (E-22's structural argument holds); the activity test re-runs with RIS on. |
| W × 1.003 / A/A | — | (i) offline-m6 | detected / pass | M4 calibration on the M6 configuration. |

### 5.4 T16 (per run)

Every M4 / M5 assertion plus: `dupmap` off in every unit except Gate 5; plants only where named; `dualMv` only in its 3.7
unit and Gate 5; pairing / RIS / light mode as named by the unit (meta.json `restir.settings` and `lightMode`); the PT
reference has the same light mode as the ReSTIR side; arena counters (incl. `RSC_T_*`, q0–q2 overflow) 0.

---

## 6. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Metal Q1 / Q2 in the larger Mode-B / RIS variants (path tree with two extra BSDF-query sites, shift with a new case) | Variant-only code; candidate loops keep only (entry, words, r, Σr); the T3 PLATFORM class and the production stress (VITE_STRESS) on the Mode-B variant; canaries. |
| R2 | Compile time of the variants (rs_initial: 4 bsdf_query call sites with RIS + Mode B) | Measured and recorded in budget.json; the inline budget of restir-api §4.5 is amended for the M6 variants (MD6). |
| R3 | Bit drift of the M4/M5 paths | MD1 + U-WGSL-BITS + U-M4-BITS + U-M5-BITS in Gate 0. |
| R4 | Gate cost (~45 Stage-B units, chains, Gate 5, plants) | Cached M4/M5 references (MD2, MD14), pilot sizing, 64² aggregates only when a pilot demands it (Q3 of restir-api), `--part` split, ≤ 12-min lock holds. |
| R5 | Crossing FP boundaries (light edges, ω_c vs ω_out) | Per-candidate geometry from stored ids (MD6); the re-intersection miss is a defined zero; T3 classifies by margin / edge. |
| R6 | Gate 5 metric noise | Debiased tile statistic and a noise-floor requirement (MD15). |

## 7. Open questions (my recommendation in bold; adopted unless the coordinator objects)

- **Q1** σ of the offline (6-slot) Gaussian maps: σ = 16 on all six layers (the interactive scale) vs σ ≈ 5.3 (the
  R = 10 offline disk equivalent). **σ = 16** for `offline-m6` (tests the shipped textures); `offline` keeps disk R 10.
- **Q2** Large-foliage scene for 3.10: Khronos Sponza (gitignored asset, 262k triangles, normal maps are M7) vs a
  generated foliage scene of the same class (hundreds of α-tested leaf cards, sun + env). **Generated** (deterministic,
  no asset dependency, flat shading; Sponza stays the manual/app check).
- **Q3** Gate-5 configurations: the interactive preset (cCap 5) is what ships; the paper's 3.25 % is at cCap 20.
  **Both are measured; the budget applies to both.**
- **Q4** Interactive default light mode B also for the PT path of the app (one global light mode). **Yes** (PLAN §1.4).

---

## Appendix A. math.md additions ([M6 addition])

§8: RIS-NEE realisation (tiles per member, candidate hashes, p̂ = 0 for same-triangle samples) and the corrected trap
statement (the UCW measure must match q; a consistent target in any measure is unbiased). §9: M(B) realised as p2/M in
ReSTIR. §16: Mode-B crossing candidates (MD6). §17: BSDF_ANALYTIC encodings, SFX_CROSS. §18: case (d-ana). §22: Gaussian
pairing generator (MD3). §27: duplication-map realisation (MD10).

## Changelog

(append-only; amendments made while implementing)

- **M6-1 (deep crossing suffix throughput, bug found by T3-2/M6).** The deep `BSDF_ANALYTIC` candidate stored `betaS`
  without the x_{d−1} factor; the Mode-B "moving + rotating crossing lights" round trips showed 17 746 deep-bsdf F
  mismatches. Fixed: `betaS = betaPostB ⊙ fac` (§1.4 / MD6 text amended). LOGIC = 0 afterwards.
- **M6-2 (inline budget, R2).** restir-api §4.5's per-pass inline budget is amended for the M6 variants: rs_initial with
  RIS-NEE + Mode B has 4 `bsdf_query` call sites (path tree NEE, RIS candidates, BSDF continuation, crossing candidate);
  replay 2, shift 1, t_forward / t_inverse 2 (tests/restir/layout.test.ts asserts the counts).
- **M6-3 (U8-8 derivation corrected before the measurement).** §5.3 claimed r²/|cos θ_z| > 1 at every receiver of
  u8_c0e_rect_b0; the light sits 0.5 m above the floor, so near it r²/cos < 1. The prediction is the sign of the
  contribution-weighted mean of r²/cos over the image: f64 quadrature (tests/restir/plant-m6.test.ts) gives 1.264 ⇒
  global **+26 %** (1541 of 8320 sampled pixels have a mean < 1). Prediction unchanged (+), derivation replaced; made
  before any GPU run of the plant.
- **M6-4 (U8-4 toy).** The pairwise-MIS toy on the package geometry with the production pairing maps and MIS twins
  (plant-m6.test.ts; 3 rounds × 6 slots, R 10) gives M_foot **+2.6 %** and global +1.0 %. M_foot + is the gating
  prediction; the global sign is reported (`detect`), as written in §5.3.
- **M6-5 (U8-7 region).** The M_behind mask (PT with the lights occluding) is not built; U8-7 gates on the global sign
  (−) only.
- **M6-6 (cached M4 PT references).** The M4 PT references of the non-env scenes in validation/out/m4/ptrefs carry the
  pre-M5 PT code hash (44d306…); the PT closure changed since (bf3620…), so the cache rule re-renders them once at seed
  4001 (sizes re-frozen in validation/out/m4/ptsize); the env scenes were already at bf3620. MD2 holds from f5c23fd on
  (the M6 work does not change the PT closure).
- **M6-7 (chain PT seeds).** The chain units (3.7 temporal, 3.11 chains / dyn, Gate 5) run through gate-m5's machinery
  (`runExternalChainUnits`) and keep its PT seeds 7001 / 107001 (cached M5 references on M5 packages; new packages
  render 7001 into validation/out/m6) instead of MD17's 6001; the chain seeds are M6's (6002 / 106002 / 6012).
- **M6-8 (T3-2/M6 bin rule).** The development test required ≥ 10⁵ round trips per crossing bin; Gate 0 now runs each
  Mode-B case in its own hold at 24 000 pairs (256²) and requires every ana / deep bin ≥ 10⁶ (§4 as written).
- **M6-9 (app).** The app's ReSTIR honours the light mode (A / A′ / B pipeline variants, `RestirKernel.setLightMode`),
  the restir panel gains the four feature toggles over the mode's preset, the app modes `offline` / `unbiased` use
  `offline-m6` / `full-m6`, and the HUD prints an M6 line (mode, pairing, RIS-NEE, dual MV, duplication map).
- **M6-10 (glass bias at rung 3.9: incoming direction after a delta event; amends restir-api D3 / Changelog B-2, MD1).**
  Rung 3.9 failed C0h (Δ_Y +0.120 %, MDB 0.015 %), G1 Glass node (+0.074 %) and G1 Principled (+0.052 %), all positive,
  concentrated at grazing incidence (C0h rows near the far edge +3.6 %, mid-image +0.1 %), every other glass unit
  passing (rough glass G3 / G4 / C0j, panes G5 / G5b, caustics G6, G7, G8). Bisection on C0h (12 frames × 16, seed
  7701/7702, vs the cached PT 6001; global / rows 40–63): offline-m6 +0.107 % / +1.64 %; M4 `offline` +0.108 % / +1.62 %;
  offline + σ 16 or + RIS-NEE identical; `rounds 0` (S = 32) +0.108 %; rung 3.1 `initial` (S = 1, no reuse) +0.120 % /
  +1.70 %; N = 2 (only the T,T path) +3.97 % in the far rows vs the closed form; the PT matches the closed form
  (|z| ≤ 1.6 per 16-row band). Hence not an M6 feature and not reuse: the path tree's own integrand differs from the
  PT's. Root cause (math.md#delta-incoming): the tree and replay took the incoming direction of the vertex after a delta
  event from positions (D3); the continuation ray starts at the Wächter–Binder offset origin, so inside a 0.02 m slab the
  position-derived segment is steeper by δ/h = 7.6e-4 in tan θ, and the delta sampler at the bottom face picks T with
  1 − F(θ') > 1 − F(θ) while the weight cancels only its own pick; near the critical angle ∂F/∂θ diverges (f64: +0.78 %
  at 80°, +4.0 % at 85°). Fix: `V = select(−ω_pos, −ω_sampled, is_delta)` in `path/pathtree.wgsl` and `path/replay.wgsl`
  (the PT's rule; no Jacobian, predicate or rcWi depends on such a V, and base and shift share the replay, so
  T∘T⁻¹ = id is unchanged). Regression test **U-RIS-1g** (restir-initial.gpu.test.ts): production tree ≡ PT sample per
  pixel on C0h and G1 Glass node (≥ 99.99 % within 1e-4, Σ within 1e-5); before the fix 99.875 % / 99.868 % (tree above
  the PT on 261 / 342 pixels, below on 66 / 4), Σ +1.19e-3 / +1.03e-3; after the fix it passes. Consequences: the
  composed WGSL of rs_initial(_dump), rs_spatial_replay / _shift and the temporal passes that include replay changes
  (U-WGSL-BITS goldens re-recorded); results change only where a delta lobe is sampled (glass, roughness-0 mirrors and
  specular), so U-M4-BITS / U-M5-BITS keep their goldens on the delta-free (i) cases and are re-recorded on the
  fixture-box cases (the box has a roughness-0 mirror); the PT closure is untouched (MD2: every PT reference stays
  valid). Gate harness: `--restir-seed-offset N` (ReSTIR seeds of the sequential units + N, PT references unchanged) for
  the fresh-seed verification.
  *Verification (M6-10).* `npm run validate -- --milestone M6 --part r39 --restir-seed-offset 900` (ReSTIR seeds
  6902 / 106902, cached PT references 6001; run m6-gate-r39-20261007-044151), 12 / 12 pass, no re-run needed:
  C0h Δ_Y +0.0042 % (MDB 0.017 %, worst tile +0.19 %; was +0.120 %), G1 Glass node −0.0035 % (MDB 0.006 %; was
  +0.074 %), G1 Principled −0.0016 % (MDB 0.007 %; was +0.052 %), and the previously passing C0i +0.0017 %, C0j −0.0002 %,
  C0k −0.0023 %, G3 −0.0000 %, G4 +0.0026 %, G5b +0.0048 %, G6 (B) −0.0059 %, G6-neg −0.0017 %, G7 furnace −0.0001 %.
  At 4× the gate size (48 frames × 16, seed 7801) C0h offline-m6 −0.001 % (z −0.3; far rows +0.019 % vs the closed form,
  z 0.5), rung 3.1 +0.001 %, G1 Glass node −0.001 %. Gate 0 touched: U-RIS-1g (new), restir-initial 26/26, restir-spatial
  18/18, restir-tframe 20/20 (U-M4-BITS: (i) unchanged, xq re-recorded), restir-m6 23/23 (U-M5-BITS re-recorded),
  restir-temporal 53/53, restir-refresh 23/23, restir-debug 16/16, restir-shift T3 t3_cases_256 and t3_glass_256 at an
  8-min budget (LOGIC 0, U-11 J 0 bad; t3_glass's ≥ 10⁶ grKm1 count needs the gate's 18 min), U-WGSL-BITS, cpu lane.
  Test-side copies of the rule: the T3-D f64 dual (tests/restir/rc-dual.ts) and the refresh test's full re-trace. The
  restir-initial frame-pass test failed on m6-enhanced before M6-10 (the interactive preset's M6 features vs an 'initial'
  batch rig) and now pins M6_OFF.
- **M6-11 (DMV-1: confidence of a dual-MV q′; rung 3.7 `ixs_d_camera_256@3.7-dualmv-f40` failure, m6-gate-r37-20261006-210627).**
  - **Symptom.** f40 failed `tost_masks` (M_disocc) and `tost_tiles` on all channels: run 1 M_disocc −0.90 %, re-run
    (disjoint seeds) +1.85 %, multiplier 3.2 at R = 1808. Every other test frame passed.
  - **Not a bias.** The q′ rule reads only the G-buffers, both cameras and the pick hash, and T4 evaluates π_p at the
    stored q′ (same mapping). A powered run on fresh seeds (906002, R = 6000, same estimator) passed every TOST:
    global Δ_Y +0.008 % (MDB 0.048 %), M_disocc +0.18 % ± 0.83 %, worst tile −0.17 %, multiplier 1. Only `chi2_red[R]`
    failed (2.73 > 2.46), a heavy-tail symptom. Pooled over the three disjoint pre-fix seed sets (R = 9616): M_disocc
    +0.29 % ± 0.68 %, global +0.014 % ± 0.011 %.
  - **Cause: variance.** f40 ends ix-d's fastest segment (0.022 per frame, frames 24 → 40) at the camera's closest
    approach. For 16 frames the camera moves left, so a strip left of the short box's silhouette (x ≈ 130–150,
    y ≈ 140–240) is disoccluded every frame. Dual MVs give ~half of those pixels a q′ about 4–5 px away on the floor,
    across the box's contact-shadow gradient. That history is converged elsewhere (c_prev ≥ 20) and was reused with
    c_p = 20.
  - **Mechanism.** The temporal candidate contributes its own estimate × c_p·r/(r + c_p) ≈ min(r, c_p), where
    r = p̂_q(Y)·J/p̂_q′(X_p) (Talbot form; contribution MIS behaves alike). For a standard q′, r ≈ 1. For a dual q′,
    r is broadly distributed, so outliers are amplified up to 20-fold. The amplified records then live on through
    temporal and spatial reuse along the strip.
  - **Measured (512 paired chains, f40).** The inverse shift of X_c is undefined for 9.7 % of dual picks (OCCLUDED
    7.4 %, O2 1.8 %) versus 0.6 % of standard ones. Dual-picked pixels carry 67 % of the per-chain M_disocc variance.
    The per-chain M_disocc median is 8 % below the reference, with a right tail up to 23× the mean in the gate
    re-run.
  - **What did not help.** Talbot MIS on dual picks only (per-pixel, unbiased) left the variance unchanged (sd 0.117
    vs 0.121). The hand-over of contribution MIS is therefore not the driver; the confidence is.
  - **Fix.** For a dual pick, `c_p = min(DMV_C_CAP = 1, c_prev)` (tpick.wgsl / t-classify.wgsl, `RS_DUAL_MV` only;
    tState flag `TS_DUAL_PICK = 65536`). The cap is a constant applied by a G-buffer-only predicate, so the MIS
    partition stays sample-independent (math.md §26 [M6 addition]).
  - **Effect at f40** (per-chain relative sd, R-independent):
    | | M_disocc | M_edge | global |
    |---|---|---|---|
    | before | 0.453 | 0.079 | 0.0089 |
    | after | 0.062 | 0.025 | 0.0039 |
    | M5 without dual MVs | 0.061 | 0.025 | 0.0038 |

    The other frames are unchanged or lower and match M5 (f32 edge 0.028 → 0.023, f48 disocc 0.027 → 0.021).
    Diagnostic with paired seeds (512 chains): per-chain M_disocc sd 0.121 → 0.023 (cap 4: 0.039; dual MVs off: 0.045,
    one outlier chain).
  - **After-fix powered run** (fresh seeds 916002, R = 6000, all 8 test frames, PT references as the gate): every frame
    passes, multiplier 1. f40: Δ_Y +0.014 % (MDB 0.022 %), M_disocc +0.16 % ± 0.09 %, worst tile −0.01 %, chi2_red 0.80.
    The other frames: |Δ_Y| ≤ 0.006 %, MDB ≤ 0.028 %.
  - **Test changes.** U-DMV-1 now also asserts that every dual pick is flagged with c_p ≤ 1 and that no pixel is
    flagged with dual MVs off. The diagnostic is `validation/gpu-tests/diag/dmv-f40.gpu.test.ts` (node-dawn, skipped
    unless `VITE_DMV_DIAG`; `VITE_DMV_OLD=1` / `VITE_DMV_CDUAL=c` patch the cap).
  - **Dual MVs off.** The composed WGSL is unchanged (U-WGSL-BITS).
- **M6-12 (Gate 5 metric: the chain ensemble stores sums; run m6-gate-gate5-20261007-115614).**
  - **Symptom.** All six Gate 5 verdicts failed with mean tile |bias| ≈ 25 000 %, global ≈ 6 560 000 % and noise floors
    9–46 %, nearly identical across scenes, c_cap and frames, while compare.py on the same chains gave sane Stage-B
    numbers (e.g. m5s_cornell_i cCap 5 Δ_Y +0.161 %).
  - **Root cause.** `dup_bias.py` read `ensemble.npz` (restir-chain-run.ts) as per-chain MEANS, but the file holds
    per-chain tile and image SUMS (`stats.replicates_from_sums`, README). Every 16² tile was 256× and the image 65 536×
    the PT; the "noise floor" was the chain SE in sum units (256× too large) over the PT mean. Its unit test wrote means
    into the npz, so the bug was never exercised.
  - **Fix.** `dup_bias.py` now reads both sides with compare.py's loaders (`compare.discover` + `load_replicates`), so
    any layout compare.py accepts works and the global Δ is compare.py's to 1e-15. Several `--ours` / `--ref`
    directories (disjoint seed sets) are pooled. Bad input exits 2 instead of printing a number.
  - **Regression test.** `validation/tools/tests/fixtures/gate5_real` is a 32² crop of the real run (32 chains of the
    cornell cCap 5 ensemble.npz, 4 PT batch PFMs; `make_fixture.py` regenerates it). `test_dup_bias.py` asserts that
    the global Δ and its SE equal compare.py's report on the fixture, that the tile means equal a hand reduction
    (sums / 256), and that the synthetic set (now written in the writer's sum format) resolves a planted 5 %. The old
    tool fails all three tests.
  - **Harness.** `npm run validate -- --milestone M6 --part gate5 --reuse-run DIR` recomputes the Gate 5 verdicts from
    an earlier gate5 run's summary.json, chains and PT references without rendering. The dupmap-ON Stage-B / dyn steps
    are informational: the duplication map is biased by design, so they no longer fail the run. MD15 gates on
    dup_bias.py and the OFF twins.
  - **Recomputed Gate 5** (run m6-gate-gate5-20261007-154618, reusing m6-gate-gate5-20261007-115614; first chain set
    and its PT reference; budget 3.25 %):

    | scene, t | cCap | R | mean b̂_t | p90 b̂_t | global | 99 % bound | noise floor | twin (Stage B) | Gate 5 |
    |---|---|---|---|---|---|---|---|---|---|
    | m5s_cornell_i, 24 | 5 | 2608 | 0.36 % | 0.92 % | +0.161 % | 0.170 % | 0.055 % | pass (−0.0006 %) | pass |
    | m5s_cornell_i, 24 | 20 | 8576 | 0.76 % | 1.95 % | +0.234 % | 0.241 % | 0.042 % | pass (+0.0004 %) | pass |
    | m5s_glossy_v1, 24 | 5 | 8864 | 0.22 % | 0.72 % | −0.126 % | 0.135 % | 0.049 % | pass (+0.004 %) | pass |
    | m5s_glossy_v1, 24 | 20 | 9648 | 0.75 % | 2.53 % | −0.124 % | 0.136 % | 0.056 % | pass (+0.009 %) | pass |
    | ixs_d_camera_256, 64 | 5 | 256 | 0.26 % | 0.87 % | +0.065 % | 0.095 % | 0.20 % | pass (+0.009 %) | pass |
    | ixs_d_camera_256, 64 | 20 | 496 | 0.89 % | 2.26 % | +0.262 % | 0.296 % | 0.21 % | pass (−0.001 %) | pass |

    The confirmatory re-run chain sets (seeds 106002, PT 107001) agree: mean b̂_t within 0.03 % of the first set. Pooled
    first + re-run gives 0.37 / 0.76 / 0.22 / 0.75 / 0.29 / 0.88 %. The twins on the same tool read 0.015–0.080 %, which is
    the debiased statistic's residual at this noise. Over all 8 ixs_d_camera test frames the largest mean b̂_t is 1.06 %
    (cCap 20, f40), the largest 99 % bound 0.39 % (cCap 20, f32). The noise floor is 0.04–0.21 % against the 1 % limit,
    so the budget is resolved with ≥ 5× margin and no further chains are needed.
