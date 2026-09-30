# ReSTIR PT core API: implementation contract for M4 (spatial, unbiased)

Status: **normative for M4** (branch `m4-restir`). Implementer agents build from this document in parallel; anything it
does not pin is the owning work package's choice, as long as it changes no interface below. A change to an interface
here is a **contract amendment**: the coordinator edits this file first, then the owners adapt.

Normative math: `docs/math.md` §1 (indices), §8–§9 (measure, MIS), §13–§22 (pdf conventions, support indicator, RNG,
path tree, reservoir, Jacobian, rc predicate, O0–O3, visibility, paired reuse), §25 (RR), §26 (confidence). Sections
marked **[M4 addition]** in math.md were added together with this contract (appendix A lists them). Style and
conventions follow `docs/decisions/bsdf-api.md`; that document stays authoritative for the BSDF itself.

**Precedence** (PLAN §4.2): PLAN §0–§2 > gap-* (gap-rc is normative for rc/O0–O3/paired reuse) > *-verify > critique >
original reports > this contract's own choices. Every place where this contract picks between sources, or pins
something no source pins, is listed in §0 with its reason.

Contents: §0 decisions · §1 work packages and module map · §2 data contracts · §3 function contracts · §4 pass graph,
bindings, dispatch, submits, pipelines · §5 RNG/seed contract · §6 tests and gates · §7 risks and open decisions ·
appendix A (math.md additions) · appendix B (normative WGSL blocks).

---

## 0. Decisions at a glance

Each item is binding. "Why" gives the source or the reason; "(math §x)" marks where it is recorded in math.md.

| # | Decision | Why |
|---|---|---|
| D1 | **Mode A only** for ReSTIR in M4. `RestirKernel` rejects light modes B and A′ (error). The Mode-B path-tree branch (`BSDF_ANALYTIC` candidates, crossings) is a compiled-out stub (`RS_MODE_B = 0`) with a fixed signature for M6. | PLAN §5 M6 "Mode-B ReSTIR"; A′ also needs `BSDF_ANALYTIC`. |
| D2 | **`initSeed` is 64 bits** (two words). The reservoir stores `seed.x, seed.y`. | M3c finding (math §15); math §17 said u32 (amended). |
| D3 | **Same-formula directions.** Every direction between two stored vertices is `normalize(pos(b) − pos(a))` from positions rebuilt from ids, in the base path, in replay and in every shift: incoming directions, the direction used for p̄/p_joint in the predicate and in jDen/jNum, and the stored `rcWi` for continuing `x_k`. Sampled direction bits are used only to trace the ray and for the escape direction to the env (`BSDF_ENV`). Throughput factors of the base path may use the sampler's `weight`. | gap-rc §3.1/§8.3 strengthened: base, replay and inverse shift then evaluate the predicate on identical inputs, so FP-boundary flips can only come from inter-pipeline code generation. (math §16) |
| D4 | **NEE endpoints are stored as the sampler's own light-local coordinates** `(entry, a, b)`: area lights and emissive triangles store `bits(u01(h_l0)), bits(u01(h_l1))` (for triangles the `(u₁,u₂)` of the area-uniform map, **not** barycentrics); delta lights and the sun store `(0, 0)`; the env stores `((i<<16)|j, h2)`. The shift re-evaluates the endpoint with the PT's own per-entry function (`light_sample_entry` / the env cell eval), so base and shift compute `z`, Λ, q, p1 bit-identically. | PLAN §1.9 light-local `(lightId,u,v)` and env `(ENV,(i<<16)|j,(du16<<16)|dv16)`; exact reproduction without an inverse map. (math §17) |
| D5 | **`endpointId`** (reservoir word 39, derived from the endpoint triple when the candidate is written): analytic light = its alias entry index in the reservoir's frame; emissive triangle = `primId` (for NEE triangle endpoints `emissive_tri(entry − nAnalytic).x`); env = `RS_ENV_ID`. M5 renumbers analytic entries on temporal selection (PLAN rule 4). | PLAN §1.9 field list, rule 4 "light indices renumbered". |
| D6 | **Cases (b) and (c) re-evaluate the end term** at the copied `x_{d−1}` from the stored endpoint (`nee_eval`, `tri_emission`/`tri_light_p1`, `envRadiance`/`p1Env`). The stored `rcRadiance` (Λ_x resp. L_e) and `aux` (p1) are still written (PLAN §1.9) and a debug check asserts they match to 1e-6 relative. | Removes any staleness class; within a frame the values are bit-identical anyway. |
| D7 | **Status words.** Slot J word: `VALID` = finite J > 0 **and** lum(F·J) > 0; `FAILED` = 0 (undefined shift, **or** defined with F = 0: occluded, back-facing, f = 0, L_e = 0, J guard); `NOT_ACCEPTED` = `0xFFFFFFFF`; `PENDING` = `0xFFFFFFFE` (transient, must never reach resample). The detailed outcome goes to a separate per-slot code word (§2.6). | math §22 "FAILED: undefined or p̂ = 0"; the numbers are identical, the code word keeps the debug distinction. (math §20, §22) |
| D8 | **Pair acceptance is evaluated once per pair**: the thread of `min(p, q)` evaluates `A0(G[min], G[max])` and writes both slots' J words; a pixel whose partner is invalid writes only its own `NOT_ACCEPTED`. `A0`'s `z` is the camera distance ‖x₁ − x₀‖, stored as f32 bits in `rsVbuf.w`. | math §22, gap-rc §7.1; "z = camera distance" pinned to the stored value. (math §22) |
| D9 | **M4 pairing maps** are uniform-disk involution textures in the M6 format (`rg8sint` layers of a `texture_2d_array<i32>`, per-(frame, round, member, slot) dihedral transform + offset). Radius R = 30 px interactive (σ-equivalent 16), R = 10 px offline (Falcor offline preset). The M6 σ = 16 Gaussian textures replace the layers without kernel changes. | PLAN §5 M4/M6, math §22 pairing-textures, reference-code §1.10. |
| D10 | **Offline mode** = S = 32 path trees per pixel streamed into one reservoir (two-level RIS, `W = Σw/(S·lum F_Y)`), 3 rounds × 6 slots. Trees are split across dispatches; `wSum` persists in the reservoir; the RIS counter is `(tree<<20)|(B<<12)|slotInVertex` so any chunking is bitwise identical. | PLAN §1.8, §3 Offline, math §16. |
| D11 | **Russian roulette** (rung 3.1b, interactive): exactly the PT's rule `q_B = min(sqrt(max_c β_c), 1)` on the RR-free prefix throughput, tested at `x_B` for `B > rrMinBounces`, after NEE and before the BSDF continuation. Resolves math open item 34 for ReSTIR by reusing the PT's (already validated) formula. | PLAN rule 11, math §25. (math §25, open item 34) |
| D12 | **`rs_primary` + `rs_initial` are separate passes** of the ReSTIR graph (PLAN §3 steps 1–2). `rs_primary` writes its own ReSTIR G-buffer textures; the M1 `primary` pass keeps running in the app for picking and G-buffer debug views (a duplicate camera ray in interactive ReSTIR mode, accepted for M4, see R10). | PLAN §3; minimal change to the M1 renderer. |
| D13 | **Gate 3 in M4 includes (vi) Mode A and (xii)** at rungs 3.1/3.1b/3.2 (PLAN §5 M4 exit lists (i)–(vi), (xii)). The ReSTIR code has no glass or alpha special cases: glass is replayed through `bsdf_sample`, G_T never passes the predicate, MASK cutouts live in traversal. Rungs 3.9 (glass) and 3.10 (alpha) in M6 add: (vi) Mode B, the glass Stage-B scenes (C0h–C0k, G1–G10), rough-glass G_R reconnection and side flips in T3 as gating, the (xiv) glass variant, alpha cutouts on reconnection segments in T3, and Sponza-foliage. Rough-glass T3 fixtures run in M4 **reported, not gating**. | Resolves PLAN §5 M4 "(i)–(vi)" vs §5 M6 "3.9 glass / 3.10 alpha"; see §6.5. |
| D14 | **Stage-B runs in M4 use sequential independent frames** through the batch accumulator (a batch = mean of F frames). Ensemble mode is built, tested bitwise against sequential frames, and exercised end-to-end on one gating unit via `ensemble.npz`. Temporal ensembles are M5. | validation-harness §3.6 design 2 is valid for temporal-off rungs; cheaper and resolution-agnostic. |
| D15 | **Ensemble member tile = the scene's render resolution** (default 256² per PLAN §3), E members in an atlas, `E·W·H ≤ 2²²`; E = 16 default, 64 only after the M0 allocation probe. | Generalises PLAN §3's 256² tiles to 512² scenes. |
| D16 | **Suffix cache** (for M5) is filled for every candidate with the definitions of §2.4; it is meaningful only when `k ≤ d−1`; spatial selection copies it verbatim. M4 reads it only in tests (U-SFX-1) and in the inspector. | PLAN §3 "fills the suffix cache", gap-temporal §5.5–5.6. |
| D17 | **Per-dispatch constants** go through a dynamic-offset uniform ring (`RsDispatch`, 256 B stride), not immediates. | PLAN §1.1 fallback "immediates → uniform ring"; no immediates in the code base yet. |
| D18 | **Queue append** is one `atomicAdd` per item (subgroup ballot is an M8 optimisation). | PLAN §1.1 fallback; correctness first. |
| D19 | **2022-criteria mode** is a runtime flag of the same `rcPairTest` (`RSF_CRIT_2022`): `D ∧ min(r(e_a), r(e_b)) ≥ α_min ∧ (kind(b) = ENV ∨ ‖b − a‖ ≥ d_min)`, `d_min = 0.02 ×` the shortest scene-bounds extent, light/env endpoints rough, `r(EV_NEE) = lobe_roughness(m, LOBE_NEE)`. Same O0–O3 machinery. | PLAN §3 mode list; enhanced-paper §3.1 (P-§7.2 d_min = 2%), GRIS §7.5. |
| D20 | **Optional symmetric `max(J, 1/J) > 11` rejection is off** (flag `RSF_J_REJECT` reserved). | enh-verify O6; not needed for unbiasedness. |
| D21 | **Single-engine validation build** (gap-rc §8.3 item 7) is **deferred**; M4 requires FP-BOUNDARY ≤ 1e-5 in the production multi-pipeline build (PLAN §5 M4 exit). | PLAN exit does not require it; cost is a large restructure. |
| D22 | **Empty and background reservoirs**: empty ⇔ `d == 0` (W = 0, F = 0, c = 1 on a hit pixel); background ⇔ `RF_BG` bit set, d = 0, c = 0. | math §16. |

---

## 1. Work packages, module map, ownership

### 1.1 Phases

1. **P0 foundation (owner WP-A, first merge, ≤ 1 day).** Lands exactly the files marked **P0** in §1.3 with the
   normative content of appendix B (constants, structs, reservoir layout and accessors, uniforms, the TS mirror),
   `bsdf_query` (§3.1, needed by WP-A and WP-B on day 1) and **compiling stubs** for every other shared WGSL module
   (API of §3 with trivial bodies, see §1.4). Nothing else. All other WPs branch from the P0 merge.
2. **WP-A … WP-E in parallel.** Each WP merges into `m4-restir` independently; the merge order is free after P0. A WP
   replaces the stub bodies of the files it owns; it never edits a file it does not own except the "may touch" lines of
   §1.2.
3. **Integration (WP-E leads).** Gate 3 pilots start as soon as WP-A (rung 3.1) and WP-C (rung 3.2) are merged.

### 1.2 Work packages

**WP-A: path tree, streaming RIS, reservoir, `rs_primary`/`rs_initial`/`rs_finalize`, kernel orchestration (rungs 3.1/3.1b).**
- Owns: all P0 files; `restir/endpoint.wgsl`; `path/pathtree.wgsl`; `path/length1.wgsl`; `passes/restir/primary.wgsl`,
  `initial.wgsl`, `finalize.wgsl`; TS `render/restir/{resources,kernel,presets,pass-primary,pass-initial,pass-finalize}.ts`;
  `validation/gpu-tests/restir-fixtures.ts`, `restir-initial.gpu.test.ts`; `tests/restir/layout.test.ts`.
- May touch (minimal, each change listed in its PR): `material/bsdf.wgsl` (add `bsdf_query`, §3.1; nothing else),
  `docs/decisions/bsdf-api.md` (document it), `lights/env-sample.wgsl` (split `env_light_sample` into draw + cell eval,
  bit-identical), `passes/pt.wgsl` (replace `pt_camera_lights` by the include of `path/length1.wgsl`, bit-identical).
- Provides stubs for: every shared module (P0). Early deliverable **A1** (≤ 3 days after P0):
  `endpoint.wgsl`, `rs_primary`, `rs_initial` with candidate dump, `rs_finalize` and a `RestirKernel` that renders
  rung 3.1 frames. WP-B and WP-E depend on A1.
- Tests: U-RES-1, U-EP-1, U-RIS-1…4, U-SFX-1, U-PT-BITS (§6.1).

**WP-B: shift core (rc predicate and k*, replay, reconnection, Jacobians incl. light/env cases, O0–O3, support
indicator, J guard, plants) + invertibility tests and the f64 dual.**
- Owns: `restir/rc.wgsl`, `path/replay.wgsl`, `restir/shift.wgsl`; `validation/gpu-tests/restir-shift.gpu.test.ts`
  (+ its own fixture helpers file `restir-shift-fixtures.ts`); `tests/restir/{rc-dual.ts,rc-dual.test.ts}`;
  `validation/scenes/make-m4.ts` (T3/case-coverage fixture packages).
- May touch: nothing else. (It consumes `bsdf_query`, `endpoint.wgsl`, the candidate dump from WP-A.)
- Early deliverable **B1** (≤ 2 days after P0): `rc.wgsl` complete (the path tree calls it). **B2**: `shift_hybrid`
  for `k = 2` and forced NEE (no replay) so WP-C's `rs_spatial_shift` produces real slots. **B3**: replay + all cases.
- Tests: T2, T3-0…T3-5, T3-D, T3-ENV, T4, T5/U7, U5, U-11, U-12, U-13, dense PSS sweeps, U-CASE (§6.1).

**WP-C: paired spatial reuse, queues, MIS, ensemble.**
- Owns: `restir/pairing.wgsl`, `restir/queue.wgsl`, `restir/mis.wgsl`; `passes/restir/{pair-accept,args,
  spatial-replay,spatial-shift,spatial-resample,ensemble-stats}.wgsl`; TS `render/restir/{stage-spatial,pairing,
  ensemble,npz}.ts`; `validation/gpu-tests/restir-spatial.gpu.test.ts`; `tests/restir/{pairing,mis-ref,mis,queue,npz}.test.ts`
  (+ `tests/restir/mis-ref.ts`).
- May touch: nothing else. `kernel.ts` imports `stage-spatial.ts` and `ensemble.ts` through the P0 stage interface
  (§4.6); WP-C never edits `kernel.ts`.
- Stubs consumed: `shift_hybrid` (P0 stub returns `FAILED`/`SC_O0_MISS`), so the spatial passes run and are
  unbiased (canonical only) before WP-B lands.
- Tests: T3-3/M4, T6(a), T6(b), T7, T14-M4, T17, U-MIS-1 (lum L = Σw), U-ENS-1/2, U-OFF-1 (§6.1).

**WP-D: debug views, pixel inspector, interactive integration.**
- Owns: `debug/restir-views.wgsl`; TS `render/restir/debug.ts`; `src/app/ui/panels/restir-panel.ts`,
  `src/app/ui/panels/restir-inspector.ts`; `validation/harness/m4-app-smoke.ts`; `validation/gpu-tests/restir-debug.gpu.test.ts`.
- May touch: `render/renderer.ts` (render mode `'restir'`: create/drive `RestirKernel.interactive()` after `primary`,
  register the views, timestamps), `src/app/integration.ts` / `src/app/ui/panel.ts` (mount the two panels, PT/ReSTIR
  toggle), `render/overlay.ts` + `debug/overlay.wgsl` (draw probe polylines), `render/probe.ts` (tag names).
- Stubs provided: hook functions of §2.11 (P0: empty bodies). Every pass owner calls the hooks at the points of §2.11.
- Tests: U-DBG-1…3 and the M4 app smoke (§6.1).

**WP-E: Gate 3 harness, Stage-B wiring, `validate --milestone M4`.**
- Owns: `render/restir/batch-runner.ts`; `validation/harness/{restir-batch-run.ts,gate-m4.ts}`;
  `tests/restir/gate-m4-config.test.ts`; the M4 sections of `docs/decisions/validation.md`; M4 rows of `validation/budget.json`.
- May touch: `render/batch-accumulator.ts` (add `runBatchUnits`, §4.4; `runBatch` unchanged),
  `validation/harness/harness.ts` (expose `renderRestirBatches`), `validation/harness/run-batches.ts`
  (`--kernel restir` and its flags, §6.4), `validation/harness/validate.ts` (M4 branch),
  `validation/tools/compare.py`/`stats.py` only for bugs found while wiring Stage B (reviewed by the coordinator).
- Stubs consumed: until WP-C lands, rung 3.2 units run the canonical-only spatial stub (must pass trivially; the
  plumbing is tested).

Dependencies: P0 → everyone; A1 → B (candidate dumps), E (rung 3.1); B1 → A (real k*); B2/B3 → C (real slots);
C → E (rung 3.2), D (interactive spatial); D needs A1 for the first views.

### 1.3 Module map (one owner per file)

```
src/core/shaders/
  restir/types.wgsl            P0/A  constants, status codes, small shared structs (appendix B.1)
  restir/reservoir.wgsl        P0/A  reservoir layout, bindings-by-define, plane accessors, ris_update (B.2)
  restir/frame.wgsl            P0/A  RestirParams, RsDispatch ring, atlas mapping, seeds, rs_rand, rs_path_* (B.3)
  restir/endpoint.wgsl         A     NEE endpoint draw/eval, emitter/env end terms, NEE visibility (§3.3)
  restir/rc.wgsl               B     vertex_from_ids, RcVertex/RcEvent, rcPairTest (Enhanced + 2022), primaryThreshold,
                                     rc_G, kstar_* (§3.4)
  restir/shift.wgsl            B     ShiftSrc/ShiftDst loaders, shift_hybrid (all cases, O0–O3, J, plants) (§3.7)
  restir/mis.wgsl              C     pairwise MIS terms, spatial resample kernel body, write-back (§3.8)
  restir/pairing.wgsl          C     dihedral transforms, pair_partner (§3.9)
  restir/queue.wgsl            C     queue append, 2D args, item decode (§3.9)
  path/pathtree.wgsl           A     unified path tree, candidates, deferred k*, streaming RIS, suffix cache (§3.5)
  path/replay.wgsl             B     replay_prefix (prefix mode and k = ∅ mode) (§3.6)
  path/length1.wgsl            A     camera-visible analytic lights + emissive x₁ + env miss (moved from pt.wgsl)
  passes/restir/primary.wgsl   A     rs_primary
  passes/restir/initial.wgsl   A     rs_initial (+ RS_DUMP_CANDIDATES test variant)
  passes/restir/finalize.wgsl  A     rs_finalize (batch accum | rsFrame | interactive colour)
  passes/restir/pair-accept.wgsl      C
  passes/restir/args.wgsl             C
  passes/restir/spatial-replay.wgsl   C  (calls shift_hybrid with RS_REPLAY = 1)
  passes/restir/spatial-shift.wgsl    C  (calls shift_hybrid with RS_REPLAY = 0)
  passes/restir/spatial-resample.wgsl C
  passes/restir/ensemble-stats.wgsl   C
  debug/restir-views.wgsl      P0/D  hook API (P0 empty bodies), view ids 400–499, probe tags 64–95 (§2.11)
src/core/render/restir/
  layout.ts                    P0/A  TS mirror of appendix B (word offsets, flags, codes, sizes)
  resources.ts                 P0/A  allocation of every M4 buffer/texture, bind group layouts G0–G3, ping-pong groups
  presets.ts                   P0/A  RestirSettings presets (§4.6)
  kernel.ts                    P0/A  RestirKernel (frame units, stage interface, interactive frame pass)
  pass-primary.ts, pass-initial.ts, pass-finalize.ts   A
  stage-spatial.ts             P0 stub/C  RestirStage for pair_accept → args → replay → shift → resample
  ensemble.ts, npz.ts          P0 stub/C  atlas, ensembleStats pipeline + readback, npz writer
  pairing.ts                   C     CPU disk-involution generator + TS mirror of the transforms
  debug.ts                     P0 stub/D  view registry entries, probe decoding for the inspector
  batch-runner.ts              E     WorkUnit → submit packing, per-batch readback (uses BatchAccumulator)
src/app/ui/panels/restir-panel.ts, restir-inspector.ts   D
validation/harness/restir-batch-run.ts, gate-m4.ts       E
validation/harness/m4-app-smoke.ts                       D
validation/scenes/make-m4.ts                             B
validation/gpu-tests/restir-fixtures.ts                  A   (others import; they add helpers in their own files)
validation/gpu-tests/restir-initial.gpu.test.ts          A
validation/gpu-tests/restir-shift.gpu.test.ts (+fixtures) B
validation/gpu-tests/restir-spatial.gpu.test.ts          C
validation/gpu-tests/restir-debug.gpu.test.ts            D
tests/restir/layout.test.ts                              A
tests/restir/rc-dual.ts, rc-dual.test.ts                 B
tests/restir/mis-ref.ts, mis.test.ts, pairing.test.ts, queue.test.ts, npz.test.ts   C
tests/restir/gate-m4-config.test.ts                      E
```

### 1.4 Stubs in P0 (exact behaviour)

| Module | Stub body | Consequence while stubbed |
|---|---|---|
| `rc.wgsl` | `rcPairTest` returns fail (margin 0); `kstar_nee` returns `B+1` (forced); `kstar_tree_pair` false; `kstar_bsdf_end` returns `treeRc` (0) | every NEE candidate is forced (case a/f), every BSDF end is `k = ∅`: valid, unbiased, maximal replay |
| `replay.wgsl` | returns `SC_O0_MISS` | every replayed shift FAILED |
| `shift.wgsl` | `shift_hybrid` returns FAILED `SC_O0_MISS`; `res_needs_replay` real (flags only) | spatial output = canonical sample (m_c = 1 for FAILED pairs): unbiased |
| `mis.wgsl` | resample copies resIn → resOut, shade = F·W | no reuse |
| `pairing.wgsl` | `pair_partner` returns invalid | every slot NOT_ACCEPTED |
| `queue.wgsl` | real (it is small; WP-C may refine) | – |
| `endpoint.wgsl` | real signatures, bodies may call `nee_sample` directly until A1 | – |
| `pathtree.wgsl` | writes an empty reservoir | rung 3.1 = L1 only |
| `restir-views.wgsl` | empty hook bodies | no views |
| TS stage-spatial / ensemble / debug | `frameUnits` return `[]` | – |

### 1.5 Rules of engagement

- One owner per file. A WP that needs a change in another WP's file files the request with the coordinator; if it is
  an interface change, the coordinator amends this contract first.
- Every WGSL module starts with the header comment convention of the code base (what it implements, `math.md#anchors`).
- New constants live in `restir/types.wgsl` (appendix B.1) and `layout.ts`; adding one is a contract amendment
  (append-only numbering, never renumber).
- GPU tests run on the Chrome lane (authoritative); dawn.node runs them as a pre-check only.

---

## 2. Data contracts

All per-pixel buffers are indexed by the **atlas pixel index** `ai = ay·atlasW + ax` (row 0 = top). In sequential and
interactive modes the atlas is the image (E = 1). Words are u32; floats are stored as `bitcast<u32>(f32)`.

### 2.1 Constants

Normative in appendix B.1 (`restir/types.wgsl`) and mirrored in `layout.ts`. Summary:
- Techniques (math §16): `RS_TECH_NEE 0, RS_TECH_BSDF_TRI 1, RS_TECH_BSDF_ANALYTIC 2, RS_TECH_BSDF_ENV 3`.
- Endpoint types = `LT_*` (`lights.wgsl`): `tri 0, point 1, spot 2, rect 3, disk 4, sun 5, env 7`.
- Lobe codes = `LOBE_*` (`bsdf.wgsl`): `D 0, S 1, G_R 2, G_T 3, NEE 4, NONE 5`; delta is a separate bit.
- rc-vertex kinds for `rcPairTest`: `RCK_SURFACE 0, RCK_LIGHT 1, RCK_ENV 2`.
- rc/endpoint word-A tags: surface `primId` (`< 0x80000000`); `RC_TAG_NEE = 0x80000000u | entry`;
  `RC_TAG_CROSS = 0xC0000000u | entry` (M6); `RC_ENV_DIR = 0xFFFFFFF1u`; `RC_NONE = 0xFFFFFFFFu`.
- `RS_ENV_ID = 0xFFFFFFFEu` (endpointId of the env).
- J-word statuses: `JW_FAILED = 0`, `JW_PENDING = 0xFFFFFFFEu`, `JW_NOT_ACCEPTED = 0xFFFFFFFFu`.
- Slot outcome codes `SC_*` (§2.6), counters `RSC_*` (§2.6), flags `RSF_*`/`RSD_*` (§2.9), pass ids `RS_PASS_*` (§5).
- Limits: `RS_MAX_D = 15`, `RS_MAX_SLOTS = 6`, `RS_MAX_TREES = 64`, `RS_MAX_ROUNDS = 4`, `RS_WG = 64` (1D kernels).

### 2.2 Reservoir record (validation layout, 160 B)

`array<vec4<u32>>`, record `i` = planes `i·10 … i·10+9`. Two physical buffers `resA`, `resB` (ping-pong).

| Plane | Byte | Word | Field | Type | Semantics |
|---|---|---|---|---|---|
| P0 | 0 | 0 | `W` | f32 | UCW (math §17); 0 for empty/background |
| | 4 | 1–3 | `F` | f32×3 | PSS integrand of the stored path **in this reservoir's own pixel domain**; p̂ = lum(F) |
| P1 | 16 | 4–5 | `seed` | u32×2 | 64-bit path seed of the tree that produced the path (D2); dupmap identity |
| | 24 | 6 | `flags` | u32 | bit table below |
| | 28 | 7 | `c` | f32 | confidence (math §26); 1 after initial on a hit pixel, 0 on background |
| P2 | 32 | 8–10 | `rcA, rcB, rcC` | u32×3 | rc vertex x_k (§2.3); `RC_NONE` for k = ∅ or empty |
| | 44 | 11 | `jDen` | f32 | the path's own joint-pdf denominator (math §18 table); 1 when unused |
| P3 | 48 | 12–14 | `rcWi` | f32×3 | ω_k at x_k (continuing: from positions, D3; case b: the NEE direction; case e: the **world** escape ω); 0 when unused |
| | 60 | 15 | `aux` | f32 | cases (b)/(c): p1 of the end term measured at x_{d−1} (0 for delta/sun); else 0 |
| P4 | 64 | 16–18 | `rcRad` | f32×3 | (b) Λ_x; (c) L_e or L_env; (deep) post-rc suffix incl. end MIS; else 0 (math §17) |
| | 76 | 19 | `wSum` | f32 | streaming-RIS Σw of the initial pass (persists across tree chunks, D10); diagnostic afterwards |
| P5 | 80 | 20–22 | `endA, endB, endC` | u32×3 | endpoint triple of the path (§2.3), for every d ≥ 2 path |
| | 92 | 23 | `lobeHist` | u32 | 4 bits per vertex x₁…x₈: `lobe | delta<<3` of the event leaving x_b; `0xF` = none. Base path only (not updated by shifts; debug/T3) |
| P6 | 96 | 24–26 | `sfxA, sfxB, sfxC` | u32×3 | suffix cache: x_{d−1} (`primId`, bits(u), bits(v)) (§2.4) |
| | 108 | 27 | `sfxFlags` | u32 | bit0 BSDF end, bit1 escape, bit2 valid (k ≤ d−1) |
| P7 | 112 | 28–30 | `sfxDir` | f32×3 | NEE end: ω_o at x_{d−1} (toward x_{d−2}); BSDF end: ω_{d−1} |
| | 124 | 31 | `sfxT` | f32 | BSDF end: t of the final ray to the first triangle (`FLT_MAX` on escape); NEE end: 0 |
| P8 | 128 | 32–34 | `betaS` | f32×3 | post-rc throughput: NEE end ∏_{j=k+1}^{d−2}, BSDF end ∏_{j=k+1}^{d−1} (f cos/p joint); 1 if k ≥ d−1 |
| | 140 | 35 | `sfxP2` | f32 | marginal BSDF pdf at x_{d−1} of the end direction (NEE dir or ω_{d−1}) |
| P9 | 144 | 36 | `nCand` | u32 | candidates streamed (all chunks) |
| | 148 | 37 | `selId` | u32 | RIS counter of the selected candidate (§5) |
| | 152 | 38 | `kMargin` | f32 | rcPairTest margin of the pair that decided k (0 if forced/∅) |
| | 156 | 39 | `endpointId` | u32 | D5 (stable light / emissive-triangle id of the endpoint; `RC_NONE` when empty) |

**`flags` bits** (appendix B.1 `RSF_*` are separate; these are `RF_*`):

| Bits | Field | Values |
|---|---|---|
| 0–3 | `d` | 2…15; **0 = empty** |
| 4–7 | `k` | 2…d; **0 = ∅** |
| 8–9 | technique | `RS_TECH_*` |
| 10–12 | endpointType | `LT_*` |
| 13 | isDelta | endpoint light is point/spot/sun |
| 14–16 | ℓ_{k−1} | lobe code at x_{k−1}; `LOBE_NEE` for forced NEE; `LOBE_NONE` for ∅ |
| 17 | δ_{k−1} | delta bit (always 0 for a passing pair; kept for T3 signatures) |
| 18–20 | ℓ_k | lobe at x_k; `LOBE_NEE` in case (b); `LOBE_NONE` for light/env vertices and ∅ |
| 21 | δ_k | |
| 22–23 | mode | 0 = A (M4); 1 = A′, 2 = B reserved |
| 24 | forced | k = d reached by forcing |
| 25 | bg | background pixel (no primary hit) |
| 26–31 | reserved | 0 |

**Per-case field contents** (letters of math §18; "–" = written as 0 / `RC_NONE` / 1 for jDen):

| Case | k | rcA–C | rcWi | jDen | aux | rcRad |
|---|---|---|---|---|---|---|
| (a) forced NEE, analytic/tri | d | `RC_TAG_NEE|entry`, u, v bits | – | 1 | – | – |
| (f) forced NEE, env | d | `RC_TAG_NEE|envEntry`, (i<<16)\|j, h2 | – | 1 | – | – |
| (b) N1 | d−1 | x_{d−1} ids | NEE dir (`nee_eval(...).dir`) | p^x_{d−2}(ω̃,ℓ)·G(x_{d−2}→x_{d−1}) | p1 | Λ_x |
| (c) B1 | d−1 | x_{d−1} ids | ω̃_{d−1} (tri end) / escape ω (env end) | p_{d−2}·G·p_{d−1}(ω_{d−1},ℓ) | p1 (tri: `tri_light_p1`; env: `p1Env`) | L_e / L_env |
| (d) emitter rc | d | emitter `primId`, bary | – | p^x_{d−1}(ω̃,ℓ)·G(x_{d−1}→z) | – | – |
| (e) env rc | d | `RC_ENV_DIR`, 0, 0 | world escape ω | p^x_{d−1}(ω,ℓ) | – | – |
| (deep) | ≤ d−2 | x_k ids | ω̃_k | p_{k−1}·G·p_k | – | betaPost ⊙ end term (MIS incl.) |
| (∅) | 0 | `RC_NONE` | – | 1 | – | – |

Writer rules: every writer writes **all ten planes** (unused words 0, jDen 1); no field is ever left uninitialised
(gris-verify O3). The WGSL accessors of appendix B.2 are the only way passes read or write records.

### 2.3 rc-vertex and endpoint encodings

| Kind | word A | word B | word C | Rebuild |
|---|---|---|---|---|
| surface vertex (rc at a scattering vertex, or a BSDF-hit emissive triangle) | `primId` | bits(bary.u) | bits(bary.v) | `vertex_from_ids(primId, u, v, fromPos)` (rc.wgsl); the hit barycentrics as returned by traversal |
| NEE light vertex, analytic area light or emissive triangle | `RC_TAG_NEE | entry` | bits(u01(h_l0)) | bits(u01(h_l1)) | `light_sample_entry(x, slot, entry, vec2f(u, v))` |
| NEE delta light / sun | `RC_TAG_NEE | entry` | 0 | 0 | same (u unused) |
| NEE env | `RC_TAG_NEE | envEntry` | `(i<<16)|j` | `h2` (`(du16<<16)|dv16`) | `env_light_sample_cell(slot, entry, i, j, h2)` |
| BSDF_ENV escape | `RC_ENV_DIR` | 0 | 0 | direction in `rcWi` (rc) or `sfxDir` (endpoint) |
| Mode-B crossing (M6) | `RC_TAG_CROSS | entry` | bits(u) | bits(v) | reserved |
| none | `RC_NONE` | 0 | 0 | – |

- The endpoint triple `endA–C` uses the same encodings (a BSDF_TRI endpoint is a surface triple; a BSDF_ENV endpoint
  is `RC_ENV_DIR` with ω in `rcWi` when k = d, else in `sfxDir`).
- For `k = ∅` and for `k = d` BSDF ends, the endpoint of a *spatially selected* sample is the source path's endpoint
  (the offset may hit a different emitter); consumers must not read `endA–C`/suffix cache for `k = ∅` (§2.4).
- Analytic `entry` < `0x3FFFFFF0`. `vertex_from_ids` builds the surface with normals flipped toward `fromPos`
  (bit-identical to `scene_surface(prim, u, v, pos − fromPos)`; §3.4).

### 2.4 Suffix cache (M5 consumer; M4 fills, copies and tests it)

Written by the path tree for every candidate. Meaningful (`sfxFlags.valid`) only when `k ≤ d−1` (x_{d−1} is on the
copied suffix). Fields: x_{d−1} ids (P6), `sfxDir`, `sfxT`, `betaS`, `sfxP2` as in §2.2; the endpoint identity is
`endA–C`. U-SFX-1 checks, for `k ≤ d−2`, `rcRad == betaS ⊙ endTerm(x_{d−1}, sfxDir, endpoint)` to 1e-5 relative.

### 2.5 ReSTIR G-buffer (atlas-sized textures)

| Texture | Format | Written by | Content |
|---|---|---|---|
| `rsVbuf` | rgba32uint | rs_primary | `(primId | BVH_MISS, bits(bary.u), bits(bary.v), bits(camDist))`, camDist = ‖pos(x₁) − x₀‖ (0 on miss) |
| `rsGeo` | rgba32float | rs_primary | `(n^g of x₁ oriented toward the camera, thr)`; thr = `primaryThreshold(x₀, pos(x₁), n^g, τ)` (0 on miss) |
| `rsL1` | rgba32float | rs_primary | length-1 radiance (math §16), a = 0 |
| `rsShade` | rgba32float | rs_spatial_resample (final round) | paired-MIS RGB estimate `m_c F_c W_c + Σ m_j G_j W_j` (0 on bg) |
| `rsFrame` | rgba32float | rs_finalize (ensemble, interactive) | final radiance `L1 + estimate` |

Readers use `textureLoad` (sample type `uint` / `unfilterable-float`). Destination domains of shifts read `rsVbuf`
(y₁ ids) and `rsGeo.w` (thr) of the destination pixel; x₀ = `frame.cam.camToWorld[3].xyz`.

### 2.6 Shift arena (`shiftArena`, one buffer, bound once per pass)

```
byte 0     header: 64 u32 (256 B)
             words 0–15   4 queue headers {counter (atomic), n, capacity, overflow (atomic)}: q0 spatial replay,
                          q1/q2 reserved for M5 temporal Q_f/Q_i, q3 reserved
             words 16–63  counters RSC_* (atomic)
byte 256   slots:  P·NS records of vec4<u32> = (bits(FJ.r), bits(FJ.g), bits(FJ.b), Jword), record (ai·NS + s)
           codes:  P·NS u32 at word offset 64 + 4·P·NS            (packed slot code, below)
           items:  P·NS u32 at word offset 64 + 5·P·NS            (queue items; capacity = P·NS)
total bytes = 256 + 24·P·NS            (P = atlas pixels, NS = RestirParams.numSlots)
```
WGSL views (appendix B.4): `struct ShiftArenaRW { hdr: array<atomic<u32>, 64>, words: array<u32> }` and the read-only
twin `ShiftArenaRO { hdr: array<u32, 64>, words: array<u32> }`.

**J word** (D7): `JW_NOT_ACCEPTED`, `JW_PENDING`, `JW_FAILED`, or the f32 bits of J (VALID). FJ = F_dst(Y)·J (RGB)
for VALID, 0 else. Writers (every slot is written exactly once per stage, no races):
- `rs_pair_accept` (thread of the smaller atlas index of the pair, D8, §3.9) writes both slots `(p, s)` and `(q, s)`:
  `NOT_ACCEPTED`; or `FAILED`/`SC_EMPTY_SRC` for an empty source; or `PENDING`/`SC_PENDING` plus a queue item when the
  source needs replay; or `PENDING`/`SC_PENDING` without an item otherwise. A pixel whose partner is invalid writes its
  own slot `NOT_ACCEPTED`.
- `rs_spatial_replay` overwrites the queued PENDING slots; `rs_spatial_shift` overwrites the PENDING slots whose source
  does not need replay (it reads its own reservoir flags). No PENDING may survive (`RSC_PENDING_LEFT`).

**Slot code word** (always written, same writer as the J word):
`code = sc | (term << 8) | (pair << 12) | ((pack2x16float(vec2f(margin, 0.0)) & 0xFFFFu) << 16)`, with
`sc` = `SC_*`, `term` = failing predicate sub-test (`RCT_D 1, RCT_R 2, RCT_F 3, RCT_I 4, RCT_GUARD 5`, 0 = none),
`pair` = pair index j of an O1/O3 failure (0 = n/a), margin = rcPairTest margin as f16.

| `SC_*` | Value | J word | Meaning |
|---|---|---|---|
| `SC_OK` | 0 | VALID | defined, F > 0 |
| `SC_NOT_ACCEPTED` | 1 | NOT_ACCEPTED | no partner / off-member / background / A0 false |
| `SC_EMPTY_SRC` | 2 | FAILED | source reservoir empty |
| `SC_O0_MISS` | 3 | FAILED | replay invalid sample, miss or early escape before y_{k−1} |
| `SC_O0_LOBE` | 4 | FAILED | copied-lobe joint pdf ≤ 0, or no non-delta lobe for a forced NEE at y_{d−1} |
| `SC_O0_TECH` | 5 | FAILED | k = ∅: different technique at d (hit vs escape, non-emissive triangle) |
| `SC_O0_LIGHT` | 6 | FAILED | endpoint light missing/invalid in the destination frame (M5; unreachable spatially) |
| `SC_O1` | 7 | FAILED | an earlier pair passes (pair index in `pair`) |
| `SC_O2` | 8 | FAILED | the rc pair fails (term, margin) |
| `SC_O3` | 9 | FAILED | k = ∅ and some pair (incl. terminal) passes |
| `SC_OCCLUDED` | 10 | FAILED | defined, reconnection/NEE segment occluded (F = 0) |
| `SC_ZERO` | 11 | FAILED | defined, F = 0 otherwise (f = 0, back-facing light, spot S = 0, L_e = 0, same-triangle NEE) |
| `SC_J_INVALID` | 12 | FAILED | J non-finite or ≤ 0 (guard) |
| `SC_O0_SUPPORT` | 13 | FAILED | support indicator 0 on a copied BSDF segment while the lobe pdf is > 0 |
| `SC_PENDING` | 14 | PENDING | accepted, not yet shifted (transient; queued iff the source needs replay) |
| `SC_NONFINITE` | 15 | FAILED | non-finite F (also counted in `RSC_SHIFT_NONFINITE`) |

Counters `RSC_*` (header words): 16 `CAND_NONFINITE`, 17 `BVH_OVERFLOW`, 18 `BVH_ITERCAP`, 19 `SHIFT_NONFINITE`,
20 `PENDING_LEFT`, 21 `SLOT_MISMATCH`, 22 `W_NONFINITE`, 23 `BASE_JDEN_INVALID`, 24 `ACCEPTED`, 25 `QUEUED`,
26 `SELECTED_SHIFTED`, 27 `EMPTY_CANON`, 28–31 reserved, 32–47 `CODE_BASE + sc` histogram, 48–63 reserved (M5).
Validation requires `CAND_NONFINITE = SHIFT_NONFINITE = PENDING_LEFT = SLOT_MISMATCH = W_NONFINITE = BVH_* = 0`
and every queue `overflow = 0`. `BASE_JDEN_INVALID / nCand ≤ 1e-5`. Replay fraction `f_r = QUEUED / ACCEPTED`
(logged, PLAN §1.10).

### 2.7 Indirect args (`rsArgs`, STORAGE | INDIRECT, 64 B)

Queue q's args at byte 16·q: `(x, y, z, pad) = (min(g, 65535), ceil(g / 65535), 1, 0)`, `g = ceil(n / 64)`; `n = 0`
gives `(0, 1, 1)` (a no-op). The args pass also writes `hdr.n = min(counter, capacity)` and sets `overflow` if
`counter > capacity`. Consumers (`@workgroup_size(64)`) compute
`item = (wid.y · nwg.x + wid.x) · 64 + lid` and exit when `item ≥ hdr.n`. Item word = `(ai << 3) | slot`.
A 1D dispatch is forbidden (silently no-ops above 4.19 M items, PLAN §1.8).

### 2.8 Pairing maps

- One `texture_2d_array<i32>` (`rg8sint`, 256 × 256 × 8 layers). Layer s holds a W_s × W_s torus involution of partner
  deltas `d ∈ [−127, 127]²`, `d = (0, 0)` = "no partner". Logical sizes `W_s` in `RestirParams.pairTexSize` (even,
  ≤ 254; M4 defaults 254, 246, 238, 230, 222, 210). Texels outside W_s are 0.
- **M4 generator** (`pairing.ts`, deterministic): PCG32 stream seeded by `(layer, R, W)`; Fisher–Yates order of the
  W² texels; for each unmatched texel a, up to 64 attempts draw an integer offset d uniformly from
  `{0 < dx² + dy² ≤ R²}` and match a with `b = (a + d) mod W` if b is unmatched and b ≠ a (`delta[a] = d`,
  `delta[b] = −d`); texels still unmatched keep (0, 0). Required: unmatched < 1%, radial KS distance to the uniform-disk
  CDF < 0.01 (T14-M4).
- **Transform per (frame t, round r, member m, slot s)**: `h = pcg4d(runSeed ^ m·φ, t, (r << 8) | s, STREAM_PAIRING)`;
  dihedral code `M = h.x & 7` (bit0 swap axes, bit1 negate x, bit2 negate y, applied in that order), offset
  `o = (h.y % W_s, h.z % W_s)`. With p the member-local pixel: `q = (M·p + o) mod W_s` (positive modulo),
  `partner = p + Mᵀ·d(q)`; valid iff d ≠ 0 and the partner lies inside the member tile. Reciprocity survives the
  transform (math §22); off-tile partners are lost symmetrically, so cross-member pairs never occur.

### 2.9 Uniforms

`RestirParams` (G0 binding 4, 128 B) and `RsDispatch` (G0 binding 7, dynamic offset, 256 B stride, 32 B used) are
normative in appendix B.3. Key fields: atlas/member sizes, E, maxBounces, `flags` (`RSF_RR 1, RSF_CRIT_2022 2,
RSF_PLANT_NO_J 4, RSF_PLANT_MARGINAL_J 8, RSF_ENSEMBLE 16, RSF_INTERACTIVE 32, RSF_J_REJECT 128`), S, NS, rounds,
rrMinBounces, τ, α_min, `wScale` (plant, 1 = off), `crit2022MinDist`, `pairTexSize`, `lightMode`. `RsDispatch`:
`t` (the frame's seed index), `passId`, `round`, `treeBase`, `treeCount`, `flags` (`RSD_FIRST_CHUNK 1,
RSD_FINAL_CHUNK 2, RSD_FINAL_ROUND 4`), `rowBase`, `rowEnd` (atlas row band). τ and α_min are uniforms, never
overrides (gap-rc §8.3).

### 2.10 Ensemble atlas and `ensembleStats`

- Atlas: `memberCols = min(E, ⌊16384 / W⌋)`, rows `⌈E / memberCols⌉`; pixel (ax, ay) → member
  `m = (ay / H)·memberCols + ax / W`, local `(ax mod W, ay mod H)`; members ≥ E are inactive (all passes skip).
  Constraint `E·W·H ≤ 2²²`.
- `ensStats` buffer (f32, rw): for levels ℓ ∈ {16, 32, 64} with `Th_ℓ = ⌈H/ℓ⌉, Tw_ℓ = ⌈W/ℓ⌉`, per member and tile an
  RGB sum (vec4, w = 0) in the order `[ℓ][m][ty][tx]`; then `global[m]` (vec4); then `mask[m][M]` (vec4, M ≤ 16). All
  sums of the **per-run (member) image of this frame**. Deterministic fixed-order workgroup reductions only (no float
  atomics): 16² tiles by a 16×16 workgroup tree reduction, 32²/64²/global/mask from those by a second fixed-order pass.
- `ensPixel` buffer (f32, rw): per member-local pixel `(Σx.rgb, 0, Σx².rgb, 0)` over members **and** frames of the
  current batch; the runner reads it per batch, adds it to f64 host accumulators and clears it.
- Mask texture (`r8uint`, member-local, region id or 255): optional (M = 0 in M4).
- Readback: `ensStats` every frame; host rows get run id `r = t·E + m`, seed label `${runSeed}:${t}:${m}`.
- `ensemble.npz` (compare.py README "Ensemble .npz format"): `tiles16/32/64 (R,Th,Tw,3)`, `global (R,3)`,
  `masks (R,M,3)` + `mask_pixels`, `pixel_sum`/`pixel_sumsq (H,W,3)` as float64, `count = R`, `channels = R,G,B`,
  `height`, `width`; stored (uncompressed) zip written by `npz.ts`; `meta.json` as the batch runs plus `seeds[R]`.

### 2.11 Debug views, probe records, hook API

View ids (DebugParams.mode; kinds per `debug-views.ts`); registered by `render/restir/debug.ts`:

| Id | Key | Kind | Written by (hook) | Content |
|---|---|---|---|---|
| 400 | `rs.c` | scalar (log) | `rsdbg_reservoir` | c |
| 401 | `rs.W` | scalar (log) | same | W |
| 402 | `rs.phat` | scalar (log) | same | lum(F) |
| 403 | `rs.FW` | vec3 | same | F·W |
| 404 | `rs.d` | code | same | d (0 empty) |
| 405 | `rs.k` | code | same | k (0 = ∅) |
| 406 | `rs.tech` | code | same | technique |
| 407 | `rs.endpoint` | code | same | endpoint type |
| 408 | `rs.lobes` | code | same | `ℓ_{k−1}·16 + ℓ_k` (+ delta bits ·256) |
| 409 | `rs.class` | code | same | L/N1/B1/E/D-NEE/D-BSDF/R (math §17) as 0…6 |
| 410 | `rs.kMargin` | scalar (signed) | same | `kMargin` |
| 420–425 | `shift.code[s]` | code | `rsdbg_slot` | SC code of slot s (source pixel) |
| 426–431 | `shift.logJ[s]` | scalar (signed) | same | log2 J (VALID only) |
| 432–437 | `shift.term[s]` | code | same | `term | pair<<4` |
| 438 | `shift.replayMask` | code | `rsdbg_slot` | bitmask of queued slots |
| 439 | `shift.acceptMask` | code | `rsdbg_accept` (spatial_shift) | bitmask of accepted slots (from the J words) |
| 440–445 | `shift.margin[s]` | scalar (signed) | `rsdbg_slot` | margin |
| 460 | `mis.mc` | scalar | `rsdbg_mis` | m_c |
| 461 | `mis.sumM` | scalar (signed) | same | `m_c(X_c) + Σ_j m_j(X_c) − 1` evaluated with the production functions at y = X_c (must be 0) |
| 462 | `mis.lumL` | scalar (signed) | same | `(lum(L_rgb) − Σw)/max(Σw, 1e-30)` (must be 0) |
| 463–468 | `mis.mj[s]` | scalar | same | m_j of partner slot s |
| 469 | `mis.k` | code | same | \|S_c\| |
| 470 | `mis.sel` | code | same | 0 canonical, 1+s partner |

Reservoir views use the stage taps `DBG_TAP_INITIAL` (written by rs_initial) and `DBG_TAP_SPATIAL` (written by the
final-round resample). HUD numbers (per frame, from the arena counters): queue occupancy per round, f_r, SC histogram,
overflow flags.

Hook API (`debug/restir-views.wgsl`, P0 empty bodies; each hook is a no-op unless its view is active or the pixel is
the probe pixel):
```wgsl
fn rsdbg_reservoir(px: vec2u, ai: u32, tap: u32);                      // after a reservoir is final (initial / resample)
fn rsdbg_candidate(px: vec2u, d: u32, tech: u32, k: u32, w: f32, lumF: f32, counter: u32, selected: bool);  // probe only
fn rsdbg_vertex(px: vec2u, path: u32, b: u32, pos: vec3f, lobeCode: u32); // probe only; path 0 = base tree, 1+s = shift into slot s, 8+s = partner→probe shift
fn rsdbg_slot(px: vec2u, s: u32, code: u32, J: f32, replayed: bool);   // shift/replay (px = source pixel)
fn rsdbg_accept(px: vec2u, mask: u32);                                   // spatial_shift, from the pixel's J words
fn rsdbg_mis(px: vec2u, k: u32, mc: f32, sumM: f32, lumRel: f32, s: u32, mj: f32, wj: f32, sel: u32); // resample; s = 0xFF for the canonical record
```
With `DEBUG_NO_BINDINGS` (test pipelines without G3) every hook compiles to an empty body.

Probe tags (`ProbeRecord.tag`, value = vec4f of bit patterns where noted): 64 reservoir plane (value = plane bits,
`seq` order = plane index; preceded by tag 65 header `(pass, round, ai, 0)`), 66 candidate, 67 vertex
`(pos.xyz, bits(path<<8 | b<<4 | lobe))`, 68 slot `(bits(code), J, bits(s), replayed)`, 69 MIS partner, 70 MIS
canonical `(k, mc, W_out, c_out)`. The inspector (WP-D) decodes these; the 3D overlay draws the base path and every
shifted path of the probe pixel as polylines.

### 2.12 Candidate dump (tests only)

`rs_initial` compiled with `RS_DUMP_CANDIDATES = 1` (and debug G3 **not** bound) binds `candDump` at G2 binding 4:
per atlas pixel `RS_DUMP_CAP = 32` records of `RS_DUMP_WORDS = 48` u32: words 0–39 = the candidate as a full
reservoir record (W field = its RIS weight w_i, `wSum` = running Σw before it, `nCand` = ordinal), words 40–47 = the
`primId` of the base path's vertices x₁…x₈ (`BVH_MISS` beyond d−1), plus a per-pixel count at the end of the buffer.
Only tree 0 is dumped.

### 2.13 Memory (validation layout)

| Item | B/px (atlas) | 1024² (E = 16, 256²) |
|---|---|---|
| `resA` + `resB` | 320 | 336 MB |
| `shiftArena` (NS = 6) | 144 | 151 MB |
| rsVbuf, rsGeo, rsL1, rsShade, rsFrame | 80 | 84 MB |
| batch accum / ensPixel | 16 / 32 per member px | ≤ 17 MB |
| **Total** | ≈ 560 | ≈ 0.59 GB (PLAN §1.9: ≈ 0.57 GB) |

E = 64 × 256² (4.2 M px) needs ≈ 2.4 GB and is allowed only after the M0 allocation probe; M4 gates do not need it
(D14). All allocations go through `pushErrorScope('out-of-memory')`.

---

## 3. Function contracts

Signatures are normative; bodies follow the pseudo-code. "Same function" means the literal WGSL function, called from
every place listed; a second implementation of the same formula is a contract violation.

### 3.1 `bsdf_query` (material/bsdf.wgsl, WP-A in P0)

```wgsl
struct BsdfQuery {
  f_lobe: vec3f,   // f_ℓ·|Ns·L| of `lobe` (LOBE_NEE: all lobes, Cycles eval incl. the spurious glass region), no support factor
  p_joint: f32,    // q(ℓ|V)·p_ℓ(L|V), valid-only (LOBE_NEE: 1; delta or unallocated lobe: 0)
  f_all: vec3f,    // all-lobe Cycles eval (NEE)
  p_marg: f32,     // Σ_{non-delta ℓ} q(ℓ|V)·p_ℓ(L|V), valid-only (MIS p2, footprints p̄)
  supp: bool,      // 1_supp(lobe, V, L) (math.md#support-indicator); true for LOBE_NEE / LOBE_NONE
}
fn bsdf_query(m: MatEval, V: vec3f, L: vec3f, lobe: u32) -> BsdfQuery;
```
One `bsdf_prepare` + one `bsdf_eval_ctx`; `supp` from the same context (G_T: `g_side` bits). It must equal
`bsdf_eval_lobe`, `bsdf_eval`, `bsdf_pdf_marginal`, `bsdf_sample_support` for the same inputs (U-PT-BITS checks
agreement to 1e-6 relative and identical `supp`). ReSTIR modules call only `bsdf_query`, `bsdf_sample`,
`material_eval` and `lobe_roughness`.

### 3.2 Frame helpers (restir/frame.wgsl, P0) — math §15

```wgsl
struct RsPix { valid: bool, member: u32, local: vec2u, localIdx: u32, ai: u32, px: vec2u }
fn rs_pix(px: vec2u) -> RsPix;                          // atlas → (member, local); invalid outside atlas/rows/inactive member
fn rs_frame_key(member: u32, t: u32, localIdx: u32) -> vec2u;   // = path_init_seed(frame.runSeed, member, t, localIdx)
fn rs_tree_seed(key: vec2u, s: u32) -> vec2u;           // s = 0: key; s ≥ 1: pcg4d(key.x, key.y, s, STREAM_TREE).xy
fn rs_jitter(key: vec2u, px: RsPix) -> vec2f;           // JITTER_IID: rand2(key.x, 0, STREAM_JITTER); else frame.jitter
fn rs_rand(key: vec2u, passId: u32, counter: u32) -> f32;  // u32_to_unit(pcg4d(key.x, key.y ^ passId·φ, counter, STREAM_RESAMPLE).x)
fn rs_path_hash(seed: vec2u, vertex: u32, slot: u32) -> u32;   // path_hash, or rs_rng_override(...) under RS_RNG_OVERRIDE
fn rs_path_u01(seed: vec2u, vertex: u32, slot: u32) -> f32;
fn rs_path_bsdf_u4(seed: vec2u, vertex: u32) -> vec4f;         // (u_lobe, u_h1, u_h2, u_rt) through rs_path_hash
fn rs_cam_pos() -> vec3f;                                // frame.cam.camToWorld[3].xyz
fn rs_t() -> u32;                                        // rsDispatch.t
```
`rs_path_hash` is the **only** path-dimension accessor in pathtree/replay (the dense PSS sweep overrides it).

### 3.3 Endpoints (restir/endpoint.wgsl, WP-A) — math §8, §9, §16

```wgsl
struct NeeEndpoint { entry: u32, a: u32, b: u32 }        // alias entry + light-local words (§2.3)
fn nee_draw(slot: LightSlot, hSel: u32, hSel2: u32, hL: vec3u) -> NeeEndpoint;   // alias_sample + (u01(h_l0), u01(h_l1)) | env (i,j,h2)
fn nee_eval(x: vec3f, ep: NeeEndpoint) -> LightSample;   // z, dir, dist, cosZ, Λ, q, p1, isDelta, isInf, analytic, prim at shading point x
fn nee_endpoint_words(ep: NeeEndpoint) -> vec3u;         // (RC_TAG_NEE | entry, a, b)
fn nee_endpoint_from_words(w: vec3u) -> NeeEndpoint;
fn nee_visible(x: SurfaceHit, xPrim: u32, ls: LightSample) -> bool;   // visible() or visibleInf() exactly as the PT
```
Invariant (U-EP-1): `nee_eval(x, nee_draw(slot, h…))` is bitwise equal to `nee_sample(x, h…)` for every entry type;
the PT is unchanged. Emitter end terms use the PT's `tri_emission`, `tri_light_p1`, `envRadiance(envUV(ω))`,
`p1Env`, `env_bsdf_mis_weight`, `mis_w2`, `nee_mis_w1` directly (no new copies).

### 3.4 Reconnection predicate (restir/rc.wgsl, WP-B) — math §19, gap-rc §3

```wgsl
struct RcVertex { pos: vec3f, ng: vec3f, kind: u32, diffuseOnly: u32 }   // kind RCK_*
struct RcEvent  { lobe: u32, delta: u32, alpha: f32, pMarg: f32 }         // alpha = perceptual roughness of the event
struct RcResult { pass: bool, margin: f32, term: u32 }                     // term RCT_* of the first failing sub-test
fn vertex_from_ids(prim: u32, u: f32, v: f32, fromPos: vec3f) -> SurfaceHit;     // == scene_surface(prim,u,v,pos−fromPos) bitwise
fn rc_vertex(s: SurfaceHit, m: MatEval) -> RcVertex;                     // kind SURFACE, diffuseOnly from m.flags
fn rc_event_bsdf(m: MatEval, lobe: u32, isDelta: bool, pMarg: f32) -> RcEvent;   // alpha = isDelta ? 0 : lobe_roughness(m, lobe)
fn rc_event_nee(m: MatEval, pMarg: f32) -> RcEvent;                      // lobe NEE, alpha = lobe_roughness(m, LOBE_NEE) (2022 mode only)
fn rc_event_none() -> RcEvent;                                           // light/env vertex (alpha FLT_MAX)
fn rcPairTest(a: RcVertex, ea: RcEvent, b: RcVertex, eb: RcEvent, thr: f32) -> RcResult;   // Enhanced, or 2022 if RSF_CRIT_2022
fn primaryThreshold(camPos: vec3f, x1: vec3f, ng1: vec3f, tau: f32) -> f32;             // gap-rc §3.6 formula
fn rc_G(a: vec3f, b: vec3f, ngB: vec3f) -> f32;                          // |n_b·(a−b)|/‖a−b‖³ (receiving vertex b)
// rs_pos_finite(x) / rs_geq_pos(a, b): integer bit compares (gap-rc §3.6), defined in types.wgsl (P0), used here
// deferred k* per candidate (math §19 pair list; gap-rc §3.4)
fn kstar_nee(treeRc: u32, B: u32, prevV: RcVertex, prevE: RcEvent, curV: RcVertex, eNee: RcEvent, thr: f32) -> vec2u; // (k, bits(margin))
fn kstar_tree_pair(prevV: RcVertex, prevE: RcEvent, curV: RcVertex, eB: RcEvent, thr: f32) -> RcResult;
fn kstar_bsdf_end(treeRc: u32, B: u32, curV: RcVertex, eB: RcEvent, endV: RcVertex, thr: f32) -> vec2u;             // (k or 0, margin)
```
- `rcPairTest` (Enhanced) is gap-rc §3.6 verbatim: D (ea), R (α ≥ α_min, `>=`), F (`kind == ENV` passes, else
  `t²/(ea.pMarg·|cos_b|) ≥ thr`), I (skipped for LIGHT/ENV/diffuseOnly, else `eb` non-delta and
  `t²/(eb.pMarg·|cos_a|) ≥ thr`); guards `t² > 1e-12`, `p̄` finite > 0, cos > 0 fail the pair; margin = min over the
  evaluated sub-tests of `log2(value/thr)` or `α − α_min`. 2022 mode per D19.
- `kstar_nee`: returns `treeRc` if non-zero; else `B` if `B ≥ 2` and `rcPairTest(prevV, prevE, curV, eNee)` passes
  (case b); else `B + 1` (forced).
- `kstar_bsdf_end`: returns `treeRc` if non-zero; else `B + 1` if `rcPairTest(curV, eB, endV, rc_event_none())` passes
  (case d/e); else 0 (∅).
- `vertex_from_ids`: `scene_surface(prim, u, v, vec3f(0))` then flip ng, ns (and set `backfacing`) iff
  `dot(ng, fromPos − pos) < 0`; bit-identical to `scene_surface(prim, u, v, pos − fromPos)`.

### 3.5 Path tree (path/pathtree.wgsl, WP-A) — math §16, §19, §25

```wgsl
fn pathtree_run(p: RsPix, key: vec2u, treeBase: u32, treeCount: u32, firstChunk: bool, finalChunk: bool);
```
Shorthand: `h(B, X) = rs_path_hash(seed, B, SLOT_X)`; `slotCur = lightsParams.cur`; `curPrim` = primId of `cur`.
Reads `rsVbuf`/`rsGeo`, streams the candidates of trees `[treeBase, treeBase+treeCount)` into `resOut[p.ai]`,
persists `wSum`, finalises W on the final chunk. Normative algorithm (per tree s):

```
seed = rs_tree_seed(key, s); thr = rsGeo.w; camPos = rs_cam_pos()
cur  = vertex_from_ids(vbuf prim, u, v, camPos);  V = normalize(camPos − cur.pos)
beta = 1 (RR-free); rrInv = 1; treeRc = 0; betaPost = 1; hist = 0xFFFFFFFF
for B = 1 … maxBounces+1:
  m = material_eval(cur, V);  curV = rc_vertex(cur, m)
  // (1) NEE (not at a delta-only vertex)
  if m.flags & MATEVAL_HAS_NON_DELTA:
    ep = nee_draw(slotCur, h(B,SEL), h(B,SEL2), (h(B,L0), h(B,L1), h(B,L2)));  ls = nee_eval(cur.pos, ep)
    if ls.valid && ls.prim != curPrim && any(ls.Lambda > 0):
      qn = bsdf_query(m, V, ls.dir, LOBE_NEE);  w1 = nee_mis_w1(ls, qn.p_marg, B)
      F  = (w1 / ls.q) * (beta * qn.f_all * ls.Lambda)
      if any(F > 0) && nee_visible(cur, curPrim, ls):
        (k, mg) = kstar_nee(treeRc, B, prevV, prevE, curV, rc_event_nee(m, qn.p_marg), thr)
        stream NEE candidate, d = B+1, W_src = rrInv (plain NEE: W_NEE = 1; M(B) = 1), fields per §2.2 case
        (deep: rcRad = betaPost * (w1/ls.q) * qn.f_all * ls.Lambda; (b): rcRad = ls.Lambda, aux = ls.p1,
         rcWi = ls.dir, jDen = prevPJoint * rc_G(prevPos, cur.pos, cur.ng); forced: rc = nee_endpoint_words(ep))
        suffix: x_{d−1} = cur ids, sfxDir = V, betaS = (k ≤ B−1 ? betaPost : 1), sfxT = 0, sfxP2 = qn.p_marg
        counter = (s<<20) | (B<<12) | 0
  // (2) Russian roulette (D11)
  if RSF_RR && B > rrMinBounces: q = min(sqrt(max3(beta)), 1); if !(rs_path_u01(seed,B,SLOT_RR) < q): break; rrInv /= q
  // (3) BSDF continuation
  bs = bsdf_sample(m, V, rs_path_bsdf_u4(seed, B)); if !bs.valid: break
  hist[B] = bs.lobe | bs.is_delta<<3
  org = offset_ray(cur.pos, sign(ng·L)·ng);  h = trace_closest_ex(org, bs.L, FLT_MAX, curPrim, BVH_MISS)
  nxt = hit ? vertex_from_ids(h.prim, h.u, h.v, cur.pos) : none
  wOut = hit ? normalize(nxt.pos − cur.pos) : bs.L                          // D3
  qb = bsdf_query(m, V, wOut, bs.lobe);  eB = rc_event_bsdf(m, bs.lobe, bs.is_delta, qb.p_marg)
  if qb.p_joint not finite-positive && !bs.is_delta: count RSC_BASE_JDEN_INVALID (still continue)
  // tree pair B (shared prefix pair (x_{B−1}, x_B), EV_BSDF at x_B)
  if treeRc == 0 && B ≥ 2 && kstar_tree_pair(prevV, prevE, curV, eB, thr).pass:
     treeRc = B; tree = {cur ids, rcWi = wOut, ℓ_{k−1} = prevE.lobe, ℓ_k = eB.lobe,
                          jDen = prevPJoint * rc_G(prevPos, cur.pos, cur.ng) * qb.p_joint}; betaPost = 1
  beta *= bs.weight;  if treeRc != 0 && treeRc < B: betaPost *= bs.weight
  if !any(beta > 0): break
  // (4) Mode-B crossings: RS_MODE_B stub (M6; D1)
  // (5) BSDF-hit endings at x_{B+1}, d = B+1
  if !hit:
     if env present: w2 = env_bsdf_mis_weight(bs.L, qb.p_marg, B, bs.is_delta); Le = envRadiance(envUV(bs.L,cg,sg))
                     F = w2 * beta * Le;  (k, mg) = kstar_bsdf_end(treeRc, B, curV, eB, envVertex, thr)
                     stream BSDF_ENV candidate (W_src = rrInv): (c) treeRc == B: rcRad = Le, aux = p1Env(bs.L);
                     (deep) treeRc < B: rcRad = betaPost * w2 * Le; (e) k == B+1: rc = RC_ENV_DIR, rcWi = bs.L,
                     jDen = qb.p_joint; suffix: sfxDir = bs.L, sfxT = FLT_MAX, flags escape, sfxP2 = qb.p_marg,
                     betaS = (treeRc < B ? betaPost : 1); endpoint (RC_ENV_DIR); counter (s<<20)|(B<<12)|1
     break
  Le = tri_emission(h.prim, h.u, h.v)
  if any(Le > 0):
     w2 = bs.is_delta ? 1 : mis_w2(tri_light_p1(cur.pos, nxt.pos, nxt.ng, h.prim), qb.p_marg, B)
     F = w2 * beta * Le;  (k, mg) = kstar_bsdf_end(treeRc, B, curV, eB, RcVertex(nxt.pos, nxt.ng, RCK_LIGHT, 0), thr)
     stream BSDF_TRI candidate: (c) rcRad = Le, aux = tri_light_p1(...); (deep) rcRad = betaPost * w2 * Le;
     (d) k == B+1: rc = (h.prim, u, v), jDen = qb.p_joint * rc_G(cur.pos, nxt.pos, nxt.ng); endpoint (h.prim, u, v);
     sfxT = h.t, sfxDir = wOut, sfxP2 = qb.p_marg; counter (s<<20)|(B<<12)|1
  prevV = curV; prevE = eB; prevPJoint = qb.p_joint; prevPos = cur.pos; cur = nxt; V = −wOut
```
- **Streaming** (`ris_update`, B.2): `w_i = lum(F_i)·W_src,i`; non-finite or ≤ 0 w is skipped (non-finite counted in
  `RSC_CAND_NONFINITE`); `u = rs_rand(key, RS_PASS_INITIAL, counter)`; on selection the whole candidate record is
  written to `resOut` immediately (no large register struct). F stores the RR-free integrand; W carries 1/∏q.
- **Finalise** (final chunk): `W = wSum / (S·lum(F_Y))` (S = numTrees), `c = 1`; `wSum == 0` ⇒ empty record (d = 0,
  W = 0, F = 0, c = 1). Background pixels (vbuf miss) write the bg record on the first chunk. `rsdbg_reservoir` is
  called after finalising.
- The candidate's `lobeHist` = tree history for vertices < B plus `LOBE_NEE` at B (NEE) or the history through B
  (BSDF end).
- `F` grouping follows the PT's expression minus `rrScale` so that U-RIS-1 compares like with like.

### 3.6 Replay (path/replay.wgsl, WP-B) — math §20 O0/O1/O3, gap-rc §6.2/§9.1

```wgsl
struct ReplayOut {
  code: u32,             // SC_OK | SC_O0_MISS | SC_O0_TECH | SC_O1 | SC_O3 | SC_ZERO (+ term, pair, margin packed as §2.6)
  margin: f32,
  yLast: SurfaceHit,     // prefix mode: y_{k−1};   ∅ mode: unused
  yLastPrim: u32,
  VLast: vec3f,          // incoming direction at y_{k−1} (from positions)
  Tp: vec3f,             // ∏ over replayed samples (prefix: y₁…y_{k−2}; ∅: y₁…y_{d−1}), RR-free
  preValid: bool,        // a pre-rc pair (y_{k−2}, y_{k−1}) exists (k−1 ≥ 2)
  preV: RcVertex, preE: RcEvent,   // y_{k−2} and its replayed event
  F: vec3f,              // ∅ mode: the replayed path's own integrand incl. ω2
}
fn replay_prefix(seed: vec2u, y1: SurfaceHit, y1Prim: u32, camPos: vec3f, thr: f32, kIdx: u32, d: u32, tech: u32) -> ReplayOut;
// kIdx = k (prefix mode, 3 ≤ k ≤ d) or 0 (∅ mode, d ≥ 2)
```
There is **no replay-state buffer**: replay and reconnection are fused in one thread (gap-perf §2.2, PLAN §1.8), and
`ReplayOut` lives in registers. Per replayed vertex b (prefix mode: b = 1 … k−2; ∅ mode: b = 1 … d−1): `material_eval` with V from positions,
`bsdf_sample(m, V, rs_path_bsdf_u4(seed, b))` (invalid ⇒ `SC_O0_MISS`), trace; miss before the last vertex ⇒
`SC_O0_MISS`; event `e_b` with `wOut` from positions (D3); pairs `2 ≤ b` tested with `rcPairTest(y_{b−1}, e_{b−1},
y_b, e_b, thr)`, a pass ⇒ `SC_O1` (prefix) / `SC_O3` (∅) with `pair = b`; `Tp *= bs.weight`, `Tp = 0` ⇒ `SC_ZERO`.
∅ mode at b = d−1: `BSDF_ENV` needs a miss, `BSDF_TRI` needs a hit on a triangle with `TRI_EMISSIVE` (else
`SC_O0_TECH`); the terminal pair (y_{d−1}, env | emitter as `RCK_LIGHT`) passing ⇒ `SC_O3` (pair d); then
`F = Tp ⊙ L·ω2` with the PT's ω2 functions at b = d−1 (1 after a delta lobe). No NEE, no RR, no crossings, no RIS.

### 3.7 Shift (restir/shift.wgsl, WP-B) — math §18, §20, §21, rules 3–7

```wgsl
struct ShiftSrc { empty: bool, flags: u32, seed: vec2u, F: vec3f, rc: vec3u, jDen: f32, rcWi: vec3f, aux: f32,
                  rcRad: vec3f, end: vec3u }
struct ShiftDst { valid: bool, prim: u32, bary: vec2f, thr: f32, camPos: vec3f }
struct ShiftOut { FJ: vec3f, J: f32, jNum: f32, code: u32 }   // code packed as §2.6; FJ, J = 0 unless SC_OK
fn shift_src_load(ai: u32) -> ShiftSrc;                // from resIn (reservoir.wgsl accessors)
fn shift_dst_load(px: vec2u) -> ShiftDst;              // rsVbuf / rsGeo / camera of the destination pixel
fn res_needs_replay(flags: u32) -> bool;               // non-empty ∧ (k > 2 ∨ k = ∅)
fn shift_hybrid(src: ShiftSrc, dst: ShiftDst) -> ShiftOut;
```
Order of evaluation (normative; every "undefined" decision precedes any "zero" decision):
1. `src.empty` ⇒ `SC_EMPTY_SRC`. `y1 = vertex_from_ids(dst.prim, bary, camPos)`.
2. Replay if `res_needs_replay` (only compiled with `RS_REPLAY = 1`; with `RS_REPLAY = 0` such a source is a caller
   bug): ∅ ⇒ return `rp.F`, J = 1 (after the NONFINITE/ZERO checks). Else `yPrev = rp.yLast`, `VPrev`, `Tp`, `pre`.
   Without replay: `yPrev = y1`, `VPrev = normalize(camPos − y1.pos)`, `Tp = 1`, no pre.
3. Case dispatch on (k, d, technique) and evaluation of the events:
   - **(a)/(f) forced NEE** (`k = d`, NEE): `ls = nee_eval(yPrev.pos, end)`; `mY` without a non-delta lobe ⇒
     `SC_O0_LOBE`; `qn = bsdf_query(mY, VPrev, ls.dir, LOBE_NEE)`; O1 on the pre-rc pair with
     `rc_event_nee(mY, qn.p_marg)` (EV_RECONNECT_NEE); no O2. F = `Tp ⊙ (w1/ls.q)·(f_all ⊙ Λ)`, w1 =
     `nee_mis_w1(ls, qn.p_marg, d−1)`; same-triangle ⇒ `SC_ZERO`; `nee_visible` fails ⇒ `SC_OCCLUDED`; J = 1.
   - **(d) emitter rc** (`k = d`, BSDF_TRI): z = `vertex_from_ids(end…, yPrev.pos)`, ω' = normalize(z − yPrev);
     `qy = bsdf_query(mY, VPrev, ω', ℓ_{k−1})`, `eY = rc_event_bsdf(mY, ℓ_{k−1}, false, qy.p_marg)`; O1 (pre, eY);
     O2 `rcPairTest(yPrev, eY, z as RCK_LIGHT)`; O0 `qy.p_joint > 0` (`SC_O0_LOBE`) and `qy.supp` (`SC_O0_SUPPORT`);
     F = `Tp ⊙ (f_lobe/p_joint)·ω2·L_e`, ω2 = `mis_w2(tri_light_p1(yPrev, z, n_z, prim), qy.p_marg, d−1)`;
     jNum = `qy.p_joint·rc_G(yPrev, z, n_z)`; visibility `visible(yPrev, z)` excluding both primitives.
   - **(e) env rc** (`k = d`, BSDF_ENV): ω = `rcWi`; as (d) with O2 = D ∧ R (kind ENV), ω2 =
     `env_bsdf_mis_weight(ω, qy.p_marg, d−1, false)`, L = `envRadiance(envUV(ω))`, jNum = `qy.p_joint`,
     visibility `visibleInf(yPrev, ω)`.
   - **(b)/(c)/(deep) surface rc** (`k ≤ d−1`): x_k = `vertex_from_ids(rc…, yPrev.pos)`, ω' = normalize(x_k − yPrev),
     `V_k = −ω'`, `mK = material_eval(x_k, V_k)`; x_k event: (b) `ls = nee_eval(x_k.pos, end)`,
     `qk = bsdf_query(mK, V_k, ls.dir, LOBE_NEE)`, `eK = rc_event_nee`; (c)/(deep) `qk = bsdf_query(mK, V_k, rcWi, ℓ_k)`,
     `eK = rc_event_bsdf(mK, ℓ_k, δ_k, qk.p_marg)`. O1 (pre, eY); O2 `rcPairTest(yPrev, eY, x_k, eK)`; O0 on `qy`
     and, for (c)/(deep), on `qk` (joint > 0 and supp). F:
     (b) `Tp ⊙ (qy.f_lobe/qy.p_joint) ⊙ (w1/ls.q)·(qk.f_all ⊙ ls.Lambda)`, w1 = `nee_mis_w1(ls, qk.p_marg, d−1)`;
     (c) `Tp ⊙ (qy.f/qy.p) ⊙ (qk.f/qk.p) ⊙ ω2·L_e` with the end term re-evaluated at x_k (D6);
     (deep) `Tp ⊙ (qy.f/qy.p) ⊙ (qk.f/qk.p) ⊙ rcRad`.
     jNum = `qy.p_joint·rc_G(yPrev, x_k, n_k)·((c)|(deep) ? qk.p_joint : 1)`; visibility `visible(yPrev, x_k)`.
4. `J = jNum / src.jDen` (1 for (a), (f), ∅); `J` not finite-positive ⇒ `SC_J_INVALID`; F non-finite ⇒ `SC_NONFINITE`;
   plants: `RSF_PLANT_NO_J` ⇒ `J = 1` after the guard; `RSF_PLANT_MARGINAL_J` ⇒ marginal pdfs in jNum (and in the base
   jDen, pathtree honours the same flag).
5. `lum(F) == 0` ⇒ `SC_ZERO`; else `SC_OK`, `FJ = F·J`.
- M(B) = 1 everywhere (`mis_M`), p1/p2 recomputed at the offset's own x_{d−1} (math §9).
- The support indicator is applied through `bsdf_query.supp` to ω' at y_{k−1} (all non-forced cases) and to ω_k at
  x_k ((c), deep); never to NEE segments (math §14).
- `RS_SHIFT_TRACE = 1` (tests): the shift additionally returns the replayed prefix `primId`s and lobes (y₂…y_{k−1})
  and the events/margins of O1/O2 for T3 and the dual.

### 3.8 MIS, resampling, write-back (restir/mis.wgsl, WP-C) — math §22, §26

```wgsl
fn mis_canonical_term(a: f32, pc: f32, cj: f32, lumH: f32) -> f32;   // a·pc/(a·pc + cj·lumH), := 1 if denominator 0
fn mis_partner_weight(a: f32, cj: f32, pj: f32, lumG: f32, k: u32) -> f32; // [cj·pj/(cj·pj + a·lumG)]/(k+1), := 0 if den 0
fn spatial_resample(p: RsPix, round: u32, finalRound: bool);
fn res_select_shifted(dst: u32, src: u32, G: vec3f, J: f32, W: f32, c: f32);   // write-back (math §18, PLAN rule 4)
```
Per hit pixel c (background: copy, shade 0): `S_c` = slots whose J word ≠ `NOT_ACCEPTED` (PENDING counts as FAILED
and increments `RSC_PENDING_LEFT`; a partner slot whose acceptance differs increments `RSC_SLOT_MISMATCH`);
`k = |S_c|`; k = 0 ⇒ output = canonical, shade F_c·W_c. Else `a = c_c/k`,
`m_c = (1 + Σ_s mis_canonical_term(a, p̂_c, c_j, lum H_s))/(k+1)`, `w_c = m_c·p̂_c·W_c` streamed first
(`rs_rand(key_c, RS_PASS_SPATIAL + r, 0)`), then each s ∈ S_c with `G = FJ(partner, s)`, `m_j`, `w_j = m_j·lum(G)·W_j`
(counter 1 + s). `W_Y = Σw / p̂_c(Y)`, `p̂_c(Y_j) = lum(G_j)/J_j`; `c_out = c_c + Σ_{S_c} c_j` (FAILED partners count);
`Σw = 0` ⇒ canonical fields with W = 0. Then `W *= rsParams.wScale` (1 unless the W-scale plant is on). Final round:
`rsShade = m_c F_c W_c + Σ m_j G_j W_j`.
Write-back of a partner sample: copy all ten planes of `resIn[j]`, then `F = G/J`, `jDen = J·jDen_j`, `W`, `c`;
nothing else changes spatially (aux/suffix/endpoint stay, D6/D16).

### 3.9 Pairing and queues (restir/pairing.wgsl, restir/queue.wgsl, WP-C)

```wgsl
struct PairResult { valid: bool, partner: vec2u }            // member-local pixel
fn dihedral_apply(code: u32, v: vec2i) -> vec2i;  fn dihedral_apply_t(code: u32, v: vec2i) -> vec2i;
fn pair_partner(local: vec2u, member: u32, t: u32, round: u32, slot: u32) -> PairResult;   // §2.8
fn pair_A0(a: vec4u, ga: vec4f, b: vec4u, gb: vec4f) -> bool;   // rsVbuf/rsGeo texels, canonical order (min index first)
fn queue_append(q: u32, item: u32);                              // atomicAdd; overflow flag instead of writing past capacity
fn queue_item(q: u32, wid: vec3u, nwg: vec3u, lid: u32) -> u32;  // item index or 0xFFFFFFFF past n
fn slot_index(ai: u32, s: u32) -> u32;                           // ai·NS + s
```
`pair_A0`: both hits, `dot(ng_a, ng_b) ≥ 0.5`, `|z_a − z_b| ≤ 0.1·min(z_a, z_b)` with z from `rsVbuf.w` (D8).

`rs_pair_accept` (normative), thread per atlas pixel p (background pixels included, no early return), for each slot s:
`pr = pair_partner(p.local, p.member, t, r, s)`; invalid ⇒ write slot (p, s) = `NOT_ACCEPTED`; else with q the partner's
atlas index: if `p.ai < q` evaluate `A = pair_A0(G[p], G[q])` once and write **both** slots (p, s) and (q, s) as in §2.6
(`RSC_ACCEPTED += 2` when A); if `p.ai > q` write nothing (the partner's thread owns the pair). Because the map is an
involution, every slot is written exactly once. `rs_spatial_shift` thread p: loads its own flags once, for each s with
J word `PENDING` and `!res_needs_replay(flags)` runs `shift_hybrid(src_p, dst = partner)` and writes (p, s); it also
calls `rsdbg_accept`. `rs_spatial_replay` thread per queue item `(ai << 3) | s`: recomputes the partner and runs
`shift_hybrid` with replay compiled in.

### 3.10 RR and Mode A with the reservoir (summary)

- RR changes only the source weight: NEE candidates at x_d (sampled at x_{d−1}) carry `1/∏_{i=1}^{d−2} q_i`,
  BSDF endings `1/∏_{i=1}^{d−1} q_i`; F and every stored field are RR-free; replay and shifts never apply RR and never
  read slot `u_rr` (math §25).
- Mode A: analytic lights are NEE-only with ω1 ≡ 1 by flag (`nee_mis_w1` → `mis_w1` with `LP_MODE_A`), never crossed,
  never an rc emitter; emissive triangles and the env always use MIS (math §9). Camera-visible analytic lights are
  length-1 terms (`path/length1.wgsl`), never in the reservoir.

---

## 4. Pass graph, bindings, dispatch, submits, pipelines

### 4.1 Frame schedule

```
frame t (all modes; E members in the atlas):
  rs_primary                          8×8 over atlas (row bands allowed)            → rsVbuf, rsGeo, rsL1
  rs_initial   × tree chunks          8×8 over atlas; RsDispatch.treeBase/Count     → res[0]  (+ candDump in tests)
  for r in 0 … rounds−1:              in = res[r%2], out = res[(r+1)%2]
     clearBuffer(shiftArena, 0, 16)   queue-0 header
     rs_pair_accept                   8×8; min-index thread writes both J words + codes of a pair; appends replay items
     rs_args                          1 workgroup                                   → rsArgs q0, hdr.n
     rs_spatial_replay                dispatchWorkgroupsIndirect(rsArgs, 0), 64/wg  (RS_REPLAY = 1)
     rs_spatial_shift                 8×8; loops over slots s; handles PENDING non-replay slots (RS_REPLAY = 0)
     rs_spatial_resample              8×8; writes out; final round: rsShade
  rs_finalize                         8×8; L = L1 + (rounds > 0 ? rsShade : F·W); T15 guards
                                      → accum (+counters) | rsFrame | interactive colour target
  rs_ensemble_stats (ensemble only)   per (member, 16² tile) + level reduction      → ensStats, ensPixel
```
Rung 3.1/3.1b: rounds = 0. Offline (rung 3.2): S = 32, rounds = 3, NS = 6, R = 10. Interactive: S = 1, rounds = 1,
NS = 3, R = 30. The final reservoir buffer index is `rounds % 2`.

### 4.2 Bind groups and per-pass bindings

G0 (explicit, shared by every ReSTIR pipeline): 0 FrameUniforms · 1–3 env (EnvParams, texEnv, sEnv) · 4 RestirParams ·
5 LightsParams · 6 `records` (ro storage) · 7 RsDispatch (uniform, dynamic offset). G1: `SceneGpu.layoutEntries()`
(5 storage buffers + textures) or an empty layout. G3: `DebugResources.layout` (uniform + 1 storage buffer) or empty
(test pipelines with `RS_DUMP_CANDIDATES`). Every storage entry of a pipeline layout counts (gap-perf §4.1).

| Pass | G1 | G2 bindings | SB (G0+G1+G2+G3) | ST | Sampled (ReSTIR) |
|---|---|---|---|---|---|
| rs_primary | scene | 0 rsVbuf (st w) · 1 rsGeo (st w) · 2 rsL1 (st w) · 3 shiftArena rw (BVH counters) | 1+5+1+1 = **8** | 3 | – |
| rs_initial | scene | 0 resOut rw · 1 shiftArena rw (counters) · 2 rsVbuf · 3 rsGeo [· 4 candDump rw, test] | 1+5+2+1 = **9** (test: 1+5+3+0) | 0 | 2 |
| rs_pair_accept | empty | 0 resIn ro · 1 shiftArena rw · 2 rsVbuf · 3 rsGeo · 4 pairTex | 1+0+2+1 = **4** | 0 | 3 |
| rs_args | empty | 0 shiftArena ro · 1 rsArgs rw | **4** | 0 | – |
| rs_spatial_replay | scene | 0 resIn ro · 1 shiftArena rw · 2 rsVbuf · 3 rsGeo · 4 pairTex | 1+5+2+1 = **9** | 0 | 3 |
| rs_spatial_shift | scene | same as replay | **9** | 0 | 3 |
| rs_spatial_resample | empty | 0 resIn ro · 1 resOut rw · 2 shiftArena ro · 3 pairTex · 4 rsShade (st w) · 5 rsVbuf | 1+0+3+1 = **5** | 1 | 2 |
| rs_finalize | empty | 0 resFinal ro · 1 accum rw · 2 counters rw · 3 rsL1 · 4 rsShade · 5 rsFrame (st w) · 6 rsVbuf [· 7 colour (st w), interactive] | 1+0+3+1 = **5** | 1–2 | 3 |
| rs_ensemble_stats | empty | 0 ensStats rw · 1 ensPixel rw · 2 rsFrame · 3 maskTex | 1+0+2+1 = **4** | 0 | 2 |

- Every pass ≤ 9 of 10 storage buffers (PLAN §1.8). `rsArgs` is never bound as storage in the pass that consumes it
  indirectly (usage-scope rule).
- `records` is unchanged by M4 (LUTs, lights cur/prev incl. the env record, alias/pmf cur/prev, id maps, env
  `rowAlias`/`colAlias`/`pdfUV`); ReSTIR M4 reads only the `cur` light slot. Cur/prev swapping in M4 exists only for
  `resA`/`resB` (per spatial round); the previous-frame V-buffer/thr/reservoir planes are M5 and are not allocated.
- Ping-pong: G2 of pair_accept, replay, shift, resample and finalize exist in two prebuilt variants (in = A or B).
- The G-buffer textures are in G2 (not G0) because rs_primary writes them as storage textures.
- Defines: `SCENE_GROUP = 1`, scene/BVH/TEX/ENV defines as `pt-kernel.ts`, `LIGHTS_GROUP = 0`, `LIGHTS_BINDING = 5`,
  `LUT_*` (records kind u32), `RS_RES_IN_BINDING`/`RS_RES_OUT_BINDING`/`RS_ARENA_BINDING`/`RS_ARENA_RW` per pass,
  `RS_REPLAY`, `RS_MODE_B = 0`, and test-only `RS_DUMP_CANDIDATES`, `RS_SHIFT_TRACE`, `RS_RNG_OVERRIDE`.

### 4.3 Dispatch shapes

Per-pixel passes: `@workgroup_size(8, 8, 1)`, `dispatchWorkgroups(⌈atlasW/8⌉, ⌈(rowEnd−rowBase)/8⌉)`, pixel =
`(gid.x, gid.y + rowBase)`. Queue passes: `@workgroup_size(64)`, 2D indirect args (§2.7). The ensemble tile pass:
`@workgroup_size(16, 16)` over (member-local 16² tiles, members). Nothing uses a 1D dispatch over items.

### 4.4 Submit splitting

- A frame is a list of **work units** (`WorkUnit { label, costHint, encode(enc) }`): primary (per row band), each
  initial tree chunk (per row band), each spatial round (its six stages as one unit, or stage × member-row band for
  large atlases; stages stay in order), finalize(+stats). Units of consecutive frames may share a submit.
- `batch-runner.ts` (WP-E) packs units into submits with target 50 ms, budget 100 ms, hard cap 200 ms (PLAN §1.8),
  learning ms/costHint from `onSubmittedWorkDone`; a submit over the hard cap fails the run; over budget is recorded.
- `RestirKernel.beginSubmit()` resets the `RsDispatch` ring cursor; each unit takes the next 256-B slot and writes it
  with `queue.writeBuffer` before the submit (ring of 512 slots; the runner never puts more units in one submit).
- `BatchAccumulator.runBatchUnits(src: { frameUnits(t: number): WorkUnit[]; beginSubmit(): void }, frames: number,
  index?: number)` (added by WP-E): clears accum and counters, runs `frames` frames `t = index·frames + i`, packs units
  into submits (calling `beginSubmit()` before each), reads back like `runBatch` (batch mean = sum / frames). The
  16-B counters keep the PT layout (nonFinite, bvhOverflow, bvhItercap, negative): `rs_finalize` fills nonFinite and
  negative; the runner adds the arena's `RSC_BVH_*` from `readCounters()` and reports every other `RSC_*` in meta.json.
- Tree chunk size and row bands start from `costHint` and adapt; the offline split-dispatch test (U-RIS-2) proves the
  result does not depend on them.

### 4.5 Pipeline variants and compile budget

- Variants per scene: MT/Woop × texture binding counts × {interactive, validation} × debug on/off; plus test-only
  pipelines. Criteria mode, plants, RR, S, NS and rounds are **uniforms** (no recompiles).
- Inline budget per pipeline: ≤ 1 `bsdf_sample` call site, ≤ 2 `bsdf_query` call sites, ≤ 2 `material_eval` call
  sites (shift/replay may loop a 2-iteration evaluation to stay inside it). A composer-level test counts the call
  expressions in each composed ReSTIR entry point (U-PT-BITS part 3).
- Compile time: ≤ 1.5 s per pipeline cold, all M4 production pipelines ≤ 4 s in parallel on the M5 Pro (measured,
  recorded in `budget.json`); test pipelines compile lazily.
- Metal state-size cliff (bsdf-api.md): no kernel holds a full reservoir, two MatEvals and a BsdfCtx at once; records
  are read plane by plane; every GPU test file includes a known-answer canary on the Chrome lane.

### 4.6 TS API (render/restir/kernel.ts, presets.ts; P0 signatures)

```ts
export interface RestirSettings {
  maxBounces: number; rr: boolean; rrMinBounces: number;
  trees: number; rounds: number; slots: number; diskRadius: number;
  criteria: 'enhanced' | '2022'; tau: number; alphaMin: number;
  plant?: { noJ?: boolean; marginalJ?: boolean; wScale?: number };
}
export const RESTIR_PRESETS: Record<'initial' | 'initial-rr' | 'offline' | 'interactive' | 'criteria2022', Partial<RestirSettings>>;
//  initial: trees 1, rounds 0, rr false · initial-rr: + rr true, rrMinBounces 1 · offline: trees 32, rounds 3, slots 6, R 10
//  interactive: trees 1, rounds 1, slots 3, R 30, rr true, rrMinBounces 3 · criteria2022: offline + criteria '2022'
export interface WorkUnit { label: string; costHint: number; encode(enc: GPUCommandEncoder): void }
export interface RestirStage { frameUnits(k: RestirKernel, t: number): WorkUnit[] }   // spatial (WP-C), ensemble (WP-C)
export interface RestirFrameOut { accum?: GPUBuffer; counters?: GPUBuffer; colorTarget?: GPUTexture; ensemble?: boolean }
export class RestirKernel {
  static create(device: GPUDevice, scene: SceneGpu, env: EnvGpuResources, o: {
    settings: RestirSettings; lightMode: LightMode /* 'A' only */; env?: PtEnvOptions; debug?: DebugResources;
    features?: Set<string>; wgslLanguageFeatures?: Set<string>; instrumentation?: { dumpCandidates?: boolean } }): Promise<RestirKernel>;
  readonly lights: LightsGpu; readonly resources: RestirResources; settings: RestirSettings;
  setView(v: { camera: CameraState; width: number; height: number; runSeed: number; members?: number; jitterMode?: JitterMode }): void;
  setSettings(s: Partial<RestirSettings>): void;
  beginSubmit(): void;
  frameUnits(t: number, out: RestirFrameOut): WorkUnit[];
  readCounters(reset: boolean): Promise<RestirCounters>;     // arena header (queues, RSC_*), f_r
  readReservoirs(which: 'final' | 0 | 1): Promise<Uint32Array>;   // tests / inspector
  destroy(): void;
}
```
Interactive: `RestirKernel.interactive(...)` returns a `RestirFramePass` with `encode(encoder, {advanced, accumulate},
timestampWrites?)` mirroring `PtFramePass` (progressive mean into the colour target; t = `frame.seedIndex`).

---

## 5. RNG and seed contract (math §15 [M4 addition])

| Stream | Formula | Used for |
|---|---|---|
| frame key | `key = pcg3d(runSeed ^ member·0x9e3779b9, t, localIdx).xy` (= `path_init_seed`) | tree-0 path seed, jitter seed (`key.x`), resampling key |
| tree seed | s = 0: `key`; s ≥ 1: `pcg4d(key.x, key.y, s, STREAM_TREE = 0x27d4eb2f).xy` | path stream of tree s |
| path | `pcg4d(seed.x, seed.y, vertex·16 + slot, STREAM_PATH).x` (existing) | BSDF dims 0–3 (replayed), NEE 4–8 and RR 9 (never replayed) |
| jitter | `rand2(key.x, 0, STREAM_JITTER)` (existing; = PT for the same member) | camera ray |
| resampling | `pcg4d(key.x, key.y ^ passId·0x9e3779b9, counter, STREAM_RESAMPLE).x` of the **resampling** pixel | streaming RIS, spatial selection |
| pairing | `pcg4d(runSeed ^ member·0x9e3779b9, t, (round<<8)|slot, STREAM_PAIRING = 0x165667b1)` | per-(t, r, m, s) dihedral code and offset |

- Pass ids: `RS_PASS_PRIMARY 0, RS_PASS_INITIAL 1, RS_PASS_TEMPORAL 8 (M5), RS_PASS_SPATIAL 16 (+round)`.
- RIS counters: initial `(tree<<20)|(B<<12)|slotInVertex` (0 = NEE, 1 = BSDF end, 2… = Mode-B crossings);
  spatial `0` canonical, `1 + s` slot s.
- **Replay reproduces the base exactly**: vertex index b of the offset equals the base's b (y₁ ↔ x₁); replay reads only
  slots 0–3 of vertices 1…k−2 (or d−1 for ∅) of the reservoir's `seed`; alpha MASK and Mode-B crossings consume no
  dimension (slots 10–13 stay reserved); τ is fixed (slot 14 unused).
- Sequential validation: t = run-global frame index (batch b covers `[b·F, (b+1)·F)`), member 0; with S = 1 the path
  stream of tree 0 equals the PT's for the same `(runSeed, t, pixel)` (U-RIS-1 relies on it). Ensemble: member id in
  every seed and in the pairing hash. Interactive: t = `frame.seedIndex`.

---

## 6. Tests and gates

### 6.1 Gate-0 registry (PLAN §7.4 M4; "10⁷ per case" = round trips per case in §6.2's case list)

| ID | Owner | Where | Fixture | Pass criterion |
|---|---|---|---|---|
| U-RES-1 | A | tests/restir/layout.test.ts + restir-initial | WGSL composed vs `layout.ts` | offsets, flag packing and pack/unpack round trip identical |
| U-EP-1 | A | restir-initial | 10⁶ random (x, hashes) per light type incl. env | `nee_eval(x, nee_draw(h))` ≡ `nee_sample(x, h)` bitwise |
| U-PT-BITS | A | restir-initial + pt suites | M3 PT fixtures | (1) PT images bit-identical before/after the env-sample/length1 refactors (hash of 64 spp on C0c, C0e, C0m, (x) quads, C0s); (2) `bsdf_query` ≡ public BSDF API to 1e-6, `supp` identical; (3) inline-budget call-site counts (§4.5) |
| U-RIS-1 | A | restir-initial | quad scenes + (i) at 64², S = 1, no RR | per pixel `wSum` = lum(L_PT − L1) of `pt_trace` with the same seed: ≥ 99.99% of pixels within 1e-4 rel (the rest are lobe/edge-boundary divergences, D3), and the sum over the agreeing pixels within 1e-5 |
| U-RIS-2 | A | restir-initial | offline S = 32 on (i) 128² | chunks {32}, {8×4}, {1×32} give bitwise-identical reservoirs and finalize output |
| U-RIS-3 | A | restir-initial | RR on | W = Σw/lum F with the RR factors of §3.10 (recomputed from the candidate dump) to 1e-6 |
| U-RIS-4 | A | restir-initial | C0e / C0c 64² | 3.1 micro: mean of F·W over 2¹⁴ frames vs PT at matched variance, z ≤ 4 (smoke; Gate 3 is the real test) |
| U-SFX-1 | A | restir-initial | t3_cases | for k ≤ d−2: `rcRad == betaS ⊙ endTerm(suffix cache)` ≤ 1e-5; case (b)/(c): stored rcRad/aux == re-evaluated end term ≤ 1e-6 (D6) |
| T2 | B | restir-shift | candidate dumps, t3_cases + scene list | self-replay reproduces base prefix ids and lobes; mismatch rate ≤ 1e-5 and every mismatch is an edge hit (both within 1e-6 of an edge) else LOGIC |
| T3-0 | B | restir-shift | every dumped candidate, self shift p→p | defined; σ equal; \|log2 J\| < 2⁻¹⁶; \|F(ȳ)/F(x̄) − 1\| < 1e-4 per channel |
| T3-1 | B | restir-shift | pairs at disk offsets 1–30 px + pairing partners | round trip defined and σ equal: LOGIC = 0 in ≥ 10⁷ per case; FP-BOUNDARY ≤ 1e-5 |
| T3-2 | — | M5 | temporal round trips | (M5) |
| T3-3/M4 | C | restir-spatial | all pixels × slots × 8 dihedral codes × offsets, 960×540 and 1024² | `A(p,q) == A(q,p)` bitwise; slot acceptedness equal on both sides; `partner(partner(p)) == p`; `RSC_SLOT_MISMATCH = 0` |
| T3-4 (U-CASE) | B | restir-shift | t3_cases variants (§6.2) | each case (a) delta/area/tri/sun, (b), (c) tri/env, (d), (e), (f), deep-NEE, deep-BSDF, ∅-TRI, ∅-ENV, each with k = 2 and k > 2, occurs ≥ 10⁵ times; mixed-lobe and mirror configurations present |
| T3-5 | B | restir-shift | chains i→j→l | `J_{i→l}` vs `J_{i→j}·J_{j→l}` ≤ 1e-4 when signatures agree |
| T3-D | B | tests/restir/rc-dual.ts (runs in the Chrome test page on dumps; own CPU unit tests) | 10⁵ dumped round-trip records per scene | f64 k*, O1/O2/O3 decisions agree with the GPU; disagreement with \|margin_f64\| > 2⁻¹⁶ = LOGIC = 0 |
| T3-ENV | B | restir-shift | t3_cases env + C0r/(xiii)/(xiv) | LOGIC = 0 on (e), (f), ∅-ENV, N1-env (b with env), k = 2 env-from-primary; dual agrees |
| T4 | B | restir-shift | all successful T3 round trips | \|log(J_fwd·J_inv)\| < 1e-4; log2 J histogram reported |
| T5 / U7 | B | restir-shift | 10⁷–10⁸ samples per side, per technique domain (NEE, BSDF_TRI, BSDF_ENV) | `E[J·1{fwd ok}]` = `Pr[inv ok]` and with h = F/(1+F); two-sample z ≤ 4 (Šidák over cases) |
| U5 | B | restir-shift | T3 restricted to light-ending paths per case × light type | round trip + reciprocity as T3/T4 |
| U-11 | B (+E) | restir-shift; Gate 3 plant | random reconnections; rendered plant | J·J⁻¹ = 1 with joint pdfs; the marginal-J plant fails Stage B (§6.3) |
| U-12 | B | restir-shift | V2 materials with Λ near the 1e-5 cutoff, V1 mixes | T3 census LOGIC = 0 |
| U-13 | B | restir-shift | synthetic MatEval with N ≠ Ng | BSDF-sampled shifted segment with Ng·L ≤ 0 < N·L gives F = 0 (support); NEE segment does not |
| Dense PSS | B | restir-shift | t3_cases, maxBounces 2 (d ≤ 4), 32×32 pixel-pair grid, 2²⁰ stratified ū per pair (`RS_RNG_OVERRIDE` rank-1 lattice over the first 12 path dims) | LOGIC = 0; every path class of the toy scene enumerated (class counts reported) |
| T6(a) | C | restir-spatial + tests/restir/mis.test.ts | random k ∈ 1…6, c's, FAILED, p̂ = 0 | production `mis_*` functions: \|Σm − 1\| < 1e-5 at y = X_c and at y = Y_j; f64 `mis-ref.ts` agreement 1e-6 |
| T6(b) | C | restir-spatial | neighbour samples of (i), (v), (xiii) | `p̂_j(X_j)/J_{j→c}` vs `p̂_j(T_{c→j}(Y_j))·J_{c→j}(Y_j)` within 1e-3 |
| T7 | C | restir-spatial | 1D toys (translation, power warp, undefined regions, partial support) with production RIS/MIS WGSL | `E[f(Y)W] = 1` (z-test, 10⁸), W-histogram identity over 64 bins; negative controls (m = 1/M, no cap) fail |
| U-MIS-1 | C | restir-spatial | (i) offline frame | `lum(rsShade) = Σw` to 1e-6 rel per pixel (view 462) |
| T14-M4 | C | tests/restir/pairing.test.ts | generator for W ∈ {254,…,210}, R ∈ {10, 30} | involution on the torus; after all 8 codes and random offsets on 960×540 / 1024², `partner(partner(p)) = p`; unmatched < 1%; radial KS < 0.01 |
| T17 | C | restir-spatial + queue.test.ts | synthetic queue with 5·2²⁰ items; capacity − 1 | every item processed exactly once (2D args); overflow flag set exactly when counter > capacity; args math for n ∈ {0, 1, 64, 65535·64, 65536·64+1} |
| U-ENS-1 | C | restir-spatial | (i) 128², E = 4 | member m of the atlas ≡ sequential frame with member m, bitwise (reservoirs and radiance) |
| U-ENS-2 | C | restir-spatial + npz.test.ts | random atlas | ensStats and ensPixel equal an f64 CPU reduction of the read-back atlas (1e-6 rel); npz round trip through compare.py's loader |
| U-OFF-1 | C | restir-spatial | offline (i) 128² | two runs bitwise identical; `RSC_PENDING_LEFT = RSC_SLOT_MISMATCH = 0`; f_r logged |
| U-DBG-1…3 | D | restir-debug | fixture frames | every view id writes the expected code/value at known pixels; view 461 and 462 are 0 to 1e-6; the probe dump decodes to the reservoir read back |
| T15/T16 | all | every GPU run | – | NaN/Inf = 0, negatives = 0, counters of §2.6 = 0; meta.json config (unbiased preset, no plants unless named, jitter iid-per-run, maxBounces = reference, Mode A) |

LOGIC/FP-BOUNDARY classes are those of math §20 (gap-rc §10.3). T3 dispatches are chunked ≤ 100 ms per submit.

### 6.2 T3 fixtures and cases (WP-B, `make-m4.ts`)

`t3_cases_256` (Mode A, b = 4, flat): Cornell box with Lambert walls; V1 GGX spheres r ∈ {0.1, 0.19, 0.21, 0.3, 0.5};
a V1 constant-mix plastic (Lambert + GGX r 0.3); V2 Principled metal r 0.25 and dielectric r 0.4; a roughness-0 mirror;
a two-sided emissive triangle mesh; point, spot, rect, disk and sun lights; `studio_small_09` env (so P(env) is
clamped). Variants: `_noenv`, `_envonly` (env NEE on and off), `_b2` (dense sweep), `t3_glass_256` (smooth glass
sphere + rough glass r 0.3 sphere; **reported only**, D13). T3/T4/T5 also run on the Stage-B scenes (i), (v) V1,
(v) V2, (vi)-A, (xiii), (xiv) overcast+rect with ≥ 10⁷ round trips each (validation-harness §8 (v) row).

### 6.3 Gate 3 in M4 (rungs 3.1, 3.1b, 3.2)

- **Rungs** (presets of §4.6, Mode A, maxBounces = the package's, env NEE per the package):
  3.1 `initial`; 3.1b `initial-rr` (RR on, rrMinBounces 1); 3.2 `offline` (S = 32, 3 rounds × 6 slots, R = 10, RR off
  per math §25). Per scene the ladder stops at the first failing rung (after its re-run); later rungs are reported
  "not run" and the gate fails.
- **Scenes** (existing packages; env scenes generated by `make-m3c.ts` into `validation/out/m3c/scenes`):
  (i) `cornell_i_512`; (ii) `ii_cornell_point_512`; (iii) `iii_spot_grazing_512`; (iv) `iv_emissive_mesh_512`;
  (v) `v_glossy_v1_sharp_512`, `v_glossy_v1_512`, `v_glossy_v2_512`; (vi) `vi_glass_mirror_A_512`;
  (x) `x_many_lights_512`; (xi) `xi_contact_512`; (xii) `xii_alpha_foliage_512`;
  C0q(d) `c0q_openbox_b13_256` (its `_bg` twin differs only on the Blender side); C0r `c0r_irradiance_256`, `c0r_mirror_256`;
  (xiii) `xiii_spheres_512x256` (heavy-tail); (xiv) `xiv_overcast_b1_512`, `_b3_512`, `_b7_512`, `xiv_overcast_rect_b3_512`,
  `xiv_kloof_b3_512`, `xiv_kloof_rect_b3_512` (kloof: heavy-tail). (xiv) Mode-B and glass variants are M6.
  Plus one gating **ensemble unit**: `c0q_openbox_b13_256` rung 3.2 rendered in ensemble mode (E = 16, 256²) and
  compared through `ensemble.npz` against the same PT reference.
  Plus one gating **2022-criteria unit**: (i) with preset `criteria2022`.
- **Statistics**: compare.py stage `B` (math §28): TOST α = 0.01, δ 0.2% global / 1% per 32² tile, channels Y/R/G/B,
  Šidák tiles, χ²_red, mean-t, KS/AD under the suite FWER `α_u = 1 − 0.99^{1/n_units}` (n_units = units × 4 channels,
  including plants and A/A); `num_eps = 1e-4` (D2 of cycles-deviations: two f32 implementations); `min_replicates` 16
  (heavy-tail 32); a failed unit is re-run once on disjoint seeds (both sides).
- **References**: our PT (`--kernel pt`), RR off, same package, seed 4001 (re-run 104001); ReSTIR seed 4002 (re-run
  104002). PT references are cached under `validation/out/m4/ptrefs/<pkg>-<configHash>/` and shared by all rungs.
- **Sizing** (PLAN §7.3): a ~5% pilot per unit and side gives SE; `stats.required_replicates` with δ_tile = 1%,
  m = #tiles, sets spp (PT) and frames per batch (ReSTIR) with B = 16 (heavy-tail 32) batches, splitting the SE budget
  by cost. If a unit needs more than 60 min on one side, enlarge its tiles to 64² and record `aggregate_enlarged`;
  δ is never loosened. The sizes are written to `report.json` and `budget.json` rows.
- **Planted controls** (must be detected; half-size repeats via `compare.py --calibrate --planted`, ≥ 9/10, and the
  full comparison must fail): (1) **omitted spatial Jacobian** (`RSF_PLANT_NO_J`, rung 3.2) on (i) and (v) V1;
  (2) **marginal pdfs in J** (`RSF_PLANT_MARGINAL_J`, U-11 negative control) on (v) V2; (3) synthetic **W × 1.003**
  (compare.py stage-B default plant) on (i) rung 3.2. **A/A**: two ReSTIR seed sets on (i) rung 3.2 must pass, and
  `--calibrate` A/A re-splits pass at the nominal rate.
- **Other invariants per unit**: T15/T16, arena counters of §2.6 = 0, BVH overflow = 0, no submit over the hard cap,
  f_r and timings recorded.

### 6.4 `npm run validate -- --milestone M4` (gate-m4.ts)

1. **Gate 0**: typecheck; `vitest --project cpu` (tests/restir + regressions); python stats tests; package determinism
   of `make-m4.ts`; Chrome GPU suites `restir-initial`, `restir-shift`, `restir-spatial`, `restir-debug` and the M3
   regressions `pt`, `bsdf`, `lights`, `env-sampling`, `glass`, `pt-glass` (they prove the refactors are
   bit-identical).
2. **Gate 3**: PT references (cached), then per scene rungs 3.1 → 3.1b → 3.2 through
   `run-batches.ts --kernel restir --preset <rung> [--members E] [--plant no-j|marginal-j] [--w-scale s]`, then
   compare.py stage B; the ensemble unit and the 2022 unit.
3. **Calibration and plants** (§6.3).
4. **App smoke** `m4-app-smoke.ts` (WP-D): render mode ReSTIR on Cornell and an HDRI scene, every M4 view renders
   without NaN, the inspector dump decodes, HUD shows f_r.
5. Output `validation/out/m4-gate-<time>/summary.{json,md}`: every unit with rung, sizes, MDB, pass/fail, re-runs,
   plants, counters, budget; `--only pkg,…` restricts Gate 3 (debugging, skips Gates 0 and calibration).
The GPU lock is taken by the scripts that use the GPU (as in M3). Expected cost: Gate 0 ≈ 45 min, Gate 3 first run
≈ 6–10 h (PT references dominate; re-runs with cached references ≈ 2–4 h): the weekly tier of PLAN §7.3.

### 6.5 Decision: (vi) and (xii) in M4 vs rungs 3.9/3.10 in M6

PLAN §5 M4 exit lists "(i)–(vi), (x), (xi), (xii)" while PLAN §7.1 puts rung 3.9 (glass) and 3.10 (alpha) in M6.
Resolution (D13): the two statements are compatible once rungs are read as feature coverage. M4 gates (vi) in
**Mode A** and (xii) at rungs 3.1/3.1b/3.2 because the M4 code paths already contain everything those scenes
exercise (smooth glass and mirror = delta replay and forced NEE; MASK = traversal). M6's rung 3.9 covers what only M6
adds or stresses: Mode B (vi), the glass Stage-B scenes C0h–C0k and G1–G10, rough-glass G_R reconnection and glass
side flips as gating T3 cases, the (xiv) glass variant; rung 3.10 covers alpha cutouts on reconnection/shift segments
as T3 cases and large-foliage scenes. If (vi)-A fails in M4 for a glass-specific reason, the coordinator may move it to
M6 by amendment (open question Q2); the unit is never silently dropped.

---

## 7. Risks and open decisions

| # | Risk / decision | Recommendation |
|---|---|---|
| R1 | f32 precision of J (products of pdfs up to ~1e6 and G up to 1e12) | Compute F and J separately with guards; FJ = F·J; T4 at 1e-4, T3-5 chains; no clamping. Reciprocal round-off is ~1e-6. |
| R2 | Queue capacity | Worst case sized (P·NS items); overflow flag fails validation; T17 covers > 4.19 M items and the 2D args. |
| R3 | Compile time (BSDF inlined per site; replay + two reconnection evaluations) | Inline budget of §4.5 checked by a test; parallel async compile; test pipelines lazy; record timings in budget.json. |
| R4 | Metal silent miscompile from per-thread state size (bsdf-api.md) | Plane-wise reservoir access, `bsdf_query` with one prepare, sequential MatEvals, known-answer canaries in every Chrome GPU test file (T3-0 is a strong one). |
| R5 | Replay cost (30% replay ≈ +17 ms at 720p) and offline S = 32 cost (~32× initial) | Compacted queue (measured 1.39×); f_r logged; tree chunks for the submit budget; offline only in validation. |
| R6 | Where the f64 dual lives | `tests/restir/rc-dual.ts`, pure TS on top of `tests/material/{bsdf-ref,glass-ref}.ts`; runs inside the Chrome GPU test on dumps (no Node APIs) and in the cpu lane on hand-built configurations. Textured materials are skipped and counted. |
| R7 | FP-BOUNDARY rate vs the 1e-5 limit (different pipelines contract FMAs differently) | D3 same-formula positions, stored thr, integer compares, `>=` everywhere; measure early at milestone B2; if it exceeds 1e-5, implement the single-engine build (D21) for T3. |
| R8 | Stage-B cost (δ 0.2%/1% needs ~4× the Stage-A PT samples per side) | Cache PT references; pilots; heavy tiles → 64² aggregates recorded; weekly tier. |
| R9 | Memory at E = 64 (≈ 2.4 GB) | E = 16 default; M4 gates do not need ensembles; E = 64 only after the M0 probe. |
| R10 | Duplicate camera ray in interactive ReSTIR (M1 primary + rs_primary) | Accept for M4 (≈ 1–2 ms at 540p); M8 lets rs_primary feed the app's G-buffer views. |
| R11 | Glass/alpha bugs block M4 through (vi)-A/(xii) | Kept in the gate per PLAN; escalation path Q2. |
| R12 | BSDF_TRI ∅-replay technique test | "Same technique" = hit on a `TRI_EMISSIVE` triangle (geometry property, symmetric); L_e = 0 texels give a defined zero. |
| R13 | `lobeHist` and endpoint fields of spatially selected ∅/k = d samples describe the source path | Documented as base-path/debug only; nothing in M4 reads them for correctness; M5 must re-derive (class R/L/E use replay or the shift). |
| R14 | U-RIS-1 tolerance: ReSTIR base uses position-derived V, the PT uses −ray direction | Rare lobe-boundary divergences; the criterion allows 1e-4 of pixels; the global sum is the strong check. |
| R15 | Stubbed shift during WP-C development hides MIS bugs that need VALID slots | WP-C's T6/T7/U-MIS-1 use synthetic slot buffers; integration run after B2. |

Open questions for the coordinator:
- **Q1** Who lands P0 (recommended: WP-A, one day, from appendix B verbatim, including `bsdf_query`), or whether the
  coordinator lands it before spawning the WPs.
- **Q2** (vi)-A and (xii) gating in M4 (D13, recommended yes) or deferral to M6.
- **Q3** Acceptance of the Gate-3 compute budget (weekly tier) and of 64² aggregate enlargement for the heaviest
  units (sharp (v), (xiii), kloofendal) if the pilot demands it.
- **Q4** Pairing radii R = 10 px (offline) and 30 px (interactive) (D9).
- **Q5** Pin of the RR survival formula to the PT's `min(sqrt(max_c β_c), 1)` (D11, resolves math open item 34).
- **Q6** Whether the 2022-criteria unit gates M4 (recommended yes: it is the A|B baseline and must be unbiased).

---

## Appendix A. math.md additions made with this contract

All are marked **[M4 addition, restir-api.md]** in math.md: §15 (tree seeds, resampling and pairing stream formulas,
RIS counters), §16 (same-formula directions D3, base NEE endpoint encoding D4, streaming finalisation with S trees,
suffix-cache definitions), §17 (64-bit seed, endpointId, encodings, empty/background), §20 (status taxonomy: defined-
zero shifts carry FAILED), §22 (A0 z = stored camera distance, once-per-pair evaluation by the min thread, M4 disk
maps), §25 (RR formula pin), open inconsistencies 36–42.

## Appendix B. Normative WGSL blocks (P0 lands them verbatim)

### B.1 `restir/types.wgsl` (constants)

```wgsl
// Techniques (math.md#path-tree)
const RS_TECH_NEE: u32 = 0u;  const RS_TECH_BSDF_TRI: u32 = 1u;  const RS_TECH_BSDF_ANALYTIC: u32 = 2u;  const RS_TECH_BSDF_ENV: u32 = 3u;
// rc vertex kinds (rcPairTest)
const RCK_SURFACE: u32 = 0u;  const RCK_LIGHT: u32 = 1u;  const RCK_ENV: u32 = 2u;
// rc / endpoint word-A tags (§2.3)
const RC_TAG_NEE: u32 = 0x80000000u;  const RC_TAG_CROSS: u32 = 0xC0000000u;  const RC_TAG_MASK: u32 = 0xC0000000u;
const RC_ENTRY_MASK: u32 = 0x3FFFFFFFu;  const RC_ENV_DIR: u32 = 0xFFFFFFF1u;  const RC_NONE: u32 = 0xFFFFFFFFu;
const RS_ENV_ID: u32 = 0xFFFFFFFEu;
// Reservoir flags (§2.2)
const RF_D_SHIFT: u32 = 0u;   const RF_K_SHIFT: u32 = 4u;   const RF_TECH_SHIFT: u32 = 8u;  const RF_EP_SHIFT: u32 = 10u;
const RF_ISDELTA: u32 = 0x2000u;  const RF_LKM1_SHIFT: u32 = 14u;  const RF_DKM1: u32 = 0x20000u;
const RF_LK_SHIFT: u32 = 18u;  const RF_DK: u32 = 0x200000u;  const RF_MODE_SHIFT: u32 = 22u;
const RF_FORCED: u32 = 0x1000000u;  const RF_BG: u32 = 0x2000000u;
// J-word statuses (§2.6)
const JW_FAILED: u32 = 0u;  const JW_PENDING: u32 = 0xFFFFFFFEu;  const JW_NOT_ACCEPTED: u32 = 0xFFFFFFFFu;
// Slot outcome codes
const SC_OK: u32 = 0u;  const SC_NOT_ACCEPTED: u32 = 1u;  const SC_EMPTY_SRC: u32 = 2u;  const SC_O0_MISS: u32 = 3u;
const SC_O0_LOBE: u32 = 4u;  const SC_O0_TECH: u32 = 5u;  const SC_O0_LIGHT: u32 = 6u;  const SC_O1: u32 = 7u;
const SC_O2: u32 = 8u;  const SC_O3: u32 = 9u;  const SC_OCCLUDED: u32 = 10u;  const SC_ZERO: u32 = 11u;
const SC_J_INVALID: u32 = 12u;  const SC_O0_SUPPORT: u32 = 13u;  const SC_PENDING: u32 = 14u;  const SC_NONFINITE: u32 = 15u;
// Predicate sub-tests (RcResult.term, slot code bits 8–11)
const RCT_NONE: u32 = 0u;  const RCT_D: u32 = 1u;  const RCT_R: u32 = 2u;  const RCT_F: u32 = 3u;  const RCT_I: u32 = 4u;  const RCT_GUARD: u32 = 5u;
// Arena counters (header word index)
const RSC_CAND_NONFINITE: u32 = 16u;  const RSC_BVH_OVERFLOW: u32 = 17u;  const RSC_BVH_ITERCAP: u32 = 18u;
const RSC_SHIFT_NONFINITE: u32 = 19u;  const RSC_PENDING_LEFT: u32 = 20u;  const RSC_SLOT_MISMATCH: u32 = 21u;
const RSC_W_NONFINITE: u32 = 22u;  const RSC_BASE_JDEN_INVALID: u32 = 23u;  const RSC_ACCEPTED: u32 = 24u;
const RSC_QUEUED: u32 = 25u;  const RSC_SELECTED_SHIFTED: u32 = 26u;  const RSC_EMPTY_CANON: u32 = 27u;  const RSC_CODE_BASE: u32 = 32u;
// RestirParams.flags / RsDispatch.flags
const RSF_RR: u32 = 1u;  const RSF_CRIT_2022: u32 = 2u;  const RSF_PLANT_NO_J: u32 = 4u;  const RSF_PLANT_MARGINAL_J: u32 = 8u;
const RSF_ENSEMBLE: u32 = 16u;  const RSF_INTERACTIVE: u32 = 32u;  const RSF_J_REJECT: u32 = 128u;
const RSD_FIRST_CHUNK: u32 = 1u;  const RSD_FINAL_CHUNK: u32 = 2u;  const RSD_FINAL_ROUND: u32 = 4u;
// Pass ids (RNG stream separation, §5) and stream constants
const RS_PASS_PRIMARY: u32 = 0u;  const RS_PASS_INITIAL: u32 = 1u;  const RS_PASS_TEMPORAL: u32 = 8u;  const RS_PASS_SPATIAL: u32 = 16u;
const STREAM_TREE: u32 = 0x27d4eb2fu;  const STREAM_PAIRING: u32 = 0x165667b1u;
// Limits
const RS_MAX_D: u32 = 15u;  const RS_MAX_SLOTS: u32 = 6u;  const RS_MAX_TREES: u32 = 64u;  const RS_MAX_ROUNDS: u32 = 4u;
const RS_RES_PLANES: u32 = 10u;  const RS_ARENA_HDR_WORDS: u32 = 64u;  const RS_WG: u32 = 64u;
const RS_DUMP_CAP: u32 = 32u;  const RS_DUMP_WORDS: u32 = 48u;
```

### B.2 `restir/reservoir.wgsl` (layout and accessors)

```wgsl
// Plane indices of a record (§2.2)
const RP_WF: u32 = 0u;  const RP_SEED: u32 = 1u;  const RP_RC: u32 = 2u;  const RP_WI: u32 = 3u;  const RP_RAD: u32 = 4u;
const RP_END: u32 = 5u;  const RP_SFX0: u32 = 6u;  const RP_SFX1: u32 = 7u;  const RP_SFX2: u32 = 8u;  const RP_DIAG: u32 = 9u;
#if RS_RES_IN_BINDING
@group(2) @binding($RS_RES_IN_BINDING) var<storage, read> resIn: array<vec4u>;
fn resin_plane(i: u32, p: u32) -> vec4u { return resIn[i * RS_RES_PLANES + p]; }
#endif
#if RS_RES_OUT_BINDING
@group(2) @binding($RS_RES_OUT_BINDING) var<storage, read_write> resOut: array<vec4u>;
fn resout_plane(i: u32, p: u32) -> vec4u { return resOut[i * RS_RES_PLANES + p]; }
fn resout_set(i: u32, p: u32, v: vec4u) { resOut[i * RS_RES_PLANES + p] = v; }
#endif
fn rf_d(f: u32) -> u32 { return f & 0xFu; }
fn rf_k(f: u32) -> u32 { return (f >> RF_K_SHIFT) & 0xFu; }
fn rf_tech(f: u32) -> u32 { return (f >> RF_TECH_SHIFT) & 3u; }
fn rf_ep(f: u32) -> u32 { return (f >> RF_EP_SHIFT) & 7u; }
fn rf_lkm1(f: u32) -> u32 { return (f >> RF_LKM1_SHIFT) & 7u; }
fn rf_lk(f: u32) -> u32 { return (f >> RF_LK_SHIFT) & 7u; }
fn rf_pack(d: u32, k: u32, tech: u32, ep: u32, isDelta: bool, lkm1: u32, dkm1: bool, lk: u32, dk: bool, forced: bool) -> u32;
fn res_empty(f: u32) -> bool { return rf_d(f) == 0u; }
/// Streaming RIS update (math.md#path-tree): true if the candidate with weight w replaces the selection.
fn ris_update(wSum: ptr<function, f32>, w: f32, u: f32) -> bool {
  if (!rs_pos_finite(w)) { return false; }
  *wSum = *wSum + w;
  return u * *wSum < w;
}
```
(`rf_pack` body and the TS mirror are straightforward; U-RES-1 checks them. `rs_pos_finite` lives in types.wgsl in
P0 and is re-exported by rc.wgsl.)

### B.3 `restir/frame.wgsl` (uniforms)

```wgsl
struct RestirParams {          // 128 B, G0 binding 4 (layout.ts RESTIR_PARAMS_SIZE)
  atlasSize: vec2u,            //   0
  memberSize: vec2u,           //   8  (= frame.resolution)
  memberCols: u32,             //  16
  memberCount: u32,            //  20  E
  maxBounces: u32,             //  24
  flags: u32,                  //  28  RSF_*
  numTrees: u32,               //  32  S
  numSlots: u32,               //  36  NS
  numRounds: u32,              //  40
  rrMinBounces: u32,           //  44
  tau: f32,                    //  48  2e-4
  alphaMin: f32,               //  52  0.2
  wScale: f32,                 //  56  plant (1 = off)
  crit2022MinDist: f32,        //  60
  pairTexSize: vec4u,          //  64  layers 0..3
  pairTexSize2: vec4u,         //  80  layers 4..7
  lightMode: u32,              //  96  0 = A
  pad0: u32, pad1: u32, pad2: u32,   // 100..111
  pad3: vec4u,                 // 112..127
}
struct RsDispatch {            // 32 B used, 256 B ring stride, G0 binding 7 (hasDynamicOffset)
  t: u32, passId: u32, round: u32, treeBase: u32,
  treeCount: u32, flags: u32, rowBase: u32, rowEnd: u32,
}
@group(0) @binding(4) var<uniform> rsParams: RestirParams;
@group(0) @binding(7) var<uniform> rsDispatch: RsDispatch;
```

### B.4 Shift arena views (restir/queue.wgsl declares the binding per pass)

```wgsl
struct ShiftArenaRW { hdr: array<atomic<u32>, 64>, words: array<u32> }
struct ShiftArenaRO { hdr: array<u32, 64>, words: array<u32> }
// slots at words[4·(ai·NS + s) .. +3]; codes at words[4·P·NS + ai·NS + s]; items at words[5·P·NS + i]
// queue q header at hdr[4q .. 4q+3] = {counter, n, capacity, overflow}
```

---

## Changelog

Amendments made while implementing the contract. Numbering is append-only; every WP reads this section before
touching a shared interface. "WP-A" entries were made while landing P0 / A1.

### Coordinator decisions on §7 (2026-09-29)

- **Q1** WP-A lands P0 (this changelog's P0 entries).
- **Q2** (vi)-A and (xii) gate in M4, in Mode A (D13 stands).
- **Q3** The weekly-tier Gate-3 budget is accepted. 64² aggregate tiles are allowed for the heaviest units only if the
  pilot demands it, recorded in `report.json`; δ is never loosened.
- **Q4** Pairing radii R = 10 px (offline) and 30 px (interactive) are accepted (D9).
- **Q5** The RR survival formula is pinned for ReSTIR (D11).
- **Q6** The 2022-criteria unit gates M4.

### P0 amendments (WP-A)

- **A1 Interactive finalize bits.** `RsDispatch.flags` of the `rs_finalize_frame` dispatch carries
  `RSD_ACCUMULATE = 8` and `RSD_ADVANCED = 16` (progressive mean, as `PtFramePass`). Appended to B.1 / `layout.ts`.
- **A2 Suffix flag names.** `SFX_BSDF_END = 1, SFX_ESCAPE = 2, SFX_VALID = 4` (word 27, §2.2). Appended to B.1.
- **A3** `RS_HIST_NONE = 0xFFFFFFFF` (lobeHist with no event). Appended to B.1.
- **A4 Binding defines are `'<n>u'` strings.** The composer's `#if` treats the number 0 as false, so
  `#if RS_RES_IN_BINDING` (B.2) would drop a binding at slot 0 (`rs_pair_accept`, `rs_finalize`, `rs_args`).
  All binding defines (`RS_RES_IN_BINDING`, `RS_RES_OUT_BINDING`, `RS_ARENA_BINDING`, `RS_VBUF_BINDING`, …) are
  strings such as `'0u'` (truthy, and `@binding(0u)` is valid WGSL). **Always** build defines with
  `resources.ts` `restirDefines(name, …)` / `restirPassDefines(name)` / `RestirKernel.defines()`; never pass numbers.
  B.2 stays verbatim.
- **A5 `RestirParams.memberBase`** (offset 100, was `pad0`): member id of atlas member 0, so a sequential run can
  render member m (U-ENS-1). `rs_pix().member = memberBase + atlas member index`; seeds and the pairing hash use it.
- **A6 Finalize estimate source.** `rs_finalize` uses `rsShade` iff the spatial stage emitted work units for the
  frame; the finalize dispatch's `RsDispatch.round` = number of executed rounds and it reads `res[executed % 2]`.
  With the P0 stub stage (`frameUnits → []`) the output is the canonical `F·W` of `res[0]` (unbiased). WP-C's stage
  must emit all rounds or none.
- **A7 `RcResult.pass` → `RcResult.ok`.** `pass` is a reserved word in WGSL. §3.4's struct is
  `struct RcResult { ok: bool, margin: f32, term: u32 }`.
- **A8 `rs_t()`** returns `frame.seedIndex` when `RSF_INTERACTIVE` is set (the renderer owns the frame uniforms),
  else `RsDispatch.t`. Every pass (incl. pairing) must use `rs_t()`, never `rsDispatch.t` directly.
- **A9 Stage interface and kernel API for stages / tests** (`kernel.ts`):
  `interface RestirStage { frameUnits(k, t): WorkUnit[]; prepare?(k): Promise<void>; destroy?(): void }` —
  `prepare` compiles the stage's pipelines; `RestirKernel.create()` / `prepare()` calls it only when the settings
  need the stage (spatial: rounds > 0; ensemble: E > 1), so a half-written stage never breaks rung-3.1 users.
  Helpers: `k.pipeline(name, extraDefines?, colorFormat?)` (async, cached), `k.pipelineSync(name)`,
  `k.resources.g2(name, inIdx, {accum, counters, colour})` (ping-pong: resample reads `res[inIdx]`, writes the other),
  `k.encodePass(enc, name, pipeline, g2, dispatch, [x, y] | {indirect, offset})` (takes a ring slot),
  `k.perPixelWorkgroups(r0, r1)`, `k.rowBands()`, `k.dispatchSlot(d)`. Tests / tools: `k.compile(file, entry,
  defines, layout, label, extraSources)`, `k.customLayout(g2Layout, scene)`, `k.customDefines(extra, scene)`,
  `k.encodeCustom(...)`. The pass table (`resources.ts RS_PASSES`) holds file, entry, G1/G3 use, the G2 layout and
  binding defines of every pass of §4.2; `rs_initial_dump` (test variant, no G3) and `rs_finalize_frame`
  (interactive, `RS_INTERACTIVE`, G2 binding 7 = colour) are pass names of their own.
- **A10 Queue details** (`queue.wgsl`, real in P0): the arena variable is `rsArena`; the capacity of every queue is
  `P·NS` computed from `rsParams` (the per-round `clearBuffer(arena, 0, 16)` also zeroes the header's capacity
  word, so it cannot be the source); `rs_count(c, n)` adds to an `RSC_*` counter; `queue_item` returns the item
  **index** (read the word with `arena_word(arena_item_word(i))`); helpers `slot_index`, `arena_slot_word`,
  `arena_code_word`, `arena_item_word`, `queue_capacity`. `rs_slot_code(sc, term, pair, margin)` packs §2.6's code
  word (types.wgsl).
- **A11 G-buffer texture declarations** live in `restir/frame.wgsl`, one define each: sampled `rsVbuf`, `rsGeo`,
  `rsL1`, `rsShade`, `rsFrame`, `pairTex` (`RS_*_BINDING`, `RS_PAIRTEX_BINDING`) and storage `rsVbufOut`,
  `rsGeoOut`, `rsL1Out`, `rsShadeOut`, `rsFrameOut` (`RS_*_W_BINDING`); `rs_vbuf(px)`, `rs_geo(px)` load texels.
- **A12 P0 bodies beyond §1.4.** `rc.wgsl`: `vertex_from_ids`, `rc_vertex`, `rc_event_*`, `rc_G` and
  `primaryThreshold` are real; `kstar_nee` / `kstar_tree_pair` / `kstar_bsdf_end` are real **in terms of
  `rcPairTest`** (which is the failing stub), so WP-B's B1 only has to replace `rcPairTest` (it may refine the rest).
  `pairing.wgsl`: `dihedral_apply(_t)` and `pair_A0` real; `mis.wgsl`: the two MIS term functions and
  `res_select_shifted` real; `endpoint.wgsl` is complete.
- **A13 Interactive API.** `RestirKernel.interactive(device, scene, env, colorFormat, o)` → `RestirFramePass` with
  `setTargets({width, height, color, frameUniforms})`, `encode(encoder, {advanced, accumulate}, timestampWrites?)`
  (one submit per frame: it resets the RsDispatch ring; timestamps bracket all ReSTIR passes), `setLights`,
  `setEnvironment`, `envParamsChanged`, `setEnvOptions`, `setSettings` + `await prepare()` when rounds go 0 → > 0.
  `RestirFrameOut` gains `interactive?: { advanced, accumulate }`.
- **A14** `pass-primary.ts`, `pass-initial.ts`, `pass-finalize.ts` are folded into `kernel.ts frameUnits` (no
  separate files).
- **A15** `res_write_empty(i, seed, bg)` (reservoir.wgsl) writes the empty / background record (all ten planes;
  endpoint triple and endpointId `RC_NONE`, lobeHist `RS_HIST_NONE`, jDen 1). The candidate dump binding
  (`candDump`, G2 binding 4) is declared in `path/pathtree.wgsl`.

### A1 amendments (WP-A)

- **A16 U-RIS-1 split** (§6.1). With D3 the base path's directions come from positions rebuilt from ids, while the PT
  uses the sampled direction and −(camera ray); the offset ray origin alone makes them differ by ~1e-5 rad, which
  glossy lobes amplify to 1e-4–1e-3 relative in 2–3% of pixels (measured, 64², 8 frames). R14's "rare lobe-boundary
  divergences" underestimated this. U-RIS-1 therefore runs twice: **(a)** with the test-only define
  `RS_PT_DIRECTIONS` (pathtree uses the PT's directions and vertex orientation): ≥ 99.98% of pixels within 1e-4 rel
  and the sum over them within 1e-5 (measured 99.98–100%, sums 1e-10–5e-8; the remaining pixels are isolated
  inter-pipeline FP events ≤ 3e-3 rel); **(b)** production D3: ≥ 99% within 1e-3 and the sum over them within 1e-4
  (measured 99.70–99.95%, sums ≤ 1e-5). The streaming-RIS bookkeeping is proven by (a) and U-RIS-3.
- **A17** `RestirKernelOptions.instrumentation` gains `initialDefines` (extra defines of `rs_initial` /
  `rs_initial_dump`, e.g. `RS_PT_DIRECTIONS`, `RS_RNG_OVERRIDE`) and `extraSources` (in-memory WGSL that overrides
  or adds shader files for every pipeline of that kernel, e.g. a test `rc.wgsl` or the dense-PSS `rs_rng_override`).
- **A18 Candidate dump details** (§2.12): only candidates with a finite w > 0 are streamed, counted (`nCand`) and
  dumped; the dump's `wSum` word is Σw before the candidate, `W` its w_i, `nCand` its ordinal; words 40–47 hold x₁…x₈
  (`0xFFFFFFFF` beyond x_{d−1}); the per-pixel count is at word `P·32·48 + ai` (`layout.ts dumpCountWord`).
  Candidates with w = 0 (e.g. F underflow) are neither streamed nor dumped.

### WP-B amendments (shift core)

- **B-1 rc.wgsl conventions** (affects WP-A: `kMargin`; WP-C/WP-D: slot-code margin field, views 410/440–445).
  `rcPairTest` reads α_min, `RSF_CRIT_2022` and `crit2022MinDist` from `rsParams` (rc.wgsl includes
  `restir/frame.wgsl`; every pipeline that includes rc.wgsl has G0). A **discrete** failure (delta event, G_T lobe,
  failed guard) reports margin `RC_MARGIN_DISCRETE = −1024` (a module constant of rc.wgsl, mirrored in
  `tests/restir/rc-dual.ts`) instead of 0, so it can never be classed FP-BOUNDARY; every margin is clamped to ±1024
  (fits the f16 slot-code field). F and I are decided as `t² ≥ thr·(p̄·|cos|)` on bit patterns (the same decision as
  `t²/(p̄·|cos|) ≥ thr`, with no division by a tiny product and no Inf); the margin is `log2(t²) − log2(thr·p̄·|cos|)`.
  PLAN rule 5 "v1: G_T never passes" is enforced in `rcPairTest` itself: `ℓ_{k−1} = G_T` fails D, `ℓ_k = G_T` fails
  the I branch (also in 2022 mode).

- **B-2 D3 strengthened: throughput factors from positions** (affects **WP-A** `path/pathtree.wgsl`, minimal change
  made by WP-B; WP-C T6(b); U-RIS-1 tolerances). D3 allowed the base path's throughput to use the sampler's `weight`
  (f/p at the SAMPLED direction) while every shift recomputes f/p at a reconnection vertex at the POSITION-DERIVED
  direction `normalize(pos(x_{k}) − pos(x_{k−1}))` (resp. the stored ω_k). The two directions differ by the ray-offset
  angle (Wächter–Binder origin offset / t), and near a grazing direction f/p is steep, so `F(x̄)` from the path tree and
  `F(T⁻¹T x̄)` from a round trip differed by up to 7e-2 relative (T3-0 self shifts on the all-lights box; WP-C's T6(b)
  failure on the C0s box, 1.387e-3 on one deep NEE→env path with ℓ₁ = S, seen from both pixels of its pair). That is
  a target/integrand defined by two formulas (a measure-small bias), not FP noise. Now **every** BSDF-sampled
  throughput factor in the path tree (`beta`, `betaPost`) and in replay (`Tp`) is `rs_path_weight(q, weight, δ)`
  (new WP-B module `path/path-weight.wgsl`, included by pathtree and replay) = `bsdf_query(m, V, wOut, ℓ).f_lobe / p_joint` at the position-derived `wOut` (the escape direction on a
  miss), the formula the shift uses at y_{k−1} and x_k; the sampler's weight is kept only for delta lobes and when
  `p_joint(wOut)` is not finite-positive (measure-small, counted by `RSC_BASE_JDEN_INVALID` in the base). F is then one
  function of the stored vertices everywhere, and T3-0 / T3-1 check `|F(x̄')/F(x̄) − 1| < 1e-4` without exceptions. The
  PT keeps the sampler's weight; the ReSTIR base differs from it per sample by the ray-offset effect (R14).
- **B-3 T3 fixtures** (§6.2, `make-m4.ts`): added `t3_rare_256` (glossy r 0.1 / 0.12 floor and back wall so pair 2 fails R:
  k > 2, ∅-TRI and ∅-ENV are frequent; an emissive panel) and `t3_cutoff_256` (U-12: V2 metallic 1 − 1.5e-5 and
  1 − 0.8e-5, V1 mix 1 − 1.2e-5, specular level 1e-5, base colour 1.2e-5, i.e. lobe weights on both sides of the
  1e-5 closure cutoff). `t3Scene()` is browser-safe (the Chrome test builds the scenes directly; the CLI writes packages).
- **B-4 T3 classification** (§6.1 T3, gap-rc §10.3): a round-trip failure of the inverse is FP-BOUNDARY iff the
  deciding pair has |margin| < 2⁻¹⁶ or the replay diverged at a triangle edge (barycentric distance < 1e-6 of the
  offset's hit); everything else (incl. an inverse ZERO/OCCLUDED after a forward OK, J or F reciprocity > 1e-4) is
  LOGIC. In the f64 dual (T3-D) a disagreement is FP-BOUNDARY iff |margin_f64| < 2⁻¹⁶ or the pair is **tangent**
  (|cos| < 1e-5 between the pair segment or an event direction and a geometric normal: the sampler supports and the
  two-sided flip switch there, so f32 vertex rounding decides; e.g. both endpoints on the same wall).

### WP-C amendments (paired spatial reuse, queues, ensemble)

- **C1 Arena write access of `rs_args` and `rs_spatial_resample`; queue clear** (affects WP-A: `resources.ts`
  `RS_PASSES` edited by WP-C, two entries; WP-D/WP-E: header semantics). §2.7 makes `rs_args` write `hdr.n` and the
  overflow flag, and §3.8 makes the resample count `RSC_PENDING_LEFT` / `RSC_SLOT_MISMATCH`, but §4.2 bound the arena
  read-only in both passes. Both now bind it `rw` (`RS_ARENA_RW: true`; storage-buffer counts unchanged: 4 and 5).
  The per-round queue clear is `clearBuffer(arena, 0, 8)` (counter and n only): the overflow flag is **sticky** until
  `readCounters(true)`, so an overflow in an early round cannot be erased by a later one; `rs_args` writes
  `hdr.capacity`. `rs_args` handles q0 only (M5 adds its queues).
- **C2 Reciprocity check in `rs_pair_accept`.** The thread also evaluates `partner(partner(p))`; if it is not p (never
  for an involution map) its own slot is NOT_ACCEPTED and `RSC_SLOT_MISMATCH` is incremented, so every slot is still
  written exactly once. Counters of the spatial passes are aggregated per workgroup before the global atomics.
- **C3 `res_needs_replay` in `rs_pair_accept`** (affects WP-B). pair-accept has no scene group and cannot include
  `shift.wgsl`; it carries a copy `pa_needs_replay` of the flags predicate (non-empty ∧ (k > 2 ∨ k = ∅)). A
  divergence would leave PENDING slots (`RSC_PENDING_LEFT`); restir-spatial checks both predicates agree on all
  flags. WP-B: if `res_needs_replay` changes, tell WP-C (or move the predicate to `reservoir.wgsl`).
- **C4 W-scale plant reaches the image.** The final-round `rsShade` is multiplied by `rsParams.wScale` as well as W
  (finalize reads rsShade when rounds > 0, so scaling W alone would change nothing visible in the last round).
- **C5 T14-M4 radial KS reference** (§2.8, §6.1): the KS distance is measured against the **lattice-uniform** radial
  CDF of `{0 < |d|² ≤ R²}` (the exact target of the generator). The continuous uniform-disk CDF r²/R² is not attainable
  by integer offsets: discretisation alone gives 0.045 at R = 10 and 0.011 at R = 30 (reported, not gated). Measured:
  unmatched 0.59–0.74%, lattice KS 0.0025–0.0067 over all 6 layers × R ∈ {10, 30}. The generator's PCG32 stream of
  layer s is `pcg32_srandom(state = (0x9e3779b9 ^ s) << 32 | R << 16 | W_s, seq = STREAM_PAIRING + s)`.
- **C6 Ensemble stats entry points** (§2.10). `passes/restir/ensemble-stats.wgsl` has two entry points with the
  `rs_ensemble_stats` bindings: `rs_ensemble_stats` (16×16 workgroups, dispatch (⌈W/16⌉, ⌈H/16⌉·E): 16² tile sums by a
  shared-memory tree, and per-pixel Σx/Σx² over members into ensPixel) and `rs_ensemble_reduce` (1D: 32²/64² tiles by
  fixed 2×2 trees, global by cascade summation of the 64² tiles, mask sums = 0 since M = 0 in M4). The second is
  compiled by `EnsembleStage.prepare` via `k.compile(..., k.pipelineLayout('rs_ensemble_stats'))`.
- **A19 U-SFX-1 fixture.** `t3_cases` (WP-B's `make-m4.ts`) is not needed for U-SFX-1: it runs on
  `restir-fixtures.ts allLightsScene()` (every endpoint type, V1/V2/mirror, env) twice, once with the production
  `rc.wgsl` and once with `TEST_RC_DR` (a test replacement of `rc.wgsl` whose `rcPairTest` is D ∧ R only, passed via
  `extraSources`), so every case (b), (c) tri/env, (d), (e), deep NEE/TRI/ENV occurs independently of the predicate.
  Other WPs may reuse `TEST_RC_DR` and `restirRig()` (`frames(n, base, perSubmit)`).
- **C7 Chunked replay dispatch (submit budget, §4.4)** (affects WP-E: units). With row bands (`k.rowBand > 0`) the
  replay of a round is emitted as one work unit per band, each `rs_args` + `rs_spatial_replay` over the item chunk
  `[c·chunk, (c+1)·chunk)`, `chunk = ⌈P·NS / bands⌉`; `RsDispatch.treeBase / treeCount` carry the chunk's item base /
  count on these two dispatches (count 0 = whole queue, the single-band default). `rs_args` writes the args of the
  chunk (`queue_chunk_n`), `hdr.n` stays the whole queue; the consumer uses `queue_item_chunk`. The result is bitwise
  independent of the chunking (every slot is written by exactly one item).

### WP-E amendments (Gate 3 harness)

- **E1 `runBatchUnits` options** (§4.4; affects nobody else): `BatchAccumulator.runBatchUnits(src, frames, index?, o?)`
  takes an optional 4th argument `{ maxUnitsPerSubmit (64), rates (Map kind → ms per costHint, carried across batches),
  onSubmit }`. The unit kind is the label up to the first `[`/`(`/`:` (`rs_initial[0+32][0]` → `rs_initial`); **unit
  labels must start with a stable kind name** (WP-A/WP-C units already do). A unit of an unknown kind is estimated at
  the submit target, so it runs alone until measured. At most 64 units share a submit (RsDispatch ring: 512 slots).
- **E2 Chunking probe** (§4.4 "start from costHint and adapt"): before batch 0 `RestirBatchRunner` renders one
  **discarded** frame `t = 0x7FFFFFF0` (scratch accum/counters, outside every run's frame range) with 128-row bands and
  1 tree per unit, learns ms/costHint per kind, then sets `kernel.treeChunk` / `kernel.rowBand` so one rs_initial unit
  is ≤ ½ the 50 ms target; a batch with a submit above the 100 ms budget halves the tree chunk (then the row band).
  Results are chunking-invariant (U-RIS-2). The arena counters are reset after the probe.
- **E3 Ensemble readback** (§2.10): the runner appends a unit `ens_copy[t]` after every frame that copies
  `ensStats` (levels + global; masks unused, M = 0) into a per-batch staging slot, so frames of one submit do not
  overwrite each other's stats; rows are decoded per batch (r = t·E + m, seeds `${runSeed}:${t}:${m}`) and `ensPixel`
  is read and cleared per batch. `ensemble.npz` omits `masks`/`mask_pixels` (M = 0) and `channels` (default R,G,B).
  WP-C's `EnsembleCollector` in `ensemble.ts` reads ensStats per frame instead; both decode the same §2.10 layout.
- **E4 `BASE_JDEN_INVALID / nCand ≤ 1e-5`** is checked against the proxy denominator pixels × frames × S (nCand is a
  per-reservoir field, not a counter); the ratio is written to meta.json `restir.baseJdenInvalidRate`.
- **E5 run-batches `--kernel restir`**: `--spp` (alias `--frames-per-batch`) is the number of frames per batch;
  `--preset`, `--members`, `--plant no-j|marginal-j`, `--w-scale`, `--max-bounces`, `--env-nee`. The Vite server of
  run-batches runs without HMR / file watching (an edit anywhere in the tree reloaded the harness page mid-run) and,
  when `node_modules` is a symlink (git worktree), with a worktree-local `cacheDir` (`.vite-cache-<wt>-harness`).
  `batch-run.ts` exports `loadSource` and `stable` (shared by `restir-batch-run.ts`; no behaviour change).
- **E6 Seeds and caches** (§6.3): PT reference 4001 / re-run 104001, ReSTIR 4002 / re-run 104002, ReSTIR A/A second
  set 5002, rendered plants 4101–4103, pilots PT 4011 / ReSTIR 4012. PT references are cached in
  `validation/out/m4/ptrefs/<pkg>-s<seed>-<spp>x<B>-<key>` with key = sha256 of {package bytes, spp, B, seed, RR off,
  hash of the TS import closure of `batch-run.ts` + the WGSL include closure of `passes/pt.wgsl`}; pilots in
  `validation/out/m4/pilots` (ReSTIR key: TS closure of `restir-batch-run.ts` + every WGSL file).
- **E7 Sizing** (§6.3): pilots PT 128 spp × B and ReSTIR 128 frames (3.1/3.1b) / 8 frames (3.2) × B (B = 16,
  heavy-tail 32); per-sample SDs of every 32² tile and the global mean of Y/R/G/B; the PT total N_R (shared by the
  rungs) and each rung's N_k minimise GPU time subject to u_R(a)/N_R + u_k(a)/N_k ≤ 1 on every aggregate
  (u = per-sample variance / (D·δ/(t_{0.99,B−1} + z_{1−0.005/m}))², D = max(R̄, 0.05·R̄_image)), × 1.25 margin, per-batch
  sizes rounded up to m·2^k (m ∈ 4…7) so cached references are reused. A side above 60 min switches that rung (PT:
  all rungs) to 64² tiles (`tile: 64` + `aggregate_note` in test.json → `aggregate_enlarged` in report.json).
- **E8 Gate-3 T16 extras**: a rung-3.2 / 2022 / ensemble unit fails T16 unless `spatialRoundsExecuted = 3` (a stubbed
  spatial stage cannot pass the final gate); ReSTIR and PT must agree on scene bytes, resolution, env NEE and
  maxBounces; plants must be named and are exempt from the unbiased-preset check.
- **B-5 One visibility term for the rc segment** (affects **WP-A** `path/pathtree.wgsl`, minimal change by WP-B). The base
  path accepts a segment through the closest-hit ray along the SAMPLED direction, every shift re-tests the rc segment
  with `visible(x_{k−1}, x_k)` (offset endpoints, shadow segment between the stored positions). The two disagree on a
  measure ~2e-6 set (a silhouette or triangle within ~1e-5 of the ray; not grazing, not near edges: measured |cos|
  0.2–0.6, segment length 0.6–3), which made T3-0 self shifts and T3-1 inverses OCCLUDED (VIS class). The path tree
  now also evaluates `visible()` for the rc segment of every candidate that owns one — the tree rc pair when it is
  set (`treeVis`, inherited by deep / (c) candidates), the N1 pair (x_{B−1}, x_B) of case (b) and the emitter segment
  of case (d) — and does not stream the candidate when it fails (F := 0). The env rc (e) needs no extra test
  (`visibleInf` traces the identical ray). The ReSTIR integrand thus carries the shift's visibility term on its rc
  segment; it differs from the PT's on a set of measure ~1e-6 (below every Stage-B tolerance).
- **B-6 |cos| guard of the pair predicate** (affects every caller of `rcPairTest`; decisions change only for pairs whose
  segment lies within |cos| < 1e-5 of a vertex plane). The dual found ~9e-4 of the recorded shifts contained a pair
  whose two vertices lie on the SAME flat wall (a shifted y_{k−1} next to the base x_k): exactly coplanar, so the f64
  cos is 0 (guard fails) while the f32 cos is rounding noise (~1e-7) and the GPU passed the pair on a huge footprint.
  Base and shifts agreed (the same f32 code), but the decision was rounding noise. `rcPairTest` now fails F / I when
  `|cos| < RC_COS_MIN = 1e-5` (term GUARD) with the continuous margin `log2(|cos|/1e-5)`; an unbiased domain
  restriction (such pairs fall back to replay / a later pair). The B-4 "tangent" exemption of the dual is removed: a
  dual disagreement is FP-BOUNDARY only by the margin rule or when it flips under an ulp-scale (16 ulp) perturbation
  of the vertex positions; U-11 J disagreements within the f32 conditioning `32·2⁻²³·(1/|cos|min + 3·|p|max/t)` are FP.
- **B-7 Platform fault in the T3 kernel; compact NEE state in the shift** (no interface change; affects nobody's call
  sites). A rare, nondeterministic Metal control-flow fault in the T3 test kernel (docs/decisions/platform-lanes.md
  "Metal quirks" Q2) made a few forced-NEE lanes take the wrong branch after `shift_hybrid`. `shift_hybrid` now keeps
  only the NEE end term and the visibility data of a light sample (not the full `LightSample`) across the rest of the
  shift; the forced visibility is `nee_visible()`'s body on those fields. The T3 harness separates a PLATFORM class
  (sentinels, then/else counts, accounting invariants) from LOGIC; production passes were stress-tested clean.

### WP-D amendments (debug views, inspector, interactive integration)

- **D1 Shift views come from the arena, not from `rsdbg_slot`** (affects nobody's call sites; WP-B/WP-C keep calling
  the hooks). Views 420–445, 438 and 439 are written by the WP-D pass `rs_debug_views` (`passes/restir/debug.wgsl`,
  new file owned by WP-D) after the spatial stage, from the J/code words of the **last executed round**
  (`RsDispatch.round` = executed rounds, as A6) and the flags of the reservoir that round read. The hook could not
  cover slots that only `rs_pair_accept` finalises (NOT_ACCEPTED, EMPTY_SRC, which would render as code 0 = OK), and
  the per-pixel masks would need a cross-thread read-modify-write of the AOV. `replayMask` is derived with the
  `res_needs_replay` rule (accepted ∧ ≠ EMPTY_SRC ∧ source needs replay). `rsdbg_slot` now only records probe tag 68
  (live shift events); `rsdbg_accept` is a no-op kept for signature stability. The pass is encoded only while a
  shift view or the probe is active; a second pass `rs_debug_fill` sets the AOV to `DBG_CODE_NONE` before the ReSTIR
  passes while a ReSTIR code view is active (pixels that no hook writes render black instead of code 0). Both use
  G1 = scene, G2 = {res[(rounds−1) % 2] ro, arena ro, rsVbuf, rsGeo, pairTex}, G3 = debug (9 storage buffers) and are
  encoded through `kernel.encodePass` with the group shape of `rs_spatial_shift`.
- **D2 View 446 `shift.thr`** (appended to the WP-D range 400–499): thr of the pixel (`rsGeo.w`), the reference of the
  margin views 410/440–445 (with B-1 those margins are `log2(footprint/thr)`, so together they are the PLAN §6
  "footprints vs thr" view).
- **D3 Stage taps of the reservoir views.** `rsdbg_reservoir(px, ai, tap)` writes views 400–410 when `dbg.tap == tap`
  or `dbg.tap == DBG_TAP_FINAL`; in the final tap the later stage of the frame overwrites (last writer wins), so
  "final" = after spatial when the spatial stage ran, else after initial. The probe records the reservoir of every
  tap regardless of `dbg.tap`. Bg / empty pixels write `DBG_CODE_NONE` for the code views 405–409 (404 d writes 0 for
  empty, NONE for background).
- **D4 Probe records** (§2.11 tags, formats pinned): 65 header = (bits(passId), bits(round), bits(ai), bits(tap)),
  header + 10 planes (tag 64) are reserved as one block of 11 consecutive records (one atomicAdd, never interleaved);
  66 candidate = (bits(d | tech<<4 | k<<8 | selected<<12), w, lumF, bits(counter)); 67 vertex = (pos.xyz render
  frame, bits(path<<8 | b<<4 | lobeCode)); 68 = (bits(code), J, bits(s), bits(replayed)); **69** = (bits(s), m, w, 0),
  s = 0xFF for the canonical record (m_c, w_c); **70** = (bits(k | sel<<8), m_c, Σm−1, lumRel) — the hook has no
  W_out / c_out (they are in the tap-SPATIAL reservoir record). New WP-D tags from `rs_debug_views` at the probe
  pixel: **71** outgoing slot (bits(code), bits(Jword), bits(s | queued<<8 | partnerValid<<9), bits(partner ai)),
  **72** incoming slot of the partner (bits(code), bits(Jword), bits(s), lum(FJ)), and tag-67 anchors of the shifted
  paths: path **16+s** = the partner's primary hit (p→partner), path **24+s** = the probe's primary hit, the partner's
  surface rc vertex and surface endpoint (partner→p). The inspector joins them with the base path's own vertices
  (the replayed prefix of k > 2 is not recorded). Path ids 1+s / 8+s stay reserved for `rsdbg_vertex` calls from
  the shift (none in M4).
- **D5 Interactive ReSTIR in the app** (`renderer.ts`): `RendererOptions.renderMode 'restir'` +
  `restirMode: 'unbiased' | 'criteria2022' | 'offline' | 'initial'` (interactive preset; + 2022 criteria; offline preset;
  interactive with rounds 0); `RendererContext.debug` passes the app's DebugResources to `RestirKernel.interactive`.
  The renderer clears the arena counter words 16–63 at the start of each interactive frame (per-frame HUD numbers);
  batch kernels are untouched. A light mode other than A falls back to the PT with a HUD note (D1). Offline mode runs
  S = 32 trees in one interactive submit: at 540p this can exceed the 100 ms submit budget (interactive viewing only;
  validation uses the batch runner's chunking).
- **D6 Debug off.** Validation pipelines (no G3, `DEBUG_NO_BINDINGS`) compile every hook to an empty body (tested). With
  debug resources bound, views/probe on vs off give bitwise-identical finalize output; the debug-enabled pipelines
  may differ from the validation pipelines by f32 contraction only (measured: 1 word of 2304 by 7e-8 rel).
- **C8 No timestamp writes on the ReSTIR passes** (affects WP-A: `kernel.ts` `RestirFramePass.encode`, edited by WP-C;
  WP-D: the HUD's `restir` GPU timing line; A13's "timestamps bracket all ReSTIR passes" is withdrawn). The app smoke
  bug "spatial reuse produces nothing" (accepted 0, q0 0/0, view 460 unwritten, image ≈ L1, plain Cornell or
  Cornell + HDRI depending on the run) was not in the spatial stage: in some page loads (typically not the first page of
  a fresh browser) Chrome 154 / Metal silently lost the frame's ReSTIR effects whenever the ReSTIR passes were
  bracketed by the app's timestamp writes (the two empty `rs-ts-begin` / `rs-ts-end` passes, and equally when the
  writes rode on the first / last real pass). No WebGPU validation, internal or OOM error was raised; the frame's GPU
  time halved. Measured in failing pages (same kernel, bind groups, ring slots and pipelines): every frame encoded
  without ReSTIR timestamp writes ran correctly (hand-encoded frames, app frames with the timestamp ring disabled,
  timestamps on the M1 primary pass only), frames with them lost the work. `RestirFramePass.encode` now ignores its
  `timestampWrites` argument (signature kept); 10/10 app page loads and the app smoke pass. Batch / validation paths
  never used timestamps. A GPU test asserts that the interactive pass encodes no compute pass with `timestampWrites`
  and renders identically with and without the argument.
