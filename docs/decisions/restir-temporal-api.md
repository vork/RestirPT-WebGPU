# ReSTIR PT temporal API: implementation contract for M5 (temporal reuse and dynamics)

Status: **normative for M5** (branch `m5-temporal`). Implementer agents build M5 from this document in parallel, as M4
was built from `docs/decisions/restir-api.md`. Anything this document does not pin is the owning work package's
choice, provided it changes no interface below. A change to an interface here is a **contract amendment**: the
coordinator edits this file first (Changelog at the end, append-only numbering), then the owners adapt.

This contract **extends** `restir-api.md`; it does not replace it. Every M4 rule (D1–D22, the whole M4 Changelog A1–A19,
B-1…B-7, C1–C8, D1–D6, E1–E8) stays binding unless a TD item below amends it explicitly, and the amendment then names
the M4 item. The Metal quirks Q1–Q3 of `docs/decisions/platform-lanes.md` apply to every new pass.

Normative math: `docs/math.md` §15 (streams), §17 (fields, suffix cache), §18 (J_P), §21 (visibility), §23 (temporal
contribution MIS with exact E_{t−1}), §24 (light and env change taxonomy), §26 (confidence). Sections marked
**[M5 addition]** in math.md were added with this contract (appendix A lists them).

**Precedence** (PLAN §4.2): PLAN §0–§2 > gap-* (gap-temporal is normative for temporal; gap-env for env dynamics;
gap-rc for the predicate) > *-verify > critique > original reports > this contract's own choices. Every place where this
contract picks between sources, or pins something no source pins, is listed in §0 with its reason.

**Coordination.** Branch `dataformats` (compact quantized vertex arena, package v2) merges into `main` **before** M5 P0
starts; P0 branches from that merge. New M5 code reaches geometry only through the abstract accessors
(`scene_surface`, `vertex_from_ids`, `trace_closest(_ex)`, `trace_any(_ex)`, `visible`, `visibleInf`, `material_eval`,
`bsdf_query`, `bsdf_sample`, `tri_emission`, `emissive_tri`, the light/env functions of §3.1) and never reads
`sceneVerts`/`sceneTris` or a vertex layout. The M2–M4 gates are re-run on the merged `main` before the M5 gate; the M5
gate re-runs only the M4 Gate-0 suites (the rung-3.1/3.2 Stage-B units are covered by the bit-identity test U-M4-BITS).

Contents: §0 decisions · §1 work packages and module map · §2 data contracts · §3 function contracts · §4 pass graph,
bindings, dispatch, submits · §5 RNG and seed contract · §6 validation design (tests, rungs, budget) · §7 risks and
open decisions · appendix A (math.md additions) · appendix B (normative WGSL blocks) · Changelog.

---

## 0. Decisions at a glance

| # | Decision | Why |
|---|---|---|
| TD1 | **Contribution MIS** (math §23) is the temporal estimator in every mode. **Talbot-exact** is a runtime variant (`temporalMis: 'talbot'`) that gates on two units. A **recompute-π_p** variant (`TM_PP_RECOMPUTE`, π_p(Y_p) recomputed by E_{t−1} instead of read from storage) and a **robust** check mode (`TM_ROBUST`, both computed, compared, stored route used) serve T6(b) and the N1-consistent plant. E3 is not implemented. | PLAN rule 9; gap-temporal §3.6, §9.1 ("all exact modes pass"). |
| TD2 | **Two reservoir buffers, in-place temporal output.** `rs_initial` writes the non-history buffer `res[w]`; the temporal stage reads `res[h]` (history = previous frame's final reservoir) **read-only** and writes its output **in place** into `res[w]`; spatial round r reads `res[(w+r)%2]`, writes `res[(w+r+1)%2]`; the final buffer `res[(w+rounds)%2]` becomes the next frame's history. `res[h]` is not written by any pass before the temporal stage's last read (phase B of `rs_t_select`). Temporal off ⇒ `w = 0`, bitwise the M4 schedule. | Invariant I1 (math §23); M4 memory (2 × 160 B) is kept; each temporal pass reads/writes only its own pixel of `res[w]`. |
| TD3 | **E_{t−1} is a set of existing resources plus four additions**: `FrameUniforms.prevCam` (exists), the **ping-ponged ReSTIR G-buffer** (`rsVbuf[2]`, `rsGeo[2]`: jittered prev V-buffer, camera distance, n^g, thr_{t−1}), `LightsParams.prev` + the pmf/alias of the prev slot + the maps stored in the **cur** slot (exist), and the new `RsTemporal` uniform (env record of t−1, change flags, generations). No previous BVH, no previous light tiles, no motion-vector texture. | gap-temporal §4 (proofs §4.2–4.3); critique X3. |
| TD4 | **Exactly one light-state commit per rendered frame.** `LightsState` stages all edits of a frame (light store, animation, env strength/tint/NEE) and commits once in `RestirKernel.advance()`. When nothing changed, the commit leaves the slot and sets `TF_LIGHTS_SAME` (prev accessors return the cur slot, translation is the identity, J_P ≡ 1). | Today env and light edits flip the slot twice per frame (survey: `setEnvironment` calls `update`), so prev ≠ frame t−1. |
| TD5 | **Per-light change bits** live in the unused word 26 of each light record of the **cur** slot (`LCB_MOVED`, `LCB_RADIO`, `LCB_ADDED`); env change bits and the global summary live in `RsTemporal.flags`. MOVED ⇔ the record words that enter the light point/direction Φ differ (point/spot: pos; sun: normal; rect/disk: pos, axisU, axisV, halfU, halfV, normal). RADIO ⇔ any other record word differs. Word 26 is excluded from `lightsChanged`. | math §24 [M5 addition]; the refresh needs rays only for MOVED. |
| TD6 | **Frame-selected evaluators take explicit state**: `_s(…, slot: LightSlot, er: EnvParams)` variants of `nee_eval`, `tri_light_p1`, `p1Env`, `env_bsdf_mis_weight`, `envRadiance`, `env_light_sample_cell`; the existing functions become wrappers with `(lightsParams.cur, envParams)` and must stay **bit-identical** (PT images and M4 ReSTIR frames, U-M4-BITS). ReSTIR code selects the frame with a u32 `fs ∈ {RS_FS_CUR, RS_FS_PREV}` through `lf_slot(fs)` / `lf_env(fs)` (read from uniforms on demand; no frame struct kept live: Metal Q1). | RTXDI `isPrevFrame` pattern (gap-temporal §8.3); one code path for T and T⁻¹. |
| TD7 | **Entry translation covers every alias entry kind**: analytic via the maps of the **cur** slot (`prevToCur` for t−1 → t, `curToPrev` for t → t−1), emissive triangles by the offset `e − nA_from + nA_to` (triangles are static), env to the target slot's `envEntry`. A missing entry, or a realized pmf ≤ 0 in the target frame, makes the shift **undefined** (`SC_O0_LIGHT`) in both directions. | gap-temporal §7; survey: triangle and env entries shift when nA changes. |
| TD8 | **J_p = J_rc · J_P**, with J_P = pmf_to(e_to)/pmf_from(e_from) for NEE-terminated paths only (1 otherwise). J_P is stored separately (tState `TS3.x`, sfxOut `jp`); weights use J_p; the write-back sets `jDen ← J_rc·jDen_src` (J_P excluded). | PLAN rule 3; math §18. |
| TD9 | **`suffix_refresh` is a separate pass pair writing scratch only** (`sfxOut` in the arena): `rs_refresh_fwd` (per pixel over the **history** records, under S_t) before T1, and `rs_refresh_inv` (indirect over **Q_i**, the canonicals whose π_p is needed, under S_{t−1}) before T4. It runs only on frames with `TF_REFRESH`. It produces, per record: the translated entry, J_P, validity, the refreshed deep suffix radiance (classes D-NEE, D-BSDF), the refreshed (b)/(c) cache values (write-back only), and the N1 end visibility. Rays are traced only for samples whose own light MOVED (or env rotated for env NEE). | PLAN §1.8/§3 step 3, rule 10; plan-review R1/WGPU-2 (never in place; pick one scope). |
| TD10 | **One shift for T, T⁻¹ and spatial**: `shift_hybrid` gains `ShiftDst.fs` and `ShiftSrc.endVis`; every light/env term inside uses the `_s` evaluators of frame `dst.fs`; `replay_prefix` gains `fs` (∅-mode end term). Spatial callers pass `fs = CUR`, `endVis = true` (bitwise M4). Classes L/E/B1/(e)/(f)/∅ are re-evaluated inside the shift; only the prefix-independent deep suffix and the N1 end visibility come from the refresh. | PLAN rule 5 (k* in the destination frame's light state; a class change is undefined both ways, open item 9); gap-temporal §5.5. |
| TD11 | **New suffix flag `SFX_DELTA_END = 8`** (word 27 bit 3): the last BSDF event of a BSDF-ended path was a delta lobe (ω2 = 1). Written by the path tree; needed to refresh deep BSDF ends (lobeHist covers only x₁…x₈). | Refresh of ω2 must match the base; d can reach 15. |
| TD12 | **q′ (temporal pixel)** = back-projection of the current jittered primary hit x₁(q) (rebuilt from ids) with `prevCam`, stochastic rounding with ξ from `STREAM_TEMPORAL_PICK`, validity `hit' ∧ dot(n^g, n^g') ≥ 0.5 ∧ |‖x₁ − o_{t−1}‖ − z'| ≤ 0.1·z'` (z' = stored camera distance of q′ at t−1), same member; if the centre tap fails, the 8-ring in a fixed order rotated by the same hash; first valid wins. G-buffer and sample-independent RNG only. | PLAN rule 9; math §23; gap-rc A0 thresholds (open item 11). |
| TD13 | **No q′ ⇒ the canonical record is left bitwise unchanged** (c = 1). A **valid q′ with an empty or non-shiftable temporal record still counts**: c_p = min(20, c_prev), w̃_p = 0, and π_p(X_c) is still required when s = c. | GRIS Eq. 17 (the partition is over producibility sets, independent of the realized X_p). |
| TD14 | c_p = min(`cCap` = 20, c_prev); c_out = 1 + c_p; an empty output (w̃_c + w̃_p = 0) is an empty record with **c = 1 + c_p**. Spatial stays uncapped and reads the post-temporal c. | PLAN rule 13; math §26. |
| TD15 | **Temporal before spatial**, every mode with temporal on: initial → [refresh fwd] → T1 → T2 → T3-A → [refresh inv] → T4 → T3-B → spatial rounds → finalize. | PLAN §3 steps 2–5. |
| TD16 | **Queues**: q1 = Q_f (forward replays), q2 = Q_i (inverse evaluations); items at `[0, P)` and `[P, 2P)` of the M4 item region; capacity P each; `rs_args` takes the queue index from `RsDispatch.flags >> RSD_QUEUE_SHIFT`. With temporal on the arena is allocated with `NS_alloc = max(slots + boostSlots, 2)`. | M4 reserved q1/q2 (§2.6); C1 sticky overflow kept. |
| TD17 | **Arena extension** appended after the M4 item region: `tState` (5 vec4u = 80 B/px) and `sfxOut` (fwd + inv, 2 × 32 B/px). Temporal counters use header words 48–63 (reserved for M5). The M4 layout and every M4 offset are unchanged. | No new storage-buffer binding (≤ 9 of 10). |
| TD18 | **`RsTemporal` uniform at G0 binding 8** (per-frame state, 128 B); static temporal settings in the free words of `RestirParams` (104–127). | Uniforms do not count against the storage-buffer limit. |
| TD19 | **History reset by config hash** (§2.10): any change of the hashed configuration clears `TF_HIST_VALID` for exactly one frame (no temporal candidates). Light edits, camera motion, env rotation/strength/tint are **not** resets. | PLAN rule 9 ("any config change resets history"), rule 10, §3 step 0. |
| TD20 | **Interactive**: a paused frame (`advanced = false`) encodes no ReSTIR pass (history must never be consumed twice); "freeze seed/frame" forces a history reset every frame (identical canonical seeds are not independent candidates). | TD2 destroys `res[h]` during the frame. |
| TD21 | **Reciprocal disocclusion boost**: `boostSlots` extra slots `s = NS … NS+NB−1` (pairing layers NS…) accepted iff `A(p,q) ∧ (dis(p) ∨ dis(q))`, evaluated once by the min thread; dis = no valid q′ (tState flag). Unbiased; default on in interactive, off in validation except one gating unit. | PLAN rule 8, math §22 "disocclusion boost"; PLAN §3 lists it under interactive. |
| TD22 | **Dual MVs are M6** (interactive-only heuristic; no gate needs them). | PLAN §5 M5 "dual MVs (or in M6)". |
| TD23 | **E2 fallback** (`refresh: 'e2'`): on `TF_REFRESH` frames deep-class temporal samples are undefined (w̃_p = 0) and deep canonicals get π_p := 0; L/N1/B1/E/∅ stay exact (the refresh pass still translates, computes J_P and the N1 end visibility). It gates on two units. The P0 stub is stricter: every record is undefined on `TF_REFRESH` frames. Both are unbiased. | gap-temporal §3.5-iii, §5.4 (v). |
| TD24 | **Presets**: `temporal` (rung 3.3: trees 1, rounds 0, temporal on, RR off), `full` (rungs 3.4–3.6: trees 1, rounds 1, slots 3, R 30, temporal on, RR off), `interactive` gains temporal on and boost 3 (RR on, D11). | math §25: RR off in 3.2–3.6. |
| TD25 | **Chains** (validation): chain id = member id (`memberBase + atlas member`), frame t = animation frame counted from the chain's reset (t = 0 has no history), every member renders the same scene state; chains are split across batches and lock chunks only at chain boundaries. | §5; M4 D14 deferred temporal ensembles to M5. |
| TD26 | **A submit never spans a frame boundary** when temporal is on (per-frame uniforms and light slots are written with `queue.writeBuffer`, which would run ahead of an earlier frame in the same submit). | WebGPU queue ordering. |
| TD27 | **New sequence packages** `ixs_*_256` with dense per-frame states (`frames[]` for every frame 0…T−1, light enable flags, env map swaps) generated by `make-m5.ts`; the M3a `ix_*` packages stay untouched (Stage-A regressions). | Survey: M3a packages store only 7 test frames; ix-e models removal as power 0. |
| TD28 | **Static temporal rungs 3.3–3.5 run at 256²** on an 8-scene subset regenerated from the M4 packages; 3.6 runs the 256² sequences. δ is never loosened. | Gate budget (§6.6). |
| TD29 | **Plant signs are checked** (new `plant_sign.py`): a plant passes only if detected (as M4) **and** its predicted sign holds on its predicted region and frames. | PLAN §5 M5 exit; M4 checked detection only (survey). |
| TD30 | **U8 ladder in M5** = C0c/C0d/C0e at 256² (b = 0 and b = 1) through rungs 3.1→3.4 plus the applicable U8 plants 1, 3, 4, 6, 9 and the temporal variants of 2 and 5; plants 7, 8, 10 need Mode B or RIS-NEE tiles and move to M6. | gap-light U8; applicability in Mode A without tiles. |
| TD31 | **N4** (fresh RIS re-draw in the refresh) is a **test-only synthetic** RIS over 8 alias candidates drawn with frame-t random numbers (no light tiles exist before M6), on a dedicated two-light scene. | gap-temporal §5.2, §9.4. |

**M4 items amended by this contract** (all additive; with temporal off every M4 behaviour is bitwise unchanged):
§4.1 / A6 "final reservoir = `res[rounds % 2]`, `rs_initial` writes `res[0]`" → `resBase()` / `finalResIndex()` (TD2);
§3.7 `ShiftSrc`, `ShiftDst`, `ShiftOut` gain `endVis`, `fs`, `F` (TD10, §3.4); §3.6 `replay_prefix` gains `fs`;
§3.3 `nee_eval` becomes a wrapper of `nee_eval_s` (TD6); A10 "capacity of every queue = P·NS" and C1 "`rs_args`
handles q0 only" → per-queue capacity and queue selection (TD16); §2.6 header words 48–63 are assigned (TD17);
B.3 `RestirParams` words 104–127 and a G0 binding 8 (TD18); A2 suffix flags gain `SFX_DELTA_END` (TD11); §4.2 the
ReSTIR G-buffer textures become ping-ponged (§2.6); D5 `endpointId` renumbering is implemented by the temporal
write-back (§3.6).

---

## 1. Work packages, module map, ownership

### 1.1 Phases

1. **P0 foundation (owner T-A, first merge, ≤ 1 day).** Lands exactly the files and edits marked **P0** in §1.3: the
   normative content of appendix B (constants, `RsTemporal`, `RestirParams` words, arena layout, tState/sfxOut
   accessors, TS mirror), the G0 binding 8, the prev G-buffer textures and their parity, the reservoir role rotation
   (TD2) with bitwise-M4 behaviour when temporal is off, the pass table entries and G2 layouts of every new pass, and
   **compiling stubs** for every new WGSL module and TS stage (§1.4). Nothing else. All other WPs branch from the P0 merge.
2. **T-A … T-E in parallel.** Each WP merges into `m5-temporal` independently after P0; merge order is free. A WP
   replaces the stub bodies of the files it owns; it edits a file it does not own only at the "may touch" lines of §1.2.
3. **Integration (T-E leads).** Rung-3.3/3.4 pilots start when T-A (A1) and T-B (B1) are merged; dynamic light units
   need T-C; env units need T-A's env record and T-C.

### 1.2 Work packages

**T-A: CPU prev-state plumbing, E_{t−1} snapshot, frame-selected evaluators, q′, config hash, history reset.**
- Owns: all P0 files; `restir/tframe.wgsl`, `restir/tpick.wgsl`; the `_s` evaluator edits in `lights/measure.wgsl`,
  `lights/env-sample.wgsl`, `lights/env.wgsl`, `restir/endpoint.wgsl`; TS `render/restir/frame-state.ts`, the M5 parts
  of `render/restir/{kernel,resources,layout,presets}.ts`; `render/lights-gpu.ts`; `src/core/scene/scene-package.ts`
  (frame format of TD27); tests `tests/restir/{tframe,light-maps,config-hash,frame-state}.test.ts`,
  `validation/gpu-tests/restir-tframe.gpu.test.ts`.
- May touch (each change listed in its PR): `render/env-gpu.ts` (env map generation id, packed-word copy for
  `RsTemporal.envPrev`), `render/pt-kernel.ts` (`applyEnvLighting` stages instead of committing), `scene/light-store.ts`
  (expose the staged change set), `render/restir/stage-spatial.ts` and `render/restir/debug.ts` (**P0 only**: the
  input index `(k.resBase + r) % 2` / `k.finalResIndex()`; nothing else), `validation/harness/batch-run.ts`
  (`loadSource` applies the new frame fields; no behaviour change for M3 packages).
- Early deliverable **A1** (≤ 3 days after P0): real `_s` evaluators with U-M4-BITS green; `RestirKernel.advance()` with
  one commit per frame, change bits, env prev record, config hash; prev G-buffer swap; real `temporal_pixel`. T-B's
  forward/inverse shifts and T-C's refresh depend on A1.
- Tests: U-TL-1…4, U-TV-1, U-TH-1, U-M4-BITS, T6(c) part (i) provenance (§6.1).

**T-B: temporal passes T1–T4, contribution MIS, Talbot and recompute variants, J_P application, confidence cap,
temporal write-back, shift frame selector, temporal queues.**
- Owns: `restir/tshift.wgsl`; `passes/restir/{t-classify,t-forward,t-select,t-inverse}.wgsl`; TS
  `render/restir/stage-temporal.ts`; the M5 edits of `restir/shift.wgsl`, `path/replay.wgsl`, `restir/queue.wgsl`,
  `passes/restir/args.wgsl` (M4 WP-B/WP-C files, owned by T-B in M5); tests
  `validation/gpu-tests/restir-temporal.gpu.test.ts` (+ `restir-temporal-fixtures.ts`),
  `tests/restir/{tmis-ref.ts,tmis.test.ts,tqueue.test.ts}`, the temporal extension of `tests/restir/rc-dual.ts`
  (T6(c) cross-evaluator; T-B may touch `rc-dual.ts` for it).
- May touch: `path/pathtree.wgsl` only via T-C (below); nothing else.
- Stubs consumed: `temporal_pixel` (P0: no q′ ⇒ temporal off, unbiased), refresh stub (P0: all records undefined on
  `TF_REFRESH` frames, unbiased), `_s` evaluators (P0: cur state).
- Early deliverable **B1** (≤ 3 days after A1): T1–T4 on camera-only motion (no light change), contribution MIS,
  identity test §9.3-5, U-TQ-1. **B2**: Talbot, recompute/robust modes, T3-2 round trips with light changes (needs C1).
- Tests: T6(a), T6(b), T3-2, T4-t, §9.3-5, §9.3-6 (amended), U-TQ-1, U-TR-2, U-TE-1 (with T-E) (§6.1).

**T-C: `suffix_refresh` and the light/env change taxonomy.**
- Owns: `restir/refresh.wgsl`; `passes/restir/refresh.wgsl`; TS `render/restir/refresh.ts` (unit builders called by
  `stage-temporal.ts`); tests `validation/gpu-tests/restir-refresh.gpu.test.ts`, `tests/restir/refresh-ref.test.ts`.
- May touch: `path/pathtree.wgsl` (write `SFX_DELTA_END`; two lines), `validation/gpu-tests/restir-initial.gpu.test.ts`
  (append U-SFX-2). It never edits `restir/shift.wgsl` (T-B owns the shift; T-C provides `endVis` and the refreshed
  radiance through `SfxRec`).
- Stubs provided (P0, T-A lands them): `refresh_record` returns `SXS_DONE | SXS_UNDEF` on every record.
- Early deliverable **C1** (≤ 4 days after A1): exact refresh for analytic lights (moved, radiometric, add/remove) and
  E2; **C2**: env (rotation, strength, tint), plants N3, N4, N7, env plants.
- Tests: §9.3-2 (light-local Jacobian: FD + PSS z-test), §9.3-3 (idempotence), §9.3-4 (refresh vs full re-trace),
  T-ENV-temporal (with T-B), U-SFX-2 (§6.1).

**T-D: M5 debug views, inspector, HUD, interactive integration, reciprocal disocclusion boost.**
- Owns: the M5 edits of `debug/restir-views.wgsl`, `passes/restir/debug.wgsl`, `render/restir/debug.ts`,
  `src/app/ui/panels/{restir-panel,restir-inspector}.ts`; the boost edits of `passes/restir/pair-accept.wgsl`,
  `restir/pairing.wgsl`, `restir/mis.wgsl` (also U8 plant 9), `render/restir/stage-spatial.ts` (M4 WP-C/WP-D files,
  owned by T-D in M5); `validation/harness/m5-app-smoke.ts`; `validation/gpu-tests/restir-debug.gpu.test.ts` (M5 cases).
- May touch: `render/renderer.ts` (interactive temporal: `advance()` per advanced frame, pause rule TD20, restirMode
  list), `src/app/integration.ts` (remove the history reset on light and env-parameter edits; reset on config change),
  `src/app/app.ts` (freeze-seed ⇒ reset, "reset history" and "freeze history" controls), `render/overlay.ts`
  (temporal probe polylines), `validation/gpu-tests/restir-spatial.gpu.test.ts` (append T3-3-boost).
- Stubs provided (P0): the hook functions of §2.11 (empty bodies).
- Tests: U-TD-1…3, T3-3-boost, M5 app smoke (§6.1).

**T-E: Gate 3 rungs 3.3–3.6, chain runner, sequence generators, masks, plants and sign checks, `validate --milestone M5`.**
- Owns: `render/restir/chain-runner.ts`; `validation/harness/{restir-chain-run.ts,gate-m5.ts}`;
  `validation/scenes/make-m5.ts` (+ generated `ixs_*`, `m5s_*`, `u8_*` packages); `validation/tools/{dynamic.py,
  dyn_masks.py,plant_sign.py}` (+ their tests under `validation/tools/tests/`); `tests/restir/gate-m5-config.test.ts`;
  the M5 section of `docs/decisions/validation.md`; M5 rows of `validation/budget.json`.
- May touch: `render/restir/ensemble.ts` (mask regions M > 0, per-test-frame readback), `render/restir/batch-runner.ts`
  (reuse its packer; no behaviour change for M4 runs), `validation/harness/{harness.ts,run-batches.ts,validate.ts,
  restir-batch-run.ts}` (M5 entry points and flags, as M4 E5), `validation/tools/{compare.py,stats.py}` (additions for
  masks from files, per-frame units and drift; reviewed by the coordinator).
- Stubs consumed: until T-B lands, chains run with the stub (temporal output = canonical); the plumbing is tested.

Dependencies: P0 → everyone; A1 → B, C, E (chains need `advance()`); B1 → E (rungs 3.3/3.4 static, ix-d/ix-d0), D (views
with data); C1 → B2 (light-change round trips), E (light sequences); C2 → E (env sequences, env plants).

### 1.3 Module map (one owner per file; M4 files list the M5 owner)

```
src/core/shaders/
  restir/types.wgsl            P0/A   + M5 constants (appendix B.1), append-only
  restir/reservoir.wgsl        P0/A   + res_write_empty_c(i, seed, c), SFX_DELTA_END accessor
  restir/frame.wgsl            P0/A   + prev G-buffer texture declarations, RestirParams M5 words
  restir/tframe.wgsl           P0/A   NEW: RsTemporal (G0 binding 8), RS_FS_*, lf_slot, lf_env, lf_cam_pos, lt_translate,
                                      lt_pmf, lt_change_bits, lt_moved, tState / sfxOut accessors (B.3)
  restir/tpick.wgsl            P0 stub/A   NEW: temporal_pixel (q′, §3.3)
  restir/endpoint.wgsl         A      + nee_eval_s, nee_endpoint_id_s; nee_eval = wrapper
  lights/measure.wgsl          A      + tri_light_p1_s, analytic_area_p1_s; wrappers bit-identical
  lights/env-sample.wgsl       A      + p1Env_s, env_bsdf_mis_weight_s, env_light_sample_cell_s; wrappers
  lights/env.wgsl              A      + envRadiance_s(uv, er); envRadiance = wrapper
  restir/shift.wgsl            B      + ShiftDst.fs, ShiftSrc.endVis, _s evaluators, U8 plants 1/3/4/6/2t
  path/replay.wgsl             B      + fs parameter (∅-mode end term)
  restir/queue.wgsl            B      + per-queue capacity and item base (q1/q2)
  restir/tshift.wgsl           P0 stub/B   NEW: temporal loaders, temporal_shift wrapper, tmis_* weights, temporal write-back
  restir/refresh.wgsl          P0 stub/C   NEW: SfxRec, refresh_record (end terms under frame fs), plants N3/N4/N7/env
  restir/mis.wgsl              D      + boost slots in the slot loop (none needed: numSlots covers them), U8 plant 9
  restir/pairing.wgsl          D      + boost acceptance helper
  path/pathtree.wgsl           (M4 A) T-C may touch: SFX_DELTA_END
  passes/restir/args.wgsl      B      queue index from RsDispatch.flags
  passes/restir/refresh.wgsl   P0 stub/C   NEW: rs_refresh_fwd (per pixel), rs_refresh_inv (indirect, Q_i)
  passes/restir/t-classify.wgsl P0 stub/B  NEW: rs_t_classify (T1)
  passes/restir/t-forward.wgsl  P0 stub/B  NEW: rs_t_forward (T2, indirect Q_f)
  passes/restir/t-select.wgsl   P0 stub/B  NEW: rs_t_select (T3, phases A and B)
  passes/restir/t-inverse.wgsl  P0 stub/B  NEW: rs_t_inverse (T4, indirect Q_i)
  passes/restir/pair-accept.wgsl D     + boost acceptance
  passes/restir/debug.wgsl     D      + temporal views from tState
  debug/restir-views.wgsl      P0/D   + temporal hooks (empty in P0), views 480–499, probe tags 73–78
src/core/render/
  lights-gpu.ts                A      staged edits, one commit per frame, change bits (word 26), TF summary
  restir/layout.ts             P0/A   TS mirror of appendix B (M5 words, arena offsets, tState/sfxOut decoders)
  restir/resources.ts          P0/A   rsVbuf[2]/rsGeo[2], parity, res roles, arena size, G0 binding 8, new G2 layouts
  restir/kernel.ts             P0/A   advance(), resBase / finalResIndex(), temporal stage hook, frame schedule (§4.1)
  restir/presets.ts            P0/A   temporal settings, presets `temporal`/`full`, interactive + temporal + boost
  restir/frame-state.ts        P0 stub/A   NEW: RestirFrameState, change diff, config hash, RsTemporal packing
  restir/stage-temporal.ts     P0 stub/B   NEW: RestirStage emitting the temporal units
  restir/refresh.ts            P0 stub/C   NEW: refresh unit builders
  restir/stage-spatial.ts      D      boost slots (P0/A: input index expression)
  restir/debug.ts              D      M5 views/tags/HUD (P0/A: final index)
  restir/chain-runner.ts       E      NEW: chains, test frames, per-frame ensemble readback
  restir/ensemble.ts           (C)    T-E may touch: masks, per-test-frame readback
  restir/batch-runner.ts       (E)    T-E may touch
src/core/scene/scene-package.ts  A    PackageFrame: dense frames, lights[id].enabled, env.map, sequence block
src/app/ui/panels/restir-panel.ts, restir-inspector.ts   D
validation/harness/restir-chain-run.ts, gate-m5.ts       E
validation/harness/m5-app-smoke.ts                       D
validation/scenes/make-m5.ts                             E
validation/tools/dynamic.py, dyn_masks.py, plant_sign.py E
validation/gpu-tests/restir-tframe.gpu.test.ts           A
validation/gpu-tests/restir-temporal.gpu.test.ts (+fixtures)  B
validation/gpu-tests/restir-refresh.gpu.test.ts          C
validation/gpu-tests/restir-debug.gpu.test.ts            D   (M5 cases appended)
tests/restir/tframe, light-maps, config-hash, frame-state .test.ts   A
tests/restir/tmis-ref.ts, tmis.test.ts, tqueue.test.ts               B
tests/restir/refresh-ref.test.ts                                     C
tests/restir/gate-m5-config.test.ts                                  E
```

### 1.4 Stubs in P0 (exact behaviour)

| Module | Stub body | Consequence while stubbed |
|---|---|---|
| `tpick.wgsl` | `temporal_pixel` returns invalid (flags `TS_DISOCC`) | no temporal candidate; T3 leaves the canonical unchanged: output = rung-3.1/3.2 output, unbiased |
| `tshift.wgsl` | `temporal_shift` returns FAILED `SC_O0_MISS`; tmis functions real (they are small, §3.6) | every temporal candidate FAILED: w̃_p = 0, π_p = 0 ⇒ W_Y = W_c up to rounding, c = 1 + c_p (unbiased); only reachable once A1's real `temporal_pixel` lands |
| `refresh.wgsl` | `refresh_record` returns status `SXS_DONE | SXS_UNDEF` | on `TF_REFRESH` frames every temporal sample and every canonical inverse is undefined (class zeroing of all classes, unbiased) |
| `_s` evaluators | bodies call the cur functions (ignore slot and env) | prev == cur; only camera-only units are meaningful until A1 |
| `stage-temporal.ts` | `frameUnits` returns `[]` | temporal off; `w = 0`; bitwise M4 |
| `refresh.ts` | unit builders return `[]` | – |
| `frame-state.ts` | `advance()` commits lights once, writes `RsTemporal` with `TF_HIST_VALID = 0` | every frame is a reset frame |
| temporal debug hooks | empty | no views |
| pass entry points | compile with their bindings, bodies call the stubs | – |

### 1.5 Rules of engagement

- One owner per file (§1.3). A WP that needs a change elsewhere asks the coordinator; interface changes are amendments.
- New constants are appended to `restir/types.wgsl` / `layout.ts` (never renumbered); adding one is an amendment.
- Every WGSL module starts with the code base's header comment (what it implements, `math.md#anchors`, owner).
- Binding defines are `'<n>u'` strings built with `restirDefines` (M4 A4).
- No `timestampWrites` on or around any ReSTIR pass, including the temporal and refresh passes (Q3, C8).
- GPU tests run on the Chrome lane (authoritative); each new GPU test file contains a known-answer canary (Q1) and any
  test kernel that loops over shifts carries the PLATFORM sentinels of B-7 (Q2).
- Nothing in M5 changes a rung-3.1/3.2 result: with `temporal: false` every M4 pipeline output is bit-identical to the
  pre-M5 build (U-M4-BITS).

---

## 2. Data contracts

Per-pixel buffers are indexed by the **atlas pixel index** `ai` (M4 §2). Words are u32; floats are `bitcast<u32>(f32)`.

### 2.1 New constants (normative in appendix B.1, mirrored in `layout.ts`)

- Frame selectors `RS_FS_CUR = 0`, `RS_FS_PREV = 1`.
- `RsTemporal.flags` `TF_*`: `HIST_VALID 1, LIGHTS_SAME 2, ENV_SAME 4, REFRESH 8, PMF_CHANGED 16, ENV_MOVED 32,
  ENV_RADIO 64, LIGHT_MOVED 128, RESET 256, CAM_SAME 512`.
- Temporal mode word `RestirParams.tMode` `TM_*`: `TALBOT 1, PP_RECOMPUTE 2, ROBUST 4, E2 8`.
- Temporal plant word `RestirParams.tPlants` `TP_*`: `N1_MIXED 1, NO_JP 2, NO_JP_ENV 4, N3_STALE 8, N4_RIS 16,
  N5_PIXEL_CENTRE 32, N6_CUR_CAM 64, N7_PER_LIGHT 128, ENV_NO_ROT_VIS 256, ENV_GAMMA_T 512, CP_PLUS1 1024,
  U8_STALE_AUX 2048, U8_SPOT_PREV_AXIS 4096`. (N1-consistent = `TP_N1_MIXED` + `TM_PP_RECOMPUTE`.)
- `RestirParams.flags` new bits: `RSF_TEMPORAL 64`, U8 plants `RSF_PLANT_U8_W1DELTA 256, RSF_PLANT_U8_NO_PK 512,
  RSF_PLANT_U8_T2 1024, RSF_PLANT_U8_ONESIDED 2048, RSF_PLANT_U8_FAILED_K 4096`.
- `RsDispatch.flags`: `RSD_QUEUE_SHIFT = 8` (bits 8–9 = queue index for `rs_args`), `RSD_PHASE_B = 32` (T3 phase).
- Light-record change bits (word 26 of the cur slot's records) `LCB_MOVED 1, LCB_RADIO 2, LCB_ADDED 4`.
- tState flags `TS_*` and sfxOut status `SXS_*`: appendix B.3.
- Pass ids: `RS_PASS_TEMPORAL 8` (select RNG, exists), `RS_PASS_T_REFRESH_FWD 9, RS_PASS_T_CLASSIFY 10,
  RS_PASS_T_FWD 11, RS_PASS_T_REFRESH_INV 12, RS_PASS_T_INV 13, RS_PASS_T_PLANT 14` (all < 16 = spatial).
- `STREAM_TEMPORAL_PICK = 0x2c1b3c6du`.
- `SFX_DELTA_END = 8u` (suffix flags word 27).
- Counters (arena header words 48–63) `RSC_T_*`: `QVALID 48, DISOCC 49, FWD_QUEUED 50, FWD_OK 51, SEL_P 52,
  INV_QUEUED 53, INV_OK 54, EMPTY_OUT 55, LIGHT_UNDEF 56, CLASS_UNDEF 57, REFRESH_RECS 58, REFRESH_RAYS 59, E2_ZEROED 60,
  ROBUST_MISMATCH 61, NONFINITE 62, PENDING_LEFT 63`.
- Views 480–499 and probe tags 73–78 (§2.11).

### 2.2 Reservoir buffers, roles, invariant I1

- Two physical buffers `resA`, `resB` (M4). The kernel keeps `h` = index of the history buffer (−1 after a reset or
  with temporal off) and derives `w = (h < 0) ? 0 : 1 − h` each frame (TD2).
- Frame t with temporal on:
  - `rs_initial` writes `res[w]` (canonical X_c, c = 1).
  - `rs_refresh_fwd`, T1, T2 read `res[h]` (bound read-only as `resIn`).
  - T3 binds `res[h]` as `resIn` (ro) and `res[w]` as `resOut` (rw); T4 and `rs_refresh_inv` bind `res[w]`
    (T4 rw as `resOut`; refresh-inv ro as `resIn`).
  - **Invariant I1** (math §23): `res[h]` holds F_{t−1}(X) exactly in its own domain and is never written during
    frame t before T3 phase B has finished. Spatial round 0 then writes `res[1 − w] = res[h]`.
  - The final buffer is `res[(w + executedRounds) % 2]`; `advance()` of frame t+1 sets `h` to it.
- Pixel-local write rule: every temporal pass writes only records of its own pixel `q` in `res[w]`; T4 reads X_c from
  `res[w][q]` only for pixels with s = c, whose record T3 phase A did not modify; T3 phase A writes `res[w][q]` only for
  s = p (and empty) outputs.
- `resBase()` / `finalResIndex()` (kernel) replace the M4 constants `0` and `rounds % 2` in `stage-spatial.ts`,
  `debug.ts` and finalize; with temporal off they return `0` and `rounds % 2` (bitwise M4).
- Memory is unchanged (2 × 160 B/px).

### 2.3 The exact E_{t−1} record

| # | State of frame t−1 | Where it lives at frame t | Written by | Notes |
|---|---|---|---|---|
| 1 | Camera (position, basis, yfov, aspect) | `FrameUniforms.prevCam` | `advance()` (validation), app (interactive) | = cam of t−1; = cam after a reset |
| 2 | Jittered primary hit per pixel (primId, bary f32 bits), camera distance z | `rsVbuf[1 − g]` (rgba32uint) | `rs_primary` of t−1 | g = G-buffer parity of frame t |
| 3 | n^g toward the t−1 camera, thr_{t−1} | `rsGeo[1 − g]` (rgba32float) | `rs_primary` of t−1 | thr stored bits (gap-rc §6.3) |
| 4 | Light records, alias table, realized pmf | `LightsParams.prev` slot in `records` | `LightsState.commit()` | = cur slot when `TF_LIGHTS_SAME` |
| 5 | Id maps t−1 ↔ t | `curToPrev` / `prevToCur` of the **cur** slot | `commit()` | identity is implied by `TF_LIGHTS_SAME` |
| 6 | Env record (cg, sg, strength, visibleToCamera, tint, flags) | `RsTemporal.envPrev` (verbatim packed `EnvParams` words of t−1) | `advance()` | env pmf[ENV] is in the prev slot's pmf |
| 7 | Env map, importance tables | shared (G0 texEnv, `records` env tables) | – | a change is a config change (reset) |
| 8 | Frame config (maxBounces, M(B), τ, α_min, criteria, BSDF tier, light mode, env map id, importance key, env NEE, atlas, cCap, tMode, plants) | config hash (§2.10) | `advance()` | any change ⇒ reset, so t−1 and t share it |
| 9 | Reservoir X_p (F, W, c, fields, suffix cache) | `res[h]` | final stage of t−1 | read-only (I1) |
| 10 | Generations (camGen, vbufGen, lightGen, pmfGen) | `RsTemporal.gensPrev` | `advance()` | T6(c) provenance |

Not needed (gap-temporal §4.4): previous BVH, previous materials/textures, previous light tiles, motion vectors,
previous RR probabilities, previous jitter offsets, previous LOD plane (LOD 0 in exact modes; M7 revisits).

### 2.4 Light state per frame

- `LightsState` gains `stage(lights, env?)` (records the pending state) and `commit(): LightsCommit` (called exactly
  once per rendered frame by `advance()`). `commit()` writes the new state into the non-current slot and flips only if
  something changed; otherwise it sets `same = true` and the kernel sets `TF_LIGHTS_SAME` (`LightsParams.prev` may then
  equal `cur`; the maps are not read).
- The pmf/alias table is rebuilt only when the weight vector changes (M3a rule): `pmfChanged` ⇔ weights differ.
- `LightsCommit { same, pmfChanged, anyMoved, anyRadio, added: number[], removed: number[], reallocated }`.
  `reallocated` (capacity growth or env table change) is a config-hash input (reset).
- Change bits: word 26 of every record of the cur slot (TD5); word 27 stays 0. A light whose stable id is new has
  `LCB_ADDED`. Removed lights are visible only through `prevToCur = LIGHT_NONE`.
- The PT kernel's `LightsGpu` keeps its M3 behaviour; the renderer stages lights into both kernels and each commits once
  per encoded frame.

### 2.5 Env record cur/prev

- `envParams` (G0 binding 1) is frame t; `RsTemporal.envPrev` holds the **same packed words** of frame t−1 (copy of the
  previous frame's `packEnvParams` output, never recomputed), so `lf_env(PREV)` under `TF_ENV_SAME` is bitwise
  `envParams`.
- `TF_ENV_MOVED` ⇔ cg or sg words differ; `TF_ENV_RADIO` ⇔ strength or tint words differ (then pmf[ENV] and every other
  entry's pmf change: `pmfChanged`). visibleToCamera changes only length-1 terms (no flag).
- Map swap, importance-resolution change, env NEE toggle, env added/removed with a table change: config hash (reset).

### 2.6 G-buffer double buffering

- `rsVbuf[0..1]`, `rsGeo[0..1]` (M4 formats). Parity `g` flips in `advance()` when the frame advances; `rs_primary`
  writes `[g]`; passes sample cur `[g]` as `rsVbuf`/`rsGeo` (M4 bindings) and prev `[1 − g]` as `rsVbufPrev`/`rsGeoPrev`
  (new sampled textures). `rsL1`, `rsShade`, `rsFrame` stay single.
- Bind groups are cached per `(pass, resIn index, parity)`; M4 pass code is unchanged (it sees "cur").

### 2.7 Uniforms

- `RestirParams` (128 B, unchanged size): offset 104 `boostSlots` (was pad1), 108 `tMode`, 112 `cCap` (f32, 20),
  116 `tPlants`, 120–127 reserved (appendix B.2). `numSlots` = slots + boostSlots when the boost is on.
- `RsTemporal` (G0 binding 8, 128 B, per frame; appendix B.2): `flags`, `histFrames`, `frameGen`, `prevGen`,
  `envPrev: EnvParams`, `gens`, `gensPrev`, `configHash`.
- `RsDispatch`: unchanged layout; temporal units use `passId`, `round` (T3 phase), `treeBase/treeCount` (item chunk of
  indirect passes, M4 C7), `flags` (`RSD_QUEUE_SHIFT`, `RSD_PHASE_B`), `rowBase/rowEnd`.

### 2.8 Arena extension, queues, temporal slot formats

```
M4 (unchanged):  hdr 64 words | slots 4·P·NS | codes P·NS | items P·NS            NS = NS_alloc (TD16)
M5 appended at word TB = 64 + 6·P·NS:
  tState   TB + 20·ai + {0..19}             5 vec4u per pixel (80 B)
  sfxOut   TB + 20·P + 16·ai + {0..7}       fwd record of the HISTORY record at ai (32 B)
           TB + 20·P + 16·ai + {8..15}      inv record of the CURRENT record at ai (32 B)
total words = 64 + 6·P·NS + 36·P            (layout.ts arenaWords(P, NS, temporal))
Q_f = q1: items at word 64 + 5·P·NS + i,       i < P, capacity P
Q_i = q2: items at word 64 + 5·P·NS + P + i,   i < P, capacity P
item word = ai << 3                          (slot bits 0)
```

**tState** (per destination pixel q; writers in parentheses; every word is written each frame by T1 for every pixel,
including background, before anything reads it):

| Word | Name | Content |
|---|---|---|
| TS0.xyz | `fwdF` | F_t(Y_p) RGB (T1 inline or T2); 0 unless OK |
| TS0.w | `fwdJ` | J word of the forward shift: `JW_FAILED`, `JW_PENDING` (queued, transient) or f32 bits of **J_p = J_rc·J_P** |
| TS1.xyz | `invF` | F_{t−1}(T⁻¹ y) RGB (T4), y = X_c (or Y_p in recompute/robust mode) |
| TS1.w | `invJ` | J word of the inverse: f32 bits of J_inv = J_rc⁻·J_P⁻¹, or FAILED |
| TS2.x | `qPrime` | atlas index of q′, `0xFFFFFFFF` if none |
| TS2.y | `cP` | bits(c_p) (0 if no q′) |
| TS2.z | `fwdCode` | slot-code word (M4 §2.6 format) of the forward shift |
| TS2.w | `flags` | `TS_*` |
| TS3.x | `jP` | bits(J_P) of the forward shift (1 for BSDF ends) |
| TS3.y | `wc` | bits(w̃_c) (T3 phase A) |
| TS3.z | `wp` | bits(w̃_p) (T3 phase A) |
| TS3.w | `invCode` | slot-code word of the inverse |
| TS4.x | `piStored` | bits(π_p(Y_p) stored route = lum(F_p^st)/J_p) (T3 phase A, s = p) |
| TS4.y | `piRecomp` | bits(π_p recomputed by T4) |
| TS4.z | `xpEntry` | endpoint entry of X_p in t−1 numbering (robust ID check), `RC_NONE` if not NEE |
| TS4.w | `cPrev` | bits(uncapped c_prev) |

**sfxOut record** (`SfxRec`, per record and direction; appendix B.3):

| Word | Name | Content |
|---|---|---|
| SX0.xyz | `rad` | deep: β_s ⊙ (end term under frame fs) (the new `rcRad`); (b): Λ_fs at x_k; (c): L_e / L_env,fs; else 0 |
| SX0.w | `aux` | (b)/(c): p1_fs at x_{d−1}; else 0 |
| SX1.x | `status` | `SXS_*` |
| SX1.y | `entryTo` | translated alias entry (NEE ends), `RC_NONE` for non-NEE ends |
| SX1.z | `jp` | bits(J_P) (1 for BSDF ends) |
| SX1.w | `gen` | `RsTemporal.frameGen` of the writing frame (stale-record check) |

A record whose sfxOut `gen` differs from the current `frameGen` is treated as not refreshed; on a `TF_REFRESH` frame a
missing refresh is a bug (`RSC_T_PENDING_LEFT`).

**Queue discipline.** At the start of the temporal stage: `clearBuffer(arena, 16, 8)` (q1 counter, n) and
`clearBuffer(arena, 32, 8)` (q2); overflow flags are sticky (C1). `rs_args` with `RSD_QUEUE_SHIFT` field q writes
`hdr[q].n = min(counter, P)`, capacity P, the sticky overflow, and args at byte `16·q` (for the item chunk of M4 C7).

### 2.9 Suffix cache usage for the refresh

- Classes (math §17, derivable): L (k = d, NEE), N1 (k = d−1, NEE), B1 (k = d−1, BSDF), E (k = d, BSDF), D-NEE /
  D-BSDF (k ≤ d−2), R (k = ∅).
- The refresh reads, for deep classes: `sfxA–C` (x_{d−1} ids), `sfxDir` (ω_o toward x_{d−2} for NEE ends; ω_{d−1} for
  BSDF ends), `betaS`, `sfxP2` (BSDF ends), `sfxFlags` (`SFX_ESCAPE`, `SFX_DELTA_END`), the endpoint triple
  `endA–C` (NEE: entry, u, v / env i,j,h2; BSDF_TRI: triangle ids); for N1/B1: the rc triple (x_k = x_{d−1}) and
  `rcWi` (B1 end direction).
- x_{d−1} is rebuilt with `scene_surface(prim, u, v, −sfxDir)` (orientation toward the incoming side; equal to the base's
  `vertex_from_ids` flip decision), V = sfxDir.
- `SFX_DELTA_END` (TD11): set by the path tree on BSDF endings whose final event was a delta lobe.
- The refresh never reads the suffix cache of ∅ or k = d records (it is invalid there, M4 §2.4).

### 2.10 Config hash and history reset

`frame-state.ts configHash(state)` = FNV-1a over: scene generation id (package hash or app scene version), atlas size,
member size, E, memberBase, light mode, `LightsCommit.reallocated`, env map id (generation on `setEnvironment`),
importance key, env NEE, BSDF tier / material version, maxBounces, criteria, τ, α_min, M(B), cCap, tMode, tPlants,
RSF plant bits, wScale, jitter mode. `advance()` clears `TF_HIST_VALID` (and sets `TF_RESET`) iff the hash differs from
the previous frame's, or `state.reset`, or the kernel was (re)allocated, or it is the first frame. With `TF_HIST_VALID`
clear, T1 sets every q′ invalid (`TS_NO_HIST`) and the frame is canonical + spatial.

Interactive triggers (T-D wires them): scene load, resize, light mode change, env map/NEE/importance change, ReSTIR
settings change, freeze seed/frame (TD20), the "reset history" control. **Not** resets: light edits, animation, camera
motion (including teleports and FOV changes, handled by q′), env rotation/strength/tint, visibleToCamera.

### 2.11 Debug views, probe tags, hooks (T-D)

View ids (registered by `render/restir/debug.ts`; data from tState / sfxOut through `rs_debug_views`, or hooks):

| Id | Key | Kind | Content |
|---|---|---|---|
| 480 | `t.qvalid` | code | 0 bg, 1 centre tap, 2 ring tap, 3 none (disoccluded), 4 no history |
| 481 | `t.motion` | vec3 | (s′ − q) in pixels (x, y, 0) |
| 482 | `t.refreshFwd` | code | sfxOut.fwd status class of X_p: 0 none, 1 analytic, 2 ray traced, 3 undefined, 4 E2, 5 zero |
| 483 | `t.refreshInv` | code | same for X_c (Q_i pixels) |
| 484 | `t.fwdCode` | code | SC of the forward shift |
| 485 | `t.invCode` | code | SC of the inverse |
| 486 | `t.logJ` | scalar (signed) | log2 J_p |
| 487 | `t.logJP` | scalar (signed) | log2 J_P |
| 488 | `t.pic` | scalar (log) | π_c |
| 489 | `t.pip` | scalar (log) | π_p (stored for s = p, recomputed for s = c) |
| 490 | `t.cprev` | scalar | c_prev (uncapped) |
| 491 | `t.cout` | scalar | c_out |
| 492 | `t.sel` | code | 0 canonical kept (no q′), 1 canonical selected, 2 temporal selected, 3 empty |
| 493 | `t.phatRel` | scalar (signed) | (lum F_t(Y_p) − lum F_p^st)/max(both) (forward vs stored p̂) |
| 494 | `t.robust` | scalar (signed) | (π_p recomputed − π_p stored)/max (robust mode) |
| 495 | `t.wp` | scalar | w̃_p/(w̃_c + w̃_p) |
| 496 | `t.lightsChanged` | code | per pixel: bit0 X_p's light moved, bit1 radiometric, bit2 undefined (removed/added) |
| 497 | `s.boost` | code | bitmask of accepted boost slots |

Reservoir views 400–410 get the tap `DBG_TAP_TEMPORAL` (written by T3 phase A for finalized pixels and phase B for the
rest; pixels with no q′ are written by phase A). Hooks (P0 empty bodies):
```wgsl
fn rsdbg_temporal(px: vec2u, ai: u32, phase: u32);            // after T3 phase A/B finalised the pixel (reads resOut + tState)
fn rsdbg_tpick(px: vec2u, sp: vec2f, tap: u32, valid: bool);  // T1, probe only
fn rsdbg_refresh(ai: u32, dir: u32, rec: SfxRec);             // refresh passes, probe only
```
Probe tags: 73 temporal header `(bits(q ai), bits(q′ ai), c_prev, bits(flags))`, 74 forward `(bits(code), J_p, J_P,
lum F)`, 75 inverse `(bits(code), J_inv, lum F_prev, π_p)`, 76 select `(w̃_c, w̃_p, π_c, bits(sel | phase<<8))`, 77
refresh `(bits(status), lum rad, aux, bits(entryTo))`, 78 pick `(s′.x, s′.y, bits(tap), bits(valid))`. The inspector
draws the forward shifted path (anchors 16+s reuse, s = 7) and the inverse path into q′ (camera at `prevCam`).
HUD (per frame, from header words 48–63 and queue headers q1/q2): Q_f/Q_i occupancy and overflow, P(s = p), temporal
replay fraction `FWD_QUEUED/QVALID`, refresh records and rays, disocclusion fraction, `histFrames`, TF flags, class-change
count.

### 2.12 Memory (validation layout)

| Item | B/px | 1024² atlas (E = 16 × 256², NS = 3) | 540p interactive (NS 3 + boost 3) |
|---|---|---|---|
| resA + resB | 320 | 336 MB | 166 MB |
| arena M4 part (24·NS) | 72 / 144 | 75 MB | 75 MB |
| tState + sfxOut | 144 | 151 MB | 75 MB |
| rsVbuf ×2, rsGeo ×2, rsL1, rsShade, rsFrame | 112 | 117 MB | 58 MB |
| ens stats / pixel | ≤ 32 per member px | ≤ 3 MB | – |
| **Total** | ≈ 650 | **≈ 0.68 GB** | **≈ 0.37 GB** |

All allocations go through `pushErrorScope('out-of-memory')`; E = 64 stays behind the M0 allocation probe.

---

## 3. Function contracts

Signatures are normative; bodies follow the pseudo-code. "Same function" means the literal WGSL function called from
every listed place; a second implementation of a formula is a contract violation.

### 3.1 Frame-selected light and env evaluators (T-A; math §24)

```wgsl
// lights/env.wgsl, lights/env-sample.wgsl, lights/measure.wgsl (wrappers keep the M3 names and bits)
fn envRadiance_s(uv: vec2f, er: EnvParams) -> vec3f;                     // envRadianceScaled(uv, er.strength * er.tint)
fn p1Env_s(d: vec3f, slot: LightSlot, er: EnvParams) -> f32;             // light_pmf(slot, slot.envEntry)·envPdfSA(d, er.cg, er.sg)
fn env_bsdf_mis_weight_s(dir: vec3f, p2: f32, B: u32, afterDelta: bool, slot: LightSlot, er: EnvParams) -> f32;
fn env_light_sample_cell_s(slot: LightSlot, er: EnvParams, entry: u32, i: u32, j: u32, h2: u32) -> LightSample;
fn tri_light_p1_s(x: vec3f, z: vec3f, ngz: vec3f, primId: u32, slot: LightSlot) -> f32;
fn analytic_area_p1_s(x: vec3f, entry: u32, z: vec3f, slot: LightSlot) -> f32;
// restir/endpoint.wgsl
fn nee_eval_s(x: vec3f, ep: NeeEndpoint, slot: LightSlot, er: EnvParams) -> LightSample;
fn nee_endpoint_id_s(ep: NeeEndpoint, slot: LightSlot) -> u32;
// restir/tframe.wgsl (ReSTIR only; needs G0 binding 8)
fn lf_slot(fs: u32) -> LightSlot;       // CUR: lightsParams.cur; PREV: TF_LIGHTS_SAME ? cur : prev; TP_N1_MIXED: always cur
fn lf_env(fs: u32) -> EnvParams;        // CUR: envParams; PREV: rsTemporal.envPrev (TP_N1_MIXED: envParams;
                                        //   TP_ENV_GAMMA_T: envPrev with cg/sg of envParams)
fn lf_cam_pos(fs: u32) -> vec3f;        // frame.cam / frame.prevCam camToWorld[3].xyz
```
- `envRadiance(uv) ≡ envRadiance_s(uv, envParams)` etc., bitwise; the M3/M4 callers keep their names. U-M4-BITS proves
  PT and ReSTIR outputs unchanged.
- The functions never read `lightsParams.cur` / `envParams` internally once they take explicit state.
- `light_sample_entry(x, slot, entry, u)` already takes the slot (analytic records and pmf of that frame; emissive
  triangles are static).

### 3.2 Entry translation and J_P (T-A; math §18, §24)

```wgsl
fn lt_translate(entry: u32, from: u32, to: u32) -> u32;   // LIGHT_NONE if absent in `to`
fn lt_pmf(entry: u32, fs: u32) -> f32;                    // light_pmf(lf_slot(fs), entry)
fn lt_change_bits(entryCur: u32) -> u32;                  // analytic: TF_LIGHTS_SAME ? 0 : word 26 of the cur record
                                                          //   (the word is relative to the slot's own predecessor); triangle: 0;
                                                          //   env: (ENV_MOVED ? LCB_MOVED : 0) | (ENV_RADIO ? LCB_RADIO : 0)
fn lt_moved(entryCur: u32) -> bool;                       // lt_change_bits & LCB_MOVED
```
`lt_translate` (normative):
```
if from == to or TF_LIGHTS_SAME or TP_N1_MIXED:   return entry
sf = lf_slot(from); st = lf_slot(to); cur = lightsParams.cur
if entry == sf.envEntry:                    return st.envEntry            (LIGHT_NONE when the env entry is absent)
if entry <  sf.nAnalytic:                   return records[(from == PREV ? cur.prevToCurOff : cur.curToPrevOff) + entry]
tri = entry − sf.nAnalytic;                 return st.nAnalytic + tri     (tri < triCount, static)
```
J_P of an NEE-terminated record translated `e_from → e_to`: `jp = lt_pmf(e_to, to) / lt_pmf(e_from, from)`; the record
is **undefined** (`SC_O0_LIGHT`) iff `e_to == LIGHT_NONE` or `lt_pmf(e_to, to) ≤ 0` (the same predicate in T and T⁻¹).
Plants: `TP_NO_JP` ⇒ jp := 1; `TP_NO_JP_ENV` ⇒ jp := 1 for env endpoints only.

### 3.3 Temporal pixel q′ (T-A, `restir/tpick.wgsl`; math §23 [M5 addition])

```wgsl
struct TPick { valid: bool, ai: u32, local: vec2u, sp: vec2f, tap: u32 }   // tap 0 centre, 1…8 ring, 9 none
fn temporal_pixel(p: RsPix, key: vec2u) -> TPick;
```
Normative algorithm (member-local coordinates; a G-buffer and sample-independent RNG function only):
```
if !TF_HIST_VALID or rsVbuf[p] is a miss:     invalid (tap 9)
x1 = vertex_from_ids(vbuf ids of p, rs_cam_pos()); ng = rsGeo[p].xyz
sp = frame_project(x1.pos, frame.prevCam)      (continuous member-local pixel position; behind camera ⇒ invalid)
h  = pcg4d(runSeed ^ member·φ, t, localIdx, STREAM_TEMPORAL_PICK)
c0 = floor(sp + (u01(h.x), u01(h.y)))
for tap in 0 … 8:  c = (tap == 0) ? c0 : c0 + RING[(tap − 1 + (h.z & 7)) & 7]      RING = (1,0),(1,1),(0,1),(−1,1),(−1,0),(−1,−1),(0,−1),(1,−1)
   if c inside the member tile and V(c) : return (c, tap)
V(c) = rsVbufPrev[c] is a hit ∧ dot(ng, rsGeoPrev[c].xyz) ≥ 0.5 ∧ |‖x1.pos − lf_cam_pos(PREV)‖ − z′| ≤ 0.1·z′,  z′ = bits(rsVbufPrev[c].w)
```
Background pixels are never temporal sources or destinations (PLAN §1.7).

### 3.4 Temporal shift (T-B; `restir/shift.wgsl` edits and `restir/tshift.wgsl`)

`restir/shift.wgsl` edits (T-B, normative):
```wgsl
struct ShiftSrc { ...M4 fields..., endVis: bool }       // N1 end visibility under the destination frame (true spatially)
struct ShiftDst { ...M4 fields..., fs: u32 }            // light/env frame of the destination domain (CUR spatially)
```
- Every light/env term in `shift_hybrid` uses `lf_slot(dst.fs)` / `lf_env(dst.fs)` with the `_s` evaluators:
  (a)/(f)/(b) `nee_eval_s`; (d) and (c)-triangle `tri_light_p1_s`; (e) and (c)-env `env_bsdf_mis_weight_s`,
  `envRadiance_s(envUV(ω, er.cg, er.sg), er)`; `replay_prefix(…, dst.fs)` for the ∅ end term.
- Case (b): `shift_finish_vis` uses `vis = visible(y_{k−1}, x_k) ∧ src.endVis` (a false endVis is a defined zero,
  `SC_OCCLUDED`, classified after every undefined test).
- Nothing else in the shift changes: O0–O3, rcPairTest with `dst.thr`, D3/B-2 directions, M(B), support indicator.
  With `fs = CUR` and `endVis = true` the shift is bitwise M4 (U-M4-BITS).

`restir/tshift.wgsl` (T-B):
```wgsl
struct TSrc { base: ShiftSrc, undefinedLight: bool, jp: f32, xpEntry: u32 }
fn tsrc_load(ai: u32, fromBuf: u32 /* 0 resIn, 1 resOut */, sfxDir: u32 /* SFX_FWD | SFX_INV */, fsTo: u32) -> TSrc;
fn tdst_cur(q: RsPix) -> ShiftDst;                       // M4 shift_dst_load + fs = CUR
fn tdst_prev(qPrime: u32) -> ShiftDst;                   // rsVbufPrev / rsGeoPrev at q′, camPos = lf_cam_pos(PREV), fs = PREV
                                                         //   TP_N5: pixel-centre prev primary re-traced; TP_N6: camPos = cur
struct TShiftOut { F: vec3f, J: f32, jP: f32, code: u32 } // F = F_dst(ȳ) (not F·J); J = J_rc·J_P; code M4 slot-code format
fn temporal_shift(s: TSrc, dst: ShiftDst) -> TShiftOut;
```
`tsrc_load` (normative): load the record plane-wise (Metal Q1); if `TF_REFRESH`, read the sfxOut record of the given
direction: `SXS_UNDEF` ⇒ `undefinedLight`; NEE ends get `entryTo` written into `end.x` (and `rc.x` when k = d);
`jp = sfx.jp`; deep records take `rcRad ← sfx.rad` (unless `TP_N3_STALE`); N1 takes `endVis = (status & SXS_VIS) != 0`.
Without `TF_REFRESH`: entries unchanged, jp = 1, endVis = true, stored rcRad.
`temporal_shift`: `undefinedLight` ⇒ `SC_O0_LIGHT`; else `o = shift_hybrid(s.base, dst)`; for SC_OK: `F = o.F`,
`J = o.J · jp`, `jP = jp`. `ShiftOut` gains the field `F` (the destination integrand before the J multiply, set for
SC_OK; spatial callers ignore it), because recovering F as `FJ / J` would divide by a possibly tiny J. J_P multiplies
NEE-terminated records only (jp = 1 otherwise by construction).

### 3.5 Suffix refresh (T-C; `restir/refresh.wgsl`; math §24 [M5 addition])

```wgsl
struct SfxRec { rad: vec3f, aux: f32, status: u32, entryTo: u32, jp: f32, gen: u32 }
fn refresh_record(ai: u32, fsFrom: u32, fsTo: u32) -> SfxRec;   // record at resIn[ai] (own frame fsFrom) evaluated under fsTo
fn sfx_store(dir: u32, ai: u32, r: SfxRec);  fn sfx_load(dir: u32, ai: u32) -> SfxRec;   // (tframe.wgsl accessors, P0)
```
Normative per record X (empty/background ⇒ status 0):
```
class = class(d, k, tech);  status = SXS_DONE
if tech == NEE:  eTo = lt_translate(entry, fsFrom, fsTo);  if eTo == NONE or lt_pmf(eTo, fsTo) ≤ 0: status |= SXS_UNDEF
                 jp = lt_pmf(eTo, fsTo)/lt_pmf(entry, fsFrom)                      (TP_NO_JP / TP_NO_JP_ENV ⇒ 1)
else:            eTo = RC_NONE, jp = 1
moved = tech == NEE and lt_moved(entryCur)       (entryCur = the frame-t index of the endpoint: eTo when fsTo = CUR, entry when fsFrom = CUR)
slot = lf_slot(fsTo); er = lf_env(fsTo)
D-NEE (k ≤ d−2, NEE):   x = scene_surface(sfx ids, −sfxDir); m = material_eval(x, sfxDir); ls = nee_eval_s(x.pos, ep(eTo), slot, er)
                         qn = bsdf_query(m, sfxDir, ls.dir, LOBE_NEE); w1 = nee_mis_w1(ls, qn.p_marg, d−1)
                         vis = moved ? nee_visible(x, prim, ls) : true            (ENV_NO_ROT_VIS plant: vis = true)
                         rad = vis && ls.valid && ls.prim != prim ? betaS * (w1/ls.q) * qn.f_all * ls.Lambda : 0     (pathtree grouping)
                         status |= SXS_DEEP | (moved ? SXS_RAY : 0) | (rad == 0 ? SXS_ZERO : 0)
D-BSDF (k ≤ d−2, BSDF): escape: L = envRadiance_s(envUV(sfxDir, er.cg, er.sg), er); w2 = DELTA_END ? 1 : env_bsdf_mis_weight_s(sfxDir, sfxP2, d−1, false, slot, er)
                         triangle: z = vertex_from_ids(end ids, x.pos); L = tri_emission(end ids); w2 = DELTA_END ? 1 : mis_w2(tri_light_p1_s(x.pos, z.pos, z.ng, prim_z, slot), sfxP2, d−1)
                         rad = betaS * (w2 * L); status |= SXS_DEEP
N1 (k = d−1, NEE):       x = scene_surface(sfx ids, −sfxDir) (= x_k); ls = nee_eval_s(x.pos, ep(eTo), slot, er)
                         rad = ls.Lambda; aux = ls.p1; vis = moved ? nee_visible(x, prim, ls) : true
                         status |= SXS_N1 | (vis ? SXS_VIS : 0) | (moved ? SXS_RAY : 0)
B1 (k = d−1, BSDF):      rad = L_e (triangle, static) or envRadiance_s(envUV(rcWi, er.cg, er.sg), er); aux = p1 of the end under slot/er
                         status |= SXS_B1
L, E, R:                 rad = 0 (the shift re-evaluates everything)
E2 (TM_E2) on a deep record:  status |= SXS_UNDEF | SXS_E2 (no evaluation)
N7 plant (TP_N7_PER_LIGHT): records whose own light has neither MOVED nor RADIO keep stored rcRad/aux (rad := stored), jp still applied
N4 plant (TP_N4_RIS), D-NEE:  endpoint re-drawn by RIS over 8 alias candidates at x with u = rs_rand(key(ai), RS_PASS_T_PLANT, j);
                         rad = betaS * N(new endpoint) * W_RIS (test-only; not a shift)
gen = rsTemporal.frameGen
```
Note for N1: the suffix cache holds x_{d−1} = x_k; `sfxDir` points toward the base's x_{k−1} (prefix-dependent), so
only the orientation of the rebuilt surface depends on it, and Λ, p1 and the visibility (M4 `visible` offsets toward the
segment's side) do not. The prefix-dependent factors of N1 (f at x_k with the offset's V, ω1 with the offset's p2) are
evaluated in the shift.
- **Visibility rule** (N1 and D-NEE): a ray is traced iff the endpoint's light MOVED between t−1 and t (analytic
  `LCB_MOVED`, env `TF_ENV_MOVED`); otherwise V = 1, which is exact because every record the refresh serves has
  p̂ > 0 in its own frame (it was selected), geometry is static, and an unmoved light keeps Φ. (math §24 [M5 addition].)
- **Idempotence**: with `fsFrom = fsTo` or an unchanged light, `rad`/`aux` equal the stored `rcRad`/`aux` to ≤ 1e-6
  relative (same functions and grouping as the path tree; §9.3-3).
- `rs_refresh_fwd`: per pixel ai over the atlas, record `res[h][ai]`, `fsFrom = PREV`, `fsTo = CUR`, store `SFX_FWD`.
  `rs_refresh_inv`: per Q_i item q, record `res[w][q]` (X_c, or Y_p in recompute/robust mode), `fsFrom = CUR`,
  `fsTo = PREV`, store `SFX_INV` (N1-mixed plant: not run; the loader then uses stored values).

### 3.6 Temporal passes T1–T4 (T-B; math §23)

Shorthand: `key = rs_frame_key(member, t, localIdx)` of the destination pixel q; `c_c = 1`; `cap = rsParams.cCap`.

**T1 `rs_t_classify`** (8×8 per pixel over the atlas, every pixel writes its tState):
```
tState[q] = cleared (qPrime = NONE, flags = 0, codes = SC_NOT_ACCEPTED-coded, J words FAILED)
if background: flags = TS_BG; rsdbg; return
pk = temporal_pixel(q, key);  if !pk.valid: flags = TS_DISOCC | (TF_HIST_VALID ? 0 : TS_NO_HIST); count; return
qP = pk.ai;  cPrev = rp_c(res[h][qP]);  cP = min(cap, cPrev)
tState: qPrime = qP, cP, cPrev, flags = TS_QVALID (| TS_PICK_RING)
src = tsrc_load(qP, resIn, SFX_FWD, CUR)
if src.base.empty:            fwdCode = SC_EMPTY_SRC, fwdJ = FAILED
elif src.undefinedLight:      fwdCode = SC_O0_LIGHT,  fwdJ = FAILED
elif res_needs_replay(flags): fwdJ = PENDING; queue_append(1, q<<3); flags |= TS_FWD_QUEUED
else:                         o = temporal_shift(src, tdst_cur(q)); store fwdF, fwdJ, jP, fwdCode; flags |= TS_FWD_DONE
```
**T2 `rs_t_forward`** (indirect over Q_f, 64/wg, RS_REPLAY = 1): for item q: reload `qP`, `src`; `o = temporal_shift`;
store; no PENDING may survive (`RSC_T_PENDING_LEFT`).

**T3 `rs_t_select`, phase A** (`RsDispatch.round = 0`; 8×8 per pixel; resIn = res[h] ro, resOut = res[w] rw):
```
if !(flags & TS_QVALID): leave res[w][q] untouched (TD13); rsdbg_temporal(A); return
wc = c_c·lum(F_c)·W_c;   wp = (fwdJ is a value) ? cP·lum(fwdF)·W_p·J_p : 0          (W_p = rp_W(res[h][qP]))
store wc, wp
if TM_TALBOT:  queue_append(2, q<<3), flags |= TS_INV_QUEUED; return               (selection in phase B)
if wc + wp == 0:  res_write_empty_c(q, seed_c, 1 + cP); flags |= TS_EMPTY_OUT | TS_FINAL; return
wSum = 0; sel = c
ris_update(&wSum, wc, rs_rand(key, RS_PASS_TEMPORAL, 0))
if ris_update(&wSum, wp, rs_rand(key, RS_PASS_TEMPORAL, 1)): sel = p
if sel == p:
   πc = lum(fwdF);  πp = lum(F_p^st)/J_p          (F_p^st = rp_F(res[h][qP]); TP_CP_PLUS1: denominator uses cP + 1)
   piStored = πp
   if TM_PP_RECOMPUTE:  res_select_temporal(q, qP, 0, 1 + cP) (W written in phase B); queue_append(2, q<<3);
                        flags |= TS_SEL_P | TS_INV_QUEUED; return
   W_Y = πp / (c_c·πc + cP·πp) · (wc + wp) / πc
   res_select_temporal(q, qP, W_Y, 1 + cP);  flags |= TS_SEL_P | TS_FINAL
   if TM_ROBUST:  queue_append(2, q<<3); flags |= TS_INV_QUEUED | TS_ROBUST          (check only; the stored route is used)
else:
   queue_append(2, q<<3); flags |= TS_SEL_C | TS_INV_QUEUED                         (finalised in phase B after T4)
```
`res_select_temporal(q, qP, W, c)` (write-back, math §18 [M5 addition]): copy all ten planes of `res[h][qP]` to
`res[w][q]`, then: `F ← fwdF`; `jDen ← (J_p / J_P)·jDen_src`; NEE ends: entry words (`end.x`, and `rc.x` when k = d)
← `RC_TAG_NEE | entryTo`, `endpointId ← nee_endpoint_id_s(entryTo, cur slot)`; deep: `rcRad ← sfx.fwd.rad` (when
`TF_REFRESH`); (b)/(c): `rcRad ← sfx.fwd.rad`, `aux ← sfx.fwd.aux` (when `TF_REFRESH`); `W ← W·rsParams.wScale`
(plant, 1 = off); `c`. Seeds, d, k, technique, rc ids, lobes, rcWi, suffix cache, lobeHist stay (R13 of M4 applies to
∅ records). A non-finite W_Y ⇒ empty record with c, `RSC_T_NONFINITE`.

**T4 `rs_t_inverse`** (indirect over Q_i, RS_REPLAY = 1; resOut = res[w] rw, never written by T4):
```
q = item; qP = tState.qPrime
src = tsrc_load(q, resOut, SFX_INV, PREV)     (the record at res[w][q]: X_c, or Y_p in recompute/robust mode)
o = temporal_shift(src, tdst_prev(qP))         (J_P⁻¹ via sfx.inv.jp; entry translated cur → prev)
πp = OK ? lum(o.F)·o.J : 0                     (TM_E2 and a deep record on a TF_REFRESH frame ⇒ 0, counted E2_ZEROED)
store invF, invJ, invCode, piRecomp = πp
```
**T3 `rs_t_select`, phase B** (`round = 1`, `RSD_PHASE_B`; per pixel; only pixels with `TS_INV_QUEUED`):
```
contribution, s = c:   πc = lum(F_c); W_Y = πc/(c_c·πc + cP·πp)·(wc + wp)/πc;  write W_Y·wScale and c = 1 + cP into res[w][q] (P0.x, P1.w only)
recompute (s = p):     W_Y = πp_recomp/(c_c·πc + cP·πp_recomp)·(wc + wp)/πc  with πc = lum(fwdF); write W, c
robust (s = p):        use piStored (as contribution); compare with piRecomp: |a − b| > 1e-3·max(a, b), or invCode ≠ OK,
                       or the translated-back entry ≠ xpEntry ⇒ RSC_T_ROBUST_MISMATCH and view 494
Talbot:                πp(X_c) = piRecomp; πp(Y_p) = lum(F_p^st)/J_p;
                       m_c = c_c·lum F_c/(c_c·lum F_c + cP·πp(X_c));  m_p = cP·πp(Y_p)/(c_c·lum fwdF + cP·πp(Y_p))
                       w_c = m_c·lum F_c·W_c; w_p = m_p·lum(fwdF)·W_p·J_p; select with the phase-A RNG counters 0/1;
                       W_Y = (w_c + w_p)/lum F(Y); s = p ⇒ res_select_temporal; s = c ⇒ write W, c; Σw = 0 ⇒ empty with c
rsdbg_temporal(B)
```
MIS helper functions (tshift.wgsl, real in P0, CPU twin `tests/restir/tmis-ref.ts`):
```wgsl
fn tmis_contrib_W(piSel: f32, piC: f32, piP: f32, cC: f32, cP: f32, wSum: f32) -> f32;   // π_s/(c_c π_c + c_p π_p)·wSum/π_c; 0 if a denominator is 0
fn tmis_talbot_mc(cC: f32, phatC: f32, cP: f32, piP: f32) -> f32;                        // := 1 if the denominator is 0
fn tmis_talbot_mp(cC: f32, phatY: f32, cP: f32, piP: f32) -> f32;                        // := 0 if the denominator is 0
```

### 3.7 Composition with spatial reuse; boost

- Order TD15. The spatial stage reads `res[w]` after T3 phase B: the post-temporal records with their c (M4 §3.8
  unchanged). c_out of spatial is uncapped; the next frame caps it through c_p.
- ReSTIR-unbiased = presets `temporal`/`full`: exact E_{t−1}, contribution MIS, cCap 20, refresh exact, no boost, RR
  off, no plants. ReSTIR-interactive = interactive preset: the same exact temporal core, RR on (D11), boost 3; M5.5 adds
  the denoiser, M6 the duplication map (biased) and dual MVs.
- **Boost** (TD21, T-D): `rs_pair_accept` loops `s < numSlots`; for `s ≥ slots`: `A_boost = A(p,q) ∧ (dis(p) ∨ dis(q))`
  with `dis = (tState.flags & TS_DISOCC) != 0`, evaluated once by the min thread; pairing layer = s (sizes of
  `PAIR_TEX_SIZES`, s ≤ 5). Resample, MIS and write-back treat boost slots as ordinary slots (`k = |S_c|`).

### 3.8 TS API (P0 signatures; T-A / T-B / T-E)

```ts
// presets.ts
export interface RestirSettings { /* M4 fields */ temporal: boolean; cCap: number; temporalMis: 'contribution' | 'talbot';
  temporalCheck?: 'none' | 'recompute' | 'robust'; refresh: 'exact' | 'e2'; boostSlots: number;
  tPlant?: { n1Mixed?: boolean; noJP?: boolean; noJPEnv?: boolean; n3Stale?: boolean; n4Ris?: boolean; n5PixelCentre?: boolean;
             n6CurCam?: boolean; n7PerLight?: boolean; envNoRotVis?: boolean; envGammaT?: boolean; cpPlus1?: boolean; u8StaleAux?: boolean; u8SpotPrevAxis?: boolean };
  plant?: { /* M4 */ noJ?: boolean; marginalJ?: boolean; wScale?: number; u8W1Delta?: boolean; u8NoPk?: boolean; u8T2?: boolean; u8OneSided?: boolean; u8FailedK?: boolean } }
export const RESTIR_PRESETS: Record<'initial' | 'initial-rr' | 'offline' | 'interactive' | 'criteria2022' | 'temporal' | 'full', Partial<RestirSettings>>;
// frame-state.ts
export interface RestirFrameState { t: number; camera: CameraState; lights: LightData[]; env?: { params: EnvParamsCpu; mapId: string }; reset?: boolean }
export interface RestirAdvance { histValid: boolean; flags: number; reasons: string[]; commit: LightsCommit; configHash: number }
// kernel.ts
class RestirKernel {
  advance(state: RestirFrameState): RestirAdvance;    // validation/chains: one call per frame, before frameUnits(state.t, out)
  advanceInteractive(frameUniforms: GPUBuffer, o: { lights?: LightData[]; env?: …; reset: boolean }): RestirAdvance;
  resBase(): number; finalResIndex(): number;
  readonly temporal: RestirStage;                     // stage-temporal.ts
  readTemporalState(): Promise<Uint32Array>;          // tState + sfxOut (tests, inspector)
}
// chain-runner.ts (T-E)
export interface ChainSpec { frames: number; testFrames: number[]; state(t: number): RestirFrameState; masks?(t: number): Uint8Array }
export class ChainRunner { run(k: RestirKernel, spec: ChainSpec, o: { members: number; chains: number; chainBase: number; runSeed: number;
  onTestFrame(t: number, stats: EnsembleFrameStats, pixel: Float64Array): void }): Promise<ChainTotals> }
```

---

## 4. Pass graph, bindings, dispatch, submits

### 4.1 Frame schedule

```
frame t (temporal on; E members in the atlas):
  advance(state_t)                    CPU: one light commit, env record, prevCam, parity g, roles h/w, RsTemporal, hash
  rs_primary                          8×8, row bands                              → rsVbuf[g], rsGeo[g], rsL1
  rs_initial × tree chunks            8×8, row bands                              → res[w] (canonical)
  -- temporal stage (stage-temporal.ts), only if temporal on --------------------------------------------------------
  clearBuffer(arena, 16, 8); clearBuffer(arena, 32, 8)                            q1, q2 counters
  rs_refresh_fwd        (TF_REFRESH && TF_HIST_VALID)   8×8 over the atlas        res[h] → sfxOut.fwd
  rs_t_classify         8×8                             T1: q′, c_p, inline shifts → tState, Q_f
  rs_args(q1) + rs_t_forward    indirect (item chunks)  T2                        → tState
  rs_t_select (phase A) 8×8                             T3-A: select / finalise s = p / queue s = c → res[w], Q_i
  rs_args(q2) + rs_refresh_inv  (TF_REFRESH) indirect   Q_i                       res[w] → sfxOut.inv
  rs_args(q2) + rs_t_inverse    indirect                T4                        → tState
  rs_t_select (phase B) 8×8                             T3-B: finalise Q_i pixels  → res[w]
  -- spatial stage (M4, input res[w]) ----------------------------------------------------------------------------------
  rounds × (pair_accept → args(q0) → spatial_replay → spatial_shift → spatial_resample)
  rs_finalize                         reads res[finalResIndex()]
  rs_ensemble_stats (ensemble)        + ens copy units at test frames (T-E)
```
Rung 3.3 (`temporal`): rounds 0 ⇒ finalize reads `res[w]` (F·W, M4 A6). With `TF_HIST_VALID` clear the temporal units
still run T1 and T3-A (flags, taps; the canonical is untouched) and skip the rest.

### 4.2 Per-pass bindings

G0 (shared, explicit): 0 FrameUniforms · 1–3 env · 4 RestirParams · 5 LightsParams · 6 `records` (ro) · 7 RsDispatch ·
**8 RsTemporal (new)**. G1 scene (`SceneGpu.layoutEntries()`, **≤ 5 storage buffers**, asserted by U-BIND-1 on the
merged dataformats layout) or empty. G3 debug (1 SB) or empty. SB = 1 + G1 + G2 + G3.

| Pass | G1 | G2 bindings | SB | Sampled |
|---|---|---|---|---|
| rs_primary | scene | 0 rsVbufOut[g] · 1 rsGeoOut[g] · 2 rsL1Out · 3 arena rw | 8 | – |
| rs_initial | scene | M4 (resOut = res[w]) | 9 | 2 |
| rs_refresh_fwd | scene | 0 resIn = res[h] ro · 1 arena rw · 2 rsVbuf · 3 rsGeo | 1+5+2+1 = **9** | 2 |
| rs_refresh_inv | scene | 0 resIn = res[w] ro · 1 arena rw · 2 rsVbuf · 3 rsGeo | **9** | 2 |
| rs_t_classify | scene | 0 resIn = res[h] ro · 1 arena rw · 2 rsVbuf · 3 rsGeo · 4 rsVbufPrev · 5 rsGeoPrev | **9** | 4 |
| rs_t_forward | scene | same as classify | **9** | 4 |
| rs_t_select | empty | 0 resIn = res[h] ro · 1 resOut = res[w] rw · 2 arena rw · 3 rsVbuf | 1+0+3+1 = **5** | 1 |
| rs_t_inverse | scene | 0 resOut = res[w] rw · 1 arena rw · 2 rsVbuf · 3 rsGeo · 4 rsVbufPrev · 5 rsGeoPrev | **9** | 4 |
| rs_args | empty | M4 (arena rw, rsArgs rw) | 4 | – |
| rs_pair_accept | empty | M4 + reads tState (arena rw already) | 4 | 3 |
| spatial / finalize / ensemble | – | M4 (cur textures of parity g) | M4 | M4 |

- No pass exceeds 9 of 10 storage buffers. `rsArgs` is never bound as storage in the pass that consumes it indirectly.
- Arena write access: every temporal pass binds the arena `read_write` (counters, tState, sfxOut, queues).
- Bind groups: prebuilt per `(pass, h/w, g)` (at most 4 variants each), cached; the kernel rebuilds on reallocation or
  `LightsGpu.version`.
- Defines: M4 set plus `RS_TEMPORAL = 1` on temporal/refresh passes, `RS_RES_IN_BINDING`/`RS_RES_OUT_BINDING` per
  pass as above, `RS_VBUF_PREV_BINDING`, `RS_GEO_PREV_BINDING`, `RS_REPLAY = 1` for T2 and T4, `RS_REPLAY = 0` for T1
  (the refresh passes do not include the shift). Test-only: `RS_TEMPORAL_TRACE` (T3-2 harness), `RS_PROVENANCE` (T6(c)).

### 4.3 Dispatch shapes

Per-pixel passes `@workgroup_size(8, 8, 1)`, 2D direct over the atlas row band (M4 §4.3). Queue passes
`@workgroup_size(64)`, 2D indirect (M4 §2.7), item chunks as C7 (`treeBase/treeCount` = item base/count). Temporal
passes contain **no loop over shifts** (one shift per thread), which keeps them outside the Q2 shape.

### 4.4 Submits and work units

- Unit labels start with a stable kind name: `rs_refresh_fwd[r0]`, `rs_t_classify[r0]`, `rs_t_forward[ci]`,
  `rs_t_select_a[r0]`, `rs_refresh_inv[ci]`, `rs_t_inverse[ci]`, `rs_t_select_b[r0]` (E1).
- `costHint`: per-pixel passes `atlasW·rows`; forward/inverse `items·(maxBounces + 1)`; refresh `atlasW·rows·2`.
- **TD26**: with temporal on, the batch/chain runner closes the submit at every frame boundary (units of one frame may
  span several submits; a submit never holds units of two frames). Target 50 ms, budget 100 ms, hard cap 200 ms (M4).
- Interactive: one submit per frame (M4); at 540p the temporal stage is expected at 3–6 ms (gap-temporal §8.2).
- The E2 chunking probe (M4) runs once per chain run on a discarded frame `t = 0x7FFFFFF0` with history invalid.

### 4.5 Pipelines, compile budget, Metal quirks

- New pipelines: refresh ×2, T1, T2, T3, T4 (+ debug variants). Inline budget per pipeline (M4 §4.5): ≤ 1 `bsdf_sample`,
  ≤ 2 `bsdf_query`, ≤ 2 `material_eval` call sites; the refresh loops its NEE/BSDF-end evaluation through one call site
  each. Budget ≤ 1.5 s per pipeline cold; all M5 production pipelines ≤ 5 s in parallel (recorded in budget.json).
- Q1: records are read plane by plane; `fs` is a u32, frame state is fetched from uniforms when needed; T4 keeps
  `TSrc` fields in registers only while the shift runs; no kernel holds a full record and two MatEvals at once.
- Q2: the T3-2 test kernel carries sentinels and then/else counters (PLATFORM class, B-7).
- Q3: no timestamp writes on any ReSTIR pass (a regression test asserts it for the temporal passes).

---

## 5. RNG and seed contract

| Stream | Formula | Used for |
|---|---|---|
| frame key | `key = pcg3d(runSeed ^ member·φ, t, localIdx).xy` (M4) | canonical tree seed, jitter, resampling key |
| temporal pick | `h = pcg4d(runSeed ^ member·φ, t, localIdx, STREAM_TEMPORAL_PICK = 0x2c1b3c6d)` | ξ = (u01(h.x), u01(h.y)) for stochastic rounding, ring rotation `h.z & 7` |
| temporal select | `rs_rand(key_q, RS_PASS_TEMPORAL, counter)`, counter 0 = canonical, 1 = temporal | s (contribution and Talbot) |
| N4 plant | `rs_rand(key(ai), RS_PASS_T_PLANT, j)`, j < 16 | synthetic RIS re-draw (test only) |
| path | M4 (`rs_path_hash` of the record's own 64-bit seed) | replay in T2/T4 (slots 0–3 of the stored seed) |

- **q′ is sample-independent**: it depends on G-buffers (cur/prev, jittered) and the pick stream, never on reservoir
  contents or the path streams (math §15, §23).
- **Validation chains**: chain id `c = memberBase + atlas member` (member in every seed and in the pairing hash);
  `t` = animation frame index counted from the chain's reset (`RsDispatch.t = t`); every chain of a run shares `runSeed`
  and the scene script. Batch b of a run covers chains `[chainBase + b·E, chainBase + (b+1)·E)`. A chain's output is a
  deterministic function of `(runSeed, c, frames 0…t, scene states 0…t)`: bitwise reproducible across runs, across
  row bands and item chunks (U-TR-2), across lock chunks (chains never split), and atlas member m ≡ a sequential run with
  `memberBase = m` (U-TE-1).
- **Interactive**: `t = frame.seedIndex` (A8); history is independent of t. Freeze seed/frame ⇒ reset every frame (TD20).
- **Seeds of the M5 gate** (disjoint from M4's): PT 7001 / re-run 107001; chains 7002 / 107002; A/A 7502 / 7503;
  pilots PT 7011, chains 7012; rendered plants 7101–7140; 4× calibration PT 7201; mask/dominance PT 7301.
- RR (D11) exists only in the canonical path tree; temporal shifts, the refresh and replay never apply RR or read
  `u_rr`. Validation rungs 3.3–3.6 run with RR off (math §25).

---

## 6. Validation design

### 6.1 Gate-0 registry (M5)

| ID | Owner | Where | Pass criterion |
|---|---|---|---|
| U-M4-BITS | A | restir-tframe + pt suites | with temporal off: rung 3.1/3.2 reservoirs and finalize output bit-identical to the pre-M5 build on t3_cases_256, (i) 128², C0s; PT images bit-identical (C0c, C0e, C0m, C0s, (x) quads); `_s` evaluators ≡ wrappers bitwise on 10⁶ random inputs per function |
| U-BIND-1 | A | tests/restir/tframe.test.ts | every pass of §4.2 ≤ 9 storage buffers with the merged scene layout |
| U-TL-1 | A | tests/restir/light-maps.test.ts + restir-tframe | §9.3-1: random add/remove/reorder/type-change sequences: `curToPrev[prevToCur[j]] = j` for survivors, NONE exactly for added/removed; WGSL `lt_translate` ≡ CPU for analytic, triangle and env entries across nA changes |
| U-TL-2 | A | tests/restir/frame-state.test.ts | one commit per frame: env + light + animation edits in one frame give `prev` = frame t−1 state bitwise; unchanged frames set `TF_LIGHTS_SAME`, pmf bitwise equal; change bits per type (TD5) |
| U-TL-3 | A | tests/restir/config-hash.test.ts | every hash input of §2.10 changes the hash; none of the non-reset edits does |
| U-TV-1 | A | restir-tframe | `rsVbufPrev`/`rsGeoPrev` at frame t ≡ `rsVbuf`/`rsGeo` of frame t−1 bitwise; `envPrev` ≡ previous packed words |
| U-TH-1 | A (+E) | restir-tframe | ixs_k frame 20: `TF_HIST_VALID = 0`, `RSC_T_QVALID = 0`; frame 21 valid |
| U-TP-1 | A | restir-tframe | `temporal_pixel`: static camera, jitter off ⇒ q′ = q for every hit pixel; known back-projections for camera translation/rotation/zoom; off-member taps never returned |
| T6(a) | B | tests/restir/tmis.test.ts + restir-temporal | random (c_c, c_p ∈ 0…20 incl. capped, p̂, π incl. zeros): contribution ĉ_c + ĉ_p = 1 where both producible, ĉ_c = 1 where π_p = 0; Talbot \|m_c + m_p − 1\| < 1e-6; E2 ĉ_c ≡ 1 on deep classes; production WGSL ≡ f64 `tmis-ref.ts` to 1e-6 |
| T6(b) | B | restir-temporal (robust mode) | every ixs sequence at its test frames: \|a − b\| ≤ 1e-3·max(a, b) and ID equality on all pixels (target 0; rare threshold cases logged, ≤ 1e-5 of pixels); with N1-mixed it fails exactly on light-dependent pixels; with N3 it fails on M_steady after a move |
| T6(c) | A + B | restir-temporal (`RS_PROVENANCE`) + rc-dual.ts | (i) every evaluator call records `(camGen, vbufGen, lightGen, pmfGen)`: = t−1 for inverse, = t for forward; (ii) 10⁵ canonicals: T⁻¹(X_c) as explicit vertices evaluated by the f64 dual under S_{t−1} equals π_p(X_c)/J_inv within 1e-4 |
| T3-2 | B | restir-temporal | temporal round trips (base in (q′, t−1) from dumped candidates rendered with the prev camera/V-buffer; forward to (q, t); inverse back): every M4 case (a)…(∅) incl. env, k = 2 and k > 2, each ≥ 10⁷, under camera translation/rotation/zoom in/out, jitter on/off, then moving/resized/rotated lights, intensity steps, add/remove, env rotation/strength: LOGIC = 0, FP-BOUNDARY ≤ 1e-5, PLATFORM ≤ 1e-8 (B-7) |
| T4-t | B | restir-temporal | successful T3-2 round trips: \|log(J_fwd·J_inv)\| < 1e-4 including J_P |
| §9.3-2 | C | restir-refresh + refresh-ref.test.ts | (a) FD area ratio of the light-local map = A_t/A_{t−1} within 1e-5 (rect, disk, triangle); (b) PSS change of variables with two lights and an intensity change and with env strength: `E_prev[h(Tx)·J_P·1{x∈D}] = E_cur[h(y)·1{y∈I}]`, 10⁸ samples, z ≤ 4 |
| §9.3-3 | C | restir-refresh | refresh idempotence on unchanged frames (analytic, triangles, env): ΔF ≤ 1e-6 relative |
| §9.3-4 | C | restir-refresh | cached refresh vs full re-trace (replay of the suffix BSDF dims from x_k with the record's seed + NEE to the stored point under frame s) ≤ 1e-4 |
| §9.3-5 | B | restir-temporal | jitter off, static camera and lights: T = identity, J = 1, F_t(Y_p) = F_p^st (≤ 1e-6), temporal output W equals the GRIS formula |
| §9.3-6′ | B | restir-temporal | (amended, open item 9) T defined ⇒ k and class preserved (10⁷ shifts incl. light edits); class-change (undefined) rate per light-change frame reported |
| T-ENV-temporal | C + B | restir-refresh, restir-temporal | refresh idempotence with env; forward/inverse round trips with γ changes (LOGIC = 0); J_P with the env-strength pmf change; PSS z-test with env |
| U-SFX-2 | C | restir-initial (M4 file, C may touch the test) | `SFX_DELTA_END` set iff the final BSDF event was delta; U-SFX-1 still passes |
| U-TQ-1 | B | tests/restir/tqueue.test.ts + restir-temporal | Q_f/Q_i: every item processed once; overflow flag exactly when counter > P; args for n ∈ {0, 1, 64, 65535·64, 65536·64 + 1}; `rs_args` queue selection |
| U-TR-2 | B | restir-temporal | chain of 16 frames with light changes bitwise independent of row bands and item chunks |
| U-TE-1 | B + E | restir-temporal | E = 4 atlas member m ≡ sequential chain with memberBase m, bitwise over 8 frames with light and env changes |
| U-TR-1 | E | tests/restir/gate-m5-config.test.ts + chain run | two chain runs bitwise identical (reservoirs, ensStats, counters) |
| U-TD-1…3 | D | restir-debug | views 480–497 write expected codes/values at known pixels; 493 = 0 on static identity frames; probe tags 73–78 decode to tState/sfxOut read back |
| T3-3-boost | D | restir-spatial (M4 file, D may append) | boost acceptance symmetric, `RSC_SLOT_MISMATCH = 0`, only pairs with a disoccluded member accepted |
| M5 app smoke | D | m5-app-smoke.ts | interactive temporal with animated camera/lights/env on Cornell + HDRI: no NaN, counters 0, HUD temporal lines, map swap resets history, pause encodes no ReSTIR pass, no timestamp writes |
| T15/T16 | all | every GPU run | NaN/Inf 0, negatives 0, all RSC error counters 0 (incl. `RSC_T_NONFINITE`, `RSC_T_PENDING_LEFT`, q1/q2 overflow), config asserts (§6.4) |

### 6.2 Scenes and sequences (T-E, `make-m5.ts`)

**Static subset at 256²** (`m5s_*`, regenerated from the M4 packages with `render = 256²`, everything else identical):
(i) `m5s_cornell_i`, (iii) `m5s_spot_grazing`, (v) `m5s_glossy_v1`, (vi) `m5s_glass_mirror_A`, (x) `m5s_many_lights`,
(xii) `m5s_alpha_foliage`, C0q(d) `m5s_c0q_openbox_b13`, (xiv) `m5s_overcast_rect_b3`. (xiii) and the kloof scenes are
reported, not gating (heavy-tail at 256² is unaffordable at δ; they gate again in M8's `validate --all`).

**Sequence packages** (`ixs_*_256`, maxBounces 3 unless stated, dense frames 0…T−1, TD27):

| Package | Script (gap-temporal §9.1, gap-env §5.2) | T | Test frames |
|---|---|---|---|
| ixs_a_point | point light on a linear path (≈ 3 px/frame shadow edge), behind a box at 20–30, stops at 40 | 81 | 1, 10, 25, 40, 80 |
| ixs_b_area | rect light translate + rotate 5°/frame; area ×1.5 at 16; intensity ×2 at 24 | 41 | 15, 16, 17, 24, 25, 40 |
| ixs_c_spot_b0, ixs_c_spot_b03 | spot rotating 2°/frame, blend 0 and 0.3 | 41 | 5, 20, 40 |
| ixs_d_camera | scripted fly path (static lights) | 65 | 8, 16, …, 64 |
| ixs_d0_jitter | static camera, jitter only | 65 | 8, 32, 64 |
| ixs_d_glossy | ixs_d_camera with a GGX r = 0.3 floor (N6) | 33 | 8, 16, 32 |
| ixs_e_addremove | lights A (point), B (rect), P 1:1; C added at 8; A ×2 at 14; B removed at 20 (true add/remove) | 41 | 8, 9, 14, 15, 20, 21, 40 |
| ixs_e_half | as ixs_e_addremove with A ×0.5 at 14 (N1-mixed positive sign) | 22 | 14, 15, 21 |
| ixs_f_combined | camera + lights, teleport at 24, FOV ramp 30°→60° over 32–48 | 49 | 8, 24, 25, 40, 48 |
| ixs_g_sun | courtyard, sun rotating | 49 | 8, 24, 48 |
| ixs_h_envrot | env rotating 1°/frame, Lambert + GGX r 0.3 objects | 41 | 1, 10, 25, 40 |
| ixs_i_envradio | env strength ×2 at 16, tint change at 24, interior rect power ×2 at 30 | 32 | 15, 16, 17, 24, 25, 30, 31 |
| ixs_j_envcombo | env rotation + fly camera + moving rect light | 41 | 8, 24, 40 |
| ixs_k_envswap | map swap at 20 (reset asserted) | 29 | 19, 20, 21, 28 |
| ixs_n4_twolights | two equal-power point lights, distance ratio 3 (f_B/f_A = 9), rough-floor so that paths are class D-NEE, a third far light moving slightly every frame (refresh every frame) | 17 | 8, 16 |

`PackageFrame` gains `lights[id].enabled` and `env.map` (T-A, scene-package.ts); `sequence: {fps, frameCount,
testFrames, notes}` at top level. `make-m5.ts` is deterministic (run twice, byte-identical, Gate 0).

**U8 scenes** (`u8_c0c_point_b0/b1`, `u8_c0d_spot_b0/b1`, `u8_c0e_rect_b0/b1`, 256²): the C0c/C0d/C0e set-ups with
b = 0 (pure DI, case (a)) and b = 1 with a diffuse side wall (cases (b), (c), (e)).

### 6.3 Rungs

| Rung | Preset | Scenes | Statistic | δ |
|---|---|---|---|---|
| 3.3 temporal, static | `temporal` | 8 × m5s | per-frame ensembles at t ∈ {1, 24} (T = 25) | stage B: 0.2% global, 1% per 32² tile |
| 3.4 full, static | `full` | 8 × m5s | per-frame ensembles at t ∈ {1, 24} | stage B |
| 3.5 time-average | `full` | m5s_cornell_i, m5s_glossy_v1, m5s_overcast_rect_b3 | per chain the mean of frames 32…287 (L = 256); replicates = chains | stage B |
| 3.6 dynamic | `full` | the 13 sequences of §6.2 except the plant-only ixs_d_glossy, ixs_e_half, ixs_n4_twolights; + Talbot on ixs_b_area and ixs_e_addremove; + E2 on the same two; + boost on ixs_d_camera (18 units) | per-frame ensembles at the test frames | dyn: 0.2% global, 2% per 64² tile, 3% per mask region |

The ladder stops per scene at the first failing rung (after its re-run), as in M4. A unit = (scene, rung, test frame);
the suite FWER counts every test frame × 4 channels, plants and A/A.

### 6.4 Ensembles, PT references, masks, statistics

- **Chains**: E = 16 members at 256² (1 M atlas px); R chains per unit from a pilot (64 chains) with the M4 sizing
  method (E7) applied per test frame and aggregate, × 1.25 margin, **floor R ≥ 256** (gap-temporal §9.1), rounded to
  multiples of E. One chain run serves all test frames of its unit. Readback only at test frames: `ensStats` (tiles 16/32/64,
  global, masks) and `ensPixel` (Σx, Σx² over members) are copied and cleared by an `ens_copy[t]` unit; one
  `f<t>/ensemble.npz` per test frame; rows = chains.
- **PT references** (our PT, RR off, same package, frame override of the test frame) at every test frame **and at t−1**,
  sized by pilot, frozen per PT code hash (M4 E7), cached under `validation/out/m5/ptrefs/<pkg>-f<t>-…`. Plant units use
  4× references (M4 calibration rule). Per-light dominance references: our PT with a single emitter enabled at the
  test frames that need them (N1, N2, N7, env plants).
- **Masks** (`dyn_masks.py`, deterministic from the frozen PT means and geometry, never from ReSTIR samples), region ids
  in priority order: 0 `M_disocc` (pixel-centre V-buffers at t−1 and t with jitter off through the production
  `temporal_pixel`, a harness mode of the chain runner), 1 `M_new` (4×4-block PT mean: L_{t−1} ≤ 0.01·mean_{t−1} and
  L_t ≥ 0.05·mean_t), 2 `M_gone` (symmetric), 3 `M_edge` (|L_t − L_{t−1}| ≥ 0.25·max, else) , 4 `M_steady`; plus
  dominance regions `M_light:<id>` (one light ≥ 70% of L) used only by plant sign checks. Uploaded as the `r8uint`
  mask texture for the test frame (M ≤ 16).
- **Statistics**: static rungs = compare.py stage B (M4 unchanged); 3.6 = stage `dyn` (exists in stats.py). New in
  `dynamic.py`: per sequence the series {Δ_f/SE_f} over test frames (reported), drift regression of Δ on f (slope within
  its 99% CI of 0; gating), and the fraction of failing tiles per frame vs Binomial(m, α′) (gating at α_u).
- **T16 for M5 units**: preset as the rung, no plants unless named, cCap 20, tMode as the unit, RR off, jitter iid,
  Mode A, chains reset at t = 0, chain length and test frames = the package's `sequence`, PT refs at the same resolved
  frame states (package frame hash), `spatialRoundsExecuted` = rounds for 3.4–3.6, temporal units executed every frame,
  all counters 0.

### 6.5 Plants (rendered; detected **and** predicted sign, TD29)

Detection as M4 (full comparison not `pass`; half-size calibrated repeats ≥ 9/10 against 4× PT references; PT A/A ≥ 9/10).
Sign check (`plant_sign.py`): the relative Δ_Y over the predicted region at the predicted frames has the predicted sign
with one-sided z ≥ 3; for "only" predictions no region/tile shows the opposite sign with z ≥ 4.

| Plant | Flag | Scene, frames | Predicted |
|---|---|---|---|
| N1-mixed | `TP_N1_MIXED` | ixs_e_addremove 14, 15 | Δ < 0 in `M_light:A` (≈ B(2) = −4.3% of A's contribution); frame 8: Δ ≪ 0 in `M_light:C` (≈ −95%) |
| N1-mixed r = 0.5 | `TP_N1_MIXED` | ixs_e_half 14, 15 | Δ > 0 in `M_light:A` (≈ +2.3%) |
| N1-consistent | `TP_N1_MIXED + TM_PP_RECOMPUTE` | ixs_a_point 10, 25 | Δ < 0 in `M_new`; no significant Δ > 0 anywhere ("darkening only") |
| N2 | `TP_NO_JP` | ixs_e_addremove 14, 15 | Δ < 0 in `M_light:A`, Δ > 0 in `M_light:B` |
| N3 | `TP_N3_STALE` | ixs_a_point 40, 80 | Δ > 0 in `M_gone`, Δ < 0 in `M_new`; T6(b) fails |
| N4 | `TP_N4_RIS` | ixs_n4_twolights 8, 16 | Δ > 0 globally (D-NEE brightening; +64% is the M → ∞ bound) |
| N5 | `TP_N5_PIXEL_CENTRE` | ixs_d0_jitter 32; ixs_d_camera 16 | Δ < 0 in `M_edge` (silhouettes) |
| N6 | `TP_N6_CUR_CAM` | ixs_d_glossy 16, 32 | detected; sign not predicted (reported) |
| N7 | `TP_N7_PER_LIGHT` | ixs_e_addremove 14, 15 | Δ < 0 in `M_light:B` (pmf_B drops, stale 1/q) |
| skip rotation refresh | `TP_ENV_NO_ROT_VIS` | ixs_h_envrot 10, 25 | Δ > 0 in `M_edge` |
| E_{t−1} with γ_t | `TP_ENV_GAMMA_T` | ixs_h_envrot 10, 25 | Δ < 0 in `M_new`, Δ > 0 in `M_gone` |
| omit J_P on env | `TP_NO_JP_ENV` | ixs_i_envradio 16, 17 | Δ < 0 in the env-dominant region, Δ > 0 in `M_light:rect` |
| c_p + 1 | `TP_CP_PLUS1` | m5s_cornell_i 3.4, t = 24 | Δ < 0 globally |
| W × 1.003 | synthetic (compare.py) | A/A run of m5s_cornell_i 3.4 | detected ≥ 9/10 |
| U8-1 ω1 < 1 for delta | `RSF_PLANT_U8_W1DELTA` | u8_c0c_point_b1, 3.4 | Δ < 0 |
| U8-3 no p_k ratio | `RSF_PLANT_U8_NO_PK` | u8_c0e_rect_b1, 3.4 | detected (sign reported) |
| U8-4 J = t_x²/t_y² | `RSF_PLANT_U8_T2` | u8_c0c_point_b0, 3.4 | detected (sign reported) |
| U8-6 one-sided ignored | `RSF_PLANT_U8_ONESIDED` | u8_c0e_rect_b1, 3.4 | Δ > 0 |
| U8-9 FAILED dropped from k | `RSF_PLANT_U8_FAILED_K` | u8_c0e_rect_b1, 3.4 | Δ < 0 (energy loss) |
| U8-2t stale aux across frames | `TP_U8_STALE_AUX` | ixs_b_area 16, 24 | detected |
| U8-5t spot profile of frame t−1 | `TP_U8_SPOT_PREV_AXIS` | ixs_c_spot_b03 20 | detected |
| A/A | seeds 7502 / 7503 | m5s_cornell_i 3.4 ×4 | pass; re-splits at nominal rate |

The U8 ladder also runs the unbiased rungs 3.1, 3.1b, 3.2, 3.3, 3.4 on the six u8 scenes (all must pass).

### 6.6 Budget estimate and cost controls

Anchors (replaced by pilot measurements, written to budget.json): `full` preset ≈ 5 ms per 256² member-frame
(≈ 80 ns/px incl. temporal, from the M4 interactive preset and the M4 3.2 rows); `temporal` ≈ 3 ms; PT ≈ 0.6 ms per
spp at 256² (M4 (i): 2.3 ms/spp at 512²); Chrome/Vite start ≈ 20 s per run-batches invocation.

| Block | Content | Estimate |
|---|---|---|
| Gate 0 | M4 suites (incl. T3 18 min × variants) + restir-tframe, restir-temporal (T3-2 ≥ 10⁷/case × motion variants), restir-refresh, restir-debug, M3 regressions, CPU/Python tests | 2.5 h |
| PT references | static 8 × 256²; dynamic ≈ 60 (test frame, t−1) pairs; dominance renders; 4× plant refs; re-run refs | 1.2 h |
| Pilots | 64 chains per unit type | 0.4 h |
| 3.3 + 3.4 | 16 units, R by pilot (≈ 1–5 k chains × 25 frames) | 2.5 h |
| 3.5 | 3 units, ≈ 64–128 chains × 288 frames | 0.6 h |
| 3.6 | 18 units, R ≈ 256–1024 × T ≤ 81 | 2.4 h |
| Plants + calibration | 21 rendered plants at 1× (chains cut after the last predicted frame) + A/A ×4 + synthetic | 1.5 h |
| U8 ladder | 6 small scenes × 5 rungs | 0.8 h |
| App smoke | – | 0.1 h |
| **Total** | first run, cold caches | **≈ 12 h** (range 10–16 h) |
| | re-run with frozen PT sizes and cached references/pilots | ≈ 10 h |

For comparison the M4 gate took 12.47 h for 63 units at 512² with ~4.6 h of Stage-B ReSTIR runs. Cost controls, in
the order the gate applies them (δ is never loosened):
1. All temporal units at 256² (TD28); static rungs on the 8-scene subset.
2. One chain run serves every test frame of its unit; PT references only at test frames and t−1 (PLAN §7.3).
3. Frozen PT sizes and the reference cache shared by rungs, modes, plants and re-runs (M4 E6/E7).
4. Pilot sizing per unit; floor R ≥ 256; per-batch rounding for cache reuse.
5. **Unit cap 30 min per side.** Over the cap: enlarge the tile aggregate one step (static 32² → 64², dyn 64² → 128²;
   masks unchanged), record `aggregate_enlarged`; the **global** aggregate is never enlarged: a unit whose global SE
   alone exceeds the cap is `infeasible` and goes to the coordinator (Q4), it is never passed.
6. Plants run at 1× with 4× PT references; their chains stop after the last predicted frame.
7. GPU lock held per chunk of whole batches, ≤ 12 min (M4 lesson); chains are never split.
8. If the pilot-sized plan exceeds 14 h, the gate runs as two required parts: `--part core` (Gate 0, 3.6, plants, A/A,
   app smoke) and `--part static` (3.3–3.5, U8 ladder), each within the weekly tier.
9. A failed unit is re-run once on disjoint seeds (both sides), as in M4.

### 6.7 `npm run validate -- --milestone M5` (gate-m5.ts)

1. **Gate 0** (§6.1): typecheck; `vitest --project cpu`; python tests (incl. dynamic.py, dyn_masks.py, plant_sign.py);
   `make-m5.ts` determinism; Chrome GPU suites restir-tframe, restir-temporal (`VITE_T3_MS` per variant as M4),
   restir-refresh, restir-debug, the M4 suites restir-initial/shift/spatial and the M3 regressions pt, bsdf, lights,
   env-sampling, glass, pt-glass; each suite under one GPU-lock hold; `m5-app-smoke.ts`; budget rows present.
2. **Gate 3**: pilots → sizing → PT references (cached, frozen) → per scene 3.3 → 3.4 → 3.5 (where listed) and per
   sequence 3.6 through `run-batches.ts --kernel restir --chains … --preset <rung> [--temporal-mis talbot]
   [--refresh e2] [--boost 3] [--tplant …]`, then compare.py (stage B / dyn) and dynamic.py.
3. **Plants and calibration** (§6.5) with `plant_sign.py`.
4. **U8 ladder** (§6.5).
5. Output `validation/out/m5-gate-<time>/summary.{json,md}` (units with rung, test frame, sizes, MDB, pass/fail,
   re-runs, plants with detected/sign, counters, timings, budget); `--only`, `--part`, `--pilot-only`,
   `--prerender-ptrefs`, `--write-budget` as in M4. Before `git worktree remove`, copy `validation/out/m5*` to the main
   checkout (M4 lesson).

---

## 7. Risks and open decisions

| # | Risk / decision | Recommendation |
|---|---|---|
| R1 | FP asymmetry between T (T1 inline / T2 replay) and T⁻¹ (T4) pipelines near predicate thresholds | One shift code path with `fs`; stored thr_{t−1}; D3/B-2 directions; T3-2 measures FP-BOUNDARY ≤ 1e-5; single-engine build only if it exceeds (M4 D21). |
| R2 | Metal Q1 state-size cliff in T2/T4 (shift + replay + temporal loaders) | Plane-wise loads, `fs` as u32, `TSrc` built right before the call, canaries in every new GPU test file. |
| R3 | Metal Q2-like control-flow faults in test kernels | T3-2 harness with sentinels and PLATFORM class (B-7); production passes have no shift loop. |
| R4 | Q3 timestamps | Forbidden on temporal passes; regression test. |
| R5 | Gate cost above the weekly tier | §6.6 controls; two-part split above 14 h. |
| R6 | Global δ = 0.2% on 256² dynamic sequences may need R > the cap on hard scenes (spot, many lights) | Pilot early (first week of integration); aggregates enlarge only for tiles/masks; `infeasible` units escalate (Q4). |
| R7 | Contribution MIS variance under strong light motion (GRIS §5.6) | Report the efficiency ratio per unit; Talbot units give the comparison. |
| R8 | Light-slot double flip per frame (today's app) | TD4, U-TL-2. |
| R9 | Triangle/env alias entries shift when nA changes | `lt_translate` handles all kinds; U-TL-1. |
| R10 | In-place temporal output read by a later pass expecting the canonical | Pixel-local write rule (§2.2); the initial tap is written by rs_initial; T4 reads only s = c pixels. |
| R11 | Pause / freeze seed in the app consuming history twice or reusing seeds | TD20. |
| R12 | N1 end visibility "trace iff moved" relies on V = 1 for every refreshed record | Holds because refreshed records have p̂ > 0 in their own frame (they were selected); §9.3-3/-4 and T6(b) catch violations. |
| R13 | ∅ and k = d records carry the source's endpoint/lobeHist after selection (M4 R13) | The refresh never reads them; the shift re-evaluates; documented. |
| R14 | N4 has no RIS tiles before M6 | Synthetic RIS re-draw plant (TD31). |
| R15 | U8 plants 5, 7, 8, 10 | 5 replaced by its temporal analogue (spot axis of t−1); 7 (Mode B), 8 and 10 (RIS-NEE tiles) move to M6. |
| R16 | Mode B temporal (crossing loop over lightsPrev) | Out of M5: the kernel rejects Mode B (D1); M6 adds it. |
| R17 | Moving emissive meshes need a previous TLAS | Out of scope (static geometry, user decision); `emissive triangles static` is asserted by the package generator. |
| R18 | The dataformats merge changes the scene group | Only abstract accessors; U-BIND-1. |
| R19 | Masks from noisy PT references | 4×4-block means, frozen references, masks never from ReSTIR data; plants' sign regions use dominance renders. |
| R20 | 3.5 time-average hides per-frame bias that alternates in sign | 3.3/3.4 per-frame units cover t = 1 and 24; 3.5 adds long-run convergence only. |

**Open questions for the coordinator** (each with the recommended answer):
- **Q1** P0 by T-A in ≤ 1 day from appendix B, after the dataformats merge. *Recommended: yes.*
- **Q2** Static temporal rungs 3.3–3.5 at 256² on the 8-scene subset instead of the 21 M4 scenes at native resolution;
  (xiii) and kloof reported only. *Recommended: yes (TD28).*
- **Q3** Gate budget ≈ 12 h (10–16 h) in the weekly tier, and the two-part split when the pilot plan exceeds 14 h.
  *Recommended: accept.*
- **Q4** Handling of a unit whose global aggregate cannot reach its SE target within the 30-min cap. *Recommended:
  raise that unit's cap to 90 min once; if still infeasible, the coordinator decides between a longer run and removing
  the sequence from gating with a written reason (never a looser δ).*
- **Q5** N4 as a synthetic RIS re-draw; N6 detection without a predicted sign. *Recommended: yes; N6's sign is
  genuinely mixed under contribution MIS (gap-temporal §9.4).*
- **Q6** U8 ladder subset (1, 3, 4, 6, 9 + temporal variants of 2 and 5; 7, 8, 10 → M6). *Recommended: yes.*
- **Q7** Boost default on in interactive, off in validation except the gating ixs_d unit. *Recommended: yes.*
- **Q8** Dual MVs in M6. *Recommended: yes.*
- **Q9** Rung 3.5 = per-chain mean over frames 32…287 on three scenes. *Recommended: yes.*
- **Q10** `rs_refresh_inv` scope = Q_i (canonicals whose π_p is needed) instead of every canonical (PLAN §3 says
  "canonicals"; the values are exact either way, only the work differs). *Recommended: Q_i.*
- **Q11** Talbot and E2 each gate on ixs_b_area and ixs_e_addremove (gap-temporal §9.1 "all exact modes pass"); E3 is
  not implemented. *Recommended: yes.*
- **Q12** New `ixs_*` packages instead of rewriting the M3a `ix_*` packages. *Recommended: yes.*

---

## Appendix A. math.md additions made with this contract

All marked **[M5 addition, restir-temporal-api.md]**: §15 (temporal pick and select streams, chain seeds), §17 (suffix
flag `SFX_DELTA_END`), §18 (J_P storage, translation of all entry kinds, undefined on a missing entry or pmf ≤ 0, write-back
excluding J_P), §23 (q′ rule, no-q′ identity, empty source counts, selection order and counters, in-place output,
select phases and the Talbot/recompute/robust variants, wScale on temporal W), §24 (refresh responsibilities split
between the refresh pass and the shift, MOVED/RADIO definitions, N1/D-NEE visibility rule, deep refresh formulas,
E2 definition, per-light refresh masks forbidden), §26 (no-q′ keeps c = 1, empty output c = 1 + c_p), open
inconsistencies 43–52.

## Appendix B. Normative WGSL blocks (P0 lands them verbatim)

### B.1 `restir/types.wgsl` (appended)

```wgsl
// M5 (restir-temporal-api.md §2.1)
const RS_FS_CUR: u32 = 0u;  const RS_FS_PREV: u32 = 1u;
const TF_HIST_VALID: u32 = 1u;  const TF_LIGHTS_SAME: u32 = 2u;  const TF_ENV_SAME: u32 = 4u;  const TF_REFRESH: u32 = 8u;
const TF_PMF_CHANGED: u32 = 16u;  const TF_ENV_MOVED: u32 = 32u;  const TF_ENV_RADIO: u32 = 64u;  const TF_LIGHT_MOVED: u32 = 128u;
const TF_RESET: u32 = 256u;  const TF_CAM_SAME: u32 = 512u;
const TM_TALBOT: u32 = 1u;  const TM_PP_RECOMPUTE: u32 = 2u;  const TM_ROBUST: u32 = 4u;  const TM_E2: u32 = 8u;
const TP_N1_MIXED: u32 = 1u;  const TP_NO_JP: u32 = 2u;  const TP_NO_JP_ENV: u32 = 4u;  const TP_N3_STALE: u32 = 8u;
const TP_N4_RIS: u32 = 16u;  const TP_N5_PIXEL_CENTRE: u32 = 32u;  const TP_N6_CUR_CAM: u32 = 64u;  const TP_N7_PER_LIGHT: u32 = 128u;
const TP_ENV_NO_ROT_VIS: u32 = 256u;  const TP_ENV_GAMMA_T: u32 = 512u;  const TP_CP_PLUS1: u32 = 1024u;
const TP_U8_STALE_AUX: u32 = 2048u;  const TP_U8_SPOT_PREV_AXIS: u32 = 4096u;
const RSF_TEMPORAL: u32 = 64u;
const RSF_PLANT_U8_W1DELTA: u32 = 256u;  const RSF_PLANT_U8_NO_PK: u32 = 512u;  const RSF_PLANT_U8_T2: u32 = 1024u;
const RSF_PLANT_U8_ONESIDED: u32 = 2048u;  const RSF_PLANT_U8_FAILED_K: u32 = 4096u;
const RSD_QUEUE_SHIFT: u32 = 8u;  const RSD_PHASE_B: u32 = 32u;
const LCB_MOVED: u32 = 1u;  const LCB_RADIO: u32 = 2u;  const LCB_ADDED: u32 = 4u;
const RS_PASS_T_REFRESH_FWD: u32 = 9u;  const RS_PASS_T_CLASSIFY: u32 = 10u;  const RS_PASS_T_FWD: u32 = 11u;
const RS_PASS_T_REFRESH_INV: u32 = 12u;  const RS_PASS_T_INV: u32 = 13u;  const RS_PASS_T_PLANT: u32 = 14u;
const STREAM_TEMPORAL_PICK: u32 = 0x2c1b3c6du;
const SFX_DELTA_END: u32 = 8u;
const RSC_T_QVALID: u32 = 48u;  const RSC_T_DISOCC: u32 = 49u;  const RSC_T_FWD_QUEUED: u32 = 50u;  const RSC_T_FWD_OK: u32 = 51u;
const RSC_T_SEL_P: u32 = 52u;  const RSC_T_INV_QUEUED: u32 = 53u;  const RSC_T_INV_OK: u32 = 54u;  const RSC_T_EMPTY_OUT: u32 = 55u;
const RSC_T_LIGHT_UNDEF: u32 = 56u;  const RSC_T_CLASS_UNDEF: u32 = 57u;  const RSC_T_REFRESH_RECS: u32 = 58u;
const RSC_T_REFRESH_RAYS: u32 = 59u;  const RSC_T_E2_ZEROED: u32 = 60u;  const RSC_T_ROBUST_MISMATCH: u32 = 61u;
const RSC_T_NONFINITE: u32 = 62u;  const RSC_T_PENDING_LEFT: u32 = 63u;
const RS_Q_SPATIAL: u32 = 0u;  const RS_Q_FWD: u32 = 1u;  const RS_Q_INV: u32 = 2u;
```

### B.2 Uniforms (`restir/frame.wgsl` edit, `restir/tframe.wgsl`)

```wgsl
// RestirParams words 104..127 (was pad1, pad2, pad3): struct size unchanged (128 B)
  boostSlots: u32,             // 104  NB (0 = off); numSlots = slots + NB
  tMode: u32,                  // 108  TM_*
  cCap: f32,                   // 112  20
  tPlants: u32,                // 116  TP_*
  pad4: u32, pad5: u32,        // 120..127
// restir/tframe.wgsl
struct RsTemporal {            // 128 B, G0 binding 8 (layout.ts RS_TEMPORAL_SIZE)
  flags: u32,                  //   0  TF_*
  histFrames: u32,             //   4  frames since the last reset (0 on a reset frame)
  frameGen: u32,               //   8  advance() counter of frame t
  prevGen: u32,                //  12  advance() counter of the E_{t−1} snapshot
  envPrev: EnvParams,          //  16  verbatim packed EnvParams words of frame t−1 (32 B)
  gens: vec4u,                 //  48  (camGen, vbufGen, lightGen, pmfGen) of frame t
  gensPrev: vec4u,             //  64  same of frame t−1
  configHash: u32,             //  80
  pad0: u32, pad1: u32, pad2: u32,   // 84..95
  pad3: vec4u, pad4: vec4u,    //  96..127
}
@group(0) @binding(8) var<uniform> rsTemporal: RsTemporal;
```

### B.3 tState / sfxOut accessors (`restir/tframe.wgsl`, real in P0)

```wgsl
const TS_QVALID: u32 = 1u;  const TS_DISOCC: u32 = 2u;  const TS_FWD_QUEUED: u32 = 4u;  const TS_FWD_DONE: u32 = 8u;
const TS_SEL_P: u32 = 16u;  const TS_SEL_C: u32 = 32u;  const TS_INV_QUEUED: u32 = 64u;  const TS_INV_DONE: u32 = 128u;
const TS_EMPTY_OUT: u32 = 256u;  const TS_NO_HIST: u32 = 512u;  const TS_PICK_RING: u32 = 1024u;  const TS_ROBUST: u32 = 2048u;
const TS_E2_ZERO: u32 = 4096u;  const TS_FINAL: u32 = 8192u;  const TS_BG: u32 = 16384u;
const SXS_DONE: u32 = 1u;  const SXS_UNDEF: u32 = 2u;  const SXS_VIS: u32 = 4u;  const SXS_RAY: u32 = 8u;  const SXS_DEEP: u32 = 16u;
const SXS_N1: u32 = 32u;  const SXS_B1: u32 = 64u;  const SXS_ZERO: u32 = 128u;  const SXS_E2: u32 = 256u;  const SXS_PLANT: u32 = 512u;
const SFX_FWD: u32 = 0u;  const SFX_INV: u32 = 1u;
fn arena_tbase() -> u32;                                  // 64 + 6·P·NS_alloc (P, NS from rsParams)
fn ts_word(ai: u32, w: u32) -> u32;                       // arena word index of tState[ai] word w (0..19)
fn sfx_word(dir: u32, ai: u32, w: u32) -> u32;            // arena word index of sfxOut[dir][ai] word w (0..7)
fn queue_item_base(q: u32) -> u32;                        // 0 (q0), 0 (q1), P (q2) within the item region
fn queue_capacity_q(q: u32) -> u32;                       // P·NS (q0), P (q1, q2)
```
(`NS_alloc = numSlots` when temporal is off; `max(numSlots, 2)` with `RSF_TEMPORAL`; `layout.ts` mirrors every function.)

---

## Changelog

Amendments made while implementing this contract. Numbering is append-only (T-prefixed per work package: `A-n`,
`B-n`, `C-n`, `D-n`, `E-n`, coordinator `Q-n`). Every WP reads this section before touching a shared interface.

### Coordinator decisions on §7 (2026-09-30)

- **Q1** Yes: T-A lands P0 now (branch `m5-temporal` on main e251b43 incl. the dataformats merge).
- **Q2** Yes: static temporal rungs 3.3–3.5 at 256² on the 8-scene subset (TD28); (xiii) and kloof reported only.
- **Q3** Accepted: gate budget ≈ 12 h, plus the two-part split (`--part core` / `--part static`) when the pilot plan exceeds 14 h.
- **Q4** A unit whose global aggregate misses its SE target within 30 min gets its cap raised **once** to 90 min; beyond
  that it escalates to the coordinator. δ is never loosened.
- **Q5** Yes to both: N4 as a synthetic RIS re-draw; N6 detection without a predicted sign.
- **Q6** Yes: U8 subset 1, 3, 4, 6, 9 + temporal variants of 2 and 5; 7, 8, 10 move to M6.
- **Q7** Yes: boost default on in interactive, off in validation except the gating ixs_d unit.
- **Q8** Dual MVs are deferred to M6.
- **Q9** Yes: rung 3.5 = per-chain mean over frames 32…287 on three scenes.
- **Q10** `rs_refresh_inv` scope = Q_i only.
- **Q11** Yes: Talbot and E2 each gate on ixs_b_area and ixs_e_addremove; E3 is not implemented.
- **Q12** New `ixs_*` packages (the M3a `ix_*` packages stay untouched).

### P0 amendments (T-A)

- **A-1 Arena indexing and NS_alloc** (affects T-B, T-C, T-D, T-E: every arena accessor). The WGSL accessors of B.3
  (`arena_tbase`, `ts_word`, `sfx_word`) return **words[] indices** (after the 64-word header), like queue.wgsl's
  `arena_*_word` (M4 A10): `arena_tbase() = 6·P·NS_alloc` there, i.e. global word `64 + 6·P·NS_alloc` of §2.8. The TS
  mirror `arenaWords(P, NS)` gains `tState`, `sfxOut`, `end` in the same indexing; `arenaBytes(P, NS, temporal)`,
  `nsAlloc(numSlots, temporal)`. NS_alloc enters the M4 layout through one line of `restir/queue.wgsl` (P0 edit of a
  T-B file): `queue_capacity() = P·rs_ns_alloc()` with `rs_ns_alloc() = RSF_TEMPORAL ? max(numSlots, 2) : numSlots`
  (bitwise M4 with temporal off; q0's capacity is P·NS_alloc). `RestirAllocation.slots` is NS_alloc.
- **A-2 `restir/tmis.wgsl`** (new file, owner T-B; affects T-B). `rs_t_select` has no scene group and cannot include
  `shift.wgsl`, so the pure `tmis_*` functions live in `restir/tmis.wgsl` (real since P0), included by `tshift.wgsl`
  and `t-select.wgsl`. The write-back `res_select_temporal` must likewise stay scene-free (T-B's choice of file).
- **A-3 Interactive preset in P0** (affects T-D). `interactive` has `temporal: true` (TD24) but `boostSlots: 0` until T-D
  lands the boost acceptance, then T-D sets 3 in `presets.ts` (T-D may touch that one line). The kernel emits temporal
  units only on frames prepared by `advance()` / `advanceInteractive()`; `frameUnits()` without a preceding advance is a
  reset frame (h = −1, w = 0, no temporal units), so every M4 caller, and the app until T-D wires
  `advanceInteractive`, is bitwise M4. `numSlots = slots + boostSlots` only with temporal on (`presets.ts numSlotsOf`).
- **A-4 G-buffer ping-pong and bind-group keys** (affects T-B, T-C, T-D, T-E). The second `rsVbuf`/`rsGeo` pair is
  allocated only with `settings.temporal` (else both entries alias one texture). `RestirResources.vbuf`, `.geo`,
  `.views.vbuf`, `.views.geo` are getters of the **current** parity; `vbufPrev`, `geoPrev` (and views) of 1 − g.
  `res.parity` is flipped by `advance()`. Every `g2()` cache key carries the parity; `render/restir/debug.ts` caches its
  G2 per (reservoir index, parity) (P0 edit beyond the input index). `g2(name, idx)` index semantics: `rs_initial` /
  `rs_initial_dump` idx = w (the output buffer); `rs_refresh_fwd`, `rs_t_classify`, `rs_t_forward` idx = h (resIn);
  `rs_t_select` idx = h (resIn = res[h], resOut = res[1 − h] = res[w]); `rs_refresh_inv` idx = w (resIn), `rs_t_inverse`
  idx = w (resOut).
- **A-5 tframe.wgsl extras** (affects T-B, T-C, T-D). Besides B.3: `TSW_*` word indices of §2.8 (TS mirror `TSW`),
  `TS_QPRIME_NONE`, `rs_tf(bit)`, `rs_tplant(bit)`, `rs_tmode(bit)`, `ts_load/ts_loadf`, `ts_store/ts_storef`,
  `ts_clear(ai, flags)` (T1's cleared record), and `struct SfxRec` itself (`sfx_store`/`sfx_load` need it). The temporal
  debug hooks of §2.11 are declared in `debug/restir-views.wgsl` only under `RS_TEMPORAL` (which includes tframe.wgsl),
  so no M4 module changes (checked by tests/restir/tframe.test.ts).
- **A-6 Kernel API** (affects T-B, T-D, T-E). `RestirKernel` gains `historyIndex()` (h, −1 = none), `resBase()` (w),
  `finalResIndex()`, `currentAdvance`, `frameState` (frame-state.ts `FrameStateTracker`), `rsTemporal` (the uniform
  buffer), `envParamsWords()`; `readTemporalState()` returns the words from tState to the end of sfxOut (decode with
  `layout.ts decodeTStateLocal / decodeSfxLocal`). `readReservoirs('final')` reads `res[finalResIndex()]`.

### A1 amendments (T-A)

- **A-7 q′ pixel convention** (affects T-B, T-D views 481/tag 78, T-E masks). §3.3's `sp` is the continuous member-local
  position with pixel **centres at integers**: `sp = frame_project(x₁, prevCam) − 0.5`, then `c₀ = ⌊sp + ξ⌋`. With
  frame_project's own convention (centres at +0.5) a static, jitter-off camera would round to q + 1 half of the time,
  contradicting U-TP-1 (q′ = q). `TPick.sp` and view 481 (`s′ − q`) use this convention: for a static camera s′ − q is 0 up to the f32 round-off of
  the back-projection (measured ≤ 1e-5 px; never exactly 0 in general, so tests use a 1e-4 px tolerance).
  A position behind the previous camera, or more than 2 px outside the member tile, has no tap.
- **A-8 Package frame format (TD27; confirms T-E's proposal with two amendments)** (affects T-E `make-m5.ts`, the chain
  runner, `batch-run.ts loadSource`). `scene.json`:
  `frames[k] = { frame, label?, camera?: {matrix, yfov}, lights?: { "<id>": { matrix?, power?, enabled?, color?,
  exposure?, sizeX?, sizeY?, spotSize?, spotBlend?, spread? } }, env?: { rotationZ?, strength?, tint?: [r,g,b],
  visibleToCamera?, map? } }` — dense (one entry per frame 0…T−1) for `ixs_*`; values are absolute; a missing field /
  light / frame keeps the base scene state; `enabled: false` ⇒ the light is absent in that frame (every light that
  ever exists is in the base `lights[]`). Top level `sequence: {fps, frameCount, testFrames, notes}` and
  `envMaps: [{ id, file, sha256, width, height, name? }]` (**array keyed by id**, file `env_<id>.exr`, written by
  `exportScenePackage({ envMaps: [{ id, env }] })` with encodeEnvExr); `frames[k].env.map` is an **id** (`'env'` =
  the base env.exr, constant `BASE_ENV_MAP`), not a file name. `readScenePackage` / `fetchScenePackage` load and
  hash-check every map (`LoadedScenePackage.envMaps: Map<id, EnvironmentData>`, `.sequence`). The one shared resolver
  is `scene-package.ts resolvePackageFrame(pkg, k) → { frame, camera: {camToWorld: Float64Array, yfov}, lights (enabled
  only, overrides applied), env?: { params: {rotationZ, strength, tint, visibleToCamera}, mapId, map } }`; the chain
  runner feeds it to `RestirKernel.advance({ t, camera, lights, env: { params, mapId } })` after
  `kernel.setEnvironment(...)` when `mapId` changes, and `loadSource` uses it for `--frame` (M3 packages resolve to
  exactly their previous state). The Blender bridge (`build_scene.py`) does **not** need the new fields: every M5 unit
  compares against our PT (§6.4, Stage B / dyn), so no `ixs_*` package is rendered by Cycles.
- **A-9 Staged light commits** (affects T-D `renderer.ts` / `integration.ts`). With `settings.temporal` the kernel's
  `LightsGpu.deferred` is on: `setLights`, `setEnvironment`, `envParamsChanged`, `setEnvOptions` only stage (their
  returned `LightsUpdate` is provisional; bind groups follow `lights.version`) and the ONE commit happens in
  `advance()` / `advanceInteractive()` (`LightsState.commit`: no flip when nothing changed ⇒ `TF_LIGHTS_SAME`, change
  bits in word 26). A frame built without advance commits the staged edits itself (a reset frame). The PT's
  `LightsGpu` is unchanged (every update flips). `setEnvironment` and `setView` invalidate history (reasons `env-map`,
  `view`); the config hash (§2.10) additionally covers every RestirSettings field and the env options.
- **A-10 `currentAdvance` during unit building** (requested by T-B; affects T-B, T-C). `RestirKernel.currentAdvance`
  (alias `frameAdvance`) returns the frame's `RestirAdvance` from `advance()` until the next `advance()`, including while
  `frameUnits()` builds the stage units (TF_REFRESH, TF_HIST_VALID for the temporal stage and the refresh builders);
  undefined on a non-advanced (reset) frame.

### T-C amendments (suffix refresh, change taxonomy)

- **C-1 `SFX_DELTA_END` only with `RSF_TEMPORAL`** (affects T-A U-M4-BITS). The path tree sets word 27 bit 3 only when
  `RSF_TEMPORAL` is set, so every temporal-off reservoir is bitwise M4. A kernel with temporal settings that is never
  advanced (the interactive preset "as shipped" before T-D's wiring) may differ from the pre-M5 build in that bit on
  delta BSDF endings; images and counters are unaffected (no M4 code reads the bit).
- **C-2 Undefined predicate of the refresh** (affects nobody; T-B consumes `SXS_UNDEF`). `refresh_entry` marks an NEE
  record undefined iff `e_to` is missing or `pmf_to(e_to)`, `pmf_from(e)` or `J_P` fails the bit test `rs_pos_finite`
  (the same predicate in T and T⁻¹; §3.2 plus the source pmf, which is > 0 for every served record). BSDF_ANALYTIC
  records (Mode B, unreachable in M5) are undefined.
- **C-3 N4 plant = single-sample value of the RIS pick, forward only** (affects T-E's N4 unit: prediction unchanged,
  Δ > 0). §3.5's `rad = β_s ⊙ N(new)·W_RIS` is an unbiased RIS estimate of the NEE integral and would not reproduce the
  +64 % of gap-temporal §5.2, which comes from keeping the per-sample integrand `F(new) = f/p` of a target-distributed
  pick. The plant draws 8 alias candidates under the frame-t slot, picks ∝ `lum(f_all·Λ)/q` and sets
  `rad = β_s ⊙ (ω1/q)·f_all·Λ` of the pick (shadow ray always traced). It acts on `fsTo = CUR` only (frame-t random
  numbers; the inverse is evaluated under S_{t−1}). Stream: `pcg4d(key.x, key.y ^ RS_PASS_T_PLANT·φ, j,
  STREAM_RESAMPLE)`: counter j < 8 gives (u_sel, u_sel2, h_l0, h_l1) of candidate j, counter 8 + j gives (h_l2, u_pick).
- **C-4 U-SFX-2 lives in `validation/gpu-tests/restir-refresh.gpu.test.ts`** (affects nobody), not in
  restir-initial.gpu.test.ts, which T-B edits concurrently.
- **C-5 Stale plants are delivered by the refresh** (affects T-B: the loader's "unless `TP_N3_STALE`" check becomes
  redundant, both are consistent). `TP_N3_STALE`: every served class returns the stored `rcRad` / `aux` and N1 gets
  `SXS_VIS` without a ray (stale visibility); entry and `J_P` are still translated. `TP_N7_PER_LIGHT`: the same for records
  whose endpoint has neither `LCB_MOVED` nor `LCB_RADIO`; for BSDF ends the endpoint is the hit emitter (triangles never
  change, env escapes use `TF_ENV_MOVED` / `TF_ENV_RADIO`). Plant outputs carry `SXS_PLANT`.
- **C-6 Details of §3.5** (affects nobody). N1 on a moved light: no ray and `SXS_VIS` clear when the re-evaluated end is
  invalid or Λ = 0 (F = 0 either way). `TP_ENV_NO_ROT_VIS` skips the ray for env NEE ends of N1 and D-NEE. D-BSDF uses
  the path tree's grouping `(β_s·ω2)·L`; `SXS_ZERO` is set on any deep record with rad = 0. Counters: `RSC_T_REFRESH_RECS`
  (non-empty records) and `RSC_T_REFRESH_RAYS`. Test-only define `RS_REFRESH_FORCE_MOVED`.
- **C-7 Unit builders** (affects T-B). `refreshFwdUnits(k, t, flags?)` / `refreshInvUnits(k, t, flags?)`: `flags` defaults
  to `k.currentAdvance?.flags` (A-10); with unknown flags the units are emitted and the entry points return early without
  `TF_REFRESH` (fwd also without `TF_HIST_VALID`). fwd: one unit per row band over `res[h]`; inv: per item chunk (one per
  row band, restir-api C7) `rs_args` with queue q2 + indirect `rs_refresh_inv` at args byte 32 over `res[w]`; not
  emitted under `TP_N1_MIXED`. Bind groups are resolved at build time. The per-pixel entry adds `RsDispatch.rowBase` to
  `gid.y` (the P0 stub did not).

### T-D amendments (debug views, interactive integration, boost)

- **D-1 Temporal debug data paths and probe formats** (affects T-B only through the §2.11 hook call sites, which
  `t-select.wgsl` / `t-classify.wgsl` already follow; nobody's interface changes). Views 480, 482–496 and the reservoir
  views 400–410 at tap `DBG_TAP_TEMPORAL` are written by `rsdbg_temporal` in T3 (phase A for pixels without
  `TS_INV_QUEUED`, phase B for the Q_i pixels; a phase-A call on a Q_i pixel is ignored): only T3 still has `res[h]`
  (stored p̂ of view 493, X_p's end technique for 496); spatial round 0 overwrites it (TD2), so `rs_debug_views` cannot
  provide them. View 481 comes from `rsdbg_tpick` (T1, `sp − local` with the A-7 convention, 0 without history or
  projection), view 497 from `rs_debug_views` (bit `s − firstBoost` of the accepted boost slots of the last round).
  Pinned definitions: 488 π_c = lum F of the temporal output record (= lum F_t(Y_p) for s = p, lum F_c for s = c and
  no q′, 0 for empty); 489 π_p = `piStored` for `TS_SEL_P`, `piRecomp` for other Q_i pixels, else 0; 491 c_out = c of
  the output record; 492 = 0 without q′, 3 `TS_EMPTY_OUT`, 2 `TS_SEL_P`, else 1; 482/483 = 0 unless `TF_REFRESH` and the
  record's `gen == frameGen`; 496: NEE ends — `SXS_UNDEF`, a stale record or a missing entry ⇒ bit 2, else the MOVED /
  RADIO bits of `lt_change_bits(entryTo)`; BSDF env ends — `TF_ENV_MOVED` / `TF_ENV_RADIO`; other ends 0. Background
  pixels of view 480 get code 0 from `rs_debug_fill` (T3 skips them). Probe tag 77's status word carries `dir << 16`
  (0 fwd = X_p at q′, 1 inv = the record at q) and bit 17 = "recorded by the refresh pass for the record stored at the
  probe pixel itself" (`rsdbg_refresh`); T3 records the fwd record of q′ and the inv record of q (bit 17 clear). New
  internal tag **79** (anchor ids `(bits(role | b << 8), prim/tag word, u, v)`, roles 0/1 X_p's x_k / surface endpoint,
  2/3 the inverse source's) is recorded by T3 (no scene group) and turned into tag-67 vertices by `rs_debug_views`:
  path 23 = T(X_p) into the probe pixel (y₁, x_k, endpoint), path 31 = the inverse into q′ (b = 0 the previous camera,
  y₁′ from the previous V-buffer, x_k, endpoint). `rs_debug_views` G2 gains binding 5 `rsVbufPrev` (a sampled texture:
  storage-buffer count unchanged) and includes `restir/tframe.wgsl`.
- **D-2 Interactive integration** (affects T-A: `kernel.ts` `RestirFramePass` gains `encodeHold()` (T-D edit, one
  method); T-E: none). `renderer.ts`: `restirMode` gains `'interactive'` (the interactive preset: temporal, RR, boost 3;
  the new default) and `'unbiased'` becomes the `full` preset (temporal, RR off, no boost), `options.temporal` switches
  temporal reuse in every mode; with temporal on each **advanced** frame calls `advanceInteractive(frameUniforms,
  { reset })` exactly once before the units are built, and a paused frame encodes no ReSTIR pass (TD20) but
  `encodeHold()` (rs_finalize_frame only: re-displays the last estimate without adding a sample). `app.ts`:
  `resetHistory()` now restarts only the progressive accumulation (light / env edits, animation, the render folder
  button); `resetTemporalHistory()` (Shift+R, the ReSTIR panel button) also resets the temporal history
  (`FrameContext.resetTemporal`), as do freeze seed / frame (TD20) and the new `render.freezeHistory`. "Freeze history" is
  defined as *temporal reuse suspended* (every frame a reset): a frozen history reused on later frames would be consumed
  twice (TD20, I1), which is biased. `prevCamera` is the camera of the previous **advanced** frame and equals `cam` only
  on the first frame and on temporal resets — before, every accumulation reset set `prevCam = cam`, which with a light
  animated together with a moving camera would have made q′ back-project with the wrong camera. A held paused frame
  keeps its debug AOV (hook `holdsFrameWhenPaused`, same view only) and the inspector keeps its last dump. The HUD's
  error totals restart only on temporal resets (light animation restarts the accumulation every frame).
- **D-3 Boost implementation** (affects T-E: `--boost` units; nobody's interface). `rs_pair_accept` reads the tState
  flag word through a mirror in `restir/pairing.wgsl` (`PAIR_TS_WORDS`, `PAIR_TSW_FLAGS`, `PAIR_TS_DISOCC`, checked
  against `layout.ts` by `tests/restir/debug-m5.test.ts`): it must not include `tframe.wgsl` (the RsTemporal uniform
  would make it a non-M4 module, `tests/restir/tframe.test.ts`). `stage-spatial.ts` sizes the replay chunks and cost
  hints by `numSlots` (slots + boost; a chunk sized by `slots` would leave boost items PENDING). On a frame of a temporal
  kernel without a temporal stage (not advanced) the flags are those of the last temporal frame (still G-buffer only,
  symmetric: unbiased); before the first temporal frame the zero-initialised arena gives no boost pair, so the
  interactive preset as shipped keeps the M4 goldens of U-M4-BITS. U8 plant 9 (`RSF_PLANT_U8_FAILED_K`, `mis.wgsl`):
  a partner whose G_j is not VALID is dropped from S_c (k, the 1/(k + 1) normaliser and c_out).

### T-B amendments (temporal passes, shift frame selector, queues)

- **B-1 `ShiftSrc.endVis` is stored inverted as `endOcc`** (affects nobody's call sites; T-C provides the N1 end
  visibility through `SfxRec.status` as before). WGSL zero-initialises `var s: ShiftSrc`, and the M4 test harnesses
  (restir-shift fixtures, the frozen t3fault copies) build sources field by field: a field `endVis` would default to
  false and turn every case (b) into `SC_OCCLUDED`. `endOcc` (false spatially, `tsrc_load`: `endOcc = SXS_N1 ∧ ¬SXS_VIS`)
  keeps every zero-initialised source bitwise M4. `ShiftDst.fs` defaults to `RS_FS_CUR = 0` for the same reason.
  `shift.wgsl` and `path/replay.wgsl` include `restir/tframe.wgsl` (the frame selectors); `replay_prefix` gains the
  trailing argument `fs` (call sites updated: shift, `restir-initial.gpu.test.ts` all-module smoke, the frozen legacy
  shift `t3fault-shift-a54ac2d.wgsl.txt`, each with `RS_FS_CUR`). U-M4-BITS, U-RIS-*, U-SFX-1 and T3 (M4) unchanged.
- **B-2 Queues** (affects T-C `rs_refresh_inv`, T-D boost units: none). `queue_append(q, item)` uses the per-queue item
  region and capacity (`tframe.wgsl queue_item_base / queue_capacity_q`) in modules compiled with `RS_TEMPORAL`, and the
  M4 body (q0 only) elsewhere, so no M4 pass changes. `rs_args` includes `tframe.wgsl` and takes
  `q = (RsDispatch.flags >> RSD_QUEUE_SHIFT) & 3`, capacity `queue_capacity_q(q)`, args at byte `16·q` (0 for every M4
  dispatch). Consumers read items at `arena_item_word(queue_item_base(q) + i)`. T2 / T4 (and T-C's refresh-inv
  builder) emit one item chunk per row band as M4 C7 (`treeBase / treeCount` = chunk base / size, 0 = whole queue).
- **B-3 T-B local tState flag `TS_ROBUST_IDMIS = 32768`** (affects T-D's flag decoding only; no word changes). T4 sets
  it in robust mode when the translated-back endpoint entry of Y_p (the SFX_INV `entryTo`, or the stored entry without a
  refresh) differs from `xpEntry` (or the inverse refresh marks the record undefined); T3 phase B counts it as
  `RSC_T_ROBUST_MISMATCH` together with the π comparison, so the ID check needs no extra tState word.
- **B-4 Write-back details** (affects T-C: status class bits). `res_select_temporal` lives in `passes/restir/t-select.wgsl`
  (scene-free, A-2). jDen ← `(J_p / J_P)·jDen_src` as §3.6. The endpointId renumbering is the analytic branch of
  `nee_endpoint_id_s` (entryTo < cur `nAnalytic` ⇒ endpointId = entryTo); emissive-triangle primIds and `RS_ENV_ID` are
  frame-independent and stay as copied (`endpoint.wgsl` cannot be included without the scene group). The refreshed deep
  `rcRad` is written only when the fwd refresh record carries `SXS_DEEP`, the (b)/(c) cache (`rcRad`, `aux`) only with
  `SXS_N1` / `SXS_B1` (else the copied values stay: a refresh that does not serve a class leaves it unchanged); the same
  status bits gate `tsrc_load` (deep `rcRad`, N1 `endOcc`). `TP_U8_STALE_AUX` keeps the copied `aux`.
- **B-5 Stale refresh records and counters** (affects T-C, T-D HUD). On a `TF_REFRESH` frame a record whose sfxOut
  `gen ≠ frameGen` or without `SXS_DONE` is treated as undefined (w̃_p = 0 / π_p = 0, unbiased) and counted in
  `RSC_T_PENDING_LEFT` (must stay 0), as is a T2 item still PENDING in phase A and a Q_i pixel phase B finds without
  `TS_INV_DONE`. Counter definitions: `LIGHT_UNDEF` = forward shifts refused by the refresh (`SXS_UNDEF`, incl. E2);
  `CLASS_UNDEF` = other undefined forward codes (O0–O3, J invalid) on `TF_REFRESH` frames (§9.3-6′ rate); `E2_ZEROED` =
  inverse evaluations whose SFX_INV record carries `SXS_E2` (tState `TS_E2_ZERO`); `NONFINITE` = non-finite or negative
  w̃ or W_Y (the record becomes empty with its c). The queued forward code is `SC_PENDING` until T2 writes it.
- **B-6 Row bands in the temporal per-pixel passes**: T1 and both T3 phases take `rs_pix(gid.x, gid.y + rowBase)` like
  every M4 per-pixel pass (the P0 stubs used `gid.xy`).

### T-E amendments (validation harness, gate-m5)

- **E-1 Chain runner API and readback** (§3.8, §6.4; affects nobody's call sites: T-B / T-D may reuse it). `ChainRunner`
  runs one BATCH of E chains: `new ChainRunner(kernel, { runSeed, budget? })`, `runBatch(spec, b, chainBase,
  onFrame(rows, chainIds))` sets `memberBase = chainBase + b·E` (setView, same allocation) and returns per-frame records
  (`histValid`, `flags`, `reasons`, temporal units). `ChainSpec` gains `beforeFrame(t)` (env map swaps through
  `setEnvironment`, T-A A-8) and `average {from, to}` (rung 3.5); `masks(t)` returns `{ names, bits: Uint16Array }`
  (bit i ⇔ region i, ≤ 16 overlapping regions: the dominance regions overlap the partition). Test frames are reduced on
  the host: a harness unit `chain_copy[t]` copies the atlas `rsFrame` (the linear L of every member) into a MAP_READ slot
  and the host reduces each member image in f64 into the rows of `ensemble.npz` (tiles 16/32/64, global, masks, per-pixel
  Σx/Σx² over chains); rung 3.5 adds `rsFrame` into an atlas accumulator (`chain_accum[t]`, a standalone 8×8 pass) over
  [from, to]. Reason: the §2.10 mask texture is one region id per pixel (no overlap), `ensPixel` accumulates over all
  frames of a batch (not per test frame) and `ensStats` has no mask reduction yet (M = 0 in M4); the host reduction needs
  no shader change. The ensemble stage still runs (unused by chains). TD26: the unit packer is drained at every frame end.
  Output per test frame `f<t>/{ensemble.npz, meta.json}` (rows = chains, seeds `runSeed:c<chain>`); the upload middleware
  takes single path components, so the page uploads `f<t>__<file>` and run-batches.ts moves them into `f<t>/`.
- **E-2 Drift test level** (§6.4 "drift regression … slope within its 99% CI of 0; gating"). The slope of the relative
  global Δ_f on f is estimated by OLS with a SANDWICH variance: the test frames of one unit share the same chains, so their
  Δ_f are correlated (covariance across chains; PT references of different frames independent). The gate uses
  |z| ≤ z_{1−α_u/2} (α_u = suite FWER unit level: PLAN §7.3 applies α_u to every rejection-type check, and PLAN outranks
  this contract's choice); the 99 % CI is reported. The failing-tile count per frame is tested against Binomial(m, 0.01)
  at α_u, as specified.
- **E-3 Mask regions below 256 px are dropped** (dyn_masks.py, §6.4): a region smaller than one 16² tile needs orders of
  magnitude more chains for its 3 % TOST than the 64² tiles that already cover it (and compare.py refuses empty masks).
  Dropped regions are listed in masks.json. Masks come from frozen PT means of fixed-size mask references (512 spp × 4,
  seed 7301) at t and t−1, independent of the comparison references (no selection on the reference noise).
- **E-4 Rung 3.5 chain floor 64** (§6.4 floor R ≥ 256 is for per-frame ensembles; gap-temporal §9.1): a 3.5 replicate is
  the mean of 256 frames of one chain, near-normal; the §6.6 estimate (64–128 chains) assumes this. Per-frame units keep 256.
- **E-5 Plant detection with chain rows** (§6.5 "as M4"). M4's rendered-plant rule compares half-size sets
  (`h = min(ref.n/2, planted.n)`), i.e. 8 of the R planted chains against 8 of 16 PT batches: meaningless when a replicate
  is one chain. plant_sign.py compares a random HALF of the planted chains with a half of the 4× PT reference (must fail
  in ≥ 9/10 repeats) and the two PT halves (control, must pass in ≥ 9/10). The synthetic W × 1.003 and the calibrate A/A
  re-splits on the A/A chain run use `dynamic.py calibrate` (compare.py --calibrate needs replicate images; the chain side
  is ensemble rows): 20 re-splits (stats.aa_split) and halves A × 1.003 vs B.
- **E-6 Plant regions that do not exist at the predicted frame.** (a) N3 at frame 80: the light stops at 40, so M_new /
  M_gone (t = 80 vs 79) are empty; (b) N5 on ixs_d0_jitter (static camera): M_edge (a t−1 → t change) is empty. A
  prediction whose region is empty is recorded "not evaluable" (never passed silently: every plant needs ≥ 1 evaluated
  prediction, and N3's frame-40 predictions remain). For N5-d0 the region is `M_sil` (dyn_masks.py --sil: pixels whose
  luminance differs by ≥ 25 % from a 4-neighbour in the PT mean at t, dilated 1 px — the silhouettes and shading edges
  where a pixel-centre primary differs from the jittered one). N5 on ixs_d_camera keeps M_edge.
- **E-7 Dominance renders** (§6.4 "our PT with a single emitter enabled"): gate-m5 derives single-emitter copies of the
  sequence package (every other light `enabled: false` in every frame; env strength 0 unless the emitter is the env, then
  every analytic light disabled) and renders them with the unchanged PT (no emitter filter in run-batches). Light names
  come from scene.json `lightNames` written by make-m5.ts (ixs_e: A/B/C; ixs_n4: A/B/C; ixs_i: R).
- **E-8 Static references** (§6.4): a static unit's scene state is the same at every frame, so one base-state PT reference
  (no frame override) per m5s/u8 package serves both test frames of 3.3 and 3.4 and rung 3.5 (joint sizing over the three
  rungs, as M4's per-scene reference).
- **E-9 Sequence scripts** (§6.2, make-m5.ts). The Cornell box has no blocks, so ixs_a…f add one white occluder box
  (x ∈ [−0.02, 0.10], z ∈ [−0.16, −0.04], h 0.2) for shadows and disocclusions; ixs_b resizes the rect with the frame
  fields `sizeX`/`sizeY` (T-A A-8) and spins it 5°/frame about its normal (it stays facing down); ixs_k swaps to the
  overcast map box-filtered 2× (map id `b`, a different texture and importance table; the asset set has no second
  sun-free HDRI); ixs_h is a floor + wall + a Lambert and a GGX r 0.3 sphere under the overcast map. If a plant's region
  proves empty in the pilot (E-6), the script is revisited, not the prediction.
- **E-10 U8 rungs 3.1/3.1b/3.2** run through the M4 sequential harness (run-batches --kernel restir) and are sized by
  gate-m4's `sizeScene` (M4 pilots 128 / 8 frames × 16) around the u8 package's frozen base-state PT reference, which
  they share with the package's 3.3 / 3.4 chain units.
- **E-11 Grouped GPU-lock holds for small renders** (§6.6 control 7; affects nobody: run-batches.ts, T-E may touch).
  `run-batches.ts --jobs FILE` runs a JSON list of ordinary invocations on one harness page under ONE lock hold.
  gate-m5 collects the small cacheable renders of a phase in a dry pass (mask references at t and t−1, the
  disocclusion runs, the dominance renders, the PT and chain pilots, every PT reference that fits one hold), runs them
  in groups of ≤ 10 min of estimated GPU work, then runs the phase for real on the caches. With the main checkout's
  gates holding the lock for hours, one lock wait per group instead of per render is the difference between hours and
  days of wall time; each hold stays ≤ 12 min.
- **B-7 T3-2 harness hook instead of `RS_TEMPORAL_TRACE`** (affects nobody). The composer has no `#define` and the kernel
  passes extra defines only to `rs_initial`, so the T3-2 / T4-t harness does not use a define: `t-select.wgsl` has
  `const TSEL_TRACE_FORCE_P: bool = false`, which the test flips to `true` through `instrumentation.extraSources` (every
  defined forward shift is then selected, robust mode runs T⁻¹(T(X_p)) through the production T1/T2/T3/refresh/T4
  passes on every pixel with a valid q′). Frames alternate reset / test so each test frame round-trips fresh canonical
  samples. Classification as M4 B-4 (FP-BOUNDARY iff the deciding pair's |margin| < 2⁻¹⁶; the replay-divergence rule
  needs the path trace and is not applied: such events count as LOGIC and are investigated).
- **B-8 U8 plants in the shift** (TD30; affects T-E's plant units only). `RSF_PLANT_U8_W1DELTA`: ω1 := 1/(1 + p2) for
  delta lights in every shift; `RSF_PLANT_U8_NO_PK`: J_rc without the p^y_k factor (cases (c) and deep);
  `RSF_PLANT_U8_ONESIDED`: back-side rect / disk samples at the offset emit with |cos| (Λ, q, p1 recomputed);
  `TP_U8_STALE_AUX`: cases (b)/(c) use the stored `aux` as p1 in ω1 / ω2 and the write-back keeps the copied `aux`;
  `TP_U8_SPOT_PREV_AXIS`: on history frames with a light change, CUR-frame shifts evaluate the spot profile with the
  spot axis of frame t−1. `RSF_PLANT_U8_T2` (J = t_x²/t_y² for point lights) is **not implemented yet**: the shift source
  carries no x_{d−1} for case (a) (see the T-B report / open issues). **Lesson (U-M4-BITS):** a plant written as a
  branch that reassigns an arithmetic result on the default path (`jNum = …; if (plant) { jNum = … }`) changed the Metal
  code of the spatial shift with the plant off (xq-3.2 / i-3.2 reservoirs differed from the M4 goldens); plants must
  enter as `select` on an input factor (`pKj = select(…, 1.0, plant)`) or as a separate call whose result replaces a
  value only under the uniform, and every plant edit re-runs U-M4-BITS.
- **D-4 Interactive colour view cached** (affects T-A: `kernel.ts`, one method; nobody's interface). `frameUnits()`
  created a new view of the interactive colour target every frame, so the finalize bind group got a new cache key (and
  a new entry in `RestirResources`' group cache) every frame, without bound (an M4 leak). `RestirKernel.colourView(tex)`
  keeps one view per colour texture (WeakMap); `frameUnits()` and `RestirFramePass.encodeHold()` use it. A new target
  is a new texture (resize, format change) and `setTargets` still drops the old finalize groups. Test: restir-debug
  "finalize bind-group cache stays bounded" (24 advanced + held frames, constant group count).
