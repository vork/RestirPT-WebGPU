# perf2 optimisation plan

Current integration status and measured results are in [§7](#7-fixed-resolution-follow-up-2026-10-10), following the recovered checkpoint in [§6](#6-recovered-second-wave-2026-10-10). The earlier estimates below are historical.

At the original planning checkpoint, no perf2 source had been edited. Five prototype diffs were exported in `/private/tmp/claude-502/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-ea73-491f-b7b7-6d0aab1ffce1/scratchpad/plan-patches/`:
- `vis-merge-shift.patch`
- `ris-hoist.patch`
- `ris-hoist-skip-needraw.patch`
- `ris-prepass.patch`
- `bvh-private-stack.patch` — port only the BVH2 half; the CWBVH half is a +10 ms regression.

The scratchpad is session-temporary, so copy these into the package worktrees early.

## 0. Ground rules for every package

1. **How the evidence is weighted.**
   - Six results are measured or trace-backed and carry the plan: the shift visibility merge, the RIS hoist, the RIS pre-pass, rs_primary reusing the M1 primary, dupmap s64, and the dn_gradient skip. Together they are worth about 11 ms on Sponza.
   - Every unmeasured estimate is assumed to realise 50–70%. The adversarial reviews cut most proposals by 2–6×.
   - Session drift is up to 40%, so every keep/drop decision uses same-session ABBA.
2. **Gating.**
   - Every change ships behind an interactive-only composer define, set by `RestirKernel.interactive` (precedent: P-4 and `RS_RES_SOA`).
   - Validation text stays byte-identical, so U-M7-BITS holds by construction.
   - Moving any define into validation text is a separate, optional step. It needs PTBITS, the validation cases of U-M8-BITS, a re-recorded U-M7-BITS and a Stage-B smoke run.
   - Reason: under relaxed math, even semantically identical restructures have changed bits. The pv stack changed PTBITS; the shift live-state lever broke nm-interactive-B.
3. **Two tiers of goldens.**
   - **Anchor:** every results-changing flag is off. These must stay bit-equal to today's goldens forever.
   - **Shipped:** all release flags are on. They are re-recorded only when a results-change item lands, with a written justification and a diagnostic.
4. **Keep/drop rule.**
   - Keep a bitwise item if it shows a gain beyond noise in ABBA, or at least 2% on the affected pass in split-phase Metal System Trace sums, and spill does not grow.
   - If a bitwise-class item unexpectedly fails U-M8-BITS and its gain is under 0.5 ms, drop it rather than re-validate.
5. **Codegen checklist.** Each package applies these to its own files:
   - Use constant upper bounds on loops (`for i < CONST { if i >= n break }`). This removes the `tint_loop_idx` guards.
   - No `return` inside `switch` (removes the `tint_volatile_zero` guards).
   - No non-constant integer `%` or `/` on hot paths.
   - Never use a module-private traversal stack on CWBVH. The separately measured workgroup-memory stack is described in §7.
   - Fewer inlined `bvh_trace` call sites. This turned out to be the biggest lever: each copy carries its own stack and code.

## 1. Ranked candidates

Gains are revised central values in ms, with ranges, at 540p N=3. "meas" means measured.

| # | Item (merged) | Class | Sponza | Cornell | Basis | Effort | WP |
|---|---|---|---|---|---|---|---|
| 1 | One `trace_any_ex` call site in `shift_finish_vis`: forced-inf, forced-local, ENV and reconnect cases merged via selects, endOcc not traced. Affects spatial shift, t_classify, t_forward, t_inverse and replay | bitwise | **4.2 meas** (shift −2.9, classify −1.0, inverse −0.3) | 0.5 (0.3–0.7) | 2 independent prototypes; U-M8-BITS 21/21 (BVH2 only) | S | WP-1 |
| 2 | Hoist `bsdf_prepare` out of the 32-candidate RIS loop and skip the dead `nee_draw` at B=1 | bitwise | **2.4 meas** | **0.6 meas** | 2 prototypes; U-M8-BITS + PTBITS 17/17 | S | WP-2a |
| 3 | Lean RIS pre-pass `rs_ris_nee` (interactive, trees=1). The record goes in the pixel's own dead RP_DIAG plane | unbiased, results change (W last-bit) | **+2.2 on top of #2** (1.5–2.5; 4.2 standalone meas) | +0.6 (1.24 standalone meas) | prototype `ris-prepass.patch` | M | WP-2d |
| 4 | Fewer traversal copies in rs_initial: one NEE site (visible + visibleInf + the k=B retest) and one post-hit retest site | bitwise | 1.2 (0.5–2) | 0.1 | analogue of #1 | S–M | WP-2b |
| 5 | rs_primary reads the M1 V-buffer (`bits(t)` in vbuf.w) instead of tracing again | unbiased, results change (ulp edge ties) | 1.2 (1.1–1.4) | 0.03 | GPU trace | M | WP-7e |
| 6 | Dense spatial slot-item queue, boost-slot gating, write-back trimming | bitwise | 1.5 (1–2.5) | 0.7 (0.5–1.2) | CPU simulation and estimate | M | WP-5 |
| 7 | Traversal: CWBVH K=2 triangle budget; alpha bit in the MT record; compile CUSTOM_ALPHA out when no MASK exists; constant loop bounds; module-private stack for BVH2 only | bitwise per ray | 1.0 (0.5–2) | 0.35 (BVH2 stack meas −0.53 before #1) | partly measured | S–M | WP-3 |
| 8 | Env-presampled light tiles inside the pre-pass (alternative: workgroup-uniform candidate ids) | unbiased, adds correlation | 1.2 (0.5–2) after #3 | 0 | estimate | M | WP-4b |
| 9 | dn_gradient skip, dead M1/finalize outputs, lossless 48 B G-buffer, à-trous zgrad hoist | bitwise | 0.8 (0.6–1.0) | 0.7 (0.5–0.8) | dn part from trace | S | WP-7 |
| 10 | Lean interactive text: debug-free variant while no view/probe is active, RS_NO_PLANTS, RS_NO_DIAG, single-chunk register accumulation | bitwise (P9 excluded) | 0.8 (0.3–1.5) | 0.4 | estimate; post-CWBVH P-4 precedent 0.7 ms | M | WP-9, WP-2c |
| 11 | Temporal: T4 split into non-replay and replay pipelines, res[w] read-only, phase B folded into T4, tState SoA, subgroup counter aggregation | bitwise | 0.8 (0.4–1.5) | 0.7 (0.4–1.0) | estimate | M | WP-6 |
| 12 | Material compile variants, TEX_XFORM, branch-free LUT lerp, no return in switch | bitwise | 0.7 (0.3–1.2) | 0.5 (0.3–0.7) | estimate | M | WP-8 |
| 13 | `envTexel` select-wrap, replacing 6 emulated i32 modulos | bitwise | 0.6 (0.3–1) | 0 | estimate | S | WP-4a |
| 14 | dupmap branch-free 64-bit compare (`s64`, memberCount==1) | bitwise | **0.55** microbench | **0.3** | microbenchmark | S | WP-7b |
| 15 | Last continuation as any-hit (Mode-A text, no emissive tris), escape env CSE, lazy NEE hashes | bitwise image (counter exceptions) | 0.4 (0.2–0.8) | 0 | estimate | S | WP-2c |
| 16 | Pairing transform table with exact fast modulo; single-member fast path in rs_pix | bitwise | 0.3 | 0.3 | estimate | M | WP-5 |
| 17 | Mode-B single emission site, area-light index list, vec4u light records | bitwise | 0 | 0.4 (crossings 0.5–1) | estimate | M | WP-2e + WP-4a |
| 18 | CWBVH tail: leaf-1 DP input, exponent-OR plane decode, CWBVH_AUTO_MIN_TRIS re-sweep | results change (ties only) | 0.4 (0–0.8) | 0 | estimate | S | WP-3 |
| 19 | App-only: tonemap once in present, dynres without reallocation, repeated-timing-submit bug while paused, early denoiser prepare | app-only | 0 at gate | 0 at gate | – | S–M | WP-7g |
| – | Enablers: guardrails, flags, spill report, quality harness | – | 0 | 0 | – | S | WP-0, WP-Q |

The first-week quick-win set is #1, #2, #9 (dn_gradient part) and #14. All are bitwise, measured and effort S, worth about 7.5 ms on Sponza and 1.6 ms on Cornell.

### Dropped or not recommended
- **Module-scope `var<private>` traversal stack on CWBVH:** measured +10.1 ms (+13%) and broke PTBITS. Only the BVH2 variant is kept, in #7.
- **Workgroup traversal stack:** slower on both scenes.
- **Wavefront path-tree split:** the compiled per-bounce pass spills more (1280 vs 1104 B, still 126 GPRs). About 1 ms for XL effort.
- **Persistent threads / dead-lane compaction:** Sponza paths are 95–98% alive at B=2–4, so about 0.5 ms net.
- **Lean any-hit kernel outside rs_initial; deferred dense shift-visibility pass:** superseded by #1 and #4. The prototype gave 3.3 ms standalone but about 0 on top of the merged call site, and costs 100 MB of arena.
- **Live-state diets:**
  - Shift: measured 0.26 ms and broke nm-interactive-B bits.
  - rs_initial: about 0.5 ms with an FMA risk from duplicating pos/ng.
  - Dropping `treeMargin`: not bitwise.
- **Confidence-gated slots:** wrong. c_p saturates within one frame, so this collapses to `slots = s0`.
- **Static-camera temporal visibility reuse:** the gate never fires (`camSame` is hard-coded false), tap 0 does not imply q′ = q, and the bias chains across frames.
- **Unshadowed NEE target, NEE roulette at B≥3, two-stage RIS, cheaper RIS target, cell-mean env radiance:** each is ≤1.2 ms with a variance cost, or superseded by #2, #3, #8 or #13.
- **Texture LOD at b≥2, ray-cone LOD, BC compression, material LOD at B≥3, metallic snap:** texel latency is not the sink. The 64² best-case probe gave at most −2.2 ms on rs_initial, and the "no textures" variant was confounded by metallic/roughness factors of 1.0.
- **y1 MatEval cache:** about +0.8 ms on Sponza, −0.1 ms on Cornell, and 100–177 MB of f32 against the data-format rule.
- **CWBVH f16 node planes:** would need a data-format exception for about 0.7 ms. Replaced by the exponent-OR variant in #18.
- **Ylitie postponing:** not additive with K=2 and puts subgroup ops in the traversal loop (Q2 risk). Revisit only if K=2 fails.
- **SBVH, BVH4, last-occluder cache, BVH2 for primary passes:** unproven at L effort, or superseded by #5. Emscripten is only useful for an optional half-day tinybvh SBVH probe through the CPU proxy. tinybvh's clipping is non-conservative, so it is not a production builder.
- **`maxTotalThreadsPerThreadgroup` profiling:** reads 1024 for every kernel on M5. Replaced by the spill-byte report.
- **TAAU:** a quality feature worth 0 ms by itself; the speed lever is the existing dynres.
- **Pipeline prewarm:** 0 ms. Only the denoiser-prepare one-liner is kept, in #19.
- **Fewer compute passes, persistent-grid queue consumers, dn_variance fusion:** Dawn already merges consecutive compute passes into one encoder and the GPU is 96% busy. Each is ≤0.1–0.3 ms.
- **Refresh skip, duplicate-pair shortcut, x_k hoist:** parked. The refresh skip's bitwise status is unproven and it only helps N1+moving. The duplicate-pair shortcut first needs a rate counter. The x_k hoist conflicts with #6.
- **Rasterised primary:** incompatible with per-pixel IID jitter.
- **`metal_use_argument_buffers`:** produced broken results.
- **`disable_robustness`:** cannot ship. Its measured 3.05 ms is the ceiling for clamp-avoiding rewrites.

## 2. Work packages

### Standard verification sets (referenced below)
- **V-BIT:**
  1. U-M7-BITS: validation text unchanged; assert the flag key is absent from every validation define set.
  2. U-M8-BITS (AoS and SoA), U-M8-MODEB, P-4 and U-M8-PTBITS with the flag on, equal to the Anchor goldens. Add the WP-0 CWBVH and Sponza-lite cases whenever CWBVH, texture or normal-map code is touched.
  3. U-M4-BITS and U-M5-BITS interactive cases, with knobs pinned.
  4. The bits rig twice with identical hashes (Q2), plus the M4/M5/M5.5/M7 app smokes.
  5. Spill bytes and MSL lint counts per pipeline: no growth in hot kernels.
- **V-SHIFT:** the T3 discriminator and the VITE_STRESS production stress with the flag forced on: RSC_PENDING_LEFT = 0, RSC_SLOT_MISMATCH = 0, and two runs with bit-identical counters. Also the T3-0, T3-1 and T3-D round trips with the flag on.
- **V-UNB:**
  - V-BIT items 1, 3, 4 and 5; Anchor goldens with the flag off unchanged.
  - Shipped goldens re-recorded with a diagnostic.
  - Stage-B units with the flag forced on: full-m6 with trees=1 in Mode A and Mode B, plus an HDRI package (m7_nm_env_256), at the unchanged δ 0.2 % / 1 %.
  - The relevant plants still detected.
- **V-Q:** the WP-Q equal-quality protocol.
- **V-PERF:**
  - Same-session ABBA, at least 2 blocks of 8 jobs, under the GPU lock using WP-0 flags, with background GPU load logged.
  - No xctrace recording during timing runs, and no timestampWrites around ReSTIR passes (Q3).
  - Frame time plus per-pass times normalised by untouched passes. For effects under 1 ms, use split-phase Metal System Trace sums.
  - Scenes: Sponza, Cornell, crossings and nm_smooth at 540p N3 and N1+moving; 720p once per package.

### WP-0 Guardrails and tooling (prerequisite, 2–3 days, no shader changes)
- **Steps:**
  1. **Perf-flag registry.** Add `RestirKernelOptions.perfFlags`, mapped to defines and included in `variantKey()`. Plumb it through `RendererOptions`, `restirRig`, `perf-run.ts` (per job) and the `--kernel-flags` option of `restir-chain-run.ts` and `restir-batch-run.ts`. The T3 fixtures (`restir-shift-fixtures.ts`, `restir-temporal-fixtures.ts`) and VITE_STRESS must be able to force flags on. Reserve all flag names now so packages do not collide in `kernel.ts`: RS_VIS_MERGE, RS_RIS_HOIST, RS_NEE_SITE, RS_LAST_ANYHIT, RS_RIS_PREPASS, RS_ENV_WRAP, RS_ENV_PRESAMPLE, CW_TRI_BUDGET, BVH_ALPHA_BIT, BVH2_PRIV_STACK, RS_DENSE_SLOTS, RS_BOOST_GATE, RS_TSEL_FOLD, RS_TSTATE_SOA, RS_AGG_COUNTERS, RS_NO_PLANTS, RS_NO_DIAG, MAT_VARIANTS, RS_PRIMARY_EXT.
  2. **Pin knobs explicitly** (slots 3, risM 32, rrMinBounces 3, dupmap true, maxBounces) in:
     - `m8-bits.ts`;
     - `restir-tframe-bits.ts` (i-interactive, xq-interactive);
     - `restir-m6-bits.ts`;
     - the N3/N1 jobs in `run-perf.ts` and `gate-m8.ts`.
     Add separately labelled rows for decision configurations.
  3. **New bits cases**, with goldens recorded on unmodified text:
     - CWBVH-forced variants of all-interactive-B, nm-interactive-B and alpha-full-m6-A;
     - a Sponza-lite case: real Sponza with the HDRI at about 160×90, CWBVH, textures, normal maps and alpha, interactive, 3 frames;
     - a P9-excluded hash mode.
     Then split the goldens into the Anchor and Shipped tiers.
  4. **Promote the profiling tools** into a tracked `validation/tools/`: `prof-driver.ts`, `xct-export.sh`, `xct_split.py`, `xctab.py`, `abba.py`, `split-dump.py`, `gpu-bg.py`. Add a report-only "spill bytes per pipeline + MSL lint" command.
- **Files:**
  - `src/core/render/restir/kernel.ts` (registry only), `src/core/render/renderer.ts` (plumbing);
  - `validation/harness/{run-perf,perf-run,gate-m8,restir-chain-run,restir-batch-run}.ts`;
  - `validation/gpu-tests/{m8-bits.ts, m8-bits.gpu.test.ts, restir-tframe-bits.ts, restir-m6-bits.ts, restir-shift-fixtures.ts, restir-temporal-fixtures.ts}`;
  - `tests/scene/m7-wgsl-bits.test.ts`;
  - `validation/tools/*`.
- **Verification:** every existing gate stays green with identical bits. Sanity-check the new goldens by running them twice.
- **Delivered (WP-0):** the interface is [perf2-api.md](perf2-api.md): the flag registry and how a package adds its flag (§1), the pinned knobs and the `INTERACTIVE_APP_DEFAULTS` hook for D1–D4 (§2), the Anchor / Shipped golden tiers and the new cases (§3), and the tools in `validation/tools/perf/` with the `spill-lint.ts` command (§4).

### WP-Q Equal-quality harness (parallel with everything, 2–3 days plus overnight compute)
- **Steps:**
  1. Export the perf Sponza setup as a package with `export-package.ts`.
  2. Add `restir: Partial<RestirSettings>` and renderer overrides to `denoise-run.ts` / `run-denoise` (`--restir JSON`), plus perfFlags.
  3. Add a `relmse` subcommand to `denoise_eval.py`.
  4. Render the PT reference through `run-batches` at 960×540 with maxBounces 3, until the noise floor is ≤10% of the smallest configuration error.
  5. Run 4 seeds × frames {16, 32, 48, 63} for static, pan and light-animation sequences.
  6. Collect metrics: raw relMSE, LDR/HDR FLIP raw and denoised, `edge_study.py` std/dt/rmse/d2, TD-I1 spike events, and 8×8 block autocorrelation of the raw error.
  7. Decision rule:
     - Equal quality means denoised FLIP and temporal std stay inside the baseline's seed-to-seed 95% CI in all three sequences, and raw relMSE × ms does not get worse.
     - Otherwise, present an ms-vs-denoised-FLIP Pareto plot.
- **Files:** `validation/harness/denoise-run.ts`, `run-denoise`, `validation/tools/denoise_eval.py`, `edge_study.py`. No `src` overlap.
- **Needed by:** WP-4b and WP-10.

### WP-1 Shift visibility, single call site (S, 2–3 days)
- **Items:** #1, plus an optional merge of the per-class visibility calls in `refresh.wgsl` (separate flag, N1+moving only, estimate 0.3–0.6 ms, needs a `refresh_record` restructure).
- **Files:** `src/core/shaders/restir/shift.wgsl` (`shift_finish_vis` under `#if RS_VIS_MERGE`); `src/core/shaders/geom/visible.wgsl` (new `vis_ray` helper under `#if`); optionally `restir/refresh.wgsl`.
- **Gain:** Sponza 4.2 ms measured. Cornell about 0.5, crossings about 1.3, nm_smooth about 1.2 (estimates).
- **Class:** bitwise.
- **Steps:**
  1. Add `vis_ray(...)` to `visible.wgsl`, reproducing the `visible` / `visibleInf` arithmetic exactly. On env lanes the finite values are computed and discarded, which is harmless. Land this first, because WP-2b needs it.
  2. Port `vis-merge-shift.patch` behind the define.
  3. Verify.
  4. Optionally do the refresh merge.
- **Verification:**
  - V-BIT including the CWBVH and Sponza-lite cases. The prototype was only checked on BVH2 fixtures.
  - V-SHIFT with the full Q2 protocol: 6 reps × 5000 frames, forced bins, and the stress on both packages. Q2 previously hit exactly the forced-NEE visibility path.
  - Expected spill: shift 800 → about 528 B, classify 736 → about 464 B, inverse 1184 → about 928 B.
  - V-PERF.
- **Dependencies:** WP-0. No conflicts: `shift.wgsl` belongs to WP-1 until the WP-9b sweep.
- **Delivered (WP-1):** two bitwise flags, both `release: true`.
  - `vis_ray` / `vis_trace` (`visible.wgsl`, gate `RS_VIS_MERGE || RS_REFRESH_VIS`; WP-2b adds `|| RS_NEE_SITE`):
    `vis_ray(a, na, b, nb, primB, dir, inf)` builds the `visible()` (inf = false) or `visibleInf()` (inf = true) ray with
    identical operands; `vis_trace(r, primA)` traces it (true = unoccluded).
  - `RS_VIS_MERGE`: `shift_finish_vis` traces once (the prototype patch, through `vis_ray`). bvh copies 4→1 (shift,
    classify), 5→2 (replay, forward), 6→3 (inverse). Spill Sponza / Cornell (B): shift 800→528 / 1072→672, classify
    736→448 / 992→592, inverse 1184→784 / 1360→992, forward 1056→640 / 1232→848, replay 1072→656 / 1232→848.
  - `RS_REFRESH_VIS` (the optional refresh merge, new registry name): the class functions return their pending ray
    (`RfOut.rayMode/ray/rayPrim`) and `refresh_record` traces it once. bvh copies 4→1 (Sponza) / 6→1 (Cornell); spill
    refresh_fwd 560→288 / 1056→384, refresh_inv 272→0 / 784→144; zero_init +1 (the pending record).
  - Gains (same-session ABBA, 2 blocks, shared GPU): 540p N3 Sponza −5.6 ms (passes: shift −3.1, classify −1.0,
    inverse −0.3), Cornell −0.9, crossings −1.6, nm_smooth −0.9; 720p N3 Sponza −9.9, Cornell −1.5. RS_REFRESH_VIS at
    540p N1+moving: refresh passes Sponza −0.8 ms (harness pass times) / −0.5 ms (−17 %, split-phase MST sums, ABBA),
    nm_smooth −0.13 / −0.10 ms (−12 %); Cornell and crossings within noise; frame time within noise.

### WP-2 rs_initial and RIS (one engineer, sequential steps, about 3 weeks)
- **Items:** #2, #4, #15 plus single-chunk accumulation, #3, #17 (Mode-B part), and constant loop bounds in its files.
- **Files:**
  - `restir/ris-nee.wgsl`;
  - `material/bsdf.wgsl` — one new helper only, after which `bsdf.wgsl` is handed to WP-8;
  - `path/{pathtree.wgsl, replay.wgsl, path-weight.wgsl, crossings.wgsl}`;
  - `restir/endpoint.wgsl`;
  - `passes/restir/initial.wgsl` (the RS_RIS_PREPASS entry point);
  - `render/restir/{kernel.ts, resources.ts}` (pass schedule and pass entry).
- **Gain:** Sponza about 6 (5–7.5); Cornell about 1.5 (1.2–1.9).
- **Class:** bitwise, except 2d, which is an unbiased results change.
- **Steps:**
  - **2a (S, measured −2.4 / −0.6). RS_RIS_HOIST.**
    - Build the context once, after the `nEntries == 0` early return, and evaluate `bsdf_query_c` / `bsdf_f_all_ctx` per candidate. Skip `nee_draw` at B=1; `rs_path_hash` is stateless, so this changes no other random draw.
    - Add the new function next to `bsdf_query` and leave `bsdf_query`'s body untouched. That keeps every other kernel's machine code byte-identical (the wrapper form changed rs_t_classify's bytes).
    - Do not move `material_eval(x1)` or the RIS call in front of `for B`: measured +5% and spill 1152 → 1296 B (the Q1 shape).
    - Do not reuse material_eval's flags context: `m.nsm` is set after `bsdf_flags`, so bump shadowing would silently pass.
    - Extend the U-PT-BITS(3) inline-budget count to cover the new entry.
  - **2b (S–M). RS_NEE_SITE, built on WP-1's `vis_ray`.**
    - `nee_visible` (visible plus visibleInf) and the k=B retest share one site, as a loop of at most 2 iterations.
    - The tree-pair retest and the end retest after the hit share one site.
    - Target: from 6 inlined `bvh_trace` copies down to 3; check with the spill report.
    - Optional follow-up: a single runtime-`any_hit` site. Higher Q2 risk; only if 2b pays.
  - **2c (S).**
    - Single-chunk register accumulation (S=1, FIRST|FINAL): no empty 10-plane write and no read-backs.
    - RS_LAST_ANYHIT, only for Mode-A text with no TRI_EMISSIVE.
    - Escape CSE: compute envUV, the cell pdf and p1Env once.
    - Lazy SEL2/L0–L2 hashes.
  - **2d (M). RS_RIS_PREPASS.**
    - Only when interactive with trees==1, E==1, and not the dump, RS_PT_DIRECTIONS or test texts. Port `ris-prepass.patch` with the hoist from 2a applied inside it.
    - Schedule: rs_primary → rs_light_tiles → rs_ris_nee → rs_initial.
    - Store `(j, W bits, mult)` in the pixel's own RP_DIAG plane, which is dead until `res_write_empty`, so no new memory is needed. rs_initial recomputes ep from j with one pcg4d, one tile word and `nee_draw_entry`.
    - Always run the loop; never gate it on the pre-pass's MatEval flags.
    - Keep RP_DIAG allocated even under WP-9's RS_NO_DIAG.
  - **2e (M).** Mode-B single candidate-emission site, keeping the same candidate order and counters, and consuming WP-4a's rect/disk index list.
- **Verification:**
  - 2a–2c: V-BIT with the CWBVH and Sponza-lite cases; U3-RIS; `rs_initial_dump` still compiles and U-RIS-1 still passes (dump keeps inline text); 2c's counter-hash exceptions (RSC_BVH_*, BASE_JDEN) written down.
  - 2d: V-UNB. The diagnostic must show `(entry, a, b)` 100% identical and W within 1e-5 relative. Run Stage-B rung units 3.9–3.11 with the flag forced on in Mode A, Mode B and HDRI; U3-RIS at M ∈ {1, 4, 16, 32}; U8-8 and U8-10 detected (U8-10 through the stored mult).
  - 2e: V-BIT plus the U8-7 plant.
  - All steps: V-PERF.
  - rs_initial has no stress harness, so use repeated-run bit identity (U-M8-BITS ×3) as the Q2 check.
- **Dependencies and conflicts:**
  - Needs WP-0; 2b needs WP-1 step 1; 2e needs WP-4a.
  - `bsdf.wgsl` goes to WP-8 after 2a.
  - `kernel.ts` / `resources.ts`: additive pass entries; merge serially with WP-5, WP-6 and WP-7.
  - `ris-nee.wgsl` goes to WP-4b after 2d.

### WP-3 Traversal and BVH (1–2 weeks)
- **Items:** #7 and #18.
- **Files:**
  - `bvh/{traverse-cwbvh.wgsl, traverse.wgsl}`;
  - `src/core/bvh/{cwbvh.ts, sah-builder.ts}`;
  - `scene/scene-data.wgsl` (`alpha_pass`);
  - `render/scene-gpu.ts` (CUSTOM_ALPHA predicate);
  - `render/renderer.ts` (the threshold constant only);
  - `tests/bvh/*`, `validation/gpu-tests/bvh.gpu.test.ts`.
- **Gain:** Sponza 1.0 (0.5–2), Cornell 0.35, crossings about 0.5.
- **Class:** bitwise; 3e changes only exact-t ties.
- **Steps:**
  - **3a.** CW_TRI_BUDGET K=2 without paired loads. Sweep K=1/2/3. Test paired loads as a separate variant and reject them if loop spill grows.
  - **3b.** BVH_ALPHA_BIT: put the alpha flag in MT record e1.w, a dead word, and call `alpha_pass` only when it is set. Compile CUSTOM_ALPHA out when no triangle is MASK (Cornell, crossings, nm_smooth). The Woop/validation records are untouched.
  - **3c.** Constant-bound traversal loops: `iter <= BVH_ITER_CAP`, `k < 32u`, `li < 127u`.
  - **3d.** BVH2_PRIV_STACK, only when `!BVH_CWBVH` and interactive. Port the BVH2 half of `bvh-private-stack.patch`. Re-measure after WP-1, because with fewer copies the gain shrinks from −0.53.
  - **3e (optional).**
    - Leaf-1 BVH2 input to the DP.
    - Exponent-OR byte decode (`(w & 0x00ff00ff) | 0x64006400`) in the existing 80 B node, so no data-format exception is needed.
    - Next-node prefetch.
  - **3f (last, after everything else has merged).** Re-sweep CWBVH_AUTO_MIN_TRIS at 2k, 8k, 32k, 65k and 130k triangles.
- **Verification:**
  - 3a–3d: T12 per-ray comparison of old vs new (closest-hit and any-hit, MT) with 0 bit differences; overflow and itercap flag tests; V-BIT with the CWBVH and Sponza-lite cases; VITE_STRESS on a CWBVH scene; `gate-m8 --part stageB` CWBVH units as a smoke run.
  - 3e: T12 against brute force with 0 unexplained differences under the existing tie rule; the `cwbvh.test.ts` mirror; gate-m8 stageB.
  - V-PERF, including the isolated T12 perf kernels.
- **Conflicts:** `scene-gpu.ts` with WP-8, as additive define lines; merge WP-3 first.
- **Delivered (WP-3 a–e, branch perf2-wp3):** five bitwise flags, all released (`RELEASE_PERF_FLAGS`; the registry got a
  `value` for parametrised release flags, CW_TRI_BUDGET ships as K = 2). The flag-free traversal text is byte-identical
  (64 define sets), so every validation text is unchanged.
  - `CW_TRI_BUDGET=K` (3a): one CWBVH loop, at most K triangles per iteration, a node step only once the triangle group is
    empty: the same node / triangle order and iteration count. Paired loads not tried. T12 sweep (Sponza, 1080p rays): K=1
    loses on primary rays (+11 %), K=2/3 win 4–6 % on incoherent rays; on top of CW_EXP_OR K=2 is best (−9…−10 %).
  - `BVH_ALPHA_BIT` (3b): MASK bit in MT record e1.w (scene-gpu.ts writes it always; no other text reads e1.w);
    alpha_pass only for flagged records; the scene key `BVH_NO_ALPHA` (no MASK triangle) removes the alpha test.
  - `BVH_CONST_LOOPS` (3c, new name): constant bounds on the iteration loop, the BVH2 leaf loop (127) and the CWBVH group
    loop (24). No effect on CWBVH when CW_TRI_BUDGET is on (that loop is already bounded).
  - `BVH2_PRIV_STACK` (3d): the BVH2 half of bvh-private-stack.patch.
  - `CW_EXP_OR` (3e, new name): exponent-OR byte decode (f16 1024 + b, exact). Leaf-1 DP input and prefetch not done.
  - Verification: per-ray bit equality vs the flag-free traversal (bvh.gpu.test.ts "perf2 WP-3": 10⁶ random + 960×540
    camera rays, procedural + Sponza, MT / Woop, BVH2 / CWBVH, BVH_STATS counters included; alpha hook; MASK-bit records;
    cyclic-graph overflow / itercap) 0 differences; m8-bits (all 27, CWBVH + Sponza-lite) twice with the flags, U-M4-BITS /
    U-M5-BITS with the flags, VITE_STRESS on the CWBVH (VITE_STRESS_CWBVH=1) twice with identical counters, 0 faults.
  - ABBA 540p N3, 2 blocks (validation/out/perf2-wp3): Sponza all CW flags −5.2 ms (K2+EXP −1.4 frame / −4.4 Σpasses,
    ALPHA −0.44); nm_smooth all BVH2 flags −4.3 ms (PRIV −2.9, ALPHA −2.1, CONST −1.3); Cornell −0.6 (PRIV −0.9, ALPHA
    −0.44, CONST −0.44). Spill: Cornell hot kernels −400…−900 B (PRIV_STACK); Sponza ±16 B wobble (net −48 B, shift +16 B).

### WP-4 Lights and environment
- **4a (parallel, S–M, bitwise):**
  - RS_ENV_WRAP: select-based wrap with a fallback to the general modulo when `x0 ∉ [-dim, 2dim)`.
  - 16 B-aligned light records read as 7 vec4u loads (`LUT_RECORDS_KIND 3` already exists).
  - A rect/disk index list in `records`, consumed by WP-2e.
  - Optional: store the realized pdf/pmf next to the alias entries, under a define, worth about 0.2 ms.
  - **Files:** `lights/{env.wgsl, env-sample.wgsl, lights.wgsl, emissive.wgsl}`, `render/lights-gpu.ts`.
  - **Gain:** Sponza 0.6, crossings 0.3, Cornell 0.
  - **Verification:** ENV-U7, 7b and 7c with 0 mismatches, adding the edge uv values 0, 1, ±0.5/dim and 1 + 0.5/dim; V-BIT (all-lights fixture with env, Sponza-lite); lights-gpu and alias tests.
- **4b (after WP-2d, M, unbiased results change):** RS_ENV_PRESAMPLE.
  - `rs_light_tiles` writes per env slot: `(entry, (i<<16)|j, h2)` from a fresh pcg stream (not h.y, which biases tables with more than 65,536 alias entries), plus dir, Λ and q. Tiles stay in the arena; a read-only alias of the arena is invalid in WebGPU.
  - Point, spot and area slots are unchanged.
  - Gate it on env presence.
  - Alternative to A/B first: workgroup-uniform candidate identities (a one-line variant).
  - Tile memory grows from 0.5 MB to 4–6 MB per member.
  - **Files:** `passes/restir/light-tiles.wgsl`, `restir/{m6-types.wgsl, ris-nee.wgsl}`, `layout.ts`.
  - **Gain:** Sponza about 1.2 on top of 2d, 0 elsewhere.
  - **Verification:**
    - V-UNB with an HDRI + Mode-B mixed-lights package; U3-RIS; U8-10 counting entry ids at the new stride.
    - Support-positivity check: no `rgb9e5` zero channels.
    - V-Q, including 8×8 block correlation before and after the denoiser, on Sponza and m6_crossings.
  - Needs user sign-off (D7).

### WP-5 Spatial reuse (M, about 1.5 weeks)
- **Items:** #6 and #16.
- **Files:**
  - `passes/restir/{pair-accept.wgsl, spatial-shift.wgsl, spatial-replay.wgsl}`;
  - `restir/{queue.wgsl, pairing.wgsl, mis.wgsl, frame.wgsl}` (`rs_pix`);
  - `render/restir/{stage-spatial.ts, resources.ts, layout.ts}`;
  - the U-RES-1 mirror and `boost.test.ts`.
- **Gain:** Sponza 1.8 (1.0–2.8), Cornell 1.0 (0.6–1.5).
- **Class:** bitwise. The ACCEPTED and SC_NOT_ACCEPTED counters get documented per-side or analytic accounting.
- **Steps:**
  1. **RS_BOOST_GATE.** T1 does an aggregated `atomicOr` of an "any disocclusion" word. While it is 0, use NS_eff = slots and treat boost slots as NOT_ACCEPTED without reading them. Debug views 420–446 use the same gate.
  2. **Write-back trimming** in `mis.wgsl`: one store per plane and a single c_j read.
  3. **RS_DENSE_SLOTS.**
     - pair_accept writes only its own 6 slots, using one canonical-ordered `pair_A0` call.
     - Appends are workgroup- or ballot-aggregated so a pixel's items stay adjacent.
     - The item region is two-ended: non-replay items fill from the top, counted in the q3 header; replay items stay at the q0 bottom, so `rs_spatial_replay`'s text is unchanged.
     - A 64-wide indirect `RS_REPLAY=0` shift keeps the `res_needs_replay` guard as a safety net.
     - `rsdbg_accept` moves to pair_accept.
     - A/B it on Cornell against two alternatives: workgroup-local compaction inside the per-pixel shift, and fusing pair_accept into the shift (0.7 / 0.6 ms).
  4. **Pairing table.** A CPU `Math.imul` pcg4d mirror per (member, t, round, slot), an exact float-reciprocal modulo, and an `rs_pix` fast path for memberCount==1.
- **Verification:**
  - V-BIT and V-SHIFT (VITE_STRESS at 3 rounds × 6 slots; RSC_QUEUED counts replay items only).
  - The pairing involution tests; a CPU test of the pcg4d mirror against the WGSL; an operand-swap test for `pair_A0`.
  - V-PERF including a `--pan` job, plus split sums for accept, shift, replay and resample.
- **Dependencies:** measure after WP-1 has merged. Owns `queue.wgsl`, `layout.ts` and `pairing.wgsl`; merges before WP-6.
- **Delivered (WP-5, branch perf2-wp5; measured on perf2 without WP-1):**
  - Flags: `RS_DENSE_SLOTS` (released), `RS_MIS_TRIM` (released; split out of the dense flag), `RS_BOOST_GATE` and
    `RS_PAIR_TABLE` (landed, not released: no gain). All bitwise; validation text unchanged (U-M7-BITS).
  - Dense queue as specified, plus: `RSD_BOOST_OPEN` (RsDispatch.flags bit 12) opens the gate on frames without T1;
    header word 30 is the gate (cleared by the temporal stage before T1); q3 = header words 12–15. Pair_accept still
    writes the gated boost slots (NOT_ACCEPTED, no partner / A0 work), so debug views need no gate.
  - ABBA (540p N3, 2 blocks, all four flags): Sponza −7.9 ms (−8.6 %), Cornell −1.35 ms; crossings −3.6 ms, nm_smooth
    −3.4 ms; `pan` 0.02 m/frame Sponza −7.5 ms, Cornell −1.0 ms; 720p (1 block) Sponza −6.6 ms, Cornell −5.0 ms. Almost all
    of it is `rs_spatial_shift` (Sponza −7.0 ms: the per-pixel slot loop was divergent); `rs_pair_accept` +0.8 ms.
  - Dropped: `RS_DENSE_SLOTS=2` (min-thread acceptance appending both sides' items): pair_accept −0.8 ms but shift
    +0.9 ms — pixel-adjacent items matter. The workgroup-local-compaction / fused alternatives were not built (the
    queue already removed the shift's divergence). The global boost gate never closes on Sponza (jittered thin geometry
    leaves disoccluded pixels every frame); a tiled gate would need a dilation by the Gaussian maps' |d| ≤ 127 px.

### WP-6 Temporal reuse (M, about 1.5 weeks, after WP-5 merges)
- **Items:** #11.
- **Files:**
  - `passes/restir/{t-classify.wgsl, t-select.wgsl, t-inverse.wgsl, t-forward.wgsl}`;
  - `restir/{tshift.wgsl, tpick.wgsl, tframe.wgsl}`;
  - `render/restir/{stage-temporal.ts, refresh.ts, resources.ts, layout.ts}`;
  - the `PAIR_TS_*` mirrors in `pairing.wgsl`, coordinated with WP-5.
- **Gain:** Sponza 0.8 (0.4–1.5), Cornell 0.7 (0.4–1.0).
- **Class:** bitwise.
- **Steps:**
  1. Split T4 into a non-replay pipeline (`RS_REPLAY=0`) and a replay pipeline, by `res_needs_replay`. Bind res[w] read-only in T4. `refresh_inv` covers both queues.
  2. RS_TSEL_FOLD: for contribution MIS, do the phase-B work at the end of T4. Talbot, robust and recompute modes keep the old path.
  3. RS_TSTATE_SOA: word-major tState, with each record assembled once and written once.
  4. RS_AGG_COUNTERS: first stub the counters out to bound the possible gain; only then subgroup-reduce at a uniform exit. `tpick_ring` loses its return-in-switch.
- **Verification:**
  - V-BIT, including U-M5-BITS temporal cases and identical RSC_T_* counters.
  - V-SHIFT; T3-2 (TSEL_TRACE_FORCE_P).
  - The robust and Talbot validation modes stay on the old text.
  - N1+moving refresh frames; spill of the new T4 variant; V-PERF.

### WP-7 Frame, primary and denoiser (about 1.5 weeks)
- **Items:** #5, #9, #14, #19.
- **Files:**
  - `passes/{primary.wgsl, gbuffer.wgsl}`;
  - `passes/restir/{primary.wgsl, dupmap.wgsl, finalize.wgsl}`;
  - `render/denoise/{denoiser.ts, shaders/dn-temporal.wgsl, dn-resolve.wgsl, dn-filter.wgsl}`;
  - `render/renderer.ts`, `render/restir/{resources.ts, kernel.ts}` (external vbuf view);
  - `placement.ts`, `validation/gpu-tests/primary.gpu.test.ts`;
  - app-only: `post/blit.wgsl`, `render/present.ts`, `app/{app.ts, dynres.ts, integration.ts}`.
- **Gain:** Sponza about 2.5 (2.0–2.8), Cornell about 1.0 (0.8–1.1).
- **Class:** bitwise, except 7e (unbiased results change).
- **Steps:**
  - **7a. dn_gradient and dn_grad_filter skip** when λ is never read on that frame.
    - Keep LAMBDA, LAMBDA_CAM and INVERSE nested under GRADIENT.
    - Derive the HUD line from `r.gradient`.
    - Include the 527-view bit in the held-plan key.
  - **7b. dupmap s64** with a uniform branch: the old loop for memberCount > 1. Use the sentinel `(0xFFFFFFFF, 0xFFFFFFFF)` with a fallback.
  - **7c. Dead outputs.**
    - PRIM_NO_BEAUTY when a ReSTIR or PT frame will overwrite the beauty.
    - RSD_NO_DISPLAY when the denoiser will run, forcing an accumulation restart when it toggles back off.
  - **7d. Lossless 48 B G-buffer** `{pos, flags, ns, motionX, albedo, motionY}`.
  - **7e. RS_PRIMARY_EXT.**
    - A separate pass entry, `rs_primary_ext`; validation G2 layouts are unchanged.
    - Enabled by a Renderer option, not as a `RestirKernel.interactive` default, because tests call that directly.
    - Store `bits(t)` in vbuf.w and rename `VBufferTexel.matId`.
    - BVH-stats views keep tracing.
    - The primary BVH counters move to DBGC_*.
  - **7f. à-trous:** hoist zgrad into a texture and issue the colour load before the skip test. A/B each change and keep only measured wins.
  - **7g. App-only:**
    - Tonemap once per internal texel, then Catmull-Rom from 9 bilinear taps, with a 1:1 fast path.
    - Dynres changes the extent only, without reallocation. Histories still reset; the SoA plane stride must come from a uniform, not `arrayLength`.
    - Fix the bug where a pause on a frame that is a multiple of 30 keeps re-timing on every vsync.
    - Call `prepareDenoiser()` alongside `prepareRestir()`.
- **Verification:**
  - **7a:** a bitwise A/B of dnHist, dnMom, dnTaa, dnAlb, dnL1 and colour over 16+ static frames, then a light-change frame; also the gradientOnCamera path.
  - **7b:** the dup region bit-identical over 8 frames, including planted key collisions; U-DUP-1 and U-DUP-VIEW. Gate 5 is unaffected.
  - **7c, 7d:** the colour target and denoiser outputs bit-identical with the denoiser on, off and toggling, and on held frames; update `primary.gpu.test`.
  - **7e:**
    - A new Renderer-level test showing external rsVbuf equals the M1 vbuf exactly, over 16 frames with camera and light motion, on all-lights, nm, glass, alpha and Sponza-lite.
    - The existing M1-vs-rsVbuf tolerance in `denoise-run.ts`, unchanged.
    - Extend the Q3 test to cover the new texture dependency on a timestamped pass.
    - The M4/M5/M5.5 app smokes.
  - **7f:** U-DN-3b and U-DN-3c.
  - **7g:** HUD present timing; screenshot diff ≤1 LSB; a dynres app trace with no spikes.
- **Delivered (WP-7 a–d, f, g; 7e later in the merge order):** flags `DN_GRAD_SKIP` (7a), `RS_DUPMAP_S64` (7b),
  `PRIM_SKIP_BEAUTY` + `RS_SKIP_DISPLAY` (7c), `GBUF_48` (7d), `DN_ZGRAD_TEX` + `DN_COLOUR_EARLY` (7f), all bitwise and
  released ([perf2-api.md](perf2-api.md) §1). Proof: `wp7-bits.gpu.test.ts` (U-WP7-APP: the app frame with and without the
  set, 31 frames incl. a light change, motion, held frames, view 527, denoiser toggling, gradientOnCamera; U-WP7-DUP:
  planted seeds incl. the sentinel), m8-bits Anchor with the kernel flags forced and Shipped with the release set,
  U-DUP-VIEW and the denoiser suite with the flags forced, spill-lint `wp7-all` (no spill growth). 7c's documented
  difference: the first frame shown without the denoiser (or the M1 beauty) restarts the progressive mean. 7g: tonemap
  once (`post/tonemap.wgsl`, U-WP7-PRESENT ≤ 1 LSB), the paused re-timing fix, the early denoiser prepare. Dropped: dynres
  without reallocation (touches the shared ReSTIR resources and the SoA plane stride in validation text for no gate gain).

### WP-8 Materials and BSDF compile variants (M, after WP-2a merges)
- **Items:** #12, plus optionally a validity-only `bsdf_sample` variant, consumed in `pathtree` only after WP-2 has merged (about 0.3 ms).
- **Files:** `material/{bsdf.wgsl, material-eval.wgsl, v1.wgsl, v2.wgsl, glass.wgsl, lut.wgsl, textures.wgsl}`, `render/scene-gpu.ts`, `render/restir/kernel.ts` (variant sync), `tests/material/luts.test.ts`.
- **Gain:** Sponza 0.7 (0.3–1.2), Cornell 0.5 (0.3–0.7).
- **Class:** bitwise.
- **Steps:**
  1. MAT_VARIANTS: MAT_HAS_V1, MAT_HAS_V2, MAT_HAS_GLASS, GLASS_NODE, REFRACTION and TEX_SLOT_* presence. Key them on the live material table and re-evaluate at frame boundaries, as `syncLightModeVariant` does, resetting history only on a real change.
  2. TEX_XFORM.
  3. Branch-free LUT lerp, with a CPU assertion that the tables are finite and non-negative.
  4. No return in switch in `tex_sample_*` and `lobe_roughness`.
- **Verification:**
  - A U-M8-MODEB-style A/B, with and without MAT_VARIANTS, bitwise on all-lights, nm, alpha, glass and a V1-only package.
  - A switching test: adding a glass material recompiles and resets history.
  - U-PT-BITS parts 2 and 3; `textures.gpu.test`; the Q1 canary; compile times recorded in `budget.json`; V-PERF.

### WP-9 Lean interactive text
- **9a (early, host-side only).** Compile the DEBUG_NO_BINDINGS variant while no ReSTIR view (400–499) or probe is active, and switch at a frame boundary without resetting history. Files: `kernel.ts`, `renderer.ts`, `restir/debug.ts`, and the app debug panel.
- **9b (last, a sweep once WP-1, 2, 5 and 6 have merged).**
  - RS_NO_PLANTS: removes the U8-*, TP_* and 2022-criteria code; 2022 criteria compile only in the criteria2022 app mode.
  - RS_NO_DIAG: skips the P9 writes and copies but keeps the allocation.
  - Files: `shift.wgsl`, `rc.wgsl`, `refresh.wgsl`, `tshift.wgsl`, `ris-nee.wgsl`, `pathtree.wgsl`, `reservoir.wgsl`, `mis.wgsl`, `t-select.wgsl`.
- **Gain:** Sponza 0.8 (0.3–1.5), Cornell 0.4.
- **Verification:** V-BIT with the P9-excluded hash; V-SHIFT; a debug-view smoke run (toggling a 4xx view switches variant, the views are populated, history is not reset).

### WP-10 Decision configurations (after WP-Q and user decisions; days of work)
- **Rule:** app-level defaults only, set in `restirAppSettings` for app mode `'interactive'`. Presets are unchanged.
- **Changes:**
  - Extend `RestirFeatureOverrides` and the panel with risM, rrMinBounces and slots.
  - Scope the maxBounces default to interactive mode; reset the glass raise when a non-glass scene loads; make `reference-flow` follow the renderer's maxBounces.
  - Add HUD and README labels, including "biased" where it applies.
- **Verification:**
  - V-Q evidence attached.
  - Gate 5 re-run at the shipped configuration: 12 chain units, the twin passes Stage B, and the dupmap bias stays ≤3.25%.
  - An N=2 Stage-B chain if slots change.
  - U1-M and U3-RIS at the new M.
  - The M5.5 denoiser gate.
  - The Anchor goldens are unaffected because the knobs are pinned.

### Dependencies, conflicts and order
- **Hard dependencies:**
  - WP-0 before everything.
  - WP-1's `vis_ray` before WP-2b; WP-2a before WP-8; WP-2d before WP-4b; WP-4a before WP-2e.
  - WP-5 merges before WP-6.
  - WP-1, 2, 5 and 6 before WP-9b.
  - WP-Q before WP-4b sign-off and before WP-10.
  - WP-3f last.
- **Files touched by more than one package:**

  | File | Packages |
  |---|---|
  | `kernel.ts` | 0, 2, 7, 8, 9a — registry lands first, the rest are additive |
  | `resources.ts` | 2, 5, 6, 7 |
  | `layout.ts` | 5, 6, 4b |
  | `queue.wgsl`, `pairing.wgsl` | owned by 5; 6 uses them |
  | `bsdf.wgsl` | 2a, then 8 |
  | `ris-nee.wgsl` | 2, then 4b, then 9b |
  | `scene-gpu.ts` | 3, 8 |
  | `renderer.ts` | 0, 3, 7, 9a, 10 |
  | `shift.wgsl` | 1, then 9b |
  | `pathtree.wgsl` | 2, then 9b |
  | `mis.wgsl` | 5, then 9b |
  | `t-select.wgsl` | 6, then 9b |

- **Merge order:** WP-0 → WP-1 → WP-2a → WP-7 (a–d) → WP-3 (a–d) → WP-4a → WP-2b/c → WP-5 → WP-2d → WP-7e → WP-8 → WP-6 → WP-2e → WP-4b → WP-9 → WP-3f → WP-10. After each merge, re-baseline main with a same-session ABBA against the previous main.
- **Staffing:** six to seven engineers in parallel worktrees:
  - E1: WP-1, then WP-9;
  - E2: WP-2;
  - E3: WP-3;
  - E4: WP-4a, then WP-4b;
  - E5: WP-5, then WP-6;
  - E6: WP-7;
  - E7: WP-8 from week 2;
  - one owner for WP-0 and WP-Q in the first days.

## 3. Decisions needed from the user

| # | Decision | Class | Measured / estimated effect | Recommendation |
|---|---|---|---|---|
| D1 | Russian roulette start for interactive mode: rrMinBounces 3 → 2 or 1, as an app default | unbiased, more variance | Measured before the packages: Sponza −5.5 (RR from B=3) / −8.5…−10 (from B=2); Cornell about 0. After the packages: about −4.5 / −7. Survivor weights go up to 8× (d=4) at 2 and about 64× (d=5) at 1 | Evaluate 2 first with WP-Q and the TD-I1 spike metrics; 1 only if equal-time quality holds |
| D2 | RIS M: 32 → 16 or 8 as the app default, or per-pixel M (32 only where there is no valid standard-tap history; MIS M kept at 32) | unbiased | Measured before the packages: Sponza −4.2 (M=16) / −6.8 (M=8); Cornell −1.1 / −1.65; crossings −1.6 / −2.4. After WP-2 and WP-4b about −1.8 / −2.8. Previously declined (m8-perf §6) | Re-decide after WP-2d; prefer M=16 or per-pixel M |
| D3 | Duplication map off by default | removes the only biased feature | Sponza −1.3 (−0.75 after s64); Cornell −0.7 / −0.4. Measured bias today ≤0.3% globally | Your call; needs one Stage-B chain of interactive Mode B with the map off |
| D4 | Spatial slots N=3 → 2 as the app default | unbiased, quality trade | Sponza about −3.4 after WP-1/5 (−4.7 before); Cornell about −1. Does not count toward the N3 targets | Only together with D2/D3 if WP-Q shows equal quality |
| D5 | maxBounces 2 for interactive mode | biased (truncation) | Sponza −5 (about −2 if D1=1); Cornell −1 | Not recommended unless D6 requires it; would need a bias study against the maxBounces-3 PT |
| D6 | What to do about the Sponza targets (35/32/51) | – | Not reachable at native resolution without biased shading-rate tricks | Choose: (a) dynres scale about 0.72–0.75 at 33 ms instead of today's 0.625; (b) biased half-rate path trees (history-age selection, not row parity; about −11 ms after the packages; L effort; darkening at disocclusions and on glossy surfaces, plus flicker); (c) re-scope the targets |
| D7 | Env-presampled tiles (WP-4b) | unbiased, adds correlation | about −1.2 ms on Sponza; each env sample is shared by about 127 pixel-candidates per frame | Sign off after the WP-Q block-correlation evidence |
| D8 | Keep interactive-only text separate from validated text permanently, or schedule a promotion pass (full Stage A/B, T3, U-M7 re-record) | maintenance | – | Keep separate; promote only large, bitwise-proven items |
| D9 | Enable Developer Mode for GPU counters | user-only system security setting | Turns the "register/occupancy-bound" inference into limiter data | Optional; only you can do it (`sudo DevToolsSecurity -enable`) |
| D10 | Quiet machine for gate runs | – | Background GPU load is 340–400 ms/s and the "Medium" perf state moved the same build by up to 50% | Close Safari, Arc and Slack during gate runs |

Not recommended under any setting: static-camera visibility reuse, skipping the B-5 retest, hardware env bilinear, texture LOD, a 3-level denoiser, and the 512 env importance cap. Each is ≤1 ms and biased or quality-negative.

## 4. Realistic end state

Values are in ms; central estimates with ranges. Column A excludes all decisions. Unmeasured items were discounted to 50–70%.

| Scene / config | Now (M8 gate) | A: all packages, no decisions | B: A + RR from B=3 + M=16 | C: A + RR from B=2, M=8, dupmap off, N=2 | D: C + biased (maxB 2, half-rate trees) | Target |
|---|---|---|---|---|---|---|
| Cornell 540p N3 | 16.2 | **12** (11–13) | 11.5 | 10 | – | 35 ✓ |
| Cornell 540p N1+moving | 14.8 | 11 (10–12) | 10.5 | 10 | – | 32 ✓ |
| Cornell 720p N3 | 28.9 | 21 (20–23) | 20.5 | 18.5 | – | 51 ✓ |
| Sponza 540p N3 | 78.7 | **62** (58–66) | 56 (52–60) | 48 (44–52) | 34–38 | 35 ✗ (D only) |
| Sponza 540p N1+moving | 71.8 | 58 (55–62) | 52 | 47 (no N change) | 33–36 | 32 ✗ |
| Sponza 720p N3 | 148.4 | 119 (112–126) | 108 | 94 | 66–72 | 51 ✗ |
| crossings 540p N3 / 720p | 25.7 / 52.1 | 20 (19–22) / 42 (39–45) | – | – | – | ✓ / ✓ |
| nm_smooth 540p N3 / 720p | 28.4 / 50.6 | 22 (21–24) / 40 (37–43) | – | – | – | ✓ / ✓ |

How column A for Sponza 540p N3 breaks down:
- Measured or trace-backed items: about 11 ms (WP-1 4.2, RIS hoist and pre-pass about 4.6, primary 1.2, dupmap 0.55, dn_gradient 0.3).
- Estimated items: about 5 ms after the realisation discount.
- Overlaps are already accounted for. #1 and #4 share the "fewer inlined traversal copies" mechanism. #2, #3, #8 and #13 all shrink the same RIS loop. The dense-queue gain is computed after the shift has already shrunk.

What this means:
- Cornell, crossings and nm_smooth meet all targets after the packages. Today only crossings at 720p misses, by 1.1 ms.
- Sponza remains about 1.8× over target at native 540p N3 with no estimator decisions, and about 1.4× with the aggressive unbiased decisions.
- Pixel cost is linear (quarter resolution measured 24%). With the packages alone, the existing dynres reaches 33 ms at a scale of about 0.72–0.75, compared with 0.625 today, which is about 40% more pixels at the target frame time. Native 35 ms needs D6(b) or a target change.

Uncertainty:
- About ±1.5 ms on the measured part.
- About ±50% on the estimated part.
- About ±10–40% session drift, which is why every package must ABBA within one session.
- The largest open question is #4: whether rs_initial benefits from fewer traversal copies the way the shift kernels did. It ranges from 0.5 to 2 ms and is cheap to test first in WP-2b.
## 5. User decisions (2026-10-09)
- **D1 / D2 / D4 (RR start, RIS M, spatial slots):** adopt only where the WP-Q equal-quality rule holds (equal-time error vs a converged reference); all remain unbiased.
- **D3:** duplication map **off** by default in interactive mode (needs one Stage-B chain of interactive Mode B with the map off).
- **D6:** **(b)** add biased half-rate path trees (history-age selection) as an interactive option for Sponza-scale scenes; bias and artefacts (disocclusion darkening, glossy, flicker) must be measured and documented; it must be a toggle, off in every validation path.
- **D9:** developer mode enabled; GPU counters still unavailable (Xcode 27.0 / macOS 26.6.2 / M5 Pro: counter sets contain only "RT Unit Active").
- **Execution:** parallel agents per package in separate worktrees, merged in the §2 order with same-session ABBA re-baselines.

### WP-Q evidence (2026-10-09)

Source: `validation/out/wpq-eq/decision.json` (harness `validation/harness/run-eq.ts` + `validation/tools/eq_eval.py`,
branch perf2-wpq). Cornell and Sponza at 960×540, static / pan / light-animation sequences, 4 seeds, frames
{16, 32, 48, 63}, against converged PT references (maxBounces 3). Rule (§2 WP-Q step 7): denoised LDR-FLIP and denoised
temporal std inside the baseline's seed-to-seed 95 % CI in all three sequences (6 checks per scene), and raw relMSE
(0.1 %-trimmed) × ms not worse than (baseline relMSE + CI) × baseline ms. Candidate ms = baseline ms × the same-session
ABBA ratio (`perf-abba1.json`; the absolute ms are from a loaded session, the ratios are what counts).

| Candidate | Cornell: ms Δ / relMSE Δ / relMSE×ms Δ | Cornell FLIP + tstd checks | Sponza: ms Δ / relMSE Δ / relMSE×ms Δ | Sponza FLIP + tstd checks | Verdict |
|---|---|---|---|---|---|
| D1 rrMinBounces 2 | −0.1 % / +0.8 % / +0.7 % (inside CI) | 6/6 pass | −7.4 % / +0.3 % / −7.2 % | 6/6 pass | **equal quality** on both: adopted |
| D1 rrMinBounces 1 | +2.7 % / +2.6 % / +5.4 % (fail) | 2/6 (tstd light +3.8 %, FLIP pan +1.3 %, tstd pan +1.9 %, FLIP static +2.0 %) | −12.0 % / +1.9 % / −10.3 % | 0/6 (FLIP +0.7…1.2 %, tstd +1.7…2.2 %) | fails: not adopted |
| D2 risM 16 | −2.4 % / +0.1 % / −2.3 % | 6/6 pass | −3.4 % / +2.2 % / −1.3 % | 5/6 (tstd static +1.1 %) | fails on Sponza: not adopted |
| D2 risM 8 | −6.2 % / +0.1 % / −6.2 % | 6/6 pass | −7.5 % / +6.9 % / −1.2 % | 2/6 (tstd light +1.0 %, pan +2.7 %, static +3.0 %; FLIP static +1.0 %) | fails on Sponza: not adopted |
| D3 duplication map off | −1.3 % / −7.4 % / −8.6 % | 0/6 (FLIP +1.9…5.1 %, tstd +3.8…6.8 %) | +2.5 % / −5.9 % / −3.6 % | 6/6 pass | fails on Cornell (denoised); **adopted by user decision** (unbiased, lower raw error on both scenes) |
| D4 slots 2 | −0.0 % / +23.4 % / +23.3 % (fail) | 0/6 (FLIP +1.8…5.4 %, tstd +6.1…8.1 %) | −4.2 % / +11.6 % / +6.8 % (fail) | 0/6 (FLIP +1.0…2.6 %, tstd +3.5…6.0 %) | fails: not adopted |

Reading: RR from bounce 2 costs nothing measurable in quality and saves about 7 % of the Sponza frame; RR from bounce 1
and RIS M 8 / 16 trade visible denoised flicker on Sponza for their time; two spatial slots lose clearly. The
duplication map is the only case where raw error and denoised quality disagree: switching it off lowers the raw relMSE
by 6–7 % (it removes the map's bias) but the denoised Cornell output gets 2–7 % worse in FLIP and temporal std, because
the map's confidence cap also damped temporal noise that the denoiser now sees.

### Applied (perf2-wpdec, 2026-10-09)

- `INTERACTIVE_APP_DEFAULTS` (`src/core/render/renderer.ts`) = `{ rrMinBounces: 2, dupmap: false }`: ReSTIR-interactive
  in the app is unbiased by default. The presets and every test pin (`INTERACTIVE_PINNED`, the bits rigs, the perf
  baselines) are unchanged, so every golden holds. The panel can switch the duplication map back on and set the RR start
  (1 / 2 / 3) per session.
- Stage-B evidence: `npm run validate -- --milestone M6 --part decisions` (opt-in part; δ and the Bonferroni count of
  the M6 gate unchanged). Results in [validation.md](validation.md#perf2-app-defaults-d1--d3).
- Performance: same-session ABBA `run-perf.ts --abba-restir app-defaults` (results in validation.md, same section).


## 6. Recovered second wave (2026-10-10)

Recovered WP-2d, WP-7e, WP-8 and WP-6 into `perf2` in that order (merge commits `5980fba`, `f86517b`,
`03de591`, `81d62dc`). Their original worktrees and uncommitted notes are preserved. This closes the interrupted
light-candidate pre-pass, primary-hit reuse/HUD attribution, material-identity and temporal-layout work.

### Accepted implementation

The release set adds `RS_RIS_PREPASS`, `RS_PRIMARY_EXT`, `MAT_VARIANTS`, `RS_LAST_ANYHIT`, `RS_TSEL_FOLD`,
`RS_TSTATE_SOA` and `RS_DEBUG_STRIP` to the prior 18 flags. Primary-hit reuse and debug stripping are renderer-only
(the standalone interactive kernel uses `KERNEL_RELEASE_PERF_FLAGS`).

- RIS selection runs in a lean pre-pass, using the pixel's dead RP_DIAG plane for the selected endpoint and weight.
  The initial path kernel reconstructs the endpoint. The original inline path remains for unsupported configurations.
- The renderer's M1 hit is reused instead of tracing the camera ray again.
- Material variants compile only the scene's material models, texture slots and transforms; field-wise loads reduce
  live state. The last continuation uses any-hit where applicable (Mode A without emissive triangles).
- Temporal selection phase B folds into T4; temporal state is word-major. The replay/non-replay split was dropped.
- With no active diagnostic view or probe, ReSTIR shaders omit debug hooks. Switching diagnostics selects a cached
  pipeline variant without resetting history; the debug-view pass retains its own instrumentation.
- The recovered HUD fix attributes held diagnostic timing submits separately. Frame timings below come from the
  independent frame harness, not sums of HUD pass times.

App quality settings stay at maxBounces 3, RIS M 32, spatial slots 3, RR start 2 and duplication map off. No new biased
flag, shading-rate reduction or reduced-resolution default is enabled.

### Fresh performance and quality

Apple M5 Pro / Metal-3, Chrome 155.0.8059.40. Each scenario has 2 blocks of 8 jobs in ABBA order (8 baseline and
8 candidate runs), 64 frames per job, 8 split pass frames. Baseline flags are explicitly the old 18-flag release set;
candidate flags are explicitly the 25-flag set above. Both sides use the same app settings, MT intersection,
textures, normal maps and automatic BVH selection (CWBVH on Sponza). The shared GPU lock was held for each job and
no xctrace ran during timing. External background GPU load was not independently logged for these runs.

| Scenario | Baseline ms | Candidate ms | Paired time reduction | Candidate FPS | Four ABBA pair reductions |
|---|---:|---:|---:|---:|---:|
| Sponza 960×540, static | 53.379 | 46.182 | 13.48% | 21.65 | 13.35–13.68% |
| Cornell 960×540, static | 12.151 | 9.507 | 21.76% | 105.18 | 21.33–22.09% |
| Sponza 1280×720, static | 94.717 | 81.869 | 13.56% | 12.21 | 13.31–13.76% |
| Sponza 960×540, moving light, N3 | 56.135 | 48.691 | 13.26% | 20.54 | 13.09–13.53% |

`validation/out/m8-perf/perf2-final.json` contains all 64 successful jobs; `perf2-final-summary.json` records paired
ratios and per-pass observations. Static denoiser time is effectively unchanged (Sponza 2.238→2.239 ms, Cornell
1.166→1.167 ms). Normalizing static pass observations by this unchanged work leaves the gains intact. Sponza initial
work is 32.516→25.227 ms plus a new 3.358 ms RIS pre-pass (net −3.931 ms); camera-hit work is 1.570→0.153 ms;
spatial shift 6.680→5.984 ms; temporal classification 3.703→3.211 ms; temporal selection 1.564→1.084 ms. These are
split-pass observations, not additive frame timestamps.

Fresh WP-Q: both configurations, two scenes, static/pan/light sequences, four seeds, 64 frames; evaluate
{16,32,48,63}, temporal window 40:64, using the existing converged PT references. The quality renderer is pinned to
`watertight:false` to match the timing jobs. **All 12 denoised FLIP/temporal-stability comparisons are inside the
baseline's seed-to-seed 95% confidence intervals.** Raw trimmed relMSE changes by −0.001% on Cornell and +0.011% on
Sponza; raw relMSE × frame time improves by 21.8% and 13.5%, respectively. Formal verdict: equal quality on both scenes.
Evidence: `validation/out/wpq-eq/resume-{base,opt,perf,decision}.json` and the 48 `wpq-resume-*` run directories.
These are finite-seed checks on the stated scenes/sequences, not a proof for all content or hardware.

### Identity, unbiasedness and compiler evidence

Bitwise flags reproduced Anchor hashes twice before release. The pre-pass diagnostic reports zero endpoint
mismatches and maximum relative W error 5.281e-7. Results change only in the expected last-bit weight arithmetic;
Shipped hashes are re-recorded for the combined release set while Anchor tables remain unchanged. Primary-hit
reuse has its separate V-buffer identity tests and app-level equal-quality evidence.

The recovered, unchanged pre-pass already passed Stage-B Cornell Mode A, crossings Mode B and HDRI chains at
frames 1/24, plus U8-8/U8-10 plant detection and controls. Evidence remains in the WP-2d worktree:
`validation/out/v-unb-RS_RIS_PREPASS-20261009-151838/summary.json` (6/6, 4078 seconds) and
`validation/out/v-unb-RS_RIS_PREPASS-20261009-174408/summary.json` (plants 10/10; controls 10/10).
These Stage-B runs are inherited evidence, not fresh full-gate reruns. Fresh combined checks cover the diagnostic,
RIS M=1/4/16/32, shift smoke, CWBVH production stress (40 frames, 3 rounds × 6 slots, no pending/mismatch/non-finite
counters) and the four-seed quality protocol. An accidentally started exhaustive shift suite was interrupted;
its partial output is not counted as a completed gate.

Fresh compiler-spill reports: `validation/out/perf2-spill/resume-{base,opt}/report.{json,md}`. No hot-kernel spill
growth. Sponza initial 800→768 B/thread, classification 464→384, temporal forward 640→576, inverse 784→752,
spatial replay 656→560, spatial shift 464→384, refresh forward 304→176. Cornell initial 944→704, classification
592→400, temporal forward 720→480, inverse 720→496, spatial replay 704→448, spatial shift 592→384, refresh forward
384→144. The pre-pass and primary extraction have zero spills. Field-wise material loads increase some syntactic
robustness-clamp counts (e.g. Sponza temporal forward 58→74) despite lower spills and measured time.
xctrace returned its known unsupported-counter/log-archive warnings; compiler events and encoder tables exported
successfully. These reports do not establish hardware-counter behavior.

### Harness repairs and rejected experiments

- Preserve array-form flag values (`CW_TRI_BUDGET=2`) instead of treating the complete token as an unknown name.
  The failed first quality attempt produced no metrics; all quality jobs above were rerun after the repair.
- `run-eq --renderer JSON` now pins the same intersector/options as performance jobs and stops on a failed render.
- Synthetic temporal/debug and denoiser fixtures now encode the chosen temporal SoA layout. Material A/B fixtures
  use the standalone kernel's release set, excluding renderer-only flags. The recovered V-UNB plant runner forwards
  kernel flags and supports explicit U8-8/U8-10 runs.
- Earlier harness bug `4b760d6` could drop release flags from ABBA variants. Historical package timings are not used
  as evidence for the combined result above; every fresh timing report records both complete flag sets.
- `RS_NO_PLANTS` was prototyped, changed Anchor hashes and provided only a small gain; removed. `RS_NO_DIAG` remains
  off (the pre-pass needs RP_DIAG scratch storage).
- Initial thread groups 16×4, 32×2 and 64×1 did not beat 8×8. CWBVH triangle budgets 1,4,8 did not beat 2. All were
  dropped. Screening reports: `perf2-resume-screen.json` and `perf2-resume-tuning.json` in `validation/out/m8-perf/`.

### Final release verification

- Production build and TypeScript checks pass; CPU suite: 590 passed, 7 intentionally skipped (64 files).
- Released M8 Anchor/Shipped, debug and denoiser suites: 63/63. M8 repeated: 36/36, all 40 logged full/P9-excluded
  hash arrays identical between runs. Anchor, Anchor_NOP9 and PT reference tables are unchanged.
- M4/M5 bitwise suites: 11/11. Default interactive frame/refresh checks: 3/3. Material/primary-hit identity: 14/14,
  including an explicitly MAT_VARIANTS-disabled baseline so the A/B remains meaningful after release.
- Production CWBVH stress repeated: identical accepted/queued/shift-code counters; zero pending, mismatch or
  non-finite errors in both runs.
- App smokes: M4 31/31, M5 87/87, M5.5 37/37, M6 16/16, M7 21/21. No console/WebGPU errors.
  M5/M5.5/M6 view checks wait for three actual ReSTIR advances after selecting a view: the first instrumented
  variant compiles asynchronously, and screen frames during compilation show the existing PT fallback. Counting
  those screen frames caused the earlier blank-view smoke failures; all required AOVs pass after waiting for
  completed ReSTIR frames. Pause/resume, history, probes, normal maps and HUD timing also pass.

Logs are `validation/out/perf2-release-{build,cpu,bits-1,bits-2,interactive,m45-bits,stress-repeat,material-primary}.log`
and the `perf2-release-*-smoke*` report directories. Full exhaustive Stage-B/shift gates were not rerun; inherited
and fresh evidence are distinguished above. Other GPU families still need their platform-lane checks.

### Remaining real-time gap

This is a measured second-wave improvement, **not a 60 FPS Sponza result**: 46.18 ms at 540p still needs about
2.77× more throughput to reach 16.67 ms. The remaining initial path-building shader dominates (~25 ms); RIS selection
and spatial shift are the next large components. Further work should target traversal/path construction and test
WP-4b environment presampling against the block-correlation and equal-quality gates. The subsequent user instruction excludes dynamic resolution; §7 uses fixed resolution throughout. Do not silently adopt smaller RIS M, two slots, earlier RR or
half-rate path trees: the existing quality failures and D6 constraints still apply.


## 7. Fixed-resolution follow-up (2026-10-10)

The recovered checkpoint `dd87b5c` was pushed to `origin/perf2` before starting this work. The user's targets
are 960×540 near 60 FPS, 1280×720 near 30 FPS, and 1920×1080 allowed below 30 FPS, preserving high quality.
**No dynamic resolution, sample-count reduction, material simplification or denoiser-quality reduction was introduced.**

### Released changes

- `CW_SCENE_STACK` (WP-3g): size private CWBVH stacks to the encoder's actual maximum wide-tree depth,
  except spatial shift (its smaller stack increased compiler spills and was removed from that pass).
  The encoder counts root depth as one; at most one pending node group is pushed per level. Both the allocation
  and overflow guard use the same scene define. Sponza's maximum wide depth is **12**, versus the old 16 entries.
- `CW_WG_STACK` (WP-3h): the initial path pass uses a column per invocation in a workgroup-memory stack.
  The pass is 8×8; `(gid.y & 7) * 8 + (gid.x & 7)` selects its unique column. Only that invocation accesses it,
  and every pop follows its push, including successive rays. No inter-invocation synchronization is required.
  Scene depth × 64 × 8 bytes is 6 KiB for Sponza. The flag activates only for `rs_initial` and its dump entry;
  other passes and BVH2 retain their established storage. This is distinct from the rejected module-private stack.

These are two **bitwise** additions to the 25-flag release. Neither golden tier was re-recorded. Sampling,
RNG, primitive order, traversal order and denoiser outputs are unchanged. No new builder is selected by the app.

### Measurements

M5 Pro, Metal 3, Chrome 155.0.8059.40. Each row uses same-session ABBA, 64 timed frames per job,
identical settings and scene assets. Baseline is the complete `dd87b5c` release flag set; optimized adds only
the two stack flags. Frame time includes the denoiser. Split-pass times and denoiser timestamps are measured
in separate reruns and must not be summed into frame time. Exact flag sets/settings and numeric samples are
committed in [perf3-results.json](perf3-results.json); full reports are `validation/out/m8-perf/perf3-final-clean.json`.

| Scene / fixed resolution | Before ms | After ms | Change | FPS |
|---|---:|---:|---:|---:|
| 540p | 46.269 | 43.891 | -5.14% | 22.78 |
| 720p | 81.875 | 77.897 | -4.86% | 12.84 |
| 1080p | 185.970 | 174.959 | -5.92% | 5.72 |
| 540p-moving | 49.666 | 46.555 | -6.26% | 21.48 |
| cornell-540p | 9.986 | 9.978 | -0.08% | 100.22 |

Sponza's initial pass falls from approximately 25.2 to 23.0 ms at 540p. The denoiser remains about 2.24 ms
(3.98 ms at 720p, 8.92 ms at 1080p), so denoiser-only changes cannot close the target gap.
The longer 16-job Sponza confirmation (`perf3-confirm-clean.json`, four ABBA groups, 512 measured frames per
configuration) averaged 48.081→44.371 ms, -7.72%, but includes one 59.039 ms baseline job.
All samples are retained in the JSON; this mean overstates the typical gain. The median ABBA-group change
is -5.17%, consistent with the matrix above.

A telemetry issue was found while checking background load: `gpu-bg.py` assumed `IOUserClientCreator`
preceded `AppUsage` in textual IOKit output. On this host, the order is reversed, assigning Chrome's work to
the preceding Safari helper. It now parses structured registry objects; two regression tests cover property
order, nesting and multiple clients. The corrected live sample attributes 983.8 GPU ms/s to Chrome, with
negligible other load. Old background **process attribution** logs are invalid; this does not affect frame
measurements or GPU-lock ownership. The final matrix and confirmation use the corrected logger (`perf3-final-clean-bg.log`): 147 samples,
other-process load mean 0.59 GPU ms/s, p95 2, maximum 17. No competing benchmark or trace ran concurrently.

### Experiments retained or rejected

- Smaller private stack alone: Sponza 46.234→44.371 ms; Cornell unchanged. Shared stack in initial adds roughly
  0.8 ms. Expanding shared stacks to spatial shift or all reuse passes regresses (43.778→44.073/44.298 ms), so
  those variants were removed.
- Four specialized à-trous pipelines: 46.234→46.269 ms total; denoiser improvement only about 0.02 ms. Removed.
- Initial workgroups 8×4: about 0.22 ms, below the affected-pass 2% threshold; removed. 8×2 regresses to
  56.920 ms. Constant-bound path loop is neutral; removed. All retained pixels still receive the same work.
- tinybvh WebAssembly is now a reproducible optional performance probe, **not a released renderer path**.
  Optimized object splits screen about 4% faster than the new TS baseline, spatial splits about 5.6%.
  Plain object splits regress. See [the pinned build, conversion and bounds findings](../research/m8-hwrt-tinybvh-gigi.md#6-webassembly-probe-revisited-2026-10-10).
  Spatial clipping is not yet proven conservative; changed primitive tie order in either alternative needs
  its own quality validation. The experimental harness selector is never enabled without an explicit job URL.

### Correctness and target status

Production build/typecheck and CPU suite pass (590 passed, 7 skipped, 64 files), including unchanged validation
shader text. M8 Anchor/Shipped/PT/Mode-B/plumbing pass three times (36 each; all 40 reported hash arrays repeat
exactly). M4/M5 predecessor-bit cases add 11 passes. New million-ray comparisons on procedural geometry and
Sponza match every closest-hit/any-hit output word and every traversal counter for MT and Woop, both private
and shared stacks, nested and budgeted traversal loops. Overflow and iteration-cap flags are zero.

App smokes: M4 31/31, M5 87/87, M5.5 37/37, M6 16/16, M7 21/21. Screenshots and logs are under
`validation/out/perf3-m*`. After the final spatial-shift guard, M8 was rerun and the 40-frame, three-round,
six-slot CWBVH production stress was repeated twice with identical counters and zero pending, mismatch,
non-finite failures. Tests selected by name leave unrelated exhaustive gates skipped; this is not
an assertion that the full Stage-B suite was rerun. The four-seed quality evidence at `dd87b5c` is inherited;
this release preserves its sampled outputs bitwise.

Compiler spill/lint comparison: `validation/out/perf2-spill/perf3-stack-final/report.{json,md}` against
`resume-opt`. Sponza initial 768→528 B/thread, temporal forward 576→544, inverse 752→720; spatial shift
stays 384. No hot-kernel spill or robustness-clamp growth; Cornell unchanged. An earlier version had +16 B
in spatial shift, which is why that pass now keeps its old stack size. As in §6, xctrace reports exit 2 during
recording but exported compiler-event/encoder tables are usable; no hardware performance counters are claimed.
`spill-lint.ts` now records and passes explicit release flags when no override is supplied, and an explicit
empty flag string correctly means no flags. Its older default-run report header could misleadingly say “none”.

**The requested Sponza targets are not reached.** At the same quality, the released 540p result still needs
about 2.6× more throughput and 720p needs about 2.3×. The fastest tinybvh experiment also remains around
24 FPS at 540p. The large remaining cost is path construction/traversal; the experiments here do not justify
promising that another local shader tweak or denoiser change will deliver the missing factor.

## 8. tinybvh quality, fresh profiling, and the remaining gap (2026-10-10)

Follow-up to the user's request to compare tinybvh visually, profile current code and investigate game-engine
techniques. Renderer/shader source is `02a56bd`; **no production shader, sampling, resolution, builder or default
was changed in this investigation**. The user judged the initial comparison visually identical. Reproduction
jobs, exact timing blocks, pass reports, quality measurements and ray comparisons are committed in
[perf4-results.json](perf4-results.json). Large captures are local, under `validation/out/perf4-*`.

### Fresh timing and bottlenecks

Apple M5 Pro, Chrome 155, existing 27 release flags, interactive ReSTIR, RR minimum 2, RIS 32, three slots,
three bounces, duplication map off, denoising on. Each tinybvh comparison is same-session ABBA with 32 warm-up
and 64 timed frames per job. Two variant measurements and two controls per comparison:

| 960×540 builder | Control ms | Variant ms | Change | Variant FPS |
|---|---:|---:|---:|---:|
| tinybvh optimized object splits | 43.807 | 42.060 | −3.99% | 23.78 |
| tinybvh spatial splits | 43.887 | 41.354 | −5.77% | 24.18 |

Fresh current 1280×720: **77.684 ms / 12.87 FPS**. Neither builder bridges the target gap. WASM builds the
acceleration structure on the CPU; rays still traverse it in WGSL compute. This is not hardware ray tracing.

The four current 540p controls agree within 0.12 ms. Their isolated pass reruns identify approximately:

| Work | 540p ms | 720p ms |
|---|---:|---:|
| Initial path trees (`rs_initial`) | 23.0 | 40.55 |
| Spatial shifts | 6.0 | 11.39 |
| Temporal classify / forward / select / inverse, combined | 5.6 | 9.11 |
| First-surface RIS light selection | 3.4 | 5.66 |
| Primary geometry | 1.7 | 2.79 |
| Denoiser (own timestamp rerun) | 2.24 | 3.98 |

**Do not sum these into frame time:** split submits change scheduling and include residual per-submit costs;
the denoiser column comes from a separate timestamp rerun. The pipelined frame measurement remains the target.
The previous compiler trace still applies to unchanged shader source: initial spills 528 B/thread, spatial shift
384 B/thread, temporal inverse 720 B/thread. This is evidence of register pressure, not a hardware occupancy or
bandwidth measurement. No new GPU hardware-counter capture is claimed.

Diagnostic ablations at 540p: no textures 35.559 ms, alpha made opaque 43.386, no environment 43.613,
`maxBounces=1` 32.516, `maxBounces=2` 40.459, RIS M=1 41.333, temporal ReSTIR off 33.845.
**These are cost probes, not proposed quality settings.** Removing textures changes material values and the
subsequent path workload, so the 8.3 ms difference is not an isolated texture-fetch cost. Disabling temporal reuse
also changes downstream spatial work. Differences cannot be added. Even the severe individual reductions do not
reach 60 FPS. Previous quality failures for fewer samples/slots remain applicable.

The million-ray standalone test explains why lower BVH SAH does not imply a large frame gain: optimized object
splits reduce box tests by about 22% but increase triangle tests by about 16%. Spatial splits reduce box tests by
about 10% and triangle tests by about 16%. These are traversal-test workloads, not measured in-frame ray counts.

### Visual and numerical comparison

36 runs = three builders × static / oscillating camera / oscillating point light × four seeds. Every run renders
64 frames at 960×540 with the same renderer, materials, RNG seeds and denoiser. Camera motion is ±0.5 m, light
motion ±1 m. Evaluation frames 16/32/48/63 are motion knots at the reference camera/light state. Comparisons use
the existing 12,288-sample PT reference; its residual relative-MSE noise floor is 0.000322. Captures use the
quality harness's rgba32float target; performance uses the production-style rgba16float target.

| Mean denoised LDR-FLIP (16 images per cell; lower is better) | Current | tiny opt | tiny spatial |
|---|---:|---:|---:|
| Static | 0.06715650 | 0.06715666 | 0.06715634 |
| Camera motion | 0.10010992 | 0.10011062 | 0.10011083 |
| Light motion | 0.11073028 | 0.11073044 | 0.11073384 |

Largest aggregate FLIP change is **0.00000356**. Mean absolute display-RGB difference from current, averaged
across the 48 matched denoised images: opt **0.00000351**, spatial **0.00001809**. Worst captured image has
545 (0.105%) / 3,084 (0.595%) pixels differing by more than one 8-bit code in any channel, respectively.
The worst isolated display-channel change is 0.704 during light motion (seed 2, frame 63);
`worst-pixel.png` shows it at 4× magnification, so aggregate scores do not conceal this outlier.
The overall differences are small, but the images are not bitwise identical. Rare path/primitive ties can spread through
resampling and denoising. Sampled temporal changes are essentially identical between builders; these short,
seeded motion sequences are not exhaustive temporal-robustness proof.

Each builder has 96 primary-hit captures (49,766,400 pixel samples). Opt changes 148 primitive IDs, spatial
817; **zero hit-versus-miss changes**. Changed IDs alone do not prove cracks, and they were not individually
classified as geometric ties by this image test.

An opt-in GPU test compares each builder in MT and Woop modes against the f64 CPU reference on **one million
rays, including 100,000 edge-directed rays**. Brute-force queries anchor a 32-ray subset. All six combinations have
zero unexplained closest-hit or any-hit discrepancies, zero random-ray precision exceptions, and zero stack
or iteration-cap flags. Expected coincident-surface ties and edge precision cases are reported, not suppressed.
The ray test uses opaque geometry; alpha/material behaviour is exercised by the full-scene image runs.

**Conclusion:** optimized object splits are a credible quality-preserving candidate for production integration.
Spatial splits also look good here, but the known conservative-clipping issue is not disproved by sampled tests.
Keep the spatial builder experimental until source fragment bounds, seams and additional difficult scenes are
covered. No new golden images were recorded and neither builder was made the default.

Review locally:
- `validation/out/perf4/review.html`: native-resolution wipe, static/camera/light selectors and ×16 difference view.
- `validation/out/perf4/static-comparison.png`: overview and native pixel crop.
- `validation/out/perf4/pan.mp4`, `light.mp4`: synchronized three-way captures. These are **offline playback**, not
  real-time FPS recordings; one captured image per four simulation frames, repeated into 24 FPS video.

### What game engines do, and what transfers here

Sources were checked on 2026-10-10; these are primary engine/vendor publications.

1. [Epic's Lumen performance guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-performance-guide-for-unreal-engine)
   budgets 4/8 ms at 1080p for GI/reflections, using cached lighting and selective ray tracing. Rough surfaces can
   reuse the GI representation instead of tracing dedicated reflection rays; probe updates have a per-frame budget.
   Its figures cover a lighting subsystem, not our complete renderer, and rely on different platforms/settings.
   We can borrow caching and ray allocation while retaining fixed output resolution. We are not adopting its
   upscaling or reduced-resolution reflection modes.
2. [AMD Brixelizer GI](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/brixelizer-gi/) uses sparse distance-field
   tracing, world-space radiance/irradiance caches and screen probes. Prior-frame lighting contributes indirect
   transport. This amortizes shading across pixels and frames; it does not evaluate a full triangle path at every
   pixel. Its scene representation and interpolation have different error modes from our triangle tracer.
3. [NVIDIA SHaRC integration](https://github.com/NVIDIA-RTX/SHARC/blob/main/docs/Integration.md) separates sparse
   path updates, cache resolve, and rendering that terminates eligible secondary paths at cached radiance.
   Its example sparse update processes about 4% of pixels. The supplied hash/cache design needs a WGSL port,
   memory budgeting and synchronization work; it is not a drop-in replacement for our ReSTIR path suffixes.
4. [Laine, Karras & Aila, wavefront path tracing](https://research.nvidia.com/index.php/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus)
   explains why divergent material code and large live register sets can make large kernels inefficient. However,
   our earlier per-bounce prototype increased spills (1280 vs 1104 B/thread). A useful new experiment must split
   traversal from shading/reservoir work with compact state; merely splitting by bounce repeats a failed idea.
5. [ReSTIR PT Enhanced](https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/) combines reciprocal
   spatial pairing, footprint criteria and joint direct/indirect reservoirs. Much of this is already implemented
   here. Its published 2–3× improvement is over an older algorithm; it is not another 2–3× waiting to be applied.
   [WebGPU's ray-tracing extension discussion](https://github.com/gpuweb/gpuweb/issues/535) remains open; native
   DXR/Vulkan/Metal RT performance is not evidence for equivalent throughput in this WGSL software traversal.

### Recommended route, with explicit checkpoints

**First, take the modest exact gain:** productionize the optimized object builder behind an explicit experimental
selector, with a worker, pinned WASM artifact, deterministic input/buffer hashes and TS fallback. Validate Cornell,
alpha-heavy geometry, glass and moving lights before promoting it. Expect the measured ~4%, not a step to 60 FPS.

**Highest-value architectural experiment: cached diffuse transport in an interactive hybrid mode.** Retain full
resolution primary geometry and direct-light visibility. Use ReSTIR for direct lighting, a world-space cache for
rough diffuse indirect transport, and explicit rays for sharp reflection/refraction. Refresh cache entries on
light/geometry changes and trace normally on cache misses/disocclusions. Full ReSTIR PT remains the reference.
This is a biased interactive approximation: do not insert cached radiance into existing path reservoirs without
re-deriving their shift targets, Jacobians and history invalidation. A separate first prototype makes that boundary
reviewable and avoids silently changing validation semantics. It offers a plausible way to reduce both the initial
path work and expensive general path shifts; it is **not yet a measured speedup or quality guarantee**.

For an initial 540p target budget, allocate roughly 2 ms primary geometry, 3 ms direct lighting, 4 ms indirect
cache update/query, 3 ms explicit glossy/transmission work, 2 ms denoising and 2 ms other work. This **16 ms budget
is a design target, not a forecast**. Prototype on the measured Sponza view before scaling scope; abandon or revise
it if update cost/quality cannot fit. Test cold start, fast turns, thin walls, foliage, glossy surfaces and moving
lights against the same reference protocol, including localized error and temporal lag, not just whole-image means.

**For a quality-equivalent path:** benchmark traversal-only queues separated from material/reservoir processing,
while keeping RNG keys and estimator arithmetic fixed. Count rays by category, measure queue traffic and register
spills, then compare complete frame time. Stop if the combined result does not beat the current initial+reuse
passes. This is lower confidence than the cache route because earlier splits and visibility queues did not pay off.

**Denoising is supporting work.** Even a hypothetical free denoiser recovers only about 2.24 ms at 540p / 3.98 ms
at 720p. Better temporal reconstruction may enable fewer *indirect* samples at equal visible quality in the hybrid
mode, but weakening the current denoiser or simply lowering RIS/slots already has adverse quality evidence.

The remaining required speedup is ~2.63× at 540p / ~2.33× at 720p. Optimizing only a small pass cannot close it.
No dynamic resolution is proposed or introduced.

### Reproduction / checks

`perf4-results.json` includes both job arrays; extract them with Python or jq into ignored output, then run:

```sh
sh validation/tools/tinybvh/build.sh
npx tsx validation/harness/run-perf.ts --jobs validation/out/perf4/profile-jobs.json --out validation/out/perf4 --tag profile
npx tsx validation/harness/run-denoise.ts --jobs validation/out/perf4/quality-jobs.json --lock-per-job
VITE_TINYBVH_QUALITY=1 npx tsx validation/harness/with-gpu-lock.ts tinybvh-quality -- npx vitest run --project chrome validation/gpu-tests/bvh.gpu.test.ts -t 'tinybvh quality'
validation/.venv/bin/python validation/tools/tinybvh/review.py
```

The quality jobs require the existing `wpq-sponza_perf_540` scene package and `wpq-ref-sponza` reference. Recreate
those with the existing equal-quality harness if absent. `review.py` produces PNGs, metrics and the viewer; the
MP4s use ffmpeg at six input captures/second, repeated to 24 FPS. The ray gate is opt-in so ordinary suites do not
require a compiled WASM artifact. Harness changes are research-only builder injection and primary-hit capture.

Build/typecheck passed; CPU suite 590 passed / 7 skipped. All 16 profile jobs and 36 capture jobs passed.
The six million-ray/intersector comparisons passed, and browser checks covered all six viewer combinations,
difference mode and playback. These checks do not replace the full unbiased Stage-B gate for a future builder release.


## 9. Potato ReSTIR

User authorized a separate quality-for-speed mode on 2026-10-10. Interactive stays the launch default.
This section records the default Potato implementation; §10 adds an optional experimental indirect cache.
No dynamic resolution, reduced render target, sparse pixel dispatch or upscaling was introduced. All output pixels
still trace primary geometry and receive freshly sampled lighting. This mode is an explicit approximation to the
longer-path image, not an equal-quality optimization or a change to the validation presets.

### Implemented settings and behavior

Potato caps the renderer's effective `maxBounces` at `min(requested, 1)` (up to two scattering vertices under the
existing Cycles convention). It uses one spatial round with one partner, no disocclusion boost, RIS M=4, RR after
bounce 1, and no temporal reservoirs, dual motion vectors or duplication map. The denoiser retains colour-history
reprojection and resolve, with three à-trous levels instead of four. Textures, alpha testing, environment lighting,
geometry, primary sampling, BVH selection and release flags are unchanged. Dropping longer paths darkens indirect
lighting and loses multi-bounce specular/transmission contributions; lower sampling/reuse increases noise.

Effective defaults are derived on mode changes without rewriting the saved renderer depth or temporal option.
The reservoir-history checkbox shows the effective false value and is disabled in Potato. The depth tooltip
explains the cap; the HUD reports effective settings. Returning to interactive restores normal depth, reuse and
four denoising passes. Explicit feature/denoiser overrides still win across the session. Denoiser controls now
write only the edited field, preventing a Potato default from becoming a persistent override of another mode.
Both evaluation harnesses use mode defaults instead of silently pinning interactive denoising, and capture metadata
now reports the actual renderer mode.

### Tiny Glade talk and transcript research

Identified the original [Rendering Tiny Glades With Entirely Too Much Ray Marching](https://www.youtube.com/watch?v=jusWW2pPnA0),
Tomasz Stachowiak, Graphics Programming Conference 2024, published 2024-12-04, duration 59:20. Retrieved the English
automatic captions with yt-dlp and reviewed the lighting/denoising sections and Q&A. Captions contain recognition
errors (especially ReSTIR, BVH, SH and author names); these notes paraphrase the content and cross-check the linked
implementation/literature. Source caption data is a local research download under `validation/out/potato/`, not
vendored into the repository. Timestamp links below lead to the original talk, not a synthetic transcript.

- [11:26–15:05](https://www.youtube.com/watch?v=jusWW2pPnA0&t=686s): soft sun shadows use shadow maps and temporal
  stabilization, deriving rejection statistics in shadow-filter space. This is a different estimator from our
  per-sample visibility and should not be silently substituted into the unbiased mode.
- [15:05–20:06](https://www.youtube.com/watch?v=jusWW2pPnA0&t=905s): screen-space contact shadows use a short depth
  march. Combining point and bilinear depth tests reduces self-shadow stair steps while rejecting false hits at
  discontinuities. The [author's ray marcher](https://gist.github.com/h3r2tic/9c8356bdaefbe80b1a22ae0aaee192db)
  is available under MIT/Apache-2.0. It also demonstrates SSR fallback. The code is not copied into this change.
- [20:59–23:50](https://www.youtube.com/watch?v=jusWW2pPnA0&t=1259s): compare approximations against a reference
  renderer. Tiny Glade uses software wide-BVH traversal over coarse collision proxies, combined with screen data.
  It is not an SDF or Radiance Cascades renderer in this talk. Its proxy geometry even omits some roofs, so this
  technique cannot simply replace Sponza geometry without testing leaks and missing occluders.
- [23:53–28:34](https://www.youtube.com/watch?v=jusWW2pPnA0&t=1433s): DDGI and screen probes were tried, with density,
  placement, filtering and response problems. ReSTIR GI was also tried, then dropped because the mostly outdoor
  lighting had sufficiently low variance. Reservoir exchanges and memory traffic cost more than their benefit.
  This is scene-dependent evidence, not a claim that ReSTIR never helps interiors or many-light scenes.
- [28:35–32:32](https://www.youtube.com/watch?v=jusWW2pPnA0&t=1715s): the shipped GI traces one ray per 16 screen
  pixels, marches screen depth first, then falls back to proxy tracing on misses. Visible hits reuse screen
  radiance. Directional lighting is projected to low-order SH, reprojected and recurrently filtered with a radius
  that shrinks as history becomes reliable. AO also guides filtering near corners. Its fixed sparse GI sampling
  is distinct from dynamic resolution, but neither is implemented in our new mode.
- [33:20–35:10](https://www.youtube.com/watch?v=jusWW2pPnA0&t=2000s): water reflections compact screen-space misses
  into a separate tracing dispatch. This improves utilization for that sparse miss workload; it does not imply
  that splitting our mostly-live initial paths will help (previous wavefront experiment failed).
- [56:13–59:04](https://www.youtube.com/watch?v=jusWW2pPnA0&t=3373s): Q&A confirms the low-variance outdoor rationale
  and that gameplay uses final gathering with temporal lighting feedback, rather than full multi-bounce path
  tracing. The reference mode uses a few bounces.

### Related primary literature and what to transfer

[Dmitry Zhdan, Fast Denoising With Self-Stabilizing Recurrent Blurs, 2020](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22699-fast-denoising-with-self-stabilizing-recurrent-blurs.pdf),
especially slides 49–54: sparse recurrent filtering, history-dependent radius and disocclusion reconstruction;
SH helps retain normal-map lighting detail. Temporal stabilization is separate from accumulation; layering temporal
accumulators can create lag. Our denoiser already feeds its first spatial level back into history, so merely
calling it recurrent would not be a new optimization. A smaller adaptive filter needs first-frame, disocclusion,
light-change and glossy-motion tests. Dropping the fourth pass is the measured first step here, not an NRD port.

[Tiago Sousa, Fast as Hell: idTech8 Global Illumination, SIGGRAPH 2025](https://advances.realtimerendering.com/s2025/content/SOUSA_SIGGRAPH_2025_Final.pdf),
slides 15–22: separate visibility queries from cached shading, update irradiance volumes in an interleaved schedule,
and shade roughly 20k active radiance-cache entries per frame. Final gathering queries screen radiance first, then
world radiance and irradiance caches, rather than shading every hit again. SH and bilateral filtering reconstruct
lighting. The hardware-RT and lower-resolution GI timings are not predictions for this full-resolution WebGPU
software tracer. Transfer the separation of expensive shading from visibility and reuse of shaded results.

[AMD GI-1.0, Boissé et al., 2022](https://gpuopen.com/download/GPUOpen2022_GI1_0.pdf) maintains incoming radiance in
screen probes and outgoing radiance in a persistent world cache. It offers a concrete placement, reprojection
and cache design for a later approximate GI mode. It also illustrates why that work is more than changing a BVH
builder: the shading signal and reuse architecture change. The published Sponza timing is on RX 6900 XT with
hardware tracing and cannot be compared directly to this laptop's WGSL traversal.

### Next architectural experiment

The next substantial opportunity is a separate approximate indirect-light gather: use the full-resolution
G-buffer, march screen depth for nearby diffuse hits, then compact misses and trace the existing BVH. Fetch
previously shaded radiance only when depth/normal/material validity succeeds; fall back to real shading otherwise.
Keep primary visibility and direct-light shadows accurate, and keep glossy/transmission on the existing path until
there is a suitable directional cache. Start with full-resolution gathering so its benefit is measurable without
changing sampling density. Instrument screen-hit rate, fallback occupancy, shading work and disocclusion recovery.
Only then evaluate a fixed sparse GI grid and SH reconstruction as an explicit Potato quality option.

A cache cannot be inserted into the existing ReSTIR path suffix while retaining its current unbiased target,
weights and Jacobians. This belongs to the approximate mode with its own error/response tests. No screen marcher,
proxy scene, lighting cache or SH reconstruction is claimed as implemented by this preset change.

### Measured outcome (M5 Pro / Chrome 155 / Metal 3)

Same-session fixed-resolution Sponza measurements: 96 measured frames after 32 warmup frames, released flags,
Mode B, MT intersections, current TypeScript CWBVH builder. 540p and 720p use ABBA (two runs per mode); 1080p
is one paired run. Times are pipelined harness throughput, including denoising, excluding the app UI and display
pacing; equivalent FPS is not a claim of measured whole-app presentation rate.

| Resolution | Interactive ms / FPS | Potato ms / FPS | Speedup |
|---|---:|---:|---:|
| 540p | 43.853 / 22.80 | 17.348 / 57.64 | 2.53× |
| 720p | 77.655 / 12.88 | 30.208 / 33.10 | 2.57× |
| 1080p | 174.817 / 5.72 | 67.166 / 14.89 | 2.60× |

540p motion-throughput probes: camera pan 17.519 ms (57.08 FPS), moving light 17.335 ms (57.69 FPS).
These are different motion schedules from the quality captures, not timings of the exported videos.

Screening measured the combined changes, rather than adding individual ablation savings: shallow paths plus
one partner and RIS 8 with temporal reuse took 25.11 ms; removing reservoirs brought that to 18.78 ms; RIS 4
brought it to 18.36 ms; earlier roulette to 17.72 ms. Three denoising passes brought it to 17.34 ms. A second
spatial partner cost about 0.8 ms; keeping two bounces cost roughly 8 ms. RIS 1 saved only about 0.5 ms against
RIS 4 before the roulette reduction, so four candidates were retained. Three versus four denoising passes had
small mixed quality differences (one seed): static FLIP .0961 vs .0967, camera .1288 vs .1275, light .1019 vs .1023.

The final 540p isolated rerun still spends roughly 11.3 ms on initial paths, 1.7 ms on primary geometry,
0.9 ms on RIS, and 1.9 ms on denoising. Reservoir temporal passes are absent. These split-submit pass costs
are diagnostics and must not be added to reconstruct pipelined frame time. Initial tracing remains the main target.

### Quality evidence and validation

12 final captures: static / camera oscillation ±0.5 m / light oscillation ±1 m, each with seeds 1–4, 64 frames
at 960×540. Compare frames 16/32/48/63 at the reference pose against the saved 12,288 spp three-bounce PT
reference (relative-MSE noise floor .000322). Matched interactive captures from §8 are reused; shipped golden
checks and fresh timing controls confirm the ordinary mode is unchanged. Captures are rgba32float; timing uses
rgba16float like the app. Display transform is exposure 1, clamp and sRGB for both modes.

| Sequence | Interactive FLIP | Potato FLIP | Interactive relative MSE | Potato relative MSE |
|---|---:|---:|---:|---:|
| static | 0.06716 | 0.09625 | 0.09164 | 0.08910 |
| pan | 0.10011 | 0.12875 | 0.21445 | 0.11929 |
| light | 0.11073 | 0.10168 | 2.92109 | 0.87452 |

This is a perceptible quality trade: FLIP is about 43% higher static and 29% higher in camera motion.
Moving-light FLIP is about 8% lower. Relative MSE can improve while perceptual error worsens because it is
sensitive to the baseline fireflies; it is not evidence of equal quality. Qualitatively, interior fill is darker and
more motion noise remains. The shorter paths particularly limit multi-bounce reflections and transmission.

Validation: 602 CPU tests passed (7 skipped), the dedicated GPU mode-switch regression passed, all 10 shipped
rendering golden checks passed (26 unrelated tests skipped), and the production build passed. The mode-switch
regression traverses interactive → Potato → Offline → Potato → interactive and compares returned modes with
fresh renderer outputs pixel for pixel. 11 screening + 12 final performance jobs and 6 screening + 12 final
quality captures passed. A Chrome UI check also confirmed interactive startup, actual dropdown switches,
image controls and video playback without WebGPU/page errors. This is not a new Stage-B unbiasedness
certification for the approximate mode.

Reproducible measurements, exact job manifests, release flags, effective renderer/denoiser settings, timing blocks
and per-seed image metrics: [`potato-results.json`](potato-results.json). Run `validation/tools/potato-review.py`
with the listed local captures to regenerate figures and videos. The local viewer is
`validation/out/potato/review.html`, with `static-comparison.png`, `pan.mp4` and `light.mp4`; the videos are
illustrative playback of captured frames, not real-time screen recordings.

Transcript provenance: yt-dlp English automatic captions for `jusWW2pPnA0`; fetched 2026-10-10. JSON3 SHA-256: `756fca9ea5669f190d704d322651870b369f787c1a5095c8818f117d2b0d2dad`.


## 10. Potato: direct ReSTIR and cached indirect lighting

**Decision: opt-in, not a replacement for the default Potato renderer.** Select **ReSTIR → Potato indirect →
cached 1/4 (experimental)** while in Potato. The 1/16 option is faster but less stable. Interactive remains the
launch default. The ordinary/unbiased estimators and default Potato settings remain unchanged. No dynamic resolution.

The user requested direct-light ReSTIR plus cached indirect, prioritizing speed and visual fidelity over accuracy.
The implementation separates full-resolution direct reservoirs from sparse indirect path estimates. Sponza benefits
from the reduced tracing cost, but cross-scene screening found unacceptable quality regressions for a universal default.

### Signal separation and reconstruction

- Primary geometry, direct-light RIS M=4 and one spatial partner remain full resolution. No temporal reservoirs.
  Only direct candidates (path length 2) enter the reservoir and its weight sum. One pixel per fixed 2×2 tile
  continues to the second scattering vertex; the selected pixel rotates over four seed indices. An optional 4×4
  variant rotates over sixteen. The existing max-bounces cap of 1 and RR-after-1 remain.
- Indirect NEE, environment/triangle endings and analytic-light crossings accumulate weighted path contributions
  outside the direct reservoir. Alpha distinguishes a traced black sample from a missing update. There is no
  multiplication by the sampling interval. The cache demodulates by albedo, reconstructs from nine neighbouring
  tiles, validates reprojected depth/normal history, then applies recurrent spatial blur.
- History caps at 32 frames, minimum update weight 0.05. Stable indirect outliers are clamped to four times cached
  luminance (minimum 0.02). Lighting changes bypass this cap and raise the update weight to 0.2. Identical environment
  uploads do not count as edits. Newly exposed surfaces reconstruct locally or start with zero confidence.
- The cache is combined with direct illumination after reservoir resampling. Its optional denoiser profile uses
  two à-trous levels, alphaMin=0.08, sigmaN=32, sigmaA=0.2, sigmaL=12, no luminance prefilter or converged guide,
  camera resolve cap 12/gamma 1.5, and light resolve cap 16. Explicit overrides win. Cache off restores the previous
  three-level Potato profile. Disabling denoising restores per-pixel indirect tracing. Cache textures allocate lazily;
  mode/stride changes invalidate history, and all textures are released on resize/destruction.

This is a **screen-space cache of one-bounce path estimates**, inspired by the sparse sampling and recurrent
filtering in the [Tiny Glade talk at 28:35](https://www.youtube.com/watch?v=jusWW2pPnA0&t=1715s) and
[Zhdan's recurrent blur presentation](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22699-fast-denoising-with-self-stabilizing-recurrent-blurs.pdf).
It does not implement Tiny Glade's screen-space ray march, proxy BVH fallback, SH representation, or radiance reuse
at secondary hits for extra bounces. Tiny Glade removed GI ReSTIR in its outdoor renderer; our direct-light ReSTIR stays.

### Performance and visual results

Fixed-resolution Sponza: Chrome 155 / Metal 3 / Apple M5 Pro. 96 measured frames after 32 warmup; eight isolated-pass
frames and eight latency samples. Old/new/new/old at 540p and 720p; one old/new pair at 1080p. The control explicitly
pins previous Potato settings. Cache numbers below use **1/4 sampling**, not the faster rejected 1/16 prototype.

| Resolution | Default Potato | Optional cached indirect | Equivalent throughput, old → new |
|---|---:|---:|---:|
| 960×540 | 17.34 ms | 14.23 ms | 57.7 → 70.3 FPS |
| 1280×720 | 30.31 ms | 24.74 ms | 33.0 → 40.4 FPS |
| 1920×1080 | 67.20 ms | 54.61 ms | 14.9 → 18.3 FPS |

Approximately 18–19% less GPU-harness frame time, excluding presentation, UI and display pacing. These are not
measured application FPS. Initial path tracing remains the largest cost. The earlier 1/16 prototype reached about
81/47/21 FPS but was less stable. Direct-only tracing reached about 7.9 ms at 540p but removed too much interior fill.

Four matched seeds, frames 16/32/48/63: Sponza display-space variability falls about 7% during camera motion and 8%
with moving light, but rises 4% when static; the dark-interior ROI gets worse in all three cases. FLIP against the
12288-spp reference changes from 0.096→0.136 static, 0.129→0.156 pan, and 0.102→0.112 moving light. Less variability
alone does not prove accuracy or absence of lag. Smoothing attenuates glossy response and isolated indirect highlights.

A further **48 captures** cover Cornell, normal-mapped glossy objects, glass/light crossings and alpha foliage,
with static camera, camera motion and moving analytic light, two seeds each at 384×384. All are finite and free of GPU
errors, but cached output has more across-seed variation in every screening category: Cornell static 0.0030→0.0075,
Cornell pan 0.0110→0.0261, glossy moving-light 0.0052→0.0098. Images show mottling and loss of glossy/faceted detail.
Those regressions are why caching remains explicitly experimental and off by default. Scene metrics and manifests:
[`potato-scenes-results.json`](potato-scenes-results.json).

Camera disocclusion uses geometry validation and local reconstruction; illumination can initially be missing or lag.
Moving lights refresh the cache, but sparse samples can produce mottling. An 80-frame power-halving/light-removal
sequence also completes with finite output. The harness now waits for Mode-A/B shader changes instead of capturing
an undenoised compilation fallback. This excludes first-use compilation latency from that recovery test. The current
scene bridge supports camera, analytic-light and environment animation, **not skinned/deforming/moving mesh animation**;
this work does not certify cached GI for animated geometry. Full mesh replacement resets the renderer.

### Reproduction and verification

`validation/tools/potato-cache-review.py` regenerates Sponza comparisons/videos from four-seed captures.
[`potato-cache-results.json`](potato-cache-results.json) records effective settings, hardware, exact timings, reference
errors and variability. Manifests explicitly pin cache stride after the opt-in decision. Cross-scene sheets come from
`validation/tools/potato-scenes-review.py`. Local viewers/artifacts: `validation/out/potato2/review.html`,
`static-comparison.png`, `pan.mp4`, `light.mp4`, and `{cornell,glossy,glass,foliage}-comparison.png`. Videos are illustrative
capture playback, not realtime performance recordings. Cached raw PFMs contain **direct/emission only**; complete
renderings are denoised PFMs, identified in metadata.

GPU tests cover direct-reservoir isolation, reconstruction energy, black/missing samples, lighting response,
depth/normal rejection, reset/held frames and mode switches. CPU tests check opt-in/default restoration and uniform
packing. Ten shipped golden tests retain their expected outputs. Actual app controls verify default interactive,
per-pixel Potato, cache opt-in, denoise off/on, return to interactive, and FSR with canvas resizing. This is not an
unbiasedness certification for Potato.

## 11. FSR 1 spatial upscaling

**FSR 1 (EASU + RCAS)** is the launch default, as requested after the comparison. Change it in
**Render → Output → upscale filter**. Internal resolution stays at the selected fixed
540p/720p/1080p preset; dynamic resolution is not enabled. Bicubic remains available. The FSR sharpening slider
sets linear RCAS strength from zero (off) to one, default 0.5. FSR works independently of the indirect cache and in
all rendering modes.

This is a WGSL FP32 port of AMD's [FSR 1 EASU and RCAS](https://github.com/GPUOpen-Effects/FidelityFX-FSR/blob/master/ffx-fsr/ffx_fsr1.h),
v1.20210629, with the MIT license retained in both shader files. EASU is a twelve-tap directionally adaptive Lanczos
approximation with nearest-2×2 anti-ringing bounds; it is more than a fixed Lanczos resize. Tone mapping runs at
internal resolution, EASU writes an output-sized RGBA16 texture, and RCAS runs in the final blit. This avoids an extra
copy pass. The port uses clamped scalar texture loads and guarded exact reciprocals instead of AMD's gather/reciprocal
approximations; RCAS enables AMD's optional noise attenuation. It is an algorithm port, not bit-identical to AMD's
FP16 assembly. UI, probe marks and raster overlays are composed afterward. Debug views retain their existing filter.
Native-size output bypasses FSR; downscaling or a mixed up/down axis uses bicubic. It requires neither motion vectors
nor a separate temporal history, so it adds no temporal ghosting of its own.

`validation/gpu-tests/fsr.gpu.test.ts` compares output with an independent scalar CPU evaluation, including FP16
intermediate quantization, within one 8-bit level. Tests cover black/white endpoints, odd dimensions, sharpening
0/0.5/1, resizing, native/downscale bypass, debug split, and forcing the required pre-tone path. Existing presentation
regressions also pass. The EASU pass has its own GPU HUD timestamp; RCAS is included in `present`.

On the same M5 Pro, isolated tone-map + upscale + present timestamps at 1920×1080 output:

| Internal resolution | Bicubic | FSR 1 | Additional presentation work |
|---|---:|---:|---:|
| 960×540 | 0.48 ms | 0.82 ms | 0.34 ms |
| 1280×720 | 0.44 ms | 1.02 ms | 0.58 ms |

These are means of ABBA runs, 32 samples per run after 16 warmup. GPU scheduling varies (individual bicubic runs
0.33–0.64 ms); do not interpret the last decimal as reliable or add these isolated sums to predict app FPS exactly.
This filter costs more than bicubic at the same internal resolution; its purpose is better presentation of a lower
fixed resolution. It cannot recover missing rendered detail, remove all shimmer, or replace temporal supersampling.
The frozen 540p Sponza comparison is subtly sharper at arches/cloth edges but also sharpens residual noise. 720p timing
uses a resized copy of the same input solely for cost measurement, not a 720p visual-quality claim.

Reproduce with `npx tsx validation/harness/with-gpu-lock.ts fsr-review -- node validation/harness/fsr-review.mjs`
after generating the Sponza capture. Local output: `validation/out/fsr/review.html`, `bicubic.png`, `fsr1.png`.
Recorded timing evidence: [`fsr-results.json`](fsr-results.json). A temporal upscaler would require a separate integration
and quality study; it is not included in this spatial implementation.


## 12. Reducing Potato cache mottling: production research and proposed experiments

Research requested after the FSR default change. **The experiments below are proposals, not implemented or measured
improvements.** Sources were checked on 2026-10-11. FSR now starts at strength 0.5; the actual Chrome startup/mode/resize
check and TypeScript check pass. Cached GI remains opt-in.

### What the current implementation actually does

`pathtree.wgsl::pt_emit` collects all indirect contributions into one RGB signal; it does not separate diffuse from
specular at the primary hit. `dn-indirect.wgsl` divides that signal by the shared albedo/Fresnel demodulation factor,
reconstructs nine neighbouring tile samples, temporally blends, and feeds spatially blurred values back into history.
That factor is a heuristic for the mixed signal, not a physically isolated diffuse irradiance cache.

The selected pixel has the same rotating phase in every N×N tile. Cache alpha records history age rather than a
variance estimate or effective independent sample count. Normal/depth are validated, but neither roughness/material
class nor secondary-ray hit distance guides this cache filter. The recurrent blur steps by N, leaving same-parity
pixel neighbourhoods. Any analytic-light/environment parameter change raises the update weight from 0.05 to 0.2
and disables the stable-history bright-sample cap across the entire image. The combined direct+indirect image then
passes through a second temporal/spatial denoiser.

**Working diagnosis:** sparse high-variance rays are being converted to spatially correlated patches, then retained
in feedback. Mixing view-dependent glossy samples with diffuse lighting makes those patches harder to filter safely.
Motion loses useful history; global lighting refresh exposes noisy estimates even in locally unchanged regions.
The rotating sample pattern and same-parity blur may contribute additional instability. These are code-based
hypotheses; isolate them with ablations before calling any one the dominant cause. Existing comparisons establish
that the problem is real, not which change fixes it (§10).

### What shipped games teach us

- **Fortnite Chapter 4 / Lumen:** Epic reports that spatiotemporal blue noise substantially reduced indirect-light
  noise without extra rays. It also removed an aliasing source by skipping foliage hits in screen traces.
  [Epic's production write-up](https://www.unrealengine.com/tech-blog/lumen-brings-real-time-global-illumination-to-fortnite-battle-royale-chapter-4?lang=en-US).
  This supports testing better sampling patterns and treating troublesome material classes explicitly, not assuming
  every surface should share the same sparse-lighting policy.
- **DOOM: The Dark Ages / idTech 8:** the GHOST talk's final gather reads screen/world/irradiance caches, stores
  directional lighting in two-band SH, and filters with normal, depth and ray-hit-distance weights. Direct lighting
  is added after this filtering; directional occlusion restores local detail. Reflection handling is separate.
  [Developer slides, 21–29](https://advances.realtimerendering.com/s2025/content/SOUSA_SIGGRAPH_2025_Final.pdf).
  The relevant lesson is to smooth a low-frequency indirect signal while reconstructing directional/material detail
  afterward. Its hardware-RT timings are not transferable to our software BVH in WebGPU.
- **Quake II path tracing:** the developers' filtering walkthrough explains why raw inter-frame differences are
  themselves noisy and why temporal adaptation needs a better change estimate.
  [Developer video series and explanation](https://developer.nvidia.com/blog/?p=14548).
  This argues against replacing our global light flag with a naive per-pixel difference threshold.

Two implementation references strengthen those lessons without being claims about a specific game's code:

- [NVIDIA NRD integration guide](https://github.com/NVIDIA-RTX/NRD): separate diffuse/specular at the primary hit,
  supply meaningful indirect hit distances, and use short-history information to detect lag. It specifically warns
  that rotating the chosen pixel in a 2×2 footprint can destabilize guides and increase disocclusion. Our full-resolution
  guides differ from NRD's quarter-resolution setup, so this is an experiment to run, not proof that our pattern is wrong.
- [AMD GI-1.0, §§2.1.2 and 2.4](https://gpuopen.com/download/GPUOpen2022_GI1_0.pdf): redistribute a fixed ray budget
  from successfully reprojected tiles to holes; use SH irradiance and history-dependent spatial filtering. This directly
  addresses missing-history regions without changing image resolution. It is a research implementation, not evidence
  that we can reproduce its quality or timings with a few parameter changes.

### Experiment order

1. **Instrument the failure before widening blur.** Add cache-only views for variance, valid history, rejection reason,
   sample phase and lighting-change response. Compare feedback on/off and cross-phase vs current N-stride blur.
   Existing across-seed deviation measures uncertainty; add motion-compensated frame differences and light-step
   settling curves to distinguish boiling from legitimate illumination changes.
2. **Stabilize the sampler at unchanged ray count.** Compare persistent representative surface samples with the
   rotating grid. Use vector spatiotemporal blue-noise dimensions for Potato's indirect direction samples, with an
   explicit dimension mapping that leaves direct reservoirs and ordinary-mode RNG/replay untouched. Do not hash the
   mask values again or expect simply randomizing the active pixel to preserve their spectral properties.
   [NVIDIA's sampling explanation](https://developer.nvidia.com/blog/rendering-in-real-time-with-spatiotemporal-blue-noise-textures-part-2/)
   also notes reduced temporal benefit under motion: this is not a complete disocclusion solution.
3. **Cache diffuse lighting separately.** Keep sharp/specular/transmission paths outside broad diffuse filtering;
   initially use a roughness/material gate, then split the actual primary BSDF contributions. Carry first-indirect-hit
   distance and confidence. Consider SH only after the split; converting arbitrary mixed RGB to SH is not the algorithm.
   This is the strongest architectural candidate for preserving glossy detail while smoothing walls.
4. **Make reconstruction depend on confidence and local change.** Track moments/effective confidence, filter uncertain
   diffuse samples more broadly with geometry/hit-distance gates, and retain sharper mature estimates. Compare short
   and long histories for statistically significant changes. Keep a robust outlier limiter during moving-light frames,
   with a fast response for real changes; do not globally expose all noise whenever one light moves. Increase local ray
   allocation for disocclusions under a fixed total budget. Both extra memory and scheduling cost must be measured.
5. **Use a persistent fallback if camera motion still dominates.** A small world-space irradiance/probe cache or
   screen-space ray hit backed by cached world lighting can retain information through occlusion. This is substantially
   more work, with leakage and update-latency risks. Stronger filtering alone cannot supply unseen lighting.

The quickest useful A/B is sampler/feedback/lighting-change control. The most promising structural change is diffuse
separation plus confidence-guided reconstruction. Keep FSR at the same strength in comparisons, and also inspect
before sharpening: RCAS can reveal residual noise but is not its source. Require comparisons at static positions,
mid-pan/disocclusion (not only the return-to-origin knots), moving lights and abrupt light removal. Use Sponza,
Cornell, glossy, glass and alpha foliage, at least four seeds for the final candidate, and assess lag and detail as well
as variance. A faster renderer that hides noise by erasing glossy detail does not meet the stated visual-fidelity goal.

### Additional performance opportunity

The measured 540p initial path pass is about 11.3 ms in default Potato, versus about 1.9 ms for the separately timed
denoiser (§10); these isolated measurements are diagnostic, not additive frame-time accounting. A dedicated
**ReSTIR DI implementation** for cached Potato could remove generic path-resampling/replay state and use compact
light reservoirs, optionally adding validated temporal light-sample reuse. [RTXDI's integration guide](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/Integration.md)
distinguishes this direct-light sampling architecture from ReSTIR PT. That is a larger potential speed target than
shaving another blur pass, but its benefit in this code is unmeasured. Reuse of stale shadow visibility would need
separate moving-light/camera tests and should never silently enter unbiased modes.


## 13. Potato cache: stability over accuracy

The user's clarified priority is a stable, pleasant image and speed, accepting bias and lost accuracy. This changes
how we judge Potato; normal/unbiased modes retain their existing behaviour. Cached GI remains an explicit choice in
**ReSTIR → Potato indirect → cached 1/4**. FSR remains the default presentation filter.

Implemented a combined cache-only experiment:

- Before sparse spatial reconstruction, weight each demodulated estimate by `1 / (1 + 4 * luminance)` in addition
  to the existing geometry/spatial weights. This robust mean suppresses rare bright paths before they spread into
  correlated patches. Constant neighbourhoods retain their value, but uneven lighting loses energy deliberately.
- Recurrent blur uses stride `N - 1` instead of `N`, mixing sampling phases through feedback. Tap count stays nine;
  primary geometry, direct lighting, ray count, resolution, buffers and pass count are unchanged.
- Keep the bright-sample limiter active while lights move, with an allowance of `2 * history luminance + 0.05`.
  The additive term allows dark surfaces to relight. Temporal update rates remain 0.05 static / 0.2 lighting changes;
  history age still affects young pixels. This is an outlier-control heuristic, not a local lighting-change detector.

These changes were measured together, so the evidence establishes the combined effect, not each change's contribution.
The stronger bias does not preserve radiometric accuracy and can diminish real glossy highlights/colour bleeding.

### Matched results

Four seeds × five scenes × static/camera/light motion × before/after, plus two light-step sequences: **122 successful
capture jobs**. All seed-1 jobs export every frame, including peak camera/light offsets (frame 20, not just the
return-to-origin knots). Other seeds export frames 16/32/48/63. All metrics use complete denoised images; cached raw
exports only contain direct light. Captures omit FSR so sharpening cannot confound the comparison.

| Scene | Static flicker reduction | Across-seed deviation reduction (static / camera / light) | Full-image linear brightness reduction |
|---|---:|---:|---:|
| Sponza | 27.4% | 30.0% / 29.0% / 25.4% | 5.8–10.0% |
| Cornell | 33.5% | 34.7% / 32.2% / 39.0% | 16.2–17.5% |
| Glossy | 33.5% | 33.1% / 29.2% / 33.1% | 15.3–17.2% |
| Glass | 41.3% | 52.3% / 43.6% / 42.7% | 17.6–21.0% |
| Alpha foliage | 4.8% | 7.8% / 4.5% / 2.7% | about 0.7% |

“Static flicker” is mean absolute frame-to-frame change of display-space 8×8 block means, frames 16–63, one seed.
Across-seed deviation uses four matched seeds at the four listed frames. These metrics include the benefit of lower
brightness and do not establish accuracy. No motion-compensated temporal score is claimed. Camera/light sequences
are visual screens; static meshes only. Screens show reduced wall/ceiling mottling, with visibly darker bounce light.
Fast Sponza pans still reveal noisy disocclusions, and glossy detail remains approximate. This is an improvement to
the existing opt-in cache, not grounds to call all Potato motion artifacts solved or to promote it automatically.

Two sequential timing runs per configuration on M5 Pro / Chrome155 / Metal3:

| Resolution | Previous cache | Stabilized cache |
|---|---:|---:|
| 960×540 | 14.23 ms | 14.23 ms (~70.3 GPU FPS) |
| 1280×720 | 24.73 ms | 24.77 ms (~40.4 GPU FPS) |

Presentation/UI are excluded. Timing differences are effectively negligible in this screen; this was not interleaved
ABBA and is not a speedup claim. 1080p was not remeasured for this shader revision.

### Response checks and reproduction

The expanded GPU test verifies uniform-signal preservation, isolated bright-sample suppression, capped lighting
spikes, black-vs-missing samples, history rejection, held-frame immutability and bounded recovery. In the synthetic
cache-only test, a black→2 step reaches >98% by 40 lighting updates; switching to black removes >97% within 16 updates.
Those bounds are not immediate response: at 30 FPS, 40 frames is 1.33 s. Real combined rendering has separately updated
direct illumination. An 80-frame scene sequence also halves analytic power at 32 and removes its last analytic light
at 56; emissive geometry remains. Its complete-image mean is not an isolated GI settling measurement.

Validation: 13 GPU tests pass across denoiser and Potato-mode suites. Production build passes. Normal modes' paths
and settings were not modified. The shader uses the same inputs and resource layouts.

- Evidence and capture manifests: [`potato-stability-results.json`](potato-stability-results.json).
- Analysis: `validation/.venv/bin/python validation/tools/potato-stability-review.py` after captures.
- Local review: `validation/out/potato3/review.html`; Sponza/glossy static, pan and moving-light MP4 comparisons.
  Video playback is slowed to 15 FPS for inspection, not a performance recording.
- `before` captures use commit 22df07e's cache shader; `after` uses this section's implementation. Capture manifests
  are embedded in the evidence JSON. Run with `run-denoise.ts --jobs FILE --lock-per-job`; timing manifests can be
  reconstructed from each report's options in that evidence.

The next structural improvement remains separating diffuse cache data from view-dependent glossy/transmitted paths
and improving the fallback for newly visible surfaces (§12). Merely extending history would trade boiling for trails.


## 14. Gentle reveal of new indirect lighting

Requested after the stability experiment: fade newly visible areas in more gently. Implemented on top of
`fb1b6f0` for cached Potato GI, without changing direct lighting, geometry, cache radiance or other rendering modes.
At combination, multiply indirect lighting by `smoothstep(1, 17, valid history age)`. History stores the complete
estimate, avoiding recursive attenuation. Age 1 contributes no GI, age 9 contributes half, age 17 contributes all.
The ramp takes 16 successful updates: about 0.27 s at 60 FPS or 0.53 s at 30 FPS. It is frame-based, not wall-clock-based.
Missing samples preserve age; rejected geometry/reset starts again. A lighting change alone does not restart it.
The existing final denoiser can soften and lengthen the visible transition.

Visual comparison: 15 matched seed-1 jobs across Sponza, Cornell, glossy, glass and foliage, covering stationary,
camera and analytic-light motion, all 64 frames exported. Sponza's newly exposed side walls show far fewer bright
patches during pans. The tradeoff is deliberate temporary darkening: new surfaces show direct illumination first.
This cannot hide direct-light noise or provide correct unseen lighting, and fast motion can keep some areas dim.

Tests verify the ramp's endpoints and midpoint, immediate direct/background visibility, unattenuated cache history,
no restart on a lighting change, and restart after rejected depth. All 13 relevant GPU tests and the production build
pass. GPU throughput remains about 14.21 ms at 540p and 24.72 ms at 720p (two sequential runs each; presentation excluded).
No extra pass, ray, resource or dynamic-resolution change was introduced.

Long stationary screens at 64/128/256 frames revealed a limitation of applying the fade before the existing final
resolve: its long averaging history retains part of the darker startup. By frame 255, whole-image mean brightness
is still 0.8% lower in Sponza, 4.9% in Cornell and 3.8% in glossy than the unfaded stabilized cache. This decreased over
the sequence, but full-image brightness is not restored immediately at update 17. Keep this explicit rather than
claiming a strict half-second end-to-end response. The cache-only synthetic test verifies that its radiance itself
is not repeatedly attenuated. Changing the final resolve's history cap would require a separate stability tradeoff.

Evidence: [`potato-fade-results.json`](potato-fade-results.json), including matched manifests, long static differences
and final timing reports. Reproduce the review with `validation/.venv/bin/python validation/tools/potato-fade-review.py`.
Local output: `validation/out/potato3/fade-review.html` and `sponza-pan-fade.mp4` (left: stabilized unfaded cache; right:
fade enabled). Videos are slowed to 15 FPS for inspection. This is a one-seed visual screen, not a new four-seed
stability claim; §13's statistics describe the preceding robust-filter change.

## 15. Persistent diffuse lighting, screen lookup and blue noise

User priority: stable, pleasant biased lighting; retain fixed resolution. Cached Potato now separates diffuse and
view-dependent indirect paths. The primary sampled lobe chooses the signal: diffuse remains sparse; reflection /
transmission continuations run per pixel and bypass the diffuse reconstruction and reveal. They still use the final
image denoiser. Diffuse demodulation uses the actual Lambert closure weight, so metals do not receive cached diffuse
lighting. Direct ReSTIR candidates remain independent of the indirect estimator.

The persistent fallback is a **surface-oriented RGB diffuse cache**, not a full GIBS implementation or a probe volume.
It has 32,768 direct-mapped entries, with complete cell tags, normal and plane-distance checks, floating-origin-aware
positions and a 120-update expiry. Mature visible estimates populate it; deterministic integer selection avoids
floating-point atomic races. Separate read/write tables make held frames and timing replays deterministic. Memory is
4 MiB for the tables plus 128 KiB for selection, alongside the two new full-resolution RGBA16F reflection/guide textures.
On disocclusion, matching world lighting fills the display while fresh screen history builds. It does not feed itself
back through the screen filter. Previously unseen surfaces still fade from direct lighting. Hash collisions reduce
coverage rather than accepting a different tagged surface.

**Dynamic lights:** any actual analytic-light or environment change advances a lighting epoch, immediately rejecting
older world entries. Visible diffuse history keeps the existing faster lighting response and bright-sample limiter;
it can still lag. New world entries inherit that filtered estimate, so this does not promise instantaneous indirect
relighting. A power change and subsequent light removal set the change flag exactly at frames 32 and 56 in the 80-frame
recovery capture. Camera movement retains world entries. Scene/history resets clear them. Animated mesh deformation
is not supported by this application path; no dynamic-mesh validation is claimed. A production surfel/probe updater
would budget fresh rays across affected regions, shorten history after significant changes, and revalidate moved
geometry. See [GIBS](https://www.ea.com/seed/amp/news/siggraph21-global-illumination-surfels) and
[RTXGI controls](https://github.com/NVIDIAGameWorks/RTXGI-DDGI/blob/main/rtxgi-sdk/include/rtxgi/ddgi/DDGIVolume.h).

Two experiments are exposed under **ReSTIR → Potato cached lighting**, both off by default:

- **Screen lookup:** eight projected probes into the current visibility buffer, actual triangle/alpha intersection,
  an any-hit BVH check against intervening geometry, then ordinary closest-hit fallback when uncertain. This is a
  candidate-lookup shortcut, not screen-radiance reuse or a hierarchical depth marcher. Sponza costs about **16.47 ms**
  instead of **15.48 ms** at 540p. It is not promoted as an optimization; measured display differences are negligible.
- **Blue-noise GI:** a bundled 64×64 RGBA CC0 texture by Christoph Peters changes secondary BSDF direction coordinates.
  Primary/direct sampling is untouched. Whole-tile translations preserve spatial correlation, and diffuse samples
  address the sparse tile grid. Per-pixel reflection paths address the full pixel grid. This is **spatial blue noise,
  not STBN**. It costs about **15.59 ms** in the same ablation. One-seed cross-scene results do not establish a general
  stability win, so it remains optional. Source, pinned revision, hashes and license are in `restir/noise/README.md`.
  Background: [Moments in Graphics](https://momentsingraphics.de/BlueNoise.html).

Final M5 Pro / Chrome 155 GPU renderer-and-denoiser timings, 96 measured frames after 32 warmup frames, two cache runs
at each resolution (presentation, UI and browser scheduling excluded):

| Fixed internal resolution | Cached diffuse + world fallback | GPU FPS | Per-pixel Potato |
|---|---:|---:|---:|
| 960×540 | 15.46 ms | 64.7 | 17.33 ms |
| 1280×720 | 26.90 ms | 37.2 | 30.27 ms |
| 1920×1080 | 59.31 ms | 16.9 | not rerun |

The preceding mixed diffuse/specular cache was faster (14.21 / 24.72 ms at 540p / 720p). This change buys separation
and persistent diffuse fallback, not a net speedup over that cache. The world passes alone cost about 0.13 ms at 540p;
the total world-on/off ablation differs by about 0.15 ms. Most added cost comes from per-pixel specular continuations.
FSR remains the fixed-resolution presentation default; no dynamic resolution is introduced.

Visual screening: 15 matched scene/motion captures against 803231e (Sponza, Cornell, glossy, glass, foliage; static,
camera motion and moving analytic light), 15 blue-noise ablations, three screen-lookup Sponza ablations and one light-step
sequence. All images are finite. In this one-seed screen, static glossy/glass frame changes fall roughly 21% / 17%,
while Sponza is roughly unchanged (+1%) and Cornell increases 8%. Full-image mean brightness is about 3–5% higher in
most static scenes; foliage is essentially unchanged. Motion frame differences include changing geometry and are not
a motion-compensated flicker metric. These mixed results do not establish a general stability improvement.

Validation: production build; 603 CPU tests (7 skipped); all 16 denoiser/Potato GPU tests; all 10 shipped GPU identity
cases. Added checks cover diffuse remodulation, specular bypass of the fade, screen/blue shader variants, world reuse,
lighting invalidation, opposite normals, nearby parallel surfaces, expiry, reset and held/timing repeatability.
Ordinary shader-text goldens remain unchanged. Per-pixel Potato remains the default; persistent diffuse is enabled
when the user selects cached Potato. Interactive ReSTIR remains the startup mode.

Evidence: [`potato-world-results.json`](potato-world-results.json), including manifests, all timing reports, source
texture spectral checks and light-step metadata. Review scripts: `validation/tools/potato-world-review.py` and
`potato-world-variants.py`. Local comparisons: `validation/out/potato4/world-review.html`, `sponza-world-pan.mp4` and
`glossy-world-pan.mp4`; videos are slowed to 15 FPS for inspection, not performance recordings.
