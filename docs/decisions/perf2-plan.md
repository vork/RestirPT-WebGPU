# perf2 optimisation plan

Nothing in the perf2 worktree was edited. The only files written are five prototype diffs, exported for engineers to port, in `/private/tmp/claude-502/-Users-mark-boss-Dev-WebGPURestirPT/4e489933-ea73-491f-b7b7-6d0aab1ffce1/scratchpad/plan-patches/`:
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
   - Never place a traversal stack at module scope on CWBVH.
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
