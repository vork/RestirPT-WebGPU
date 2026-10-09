# perf2 API: perf flags, pinned knobs, golden tiers, profiling tools (WP-0)

This note is the interface the perf2 work packages build on ([perf2-plan.md](perf2-plan.md) §0, §2 WP-0, §5). It covers
the perf-flag registry and how a package adds its flag (§1), the pinned interactive knobs and the app-default hook of the
user decisions (§2), the bits goldens and their Anchor / Shipped tiers (§3), and the profiling tools in
`validation/tools/perf/` (§4). WP-0 changed no shader: every validation text is byte-identical (U-M7-BITS) and every
existing golden holds.

## 1. Perf-flag registry

`src/core/render/restir/perf-flags.ts` holds `PERF_FLAGS`, one entry per composer define. All names are reserved now, so
the packages never collide:

| Flag | WP | Class | What |
|---|---|---|---|
| `RS_VIS_MERGE` | WP-1 | bitwise | one `trace_any_ex` call site in `shift_finish_vis` |
| `RS_REFRESH_VIS` | WP-1 | bitwise | `refresh_record` traces the per-class visibility ray once (added by WP-1, the plan's separate refresh flag) |
| `RS_RIS_HOIST` | WP-2a | bitwise | `bsdf_prepare` hoisted out of the RIS loop; dead `nee_draw` at B=1 skipped |
| `RS_NEE_SITE` | WP-2b | bitwise | one NEE site and one post-hit retest site in rs_initial |
| `RS_LAST_ANYHIT` | WP-2c | bitwise (exception: counters) | last continuation as any-hit |
| `RS_RIS_PREPASS` | WP-2d | unbiased | lean RIS pre-pass `rs_ris_nee` |
| `RS_ENV_WRAP` | WP-4a | bitwise | `envTexel` select-wrap |
| `RS_ENV_PRESAMPLE` | WP-4b | unbiased | env-presampled light tiles |
| `CW_TRI_BUDGET` | WP-3 | bitwise | CWBVH triangle budget (value = K, e.g. `CW_TRI_BUDGET=2`) |
| `BVH_ALPHA_BIT` | WP-3 | bitwise | alpha bit in the MT record |
| `BVH2_PRIV_STACK` | WP-3 | bitwise | module-private traversal stack (BVH2 only) |
| `BVH_CONST_LOOPS` | WP-3 | bitwise | constant-bound traversal loops (added by WP-3, step 3c) |
| `CW_EXP_OR` | WP-3 | bitwise | CWBVH exponent-OR child-box byte decode (added by WP-3, step 3e) |
| `RS_DENSE_SLOTS` | WP-5 | bitwise | dense spatial slot-item queue |
| `RS_BOOST_GATE` | WP-5 | bitwise | boost-slot gating |
| `RS_MIS_TRIM` | WP-5 | bitwise | spatial write-back trimming (added by WP-5) |
| `RS_PAIR_TABLE` | WP-5 | bitwise | pairing transforms per workgroup, exact fast modulo, `rs_pix` single-member fast path (added by WP-5) |
| `RS_TSEL_FOLD` | WP-6 | bitwise | T4 split / phase B folded into T4 |
| `RS_TSTATE_SOA` | WP-6 | bitwise | tState plane-major |
| `RS_AGG_COUNTERS` | WP-6 | bitwise | subgroup-aggregated counters |
| `RS_NO_PLANTS` | WP-9b | bitwise | plant code compiled out |
| `RS_NO_DIAG` | WP-9b | bitwise (exception: P9) | P9 diagnostic writes skipped |
| `MAT_VARIANTS` | WP-8 | bitwise | material compile variants |
| `RS_PRIMARY_EXT` | WP-7e | unbiased | rs_primary reads the M1 V-buffer |
| `RS_HALF_RATE` | D6 | biased | half-rate path trees (interactive toggle only) |
| `DN_GRAD_SKIP` | WP-7a | bitwise | denoiser: dn_gradient / dn_grad_filter skipped when no pass reads λ (host only) |
| `RS_DUPMAP_S64` | WP-7b | bitwise | rs_dupmap: one 64-bit seed compare per tap (memberCount 1, sentinel texels) |
| `PRIM_SKIP_BEAUTY` | WP-7c | bitwise | M1 primary: no placeholder beauty when ReSTIR / PT overwrites the colour target |
| `RS_SKIP_DISPLAY` | WP-7c | bitwise | rs_finalize_frame: no accumulation / colour store when the denoiser displays |
| `GBUF_48` | WP-7d | bitwise | lossless 48 B G-buffer (M1 primary, denoiser) |
| `DN_ZGRAD_TEX` | WP-7f | bitwise | denoiser: the à-trous depth gradient written once by dn_variance |
| `DN_COLOUR_EARLY` | WP-7f | bitwise | denoiser: the à-trous colour load before the geometric skip test |

Each entry has `wp`, `cls` (`bitwise` / `unbiased` / `biased`), optional `except` (`'P9'`, `'counters'`), `landed`
(the WGSL exists) and `release` (on in the app by default).

### Flow

- `RestirKernelOptions.perfFlags` (a `PerfFlagsInput`: `'A,B=2'`, `['A', 'B']` or `{ A: 1, B: 2 }`) is normalised once.
  Unknown names throw; `0` / `false` entries are dropped; keys are sorted.
- `RestirKernel.perfDefines()` returns the composer defines. A flag that is off is **absent**, never `NAME: 0`. The
  composer treats an undefined identifier in `#if` as false, so a flag-free kernel composes the pre-perf2 text.
- WP-7 added the names of its app-only items (the M1 primary, the interactive finalize, the denoiser). The denoiser's
  pipelines get the renderer's set too (`Denoiser.create({ perfFlags })`; the renderer recompiles the denoiser when the
  set changes).
- The defines reach every ReSTIR pipeline: `kernel.defines()` (all standard passes and stages), `customDefines()` (the
  T3 / test pipelines), the debug views (`RestirDebugPass`, compiled once at creation), and the renderer's M1 primary
  pass (`Renderer.compile`, key `format|stats|flags`).
- `variantKey()` appends `|<flags key>` only when flags are set. A flag-free key is the pre-perf2 key.
  `setPerfFlags(f)` switches the set at a frame boundary: a new variant (`await prepare()`) and a history reset
  (`frameState.invalidate('perf-flags')`).
- Defaults: `RestirKernel.create` (every validation caller) has no flags. `RestirKernel.interactive` (the app) uses
  `RELEASE_PERF_FLAGS` (the `release: true` entries; empty today).

### Where flags are set

| Place | How |
|---|---|
| App / renderer | `RendererOptions.restirKernel.perfFlags` (default `RELEASE_PERF_FLAGS`); `setOptions` switches the kernel and the primary variant |
| Perf jobs | `PerfOptions.perfFlags` per job (`perf-run.ts`); `run-perf.ts --kernel-flags A,B` for all jobs; `--abba A,B` builds same-session ABBA jobs; the report records `perfFlags` |
| Validation runs | `run-batches.ts --kernel restir … --kernel-flags A,B` (batches and chains; `restir-batch-run.ts` / `restir-chain-run.ts` `perfFlags`). Recorded as `meta.config.perfFlags` (absent without flags, so config hashes of flag-free runs are unchanged). Biased flags are refused unless `--allow-biased-flags`, and they make `t16.validationModeUnbiased` false |
| GPU tests | `VITE_PERF_FLAGS=A,B npx vitest run --project chrome …` forces flags on in every `restirRig` (the bits rigs, `VITE_STRESS`) and in the T3 kernels of `restir-shift.gpu.test.ts` / `restir-temporal.gpu.test.ts` (`testPerfFlags()`); test pipelines get them through `customDefines()` |
| Profiling | `prof-driver.ts --kernel-flags`, `spill-lint.ts --kernel-flags` |

### How a package adds its flag

1. Use the reserved name. In WGSL, gate the change with `#if NAME` … `#else` (old text) … `#endif`. A flag that also
   needs host changes (a pass, a buffer) reads `kernel.perfFlags.NAME`.
2. Set `landed: true` in `PERF_FLAGS`. The CPU test fails if a shader references a flag still marked reserved.
3. Prove it. Bitwise class: `VITE_PERF_FLAGS=NAME` on `m8-bits` (the Anchor goldens, both layouts, the CWBVH and
   Sponza-lite cases) and the other V-BIT suites, plus the spill / lint report (no growth in hot kernels). Results-changing
   class: V-UNB (Stage-B units with `--kernel-flags NAME`), Anchor unchanged, Shipped re-recorded (§3).
4. Measure with `run-perf.ts --abba NAME` and `abba.py`.
5. To ship it in the app, set `release: true` (never for a biased flag). A results-changing release flag also
   re-records the Shipped goldens, with the justification.
6. Moving a flag into the validation text is the separate, optional promotion step of perf2-plan.md §0 rule 2.

## 2. Pinned knobs and app defaults

- `INTERACTIVE_PINNED` (`presets.ts`) = `{ slots: 3, risM: 32, rrMinBounces: 3, dupmap: true }`, the interactive preset's
  values (a CPU test checks this). It is spread explicitly into the interactive cases of `m8-bits.ts`,
  `restir-tframe-bits.ts` and `restir-m6-bits.ts`, and into every baseline job of `run-perf.ts` (`pinnedJob`) and
  `gate-m8.ts`. maxBounces is pinned per case and per perf scene (`PERF_SCENES[*].maxBounces`: cornell 3, sponza 3,
  crossings 4, nm_smooth 3). Pinning changed no hash.
- `INTERACTIVE_APP_DEFAULTS` (`renderer.ts`) is the app-level override of app mode `'interactive'`, applied by
  `restirAppSettings` over the interactive preset. The presets are unchanged (WP-10 rule).
  - User decision D3 lands as `{ dupmap: false }` here, together with its Stage-B chain. Until then it is empty and the
    app keeps the duplication map.
  - The per-session switch already exists: the panel's "duplication map" toggle (`restirFeatures.dupmap`).
  - D1 / D2 / D4 land here only where WP-Q's equal-quality rule holds.
- Decision rows: `run-perf.ts --decisions` adds labelled 540p N3 rows per scene from `DECISION_CONFIGS` (D1 rrMin2,
  D2 risM16, D3 dupmapOff, D4 slots2, D6 halfRate). A row whose flag has not landed is skipped with a note.

## 3. Bits goldens: Anchor and Shipped tiers

`validation/gpu-tests/m8-bits.gpu.test.ts`:

- **Anchor** (`ANCHOR`, `ANCHOR_NOP9`): every results-changing flag off; bit-equal forever.
  - The test runs with the bitwise subset of `VITE_PERF_FLAGS`.
  - A flag with `except: ['P9']` is compared on the P9-excluded hash only.
  - A flag with `except: ['counters']` is compared on the reservoir and image fields only.
- **Shipped** (`SHIPPED`, `SHIPPED_NOP9`): `RELEASE_PERF_FLAGS ∪ VITE_PERF_FLAGS` on.
  - While that set adds nothing over the Anchor run, the tier is the Anchor table, checked structurally.
  - Otherwise every case runs again with the shipped flags.
  - A package that releases a results-changing flag replaces the affected entries, with a written justification and a
    diagnostic.
- **P9-excluded hash mode**: `m8BitsCaseHashes` returns `full` and `noP9`. For `noP9`, reservoir words 36–39 (plane P9:
  nCand, selId, kMargin, endpointId) are zeroed before hashing; the image and the counters are unchanged.

New cases (`PERF2_BITS_CASES`), recorded on the unmodified perf2 text (6d46432), twice, with identical results:

| Case | Scene | Configuration |
|---|---|---|
| `all-interactive-B-cwbvh` | all-lights fixture | interactive (pinned), Mode B, CWBVH + MT, 5 frames with motion |
| `nm-interactive-B-cwbvh` | m7_nm_smooth_256 | interactive (pinned), Mode B, CWBVH + MT, 4 frames with motion |
| `alpha-full-m6-A-cwbvh` | m5s_alpha_foliage | full-m6, Mode A, CWBVH + Woop, 3 frames (bit-equal to the BVH2 case) |
| `sponza-lite-interactive-B` | Sponza + kloofendal HDRI | 160×90, interactive (pinned), Mode B compiled as Mode A (P-4), SoA, CWBVH + MT, exact texels, the perf auto setup, 3 frames with motion |

The existing six cases gained P9-excluded goldens. All 10 cases are in the Anchor tier, and Shipped = Anchor.

## 4. Profiling tools (`validation/tools/perf/`)

These were promoted from `validation/out/perf2-profile/tools` and `counters/tools`. Hard-coded paths were removed; the
Python scripts import `xctab` from their own directory. Chrome tools do not take the GPU lock unless stated.

| Tool | Use |
|---|---|
| `prof-driver.ts` | Headless-Chrome driver. `--mode dump` (Dawn `dump_shaders` → JSONL), `--mode xct` (Metal System Trace attached to Chrome's GPU process in the split or frame phase; `--target all --trigger CMD` for counters), `--mode perf` (custom Chrome args / Dawn toggles). Takes `--kernel-flags`. Run it under `with-gpu-lock.ts`. |
| `split-dump.py DUMP.jsonl OUTDIR` | Dump → `<entry>[.k].metal` + `wgsl/` + `INDEX.txt` |
| `msl_lint.py MSL_DIR [--json]` | Per kernel: MSL bytes, robustness clamps, `tint_loop_idx` loops, `tint_volatile_zero` guards, zero-inits, Tint int div/mod helpers, `bvh_trace` copies after inlining |
| `xct-export.sh TRACE PREFIX [schema…]` | xctrace tables (default: all used here) |
| `xct_spill.py PREFIX [--json]` | Spilled bytes per thread per pipeline (graphics-compiler-spill-events) |
| `xct_split.py PREFIX` / `xct_groups.py PREFIX` | Per-pass GPU busy time + spill (split phase) / per-encoder time (frame phase) |
| `abba.py REPORT.json… [--keys] [--json]` | Same-session ABBA deltas: frame, Σpasses, denoiser, per pass; labels `<group> <role> #i` |
| `gpu-bg.py [secs]` / `gpu-bg.py --loop s LOG` | Background GPU load per process; the loop logs the lock holder |
| `ctr_export.sh`, `ctr_analyse.py`, `ctr_ids.py`, `mk_template.py`, `perfstate_split.py` | GPU-counter traces (unavailable on M5 Pro / Xcode 27.0: only "RT Unit Active"), and the perf-state clock-sensitivity split |
| `pso-stats.swift` | Compiles dumped MSL and prints the pipeline limits |
| **`spill-lint.ts`** | The report-only "spill bytes per pipeline + MSL lint" command. Takes the GPU lock itself around each Chrome run. |

```
npx tsx validation/tools/perf/spill-lint.ts --scene sponza,cornell [--res 480x270] [--kernel-flags A,B] [--tag T] \
    [--compare validation/out/perf2-spill/<base-tag>/report.json] [--no-dump | --no-trace]
```

`--no-dump` / `--no-trace` skip a step and reuse that step's result from an earlier run with the same tag. xctrace exits
with code 2 because of the corrupt-log-archive warning, which it reports on every recording here. If the trace was
saved, this is treated as a warning, because the Metal tables are complete.

It writes `validation/out/perf2-spill/<tag>/report.{json,md}`. With `--compare`, deltas are shown and growth in a hot
kernel is marked `!`.

ABBA timing:

```
npx tsx validation/harness/run-perf.ts --abba RS_VIS_MERGE --only sponza,cornell --res 540p --blocks 2 --tag abba-vis
python3 validation/tools/perf/abba.py validation/out/m8-perf/abba-vis.json
```

Neither command records with xctrace, and neither puts timestamps around ReSTIR passes (Q3).

### Baseline (perf2 text 6d46432 = WP-0, no flags)

`spill-lint.ts --scene sponza,cornell --tag baseline-6d46432` (480×270; the report is in
`validation/out/perf2-spill/baseline-6d46432/`; pass it to `--compare`). Sponza runs CWBVH with textures, normal maps and
alpha, compiled as Mode-A text (P-4). Cornell runs BVH2 in Mode B. The Sponza spill values reproduce the profile run
(`validation/out/perf2-profile`). Where the two scenes differ, the lint values are given as Sponza / Cornell.

| kernel | spill B (Sponza / Cornell) | MSL bytes | clamps | loop_idx | volatile | zero_init | divmod | bvh_copies |
|---|---|---|---|---|---|---|---|---|
| rs_initial | 1152 / 1808 | 147510 / 153517 | 63 | 6 / 8 | 8 / 5 | 16 | 9 | 6 / 8 |
| rs_spatial_shift | 800 / 1072 | 126391 / 125097 | 60 | 6 | 8 / 5 | 19 | 10 | 4 |
| rs_spatial_replay | 1072 / 1232 | 139473 / 140874 | 61 | 6 | 8 / 5 | 23 | 12 | 5 |
| rs_t_classify | 736 / 992 | 142706 / 141969 | 75 | 4 | 16 / 13 | 20 | 6 | 4 |
| rs_t_forward | 1056 / 1232 | 137772 / 139734 | 60 | 6 | 8 / 5 | 24 | 8 | 5 |
| rs_t_inverse | 1184 / 1360 | 139281 / 141243 | 60 | 6 | 8 / 5 | 23 | 8 | 6 |
| rs_refresh_fwd | 560 / 1056 | 111933 / 111720 | 58 | 4 | 3 / 0 | 19 / 22 | 12 | 4 / 6 |
| rs_refresh_inv | 272 / 784 | 112057 / 111844 | 60 | 4 | 3 / 0 | 18 / 21 | 10 | 4 / 6 |
| rs_t_select | 0 / 0 | 53948 / 54410 | 25 | 0 | 0 | 2 | 4 | 0 |
| rs_primary | 0 / 144 | 50659 / 44794 | 33 | 4 | 3 / 0 | 5 | 5 | 1 |
| primary | 0 / 144 | 56507 / 50604 | 33 | 2 | 3 / 0 | 4 | 3 | 1 |
| rs_spatial_resample | 0 / 0 | 37586 | 20 | 6 | 0 | 2 | 8 | 0 |
| rs_pair_accept | 0 / 0 | 22952 | 19 | 2 | 0 | 1 | 7 | 0 |
| rs_dupmap | 0 / 0 | 9097 | 6 | 0 | 0 | 1 | 3 | 0 |

Spill is bytes per thread from the Metal compiler; 0 means the kernel ran in the trace without a spill event. Only the
hot kernels are listed; the full tables, including the denoiser and the debug passes, are in the report.

### A/A check of the ABBA pipeline

`run-perf.ts --abba RS_VIS_MERGE --only cornell --res 540p --blocks 1 --frames 32` with the flag still unlanded, so the
variant runs the same text as the base. The report is in `validation/out/perf2-wp0/abba-smoke.json`.

- The flag reached the jobs: the report records `perfFlags`, and the pins hold (maxBounces 3, risM 32, dupmap on).
- abba.py still measured +2.4 ms (+12 %; the two pairs gave +0.8 and +4.0 ms). Another agent's GPU jobs ran between
  the lock holds.
- That is the noise floor under shared load. Keep/drop runs need the quiet machine (D10), at least 2 blocks, and
  gpu-bg.py logging.
