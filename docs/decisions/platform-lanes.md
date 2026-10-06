# Platform lanes (M0 lane diff)

Generated at M0 from `vitest --project chrome` and `--project node-dawn` (`validation/gpu-tests/smoke.gpu.test.ts`).
The Chrome lane is authoritative for every gate. dawn.node is a fast pre-check only (plan §1.1).

| Item | Chrome 154 (headless, Playwright) | dawn.node (webgpu@0.6.1, Node 26) |
|---|---|---|
| adapter | apple / metal-3 | apple / metal-3 |
| features (device, after profile request) | bgra8unorm-storage, core-features-and-limits, float32-filterable, rg11b10ufloat-renderable, shader-f16, subgroups, texture-compression-bc, texture-formats-tier1, texture-formats-tier2, timestamp-query | bgra8unorm-storage, core-features-and-limits, float32-filterable, rg11b10ufloat-renderable, shader-f16, subgroups, texture-compression-bc, texture-formats-tier1, texture-formats-tier2, timestamp-query |
| features only in one lane | — | — |
| WGSL language features only in one lane | — | — |
| device limits differing | — (identical after profile clamp) | — |
| limits below profile | — | — |
| `x != x` NaN self-compare | FOLDED (false) — use bit tests | FOLDED (false) — use bit tests |
| FMA contraction of a*b+c | True | True |
| WGSL language features (both lanes) | buffer_view, immediate_address_space, linear_indexing, packed_4x8_integer_dot_product, pointer_composite_access, readonly_and_readwrite_storage_textures, subgroup_id, subgroup_uniformity, swizzle_assignment, texture_and_sampler_let, texture_formats_tier1, uniform_buffer_standard_layout, unrestricted_pointer_parameters | identical to Chrome |

Headless Chrome 154 exposes the hardware Metal adapter without `--enable-unsafe-webgpu` (`npm run smoke:chrome`), so the vitest Chrome lane runs without the flag.

## Metal quirks

### Q1 Per-thread state size cliff (M3a/M3b)

The Metal compiler silently miscompiles compute kernels whose per-thread state crosses a size threshold (wrong values or
no writes, no error). That is why `BsdfCtx` and `MatEval` are compact (docs/decisions/bsdf-api.md "BsdfCtx size") and
why ReSTIR kernels read reservoirs plane by plane (restir-api.md §4.5).

### Q2 Rare nondeterministic control-flow fault in the T3 shift test kernel (M4, 2026-09-30)

**Symptom.** In `restir-shift.gpu.test.ts`'s T3 kernel (loop `for ps in 0 … nPass−1 { o = shift_hybrid(…); if (ps == 0) {…}
else {…}; if (last) {…} }`) a few lanes took the `else` branch at `ps == 0`. Only lanes whose shift ran the forced-NEE
path (SH_FORCED: cases a-tri, f-env, a-delta), in batches of 3 / 7 / 11 lanes of one SIMD group. With `J0 = 0`,
`c0 = 0` as initial values (SC_OK == 0) this read as a forward shift "OK, J = 0" (two WP-B "anomalies") and broke the
harness's accounting invariants (Σ forward codes = trials; mode 0: no inverse codes; mode 1: Σ inverse codes = fwdOk)
in ~30 more lanes over the M4 runs (found by the coordinator's review, `t3scan.py`). The shift results of those
lanes were correct; only the branch was wrong.

**Evidence that it is a platform fault, not a logic bug.**
- The generated MSL (Dawn `--enable-dawn-features=dump_shaders,disable_symbol_renaming`, Chrome stderr through
  `DEBUG=pw:browser*`, local `vitest.dump.config.ts`) of the kernel is byte-identical across processes, and the branch
  is a plain `if ((ps == 0u))` on a function-local loop counter (MSL line 3924 of the dump).
- The same kernel on the same inputs (same frames, seeds, fixture) faulted in 2 of 3 runs of one process (1 event
  each) and in 0 of 6 runs of a later process: nondeterministic at run time on identical code and data.
- It appeared from commit 3529eda on (7 of ~20 runs), not at c807aac (~26 runs); 3529eda only added a guard in
  `rcPairTest` that the forced k = 2 path does not execute, i.e. it changed code generation, not semantics.
- Rates (T3-0 self shifts, t3_cases_256_noenv, only the forced-NEE case bins active, `VITE_DISC=…:forced`):
  6e02a93 harness + shift before the restructure: **2 events in 1.42e9 shifts (9 runs)**; the same harness with the
  restructured shift (below): **0 in 9.5e8 (6 runs)**; the instrumented harness (below) with either shift: **0 in
  7.2e9** (all bins and forced-only, t3_cases_256_noenv and t3_cutoff_256). The before/after difference is not
  statistically conclusive at these rates (expected 1.3 events after, observed 0).

**Production passes.** `rs_spatial_shift` / `rs_spatial_replay` have the same shape (a loop over slots around an
inlined `shift_hybrid`, then a store indexed by the loop counter), so a fault there would misroute a slot and show as
`RSC_PENDING_LEFT` / `RSC_SLOT_MISMATCH` or as a run-to-run difference. Stress (`VITE_STRESS`, 3 rounds × 6 slots,
18000 frames each on t3_cases_256_noenv and t3_cutoff_256, 2.7e10 accepted slots per run): all three counters 0, and
two repeated runs of the same build gave **bit-identical** arena counters (new build: 2 runs; pre-restructure build:
2 runs). No fault in 1.1e11 production slot shifts.

**Workarounds (m4-t3fault).**
1. `shift_hybrid` keeps the NEE end term and its visibility data in a few registers (`neeT`, `visPos`, `visN`,
   `visPrim`, `visInf`) computed inside the iteration that evaluates the light sample, instead of a full
   `LightSample` live across the rest of the shift and the visibility traversal (less live state on the forced path,
   in the spirit of Q1). Arithmetic is unchanged (T3-0 / T3-1 / T3-D / T6(b) / U-SFX-1 / U-RIS-* pass).
2. The T3 harness is robust against such a fault: sentinels (`c0 = 0xFFFFFFFF`, `J0 = −1`), the shift result captured
   right after the call, a per-thread count of then/else executions checked at the last pass (kernel counter
   `PLATFORM`, violation kind 20 with ps, nPass, mode, then/else counts, frame and trial), and the accounting
   invariants checked in `t3Report`. PLATFORM events are reported apart from LOGIC; the Gate-0 tests fail only on a
   PLATFORM rate above 1e-8 per shift (a flood would indicate something systematic).

**If it reappears.** Re-run the discriminator (`VITE_DISC="t3_cases_256_noenv:5000:6:<build>:forced"`, builds `new`,
`legacy`, `*-oldharness`, `*-noguard`; frozen copies `validation/gpu-tests/t3fault-*`) and the production stress
(`VITE_STRESS`); compare MSL dumps with `vitest.dump.config.ts`-style launch args. Each discriminator run of 6 reps
× 5000 frames takes ~6.6 min of GPU time.

### Q3 Timestamp writes around the interactive ReSTIR passes lose the frame's work (M4, 2026-09-30)

**Symptom.** `RestirFramePass.encode` put the app's frame timestamps on two empty compute passes bracketing the ReSTIR
passes. In some page loads of Chrome 154 on Metal, usually not the first page of a fresh browser, the frame's ReSTIR
work was then silently lost:
- no validation, internal or OOM error;
- frame GPU time about halved;
- the shift arena was never written and the image was about L1 only.

**Isolation.**
- **Where it happened:** only in pages that were in the failing state.
- **Always failed:** frames that had those timestamp writes. This held even when the writes were moved onto the first and
  last real ReSTIR passes, and with a fresh query set.
- **Always worked:**
  - frames without the writes;
  - frames timing only the primary pass;
  - the batch kernel, which never had them.

**Rule.** No `timestampWrites` on or around the ReSTIR passes (restir-api.md changelog C8). A regression test in
restir-spatial.gpu.test.ts asserts that no ReSTIR compute pass is encoded with `timestampWrites`. If per-pass ReSTIR GPU
timing is needed, measure it in a separate timing submit.

**Status.** Not reduced to a minimal repro outside the app; it needs a multi-page browser state.

### Q4 Hardware filtering of rgb9e5ufloat / rgba16float is lower precision than the texels (P2, 2026-09-30)

**Measured** (`validation/gpu-tests/env-format.gpu.test.ts`, ENV-F, Chrome 154 / Metal, M5 Pro): bilinear
`textureSampleLevel` of env texels stored exactly in rgb9e5ufloat or rgba16float vs the SAME values stored as
rgba32float, 10⁶ random uv (incl. the u seam and the pole rows): worst relative difference 2⁻¹¹ … 2⁻¹⁰·² (the three
validation HDRIs, a random RGBE map; ≈ 13–87% of lookups differ), and 15 of 10⁶ lookups above 2⁻²⁰ for the C0s f16
sun map. Texel-centre lookups (no interpolation) are exact. The filter evidently interpolates at half precision.

**Rule.** Validation mode keeps the env texture rgba32float (`env-gpu.ts` `ENV_COMPACT_IN_VALIDATION = false`); only
interactive mode uses the smallest exact format (data-formats.md §B9, E-3). Re-run ENV-F before changing that.

**M5 update (restir-temporal-api.md C-10).** The env lookup no longer uses the hardware filter: `env.wgsl envTexel` is an
explicit f32 bilinear over four `textureLoad`s (exact texel values in every format). ENV-U7b (env.gpu.test.ts, Chrome
154 / Metal): `envRadiance` on studio_small_09 1k with the smallest exact format (rgb9e5ufloat) is bit-identical to
rgba32float at 2·10⁵ random uv incl. the seam and pole rows (0 mismatches), and ≤ 1.2·10⁻⁷ from an f64 bilinear at the
f32 texel coordinate. The Q4 precision loss therefore no longer applies to the renderer; the compact formats could be
allowed in validation mode too (ENV-F keeps measuring the hardware filter only). `ENV_COMPACT_IN_VALIDATION` is left
false pending a coordinator decision (memory only; no correctness reason remains).
