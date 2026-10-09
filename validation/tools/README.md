# validation/tools: comparison statistics (M2)

This directory holds `compare.py` and `stats.py`. They implement plan §7.3,
[math.md §28](../../docs/math.md#validation-stats), val §3 and the review fixes V3/V8.

| File | Role |
|---|---|
| `stats.py` | Pure numpy/scipy. Replicate aggregation, Welch/TOST, multiplicity, distribution checks, metrics, sizing, suite FWER, and the calibration helpers (A/A splits, plants). |
| `compare.py` | CLI with three modes: compare, `--curve` and `--calibrate`. Reads inputs and writes `report.json`, PNGs and `index.html`. |
| `report/` | `plots.py` (matplotlib figures), `template.html` and `render_index()` (the static HTML report). |
| `tests/` | pytest suite on synthetic data with known truth. |
| `imageio_util.py` | PFM/EXR IO. Arrays are float64 HxWxC with row 0 at the top. Not owned by M2. |
| `analytic_check.py` | (M3a) A replicate set (ours or Cycles) vs a package's analytic `expected`: C0f constant, or a supersampled f64 image for b = 0 point/rect/disk light on a plane. |

Run the tests:

```
validation/.venv/bin/python -m pytest validation/tools/tests -q
```

The suite has 47 tests and runs in about 6 s.

## CLI

```
# Gate 2/3 comparison (exit 0 = all gating checks pass, 1 = fail / re-run required, 2 = input error)
validation/.venv/bin/python validation/tools/compare.py --ours OURS_DIR --ref REF_DIR --test test.json --out OUT_DIR [--frame F]

# confirmatory re-run of a failed unit on DISJOINT seeds (suite FWER rule)
compare.py --ours OURS2 --ref REF2 --test test.json --out OUT2 --rerun-of OUT/report.json

# Gate 4 convergence: prefix means at N = 2^k batches
compare.py --curve --ours OURS_DIR --ref REF_DIR --test test.json --out OUT_DIR

# Gate 1 calibration on one replicate set: >= 20 A/A re-splits + delta-scale plants (detect >= 9/10)
compare.py --calibrate --ref REF_DIR --test test.json --out OUT_DIR [--splits 20] [--repeats 10] [--seed 0]

# ... plus a RENDERED plant (e.g. Cycles light power x1.0075) on seeds disjoint from REF_DIR
compare.py --calibrate --ref REF_DIR --planted PLANT_DIR --test test.json --out OUT_DIR [--plant-name NAME]

# M5.5 denoiser (docs/decisions/denoiser.md §11; gate-m55.ts): LDR / HDR FLIP of raw and denoised images vs a PT reference
denoise_eval.py flip --ref REF/mean.pfm --pairs raw_f16.pfm:dn_f16.pfm,... --out flip.json [--png DIR]
# frames until the denoised regional mean recovers 95 % of each ix-e step (tiles.bin of several seeds; step masks from PT refs)
denoise_eval.py recovery --runs DIR,DIR --steps 32,56,80 --hold 24 --refs BEFORE.pfm:AFTER.pfm,... --names a,b,c --out rec.json
```

### Input directory layouts

In each directory, the first layout that matches wins.

| Files | Meaning | Seeds recorded from |
|---|---|---|
| `batch_###.{pfm,exr}` + `meta.json` | PT batch means (B replicates) | `meta.json` `"seeds"` (list, one per batch) |
| `run_###.{pfm,exr}` + `meta.json` | ReSTIR ensemble runs as images (R) | `meta.json` `"seeds"` |
| `seed_###.{exr,pfm}` + `manifest.json` | Cycles seeds (K) | file numbers, or `"seeds"` |
| `f{frame}_s{seed}.{exr,pfm}` + `manifest.json` | Per-frame seeds; select the frame with `--frame` or test.json `"frame"` | `s{seed}` |
| `*.npz` (`ensemble.npz` preferred) + `meta.json` | Pre-reduced ensemble aggregates (below) | `meta.json` `"seeds"` |

- Either file name, `meta.json` or `manifest.json`, is accepted on either side.
- Every scalar key whose name contains `hash` is copied into `report.json` → `provenance.{ours,ref}.hashes`, recursively (nested keys are dotted). So is every key that is version-like (`*version*`, `blender`, `chrome`, `commit`, `git`), along with the sha256 of the meta file.
- Images are read with `imageio_util.read_image`, so PFM is bottom-to-top on disk and is flipped to row 0 = top. Alpha is dropped.

### Ensemble `.npz` format (the `ensembleStats` read-back)

| Key | Shape | Content |
|---|---|---|
| `tiles16`, `tiles32`, `tiles64` | (R, Th, Tw, C) | Per-run tile **sums** over pixels. Edge tiles may be partial; the pixel counts are derived from H, W. |
| `global` | (R, C) | Per-run image sum |
| `masks` | (R, M, C) | Per-run mask-region sums (needed when test.json declares masks) |
| `mask_pixels` | (M,) | Pixel count per mask |
| `mask_names` | (M,) str | Optional |
| `pixel_sum`, `pixel_sumsq` | (H, W, C) | Σ_r x and Σ_r x² over runs (per-pixel maps only) |
| `count` | scalar | R |
| `channels` | (C,) str | Default `R,G,B`. Y is derived linearly for sums. It is **not** derived for `pixel_sumsq`, so per-pixel Y maps are skipped unless the file carries a `Y` channel. Include `Y` (C = 4) to get them. |
| `height`, `width` | scalar | Only needed without `pixel_sum` |

The sums are divided by the pixel counts on load. Per-run values stay separate, because uncertainty comes from the spread across runs.

### test.json

```jsonc
{
  "name": "cornell-i-stageA",
  "stage": "A",                 // "A" PT vs Cycles | "B" ReSTIR vs PT | "dyn" dynamic per-frame
  "tile": 32,                   // gate tile (defaults: A 32, B 32, dyn 64); a change is recorded as aggregate_enlarged
  "aggregate_note": "why",      // recorded with aggregate_enlarged
  "channels": ["Y","R","G","B"],
  "delta": {"global": 0.005, "tile": 0.02, "mask": 0.02},   // optional; LOOSER than the stage default => warning + report.notes.delta_loosened
  "alpha": 0.01,                // TOST α per side
  "n_units": 64,                // scenes × configs × gates × {Y,R,G,B} in this suite run -> α_u
  "alpha_suite": 0.01,
  "masks": [{"name": "disocclusion", "rect": [x0, y0, x1, y1]}, {"name": "m2", "file": "mask.png"}],
  "frame": 12,
  "tier": "tight",              // or "heavy-tail"; copied to report.json
  "min_replicates": 16,
  "checks": {"chi2_red": true},  // set false to make a check informational (recorded in notes.non_gating_checks)
  "num_eps": 1e-4,              // numerical-equivalence floor (default 0 = off): tiles with SE_Δ ≤ num_eps·den leave the
                                // Δ = 0 rejection family and must MATCH |Δ| ≤ 2·num_eps·den + z_Šidák·SE_Δ
                                // (check numeric_tiles); recorded in notes.numeric_floor (docs/decisions/cycles-deviations.md D2)
  "curve": {"slope_range": [-1.1, -0.9], "gate": false},
  "calibration": {"plants": [{"name": "light x1.0075", "kind": "scale", "factor": 1.0075},
                             {"name": "32² +3%", "kind": "region", "factor": 1.03, "size": 32, "x0": 64, "y0": 32}]}
}
```

Stage defaults (plan §7.3):
- A: δ 0.5% global, 2% tile, 2% mask;
- B: δ 0.2% global, 1% tile, 1% mask;
- dyn: δ 0.2% global, 2% on 64² tiles, 3% mask.

### Outputs

- **`report.json`**: strict JSON; a non-finite value becomes `null` or ±1e308. It contains:
  - `status`: `pass`, `rerun_required`, `pass_on_rerun`, `fail` or `invalid_rerun`; also `gate_passed`;
  - `checks[]` (name, channel, passed, gating, value, threshold, …) and `failed_checks`;
  - `channels.{Y,R,G,B}`: global, tile and mask statistics, with χ²_red, mean-t and KS/AD;
  - `mdb` per channel, with `powered_*` flags (MDB ≤ δ);
  - `sizing`: the SE target, the ratio against it, and `replicate_multiplier_needed`;
  - `metrics`: relMSE, relMSE_corr, MAPE, RMSRB and HDR-FLIP on Y and RGB converged means, plus `noise_floor_ref_halves` (the same metrics on reference half vs half);
  - `tier`, `notes` (δ loosened, aggregate enlarged, non-gating checks), `provenance`, `seeds`, `inputs` and `images`.
- **PNGs**:
  - `side_by_side.png`: identical exposure, sRGB;
  - `rel_diff.png`: signed Y relative difference clipped at ±δ_tile, with the dark-floor denominator;
  - `t_map.png`: per-pixel |t| clipped at 6, the Šidák contour at α_u over H·W pixels, and a BH-FDR q ≤ 0.01 mask;
  - `tile_t.png`: signed tile t, with TOST failures (×) and Šidák rejections (□) marked;
  - `flip.png`;
  - `z_hist_qq.png`: histogram and QQ plot against t(ν);
  - `curve.png`: curve mode only.
- **`index.html`**: a static report of all of the above.

## Statistical design (normative sources and the decisions taken here)

**Replicates only.** Every aggregate (tile, mask, global, and each pixel for the maps) is computed per replicate. Its mean, SE = s/√n and ν = n−1 come from those values. Per-pixel variances are never pooled. Uncertainty for ensemble `.npz` inputs also comes from the per-run tile, mask and global rows.

**Welch.** Welch t with Welch–Satterthwaite ν, checked against `scipy.stats.ttest_ind(equal_var=False)`. Edge cases are guarded:
- SE_Δ = 0 gives t = 0 when Δ = 0, otherwise ±inf, and ν = NaN.
- One side with SE = 0 and ν = ∞ (an exact/analytic value) gives the other side's ν.

**TOST (equivalence is the pass rule).** The rule is `|Δ| + t_{1−α,ν}·SE_Δ < margin`, with α = 0.01 per side (a 98% CI).
- margin = δ·R̄_A when R̄_A ≥ 0.05·R̄_image.
- Otherwise margin = δ·0.05·R̄_image, an absolute margin for dark tiles.
- Zero-variance aggregates (every replicate identical on both sides) must satisfy |Δ| ≤ 1e-6·R̄_image.
- R̄_image is the reference global mean of the same channel.
- Relative quantities (Δ/denominator and bound/margin) never divide 0/0.
- On an all-black image the margin is 0, so only exact equality passes.
- The pass condition is written in absolute form. It is algebraically identical to the math.md relative form when R̄ > 0.

**Suite FWER and which α applies where.**
- α_u = 1 − (1−α_suite)^{1/n_units} applies to every rejection-type check: Šidák tiles, χ²_red, mean-t and KS/AD.
- TOST keeps α = 0.01 per side. Its false failures are controlled by power (the sizing rule), not by α (V8).
- The Šidák family is the gate tile grid of one channel (m tiles), at FWER α_u: a tile rejects if p < 1 − (1−α_u)^{1/m}. The per-pixel |t| map uses the same α_u over H·W pixels (visual only).
- BH-FDR at q = 0.01 is used for visual maps only.

**Distribution checks.** Each runs over the finite tile t values of one channel.
- χ²_red = mean t², with expectation E = mean(ν_A/(ν_A−2)) over tiles with ν > 2. It passes when χ²_red < E·(1 + 4√(2/m)), the normative bound; a moment-based p-value is also reported, for information.
- mean-t: z = mean(t)/√(E/m). This is the exact form of "SE 1/√m", which it approaches for large ν. It rejects when p < α_u.
- KS and AD of the probability-integral transforms u_A = F_{ν_A}(t_A) against U(0,1). When every tile has the same ν this is identical to "KS vs t(ν)", and it stays exact when Welch ν varies by tile. The AD p-value uses the Marsaglia–Marsaglia asymptotic ADinf. KS and AD are Bonferroni-combined: the check passes when min(p_KS, p_AD) ≥ α_u/2.
- With fewer than 8 finite tiles a check is "not applicable" and does not fail.
- Zero-variance tiles are excluded here, because they are judged by the 1e-6 rule.

**Tension to be aware of.** The rejection-type checks test Δ = 0, not |Δ| < δ. In Stage A, a *real* sub-δ model mismatch that is diffuse over many tiles (for example a uniform 0.2% bias with SE sized at δ/6.4) will fail mean-t or χ²_red even though TOST passes. Detecting that is exactly what they are there for (val §3.2). If a Stage-A scene is known to be "model-approximate", switch the check to informational through `"checks": {"mean_t": false}`. That choice is recorded in `notes.non_gating_checks`; it is never silent.

**MDB.** MDB = (z_{1−α/2} + z_{0.9})·SE_Δ/denominator ≈ 3.86·SE_rel at α = 0.01 and 90% power. It is reported for global, tiles (max and median) and masks. `test_mdb_power_matches_claim` checks that a bias equal to the MDB is detected in about 90% of fresh draws. The z-based formula is slightly optimistic for Welch t with finite ν: the measured power is about 0.85–0.9.

**Sizing rule.** `sizing_target(δ, m, ν) = δ/(t_{1−α,ν} + z_{1−0.005/m})`. This is δ/4.90 globally and δ/6.44 at m = 256.
- `required_replicates(sd_rel, δ, m, …)` returns the smallest n satisfying the rule, computed by binary search that includes the n-dependence of Welch ν. The input `sd_rel` is the per-replicate relative SD; get it from a pilot with `pilot_sd(SE_pilot, n_pilot)`.
- It can fix the other side's SE (an existing reference) or size both sides with the same n. It returns `None` when the target is unreachable, meaning: enlarge the aggregate, never loosen δ.
- compare.py reports `replicate_multiplier_needed` = max(1, (SE/target)²) per channel. That is how many times more replicates (or spp) would be needed.

**Dark-tile classification** uses the reference tile mean R̄_tile (the reference side of the comparison, which is our PT in Stage B).

**min_replicates.** The heavy-tail rule (≥ 16 replicates) is a gating check, `min_replicates[all]`. The calibration helpers relax it to the half size, because A/A halves of K = 16 are 8 vs 8, as in val §3.9.

**Confirmatory re-run.** A first failure yields `rerun_required` (exit 1). `--rerun-of` applies `confirmatory_decision`:
- The unit passes (`pass_on_rerun`) only if the re-run passes **and** the seed sets are provably disjoint on both sides, compared as `ours:<seed>` and `ref:<seed>` labels.
- Unknown or overlapping seeds give `invalid_rerun`.
- For batch inputs, `meta.json` must list `"seeds"`.

## Calibration (Gate 1-lite)

- **`aa_split(reps, spec, n_splits ≥ 20)`** splits one replicate set into disjoint random halves and runs the full gate on half A vs half B.
  - It reports the pooled per-tile false-positive rate at α = 0.05 and 0.01, the failure rate of each rejection-type check against α_u, and the TOST pass rate (a power diagnostic).
  - `ok` is false when any rate is significantly *above* nominal: one-sided binomial p < 1e-3.
  - The re-splits share data, so the counts are over-dispersed and the binomial test is approximate. The 1e-3 level leaves room for that.
- **`plant(kind, stack, factor=…)`** plants a known bias. `scale` covers light ×1.0075 in a one-light scene and W ×1.003. `region` covers a 32² block +3%, located by (x0, y0, size). `channel` covers a chromatic bug.
- **`plant_detection(stack, spec, plant_fn, n_repeats=10)`** re-splits each repeat into halves A and B and runs two comparisons:
  - plant(A) vs B, which must fail in ≥ 9/10 repeats (`detected`);
  - A vs B, which must pass in ≥ 9/10 repeats (`powered`).
  - Both are required (`calibrated`). An under-powered TOST "detects" every plant, so detection alone proves nothing (V8).
  - `significant_rate` separately counts rejections of Δ = 0.
- **`--calibrate`** runs both on `--ref` with the stage's default plants:
  - A: ×1.0075, plus a +3% 32² block on the brightest 32-aligned block;
  - B/dyn: ×1.003.
  - The c_p+1 MIS plant needs the renderer, so it is out of scope here.
  - `"calibration": {"plants": []}` (an explicit empty list) disables the synthetic plants.
- **`--planted DIR`** (`rendered_plant_detection`) adds a plant that was *rendered*, e.g. the scene with light power
  ×1.0075 or spot blend 0.16 rendered by Cycles:
  - the seed sets of `--ref` and `--planted` must be known and **disjoint** (input error otherwise): a shared seed
    would correlate the two sides and void Welch's independence assumption;
  - each of the `--repeats` repeats splits `--ref` into halves A/B and draws a half-size subset P of the planted set;
    P vs B must fail (`detected`) and A vs B must pass (`powered`), both in ≥ 9/10 repeats, as for synthetic plants;
  - `report.json` → `calibration.rendered_plant` also records how many planted failures were equivalence (TOST)
    failures, a histogram of the failed checks, the median planted Δ and the control MDBs (global, tile max/median).

## What the tests establish (synthetic data, known truth)

1. **A/A with heavy tails.** The noise is a lognormal × Pareto(α = 2.5) mixture, with mean exactly 1. The test runs 256² images, 16² tiles and 16 vs 16 replicates, with 20 re-splits. The per-tile FP rate is within [0.4, 1.8]× nominal; measured values are 0.044–0.049 at 0.05 and 0.008–0.009 at 0.01. χ²_red stays within its bound. A 1.5× SE under-estimate is caught by `aa_split`.
2. **Plants.** These are detected and calibrated when powered: light ×1.0075, a 32² block +3%, a 1% plant with Stage A δ, W ×1.003, and a 0.3% R-only chromatic bug. With MDB ≤ 0.3% the rejection tests see the 0.3% plant. Under-powered runs (4 vs 4 replicates, noisy) show three things:
   - the 0.3% plant is mostly not significant;
   - the reported MDB is above 0.3%;
   - TOST never passes, for either the planted or the unplanted data.
3. **Dark tiles and guards.** Covered: the absolute margin, the 5% threshold, zero-variance matching to 1e-6·R̄_image, all-black images, and no NaN in any output.
4. **Welch** matches scipy (t, ν, p) for balanced and unbalanced n and variance.
5. **Formulas on hand-computed cases.** MAPE, relMSE, relMSE_corr, MSE_corr, RMSRB, the slope fit, BH (matches `scipy.stats.false_discovery_control`), the Šidák constants (4.42 at m = 1024; 5.5 at m = 262,144), and the sizing constants (δ/4.90, δ/6.44).
6. **End to end.** The fixtures are PFM batches with `meta.json` vs EXR seeds with `manifest.json`. The tests check:
   - a pass, with hashes copied, all PNGs written, index.html produced, and strict JSON;
   - a vertically flipped `ours` set fails, which catches orientation bugs;
   - a 1% plant fails, then a confirmatory re-run on disjoint seeds passes, while an overlapping-seed re-run is `invalid_rerun`;
   - the `f{frame}_s{seed}` layout with `--frame`, and an `.npz` ensemble;
   - `--curve`, with slope ≈ −1 and BNR present;
   - `--calibrate`;
   - CLI exit codes, including the loosened-δ warning.

## perf/ — perf2 profiling tools

`validation/tools/perf/` holds the perf2 profiling tools: the headless-Chrome profiling driver (`prof-driver.ts`), the
Dawn shader-dump splitter and MSL lint, Metal System Trace exports (pass times, spill bytes), ABBA analysis, background
GPU load, and the report-only `spill-lint.ts` command (spill bytes per pipeline + MSL lint). Usage, inputs and the
perf-flag registry they work with: [docs/decisions/perf2-api.md](../../docs/decisions/perf2-api.md) §4. They are not
part of any gate.
