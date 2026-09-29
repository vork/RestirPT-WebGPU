"""Unit tests for stats.py with known ground truth (plan §7.3; math.md §28)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as st

import stats as S
from conftest import base_image, heavy_noise, replicate_stack


# ---------------------------------------------------------------- (4) Welch vs scipy

@pytest.mark.parametrize("nx,nr,sx,sr", [(16, 16, 1.0, 1.0), (8, 40, 0.3, 2.0), (64, 5, 5.0, 0.1), (2, 3, 1.0, 1.0)])
def test_welch_matches_scipy(rng, nx, nr, sx, sr):
    x = rng.normal(1.0, sx, (nx, 7))
    r = rng.normal(1.2, sr, (nr, 7))
    d, se, t, nu = S.welch_summaries(S.summarize(x), S.summarize(r))
    ref = st.ttest_ind(x, r, axis=0, equal_var=False)
    np.testing.assert_allclose(t, ref.statistic, rtol=1e-12)
    np.testing.assert_allclose(nu, ref.df, rtol=1e-12)
    np.testing.assert_allclose(S.p_two_sided(t, nu), ref.pvalue, rtol=1e-9)
    np.testing.assert_allclose(d, x.mean(0) - r.mean(0))


def test_welch_zero_variance_guards():
    d, se, t, nu = S.welch([1.0, 1.0, 2.0], [0.0, 0.0, 0.0], [15, 15, 15], [1.0, 0.5, 2.0], [0.0, 0.0, 0.0], [15, 15, 15])
    assert list(t) == [0.0, np.inf, 0.0]
    assert np.all(np.isnan(nu)) and np.all(se == 0)
    # one side exactly known (ν = inf, SE = 0) -> normal limit on the other side's dof
    _, _, t, nu = S.welch(1.1, 0.05, 15, 1.0, 0.0, np.inf)
    assert t == pytest.approx(2.0) and nu == pytest.approx(15.0)
    _, _, _, nu = S.welch(1.1, 0.05, np.inf, 1.0, 0.0, 15)
    assert np.isinf(nu)


def test_summarize_const_flag():
    v = np.array([[0.1, 1.0], [0.1, 2.0], [0.1, 3.0]])
    s = S.summarize(v)
    assert s.const.tolist() == [True, False] and s.se[0] == 0.0
    with pytest.raises(ValueError):
        S.summarize(v[:1])


# ---------------------------------------------------------------- (3) TOST dark tiles, zero variance, 0/0

def test_tost_relative_and_boundary():
    tc = st.t.ppf(0.99, 30)
    # relative: Δ/R̄ = 1%, SE/R̄ = 0.1% -> bound 1% + 0.1%·t_{0.99,30} < 2% passes; δ = 1% fails
    r = S.tost(0.01, 0.001, 30, 1.0, 1.0, 0.02)
    assert bool(r["passed"]) and r["bound"] == pytest.approx(0.01 + tc * 0.001)
    assert not bool(S.tost(0.01, 0.001, 30, 1.0, 1.0, 0.01)["passed"])
    assert r["rel"] == pytest.approx(0.01) and not bool(r["dark"])


def test_tost_dark_tile_uses_absolute_margin():
    # R̄_tile = 0.01·R̄_image (dark). A 10% relative error on it is only 0.001·R̄_image in absolute
    # terms, well inside the absolute margin δ·0.05·R̄_image = 0.001 (δ = 2%) ... at 0.0005 it passes.
    r = S.tost(0.0005, 0.00001, 30, 0.01, 1.0, 0.02)
    assert bool(r["dark"]) and r["margin"] == pytest.approx(0.02 * 0.05 * 1.0)
    assert bool(r["passed"])
    assert r["rel"] == pytest.approx(0.0005 / 0.05)       # reported against 0.05·R̄_image, not R̄_tile
    # the same tile judged relatively would fail (5% > 2%)
    assert not bool(S.tost(0.0005, 0.00001, 30, 0.01, 0.01, 0.02)["passed"])
    # exactly at the 5% threshold the relative rule applies
    assert not bool(S.tost(0.0, 0.0, 30, 0.05, 1.0, 0.02)["dark"])


def test_tost_zero_variance_and_zero_image():
    # both engines exactly zero: pass, no NaN anywhere
    r = S.tost(0.0, 0.0, np.nan, 0.0, 1.0, 0.02)
    assert bool(r["passed"]) and bool(r["zero_var"])
    for k in ("rel", "se_rel", "ratio", "bound"):
        assert np.isfinite(r[k])
    # zero variance but a 1e-7·R̄_image mismatch passes, 1e-5 fails (must match to 1e-6·R̄_image)
    assert bool(S.tost(1e-7, 0.0, np.nan, 0.3, 1.0, 0.02)["passed"])
    assert not bool(S.tost(1e-5, 0.0, np.nan, 0.3, 1.0, 0.02)["passed"])
    # an all-black image: R̄_image = 0 -> margins are 0; identical passes, any difference fails, no NaN
    r = S.tost(np.array([0.0, 1e-9]), np.array([0.0, 0.0]), np.array([np.nan, np.nan]), np.zeros(2), 0.0, 0.02)
    assert r["passed"].tolist() == [True, False]
    assert np.all(np.isfinite(r["rel"]) | np.isinf(r["rel"])) and not np.any(np.isnan(r["ratio"]))
    # noisy data on a black reference image can never pass (margin 0)
    assert not bool(S.tost(0.0, 1e-3, 20, 0.0, 0.0, 0.02)["passed"])


def test_evaluate_gate_black_and_dark_regions(rng):
    base = base_image(128, 128)
    x = replicate_stack(rng, 16, base, 0.2)
    r = replicate_stack(rng, 16, base, 0.2)
    spec = S.GateSpec.for_stage("A", n_units=4)
    res = S.evaluate_gate(S.aggregate_stack(x), S.aggregate_stack(r), spec)
    t = res.summary["channels"]["Y"]["tiles"]
    assert t["zero_var"] == 1 and t["dark"] >= 1
    assert res.passed, res.summary["failed_checks"]
    json_like = [c["value"] for c in res.checks]
    assert not any(isinstance(v, float) and math.isnan(v) for v in json_like)


# ---------------------------------------------------------------- multiplicity, sizing, suite

def test_sidak_and_alpha_unit():
    assert S.sidak_alpha(0.01, 1) == pytest.approx(0.01)
    a = S.sidak_alpha(0.01, 1024)
    assert (1 - a) ** 1024 == pytest.approx(0.99, rel=1e-12)
    assert float(S.sidak_threshold(0.01, 1024, np.inf)) == pytest.approx(4.42, abs=0.01)   # val §3.2
    assert float(S.sidak_threshold(0.01, 262144, np.inf)) == pytest.approx(5.5, abs=0.05)
    assert S.alpha_unit(1) == pytest.approx(0.01)
    assert S.alpha_unit(64) == pytest.approx(1 - 0.99 ** (1 / 64))


def test_bh_matches_scipy(rng):
    p = rng.random(500) ** 3
    np.testing.assert_allclose(S.bh_qvalues(p), st.false_discovery_control(p), rtol=1e-12)
    p2 = p.copy()
    p2[::7] = np.nan
    q = S.bh_qvalues(p2)
    ok = ~np.isnan(p2)
    np.testing.assert_allclose(q[ok], st.false_discovery_control(p2[ok]), rtol=1e-12)
    assert np.all(np.isnan(q[~ok]))


def test_sizing_rule_constants_and_required_replicates():
    # δ/6.4 for m = 256 tiles and δ/4.9 globally at α = 0.01 (plan §7.3, review V3)
    assert 0.01 / S.sizing_target(0.01, 256) == pytest.approx(6.44, abs=0.02)
    assert 0.01 / S.sizing_target(0.01, 1) == pytest.approx(4.90, abs=0.01)
    sd, delta, m = 0.05, 0.01, 256
    n = S.required_replicates(sd, delta, m)
    def se_ok(k):
        _, se, _, nu = S.welch(0, sd / math.sqrt(k), k - 1, 0, 0, np.inf)
        return float(se) <= S.sizing_target(delta, m, float(nu))
    assert se_ok(n) and not se_ok(n - 1)
    # both sides with the same n need more replicates than one side against an exact reference
    assert S.required_replicates(sd, delta, m, sd_other_rel_same_n=sd) > n
    # a reference whose SE already exceeds the target can never be fixed by more of "ours"
    assert S.required_replicates(sd, delta, m, se_other_rel=S.sizing_target(delta, m) * 1.01, nu_other=15) is None
    assert S.pilot_sd(0.01, 16) == pytest.approx(0.04)


def test_confirmatory_decision():
    assert S.confirmatory_decision(True)["status"] == "pass"
    assert S.confirmatory_decision(False)["status"] == "rerun_required"
    assert S.confirmatory_decision(False, True, [1, 2], [3, 4])["status"] == "pass_on_rerun"
    assert S.confirmatory_decision(False, False, [1, 2], [3, 4])["status"] == "fail"
    assert S.confirmatory_decision(False, True, [1, 2], [2, 3])["status"] == "invalid_rerun"
    assert S.confirmatory_decision(False, True, None, [2, 3])["status"] == "invalid_rerun"


# ---------------------------------------------------------------- distribution checks

def test_distribution_checks_under_h0_and_shift(rng):
    nu = np.full(2000, 15.0)
    t = st.t.rvs(15, size=2000, random_state=rng)
    c2 = S.chi2_red(t, nu)
    assert c2["expect"] == pytest.approx(15 / 13) and c2["passed"]
    assert S.mean_t(t, nu, 0.01)["passed"]
    assert S.ks_ad(t, nu, 0.01)["passed"]
    # a +0.5 shift in every t_A: mean-t and KS/AD must reject
    assert not S.mean_t(t + 0.5, nu, 0.01)["passed"]
    assert not S.ks_ad(t + 0.5, nu, 0.01)["passed"]
    # inflated spread (under-estimated SE) must break χ²_red
    assert not S.chi2_red(1.5 * t, nu)["passed"]
    # tiny families are "not applicable" rather than silently failing
    assert S.chi2_red(t[:3], nu[:3])["applicable"] is False


def test_ad_asymptotic_matches_scipy_uniform_reference(rng):
    # compare our A² for uniform data with scipy's anderson_ksamp-free check via simulation quantile:
    # under H0, P(p_AD < 0.05) ≈ 0.05
    hits = 0
    for _ in range(400):
        t = st.t.rvs(20, size=300, random_state=rng)
        hits += S.ks_ad(t, np.full(300, 20.0), 0.01)["p_ad"] < 0.05
    assert 0.02 <= hits / 400 <= 0.09


# ---------------------------------------------------------------- (5) metrics on hand-computed cases

def test_mape_hand_case():
    i = np.array([[1.0, 2.0]])[..., None]
    g = np.array([[1.0, 3.0]])[..., None]
    # mean(g) = 2 -> denominators 0.02 + 1, 0.02 + 3
    assert S.mape(i, g) == pytest.approx((0 / 1.02 + 1 / 3.02) / 2)
    # RGB inputs go through Rec.709 grey
    rgb = np.ones((1, 2, 3))
    assert S.mape(rgb * 1.1, rgb) == pytest.approx(0.1 / 1.01)


def test_relmse_hand_case():
    r = np.array([1.0, 3.0])            # mean 2 -> normalised r = [0.5, 1.5]
    x = np.array([2.0, 3.0])            # normalised x = [1.0, 1.5]
    assert S.rel_mse(x, r) == pytest.approx((0.25 / (0.25 + 0.01) + 0.0) / 2)
    assert S.rel_mse(x, r, normalize=False) == pytest.approx((1 / 1.01 + 0) / 2)
    se_r = np.array([0.2, 0.4])         # normalised [0.1, 0.2]
    assert S.rel_mse_corr(x, r, se_r) == pytest.approx(((0.25 - 0.01) / 0.26 + (0 - 0.04) / 2.26) / 2)
    assert S.mse_corr(x, r, se_r) == pytest.approx(((0.25 - 0.01) + (0 - 0.04)) / 2)
    # scale invariance of the normalised forms
    assert S.rel_mse(10 * x, 10 * r) == pytest.approx(S.rel_mse(x, r))
    assert float(S.bnr(0.0, 0.0)) == 0.0 and float(S.bnr(0.3, 0.1)) == pytest.approx(3.0)


def test_rmsrb_and_slope():
    r = np.full(100, 2.0)
    x = r * 1.01
    assert S.rmsrb(x, r, np.zeros(100), np.zeros(100)) == pytest.approx(0.01)
    n = 2.0 ** np.arange(8)
    fit = S.slope_fit(n, 3.0 / n)
    assert fit["slope"] == pytest.approx(-1.0) and fit["intercept"] == pytest.approx(math.log(3.0))


def test_hdr_flip_identity_and_difference(rng):
    a = rng.random((32, 32, 3)) + 0.1
    m0, emap = S.hdr_flip(a, a)
    assert m0 == pytest.approx(0.0, abs=1e-6) and emap.shape == (32, 32)
    m1, _ = S.hdr_flip(a, a * 1.5)
    assert m1 > 0.01


# ---------------------------------------------------------------- aggregation

def test_tiles_partial_edges_and_masks(rng):
    img = rng.random((40, 50, 3))
    tm = S.tile_means(img, 16)
    assert tm.shape == (3, 4, 3)
    np.testing.assert_allclose(tm[2, 3], img[32:40, 48:50].mean(axis=(0, 1)))
    masks = np.zeros((2, 40, 50), bool)
    masks[0, :10, :10] = True
    masks[1, 20:, 5:9] = True
    mm = S.mask_means(img, masks)
    np.testing.assert_allclose(mm[1], img[20:, 5:9].mean(axis=(0, 1)))
    y = S.channelize(img, ("Y", "B"))
    np.testing.assert_allclose(y[..., 0], img @ S.LUMA_709)


def test_replicates_from_sums_matches_images(rng):
    stack = replicate_stack(rng, 8, base_image(64, 96), 0.5)
    masks = np.zeros((1, 64, 96), bool)
    masks[0, 10:30, 40:90] = True
    a = S.aggregate_stack(stack, masks=masks, mask_names=["m"])
    sums = {f"tiles{s}": S.tile_sums(stack, s) for s in (16, 32, 64)}
    sums.update(**{"global": stack.sum(axis=(1, 2)), "masks": np.einsum("nhwc,mhw->nmc", stack, masks.astype(float)),
                   "mask_pixels": masks.sum(axis=(1, 2)), "pixel_sum": stack.sum(0),
                   "pixel_sumsq": (stack ** 2).sum(0), "count": 8, "channels": np.array(["R", "G", "B"])})
    b = S.replicates_from_sums(sums)
    for s in (16, 32, 64):
        np.testing.assert_allclose(b.tiles[s], a.tiles[s], rtol=1e-12)
    np.testing.assert_allclose(b.glob, a.glob, rtol=1e-12)
    np.testing.assert_allclose(b.masks, a.masks, rtol=1e-12)
    pa, pb = a.pixel_summary(), b.pixel_summary()
    np.testing.assert_allclose(pb.mean[..., 1:], pa.mean[..., 1:], rtol=1e-12)
    assert np.all(np.isnan(pb.se[..., 0]))              # per-pixel Y variance is not derivable from RGB sums
    np.testing.assert_allclose(pb.se[..., 1:], pa.se[..., 1:], rtol=1e-6, atol=1e-12)


def test_heavy_noise_is_unbiased(rng):
    x = heavy_noise(rng, 2_000_000, 1.0)
    assert x.min() >= 0
    assert x.mean() == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------- numerical-equivalence floor (num_eps)

def _near_deterministic(rng, bias_rel: float, n: int = 16):
    """Two f32-like implementations of a deterministic image: per-replicate noise 1e-7, relative offset bias_rel."""
    img = 0.2 + 0.6 * rng.random((128, 128, 3))
    a = img[None] * (1 + bias_rel) + rng.normal(0, 1e-7, (n, 128, 128, 3))
    b = img[None] + rng.normal(0, 1e-7, (n, 128, 128, 3))
    return S.aggregate_stack(a, tile_sizes=(32,)), S.aggregate_stack(b, tile_sizes=(32,))


def test_num_eps_floor(rng):
    spec = S.GateSpec.for_stage("A", n_units=8)
    a, b = _near_deterministic(rng, 3e-6)
    assert not S.evaluate_gate(a, b, spec).passed  # f32-level offset: the Δ = 0 tests have unbounded power
    r = S.evaluate_gate(a, b, spec.with_(num_eps=1e-4))
    assert r.passed and r.summary["channels"]["Y"]["tiles"]["numeric_det"] == 16
    a, b = _near_deterministic(rng, 5e-4)  # a real sub-δ bias above the floor is still caught (it must MATCH)
    r = S.evaluate_gate(a, b, spec.with_(num_eps=1e-4))
    assert not r.passed and "numeric_tiles[Y]" in r.summary["failed_checks"]
