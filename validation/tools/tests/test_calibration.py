"""Harness calibration on synthetic replicates with KNOWN truth (plan §7.3 Calibration, Gate 1-lite).

(1) A/A with heavy-tailed noise -> tile false-positive rate ≈ nominal, χ²_red within bound.
(2) Planted biases detected at the claimed MDB when powered; not detected when under-powered,
    and TOST never passes on noise (equivalence cannot be certified by an under-powered test).
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest

import stats as S
from conftest import base_image, heavy_noise, replicate_stack


# ---------------------------------------------------------------- (1) A/A

@pytest.mark.parametrize("seed,scale,sigma", [(1, 1.0, 0.6), (2, 1.0, 1.0)])
def test_aa_split_heavy_tails_nominal_rates(seed, scale, sigma):
    rng = np.random.default_rng(seed)
    base = base_image(256, 256)
    stack = replicate_stack(rng, 32, base, scale, sigma)        # 16 vs 16 per split
    reps = S.aggregate_stack(stack, tile_sizes=(16,))
    spec = S.GateSpec.for_stage("B", tile=16, n_units=1)
    aa = S.aa_split(reps, spec, n_splits=20, rng=seed)
    for a, r in aa["per_tile"].items():
        # pooled over 20 splits × 4 channels × 255 non-zero tiles
        assert 0.4 * float(a) <= r["rate"] <= 1.8 * float(a), (a, r)
    rc = aa["rejection_checks"]
    assert rc["chi2_red"]["count"] <= 1, rc["chi2_red"]      # χ²_red within its bound
    assert rc["sidak_tiles"]["rate"] <= 0.05
    assert rc["mean_t"]["rate"] <= 0.05 and rc["ks_ad"]["rate"] <= 0.05
    assert aa["ok"]


def test_aa_split_detects_broken_se(monkeypatch):
    """A harness whose SE is under-estimated (here by 1.5×, as naive per-pixel variance sums or
    pseudo-replication would do) must show an inflated A/A false-positive rate and fail calibration."""
    rng = np.random.default_rng(7)
    stack = replicate_stack(rng, 32, base_image(128, 128), 1.0)
    reps = S.aggregate_stack(stack, tile_sizes=(16,))
    spec = S.GateSpec.for_stage("B", tile=16)
    assert S.aa_split(reps, spec, n_splits=20, rng=3)["ok"]
    honest = S.summarize

    def broken(values):
        s = honest(values)
        s.se = s.se / 1.5
        return s

    monkeypatch.setattr(S, "summarize", broken)
    aa = S.aa_split(reps, spec, n_splits=20, rng=3)
    assert not aa["ok"]
    assert aa["per_tile"]["0.01"]["rate"] > 0.03


def test_aa_split_requires_20_splits():
    reps = S.aggregate_stack(np.ones((4, 16, 16, 3)), tile_sizes=(16,))
    with pytest.raises(ValueError):
        S.aa_split(reps, S.GateSpec.for_stage("A", tile=16), n_splits=10)


# ---------------------------------------------------------------- (2) MDB is what it claims to be

@pytest.mark.parametrize("n", [16, 64])
def test_mdb_power_matches_claim(n):
    """Bias equal to the reported MDB is detected (two-sided α = 0.01) with ≈ 90% power."""
    rng = np.random.default_rng(99 + n)
    sd = 0.02                                                   # per-replicate relative SD of the aggregate
    se_true = sd * np.sqrt(2.0 / n)
    bias = float(S.mdb(se_true, 0.01, 0.9))
    hits = 0
    reps = 400
    for _ in range(reps):
        x = 1.0 + bias + sd * rng.standard_normal(n)
        r = 1.0 + sd * rng.standard_normal(n)
        _, _, t, nu = S.welch_summaries(S.summarize(x), S.summarize(r))
        hits += float(S.p_two_sided(t, nu)) < 0.01
    rate = hits / reps
    # z-based MDB is slightly optimistic for Welch-t with finite ν; 0.9 nominal
    assert 0.78 <= rate <= 0.96, rate


def _powered_stack(rng, n=32, scale=0.12):
    return replicate_stack(rng, n, base_image(128, 128), scale)


@pytest.mark.parametrize("stage,kind,factor,region", [
    ("A", "scale", 1.0075, None),            # Stage A: light ×1.0075
    ("A", "region", 1.03, (64, 32, 32)),     # Stage A: one 32² region +3%
    ("A", "scale", 1.01, None),              # 1% plant against δ_global = 0.5%
    ("B", "scale", 1.003, None),             # Stage B: W ×1.003
    ("B", "channel", 1.003, None),           # chromatic 0.3% bug (R only) against δ_global = 0.2%
])
def test_plants_detected_when_powered(stage, kind, factor, region):
    rng = np.random.default_rng(zlib.crc32(repr((stage, kind, factor)).encode()))
    stack = _powered_stack(rng, 32, 0.12 if stage == "A" else 0.05)
    spec = S.GateSpec.for_stage(stage, n_units=8)
    res = S.plant_detection(stack, spec, lambda s: S.plant(kind, s, factor=factor, region=region, channel=0),
                            n_repeats=10, rng=5)
    assert res["powered"], res                  # the unplanted control passes (so detection means something)
    assert res["detected"], res                 # the planted gate fails in ≥ 9/10
    assert res["calibrated"]
    assert res["mdb_global_median"] <= abs(factor - 1) or kind != "scale"


def test_plant_03pct_detected_at_mdb_significance():
    """0.3% global bias with SE sized so MDB ≤ 0.3%: the rejection tests (not just TOST) see it."""
    rng = np.random.default_rng(11)
    stack = _powered_stack(rng, 32, 0.05)
    spec = S.GateSpec.for_stage("B", n_units=1)
    res = S.plant_detection(stack, spec, lambda s: S.plant("scale", s, factor=1.003), n_repeats=10, rng=1)
    assert res["mdb_global_median"] <= 0.003
    assert res["significant_count"] >= 9, res


def test_underpowered_plants_not_detected_and_tost_never_passes():
    """Few noisy replicates: a 0.3% plant is invisible to the significance tests, and the gate still
    FAILS (TOST cannot certify equivalence on noise) — both planted and unplanted."""
    rng = np.random.default_rng(21)
    base = base_image(64, 64)
    spec = S.GateSpec.for_stage("B", n_units=1, min_replicates=2)
    sig = 0
    for i in range(20):
        x = replicate_stack(rng, 4, base, 1.0, 1.0)
        r = replicate_stack(rng, 4, base, 1.0, 1.0)
        planted = S.aggregate_stack(S.plant("scale", x, factor=1.003))
        res_p = S.evaluate_gate(planted, S.aggregate_stack(r), spec)
        res_c = S.evaluate_gate(S.aggregate_stack(x), S.aggregate_stack(r), spec)
        sig += S.significant(res_p, alpha_global=0.01)
        for res in (res_p, res_c):
            assert not res.passed
            assert not any(c["passed"] for c in res.checks if c["name"] == "tost_global")
        g = res_p.summary["channels"]["Y"]["global_"]
        assert g["mdb"] > 0.003                           # MDB says it cannot see 0.3%
    assert sig <= 4, sig


def test_min_replicates_check():
    rng = np.random.default_rng(3)
    base = base_image(64, 64)
    x = S.aggregate_stack(replicate_stack(rng, 8, base, 0.01))
    r = S.aggregate_stack(replicate_stack(rng, 8, base, 0.01))
    res = S.evaluate_gate(x, r, S.GateSpec.for_stage("A"))
    assert "min_replicates[all]" in res.summary["failed_checks"]


def test_plant_helpers():
    a = np.ones((2, 8, 8, 3))
    assert np.all(S.plant("scale", a, factor=1.5) == 1.5)
    p = S.plant("region", a, factor=2.0, region=(2, 4, 3))
    assert p[:, 4:7, 2:5].min() == 2.0 and p.sum() == a.sum() + 2 * 9 * 3
    p = S.plant("region", a, factor=2.0, region=(0, 0, 4, 2))
    assert p[0, :2, :4].min() == 2.0 and p[0, 2:, :].max() == 1.0
    assert np.all(S.plant("channel", a, factor=3.0, channel=2)[..., 2] == 3.0)
    with pytest.raises(ValueError):
        S.plant("bogus", a, factor=1.0)
