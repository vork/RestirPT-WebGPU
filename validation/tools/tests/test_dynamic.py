"""M5 tools (restir-temporal-api.md §6.4, §6.5): dyn_masks.py, dynamic.py, plant_sign.py on synthetic data with a known
truth (OWNER T-E)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import dyn_masks as DM
import dynamic as DY
import plant_sign as PS
from imageio_util import write_pfm


# ------------------------------------------------------------------------------------------------ dyn_masks

def test_partition_priorities():
    h = w = 64
    lt = np.full((h, w), 1.0)
    lp = np.full((h, w), 1.0)
    lt[0:8, 0:8] = 0.0          # gone: dark now, bright before
    lp[8:16, 0:8] = 0.0         # new: dark before, bright now
    lt[16:24, 0:8] = 1.5        # edge: +50 %
    dis = np.zeros((h, w), bool)
    dis[16:20, 0:8] = True      # disocclusion wins over edge
    m = DM.partition(lt, lp, float(lt.mean()), float(lp.mean()), dis)
    assert m["M_gone"][0:8, 0:8].all() and m["M_new"][8:16, 0:8].all()
    assert m["M_disocc"][16:20, 0:8].all() and m["M_edge"][20:24, 0:8].all() and not m["M_edge"][16:20, 0:8].any()
    assert m["M_steady"][40:, 40:].all()
    total = sum(v.astype(int) for v in m.values())
    assert (total == 1).all(), "the partition covers every pixel exactly once"


def test_block_luma_and_dominance():
    img = np.zeros((8, 8, 3))
    img[0, 0] = 16.0
    b = DM.block_luma(img)
    assert np.isclose(b[0, 0], 1.0) and np.isclose(b[3, 3], 1.0) and b[4, 4] == 0
    full = np.ones((4, 4))
    only = np.full((4, 4), 0.69)
    only[0, 0] = 0.71
    d = DM.dominance(full, only)
    assert d[0, 0] and d.sum() == 1


def test_build_writes_bits(tmp_path: Path):
    rng = np.random.default_rng(0)
    for name, scale in (("t", 1.0), ("p", 1.0)):
        d = tmp_path / name
        d.mkdir()
        for b in range(2):
            img = np.full((64, 64, 3), 0.5) * scale + 0.001 * rng.random((64, 64, 3))
            if name == "t":
                img[:32, :32] *= 2.0     # edge region (32² ≥ 256 px)
            write_pfm(d / f"batch_{b:03d}.pfm", img.astype(np.float32))
    meta = DM.build(tmp_path / "t", tmp_path / "p", 5, tmp_path / "out")
    assert meta["names"] == ["M_edge", "M_steady"] and "M_new" in meta["dropped"]
    bits = np.fromfile(tmp_path / "out/f5/masks.bin", dtype="<u2").reshape(64, 64)
    assert (bits[:32, :32] == 1).all() and (bits[40:, 40:] == 2).all()
    assert (tmp_path / "out/f5/mask_1.pfm").is_file()


# ------------------------------------------------------------------------------------------------ dynamic

def _drift_sample(rng, slope: float, R=256, F=5, rho=0.8):
    frames = np.arange(F) * 8.0
    cov = rho * np.ones((F, F)) + (1 - rho) * np.eye(F)
    L = np.linalg.cholesky(cov)
    rows = 1.0 + 0.05 * (rng.standard_normal((R, F)) @ L.T) + slope * frames
    ref_means = np.ones(F) + 0.001 * rng.standard_normal(F)
    ref_ses = np.full(F, 0.001)
    return frames, rows, ref_means, ref_ses


def test_drift_null_calibrated():
    """Under H0 the sandwich z is ~N(0,1) even with strongly correlated frames (same chains)."""
    rng = np.random.default_rng(1)
    zs = []
    for _ in range(400):
        f, rows, rm, rs = _drift_sample(rng, 0.0)
        rm = 1.0 + rs * rng.standard_normal(rm.size)
        zs.append(DY.drift(f, rows, rm, rs)["z"])
    zs = np.array(zs)
    assert abs(zs.mean()) < 0.2 and 0.8 < zs.std() < 1.2


def test_drift_detects_slope():
    rng = np.random.default_rng(2)
    f, rows, rm, rs = _drift_sample(rng, 2e-3)
    r = DY.drift(f, rows, np.ones(5), rs)
    assert r["z"] > 5 and r["ci99"][0] > 0


def test_tile_binomial():
    assert DY.tile_binomial(0, 16) == 1.0
    assert DY.tile_binomial(1, 16) == pytest.approx(1 - 0.99 ** 16)
    assert DY.tile_binomial(6, 16) < 1e-8


def _npz(path: Path, imgs: np.ndarray, masks: list[np.ndarray] | None = None, names=None):
    """Ensemble npz (sums) from per-run images (R, H, W, 3)."""
    R, H, W, _ = imgs.shape
    d = {}
    for s in (16, 32, 64):
        th, tw = -(-H // s), -(-W // s)
        t = np.zeros((R, th, tw, 3))
        for ty in range(th):
            for tx in range(tw):
                t[:, ty, tx] = imgs[:, ty * s:(ty + 1) * s, tx * s:(tx + 1) * s].sum(axis=(1, 2))
        d[f"tiles{s}"] = t
    d["global"] = imgs.sum(axis=(1, 2))
    if masks:
        d["masks"] = np.stack([imgs[:, m].sum(axis=1) for m in masks], axis=1)
        d["mask_pixels"] = np.array([m.sum() for m in masks], dtype=np.float64)
        d["mask_names"] = np.array(names)
    d["pixel_sum"] = imgs.sum(axis=0)
    d["pixel_sumsq"] = (imgs ** 2).sum(axis=0)
    d["count"] = np.array(R)
    d["channels"] = np.array(["R", "G", "B"])
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "ensemble.npz", **d)
    (path / "meta.json").write_text(json.dumps({"seeds": [f"s:c{i}" for i in range(R)], "chains": R}))


def _pt(path: Path, imgs: np.ndarray, seed0=1000):
    path.mkdir(parents=True, exist_ok=True)
    for b, im in enumerate(imgs):
        write_pfm(path / f"batch_{b:03d}.pfm", im.astype(np.float32))
    (path / "meta.json").write_text(json.dumps({"seeds": [f"{seed0}:{b}" for b in range(len(imgs))]}))


def test_merge_npz(tmp_path: Path):
    rng = np.random.default_rng(3)
    a, b = rng.random((16, 64, 64, 3)), rng.random((16, 64, 64, 3))
    _npz(tmp_path / "c0", a)
    _npz(tmp_path / "c1", b)
    (tmp_path / "c1/meta.json").write_text(json.dumps({"seeds": [f"s:c{16 + i}" for i in range(16)], "chains": 16}))
    DY.merge_npz(tmp_path / "m", [tmp_path / "c0", tmp_path / "c1"])
    z = np.load(tmp_path / "m/ensemble.npz")
    assert int(z["count"]) == 32 and np.allclose(z["global"], np.concatenate([a, b]).sum(axis=(1, 2)))


def test_sequence_no_drift(tmp_path: Path):
    rng = np.random.default_rng(4)
    H = W = 64
    truth = np.full((H, W, 3), 0.5)
    frames = []
    base = rng.standard_normal((64, 1, 1, 1)) * 0.01          # per-chain offset shared across frames (correlation)
    for f in (4, 8, 12):
        ours = truth * (1 + base + 0.02 * rng.standard_normal((64, H, W, 3)))
        ref = truth * (1 + 0.02 * rng.standard_normal((16, H, W, 3)))
        _npz(tmp_path / f"o{f}", ours)
        _pt(tmp_path / f"r{f}", ref)
        frames.append({"frame": f, "ours": str(tmp_path / f"o{f}"), "ref": str(tmp_path / f"r{f}")})
    (tmp_path / "test.json").write_text(json.dumps({"stage": "dyn", "channels": ["Y", "R", "G", "B"], "n_units": 10}))
    (tmp_path / "seq.json").write_text(json.dumps({"test": "test.json", "frames": frames, "n_units": 10}))
    r = DY.sequence(tmp_path / "seq.json", tmp_path / "out")
    assert r["ok"], r


# ------------------------------------------------------------------------------------------------ plant_sign

@pytest.mark.parametrize("sign,factor,expect", [("-", 0.95, True), ("+", 0.95, False), ("-", 1.0, False)])
def test_plant_sign(tmp_path: Path, sign, factor, expect):
    rng = np.random.default_rng(5)
    H = W = 64
    truth = np.full((H, W, 3), 0.5)
    region = np.zeros((H, W), bool)
    region[:32, :32] = True
    rest = ~region
    ours = truth * (1 + 0.02 * rng.standard_normal((64, H, W, 3)))
    ours[:, region] *= factor
    ref = truth * (1 + 0.02 * rng.standard_normal((64, H, W, 3)))
    names = ["M_light:A", "M_other"]
    _npz(tmp_path / "o", ours, [region, rest], names)
    _pt(tmp_path / "r", ref)
    write_pfm(tmp_path / "m0.pfm", np.repeat(region.astype(np.float32)[..., None], 3, axis=2))
    write_pfm(tmp_path / "m1.pfm", np.repeat(rest.astype(np.float32)[..., None], 3, axis=2))
    (tmp_path / "test.json").write_text(json.dumps({"stage": "dyn", "n_units": 10, "min_replicates": 16,
                                                     "masks": [{"name": names[0], "file": "m0.pfm"}, {"name": names[1], "file": "m1.pfm"}]}))
    cfg = {"name": "t", "test": "test.json", "frames": [{"frame": 3, "ours": "o", "ref": "r"}],
           "predict": [{"frame": 3, "region": "M_light:A", "sign": sign}], "only": "-" if sign == "-" else None}
    (tmp_path / "plant.json").write_text(json.dumps(cfg))
    rep = PS.run(tmp_path / "plant.json", tmp_path / "out")
    assert rep["ok"] is expect, rep


def test_calibrate_scale_plant(tmp_path: Path):
    rng = np.random.default_rng(6)
    H = W = 64
    truth = np.full((H, W, 3), 0.5)
    ours = truth * (1 + 0.004 * rng.standard_normal((256, 1, 1, 1)) + 0.02 * rng.standard_normal((256, H, W, 3)))
    _npz(tmp_path / "o", ours)
    (tmp_path / "test.json").write_text(json.dumps({"stage": "B", "n_units": 10, "min_replicates": 16}))
    r = DY.calibrate(tmp_path / "o", tmp_path / "test.json", tmp_path / "out", factor=1.02)
    assert r["plant"]["gate_fail_count"] >= 9 and r["aa"]["ok"], r["plant"]


def test_plant_sign_region_missing(tmp_path: Path):
    """A predicted region that dyn_masks dropped is not evaluable; with no evaluable prediction the plant fails."""
    rng = np.random.default_rng(7)
    H = W = 64
    truth = np.full((H, W, 3), 0.5)
    _npz(tmp_path / "o", truth * (0.95 + 0.02 * rng.standard_normal((64, H, W, 3))))
    _pt(tmp_path / "r", truth * (1 + 0.02 * rng.standard_normal((32, H, W, 3))))
    (tmp_path / "test.json").write_text(json.dumps({"stage": "dyn", "n_units": 10}))
    cfg = {"name": "t", "test": "test.json", "frames": [{"frame": 3, "ours": "o", "ref": "r"}],
           "predict": [{"frame": 3, "region": "M_new", "sign": "-"}], "only": None}
    (tmp_path / "plant.json").write_text(json.dumps(cfg))
    rep = PS.run(tmp_path / "plant.json", tmp_path / "out")
    assert rep["detected"] and rep["evaluated_predictions"] == 0 and rep["ok"] is False


def test_merge_npz_deterministic(tmp_path: Path):
    rng = np.random.default_rng(8)
    _npz(tmp_path / "c0", rng.random((4, 32, 32, 3)))
    DY.merge_npz(tmp_path / "a", [tmp_path / "c0"])
    DY.merge_npz(tmp_path / "b", [tmp_path / "c0"])
    assert (tmp_path / "a/ensemble.npz").read_bytes() == (tmp_path / "b/ensemble.npz").read_bytes()
    z = np.load(tmp_path / "a/ensemble.npz")
    assert int(z["count"]) == 4 and str(z["channels"][0]) == "R"


def test_npz_128_tiles_from_64(tmp_path: Path):
    """Enlarged 128² aggregates of an ensemble npz are the block sums of its 64² tiles (Changelog E-15)."""
    import stats as S
    rng = np.random.default_rng(9)
    imgs = rng.random((5, 200, 136, 3))
    _npz(tmp_path / "o", imgs)
    z = dict(np.load(tmp_path / "o/ensemble.npz"))
    reps = S.replicates_from_sums(z, ("Y", "R", "G", "B"), (128,))
    direct = S.aggregate_stack(imgs, ("Y", "R", "G", "B"), (128,))
    assert np.allclose(reps.tiles[128], direct.tiles[128])
