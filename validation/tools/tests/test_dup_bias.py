"""dup_bias.py (Gate 5, restir-m6-api.md MD15 / Changelog M6-12).

(1) Real file layouts: a crop of a real gate run (ensemble.npz of per-chain SUMS from restir-chain-run.ts, PT
    batch_###.pfm means; fixtures/gate5_real/make_fixture.py) -> the global Δ equals compare.py's on the same inputs and
    the tile means equal a hand reduction (sums / pixel count). The first version read the npz sums as means: every
    tile was 256× and the image 65 536× the reference (Gate 5 run m6-gate-gate5-20261007-115614).
(2) Synthetic data in the writer's format (sums): the noise-debiased tile bias resolves a planted 5 % bias and stays
    below 1 % on unbiased data of the gate's size.
"""
import json
from pathlib import Path

import numpy as np

import compare
import stats as S
from dup_bias import evaluate, main
from imageio_util import read_image, write_pfm

FIX = Path(__file__).resolve().parent / "fixtures" / "gate5_real"
LUMA = np.array([0.2126, 0.7152, 0.0722])


def test_real_layout_matches_compare_and_hand_reduction(tmp_path):
    res = evaluate(FIX / "ours", FIX / "ref", tile=16)
    assert res["chains"] == 32 and res["ref_batches"] == 4 and res["tiles"] == 4
    # compare.py on the same directories (its CLI, its report)
    test = tmp_path / "test.json"
    test.write_text(json.dumps(dict(name="gate5-fixture", stage="B", channels=["Y"], n_units=1, tier="tight", min_replicates=2)))
    compare.main(["--ours", str(FIX / "ours"), "--ref", str(FIX / "ref"), "--test", str(test), "--out", str(tmp_path / "cmp")])
    rep = json.loads((tmp_path / "cmp" / "report.json").read_text())
    g = rep["channels"]["Y"]["global_"]
    assert abs(res["global_rel"] - g["rel"]) < 1e-12, (res["global_rel"], g["rel"])
    assert abs(res["global_se_rel"] - g["se_rel"]) < 1e-12
    # hand reduction: npz tile SUMS / 256 vs the PFM batch means
    with np.load(FIX / "ours" / "ensemble.npz") as z:
        assert int(z["count"]) == z["tiles16"].shape[0] and int(z["height"]) == 32
        ours_t = (z["tiles16"] @ LUMA).mean(0) / 256.0
        ours_g = (z["global"] @ LUMA).mean() / (32 * 32)
    ref = np.stack([read_image(p, drop_alpha=True)[..., :3] @ LUMA for p in sorted((FIX / "ref").glob("batch_*.pfm"))])
    ref_t = ref.reshape(-1, 2, 16, 2, 16).mean(axis=(2, 4)).mean(0)
    np.testing.assert_allclose(ours_g / ref.mean() - 1, res["global_rel"], atol=1e-12)
    np.testing.assert_allclose(np.mean((ours_t - ref_t) / ref_t), res["mean_signed_tile_rel"], atol=1e-12)
    # a units bug (sums read as means) shows up as ~25 000 %; the real dupmap bias here is a few per cent at most
    assert res["mean_tile_bias"] < 0.05 and abs(res["global_rel"]) < 0.05, res


def test_pooling_and_cli(tmp_path):
    a = evaluate(FIX / "ours", FIX / "ref")
    p = evaluate([FIX / "ours", FIX / "ours"], [FIX / "ref", FIX / "ref"])
    assert p["chains"] == 2 * a["chains"] and p["ref_batches"] == 2 * a["ref_batches"]
    assert abs(p["global_rel"] - a["global_rel"]) < 1e-12
    out = tmp_path / "o"
    code = main(["--ours", str(FIX / "ours"), "--ref", str(FIX / "ref"), "--out", str(out)])
    r = json.loads((out / "report.json").read_text())
    assert code == (0 if r["ok"] else 1) and r["version"] == 2
    # mismatched image sizes are an input error (exit 2), not a number
    bad = tmp_path / "bad"
    bad.mkdir()
    write_pfm(bad / "batch_000.pfm", np.ones((16, 16, 3), np.float32))
    write_pfm(bad / "batch_001.pfm", np.ones((16, 16, 3), np.float32))
    assert main(["--ours", str(FIX / "ours"), "--ref", str(bad), "--out", str(tmp_path / "o2")]) == 2


def _make(tmp: Path, bias: float, seed: int) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    h, w, R, B = 64, 64, 2048, 16
    truth = 0.5 + 0.4 * np.sin(np.linspace(0, 3, w))[None, :, None] * np.ones((h, w, 3))
    ref = tmp / "ref"
    ref.mkdir()
    for b in range(B):
        write_pfm(ref / f"batch_{b:03d}.pfm", (truth + rng.normal(0, 0.02, truth.shape)).astype(np.float32))
    ours = tmp / "ours"
    ours.mkdir()
    rows = truth[None] * (1 + bias) + rng.normal(0, 0.3, (R, h, w, 3))
    # the chain writer's format: per-chain SUMS (restir-chain-run.ts / stats.replicates_from_sums)
    np.savez(ours / "ensemble.npz", tiles16=S.tile_sums(rows, 16), **{"global": rows.sum(axis=(1, 2))},
             count=np.array(R), channels=np.array(["R", "G", "B"]), height=np.array(h), width=np.array(w))
    return ours, ref


def test_unbiased_small_and_planted_detected(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    o, r = _make(tmp_path / "a", 0.0, 1)
    res = evaluate(o, r)
    assert res["resolvable"] and res["ok"], res
    assert res["mean_tile_bias"] < 0.01
    o, r = _make(tmp_path / "b", 0.05, 2)
    res = evaluate(o, r)
    assert not res["ok_mean"] and not res["ok_global"]
    assert abs(res["global_rel"] - 0.05) < 0.005
    assert abs(res["mean_tile_bias"] - 0.05) < 0.005
