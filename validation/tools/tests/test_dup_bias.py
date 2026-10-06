"""dup_bias.py (Gate 5, restir-m6-api.md MD15): the noise-debiased tile bias resolves a planted 5 % bias and stays
below 1 % on unbiased synthetic data of the gate's size."""
from pathlib import Path

import numpy as np

from dup_bias import evaluate
from imageio_util import write_pfm


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
    t16 = rows.reshape(R, h // 16, 16, w // 16, 16, 3).mean(axis=(2, 4))
    np.savez(ours / "ensemble.npz", tiles16=t16, **{"global": rows.mean(axis=(1, 2))})
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
