"""(6) End-to-end: synthetic ours/ref directories in the documented layouts -> compare.py -> report."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import compare
import stats as S
from conftest import TOOLS, base_image, replicate_stack
from imageio_util import write_exr, write_pfm

H = W = 128


def write_ours(d: Path, stack: np.ndarray, seeds, **meta) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(stack):
        write_pfm(d / f"batch_{i:03d}.pfm", img)
    (d / "meta.json").write_text(json.dumps(dict(seeds=list(seeds), spp_per_batch=256, config_hash="cfg-ours-abc",
                                                 chrome_version="140.0", **meta)))
    return d


def write_ref(d: Path, stack: np.ndarray, seeds, frame: int | None = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for s, img in zip(seeds, stack):
        write_exr(d / (f"seed_{s:03d}.exr" if frame is None else f"f{frame}_s{s}.exr"), img)
    (d / "manifest.json").write_text(json.dumps({"blender": "5.1.2", "scene_hash": "scene-123",
                                                 "cache": {"config_hash": "cfg-ref-xyz"}, "scene.cycles.samples": 4096}))
    return d


def write_test(d: Path, **kw) -> Path:
    t = dict(name="synthetic-stageA", stage="A", channels=["Y", "R", "G", "B"], n_units=4, tier="tight",
             masks=[dict(name="centre", rect=[40, 40, 88, 88])])
    t.update(kw)
    p = d / "test.json"
    p.write_text(json.dumps(t))
    return p


@pytest.fixture
def dirs(tmp_path):
    rng = np.random.default_rng(2024)
    base = base_image(H, W)
    ours = replicate_stack(rng, 16, base, 0.15)
    ref = replicate_stack(rng, 16, base, 0.15)
    return tmp_path, base, ours, ref


def run(argv) -> int:
    return compare.main([str(a) for a in argv])


def test_e2e_pass_pfm_vs_exr(dirs):
    tmp, base, ours, ref = dirs
    o = write_ours(tmp / "ours", ours, range(100, 116))
    r = write_ref(tmp / "ref", ref, range(16))
    t = write_test(tmp)
    out = tmp / "out"
    assert run(["--ours", o, "--ref", r, "--test", t, "--out", out]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["status"] == "pass" and rep["gate_passed"]
    assert rep["provenance"]["ours"]["hashes"] == {"config_hash": "cfg-ours-abc"}
    assert rep["provenance"]["ref"]["hashes"] == {"scene_hash": "scene-123", "cache.config_hash": "cfg-ref-xyz"}
    assert rep["provenance"]["ref"]["versions"]["blender"] == "5.1.2"
    assert rep["seeds"]["ref"] == [str(s) for s in range(16)]
    assert rep["tier"] == "tight"
    names = {(c["name"], c["channel"]) for c in rep["checks"]}
    for ch in "YRGB":
        for chk in ("tost_global", "tost_tiles", "tost_masks", "sidak_tiles", "chi2_red", "mean_t", "ks_ad"):
            assert (chk, ch) in names
    assert set(rep["mdb"]) == set("YRGB") and rep["mdb"]["Y"]["global_"] > 0
    for k in ("relMSE", "relMSE_corr", "MAPE", "RMSRB", "HDR_FLIP", "noise_floor_ref_halves"):
        assert k in rep["metrics"], k
    for f in ("side_by_side.png", "rel_diff.png", "t_map.png", "tile_t.png", "flip.png", "z_hist_qq.png", "index.html"):
        assert (out / f).is_file(), f
    html = (out / "index.html").read_text()
    assert "synthetic-stageA" in html and "✓ pass" in html
    # report.json is strict JSON (no NaN / Infinity)
    json.loads((out / "report.json").read_text(), parse_constant=lambda c: pytest.fail(f"non-finite {c}"))


def test_e2e_orientation_flip_is_caught(dirs):
    """PFM is stored bottom-to-top; a reader/writer that forgets the flip must fail the gate."""
    tmp, base, ours, ref = dirs
    o = write_ours(tmp / "ours", ours[:, ::-1], range(100, 116))
    r = write_ref(tmp / "ref", ref, range(16))
    assert run(["--ours", o, "--ref", r, "--test", write_test(tmp), "--out", tmp / "out"]) == 1


def test_e2e_planted_bias_fails_and_confirmatory_rerun(dirs):
    tmp, base, ours, ref = dirs
    o = write_ours(tmp / "ours_bad", ours * 1.01, range(100, 116))
    r = write_ref(tmp / "ref", ref, range(16))
    t = write_test(tmp)
    assert run(["--ours", o, "--ref", r, "--test", t, "--out", tmp / "out1"]) == 1
    rep1 = json.loads((tmp / "out1" / "report.json").read_text())
    assert rep1["status"] == "rerun_required"
    assert any(f.startswith("tost_global") for f in rep1["failed_checks"])
    # confirmatory re-run on DISJOINT seeds that passes -> pass_on_rerun (exit 0)
    rng = np.random.default_rng(77)
    o2 = write_ours(tmp / "ours2", replicate_stack(rng, 16, base, 0.15), range(200, 216))
    r2 = write_ref(tmp / "ref2", replicate_stack(rng, 16, base, 0.15), range(300, 316))
    assert run(["--ours", o2, "--ref", r2, "--test", t, "--out", tmp / "out2",
                "--rerun-of", tmp / "out1" / "report.json"]) == 0
    assert json.loads((tmp / "out2" / "report.json").read_text())["status"] == "pass_on_rerun"
    # a "re-run" that reuses the first run's seeds is not a confirmatory re-run
    assert run(["--ours", o2, "--ref", r, "--test", t, "--out", tmp / "out3",
                "--rerun-of", tmp / "out1" / "report.json"]) == 1
    assert json.loads((tmp / "out3" / "report.json").read_text())["status"] == "invalid_rerun"


def test_e2e_frame_seed_layout_and_npz_ensemble(dirs):
    tmp, base, ours, ref = dirs
    r = tmp / "ref"
    write_ref(r, ref, range(16), frame=3)
    write_ref(r, ref * 1.05, range(16), frame=4)
    # ours as a pre-reduced ensemble (.npz, ensembleStats format)
    e = tmp / "ens"
    e.mkdir()
    np.savez(e / "ensemble.npz", **{f"tiles{s}": S.tile_sums(ours, s) for s in (16, 32, 64)},
             **{"global": ours.sum(axis=(1, 2)), "pixel_sum": ours.sum(0), "pixel_sumsq": (ours ** 2).sum(0),
                "count": np.array(16), "channels": np.array(["R", "G", "B"]),
                "masks": np.einsum("nhwc,mhw->nmc", ours, _rect_mask()), "mask_pixels": _rect_mask().sum(axis=(1, 2)),
                "mask_names": np.array(["centre"])})
    (e / "meta.json").write_text(json.dumps({"seeds": list(range(500, 516)), "config_hash": "ens"}))
    t = write_test(tmp)
    assert run(["--ours", e, "--ref", r, "--test", t, "--out", tmp / "o3", "--frame", 3]) == 0
    assert run(["--ours", e, "--ref", r, "--test", t, "--out", tmp / "o4", "--frame", 4]) == 1
    # ambiguous frame without --frame is a usage error
    assert run(["--ours", e, "--ref", r, "--test", t, "--out", tmp / "o5"]) == 2


def _rect_mask():
    m = np.zeros((1, H, W))
    m[0, 40:88, 40:88] = 1
    return m


def test_e2e_curve_mode(tmp_path):
    rng = np.random.default_rng(5)
    base = base_image(H, W)
    o = write_ours(tmp_path / "ours", replicate_stack(rng, 32, base, 1.0), range(32))
    r = write_ref(tmp_path / "ref", replicate_stack(rng, 16, base, 0.02), range(16))
    t = write_test(tmp_path, curve={"slope_range": [-1.1, -0.9], "gate": False})
    out = tmp_path / "curve"
    assert run(["--curve", "--ours", o, "--ref", r, "--test", t, "--out", out]) == 0
    rep = json.loads((out / "report.json").read_text())
    ns = [p["N"] for p in rep["curve"]["points"]]
    assert ns == [1, 2, 4, 8, 16, 32]
    assert -1.3 < rep["curve"]["fit"]["slope"] < -0.7
    assert all(p["BNR"] is not None for p in rep["curve"]["points"][1:])
    assert (out / "curve.png").is_file() and (out / "index.html").is_file()


def test_e2e_calibrate_mode(tmp_path):
    rng = np.random.default_rng(9)
    r = write_ref(tmp_path / "ref", replicate_stack(rng, 32, base_image(H, W), 0.12), range(32))
    t = write_test(tmp_path, masks=[])
    out = tmp_path / "cal"
    assert run(["--calibrate", "--ref", r, "--test", t, "--out", out]) == 0
    rep = json.loads((out / "report.json").read_text())
    cal = rep["calibration"]
    assert cal["aa"]["ok"] and cal["aa"]["n_splits"] == 20
    assert [p["name"] for p in cal["plants"]] == ["light x1.0075", "32x32 region +3%"]
    assert all(p["calibrated"] for p in cal["plants"])


def test_cli_subprocess_exit_codes(dirs):
    tmp, base, ours, ref = dirs
    o = write_ours(tmp / "ours", ours, range(100, 116))
    r = write_ref(tmp / "ref", ref, range(16))
    t = write_test(tmp, delta={"global": 0.01})              # loosened δ must be recorded, not silent
    p = subprocess.run([sys.executable, str(TOOLS / "compare.py"), "--ours", str(o), "--ref", str(r),
                        "--test", str(t), "--out", str(tmp / "out")], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "loosened" in p.stderr
    rep = json.loads((tmp / "out" / "report.json").read_text())
    assert rep["notes"]["delta_loosened"]["global"] == {"default": 0.005, "used": 0.01}
    p = subprocess.run([sys.executable, str(TOOLS / "compare.py"), "--ours", str(tmp / "nope"), "--ref", str(r),
                        "--test", str(t), "--out", str(tmp / "out")], capture_output=True, text=True)
    assert p.returncode == 2
