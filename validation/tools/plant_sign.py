#!/usr/bin/env python3
"""Planted-control checks of M5 with PREDICTED SIGNS (restir-temporal-api.md §6.5, TD29). OWNER T-E.

A temporal plant passes only if it is
  (1) detected as in M4: the rendered plant (ReSTIR chains with the plant flag, ensemble.npz at the test frame) vs a 4x
      PT reference fails the Stage-dyn gate in >= 9/10 half-size repeats while the PT A/A control passes in >= 9/10
      (rows_plant_detection: half of the planted CHAINS vs a PT half), on at least one predicted frame, and the full comparison (compare.py, run by the
      gate) is not `pass`; and
  (2) its predicted sign holds: the relative Y difference Delta = (mean_ours - mean_ref)/mean_ref over the predicted
      region (a mask region of the test, or "global") at EVERY predicted frame has the predicted sign with one-sided
      z = Delta/SE >= 3; and, for "only" predictions ("darkening only"), no mask region and no gate tile shows the
      opposite sign with z >= 4.
Predictions with sign "?" (N6; coordinator Q5) and "detect" (U8-3/U8-4 "sign reported", U8-2t/5t "detected") only need
detection; their Delta and z are reported.

Usage (one plant, several frames):
  plant_sign.py --spec plant.json --out OUT
plant.json:
  {"name": "N2", "test": "test.json",               # stage dyn / B, tile, masks (the same mask order as the npz)
   "frames": [{"frame": 14, "ours": DIR_with_ensemble.npz, "ref": PT_DIR_4x, "test": optional per-frame test.json}, ...],
   "predict": [{"frame": 14, "region": "M_light:0", "sign": "-"}, ...],
   "only": "-" | "+" | null, "repeats": 10, "seed": 0}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import stats as S  # noqa: E402
from compare import build_spec, discover, load_masks, load_replicates, load_stack, write_json  # noqa: E402

Z_SIGN, Z_ONLY = 3.0, 4.0


def region_delta(ours: S.Replicates, ref: S.Replicates, region: str, ch: int = 0) -> dict:
    """Relative Welch Δ of the channel `ch` (0 = Y) on `region` ('global' or a mask name)."""
    if region == "global":
        x, r = ours.glob[:, ch], ref.glob[:, ch]
    else:
        if ours.masks is None or ref.masks is None:
            raise ValueError(f"region {region}: no masks on one side")
        if region not in ours.mask_names or region not in ref.mask_names:
            raise ValueError(f"region {region} not in masks {ours.mask_names} / {ref.mask_names}")
        x, r = ours.masks[:, ours.mask_names.index(region), ch], ref.masks[:, ref.mask_names.index(region), ch]
    sx, sr = S.summarize(x[:, None]), S.summarize(r[:, None])
    d, se, t, nu = S.welch(sx.mean, sx.se, sx.nu, sr.mean, sr.se, sr.nu)
    den = float(sr.mean[0])
    rel = float(d[0]) / den if den > 0 else float("nan")
    se_rel = float(se[0]) / den if den > 0 else float("nan")
    z = float(t[0])
    return dict(region=region, rel=rel, se_rel=se_rel, z=z, nu=float(nu[0]), ref_mean=den)


def opposite_hits(ours: S.Replicates, ref: S.Replicates, sign: str, tile: int, z_min: float = Z_ONLY) -> list[dict]:
    """Mask regions and gate tiles whose Y difference has the sign opposite to `sign` with |z| >= z_min."""
    want = 1.0 if sign == "+" else -1.0
    hits = []
    for name in (ours.mask_names if ours.masks is not None else ()):
        r = region_delta(ours, ref, name)
        if -want * r["z"] >= z_min:
            hits.append(dict(kind="mask", **r))
    xo, xr = S.summarize(ours.tiles[tile][..., 0]), S.summarize(ref.tiles[tile][..., 0])
    d, se, t, _ = S.welch(xo.mean, xo.se, xo.nu, xr.mean, xr.se, xr.nu)
    for (ty, tx) in zip(*np.nonzero(-want * t >= z_min)):
        hits.append(dict(kind="tile", tile=[int(ty), int(tx)], z=float(t[ty, tx]),
                         rel=float(d[ty, tx] / xr.mean[ty, tx]) if xr.mean[ty, tx] > 0 else None))
    return hits


def rows_plant_detection(ref: S.Replicates, planted: S.Replicates, spec: S.GateSpec, n_repeats: int = 10, rng=0) -> dict:
    """Rendered-plant detection with chain rows (M4 rule adapted to R >> B, Changelog E-5): per repeat the PT reference
    splits into disjoint halves A/B and HALF of the planted chains (random) is compared with B (must FAIL); A vs B is the
    control (must PASS); both in >= ceil(0.9 * n_repeats) repeats."""
    rng = np.random.default_rng(rng)
    required = math.ceil(0.9 * n_repeats)
    sp = spec.with_(min_replicates=2)
    fails = ctrl = 0
    hist: dict[str, int] = {}
    for _ in range(n_repeats):
        a, b = S._halves(ref.n, rng)
        p = np.sort(rng.permutation(planted.n)[: planted.n // 2])
        rp = S.evaluate_gate(planted.subset(p), ref.subset(b), sp)
        rc = S.evaluate_gate(ref.subset(a), ref.subset(b), sp)
        fails += not rp.passed
        ctrl += rc.passed
        for n in rp.summary["failed_checks"]:
            hist[n] = hist.get(n, 0) + 1
    return dict(n_repeats=n_repeats, required=required, gate_fail_count=fails, control_pass_count=ctrl, half_planted=planted.n // 2,
                half_ref=ref.n // 2, detected=fails >= required, powered=ctrl >= required, calibrated=bool(fails >= required and ctrl >= required),
                failed_checks_histogram=dict(sorted(hist.items(), key=lambda kv: -kv[1])))


def sign_holds(p: dict, r: dict) -> bool:
    s = p["sign"]
    if s in ("?", "detect"):
        return True
    return (r["z"] >= Z_SIGN) if s == "+" else (-r["z"] >= Z_SIGN)


def load_side(d: Path, test: dict, test_dir: Path, spec: S.GateSpec, frame=None) -> tuple[S.Replicates, np.ndarray | None]:
    src = discover(Path(d), frame)
    tiles = sorted({16, 32, 64, spec.tile})
    if src.kind == "npz":
        reps, _ = load_replicates(src, spec.channels, tiles)
        return reps, None
    stack = load_stack(src)
    h, w = stack.shape[1:3]
    masks, names = load_masks(test, test_dir, h, w)
    return S.aggregate_stack(stack, spec.channels, tiles, masks, names), stack


def run(spec_path: Path, out: Path) -> dict:
    cfg = json.loads(Path(spec_path).read_text())
    base = Path(spec_path).parent
    res_frames = []
    detected_any = False
    signs_ok = True
    only_ok = True
    evaluated = 0
    for fr in cfg["frames"]:
        tpath = base / fr.get("test", cfg["test"])
        test = json.loads(tpath.read_text())
        spec, _ = build_spec(test)
        ours, _ = load_side(base / fr["ours"] if not Path(fr["ours"]).is_absolute() else Path(fr["ours"]), test, tpath.parent, spec)
        ref, _ = load_side(base / fr["ref"] if not Path(fr["ref"]).is_absolute() else Path(fr["ref"]), test, tpath.parent, spec)
        if (ours.mask_names or ()) != (ref.mask_names or ()):
            raise ValueError(f"frame {fr['frame']}: mask names differ: ours {ours.mask_names} ref {ref.mask_names}")
        det = rows_plant_detection(ref, ours, spec, n_repeats=int(cfg.get("repeats", 10)), rng=int(cfg.get("seed", 0)) + int(fr["frame"]))
        detected_any |= bool(det["calibrated"])
        preds = []
        for p in [p for p in cfg["predict"] if int(p["frame"]) == int(fr["frame"])]:
            if p["region"] != "global" and p["region"] not in (ours.mask_names or ()):
                # the region is empty at this frame (dyn_masks dropped it): not evaluable, recorded (Changelog E-6)
                preds.append(dict(p, holds=None, note="region empty at this frame (dropped by dyn_masks.py)"))
                continue
            r = region_delta(ours, ref, p["region"])
            ok = sign_holds(p, r)
            signs_ok &= ok
            evaluated += 1
            preds.append(dict(p, **r, holds=ok))
        hits = opposite_hits(ours, ref, cfg["only"], spec.tile) if cfg.get("only") in ("+", "-") else []
        only_ok &= not hits
        res_frames.append(dict(frame=fr["frame"], detection={k: v for k, v in det.items() if k != "failed_checks_histogram"},
                               failed_checks=det["failed_checks_histogram"], predictions=preds, only_hits=hits,
                               global_Y=region_delta(ours, ref, "global")))
    ok = bool(detected_any and signs_ok and only_ok and evaluated > 0)
    rep = dict(tool="plant_sign.py", name=cfg.get("name"), ok=ok, evaluated_predictions=evaluated, detected=bool(detected_any), signs_hold=bool(signs_ok),
               only=cfg.get("only"), only_ok=bool(only_ok), z_sign=Z_SIGN, z_only=Z_ONLY, frames=res_frames,
               rule="detected (>= 9/10 half-size repeats, PT A/A >= 9/10) on >= 1 predicted frame AND the predicted sign with "
                    "one-sided z >= 3 at every predicted frame AND (only-predictions) no opposite-sign region/tile with z >= 4")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report.json", rep)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rep = run(a.spec, a.out)
    print(json.dumps({"name": rep["name"], "ok": rep["ok"], "detected": rep["detected"], "signs_hold": rep["signs_hold"], "only_ok": rep["only_ok"],
                      "predictions": [dict(frame=f["frame"], **{k: p.get(k) for k in ("region", "sign", "rel", "z", "holds")}) for f in rep["frames"] for p in f["predictions"]]},
                     default=lambda x: None if isinstance(x, float) and not math.isfinite(x) else x))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
