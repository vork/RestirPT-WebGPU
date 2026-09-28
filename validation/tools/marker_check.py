"""Check renders of calibration packages against their analytic answers (plan §5 M2 Gate 1-lite: C0a, C0b, C0p).

  marker_check.py <render_dir> [--package DIR] [--tol-px 0.1] [--json]
  marker_check.py --image img.exr --package DIR --frame 0

<render_dir> is a render_reference.py output (manifest.json names the package and the EXRs); the renderer's
own EXR/PFM outputs can be checked with --image. Expected answers live in the package's scene.json "expected"
(written by validation/blender/calib_scenes.py). Image convention: row 0 = TOP, pixel (c, r) centre (c+.5, r+.5).

Per marker: w = max(dot(px, class/|class|_1) - threshold, 0) over the marker's window; centroid = sum(w·xy)/sum(w);
error = |centroid - expected centroid_px|. Also: the weighted mean colour must point along the class colour
(identifies the marker, so flips/rotations fail), stray mass outside all windows must be ~0, probes (pixels with
exactly known values) must match to probe_tolerance, and "constant" packages must match everywhere.
Exit codes: 0 pass, 1 fail, 2 bad input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imageio_util import read_image  # noqa: E402


def marker_centroid(img: np.ndarray, window: list[int], cls: list[float], threshold: float) -> dict[str, Any]:
    x0, x1, y0, y1 = window
    sub = img[y0:y1, x0:x1, :3].astype(np.float64)
    c = np.asarray(cls, float)
    w = np.clip(sub @ (c / c.sum()) - threshold, 0, None)
    mass = float(w.sum())
    if mass <= 0:
        return {"mass": 0.0, "centroid_px": None, "class_cos": None}
    ys, xs = np.mgrid[y0:y1, x0:x1] + 0.5
    mean_col = (sub * w[..., None]).sum(axis=(0, 1)) / mass
    cos = float(mean_col @ c / (np.linalg.norm(mean_col) * np.linalg.norm(c) + 1e-300))
    return {"mass": mass, "centroid_px": [float((w * xs).sum() / mass), float((w * ys).sum() / mass)], "class_cos": cos}


def check_frame(img: np.ndarray, expected: dict[str, Any], frame_exp: dict[str, Any], tol_px: float | None = None) -> dict[str, Any]:
    kind = expected["kind"]
    H, W = img.shape[:2]
    rgb = img[..., :3].astype(np.float64)
    out: dict[str, Any] = {"kind": kind, "ok": True, "failures": []}
    if kind == "constant":
        v = np.asarray(expected["value"], float)
        err = float(np.abs(rgb - v).max())
        out["max_abs_err"] = err
        if err > float(expected.get("tolerance", 1e-4)):
            out["failures"].append(f"constant: max |px - {v.tolist()}| = {err:.3g} > {expected.get('tolerance', 1e-4)}")
    elif kind == "markers":
        tol = float(tol_px if tol_px is not None else expected.get("tolerance_px", 0.1))
        out["tolerance_px"] = tol
        mask = np.zeros((H, W), bool)
        results = []
        thr_stray = max([m["threshold"] for m in frame_exp["markers"]] + [0.0])
        for m in frame_exp["markers"]:
            x0, x1, y0, y1 = m["window"]
            mask[y0:y1, x0:x1] = True
            if m.get("partial") or "centroid_px" not in m:
                continue
            r = marker_centroid(img, m["window"], m["class"], m["threshold"])
            rec: dict[str, Any] = {"name": m["name"], "expected_px": m["centroid_px"], "got_px": r["centroid_px"],
                                   "center_px": m.get("center_px"), "mass_ratio": r["mass"] / m["mass"] if m.get("mass") else None,
                                   "class_cos": r["class_cos"]}
            if r["centroid_px"] is None:
                rec["err_px"] = float("inf")
                out["failures"].append(f"{m['name']}: no marker in window {m['window']}")
            else:
                d = np.subtract(r["centroid_px"], m["centroid_px"])
                rec["err_px"] = float(np.hypot(*d))
                rec["dxy_px"] = [float(d[0]), float(d[1])]
                if rec["err_px"] > tol:
                    out["failures"].append(f"{m['name']}: centroid error {rec['err_px']:.4f} px > {tol}")
                if r["class_cos"] < 0.99:
                    out["failures"].append(f"{m['name']}: colour {r['class_cos']:.3f} not along class {m['class']}")
                if rec["mass_ratio"] is not None and abs(rec["mass_ratio"] - 1) > 0.02:
                    out["failures"].append(f"{m['name']}: mass ratio {rec['mass_ratio']:.4f} (expected 1 ± 0.02)")
            results.append(rec)
        out["markers"] = results
        out["max_err_px"] = max((r["err_px"] for r in results), default=None)
        stray = np.clip(rgb.max(axis=2) - thr_stray, 0, None)
        stray_mass = float(stray[~mask].sum())
        total = float(sum(m.get("mass", 0.0) for m in frame_exp["markers"])) or 1.0
        out["stray_mass_ratio"] = stray_mass / total
        if stray_mass / total > 1e-3:
            ys, xs = np.nonzero(stray * ~mask)
            out["failures"].append(f"stray bright mass outside marker windows: {stray_mass / total:.3g} of the marker mass "
                                   f"(e.g. pixel ({xs[0]}, {ys[0]}))")
    else:
        raise ValueError(f"unknown expected kind {kind!r}")
    ptol = float(expected.get("probe_tolerance", expected.get("tolerance", 1e-4)))
    perr = 0.0
    for p in frame_exp.get("probes", []):
        c, r = p["pixel"]
        e = float(np.abs(rgb[r, c] - np.asarray(p["value"], float)).max())
        perr = max(perr, e)
        if e > ptol:
            out["failures"].append(f"probe ({c}, {r}) {p.get('what', '')}: got {rgb[r, c].tolist()} want {p['value']} (|err| {e:.3g})")
    out["probes"] = len(frame_exp.get("probes", []))
    out["probe_max_err"] = perr
    out["ok"] = not out["failures"]
    return out


def check_render_dir(render_dir: Path, package: Path | None = None, tol_px: float | None = None) -> dict[str, Any]:
    man = json.loads((render_dir / "manifest.json").read_text())
    pkg = package or Path(man["package"])
    sj = json.loads((pkg / "scene.json").read_text())
    exp = sj.get("expected")
    if exp is None:
        raise ValueError(f"{pkg}: scene.json has no 'expected'")
    per = []
    for r in man["renders"]:
        img = read_image(render_dir / r["file"], drop_alpha=True)
        fe = exp["frames"][str(r["frame"])]
        res = check_frame(img, exp, fe, tol_px)
        res.update(file=r["file"], frame=r["frame"], seed=r["seed"])
        per.append(res)
    errs = [r["max_err_px"] for r in per if r.get("max_err_px") is not None]
    return {"package": sj.get("name"), "render_dir": str(render_dir), "renders": len(per), "ok": all(r["ok"] for r in per),
            "max_err_px": max(errs) if errs else None, "max_probe_err": max((r["probe_max_err"] for r in per), default=None),
            "max_const_err": max((r.get("max_abs_err", 0.0) for r in per), default=None),
            "failures": [f"{r['file']}: {f}" for r in per for f in r["failures"]], "per_render": per}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("render_dir", nargs="?", type=Path)
    ap.add_argument("--package", type=Path)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--tol-px", type=float)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.image:
            if not args.package:
                raise ValueError("--image needs --package")
            sj = json.loads((args.package / "scene.json").read_text())
            res = check_frame(read_image(args.image, drop_alpha=True), sj["expected"], sj["expected"]["frames"][str(args.frame)], args.tol_px)
        else:
            if not args.render_dir:
                raise ValueError("need a render_dir or --image")
            res = check_render_dir(args.render_dir, args.package, args.tol_px)
    except (OSError, ValueError, KeyError) as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(res))
    else:
        print(f"{'OK' if res['ok'] else 'FAIL'} {res.get('package', args.image)}: max centroid err {res.get('max_err_px')} px, "
              f"max probe err {res.get('max_probe_err', res.get('probe_max_err'))}")
        for f in res["failures"][:20]:
            print("  " + f)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
