#!/usr/bin/env python3
"""Replicate sets vs a package's analytic "expected" (plan §7.2 calibration scenes; validation/scenes/make-m3a.ts).

  analytic_check.py --dir REPLICATES --package validation/scenes/c0f_furnace_b3_256 [--frame F] [--json]

REPLICATES is any compare.py input directory (our PT batches or Cycles seeds). Supported "expected" kinds:
  constant      every pixel = value (C0f furnace L_e(1 - rho^(b+2))/(1 - rho)): image mean per channel.
  env-analytic  (M3c) the image written by validation/tools/env-expected.ts, passed with --expected-image FILE.pfm
                (f64 ray casting + env quadrature of the package; C0q/C0r/C0s).
  direct-plane  b = 0 direct lighting of the Lambert floor y = 0 (the only geometry) by one light: the exact
                box-filtered image is built by f64 supersampling (8x8 per pixel) of the closed form
                point: rho/pi * Phi/(4 pi) * h/d^3; rect/disk: rho/pi * L_e * E_polygon (disk = 512-gon, area
                corrected), one-sided. Sun / spread / occluded scenes are not imaged (their "expected" is text only).
Rule (as compare.py TOST, delta 0.5% global / 2% per 32^2 tile, dark-tile absolute margin): |D| + t_{0.99,n-1} SE
< margin, plus a bias test |D|/SE < 4.5 (the analytic side has no variance). The supersampling error of the analytic
image is far below both. Exit 0 pass, 1 fail, 2 input error / not applicable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare as C  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722])


def camera_dirs(cam: dict, W: int, H: int, ss: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit world directions (H, W, ss*ss, 3) of the math.md#raster camera, subpixel grid (i+.5)/ss."""
    M = np.array(cam["matrix"], float).reshape(4, 4).T
    ty = math.tan(cam["yfov"] / 2)
    tx = ty * W / H
    s = (np.arange(ss) + 0.5) / ss
    su, sv = np.meshgrid(s, s)
    c = np.arange(W)[None, :, None] + su.reshape(1, 1, -1)
    r = np.arange(H)[:, None, None]
    v = sv.reshape(1, 1, -1)
    dx = (2 * c / W - 1) * tx
    dy = (2 * (H - 1 - r + v) / H - 1) * ty
    dx, dy = np.broadcast_arrays(dx, dy)
    d = np.stack([dx, dy, -np.ones_like(dx)], -1) @ M[:3, :3].T
    return M[:3, 3], d / np.linalg.norm(d, axis=-1, keepdims=True)


def polygon_irradiance(x: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Irradiance at points x (N,3) on the floor (normal +Y) from a unit-radiance polygon (V,3) above the horizon."""
    s = np.zeros(len(x))
    V = len(verts)
    for i in range(V):
        a = verts[i] - x
        b = verts[(i + 1) % V] - x
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        cr = np.cross(a, b)
        cl = np.linalg.norm(cr, axis=1)
        ang = np.arccos(np.clip((a * b).sum(1), -1, 1))
        s += np.where(cl > 0, ang * cr[:, 1] / np.where(cl > 0, cl, 1), 0)
    return 0.5 * np.abs(s)


def direct_plane_image(sj: dict, ss: int = 8) -> np.ndarray:
    exp = sj["expected"]
    W, H = sj["render"]["width"], sj["render"]["height"]
    if len(sj["lights"]) != 1 or exp.get("light") not in ("point", "rect", "disk"):
        raise ValueError(f"direct-plane {exp.get('light')!r}: not imaged")
    rho = np.asarray(exp["rho"] if isinstance(exp["rho"], list) else [exp["rho"]] * 3, float)
    L = sj["lights"][0]
    M = np.array(L["matrix"], float).reshape(4, 4).T
    c, X, Y, aL = M[:3, 3], M[:3, 0], M[:3, 1], -M[:3, 2]
    color = np.asarray(L["color"], float) * L["power"] * 2.0 ** L.get("exposure", 0)
    o, d = camera_dirs(sj["camera"], W, H, ss)
    t = -o[1] / d[..., 1]
    x = o + t[..., None] * d
    hs = float(exp.get("plane_half_size", 3))
    hit = (t > 0) & (np.abs(x[..., 0]) <= hs) & (np.abs(x[..., 2]) <= hs)
    xs = x[hit]
    if exp["light"] == "point":
        dv = c - xs
        dist = np.linalg.norm(dv, axis=1)
        val = (1 / math.pi) * (1 / (4 * math.pi)) * dv[:, 1] / dist ** 3
    else:
        if exp["light"] == "rect":
            hx, hy = L["sizeX"] / 2, L["sizeY"] / 2
            verts = np.array([c - hx * X - hy * Y, c + hx * X - hy * Y, c + hx * X + hy * Y, c - hx * X + hy * Y])
            A = L["sizeX"] * L["sizeY"]
        else:
            n = 512
            R = L["sizeX"] / 2
            ang = 2 * math.pi * np.arange(n) / n
            # circumscribed-area-matched n-gon: same area as the disk (radius scaled by sqrt(pi / (n/2 sin(2pi/n))))
            Rp = R * math.sqrt(math.pi / (n / 2 * math.sin(2 * math.pi / n)))
            verts = c + Rp * (np.cos(ang)[:, None] * X + np.sin(ang)[:, None] * Y)
            A = math.pi * R * R
        if L.get("spread", math.pi) < math.pi - 1e-9:
            raise ValueError("spread < pi: not imaged")
        Le = 1 / (math.pi * A)
        front = ((xs - c) @ aL) > 0
        val = np.where(front, (1 / math.pi) * Le * polygon_irradiance(xs, verts), 0.0)
    img = np.zeros(hit.shape)
    img[hit] = val
    img = img.mean(-1)
    return img[..., None] * (rho * color)[None, None, :]


NUM_EPS = 1e-4  # relative accuracy claimed for the analytic image (8x8 supersampling, 512-gon disk); floors the bias test


def check(reps: np.ndarray, E: np.ndarray, tile: int = 32, d_glob: float = 0.005, d_tile: float = 0.02) -> dict:
    n = reps.shape[0]
    tq = sps.t.ppf(0.99, n - 1)
    out: dict = {"n": int(n), "channels": {}}
    ok = True
    H, W = E.shape[:2]
    for ci, ch in enumerate(("R", "G", "B", "Y")):
        x = reps @ LUMA if ch == "Y" else reps[..., ci]
        e = E @ LUMA if ch == "Y" else E[..., ci]
        m = x.reshape(n, -1).mean(1)
        eg = e.mean()
        D, se = m.mean() - eg, m.std(ddof=1) / math.sqrt(n)
        sz = math.hypot(se, NUM_EPS * eg)
        g_ok = abs(D) + tq * se < d_glob * eg and abs(D) / sz < 4.5
        worst, fails = 0.0, 0
        for y0 in range(0, H, tile):
            for x0 in range(0, W, tile):
                xt = x[:, y0:y0 + tile, x0:x0 + tile].reshape(n, -1).mean(1)
                et = e[y0:y0 + tile, x0:x0 + tile].mean()
                Dt, st = xt.mean() - et, xt.std(ddof=1) / math.sqrt(n)
                margin = d_tile * max(et, 0.05 * eg)
                szt = math.hypot(st, NUM_EPS * max(et, 0.05 * eg))
                t_ok = abs(Dt) + tq * st < margin and abs(Dt) / szt < 4.5 + math.sqrt(2 * math.log(max(H * W / tile / tile, 1)))
                fails += not t_ok
                worst = max(worst, abs(Dt) / max(et, 0.05 * eg, 1e-30))
        ok &= g_ok and fails == 0
        out["channels"][ch] = {"expected_mean": eg, "rel": D / eg if eg else 0.0, "z": D / sz if sz else 0.0, "se_rel": se / eg if eg else 0.0,
                               "global_ok": bool(g_ok), "tiles_failed": fails, "worst_tile_rel": worst}
    out["ok"] = bool(ok)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--frame", type=int)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--expected-image", type=Path, help="expected image (PFM/EXR) for expected.kind 'env-analytic'")
    a = ap.parse_args(argv)
    sj = json.loads((a.package / "scene.json").read_text())
    exp = sj.get("expected") or {}
    src = C.discover(a.dir, a.frame)
    reps = np.stack([C.read_image(p)[..., :3] for _, p in src.files]).astype(np.float64)
    if exp.get("kind") == "constant":
        E = np.broadcast_to(np.asarray(exp["value"], float), reps.shape[1:]).copy()
        res = check(reps, E, tile=max(reps.shape[1:3]))  # one "tile" = the image: the constant must hold on average
    elif exp.get("kind") == "env-analytic":
        if a.expected_image is None:
            print(json.dumps({"ok": None, "skipped": "env-analytic needs --expected-image"}) if a.json else "skipped: --expected-image")
            return 2
        E = C.read_image(a.expected_image)[..., :3].astype(np.float64)
        if E.shape != reps.shape[1:]:
            raise SystemExit(f"expected image {E.shape} != replicates {reps.shape[1:]}")
        res = check(reps, E)
        res["expected_image"] = str(a.expected_image)
    elif exp.get("kind") == "direct-plane":
        try:
            E = direct_plane_image(sj)
        except ValueError as e:
            print(json.dumps({"ok": None, "skipped": str(e)}) if a.json else f"skipped: {e}")
            return 2
        res = check(reps, E)
    else:
        print(json.dumps({"ok": None, "skipped": f"expected kind {exp.get('kind')!r}"}) if a.json else "no analytic expectation")
        return 2
    res.update(kind=exp["kind"], dir=str(a.dir), package=str(a.package))
    if a.json:
        print(json.dumps(res))
    else:
        for ch, v in res["channels"].items():
            print(f"{ch}: expected {v['expected_mean']:.6g} rel {v['rel']:+.3e} (z {v['z']:+.2f}, SE {v['se_rel']:.2e}) global {'ok' if v['global_ok'] else 'FAIL'}, "
                  f"tiles failed {v['tiles_failed']}, worst tile {v['worst_tile_rel']:.3e}")
        print("PASS" if res["ok"] else "FAIL")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
