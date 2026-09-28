"""Orientation checks (plan §5 M0 'IO orientation test'; M2 C0a placeholder).

  orientation_check.py pfm <path>                       # asserts I(c, r) = (c, r, c + r), row 0 = top
  orientation_check.py exr-marker <path> --expect top-left   # brightest red marker's quadrant

Exit codes: 0 = pass, 1 = mismatch, 2 = unreadable/invalid input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imageio_util import read_image  # noqa: E402

QUADRANTS = ("top-left", "top-right", "bottom-left", "bottom-right")


def expected_pattern(width: int, height: int) -> np.ndarray:
    """Mirror of orientationPattern in src/core/io/pfm.ts."""
    r, c = np.mgrid[0:height, 0:width].astype(np.float64)
    return np.stack([c, r, c + r], axis=-1)


def diagnose(img: np.ndarray) -> str:
    """Name the transform that maps the expected pattern onto img, if it is a simple one."""
    h, w = img.shape[:2]
    exp = expected_pattern(w, h)
    for name, t in (("vertically flipped", exp[::-1]), ("horizontally flipped", exp[:, ::-1]),
                    ("rotated 180", exp[::-1, ::-1])):
        if np.array_equal(img, t):
            return name
    return "unknown mismatch"


def check_pfm(path: Path) -> int:
    img = read_image(path)
    h, w, ch = img.shape
    if ch != 3:
        print(f"FAIL {path}: expected 3 channels, got {ch}")
        return 1
    exp = expected_pattern(w, h)
    bad = np.argwhere(np.any(img != exp, axis=-1))
    if bad.size:
        r, c = bad[0]
        print(f"FAIL {path}: {len(bad)}/{w * h} pixels mismatch ({diagnose(img)}); "
              f"first at (c={c}, r={r}): got {img[r, c].tolist()}, want {exp[r, c].tolist()}")
        return 1
    print(f"OK {path}: {w}x{h} orientation pattern, row 0 = top")
    return 0


def find_red_marker(img: np.ndarray, frac: float = 0.5) -> tuple[float, float] | None:
    """Centroid (x, y), row 0 = top, of red-dominant pixels within `frac` of the brightest red value."""
    rgb = img[:, :, :3] if img.shape[2] >= 3 else np.repeat(img, 3, axis=2)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    redness = np.where((r > 0) & (r > 2.0 * np.maximum(g, b)), r - np.maximum(g, b), 0.0)
    peak = redness.max()
    if not np.isfinite(peak) or peak <= 0:
        return None
    ys, xs = np.nonzero(redness >= frac * peak)
    wts = redness[ys, xs]
    return float(np.average(xs, weights=wts)) + 0.5, float(np.average(ys, weights=wts)) + 0.5


def quadrant(x: float, y: float, w: int, h: int) -> str:
    return f"{'top' if y < h / 2 else 'bottom'}-{'left' if x < w / 2 else 'right'}"


def check_exr_marker(path: Path, expect: str | None, as_json: bool) -> int:
    img = read_image(path, drop_alpha=True)
    h, w = img.shape[:2]
    m = find_red_marker(img)
    if m is None:
        print(f"FAIL {path}: no red-dominant pixels found")
        return 1
    q = quadrant(m[0], m[1], w, h)
    ok = expect is None or q == expect
    if as_json:
        print(json.dumps({"path": str(path), "width": w, "height": h, "centroid": m, "quadrant": q,
                          "expect": expect, "ok": ok}))
    else:
        print(f"{'OK' if ok else 'FAIL'} {path}: red marker centroid (x={m[0]:.2f}, y={m[1]:.2f}) "
              f"in {q} (row 0 = top){'' if expect is None else f', expected {expect}'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pfm", help="assert the M0 (c, r, c+r) orientation pattern")
    p.add_argument("path", type=Path)
    e = sub.add_parser("exr-marker", help="report the quadrant of the brightest red marker")
    e.add_argument("path", type=Path)
    e.add_argument("--expect", choices=QUADRANTS)
    e.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "pfm":
            return check_pfm(args.path)
        return check_exr_marker(args.path, args.expect, args.json)
    except (OSError, ValueError, KeyError) as err:
        print(f"ERROR {args.path}: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
