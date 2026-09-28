"""Check a Cycles reference EXR with OpenImageIO: float32, ZIP, RGB, all finite; print stats.

Runs in Blender's bundled python (has numpy + OpenImageIO) or inside Blender itself:
  /Applications/Blender.app/Contents/Resources/5.1/python/bin/python3.13 validation/blender/verify_exr.py a.exr [b.exr ...]
Rows are returned top-down (EXR/OIIO scanline 0 = first row of the array).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import OpenImageIO as oiio


class ExrCheckError(AssertionError):
    pass


def read_exr(path: str) -> tuple[np.ndarray, Any]:
    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise ExrCheckError(f"{path}: {oiio.geterror()}")
    try:
        spec = inp.spec()
        px = inp.read_image(0, 0, 0, spec.nchannels, "float")
    finally:
        inp.close()
    if px is None:
        raise ExrCheckError(f"{path}: read failed")
    return np.asarray(px, dtype=np.float32).reshape(spec.height, spec.width, spec.nchannels), spec


def verify_exr(path: str, shape: tuple[int, int] | None = None, channels: tuple[str, ...] = ("R", "G", "B")) -> dict[str, Any]:
    px, spec = read_exr(path)
    info: dict[str, Any] = {
        "path": path,
        "shape": list(px.shape),
        "file_format": str(spec.format),  # native pixel type in the file
        "dtype": str(px.dtype),
        "channels": list(spec.channelnames),
        "compression": spec.get_string_attribute("compression"),
        "finite": bool(np.isfinite(px).all()),
        "mean": [float(x) for x in px.reshape(-1, px.shape[2]).mean(axis=0)],
        "min": float(px.min()),
        "max": float(px.max()),
    }
    errs = []
    if info["file_format"] != "float":
        errs.append(f"pixel type {info['file_format']} != float (32-bit)")
    if tuple(info["channels"]) != channels:
        errs.append(f"channels {info['channels']} != {list(channels)}")
    if info["compression"] != "zip":
        errs.append(f"compression {info['compression']!r} != 'zip'")
    if not info["finite"]:
        errs.append("non-finite pixels")
    if info["min"] < 0:
        errs.append(f"negative pixel {info['min']}")
    if shape is not None and tuple(px.shape[:2]) != tuple(shape):
        errs.append(f"shape {px.shape[:2]} != {shape}")
    if errs:
        raise ExrCheckError(f"{path}: " + "; ".join(errs))
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--shape", type=int, nargs=2, metavar=("H", "W"))
    args = ap.parse_args()
    ok = True
    for p in args.paths:
        try:
            print(json.dumps(verify_exr(p, tuple(args.shape) if args.shape else None)))
        except ExrCheckError as e:
            ok = False
            print(f"FAIL {e}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
