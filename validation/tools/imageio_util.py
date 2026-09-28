"""Image IO for the validation pipeline (plan §1.2, §5 M0, §7.3).

Convention: every array returned or accepted here is float, shape HxWxC, ROW 0 = TOP.
- PFM stores scanlines bottom-to-top; scale < 0 means little-endian (mirror of src/core/io/pfm.ts).
- EXR goes through OpenImageIO, which presents scanlines top-down (Blender/Cycles EXRs are top-down).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np

_PFM_HEADER = re.compile(rb"^(PF|Pf)\s+(\d+)\s+(\d+)\s+([-+0-9.eE]+)\s")


def read_pfm(path: str | Path) -> np.ndarray:
    data = Path(path).read_bytes()
    m = _PFM_HEADER.match(data)
    if not m:
        raise ValueError(f"{path}: not a PFM (bad header)")
    channels = 3 if m.group(1) == b"PF" else 1
    w, h, scale = int(m.group(2)), int(m.group(3)), float(m.group(4))
    dtype = "<f4" if scale < 0 else ">f4"
    n = w * h * channels
    body = np.frombuffer(data, dtype=dtype, count=n, offset=m.end())
    # stored bottom-to-top -> flip to row 0 = top
    return body.reshape(h, w, channels)[::-1].astype(np.float64)


def write_pfm(path: str | Path, img: np.ndarray) -> None:
    """Inverse of read_pfm; byte-identical to encodePFM in src/core/io/pfm.ts (little-endian, scale -1.0)."""
    a = _as_hwc(img)
    c = a.shape[2]
    if c not in (1, 3):
        raise ValueError(f"PFM supports 1 or 3 channels, got {c}")
    h, w = a.shape[:2]
    header = f"{'PF' if c == 3 else 'Pf'}\n{w} {h}\n-1.0\n".encode("ascii")
    body = np.ascontiguousarray(a[::-1], dtype="<f4").tobytes()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(header + body)


def read_exr(path: str | Path, *, drop_alpha: bool = False, channels: Sequence[str] | None = None) -> np.ndarray:
    """Read an EXR (or any OIIO format) as float64 HxWxC, row 0 = top.

    `channels` selects channels by name (e.g. ["R", "G", "B"] or multilayer names like
    "ViewLayer.Combined.R"); `drop_alpha` removes a channel named A/alpha (or the 4th of RGBA).
    """
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise OSError(f"{path}: {oiio.geterror()}")
    try:
        spec = inp.spec()
        if (spec.x, spec.y, spec.width, spec.height) != (spec.full_x, spec.full_y, spec.full_width, spec.full_height):
            raise ValueError(f"{path}: data window != display window; crop/pad not supported")
        pixels = inp.read_image(0, 0, 0, spec.nchannels, "float")
        if pixels is None:
            raise OSError(f"{path}: {inp.geterror()}")
        names = list(spec.channelnames)
    finally:
        inp.close()
    a = _as_hwc(np.asarray(pixels, dtype=np.float64))
    if channels is not None:
        missing = [c for c in channels if c not in names]
        if missing:
            raise KeyError(f"{path}: channels {missing} not in {names}")
        a = a[:, :, [names.index(c) for c in channels]]
    elif drop_alpha:
        keep = [i for i, n in enumerate(names) if n.split(".")[-1].upper() not in ("A", "ALPHA")]
        a = a[:, :, keep]
    return a


def write_exr(path: str | Path, img: np.ndarray, channel_names: Sequence[str] | None = None) -> None:
    """Write float32 EXR with ZIP compression, row 0 = top."""
    import OpenImageIO as oiio

    a = np.ascontiguousarray(_as_hwc(img), dtype=np.float32)
    h, w, c = a.shape
    spec = oiio.ImageSpec(w, h, c, "float")
    if channel_names is None:
        channel_names = {1: ["Y"], 2: ["Y", "A"], 3: ["R", "G", "B"], 4: ["R", "G", "B", "A"]}.get(c)
    if channel_names is not None:
        spec.channelnames = tuple(channel_names)
    spec.attribute("compression", "zip")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = oiio.ImageOutput.create(str(p))
    if out is None:
        raise OSError(f"{path}: {oiio.geterror()}")
    try:
        if not out.open(str(p), spec) or not out.write_image(a):
            raise OSError(f"{path}: {out.geterror()}")
    finally:
        out.close()


def read_image(path: str | Path, *, drop_alpha: bool = False) -> np.ndarray:
    """Dispatch on extension: .pfm -> read_pfm, anything else -> OpenImageIO (EXR etc.)."""
    if Path(path).suffix.lower() == ".pfm":
        return read_pfm(path)
    return read_exr(path, drop_alpha=drop_alpha)


def write_image(path: str | Path, img: np.ndarray) -> None:
    if Path(path).suffix.lower() == ".pfm":
        write_pfm(path, img)
    else:
        write_exr(path, img)


def _as_hwc(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3:
        raise ValueError(f"expected HxW or HxWxC, got shape {a.shape}")
    return a
