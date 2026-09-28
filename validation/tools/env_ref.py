"""OpenImageIO reference decode for the env loaders (ENV-U1; plan §1.4b, math.md#env-mapping).

  python validation/tools/env_ref.py dump <image.hdr|.exr> <out.f32>

Reads the image with OIIO (the library Blender uses for .hdr and .exr) as float32 and writes the raw
little-endian float32 pixels, RGB(A) interleaved, ROWS TOP-DOWN (OIIO scanline order, row 0 = top of the
picture = zenith for an equirect). Prints one JSON line to stdout:
  {"width": W, "height": H, "channels": C, "channelnames": [...], "format": "...", "compression": "...", "sha256": "..."}
where sha256 is over the dumped bytes. The TS side flips its bottom-up rows before comparing.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import OpenImageIO as oiio


def dump(src: str, dst: str) -> dict:
    inp = oiio.ImageInput.open(src)
    if inp is None:
        raise RuntimeError(f"OIIO cannot open {src}: {oiio.geterror()}")
    spec = inp.spec()
    px = inp.read_image(0, 0, 0, spec.nchannels, oiio.FLOAT)
    inp.close()
    if px is None:
        raise RuntimeError(f"OIIO read failed for {src}: {oiio.geterror()}")
    arr = np.ascontiguousarray(np.asarray(px, dtype="<f4").reshape(spec.height, spec.width, spec.nchannels))
    raw = arr.tobytes()
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Path(dst).write_bytes(raw)
    return {
        "width": spec.width,
        "height": spec.height,
        "channels": spec.nchannels,
        "channelnames": list(spec.channelnames),
        "format": str(spec.format),
        "compression": spec.get_string_attribute("compression"),
        "oiio": oiio.__version__,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[1] != "dump":
        print(__doc__, file=sys.stderr)
        return 2
    print(json.dumps(dump(argv[2], argv[3])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
