"""M7 U-TAN-B (docs/decisions/m7-api.md §6.2): Blender's own MikkTSpace on a scene package mesh.

  Blender -b --factory-startup --python-exit-code 1 -P validation/blender/tangent_dump.py -- --package DIR --out FILE

Builds the package exactly as the reference renders do (build_scene.build_from_package: one mesh, faces in primId order,
custom normals on smooth packages, UV map with v_b = 1 - v), runs Mesh.calc_tangents(uvmap=UVMap) and writes, per
loop (corner 3·primId + k), the tangent in the glTF frame (C^T applied) and the bitangent sign: float32 LE
[tx, ty, tz, sign] × corners. No render, no GPU.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_scene as bs  # noqa: E402


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    built = bs.build_from_package(a.package)
    me = built["mesh"].data
    me.calc_tangents(uvmap=bs.UV_MAP)
    n = len(me.loops)
    t = np.empty(3 * n, np.float32)
    me.loops.foreach_get("tangent", t)
    s = np.empty(n, np.float32)
    me.loops.foreach_get("bitangent_sign", s)
    tb = t.reshape(n, 3)
    tg = np.empty_like(tb)            # Blender (x, y, z) -> glTF (x, z, -y)  (C^T, C = R_x(+90°))
    tg[:, 0], tg[:, 1], tg[:, 2] = tb[:, 0], tb[:, 2], -tb[:, 1]
    out = np.concatenate([tg, s[:, None]], axis=1).astype("<f4")
    Path(a.out).write_bytes(out.tobytes())
    print(f"[tangent_dump] {n} loops -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
