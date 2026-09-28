"""Calibration scene packages C0a / C0b / C0p (plan §5 M2, §7.2; docs/decisions/scene-bridge.md).

Writes scene packages (scene.json + geometry.bin [+ env.exr]) directly with numpy, in the glTF canonical
frame, plus the analytic answers in scene.json "expected". No bpy: runs in any python with numpy and
OpenImageIO. Canonical generator: Blender's bundled python (the env EXR goes through Blender's OIIO):

  /Applications/Blender.app/Contents/Resources/5.1/python/bin/python3.13 validation/blender/calib_scenes.py \
      [--out validation/scenes] [--only c0a_512 c0p_512 ...]

Packages
  c0a_512, c0a_640x360, c0a_360x640   five emissive squares (TL red, TR green, BL blue, BR white, yellow at
                                      (0.3W, 0.62H)) on a plane facing the camera, black background
  c0a_far_512                         c0a_512 translated by (1000, 0, -500) m (recentring test)
  c0b_512                             camera facing a large emissive quad, L_e = strength*color = (1, 0.5, 0.25)
  c0p_512, c0p_640x360, c0p_360x640   synthetic 512x256 env (octants + asymmetric markers), 12 frames:
                                      cameras +X, -Z, +Y, -Y (glTF) x rotation gamma in {0, +30 deg, -90 deg}
  c0p_hidden_512                      c0p env with visibleToCamera false (image must be black)

"expected" (image pixels, row 0 = TOP, pixel (c, r) centre at (c + 0.5, r + 0.5)):
  {"kind": "markers" | "constant", "frames": {"<frame>": {"markers": [...], "probes": [...]}}, ...}
  marker: name, class (unit RGB direction), window [x0, x1, y0, y1] (pixel index ranges, end exclusive),
          threshold T, centroid_px (weighted centroid with w = max(dot(px, class/|class|_1) - T, 0) over the
          window, of the exact box-filtered image), center_px (projection of the marker's geometric centre),
          mass (sum of w), partial (window clipped by the image: not checked, only masked)
  probe:  pixel [c, r] whose exact value is known (fully covered / constant region), value [r, g, b]
C0a/C0b answers are exact (axis-aligned squares on a plane parallel to the image plane: pixel = L_e x
coverage). C0p answers come from an f64 re-implementation of math.md#env-mapping (+ Cycles bilinear REPEAT
lookup) supersampled 8x8 per pixel; center_px uses the math.md envDir inverse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_OUT = REPO / "validation" / "scenes"

# glTF (x, y, z) -> Blender (x, -z, y); only used for documentation/cross-checks here (build_scene applies it)
C_GLTF_TO_BLENDER = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


# --------------------------------------------------------------------------------------------------
# package writer (shared with tests)


def _mat4_colmajor(R: np.ndarray, t: np.ndarray) -> list[float]:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return [float(x) for x in M.T.reshape(-1)]  # column-major


def env_sha256(texels_bottom_up_rgba: np.ndarray) -> str:
    """scene-bridge.md env.sha256 (ENV-U9): SHA-256 of env.exr's pixels as written = float32 little-endian,
    RGBA interleaved, rows TOP-DOWN. Input is the GPU layout (rows bottom-up, row 0 = nadir)."""
    a = np.asarray(texels_bottom_up_rgba)
    assert a.ndim == 3 and a.shape[2] == 4, a.shape
    return hashlib.sha256(np.ascontiguousarray(a[::-1], dtype="<f4").tobytes()).hexdigest()


def write_env_exr(path: Path, texels_bottom_up_rgba: np.ndarray) -> None:
    """float32 ZIP RGBA EXR, rows TOP-DOWN (EXR convention), exactly the GPU texel array (scene-bridge.md)."""
    import OpenImageIO as oiio

    rgba = np.ascontiguousarray(np.asarray(texels_bottom_up_rgba, dtype=np.float32)[::-1])
    h, w, _ = rgba.shape
    spec = oiio.ImageSpec(w, h, 4, "float")
    spec.channelnames = ("R", "G", "B", "A")
    spec.alpha_channel = 3
    spec.attribute("compression", "zip")
    spec.attribute("DateTime", "")  # no capDate header: byte-identical regeneration (package hash = cache key)
    out = oiio.ImageOutput.create(str(path))
    if out is None or not out.open(str(path), spec) or not out.write_image(rgba):
        raise OSError(f"{path}: {oiio.geterror()}")
    out.close()


def write_package(out_dir: Path, scene: dict[str, Any], arrays: dict[str, tuple[np.ndarray, str, int]],
                  env_texels: np.ndarray | None = None, extra_files: dict[str, bytes] | None = None) -> Path:
    """arrays: buffer name -> (array, dtype 'f32'|'u32', components). Writes geometry.bin with 4-byte aligned
    blobs in the given order, fills scene['buffers'], writes env.exr when env_texels (H x W x 4, bottom-up)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    blob = bytearray()
    bufs: dict[str, Any] = {}
    for name, (arr, dtype, comps) in arrays.items():
        np_dt = {"f32": "<f4", "u32": "<u4"}[dtype]
        a = np.ascontiguousarray(arr, dtype=np_dt).reshape(-1)
        if a.size % comps:
            raise ValueError(f"{name}: {a.size} values not a multiple of {comps}")
        off = len(blob)
        blob += a.tobytes()
        bufs[name] = {"offset": off, "length": a.nbytes, "dtype": dtype, "components": comps}
    (out_dir / "geometry.bin").write_bytes(bytes(blob))
    scene = {**scene, "buffers": bufs}
    if env_texels is not None:
        write_env_exr(out_dir / scene["env"]["file"], env_texels)
    for fn, data in (extra_files or {}).items():
        (out_dir / fn).write_bytes(data)
    (out_dir / "scene.json").write_text(json.dumps(scene, indent=1) + "\n")
    return out_dir


def base_scene(name: str, W: int, H: int, max_bounces: int, cam_matrix: list[float], yfov: float) -> dict[str, Any]:
    return {
        "format": "restir-scene-package", "version": 1, "name": name,
        "source": {"uri": "validation/blender/calib_scenes.py", "sha256": None},
        "flatShaded": True,
        "materials": [], "textures": [], "lights": [], "lightMode": "A",
        "camera": {"matrix": cam_matrix, "yfov": yfov, "znear": 1e-4},
        "env": None,
        "render": {"width": W, "height": H, "maxBounces": max_bounces},
    }


def emissive_v1(name: str, color: tuple[float, float, float], strength: float) -> dict[str, Any]:
    return {"name": name, "model": "v1",
            "v1": {"diffuse": [0.0, 0.0, 0.0], "glossy": [0.0, 0.0, 0.0], "roughness": 0.5, "mix": 0.0},
            "emission": {"color": list(color), "strength": strength}, "emissionSampling": "FRONT_BACK"}


class QuadMesh:
    """Unwelded flat quads (2 triangles each) -> package buffers."""

    def __init__(self) -> None:
        self.pos: list[np.ndarray] = []
        self.tri_mat: list[int] = []

    def add(self, corners: np.ndarray, mat: int) -> None:
        """corners: 4x3 in order around the quad (glTF world)."""
        self.pos.append(np.asarray(corners, dtype=np.float64))
        self.tri_mat += [mat, mat]

    def arrays(self) -> dict[str, tuple[np.ndarray, str, int]]:
        n = len(self.pos)
        P = np.concatenate(self.pos) if n else np.zeros((0, 3))
        N = []
        for q in self.pos:
            nrm = np.cross(q[1] - q[0], q[2] - q[0])
            N += [nrm / np.linalg.norm(nrm)] * 4
        N = np.array(N) if n else np.zeros((0, 3))
        UV = np.tile(np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64), (n, 1))  # glTF uv (v down)
        idx = np.concatenate([np.array([[0, 1, 2], [0, 2, 3]]) + 4 * i for i in range(n)]) if n else np.zeros((0, 3))
        return {
            "positions": (P, "f32", 3), "normals": (N, "f32", 3), "uv0": (UV, "f32", 2),
            "indices": (idx, "u32", 3), "triMaterial": (np.array(self.tri_mat), "u32", 1),
        }


# --------------------------------------------------------------------------------------------------
# camera model (math.md §3): image coords x right, y DOWN from the top edge, pixel (c, r) = [c, c+1] x [r, r+1]


def look_rotation(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera-to-world rotation (columns X, Y, Z of the camera) looking along `forward` (camera -Z)."""
    z = -np.asarray(forward, float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def project(points: np.ndarray, R: np.ndarray, t: np.ndarray, yfov: float, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """World points (N,3) -> image (x, y) (row 0 top) and camera-space depth (-z_cam, > 0 in front)."""
    pc = (np.asarray(points, float) - t) @ R  # R^T (p - t)
    ty = math.tan(yfov / 2)
    tx = ty * W / H
    depth = -pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (pc[:, 0] / depth / tx + 1) * W / 2
        y = (1 - pc[:, 1] / depth / ty) * H / 2
    return np.stack([x, y], axis=1), depth


def pixel_dirs(xs: np.ndarray, ys: np.ndarray, R: np.ndarray, yfov: float, W: int, H: int) -> np.ndarray:
    """Image coords -> unit world directions (glTF)."""
    ty = math.tan(yfov / 2)
    tx = ty * W / H
    d = np.stack([(2 * xs / W - 1) * tx, (1 - 2 * ys / H) * ty, -np.ones_like(xs)], axis=-1)
    d = d @ R.T
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


# --------------------------------------------------------------------------------------------------
# C0a / C0b

C0A_MARKERS = [  # name, image-fraction centre (+ sub-pixel offset in px at the variant), colour, strength
    ("red_tl", (0.16, 0.14), (0.23, 0.61), (1.0, 0.0, 0.0), 3.0),
    ("green_tr", (0.83, 0.17), (0.41, 0.12), (0.0, 1.0, 0.0), 2.0),
    ("blue_bl", (0.18, 0.85), (0.77, 0.34), (0.0, 0.0, 1.0), 4.0),
    ("white_br", (0.86, 0.82), (0.05, 0.58), (1.0, 1.0, 1.0), 1.5),
    ("yellow", (0.30, 0.62), (0.0, 0.0), (1.0, 1.0, 0.0), 2.5),
]
C0A_SIDE_PX_AT_512 = 12.3  # square side in pixels at H = 512 (scaled by H/512 for other variants)
C0A_VFOV = math.radians(40.0)
C0A_DEPTH = 10.0


def _coverage_1d(a: float, b: float, lo: int, hi: int) -> np.ndarray:
    c = np.arange(lo, hi, dtype=np.float64)
    return np.clip(np.minimum(c + 1, b) - np.maximum(c, a), 0, None)


def _square_expected(name: str, cls: tuple[float, ...], Le: np.ndarray, x0: float, x1: float, y0: float, y1: float,
                     W: int, H: int) -> dict[str, Any]:
    """Exact box-filtered image of an axis-aligned radiance square [x0,x1]x[y0,y1] (image px)."""
    cls_v = np.asarray(cls, float)
    wdot = float(Le @ (cls_v / cls_v.sum()))
    wx0, wx1 = int(math.floor(x0)) - 2, int(math.ceil(x1)) + 2
    wy0, wy1 = int(math.floor(y0)) - 2, int(math.ceil(y1)) + 2
    ox, oy = _coverage_1d(x0, x1, wx0, wx1), _coverage_1d(y0, y1, wy0, wy1)
    cx = float(((np.arange(wx0, wx1) + 0.5) * ox).sum() / ox.sum())
    cy = float(((np.arange(wy0, wy1) + 0.5) * oy).sum() / oy.sum())
    inside = [int(math.ceil(x0)), int(math.floor(x1)) - 1, int(math.ceil(y0)), int(math.floor(y1)) - 1]
    probes = []
    if inside[1] >= inside[0] and inside[3] >= inside[2]:  # fully covered pixels: value exactly L_e
        for c, r in ((inside[0], inside[2]), (inside[1], inside[3]), ((inside[0] + inside[1]) // 2, (inside[2] + inside[3]) // 2)):
            probes.append({"pixel": [c, r], "value": [float(v) for v in Le], "what": f"{name} interior"})
    return {
        "marker": {
            "name": name, "class": list(cls), "threshold": 0.0,
            "window": [wx0, wx1, wy0, wy1], "partial": not (wx0 >= 0 and wy0 >= 0 and wx1 <= W and wy1 <= H),
            "centroid_px": [cx, cy], "center_px": [(x0 + x1) / 2, (y0 + y1) / 2], "bounds_px": [x0, x1, y0, y1],
            "mass": wdot * (x1 - x0) * (y1 - y0), "L_e": [float(v) for v in Le],
        },
        "probes": probes,
    }


def make_c0a(name: str, W: int, H: int, offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[dict, dict]:
    off = np.asarray(offset, float)
    R = np.eye(3)
    ty = math.tan(C0A_VFOV / 2)
    tx = ty * W / H
    D = C0A_DEPTH
    side = C0A_SIDE_PX_AT_512 * H / 512

    def img_to_world(x: float, y: float) -> np.ndarray:
        return np.array([(2 * x / W - 1) * tx * D, (1 - 2 * y / H) * ty * D, -D]) + off

    mesh = QuadMesh()
    mats = []
    for i, (mname, (fx, fy), (dx, dy), col, s) in enumerate(C0A_MARKERS):
        cx, cy = fx * W + dx, fy * H + dy
        x0, x1, y0, y1 = cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2
        # counter-clockwise seen from the camera (normal toward +Z_g, i.e. toward the camera)
        mesh.add(np.array([img_to_world(x0, y1), img_to_world(x1, y1), img_to_world(x1, y0), img_to_world(x0, y0)]), i)
        mats.append(emissive_v1(mname, col, s))
    arrays = mesh.arrays()
    cam = _mat4_colmajor(R, off)
    scene = base_scene(name, W, H, 0, cam, C0A_VFOV)
    scene["materials"] = mats

    # expected from the float32-stored geometry, projected in f64 with the stored camera
    P32 = arrays["positions"][0].astype(np.float32).astype(np.float64)
    t32 = np.asarray(cam[12:15], np.float32).astype(np.float64)
    markers, probes = [], []
    for i, (mname, _, _, col, s) in enumerate(C0A_MARKERS):
        xy, depth = project(P32[4 * i:4 * i + 4], R, t32, C0A_VFOV, W, H)
        assert (depth > 0).all()
        x0, x1 = float(xy[:, 0].min()), float(xy[:, 0].max())
        y0, y1 = float(xy[:, 1].min()), float(xy[:, 1].max())
        e = _square_expected(mname, col, np.asarray(col) * s, x0, x1, y0, y1, W, H)
        markers.append(e["marker"])
        probes += e["probes"]
    probes += [{"pixel": [2, 2], "value": [0.0, 0.0, 0.0], "what": "background"},
               {"pixel": [W // 2, H // 3], "value": [0.0, 0.0, 0.0], "what": "background"}]
    scene["expected"] = {
        "kind": "markers", "tolerance_px": 0.1, "probe_tolerance": 1e-4, "background": [0.0, 0.0, 0.0],
        "note": "exact box-filtered coverage of the float32-stored squares; row 0 = top; pixel centre = +0.5",
        "frames": {"0": {"markers": markers, "probes": probes}},
    }
    return scene, arrays


def make_c0b(name: str, W: int, H: int) -> tuple[dict, dict]:
    color, strength = (0.5, 0.25, 0.125), 2.0  # L_e = (1, 0.5, 0.25)
    mesh = QuadMesh()
    D, S = 2.0, 50.0
    mesh.add(np.array([[-S, -S, -D], [S, -S, -D], [S, S, -D], [-S, S, -D]]), 0)
    scene = base_scene(name, W, H, 0, _mat4_colmajor(np.eye(3), np.zeros(3)), math.radians(40.0))
    scene["materials"] = [emissive_v1("emitter", color, strength)]
    Le = [c * strength for c in color]
    scene["expected"] = {"kind": "constant", "value": Le, "tolerance": 1e-4,
                         "note": "every pixel is fully covered by the emitter: pixel = strength*color exactly",
                         "frames": {"0": {}}}
    return scene, mesh.arrays()


# --------------------------------------------------------------------------------------------------
# C0p env orientation calibration (math.md#env-mapping)

ENV_W, ENV_H = 512, 256
OCTANT_UPPER = [(0.80, 0.20, 0.20), (0.20, 0.80, 0.20), (0.20, 0.20, 0.80), (0.80, 0.80, 0.20)]
OCTANT_LOWER = [(0.20, 0.80, 0.80), (0.80, 0.20, 0.80), (0.50, 0.50, 0.50), (0.10, 0.10, 0.10)]
MARKER_VALUE = 100.0
MARKER_THRESHOLD = 2.0  # > every octant value, << marker value
# name: (class colour, list of (col0, col1, row0, row1) texel blocks, inclusive, rows BOTTOM-UP)
ENV_MARKERS: dict[str, tuple[tuple[float, float, float], list[tuple[int, int, int, int]]]] = {
    # "L" glyph at +X_g (u = 0.5, v = 0.5): vertical stroke + foot to the right at the bottom
    "L_plusX": ((1.0, 1.0, 1.0), [(254, 255, 126, 135), (256, 259, 126, 127)]),
    "dot_minusZ": ((1.0, 0.0, 0.0), [(127, 128, 127, 128)]),         # u = 0.25
    "dot_plusZ_hi": ((0.0, 1.0, 0.0), [(383, 384, 133, 134)]),       # u = 0.75, above the equator
    "dot_plusZ_lo": ((0.0, 1.0, 0.0), [(383, 384, 121, 122)]),       # u = 0.75, below the equator
    "seam_bar": ((0.0, 0.0, 1.0), [(506, 511, 127, 128), (0, 3, 127, 128)]),  # crosses u = 0/1, 6 + 4 texels
    "pole_hi": ((1.0, 1.0, 0.0), [(319, 320, 242, 243)]),            # v ~ 0.95, u = 0.625
    "pole_lo": ((1.0, 0.0, 1.0), [(63, 64, 7, 8)]),                  # v ~ 0.03, u = 0.125
}


def c0p_texels() -> np.ndarray:
    """H x W x 4 float32, rows BOTTOM-UP (row j: v in [j/H, (j+1)/H]), alpha 1."""
    T = np.zeros((ENV_H, ENV_W, 4), np.float32)
    T[..., 3] = 1
    for q in range(4):
        T[ENV_H // 2:, q * 128:(q + 1) * 128, :3] = OCTANT_UPPER[q]
        T[:ENV_H // 2, q * 128:(q + 1) * 128, :3] = OCTANT_LOWER[q]
    for cls, blocks in ENV_MARKERS.values():
        for c0, c1, r0, r1 in blocks:
            T[r0:r1 + 1, c0:c1 + 1, :3] = np.asarray(cls, np.float32) * MARKER_VALUE
    return T


# Planted errors (plan §5 M2 exit: C0p must detect them): the same mapping with one deliberate bug.
PLANTS = ("u_half_texel", "v_flip", "gamma_sign", "no_C")


def env_uv(d: np.ndarray, gamma: float, plant: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """math.md#env-mapping: glTF direction -> (u, v). `plant` injects one of PLANTS."""
    if plant == "gamma_sign":
        gamma = -gamma
    cg, sg = math.cos(gamma), math.sin(gamma)
    if plant == "no_C":  # the glTF direction used as if it were Z-up (C omitted)
        bx, by, bz = cg * d[..., 0] - sg * d[..., 1], sg * d[..., 0] + cg * d[..., 1], d[..., 2]
    else:
        bx = cg * d[..., 0] + sg * d[..., 2]
        by = sg * d[..., 0] - cg * d[..., 2]
        bz = d[..., 1]
    u = (np.arctan2(by, bx) - math.pi) / (-2 * math.pi)
    v = (np.arccos(np.clip(bz, -1, 1)) - math.pi) / (-math.pi)
    if plant == "u_half_texel":
        u = u + 0.5 / ENV_W
    if plant == "v_flip":
        v = 1 - v
    return u, v


def env_dir(u: np.ndarray, v: np.ndarray, gamma: float, plant: str | None = None) -> np.ndarray:
    """math.md#env-mapping inverse envDir(u, v) (inverse of env_uv with the same plant)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    if plant == "u_half_texel":
        u = u - 0.5 / ENV_W
    if plant == "v_flip":
        v = 1 - v
    if plant == "gamma_sign":
        gamma = -gamma
    phi = -2 * math.pi * u + math.pi
    theta = -math.pi * v + math.pi
    b = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    cg, sg = math.cos(gamma), math.sin(gamma)
    r = np.stack([cg * b[..., 0] + sg * b[..., 1], -sg * b[..., 0] + cg * b[..., 1], b[..., 2]], axis=-1)
    if plant == "no_C":
        return r
    return np.stack([r[..., 0], r[..., 2], -r[..., 1]], axis=-1)


def env_lookup(T: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cycles bilinear lookup with EXTENSION_REPEAT on both axes (rows bottom-up), f64."""
    Hh, Ww = T.shape[:2]
    x = u * Ww - 0.5
    y = v * Hh - 0.5
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = (x - ix)[..., None], (y - iy)[..., None]
    ix0, iy0 = ix.astype(np.int64) % Ww, iy.astype(np.int64) % Hh
    ix1, iy1 = (ix0 + 1) % Ww, (iy0 + 1) % Hh
    T = T[..., :3].astype(np.float64)
    return ((1 - fy) * ((1 - fx) * T[iy0, ix0] + fx * T[iy0, ix1]) + fy * ((1 - fx) * T[iy1, ix0] + fx * T[iy1, ix1]))


def render_env_window(T: np.ndarray, gamma: float, R: np.ndarray, yfov: float, W: int, H: int,
                      win: tuple[int, int, int, int], ss: int = 8, plant: str | None = None) -> np.ndarray:
    """Box-filtered (ss x ss midpoint) f64 render of the env over window pixels [x0,x1) x [y0,y1)."""
    x0, x1, y0, y1 = win
    o = (np.arange(ss) + 0.5) / ss
    xs = (np.arange(x0, x1)[:, None] + o[None, :]).reshape(-1)
    ys = (np.arange(y0, y1)[:, None] + o[None, :]).reshape(-1)
    X, Y = np.meshgrid(xs, ys)  # (ny*ss, nx*ss)
    d = pixel_dirs(X, Y, R, yfov, W, H)
    u, v = env_uv(d, gamma, plant)
    L = env_lookup(T, u, v)
    ny, nx = y1 - y0, x1 - x0
    return L.reshape(ny, ss, nx, ss, 3).mean(axis=(1, 3))


def _marker_uv_footprint(blocks: list[tuple[int, int, int, int]]) -> list[tuple[float, float, float, float]]:
    """Bilinear support of each block in (u, v): texel i influences x = uW - 0.5 in (i - 1, i + 1)."""
    out = []
    for c0, c1, r0, r1 in blocks:
        out.append(((c0 - 0.5) / ENV_W, (c1 + 1.5) / ENV_W, (r0 - 0.5) / ENV_H, (r1 + 1.5) / ENV_H))
    return out


def _marker_center_uv(blocks: list[tuple[int, int, int, int]]) -> tuple[float, float]:
    """Area-weighted centre of the texel blocks in (u, v) (handles the seam by unwrapping)."""
    us, vs, ws = [], [], []
    ref = (blocks[0][0] + blocks[0][1] + 1) / 2 / ENV_W
    for c0, c1, r0, r1 in blocks:
        area = (c1 - c0 + 1) * (r1 - r0 + 1)
        u = (c0 + c1 + 1) / 2 / ENV_W
        u += round(ref - u)  # unwrap near the reference block
        us.append(u), vs.append((r0 + r1 + 1) / 2 / ENV_H), ws.append(area)
    return float(np.average(us, weights=ws)), float(np.average(vs, weights=ws))


def c0p_frame_expected(T: np.ndarray, gamma: float, R: np.ndarray, yfov: float, W: int, H: int,
                       plant: str | None = None) -> dict[str, Any]:
    t0 = np.zeros(3)
    markers = []
    for name, (cls, blocks) in ENV_MARKERS.items():
        pts = []
        for ua, ub, va, vb in _marker_uv_footprint(blocks):
            s = np.linspace(0, 1, 65)
            edge = np.concatenate([np.stack([ua + (ub - ua) * s, np.full_like(s, va)], 1),
                                   np.stack([ua + (ub - ua) * s, np.full_like(s, vb)], 1),
                                   np.stack([np.full_like(s, ua), va + (vb - va) * s], 1),
                                   np.stack([np.full_like(s, ub), va + (vb - va) * s], 1)])
            pts.append(env_dir(edge[:, 0], edge[:, 1], gamma, plant))
        dirs = np.concatenate(pts)
        xy, depth = project(dirs, R, t0, yfov, W, H)
        if not (depth > 1e-3).all():
            continue
        wx0, wx1 = int(math.floor(xy[:, 0].min())) - 1, int(math.ceil(xy[:, 0].max())) + 1
        wy0, wy1 = int(math.floor(xy[:, 1].min())) - 1, int(math.ceil(xy[:, 1].max())) + 1
        if wx1 <= 0 or wy1 <= 0 or wx0 >= W or wy0 >= H:
            continue
        partial = not (wx0 >= 0 and wy0 >= 0 and wx1 <= W and wy1 <= H)
        uc, vc = _marker_center_uv(blocks)
        cxy, _ = project(env_dir(np.array([uc]), np.array([vc]), gamma, plant), R, t0, yfov, W, H)
        m: dict[str, Any] = {"name": name, "class": list(cls), "threshold": MARKER_THRESHOLD,
                             "window": [max(wx0, 0), min(wx1, W), max(wy0, 0), min(wy1, H)], "partial": partial,
                             "center_uv": [uc, vc], "center_px": [float(cxy[0, 0]), float(cxy[0, 1])]}
        if not partial:
            img = render_env_window(T, gamma, R, yfov, W, H, (wx0, wx1, wy0, wy1), plant=plant)
            c = np.asarray(cls, float)
            w = np.clip(img @ (c / c.sum()) - MARKER_THRESHOLD, 0, None)
            if w.sum() <= 0:
                continue
            ys, xs = np.mgrid[wy0:wy1, wx0:wx1] + 0.5
            m["centroid_px"] = [float((w * xs).sum() / w.sum()), float((w * ys).sum() / w.sum())]
            m["mass"] = float(w.sum())
        markers.append(m)
    # probes: pixels whose (dilated) footprint samples all hit the same 4 texels of one constant colour
    probes = []
    grid = [(int((i + 0.5) * W / 7), int((j + 0.5) * H / 7)) for j in range(7) for i in range(7)]
    for c, r in grid:
        o = np.linspace(-1.0, 2.0, 13)  # pixel dilated by 1 px on each side
        X, Y = np.meshgrid(c + o, r + o)
        d = pixel_dirs(X, Y, R, yfov, W, H)
        u, v = env_uv(d, gamma, plant)
        x, y = u * ENV_W - 0.5, v * ENV_H - 0.5
        ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
        cols = np.concatenate([T[iy % ENV_H, ix % ENV_W, :3].reshape(-1, 3), T[iy % ENV_H, (ix + 1) % ENV_W, :3].reshape(-1, 3),
                               T[(iy + 1) % ENV_H, ix % ENV_W, :3].reshape(-1, 3), T[(iy + 1) % ENV_H, (ix + 1) % ENV_W, :3].reshape(-1, 3)])
        if np.all(cols == cols[0]):
            probes.append({"pixel": [c, r], "value": [float(x) for x in cols[0]], "what": "constant octant"})
    return {"markers": markers, "probes": probes}


C0P_CAMERAS = {  # name: forward, up (glTF)
    "plusX": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "minusZ": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "plusY": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
    "minusY": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
}
C0P_GAMMAS_DEG = (0.0, 30.0, -90.0)
C0P_VFOV_DEG = {(512, 512): 100.0, (640, 360): 100.0, (360, 640): 120.0}
C0P_CAM_POS = (3.0, 1.0, -2.0)  # arbitrary: env lookups ignore the camera position


def make_c0p(name: str, W: int, H: int, visible: bool = True, frames: list[tuple[str, float]] | None = None) -> tuple[dict, dict, np.ndarray]:
    T = c0p_texels()
    yfov = math.radians(C0P_VFOV_DEG[(W, H)])
    combos = frames or [(cam, g) for cam in C0P_CAMERAS for g in C0P_GAMMAS_DEG]
    t = np.asarray(C0P_CAM_POS)
    frame_list, expected = [], {}
    for k, (cam, gdeg) in enumerate(combos):
        f, up = C0P_CAMERAS[cam]
        R = look_rotation(np.asarray(f), np.asarray(up))
        # snap the rotation to what float32 JSON consumers see (values are exactly 0/±1 here anyway)
        M = _mat4_colmajor(R, t)
        g = math.radians(gdeg)
        frame_list.append({"frame": k, "label": f"{cam} gamma={gdeg:+.0f}deg", "camera": {"matrix": M, "yfov": yfov},
                           "env": {"rotationZ": g, "strength": 1.0}})
        if visible:
            e = c0p_frame_expected(T, g, R, yfov, W, H)
            if not any("centroid_px" in m for m in e["markers"]):
                raise AssertionError(f"{name} frame {k} ({cam}, {gdeg}): no fully visible marker")
            expected[str(k)] = e
        else:
            expected[str(k)] = {"markers": [], "probes": [{"pixel": [W // 2, H // 2], "value": [0.0, 0.0, 0.0], "what": "hidden env"}]}
    scene = base_scene(name, W, H, 0, frame_list[0]["camera"]["matrix"], yfov)
    scene["env"] = {"file": "env.exr", "strength": 1.0, "tint": [1.0, 1.0, 1.0], "rotationZ": frame_list[0]["env"]["rotationZ"],
                    "visibleToCamera": visible, "sampling": "AUTOMATIC",
                    "width": ENV_W, "height": ENV_H, "sha256": env_sha256(T)}
    scene["frames"] = frame_list
    if visible:
        scene["expected"] = {
            "kind": "markers", "tolerance_px": 0.1, "report_tolerance_px": 0.5, "probe_tolerance": 1e-4,
            "note": "centroid_px: f64 math.md#env-mapping + bilinear REPEAT lookup, 8x8 box supersampling per pixel, "
                    "w = max(dot(px, class/|class|_1) - threshold, 0); center_px: envDir inverse of the texel-block centre",
            "frames": expected,
        }
    else:
        scene["expected"] = {"kind": "constant", "value": [0.0, 0.0, 0.0], "tolerance": 0.0, "frames": expected}
    return scene, QuadMesh().arrays(), T


# --------------------------------------------------------------------------------------------------


def build_all(out: Path, only: list[str] | None = None) -> dict[str, str]:
    jobs: dict[str, Any] = {
        "c0a_512": lambda n: make_c0a(n, 512, 512),
        "c0a_640x360": lambda n: make_c0a(n, 640, 360),
        "c0a_360x640": lambda n: make_c0a(n, 360, 640),
        "c0a_far_512": lambda n: make_c0a(n, 512, 512, offset=(1000.0, 0.0, -500.0)),
        "c0b_512": lambda n: make_c0b(n, 512, 512),
        "c0p_512": lambda n: make_c0p(n, 512, 512),
        "c0p_640x360": lambda n: make_c0p(n, 640, 360),
        "c0p_360x640": lambda n: make_c0p(n, 360, 640),
        "c0p_hidden_512": lambda n: make_c0p(n, 512, 512, visible=False, frames=[("plusX", 0.0)]),
    }
    done = {}
    for name, fn in jobs.items():
        if only and name not in only:
            continue
        res = fn(name)
        scene, arrays = res[0], res[1]
        env = res[2] if len(res) > 2 else None
        write_package(out / name, scene, arrays, env)
        fr = scene["expected"]["frames"]
        nm = sum(1 for f in fr.values() for m in f.get("markers", []) if "centroid_px" in m)
        npr = sum(len(f.get("probes", [])) for f in fr.values())
        print(f"[calib] {name}: {len(fr)} frame(s), {nm} checked markers, {npr} probes -> {out / name}", flush=True)
        done[name] = str(out / name)
    return done


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args(argv)
    build_all(args.out, args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
