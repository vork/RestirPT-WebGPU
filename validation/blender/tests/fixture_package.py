"""Synthetic scene package exercising the whole bridge contract (no bpy; numpy + OpenImageIO).

make_bridge_fixture(dir) writes a package with:
- 3 emissive-textured Principled quads (nearest filter, sRGB 4x4 PNG with coloured 2x2 quadrants): no
  transform / KHR_texture_transform offset (0.5, 0) / rotation 90 deg + offset (0, 1), mixed wraps
- 1 smooth-shaded quad with constant tilted vertex normals and COLOR_0 (V1 glossy material)
- a second PNG used by metallicRoughness / normal / specular textures (structural checks only)
- 5 lights (point, spot, rect, disk, sun) behind the camera, lightMode B, rect visible to the camera
- a 64x32 env (hidden from the camera), 2 frames (light power/matrix, env rotation/strength, camera yfov)
It returns the facts the Blender-side checks compare against.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calib_scenes as cal  # noqa: E402

TEX_QUADRANTS = {  # PNG pixel rows TOP-first: (row block, col block) -> RGB8
    (0, 0): (200, 30, 30),    # top-left     red
    (0, 1): (30, 200, 30),    # top-right    green
    (1, 0): (30, 30, 200),    # bottom-left  blue
    (1, 1): (220, 220, 220),  # bottom-right white
}
EMISSIVE_FACTOR = (1.0, 0.5, 1.0)
EMISSIVE_STRENGTH = 2.0
N0_GLTF = np.array([0.3, 0.4, 0.866])
N0_GLTF = N0_GLTF / np.linalg.norm(N0_GLTF)
TRANSFORMS = [
    None,
    [1.0, 0.0, 0.5, 0.0, 1.0, 0.0],  # offset (0.5, 0)
    # rotation 90 deg, scale 1, offset (0, 1): [c·sx, s·sy, ox; -s·sx, c·sy, oy]
    [math.cos(math.pi / 2), math.sin(math.pi / 2), 0.0, -math.sin(math.pi / 2), math.cos(math.pi / 2), 1.0],
]
WRAPS = [("repeat", "repeat"), ("repeat", "repeat"), ("clamp-to-edge", "mirror-repeat")]


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def tex_pixels() -> np.ndarray:
    px = np.zeros((4, 4, 4), np.uint8)
    px[..., 3] = 255
    for (rb, cb), rgb in TEX_QUADRANTS.items():
        px[2 * rb:2 * rb + 2, 2 * cb:2 * cb + 2, :3] = rgb
    return px


def write_png(path: Path, px: np.ndarray) -> None:
    import OpenImageIO as oiio

    h, w, c = px.shape
    out = oiio.ImageOutput.create(str(path))
    spec = oiio.ImageSpec(w, h, c, "uint8")
    if out is None or not out.open(str(path), spec) or not out.write_image(np.ascontiguousarray(px)):
        raise OSError(f"{path}: {oiio.geterror()}")
    out.close()


def _wrap(t: float, mode: str) -> float:
    if mode == "repeat":
        return t - math.floor(t)
    if mode == "clamp-to-edge":
        return min(max(t, 0.0), 1.0 - 1e-9)
    if mode == "mirror-repeat":
        f = t - 2 * math.floor(t / 2)
        return f if f <= 1 else 2 - f
    raise ValueError(mode)


def expected_emission(u: float, v: float, tf: list[float] | None, wraps: tuple[str, str]) -> list[float]:
    """Emission radiance at glTF uv (u, v) on a textured quad (nearest texel of the 4x4 PNG)."""
    if tf is not None:
        a, b, c, d, e, f = tf
        u, v = a * u + b * v + c, d * u + e * v + f
    u, v = _wrap(u, wraps[0]), _wrap(v, wraps[1])
    px = tex_pixels()
    i, j = min(int(v * 4), 3), min(int(u * 4), 3)
    lin = srgb_to_linear(px[i, j, :3] / 255.0)
    return [float(x) for x in lin * np.asarray(EMISSIVE_FACTOR) * EMISSIVE_STRENGTH]


def _mat(R: np.ndarray, t: list[float]) -> list[float]:
    return cal._mat4_colmajor(R, np.asarray(t, float))


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def make_bridge_fixture(out_dir: Path, *, W: int = 512, H: int = 512) -> dict[str, Any]:
    D, S = 3.0, 0.6
    xs = [-0.7, 0.0, 0.7]
    P, N, UV, C0, idx, tm = [], [], [], [], [], []
    materials: list[dict[str, Any]] = []
    textures = [
        {"file": "tex_0.png", "wrapS": "repeat", "wrapT": "repeat", "filter": "nearest"},
        {"file": "tex_0.png", "wrapS": "repeat", "wrapT": "repeat", "filter": "nearest"},
        {"file": "tex_0.png", "wrapS": "clamp-to-edge", "wrapT": "mirror-repeat", "filter": "nearest"},
        {"file": "tex_1.png", "wrapS": "repeat", "wrapT": "repeat", "filter": "linear"},
    ]

    def quad(x0: float, x1: float, y0: float, y1: float, z: float, mat: int, normal: np.ndarray | None = None,
             color: list[float] | None = None) -> None:
        base = len(P)
        # corners TL, TR, BR, BL as seen from the camera (+Y up); glTF uv origin top-left
        for (x, y), uv in zip(((x0, y1), (x1, y1), (x1, y0), (x0, y0)), ((0, 0), (1, 0), (1, 1), (0, 1))):
            P.append([x, y, z])
            N.append(list(normal) if normal is not None else [0.0, 0.0, 1.0])
            UV.append(list(uv))
            C0.append(color or [1.0, 1.0, 1.0, 1.0])
        idx.extend([[base, base + 3, base + 2], [base, base + 2, base + 1]])  # CCW seen from +Z (camera side)
        tm.extend([mat, mat])

    for k, x in enumerate(xs):
        ref: dict[str, Any] = {"texture": k, "texCoord": 0}
        if TRANSFORMS[k] is not None:
            ref["transform"] = TRANSFORMS[k]
        materials.append({
            "name": f"tex{k}", "model": "principled",
            "baseColorFactor": [0.0, 0.0, 0.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0, "ior": 1.5,
            "specularFactor": 0.0, "specularColorFactor": [1.0, 1.0, 1.0], "transmissionFactor": 0.0,
            "emissiveFactor": list(EMISSIVE_FACTOR), "emissiveStrength": EMISSIVE_STRENGTH, "emissiveTexture": ref,
            "alphaMode": "OPAQUE", "alphaCutoff": 0.5, "doubleSided": True,
            "emission": {"color": list(EMISSIVE_FACTOR), "strength": EMISSIVE_STRENGTH}, "emissionSampling": "FRONT_BACK",
        })
        quad(x - S / 2, x + S / 2, 0.1, 0.1 + S, -D, k)
    # a 'kitchen sink' principled material (structural checks only), not on screen
    materials.append({
        "name": "sink", "model": "principled", "baseColorFactor": [0.8, 0.7, 0.6, 0.9], "metallicFactor": 0.5,
        "roughnessFactor": 0.4, "ior": 1.45, "specularFactor": 0.8, "specularColorFactor": [0.9, 1.0, 0.8],
        "transmissionFactor": 0.25, "emissiveFactor": [0.0, 0.0, 0.0], "emissiveStrength": 1.0,
        "baseColorTexture": {"texture": 0, "texCoord": 0}, "metallicRoughnessTexture": {"texture": 3, "texCoord": 0},
        "normalTexture": {"texture": 3, "texCoord": 0, "scale": 0.7}, "specularTexture": {"texture": 3, "texCoord": 0},
        "specularColorTexture": {"texture": 3, "texCoord": 0}, "transmissionTexture": {"texture": 3, "texCoord": 0},
        "alphaMode": "MASK", "alphaCutoff": 0.4, "doubleSided": True, "emissionSampling": "NONE",
    })
    quad(-5.0, -4.0, 0.0, 1.0, 5.0, 3)  # behind the camera
    # smooth quad with constant tilted normals + COLOR_0, V1 glossy
    materials.append({"name": "v1gloss", "model": "v1",
                      "v1": {"diffuse": [0.5, 0.4, 0.3], "glossy": [0.9, 0.9, 0.9], "roughness": 0.3, "mix": 0.25},
                      "emission": {"color": [0.0, 0.0, 0.0], "strength": 0.0}, "emissionSampling": "FRONT_BACK"})
    quad(-0.4, 0.4, -0.9, -0.1, -D, 4, normal=N0_GLTF, color=[0.2, 0.4, 0.6, 1.0])

    arrays = {
        "positions": (np.array(P), "f32", 3), "normals": (np.array(N), "f32", 3), "uv0": (np.array(UV), "f32", 2),
        "color0": (np.array(C0), "f32", 4), "indices": (np.array(idx), "u32", 3), "triMaterial": (np.array(tm), "u32", 1),
    }
    behind = [0.0, 0.5, 4.0]
    lights = [
        {"id": 0, "name": "pt", "type": "point", "color": [1.0, 0.9, 0.8], "power": 50.0, "exposure": 0.5,
         "matrix": _mat(np.eye(3), [0.5, 1.0, 3.0]), "visibleToCamera": False},
        {"id": 3, "name": "spot", "type": "spot", "color": [1.0, 1.0, 1.0], "power": 30.0, "exposure": 0.0,
         "matrix": _mat(rot_x(-0.3), [-0.5, 1.2, 3.5]), "spotSize": math.radians(40), "spotBlend": 0.15, "visibleToCamera": False},
        {"id": 7, "name": "rect", "type": "rect", "color": [0.8, 0.8, 1.0], "power": 20.0, "exposure": 0.0,
         "matrix": _mat(rot_x(0.4), behind), "sizeX": 0.5, "sizeY": 0.25, "spread": math.radians(60), "visibleToCamera": True},
        {"id": 8, "name": "disk", "type": "disk", "color": [1.0, 1.0, 1.0], "power": 10.0, "exposure": -1.0,
         "matrix": _mat(rot_x(-0.2), [1.0, 0.2, 4.5]), "sizeX": 0.3, "spread": math.pi, "visibleToCamera": False},
        {"id": 9, "name": "sun", "type": "sun", "color": [1.0, 0.95, 0.9], "power": 3.0, "exposure": 0.0,
         "matrix": _mat(rot_x(-1.0), [0.0, 10.0, 0.0]), "visibleToCamera": False},
    ]
    cam = _mat(np.eye(3), [0.0, 0.3, 0.0])
    yfov = math.radians(40.0)
    env = np.random.default_rng(7).uniform(0.0, 2.0, (32, 64, 4)).astype(np.float32)
    env[..., 3] = 1.0
    scene = cal.base_scene("bridge_fixture", W, H, 0, cam, yfov)
    scene["version"] = 1  # bridge fixture: unquantized smooth geometry (package v1 layout, still accepted by build_scene.py)
    scene.update({
        "flatShaded": False, "materials": materials, "textures": textures, "lights": lights, "lightMode": "B",
        "env": {"file": "env.exr", "strength": 0.5, "tint": [1.0, 0.9, 0.8], "rotationZ": 0.25, "visibleToCamera": False,
                "sampling": "AUTOMATIC", "sha256": cal.env_sha256(env)},
        "frames": [
            {"frame": 0, "camera": {"matrix": cam, "yfov": yfov}},
            {"frame": 5, "camera": {"matrix": _mat(np.eye(3), [0.0, 0.3, 0.1]), "yfov": math.radians(42.0)},
             "lights": {"0": {"matrix": _mat(np.eye(3), [0.6, 1.0, 3.0]), "power": 60.0}},
             "env": {"rotationZ": -0.5, "strength": 0.75}},
        ],
    })
    cal.write_package(out_dir, scene, arrays, env)
    write_png(out_dir / "tex_0.png", tex_pixels())
    t1 = np.random.default_rng(3).integers(0, 256, (8, 8, 4), dtype=np.uint8)
    write_png(out_dir / "tex_1.png", t1)

    # facts for the Blender checks: textured quad quadrant centres (glTF uv) -> image pixel + expected radiance
    probes = []
    ty = math.tan(yfov / 2)
    for k, x in enumerate(xs):
        for (u, v) in ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
            wx = x - S / 2 + u * S
            wy = 0.1 + S - v * S
            p = np.array([wx, wy - 0.3, -D])  # camera at y = 0.3
            px = (p[0] / D / (ty * W / H) + 1) * W / 2
            py = (1 - p[1] / D / ty) * H / 2
            probes.append({"quad": k, "uv": [u, v], "pixel": [int(px), int(py)],
                           "value": expected_emission(u, v, TRANSFORMS[k], WRAPS[k])})
    p = np.array([0.0, -0.5 - 0.3, -D])
    normal_px = [int((p[0] / D / ty + 1) * W / 2), int((1 - p[1] / D / ty) * H / 2)]
    return {"probes": probes, "normal_pixel": normal_px, "n0_gltf": N0_GLTF.tolist(), "arrays": {k: v[0].tolist() for k, v in arrays.items()},
            "scene": scene}


if __name__ == "__main__":
    import json

    info = make_bridge_fixture(Path(sys.argv[1]))
    print(json.dumps({"probes": info["probes"], "normal_pixel": info["normal_pixel"]}))
