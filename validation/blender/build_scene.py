"""Scene bridge, Blender side: rebuild a scene package for Cycles (plan §5 M2, §1.2, §1.4, §1.4b, §7.5;
docs/decisions/scene-bridge.md is the contract with src/core/scene/scene-package.ts).

  built = build_from_package(package_dir)          # empty scene + ONE mesh + materials + lights + camera + world
  apply_frame(built, package["frames"][k])         # per-frame camera / light / env state
  cfg = settings_cfg(built, spp=..., seed=...)     # cycles_settings.apply_settings cfg for the current state

Conventions (math.md §3, §5):
- C = R_x(+90°): glTF (x, y, z) -> Blender (x, -z, y), applied exactly once to positions, normals, object
  matrices (M_b = C·M_g; object-local axes unchanged, so lights still emit along local -Z and cameras look down -Z).
- UVs: Blender v = 1 - v_glTF. KHR_texture_transform: Mapping node in Blender UV space via the glTF importer's
  texture_transform_gltf_to_blender (the package's 2x3 matrix is decomposed back to offset/rotation/scale).
- Faces are created in primId order (face index == primId), never merged or validated away.
- Smooth shading: the package's per-vertex normals are stored as a float 'custom_normal' POINT attribute
  (Blender 4.5+ free normals: bit-exact, unlike normals_split_custom_set's int16 encoding).
- Anything that cannot be represented exactly raises BridgeError (contract: hard error, not a warning).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cycles_settings as cs  # noqa: E402

C3 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # R_x(+90°)
C4 = np.eye(4)
C4[:3, :3] = C3
UV_MAP = "UVMap"
COLOR_ATTR = "Col"
_DTYPES = {"f32": "<f4", "u32": "<u4"}
_WRAP_EXT = {"repeat": "REPEAT", "clamp-to-edge": "EXTEND", "mirror-repeat": "MIRROR"}


class BridgeError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------------
# package IO


def load_package(package_dir: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    pkg = Path(package_dir)
    scene = json.loads((pkg / "scene.json").read_text())
    if scene.get("format") != "restir-scene-package" or scene.get("version") != 1:
        raise BridgeError(f"{pkg}: not a restir-scene-package v1 ({scene.get('format')!r}, {scene.get('version')!r})")
    blob = (pkg / "geometry.bin").read_bytes()
    arrays: dict[str, np.ndarray] = {}
    for name, b in scene.get("buffers", {}).items():
        dt = _DTYPES.get(b["dtype"])
        if dt is None:
            raise BridgeError(f"buffer {name}: unsupported dtype {b['dtype']!r}")
        off, length, comps = int(b["offset"]), int(b["length"]), int(b["components"])
        if off % 4 or length % (4 * comps) or off + length > len(blob):
            raise BridgeError(f"buffer {name}: bad range offset={off} length={length} comps={comps} (bin {len(blob)} B)")
        arrays[name] = np.frombuffer(blob, dtype=dt, count=length // 4, offset=off).reshape(-1, comps)
    return scene, arrays


def package_files(package_dir: str | Path) -> list[Path]:
    """Every file that defines the package (for hashing): scene.json, geometry.bin, tex_*.png, env file."""
    pkg = Path(package_dir)
    scene = json.loads((pkg / "scene.json").read_text())
    files = [pkg / "scene.json", pkg / "geometry.bin"]
    files += [pkg / t["file"] for t in scene.get("textures", [])]
    if scene.get("env"):
        files.append(pkg / scene["env"]["file"])
    return files


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_sha256(package_dir: str | Path) -> tuple[str, dict[str, str]]:
    per = {p.name: sha256_file(p) for p in package_files(package_dir)}
    h = hashlib.sha256("".join(f"{k}\0{v}\n" for k, v in sorted(per.items())).encode())
    return h.hexdigest(), per


# --------------------------------------------------------------------------------------------------
# frames and matrices


def gltf_matrix(m: list[float], what: str) -> np.ndarray:
    """Column-major 16 floats -> 4x4 (rows), asserting rotation + translation only (no scale/shear)."""
    if len(m) != 16:
        raise BridgeError(f"{what}: matrix needs 16 floats, got {len(m)}")
    M = np.asarray(m, dtype=np.float64).reshape(4, 4).T
    R = M[:3, :3]
    if not np.allclose(M[3], [0, 0, 0, 1], atol=1e-7):
        raise BridgeError(f"{what}: not affine: last row {M[3]}")
    if not np.allclose(R.T @ R, np.eye(3), atol=2e-5) or np.linalg.det(R) < 0:
        raise BridgeError(f"{what}: rotation part is not a proper rotation (scale/shear/mirror not allowed): {R.tolist()}")
    return M


def set_object_matrix(obj: bpy.types.Object, M_gltf: np.ndarray) -> None:
    """matrix_world = C @ M with scale exactly 1 (location + quaternion)."""
    Mb = C4 @ M_gltf
    obj.rotation_mode = "QUATERNION"
    obj.location = Vector(Mb[:3, 3].tolist())
    q = Matrix(Mb[:3, :3].tolist()).to_quaternion()
    obj.rotation_quaternion = Quaternion(q).normalized()
    obj.scale = (1.0, 1.0, 1.0)


# --------------------------------------------------------------------------------------------------
# mesh


def build_mesh(scene_json: dict[str, Any], arrays: dict[str, np.ndarray], materials: list[bpy.types.Material]) -> bpy.types.Object | None:
    for req in ("positions", "indices", "triMaterial"):
        if req not in arrays:
            raise BridgeError(f"missing buffer {req!r}")
    P = arrays["positions"].astype(np.float32)
    idx = arrays["indices"].astype(np.int64)
    tm = arrays["triMaterial"].reshape(-1).astype(np.int64)
    T, V = idx.shape[0], P.shape[0]
    if T == 0:
        return None
    if idx.shape[1] != 3 or P.shape[1] != 3:
        raise BridgeError("positions must be vec3 and indices uvec3")
    if tm.shape[0] != T:
        raise BridgeError(f"triMaterial has {tm.shape[0]} entries for {T} triangles")
    if idx.max() >= V:
        raise BridgeError(f"index {idx.max()} out of range ({V} vertices)")
    if tm.max() >= len(materials):
        raise BridgeError(f"material index {tm.max()} out of range ({len(materials)} materials)")
    if T >= 1 << 24:
        raise BridgeError(f"{T} triangles > 2^24 (plan §1.3)")

    me = bpy.data.meshes.new("package_mesh")
    Pb = np.empty_like(P)  # (x, y, z) -> (x, -z, y), exact in float32
    Pb[:, 0], Pb[:, 1], Pb[:, 2] = P[:, 0], -P[:, 2], P[:, 1]
    me.vertices.add(V)
    me.vertices.foreach_set("co", Pb.reshape(-1))
    me.loops.add(3 * T)
    me.loops.foreach_set("vertex_index", idx.reshape(-1).astype(np.int32))
    me.polygons.add(T)
    me.polygons.foreach_set("loop_start", (np.arange(T, dtype=np.int32) * 3))
    me.update(calc_edges=True)
    if len(me.polygons) != T or len(me.vertices) != V:
        raise BridgeError(f"Blender changed the topology: {len(me.polygons)} faces / {len(me.vertices)} verts")
    for m in materials:
        me.materials.append(m)
    me.polygons.foreach_set("material_index", tm.astype(np.int32))

    if "uv0" in arrays:
        uv = arrays["uv0"].astype(np.float32)
        if uv.shape != (V, 2):
            raise BridgeError(f"uv0 shape {uv.shape} != ({V}, 2)")
        uvb = uv[idx.reshape(-1)].copy()
        uvb[:, 1] = 1.0 - uvb[:, 1]
        layer = me.uv_layers.new(name=UV_MAP)
        layer.data.foreach_set("uv", uvb.reshape(-1))
    if "color0" in arrays:
        col = arrays["color0"].astype(np.float32)
        if col.shape != (V, 4):
            raise BridgeError(f"color0 shape {col.shape} != ({V}, 4)")
        ca = me.color_attributes.new(COLOR_ATTR, "FLOAT_COLOR", "POINT")
        ca.data.foreach_set("color", col.reshape(-1))
        me.color_attributes.active_color = ca
        me.color_attributes.render_color_index = me.color_attributes.find(COLOR_ATTR)

    if scene_json.get("flatShaded", True):
        me.shade_flat()
    else:
        if "normals" not in arrays:
            raise BridgeError("smooth shading needs the 'normals' buffer")
        N = arrays["normals"].astype(np.float32)
        if N.shape != (V, 3):
            raise BridgeError(f"normals shape {N.shape} != ({V}, 3)")
        Nb = np.empty_like(N)
        Nb[:, 0], Nb[:, 1], Nb[:, 2] = N[:, 0], -N[:, 2], N[:, 1]
        me.shade_smooth()
        a = me.attributes.new("custom_normal", "FLOAT_VECTOR", "POINT")
        a.data.foreach_set("vector", Nb.reshape(-1))
        me.update()
        if me.normals_domain != "POINT" or not me.has_custom_normals:
            raise BridgeError(f"custom normals not active (normals_domain={me.normals_domain})")
        got = np.empty(3 * V, np.float32)
        me.vertex_normals.foreach_get("vector", got)
        if not np.array_equal(got.reshape(-1, 3), Nb):
            raise BridgeError(f"custom normals not bit-exact (max err {np.abs(got.reshape(-1, 3) - Nb).max()})")
    ob = bpy.data.objects.new("package_mesh", me)
    return ob


# --------------------------------------------------------------------------------------------------
# materials


class _Tex:
    """Image datablocks per (texture index, colorspace); one PNG may be both sRGB and Non-Color."""

    def __init__(self, pkg: Path, textures: list[dict[str, Any]]):
        self.pkg, self.textures = pkg, textures
        self.cache: dict[tuple[int, str], bpy.types.Image] = {}

    def image(self, i: int, colorspace: str) -> bpy.types.Image:
        key = (i, colorspace)
        if key not in self.cache:
            t = self.textures[i]
            f = self.pkg / t["file"]
            if f.suffix.lower() != ".png":
                raise BridgeError(f"texture {i}: {f.name} is not a PNG (KTX2 etc. cannot be bridged exactly)")
            img = bpy.data.images.load(str(f), check_existing=False)
            img.name = f"tex{i}_{'srgb' if colorspace == 'sRGB' else 'data'}"
            img.colorspace_settings.name = colorspace
            img.alpha_mode = "CHANNEL_PACKED"  # raw RGBA, no (un)premultiplication: equals our texel fetch
            self.cache[key] = img
        return self.cache[key]


def _decompose_transform(tf: list[float], what: str) -> tuple[list[float], float, list[float]]:
    """Package 2x3 [a b c; d e f] = [c·sx, s·sy, ox; -s·sx, c·sy, oy] (gltf-loader.ts textureTransformMatrix)
    -> KHR_texture_transform offset, rotation, scale."""
    a, b, c, d, e, f = (float(x) for x in tf)
    sx = math.hypot(a, d)
    if sx == 0:
        raise BridgeError(f"{what}: degenerate texture transform {tf}")
    rot = math.atan2(-d, a)
    cr, sr = math.cos(rot), math.sin(rot)
    sy = b * sr + e * cr
    rebuilt = [cr * sx, sr * sy, c, -sr * sx, cr * sy, f]
    if max(abs(x - y) for x, y in zip(rebuilt, (a, b, c, d, e, f))) > 1e-5 * max(1.0, abs(sx), abs(sy)):
        raise BridgeError(f"{what}: texture transform {tf} is not offset·rotation·scale (skew)")
    return [c, f], rot, [sx, sy]


def _gltf_to_blender_transform(offset: list[float], rotation: float, scale: list[float]) -> dict[str, Any]:
    """io_scene_gltf2/blender/com/conversion.py texture_transform_gltf_to_blender (Blender 5.1.2), verbatim."""
    return {
        "offset": [offset[0] + scale[1] * math.sin(rotation), 1 - offset[1] - scale[1] * math.cos(rotation)],
        "rotation": rotation,
        "scale": [scale[0], scale[1]],
    }


class _MatBuilder:
    def __init__(self, mat: bpy.types.Material, tex: _Tex):
        self.mat, self.tex = mat, tex
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.out = self.nt.nodes.new("ShaderNodeOutputMaterial")
        self.out.target = "ALL"

    def link(self, a: Any, b: Any) -> None:
        self.nt.links.new(a, b)

    def new(self, kind: str, **props: Any) -> Any:
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def texture(self, ref: dict[str, Any], colorspace: str, what: str) -> Any:
        """Image Texture node (Linear/Closest, extension from wrap, UV map, optional Mapping)."""
        if int(ref.get("texCoord", 0)) != 0:
            raise BridgeError(f"{what}: texCoord {ref.get('texCoord')} (only TEXCOORD_0 is exported)")
        i = int(ref["texture"])
        t = self.tex.textures[i]
        n = self.new("ShaderNodeTexImage")
        n.image = self.tex.image(i, colorspace)
        filt = t.get("filter", "linear")
        if filt == "nearest":
            n.interpolation = "Closest"
            n["restir_interpolation"] = "Closest"  # cycles_settings asserts this instead of 'Linear'
        elif filt == "linear":
            n.interpolation = "Linear"
        else:
            raise BridgeError(f"texture {i}: filter {filt!r}")
        ws, wt = t.get("wrapS", "repeat"), t.get("wrapT", "repeat")
        for w in (ws, wt):
            if w not in _WRAP_EXT:
                raise BridgeError(f"texture {i}: wrap {w!r}")
        uv = self.new("ShaderNodeUVMap", uv_map=UV_MAP)
        sock = uv.outputs["UV"]
        if ref.get("transform") is not None:
            off, rot, sc = _decompose_transform(ref["transform"], what)
            tb = _gltf_to_blender_transform(off, rot, sc)
            mp = self.new("ShaderNodeMapping", vector_type="POINT")
            mp.inputs["Location"].default_value = (tb["offset"][0], tb["offset"][1], 0.0)
            mp.inputs["Rotation"].default_value = (0.0, 0.0, tb["rotation"])
            mp.inputs["Scale"].default_value = (tb["scale"][0], tb["scale"][1], 1.0)
            self.link(sock, mp.inputs["Vector"])
            sock = mp.outputs["Vector"]
        if ws == wt:
            n.extension = _WRAP_EXT[ws]
        else:  # the glTF importer's per-axis wrap: Separate XYZ -> WRAP/PINGPONG -> Combine XYZ, image EXTEND
            n.extension = "EXTEND"
            sep = self.new("ShaderNodeSeparateXYZ")
            com = self.new("ShaderNodeCombineXYZ")
            self.link(sock, sep.inputs[0])
            for k, w in enumerate((ws, wt)):
                s = sep.outputs[k]
                if w == "repeat":
                    m = self.new("ShaderNodeMath", operation="WRAP")
                    m.inputs[1].default_value, m.inputs[2].default_value = 0.0, 1.0
                    self.link(s, m.inputs[0])
                    s = m.outputs[0]
                elif w == "mirror-repeat":
                    m = self.new("ShaderNodeMath", operation="PINGPONG")
                    m.inputs[1].default_value = 1.0
                    self.link(s, m.inputs[0])
                    s = m.outputs[0]
                self.link(s, com.inputs[k])
            sock = com.outputs[0]
        self.link(sock, n.inputs["Vector"])
        return n

    def color_times(self, factor: list[float], sockets: list[Any]) -> Any:
        """factor (rgb) × each colour socket (Mix RGBA MULTIPLY chain); returns the output socket or None."""
        out = None
        for s in sockets:
            if out is None:
                out = s
                continue
            m = self.new("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
            m.inputs["Factor"].default_value = 1.0
            self.link(out, m.inputs[6])
            self.link(s, m.inputs[7])
            out = m.outputs[2]
        if out is None:
            return None
        if list(factor) != [1.0, 1.0, 1.0]:
            m = self.new("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY")
            m.inputs["Factor"].default_value = 1.0
            self.link(out, m.inputs[6])
            m.inputs[7].default_value = (*factor, 1.0)
            out = m.outputs[2]
        return out

    def scalar_times(self, factor: float, socket: Any) -> Any:
        if factor == 1.0:
            return socket
        m = self.new("ShaderNodeMath", operation="MULTIPLY")
        self.link(socket, m.inputs[0])
        m.inputs[1].default_value = factor
        return m.outputs[0]


def _v3(x: Any, what: str) -> list[float]:
    v = [float(c) for c in x]
    if len(v) != 3:
        raise BridgeError(f"{what}: expected 3 components, got {x!r}")
    return v


def _emission(m: dict[str, Any]) -> tuple[list[float], float]:
    e = m.get("emission")
    if e is not None:
        return _v3(e.get("color", [0, 0, 0]), "emission.color"), float(e.get("strength", 0.0))
    p = m.get("principled", m)
    return _v3(p.get("emissiveFactor", [0, 0, 0]), "emissiveFactor"), float(p.get("emissiveStrength", 1.0))


def build_material_v1(b: _MatBuilder, m: dict[str, Any]) -> dict[str, Any]:
    v = m["v1"]
    diff = b.new("ShaderNodeBsdfDiffuse")
    diff.inputs["Color"].default_value = (*_v3(v["diffuse"], "v1.diffuse"), 1.0)
    diff.inputs["Roughness"].default_value = 0.0
    glos = b.new("ShaderNodeBsdfAnisotropic")  # 'Glossy BSDF' (bl_idname kept from 4.0)
    glos.distribution = "GGX"
    glos.inputs["Color"].default_value = (*_v3(v["glossy"], "v1.glossy"), 1.0)
    glos.inputs["Roughness"].default_value = float(v["roughness"])
    glos.inputs["Anisotropy"].default_value = 0.0
    mix = b.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = float(v["mix"])
    b.link(diff.outputs[0], mix.inputs[1])
    b.link(glos.outputs[0], mix.inputs[2])
    surf = mix.outputs[0]
    col, strength = _emission(m)
    if strength > 0 and any(c != 0 for c in col):
        em = b.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*col, 1.0)
        em.inputs["Strength"].default_value = strength
        add = b.new("ShaderNodeAddShader")
        b.link(surf, add.inputs[0])
        b.link(em.outputs[0], add.inputs[1])
        surf = add.outputs[0]
    b.link(surf, b.out.inputs["Surface"])
    return {"model": "v1", "diffuse": v["diffuse"], "glossy": v["glossy"], "roughness": v["roughness"], "mix": v["mix"],
            "emission": [c * strength for c in col]}


def build_material_principled(b: _MatBuilder, m: dict[str, Any], has_color0: bool) -> dict[str, Any]:
    p = m.get("principled", m)
    name = m.get("name", "?")
    bsdf = b.new("ShaderNodeBsdfPrincipled")
    bsdf.distribution = "GGX"
    bsdf.subsurface_method = "BURLEY"
    I = bsdf.inputs
    # Tier-1 inputs only; everything else explicitly neutral (math.md#bsdf-v2)
    for k, v in (("Diffuse Roughness", 0.0), ("Subsurface Weight", 0.0), ("Anisotropic", 0.0), ("Anisotropic Rotation", 0.0),
                 ("Coat Weight", 0.0), ("Sheen Weight", 0.0), ("Thin Film Thickness", 0.0)):
        I[k].default_value = v
    bc = [float(x) for x in p.get("baseColorFactor", [1, 1, 1, 1])]
    if len(bc) != 4:
        raise BridgeError(f"{name}: baseColorFactor needs 4 components")
    alpha_mode = p.get("alphaMode", "OPAQUE")
    if alpha_mode == "BLEND":
        raise BridgeError(f"{name}: alphaMode BLEND cannot be bridged exactly (contract: hard error)")
    if alpha_mode not in ("OPAQUE", "MASK"):
        raise BridgeError(f"{name}: alphaMode {alpha_mode!r}")
    cutoff = float(p.get("alphaCutoff", 0.5))

    # base colour = factor × texture(sRGB) × COLOR_0 ; alpha = factor.a × texture.a × COLOR_0.a
    col_socks, alpha_socks = [], []
    if p.get("baseColorTexture"):
        t = b.texture(p["baseColorTexture"], "sRGB", f"{name}.baseColorTexture")
        col_socks.append(t.outputs["Color"])
        alpha_socks.append(t.outputs["Alpha"])
    if has_color0:
        vc = b.new("ShaderNodeVertexColor", layer_name=COLOR_ATTR)
        col_socks.append(vc.outputs["Color"])
        alpha_socks.append(vc.outputs["Alpha"])
    s = b.color_times(bc[:3], col_socks)
    if s is None:
        I["Base Color"].default_value = (*bc[:3], 1.0)
    else:
        b.link(s, I["Base Color"])
    if alpha_mode == "OPAQUE":
        I["Alpha"].default_value = 1.0
    else:
        if alpha_socks:
            a = alpha_socks[0]
            for s2 in alpha_socks[1:]:
                mm = b.new("ShaderNodeMath", operation="MULTIPLY")
                b.link(a, mm.inputs[0])
                b.link(s2, mm.inputs[1])
                a = mm.outputs[0]
            a = b.scalar_times(bc[3], a)
            lt = b.new("ShaderNodeMath", operation="LESS_THAN")  # 1 - (alpha < cutoff), as the glTF importer
            b.link(a, lt.inputs[0])
            lt.inputs[1].default_value = cutoff
            sub = b.new("ShaderNodeMath", operation="SUBTRACT")
            sub.inputs[0].default_value = 1.0
            b.link(lt.outputs[0], sub.inputs[1])
            b.link(sub.outputs[0], I["Alpha"])
        else:
            I["Alpha"].default_value = 0.0 if bc[3] < cutoff else 1.0

    metal, rough = float(p.get("metallicFactor", 1.0)), float(p.get("roughnessFactor", 1.0))
    if p.get("metallicRoughnessTexture"):
        t = b.texture(p["metallicRoughnessTexture"], "Non-Color", f"{name}.metallicRoughnessTexture")
        sep = b.new("ShaderNodeSeparateColor")
        b.link(t.outputs["Color"], sep.inputs[0])
        b.link(b.scalar_times(metal, sep.outputs["Blue"]), I["Metallic"])
        b.link(b.scalar_times(rough, sep.outputs["Green"]), I["Roughness"])
    else:
        I["Metallic"].default_value, I["Roughness"].default_value = metal, rough

    I["IOR"].default_value = float(p.get("ior", 1.5))
    spec = 0.5 * float(p.get("specularFactor", 1.0))
    if p.get("specularTexture"):
        t = b.texture(p["specularTexture"], "Non-Color", f"{name}.specularTexture")
        b.link(b.scalar_times(spec, t.outputs["Alpha"]), I["Specular IOR Level"])
    else:
        I["Specular IOR Level"].default_value = spec
    sc = _v3(p.get("specularColorFactor", [1, 1, 1]), "specularColorFactor")
    if p.get("specularColorTexture"):
        t = b.texture(p["specularColorTexture"], "sRGB", f"{name}.specularColorTexture")
        b.link(b.color_times(sc, [t.outputs["Color"]]), I["Specular Tint"])
    else:
        I["Specular Tint"].default_value = (*sc, 1.0)
    tr = float(p.get("transmissionFactor", 0.0))
    if p.get("transmissionTexture"):
        t = b.texture(p["transmissionTexture"], "Non-Color", f"{name}.transmissionTexture")
        sep = b.new("ShaderNodeSeparateColor")
        b.link(t.outputs["Color"], sep.inputs[0])
        b.link(b.scalar_times(tr, sep.outputs["Red"]), I["Transmission Weight"])
    else:
        I["Transmission Weight"].default_value = tr

    ecol, estr = _emission(m)
    if p.get("emissiveTexture"):
        t = b.texture(p["emissiveTexture"], "sRGB", f"{name}.emissiveTexture")
        b.link(b.color_times(ecol, [t.outputs["Color"]]), I["Emission Color"])
    else:
        I["Emission Color"].default_value = (*ecol, 1.0)
    I["Emission Strength"].default_value = estr

    if p.get("normalTexture"):
        nref = p["normalTexture"]
        t = b.texture(nref, "Non-Color", f"{name}.normalTexture")
        nm = b.new("ShaderNodeNormalMap", space="TANGENT", uv_map=UV_MAP)
        nm.inputs["Strength"].default_value = float(nref.get("scale", 1.0))
        b.link(t.outputs["Color"], nm.inputs["Color"])
        b.link(nm.outputs["Normal"], I["Normal"])
    b.link(bsdf.outputs[0], b.out.inputs["Surface"])
    return {"model": "principled", "alphaMode": alpha_mode, "emission": [c * estr for c in ecol],
            "textures": sorted(k for k in p if k.endswith("Texture") and p[k])}


def build_materials(scene_json: dict[str, Any], pkg: Path, has_color0: bool) -> tuple[list[bpy.types.Material], dict[str, Any]]:
    tex = _Tex(pkg, scene_json.get("textures", []))
    mats, info, sampling = [], [], {}
    for i, m in enumerate(scene_json.get("materials", [])):
        mat = bpy.data.materials.new(f"m{i:03d}_{m.get('name', '')}"[:63])
        if mat.node_tree is None:  # Blender 5 materials always have a node tree; older ones need use_nodes
            mat.use_nodes = True
        b = _MatBuilder(mat, tex)
        model = m.get("model")
        if model == "v1":
            d = build_material_v1(b, m)
        elif model == "principled":
            d = build_material_principled(b, m, has_color0)
        else:
            raise BridgeError(f"material {i}: unknown model {model!r}")
        es = m.get("emissionSampling", "FRONT_BACK")
        if es not in ("FRONT_BACK", "NONE", "AUTO", "FRONT", "BACK"):
            raise BridgeError(f"material {i}: emissionSampling {es!r}")
        mat.cycles.emission_sampling = es
        sampling[mat.name] = es
        mats.append(mat)
        info.append({"index": i, "blender_name": mat.name, **d, "emissionSampling": es})
    return mats, {"materials": info, "emission_sampling": sampling,
                  "images": {img.name: {"file": Path(img.filepath).name, "colorspace": img.colorspace_settings.name}
                             for img in tex.cache.values()}}


# --------------------------------------------------------------------------------------------------
# lights, camera, world


_LIGHT_TYPES = {"point": "POINT", "spot": "SPOT", "rect": "AREA", "disk": "AREA", "sun": "SUN"}


def build_light(L: dict[str, Any], light_mode: str) -> tuple[bpy.types.Object, dict[str, Any]]:
    t = L["type"]
    if t not in _LIGHT_TYPES:
        raise BridgeError(f"light {L.get('id')}: type {t!r}")
    name = f"light{int(L['id']):03d}_{L.get('name', '')}"[:63]
    ld = bpy.data.lights.new(name, _LIGHT_TYPES[t])
    ob = bpy.data.objects.new(name, ld)
    set_object_matrix(ob, gltf_matrix(L["matrix"], f"light {name}"))
    vis = bool(L.get("visibleToCamera", False))
    if light_mode == "A" and t in ("rect", "disk") and vis:
        raise BridgeError(f"light {name}: Mode A with a camera-visible area light (Cycles would not show it; plan §1.4)")
    spec: dict[str, Any] = {
        "visible_camera": vis, "exposure": float(L.get("exposure", 0.0)),
        "energy": float(L["power"]), "color": tuple(_v3(L.get("color", [1, 1, 1]), f"{name}.color")),
    }
    if t == "spot":
        spec["spot_size"] = float(L["spotSize"])
        spec["spot_blend"] = float(L.get("spotBlend", 0.0))
    if t in ("rect", "disk"):
        spec["spread"] = float(L.get("spread", math.pi))
        if t == "rect":
            spec.update(shape="RECTANGLE", size=float(L["sizeX"]), size_y=float(L["sizeY"]))
        else:
            if L.get("sizeY") is not None and abs(float(L["sizeY"]) - float(L["sizeX"])) > 1e-9:
                raise BridgeError(f"light {name}: disk with sizeY != sizeX (ellipse) not in the contract")
            spec.update(shape="DISK", size=float(L["sizeX"]))
    # set now (apply_settings re-asserts the same values through the cycles_settings tables)
    for k in ("energy", "color", "exposure", "spot_size", "spot_blend", "spread", "shape", "size", "size_y"):
        if k in spec:
            setattr(ld, k, spec[k])
    ob.visible_camera = vis
    ld.cycles.use_multiple_importance_sampling = light_mode == "B"
    if t in ("point", "spot"):
        ld.shadow_soft_size = 0.0
        ld.use_soft_falloff = False
    if t == "sun":
        ld.angle = 0.0
    return ob, spec


def build_camera(cam: dict[str, Any]) -> bpy.types.Object:
    cd = bpy.data.cameras.new("camera")
    co = bpy.data.objects.new("camera", cd)
    set_object_matrix(co, gltf_matrix(cam["matrix"], "camera"))
    cd.type = "PERSP"
    cd.sensor_fit = "VERTICAL"
    cd.angle_y = float(cam["yfov"])
    cd.clip_start = float(cam.get("znear", 1e-4))
    cd.clip_end = 1e5
    return co


def image_pixels_sha256(img: bpy.types.Image) -> str:
    """ENV-U9 (scene-bridge.md): SHA-256 of float32 LE RGBA rows TOP-DOWN (Blender rows are bottom-up -> flip)."""
    w, h = img.size
    if img.channels != 4:
        raise BridgeError(f"{img.name}: {img.channels} channels in image.pixels (expected 4)")
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    return hashlib.sha256(np.ascontiguousarray(px.reshape(h, w, 4)[::-1], dtype="<f4").tobytes()).hexdigest()


def exr_file_sha256(path: Path) -> tuple[str, tuple[int, int]]:
    """Same hash computed from the EXR file itself via OpenImageIO (missing alpha -> 1)."""
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise BridgeError(f"{path}: {oiio.geterror()}")
    spec = inp.spec()
    px = np.asarray(inp.read_image(0, 0, 0, spec.nchannels, "float"), np.float32).reshape(spec.height, spec.width, spec.nchannels)
    inp.close()
    names = list(spec.channelnames)
    if names[:3] != ["R", "G", "B"] or len(names) not in (3, 4):
        raise BridgeError(f"{path}: channels {names} (expected RGB or RGBA)")
    rgba = np.ones((spec.height, spec.width, 4), np.float32)
    rgba[..., :spec.nchannels] = px
    return hashlib.sha256(np.ascontiguousarray(rgba, dtype="<f4").tobytes()).hexdigest(), (spec.width, spec.height)


def build_env(scene: bpy.types.Scene, env: dict[str, Any], pkg: Path) -> tuple[bpy.types.World, dict[str, Any]]:
    path = pkg / env["file"]
    wcfg = {"hdri": str(path), "rotation_z": float(env.get("rotationZ", 0.0)), "strength": float(env.get("strength", 1.0)),
            "tint": tuple(_v3(env.get("tint", [1, 1, 1]), "env.tint"))}
    w = cs.build_world(scene, wcfg)
    tex = next(n for n in w.node_tree.nodes if n.bl_idname == "ShaderNodeTexEnvironment")
    img = tex.image
    img.colorspace_settings.name = cs.WORKING_SPACE
    img.alpha_mode = "NONE"  # plan §1.4b / env table: alpha ignored
    w.cycles.sampling_method = env.get("sampling", "AUTOMATIC")
    w.cycles_visibility.camera = bool(env.get("visibleToCamera", True))
    got = image_pixels_sha256(img)
    file_hash, (fw, fh) = exr_file_sha256(path)
    info = {"file": env["file"], "size": list(img.size), "image_sha256": got, "exr_file_sha256": file_hash,
            "declared_sha256": env.get("sha256"), "colorspace": img.colorspace_settings.name}
    if tuple(img.size) != (fw, fh):
        raise BridgeError(f"env: Blender image size {tuple(img.size)} != EXR {(fw, fh)}")
    if got != file_hash:
        raise BridgeError(f"ENV-U9: Blender image.pixels hash {got} != EXR pixel hash {file_hash} (Blender altered texels)")
    if env.get("sha256") and got != env["sha256"]:
        raise BridgeError(f"ENV-U9: Blender image.pixels hash {got} != package env.sha256 {env['sha256']}")
    info["hash_ok"] = True
    return w, info


def build_env_constant(scene: bpy.types.Scene, env: dict[str, Any], pkg: Path) -> tuple[bpy.types.World, dict[str, Any]]:
    """M3c C0q variant (env "blenderWorld": "constant"): the package's env texture must be one constant colour c; Blender
    gets a plain Background node with Color = c * tint and the package strength. Cycles then has NO background light
    (a constant world is not spatially varying, light.cpp:280-300): BSDF-only env, weight 1. Our renderer samples the
    same constant texture (with env NEE unless env.sampling is NONE); both must agree (plan §7.2 C0q)."""
    import OpenImageIO as oiio

    path = pkg / env["file"]
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise BridgeError(f"{path}: {oiio.geterror()}")
    spec = inp.spec()
    px = np.asarray(inp.read_image(0, 0, 0, spec.nchannels, "float"), np.float32).reshape(-1, spec.nchannels)[:, :3]
    inp.close()
    if not np.all(px == px[0]):
        raise BridgeError("blenderWorld 'constant' needs a constant env texture")
    tint = _v3(env.get("tint", [1, 1, 1]), "env.tint")
    color = tuple(float(px[0][k]) * float(tint[k]) for k in range(3))
    w = cs.build_world(scene, {"color": color, "strength": float(env.get("strength", 1.0))})
    w.cycles.sampling_method = env.get("sampling", "AUTOMATIC")
    w.cycles_visibility.camera = bool(env.get("visibleToCamera", True))
    file_hash, (fw, fh) = exr_file_sha256(path)
    if env.get("sha256") and file_hash != env["sha256"]:
        raise BridgeError(f"ENV-U9: EXR pixel hash {file_hash} != package env.sha256 {env['sha256']}")
    return w, {"file": env["file"], "size": [fw, fh], "blenderWorld": "constant", "color": list(color), "exr_file_sha256": file_hash,
               "declared_sha256": env.get("sha256"), "hash_ok": True}


def _world_nodes(w: bpy.types.World) -> tuple[Any, Any, Any]:
    nodes = w.node_tree.nodes
    mp = next(n for n in nodes if n.bl_idname == "ShaderNodeMapping")
    bg = next(n for n in nodes if n.bl_idname == "ShaderNodeBackground")
    tint = next(n for n in nodes if n.bl_idname == "ShaderNodeVectorMath")
    return mp, bg, tint


# --------------------------------------------------------------------------------------------------
# entry points


def build_from_package(package_dir: str | Path, *, reset: bool = True) -> dict[str, Any]:
    """Empty scene -> package scene. Returns {'scene', 'mesh', 'camera', 'lights' {id: obj}, 'world', 'package',
    'state', 'manifest', ...}; see settings_cfg() for the cycles_settings cfg."""
    pkg = Path(package_dir).resolve()
    sj, arrays = load_package(pkg)
    if reset:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    light_mode = sj.get("lightMode", "A")
    if light_mode not in ("A", "B"):
        raise BridgeError(f"lightMode {light_mode!r}")

    has_color0 = "color0" in arrays
    mats, mat_info = build_materials(sj, pkg, has_color0)
    mesh = build_mesh(sj, arrays, mats)
    if mesh is not None:
        scene.collection.objects.link(mesh)

    lights: dict[int, bpy.types.Object] = {}
    light_specs: dict[int, dict[str, Any]] = {}
    for L in sj.get("lights", []):
        lid = int(L["id"])
        if lid in lights:
            raise BridgeError(f"duplicate light id {lid}")
        ob, spec = build_light(L, light_mode)
        scene.collection.objects.link(ob)
        lights[lid], light_specs[lid] = ob, spec

    cam = build_camera(sj["camera"])
    scene.collection.objects.link(cam)
    scene.camera = cam

    world, env_info = None, None
    if sj.get("env"):
        builder = build_env_constant if sj["env"].get("blenderWorld") == "constant" else build_env
        world, env_info = builder(scene, sj["env"], pkg)
    else:
        scene.world = None

    r = sj.get("render", {})
    scene.render.resolution_x, scene.render.resolution_y = int(r.get("width", 512)), int(r.get("height", 512))
    scene.render.resolution_percentage = 100
    bpy.context.view_layer.update()  # matrix_world is only valid after a depsgraph update

    psha, per_file = package_sha256(pkg)
    T = int(arrays["indices"].shape[0]) if "indices" in arrays else 0
    built = {
        "scene": scene, "mesh": mesh, "camera": cam, "lights": lights, "world": world, "materials": mats,
        "package": sj, "package_dir": str(pkg), "light_mode": light_mode,
        "state": {"yfov": float(sj["camera"]["yfov"]), "znear": float(sj["camera"].get("znear", 1e-4)),
                  "light_specs": light_specs, "frame": None},
        "manifest": {
            "package": str(pkg), "package_sha256": psha, "package_files": per_file,
            "name": sj.get("name"), "blender": bpy.app.version_string,
            "conversion": "C = R_x(+90deg): glTF (x,y,z) -> Blender (x,-z,y); matrix_world = C @ M; uv v_b = 1 - v",
            "mesh": None if mesh is None else {
                "object": mesh.name, "triangles": T, "vertices": int(arrays["positions"].shape[0]),
                "faces_in_primId_order": True, "flatShaded": bool(sj.get("flatShaded", True)),
                "uv0": "uv0" in arrays, "color0": has_color0,
            },
            **mat_info,
            "lights": {str(k): {"object": o.name, "type": o.data.type, **{kk: (list(v) if isinstance(v, tuple) else v)
                                                                       for kk, v in light_specs[k].items()}}
                       for k, o in lights.items()},
            "light_mode": light_mode,
            "camera": {"object": cam.name, "yfov": cam.data.angle_y, "clip_start": cam.data.clip_start,
                       "matrix_world": [list(row) for row in cam.matrix_world]},
            "env": env_info,
            "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        },
    }
    return built


def apply_frame(built: dict[str, Any], frame: dict[str, Any] | None) -> None:
    """Apply one resolved frame state (scene-bridge.md 'frames'). None = the package's base state."""
    sj = built["package"]
    base_cam = sj["camera"]
    fcam = (frame or {}).get("camera") or {}
    set_object_matrix(built["camera"], gltf_matrix(fcam.get("matrix", base_cam["matrix"]), "frame camera"))
    built["state"]["yfov"] = float(fcam.get("yfov", base_cam["yfov"]))
    built["camera"].data.angle_y = built["state"]["yfov"]
    base_lights = {int(L["id"]): L for L in sj.get("lights", [])}
    flights = (frame or {}).get("lights") or {}
    for lid, ob in built["lights"].items():
        fl = flights.get(str(lid), flights.get(lid, {})) or {}
        bl = base_lights[lid]
        set_object_matrix(ob, gltf_matrix(fl.get("matrix", bl["matrix"]), f"frame light {lid}"))
        p = float(fl.get("power", bl["power"]))
        ob.data.energy = p
        built["state"]["light_specs"][lid]["energy"] = p
    unknown = set(map(str, flights)) - set(map(str, built["lights"]))
    if unknown:
        raise BridgeError(f"frame {frame.get('frame') if frame else None}: unknown light ids {sorted(unknown)}")
    if built["world"] is not None and sj["env"].get("blenderWorld") == "constant":
        if (frame or {}).get("env"):
            raise BridgeError("blenderWorld 'constant' does not support per-frame env parameters")
    elif built["world"] is not None:
        env = sj["env"]
        fe = (frame or {}).get("env") or {}
        mp, bg, tint = _world_nodes(built["world"])
        mp.inputs["Rotation"].default_value = (0.0, 0.0, float(fe.get("rotationZ", env.get("rotationZ", 0.0))))
        bg.inputs["Strength"].default_value = float(fe.get("strength", env.get("strength", 1.0)))
        tint.inputs[1].default_value = tuple(_v3(fe.get("tint", env.get("tint", [1, 1, 1])), "frame env tint"))
    elif (frame or {}).get("env"):
        raise BridgeError("frame has env parameters but the package has no env")
    built["state"]["frame"] = None if frame is None else frame.get("frame")
    bpy.context.view_layer.update()


def frame_state(built: dict[str, Any]) -> dict[str, Any]:
    """Read back the frame-dependent Blender state (for manifests)."""
    out: dict[str, Any] = {
        "camera": {"matrix_world": [list(r) for r in built["camera"].matrix_world], "angle_y": built["camera"].data.angle_y},
        "lights": {str(k): {"energy": o.data.energy, "matrix_world": [list(r) for r in o.matrix_world]} for k, o in built["lights"].items()},
    }
    if built["world"] is not None and built["package"]["env"].get("blenderWorld") == "constant":
        bg = next(n for n in built["world"].node_tree.nodes if n.bl_idname == "ShaderNodeBackground")
        out["env"] = {"constant": list(bg.inputs["Color"].default_value), "strength": bg.inputs["Strength"].default_value}
    elif built["world"] is not None:
        mp, bg, tint = _world_nodes(built["world"])
        out["env"] = {"rotation": list(mp.inputs["Rotation"].default_value), "strength": bg.inputs["Strength"].default_value,
                      "tint": list(tint.inputs[1].default_value)}
    return out


def settings_cfg(built: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """cycles_settings.apply_settings cfg for the CURRENT state (call after apply_frame)."""
    sj = built["package"]
    r = sj.get("render", {})
    env = sj.get("env")
    cfg: dict[str, Any] = {
        "resolution": (int(r.get("width", 512)), int(r.get("height", 512))),
        "max_bounces": int(r.get("maxBounces", 3)),
        "light_mis": built["light_mode"] == "B",
        "camera": {"vfov_rad": built["state"]["yfov"], "clip_start": built["state"]["znear"], "clip_end": 1e5},
        "lights": {o.name: dict(built["state"]["light_specs"][k]) for k, o in built["lights"].items()},
        "emission_sampling": "FRONT_BACK",
        "material_emission_sampling": dict(built["manifest"]["emission_sampling"]),
        "world": None if env is None else {
            "keep_existing": True, "sampling_method": env.get("sampling", "AUTOMATIC"),
            "visible_camera": bool(env.get("visibleToCamera", True)),
        },
    }
    # Per-package Cycles overrides (scene-bridge.md "cycles"): only the documented deviation D4 (light tree) so far.
    for k, v in (sj.get("cycles") or {}).items():
        if k != "use_light_tree":
            raise BridgeError(f"scene.json cycles.{k}: unsupported override")
        cfg[k] = bool(v)
    cfg.update(overrides)
    return cfg
