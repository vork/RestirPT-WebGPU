"""(viii-L) reference dump of a USD file with OpenUSD (pxr) for the loader-fidelity gate (docs/decisions/m7-api.md §3.2).

  /Applications/Blender.app/Contents/Resources/5.2/python/bin/python3.13 validation/tools/usd_pxr_dump.py FILE.usd[a|c|z] --out DUMP.json

UsdImaging conventions (as validation/spikes/usd/pxr_dump.py): instance proxies expanded, abstract (class) prims and
PointInstancer prototype subtrees not drawn directly, invisible / guide / proxy prims pruned. Everything geometric is
given in the renderer's canonical frame (glTF: +Y up, metres): p_c = W · p_usd with W = (Z-up ? R_x(−90°) : I) · mpu, the
same stage conversion as usd-lights.ts stageMatrix. Per draw: world-space (canonical) bounding box of the mesh points
and the triangle count of its faces (fan triangulation: n − 2 per n-gon) per bound material. PointInstancer instances are
draws named "<instancer path>[<i>]". Materials: the UsdPreviewSurface inputs (authored values, else the spec defaults) and,
for connected inputs, the UsdUVTexture binding (file, output channel, wrapS, wrapT, sourceColorSpace, scale, bias,
primvar varname). Lights: UsdLux fields + the canonical world matrix (column-major, column vectors). Cameras: vertical
FOV and canonical world matrix.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade

T = Usd.TimeCode.Default()


def val(attr: Usd.Attribute) -> Any:
    v = attr.Get(T) if attr else None
    if v is None:
        return None
    if isinstance(v, (Gf.Vec2f, Gf.Vec3f, Gf.Vec3d, Gf.Vec4f, Gf.Vec2d, Gf.Vec4d)):
        return [float(x) for x in v]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(v, "path"):
        return str(v.path)
    return str(v)


def stage_matrix(stage: Usd.Stage) -> Gf.Matrix4d:
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    s = Gf.Matrix4d().SetScale(Gf.Vec3d(mpu, mpu, mpu))
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z:
        return s * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90.0))   # row vectors: p·S·R
    return s


def colmajor(m: Gf.Matrix4d) -> list[float]:
    """USD row-vector matrix → column-major array for column vectors (the same 16 numbers in row order)."""
    return [float(m[i][j]) for i in range(4) for j in range(4)]


PS_DEFAULTS: dict[str, Any] = {
    "diffuseColor": [0.18, 0.18, 0.18], "emissiveColor": [0.0, 0.0, 0.0], "useSpecularWorkflow": 0.0, "specularColor": [0.0, 0.0, 0.0],
    "metallic": 0.0, "roughness": 0.5, "clearcoat": 0.0, "clearcoatRoughness": 0.01, "opacity": 1.0, "opacityThreshold": 0.0, "ior": 1.5,
    "normal": [0.0, 0.0, 1.0], "occlusion": 1.0,
}


def texture_binding(inp: UsdShade.Input) -> dict[str, Any] | None:
    srcs = inp.GetConnectedSources()[0] if hasattr(inp, "GetConnectedSources") else []
    if not srcs:
        return None
    s = srcs[0]
    sh = UsdShade.Shader(s.source.GetPrim())
    out: dict[str, Any] = {"shader": str(sh.GetPath()), "id": str(sh.GetIdAttr().Get() or ""), "channel": str(s.sourceName)}
    if out["id"] != "UsdUVTexture":
        return out
    def g(n):
        i = sh.GetInput(n)
        return val(i.GetAttr()) if i and i.GetAttr().HasAuthoredValue() else None
    f = sh.GetInput("file")
    out.update({"file": str(f.Get().path) if f and f.Get() else None, "wrapS": g("wrapS"), "wrapT": g("wrapT"), "sourceColorSpace": g("sourceColorSpace"),
                "scale": g("scale"), "bias": g("bias")})
    st = sh.GetInput("st")
    if st:
        ss = st.GetConnectedSources()[0]
        if ss:
            rd = UsdShade.Shader(ss[0].source.GetPrim())
            out["reader"] = str(rd.GetIdAttr().Get() or "")
            vn = rd.GetInput("varname")
            out["varname"] = val(vn.GetAttr()) if vn else None
    return out


def dump_material(prim: Usd.Prim) -> dict[str, Any]:
    m = UsdShade.Material(prim)
    out: dict[str, Any] = {"path": str(prim.GetPath()), "shader": None, "inputs": {}, "textures": {}}
    src = m.ComputeSurfaceSource()
    shader = src[0] if src and src[0] else None
    if not shader:
        return out
    out["shader"] = str(shader.GetIdAttr().Get() or "")
    for name, d in PS_DEFAULTS.items():
        inp = shader.GetInput(name)
        tb = texture_binding(inp) if inp else None
        if tb:
            out["textures"][name] = tb
        out["inputs"][name] = val(inp.GetAttr()) if inp and inp.GetAttr().HasAuthoredValue() else d
    return out


def bound_material(prim: Usd.Prim) -> str | None:
    m, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    return str(m.GetPath()) if m else None


def world_bbox(points, M: Gf.Matrix4d) -> list[float] | None:
    if not len(points):
        return None
    ps = [M.Transform(Gf.Vec3d(*p)) for p in points]
    return [min(p[k] for p in ps) for k in range(3)] + [max(p[k] for p in ps) for k in range(3)]


def dump_mesh(prim: Usd.Prim, world: Gf.Matrix4d, W: Gf.Matrix4d, materials: dict[str, Any], path: str) -> dict[str, Any]:
    mesh = UsdGeom.Mesh(prim)
    counts = list(mesh.GetFaceVertexCountsAttr().Get(T) or [])
    points = mesh.GetPointsAttr().Get(T) or []
    tri_per_face = [max(0, int(n) - 2) for n in counts]
    base = bound_material(prim)
    tris_by_mat: dict[str, int] = {}
    covered = [False] * len(counts)
    for s in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
        sm = bound_material(s.GetPrim()) or base or "<none>"
        for f in list(s.GetIndicesAttr().Get(T) or []):
            if 0 <= f < len(counts) and not covered[f]:
                covered[f] = True
                tris_by_mat[sm] = tris_by_mat.get(sm, 0) + tri_per_face[f]
    rest = sum(t for t, c in zip(tri_per_face, covered) if not c)
    if rest:
        tris_by_mat[base or "<none>"] = tris_by_mat.get(base or "<none>", 0) + rest
    for mp in set(tris_by_mat) | ({base} if base else set()):
        if mp and mp != "<none>" and mp not in materials:
            materials[mp] = dump_material(prim.GetStage().GetPrimAtPath(mp))
    st = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    return {"path": path, "mesh": str(prim.GetPath()), "points": len(points), "faces": len(counts), "triangles": sum(tri_per_face), "material": base,
            "trianglesByMaterial": tris_by_mat, "bbox": world_bbox(points, world * W), "hasSt": bool(st and st.HasValue()),
            "hasNormals": bool(mesh.GetNormalsAttr().HasAuthoredValue()), "orientation": str(mesh.GetOrientationAttr().Get(T))}


LIGHT_TYPES = {"SphereLight": "sphere", "RectLight": "rect", "DiskLight": "disk", "DistantLight": "distant", "CylinderLight": "cylinder", "DomeLight": "dome"}


def dump_light(prim: Usd.Prim, world: Gf.Matrix4d, W: Gf.Matrix4d) -> dict[str, Any]:
    t = prim.GetTypeName()
    L = UsdLux.LightAPI(prim)
    e: dict[str, Any] = {"path": str(prim.GetPath()), "type": LIGHT_TYPES.get(t, t), "intensity": val(L.GetIntensityAttr()), "exposure": val(L.GetExposureAttr()),
                         "color": val(L.GetColorAttr()), "normalize": val(L.GetNormalizeAttr()), "enableColorTemperature": val(L.GetEnableColorTemperatureAttr()),
                         "colorTemperature": val(L.GetColorTemperatureAttr()), "world": colmajor(world * W), "usdWorld": colmajor(world)}
    if t == "SphereLight":
        s = UsdLux.SphereLight(prim)
        e["radius"] = val(s.GetRadiusAttr()); e["treatAsPoint"] = val(s.GetTreatAsPointAttr())
    elif t == "DiskLight":
        e["radius"] = val(UsdLux.DiskLight(prim).GetRadiusAttr())
    elif t == "RectLight":
        r = UsdLux.RectLight(prim)
        e["width"] = val(r.GetWidthAttr()); e["height"] = val(r.GetHeightAttr())
    elif t == "DistantLight":
        e["angle"] = val(UsdLux.DistantLight(prim).GetAngleAttr())
    if prim.HasAPI(UsdLux.ShapingAPI):
        sh = UsdLux.ShapingAPI(prim)
        e["shaping"] = {"coneAngle": val(sh.GetShapingConeAngleAttr()), "coneSoftness": val(sh.GetShapingConeSoftnessAttr()),
                        "focus": val(sh.GetShapingFocusAttr()), "focusTint": val(sh.GetShapingFocusTintAttr())}
    return e


def dump_file(path: Path) -> dict[str, Any]:
    stage = Usd.Stage.Open(str(path))
    W = stage_matrix(stage)
    xf = UsdGeom.XformCache(T)
    out: dict[str, Any] = {"source": "pxr", "file": path.name, "stage": {"upAxis": str(UsdGeom.GetStageUpAxis(stage)),
                           "metersPerUnit": float(UsdGeom.GetStageMetersPerUnit(stage)), "doc": stage.GetRootLayer().documentation},
                           "draws": [], "lights": [], "materials": [], "pointInstancers": [], "cameras": []}
    materials: dict[str, Any] = {}
    pred = Usd.TraverseInstanceProxies(Usd.PrimIsActive & Usd.PrimIsDefined & ~Usd.PrimIsAbstract)
    it = iter(Usd.PrimRange(stage.GetPseudoRoot(), pred))
    for prim in it:
        img = UsdGeom.Imageable(prim)
        if img and img.ComputeVisibility(T) == UsdGeom.Tokens.invisible:
            it.PruneChildren(); continue
        if img and img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            it.PruneChildren(); continue
        if prim.IsA(UsdGeom.PointInstancer):
            pi = UsdGeom.PointInstancer(prim)
            protos = [str(p) for p in pi.GetPrototypesRel().GetTargets()]
            idx = list(pi.GetProtoIndicesAttr().Get(T) or [])
            w = xf.GetLocalToWorldTransform(prim)
            xfs = pi.ComputeInstanceTransformsAtTime(T, T)
            out["pointInstancers"].append({"path": str(prim.GetPath()), "prototypes": protos, "instances": len(idx)})
            for k, (p, x) in enumerate(zip(idx, xfs)):
                proto = stage.GetPrimAtPath(protos[p])
                for q in Usd.PrimRange(proto):
                    if q.IsA(UsdGeom.Mesh):
                        rel = xf.GetLocalToWorldTransform(q) * xf.GetLocalToWorldTransform(proto).GetInverse()
                        out["draws"].append(dump_mesh(q, rel * Gf.Matrix4d(x) * w, W, materials, f"{prim.GetPath()}[{k}]"))
            it.PruneChildren()
            continue
        if prim.IsA(UsdGeom.Mesh):
            out["draws"].append(dump_mesh(prim, xf.GetLocalToWorldTransform(prim), W, materials, str(prim.GetPath())))
        elif prim.HasAPI(UsdLux.LightAPI) or prim.GetTypeName() in LIGHT_TYPES:
            out["lights"].append(dump_light(prim, xf.GetLocalToWorldTransform(prim), W))
        elif prim.IsA(UsdGeom.Camera):
            c = UsdGeom.Camera(prim)
            fl, va = val(c.GetFocalLengthAttr()), val(c.GetVerticalApertureAttr())
            out["cameras"].append({"path": str(prim.GetPath()), "yfov": 2 * math.atan(va / (2 * fl)) if fl and va else None,
                                   "world": colmajor(xf.GetLocalToWorldTransform(prim) * W)})
    out["materials"] = sorted(materials.values(), key=lambda m: m["path"])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dump_file(a.file), indent=1) + "\n")
    print(f"[usd_pxr_dump] {a.file} -> {a.out}")


if __name__ == "__main__":
    main()
