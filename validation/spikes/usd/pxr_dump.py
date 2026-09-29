"""Reference dump of USD files with OpenUSD (pxr), in the same schema as the LightUSD spike page.

Run with Blender's bundled Python (pxr 0.25.8 importable there):
  /Applications/Blender.app/Contents/Resources/5.2/python/bin/python3.13 \
      validation/spikes/usd/pxr_dump.py validation/assets/usd-spike/*.usd? --out validation/out/usd-spike

Schema (shared with spike.ts, see SceneDump there): matrices are 16 floats, USD row-major for row
vectors (translation at 12..14), which is also the column-major layout for column vectors.
Draws follow UsdImaging conventions: instance proxies expanded, abstract (class) prims and
PointInstancer prototype subtrees are not drawn directly, invisible prims skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade

T = Usd.TimeCode.Default()


def mat(m: Gf.Matrix4d) -> list[float]:
    return [float(m[i][j]) for i in range(4) for j in range(4)]


def val(attr: Usd.Attribute) -> Any:
    v = attr.Get(T) if attr else None
    if v is None:
        return None
    if isinstance(v, (Gf.Vec3f, Gf.Vec3d, Gf.Vec2f, Gf.Vec4f)):
        return [float(x) for x in v]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return str(v)


LIGHT_TYPES = {
    "SphereLight": "sphere",
    "RectLight": "rect",
    "DiskLight": "disk",
    "DistantLight": "distant",
    "CylinderLight": "cylinder",
    "DomeLight": "dome",
    "DomeLight_1": "dome",
}


def dump_light(prim: Usd.Prim, xf_cache: UsdGeom.XformCache) -> dict[str, Any]:
    t = prim.GetTypeName()
    L = UsdLux.LightAPI(prim)
    e: dict[str, Any] = {
        "path": str(prim.GetPath()),
        "type": LIGHT_TYPES.get(t, t),
        "intensity": val(L.GetIntensityAttr()),
        "exposure": val(L.GetExposureAttr()),
        "color": val(L.GetColorAttr()),
        "normalize": val(L.GetNormalizeAttr()),
        "enableColorTemperature": val(L.GetEnableColorTemperatureAttr()),
        "colorTemperature": val(L.GetColorTemperatureAttr()),
        "radius": None,
        "width": None,
        "height": None,
        "angle": None,
        "treatAsPoint": None,
        "shaping": None,
        "world": mat(xf_cache.GetLocalToWorldTransform(prim)),
    }
    if t == "SphereLight":
        s = UsdLux.SphereLight(prim)
        e["radius"] = val(s.GetRadiusAttr())
        e["treatAsPoint"] = val(s.GetTreatAsPointAttr())
    elif t == "DiskLight":
        e["radius"] = val(UsdLux.DiskLight(prim).GetRadiusAttr())
    elif t == "RectLight":
        r = UsdLux.RectLight(prim)
        e["width"] = val(r.GetWidthAttr())
        e["height"] = val(r.GetHeightAttr())
    elif t == "DistantLight":
        e["angle"] = val(UsdLux.DistantLight(prim).GetAngleAttr())
    elif t == "CylinderLight":
        c = UsdLux.CylinderLight(prim)
        e["radius"] = val(c.GetRadiusAttr())
        e["length"] = val(c.GetLengthAttr())
    if prim.HasAPI(UsdLux.ShapingAPI):
        sh = UsdLux.ShapingAPI(prim)
        e["shaping"] = {
            "coneAngle": val(sh.GetShapingConeAngleAttr()),
            "coneSoftness": val(sh.GetShapingConeSoftnessAttr()),
            "focus": val(sh.GetShapingFocusAttr()),
            "focusTint": val(sh.GetShapingFocusTintAttr()),
        }
    return e


PS_INPUTS = (
    "diffuseColor", "emissiveColor", "useSpecularWorkflow", "specularColor", "metallic", "roughness",
    "clearcoat", "clearcoatRoughness", "opacity", "opacityThreshold", "ior", "normal", "occlusion",
    "specular",  # Blender's non-standard input (scene-io §5.3)
)
PS_DEFAULTS: dict[str, Any] = {
    "diffuseColor": [0.18, 0.18, 0.18], "emissiveColor": [0.0, 0.0, 0.0], "useSpecularWorkflow": 0.0,
    "specularColor": [0.0, 0.0, 0.0], "metallic": 0.0, "roughness": 0.5, "clearcoat": 0.0,
    "clearcoatRoughness": 0.01, "opacity": 1.0, "opacityThreshold": 0.0, "ior": 1.5,
    "normal": [0.0, 0.0, 1.0], "occlusion": 1.0,
}


def dump_material(mat_prim: Usd.Prim) -> dict[str, Any]:
    m = UsdShade.Material(mat_prim)
    out: dict[str, Any] = {"path": str(mat_prim.GetPath()), "shader": None, "inputs": {}, "authored": []}
    src = m.ComputeSurfaceSource()
    shader = src[0] if src and src[0] else None
    if not shader:
        return out
    out["shader"] = str(shader.GetIdAttr().Get() or "")
    for name in PS_INPUTS:
        inp = shader.GetInput(name)
        if inp and inp.GetAttr().HasAuthoredValue():
            out["inputs"][name] = val(inp.GetAttr())
            out["authored"].append(name)
        elif inp and inp.HasConnectedSource():
            out["inputs"][name] = "<connected>"
            out["authored"].append(name)
        elif name in PS_DEFAULTS:
            out["inputs"][name] = PS_DEFAULTS[name]
    return out


def bound_material(prim: Usd.Prim) -> str | None:
    m, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    return str(m.GetPath()) if m else None


def dump_mesh(prim: Usd.Prim, xf_cache: UsdGeom.XformCache, materials: dict[str, Any]) -> dict[str, Any]:
    mesh = UsdGeom.Mesh(prim)
    counts = list(mesh.GetFaceVertexCountsAttr().Get(T) or [])
    points = mesh.GetPointsAttr().Get(T) or []
    tri_per_face = [max(0, int(n) - 2) for n in counts]
    base = bound_material(prim)
    tris_by_mat: dict[str, int] = {}
    covered = [False] * len(counts)
    subsets = UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets()
    for s in subsets:
        idx = list(s.GetIndicesAttr().Get(T) or [])
        sm = bound_material(s.GetPrim()) or base or "<none>"
        for f in idx:
            if 0 <= f < len(counts) and not covered[f]:
                covered[f] = True
                tris_by_mat[sm] = tris_by_mat.get(sm, 0) + tri_per_face[f]
    rest = sum(t for t, c in zip(tri_per_face, covered) if not c)
    if rest:
        key = base or "<none>"
        tris_by_mat[key] = tris_by_mat.get(key, 0) + rest
    for mp in set(tris_by_mat) | ({base} if base else set()):
        if mp and mp != "<none>" and mp not in materials:
            materials[mp] = dump_material(prim.GetStage().GetPrimAtPath(mp))
    return {
        "path": str(prim.GetPath()),
        "points": len(points),
        "bbox": ([min(p[k] for p in points) for k in range(3)] + [max(p[k] for p in points) for k in range(3)])
        if len(points) else None,
        "faces": len(counts),
        "triangles": sum(tri_per_face),
        "material": base,
        "trianglesByMaterial": tris_by_mat,
        "geomSubsets": len(subsets),
        "subdivisionScheme": str(mesh.GetSubdivisionSchemeAttr().Get(T)),
        "orientation": str(mesh.GetOrientationAttr().Get(T)),
        "doubleSided": bool(mesh.GetDoubleSidedAttr().Get(T)),
        "world": mat(xf_cache.GetLocalToWorldTransform(prim)),
    }


def dump_file(path: Path) -> dict[str, Any]:
    stage = Usd.Stage.Open(str(path))
    xf_cache = UsdGeom.XformCache(T)
    root = stage.GetRootLayer()
    out: dict[str, Any] = {
        "source": "pxr",
        "file": path.name,
        "ok": True,
        "stage": {
            "upAxis": str(UsdGeom.GetStageUpAxis(stage)),
            "metersPerUnit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "doc": root.documentation,
        },
        "draws": [],
        "lights": [],
        "materials": [],
        "pointInstancers": [],
        "cameras": [],
    }
    materials: dict[str, Any] = {}
    pred = Usd.TraverseInstanceProxies(Usd.PrimIsActive & Usd.PrimIsDefined & ~Usd.PrimIsAbstract)
    it = iter(Usd.PrimRange(stage.GetPseudoRoot(), pred))
    for prim in it:
        img = UsdGeom.Imageable(prim)
        if img and img.ComputeVisibility(T) == UsdGeom.Tokens.invisible:
            it.PruneChildren()
            continue
        if img and img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            it.PruneChildren()
            continue
        if prim.IsA(UsdGeom.PointInstancer):
            pi = UsdGeom.PointInstancer(prim)
            protos = [str(p) for p in pi.GetPrototypesRel().GetTargets()]
            proto_idx = list(pi.GetProtoIndicesAttr().Get(T) or [])
            # World = instance-relative transform * instancer local-to-world (row-vector convention).
            w = xf_cache.GetLocalToWorldTransform(prim)
            xfs = pi.ComputeInstanceTransformsAtTime(T, T)
            out["pointInstancers"].append({
                "path": str(prim.GetPath()),
                "world": mat(w),
                "prototypes": protos,
                "instances": [
                    {"proto": protos[pidx] if 0 <= pidx < len(protos) else None, "world": mat(Gf.Matrix4d(x) * w)}
                    for pidx, x in zip(proto_idx, xfs)
                ],
            })
            it.PruneChildren()  # prototypes are imaged only through the instancer
            continue
        if prim.IsA(UsdGeom.Mesh):
            out["draws"].append(dump_mesh(prim, xf_cache, materials))
        elif prim.HasAPI(UsdLux.LightAPI) or prim.GetTypeName() in LIGHT_TYPES:
            out["lights"].append(dump_light(prim, xf_cache))
        elif prim.IsA(UsdGeom.Camera):
            c = UsdGeom.Camera(prim)
            out["cameras"].append({
                "path": str(prim.GetPath()),
                "focalLength": val(c.GetFocalLengthAttr()),
                "horizontalAperture": val(c.GetHorizontalApertureAttr()),
                "verticalAperture": val(c.GetVerticalApertureAttr()),
                "world": mat(xf_cache.GetLocalToWorldTransform(prim)),
            })
    # PointInstancer prototypes still need their materials in the table.
    for pi in out["pointInstancers"]:
        for pp in pi["prototypes"]:
            prim = stage.GetPrimAtPath(pp)
            for p in Usd.PrimRange(prim):
                if p.IsA(UsdGeom.Mesh):
                    mp = bound_material(p)
                    if mp and mp not in materials:
                        materials[mp] = dump_material(stage.GetPrimAtPath(mp))
    out["materials"] = sorted(materials.values(), key=lambda m: m["path"])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("validation/out/usd-spike"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.files:
        d = dump_file(f)
        dst = args.out / f"pxr.{f.name}.json"
        dst.write_text(json.dumps(d, indent=1))
        print(f"[pxr_dump] {f.name}: {len(d['draws'])} draws, {len(d['lights'])} lights, "
              f"{len(d['materials'])} materials, {len(d['pointInstancers'])} instancers -> {dst}")


if __name__ == "__main__":
    main()
