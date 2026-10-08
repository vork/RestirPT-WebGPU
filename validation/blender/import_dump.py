"""(vii-L) Blender 5.2.2's stock glTF importer as the reference for our glTF loader (docs/decisions/m7-api.md §3.2).

  Blender -b --factory-startup --python-exit-code 1 -P validation/blender/import_dump.py -- --file A.glb --out D.json [--shading NORMALS|FLAT]

bpy.ops.import_scene.gltf (export_import_convert_lighting_mode SPEC, import_shading as given), then a JSON dump in the
renderer's canonical frame (glTF: x, y, z = Blender x, z, −y; C^T applied): total triangles (evaluated meshes, loop
triangles of non-zero area), triangles per material, the world bbox of every mesh vertex, per material the Principled BSDF inputs
(value or "linked", the Normal Map strength), and per light type, energy (W), colour, spot size / blend and the world
matrix (column-major). No render, no GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def canon(v):
    return [v[0], v[2], -v[1]]


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shading", default="NORMALS")
    a = ap.parse_args(argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    res = bpy.ops.import_scene.gltf(filepath=a.file, export_import_convert_lighting_mode="SPEC", import_shading=a.shading)
    if "FINISHED" not in res:
        raise SystemExit(f"import failed: {res}")
    dg = bpy.context.evaluated_depsgraph_get()
    tris, by_mat = 0, {}
    mn, mx = [float("inf")] * 3, [float("-inf")] * 3
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        me.calc_loop_triangles()
        for t in me.loop_triangles:
            if not t.area > 0.0:      # zero-area triangles: our loader drops them (dense primIds); they render nothing
                continue
            tris += 1
            m = o.material_slots[t.material_index].material if o.material_slots else None
            k = m.name if m else "<none>"
            by_mat[k] = by_mat.get(k, 0) + 1
        M = o.matrix_world
        for v in me.vertices:
            p = canon(M @ v.co)
            for k in range(3):
                mn[k] = min(mn[k], p[k]); mx[k] = max(mx[k], p[k])
        oe.to_mesh_clear()
    mats = []
    for m in bpy.data.materials:
        if not m.node_tree:
            continue
        ps = next((n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
        if ps is None:
            continue
        def inp(name):
            i = ps.inputs[name]
            v = i.default_value
            return {"linked": i.is_linked, "value": list(v) if hasattr(v, "__len__") else float(v)}
        nm = None
        if ps.inputs["Normal"].is_linked:
            src = ps.inputs["Normal"].links[0].from_node
            if src.bl_idname == "ShaderNodeNormalMap":
                nm = {"strength": float(src.inputs["Strength"].default_value), "space": src.space}
        mats.append({"name": m.name, "baseColor": inp("Base Color"), "metallic": inp("Metallic"), "roughness": inp("Roughness"), "ior": inp("IOR"),
                     "transmission": inp("Transmission Weight"), "emissionColor": inp("Emission Color"), "emissionStrength": inp("Emission Strength"),
                     "alpha": inp("Alpha"), "normalMap": nm, "blend_method": getattr(m, "blend_method", None)})
    lights = []
    for o in bpy.data.objects:
        if o.type != "LIGHT":
            continue
        L = o.data
        M = o.matrix_world
        # column-major canonical world matrix: columns = canonical images of the local axes, translation
        cols = [canon(M.col[i][:3]) for i in range(3)] + [canon(M.col[3][:3])]
        lights.append({"name": o.name, "type": L.type, "energy": float(L.energy), "color": list(L.color),
                       "spotSize": float(getattr(L, "spot_size", 0.0)), "spotBlend": float(getattr(L, "spot_blend", 0.0)),
                       "shadowSoftSize": float(getattr(L, "shadow_soft_size", 0.0)),
                       "matrix": [c for col in cols for c in (col + [0.0])][:12] + cols[3] + [1.0]})
    out = {"file": a.file, "shading": a.shading, "blender": bpy.app.version_string, "triangles": tris, "trianglesByMaterial": by_mat,
           "bbox": mn + mx, "materials": mats, "lights": lights}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"[import_dump] {a.file}: {tris} triangles, {len(mats)} materials, {len(lights)} lights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
