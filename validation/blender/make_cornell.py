"""Procedural Cornell box (plan §5 M0, §7.2 scene (i)); metres, built Z-up in Blender.

Geometry follows the classic Cornell measurements scaled to a 0.555 m cube: open side at -Y, camera
at -Y looking +Y, red wall at -X (left in the image), green at +X, one-sided RECTANGLE area light
1 mm below the ceiling emitting down (-Z). Flat shading everywhere.

Materials: 'diffuse' = V1 (Diffuse BSDF, roughness 0, used for renders and the .blend);
'principled' = Principled(base, metallic 0, roughness 1, Specular IOR Level 0) = glTF
KHR_materials_specular specularFactor 0, used for the GLB/USD exports because the exporters only
translate Principled.

CLI:
  Blender -b --factory-startup -P validation/blender/make_cornell.py -- --out validation/assets/cornell
Writes cornell.glb (Y-up; the area light is dropped by the glTF exporter), cornell.usda/.usdc
(Blender USD default: upAxis Z, metersPerUnit 1, RectLight kept), cornell.blend, cornell.meta.json
(camera and light in the glTF frame, for loaders that lose the light).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cycles_settings as cs  # noqa: E402

L = 0.555  # interior edge (m)
H2 = L / 2
REFLECTANCE = {
    "white": (0.725, 0.71, 0.68),
    "red": (0.63, 0.065, 0.05),
    "green": (0.14, 0.45, 0.091),
}
LIGHT_SIZE = (0.13, 0.105)
LIGHT_POWER_W = 4.0  # L = P/(pi A) ~ 93 W/m^2/sr; white floor under the light ~0.9 at b=7
VFOV_DEG = 39.3
CAMERA_POS = (0.0, -H2 - 0.8, 0.273)


def _mesh_object(name: str, verts: list[tuple[float, float, float]], faces: list[tuple[int, ...]], mat: bpy.types.Material) -> bpy.types.Object:
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.shade_flat()
    me.validate()
    me.update()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    return ob


def _oriented_quad(corners: list[Vector], toward: Vector) -> list[tuple[float, float, float]]:
    """Winding so the face normal points toward `toward` (interior of the box)."""
    n = (corners[1] - corners[0]).cross(corners[2] - corners[0])
    c = sum(corners, Vector()) / 4
    if n.dot(toward - c) < 0:
        corners = corners[::-1]
    return [tuple(v) for v in corners]


def _cuboid(sx: float, sy: float, sz: float) -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]]]:
    """Axis-aligned box with its base at z=0, outward normals."""
    x, y = sx / 2, sy / 2
    v = [(-x, -y, 0), (x, -y, 0), (x, y, 0), (-x, y, 0), (-x, -y, sz), (x, -y, sz), (x, y, sz), (-x, y, sz)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return v, f


def set_material_model(mat: bpy.types.Material, color: tuple[float, float, float], model: str) -> None:
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    rgba = (*color, 1.0)
    if model == "diffuse":
        bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
        bsdf.inputs["Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = 0.0
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    elif model == "principled":
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.distribution = "GGX"
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 1.0
        bsdf.inputs["Specular IOR Level"].default_value = 0.0
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    else:
        raise ValueError(model)
    mat.diffuse_color = rgba  # viewport / fallback exporters


def set_materials(model: str) -> None:
    for name, color in REFLECTANCE.items():
        set_material_model(bpy.data.materials[name], color, model)


def build_cornell(scene: bpy.types.Scene, *, light_power: float = LIGHT_POWER_W, material_model: str = "diffuse", resolution: tuple[int, int] = (512, 512)) -> dict[str, Any]:
    """Build into `scene` (expects an empty scene). Returns a spec for cycles_settings cfg keys
    'camera' and 'lights' plus a glTF-frame description."""
    mats = {}
    for name in REFLECTANCE:
        m = bpy.data.materials.new(name)
        mats[name] = m
    set_materials(material_model)

    col = scene.collection
    inside = Vector((0, 0, H2))
    xs, ys, zs = (-H2, H2), (-H2, H2), (0.0, L)
    walls = {
        "floor": ([(x, y, 0.0) for x, y in ((xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1]))], "white"),
        "ceiling": ([(x, y, L) for x, y in ((xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1]))], "white"),
        "back_wall": ([(x, H2, z) for x, z in ((xs[0], zs[0]), (xs[1], zs[0]), (xs[1], zs[1]), (xs[0], zs[1]))], "white"),
        "left_wall": ([(-H2, y, z) for y, z in ((ys[0], zs[0]), (ys[1], zs[0]), (ys[1], zs[1]), (ys[0], zs[1]))], "red"),
        "right_wall": ([(H2, y, z) for y, z in ((ys[0], zs[0]), (ys[1], zs[0]), (ys[1], zs[1]), (ys[0], zs[1]))], "green"),
    }
    for name, (corners, mname) in walls.items():
        v = _oriented_quad([Vector(c) for c in corners], inside)
        col.objects.link(_mesh_object(name, v, [(0, 1, 2, 3)], mats[mname]))

    # Classic blocks (mm -> m): x_b = (278 - x_c)/1000, y_b = z_c/1000 - L/2.
    for name, size, loc, rot_deg in (
        ("short_box", (0.165, 0.165, 0.165), (0.0925, -0.1085, 0.0), -17.0),
        ("tall_box", (0.165, 0.165, 0.330), (-0.090, 0.0735, 0.0), 17.0),
    ):
        v, f = _cuboid(*size)
        ob = _mesh_object(name, v, f, mats["white"])
        ob.location = loc
        ob.rotation_euler = (0.0, 0.0, math.radians(rot_deg))
        col.objects.link(ob)

    ld = bpy.data.lights.new("ceiling_light", "AREA")
    ld.shape = "RECTANGLE"
    ld.size, ld.size_y = LIGHT_SIZE
    ld.energy = light_power
    ld.spread = math.pi
    ld.normalize = True
    lo = bpy.data.objects.new("ceiling_light", ld)
    lo.location = (0.0, 0.0, L - 1e-3)  # identity rotation: emits toward -Z (down)
    col.objects.link(lo)

    cd = bpy.data.cameras.new("camera")
    co = bpy.data.objects.new("camera", cd)
    co.location = CAMERA_POS
    co.rotation_euler = (math.pi / 2, 0.0, 0.0)  # -Z_cam -> +Y_world, +Y_cam -> +Z_world
    col.objects.link(co)
    scene.camera = co
    cs.apply_camera(co, math.radians(VFOV_DEG))
    scene.render.resolution_x, scene.render.resolution_y = resolution

    light_spec = {"visible_camera": True, "spread": math.pi}
    lo.visible_camera = True
    return {
        "camera": {"vfov_deg": VFOV_DEG, "clip_start": 1e-4, "clip_end": 1e5},
        "lights": {"ceiling_light": light_spec},
        "gltf_frame": _gltf_frame_meta(light_power),
    }


def _to_gltf(v: tuple[float, float, float]) -> list[float]:
    """Blender Z-up -> glTF Y-up (inverse of C = R_x(+90 deg)): (x, y, z) -> (x, z, -y)."""
    return [float(v[0]), float(v[2]), float(-v[1])]


def _gltf_frame_meta(light_power: float) -> dict[str, Any]:
    return {
        "units": "metres, glTF frame (+Y up), un-recentred",
        "camera": {"position": _to_gltf(CAMERA_POS), "forward": [0.0, 0.0, -1.0], "up": [0.0, 1.0, 0.0], "vfov_deg": VFOV_DEG, "sensor_fit": "VERTICAL"},
        "lights": [{
            "name": "ceiling_light", "type": "rect", "power_W": light_power, "color": [1.0, 1.0, 1.0],
            "position": _to_gltf((0.0, 0.0, L - 1e-3)), "normal": [0.0, -1.0, 0.0],
            "size_x": LIGHT_SIZE[0], "size_along_minus_z_gltf": LIGHT_SIZE[1], "spread_deg": 180.0,
            "one_sided": True, "visible_camera": True,
        }],
        "reflectance": REFLECTANCE,
    }


def export_assets(out: Path) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    glb, usda, usdc = out / "cornell.glb", out / "cornell.usda", out / "cornell.usdc"
    bpy.ops.export_scene.gltf(
        filepath=str(glb), export_format="GLB", export_yup=True, export_lights=True, export_cameras=True,
        export_apply=True, export_materials="EXPORT", export_normals=True,
    )
    for p in (usda, usdc):
        bpy.ops.wm.usd_export(
            filepath=str(p), export_lights=True, export_cameras=True, export_materials=True,
            generate_preview_surface=True, convert_world_material=False, export_normals=True,
        )
    return {"glb": str(glb), "usda": str(usda), "usdc": str(usdc)}


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("validation/assets/cornell"))
    ap.add_argument("--light-power", type=float, default=LIGHT_POWER_W)
    ap.add_argument("--resolution", type=int, nargs=2, default=(512, 512))
    args = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    spec = build_cornell(scene, light_power=args.light_power, material_model="principled", resolution=tuple(args.resolution))
    files = export_assets(args.out)

    set_materials("diffuse")  # the .blend keeps the V1 validation materials and §7.5 settings
    # Mode B (per-light MIS on) so the camera-visible light shows; Mode A would hide it (cycles_settings).
    manifest = cs.apply_settings(scene, {"camera": spec["camera"], "lights": spec["lights"], "resolution": tuple(args.resolution), "light_mis": True})
    blend = args.out / "cornell.blend"
    bpy.context.preferences.filepaths.save_version = 0  # no cornell.blend1 backups
    bpy.ops.wm.save_as_mainfile(filepath=str(blend.resolve()), compress=True)
    files["blend"] = str(blend)
    meta = {"generator": "validation/blender/make_cornell.py", "blender": bpy.app.version_string, "files": files,
            **spec["gltf_frame"], "blend_manifest_entries": len(manifest),
            "notes": ["glTF exporter drops AREA lights; use 'lights' above",
                      "GLB/USD materials: Principled roughness 1, metallic 0, Specular IOR Level 0 (glTF specular 0)",
                      "USD exported with Blender defaults: upAxis Z, metersPerUnit 1",
                      "blend saved in Mode B (light MIS on): Cycles 5.1.2 hides camera-visible area lights when no light has MIS"]}
    (args.out / "cornell.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"make_cornell": files}))


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
