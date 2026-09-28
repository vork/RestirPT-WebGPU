"""Build the LightUSD spike scene in headless Blender and export it as .usda/.usdc/.usdz.

Run:
  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
      --python validation/spikes/usd/make_usd.py -- --out validation/assets/usd-spike

Scene: floor plane + two boxes (Principled base color/roughness/metallic; one box carries two
materials -> GeomSubset), a linked-duplicate box and a collection instance (use_instancing=True),
and one light of each Blender type the plan cares about (plan §1.2, scene-io §6).
Also writes blender_truth.json with the Blender-side light/object parameters for the comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("validation/assets/usd-spike"))
    p.add_argument("--name", default="spike_blender")
    p.add_argument("--perf-out", type=Path, default=None,
                   help="also write perf.usdc (~655k triangles; use a gitignored dir) for load timings")
    return p.parse_args(argv)


def principled(name: str, base: tuple[float, float, float], rough: float, metal: float) -> bpy.types.Material:
    m = bpy.data.materials.new(name)
    if m.node_tree is None:  # Blender 5.x materials always have a node tree
        m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    return m


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def add_light(
    name: str,
    kind: str,
    loc: tuple[float, float, float],
    rot_deg: tuple[float, float, float],
    energy: float,
    color: tuple[float, float, float],
    **props: object,
) -> bpy.types.Object:
    data = bpy.data.lights.new(name, kind)
    data.energy = energy
    data.color = color
    for k, v in props.items():
        setattr(data, k, v)
    ob = bpy.data.objects.new(name, data)
    ob.location = loc
    ob.rotation_euler = Euler([math.radians(a) for a in rot_deg], "XYZ")
    bpy.context.scene.collection.objects.link(ob)
    return ob


def build() -> None:
    clear_scene()
    scene = bpy.context.scene
    col = scene.collection

    m_floor = principled("M_Floor", (0.8, 0.8, 0.8), 0.5, 0.0)
    m_red = principled("M_Red", (0.8, 0.1, 0.1), 0.3, 0.0)
    m_gold = principled("M_Gold", (1.0, 0.766, 0.336), 0.2, 1.0)
    m_blue = principled("M_Blue", (0.1, 0.2, 0.8), 0.7, 0.25)

    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.data.materials.append(m_floor)

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-1.0, 0.0, 0.5))
    box_a = bpy.context.active_object
    box_a.name = "BoxA"
    box_a.data.name = "BoxMesh"
    box_a.data.materials.append(m_red)
    box_a.data.materials.append(m_gold)
    # Top face (+Z normal) gets material slot 1 -> Blender writes a materialBind GeomSubset.
    for poly in box_a.data.polygons:
        poly.material_index = 1 if poly.normal.z > 0.5 else 0

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(1.2, 0.3, 0.75))
    box_b = bpy.context.active_object
    box_b.name = "BoxB"
    box_b.rotation_euler = Euler((0.0, 0.0, math.radians(30.0)), "XYZ")
    box_b.scale = (0.8, 1.2, 1.5)
    box_b.data.materials.append(m_blue)

    # Linked duplicate (shares BoxMesh).
    box_c = bpy.data.objects.new("BoxA_Linked", box_a.data)
    box_c.location = (-1.0, 2.0, 0.5)
    box_c.rotation_euler = Euler((0.0, 0.0, math.radians(45.0)), "XYZ")
    col.objects.link(box_c)

    # Collection instance -> dupli instance (what use_instancing actually targets).
    proto_col = bpy.data.collections.new("ProtoCol")
    col.children.link(proto_col)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.3, location=(0, 0, 0))
    ico = bpy.context.active_object
    ico.name = "ProtoIco"
    for c in list(ico.users_collection):
        c.objects.unlink(ico)
    proto_col.objects.link(ico)
    ico.data.materials.append(m_gold)
    proto_col.hide_render = False
    view_layer_col = bpy.context.view_layer.layer_collection.children["ProtoCol"]
    view_layer_col.exclude = True  # prototype itself not rendered, only the instances
    for i, pos in enumerate([(2.5, -1.5, 0.3), (2.5, -0.5, 0.3)]):
        inst = bpy.data.objects.new(f"IcoInstance{i}", None)
        inst.instance_type = "COLLECTION"
        inst.instance_collection = proto_col
        inst.location = pos
        col.objects.link(inst)

    add_light("PointR0", "POINT", (0.0, -2.0, 3.0), (0, 0, 0), 100.0, (1.0, 0.9, 0.8), shadow_soft_size=0.0)
    add_light("PointR025", "POINT", (-2.5, 1.0, 2.0), (0, 0, 0), 50.0, (0.8, 0.9, 1.0),
              shadow_soft_size=0.25, exposure=1.0)
    add_light("Spot60", "SPOT", (2.0, -2.0, 3.5), (35.0, 0.0, 45.0), 300.0, (1.0, 1.0, 1.0),
              shadow_soft_size=0.0, spot_size=math.radians(60.0), spot_blend=0.15)
    add_light("AreaRect", "AREA", (0.0, 1.5, 2.5), (-20.0, 10.0, 0.0), 80.0, (1.0, 0.95, 0.9),
              shape="RECTANGLE", size=1.0, size_y=0.5)
    add_light("AreaDisk", "AREA", (-1.5, -1.5, 2.2), (15.0, -25.0, 0.0), 60.0, (0.9, 1.0, 0.9),
              shape="DISK", size=0.8, normalize=False)
    add_light("Sun", "SUN", (0.0, 0.0, 10.0), (40.0, 0.0, 120.0), 3.0, (1.0, 0.98, 0.95),
              angle=math.radians(2.0))

    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    cam.location = (6.0, -6.0, 4.0)
    cam.rotation_euler = Euler((math.radians(65), 0, math.radians(45)), "XYZ")
    col.objects.link(cam)
    scene.camera = cam


def truth() -> dict:
    """Blender-side parameters, in Blender world (Z-up) coordinates."""
    out: dict = {"lights": {}, "objects": {}}
    bpy.context.view_layer.update()
    for ob in bpy.context.scene.objects:
        mw = [list(r) for r in ob.matrix_world]
        if ob.type == "LIGHT":
            d = ob.data
            e = {
                "type": d.type,
                "energy": d.energy,
                "color": list(d.color),
                "exposure": d.exposure,
                "normalize": d.normalize,
                "use_temperature": d.use_temperature,
                "temperature": d.temperature,
                "matrix_world": mw,
            }
            if d.type in ("POINT", "SPOT"):
                e["radius"] = d.shadow_soft_size
            if d.type == "SPOT":
                e["spot_size_deg"] = math.degrees(d.spot_size)
                e["spot_blend"] = d.spot_blend
            if d.type == "AREA":
                e.update(shape=d.shape, size=d.size, size_y=d.size_y, spread_deg=math.degrees(d.spread))
            if d.type == "SUN":
                e["angle_deg"] = math.degrees(d.angle)
            out["lights"][ob.name] = e
        else:
            out["objects"][ob.name] = {"type": ob.type, "matrix_world": mw}
    return out


def export(out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    common = dict(
        export_lights=True,
        export_materials=True,
        generate_preview_surface=True,
        use_instancing=True,
        export_cameras=True,
        convert_world_material=False,
        evaluation_mode="RENDER",
    )
    for ext in ("usda", "usdc", "usdz"):
        bpy.ops.wm.usd_export(filepath=str(out_dir / f"{name}.{ext}"), **common)
    # Y-up variant (convert_orientation) to exercise upAxis handling on Blender output.
    bpy.ops.wm.usd_export(filepath=str(out_dir / f"{name}_yup.usda"), convert_orientation=True,
                          export_global_forward_selection="NEGATIVE_Z", export_global_up_selection="Y",
                          **common)
    (out_dir / f"{name}.truth.json").write_text(json.dumps(truth(), indent=1))


def convert_hand(out_dir: Path) -> None:
    """spike_hand.usda (hand-authored, checked in) -> .usdc and .usdz via Blender's bundled pxr."""
    from pxr import Sdf, UsdUtils

    src = out_dir / "spike_hand.usda"
    if not src.exists():
        return
    Sdf.Layer.FindOrOpen(str(src)).Export(str(out_dir / "spike_hand.usdc"))
    UsdUtils.CreateNewUsdzPackage(str(out_dir / "spike_hand.usdc"), str(out_dir / "spike_hand.usdz"))


def export_perf(out_dir: Path) -> None:
    """~655k-triangle scene (8 ico spheres, 2 materials, 1 rect light) for parse/transfer timings."""
    clear_scene()
    mats = [principled("M_A", (0.8, 0.2, 0.2), 0.4, 0.0), principled("M_B", (0.2, 0.2, 0.8), 0.2, 1.0)]
    for i in range(8):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=7, radius=0.5, location=(i % 4 * 1.2, i // 4 * 1.2, 0.5))
        bpy.context.active_object.data.materials.append(mats[i % 2])
    add_light("Key", "AREA", (0, 0, 4), (0, 0, 0), 200.0, (1, 1, 1), shape="RECTANGLE", size=2.0, size_y=1.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.usd_export(filepath=str(out_dir / "perf.usdc"), export_lights=True, export_materials=True)


def main() -> None:
    args = parse_args()
    build()
    export(args.out.resolve(), args.name)
    convert_hand(args.out.resolve())
    if args.perf_out:
        export_perf(args.perf_out.resolve())
    print(f"[make_usd] wrote {args.name}.* to {args.out.resolve()}")


if __name__ == "__main__":
    main()
