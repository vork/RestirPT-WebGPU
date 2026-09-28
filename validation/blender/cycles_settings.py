"""Table-driven Cycles settings for reference renders (plan §7.5, cyc §1/§5/§7, cyc-verify §5).

`apply_settings(scene, cfg)` sets every property from explicit tables, then reads every one back and
returns a manifest {rna path: value}. Any mismatch raises `SettingsError`. RNA names were checked by
introspection in Blender 5.1.2 (factory startup).

cfg keys (all optional except where noted; see DEFAULT_CFG):
  spp, max_bounces, seed, resolution (W, H), device ('GPU' = Metal only, or 'CPU'),
  min_light_bounces (0 keeps Cycles RR), emission_sampling ('FRONT_BACK' | 'NONE' ...),
  light_mis (per-light MIS: False = plan Mode A, True = Mode B),
  light_defaults {visible_camera, exposure, spread}, lights {object name: overrides},
  camera {vfov_deg | vfov_rad, clip_start, clip_end} or None (then VERTICAL fit is only asserted),
  world None (black) or a world dict (see _world_rows), output_path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import bpy

WORKING_SPACE = "Linear Rec.709"

DEFAULT_CFG: dict[str, Any] = {
    "spp": 128,
    "max_bounces": 3,
    "seed": 0,
    "resolution": (512, 512),
    "device": "GPU",
    "min_light_bounces": 0,
    "emission_sampling": "FRONT_BACK",
    "light_mis": False,
    "light_defaults": {"visible_camera": False, "exposure": 0.0, "spread": math.pi},
    "lights": {},
    "camera": None,
    "world": None,
    "output_path": None,
}


class SettingsError(RuntimeError):
    pass


@dataclass
class Row:
    """One table entry: `owner.attr = value` (or only asserted when `assign` is False)."""

    owner: Any
    path: str  # manifest prefix, e.g. 'scene.cycles'
    attr: str
    value: Any
    assign: bool = True

    @property
    def key(self) -> str:
        return f"{self.path}.{self.attr}"


def _q(name: str) -> str:
    return '["' + name.replace('"', '\\"') + '"]'


def _plain(v: Any) -> Any:
    """RNA value -> JSON-friendly python value."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return float(v)
    if hasattr(v, "name") and hasattr(v, "bl_rna"):  # ID pointer
        return v.name
    try:
        return [_plain(x) for x in v]
    except TypeError:
        return repr(v)


def _equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        if isinstance(expected, bool) or isinstance(actual, bool):
            return expected == actual
        return math.isclose(float(expected), float(actual), rel_tol=1e-5, abs_tol=1e-9)
    if isinstance(expected, (list, tuple)):
        a = list(actual)
        return len(a) == len(expected) and all(_equal(e, x) for e, x in zip(expected, a))
    if expected is None:
        return actual is None
    return expected == actual


def run_rows(rows: list[Row]) -> dict[str, Any]:
    """Assign all rows in order, then read all back (later writes may clobber earlier ones)."""
    for r in rows:
        if r.assign:
            try:
                setattr(r.owner, r.attr, r.value)
            except (AttributeError, TypeError, ValueError) as e:
                raise SettingsError(f"cannot set {r.key} = {r.value!r}: {e}") from e
    manifest: dict[str, Any] = {}
    bad: list[str] = []
    for r in rows:
        actual = getattr(r.owner, r.attr)
        if not _equal(r.value, actual):
            bad.append(f"{r.key}: expected {r.value!r}, read back {_plain(actual)!r}")
        manifest[r.key] = _plain(actual)
    if bad:
        raise SettingsError("read-back mismatch:\n  " + "\n  ".join(bad))
    return manifest


# --- device -------------------------------------------------------------------------------------


def enable_metal() -> dict[str, Any]:
    """Enable only METAL devices. --factory-startup prefs are not saved, so call this every run."""
    cp = bpy.context.preferences.addons["cycles"].preferences
    cp.compute_device_type = "METAL"
    cp.refresh_devices()
    used: list[str] = []
    for d in cp.devices:
        d.use = d.type == "METAL"
        if d.use:
            used.append(d.name)
    if not used or cp.compute_device_type != "METAL" or not cp.has_active_device():
        raise SettingsError(f"no METAL device available: {[(d.name, d.type) for d in cp.devices]}")
    p = "preferences.addons['cycles'].preferences"
    m: dict[str, Any] = {f"{p}.compute_device_type": cp.compute_device_type}
    for d in cp.devices:
        m[f"{p}.devices{_q(d.name)}.use"] = bool(d.use)
    return m


# --- tables -------------------------------------------------------------------------------------


def _sampling_rows(scene: bpy.types.Scene, cfg: dict[str, Any]) -> list[Row]:
    c, b = scene.cycles, int(cfg["max_bounces"])
    table: list[tuple[str, Any]] = [
        ("device", cfg["device"]),
        ("samples", int(cfg["spp"])),
        ("time_limit", 0.0),
        ("use_adaptive_sampling", False),
        ("use_denoising", False),
        ("use_preview_denoising", False),
        ("sample_clamp_direct", 0.0),
        ("sample_clamp_indirect", 0.0),
        ("blur_glossy", 0.0),
        ("light_sampling_threshold", 0.0),
        ("caustics_reflective", True),
        ("caustics_refractive", True),
        ("use_fast_gi", False),
        ("use_guiding", False),
        ("max_bounces", b),
        ("diffuse_bounces", b),
        ("glossy_bounces", b),
        ("transmission_bounces", b),
        ("volume_bounces", 0),
        ("transparent_max_bounces", 1024),  # always: Mode-B light pass-throughs consume these
        ("min_light_bounces", int(cfg["min_light_bounces"])),
        ("min_transparent_bounces", 0),
        ("pixel_filter_type", "BOX"),  # Blender forces width 1.0 for BOX
        ("filter_width", 1.0),
        ("sampling_pattern", "TABULATED_SOBOL"),  # explicit: new scenes default to AUTOMATIC (C7)
        ("scrambling_distance", 1.0),
        ("auto_scrambling_distance", False),
        ("seed", int(cfg["seed"])),
        ("use_animated_seed", False),
        ("film_exposure", 1.0),
        ("use_light_tree", True),
        ("texture_limit_render", "OFF"),
        ("use_sample_subset", False),
        ("shading_system", False),  # SVM, not OSL
        ("direct_light_sampling_type", "MULTIPLE_IMPORTANCE_SAMPLING"),
        ("use_layer_samples", "USE"),
    ]
    return [Row(c, "scene.cycles", k, v) for k, v in table]


def _render_rows(scene: bpy.types.Scene, cfg: dict[str, Any]) -> list[Row]:
    r = scene.render
    w, h = cfg["resolution"]
    table: list[tuple[str, Any]] = [
        ("engine", "CYCLES"),
        ("resolution_x", int(w)),
        ("resolution_y", int(h)),
        ("resolution_percentage", 100),
        ("pixel_aspect_x", 1.0),
        ("pixel_aspect_y", 1.0),
        ("use_border", False),
        ("film_transparent", False),
        ("use_motion_blur", False),
        ("use_compositing", False),
        ("use_sequencer", False),
        ("dither_intensity", 0.0),
        ("use_simplify", False),
    ]
    rows = [Row(r, "scene.render", k, v) for k, v in table]
    if cfg.get("output_path"):
        rows.append(Row(r, "scene.render", "filepath", str(cfg["output_path"])))
    for vl in scene.view_layers:
        p = f"scene.view_layers{_q(vl.name)}"
        rows += [Row(vl, p, "samples", 0), Row(vl.cycles, p + ".cycles", "use_denoising", False)]
    return rows


def _output_rows(scene: bpy.types.Scene) -> list[Row]:
    im, vs = scene.render.image_settings, scene.view_settings
    p = "scene.render.image_settings"
    return [
        Row(im, p, "media_type", "IMAGE"),
        Row(im, p, "file_format", "OPEN_EXR"),  # must precede depth/codec
        Row(im, p, "color_depth", "32"),
        Row(im, p, "exr_codec", "ZIP"),
        Row(im, p, "color_mode", "RGB"),
        Row(im, p, "color_management", "FOLLOW_SCENE"),
        Row(im.linear_colorspace_settings, p + ".linear_colorspace_settings", "name", WORKING_SPACE, False),
        Row(vs, "scene.view_settings", "view_transform", "Standard"),
        Row(vs, "scene.view_settings", "look", "None"),
        Row(vs, "scene.view_settings", "exposure", 0.0),
        Row(vs, "scene.view_settings", "gamma", 1.0),
        Row(bpy.data.colorspace, "bpy.data.colorspace", "working_space", WORKING_SPACE, False),
    ]


def camera_rows(cam_obj: bpy.types.Object, vfov_rad: float, clip_start: float = 1e-4, clip_end: float = 1e5) -> list[Row]:
    """Pinhole camera with exact vertical FOV (never angle_x; cyc-verify C1)."""
    cam = cam_obj.data
    p = f"objects{_q(cam_obj.name)}.data"
    return [
        Row(cam, p, "type", "PERSP"),
        Row(cam, p, "sensor_fit", "VERTICAL"),
        Row(cam, p, "angle_y", float(vfov_rad)),
        Row(cam, p, "clip_start", float(clip_start)),
        Row(cam, p, "clip_end", float(clip_end)),
        Row(cam, p, "shift_x", 0.0),
        Row(cam, p, "shift_y", 0.0),
        Row(cam.dof, p + ".dof", "use_dof", False),
    ]


def apply_camera(cam_obj: bpy.types.Object, vfov_rad: float, clip_start: float = 1e-4, clip_end: float = 1e5) -> dict[str, Any]:
    return run_rows(camera_rows(cam_obj, vfov_rad, clip_start, clip_end))


def light_rows(obj: bpy.types.Object, spec: dict[str, Any]) -> list[Row]:
    """Light defaults (plan §7.5 Lights). `spec`: mis, visible_camera, exposure, spread, [energy, color,
    spot_size, spot_blend, shape, size, size_y]. Object scale must already be 1 (it scales area/cones)."""
    L = obj.data
    p = f"objects{_q(obj.name)}"
    rows = [
        Row(obj, p, "scale", (1.0, 1.0, 1.0), False),
        Row(obj, p, "visible_camera", bool(spec["visible_camera"])),
        Row(L, p + ".data", "normalize", True),
        Row(L, p + ".data", "exposure", float(spec.get("exposure", 0.0))),
        Row(L, p + ".data", "use_temperature", False),
        Row(L, p + ".data", "use_shadow", True),
        Row(L.cycles, p + ".data.cycles", "use_multiple_importance_sampling", bool(spec["mis"])),
        Row(L.cycles, p + ".data.cycles", "is_caustics_light", False),
        Row(L.cycles, p + ".data.cycles", "is_portal", False),
        Row(L.cycles, p + ".data.cycles", "max_bounces", 1024),
    ]
    for k in ("energy", "color"):
        if k in spec:
            rows.append(Row(L, p + ".data", k, spec[k]))
    if L.type in ("POINT", "SPOT"):
        rows += [Row(L, p + ".data", "shadow_soft_size", 0.0), Row(L, p + ".data", "use_soft_falloff", False)]
    if L.type == "SPOT":
        rows += [Row(L, p + ".data", k, spec[k]) for k in ("spot_size", "spot_blend") if k in spec]
    if L.type == "SUN":
        rows.append(Row(L, p + ".data", "angle", 0.0))
    if L.type == "AREA":
        rows.append(Row(L, p + ".data", "spread", float(spec.get("spread", math.pi))))
        rows += [Row(L, p + ".data", k, spec[k]) for k in ("shape", "size", "size_y") if k in spec]
    if L.node_tree is not None:  # new lights get Emission (1,1,1) x 1; anything else is unsupported
        for n in L.node_tree.nodes:
            if n.bl_idname == "ShaderNodeEmission":
                np = f"{p}.data.node_tree.nodes{_q(n.name)}"
                rows.append(Row(n.inputs["Strength"], np + '.inputs["Strength"]', "default_value", 1.0, False))
                rows.append(Row(n.inputs["Color"], np + '.inputs["Color"]', "default_value", (1.0, 1.0, 1.0, 1.0), False))
    return rows


def _iter_node_trees() -> list[tuple[str, bpy.types.NodeTree]]:
    out: list[tuple[str, bpy.types.NodeTree]] = []
    for m in bpy.data.materials:
        if m.node_tree is not None:
            out.append((f"materials{_q(m.name)}.node_tree", m.node_tree))
    for w in bpy.data.worlds:
        if w.node_tree is not None:
            out.append((f"worlds{_q(w.name)}.node_tree", w.node_tree))
    for g in bpy.data.node_groups:
        if g.bl_idname == "ShaderNodeTree":
            out.append((f"node_groups{_q(g.name)}", g))
    return out


def material_object_rows(scene: bpy.types.Scene, emission_sampling: str) -> list[Row]:
    """GGX everywhere, no bump correction, fixed emission sampling, Linear images, no terminator
    offsets, no MNEE (plan §7.5 Materials and objects)."""
    rows: list[Row] = []
    for m in bpy.data.materials:
        p = f"materials{_q(m.name)}.cycles"
        rows += [Row(m.cycles, p, "use_bump_map_correction", False), Row(m.cycles, p, "emission_sampling", emission_sampling)]
    for tp, tree in _iter_node_trees():
        for n in tree.nodes:
            np = f"{tp}.nodes{_q(n.name)}"
            if "distribution" in n.bl_rna.properties:
                rows.append(Row(n, np, "distribution", "GGX"))
            if n.bl_idname in ("ShaderNodeTexImage", "ShaderNodeTexEnvironment"):
                rows.append(Row(n, np, "interpolation", "Linear"))
    for o in scene.objects:
        if o.type in ("MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES", "POINTCLOUD"):
            p = f"objects{_q(o.name)}"
            rows += [
                Row(o, p, "shadow_terminator_geometry_offset", 0.0),
                Row(o, p, "shadow_terminator_shading_offset", 0.0),
                Row(o.cycles, p + ".cycles", "is_caustics_caster", False),
                Row(o.cycles, p + ".cycles", "is_caustics_receiver", False),
            ]
    return rows


def build_world(scene: bpy.types.Scene, wcfg: dict[str, Any]) -> bpy.types.World:
    """Minimal world graph (M2 owns the final one). wcfg: {'color', 'strength'} constant, or
    {'hdri': path, 'rotation_z', 'strength', 'tint'} equirect; Linear interpolation."""
    w = bpy.data.worlds.new("reference_world")
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = float(wcfg.get("strength", 1.0))
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    if "hdri" in wcfg:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.vector_type = "POINT"
        mp.inputs["Rotation"].default_value = (0.0, 0.0, float(wcfg.get("rotation_z", 0.0)))
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(str(wcfg["hdri"]), check_existing=True)
        env.image.colorspace_settings.name = WORKING_SPACE
        env.projection = "EQUIRECTANGULAR"
        env.interpolation = "Linear"
        tint = nt.nodes.new("ShaderNodeVectorMath")
        tint.operation = "MULTIPLY"
        tint.inputs[1].default_value = tuple(wcfg.get("tint", (1.0, 1.0, 1.0)))
        nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], tint.inputs[0])
        nt.links.new(tint.outputs["Vector"], bg.inputs["Color"])
    else:
        c = wcfg.get("color", (1.0, 1.0, 1.0))
        bg.inputs["Color"].default_value = (c[0], c[1], c[2], 1.0)
    scene.world = w
    return w


def _world_rows(scene: bpy.types.Scene, cfg: dict[str, Any]) -> list[Row]:
    wcfg = cfg["world"]
    if wcfg is None:
        return [Row(scene, "scene", "world", None)]  # no world: empty (black) background graph
    w = scene.world if scene.world is not None and wcfg.get("keep_existing") else build_world(scene, wcfg)
    p = f"worlds{_q(w.name)}"
    rows = [
        Row(w.cycles, p + ".cycles", "sampling_method", wcfg.get("sampling_method", "AUTOMATIC")),
        Row(w.cycles, p + ".cycles", "max_bounces", 1024),
        Row(w.cycles, p + ".cycles", "is_caustics_light", False),
    ]
    vis = w.cycles_visibility
    rows.append(Row(vis, p + ".cycles_visibility", "camera", bool(wcfg.get("visible_camera", True))))
    rows += [Row(vis, p + ".cycles_visibility", k, True) for k in ("diffuse", "glossy", "transmission", "shadow", "scatter")]
    return rows


# --- entry point --------------------------------------------------------------------------------


def resolve_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    full = {**DEFAULT_CFG, **cfg}
    full["light_defaults"] = {**DEFAULT_CFG["light_defaults"], **cfg.get("light_defaults", {})}
    return full


def apply_settings(scene: bpy.types.Scene, cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply plan §7.5 to `scene` and return the read-back manifest. Raises SettingsError."""
    cfg = resolve_cfg(cfg)
    manifest: dict[str, Any] = {}
    if cfg["device"] == "GPU":
        manifest.update(enable_metal())
    rows = _sampling_rows(scene, cfg) + _render_rows(scene, cfg) + _output_rows(scene) + _world_rows(scene, cfg)
    cam = cfg.get("camera")
    if scene.camera is None:
        raise SettingsError("scene has no camera")
    if cam is not None:
        vfov = cam["vfov_rad"] if "vfov_rad" in cam else math.radians(cam["vfov_deg"])
        rows += camera_rows(scene.camera, vfov, cam.get("clip_start", 1e-4), cam.get("clip_end", 1e5))
    else:
        rows.append(Row(scene.camera.data, f"objects{_q(scene.camera.name)}.data", "sensor_fit", "VERTICAL", False))
    for o in scene.objects:
        if o.type == "LIGHT":
            spec = {"mis": cfg["light_mis"], **cfg["light_defaults"], **cfg["lights"].get(o.name, {})}
            rows += light_rows(o, spec)
    rows += material_object_rows(scene, cfg["emission_sampling"])
    manifest.update(run_rows(rows))
    _assert_invariants(scene, cfg)
    return manifest


def _assert_invariants(scene: bpy.types.Scene, cfg: dict[str, Any]) -> None:
    # Cycles 5.1.2 runs lights_intersect() only if some light has MIS (kernel_data.integrator.use_light_mis,
    # intersect_closest.h:426), for camera rays too. In Mode A (no MIS light) an area light with
    # visible_camera=True is therefore NOT seen by the camera (verified by render); refuse that combination.
    lights = [o for o in scene.objects if o.type == "LIGHT"]
    if not any(o.data.cycles.use_multiple_importance_sampling for o in lights):
        seen = [o.name for o in lights if o.data.type == "AREA" and o.visible_camera]
        if seen:
            raise SettingsError(f"area lights {seen} have visible_camera=True but no light uses MIS: "
                                "Cycles would not show them to the camera; set visible_camera False (Mode A) or use MIS")
    w = scene.world
    if w is None:
        return
    if w.cycles.max_bounces < int(cfg["max_bounces"]) + 1:
        raise SettingsError("world.cycles.max_bounces < max_bounces + 1 biases env lighting low")
    if w.node_tree is not None:
        banned = [n.name for n in w.node_tree.nodes if n.bl_idname in ("ShaderNodeLightPath", "ShaderNodeLightFalloff")]
        if banned:
            raise SettingsError(f"world has ray-type dependent nodes: {banned}")
