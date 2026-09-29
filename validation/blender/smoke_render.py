"""M0 Blender smoke test (plan §5 M0): Cornell on Metal -> 32-bit EXR, timing budget, EXR orientation.

  Blender -b --factory-startup --python-exit-code 1 -P validation/blender/smoke_render.py -- \
      [--out-dir validation/out/m0] [--budget validation/budget.json] [--skip-sponza] [--skip-marker]

Timing model per (scene, max_bounces): T(n) = overhead + n * t_spp, where overhead = scene sync + BVH +
kernel load, measured as a warm 1-spp render (min of repeats). seconds = T(n) - T(1) (n-1 samples),
s_per_4096spp = seconds / (n-1) * 4096. The first render of the process also pays Metal device init and
kernel compile/load; `metal_first_compile_s` = T_first(1 spp) - overhead (depends on the Metal shader cache).

Every render group holds the shared GPU lock (gpu_lock.py; protocol in validation/harness/gpu-lock.ts).
Sponza needs `python3 validation/blender/fetch_sponza.py` first.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cycles_settings as cs  # noqa: E402
import make_cornell  # noqa: E402
from gpu_lock import gpu_lock as _gpu_lock  # noqa: E402
from verify_exr import read_exr, verify_exr  # noqa: E402

def _log(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


@contextlib.contextmanager
def gpu_lock() -> Iterator[None]:
    """Hold the shared GPU lock (gpu_lock.py, holder file smoke_render-<pid>) for one render group."""
    with _gpu_lock("smoke_render", log=_log) as waited:
        if waited > 1:
            _log(f"waited {waited:.0f} s for GPU lock")
        yield


def reset() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)  # also resets prefs: apply_settings re-enables Metal
    return bpy.context.scene


def render(scene: bpy.types.Scene, spp: int) -> float:
    scene.cycles.samples = spp
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=False)
    return time.perf_counter() - t0


def save_exr(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.data.images["Render Result"].save_render(str(path))


def save_preview(exr: Path, exposure: float = 1.0) -> None:
    """sRGB 8-bit PNG next to the EXR, for eyeballing only."""
    import OpenImageIO as oiio

    px, _ = read_exr(str(exr))
    x = np.clip(px * exposure, 0, 1)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    out = oiio.ImageOutput.create(str(exr.with_suffix(".png")))
    out.open(str(exr.with_suffix(".png")), oiio.ImageSpec(px.shape[1], px.shape[0], px.shape[2], "uint8"))
    out.write_image((srgb * 255 + 0.5).astype(np.uint8))
    out.close()


def time_config(scene: bpy.types.Scene, cfg: dict[str, Any], spps: list[int], repeats: int, exr: Path, label: str) -> dict[str, Any]:
    """Apply settings, measure overhead and T(n) for each n, save the highest-spp EXR."""
    manifest = cs.apply_settings(scene, {**cfg, "spp": 1})
    render(scene, 8)  # warm-up (scene features, specialised kernels)
    overhead = min(render(scene, 1) for _ in range(max(2, repeats)))
    res = scene.render.resolution_x
    entries, walls = [], {}
    for n in spps:
        wall = min(render(scene, n) for _ in range(repeats))
        walls[n] = wall
        secs = wall - overhead
        entries.append({
            "scene": label, "res": res, "max_bounces": cfg["max_bounces"], "spp_measured": n,
            "seconds": round(secs, 4), "s_per_4096spp": round(secs / (n - 1) * 4096, 3),
            "wall_s": round(wall, 4), "overhead_s": round(overhead, 4),
        })
        print(f"[smoke] {label} b={cfg['max_bounces']} {n} spp: wall {wall:.3f} s, overhead {overhead:.3f} s, "
              f"-> {entries[-1]['s_per_4096spp']:.2f} s/4096spp", flush=True)
    if len(spps) >= 2:
        a, b = spps[0], spps[-1]
        slope = (walls[b] - walls[a]) / (b - a)
        for e in entries:
            e["s_per_4096spp_slope"] = round(slope * 4096, 3)  # overhead-free cross-check
    manifest["scene.cycles.samples"] = scene.cycles.samples  # the saved EXR's spp
    save_exr(exr)
    info = verify_exr(str(exr), (scene.render.resolution_y, scene.render.resolution_x))
    save_preview(exr)
    return {"entries": entries, "exr": info, "manifest_size": len(manifest), "manifest": manifest}


# --- scenes -------------------------------------------------------------------------------------


def cornell(args: argparse.Namespace, results: dict[str, Any]) -> None:
    scene = reset()
    spec = make_cornell.build_cornell(scene, resolution=(args.res, args.res))
    # Mode B (per-light MIS on): in Mode A Cycles hides the camera-visible area light (cycles_settings).
    base = {"camera": spec["camera"], "lights": spec["lights"], "resolution": (args.res, args.res), "seed": 0, "light_mis": True}
    with gpu_lock():
        cs.apply_settings(scene, {**base, "spp": 1, "max_bounces": args.bounces[0]})
        t_first = render(scene, 1)
        runs = {}
        for b in args.bounces:
            runs[b] = time_config(scene, {**base, "max_bounces": b}, args.cornell_spp, args.repeats,
                                  args.out_dir / f"cornell_b{b}_{args.cornell_spp[-1]}spp.exr", "cornell")
    overhead = min(e["overhead_s"] for r in runs.values() for e in r["entries"])
    results["first_render_1spp_s"] = round(t_first, 3)
    results["metal_first_compile_s"] = round(t_first - overhead, 3)
    results.setdefault("scenes", {})["cornell"] = (
        "make_cornell.build_cornell: 0.555 m box, 5 quads + 2 boxes (flat), V1 Diffuse BSDF, one-sided rect light "
        f"{make_cornell.LIGHT_SIZE[0]}x{make_cornell.LIGHT_SIZE[1]} m {make_cornell.LIGHT_POWER_W} W, Mode B (MIS on), no world")
    results["cornell"] = {str(b): {k: v for k, v in r.items() if k != "manifest"} for b, r in runs.items()}
    last = runs[args.bounces[-1]]["manifest"]
    (args.out_dir / "cornell_manifest.json").write_text(json.dumps(last, indent=1, sort_keys=True) + "\n")
    results["entries"] += [e for r in runs.values() for e in r["entries"]]


def marker(args: argparse.Namespace, results: dict[str, Any]) -> None:
    """Emissive quads covering exact pixel footprints, placed with the plan §1.2 mapping (raster y from
    the bottom): image pixel (c, r) = raster [c, c+1] x [H-1-r, H-r]. Checks which EXR row they land in."""
    W, H, vfov = 64, 48, math.radians(40.0)
    ty = math.tan(vfov / 2)
    scene = reset()
    co = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))  # identity: looks -Z, up +Y
    scene.collection.objects.link(co)
    scene.camera = co

    def quad(name: str, c0: int, c1: int, r: int, rgb: tuple[float, float, float]) -> None:
        y0, y1 = H - 1 - r, H - r
        xs = [(2 * rx / W - 1) * ty * W / H for rx in (c0, c1 + 1)]
        ys = [(2 * ry / H - 1) * ty for ry in (y0, y1)]
        v = [(xs[0], ys[0], -1.0), (xs[1], ys[0], -1.0), (xs[1], ys[1], -1.0), (xs[0], ys[1], -1.0)]
        me = bpy.data.meshes.new(name)
        me.from_pydata(v, [], [(0, 1, 2, 3)])
        m = bpy.data.materials.new(name)
        nt = m.node_tree
        nt.nodes.clear()
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (*rgb, 1.0)
        em.inputs["Strength"].default_value = 1.0
        nt.links.new(em.outputs["Emission"], nt.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
        me.materials.append(m)
        scene.collection.objects.link(bpy.data.objects.new(name, me))

    red = (3, 4, 2)  # image columns 3..4, image row 2 (near the TOP-left)
    green = (W - 6, W - 6, H - 4)  # one pixel near the BOTTOM-right
    quad("red_marker", *red, (1.0, 0.0, 0.0))
    quad("green_marker", *green, (0.0, 1.0, 0.0))
    exr = args.out_dir / "orientation_marker.exr"
    with gpu_lock():
        cs.apply_settings(scene, {"camera": {"vfov_rad": vfov}, "resolution": (W, H), "spp": 16, "max_bounces": 0})
        render(scene, 16)
    save_exr(exr)
    verify_exr(str(exr), (H, W))
    px, _ = read_exr(str(exr))
    red_rows = sorted({int(r) for r in np.nonzero(px[..., 0] > 0.5)[0]})
    red_cols = sorted({int(c) for c in np.nonzero(px[..., 0] > 0.5)[1]})
    green_rows = sorted({int(r) for r in np.nonzero(px[..., 1] > 0.5)[0]})
    green_cols = sorted({int(c) for c in np.nonzero(px[..., 1] > 0.5)[1]})
    lit = px.max(axis=2) > 1e-6
    expect = np.zeros_like(lit)
    expect[red[2], red[0]:red[1] + 1] = True
    expect[green[2], green[0]] = True
    out = {
        "resolution": [W, H], "vfov_deg": 40.0,
        "red_expected": {"rows": [red[2]], "cols": [red[0], red[1]]}, "red_found": {"rows": red_rows, "cols": red_cols},
        "green_expected": {"rows": [green[2]], "cols": [green[0]]}, "green_found": {"rows": green_rows, "cols": green_cols},
        "marker_values": {"red": px[red[2], red[0]:red[1] + 1, 0].tolist(), "green": [float(px[green[2], green[0], 1])]},
        "exact_footprint": bool((lit == expect).all()),
        "max_leak_outside": float(px[~expect].max()) if (~expect).any() else 0.0,
    }
    out["exr_row0_is_top"] = red_rows == [red[2]] and green_rows == [green[2]] and red_cols == [red[0], red[1]] and green_cols == [green[0]]
    results["orientation"] = out
    print(f"[smoke] orientation: {json.dumps(out)}", flush=True)
    if not out["exr_row0_is_top"]:
        raise AssertionError(f"EXR orientation check failed: {out}")


def sponza(args: argparse.Namespace, results: dict[str, Any]) -> None:
    gltf = args.sponza_dir / "Sponza.gltf"
    if not gltf.exists():
        raise FileNotFoundError(f"{gltf} missing: run python3 validation/blender/fetch_sponza.py")
    scene = reset()
    bpy.ops.import_scene.gltf(filepath=str(gltf), import_shading="FLAT")
    mn, mx = Vector((1e30,) * 3), Vector((-1e30,) * 3)
    tris = 0
    for o in scene.objects:
        if o.type == "MESH":
            for c in o.bound_box:
                w = o.matrix_world @ Vector(c)
                mn, mx = Vector(map(min, mn, w)), Vector(map(max, mx, w))
            tris += sum(len(p.vertices) - 2 for p in o.data.polygons)
    co = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    # Ground floor of the atrium near its +X end (open-sky court spans |x| < ~9 m, floor z ~ 0).
    co.location = (0.6 * mx.x, 0.3, 1.5)
    co.rotation_euler = (math.radians(95.0), 0.0, math.radians(90.0))  # look down -X, slightly up
    scene.collection.objects.link(co)
    scene.camera = co
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 8.0
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(12.0), math.radians(-8.0), 0.0)  # high sun into the open court
    scene.collection.objects.link(so)
    base = {"camera": {"vfov_deg": 60.0}, "resolution": (args.res, args.res), "seed": 0,
            "lights": {"sun": {"visible_camera": False}}}
    with gpu_lock():
        run = time_config(scene, {**base, "max_bounces": 3}, args.sponza_spp, args.repeats,
                          args.out_dir / f"sponza_b3_{args.sponza_spp[-1]}spp.exr", "sponza")
    results.setdefault("scenes", {})["sponza"] = (
        f"Khronos glTF-Sample-Assets Sponza (glTF, {tris} tris, {len(bpy.data.materials)} materials, "
        f"{len(bpy.data.images)} images), import_shading FLAT, sun 8 W/m2 angle 0, no world, vfov 60 deg")
    results["sponza"] = {"triangles": tris, "bbox_min": list(mn), "bbox_max": list(mx),
                         **{k: v for k, v in run.items() if k != "manifest"}}
    results["entries"] += run["entries"]


# --- main ---------------------------------------------------------------------------------------


def write_budget(path: Path, results: dict[str, Any], device: str, cold_compile_s: float | None) -> None:
    old: dict[str, Any] = {}
    if path.exists():
        old = json.loads(path.read_text())
    key = lambda e: (e["scene"], e["res"], e["max_bounces"], e["spp_measured"])  # noqa: E731
    new_keys = {key(e) for e in results["entries"]}
    entries = [e for e in old.get("entries", []) if key(e) not in new_keys] + results["entries"]
    firsts = [v for v in (results.get("metal_first_compile_s"), old.get("metal_first_compile_s"), cold_compile_s) if v is not None]
    budget = {
        "blender": bpy.app.version_string.split(" ")[0],
        "device": device,
        # Budget the cold case: the largest first-render excess seen (warm Metal shader cache gives ~0).
        "metal_first_compile_s": max(firsts) if firsts else None,
        "metal_first_compile_this_run_s": results.get("metal_first_compile_s"),
        "first_render_1spp_this_run_s": results.get("first_render_1spp_s"),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": "wall time of bpy.ops.render.render (no file write); seconds = T(n) - T(1 spp); "
                  "s_per_4096spp = seconds/(n-1)*4096; s_per_4096spp_slope = (T(n_max)-T(n_min))/(n_max-n_min)*4096",
        "scenes": {**old.get("scenes", {}), **results.get("scenes", {})},
        "notes": [
            "metal_first_compile_s = largest first-render excess over a warm 1-spp render seen so far (cold Metal "
            "shader cache: kernels compile in MTLCompilerService); with a warm cache the first render costs ~0.3 s",
            "timings exclude EXR writing; 512x512, BOX filter, TABULATED_SOBOL, RR on (min_light_bounces 0), light tree on",
        ],
        "entries": sorted(entries, key=key),
    }
    path.write_text(json.dumps(budget, indent=2) + "\n")


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("validation/out/m0"))
    ap.add_argument("--budget", type=Path, default=Path("validation/budget.json"))
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--bounces", type=int, nargs="+", default=[3, 7])
    ap.add_argument("--cornell-spp", type=int, nargs="+", default=[128, 512])
    ap.add_argument("--sponza-spp", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--sponza-dir", type=Path, default=Path("validation/assets/downloaded/sponza"))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cold-compile-s", type=float, help="cold-cache first-compile time measured by an earlier run")
    ap.add_argument("--skip-cornell", action="store_true")
    ap.add_argument("--skip-sponza", action="store_true")
    ap.add_argument("--skip-marker", action="store_true")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"entries": []}
    reset()
    cs.enable_metal()
    cp = bpy.context.preferences.addons["cycles"].preferences
    device = " ".join(d.name for d in cp.devices if d.use) + " METAL"
    results["device"] = device
    if not args.skip_cornell:
        cornell(args, results)
    if not args.skip_marker:
        marker(args, results)
    if not args.skip_sponza:
        sponza(args, results)
    (args.out_dir / "smoke_results.json").write_text(json.dumps(results, indent=1) + "\n")
    if results["entries"]:
        write_budget(args.budget, results, device, args.cold_compile_s)
    print("[smoke] done", json.dumps({k: results.get(k) for k in ("device", "metal_first_compile_s", "first_render_1spp_s")}), flush=True)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
