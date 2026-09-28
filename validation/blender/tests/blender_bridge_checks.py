"""Blender-side bridge checks on the fixture package (run inside Blender; driven by run_m2_tests.py).

  Blender -b --factory-startup --python-exit-code 1 -P validation/blender/tests/blender_bridge_checks.py -- \
      --package DIR --facts facts.json --out DIR

Checks: faces in primId order with C-converted positions (bit-exact), UV v-flip, material indices, COLOR_0,
float custom normals, material node trees (V1, Principled, textures, KHR_texture_transform numerically),
lights / camera / world vs C·M, apply_frame, cycles_settings acceptance, renders (textured-emission quadrant
probes: UV orientation + transform + sRGB + nearest; smooth-normal probe), and hard errors for things the
bridge must refuse. Writes <out>/checks.json; exits 1 on any failure.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "tools"))
import build_scene as bs  # noqa: E402
import cycles_settings as cs  # noqa: E402
from render_reference import gpu_lock  # noqa: E402
from verify_exr import read_exr  # noqa: E402

RESULTS: list[dict[str, Any]] = []


def check(name: str, fn: Callable[[], Any]) -> Any:
    try:
        detail = fn()
        RESULTS.append({"name": name, "ok": True, "detail": detail})
        print(f"[bridge-checks] OK   {name}" + (f": {detail}" if detail is not None else ""), flush=True)
        return detail
    except Exception as e:  # noqa: BLE001
        RESULTS.append({"name": name, "ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})
        print(f"[bridge-checks] FAIL {name}: {type(e).__name__}: {e}", flush=True)
        return None


def mat4(M: Any) -> np.ndarray:
    return np.array([list(r) for r in M], dtype=np.float64)


def expect_close(a: Any, b: Any, tol: float, what: str) -> float:
    err = float(np.abs(np.asarray(a, float) - np.asarray(b, float)).max())
    if err > tol:
        raise AssertionError(f"{what}: max err {err:.3g} > {tol} ({np.asarray(a).tolist()} vs {np.asarray(b).tolist()})")
    return err


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--facts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    facts = json.loads(args.facts.read_text())
    sj = facts["scene"]
    A = {k: np.asarray(v) for k, v in facts["arrays"].items()}
    C = bs.C3

    holder: dict[str, Any] = {}
    check("build_from_package", lambda: holder.update(b=bs.build_from_package(args.package)))
    if "b" not in holder:
        (args.out / "checks.json").write_text(json.dumps({"ok": False, "results": RESULTS}, indent=1, default=str) + "\n")
        return 1
    built = holder["b"]
    me = built["mesh"].data
    T, V = len(A["indices"]), len(A["positions"])

    def mesh_topology() -> str:
        assert len(me.polygons) == T and len(me.vertices) == V
        lv = np.empty(3 * T, np.int32)
        me.loops.foreach_get("vertex_index", lv)
        assert np.array_equal(lv.reshape(-1, 3), A["indices"]), "face vertex order != package indices (primId order)"
        co = np.empty(3 * V, np.float32)
        me.vertices.foreach_get("co", co)
        want = (A["positions"].astype(np.float32).astype(np.float64) @ C.T).astype(np.float32)
        assert np.array_equal(co.reshape(-1, 3), want), "positions != C·p (bit-exact)"
        mi = np.empty(T, np.int32)
        me.polygons.foreach_get("material_index", mi)
        assert np.array_equal(mi, A["triMaterial"].reshape(-1))
        return f"{T} faces in primId order, positions bit-exact"
    check("mesh topology/positions/material indices", mesh_topology)

    def mesh_attrs() -> str:
        uv = np.empty(6 * T, np.float32)
        me.uv_layers[bs.UV_MAP].data.foreach_get("uv", uv)
        want = A["uv0"].astype(np.float32)[A["indices"].reshape(-1)].copy()
        want[:, 1] = np.float32(1.0) - want[:, 1]
        assert np.array_equal(uv.reshape(-1, 2), want), "uv != (u, 1 - v)"
        col = np.empty(4 * V, np.float32)
        me.color_attributes[bs.COLOR_ATTR].data.foreach_get("color", col)
        assert np.array_equal(col.reshape(-1, 4), A["color0"].astype(np.float32))
        vn = np.empty(3 * V, np.float32)
        me.vertex_normals.foreach_get("vector", vn)
        wantn = (A["normals"].astype(np.float32).astype(np.float64) @ C.T).astype(np.float32)
        assert np.array_equal(vn.reshape(-1, 3), wantn), "vertex normals != C·n (bit-exact)"
        assert me.normals_domain == "POINT"
        return "uv v-flip, COLOR_0, custom normals bit-exact"
    check("mesh uv/color/normals", mesh_attrs)

    def materials() -> dict[str, Any]:
        mats = built["materials"]
        assert len(mats) == len(sj["materials"])
        out = {}
        for m, pm in zip(mats, sj["materials"]):
            nodes = m.node_tree.nodes
            for n in nodes:
                if "distribution" in n.bl_rna.properties:
                    assert n.distribution == "GGX", (m.name, n.name, n.distribution)
            assert m.cycles.emission_sampling == pm["emissionSampling"]
            if pm["model"] == "v1":
                g = next(n for n in nodes if n.bl_idname == "ShaderNodeBsdfAnisotropic")
                d = next(n for n in nodes if n.bl_idname == "ShaderNodeBsdfDiffuse")
                mix = next(n for n in nodes if n.bl_idname == "ShaderNodeMixShader")
                expect_close(g.inputs["Roughness"].default_value, pm["v1"]["roughness"], 1e-7, "glossy roughness")
                expect_close(mix.inputs["Fac"].default_value, pm["v1"]["mix"], 1e-7, "mix")
                expect_close(list(d.inputs["Color"].default_value)[:3], pm["v1"]["diffuse"], 1e-7, "diffuse")
                assert d.inputs["Roughness"].default_value == 0.0
                assert mix.inputs[1].links[0].from_node == d and mix.inputs[2].links[0].from_node == g
            else:
                p = next(n for n in nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")
                expect_close(p.inputs["IOR"].default_value, pm["ior"], 1e-6, "ior")
                if not p.inputs["Specular IOR Level"].is_linked:
                    expect_close(p.inputs["Specular IOR Level"].default_value, 0.5 * pm["specularFactor"], 1e-7, "spec level")
                if not p.inputs["Emission Strength"].is_linked:
                    expect_close(p.inputs["Emission Strength"].default_value, pm["emission"]["strength"] if "emission" in pm else pm["emissiveStrength"], 1e-7, "emission strength")
                for k in ("Coat Weight", "Sheen Weight", "Subsurface Weight", "Diffuse Roughness"):
                    assert p.inputs[k].default_value == 0.0, k
                if pm["alphaMode"] == "MASK":
                    assert p.inputs["Alpha"].is_linked
            out[m.name] = sorted({n.bl_idname for n in nodes})
        return {"n": len(mats)}
    check("material node trees", materials)

    def tex_transforms() -> str:
        rng = np.random.default_rng(1)
        uv = rng.uniform(-1, 2, (200, 2))
        n_checked = 0
        for m, pm in zip(built["materials"], sj["materials"]):
            ref = pm.get("emissiveTexture")
            if not ref or ref.get("transform") is None:
                continue
            mp = next(n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeMapping")
            assert mp.vector_type == "POINT"
            loc, rot, sc = (np.array(mp.inputs[k].default_value) for k in ("Location", "Rotation", "Scale"))
            assert rot[0] == 0 and rot[1] == 0
            a, b, c, d, e, f = ref["transform"]
            want = np.stack([a * uv[:, 0] + b * uv[:, 1] + c, 1 - (d * uv[:, 0] + e * uv[:, 1] + f)], 1)
            ub = np.stack([uv[:, 0], 1 - uv[:, 1]], 1) * sc[:2]
            cr, sr = math.cos(rot[2]), math.sin(rot[2])
            got = np.stack([cr * ub[:, 0] - sr * ub[:, 1], sr * ub[:, 0] + cr * ub[:, 1]], 1) + loc[:2]
            expect_close(got, want, 2e-6, f"{m.name} Mapping(POINT) vs package transform")
            n_checked += 1
        assert n_checked == 2
        return "Blender Mapping node == package 2x3 transform in v-flipped space (2 materials)"
    check("KHR_texture_transform mapping", tex_transforms)

    def lights() -> dict[str, Any]:
        mode_b = sj["lightMode"] == "B"
        errs = {}
        for L in sj["lights"]:
            ob = built["lights"][L["id"]]
            ld = ob.data
            M = np.asarray(L["matrix"], float).reshape(4, 4).T
            errs[ob.name] = expect_close(mat4(ob.matrix_world), bs.C4 @ M, 2e-6, f"{ob.name} matrix_world")
            assert tuple(ob.scale) == (1.0, 1.0, 1.0)
            want_type = {"point": "POINT", "spot": "SPOT", "rect": "AREA", "disk": "AREA", "sun": "SUN"}[L["type"]]
            assert ld.type == want_type
            expect_close(ld.energy, L["power"], 1e-6, "energy")
            expect_close(list(ld.color), L["color"], 1e-7, "color")
            expect_close(ld.exposure, L["exposure"], 1e-7, "exposure")
            assert ld.cycles.use_multiple_importance_sampling == mode_b
            assert ob.visible_camera == L["visibleToCamera"]
            if L["type"] == "spot":
                expect_close([ld.spot_size, ld.spot_blend], [L["spotSize"], L["spotBlend"]], 1e-6, "spot")
            if L["type"] in ("point", "spot"):
                assert ld.shadow_soft_size == 0 and not ld.use_soft_falloff
            if L["type"] == "rect":
                assert ld.shape == "RECTANGLE"
                expect_close([ld.size, ld.size_y, ld.spread], [L["sizeX"], L["sizeY"], L["spread"]], 1e-6, "rect")
            if L["type"] == "disk":
                assert ld.shape == "DISK"
                expect_close([ld.size, ld.spread], [L["sizeX"], L["spread"]], 1e-6, "disk")
            if L["type"] == "sun":
                assert ld.angle == 0.0
        # emission axis: light local -Z, as a world vector, must equal C·(M·(0,0,-1))
        return {"max_matrix_err": max(errs.values())}
    check("lights", lights)

    def camera_world() -> dict[str, Any]:
        co = built["camera"]
        M = np.asarray(sj["camera"]["matrix"], float).reshape(4, 4).T
        e = expect_close(mat4(co.matrix_world), bs.C4 @ M, 2e-6, "camera matrix_world")
        assert co.data.sensor_fit == "VERTICAL"
        expect_close(co.data.angle_y, sj["camera"]["yfov"], 1e-7, "angle_y")
        # camera looks down Blender -Z_cam: forward must be C·(glTF forward)
        fwd_b = -mat4(co.matrix_world)[:3, 2]
        expect_close(fwd_b, C @ -M[:3, 2], 1e-6, "camera forward")
        w = built["world"]
        mp, bg, tint = bs._world_nodes(w)
        expect_close(mp.inputs["Rotation"].default_value[2], sj["env"]["rotationZ"], 1e-7, "env rotation")
        expect_close(bg.inputs["Strength"].default_value, sj["env"]["strength"], 1e-7, "env strength")
        expect_close(list(tint.inputs[1].default_value), sj["env"]["tint"], 1e-7, "env tint")
        assert w.cycles_visibility.camera == sj["env"]["visibleToCamera"]
        assert mp.vector_type == "POINT"
        env_info = built["manifest"]["env"]
        assert env_info["hash_ok"] and env_info["image_sha256"] == sj["env"]["sha256"]
        return {"camera_err": e, "env_sha256": env_info["image_sha256"][:16]}
    check("camera + world (+ ENV-U9 hash)", camera_world)

    def frames() -> str:
        f5 = next(f for f in sj["frames"] if f["frame"] == 5)
        bs.apply_frame(built, f5)
        M = np.asarray(f5["lights"]["0"]["matrix"], float).reshape(4, 4).T
        expect_close(mat4(built["lights"][0].matrix_world), bs.C4 @ M, 2e-6, "frame light matrix")
        assert built["lights"][0].data.energy == 60.0
        expect_close(built["camera"].data.angle_y, f5["camera"]["yfov"], 1e-7, "frame yfov")
        mp, bg, _ = bs._world_nodes(built["world"])
        expect_close([mp.inputs["Rotation"].default_value[2], bg.inputs["Strength"].default_value], [-0.5, 0.75], 1e-7, "frame env")
        cfg = bs.settings_cfg(built, spp=4, device="GPU")
        man = cs.apply_settings(built["scene"], cfg)
        assert man[f'objects["{built["lights"][0].name}"].data.energy'] == 60.0
        bs.apply_frame(built, None)
        assert built["lights"][0].data.energy == sj["lights"][0]["power"]
        return "frame 5 applied and read back through cycles_settings; base state restored"
    check("apply_frame", frames)

    def settings() -> int:
        man = cs.apply_settings(built["scene"], bs.settings_cfg(built, spp=16, device="GPU"))
        interp = {k: v for k, v in man.items() if k.endswith(".interpolation")}
        assert any(v == "Closest" for v in interp.values()) and any(v == "Linear" for v in interp.values())
        sink = next(m.name for m in built["materials"] if m.name.endswith("sink"))
        assert man[f'materials["{sink}"].cycles.emission_sampling'] == "NONE"
        return len(man)
    check("cycles_settings accepts the bridged scene", settings)

    def render_probes() -> dict[str, Any]:
        cs.apply_settings(built["scene"], bs.settings_cfg(built, spp=16, device="GPU", max_bounces=0))
        path = args.out / "fixture_render.exr"
        with gpu_lock():
            bpy.ops.render.render(write_still=False)
        bpy.data.images["Render Result"].save_render(str(path))
        img, _ = read_exr(str(path))
        worst = 0.0
        for p in facts["probes"]:
            c, r = p["pixel"]
            got = img[r, c, :3]
            err = float(np.abs(got - np.asarray(p["value"])).max() / max(max(p["value"]), 1e-6))
            worst = max(worst, err)
            if err > 2e-3:
                raise AssertionError(f"quad {p['quad']} uv {p['uv']} pixel {p['pixel']}: got {got.tolist()} want {p['value']}")
        return {"probes": len(facts["probes"]), "max_rel_err": worst}
    check("render: textured emission (UV orientation, transform, sRGB, nearest)", render_probes)

    def render_normals() -> dict[str, Any]:
        m = next(m for m in built["materials"] if m.name.endswith("v1gloss"))
        nt = m.node_tree
        nt.nodes.clear()
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        ma = nt.nodes.new("ShaderNodeVectorMath")
        ma.operation = "MULTIPLY_ADD"
        ma.inputs[1].default_value = (0.5, 0.5, 0.5)
        ma.inputs[2].default_value = (0.5, 0.5, 0.5)
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(geo.outputs["Normal"], ma.inputs[0])
        nt.links.new(ma.outputs[0], em.inputs["Color"])
        nt.links.new(em.outputs[0], out.inputs["Surface"])
        cs.apply_settings(built["scene"], bs.settings_cfg(built, spp=4, device="GPU", max_bounces=0))
        path = args.out / "fixture_normals.exr"
        with gpu_lock():
            bpy.ops.render.render(write_still=False)
        bpy.data.images["Render Result"].save_render(str(path))
        img, _ = read_exr(str(path))
        c, r = facts["normal_pixel"]
        want = 0.5 * (C @ np.asarray(facts["n0_gltf"])) + 0.5
        e = expect_close(img[r, c, :3], want, 1e-4, "shading normal (Geometry.Normal) at the smooth quad")
        return {"pixel": [c, r], "got": img[r, c, :3].tolist(), "want": want.tolist(), "err": e}
    check("render: smooth custom normals reach Cycles", render_normals)

    # --- hard errors -----------------------------------------------------------------------------
    def expect_bridge_error(mutate: Callable[[dict[str, Any], Path], None], what: str) -> Callable[[], str]:
        def run() -> str:
            with tempfile.TemporaryDirectory() as td:
                pkg = Path(td) / "pkg"
                shutil.copytree(args.package, pkg)
                s = json.loads((pkg / "scene.json").read_text())
                mutate(s, pkg)
                (pkg / "scene.json").write_text(json.dumps(s))
                try:
                    bs.build_from_package(pkg)
                except bs.BridgeError as e:
                    return f"BridgeError: {str(e)[:90]}"
                raise AssertionError(f"{what}: no BridgeError")
        return run

    def m_blend(s: dict[str, Any], _: Path) -> None:
        s["materials"][3]["alphaMode"] = "BLEND"

    def m_mode_a(s: dict[str, Any], _: Path) -> None:
        s["lightMode"] = "A"

    def m_scale(s: dict[str, Any], _: Path) -> None:
        s["camera"]["matrix"] = [2.0 * x if i in (0, 5, 10) else x for i, x in enumerate(s["camera"]["matrix"])]

    def m_ktx(s: dict[str, Any], pkg: Path) -> None:
        (pkg / "tex_1.ktx2").write_bytes(b"\xabKTX 20\xbb")
        s["textures"][3]["file"] = "tex_1.ktx2"

    def m_hash(s: dict[str, Any], _: Path) -> None:
        s["env"]["sha256"] = "0" * 64

    def m_texcoord(s: dict[str, Any], _: Path) -> None:
        s["materials"][0]["emissiveTexture"]["texCoord"] = 1

    for mut, what in ((m_blend, "BLEND alpha"), (m_mode_a, "Mode A + camera-visible rect light"),
                      (m_scale, "scaled camera matrix"), (m_ktx, "KTX2 texture"), (m_hash, "wrong env.sha256"),
                      (m_texcoord, "TEXCOORD_1 reference")):
        check(f"hard error: {what}", expect_bridge_error(mut, what))

    ok = all(r["ok"] for r in RESULTS)
    (args.out / "checks.json").write_text(json.dumps({"ok": ok, "results": RESULTS}, indent=1, default=str) + "\n")
    print(f"[bridge-checks] {'PASS' if ok else 'FAIL'} {sum(r['ok'] for r in RESULTS)}/{len(RESULTS)}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []))
