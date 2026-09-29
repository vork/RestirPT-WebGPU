"""M2 Blender-side test driver (plan §5 M2: scene bridge, render_reference, C0a/C0b/C0p calibration).

  /Applications/Blender.app/Contents/Resources/5.2/python/bin/python3.13 validation/blender/tests/run_m2_tests.py \
      [--blender /Applications/Blender.app/Contents/MacOS/Blender] [--out validation/out/m2] [--quick]

Steps (each recorded in <out>/summary.json; exit 1 on any failure):
 1. calib_scenes.py regenerated into a temp dir == validation/scenes (byte-identical: packages are current)
 2. bridge fixture (tests/fixture_package.py) -> Blender checks (tests/blender_bridge_checks.py)
 3. render_reference.py for every calibration package (cache under <out>/refs), marker_check vs "expected":
    C0b pixel == L_e (<= 1e-4), C0a centroids <= 0.1 px, C0p centroids <= 0.1 px (gate is Cycles-vs-ours;
    vs analytic we report and require <= 0.5 px), probes exact
 4. planted errors must be detected: C0a/C0p image flips, C0a +0.2 px centroid shift, and the four C0p plants
    (u + 0.5 texel, v flip, gamma sign, missing C) recomputed as planted expectations
 5. render_reference re-run on C0b and C0a -> cache hit (no render)
"""
from __future__ import annotations

import argparse
import copy
import filecmp
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
BL = HERE.parent
REPO = BL.parent.parent
sys.path.insert(0, str(BL))
sys.path.insert(0, str(REPO / "validation" / "tools"))
import calib_scenes as cal  # noqa: E402
import fixture_package as fp  # noqa: E402  (tests dir is on sys.path as the script dir)
import marker_check as mc  # noqa: E402
from imageio_util import read_image  # noqa: E402

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
SCENES = REPO / "validation" / "scenes"
RENDER_PLAN = {  # package: (spp, seeds)
    "c0b_512": (16, "0..1"),
    "c0a_512": (64, "0"),
    "c0a_640x360": (64, "0"),
    "c0a_360x640": (64, "0"),
    "c0a_far_512": (64, "0"),
    "c0p_512": (256, "0"),
    "c0p_640x360": (256, "0"),
    "c0p_360x640": (256, "0"),
    "c0p_hidden_512": (16, "0"),
}

SUMMARY: dict[str, Any] = {"steps": {}}


def step(name: str, ok: bool, **detail: Any) -> None:
    SUMMARY["steps"][name] = {"ok": bool(ok), **detail}
    print(f"[m2-tests] {'PASS' if ok else 'FAIL'} {name} " + json.dumps(detail, default=str)[:400], flush=True)


def blender(script: Path, args: list[str], blender_bin: str) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    p = subprocess.run([blender_bin, "-b", "--factory-startup", "--python-exit-code", "1", "-P", str(script), "--", *args],
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr, time.perf_counter() - t0


def render_ref(pkg: Path, out: Path, spp: int, seeds: str, blender_bin: str) -> dict[str, Any]:
    code, log, wall = blender(BL / "render_reference.py", ["--package", str(pkg), "--out", str(out), "--spp", str(spp),
                                                           "--seeds", seeds], blender_bin)
    lines = [ln for ln in log.splitlines() if ln.startswith("[render_reference] RESULT ")]
    if code != 0 or not lines:
        raise RuntimeError(f"render_reference failed ({code}) for {pkg.name}:\n{log[-3000:]}")
    res = json.loads(lines[-1].split("RESULT ", 1)[1])
    res["wall_s"] = round(wall, 3)
    return res


def check_scenes_current() -> None:
    with tempfile.TemporaryDirectory() as td:
        cal.build_all(Path(td))
        diffs = []
        for d in sorted(Path(td).iterdir()):
            ref = SCENES / d.name
            if not ref.exists():
                diffs.append(f"{d.name}: missing in validation/scenes")
                continue
            cmp = filecmp.dircmp(d, ref)
            files = sorted(set(cmp.left_list) | set(cmp.right_list))
            for f in files:
                if not (d / f).exists() or not (ref / f).exists() or not filecmp.cmp(d / f, ref / f, shallow=False):
                    diffs.append(f"{d.name}/{f}")
    step("calib packages current (byte-identical regeneration)", not diffs, diffs=diffs)


def bridge_fixture(out: Path, blender_bin: str) -> None:
    fx = out / "fixture"
    info = fp.make_bridge_fixture(fx / "package")
    (fx / "facts.json").write_text(json.dumps(info))
    code, log, wall = blender(HERE / "blender_bridge_checks.py",
                              ["--package", str(fx / "package"), "--facts", str(fx / "facts.json"), "--out", str(fx)], blender_bin)
    res = json.loads((fx / "checks.json").read_text()) if (fx / "checks.json").exists() else {"ok": False, "results": []}
    step("bridge fixture checks (Blender)", code == 0 and res["ok"], wall_s=round(wall, 1),
         passed=f"{sum(r['ok'] for r in res['results'])}/{len(res['results'])}",
         failures=[f"{r['name']}: {r.get('error')}" for r in res["results"] if not r["ok"]] or ([] if code == 0 else [log[-2000:]]))


def calibration(out: Path, blender_bin: str, quick: bool) -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    timings = {}
    for name, (spp, seeds) in RENDER_PLAN.items():
        if quick and name not in ("c0b_512", "c0a_512", "c0p_512"):
            continue
        res = render_ref(SCENES / name, out / "refs", spp, seeds, blender_bin)
        dirs[name] = Path(res["dir"])
        timings[name] = {k: res.get(k) for k in ("cache_hit", "renders", "renders_total_s", "wall_s")}
        chk = mc.check_render_dir(Path(res["dir"]))
        tol_ok = True
        if name.startswith("c0p") and chk.get("max_err_px") is not None:
            tol_ok = chk["max_err_px"] <= 0.5
        ok = (chk["ok"] if not name.startswith("c0p") else (tol_ok and not [f for f in chk["failures"] if "centroid" not in f]))
        step(f"calibration {name}", ok, spp=spp, seeds=seeds, renders=chk["renders"], max_err_px=chk["max_err_px"],
             max_probe_err=chk["max_probe_err"], max_const_err=chk["max_const_err"],
             within_0p1px=chk["ok"], failures=chk["failures"][:5], timing=timings[name])
    SUMMARY["render_timings"] = timings
    return dirs


def _first_img(d: Path, frame: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    man = json.loads((d / "manifest.json").read_text())
    r = man["renders"][0] if frame is None else next(x for x in man["renders"] if x["frame"] == frame)
    return read_image(d / r["file"], drop_alpha=True), r


def plants(dirs: dict[str, Path]) -> None:
    # C0a: image flips and a 0.2 px centroid shift must fail
    if "c0a_512" in dirs:
        sj = json.loads((SCENES / "c0a_512" / "scene.json").read_text())
        img, _ = _first_img(dirs["c0a_512"])
        exp, fe = sj["expected"], sj["expected"]["frames"]["0"]
        res = {"vflip": not mc.check_frame(img[::-1], exp, fe)["ok"], "hflip": not mc.check_frame(img[:, ::-1], exp, fe)["ok"]}
        shifted = copy.deepcopy(fe)
        for m in shifted["markers"]:
            m["centroid_px"][0] += 0.2
        res["shift_0.2px"] = not mc.check_frame(img, exp, shifted)["ok"]
        step("C0a planted errors detected", all(res.values()), detected=res)
    # C0p: planted mapping bugs, recomputed expectations, compared with the (correct) Cycles images
    for name in ("c0p_512", "c0p_640x360", "c0p_360x640"):
        if name not in dirs:
            continue
        sj = json.loads((SCENES / name / "scene.json").read_text())
        W, H = sj["render"]["width"], sj["render"]["height"]
        T = cal.c0p_texels()
        man = json.loads((dirs[name] / "manifest.json").read_text())
        frames = {f["frame"]: f for f in sj["frames"]}
        det: dict[str, Any] = {}
        for plant in cal.PLANTS:
            failed, affected = 0, 0
            for r in man["renders"]:
                f = frames[r["frame"]]
                M = np.asarray(f["camera"]["matrix"], float).reshape(4, 4).T
                g = f["env"]["rotationZ"]
                if plant == "gamma_sign" and g == 0:
                    continue  # identical mapping: nothing to detect
                affected += 1
                pe = cal.c0p_frame_expected(T, g, M[:3, :3], f["camera"]["yfov"], W, H, plant=plant)
                img = read_image(dirs[name] / r["file"], drop_alpha=True)
                if not mc.check_frame(img, sj["expected"], pe)["ok"]:
                    failed += 1
            det[plant] = {"frames_failed": failed, "frames_affected": affected}
        vimg, _ = _first_img(dirs[name])
        det["image_vflip"] = not mc.check_frame(vimg[::-1], sj["expected"], sj["expected"]["frames"]["0"])["ok"]
        ok = det["image_vflip"] and all(det[p]["frames_failed"] == det[p]["frames_affected"] > 0 for p in cal.PLANTS)
        step(f"C0p planted errors detected ({name})", ok, detected=det)


def cache_demo(out: Path, blender_bin: str) -> None:
    res = {}
    for name in ("c0b_512", "c0a_512"):
        spp, seeds = RENDER_PLAN[name]
        r = render_ref(SCENES / name, out / "refs", spp, seeds, blender_bin)
        res[name] = {"cache_hit": r["cache_hit"], "wall_s": r["wall_s"], "dir": Path(r["dir"]).name}
    step("render_reference cache hit on re-run", all(v["cache_hit"] for v in res.values()), runs=res)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blender", default=BLENDER)
    ap.add_argument("--out", type=Path, default=REPO / "validation" / "out" / "m2")
    ap.add_argument("--quick", action="store_true", help="only c0b_512, c0a_512, c0p_512")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    check_scenes_current()
    bridge_fixture(args.out, args.blender)
    dirs = calibration(args.out, args.blender, args.quick)
    plants(dirs)
    cache_demo(args.out, args.blender)
    ok = all(s["ok"] for s in SUMMARY["steps"].values())
    SUMMARY.update(ok=ok, total_s=round(time.perf_counter() - t0, 1))
    (args.out / "summary.json").write_text(json.dumps(SUMMARY, indent=1, default=str) + "\n")
    print(f"[m2-tests] {'ALL PASS' if ok else 'FAILURES'} in {SUMMARY['total_s']} s -> {args.out / 'summary.json'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
