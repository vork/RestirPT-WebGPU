"""Cycles reference renders of a scene package (plan §5 M2 render_reference.py, §7.5; scene-bridge.md).

  Blender -b --factory-startup --python-exit-code 1 -P validation/blender/render_reference.py -- \
      --package validation/scenes/c0b_512 --out validation/out/refs --spp 64 --seeds 0..3 \
      [--frames all|0,3,5] [--max-bounces b] [--device GPU|CPU] [--force]

Output: <out>/<package name>-<key16>/f{frame:04d}_s{seed:03d}.exr (+ manifest.json, written last with
"complete": true). Cycles seed = splitmix64((seed << 32) | frame) & 0x7fffffff (see cycles_seed()).

Cache: key = sha256 over (package file hashes, render args, sha256 of every validation/blender/*.py, Blender
version). If <out>/<name>-<key16>/manifest.json is complete and every EXR exists, nothing is rendered.

GPU lock: renders run inside the /tmp/restirpt-gpu.lock mkdir lock (plan §7.5 Orchestration), released in a
finally block and atexit / SIGINT / SIGTERM.

The last stdout line is  `[render_reference] RESULT {json}`  with keys dir, cache_hit, key, renders.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import bpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_scene as bs  # noqa: E402
import cycles_settings as cs  # noqa: E402
from verify_exr import verify_exr  # noqa: E402

LOCK = Path("/tmp/restirpt-gpu.lock")
_lock_held = False
MASK64 = (1 << 64) - 1


# --- GPU lock -----------------------------------------------------------------------------------


def _release_lock() -> None:
    global _lock_held
    if _lock_held:
        with contextlib.suppress(OSError):
            LOCK.rmdir()
        _lock_held = False


@contextlib.contextmanager
def gpu_lock() -> Iterator[float]:
    """until mkdir /tmp/restirpt-gpu.lock; do sleep 5; done ... rmdir. Yields the seconds waited."""
    global _lock_held
    t0 = time.perf_counter()
    while True:
        try:
            LOCK.mkdir()
            break
        except FileExistsError:
            time.sleep(5)
    _lock_held = True
    waited = time.perf_counter() - t0
    if waited > 1:
        print(f"[render_reference] waited {waited:.0f} s for the GPU lock", flush=True)
    try:
        yield waited
    finally:
        _release_lock()


atexit.register(_release_lock)
for _sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(_sig, lambda s, f: (_release_lock(), sys.exit(128 + s)))


# --- seeds, args, cache key ---------------------------------------------------------------------


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def cycles_seed(seed: int, frame: int) -> int:
    """Cycles scene.cycles.seed for replicate `seed` at `frame`: hash(seed, frame) & 0x7fffffff."""
    return splitmix64(((seed & 0xFFFFFFFF) << 32) | (frame & 0xFFFFFFFF)) & 0x7FFFFFFF


def parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if ".." in part:
            a, b = part.split("..")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    if not out or len(set(out)) != len(out) or min(out) < 0 or max(out) > 999:
        raise SystemExit(f"--seeds {s!r}: need distinct seeds in 0..999")
    return out


def resolve_frames(sj: dict[str, Any], spec: str) -> list[dict[str, Any] | None]:
    frames = sj.get("frames") or []
    if not frames:
        if spec not in ("all", "0"):
            raise SystemExit(f"--frames {spec!r}: package has no frames (only the base state, frame 0)")
        return [None]
    by_id = {int(f["frame"]): f for f in frames}
    if len(by_id) != len(frames):
        raise SystemExit("package frames have duplicate 'frame' numbers")
    if spec == "all":
        return [by_id[k] for k in sorted(by_id)]
    want = [int(x) for x in spec.split(",") if x.strip()]
    missing = [k for k in want if k not in by_id]
    if missing:
        raise SystemExit(f"--frames: {missing} not in package frames {sorted(by_id)}")
    return [by_id[k] for k in want]


def scripts_sha256() -> dict[str, str]:
    return {p.name: bs.sha256_file(p) for p in sorted(HERE.glob("*.py"))}


def cache_key(pkg_files: dict[str, str], render_args: dict[str, Any], scripts: dict[str, str], blender: str) -> str:
    doc = json.dumps({"package_files": pkg_files, "render_args": render_args, "scripts": scripts, "blender": blender},
                     sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(doc.encode()).hexdigest()


def cached(out_dir: Path) -> dict[str, Any] | None:
    m = out_dir / "manifest.json"
    if not m.exists():
        return None
    try:
        man = json.loads(m.read_text())
    except json.JSONDecodeError:
        return None
    if not man.get("complete") or not all((out_dir / r["file"]).exists() for r in man.get("renders", [])):
        return None
    return man


# --- main ---------------------------------------------------------------------------------------


def _diff(base: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cur.items() if base.get(k) != v}


def run(args: argparse.Namespace) -> dict[str, Any]:
    pkg = Path(args.package).resolve()
    sj, _ = bs.load_package(pkg)
    psha, per_file = bs.package_sha256(pkg)
    frames = resolve_frames(sj, args.frames)
    frame_ids = [0 if f is None else int(f["frame"]) for f in frames]
    seeds = parse_seeds(args.seeds)
    max_b = int(args.max_bounces if args.max_bounces is not None else sj.get("render", {}).get("maxBounces", 3))
    render_args = {"spp": int(args.spp), "seeds": seeds, "frames": frame_ids, "max_bounces": max_b, "device": args.device}
    blender = bpy.app.version_string
    scripts = scripts_sha256()
    key = cache_key(per_file, render_args, scripts, blender)
    out_dir = Path(args.out).resolve() / f"{sj.get('name', pkg.name)}-{key[:16]}"

    man = None if args.force else cached(out_dir)
    if man is not None:
        print(f"[render_reference] cache hit {out_dir}", flush=True)
        return {"dir": str(out_dir), "cache_hit": True, "key": key, "renders": len(man["renders"])}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    t_build = time.perf_counter()
    built = bs.build_from_package(pkg)
    t_build = time.perf_counter() - t_build
    scene = built["scene"]
    W, H = scene.render.resolution_x, scene.render.resolution_y
    renders: list[dict[str, Any]] = []
    base_manifest: dict[str, Any] | None = None
    manifest_diffs: dict[str, Any] = {}
    lock_wait = 0.0
    t_all = time.perf_counter()
    with gpu_lock() as waited:
        lock_wait = waited
        for fr, fid in zip(frames, frame_ids):
            bs.apply_frame(built, fr)
            for seed in seeds:
                cseed = cycles_seed(seed, fid)
                cfg = bs.settings_cfg(built, spp=int(args.spp), max_bounces=max_b, device=args.device, seed=cseed)
                t0 = time.perf_counter()
                m = cs.apply_settings(scene, cfg)
                t_set = time.perf_counter() - t0
                if base_manifest is None:
                    base_manifest = m
                else:
                    d = _diff(base_manifest, m)
                    if d:
                        manifest_diffs[f"f{fid:04d}_s{seed:03d}"] = d
                name = f"f{fid:04d}_s{seed:03d}.exr"
                t0 = time.perf_counter()
                bpy.ops.render.render(write_still=False)
                t_render = time.perf_counter() - t0
                t0 = time.perf_counter()
                bpy.data.images["Render Result"].save_render(str(out_dir / name))
                t_save = time.perf_counter() - t0
                info = verify_exr(str(out_dir / name), (H, W))
                renders.append({"frame": fid, "seed": seed, "cycles_seed": cseed, "file": name,
                                "render_s": round(t_render, 4), "save_s": round(t_save, 4), "settings_s": round(t_set, 4),
                                "mean": info["mean"], "state": bs.frame_state(built)})
                print(f"[render_reference] {name} seed={cseed} {t_render:.3f} s", flush=True)
    t_all = time.perf_counter() - t_all

    manifest = {
        "complete": True,
        "generator": "validation/blender/render_reference.py",
        "key": key, "blender": blender, "device": args.device,
        "package": str(pkg), "package_name": sj.get("name"), "package_sha256": psha, "package_files": per_file,
        "scripts_sha256": scripts, "render_args": render_args,
        "seed_rule": "cycles_seed = splitmix64((seed << 32) | frame) & 0x7fffffff",
        "resolution": [W, H],
        "timing": {"build_s": round(t_build, 3), "renders_total_s": round(t_all, 3), "lock_wait_s": round(lock_wait, 3)},
        "build_manifest": built["manifest"],
        "settings_manifest": base_manifest,
        "settings_manifest_diffs": manifest_diffs,
        "renders": renders,
    }
    tmp = out_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, out_dir / "manifest.json")
    return {"dir": str(out_dir), "cache_hit": False, "key": key, "renders": len(renders),
            "renders_total_s": round(t_all, 3), "build_s": round(t_build, 3)}


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--spp", required=True, type=int)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--frames", default="all")
    ap.add_argument("--max-bounces", type=int)
    ap.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    ap.add_argument("--force", action="store_true", help="ignore the cache")
    args = ap.parse_args(argv)
    res = run(args)
    print("[render_reference] RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
