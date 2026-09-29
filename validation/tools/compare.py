#!/usr/bin/env python3
"""compare.py — replicate-based statistical comparison of two image sets (plan §5 M2, §7.3).

Modes
  compare (default)  compare.py --ours DIR --ref DIR --test test.json --out DIR [--frame F] [--rerun-of report.json]
  curve              compare.py --curve --ours DIR --ref DIR --test test.json --out DIR
  calibrate (Gate 1) compare.py --calibrate --ref DIR --test test.json --out DIR [--splits 20] [--repeats 10]
                     [--planted DIR]   (+ a rendered plant on disjoint seeds that must be detected vs --ref)

Input directories (one replicate set each; the first matching layout wins):
  batch_###.{pfm,exr}  PT batch means (ours)             + meta.json
  run_###.{pfm,exr}    ReSTIR ensemble runs (images)      + meta.json
  seed_###.{exr,pfm}   Cycles seeds                        + manifest.json
  f{frame}_s{seed}.{exr,pfm}  per-frame seeds (select with --frame or test.json "frame")
  *.npz                pre-reduced ensemble aggregates (format in README.md)

Outputs in --out: report.json, PNG figures, index.html. Exit code 0 = every gating check passed,
1 = a gating check failed (or a confirmatory re-run is required), 2 = usage / input error.
See validation/tools/README.md for the test.json schema and statistical design decisions.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats as S  # noqa: E402
from imageio_util import read_image  # noqa: E402

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("batch", re.compile(r"^batch_(\d+)\.(pfm|exr)$", re.I)),
    ("run", re.compile(r"^run_(\d+)\.(pfm|exr)$", re.I)),
    ("seed", re.compile(r"^seed_(\d+)\.(pfm|exr)$", re.I)),
    ("frame_seed", re.compile(r"^f(\d+)_s(\d+)\.(pfm|exr)$", re.I)),
]
META_NAMES = ("meta.json", "manifest.json")


class InputError(Exception):
    pass


# ------------------------------------------------------------------------------------------
# Input discovery and loading
# ------------------------------------------------------------------------------------------

@dataclass
class Source:
    dir: Path
    kind: str
    files: list[tuple[int, Path]] = field(default_factory=list)
    seeds: list | None = None
    meta: dict = field(default_factory=dict)
    meta_file: Path | None = None
    npz: Path | None = None
    frame: int | None = None

    def describe(self) -> dict:
        d = dict(dir=str(self.dir), kind=self.kind, n=len(self.files) if self.files else None,
                 meta_file=None if self.meta_file is None else self.meta_file.name)
        if self.frame is not None:
            d["frame"] = self.frame
        if self.npz is not None:
            d["npz"] = self.npz.name
        if self.files:
            d["first"], d["last"] = self.files[0][1].name, self.files[-1][1].name
        return d


def discover(d: Path, frame: int | None = None) -> Source:
    d = Path(d)
    if not d.is_dir():
        raise InputError(f"{d}: not a directory")
    meta, meta_file = {}, None
    for name in META_NAMES:
        if (d / name).is_file():
            meta_file = d / name
            meta = json.loads(meta_file.read_text())
            break
    names = sorted(p.name for p in d.iterdir() if p.is_file())
    for kind, pat in PATTERNS:
        hits = [(pat.match(n), n) for n in names]
        hits = [(m, n) for m, n in hits if m]
        if not hits:
            continue
        if kind == "frame_seed":
            frames = sorted({int(m.group(1)) for m, _ in hits})
            if frame is None:
                if len(frames) != 1:
                    raise InputError(f"{d}: frames {frames} present; select one with --frame or test.json 'frame'")
                frame = frames[0]
            hits = [(m, n) for m, n in hits if int(m.group(1)) == frame]
            if not hits:
                raise InputError(f"{d}: no files for frame {frame}")
            files = sorted((int(m.group(2)), d / n) for m, n in hits)
            seeds = [i for i, _ in files]
        else:
            files = sorted((int(m.group(1)), d / n) for m, n in hits)
            seeds = [i for i, _ in files] if kind == "seed" else None
        if isinstance(meta.get("seeds"), list) and len(meta["seeds"]) == len(files):
            seeds = list(meta["seeds"])
        return Source(d, kind, files, seeds, meta, meta_file, frame=frame)
    npzs = sorted(d.glob("*.npz"))
    if npzs:
        pick = d / "ensemble.npz" if (d / "ensemble.npz").is_file() else npzs[0]
        seeds = meta.get("seeds") if isinstance(meta.get("seeds"), list) else None
        return Source(d, "npz", [], seeds, meta, meta_file, npz=pick, frame=frame)
    raise InputError(f"{d}: no batch_/run_/seed_/f*_s* images or .npz found")


def load_replicates(src: Source, channels, tile_sizes, masks=None, mask_names=(), checkpoints=(),
                    halves: bool = False) -> tuple[S.Replicates, dict | None]:
    """Stream the replicate images into aggregates. With halves=True also returns per-pixel sums of
    the first and second half (for the A/A noise-floor metrics)."""
    if src.kind == "npz":
        with np.load(src.npz, allow_pickle=False) as z:
            reps = S.replicates_from_sums({k: z[k] for k in z.files}, channels)
        if masks is not None and reps.masks is None:
            raise InputError(f"{src.npz}: test.json declares masks but the npz has no 'masks' sums")
        return reps, None
    acc = S.ReplicateAccumulator(channels, tile_sizes, masks, mask_names, checkpoints)
    n = len(src.files)
    half = {"a": None, "b": None, "n": n // 2} if halves and n >= 4 else None
    for i, (_, path) in enumerate(src.files):
        img = read_image(path, drop_alpha=True)
        if img.shape[-1] > 3:
            img = img[..., :3]
        acc.add(img)
        if half is not None and i < 2 * half["n"]:
            key = "a" if i < half["n"] else "b"
            x = S.channelize(img, channels)
            half[key] = x.copy() if half[key] is None else half[key] + x
    return acc.finish(), half


def load_stack(src: Source) -> np.ndarray:
    if src.kind == "npz":
        raise InputError("calibration needs replicate images, not pre-reduced .npz aggregates")
    imgs = []
    for _, p in src.files:
        a = read_image(p, drop_alpha=True)
        imgs.append(a[..., :3] if a.shape[-1] > 3 else a)
    return np.stack(imgs)


def load_masks(test: dict, test_dir: Path, h: int, w: int) -> tuple[np.ndarray | None, tuple[str, ...]]:
    specs = test.get("masks") or []
    if not specs:
        return None, ()
    out, names = [], []
    for i, m in enumerate(specs):
        name = m.get("name", f"mask{i}")
        if "rect" in m:
            x0, y0, x1, y1 = (int(v) for v in m["rect"])
            a = np.zeros((h, w), bool)
            a[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w)] = True
        elif "file" in m:
            img = read_image(test_dir / m["file"], drop_alpha=False)
            if img.shape[:2] != (h, w):
                raise InputError(f"mask {name}: shape {img.shape[:2]} != image {(h, w)}")
            a = img[..., 0] > 0.5
        else:
            raise InputError(f"mask {name}: needs 'rect' [x0,y0,x1,y1] or 'file'")
        if not a.any():
            raise InputError(f"mask {name}: empty")
        out.append(a)
        names.append(name)
    return np.stack(out), tuple(names)


def source_channels(src: Source) -> int:
    if src.kind == "npz":
        with np.load(src.npz, allow_pickle=False) as z:
            names = [str(c) for c in z["channels"].tolist()] if "channels" in z.files else ["R", "G", "B"]
        return 3 if all(c in names for c in "RGB") else 1
    return min(read_image(src.files[0][1], drop_alpha=True).shape[-1], 3)


def image_size(src: Source) -> tuple[int, int]:
    if src.kind == "npz":
        with np.load(src.npz, allow_pickle=False) as z:
            if "pixel_sum" in z.files:
                return tuple(z["pixel_sum"].shape[:2])
            return int(z["height"]), int(z["width"])
    return read_image(src.files[0][1], drop_alpha=True).shape[:2]


# ------------------------------------------------------------------------------------------
# test.json -> GateSpec
# ------------------------------------------------------------------------------------------

def build_spec(test: dict) -> tuple[S.GateSpec, dict]:
    stage = test.get("stage", "A")
    if stage not in S.STAGE_DEFAULTS:
        raise InputError(f"test.json stage {stage!r} not in {sorted(S.STAGE_DEFAULTS)}")
    dflt = S.STAGE_DEFAULTS[stage]
    delta = test.get("delta") or {}
    ov = dict(delta_global=delta.get("global"), delta_tile=delta.get("tile"), delta_mask=delta.get("mask"),
              tile=test.get("tile"), alpha_tost=test.get("alpha"), n_units=test.get("n_units"),
              alpha_suite=test.get("alpha_suite"), min_replicates=test.get("min_replicates"),
              dark_frac=test.get("dark_frac"), zero_tol=test.get("zero_tol"), power=test.get("power"),
              fdr_q=test.get("fdr_q"), num_eps=test.get("num_eps"))
    if test.get("channels"):
        ov["channels"] = tuple(test["channels"])
    if test.get("checks"):
        ov["gating"] = dict(test["checks"])
    spec = S.GateSpec.for_stage(stage, **ov)
    notes = {}
    loosened = {k: dict(default=dflt[f"delta_{k}"], used=getattr(spec, f"delta_{k}"))
                for k in ("global", "tile", "mask") if getattr(spec, f"delta_{k}") > dflt[f"delta_{k}"]}
    if loosened:
        notes["delta_loosened"] = loosened
        print(f"WARNING: δ loosened vs stage {stage} defaults: {loosened}", file=sys.stderr)
    if spec.tile != dflt["tile"]:
        notes["aggregate_enlarged" if spec.tile > dflt["tile"] else "aggregate_reduced"] = dict(
            default_tile=dflt["tile"], used_tile=spec.tile, reason=test.get("aggregate_note"))
    if spec.alpha_tost > 0.01:
        notes["alpha_loosened"] = spec.alpha_tost
    if spec.num_eps > 0:
        notes["numeric_floor"] = dict(num_eps=spec.num_eps, rule="tiles with SE_delta <= num_eps*denominator leave the "
                                      "Δ = 0 rejection family and must match |Δ| <= 2*num_eps*denominator + z_sidak*SE_delta; TOST unchanged",
                                      reason=test.get("num_eps_note"))
    disabled = [k for k, v in spec.gating.items() if not v]
    if disabled:
        notes["non_gating_checks"] = disabled
    return spec, notes


# ------------------------------------------------------------------------------------------
# Provenance
# ------------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _collect(meta, pred, prefix="") -> dict:
    out = {}
    if isinstance(meta, dict):
        for k, v in meta.items():
            key = f"{prefix}{k}"
            if pred(str(k)) and not isinstance(v, (dict, list)):
                out[key] = v
            elif isinstance(v, dict):
                out.update(_collect(v, pred, key + "."))
    return out


def provenance(src: Source) -> dict:
    d = dict(meta_file=None if src.meta_file is None else str(src.meta_file),
             meta_sha256=None if src.meta_file is None else _sha256(src.meta_file),
             hashes=_collect(src.meta, lambda k: "hash" in k.lower()),
             versions=_collect(src.meta, lambda k: "version" in k.lower() or k.lower() in ("blender", "chrome", "commit", "git")))
    if src.npz is not None:
        d["npz_sha256"] = _sha256(src.npz)
    return d


def seed_labels(src: Source) -> list[str] | None:
    return None if src.seeds is None else [str(s) for s in src.seeds]


# ------------------------------------------------------------------------------------------
# JSON helpers
# ------------------------------------------------------------------------------------------

def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if math.isnan(x):
            return None
        if math.isinf(x):
            return 1e308 if x > 0 else -1e308
        return x
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(jsonable(obj), indent=1, allow_nan=False) + "\n")


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _ch(reps: S.Replicates, names) -> list[int]:
    return [reps.channels.index(c) for c in names]


# ------------------------------------------------------------------------------------------
# Modes
# ------------------------------------------------------------------------------------------

def run_compare(args, test: dict, test_dir: Path, out: Path) -> int:
    from report import plots, render_index

    spec, notes = build_spec(test)
    frame = args.frame if args.frame is not None else test.get("frame")
    so, sr = discover(args.ours, frame), discover(args.ref, frame)
    h, w = image_size(sr)
    if image_size(so) != (h, w):
        raise InputError(f"image size mismatch: ours {image_size(so)} vs ref {(h, w)}")
    masks, mask_names = load_masks(test, test_dir, h, w)
    # always aggregate Y (metrics) and, for colour sources, R/G/B (FLIP, figures); gate on spec.channels
    channels = ("Y", "R", "G", "B") if source_channels(sr) >= 3 and source_channels(so) >= 3 else ("Y",)
    missing = [c for c in spec.channels if c not in channels]
    if missing:
        raise InputError(f"channels {missing} requested but the inputs are single-channel")
    tiles = sorted({16, 32, 64, spec.tile})
    ours, _ = load_replicates(so, channels, tiles, masks, mask_names)
    ref, half = load_replicates(sr, channels, tiles, masks, mask_names, halves=True)

    res = S.evaluate_gate(ours, ref, spec)
    report: dict = dict(tool="compare.py", version=1, mode="compare", created=_now(),
                        test=dict(test, path=str(args.test), sha256=_sha256(args.test)),
                        tier=test.get("tier", "tight"), notes=notes, spec=spec.to_json(),
                        inputs=dict(ours=so.describe(), ref=sr.describe()),
                        provenance=dict(ours=provenance(so), ref=provenance(sr)),
                        seeds=dict(ours=seed_labels(so), ref=seed_labels(sr)),
                        gate_passed=res.passed, checks=res.checks, failed_checks=res.summary["failed_checks"],
                        channels=res.summary["channels"], alpha_u=res.summary["alpha_u"])
    report["mdb"] = {ch: dict(global_=v["global_"]["mdb"], tile_max=v["tiles"]["mdb_max"],
                              tile_median=v["tiles"]["mdb_median"],
                              masks={m["name"]: m["mdb"] for m in v["masks"]},
                              powered_global=(v["global_"]["mdb"] or 0) <= spec.delta_global,
                              powered_tiles=(v["tiles"]["mdb_max"] or 0) <= spec.delta_tile)
                     for ch, v in res.summary["channels"].items()}
    report["sizing"] = {ch: dict(global_se_rel=v["global_"]["se_rel"], global_target=v["global_"]["sizing_target"],
                                 global_ratio=v["global_"]["sizing_ratio"],
                                 tile_target=v["tiles"]["sizing_target"], tile_ratio_max=v["tiles"]["sizing_ratio_max"],
                                 replicate_multiplier_needed=max(v["global_"]["replicate_multiplier_needed"] or 1,
                                                                 v["tiles"]["replicate_multiplier_needed"] or 1))
                        for ch, v in res.summary["channels"].items()}

    # ---- metrics on converged means (Y) and figures
    images: dict[str, str] = {}
    px, pr = ours.pixel_summary(), ref.pixel_summary()
    y = channels.index("Y")
    metrics: dict = {}
    if px is not None and pr is not None:
        xm, rm = px.mean, pr.mean
        metrics.update(relMSE=S.rel_mse(xm[..., y], rm[..., y]),
                       relMSE_corr=S.rel_mse_corr(xm[..., y], rm[..., y], pr.se[..., y]),
                       MAPE=S.mape(xm[..., y:y + 1], rm[..., y:y + 1]),
                       RMSRB=S.rmsrb(xm[..., y], rm[..., y], px.se[..., y], pr.se[..., y]))
        rgb = [channels.index(c) for c in "RGB"] if all(c in channels for c in "RGB") else [y]
        try:
            fmean, fmap = S.hdr_flip(rm[..., rgb], xm[..., rgb])
            metrics["HDR_FLIP"] = fmean
            images["HDR-FLIP map"] = plots.flip_map(fmap, fmean, out / "flip.png")
        except Exception as e:  # FLIP is a report metric; never let it break the gate
            metrics["HDR_FLIP_error"] = repr(e)
        if half is not None:
            ha, hb = half["a"] / half["n"], half["b"] / half["n"]
            nf = dict(half_size=half["n"], relMSE=S.rel_mse(ha[..., y], hb[..., y]),
                      MAPE=S.mape(ha[..., y:y + 1], hb[..., y:y + 1]))
            try:
                nf["HDR_FLIP"] = S.hdr_flip(hb[..., rgb], ha[..., rgb])[0]
            except Exception as e:
                nf["HDR_FLIP_error"] = repr(e)
            metrics["noise_floor_ref_halves"] = nf
        images["side by side"] = plots.side_by_side(rm[..., rgb], xm[..., rgb], out / "side_by_side.png")
        denom = np.maximum(np.abs(rm[..., y]), spec.dark_frac * abs(float(np.mean(rm[..., y]))))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(denom > 0, (xm[..., y] - rm[..., y]) / np.where(denom > 0, denom, 1), 0.0)
        images["signed relative difference"] = plots.rel_diff(rel, spec.delta_tile, out / "rel_diff.png",
                                                              f"Y, clipped ±{100 * spec.delta_tile:g}%")
        _, _, tpx, nupx = S.welch(px.mean[..., y], px.se[..., y], px.nu[..., y], pr.mean[..., y], pr.se[..., y], pr.nu[..., y])
        if np.any(np.isfinite(tpx)):
            ppx = S.p_two_sided(tpx, nupx)
            thr = float(np.nanmedian(S.sidak_threshold(res.summary["alpha_u"], h * w, nupx[np.isfinite(nupx)])))
            images["|t| map"] = plots.t_map(np.abs(tpx), thr, S.bh_reject(ppx, spec.fdr_q), out / "t_map.png", "Y")
    report["metrics"] = metrics
    a = res.arrays["Y"] if "Y" in res.arrays else next(iter(res.arrays.values()))
    images["tile t heatmap"] = plots.tile_t(a["tile_t"], a["tile_pass"], a["tile_sidak"], spec.tile, out / "tile_t.png", "Y")
    images["z histogram + QQ"] = plots.z_hist_qq(a["tile_t"], a["tile_nu"], out / "z_hist_qq.png", "Y")
    report["images"] = images

    # ---- suite decision (confirmatory re-run rule)
    if args.rerun_of:
        prev = json.loads(Path(args.rerun_of).read_text())
        pseeds = prev.get("seeds") or {}
        first = None if pseeds.get("ours") is None or pseeds.get("ref") is None else \
            [f"ours:{s}" for s in pseeds["ours"]] + [f"ref:{s}" for s in pseeds["ref"]]
        mine = None if so.seeds is None or sr.seeds is None else \
            [f"ours:{s}" for s in so.seeds] + [f"ref:{s}" for s in sr.seeds]
        dec = S.confirmatory_decision(bool(prev.get("gate_passed")), res.passed, first, mine)
        dec["rerun_of"] = str(args.rerun_of)
    else:
        dec = S.confirmatory_decision(res.passed)
    report["decision"] = dec
    report["status"] = dec["status"]
    write_json(out / "report.json", report)
    render_index(jsonable(report), out)
    return 0 if dec["passed"] else 1


def run_curve(args, test: dict, test_dir: Path, out: Path) -> int:
    from report import plots, render_index

    spec, notes = build_spec(test)
    frame = args.frame if args.frame is not None else test.get("frame")
    so, sr = discover(args.ours, frame), discover(args.ref, frame)
    if so.kind == "npz":
        raise InputError("--curve needs per-batch images in --ours")
    n = len(so.files)
    ks = [2 ** k for k in range(int(math.log2(n)) + 1)] if n >= 1 else []
    channels = ("Y",)
    ours, _ = load_replicates(so, channels, (spec.tile,), checkpoints=ks)
    ref, _ = load_replicates(sr, channels, (spec.tile,))
    pr = ref.pixel_summary()
    gr = S.summarize(ref.glob)
    rows = []
    for N in ks:
        xm = ours.prefix[N][..., 0] / N
        rmc = S.rel_mse_corr(xm, pr.mean[..., 0], pr.se[..., 0])
        mc = S.mse_corr(xm, pr.mean[..., 0], pr.se[..., 0])
        row = dict(N=N, relMSE_corr=rmc, relMSE=S.rel_mse(xm, pr.mean[..., 0]), MSE_corr=mc, N_MSE_corr=N * mc,
                   N_relMSE_corr=N * rmc, BNR=None)
        if N >= 2:
            gx = S.summarize(ours.glob[:N])
            d, se, _, _ = S.welch(gx.mean[0], gx.se[0], gx.nu[0], gr.mean[0], gr.se[0], gr.nu[0])
            row["BNR"] = float(S.bnr(d, se))
        rows.append(row)
    fit = S.slope_fit([r["N"] for r in rows], [r["relMSE_corr"] for r in rows])
    cfg = test.get("curve") or {}
    lo, hi = cfg.get("slope_range", [-1.1, -0.9])
    slope_ok = fit["slope"] is not None and lo <= fit["slope"] <= hi
    gating = bool(cfg.get("gate", False))
    checks = [dict(name="curve_slope", channel="Y", passed=slope_ok, gating=gating, value=fit["slope"],
                   threshold=[lo, hi])]
    passed = slope_ok or not gating
    images = dict(convergence=plots.curve([r["N"] for r in rows], [r["relMSE_corr"] for r in rows],
                                          [r["N_MSE_corr"] for r in rows],
                                          [np.nan if r["BNR"] is None else r["BNR"] for r in rows], fit,
                                          out / "curve.png"))
    report = dict(tool="compare.py", version=1, mode="curve", created=_now(),
                  test=dict(test, path=str(args.test), sha256=_sha256(args.test)), notes=notes,
                  inputs=dict(ours=so.describe(), ref=sr.describe()),
                  provenance=dict(ours=provenance(so), ref=provenance(sr)),
                  curve=dict(points=rows, fit=fit, slope_range=[lo, hi], gating=gating),
                  checks=checks, failed_checks=[] if passed else ["curve_slope[Y]"],
                  gate_passed=passed, status="pass" if passed else "fail", images=images)
    write_json(out / "report.json", report)
    render_index(jsonable(report), out)
    return 0 if passed else 1


def default_plants(stage: str) -> list[dict]:
    if stage == "A":
        return [dict(name="light x1.0075", kind="scale", factor=1.0075),
                dict(name="32x32 region +3%", kind="region", factor=1.03, size=32)]
    return [dict(name="W x1.003", kind="scale", factor=1.003)]


def rendered_plant(args, sr: Source, spec: S.GateSpec, stack: np.ndarray, masks, mask_names, frame) -> dict:
    """--planted: a separately rendered planted scene (e.g. Cycles light x1.0075) vs --ref. The seed sets must be
    known and disjoint (independent replicates; a shared seed would correlate the two sides and void Welch)."""
    sp = discover(args.planted, frame)
    rs, ps = seed_labels(sr), seed_labels(sp)
    if rs is None or ps is None:
        raise InputError("--planted: seeds unknown on one side (need seed_###/f*_s* names or meta 'seeds')")
    common = sorted(set(rs) & set(ps))
    if common:
        raise InputError(f"--planted: seed sets overlap with --ref ({common[:8]}); render the plant on disjoint seeds")
    pstack = load_stack(sp)
    if pstack.shape[1:] != stack.shape[1:]:
        raise InputError(f"--planted: image shape {pstack.shape[1:]} != ref {stack.shape[1:]}")
    tiles = sorted({16, 32, 64, spec.tile})
    ref_r = S.aggregate_stack(stack, spec.channels, tiles, masks, mask_names)
    pl_r = S.aggregate_stack(pstack, spec.channels, tiles, masks, mask_names)
    r = S.rendered_plant_detection(ref_r, pl_r, spec, n_repeats=args.repeats, rng=args.seed + 101)
    r.update(name=args.plant_name or sp.meta.get("package_name") or sp.dir.name, seeds_ref=rs, seeds_planted=ps,
             _source=sp.describe(), _provenance=provenance(sp))
    return r


def run_calibrate(args, test: dict, test_dir: Path, out: Path) -> int:
    from report import render_index

    spec, notes = build_spec(test)
    frame = args.frame if args.frame is not None else test.get("frame")
    sr = discover(args.ref, frame)
    stack = load_stack(sr)
    h, w = stack.shape[1:3]
    masks, mask_names = load_masks(test, test_dir, h, w)
    tiles = sorted({16, 32, 64, spec.tile})
    reps = S.aggregate_stack(stack, spec.channels, tiles, masks, mask_names)
    aa = S.aa_split(reps, spec, n_splits=args.splits, rng=args.seed)
    cal = test.get("calibration") or {}
    # an explicit "plants": [] disables the synthetic plants (e.g. a rendered-plant-only calibration)
    plants_cfg = cal["plants"] if isinstance(cal.get("plants"), list) else default_plants(spec.stage)
    ymean = S.tile_means(S.channelize(stack.mean(axis=0), ("Y",)), 32)[..., 0]
    results = []
    ok = aa["ok"]
    for i, p in enumerate(plants_cfg):
        kind, factor = p["kind"], float(p["factor"])
        region = None
        if kind == "region":
            size = int(p.get("size", 32))
            if "x0" in p and "y0" in p:
                region = (int(p["x0"]), int(p["y0"]), size)
            else:  # brightest 32-aligned block (never a dark tile)
                ty, tx = np.unravel_index(int(np.argmax(ymean)), ymean.shape)
                region = (int(tx) * 32, int(ty) * 32, size)
        ch = None if kind != "channel" else int(p.get("channel", 0))
        r = S.plant_detection(stack, spec, lambda s, k=kind, f=factor, rg=region, c=ch: S.plant(k, s, factor=f, region=rg, channel=c),
                              n_repeats=args.repeats, rng=args.seed + 1 + i, masks=masks, mask_names=mask_names)
        r.update(name=p.get("name", f"{kind} x{factor}"), kind=kind, factor=factor, region=region)
        results.append(r)
        ok &= r["calibrated"]
    rendered = None
    inputs, prov = dict(ref=sr.describe()), dict(ref=provenance(sr))
    if args.planted is not None:
        rendered = rendered_plant(args, sr, spec, stack, masks, mask_names, frame)
        inputs["planted"], prov["planted"] = rendered.pop("_source"), rendered.pop("_provenance")
        ok &= rendered["calibrated"]
    report = dict(tool="compare.py", version=1, mode="calibrate", created=_now(),
                  test=dict(test, path=str(args.test), sha256=_sha256(args.test)), notes=notes, spec=spec.to_json(),
                  inputs=inputs, provenance=prov,
                  calibration=dict(aa=aa, plants=results, rendered_plant=rendered), gate_passed=bool(ok),
                  status="pass" if ok else "fail",
                  failed_checks=([] if aa["ok"] else ["aa_split"]) + [f"plant:{r['name']}" for r in results if not r["calibrated"]]
                  + ([] if rendered is None or rendered["calibrated"] else [f"rendered_plant:{rendered['name']}"]))
    write_json(out / "report.json", report)
    render_index(jsonable(report), out)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", type=Path, help="our replicate directory")
    ap.add_argument("--ref", type=Path, required=True, help="reference replicate directory")
    ap.add_argument("--test", type=Path, required=True, help="test.json")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--frame", type=int, default=None, help="frame for f{frame}_s{seed} inputs")
    ap.add_argument("--rerun-of", type=Path, default=None, help="report.json of the failed first run (confirmatory re-run)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--curve", action="store_true", help="convergence mode (prefix means at N = 2^k)")
    mode.add_argument("--calibrate", action="store_true", help="Gate 1: A/A re-splits + planted biases on --ref")
    ap.add_argument("--planted", type=Path, default=None,
                    help="calibrate: separately rendered planted replicates (disjoint seeds) that must be detected vs --ref")
    ap.add_argument("--plant-name", default=None, help="label of the --planted plant in report.json")
    ap.add_argument("--splits", type=int, default=20, help="A/A re-splits (>= 20)")
    ap.add_argument("--repeats", type=int, default=10, help="plant repeats (detect >= 9/10)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for re-splits")
    args = ap.parse_args(argv)
    try:
        test = json.loads(args.test.read_text())
        args.out.mkdir(parents=True, exist_ok=True)
        if args.calibrate:
            return run_calibrate(args, test, args.test.parent, args.out)
        if args.ours is None:
            ap.error("--ours is required unless --calibrate")
        if args.curve:
            return run_curve(args, test, args.test.parent, args.out)
        return run_compare(args, test, args.test.parent, args.out)
    except (InputError, FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"compare.py: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
