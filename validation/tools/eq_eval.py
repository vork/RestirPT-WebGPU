"""perf2 WP-Q equal-quality evaluation (docs/decisions/perf2-plan.md, WP-Q; driver: validation/harness/run-eq.ts).

ref-merge  merge PT reference batches (run-batches.ts batch_###.pfm, possibly several --batch-offset chunk dirs) into
           ref.pfm (mean) + ref_var.pfm (per-pixel variance of the mean, from the batch-to-batch spread) + ref.json
           (batches, spp, the relMSE noise floor mean Var/(ref^2 + eps)); --delete-batches removes the batch files.
run        metrics of one run-denoise.ts flip run (one config, scene, sequence, seed) against the reference:
             relMSE raw / denoised (denoise_eval.relmse, eps 0.01; plain and 0.1 %-trimmed) at the eval frames
             LDR / HDR FLIP raw / denoised (denoise_eval.flip_pair) at the eval frames
             edge_study.stats on the window frames (display luminance; masks edge / interior / all): std, dt, rmse,
               d2, bias, for dn and raw
             TD-I1 spike events (window frames): onsets of runs of >= 2 consecutive frames with raw luminance
               > 4 x max(ref, floor) (the (1 + c_p)x history spike persisting ~c_p frames), per Mpx and frame; and
               denoised onsets > 1.5 x max(ref, floor) (single-frame), per Mpx and frame
             8x8 block correlation of the raw error (eval frames): e = (raw - ref) / (ref + 0.05 mean ref) luminance,
               winsorised at the 99.9th percentile of |e|; vif8 = 64 Var(8x8 block means) / Var(e) (1 = independent
               pixels; > 1 = errors shared inside tiles, e.g. env-presampled tiles), lag-1 autocorrelation in x and y
           --delete removes the run's PFMs afterwards except --keep (default dn_f63.pfm,raw_f63.pfm of seed 1).
seq        cross-seed metrics of the runs of one (config, scene, sequence), merged into each run's metrics.json:
             tstd   motion-compensated temporal std of the displayed luminance: per pixel, the temporal std over the
                    window of r_t = I_t,s - mean_seeds I_t, x sqrt(n / (n - 1)) (n seeds). The seed mean removes the
                    deterministic camera / light motion; for a static camera it equals the plain temporal std in
                    expectation. Masks edge / interior / all. This is the rule's "temporal std" in every sequence.
             spikes in moving sequences: as in `run`, against the seed mean of the denoised frame t (the PT reference
                    shows the base state only) instead of the PT reference.
           --delete removes the PFMs afterwards except --keep in the first dir.
summary    aggregate the metrics.json files of one configuration: per scene and sequence the seed mean, sd and the 95 %
           t confidence half-width (seed-to-seed, n seeds) of every scalar metric.
decide     the WP-Q rule for candidates against the baseline summary: equal quality iff, in every sequence of a scene,
           denoised LDR-FLIP and denoised motion-compensated temporal std (tstd, all pixels) are not above the baseline's seed-to-seed 95 % CI
           (mean + half-width; below it is better and passes) AND raw relMSE x ms (relMSE averaged over the sequences, ms =
           run-perf frame time, candidate = baseline ms x the same-session ABBA ratio) is not worse than the baseline's
           beyond its seed-to-seed CI (the strict comparison is reported as pass_strict). Writes decision.json and, for
           every scene, a ms-vs-denoised-FLIP Pareto plot (pareto_<scene>.png).

    python eq_eval.py ref-merge --dirs D1,D2 --out-dir REFDIR [--delete-batches]
    python eq_eval.py run --ref-dir REFDIR --dir RUNDIR --eval 16,32,48,63 --window 40:64 --out metrics.json [--delete]
    python eq_eval.py summary --config NAME --metrics 'validation/out/wpq-*/metrics.json' --out summary.json
    python eq_eval.py decide --baseline S.json --candidates S1.json,S2.json --perf perf.json --out decision.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from denoise_eval import LUMA, RELMSE_EPS, edge_masks, flip_pair, flip_version, relmse, relmse_floor  # noqa: E402
from edge_study import all_mask, disp_lum, stats  # noqa: E402
from imageio_util import read_pfm, write_pfm  # noqa: E402

T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 12: 2.201, 16: 2.131}


def t975(n: int) -> float:
    if n < 2:
        return float("nan")
    keys = sorted(T975)
    for k in keys:
        if n <= k:
            return T975[k]
    return 1.96


# ---------------------------------------------------------------------------------------------------- ref-merge
def cmd_ref_merge(a: argparse.Namespace) -> int:
    files: dict[str, Path] = {}
    spp = None
    for d in a.dirs.split(","):
        dd = Path(d)
        meta = json.loads((dd / "meta.json").read_text()) if (dd / "meta.json").exists() else {}
        s = meta.get("sppPerBatch", meta.get("spp"))
        if s is not None:
            if spp is not None and s != spp:
                raise SystemExit(f"{d}: spp {s} != {spp}")
            spp = s
        for f in sorted(dd.glob("batch_*.pfm")):
            if f.name in files:
                raise SystemExit(f"duplicate batch {f.name} in {d} and {files[f.name].parent}")
            files[f.name] = f
    if len(files) < 2:
        raise SystemExit("need >= 2 batches")
    s1 = s2 = None
    for f in files.values():
        x = read_pfm(f).astype(np.float64)
        s1 = x if s1 is None else s1 + x
        s2 = x * x if s2 is None else s2 + x * x
    n = len(files)
    mean = s1 / n
    var_batch = np.maximum(s2 / n - mean * mean, 0) * n / (n - 1)
    var_mean = var_batch / n
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_pfm(out / "ref.pfm", mean.astype(np.float32))
    write_pfm(out / "ref_var.pfm", var_mean.astype(np.float32))
    floor = relmse_floor(mean, var_mean)
    info = {"batches": n, "spp_per_batch": spp, "spp": (spp or 0) * n, "eps": RELMSE_EPS, "noise_floor_relmse": floor,
            "noise_floor_relmse_trim": float(np.sort((var_mean / (mean ** 2 + RELMSE_EPS)).mean(-1).ravel())[: int(mean.shape[0] * mean.shape[1] * 0.999)].mean()),
            "per_batch_relmse": floor * n, "mean_lum": float((mean @ LUMA).mean()), "sources": sorted({str(f.parent) for f in files.values()})}
    (out / "ref.json").write_text(json.dumps(info, indent=1))
    print(f"ref: {n} batches x {spp} spp; noise floor relMSE {floor:.4g} (1 batch: {floor * n:.4g}); mean lum {info['mean_lum']:.4g}")
    if a.delete_batches:
        for f in files.values():
            f.unlink()
        for d in a.dirs.split(","):
            m = Path(d) / "mean.pfm"
            if m.exists():
                m.unlink()
    return 0


# ---------------------------------------------------------------------------------------------------- run
def block_corr(raw: np.ndarray, ref: np.ndarray) -> dict:
    lr, lf = raw @ LUMA, ref @ LUMA
    e = (lr - lf) / (lf + 0.05 * float(lf.mean()))
    c = float(np.quantile(np.abs(e), 0.999))
    e = np.clip(e, -c, c)
    e = e - e.mean()
    v = float(e.var())
    H, W = (e.shape[0] // 8) * 8, (e.shape[1] // 8) * 8
    b = e[:H, :W].reshape(H // 8, 8, W // 8, 8).mean(axis=(1, 3))
    lagx = float((e[:, 1:] * e[:, :-1]).mean() / v)
    lagy = float((e[1:] * e[:-1]).mean() / v)
    return {"vif8": float(64 * b.var() / v), "lag1x": lagx, "lag1y": lagy}


def spikes(raw_l: np.ndarray, dn_l: np.ndarray, ref_l: np.ndarray) -> dict:
    floor = 0.02 * float(ref_l.mean())
    thr = np.maximum(ref_l, floor)
    s = raw_l > 4 * thr[None]                       # [T, H, W]
    persist = s[1:] & s[:-1]                         # spike at t and t-1
    onset = persist[1:] & ~s[:-2]                    # run of >= 2 starting at t-1 (frames t-2 not spiking)
    d = dn_l > 1.5 * thr[None]
    d_on = d[1:] & ~d[:-1]
    mpx = ref_l.size / 1e6
    return {"raw_spike_events_per_mpx_frame": float(onset.sum() / mpx / onset.shape[0]),
            "raw_spike_pixels_frac": float(s.mean()), "dn_spike_onsets_per_mpx_frame": float(d_on.sum() / mpx / d_on.shape[0]),
            "dn_spike_pixels_frac": float(d.mean())}


def cmd_run(a: argparse.Namespace) -> int:
    rd = Path(a.ref_dir)
    ref = read_pfm(rd / "ref.pfm")
    ref_info = json.loads((rd / "ref.json").read_text()) if (rd / "ref.json").exists() else {}
    floor = ref_info.get("noise_floor_relmse")
    d = Path(a.dir)
    meta = json.loads((d / "meta.json").read_text())
    ev = [int(x) for x in a.eval.split(",")]
    f0, f1 = (int(x) for x in a.window.split(":"))
    rows = []
    for t in ev:
        raw, dn = read_pfm(d / f"raw_f{t}.pfm"), read_pfm(d / f"dn_f{t}.pfm")
        if raw.shape != ref.shape:
            raise SystemExit(f"{d}: shape {raw.shape} != ref {ref.shape}")
        l_raw, h_raw, _ = flip_pair(ref, raw)
        l_dn, h_dn, _ = flip_pair(ref, dn)
        rows.append({"frame": t, "relmse_raw": relmse(raw, ref), "relmse_dn": relmse(dn, ref), "relmse_raw_trim": relmse(raw, ref, trim=1e-3),
                     "relmse_dn_trim": relmse(dn, ref, trim=1e-3), "flip_ldr_raw": l_raw, "flip_hdr_raw": h_raw, "flip_ldr_dn": l_dn, "flip_hdr_dn": h_dn,
                     **{f"raw_{k}": v for k, v in block_corr(raw, ref).items()}, "mean_lum_raw": float((raw @ LUMA).mean()), "mean_lum_dn": float((dn @ LUMA).mean())})
    m: dict[str, float] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "frame"}
    m["mean_lum_ref"] = float((ref @ LUMA).mean())
    # window: temporal stability (edge_study) and spikes
    edge, interior = edge_masks(ref)
    masks = {"edge": edge, "interior": interior, "all": all_mask(ref)}
    ref_l = disp_lum(ref)
    win = {}
    for img in ("dn", "raw"):
        fr = np.stack([disp_lum(read_pfm(d / f"{img}_f{t}.pfm")) for t in range(f0, f1)])
        s = stats(fr, ref_l, masks, True, None, f0)
        win[img] = s
        for mk, e in s.items():
            for q, v in e.items():
                if q != "n":
                    m[f"{img}_{mk}_{q}"] = v
    lin = {img: np.stack([read_pfm(d / f"{img}_f{t}.pfm") @ LUMA for t in range(f0, f1)]) for img in ("raw", "dn")}
    m.update(spikes(lin["raw"], lin["dn"], ref @ LUMA))
    if floor is not None:
        m["ref_noise_floor"] = floor
    st = meta.get("renderer", {}).get("settings") or {}
    out = {"run": str(d), "package": meta.get("package"), "seed": meta.get("seed"), "eval": ev, "window": [f0, f1],
           "renderer": meta.get("renderer"), "panOsc": meta.get("panOsc"), "lightOsc": meta.get("lightOsc"), "ok": meta.get("ok"), "errors": meta.get("errors"),
           "settings_key": {k: st.get(k) for k in ("slots", "boostSlots", "risM", "rrMinBounces", "dupmap", "cCap", "maxBounces", "rr")},
           "flip_evaluator": flip_version(), "metrics": m, "rows": rows, "masks_n": {k: int(v.sum()) for k, v in masks.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{d.name}: relMSE raw {m['relmse_raw']:.4g} dn {m['relmse_dn']:.4g} | FLIP LDR raw {m['flip_ldr_raw']:.4f} dn {m['flip_ldr_dn']:.4f} | "
          f"dn std all {m['dn_all_std']:.4f} edge {m['dn_edge_std']:.4f} | spikes {m['raw_spike_events_per_mpx_frame']:.1f}/Mpx/fr | vif8 {m['raw_vif8']:.3f}")
    if a.delete:
        keep = set(a.keep.split(",")) if a.keep else set()
        for f in d.glob("*.pfm"):
            if f.name not in keep:
                f.unlink()
    return 0


# ---------------------------------------------------------------------------------------------------- seq
def cmd_seq(a: argparse.Namespace) -> int:
    ref = read_pfm(Path(a.ref_dir) / "ref.pfm")
    dirs = [Path(d) for d in a.dirs.split(",")]
    f0, f1 = (int(x) for x in a.window.split(":"))
    n = len(dirs)
    edge, interior = edge_masks(ref)
    masks = {"edge": edge, "interior": interior, "all": all_mask(ref)}
    res: list[dict] = [{} for _ in dirs]
    lin_dn = None
    for img in ("dn", "raw"):
        st = np.stack([np.stack([disp_lum(read_pfm(d / f"{img}_f{t}.pfm")).astype(np.float32) for t in range(f0, f1)]) for d in dirs])  # [seed, T, H, W]
        r = st - st.mean(axis=0, keepdims=True)
        sd = r.std(axis=1) * np.sqrt(n / (n - 1)) if n > 1 else np.full(st.shape[0:1] + st.shape[2:], np.nan)
        for k in range(n):
            for mk, m in masks.items():
                res[k][f"{img}_{mk}_tstd"] = float(sd[k][m].mean())
        del st, r, sd
    if a.motion:
        lin_dn = np.stack([np.stack([read_pfm(d / f"dn_f{t}.pfm") @ LUMA for t in range(f0, f1)]) for d in dirs]).astype(np.float32)
        dn_mean = lin_dn.mean(axis=0)
        floor_ref = ref @ LUMA
        for k, d in enumerate(dirs):
            raw = np.stack([read_pfm(d / f"raw_f{t}.pfm") @ LUMA for t in range(f0, f1)]).astype(np.float32)
            thr = np.maximum(dn_mean, 0.02 * float(floor_ref.mean()))
            s = raw > 4 * thr
            onset = (s[1:] & s[:-1])[1:] & ~s[:-2]
            dd = lin_dn[k] > 1.5 * thr
            d_on = dd[1:] & ~dd[:-1]
            mpx = floor_ref.size / 1e6
            res[k].update({"raw_spike_events_per_mpx_frame": float(onset.sum() / mpx / onset.shape[0]), "raw_spike_pixels_frac": float(s.mean()),
                           "dn_spike_onsets_per_mpx_frame": float(d_on.sum() / mpx / d_on.shape[0]), "dn_spike_pixels_frac": float(dd.mean()), "spike_ref": "seed-mean denoised"})
    for k, d in enumerate(dirs):
        mp = d / "metrics.json"
        j = json.loads(mp.read_text())
        j["metrics"].update({q: v for q, v in res[k].items() if not isinstance(v, str)})
        j["seq"] = {"seeds": n, "window": [f0, f1], "dirs": [str(x) for x in dirs], "spike_ref": res[k].get("spike_ref", "PT reference")}
        mp.write_text(json.dumps(j, indent=1))
    print(f"seq {dirs[0].name}..: dn tstd all " + " ".join(f"{r['dn_all_tstd']:.4f}" for r in res))
    if a.delete:
        keep = set(a.keep.split(",")) if a.keep else set()
        for k, d in enumerate(dirs):
            for f in d.glob("*.pfm"):
                if not (k == 0 and f.name in keep):
                    f.unlink()
    return 0


# ---------------------------------------------------------------------------------------------------- summary
def summarize(files: list[Path], config: str) -> dict:
    groups: dict[tuple[str, str], list[dict]] = {}
    for f in files:
        j = json.loads(f.read_text())
        lab = j.get("label") or {}
        groups.setdefault((lab.get("scene", "?"), lab.get("seq", "?")), []).append(j)
    out: dict = {"config": config, "scenes": {}}
    for (sc, seq), runs in sorted(groups.items()):
        keys = sorted(runs[0]["metrics"])
        n = len(runs)
        agg = {}
        for k in keys:
            v = np.array([r["metrics"].get(k, np.nan) for r in runs], float)
            mu, sd = float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if n > 1 else float("nan")
            agg[k] = {"mean": mu, "sd": sd, "ci": t975(n) * sd / np.sqrt(n) if n > 1 else float("nan"), "values": v.tolist()}
        out["scenes"].setdefault(sc, {})[seq] = {"n": n, "seeds": [r.get("seed") for r in runs], "settings": runs[0].get("settings_key"), "metrics": agg}
    return out


def cmd_summary(a: argparse.Namespace) -> int:
    files = sorted(Path(p) for p in glob.glob(a.metrics))
    if not files:
        raise SystemExit(f"no files match {a.metrics}")
    s = summarize(files, a.config)
    Path(a.out).write_text(json.dumps(s, indent=1))
    for sc, seqs in s["scenes"].items():
        for seq, g in seqs.items():
            m = g["metrics"]
            f = lambda k: f"{m[k]['mean']:.4g}±{m[k]['ci']:.2g}"  # noqa: E731
            print(f"{a.config:>14} {sc:>8} {seq:>6} n={g['n']}: relMSE raw {f('relmse_raw')} dn {f('relmse_dn')} | FLIP dn {f('flip_ldr_dn')} raw {f('flip_ldr_raw')} | "
                  f"tstd dn {f('dn_all_tstd')} | spikes {f('raw_spike_events_per_mpx_frame')} | vif8 {f('raw_vif8')}")
    return 0


# ---------------------------------------------------------------------------------------------------- decide
RULE_METRICS = ("flip_ldr_dn", "dn_all_tstd")


def perf_ms(perf: dict, config: str, scene: str) -> float | None:
    """Frame ms of `config`: the baseline's mean, times the same-session paired (ABBA) ratio for a candidate."""
    cs = perf.get("configs", {})
    c, b = cs.get(config, {}).get(scene), cs.get(perf.get("baseline", "baseline"), {}).get(scene)
    if c is None:
        return None
    if b is not None and config != perf.get("baseline", "baseline") and c.get("ratio_vs_baseline") is not None:
        return float(b["frame_ms"]) * float(c["ratio_vs_baseline"])
    return float(c["frame_ms"])


def cmd_decide(a: argparse.Namespace) -> int:
    base = json.loads(Path(a.baseline).read_text())
    cands = [json.loads(Path(p).read_text()) for p in a.candidates.split(",") if p]
    perf = json.loads(Path(a.perf).read_text()) if a.perf else {}
    res: dict = {"rule": "denoised LDR-FLIP and denoised motion-compensated temporal std (tstd, all pixels) <= baseline mean + seed-to-seed 95% CI half-width in every "
                         "sequence, and raw relMSE (mean over sequences) x frame ms <= (baseline relMSE + its seed-to-seed 95% CI) x baseline ms; candidate ms = baseline ms x same-session ABBA ratio", "baseline": base["config"], "candidates": {}}
    for c in cands:
        cr: dict = {"scenes": {}}
        all_ok = True
        for sc, seqs in base["scenes"].items():
            if sc not in c["scenes"]:
                continue
            sr: dict = {"sequences": {}}
            ok = True
            for seq, b in seqs.items():
                cc = c["scenes"][sc].get(seq)
                if cc is None:
                    ok = False
                    sr["sequences"][seq] = {"missing": True}
                    continue
                q = {}
                for k in RULE_METRICS:
                    bm, bci, cm = b["metrics"][k]["mean"], b["metrics"][k]["ci"], cc["metrics"][k]["mean"]
                    q[k] = {"base": bm, "base_ci": bci, "cand": cm, "cand_ci": cc["metrics"][k]["ci"], "rel": cm / bm - 1, "pass": cm <= bm + bci,
                            "inside_ci": abs(cm - bm) <= bci}
                    ok = ok and q[k]["pass"]
                for k in ("relmse_raw", "relmse_dn", "flip_ldr_raw", "flip_hdr_dn", "dn_all_std", "dn_edge_tstd", "dn_edge_d2", "raw_spike_events_per_mpx_frame", "raw_vif8"):
                    q[k] = {"base": b["metrics"][k]["mean"], "base_ci": b["metrics"][k]["ci"], "cand": cc["metrics"][k]["mean"], "rel": cc["metrics"][k]["mean"] / b["metrics"][k]["mean"] - 1 if b["metrics"][k]["mean"] else None}
                sr["sequences"][seq] = q
            # per-seed raw relMSE averaged over the sequences (seed k of every sequence), its mean and 95 % CI half-width
            def seed_avg(summ: dict) -> np.ndarray:
                v = [np.array(summ["scenes"][sc][q]["metrics"]["relmse_raw"]["values"]) for q in seqs if q in summ["scenes"][sc]]
                k = min(len(x) for x in v)
                return np.mean([x[:k] for x in v], axis=0)
            vb, vc = seed_avg(base), seed_avg(c)
            rb, rc = float(vb.mean()), float(vc.mean())
            rb_ci = float(t975(len(vb)) * vb.std(ddof=1) / np.sqrt(len(vb))) if len(vb) > 1 else 0.0
            mb, mc = perf_ms(perf, base["config"], sc), perf_ms(perf, c["config"], sc)
            sr["relmse_raw"] = {"base": rb, "base_ci": rb_ci, "cand": rc, "cand_ci": float(t975(len(vc)) * vc.std(ddof=1) / np.sqrt(len(vc))) if len(vc) > 1 else 0.0}
            sr["ms"] = {"base": mb, "cand": mc}
            if mb is not None and mc is not None:
                # not worse beyond the baseline's seed-to-seed CI (raw relMSE is heavy-tailed: fireflies), strict ratio reported
                sr["relmse_x_ms"] = {"base": rb * mb, "base_ci": rb_ci * mb, "cand": rc * mc, "rel": rc * mc / (rb * mb) - 1,
                                     "pass": rc * mc <= (rb + rb_ci) * mb, "pass_strict": rc * mc <= rb * mb}
                ok = ok and sr["relmse_x_ms"]["pass"]
            else:
                sr["relmse_x_ms"] = {"pass": None, "note": "no run-perf timing"}
                ok = False
            sr["equal_quality"] = ok
            all_ok = all_ok and ok
            cr["scenes"][sc] = sr
        cr["equal_quality_all_scenes"] = all_ok
        res["candidates"][c["config"]] = cr
    Path(a.out).write_text(json.dumps(res, indent=1))
    for name, cr in res["candidates"].items():
        for sc, sr in cr["scenes"].items():
            parts = []
            for seq, q in sr["sequences"].items():
                if q.get("missing"):
                    parts.append(f"{seq} MISSING")
                    continue
                parts.append(f"{seq}: FLIPdn {q['flip_ldr_dn']['rel']:+.1%}{'' if q['flip_ldr_dn']['pass'] else '!'} tstd {q['dn_all_tstd']['rel']:+.1%}{'' if q['dn_all_tstd']['pass'] else '!'}")
            x = sr["relmse_x_ms"]
            xs = f"relMSE×ms {x['rel']:+.1%}" if "rel" in x else "relMSE×ms n/a"
            print(f"{name:>14} {sc:>8}: {'EQUAL' if sr['equal_quality'] else 'NOT EQUAL':>9} | {' | '.join(parts)} | relMSE {sr['relmse_raw']['cand'] / sr['relmse_raw']['base'] - 1:+.1%} ms {sr['ms']['cand']} vs {sr['ms']['base']} {xs}")
    if a.plot_dir:
        pareto(base, cands, perf, Path(a.plot_dir))
    return 0


def pareto(base: dict, cands: list[dict], perf: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    for sc in base["scenes"]:
        fig, axs = plt.subplots(1, len(base["scenes"][sc]), figsize=(4.2 * len(base["scenes"][sc]), 3.6), squeeze=False)
        for ax, seq in zip(axs[0], base["scenes"][sc]):
            for s in [base, *cands]:
                g = s["scenes"].get(sc, {}).get(seq)
                ms = perf_ms(perf, s["config"], sc)
                if g is None or ms is None:
                    continue
                m = g["metrics"]["flip_ldr_dn"]
                ax.errorbar(ms, m["mean"], yerr=m["ci"], fmt="o" if s is base else "s", capsize=3, label=s["config"])
                ax.annotate(s["config"], (ms, m["mean"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_title(f"{sc} / {seq}")
            ax.set_xlabel("frame ms (run-perf, 540p N3)")
            ax.set_ylabel("denoised LDR-FLIP (mean ± 95% CI)")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"pareto_{sc}.png", dpi=120)
        plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ref-merge")
    r.add_argument("--dirs", required=True)
    r.add_argument("--out-dir", dest="out_dir", required=True)
    r.add_argument("--delete-batches", dest="delete_batches", action="store_true")
    u = sub.add_parser("run")
    u.add_argument("--ref-dir", dest="ref_dir", required=True)
    u.add_argument("--dir", required=True)
    u.add_argument("--eval", default="16,32,48,63")
    u.add_argument("--window", default="40:64")
    u.add_argument("--label", help="JSON {config, scene, seq, seed} stored in metrics.json")
    u.add_argument("--out", required=True)
    u.add_argument("--delete", action="store_true")
    u.add_argument("--keep", default="")
    q = sub.add_parser("seq")
    q.add_argument("--ref-dir", dest="ref_dir", required=True)
    q.add_argument("--dirs", required=True)
    q.add_argument("--window", default="40:64")
    q.add_argument("--motion", action="store_true")
    q.add_argument("--delete", action="store_true")
    q.add_argument("--keep", default="")
    s = sub.add_parser("summary")
    s.add_argument("--config", required=True)
    s.add_argument("--metrics", required=True, help="glob of metrics.json files")
    s.add_argument("--out", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--baseline", required=True)
    d.add_argument("--candidates", required=True)
    d.add_argument("--perf")
    d.add_argument("--out", required=True)
    d.add_argument("--plot-dir", dest="plot_dir")
    a = p.parse_args()
    if a.cmd == "run" and a.label:
        rc = cmd_run(a)
        j = json.loads(Path(a.out).read_text())
        j["label"] = json.loads(a.label)
        Path(a.out).write_text(json.dumps(j, indent=1))
        return rc
    return {"ref-merge": cmd_ref_merge, "run": cmd_run, "seq": cmd_seq, "summary": cmd_summary, "decide": cmd_decide}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
