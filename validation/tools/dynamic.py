#!/usr/bin/env python3
"""Sequence-level statistics of the dynamic rung 3.6 (restir-temporal-api.md §6.4; PLAN §7.3 dynamic). OWNER T-E.

The per-frame gate is compare.py stage `dyn` (TOST 0.2 % global, 2 % per 64² tile, 3 % per mask region, per test
frame). This tool adds, per sequence unit (one chain run serving every test frame):
  series  {Delta_f / SE_f} of the global relative difference per channel over the test frames (reported);
  drift   OLS regression of the relative global Delta_f on the frame number f, with the SANDWICH covariance: the
          ReSTIR rows of different test frames come from the SAME chains (covariance estimated across chains), the PT
          references of different frames are independent. Gating: |slope| / SE(slope) <= z_{1-alpha_u/2} (alpha_u = the
          suite FWER unit level, PLAN §7.3 "applies to every rejection-type check"); the 99 % CI is reported.
  tiles   per frame and channel the number k of gate tiles whose Welch test rejects Delta = 0 at alpha' = 0.01, against
          Binomial(m, alpha'): gating P(K >= k) >= alpha_u.
Also: merge of ensemble.npz chunks (GPU-lock chunks of one unit are concatenated in batch order), and `calibrate` on
one chain run (the §6.5 synthetic W x 1.003 and the A/A re-splits; compare.py --calibrate needs images, chains are
ensemble rows): >= 20 A/A re-splits of the rows (stats.aa_split) and, in >= 9/10 repeats, halves A/B with A x 1.003
failing against B while A vs B passes.
Usage:
  dynamic.py sequence --spec seq.json --out OUT      seq.json: {"test": test.json, "n_units": N,
                                                                 "frames": [{"frame": f, "ours": DIR, "ref": DIR}, ...]}
  dynamic.py merge-npz --out DIR CHUNK_DIR...       (f<t>/ensemble.npz of each chunk -> DIR/ensemble.npz; meta seeds merged)
  dynamic.py calibrate --ours NPZ_DIR --test test.json --out OUT [--factor 1.003] [--splits 20] [--repeats 10]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.stats as st

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import stats as S  # noqa: E402
from compare import build_spec, discover, load_masks, load_replicates, write_json  # noqa: E402

ALPHA_TILE = 0.01


def drift(frames: np.ndarray, rel_ours_rows: np.ndarray, ref_means: np.ndarray, ref_ses: np.ndarray) -> dict:
    """OLS slope of d_f = (mean_ours_f - mean_ref_f)/mean_ref_f on f with the sandwich variance.

    rel_ours_rows: (R, F) per-chain values of the channel mean at each frame (same chains across frames).
    ref_means / ref_ses: (F,) PT means and SEs (independent across frames).
    """
    f = np.asarray(frames, dtype=np.float64)
    F = f.size
    if F < 2:
        return dict(slope=None, se=None, z=None, note="fewer than 2 test frames")
    R = rel_ours_rows.shape[0]
    mo = rel_ours_rows.mean(axis=0)
    cov_o = np.cov(rel_ours_rows, rowvar=False, ddof=1).reshape(F, F) / R
    d = (mo - ref_means) / ref_means
    # delta method for d_f = o_f / r_f - 1: var from o (covariance) and r (independent)
    J_o = np.diag(1.0 / ref_means)
    cov = J_o @ cov_o @ J_o + np.diag((mo / ref_means ** 2) ** 2 * ref_ses ** 2)
    fc = f - f.mean()
    sxx = float((fc ** 2).sum())
    if sxx == 0:
        return dict(slope=None, se=None, z=None, note="all test frames equal")
    w = fc / sxx
    slope = float(w @ d)
    se = float(math.sqrt(max(w @ cov @ w, 0.0)))
    z = slope / se if se > 0 else (0.0 if slope == 0 else math.copysign(math.inf, slope))
    return dict(slope=slope, se=se, z=z, ci99=[slope - 2.5758 * se, slope + 2.5758 * se], deltas=d.tolist(),
                delta_se=np.sqrt(np.diag(cov)).tolist())


def tile_binomial(k: int, m: int, alpha: float = ALPHA_TILE) -> float:
    """P(K >= k) for K ~ Binomial(m, alpha)."""
    return float(st.binom.sf(k - 1, m, alpha)) if k > 0 else 1.0


def load_pair(fr: dict, test: dict, test_dir: Path, spec: S.GateSpec) -> tuple[S.Replicates, S.Replicates]:
    tiles = sorted({16, 32, 64, spec.tile})
    so, sr = discover(Path(fr["ours"])), discover(Path(fr["ref"]))
    from compare import image_size
    h, w = image_size(sr)
    masks, names = load_masks(test, test_dir, h, w)
    ours, _ = load_replicates(so, spec.channels, tiles, masks, names)
    ref, _ = load_replicates(sr, spec.channels, tiles, masks, names)
    return ours, ref


def sequence(spec_path: Path, out: Path) -> dict:
    cfg = json.loads(Path(spec_path).read_text())
    base = Path(spec_path).parent
    res = dict(tool="dynamic.py", mode="sequence", frames=[], channels={})
    alpha_u = None
    rows: dict[str, list[np.ndarray]] = {}
    ref_m: dict[str, list[float]] = {}
    ref_se: dict[str, list[float]] = {}
    seeds0 = None
    fs = []
    tile_ok = True
    for fr in cfg["frames"]:
        tpath = base / fr.get("test", cfg["test"])
        test = json.loads(tpath.read_text())
        test.setdefault("n_units", cfg.get("n_units", 1))
        spec, _ = build_spec(test)
        alpha_u = spec.alpha_u
        ours, ref = load_pair(fr, test, tpath.parent, spec)
        meta = json.loads((Path(fr["ours"]) / "meta.json").read_text()) if (Path(fr["ours"]) / "meta.json").is_file() else {}
        if seeds0 is None:
            seeds0 = meta.get("seeds")
        elif meta.get("seeds") is not None and meta.get("seeds") != seeds0:
            raise ValueError(f"frame {fr['frame']}: chain rows differ from the first test frame (drift covariance needs the same chains)")
        fs.append(int(fr["frame"]))
        per = dict(frame=int(fr["frame"]), channels={})
        for ci, ch in enumerate(spec.channels):
            rows.setdefault(ch, []).append(ours.glob[:, ci])
            sr = S.summarize(ref.glob[:, ci:ci + 1])
            ref_m.setdefault(ch, []).append(float(sr.mean[0]))
            ref_se.setdefault(ch, []).append(float(sr.se[0]))
            so = S.summarize(ours.glob[:, ci:ci + 1])
            d, se, t, nu = S.welch(so.mean, so.se, so.nu, sr.mean, sr.se, sr.nu)
            # tile family at the gate tile size
            xo, xr = S.summarize(ours.tiles[spec.tile][..., ci]), S.summarize(ref.tiles[spec.tile][..., ci])
            _, _, tt, tnu = S.welch(xo.mean, xo.se, xo.nu, xr.mean, xr.se, xr.nu)
            p = S.p_two_sided(tt, tnu)
            valid = np.isfinite(p)
            k, m = int((p[valid] < ALPHA_TILE).sum()), int(valid.sum())
            pb = tile_binomial(k, m)
            ok = pb >= alpha_u
            tile_ok &= ok
            per["channels"][ch] = dict(rel=float(d[0] / sr.mean[0]) if sr.mean[0] > 0 else None, z=float(t[0]),
                                       tiles_rejected=k, tiles=m, binom_p=pb, binom_ok=bool(ok))
        res["frames"].append(per)
    drift_ok = True
    for ch in rows:
        R = np.stack(rows[ch], axis=1)
        dr = drift(np.array(fs), R, np.array(ref_m[ch]), np.array(ref_se[ch]))
        zc = float(st.norm.ppf(1 - alpha_u / 2)) if alpha_u else 2.5758
        ok = dr.get("z") is None or abs(dr["z"]) <= zc
        drift_ok &= bool(ok)
        res["channels"][ch] = dict(drift=dr, drift_z_crit=zc, drift_ok=bool(ok),
                                   series_z=[f["channels"][ch]["z"] for f in res["frames"]])
    res.update(alpha_u=alpha_u, alpha_tile=ALPHA_TILE, drift_ok=bool(drift_ok), tiles_ok=bool(tile_ok), ok=bool(drift_ok and tile_ok))
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "dynamic.json", res)
    return res


def merge_npz(out: Path, chunks: list[Path]) -> dict:
    """Concatenate the per-chain rows of ensemble.npz chunks (batch order) and add the pixel moments."""
    zs = [dict(np.load(c / "ensemble.npz", allow_pickle=False)) for c in chunks]
    cat = ("tiles16", "tiles32", "tiles64", "global", "masks")
    add = ("pixel_sum", "pixel_sumsq")
    m = {}
    for k in zs[0]:
        if k in cat:
            m[k] = np.concatenate([z[k] for z in zs], axis=0)
        elif k in add:
            m[k] = np.sum([z[k] for z in zs], axis=0)
        elif k == "count":
            m[k] = np.array(sum(int(z[k]) for z in zs), dtype=np.int64)
        else:
            for z in zs[1:]:
                if not np.array_equal(z[k], zs[0][k]):
                    raise ValueError(f"merge-npz: key {k} differs between chunks")
            m[k] = zs[0][k]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "ensemble.npz", **m)
    metas = [json.loads((c / "meta.json").read_text()) for c in chunks]
    meta = dict(metas[0])
    meta["seeds"] = [s for x in metas for s in x.get("seeds", [])]
    meta["chains"] = sum(int(x.get("chains", 0)) for x in metas)
    meta["chunks"] = [dict(dir=str(c), batchOffset=x.get("batchOffset"), chains=x.get("chains")) for c, x in zip(chunks, metas)]
    meta["ok"] = all(x.get("ok") for x in metas)
    meta["errors"] = [e for x in metas for e in x.get("errors", [])]
    if len(set(meta["seeds"])) != len(meta["seeds"]):
        raise ValueError("merge-npz: duplicate chain seeds across chunks")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return dict(rows=int(m["count"]), chunks=len(chunks))


def scaled(r: S.Replicates, f: float) -> S.Replicates:
    return S.Replicates(channels=r.channels, height=r.height, width=r.width, tiles={k: v * f for k, v in r.tiles.items()}, glob=r.glob * f,
                        masks=None if r.masks is None else r.masks * f, mask_names=r.mask_names)


def calibrate(ours: Path, test_path: Path, out: Path, factor: float = 1.003, splits: int = 20, repeats: int = 10, seed: int = 0) -> dict:
    test = json.loads(Path(test_path).read_text())
    spec, _ = build_spec(test)
    reps, _ = load_replicates(discover(Path(ours)), spec.channels, sorted({16, 32, 64, spec.tile}))
    aa = S.aa_split(reps, spec, n_splits=splits, rng=seed)
    rng = np.random.default_rng(seed + 1)
    sp = spec.with_(min_replicates=2)
    fails = ctrl = 0
    for _ in range(repeats):
        a, b = S._halves(reps.n, rng)
        fails += not S.evaluate_gate(scaled(reps.subset(a), factor), reps.subset(b), sp).passed
        ctrl += S.evaluate_gate(reps.subset(a), reps.subset(b), sp).passed
    req = math.ceil(0.9 * repeats)
    plant = dict(factor=factor, n_repeats=repeats, gate_fail_count=fails, control_pass_count=ctrl, calibrated=bool(fails >= req and ctrl >= req))
    rep = dict(tool="dynamic.py", mode="calibrate", aa=aa, plant=plant, ok=bool(aa["ok"] and plant["calibrated"]))
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report.json", rep)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("sequence")
    a1.add_argument("--spec", type=Path, required=True)
    a1.add_argument("--out", type=Path, required=True)
    a2 = sub.add_parser("merge-npz")
    a2.add_argument("--out", type=Path, required=True)
    a2.add_argument("chunks", type=Path, nargs="+")
    a3 = sub.add_parser("calibrate")
    a3.add_argument("--ours", type=Path, required=True)
    a3.add_argument("--test", type=Path, required=True)
    a3.add_argument("--out", type=Path, required=True)
    a3.add_argument("--factor", type=float, default=1.003)
    a3.add_argument("--splits", type=int, default=20)
    a3.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args(argv)
    if a.cmd == "calibrate":
        r = calibrate(a.ours, a.test, a.out, a.factor, a.splits, a.repeats)
        print(json.dumps(dict(ok=r["ok"], aa=r["aa"]["ok"], plant=r["plant"])))
        return 0 if r["ok"] else 1
    if a.cmd == "sequence":
        r = sequence(a.spec, a.out)
        print(json.dumps(dict(ok=r["ok"], drift_ok=r["drift_ok"], tiles_ok=r["tiles_ok"],
                              drift_z={ch: v["drift"].get("z") for ch, v in r["channels"].items()})))
        return 0 if r["ok"] else 1
    print(json.dumps(merge_npz(a.out, a.chunks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
