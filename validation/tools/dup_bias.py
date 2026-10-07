#!/usr/bin/env python3
"""Gate 5 (PLAN §7.1; restir-m6-api.md MD15): bias of the duplication map (Enhanced's 3.25 % budget).

Inputs: the chains of one configuration with the duplication map ON at one test frame and the PT reference of that
frame. Both sides are read exactly as compare.py reads them (compare.discover + compare.load_replicates): the chain
side is an ensemble.npz of per-chain tile / image SUMS (stats.replicates_from_sums divides by the tile pixel counts;
restir-m6-api.md Changelog M6-12), the PT side batch_###.pfm batch means; any layout compare.py accepts works on
either side. Several --ours / --ref directories (disjoint seed sets of the same frame and configuration, e.g. a unit
and its confirmatory re-run) are pooled replicate by replicate.
Per 16² tile t (luminance Y):
    Δ_t = mean_ours − mean_ref,  SE_t = sqrt(SE_ours² + SE_ref²),  D_t = max(R̄_t, 0.05·R̄_image)  (compare.py dark-tile rule)
    b̂_t = sqrt(max(Δ_t² − SE_t², 0)) / D_t          (noise-debiased |bias| estimate, E[Δ²] = bias² + SE²)
Reported / gated:
    mean_t b̂_t ≤ budget (3.25 %), the 99 % upper bound (|Δ_g| + z_.995·SE_g)/R̄_g of the global relative |bias| ≤ budget,
    and the noise floor mean_t SE_t/D_t ≤ 1 % (so the budget is resolvable at this size).
Usage: dup_bias.py --ours DIR [DIR ...] --ref PT_DIR [PT_DIR ...] --out OUT [--tile 16] [--budget 0.0325] [--floor 0.01]
       [--frame F]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import compare as C  # noqa: E402
import stats as S  # noqa: E402

Z995 = 2.5758293035489


def _paths(p: Path | str | Sequence[Path | str]) -> list[Path]:
    return [Path(p)] if isinstance(p, (str, Path)) else [Path(x) for x in p]


def load_side(dirs: Sequence[Path], tile: int, frame: int | None = None) -> tuple[np.ndarray, np.ndarray, tuple[int, int], list[dict]]:
    """Per-replicate Y tile means [N, th, tw] and image means [N] of the pooled directories (compare.py's loaders)."""
    tiles, glob, size, desc = [], [], None, []
    for d in dirs:
        src = C.discover(d, frame)
        reps, _ = C.load_replicates(src, ("Y",), [tile])
        if size is not None and (reps.height, reps.width) != size:
            raise C.InputError(f"{d}: image size {(reps.height, reps.width)} != {size} of the other inputs")
        size = (reps.height, reps.width)
        tiles.append(reps.tiles[tile][..., 0])
        glob.append(reps.glob[:, 0])
        desc.append(dict(src.describe(), n=int(reps.glob.shape[0])))
    return np.concatenate(tiles), np.concatenate(glob), size, desc


def evaluate(ours, ref, tile: int = 16, budget: float = 0.0325, floor: float = 0.01, frame: int | None = None) -> dict:
    to, go, so_size, od = load_side(_paths(ours), tile, frame)
    tr, gr_, sr_size, rd = load_side(_paths(ref), tile, frame)
    if so_size != sr_size:
        raise ValueError(f"image sizes differ: ours {so_size} ref {sr_size}")
    tx, trs = S.summarize(to), S.summarize(tr)
    gx, grs = S.summarize(go), S.summarize(gr_)
    gr = float(grs.mean)
    D = np.maximum(trs.mean, 0.05 * gr)
    ok_t = D > 0
    Dsafe = np.where(ok_t, D, 1)
    delta, se = tx.mean - trs.mean, np.sqrt(tx.se ** 2 + trs.se ** 2)
    b = np.sqrt(np.maximum(delta ** 2 - se ** 2, 0.0)) / Dsafe
    noise = se / Dsafe
    dg, seg = float(gx.mean - grs.mean), float(np.hypot(gx.se, grs.se))
    rel_g = dg / gr
    upper = (abs(dg) + Z995 * seg) / gr
    mean_b = float(b[ok_t].mean())
    mean_noise = float(noise[ok_t].mean())
    signed = delta / Dsafe
    res = dict(tool="dup_bias.py", version=2, tile=tile, chains=int(to.shape[0]), ref_batches=int(tr.shape[0]),
               inputs=dict(ours=od, ref=rd), budget=budget, noise_floor_max=floor,
               mean_tile_bias=mean_b, median_tile_bias=float(np.median(b[ok_t])), p90_tile_bias=float(np.quantile(b[ok_t], 0.9)),
               max_tile_bias=float(b[ok_t].max()), mean_signed_tile_rel=float(signed[ok_t].mean()),
               global_rel=float(rel_g), global_se_rel=float(seg / gr), global_abs_upper99=float(upper), noise_floor=mean_noise,
               noise_floor_ours=float((tx.se / Dsafe)[ok_t].mean()), noise_floor_ref=float((trs.se / Dsafe)[ok_t].mean()),
               tiles=int(ok_t.sum()),
               ok_mean=bool(mean_b <= budget), ok_global=bool(upper <= budget), resolvable=bool(mean_noise <= floor))
    res["ok"] = bool(res["ok_mean"] and res["ok_global"] and res["resolvable"])
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", type=Path, nargs="+", required=True)
    ap.add_argument("--ref", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--budget", type=float, default=0.0325)
    ap.add_argument("--floor", type=float, default=0.01)
    ap.add_argument("--frame", type=int, default=None)
    a = ap.parse_args(argv)
    try:
        res = evaluate(a.ours, a.ref, a.tile, a.budget, a.floor, a.frame)
    except (C.InputError, ValueError) as e:
        print(f"dup_bias.py: {e}", file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "inputs"}))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
