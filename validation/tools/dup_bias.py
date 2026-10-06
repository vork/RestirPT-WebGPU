#!/usr/bin/env python3
"""Gate 5 (PLAN §7.1; restir-m6-api.md MD15): bias of the duplication map (Enhanced's 3.25 % budget).

Inputs: the chains of one configuration with the duplication map ON at one test frame (ensemble.npz: rows = chains,
`tiles16` [R, th, tw, 3], `global` [R, 3]) and the PT reference of that frame (batch_###.pfm; replicate = batch mean).
Per 16² tile t (luminance Y):
    Δ_t = mean_ours − mean_ref,  SE_t = sqrt(SE_ours² + SE_ref²),  D_t = max(R̄_t, 0.05·R̄_image)  (compare.py dark-tile rule)
    b̂_t = sqrt(max(Δ_t² − SE_t², 0)) / D_t          (noise-debiased |bias| estimate, E[Δ²] = bias² + SE²)
Reported / gated:
    mean_t b̂_t ≤ budget (3.25 %), the 99 % upper bound (|Δ_g| + z_.995·SE_g)/R̄_g of the global relative |bias| ≤ budget,
    and the noise floor mean_t SE_t/D_t ≤ 1 % (so the budget is resolvable at this size).
Usage: dup_bias.py --ours DIR_with_ensemble.npz --ref PT_DIR --out OUT [--tile 16] [--budget 0.0325] [--floor 0.01]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from imageio_util import read_image  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722])
Z995 = 2.5758293035489


def ours_side(d: Path, tile: int) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    with np.load(d / "ensemble.npz", allow_pickle=False) as z:
        t = z[f"tiles{tile}"] @ LUMA          # [R, th, tw]
        g = z["global"] @ LUMA                # [R]
    R = t.shape[0]
    return t.mean(0), t.std(0, ddof=1) / np.sqrt(R), float(g.mean()), float(g.std(ddof=1) / np.sqrt(R)), R


def ref_side(d: Path, tile: int) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    files = sorted(d.glob("batch_*.pfm"))
    tiles, glob = [], []
    for f in files:
        img = read_image(f, drop_alpha=True)[..., :3] @ LUMA
        h, w = img.shape
        th, tw = -(-h // tile), -(-w // tile)
        pad = np.full((th * tile, tw * tile), np.nan)
        pad[:h, :w] = img
        tiles.append(np.nanmean(pad.reshape(th, tile, tw, tile), axis=(1, 3)))
        glob.append(img.mean())
    T, G = np.stack(tiles), np.array(glob)
    B = len(files)
    return T.mean(0), T.std(0, ddof=1) / np.sqrt(B), float(G.mean()), float(G.std(ddof=1) / np.sqrt(B)), B


def evaluate(ours: Path, ref: Path, tile: int = 16, budget: float = 0.0325, floor: float = 0.01) -> dict:
    mo, so, go, sgo, R = ours_side(ours, tile)
    mr, sr, gr, sgr, B = ref_side(ref, tile)
    if mo.shape != mr.shape:
        raise ValueError(f"tile grids differ: ours {mo.shape} ref {mr.shape}")
    D = np.maximum(mr, 0.05 * gr)
    ok_t = D > 0
    delta, se = mo - mr, np.sqrt(so ** 2 + sr ** 2)
    b = np.sqrt(np.maximum(delta ** 2 - se ** 2, 0.0)) / np.where(ok_t, D, 1)
    noise = se / np.where(ok_t, D, 1)
    dg, seg = go - gr, float(np.hypot(sgo, sgr))
    rel_g = dg / gr
    upper = (abs(dg) + Z995 * seg) / gr
    mean_b = float(b[ok_t].mean())
    mean_noise = float(noise[ok_t].mean())
    res = dict(tool="dup_bias.py", tile=tile, chains=R, ref_batches=B, budget=budget, noise_floor_max=floor,
               mean_tile_bias=mean_b, median_tile_bias=float(np.median(b[ok_t])), p90_tile_bias=float(np.quantile(b[ok_t], 0.9)),
               global_rel=float(rel_g), global_se_rel=float(seg / gr), global_abs_upper99=float(upper), noise_floor=mean_noise,
               tiles=int(ok_t.sum()),
               ok_mean=bool(mean_b <= budget), ok_global=bool(upper <= budget), resolvable=bool(mean_noise <= floor))
    res["ok"] = bool(res["ok_mean"] and res["ok_global"] and res["resolvable"])
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--budget", type=float, default=0.0325)
    ap.add_argument("--floor", type=float, default=0.01)
    a = ap.parse_args(argv)
    res = evaluate(a.ours, a.ref, a.tile, a.budget, a.floor)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
