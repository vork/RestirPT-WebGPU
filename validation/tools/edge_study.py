"""Edge study (docs/decisions/denoiser.md Changelog DN-16): temporal stability and error of the displayed image on
silhouette pixels vs interior pixels, for runs of validation/harness/run-denoise.ts (flip mode, consecutive eval
frames), against a PT reference.

Masks (from the PT reference of the frame): `edge` = denoise_eval.edge_masks (≥ 25 % luminance step to a 4-neighbour,
dilated 1 px), `interior` = pixels ≥ 2 px from any edge, `all` = every pixel above the 2 % luminance floor (perf2 WP-Q). With --ids (an int32 .npy material-id map at pixel centres,
-1 = background) the edge pixels are split by the pair of ids across the boundary, e.g. box|bg and box|wall.

Per run and image (dn = the displayed output, raw = rsFrame), display luminance (Standard view: clamp, sRGB OETF):
  std    per-pixel temporal std over the frames                       (static camera only)
  dt     mean |I_t − I_{t−1}|                                         (static camera only)
  bias   |mean_t I_t − I_ref|                                         (static camera only)
  rmse   sqrt(mean_t (I_t − I_ref)²)  (per-frame refs with --frame-refs; all eval frames otherwise)
  d2     mean |I_t − 2 I_{t−1} + I_{t−2}|                             (motion: cancels constant-velocity change)

    python edge_study.py --ref REF.pfm [--ids ids.npy] --runs label=DIR,... --frames 40:64 [--frame-refs t=REF,...] --out stats.json
    python edge_study.py crops --runs label=DIR,... --frame 63 --box x0,y0,x1,y1 --zoom 8 --out DIR [--ref REF.pfm] [--std [--std-gain 10]]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from denoise_eval import LUMA, edge_masks, srgb_oetf, write_png  # noqa: E402
from imageio_util import read_pfm  # noqa: E402


def disp_lum(img: np.ndarray) -> np.ndarray:
    return srgb_oetf(img) @ LUMA


def all_mask(ref: np.ndarray) -> np.ndarray:
    """perf2 WP-Q: every pixel above the edge_masks luminance floor (2 % of the mean reference luminance)."""
    lum = ref @ LUMA
    return lum > 0.02 * float(lum.mean())


def id_masks(ids: np.ndarray, edge: np.ndarray) -> dict[str, np.ndarray]:
    """Edge pixels split by the (sorted) id pair across a 4-neighbour boundary within 1 px."""
    out: dict[str, np.ndarray] = {}
    H, W = ids.shape
    for dy, dx in ((0, 1), (1, 0)):
        a, b = ids[: H - dy, : W - dx], ids[dy:, dx:]
        diff = a != b
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        for u, v in set(zip(lo[diff].tolist(), hi[diff].tolist())):
            m = np.zeros((H, W), bool)
            sel = diff & (lo == u) & (hi == v)
            m[: H - dy, : W - dx] |= sel
            m[dy:, dx:] |= sel
            key = f'{u}|{v}'
            out[key] = out.get(key, np.zeros((H, W), bool)) | m
    for k, m in out.items():
        d = m.copy()
        for yy in (-1, 0, 1):
            for xx in (-1, 0, 1):
                d |= np.roll(np.roll(m, yy, 0), xx, 1)
        out[k] = d
    return out


def stats(frames: np.ndarray, ref_l: np.ndarray | None, masks: dict[str, np.ndarray], static: bool, frame_refs: dict[int, np.ndarray] | None, f0: int) -> dict:
    r: dict[str, dict] = {}
    std = frames.std(axis=0)
    dt = np.abs(np.diff(frames, axis=0)).mean(axis=0)
    d2 = np.abs(frames[2:] - 2 * frames[1:-1] + frames[:-2]).mean(axis=0)
    bias = np.abs(frames.mean(axis=0) - ref_l) if ref_l is not None else None
    if frame_refs:
        se = np.mean([(frames[t - f0] - disp_lum(fr)) ** 2 for t, fr in frame_refs.items()], axis=0)
    elif ref_l is not None:
        se = ((frames - ref_l[None]) ** 2).mean(axis=0)
    else:
        se = None
    for k, m in masks.items():
        if not m.any():
            continue
        e = {'n': int(m.sum()), 'd2': float(d2[m].mean())}
        if static:
            e.update(std=float(std[m].mean()), dt=float(dt[m].mean()))
            if bias is not None:
                e['bias'] = float(bias[m].mean())
        if se is not None:
            e['rmse'] = float(np.sqrt(se[m].mean()))
        r[k] = e
    return r


def cmd_stats(a: argparse.Namespace) -> int:
    f0, f1 = (int(x) for x in a.frames.split(':'))
    frame_refs = {int(t): read_pfm(p) for t, p in (s.split('=') for s in a.frame_refs.split(','))} if a.frame_refs else None
    ref = read_pfm(a.ref)
    edge, interior = edge_masks(ref)
    if frame_refs:   # moving camera: the union of the per-frame edge masks
        for fr in frame_refs.values():
            e2, i2 = edge_masks(fr)
            edge |= e2
            interior &= i2
    masks = {'edge': edge, 'interior': interior, 'all': all_mask(ref)}
    if a.ids:
        ids = np.load(a.ids)
        for k, m in id_masks(ids, edge).items():
            masks[f'edge {k}'] = m & edge
    ref_l = None if frame_refs else disp_lum(ref)
    out = {'frames': [f0, f1], 'ref': a.ref, 'frame_refs': a.frame_refs, 'runs': {}}
    for item in a.runs.split(','):
        label, d = item.split('=')
        for img in a.images.split(','):
            p = [Path(d) / f'{img}_f{t}.pfm' for t in range(f0, f1)]
            if not all(x.exists() for x in p):
                continue
            fr = np.stack([disp_lum(read_pfm(x)) for x in p])
            s = stats(fr, ref_l, masks, frame_refs is None, frame_refs, f0)
            out['runs'][f'{label}/{img}'] = s
            line = ' | '.join(f"{k} " + ' '.join(f"{q} {v:.4f}" for q, v in e.items() if q != 'n') for k, e in s.items())
            print(f'{label + "/" + img:>22}: {line}')
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


def cmd_crops(a: argparse.Namespace) -> int:
    x0, y0, x1, y1 = (int(v) for v in a.box.split(','))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    srcs = [s.split('=') for s in a.runs.split(',')]
    if a.ref:
        srcs.insert(0, ['ref', a.ref])
    for label, p in srcs:
        pp = Path(p)
        if a.std and pp.suffix != '.pfm':   # per-pixel temporal std of the display luminance over --frames, × --std-gain
            f0, f1 = (int(v) for v in a.frames.split(':'))
            st = np.stack([disp_lum(read_pfm(pp / f'{a.image}_f{t}.pfm')[y0:y1, x0:x1]) for t in range(f0, f1)])
            c = np.repeat(np.clip(st.std(axis=0) * a.std_gain, 0, 1)[..., None], 3, axis=2)
        elif a.std:
            c = np.zeros((y1 - y0, x1 - x0, 3))
        else:
            img = read_pfm(pp if pp.suffix == '.pfm' else pp / f'{a.image}_f{a.frame}.pfm')
            c = srgb_oetf(img[y0:y1, x0:x1])
        z = np.repeat(np.repeat(c, a.zoom, 0), a.zoom, 1)
        write_png(out / f'{label}.png', z)
        tiles.append(z)
    pad = np.ones((tiles[0].shape[0], 4, 3))
    strip = np.concatenate([x for t in tiles for x in (t, pad)][:-1], axis=1)
    write_png(out / 'strip.png', strip)
    print(f'{out}/strip.png: ' + ', '.join(s[0] for s in srcs))
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == 'crops':
        ap = argparse.ArgumentParser()
        ap.add_argument('cmd')
        ap.add_argument('--runs', required=True)
        ap.add_argument('--frame', type=int, default=63)
        ap.add_argument('--image', default='dn')
        ap.add_argument('--box', required=True)
        ap.add_argument('--zoom', type=int, default=8)
        ap.add_argument('--ref')
        ap.add_argument('--std', action='store_true', help='temporal std maps (display luminance) instead of images')
        ap.add_argument('--std-gain', type=float, default=10.0)
        ap.add_argument('--frames', default='40:64')
        ap.add_argument('--out', required=True)
        return cmd_crops(ap.parse_args())
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', required=True)
    ap.add_argument('--ids')
    ap.add_argument('--runs', required=True)
    ap.add_argument('--images', default='dn,raw')
    ap.add_argument('--frames', default='40:64')
    ap.add_argument('--frame-refs')
    ap.add_argument('--out', required=True)
    return cmd_stats(ap.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
