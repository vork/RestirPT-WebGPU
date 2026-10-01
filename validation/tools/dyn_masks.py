#!/usr/bin/env python3
"""Region masks of the dynamic (rung 3.6) units and the temporal plants (restir-temporal-api.md §6.4, §6.5; R19).

OWNER T-E. Masks are deterministic functions of the FROZEN PT reference means (and of the production temporal_pixel
through the chain runner's `disocc` harness mode), never of ReSTIR samples.

Partition of the member image (region ids in priority order; each pixel belongs to the first region that matches):
  M_disocc  the chain runner's pixel-centre T1 flags (jitter off): no valid temporal pixel q' at frame t (disocc.bin)
  M_new     4x4-block PT means: L_{t-1} <= 0.01 * mean_{t-1}  and  L_t >= 0.05 * mean_t
  M_gone    symmetric:          L_t     <= 0.01 * mean_t      and  L_{t-1} >= 0.05 * mean_{t-1}
  M_edge    |L_t - L_{t-1}| >= 0.25 * max(L_t, L_{t-1})
  M_steady  everything else
(L = luminance Y of the 4x4-block mean; mean_x = the image mean of frame x.) Empty regions are dropped (compare.py
refuses empty masks), as are regions below MIN_REGION_PX = 256 pixels; both are listed in masks.json "dropped".
Silhouettes (plant N5 only; --sil): M_sil = pixels whose luminance differs from a 4-neighbour by >= 25 % of the larger
(PT mean at t, pixel level), dilated by one pixel: the geometric/shading edges where a pixel-centre primary (N5) differs
from the jittered one even with a static camera (Changelog E-6).
Signed change regions (plants only, with --sil): M_up / M_down = the 4x4 cells with |L_t - L_{t-1}| >= 25 % of the larger
that got brighter / darker (the sign-resolved M_edge; M_new / M_gone are their extreme cases and are empty for slow light
or env motion, Changelog E-13).
Dominance regions (plant sign checks only, not a partition): M_light:<name> = 4x4 blocks where the PT render of frame t
with only that emitter enabled carries >= 70 % of the full PT's L_t.

Output per test frame (directory OUT/f<t>/):
  masks.json  {"names": [...], "width", "height", "dropped": [...], "pixels": {...}, "sources": {...}}
  masks.bin   uint16 little-endian per member-local pixel (row 0 = top): bit i <=> region names[i]  (chain runner)
  mask_<i>.pfm  1.0 inside region i (compare.py test.json "masks": [{"name", "file"}])
Usage:
  dyn_masks.py --ref-t PT_DIR_t --ref-prev PT_DIR_t-1 --frame t --out DIR [--disocc disocc.bin]
               [--dominance NAME=PT_DIR ...] [--no-partition] [--sil]
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

from imageio_util import read_image, write_pfm  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722])
BLOCK = 4
NEW_LOW, NEW_HIGH, EDGE_FRAC, DOMINANCE = 0.01, 0.05, 0.25, 0.70
# Regions smaller than one 16² tile are dropped (their 3 % TOST would need orders of magnitude more chains than the
# 64² tiles that already cover them; restir-temporal-api.md Changelog E-3).
MIN_REGION_PX = 256
PARTITION = ("M_disocc", "M_new", "M_gone", "M_edge", "M_steady")


def mean_image(d: Path) -> np.ndarray:
    """Mean of the batch_###.pfm replicates of a PT reference directory (H, W, 3), row 0 = top."""
    files = sorted(Path(d).glob("batch_*.pfm"))
    if not files:
        mean = Path(d) / "mean.pfm"
        if not mean.is_file():
            raise FileNotFoundError(f"{d}: no batch_*.pfm / mean.pfm")
        return read_image(mean, drop_alpha=True)[..., :3].astype(np.float64)
    acc = None
    for f in files:
        x = read_image(f, drop_alpha=True)[..., :3].astype(np.float64)
        acc = x if acc is None else acc + x
    return acc / len(files)


def block_luma(img: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """4x4-block mean luminance expanded back to pixel resolution (edge blocks partial)."""
    y = img @ LUMA
    h, w = y.shape
    bh, bw = -(-h // block), -(-w // block)
    pad = np.zeros((bh * block, bw * block))
    cnt = np.zeros_like(pad)
    pad[:h, :w] = y
    cnt[:h, :w] = 1
    s = pad.reshape(bh, block, bw, block).sum(axis=(1, 3))
    c = cnt.reshape(bh, block, bw, block).sum(axis=(1, 3))
    m = s / np.maximum(c, 1)
    return np.repeat(np.repeat(m, block, axis=0), block, axis=1)[:h, :w]


def partition(l_t: np.ndarray, l_p: np.ndarray, mean_t: float, mean_p: float, disocc: np.ndarray | None) -> dict[str, np.ndarray]:
    """Priority partition of §6.4 from 4x4-block luminances (pure)."""
    h, w = l_t.shape
    taken = np.zeros((h, w), bool)
    out: dict[str, np.ndarray] = {}
    cand = {
        "M_disocc": disocc.astype(bool) if disocc is not None else np.zeros((h, w), bool),
        "M_new": (l_p <= NEW_LOW * mean_p) & (l_t >= NEW_HIGH * mean_t),
        "M_gone": (l_t <= NEW_LOW * mean_t) & (l_p >= NEW_HIGH * mean_p),
        "M_edge": np.abs(l_t - l_p) >= EDGE_FRAC * np.maximum(l_t, l_p),
        "M_steady": np.ones((h, w), bool),
    }
    for name in PARTITION:
        m = cand[name] & ~taken
        # M_edge needs a non-zero reference (both black ⇒ steady)
        if name == "M_edge":
            m &= np.maximum(l_t, l_p) > 0
        out[name] = m
        taken |= m
    return out


def silhouettes(img: np.ndarray) -> np.ndarray:
    y = img @ LUMA
    e = np.zeros(y.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        a, b = y[: y.shape[0] - dy, : y.shape[1] - dx], y[dy:, dx:]
        m = np.abs(a - b) >= EDGE_FRAC * np.maximum(np.maximum(a, b), 1e-30)
        m &= np.maximum(a, b) > 0
        e[: y.shape[0] - dy, : y.shape[1] - dx] |= m
        e[dy:, dx:] |= m
    d = e.copy()
    d[1:] |= e[:-1]; d[:-1] |= e[1:]; d[:, 1:] |= e[:, :-1]; d[:, :-1] |= e[:, 1:]
    return d


def dominance(l_full: np.ndarray, l_only: np.ndarray) -> np.ndarray:
    return (l_full > 0) & (l_only >= DOMINANCE * l_full)


def write_masks(out: Path, masks: dict[str, np.ndarray], sources: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n, m in masks.items() if int(m.sum()) >= MIN_REGION_PX]
    dropped = [n for n, m in masks.items() if int(m.sum()) < MIN_REGION_PX]
    if len(names) > 16:
        raise ValueError(f"{len(names)} regions > 16 (ensemble mask limit)")
    first = next(iter(masks.values()))
    h, w = first.shape
    bits = np.zeros((h, w), np.uint16)
    for i, n in enumerate(names):
        bits |= (masks[n].astype(np.uint16) << i)
        write_pfm(out / f"mask_{i}.pfm", np.repeat(masks[n].astype(np.float32)[..., None], 3, axis=2))
    bits.astype("<u2").tofile(out / "masks.bin")
    meta = {"names": names, "width": w, "height": h, "dropped": dropped,
            "pixels": {n: int(masks[n].sum()) for n in masks}, "sources": sources,
            "rule": "restir-temporal-api.md §6.4 (validation/tools/dyn_masks.py)",
            "test_json_masks": [{"name": n, "file": f"mask_{i}.pfm"} for i, n in enumerate(names)]}
    (out / "masks.json").write_text(json.dumps(meta, indent=1))
    return meta


def build(ref_t: Path, ref_prev: Path | None, frame: int, out: Path, disocc: Path | None = None,
          dominance_refs: dict[str, Path] | None = None, with_partition: bool = True, sil: bool = False) -> dict:
    img_t = mean_image(ref_t)
    l_t = block_luma(img_t)
    mean_t = float((img_t @ LUMA).mean())
    masks: dict[str, np.ndarray] = {}
    if with_partition:
        if ref_prev is None:
            raise ValueError("partition needs --ref-prev")
        img_p = mean_image(ref_prev)
        l_p, mean_p = block_luma(img_p), float((img_p @ LUMA).mean())
        dis = None
        if disocc is not None:
            dis = np.fromfile(disocc, dtype=np.uint8).reshape(l_t.shape) > 0
        masks.update(partition(l_t, l_p, mean_t, mean_p, dis))
    if sil:
        masks["M_sil"] = silhouettes(img_t)
        if with_partition:
            edge = (np.abs(l_t - l_p) >= EDGE_FRAC * np.maximum(l_t, l_p)) & (np.maximum(l_t, l_p) > 0)
            masks["M_up"] = edge & (l_t > l_p)
            masks["M_down"] = edge & (l_t < l_p)
    for name, d in (dominance_refs or {}).items():
        masks[f"M_light:{name}"] = dominance(l_t, block_luma(mean_image(d)))
    sources = {"ref_t": str(ref_t), "ref_prev": None if ref_prev is None else str(ref_prev), "frame": frame,
               "disocc": None if disocc is None else str(disocc),
               "dominance": {k: str(v) for k, v in (dominance_refs or {}).items()}}
    return write_masks(out / f"f{frame}", masks, sources)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-t", type=Path, required=True)
    ap.add_argument("--ref-prev", type=Path)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--disocc", type=Path)
    ap.add_argument("--dominance", action="append", default=[], help="NAME=PT_DIR (single-emitter PT at frame t)")
    ap.add_argument("--no-partition", action="store_true")
    ap.add_argument("--sil", action="store_true", help="add M_sil (plant N5)")
    a = ap.parse_args(argv)
    dom = dict(x.split("=", 1) for x in a.dominance)
    meta = build(a.ref_t, a.ref_prev, a.frame, a.out, a.disocc, {k: Path(v) for k, v in dom.items()}, not a.no_partition, a.sil)
    print(json.dumps({"frame": a.frame, "names": meta["names"], "dropped": meta["dropped"], "pixels": meta["pixels"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
