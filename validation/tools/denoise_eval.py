"""M5.5 denoiser evaluation (docs/decisions/denoiser.md §11).

flip      mean LDR-FLIP (Standard view: exposure 1, clamp, sRGB OETF: the app's default display) and HDR-FLIP of the
          raw 1-frame ReSTIR images and of the denoised images against a PT reference; per image, per scene means
          and the ratio raw / denoised. Optionally writes display PNGs (reference, raw, denoised, FLIP maps).
stability temporal stability of the displayed image with a static camera: per-pixel temporal std and mean frame-to-frame
          change of the display-encoded luminance on edge pixels (silhouettes, texture edges: ≥ 25 % luminance step to a
          4-neighbour in the PT reference, dilated 1 px) and on interior pixels.
recovery  frames until the denoised regional mean recovers 95 % of each ix-e step (tiles.bin of several seeds, the
          step masks from PT references before / after each step).
relmse    perf2 WP-Q: relative MSE mean_{pixel,channel} (x - ref)^2 / (ref^2 + eps) (eps 0.01, Rousselle et al. 2011) of
          images against a reference; also the 0.1 %-trimmed variant (the largest per-pixel terms dropped: robust to
          a few fireflies) and, with --ref-var (per-pixel variance of the reference mean), the reference noise floor
          mean Var/(ref^2 + eps), which a measured relMSE contains additively.

    python denoise_eval.py flip --ref mean.pfm --pairs raw_f16.pfm:dn_f16.pfm,... --out flip.json [--png DIR]
    python denoise_eval.py stability --ref REF.pfm --runs label=DIR,... --frames 48:64 --out stab.json
    python denoise_eval.py recovery --runs DIR,DIR --steps 32,56,80 --hold 24 --refs B0.pfm:A0.pfm,... --names a,b --out rec.json
    python denoise_eval.py relmse --ref REF.pfm [--ref-var VAR.pfm] --images a.pfm,b.pfm [--eps 0.01] --out relmse.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imageio_util import read_pfm  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722])
TILE = 8


def srgb_oetf(x: np.ndarray) -> np.ndarray:
    c = np.clip(x, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def flip_version() -> str:
    try:
        from importlib.metadata import version
        return version("flip-evaluator")
    except Exception:  # pragma: no cover
        return "unknown"


def flip_pair(ref: np.ndarray, test: np.ndarray) -> tuple[float, float, np.ndarray]:
    import flip_evaluator as flip
    ldr_ref = srgb_oetf(ref).astype(np.float32)
    ldr_test = srgb_oetf(test).astype(np.float32)
    emap, ldr, _ = flip.evaluate(ldr_ref, ldr_test, "LDR", inputsRGB=True, applyMagma=False)
    _, hdr, _ = flip.evaluate(np.maximum(ref, 0).astype(np.float32), np.maximum(test, 0).astype(np.float32), "HDR", inputsRGB=False, applyMagma=False)
    return float(ldr), float(hdr), np.asarray(emap)


def write_png(path: Path, img: np.ndarray) -> None:
    import OpenImageIO as oiio
    a = np.clip(img, 0, 1)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    a = np.ascontiguousarray((a * 255 + 0.5).astype(np.uint8))
    buf = oiio.ImageBuf(oiio.ImageSpec(a.shape[1], a.shape[0], a.shape[2], oiio.UINT8))
    buf.set_pixels(oiio.ROI(), a)
    if not buf.write(str(path)):
        raise RuntimeError(f"{path}: {buf.geterror()}")


def cmd_flip(a: argparse.Namespace) -> int:
    ref = read_pfm(a.ref)
    rows = []
    for k, pair in enumerate(a.pairs.split(",")):
        raw_p, dn_p = pair.split(":")
        raw, dn = read_pfm(raw_p), read_pfm(dn_p)
        if raw.shape != ref.shape or dn.shape != ref.shape:
            raise SystemExit(f"shape mismatch: ref {ref.shape} raw {raw.shape} dn {dn.shape}")
        l_raw, h_raw, m_raw = flip_pair(ref, raw)
        l_dn, h_dn, m_dn = flip_pair(ref, dn)
        rows.append({"raw": raw_p, "dn": dn_p, "ldr_raw": l_raw, "ldr_dn": l_dn, "hdr_raw": h_raw, "hdr_dn": h_dn,
                     "rmse_raw": float(np.sqrt(np.mean((raw - ref) ** 2))), "rmse_dn": float(np.sqrt(np.mean((dn - ref) ** 2))),
                     "mean_ref": float(np.mean(ref @ LUMA)), "mean_raw": float(np.mean(raw @ LUMA)), "mean_dn": float(np.mean(dn @ LUMA))})
        if a.png and k == len(a.pairs.split(",")) - 1:
            d = Path(a.png)
            d.mkdir(parents=True, exist_ok=True)
            write_png(d / "ref.png", srgb_oetf(ref))
            write_png(d / "raw.png", srgb_oetf(raw))
            write_png(d / "dn.png", srgb_oetf(dn))
            write_png(d / "flip_raw.png", m_raw)
            write_png(d / "flip_dn.png", m_dn)
    mean = lambda key: float(np.mean([r[key] for r in rows]))  # noqa: E731
    out = {
        "flip_evaluator": flip_version(), "metric": "mean LDR-FLIP of the Standard view (exposure 1, clamp, sRGB OETF); HDR-FLIP reported",
        "n": len(rows), "ldr_raw": mean("ldr_raw"), "ldr_dn": mean("ldr_dn"), "hdr_raw": mean("hdr_raw"), "hdr_dn": mean("hdr_dn"),
        "rmse_raw": mean("rmse_raw"), "rmse_dn": mean("rmse_dn"), "rows": rows,
    }
    out["ratio_ldr"] = out["ldr_raw"] / max(out["ldr_dn"], 1e-12)
    out["ratio_hdr"] = out["hdr_raw"] / max(out["hdr_dn"], 1e-12)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"FLIP LDR raw {out['ldr_raw']:.4f} dn {out['ldr_dn']:.4f} ratio {out['ratio_ldr']:.2f} | HDR raw {out['hdr_raw']:.4f} dn {out['hdr_dn']:.4f} ratio {out['ratio_hdr']:.2f} ({len(rows)} images, flip-evaluator {out['flip_evaluator']})")
    return 0


RELMSE_EPS = 0.01


def relmse(img: np.ndarray, ref: np.ndarray, eps: float = RELMSE_EPS, trim: float = 0.0) -> float:
    """mean over pixels and channels of (img - ref)^2 / (ref^2 + eps); `trim` drops that fraction of the largest
    per-pixel (channel-mean) terms."""
    t = ((img.astype(np.float64) - ref) ** 2 / (ref.astype(np.float64) ** 2 + eps)).mean(axis=-1).reshape(-1)
    if trim > 0:
        k = int(len(t) * (1 - trim))
        t = np.partition(t, k)[:k]
    return float(t.mean())


def relmse_floor(ref: np.ndarray, ref_var: np.ndarray, eps: float = RELMSE_EPS) -> float:
    """Expected relMSE of the reference mean against its own expectation: mean Var(ref) / (ref^2 + eps)."""
    return float((ref_var.astype(np.float64) / (ref.astype(np.float64) ** 2 + eps)).mean())


def cmd_relmse(a: argparse.Namespace) -> int:
    ref = read_pfm(a.ref)
    rows = []
    for p in a.images.split(","):
        img = read_pfm(p)
        if img.shape != ref.shape:
            raise SystemExit(f"shape mismatch: ref {ref.shape} {p} {img.shape}")
        rows.append({"image": p, "relmse": relmse(img, ref, a.eps), "relmse_trim": relmse(img, ref, a.eps, 1e-3)})
    out = {"eps": a.eps, "metric": "mean (x - ref)^2 / (ref^2 + eps) over pixels and channels; trim drops the top 0.1 % pixels",
           "n": len(rows), "relmse": float(np.mean([r["relmse"] for r in rows])), "relmse_trim": float(np.mean([r["relmse_trim"] for r in rows])), "rows": rows}
    if a.ref_var:
        out["ref_noise_floor"] = relmse_floor(ref, read_pfm(a.ref_var), a.eps)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"relMSE {out['relmse']:.5g} (trimmed {out['relmse_trim']:.5g}) over {len(rows)} images" + (f"; reference noise floor {out['ref_noise_floor']:.4g}" if a.ref_var else ""))
    return 0


def edge_masks(ref: np.ndarray, step: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    lum = ref @ LUMA
    floor = 0.02 * float(lum.mean())
    edge = np.zeros(lum.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        a, b = lum[: lum.shape[0] - dy, : lum.shape[1] - dx], lum[dy:, dx:]
        e = np.abs(a - b) >= step * np.maximum(np.maximum(a, b), floor)
        edge[: lum.shape[0] - dy, : lum.shape[1] - dx] |= e
        edge[dy:, dx:] |= e
    dil = edge.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            dil |= np.roll(np.roll(edge, dy, 0), dx, 1)
    far = dil.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            far |= np.roll(np.roll(dil, dy, 0), dx, 1)
    interior = ~far & (lum > floor)
    return dil, interior


def cmd_stability(a: argparse.Namespace) -> int:
    ref = read_pfm(a.ref)
    edge, interior = edge_masks(ref)
    f0, f1 = (int(x) for x in a.frames.split(":"))
    if a.motion:
        return stability_motion(a, f0, f1)
    out = {"frames": [f0, f1], "edge_pixels": int(edge.sum()), "interior_pixels": int(interior.sum()), "metric": "display luminance (Standard view), per-pixel temporal std and mean |frame-to-frame change|", "runs": {}}
    for item in a.runs.split(","):
        label, d = item.split("=")
        frames = [srgb_oetf(read_pfm(Path(d) / f"{a.image}_f{t}.pfm")) @ LUMA for t in range(f0, f1)]
        st = np.stack(frames)
        std = st.std(axis=0)
        dif = np.abs(np.diff(st, axis=0)).mean(axis=0)
        r = {"edge_std": float(std[edge].mean()), "interior_std": float(std[interior].mean()), "edge_dt": float(dif[edge].mean()), "interior_dt": float(dif[interior].mean())}
        if a.flip:
            r["ldr_flip_last"] = flip_pair(ref, read_pfm(Path(d) / f"{a.image}_f{f1 - 1}.pfm"))[0]
        out["runs"][label] = r
        print(f"{label:>18}: edge std {r['edge_std']:.4f} dt {r['edge_dt']:.4f} | interior std {r['interior_std']:.4f} dt {r['interior_dt']:.4f}" + (f" | FLIP {r['ldr_flip_last']:.4f}" if a.flip else ""))
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


def stability_motion(a: argparse.Namespace, f0: int, f1: int) -> int:
    """Slow constant-velocity camera motion: the second temporal difference |I_t − 2 I_{t−1} + I_{t−2}| cancels the
    (locally linear) motion and keeps flicker; edges from each run's own frames (≥ 25 % luminance steps, dilated)."""
    out = {"frames": [f0, f1], "metric": "mean |second temporal difference| of the display luminance", "runs": {}}
    for item in a.runs.split(","):
        label, d = item.split("=")
        st = np.stack([srgb_oetf(read_pfm(Path(d) / f"{a.image}_f{t}.pfm")) @ LUMA for t in range(f0, f1)])
        d2 = np.abs(st[2:] - 2 * st[1:-1] + st[:-2])
        e_all = np.zeros(d2.shape, bool)
        i_all = np.zeros(d2.shape, bool)
        for k in range(d2.shape[0]):
            e, i = edge_masks(np.repeat(st[k + 1][:, :, None], 3, 2) / np.array([0.2126 + 0.7152 + 0.0722]))
            e_all[k], i_all[k] = e, i
        r = {"edge_d2": float(d2[e_all].mean()), "interior_d2": float(d2[i_all].mean())}
        out["runs"][label] = r
        print(f"{label:>18}: edge |d2| {r['edge_d2']:.4f} | interior |d2| {r['interior_d2']:.4f}")
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


def tile_means(img: np.ndarray, tx: int, ty: int) -> np.ndarray:
    lum = img @ LUMA
    h, w = lum.shape
    out = np.zeros((ty, tx))
    for y in range(ty):
        for x in range(tx):
            out[y, x] = lum[y * TILE:min(h, (y + 1) * TILE), x * TILE:min(w, (x + 1) * TILE)].mean()
    return out.reshape(-1)


def recovery_frames(r: np.ndarray, tol: float, horizon: int) -> int | None:
    """Smallest k (the step frame is k = 1) with |1 − r| ≤ tol from frame k − 1 through horizon − 1 (None: never)."""
    for k in range(1, horizon + 1):
        if np.all(np.abs(1 - r[k - 1:horizon]) <= tol):
            return k
    return None


def cmd_recovery(a: argparse.Namespace) -> int:
    runs = [Path(p) for p in a.runs.split(",")]
    metas = [json.loads((d / "meta.json").read_text()) for d in runs]
    tiles = metas[0]["tiles"]
    tx, ty = tiles["x"], tiles["y"]
    nt = tx * ty
    frames = metas[0]["frames"]
    data = np.stack([np.fromfile(d / "tiles.bin", dtype="<f4").reshape(frames, nt, 2) for d in runs])  # [seed, frame, tile, (dn, raw)]
    steps = [int(s) for s in a.steps.split(",")]
    refs = [p.split(":") for p in a.refs.split(",")]
    names = a.names.split(",")
    hold, tol, horizon = a.hold, a.tol, 16
    out_steps = []
    ok = True
    for s, (rb, ra), name in zip(steps, refs, names):
        before, after = read_pfm(rb), read_pfm(ra)
        tb, ta = tile_means(before, tx, ty), tile_means(after, tx, ty)
        floor = 0.05 * float(np.mean(before @ LUMA))
        mask = np.abs(ta - tb) >= a.mask_rel * np.maximum(tb, floor)
        if not mask.any():
            out_steps.append({"step": name, "frame": s, "error": "empty mask"})
            ok = False
            continue
        res = {"step": name, "frame": s, "mask_tiles": int(mask.sum()), "mask_fraction": float(mask.mean()),
               "pt_before": float(tb[mask].mean()), "pt_after": float(ta[mask].mean())}
        for side, ch in (("dn", 0), ("raw", 1)):
            m = data[:, :, mask, ch].mean(axis=(0, 2))  # per frame
            mb = float(m[s - 8:s].mean())
            ma = float(m[s + hold - 8:s + hold].mean())
            r = (m[s:s + horizon] - mb) / (ma - mb)
            k = recovery_frames(r, tol, horizon)
            res[side] = {"m_before": mb, "m_after": ma, "r": [float(x) for x in r], "frames_to_95": k,
                         "steady_vs_pt_after": (ma - res["pt_after"]) / res["pt_after"]}
        res["pass"] = res["dn"]["frames_to_95"] is not None and res["dn"]["frames_to_95"] <= a.max_frames
        ok = ok and res["pass"]
        out_steps.append(res)
        print(f"{name} @ {s}: mask {res['mask_tiles']} tiles, PT {res['pt_before']:.4g} -> {res['pt_after']:.4g}; denoised recovers 95 % in "
              f"{res['dn']['frames_to_95']} frames (raw {res['raw']['frames_to_95']}); r = {', '.join(f'{x:.3f}' for x in res['dn']['r'][:10])}")
    Path(a.out).write_text(json.dumps({"ok": ok, "tol": tol, "max_frames": a.max_frames, "seeds": len(runs), "steps": out_steps}, indent=1))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("flip")
    f.add_argument("--ref", required=True)
    f.add_argument("--pairs", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--png")
    t = sub.add_parser("stability")
    t.add_argument("--ref", required=True)
    t.add_argument("--runs", required=True, help="label=DIR,...")
    t.add_argument("--frames", required=True, help="from:to (exclusive)")
    t.add_argument("--image", default="dn", help="dn (the displayed colour) or raw")
    t.add_argument("--flip", action="store_true")
    t.add_argument("--motion", action="store_true", help="slow camera motion: second temporal difference")
    t.add_argument("--out", required=True)
    r = sub.add_parser("recovery")
    r.add_argument("--runs", required=True)
    r.add_argument("--steps", required=True)
    r.add_argument("--refs", required=True, help="before:after PFM pairs, one per step")
    r.add_argument("--names", required=True)
    r.add_argument("--hold", type=int, default=24)
    r.add_argument("--tol", type=float, default=0.05)
    r.add_argument("--mask-rel", dest="mask_rel", type=float, default=0.10)
    r.add_argument("--max-frames", dest="max_frames", type=int, default=8)
    r.add_argument("--out", required=True)
    q = sub.add_parser("relmse")
    q.add_argument("--ref", required=True)
    q.add_argument("--ref-var", dest="ref_var", help="per-pixel variance of the reference mean (PFM)")
    q.add_argument("--images", required=True)
    q.add_argument("--eps", type=float, default=RELMSE_EPS)
    q.add_argument("--out", required=True)
    a = p.parse_args()
    return {"flip": cmd_flip, "stability": cmd_stability, "recovery": cmd_recovery, "relmse": cmd_relmse}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
