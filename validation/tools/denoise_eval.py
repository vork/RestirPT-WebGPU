"""M5.5 denoiser evaluation (docs/decisions/denoiser.md §11).

flip      mean LDR-FLIP (Standard view: exposure 1, clamp, sRGB OETF: the app's default display) and HDR-FLIP of the
          raw 1-frame ReSTIR images and of the denoised images against a PT reference; per image, per scene means
          and the ratio raw / denoised. Optionally writes display PNGs (reference, raw, denoised, FLIP maps).
recovery  frames until the denoised regional mean recovers 95 % of each ix-e step (tiles.bin of several seeds, the
          step masks from PT references before / after each step).

    python denoise_eval.py flip --ref mean.pfm --pairs raw_f16.pfm:dn_f16.pfm,... --out flip.json [--png DIR]
    python denoise_eval.py recovery --runs DIR,DIR --steps 32,56,80 --hold 24 --refs B0.pfm:A0.pfm,... --names a,b --out rec.json
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
    a = p.parse_args()
    return cmd_flip(a) if a.cmd == "flip" else cmd_recovery(a)


if __name__ == "__main__":
    raise SystemExit(main())
