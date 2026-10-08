#!/usr/bin/env python3
"""Regenerates the Gate-5 real-format fixture (restir-m6-api.md Changelog M6-12) from a real gate run.

Source (M6 gate run m6-gate-gate5-20261007-115614, unit m5s_cornell_i@G5-cap5-dup-f24):
  ours: chains/m5s_cornell_i@G5-cap5-dup/f24/ensemble.npz (restir-chain-run.ts: per-chain tile / image SUMS)
  ref:  validation/out/m5/ptrefs/m5s_cornell_i-fbase-s7001-8192x16-33734be3d1b2d3db/batch_###.pfm (batch means)
Crop: pixels [64:96, 64:96] (16² tiles [4:6, 4:6], 32² tile [2, 2]; no 64² tile fits), the first 32 chains and the first
4 PT batches. The tile sums are copied verbatim; `global` is the crop's sum (= sum of its 16² tile sums); the pixel
moments are the full ensemble's crop scaled to 32 chains (not used by the gate metrics). Every key, dtype and the
sums convention are the writer's.
Usage: make_fixture.py RUN_ROOT (the worktree holding validation/out/...)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from imageio_util import read_image, write_pfm  # noqa: E402

HERE = Path(__file__).resolve().parent
Y0, X0, N, R, B = 64, 64, 32, 32, 4


def main(root: Path) -> None:
    run = root / "validation/out/m6-gate-gate5-20261007-115614"
    src = run / "chains/m5s_cornell_i@G5-cap5-dup/f24"
    ref = root / "validation/out/m5/ptrefs/m5s_cornell_i-fbase-s7001-8192x16-33734be3d1b2d3db"
    with np.load(src / "ensemble.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    full = int(d["count"])
    t16 = d["tiles16"][:R, Y0 // 16:(Y0 + N) // 16, X0 // 16:(X0 + N) // 16].copy()
    out = dict(tiles16=t16, tiles32=d["tiles32"][:R, Y0 // 32:(Y0 + N) // 32, X0 // 32:(X0 + N) // 32].copy(),
               **{"global": t16.sum(axis=(1, 2))},
               pixel_sum=d["pixel_sum"][Y0:Y0 + N, X0:X0 + N] * (R / full),
               pixel_sumsq=d["pixel_sumsq"][Y0:Y0 + N, X0:X0 + N] * (R / full),
               count=np.array(R, dtype=d["count"].dtype), channels=d["channels"],
               height=np.array(N, dtype=d["height"].dtype), width=np.array(N, dtype=d["width"].dtype))
    (HERE / "ours").mkdir(exist_ok=True)
    np.savez(HERE / "ours/ensemble.npz", **out)
    meta = json.loads((src / "meta.json").read_text())
    (HERE / "ours/meta.json").write_text(json.dumps(dict(kind=meta["kind"], kernel=meta["kernel"], seed=meta["seed"], chains=R,
                                                         width=N, height=N, frame=24, source=str(src.relative_to(root))), indent=1))
    (HERE / "ref").mkdir(exist_ok=True)
    for b in range(B):
        write_pfm(HERE / f"ref/batch_{b:03d}.pfm", read_image(ref / f"batch_{b:03d}.pfm", drop_alpha=True)[Y0:Y0 + N, X0:X0 + N, :3])
    m = json.loads((ref / "meta.json").read_text())
    (HERE / "ref/meta.json").write_text(json.dumps(dict(kind=m["kind"], kernel=m["kernel"], seed=m["seed"], sppPerBatch=m["sppPerBatch"],
                                                        batches=B, width=N, height=N, source=str(ref.relative_to(root))), indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
