#!/usr/bin/env python3
"""Env rows of validation/budget.json (plan §5 M3c exit "budget.json has env rows"; env §5.4 "Cycles s/4096 spp for C0q,
C0r, (xiii), (xiv)"), measured from an M3c gate run: Cycles from the cached reference manifests (per-seed render_s,
EXR writing excluded), our PT from the batch meta.json (per-batch wall ms in the page).

  budget_env.py --gate validation/out/m3c-gate-<time> [--budget validation/budget.json]

Upserts `env_entries` (one row per package: scene, width, height, max_bounces, spp, cycles_s_per_4096spp,
ours_s_per_4096spp, ratio) and `env_method`. s_per_4096spp = median(seconds per replicate) / spp * 4096.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENES = ("c0q_lambert_b1_256", "c0q_openbox_b13_256", "c0r_irradiance_256", "c0s_45_nee_256", "c0s_45_none_256",
          "xiii_spheres_512x256", "xiv_overcast_b3_512", "xiv_overcast_rect_b3_512", "xiv_overcast_b7_512", "xiv_kloof_b3_512")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True, type=Path)
    ap.add_argument("--budget", type=Path, default=ROOT / "validation/budget.json")
    a = ap.parse_args()
    summary = json.loads((a.gate / "summary.json").read_text())
    refs, ours = {}, {}
    for st in summary["steps"]:
        d = st.get("data") or {}
        name = st["name"]
        if name.startswith("Cycles reference ") and d.get("dir"):
            refs.setdefault(name.split()[2], d["dir"])
        if name.startswith("PT batches ") and d.get("dir"):
            ours.setdefault(name.split()[2], d["dir"])
    rows = []
    for s in SCENES:
        if s not in refs:
            continue
        man = json.loads((ROOT / refs[s] / "manifest.json").read_text())
        spp = man["render_args"]["spp"]
        W, H = man["resolution"]
        cyc = statistics.median(r["render_s"] for r in man["renders"]) / spp * 4096
        row = {"scene": s, "width": W, "height": H, "max_bounces": man["render_args"]["max_bounces"], "spp": spp,
               "cycles_s_per_4096spp": round(cyc, 3)}
        if s in ours:
            meta = json.loads((ROOT / ours[s] / "meta.json").read_text())
            o = statistics.median(meta["timings"]["batchMs"]) / 1000 / meta["sppPerBatch"] * 4096
            row.update(ours_s_per_4096spp=round(o, 3), cycles_over_ours=round(cyc / o, 2))
        rows.append(row)
    b = json.loads(a.budget.read_text())
    b["env_method"] = ("M3c env scenes (validation/scenes/make-m3c.ts): Cycles = median per-seed bpy render time of the gate's "
                       "reference (EXR write excluded, ~0.3 s fixed overhead included) / spp * 4096; ours = median batch wall time in "
                       f"Chrome (readback included) / spp * 4096; measured {dt.datetime.now().astimezone().isoformat(timespec='seconds')} "
                       f"from {a.gate.name}")
    b["env_entries"] = rows
    a.budget.write_text(json.dumps(b, indent=2) + "\n")
    for r in rows:
        print(json.dumps(r))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
