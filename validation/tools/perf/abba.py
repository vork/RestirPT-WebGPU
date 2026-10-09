#!/usr/bin/env python3
"""Same-session ABBA analysis of run-perf / prof-driver reports (perf2 V-PERF; docs/decisions/perf2-api.md §4).

usage: abba.py REPORT.json [REPORT.json ...] [--keys rs_initial,rs_spatial_shift,...] [--json out.json]

Job labels are '<group> <role> #<i>' (run-perf.ts --abba writes '<scene>@<res> N3 base #i' / '<scene>@<res> N3 <FLAGS> #i';
the profile runs used '<scene> <variant> #i'). Within one report and group, every variant pair (two consecutive variant
jobs) is compared with the mean of its bracketing base jobs (B V V B, or B V V B B V V B: the base on either side). Per
pair: Δ of the frame time, Σpasses, denoiser total and the per-pass means of --keys; then the mean / min / max over the
pairs of a (group, variant). Report-only: no thresholds."""
import json, re, statistics, sys

DEFAULT_KEYS = ['rs_initial', 'rs_spatial_shift', 'rs_t_classify', 'primary', 'rs_primary', 'rs_t_select', 'rs_t_inverse', 'rs_t_forward',
                'rs_spatial_replay', 'rs_pair_accept', 'rs_dupmap', 'rs_spatial_resample']


def load(path):
    out = []
    for r in json.load(open(path))['reports']:
        p = {x['name']: x['meanMs'] for x in r.get('passes') or []}
        p['frame'] = r['frame']['meanMs']; p['sum'] = r.get('passTotalMs') or 0.0; p['dn'] = (r.get('denoiser') or {}).get('totalMs', 0.0)
        label = re.sub(r'\s*#\d+$', '', r['label'].strip())
        group, role = label.rsplit(' ', 1) if ' ' in label else ('', label)
        out.append((group, role, p, r.get('ok', True)))
    return out


def main(argv):
    keys = DEFAULT_KEYS; out_json = None; files = []
    i = 0
    while i < len(argv):
        if argv[i] == '--keys': keys = argv[i + 1].split(','); i += 2
        elif argv[i] == '--json': out_json = argv[i + 1]; i += 2
        else: files.append(argv[i]); i += 1
    if not files:
        print(__doc__); return 2
    pairs = {}
    for f in files:
        rows = load(f)
        groups = {}
        for g, role, p, ok in rows: groups.setdefault(g, []).append((role, p, ok))
        for g, seq in groups.items():
            k = 0
            while k < len(seq):
                if seq[k][0] == 'base': k += 1; continue
                # a variant pair seq[k], seq[k+1] with the nearest base before and after
                if k + 1 >= len(seq) or seq[k + 1][0] != seq[k][0]: k += 1; continue
                before = next((seq[j] for j in range(k - 1, -1, -1) if seq[j][0] == 'base'), None)
                after = next((seq[j] for j in range(k + 2, len(seq)) if seq[j][0] == 'base'), None)
                if before is None or after is None: k += 2; continue
                row = {}
                for key in ['frame', 'sum', 'dn'] + keys:
                    b = (before[1].get(key, 0) + after[1].get(key, 0)) / 2; v = (seq[k][1].get(key, 0) + seq[k + 1][1].get(key, 0)) / 2
                    row[key] = (b, v, v - b, (v / b - 1) * 100 if b else 0.0)
                pairs.setdefault((g, seq[k][0]), []).append(row)
                k += 2
    if not pairs:
        print('no (base, variant, variant, base) sequences found'); return 1
    summary = {}
    print(f"{'group / variant':48s} {'n':>2s} {'base':>8s} {'var':>8s} {'Δframe':>8s} {'Δ%':>6s} {'[min, max] Δ':>16s} | Σpasses Δ | dn Δ | per-pass Δ (ms)")
    for (g, role), rs in pairs.items():
        d = [r['frame'][2] for r in rs]
        mean = lambda key, i: statistics.mean(r[key][i] for r in rs)
        per = {key: mean(key, 2) for key in keys if any(r[key][0] or r[key][1] for r in rs)}
        summary[f'{g} {role}'] = {'pairs': len(rs), 'frame': {'base': mean('frame', 0), 'variant': mean('frame', 1), 'delta': mean('frame', 2), 'pct': mean('frame', 3), 'min': min(d), 'max': max(d)},
                                  'sum': mean('sum', 2), 'dn': mean('dn', 2), 'passes': per}
        pp = ', '.join(f'{key} {v:+.2f}' for key, v in sorted(per.items(), key=lambda x: x[1]) if abs(v) >= 0.05)
        print(f"{(g + ' ' + role)[:48]:48s} {len(rs):2d} {mean('frame', 0):8.2f} {mean('frame', 1):8.2f} {mean('frame', 2):+8.2f} {mean('frame', 3):+6.1f} [{min(d):+6.2f}, {max(d):+6.2f}] | {mean('sum', 2):+7.2f} | {mean('dn', 2):+5.2f} | {pp}")
    if out_json: json.dump(summary, open(out_json, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
