#!/usr/bin/env python3
"""Spill bytes per pipeline from a Metal System Trace export (perf2; docs/decisions/perf2-api.md §4).

usage: xct_spill.py EXPORT_PREFIX [--json out.json] [--proc 'Google Chrome Helper (GPU)']
EXPORT_PREFIX: the prefix given to xct-export.sh (needs metal-application-encoders-list.xml and
graphics-compiler-spill-events.xml). The compiler's spill events carry the encoder id; encoder labels are Dawn's
'Dawn_ComputePassEncoder_<pass label>' (Chrome with use_user_defined_labels_in_backend; prof-driver.ts sets it), reduced
to the pass kind ('rs_spatial_shift[0][0]' -> 'rs_spatial_shift'). Output: kind -> sorted distinct spilled bytes per
thread ([] when the kind ran in the trace but never spilled). Report-only."""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xctab


def hexid(v):
    return int(v[0], 16) if v[0] and str(v[0]).startswith('0x') else None


def spills(pre, proc='Google Chrome Helper (GPU)'):
    kind = lambda l: l.split('[')[0].split('#')[0].strip()
    lab = {}
    for r in xctab.read(pre + 'metal-application-encoders-list.xml'):
        eid = hexid(r['encoder-id'])
        if eid is not None: lab[eid] = (r['encoder-label'][0] or '').replace('Dawn_ComputePassEncoder_', '')
    out = {kind(l): set() for l in lab.values() if l}
    for r in xctab.read(pre + 'graphics-compiler-spill-events.xml'):
        if not (r['process'][0] and proc in r['process'][0]): continue
        eid = int(r['encoder-id'][1]); l = lab.get(eid)
        out.setdefault(kind(l) if l else f'?{eid:x}', set()).add(int(r['spilled-bytes'][1]))
    return {k: sorted(v) for k, v in out.items()}


if __name__ == '__main__':
    if len(sys.argv) < 2: print(__doc__); sys.exit(2)
    a = sys.argv[1:]
    proc = a[a.index('--proc') + 1] if '--proc' in a else 'Google Chrome Helper (GPU)'
    res = spills(a[0], proc)
    for k, v in sorted(res.items(), key=lambda x: -max(x[1] or [0])):
        print(f'{k:24s} spill/thread B {v}')
    if '--json' in a: json.dump(res, open(a[a.index('--json') + 1], 'w'), indent=1)
