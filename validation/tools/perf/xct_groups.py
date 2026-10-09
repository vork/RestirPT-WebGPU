"""Per-encoder GPU time from an exported metal-gpu-intervals + metal-application-encoders-list (+ spill events)."""
import sys, collections, re
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xctab
def hexid(v): return int(v[0], 16) if v[0] and v[0].startswith('0x') else None
def analyse(prefix, proc_sub='Google Chrome Helper (GPU)'):
    E = xctab.read(prefix + 'metal-application-encoders-list.xml')
    lab = {}
    for r in E:
        eid = hexid(r['encoder-id']); 
        if eid is not None: lab[eid] = (r['encoder-label'][0] or '').replace('Dawn_ComputePassEncoder_', '')
    G = xctab.read(prefix + 'metal-gpu-intervals.xml')
    per = collections.defaultdict(float); cnt = collections.Counter(); first = {}; last = {}
    span = collections.defaultdict(lambda: [1e30, 0])
    for r in G:
        if not (r['process'][0] and proc_sub in r['process'][0]): continue
        if r['event-depth'][0] != '0': continue
        eid = hexid(r['encoder-id'])
        dur = int(r['duration'][1]); st = int(r['start'][1])
        per[eid] += dur
        s = span[eid]; s[0] = min(s[0], st); s[1] = max(s[1], st + dur)
    byl = collections.defaultdict(list)
    for eid, d in per.items():
        byl[lab.get(eid, f'?{eid:x}')].append((d, span[eid][1] - span[eid][0]))
    return lab, byl
if __name__ == '__main__':
    lab, byl = analyse(sys.argv[1])
    rows = []
    for l, v in byl.items():
        busy = sorted(x[0] for x in v); wall = sorted(x[1] for x in v)
        rows.append((sum(busy)/len(busy)/1e6, l, len(v), busy[len(busy)//2]/1e6, wall[len(wall)//2]/1e6))
    for m, l, n, med, wmed in sorted(rows, reverse=True):
        print(f'{m:8.3f} ms mean busy  median {med:8.3f}  median wall-span {wmed:8.3f}  n={n:4d}  {l[:150]}')
