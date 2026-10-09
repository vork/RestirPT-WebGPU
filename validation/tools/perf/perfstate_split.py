"""perfstate_split.py <export-dir/>: per-kernel GPU busy time of encoders that ran entirely inside a 'Maximum' vs a
'Medium' GPU performance-state interval (gpu-performance-state-intervals) of a split-phase MST trace. The ratio
Medium/Maximum is a clock-sensitivity proxy (core-clock-bound kernels slow down more at the lower state)."""
import sys, os, collections, statistics, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xctab
pre = sys.argv[1]
P = xctab.read(pre + 'gpu-performance-state-intervals.xml')
ps = sorted((int(r['start'][1]), int(r['start'][1]) + int(r['duration'][1]), r['gpu-performance-state'][0]) for r in P)
pst = [p[0] for p in ps]
lab = {}
for r in xctab.read(pre + 'metal-application-encoders-list.xml'):
    v = r['encoder-id'][0]
    if v and v.startswith('0x'): lab[int(v, 16)] = (r['encoder-label'][0] or '').replace('Dawn_ComputePassEncoder_', '')
enc = collections.defaultdict(lambda: [1 << 62, 0, 0, None])
for r in xctab.read(pre + 'metal-gpu-intervals.xml'):
    if r['event-depth'][0] != '0' or not (r['process'][0] and 'Chrome Helper (GPU)' in r['process'][0]): continue
    v = r['encoder-id'][0]
    if not (v and v.startswith('0x')): continue
    eid = int(v, 16); s = int(r['start'][1]); e = s + int(r['duration'][1]); x = enc[eid]
    x[0] = min(x[0], s); x[1] = max(x[1], e); x[2] += e - s; x[3] = lab.get(eid, '?').split('[')[0].split('#')[0].strip()
by = collections.defaultdict(lambda: collections.defaultdict(list))
for eid, (s, e, b, k) in enc.items():
    i = bisect.bisect_right(pst, s) - 1
    if i < 0 or not (ps[i][0] <= s and e <= ps[i][1]): continue
    by[k][ps[i][2]].append(b / 1e6)
print(f'{"kernel":22s} {"Maximum ms (n)":>18s} {"Medium ms (n)":>18s}  Medium/Maximum')
rows = []
for k, d in by.items():
    mx, md = d.get('Maximum', []), d.get('Medium', [])
    if not mx or not md: continue
    rows.append((statistics.median(mx), k, len(mx), statistics.median(md), len(md)))
for mx, k, nx, md, nd in sorted(rows, reverse=True):
    print(f'{k:22s} {mx:10.3f} ({nx:3d})    {md:10.3f} ({nd:3d})    {md / mx:6.3f}')
