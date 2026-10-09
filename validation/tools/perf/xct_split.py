"""Per-pass GPU busy time + spill bytes from a split-phase (one compute pass per command buffer) trace export."""
import sys, collections, statistics
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xctab
pre = sys.argv[1]; proc = 'Google Chrome Helper (GPU)'
def hexid(v): return int(v[0], 16) if v[0] and str(v[0]).startswith('0x') else None
E = xctab.read(pre + 'metal-application-encoders-list.xml')
lab = {}; cb_of = {}
for r in E:
    eid = hexid(r['encoder-id'])
    if eid is None: continue
    lab[eid] = (r['encoder-label'][0] or '').replace('Dawn_ComputePassEncoder_', '')
    cb_of[eid] = hexid(r['cmdbuffer-id'])
G = xctab.read(pre + 'metal-gpu-intervals.xml')
busy = collections.defaultdict(int); span = {}
other_busy = 0; t_min = 1 << 62; t_max = 0
for r in G:
    if r['event-depth'][0] != '0': continue
    st = int(r['start'][1]); du = int(r['duration'][1])
    if not (r['process'][0] and proc in r['process'][0]):
        other_busy += du; continue
    t_min = min(t_min, st); t_max = max(t_max, st + du)
    eid = hexid(r['encoder-id']); busy[eid] += du
    s = span.setdefault(eid, [st, st + du]); s[0] = min(s[0], st); s[1] = max(s[1], st + du)
kind = lambda l: l.split('[')[0].split('#')[0].strip()
per = collections.defaultdict(list); perw = collections.defaultdict(list); pieces = collections.Counter()
for eid, b in busy.items():
    l = lab.get(eid)
    if l is None: continue
    per[kind(l)].append(b / 1e6); perw[kind(l)].append((span[eid][1] - span[eid][0]) / 1e6)
# spills
spill = collections.defaultdict(set)
try:
    for r in xctab.read(pre + 'graphics-compiler-spill-events.xml'):
        if not (r['process'][0] and proc in r['process'][0]): continue
        eid = int(r['encoder-id'][1]); l = lab.get(eid)
        spill[kind(l) if l else f'?{eid:x}'].add(int(r['spilled-bytes'][1]))
except Exception as e: print('spill read failed', e)
print(f'chrome GPU window {(t_max - t_min)/1e9:.2f} s; other processes busy {other_busy/1e6:.0f} ms in that trace')
tot = 0
rows = []
for k, v in per.items():
    # encoders per frame of the same kind (e.g. rs_t_select twice, rs_args several): count per occurrence
    rows.append((statistics.mean(v), k, len(v), statistics.median(v), statistics.median(perw[k]), sorted(spill.get(k, []))))
for m, k, n, med, wmed, sp in sorted(rows, reverse=True):
    print(f'{k:22s} mean {m:8.3f} ms  median {med:8.3f}  wall-span med {wmed:8.3f}  n={n:4d}  spill/thread B {sp}')
for k, v in spill.items():
    if k not in per: print('spill without timing', k, sorted(v))
