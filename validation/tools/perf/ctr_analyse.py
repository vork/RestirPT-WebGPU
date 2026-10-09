"""Per-kernel GPU counters from a split-phase (one compute pass per command buffer) trace.
usage: python3 ctr_analyse.py <trace> <export-dir/> [--proc 'Google Chrome Helper (GPU) (PID)']
Exports metal-gpu-intervals, gpu-counter-info, gpu-counter-value (streamed), metal-shader-profiler-intervals.
Kernel label: metal-application-encoders-list when present, else the GPU instrument's event-label
('Command Buffer N:Dawn_ComputePassEncoder_<label>'). Counter samples (gpu-counter-value timestamps) are attributed to
the Chrome GPU-process encoder interval that contains them; per kernel the mean / max of each counter is reported."""
import bisect, collections, json, os, re, subprocess, sys, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xctab

trace, out = sys.argv[1], sys.argv[2]
proc = sys.argv[sys.argv.index('--proc') + 1] if '--proc' in sys.argv else 'Google Chrome Helper (GPU)'
os.makedirs(out, exist_ok=True)
def export(schema, extra=''):
    path = os.path.join(out, f'{schema}{extra.replace("=", "").replace(" ", "")}.xml')
    if not os.path.exists(path):
        x = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"{extra}]'
        with open(path, 'wb') as f: subprocess.run(['xcrun', 'xctrace', 'export', '--input', trace, '--xpath', x], stdout=f, stderr=subprocess.DEVNULL)
    return path
kind = lambda l: l.split('[')[0].split('#')[0].strip()

# encoder intervals of the Chrome GPU process
G = xctab.read(export('metal-gpu-intervals'))
iv = []
for r in G:
    if r['event-depth'][0] != '0' or not (r['process'][0] and proc in r['process'][0]): continue
    m = re.search(r'Dawn_ComputePassEncoder_([^\s]+)', r['event-label'][0] or '')
    lab = m.group(1) if m else (r['event-label'][0] or '?').split('  (')[0]
    st = int(r['start'][1]); du = int(r['duration'][1])
    iv.append((st, st + du, kind(lab), r['encoder-id'][1]))
iv.sort()
starts = [x[0] for x in iv]
busy = collections.defaultdict(int); encs = collections.defaultdict(set)
for s, e, k, eid in iv: busy[k] += e - s; encs[k].add(eid)
nenc = {k: len(v) for k, v in encs.items()}  # GPU intervals of one encoder can be split (preemption): count encoders
def find(t):
    i = bisect.bisect_right(starts, t) - 1
    return iv[i][2] if i >= 0 and iv[i][0] <= t <= iv[i][1] else None

# counter names
toc = subprocess.run(['xcrun', 'xctrace', 'export', '--input', trace, '--toc'], capture_output=True, text=True).stdout
names = {}
for p in sorted(set(re.findall(r'schema="gpu-counter-info" counter-profile="(\d+)"', toc))):
    for r in xctab.read(export('gpu-counter-info', f' and @counter-profile="{p}"')):
        names[r['counter-id'][1]] = (r['name'][1], r['type'][1], r['description'][1])
# counter samples (streamed)
acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0, 0.0]))
ids = {}; nrows = 0; nin = 0
p = subprocess.Popen(['xcrun', 'xctrace', 'export', '--input', trace, '--xpath', '/trace-toc/run[@number="1"]/data/table[@schema="gpu-counter-value"]'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
for ev, el in ET.iterparse(p.stdout, events=('end',)):
    if el.tag != 'row': continue
    vals = []
    for c in el:
        if 'id' in c.attrib: ids[c.attrib['id']] = c.text
        vals.append(ids[c.attrib['ref']] if 'ref' in c.attrib else c.text)
    nrows += 1; el.clear()
    k = find(int(vals[0]))
    if k is None: continue
    nin += 1; a = acc[k][vals[1]]; v = float(vals[2]); a[0] += 1; a[1] += v; a[2] = max(a[2], v)
# shader timeline (per-shader execution intervals inside the kicks)
sh = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
try:
    for r in xctab.read(export('metal-shader-profiler-intervals')):
        if not (r['process'][0] and proc in r['process'][0]): continue
        st = int(r['start'][1]); du = int(r['duration'][1]); k = find(st + du // 2)
        s = sh[k or '?'][r['name'][1]]; s[0] += du; s[1] += float(r['percent-of-kick'][1] or 0)
except Exception as e: print('shader timeline:', e)

print(f'gpu intervals {len(iv)} kinds {len(busy)}; counter rows {nrows} (inside Chrome encoders {nin}); counters known: {names}')
res = {}
for k in sorted(busy, key=lambda k: -busy[k] / nenc[k]):
    c = {names.get(cid, (cid,))[0]: {'mean': s[1] / s[0], 'max': s[2], 'n': s[0]} for cid, s in acc[k].items()}
    t = {n: {'ms': v[0] / 1e6} for n, v in sh.get(k, {}).items()}
    res[k] = {'meanBusyMs': busy[k] / nenc[k] / 1e6, 'n': nenc[k], 'counters': c, 'shaderTimeline': t}
    cs = ', '.join(f'{n} mean {x["mean"]:.3g} max {x["max"]:.3g} (n={x["n"]})' for n, x in c.items()) or '-'
    ts = ', '.join(f'{n} {x["ms"]:.1f} ms' for n, x in sorted(t.items(), key=lambda q: -q[1]['ms'])[:3]) or '-'
    print(f'{k:22s} busy {busy[k] / nenc[k] / 1e6:8.3f} ms n={nenc[k]:4d} | counters: {cs} | shader timeline: {ts}')
json.dump({'trace': trace, 'counterNames': names, 'kernels': res}, open(os.path.join(out, 'per-kernel.json'), 'w'), indent=1)
