"""ctr_ids.py <trace> [maxrows]: stream gpu-counter-value and print row count, distinct counter ids with value stats."""
import subprocess, sys, collections, xml.etree.ElementTree as ET
tr = sys.argv[1]; maxrows = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
p = subprocess.Popen(['xcrun', 'xctrace', 'export', '--input', tr, '--xpath',
                      '/trace-toc/run[@number="1"]/data/table[@schema="gpu-counter-value"]'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
ids = {}; st = collections.defaultdict(lambda: [0, 0.0, 0.0, 0]); n = 0
for ev, el in ET.iterparse(p.stdout, events=('end',)):
    if el.tag != 'row':
        continue
    vals = []
    for c in el:
        if 'id' in c.attrib: ids[c.attrib['id']] = c.text
        vals.append(ids[c.attrib['ref']] if 'ref' in c.attrib else c.text)
    cid, v = vals[1], float(vals[2])
    s = st[cid]; s[0] += 1; s[1] += v; s[2] = max(s[2], v); s[3] += v != 0
    n += 1; el.clear()
    if n >= maxrows: p.kill(); break
print(f'rows {n} counters {len(st)}: ' + ', '.join(f'{k}:n={s[0]} mean={s[1]/s[0]:.3g} max={s[2]:.3g} nz={s[3]}' for k, s in sorted(st.items(), key=lambda x: int(x[0]))))
