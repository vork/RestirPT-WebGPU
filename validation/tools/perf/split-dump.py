"""Splits the console JSONL of a dump_shaders run into msl/<entry>[.k].metal and msl/wgsl/<k>-<first entry>.wgsl."""
import json, re, sys, os, collections
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out + '/wgsl', exist_ok=True)
seen = collections.Counter(); idx = []
for i, l in enumerate(open(src)):
    d = json.loads(l); x = d['x']
    if x.startswith('/* Dumped generated MSL */'):
        m = re.search(r'kernel void \w+\([^)]*\)[^{]*\{(.*?)\n\}', x, re.S)
        inner = re.findall(r'\((\w+)_inner\(', m.group(1)) if m else []
        name = inner[-1] if inner else 'unknown'
        seen[name] += 1
        fn = f'{name}.metal' if seen[name] == 1 else f'{name}.{seen[name]}.metal'
        open(f'{out}/{fn}', 'w').write(x); idx.append((i, 'msl', fn, len(x)))
    elif x.startswith('// Dumped WGSL:'):
        eps = re.findall(r'@compute[^\n]*\n\s*fn (\w+)', x) or re.findall(r'fn (\w+)\(', x)[:1]
        fn = f'wgsl/{i:02d}-{eps[0] if eps else "module"}.wgsl'
        open(f'{out}/{fn}', 'w').write(x); idx.append((i, 'wgsl', fn, len(x)))
with open(f'{out}/INDEX.txt', 'w') as f:
    for r in idx: f.write(f'{r[0]:3d} {r[1]:4s} {r[3]:8d} {r[2]}\n')
print(open(f'{out}/INDEX.txt').read())
