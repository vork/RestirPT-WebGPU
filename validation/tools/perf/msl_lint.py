#!/usr/bin/env python3
"""MSL lint of a Dawn dump_shaders dump (perf2 codegen checklist, perf2-plan.md §0 rule 5; docs/decisions/perf2-api.md §4).

usage: msl_lint.py MSL_DIR [--json out.json] [--all]
MSL_DIR holds <entry>[.k].metal files (split-dump.py). Per kernel (first dump of each entry; --all: every file), counts:
  bytes         MSL size
  clamps        robustness index clamps: min(...) used as an array index ('[min(') or bounded by a runtime length /
                texture size (tint_array_length…, get_width / get_height / get_depth / get_array_size / get_num_mip_levels)
  loop_idx      loops carrying a Tint loop-iteration guard ('tint_loop_idx' declarations): non-constant loop bounds
  volatile      'tint_volatile_zero' guards (a return inside a switch, rule 5)
  zero_init     zero-initialised locals / returns ('T x = {};', 'return T{};')
  divmod        calls of Tint's non-constant integer division / modulo helpers (tint_div_*, tint_mod_*)
  bvh_copies    bvh_trace instances after full inlining from the kernel entry (call-graph path count; rule 5: fewer
                inlined traversal copies — each carries its own stack and code)
Report-only."""
import json, os, re, sys

LEN = re.compile(r'tint_array_length|get_width\(|get_height\(|get_depth\(|get_array_size\(|get_num_mip_levels\(')
FUNC = re.compile(r'^(?:[A-Za-z_][\w:<>, ]*?\s+)([A-Za-z_]\w*)\(([^;{]*)\)\s*\{', re.M)


def balanced(s, i):
    """s[i] == '(' -> index after the matching ')'."""
    d = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == '(': d += 1
        elif c == ')':
            d -= 1
            if d == 0: return j + 1
    return len(s)


def functions(src):
    """Top-level function name -> body text (MSL as Tint writes it: definitions start at column 0)."""
    out = {}
    for m in FUNC.finditer(src):
        b = m.end() - 1; d = 0
        for j in range(b, len(src)):
            if src[j] == '{': d += 1
            elif src[j] == '}':
                d -= 1
                if d == 0: break
        out.setdefault(m.group(1), src[b:j + 1])
    return out


def inlined_copies(src, target='bvh_trace'):
    fns = functions(src)
    calls = {f: {} for f in fns}
    for f, body in fns.items():
        for name in re.findall(r'\b([A-Za-z_]\w*)\(', body):
            if name in fns and name != f: calls[f][name] = calls[f].get(name, 0) + 1
    memo = {}
    def inst(f, depth=0):
        if f == target: return 1
        if f in memo or depth > 200: return memo.get(f, 0)
        memo[f] = sum(c * inst(g, depth + 1) for g, c in calls[f].items())
        return memo[f]
    roots = [f for f in fns if f.endswith('_inner')] or [f for f in fns if re.search(r'^kernel\s', src, re.M)]
    return max((inst(r) for r in roots), default=0)


def lint(src):
    clamps = 0
    for m in re.finditer(r'\bmin\(', src):
        start = m.end() - 1
        arg = src[start:balanced(src, start)]
        if src[m.start() - 1] == '[' or LEN.search(arg): clamps += 1
    return {
        'bytes': len(src.encode()),
        'clamps': clamps,
        'loop_idx': len(re.findall(r'\btint_loop_idx\s*=', src)),
        'volatile': len(re.findall(r'\btint_volatile_zero\s*==', src)),
        'zero_init': len(re.findall(r'=\s*\{\}\s*;', src)) + len(re.findall(r'\breturn\s+\w+\{\}\s*;', src)),
        'divmod': len(re.findall(r'\btint_(?:div|mod)_\w*\(', src)) - len(re.findall(r'^\s*\w+\s+tint_(?:div|mod)_\w*\(', src, re.M)),
        'bvh_copies': inlined_copies(src),
    }


def main(argv):
    if not argv: print(__doc__); return 2
    d = argv[0]; out = argv[argv.index('--json') + 1] if '--json' in argv else None; every = '--all' in argv
    res = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith('.metal'): continue
        name = f[:-6]
        if not every and re.search(r'\.\d+$', name): continue
        res[name] = lint(open(os.path.join(d, f)).read())
    cols = ['bytes', 'clamps', 'loop_idx', 'volatile', 'zero_init', 'divmod', 'bvh_copies']
    print(f"{'kernel':24s} " + ' '.join(f'{c:>11s}' for c in cols))
    for k, v in sorted(res.items(), key=lambda x: -x[1]['bytes']):
        print(f'{k:24s} ' + ' '.join(f'{v[c]:11d}' for c in cols))
    if out: json.dump(res, open(out, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
