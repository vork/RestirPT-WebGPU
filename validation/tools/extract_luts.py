#!/usr/bin/env python3
"""extract_luts.py — generate src/core/render/luts/cycles-luts.ts from Cycles' shader.tables (plan §1.5; math.md#bsdf-v2).

  python validation/tools/extract_luts.py [--src shader.tables] [--out src/core/render/luts/cycles-luts.ts]

Without --src the file is downloaded from SOURCE_URL (Blender tag v5.1.2). The script:
  * extracts table_ggx_gen_schlick_ior_s[4096], table_ggx_gen_schlick_s[4096], table_ggx_E[1024], table_ggx_Eavg[32]
    verbatim (decimal literals as written, the 'f' suffix dropped);
  * checks every literal: decimal -> f64 -> f32 (what `new Float32Array([...])` does) equals the correctly rounded
    decimal -> f32 (what the C compiler does for `0.123456f`), so the GPU sees bit-identical floats to Cycles;
  * checks the element counts and the table sums against gap-bsdf §3.5 (to 1e-5);
  * records the source URL, the sha256 of the source file and the sha256 of each table's little-endian f32 bytes.
Stdlib only (no numpy), so it runs with any python3.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import struct
import sys
import urllib.request
from fractions import Fraction
from pathlib import Path

SOURCE_URL = 'https://raw.githubusercontent.com/blender/blender/v5.1.2/intern/cycles/scene/shader.tables'
# (name, count, expected sum from gap-bsdf §3.5 / math.md#bsdf-v2)
TABLES = [
    ('table_ggx_gen_schlick_ior_s', 4096, 184.247126),
    ('table_ggx_gen_schlick_s', 4096, 764.768581),
    ('table_ggx_E', 1024, 849.736018),
    ('table_ggx_Eavg', 32, 25.879163),
]
# Tier-2 (MULTI_GGX) glass tables, gap-glass §2.7 (checksums verified there). Written to a SEPARATE module so the
# Tier-1 block (cycles-luts.ts) is unchanged; M3b uses them only in the CPU tests U-G4 (albedo, η > 1) and U-G9
# (checksums, energy-scale multipliers); the GPU renderer is Tier 1 (validation forces distribution = 'GGX').
TABLES_GLASS = [
    ('table_ggx_glass_E', 4096, 3758.664856),
    ('table_ggx_glass_Eavg', 256, 235.910746),
    ('table_ggx_glass_inv_E', 4096, 3477.751955),
    ('table_ggx_glass_inv_Eavg', 256, 216.090299),
]
REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / 'src/core/render/luts/cycles-luts.ts'
DEFAULT_GLASS_OUT = REPO / 'src/core/render/luts/cycles-glass-luts.ts'


def f32_bits(x: float) -> int:
    return struct.unpack('<I', struct.pack('<f', x))[0]


def bits_f32(b: int) -> float:
    return struct.unpack('<f', struct.pack('<I', b))[0]


def correctly_rounded_f32(lit: str) -> int:
    """Bits of the f32 nearest to the exact decimal value (ties to even)."""
    exact = Fraction(lit)
    cand = f32_bits(float(exact))  # float() is correctly rounded to f64; f32 pack rounds again (maybe double rounding)
    best = None
    for b in (cand - 1, cand, cand + 1):
        if b < 0:
            continue
        v = bits_f32(b)
        if not math.isfinite(v):
            continue
        d = abs(Fraction(v) - exact)
        key = (d, b & 1)  # ties to even mantissa
        if best is None or key < best[0]:
            best = (key, b)
    assert best is not None
    return best[1]


def parse_table(text: str, name: str, count: int) -> list[str]:
    m = re.search(r'static const float ' + re.escape(name) + r'\[(\d+)\]\s*=\s*\{(.*?)\};', text, re.S)
    if not m:
        raise SystemExit(f'{name} not found')
    if int(m.group(1)) != count:
        raise SystemExit(f'{name}: declared size {m.group(1)} != {count}')
    lits = [t.strip() for t in m.group(2).replace('\n', ' ').split(',') if t.strip()]
    lits = [t[:-1] if t.endswith('f') else t for t in lits]
    if len(lits) != count:
        raise SystemExit(f'{name}: {len(lits)} literals != {count}')
    for t in lits:
        if not re.fullmatch(r'-?\d+(\.\d+)?([eE][-+]?\d+)?', t):
            raise SystemExit(f'{name}: unexpected literal {t!r}')
    return lits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', help='local copy of shader.tables (default: download SOURCE_URL)')
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    ap.add_argument('--glass-out', default=str(DEFAULT_GLASS_OUT))
    args = ap.parse_args()

    if args.src:
        raw = Path(args.src).read_bytes()
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as r:
            raw = r.read()
    src_sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode('utf-8')
    write_module(text, src_sha, TABLES, Path(args.out),
                 '// Cycles precomputed microfacet albedo tables, copied verbatim (plan §1.5; math.md#bsdf-v2; gap-bsdf §3.5).')
    write_module(text, src_sha, TABLES_GLASS, Path(args.glass_out),
                 '// Cycles Tier-2 (MULTI_GGX) glass albedo tables, copied verbatim (gap-glass §2.7; CPU tests U-G4 / U-G9 only).',
                 checksum_name='GLASS_LUT_CHECKSUMS')
    return 0


def write_module(text: str, src_sha: str, tables: list[tuple[str, int, float]], out: Path, title: str,
                 checksum_name: str = 'LUT_CHECKSUMS') -> None:
    out_lines = [
        '// GENERATED by validation/tools/extract_luts.py — do not edit by hand.',
        title,
        f'// Source: {SOURCE_URL}',
        f'// Source sha256: {src_sha}',
        '// Every literal was checked: decimal -> f64 -> f32 (Float32Array) == correctly rounded decimal -> f32 (C `0.1f`).',
        '// Layout (lookup_table.h): 3D tables data[z*n*n + y*n + x], 2D data[y*n + x]; x = rough, y = mu, z per table.',
        '',
        f"export const LUT_SOURCE_URL = '{SOURCE_URL}';",
        f"export const LUT_SOURCE_SHA256 = '{src_sha}';",
        '',
    ]
    meta = []
    for name, count, want_sum in tables:
        lits = parse_table(text, name, count)
        bits = []
        for t in lits:
            via_f64 = f32_bits(float(t))
            exact = correctly_rounded_f32(t)
            if via_f64 != exact:
                raise SystemExit(f'{name}: literal {t} double-rounds differently (f64 path {via_f64:#x} vs {exact:#x})')
            bits.append(exact)
        vals = [bits_f32(b) for b in bits]
        s = math.fsum(vals)
        if abs(s - want_sum) > 1e-5:
            raise SystemExit(f'{name}: sum {s:.6f} != expected {want_sum}')
        tsha = hashlib.sha256(b''.join(struct.pack('<I', b) for b in bits)).hexdigest()
        meta.append((name, count, s, tsha))
        ident = name.upper()
        out_lines.append(f'/** {name}[{count}] (sum {s:.6f}, f32 LE sha256 {tsha}). */')
        out_lines.append(f'export const {ident} = new Float32Array([')
        for i in range(0, count, 16):
            out_lines.append('  ' + ', '.join(lits[i:i + 16]) + ',')
        out_lines.append(']);')
        out_lines.append('')
        print(f'{name}: {count} floats, sum {s:.6f}, sha256(f32) {tsha}')

    out_lines.append('/** Per-table checksums recorded at extraction (tests/material/luts.test.ts re-verifies them). */')
    out_lines.append(f'export const {checksum_name} = {{')
    for name, count, s, tsha in meta:
        out_lines.append(f"  {name}: {{ count: {count}, sum: {s:.6f}, sha256F32: '{tsha}' }},")
    out_lines.append('} as const;')
    out_lines.append('')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(out_lines))
    print(f'wrote {out} (source sha256 {src_sha})')


if __name__ == '__main__':
    sys.exit(main())
