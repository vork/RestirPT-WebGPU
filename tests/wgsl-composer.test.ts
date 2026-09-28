import { describe, expect, it } from 'vitest';
import { composeWgsl, WgslComposeError, mapCompilationMessages } from '../src/core/gpu/wgsl-composer.ts';

const src = {
  'a.wgsl': '#include "lib/b.wgsl"\n#include "lib/b.wgsl"\nfn a() -> u32 { return $X; }\n#if X > 2 && !OFF\nconst BIG: bool = true;\n#else\nconst BIG: bool = false;\n#endif',
  'lib/b.wgsl': 'requires unrestricted_pointer_parameters;\n#include "./c.wgsl"\nfn b() {}',
  'lib/c.wgsl': 'fn c() {}',
  'cyc1.wgsl': '#include "cyc2.wgsl"',
  'cyc2.wgsl': '#include "cyc1.wgsl"',
  'bad.wgsl': 'requires chromium_print;',
  'undef.wgsl': 'const Y = $NOPE;',
  'nested.wgsl': '#ifdef A\n#ifdef B\nAB\n#elif C\nAC\n#else\nA_\n#endif\n#else\nNA\n#endif',
};
const lf = new Set(['unrestricted_pointer_parameters']);

describe('wgsl composer', () => {
  it('includes once, resolves relative paths, hoists requires, substitutes and evaluates #if', () => {
    const r = composeWgsl('a.wgsl', { sources: src, defines: { X: 3 }, wgslLanguageFeatures: lf });
    expect(r.code.split('\n')[0]).toBe('requires unrestricted_pointer_parameters;');
    expect(r.code.match(/fn b\(\)/g)?.length).toBe(1);
    expect(r.code).toContain('fn c() {}');
    expect(r.code).toContain('return 3;');
    expect(r.code).toContain('const BIG: bool = true;');
    const lineOfA = r.code.split('\n').findIndex((l) => l.includes('fn a()'));
    expect(r.lineMap[lineOfA]).toEqual({ file: 'a.wgsl', line: 3 });
  });
  it('evaluates #else and nested branches', () => {
    expect(composeWgsl('a.wgsl', { sources: src, defines: { X: 1 }, wgslLanguageFeatures: lf }).code).toContain('const BIG: bool = false;');
    const n = (d: Record<string, boolean>) => composeWgsl('nested.wgsl', { sources: src, defines: d }).code.trim();
    expect(n({ A: true, B: true })).toBe('AB');
    expect(n({ A: true, C: true })).toBe('AC');
    expect(n({ A: true })).toBe('A_');
    expect(n({})).toBe('NA');
  });
  it('handles cycles, rejects missing files, non-shipped features, unsupported requires and undefined defines', () => {
    // include-once makes mutual includes benign (WGSL module-scope order does not matter)
    expect(composeWgsl('cyc1.wgsl', { sources: src }).code).toBe('');
    expect(() => composeWgsl('missing.wgsl', { sources: src })).toThrow(WgslComposeError);
    expect(() => composeWgsl('bad.wgsl', { sources: src })).toThrow(/not a shipped/);
    expect(() => composeWgsl('a.wgsl', { sources: src, defines: { X: 1 } })).toThrow(/requires unrestricted_pointer_parameters/);
    expect(() => composeWgsl('undef.wgsl', { sources: src })).toThrow(/undefined define/);
  });
  it('maps compilation messages back to source lines', () => {
    const r = composeWgsl('a.wgsl', { sources: src, defines: { X: 3 }, wgslLanguageFeatures: lf });
    const idx = r.code.split('\n').findIndex((l) => l.includes('fn a()'));
    const msgs = mapCompilationMessages([{ type: 'error', lineNum: idx + 1, linePos: 4, message: 'boom' } as GPUCompilationMessage], r.lineMap);
    expect(msgs[0]).toBe('error: a.wgsl:3:4: boom');
  });
});
