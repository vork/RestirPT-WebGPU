// Integer alias table (alias.ts; math.md#light-selection, open item 5): realized pmf from the stored integers,
// exact bucket uniformity with power-of-two padding, padding never selectable, determinism, CPU sampler ≡ realized pmf.
import { describe, expect, it } from 'vitest';
import { ALIAS_ONE, aliasLog2, aliasSample, buildAliasTable, packAliasEntries, realizedPmf } from '../../src/core/render/alias.ts';

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s; };
}

describe('alias table', () => {
  it('pads to a power of two ≥ 2 and returns undefined for Σw = 0', () => {
    expect(aliasLog2(1)).toBe(1);
    expect(aliasLog2(2)).toBe(1);
    expect(aliasLog2(3)).toBe(2);
    expect(aliasLog2(1025)).toBe(11);
    expect(buildAliasTable([0, 0])).toBeUndefined();
    expect(buildAliasTable([])).toBeUndefined();
    expect(() => buildAliasTable([1, -1])).toThrow();
    expect(() => buildAliasTable([1, NaN])).toThrow();
  });

  it('realized pmf sums to 1 exactly, is within 2^-16/n of the target, and every w > 0 keeps pmf > 0', () => {
    const rnd = lcg(3);
    for (const n of [1, 2, 3, 5, 17, 100, 1000, 4097]) {
      const w: number[] = Array.from({ length: n }, (_, i) => (i % 7 === 3 ? 0 : (rnd() / 2 ** 32) ** 4 * 1e3 + (i % 11 === 0 ? 1e-9 : 0)));
      if (!w.some((x) => x > 0)) w[0] = 1;
      const t = buildAliasTable(w)!;
      expect(t.n).toBe(2 ** t.log2n);
      expect(t.n).toBeGreaterThanOrEqual(Math.max(2, n));
      let s = 0;
      const sum = w.reduce((a, b) => a + b, 0);
      for (let i = 0; i < n; i++) {
        s += t.pmf[i];
        if (w[i] > 0) expect(t.pmf[i], `entry ${i}`).toBeGreaterThan(0);
        else expect(t.pmf[i]).toBe(0);
        // each bucket's threshold is off by ≤ ½ ulp of 2^-16 (plus the ≥ 1 floor), and an entry appears in ≤ n buckets
        expect(Math.abs(t.pmf[i] - w[i] / sum)).toBeLessThan(n * 1.5 / (ALIAS_ONE * t.n) + 1e-15);
      }
      expect(Math.abs(s - 1)).toBeLessThan(1e-12);
      for (let b = 0; b < t.n; b++) expect(t.q[b]).toBeLessThanOrEqual(65535);
    }
  });

  it('padding entries (and zero-weight entries) are never selected: exhaustive over all 2^16 thresholds per bucket', () => {
    const w = [3, 0, 1, 0.5, 2]; // n = 8, three padding entries
    const t = buildAliasTable(w)!;
    expect(t.n).toBe(8);
    const counts = new Float64Array(t.n);
    for (let b = 0; b < t.n; b++) {
      for (let thr = 0; thr < 65536; thr++) counts[aliasSample(t, ((b << (32 - t.log2n)) | thr) >>> 0)]++;
    }
    for (let k = 0; k < t.n; k++) {
      const p = counts[k] / (65536 * t.n);
      if (k < w.length) expect(p).toBe(t.pmf[k]); // the sampler realizes exactly the stored pmf
      else expect(counts[k]).toBe(0);
    }
    expect(counts[1]).toBe(0); // the zero-weight real entry
  });

  it('log2n > 16 uses the second hash for the threshold (bucket bits and threshold bits independent)', () => {
    const n = 70000;
    const w = Array.from({ length: n }, (_, i) => 1 + (i % 3));
    const t = buildAliasTable(w)!;
    expect(t.log2n).toBe(17);
    // exhaustive over thresholds for a few buckets: counts match q
    for (const b of [0, 12345, 69999, 100000]) {
      let keep = 0;
      for (let thr = 0; thr < 65536; thr++) if (aliasSample(t, (b << 15) >>> 0, thr) === b) keep++;
      expect(keep).toBe(t.alias[b] === b ? 65536 : t.q[b]);
    }
  });

  it('is deterministic (bitwise identical rebuilds) and packs (q, alias) pairs', () => {
    const w = Array.from({ length: 300 }, (_, i) => Math.sin(i) ** 2 + 0.01);
    const a = buildAliasTable(w)!, b = buildAliasTable([...w])!;
    expect(a.q).toEqual(b.q);
    expect(a.alias).toEqual(b.alias);
    expect(a.pmf).toEqual(b.pmf);
    const p = packAliasEntries(a);
    expect(p.length).toBe(2 * a.n);
    expect(p[2 * 7]).toBe(a.q[7]);
    expect(p[2 * 7 + 1]).toBe(a.alias[7]);
    expect(realizedPmf(a.q, a.alias, a.count)).toEqual(a.pmf);
  });

  it('χ² of the CPU sampler over random hashes matches the realized pmf', () => {
    const w = [5, 1, 0.25, 3, 0.75, 2, 0.1];
    const t = buildAliasTable(w)!;
    const rnd = lcg(11);
    const N = 400_000;
    const c = new Float64Array(w.length);
    for (let i = 0; i < N; i++) c[aliasSample(t, rnd())]++;
    let chi2 = 0;
    for (let k = 0; k < w.length; k++) chi2 += (c[k] - N * t.pmf[k]) ** 2 / (N * t.pmf[k]);
    expect(chi2).toBeLessThan(22.46); // χ²_6 at p = 0.001
  });
});
