// HUD timing attribution (timestamps.ts attributeFrame): the present render pass's early begin sample (Metal) must not
// absorb the untimed ReSTIR passes (Q3); the untimed remainder is reported separately and the lines add up to the span.
import { describe, expect, it } from 'vitest';
import { attributeFrame } from '../../src/core/render/timestamps.ts';

const ns = (ms: number) => BigInt(Math.round(ms * 1e6)) + 1_000_000_000n;

describe('attributeFrame', () => {
  it('clamps an early render-pass begin to the previous timed end and reports the untimed rest', () => {
    // raw pairs seen on Sponza (Chrome / Metal): present begins during the primary pass
    const r = attributeFrame([
      { name: 'primary', begin: ns(0), end: ns(0.4) },
      { name: 'present', begin: ns(0.26), end: ns(9.37) },
      { name: 'tonemap', begin: ns(9.24), end: ns(9.31) },
    ]);
    expect(r.perName.get('primary')).toBeCloseTo(0.4, 6);
    expect(r.perName.get('tonemap')).toBeCloseTo(0.07, 6);
    expect(r.perName.get('present')).toBeCloseTo(0.06, 6);
    expect(r.spanMs).toBeCloseTo(9.37, 6);
    expect(r.untimedMs).toBeCloseTo(9.37 - 0.53, 6);
  });
  it('breaks end ties by reservation (encode) order', () => {
    const r = attributeFrame([
      { name: 'primary', begin: ns(0), end: ns(0.9) },
      { name: 'tonemap', begin: ns(37.09), end: ns(37.22) },
      { name: 'present', begin: ns(0.33), end: ns(37.22) },
    ]);
    expect(r.perName.get('tonemap')).toBeCloseTo(0.13, 6);
    expect(r.perName.get('present')).toBe(0);
    expect(r.untimedMs).toBeCloseTo(37.22 - 1.03, 6);
  });
  it('drops unwritten pairs and keeps their names at 0', () => {
    const r = attributeFrame([{ name: 'primary', begin: ns(0), end: ns(1) }, { name: 'restir', begin: 0n, end: 0n }]);
    expect(r.perName.get('restir')).toBe(0);
    expect(r.spanMs).toBeCloseTo(1, 6);
    expect(r.untimedMs).toBe(0);
  });
});
