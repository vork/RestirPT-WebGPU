// T6(a) CPU part (restir-api.md §6.1; math.md#paired-mis): partition of unity of the defensive pairwise MIS at the
// canonical sample and at every shifted sample, J-freedom of the partner weight, and the f64 reference itself.
import { describe, expect, it } from 'vitest';
import { misCanonicalTerm, misPartnerWeight, misPartnerWeightExplicitJ, misWeightsAt, pairedWeightsRef } from './mis-ref.ts';
import { Pcg32 } from '../../src/core/render/restir/pairing.ts';

const rng = new Pcg32(7, 7, 0, 99);
const u = () => rng.next() / 2 ** 32;
const pick = <T>(a: T[]) => a[Math.floor(u() * a.length)];
const conf = () => pick([1, 2, 3.7, 7, 20, 49, 343]);
const val = (pZero: number) => (u() < pZero ? 0 : Math.exp(8 * (u() - 0.5)));

describe('T6(a) CPU: pairwise MIS partition of unity', () => {
  it('Σm = 1 at y = X_c and at y = Y_j for random k ∈ 1…6, confidences, FAILED slots and p̂ = 0 (1e5 configs)', () => {
    let worst = 0;
    for (let n = 0; n < 100000; n++) {
      const k = 1 + Math.floor(u() * 6);
      const cc = conf();
      const c = Array.from({ length: k }, conf);
      // at an arbitrary point y (covers X_c: pFrom = lum H; Y_j: pFrom[j] = p̂_j(X_j)/J, pcY = lum G/J)
      const pcY = val(0.15);
      const pFrom = Array.from({ length: k }, () => val(0.25));
      const { mc, mj } = misWeightsAt(cc, c, pcY, pFrom);
      const s = mc + mj.reduce((x, y) => x + y, 0);
      worst = Math.max(worst, Math.abs(s - 1));
      expect(mc).toBeGreaterThanOrEqual(1 / (k + 1) - 1e-15);
      for (const m of mj) expect(m).toBeGreaterThanOrEqual(0);
    }
    console.log(`[T6(a) CPU] max |Σm − 1| = ${worst.toExponential(2)}`);
    expect(worst).toBeLessThan(1e-12);
  });

  it('the J-free partner weight equals the J-explicit GRIS form for every J > 0', () => {
    for (let n = 0; n < 20000; n++) {
      const k = 1 + Math.floor(u() * 6), a = conf() / k, cj = conf(), pj = val(0.1), lumG = val(0.1), J = Math.exp(20 * (u() - 0.5));
      const f = misPartnerWeight(a, cj, pj, lumG, k), e = misPartnerWeightExplicitJ(a, cj, pj, lumG, J, k);
      expect(Math.abs(f - e)).toBeLessThanOrEqual(1e-12 * Math.max(1, f));
    }
  });

  it('degenerate cases: FAILED partners get m_j = 0, the canonical keeps its share; k = 0 is the canonical', () => {
    const r = pairedWeightsRef(1, 2, 0.5, [{ cj: 1, pj: 3, Wj: 1, lumG: 0, J: 0, valid: false, lumH: 0 }]);
    expect(r.mj).toEqual([0]);
    expect(r.mc).toBe(1);
    expect(r.w[0]).toBe(1);
    expect(r.cOut).toBe(2);
    expect(misCanonicalTerm(1, 0, 1, 0)).toBe(1);
    expect(misPartnerWeight(1, 1, 0, 0, 1)).toBe(0);
    const r0 = pairedWeightsRef(1, 2, 0.5, []);
    expect(r0.wSum).toBe(1);
  });
});
