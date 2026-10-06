// T6(a) CPU part (restir-temporal-api.md §6.1; math.md#temporal): the temporal contribution weights are a partition of
// unity over the producibility sets (ĉ_c + ĉ_p = 1 where both are producible, ĉ_c = 1 where π_p = 0, capped and
// uncapped c_p, zeros), Talbot m_c + m_p = 1, E2 (π_p := 0 on deep classes) gives ĉ_c ≡ 1, and the W_Y formula
// satisfies W_Y·p̂(Y) = Σw̃·π_s(Y)/(c_c π_c + c_p π_p); an exact toy expectation checks unbiasedness. OWNER T-B.
import { describe, expect, it } from 'vitest';
import { contribWeights, temporalSelectContrib, tmisContribW, tmisTalbotMc, tmisTalbotMp } from './tmis-ref.ts';

/** Deterministic PRNG (mulberry32). */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** A π-like value: 0 with probability 0.2, else log-uniform over [1e-6, 1e4]. */
const piVal = (r: () => number) => (r() < 0.2 ? 0 : 10 ** (-6 + 10 * r()));

describe('T6(a) CPU: temporal contribution MIS', () => {
  it('ĉ_c + ĉ_p = 1 where both producible; ĉ_c = 1 where π_p = 0 (c_p ∈ 0…20 incl. capped c_prev)', () => {
    const r = rng(1);
    for (let n = 0; n < 200000; n++) {
      const cPrev = r() < 0.3 ? 20 + 400 * r() : Math.floor(21 * r());
      const cP = Math.min(20, cPrev);
      const piC = r() < 0.05 ? 0 : piVal(r), piP = piVal(r);
      const w = contribWeights(1, cP, piC, piP);
      if (piC > 0 && piP > 0 && cP > 0) expect(Math.abs(w.cc + w.cp - 1)).toBeLessThan(1e-12);
      if (piC > 0 && (piP === 0 || cP === 0)) { expect(w.cc).toBe(1); expect(w.cp).toBe(0); }
      if (piC === 0 && piP === 0) { expect(w.cc + w.cp).toBe(0); }
    }
  });

  it('E2: π_p := 0 on deep classes ⇒ ĉ_c ≡ 1 for every canonical with p̂ > 0', () => {
    const r = rng(2);
    for (let n = 0; n < 20000; n++) expect(contribWeights(1, Math.floor(21 * r()), piVal(r) + 1e-9, 0).cc).toBe(1);
  });

  it('Talbot |m_c + m_p − 1| < 1e-6 at the same point (both denominators equal); edge cases := 1 / := 0', () => {
    const r = rng(3);
    for (let n = 0; n < 200000; n++) {
      const cP = Math.min(20, Math.floor(30 * r()));
      const ph = piVal(r), pp = piVal(r);
      const mc = tmisTalbotMc(1, ph, cP, pp), mp = tmisTalbotMp(1, ph, cP, pp);
      if (ph + cP * pp > 0) expect(Math.abs(mc + mp - 1)).toBeLessThan(1e-6);
      else { expect(mc).toBe(1); expect(mp).toBe(0); }
    }
  });

  it('W_Y·p̂(Y) = Σw̃·π_s(Y)/(c_c π_c + c_p π_p); zero / non-finite denominators give 0', () => {
    const r = rng(4);
    for (let n = 0; n < 100000; n++) {
      const cP = Math.floor(21 * r()), piC = piVal(r) + 1e-12, piP = piVal(r), wSum = piVal(r) + 1e-12;
      for (const [piSel] of [[piC], [piP]]) {
        const W = tmisContribW(piSel, piC, piP, 1, cP, wSum);
        expect(Math.abs(W * piC - (wSum * piSel) / (piC + cP * piP))).toBeLessThanOrEqual(1e-12 * Math.max(1, Math.abs(W * piC)));
      }
    }
    expect(tmisContribW(1, 0, 1, 1, 1, 1)).toBe(0);
    expect(tmisContribW(1, 1, 0, 1, 0, 1)).toBe(1);
    expect(tmisContribW(1, 1, Infinity, 1, 1, 1)).toBe(0);
  });

  it('selection: no valid forward shift ⇒ canonical kept with c = 1 + c_p (TD13); both weights 0 ⇒ empty, c = 1 + c_p', () => {
    const base = { pc: 2, Wc: 0.5, cPrev: 35, cap: 20, fwdOk: false, lumFY: 0, Jp: 0, Wp: 0, lumFst: 0, piPXc: 0.7, u0: 0.3, u1: 0.9 };
    const a = temporalSelectContrib(base);
    expect(a.sel).toBe('c');
    expect(a.c).toBe(21);
    expect(a.W).toBeCloseTo((2 * 0.5) / (2 + 20 * 0.7), 12);   // π_p(X_c) still enters (producibility partition)
    const e = temporalSelectContrib({ ...base, Wc: 0 });
    expect(e.sel).toBe('empty');
    expect(e.c).toBe(21);
  });

  it('unbiasedness of the one-candidate-each estimator on a discrete toy domain (exact expectation)', () => {
    // Domain {y0, y1, y2}; target f; the canonical draws X_c ~ p_c, the temporal X_p ~ p_p is shifted by the identity
    // (J = 1) and is producible only on {y0, y1}. E[f(Y)·W_Y] must equal Σ f exactly (enumerate both draws and u).
    const f = [1.0, 2.5, 0.7], pcs = [0.5, 0.3, 0.2], pps = [0.6, 0.4, 0], cP = 4;
    // W_c = 1/p_c(x), W_p = 1/p_p(x); π_c = f, π_p = f/…: π_p(y) = p̂_{t−1}(T⁻¹y)·J = f(y) on {y0,y1}, 0 on y2.
    const piP = (y: number) => (pps[y] > 0 ? f[y] : 0);
    let E = 0;
    const U = 400;
    for (let xc = 0; xc < 3; xc++) for (let xp = 0; xp < 3; xp++) {
      const pr = pcs[xc] * pps[xp];
      if (pr === 0) continue;
      for (let k = 0; k < U; k++) {
        const u = (k + 0.5) / U;
        const o = temporalSelectContrib({ pc: f[xc], Wc: 1 / pcs[xc], cPrev: cP, cap: 20, fwdOk: true, lumFY: f[xp], Jp: 1, Wp: 1 / pps[xp], lumFst: f[xp], piPXc: piP(xc), u0: 0, u1: u });
        const y = o.sel === 'p' ? xp : xc;
        E += (pr / U) * f[y] * o.W;
      }
    }
    expect(E).toBeCloseTo(f.reduce((a, b) => a + b, 0), 2);
  });
});
