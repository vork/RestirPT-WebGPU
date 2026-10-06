// U8 plant 9 (RSF_PLANT_U8_FAILED_K, mis.wgsl spatial_resample; restir-temporal-api.md §6.5, Changelog D-5): sign of the
// planted bias, confirmed on the T7 GRIS toy in f64 with the production MIS formulas (mis-ref.ts twins of
// mis_canonical_term / mis_partner_weight) and the production S_c rule: K = 4 pixels on [0, 1), f_i(x) = 1 + ½ sin 2π(x + ψ_i),
// initial RIS with 4 uniform candidates, one round of fully connected paired reuse into pixel 0 (k = 3), translation
// shifts T_{i→j}(x) = fract(x + φ_i − φ_j) undefined on a pair-specific base interval (symmetric).
//   plant off  S_c = every accepted partner (VALID or FAILED)              ⇒ E[f_0(Y)·W] = ∫f_0 = 1 (unbiased)
//   plant on   partners whose G_j is not VALID are dropped from S_c (k, a = c_c/k, m_c, c_out)
//              ⇒ the remaining techniques' weights are renormalised over fewer techniques exactly when the dropped
//                one contributes nothing: Σ_t E[m_t] > 1 at points the dropped technique can produce ⇒ Δ > 0.
import { describe, expect, it } from 'vitest';
import { misCanonicalTerm, misPartnerWeight } from './mis-ref.ts';

const K = 4;
const PSI = [0.1, 0.4, 0.7, 0.25];
const PHI = [0, 0.17, 0.55, 0.81];
const f = (i: number, x: number) => 1 + 0.5 * Math.sin(2 * Math.PI * (x + PSI[i]));
const fract = (x: number) => x - Math.floor(x);
/** T_{i→j}(x): y, or undefined where the base coordinate falls in the pair's hole (width 0.4). */
function T(i: number, j: number, x: number): number | undefined {
  const u = fract(x + PHI[i]);
  const lo = fract(0.13 * (Math.min(i, j) * 4 + Math.max(i, j)) + 0.07);
  if (fract(u - lo) < 0.4) return undefined;
  return fract(u - PHI[j]);
}
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

/** E[f_0(Y)·W] of pixel 0 after one round (mean, standard error, fraction of trials with a dropped partner). */
function run(plant: boolean, n: number, seed: number): { mean: number; se: number; dropped: number } {
  const r = rng(seed);
  let s1 = 0, s2 = 0, dropped = 0;
  const X = new Float64Array(K), W = new Float64Array(K);
  for (let it = 0; it < n; it++) {
    for (let i = 0; i < K; i++) {           // initial RIS (4 uniform candidates, target f_i)
      let wSum = 0, sel = -1;
      for (let m = 0; m < 4; m++) { const x = r(); const w = f(i, x); wSum += w; if (r() * wSum < w) sel = x; }
      X[i] = sel; W[i] = wSum / (4 * f(i, sel));
    }
    const c = 0, pc = f(c, X[c]), cc = 1;
    // S_c: every partner (all accepted); G_j = f_c(T_{j→c}(X_j))·|J| (|J| = 1), H_j = f_j(T_{c→j}(X_c))
    const S: number[] = [];
    for (let j = 1; j < K; j++) {
      const y = T(j, c, X[j]);
      const valid = y !== undefined && f(c, y) > 0;
      if (plant && !valid) continue;         // U8 plant 9: failed neighbours dropped from S_c
      S.push(j);
    }
    if (S.length < K - 1) dropped++;
    let est: number;
    if (S.length === 0) est = f(c, X[c]) * W[c];   // no partner: the canonical as is (mis.wgsl k = 0 branch)
    else {
      const k = S.length, a = cc / k;
      let sumT = 1;
      for (const j of S) { const h = T(c, j, X[c]); sumT += misCanonicalTerm(a, pc, 1, h === undefined ? 0 : f(j, h)); }
      const mc = sumT / (k + 1);
      let wSum = mc * pc * W[c], selF = pc, selP = pc;
      for (const j of S) {
        const y = T(j, c, X[j]);
        if (y === undefined) { r(); continue; }
        const G = f(c, y);
        const w = misPartnerWeight(a, 1, f(j, X[j]), G, k) * G * W[j];
        wSum += w;
        if (r() * wSum < w) { selF = G; selP = G; }
      }
      est = selF * (wSum / selP);                  // f_c(Y)·W_Y, W_Y = Σw / p̂_c(Y)
    }
    s1 += est; s2 += est * est;
  }
  const mean = s1 / n;
  return { mean, se: Math.sqrt(Math.max(s2 / n - mean * mean, 0) / n), dropped: dropped / n };
}

describe('U8 plant 9 (FAILED dropped from k): predicted sign', () => {
  it('plant off is unbiased; plant on brightens (Δ > 0) on the T7 toy', () => {
    const n = 400_000;
    const off = run(false, n, 1), on = run(true, n, 2);
    const zOff = (off.mean - 1) / off.se, zOn = (on.mean - 1) / on.se;
    console.log(`[plant9] off E=${off.mean.toFixed(5)} z=${zOff.toFixed(2)}; on E=${on.mean.toFixed(5)} (${((on.mean - 1) * 100).toFixed(2)}%) z=${zOn.toFixed(2)}, dropped-partner trials ${(on.dropped * 100).toFixed(1)}%`);
    expect(Math.abs(zOff)).toBeLessThan(4.5);
    expect(zOn).toBeGreaterThan(6);
  });
});
