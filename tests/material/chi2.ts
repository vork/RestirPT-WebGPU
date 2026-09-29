// Pearson χ² goodness-of-fit for sampler tests (Mitsuba chi2.py / pbrt bsdfs_test methodology; validation-harness
// T8): observed histogram vs expected counts from the integrated pdf, low-expectation cells pooled, plus one
// "outside" cell for the probability mass the histogram does not cover (rejections, other lobes).

import { gammaQ } from '../../src/core/render/chi2.ts';

export { gammaQ, lnGamma } from '../../src/core/render/chi2.ts';

export interface Chi2Result { chi2: number; dof: number; pValue: number; cells: number; pooledExpected: number; outsideObs: number; outsideExp: number }

/**
 * obs[i] / exp[i] per histogram bin; `total` = number of sampler invocations (the multinomial size). The mass not
 * covered by the bins forms the "outside" cell (total − Σobs vs total − Σexp). Cells with expected < minExp are pooled.
 */
export function chi2Test(obs: ArrayLike<number>, exp: ArrayLike<number>, total: number, minExp = 5): Chi2Result {
  let sumObs = 0, sumExp = 0;
  const cells: [number, number][] = [];
  let pObs = 0, pExp = 0;
  for (let i = 0; i < obs.length; i++) {
    const o = obs[i], e = Math.max(exp[i], 0);
    sumObs += o; sumExp += e;
    if (e < minExp) { pObs += o; pExp += e; } else cells.push([o, e]);
  }
  const outsideObs = total - sumObs, outsideExp = Math.max(total - sumExp, 0);
  if (outsideExp < minExp) { pObs += outsideObs; pExp += outsideExp; } else cells.push([outsideObs, outsideExp]);
  const pooledExpected = pExp;
  if (pExp > 0 || pObs > 0) {
    if (pExp < minExp && cells.length) {
      let k = 0;
      for (let i = 1; i < cells.length; i++) if (cells[i][1] < cells[k][1]) k = i;
      cells[k][0] += pObs; cells[k][1] += pExp;
    } else cells.push([pObs, Math.max(pExp, 1e-300)]);
  }
  let chi2 = 0;
  for (const [o, e] of cells) chi2 += (o - e) ** 2 / e;
  const dof = cells.length - 1;
  const pValue = dof > 0 ? gammaQ(dof / 2, chi2 / 2) : 1;
  return { chi2, dof, pValue, cells: cells.length, pooledExpected, outsideObs, outsideExp };
}

/** Šidák-corrected per-test significance for `tests` independent tests at family level alpha. */
export const sidak = (alpha: number, tests: number) => 1 - (1 - alpha) ** (1 / tests);

/** Bin edges of [a, b] (n bins) that are equal-probability under λ·Cauchy(x0, w)|[a,b] + (1−λ)·Uniform[a,b]. */
export function adaptiveEdges(n: number, a: number, b: number, x0: number, w: number, lambda: number): Float32Array {
  const A = Math.atan((a - x0) / w), B = Math.atan((b - x0) / w);
  const G = (x: number) => lambda * (Math.atan((x - x0) / w) - A) / (B - A) + (1 - lambda) * (x - a) / (b - a);
  const out = new Float32Array(n + 1);
  out[0] = a; out[n] = b;
  for (let i = 1; i < n; i++) {
    const t = i / n;
    let lo = a, hi = b;
    for (let k = 0; k < 200 && hi - lo > 1e-300; k++) { const mid = 0.5 * (lo + hi); if (G(mid) < t) lo = mid; else hi = mid; }
    out[i] = Math.fround(0.5 * (lo + hi));
  }
  for (let i = 1; i <= n; i++) if (!(out[i] > out[i - 1])) throw new Error(`adaptiveEdges: non-increasing f32 edges at ${i} (w = ${w})`);
  return out;
}
