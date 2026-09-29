// χ² tail probability for the sampler tests and the env splat debug view (HUD): ln Γ and the regularized upper
// incomplete gamma Q(a, x), so p = Q(dof/2, χ²/2).
/** ln Γ(x), Lanczos (g = 7, n = 9), |rel err| < 1e-13 for x > 0. */
export function lnGamma(x: number): number {
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Regularized upper incomplete gamma Q(a, x) (Numerical Recipes gser / gcf). */
export function gammaQ(a: number, x: number): number {
  if (x <= 0) return 1;
  const gln = lnGamma(a);
  if (x < a + 1) {
    let ap = a, sum = 1 / a, del = sum;
    for (let n = 0; n < 100000; n++) { ap += 1; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-15) break; }
    return 1 - sum * Math.exp(-x + a * Math.log(x) - gln);
  }
  const FPMIN = 1e-300;
  let b = x + 1 - a, c = 1 / FPMIN, d = 1 / b, h = d;
  for (let i = 1; i < 100000; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = b + an / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return Math.exp(-x + a * Math.log(x) - gln) * h;
}


/** χ² p-value of `chi2` at `dof` degrees of freedom. */
export const chi2PValue = (dof: number, chi2: number): number => (dof > 0 ? gammaQ(dof / 2, chi2 / 2) : 1);

/** Pearson χ² after merging adjacent cells (index order) until each merged cell expects ≥ minExp. */
export function chi2Merged(obs: ArrayLike<number>, exp: ArrayLike<number>, minExp = 50): { chi2: number; dof: number; p: number; cells: number } {
  let chi2 = 0, cells = 0, o = 0, e = 0;
  for (let k = 0; k < obs.length; k++) {
    o += obs[k]; e += exp[k];
    if (e >= minExp) { chi2 += (o - e) ** 2 / e; cells++; o = 0; e = 0; }
  }
  if (e > 0) { chi2 += (o - e) ** 2 / e; cells++; }
  return { chi2, dof: cells - 1, p: chi2PValue(cells - 1, chi2), cells };
}
