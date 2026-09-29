// f64 reference of the paired-spatial MIS (restir-api.md §3.8; math.md#paired-mis; gap-rc §7.3). Used by
// tests/restir/mis.test.ts (CPU) and the GPU tests T6(a) / T7 (restir-spatial.gpu.test.ts). Pure TS, no Node APIs.

/** a·pc/(a·pc + cj·lumH), := 1 if the denominator is 0. */
export function misCanonicalTerm(a: number, pc: number, cj: number, lumH: number): number {
  const den = a * pc + cj * lumH;
  return den > 0 ? (a * pc) / den : 1;
}
/** [cj·pj/(cj·pj + a·lumG)]/(k+1), := 0 if the denominator is 0. */
export function misPartnerWeight(a: number, cj: number, pj: number, lumG: number, k: number): number {
  const den = cj * pj + a * lumG;
  return (den > 0 ? (cj * pj) / den : 0) / (k + 1);
}

/**
 * Pairwise MIS weights of all k+1 techniques at one point y of the canonical domain:
 * `pcY` = p̂_c(y), `pFrom[j]` = p̂_{←j}(y) (0 where T_{c→j} is undefined), `c[j]` partner confidences, `cc` canonical.
 * Returns m_c(y) and m_j(y); their sum is 1 (partition of unity).
 */
export function misWeightsAt(cc: number, c: number[], pcY: number, pFrom: number[]): { mc: number; mj: number[] } {
  const k = c.length;
  const a = cc / k;
  let s = 1;
  for (let j = 0; j < k; j++) s += misCanonicalTerm(a, pcY, c[j], pFrom[j]);
  return { mc: s / (k + 1), mj: c.map((cj, j) => misPartnerWeight(a, cj, pFrom[j], pcY, k)) };
}

export interface PairedSlotRef {
  /** Partner confidence c_j, p̂_j(X_j) = lum F_j, W_j. */
  cj: number; pj: number; Wj: number;
  /** Partner slot G_j = F_c(Y_j)·J_{j→c} (luminance) and J_j; valid = false for FAILED / PENDING. */
  lumG: number; J: number; valid: boolean;
  /** Own slot H_j = F_j(T_{c→j}(X_c))·J_{c→j} (luminance), 0 when the own slot is FAILED / PENDING. */
  lumH: number;
}

/** Resampling weights of the canonical (index 0) and the partners (1 + j) as spatial_resample computes them. */
export function pairedWeightsRef(cc: number, pc: number, Wc: number, slots: PairedSlotRef[]): { mc: number; mj: number[]; w: number[]; wSum: number; cOut: number } {
  const k = slots.length;
  if (k === 0) return { mc: 1, mj: [], w: [pc * Wc], wSum: pc * Wc, cOut: cc };
  const a = cc / k;
  let s = 1;
  for (const sl of slots) s += misCanonicalTerm(a, pc, sl.cj, sl.lumH);
  const mc = s / (k + 1);
  const mj = slots.map((sl) => (sl.valid ? misPartnerWeight(a, sl.cj, sl.pj, sl.lumG, k) : 0));
  const w = [mc * pc * Wc, ...slots.map((sl, j) => mj[j] * (sl.valid ? sl.lumG : 0) * sl.Wj)];
  return { mc, mj, w, wSum: w.reduce((x, y) => x + y, 0), cOut: cc + slots.reduce((x, sl) => x + sl.cj, 0) };
}

/**
 * The J-explicit form of m_j(Y_j) (GRIS Eq. 38 with p̂_{←j}(Y_j) = p̂_j(X_j)/J and p̂_c(Y_j) = lum(G)/J). Equal to
 * misPartnerWeight(a, cj, pj, lumG, k) for every J > 0 (J cancels): mis.test.ts checks it.
 */
export function misPartnerWeightExplicitJ(a: number, cj: number, pj: number, lumG: number, J: number, k: number): number {
  const pFrom = pj / J, pcY = lumG / J;
  const den = cj * pFrom + a * pcY;
  return (den > 0 ? (cj * pFrom) / den : 0) / (k + 1);
}
