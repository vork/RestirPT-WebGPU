// f64 reference of the temporal MIS (restir-temporal-api.md §3.6 "MIS helper functions", TD1, TD13, TD14; math.md#temporal
// [M5 addition], #confidence). CPU twin of shaders/restir/tmis.wgsl and of the selection / W_Y formulas of
// passes/restir/t-select.wgsl. Used by tests/restir/tmis.test.ts (T6(a) CPU) and restir-temporal.gpu.test.ts (T6(a)
// WGSL ≡ f64, §9.3-5 GRIS formula). OWNER T-B. Pure TS.

const posFinite = (x: number) => Number.isFinite(x) && x > 0;

/** π_s/(c_c π_c + c_p π_p)·wSum/π_c; 0 if a denominator is not finite-positive (tmis_contrib_W). */
export function tmisContribW(piSel: number, piC: number, piP: number, cC: number, cP: number, wSum: number): number {
  const den = cC * piC + cP * piP;
  if (!posFinite(den) || !posFinite(piC)) return 0;
  return (piSel / den) * (wSum / piC);
}
/** Talbot m_c = c_c p̂_c/(c_c p̂_c + c_p π_p(X_c)); 1 if the denominator is 0 (tmis_talbot_mc). */
export function tmisTalbotMc(cC: number, phatC: number, cP: number, piP: number): number {
  const a = cC * phatC, den = a + cP * piP;
  return posFinite(den) ? a / den : 1;
}
/** Talbot m_p = c_p π_p(Y_p)/(c_c p̂(Y_p) + c_p π_p(Y_p)); 0 if the denominator is 0 (tmis_talbot_mp). */
export function tmisTalbotMp(cC: number, phatY: number, cP: number, piP: number): number {
  const b = cP * piP, den = cC * phatY + b;
  return posFinite(den) ? b / den : 0;
}

/** Contribution-MIS contribution weights ĉ_c(y), ĉ_p(y) at a point y of the canonical domain (GRIS Eq. 17 with
 *  c_c π_c / c_p π_p): π_c = p̂_t(y), π_p = π_p(y) (0 outside the image of T). */
export function contribWeights(cC: number, cP: number, piC: number, piP: number): { cc: number; cp: number } {
  const den = cC * piC + cP * piP;
  if (!(den > 0)) return { cc: 0, cp: 0 };
  return { cc: (cC * piC) / den, cp: (cP * piP) / den };
}

/** Inputs of one temporal selection at pixel q (validation layout values). */
export interface TSelectIn {
  /** Canonical: p̂_c(X_c) = lum F_c, W_c. */
  pc: number; Wc: number;
  /** Temporal: c_prev (uncapped), cap; forward shift valid?, lum F_t(Y_p), J_p, W_p, lum F_p^st. */
  cPrev: number; cap: number; fwdOk: boolean; lumFY: number; Jp: number; Wp: number; lumFst: number;
  /** π_p(X_c) (T4; used only when s = c). */
  piPXc: number;
  /** Selection random numbers (counters 0 and 1 of RS_PASS_TEMPORAL). */
  u0: number; u1: number;
}
export interface TSelectOut { sel: 'c' | 'p' | 'empty'; W: number; c: number; wc: number; wp: number }

/** Streaming-RIS update (reservoir.wgsl ris_update): true if w replaces the selection. */
function risUpdate(s: { wSum: number }, w: number, u: number): boolean {
  if (!posFinite(w)) return false;
  s.wSum += w;
  return u * s.wSum < w;
}

/** Contribution MIS selection and W_Y (t-select.wgsl phases A + B, contribution mode). */
export function temporalSelectContrib(i: TSelectIn): TSelectOut {
  const cP = Math.min(i.cap, i.cPrev);
  const wc = i.pc * i.Wc;
  const wp = i.fwdOk ? cP * i.lumFY * i.Wp * i.Jp : 0;
  if (!(wc + wp > 0)) return { sel: 'empty', W: 0, c: 1 + cP, wc, wp };
  const s = { wSum: 0 };
  risUpdate(s, wc, i.u0);
  const selP = risUpdate(s, wp, i.u1);
  if (selP) {
    const piC = i.lumFY, piP = i.lumFst / i.Jp;
    return { sel: 'p', W: tmisContribW(piP, piC, piP, 1, cP, s.wSum), c: 1 + cP, wc, wp };
  }
  return { sel: 'c', W: tmisContribW(i.pc, i.pc, i.piPXc, 1, cP, s.wSum), c: 1 + cP, wc, wp };
}
