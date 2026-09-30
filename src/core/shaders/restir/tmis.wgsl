// Temporal MIS weights (restir-temporal-api.md §3.6 "MIS helper functions", TD1; math.md#temporal: contribution MIS
// with the exact E_{t−1}, Talbot-exact). OWNER T-B (real since P0; CPU twin tests/restir/tmis-ref.ts). Pure functions:
// no bindings, so the scene-less rs_t_select can include them (restir-temporal-api.md Changelog A-2).
#include "restir/types.wgsl"

/// Contribution-MIS unbiased contribution weight of the selected sample: π_s/(c_c π_c + c_p π_p) · wSum/π_c;
/// 0 if a denominator is not finite-positive.
fn tmis_contrib_W(piSel: f32, piC: f32, piP: f32, cC: f32, cP: f32, wSum: f32) -> f32 {
  let den = cC * piC + cP * piP;
  if (!rs_pos_finite(den) || !rs_pos_finite(piC)) { return 0.0; }
  return piSel / den * (wSum / piC);
}
/// Talbot m_c = c_c p̂_c/(c_c p̂_c + c_p π_p(X_c)); := 1 if the denominator is 0.
fn tmis_talbot_mc(cC: f32, phatC: f32, cP: f32, piP: f32) -> f32 {
  let a = cC * phatC;
  let den = a + cP * piP;
  if (!rs_pos_finite(den)) { return 1.0; }
  return a / den;
}
/// Talbot m_p = c_p π_p(Y_p)/(c_c p̂(Y_p) + c_p π_p(Y_p)); := 0 if the denominator is 0.
fn tmis_talbot_mp(cC: f32, phatY: f32, cP: f32, piP: f32) -> f32 {
  let b = cP * piP;
  let den = cC * phatY + b;
  if (!rs_pos_finite(den)) { return 0.0; }
  return b / den;
}
