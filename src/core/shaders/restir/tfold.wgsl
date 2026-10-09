// perf2 WP-6 (RS_TSEL_FOLD; docs/decisions/perf2-plan.md §2 WP-6 step 2): T3 phase B folded into T4 for contribution
// MIS without a check mode (RsDispatch flag RSD_TFOLD, set by stage-temporal.ts). Included by rs_t_inverse only.
// The arithmetic is passes/restir/t-select.wgsl tsel_phase_b's s = c branch, verbatim (bitwise: same operands in the
// same order; π_p(X_c) is the value T4 just stored as TSW_PIRECOMP). Each T4 item writes only its own res[w][q] (W, c;
// or the empty record), which no other T4 item reads, so finishing it here equals finishing it after every T4 item.
// Talbot, recompute and robust keep phase B (t-select.wgsl). The debug views / probe of these pixels are still
// recorded by T3 phase B (t-select.wgsl: an RSD_TFOLD phase-B dispatch only records them, emitted while a view is on).

/// = t-select.wgsl tsel_cpw.
fn tfold_cpw(cP: f32) -> f32 { return select(cP, cP + 1.0, rs_tplant(TP_CP_PLUS1)); }

/// = t-select.wgsl tsel_write_wc.
fn tfold_write_wc(q: u32, W: f32, c: f32) {
  var p0 = resout_plane(q, RP_WF);
  p0.x = bitcast<u32>(W * rsParams.wScale);
  resout_set(q, RP_WF, p0);
  var p1 = resout_plane(q, RP_SEED);
  p1.w = bitcast<u32>(c);
  resout_set(q, RP_SEED, p1);
}

/// = t-select.wgsl tsel_empty.
fn tfold_empty(q: u32, c: f32) {
  let p = rs_pix(vec2u(q % rsParams.atlasSize.x, q / rsParams.atlasSize.x));
  res_write_empty_c(p.ai, rs_frame_key(p.member, rs_t(), p.localIdx), c);
  rs_count(RSC_T_EMPTY_OUT, 1u);
}

/// Phase B of an s = c pixel q after its inverse shift (flags: T4's flags incl. TS_INV_DONE; piR: π_p(X_c) as stored).
/// Returns the final flag word.
fn tfold_phase_b(q: u32, flags0: u32, piR: f32) -> u32 {
  var flags = flags0 | TS_FINAL;
  let cP = ts_loadf(q, TSW_CP);
  let cOut = 1.0 + cP;
  // contribution MIS, s = c: π_c = p̂_t(X_c), π_p(X_c) by T4
  let piC = luminance(rp_F(resout_plane(q, RP_WF)));
  let wSum = ts_loadf(q, TSW_WC) + ts_loadf(q, TSW_WP);
  let W = tmis_contrib_W(piC, piC, piR, 1.0, tfold_cpw(cP), wSum);
  if (is_finite(W)) { tfold_write_wc(q, W, cOut); } else { rs_count(RSC_T_NONFINITE, 1u); tfold_empty(q, cOut); flags |= TS_EMPTY_OUT; }
  return flags;
}
