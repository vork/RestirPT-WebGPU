// T1 rs_t_classify (restir-temporal-api.md §3.6, §4.1): per pixel over the atlas, clears tState, picks q′
// (tpick.wgsl temporal_pixel), reads c_prev, runs the forward shift inline for k ≤ 2 sources or queues it on Q_f.
// OWNER T-B (P0 stub body by T-A: clear + classification only; the stub temporal_pixel never finds a q′).
// G2: 0 resIn = res[h] ro · 1 arena rw · 2 rsVbuf · 3 rsGeo · 4 rsVbufPrev · 5 rsGeoPrev (§4.2). RS_REPLAY = 0.
#include "restir/tshift.wgsl"
#include "restir/tpick.wgsl"
#include "debug/restir-views.wgsl"

@compute @workgroup_size(8, 8, 1)
fn rs_t_classify(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(gid.xy);
  if (!p.valid) { return; }
  if (rs_vbuf(p.px).x == 0xFFFFFFFFu) { ts_clear(p.ai, TS_BG); return; }
  let key = rs_frame_key(p.member, rs_t(), p.localIdx);
  let pk = temporal_pixel(p, key);
  rsdbg_tpick(p.px, pk.sp, pk.tap, pk.valid);
  if (!pk.valid) {
    ts_clear(p.ai, TS_DISOCC | select(TS_NO_HIST, 0u, rs_tf(TF_HIST_VALID)));
    rs_count(RSC_T_DISOCC, 1u);
    return;
  }
  // P0: unreachable (stub pick). T-B: c_p, the inline forward shift / Q_f (§3.6 T1).
  ts_clear(p.ai, TS_DISOCC);
}
