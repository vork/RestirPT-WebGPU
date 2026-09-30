// T3 rs_t_select (restir-temporal-api.md §3.6, §4.1): phase A (RsDispatch.round = 0) selects between the canonical
// and the shifted temporal candidate (contribution MIS), finalises s = p and queues s = c on Q_i; phase B
// (RSD_PHASE_B) finalises the Q_i pixels after T4. Pixel-local, in place into res[w] (TD2, I1).
// OWNER T-B (P0 stub body by T-A: pixels without q′ keep the canonical bitwise, TD13; the stub pick never finds one).
// G2: 0 resIn = res[h] ro · 1 resOut = res[w] rw · 2 arena rw · 3 rsVbuf (§4.2). No scene group.
#include "restir/tframe.wgsl"
#include "restir/tmis.wgsl"
#include "restir/reservoir.wgsl"
#include "debug/restir-views.wgsl"

@compute @workgroup_size(8, 8, 1)
fn rs_t_select(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(gid.xy);
  if (!p.valid) { return; }
  let phase = select(0u, 1u, (rsDispatch.flags & RSD_PHASE_B) != 0u);
  if (rs_vbuf(p.px).x == 0xFFFFFFFFu) { return; }
  let flags = ts_load(p.ai, TSW_FLAGS);
  if ((flags & TS_QVALID) == 0u) {                       // TD13: no q′ ⇒ res[w][q] untouched
    if (phase == 0u) { rsdbg_temporal(p.px, p.ai, 0u); }
    return;
  }
  // P0: unreachable (stub pick). T-B: phase A / phase B of §3.6.
  rsdbg_temporal(p.px, p.ai, phase);
}
