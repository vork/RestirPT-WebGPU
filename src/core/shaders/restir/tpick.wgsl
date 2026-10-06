// Temporal pixel q′ (restir-temporal-api.md §3.3, TD12; math.md#temporal [M5 addition] "q′ rule"). OWNER T-A.
// Back-projection of the current jittered primary hit x₁(q) (rebuilt from the V-buffer ids) with the previous camera,
// stochastic rounding with ξ from STREAM_TEMPORAL_PICK, then the centre tap and the 8-ring (fixed order rotated by the
// pick hash); first tap inside the member tile with a previous hit, dot(n^g, n^g′) ≥ 0.5 and
// |‖x₁ − o_{t−1}‖ − z′| ≤ 0.1·z′ wins. A function of the G-buffers and the pick stream only (sample-independent).
// sp is the continuous member-local position with pixel CENTRES at integers (frame_project − 0.5), so a static camera
// with jitter off gives sp = q and c₀ = ⌊sp + ξ⌋ = q (restir-temporal-api.md Changelog A-7).
// Needs the cur G-buffer (RS_VBUF_BINDING, RS_GEO_BINDING), the previous one (RS_VBUF_PREV_BINDING, RS_GEO_PREV_BINDING)
// and the scene group (vertex_from_ids).
#include "restir/tframe.wgsl"
#include "restir/rc.wgsl"

struct TPick { valid: bool, ai: u32, local: vec2u, sp: vec2f, tap: u32 }   // tap 0 centre, 1…8 ring, 9 none

fn tpick_none() -> TPick { return TPick(false, TS_QPRIME_NONE, vec2u(0u), vec2f(-1.0), 9u); }

/// Ring offset k (0…7): (1,0),(1,1),(0,1),(−1,1),(−1,0),(−1,−1),(0,−1),(1,−1).
fn tpick_ring(k: u32) -> vec2i {
  switch (k & 7u) {
    case 0u: { return vec2i(1, 0); }
    case 1u: { return vec2i(1, 1); }
    case 2u: { return vec2i(0, 1); }
    case 3u: { return vec2i(-1, 1); }
    case 4u: { return vec2i(-1, 0); }
    case 5u: { return vec2i(-1, -1); }
    case 6u: { return vec2i(0, -1); }
    default: { return vec2i(1, -1); }
  }
}

/// The pick hash h = pcg4d(runSeed ^ member·φ, t, localIdx, STREAM_TEMPORAL_PICK) (§5).
fn tpick_hash(p: RsPix) -> vec4u {
  return pcg4d(vec4u(frame.runSeed ^ (p.member * 0x9e3779b9u), rs_t(), p.localIdx, STREAM_TEMPORAL_PICK));
}

#if RS_VBUF_BINDING && RS_GEO_BINDING && RS_VBUF_PREV_BINDING && RS_GEO_PREV_BINDING
fn temporal_pixel(p: RsPix, key: vec2u) -> TPick {
  var r = tpick_none();
  if (!rs_tf(TF_HIST_VALID)) { return r; }
  let vb = rs_vbuf(p.px);
  if (vb.x == 0xFFFFFFFFu) { return r; }                 // background: never a temporal destination (PLAN §1.7)
  let x1 = vertex_from_ids(vb.x, bitcast<f32>(vb.y), bitcast<f32>(vb.z), rs_cam_pos());
  let ng = rs_geo(p.px).xyz;
  if (!(frame_view_depth(x1.pos, frame.prevCam) >= 1e-12)) { return r; }   // behind the previous camera
  let sp = frame_project(x1.pos, frame.prevCam) - vec2f(0.5);
  r.sp = sp;
  let h = tpick_hash(p);
  let xi = vec2f(u32_to_unit(h.x), u32_to_unit(h.y));
  let W = rsParams.memberSize.x;
  let H = rsParams.memberSize.y;
  // Far-off (or non-finite) positions are clamped before the integer conversion: no tap can be inside then.
  let lim = vec2f(f32(W) + 2.0, f32(H) + 2.0);
  let fc = floor(sp + xi);
  let inRange = fc.x >= -2.0 && fc.y >= -2.0 && fc.x <= lim.x && fc.y <= lim.y;
  if (!inRange) { return r; }
  let c0 = vec2i(fc);
  let origin = p.px - p.local;                           // member tile origin in the atlas
  let dist = length(x1.pos - lf_cam_pos(RS_FS_PREV));
  let rot = h.z & 7u;
  for (var tap = 0u; tap < 9u; tap++) {
    var c = c0;
    if (tap > 0u) { c = c0 + tpick_ring(tap - 1u + rot); }
    if (c.x < 0 || c.y < 0 || c.x >= i32(W) || c.y >= i32(H)) { continue; }
    let apx = origin + vec2u(c);
    let vp = rs_vbuf_prev(apx);
    if (vp.x == 0xFFFFFFFFu) { continue; }
    if (!(dot(ng, rs_geo_prev(apx).xyz) >= 0.5)) { continue; }
    let zp = bitcast<f32>(vp.w);
    if (!(abs(dist - zp) <= 0.1 * zp)) { continue; }
    r.valid = true;
    r.ai = apx.y * rsParams.atlasSize.x + apx.x;
    r.local = vec2u(c);
    r.tap = tap;
    return r;
  }
#if RS_DUAL_MV
  // M6 MD11 (dual motion vectors, interactive): every standard tap failed (disocclusion). The previous hit at the centre
  // tap c₀ (the occluder at t−1) is projected with the CURRENT camera; its motion applied to this pixel gives the dual
  // position, whose taps use the unchanged validity rule. G-buffers and the pick stream only (sample-independent).
  if (c0.x >= 0 && c0.y >= 0 && c0.x < i32(W) && c0.y < i32(H)) {
    let vo = rs_vbuf_prev(origin + vec2u(c0));
    if (vo.x != 0xFFFFFFFFu) {
      let yOcc = vertex_from_ids(vo.x, bitcast<f32>(vo.y), bitcast<f32>(vo.z), lf_cam_pos(RS_FS_PREV)).pos;
      if (frame_view_depth(yOcc, frame.cam) >= 1e-12) {
        let sOcc = frame_project(yOcc, frame.cam) - vec2f(0.5);
        let spD = vec2f(p.local) - (sOcc - vec2f(c0));
        let fd = floor(spD + xi);
        if (fd.x >= -2.0 && fd.y >= -2.0 && fd.x <= lim.x && fd.y <= lim.y) {
          let cd = vec2i(fd);
          for (var tap = 0u; tap < 9u; tap++) {
            var c = cd;
            if (tap > 0u) { c = cd + tpick_ring(tap - 1u + rot); }
            if (c.x < 0 || c.y < 0 || c.x >= i32(W) || c.y >= i32(H)) { continue; }
            let apx = origin + vec2u(c);
            let vp = rs_vbuf_prev(apx);
            if (vp.x == 0xFFFFFFFFu) { continue; }
            if (!(dot(ng, rs_geo_prev(apx).xyz) >= 0.5)) { continue; }
            let zp = bitcast<f32>(vp.w);
            if (!(abs(dist - zp) <= 0.1 * zp)) { continue; }
            r.valid = true;
            r.ai = apx.y * rsParams.atlasSize.x + apx.x;
            r.local = vec2u(c);
            r.tap = 10u + tap;
            rs_count(29u, 1u);                                 // RSC_T_DUAL (m6-types.wgsl)
            return r;
          }
        }
      }
    }
  }
#endif
  return r;
}
#endif
