// rs_finalize (restir-api.md §4.1): L = L1 + estimate, estimate = rsShade when the spatial stage ran this frame
// (RsDispatch.round = number of executed spatial rounds > 0, Changelog A6) else F·W of the final reservoir (rung 3.1).
// T15 guards: a non-finite L is dropped and counted (counters[0]), negative components are counted (counters[3]).
// Outputs: rsFrame (every mode); batch: accum[localIdx] += L (sequential runs only, not RSF_ENSEMBLE);
// RS_INTERACTIVE: progressive mean (accum w = count, RSD_ACCUMULATE / RSD_ADVANCED) into the colour target.
// G2: 0 resFinal ro · 1 accum rw · 2 counters rw · 3 rsL1 · 4 rsShade · 5 rsFrame (st w) · 6 rsVbuf [· 7 colour].
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"

@group(2) @binding(1) var<storage, read_write> accum: array<vec4f>;
@group(2) @binding(2) var<storage, read_write> counters: array<atomic<u32>, 4>;   // nonFinite, bvhOverflow, bvhItercap, negative
#if RS_INTERACTIVE
@group(2) @binding(7) var colorOut: texture_storage_2d<$COLOR_FORMAT, write>;
#endif

const RS_CNT_NONFINITE: u32 = 0u;
const RS_CNT_NEGATIVE: u32 = 3u;
#if RS_SKIP_DISPLAY
const RSD_NO_DISPLAY: u32 = 128u;   // perf2 WP-7c (kernel.ts K_RSD_NO_DISPLAY): the denoiser writes the colour target
#endif

fn rs_finalize_radiance(p: RsPix) -> vec3f {
  let L1 = textureLoad(rsL1, p.px, 0).rgb;
  var est = vec3f(0.0);
  if (rsDispatch.round > 0u) {
    est = textureLoad(rsShade, p.px, 0).rgb;
  } else {
    let p0 = resin_plane(p.ai, RP_WF);
    let W = rp_W(p0);
    if (W != 0.0) { est = rp_F(p0) * W; }
  }
  return L1 + est;
}

#if !RS_INTERACTIVE
@compute @workgroup_size(8, 8, 1)
fn rs_finalize(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
  var L = rs_finalize_radiance(p);
  let ok = all_finite3(L);
  if (!ok) { atomicAdd(&counters[RS_CNT_NONFINITE], 1u); L = vec3f(0.0); }
  else if (any(L < vec3f(0.0))) { atomicAdd(&counters[RS_CNT_NEGATIVE], 1u); }
  textureStore(rsFrameOut, p.px, vec4f(L, 1.0));
  if (ok && (rsParams.flags & RSF_ENSEMBLE) == 0u) { accum[p.localIdx] += vec4f(L, 0.0); }
}
#else
@compute @workgroup_size(8, 8, 1)
fn rs_finalize_frame(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
  var c = rs_finalize_radiance(p);
  if (!all_finite3(c)) { c = vec3f(0.0); atomicAdd(&counters[RS_CNT_NONFINITE], 1u); }
  textureStore(rsFrameOut, p.px, vec4f(c, 1.0));
#if RS_SKIP_DISPLAY
  // perf2 WP-7c: the accumulation is left stale while the denoiser displays; the renderer restarts it (no
  // RSD_ACCUMULATE) on the first frame shown without the denoiser
  if ((rsDispatch.flags & RSD_NO_DISPLAY) != 0u) { return; }
#endif
  let restart = (frame.flags & (FRAME_RESET_HISTORY | FRAME_CAMERA_MOVED)) != 0u || (rsDispatch.flags & RSD_ACCUMULATE) == 0u;
  var a = select(accum[p.localIdx], vec4f(0.0), restart);
  if (restart || (rsDispatch.flags & RSD_ADVANCED) != 0u) { a += vec4f(c, 1.0); }
  accum[p.localIdx] = a;
  textureStore(colorOut, p.local, vec4f(a.rgb / max(a.w, 1.0), 1.0));
}
#endif
