// Potato-only, deliberately biased screen-space radiance cache. Sparse one-bounce estimates are demodulated,
// reconstructed across a fixed 3x3 tile neighbourhood, and accumulated with validated reprojection. A black traced
// sample (alpha 1) is data; an untraced pixel (alpha 0) is not. No multiplication by the sparse sampling interval.
// Direct illumination stays in its own ReSTIR reservoir. This pass adds the indirect cache only after resampling.
#include "denoise/dn-common.wgsl"
#include "passes/gbuffer.wgsl"
#if GBUF_48
@group(1) @binding(0) var<storage, read> gbuf: array<GBufStore>;
#else
@group(1) @binding(0) var<storage, read> gbuf: array<GBufTexel>;
#endif
@group(1) @binding(1) var raw: texture_2d<f32>;
@group(1) @binding(2) var history: texture_2d<f32>;
@group(1) @binding(3) var geoPrev: texture_2d<u32>;
@group(1) @binding(4) var direct: texture_2d<f32>;
@group(1) @binding(5) var cacheOut: texture_storage_2d<rgba16float, write>;
@group(1) @binding(6) var combined: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn dn_indirect(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.xy;
  if (any(p >= dn.size)) { return; }
  let g = gbuf[p.y * dn.size.x + p.x];
  let d = textureLoad(direct, p, 0).rgb;
  if ((g.flags & GB_HIT) == 0u) {
    textureStore(cacheOut, p, vec4f(0.0));
    textureStore(combined, p, vec4f(d, 1.0));
    return;
  }
  let N = max(dn.indirectStride, 1u);
  let phase = frame.seedIndex % (N * N);
#if DN_GI_BLUR
  let base = vec2i(p);
#else
  let base = vec2i(p / vec2u(N) * vec2u(N) + vec2u(phase % N, phase / N));
#endif
  let dist = length(g.pos - frame.cam.camToWorld[3].xyz);
  // Plane-distance gate plus a Euclidean bound: avoid crossing walls and borrowing far-away surfaces.
  let footprint = max(2.0 * dist * frame.cam.tanHalfY / f32(dn.size.y), 1e-5);
  var c = vec3f(0.0);
  var w = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let q = base + vec2i(x, y) * i32(N);
      if (!dn_in_image(q)) { continue; }
      let sample = textureLoad(raw, q, 0);
      if (sample.a == 0.0) { continue; }
      let v = gbuf[u32(q.y) * dn.size.x + u32(q.x)];
      let delta = v.pos - g.pos;
      let normal = max(dot(g.ns, v.ns), 0.0);
      if ((v.flags & GB_HIT) == 0u || normal < 0.5 || length(delta) > footprint * f32(N) * 8.0) { continue; }
      let plane = max(abs(dot(delta, g.ns)), abs(dot(delta, v.ns)));
      if (plane > footprint * 4.0) { continue; }
      let offset = vec2f(q) - vec2f(p);
      let weight = exp(-dot(offset, offset) / (2.0 * f32(N * N))) * pow(normal, 4.0);
#if DN_GI_BLUR
      c += sample.rgb * weight;
#else
      c += dn_fp16v(sample.rgb / dn_demod_factor(v.albedo)) * weight;
#endif
      w += weight;
    }
  }
  if (w > 0.0) { c /= w; }
#if DN_GI_BLUR
  let age = textureLoad(raw, p, 0).a;
#else
  var h = vec4f(0.0);
  var hw = 0.0;
  if (!dn_flag(DNF_RESET) && frame_view_depth(g.pos, frame.prevCam) > 1e-12) {
    let z = length(g.pos - frame.prevCam.camToWorld[3].xyz);
    let sp = select(vec2f(p), frame_project(g.pos, frame.prevCam) - vec2f(0.5), (frame.flags & FRAME_CAMERA_MOVED) != 0u);
    if (all(sp >= vec2f(-1.0)) && all(sp < vec2f(dn.size))) {
      let b = vec2i(floor(sp));
      let f = fract(sp);
      for (var j = 0u; j < 4u; j++) {
        let o = vec2i(i32(j & 1u), i32(j >> 1u));
        let q = b + o;
        if (!dn_in_image(q)) { continue; }
        let prev = textureLoad(geoPrev, q, 0).xy;
        let zp = dn_guide_dist(prev);
        if (!(zp > 0.0) || dot(g.ns, dn_guide_normal(prev)) < 0.5 || abs(z - zp) > max(footprint * 2.0, 0.005 * z)) { continue; }
        let v = textureLoad(history, q, 0);
        if (v.a <= 0.0) { continue; }
        let weight = select(1.0-f.x, f.x, o.x == 1) * select(1.0-f.y, f.y, o.y == 1);
        h += v * weight;
        hw += weight;
      }
    }
  }
  if (hw > 1e-4) { h /= hw; } else { h = vec4f(0.0); }
  var age = min(h.a + 1.0, 32.0);
  if (w > 0.0) {
    if (h.a > 0.0) {
      let alpha = max(1.0 / age, select(0.05, 0.2, dn_flag(DNF_CACHE_LIGHT)));
      // Biased suppression of isolated bright indirect samples. Broad lighting changes bypass the cap.
      if (!dn_flag(DNF_CACHE_LIGHT)) {
        let limit = max(luminance(h.rgb) * 4.0, 0.02);
        c *= min(1.0, limit / max(luminance(c), 1e-6));
      }
      c = mix(h.rgb, c, alpha);
    }
  } else {
    c = h.rgb;
    age = h.a; // No compatible sample or history: zero confidence, retry reconstruction next frame.
  }
#endif
  c = dn_fp16v(c);
  textureStore(cacheOut, p, vec4f(c, age));
  textureStore(combined, p, vec4f(d + c * dn_demod_factor(g.albedo), 1.0));
}
