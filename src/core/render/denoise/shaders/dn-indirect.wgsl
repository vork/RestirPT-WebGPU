// Potato-only, deliberately biased diffuse cache with an optional world-space fallback. Sparse one-bounce estimates are demodulated,
// reconstructed across a fixed 3x3 tile neighbourhood, and accumulated with validated reprojection. A black traced
// sample (alpha 1) is data; an untraced pixel (alpha 0) is not. No multiplication by the sparse sampling interval.
// Direct illumination stays in its own ReSTIR reservoir. This pass adds the indirect cache only after resampling.
#include "denoise/dn-common.wgsl"
#include "passes/gbuffer.wgsl"
#include "denoise/dn-world-common.wgsl"
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

@group(1) @binding(7) var specular: texture_2d<f32>;
@group(1) @binding(8) var diffuseGuide: texture_2d<f32>;
@group(1) @binding(9) var<storage, read> worldCache: array<GiWorldEntry>;

fn gi_diffuse_factor(p: vec2u, albedo: vec3f) -> vec3f {
  // A 1x1 guide is the synthetic/legacy test path. Production always supplies the exact Lambert weight.
  if (all(textureDimensions(diffuseGuide) == vec2u(1u))) { return dn_demod_factor(albedo); }
  return textureLoad(diffuseGuide, p, 0).rgb;
}

@compute @workgroup_size(8, 8, 1)
fn dn_indirect(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.xy;
  if (any(p >= dn.size)) { return; }
  let g = gbuf[p.y * dn.size.x + p.x];
  let d = textureLoad(direct, p, 0).rgb + textureLoad(specular, p, 0).rgb;
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
#if DN_GI_BLUR
      // Mix adjacent phases instead of feeding independent N-stride subgrids back into history.
      let q = base + vec2i(x, y) * i32(max(N - 1u, 1u));
#else
      let q = base + vec2i(x, y) * i32(N);
#endif
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
      var weight = exp(-dot(offset, offset) / (2.0 * f32(N * N))) * pow(normal, 4.0);
#if DN_GI_BLUR
      c += sample.rgb * weight;
#else
      let irradiance = dn_fp16v(sample.rgb / max(gi_diffuse_factor(vec2u(q), v.albedo), vec3f(0.02)));
      // Deliberately biased robust mean: stop rare bright paths becoming persistent spatial patches.
      // Equal-valued neighbourhoods retain their brightness; only uneven estimates are downweighted.
      weight /= 1.0 + 4.0 * luminance(irradiance);
      c += irradiance * weight;
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
      // Keep outlier suppression during moving-light frames too. The additive allowance lets
      // previously dark surfaces brighten, while the unchanged lighting alpha limits response lag.
      let limit = luminance(h.rgb) * 2.0 + 0.05;
      c *= min(1.0, limit / max(luminance(c), 1e-6));
      c = mix(h.rgb, c, alpha);
    }
  } else {
    c = h.rgb;
    age = h.a; // No compatible sample or history: zero confidence, retry reconstruction next frame.
  }
#endif
  c = dn_fp16v(c);
  // Keep full radiance in history: feeding the fade back would continually darken the cache.
  textureStore(cacheOut, p, vec4f(c, age));
  // Newly visible surfaces start with direct lighting, then reveal GI over 16 valid updates.
  // Smoothstep has zero slope at both ends; only rejected/reset history restarts the ramp.
  let reveal = smoothstep(1.0, 17.0, age);
  var indirect = c * reveal;
#if DN_GI_BLUR
  if (dn_flag(DNF_WORLD_CACHE) && reveal < 1.0) {
    let pos = g.pos + frame.origin;
    let entry = worldCache[gi_world_index(pos, g.ns)];
    if (gi_world_valid(entry, pos, g.ns)) {
      // Only the displayed fallback uses persistent lighting. Fresh local history remains independent.
      indirect = mix(entry.irradiance, c, reveal);
    }
  }
#endif
  textureStore(combined, p, vec4f(d + indirect * gi_diffuse_factor(p, g.albedo), 1.0));
}
