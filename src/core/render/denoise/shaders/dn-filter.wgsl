// dn_variance + dn_atrous (docs/decisions/denoiser.md §6, §7): the variance estimate (temporal, or the 7×7 bilateral
// spatial estimate while the history is shorter than 4 frames) and the variance-guided à-trous iterations with the SVGF
// edge-stopping functions (depth, normal, luminance) and B3-spline weights. Iteration 0 feeds its output back as the
// colour history (SVGF); the last iteration remodulates (a′·filtered + L1) into the colour target (background: the
// input passes through).
//   dn_variance  G1: 0 dnAtrous[0] · 1 dnMom[cur] · 2 dnGeo[cur] · 3 dnAtrous[1] (w)
//   dn_atrous    G1: 0 dnAtrous[src] · 1 dnGeo[cur] · 2 DnIter · 3 dnAtrous[dst] (w) · 4 dnHist[cur] (w) · 5 dnOut (w)
//                · 6 input · 7 dnL1[cur] (accumulated L1, DN-2) · 8 dnAlb[cur] (the accumulated demodulation factor ā, Changelog DN-1)
#include "denoise/dn-common.wgsl"

/// One-sided depth gradient per axis at p (the smaller difference, so a silhouette does not inflate it).
fn dn_zgrad(geo: texture_2d<u32>, p: vec2i, z: f32) -> vec2f {
  var g = vec2f(0.0);
  for (var a = 0; a < 2; a++) {
    var best = 1e30;
    for (var s = -1; s <= 1; s += 2) {
      var c = p;
      c[a] += s;
      if (!dn_in_image(c)) { continue; }
      let zq = dn_guide_dist(textureLoad(geo, c, 0).xy);
      if (zq > 0.0) { best = min(best, abs(zq - z)); }
    }
    g[a] = select(0.0, best, best < 1e29);
  }
  return g;
}

/// SVGF depth and normal edge stops for a tap at offset d (pixels) from the centre.
fn dn_w_geo(zc: f32, zg: vec2f, nc: vec3f, zq: f32, nq: vec3f, d: vec2f) -> f32 {
  let phi = dn.sigmaZ * (abs(zg.x * d.x) + abs(zg.y * d.y)) + 1e-3 * zc;
  let wz = exp(-abs(zc - zq) / phi);
  let wn = pow(max(dot(nc, nq), 0.0), dn.sigmaN);
  return wz * wn;
}

#if DN_VARIANCE
@group(1) @binding(0) var atrousIn: texture_2d<f32>;
@group(1) @binding(1) var momCur: texture_2d<f32>;
@group(1) @binding(2) var geoCur: texture_2d<u32>;
@group(1) @binding(3) var atrousOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn dn_variance(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (any(gid.xy >= dn.size)) { return; }
  let c = textureLoad(atrousIn, p, 0);
  let gc = textureLoad(geoCur, p, 0).xy;
  let zc = dn_guide_dist(gc);
  if (!(zc > 0.0)) { textureStore(atrousOut, p, c); return; }
  let n = textureLoad(momCur, p, 0).z;
  var v = c.a;
  if (n < 4.0) {
    let nc = dn_guide_normal(gc);
    let zg = dn_zgrad(geoCur, p, zc);
    var s = vec3f(0.0);   // Σw, Σw·l, Σw·l²
    for (var dy = -3; dy <= 3; dy++) {
      for (var dx = -3; dx <= 3; dx++) {
        let q = p + vec2i(dx, dy);
        if (!dn_in_image(q)) { continue; }
        let gq = textureLoad(geoCur, q, 0).xy;
        let zq = dn_guide_dist(gq);
        if (!(zq > 0.0)) { continue; }
        let w = dn_w_geo(zc, zg, nc, zq, dn_guide_normal(gq), vec2f(f32(dx), f32(dy)));
        let l = luminance(textureLoad(atrousIn, q, 0).rgb);
        s += vec3f(w, w * l, w * l * l);
      }
    }
    let m = s.y / s.x;
    v = max(s.z / s.x - m * m, 0.0) * 4.0 / max(n, 1.0);   // SVGF: boost the variance of young histories
  }
  textureStore(atrousOut, p, vec4f(c.rgb, dn_fp16(v)));
  debug_write1(gid.xy, DNV_VARIANCE, v);
}
#endif

#if DN_ATROUS
struct DnIter { iter: u32, step: u32, flags: u32, pad: u32 }
const DNI_FEEDBACK: u32 = 1u;   // iteration 0: write the output as the colour history (SVGF)
const DNI_FINAL: u32 = 2u;      // last iteration: remodulate into the colour target
const DNI_COPY: u32 = 4u;       // no filtering (0 iterations)

@group(1) @binding(0) var atrousIn: texture_2d<f32>;
@group(1) @binding(1) var geoCur: texture_2d<u32>;
@group(1) @binding(2) var<uniform> it: DnIter;
@group(1) @binding(3) var atrousOut: texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var histOut: texture_storage_2d<rgba16float, write>;
@group(1) @binding(5) var colourOut: texture_storage_2d<rgba16float, write>;   // dnOut: the remodulated output (dn_resolve, DN-6)
@group(1) @binding(6) var inputTex: texture_2d<f32>;
@group(1) @binding(7) var l1Tex: texture_2d<f32>;
@group(1) @binding(8) var albTex: texture_2d<f32>;


/// 3×3 Gaussian of the variance around p (SVGF prefilter of the luminance edge stop).
fn dn_var3(p: vec2i) -> f32 {
  var s = 0.0;
  var ws = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let q = p + vec2i(dx, dy);
      if (!dn_in_image(q)) { continue; }
      let w = select(0.5, 0.25, dx != 0) * select(0.5, 0.25, dy != 0);
      s += w * textureLoad(atrousIn, q, 0).a;
      ws += w;
    }
  }
  return s / ws;
}

@compute @workgroup_size(8, 8, 1)
fn dn_atrous(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (any(gid.xy >= dn.size)) { return; }
  let c = textureLoad(atrousIn, p, 0);
  let gc = textureLoad(geoCur, p, 0).xy;
  let zc = dn_guide_dist(gc);
  let isFinal = (it.flags & DNI_FINAL) != 0u;
  if (!(zc > 0.0)) {                                    // background: pass the input through
    if (isFinal) { textureStore(colourOut, p, vec4f(textureLoad(inputTex, p, 0).rgb, 1.0)); }
    else { textureStore(atrousOut, p, c); }
    if ((it.flags & DNI_FEEDBACK) != 0u) { textureStore(histOut, p, vec4f(0.0)); }
    return;
  }
  var res = c;
  if ((it.flags & DNI_COPY) == 0u) {
    let nc = dn_guide_normal(gc);
    let zg = dn_zgrad(geoCur, p, zc);
    let lc = luminance(c.rgb);
    let ac = textureLoad(albTex, p, 0).rgb;
    let invA = select(0.0, 1.0 / dn.sigmaA, dn.sigmaA > 0.0);
    let phiL = dn.sigmaL * sqrt(max(dn_var3(p), 0.0)) + 1e-6;
    let stp = i32(it.step);
    var b3 = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
    var sc = vec3f(0.0);
    var sv = 0.0;
    var sw = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
      for (var dx = -2; dx <= 2; dx++) {
        let h = b3[dx + 2] * b3[dy + 2];
        if (dx == 0 && dy == 0) { sc += h * c.rgb; sv += h * h * c.a; sw += h; continue; }
        let d = vec2i(dx, dy) * stp;
        let q = p + d;
        if (!dn_in_image(q)) { continue; }
        let gq = textureLoad(geoCur, q, 0).xy;
        let zq = dn_guide_dist(gq);
        if (!(zq > 0.0)) { continue; }
        let cq = textureLoad(atrousIn, q, 0);
        let da = textureLoad(albTex, q, 0).rgb - ac;
        let wl = exp(-abs(lc - luminance(cq.rgb)) / phiL - (abs(da.x) + abs(da.y) + abs(da.z)) * (invA / 3.0));   // DN-5 albedo stop
        let w = h * dn_w_geo(zc, zg, nc, zq, dn_guide_normal(gq), vec2f(d)) * wl;
        sc += w * cq.rgb;
        sv += w * w * cq.a;
        sw += w;
      }
    }
    res = vec4f(sc / sw, sv / (sw * sw));
  }
  res = vec4f(dn_fp16v(res.rgb), dn_fp16(res.a));
  if ((it.flags & DNI_FEEDBACK) != 0u) { textureStore(histOut, p, vec4f(res.rgb, 0.0)); }
  debug_write3(gid.xy, DNV_LEVEL0 + it.iter, res.rgb);
  if (isFinal) {
    let L1 = textureLoad(l1Tex, p, 0).rgb;
    textureStore(colourOut, p, vec4f(res.rgb * textureLoad(albTex, p, 0).rgb + L1, 1.0));
  } else {
    textureStore(atrousOut, p, res);
  }
}
#endif
