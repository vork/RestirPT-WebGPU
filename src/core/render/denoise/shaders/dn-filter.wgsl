// dn_variance + dn_atrous (docs/decisions/denoiser.md §6, §7): the variance estimate (temporal, or the 7×7 bilateral
// spatial estimate while the history is shorter than 4 frames) and the variance-guided à-trous iterations with the SVGF
// edge-stopping functions (depth, normal, luminance) and B3-spline weights. Iteration 0 feeds its output back as the
// colour history (SVGF); the last iteration remodulates (a′·filtered + L1) into the colour target (background: the
// input passes through).
//   dn_variance  G1: 0 dnAtrous[0] · 1 dnMom[cur] · 2 dnGeo[cur] · 3 dnAtrous[1] (w) · 4 dnLumG (r32float, w: DN-13 guide)
//                · 5 dnAlb[cur] · 6 dnL1[cur] · 7 dnTaa[prev] · 8 dnTap (rgba32uint, w: per-tap data of the à-trous, DN-14)
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

const DN_LOG2E: f32 = 1.4426950408889634;
const DN_LW_GEO_MIN: f32 = -19.931568569324174;   // log2(10⁻⁶), DN-15: à-trous taps below this geometric weight are skipped

/// SVGF depth and normal edge stops for a tap at offset d (pixels) from the centre.
fn dn_w_geo(zc: f32, zg: vec2f, nc: vec3f, zq: f32, nq: vec3f, d: vec2f) -> f32 {
  let phi = dn.sigmaZ * (abs(zg.x * d.x) + abs(zg.y * d.y)) + 1e-3 * zc;
  let wz = exp(-abs(zc - zq) / phi);
  let wn = pow(max(dot(nc, nq), 0.0), dn.sigmaN);
  return wz * wn;
}

/// DN-15: log2 of dn_w_geo, so an à-trous tap needs one exp2 for all its edge stops (pow = exp2·log2 anyway).
fn dn_lw_geo(zc: f32, zg: vec2f, nc: vec3f, zq: f32, nq: vec3f, d: vec2f) -> f32 {
  let phi = dn.sigmaZ * (abs(zg.x * d.x) + abs(zg.y * d.y)) + 1e-3 * zc;
  return -abs(zc - zq) / phi * DN_LOG2E + dn.sigmaN * log2(max(dot(nc, nq), 1e-30));
}

/// B3-spline weight of tap offset k ∈ [−2, 2].
fn dn_b3(k: i32) -> f32 { return select(select(0.0625, 0.25, abs(k) == 1), 0.375, k == 0); }

/// DN-14 dnTap accessors: ā and the DN-9 guide luminance (−1: none).
fn dn_tap_alb(t: vec4u) -> vec3f { return vec3f(unpack2x16float(t.z), unpack2x16float(t.w).x); }
fn dn_tap_guide(t: vec4u) -> f32 { return unpack2x16float(t.w).y; }

#if DN_VARIANCE
@group(1) @binding(0) var atrousIn: texture_2d<f32>;
@group(1) @binding(1) var momCur: texture_2d<f32>;
@group(1) @binding(2) var geoCur: texture_2d<u32>;
@group(1) @binding(3) var atrousOut: texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var lumGOut: texture_storage_2d<r32float, write>;
@group(1) @binding(5) var albCur: texture_2d<f32>;
@group(1) @binding(6) var l1Cur: texture_2d<f32>;
@group(1) @binding(7) var taaPrev: texture_2d<f32>;
@group(1) @binding(8) var tapOut: texture_storage_2d<rgba32uint, write>;

/// DN-14: everything an à-trous tap reads besides its colour, written once per frame as one rgba32uint texel:
/// x, y = the dnGeo guide (f32 distance, oct normal); z, w = binary16 (ā.r, ā.g), (ā.b, g) with g = the DN-9 luminance
/// guide lum((T̄ − L̄1)/ā) of the converged previous output, or −1 where it does not apply (guide off, camera moved,
/// n_t < 8). One fetch instead of two (texture-fetch bound: −DN-15).
fn dn_tap(p: vec2i, geo: vec2u) -> vec4u {
  let a = textureLoad(albCur, p, 0).rgb;
  var g = -1.0;
  let t = textureLoad(taaPrev, p, 0);
  if (dn_flag(DNF_GUIDE) && (frame.flags & FRAME_CAMERA_MOVED) == 0u && t.a >= 8.0) {
    g = dn_fp16(luminance(max(t.rgb - textureLoad(l1Cur, p, 0).rgb, vec3f(0.0)) / max(a, vec3f(1e-3))));
  }
  let a16 = dn_fp16v(a);
  return vec4u(geo, pack2x16float(a16.xy), pack2x16float(vec2f(a16.z, g)));
}

/// DN-13: the luminance guide of the à-trous stop: a binomial prefilter (radius dn.lumPre) of the integrated colour's
/// luminance over geometrically compatible neighbours (weights independent of the noisy values).
fn dn_lum_guide(p: vec2i, zc: f32, zg: vec2f, nc: vec3f) -> f32 {
  let R = i32(dn.lumPre);
  if (R == 0) { return luminance(textureLoad(atrousIn, p, 0).rgb); }
  var s = 0.0;
  var ws = 0.0;
  for (var dy = -R; dy <= R; dy++) {
    for (var dx = -R; dx <= R; dx++) {
      let q = p + vec2i(dx, dy);
      if (!dn_in_image(q)) { continue; }
      let gq = textureLoad(geoCur, q, 0).xy;
      let zq = dn_guide_dist(gq);
      if (!(zq > 0.0)) { continue; }
      let hb = select(select(0.25, 0.5, abs(dx) == 0), select(select(0.0625, 0.25, abs(dx) == 1), 0.375, dx == 0), R == 2)
             * select(select(0.25, 0.5, abs(dy) == 0), select(select(0.0625, 0.25, abs(dy) == 1), 0.375, dy == 0), R == 2);
      let w = hb * dn_w_geo(zc, zg, nc, zq, dn_guide_normal(gq), vec2f(f32(dx), f32(dy)));
      s += w * luminance(textureLoad(atrousIn, q, 0).rgb);
      ws += w;
    }
  }
  return s / max(ws, 1e-12);
}

@compute @workgroup_size(8, 8, 1)
fn dn_variance(@builtin(global_invocation_id) gid: vec3u) {
  let p = vec2i(gid.xy);
  if (any(gid.xy >= dn.size)) { return; }
  let c = textureLoad(atrousIn, p, 0);
  let gc = textureLoad(geoCur, p, 0).xy;
  let zc = dn_guide_dist(gc);
  if (!(zc > 0.0)) { textureStore(atrousOut, p, c); textureStore(lumGOut, p, vec4f(0.0)); textureStore(tapOut, p, vec4u(0u)); return; }
  textureStore(tapOut, p, dn_tap(p, gc));
  let n = textureLoad(momCur, p, 0).z;
  textureStore(lumGOut, p, vec4f(dn_lum_guide(p, zc, dn_zgrad(geoCur, p, zc), dn_guide_normal(gc)), 0.0, 0.0, 0.0));
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
@group(1) @binding(1) var tapTex: texture_2d<u32>;   // DN-14: (dnGeo guide, ā, DN-9 guide luminance or −1), from dn_variance
@group(1) @binding(2) var<uniform> it: DnIter;
@group(1) @binding(3) var atrousOut: texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var histOut: texture_storage_2d<rgba16float, write>;
@group(1) @binding(5) var colourOut: texture_storage_2d<rgba16float, write>;   // dnOut: the remodulated output (dn_resolve, DN-6)
@group(1) @binding(6) var inputTex: texture_2d<f32>;
@group(1) @binding(7) var l1Tex: texture_2d<f32>;
@group(1) @binding(8) var albTex: texture_2d<f32>;
@group(1) @binding(9) var momCur: texture_2d<f32>;    // DN-12: the colour-history length n
@group(1) @binding(10) var lumG: texture_2d<f32>;     // DN-13: the prefiltered luminance guide


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
  let tc = textureLoad(tapTex, p, 0);
  let gc = tc.xy;
  let zc = dn_guide_dist(gc);
  let isFinal = (it.flags & DNI_FINAL) != 0u;
  if (!(zc > 0.0)) {                                    // background: pass the input through
    if (isFinal) { textureStore(colourOut, p, vec4f(dn_fp16v(textureLoad(inputTex, p, 0).rgb), 1.0)); }
    else { textureStore(atrousOut, p, c); }
    if ((it.flags & DNI_FEEDBACK) != 0u) { textureStore(histOut, p, vec4f(0.0)); }
    return;
  }
  var res = c;
  if ((it.flags & DNI_COPY) == 0u) {
    let nc = dn_guide_normal(gc);
    let zg = dn_zgrad(tapTex, p, zc);
    let ac = dn_tap_alb(tc);
    // DN-9: with a converged output (static camera and lighting, n_t ≥ 8 at both pixels) the luminance stop compares
    // the demodulated previous output instead of this frame's noisy values: weights that depend on the noise being
    // filtered pull the mean toward the mode of right-skewed Monte-Carlo noise (the residual darkening of DN-7).
    let gl = dn_tap_guide(tc);
    let useGuide = gl >= 0.0;
    let pre = dn.lumPre > 0u && it.iter == 0u;   // DN-13: later levels filter already-smoothed values
    var lc = luminance(c.rgb);
    if (useGuide) { lc = gl; } else if (pre) { lc = textureLoad(lumG, p, 0).x; }
    let invA = select(0.0, 1.0 / dn.sigmaA, dn.sigmaA > 0.0);
    // DN-12: no luminance stop on young histories (resets, disocclusions): with few, right-skewed samples, weights that
    // depend on the noisy values pull the mean toward the mode (−25 % on a reset after an added spot light)
    let lumStop = textureLoad(momCur, p, 0).z >= dn.lumMinN;
    let phiL = select(1e30, dn.sigmaL * sqrt(max(dn_var3(p), 0.0)) + 1e-6, lumStop);
    let invPhiL = DN_LOG2E / phiL;
    let kA = invA / 3.0 * DN_LOG2E;
    let stp = i32(it.step);
    var sc = vec3f(0.0);
    var sv = 0.0;
    var sw = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
      for (var dx = -2; dx <= 2; dx++) {
        let h = dn_b3(dx) * dn_b3(dy);   // no dynamically indexed array: it would live in private memory
        if (dx == 0 && dy == 0) { sc += h * c.rgb; sv += h * h * c.a; sw += h; continue; }
        let d = vec2i(dx, dy) * stp;
        let q = p + d;
        if (!dn_in_image(q)) { continue; }
        let tq = textureLoad(tapTex, q, 0);
        let gq = tq.xy;
        let zq = dn_guide_dist(gq);
        if (!(zq > 0.0)) { continue; }
        let lg = dn_lw_geo(zc, zg, nc, zq, dn_guide_normal(gq), vec2f(d));
        if (lg < DN_LW_GEO_MIN) { continue; }   // DN-15: a negligible geometric weight: skip the colour and tap reads
        let cq = textureLoad(atrousIn, q, 0);
        let da = dn_tap_alb(tq) - ac;
        let glq = dn_tap_guide(tq);
        var lq: f32;
        if (useGuide && glq >= 0.0) { lq = glq; } else if (pre) { lq = textureLoad(lumG, q, 0).x; } else { lq = luminance(cq.rgb); }
        let w = h * exp2(lg - abs(lc - lq) * invPhiL - (abs(da.x) + abs(da.y) + abs(da.z)) * kA);   // DN-5 albedo stop
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
    textureStore(colourOut, p, vec4f(dn_fp16v(res.rgb * textureLoad(albTex, p, 0).rgb + L1), 1.0));
  } else {
    textureStore(atrousOut, p, res);
  }
}

// M8 (m8-perf.md §7, P-6): à-trous levels of step s = DN_TILE_STEP ∈ {1, 2, 4} with a workgroup-memory tile (the
// denoiser uses it for s = 1: −35 % on that level; s = 2 measured neutral, s = 4 slower: a 20 KB tile costs occupancy). Every tap
// p + s·d (|d| ≤ 2) of the 8 × 8 group lies in one contiguous (8 + 4s)² tile, loaded once (144 / 256 / 576 tap and colour
// texels instead of 64 × 24 texture reads); the per-tap arithmetic, its order and every skip rule are dn_atrous' (keep
// the two in sync); dn_zgrad / dn_var3 (±1 pixel) keep their texture reads. A strided-lattice variant (one residue class per group) lost the cache locality of
// neighbouring threads and was slower (m8-perf.md §7).
#if DN_TILE_STEP
const DN_AT_S: u32 = u32($DN_TILE_STEP);
const DN_AT_T: u32 = 8u + 4u * DN_AT_S;   // tile side
var<workgroup> atTap: array<vec4u, DN_AT_T * DN_AT_T>;
var<workgroup> atCol: array<vec4f, DN_AT_T * DN_AT_T>;
var<workgroup> atLum: array<f32, DN_AT_T * DN_AT_T>;

@compute @workgroup_size(8, 8, 1)
fn dn_atrous_tile(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) li: u32) {
  let stp = DN_AT_S;
  let o = vec2i(wid.xy * 8u);                          // first pixel of the group
  let pre = dn.lumPre > 0u && it.iter == 0u;          // DN-13: later levels filter already-smoothed values
  for (var k = li; k < DN_AT_T * DN_AT_T; k += 64u) {
    let q = o + vec2i(i32(k % DN_AT_T), i32(k / DN_AT_T)) - vec2i(2 * i32(DN_AT_S));
    if (dn_in_image(q)) {
      atTap[k] = textureLoad(tapTex, q, 0);
      atCol[k] = textureLoad(atrousIn, q, 0);
      if (pre) { atLum[k] = textureLoad(lumG, q, 0).x; }
    }
  }
  workgroupBarrier();
  let p = o + vec2i(lid.xy);
  if (any(vec2u(p) >= dn.size)) { return; }
  let gid = vec2u(p);
  let ci = (lid.y + 2u * DN_AT_S) * DN_AT_T + lid.x + 2u * DN_AT_S;
  let c = atCol[ci];
  let tc = atTap[ci];
  let gc = tc.xy;
  let zc = dn_guide_dist(gc);
  let isFinal = (it.flags & DNI_FINAL) != 0u;
  if (!(zc > 0.0)) {                                    // background: pass the input through
    if (isFinal) { textureStore(colourOut, p, vec4f(dn_fp16v(textureLoad(inputTex, p, 0).rgb), 1.0)); }
    else { textureStore(atrousOut, p, c); }
    if ((it.flags & DNI_FEEDBACK) != 0u) { textureStore(histOut, p, vec4f(0.0)); }
    return;
  }
  var res = c;
  if ((it.flags & DNI_COPY) == 0u) {
    let nc = dn_guide_normal(gc);
    let zg = dn_zgrad(tapTex, p, zc);
    let ac = dn_tap_alb(tc);
    // DN-9: with a converged output (static camera and lighting, n_t ≥ 8 at both pixels) the luminance stop compares
    // the demodulated previous output instead of this frame's noisy values: weights that depend on the noise being
    // filtered pull the mean toward the mode of right-skewed Monte-Carlo noise (the residual darkening of DN-7).
    let gl = dn_tap_guide(tc);
    let useGuide = gl >= 0.0;
    var lc = luminance(c.rgb);
    if (useGuide) { lc = gl; } else if (pre) { lc = atLum[ci]; }
    let invA = select(0.0, 1.0 / dn.sigmaA, dn.sigmaA > 0.0);
    // DN-12: no luminance stop on young histories (resets, disocclusions): with few, right-skewed samples, weights that
    // depend on the noisy values pull the mean toward the mode (−25 % on a reset after an added spot light)
    let lumStop = textureLoad(momCur, p, 0).z >= dn.lumMinN;
    let phiL = select(1e30, dn.sigmaL * sqrt(max(dn_var3(p), 0.0)) + 1e-6, lumStop);
    let invPhiL = DN_LOG2E / phiL;
    let kA = invA / 3.0 * DN_LOG2E;
    let st = i32(stp);
    var sc = vec3f(0.0);
    var sv = 0.0;
    var sw = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
      for (var dx = -2; dx <= 2; dx++) {
        let h = dn_b3(dx) * dn_b3(dy);   // no dynamically indexed array: it would live in private memory
        if (dx == 0 && dy == 0) { sc += h * c.rgb; sv += h * h * c.a; sw += h; continue; }
        let d = vec2i(dx, dy) * st;
        let q = p + d;
        if (!dn_in_image(q)) { continue; }
        let qi = u32(i32(ci) + (dy * i32(DN_AT_T) + dx) * st);
        let tq = atTap[qi];
        let gq = tq.xy;
        let zq = dn_guide_dist(gq);
        if (!(zq > 0.0)) { continue; }
        let lg = dn_lw_geo(zc, zg, nc, zq, dn_guide_normal(gq), vec2f(d));
        if (lg < DN_LW_GEO_MIN) { continue; }   // DN-15: a negligible geometric weight: skip the colour
        let cq = atCol[qi];
        let da = dn_tap_alb(tq) - ac;
        let glq = dn_tap_guide(tq);
        var lq: f32;
        if (useGuide && glq >= 0.0) { lq = glq; } else if (pre) { lq = atLum[qi]; } else { lq = luminance(cq.rgb); }
        let w = h * exp2(lg - abs(lc - lq) * invPhiL - (abs(da.x) + abs(da.y) + abs(da.z)) * kA);   // DN-5 albedo stop
        sc += w * cq.rgb;
        sv += w * w * cq.a;
        sw += w;
      }
    }
    res = vec4f(sc / sw, sv / (sw * sw));
  }
  res = vec4f(dn_fp16v(res.rgb), dn_fp16(res.a));
  if ((it.flags & DNI_FEEDBACK) != 0u) { textureStore(histOut, p, vec4f(res.rgb, 0.0)); }
  debug_write3(gid, DNV_LEVEL0 + it.iter, res.rgb);
  if (isFinal) {
    let L1 = textureLoad(l1Tex, p, 0).rgb;
    textureStore(colourOut, p, vec4f(dn_fp16v(res.rgb * textureLoad(albTex, p, 0).rgb + L1), 1.0));
  } else {
    textureStore(atrousOut, p, res);
  }
}
#endif
#endif
