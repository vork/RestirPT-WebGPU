// CWBVH traversal (BVH_CWBVH; layout contract in src/core/bvh/cwbvh.ts; docs/decisions/m8-perf.md §3). Included by
// bvh/traverse.wgsl in place of the BVH2 bvh_trace: same API, same Hit, same triangle loop body (MT or Woop + primId
// tail, skip prims, alpha_pass, the B-shadow glass plant), same flags (stack overflow, iteration cap) and BVH_STATS.
// Node groups follow Ylitie et al. 2017 / tinybvh traverse_cwbvh (a group = (base index, hit bits << 24 | imask);
// triangle groups are consumed at once). Child boxes are tested in the tinybvh form t = q·adj + orig (four children
// per vector op) with a slack that bounds the f32 rounding of that form, so a child whose exact (real-arithmetic)
// quantized box the ray overlaps is never culled; the encoder makes the quantized box contain the child's box. A
// triangle BVH2 would accept is therefore never culled either (hit equivalence, T12).
// bvh_nodes: array<vec4u> holds the 20-word nodes (never read through f32: NaN payloads); bvh_tris the triangle records in
// CWBVH order.

// 8ε (ε = 2^-24): the slab slack factor (see bvh_trace)
const CW_SLACK: f32 = 4.76837158203125e-7;
/// The four bytes of a word as f32 (slots 0–3 of a half).
#if CW_EXP_OR
// perf2 WP-3e: exponent-OR byte decode. A byte b in the mantissa of the f16 1024.0 (0x6400) is the half 1024 + b
// exactly; unpacking two halves and subtracting 1024 gives the bytes as f32, bit-equal to the integer conversion.
fn cw_bytes4(x: u32) -> vec4f {
  let lo = unpack2x16float((x & 0x00ff00ffu) | 0x64006400u) - vec2f(1024.0);
  let hi = unpack2x16float(((x >> 8u) & 0x00ff00ffu) | 0x64006400u) - vec2f(1024.0);
  return vec4f(lo.x, hi.x, lo.y, hi.y);
}
#else
fn cw_bytes4(x: u32) -> vec4f { return vec4f((vec4u(x) >> vec4u(0u, 8u, 16u, 24u)) & vec4u(0xffu)); }
#endif
fn cw_node(i: u32) -> vec4u { return bvh_nodes[i]; }
#if CW_TRI_BUDGET
// perf2 WP-3a: triangles tested per outer iteration; the outer bound never ends traversal before the iteration cap does
// (a node step leaves at most 24 triangle bits, so at most 24 outer iterations per node step)
const CW_TRI_K: u32 = $CW_TRI_BUDGETu;
const CW_OUTER_CAP: u32 = (BVH_ITER_CAP + 2u) * 24u;
#endif

fn bvh_trace(o: vec3f, d: vec3f, tmax: f32, any_hit: bool, skipA: u32, skipB: u32) -> Hit {
  var hit = Hit(tmax, 0.0, 0.0, BVH_MISS);
  let rd = vec3f(bvh_safe_rcp(d.x), bvh_safe_rcp(d.y), bvh_safe_rcp(d.z));
#if WATERTIGHT
  let wr = woop_setup(d);
  let nTris4 = arrayLength(&bvh_tris);
#endif
  let oct = select(0u, 4u, d.x < 0.0) | select(0u, 2u, d.y < 0.0) | select(0u, 1u, d.z < 0.0);
  let octinv4 = (7u - oct) * 0x01010101u;
  var stack: array<vec2u, 16>;
  var sp = 0u;
  var ng = vec2u(0u, 0x80000000u);   // the root: slot bit 31 of a virtual parent, imask 0 ⇒ node 0
  var tg = vec2u(0u, 0u);
  var iter = 0u;
#if CW_TRI_BUDGET
  // perf2 WP-3a: one loop; a node step (or the pop that precedes it) only once the triangle group is empty, then at most
  // CW_TRI_K triangles per iteration. Same node / triangle order and the same iteration count (node steps) as the nested
  // loops of the #else text, so every hit, any-hit answer, flag and BVH_STATS counter is unchanged.
  for (var s = 0u; s < CW_OUTER_CAP; s += 1u) {
    if (tg.y == 0u) {
      if (ng.y <= 0x00ffffffu) {
        if (sp == 0u) { break; }
        sp -= 1u;
        ng = stack[sp];
      }
#elif BVH_CONST_LOOPS
  // perf2 WP-3c: constant loop bound (no Tint loop guard); iteration iter + 1 of the old loop, same cap and flag
  for (; iter <= BVH_ITER_CAP; iter += 1u) {
#else
  loop {
#endif
#if BVH_CONST_LOOPS && !CW_TRI_BUDGET
    if (iter == BVH_ITER_CAP) { bvh_flags |= BVH_FLAG_ITERCAP; break; }
#else
    iter += 1u;
    if (iter > BVH_ITER_CAP) { bvh_flags |= BVH_FLAG_ITERCAP; break; }
#endif
#if BVH_STATS
    bvh_st_steps += 1u;
#endif
    if (ng.y > 0x00ffffffu) {
      let imask = ng.y & 0xffu;
      let cbi = firstLeadingBit(ng.y);
      ng.y &= ~(1u << cbi);
      if (ng.y > 0x00ffffffu) {
        if (sp < 16u) {
          stack[sp] = ng;
          sp += 1u;
#if BVH_STATS
          bvh_flags = max(bvh_flags, (bvh_flags & 0xffu) | (sp << 8u));
#endif
        } else {
          bvh_flags |= BVH_FLAG_OVERFLOW;
        }
      }
      let slot = (cbi - 24u) ^ (octinv4 & 0xffu);
      let rel = countOneBits(imask & ~(0xffffffffu << slot));
      let ni = (ng.x + rel) * 5u;
      let n0 = cw_node(ni);
      let n1 = cw_node(ni + 1u);
      let n2 = cw_node(ni + 2u);
      let n3 = cw_node(ni + 3u);
      let n4 = cw_node(ni + 4u);
      // Child slab distances t = q·(2^e·rd) + (p − o)·rd with a conservative slack (§3 of m8-perf.md): with
      // O = (p − o)·rd and A = 2^e·rd per axis (A exact: power-of-two scale), every computed plane distance is within
      // 4.6·ε·(|O| + 255·|A|) of the exact one, so widening each interval by S = 8ε·(|O| + 255·|A|) never culls a child box
      // the exact (real-arithmetic) slab test accepts. Near / far bytes are picked by the ray's direction signs.
      let p = bitcast<vec3f>(n0.xyz);
      let ex = bitcast<i32>(n0.w << 24u) >> 24u;
      let ey = bitcast<i32>(n0.w << 16u) >> 24u;
      let ez = bitcast<i32>(n0.w << 8u) >> 24u;
      let adj = vec3f(bitcast<f32>(u32(ex + 127) << 23u), bitcast<f32>(u32(ey + 127) << 23u), bitcast<f32>(u32(ez + 127) << 23u)) * rd;
      let orig = (p - o) * rd;
      let slack = CW_SLACK * (abs(orig) + 255.0 * abs(adj));
      let oN = orig - slack;
      let oF = orig + slack;
      var hitmask = 0u;
      for (var hf = 0u; hf < 2u; hf++) {
        let second = hf == 1u;
        let meta4 = select(n1.z, n1.w, second);
        let isInner4 = (meta4 & (meta4 << 1u)) & 0x10101010u;
        let innerMask4 = ((isInner4 << 3u) >> 7u) * 0xffu;
        let bitIndex4 = (meta4 ^ (octinv4 & innerMask4)) & 0x1f1f1f1fu;
        let childBits4 = (meta4 >> 5u) & 0x07070707u;
        let lox = select(n2.x, n2.y, second);
        let loy = select(n2.z, n2.w, second);
        let loz = select(n3.x, n3.y, second);
        let hix = select(n3.z, n3.w, second);
        let hiy = select(n4.x, n4.y, second);
        let hiz = select(n4.z, n4.w, second);
        let tnx = cw_bytes4(select(lox, hix, rd.x < 0.0)) * adj.x + oN.x;
        let tny = cw_bytes4(select(loy, hiy, rd.y < 0.0)) * adj.y + oN.y;
        let tnz = cw_bytes4(select(loz, hiz, rd.z < 0.0)) * adj.z + oN.z;
        let tfx = cw_bytes4(select(hix, lox, rd.x < 0.0)) * adj.x + oF.x;
        let tfy = cw_bytes4(select(hiy, loy, rd.y < 0.0)) * adj.y + oF.y;
        let tfz = cw_bytes4(select(hiz, loz, rd.z < 0.0)) * adj.z + oF.z;
        let cmin = max(max(tnx, tny), max(tnz, vec4f(0.0)));
        let cmax = min(min(tfx, tfy), min(tfz, vec4f(hit.t)));
        let ok = cmin <= cmax;
        let cb = (vec4u(childBits4) >> vec4u(0u, 8u, 16u, 24u)) & vec4u(0xffu);
        let bi = (vec4u(bitIndex4) >> vec4u(0u, 8u, 16u, 24u)) & vec4u(0xffu);
        let bits = select(vec4u(0u), cb << bi, ok);
        hitmask |= bits.x | bits.y | bits.z | bits.w;
#if BVH_STATS
        bvh_st_box += countOneBits(childBits4 & 0x01010101u) + countOneBits(childBits4 & 0x02020202u);   // non-empty slots
#endif
      }
      ng = vec2u(n1.x, (hitmask & 0xff000000u) | (n0.w >> 24u));
      tg = vec2u(n1.y, hitmask & 0x00ffffffu);
    } else {
      tg = ng;
      ng = vec2u(0u);
    }
#if CW_TRI_BUDGET
    }
    for (var k = 0u; k < CW_TRI_K; k += 1u) {
      if (tg.y == 0u) { break; }
#elif BVH_CONST_LOOPS
    for (var k = 0u; k < 24u; k += 1u) {   // a triangle group has at most 24 bits
      if (tg.y == 0u) { break; }
#else
    loop {
      if (tg.y == 0u) { break; }
#endif
      let ti = firstLeadingBit(tg.y);
      tg.y &= ~(1u << ti);
      let i = tg.x + ti;
#if BVH_STATS
      bvh_st_tri += 1u;
#endif
      let base = 3u * i;
#if WATERTIGHT
      let p0 = bvh_tris[base];
      let p1 = bvh_tris[base + 1u];
      let p2 = bvh_tris[base + 2u];
      let r = isect_woop(o, wr, p0.xyz, p1.xyz, p2.xyz, vec3u(bitcast<u32>(p0.w), bitcast<u32>(p1.w), bitcast<u32>(p2.w)), hit.t);
      if (!r.ok) { continue; }
      let prim = bitcast<u32>(bvh_tris[nTris4 - 1u - (i >> 2u)][i & 3u]); // primId tail, read from the end
#else
      let v0 = bvh_tris[base];
#if BVH_ALPHA_BIT
      let e1 = bvh_tris[base + 1u];   // e1.w: the MASK bit (bit 0) written by scene-gpu.ts (perf2 WP-3b)
      let r = isect_mt(o, d, v0.xyz, e1.xyz, bvh_tris[base + 2u].xyz, hit.t);
#else
      let r = isect_mt(o, d, v0.xyz, bvh_tris[base + 1u].xyz, bvh_tris[base + 2u].xyz, hit.t);
#endif
      if (!r.ok) { continue; }
      let prim = bitcast<u32>(v0.w);
#endif
      if (prim == skipA || prim == skipB || prim == BVH_MISS) { continue; }
#if BVH_ALPHA_BIT && BVH_NO_ALPHA
      // perf2 WP-3b: no MASK triangle in the scene (scene-gpu.ts): alpha_pass is constant true
#elif BVH_ALPHA_BIT && !WATERTIGHT
      if ((bitcast<u32>(e1.w) & 1u) != 0u && !alpha_pass(prim, r.u, r.v)) { continue; }
#else
      if (!alpha_pass(prim, r.u, r.v)) { continue; }
#endif
#if GLASS_PLANT == 5
      if (any_hit && glass_plant_transparent(prim)) { continue; }   // Gate-1 plant B-shadow (validation only)
#endif
      hit = Hit(r.t, r.u, r.v, prim);
      if (any_hit) { return hit; }
    }
#if !CW_TRI_BUDGET
    if (ng.y <= 0x00ffffffu) {
      if (sp == 0u) { break; }
      sp -= 1u;
      ng = stack[sp];
    }
#endif
  }
  return hit;
}
