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
fn cw_bytes4(x: u32) -> vec4f { return vec4f((vec4u(x) >> vec4u(0u, 8u, 16u, 24u)) & vec4u(0xffu)); }
fn cw_node(i: u32) -> vec4u { return bvh_nodes[i]; }

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
  loop {
    iter += 1u;
    if (iter > BVH_ITER_CAP) { bvh_flags |= BVH_FLAG_ITERCAP; break; }
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
    loop {
      if (tg.y == 0u) { break; }
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
      let r = isect_mt(o, d, v0.xyz, bvh_tris[base + 1u].xyz, bvh_tris[base + 2u].xyz, hit.t);
      if (!r.ok) { continue; }
      let prim = bitcast<u32>(v0.w);
#endif
      if (prim == skipA || prim == skipB || prim == BVH_MISS) { continue; }
      if (!alpha_pass(prim, r.u, r.v)) { continue; }
#if GLASS_PLANT == 5
      if (any_hit && glass_plant_transparent(prim)) { continue; }   // Gate-1 plant B-shadow (validation only)
#endif
      hit = Hit(r.t, r.u, r.v, prim);
      if (any_hit) { return hit; }
    }
    if (ng.y <= 0x00ffffffu) {
      if (sp == 0u) { break; }
      sp -= 1u;
      ng = stack[sp];
    }
  }
  return hit;
}
