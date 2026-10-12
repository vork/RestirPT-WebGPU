// Cached Potato: ordinary indirect path integration, with no reservoir/reconnection state.
// The primary vertex still uses the established direct-light reservoir and spatial reuse.
// Stateless RNG dimensions, RR and MIS match the general path tree. Visibility is supplied by
// actual continuation and shadow rays; indirect paths need no extra reconnection visibility rays.
fn potato_add(F: vec3f, rrInv: f32, diffuse: bool) {
  let lum = luminance(F);
  let w = lum * rrInv;
  if (rs_pos_finite(w) && lum > 0.0) {
    let c = F * (w / lum);
    if (diffuse) { ptIndirect += c; } else { ptSpecular += c; }
  }
}

fn potato_indirect(first: SurfaceHit, firstPrim: u32, firstV: vec3f, seed: vec2u,
                   firstBeta: vec3f, firstRr: f32, diffuse: bool, lastB: u32) {
  var cur = first;
  var prim = firstPrim;
  var V = firstV;
  var beta = firstBeta;
  var rrInv = firstRr;
  for (var B = 2u; B <= lastB; B++) {
    let m = material_eval(cur, V);
    if ((m.flags & MATEVAL_HAS_NON_DELTA) != 0u) {
      let ep = nee_draw(lightsParams.cur, rs_path_hash(seed, B, SLOT_SEL), rs_path_hash(seed, B, SLOT_SEL2),
                       vec3u(rs_path_hash(seed, B, SLOT_L0), rs_path_hash(seed, B, SLOT_L1), rs_path_hash(seed, B, SLOT_L2)));
      let ls = nee_eval(cur.pos, ep);
      if (ls.valid && ls.prim != prim && any(ls.Lambda > vec3f(0.0))) {
        let qn = bsdf_query(m, V, ls.dir, LOBE_NEE);
        let F = (nee_mis_w1(ls, qn.p_marg, B) / ls.q) * (beta * qn.f_all * ls.Lambda);
        if (any(F > vec3f(0.0)) && nee_visible(cur, prim, ls)) { potato_add(F, rrInv, diffuse); }
      }
    }
    if ((rsParams.flags & RSF_RR) != 0u && B > rsParams.rrMinBounces) {
      let q = min(sqrt(max(max(beta.x, beta.y), beta.z)), 1.0);
      if (!(rs_path_u01(seed, B, SLOT_RR) < q)) { break; }
      rrInv /= q;
    }
    let bs = bsdf_sample(m, V, rs_path_bsdf_u4(seed, B));
    if (!bs.valid) { break; }
    let org = offset_ray(cur.pos, select(-cur.ng, cur.ng, dot(cur.ng, bs.L) >= 0.0));
#if RS_LAST_ANYHIT && !RS_MODE_B
    let h = bvh_trace(org, bs.L, FLT_MAX, B == lastB, prim, BVH_MISS);
    if (B == lastB && h.primId != BVH_MISS) { break; }
#else
    let h = trace_closest_ex(org, bs.L, FLT_MAX, prim, BVH_MISS);
#endif
    let hit = h.primId != BVH_MISS;
    var nxt = cur;
    var wOut = bs.L;
    if (hit) {
      nxt = vertex_from_ids(h.primId, h.u, h.v, cur.pos);
      wOut = normalize(nxt.pos - cur.pos);
    }
    let qb = bsdf_query(m, V, wOut, bs.lobe);
#if RS_MODE_B
    if (cross_enabled(bs.is_delta)) {
      for (var li = 0u; li < lightsParams.cur.nAnalytic; li++) {
        let light = light_load(lightsParams.cur, li);
        if (light.kind != LT_RECT && light.kind != LT_DISK) { continue; }
        let ch = cross_ray(light, cur.pos, wOut, select(FLT_MAX, length(nxt.pos - cur.pos), hit));
        if (!ch.hit) { continue; }
        let z = cross_point(light, ch.xy);
        let wc = normalize(z - cur.pos);
        var fac = bs.weight;
        var p2 = 0.0;
        if (!bs.is_delta) {
          let qc = bsdf_query(m, V, wc, bs.lobe);
          fac = rs_path_weight(qc, bs.weight, false);
          p2 = qc.p_marg;
        }
        let w2 = select(mis_w2(analytic_area_p1(cur.pos, li, z), p2, B), 1.0, bs.is_delta);
        potato_add(w2 * beta * fac * area_radiance(light, -wc), rrInv, diffuse);
      }
    }
#endif
    beta *= rs_path_weight(qb, bs.weight, bs.is_delta);
    if (!any(beta > vec3f(0.0))) { break; }
    if (!hit) {
      if ((envParams.flags & ENV_FLAG_PRESENT) != 0u) {
        let w2 = env_bsdf_mis_weight(bs.L, qb.p_marg, B, bs.is_delta);
        potato_add(w2 * beta * envRadiance(envUV(bs.L, envParams.cg, envParams.sg)), rrInv, diffuse);
      }
      break;
    }
    let Le = tri_emission(h.primId, h.u, h.v);
    if (any(Le > vec3f(0.0))) {
      let p1 = tri_light_p1(cur.pos, nxt.pos, nxt.ng, h.primId);
      let w2 = select(mis_w2(p1, qb.p_marg, B), 1.0, bs.is_delta);
      potato_add(w2 * beta * Le, rrInv, diffuse);
    }
    cur = nxt;
    prim = h.primId;
    V = select(-wOut, -bs.L, bs.is_delta);
  }
}
