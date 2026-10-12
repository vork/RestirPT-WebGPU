// rs_initial (restir-api.md §4.1 step 2, §3.5; math.md#path-tree): the path tree(s) of each atlas pixel streamed into
// resOut[ai] with streaming RIS. Tree chunks: RsDispatch.treeBase/treeCount, RSD_FIRST_CHUNK initialises the record,
// RSD_FINAL_CHUNK finalises W; Σw persists in the reservoir between chunks (D10). Background pixels (vbuf miss) get
// the background record on the first chunk.
// G2: 0 resOut rw · 1 shiftArena rw (counters) · 2 rsVbuf · 3 rsGeo [· 4 candDump rw, RS_DUMP_CANDIDATES].
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"
#include "restir/queue.wgsl"
#include "path/pathtree.wgsl"
#include "debug/restir-views.wgsl"

@compute @workgroup_size(8, 8, 1)
fn rs_initial(@builtin(global_invocation_id) gid: vec3u) {
#if CW_WG_STACK && CW_WG_STACK_ACTIVE && BVH_CWBVH
  cw_lane = (gid.y & 7u) * 8u + (gid.x & 7u);
#endif
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
#if RS_POTATO_DIRECT
  let primary = rs_vbuf(p.px);
  if (primary.x != BVH_MISS) {
    let cur = vertex_from_ids(primary.x, bitcast<f32>(primary.y), bitcast<f32>(primary.z), rs_cam_pos());
    if (potato_nee_only(material_eval(cur, normalize(rs_cam_pos() - cur.pos)))) { return; }
  }
#endif
#if RS_POTATO_CACHE
  ptSpecularActive = true;
  ptSpecularDistance = 1.0;
  ptIndirect = vec3f(0.0);
  ptSpecular = vec3f(0.0);
  ptGuide = vec4f(0.0);
  textureStore(ptSpecularOut, p.px, vec4f(0.0));
  textureStore(ptGuideOut, p.px, vec4f(0.0));
  textureStore(ptIndirectOut, p.px, vec4f(0.0));
#endif
  let key = rs_frame_key(p.member, rs_t(), p.localIdx);
  let first = (rsDispatch.flags & RSD_FIRST_CHUNK) != 0u;
  let lastChunk = (rsDispatch.flags & RSD_FINAL_CHUNK) != 0u;
  if (rs_vbuf(p.px).x == 0xFFFFFFFFu) {
    if (first) { res_write_empty(p.ai, key, true); }
    if (lastChunk) { rsdbg_reservoir(p.px, p.ai, 1u); }
    return;
  }
  pathtree_run(p, key, rsDispatch.treeBase, rsDispatch.treeCount, first, lastChunk);
#if RS_POTATO_CACHE
  textureStore(ptGuideOut, p.px, ptGuide);
  textureStore(ptSpecularOut, p.px, vec4f(select(vec3f(0.0), clamp(ptSpecular, vec3f(0.0), vec3f(65504.0)), all_finite3(ptSpecular)), select(0.0, ptSpecularDistance, ptSpecularActive)));
  if (pt_indirect_active(p.local)) {
    let valid = all_finite3(ptIndirect);
    textureStore(ptIndirectOut, p.px, vec4f(select(vec3f(0.0), clamp(ptIndirect, vec3f(0.0), vec3f(65504.0)), valid), 1.0));
  }
#endif
}

#if RS_RIS_PREPASS && RS_RIS_NEE
// perf2 WP-2d (RS_RIS_PREPASS; interactive, trees = 1, no candidate dump / test text): the RIS-NEE selection at x₁ of
// tree treeBase as its own lean pass before rs_initial (schedule rs_primary → rs_light_tiles → rs_ris_nee → rs_initial).
// x₁, V and the MatEval are rs_initial's own (vertex_from_ids, D3 direction, material_eval); the loop always runs (never
// gated on this pass's MatEval flags: rs_initial decides with its own). Record → resOut[ai] plane RP_DIAG
// (ris-nee.wgsl ris_prepass_record); background pixels write nothing (rs_initial gives them the background record).
@compute @workgroup_size(8, 8, 1)
fn rs_ris_nee(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
  let vb = rs_vbuf(p.px);
  if (vb.x == 0xFFFFFFFFu) { return; }
  let key = rs_frame_key(p.member, rs_t(), p.localIdx);
  let s = rsDispatch.treeBase;
  let camPos = rs_cam_pos();
  let cur = vertex_from_ids(vb.x, bitcast<f32>(vb.y), bitcast<f32>(vb.z), camPos);
  let V = normalize(camPos - cur.pos);
  let m = material_eval(cur, V);
  resout_set(p.ai, RP_DIAG, ris_prepass_record(ris_nee_select(p, key, rs_tree_seed(key, s), s, cur, vb.x, m, V)));
}
#endif

#if RS_POTATO_DIRECT && RS_POTATO_CACHE
// Separate lean entry points: the compiler need not keep path-resampling state live while tracing GI.
@compute @workgroup_size(8, 8, 1)
fn rs_potato_light(@builtin(global_invocation_id) gid: vec3u) {
#if CW_WG_STACK && CW_WG_STACK_ACTIVE && BVH_CWBVH
  cw_lane = (gid.y & 7u) * 8u + (gid.x & 7u);
#endif
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));
  if (!p.valid) { return; }
  let vb = rs_vbuf(p.px);
  if (vb.x == BVH_MISS) { return; }
  let cur = vertex_from_ids(vb.x, bitcast<f32>(vb.y), bitcast<f32>(vb.z), rs_cam_pos());
  let V = normalize(rs_cam_pos() - cur.pos);
  let m = material_eval(cur, V);
  if (!potato_nee_only(m)) { return; }
  let key = rs_frame_key(p.member, rs_t(), p.localIdx);
  let seed = rs_tree_seed(key, 0u);
#if RS_RIS_NEE
#if RS_RIS_PREPASS
  let sel = ris_prepass_sel(resout_plane(p.ai, RP_DIAG), seed);
#else
  let sel = ris_nee_select(p, key, seed, 0u, cur, vb.x, m, V);
#endif
  let ep = sel.ep;
  let W = sel.W;
#else
  let ep = nee_draw(lightsParams.cur, rs_path_hash(seed, 1u, SLOT_SEL), rs_path_hash(seed, 1u, SLOT_SEL2),
    vec3u(rs_path_hash(seed, 1u, SLOT_L0), rs_path_hash(seed, 1u, SLOT_L1), rs_path_hash(seed, 1u, SLOT_L2)));
  let W = 1.0;
#endif
  res_write_empty(p.ai, key, false);
  textureStore(ptIndirectOut, p.px, vec4f(0.0));
  textureStore(ptSpecularOut, p.px, vec4f(0.0));
  textureStore(ptGuideOut, p.px, vec4f(bsdf_prepare(m,V).w_d, m.roughness));
  let ls = nee_eval(cur.pos, ep);
  if (!ls.valid || ls.prim == vb.x || !any(ls.Lambda > vec3f(0.0))) { return; }
  let q = bsdf_query(m, V, ls.dir, LOBE_NEE);
  let F = (1.0 / ls.q) * (q.f_all * ls.Lambda);
  let w = luminance(F) * W;
  if (!rs_pos_finite(w) || !nee_visible(cur, vb.x, ls)) { return; }
  let end = nee_endpoint_words(ep);
  let flags = rf_pack(2u, 2u, RS_TECH_NEE, ls.kind, ls.isDelta, LOBE_NEE, false, LOBE_NONE, false, true);
  pt_emit(p.ai, true, 0xFFFFFFFFu, W, F, seed, flags, end, 1.0, vec3f(0.0), 0.0, vec3f(0.0), w,
    end, pt_hist_set(RS_HIST_NONE, 1u, LOBE_NEE), vb.xyz, 0u, V, 0.0, vec3f(1.0), q.p_marg, 1u, 4096u, 0.0, nee_endpoint_id(ep));
}

@compute @workgroup_size(8, 8, 1)
fn rs_potato_indirect(@builtin(global_invocation_id) gid: vec3u) {
#if CW_WG_STACK && CW_WG_STACK_ACTIVE && BVH_CWBVH
  cw_lane = (gid.y & 7u) * 8u + (gid.x & 7u);
#endif
#if RS_POTATO_DIFFUSE
  let N = u32($RS_POTATO_CACHE);
  let phase = rs_t() % (N*N);
  let p = rs_pix(vec2u(gid.x, gid.y) * N + vec2u(phase % N, rsDispatch.rowBase / N * N + phase / N));
  if (p.px.y < rsDispatch.rowBase || p.px.y >= rsDispatch.rowEnd) { return; }
#else
  let p = rs_pix(vec2u(gid.x, gid.y + rsDispatch.rowBase));

#endif
  if (!p.valid) { return; }
  let vb = rs_vbuf(p.px);
  if (vb.x == BVH_MISS) { return; }
  let cur = vertex_from_ids(vb.x, bitcast<f32>(vb.y), bitcast<f32>(vb.z), rs_cam_pos());
  let V = normalize(rs_cam_pos() - cur.pos);
  let m = material_eval(cur, V);
  if (!potato_nee_only(m)) { return; }
  let ctx = bsdf_prepare(m, V);
  ptIndirect = vec3f(0.0); ptSpecular = vec3f(0.0); ptSpecularDistance = 1.0;
  let diffuseActive = pt_indirect_active(p.local);
  ptSpecularActive = true;
#if RS_POTATO_SPECULAR
  if (!pt_force_refresh() && textureLoad(ptFeedback, p.px, 0).g >= 8.0) {
    ptSpecularActive = ((p.local.x + p.local.y + rs_t()) & 1u) == 0u;
  }
#endif
  let key = rs_frame_key(p.member, rs_t(), p.localIdx);
  let seed = rs_tree_seed(key, 0u);
  let bs = bsdf_sample(m, V, rs_path_bsdf_u4(seed, 1u));
#if RS_POTATO_DIFFUSE
#if RS_POTATO_SPECULAR
  let trace = bs.valid && select(true, diffuseActive, bs.lobe == LOBE_D);
#else
  let trace = bs.valid && bs.lobe == LOBE_D && diffuseActive;
#endif
#else
  let trace = bs.valid && bs.lobe != LOBE_D && ptSpecularActive;
#endif
  if (trace) {
    let org = offset_ray(cur.pos, select(-cur.ng, cur.ng, dot(cur.ng, bs.L) >= 0.0));
    let h = trace_closest_ex(org, bs.L, FLT_MAX, vb.x, BVH_MISS);
    if (h.primId != BVH_MISS) {
      let nxt = vertex_from_ids(h.primId, h.u, h.v, cur.pos);
      let wi = normalize(nxt.pos - cur.pos);
      let q = bsdf_query(m, V, wi, bs.lobe);
      if (bs.lobe != LOBE_D) { ptSpecularDistance = min(h.t + 1.0, 65504.0); }
      potato_indirect(nxt, h.primId, select(-wi, -bs.L, bs.is_delta), seed,
        rs_path_weight(q, bs.weight, bs.is_delta), 1.0, bs.lobe == LOBE_D, rsParams.maxBounces + 1u);
    }
  }
#if RS_POTATO_DIFFUSE
  textureStore(ptIndirectOut, p.px, vec4f(select(vec3f(0.0), clamp(ptIndirect, vec3f(0.0), vec3f(65504.0)), all_finite3(ptIndirect)), select(0.0, 1.0, diffuseActive)));
#if RS_POTATO_SPECULAR
  textureStore(ptSpecularOut, p.px, vec4f(select(vec3f(0.0), clamp(ptSpecular, vec3f(0.0), vec3f(65504.0)), all_finite3(ptSpecular)), ptSpecularDistance));
#endif
#else
  textureStore(ptGuideOut, p.px, vec4f(ctx.w_d, m.roughness));
  textureStore(ptSpecularOut, p.px, vec4f(select(vec3f(0.0), clamp(ptSpecular, vec3f(0.0), vec3f(65504.0)), all_finite3(ptSpecular)), select(0.0, ptSpecularDistance, ptSpecularActive)));
#endif
}
#endif
