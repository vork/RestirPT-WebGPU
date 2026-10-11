// Potato diffuse continuation: find a likely triangle in the current primary visibility buffer first.
// Eight exponentially spaced probes, then an any-hit visibility check. Offscreen/uncertain/occluded cases
// fall back to the ordinary closest-hit BVH. Never sample old screen radiance or accept a depth-only hit.
fn pt_screen_hit(o: vec3f, d: vec3f, skip: u32, p: RsPix) -> Hit {
  let miss = Hit(FLT_MAX, 0.0, 0.0, BVH_MISS);
  let base = max(frame.sceneDiag * 0.0005, 0.001);
  var last = skip;
  for (var step = 0u; step < 8u; step++) {
    let t = base * exp2(f32(step));
    let uv = frame_project(o + d * t, frame.cam);
    if (any(uv < vec2f(0.0)) || any(uv >= vec2f(rsParams.memberSize))) { break; }
    let prim = rs_vbuf(p.px - p.local + vec2u(uv)).x;
    if (prim == BVH_MISS || prim == skip || prim == last) { continue; }
    last = prim;
    let tri = sceneTris[prim];
    let a = scene_vertex_pos(tri.x);
    let b = scene_vertex_pos(tri.y);
    let c = scene_vertex_pos(tri.z);
    let h = isect_mt(o, d, a, b - a, c - a, base * 256.0);
    if (!h.ok || !alpha_pass(prim, h.u, h.v)) { continue; }
    // Camera-visible does not imply ray-visible: protect thin walls and offscreen occluders.
    if (!trace_any_ex(o, d, h.t, skip, prim)) { return Hit(h.t, h.u, h.v, prim); }
    break;
  }
  return miss;
}
