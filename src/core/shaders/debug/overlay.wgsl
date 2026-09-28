// Raster overlay: world-space line lists (gizmos, probe paths) drawn onto the canvas after present.
// Depth test is manual: the fragment's linear depth is compared against the internal-resolution linear depth
// texture from the primary pass (no depth attachment, so it works at canvas resolution). Lines flagged
// ON_TOP skip the test; hidden fragments are drawn with `hiddenAlpha` (0 = discarded).
// Vertex positions are in the render-internal recentred frame (overlay.ts subtracts the origin in f64).

struct OverlayParams {
  viewProj: mat4x4f,     // internal frame -> clip (reversed infinite perspective, internal-res aspect)
  worldToCam: mat4x4f,
  srcSize: vec2u,        // depth texture size (internal resolution)
  dstSize: vec2u,        // canvas size
  depthKind: u32,        // 0: view-space depth (-z_cam), 1: ray distance |p - cam|
  hiddenAlpha: f32,
  depthEps: f32,         // relative tolerance
  hasDepth: u32,         // 0: no depth texture bound yet -> everything visible
}

const OVERLAY_ON_TOP: u32 = 1u;

@group(0) @binding(0) var<uniform> O: OverlayParams;
@group(0) @binding(1) var depthTex: texture_2d<f32>;

struct VsIn {
  @location(0) pos: vec3f,
  @location(1) color: vec4f,
  @location(2) flags: u32,
}

struct VsOut {
  @builtin(position) clip: vec4f,
  @location(0) color: vec4f,
  @location(1) viewPos: vec3f,
  @location(2) @interpolate(flat) flags: u32,
}

@vertex
fn vs_overlay(v: VsIn) -> VsOut {
  var o: VsOut;
  o.clip = O.viewProj * vec4f(v.pos, 1.0);
  o.color = v.color;
  o.viewPos = (O.worldToCam * vec4f(v.pos, 1.0)).xyz;
  o.flags = v.flags;
  return o;
}

@fragment
fn fs_overlay(i: VsOut) -> @location(0) vec4f {
  var c = i.color;
  if ((i.flags & OVERLAY_ON_TOP) == 0u && O.hasDepth != 0u) {
    let px = min(vec2u(i.clip.xy / vec2f(O.dstSize) * vec2f(O.srcSize)), O.srcSize - 1u);
    let scene = textureLoad(depthTex, px, 0).r;
    let bits = bitcast<u32>(scene) & 0x7fffffffu;
    let valid = scene > 0.0 && bits < 0x7f800000u && scene < 1e30;
    let d = select(length(i.viewPos), -i.viewPos.z, O.depthKind == 0u);
    if (valid && d > scene * (1.0 + O.depthEps)) {
      if (O.hiddenAlpha <= 0.0) { discard; }
      c.a *= O.hiddenAlpha;
    }
  }
  return c;
}
