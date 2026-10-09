// perf2 WP-7g: the present's view transform once per internal texel (app only). Linear 'color' -> resolve_display
// (exposure + view transform; magenta for a non-finite texel when highlighted) -> dispOut (rgba16float, display-encoded,
// filterable), which blit.wgsl upscales (BLIT_PRETONED). Before, every canvas pixel re-ran the transform on each of its
// 16 bicubic taps.
#include "post/resolve.wgsl"

struct ToneParams {
  size: vec2u,
  exposure: f32,       // 2^EV
  tonemap: u32,        // TONEMAP_*
  nonFinite: u32,      // 1: non-finite texels are magenta
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var colorTex: texture_2d<f32>;
@group(0) @binding(1) var dispOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> T: ToneParams;

@compute @workgroup_size(8, 8, 1)
fn tonemap(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= T.size)) { return; }
  let c = textureLoad(colorTex, gid.xy, 0).rgb;
  var d = resolve_display(c, T.exposure, T.tonemap);
  if (T.nonFinite != 0u && !all_finite3(c)) { d = vec3f(1.0, 0.0, 1.0); }
  textureStore(dispOut, gid.xy, vec4f(d, 1.0));
}
