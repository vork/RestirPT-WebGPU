// Test-pattern "renderer" used when no scene/renderer is wired: ray-casts an analytic scene (checker ground at
// world y = 0, a unit sphere, an HDR sky) with the frame camera, plus a strip of calibration patches for the
// exposure/view-transform check. Exercises frame uniforms, debug AOVs, probe records, counters and linear depth.
#include "common/frame.wgsl"
#include "debug/debug-common.wgsl"

@group(0) @binding(1) var colorOut: texture_storage_2d<$COLOR_FORMAT, write>;
@group(0) @binding(2) var depthOut: texture_storage_2d<r32float, write>;

// Calibration strip (top 10% of the image): scene-linear values, exposure/tonemap applied by the blit.
const PATCHES = array<f32, 8>(0.0, 0.001, 0.01, 0.05, 0.18, 0.5, 1.0, 4.0);

fn sky(d: vec3f) -> vec3f {
  let t = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
  let sun = pow(max(dot(d, normalize(vec3f(0.4, 0.6, -0.5))), 0.0), 512.0) * 40.0;
  return mix(vec3f(0.9, 0.85, 0.8), vec3f(0.25, 0.45, 0.9), t) + vec3f(sun);
}

@compute @workgroup_size(8, 8, 1)
fn test_pattern(@builtin(global_invocation_id) gid: vec3u) {
  let p = gid.xy;
  let res = frame.resolution;
  if (any(p >= res)) { return; }
  let uvPix = (vec2f(p) + 0.5) * frame.invResolution;
  debug_write3(p, DBG_TEST_UV, vec3f(uvPix, 0.0));

  if (uvPix.y < 0.1) {
    let i = min(u32(uvPix.x * 8.0), 7u);
    let v = PATCHES[i];
    textureStore(colorOut, p, vec4f(v, v, v, 1.0));
    textureStore(depthOut, p, vec4f(0.0));
    debug_write_code(p, DBG_TEST_CELL, i);
    return;
  }

  let ray = frame_camera_ray(p, frame_pixel_jitter(p));
  var col = sky(ray.d);
  var t = FLT_MAX;
  var n = vec3f(0.0);
  var cell = DBG_CODE_NONE;

  // ground plane at world y = 0 (internal y = -origin.y)
  let groundY = -frame.origin.y;
  if (abs(ray.d.y) > 1e-8) {
    let tg = (groundY - ray.o.y) / ray.d.y;
    if (tg > 0.0 && tg < t) {
      t = tg;
      n = vec3f(0.0, 1.0, 0.0);
      let hp = ray.o + tg * ray.d + frame.origin;
      let c = vec2i(floor(hp.xz));
      cell = u32(c.x & 0xffff) | (u32(c.y & 0xffff) << 16u);
      let chk = ((c.x + c.y) & 1) == 0;
      let fog = exp(-tg * 0.02);
      col = mix(sky(ray.d), select(vec3f(0.08), vec3f(0.6), chk) * (0.3 + 0.7 * max(dot(n, normalize(vec3f(0.4, 0.6, -0.5))), 0.0)), fog);
    }
  }
  // unit sphere centred at world (0, 1, 0)
  let oc = ray.o - (vec3f(0.0, 1.0, 0.0) - frame.origin);
  let b = dot(oc, ray.d);
  let disc = b * b - (dot(oc, oc) - 1.0);
  if (disc > 0.0) {
    let ts = -b - sqrt(disc);
    if (ts > 0.0 && ts < t) {
      t = ts;
      n = normalize(oc + ts * ray.d);
      cell = 0xfffeu;
      let l = normalize(vec3f(0.4, 0.6, -0.5));
      col = vec3f(0.8, 0.3, 0.2) * (0.1 + max(dot(n, l), 0.0)) + 0.2 * sky(reflect(ray.d, n));
    }
  }

  let hit = t < FLT_MAX;
  let viewZ = select(0.0, t * -dot(ray.d, frame.cam.camToWorld[2].xyz), hit);
  textureStore(colorOut, p, vec4f(col, 1.0));
  textureStore(depthOut, p, vec4f(viewZ, 0.0, 0.0, 0.0));
  debug_write1(p, DBG_TEST_DEPTH, select(0.0, t, hit));
  debug_write3(p, DBG_TEST_NORMAL, n);
  debug_write_code(p, DBG_TEST_CELL, cell);
  debug_check_finite(col);
  probe_record(p, PROBE_TAG_USER, vec4f(col, t));
  probe_record(p, PROBE_TAG_USER + 1u, vec4f(ray.d, f32(frame.frameIndex)));
}
