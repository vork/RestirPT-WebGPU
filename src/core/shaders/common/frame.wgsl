// Per-frame uniforms (mirror of src/core/render/frame-uniforms.ts; keep offsets in sync, see FRAME_LAYOUT).
// All positions are in the render-internal recentred frame: p_int = p_world - origin (math.md#raster).
// Binding: @group(0) @binding(0) unless the includer defines FRAME_CUSTOM_BINDING and declares `frame` itself.
#include "common/math.wgsl"
#include "common/rng.wgsl"

struct CameraFrame {
  camToWorld: mat4x4f,   // columns: right, up, back (+Z; the camera looks down -Z), position
  worldToCam: mat4x4f,
  yfov: f32,             // vertical fov, radians (sensor_fit VERTICAL)
  tanHalfY: f32,
  aspect: f32,           // W / H of the internal resolution
  znear: f32,
}

struct FrameUniforms {
  cam: CameraFrame,        // offset 0
  prevCam: CameraFrame,    // 144 (equals cam on the first frame and after a reset)
  resolution: vec2u,       // 288 internal resolution
  invResolution: vec2f,    // 296
  frameIndex: u32,         // 304 frames since history reset (frozen while paused / FRAME_FREEZE_FRAME)
  seedIndex: u32,          // 308 the per-frame RNG index (frozen under FRAME_FREEZE_SEED)
  runSeed: u32,            // 312
  flags: u32,              // 316 FRAME_* bits
  jitterMode: u32,         // 320 JITTER_* below
  _pad0: u32,              // 324
  jitter: vec2f,           // 328 CPU jitter for JITTER_NONE / JITTER_R2 (all pixels)
  origin: vec3f,           // 336 recentring offset O (world = internal + O), informational in f32
  exposure: f32,           // 348 linear multiplier 2^EV
  time: f32,               // 352 seconds since start (paused time does not advance)
  dt: f32,                 // 356 seconds, clamped to 0.1
  sceneDiag: f32,          // 360 bounding-box diagonal (m)
  _pad1: f32,              // 364
}                          // 368 bytes

const FRAME_PAUSED: u32 = 1u;
const FRAME_FREEZE_SEED: u32 = 2u;
const FRAME_FREEZE_FRAME: u32 = 4u;
const FRAME_RESET_HISTORY: u32 = 8u;   // set for exactly one frame
const FRAME_CAMERA_MOVED: u32 = 16u;   // cam != prevCam

const JITTER_NONE: u32 = 0u;           // pixel centre (+ frame.jitter, normally 0.5)
const JITTER_IID: u32 = 1u;            // validation: hash(runSeed, t, p, JITTER), i.i.d. per run and frame
const JITTER_R2: u32 = 2u;             // interactive: R2 sequence with a per-run Cranley-Patterson rotation (CPU)

#if !FRAME_CUSTOM_BINDING
@group(0) @binding(0) var<uniform> frame: FrameUniforms;
#endif

/// Per-pixel, per-frame seed. Streams (path / resample / jitter) are separated by rng.wgsl stream ids.
fn frame_pixel_seed(pixel: vec2u) -> u32 {
  return pcg3d(vec3u(frame.runSeed, frame.seedIndex, pixel.y * frame.resolution.x + pixel.x)).x;
}

/// Subpixel jitter (u, v) in [0,1)^2. math.md#raster
fn frame_pixel_jitter(pixel: vec2u) -> vec2f {
  if (frame.jitterMode == JITTER_IID) { return rand2(frame_pixel_seed(pixel), 0u, STREAM_JITTER); }
  return frame.jitter;
}

/// Unnormalised camera-space direction for image pixel (c, r) (row 0 = top) and subpixel (u, v). math.md#raster
fn frame_dir_cam(pixel: vec2u, uv: vec2f, cam: CameraFrame) -> vec3f {
  let W = f32(frame.resolution.x);
  let H = f32(frame.resolution.y);
  let tx = cam.tanHalfY * W / H;
  let x = (2.0 * (f32(pixel.x) + uv.x) / W - 1.0) * tx;
  let y = (2.0 * (H - 1.0 - f32(pixel.y) + uv.y) / H - 1.0) * cam.tanHalfY;
  return vec3f(x, y, -1.0);
}

struct CameraRay { o: vec3f, d: vec3f }

/// Primary ray of the current camera (internal frame). math.md#raster
fn frame_camera_ray(pixel: vec2u, uv: vec2f) -> CameraRay {
  let dc = frame_dir_cam(pixel, uv, frame.cam);
  let m = frame.cam.camToWorld;
  let d = normalize(m[0].xyz * dc.x + m[1].xyz * dc.y + m[2].xyz * dc.z);
  return CameraRay(m[3].xyz, d);
}

/// Continuous image coordinates (col, row; pixel centres at +0.5, row 0 = top) of internal-frame point p as seen
/// by `cam`. Returns (-1, -1) behind the camera. Used for motion vectors / reprojection. math.md#raster
fn frame_project(p: vec3f, cam: CameraFrame) -> vec2f {
  let pc = (cam.worldToCam * vec4f(p, 1.0)).xyz;
  if (pc.z > -1e-12) { return vec2f(-1.0); }
  let ndc = vec2f(pc.x / (-pc.z * cam.tanHalfY * cam.aspect), pc.y / (-pc.z * cam.tanHalfY));
  let res = vec2f(frame.resolution);
  return vec2f((ndc.x + 1.0) * 0.5 * res.x, res.y - (ndc.y + 1.0) * 0.5 * res.y);
}

/// Linear view depth (distance along -Z) of an internal-frame point for `cam`.
fn frame_view_depth(p: vec3f, cam: CameraFrame) -> f32 {
  return -(cam.worldToCam * vec4f(p, 1.0)).z;
}
