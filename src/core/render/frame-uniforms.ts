// FrameUniforms: CPU packer for the per-frame uniform block. WGSL mirror: src/core/shaders/common/frame.wgsl.
// Camera matrices arrive in UN-recentred world space (f64); the recentring offset is subtracted here in f64 so the
// f32 uniform keeps full precision near the camera (plan §1.2, math.md#raster).
import type { Bounds } from '../scene/types.ts';

export const FRAME_PAUSED = 1;
export const FRAME_FREEZE_SEED = 2;
export const FRAME_FREEZE_FRAME = 4;
export const FRAME_RESET_HISTORY = 8;
export const FRAME_CAMERA_MOVED = 16;

export const JITTER_NONE = 0;
export const JITTER_IID = 1;
export const JITTER_R2 = 2;
export type JitterMode = typeof JITTER_NONE | typeof JITTER_IID | typeof JITTER_R2;

/** Byte offsets; must match frame.wgsl (tests/app/frame-uniforms.test.ts checks the arithmetic). */
export const FRAME_LAYOUT = {
  cam: 0,
  prevCam: 144,
  cameraSize: 144,
  resolution: 288,
  invResolution: 296,
  frameIndex: 304,
  seedIndex: 308,
  runSeed: 312,
  flags: 316,
  jitterMode: 320,
  jitter: 328,
  origin: 336,
  exposure: 348,
  time: 352,
  dt: 356,
  sceneDiag: 360,
  size: 368,
} as const;

export interface CameraState {
  /** Camera-to-world, column-major 4x4, UN-recentred world (rigid: rotation + translation). */
  camToWorld: Float64Array | Float32Array | number[];
  /** Vertical field of view, radians. */
  yfov: number;
  znear?: number;
}

export interface FrameUniformInput {
  camera: CameraState;
  prevCamera: CameraState;
  width: number;
  height: number;
  frameIndex: number;
  seedIndex: number;
  runSeed: number;
  flags: number;
  jitterMode: JitterMode;
  jitter: [number, number];
  origin: [number, number, number];
  exposure: number;
  time: number;
  dt: number;
  sceneDiag: number;
}

/** Render-internal recentring offset O: the bounds centre (plan §1.2). Everything subtracts the same O. */
export function computeRenderOrigin(bounds: Bounds | undefined): [number, number, number] {
  if (!bounds) return [0, 0, 0];
  const c = [0, 1, 2].map((i) => 0.5 * (bounds.min[i] + bounds.max[i]));
  return c.every(Number.isFinite) ? [c[0], c[1], c[2]] : [0, 0, 0];
}

export function boundsDiagonal(bounds: Bounds | undefined): number {
  if (!bounds) return 1;
  const d = Math.hypot(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2]);
  return Number.isFinite(d) && d > 0 ? d : 1;
}

/** Inverse of a rigid column-major transform (R | t): (R^T | -R^T t). */
export function rigidInverse(m: ArrayLike<number>): Float64Array {
  const o = new Float64Array(16);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[c * 4 + r] = m[r * 4 + c];
  for (let r = 0; r < 3; r++) o[12 + r] = -(o[r] * m[12] + o[4 + r] * m[13] + o[8 + r] * m[14]);
  o[15] = 1;
  return o;
}

/** Recentred copy: translation minus origin, rotation columns re-orthonormalised (drops scale from file cameras). */
export function recentre(m: ArrayLike<number>, origin: readonly number[]): Float64Array {
  const o = new Float64Array(16);
  for (let c = 0; c < 3; c++) {
    const len = Math.hypot(m[c * 4], m[c * 4 + 1], m[c * 4 + 2]) || 1;
    for (let r = 0; r < 3; r++) o[c * 4 + r] = m[c * 4 + r] / len;
  }
  for (let r = 0; r < 3; r++) o[12 + r] = m[12 + r] - origin[r];
  o[15] = 1;
  return o;
}

function writeCamera(f32: Float32Array, byteOffset: number, cam: CameraState, origin: readonly number[], aspect: number): void {
  const base = byteOffset / 4;
  const c2w = recentre(cam.camToWorld, origin);
  const w2c = rigidInverse(c2w);
  f32.set(c2w, base);
  f32.set(w2c, base + 16);
  const t = Math.tan(cam.yfov / 2);
  f32[base + 32] = cam.yfov;
  f32[base + 33] = t;
  f32[base + 34] = aspect;
  f32[base + 35] = cam.znear ?? 1e-4;
}

export function camerasEqual(a: CameraState, b: CameraState): boolean {
  if (a.yfov !== b.yfov) return false;
  for (let i = 0; i < 16; i++) if (a.camToWorld[i] !== b.camToWorld[i]) return false;
  return true;
}

/** Pack into `out` (FRAME_LAYOUT.size bytes). */
export function packFrameUniforms(u: FrameUniformInput, out: ArrayBuffer = new ArrayBuffer(FRAME_LAYOUT.size)): ArrayBuffer {
  const f32 = new Float32Array(out);
  const u32 = new Uint32Array(out);
  const aspect = u.width / u.height;
  writeCamera(f32, FRAME_LAYOUT.cam, u.camera, u.origin, aspect);
  writeCamera(f32, FRAME_LAYOUT.prevCam, u.prevCamera, u.origin, aspect);
  const L = FRAME_LAYOUT;
  u32[L.resolution / 4] = u.width;
  u32[L.resolution / 4 + 1] = u.height;
  f32[L.invResolution / 4] = 1 / u.width;
  f32[L.invResolution / 4 + 1] = 1 / u.height;
  u32[L.frameIndex / 4] = u.frameIndex >>> 0;
  u32[L.seedIndex / 4] = u.seedIndex >>> 0;
  u32[L.runSeed / 4] = u.runSeed >>> 0;
  let flags = u.flags;
  if (!camerasEqual(u.camera, u.prevCamera)) flags |= FRAME_CAMERA_MOVED;
  u32[L.flags / 4] = flags >>> 0;
  u32[L.jitterMode / 4] = u.jitterMode;
  u32[L.jitterMode / 4 + 1] = 0;
  f32[L.jitter / 4] = u.jitter[0];
  f32[L.jitter / 4 + 1] = u.jitter[1];
  f32.set(u.origin, L.origin / 4);
  f32[L.exposure / 4] = u.exposure;
  f32[L.time / 4] = u.time;
  f32[L.dt / 4] = u.dt;
  f32[L.sceneDiag / 4] = u.sceneDiag;
  f32[L.sceneDiag / 4 + 1] = 0;
  return out;
}

/** R2 low-discrepancy jitter with a per-run Cranley–Patterson rotation (interactive mode only; plan §1.2). */
export function r2Jitter(frameIndex: number, runSeed: number): [number, number] {
  const g = 1.32471795724474602596;
  const a1 = 1 / g;
  const a2 = 1 / (g * g);
  const rot = [(Math.imul(runSeed, 0x9e3779b1) >>> 0) / 2 ** 32, (Math.imul(runSeed, 0x85ebca77) >>> 0) / 2 ** 32];
  const fr = (x: number) => x - Math.floor(x);
  return [fr(0.5 + a1 * frameIndex + rot[0]), fr(0.5 + a2 * frameIndex + rot[1])];
}

/** Owns the GPU uniform buffer for FrameUniforms. */
export class FrameUniformBuffer {
  readonly buffer: GPUBuffer;
  private readonly scratch = new ArrayBuffer(FRAME_LAYOUT.size);
  last: FrameUniformInput | undefined;

  constructor(private readonly device: GPUDevice) {
    this.buffer = device.createBuffer({
      label: 'frame-uniforms',
      size: FRAME_LAYOUT.size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  write(u: FrameUniformInput): void {
    this.last = u;
    this.device.queue.writeBuffer(this.buffer, 0, packFrameUniforms(u, this.scratch));
  }

  destroy(): void { this.buffer.destroy(); }
}
