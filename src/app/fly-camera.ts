// Fly camera (plan §5 M1): DOM-free state + integration so it can be unit tested (tests/app/fly-camera.test.ts).
// DOM wiring lives in fly-controls.ts. World space = glTF canonical, UN-recentred (f64).
import {
  DEG, lookAtYawPitch, matrixToPose, poseMatrix, quatFromYawPitch, slerp, yawPitchFromQuat, type Quat, type Vec3,
} from './camera-math.ts';

export type MoveAction = 'forward' | 'back' | 'left' | 'right' | 'up' | 'down';

/** KeyboardEvent.code -> action (layout independent: 'KeyW' is the physical W position on AZERTY too). */
export const KEY_ACTIONS: Readonly<Record<string, MoveAction>> = {
  KeyW: 'forward', KeyS: 'back', KeyA: 'left', KeyD: 'right', KeyE: 'up', KeyQ: 'down',
};

export function keyToAction(code: string): MoveAction | undefined { return KEY_ACTIONS[code]; }

export const PITCH_LIMIT = 89 * DEG;
export const MAX_DT = 0.1;
export const WHEEL_STEP = 1.25;       // speed factor per wheel notch
export const WHEEL_STEPS_MAX = 40;
export const SHIFT_FACTOR = 4;
export const CTRL_FACTOR = 0.25;

export function clampPitch(p: number): number { return Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, p)); }
export function clampDt(dt: number): number { return Number.isFinite(dt) ? Math.max(0, Math.min(MAX_DT, dt)) : 0; }

/** Speed in m/s: scene diagonal / 10 per second, times WHEEL_STEP^steps, Shift x4, Ctrl x0.25. */
export function flySpeed(diag: number, wheelSteps: number, shift: boolean, ctrl: boolean): number {
  return (diag / 10) * WHEEL_STEP ** wheelSteps * (shift ? SHIFT_FACTOR : 1) * (ctrl ? CTRL_FACTOR : 1);
}

/** Unit move direction: WASD in the world XZ plane (yaw only), E/Q along world ±Y. Zero vector if idle. */
export function moveDirection(held: ReadonlySet<MoveAction>, yaw: number): Vec3 {
  const f = (held.has('forward') ? 1 : 0) - (held.has('back') ? 1 : 0);
  const r = (held.has('right') ? 1 : 0) - (held.has('left') ? 1 : 0);
  const u = (held.has('up') ? 1 : 0) - (held.has('down') ? 1 : 0);
  const s = Math.sin(yaw);
  const c = Math.cos(yaw);
  // forward_h = (-sin, 0, -cos), right = (cos, 0, -sin)
  const d: Vec3 = [-s * f + c * r, u, -c * f - s * r];
  const l = Math.hypot(...d);
  return l > 0 ? [d[0] / l, d[1] / l, d[2] / l] : [0, 0, 0];
}

export interface CameraPose { position: Vec3; quaternion: Quat; yfov: number }
export interface TrackKey { t: number; position: Vec3; quaternion: Quat; yfov: number }

/** Recorded camera track (plan: tracks.camera). Linear position/yfov, slerp orientation. */
export class CameraTrack {
  constructor(public keys: TrackKey[] = []) {}

  get duration(): number { return this.keys.length ? this.keys[this.keys.length - 1].t : 0; }

  push(k: TrackKey): void {
    if (this.keys.length && k.t < this.keys[this.keys.length - 1].t) throw new Error('track keys must be time-ordered');
    this.keys.push({ t: k.t, position: [...k.position], quaternion: [...k.quaternion], yfov: k.yfov });
  }

  /** Pose at time t (clamped to [0, duration], or wrapped when loop). Binary search. */
  sample(t: number, loop = false): CameraPose | undefined {
    const k = this.keys;
    if (!k.length) return undefined;
    const T = this.duration;
    if (loop && T > 0) t = ((t % T) + T) % T;
    if (t <= k[0].t) return { position: [...k[0].position], quaternion: [...k[0].quaternion], yfov: k[0].yfov };
    if (t >= T) { const e = k[k.length - 1]; return { position: [...e.position], quaternion: [...e.quaternion], yfov: e.yfov }; }
    let lo = 0;
    let hi = k.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (k[mid].t <= t) lo = mid; else hi = mid; }
    const a = k[lo];
    const b = k[hi];
    const u = b.t > a.t ? (t - a.t) / (b.t - a.t) : 0;
    return {
      position: [0, 1, 2].map((i) => a.position[i] + (b.position[i] - a.position[i]) * u) as Vec3,
      quaternion: slerp(a.quaternion, b.quaternion, u),
      yfov: a.yfov + (b.yfov - a.yfov) * u,
    };
  }

  toJSON(): { version: 1; frame: string; keys: TrackKey[] } {
    return { version: 1, frame: 'glTF world (right-handed, +Y up, metres, un-recentred)', keys: this.keys };
  }

  static fromJSON(j: unknown): CameraTrack {
    const keys = (j as { keys?: unknown }).keys;
    if (!Array.isArray(keys)) throw new Error('camera track: missing keys');
    const t = new CameraTrack();
    for (const k of keys as TrackKey[]) {
      const ok = Number.isFinite(k.t) && k.position?.length === 3 && k.quaternion?.length === 4 && Number.isFinite(k.yfov);
      if (!ok) throw new Error('camera track: malformed key');
      t.push(k);
    }
    return t;
  }
}

export interface Bookmark { position: Vec3; quaternion: Quat; yfov: number }

export interface FlyCameraOptions {
  sceneDiag?: number;
  /** Radians per CSS pixel of mouse movement. */
  sensitivity?: number;
  invertY?: boolean;
}

export class FlyCamera {
  position: Vec3 = [0, 1.5, 6];
  yaw = 0;
  pitch = 0;
  /** Orientation. Equal to quatFromYawPitch(yaw, pitch) except after setPose() with roll (file cameras, playback). */
  quaternion: Quat = [0, 0, 0, 1];
  yfov = 40 * DEG;
  sceneDiag = 10;
  sensitivity = 0.0025;
  invertY = false;
  wheelSteps = 0;
  readonly held = new Set<MoveAction>();
  shift = false;
  ctrl = false;
  /** Paused: mouse look is ignored (update() is not called by the app while paused). */
  frozen = false;
  /** Pose restored by Home. */
  initialPose: CameraPose;
  readonly bookmarks: (Bookmark | undefined)[] = new Array(10).fill(undefined);

  recording: CameraTrack | undefined;
  private recordT = 0;
  playing: { track: CameraTrack; t: number; loop: boolean } | undefined;
  /** Incremented whenever the pose changes (cheap "moved" test). */
  version = 0;

  constructor(opts: FlyCameraOptions = {}) {
    if (opts.sceneDiag) this.sceneDiag = opts.sceneDiag;
    if (opts.sensitivity !== undefined) this.sensitivity = opts.sensitivity;
    if (opts.invertY !== undefined) this.invertY = opts.invertY;
    this.quaternion = quatFromYawPitch(this.yaw, this.pitch);
    this.initialPose = this.pose();
  }

  get speed(): number { return flySpeed(this.sceneDiag, this.wheelSteps, this.shift, this.ctrl); }

  pose(): CameraPose { return { position: [...this.position], quaternion: [...this.quaternion], yfov: this.yfov }; }

  /** Set an exact pose (keeps roll until the user rotates). */
  setPose(p: CameraPose): void {
    this.position = [...p.position];
    this.quaternion = [...p.quaternion];
    this.yfov = p.yfov;
    const yp = yawPitchFromQuat(p.quaternion);
    this.yaw = yp.yaw;
    this.pitch = clampPitch(yp.pitch);
    this.version++;
  }

  setFromMatrix(camToWorld: ArrayLike<number>, yfov: number): void {
    const { position, quaternion } = matrixToPose(camToWorld);
    this.setPose({ position, quaternion, yfov });
  }

  lookAt(eye: Vec3, target: Vec3): void {
    const yp = lookAtYawPitch(eye, target);
    this.position = [...eye];
    this.yaw = yp.yaw;
    this.pitch = clampPitch(yp.pitch);
    this.quaternion = quatFromYawPitch(this.yaw, this.pitch);
    this.version++;
  }

  /** Frame an axis-aligned box: centre in view, whole diagonal visible, looking down -Z. */
  frameBounds(min: Vec3, max: Vec3): void {
    const c: Vec3 = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    const r = 0.5 * Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    const dist = (r / Math.tan(this.yfov / 2)) * 1.05;
    this.lookAt([c[0], c[1] + 0.15 * r, c[2] + dist], c);
  }

  /** Mouse look by CSS-pixel deltas. */
  look(dx: number, dy: number): void {
    if (this.playing || this.frozen || (dx === 0 && dy === 0)) return;
    this.yaw -= dx * this.sensitivity;
    this.pitch = clampPitch(this.pitch - (this.invertY ? -dy : dy) * this.sensitivity);
    this.yaw = Math.atan2(Math.sin(this.yaw), Math.cos(this.yaw));
    this.quaternion = quatFromYawPitch(this.yaw, this.pitch);
    this.version++;
  }

  /** One wheel notch per call; deltaY > 0 (scroll down) slows down. */
  wheel(deltaY: number): void {
    if (deltaY === 0) return;
    this.wheelSteps = Math.max(-WHEEL_STEPS_MAX, Math.min(WHEEL_STEPS_MAX, this.wheelSteps - Math.sign(deltaY)));
  }

  /** Integrate one frame. dt is clamped to MAX_DT (frame-rate independent, no teleport after a stall). */
  update(dtIn: number): boolean {
    const dt = clampDt(dtIn);
    let moved = false;
    if (this.playing) {
      this.playing.t += dt;
      const p = this.playing.track.sample(this.playing.t, this.playing.loop);
      if (p) { this.setPose(p); moved = true; }
      if (!this.playing.loop && this.playing.t >= this.playing.track.duration) this.playing = undefined;
    } else {
      const d = moveDirection(this.held, this.yaw);
      if (d[0] !== 0 || d[1] !== 0 || d[2] !== 0) {
        const s = this.speed * dt;
        this.position = [this.position[0] + d[0] * s, this.position[1] + d[1] * s, this.position[2] + d[2] * s];
        this.version++;
        moved = true;
      }
    }
    if (this.recording) {
      this.recordT += dt;
      this.recording.push({ t: this.recordT, ...this.pose() });
    }
    return moved;
  }

  camToWorld(): Float64Array { return poseMatrix(this.position, this.quaternion); }

  reset(): void { this.stopPlayback(); this.setPose(this.initialPose); }

  saveBookmark(slot: number): void { this.bookmarks[slot] = this.pose(); }
  recallBookmark(slot: number): boolean {
    const b = this.bookmarks[slot];
    if (!b) return false;
    this.stopPlayback();
    this.setPose(b);
    return true;
  }

  startRecording(): void {
    this.recording = new CameraTrack();
    this.recordT = 0;
    this.recording.push({ t: 0, ...this.pose() });
  }
  stopRecording(): CameraTrack | undefined { const r = this.recording; this.recording = undefined; return r; }
  play(track: CameraTrack, loop = false): void {
    if (!track.keys.length) return;
    this.playing = { track, t: 0, loop };
    const p = track.sample(0);
    if (p) this.setPose(p);
  }
  stopPlayback(): void { this.playing = undefined; }
}
