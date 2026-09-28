// Keyframe animation for the camera, every analytic light and the environment (plan §5 M3a "Animation", §3 step 0
// "Tracks", §1.4b "γ and strength are timeline tracks").
//
// Model: tracks keyed by target ('camera' | 'env' | 'light:<id>'), each with independent channels:
//   position  vec3  lerp        rotation  quat [x,y,z,w]  slerp (shortest path)      yfov   scalar lerp (camera)
//   power     W     lerp        color     rgb             lerp                        rotationZ / strength (env)
// A key's `interp` governs the segment that STARTS at it: 'linear' interpolates to the next key, 'step' holds its
// value until the next key (left-closed: at exactly the next key's time the next value applies). Before the first
// key the first value holds, after the last key the last value holds (or the timeline wraps when looping).
//
// Frames export (docs/decisions/scene-bridge.md "frames"): frame k is the state resolved at t = k / fps, written as
// CONSTANT values (Blender keys each frame CONSTANT: no sub-frame interpolation, no motion blur), with light matrices
// rounded to f32 exactly as the LightStore / renderer hold them, so our frame k and Cycles' frame k see identical
// values. The bridge carries camera matrix/yfov, light matrix/power and env rotationZ/strength; an animated light
// colour cannot be represented and is a hard error (bridge rule: never approximate silently).
import type { LightData } from './types.ts';
import type { PackageFrame } from './scene-package.ts';

export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];
export type Interp = 'linear' | 'step';
export type ChannelKind = 'position' | 'rotation' | 'power' | 'color' | 'yfov' | 'rotationZ' | 'strength';
export type TargetId = 'camera' | 'env' | `light:${number}`;

export const CHANNEL_SIZE: Record<ChannelKind, number> = { position: 3, rotation: 4, power: 1, color: 3, yfov: 1, rotationZ: 1, strength: 1 };
export const TARGET_CHANNELS = {
  camera: ['position', 'rotation', 'yfov'],
  env: ['rotationZ', 'strength'],
  light: ['position', 'rotation', 'power', 'color'],
} as const satisfies Record<string, readonly ChannelKind[]>;

export interface Key { t: number; v: number[]; interp: Interp }
export interface Track { target: TargetId; channels: Partial<Record<ChannelKind, Key[]>> }

export interface AnimationJson {
  version: 1;
  duration: number;
  fps: number;
  loop: boolean;
  tracks: Track[];
}

export class AnimationError extends Error {}

export const lightTarget = (id: number): TargetId => `light:${id}`;
export const targetLightId = (t: TargetId): number | undefined => (t.startsWith('light:') ? Number(t.slice(6)) : undefined);
const kindOf = (t: TargetId): keyof typeof TARGET_CHANNELS => (t === 'camera' ? 'camera' : t === 'env' ? 'env' : 'light');

/** Key-time equality tolerance (seconds). */
export const KEY_EPS = 1e-9;

// ---- small f64 math ----------------------------------------------------------------------------------------------

export function quatNormalize(q: readonly number[]): Quat {
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

export function quatMul(a: readonly number[], b: readonly number[]): Quat {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

export function quatAxisAngle(axis: readonly number[], angle: number): Quat {
  const l = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const s = Math.sin(angle / 2) / l;
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
}

export function quatRotate(q: readonly number[], v: readonly number[]): Vec3 {
  const [x, y, z, w] = q;
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
}

/** Spherical interpolation along the SHORTER arc (q and −q are the same rotation). Result is unit length. */
export function slerpShortest(a: readonly number[], b: readonly number[], t: number): Quat {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const s = d < 0 ? -1 : 1;
  d *= s;
  if (d > 0.9995) {
    return quatNormalize([0, 1, 2, 3].map((i) => a[i] + (s * b[i] - a[i]) * t));
  }
  const th = Math.acos(Math.min(1, d));
  const sn = Math.sin(th);
  const wa = Math.sin((1 - t) * th) / sn;
  const wb = (s * Math.sin(t * th)) / sn;
  return quatNormalize([0, 1, 2, 3].map((i) => wa * a[i] + wb * b[i]));
}

/** Column-major rigid matrix from position + unit quaternion (f64). */
export function poseToMatrix(p: readonly number[], q: readonly number[]): Float64Array {
  const [x, y, z, w] = quatNormalize(q);
  const m = new Float64Array(16);
  m[0] = 1 - 2 * (y * y + z * z); m[1] = 2 * (x * y + z * w); m[2] = 2 * (x * z - y * w);
  m[4] = 2 * (x * y - z * w); m[5] = 1 - 2 * (x * x + z * z); m[6] = 2 * (y * z + x * w);
  m[8] = 2 * (x * z + y * w); m[9] = 2 * (y * z - x * w); m[10] = 1 - 2 * (x * x + y * y);
  m[12] = p[0]; m[13] = p[1]; m[14] = p[2]; m[15] = 1;
  return m;
}

/** Position + unit quaternion of a rigid (or uniformly scaled) column-major matrix. */
export function matrixToPoseQ(m: ArrayLike<number>): { position: Vec3; quaternion: Quat } {
  const c = [0, 1, 2].map((j) => {
    const l = Math.hypot(m[j * 4], m[j * 4 + 1], m[j * 4 + 2]) || 1;
    return [m[j * 4] / l, m[j * 4 + 1] / l, m[j * 4 + 2] / l];
  });
  const m00 = c[0][0], m10 = c[0][1], m20 = c[0][2];
  const m01 = c[1][0], m11 = c[1][1], m21 = c[1][2];
  const m02 = c[2][0], m12 = c[2][1], m22 = c[2][2];
  const tr = m00 + m11 + m22;
  let q: Quat;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    q = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s];
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    q = [0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s];
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    q = [(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s];
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    q = [(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s];
  }
  return { position: [m[12], m[13], m[14]], quaternion: quatNormalize(q) };
}

/** Unit quaternion whose local −Z points along `dir` (local +Y as close to world `up` as possible). */
export function quatLookDir(dir: readonly number[], up: readonly number[] = [0, 1, 0]): Quat {
  const l = Math.hypot(dir[0], dir[1], dir[2]) || 1;
  const z: Vec3 = [-dir[0] / l, -dir[1] / l, -dir[2] / l];
  let u = up;
  if (Math.abs(z[0] * u[0] + z[1] * u[1] + z[2] * u[2]) > 0.999) u = Math.abs(z[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1];
  let x: Vec3 = [u[1] * z[2] - u[2] * z[1], u[2] * z[0] - u[0] * z[2], u[0] * z[1] - u[1] * z[0]];
  const xl = Math.hypot(...x);
  x = [x[0] / xl, x[1] / xl, x[2] / xl];
  const y: Vec3 = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  return matrixToPoseQ([x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, z[0], z[1], z[2], 0, 0, 0, 0, 1]).quaternion;
}

// ---- channel evaluation ------------------------------------------------------------------------------------------

/** Value of a channel at time t (keys sorted by t). */
export function evalChannel(keys: readonly Key[], kind: ChannelKind, t: number): number[] {
  if (!keys.length) throw new AnimationError('empty channel');
  if (t <= keys[0].t) return [...keys[0].v];
  const last = keys[keys.length - 1];
  if (t >= last.t) return [...last.v];
  let lo = 0, hi = keys.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (keys[mid].t <= t) lo = mid; else hi = mid; }
  const a = keys[lo], b = keys[hi];
  if (a.interp === 'step') return [...a.v];
  const u = (t - a.t) / (b.t - a.t);
  if (kind === 'rotation') return slerpShortest(a.v, b.v, u);
  return a.v.map((x, i) => x + (b.v[i] - x) * u);
}

// ---- resolved state -----------------------------------------------------------------------------------------------

export interface BaseCamera { position: Vec3; quaternion: Quat; yfov: number }
export interface BaseEnv { rotationZ: number; strength: number }
export interface BaseState {
  camera?: BaseCamera;
  lights: readonly LightData[];
  env?: BaseEnv;
}

export interface ResolvedCamera { position: Vec3; quaternion: Quat; yfov: number; matrix: Float64Array }
export interface ResolvedLight { id: number; matrix: Float32Array; power: number; color: Vec3; animated: ChannelKind[] }
export interface ResolvedState {
  t: number;
  /** Only when the camera has a track. */
  camera?: ResolvedCamera;
  /** Only lights that have a track (others keep their base state). */
  lights: Map<number, ResolvedLight>;
  env?: { rotationZ: number; strength: number };
}

export interface AnimationChange { kind: 'keys' | 'settings' | 'load'; target?: TargetId }

export class Animation {
  duration = 10;
  fps = 24;
  loop = true;
  private readonly tracks = new Map<TargetId, Track>();
  private readonly listeners = new Set<(c: AnimationChange) => void>();

  onChange(cb: (c: AnimationChange) => void): () => void { this.listeners.add(cb); return () => this.listeners.delete(cb); }
  private notify(c: AnimationChange): void { for (const cb of [...this.listeners]) cb(c); }
  /** Notify listeners after a series of setKey(..., notify = false) calls. */
  touch(target?: TargetId): void { this.notify({ kind: 'keys', target }); }

  get targets(): TargetId[] { return [...this.tracks.keys()]; }
  track(target: TargetId): Readonly<Track> | undefined { return this.tracks.get(target); }
  hasTrack(target: TargetId): boolean { const t = this.tracks.get(target); return !!t && Object.values(t.channels).some((k) => k && k.length); }
  get isEmpty(): boolean { return this.targets.every((t) => !this.hasTrack(t)); }

  /** Insert or replace (same time within KEY_EPS) a key. */
  setKey(target: TargetId, channel: ChannelKind, t: number, v: readonly number[], interp: Interp = 'linear', notify = true): void {
    if (!(TARGET_CHANNELS[kindOf(target)] as readonly ChannelKind[]).includes(channel)) throw new AnimationError(`${target} has no '${channel}' channel`);
    if (!Number.isFinite(t) || t < 0) throw new AnimationError(`bad key time ${t}`);
    if (v.length !== CHANNEL_SIZE[channel] || v.some((x) => !Number.isFinite(x))) throw new AnimationError(`bad ${channel} value [${v.join(', ')}]`);
    if (channel === 'power' && v[0] < 0) throw new AnimationError('power must be ≥ 0');
    if (channel === 'color' && v.some((x) => x < 0)) throw new AnimationError('colour must be ≥ 0');
    if (channel === 'yfov' && !(v[0] > 0 && v[0] < Math.PI)) throw new AnimationError('yfov must be in (0, π)');
    const val = channel === 'rotation' ? quatNormalize(v) : [...v];
    let tr = this.tracks.get(target);
    if (!tr) { tr = { target, channels: {} }; this.tracks.set(target, tr); }
    const keys = (tr.channels[channel] ??= []);
    const i = keys.findIndex((k) => Math.abs(k.t - t) <= KEY_EPS);
    const key: Key = { t: i >= 0 ? keys[i].t : t, v: val, interp };
    if (i >= 0) keys[i] = key;
    else { keys.push(key); keys.sort((a, b) => a.t - b.t); }
    if (t > this.duration) this.duration = t;
    if (notify) this.notify({ kind: 'keys', target });
  }

  /** Remove keys at time t (all channels of the target when `channel` is omitted). Returns how many were removed. */
  removeKey(target: TargetId, t: number, channel?: ChannelKind): number {
    const tr = this.tracks.get(target);
    if (!tr) return 0;
    let n = 0;
    for (const ch of Object.keys(tr.channels) as ChannelKind[]) {
      if (channel && ch !== channel) continue;
      const keys = tr.channels[ch]!;
      const before = keys.length;
      tr.channels[ch] = keys.filter((k) => Math.abs(k.t - t) > KEY_EPS);
      n += before - tr.channels[ch]!.length;
      if (!tr.channels[ch]!.length) delete tr.channels[ch];
    }
    if (!Object.keys(tr.channels).length) this.tracks.delete(target);
    if (n) this.notify({ kind: 'keys', target });
    return n;
  }

  removeTarget(target: TargetId): Track | undefined {
    const tr = this.tracks.get(target);
    if (tr) { this.tracks.delete(target); this.notify({ kind: 'keys', target }); }
    return tr ? cloneTrack(tr) : undefined;
  }

  /** Re-attach a track (undo) or move one to another target (light type change → new id). */
  setTrack(target: TargetId, track: Track | undefined): void {
    if (track && Object.keys(track.channels).length) {
      const c = cloneTrack(track);
      c.target = target;
      this.tracks.set(target, c);
    } else this.tracks.delete(target);
    this.notify({ kind: 'keys', target });
  }

  replaceChannel(target: TargetId, channel: ChannelKind, keys: readonly Key[]): void {
    let tr = this.tracks.get(target);
    if (!tr) { tr = { target, channels: {} }; this.tracks.set(target, tr); }
    if (keys.length) tr.channels[channel] = keys.map((k) => ({ t: k.t, v: [...k.v], interp: k.interp })).sort((a, b) => a.t - b.t);
    else delete tr.channels[channel];
    if (!Object.keys(tr.channels).length) this.tracks.delete(target);
    const end = keys.reduce((m, k) => Math.max(m, k.t), 0);
    if (end > this.duration) this.duration = end;
    this.notify({ kind: 'keys', target });
  }

  /** Sorted unique key times (of one target, or all). */
  keyTimes(target?: TargetId): number[] {
    const ts: number[] = [];
    for (const tr of this.tracks.values()) {
      if (target && tr.target !== target) continue;
      for (const keys of Object.values(tr.channels)) for (const k of keys!) ts.push(k.t);
    }
    ts.sort((a, b) => a - b);
    return ts.filter((t, i) => i === 0 || t - ts[i - 1] > KEY_EPS);
  }

  setSettings(s: Partial<Pick<Animation, 'duration' | 'fps' | 'loop'>>): void {
    if (s.duration !== undefined) { if (!(s.duration > 0)) throw new AnimationError('duration must be > 0'); this.duration = s.duration; }
    if (s.fps !== undefined) { if (!(s.fps > 0)) throw new AnimationError('fps must be > 0'); this.fps = s.fps; }
    if (s.loop !== undefined) this.loop = s.loop;
    this.notify({ kind: 'settings' });
  }

  /** Map a playback time into the animation's domain (wrap when looping, clamp otherwise). */
  wrapTime(t: number): number {
    if (this.loop && this.duration > 0) return ((t % this.duration) + this.duration) % this.duration;
    return Math.max(0, Math.min(this.duration, t));
  }

  /** Resolve every animated target at time t (not wrapped: pass wrapTime(t) for playback). */
  evaluate(t: number, base: BaseState): ResolvedState {
    const out: ResolvedState = { t, lights: new Map() };
    const cam = this.tracks.get('camera');
    if (cam && Object.keys(cam.channels).length) {
      const b = base.camera ?? { position: [0, 0, 0] as Vec3, quaternion: [0, 0, 0, 1] as Quat, yfov: 40 * Math.PI / 180 };
      const position = (cam.channels.position ? evalChannel(cam.channels.position, 'position', t) : [...b.position]) as Vec3;
      const quaternion = (cam.channels.rotation ? evalChannel(cam.channels.rotation, 'rotation', t) : [...b.quaternion]) as Quat;
      const yfov = cam.channels.yfov ? evalChannel(cam.channels.yfov, 'yfov', t)[0] : b.yfov;
      out.camera = { position, quaternion, yfov, matrix: poseToMatrix(position, quaternion) };
    }
    for (const l of base.lights) {
      const tr = this.tracks.get(lightTarget(l.id));
      if (!tr || !Object.keys(tr.channels).length) continue;
      const ch = tr.channels;
      const animated = Object.keys(ch) as ChannelKind[];
      let matrix: Float32Array;
      if (ch.position || ch.rotation) {
        const bp = matrixToPoseQ(l.matrix);
        const p = ch.position ? evalChannel(ch.position, 'position', t) : [l.matrix[12], l.matrix[13], l.matrix[14]];
        const q = ch.rotation ? evalChannel(ch.rotation, 'rotation', t) : bp.quaternion;
        matrix = rigidF32(poseToMatrix(p, q));
        if (!ch.rotation) for (let i = 0; i < 12; i++) matrix[i] = l.matrix[i]; // keep the exact base rotation
      } else matrix = Float32Array.from(l.matrix);
      const power = ch.power ? evalChannel(ch.power, 'power', t)[0] : l.power;
      const color = (ch.color ? evalChannel(ch.color, 'color', t) : [...l.color]) as Vec3;
      out.lights.set(l.id, { id: l.id, matrix, power, color, animated });
    }
    const env = this.tracks.get('env');
    if (env && Object.keys(env.channels).length) {
      const b = base.env ?? { rotationZ: 0, strength: 1 };
      out.env = {
        rotationZ: env.channels.rotationZ ? evalChannel(env.channels.rotationZ, 'rotationZ', t)[0] : b.rotationZ,
        strength: env.channels.strength ? evalChannel(env.channels.strength, 'strength', t)[0] : b.strength,
      };
    }
    return out;
  }

  toJSON(): AnimationJson {
    const tracks = [...this.tracks.values()]
      .filter((t) => Object.keys(t.channels).length)
      .sort((a, b) => (a.target < b.target ? -1 : a.target > b.target ? 1 : 0))
      .map(cloneTrack);
    return { version: 1, duration: this.duration, fps: this.fps, loop: this.loop, tracks };
  }

  load(j: AnimationJson): void {
    if (!j || j.version !== 1) throw new AnimationError(`unsupported animation version ${String(j?.version)}`);
    const next = new Animation();
    next.duration = j.duration; next.fps = j.fps; next.loop = j.loop;
    for (const tr of j.tracks ?? []) {
      for (const [ch, keys] of Object.entries(tr.channels) as [ChannelKind, Key[]][]) {
        for (const k of keys) next.setKey(tr.target, ch, k.t, k.v, k.interp === 'step' ? 'step' : 'linear', false);
      }
    }
    if (!(next.duration > 0) || !(next.fps > 0)) throw new AnimationError('bad duration/fps');
    this.tracks.clear();
    for (const [k, v] of next.tracks) {
      // keep the stored values bit-exact (setKey normalises quaternions; re-normalising a unit quaternion is a no-op
      // in most cases but not always to the last bit)
      const src = j.tracks.find((t) => t.target === k)!;
      for (const ch of Object.keys(v.channels) as ChannelKind[]) {
        v.channels[ch] = src.channels[ch]!.map((key): Key => ({ t: key.t, v: [...key.v], interp: key.interp === 'step' ? 'step' : 'linear' }))
          .sort((a, b) => a.t - b.t);
      }
      this.tracks.set(k, v);
    }
    this.duration = j.duration; this.fps = j.fps; this.loop = j.loop;
    this.notify({ kind: 'load' });
  }

  static fromJSON(j: AnimationJson): Animation { const a = new Animation(); a.load(j); return a; }
}

export function cloneTrack(t: Readonly<Track>): Track {
  const channels: Track['channels'] = {};
  for (const [ch, keys] of Object.entries(t.channels) as [ChannelKind, Key[]][]) channels[ch] = keys.map((k) => ({ t: k.t, v: [...k.v], interp: k.interp }));
  return { target: t.target, channels };
}

/** f64 rigid matrix → f32 (as the LightStore / GPU hold it). */
export function rigidF32(m: ArrayLike<number>): Float32Array {
  const out = Float32Array.from(m);
  out[3] = 0; out[7] = 0; out[11] = 0; out[15] = 1;
  return out;
}

// ---- presets ------------------------------------------------------------------------------------------------------

export interface PresetResult { position?: Key[]; rotation?: Key[] }

/** Orbit around `center` in the horizontal plane through the start position (radius = current horizontal distance
 *  unless given). `aim` also keys the rotation so local −Z looks at the centre (spot/area lights, camera). */
export function presetOrbit(o: {
  start: Vec3; center: Vec3; period: number; turns?: number; t0?: number; radius?: number; aim?: boolean; samplesPerTurn?: number; clockwise?: boolean;
}): PresetResult {
  const turns = o.turns ?? 1, t0 = o.t0 ?? 0, n = Math.max(4, Math.round(o.samplesPerTurn ?? 24));
  const dx = o.start[0] - o.center[0], dz = o.start[2] - o.center[2];
  const r = o.radius ?? (Math.hypot(dx, dz) || 1);
  const a0 = Math.atan2(dz, dx);
  const sgn = o.clockwise ? -1 : 1;
  const total = Math.max(1, Math.round(n * turns));
  const position: Key[] = [];
  const rotation: Key[] = [];
  for (let i = 0; i <= total; i++) {
    const f = i / n;
    const a = a0 + sgn * 2 * Math.PI * f;
    const p: Vec3 = [o.center[0] + r * Math.cos(a), o.start[1], o.center[2] + r * Math.sin(a)];
    const t = t0 + f * o.period;
    position.push({ t, v: p, interp: 'linear' });
    if (o.aim) rotation.push({ t, v: quatLookDir([o.center[0] - p[0], o.center[1] - p[1], o.center[2] - p[2]]), interp: 'linear' });
  }
  return o.aim ? { position, rotation } : { position };
}

/** Vertical bob: y(t) = y0 + A·sin(2π (t − t0)/T), sampled `samplesPerPeriod` times per period. */
export function presetBob(o: { start: Vec3; amplitude: number; period: number; cycles?: number; t0?: number; samplesPerPeriod?: number }): PresetResult {
  const cycles = o.cycles ?? 1, t0 = o.t0 ?? 0, n = Math.max(4, Math.round(o.samplesPerPeriod ?? 16));
  const total = Math.max(1, Math.round(n * cycles));
  const position: Key[] = [];
  for (let i = 0; i <= total; i++) {
    const f = i / n;
    position.push({ t: t0 + f * o.period, v: [o.start[0], o.start[1] + o.amplitude * Math.sin(2 * Math.PI * f), o.start[2]], interp: 'linear' });
  }
  return { position };
}

/** Sweep (e.g. a spot light): rotation q(t) = R(axis, angle·sin(2π (t − t0)/T))·q0 about a WORLD axis. */
export function presetSweep(o: { start: Quat; angle: number; period: number; cycles?: number; t0?: number; axis?: Vec3; samplesPerPeriod?: number }): PresetResult {
  const cycles = o.cycles ?? 1, t0 = o.t0 ?? 0, n = Math.max(4, Math.round(o.samplesPerPeriod ?? 16));
  const axis = o.axis ?? [0, 1, 0];
  const total = Math.max(1, Math.round(n * cycles));
  const rotation: Key[] = [];
  for (let i = 0; i <= total; i++) {
    const f = i / n;
    const q = quatNormalize(quatMul(quatAxisAngle(axis, o.angle * Math.sin(2 * Math.PI * f)), o.start));
    rotation.push({ t: t0 + f * o.period, v: q, interp: 'linear' });
  }
  return { rotation };
}

// ---- frames export (scene-bridge.md) ------------------------------------------------------------------------------

export class FrameExportError extends Error {}

export interface FrameExportOptions {
  fps?: number;
  /** Frame numbers to export (default: 0..round(duration·fps)). */
  frames?: readonly number[];
  base: BaseState;
  /** Wrap times into the loop (default false: frames past the end hold the last key). */
  wrap?: boolean;
}

/** All frame numbers of the animation: 0..round(duration·fps). */
export function allFrames(anim: Animation, fps = anim.fps): number[] {
  const n = Math.round(anim.duration * fps);
  return Array.from({ length: n + 1 }, (_, i) => i);
}

/** Resolved per-frame states in the scene-package `frames` format (CONSTANT semantics, see the header). */
export function exportFrames(anim: Animation, o: FrameExportOptions): PackageFrame[] {
  const fps = o.fps ?? anim.fps;
  if (!(fps > 0)) throw new FrameExportError('fps must be > 0');
  const frames = o.frames ?? allFrames(anim, fps);
  const ids = new Set(o.base.lights.map((l) => l.id));
  for (const tg of anim.targets) {
    const id = targetLightId(tg);
    if (id !== undefined && !ids.has(id) && anim.hasTrack(tg)) throw new FrameExportError(`track for unknown light ${id}`);
  }
  if (anim.hasTrack('env') && !o.base.env) throw new FrameExportError('env track but the scene has no environment');
  const seen = new Set<number>();
  return frames.map((k) => {
    if (!Number.isInteger(k) || k < 0) throw new FrameExportError(`bad frame number ${k}`);
    if (seen.has(k)) throw new FrameExportError(`duplicate frame ${k}`);
    seen.add(k);
    const t = frameTime(k, fps);
    const s = anim.evaluate(o.wrap ? anim.wrapTime(t) : t, o.base);
    const f: PackageFrame = { frame: k };
    if (s.camera) f.camera = { matrix: Array.from(s.camera.matrix), yfov: s.camera.yfov };
    if (s.lights.size) {
      const lights: NonNullable<PackageFrame['lights']> = {};
      for (const l of [...s.lights.values()].sort((a, b) => a.id - b.id)) {
        const b = o.base.lights.find((x) => x.id === l.id)!;
        if (l.color.some((c, i) => c !== b.color[i])) {
          throw new FrameExportError(`frame ${k}: light ${l.id} '${b.name}' has an animated colour, which the scene bridge cannot represent (frames carry matrix + power only)`);
        }
        lights[String(l.id)] = { matrix: Array.from(l.matrix), power: l.power };
      }
      f.lights = lights;
    }
    if (s.env) f.env = { rotationZ: s.env.rotationZ, strength: s.env.strength };
    return f;
  });
}

/** t = k / fps (the one definition shared by playback in validation mode and the frames export). */
export function frameTime(k: number, fps: number): number { return k / fps; }
