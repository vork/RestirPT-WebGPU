// Fixtures of the temporal GPU tests (restir-temporal-api.md §6.1; T-B): camera paths in the box scene, a chain driver
// (advance() + frameUnits per frame, one submit per frame, TD26), tState / reservoir readback and CPU decoders.
// OWNER T-B.
import { readBuffer } from '../../src/core/gpu/readback.ts';
import type { EnvParamsCpu } from '../../src/core/render/env-gpu.ts';
import type { CameraState } from '../../src/core/render/frame-uniforms.ts';
import type { RestirCounters, RestirKernel } from '../../src/core/render/restir/kernel.ts';
import { RES_WORDS, RW, decodeTStateLocal, type TStateRecord } from '../../src/core/render/restir/layout.ts';
import type { LightData } from '../../src/core/scene/types.ts';
import { boxCamera, type RestirRig } from './restir-fixtures.ts';

/** boxCamera translated by (dx, dy, dz) and yawed by `yaw` radians about +Y (column-major camToWorld). */
export function movedCamera(dx: number, dy: number, dz: number, yaw = 0, yfov?: number): CameraState {
  const c = boxCamera();
  const m = c.camToWorld.slice();
  const cs = Math.cos(yaw), sn = Math.sin(yaw);
  for (const col of [0, 4, 8]) {                          // rotate the basis columns about +Y
    const x = m[col], z = m[col + 2];
    m[col] = cs * x + sn * z;
    m[col + 2] = -sn * x + cs * z;
  }
  m[12] += dx; m[13] += dy; m[14] += dz;
  return { camToWorld: m, yfov: yfov ?? c.yfov };
}

export interface ChainFrame { t: number; camera: CameraState; lights: LightData[]; reset?: boolean; env?: { params: EnvParamsCpu; mapId: string } }
export interface FrameResult { t: number; histValid: boolean; flags: number; counters: RestirCounters }

/** Run frames through RestirKernel.advance + frameUnits (one submit per frame). `after(f, res)` runs after each frame's
 *  GPU work (readbacks); counters are read with reset per frame. */
export async function runChain(rig: RestirRig, frames: ChainFrame[], after?: (f: ChainFrame, r: FrameResult) => Promise<void> | void): Promise<FrameResult[]> {
  const k = rig.kernel, device = rig.g.device;
  const clear = device.createCommandEncoder();
  clear.clearBuffer(rig.accum); clear.clearBuffer(rig.counters); clear.clearBuffer(k.resources.arena, 0, 256);
  device.queue.submit([clear.finish()]);
  const out: FrameResult[] = [];
  for (const f of frames) {
    const adv = k.advance({ t: f.t, camera: f.camera, lights: f.lights, reset: f.reset, env: f.env });
    k.beginSubmit();
    const enc = device.createCommandEncoder({ label: `chain-${f.t}` });
    for (const u of k.frameUnits(f.t, { accum: rig.accum, counters: rig.counters })) u.encode(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const r = { t: f.t, histValid: adv.histValid, flags: adv.flags, counters: await k.readCounters(true) };
    out.push(r);
    await after?.(f, r);
  }
  return out;
}

/** Temporal state of the last frame: tState per pixel, the history (res[h]) and output (res[w]) reservoirs. */
export interface TemporalSnapshot {
  P: number; NS: number; h: number; w: number;
  ts: (ai: number) => TStateRecord;
  hist: Uint32Array; out: Uint32Array; histF: Float32Array; outF: Float32Array;
  rec: (buf: Uint32Array, ai: number, word: number) => number;
  recF: (buf: Float32Array, ai: number, word: number) => number;
}

export async function temporalSnapshot(k: RestirKernel): Promise<TemporalSnapshot> {
  const res = k.resources;
  const P = res.pixels, NS = res.alloc.slots;
  const words = await k.readTemporalState();
  const h = k.historyIndex(), w = k.resBase();
  const hist = h >= 0 ? await k.readReservoirs(h as 0 | 1) : new Uint32Array(P * RES_WORDS);
  const out = await k.readReservoirs(w as 0 | 1);
  return {
    P, NS, h, w, hist, out,
    histF: new Float32Array(hist.buffer), outF: new Float32Array(out.buffer),
    ts: (ai) => decodeTStateLocal(words, P, NS, ai),
    rec: (buf, ai, word) => buf[ai * RES_WORDS + word],
    recF: (buf, ai, word) => buf[ai * RES_WORDS + word],
  };
}

export const lum = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;
/** lum F of record ai of a reservoir buffer (as f32 words). */
export const recLumF = (f: Float32Array, ai: number) => lum(f[ai * RES_WORDS + RW.F], f[ai * RES_WORDS + RW.F + 1], f[ai * RES_WORDS + RW.F + 2]);

/** Read an arbitrary GPU buffer range as u32. */
export async function readU32(device: GPUDevice, b: GPUBuffer, bytes = b.size): Promise<Uint32Array> {
  return new Uint32Array(await readBuffer(device, b, bytes));
}

/** Lights of frame t for the light-change tests: light `id` translated by t·dp and its power scaled by `power(t)`. */
export function animatedLights(base: LightData[], t: number, o: { id: number; dp?: [number, number, number]; power?: (t: number) => number }[]): LightData[] {
  return base.map((l) => {
    const a = o.find((x) => x.id === l.id);
    if (!a) return l;
    const m = Array.from(l.matrix as ArrayLike<number>);
    if (a.dp) { m[12] += t * a.dp[0]; m[13] += t * a.dp[1]; m[14] += t * a.dp[2]; }
    return { ...l, matrix: Float32Array.from(m), power: l.power * (a.power ? a.power(t) : 1) };
  });
}

/** FNV-1a over u32 words (hex). */
export function hashU32(u: Uint32Array): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < u.length; i++) {
    let x = u[i];
    for (let k = 0; k < 4; k++) { h ^= x & 0xff; h = Math.imul(h, 0x01000193) >>> 0; x >>>= 8; }
  }
  return h.toString(16).padStart(8, '0');
}
