// U-M5-BITS rig (restir-m6-api.md §7, M6 Gate 0): temporal chains with every M6 feature OFF must be bitwise the M5
// build (main f5c23fd). Renders short chains (advance + frameUnits per frame, history on) on two fixtures and hashes
// the final reservoirs, the frame's mean image, the finalize counters and the arena counters of every frame. The
// goldens in restir-m6.gpu.test.ts were recorded with this file on the unmodified M5 build (VITE_M6_BITS_RECORD=1).
// Cases cover: static history ('full' preset), a moving light + camera + env rotation (refresh, J_P, inverse shifts),
// the interactive preset as it was in M5 (RR, boost 3, cCap 5; reset-free chain; perf2 WP-0: knobs pinned with
// INTERACTIVE_PINNED, the preset's values) and Talbot.
import { readBuffer } from '../../src/core/gpu/readback.ts';
import type { LightData, SceneData } from '../../src/core/scene/types.ts';
import { INTERACTIVE_PINNED, M6_OFF, type RestirPresetName } from '../../src/core/render/restir/presets.ts';
import { allLightsScene, bitFixtureScene, boxCamera, hashF32, restirRig } from './restir-fixtures.ts';

export interface M5BitsCase {
  name: string; scene: 'x_quads' | 'all'; preset: RestirPresetName; settings: Record<string, unknown>; frames: number;
  /** Per frame: move light id 1 by dx (m) and rotate the camera yaw by dyaw (rad) from frame `from` on. */
  motion?: { from: number; dx: number; dyaw: number; envRot?: number };
}

export const M5_BITS_CASES: M5BitsCase[] = [
  { name: 'xq-full-static', scene: 'x_quads', preset: 'full', settings: { maxBounces: 3 }, frames: 5 },
  { name: 'all-full-motion', scene: 'all', preset: 'full', settings: { maxBounces: 3 }, frames: 6, motion: { from: 3, dx: 0.04, dyaw: 0.01, envRot: 0.05 } },
  { name: 'xq-interactive-M5', scene: 'x_quads', preset: 'interactive', settings: { maxBounces: 3, ...INTERACTIVE_PINNED, rounds: 1, slots: 3, boostSlots: 3, cCap: 5, rr: true, rrMinBounces: 3, temporal: true, ...M6_OFF }, frames: 5, motion: { from: 2, dx: 0.03, dyaw: 0.0 } },
  { name: 'all-talbot', scene: 'all', preset: 'full', settings: { maxBounces: 2, temporalMis: 'talbot' }, frames: 4, motion: { from: 2, dx: 0.05, dyaw: 0.0 } },
];

function yawed(cam: { camToWorld: number[]; yfov: number }, a: number): { camToWorld: number[]; yfov: number } {
  if (a === 0) return cam;
  const m = cam.camToWorld.slice();
  const c = Math.cos(a), s = Math.sin(a);
  for (const col of [0, 4, 8]) {          // rotate the three basis columns about +Y
    const x = m[col], z = m[col + 2];
    m[col] = c * x + s * z; m[col + 2] = -s * x + c * z;
  }
  return { camToWorld: m, yfov: cam.yfov };
}

function movedLights(lights: LightData[], dx: number): LightData[] {
  return lights.map((l) => {
    if (l.id !== 1) return l;
    const m = new Float32Array(l.matrix);
    m[12] += dx;
    return { ...l, matrix: m };
  });
}

/** Hashes of one case per frame: final reservoirs, mean image of the frame, finalize counters, arena RSC counters. */
export async function m5BitsCase(c: M5BitsCase, extra: Record<string, unknown> = {}): Promise<string[]> {
  const scene: SceneData = c.scene === 'all' ? allLightsScene() : bitFixtureScene('x_quads');
  const W = 64, H = 64;
  const rig = await restirRig(scene, W, H, { preset: c.preset, settings: { ...c.settings, ...extra } as never, seed: 29 });
  const k = rig.kernel, device = rig.g.device;
  await k.prepare();
  const out: string[] = [];
  let lights = scene.lights;
  let cam = boxCamera();
  let envRot = 0;
  const env0 = scene.env;
  for (let f = 0; f < c.frames; f++) {
    const t = 3 + f;
    if (c.motion && f >= c.motion.from) {
      lights = movedLights(lights, c.motion.dx);
      cam = yawed(cam, c.motion.dyaw);
      envRot += c.motion.envRot ?? 0;
    }
    const env = env0 && c.motion?.envRot ? { params: { ...k.envResources.params, rotationZ: envRot }, mapId: 'synth' } : undefined;
    const clear = device.createCommandEncoder();
    clear.clearBuffer(rig.accum); clear.clearBuffer(rig.counters);
    device.queue.submit([clear.finish()]);
    k.advance({ t, camera: cam, lights, env: env as never });
    k.beginSubmit();
    const enc = device.createCommandEncoder();
    for (const u of k.frameUnits(t, { accum: rig.accum, counters: rig.counters })) u.encode(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const sum = new Float32Array(await readBuffer(device, rig.accum, W * H * 16));
    const img = new Float32Array(W * H * 3);
    for (let i = 0; i < W * H; i++) for (let ch = 0; ch < 3; ch++) img[3 * i + ch] = sum[4 * i + ch];
    const fin = await k.readReservoirs('final');
    const cnt = Array.from(new Uint32Array(await readBuffer(device, rig.counters, 16)));
    const ar = await k.readCounters(true);
    const rsc = [ar.rsc.accepted, ar.rsc.queued, ar.rsc.selectedShifted, ar.rsc.tFwdOk, ar.rsc.tSelP, ar.rsc.tInvOk, ar.rsc.tRefreshRecs, ar.rsc.tRefreshRays, ar.rsc.pendingLeft, ar.rsc.tPendingLeft];
    out.push(`${hashF32(new Float32Array(fin.buffer))}:${hashF32(img)}:${cnt.join(',')}:${rsc.join(',')}`);
  }
  rig.destroy();
  return out;
}

