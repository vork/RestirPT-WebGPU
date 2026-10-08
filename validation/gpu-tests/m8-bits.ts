// U-M8-BITS rig (docs/decisions/m8-perf.md §4, M8 Gate 0): every M8 optimisation that must not change results is
// checked against hashes recorded on the pre-M8 code (eda22be; recorded with VITE_M8_BITS_RECORD=1 before any ReSTIR /
// PT shader changed — U-M7-BITS showed the composed WGSL of every validation pipeline unchanged at that point).
//   ReSTIR  short temporal chains (advance + frameUnits per frame) of the shipped interactive configuration (every M6
//           feature incl. the duplication map, light mode B, RR, boost) and of the validation presets full-m6 / offline-m6
//           in light mode B, on the all-lights fixture (point / spot / rect / disk / sun / emissive quad / env, the
//           roughness-0 mirror), the normal-mapped smooth package, a glass package and the alpha-foliage package; per
//           frame: final reservoirs, the frame's mean image, the finalize counters and the arena counters.
//   PT      the batch kernel (validation configuration) on the same scenes: hash of the mean image and the counters.
import { readBuffer } from '../../src/core/gpu/readback.ts';
import type { LightData, SceneData } from '../../src/core/scene/types.ts';
import type { RestirPresetName, RestirSettings } from '../../src/core/render/restir/presets.ts';
import type { LightMode } from '../../src/core/render/lights-gpu.ts';
import { fetchScenePackage } from '../../src/core/scene/scene-package.ts';
import { allLightsScene, boxCamera, gpuScene, hashF32, restirRig } from './restir-fixtures.ts';
import { PtKernel } from '../../src/core/render/pt-kernel.ts';
import { BatchAccumulator } from '../../src/core/render/batch-accumulator.ts';
import { JITTER_IID } from '../../src/core/render/frame-uniforms.ts';

export type M8BitsScene = 'all' | 'nm_smooth' | 'glass' | 'alpha';
export interface M8BitsCase {
  name: string; scene: M8BitsScene; preset: RestirPresetName; settings: Partial<RestirSettings>; lightMode: LightMode; frames: number;
  /** Per frame from `from` on: move light index 0 by dx (m) and yaw the camera by dyaw (rad). */
  motion?: { from: number; dx: number; dyaw: number };
}

const INTERACTIVE: Partial<RestirSettings> = { maxBounces: 3 };
export const M8_BITS_CASES: M8BitsCase[] = [
  { name: 'all-interactive-B', scene: 'all', preset: 'interactive', settings: INTERACTIVE, lightMode: 'B', frames: 5, motion: { from: 2, dx: 0.03, dyaw: 0.01 } },
  { name: 'all-full-m6-B', scene: 'all', preset: 'full-m6', settings: { maxBounces: 3 }, lightMode: 'B', frames: 4, motion: { from: 2, dx: 0.04, dyaw: 0.0 } },
  { name: 'all-offline-m6-B', scene: 'all', preset: 'offline-m6', settings: { maxBounces: 3, trees: 4 }, lightMode: 'B', frames: 1 },
  { name: 'nm-interactive-B', scene: 'nm_smooth', preset: 'interactive', settings: INTERACTIVE, lightMode: 'B', frames: 4, motion: { from: 2, dx: 0.02, dyaw: 0.01 } },
  { name: 'glass-interactive-B', scene: 'glass', preset: 'interactive', settings: { maxBounces: 4 }, lightMode: 'B', frames: 4, motion: { from: 2, dx: 0.0, dyaw: 0.01 } },
  { name: 'alpha-full-m6-A', scene: 'alpha', preset: 'full-m6', settings: { maxBounces: 3 }, lightMode: 'A', frames: 3 },
];

const PKG: Record<Exclude<M8BitsScene, 'all'>, string> = {
  nm_smooth: '/validation/out/m7/scenes/m7_nm_smooth_256/',
  glass: '/validation/scenes/g8_cornell_glass_512/',
  alpha: '/validation/scenes/m5s_alpha_foliage/',
};

export async function m8BitsScene(s: M8BitsScene): Promise<{ scene: SceneData; cam: { camToWorld: number[]; yfov: number } }> {
  if (s === 'all') return { scene: allLightsScene(), cam: boxCamera() };
  const p = await fetchScenePackage(PKG[s]);
  return { scene: p.scene, cam: { camToWorld: Array.from(p.camera.matrix), yfov: p.camera.yfov } };
}

function yawed(cam: { camToWorld: number[]; yfov: number }, a: number): { camToWorld: number[]; yfov: number } {
  if (a === 0) return cam;
  const m = cam.camToWorld.slice();
  const c = Math.cos(a), s = Math.sin(a);
  for (const col of [0, 4, 8]) { const x = m[col], z = m[col + 2]; m[col] = c * x + s * z; m[col + 2] = -s * x + c * z; }
  return { camToWorld: m, yfov: cam.yfov };
}

function moved(lights: LightData[], dx: number): LightData[] {
  return lights.map((l, i) => { if (i !== 0 || dx === 0) return l; const m = new Float32Array(l.matrix); m[12] += dx; return { ...l, matrix: m }; });
}

/** Per frame: `final reservoirs:image:finalize counters:arena counters` hashes. */
export async function m8BitsCase(c: M8BitsCase): Promise<string[]> {
  const { scene, cam: cam0 } = await m8BitsScene(c.scene);
  const W = 64, H = 64;
  const rig = await restirRig(scene, W, H, { preset: c.preset, settings: c.settings, seed: 31, lightMode: c.lightMode, cam: cam0 });
  const k = rig.kernel, device = rig.g.device;
  await k.prepare();
  const out: string[] = [];
  let lights = scene.lights;
  let cam = cam0;
  for (let f = 0; f < c.frames; f++) {
    const t = 3 + f;
    if (c.motion && f >= c.motion.from) { lights = moved(lights, c.motion.dx); cam = yawed(cam, c.motion.dyaw); }
    const clear = device.createCommandEncoder();
    clear.clearBuffer(rig.accum); clear.clearBuffer(rig.counters);
    device.queue.submit([clear.finish()]);
    if (k.settings.temporal) k.advance({ t, camera: cam, lights });
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

/** PT batch kernel (validation configuration, Woop, BVH2) on the case scene: `image:counters`. */
export async function m8PtBits(s: M8BitsScene, lightMode: LightMode): Promise<string> {
  const { scene, cam } = await m8BitsScene(s);
  const g = await gpuScene(scene);
  const kernel = await PtKernel.create(g.device, g.gpu, g.env, { features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures, maxBounces: s === 'glass' ? 4 : 3, lightMode });
  kernel.setView({ camera: cam, width: 64, height: 64, runSeed: 17, jitterMode: JITTER_IID });
  const acc = new BatchAccumulator(g.device, 64, 64, { maxSamplesPerDispatch: 1 });
  const b = await acc.runBatch((enc, d, a, c) => kernel.encode(enc, d, a, c), 4, 0);
  kernel.destroy(); acc.destroy(); g.destroy();
  return `${hashF32(b.mean)}:${b.counters.join(',')}`;
}
