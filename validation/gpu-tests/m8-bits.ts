// U-M8-BITS rig (docs/decisions/m8-perf.md §4, M8 Gate 0): every M8 optimisation that must not change results is
// checked against hashes recorded on the pre-M8 code (eda22be; recorded with VITE_M8_BITS_RECORD=1 before any ReSTIR /
// PT shader changed — U-M7-BITS showed the composed WGSL of every validation pipeline unchanged at that point).
//   ReSTIR  short temporal chains (advance + frameUnits per frame) of the shipped interactive configuration (every M6
//           feature incl. the duplication map, light mode B, RR, boost) and of the validation presets full-m6 / offline-m6
//           in light mode B, on the all-lights fixture (point / spot / rect / disk / sun / emissive quad / env, the
//           roughness-0 mirror), the normal-mapped smooth package, a glass package and the alpha-foliage package; per
//           frame: final reservoirs, the frame's mean image, the finalize counters and the arena counters.
//   PT      the batch kernel (validation configuration) on the same scenes: hash of the mean image and the counters.
// perf2 WP-0 (docs/decisions/perf2-api.md): the interactive knobs are pinned (INTERACTIVE_PINNED); PERF2_BITS_CASES add
// CWBVH-forced variants (the app's MT intersection for the interactive cases, Woop for the validation preset) and a
// Sponza-lite case (real Sponza + HDRI at 160×90, CWBVH, MT, textures, normal maps, alpha; the interactive kernel's P-4 /
// SoA options); every case also yields a P9-excluded hash (reservoir plane P9 = words 36–39 zeroed: nCand, selId,
// kMargin, endpointId) for packages whose documented exception is the diagnostic plane (RS_NO_DIAG).
import { readBuffer } from '../../src/core/gpu/readback.ts';
import type { LightData, SceneData } from '../../src/core/scene/types.ts';
import { INTERACTIVE_PINNED, type RestirPresetName, type RestirSettings } from '../../src/core/render/restir/presets.ts';
import { RES_PLANES } from '../../src/core/render/restir/layout.ts';
import type { PerfFlagsInput } from '../../src/core/render/restir/perf-flags.ts';
import type { LightMode } from '../../src/core/render/lights-gpu.ts';
import { fetchScenePackage } from '../../src/core/scene/scene-package.ts';
import { allLightsScene, boxCamera, gpuScene, hashF32, restirRig, type GpuSceneOptions } from './restir-fixtures.ts';
import { loadGltfPerfScene } from '../harness/perf-run.ts';
import { PtKernel } from '../../src/core/render/pt-kernel.ts';
import { BatchAccumulator } from '../../src/core/render/batch-accumulator.ts';
import { JITTER_IID } from '../../src/core/render/frame-uniforms.ts';

export type M8BitsScene = 'all' | 'noarea' | 'nm_smooth' | 'glass' | 'alpha' | 'sponza_lite';
export interface M8BitsCase {
  name: string; scene: M8BitsScene; preset: RestirPresetName; settings: Partial<RestirSettings>; lightMode: LightMode; frames: number;
  /** Per frame from `from` on: move light index 0 by dx (m) and yaw the camera by dyaw (rad). */
  motion?: { from: number; dx: number; dyaw: number };
  /** perf2: scene upload (default BVH2 + Woop, exact texels), atlas size (default 64²), interactive kernel options. */
  gpu?: GpuSceneOptions;
  size?: [number, number];
  kernel?: { resLayout?: 'aos' | 'soa'; modeBNeedsAreaLights?: boolean };
}

/** perf2 WP-0: interactive knobs pinned (the preset's values: the goldens are unchanged). */
const INTERACTIVE: Partial<RestirSettings> = { maxBounces: 3, ...INTERACTIVE_PINNED };
export const M8_BITS_CASES: M8BitsCase[] = [
  { name: 'all-interactive-B', scene: 'all', preset: 'interactive', settings: INTERACTIVE, lightMode: 'B', frames: 5, motion: { from: 2, dx: 0.03, dyaw: 0.01 } },
  { name: 'all-full-m6-B', scene: 'all', preset: 'full-m6', settings: { maxBounces: 3 }, lightMode: 'B', frames: 4, motion: { from: 2, dx: 0.04, dyaw: 0.0 } },
  { name: 'all-offline-m6-B', scene: 'all', preset: 'offline-m6', settings: { maxBounces: 3, trees: 4 }, lightMode: 'B', frames: 1 },
  { name: 'nm-interactive-B', scene: 'nm_smooth', preset: 'interactive', settings: INTERACTIVE, lightMode: 'B', frames: 4, motion: { from: 2, dx: 0.02, dyaw: 0.01 } },
  { name: 'glass-interactive-B', scene: 'glass', preset: 'interactive', settings: { maxBounces: 4, ...INTERACTIVE_PINNED }, lightMode: 'B', frames: 4, motion: { from: 2, dx: 0.0, dyaw: 0.01 } },
  { name: 'alpha-full-m6-A', scene: 'alpha', preset: 'full-m6', settings: { maxBounces: 3 }, lightMode: 'A', frames: 3 },
];

/** perf2 WP-0 (perf2-plan.md §2 WP-0 step 3): goldens recorded on the unmodified perf2 text (6d46432). */
const CW_MT: GpuSceneOptions = { bvhKind: 'cwbvh', watertight: false };
export const PERF2_BITS_CASES: M8BitsCase[] = [
  { name: 'all-interactive-B-cwbvh', scene: 'all', preset: 'interactive', settings: INTERACTIVE, lightMode: 'B', frames: 5, motion: { from: 2, dx: 0.03, dyaw: 0.01 }, gpu: CW_MT },
  { name: 'nm-interactive-B-cwbvh', scene: 'nm_smooth', preset: 'interactive', settings: INTERACTIVE, lightMode: 'B', frames: 4, motion: { from: 2, dx: 0.02, dyaw: 0.01 }, gpu: CW_MT },
  { name: 'alpha-full-m6-A-cwbvh', scene: 'alpha', preset: 'full-m6', settings: { maxBounces: 3 }, lightMode: 'A', frames: 3, gpu: { bvhKind: 'cwbvh', watertight: true } },
  // the app's Sponza configuration at 160×90: interactive preset (pinned), light mode B compiled as Mode A (P-4: no
  // rect / disk light), SoA reservoirs, CWBVH + MT, exact texels; the perf auto setup (warm point light) + the HDRI
  { name: 'sponza-lite-interactive-B', scene: 'sponza_lite', preset: 'interactive', settings: INTERACTIVE, lightMode: 'B', frames: 3, motion: { from: 2, dx: 0.05, dyaw: 0.005 },
    gpu: CW_MT, size: [160, 90], kernel: { resLayout: 'soa', modeBNeedsAreaLights: true } },
];

const SPONZA = '/validation/assets/downloaded/sponza/Sponza.gltf';
const SPONZA_HDRI = '/validation/assets/downloaded/hdri/kloofendal_48d_partly_cloudy_puresky_1k.hdr';

const PKG: Record<Exclude<M8BitsScene, 'all' | 'noarea' | 'sponza_lite'>, string> = {
  nm_smooth: '/validation/out/m7/scenes/m7_nm_smooth_256/',
  glass: '/validation/scenes/g8_cornell_glass_512/',
  alpha: '/validation/scenes/m5s_alpha_foliage/',
};

export async function m8BitsScene(s: M8BitsScene): Promise<{ scene: SceneData; cam: { camToWorld: number[]; yfov: number } }> {
  if (s === 'all') return { scene: allLightsScene(), cam: boxCamera() };
  if (s === 'noarea') {   // every endpoint type but the crossable rect / disk lights (point, spot, sun, emissive quad, env)
    const sc = allLightsScene();
    return { scene: { ...sc, lights: sc.lights.filter((l) => l.type !== 'rect' && l.type !== 'disk') }, cam: boxCamera() };
  }
  if (s === 'sponza_lite') {
    const { scene, camera } = await loadGltfPerfScene(SPONZA, { env: SPONZA_HDRI });
    return { scene, cam: { camToWorld: Array.from(camera.camToWorld), yfov: camera.yfov } };
  }
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

/** Reservoir words of plane P9 (nCand, selId, kMargin, endpointId) zeroed: the P9-excluded hash mode (AoS input). */
export function zeroP9(aos: Uint32Array): Uint32Array {
  const out = aos.slice();
  const rec = RES_PLANES * 4;
  for (let i = 0; i < out.length; i += rec) out.fill(0, i + 36, i + 40);
  return out;
}

export interface M8BitsOptions {
  /** Reservoir layout (default: the case's kernel option, else the validation AoS). */
  resLayout?: 'aos' | 'soa';
  /** perf2: perf flags of the kernel (default: restirRig's, i.e. VITE_PERF_FLAGS). */
  perfFlags?: PerfFlagsInput;
}

/** Per frame: `final reservoirs:image:finalize counters:arena counters` hashes. */
export async function m8BitsCase(c: M8BitsCase, opts: M8BitsOptions = {}): Promise<string[]> {
  return (await m8BitsCaseHashes(c, opts)).full;
}

/** perf2: both hash modes of one run — `full` (as m8BitsCase) and `noP9` (reservoir plane P9 zeroed before hashing). */
export async function m8BitsCaseHashes(c: M8BitsCase, opts: M8BitsOptions = {}): Promise<{ full: string[]; noP9: string[] }> {
  const { scene, cam: cam0 } = await m8BitsScene(c.scene);
  const [W, H] = c.size ?? [64, 64];
  const rig = await restirRig(scene, W, H, {
    preset: c.preset, settings: c.settings, seed: 31, lightMode: c.lightMode, cam: cam0, resLayout: opts.resLayout ?? c.kernel?.resLayout,
    modeBNeedsAreaLights: c.kernel?.modeBNeedsAreaLights, gpu: c.gpu, ...(opts.perfFlags !== undefined ? { perfFlags: opts.perfFlags } : {}),
  });
  if ((c.gpu?.bvhKind ?? 'bvh2') !== rig.g.gpu.bvhKind) throw new Error(`${c.name}: scene uploaded as ${rig.g.gpu.bvhKind}, want ${c.gpu?.bvhKind}`);
  const k = rig.kernel, device = rig.g.device;
  await k.prepare();
  const out: string[] = [];
  const outNoP9: string[] = [];
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
    const tail = `${hashF32(img)}:${cnt.join(',')}:${rsc.join(',')}`;
    out.push(`${hashF32(new Float32Array(fin.buffer))}:${tail}`);
    const z = zeroP9(fin);
    outNoP9.push(`${hashF32(new Float32Array(z.buffer))}:${tail}`);
  }
  rig.destroy();
  return { full: out, noP9: outNoP9 };
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
