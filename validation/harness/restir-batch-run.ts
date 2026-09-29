// Browser side of the Stage-B ReSTIR validation renders (restir-api.md §6.3/§6.4, D14; OWNER WP-E): load a scene
// package, render `batches` × `framesPerBatch` sequential independent ReSTIR frames with a rung preset through
// RestirBatchRunner (submit packing, per-batch readback of the linear accumulation) and upload batch_###.pfm +
// mean.pfm + meta.json to validation/out/<run>/ (the compare.py batch layout, identical to the PT's). Ensemble mode
// (members E > 1) writes ensemble.npz + meta.json instead (rows r = t·E + m, restir-api.md §2.10).
//
// T15/T16 (restir-api.md §6.1): NaN/Inf = 0, negatives = 0, BVH overflow = 0, RSC error counters = 0, queue overflow
// = 0 are errors of the run; meta.json records the config assertions (unbiased preset, plants only when named, internal
// scale 1, no denoiser, linear-accumulation readback, jitter iid-per-run, maxBounces, Mode A) for gate-m4.ts.
import { describeContext, type GpuContext } from '../../src/core/gpu/device.ts';
import { encodePFM } from '../../src/core/io/pfm.ts';
import { sha256Hex } from '../../src/core/io/zlib.ts';
import type { SubmitBudget } from '../../src/core/render/batch-accumulator.ts';
import { createEnvResources, destroyEnvResources, writeEnvParams } from '../../src/core/render/env-gpu.ts';
import { computeRenderOrigin, JITTER_IID } from '../../src/core/render/frame-uniforms.ts';
import type { PtEnvOptions } from '../../src/core/render/pt-kernel.ts';
import { RestirBatchRunner } from '../../src/core/render/restir/batch-runner.ts';
import { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import { writeNpz } from '../../src/core/render/restir/npz.ts';
import { RESTIR_PRESETS, restirSettings, type RestirPresetName, type RestirSettings } from '../../src/core/render/restir/presets.ts';
import { SceneGpu } from '../../src/core/render/scene-gpu.ts';
import { loadSource, stable } from './batch-run.ts';
import { uploadFile } from './export-package.ts';

export type RestirPlantName = 'no-j' | 'marginal-j';

export interface RenderRestirBatchesOptions {
  run: string;
  /** Scene package directory URL (e.g. '/validation/scenes/cornell_i_512/'). */
  package: string;
  /** Rung preset: 'initial' (3.1), 'initial-rr' (3.1b), 'offline' (3.2), 'criteria2022' (3.2, 2022 criteria). */
  preset: RestirPresetName;
  /** Frames per batch F (a batch is the mean of F sequential independent frames). */
  framesPerBatch: number;
  batches: number;
  /** Render batches [batchOffset, batchOffset + batches) of a longer sequential run (bitwise identical, measured; E = 1 only). */
  batchOffset?: number;
  seed: number;
  /** Ensemble members E (> 1: atlas of E members per frame, ensemble.npz output). Default 1. */
  members?: number;
  /** Planted controls (restir-api.md §6.3): omitted spatial J, marginal pdfs in J. */
  plant?: RestirPlantName;
  /** W × s plant (RestirParams.wScale; 1 = off). */
  wScale?: number;
  /** Override the package's max bounces (T16 requires it to equal the PT reference's). */
  maxBounces?: number;
  /** Extra setting overrides (debugging only; recorded in the config hash). */
  settings?: Partial<RestirSettings>;
  env?: PtEnvOptions;
  chromeVersion?: string;
  budget?: Partial<SubmitBudget>;
  watertight?: boolean;
  writeMean?: boolean;
}

export interface RenderRestirBatchesReport { ok: boolean; run: string; files: string[]; meta: Record<string, unknown>; errors: string[] }

/** Presets that are unbiased validation modes (T16: "validation mode is unbiased"). */
export const UNBIASED_PRESETS: readonly RestirPresetName[] = ['initial', 'initial-rr', 'offline', 'criteria2022'];

export async function renderRestirBatches(ctx: GpuContext, o: RenderRestirBatchesOptions): Promise<RenderRestirBatchesReport> {
  const t0 = performance.now();
  const errors: string[] = [];
  if (!(o.framesPerBatch >= 1 && o.batches >= 1)) throw new Error('framesPerBatch and batches must be ≥ 1');
  if (!(o.preset in RESTIR_PRESETS)) throw new Error(`unknown preset ${o.preset}`);
  const E = o.members ?? 1;
  const off = o.batchOffset ?? 0;
  if (off && E > 1) throw new Error('batchOffset is for sequential runs only (ensemble rows are frame-indexed in one npz)');
  const { device, features, wgslLanguageFeatures } = ctx;
  const src = await loadSource({ package: o.package });
  if (!src.size) throw new Error(`${o.package}: package without a render size`);
  const [W, H] = src.size;   // internal scale 1 (T16): always the package's resolution
  const tLoad = performance.now();
  const origin = computeRenderOrigin(src.scene.bounds);
  const watertight = o.watertight ?? true;
  const gpu = await SceneGpu.create(device, src.scene, origin, { textureMode: 'validation', watertight, features, wgslLanguageFeatures });
  const env = await createEnvResources(device, src.scene.env);
  if (src.frame?.env) writeEnvParams(device, env, src.frame.env);
  const pkgEnvSampling = (src.source.envSampling as string | undefined) ?? 'AUTOMATIC';
  const envOpts: PtEnvOptions = { ...o.env, nee: o.env?.nee ?? pkgEnvSampling !== 'NONE' };
  const pkgBounces = typeof src.source.maxBounces === 'number' ? src.source.maxBounces : undefined;
  const maxBounces = o.maxBounces ?? pkgBounces ?? 3;
  const lightMode = (src.source.lightMode as string | undefined) ?? 'A';
  if (lightMode !== 'A') throw new Error(`${o.package}: light mode ${lightMode}; M4 ReSTIR is Mode A only (restir-api.md D1)`);
  const plant = o.plant || (o.wScale !== undefined && o.wScale !== 1)
    ? { ...(o.plant === 'no-j' ? { noJ: true } : {}), ...(o.plant === 'marginal-j' ? { marginalJ: true } : {}), ...(o.wScale !== undefined && o.wScale !== 1 ? { wScale: o.wScale } : {}) }
    : undefined;
  const settings = restirSettings(o.preset, { maxBounces, ...o.settings, ...(plant ? { plant } : {}) });

  const kernel = await RestirKernel.create(device, gpu, env, { settings, lightMode: 'A', env: envOpts, features, wgslLanguageFeatures });
  kernel.setView({ camera: { camToWorld: src.camera.camToWorld as number[], yfov: src.camera.yfov }, width: W, height: H, runSeed: o.seed >>> 0, members: E, jitterMode: JITTER_IID });
  await kernel.prepare();
  const runner = new RestirBatchRunner(kernel, { budget: o.budget, framesPerBatch: o.framesPerBatch, batches: o.batches, runSeed: o.seed >>> 0 });
  const envInfo = env.present ? { nee: kernel.lights.summary().env, pEnv: kernel.lights.state.envPmf(), strength: env.params.strength, tint: env.params.tint, rotationZ: env.params.rotationZ } : undefined;
  const tSetup = performance.now();

  const files: string[] = [];
  const batchMs: number[] = [];
  let spatialRounds = 0;
  try {
    for (let b = off; b < off + o.batches; b++) {
      const r = await runner.runBatch(o.framesPerBatch, b);
      batchMs.push(r.wallMs);
      spatialRounds = kernel.lastRounds;
      if (E === 1) {
        const name = `batch_${String(b).padStart(3, '0')}.pfm`;
        await uploadFile(o.run, name, encodePFM({ width: W, height: H, channels: 3, data: r.mean }));
        files.push(name);
      }
    }
    if (E === 1 && (o.writeMean ?? true)) {
      await uploadFile(o.run, 'mean.pfm', encodePFM({ width: W, height: H, channels: 3, data: runner.acc.overallMean() }));
      files.push('mean.pfm');
    }
    if (E > 1) {
      const c = runner.ensemble!;
      const R = c.count;
      const arrays = [
        ...c.levels.map((lv, i) => ({ name: `tiles${lv.l}`, shape: [R, lv.th, lv.tw, 3], dtype: '<f8' as const, data: c.tiles[i].subarray(0, R * lv.th * lv.tw * 3) })),
        { name: 'global', shape: [R, 3], dtype: '<f8' as const, data: c.global.subarray(0, R * 3) },
        { name: 'pixel_sum', shape: [H, W, 3], dtype: '<f8' as const, data: c.pixelSum },
        { name: 'pixel_sumsq', shape: [H, W, 3], dtype: '<f8' as const, data: c.pixelSumSq },
        { name: 'count', shape: [], dtype: '<i8' as const, data: BigInt64Array.from([BigInt(R)]) },
        { name: 'height', shape: [], dtype: '<i8' as const, data: BigInt64Array.from([BigInt(H)]) },
        { name: 'width', shape: [], dtype: '<i8' as const, data: BigInt64Array.from([BigInt(W)]) },
      ];
      await uploadFile(o.run, 'ensemble.npz', writeNpz(arrays));
      files.push('ensemble.npz');
    }
  } finally {
    runner.destroy(); kernel.destroy(); gpu.destroy(); destroyEnvResources(env);
  }
  errors.push(...runner.invariantErrors());
  const T = runner.totals;
  // BASE_JDEN_INVALID / nCand ≤ 1e-5 (§2.6); nCand is per reservoir, so the proxy denominator is pixels × frames × S
  // (a lower bound of the candidates on hit pixels when every tree yields ≥ 1 candidate).
  const candProxy = W * H * E * o.framesPerBatch * o.batches * settings.trees;
  const jdenRate = T.rsc.baseJdenInvalid / Math.max(1, candProxy);
  if (jdenRate > 1e-5) errors.push(`BASE_JDEN_INVALID ${T.rsc.baseJdenInvalid} / ${candProxy} candidates (proxy) = ${jdenRate.toExponential(2)} > 1e-5`);
  // rounds > 0 presets must actually run the spatial stage (A6: all rounds or none); a stub stage is recorded.
  if (settings.rounds > 0 && spatialRounds !== settings.rounds) {
    console.warn(`[restir-batch-run] spatial stage executed ${spatialRounds}/${settings.rounds} rounds (stub stage: canonical-only output)`);
  }

  const unbiased = UNBIASED_PRESETS.includes(o.preset) && !plant;
  const config = {
    kernel: 'restir', preset: o.preset, framesPerBatch: o.framesPerBatch, members: E, width: W, height: H, jitter: 'iid-per-run', filter: 'box-1px',
    scene: src.source.packageSha256, frame: null, textureMode: 'validation', intersector: watertight ? 'woop-watertight' : 'moller-trumbore',
    maxBounces, lightMode: 'A', settings, env: envInfo ? { nee: envInfo.nee } : 'none', plant: plant ?? null,
  };
  const configHash = await sha256Hex(new TextEncoder().encode(stable(config)));
  const info = describeContext(ctx) as { vendor?: string; architecture?: string; description?: string };
  const ua = navigator.userAgent;
  const seeds = E === 1
    ? Array.from({ length: o.batches }, (_, b) => `${o.seed}:${off + b}`)
    : runner.ensemble!.seeds;
  const meta = {
    kind: E === 1 ? 'batches' : 'ensemble', kernel: 'restir', seed: o.seed, sppPerBatch: o.framesPerBatch, framesPerBatch: o.framesPerBatch, batches: o.batches,
    members: E, width: W, height: H, configHash, config, ...(off ? { batchOffset: off } : {}),
    // T16 config assertions (gate-m4.ts checks them against the PT reference's meta.json)
    t16: {
      validationModeUnbiased: unbiased, plantsNamed: plant ? Object.keys(plant) : [], internalScale: 1, denoiser: 'none', upscaler: 'none',
      readback: E === 1 ? 'linear accumulation buffer (f32 Σ L per frame) / frames, f64 division'
        : 'linear rsFrame radiance reduced by ensStats / ensPixel (f32 sums per frame, f64 host rows); no accumulation buffer', jitterMode: 'iid-per-run', maxBounces, lightMode: 'A',
      spatialRoundsExecuted: spatialRounds, spatialRoundsRequested: settings.rounds,
    },
    restir: {
      settings, counters: T.rsc, codes: T.codes, fr: T.fr, queueOverflow: T.queueOverflow, queueMaxCounter: T.queueMaxCounter,
      baseJdenInvalidRate: jdenRate, chunking: T.chunking, rates: T.rates, probeMs: T.probeMs,
      ...(envInfo ? { env: envInfo } : {}),
    },
    chromeVersion: o.chromeVersion ?? /(?:Headless)?Chrome\/([\d.]+)/.exec(ua)?.[1] ?? 'unknown',
    userAgent: ua,
    adapterInfo: { vendor: info.vendor, architecture: info.architecture, description: info.description },
    seeds,
    sampleIndexing: E === 1
      ? 'batch b = run-global frames t in [b*F, (b+1)*F), member 0; frame key = pcg3d(runSeed, t, pixelIndex) (restir-api.md §5)'
      : 'row r = t*E + m: frame t, atlas member m; frame key = pcg3d(runSeed ^ m*0x9e3779b9, t, localIdx) (restir-api.md §5, §2.10)',
    image: E === 1 ? { format: 'PFM RGB float32', rowOrder: 'PFM bottom-to-top (decodes to row 0 = top)', value: 'batch mean radiance' } : { format: 'ensemble.npz (compare.py README)' },
    scene: { ...src.source, triangles: src.scene.geometry.indices.length / 3, env: !!src.scene.env, origin },
    counters: T.counters, submits: T.submits,
    timings: { loadMs: tLoad - t0, setupMs: tSetup - tLoad, batchMs, totalMs: performance.now() - t0 },
    files, ok: errors.length === 0, errors,
    createdAt: new Date().toISOString(),
  };
  await uploadFile(o.run, 'meta.json', new TextEncoder().encode(JSON.stringify(meta, null, 1)));
  return { ok: errors.length === 0, run: o.run, files: [...files, 'meta.json'], meta, errors };
}
