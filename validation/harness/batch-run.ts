// Browser side of batch validation renders (plan §5 M2, §7.3): load a scene package (or a .glb/.usd URL), render
// `batches` × `spp` with a validation kernel through BatchAccumulator (submit budget, per-batch readback), and upload
// batch_###.pfm + meta.json to validation/out/<run>/ (compare.py input layout).
import { describeContext, type GpuContext } from '../../src/core/gpu/device.ts';
import { encodePFM } from '../../src/core/io/pfm.ts';
import { sha256Hex } from '../../src/core/io/zlib.ts';
import { BatchAccumulator, type SubmitBudget } from '../../src/core/render/batch-accumulator.ts';
import { EMISSION_COUNTERS, EmissionKernel } from '../../src/core/render/emission-kernel.ts';
import { PT_COUNTERS, PtKernel, type PtEnvOptions, type PtPlant, type PtTechnique } from '../../src/core/render/pt-kernel.ts';
import { createEnvResources, destroyEnvResources, writeEnvParams } from '../../src/core/render/env-gpu.ts';
import { computeRenderOrigin } from '../../src/core/render/frame-uniforms.ts';
import { SceneGpu } from '../../src/core/render/scene-gpu.ts';
import { parseLightMode } from '../../src/core/render/lights-gpu.ts';
import { loadScene as loadGltfScene } from '../../src/core/scene/load-scene.ts';
import { fetchScenePackage } from '../../src/core/scene/scene-package.ts';
import type { SceneData } from '../../src/core/scene/types.ts';
import { isUsdName, loadUsd } from '../../src/core/scene/usd/load-usd.ts';
import { packageSha256, uploadFile } from './export-package.ts';

/** 'emission': M2 length-1 kernel; 'pt': the reference path tracer (Mode A; env NEE + MIS since M3c). */
export type ValidationKernel = 'emission' | 'pt';

export interface RenderBatchesOptions {
  /** Output directory name under validation/out/ (safe path component). */
  run: string;
  /** Scene package directory URL (e.g. '/validation/scenes/C0b/'); render size defaults to the package's. */
  package?: string;
  /** Alternatively a .glb/.gltf/.usd* URL (first file camera). */
  sceneUrl?: string;
  kernel: ValidationKernel;
  spp: number;
  batches: number;
  width?: number;
  height?: number;
  seed: number;
  /** Browser version from the driver (Playwright browser.version()); falls back to the user agent. */
  chromeVersion?: string;
  budget?: Partial<SubmitBudget>;
  /** Package frame index (scene.json "frames": camera / env / light overrides), e.g. the C0p γ variants. */
  frame?: number;
  /** Woop watertight intersection (default true: Möller–Trumbore leaks ~1e-4 of the rays through shared edges). */
  watertight?: boolean;
  /** Also upload mean.pfm (mean over all batches; compare.py ignores it, marker_check.py --image uses it). Default true. */
  writeMean?: boolean;
  /** kernel 'pt': Cycles max_bounces (default: the package's render.maxBounces, else 3). */
  maxBounces?: number;
  /** kernel 'pt': Russian roulette (default off; RR is unbiased but not part of Cycles-parity semantics). */
  rr?: boolean;
  /** kernel 'pt': estimator (default 'mis'; 'nee'/'bsdf' are the T9d single-technique estimators). */
  technique?: PtTechnique;
  /** kernel 'pt': Gate-1 planted bias (emitter scale / path drop); recorded in meta.json and the config hash. */
  plant?: PtPlant;
  /** kernel 'pt': light mode override ('A' | 'B' | 'A′'; default the package's lightMode). */
  lightMode?: string;
  /**
   * kernel 'pt' (M3c): env sampling options. `nee` defaults to the package's env.sampling (NONE → off). `strengthScale`
   * is the planted bias "env strength ×s" (applied to the rendered env only). Recorded in meta.json and the config hash.
   */
  env?: PtEnvOptions & { strengthScale?: number };
}

export interface RenderBatchesReport {
  ok: boolean;
  run: string;
  files: string[];
  meta: Record<string, unknown>;
  errors: string[];
}

export const stable = (v: unknown): string => JSON.stringify(v, (_k, x) => (x && typeof x === 'object' && !Array.isArray(x)
  ? Object.fromEntries(Object.entries(x as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : 1))) : x));

interface FrameOverride { env?: { rotationZ?: number; strength?: number; tint?: [number, number, number] } }

export async function loadSource(o: Pick<RenderBatchesOptions, 'package' | 'sceneUrl' | 'frame'>): Promise<{ scene: SceneData; camera: { camToWorld: ArrayLike<number>; yfov: number }; size?: [number, number]; source: Record<string, unknown>; frame?: FrameOverride }> {
  if (o.package) {
    const p = await fetchScenePackage(o.package);
    const hash = await packageSha256(p.files); // exactly the files build_scene.py hashes
    let camera = { camToWorld: p.camera.matrix as ArrayLike<number>, yfov: p.camera.yfov };
    let scene = p.scene;
    let frame: FrameOverride | undefined;
    if (o.frame !== undefined) {
      const f = p.frames?.find((x) => x.frame === o.frame);
      if (!f) throw new Error(`${o.package}: no frame ${o.frame} in scene.json frames`);
      if (f.camera) camera = { camToWorld: Float64Array.from(f.camera.matrix), yfov: f.camera.yfov };
      if (f.lights) {
        scene = { ...scene, lights: scene.lights.map((l) => {
          const ov = f.lights![String(l.id)];
          return ov ? { ...l, ...(ov.matrix ? { matrix: new Float32Array(ov.matrix) } : {}), ...(ov.power !== undefined ? { power: ov.power } : {}) } : l;
        }) };
      }
      frame = { env: f.env };
    }
    return {
      scene, camera, size: [p.render.width, p.render.height], frame,
      source: { kind: 'package', url: o.package, name: p.json.name, packageSha256: hash.sha256, files: hash.files, lightMode: p.lightMode, maxBounces: p.render.maxBounces, frame: o.frame ?? null,
        ...(p.json.env ? { envSampling: p.json.env.sampling } : {}) },
    };
  }
  if (!o.sceneUrl) throw new Error('renderBatches: need package or sceneUrl');
  const bytes = new Uint8Array(await (await fetch(o.sceneUrl)).arrayBuffer());
  const fileSha256 = await sha256Hex(bytes);
  const scene = isUsdName(o.sceneUrl) ? (await loadUsd(o.sceneUrl)).scene : (await loadGltfScene(o.sceneUrl)).scene;
  const cam = scene.cameras[0];
  if (!cam) throw new Error(`${o.sceneUrl}: no camera in the file`);
  return { scene, camera: { camToWorld: cam.matrix, yfov: cam.yfov }, source: { kind: 'file', url: o.sceneUrl, name: scene.name, fileSha256 } };
}

export async function renderBatches(ctx: GpuContext, o: RenderBatchesOptions): Promise<RenderBatchesReport> {
  const t0 = performance.now();
  const errors: string[] = [];
  if (o.kernel !== 'emission' && o.kernel !== 'pt') throw new Error(`unknown kernel ${o.kernel}`);
  if (!(o.spp >= 1 && o.batches >= 1)) throw new Error('spp and batches must be ≥ 1');
  const { device, features, wgslLanguageFeatures } = ctx;
  const src = await loadSource(o);
  const W = o.width ?? src.size?.[0] ?? 512, H = o.height ?? src.size?.[1] ?? 512;
  const tLoad = performance.now();
  const origin = computeRenderOrigin(src.scene.bounds);
  const watertight = o.watertight ?? true;
  const gpu = await SceneGpu.create(device, src.scene, origin, { textureMode: 'validation', watertight, features, wgslLanguageFeatures });
  const env = await createEnvResources(device, src.scene.env);
  if (src.frame?.env) writeEnvParams(device, env, src.frame.env);
  if (o.env?.strengthScale !== undefined) writeEnvParams(device, env, { strength: env.params.strength * o.env.strengthScale });
  const pkgEnvSampling = (src.source.envSampling as string | undefined) ?? 'AUTOMATIC';
  const envOpts: PtEnvOptions = { ...o.env, nee: o.env?.nee ?? pkgEnvSampling !== 'NONE' };
  const view = { camera: { camToWorld: src.camera.camToWorld as number[], yfov: src.camera.yfov }, width: W, height: H, runSeed: o.seed };
  const pkgBounces = typeof src.source.maxBounces === 'number' ? src.source.maxBounces : undefined;
  const maxBounces = o.maxBounces ?? pkgBounces ?? 3;
  // plan §1.4 light modes: the package's mode unless overridden (U9: the same package rendered in A and B)
  const lightMode = parseLightMode(o.lightMode ?? (src.source.lightMode as string | undefined), 'A');
  const kernel = o.kernel === 'pt'
    ? await PtKernel.create(device, gpu, env, { features, wgslLanguageFeatures, maxBounces, rr: o.rr ?? false, technique: o.technique ?? 'mis', lightMode, plant: o.plant, env: envOpts })
    : await EmissionKernel.create(device, gpu, env, { features, wgslLanguageFeatures });
  kernel.setView(view);
  const envTable = kernel instanceof PtKernel ? kernel.lights.state.env : undefined;
  const envInfo = kernel instanceof PtKernel && env.present ? {
    nee: kernel.lights.summary().env, pEnv: kernel.lights.state.envPmf(), strength: env.params.strength, tint: env.params.tint, rotationZ: env.params.rotationZ,
    ...(envTable ? { importance: { Wm: envTable.Wm, Hm: envTable.Hm, cap: envTable.options.cap, floors: envTable.options.floors, buildMs: Math.round(envTable.buildMs) } } : {}),
    ...(o.env?.misPower ? { misPower: true } : {}), ...(o.env?.plant ? { plant: o.env.plant } : {}), ...(o.env?.strengthScale !== undefined ? { strengthScale: o.env.strengthScale } : {}),
  } : undefined;
  const ptInfo = kernel instanceof PtKernel
    ? { maxBounces, rr: kernel.settings.rr, technique: kernel.settings.technique, lightMode, lights: kernel.lights.summary(), ...(o.plant ? { plant: o.plant } : {}), ...(envInfo ? { env: envInfo } : {}) }
    : undefined;
  const acc = new BatchAccumulator(device, W, H, o.budget);
  const tSetup = performance.now();

  const files: string[] = [];
  const counters = { nonFinite: 0, bvhOverflow: 0, bvhItercap: 0, negative: 0 };
  const submits = { total: 0, maxMs: 0, overBudget: 0, overHardCap: 0, perBatch: [] as number[] };
  const batchMs: number[] = [];
  try {
    for (let b = 0; b < o.batches; b++) {
      const r = await acc.runBatch((enc, d, a, c) => kernel.encode(enc, d, a, c), o.spp, b);
      counters.nonFinite += r.counters[EMISSION_COUNTERS.nonFinite];
      counters.bvhOverflow += r.counters[EMISSION_COUNTERS.bvhOverflow];
      counters.bvhItercap += r.counters[EMISSION_COUNTERS.bvhItercap];
      if (o.kernel === 'pt') counters.negative += r.counters[PT_COUNTERS.negative];
      submits.total += r.submits; submits.maxMs = Math.max(submits.maxMs, r.maxSubmitMs);
      submits.overBudget += r.overBudget; submits.overHardCap += r.overHardCap; submits.perBatch.push(r.submits);
      batchMs.push(r.wallMs);
      const name = `batch_${String(b).padStart(3, '0')}.pfm`;
      await uploadFile(o.run, name, encodePFM({ width: W, height: H, channels: 3, data: r.mean }));
      files.push(name);
    }
    if (o.writeMean ?? true) {
      await uploadFile(o.run, 'mean.pfm', encodePFM({ width: W, height: H, channels: 3, data: acc.overallMean() }));
      files.push('mean.pfm');
    }
  } finally {
    kernel.destroy(); acc.destroy(); gpu.destroy(); destroyEnvResources(env);
  }
  if (counters.nonFinite) errors.push(`${counters.nonFinite} NaN/Inf samples (T15)`);
  if (counters.negative) errors.push(`${counters.negative} negative samples (T15)`);
  if (counters.bvhOverflow || counters.bvhItercap) errors.push(`BVH overflow ${counters.bvhOverflow}, iteration cap ${counters.bvhItercap}`);
  if (submits.overHardCap) errors.push(`${submits.overHardCap} submits above the 200 ms hard cap`);

  const config = {
    kernel: o.kernel, sppPerBatch: o.spp, width: W, height: H, jitter: 'iid-per-run', filter: 'box-1px',
    scene: src.source.packageSha256 ?? src.source.fileSha256, frame: o.frame ?? null, textureMode: 'validation', intersector: watertight ? 'woop-watertight' : 'moller-trumbore',
    ...(ptInfo ? {
      maxBounces: ptInfo.maxBounces, rr: ptInfo.rr, technique: ptInfo.technique, lightMode, ...(o.plant ? { plant: o.plant } : {}),
      env: envInfo ? { nee: envInfo.nee, importance: envInfo.importance ?? null, misPower: !!envInfo.misPower, plant: envInfo.plant ?? null, strengthScale: envInfo.strengthScale ?? null } : 'none',
    } : {}),
  };
  const configHash = await sha256Hex(new TextEncoder().encode(stable(config)));
  const info = describeContext(ctx) as { vendor?: string; architecture?: string; description?: string };
  const ua = navigator.userAgent;
  const meta = {
    kind: 'batches', kernel: o.kernel, seed: o.seed, sppPerBatch: o.spp, batches: o.batches, width: W, height: H,
    configHash, config,
    chromeVersion: o.chromeVersion ?? /(?:Headless)?Chrome\/([\d.]+)/.exec(ua)?.[1] ?? 'unknown',
    userAgent: ua,
    adapterInfo: { vendor: info.vendor, architecture: info.architecture, description: info.description },
    // replicate identifiers for compare.py (confirmatory re-runs need disjoint seeds)
    seeds: Array.from({ length: o.batches }, (_, b) => `${o.seed}:${b}`),
    sampleIndexing: 'batch b = run-global samples [b*sppPerBatch, (b+1)*sppPerBatch); jitter = rand2(pcg3d(seed, k, pixelIndex).x, 0, STREAM_JITTER)'
      + (o.kernel === 'pt' ? '; path stream = pcg3d(pcg3d(seed, k, pixelIndex).x, vertex*16 + slot, STREAM_PATH).x (math.md#rng-layout)' : ''),
    image: { format: 'PFM RGB float32', rowOrder: 'PFM bottom-to-top (decodes to row 0 = top)', value: 'batch mean radiance' },
    scene: {
      ...src.source, triangles: src.scene.geometry.indices.length / 3, env: !!src.scene.env, origin,
      ...(kernel instanceof EmissionKernel ? { cameraVisibleLights: kernel.cameraVisibleLights } : {}),
    },
    ...(ptInfo ? { pt: ptInfo } : {}),
    counters, submits,
    timings: { loadMs: tLoad - t0, setupMs: tSetup - tLoad, batchMs, totalMs: performance.now() - t0 },
    files, ok: errors.length === 0, errors,
    createdAt: new Date().toISOString(),
  };
  await uploadFile(o.run, 'meta.json', new TextEncoder().encode(JSON.stringify(meta, null, 1)));
  return { ok: errors.length === 0, run: o.run, files: [...files, 'meta.json'], meta, errors };
}
