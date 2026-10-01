// Browser side of the M5 temporal validation chains (restir-temporal-api.md §6.3–§6.5, TD25, TD26; OWNER T-E).
// Loads a scene package, builds a temporal ReSTIR kernel on an ensemble atlas (E chains per batch), runs `chains`
// chains through the package's frames (sequence packages: every frame's resolved state; static packages: the base state
// every frame) with ChainRunner and uploads, per reduced frame, `f<t>/ensemble.npz` + `f<t>/meta.json` (rows = chains,
// compare.py's ensemble format incl. mask regions) and `avg/…` for rung 3.5, plus the run's `meta.json` (T15/T16:
// counters, per-frame history validity, config). Mode `disocc` renders frames t−1 and t with jitter off at E = 1 and
// uploads the production T1 disocclusion flags as `f<t>/disocc.bin` (M_disocc, dyn_masks.py), for every test frame t.
import { describeContext, type GpuContext } from '../../src/core/gpu/device.ts';
import { sha256Hex } from '../../src/core/io/zlib.ts';
import type { SubmitBudget } from '../../src/core/render/batch-accumulator.ts';
import { createEnvResources, destroyEnvResources, writeEnvParams, type EnvGpuResources } from '../../src/core/render/env-gpu.ts';
import { computeRenderOrigin, JITTER_IID, JITTER_NONE } from '../../src/core/render/frame-uniforms.ts';
import type { PtEnvOptions } from '../../src/core/render/pt-kernel.ts';
import { ChainFrameCollector, ChainRunner, type ChainMasks, type ChainSpec } from '../../src/core/render/restir/chain-runner.ts';
import { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import { writeNpz, type NpzArray } from '../../src/core/render/restir/npz.ts';
import { RESTIR_PRESETS, restirSettings, type RestirPresetName, type RestirSettings } from '../../src/core/render/restir/presets.ts';
import { SceneGpu } from '../../src/core/render/scene-gpu.ts';
import { BASE_ENV_MAP, fetchScenePackage, resolvePackageFrame } from '../../src/core/scene/scene-package.ts';
import { packageSha256, uploadFile } from './export-package.ts';
import { stable } from './batch-run.ts';

export type TPlantName = keyof NonNullable<RestirSettings['tPlant']>;
export type U8PlantName = 'u8W1Delta' | 'u8NoPk' | 'u8T2' | 'u8OneSided' | 'u8FailedK';

export interface RenderRestirChainsOptions {
  run: string;
  package: string;
  /** Rung preset: 'temporal' (3.3), 'full' (3.4–3.6); any preset for tests. */
  preset: RestirPresetName;
  /** Chains of this invocation (multiple of members). */
  chains: number;
  /** First batch index of this invocation (GPU-lock chunks of one unit: batches [offset, offset + chains/E)). */
  batchOffset?: number;
  /** Chain id of batch 0's member 0 (default 0). */
  chainBase?: number;
  members?: number;
  seed: number;
  /** Frames per chain (default: the package's sequence.frameCount; static packages: required). */
  frames?: number;
  /** Test frames (default: the package's sequence.testFrames). */
  testFrames?: number[];
  /** Rung 3.5: per-chain mean over [from, to]. */
  average?: { from: number; to: number };
  /** Directory URL with f<t>/masks.json + f<t>/masks.bin (Uint16 bits per member-local pixel) per test frame. */
  masks?: string;
  temporalMis?: 'contribution' | 'talbot';
  temporalCheck?: 'none' | 'recompute' | 'robust';
  refresh?: 'exact' | 'e2';
  boostSlots?: number;
  tPlants?: TPlantName[];
  u8Plants?: U8PlantName[];
  wScale?: number;
  maxBounces?: number;
  env?: PtEnvOptions;
  /** 'disocc': M_disocc harness mode for every test frame t (a 2-frame chain t−1, t; E = 1; jitter off). */
  mode?: 'chains' | 'disocc';
  chromeVersion?: string;
  budget?: Partial<SubmitBudget>;
}

export interface RenderRestirChainsReport { ok: boolean; run: string; files: string[]; meta: Record<string, unknown>; errors: string[] }

/** Temporal plants are validation-only and never part of an unbiased unit. */
export const UNBIASED_CHAIN_PRESETS: readonly RestirPresetName[] = ['temporal', 'full', 'initial', 'initial-rr', 'offline', 'criteria2022'];

async function fetchBytes(url: string): Promise<Uint8Array> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

export async function renderRestirChains(ctx: GpuContext, o: RenderRestirChainsOptions): Promise<RenderRestirChainsReport> {
  const t0 = performance.now();
  const errors: string[] = [];
  if (!(o.preset in RESTIR_PRESETS)) throw new Error(`unknown preset ${o.preset}`);
  const { device, features, wgslLanguageFeatures } = ctx;
  const p = await fetchScenePackage(o.package);
  const hash = await packageSha256(p.files);
  const [W, H] = [p.render.width, p.render.height];
  const disocc = o.mode === 'disocc';
  const E = disocc ? 1 : o.members ?? 16;
  if (!disocc && (o.chains < E || o.chains % E)) throw new Error(`chains ${o.chains} must be a positive multiple of members ${E}`);
  const seqFrames = p.sequence?.frameCount;
  let frames = o.frames ?? seqFrames;
  let testFrames = o.testFrames ?? p.sequence?.testFrames ?? [];
  if (!frames) throw new Error(`${o.package}: no sequence block; --frames is required`);
  if (seqFrames && frames > seqFrames) throw new Error(`frames ${frames} > the package's ${seqFrames}`);
  const disoccFrames = disocc ? [...testFrames] : [];
  if (disocc && !(disoccFrames.length && disoccFrames.every((t) => t >= 1))) throw new Error('disocc mode: test frames t ≥ 1');
  // frame index map: chain frame i → package frame (disocc: t−1, t of the current test frame)
  let tDis = disoccFrames[0] ?? 0;
  const pkgFrame = (i: number) => (disocc ? tDis - 1 + i : i);
  if (disocc) { frames = 2; testFrames = []; }

  const origin = computeRenderOrigin(p.scene.bounds, p.scene.quant);
  const gpu = await SceneGpu.create(device, p.scene, origin, { textureMode: 'validation', watertight: true, features, wgslLanguageFeatures });
  const f0 = resolvePackageFrame(p, pkgFrame(0));
  let envMapId = f0.env?.mapId ?? BASE_ENV_MAP;
  let env: EnvGpuResources = await createEnvResources(device, f0.env?.map ?? p.scene.env);
  const oldEnvs: EnvGpuResources[] = [];
  if (f0.env) writeEnvParams(device, env, f0.env.params);
  const pkgEnvSampling = p.json.env?.sampling ?? 'AUTOMATIC';
  const envOpts: PtEnvOptions = { ...o.env, nee: o.env?.nee ?? pkgEnvSampling !== 'NONE' };
  const maxBounces = o.maxBounces ?? p.render.maxBounces ?? 3;
  if ((p.lightMode ?? 'A') !== 'A') throw new Error(`${o.package}: light mode ${p.lightMode}; M5 ReSTIR is Mode A only (restir-api.md D1)`);
  const tPlant = o.tPlants?.length ? Object.fromEntries(o.tPlants.map((n) => [n, true])) as RestirSettings['tPlant'] : undefined;
  const plant = (o.u8Plants?.length || (o.wScale !== undefined && o.wScale !== 1))
    ? { ...Object.fromEntries((o.u8Plants ?? []).map((n) => [n, true])), ...(o.wScale !== undefined && o.wScale !== 1 ? { wScale: o.wScale } : {}) } : undefined;
  const settings = restirSettings(o.preset, {
    maxBounces, ...(o.temporalMis ? { temporalMis: o.temporalMis } : {}), ...(o.temporalCheck ? { temporalCheck: o.temporalCheck } : {}),
    ...(o.refresh ? { refresh: o.refresh } : {}), ...(o.boostSlots !== undefined ? { boostSlots: o.boostSlots } : {}),
    ...(tPlant ? { tPlant } : {}), ...(plant ? { plant } : {}),
  });
  const kernel = await RestirKernel.create(device, gpu, env, { settings, lightMode: 'A', env: envOpts, features, wgslLanguageFeatures });
  const jitterMode = disocc ? JITTER_NONE : JITTER_IID;
  kernel.setView({ camera: { camToWorld: Array.from(f0.camera.camToWorld), yfov: f0.camera.yfov }, width: W, height: H, runSeed: o.seed >>> 0, members: E, jitterMode, jitter: [0.5, 0.5] });
  await kernel.prepare();
  const runner = new ChainRunner(kernel, { runSeed: o.seed >>> 0, budget: o.budget });

  // masks per test frame (fetched once)
  const masks = new Map<number, ChainMasks>();
  if (o.masks && !disocc) {
    const base = o.masks.endsWith('/') ? o.masks : `${o.masks}/`;
    for (const t of testFrames) {
      const j = JSON.parse(new TextDecoder().decode(await fetchBytes(`${base}f${t}/masks.json`))) as { names: string[]; width: number; height: number };
      if (j.width !== W || j.height !== H) throw new Error(`masks f${t}: ${j.width}x${j.height} != ${W}x${H}`);
      const b = await fetchBytes(`${base}f${t}/masks.bin`);
      masks.set(t, { names: j.names, bits: new Uint16Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)) });
    }
  }
  const resolved = new Map<number, ReturnType<typeof resolvePackageFrame>>();
  const frameOf = (i: number) => { const k = pkgFrame(i); let r = resolved.get(k); if (!r) { r = resolvePackageFrame(p, k); resolved.set(k, r); } return r; };
  const spec: ChainSpec = {
    frames, testFrames, ...(o.average ? { average: o.average } : {}),
    state: (t) => {
      const r = frameOf(t);
      return { t, camera: { camToWorld: Array.from(r.camera.camToWorld), yfov: r.camera.yfov }, lights: r.lights, ...(r.env ? { env: { params: r.env.params, mapId: r.env.mapId } } : {}) };
    },
    beforeFrame: async (t) => {
      const r = frameOf(t);
      if (r.env && r.env.mapId !== envMapId) {                 // map swap: new env resources (config change ⇒ reset, §2.5)
        oldEnvs.push(env);
        env = await createEnvResources(device, r.env.map);
        writeEnvParams(device, env, r.env.params);
        kernel.setEnvironment(env);
        envMapId = r.env.mapId;
      }
    },
    masks: (t) => masks.get(t),
  };
  const collectors = new Map<number | 'avg', ChainFrameCollector>();
  const tSetup = performance.now();
  const files: string[] = [];
  const batches = disocc ? 0 : o.chains / E;
  const off = o.batchOffset ?? 0;
  const chainBase = o.chainBase ?? 0;
  let histPattern: boolean[] = [];
  try {
    for (let b = off; b < off + batches; b++) {
      const recs = await runner.runBatch(spec, b, chainBase, (rows, ids) => {
        let c = collectors.get(rows.frame);
        if (!c) { c = new ChainFrameCollector(); collectors.set(rows.frame, c); }
        c.add(rows, ids, o.seed >>> 0);
      });
      if (b === off) histPattern = recs.map((r) => r.histValid);
    }
    for (const t of disoccFrames) {
      tDis = t;
      await runner.runBatch(spec, 0, chainBase, () => undefined);
      // each disocclusion "batch" is a different 2-frame chain (t−1, t): their reset patterns legitimately differ
      // (ixs_k: the map swap at 20 resets frame 1 of the f20 chain), so the cross-batch check does not apply here
      runner.totals.resetMismatches.length = 0;
      const m = await runner.disocclusionMask();
      const name = `f${t}__disocc.bin`;
      await uploadFile(o.run, name, m);
      files.push(name);
    }
    for (const [f, c] of collectors) {
      const dir = f === 'avg' ? 'avg' : `f${f}`;
      await uploadFile(o.run, `${dir}__ensemble.npz`, writeNpz(c.npzArrays() as NpzArray[]));
      files.push(`${dir}__ensemble.npz`);
      if (c.nonFinite) errors.push(`frame ${f}: ${c.nonFinite} non-finite member pixels in the reduction`);
    }
  } finally {
    runner.destroy(); kernel.destroy(); gpu.destroy(); destroyEnvResources(env);
    for (const e of oldEnvs) destroyEnvResources(e);
  }
  errors.push(...runner.invariantErrors());
  const T = runner.totals;
  // T16 "any config change resets history": within a chain, history is invalid exactly at t = 0 and at the package's
  // declared reset frames (map swaps); every other frame must carry valid history and emit temporal units.
  const resetFrames = new Set<number>([0, ...((p.json as { resetFrames?: number[] }).resetFrames ?? []).filter(() => !disocc)]);
  const temporalOn = settings.temporal;
  if (temporalOn) {
    for (const r of T.frameRecords) {
      const expectValid = !resetFrames.has(r.t);
      if (r.histValid !== expectValid) errors.push(`frame ${r.t}: histValid ${r.histValid}, expected ${expectValid} (${r.reasons.join(', ') || 'no reason'}) (T16 history resets)`);
      if (r.temporalUnits === 0) errors.push(`frame ${r.t}: no temporal units emitted (T16)`);
    }
  }
  const unbiased = UNBIASED_CHAIN_PRESETS.includes(o.preset) && !tPlant && !plant;
  const envInfo = env.present ? { nee: envOpts.nee !== false } : undefined;
  const config = {
    kernel: 'restir-chains', preset: o.preset, members: E, width: W, height: H, jitter: disocc ? 'none (pixel centre)' : 'iid-per-run', filter: 'box-1px',
    scene: hash.sha256, frames, testFrames, average: o.average ?? null, textureMode: 'validation', intersector: 'woop-watertight',
    maxBounces, lightMode: 'A', settings, env: envInfo ? { nee: envInfo.nee } : 'none', plants: { temporal: o.tPlants ?? [], u8: o.u8Plants ?? [], wScale: o.wScale ?? 1 },
    masks: o.masks ?? null, mode: o.mode ?? 'chains',
  };
  const configHash = await sha256Hex(new TextEncoder().encode(stable(config)));
  const info = describeContext(ctx) as { vendor?: string; architecture?: string; description?: string };
  const ua = navigator.userAgent;
  const baseMeta = {
    kind: 'chains', kernel: 'restir', seed: o.seed, members: E, chains: T.chains, batches: T.batches, batchOffset: off, chainBase,
    width: W, height: H, configHash, config,
    t16: {
      validationModeUnbiased: unbiased, plantsNamed: [...(o.tPlants ?? []), ...(o.u8Plants ?? []), ...(o.wScale !== undefined && o.wScale !== 1 ? ['wScale'] : [])],
      internalScale: 1, denoiser: 'none', upscaler: 'none',
      readback: 'linear rsFrame radiance per chain (f32 texels, f64 host reduction per test frame); no accumulation buffer',
      jitterMode: disocc ? 'none' : 'iid-per-run', maxBounces, lightMode: 'A', temporal: temporalOn, cCap: settings.cCap, rr: settings.rr,
      spatialRoundsRequested: settings.rounds, spatialRoundsExecuted: kernel.lastRounds, historyPattern: histPattern, resetFrames: [...resetFrames],
    },
    restir: { settings, counters: T.rsc, codes: T.codes, queueOverflow: T.queueOverflow, queueMaxCounter: T.queueMaxCounter, rates: T.rates, probeMs: T.probeMs, frameRecords: T.frameRecords },
    chromeVersion: o.chromeVersion ?? /(?:Headless)?Chrome\/([\d.]+)/.exec(ua)?.[1] ?? 'unknown', userAgent: ua,
    adapterInfo: { vendor: info.vendor, architecture: info.architecture, description: info.description },
    sampleIndexing: 'row r = chain c = chainBase + b*E + m (batch b, atlas member m); frame key = pcg3d(runSeed ^ c*0x9e3779b9, t, localIdx); t = frame from the chain reset (restir-temporal-api.md §5, TD25)',
    scene: { url: o.package, name: p.json.name, packageSha256: hash.sha256, sequence: p.sequence ?? null, origin },
    counters: T.counters, submits: T.submits,
    timings: { setupMs: tSetup - t0, batchMs: T.batchMs, totalMs: performance.now() - t0 },
    ok: errors.length === 0, errors, createdAt: new Date().toISOString(),
  };
  for (const [f, c] of collectors) {
    const dir = f === 'avg' ? 'avg' : `f${f}`;
    const m = { ...baseMeta, frame: f, seeds: c.seeds, image: { format: 'ensemble.npz (compare.py README)', frame: f, masks: c.shape?.maskNames ?? [] } };
    await uploadFile(o.run, `${dir}__meta.json`, new TextEncoder().encode(JSON.stringify(m, null, 1)));
    files.push(`${dir}__meta.json`);
  }
  const meta = { ...baseMeta, files };
  await uploadFile(o.run, 'meta.json', new TextEncoder().encode(JSON.stringify(meta, null, 1)));
  return { ok: errors.length === 0, run: o.run, files: [...files, 'meta.json'], meta, errors };
}
