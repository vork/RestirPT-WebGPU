// Reference path tracer kernels (plan §3 mode PT, §5 M3a). WGSL: shaders/passes/pt.wgsl.
//   PtKernel     validation batches (BatchAccumulator SampleEncoder: sample-indexed seeds and jitter, row bands)
//   PtFramePass  interactive progressive rendering (one sample per pixel per frame into the renderer's colour target)
//
// Bind groups (storage buffers 8 of 10 in both variants):
//   G0  0 FrameUniforms | 1..3 env (uniform, texEnv, sEnv) | 4 PtParams | 5 LightsParams | 6 records (lights-gpu.ts)
//   G1  scene (SceneGpu: bvh nodes/tris, vertices, tris, materials, textures)
//   G2  batch: 0 accum (rgb sum) | 1 counters (NaN/Inf, BVH overflow, BVH itercap, negative samples)
//       frame: 0 colour target (write) | 1 accum (rgb sum, w = count) | 2 counters
// The BSDF is material/material-eval.wgsl + bsdf.wgsl (docs/decisions/bsdf-api.md); its Cycles LUTs live in `records`
// at LUT_RECORDS_BASE (lights-gpu.ts), read as u32 bits.
// Env lighting (M3c): the env importance tables live in `records` too (no extra binding); the env is an alias entry
// unless env NEE is off (PtEnvOptions.nee = false ≡ Cycles world sampling_method NONE). See applyEnvLighting().
import { composeWgsl, createCheckedShaderModule } from '../gpu/wgsl-composer.ts';
import { shaderSources } from '../shaders/index.ts';
import type { LightData } from '../scene/types.ts';
import { envBindGroupEntries, envBindGroupLayoutEntries, envDefines, envImportanceFor, type EnvGpuResources } from './env-gpu.ts';
import {
  FrameUniformBuffer, JITTER_IID, JITTER_NONE, boundsDiagonal, type CameraState, type JitterMode,
} from './frame-uniforms.ts';
import { LightsGpu, LUT_RECORDS_BASE, type LightMode, type LightsUpdate } from './lights-gpu.ts';
import { lutDefines } from './luts/lut-layout.ts';
import { recentrePositions, type SceneGpu } from './scene-gpu.ts';

export const PT_PARAMS_SIZE = 48;
export const PT_COUNTER_BYTES = 16;
export const PT_COUNTERS = { nonFinite: 0, bvhOverflow: 1, bvhItercap: 2, negative: 3 } as const;
export const PT_FLAGS = { rr: 1, neeOnly: 2, bsdfOnly: 4, accumulate: 8, advanced: 16 } as const;
const ENV_BINDING_BASE = 1;
const SCENE_GROUP = 1;
const LIGHTS_BINDING = 5;

/** 'mis' = the production estimator; 'nee' / 'bsdf' = single-technique estimators for T9d (bsdf: no delta lights). */
export type PtTechnique = 'mis' | 'nee' | 'bsdf';

export interface PtSettings {
  /** Cycles max_bounces N (≤ N+1 scattering vertices). */
  maxBounces: number;
  /** Russian roulette (off by default; unbiased, interactive speed-up). */
  rr?: boolean;
  /** RR only at vertices B > rrMinBounces (default 3). */
  rrMinBounces?: number;
  technique?: PtTechnique;
  /** Gate-1 planted biases (plan §7.3; validation only). Defaults are the identity. */
  plant?: PtPlant;
}

/** Planted biases for the our-PT-vs-Cycles calibration: every emitter ×emitScale; drop (terminate, no compensation)
 *  the path with probability dropProb at scattering vertex dropBounce; glass (M3b, compiled in: material/glass.wgsl
 *  GLASS_PLANT): 'eta2' = a 1/η² radiance scaling of the BTDF (B-η), 'pr-half' = R/T chosen with 0.5 instead of P_R
 *  without pdf compensation, 'tint' = Principled glass C instead of √C (B-tint), 'side' = η not inverted on backfaces
 *  (B-side), 'shadow' = shadow rays pass through glass (B-shadow). */
export type GlassPlant = 'eta2' | 'pr-half' | 'tint' | 'side' | 'shadow';
export const GLASS_PLANTS: readonly GlassPlant[] = ['eta2', 'pr-half', 'tint', 'side', 'shadow'];
export interface PtPlant { emitScale?: number; dropProb?: number; dropBounce?: number; glass?: GlassPlant }

/** Env planted biases (validation only, env §5.3). 'strength' (env strength ×1.0075) is applied to the env params by
 *  the caller; 'pdfFromTargets' is an importance-table option; the others are WGSL defines (env-sample.wgsl). */
export type PtEnvPlant = 'missingSin' | 'w2WithoutPmf' | 'doubleCount' | 'pdfFromTargets';
const ENV_PLANT_CODE: Record<PtEnvPlant, number> = { missingSin: 1, w2WithoutPmf: 2, doubleCount: 3, pdfFromTargets: 0 };

/** Env sampling configuration (M3c; plan §1.4b UI "importance resolution", "Env NEE on/off"). */
export interface PtEnvOptions {
  /** Env NEE (default true ≡ Cycles world sampling_method AUTOMATIC); false = BSDF-only env (NONE). */
  nee?: boolean;
  /** Importance resolution cap: W_m = largest power of two ≤ min(W, cap) (default 4096, validation). */
  importanceCap?: number;
  /** Defensive floors (default true; false = negative control). */
  floors?: boolean;
  /** Power instead of balance heuristic for the env (negative control; compile-time). */
  misPower?: boolean;
  plant?: PtEnvPlant;
}

export interface PtKernelOptions extends Partial<PtSettings> {
  /** plan §1.4: 'A' (default), 'B' (pass-through + MIS), 'A′' (pass-through after delta lobes, weight 1). */
  lightMode?: LightMode;
  /** Tests (gap-light U10): pt_batch writes (vertex-sequence hash, crossing candidates, vertices) instead of radiance. */
  probe?: boolean;
  env?: PtEnvOptions;
  features?: Set<string>;
  wgslLanguageFeatures?: Set<string>;
}

export interface PtView {
  camera: CameraState;
  width: number;
  height: number;
  runSeed: number;
  /** Default JITTER_IID (validation). JITTER_NONE uses `jitter` (default pixel centre). */
  jitterMode?: JitterMode;
  jitter?: [number, number];
}

export interface PtDispatch { sampleBase: number; sampleCount: number; rowBase: number; rows: number }

function ptFlags(s: PtSettings): number {
  let f = s.rr ? PT_FLAGS.rr : 0;
  if (s.technique === 'nee') f |= PT_FLAGS.neeOnly;
  if (s.technique === 'bsdf') f |= PT_FLAGS.bsdfOnly;
  return f;
}

function g0LayoutEntries(): GPUBindGroupLayoutEntry[] {
  const c = GPUShaderStage.COMPUTE;
  return [
    { binding: 0, visibility: c, buffer: { type: 'uniform' } },
    ...envBindGroupLayoutEntries(ENV_BINDING_BASE, c),
    { binding: 4, visibility: c, buffer: { type: 'uniform', minBindingSize: PT_PARAMS_SIZE } },
    { binding: LIGHTS_BINDING, visibility: c, buffer: { type: 'uniform' } },
    { binding: LIGHTS_BINDING + 1, visibility: c, buffer: { type: 'read-only-storage' } },
  ];
}

async function compilePt(device: GPUDevice, scene: SceneGpu, layouts: GPUBindGroupLayout[], entry: 'pt_batch' | 'pt_frame',
  opts: { colorFormat?: string; features?: Set<string>; wgslLanguageFeatures?: Set<string>; probe?: boolean; plant?: PtPlant; env?: PtEnvOptions }): Promise<GPUComputePipeline> {
  const shader = composeWgsl('passes/pt.wgsl', {
    sources: shaderSources,
    defines: {
      ...scene.defines(SCENE_GROUP), ...envDefines(0, ENV_BINDING_BASE), LIGHTS_GROUP: 0, LIGHTS_BINDING,
      ...lutDefines({ base: LUT_RECORDS_BASE, recordsKind: 'u32' }),
      PT_INTERACTIVE: entry === 'pt_frame', PT_PROBE: !!opts.probe, GLASS_PLANT: opts.plant?.glass ? GLASS_PLANTS.indexOf(opts.plant.glass) + 1 : 0,
      ...(opts.colorFormat ? { COLOR_FORMAT: opts.colorFormat } : {}),
      ENV_PLANT: opts.env?.plant ? ENV_PLANT_CODE[opts.env.plant] : 0, ENV_MIS_POWER: !!opts.env?.misPower,
    },
    features: opts.features, wgslLanguageFeatures: opts.wgslLanguageFeatures,
  });
  const module = await createCheckedShaderModule(device, shader, entry);
  return device.createComputePipelineAsync({
    label: entry,
    layout: device.createPipelineLayout({ label: entry, bindGroupLayouts: layouts }),
    compute: { module, entryPoint: entry },
  });
}

/**
 * The env as a light for `lights` (M3c): importance tables (built synchronously unless the caller attached them, e.g.
 * the app's Worker build) + the current strength/tint. No env, or env NEE off → no alias entry (escapes ω2 = 1).
 */
export function applyEnvLighting(lights: LightsGpu, env: EnvGpuResources, o: PtEnvOptions = {}): LightsUpdate {
  if (!env.present || o.nee === false) return lights.setEnvironment(undefined);
  const table = envImportanceFor(env, { cap: o.importanceCap, floors: o.floors, plantPdfFromTargets: o.plant === 'pdfFromTargets' });
  if (!table) return lights.setEnvironment(undefined);
  return lights.setEnvironment({ table, strength: env.params.strength, tint: env.params.tint, nee: true });
}

function packParams(d: { sampleBase: number; sampleCount: number; rowBase: number; rowEnd: number }, s: PtSettings, extraFlags = 0): Uint32Array {
  const u = new Uint32Array(PT_PARAMS_SIZE / 4);
  u.set([d.sampleBase >>> 0, d.sampleCount, d.rowBase, d.rowEnd, s.maxBounces, ptFlags(s) | extraFlags, s.rrMinBounces ?? 3, 0]);
  const f = new Float32Array(u.buffer);
  f[8] = s.plant?.emitScale ?? 1;
  f[9] = s.plant?.dropProb ?? 0;
  u[10] = s.plant?.dropBounce ?? 0;
  return u;
}

/** Validation batches of the reference PT (BatchAccumulator SampleEncoder). */
export class PtKernel {
  readonly frame: FrameUniformBuffer;
  readonly lights: LightsGpu;
  readonly settings: PtSettings;
  private readonly params: GPUBuffer;
  private g0!: GPUBindGroup;
  private g0Version = -1;
  private readonly g1: GPUBindGroup;
  private g2: GPUBindGroup | undefined;
  private g2Key: [GPUBuffer, GPUBuffer] | undefined;
  private view: PtView | undefined;
  readonly envOptions: PtEnvOptions;

  private constructor(
    readonly device: GPUDevice,
    readonly scene: SceneGpu,
    private env: EnvGpuResources,
    private readonly pipeline: GPUComputePipeline,
    private readonly layouts: GPUBindGroupLayout[],
    opts: PtKernelOptions,
  ) {
    this.settings = { maxBounces: opts.maxBounces ?? 3, rr: opts.rr ?? false, rrMinBounces: opts.rrMinBounces ?? 3, technique: opts.technique ?? 'mis', plant: opts.plant };
    this.envOptions = { ...opts.env };
    this.frame = new FrameUniformBuffer(device);
    this.params = device.createBuffer({ label: 'pt-params', size: PT_PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.lights = new LightsGpu(device, scene.scene, scene.origin, recentrePositions(scene.scene.geometry.positions, scene.origin), { lightMode: opts.lightMode ?? 'A', label: 'pt-lights' });
    applyEnvLighting(this.lights, env, this.envOptions);
    this.g1 = device.createBindGroup({ label: 'pt-g1', layout: layouts[1], entries: scene.bindGroupEntries() });
  }

  static async create(device: GPUDevice, scene: SceneGpu, env: EnvGpuResources, opts: PtKernelOptions = {}): Promise<PtKernel> {
    const c = GPUShaderStage.COMPUTE;
    const g0 = device.createBindGroupLayout({ label: 'pt-g0', entries: g0LayoutEntries() });
    const g1 = device.createBindGroupLayout({ label: 'pt-g1', entries: scene.layoutEntries(c) });
    const g2 = device.createBindGroupLayout({
      label: 'pt-g2',
      entries: [
        { binding: 0, visibility: c, buffer: { type: 'storage' } },
        { binding: 1, visibility: c, buffer: { type: 'storage', minBindingSize: PT_COUNTER_BYTES } },
      ],
    });
    const pipeline = await compilePt(device, scene, [g0, g1, g2], 'pt_batch', opts);
    return new PtKernel(device, scene, env, pipeline, [g0, g1, g2], opts);
  }

  private ensureG0(): void {
    if (this.g0Version === this.lights.version && this.g0) return;
    this.g0 = this.device.createBindGroup({
      label: 'pt-g0',
      layout: this.layouts[0],
      entries: [
        { binding: 0, resource: { buffer: this.frame.buffer } },
        ...envBindGroupEntries(this.env, ENV_BINDING_BASE),
        { binding: 4, resource: { buffer: this.params } },
        { binding: LIGHTS_BINDING, resource: { buffer: this.lights.params } },
        { binding: LIGHTS_BINDING + 1, resource: { buffer: this.lights.records } },
      ],
    });
    this.g0Version = this.lights.version;
  }

  setEnvironment(env: EnvGpuResources): void { this.env = env; this.g0Version = -1; applyEnvLighting(this.lights, env, this.envOptions); }

  /** Env strength/tint changed (radiometric: the pmf is rebuilt; rotation never changes it). */
  envParamsChanged(): LightsUpdate { return applyEnvLighting(this.lights, this.env, this.envOptions); }

  /** New light list (stable ids; the alias table is rebuilt only when powers or the set change). */
  setLights(lights: readonly LightData[]): LightsUpdate { return this.lights.update(lights); }

  setSettings(s: Partial<PtSettings>): void { Object.assign(this.settings, s); }

  setView(v: PtView): void {
    this.view = v;
    const jm = v.jitterMode ?? JITTER_IID;
    this.frame.write({
      camera: v.camera, prevCamera: v.camera, width: v.width, height: v.height,
      frameIndex: 0, seedIndex: 0, runSeed: v.runSeed >>> 0, flags: 0, jitterMode: jm,
      jitter: jm === JITTER_NONE ? (v.jitter ?? [0.5, 0.5]) : [0.5, 0.5],
      origin: this.scene.origin, exposure: 1, time: 0, dt: 0, sceneDiag: boundsDiagonal(this.scene.scene.bounds),
    });
  }

  /** Encode one dispatch (params via queue.writeBuffer: submit before encoding the next dispatch). */
  encode(encoder: GPUCommandEncoder, d: PtDispatch, accum: GPUBuffer, counters: GPUBuffer): void {
    const v = this.view;
    if (!v) throw new Error('PtKernel.setView() first');
    this.ensureG0();
    if (this.g2Key?.[0] !== accum || this.g2Key?.[1] !== counters) {
      this.g2 = this.device.createBindGroup({
        label: 'pt-g2', layout: this.layouts[2],
        entries: [{ binding: 0, resource: { buffer: accum } }, { binding: 1, resource: { buffer: counters, size: PT_COUNTER_BYTES } }],
      });
      this.g2Key = [accum, counters];
    }
    this.device.queue.writeBuffer(this.params, 0, packParams({ ...d, rowEnd: d.rowBase + d.rows }, this.settings));
    const pass = encoder.beginComputePass({ label: 'pt' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.g0);
    pass.setBindGroup(1, this.g1);
    pass.setBindGroup(2, this.g2!);
    pass.dispatchWorkgroups(Math.ceil(v.width / 8), Math.ceil(d.rows / 8));
    pass.end();
  }

  destroy(): void {
    this.frame.destroy();
    this.params.destroy();
    this.lights.destroy();
  }
}

export interface PtFrameTargets { width: number; height: number; color: GPUTexture; frameUniforms: GPUBuffer }

/** Interactive PT pass: one sample per pixel per frame, progressive mean into the colour target (replaces the M1 beauty). */
export class PtFramePass {
  readonly lights: LightsGpu;
  readonly settings: PtSettings;
  private readonly params: GPUBuffer;
  readonly counters: GPUBuffer;
  private accum: GPUBuffer | undefined;
  private targets: PtFrameTargets | undefined;
  private g0: GPUBindGroup | undefined;
  private g0Key = '';
  private g2: GPUBindGroup | undefined;
  private readonly g1: GPUBindGroup;
  readonly envOptions: PtEnvOptions;

  private constructor(
    readonly device: GPUDevice,
    readonly scene: SceneGpu,
    private env: EnvGpuResources,
    private readonly pipeline: GPUComputePipeline,
    private readonly layouts: GPUBindGroupLayout[],
    readonly colorFormat: GPUTextureFormat,
    opts: PtKernelOptions,
  ) {
    this.settings = { maxBounces: opts.maxBounces ?? 3, rr: opts.rr ?? false, rrMinBounces: opts.rrMinBounces ?? 3, technique: opts.technique ?? 'mis' };
    this.envOptions = { ...opts.env };
    this.params = device.createBuffer({ label: 'pt-frame-params', size: PT_PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.counters = device.createBuffer({ label: 'pt-frame-counters', size: PT_COUNTER_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.lights = new LightsGpu(device, scene.scene, scene.origin, recentrePositions(scene.scene.geometry.positions, scene.origin), { lightMode: opts.lightMode ?? 'A', label: 'pt-frame-lights' });
    applyEnvLighting(this.lights, env, this.envOptions);
    this.g1 = device.createBindGroup({ label: 'pt-frame-g1', layout: layouts[1], entries: scene.bindGroupEntries() });
  }

  static async create(device: GPUDevice, scene: SceneGpu, env: EnvGpuResources, colorFormat: GPUTextureFormat, opts: PtKernelOptions = {}): Promise<PtFramePass> {
    const c = GPUShaderStage.COMPUTE;
    const g0 = device.createBindGroupLayout({ label: 'pt-frame-g0', entries: g0LayoutEntries() });
    const g1 = device.createBindGroupLayout({ label: 'pt-frame-g1', entries: scene.layoutEntries(c) });
    const g2 = device.createBindGroupLayout({
      label: `pt-frame-g2-${colorFormat}`,
      entries: [
        { binding: 0, visibility: c, storageTexture: { access: 'write-only', format: colorFormat } },
        { binding: 1, visibility: c, buffer: { type: 'storage' } },
        { binding: 2, visibility: c, buffer: { type: 'storage', minBindingSize: PT_COUNTER_BYTES } },
      ],
    });
    const pipeline = await compilePt(device, scene, [g0, g1, g2], 'pt_frame', { ...opts, colorFormat });
    return new PtFramePass(device, scene, env, pipeline, [g0, g1, g2], colorFormat, opts);
  }

  setEnvironment(env: EnvGpuResources): void { this.env = env; this.g0Key = ''; applyEnvLighting(this.lights, env, this.envOptions); }

  /** Env strength/tint changed (pmf rebuild), or the env NEE toggle / importance resolution (PtEnvOptions.nee /
   *  importanceCap; the table must already be attached to the EnvGpuResources for a non-blocking change). */
  setEnvOptions(o: Partial<PtEnvOptions> = {}): LightsUpdate {
    Object.assign(this.envOptions, o);
    const r = applyEnvLighting(this.lights, this.env, this.envOptions);
    if (r.reallocated) this.g0Key = '';
    return r;
  }
  setLights(lights: readonly LightData[]): LightsUpdate { return this.lights.update(lights); }
  setSettings(s: Partial<PtSettings>): void { Object.assign(this.settings, s); }

  /** Bind the renderer's targets (colour texture + frame uniforms); reallocates the accumulation on resize. */
  setTargets(t: PtFrameTargets): void {
    const old = this.targets;
    if (!old || old.width !== t.width || old.height !== t.height) {
      this.accum?.destroy();
      this.accum = this.device.createBuffer({ label: 'pt-frame-accum', size: t.width * t.height * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    }
    this.targets = t;
    this.g2 = this.device.createBindGroup({
      label: 'pt-frame-g2', layout: this.layouts[2],
      entries: [
        { binding: 0, resource: t.color.createView() },
        { binding: 1, resource: { buffer: this.accum! } },
        { binding: 2, resource: { buffer: this.counters } },
      ],
    });
    this.g0Key = '';
  }

  private ensureG0(): void {
    const t = this.targets!;
    const key = `${this.lights.version}`;
    if (this.g0 && this.g0Key === key) return;
    this.g0 = this.device.createBindGroup({
      label: 'pt-frame-g0', layout: this.layouts[0],
      entries: [
        { binding: 0, resource: { buffer: t.frameUniforms } },
        ...envBindGroupEntries(this.env, ENV_BINDING_BASE),
        { binding: 4, resource: { buffer: this.params } },
        { binding: LIGHTS_BINDING, resource: { buffer: this.lights.params } },
        { binding: LIGHTS_BINDING + 1, resource: { buffer: this.lights.records } },
      ],
    });
    this.g0Key = key;
  }

  /** Encode after the primary pass (the frame uniforms must already hold this frame's camera/seed/flags). */
  encode(encoder: GPUCommandEncoder, frame: { advanced: boolean; accumulate: boolean }, timestampWrites?: GPUComputePassTimestampWrites): boolean {
    const t = this.targets;
    if (!t || !this.g2) return false;
    this.ensureG0();
    const extra = (frame.accumulate ? PT_FLAGS.accumulate : 0) | (frame.advanced ? PT_FLAGS.advanced : 0);
    this.device.queue.writeBuffer(this.params, 0, packParams({ sampleBase: 0, sampleCount: 1, rowBase: 0, rowEnd: t.height }, this.settings, extra));
    const pass = encoder.beginComputePass({ label: 'pt-frame', timestampWrites });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.g0!);
    pass.setBindGroup(1, this.g1);
    pass.setBindGroup(2, this.g2);
    pass.dispatchWorkgroups(Math.ceil(t.width / 8), Math.ceil(t.height / 8));
    pass.end();
    return true;
  }

  destroy(): void {
    this.params.destroy();
    this.counters.destroy();
    this.accum?.destroy();
    this.lights.destroy();
  }
}

export { JITTER_IID, JITTER_NONE };
