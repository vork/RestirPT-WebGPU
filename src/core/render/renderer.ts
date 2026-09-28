// M1 renderer: scene + env GPU resources and the `primary` pass (plan §3 step 1, §5 M1). App-independent: the app
// glue (src/app/integration.ts) feeds it SceneData, EnvironmentData, render targets and frame encoders; GPU tests
// drive it directly.
//
// Bind groups of the primary pass (storage buffers 8 of 10, storage textures 3 of 8):
//   G0  0 FrameUniforms | 1..3 env (uniform, texEnv, sEnv) | 4 PrimaryParams          (plan §1.8: texEnv+sEnv in G0)
//   G1  scene: 0 bvh nodes | 1 bvh tris | 2 vertices | 3 tris | 4 materials | 8.. textures   (scene-gpu.ts)
//   G2  outputs: 0 color (write) | 1 linear depth | 2 V-buffer rgba32uint | 3 G-buffer | 4 accumulation
//   G3  debug (DebugResources.layout)
// Pipeline variants: colour format × BVH_STATS (compiled lazily when a BVH view is selected) × MT/Woop × the scene's
// texture binding counts. Switching views never recompiles otherwise (debugMode is a uniform).
// M3a: renderMode 'pt' (the app's default; the Renderer default stays 'albedo' for the M1 tests) adds the reference path tracer pass (pt-kernel.ts PtFramePass) after `primary`: one
// sample per pixel per frame, progressive mean into the colour target (replaces the M1 albedo placeholder beauty, which
// remains available as renderMode 'albedo'). Lights: setLights() (cur/prev light buffers, deterministic alias rebuild).
import { composeWgsl, createCheckedShaderModule } from '../gpu/wgsl-composer.ts';
import { shaderSources } from '../shaders/index.ts';
import type { EnvironmentData, SceneData } from '../scene/types.ts';
import type { DebugViewDef } from './debug-views.ts';
import {
  createEnvResources, destroyEnvResources, envBindGroupEntries, envBindGroupLayoutEntries, envDefines, envMemoryReport,
  writeEnvParams, type EnvGpuResources, type EnvParamsCpu,
} from './env-gpu.ts';
import { SceneGpu, type BvhBuilder } from './scene-gpu.ts';
import type { LightData } from '../scene/types.ts';
import type { LightsUpdate } from './lights-gpu.ts';
import { PtFramePass } from './pt-kernel.ts';
import type { TexturePathMode } from './textures-gpu.ts';

export const GBUF_TEXEL_BYTES = 80;
export const PRIMARY_PARAMS_SIZE = 16;
export const PRIM_ACCUMULATE = 1;
export const PRIM_ADVANCED = 2;
/** τ in thr = τ·R²_pri (math.md#rc-predicate). */
export const THR_TAU = 2e-4;
const ENV_BINDING_BASE = 1;
const SCENE_GROUP = 1;

/** Extra debug views written by gbuffer-views.wgsl (ids must match its DBG_* consts). */
export const EXTRA_VIEWS: DebugViewDef[] = [
  { id: 110, key: 'gbuffer.motionHue', label: 'Motion (hue = dir, value = length)', group: 'G-buffer', source: 'primary', kind: 'vec3', range: [0, 1] },
  { id: 111, key: 'check.nonFinite', label: 'NaN/Inf mask (1 colour, 2 vectors, 4 scalars)', group: 'Checks', source: 'primary', kind: 'code' },
  { id: 112, key: 'gbuffer.hitDist', label: 'Primary hit distance t', group: 'G-buffer', source: 'primary', kind: 'scalar', range: [0.1, 100], log: true, colormap: 'turbo' },
];
/** Probe tags written by primary.wgsl. */
export const PRIMARY_PROBE_TAGS: [number, string][] = [
  [16, 'primary.hit t|u|v|primId'], [17, 'primary.pos xyz|viewZ'], [18, 'primary.Ns xyz|matId'],
  [19, 'primary.bvh steps|box|tri|flags'], [20, 'primary.sample rgb|n'],
];
export const isBvhStatsView = (mode: number): boolean => mode >= 200 && mode < 300;

export interface RendererOptions {
  textureMode: TexturePathMode;
  watertight: boolean;
  accumulate: boolean;
  thrTau: number;
  /** 'pt': reference path tracer beauty (M3a); 'albedo': the M1 placeholder (albedo on hits, env on misses). */
  renderMode: 'pt' | 'albedo';
  /** PT: Cycles max_bounces N. */
  maxBounces: number;
  /** PT: Russian roulette (unbiased; off by default, plan §2 rule 11). */
  rr: boolean;
}

export interface RendererTargets {
  width: number;
  height: number;
  color: GPUTexture;
  colorFormat: GPUTextureFormat;
  depth: GPUTexture;
  frameUniforms: GPUBuffer;
}

export interface RendererContext {
  device: GPUDevice;
  /** Group-3 layout (DebugResources.layout) and the bind group to use for it. */
  debugLayout: GPUBindGroupLayout;
  features?: Set<string>;
  wgslLanguageFeatures?: Set<string>;
  buildBvh?: BvhBuilder;
}

interface SceneState {
  gpu: SceneGpu;
  /** PT pass (undefined while compiling or after a compile error; the albedo placeholder is shown meanwhile). */
  pt?: PtFramePass;
  ptPending?: Promise<PtFramePass | undefined>;
  layout: GPUBindGroupLayout;
  group: GPUBindGroup;
  pipelines: Map<string, GPUComputePipeline>;
  pending: Map<string, Promise<GPUComputePipeline | undefined>>;
}

interface TargetState {
  t: RendererTargets;
  gbuf: GPUBuffer;
  accum: GPUBuffer;
  vbuf: GPUTexture;
  group: GPUBindGroup;
  frameGroup: GPUBindGroup;
}

export class Renderer {
  readonly device: GPUDevice;
  /** Defaults are the validation path: exact textures and Woop watertight intersection (Möller–Trumbore leaks through
   *  the shared diagonal of a quad; plan §1.3). The interactive app opts into MT explicitly (src/app/integration.ts). */
  readonly options: RendererOptions = { textureMode: 'validation', watertight: true, accumulate: true, thrTau: THR_TAU, renderMode: 'albedo', maxBounces: 3, rr: false };
  env!: EnvGpuResources;
  sceneData: SceneData | undefined;
  origin: [number, number, number] = [0, 0, 0];
  /** Last compile/upload error (shown in the HUD). */
  lastError: string | undefined;
  private state: SceneState | undefined;
  private targets: TargetState | undefined;
  private readonly g0Layout: GPUBindGroupLayout;
  private readonly g2Layouts = new Map<string, GPUBindGroupLayout>();
  private readonly params: GPUBuffer;
  private readonly paramScratch = new ArrayBuffer(PRIMARY_PARAMS_SIZE);
  private generation = 0;
  private busy = 0;

  private constructor(private readonly ctx: RendererContext) {
    this.device = ctx.device;
    this.g0Layout = this.device.createBindGroupLayout({
      label: 'primary-g0',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ...envBindGroupLayoutEntries(ENV_BINDING_BASE, GPUShaderStage.COMPUTE),
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', minBindingSize: PRIMARY_PARAMS_SIZE } },
      ],
    });
    this.params = this.device.createBuffer({ label: 'primary-params', size: PRIMARY_PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  static async create(ctx: RendererContext, opts: Partial<RendererOptions> = {}): Promise<Renderer> {
    const r = new Renderer(ctx);
    Object.assign(r.options, opts);
    r.env = await createEnvResources(ctx.device, undefined, 'env');
    return r;
  }

  get scene(): SceneGpu | undefined { return this.state?.gpu; }
  /** True while a scene upload / recompile is in flight. */
  get loading(): boolean { return this.busy > 0; }
  get ready(): boolean { return !!this.state && !!this.targets && this.state.pipelines.size > 0; }

  // ---- scene ----------------------------------------------------------------------------------------------------

  /** Upload a scene (recentred by `origin`), build its BVH and compile the primary pipeline. Latest call wins. */
  async setScene(scene: SceneData, origin: [number, number, number]): Promise<SceneGpu | undefined> {
    const gen = ++this.generation;
    this.busy++;
    try {
      const gpu = await SceneGpu.create(this.device, scene, origin, {
        textureMode: this.options.textureMode, watertight: this.options.watertight, buildBvh: this.ctx.buildBvh,
        features: this.ctx.features, wgslLanguageFeatures: this.ctx.wgslLanguageFeatures,
      });
      const state = this.makeState(gpu);
      const key = this.variantKey(this.targets?.t.colorFormat ?? 'rgba32float', false);
      const ptP = this.compilePt(state, (this.targets?.t.colorFormat ?? 'rgba32float') as GPUTextureFormat);
      const pipe = await this.compile(state, key);
      await ptP;
      if (gen !== this.generation) { state.pt?.destroy(); gpu.destroy(); return undefined; }
      if (!pipe) { state.pt?.destroy(); gpu.destroy(); throw new Error(this.lastError ?? 'primary pipeline failed'); }
      state.pipelines.set(key, pipe);
      const old = this.state;
      this.state = state;
      this.sceneData = scene;
      this.origin = origin;
      old?.pt?.destroy();
      old?.gpu.destroy();
      this.lastError = undefined;
      return gpu;
    } finally {
      this.busy--;
    }
  }

  /** Change texture path / watertight intersection; re-uploads or recompiles as needed. */
  async setOptions(o: Partial<RendererOptions>): Promise<void> {
    const texChanged = o.textureMode !== undefined && o.textureMode !== this.options.textureMode;
    const wtChanged = o.watertight !== undefined && o.watertight !== this.options.watertight;
    Object.assign(this.options, o);
    this.state?.pt?.setSettings({ maxBounces: this.options.maxBounces, rr: this.options.rr });
    if (!this.sceneData) return;
    if (texChanged || wtChanged) await this.setScene(this.sceneData, this.origin);
  }

  /** Re-upload the current scene with the current options (texture path / watertight changed in place). */
  async reload(): Promise<void> {
    if (this.sceneData) await this.setScene(this.sceneData, this.origin);
  }

  private makeState(gpu: SceneGpu): SceneState {
    const layout = this.device.createBindGroupLayout({ label: 'primary-g1-scene', entries: gpu.layoutEntries(GPUShaderStage.COMPUTE) });
    const group = this.device.createBindGroup({ label: 'primary-g1-scene', layout, entries: gpu.bindGroupEntries() });
    return { gpu, layout, group, pipelines: new Map(), pending: new Map() };
  }

  /** Compile the PT pass for `state` (never throws; errors go to lastError and the placeholder beauty stays). */
  private compilePt(state: SceneState, colorFormat: GPUTextureFormat): Promise<PtFramePass | undefined> {
    if (state.ptPending && state.pt?.colorFormat === colorFormat) return state.ptPending;
    const p = (async () => {
      try {
        const pt = await PtFramePass.create(this.device, state.gpu, this.env, colorFormat, {
          maxBounces: this.options.maxBounces, rr: this.options.rr, features: this.ctx.features, wgslLanguageFeatures: this.ctx.wgslLanguageFeatures,
        });
        const tg = this.targets;
        if (tg) pt.setTargets({ width: tg.t.width, height: tg.t.height, color: tg.t.color, frameUniforms: tg.t.frameUniforms });
        state.pt?.destroy();
        state.pt = pt;
        return pt;
      } catch (e) {
        this.lastError = `PT: ${e instanceof Error ? e.message : String(e)}`;
        console.error(e);
        return undefined;
      }
    })();
    state.ptPending = p;
    return p;
  }

  /** Analytic lights for the PT (stable ids; the alias table is rebuilt only when powers or the set change). */
  setLights(lights: readonly LightData[]): LightsUpdate | undefined {
    return this.state?.pt?.setLights(lights);
  }

  private variantKey(colorFormat: string, stats: boolean): string { return `${colorFormat}|${stats ? 'stats' : 'plain'}`; }

  private g2Layout(colorFormat: GPUTextureFormat): GPUBindGroupLayout {
    let l = this.g2Layouts.get(colorFormat);
    if (!l) {
      const c = GPUShaderStage.COMPUTE;
      l = this.device.createBindGroupLayout({
        label: `primary-g2-${colorFormat}`,
        entries: [
          { binding: 0, visibility: c, storageTexture: { access: 'write-only', format: colorFormat } },
          { binding: 1, visibility: c, storageTexture: { access: 'write-only', format: 'r32float' } },
          { binding: 2, visibility: c, storageTexture: { access: 'write-only', format: 'rgba32uint' } },
          { binding: 3, visibility: c, buffer: { type: 'storage' } },
          { binding: 4, visibility: c, buffer: { type: 'storage' } },
        ],
      });
      this.g2Layouts.set(colorFormat, l);
    }
    return l;
  }

  /** Compose + compile one primary variant for `state` (never throws; errors go to lastError). */
  private compile(state: SceneState, key: string): Promise<GPUComputePipeline | undefined> {
    const existing = state.pending.get(key);
    if (existing) return existing;
    const [colorFormat, stats] = key.split('|');
    const p = (async () => {
      try {
        const shader = composeWgsl('passes/primary.wgsl', {
          sources: shaderSources,
          defines: {
            COLOR_FORMAT: colorFormat, BVH_STATS: stats === 'stats',
            ...state.gpu.defines(SCENE_GROUP), ...envDefines(0, ENV_BINDING_BASE),
          },
          features: this.ctx.features, wgslLanguageFeatures: this.ctx.wgslLanguageFeatures,
        });
        const module = await createCheckedShaderModule(this.device, shader, `primary-${key}`);
        return await this.device.createComputePipelineAsync({
          label: `primary-${key}`,
          layout: this.device.createPipelineLayout({
            label: 'primary', bindGroupLayouts: [this.g0Layout, state.layout, this.g2Layout(colorFormat as GPUTextureFormat), this.ctx.debugLayout],
          }),
          compute: { module, entryPoint: 'primary' },
        });
      } catch (e) {
        this.lastError = e instanceof Error ? e.message : String(e);
        console.error(e);
        return undefined;
      }
    })();
    state.pending.set(key, p);
    void p.then((pipe) => { if (pipe) state.pipelines.set(key, pipe); });
    return p;
  }

  /** Compile a variant now (e.g. BVH_STATS before a timing run). */
  async warmup(stats: boolean): Promise<void> {
    const s = this.state;
    if (!s || !this.targets) return;
    await this.compile(s, this.variantKey(this.targets.t.colorFormat, stats));
  }

  // ---- environment --------------------------------------------------------------------------------------------

  async setEnvironment(env: EnvironmentData | undefined): Promise<void> {
    const next = await createEnvResources(this.device, env, 'env');
    const old = this.env;
    this.env = next;
    this.state?.pt?.setEnvironment(next);
    if (this.targets) this.targets.frameGroup = this.frameGroup(this.targets.t);
    destroyEnvResources(old);
  }

  setEnvParams(p: Partial<EnvParamsCpu>): void { writeEnvParams(this.device, this.env, p); }

  // ---- targets --------------------------------------------------------------------------------------------------

  private frameGroup(t: RendererTargets): GPUBindGroup {
    return this.device.createBindGroup({
      label: 'primary-g0',
      layout: this.g0Layout,
      entries: [
        { binding: 0, resource: { buffer: t.frameUniforms } },
        ...envBindGroupEntries(this.env, ENV_BINDING_BASE),
        { binding: 4, resource: { buffer: this.params } },
      ],
    });
  }

  /** (Re)allocate the G-buffer, V-buffer and accumulation for new targets. */
  resize(t: RendererTargets): void {
    const old = this.targets;
    const n = t.width * t.height;
    const sameSize = old && old.t.width === t.width && old.t.height === t.height;
    const gbuf = sameSize ? old.gbuf : this.device.createBuffer({ label: 'gbuffer', size: n * GBUF_TEXEL_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const accum = sameSize ? old.accum : this.device.createBuffer({ label: 'accum', size: n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const vbuf = sameSize ? old.vbuf : this.device.createTexture({
      label: 'vbuffer', size: [t.width, t.height], format: 'rgba32uint',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    if (old && !sameSize) { old.gbuf.destroy(); old.accum.destroy(); old.vbuf.destroy(); }
    const group = this.device.createBindGroup({
      label: 'primary-g2',
      layout: this.g2Layout(t.colorFormat),
      entries: [
        { binding: 0, resource: t.color.createView() },
        { binding: 1, resource: t.depth.createView() },
        { binding: 2, resource: vbuf.createView() },
        { binding: 3, resource: { buffer: gbuf } },
        { binding: 4, resource: { buffer: accum } },
      ],
    });
    this.targets = { t, gbuf, accum, vbuf, group, frameGroup: this.frameGroup(t) };
    const s = this.state;
    if (s?.pt && s.pt.colorFormat !== t.colorFormat) void this.compilePt(s, t.colorFormat);
    else s?.pt?.setTargets({ width: t.width, height: t.height, color: t.color, frameUniforms: t.frameUniforms });
  }

  get gbuffer(): GPUBuffer | undefined { return this.targets?.gbuf; }
  get vbuffer(): GPUTexture | undefined { return this.targets?.vbuf; }

  // ---- frame ----------------------------------------------------------------------------------------------------

  /**
   * Encode the primary pass. `timestamps` is called only when the pass is actually encoded. Returns false when not
   * ready (no scene/targets yet, or a variant is still compiling on first use).
   */
  encode(
    encoder: GPUCommandEncoder,
    frame: { advanced: boolean; debugMode: number; debugGroup: GPUBindGroup; /** skip the PT pass (primary timing) */ noPt?: boolean },
    timestamps?: () => GPUComputePassTimestampWrites | undefined,
    ptTimestamps?: () => GPUComputePassTimestampWrites | undefined,
  ): boolean {
    const s = this.state;
    const tg = this.targets;
    if (!s || !tg) return false;
    const plainKey = this.variantKey(tg.t.colorFormat, false);
    let pipe: GPUComputePipeline | undefined;
    if (isBvhStatsView(frame.debugMode)) {
      const k = this.variantKey(tg.t.colorFormat, true);
      pipe = s.pipelines.get(k);
      if (!pipe) void this.compile(s, k);
    }
    pipe ??= s.pipelines.get(plainKey);
    if (!pipe) { void this.compile(s, plainKey); return false; }
    const u32 = new Uint32Array(this.paramScratch);
    const f32 = new Float32Array(this.paramScratch);
    u32[0] = (this.options.accumulate ? PRIM_ACCUMULATE : 0) | (frame.advanced ? PRIM_ADVANCED : 0);
    f32[1] = this.options.thrTau;
    this.device.queue.writeBuffer(this.params, 0, this.paramScratch);
    const pass = encoder.beginComputePass({ label: 'primary', timestampWrites: timestamps?.() });
    pass.setPipeline(pipe);
    pass.setBindGroup(0, tg.frameGroup);
    pass.setBindGroup(1, s.group);
    pass.setBindGroup(2, tg.group);
    pass.setBindGroup(3, frame.debugGroup);
    pass.dispatchWorkgroups(Math.ceil(tg.t.width / 8), Math.ceil(tg.t.height / 8));
    pass.end();
    // PT beauty after the primary pass (same jitter/seed; overwrites the placeholder colour). Skipped for BVH-stat views.
    if (this.options.renderMode === 'pt' && s.pt && !frame.noPt && !isBvhStatsView(frame.debugMode)) {
      s.pt.encode(encoder, { advanced: frame.advanced, accumulate: this.options.accumulate }, ptTimestamps?.());
    }
    return true;
  }

  /**
   * GPU time of the primary pass alone (timestamp queries around each of `iterations` back-to-back dispatches in one
   * submit, so the GPU stays busy and clocked up). Returns per-dispatch ms, or undefined without timestamp-query.
   * Uses the current targets/uniforms; the accumulation buffer is left as a single fresh sample.
   */
  async timePrimary(debugGroup: GPUBindGroup, iterations = 16): Promise<number[] | undefined> {
    if (!this.device.features.has('timestamp-query') || !this.ready) return undefined;
    const n = iterations * 2;
    const qs = this.device.createQuerySet({ type: 'timestamp', count: n });
    const res = this.device.createBuffer({ size: n * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = this.device.createBuffer({ size: n * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder({ label: 'primary-timing' });
    for (let i = 0; i < iterations; i++) {
      this.encode(enc, { advanced: false, debugMode: 0, debugGroup, noPt: true }, () => ({ querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 }));
    }
    enc.resolveQuerySet(qs, 0, n, res, 0);
    enc.copyBufferToBuffer(res, 0, read, 0, n * 8);
    this.device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const t = new BigInt64Array(read.getMappedRange().slice(0));
    read.unmap();
    for (const x of [qs, res, read]) x.destroy();
    return Array.from({ length: iterations }, (_, i) => Number(t[2 * i + 1] - t[2 * i]) / 1e6);
  }

  /** HUD summary lines. */
  hudLines(): string[] {
    const out: string[] = [];
    const g = this.state?.gpu;
    if (g) {
      const b = g.bvh.stats;
      const mib = (x: number) => (x / 2 ** 20).toFixed(1);
      out.push(`tris ${g.stats.triangles}  BVH ${b.nodeCount} nodes depth ${b.maxDepth} SAH ${b.sahCost.toFixed(1)} (${g.stats.bvhMs.toFixed(0)} ms)`);
      out.push(`isect ${g.watertight ? 'Woop (watertight)' : 'Möller–Trumbore'}  textures ${this.options.textureMode} ${mib(g.stats.textureBytes)} MiB  geom ${mib(g.stats.geometryBytes)} MiB`);
    }
    out.push(envMemoryReport(this.env).text + (this.env.present ? `  γ ${(this.env.params.rotationZ * 180 / Math.PI).toFixed(1)}° s ${this.env.params.strength}` : ''));
    const pt = this.state?.pt;
    if (this.options.renderMode === 'pt') {
      const l = pt?.lights.summary();
      out.push(pt
        ? `PT max_bounces ${pt.settings.maxBounces}${pt.settings.rr ? ' RR' : ''}  lights ${l!.analytic} analytic + ${l!.emissiveTriangles} emissive tris (Mode ${pt.lights.lightMode})`
        : 'PT: compiling (albedo placeholder shown)');
    }
    if (this.loading) out.push('renderer: uploading / compiling ...');
    if (this.lastError) out.push(`renderer error: ${this.lastError.split('\n')[0]}`);
    return out;
  }

  destroy(): void {
    this.state?.pt?.destroy();
    this.state?.gpu.destroy();
    if (this.targets) { this.targets.gbuf.destroy(); this.targets.accum.destroy(); this.targets.vbuf.destroy(); }
    destroyEnvResources(this.env);
    this.params.destroy();
  }
}
