// ReSTIR PT kernel orchestration (restir-api.md §4, P0/WP-A): frame work units, the stage interface for the spatial
// and ensemble stages (WP-C), pipelines, G0/G1/G3 bind groups, the RsDispatch ring (D17), counters/reservoir readback
// and the interactive frame pass. Light modes A, B and A′ (restir-m6-api MD9 lifted D1's Mode-A-only restriction).
//
// Frame schedule (§4.1): rs_primary (row bands) → rs_initial × tree chunks (row bands) → spatial stage units (rounds)
// → rs_finalize (+ ensemble stage units). The estimate in finalize is rsShade when the spatial stage emitted units
// for this frame, else F·W of res[0] (rung 3.1; also the canonical-only output while the spatial stage is stubbed).
// M5 (restir-temporal-api.md §4.1, TD2, TD15; P0/A by T-A): with settings.temporal, advance(state) runs once per frame
// before frameUnits(t): it commits the light state, flips the G-buffer parity g and rotates the reservoir roles —
// h = history (the previous frame's final buffer, −1 after a reset), w = 1 − h (0 after a reset): rs_initial writes
// res[w], the temporal stage (stage-temporal.ts) reads res[h] and writes res[w] in place, spatial round r reads
// res[(w + r) % 2], finalize reads res[finalResIndex()]. Without a preceding advance() (temporal off, M4 callers) the
// frame is a reset: w = 0 and no temporal unit is emitted — bitwise the M4 schedule.
import { composeWgsl, createCheckedShaderModule, type Defines } from '../../gpu/wgsl-composer.ts';
import { readBuffer } from '../../gpu/readback.ts';
import { shaderSources } from '../../shaders/index.ts';
import type { LightData } from '../../scene/types.ts';
import type { DebugResources } from '../debug-views.ts';
import { envBindGroupEntries, envBindGroupLayoutEntries, type EnvGpuResources } from '../env-gpu.ts';
import { FrameUniformBuffer, JITTER_IID, JITTER_NONE, boundsDiagonal, type CameraState, type JitterMode } from '../frame-uniforms.ts';
import { LightsGpu, type LightMode, type LightsUpdate } from '../lights-gpu.ts';
import { envImportanceKey, packEnvParams } from '../env-gpu.ts';
import { applyEnvLighting, type PtEnvOptions } from '../pt-kernel.ts';
import { recentrePositions, type SceneGpu } from '../scene-gpu.ts';
import {
  RES_BYTES, RES_PLANES, RESTIR_PARAMS_SIZE, RS_DISPATCH_RING, RS_DISPATCH_SIZE, RS_DISPATCH_STRIDE, RS_TEMPORAL_SIZE, RSC, RS_WGSL_CONSTS as K,
  ARENA_HDR_BYTES, arenaWords, nsAlloc, packRestirParams, packRsDispatch, packRsTemporal, queueHdr, type RscName, type RsDispatchCpu,
} from './layout.ts';
import { m6Defines, m7NmPlantDefine, numSlotsOf, pairTexSizes, restirFlags, restirSettings, tModeOf, tPlantsOf, validateSettings, type RestirSettings } from './presets.ts';
import { RS_M6_CONSTS as K6, arenaM6Base } from './layout.ts';
import { G0_BINDING, RS_PASSES, RestirResources, createUniforms, g2LayoutEntries, restirCommonDefines, restirDefines, type RsPassName } from './resources.ts';
import { SpatialStage } from './stage-spatial.ts';
import { EnsembleStage } from './ensemble.ts';
import { TemporalStage } from './stage-temporal.ts';
import { FrameStateTracker, type ConfigHashInput, type RestirAdvance, type RestirFrameState, type RestirInteractiveAdvance } from './frame-state.ts';
import { RELEASE_PERF_FLAGS, normalizePerfFlags, perfFlagDefines, perfFlagsKey, type PerfFlags, type PerfFlagsInput } from './perf-flags.ts';

export type { RestirSettings } from './presets.ts';
export type { RestirAdvance, RestirFrameState } from './frame-state.ts';
export { RESTIR_PRESETS } from './presets.ts';
export { PERF_FLAGS, RELEASE_PERF_FLAGS, normalizePerfFlags, perfFlagDefines, perfFlagsKey, type PerfFlagName, type PerfFlags, type PerfFlagsInput } from './perf-flags.ts';

export interface WorkUnit { label: string; costHint: number; encode(enc: GPUCommandEncoder): void }
/** A stage of the frame graph (spatial = WP-C stage-spatial.ts, ensemble = WP-C ensemble.ts). `prepare` compiles its
 *  pipelines (called by RestirKernel.prepare() when the settings / view need the stage). */
export interface RestirStage { frameUnits(k: RestirKernel, t: number): WorkUnit[]; prepare?(k: RestirKernel): Promise<void>; destroy?(): void }
export interface RestirFrameOut {
  accum?: GPUBuffer; counters?: GPUBuffer; colorTarget?: GPUTexture; ensemble?: boolean;
  /** Interactive progressive mean (RestirFramePass). */
  interactive?: { advanced: boolean; accumulate: boolean; /** perf2 WP-7c (RS_SKIP_DISPLAY): the denoiser writes the colour target */ noDisplay?: boolean };
}

export interface RestirQueueHeader { counter: number; n: number; capacity: number; overflow: number }
export interface RestirCounters {
  queues: RestirQueueHeader[];
  rsc: Record<RscName, number>;
  /** Histogram of slot outcome codes (SC_*), header words 32–47. */
  codes: number[];
  /** Replay fraction f_r = QUEUED / ACCEPTED (0 when nothing was accepted). */
  fr: number;
  raw: Uint32Array;
}

export interface RestirView {
  camera: CameraState; width: number; height: number; runSeed: number;
  /** Ensemble members E in the atlas (default 1 = sequential). */
  members?: number;
  /** Member id of atlas member 0 (sequential runs of member m; default 0). */
  memberBase?: number;
  jitterMode?: JitterMode; jitter?: [number, number];
}

export interface RestirKernelOptions {
  settings: Partial<RestirSettings>;
  lightMode?: LightMode;
  /** M8 (m8-perf.md §5, P-4): compile light modes B / A′ as the Mode-A pipeline text while no rect / disk light exists
   *  (nothing a BSDF ray could cross: the two estimators are bitwise equal, U-M8-MODEB). Off by default (every validation
   *  caller keeps the requested text); RestirKernel.interactive (the app) turns it on. */
  modeBNeedsAreaLights?: boolean;
  /** M8 (m8-perf.md §8, P-7): reservoir planes record-major ('aos', the M4 layout; default: every validation caller) or
   *  plane-major ('soa', composer define RS_RES_SOA; RestirKernel.interactive). readReservoirs() always returns AoS. */
  resLayout?: 'aos' | 'soa';
  /** perf2 (docs/decisions/perf2-api.md): interactive-only optimisation defines from the registry in perf-flags.ts.
   *  Default: none (every validation caller); RestirKernel.interactive defaults to RELEASE_PERF_FLAGS. Part of
   *  variantKey(); setPerfFlags() switches them at a frame boundary (recompile + history reset). */
  perfFlags?: PerfFlagsInput;
  env?: PtEnvOptions;
  debug?: DebugResources;
  features?: Set<string>;
  wgslLanguageFeatures?: Set<string>;
  /** Tests: candidate dump; extra defines of rs_initial (e.g. RS_PT_DIRECTIONS, RS_RNG_OVERRIDE). */
  instrumentation?: { dumpCandidates?: boolean; initialDefines?: Defines; extraSources?: Record<string, string> };
}

const ENV_BINDING_BASE = 1;
const SCENE_GROUP = 1;
/** Scene generation ids for the config hash (§2.10): one per SceneGpu object. */
const sceneGens = new WeakMap<object, number>();
let nextSceneGen = 1;
const sceneGen = (s: object): number => { let g = sceneGens.get(s); if (g === undefined) { g = nextSceneGen++; sceneGens.set(s, g); } return g; };

export class RestirKernel {
  readonly frame: FrameUniformBuffer;
  readonly lights: LightsGpu;
  settings: RestirSettings;
  readonly envOptions: PtEnvOptions;
  readonly spatial: RestirStage;
  readonly ensemble: RestirStage;
  /** M5 temporal stage (T-B, stage-temporal.ts); emits units only on frames prepared by advance() with temporal on. */
  readonly temporal: RestirStage;
  /** M5 per-frame state (frame-state.ts): light commit, env record, config hash, history validity, RsTemporal. */
  readonly frameState: FrameStateTracker;
  /** Rows per work unit for per-pixel passes (default: the whole atlas) and trees per rs_initial unit (default S). */
  rowBand = 0;
  treeChunk = 0;
  /** Spatial rounds executed by the last frameUnits() (0 while the spatial stage emits nothing). */
  lastRounds = 0;
  private res: RestirResources | undefined;
  private view: RestirView | undefined;
  private external: { frameUniforms: GPUBuffer; width: number; height: number } | undefined;
  private readonly params: GPUBuffer;
  private readonly ring: GPUBuffer;
  /** RsTemporal uniform (G0 binding 8, restir-temporal-api.md §2.7). */
  readonly rsTemporal: GPUBuffer;
  /** Reservoir roles of the current frame (TD2): h = history index (−1: none), w = the non-history buffer. */
  private roleH = -1;
  private roleW = 0;
  /** advance() prepared the next frameUnits() call (temporal units allowed). */
  private advanced: RestirAdvance | undefined;
  /** The RestirAdvance of the frame whose units frameUnits() builds / built last (undefined: a non-advanced frame). */
  private builtAdv: RestirAdvance | undefined;
  /** Final reservoir index of the last frameUnits() (the next frame's history when it stays valid). */
  private lastFinal = 0;
  /** The last frame whose units were built had temporal units (its final buffer is a valid history candidate). */
  private lastWasAdvanced = false;
  /** Env map generation (bumps on setEnvironment; a config-hash input). */
  private envMapGen = 0;
  private ringCursor = 0;
  private g0: GPUBindGroup | undefined;
  private g0Key = '';
  private readonly g1Scene: GPUBindGroup;
  private readonly g1Empty: GPUBindGroup;
  private readonly g3Empty: GPUBindGroup;
  readonly layouts: { g0: GPUBindGroupLayout; g1Scene: GPUBindGroupLayout; empty: GPUBindGroupLayout; g3: GPUBindGroupLayout };
  private readonly g2Layouts = new Map<string, GPUBindGroupLayout>();
  private readonly pipelines = new Map<string, Promise<GPUComputePipeline>>();
  private prepared = '';
  /** perf2: the normalised perf-flag set (sorted; absent = off). */
  private perfFlagSet: PerfFlags = {};

  /** Light mode (M6 MD9: A, B or A′; B / A′ compile the crossing variant RS_MODE_B). */
  lightMode: LightMode;

  private constructor(readonly device: GPUDevice, readonly scene: SceneGpu, private env: EnvGpuResources, readonly o: RestirKernelOptions) {
    this.lightMode = o.lightMode ?? 'A';
    this.perfFlagSet = normalizePerfFlags(o.perfFlags);
    this.settings = restirSettings(undefined, o.settings);
    this.envOptions = { ...o.env };
    this.frame = new FrameUniformBuffer(device);
    ({ params: this.params, ring: this.ring } = createUniforms(device));
    this.rsTemporal = device.createBuffer({ label: 'rs-temporal', size: RS_TEMPORAL_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.lights = new LightsGpu(device, scene.scene, scene.origin, recentrePositions(scene.scene.geometry.positions, scene.origin), { lightMode: this.lightMode, label: 'rs-lights' });
    applyEnvLighting(this.lights, env, this.envOptions);
    this.variantLightMode = this.desiredVariantLightMode();
    const c = GPUShaderStage.COMPUTE;
    const empty = device.createBindGroupLayout({ label: 'rs-empty', entries: [] });
    this.layouts = {
      g0: device.createBindGroupLayout({
        label: 'rs-g0',
        entries: [
          { binding: G0_BINDING.frame, visibility: c, buffer: { type: 'uniform' } },
          ...envBindGroupLayoutEntries(ENV_BINDING_BASE, c),
          { binding: G0_BINDING.params, visibility: c, buffer: { type: 'uniform', minBindingSize: RESTIR_PARAMS_SIZE } },
          { binding: G0_BINDING.lights, visibility: c, buffer: { type: 'uniform' } },
          { binding: G0_BINDING.records, visibility: c, buffer: { type: 'read-only-storage' } },
          { binding: G0_BINDING.dispatch, visibility: c, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: RS_DISPATCH_SIZE } },
          { binding: G0_BINDING.temporal, visibility: c, buffer: { type: 'uniform', minBindingSize: RS_TEMPORAL_SIZE } },
        ],
      }),
      g1Scene: device.createBindGroupLayout({ label: 'rs-g1', entries: scene.layoutEntries(c) }),
      empty,
      g3: o.debug?.layout ?? empty,
    };
    this.g1Scene = device.createBindGroup({ label: 'rs-g1', layout: this.layouts.g1Scene, entries: scene.bindGroupEntries() });
    this.g1Empty = device.createBindGroup({ label: 'rs-g1-empty', layout: empty, entries: [] });
    this.g3Empty = this.g1Empty;
    this.spatial = new SpatialStage();
    this.ensemble = new EnsembleStage();
    this.temporal = new TemporalStage();
    this.frameState = new FrameStateTracker();
    // M5 TD4: with temporal on, light / env edits are staged and committed once per frame (advance()).
    this.lights.deferred = !!this.settings.temporal;
  }

  static async create(device: GPUDevice, scene: SceneGpu, env: EnvGpuResources, o: RestirKernelOptions): Promise<RestirKernel> {
    const k = new RestirKernel(device, scene, env, o);
    // (RS_PRIMARY_EXT: rs_vtrace compiles in prepare() only when no external V-buffer serves the view)
    const names: RsPassName[] = [k.primaryExt() ? 'rs_primary_ext' : 'rs_primary', 'rs_initial', 'rs_finalize'];
    if (o.instrumentation?.dumpCandidates) names.push('rs_initial_dump');
    await Promise.all(names.map((n) => k.pipeline(n)));
    await k.prepare();
    return k;
  }

  /** Interactive ReSTIR (renderer mode 'restir'): progressive mean into the renderer's colour target. */
  static async interactive(device: GPUDevice, scene: SceneGpu, env: EnvGpuResources, colorFormat: GPUTextureFormat,
    o: Omit<RestirKernelOptions, 'settings'> & { settings?: Partial<RestirSettings> } = {}): Promise<RestirFramePass> {
    const k = await RestirKernel.create(device, scene, env, {
      modeBNeedsAreaLights: true, resLayout: 'soa', ...o, perfFlags: o.perfFlags ?? RELEASE_PERF_FLAGS, settings: { ...restirSettings('interactive'), ...o.settings },
    });
    await k.pipeline('rs_finalize_frame', {}, colorFormat);
    return new RestirFramePass(k, colorFormat);
  }

  // ---------------------------------------------------------------------------------------------- pipelines

  /** G2 layout of a pass (the interactive finalize also depends on the colour format). */
  g2Layout(name: RsPassName, colorFormat?: GPUTextureFormat): GPUBindGroupLayout {
    const key = `${name}:${colorFormat ?? ''}`;
    let l = this.g2Layouts.get(key);
    if (!l) {
      l = this.device.createBindGroupLayout({ label: `rs-g2-${key}`, entries: g2LayoutEntries(name, colorFormat) });
      this.g2Layouts.set(key, l);
    }
    return l;
  }

  /** Composer defines shared by every ReSTIR pipeline (+ the pass's own). */
  defines(name: RsPassName, extra: Defines = {}): Defines {
    return restirDefines(name, { sceneDefines: this.scene.defines(SCENE_GROUP), debug: !!this.o.debug, extra: { ...this.m6Defines(), NM_PLANT: m7NmPlantDefine(this.settings, name), ...this.layoutDefines(), ...this.perfDefines(), ...extra } });
  }
  /** M8 P-7: RS_RES_SOA only for plane-major kernels (the key is absent otherwise: the M7 text). */
  layoutDefines(): Record<string, number> { return this.o.resLayout === 'soa' ? { RS_RES_SOA: 1 } : {}; }
  /** perf2: composer defines of the kernel's perf flags (empty without flags: the validation text). */
  perfDefines(): Record<string, number> { return perfFlagDefines(this.perfFlagSet); }
  /** perf2: the normalised perf-flag set of the kernel. */
  get perfFlags(): PerfFlags { return this.perfFlagSet; }
  /** perf2: switch the perf flags (a new pipeline variant: `await prepare()` before the next frame; the temporal history
   *  resets, as for a light-mode variant switch). Returns true when the set changed. */
  setPerfFlags(f: PerfFlagsInput): boolean {
    const next = normalizePerfFlags(f);
    if (perfFlagsKey(next) === perfFlagsKey(this.perfFlagSet)) return false;
    this.perfFlagSet = next;
    this.frameState.invalidate('perf-flags');
    return true;
  }
  /** M8 P-7: u32 words between consecutive planes of a record's plane 0 … (denoiser: 10 AoS, 1 SoA for plane 0). */
  get resPlaneStride(): number { return this.o.resLayout === 'soa' ? 1 : RES_PLANES;
  }

  /** M6 pipeline-variant defines of the current settings / light mode (restir-m6-api.md MD1). */
  m6Defines(): Record<string, number> { return m6Defines(this.settings, this.variantLightMode); }
  /** The light mode whose pipeline text is compiled (= lightMode unless modeBNeedsAreaLights finds no rect / disk light). */
  private variantLightMode: LightMode = 'A';
  private desiredVariantLightMode(): LightMode {
    return this.o.modeBNeedsAreaLights && this.lightMode !== 'A' && !this.lights.hasAreaLights ? 'A' : this.lightMode;
  }
  /** M8 P-4: re-evaluate the compiled light-mode text at a frame boundary (call before checking isPrepared()): a switch
   *  is a new variant (prepare() compiles it) and resets the temporal history. Returns true when it switched. */
  syncLightModeVariant(): boolean {
    const m = this.desiredVariantLightMode();
    if (m === this.variantLightMode) return false;
    this.variantLightMode = m;
    this.frameState.invalidate('light-mode-variant');
    return true;
  }
  /** Cache key of the current pipeline variant. */
  variantKey(): string {
    const d = this.m6Defines();
    const pf = perfFlagsKey(this.perfFlagSet);   // perf2: '' without flags (the pre-perf2 key)
    return `${d.RS_RIS_NEE}${d.RS_MODE_B}${d.RS_DUAL_MV}${d.RS_DUPMAP}${d.RS_PLANT_T2}${d.RS_PLANT_SMOOTH_J}${m7NmPlantDefine(this.settings, 'rs_spatial_shift')}${pf ? `|${pf}` : ''}`;
  }

  /** Compile (once) the pipeline of a standard pass; `extra` defines give test variants (their own cache key). */
  pipeline(name: RsPassName, extra: Defines = {}, colorFormat?: GPUTextureFormat): Promise<GPUComputePipeline> {
    const vk = this.variantKey();
    const key = `${name}:${JSON.stringify(extra)}:${colorFormat ?? ''}:${vk}`;
    let p = this.pipelines.get(key);
    if (!p) {
      const d = RS_PASSES[name];
      const std = Object.keys(extra).length === 0;
      const inst = (name === 'rs_initial' || name === 'rs_initial_dump') ? this.o.instrumentation?.initialDefines ?? {} : {};
      p = this.compile(d.file, d.entry, this.defines(name, { ...inst, ...extra, ...(colorFormat ? { COLOR_FORMAT: colorFormat } : {}) }),
        this.pipelineLayout(name, colorFormat), `${name}${std ? '' : ':' + JSON.stringify(extra)}`, this.o.instrumentation?.extraSources ?? {})
        .then((pl) => { if (std) this.ready.set(`${name}:${colorFormat ?? ''}:${vk}`, pl); return pl; });
      this.pipelines.set(key, p);
      p.catch(() => this.pipelines.delete(key));
    }
    return p;
  }

  /** Pipeline layout [G0, G1 scene | empty, G2, G3 debug | empty] of a pass. */
  pipelineLayout(name: RsPassName, colorFormat?: GPUTextureFormat, g2?: GPUBindGroupLayout): GPUPipelineLayout {
    const d = RS_PASSES[name];
    return this.device.createPipelineLayout({
      label: `rs-${name}`,
      bindGroupLayouts: [this.layouts.g0, d.scene ? this.layouts.g1Scene : this.layouts.empty, g2 ?? this.g2Layout(name, colorFormat),
        d.debug && this.o.debug ? this.layouts.g3 : this.layouts.empty],
    });
  }

  /** Compile a module from the shader tree (or an extra in-memory source for tests) with the given defines. */
  async compile(file: string, entry: string, defines: Defines, layout: GPUPipelineLayout, label = entry, extraSources: Record<string, string> = {}): Promise<GPUComputePipeline> {
    const shader = composeWgsl(file, { sources: { ...shaderSources, ...extraSources }, defines, features: this.o.features, wgslLanguageFeatures: this.o.wgslLanguageFeatures });
    const module = await createCheckedShaderModule(this.device, shader, label);
    return this.device.createComputePipelineAsync({ label, layout, compute: { module, entryPoint: entry } });
  }

  /** Compile the stages the current settings / view need (spatial when rounds > 0, ensemble when E > 1). */
  async prepare(): Promise<void> {
    const key = this.prepareKey();
    if (key === this.prepared) return;
    // the frame passes of the current variant (cached; a variant change recompiles them, MD1)
    const base: RsPassName[] = [...this.primaryPasses(), 'rs_initial', 'rs_finalize'];
    if (this.o.instrumentation?.dumpCandidates) base.push('rs_initial_dump');
    if (this.settings.risNee) base.push('rs_light_tiles');
    if (this.settings.dupmap && this.settings.temporal) base.push('rs_dupmap');
    await Promise.all(base.map((n) => this.pipeline(n)));
    if (this.colorFormat) await this.pipeline('rs_finalize_frame', {}, this.colorFormat);
    if (this.settings.rounds > 0) await this.spatial.prepare?.(this);
    if ((this.view?.members ?? 1) > 1) await this.ensemble.prepare?.(this);
    if (this.settings.temporal) await this.temporal.prepare?.(this);
    // a prepare() of an older variant that finishes after the settings changed again must not mark the new one ready
    if (key === this.prepareKey()) this.prepared = key;
  }

  /** Key of what prepare() compiles: the stages the settings / view need and the pipeline variant. */
  prepareKey(): string {
    return `${this.settings.rounds > 0}:${(this.view?.members ?? 1) > 1}:${this.settings.temporal}:${this.variantKey()}${this.primaryExt() ? `:x${this.extVsrc() ? 1 : 0}` : ''}`;
  }

  // ---- perf2 WP-7e (RS_PRIMARY_EXT) -------------------------------------------------------------------------------
  private externalVbuf: { tex: GPUTexture; view: GPUTextureView } | undefined;
  /** RS_PRIMARY_EXT: rs_primary_ext replaces rs_primary (the primary hit read from a V-buffer texel). */
  primaryExt(): boolean { return !!this.perfFlagSet.RS_PRIMARY_EXT; }
  /** The frame's primary passes: rs_primary, or rs_primary_ext (+ rs_vtrace without a usable external V-buffer). */
  primaryPasses(): RsPassName[] {
    if (!this.primaryExt()) return ['rs_primary'];
    return this.extVsrc() ? ['rs_primary_ext'] : ['rs_vtrace', 'rs_primary_ext'];
  }
  /** Interactive (RS_PRIMARY_EXT): the renderer's M1 V-buffer of this frame (primId, bits(u), bits(v), bits(t)), written
   *  by its primary pass before the ReSTIR passes with the same camera ray. Undefined: the kernel traces its own
   *  (rs_vtrace). A change of source is a new prepare() key (rs_vtrace compiles only when needed). */
  setExternalVbuf(tex: GPUTexture | undefined): void {
    if (tex === this.externalVbuf?.tex) return;
    this.externalVbuf = tex ? { tex, view: tex.createView() } : undefined;
  }
  /** The external V-buffer when it can serve this kernel: interactive, one member, atlas-sized. */
  private extVsrc(): GPUTextureView | undefined {
    const e = this.externalVbuf, a = this.res?.alloc;
    if (!e || !this.external || !a || a.members !== 1 || e.tex.width !== a.atlasW || e.tex.height !== a.atlasH) return undefined;
    return e.view;
  }
  /** Every pipeline the current settings, light mode and view need is compiled (prepare() finished for them). A frame
   *  must not be encoded otherwise: setSettings() / setLightMode() switch the variant at once, the compile is async. */
  isPrepared(): boolean { return this.prepared === this.prepareKey(); }

  // ---------------------------------------------------------------------------------------------- state

  get resources(): RestirResources {
    if (!this.res) throw new Error('RestirKernel: setView() first');
    return this.res;
  }
  get currentView(): RestirView | undefined { return this.view; }
  /** Current light / env state changes (mirror PtFramePass). */
  setLights(lights: readonly LightData[]): LightsUpdate { return this.lights.update(lights); }
  setEnvironment(env: EnvGpuResources): void {
    this.env = env; this.g0Key = ''; this.envMapGen++;
    this.frameState.invalidate('env-map');               // a map swap is a config change (§2.5)
    applyEnvLighting(this.lights, env, this.envOptions);
  }
  envParamsChanged(): LightsUpdate { return applyEnvLighting(this.lights, this.env, this.envOptions); }
  setEnvOptions(o: Partial<PtEnvOptions> = {}): LightsUpdate {
    Object.assign(this.envOptions, o);
    const r = applyEnvLighting(this.lights, this.env, this.envOptions);
    if (r.reallocated) this.g0Key = '';
    return r;
  }

  /** M6 (MD9): switch the light mode (A / B / A′). A new pipeline variant: `await prepare()` before the next frame; the
   *  config hash changes, so the history resets. */
  setLightMode(m: LightMode): void {
    if (m === this.lightMode) return;
    this.lightMode = m;
    this.lights.setLightMode(m);
    this.variantLightMode = this.desiredVariantLightMode();
    this.g0Key = '';
    this.writeParams();
  }

  setSettings(s: Partial<RestirSettings>): void {
    const next = { ...this.settings, ...s };
    validateSettings(next);
    const realloc = numSlotsOf(next) !== numSlotsOf(this.settings) || next.temporal !== this.settings.temporal
      || next.risNee !== this.settings.risNee || next.dupmap !== this.settings.dupmap;
    this.settings = next;
    this.lights.deferred = !!next.temporal;
    if (!next.temporal && this.lights.hasPending) this.lights.commit();
    if (realloc && this.res) this.allocate();
    this.writeParams();
  }

  /** Sequential / ensemble validation view: the kernel owns the frame uniforms (member tile = width × height). */
  setView(v: RestirView): void {
    this.view = v;
    this.frameState.invalidate('view');
    this.external = undefined;
    const jm = v.jitterMode ?? JITTER_IID;
    this.frame.write({
      camera: v.camera, prevCamera: v.camera, width: v.width, height: v.height,
      frameIndex: 0, seedIndex: 0, runSeed: v.runSeed >>> 0, flags: 0, jitterMode: jm,
      jitter: jm === JITTER_NONE ? (v.jitter ?? [0.5, 0.5]) : [0.5, 0.5],
      origin: this.scene.origin, exposure: 1, time: 0, dt: 0, sceneDiag: boundsDiagonal(this.scene.scene.bounds),
    });
    this.g0Key = '';
    this.allocate();
    this.writeParams();
  }

  /** Interactive: the renderer owns the frame uniforms (camera, seedIndex, jitter, flags); E = 1. */
  setExternalFrame(frameUniforms: GPUBuffer, width: number, height: number): void {
    const same = this.res && this.res.alloc.atlasW === width && this.res.alloc.atlasH === height;
    this.external = { frameUniforms, width, height };
    this.view = undefined;
    this.g0Key = '';
    if (!same) this.allocate();
    this.writeParams();
  }

  private memberSize(): [number, number] {
    if (this.external) return [this.external.width, this.external.height];
    if (!this.view) throw new Error('RestirKernel: setView() first');
    return [this.view.width, this.view.height];
  }

  private allocate(): void {
    const [W, H] = this.memberSize();
    const E = this.view?.members ?? 1;
    const memberCols = Math.min(E, Math.floor(16384 / W));
    const atlasW = memberCols * W, atlasH = Math.ceil(E / memberCols) * H;
    if (E * W * H > 2 ** 22) throw new Error(`RestirKernel: E·W·H = ${E * W * H} > 2^22 (restir-api.md D15)`);
    const temporal = !!this.settings.temporal;
    const m6 = { dup: !!this.settings.dupmap && temporal, tileMembers: this.settings.risNee ? E : 0 };
    const a = {
      atlasW, atlasH, memberW: W, memberH: H, members: E, memberCols, slots: nsAlloc(numSlotsOf(this.settings), temporal),
      dump: !!this.o.instrumentation?.dumpCandidates, temporal, ...(m6.dup || m6.tileMembers > 0 ? { m6 } : {}),
    };
    const old = this.res;
    if (old && JSON.stringify(old.alloc) === JSON.stringify(a)) return;
    old?.destroy();
    // A new allocation holds no history (TD19: the kernel was (re)allocated ⇒ reset).
    this.roleH = -1; this.roleW = 0; this.lastFinal = 0; this.lastWasAdvanced = false; this.advanced = undefined;
    this.frameState.invalidate('reallocated');
    this.device.pushErrorScope('out-of-memory');
    this.res = new RestirResources(this.device, a, (name) => this.g2Layout(name, name === 'rs_finalize_frame' ? this.colorFormat : undefined));
    void this.device.popErrorScope().then((e) => { if (e) console.error(`RestirKernel: out of memory allocating the ${atlasW}×${atlasH} atlas: ${e.message}`); });
  }
  /** Colour format of the interactive finalize (set by RestirFramePass). */
  colorFormat: GPUTextureFormat | undefined;

  private writeParams(): void {
    if (!this.res) return;
    const s = this.settings, a = this.res.alloc;
    const bmin = this.scene.scene.bounds?.min ?? [0, 0, 0], bmax = this.scene.scene.bounds?.max ?? [1, 1, 1];
    const minExtent = Math.min(...[0, 1, 2].map((i) => Math.max(bmax[i] - bmin[i], 0)).filter((x) => x > 0), Infinity);
    let flags = restirFlags(s);
    if (a.members > 1) flags |= K.RSF_ENSEMBLE;
    if (this.external) flags |= K.RSF_INTERACTIVE;
    const m6On = !!a.m6;
    const P = a.atlasW * a.atlasH;
    this.device.queue.writeBuffer(this.params, 0, packRestirParams({
      atlasSize: [a.atlasW, a.atlasH], memberSize: [a.memberW, a.memberH], memberCols: a.memberCols, memberCount: a.members,
      maxBounces: s.maxBounces, flags, numTrees: s.trees, numSlots: numSlotsOf(s), numRounds: s.rounds, rrMinBounces: s.rrMinBounces,
      tau: s.tau, alphaMin: s.alphaMin, wScale: s.plant?.wScale ?? 1, crit2022MinDist: Number.isFinite(minExtent) ? 0.02 * minExtent : 0,
      pairTexSize: pairTexSizes(s), lightMode: this.lightMode === 'B' ? 2 : this.lightMode === 'A′' ? 1 : 0, memberBase: this.view?.memberBase ?? 0,
      boostSlots: s.temporal ? s.boostSlots : 0, tMode: tModeOf(s), cCap: s.cCap, tPlants: tPlantsOf(s),
      m6Base: m6On ? arenaM6Base(P, a.slots, !!a.temporal) : 0, risM: s.risNee ? s.risM : 0,
    }));
  }

  private ensureG0(): GPUBindGroup {
    const fb = this.external?.frameUniforms ?? this.frame.buffer;
    const key = `${this.lights.version}`;
    if (this.g0 && this.g0Key === key) return this.g0;
    this.g0 = this.device.createBindGroup({
      label: 'rs-g0', layout: this.layouts.g0,
      entries: [
        { binding: G0_BINDING.frame, resource: { buffer: fb } },
        ...envBindGroupEntries(this.env, ENV_BINDING_BASE),
        { binding: G0_BINDING.params, resource: { buffer: this.params } },
        { binding: G0_BINDING.lights, resource: { buffer: this.lights.params } },
        { binding: G0_BINDING.records, resource: { buffer: this.lights.records } },
        { binding: G0_BINDING.dispatch, resource: { buffer: this.ring, size: RS_DISPATCH_SIZE } },
        { binding: G0_BINDING.temporal, resource: { buffer: this.rsTemporal } },
      ],
    });
    this.g0Key = key;
    return this.g0;
  }

  // ---------------------------------------------------------------------------------------------- encoding

  /** Reset the RsDispatch ring cursor: call before encoding the units of each submit (§4.4). */
  beginSubmit(): void { this.ringCursor = 0; }

  /** Write one RsDispatch into the next ring slot (queue.writeBuffer, lands before the submit) → dynamic offset. */
  dispatchSlot(d: Partial<RsDispatchCpu>): number {
    if (this.ringCursor >= RS_DISPATCH_RING) throw new Error(`RestirKernel: more than ${RS_DISPATCH_RING} dispatches in one submit (call beginSubmit() per submit)`);
    const a = this.resources.alloc;
    const off = this.ringCursor++ * RS_DISPATCH_STRIDE;
    this.device.queue.writeBuffer(this.ring, off, packRsDispatch({
      t: 0, passId: 0, round: 0, treeBase: 0, treeCount: 0, flags: 0, rowBase: 0, rowEnd: a.atlasH, ...d,
    }));
    return off;
  }

  /**
   * Encode one compute pass: G0 (+ this dispatch's RsDispatch), G1, G2, G3. `work` = [x, y] workgroups, or an indirect
   * args buffer + offset. Per-pixel passes use perPixelWorkgroups(rowBase, rowEnd).
   */
  encodePass(enc: GPUCommandEncoder, name: RsPassName, pipeline: GPUComputePipeline, g2: GPUBindGroup, d: Partial<RsDispatchCpu>,
    work: [number, number] | { indirect: GPUBuffer; offset: number }, timestampWrites?: GPUComputePassTimestampWrites): void {
    const def = RS_PASSES[name];
    const off = this.dispatchSlot(d);
    const pass = enc.beginComputePass({ label: name, timestampWrites });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.ensureG0(), [off]);
    pass.setBindGroup(1, def.scene ? this.g1Scene : this.g1Empty);
    pass.setBindGroup(2, g2);
    pass.setBindGroup(3, def.debug && this.o.debug ? this.o.debug.bindGroup : this.g3Empty);
    if (Array.isArray(work)) pass.dispatchWorkgroups(work[0], work[1]);
    else pass.dispatchWorkgroupsIndirect(work.indirect, work.offset);
    pass.end();
  }

  /** Test / tool pipelines: layout [G0, G1 scene | empty, `g2`, empty] (no debug group; compose with DEBUG_NO_BINDINGS). */
  customLayout(g2: GPUBindGroupLayout, scene = true): GPUPipelineLayout {
    return this.device.createPipelineLayout({ label: 'rs-custom', bindGroupLayouts: [this.layouts.g0, scene ? this.layouts.g1Scene : this.layouts.empty, g2, this.layouts.empty] });
  }
  /** Defines of a custom pipeline (scene + env + lights + LUTs + the given pass bindings; DEBUG_NO_BINDINGS). */
  customDefines(extra: Defines, scene = true): Defines {
    return { ...restirCommonDefines(scene ? this.scene.defines(SCENE_GROUP) : undefined, false), ...this.m6Defines(), ...this.perfDefines(), ...extra };
  }
  /** Encode a custom pipeline with the kernel's G0 (+ one RsDispatch) and G1. */
  encodeCustom(enc: GPUCommandEncoder, pipeline: GPUComputePipeline, g2: GPUBindGroup, d: Partial<RsDispatchCpu>, work: [number, number], scene = true): void {
    const off = this.dispatchSlot(d);
    const pass = enc.beginComputePass({ label: pipeline.label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.ensureG0(), [off]);
    pass.setBindGroup(1, scene ? this.g1Scene : this.g1Empty);
    pass.setBindGroup(2, g2);
    pass.setBindGroup(3, this.g3Empty);
    pass.dispatchWorkgroups(work[0], work[1]);
    pass.end();
  }

  perPixelWorkgroups(rowBase: number, rowEnd: number): [number, number] {
    return [Math.ceil(this.resources.alloc.atlasW / 8), Math.ceil((rowEnd - rowBase) / 8)];
  }

  /** Row bands [rowBase, rowEnd) of the atlas for per-pixel units (multiples of 8 rows). */
  rowBands(): [number, number][] {
    const H = this.resources.alloc.atlasH;
    const band = this.rowBand > 0 ? Math.max(8, Math.floor(this.rowBand / 8) * 8) : H;
    const out: [number, number][] = [];
    for (let r = 0; r < H; r += band) out.push([r, Math.min(H, r + band)]);
    return out;
  }

  private readonly ready = new Map<string, GPUComputePipeline>();
  /** Synchronous access to a compiled standard pipeline (after `await pipeline(name)`). */
  pipelineSync(name: RsPassName, colorFormat?: GPUTextureFormat): GPUComputePipeline {
    const p = this.ready.get(`${name}:${colorFormat ?? ''}:${this.variantKey()}`);
    if (!p) throw new Error(`RestirKernel: pipeline ${name} not compiled (await kernel.pipeline('${name}'))`);
    return p;
  }

  // ---------------------------------------------------------------------------------------------- M5 frame state

  /** M5 (restir-temporal-api.md §3.8, §4.1): prepare frame `state.t` of a validation run / chain. Exactly one call per
   *  frame, before frameUnits(state.t): one light commit (TD4), the env record of t−1, prevCam, the G-buffer parity,
   *  the reservoir roles h/w, the config hash (TD19) and the RsTemporal uniform. */
  advance(state: RestirFrameState): RestirAdvance {
    if (!this.view) throw new Error('RestirKernel.advance: setView() first (interactive kernels use advanceInteractive)');
    const r = this.frameState.advance(this, state);
    this.applyAdvance(r);
    return r;
  }

  /** Interactive variant (T-D wires it in renderer.ts): the renderer owns the frame uniforms; `reset` forces a history
   *  reset (freeze seed, "reset history", config changes of the app). */
  advanceInteractive(frameUniforms: GPUBuffer, o: RestirInteractiveAdvance): RestirAdvance {
    if (!this.external || this.external.frameUniforms !== frameUniforms) throw new Error('RestirKernel.advanceInteractive: setExternalFrame(frameUniforms, …) first');
    const r = this.frameState.advanceInteractive(this, o);
    this.applyAdvance(r);
    return r;
  }

  private applyAdvance(r: RestirAdvance): void {
    const res = this.resources;
    // History = the previous frame's final buffer, only if that frame was itself an advanced frame of this allocation.
    const hist = r.histValid && this.lastWasAdvanced;
    if (r.histValid && !hist) throw new Error('RestirKernel.advance: history valid but the previous frame was not advanced (frame-state bug)');
    this.builtAdv = undefined;
    this.roleH = hist ? this.lastFinal : -1;
    this.roleW = this.roleH < 0 ? 0 : 1 - this.roleH;
    if (res.alloc.temporal) res.parity ^= 1;           // rs_primary of this frame writes the other G-buffer (§2.6)
    this.device.queue.writeBuffer(this.rsTemporal, 0, packRsTemporal(r.temporal));
    this.advanced = r;
  }

  /** The last frame built by frameUnits() was an advanced temporal frame (its final buffer can be history). */
  get previousFrameAdvanced(): boolean { return this.lastWasAdvanced; }

  /** Inputs of the config hash (§2.10) for the current kernel state. */
  configHashInput(jitterMode: number, envMapId: string): ConfigHashInput {
    const a = this.resources.alloc, s = this.settings;
    return {
      sceneGen: sceneGen(this.scene), atlas: [a.atlasW, a.atlasH], member: [a.memberW, a.memberH], members: a.members,
      memberBase: this.view?.memberBase ?? 0, lightMode: this.lights.lightMode, lightsLayout: this.lights.version,
      envMapGen: this.envMapGen, envMapId,
      importanceKey: envImportanceKey({ cap: this.envOptions.importanceCap, floors: this.envOptions.floors, plantPdfFromTargets: this.envOptions.plant === 'pdfFromTargets' }),
      envNee: this.envOptions.nee !== false && this.env.present, settings: { restir: s, env: this.envOptions }, flags: restirFlags(s), tMode: tModeOf(s), tPlants: tPlantsOf(s),
      jitterMode, misM: s.risNee ? s.risM : 1,
    };
  }

  /** Index of the buffer rs_initial writes (w; 0 with temporal off or after a reset). */
  resBase(): number { return this.roleW; }
  /** History buffer index h of the current frame (−1: no history). */
  historyIndex(): number { return this.roleH; }
  /** The buffer holding the frame's final reservoirs: res[(w + executed spatial rounds) % 2]. */
  finalResIndex(): number { return (this.roleW + this.lastRounds) % 2; }
  /** The RestirAdvance of the frame being prepared / built: set by advance(), readable by the stages while frameUnits()
   *  builds the frame's units and until the next advance() (undefined: a non-advanced frame ⇒ no temporal units;
   *  Changelog A-10). */
  get currentAdvance(): RestirAdvance | undefined { return this.advanced ?? this.builtAdv; }
  /** Alias of currentAdvance for the stages (T-B / T-C). */
  get frameAdvance(): RestirAdvance | undefined { return this.currentAdvance; }
  /** Packed EnvParams words of the env currently bound (the next frame's envPrev, §2.5). */
  envParamsWords(): ArrayBuffer { return packEnvParams(this.env.params, this.env.present); }
  get envResources(): EnvGpuResources { return this.env; }

  /** The work units of frame t (§4.1, §4.4). */
  frameUnits(t: number, out: RestirFrameOut): WorkUnit[] {
    const res = this.resources;
    const a = res.alloc, s = this.settings;
    const units: WorkUnit[] = [];
    const adv = this.advanced;
    this.advanced = undefined;
    this.builtAdv = adv;
    if (!adv) {                                           // not advanced: a reset frame (bitwise M4 schedule)
      this.roleH = -1; this.roleW = 0;
      if (this.lights.hasPending) this.lights.commit();   // staged edits of a non-advanced frame apply now
    }
    const w = this.roleW;
    const bands = this.rowBands();
    // perf2 WP-7e (RS_PRIMARY_EXT): rs_primary_ext reads the hit from the M1 V-buffer (or rs_vtrace's) instead of tracing
    for (const name of this.primaryPasses()) {
      const primary = this.pipelineSync(name);
      const g2 = name === 'rs_primary_ext' ? res.g2(name, 0, { vsrc: this.extVsrc() }) : res.g2(name);
      for (const [r0, r1] of bands) {
        units.push({
          label: `${name}[${r0}]`, costHint: a.atlasW * (r1 - r0),
          encode: (enc) => this.encodePass(enc, name, primary, g2, { t, passId: K.RS_PASS_PRIMARY, rowBase: r0, rowEnd: r1 }, this.perPixelWorkgroups(r0, r1)),
        });
      }
    }
    // M6 (MD4): the frame's light tiles (per member) before the path trees
    if (s.risNee) {
      const lt = this.pipelineSync('rs_light_tiles');
      units.push({
        label: 'rs_light_tiles', costHint: a.members * K6.RS_TILES * K6.RS_TILE_SIZE / 64,
        encode: (enc) => this.encodePass(enc, 'rs_light_tiles', lt, res.g2('rs_light_tiles'), { t, passId: K6.RS_PASS_LIGHT_TILES },
          [K6.RS_TILE_SIZE / 64, K6.RS_TILES * a.members]),
      });
    }
    const dump = !!this.o.instrumentation?.dumpCandidates;
    const initName: RsPassName = dump ? 'rs_initial_dump' : 'rs_initial';
    const initial = this.pipelineSync(initName);
    const chunk = this.treeChunk > 0 ? Math.min(this.treeChunk, s.trees) : s.trees;
    for (let tb = 0; tb < s.trees; tb += chunk) {
      const tc = Math.min(chunk, s.trees - tb);
      const flags = (tb === 0 ? K.RSD_FIRST_CHUNK : 0) | (tb + tc >= s.trees ? K.RSD_FINAL_CHUNK : 0);
      for (const [r0, r1] of bands) {
        units.push({
          label: `rs_initial[${tb}+${tc}][${r0}]`, costHint: a.atlasW * (r1 - r0) * tc * (s.maxBounces + 1),
          encode: (enc) => {
            if (dump && tb === 0 && r0 === 0) enc.clearBuffer(res.candDump!);
            this.encodePass(enc, initName, initial, res.g2(initName, w), { t, passId: K.RS_PASS_INITIAL, treeBase: tb, treeCount: tc, flags, rowBase: r0, rowEnd: r1 }, this.perPixelWorkgroups(r0, r1));
          },
        });
      }
    }
    // Temporal stage (TD15: before spatial), only on frames prepared by advance() with temporal on.
    if (adv && s.temporal && a.temporal) units.push(...this.temporal.frameUnits(this, t));
    const spatial = s.rounds > 0 ? this.spatial.frameUnits(this, t) : [];
    units.push(...spatial);
    const rounds = spatial.length > 0 ? s.rounds : 0;
    this.lastRounds = rounds;
    this.lastFinal = this.finalResIndex();
    this.lastWasAdvanced = !!adv && s.temporal && !!a.temporal;
    // M6 (MD10): the duplication counts of this frame's final reservoirs (read by the next frame's T1 at q′)
    if (s.dupmap && s.temporal && a.temporal) {
      const dm = this.pipelineSync('rs_dupmap');
      const fin = this.lastFinal;
      units.push({
        label: 'rs_dupmap', costHint: a.atlasW * a.atlasH,
        encode: (enc) => this.encodePass(enc, 'rs_dupmap', dm, res.g2('rs_dupmap', fin), { t, passId: K6.RS_PASS_DUPMAP },
          [Math.ceil(a.atlasW / 16), Math.ceil(a.atlasH / 16)]),
      });
    }
    const interactive = !!out.interactive;
    const fname: RsPassName = interactive ? 'rs_finalize_frame' : 'rs_finalize';
    if (!out.accum || !out.counters) throw new Error('RestirKernel.frameUnits: out.accum and out.counters are required');
    const colour = interactive && out.colorTarget ? this.colourView(out.colorTarget) : undefined;
    const fin = this.pipelineSync(fname, interactive ? this.colorFormat : undefined);
    const fflags = interactive ? ((out.interactive!.accumulate ? K.RSD_ACCUMULATE : 0) | (out.interactive!.advanced ? K.RSD_ADVANCED : 0) | this.noDisplayBit(out.interactive!.noDisplay)) : 0;
    const g2 = res.g2(fname, this.lastFinal, { accum: out.accum, counters: out.counters, colour });
    for (const [r0, r1] of bands) {
      units.push({
        label: `rs_finalize[${r0}]`, costHint: a.atlasW * (r1 - r0),
        encode: (enc) => this.encodePass(enc, fname, fin, g2, { t, passId: 0, round: rounds, flags: fflags, rowBase: r0, rowEnd: r1 }, this.perPixelWorkgroups(r0, r1)),
      });
    }
    if (a.members > 1 || out.ensemble) units.push(...this.ensemble.frameUnits(this, t));
    return units;
  }

  /** perf2 WP-7c: RSD_NO_DISPLAY (finalize.wgsl, RS_SKIP_DISPLAY only; 0 without the flag). */
  noDisplayBit(noDisplay: boolean | undefined): number { return noDisplay && this.perfFlagSet.RS_SKIP_DISPLAY ? K_RSD_NO_DISPLAY : 0; }

  /** One view per interactive colour target (M5 T-D, restir-temporal-api.md Changelog D-4): a fresh createView() per
   *  frame gave the finalize bind group a new cache key every frame (the resources' group cache grew without bound).
   *  A new target (resize, format change) is a new texture, hence a new view; RestirFramePass.setTargets drops the
   *  finalize groups of the old one (forgetExternalGroups). */
  colourView(tex: GPUTexture): GPUTextureView {
    let v = this.colourViews.get(tex);
    if (!v) { v = tex.createView(); this.colourViews.set(tex, v); }
    return v;
  }
  private readonly colourViews = new WeakMap<GPUTexture, GPUTextureView>();

  /** Read (and optionally clear) the arena header: queue headers, RSC_* counters, the SC histogram, f_r. */
  async readCounters(reset: boolean): Promise<RestirCounters> {
    const res = this.resources;
    const raw = new Uint32Array(await readBuffer(this.device, res.arena, RestirResources.arenaHeaderBytes));
    if (reset) {
      const enc = this.device.createCommandEncoder({ label: 'rs-counters-reset' });
      enc.clearBuffer(res.arena, 0, RestirResources.arenaHeaderBytes);
      this.device.queue.submit([enc.finish()]);
    }
    const queues = [0, 1, 2, 3].map((q) => { const h = queueHdr(q); return { counter: raw[h.counter], n: raw[h.n], capacity: raw[h.capacity], overflow: raw[h.overflow] }; });
    const rsc = Object.fromEntries(Object.entries(RSC).map(([k, w]) => [k, raw[w]])) as Record<RscName, number>;
    const codes = Array.from(raw.subarray(K.RSC_CODE_BASE, K.RSC_CODE_BASE + 16));
    return { queues, rsc, codes, fr: rsc.accepted > 0 ? rsc.queued / rsc.accepted : 0, raw };
  }

  /** Read back a reservoir buffer ('final' = the output of the last frame's last stage). */
  async readReservoirs(which: 'final' | 0 | 1): Promise<Uint32Array> {
    const res = this.resources;
    const idx = which === 'final' ? this.lastFinal : which;
    const raw = new Uint32Array(await readBuffer(this.device, res.res[idx], res.pixels * RES_BYTES));
    if (this.o.resLayout !== 'soa') return raw;
    const n = res.pixels, aos = new Uint32Array(raw.length);   // plane-major → record-major (M8 P-7)
    for (let p = 0; p < RES_PLANES; p++) for (let i = 0; i < n; i++) aos.set(raw.subarray((p * n + i) * 4, (p * n + i) * 4 + 4), (i * RES_PLANES + p) * 4);
    return aos;
  }

  /** M5 (§3.8): the arena's tState + sfxOut region as u32 words, starting at tState (decode with layout.ts
   *  decodeTStateLocal / decodeSfxLocal). */
  async readTemporalState(): Promise<Uint32Array> {
    const res = this.resources;
    if (!res.alloc.temporal) throw new Error('RestirKernel.readTemporalState: temporal is off');
    const aw = arenaWords(res.pixels, res.alloc.slots);
    const off = ARENA_HDR_BYTES + 4 * aw.tState;
    const bytes = 4 * (aw.end - aw.tState);
    const staging = this.device.createBuffer({ label: 'rs-tstate-copy', size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const enc = this.device.createCommandEncoder({ label: 'rs-tstate-copy' });
    enc.copyBufferToBuffer(res.arena, off, staging, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    const out = new Uint32Array(await readBuffer(this.device, staging, bytes));
    staging.destroy();
    return out;
  }

  /** Read back the candidate dump (instrumentation.dumpCandidates). */
  async readCandidateDump(): Promise<Uint32Array> {
    const d = this.resources.candDump;
    if (!d) throw new Error('RestirKernel: create with instrumentation.dumpCandidates');
    return new Uint32Array(await readBuffer(this.device, d, d.size));
  }

  destroy(): void {
    this.spatial.destroy?.();
    this.ensemble.destroy?.();
    this.res?.destroy();
    this.frame.destroy();
    this.params.destroy();
    this.ring.destroy();
    this.rsTemporal.destroy();
    this.temporal.destroy?.();
    this.lights.destroy();
  }
}

/** perf2 WP-7c: RsDispatch.flags bit of finalize.wgsl's RSD_NO_DISPLAY (RS_SKIP_DISPLAY text only). */
export const K_RSD_NO_DISPLAY = 128;

/** `vbuf` (perf2 WP-7e, RS_PRIMARY_EXT): the renderer's M1 V-buffer, written by its primary pass earlier in the frame. */
export interface RestirFrameTargets { width: number; height: number; color: GPUTexture; frameUniforms: GPUBuffer; vbuf?: GPUTexture }

/** Interactive ReSTIR pass (renderer mode 'restir'; mirrors PtFramePass). One submit per frame: encode() resets the
 *  RsDispatch ring. t = frame.seedIndex (RSF_INTERACTIVE). */
export class RestirFramePass {
  readonly counters: GPUBuffer;
  private accum: GPUBuffer | undefined;
  private targets: RestirFrameTargets | undefined;

  constructor(readonly kernel: RestirKernel, readonly colorFormat: GPUTextureFormat) {
    kernel.colorFormat = colorFormat;
    this.counters = kernel.device.createBuffer({ label: 'rs-frame-counters', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  }

  get lights(): LightsGpu { return this.kernel.lights; }
  get settings(): RestirSettings { return this.kernel.settings; }
  setLights(lights: readonly LightData[]): LightsUpdate { return this.kernel.setLights(lights); }
  setEnvironment(env: EnvGpuResources): void { this.kernel.setEnvironment(env); }
  envParamsChanged(): LightsUpdate { return this.kernel.envParamsChanged(); }
  setEnvOptions(o: Partial<PtEnvOptions> = {}): LightsUpdate { return this.kernel.setEnvOptions(o); }
  /** Settings changes that need a new stage (rounds 0 → > 0) require `await prepare()`. */
  setSettings(s: Partial<RestirSettings>): void { this.kernel.setSettings(s); }
  prepare(): Promise<void> { return this.kernel.prepare(); }
  /** The current variant is compiled: encode() / encodeHold() may run (else `await prepare()` first). */
  get ready(): boolean { return this.kernel.isPrepared(); }

  setTargets(t: RestirFrameTargets): void {
    const old = this.targets;
    if (!old || old.width !== t.width || old.height !== t.height) {
      this.accum?.destroy();
      this.accum = this.kernel.device.createBuffer({ label: 'rs-frame-accum', size: t.width * t.height * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    }
    this.targets = t;
    this.kernel.setExternalFrame(t.frameUniforms, t.width, t.height);
    this.kernel.setExternalVbuf(t.vbuf);
    this.kernel.resources.forgetExternalGroups();
  }

  /** Encode after the renderer's primary pass (frame uniforms already hold this frame's camera / seed / flags). */
  encode(encoder: GPUCommandEncoder, frame: { advanced: boolean; accumulate: boolean; noDisplay?: boolean }, timestampWrites?: GPUComputePassTimestampWrites): boolean {
    const t = this.targets;
    if (!t || !this.accum) return false;
    this.kernel.beginSubmit();
    const units = this.kernel.frameUnits(0, { accum: this.accum, counters: this.counters, colorTarget: t.color, interactive: frame });
    // No timestamp writes on (or around) the ReSTIR passes (restir-api.md Changelog C8): with timestampWrites on the
    // bracketing passes, Chrome 154 / Metal silently dropped the frame's ReSTIR work in some page loads (no WebGPU error;
    // arena header 0, q0 capacity 0, image ≈ L1; app smoke, plain Cornell and Cornell + HDRI alike). The parameter is
    // kept for signature stability and ignored.
    void timestampWrites;
    for (const u of units) u.encode(encoder);
    return true;
  }

  /** M5 paused frame with temporal on (restir-temporal-api.md TD20, Changelog D-2; T-D): no ReSTIR pass runs (the
   *  history must never be consumed twice); only rs_finalize_frame re-displays the last frame's estimate (rsShade /
   *  res[finalResIndex()], rsL1 of the current parity: untouched since) without adding a sample. */
  encodeHold(encoder: GPUCommandEncoder, frame: { accumulate: boolean; noDisplay?: boolean }): boolean {
    const t = this.targets, k = this.kernel;
    if (!t || !this.accum) return false;
    k.beginSubmit();
    const a = k.resources.alloc;
    const g2 = k.resources.g2('rs_finalize_frame', k.finalResIndex(), { accum: this.accum, counters: this.counters, colour: k.colourView(t.color) });
    k.encodePass(encoder, 'rs_finalize_frame', k.pipelineSync('rs_finalize_frame', this.colorFormat), g2,
      { t: 0, passId: 0, round: k.lastRounds, flags: (frame.accumulate ? K.RSD_ACCUMULATE : 0) | k.noDisplayBit(frame.noDisplay), rowBase: 0, rowEnd: a.atlasH }, k.perPixelWorkgroups(0, a.atlasH));
    return true;
  }

  destroy(): void {
    this.counters.destroy();
    this.accum?.destroy();
    this.kernel.destroy();
  }
}

export { JITTER_IID, JITTER_NONE };
