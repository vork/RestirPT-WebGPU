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
// M4 (WP-D): renderMode 'restir' drives RestirKernel.interactive() after `primary` (the M1 primary keeps running for
// picking, depth and the G-buffer views; restir-api.md D12/R10). restirMode picks the settings preset (PLAN §3 modes:
// ReSTIR-unbiased, ReSTIR-2022-criteria, Offline; 'initial' = rung 3.1 without spatial reuse). The ReSTIR debug views
// (400–499), the probe inspector records and the arena HUD come from render/restir/debug.ts; they encode work only while
// a ReSTIR view or the probe is active. Light modes: M4 shipped Mode A only (restir-api D1); M6 lifted that (restir-m6-api
// MD9): the kernel takes A, B and A′ (setLightMode), and the app default is B.
// M5 (T-D; restir-temporal-api.md TD19–TD21, §2.10, §3.7, Changelog D-2): restirMode adds ReSTIR-interactive
// (interactive preset: temporal, RR, boost 3) next to ReSTIR-unbiased (the `full` preset: temporal, RR off, no boost);
// options.temporal switches temporal reuse for every mode. With temporal on, every ADVANCED frame calls
// RestirKernel.advanceInteractive once (the one light commit of the frame, env record, config hash, history reset on
// `resetTemporal`), then encodes the frame; a paused frame (advanced = false) encodes no ReSTIR pass (TD20: the
// history must never be consumed twice) and only re-displays the last ReSTIR frame (RestirFramePass.encodeHold).
// Light / env-parameter edits and camera motion never reset the temporal history (they are handled by the refresh
// and q′); a config change resets through the config hash, a scene / map swap / resize through the kernel.
// M5.5 (docs/decisions/denoiser.md): the A-SVGF-lite denoiser runs after the ReSTIR finalize (input rsFrame / rsL1, the
// ReSTIR temporal gradient from tState) or after the PT (its 1-spp sample: the PT's accumulation is off meanwhile) and
// overwrites the colour target (PLAN §3 step 7: accumulate | denoise). Default on in ReSTIR-interactive, off elsewhere,
// forced off in ReSTIR-unbiased (DN4); the toggle is remembered per mode. A held (paused) frame re-encodes the last
// denoiser plan (DN9); afterSubmit() runs the HUD timing in a separate submit (Q3).
import { composeWgsl, createCheckedShaderModule } from '../gpu/wgsl-composer.ts';
import { shaderSources } from '../shaders/index.ts';
import type { EnvironmentData, SceneData } from '../scene/types.ts';
import type { DebugResources, DebugViewDef } from './debug-views.ts';
import {
  createEnvResources, destroyEnvResources, envBindGroupEntries, envBindGroupLayoutEntries, envDefines, envImportanceKey, envMemoryReport,
  writeEnvParams, type EnvGpuResources, type EnvParamsCpu,
} from './env-gpu.ts';
import { EnvDebugPass, isEnvDebugView } from './env-debug.ts';
import { ShadingDebugPass, isShadingDebugView } from './shading-debug.ts';
import { buildEnvImportanceAsync } from '../scene/env/env-importance-client.ts';
import { ENV_IMPORTANCE_CAP_INTERACTIVE, envImportanceBytes } from '../scene/env/env-importance.ts';
import { SceneGpu, type BvhBuilder, type BvhKind } from './scene-gpu.ts';
import type { LightData } from '../scene/types.ts';
import type { LightsUpdate } from './lights-gpu.ts';
import { PtFramePass } from './pt-kernel.ts';
import type { LightMode } from './lights-gpu.ts';
import type { TexturePathMode } from './textures-gpu.ts';
import { RestirKernel, type RestirAdvance, type RestirFramePass } from './restir/kernel.ts';
import { DEFAULT_RESTIR_SETTINGS, RESTIR_PRESETS, type RestirSettings } from './restir/presets.ts';
import { RestirDebugPass, RestirHud } from './restir/debug.ts';
import { Denoiser, type DenoiseFrame } from './denoise/denoiser.ts';
import { denoiseModeKey, denoiserAllowed, denoiserDefault, type DenoiserSettings } from './denoise/layout.ts';
import { arenaWords, RS_WGSL_CONSTS } from './restir/layout.ts';

/** App ReSTIR modes (PLAN §3, restir-temporal-api.md §3.7): ReSTIR-interactive (interactive preset: temporal, RR, boost
 *  3), ReSTIR-unbiased (the `full` preset: temporal, RR off, no boost), ReSTIR-2022-criteria (interactive preset with the
 *  2022 criteria), Offline, initial only (no spatial reuse: rung 3.1, or rung 3.3 with temporal on). */
export type RestirAppMode = 'interactive' | 'unbiased' | 'criteria2022' | 'offline' | 'initial';
export const RESTIR_APP_MODES: Record<RestirAppMode, string> = {
  interactive: 'ReSTIR-interactive (S 1, 1 round × 3 + boost 3, σ 16, RIS-NEE, dual MV, dup map, RR)',
  unbiased: 'ReSTIR-unbiased (S 1, 1 round × 3, σ 16, RIS-NEE)', criteria2022: 'ReSTIR-2022-criteria', offline: 'Offline (S 32, 3 rounds × 6, σ 16, RIS-NEE)',
  initial: 'initial only (rung 3.1 / 3.3)',
};
/** Settings of an app ReSTIR mode (maxBounces and temporal from the renderer options). Complete, not a partial: the
 *  kernel merges setSettings() onto its current settings, so a partial let a field the new mode's preset leaves unset
 *  carry over from the previous mode (interactive's boostSlots 3 + offline's 6 slots = 9 > RS_MAX_SLOTS threw on a
 *  switch to Offline). A freshly compiled kernel starts from DEFAULT_RESTIR_SETTINGS, so first-compile settings are
 *  unchanged. */
export function restirAppSettings(mode: RestirAppMode, maxBounces: number, temporal = true, features: Partial<RestirSettings> = {}): RestirSettings {
  const base = mode === 'offline' ? RESTIR_PRESETS['offline-m6'] : mode === 'unbiased' ? RESTIR_PRESETS['full-m6']
    : mode === 'initial' ? { ...RESTIR_PRESETS.interactive, rounds: 0 } : RESTIR_PRESETS.interactive;
  return { ...DEFAULT_RESTIR_SETTINGS, ...base, criteria: mode === 'criteria2022' ? '2022' : 'enhanced', maxBounces, temporal, ...features };
}

/** M6 feature toggles of the app (restir-m6-api.md MD13): overrides of the mode's preset (empty = the preset's). */
export type RestirFeatureOverrides = Partial<Pick<RestirSettings, 'pairing' | 'risNee' | 'dualMv' | 'dupmap'>>;

/** M8 (m8-perf.md §3): 'auto' picks the CWBVH from this many triangles on. */
export const CWBVH_AUTO_MIN_TRIS = 65536;
export function resolveBvhKind(k: BvhKind | 'auto', scene: SceneData): BvhKind {
  return k === 'auto' ? (scene.geometry.indices.length / 3 >= CWBVH_AUTO_MIN_TRIS ? 'cwbvh' : 'bvh2') : k;
}

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
  /** M8 (docs/decisions/m8-perf.md §3): acceleration structure, BVH2 (default: every validation path), CWBVH, or 'auto'
   *  (CWBVH from CWBVH_AUTO_MIN_TRIS triangles on: measured faster on Sponza, slower on small scenes). */
  bvhKind: BvhKind | 'auto';
  accumulate: boolean;
  thrTau: number;
  /** 'pt': reference path tracer beauty (M3a); 'albedo': the M1 placeholder (albedo on hits, env on misses). */
  renderMode: 'pt' | 'albedo' | 'restir';
  /** renderMode 'restir': the settings preset (PLAN §3 modes). */
  restirMode: RestirAppMode;
  /** renderMode 'restir': temporal reuse (M5; every mode). */
  temporal: boolean;
  /** renderMode 'restir': M6 feature toggles over the mode's preset (σ 16 pairing, RIS-NEE, dual MVs, duplication map). */
  restirFeatures: RestirFeatureOverrides;
  /** PT: Cycles max_bounces N. */
  maxBounces: number;
  /** PT: Russian roulette (unbiased; off by default, plan §2 rule 11). */
  rr: boolean;
  /** PT and ReSTIR: plan §1.4 light mode: 'A' NEE-only analytic lights, 'B' pass-through + MIS (area lights visible in
   *  mirrors / through smooth glass; the product default once Gate 3.11 passed, restir-m6-api.md MD9 / Q4), 'A′'
   *  pass-through after delta lobes only. */
  lightMode: LightMode;
  /** Env NEE (M3c; ≡ Cycles world sampling_method AUTOMATIC). Off = BSDF-only env (NONE). */
  envNee: boolean;
  /** Env importance resolution cap (W_m ≤ cap; plan §1.4b: 2048 interactively, 4096 in validation). */
  envImportanceCap: number;
  /** M5.5: the denoiser toggle of the current mode (denoiser.md §8; remembered per mode in denoiseByMode). */
  denoise: boolean;
  /** M8 (m8-perf.md): kernel options of the interactive ReSTIR (A/B measurements; default the interactive kernel's). */
  restirKernel?: { resLayout?: 'aos' | 'soa'; modeBNeedsAreaLights?: boolean };
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
  /** The app's debug resources (group 3): ReSTIR mode binds them in its own passes and runs the rs_debug passes. */
  debug?: DebugResources;
}

interface RestirState {
  pass: RestirFramePass; dbg?: RestirDebugPass; hud: RestirHud;
  /** The RestirAdvance of the last advanced temporal frame (HUD, smoke). */
  lastAdvance?: RestirAdvance;
  /** Temporal frames encoded / paused frames held since the pass was created (smoke: TD20). */
  temporalFrames: number; heldFrames: number;
  /** The compile of the current pipeline variant (light mode / M6 feature toggles / stages; never rejects). */
  variant?: { key: string; p: Promise<void> };
  /** restirError of the last failed variant compile (cleared when a later variant compiles). */
  variantError?: string;
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
  /** Env sampling debug views (M3c), compiled on first use. */
  envDebug?: EnvDebugPass;
  envDebugPending?: Promise<EnvDebugPass | undefined>;
  shadingDebug?: ShadingDebugPass;
  shadingDebugPending?: Promise<ShadingDebugPass | undefined>;
  /** ReSTIR (renderMode 'restir'), compiled on first use. */
  rs?: RestirState;
  rsPending?: Promise<RestirState | undefined>;
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
  readonly options: RendererOptions = {
    textureMode: 'validation', watertight: true, bvhKind: 'bvh2', accumulate: true, thrTau: THR_TAU, renderMode: 'albedo', restirMode: 'interactive', temporal: true, restirFeatures: {}, maxBounces: 3, rr: false,
    lightMode: 'A', envNee: true, envImportanceCap: ENV_IMPORTANCE_CAP_INTERACTIVE, denoise: false,
  };
  /** M5.5: denoiser toggle per mode (denoiseModeKey); a mode without an entry starts at denoiserDefault. */
  readonly denoiseByMode: Record<string, boolean> = {};
  /** M5.5: the denoiser (compiled on first use) and its last compile error. */
  denoiser: Denoiser | undefined;
  denoiserError: string | undefined;
  private denoiserPending: Promise<Denoiser | undefined> | undefined;
  private denoiseModeKey = '';
  /** The denoiser ran in the last encoded frame. */
  denoisedLastFrame = false;
  private advancedFrames = 0;
  /** RestirAdvance of the frame being encoded (undefined: no temporal advance this frame). */
  private frameAdv: RestirAdvance | undefined;
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
  private lights: readonly LightData[] | undefined;
  /** Last ReSTIR compile error (HUD). */
  restirError: string | undefined;

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
        textureMode: this.options.textureMode, watertight: this.options.watertight, bvhKind: resolveBvhKind(this.options.bvhKind, scene), buildBvh: this.ctx.buildBvh,
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
      old?.envDebug?.destroy();
      old?.shadingDebug?.destroy();
      destroyRestir(old?.rs);
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
    const wtChanged = (o.watertight !== undefined && o.watertight !== this.options.watertight) || (o.bvhKind !== undefined && o.bvhKind !== this.options.bvhKind);
    Object.assign(this.options, o);
    this.state?.pt?.setSettings({ maxBounces: this.options.maxBounces, rr: this.options.rr });
    if (this.state?.pt && this.state.pt.lights.lightMode !== this.options.lightMode) this.state.pt.lights.setLightMode(this.options.lightMode);
    const rs = this.state?.rs;
    if (rs) {
      // M6: the light mode and the feature toggles are pipeline variants (MD1, MD9): recompiled lazily by prepare()
      // until it is compiled the frames show the PT beauty (encodeRestir), never a half-switched kernel
      rs.pass.kernel.setLightMode(this.options.lightMode);
      rs.pass.setSettings(this.restirSettings());
      await this.prepareRestirVariant(rs);
    }
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
          maxBounces: this.options.maxBounces, rr: this.options.rr, lightMode: this.options.lightMode, features: this.ctx.features, wgslLanguageFeatures: this.ctx.wgslLanguageFeatures,
          env: { nee: this.options.envNee, importanceCap: this.options.envImportanceCap },
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
    this.lights = lights;
    const r = this.state?.rs?.pass.setLights(lights);
    return this.state?.pt?.setLights(lights) ?? r;
  }

  /** Compile the ReSTIR pass for `state` (never throws; errors go to restirError and the PT beauty stays). */
  private compileRestir(state: SceneState, colorFormat: GPUTextureFormat): Promise<RestirState | undefined> {
    if (state.rsPending && state.rs?.pass.colorFormat === colorFormat) return state.rsPending;
    if (state.rsPending && !state.rs) return state.rsPending;
    const p = (async () => {
      try {
        const debug = this.ctx.debug;
        const pass = await RestirKernel.interactive(this.device, state.gpu, this.env, colorFormat, {
          settings: this.restirSettings(), lightMode: this.options.lightMode, debug,
          env: { nee: this.options.envNee, importanceCap: this.options.envImportanceCap },
          features: this.ctx.features, wgslLanguageFeatures: this.ctx.wgslLanguageFeatures, ...this.options.restirKernel,
        });
        const dbg = debug ? await RestirDebugPass.create(pass.kernel, debug) : undefined;
        if (this.lights) pass.setLights(this.lights);
        const tg = this.targets;
        if (tg) pass.setTargets({ width: tg.t.width, height: tg.t.height, color: tg.t.color, frameUniforms: tg.t.frameUniforms });
        destroyRestir(state.rs);
        state.rs = { pass, dbg, hud: new RestirHud(this.device), temporalFrames: 0, heldFrames: 0 };
        this.restirError = undefined;
        return state.rs;
      } catch (e) {
        this.restirError = `ReSTIR: ${e instanceof Error ? e.message : String(e)}`;
        console.error(e);
        return undefined;
      }
    })();
    state.rsPending = p;
    return p;
  }

  /** Compile the current pipeline variant of `rs` (once per variant; never throws: a failure goes to restirError and
   *  the PT beauty stays, as for compileRestir). */
  private prepareRestirVariant(rs: RestirState): Promise<void> {
    const k = rs.pass.kernel;
    if (k.isPrepared()) return Promise.resolve();
    const key = k.prepareKey();
    if (rs.variant?.key === key) return rs.variant.p;
    const p = rs.pass.prepare().then(() => {
      if (rs.variantError !== undefined && this.restirError === rs.variantError) this.restirError = undefined;
      rs.variantError = undefined;
    }, (e: unknown) => {
      rs.variantError = this.restirError = `ReSTIR: ${e instanceof Error ? e.message : String(e)}`;
      console.error(e);
    });
    rs.variant = { key, p };
    return p;
  }

  /** The ReSTIR settings of the current app options (mode preset ⊕ M6 feature toggles). */
  restirSettings(): Partial<RestirSettings> {
    return restirAppSettings(this.options.restirMode, this.options.maxBounces, this.options.temporal, this.options.restirFeatures);
  }

  /** The interactive ReSTIR pass (undefined until renderMode 'restir' compiled it). */
  get restir(): RestirFramePass | undefined { return this.state?.rs?.pass; }
  get restirHud(): RestirHud | undefined { return this.state?.rs?.hud; }
  /** M5: the last advanced temporal frame's RestirAdvance and the temporal / held frame counts (HUD, app smoke). */
  get restirTemporal(): { lastAdvance?: RestirAdvance; temporalFrames: number; heldFrames: number } | undefined {
    const rs = this.state?.rs;
    return rs ? { lastAdvance: rs.lastAdvance, temporalFrames: rs.temporalFrames, heldFrames: rs.heldFrames } : undefined;
  }
  /** Compile ReSTIR now (e.g. before switching the mode in a test). */
  async prepareRestir(): Promise<RestirFramePass | undefined> {
    const s = this.state;
    if (!s || !this.targets) return undefined;
    return (await this.compileRestir(s, this.targets.t.colorFormat))?.pass;
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
    const next = await createEnvResources(this.device, env, 'env', { mode: this.options.textureMode });
    await this.attachImportance(next);   // Worker-built tables (plan §1.4b) before the PT sees the env
    const old = this.env;
    this.env = next;
    this.state?.pt?.setEnvironment(next);
    this.state?.rs?.pass.setEnvironment(next);
    if (this.targets) this.targets.frameGroup = this.frameGroup(this.targets.t);
    destroyEnvResources(old);
  }

  /** Env rotation/strength/tint/visibility (UI or timeline). Strength/tint rebuild the pmf (P(env)); rotation never does. */
  setEnvParams(p: Partial<EnvParamsCpu>): void {
    writeEnvParams(this.device, this.env, p);
    this.state?.pt?.setEnvOptions();
    this.state?.rs?.pass.setEnvOptions();
  }

  /** Build the importance tables of `res` for the current cap in the Worker (no-op when cached). */
  private async attachImportance(res: EnvGpuResources): Promise<void> {
    if (!res.present || !res.source) return;
    const o = { cap: this.options.envImportanceCap };
    const key = envImportanceKey(o);
    if (res.importanceKey === key) return;
    const s = res.source;
    res.importance = await buildEnvImportanceAsync(s.texels, s.width, s.height, o);
    res.importanceKey = key;
  }

  /** Env NEE on/off and the importance resolution (a config change: the caller resets the accumulation). */
  async setEnvSampling(o: { nee?: boolean; importanceCap?: number }): Promise<void> {
    if (o.importanceCap !== undefined && o.importanceCap !== this.options.envImportanceCap) {
      this.options.envImportanceCap = o.importanceCap;
      await this.attachImportance(this.env);
    }
    if (o.nee !== undefined) this.options.envNee = o.nee;
    this.state?.pt?.setEnvOptions({ nee: this.options.envNee, importanceCap: this.options.envImportanceCap });
    this.state?.rs?.pass.setEnvOptions({ nee: this.options.envNee, importanceCap: this.options.envImportanceCap });
  }

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
    if (s?.rs && s.rs.pass.colorFormat !== t.colorFormat) void this.compileRestir(s, t.colorFormat);
    else s?.rs?.pass.setTargets({ width: t.width, height: t.height, color: t.color, frameUniforms: t.frameUniforms });
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
    frame: {
      advanced: boolean; debugMode: number; debugGroup: GPUBindGroup; /** skip the PT pass (primary timing) */ noPt?: boolean;
      /** M5: reset the ReSTIR temporal history this frame (explicit reset, freeze seed / frame / history; TD19, TD20). */
      resetTemporal?: boolean;
      /** M5.5: the progressive accumulation restarts this frame (FrameContext.resetHistory): resets the denoiser history in
       *  modes without a ReSTIR temporal gradient (PT, temporal off; denoiser.md §6). */
      resetHistory?: boolean;
    },
    timestamps?: () => GPUComputePassTimestampWrites | undefined,
    ptTimestamps?: () => GPUComputePassTimestampWrites | undefined,
    rsTimestamps?: () => GPUComputePassTimestampWrites | undefined,
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
    let restirDone = false;
    this.frameAdv = undefined;
    if (this.options.renderMode === 'restir' && !frame.noPt && !isBvhStatsView(frame.debugMode)) {
      if (!s.rs) void this.compileRestir(s, tg.t.colorFormat);
      else restirDone = this.encodeRestir(encoder, s.rs, frame.advanced, !!frame.resetTemporal, tg.t.frameUniforms, rsTimestamps);
    }
    let ptDone = false;
    const dn = this.denoiseWanted() && !frame.noPt && !isBvhStatsView(frame.debugMode);
    if ((this.options.renderMode === 'pt' || (this.options.renderMode === 'restir' && !restirDone)) && s.pt && !frame.noPt && !isBvhStatsView(frame.debugMode)) {
      // M5.5: the denoiser needs the PT's 1-spp frame sample (accumulate | denoise, PLAN §3 step 7)
      s.pt.encode(encoder, { advanced: frame.advanced, accumulate: this.options.accumulate && !(dn && this.options.renderMode === 'pt' && this.denoiser) }, ptTimestamps?.());
      ptDone = true;
    }
    this.denoisedLastFrame = false;
    if (dn && (restirDone || (ptDone && this.options.renderMode === 'pt'))) {
      this.denoisedLastFrame = this.encodeDenoiser(encoder, s, tg, frame, restirDone ? 'restir' : 'pt');
    }
    if (frame.advanced) this.advancedFrames++;
    if (isEnvDebugView(frame.debugMode) && s.pt) {
      if (!s.envDebug) void this.compileEnvDebug(s);
      else s.envDebug.encode(encoder, { mode: frame.debugMode, env: this.env, lights: s.pt.lights, frameUniforms: tg.t.frameUniforms, width: tg.t.width, height: tg.t.height, debugGroup: frame.debugGroup, reset: false });
    }
    if (isShadingDebugView(frame.debugMode)) {   // M7 shading-normal views (m7-api.md §7)
      if (!s.shadingDebug) void this.compileShadingDebug(s);
      else s.shadingDebug.encode(encoder, { mode: frame.debugMode, frameUniforms: tg.t.frameUniforms, width: tg.t.width, height: tg.t.height, debugGroup: frame.debugGroup });
    }
    return true;
  }

  /**
   * ReSTIR frame: counters clear + code-view fill, (M5: advanceInteractive), the kernel's passes, shift views, arena
   * header copy (one encoder). M5 temporal: a paused frame only re-displays the last frame (TD20); `resetTemporal`
   * resets the history (and the HUD error totals).
   */
  private encodeRestir(encoder: GPUCommandEncoder, rs: RestirState, advanced: boolean, resetTemporal: boolean, frameUniforms: GPUBuffer,
    ts?: () => GPUComputePassTimestampWrites | undefined): boolean {
    const k = rs.pass.kernel;
    // A light-mode / feature toggle switches the kernel's pipeline variant at once and compiles it asynchronously
    // (setOptions awaits it, but frames keep coming): until it is compiled the PT beauty is shown (before advancing the
    // temporal state, so the history is not consumed by a frame that never ran).
    k.syncLightModeVariant();   // M8 P-4: the Mode-A text while no rect / disk light exists (a frame boundary)
    if (!rs.pass.ready) { void this.prepareRestirVariant(rs); return false; }
    let arena: GPUBuffer;
    try { arena = k.resources.arena; } catch { return false; }
    const temporal = !!k.settings.temporal;
    if (temporal && !advanced) {
      if (!rs.pass.encodeHold(encoder, { accumulate: this.options.accumulate })) return false;
      rs.heldFrames++;
      return true;
    }
    let adv: RestirAdvance | undefined;
    if (temporal) {
      adv = k.advanceInteractive(frameUniforms, { reset: resetTemporal });
      if (resetTemporal) rs.hud.resetTotals();
    }
    rs.hud.encodeBegin(encoder, arena);
    rs.dbg?.encodeBegin(encoder);
    const ok = rs.pass.encode(encoder, { advanced, accumulate: this.options.accumulate }, ts?.());
    if (!ok) return false;
    rs.dbg?.encodeViews(encoder, { rounds: k.lastRounds });
    rs.hud.encodeEnd(encoder, k.resources.arena, adv);
    if (adv) { rs.lastAdvance = adv; rs.temporalFrames++; }
    this.frameAdv = adv;
    return true;
  }

  // ---- M5.5 denoiser (docs/decisions/denoiser.md) -------------------------------------------------------------------

  /** Follow mode switches: each mode keeps its own toggle (default on in ReSTIR-interactive only). */
  private syncDenoiseMode(): void {
    const o = this.options;
    const key = denoiseModeKey(o.renderMode, o.restirMode);
    if (key === this.denoiseModeKey) return;
    this.denoiseModeKey = key;
    o.denoise = this.denoiseByMode[key] ?? denoiserDefault(o.renderMode, o.restirMode);
  }
  /** The denoiser may run in the current mode (never in ReSTIR-unbiased: DN4). */
  get denoiseAllowed(): boolean { return denoiserAllowed(this.options.renderMode, this.options.restirMode); }
  /** Toggle the denoiser of the current mode (remembered per mode). */
  setDenoise(on: boolean): void {
    this.syncDenoiseMode();
    this.options.denoise = on;
    this.denoiseByMode[this.denoiseModeKey] = on;
  }
  /** The toggle of the current mode is on and the mode allows the denoiser. */
  denoiseWanted(): boolean {
    this.syncDenoiseMode();
    return this.options.denoise && this.denoiseAllowed;
  }
  /** Denoiser settings (iterations, α_min, gradient ramp, edge stops); applied on the next frame. */
  setDenoiserSettings(s: Partial<DenoiserSettings>): void {
    Object.assign(this.pendingDenoiserSettings, s);
    this.denoiser?.setSettings(this.pendingDenoiserSettings);
  }
  readonly pendingDenoiserSettings: Partial<DenoiserSettings> = {};

  private compileDenoiser(colorFormat: GPUTextureFormat): Promise<Denoiser | undefined> {
    if (this.denoiserPending) return this.denoiserPending;
    const p = (async () => {
      try {
        const d = await Denoiser.create(this.device, { debugLayout: this.ctx.debugLayout, colorFormat, features: this.ctx.features, wgslLanguageFeatures: this.ctx.wgslLanguageFeatures });
        d.setSettings(this.pendingDenoiserSettings);
        this.denoiser?.destroy();
        this.denoiser = d;
        this.denoiserError = undefined;
        return d;
      } catch (e) {
        this.denoiserError = `denoiser: ${e instanceof Error ? e.message : String(e)}`;
        console.error(e);
        return undefined;
      } finally {
        this.denoiserPending = undefined;
      }
    })();
    this.denoiserPending = p;
    return p;
  }

  /** Compile the denoiser now (tests, harness). */
  async prepareDenoiser(): Promise<Denoiser | undefined> {
    const tg = this.targets;
    if (!tg) return undefined;
    if (this.denoiser?.colorFormat === tg.t.colorFormat) return this.denoiser;
    return this.compileDenoiser(tg.t.colorFormat);
  }

  private encodeDenoiser(enc: GPUCommandEncoder, s: SceneState, tg: TargetState,
    frame: { advanced: boolean; debugGroup: GPUBindGroup; resetTemporal?: boolean; resetHistory?: boolean }, kind: 'restir' | 'pt'): boolean {
    const d = this.denoiser;
    if (!d || d.colorFormat !== tg.t.colorFormat) { void this.compileDenoiser(tg.t.colorFormat); return false; }
    d.resize(tg.t.width, tg.t.height);
    const f: DenoiseFrame = {
      kind, advanced: frame.advanced, reset: !!frame.resetHistory, frameUniforms: tg.t.frameUniforms, gbuf: tg.gbuf, colour: tg.t.color, debugGroup: frame.debugGroup,
    };
    if (kind === 'restir') {
      const k = s.rs!.pass.kernel;
      const res = k.resources;
      const adv = this.frameAdv;
      const temporal = !!k.settings.temporal && !!res.alloc.temporal;
      const fl = adv?.flags ?? 0;
      f.radiance = res.frameTex;
      f.l1 = res.l1;
      f.restir = {
        arena: res.arena, resW: res.res[k.resBase()], resFinal: res.res[k.finalResIndex()],
        tsBase: RS_WGSL_CONSTS.RS_ARENA_HDR_WORDS + arenaWords(res.pixels, res.alloc.slots).tState, resPlanes: k.resPlaneStride,
        gradient: !!adv && adv.histValid && temporal && k.settings.temporalMis === 'contribution',
        lightingChanged: (fl & RS_WGSL_CONSTS.TF_LIGHTS_SAME) === 0 || (fl & RS_WGSL_CONSTS.TF_ENV_SAME) === 0,
        inverse: k.lastRounds <= 1,
      };
      // ReSTIR with temporal reuse: the denoiser history resets with the ReSTIR history (config, scene, resize, env map,
      // explicit reset, freeze); light / env edits are the gradient's job (denoiser.md §6).
      if (temporal) f.reset = (!!adv && !adv.histValid) || !!frame.resetTemporal;
    }
    return d.encode(enc, f);
  }

  /** M5.5: call right after the frame's queue.submit: every 30th advanced frame the HUD timing re-runs the frame's
   *  denoiser passes in a separate submit with timestamp writes (DN9, Q3). */
  afterSubmit(): void {
    const d = this.denoiser;
    if (!d || !this.denoisedLastFrame || this.advancedFrames % 30 !== 0) return;
    void d.time(1);
  }

  private compileShadingDebug(s: SceneState): Promise<ShadingDebugPass | undefined> {
    s.shadingDebugPending ??= ShadingDebugPass.create(this.device, s.gpu, this.ctx.debugLayout, { features: this.ctx.features, wgslLanguageFeatures: this.ctx.wgslLanguageFeatures })
      .then((p) => { s.shadingDebug = p; return p; }, (e: unknown) => { this.lastError = `shading debug: ${e instanceof Error ? e.message : String(e)}`; console.error(e); return undefined; });
    return s.shadingDebugPending;
  }

  private compileEnvDebug(s: SceneState): Promise<EnvDebugPass | undefined> {
    s.envDebugPending ??= EnvDebugPass.create(this.device, s.gpu, this.ctx.debugLayout, { features: this.ctx.features, wgslLanguageFeatures: this.ctx.wgslLanguageFeatures })
      .then((p) => { s.envDebug = p; return p; }, (e: unknown) => { this.lastError = `env debug: ${e instanceof Error ? e.message : String(e)}`; console.error(e); return undefined; });
    return s.envDebugPending;
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
      out.push(`isect ${g.watertight ? 'Woop (watertight)' : 'Möller–Trumbore'} ${g.bvhKind === 'cwbvh' ? 'CWBVH' : 'BVH2'}  textures ${this.options.textureMode} ${mib(g.stats.textureBytes)} MiB  geom ${mib(g.stats.geometryBytes)} MiB`);
    }
    out.push(envMemoryReport(this.env).text + (this.env.present ? `  γ ${(this.env.params.rotationZ * 180 / Math.PI).toFixed(1)}° s ${this.env.params.strength}` : ''));
    const pt = this.state?.pt;
    if (this.env.present && pt) {
      const l = pt.lights.summary();
      const t = pt.lights.state.env;
      out.push(l.env
        ? `env NEE on  P(env) ${l.pEnv.toFixed(4)}  importance ${t?.Wm}x${t?.Hm} (${((t ? envImportanceBytes(t) : 0) / 2 ** 20).toFixed(1)} MiB, built ${t?.buildMs.toFixed(0)} ms)`
        : 'env NEE off (BSDF-only env, ≡ Cycles sampling_method NONE)');
      const ed = this.state?.envDebug;
      if (ed) {
        void ed.updateChi2(pt.lights);
        if (ed.chi2) out.push(`env splat χ² ${ed.chi2.chi2.toFixed(0)} / dof ${ed.chi2.dof}  p ${ed.chi2.p.toExponential(2)}  (${(ed.chi2.total / 1e6).toFixed(0)} M samples)`);
      }
    }
    if (this.options.renderMode === 'pt') {
      const l = pt?.lights.summary();
      out.push(pt
        ? `PT max_bounces ${pt.settings.maxBounces}${pt.settings.rr ? ' RR' : ''}  lights ${l!.analytic} analytic + ${l!.emissiveTriangles} emissive tris (Mode ${pt.lights.lightMode})`
        : 'PT: compiling (albedo placeholder shown)');
    }
    if (this.options.renderMode === 'restir') {
      const rs = this.state?.rs;
      if (!rs) out.push(this.restirError ?? 'ReSTIR: compiling (PT shown)');
      else {
        const st = rs.pass.settings;
        out.push(`ReSTIR ${this.options.restirMode}: S ${st.trees}  rounds ${st.rounds} × ${st.slots}${st.temporal && st.boostSlots ? ` + boost ${st.boostSlots}` : ''} slots  R ${st.diskRadius}  ${st.criteria}${st.rr ? ' RR' : ''}  max_bounces ${st.maxBounces}`
          + `  temporal ${st.temporal ? `on (c_cap ${st.cCap}, ${st.temporalMis}${st.refresh === 'e2' ? ', E2' : ''}; ${rs.temporalFrames} frames, ${rs.heldFrames} held)` : 'off'}`);
        out.push(`  M6: Mode ${rs.pass.kernel.lightMode}  pairing ${st.pairing === 'gauss' ? `gauss σ ${st.pairSigma}` : `disk R ${st.diskRadius}`}  RIS-NEE ${st.risNee ? `M ${st.risM}` : 'off'}`
          + `  dual MV ${st.dualMv ? 'on' : 'off'}  dup map ${st.dupmap && st.temporal ? 'on (biased)' : 'off'}`);
        out.push(...rs.hud.lines());
      }
    }
    out.push(this.denoiserHudLine());
    if (this.loading) out.push('renderer: uploading / compiling ...');
    if (this.lastError) out.push(`renderer error: ${this.lastError.split('\n')[0]}`);
    return out;
  }

  /** HUD line of the denoiser (state, settings, GPU ms of the separate timing submits). */
  denoiserHudLine(): string {
    if (!this.denoiseAllowed) return `denoiser: off (not available in ${this.options.renderMode === 'restir' ? `ReSTIR-${this.options.restirMode}` : this.options.renderMode})`;
    if (!this.denoiseWanted()) return 'denoiser: off';
    const d = this.denoiser;
    if (!d) return this.denoiserError ?? 'denoiser: compiling ...';
    const st = d.settings;
    const t = d.timingAverage();
    const fl = d.flags;
    const grad = d.kind === 'restir' ? ((fl & 32) ? `gradient ${(fl & 2) ? 'on (lights / env changed)' : 'gated (static lighting)'}` : 'gradient n/a') : 'no gradient (PT)';
    return `denoiser A-SVGF-lite: ${st.iterations} iters, α_min ${st.alphaMin}, resolve ${st.resolve ? 'on' : 'off'}, ${grad}, ${d.framesSinceReset} frames since reset`
      + (t ? `  GPU ${t.totalMs.toFixed(3)} ms (${t.passes.map((p) => `${p.name.replace('dn_', '')} ${p.ms.toFixed(2)}`).join(', ')})` : '');
  }

  destroy(): void {
    this.state?.pt?.destroy();
    this.state?.envDebug?.destroy();
    this.state?.shadingDebug?.destroy();
    destroyRestir(this.state?.rs);
    this.denoiser?.destroy();
    this.denoiser = undefined;
    this.state?.gpu.destroy();
    if (this.targets) { this.targets.gbuf.destroy(); this.targets.accum.destroy(); this.targets.vbuf.destroy(); }
    destroyEnvResources(this.env);
    this.params.destroy();
  }
}

function destroyRestir(rs: RestirState | undefined): void {
  if (!rs) return;
  rs.hud.destroy();
  rs.pass.destroy();
}
