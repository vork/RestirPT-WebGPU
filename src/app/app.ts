// App shell (plan §5 M1): canvas + DPR handling, internal-resolution targets, frame loop, fly camera, UI, HUD,
// probe, timestamps, overlay and the loading plumbing. The integrator wires the renderer through AppHooks:
//   onSceneLoaded(scene)        upload geometry/BVH/materials (use app.origin for recentring)
//   renderFrame(encoder, ctx)   encode the render passes into ctx.targets (color, depth, debug group 3)
//   registerDebugView(def)      add debug views without recompiling (debugMode/debugTap are uniforms)
import { createGpuContext, type GpuContext } from '../core/gpu/device.ts';
import {
  DebugResources, DebugViewRegistry, DBGC, applyViewDefaults, defaultDebugSettings,
  type DebugSettings, type DebugViewDef,
} from '../core/render/debug-views.ts';
import {
  FRAME_FREEZE_FRAME, FRAME_FREEZE_SEED, FRAME_PAUSED, FRAME_RESET_HISTORY, FrameUniformBuffer, JITTER_IID, JITTER_NONE,
  JITTER_R2, boundsDiagonal, computeRenderOrigin, r2Jitter, type CameraState, type FrameUniformInput, type JitterMode,
} from '../core/render/frame-uniforms.ts';
import { Overlay } from '../core/render/overlay.ts';
import { Presenter, internalResolution, type PresentSettings, type ResolutionPreset } from '../core/render/present.ts';
import { ProbeRing, pickPixel, type ProbeFrame } from '../core/render/probe.ts';
import { GpuTimestamps, RollingAverage } from '../core/render/timestamps.ts';
import type { EnvironmentData, SceneData } from '../core/scene/types.ts';
import { DEG, type Vec3 } from './camera-math.ts';
import { CameraTrack, FlyCamera, clampDt } from './fly-camera.ts';
import { FlyControls, type PickEvent } from './fly-controls.ts';
import {
  installDropTarget, sourcesFromFiles, sourcesFromQuery, type SceneLoader, type SceneSource,
} from './loader.ts';
import { TestPattern } from './test-pattern.ts';
import { Hud } from './ui/hud.ts';
import { LoadingOverlay } from './ui/loading.ts';
import { buildPanel, type PanelHandle } from './ui/panel.ts';
import { ProbePanel } from './ui/probe-panel.ts';

export type ColorFormat = 'rgba32float' | 'rgba16float';

/** Render targets at internal resolution. A new object is created on every resize/format change. */
export interface RenderTargets {
  width: number;
  height: number;
  /** Linear radiance the renderer writes (STORAGE_BINDING | TEXTURE_BINDING | COPY_SRC | COPY_DST). */
  color: GPUTexture;
  colorFormat: ColorFormat;
  /** Linear depth (r32float, view-space -z_cam; 0 = background) from the primary pass: overlay depth test. */
  depth: GPUTexture;
  /** Group-3 debug binding (params + DebugBuffer), debugOut texture. */
  debug: DebugResources;
  /** FrameUniforms buffer (frame.wgsl, @group(0) @binding(0) by default). */
  frameUniforms: GPUBuffer;
}

export interface FrameContext {
  device: GPUDevice;
  targets: RenderTargets;
  uniforms: FrameUniformInput;
  /** True when frameIndex advanced this frame (false while paused without a step). */
  advanced: boolean;
  resetHistory: boolean;
  scene: SceneData | undefined;
  debug: DebugSettings;
  /** timestampWrites for a pass (undefined when unsupported or out of slots). */
  timestamps: (passName: string) => GPUComputePassTimestampWrites | undefined;
}

export interface EnvParams {
  url: string;
  strength: number;
  rotationDeg: number;
  tint: { r: number; g: number; b: number };
  visibleToCamera: boolean;
}

export interface AppHooks {
  onSceneLoaded?(scene: SceneData, app: App): void | Promise<void>;
  onEnvironmentLoaded?(env: EnvironmentData, app: App): void | Promise<void>;
  onEnvironmentParams?(p: EnvParams, app: App): void;
  onTargetsResized?(targets: RenderTargets, app: App): void;
  renderFrame?(encoder: GPUCommandEncoder, ctx: FrameContext): void;
  onResetHistory?(app: App): void;
  onPick?(e: PickEvent & { pixel: [number, number] }, app: App): void;
  /** Extra HUD lines (e.g. replay fraction) appended every HUD refresh. */
  hudLines?(app: App): string[];
}

export interface BeforeFrameInfo {
  now: number;
  /** Wall-clock seconds since the previous frame (unclamped) and the clamped camera dt. */
  rawDt: number;
  dt: number;
  /** The frame index advances this frame (not paused, or a single step). */
  advance: boolean;
}

export interface RenderSettings {
  resolution: ResolutionPreset;
  colorFormat: ColorFormat;
  jitter: 'none' | 'iid' | 'r2';
  paused: boolean;
  freezeSeed: boolean;
  freezeFrame: boolean;
  overlay: boolean;
}

export interface AppOptions {
  canvas: HTMLCanvasElement;
  /** Element that hosts the HUD, panels and the loading overlay (defaults to document.body). */
  ui?: HTMLElement;
  gpu?: GpuContext;
  loader?: SceneLoader;
  hooks?: AppHooks;
  /** Fixed run seed (validation); default random, or ?seed=N. */
  runSeed?: number;
}

const JITTER_CODE: Record<RenderSettings['jitter'], JitterMode> = { none: JITTER_NONE, iid: JITTER_IID, r2: JITTER_R2 };

export class App {
  readonly device: GPUDevice;
  readonly canvas: HTMLCanvasElement;
  readonly presenter: Presenter;
  readonly debug: DebugResources;
  readonly timestamps: GpuTimestamps;
  readonly probe: ProbeRing;
  readonly overlay: Overlay;
  readonly frameUniforms: FrameUniformBuffer;
  readonly camera = new FlyCamera();
  readonly controls: FlyControls;
  readonly hud: Hud;
  readonly loading: LoadingOverlay;
  readonly probePanel: ProbePanel;
  panel: PanelHandle | undefined;
  hooks: AppHooks;
  loader: SceneLoader | undefined;

  targets!: RenderTargets;
  scene: SceneData | undefined;
  env: EnvironmentData | undefined;
  origin: [number, number, number] = [0, 0, 0];
  sceneDiag = 10;
  runSeed: number;

  readonly render: RenderSettings = {
    resolution: '540p', colorFormat: 'rgba32float', jitter: 'r2', paused: false, freezeSeed: false, freezeFrame: false, overlay: true,
  };
  readonly present: PresentSettings = { exposureEV: 0, tonemap: 'standard', filter: 'bilinear', highlightNonFinite: true };
  readonly debugSettings: DebugSettings = defaultDebugSettings();
  readonly envParams: EnvParams = { url: '', strength: 1, rotationDeg: 0, tint: { r: 1, g: 1, b: 1 }, visibleToCamera: true };
  hudVisible = true;

  frameIndex = 0;
  seedIndex = 0;
  frameCounter = 0;
  time = 0;
  private stepPending = false;
  private resetPending = true;
  private lastT: number | undefined;
  private lastCamera: CameraState | undefined;
  private readonly cpuFrame = new RollingAverage();
  private lastHud = 0;
  private testPattern: TestPattern;
  private canvasSize: [number, number] = [0, 0];
  private running = false;
  private loadAbort: AbortController | undefined;
  private fatal: string | undefined;
  /** Counter totals shown in the HUD (NaN/Inf restart at every history reset). */
  readonly totals = { nan: 0, inf: 0, bvhOverflow: 0, bvhItercap: 0, queueOverflow: 0, probeOverflow: 0 };
  /** Same counters since start, never reset, plus how many frames were actually read back (gates use these). */
  readonly runTotals = { nan: 0, inf: 0, bvhOverflow: 0, bvhItercap: 0, queueOverflow: 0, probeOverflow: 0, framesRead: 0 };
  axesHandle: number | undefined;
  /** Called at the start of every frame, before the camera update and the uniforms (editor, timeline). */
  readonly beforeFrame = new Set<(f: BeforeFrameInfo) => void>();
  /** While true the frame loop submits no GPU work (e.g. a Cycles reference render holds the GPU, plan §7.5). */
  suspended = false;

  static async create(opts: AppOptions): Promise<App> {
    const gpu = opts.gpu ?? await createGpuContext(navigator.gpu, { label: 'app' });
    const app = new App(gpu, opts);
    await app.init();
    return app;
  }

  private constructor(readonly gpu: GpuContext, opts: AppOptions) {
    this.device = gpu.device;
    this.canvas = opts.canvas;
    this.hooks = opts.hooks ?? {};
    this.loader = opts.loader;
    const q = new URLSearchParams(location.search);
    const seedParam = q.get('seed');
    this.runSeed = opts.runSeed ?? (seedParam !== null && /^\d+$/.test(seedParam) ? Number(seedParam) >>> 0 : crypto.getRandomValues(new Uint32Array(1))[0]);
    const res = q.get('res');
    if (res === '540p' || res === '720p' || res === '1080p' || res === 'native') this.render.resolution = res;
    if (q.get('hud') === '0') this.hudVisible = false;

    this.presenter = new Presenter(this.device, this.canvas);
    this.debug = new DebugResources(this.device, new DebugViewRegistry());
    this.timestamps = new GpuTimestamps(this.device);
    this.probe = new ProbeRing(this.device, 3);
    this.overlay = new Overlay(this.device, this.presenter.format);
    this.frameUniforms = new FrameUniformBuffer(this.device);
    this.testPattern = new TestPattern(this.device, this.debug.layout, this.render.colorFormat);
    this.controls = new FlyControls(this.canvas, this.camera);

    const ui = opts.ui ?? document.body;
    this.hud = new Hud(ui);
    this.hud.setVisible(this.hudVisible);
    this.loading = new LoadingOverlay(ui);
    this.probePanel = new ProbePanel(ui);
    this.probe.listeners.add((f) => this.onProbeFrame(f));
  }

  private async init(): Promise<void> {
    await Promise.all([this.presenter.init(), this.debug.init(), this.overlay.init(), this.testPattern.init()]);
    this.device.lost.then((info) => { this.fatal = `Device lost (${info.reason}): ${info.message}`; this.loading.start('WebGPU'); this.loading.error(this.fatal); });
    this.observeCanvas();
    this.resizeTargets(true);
    this.setDefaultCamera();
    this.installAxes();
    this.controls.onPick.add((e) => this.handlePick(e));
    this.controls.extraKeys = (e) => this.handleKey(e);
    installDropTarget(document.body, (files) => { void this.loadFiles(files); }, (on) => document.body.classList.toggle('drop-hover', on));
    this.panel = buildPanel(this);
    const qs = sourcesFromQuery(location.search, location.href);
    for (const s of qs.ignored) console.warn(`ignored URL parameter: ${s}`);
    if (qs.scene) void this.loadScene(qs.scene).then(() => (qs.env ? this.loadEnvironment(qs.env) : undefined));
    else if (qs.env) void this.loadEnvironment(qs.env);
  }

  // ---- public API -----------------------------------------------------------------------------------------------

  start(): void {
    if (this.running) return;
    this.running = true;
    requestAnimationFrame(this.tick);
  }

  registerDebugView(def: DebugViewDef): DebugViewDef { return this.debug.registry.register(def); }

  setHooks(h: AppHooks): void { this.hooks = { ...this.hooks, ...h }; }

  /** Request a history reset (fires hooks.onResetHistory and FRAME_RESET_HISTORY on the next frame). */
  resetHistory(): void { this.resetPending = true; }
  step(): void { this.stepPending = true; }
  setPaused(p: boolean): void { this.render.paused = p; this.camera.frozen = p; this.panel?.refresh(); }

  setResolution(p: ResolutionPreset): void { this.render.resolution = p; this.resizeTargets(); }
  setColorFormat(f: ColorFormat): void {
    if (f === this.render.colorFormat) return;
    this.render.colorFormat = f;
    this.testPattern = new TestPattern(this.device, this.debug.layout, f);
    void this.testPattern.init();
    this.resizeTargets(true);
  }

  selectDebugView(id: number): void {
    this.debugSettings.mode = id;
    applyViewDefaults(this.debugSettings, this.debug.registry.get(id));
    this.panel?.refresh();
  }

  /** Recorded or loaded camera track -> JSON download (tracks.camera). */
  exportTrack(track: CameraTrack | undefined = this.lastTrack): void {
    if (!track) return;
    const blob = new Blob([JSON.stringify({ camera: track.toJSON() }, null, 1)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${(this.scene?.name ?? 'scene').replace(/[^\w.-]+/g, '_')}.camera-track.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  lastTrack: CameraTrack | undefined;

  toggleRecording(): void {
    if (this.camera.recording) this.lastTrack = this.camera.stopRecording();
    else this.camera.startRecording();
    this.panel?.refresh();
  }
  playTrack(loop = false): void { if (this.lastTrack) this.camera.play(this.lastTrack, loop); }
  async importTrack(file: File): Promise<void> {
    const j = JSON.parse(await file.text()) as { camera?: unknown };
    this.lastTrack = CameraTrack.fromJSON(j.camera ?? j);
  }

  /** Jump to SceneData.cameras[i] (exact pose, keeps roll until the user rotates). */
  useFileCamera(i = 0): boolean {
    const c = this.scene?.cameras[i];
    if (!c) return false;
    this.camera.setFromMatrix(c.matrix, c.yfov);
    this.panel?.refresh();
    return true;
  }

  async loadFiles(files: File[]): Promise<void> {
    const s = sourcesFromFiles(files);
    if (s.ignored.length) console.warn('ignored dropped files:', s.ignored);
    if (s.scene) await this.loadScene(s.scene);
    if (s.env) await this.loadEnvironment(s.env);
    if (!s.scene && !s.env) { this.loading.start('Drop'); this.loading.error(`No scene (.glb/.gltf/.usd*) or environment (.hdr/.exr) in: ${files.map((f) => f.name).join(', ')}`); }
  }

  async loadScene(src: SceneSource): Promise<void> {
    this.loading.start(`Loading ${src.name}`);
    if (!this.loader) { this.loading.error('No scene loader is wired into the app (integrator: pass AppOptions.loader).'); return; }
    this.loadAbort?.abort();
    const ac = (this.loadAbort = new AbortController());
    try {
      const t0 = performance.now();
      const scene = await this.loader.loadScene(src, (e) => this.loading.progress(e.fraction, e.stage), ac.signal);
      if (ac.signal.aborted) return;
      this.loading.warn(...scene.warnings);
      this.setScene(scene);
      this.loading.progress(0.95, 'uploading to GPU...');
      await this.hooks.onSceneLoaded?.(scene, this);
      this.loading.done(`${scene.name}: ${scene.geometry.indices.length / 3} triangles in ${((performance.now() - t0) / 1000).toFixed(2)} s`);
    } catch (e) {
      if (ac.signal.aborted) return;
      console.error(e);
      this.loading.error(`Failed to load ${src.name}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async loadEnvironment(src: SceneSource): Promise<void> {
    this.loading.start(`Loading environment ${src.name}`);
    if (!this.loader?.loadEnvironment) { this.loading.error('No environment loader is wired into the app.'); return; }
    try {
      const env = await this.loader.loadEnvironment(src, (e) => this.loading.progress(e.fraction, e.stage), new AbortController().signal);
      this.env = env;
      this.envParams.url = src.kind === 'url' ? src.url : src.name;
      this.envParams.strength = env.strength;
      this.envParams.rotationDeg = env.rotationZ / DEG;
      this.envParams.tint = { r: env.tint[0], g: env.tint[1], b: env.tint[2] };
      this.envParams.visibleToCamera = env.visibleToCamera;
      await this.hooks.onEnvironmentLoaded?.(env, this);
      this.resetHistory();
      this.panel?.refresh();
      this.loading.done(`${env.name}: ${env.width}x${env.height}`);
    } catch (e) {
      console.error(e);
      this.loading.error(`Failed to load ${src.name}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  envParamsChanged(): void { this.hooks.onEnvironmentParams?.(this.envParams, this); }

  /** Adopt a scene: recentring origin, camera speed scale, initial/file camera, history reset. */
  setScene(scene: SceneData): void {
    this.scene = scene;
    this.origin = computeRenderOrigin(scene.bounds);
    this.sceneDiag = boundsDiagonal(scene.bounds);
    this.overlay.setOrigin(this.origin);
    this.camera.sceneDiag = this.sceneDiag;
    this.camera.wheelSteps = 0;
    if (this.axesHandle !== undefined) { this.overlay.remove(this.axesHandle); this.axesHandle = undefined; }
    if (!this.useFileCamera(0)) this.camera.frameBounds(scene.bounds.min as Vec3, scene.bounds.max as Vec3);
    this.camera.initialPose = this.camera.pose();
    this.lastCamera = undefined;
    this.resetHistory();
    this.panel?.refresh();
  }

  // ---- internals ------------------------------------------------------------------------------------------------

  private setDefaultCamera(): void {
    this.camera.sceneDiag = this.sceneDiag;
    this.camera.lookAt([0, 1.6, 6], [0, 1, 0]);
    this.camera.initialPose = this.camera.pose();
  }

  private installAxes(): void {
    // world axes at the origin (X red, Y green, Z blue), depth-tested: demonstrates the overlay without a scene
    const L = 3;
    const pts = [0, 0, 0, L, 0, 0, 0, 0, 0, 0, L, 0, 0, 0, 0, 0, 0, L];
    const col = [1, 0.2, 0.2, 1, 1, 0.2, 0.2, 1, 0.2, 1, 0.2, 1, 0.2, 1, 0.2, 1, 0.3, 0.5, 1, 1, 0.3, 0.5, 1, 1];
    this.axesHandle = this.overlay.addLines(pts, col, false);
  }

  private observeCanvas(): void {
    const apply = (w: number, h: number) => {
      const max = this.device.limits.maxTextureDimension2D;
      const cw = Math.max(1, Math.min(max, Math.round(w)));
      const ch = Math.max(1, Math.min(max, Math.round(h)));
      if (cw === this.canvasSize[0] && ch === this.canvasSize[1]) return;
      this.canvasSize = [cw, ch];
      this.canvas.width = cw;
      this.canvas.height = ch;
      this.resizeTargets();
    };
    const ro = new ResizeObserver((entries) => {
      const e = entries[0];
      const dp = e.devicePixelContentBoxSize?.[0];
      const cb = e.contentBoxSize[0];
      const dpr = devicePixelRatio;
      // Exact device pixels when available. DevTools/Playwright DPR emulation reports the unscaled box there, so
      // fall back to CSS size x DPR when the two disagree by more than rounding.
      if (dp && Math.abs(dp.inlineSize - cb.inlineSize * dpr) <= dpr + 1) apply(dp.inlineSize, dp.blockSize);
      else apply(cb.inlineSize * dpr, cb.blockSize * dpr);
    });
    try { ro.observe(this.canvas, { box: 'device-pixel-content-box' }); } catch { ro.observe(this.canvas); }
    const r = this.canvas.getBoundingClientRect();
    apply(r.width * devicePixelRatio, r.height * devicePixelRatio);
  }

  private resizeTargets(force = false): void {
    const [cw, ch] = this.canvasSize[0] > 0 ? this.canvasSize : [960, 540];
    const [w, h] = internalResolution(this.render.resolution, cw, ch);
    const t = this.targets;
    if (!force && t && t.width === w && t.height === h && t.colorFormat === this.render.colorFormat) return;
    t?.color.destroy();
    t?.depth.destroy();
    this.debug.resize(w, h);
    const color = this.device.createTexture({
      label: 'color',
      size: [w, h],
      format: this.render.colorFormat,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    });
    const depth = this.device.createTexture({
      label: 'linear-depth',
      size: [w, h],
      format: 'r32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    });
    this.targets = { width: w, height: h, color, colorFormat: this.render.colorFormat, depth, debug: this.debug, frameUniforms: this.frameUniforms.buffer };
    this.overlay.setDepth(depth, 'view-z');
    const pp = this.debugSettings.probePixel;
    this.debugSettings.probePixel = [Math.min(pp[0], w - 1), Math.min(pp[1], h - 1)];
    this.resetHistory();
    this.hooks.onTargetsResized?.(this.targets, this);
  }

  private handlePick(e: PickEvent): void {
    const pixel = pickPixel(e.x, e.y, e.width, e.height, this.targets.width, this.targets.height);
    if (e.alt || this.panel?.probeMode()) {
      this.debugSettings.probePixel = pixel;
      this.debugSettings.probeEnabled = true;
      this.probePanel.setVisible(true);
      this.panel?.refresh();
    }
    this.hooks.onPick?.({ ...e, pixel }, this);
  }

  private handleKey(e: KeyboardEvent): boolean {
    if (e.ctrlKey || e.altKey) return false;
    switch (e.code) {
      case 'KeyP': this.setPaused(!this.render.paused); return true;
      case 'Period': this.step(); return true;
      case 'KeyH': this.hudVisible = !this.hudVisible; this.hud.setVisible(this.hudVisible); this.panel?.refresh(); return true;
      case 'KeyR': if (e.shiftKey) { this.resetHistory(); return true; } return false;
      default: return false;
    }
  }

  private onProbeFrame(f: ProbeFrame): void {
    const c = f.counters;
    this.totals.nan += c[DBGC.NAN];
    this.totals.inf += c[DBGC.INF];
    this.totals.bvhOverflow += c[DBGC.BVH_OVERFLOW];
    this.totals.bvhItercap += c[DBGC.BVH_ITERCAP];
    this.totals.queueOverflow += c[DBGC.QUEUE_OVERFLOW];
    this.totals.probeOverflow += c[DBGC.PROBE_OVERFLOW];
    const r = this.runTotals;
    r.nan += c[DBGC.NAN]; r.inf += c[DBGC.INF];
    r.bvhOverflow += c[DBGC.BVH_OVERFLOW]; r.bvhItercap += c[DBGC.BVH_ITERCAP];
    r.queueOverflow += c[DBGC.QUEUE_OVERFLOW]; r.probeOverflow += c[DBGC.PROBE_OVERFLOW];
    r.framesRead++;
  }

  private readonly tick = (now: number): void => {
    if (!this.running) return;
    requestAnimationFrame(this.tick);
    if (this.fatal || this.suspended) { this.lastT = undefined; return; }
    try {
      this.frame(now);
    } catch (e) {
      this.fatal = e instanceof Error ? e.message : String(e);
      console.error(e);
      this.loading.start('Render error');
      this.loading.error(this.fatal);
    }
  };

  private frame(now: number): void {
    const rawDt = this.lastT === undefined ? 0 : (now - this.lastT) / 1000;
    this.lastT = now;
    this.cpuFrame.push(rawDt);
    const dt = clampDt(rawDt);
    if (this.canvas.width === 0 || this.canvas.height === 0) return;

    const r = this.render;
    const advance = !r.paused || this.stepPending;
    this.stepPending = false;
    for (const cb of this.beforeFrame) cb({ now, rawDt, dt, advance });
    const reset = this.resetPending;
    this.resetPending = false;
    if (reset) { this.frameIndex = 0; this.totals.nan = this.totals.inf = 0; this.hooks.onResetHistory?.(this); }
    if (advance) {
      this.camera.update(dt);
      this.time += dt;
    }
    const cur: CameraState = { camToWorld: this.camera.camToWorld(), yfov: this.camera.yfov, znear: 1e-3 };
    const prev = reset || !this.lastCamera ? cur : this.lastCamera;
    this.lastCamera = cur;

    let flags = 0;
    if (r.paused) flags |= FRAME_PAUSED;
    if (r.freezeSeed) flags |= FRAME_FREEZE_SEED;
    if (r.freezeFrame) flags |= FRAME_FREEZE_FRAME;
    if (reset) flags |= FRAME_RESET_HISTORY;
    const jm = JITTER_CODE[r.jitter];
    const uniforms: FrameUniformInput = {
      camera: cur, prevCamera: prev,
      width: this.targets.width, height: this.targets.height,
      frameIndex: this.frameIndex, seedIndex: this.seedIndex, runSeed: this.runSeed,
      flags, jitterMode: jm,
      jitter: jm === JITTER_R2 ? r2Jitter(this.seedIndex, this.runSeed) : [0.5, 0.5],
      origin: this.origin,
      exposure: 2 ** this.present.exposureEV,
      time: this.time, dt: advance ? dt : 0,
      sceneDiag: this.sceneDiag,
    };
    this.frameUniforms.write(uniforms);
    this.debug.update(this.debugSettings, this.frameIndex);

    const enc = this.device.createCommandEncoder({ label: 'frame' });
    this.timestamps.beginFrame();
    this.debug.beginFrame(enc);
    const ctx: FrameContext = {
      device: this.device, targets: this.targets, uniforms, advanced: advance, resetHistory: reset, scene: this.scene,
      debug: this.debugSettings, timestamps: (n) => this.timestamps.pass(n),
    };
    if (this.hooks.renderFrame) this.hooks.renderFrame(enc, ctx);
    else this.testPattern.encode(enc, this.targets, this.timestamps.pass('test-pattern'));
    if (this.debugSettings.mode !== 0) this.debug.encodeResolve(enc, this.timestamps.pass('debug-resolve'));
    const ds = this.debugSettings;
    this.presenter.encode(
      enc,
      { color: this.targets.color, debugOut: this.debug.debugOut },
      this.present,
      { active: ds.mode !== 0, split: ds.split, splitPos: ds.splitPos, probe: ds.probeEnabled, probePixel: ds.probePixel },
      r.overlay ? { overlay: this.overlay, camera: cur } : undefined,
      this.timestamps.pass('present'),
    );
    this.probe.encodeCopy(enc, this.debug.buffer, this.frameCounter);
    this.timestamps.resolve(enc);
    this.device.queue.submit([enc.finish()]);
    this.timestamps.afterSubmit();
    this.probe.afterSubmit();

    this.frameCounter++;
    if (advance) {
      if (!r.freezeFrame) this.frameIndex++;
      if (!r.freezeSeed && !r.freezeFrame) this.seedIndex++;
    }
    if (now - this.lastHud > 250) { this.lastHud = now; this.updateHud(); }
  }

  private updateHud(): void {
    const cam = this.camera;
    const avgDt = this.cpuFrame.value;
    this.hud.update({
      fps: avgDt > 0 ? 1 / avgDt : 0,
      cpuMs: avgDt * 1000,
      passes: this.timestamps.averages(),
      timestampsSupported: this.timestamps.supported,
      counters: this.probe.latest?.counters,
      totals: this.totals,
      internal: [this.targets.width, this.targets.height],
      canvas: [this.canvas.width, this.canvas.height],
      dpr: devicePixelRatio,
      frameIndex: this.frameIndex,
      paused: this.render.paused,
      camera: {
        position: cam.position, yawDeg: cam.yaw / DEG, pitchDeg: cam.pitch / DEG, speed: cam.speed,
        recording: !!cam.recording, playing: !!cam.playing,
      },
      scene: this.scene?.name,
      extra: this.hooks.hudLines?.(this),
    });
    const aovIsCode = this.debug.activeView()?.kind === 'code';
    this.probePanel.update(this.debugSettings.probePixel, this.probe.latest, aovIsCode);
    this.panel?.refreshMonitors();
  }
}
