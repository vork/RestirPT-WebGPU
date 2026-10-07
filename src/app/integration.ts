// M1/M2 integration: wires the real loaders (glTF Worker, USD Worker, HDRI Worker) and the Renderer
// (scene/BVH/textures/env upload + primary pass) into the app shell through SceneLoader and AppHooks. The shell keeps
// its test pattern until the first scene or environment is ready.
// Env load mode follows the renderer's texture path (validation: exact texels, lossless codecs, negatives rejected;
// interactive: clamps/downsamples) unless IntegrationOptions.envMode pins it. Dev builds add "Export for Cycles"
// (scene package → validation/out/export-<id>/ through the harness upload middleware).
// M5 (T-D; restir-temporal-api.md §2.10, TD19–TD20, Changelog D-2): light / env-parameter edits and timeline playback
// restart only the progressive accumulation (app.resetHistory); the ReSTIR temporal history survives them (refresh +
// q′). The temporal history resets on config changes (config hash in the kernel), scene load / resize (new allocation),
// env map swaps (RestirKernel.setEnvironment) and the explicit controls (FrameContext.resetTemporal).
import type { GpuContext } from '../core/gpu/device.ts';
import { registerProbeTag } from '../core/render/probe.ts';
import { EXTRA_VIEWS, PRIMARY_PROBE_TAGS, Renderer } from '../core/render/renderer.ts';
import { ENV_DEBUG_VIEWS } from '../core/render/env-debug.ts';
import { RESTIR_PROBE_TAGS, RESTIR_VIEWS } from '../core/render/restir/debug.ts';
import { DENOISER_VIEWS } from '../core/render/denoise/layout.ts';
import { addRestirPanel, type RestirPanelHandle } from './ui/panels/restir-panel.ts';
import { addDenoiserPanel, type DenoiserPanelHandle } from './ui/panels/denoiser-panel.ts';
import { RestirInspector } from './ui/panels/restir-inspector.ts';
import { DBG } from '../core/render/debug-views.ts';
import { boundsDiagonal } from '../core/render/frame-uniforms.ts';
import { emptyScene } from '../core/render/scene-gpu.ts';
import { loadEnvironment, type EnvLoadMode } from '../core/scene/env/load-env.ts';
import { loadScene as loadGltfScene } from '../core/scene/load-scene.ts';
import { loadUsd } from '../core/scene/usd/load-usd.ts';
import type { EnvironmentData, SceneData } from '../core/scene/types.ts';
import { ensureLightStore, type LightStore } from '../core/scene/light-store.ts';
import type { App, AppHooks } from './app.ts';
import { DEG } from './camera-math.ts';
import { extensionOf, type SceneLoader } from './loader.ts';

export interface Integration {
  loader: SceneLoader;
  hooks: AppHooks;
  /** Available once the app called the first hook (targets resize during App.init). */
  renderer(): Renderer | undefined;
  /** Resolves when the first real scene (or env-only empty scene) has been uploaded and compiled. */
  sceneReady: Promise<void>;
  /** M4 (WP-D): the ReSTIR panel and pixel inspector (available once the renderer exists). */
  restirUi(): { panel?: RestirPanelHandle; inspector?: RestirInspector; denoiser?: DenoiserPanelHandle };
  /** Dev: export the current scene/camera/env as a scene package to validation/out/export-<id>/ (returns the dir). */
  exportForCycles(app: App, cfg?: Partial<ExportConfig>): Promise<string | undefined>;
}

export interface ExportConfig { width: number; height: number; maxBounces: number; lightMode: 'A' | 'B' | 'A′'; status: string }

export interface IntegrationOptions {
  envMode?: EnvLoadMode;
}

export function createIntegration(gpu: GpuContext, opts: IntegrationOptions = {}): Integration {
  let renderer: Renderer | undefined;
  let rendererP: Promise<Renderer> | undefined;
  let resolveReady!: () => void;
  const sceneReady = new Promise<void>((r) => { resolveReady = r; });
  let pendingScenes = 0;
  // Analytic lights (M3a): the scene's LightStore is the single source of truth (editor, animation, loaders write it).
  // Changes only mark the lights dirty; renderFrame pushes store.list() to the renderer ONCE per frame, which is the one
  // place the cur/prev light buffers flip (id maps + alias rebuild only when the powers/set changed).
  let lightStore: LightStore | undefined;
  let lightUnsub: (() => void) | undefined;
  let lightsDirty = false;
  const restirUi: { panel?: RestirPanelHandle; inspector?: RestirInspector; denoiser?: DenoiserPanelHandle } = {};

  const ensure = (app: App): Promise<Renderer> => {
    rendererP ??= (async () => {
      for (const [tag, name] of [...PRIMARY_PROBE_TAGS, ...RESTIR_PROBE_TAGS]) registerProbeTag(tag, name);
      for (const v of [...EXTRA_VIEWS, ...ENV_DEBUG_VIEWS, ...RESTIR_VIEWS, ...DENOISER_VIEWS]) if (!app.debug.registry.get(v.id)) app.registerDebugView(v);
      app.render.jitter = 'iid'; // plan §1.2: i.i.d. per-run/per-frame jitter; the panel offers R2 and pixel centre
      const r = await Renderer.create({ device: gpu.device, debugLayout: app.debug.layout, debug: app.debug, features: gpu.features, wgslLanguageFeatures: gpu.wgslLanguageFeatures },
        { watertight: false, renderMode: 'pt', lightMode: 'B' }); // interactive default: Mode B after rung 3.11 (restir-m6-api MD9); MT (the panel toggles Woop; validation paths default to Woop); PT beauty (M3a)
      renderer = r;
      if (app.targets) r.resize(app.targets);
      addRendererPanel(app, r);
      restirUi.inspector = new RestirInspector(app);
      restirUi.panel = addRestirPanel(app, r, restirUi.inspector, 4);
      restirUi.denoiser = addDenoiserPanel(app, r, 5);
      app.panel?.refresh();
      return r;
    })();
    return rendererP;
  };

  const renderFrame: AppHooks['renderFrame'] = (encoder, ctx) => {
    if (lightsDirty && lightStore && renderer?.setLights(lightStore.list())) lightsDirty = false;
    renderer?.encode(encoder, { advanced: ctx.advanced, debugMode: ctx.debug.mode, debugGroup: ctx.targets.debug.bindGroup, resetTemporal: ctx.resetTemporal, resetHistory: ctx.resetHistory },
      () => ctx.timestamps('primary'), () => ctx.timestamps('pt'), () => ctx.timestamps('restir'));
  };
  /** ReSTIR with temporal reuse is on: paused frames re-display the last frame (TD20). */
  const restirTemporal = () => !!renderer && renderer.options.renderMode === 'restir' && !!renderer.restir?.settings.temporal;

  const adoptScene = async (app: App, scene: SceneData, origin: [number, number, number]) => {
    const r = await ensure(app);
    pendingScenes++;
    try {
      const store = ensureLightStore(scene); // before the upload: scene.lights mirrors the store (stable ids)
      // plan §1.10: interactive default ≥ 4 bounces when glass is present (each glass interface is a bounce)
      const hasGlass = scene.materials.some((m) => m.model === 'glass' || m.model === 'refraction' || m.transmissionFactor > 1e-5);
      if (hasGlass && r.options.maxBounces < 4) { await r.setOptions({ maxBounces: 4 }); app.panel?.refresh(); }
      const g = await r.setScene(scene, origin);
      if (!g) return; // superseded by a newer scene
      lightUnsub?.();
      lightStore = store;
      lightsDirty = true;
      // Any light change restarts the progressive accumulation (PT and ReSTIR display); the ReSTIR temporal history is
      // kept (M5: the kernel stages the edit, commits once per frame and refreshes the affected records).
      lightUnsub = store.onChange(() => { lightsDirty = true; app.resetHistory(); });
      if (g.warnings.length) app.loading.warn(...g.warnings);
      // Scene-scaled default ranges for the distance views (the registry entries are the live defaults).
      const diag = boundsDiagonal(scene.bounds);
      for (const id of [DBG.GB_DEPTH, 112]) {
        const v = app.debug.registry.get(id);
        if (v) v.range = [Math.max(1e-4, diag * 1e-2), diag];
      }
      app.setHooks({ renderFrame });
      app.resetHistory();
      resolveReady();
    } finally {
      pendingScenes--;
    }
  };

  const loader: SceneLoader = {
    async loadScene(src, progress, signal) {
      const ext = extensionOf(src.name);
      if (ext.startsWith('usd')) {
        progress({ stage: 'parsing USD (LightUSD worker)' });
        const res = await loadUsd(src.kind === 'url' ? src.url : src.main);
        if (signal.aborted) throw new Error('aborted');
        const ms = res.stats.ms;
        console.info(`[scene] ${res.scene.name}: ${res.stats.triangles} tris, ${res.stats.draws} draws, upAxis ${res.stats.upAxis}, ` +
          `metersPerUnit ${res.stats.metersPerUnit}, ${res.scene.lights.length} lights, total ${(ms.totalMs ?? 0).toFixed(0)} ms`);
        progress({ stage: 'building BVH + uploading', fraction: 0.9 });
        return res.scene;
      }
      progress({ stage: 'parsing glTF (worker)' });
      const res = await loadGltfScene(src.kind === 'url' ? src.url : src.files);
      if (signal.aborted) throw new Error('aborted');
      const ms = res.stats.ms;
      console.info(`[scene] ${res.scene.name}: ${res.scene.geometry.indices.length / 3} tris, parse ${ms.parse.toFixed(0)} ms, images ${ms.images.toFixed(0)} ms, flatten ${ms.flatten.toFixed(0)} ms`);
      progress({ stage: 'building BVH + uploading', fraction: 0.9 });
      return res.scene;
    },
    async loadEnvironment(src, progress) {
      progress({ stage: 'decoding environment (worker)' });
      const mode: EnvLoadMode = opts.envMode ?? (renderer?.options.textureMode === 'interactive' ? 'interactive' : 'validation');
      const res = await loadEnvironment(src.kind === 'url' ? src.url : src.main, { mode, name: src.name });
      console.info(`[env] ${src.name}: ${mode} mode, ${res.env.width}x${res.env.height}`);
      for (const w of res.warnings) console.info(`[env] ${w}`);
      return res.env;
    },
  };

  const hooks: AppHooks = {
    onTargetsResized(targets, app) {
      if (renderer) renderer.resize(targets);
      else void ensure(app);
    },
    async onSceneLoaded(scene, app) {
      app.loading.progress(0.95, 'building BVH, uploading textures, compiling the primary pass...');
      await adoptScene(app, scene, app.origin);
    },
    async onEnvironmentLoaded(env: EnvironmentData, app) {
      const r = await ensure(app);
      await r.setEnvironment(env);
      // Env without a scene: render the background over an empty BVH.
      if (!r.scene && !r.loading && pendingScenes === 0 && !app.scene) await adoptScene(app, emptyScene('environment'), [0, 0, 0]);
    },
    onEnvironmentParams(p, app) {
      renderer?.setEnvParams({ rotationZ: p.rotationDeg * DEG, strength: p.strength, tint: [p.tint.r, p.tint.g, p.tint.b], visibleToCamera: p.visibleToCamera });
      const r = renderer;
      if (r && (r.options.envNee !== p.nee || r.options.envImportanceCap !== p.importanceRes)) {
        // Config change (plan §2 rule 10: env-NEE toggle / importance resolution reset history); tables built in the Worker.
        void r.setEnvSampling({ nee: p.nee, importanceCap: p.importanceRes }).then(() => app.resetHistory());
      }
      app.resetHistory();
    },
    // With temporal reuse the HUD error totals restart only on temporal resets (renderer.ts); light / env animation
    // restarts the accumulation every frame and must not hide errors.
    onResetHistory: () => { if (!restirTemporal()) renderer?.restirHud?.resetTotals(); },
    // M5.5: a held frame re-runs the denoiser plan (identical output), so the AOV of a denoiser view holds as well (PT
    // too: its held sample is identical).
    holdsFrameWhenPaused: () => restirTemporal() || !!renderer?.denoisedLastFrame,
    afterSubmit: () => renderer?.afterSubmit(),
    hudLines: (app) => {
      const lines = renderer?.hudLines() ?? [];
      restirUi.panel?.refresh();
      restirUi.denoiser?.refresh();
      const env = renderer?.env;
      if (env) {
        const l = lines.find((x) => x.startsWith('env NEE'));
        app.envParams.info = env.present ? `${env.width}x${env.height} rgba32float ${(env.width * env.height * 16 / 2 ** 20).toFixed(1)} MiB\n${l ?? ''}` : 'no environment';
      }
      return lines;
    },
  };

  return {
    loader, hooks, renderer: () => renderer, sceneReady, restirUi: () => restirUi,
    exportForCycles: async (app, cfg = {}) => (renderer
      ? exportForCycles(app, renderer, { width: 512, height: 512, maxBounces: 3, lightMode: 'A', status: '', ...cfg })
      : undefined),
  };
}

/** 'Renderer' folder: texture path, watertight intersection, accumulation. */
function addRendererPanel(app: App, r: Renderer): void {
  const pane = app.panel?.pane;
  if (!pane) return;
  const f = pane.addFolder({ title: 'Renderer', expanded: false, index: 3 });
  const o = r.options;
  const reupload = () => { app.loading.start('Renderer'); void r.reload().then(() => { app.loading.done('re-uploaded'); app.resetHistory(); }, (e: unknown) => app.loading.error(String(e))); };
  f.addBinding(o, 'textureMode', { label: 'textures', options: { 'validation (no resample)': 'validation', 'interactive (mips)': 'interactive' } })
    .on('change', () => { if (r.sceneData) reupload(); });
  f.addBinding(o, 'watertight', { label: 'watertight (Woop)' }).on('change', () => { if (r.sceneData) reupload(); });
  f.addBinding(o, 'accumulate', { label: 'accumulate' }).on('change', () => app.resetHistory());
  // M3a reference path tracer (PT) vs the M1 albedo placeholder; bounce count and Russian roulette.
  f.addBinding(o, 'renderMode', { label: 'mode', options: { 'PT (reference)': 'pt', 'ReSTIR PT (M4)': 'restir', 'albedo (M1)': 'albedo' } })
    .on('change', () => { if (o.renderMode === 'restir') void r.prepareRestir(); app.resetHistory(); app.panel?.refresh(); });
  f.addBinding(o, 'maxBounces', { label: 'max bounces', min: 0, max: 13, step: 1 })
    .on('change', () => { void r.setOptions({ maxBounces: o.maxBounces }); app.resetHistory(); });
  f.addBinding(o, 'rr', { label: 'Russian roulette' }).on('change', () => { void r.setOptions({ rr: o.rr }); app.resetHistory(); });
  // M3b light modes (plan §1.4): A = analytic lights NEE-only (smooth mirrors / glass never show them, no caustics);
  // B = area lights hittable by BSDF rays (pass-through, MIS); A′ = hittable only after a delta lobe (same expectation as B)
  f.addBinding(o, 'lightMode', { label: 'light mode', options: { 'A (NEE only)': 'A', 'B (pass-through + MIS)': 'B', 'A′ (after delta lobes)': 'A′' } })
    .on('change', () => { void r.setOptions({ lightMode: o.lightMode }); app.resetHistory(); });
  if (import.meta.env.DEV) addExportFolder(app, r);
}

/** Dev only: "Export for Cycles" → scene package in validation/out/export-<id>/ (docs/decisions/scene-bridge.md). */
function addExportFolder(app: App, r: Renderer): void {
  const pane = app.panel?.pane;
  if (!pane) return;
  const f = pane.addFolder({ title: 'Export for Cycles (dev)', expanded: false, index: 4 });
  const cfg: ExportConfig = { width: 512, height: 512, maxBounces: 3, lightMode: 'A', status: '' };
  f.addBinding(cfg, 'width', { min: 16, max: 8192, step: 1 });
  f.addBinding(cfg, 'height', { min: 16, max: 8192, step: 1 });
  f.addBinding(cfg, 'maxBounces', { label: 'max bounces', min: 0, max: 64, step: 1 });
  f.addBinding(cfg, 'lightMode', { label: 'light mode', options: { 'A (NEE only)': 'A', 'B (MIS)': 'B', 'A′ (Cycles: MIS)': 'A′' } });
  f.addButton({ title: 'Export for Cycles' }).on('click', () => { void exportForCycles(app, r, cfg); });
  f.addBinding(cfg, 'status', { readonly: true, multiline: true, rows: 3 });
}

export async function exportForCycles(app: App, r: Renderer, cfg: ExportConfig): Promise<string | undefined> {
  const scene = r.sceneData ?? app.scene;
  if (!scene) { cfg.status = 'no scene loaded'; app.panel?.refresh(); return undefined; }
  app.loading.start('Export for Cycles');
  try {
    const { exportAndUpload } = await import('../../validation/harness/export-package.ts');
    const p = app.envParams;
    const env = app.env ? {
      ...app.env, strength: p.strength, rotationZ: p.rotationDeg * DEG, tint: [p.tint.r, p.tint.g, p.tint.b] as [number, number, number],
      visibleToCamera: p.visibleToCamera,
    } : undefined;
    const run = `export-${new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-')}`;
    const res = await exportAndUpload({ ...scene, env }, {
      camera: { matrix: app.camera.camToWorld(), yfov: app.camera.yfov, znear: 1e-4 },
      render: { width: cfg.width, height: cfg.height, maxBounces: cfg.maxBounces },
      lightMode: cfg.lightMode, envSampling: p.nee ? 'AUTOMATIC' : 'NONE',
      source: { uri: scene.name },
    }, run);
    cfg.status = `${res.dir}\n${res.files.length} files, ${(res.bytes / 1024).toFixed(0)} KiB\nsha256 ${res.sha256.slice(0, 16)}`;
    app.loading.done(`exported ${res.dir}`);
    console.info(`[export] ${res.dir}: ${res.files.join(', ')} (package sha256 ${res.sha256})`);
    return res.dir;
  } catch (e) {
    cfg.status = `failed: ${e instanceof Error ? e.message : String(e)}`;
    app.loading.error(`Export for Cycles failed: ${e instanceof Error ? e.message : String(e)}`);
    return undefined;
  } finally {
    app.panel?.refresh();
  }
}
