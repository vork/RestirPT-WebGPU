// M1 integration: wires the real loaders (glTF Worker, HDRI Worker) and the Renderer (scene/BVH/textures/env upload
// + primary pass) into the app shell through SceneLoader and AppHooks. The shell keeps its test pattern until the
// first scene or environment is ready.
import type { GpuContext } from '../core/gpu/device.ts';
import { registerProbeTag } from '../core/render/probe.ts';
import { EXTRA_VIEWS, PRIMARY_PROBE_TAGS, Renderer } from '../core/render/renderer.ts';
import { DBG } from '../core/render/debug-views.ts';
import { boundsDiagonal } from '../core/render/frame-uniforms.ts';
import { emptyScene } from '../core/render/scene-gpu.ts';
import { loadEnvironment, type EnvLoadMode } from '../core/scene/env/load-env.ts';
import { loadScene as loadGltfScene } from '../core/scene/load-scene.ts';
import type { EnvironmentData, SceneData } from '../core/scene/types.ts';
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
}

export interface IntegrationOptions {
  envMode?: EnvLoadMode;
}

export function createIntegration(gpu: GpuContext, opts: IntegrationOptions = {}): Integration {
  let renderer: Renderer | undefined;
  let rendererP: Promise<Renderer> | undefined;
  let resolveReady!: () => void;
  const sceneReady = new Promise<void>((r) => { resolveReady = r; });
  let pendingScenes = 0;

  const ensure = (app: App): Promise<Renderer> => {
    rendererP ??= (async () => {
      for (const [tag, name] of PRIMARY_PROBE_TAGS) registerProbeTag(tag, name);
      for (const v of EXTRA_VIEWS) if (!app.debug.registry.get(v.id)) app.registerDebugView(v);
      app.render.jitter = 'iid'; // plan §1.2: i.i.d. per-run/per-frame jitter; the panel offers R2 and pixel centre
      const r = await Renderer.create({ device: gpu.device, debugLayout: app.debug.layout, features: gpu.features, wgslLanguageFeatures: gpu.wgslLanguageFeatures });
      renderer = r;
      if (app.targets) r.resize(app.targets);
      addRendererPanel(app, r);
      app.panel?.refresh();
      return r;
    })();
    return rendererP;
  };

  const renderFrame: AppHooks['renderFrame'] = (encoder, ctx) => {
    renderer?.encode(encoder, { advanced: ctx.advanced, debugMode: ctx.debug.mode, debugGroup: ctx.targets.debug.bindGroup }, () => ctx.timestamps('primary'));
  };

  const adoptScene = async (app: App, scene: SceneData, origin: [number, number, number]) => {
    const r = await ensure(app);
    pendingScenes++;
    try {
      const g = await r.setScene(scene, origin);
      if (!g) return; // superseded by a newer scene
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
      if (ext.startsWith('usd')) throw new Error('USD scenes are not wired yet (M2: UsdSceneSource, docs/decisions/usd.md)');
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
      const res = await loadEnvironment(src.kind === 'url' ? src.url : src.main, { mode: opts.envMode ?? 'interactive', name: src.name });
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
      app.resetHistory();
    },
    hudLines: () => renderer?.hudLines() ?? [],
  };

  return { loader, hooks, renderer: () => renderer, sceneReady };
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
}
