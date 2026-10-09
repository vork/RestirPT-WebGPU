// Entry point: device, M1 integration (loaders + renderer hooks), app shell. Loads the Cornell box by default;
// ?scene=<url>&env=<url> override it, ?pattern=1 keeps the analytic test pattern (shell tests, tests/app/e2e-app.ts).
import { createGpuContext } from '../core/gpu/device.ts';
import { App, type AppHooks } from './app.ts';
import { installEditor } from './editor/index.ts';
import { createIntegration, type Integration } from './integration.ts';
import type { SceneLoader } from './loader.ts';

declare global {
  interface Window {
    /** Debug/automation handle (Playwright screenshots, console poking). */
    __app?: App;
    __integration?: Integration;
    /** Uncaptured WebGPU errors (validation / OOM) seen by the device, for automation. */
    __webgpuErrors?: string[];
  }
}

// Relative to Vite's base ('/' in dev, '/<repo>/' on GitHub Pages); a build can pick another default scene with
// VITE_DEFAULT_SCENE (the Pages build uses the lit Cornell box with point + spot lights).
export const DEFAULT_SCENE_URL = `${import.meta.env.BASE_URL}${import.meta.env.VITE_DEFAULT_SCENE ?? 'validation/assets/cornell/cornell.glb'}`;

export async function boot(loader?: SceneLoader, hooks?: AppHooks): Promise<App> {
  const canvas = document.getElementById('view') as HTMLCanvasElement;
  const ui = document.getElementById('ui') ?? document.body;
  if (!navigator.gpu) throw new Error('navigator.gpu is missing: this browser does not expose WebGPU');
  const errors: string[] = (window.__webgpuErrors = []);
  const gpu = await createGpuContext(navigator.gpu, {
    label: 'app',
    onUncapturedError: (msg) => { if (errors.length < 100) errors.push(msg); console.error('[webgpu uncaptured]', msg); },
  });
  const q = new URLSearchParams(location.search);
  const integration = loader || hooks || q.has('pattern') ? undefined : createIntegration(gpu);
  window.__integration = integration;
  const app = await App.create({ canvas, ui, gpu, loader: loader ?? integration?.loader, hooks: hooks ?? integration?.hooks });
  window.__app = app;
  if (integration) installEditor(app, integration); // M3a light editor, timeline, compare view
  app.start();
  if (integration && !q.has('scene')) {
    const url = new URL(DEFAULT_SCENE_URL, location.href).href;
    void app.loadScene({ kind: 'url', url, name: DEFAULT_SCENE_URL.split('/').pop() ?? 'scene' });
  }
  return app;
}

boot().catch((e: unknown) => {
  const msg = `WebGPU init failed: ${e instanceof Error ? e.message : String(e)}`;
  const el = document.getElementById('fatal');
  if (el) { el.textContent = msg; el.hidden = false; }
  console.error(e);
});
