// Entry point: device, M1 integration (loaders + renderer hooks), app shell. Loads the Cornell box by default;
// ?scene=<url>&env=<url> override it, ?pattern=1 keeps the analytic test pattern (shell tests, tests/app/e2e-app.ts).
import { createGpuContext } from '../core/gpu/device.ts';
import { App, type AppHooks } from './app.ts';
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

export const DEFAULT_SCENE_URL = '/validation/assets/cornell/cornell.glb';

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
  app.start();
  if (integration && !q.has('scene')) {
    const url = new URL(DEFAULT_SCENE_URL, location.href).href;
    void app.loadScene({ kind: 'url', url, name: 'cornell.glb' });
  }
  return app;
}

boot().catch((e: unknown) => {
  const msg = `WebGPU init failed: ${e instanceof Error ? e.message : String(e)}`;
  const el = document.getElementById('fatal');
  if (el) { el.textContent = msg; el.hidden = false; }
  console.error(e);
});
