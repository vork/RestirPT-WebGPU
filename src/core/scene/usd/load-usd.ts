// USD scene loading entry point (docs/decisions/usd.md: LightUSD rc4 `next`, called from our own module Worker).
// URL / File / bytes → SceneData. The Worker path is used in browsers; Node tests (and useWorker: false) run inline.
import type { UsdConvertOptions, UsdSceneStats } from './usd-scene.ts';
import type { SceneData } from '../types.ts';

export interface UsdLoadResult {
  scene: SceneData;
  stats: UsdSceneStats & { ms: Record<string, number> };
}

export interface LoadUsdOptions extends UsdConvertOptions { useWorker?: boolean }

export interface UsdWorkerRequest { id: number; bytes: ArrayBuffer; name: string; opts: UsdConvertOptions }
export type UsdWorkerResponse = { id: number; ok: true; result: UsdLoadResult } | { id: number; ok: false; error: string };

export const isUsdName = (name: string): boolean => /\.usd[acz]?$/i.test(name.split(/[?#]/)[0]);

export async function loadUsdInline(bytes: Uint8Array, name: string, opts: UsdConvertOptions = {}): Promise<UsdLoadResult> {
  // dynamic: keeps LightUSD + the adapter out of the main-thread bundle (they normally run in the Worker)
  const [{ extractUsd }, { usdToScene }] = await Promise.all([import('./usd-native.ts'), import('./usd-scene.ts')]);
  const raw = await extractUsd(bytes, name);
  const t0 = performance.now();
  const { scene, stats } = usdToScene(raw, opts);
  return { scene, stats: { ...stats, ms: { ...raw.timings, convertMs: performance.now() - t0 } } };
}

let nextId = 1;

/** Load a USD scene from a URL, a File/Blob or bytes. */
export async function loadUsd(input: string | URL | Blob | Uint8Array | ArrayBuffer, opts: LoadUsdOptions = {}): Promise<UsdLoadResult> {
  let bytes: Uint8Array;
  let name: string;
  if (typeof input === 'string' || input instanceof URL) {
    const url = new URL(String(input), globalThis.location?.href).href;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
    bytes = new Uint8Array(await r.arrayBuffer());
    name = decodeURIComponent(url.split(/[?#]/)[0].split('/').pop() ?? 'scene.usd');
  } else if (typeof Blob !== 'undefined' && input instanceof Blob) {
    bytes = new Uint8Array(await input.arrayBuffer());
    name = (input as Blob & { name?: string }).name ?? 'scene.usd';
  } else {
    bytes = input instanceof Uint8Array ? input.slice() : new Uint8Array((input as ArrayBuffer).slice(0));
    name = 'scene.usd';
  }
  const { useWorker, ...conv } = opts;
  if (!(useWorker ?? (typeof Worker !== 'undefined' && typeof window !== 'undefined'))) return loadUsdInline(bytes, name, conv);
  const worker = new Worker(new URL('./usd-loader.worker.ts', import.meta.url), { type: 'module', name: 'usd-loader' });
  const id = nextId++;
  try {
    return await new Promise<UsdLoadResult>((resolve, reject) => {
      worker.onmessage = (ev: MessageEvent<UsdWorkerResponse>) => {
        if (ev.data.id !== id) return;
        if (ev.data.ok) resolve(ev.data.result); else reject(new Error(`USD load failed: ${ev.data.error}`));
      };
      worker.onerror = (ev) => reject(new Error(`USD worker error: ${ev.message}`));
      const buf = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength ? bytes.buffer as ArrayBuffer : bytes.slice().buffer as ArrayBuffer;
      worker.postMessage({ id, bytes: buf, name, opts: conv } satisfies UsdWorkerRequest, [buf]);
    });
  } finally {
    worker.terminate();
  }
}
