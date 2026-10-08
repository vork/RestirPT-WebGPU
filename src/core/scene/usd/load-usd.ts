// USD scene loading entry point (docs/decisions/usd.md: LightUSD rc4 `next`, called from our own module Worker).
// URL / File / bytes → SceneData. The Worker path is used in browsers; Node tests (and useWorker: false) run inline.
import type { UsdConvertOptions, UsdSceneStats } from './usd-scene.ts';
import type { SceneData } from '../types.ts';

export interface UsdLoadResult {
  scene: SceneData;
  stats: UsdSceneStats & { ms: Record<string, number> };
}

export interface LoadUsdOptions extends UsdConvertOptions { useWorker?: boolean }

/** M7: where texture files referenced by the USD layer live: a base URL (browser / Worker: fetched relative to it) or a
 *  directory (Node: read from disk). Defaults to the USD file's own URL for URL loads. */
export interface UsdAssetOptions { assetBase?: string }

export interface UsdWorkerRequest { id: number; bytes: ArrayBuffer; name: string; opts: UsdConvertOptions & UsdAssetOptions }
export type UsdWorkerResponse = { id: number; ok: true; result: UsdLoadResult } | { id: number; ok: false; error: string };

export const isUsdName = (name: string): boolean => /\.usd[acz]?$/i.test(name.split(/[?#]/)[0]);

/** Resolve a texture asset path against `assetBase` (URL → fetch, directory → Node fs). */
function assetResolver(base: string | undefined): ((p: string) => Promise<Uint8Array | undefined>) | undefined {
  if (!base) return undefined;
  const isUrl = /^(https?|blob|file|data):/.test(base) || (typeof window !== 'undefined' || typeof (globalThis as { WorkerGlobalScope?: unknown }).WorkerGlobalScope !== 'undefined');
  return async (p: string) => {
    try {
      if (isUrl) {
        const r = await fetch(new URL(p, base).href);
        if (!r.ok || (r.headers.get('content-type') ?? '').includes('text/html')) return undefined;
        return new Uint8Array(await r.arrayBuffer());
      }
      const fsName = 'node:fs/promises', pathName = 'node:path';
      const fs = (await import(/* @vite-ignore */ fsName)) as typeof import('node:fs/promises');
      const path = (await import(/* @vite-ignore */ pathName)) as typeof import('node:path');
      return new Uint8Array(await fs.readFile(path.resolve(base, p)));
    } catch { return undefined; }
  };
}

/** Decode texture bytes to RGBA8: PNG everywhere (io/png.ts), other formats through the browser decoder. */
async function decodeAssets(assets: Record<string, Uint8Array>): Promise<Record<string, { width: number; height: number; pixels: Uint8Array } | undefined>> {
  const { decodePng, isPng } = await import('../../io/png.ts');
  const { browserImageDecoder, canDecodeInThisRuntime } = await import('../image-decode.ts');
  const out: Record<string, { width: number; height: number; pixels: Uint8Array } | undefined> = {};
  for (const [p, b] of Object.entries(assets)) {
    try {
      if (isPng(b)) out[p] = await decodePng(b);
      else if (canDecodeInThisRuntime()) out[p] = (await browserImageDecoder(b, /\.jpe?g$/i.test(p) ? 'image/jpeg' : 'image/png', p)) ?? undefined;
    } catch { out[p] = undefined; }
  }
  return out;
}

export async function loadUsdInline(bytes: Uint8Array, name: string, opts: UsdConvertOptions & UsdAssetOptions = {}): Promise<UsdLoadResult> {
  // dynamic: keeps LightUSD + the adapter out of the main-thread bundle (they normally run in the Worker)
  const [{ extractUsd }, { usdToScene }, { withSceneTangents }] = await Promise.all([import('./usd-native.ts'), import('./usd-scene.ts'), import('../tangents.ts')]);
  const { assetBase, ...conv } = opts;
  const raw = await extractUsd(bytes, name, assetResolver(assetBase));
  const t0 = performance.now();
  const images = await decodeAssets(raw.assets);
  const r = usdToScene(raw, conv, images);
  if (raw.missingAssets.length) r.scene.warnings.push(`USD texture files not found: ${raw.missingAssets.join(', ')}`);
  // M7: Blender-semantics MikkTSpace tangents for normal-mapped materials (tangents.ts; a no-op otherwise)
  const tg = await withSceneTangents(r.scene, { quantized: r.scene.quant?.mode === 'quantized', flatFaceNormals: true });
  return { scene: tg.scene, stats: { ...r.stats, ms: { ...raw.timings, convertMs: performance.now() - t0 } } };
}

let nextId = 1;

/** Load a USD scene from a URL, a File/Blob or bytes. */
export async function loadUsd(input: string | URL | Blob | Uint8Array | ArrayBuffer, opts: LoadUsdOptions & UsdAssetOptions = {}): Promise<UsdLoadResult> {
  let bytes: Uint8Array;
  let name: string;
  if (typeof input === 'string' || input instanceof URL) {
    const url = new URL(String(input), globalThis.location?.href).href;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`fetch ${url}: HTTP ${r.status}`);
    bytes = new Uint8Array(await r.arrayBuffer());
    name = decodeURIComponent(url.split(/[?#]/)[0].split('/').pop() ?? 'scene.usd');
    opts = { assetBase: url, ...opts };
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
