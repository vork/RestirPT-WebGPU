// LightUSD rc4 spike page (plan §5 M0). Loads each test file with
//   direct = our own module Worker over lightusd_next.js RenderStream (usd-direct.worker.ts; default Vite config)
//   next   = stock LightUSDWorkerLoader, backend 'next'  (RenderStream / tydra-next, lightusd_next.wasm)
//   legacy = stock LightUSDWorkerLoader, default backend (LightUSDLoaderNative / Tydra, lightusd.wasm)
//   three  = three r186 USDLoader on the main thread (fallback-1)
// and publishes normalized SceneDumps on window.__USD_SPIKE__ for run-spike.ts.
// The stock loader needs vite.lightusd-stock.config.ts, so it is only imported when next/legacy is requested.
import type { LightUSDInitOptions } from 'lightusd';
import { USDLoader } from 'three/examples/jsm/loaders/USDLoader.js';
import type { SceneDump } from './dump-types.ts';
import { normalizeLegacy, normalizeNext, normalizeThree } from './normalize.ts';
import type { DirectRequest, DirectResponse } from './usd-direct.worker.ts';

type Mode = 'direct' | 'next' | 'legacy' | 'three';
interface SpikeResult {
  done: boolean;
  userAgent: string;
  crossOriginIsolated: boolean;
  sharedArrayBuffer: boolean;
  init: Record<string, { ms: number; error?: string }>;
  dumps: SceneDump[];
}

const assetUrls = {
  ...import.meta.glob('../../assets/usd-spike/*.{usda,usdc,usdz}', { query: '?url', import: 'default', eager: true }),
  // large timing scene written by make_usd.py --perf-out validation/out/usd-spike/perf (gitignored)
  ...import.meta.glob('../../out/usd-spike/perf/*.usdc', { query: '?url', import: 'default', eager: true }),
} as Record<string, string>;
const byName = new Map(Object.entries(assetUrls).map(([k, v]) => [k.split('/').pop()!, v]));

const params = new URLSearchParams(location.search);
const files = (params.get('files')?.split(',') ?? [...byName.keys()].filter((f) => !f.startsWith('perf')).sort()).filter((f) => byName.has(f));
const modes = (params.get('modes')?.split(',') ?? ['direct', 'three']) as Mode[];
const LOAD_TIMEOUT_MS = 60_000;

const logEl = document.getElementById('log')!;
const log = (s: string): void => {
  logEl.textContent += s + '\n';
  console.log(s);
};

const result: SpikeResult = {
  done: false,
  userAgent: navigator.userAgent,
  crossOriginIsolated: globalThis.crossOriginIsolated === true,
  sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
  init: {},
  dumps: [],
};
(window as unknown as { __USD_SPIKE__: SpikeResult }).__USD_SPIKE__ = result;

const withTimeout = <T>(p: Promise<T>, ms: number, what: string): Promise<T> =>
  Promise.race([p, new Promise<T>((_, rej) => setTimeout(() => rej(new Error(`${what}: timeout ${ms} ms`)), ms))]);

function failed(source: string, file: string, e: unknown): SceneDump {
  return {
    source, file, ok: false, error: String((e as Error)?.stack ?? e),
    stage: { upAxis: null, metersPerUnit: null }, draws: [], lights: [], materials: [], pointInstancers: [], cameras: [],
  };
}

async function runDirect(): Promise<void> {
  const source = 'lightusd-direct';
  const t0 = performance.now();
  const worker = new Worker(new URL('./usd-direct.worker.ts', import.meta.url), { type: 'module' });
  let nextId = 0;
  const call = (name: string, bytes: ArrayBuffer): Promise<DirectResponse> =>
    new Promise((res, rej) => {
      const id = ++nextId;
      const onMsg = (e: MessageEvent<DirectResponse>): void => {
        if (e.data.id !== id) return;
        worker.removeEventListener('message', onMsg);
        res(e.data);
      };
      worker.addEventListener('message', onMsg);
      worker.onerror = (e) => rej(new Error(`worker error: ${e.message}`));
      worker.postMessage({ id, name, bytes } satisfies DirectRequest, [bytes]);
    });
  result.init[source] = { ms: 0 };
  for (const f of files) {
    const t1 = performance.now();
    try {
      const bytes = await (await fetch(byName.get(f)!)).arrayBuffer();
      const tFetch = performance.now();
      const r = await withTimeout(call(f, bytes), LOAD_TIMEOUT_MS, `direct ${f}`);
      if (!r.ok) throw new Error(r.error);
      const loadMs = performance.now() - t1;
      const t2 = performance.now();
      const dump = normalizeNext(r.data!, f, source);
      dump.timings = { loadMs, fetchMs: tFetch - t1, ...r.timings, normalizeMs: performance.now() - t2 };
      const adapted = normalizeNext(r.data!, f, `${source}+adapter`, true);
      adapted.timings = dump.timings;
      result.dumps.push(adapted);
      if (result.init[source].ms === 0) result.init[source] = { ms: r.timings!.wasmInitMs, workerSpawnToFirstResultMs: performance.now() - t0 } as { ms: number };
      result.dumps.push(dump);
      log(`[${source}] ${f}: ${loadMs.toFixed(1)} ms (begin ${r.timings!.beginMs.toFixed(1)} ms), ${dump.draws.length} draws, ` +
        `${dump.lights.length} lights, ${dump.materials.length} materials, ${dump.pointInstancers.length} instancers`);
    } catch (e) {
      result.dumps.push(failed(source, f, e));
      log(`[${source}] ${f}: FAILED ${e}`);
    }
  }
  worker.terminate();
}

async function runLightUSD(mode: 'next' | 'legacy'): Promise<void> {
  const { LightUSDWorkerLoader } = await import('lightusd');
  const opts: LightUSDInitOptions = mode === 'next' ? { backend: 'next' } : {};
  const source = `lightusd-${mode}`;
  let loader = new LightUSDWorkerLoader();
  const t0 = performance.now();
  try {
    await withTimeout(loader.init(opts), LOAD_TIMEOUT_MS, `${mode} init`);
    result.init[source] = { ms: performance.now() - t0 };
    log(`[${source}] worker + wasm init ${(performance.now() - t0).toFixed(1)} ms`);
  } catch (e) {
    result.init[source] = { ms: performance.now() - t0, error: String(e) };
    log(`[${source}] init FAILED: ${e}`);
    for (const f of files) result.dumps.push(failed(source, f, e));
    return;
  }
  for (const f of files) {
    const t1 = performance.now();
    try {
      const res = await withTimeout(loader.load(byName.get(f)!, opts), LOAD_TIMEOUT_MS, `${mode} ${f}`);
      const loadMs = performance.now() - t1;
      const t2 = performance.now();
      const dump = mode === 'next' ? normalizeNext(res._data, f) : normalizeLegacy(res._data, f);
      dump.timings = { loadMs, normalizeMs: performance.now() - t2 };
      result.dumps.push(dump);
      log(`[${source}] ${f}: ${loadMs.toFixed(1)} ms, ${dump.draws.length} draws, ${dump.lights.length} lights, ` +
        `${dump.materials.length} materials, ${dump.pointInstancers.length} instancers`);
    } catch (e) {
      result.dumps.push(failed(source, f, e));
      log(`[${source}] ${f}: FAILED ${e}`);
      loader.dispose(); // a trapped wasm instance cannot be reused
      loader = new LightUSDWorkerLoader();
      await loader.init(opts).catch(() => undefined);
    }
  }
  loader.dispose();
}

async function runThree(): Promise<void> {
  const loader = new USDLoader();
  for (const f of files) {
    const t1 = performance.now();
    try {
      const group = await withTimeout(loader.loadAsync(byName.get(f)!), LOAD_TIMEOUT_MS, `three ${f}`);
      const loadMs = performance.now() - t1;
      const dump = normalizeThree(group, f);
      dump.timings = { loadMs };
      result.dumps.push(dump);
      log(`[three] ${f}: ${loadMs.toFixed(1)} ms, ${dump.draws.length} draws, ${dump.lights.length} lights`);
    } catch (e) {
      result.dumps.push(failed('three', f, e));
      log(`[three] ${f}: FAILED ${e}`);
    }
  }
}

async function main(): Promise<void> {
  log(`files: ${files.join(', ')}\nmodes: ${modes.join(', ')}\ncrossOriginIsolated=${result.crossOriginIsolated}`);
  for (const m of modes) {
    if (m === 'three') await runThree();
    else if (m === 'direct') await runDirect();
    else await runLightUSD(m);
  }
  result.done = true;
  log('done');
}

main().catch((e) => {
  log(`fatal: ${e}`);
  result.done = true;
});
