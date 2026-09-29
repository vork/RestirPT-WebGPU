// Main-thread client for env-importance.worker.ts. Falls back to an inline build where Workers are unavailable (Node).
import { buildEnvImportance, type EnvImportance, type EnvImportanceOptions } from './env-importance.ts';
import type { EnvImportanceRequest } from './env-importance.worker.ts';

let worker: Worker | undefined;
let nextId = 1;
const pending = new Map<number, { resolve: (t: EnvImportance) => void; reject: (e: Error) => void }>();

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL('./env-importance.worker.ts', import.meta.url), { type: 'module', name: 'env-importance' });
  worker.onmessage = (ev: MessageEvent<{ id: number; table?: EnvImportance; error?: string }>) => {
    const p = pending.get(ev.data.id);
    if (!p) return;
    pending.delete(ev.data.id);
    if (ev.data.table) p.resolve(ev.data.table); else p.reject(new Error(ev.data.error ?? 'env importance worker failed'));
  };
  worker.onerror = (ev) => {
    for (const p of pending.values()) p.reject(new Error(`env importance worker error: ${ev.message}`));
    pending.clear();
    worker?.terminate(); worker = undefined;
  };
  return worker;
}

/** Build the env importance tables off the main thread (texels are copied; the caller's array stays usable). */
export function buildEnvImportanceAsync(texels: Float32Array, width: number, height: number, options?: EnvImportanceOptions): Promise<EnvImportance> {
  if (typeof Worker === 'undefined' || typeof window === 'undefined') return Promise.resolve(buildEnvImportance(texels, width, height, options));
  const w = getWorker();
  const id = nextId++;
  const copy = texels.slice();
  const msg: EnvImportanceRequest = { id, texels: copy, width, height, options };
  return new Promise<EnvImportance>((resolve, reject) => {
    pending.set(id, { resolve, reject });
    w.postMessage(msg, [copy.buffer as ArrayBuffer]);
  });
}
