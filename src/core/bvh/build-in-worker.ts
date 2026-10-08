// Main-thread client for sah-builder.worker.ts. Inputs are COPIED by default so the caller's SceneGeometry stays
// usable; pass `transferInputs: true` to hand the arrays over (they are detached afterwards).
import type { BvhBuildOptions } from './sah-builder.ts';
import type { BvhData } from './layout.ts';
import type { BvhBuildRequest } from './sah-builder.worker.ts';

let worker: Worker | undefined;
let nextId = 1;
const pending = new Map<number, { resolve: (b: BvhData) => void; reject: (e: Error) => void }>();

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL('./sah-builder.worker.ts', import.meta.url), { type: 'module', name: 'bvh-builder' });
  worker.onmessage = (ev: MessageEvent<{ id: number; bvh?: BvhData; error?: string }>) => {
    const p = pending.get(ev.data.id);
    if (!p) return;
    pending.delete(ev.data.id);
    if (ev.data.bvh) p.resolve(ev.data.bvh); else p.reject(new Error(ev.data.error ?? 'BVH worker failed'));
  };
  worker.onerror = (ev) => {
    for (const p of pending.values()) p.reject(new Error(`BVH worker error: ${ev.message}`));
    pending.clear();
    worker?.terminate(); worker = undefined;
  };
  return worker;
}

export function buildBvhInWorker(
  positions: Float32Array, indices: Uint32Array, options?: BvhBuildOptions & { cwbvh?: boolean }, transferInputs = false,
): Promise<BvhData> {
  const w = getWorker();
  const id = nextId++;
  const pos = transferInputs ? positions : positions.slice();
  const idx = transferInputs ? indices : indices.slice();
  const msg: BvhBuildRequest = { id, positions: pos, indices: idx, options };
  return new Promise<BvhData>((resolve, reject) => {
    pending.set(id, { resolve, reject });
    w.postMessage(msg, [pos.buffer as ArrayBuffer, ...(idx.buffer === pos.buffer ? [] : [idx.buffer as ArrayBuffer])]);
  });
}

export function terminateBvhWorker(): void { worker?.terminate(); worker = undefined; }
