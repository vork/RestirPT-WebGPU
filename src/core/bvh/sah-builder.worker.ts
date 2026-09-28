// Worker entry for the SAH build (plan §1.1: no SharedArrayBuffer; results come back via transfer lists).
/// <reference lib="webworker" />
import { buildBvh, type BvhBuildOptions } from './sah-builder.ts';
import { bvhTransferList } from './layout.ts';

export interface BvhBuildRequest { id: number; positions: Float32Array; indices: Uint32Array; options?: BvhBuildOptions }

self.onmessage = (ev: MessageEvent<BvhBuildRequest>) => {
  const { id, positions, indices, options } = ev.data;
  try {
    const bvh = buildBvh(positions, indices, options);
    (self as unknown as DedicatedWorkerGlobalScope).postMessage({ id, bvh }, bvhTransferList(bvh));
  } catch (e) {
    (self as unknown as DedicatedWorkerGlobalScope).postMessage({ id, error: e instanceof Error ? e.message : String(e) });
  }
};
