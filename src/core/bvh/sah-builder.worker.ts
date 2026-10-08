// Worker entry for the SAH build (plan §1.1: no SharedArrayBuffer; results come back via transfer lists).
/// <reference lib="webworker" />
import { buildBvh, type BvhBuildOptions } from './sah-builder.ts';
import { bvhTransferList } from './layout.ts';
import { buildCwbvhFromMesh } from './cwbvh.ts';

/** cwbvh (M8): build the leaf-≤ 3 BVH2 and its CWBVH (BvhData.cwbvh) instead of the default BVH2. */
export interface BvhBuildRequest { id: number; positions: Float32Array; indices: Uint32Array; options?: BvhBuildOptions & { cwbvh?: boolean } }

self.onmessage = (ev: MessageEvent<BvhBuildRequest>) => {
  const { id, positions, indices, options } = ev.data;
  try {
    let bvh;
    if (options?.cwbvh) { const r = buildCwbvhFromMesh(positions, indices, options); bvh = { ...r.bvh2, cwbvh: r.cw }; }
    else bvh = buildBvh(positions, indices, options);
    (self as unknown as DedicatedWorkerGlobalScope).postMessage({ id, bvh }, bvhTransferList(bvh));
  } catch (e) {
    (self as unknown as DedicatedWorkerGlobalScope).postMessage({ id, error: e instanceof Error ? e.message : String(e) });
  }
};
