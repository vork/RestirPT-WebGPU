// BVH2 GPU layout (Aila–Laine / tinybvh BVH_GPU), shared by the builder, the CPU reference and traverse.wgsl.
// Plan §1.3. The contract (do not change without updating traverse.wgsl):
//
//   nodes : array<vec4f>, 4 per interior node, root = node 0
//           [lmin.xyz | bitcast(leftRef)] [lmax.xyz | bitcast(rightRef)] [rmin.xyz | 0] [rmax.xyz | 0]
//   ref   : leaf  = 0x80000000 | (count << 24) | firstTri   (count 1..127, firstTri < 2^24, leaf-ordered tris)
//           else  = interior node index
//   tris  : array<vec4f>, 3 per leaf-ordered triangle (Möller–Trumbore): [v0 | bitcast(primId)] [e1 | 0] [e2 | 0]
//   trisW : array<vec4f>, 3 per leaf-ordered triangle (Woop watertight): [v0 | bitcast(vid0)] [v1 | vid1] [v2 | vid2]
//           followed by a primId TAIL of ceil(N/4) vec4s read from the END of the array: leaf tri i's primId is
//           component (i & 3) of vec4 (arrayLength − 1 − (i >> 2)). Bind the whole buffer (arrayLength must be exact).
//           vidK = position-welded global vertex id (smallest vertex index with the same position), so edges shared
//           by unwelded (flat-shaded) triangles still get one canonical endpoint order.
//   v0,v1,v2 are the ORIGINAL index order of SceneGeometry.indices[3·primId + 0..2]; hit barycentrics (1−u−v, u, v).

import type { CwbvhData } from './cwbvh.ts';

export const NODE_VEC4S = 4;
export const NODE_FLOATS = 16;
export const TRI_VEC4S = 3;
export const TRI_FLOATS = 12;
export const LEAF_BIT = 0x80000000;
export const LEAF_MAX_COUNT = 127;
export const MAX_TRIANGLES = 1 << 24; // plan §1.3: scenes above 2^24 triangles are rejected
export const BVH_MISS = 0xffffffff;
export const BVH_STACK_SIZE = 32;
export const BVH_MAX_DEPTH = 30; // ≤ 30 pushes, fits the 32-entry private stack
export const BVH_ITER_CAP = 1 << 16;
/** Bits of bvh_stats().w (flags word). */
export const BVH_FLAG_OVERFLOW = 1;
export const BVH_FLAG_ITERCAP = 2;
/** bvh_stats().w bits 8..15 hold the maximum stack depth reached (BVH_STATS builds only). */
export const BVH_FLAG_MAXSP_SHIFT = 8;

export function encodeLeaf(count: number, firstTri: number): number {
  if (count < 1 || count > LEAF_MAX_COUNT || firstTri < 0 || firstTri >= MAX_TRIANGLES) {
    throw new RangeError(`bad leaf (count=${count}, firstTri=${firstTri})`);
  }
  return (LEAF_BIT | (count << 24) | firstTri) >>> 0;
}
export const isLeafRef = (ref: number): boolean => (ref & LEAF_BIT) !== 0;
export const leafCount = (ref: number): number => (ref >>> 24) & 127;
export const leafFirst = (ref: number): number => ref & 0xffffff;

/** Index of leaf tri i's primId in the Uint32 view of trisW (the tail is read from the end). */
export function woopPrimIdIndex(trisWLengthVec4: number, i: number): number {
  return (trisWLengthVec4 - 1 - (i >> 2)) * 4 + (i & 3);
}
export function woopFloatLength(triCount: number): number {
  return (TRI_VEC4S * triCount + Math.ceil(triCount / 4)) * 4;
}

export interface BvhStats {
  triCount: number;           // triangles in the BVH (excludes non-finite ones)
  skippedNonFinite: number;   // non-finite triangles left out of the BVH (never hit)
  nodeCount: number;          // interior nodes
  leafCount: number;
  maxDepth: number;           // max leaf depth = max interior nodes on a root→leaf path (= stack pushes bound)
  maxLeafSize: number;
  avgLeafSize: number;
  /** SAH cost with C_trav = C_isect = 1, normalised by the root surface area. */
  sahCost: number;
  forcedSplits: number;       // median / index splits (depth cap, identical centroids, tiny root)
  buildMs: number;
}

export interface BvhData {
  /** NODE_FLOATS per interior node. Child refs are bit patterns; read them through nodesU32. */
  nodes: Float32Array;
  /** MT triangles, TRI_FLOATS per leaf-ordered triangle (w of the first vec4 = primId bits). Empty if not built. */
  tris: Float32Array;
  /** Woop triangles + primId tail (see header). Empty if not built. */
  trisW: Float32Array;
  /** leaf-order index → primId. */
  primOrder: Uint32Array;
  /** position-welded vertex id per vertex (see header). */
  weldedVid: Uint32Array;
  bounds: { min: [number, number, number]; max: [number, number, number] };
  stats: BvhStats;
  /** M8: the CWBVH collapsed from this BVH2 (built with leaves ≤ 3) when requested (src/core/bvh/cwbvh.ts). */
  cwbvh?: CwbvhData;
}

export const u32View = (a: Float32Array): Uint32Array => new Uint32Array(a.buffer, a.byteOffset, a.length);

/** Transfer list for posting a BvhData across a Worker boundary. */
export function bvhTransferList(b: BvhData): ArrayBuffer[] {
  const bufs = [b.nodes.buffer, b.tris.buffer, b.trisW.buffer, b.primOrder.buffer, b.weldedVid.buffer] as ArrayBuffer[];
  if (b.cwbvh) bufs.push(b.cwbvh.nodes.buffer as ArrayBuffer, b.cwbvh.tris.buffer as ArrayBuffer, b.cwbvh.trisW.buffer as ArrayBuffer, b.cwbvh.primOrder.buffer as ArrayBuffer);
  return [...new Set(bufs)];
}

export interface BvhGpuBuffers { nodes: GPUBuffer; tris: GPUBuffer; triCount: number; watertight: boolean }

/**
 * Upload nodes + the chosen triangle layout (MT or Woop) as STORAGE buffers (plan §1.1: mappedAtCreation).
 * Bind both with their whole size: the Woop primId tail is addressed via arrayLength().
 */
export function uploadBvh(device: GPUDevice, bvh: BvhData, opts: { watertight: boolean; label?: string; extraUsage?: number }): BvhGpuBuffers {
  const tris = opts.watertight ? bvh.trisW : bvh.tris;
  if (tris.length === 0) throw new Error(`BVH was built without the ${opts.watertight ? 'Woop' : 'MT'} triangle layout`);
  const usage = GPUBufferUsage.STORAGE | (opts.extraUsage ?? 0);
  const make = (src: Float32Array, name: string) => {
    const buf = device.createBuffer({ size: Math.max(16, src.byteLength), usage, mappedAtCreation: true, label: `${opts.label ?? 'bvh'}.${name}` });
    new Float32Array(buf.getMappedRange()).set(src);
    buf.unmap();
    return buf;
  };
  return { nodes: make(bvh.nodes, 'nodes'), tris: make(tris, opts.watertight ? 'trisW' : 'tris'), triCount: bvh.primOrder.length, watertight: opts.watertight };
}
