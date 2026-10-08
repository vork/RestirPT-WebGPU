// CWBVH (compressed 8-wide BVH, Ylitie, Karras & Laine 2017; the tinybvh BVH8_CWBVH node layout) built by collapsing
// our binned-SAH BVH2 (docs/decisions/m8-perf.md §3; gap-perf §10.2 encoder; m8-hwrt-tinybvh-gigi.md §5.2). Traversal:
// shaders/bvh/traverse-cwbvh.wgsl (included by traverse.wgsl under BVH_CWBVH).
//
// Node (80 B = 5 vec4u, 20 words, little endian):
//   w0..2  p = node box min (f32 bits)
//   w3     e.x | e.y << 8 | e.z << 16 (int8 exponents: child box scale 2^e per axis) | imask << 24 (interior slots)
//   w4     baseChild: index of the first interior child (interior children are contiguous, in slot order)
//   w5     baseTri: index of the node's first triangle (the triangles of its leaf slots are contiguous, in slot order)
//   w6, w7 meta of slots 0–3, 4–7 (one byte each): interior 0b001_11sss (= 0x20 | (24 + s)); leaf (unary(count) << 5) |
//          offset (offset of its first triangle relative to baseTri, count 1..3); empty 0
//   w8..19 quantized child boxes, bytes per slot: qlo.x (w8 slots 0–3, w9 4–7), qlo.y (w10, w11), qlo.z (w12, w13),
//          qhi.x (w14, w15), qhi.y (w16, w17), qhi.z (w18, w19)
// Child box = [p + qlo·2^e, p + qhi·2^e] evaluated in f32 (one rounding: qlo·2^e is exact). The encoder picks qlo / qhi
// so that the DEQUANTIZED f32 box contains the child's f32 box (floor / ceil in exact f64 arithmetic; rounding to the
// nearest f32 is monotonic and the child bounds are f32), so traversal with the same robust slab test as BVH2 never
// culls a triangle BVH2 would test: closest-hit t / primId and any-hit results equal BVH2's up to the documented tie rule
// (exact-t ties may resolve to another triangle: traversal order differs) and FMA contraction differences between
// separately compiled pipelines (T12 CWBVH, m8-perf.md §3).
// Triangles: the BVH2's own records (MT `tris` / Woop `trisW` + primId tail), bit-copied into CWBVH order, so both
// intersectors and the canonical Woop edge order are unchanged.
import { NODE_FLOATS, TRI_FLOATS, isLeafRef, leafCount, leafFirst, u32View, woopFloatLength, woopPrimIdIndex, type BvhData, type BvhGpuBuffers } from './layout.ts';
import { buildBvh } from './sah-builder.ts';

export const CW_NODE_WORDS = 20;
export const CW_NODE_VEC4S = 5;
export const CW_MAX_LEAF = 3;
/** Group stack entries of traverse-cwbvh.wgsl (one entry per open node group; depth ≤ the 8-wide tree depth). */
export const CW_STACK_SIZE = 16;
/** Smallest exponent of a child box scale (2^-100 · 255 keeps every dequantized product a normal f32). */
export const CW_MIN_EXP = -100;

export interface CwbvhStats {
  nodeCount: number;
  triRefs: number;
  /** Max 8-wide depth (root = 1): bounds the group stack (one push per level). */
  maxDepth: number;
  avgChildren: number;
  leafSlots: number;
  sahCost: number;
  buildMs: number;
}

export interface CwbvhData {
  /** CW_NODE_WORDS per node; root = node 0. */
  nodes: Uint32Array;
  /** MT records (TRI_FLOATS per triangle) in CWBVH order; empty if the BVH2 had none. */
  tris: Float32Array;
  /** Woop records + primId tail in CWBVH order; empty if the BVH2 had none. */
  trisW: Float32Array;
  /** CWBVH triangle index → primId. */
  primOrder: Uint32Array;
  stats: CwbvhStats;
}

interface Item { lo: [number, number, number]; hi: [number, number, number]; ref: number }

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const halfArea = (it: Item) => {
  const dx = it.hi[0] - it.lo[0], dy = it.hi[1] - it.lo[1], dz = it.hi[2] - it.lo[2];
  return dx * dy + dy * dz + dz * dx;
};

/** Collapse a BVH2 (leaves ≤ 3 triangles) to CWBVH with the SAH-optimal dynamic-programming collapse (Ylitie et al.
 *  2017; m8-hwrt-tinybvh-gigi.md §2: tinybvh's conversion; gap-perf §10.2's greedy collapse measured worse, m8-perf §3). */
export function buildCwbvh(b: BvhData): CwbvhData {
  const t0 = now();
  const nu = u32View(b.nodes);
  const nNodes2 = b.nodes.length / NODE_FLOATS;
  const child = (node: number, side: 0 | 1): Item => {
    const o = node * NODE_FLOATS + side * 8;
    // boxes: side·8 + (0..2 min, 4..6 max); refs: left in float 3 (lmin.w), right in float 7 (lmax.w) (layout.ts)
    return { lo: [b.nodes[o], b.nodes[o + 1], b.nodes[o + 2]], hi: [b.nodes[o + 4], b.nodes[o + 5], b.nodes[o + 6]], ref: nu[node * NODE_FLOATS + 3 + 4 * side] };
  };
  for (let i = 0; i < nNodes2; i++) for (const s of [0, 1] as const) {
    const r = nu[i * NODE_FLOATS + 3 + 4 * s];
    if (isLeafRef(r) && leafCount(r) > CW_MAX_LEAF) throw new RangeError(`CWBVH needs BVH2 leaves of ≤ ${CW_MAX_LEAF} triangles (leaf of ${leafCount(r)})`);
  }
  // Subtree triangle ranges (the builder's leaves are depth-first, so a subtree's triangles are one contiguous range).
  const subFirst = new Int32Array(nNodes2), subCount = new Int32Array(nNodes2), subEnd = new Int32Array(nNodes2);
  const refRange = (r: number): [number, number, number] => (isLeafRef(r) ? [leafFirst(r), leafCount(r), leafFirst(r) + leafCount(r)] : [subFirst[r], subCount[r], subEnd[r]]);
  for (let i = nNodes2 - 1; i >= 0; i--) {   // children have larger indices than their parent (preorder)
    const [fa, ca, ea] = refRange(nu[i * NODE_FLOATS + 3]), [fb, cb, eb] = refRange(nu[i * NODE_FLOATS + 7]);
    subFirst[i] = Math.min(fa, fb); subCount[i] = ca + cb; subEnd[i] = Math.max(ea, eb);
  }
  // SAH-optimal collapse (Ylitie et al. 2017 §4.1; tinybvh's BVH8_CWBVH conversion): C(n, i) = cost of subtree n as a
  // forest of ≤ i roots; C(n, 1) = min(leaf: c_prim·A·T if T ≤ 3, internal: c_node·A + dist(n, 8)); C(n, i) =
  // min(C(n, i − 1), dist(n, i)); dist(n, j) = min_k C(left, k) + C(right, j − k). Bottom-up over the BVH2.
  const C_NODE = 1.0, C_PRIM = 0.3;
  const area = (it: Item) => Math.max(0, halfArea(it));
  const C = new Float64Array(nNodes2 * 8).fill(Infinity);
  const K = new Uint8Array(nNodes2 * 9);
  const fewer = new Uint8Array(nNodes2 * 8);
  const leafDec = new Uint8Array(nNodes2);
  const cOf = (it: Item, i: number) => (isLeafRef(it.ref) ? C_PRIM * area(it) * leafCount(it.ref) : C[it.ref * 8 + i]);
  const dist = new Float64Array(9);
  for (let n = nNodes2 - 1; n >= 0; n--) {
    const a = child(n, 0), bb = child(n, 1);
    const A = area({ lo: [Math.min(a.lo[0], bb.lo[0]), Math.min(a.lo[1], bb.lo[1]), Math.min(a.lo[2], bb.lo[2])], hi: [Math.max(a.hi[0], bb.hi[0]), Math.max(a.hi[1], bb.hi[1]), Math.max(a.hi[2], bb.hi[2])], ref: 0 });
    for (let j = 2; j <= 8; j++) {
      let best = Infinity, bk = 1;
      for (let k = 1; k < j; k++) { const v = cOf(a, Math.min(k, 7)) + cOf(bb, Math.min(j - k, 7)); if (v < best) { best = v; bk = k; } }
      dist[j] = best; K[n * 9 + j] = bk;
    }
    const internal = C_NODE * A + dist[8];
    const T = subCount[n];
    const leafC = n !== 0 && T <= CW_MAX_LEAF && subEnd[n] - subFirst[n] === T ? C_PRIM * A * T : Infinity;
    leafDec[n] = leafC <= internal ? 1 : 0;
    C[n * 8 + 1] = Math.min(leafC, internal);
    for (let i = 2; i <= 7; i++) {
      const prev = C[n * 8 + i - 1];
      if (prev <= dist[i]) { C[n * 8 + i] = prev; fewer[n * 8 + i] = 1; } else C[n * 8 + i] = dist[i];
    }
  }
  /** The ≤ j roots of item `it` in the optimal forest. */
  const roots = (it: Item, j: number, out: Item[]) => {
    if (isLeafRef(it.ref) || j <= 1) { out.push(it); return; }
    const r = it.ref;
    let i = Math.min(j, 7);
    while (i > 1 && fewer[r * 8 + i]) i--;
    if (i === 1) { out.push(it); return; }
    const k = K[r * 9 + i];
    roots(child(r, 0), k, out); roots(child(r, 1), i - k, out);
  };
  /** Children of the CWBVH node made from BVH2 node n: its optimal 8-root forest; leaf decisions become leaf refs. */
  const distribute = (n: number): Item[] => {
    const out: Item[] = [];
    const k = K[n * 9 + 8];
    roots(child(n, 0), k, out); roots(child(n, 1), 8 - k, out);
    return out.map((it) => (!isLeafRef(it.ref) && leafDec[it.ref] ? { ...it, ref: (0x80000000 | (subCount[it.ref] << 24) | subFirst[it.ref]) >>> 0 } : it));
  };
  const rootBox = { lo: [b.bounds.min[0], b.bounds.min[1], b.bounds.min[2]], hi: [b.bounds.max[0], b.bounds.max[1], b.bounds.max[2]] };
  const rootArea = Math.max(1e-30, (rootBox.hi[0] - rootBox.lo[0]) * (rootBox.hi[1] - rootBox.lo[1]) + (rootBox.hi[1] - rootBox.lo[1]) * (rootBox.hi[2] - rootBox.lo[2]) + (rootBox.hi[2] - rootBox.lo[2]) * (rootBox.hi[0] - rootBox.lo[0]));

  const words: number[] = [];
  const triOrder: number[] = [];           // CWBVH triangle i → BVH2 leaf-ordered triangle index
  const queue: { node: number; depth: number }[] = [{ node: 0, depth: 1 }];
  let nodeCount = 1, maxDepth = 1, childSum = 0, leafSlots = 0, sah = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const { node: bnode, depth } = queue[qi];
    const items = distribute(bnode);
    if (items.length > 8 || items.length < 1) throw new Error(`CWBVH: ${items.length} children`);
    // Node box (f32 values: the children's boxes are f32).
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const it of items) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], it.lo[k]); hi[k] = Math.max(hi[k], it.hi[k]); }
    const nodeArea = (hi[0] - lo[0]) * (hi[1] - lo[1]) + (hi[1] - lo[1]) * (hi[2] - lo[2]) + (hi[2] - lo[2]) * (hi[0] - lo[0]);
    sah += nodeArea / rootArea;
    // Octant-ordered slot assignment: cost(child, s) = dot(sign_s, centroid − node centroid), greedy smallest first.
    const c = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])];
    const cost: { i: number; s: number; v: number }[] = [];
    items.forEach((it, i) => {
      const d = [0, 1, 2].map((k) => 0.5 * (it.lo[k] + it.hi[k]) - c[k]);
      for (let s = 0; s < 8; s++) cost.push({ i, s, v: (s & 4 ? -d[0] : d[0]) + (s & 2 ? -d[1] : d[1]) + (s & 1 ? -d[2] : d[2]) });
    });
    cost.sort((x, y) => x.v - y.v || x.i - y.i || x.s - y.s);
    const slotOf = new Array<number>(items.length).fill(-1);
    const used = new Array<boolean>(8).fill(false);
    for (const e of cost) if (slotOf[e.i] < 0 && !used[e.s]) { slotOf[e.i] = e.s; used[e.s] = true; }
    const bySlot: (Item | undefined)[] = new Array(8).fill(undefined);
    items.forEach((it, i) => { bySlot[slotOf[i]] = it; });
    // Exponents: 2^e · 255 ≥ extent, e ≥ CW_MIN_EXP.
    const e = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      const ext = hi[k] - lo[k];
      let ek = ext > 0 ? Math.ceil(Math.log2(ext / 255)) : CW_MIN_EXP;
      ek = Math.max(CW_MIN_EXP, ek);
      while (2 ** ek * 255 < ext) ek++;
      if (ek > 127) throw new RangeError('CWBVH: node extent too large for an int8 exponent');
      e[k] = ek;
    }
    const baseChild = nodeCount;
    const baseTri = triOrder.length;
    let imask = 0;
    const meta = new Array<number>(8).fill(0);
    const qlo = [new Array<number>(8).fill(0), new Array<number>(8).fill(0), new Array<number>(8).fill(0)];
    const qhi = [new Array<number>(8).fill(0), new Array<number>(8).fill(0), new Array<number>(8).fill(0)];
    for (let s = 0; s < 8; s++) {
      const it = bySlot[s];
      if (!it) continue;
      for (let k = 0; k < 3; k++) {
        const sc = 2 ** e[k];
        let a = Math.floor((it.lo[k] - lo[k]) / sc), z = Math.ceil((it.hi[k] - lo[k]) / sc);
        a = Math.min(255, Math.max(0, a)); z = Math.min(255, Math.max(0, z));
        // the f32 dequantization the shader computes must contain the child box (exact f64 ⇒ holds; asserted)
        if (Math.fround(lo[k] + a * sc) > it.lo[k] || Math.fround(lo[k] + z * sc) < it.hi[k]) throw new Error('CWBVH: non-conservative quantization');
        qlo[k][s] = a; qhi[k][s] = z;
      }
      if (isLeafRef(it.ref)) {
        const cnt = leafCount(it.ref), first = leafFirst(it.ref);
        const off = triOrder.length - baseTri;
        meta[s] = (((1 << cnt) - 1) << 5) | off;
        for (let j = 0; j < cnt; j++) triOrder.push(first + j);
        leafSlots++;
      } else {
        imask |= 1 << s;
        meta[s] = 0x20 | (24 + s);
      }
    }
    // interior children in slot order → consecutive node indices (BFS)
    for (let s = 0; s < 8; s++) {
      const it = bySlot[s];
      if (it && !isLeafRef(it.ref)) { queue.push({ node: it.ref, depth: depth + 1 }); nodeCount++; }
    }
    if (triOrder.length - baseTri > 24) throw new Error('CWBVH: more than 24 triangles in one node');
    childSum += items.length;
    maxDepth = Math.max(maxDepth, depth);
    const f = new Float32Array(3); f.set(lo);
    const fu = new Uint32Array(f.buffer);
    const pack4 = (a: number[], o: number) => (a[o] | (a[o + 1] << 8) | (a[o + 2] << 16) | (a[o + 3] << 24)) >>> 0;
    words.push(fu[0], fu[1], fu[2],
      ((e[0] & 0xff) | ((e[1] & 0xff) << 8) | ((e[2] & 0xff) << 16) | (imask << 24)) >>> 0,
      baseChild, baseTri, pack4(meta, 0), pack4(meta, 4),
      pack4(qlo[0], 0), pack4(qlo[0], 4), pack4(qlo[1], 0), pack4(qlo[1], 4), pack4(qlo[2], 0), pack4(qlo[2], 4),
      pack4(qhi[0], 0), pack4(qhi[0], 4), pack4(qhi[1], 0), pack4(qhi[1], 4), pack4(qhi[2], 0), pack4(qhi[2], 4));
  }
  if (nodeCount !== queue.length) throw new Error('CWBVH: node count mismatch');

  // Triangles bit-copied from the BVH2 records into CWBVH order.
  const n = triOrder.length;
  const hasMT = b.tris.length > 0, hasW = b.trisW.length > 0;
  const tris = hasMT ? new Float32Array(n * TRI_FLOATS) : new Float32Array(0);
  const trisW = hasW ? new Float32Array(woopFloatLength(n)) : new Float32Array(0);
  const trisU = u32View(tris), trisWU = u32View(trisW), srcU = u32View(b.tris), srcWU = u32View(b.trisW);
  const primOrder = new Uint32Array(n);
  const srcWVec4 = b.trisW.length / 4, dstWVec4 = trisW.length / 4;
  for (let i = 0; i < n; i++) {
    const L = triOrder[i];
    primOrder[i] = b.primOrder[L];
    if (hasMT) trisU.set(srcU.subarray(L * TRI_FLOATS, (L + 1) * TRI_FLOATS), i * TRI_FLOATS);
    if (hasW) {
      trisWU.set(srcWU.subarray(L * TRI_FLOATS, (L + 1) * TRI_FLOATS), i * TRI_FLOATS);
      trisWU[woopPrimIdIndex(dstWVec4, i)] = srcWU[woopPrimIdIndex(srcWVec4, L)];
    }
  }
  return {
    nodes: Uint32Array.from(words), tris, trisW, primOrder,
    stats: { nodeCount, triRefs: n, maxDepth, avgChildren: childSum / nodeCount, leafSlots, sahCost: sah, buildMs: now() - t0 },
  };
}

/** BVH2 with leaves of ≤ 3 triangles (the CWBVH input) + its CWBVH. */
export function buildCwbvhFromMesh(positions: Float32Array, indices: Uint32Array, opts: { mt?: boolean; woop?: boolean } = {}): { bvh2: BvhData; cw: CwbvhData } {
  const bvh2 = buildBvh(positions, indices, { maxLeafSize: CW_MAX_LEAF, mt: opts.mt ?? true, woop: opts.woop ?? true });
  return { bvh2, cw: buildCwbvh(bvh2) };
}

/** Upload the CWBVH nodes + the chosen triangle layout (same binding shapes as uploadBvh: nodes read as vec4f). */
export function uploadCwbvh(device: GPUDevice, cw: CwbvhData, opts: { watertight: boolean; label?: string; extraUsage?: number }): BvhGpuBuffers {
  const tris = opts.watertight ? cw.trisW : cw.tris;
  if (tris.length === 0) throw new Error(`CWBVH was built without the ${opts.watertight ? 'Woop' : 'MT'} triangle layout`);
  const usage = GPUBufferUsage.STORAGE | (opts.extraUsage ?? 0);
  const make = (src: ArrayBufferView, name: string) => {
    const buf = device.createBuffer({ size: Math.max(16, src.byteLength), usage, mappedAtCreation: true, label: `${opts.label ?? 'cwbvh'}.${name}` });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(src.buffer, src.byteOffset, src.byteLength));
    buf.unmap();
    return buf;
  };
  return { nodes: make(cw.nodes, 'nodes'), tris: make(tris, opts.watertight ? 'trisW' : 'tris'), triCount: cw.primOrder.length, watertight: opts.watertight };
}
