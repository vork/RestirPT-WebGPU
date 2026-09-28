// Binned-SAH BVH2 builder emitting the Aila–Laine GPU layout (plan §1.3; layout contract in layout.ts).
// 16 centroid bins per axis, C_trav = C_isect = 1, leaves of 1..4 triangles, depth cap 30 (forced median splits
// guarantee every leaf fits within the cap). Degenerate-safe: zero-extent centroid bounds fall back to index
// halving; non-finite triangles are left out (never hit) and counted. Single-threaded; run it in the Worker.
import {
  BVH_MAX_DEPTH, BVH_MISS, MAX_TRIANGLES, NODE_FLOATS, TRI_FLOATS, encodeLeaf, u32View, woopFloatLength, woopPrimIdIndex,
  type BvhData, type BvhStats,
} from './layout.ts';

export interface BvhBuildOptions {
  maxLeafSize?: number;   // default 4
  bins?: number;          // default 16
  maxDepth?: number;      // default 30
  /** Which triangle layouts to emit (default both). */
  mt?: boolean;
  woop?: boolean;
}

const C_TRAV = 1;
const C_ISECT = 1;

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());

export function buildBvh(positions: Float32Array, indices: Uint32Array, opts: BvhBuildOptions = {}): BvhData {
  const t0 = now();
  const maxLeaf = opts.maxLeafSize ?? 4;
  const nBins = opts.bins ?? 16;
  const maxDepth = opts.maxDepth ?? BVH_MAX_DEPTH;
  const emitMT = opts.mt ?? true;
  const emitWoop = opts.woop ?? true;
  const nTriIn = Math.floor(indices.length / 3);
  if (nTriIn > MAX_TRIANGLES) throw new RangeError(`scene has ${nTriIn} triangles (> 2^24, plan §1.3)`);

  // Per-triangle bounds + centroids; non-finite triangles are skipped.
  const bmin = new Float32Array(nTriIn * 3), bmax = new Float32Array(nTriIn * 3), cen = new Float32Array(nTriIn * 3);
  const idx = new Uint32Array(nTriIn);
  let n = 0, skipped = 0;
  for (let p = 0; p < nTriIn; p++) {
    const a = indices[3 * p] * 3, b = indices[3 * p + 1] * 3, c = indices[3 * p + 2] * 3;
    let ok = true;
    for (let k = 0; k < 3; k++) {
      const x = positions[a + k], y = positions[b + k], z = positions[c + k];
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) { ok = false; break; }
      const lo = Math.min(x, y, z), hi = Math.max(x, y, z);
      bmin[3 * p + k] = lo; bmax[3 * p + k] = hi; cen[3 * p + k] = 0.5 * lo + 0.5 * hi;
    }
    if (ok) idx[n++] = p; else skipped++;
  }

  const stats: BvhStats = {
    triCount: n, skippedNonFinite: skipped, nodeCount: 0, leafCount: 0, maxDepth: 0, maxLeafSize: 0, avgLeafSize: 0,
    sahCost: 0, forcedSplits: 0, buildMs: 0,
  };
  if (n === 0) return emptyBvh(positions.length / 3, stats, t0, emitMT, emitWoop);

  // Root bounds.
  const root = new Float64Array(6).fill(0);
  root[0] = root[1] = root[2] = Infinity; root[3] = root[4] = root[5] = -Infinity;
  for (let i = 0; i < n; i++) {
    const p = idx[i] * 3;
    for (let k = 0; k < 3; k++) {
      if (bmin[p + k] < root[k]) root[k] = bmin[p + k];
      if (bmax[p + k] > root[3 + k]) root[3 + k] = bmax[p + k];
    }
  }
  const rootArea = halfArea(root, 0);

  const maxNodes = Math.max(1, n - 1);
  const nodes = new Float32Array(maxNodes * NODE_FLOATS);
  const nodesU = u32View(nodes);
  let nodeCount = 1; // node 0 = root

  // Scratch for binning: per axis per bin [minx,miny,minz,maxx,maxy,maxz] and counts; sweep areas.
  const binB = new Float64Array(3 * nBins * 6);
  const binN = new Int32Array(3 * nBins);
  const rArea = new Float64Array(nBins);
  const rCnt = new Int32Array(nBins);
  const acc = new Float64Array(6);
  const pc0 = [0, 0, 0], pk = [0, 0, 0]; // binning parameters of the last findSahSplit (reused by partitionBin)

  // Task stack. Each task = a range [s, e) whose bounds are known; it is either the root (interior forced) or
  // a child slot of an already-written interior node (parent, side 0 = left / 1 = right).
  interface Task { s: number; e: number; depth: number; parent: number; side: number; b: Float64Array }
  const stack: Task[] = [{ s: 0, e: n, depth: 0, parent: -1, side: 0, b: root }];
  let sah = 0, leafTris = 0;

  const writeChild = (parent: number, side: number, b: Float64Array, ref: number) => {
    const o = parent * NODE_FLOATS + side * 8;
    nodes[o] = b[0]; nodes[o + 1] = b[1]; nodes[o + 2] = b[2];
    nodes[o + 4] = b[3]; nodes[o + 5] = b[4]; nodes[o + 6] = b[5];
    // Left ref in lmin.w (float 3), right ref in lmax.w (float 7); rmin.w / rmax.w (floats 11, 15) stay 0.
    nodesU[parent * NODE_FLOATS + 3 + 4 * side] = ref;
  };

  while (stack.length) {
    const task = stack.pop()!;
    const { s, e, depth, parent, side, b } = task;
    const cnt = e - s;
    const area = halfArea(b, 0);
    const isRoot = parent < 0;

    // Decide: leaf or split. Leaf depth = depth (the number of interior nodes above it).
    const levelsNeeded = cnt > maxLeaf ? Math.ceil(Math.log2(cnt / maxLeaf)) : 0;
    const forceMedian = depth + levelsNeeded >= maxDepth && cnt > maxLeaf;
    let axis = -1, splitBin = -1, splitCost = Infinity;
    if (!forceMedian && cnt > 1) {
      ({ axis, splitBin, splitCost } = findSahSplit(s, e));
    }
    const leafCost = C_ISECT * cnt * area;
    if (!isRoot && cnt <= maxLeaf && (cnt === 1 || leafCost <= C_TRAV * area + splitCost || axis < 0)) {
      writeChild(parent, side, b, encodeLeaf(cnt, s));
      stats.leafCount++; leafTris += cnt;
      if (cnt > stats.maxLeafSize) stats.maxLeafSize = cnt;
      if (depth > stats.maxDepth) stats.maxDepth = depth;
      if (rootArea > 0) sah += (C_ISECT * cnt * area) / rootArea;
      continue;
    }

    // Interior node.
    const node = isRoot ? 0 : nodeCount++;
    if (!isRoot) writeChild(parent, side, b, node);
    if (rootArea > 0) sah += (C_TRAV * area) / rootArea;

    let mid: number;
    if (cnt === 1) {
      // Only reachable for a single-triangle root: both children reference the same triangle.
      stack.push({ s, e, depth: depth + 1, parent: node, side: 1, b: copyB(b) }, { s, e, depth: depth + 1, parent: node, side: 0, b: copyB(b) });
      stats.forcedSplits++;
      continue;
    }
    if (axis >= 0 && !forceMedian) {
      mid = partitionBin(s, e, axis, splitBin);
    } else {
      mid = medianSplit(s, e);
      stats.forcedSplits++;
    }
    if (mid <= s || mid >= e) { mid = (s + e) >>> 1; stats.forcedSplits++; } // cannot happen; belt and braces
    const lb = rangeBounds(s, mid), rb = rangeBounds(mid, e);
    // Right pushed first → left processed first → depth-first preorder, left subtree contiguous.
    stack.push({ s: mid, e, depth: depth + 1, parent: node, side: 1, b: rb }, { s, e: mid, depth: depth + 1, parent: node, side: 0, b: lb });
  }

  stats.nodeCount = nodeCount;
  stats.sahCost = sah;
  stats.avgLeafSize = leafTris / Math.max(1, stats.leafCount);

  // ------------------------------------------------------------------------------------------------------------
  function findSahSplit(s: number, e: number): { axis: number; splitBin: number; splitCost: number } {
    // Centroid bounds.
    let c0x = Infinity, c0y = Infinity, c0z = Infinity, c1x = -Infinity, c1y = -Infinity, c1z = -Infinity;
    for (let i = s; i < e; i++) {
      const p = idx[i] * 3;
      const x = cen[p], y = cen[p + 1], z = cen[p + 2];
      if (x < c0x) c0x = x; if (x > c1x) c1x = x;
      if (y < c0y) c0y = y; if (y > c1y) c1y = y;
      if (z < c0z) c0z = z; if (z > c1z) c1z = z;
    }
    const ex = c1x - c0x, ey = c1y - c0y, ez = c1z - c0z;
    if (!(ex > 0) && !(ey > 0) && !(ez > 0)) return { axis: -1, splitBin: -1, splitCost: Infinity };
    const kx = ex > 0 ? nBins / ex : 0, ky = ey > 0 ? nBins / ey : 0, kz = ez > 0 ? nBins / ez : 0;
    binN.fill(0);
    for (let j = 0; j < 3 * nBins; j++) {
      const o = j * 6;
      binB[o] = binB[o + 1] = binB[o + 2] = Infinity; binB[o + 3] = binB[o + 4] = binB[o + 5] = -Infinity;
    }
    const last = nBins - 1;
    for (let i = s; i < e; i++) {
      const p = idx[i] * 3;
      const bx = Math.min(last, ((cen[p] - c0x) * kx) | 0);
      const by = Math.min(last, ((cen[p + 1] - c0y) * ky) | 0) + nBins;
      const bz = Math.min(last, ((cen[p + 2] - c0z) * kz) | 0) + 2 * nBins;
      const x0 = bmin[p], y0 = bmin[p + 1], z0 = bmin[p + 2], x1 = bmax[p], y1 = bmax[p + 1], z1 = bmax[p + 2];
      for (let q = 0; q < 3; q++) {
        const bin = q === 0 ? bx : q === 1 ? by : bz;
        binN[bin]++;
        const o = bin * 6;
        if (x0 < binB[o]) binB[o] = x0; if (y0 < binB[o + 1]) binB[o + 1] = y0; if (z0 < binB[o + 2]) binB[o + 2] = z0;
        if (x1 > binB[o + 3]) binB[o + 3] = x1; if (y1 > binB[o + 4]) binB[o + 4] = y1; if (z1 > binB[o + 5]) binB[o + 5] = z1;
      }
    }
    let bestAxis = -1, bestBin = -1, bestCost = Infinity;
    const ext = [ex, ey, ez];
    for (let a = 0; a < 3; a++) {
      if (!(ext[a] > 0)) continue;
      const base = a * nBins;
      // Right-to-left sweep: rArea[i] / rCnt[i] = bins [i, nBins).
      resetAcc(); let c = 0;
      for (let i = last; i >= 1; i--) {
        growAcc(binB, (base + i) * 6); c += binN[base + i];
        rArea[i] = halfArea(acc, 0); rCnt[i] = c;
      }
      resetAcc(); c = 0;
      for (let i = 1; i <= last; i++) {
        growAcc(binB, (base + i - 1) * 6); c += binN[base + i - 1];
        if (c === 0 || rCnt[i] === 0) continue;
        const cost = C_ISECT * (c * halfArea(acc, 0) + rCnt[i] * rArea[i]);
        if (cost < bestCost) { bestCost = cost; bestAxis = a; bestBin = i; }
      }
    }
    // partitionBin must re-bin with exactly these parameters.
    pc0[0] = c0x; pc0[1] = c0y; pc0[2] = c0z; pk[0] = kx; pk[1] = ky; pk[2] = kz;
    return { axis: bestAxis, splitBin: bestBin, splitCost: bestCost };

    function resetAcc() { acc[0] = acc[1] = acc[2] = Infinity; acc[3] = acc[4] = acc[5] = -Infinity; }
    function growAcc(src: Float64Array, o: number) { grow(acc, src, o); }
  }

  function partitionBin(s: number, e: number, axis: number, splitBin: number): number {
    const c0 = pc0[axis], k = pk[axis], last = nBins - 1;
    let i = s, j = e - 1;
    while (i <= j) {
      const bi = Math.min(last, ((cen[idx[i] * 3 + axis] - c0) * k) | 0);
      if (bi < splitBin) i++;
      else { const t = idx[i]; idx[i] = idx[j]; idx[j] = t; j--; }
    }
    return i;
  }

  function medianSplit(s: number, e: number): number {
    // Object median along the largest centroid extent; index halving if all centroids coincide.
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = s; i < e; i++) for (let k = 0; k < 3; k++) { const v = cen[idx[i] * 3 + k]; if (v < lo[k]) lo[k] = v; if (v > hi[k]) hi[k] = v; }
    const ext = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    const a = ext[0] >= ext[1] && ext[0] >= ext[2] ? 0 : ext[1] >= ext[2] ? 1 : 2;
    const mid = (s + e) >>> 1;
    if (ext[a] > 0) {
      const sub = Array.from(idx.subarray(s, e)).sort((p, q) => cen[p * 3 + a] - cen[q * 3 + a] || p - q);
      idx.set(sub, s);
    }
    return mid;
  }

  function rangeBounds(s: number, e: number): Float64Array {
    const r = new Float64Array(6);
    r[0] = r[1] = r[2] = Infinity; r[3] = r[4] = r[5] = -Infinity;
    for (let i = s; i < e; i++) {
      const p = idx[i] * 3;
      for (let k = 0; k < 3; k++) {
        if (bmin[p + k] < r[k]) r[k] = bmin[p + k];
        if (bmax[p + k] > r[3 + k]) r[3 + k] = bmax[p + k];
      }
    }
    return r;
  }

  // ------------------------------------------------------------------------------------------------------------
  // Leaf-ordered triangle arrays.
  const primOrder = idx.slice(0, n);
  const weldedVid = weldVertices(positions);
  const tris = emitMT ? new Float32Array(n * TRI_FLOATS) : new Float32Array(0);
  const trisW = emitWoop ? new Float32Array(woopFloatLength(n)) : new Float32Array(0);
  const trisU = u32View(tris), trisWU = u32View(trisW);
  const wVec4 = trisW.length / 4;
  for (let i = 0; i < n; i++) {
    const p = primOrder[i];
    const ia = indices[3 * p], ib = indices[3 * p + 1], ic = indices[3 * p + 2];
    const a = ia * 3, b = ib * 3, c = ic * 3;
    if (emitMT) {
      const o = i * TRI_FLOATS;
      for (let k = 0; k < 3; k++) {
        tris[o + k] = positions[a + k];
        tris[o + 4 + k] = positions[b + k] - positions[a + k]; // exact in f64, rounded once to f32
        tris[o + 8 + k] = positions[c + k] - positions[a + k];
      }
      trisU[o + 3] = p; trisU[o + 7] = 0; trisU[o + 11] = 0;
    }
    if (emitWoop) {
      const o = i * TRI_FLOATS;
      for (let k = 0; k < 3; k++) { trisW[o + k] = positions[a + k]; trisW[o + 4 + k] = positions[b + k]; trisW[o + 8 + k] = positions[c + k]; }
      trisWU[o + 3] = weldedVid[ia]; trisWU[o + 7] = weldedVid[ib]; trisWU[o + 11] = weldedVid[ic];
      trisWU[woopPrimIdIndex(wVec4, i)] = p;
    }
  }

  stats.buildMs = now() - t0;
  return {
    nodes: nodes.slice(0, nodeCount * NODE_FLOATS), tris, trisW, primOrder, weldedVid,
    bounds: { min: [root[0], root[1], root[2]], max: [root[3], root[4], root[5]] }, stats,
  };
}

function halfArea(b: Float64Array, o: number): number {
  const dx = b[o + 3] - b[o], dy = b[o + 4] - b[o + 1], dz = b[o + 5] - b[o + 2];
  if (!(dx >= 0) || !(dy >= 0) || !(dz >= 0)) return 0;
  return dx * dy + dy * dz + dz * dx;
}
function grow(dst: Float64Array, src: Float64Array, o: number) {
  if (src[o] < dst[0]) dst[0] = src[o]; if (src[o + 1] < dst[1]) dst[1] = src[o + 1]; if (src[o + 2] < dst[2]) dst[2] = src[o + 2];
  if (src[o + 3] > dst[3]) dst[3] = src[o + 3]; if (src[o + 4] > dst[4]) dst[4] = src[o + 4]; if (src[o + 5] > dst[5]) dst[5] = src[o + 5];
}
const copyB = (b: Float64Array): Float64Array => Float64Array.from(b);

/** Position-welded vertex ids: vid[v] = smallest vertex index with the same position (−0 == +0). */
export function weldVertices(positions: Float32Array): Uint32Array {
  const nv = Math.floor(positions.length / 3);
  const out = new Uint32Array(nv);
  let size = 1;
  while (size < nv * 2) size <<= 1;
  const table = new Int32Array(Math.max(1, size)).fill(-1);
  const bits = new Uint32Array(positions.buffer, positions.byteOffset, nv * 3);
  const mask = size - 1;
  for (let v = 0; v < nv; v++) {
    const x = positions[3 * v], y = positions[3 * v + 1], z = positions[3 * v + 2];
    const hx = x === 0 ? 0 : bits[3 * v], hy = y === 0 ? 0 : bits[3 * v + 1], hz = z === 0 ? 0 : bits[3 * v + 2];
    let h = (Math.imul(hx, 0x9e3779b1) ^ Math.imul(hy, 0x85ebca77) ^ Math.imul(hz, 0xc2b2ae3d)) >>> 0;
    h = (h ^ (h >>> 15)) & mask;
    for (;;) {
      const w = table[h];
      if (w < 0) { table[h] = v; out[v] = v; break; }
      if (positions[3 * w] === x && positions[3 * w + 1] === y && positions[3 * w + 2] === z) { out[v] = w; break; }
      h = (h + 1) & mask;
    }
  }
  return out;
}

/** No finite triangles: a root whose two leaves reference one zero-size dummy triangle (primId BVH_MISS) that
 *  can never be hit (MT a == 0, Woop det == 0), so traversal needs no special case. */
function emptyBvh(nv: number, stats: BvhStats, t0: number, mt: boolean, woop: boolean): BvhData {
  const nodes = new Float32Array(NODE_FLOATS);
  const nu = u32View(nodes);
  nu[3] = encodeLeaf(1, 0); nu[7] = encodeLeaf(1, 0);
  const tris = mt ? new Float32Array(TRI_FLOATS) : new Float32Array(0);
  if (mt) u32View(tris)[3] = BVH_MISS;
  const trisW = woop ? new Float32Array(woopFloatLength(1)) : new Float32Array(0);
  if (woop) { const u = u32View(trisW); u[3] = 0; u[7] = 1; u[11] = 2; u[woopPrimIdIndex(trisW.length / 4, 0)] = BVH_MISS; }
  stats.nodeCount = 1; stats.leafCount = 2; stats.buildMs = now() - t0;
  const weldedVid = new Uint32Array(Math.max(0, Math.floor(nv)));
  for (let i = 0; i < weldedVid.length; i++) weldedVid[i] = i;
  return { nodes, tris, trisW, primOrder: new Uint32Array([BVH_MISS]), weldedVid, bounds: { min: [0, 0, 0], max: [0, 0, 0] }, stats };
}
