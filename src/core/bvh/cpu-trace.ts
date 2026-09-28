// CPU reference ray casting in float64 (tests only): brute force over all triangles, and a traversal of the
// built BVH (same node layout as the GPU) with conservatively padded boxes. Triangles are always read from the
// ORIGINAL SceneGeometry arrays through primId, so the BVH's triangle layouts are not trusted here.
import { BVH_MISS, NODE_FLOATS, isLeafRef, leafCount, leafFirst, u32View, type BvhData } from './layout.ts';

export interface CpuHit { t: number; u: number; v: number; primId: number }
export const cpuMiss = (tmax = Infinity): CpuHit => ({ t: tmax, u: 0, v: 0, primId: BVH_MISS });

/** f64 Möller–Trumbore, two-sided, edges inclusive. Returns t (or NaN) and writes u, v into `uv`. */
export function intersectTri64(
  ox: number, oy: number, oz: number, dx: number, dy: number, dz: number,
  positions: Float32Array, indices: Uint32Array, prim: number, uv?: Float64Array, baryPad = 0,
): number {
  const a = indices[3 * prim] * 3, b = indices[3 * prim + 1] * 3, c = indices[3 * prim + 2] * 3;
  const v0x = positions[a], v0y = positions[a + 1], v0z = positions[a + 2];
  const e1x = positions[b] - v0x, e1y = positions[b + 1] - v0y, e1z = positions[b + 2] - v0z;
  const e2x = positions[c] - v0x, e2y = positions[c + 1] - v0y, e2z = positions[c + 2] - v0z;
  const hx = dy * e2z - dz * e2y, hy = dz * e2x - dx * e2z, hz = dx * e2y - dy * e2x;
  const det = e1x * hx + e1y * hy + e1z * hz;
  if (det === 0) return NaN;
  const f = 1 / det;
  const sx = ox - v0x, sy = oy - v0y, sz = oz - v0z;
  const u = f * (sx * hx + sy * hy + sz * hz);
  if (u < -baryPad || u > 1 + baryPad) return NaN;
  const qx = sy * e1z - sz * e1y, qy = sz * e1x - sx * e1z, qz = sx * e1y - sy * e1x;
  const v = f * (dx * qx + dy * qy + dz * qz);
  if (v < -baryPad || u + v > 1 + baryPad) return NaN;
  const t = f * (e2x * qx + e2y * qy + e2z * qz);
  if (uv) { uv[0] = u; uv[1] = v; }
  return t;
}

const uvScratch = new Float64Array(2);

/** Brute-force closest hit over every triangle, t in (0, tmax). Ties keep the smaller primId. */
export function bruteClosest(positions: Float32Array, indices: Uint32Array, o: ArrayLike<number>, d: ArrayLike<number>, tmax = Infinity): CpuHit {
  const hit = cpuMiss(tmax);
  const nTri = indices.length / 3;
  for (let p = 0; p < nTri; p++) {
    const t = intersectTri64(o[0], o[1], o[2], d[0], d[1], d[2], positions, indices, p, uvScratch);
    if (t > 0 && t < hit.t) { hit.t = t; hit.u = uvScratch[0]; hit.v = uvScratch[1]; hit.primId = p; }
  }
  return hit;
}

export function bruteAny(positions: Float32Array, indices: Uint32Array, o: ArrayLike<number>, d: ArrayLike<number>, tmax: number): boolean {
  const nTri = indices.length / 3;
  for (let p = 0; p < nTri; p++) {
    const t = intersectTri64(o[0], o[1], o[2], d[0], d[1], d[2], positions, indices, p);
    if (t > 0 && t < tmax) return true;
  }
  return false;
}

/**
 * f64 traversal of the built BVH. Box tests are padded by `boxPad` (relative to the box extent + 1e-7 absolute)
 * so f32 box rounding can never cull a hit; the triangle test is exact-in-f64 on the original vertices.
 * `anyHit` returns at the first hit in (0, tmax); hits on skipA / skipB are ignored (endpoint exclusion).
 * Ties keep the smaller primId (matches bruteClosest).
 */
export function bvhTrace64(
  bvh: BvhData, positions: Float32Array, indices: Uint32Array, o: ArrayLike<number>, d: ArrayLike<number>,
  tmax = Infinity, anyHit = false, skipA = BVH_MISS, skipB = BVH_MISS, boxPad = 1e-6,
): CpuHit {
  const nodes = bvh.nodes, nu = u32View(bvh.nodes), order = bvh.primOrder;
  const ox = o[0], oy = o[1], oz = o[2], dx = d[0], dy = d[1], dz = d[2];
  const rx = 1 / dx, ry = 1 / dy, rz = 1 / dz;
  const hit = cpuMiss(tmax);
  const stack = new Uint32Array(64);
  let sp = 0;
  let ref = 0; // start at the root interior node
  const slab = (b: number): boolean => {
    const x0 = nodes[b], y0 = nodes[b + 1], z0 = nodes[b + 2], x1 = nodes[b + 4], y1 = nodes[b + 5], z1 = nodes[b + 6];
    const px = (x1 - x0) * boxPad + 1e-7, py = (y1 - y0) * boxPad + 1e-7, pz = (z1 - z0) * boxPad + 1e-7;
    let tn = 0, tf = hit.t;
    // Direction components of exactly 0 → 1/0 = ±Inf; handle the parallel slab explicitly.
    if (dx === 0) { if (ox < x0 - px || ox > x1 + px) return false; } else {
      const a = (x0 - px - ox) * rx, c = (x1 + px - ox) * rx; tn = Math.max(tn, Math.min(a, c)); tf = Math.min(tf, Math.max(a, c));
    }
    if (dy === 0) { if (oy < y0 - py || oy > y1 + py) return false; } else {
      const a = (y0 - py - oy) * ry, c = (y1 + py - oy) * ry; tn = Math.max(tn, Math.min(a, c)); tf = Math.min(tf, Math.max(a, c));
    }
    if (dz === 0) { if (oz < z0 - pz || oz > z1 + pz) return false; } else {
      const a = (z0 - pz - oz) * rz, c = (z1 + pz - oz) * rz; tn = Math.max(tn, Math.min(a, c)); tf = Math.min(tf, Math.max(a, c));
    }
    return tn <= tf;
  };
  for (;;) {
    if (isLeafRef(ref)) {
      const first = leafFirst(ref), cnt = leafCount(ref);
      for (let i = first; i < first + cnt; i++) {
        const p = order[i];
        if (p === BVH_MISS || p === skipA || p === skipB) continue;
        const t = intersectTri64(ox, oy, oz, dx, dy, dz, positions, indices, p, uvScratch);
        if (t > 0 && (t < hit.t || (t === hit.t && p < hit.primId))) {
          hit.t = t; hit.u = uvScratch[0]; hit.v = uvScratch[1]; hit.primId = p;
          if (anyHit) return hit;
        }
      }
    } else {
      const base = ref * NODE_FLOATS;
      const hl = slab(base), hr = slab(base + 8);
      const l = nu[base + 3], r = nu[base + 7];
      if (hl && hr) { stack[sp++] = r; ref = l; continue; }
      if (hl) { ref = l; continue; }
      if (hr) { ref = r; continue; }
    }
    if (sp === 0) break;
    ref = stack[--sp];
  }
  return hit;
}

/** Structural invariants of a built BVH; returns a list of violations (empty = valid). */
export function checkBvhInvariants(bvh: BvhData, positions: Float32Array, indices: Uint32Array, maxLeaf = 4, maxDepth = 30): string[] {
  const errs: string[] = [];
  const nu = u32View(bvh.nodes), nodes = bvh.nodes;
  const nNodes = nodes.length / NODE_FLOATS;
  const seen = new Uint8Array(indices.length / 3);
  const leafSeen = new Uint8Array(bvh.primOrder.length);
  const visitedNode = new Uint8Array(nNodes);
  // DFS carrying the ancestor boxes that the subtree must lie within.
  const stack: { ref: number; depth: number; boxes: number[] }[] = [{ ref: 0, depth: 0, boxes: [] }];
  while (stack.length && errs.length < 20) {
    const { ref, depth, boxes } = stack.pop()!;
    if (isLeafRef(ref)) {
      const first = leafFirst(ref), cnt = leafCount(ref);
      if (cnt < 1 || cnt > maxLeaf) errs.push(`leaf count ${cnt}`);
      if (depth > maxDepth) errs.push(`leaf depth ${depth} > ${maxDepth}`);
      for (let i = first; i < first + cnt; i++) {
        if (i >= bvh.primOrder.length) { errs.push(`leaf tri ${i} out of range`); continue; }
        leafSeen[i]++;
        const p = bvh.primOrder[i];
        if (p === BVH_MISS) continue;
        seen[p]++;
        for (let k = 0; k < 3; k++) {
          const v = indices[3 * p + k] * 3;
          for (const b of boxes) {
            for (let a = 0; a < 3; a++) {
              const x = positions[v + a];
              if (x < nodes[b + a] || x > nodes[b + 4 + a]) { errs.push(`prim ${p} outside ancestor box`); break; }
            }
          }
        }
      }
      continue;
    }
    if (ref >= nNodes) { errs.push(`node ref ${ref} out of range`); continue; }
    if (visitedNode[ref]++) { errs.push(`node ${ref} reached twice`); continue; }
    const base = ref * NODE_FLOATS;
    if (nu[base + 11] !== 0 || nu[base + 15] !== 0) errs.push(`node ${ref} rmin.w/rmax.w not 0`);
    stack.push({ ref: nu[base + 7], depth: depth + 1, boxes: [...boxes, base + 8] });
    stack.push({ ref: nu[base + 3], depth: depth + 1, boxes: [...boxes, base] });
  }
  for (let i = 0; i < leafSeen.length; i++) if (leafSeen[i] === 0) { errs.push(`leaf tri ${i} unreferenced`); break; }
  const finite = (p: number) => { for (let k = 0; k < 3; k++) for (let a = 0; a < 3; a++) if (!Number.isFinite(positions[indices[3 * p + k] * 3 + a])) return false; return true; };
  for (let p = 0; p < seen.length; p++) {
    if (seen[p] > 1 && bvh.stats.triCount > 1) { errs.push(`prim ${p} referenced ${seen[p]}x`); break; }
    if (seen[p] === 0 && finite(p) && bvh.stats.triCount > 1) { errs.push(`prim ${p} missing`); break; }
  }
  for (let i = 0; i < nNodes; i++) if (!visitedNode[i]) { errs.push(`node ${i} unreachable`); break; }
  return errs;
}
