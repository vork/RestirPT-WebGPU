import { describe, expect, it } from 'vitest';
import { buildBvh, weldVertices } from '../../src/core/bvh/sah-builder.ts';
import { bruteAny, bruteClosest, bvhTrace64, checkBvhInvariants } from '../../src/core/bvh/cpu-trace.ts';
import {
  BVH_MISS, LEAF_BIT, NODE_FLOATS, TRI_FLOATS, encodeLeaf, isLeafRef, leafCount, leafFirst, u32View, woopPrimIdIndex,
} from '../../src/core/bvh/layout.ts';
import { icosphere, loadGltfMesh, meshBounds, proceduralScene, randomDir, rng, type Mesh } from './fixtures.ts';

function randomRays(m: Mesh, n: number, seed: number) {
  const r = rng(seed), b = meshBounds(m);
  const diag = Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
  const rays: { o: number[]; d: number[]; tmax: number }[] = [];
  for (let i = 0; i < n; i++) {
    const o = [0, 1, 2].map((k) => Math.fround(b.min[k] + (b.max[k] - b.min[k]) * (1.2 * r() - 0.1)));
    rays.push({ o, d: randomDir(r).map(Math.fround), tmax: Math.fround(r() * diag) });
  }
  return rays;
}

describe('BVH layout encoding', () => {
  it('leaf refs round-trip and reject out-of-contract values', () => {
    const ref = encodeLeaf(4, 0xabcdef);
    expect(isLeafRef(ref)).toBe(true);
    expect(leafCount(ref)).toBe(4);
    expect(leafFirst(ref)).toBe(0xabcdef);
    expect(ref >>> 0).toBe((LEAF_BIT | (4 << 24) | 0xabcdef) >>> 0);
    expect(isLeafRef(12345)).toBe(false);
    expect(() => encodeLeaf(0, 0)).toThrow();
    expect(() => encodeLeaf(128, 0)).toThrow();
    expect(() => encodeLeaf(1, 1 << 24)).toThrow();
  });
});

describe('SAH builder (procedural stress scene)', () => {
  const m = proceduralScene(7);
  const bvh = buildBvh(m.positions, m.indices);

  it('is structurally valid (every prim once, inside all ancestor boxes, leaves 1..4, depth ≤ 30)', () => {
    console.log('BVH_STATS procedural', JSON.stringify(bvh.stats));
    expect(checkBvhInvariants(bvh, m.positions, m.indices)).toEqual([]);
    expect(bvh.stats.triCount).toBe(m.indices.length / 3);
    expect(bvh.stats.maxLeafSize).toBeLessThanOrEqual(4);
    expect(bvh.stats.maxDepth).toBeLessThanOrEqual(30);
    expect(Number.isFinite(bvh.stats.sahCost)).toBe(true);
    expect(bvh.nodes.length).toBe(bvh.stats.nodeCount * NODE_FLOATS);
  });

  it('emits the MT and Woop triangle layouts per the contract', () => {
    const tu = u32View(bvh.tris), wu = u32View(bvh.trisW);
    const w4 = bvh.trisW.length / 4;
    for (let i = 0; i < bvh.primOrder.length; i += 97) {
      const p = bvh.primOrder[i];
      expect(tu[i * TRI_FLOATS + 3]).toBe(p);
      expect(wu[woopPrimIdIndex(w4, i)]).toBe(p);
      for (let k = 0; k < 3; k++) {
        const vi = m.indices[3 * p + k];
        expect(wu[i * TRI_FLOATS + 4 * k + 3]).toBe(bvh.weldedVid[vi]);
        for (let a = 0; a < 3; a++) expect(bvh.trisW[i * TRI_FLOATS + 4 * k + a]).toBe(m.positions[3 * vi + a]);
      }
      const v0 = m.indices[3 * p] * 3, v1 = m.indices[3 * p + 1] * 3;
      for (let a = 0; a < 3; a++) {
        expect(bvh.tris[i * TRI_FLOATS + a]).toBe(m.positions[v0 + a]);
        expect(bvh.tris[i * TRI_FLOATS + 4 + a]).toBe(Math.fround(m.positions[v1 + a] - m.positions[v0 + a]));
      }
    }
  });

  it('f64 BVH traversal equals f64 brute force (closest primId and any-hit), 4000 rays', () => {
    let hits = 0;
    for (const { o, d, tmax } of randomRays(m, 4000, 11)) {
      const a = bvhTrace64(bvh, m.positions, m.indices, o, d);
      const b = bruteClosest(m.positions, m.indices, o, d);
      expect(a.primId).toBe(b.primId);
      if (b.primId !== BVH_MISS) { hits++; expect(a.t).toBe(b.t); }
      expect(bvhTrace64(bvh, m.positions, m.indices, o, d, tmax, true).primId !== BVH_MISS).toBe(bruteAny(m.positions, m.indices, o, d, tmax));
    }
    expect(hits).toBeGreaterThan(1000);
  });
});

describe('SAH builder degenerate inputs', () => {
  const tri = (pts: number[][]): Mesh => ({ positions: new Float32Array(pts.flat()), indices: new Uint32Array(pts.map((_, i) => i)) });

  it('empty geometry yields a never-hit root', () => {
    const bvh = buildBvh(new Float32Array(0), new Uint32Array(0));
    expect(bvh.stats.triCount).toBe(0);
    expect(bvh.primOrder[0]).toBe(BVH_MISS);
    expect(bvhTrace64(bvh, new Float32Array(0), new Uint32Array(0), [0, 0, -1], [0, 0, 1]).primId).toBe(BVH_MISS);
  });

  it('single triangle: interior root, both children reference it', () => {
    const m = tri([[0, 0, 0], [1, 0, 0], [0, 1, 0]]);
    const bvh = buildBvh(m.positions, m.indices);
    expect(bvh.stats.nodeCount).toBe(1);
    expect(checkBvhInvariants(bvh, m.positions, m.indices)).toEqual([]);
    expect(bvhTrace64(bvh, m.positions, m.indices, [0.2, 0.2, -1], [0, 0, 1]).primId).toBe(0);
  });

  it('identical centroids (1000 copies) split by index halving', () => {
    const pts: number[][] = [];
    for (let i = 0; i < 1000; i++) pts.push([0, 0, 0], [1, 0, 0], [0, 1, 0]);
    const m = tri(pts);
    const bvh = buildBvh(m.positions, m.indices);
    expect(checkBvhInvariants(bvh, m.positions, m.indices)).toEqual([]);
    expect(bvh.stats.forcedSplits).toBeGreaterThan(0);
    expect(bvh.stats.maxDepth).toBeLessThanOrEqual(30);
  });

  it('pathological geometric spacing respects the depth cap (forced median splits)', () => {
    const pts: number[][] = [];
    for (let i = 0; i < 3000; i++) { const x = 2 ** (-(i % 120)) * (1 + i / 3000); pts.push([x, 0, 0], [x, 1e-3, 0], [x, 0, 1e-3]); }
    const m = tri(pts);
    const bvh = buildBvh(m.positions, m.indices);
    console.log('BVH_STATS pathological', JSON.stringify(bvh.stats));
    expect(checkBvhInvariants(bvh, m.positions, m.indices)).toEqual([]);
    expect(bvh.stats.maxDepth).toBeLessThanOrEqual(30);
    const tight = buildBvh(m.positions, m.indices, { maxDepth: 12 });
    expect(checkBvhInvariants(tight, m.positions, m.indices, 4, 12)).toEqual([]);
    expect(tight.stats.forcedSplits).toBeGreaterThan(0);
  });

  it('non-finite triangles are left out and counted', () => {
    const m = tri([[0, 0, 0], [1, 0, 0], [0, 1, 0], [NaN, 0, 0], [1, 1, 1], [2, 2, 2], [0, 0, 1], [1, 0, 1], [0, 1, 1]]);
    const bvh = buildBvh(m.positions, m.indices);
    expect(bvh.stats.triCount).toBe(2);
    expect(bvh.stats.skippedNonFinite).toBe(1);
    expect(Array.from(bvh.primOrder).sort()).toEqual([0, 2]);
    expect(checkBvhInvariants(bvh, m.positions, m.indices)).toEqual([]);
  });

  it('welded vertex ids unify unwelded (flat-shaded) duplicates, −0 == +0', () => {
    const p = new Float32Array([0, 0, 0, 1, 2, 3, -0, 0, 0, 1, 2, 3, 4, 5, 6]);
    expect(Array.from(weldVertices(p))).toEqual([0, 1, 0, 1, 4]);
  });

  it('icosphere watertight input: welded ids equal original ids for an indexed mesh', () => {
    const m = icosphere(3);
    const w = weldVertices(m.positions);
    for (let i = 0; i < w.length; i++) expect(w[i]).toBe(i);
  });
});

describe('Sponza build (if downloaded)', () => {
  it('builds 262k triangles in < 1 s with valid structure', async () => {
    const m = await loadGltfMesh();
    if (!m) { console.warn('Sponza not present (validation/blender/fetch_sponza.py); skipping'); return; }
    buildBvh(m.positions, m.indices); // warm-up JIT
    const t0 = performance.now();
    const bvh = buildBvh(m.positions, m.indices);
    const ms = performance.now() - t0;
    console.log('BVH_STATS sponza', JSON.stringify({ ...bvh.stats, wallMs: ms }));
    expect(m.indices.length / 3).toBeGreaterThan(260_000);
    expect(ms).toBeLessThan(1000);
    expect(checkBvhInvariants(bvh, m.positions, m.indices)).toEqual([]);
    for (const { o, d } of randomRays(m, 200, 3)) {
      expect(bvhTrace64(bvh, m.positions, m.indices, o, d).primId).toBe(bruteClosest(m.positions, m.indices, o, d).primId);
    }
  }, 120_000);
});
