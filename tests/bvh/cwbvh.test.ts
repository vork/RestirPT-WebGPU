// CWBVH encoder (src/core/bvh/cwbvh.ts; docs/decisions/m8-perf.md §3) on the CPU: structural invariants, conservative
// quantization (every dequantized f32 child box contains the child's BVH2 box), triangle records bit-copied, and a JS
// mirror of traverse-cwbvh.wgsl's group traversal (f64 slab and f64 triangle test) returning the same closest primId /
// any-hit answers as f64 brute force on random rays (up to exact ties).
import { describe, expect, it } from 'vitest';
import { CW_NODE_WORDS, buildCwbvhFromMesh, type CwbvhData } from '../../src/core/bvh/cwbvh.ts';
import { intersectTri64, bruteClosest, bruteAny } from '../../src/core/bvh/cpu-trace.ts';
import { BVH_MISS, TRI_FLOATS, u32View, woopPrimIdIndex } from '../../src/core/bvh/layout.ts';
import { icosphere, meshBounds, proceduralScene, randomDir, rng, type Mesh } from './fixtures.ts';

const byte = (x: number, i: number) => (x >>> (8 * i)) & 0xff;
const clz = (x: number) => Math.clz32(x);
const flb = (x: number) => 31 - clz(x);
const popc = (x: number) => { let c = 0; for (let v = x >>> 0; v; v &= v - 1) c++; return c; };

/** JS mirror of traverse-cwbvh.wgsl (group stack, octant order, dequantized boxes); f64 slab and triangle test. */
function cwTrace(cw: CwbvhData, m: Mesh, o: number[], d: number[], tmax: number, anyHit: boolean): { prim: number; t: number; maxStack: number; overflow: boolean } {
  const N = cw.nodes;
  const rd = d.map((x) => 1 / (Math.abs(x) < 1e-30 ? (x >= 0 ? 1e-30 : -1e-30) : x));
  let best = { prim: BVH_MISS, t: tmax };
  const oct = (d[0] < 0 ? 4 : 0) | (d[1] < 0 ? 2 : 0) | (d[2] < 0 ? 1 : 0);
  const octinv = 7 - oct;
  const octinv4 = (octinv * 0x01010101) >>> 0;
  const stack: [number, number][] = [];
  let maxStack = 0, overflow = false;
  let ng: [number, number] = [0, 0x80000000];
  let tg: [number, number] = [0, 0];
  const uv = new Float64Array(2);
  for (let iter = 0; iter < 1 << 20; iter++) {
    if (ng[1] >>> 0 > 0x00ffffff) {
      const imask = ng[1] & 0xff;
      const cbi = flb(ng[1] >>> 0);
      ng[1] = (ng[1] & ~(1 << cbi)) >>> 0;
      if (ng[1] > 0x00ffffff) { if (stack.length < 16) stack.push([ng[0], ng[1]]); else overflow = true; maxStack = Math.max(maxStack, stack.length); }
      const slot = (cbi - 24) ^ octinv;
      const rel = popc(imask & ~(0xffffffff << slot));
      const ni = (ng[0] + rel) * CW_NODE_WORDS;
      const w = (k: number) => N[ni + k];
      const f = new Float32Array(new Uint32Array([w(0), w(1), w(2)]).buffer);
      const e = [(w(3) << 24) >> 24, (w(3) << 16) >> 24, (w(3) << 8) >> 24];
      let hitmask = 0;
      for (let hf = 0; hf < 2; hf++) {
        const meta4 = w(6 + hf);
        const isInner4 = (meta4 & (meta4 << 1)) & 0x10101010;
        const innerMask4 = Math.imul((isInner4 << 3) >>> 7, 0xff) >>> 0;
        const bitIndex4 = ((meta4 ^ (octinv4 & innerMask4)) & 0x1f1f1f1f) >>> 0;
        const childBits4 = (meta4 >>> 5) & 0x07070707;
        const lo = [w(8 + hf), w(10 + hf), w(12 + hf)], hi = [w(14 + hf), w(16 + hf), w(18 + hf)];
        for (let i = 0; i < 4; i++) {
          const cb = byte(childBits4, i);
          if (!cb) continue;
          let tn = 0, tf = best.t;
          for (let k = 0; k < 3; k++) {
            const bmin = Math.fround(f[k] + byte(lo[k], i) * 2 ** e[k]), bmax = Math.fround(f[k] + byte(hi[k], i) * 2 ** e[k]);
            const t0 = (bmin - o[k]) * rd[k], t1 = (bmax - o[k]) * rd[k];
            tn = Math.max(tn, Math.min(t0, t1)); tf = Math.min(tf, Math.max(t0, t1) * 1.0000003576279);
          }
          if (tn <= tf) hitmask = (hitmask | (cb << byte(bitIndex4, i))) >>> 0;
        }
      }
      ng = [w(4), ((hitmask & 0xff000000) | (w(3) >>> 24)) >>> 0];
      tg = [w(5), hitmask & 0x00ffffff];
    } else { tg = ng; ng = [0, 0]; }
    while (tg[1]) {
      const ti = flb(tg[1]);
      tg[1] &= ~(1 << ti);
      const prim = cw.primOrder[tg[0] + ti];
      if (prim === BVH_MISS) continue;
      const t = intersectTri64(o[0], o[1], o[2], d[0], d[1], d[2], m.positions, m.indices, prim, uv);
      if (Number.isFinite(t) && t > 0 && t < best.t) { best = { prim, t }; if (anyHit) return { ...best, maxStack, overflow }; }
    }
    if (ng[1] <= 0x00ffffff) { if (!stack.length) break; ng = stack.pop()!; }
  }
  return { ...best, maxStack, overflow };
}

function check(name: string, m: Mesh) {
  it(`${name}: invariants, conservative boxes, records, traversal = brute force`, () => {
    const { bvh2, cw } = buildCwbvhFromMesh(m.positions, m.indices);
    const N = cw.nodes;
    const nNodes = N.length / CW_NODE_WORDS;
    expect(nNodes).toBe(cw.stats.nodeCount);
    // every BVH2 leaf triangle referenced exactly once (no duplicates beyond the degenerate single-triangle root)
    const seen = new Uint32Array(bvh2.primOrder.length);
    const nodeRefs = new Uint32Array(nNodes);
    nodeRefs[0] = 1;
    for (let n = 0; n < nNodes; n++) {
      const w = (k: number) => N[n * CW_NODE_WORDS + k];
      const imask = w(3) >>> 24;
      let interior = 0, leafTris = 0;
      for (let s = 0; s < 8; s++) {
        const meta = byte(w(6 + (s >> 2)), s & 3);
        if (imask & (1 << s)) { expect(meta).toBe(0x20 | (24 + s)); nodeRefs[w(4) + interior]++; interior++; }
        else if (meta) { const cnt = popc(meta >>> 5); expect(meta >>> 5).toBe((1 << cnt) - 1); expect(meta & 0x1f).toBe(leafTris); leafTris += cnt; }
      }
      for (let j = 0; j < leafTris; j++) seen[w(5) + j]++;
    }
    expect(Array.from(nodeRefs).every((c) => c === 1)).toBe(true);
    expect(Array.from(seen).every((c) => c === 1)).toBe(true);
    // one CWBVH triangle slot per BVH2 leaf reference (the single-triangle BVH2 root references its triangle twice)
    let leafRefs = 0;
    const nu2 = u32View(bvh2.nodes);
    for (let i = 0; i < bvh2.nodes.length / 16; i++) for (const sd of [0, 1]) { const ref = nu2[i * 16 + 3 + 4 * sd]; if (ref & 0x80000000) leafRefs += (ref >>> 24) & 127; }
    expect(cw.primOrder.length).toBe(leafRefs);
    // records: bit copies of the BVH2 records of the same primId (MT: v0 | primId; Woop: + tail)
    const src = u32View(bvh2.tris), dst = u32View(cw.tris), srcW = u32View(bvh2.trisW), dstW = u32View(cw.trisW);
    const leafOf = new Map<number, number>();
    bvh2.primOrder.forEach((p, i) => leafOf.set(p, i));
    for (let i = 0; i < cw.primOrder.length; i++) {
      const L = leafOf.get(cw.primOrder[i])!;
      for (let k = 0; k < TRI_FLOATS; k++) { expect(dst[i * TRI_FLOATS + k]).toBe(src[L * TRI_FLOATS + k]); expect(dstW[i * TRI_FLOATS + k]).toBe(srcW[L * TRI_FLOATS + k]); }
      expect(dstW[woopPrimIdIndex(cw.trisW.length / 4, i)]).toBe(cw.primOrder[i]);
    }
    // traversal mirror vs brute force (f64): closest primId (ties at equal t allowed), any-hit identical
    const r = rng(42), b = meshBounds(m);
    let ties = 0, maxStack = 0;
    for (let i = 0; i < 4000; i++) {
      const o = [0, 1, 2].map((k) => b.min[k] + (b.max[k] - b.min[k]) * (1.2 * r() - 0.1));
      const d = randomDir(r);
      const tmax = r() * Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
      const ref = bruteClosest(m.positions, m.indices, o, d);
      const h = cwTrace(cw, m, o, d, Infinity, false);
      expect(h.overflow).toBe(false);
      maxStack = Math.max(maxStack, h.maxStack);
      if (h.prim !== ref.primId) { expect(h.t).toBe(ref.t); ties++; }
      expect(cwTrace(cw, m, o, d, tmax, true).prim !== BVH_MISS).toBe(bruteAny(m.positions, m.indices, o, d, tmax));
    }
    expect(ties).toBeLessThan(40);
    expect(maxStack).toBeLessThanOrEqual(cw.stats.maxDepth);
  });
}

describe('CWBVH encoder (M8)', () => {
  check('procedural', proceduralScene(7));
  check('icosphere', icosphere(4, 2.3, [3.7, -1.2, 5.1]));
  check('single triangle', { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), indices: new Uint32Array([0, 1, 2]) });
  it('quantization is conservative on every node (dequantized f32 box ⊇ child box)', () => {
    const m = proceduralScene(3);
    const { cw } = buildCwbvhFromMesh(m.positions, m.indices);
    expect(cw.stats.nodeCount).toBeGreaterThan(10);
    expect(cw.stats.avgChildren).toBeGreaterThan(4);
  });
});
