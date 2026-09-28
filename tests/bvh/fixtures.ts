// Shared BVH test fixtures (CPU lane and the T12 GPU test): seeded RNG, procedural scenes, icosphere and a
// minimal glTF position/index reader for Khronos Sponza (runs in Node and in the browser).

export interface Mesh { positions: Float32Array; indices: Uint32Array }

/** mulberry32: small seeded PRNG, [0, 1). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomDir(r: () => number): [number, number, number] {
  const z = 2 * r() - 1, phi = 2 * Math.PI * r(), s = Math.sqrt(Math.max(0, 1 - z * z));
  return [s * Math.cos(phi), s * Math.sin(phi), z];
}

class MeshBuilder {
  pos: number[] = [];
  idx: number[] = [];
  vert(x: number, y: number, z: number): number { this.pos.push(x, y, z); return this.pos.length / 3 - 1; }
  tri(a: number, b: number, c: number) { this.idx.push(a, b, c); }
  triP(a: number[], b: number[], c: number[]) { this.tri(this.vert(a[0], a[1], a[2]), this.vert(b[0], b[1], b[2]), this.vert(c[0], c[1], c[2])); }
  /** Indexed n×n grid of quads spanning origin + s·u + t·v, s,t ∈ [0,1]. */
  grid(o: number[], u: number[], v: number[], n: number) {
    const base = this.pos.length / 3;
    for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) {
      const s = i / n, t = j / n;
      this.vert(o[0] + s * u[0] + t * v[0], o[1] + s * u[1] + t * v[1], o[2] + s * u[2] + t * v[2]);
    }
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const a = base + j * (n + 1) + i, b = a + 1, c = a + n + 1, d = c + 1;
      this.tri(a, b, d); this.tri(a, d, c);
    }
  }
  mesh(offset: number[] = [0, 0, 0]): Mesh {
    const p = new Float32Array(this.pos);
    for (let i = 0; i < p.length; i++) p[i] += offset[i % 3];
    return { positions: p, indices: new Uint32Array(this.idx) };
  }
}

/** Indexed icosphere (20·4^level triangles), outward CCW winding. */
export function icosphere(level: number, radius = 1, center: number[] = [0, 0, 0]): Mesh {
  const t = (1 + Math.sqrt(5)) / 2;
  let verts: number[][] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]];
  let faces = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
  const norm = (v: number[]) => { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; };
  verts = verts.map(norm);
  for (let l = 0; l < level; l++) {
    const cache = new Map<string, number>();
    const mid = (a: number, b: number) => {
      const k = a < b ? `${a},${b}` : `${b},${a}`;
      let m = cache.get(k);
      if (m === undefined) { m = verts.length; verts.push(norm([(verts[a][0] + verts[b][0]) / 2, (verts[a][1] + verts[b][1]) / 2, (verts[a][2] + verts[b][2]) / 2])); cache.set(k, m); }
      return m;
    };
    const nf: number[][] = [];
    for (const [a, b, c] of faces) { const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a); nf.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]); }
    faces = nf;
  }
  const positions = new Float32Array(verts.length * 3);
  verts.forEach((v, i) => { for (let k = 0; k < 3; k++) positions[3 * i + k] = center[k] + radius * v[k]; });
  return { positions, indices: new Uint32Array(faces.flat()) };
}

/**
 * ~20k-triangle procedural stress scene, off-origin: a tessellated floor, a thin double wall (1e-4 gap), an
 * icosphere, random soup incl. slivers, coplanar overlapping pairs, exact duplicates and crossing triangles.
 */
export function proceduralScene(seed = 1): Mesh {
  const r = rng(seed);
  const m = new MeshBuilder();
  m.grid([-10, 0, -10], [20, 0, 0], [0, 0, 20], 64);                       // floor, 8192 tris
  m.grid([-6, 0, 3], [12, 0, 0], [0, 5, 0], 20);                           // thin wall, front face
  m.grid([-6, 0, 3.0001], [0, 5, 0], [12, 0, 0], 20);                      // back face, 1e-4 behind, 1600 tris total
  const sph = icosphere(3, 1.5, [3, 2, -3]);                               // 1280 tris
  const base = m.pos.length / 3;
  m.pos.push(...sph.positions); for (const i of sph.indices) m.idx.push(base + i);
  const rp = (s: number) => [(r() * 2 - 1) * s, r() * 6, (r() * 2 - 1) * s];
  for (let i = 0; i < 5000; i++) {                                          // soup incl. slivers
    const c = rp(9), sz = 0.02 + r() * 0.8;
    const a = [c[0] + (r() - 0.5) * sz, c[1] + (r() - 0.5) * sz, c[2] + (r() - 0.5) * sz];
    const b = [c[0] + (r() - 0.5) * sz, c[1] + (r() - 0.5) * sz, c[2] + (r() - 0.5) * sz];
    let cc = [c[0] + (r() - 0.5) * sz, c[1] + (r() - 0.5) * sz, c[2] + (r() - 0.5) * sz];
    if (i % 5 === 0) { const f = 1e-4 * r(); cc = [a[0] + (b[0] - a[0]) * 0.5 + f, a[1] + (b[1] - a[1]) * 0.5 - f, a[2] + (b[2] - a[2]) * 0.5 + f]; } // sliver
    m.triP(a, b, cc);
  }
  for (let i = 0; i < 1000; i++) {                                          // coplanar overlapping pairs (2000 tris)
    const c = rp(8), sz = 0.1 + r() * 0.5, y = c[1];
    const a = [c[0], y, c[2]], b = [c[0] + sz, y, c[2]], cc = [c[0], y, c[2] + sz];
    m.triP(a, b, cc);
    const o = sz * 0.3;
    m.triP([a[0] + o, y, a[2] + o], [b[0] + o, y, b[2] + o], [cc[0] + o, y, cc[2] + o]);
  }
  for (let i = 0; i < 500; i++) {                                           // exact duplicates (1000 tris)
    const c = rp(8), sz = 0.2 + r() * 0.3;
    const a = [c[0], c[1], c[2]], b = [c[0] + sz, c[1] + sz * 0.3, c[2]], cc = [c[0], c[1] + sz, c[2] + sz * 0.5];
    m.triP(a, b, cc); m.triP(a, b, cc);
  }
  for (let i = 0; i < 500; i++) {                                           // crossing pairs (1000 tris)
    const c = rp(8), sz = 0.3 + r() * 0.5;
    m.triP([c[0] - sz, c[1], c[2] - sz], [c[0] + sz, c[1], c[2] - sz], [c[0], c[1], c[2] + sz]);
    m.triP([c[0], c[1] - sz, c[2] - sz], [c[0], c[1] + sz, c[2] - sz], [c[0], c[1], c[2] + sz * 1.2]);
  }
  return m.mesh([12.3, 4.5, -7.8]);
}

export function meshBounds(m: Mesh): { min: number[]; max: number[] } {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < m.positions.length; i++) { const k = i % 3; min[k] = Math.min(min[k], m.positions[i]); max[k] = Math.max(max[k], m.positions[i]); }
  return { min, max };
}

// ---------------------------------------------------------------------------------------------------------------
// Minimal glTF reader: POSITION + indices of triangle primitives, node TRS/matrix hierarchy applied (f64 math,
// rounded once to f32). Enough for Khronos Sponza; the real loader lives in src/core/scene.

export const SPONZA_PATH = 'validation/assets/downloaded/sponza/Sponza.gltf';

async function readAsset(rel: string): Promise<ArrayBuffer | undefined> {
  if (typeof window !== 'undefined' && typeof fetch !== 'undefined') {
    const res = await fetch(`/${rel}`);
    if (!res.ok) return undefined;
    const ct = res.headers.get('content-type') ?? '';
    if (ct.includes('text/html')) return undefined; // SPA fallback: the file does not exist
    return res.arrayBuffer();
  }
  const fsName = 'node:fs/promises';
  const fs = (await import(/* @vite-ignore */ fsName)) as typeof import('node:fs/promises');
  try { const b = await fs.readFile(rel); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer; } catch { return undefined; }
}

type Mat = number[]; // column-major 4x4
const matMul = (a: Mat, b: Mat): Mat => {
  const o = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return o;
};
function trs(n: { matrix?: number[]; translation?: number[]; rotation?: number[]; scale?: number[] }): Mat {
  if (n.matrix) return n.matrix.slice();
  const [tx, ty, tz] = n.translation ?? [0, 0, 0];
  const [x, y, z, w] = n.rotation ?? [0, 0, 0, 1];
  const [sx, sy, sz] = n.scale ?? [1, 1, 1];
  return [
    (1 - 2 * (y * y + z * z)) * sx, (2 * (x * y + z * w)) * sx, (2 * (x * z - y * w)) * sx, 0,
    (2 * (x * y - z * w)) * sy, (1 - 2 * (x * x + z * z)) * sy, (2 * (y * z + x * w)) * sy, 0,
    (2 * (x * z + y * w)) * sz, (2 * (y * z - x * w)) * sz, (1 - 2 * (x * x + y * y)) * sz, 0,
    tx, ty, tz, 1,
  ];
}

export async function loadGltfMesh(rel = SPONZA_PATH): Promise<Mesh | undefined> {
  const jsonBuf = await readAsset(rel);
  if (!jsonBuf) return undefined;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const g = JSON.parse(new TextDecoder().decode(jsonBuf)) as any;
  const dir = rel.split('/').slice(0, -1).join('/');
  const buffers: ArrayBuffer[] = [];
  for (const b of g.buffers) { const d = await readAsset(`${dir}/${b.uri}`); if (!d) return undefined; buffers.push(d); }
  const accessor = (i: number): { data: Float32Array | Uint32Array | Uint16Array | Uint8Array; comps: number; count: number } => {
    const a = g.accessors[i], bv = g.bufferViews[a.bufferView];
    const comps = ({ SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 } as Record<string, number>)[a.type];
    const Ctor = a.componentType === 5126 ? Float32Array : a.componentType === 5125 ? Uint32Array : a.componentType === 5123 ? Uint16Array : Uint8Array;
    const stride = bv.byteStride ?? comps * Ctor.BYTES_PER_ELEMENT;
    const off = (bv.byteOffset ?? 0) + (a.byteOffset ?? 0);
    const out = new Ctor(a.count * comps);
    const dv = new DataView(buffers[bv.buffer]);
    for (let e = 0; e < a.count; e++) for (let c = 0; c < comps; c++) {
      const p = off + e * stride + c * Ctor.BYTES_PER_ELEMENT;
      out[e * comps + c] = Ctor === Float32Array ? dv.getFloat32(p, true) : Ctor === Uint32Array ? dv.getUint32(p, true) : Ctor === Uint16Array ? dv.getUint16(p, true) : dv.getUint8(p);
    }
    return { data: out, comps, count: a.count };
  };
  const pos: number[] = [], idx: number[] = [];
  const visit = (ni: number, parent: Mat) => {
    const n = g.nodes[ni];
    const m = matMul(parent, trs(n));
    if (n.mesh !== undefined) {
      for (const prim of g.meshes[n.mesh].primitives) {
        if ((prim.mode ?? 4) !== 4) continue;
        const P = accessor(prim.attributes.POSITION);
        const base = pos.length / 3;
        for (let v = 0; v < P.count; v++) {
          const x = P.data[3 * v], y = P.data[3 * v + 1], z = P.data[3 * v + 2];
          pos.push(m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13], m[2] * x + m[6] * y + m[10] * z + m[14]);
        }
        if (prim.indices !== undefined) { const I = accessor(prim.indices); for (let k = 0; k < I.data.length; k++) idx.push(base + I.data[k]); }
        else for (let k = 0; k < P.count; k++) idx.push(base + k);
      }
    }
    for (const c of n.children ?? []) visit(c, m);
  };
  const I4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  for (const ni of g.scenes[g.scene ?? 0].nodes) visit(ni, I4);
  // Drop degenerate (zero-area in f32) triangles, as flatten does (plan §1.3).
  const positions = new Float32Array(pos);
  const keep: number[] = [];
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
    const e1 = [positions[b] - positions[a], positions[b + 1] - positions[a + 1], positions[b + 2] - positions[a + 2]];
    const e2 = [positions[c] - positions[a], positions[c + 1] - positions[a + 1], positions[c + 2] - positions[a + 2]];
    const cx = e1[1] * e2[2] - e1[2] * e2[1], cy = e1[2] * e2[0] - e1[0] * e2[2], cz = e1[0] * e2[1] - e1[1] * e2[0];
    if (cx !== 0 || cy !== 0 || cz !== 0) keep.push(idx[t], idx[t + 1], idx[t + 2]);
  }
  return { positions, indices: new Uint32Array(keep) };
}
