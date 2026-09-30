// Fixtures for the quantized-geometry GPU tests (docs/decisions/data-formats.md §D P1): lattice-snapped meshes for
// T12, the quantized xi_contact package and quantized Sponza (both lanes), seam extraction for T12-Q.
import type { Mesh } from '../../tests/bvh/fixtures.ts';
import { loadGltf } from '../../src/core/scene/gltf-loader.ts';
import { choosePosLog2, snapToLattice } from '../../src/core/scene/quantize.ts';
import { readScenePackage } from '../../src/core/scene/scene-package.ts';
import type { SceneData } from '../../src/core/scene/types.ts';

export const SPONZA_DIR = 'validation/assets/downloaded/sponza';

/** Bytes of a repo file: fetch in the browser (Vite serves the repo root), fs in Node. undefined if missing. */
export async function readBytes(rel: string): Promise<Uint8Array | undefined> {
  if (typeof window !== 'undefined' && typeof fetch !== 'undefined') {
    const res = await fetch(`/${rel}`);
    if (!res.ok) return undefined;
    if ((res.headers.get('content-type') ?? '').includes('text/html')) return undefined; // SPA fallback: missing file
    return new Uint8Array(await res.arrayBuffer());
  }
  const fsName = 'node:fs/promises';
  const fs = (await import(/* @vite-ignore */ fsName)) as typeof import('node:fs/promises');
  try { return new Uint8Array(await fs.readFile(rel)); } catch { return undefined; }
}

/** A validation package (v2) by name, read with readScenePackage (lattice verified). */
export async function loadPackage(name: string): Promise<SceneData | undefined> {
  const dir = `validation/scenes/${name}`;
  const sj = await readBytes(`${dir}/scene.json`);
  if (!sj) return undefined;
  const json = JSON.parse(new TextDecoder().decode(sj)) as { textures?: { file: string }[]; env?: { file: string } | null };
  const files = new Map<string, Uint8Array>([['scene.json', sj]]);
  for (const f of ['geometry.bin', ...(json.textures ?? []).map((t) => t.file), ...(json.env ? [json.env.file] : [])]) {
    const b = await readBytes(`${dir}/${f}`);
    if (!b) return undefined;
    files.set(f, b);
  }
  return (await readScenePackage(files)).scene;
}

/** Khronos Sponza through the real loader (quantized, no image decode, no tangents); undefined if not downloaded. */
export async function loadSponzaQuantized(): Promise<SceneData | undefined> {
  const jb = await readBytes(`${SPONZA_DIR}/Sponza.gltf`);
  if (!jb) return undefined;
  const json = JSON.parse(new TextDecoder().decode(jb)) as { buffers?: { uri?: string }[]; images?: { uri?: string }[] };
  const resources: Record<string, Uint8Array> = {};
  for (const r of [...(json.buffers ?? []), ...(json.images ?? [])]) {
    if (!r.uri || r.uri.startsWith('data:') || resources[r.uri]) continue;
    const b = await readBytes(`${SPONZA_DIR}/${decodeURIComponent(r.uri)}`);
    if (!b) return undefined;
    resources[r.uri] = b;
  }
  return (await loadGltf({ kind: 'gltf', json: json as never, resources, name: 'Sponza.gltf' }, { tangents: false })).scene;
}

/** Snap a mesh to its global power-of-two lattice (quantizeScene steps 1–2: P21 lattice, drop collapsed triangles). */
export function latticeMesh(m: Mesh): Mesh & { posLog2: number; dropped: number } {
  let mn = Infinity, mx = -Infinity, ext = 0, maxAbs = 0;
  for (let k = 0; k < 3; k++) {
    mn = Infinity; mx = -Infinity;
    for (let i = k; i < m.positions.length; i += 3) { mn = Math.min(mn, m.positions[i]); mx = Math.max(mx, m.positions[i]); maxAbs = Math.max(maxAbs, Math.abs(m.positions[i])); }
    ext = Math.max(ext, mx - mn);
  }
  const k = choosePosLog2(ext, maxAbs);
  if (k === null) throw new Error('mesh above the lattice precision floor');
  const s = 2 ** k;
  const positions = m.positions.map((x) => snapToLattice(x, s));
  const keep: number[] = [];
  let dropped = 0;
  for (let t = 0; t < m.indices.length; t += 3) {
    const [a, b, c] = [m.indices[t] * 3, m.indices[t + 1] * 3, m.indices[t + 2] * 3];
    const e1 = [0, 1, 2].map((j) => positions[b + j] - positions[a + j]), e2 = [0, 1, 2].map((j) => positions[c + j] - positions[a + j]);
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    if (n[0] === 0 && n[1] === 0 && n[2] === 0) { dropped++; continue; }
    keep.push(m.indices[t], m.indices[t + 1], m.indices[t + 2]);
  }
  return { positions, indices: Uint32Array.from(keep), posLog2: k, dropped };
}

export interface Seams {
  /** [vertex a, vertex b, triangles...] for edges shared (by identical position bits) by ≥ 2 triangles. */
  edges: { a: number; b: number; tris: number[] }[];
  /** vertex (first index with those position bits) and its triangle fan (≥ 3 triangles). */
  verts: { v: number; tris: number[] }[];
}

/** Edges / vertices shared by identical position bits. `cross`: keep only those touching ≥ 2 materials (seams between
 *  different meshes; Sponza) — otherwise every shared edge (xi_contact: contacts built from identical coordinates). */
export function findSeams(positions: Float32Array, indices: Uint32Array, triMaterial: Uint32Array, cross: boolean): Seams {
  const bits = new Uint32Array(positions.buffer, positions.byteOffset, positions.length);
  const canon = new Map<string, number>();
  const cid = new Int32Array(positions.length / 3);
  for (let v = 0; v < cid.length; v++) {
    const k = `${bits[3 * v]},${bits[3 * v + 1]},${bits[3 * v + 2]}`;
    let c = canon.get(k);
    if (c === undefined) { c = v; canon.set(k, v); }
    cid[v] = c;
  }
  const edgeMap = new Map<string, { a: number; b: number; tris: number[] }>();
  const vertMap = new Map<number, number[]>();
  for (let t = 0; t < indices.length / 3; t++) {
    for (let e = 0; e < 3; e++) {
      const a = cid[indices[3 * t + e]], b = cid[indices[3 * t + ((e + 1) % 3)]];
      const k = a < b ? `${a},${b}` : `${b},${a}`;
      let r = edgeMap.get(k);
      if (!r) { r = { a: Math.min(a, b), b: Math.max(a, b), tris: [] }; edgeMap.set(k, r); }
      r.tris.push(t);
      const vt = vertMap.get(a) ?? [];
      vt.push(t); vertMap.set(a, vt);
    }
  }
  const multi = (tris: number[]) => !cross || new Set(tris.map((t) => triMaterial[t])).size > 1;
  return {
    edges: [...edgeMap.values()].filter((r) => r.tris.length >= 2 && multi(r.tris)),
    verts: [...vertMap.entries()].filter(([, tris]) => tris.length >= 3 && multi(tris)).map(([v, tris]) => ({ v, tris })),
  };
}
