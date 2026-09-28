// Flatten the default glTF scene into one world-space triangle soup (plan §1.3; scene-io §3.2, §7.1).
//
// - World matrix per node; positions transformed, normals by the inverse-transpose and renormalized.
// - det(M) < 0 flips the winding (and sets TRI_FLIPPED) so CCW stays the front face in world space.
// - Missing NORMAL → flat normals (the primitive is unwelded: 3 unique vertices per triangle).
// - MikkTSpace tangents (glTF convention: w negated, like gltf-transform `tangents()`) when the material has a
//   normal map and the primitive has TEXCOORD_0. MikkTSpace needs unwelded input, so those primitives are
//   unwelded too. Tangents are generated from WORLD-space positions/normals, so mirrored instances get the
//   correct handedness without a det fix-up. Everything else gets zero tangents.
// - Zero-area and non-finite triangles are dropped (logged). primIds are DENSE after the drop: primId is the
//   triangle's index in the final `indices` array, i.e. dropped triangles never get an id and later ones shift
//   down. primIds are stable for a given file + loader version (plan §1.3 "stable primId").
// - Scenes with more than 2^24 triangles are rejected (BVH leaf encoding, firstTri < 2^24).
import type { Accessor, Document, Material, Primitive } from '@gltf-transform/core';
import { TRI_ALPHA_MASK, TRI_EMISSIVE, TRI_FLIPPED, type Bounds, type SceneGeometry } from './types.ts';

export const MAX_TRIANGLES = 1 << 24;

// glTF primitive modes
const MODE_TRIANGLES = 4;
const MODE_TRIANGLE_STRIP = 5;
const MODE_TRIANGLE_FAN = 6;

/** Per-material facts flatten needs (computed by the material mapper in gltf-loader.ts). */
export interface FlattenMaterialInfo {
  /** Index into SceneData.materials. */
  index: number;
  /** alphaMode MASK (BLEND is converted to MASK 0.5 before flattening). */
  masked: boolean;
  /** baseColorFactor.a × baseColorTexture.a is known to be constant 1 (COLOR_0 alpha is checked per primitive). */
  alphaConstOne: boolean;
  /** emissiveFactor · emissiveStrength > 0. */
  emissive: boolean;
  /** Has a normal map, so MikkTSpace tangents are needed. */
  wantsTangents: boolean;
}

export type TangentGenerator = (position: Float32Array, normal: Float32Array, texcoord: Float32Array) => Float32Array;

export interface FlattenOptions {
  /** Maps a glTF material (null = primitive without material) to its SceneData material. */
  materialInfo: (material: Material | null, hasColor0: boolean) => FlattenMaterialInfo;
  /** MikkTSpace generator (mikktspace package). Omit to emit zero tangents everywhere. */
  generateTangents?: TangentGenerator;
  warnings: string[];
  maxTriangles?: number;
}

export interface FlattenStats {
  triangles: number;
  vertices: number;
  droppedDegenerate: number;
  droppedNonFinite: number;
  flippedTriangles: number;
  unweldedPrimitives: number;
  tangentPrimitives: number;
  repairedNormals: number;
}

export interface FlattenResult { geometry: SceneGeometry; bounds: Bounds; stats: FlattenStats }

interface PrimInstance {
  prim: Primitive;
  world: number[];       // column-major 4x4
  tris: Uint32Array;     // triangle-list indices into the primitive's vertices (authored winding)
  vertexCount: number;
  unweld: boolean;
  flat: boolean;
  tangents: boolean;
  hasColor0: boolean;
  info: FlattenMaterialInfo;
}

export function flattenScene(doc: Document, opts: FlattenOptions): FlattenResult {
  const warn = (m: string) => opts.warnings.push(m);
  const root = doc.getRoot();
  const scene = root.getDefaultScene() ?? root.listScenes()[0];
  const maxTris = opts.maxTriangles ?? MAX_TRIANGLES;
  const instances: PrimInstance[] = [];
  let skippedModes = 0;

  // ---- pass 1: collect primitive instances ----
  scene?.traverse((node) => {
    const mesh = node.getMesh();
    if (!mesh) return;
    const world = node.getWorldMatrix() as unknown as number[];
    for (const prim of mesh.listPrimitives()) {
      const pos = prim.getAttribute('POSITION');
      if (!pos) { warn(`mesh '${mesh.getName()}': primitive without POSITION skipped`); continue; }
      const tris = triangleList(prim, pos.getCount());
      if (!tris) { skippedModes++; continue; }
      const hasColor0 = prim.getAttribute('COLOR_0') !== null;
      const info = opts.materialInfo(prim.getMaterial(), hasColor0);
      const flat = prim.getAttribute('NORMAL') === null;
      const tangents = !!opts.generateTangents && info.wantsTangents && prim.getAttribute('TEXCOORD_0') !== null;
      instances.push({ prim, world, tris, vertexCount: pos.getCount(), unweld: flat || tangents, flat, tangents, hasColor0, info });
    }
  });
  if (skippedModes) warn(`${skippedModes} non-triangle primitive(s) (points/lines) skipped`);

  let totalVerts = 0, totalTris = 0;
  for (const inst of instances) {
    const nt = inst.tris.length / 3;
    totalTris += nt;
    totalVerts += inst.unweld ? nt * 3 : inst.vertexCount;
  }
  if (totalTris > maxTris) throw new Error(`scene has ${totalTris} triangles; the limit is ${maxTris} (2^24, BVH leaf encoding)`);
  if (totalVerts >= 2 ** 32) throw new Error(`scene has ${totalVerts} vertices; u32 indices overflow`);

  const anyColor = instances.some((i) => i.hasColor0);
  const positions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(totalVerts * 3);
  const tangents = new Float32Array(totalVerts * 4);
  const uv0 = new Float32Array(totalVerts * 2);
  const color0 = anyColor ? new Float32Array(totalVerts * 4).fill(1) : undefined;
  let indices = new Uint32Array(totalTris * 3);
  let triMaterial = new Uint32Array(totalTris);
  let triFlags = new Uint32Array(totalTris);

  const stats: FlattenStats = {
    triangles: 0, vertices: totalVerts, droppedDegenerate: 0, droppedNonFinite: 0, flippedTriangles: 0,
    unweldedPrimitives: 0, tangentPrimitives: 0, repairedNormals: 0,
  };
  const bmin = [Infinity, Infinity, Infinity], bmax = [-Infinity, -Infinity, -Infinity];
  let vBase = 0, tOut = 0;

  // ---- pass 2: transform and emit ----
  for (const inst of instances) {
    const { prim, world: m } = inst;
    const P = readAttr(prim.getAttribute('POSITION'))!;
    const N = inst.flat ? null : readAttr(prim.getAttribute('NORMAL'));
    const T = readAttr(prim.getAttribute('TEXCOORD_0'));
    const C = readAttr(prim.getAttribute('COLOR_0'));
    // Upper 3x3 (column-major m[c*4+r]), its determinant and inverse-transpose (cofactor / det).
    const a00 = m[0], a10 = m[1], a20 = m[2], a01 = m[4], a11 = m[5], a21 = m[6], a02 = m[8], a12 = m[9], a22 = m[10];
    const c00 = a11 * a22 - a12 * a21, c01 = a12 * a20 - a10 * a22, c02 = a10 * a21 - a11 * a20;
    const c10 = a02 * a21 - a01 * a22, c11 = a00 * a22 - a02 * a20, c12 = a01 * a20 - a00 * a21;
    const c20 = a01 * a12 - a02 * a11, c21 = a02 * a10 - a00 * a12, c22 = a00 * a11 - a01 * a10;
    const det = a00 * c00 + a01 * c01 + a02 * c02;
    const flipped = det < 0;
    if (!(Math.abs(det) > 0) || !Number.isFinite(det)) warn(`mesh instance with singular world matrix (det=${det})`);
    const invDet = det !== 0 ? 1 / det : 0;
    // inverse-transpose rows: (M^-T)[r][c] = cof[r][c] / det
    const n00 = c00 * invDet, n01 = c01 * invDet, n02 = c02 * invDet;
    const n10 = c10 * invDet, n11 = c11 * invDet, n12 = c12 * invDet;
    const n20 = c20 * invDet, n21 = c21 * invDet, n22 = c22 * invDet;

    const writeVertex = (dst: number, src: number) => {
      const x = P.data[src * P.size], y = P.data[src * P.size + 1], z = P.data[src * P.size + 2];
      positions[dst * 3] = a00 * x + a01 * y + a02 * z + m[12];
      positions[dst * 3 + 1] = a10 * x + a11 * y + a12 * z + m[13];
      positions[dst * 3 + 2] = a20 * x + a21 * y + a22 * z + m[14];
      if (N) {
        const nx = N.data[src * N.size], ny = N.data[src * N.size + 1], nz = N.data[src * N.size + 2];
        const wx = n00 * nx + n01 * ny + n02 * nz, wy = n10 * nx + n11 * ny + n12 * nz, wz = n20 * nx + n21 * ny + n22 * nz;
        const l = Math.hypot(wx, wy, wz);
        const s = l > 0 && Number.isFinite(l) ? 1 / l : 0; // 0 marks the normal for repair below
        normals[dst * 3] = wx * s; normals[dst * 3 + 1] = wy * s; normals[dst * 3 + 2] = wz * s;
      }
      if (T) { uv0[dst * 2] = T.data[src * T.size]; uv0[dst * 2 + 1] = T.data[src * T.size + 1]; }
      if (C && color0) {
        color0[dst * 4] = C.data[src * C.size]; color0[dst * 4 + 1] = C.data[src * C.size + 1];
        color0[dst * 4 + 2] = C.data[src * C.size + 2]; color0[dst * 4 + 3] = C.size === 4 ? C.data[src * C.size + 3] : 1;
      }
    };

    const nt = inst.tris.length / 3;
    const tris = inst.tris;
    // corner order: swap corners 1 and 2 when the transform mirrors
    const k1 = flipped ? 2 : 1, k2 = flipped ? 1 : 2;
    if (inst.unweld) {
      stats.unweldedPrimitives++;
      for (let t = 0; t < nt; t++) {
        writeVertex(vBase + 3 * t, tris[3 * t]);
        writeVertex(vBase + 3 * t + 1, tris[3 * t + k1]);
        writeVertex(vBase + 3 * t + 2, tris[3 * t + k2]);
      }
    } else {
      for (let v = 0; v < inst.vertexCount; v++) writeVertex(vBase + v, v);
    }
    const vCount = inst.unweld ? nt * 3 : inst.vertexCount;
    const corner = (t: number, c: 0 | 1 | 2) => (inst.unweld ? vBase + 3 * t + c : vBase + tris[3 * t + (c === 0 ? 0 : c === 1 ? k1 : k2)]);

    // Flat normals from the final (world, post-flip) winding.
    if (inst.flat) {
      for (let t = 0; t < nt; t++) {
        const f = faceNormal(positions, vBase + 3 * t, vBase + 3 * t + 1, vBase + 3 * t + 2);
        for (let c = 0; c < 3; c++) normals.set(f, (vBase + 3 * t + c) * 3);
      }
    } else {
      stats.repairedNormals += repairNormals(positions, normals, vBase, vCount, nt, corner);
    }

    if (inst.tangents && opts.generateTangents) {
      const range = (arr: Float32Array, n: number) => arr.subarray(vBase * n, (vBase + vCount) * n);
      try {
        const tan = opts.generateTangents(range(positions, 3), range(normals, 3), range(uv0, 2));
        if (tan.length !== vCount * 4) throw new Error(`MikkTSpace returned ${tan.length} floats, expected ${vCount * 4}`);
        for (let i = 3; i < tan.length; i += 4) tan[i] = -tan[i]; // glTF uv convention (matches gltf-transform)
        tangents.set(tan, vBase * 4);
        stats.tangentPrimitives++;
      } catch (e) {
        warn(`MikkTSpace failed on a primitive (${e instanceof Error ? e.message : String(e)}); tangents left zero`);
      }
    }

    // Colour alpha < 1 anywhere forces the any-hit test on a MASK material.
    let colorAlphaOne = true;
    if (C && C.size === 4) for (let i = 3; i < C.data.length; i += 4) if (C.data[i] < 1) { colorAlphaOne = false; break; }
    const info = inst.info;
    let flags = 0;
    if (flipped) flags |= TRI_FLIPPED;
    if (info.emissive) flags |= TRI_EMISSIVE;
    if (info.masked && !(info.alphaConstOne && colorAlphaOne)) flags |= TRI_ALPHA_MASK;

    for (let t = 0; t < nt; t++) {
      const i0 = corner(t, 0), i1 = corner(t, 1), i2 = corner(t, 2);
      const q = triQuality(positions, i0, i1, i2);
      if (q === 'nonfinite') { stats.droppedNonFinite++; continue; }
      if (q === 'degenerate') { stats.droppedDegenerate++; continue; }
      indices[3 * tOut] = i0; indices[3 * tOut + 1] = i1; indices[3 * tOut + 2] = i2;
      triMaterial[tOut] = info.index;
      triFlags[tOut] = flags;
      if (flipped) stats.flippedTriangles++;
      for (const vi of [i0, i1, i2]) {
        for (let k = 0; k < 3; k++) {
          const p = positions[vi * 3 + k];
          if (p < bmin[k]) bmin[k] = p;
          if (p > bmax[k]) bmax[k] = p;
        }
      }
      tOut++;
    }
    vBase += vCount;
  }

  if (stats.droppedDegenerate) warn(`dropped ${stats.droppedDegenerate} zero-area triangle(s); primIds are dense over the kept triangles`);
  if (stats.droppedNonFinite) warn(`dropped ${stats.droppedNonFinite} triangle(s) with non-finite positions`);
  if (stats.repairedNormals) warn(`${stats.repairedNormals} zero/non-finite vertex normal(s) replaced by face normals`);
  if (tOut < totalTris) {
    indices = indices.slice(0, tOut * 3);
    triMaterial = triMaterial.slice(0, tOut);
    triFlags = triFlags.slice(0, tOut);
  }
  stats.triangles = tOut;
  const bounds: Bounds = tOut > 0
    ? { min: [bmin[0], bmin[1], bmin[2]], max: [bmax[0], bmax[1], bmax[2]] }
    : { min: [0, 0, 0], max: [0, 0, 0] };
  return { geometry: { positions, normals, tangents, uv0, color0, indices, triMaterial, triFlags }, bounds, stats };
}

// ---------------------------------------------------------------------------------------------------------------

/** Triangle-list indices for TRIANGLES / TRIANGLE_STRIP / TRIANGLE_FAN (glTF spec winding); null otherwise. */
export function triangleList(prim: Primitive, vertexCount: number): Uint32Array | null {
  const mode = prim.getMode();
  const idxAcc = prim.getIndices();
  const src = idxAcc ? (idxAcc.getArray() as ArrayLike<number>) : null;
  const n = src ? src.length : vertexCount;
  const at = (i: number) => (src ? src[i] : i);
  if (mode === MODE_TRIANGLES) {
    const out = new Uint32Array(n - (n % 3));
    for (let i = 0; i < out.length; i++) out[i] = at(i);
    return out;
  }
  if (mode === MODE_TRIANGLE_STRIP) {
    const nt = Math.max(0, n - 2), out = new Uint32Array(nt * 3);
    // glTF: p_i = {v_i, v_{i+(1+i%2)}, v_{i+(2-i%2)}}
    for (let i = 0; i < nt; i++) { out[3 * i] = at(i); out[3 * i + 1] = at(i + 1 + (i % 2)); out[3 * i + 2] = at(i + 2 - (i % 2)); }
    return out;
  }
  if (mode === MODE_TRIANGLE_FAN) {
    const nt = Math.max(0, n - 2), out = new Uint32Array(nt * 3);
    // glTF: p_i = {v_{i+1}, v_{i+2}, v_0}
    for (let i = 0; i < nt; i++) { out[3 * i] = at(i + 1); out[3 * i + 1] = at(i + 2); out[3 * i + 2] = at(0); }
    return out;
  }
  return null;
}

interface Attr { data: Float32Array; size: number }

/** Float view of an accessor; normalized integer data (if dequantize() left any) is mapped per the glTF spec. */
function readAttr(acc: Accessor | null): Attr | null {
  if (!acc) return null;
  const arr = acc.getArray();
  if (!arr) return null;
  const size = acc.getElementSize();
  if (arr instanceof Float32Array) return { data: arr, size };
  const out = new Float32Array(arr.length);
  if (acc.getNormalized()) {
    const div = arr instanceof Int8Array ? 127 : arr instanceof Uint8Array ? 255 : arr instanceof Int16Array ? 32767
      : arr instanceof Uint16Array ? 65535 : arr instanceof Uint32Array ? 4294967295 : 1;
    const signed = arr instanceof Int8Array || arr instanceof Int16Array;
    for (let i = 0; i < arr.length; i++) out[i] = signed ? Math.max(arr[i] / div, -1) : arr[i] / div;
  } else {
    for (let i = 0; i < arr.length; i++) out[i] = arr[i];
  }
  return { data: out, size };
}

function faceNormal(p: Float32Array, i0: number, i1: number, i2: number): [number, number, number] {
  const ax = p[i1 * 3] - p[i0 * 3], ay = p[i1 * 3 + 1] - p[i0 * 3 + 1], az = p[i1 * 3 + 2] - p[i0 * 3 + 2];
  const bx = p[i2 * 3] - p[i0 * 3], by = p[i2 * 3 + 1] - p[i0 * 3 + 1], bz = p[i2 * 3 + 2] - p[i0 * 3 + 2];
  const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
  const l = Math.hypot(nx, ny, nz);
  return l > 0 && Number.isFinite(l) ? [nx / l, ny / l, nz / l] : [0, 0, 0];
}

/** Zero-area test on the stored f32 positions (what the GPU intersects), evaluated in f64. */
function triQuality(p: Float32Array, i0: number, i1: number, i2: number): 'ok' | 'degenerate' | 'nonfinite' {
  for (const i of [i0, i1, i2]) {
    if (!Number.isFinite(p[i * 3]) || !Number.isFinite(p[i * 3 + 1]) || !Number.isFinite(p[i * 3 + 2])) return 'nonfinite';
  }
  const ax = p[i1 * 3] - p[i0 * 3], ay = p[i1 * 3 + 1] - p[i0 * 3 + 1], az = p[i1 * 3 + 2] - p[i0 * 3 + 2];
  const bx = p[i2 * 3] - p[i0 * 3], by = p[i2 * 3 + 1] - p[i0 * 3 + 1], bz = p[i2 * 3 + 2] - p[i0 * 3 + 2];
  const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
  const a2 = nx * nx + ny * ny + nz * nz;
  if (!Number.isFinite(a2)) return 'nonfinite';
  return a2 > 0 ? 'ok' : 'degenerate';
}

/** Replace zero normals (marked by writeVertex) with the normalized sum of adjacent face normals. */
function repairNormals(pos: Float32Array, nrm: Float32Array, vBase: number, vCount: number, nt: number,
  corner: (t: number, c: 0 | 1 | 2) => number): number {
  let bad = 0;
  for (let v = vBase; v < vBase + vCount; v++) if (nrm[v * 3] === 0 && nrm[v * 3 + 1] === 0 && nrm[v * 3 + 2] === 0) bad++;
  if (!bad) return 0;
  const acc = new Float64Array(vCount * 3);
  for (let t = 0; t < nt; t++) {
    const i0 = corner(t, 0), i1 = corner(t, 1), i2 = corner(t, 2);
    const f = faceNormal(pos, i0, i1, i2);
    for (const vi of [i0, i1, i2]) { const o = (vi - vBase) * 3; acc[o] += f[0]; acc[o + 1] += f[1]; acc[o + 2] += f[2]; }
  }
  for (let v = vBase; v < vBase + vCount; v++) {
    if (nrm[v * 3] !== 0 || nrm[v * 3 + 1] !== 0 || nrm[v * 3 + 2] !== 0) continue;
    const o = (v - vBase) * 3;
    const l = Math.hypot(acc[o], acc[o + 1], acc[o + 2]);
    if (l > 0) { nrm[v * 3] = acc[o] / l; nrm[v * 3 + 1] = acc[o + 1] / l; nrm[v * 3 + 2] = acc[o + 2] / l; }
    else nrm[v * 3 + 1] = 1; // isolated/degenerate vertex: any unit vector (never shaded; its triangles are dropped)
  }
  return bad;
}
