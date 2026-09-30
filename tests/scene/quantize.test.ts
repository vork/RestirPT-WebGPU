// quantizeScene + the GPU vertex arena (docs/decisions/data-formats.md §D P0/P1 tests U-Q1…U-Q7): lattice choice,
// round trips, error bounds, idempotency, watertightness (CPU part), re-weld, packer exactness, package v2 checks.
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { computeRenderOrigin } from '../../src/core/render/frame-uniforms.ts';
import { recentrePositions } from '../../src/core/render/scene-gpu.ts';
import { VERTEX_FORMAT_Q, decodeVertex, f32bits, packVertexArena } from '../../src/core/gpu/vertex-format.ts';
import {
  C_OCT15, C_OCT16, C_UNORM8, DEFAULT_UV_TOLERANCE, POS_MAX_OFFSET, QuantizationError, assertQuantized, choosePosLog2, octCodeExact,
  octDecode, octEncode, octSnap, quantizeScene, reweld, uvWorstTexel, vertexLatticeGroups,
} from '../../src/core/scene/quantize.ts';
import { ScenePackageError, exportScenePackage, readScenePackage } from '../../src/core/scene/scene-package.ts';
import { TRI_FLAT, type MaterialData, type SceneData, type SceneGeometry, type TextureData } from '../../src/core/scene/types.ts';

const f32 = Math.fround;
const deg = (rad: number) => rad * 180 / Math.PI;

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const MAT: MaterialData = {
  name: 'm', model: 'principled', baseColorFactor: [0.8, 0.8, 0.8, 1], metallicFactor: 0, roughnessFactor: 0.5, emissiveFactor: [0, 0, 0],
  emissiveStrength: 1, ior: 1.5, specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true,
};
const tex = (w: number, h: number): TextureData => ({ name: 't', width: w, height: h, pixels: new Uint8Array(w * h * 4), wrapS: 'repeat', wrapT: 'repeat', filter: 'linear' });

/** Triangle soup (unwelded) with face normals and the given uvs / materials. */
function soup(tris: number[][][], o: { uv?: number[][][]; mat?: number[]; normals?: number[][][]; color?: number[][][] } = {}): SceneGeometry {
  const P: number[] = [], N: number[] = [], U: number[] = [], C: number[] = [], I: number[] = [];
  tris.forEach((t, k) => {
    const [a, b, c] = t;
    const e1 = b.map((x, i) => x - a[i]), e2 = c.map((x, i) => x - a[i]);
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const l = Math.hypot(n[0], n[1], n[2]);
    for (let j = 0; j < 3; j++) {
      I.push(P.length / 3);
      P.push(...t[j]);
      N.push(...(o.normals?.[k][j] ?? n.map((x) => x / l)));
      U.push(...(o.uv?.[k][j] ?? [0, 0]));
      if (o.color) C.push(...o.color[k][j]);
    }
  });
  const nV = P.length / 3;
  return {
    positions: Float32Array.from(P), normals: Float32Array.from(N), tangents: new Float32Array(nV * 4), uv0: Float32Array.from(U),
    ...(o.color ? { color0: Float32Array.from(C) } : {}),
    indices: Uint32Array.from(I), triMaterial: Uint32Array.from(o.mat ?? tris.map(() => 0)), triFlags: new Uint32Array(tris.length),
  };
}

const sceneOf = (geometry: SceneGeometry, materials: MaterialData[] = [MAT], textures: TextureData[] = []): SceneData => ({
  name: 't', geometry, materials, textures, lights: [], cameras: [], bounds: { min: [0, 0, 0], max: [0, 0, 0] }, warnings: [],
});

const bitsEqual = (a: Float32Array | Uint32Array, b: Float32Array | Uint32Array) =>
  a.length === b.length && new Uint32Array(a.buffer, a.byteOffset, a.length).every((x, i) => x === new Uint32Array(b.buffer, b.byteOffset, b.length)[i]);

// ---------------------------------------------------------------------------------------------------------------

describe('U-Q1 positions: global power-of-two lattice (P21)', () => {
  it('k selection: extent term, |coordinate| term (c0a_far), precision floor → lossless', () => {
    expect(choosePosLog2(29.77, 25)).toBe(-16);                          // Sponza
    expect(choosePosLog2(0.555, 0.555)).toBe(-21);                       // Cornell
    expect(choosePosLog2(5.3, 1003)).toBe(-14);                          // c0a_far: A = 1003 m binds
    expect(choosePosLog2(2.5, 2)).toBe(-19);                             // xi_contact
    expect(choosePosLog2(2 ** 21 * 2 ** -10 - 1, 10)).toBe(-10);         // just inside the 2 km floor
    expect(choosePosLog2(3000, 1500)).toBeNull();                        // > 2 km → lossless fallback
    // every offset fits 21 bits after rounding both ends; |n| < 2^24 (world coordinate exact in f32)
    for (const [E, A] of [[29.77, 25], [1, 1003], [1e-3, 1e-3], [2000, 2000]]) {
      const k = choosePosLog2(E, A)!;
      expect(E / 2 ** k + 1).toBeLessThanOrEqual(POS_MAX_OFFSET);
      expect(A / 2 ** k).toBeLessThanOrEqual(2 ** 24 - 1);
    }
    const big = quantizeScene(sceneOf(soup([[[0, 0, 0], [3000, 0, 0], [0, 1, 0]]])));
    expect(big.scene.quant!.mode).toBe('lossless');
    expect(big.scene.warnings.some((w) => /quantization floor/.test(w))).toBe(true);
  });

  it('snap error ≤ s/2 per axis, values exact in f32, idempotent bit for bit', () => {
    const R = rng(7);
    const tris: number[][][] = [];
    for (let t = 0; t < 2000; t++) {
      const c = [R() * 40 - 20, R() * 10, R() * 30 - 15];
      tris.push([0, 1, 2].map(() => c.map((x) => x + (R() - 0.5) * 0.5)));
    }
    const g = soup(tris);
    const { scene: q, stats } = quantizeScene(sceneOf(g));
    const s = 2 ** q.quant!.posLog2;
    expect(q.quant!.posLog2).toBe(-15); // extent ≈ 40.5 m
    expect(stats.maxPosErrAxis).toBeLessThanOrEqual(s / 2);
    expect(stats.maxPosErr).toBeLessThanOrEqual(s / 2 * Math.sqrt(3));
    for (const x of q.geometry.positions) { expect(Number.isInteger(x / s)).toBe(true); expect(f32(x)).toBe(x); }
    // idempotency: quantizeScene(quantizeScene(x)) === quantizeScene(x)
    const q2 = quantizeScene(q).scene;
    for (const k of ['positions', 'normals', 'uv0', 'tangents', 'indices', 'triMaterial', 'triFlags'] as const) expect(bitsEqual(q2.geometry[k], q.geometry[k]), k).toBe(true);
    expect(q2.quant).toEqual(q.quant);
    expect(() => assertQuantized(q.geometry, q.quant!)).not.toThrow();
  });
});

describe('U-Q2 exactness: recentring, MT edges, world export', () => {
  it('recentred positions and edge vectors are exact in f32 on the lattice (synthetic + every validation package)', async () => {
    const check = (sc: SceneData): number => {
      const O = computeRenderOrigin(sc.bounds, sc.quant);
      const r = recentrePositions(sc.geometry.positions, O);
      let inexact = 0;
      for (let i = 0; i < r.length; i++) if (r[i] !== sc.geometry.positions[i] - O[i % 3]) inexact++;
      const p = r, idx = sc.geometry.indices;
      for (let t = 0; t < idx.length; t += 3) for (const c of [1, 2]) for (let k = 0; k < 3; k++) {
        const d = p[3 * idx[t + c] + k] - p[3 * idx[t] + k];
        if (f32(d) !== d) inexact++;
      }
      return inexact;
    };
    const R = rng(3);
    const tris = Array.from({ length: 500 }, () => [0, 1, 2].map(() => [1e4 + R() * 3, R() * 3, -5e3 + R() * 3]));
    expect(check(quantizeScene(sceneOf(soup(tris))).scene)).toBe(0);
    const dir = 'validation/scenes';
    const pkgs = existsSync(dir) ? readdirSync(dir).filter((d) => existsSync(`${dir}/${d}/scene.json`)) : [];
    let total = 0;
    for (const d of pkgs) {
      const files = Object.fromEntries(readdirSync(`${dir}/${d}`).map((f) => [f, new Uint8Array(readFileSync(`${dir}/${d}/${f}`))]));
      const p = await readScenePackage(files);
      total += check(p.scene);
      // the GPU vertex arena recodes every package losslessly (throws otherwise)
      const O = computeRenderOrigin(p.scene.bounds, p.scene.quant);
      expect(packVertexArena(p.scene.geometry, recentrePositions(p.scene.geometry.positions, O), p.scene.quant, O).format, d).toBe(VERTEX_FORMAT_Q);
      for (let i = 0; i < p.scene.geometry.positions.length; i++) expect(Math.abs(p.scene.geometry.positions[i] / 2 ** p.scene.quant!.posLog2)).toBeLessThan(2 ** 24);
    }
    expect(total).toBe(0);
  });
});

describe('U-Q4 / U-Q5 unit vectors: octahedral 2 × 16 snorm normals, 2 × 15 + sign tangents', () => {
  const dirs = (n: number, seed: number): number[][] => {
    const R = rng(seed), out: number[][] = [];
    for (let i = 0; i < n; i++) {
      const z = 2 * R() - 1, phi = 2 * Math.PI * R(), r = Math.sqrt(1 - z * z);
      out.push([r * Math.cos(phi), r * Math.sin(phi), z]);
    }
    // adversarial: axes, diagonals, equator, fold boundary (x = 0 / y = 0 below the equator), near-poles
    for (const s of [1, -1]) for (let a = 0; a < 3; a++) { const v = [0, 0, 0]; v[a] = s; out.push(v); }
    for (let k = 0; k < 2000; k++) {
      const t = (k / 2000) * 2 * Math.PI, e = 1e-7 * (k % 7);
      out.push([Math.cos(t), Math.sin(t), e]);
      out.push([0, Math.cos(t), -Math.abs(Math.sin(t))]);
      out.push([Math.cos(t), 0, -Math.abs(Math.sin(t))]);
      const z = 1 - 1e-9 * k;
      out.push([Math.sqrt(1 - z * z), 0, -z]);
    }
    return out;
  };
  const worst = (bits: 16 | 15, v: number[][]) => {
    const o = new Float32Array(3);
    let w = 0;
    for (const d of v) {
      octSnap(d[0], d[1], d[2], bits, o, 0);
      const c = (o[0] * d[0] + o[1] * d[1] + o[2] * d[2]) / Math.hypot(o[0], o[1], o[2]);
      const s = Math.hypot(o[1] * d[2] - o[2] * d[1], o[2] * d[0] - o[0] * d[2], o[0] * d[1] - o[1] * d[0]) / Math.hypot(o[0], o[1], o[2]);
      w = Math.max(w, deg(Math.atan2(s, c)));
      octCodeExact(o, 0, bits); // the decoded value is recoverable bit for bit (packer)
    }
    return w;
  };

  it('normals ≤ 0.0025° (precise encoder); axis vectors exact; decoded values always re-encode exactly', () => {
    const w = worst(16, dirs(300_000, 11));
    console.log(`U-Q4 oct16 worst ${w.toFixed(5)} deg`);
    expect(w).toBeLessThanOrEqual(0.0025);
    const o = new Float32Array(3);
    for (const a of [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]) {
      octSnap(a[0], a[1], a[2], 16, o, 0);
      expect(Array.from(o)).toEqual(a);
    }
  });

  it('tangents ≤ 0.0050° (oct 2 × 15); the bitangent sign survives quantizeScene', () => {
    const w = worst(15, dirs(200_000, 12));
    console.log(`U-Q5 oct15 worst ${w.toFixed(5)} deg`);
    expect(w).toBeLessThanOrEqual(0.005);
    const g = soup([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[2, 0, 0], [3, 0, 0], [2, 1, 0]]]);
    g.tangents.set([0.6, 0.8, 0, 1, 0.6, 0.8, 0, 1, 0.6, 0.8, 0, 1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1]);
    const q = quantizeScene(sceneOf(g)).scene.geometry;
    const ws = Array.from({ length: q.positions.length / 3 }, (_, v) => q.tangents[4 * v + 3]);
    expect(ws.filter((x) => x === 1).length).toBeGreaterThan(0);
    expect(ws.filter((x) => x === -1).length).toBeGreaterThan(0);
    expect(ws.every((x) => x === 1 || x === -1)).toBe(true);
  });

  it('TS decode mirror = the WGSL formula (f32 element-wise), constants = f32(1/(2^b − 1))', () => {
    expect(C_OCT16).toBe(f32(1 / 32767));
    expect(C_OCT15).toBe(f32(1 / 16383));
    expect(f32bits(C_OCT16)).toBe(0x38000100);
    expect(f32bits(C_UNORM8)).toBe(0x3b808081);
    const [qx, qy] = octEncode(0.3, -0.5, -0.8, 16);
    const d = octDecode(qx, qy, C_OCT16) as number[];
    const ex = f32(qx * C_OCT16), ey = f32(qy * C_OCT16), vz = f32(f32(1 - Math.abs(ex)) - Math.abs(ey)), t = Math.max(-vz, 0);
    const vx = f32(ex + (ex >= 0 ? -t : t)), vy = f32(ey + (ey >= 0 ? -t : t));
    const inv = f32(1 / f32(Math.sqrt(f32(f32(f32(vx * vx) + f32(vy * vy)) + f32(vz * vz)))));
    expect(d).toEqual([f32(vx * inv), f32(vy * inv), f32(vz * inv)]);
  });
});

describe('U-Q6 UVs: per-material dyadic lattices, τ bound, wide fallback', () => {
  it('lattice decode is exact, τ holds per texture ref (incl. KHR_texture_transform), 1 − v stays exact', () => {
    const R = rng(5);
    const tris: number[][][] = [], uv: number[][][] = [];
    for (let t = 0; t < 300; t++) {
      const c = [R() * 4, R() * 4, 0];
      tris.push([[c[0], c[1], 0], [c[0] + 0.1, c[1], 0], [c[0], c[1] + 0.1, 0]]);
      uv.push([0, 1, 2].map(() => [R() * 3 - 1, R() * 2]));
    }
    const xf: [number, number, number, number, number, number] = [0, 2, 0.25, -3, 0, 0.5]; // rotated 90°, scaled (2, 3)
    const mats: MaterialData[] = [{ ...MAT, baseColorTexture: { texture: 0, texCoord: 0 }, normalTexture: { texture: 1, texCoord: 0, transform: xf, scale: 1 } }];
    const texs = [tex(1024, 1024), tex(2048, 1024)];
    const { scene: q, stats } = quantizeScene(sceneOf(soup(tris, { uv }), mats, texs));
    const L = q.quant!.uv[0];
    expect(L.wide).toBe(false);
    const worst = uvWorstTexel([mats[0].baseColorTexture!, mats[0].normalTexture!], texs, 2 ** L.ku, 2 ** L.kv);
    expect(worst).toBeLessThanOrEqual(DEFAULT_UV_TOLERANCE);
    expect(stats.uv[0].worstTexel).toBe(worst);
    // measured texel error of every corner ≤ the bound
    const g0 = soup(tris, { uv }).uv0;
    let measured = 0;
    for (let t = 0; t < 300; t++) for (let c = 0; c < 3; c++) {
      const v = q.geometry.indices[3 * t + c], s = 3 * t + c;
      const du = q.geometry.uv0[2 * v] - g0[2 * s], dv = q.geometry.uv0[2 * v + 1] - g0[2 * s + 1];
      measured = Math.max(measured, 1024 * Math.abs(du), 1024 * Math.abs(dv), 2048 * Math.abs(xf[0] * du + xf[1] * dv), 1024 * Math.abs(xf[3] * du + xf[4] * dv));
    }
    expect(measured).toBeLessThanOrEqual(worst * (1 + 1e-12));
    for (let v = 0; v < q.geometry.uv0.length / 2; v++) {
      const u = q.geometry.uv0[2 * v], w = q.geometry.uv0[2 * v + 1];
      expect(Number.isInteger(u / 2 ** L.ku - L.baseU) && Number.isInteger(w / 2 ** L.kv - L.baseV)).toBe(true);
      expect(f32(1 - w)).toBe(1 - w); // build_scene.py's v_b = 1 − v is exact on the lattice
    }
  });

  it('wide fallback when τ fails (large UV range on a big texture), τ = 0 keeps every textured material f32', () => {
    const tris = [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]];
    const uv = [[[-27.8, 0.1], [29.7, 0.2], [0.3, 0.3]]];
    const mats: MaterialData[] = [{ ...MAT, baseColorTexture: { texture: 0, texCoord: 0 } }, { ...MAT }];
    const r = quantizeScene(sceneOf(soup(tris, { uv }), mats, [tex(1024, 1024)]));
    expect(r.stats.wideMaterials).toEqual([0]);
    expect(Array.from(r.scene.geometry.uv0.subarray(0, 2))).toEqual([f32(-27.8), f32(0.1)]); // untouched f32
    const z = quantizeScene(sceneOf(soup(tris, { uv: [[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]] }), mats, [tex(4, 4)]), { uvTolerance: 0 });
    expect(z.scene.quant!.uv[0].wide).toBe(true);
    expect(z.scene.quant!.uv[1].wide).toBe(false); // untextured: any lattice is exact enough
  });

  it('a vertex shared by two materials with different lattices is duplicated (one lattice per vertex)', () => {
    const g: SceneGeometry = {
      positions: Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]), normals: Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
      tangents: new Float32Array(16), uv0: Float32Array.from([0.1, 0.1, 5, 0.2, 0.3, 0.9, 0.7, 0.7]),
      indices: Uint32Array.from([0, 1, 2, 1, 3, 2]), triMaterial: Uint32Array.from([0, 1]), triFlags: new Uint32Array(2),
    };
    const mats: MaterialData[] = [{ ...MAT, baseColorTexture: { texture: 0, texCoord: 0 } }, { ...MAT, baseColorTexture: { texture: 0, texCoord: 0, transform: [4, 0, 0, 0, 4, 0] } }];
    const r = quantizeScene(sceneOf(g, mats, [tex(256, 256)]));
    expect(r.stats.uvDuplicatedVertices).toBe(2);
    expect(() => vertexLatticeGroups(r.scene.geometry, r.scene.quant!.uv)).not.toThrow();
  });
});

describe('U-Q7 COLOR_0', () => {
  it('unorm8 sources stay lossless (rgba8), 255 → 1.0, other values go rgba16', () => {
    const tris = [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]];
    const c8 = [[[0, 1 / 255, 128 / 255, 1], [f32(37 / 255), 1, 0, 1], [1, 1, 1, 1]].map((c) => c.map(f32))];
    const q8 = quantizeScene(sceneOf(soup(tris, { color: c8 })));
    expect(q8.scene.quant!.color).toBe('rgba8');
    const c = q8.scene.geometry.color0!;
    for (let v = 0; v < 3; v++) expect(c[4 * v + 3]).toBe(1);
    for (let i = 0; i < c.length; i++) expect(c[i]).toBe(f32(Math.round(c[i] * 255) * C_UNORM8));
    const q16 = quantizeScene(sceneOf(soup(tris, { color: [[[0.3, 0.123456, 0.5, 1], [0.1, 0.2, 0.3, 0.4], [1, 1, 1, 1]]] })));
    expect(q16.scene.quant!.color).toBe('rgba16');
    expect(Math.abs(q16.scene.geometry.color0![1] - 0.123456)).toBeLessThanOrEqual(0.5 / 65535 + 1e-8);
  });
});

describe('U-Q3 watertightness (CPU), degenerate drop, re-weld', () => {
  it('bit-identical input vertices across meshes/materials → bit-identical lattice points; near pairs ≤ 1 step apart', () => {
    const R = rng(9);
    const shared = [[R() * 7.3, R() * 2.1, R() * 5.5], [R() * 7.3, R() * 2.1, R() * 5.5]];
    const tris = [[shared[0], shared[1], [0.1, 3, 0.2]], [shared[1], shared[0], [4, -1, 3]]];
    const near = [shared[0][0] + 3e-7, shared[0][1], shared[0][2]];
    tris.push([near, [5, 5, 5], [5, 6, 5]]);
    const { scene: q } = quantizeScene(sceneOf(soup(tris, { mat: [0, 1, 0] }), [MAT, { ...MAT, name: 'b' }]));
    const P = (t: number, c: number) => Array.from(q.geometry.positions.subarray(3 * q.geometry.indices[3 * t + c], 3 * q.geometry.indices[3 * t + c] + 3));
    expect(P(0, 0)).toEqual(P(1, 1));
    expect(P(0, 1)).toEqual(P(1, 0));
    const s = 2 ** q.quant!.posLog2;
    expect(Math.abs(P(2, 0)[0] - P(0, 0)[0])).toBeLessThanOrEqual(s);
  });

  it('triangles that collapse on the lattice are dropped (dense primIds, warning); valid ones keep their order', () => {
    const tris = [[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 0], [1, 0, 0], [0.5, 1e-9, 0]], [[2, 0, 0], [3, 0, 0], [2, 1, 0]]];
    const { scene: q, stats } = quantizeScene(sceneOf(soup(tris, { mat: [0, 1, 0] }), [MAT, { ...MAT, name: 'b' }]));
    expect(stats.droppedDegenerate).toBe(1);
    expect(q.geometry.indices.length / 3).toBe(2);
    expect(Array.from(q.geometry.triMaterial)).toEqual([0, 0]);
    expect(q.warnings.some((w) => /became degenerate/.test(w))).toBe(true);
  });

  it('re-weld is lossless: every corner keeps its (p, n, uv, t, colour); identical tuples merge', () => {
    const R = rng(4);
    const quadsP: number[][][] = [], quadsUv: number[][][] = [];
    for (let k = 0; k < 50; k++) {
      const o = [R() * 5, R() * 5, 0];
      const a = [o[0], o[1], 0], b = [o[0] + 1, o[1], 0], c = [o[0] + 1, o[1] + 1, 0], d = [o[0], o[1] + 1, 0];
      quadsP.push([a, b, c], [a, c, d]);
      quadsUv.push([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]);
    }
    const g = soup(quadsP, { uv: quadsUv });
    const q = quantizeScene(sceneOf(g)).scene.geometry;
    expect(q.positions.length / 3).toBe(200); // 6 → 4 corners per quad
    const pre = quantizeScene(sceneOf(g)).scene; // deterministic
    expect(bitsEqual(pre.geometry.positions, q.positions)).toBe(true);
    // reweld itself: corner tuples preserved
    const w = reweld(g);
    for (let i = 0; i < g.indices.length; i++) {
      const a = g.indices[i], b = w.indices[i];
      expect([...w.positions.subarray(3 * b, 3 * b + 3), ...w.uv0.subarray(2 * b, 2 * b + 2)]).toEqual([...g.positions.subarray(3 * a, 3 * a + 3), ...g.uv0.subarray(2 * a, 2 * a + 2)]);
    }
  });

  it('TRI_FLAT: faces whose corner normals equal the face normal; smooth faces keep their normals', () => {
    const n = Math.SQRT1_2;
    const g = soup([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[2, 0, 0], [3, 0, 0], [2, 1, 0]]], { normals: [[[0, 0, 1], [0, 0, 1], [0, 0, 1]], [[n, 0, n], [n, 0, n], [n, 0, n]]] });
    const q = quantizeScene(sceneOf(g)).scene.geometry;
    expect(Array.from(q.triFlags)).toEqual([TRI_FLAT, 0]);
  });
});

describe('vertex arena packer (VERTEX_FORMAT 1): lossless recoding or throw', () => {
  const build = () => {
    const R = rng(21);
    const tris: number[][][] = [], uv: number[][][] = [], col: number[][][] = [], normals: number[][][] = [];
    for (let t = 0; t < 400; t++) {
      const c = [R() * 20 - 10, R() * 5, R() * 20 - 10];
      tris.push([0, 1, 2].map(() => c.map((x) => x + R() - 0.5)));
      uv.push([0, 1, 2].map(() => [R() * 8 - 4, R() * 2]));
      col.push([0, 1, 2].map(() => [R(), R(), R(), 1].map((x) => f32(Math.round(x * 255) / 255))));
      normals.push([0, 1, 2].map(() => { const z = 2 * R() - 1, p = 2 * Math.PI * R(), r = Math.sqrt(1 - z * z); return [r * Math.cos(p), r * Math.sin(p), z]; }));
    }
    const mats: MaterialData[] = [{ ...MAT, baseColorTexture: { texture: 0, texCoord: 0 } }, { ...MAT, baseColorTexture: { texture: 1, texCoord: 0 } }];
    return quantizeScene(sceneOf(soup(tris, { uv, normals, color: col, mat: tris.map((_, i) => i % 2) }), mats, [tex(512, 512), tex(1 << 14, 1 << 14)])).scene;
  };

  it('decodes every record back to the exact SceneGeometry bits (positions, normals, uv incl. wide, rgba8 colour)', () => {
    const q = build();
    expect(q.quant!.uv[1].wide).toBe(true);   // 16k² texture → τ fails → f32 UVs
    const O = computeRenderOrigin(q.bounds, q.quant);
    const rec = recentrePositions(q.geometry.positions, O);
    const a = packVertexArena(q.geometry, rec, q.quant, O);
    expect(a.format).toBe(VERTEX_FORMAT_Q);
    expect(a.wideCount).toBeGreaterThan(0);
    expect(a.bytes.records).toBe(16 * q.geometry.positions.length / 3);
    const grp = vertexLatticeGroups(q.geometry, q.quant!.uv);
    for (let v = 0; v < a.vertexCount; v++) {
      const d = decodeVertex(a, v, q.quant!.uv[grp[v]]);
      expect(d.p.map(f32bits)).toEqual(Array.from(rec.subarray(3 * v, 3 * v + 3), f32bits));
      expect(d.uv.map(f32bits)).toEqual(Array.from(q.geometry.uv0.subarray(2 * v, 2 * v + 2), f32bits));
      expect(d.n.map(f32bits)).toEqual(Array.from(q.geometry.normals.subarray(3 * v, 3 * v + 3), f32bits));
      expect(d.color.map(f32bits)).toEqual(Array.from(q.geometry.color0!.subarray(4 * v, 4 * v + 4), f32bits));
    }
  });

  it('throws on off-lattice values and on an origin that is not a lattice point', () => {
    const q = build();
    const O = computeRenderOrigin(q.bounds, q.quant);
    const bad = { ...q.geometry, positions: q.geometry.positions.slice() };
    bad.positions[q.geometry.indices[0] * 3] = f32(bad.positions[q.geometry.indices[0] * 3] + 1e-3 * 2 ** q.quant!.posLog2 + 1e-6);
    expect(() => packVertexArena(bad, recentrePositions(bad.positions, O), q.quant, O)).toThrow(QuantizationError);
    const badN = { ...q.geometry, normals: q.geometry.normals.slice() };
    badN.normals[0] = f32(badN.normals[0] + 1e-4);
    expect(() => packVertexArena(badN, recentrePositions(q.geometry.positions, O), q.quant, O)).toThrow(QuantizationError);
    const O2: [number, number, number] = [O[0] + 2 ** q.quant!.posLog2 / 2, O[1], O[2]];
    expect(() => packVertexArena(q.geometry, recentrePositions(q.geometry.positions, O2), q.quant, O2)).toThrow(/not on the/);
  });
});

describe('scene package v2', () => {
  it('writes the quant block, round-trips bit-exactly, rejects an off-lattice geometry.bin', async () => {
    const R = rng(2);
    const tris = Array.from({ length: 20 }, () => [0, 1, 2].map(() => [R(), R(), R()]));
    const q = quantizeScene(sceneOf(soup(tris))).scene;
    const opts = { camera: { matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 3, 1], yfov: 0.8 }, render: { width: 8, height: 8, maxBounces: 1 }, lightMode: 'A' as const };
    const pkg = await exportScenePackage(q, opts);
    expect(pkg.json.version).toBe(2);
    expect(pkg.json.quant).toEqual(q.quant);
    expect(pkg.json.flatShaded).toBe(true);
    const back = await readScenePackage(pkg.files);
    expect(back.scene.quant).toEqual(q.quant);
    for (const k of ['positions', 'normals', 'uv0'] as const) expect(bitsEqual(back.scene.geometry[k], q.geometry[k]), k).toBe(true);
    // off-lattice: flip the lowest mantissa bit of one position
    const bin = pkg.files.get('geometry.bin')!.slice();
    const off = pkg.json.buffers.positions.offset;
    const dv = new DataView(bin.buffer);
    dv.setUint32(off, dv.getUint32(off, true) ^ 1, true);
    const files = new Map(pkg.files); files.set('geometry.bin', bin);
    await expect(readScenePackage(files)).rejects.toThrow(ScenePackageError);
    // an unquantized scene cannot be exported
    await expect(exportScenePackage(sceneOf(soup(tris)), opts)).rejects.toThrow(/not quantized/);
  });
});
