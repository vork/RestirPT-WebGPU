// M7 tangents (docs/decisions/m7-api.md §1.2): the vertex arena's tangent section (Q oct 2 × 15 + sign, F32) round
// trips bit-exactly; sceneTangents (Blender semantics: whole-mesh MikkTSpace, corner normals as Blender sees them, glTF
// sign convention, vertex split) gives every corner its own MikkTSpace tangent; the sign convention reproduces Blender's
// B = sign·(n × t) under the bridge's v flip; packages with a normal map get tangents on read; the CPU mirror of the
// Cycles Normal Map node (used by the GPU test U-NM-1) behaves like svm_node_normal_map on hand-checked inputs.
import { describe, expect, it } from 'vitest';
import { computeRenderOrigin } from '../../src/core/render/frame-uniforms.ts';
import { recentrePositions } from '../../src/core/render/scene-gpu.ts';
import { decodeTangentWord, decodeVertex, f32bits, packVertexArena, tangentWord } from '../../src/core/gpu/vertex-format.ts';
import { loadMikkTSpace } from '../../src/core/scene/gltf-loader.ts';
import { C_OCT15, octDecode, quantizeScene } from '../../src/core/scene/quantize.ts';
import { exportScenePackage, readScenePackage } from '../../src/core/scene/scene-package.ts';
import { sceneTangents, tangentSoup } from '../../src/core/scene/tangents.ts';
import type { MaterialData, SceneData, SceneGeometry, TextureData } from '../../src/core/scene/types.ts';
import { normalMapCycles } from '../../validation/scenes/m7-kit.ts';

const MAT: MaterialData = {
  name: 'nm', model: 'principled', baseColorFactor: [0.8, 0.8, 0.8, 1], metallicFactor: 0, roughnessFactor: 0.5, emissiveFactor: [0, 0, 0],
  emissiveStrength: 1, ior: 1.5, specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true,
  normalTexture: { texture: 0, texCoord: 0, scale: 1 },
};
const TEX: TextureData = { name: 'n', width: 4, height: 4, pixels: new Uint8Array(64).fill(128), wrapS: 'repeat', wrapT: 'repeat', filter: 'linear' };

/** Smooth UV sphere (shared vertices, analytic normals, glTF UVs u = φ/2π, v = θ/π), seam duplicated. */
function uvSphere(nLat: number, nLon: number, r = 1): SceneGeometry {
  const P: number[] = [], N: number[] = [], U: number[] = [], I: number[] = [];
  for (let i = 0; i <= nLat; i++) {
    const th = Math.PI * (0.02 + 0.96 * i / nLat);     // no exact poles (no degenerate fans)
    for (let j = 0; j <= nLon; j++) {
      const ph = 2 * Math.PI * j / nLon;
      const n = [Math.sin(th) * Math.cos(ph), Math.cos(th), -Math.sin(th) * Math.sin(ph)];
      P.push(r * n[0], r * n[1], r * n[2]); N.push(...n); U.push(j / nLon, i / nLat);
    }
  }
  const at = (i: number, j: number) => i * (nLon + 1) + j;
  for (let i = 0; i < nLat; i++) for (let j = 0; j < nLon; j++) I.push(at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j), at(i + 1, j + 1), at(i, j + 1));
  const nV = P.length / 3, nT = I.length / 3;
  return { positions: Float32Array.from(P), normals: Float32Array.from(N), tangents: new Float32Array(nV * 4), uv0: Float32Array.from(U),
    indices: Uint32Array.from(I), triMaterial: new Uint32Array(nT), triFlags: new Uint32Array(nT) };
}
const sceneOf = (g: SceneGeometry): SceneData => ({ name: 's', geometry: g, materials: [MAT], textures: [TEX], lights: [], cameras: [], bounds: { min: [-1, -1, -1], max: [1, 1, 1] }, warnings: [] });
const dot = (a: ArrayLike<number>, ao: number, b: ArrayLike<number>, bo: number) => a[ao] * b[bo] + a[ao + 1] * b[bo + 1] + a[ao + 2] * b[bo + 2];

describe('M7 tangents', () => {
  it('tangent words: oct-15 + present + sign round trip (TS mirror of scene_vertex_tangent)', () => {
    const t = new Float32Array(4 * 3);
    octDecode(1234, -5678, C_OCT15, t, 0); t[3] = -1;
    octDecode(-16383, 7, C_OCT15, t, 4); t[7] = 1;
    // vertex 2: absent (w = 0)
    for (let v = 0; v < 2; v++) {
      const d = decodeTangentWord(tangentWord(t, v));
      for (let k = 0; k < 4; k++) expect(f32bits(d[k])).toBe(f32bits(t[4 * v + k]));
    }
    expect(tangentWord(t, 2)).toBe(0);
    expect(decodeTangentWord(0)).toEqual([0, 0, 0, 0]);
  });

  it('sceneTangents: per-corner MikkTSpace tangents on the vertices (split where corners disagree), oct-15 when quantized', async () => {
    const gen = await loadMikkTSpace();
    const q = quantizeScene(sceneOf(uvSphere(12, 24)), { mode: 'quantized' }).scene;
    const { scene, stats } = sceneTangents(q, gen, { quantized: true, flatFaceNormals: false });
    const g = scene.geometry;
    // the corner tangents of a direct MikkTSpace call (same soup), negated w
    const soup = tangentSoup(q.geometry, false);
    const ref = gen(soup.position, soup.normal, soup.texcoord);
    let maxDeg = 0;
    for (let c = 0; c < g.indices.length; c++) {
      const v = g.indices[c];
      expect(g.tangents[4 * v + 3]).toBe(-Math.sign(ref[4 * c + 3]));
      const a3 = [g.tangents[4 * v], g.tangents[4 * v + 1], g.tangents[4 * v + 2]], b3 = [ref[4 * c], ref[4 * c + 1], ref[4 * c + 2]];
      const cr = Math.hypot(a3[1] * b3[2] - a3[2] * b3[1], a3[2] * b3[0] - a3[0] * b3[2], a3[0] * b3[1] - a3[1] * b3[0]);
      maxDeg = Math.max(maxDeg, Math.atan2(cr, dot(a3, 0, b3, 0)) * 180 / Math.PI);   // atan2: acos loses ~0.02° at |t| = 1 ± ulp
      // positions / normals / uvs of the (possibly split) vertex are the original vertex's
      const v0 = q.geometry.indices[c];
      for (let k = 0; k < 3; k++) expect(f32bits(g.positions[3 * v + k])).toBe(f32bits(q.geometry.positions[3 * v0 + k]));
    }
    expect(maxDeg).toBeLessThan(0.0051);      // oct-15 bound (data-formats.md §B4)
    expect(stats.zeroTangents).toBe(0);
    // the arena packs them losslessly
    const origin = computeRenderOrigin(scene.bounds, scene.quant);
    const a = packVertexArena(g, recentrePositions(g.positions, origin), scene.quant, origin, { tangents: true });
    expect(a.tangents).toBe(true);
    const d = decodeVertex(a, 5, scene.quant!.uv[0]);
    for (let k = 0; k < 4; k++) expect(f32bits(d.t![k])).toBe(f32bits(g.tangents[20 + k]));
  });

  it('sign convention: B = w·(n × t) is Blender\'s sign·(n × t) under the bridge v flip (plane facing +z)', async () => {
    const gen = await loadMikkTSpace();
    // quad in z = 0, glTF uv: u = x, v = 1 − y (image top-left origin); Blender v_b = y
    const P = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], N = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], U = [0, 1, 1, 1, 1, 0, 0, 0];
    const g: SceneGeometry = { positions: Float32Array.from(P), normals: Float32Array.from(N), tangents: new Float32Array(16), uv0: Float32Array.from(U),
      indices: Uint32Array.from([0, 1, 2, 0, 2, 3]), triMaterial: new Uint32Array(2), triFlags: new Uint32Array(2) };
    const { scene } = sceneTangents(sceneOf(g), gen, { quantized: false, flatFaceNormals: false });
    const t = scene.geometry.tangents;
    expect(t[0]).toBeCloseTo(1, 6); expect(t[1]).toBeCloseTo(0, 6);   // T = dP/du = +x
    expect(t[3]).toBe(1);                                               // B = n × t = +y = dP/dv_Blender
    // mirrored u (u = 1 − x): T = −x and B must still be +y ⇒ w = −1 (n × (−x) = −y)
    const gm = { ...g, uv0: Float32Array.from([1, 1, 0, 1, 0, 0, 1, 0]), tangents: new Float32Array(16) };
    const tm = sceneTangents(sceneOf(gm), gen, { quantized: false, flatFaceNormals: false }).scene.geometry.tangents;
    expect(tm[0]).toBeCloseTo(-1, 6);
    expect(tm[3]).toBe(-1);
  });

  it('readScenePackage gives normal-mapped packages tangents (and leaves other packages alone)', async () => {
    const q = quantizeScene(sceneOf(uvSphere(6, 12)), { mode: 'quantized' }).scene;
    const pkg = await exportScenePackage(q, { camera: { matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 4, 1], yfov: 0.8 }, render: { width: 8, height: 8, maxBounces: 1 }, lightMode: 'A', flatShaded: false });
    const r = await readScenePackage(pkg.files);
    const t = r.scene.geometry.tangents;
    let present = 0;
    for (let v = 0; v < t.length / 4; v++) if (t[4 * v + 3] !== 0) present++;
    expect(present).toBe(t.length / 4);
    const noNm = { ...q, materials: [{ ...MAT, normalTexture: undefined }] };
    const r2 = await readScenePackage((await exportScenePackage(noNm, { camera: { matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 4, 1], yfov: 0.8 }, render: { width: 8, height: 8, maxBounces: 1 }, lightMode: 'A', flatShaded: false })).files);
    expect(r2.scene.geometry.tangents.every((x) => x === 0)).toBe(true);
    expect(r2.scene.geometry.positions.length).toBe(q.geometry.positions.length);
  });

  it('normalMapCycles: flat texel ⇒ n; strength scales xy and mixes z; zero ⇒ fallback', () => {
    const T: [number, number, number] = [1, 0, 0], n: [number, number, number] = [0, 0, 1];
    const flat = normalMapCycles([0.5, 0.5, 1], 1, T, 1, n)!;
    expect(flat[0]).toBeCloseTo(0, 12); expect(flat[2]).toBeCloseTo(1, 12);
    // tilt toward +x: c = (0.6, 0, 0.8) at strength 0.5 → (0.3, 0, mix(1, 0.8, 0.5) = 0.9)
    const c = [0.6, 0, 0.8], rgb = c.map((x) => x / 2 + 0.5) as [number, number, number];
    const h = normalMapCycles(rgb, 0.5, T, 1, n)!;
    const l = Math.hypot(0.3, 0.9);
    expect(h[0]).toBeCloseTo(0.3 / l, 12); expect(h[2]).toBeCloseTo(0.9 / l, 12);
    // sign −1 mirrors the y axis: c.y > 0 tilts toward −(n × t) = −y
    const m = normalMapCycles([0.5, 0.8, 0.9], 1, T, -1, n)!;
    expect(m[1]).toBeLessThan(0);
    expect(normalMapCycles([0.5, 0.5, 0.5], 1, [0, 0, 0], 0, [0, 0, 0])).toBeNull();
  });
});
