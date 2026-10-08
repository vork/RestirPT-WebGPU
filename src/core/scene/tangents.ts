// M7 tangents with Blender semantics (docs/decisions/m7-api.md §1.2; math.md#normal-maps).
// Tangents are never exported (scene-bridge.md): Blender recomputes MikkTSpace on the package mesh, ONE object, all faces,
// from its corner normals (custom normals on smooth packages, face normals on flat ones) and the UV map. sceneTangents()
// does the same on our side, so both renderers start the Normal Map node from the same frame:
//   - MikkTSpace (the Rust port, gltf-loader.ts loadMikkTSpace) over the triangle soup of the WHOLE mesh in primId order;
//     MikkTSpace welds corners by VALUE (position, normal, uv), so the soup is equivalent to Blender's indexed mesh;
//   - corner normal = the stored vertex normal, or the f32 face normal of a TRI_FLAT face in a flat-shaded scene;
//   - UVs in the glTF convention, w negated afterwards (glTF / gltf-transform convention; the flip v_b = 1 − v of the
//     bridge flips MikkTSpace's bitangent, so B = w·(n × t) is the same vector Blender's sign·(n × t) gives);
//   - quantized scenes: oct 2 × 15 snap of t.xyz, w = ±1 (data-formats.md §B4); lossless scenes keep the f32 output;
//   - a vertex whose corners receive different tangents is split (attributes copied; first corner keeps the slot), so
//     the per-vertex tangent the GPU interpolates is exactly each corner's MikkTSpace tangent.
// Only scenes with a normal-mapped material get tangents (sceneHasNormalMaps); every other scene is returned unchanged.
import { octSnap } from './quantize.ts';
import { TRI_FLAT, type SceneData, type SceneGeometry } from './types.ts';
import type { TangentGenerator } from './quantize.ts';

export interface TangentStats { corners: number; splitVertices: number; zeroTangents: number; ms: number }

export const sceneHasNormalMaps = (scene: Pick<SceneData, 'materials'>): boolean => scene.materials.some((m) => !!m.normalTexture);

const f32 = Math.fround;

/** Per-corner MikkTSpace input (triangle soup) of `g`. */
export function tangentSoup(g: SceneGeometry, flatFaceNormals: boolean): { position: Float32Array; normal: Float32Array; texcoord: Float32Array } {
  const nT = g.indices.length / 3;
  const position = new Float32Array(nT * 9), normal = new Float32Array(nT * 9), texcoord = new Float32Array(nT * 6);
  for (let t = 0; t < nT; t++) {
    const ia = g.indices[3 * t], ib = g.indices[3 * t + 1], ic = g.indices[3 * t + 2];
    let fn: number[] | undefined;
    if (flatFaceNormals && (g.triFlags[t] & TRI_FLAT)) {
      const P = g.positions;
      const e1 = [f32(P[3 * ib] - P[3 * ia]), f32(P[3 * ib + 1] - P[3 * ia + 1]), f32(P[3 * ib + 2] - P[3 * ia + 2])];
      const e2 = [f32(P[3 * ic] - P[3 * ia]), f32(P[3 * ic + 1] - P[3 * ia + 1]), f32(P[3 * ic + 2] - P[3 * ia + 2])];
      const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      const l = Math.hypot(n[0], n[1], n[2]) || 1;
      fn = [f32(n[0] / l), f32(n[1] / l), f32(n[2] / l)];
    }
    [ia, ib, ic].forEach((v, k) => {
      const c = 3 * t + k;
      for (let j = 0; j < 3; j++) { position[3 * c + j] = g.positions[3 * v + j]; normal[3 * c + j] = fn ? fn[j] : g.normals[3 * v + j]; }
      texcoord[2 * c] = g.uv0[2 * v]; texcoord[2 * c + 1] = g.uv0[2 * v + 1];
    });
  }
  return { position, normal, texcoord };
}

/**
 * The scene with Blender-semantics MikkTSpace tangents (module comment). `quantized`: oct-15 snap (the vertex packer
 * requires exact codes). `flatFaceNormals`: TRI_FLAT faces feed their face normal (a flat-shaded package; Blender
 * shade_flat), else the stored vertex normal (a smooth package: Blender's custom normals on every face).
 */
export function sceneTangents(scene: SceneData, gen: TangentGenerator, o: { quantized: boolean; flatFaceNormals: boolean }): { scene: SceneData; stats: TangentStats } {
  const t0 = performance.now();
  const g = scene.geometry;
  const nT = g.indices.length / 3;
  const nV0 = g.positions.length / 3;
  const soup = tangentSoup(g, o.flatFaceNormals);
  const tan = gen(soup.position, soup.normal, soup.texcoord);
  if (tan.length !== nT * 12) throw new Error(`MikkTSpace returned ${tan.length} floats for ${nT * 3} corners`);
  const snapped = new Float32Array(3);
  let zero = 0;
  for (let c = 0; c < nT * 3; c++) {
    const x = tan[4 * c], y = tan[4 * c + 1], z = tan[4 * c + 2], w = -tan[4 * c + 3];   // glTF convention: w negated
    if (!(Math.hypot(x, y, z) > 0) || !Number.isFinite(x + y + z) || w === 0) { tan.fill(0, 4 * c, 4 * c + 4); zero++; continue; }
    if (o.quantized) {
      octSnap(x, y, z, 15, snapped, 0);
      tan[4 * c] = snapped[0]; tan[4 * c + 1] = snapped[1]; tan[4 * c + 2] = snapped[2]; tan[4 * c + 3] = w < 0 ? -1 : 1;
    } else tan[4 * c + 3] = w;
  }
  // assign corners → vertices, splitting vertices whose corners disagree
  const tb = new Uint32Array(tan.buffer, tan.byteOffset, tan.length);
  const key = (c: number) => `${tb[4 * c]},${tb[4 * c + 1]},${tb[4 * c + 2]},${tb[4 * c + 3]}`;
  const slot = new Map<string, number>();     // `${v}|${tangent bits}` → output vertex
  const owner = new Int32Array(nV0).fill(-1); // first corner of each original vertex
  const extra: number[] = [];                 // original vertex of each appended copy
  const indices = new Uint32Array(g.indices.length);
  const tangentOf: number[] = [];             // output vertex → corner that defined its tangent
  for (let c = 0; c < nT * 3; c++) {
    const v = g.indices[c];
    const k = `${v}|${key(c)}`;
    let out = slot.get(k);
    if (out === undefined) {
      if (owner[v] < 0) { owner[v] = c; out = v; } else { out = nV0 + extra.length; extra.push(v); }
      slot.set(k, out);
      tangentOf[out] = c;
    }
    indices[c] = out;
  }
  const nV = nV0 + extra.length;
  const copy = (a: Float32Array, n: number): Float32Array => {
    const r = new Float32Array(nV * n);
    r.set(a.subarray(0, nV0 * n));
    extra.forEach((v, i) => r.set(a.subarray(v * n, v * n + n), (nV0 + i) * n));
    return r;
  };
  const tangents = new Float32Array(nV * 4);
  for (let v = 0; v < nV; v++) { const c = tangentOf[v]; if (c !== undefined) tangents.set(tan.subarray(4 * c, 4 * c + 4), 4 * v); }
  const geometry: SceneGeometry = {
    positions: copy(g.positions, 3), normals: copy(g.normals, 3), tangents, uv0: copy(g.uv0, 2),
    ...(g.color0 ? { color0: copy(g.color0, 4) } : {}),
    indices, triMaterial: g.triMaterial, triFlags: g.triFlags,
  };
  return { scene: { ...scene, geometry }, stats: { corners: nT * 3, splitVertices: extra.length, zeroTangents: zero, ms: performance.now() - t0 } };
}

/** sceneTangents with the MikkTSpace module loaded on demand; scenes without normal maps are returned unchanged. */
export async function withSceneTangents(scene: SceneData, o: { quantized: boolean; flatFaceNormals: boolean }): Promise<{ scene: SceneData; stats?: TangentStats }> {
  if (!sceneHasNormalMaps(scene) || scene.geometry.indices.length === 0) return { scene };
  const { loadMikkTSpace } = await import('./gltf-loader.ts');
  return sceneTangents(scene, await loadMikkTSpace(), o);
}
