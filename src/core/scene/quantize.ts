// quantizeScene: the ONE lossy step of the geometry pipeline (docs/decisions/data-formats.md §0, §B0–§B6).
// It runs right after every loader (glTF flatten, USD, the validation scene kit, v1 packages) and writes the
// DEQUANTIZED f32 values back into SceneGeometry. Everything downstream (BVH, lights, CPU tracer, the package export
// for Blender, the GPU packer) sees exactly those values; the GPU vertex format (render/vertex-format.ts) is a
// lossless recoding of them and throws if a value does not round-trip bit for bit (PLAN §2 rule 14 for geometry).
//
// Order (data-formats.md §B0, it matters):
//   0 TRI_FLAT detection on the INPUT (every corner normal = the face normal) — before anything moves
//   1 positions → one global power-of-two lattice (P21: 3 × 21 bit offsets, step 2^k)
//   2 drop triangles that became degenerate (logged; primIds stay dense)
//   3 flat normals from the SNAPPED positions (vertices used only by TRI_FLAT triangles)
//   4 oct 2 × 16 snorm normals (precise encoder, fixed point of encode∘decode)
//   5 per-material dyadic UV lattices (2 × 16 bit) with a per-material f32 "wide" fallback when the worst texel
//     error exceeds τ; a vertex used by two lattice groups is duplicated
//   6 COLOR_0 → rgba8 (unorm8 sources, lossless) or rgba16 (exact products f32(q)·f32(1/(2^b − 1)))
//   7 MikkTSpace on the snapped positions / normals / UVs (glTF tangent ranges), then oct 2 × 15 + sign tangents
//   8 lossless re-weld of identical (p, n, uv, t, colour, lattice group) tuples
// quantizeScene(quantizeScene(x)) === quantizeScene(x) bit for bit: a scene that carries `quant` reuses its stored
// lattices (re-deriving k from snapped data is unsafe: the extent can cross a power of two).
// Lossless mode is the identity (loader-fidelity / E2E stock-import gates, and scenes above the precision floor).
import type { MaterialData, SceneData, SceneGeometry, SceneQuant, TextureRef, UvLattice } from './types.ts';
import { TRI_FLAT } from './types.ts';

export const POS_BITS = 21;
/** Largest per-axis lattice offset (21 bits). */
export const POS_MAX_OFFSET = 2 ** POS_BITS - 1;
/** Precision floor: steps coarser than 2^-10 m (≈ 0.98 mm, scenes > 2 km) fall back to lossless f32 (§B1). */
export const DEFAULT_MAX_POS_LOG2 = -10;
/** UV fidelity τ: worst texel error on the largest bound texture after KHR_texture_transform (§B5, E-12). */
export const DEFAULT_UV_TOLERANCE = 1 / 8;
/** Finest UV / position lattice exponent considered (range-0 attributes). */
export const MIN_LOG2 = -24;
/** 1 − cos of the largest corner-normal deviation that still counts as a flat face (≈ 1e-4 rad). */
export const FLAT_COS_TOL = 5e-9;
export const UV_Q_MAX = 65535;

const f32 = Math.fround;
/** Decode constants: the f32 roundings of 1/(2^b − 1); WGSL uses the same bits (vertex-format.ts, scene-data.wgsl). */
export const C_OCT16 = f32(1 / 32767);
export const C_OCT15 = f32(1 / 16383);
export const C_UNORM8 = f32(1 / 255);
export const C_UNORM16 = f32(1 / 65535);

export type QuantMode = 'quantized' | 'lossless';
export type TangentGenerator = (position: Float32Array, normal: Float32Array, texcoord: Float32Array) => Float32Array;

export interface QuantizeOptions {
  /** Default 'quantized'. */
  mode?: QuantMode;
  /** τ in texels (default 1/8). 0 = every textured material keeps f32 UVs. */
  uvTolerance?: number;
  /** Precision floor exponent (default −10). */
  maxPosLog2?: number;
  /** MikkTSpace generator + the unwelded vertex ranges that want tangents (glTF flatten). */
  generateTangents?: TangentGenerator;
  tangentRanges?: readonly { vBase: number; vCount: number }[];
}

export interface QuantizeStats {
  mode: QuantMode;
  posLog2: number;
  step: number;
  /** Worst per-axis / Euclidean displacement of a referenced vertex (m). */
  maxPosErrAxis: number;
  maxPosErr: number;
  droppedDegenerate: number;
  flatTriangles: number;
  /** Worst angle (degrees) between an input normal and its oct-snapped value (smooth vertices only). */
  maxNormalErrDeg: number;
  maxTangentErrDeg: number;
  /** Per material: lattice and worst texel error (0 for wide or untextured). */
  uv: { material: number; ku: number; kv: number; wide: boolean; worstTexel: number; maxUvErr: number }[];
  wideMaterials: number[];
  uvDuplicatedVertices: number;
  colorFormat: SceneQuant['color'];
  verticesIn: number;
  verticesOut: number;
  ms: number;
}

// ---------------------------------------------------------------------------------------------------------------
// Scalar helpers

/** Smallest integer k with 2^k ≥ x (x > 0, finite). */
export function ceilLog2(x: number): number {
  let k = Math.ceil(Math.log2(x));
  while (2 ** (k - 1) >= x) k--;
  while (2 ** k < x) k++;
  return k;
}

/** Position lattice exponent k for extent E and max |coordinate| A (§B1), or null when above the floor. */
export function choosePosLog2(extent: number, maxAbs: number, maxLog2 = DEFAULT_MAX_POS_LOG2): number | null {
  let k = MIN_LOG2 * 4;
  if (extent > 0) k = Math.max(k, ceilLog2(extent / (POS_MAX_OFFSET - 1)));
  if (maxAbs > 0) k = Math.max(k, ceilLog2(maxAbs / (2 ** 24 - 1)));
  return k > maxLog2 ? null : k;
}

/** Nearest lattice point (ties toward +∞, JS Math.round); `+ 0` maps −0 to +0 (the GPU decode never yields −0). */
export const snapToLattice = (x: number, s: number): number => Math.round(x / s) * s + 0;

// ---------------------------------------------------------------------------------------------------------------
// Octahedral unit vectors (Stubbe fold). Decode = f32 mirror of scene-data.wgsl vq_oct (element-wise f32 rounding).

export function octDecode(qx: number, qy: number, c: number, out: Float32Array | number[] = [0, 0, 0], o = 0): Float32Array | number[] {
  const ex = f32((qx | 0) * c), ey = f32((qy | 0) * c); // integer codes: `| 0` also maps −0 to +0 like the GPU's i32
  const vz = f32(f32(1 - Math.abs(ex)) - Math.abs(ey));
  const t = Math.max(-vz, 0);
  const vx = f32(ex + (ex >= 0 ? -t : t));
  const vy = f32(ey + (ey >= 0 ? -t : t));
  const l2 = f32(f32(f32(vx * vx) + f32(vy * vy)) + f32(vz * vz));
  const inv = f32(1 / f32(Math.sqrt(l2)));
  out[o] = f32(vx * inv); out[o + 1] = f32(vy * inv); out[o + 2] = f32(vz * inv);
  return out;
}

const octTmp = [0, 0, 0];
/** Precise oct encoder: the best of the 4 floor/ceil candidates by angle after the f32 decode. q ∈ [−m, m]. */
export function octEncode(x: number, y: number, z: number, bits: 16 | 15): [number, number] {
  const m = bits === 16 ? 32767 : 16383, c = bits === 16 ? C_OCT16 : C_OCT15;
  const l1 = Math.abs(x) + Math.abs(y) + Math.abs(z);
  let px = x / l1, py = y / l1;
  if (z < 0) {
    const ox = px, oy = py;
    px = (1 - Math.abs(oy)) * (ox >= 0 ? 1 : -1);
    py = (1 - Math.abs(ox)) * (oy >= 0 ? 1 : -1);
  }
  const fx = px * m, fy = py * m;
  const cx = [Math.floor(fx), Math.ceil(fx)], cy = [Math.floor(fy), Math.ceil(fy)];
  const len = Math.hypot(x, y, z);
  let best = -Infinity, bx = 0, by = 0;
  for (const qx0 of cx) for (const qy0 of cy) {
    const qx = Math.max(-m, Math.min(m, qx0)), qy = Math.max(-m, Math.min(m, qy0));
    octDecode(qx, qy, c, octTmp);
    // angle, not the raw dot: |decode| deviates from 1 by ~1e-7, far more than 1 − cos between neighbouring codes
    const d = (octTmp[0] * x + octTmp[1] * y + octTmp[2] * z) / (len * Math.hypot(octTmp[0], octTmp[1], octTmp[2]));
    if (d > best) { best = d; bx = qx | 0; by = qy | 0; }
  }
  return [bx, by];
}

/** Oct-snap a unit vector: out = decode(encode(v)) (f32). Returns the code. */
export function octSnap(x: number, y: number, z: number, bits: 16 | 15, out: Float32Array, o: number): [number, number] {
  const q = octEncode(x, y, z, bits);
  octDecode(q[0], q[1], bits === 16 ? C_OCT16 : C_OCT15, out, o);
  return q;
}

/**
 * An oct code whose f32 decode equals the stored value bit for bit; throws if none exists (value off the lattice).
 * encode∘decode is not always a fixed point (e.g. x = 0 on the folded lower hemisphere: the two mirror codes decode to
 * ±tiny x), so the neighbourhood of the encoder's code and of its fold mirrors is searched.
 */
export function octCodeExact(v: ArrayLike<number>, o: number, bits: 16 | 15): [number, number] {
  const m = bits === 16 ? 32767 : 16383, c = bits === 16 ? C_OCT16 : C_OCT15;
  const hit = (qx: number, qy: number) => {
    if (qx < -m || qx > m || qy < -m || qy > m) return false;
    octDecode(qx, qy, c, octTmp);
    return Object.is(octTmp[0], v[o]) && Object.is(octTmp[1], v[o + 1]) && Object.is(octTmp[2], v[o + 2]); // bits: −0 ≠ +0
  };
  const q = octEncode(v[o], v[o + 1], v[o + 2], bits);
  if (hit(q[0], q[1])) return q;
  for (const [sx, sy] of [[1, 1], [-1, 1], [1, -1], [-1, -1]]) {
    for (let r = 0; r <= 2; r++) for (let dx = -r; dx <= r; dx++) for (let dy = -r; dy <= r; dy++) {
      if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
      const qx = sx * q[0] + dx, qy = sy * q[1] + dy;
      if (hit(qx, qy)) return [qx, qy];
    }
  }
  octDecode(q[0], q[1], c, octTmp);
  throw new Error(`oct${bits} value (${v[o]}, ${v[o + 1]}, ${v[o + 2]}) is not on the lattice (nearest code decodes to ${octTmp.join(', ')})`);
}

// ---------------------------------------------------------------------------------------------------------------
// UV lattices (§B5)

const TEX_KEYS = ['baseColorTexture', 'metallicRoughnessTexture', 'normalTexture', 'emissiveTexture', 'transmissionTexture',
  'specularTexture', 'specularColorTexture'] as const;

export function materialTextureRefs(m: MaterialData): TextureRef[] {
  return TEX_KEYS.map((k) => m[k]).filter((r): r is TextureRef => !!r);
}

/** Worst texel error of a UV lattice with steps (su, sv) over the material's texture refs (after the transform). */
export function uvWorstTexel(refs: readonly TextureRef[], textures: readonly { width: number; height: number }[], su: number, sv: number): number {
  let worst = 0;
  for (const r of refs) {
    const t = textures[r.texture];
    if (!t) continue;
    const [a, b, , d, e] = r.transform ?? [1, 0, 0, 0, 1, 0];
    worst = Math.max(worst, t.width * (Math.abs(a) * su / 2 + Math.abs(b) * sv / 2), t.height * (Math.abs(d) * su / 2 + Math.abs(e) * sv / 2));
  }
  return worst;
}

/** Lattice exponent and base for one UV axis over [lo, hi] (finite): uv = (q + base)·2^k, q ∈ [0, 65535]. */
export function uvAxisLattice(lo: number, hi: number): { k: number; base: number } {
  let k = MIN_LOG2;
  if (hi > lo) k = Math.max(k, ceilLog2((hi - lo) / (UV_Q_MAX - 1)));
  const maxAbs = Math.max(Math.abs(lo), Math.abs(hi));
  if (maxAbs > 0) k = Math.max(k, ceilLog2(maxAbs / (2 ** 24 - 1 - UV_Q_MAX)));
  return { k, base: Math.floor(lo / 2 ** k) };
}

export const latticeKey = (l: UvLattice): string => (l.wide ? 'wide' : `${l.ku},${l.kv},${l.baseU},${l.baseV}`);

// ---------------------------------------------------------------------------------------------------------------
// COLOR_0

const COLOR_EXACT8 = (c: number) => {
  const q = Math.round(c * 255);
  return q >= 0 && q <= 255 && (f32(q / 255) === c || f32(q * C_UNORM8) === c);
};
export const decodeUnorm = (q: number, c: number): number => f32(q * c);

// ---------------------------------------------------------------------------------------------------------------

function referencedVertices(g: SceneGeometry): Uint8Array {
  const used = new Uint8Array(g.positions.length / 3);
  for (let i = 0; i < g.indices.length; i++) used[g.indices[i]] = 1;
  return used;
}

function faceNormal64(p: ArrayLike<number>, i0: number, i1: number, i2: number): [number, number, number, number] {
  const ax = p[i1 * 3] - p[i0 * 3], ay = p[i1 * 3 + 1] - p[i0 * 3 + 1], az = p[i1 * 3 + 2] - p[i0 * 3 + 2];
  const bx = p[i2 * 3] - p[i0 * 3], by = p[i2 * 3 + 1] - p[i0 * 3 + 1], bz = p[i2 * 3 + 2] - p[i0 * 3 + 2];
  const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
  return [nx, ny, nz, Math.hypot(nx, ny, nz)];
}

const angleDeg = (a: ArrayLike<number>, ao: number, b: ArrayLike<number>, bo: number): number => {
  const la = Math.hypot(a[ao], a[ao + 1], a[ao + 2]), lb = Math.hypot(b[bo], b[bo + 1], b[bo + 2]);
  const d = (a[ao] * b[bo] + a[ao + 1] * b[bo + 1] + a[ao + 2] * b[bo + 2]) / (la * lb);
  // atan2 of |a×b| and a·b is accurate near 0 (acos is not)
  const cx = a[ao + 1] * b[bo + 2] - a[ao + 2] * b[bo + 1], cy = a[ao + 2] * b[bo] - a[ao] * b[bo + 2], cz = a[ao] * b[bo + 1] - a[ao + 1] * b[bo];
  return Math.atan2(Math.hypot(cx, cy, cz) / (la * lb), d) * 180 / Math.PI;
};

/**
 * Quantize a scene (see the header). Returns a NEW SceneData (input arrays are not modified) with `quant` set.
 */
export function quantizeScene(scene: SceneData, opts: QuantizeOptions = {}): { scene: SceneData; stats: QuantizeStats } {
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const mode = opts.mode ?? 'quantized';
  const g0 = scene.geometry;
  const nV0 = g0.positions.length / 3;
  const stats: QuantizeStats = {
    mode, posLog2: 0, step: 0, maxPosErrAxis: 0, maxPosErr: 0, droppedDegenerate: 0, flatTriangles: 0, maxNormalErrDeg: 0,
    maxTangentErrDeg: 0, uv: [], wideMaterials: [], uvDuplicatedVertices: 0, colorFormat: g0.color0 ? 'rgba16' : 'none',
    verticesIn: nV0, verticesOut: nV0, ms: 0,
  };
  const done = (s: SceneData) => { stats.ms = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0; return { scene: s, stats }; };
  const lossless = (why?: string): { scene: SceneData; stats: QuantizeStats } => {
    stats.mode = 'lossless';
    return done({ ...scene, warnings: why ? [...scene.warnings, why] : scene.warnings, quant: { mode: 'lossless', posLog2: 0, uv: [], uvTolerance: 0, normal: 'f32', tangent: 'f32', color: g0.color0 ? 'f32' : 'none' } });
  };
  if (mode === 'lossless') return lossless();
  const prior = scene.quant?.mode === 'quantized' ? scene.quant : undefined;
  const tau = prior?.uvTolerance ?? opts.uvTolerance ?? DEFAULT_UV_TOLERANCE;

  // ---- 0: TRI_FLAT detection on the input ----
  const nT0 = g0.indices.length / 3;
  let triFlags = g0.triFlags.slice();
  if (!prior) {
    for (let t = 0; t < nT0; t++) {
      const i0 = g0.indices[3 * t], i1 = g0.indices[3 * t + 1], i2 = g0.indices[3 * t + 2];
      const [nx, ny, nz, l] = faceNormal64(g0.positions, i0, i1, i2);
      if (!(l > 0) || !Number.isFinite(l)) continue;
      let flat = true;
      for (const v of [i0, i1, i2]) {
        const mx = g0.normals[3 * v], my = g0.normals[3 * v + 1], mz = g0.normals[3 * v + 2];
        const lm = Math.hypot(mx, my, mz);
        if (!(lm > 0) || 1 - (mx * nx + my * ny + mz * nz) / (lm * l) > FLAT_COS_TOL) { flat = false; break; }
      }
      if (flat) triFlags[t] |= TRI_FLAT;
    }
  }

  // ---- 1: positions ----
  const used0 = referencedVertices(g0);
  let ext = 0, maxAbs = 0;
  {
    const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (let v = 0; v < nV0; v++) {
      if (!used0[v]) continue;
      for (let k = 0; k < 3; k++) {
        const x = g0.positions[3 * v + k];
        if (!Number.isFinite(x)) continue;
        if (x < mn[k]) mn[k] = x;
        if (x > mx[k]) mx[k] = x;
        maxAbs = Math.max(maxAbs, Math.abs(x));
      }
    }
    for (let k = 0; k < 3; k++) if (mx[k] >= mn[k]) ext = Math.max(ext, mx[k] - mn[k]);
  }
  let posLog2: number;
  if (prior) posLog2 = prior.posLog2;
  else {
    const k = choosePosLog2(ext, maxAbs, opts.maxPosLog2);
    if (k === null) return lossless(`scene extent ${ext.toFixed(1)} m exceeds the quantization floor (step > 2^${opts.maxPosLog2 ?? DEFAULT_MAX_POS_LOG2} m): lossless f32 geometry`);
    posLog2 = k;
  }
  const s = 2 ** posLog2;
  stats.posLog2 = posLog2; stats.step = s;
  let positions: Float32Array = new Float32Array(g0.positions.length);
  for (let v = 0; v < nV0; v++) {
    let e2 = 0;
    for (let k = 0; k < 3; k++) {
      const x = g0.positions[3 * v + k];
      const q = snapToLattice(x, s);
      positions[3 * v + k] = q;
      if (used0[v] && Number.isFinite(x)) { const d = Math.abs(q - x); stats.maxPosErrAxis = Math.max(stats.maxPosErrAxis, d); e2 += d * d; }
    }
    if (used0[v]) stats.maxPosErr = Math.max(stats.maxPosErr, Math.sqrt(e2));
  }

  // ---- 2: drop degenerate-after-snap triangles (dense primIds) ----
  let indices = g0.indices.slice(), triMaterial = g0.triMaterial.slice();
  {
    let out = 0;
    for (let t = 0; t < nT0; t++) {
      const i0 = indices[3 * t], i1 = indices[3 * t + 1], i2 = indices[3 * t + 2];
      const [, , , l] = faceNormal64(positions, i0, i1, i2);
      if (!(l > 0) || !Number.isFinite(l)) { stats.droppedDegenerate++; continue; }
      indices[3 * out] = i0; indices[3 * out + 1] = i1; indices[3 * out + 2] = i2;
      triMaterial[out] = triMaterial[t]; triFlags[out] = triFlags[t];
      out++;
    }
    if (out < nT0) { indices = indices.slice(0, 3 * out); triMaterial = triMaterial.slice(0, out); triFlags = triFlags.slice(0, out); }
  }
  const nT = indices.length / 3;

  // ---- 3 + 4: normals (flat vertices: face normal of the snapped triangle), oct 2 × 16 ----
  const normals = new Float32Array(g0.normals.length);
  const used = new Uint8Array(nV0);        // 1 = referenced; 2 = referenced by a non-flat triangle
  const flatNormalOf = new Int32Array(nV0).fill(-1);
  for (let t = 0; t < nT; t++) {
    const flat = (triFlags[t] & TRI_FLAT) !== 0;
    if (flat) stats.flatTriangles++;
    for (let c = 0; c < 3; c++) {
      const v = indices[3 * t + c];
      if (!flat) used[v] = 2; else { if (!used[v]) used[v] = 1; if (flatNormalOf[v] < 0) flatNormalOf[v] = t; }
    }
  }
  for (let v = 0; v < nV0; v++) {
    let x = g0.normals[3 * v], y = g0.normals[3 * v + 1], z = g0.normals[3 * v + 2];
    if (used[v] === 1 && !prior) {
      const t = flatNormalOf[v];
      const [nx, ny, nz, l] = faceNormal64(positions, indices[3 * t], indices[3 * t + 1], indices[3 * t + 2]);
      x = nx / l; y = ny / l; z = nz / l;
    }
    if (!used[v] || prior) {
      normals[3 * v] = x; normals[3 * v + 1] = y; normals[3 * v + 2] = z;
      if (used[v]) octCodeExact(normals, 3 * v, 16); // already quantized: must be on the lattice (idempotency)
      continue;
    }
    if (!(Math.hypot(x, y, z) > 0) || !Number.isFinite(x + y + z)) { x = 0; y = 1; z = 0; }
    octSnap(x, y, z, 16, normals, 3 * v);
    if (used[v] === 2) stats.maxNormalErrDeg = Math.max(stats.maxNormalErrDeg, angleDeg(g0.normals, 3 * v, normals, 3 * v));
  }

  // ---- 5: UV lattices per material (+ vertex duplication across lattice groups) ----
  const nMat = scene.materials.length;
  let lattices: UvLattice[];
  if (prior) {
    if (prior.uv.length !== nMat) throw new Error(`quant.uv has ${prior.uv.length} lattices for ${nMat} materials`);
    lattices = prior.uv.map((l) => ({ ...l }));
  } else {
    const lo = new Float64Array(nMat * 2).fill(Infinity), hi = new Float64Array(nMat * 2).fill(-Infinity);
    const nonFinite = new Uint8Array(nMat);
    for (let t = 0; t < nT; t++) {
      const m = triMaterial[t];
      for (let c = 0; c < 3; c++) {
        const v = indices[3 * t + c];
        for (let a = 0; a < 2; a++) {
          const x = g0.uv0[2 * v + a] ?? 0;
          if (!Number.isFinite(x)) { nonFinite[m] = 1; continue; }
          if (x < lo[2 * m + a]) lo[2 * m + a] = x;
          if (x > hi[2 * m + a]) hi[2 * m + a] = x;
        }
      }
    }
    lattices = scene.materials.map((mat, m) => {
      const lu = hi[2 * m] >= lo[2 * m] ? uvAxisLattice(lo[2 * m], hi[2 * m]) : { k: MIN_LOG2, base: 0 };
      const lv = hi[2 * m + 1] >= lo[2 * m + 1] ? uvAxisLattice(lo[2 * m + 1], hi[2 * m + 1]) : { k: MIN_LOG2, base: 0 };
      const worst = uvWorstTexel(materialTextureRefs(mat), scene.textures, 2 ** lu.k, 2 ** lv.k);
      const wide = nonFinite[m] === 1 || worst > tau;
      return { ku: lu.k, kv: lv.k, baseU: lu.base, baseV: lv.base, wide };
    });
  }
  // group per vertex; duplicate on conflict
  const keyOf = lattices.map(latticeKey);
  const groupIds = new Map<string, number>();
  const matGroup = keyOf.map((k) => { let id = groupIds.get(k); if (id === undefined) { id = groupIds.size; groupIds.set(k, id); } return id; });
  const groupLattice: UvLattice[] = [];
  lattices.forEach((l, m) => { groupLattice[matGroup[m]] ??= l; });
  const vGroup = new Int32Array(nV0).fill(-1);
  const dupOf = new Map<string, number>(); // `${v}|${group}` → duplicate vertex index
  const extra: number[] = []; // source vertex of each duplicate
  for (let t = 0; t < nT; t++) {
    const gid = matGroup[triMaterial[t]];
    for (let c = 0; c < 3; c++) {
      const v = indices[3 * t + c];
      if (vGroup[v] < 0) { vGroup[v] = gid; continue; }
      if (vGroup[v] === gid) continue;
      const key = `${v}|${gid}`;
      let d = dupOf.get(key);
      if (d === undefined) { d = nV0 + extra.length; extra.push(v); dupOf.set(key, d); }
      indices[3 * t + c] = d;
    }
  }
  stats.uvDuplicatedVertices = extra.length;
  const nV = nV0 + extra.length;
  const src = (v: number) => (v < nV0 ? v : extra[v - nV0]);
  const grow = (a: Float32Array, n: number): Float32Array => {
    if (!extra.length) return a;
    const o = new Float32Array(nV * n);
    o.set(a);
    extra.forEach((v, i) => o.set(a.subarray(v * n, v * n + n), (nV0 + i) * n));
    return o;
  };
  positions = grow(positions, 3);
  const normalsG = grow(normals, 3);
  const vGroupG = new Int32Array(nV);
  vGroupG.set(vGroup);
  for (const [key, d] of dupOf) vGroupG[d] = Number(key.slice(key.indexOf('|') + 1));
  const uv0 = new Float32Array(nV * 2);
  const matWorst = new Float64Array(nMat), matUvErr = new Float64Array(nMat);
  for (let v = 0; v < nV; v++) {
    const sv = src(v);
    const u = g0.uv0[2 * sv] ?? 0, w = g0.uv0[2 * sv + 1] ?? 0;
    const gid = vGroupG[v];
    const L = gid >= 0 ? groupLattice[gid] : undefined;
    if (!L || L.wide) { uv0[2 * v] = u; uv0[2 * v + 1] = w; continue; }
    const su = 2 ** L.ku, sw = 2 ** L.kv;
    const qu = Math.round(u / su) - L.baseU, qv = Math.round(w / sw) - L.baseV;
    if (qu < 0 || qu > UV_Q_MAX || qv < 0 || qv > UV_Q_MAX) throw new Error(`uv (${u}, ${w}) outside its lattice (q = ${qu}, ${qv})`);
    uv0[2 * v] = (qu + L.baseU) * su;
    uv0[2 * v + 1] = (qv + L.baseV) * sw;
  }
  lattices.forEach((L, m) => {
    const worst = L.wide ? 0 : uvWorstTexel(materialTextureRefs(scene.materials[m]), scene.textures, 2 ** L.ku, 2 ** L.kv);
    matWorst[m] = worst;
    if (L.wide) stats.wideMaterials.push(m);
  });
  for (let t = 0; t < nT; t++) {
    const m = triMaterial[t];
    for (let c = 0; c < 3; c++) {
      const v = indices[3 * t + c], sv = src(v);
      matUvErr[m] = Math.max(matUvErr[m], Math.abs(uv0[2 * v] - (g0.uv0[2 * sv] ?? 0)), Math.abs(uv0[2 * v + 1] - (g0.uv0[2 * sv + 1] ?? 0)));
    }
  }
  stats.uv = lattices.map((L, m) => ({ material: m, ku: L.ku, kv: L.kv, wide: L.wide, worstTexel: matWorst[m], maxUvErr: matUvErr[m] }));

  // ---- 6: COLOR_0 ----
  let color0: Float32Array | undefined;
  let colorFormat: SceneQuant['color'] = 'none';
  if (g0.color0) {
    const c0 = grow(g0.color0, 4);
    const refd = new Uint8Array(nV);
    for (let i = 0; i < indices.length; i++) refd[indices[i]] = 1;
    let all8 = prior ? prior.color === 'rgba8' : true;
    let clamped = 0;
    if (!prior) {
      for (let v = 0; v < nV && all8; v++) if (refd[v]) for (let c = 0; c < 4; c++) if (!COLOR_EXACT8(c0[4 * v + c])) { all8 = false; break; }
    }
    colorFormat = all8 ? 'rgba8' : 'rgba16';
    const m = all8 ? 255 : 65535, cc = all8 ? C_UNORM8 : C_UNORM16;
    color0 = new Float32Array(nV * 4);
    for (let i = 0; i < color0.length; i++) {
      let x = c0[i];
      if (!(x >= 0)) { if (x !== 0) clamped++; x = 0; }
      if (x > 1) { clamped++; x = 1; }
      color0[i] = decodeUnorm(Math.round(x * m), cc);
    }
    if (clamped) scene = { ...scene, warnings: [...scene.warnings, `COLOR_0: ${clamped} component(s) outside [0, 1] clamped (unorm${all8 ? 8 : 16} vertex colour)`] };
  }
  stats.colorFormat = colorFormat;

  // ---- 7: tangents (MikkTSpace on the snapped data, then oct 2 × 15 + sign) ----
  let tangents = grow(g0.tangents.length === nV0 * 4 ? g0.tangents : new Float32Array(nV0 * 4), 4);
  if (opts.generateTangents && opts.tangentRanges?.length) {
    tangents = new Float32Array(nV * 4);
    const w: string[] = [];
    generateRangeTangents({ positions, normals: normalsG, uv0, tangents }, opts.tangentRanges, opts.generateTangents, w);
    if (w.length) scene = { ...scene, warnings: [...scene.warnings, ...w] };
  }
  {
    const snapped = new Float32Array(3);
    for (let v = 0; v < nV; v++) {
      const x = tangents[4 * v], y = tangents[4 * v + 1], z = tangents[4 * v + 2], w = tangents[4 * v + 3];
      if (w === 0 || !(Math.hypot(x, y, z) > 0) || !Number.isFinite(x + y + z)) { tangents.fill(0, 4 * v, 4 * v + 4); continue; }
      if (prior && !opts.generateTangents) { octCodeExact(tangents, 4 * v, 15); continue; }
      octSnap(x, y, z, 15, snapped, 0);
      stats.maxTangentErrDeg = Math.max(stats.maxTangentErrDeg, angleDeg(tangents, 4 * v, snapped, 0));
      tangents[4 * v] = snapped[0]; tangents[4 * v + 1] = snapped[1]; tangents[4 * v + 2] = snapped[2]; tangents[4 * v + 3] = w < 0 ? -1 : 1;
    }
  }

  // ---- 8: re-weld ----
  const welded = reweld({ positions, normals: normalsG, tangents, uv0, color0, indices, triMaterial, triFlags }, vGroupG);
  stats.verticesOut = welded.positions.length / 3;

  const quant: SceneQuant = { mode: 'quantized', posLog2, uv: lattices, uvTolerance: tau, normal: 'oct16', tangent: 'oct15', color: colorFormat };
  const warnings = [...scene.warnings];
  if (stats.droppedDegenerate) warnings.push(`quantize: dropped ${stats.droppedDegenerate} triangle(s) that became degenerate on the 2^${posLog2} m lattice; primIds are dense over the kept triangles`);
  const out: SceneData = { ...scene, geometry: welded, bounds: boundsOf(welded), warnings, quant };
  if (!welded.color0) delete out.geometry.color0;
  return done(out);
}

/** MikkTSpace (glTF convention: w negated, like gltf-transform `tangents()`) on unwelded vertex ranges, written into
 *  g.tangents. Used by quantizeScene (on the snapped data) and by the lossless glTF path. */
export function generateRangeTangents(g: Pick<SceneGeometry, 'positions' | 'normals' | 'uv0' | 'tangents'>,
  ranges: readonly { vBase: number; vCount: number }[], gen: TangentGenerator, warnings: string[]): number {
  let ok = 0;
  for (const r of ranges) {
    const sub = (a: Float32Array, n: number) => a.subarray(r.vBase * n, (r.vBase + r.vCount) * n);
    try {
      const tan = gen(sub(g.positions, 3), sub(g.normals, 3), sub(g.uv0, 2));
      if (tan.length !== r.vCount * 4) throw new Error(`MikkTSpace returned ${tan.length} floats, expected ${r.vCount * 4}`);
      for (let i = 3; i < tan.length; i += 4) tan[i] = -tan[i]; // glTF uv convention (matches gltf-transform)
      g.tangents.set(tan, r.vBase * 4);
      ok++;
    } catch (e) {
      warnings.push(`MikkTSpace failed on a primitive (${e instanceof Error ? e.message : String(e)}); tangents left zero`);
    }
  }
  return ok;
}

/** Merge vertices whose (position, normal, uv, tangent, colour, lattice group) words are identical (lossless).
 *  Keeps first-occurrence order over the ORIGINAL vertex order; unreferenced vertices are removed. */
export function reweld(g: SceneGeometry, group?: Int32Array): SceneGeometry {
  const nV = g.positions.length / 3;
  const W = 3 + 3 + 2 + 4 + (g.color0 ? 4 : 0) + 1;
  const pu = new Uint32Array(g.positions.buffer, g.positions.byteOffset, g.positions.length);
  const nu = new Uint32Array(g.normals.buffer, g.normals.byteOffset, g.normals.length);
  const uu = new Uint32Array(g.uv0.buffer, g.uv0.byteOffset, g.uv0.length);
  const tu = new Uint32Array(g.tangents.buffer, g.tangents.byteOffset, g.tangents.length);
  const cu = g.color0 ? new Uint32Array(g.color0.buffer, g.color0.byteOffset, g.color0.length) : undefined;
  const keyW = new Uint32Array(W);
  const words = (v: number, out: Uint32Array) => {
    let o = 0;
    for (let k = 0; k < 3; k++) out[o++] = pu[3 * v + k];
    for (let k = 0; k < 3; k++) out[o++] = nu[3 * v + k];
    out[o++] = uu[2 * v] ?? 0; out[o++] = uu[2 * v + 1] ?? 0;
    for (let k = 0; k < 4; k++) out[o++] = tu[4 * v + k] ?? 0;
    if (cu) for (let k = 0; k < 4; k++) out[o++] = cu[4 * v + k];
    out[o++] = group ? group[v] >>> 0 : 0;
  };
  const used = new Uint8Array(nV);
  for (let i = 0; i < g.indices.length; i++) used[g.indices[i]] = 1;
  let cap = 16;
  while (cap < 2 * nV) cap *= 2;
  const table = new Int32Array(cap).fill(-1);  // → new vertex id
  const reps: number[] = [];                   // new id → representative old vertex
  const repWords: Uint32Array[] = [];
  const remap = new Int32Array(nV).fill(-1);
  for (let v = 0; v < nV; v++) {
    if (!used[v]) continue;
    words(v, keyW);
    let h = 0x811c9dc5;
    for (let k = 0; k < W; k++) { h ^= keyW[k]; h = Math.imul(h, 0x01000193); h ^= h >>> 15; }
    let slot = (h >>> 0) & (cap - 1);
    for (;;) {
      const id = table[slot];
      if (id < 0) {
        const nid = reps.length;
        table[slot] = nid; reps.push(v); repWords.push(keyW.slice());
        remap[v] = nid;
        break;
      }
      const rw = repWords[id];
      let eq = true;
      for (let k = 0; k < W; k++) if (rw[k] !== keyW[k]) { eq = false; break; }
      if (eq) { remap[v] = id; break; }
      slot = (slot + 1) & (cap - 1);
    }
  }
  const n = reps.length;
  const pick = (a: Float32Array, c: number) => { const o = new Float32Array(n * c); reps.forEach((v, i) => { for (let k = 0; k < c; k++) o[i * c + k] = a[v * c + k]; }); return o; };
  const indices = new Uint32Array(g.indices.length);
  for (let i = 0; i < indices.length; i++) indices[i] = remap[g.indices[i]];
  const out: SceneGeometry = {
    positions: pick(g.positions, 3), normals: pick(g.normals, 3), tangents: pick(g.tangents, 4), uv0: pick(g.uv0, 2),
    indices, triMaterial: g.triMaterial, triFlags: g.triFlags,
  };
  if (g.color0) out.color0 = pick(g.color0, 4);
  return out;
}

export function boundsOf(g: SceneGeometry): SceneData['bounds'] {
  const mn: [number, number, number] = [Infinity, Infinity, Infinity], mx: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < g.indices.length; i++) {
    const v = g.indices[i];
    for (let k = 0; k < 3; k++) { const x = g.positions[3 * v + k]; if (x < mn[k]) mn[k] = x; if (x > mx[k]) mx[k] = x; }
  }
  return g.indices.length ? { min: mn, max: mx } : { min: [0, 0, 0], max: [0, 0, 0] };
}

// ---------------------------------------------------------------------------------------------------------------
// Verification (package reader, GPU packer): every value must sit exactly on its stored lattice.

export class QuantizationError extends Error {}

/** Per-vertex lattice group from the triangles' materials; throws if a vertex is used by two different lattices. */
export function vertexLatticeGroups(g: SceneGeometry, lattices: readonly UvLattice[]): Int32Array {
  const keys = lattices.map(latticeKey);
  const out = new Int32Array(g.positions.length / 3).fill(-1);
  for (let t = 0; t < g.triMaterial.length; t++) {
    const m = g.triMaterial[t];
    for (let c = 0; c < 3; c++) {
      const v = g.indices[3 * t + c];
      if (out[v] < 0) out[v] = m;
      else if (out[v] !== m && keys[out[v]] !== keys[m]) throw new QuantizationError(`vertex ${v} is used by materials ${out[v]} and ${m} with different UV lattices`);
    }
  }
  return out;
}

/** Throws QuantizationError unless every referenced value is on the lattices of `q` (package v2 reader, U-Q tests). */
export function assertQuantized(g: SceneGeometry, q: SceneQuant, what = 'scene'): void {
  if (q.mode !== 'quantized') return;
  const fail = (m: string): never => { throw new QuantizationError(`${what}: ${m}`); };
  const s = 2 ** q.posLog2;
  const nV = g.positions.length / 3;
  const used = new Uint8Array(nV);
  for (let i = 0; i < g.indices.length; i++) used[g.indices[i]] = 1;
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  const nrm = new Float32Array(3);
  for (let v = 0; v < nV; v++) {
    if (!used[v]) continue;
    for (let k = 0; k < 3; k++) {
      const x = g.positions[3 * v + k];
      const n = x / s;
      if (!Number.isInteger(n) || Math.abs(n) > 2 ** 24 - 1) fail(`position ${v}.${k} = ${x} is not on the 2^${q.posLog2} lattice`);
      if (n < mn[k]) mn[k] = n;
      if (n > mx[k]) mx[k] = n;
    }
    nrm.set(g.normals.subarray(3 * v, 3 * v + 3));
    try { octCodeExact(nrm, 0, 16); } catch (e) { fail(`normal ${v}: ${(e as Error).message}`); }
    const tw = g.tangents[4 * v + 3];
    if (tw !== 0 && tw !== undefined) {
      if (tw !== 1 && tw !== -1) fail(`tangent ${v}: w = ${tw}`);
      try { octCodeExact(g.tangents, 4 * v, 15); } catch (e) { fail(`tangent ${v}: ${(e as Error).message}`); }
    }
  }
  for (let k = 0; k < 3; k++) if (mx[k] - mn[k] > POS_MAX_OFFSET) fail(`axis ${k} spans ${mx[k] - mn[k]} steps > 2^21 − 1`);
  const grp = vertexLatticeGroups(g, q.uv);
  for (let v = 0; v < nV; v++) {
    const m = grp[v];
    if (m < 0) continue;
    const L = q.uv[m];
    if (L.wide) continue;
    const qu = g.uv0[2 * v] / 2 ** L.ku - L.baseU, qv = g.uv0[2 * v + 1] / 2 ** L.kv - L.baseV;
    if (!Number.isInteger(qu) || !Number.isInteger(qv) || qu < 0 || qv < 0 || qu > UV_Q_MAX || qv > UV_Q_MAX) {
      fail(`uv ${v} = (${g.uv0[2 * v]}, ${g.uv0[2 * v + 1]}) is not on material ${m}'s lattice`);
    }
  }
  if (g.color0) {
    if (q.color === 'none' || q.color === 'f32') fail(`COLOR_0 present but quant.color = ${q.color}`);
    const m = q.color === 'rgba8' ? 255 : 65535, c = q.color === 'rgba8' ? C_UNORM8 : C_UNORM16;
    for (let v = 0; v < nV; v++) {
      if (!used[v]) continue;
      for (let k = 0; k < 4; k++) {
        const x = g.color0[4 * v + k];
        if (decodeUnorm(Math.round(x * m), c) !== x) fail(`COLOR_0 ${v}.${k} = ${x} is not ${q.color}`);
      }
    }
  }
}
