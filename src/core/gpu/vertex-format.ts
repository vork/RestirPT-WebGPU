// GPU vertex arena (docs/decisions/data-formats.md §B1–§B6, §D P1). WGSL mirror: shaders/scene/scene-data.wgsl.
// A LOSSLESS recoding of a quantized SceneGeometry (scene/quantize.ts): the packer decodes every record with the
// TS mirror below and throws unless positions, UVs and colours come back bit-exactly and normals are exact oct codes.
//
// One storage buffer `sceneVerts: array<vec4u>` (the scene group stays at 5 storage buffers, Metal ≤ 9 per pass):
//   vec4 0   header A: posBase.xyz (i32 = n_min − n_O), posScale (f32 bits, 2^k)
//   vec4 1   header B: vertexCount, uvWideWord (word offset of the wide-UV section), colorWord (word offset of the
//            colour section), colorFormat (0 none = white, 1 rgba8, 2 rgba16)
//   VERTEX_FORMAT 1 (Q, 16 B per vertex) — vec4 2 + i:
//     x, y  P21 position: x | y << 21 (low 11 bits of y), y >> 11 | z << 10 (21-bit offsets n − n_min)
//     z     oct 2 × 16 snorm shading normal (x in bits 0..15, y in 16..31); ignored on TRI_FLAT triangles
//     w     UV: 2 × 16 bit lattice code (q_u | q_v << 16) of the vertex's material lattice, or the wide-UV index
//   VERTEX_FORMAT 0 (F32 / lossless, 48 B per vertex) — vec4 2 + 3i .. 2 + 3i + 2 (f32 bits):
//     [p.xyz recentred, uv.x] [n.xyz, uv.y] [COLOR_0 rgba (1 when absent)]
//   wide-UV section (Q): 2 words (u, v f32 bits) per wide vertex
//   colour section (Q): rgba8 1 word / rgba16 2 words per vertex; decode f32(q) · f32(1/(2^b − 1)) (exact product)
// Decoding positions / UVs is dyadic (integer × 2^k): exact on CPU and GPU even under fast-math reassociation or FMA.
import {
  C_OCT16, C_UNORM16, C_UNORM8, POS_MAX_OFFSET, QuantizationError, UV_Q_MAX, octCodeExact, octDecode, vertexLatticeGroups,
} from '../scene/quantize.ts';
import type { SceneGeometry, SceneQuant, UvLattice } from '../scene/types.ts';

export const VERTEX_FORMAT_F32 = 0;
export const VERTEX_FORMAT_Q = 1;
export type VertexFormat = typeof VERTEX_FORMAT_F32 | typeof VERTEX_FORMAT_Q;
export const VQ_HEADER_VEC4S = 2;
export const VERTEX_BYTES_Q = 16;
export const VERTEX_BYTES_F32 = 48;
export const COLOR_NONE = 0;
export const COLOR_RGBA8 = 1;
export const COLOR_RGBA16 = 2;

const f32 = Math.fround;
const F = new Float32Array(1), U = new Uint32Array(F.buffer);
export const f32bits = (x: number): number => { F[0] = x; return U[0]; };
export const bitsF32 = (u: number): number => { U[0] = u >>> 0; return F[0]; };
/** 2^k as f32 bits (k ∈ [−126, 127]); WGSL: bitcast<f32>(u32(k + 127) << 23). */
export const pow2Bits = (k: number): number => {
  if (!Number.isInteger(k) || k < -126 || k > 127) throw new QuantizationError(`2^${k} is not a normal f32`);
  return ((k + 127) << 23) >>> 0;
};

export interface VertexArena {
  words: Uint32Array;
  format: VertexFormat;
  vertexCount: number;
  wideCount: number;
  colorFormat: number;
  /** Bytes per section (reporting). */
  bytes: { header: number; records: number; wideUv: number; color: number; total: number };
}

/** Per-material UV lattice words for MaterialGpu (flags bits 16..31 = ku + 128, kv + 128; MAT_UV_WIDE; uvBase). */
export const MAT_UV_WIDE = 32;
export function materialUvWords(l: UvLattice | undefined): { flagBits: number; baseU: number; baseV: number } {
  if (!l) return { flagBits: 0, baseU: 0, baseV: 0 };
  if (l.wide) return { flagBits: MAT_UV_WIDE, baseU: 0, baseV: 0 };
  if (l.ku < -128 || l.ku > 127 || l.kv < -128 || l.kv > 127) throw new QuantizationError(`uv lattice exponent out of i8 range (${l.ku}, ${l.kv})`);
  return { flagBits: (((l.ku + 128) << 16) | ((l.kv + 128) << 24)) >>> 0, baseU: l.baseU | 0, baseV: l.baseV | 0 };
}

/** Is `origin` a lattice point of the scene (n_O integral)? The Q format requires it (computeRenderOrigin snaps it). */
export function originOnLattice(origin: readonly number[], q: SceneQuant): boolean {
  const s = 2 ** q.posLog2;
  return origin.every((o) => Number.isInteger(o / s) && Math.abs(o / s) < 2 ** 52);
}

/**
 * Pack the vertex arena. `recentred` = f32(p − origin) (the BVH / light frame). Quantized scenes → Q format; lossless or
 * unquantized scenes → F32. Throws QuantizationError unless every Q record decodes (TS mirror) to the exact bits.
 */
export function packVertexArena(g: SceneGeometry, recentred: Float32Array, quant: SceneQuant | undefined, origin: readonly number[]): VertexArena {
  const nV = g.positions.length / 3;
  if (!quant || quant.mode !== 'quantized') return packF32(g, recentred);
  const fail = (m: string): never => { throw new QuantizationError(`vertex packer: ${m}`); };
  if (!originOnLattice(origin, quant)) fail(`render origin (${origin.join(', ')}) is not on the 2^${quant.posLog2} lattice (use computeRenderOrigin(bounds, quant))`);
  const s = 2 ** quant.posLog2;
  const nO = origin.map((o) => o / s);
  // lattice ints and their minimum (over every vertex: unreferenced vertices must decode too)
  const n = new Float64Array(nV * 3);
  const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nV * 3; i++) {
    const v = g.positions[i] / s;
    if (!Number.isInteger(v)) fail(`position ${i / 3 | 0}.${i % 3} = ${g.positions[i]} is off the 2^${quant.posLog2} lattice`);
    n[i] = v;
    const k = i % 3;
    if (v < mn[k]) mn[k] = v;
    if (v > mx[k]) mx[k] = v;
  }
  if (nV === 0) { mn.fill(0); mx.fill(0); }
  for (let k = 0; k < 3; k++) if (mx[k] - mn[k] > POS_MAX_OFFSET) fail(`axis ${k} spans ${mx[k] - mn[k]} lattice steps (> 2^21 − 1)`);
  const posBase = [0, 1, 2].map((k) => mn[k] - nO[k]);
  if (posBase.some((b) => Math.abs(b) > 2 ** 23)) fail(`posBase ${posBase} out of range`);
  const groups = vertexLatticeGroups(g, quant.uv);
  const wide: number[] = [];
  const wideIndex = new Int32Array(nV).fill(-1);
  for (let v = 0; v < nV; v++) if (groups[v] >= 0 && quant.uv[groups[v]].wide) { wideIndex[v] = wide.length; wide.push(v); }
  const colorFormat = !g.color0 ? COLOR_NONE : quant.color === 'rgba8' ? COLOR_RGBA8 : quant.color === 'rgba16' ? COLOR_RGBA16 : fail(`COLOR_0 with quant.color '${quant.color}'`);
  const recWords = (VQ_HEADER_VEC4S + nV) * 4;
  const wideWord = recWords;
  const colorWord = wideWord + 2 * wide.length;
  const colorWords = colorFormat === COLOR_NONE ? 0 : colorFormat === COLOR_RGBA8 ? nV : 2 * nV;
  const total = Math.ceil((colorWord + colorWords) / 4) * 4;
  const words = new Uint32Array(Math.max(total, (VQ_HEADER_VEC4S + 1) * 4));
  words[0] = posBase[0] >>> 0; words[1] = posBase[1] >>> 0; words[2] = posBase[2] >>> 0; words[3] = pow2Bits(quant.posLog2);
  words[4] = nV; words[5] = wideWord; words[6] = colorWord; words[7] = colorFormat;
  for (let v = 0; v < nV; v++) {
    const o = (VQ_HEADER_VEC4S + v) * 4;
    const x = n[3 * v] - mn[0], y = n[3 * v + 1] - mn[1], z = n[3 * v + 2] - mn[2];
    words[o] = (x | ((y & 0x7ff) << 21)) >>> 0;
    words[o + 1] = ((y >>> 11) | (z << 10)) >>> 0;
    let q: [number, number];
    try { q = octCodeExact(g.normals, 3 * v, 16); } catch (e) { return fail(`normal ${v}: ${(e as Error).message}`); }
    words[o + 2] = ((q[0] & 0xffff) | ((q[1] & 0xffff) << 16)) >>> 0;
    const m = groups[v];
    if (m < 0) words[o + 3] = 0;
    else if (wideIndex[v] >= 0) words[o + 3] = wideIndex[v];
    else {
      const L = quant.uv[m];
      const qu = g.uv0[2 * v] / 2 ** L.ku - L.baseU, qv = g.uv0[2 * v + 1] / 2 ** L.kv - L.baseV;
      if (!Number.isInteger(qu) || !Number.isInteger(qv) || qu < 0 || qv < 0 || qu > UV_Q_MAX || qv > UV_Q_MAX) {
        fail(`uv ${v} = (${g.uv0[2 * v]}, ${g.uv0[2 * v + 1]}) is off material ${m}'s lattice`);
      }
      words[o + 3] = (qu | (qv << 16)) >>> 0;
    }
  }
  wide.forEach((v, i) => { words[wideWord + 2 * i] = f32bits(g.uv0[2 * v]); words[wideWord + 2 * i + 1] = f32bits(g.uv0[2 * v + 1]); });
  if (g.color0 && colorFormat !== COLOR_NONE) {
    const m = colorFormat === COLOR_RGBA8 ? 255 : 65535;
    for (let v = 0; v < nV; v++) {
      const q = [0, 1, 2, 3].map((c) => Math.round(g.color0![4 * v + c] * m));
      if (colorFormat === COLOR_RGBA8) words[colorWord + v] = (q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24)) >>> 0;
      else { words[colorWord + 2 * v] = (q[0] | (q[1] << 16)) >>> 0; words[colorWord + 2 * v + 1] = (q[2] | (q[3] << 16)) >>> 0; }
    }
  }
  const arena: VertexArena = {
    words, format: VERTEX_FORMAT_Q, vertexCount: nV, wideCount: wide.length, colorFormat,
    bytes: { header: 32, records: nV * VERTEX_BYTES_Q, wideUv: 8 * wide.length, color: 4 * colorWords, total: words.byteLength },
  };
  verifyArena(arena, g, recentred, quant, groups);
  return arena;
}

function packF32(g: SceneGeometry, recentred: Float32Array): VertexArena {
  const nV = recentred.length / 3;
  const words = new Uint32Array(Math.max(VQ_HEADER_VEC4S + 3 * nV, VQ_HEADER_VEC4S + 3) * 4);
  const f = new Float32Array(words.buffer);
  words[3] = pow2Bits(0);
  words[4] = nV;
  for (let i = 0; i < nV; i++) {
    const o = (VQ_HEADER_VEC4S + 3 * i) * 4;
    f[o] = recentred[3 * i]; f[o + 1] = recentred[3 * i + 1]; f[o + 2] = recentred[3 * i + 2]; f[o + 3] = g.uv0[2 * i] ?? 0;
    f[o + 4] = g.normals[3 * i]; f[o + 5] = g.normals[3 * i + 1]; f[o + 6] = g.normals[3 * i + 2]; f[o + 7] = g.uv0[2 * i + 1] ?? 0;
    for (let c = 0; c < 4; c++) f[o + 8 + c] = g.color0 ? g.color0[4 * i + c] : 1;
  }
  return {
    words, format: VERTEX_FORMAT_F32, vertexCount: nV, wideCount: 0, colorFormat: COLOR_NONE,
    bytes: { header: 32, records: nV * VERTEX_BYTES_F32, wideUv: 0, color: 0, total: words.byteLength },
  };
}

// ---------------------------------------------------------------------------------------------------------------
// TS decode mirror (= scene-data.wgsl vq_*). Positions are in the render (recentred) frame.

export interface DecodedVertex { p: [number, number, number]; n: [number, number, number]; uv: [number, number]; color: [number, number, number, number] }

const sext16 = (x: number) => (x << 16) >> 16;

export function decodeVertex(a: VertexArena, i: number, lattice: UvLattice | undefined): DecodedVertex {
  const w = a.words;
  if (a.format === VERTEX_FORMAT_F32) {
    const o = (VQ_HEADER_VEC4S + 3 * i) * 4, b = (k: number) => bitsF32(w[o + k]);
    return { p: [b(0), b(1), b(2)], n: [b(4), b(5), b(6)], uv: [b(3), b(7)], color: [b(8), b(9), b(10), b(11)] };
  }
  const o = (VQ_HEADER_VEC4S + i) * 4;
  const scale = bitsF32(w[3]);
  const base = [w[0] | 0, w[1] | 0, w[2] | 0];
  const x = w[o] & 0x1fffff, y = (w[o] >>> 21) | ((w[o + 1] & 0x3ff) << 11), z = (w[o + 1] >>> 10) & 0x1fffff;
  const p: [number, number, number] = [f32(f32(x + base[0]) * scale), f32(f32(y + base[1]) * scale), f32(f32(z + base[2]) * scale)];
  const nn = octDecode(sext16(w[o + 2] & 0xffff), sext16(w[o + 2] >>> 16), C_OCT16) as number[];
  let uv: [number, number] = [0, 0];
  if (lattice?.wide) uv = [bitsF32(w[w[5] + 2 * w[o + 3]]), bitsF32(w[w[5] + 2 * w[o + 3] + 1])];
  else if (lattice) uv = [f32(f32((w[o + 3] & 0xffff) + lattice.baseU) * f32(2 ** lattice.ku)), f32(f32((w[o + 3] >>> 16) + lattice.baseV) * f32(2 ** lattice.kv))];
  let color: [number, number, number, number] = [1, 1, 1, 1];
  if (w[7] === COLOR_RGBA8) { const c = w[w[6] + i]; color = [c & 0xff, (c >>> 8) & 0xff, (c >>> 16) & 0xff, c >>> 24].map((q) => f32(q * C_UNORM8)) as typeof color; }
  else if (w[7] === COLOR_RGBA16) {
    const c0 = w[w[6] + 2 * i], c1 = w[w[6] + 2 * i + 1];
    color = [c0 & 0xffff, c0 >>> 16, c1 & 0xffff, c1 >>> 16].map((q) => f32(q * C_UNORM16)) as typeof color;
  }
  return { p, n: [nn[0], nn[1], nn[2]], uv, color };
}

function verifyArena(a: VertexArena, g: SceneGeometry, recentred: Float32Array, q: SceneQuant, groups: Int32Array): void {
  for (let v = 0; v < a.vertexCount; v++) {
    const d = decodeVertex(a, v, groups[v] >= 0 ? q.uv[groups[v]] : undefined);
    const bad = (what: string, got: readonly number[], want: readonly number[]) => {
      for (let k = 0; k < want.length; k++) if (f32bits(got[k]) !== f32bits(want[k])) {
        throw new QuantizationError(`vertex packer: ${what} of vertex ${v} does not round-trip (${got.join(', ')} vs ${want.join(', ')})`);
      }
    };
    bad('position', d.p, [recentred[3 * v], recentred[3 * v + 1], recentred[3 * v + 2]]);
    bad('normal', d.n, [g.normals[3 * v], g.normals[3 * v + 1], g.normals[3 * v + 2]]);
    if (groups[v] >= 0) bad('uv', d.uv, [g.uv0[2 * v], g.uv0[2 * v + 1]]);
    if (g.color0) bad('COLOR_0', d.color, [g.color0[4 * v], g.color0[4 * v + 1], g.color0[4 * v + 2], g.color0[4 * v + 3]]);
  }
}
