// CPU checks for the M1 integration: WGSL struct layouts of the scene/G-buffer records vs the TS packers, primary
// pass composition for every variant axis, and the vertex/triangle/material packers.
import { describe, expect, it } from 'vitest';
import { composeWgsl } from '../../src/core/gpu/wgsl-composer.ts';
import { GBUF_TEXEL_BYTES, PRIMARY_PARAMS_SIZE } from '../../src/core/render/renderer.ts';
import {
  MATERIAL_LAYOUT, MAT_ALPHA_MASK, MAT_V1, TRI_FLAGS_SHIFT, VERTEX_BYTES, packMaterials, packTris, packVertices, recentrePositions,
} from '../../src/core/render/scene-gpu.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import type { MaterialData, SceneGeometry } from '../../src/core/scene/types.ts';

type Layout = { size: number; align: number; offsets: Record<string, number> };

/** WGSL host-shareable layout of every struct in `src` (in declaration order). */
function structLayouts(src: string): Map<string, Layout> {
  const code = src.replace(/\/\/.*$/gm, '');
  const out = new Map<string, Layout>();
  const type = (t: string): { size: number; align: number } => {
    t = t.trim();
    if (/^(f32|u32|i32|atomic<u32>)$/.test(t)) return { size: 4, align: 4 };
    const m = /^vec([234])[fui]$/.exec(t);
    if (m) return m[1] === '2' ? { size: 8, align: 8 } : m[1] === '3' ? { size: 12, align: 16 } : { size: 16, align: 16 };
    if (t === 'mat4x4f') return { size: 64, align: 16 };
    const s = out.get(t);
    if (!s) throw new Error(`unknown type ${t}`);
    return s;
  };
  for (const m of code.matchAll(/struct\s+(\w+)\s*\{([^}]*)\}/g)) {
    if (/:\s*bool\b/.test(m[2])) continue; // not host-shareable (private structs)
    let off = 0, align = 1;
    const offsets: Record<string, number> = {};
    for (const f of m[2].split(',').map((s) => s.trim()).filter(Boolean)) {
      const fm = /^(\w+)\s*:\s*(.+)$/.exec(f);
      if (!fm) continue;
      if (fm[2].includes('array')) continue;
      const t = type(fm[2]);
      off = Math.ceil(off / t.align) * t.align;
      offsets[fm[1]] = off;
      off += t.size;
      align = Math.max(align, t.align);
    }
    out.set(m[1], { size: Math.ceil(off / align) * align, align, offsets });
  }
  return out;
}

const baseDefines = {
  COLOR_FORMAT: 'rgba32float', SCENE_GROUP: 1, BVH_DECLARE_BINDINGS: true, BVH_GROUP: 1, BVH_BINDING_NODES: 0, BVH_BINDING_TRIS: 1,
  CUSTOM_ALPHA: true, TEX_GROUP: 1, TEX_BINDING_BASE: 8, TEX_ARRAYS: 2, TEX_SAMPLERS: 3, ENV_GROUP: 0, ENV_BINDING: 1,
  WATERTIGHT: false, BVH_STATS: false,
};
const primary = (d: Record<string, string | number | boolean> = {}) =>
  composeWgsl('passes/primary.wgsl', { sources: shaderSources, defines: { ...baseDefines, ...d } }).code;

describe('M1 scene / G-buffer layouts', () => {
  const L = structLayouts(primary());

  it('MaterialGpu matches MATERIAL_LAYOUT', () => {
    const m = L.get('MaterialGpu')!;
    expect(m.size).toBe(MATERIAL_LAYOUT.size);
    for (const [k, v] of Object.entries(MATERIAL_LAYOUT)) if (k !== 'size') expect(m.offsets[k], k).toBe(v);
  });

  it('SceneVertex, GBufTexel, PrimaryParams sizes', () => {
    expect(L.get('SceneVertex')!.size).toBe(VERTEX_BYTES);
    expect(L.get('GBufTexel')!.size).toBe(GBUF_TEXEL_BYTES);
    expect(L.get('GBufTexel')!.offsets).toMatchObject({ ng: 0, thr: 12, ns: 16, viewZ: 28, pos: 32, matId: 44, albedo: 48, flags: 60, motion: 64 });
    expect(L.get('PrimaryParams')!.size).toBe(PRIMARY_PARAMS_SIZE);
  });

  it('primary composes for every variant axis', () => {
    for (const WATERTIGHT of [false, true]) for (const BVH_STATS of [false, true]) for (const [a, s] of [[0, 0], [1, 1], [16, 8]]) {
      for (const COLOR_FORMAT of ['rgba32float', 'rgba16float']) {
        const code = primary({ WATERTIGHT, BVH_STATS, TEX_ARRAYS: a, TEX_SAMPLERS: s, COLOR_FORMAT });
        expect(code).toContain('fn primary(');
        expect(code).toContain(`texture_storage_2d<${COLOR_FORMAT}, write>`);
        expect(code.includes('bvh_st_steps')).toBe(BVH_STATS);
        expect(code.match(/^fn alpha_pass\(/gm)).toHaveLength(1);
      }
    }
    // default (no CUSTOM_ALPHA): traverse.wgsl supplies the pass-through hook exactly once
    const plain = primary({ CUSTOM_ALPHA: false });
    expect(plain.match(/^fn alpha_pass\(/gm)).toHaveLength(1);
  });
});

describe('scene packers', () => {
  const g: SceneGeometry = {
    positions: Float32Array.from([1e4 + 0.25, 2, 3, 1e4 + 1.25, 2, 3, 1e4, 3, 3]),
    normals: Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    tangents: new Float32Array(12),
    uv0: Float32Array.from([0, 0, 1, 0, 0, 1]),
    indices: Uint32Array.from([0, 1, 2]),
    triMaterial: Uint32Array.from([7]),
    triFlags: Uint32Array.from([5]),
  };

  it('recentres in f64 before the f32 cast', () => {
    const r = recentrePositions(g.positions, [1e4 + 0.125, 2, 3]);
    expect(r[0]).toBe(Math.fround(g.positions[0] - (1e4 + 0.125)));
    expect(r[3]).toBe(Math.fround(g.positions[3] - (1e4 + 0.125)));
  });

  it('vertices: p, uv, n, colour (white when COLOR_0 is absent)', () => {
    const v = packVertices(g, recentrePositions(g.positions, [0, 0, 0]));
    expect(v.length).toBe(3 * VERTEX_BYTES / 4);
    expect(Array.from(v.subarray(12, 24))).toEqual([Math.fround(1e4 + 1.25), 2, 3, 1, 0, 0, 1, 0, 1, 1, 1, 1]);
  });

  it('tris: indices + material | flags << 24', () => {
    const t = packTris(g);
    expect(Array.from(t)).toEqual([0, 1, 2, (7 | (5 << TRI_FLAGS_SHIFT)) >>> 0]);
  });

  it('materials: factors, emission·strength, flags, invalid texture slots', () => {
    const m: MaterialData = {
      name: 'x', baseColorFactor: [0.1, 0.2, 0.3, 0.4], metallicFactor: 0.5, roughnessFactor: 0.6, emissiveFactor: [1, 2, 3], emissiveStrength: 2,
      ior: 1.45, specularFactor: 0.7, specularColorFactor: [1, 1, 0.5], transmissionFactor: 0.1, alphaMode: 'MASK', alphaCutoff: 0.33,
      doubleSided: false, model: 'v1', v1: { diffuse: [0.8, 0.7, 0.6], glossy: [0.1, 0.1, 0.1], roughness: 0.3, mix: 0.25 },
    };
    const dv = new DataView(packMaterials([m], null));
    const f = (o: number) => dv.getFloat32(o, true);
    expect(f(MATERIAL_LAYOUT.baseColor + 12)).toBeCloseTo(0.4, 7);
    expect([f(MATERIAL_LAYOUT.emission), f(MATERIAL_LAYOUT.emission + 4), f(MATERIAL_LAYOUT.emission + 8)]).toEqual([2, 4, 6]);
    expect(f(MATERIAL_LAYOUT.alphaCutoff)).toBeCloseTo(0.33, 7);
    expect(dv.getUint32(MATERIAL_LAYOUT.flags, true)).toBe(MAT_ALPHA_MASK | MAT_V1);
    expect(f(MATERIAL_LAYOUT.v1Diffuse)).toBeCloseTo(0.8, 7);
    expect(f(MATERIAL_LAYOUT.v1Mix)).toBeCloseTo(0.25, 7);
    expect(dv.getUint32(MATERIAL_LAYOUT.texBaseColor + 12, true) >>> 31).toBe(0); // invalid slot
    expect(f(MATERIAL_LAYOUT.texBaseColor)).toBe(1); // identity transform
  });
});
