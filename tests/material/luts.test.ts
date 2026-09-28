// T8 / gap-bsdf U-1 (CPU part): the extracted Cycles LUTs (checksums, source provenance) and the reader test
// vectors of gap-bsdf §13.1–13.3 / cycles-conventions §4.5, through the f64 reference port of lookup_table.h.
import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import {
  LUT_CHECKSUMS, LUT_SOURCE_SHA256, LUT_SOURCE_URL, TABLE_GGX_E, TABLE_GGX_EAVG, TABLE_GGX_GEN_SCHLICK_IOR_S, TABLE_GGX_GEN_SCHLICK_S,
} from '../../src/core/render/luts/cycles-luts.ts';
import { LUT_LAYOUT, lutDefines, lutFloats } from '../../src/core/render/luts/lut-layout.ts';
import { composeWgsl } from '../../src/core/gpu/wgsl-composer.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { F0FromIor, S_ior, S_s, ggxE, ggxEavg } from './bsdf-ref.ts';

const zIor = (eta: number) => Math.sqrt(Math.abs((eta - 1) / (eta + 1)));

describe('Cycles LUTs (shader.tables @ v5.1.2)', () => {
  it('provenance + per-table count, sum (gap-bsdf §3.5) and f32 sha256', () => {
    expect(LUT_SOURCE_URL).toContain('/v5.1.2/intern/cycles/scene/shader.tables');
    expect(LUT_SOURCE_SHA256).toMatch(/^[0-9a-f]{64}$/);
    const tables: [keyof typeof LUT_CHECKSUMS, Float32Array, number][] = [
      ['table_ggx_gen_schlick_ior_s', TABLE_GGX_GEN_SCHLICK_IOR_S, 184.247126],
      ['table_ggx_gen_schlick_s', TABLE_GGX_GEN_SCHLICK_S, 764.768581],
      ['table_ggx_E', TABLE_GGX_E, 849.736018],
      ['table_ggx_Eavg', TABLE_GGX_EAVG, 25.879163],
    ];
    for (const [name, t, sum] of tables) {
      const ck = LUT_CHECKSUMS[name];
      expect(t.length, name).toBe(ck.count);
      const s = t.reduce((a, b) => a + b, 0);
      expect(Math.abs(s - sum), name).toBeLessThan(1e-5);
      expect(createHash('sha256').update(new Uint8Array(t.buffer, t.byteOffset, t.byteLength)).digest('hex'), name).toBe(ck.sha256F32);
    }
  });

  it('block layout: offsets of math.md#bsdf-v2, 9248 floats', () => {
    const b = lutFloats();
    expect(b.length).toBe(9248);
    expect(b[LUT_LAYOUT.ggxGenSchlickIorS + 4095]).toBe(TABLE_GGX_GEN_SCHLICK_IOR_S[4095]);
    expect(b[LUT_LAYOUT.ggxGenSchlickS]).toBe(TABLE_GGX_GEN_SCHLICK_S[0]);
    expect(b[LUT_LAYOUT.ggxE + 1023]).toBe(TABLE_GGX_E[1023]);
    expect(b[LUT_LAYOUT.ggxEavg + 31]).toBe(TABLE_GGX_EAVG[31]);
  });

  it('S_ior test vectors (gap-bsdf §13.1, cycles-conventions §4.5)', () => {
    // [ior, r, [mu=1, 0.7, 0.4, 0.1]]
    const rows: [number, number, number[]][] = [
      [1.5, 0.0, [0.0000, 0.0117, 0.0955, 0.5628]],
      [1.5, 0.25, [0.0001, 0.0126, 0.0927, 0.4092]],
      [1.5, 0.5, [0.0006, 0.0138, 0.0639, 0.1705]],
      [1.5, 1.0, [0.0014, 0.0058, 0.0161, 0.0403]],
      [1.33, 0.25, [0.0000, 0.0092, 0.0763, 0.3857]],
      [1.33, 0.5, [0.0004, 0.0105, 0.0529, 0.1539]],
      [1.33, 1.0, [0.0010, 0.0044, 0.0130, 0.0344]],
      [2.0, 0.25, [0.0001, 0.0148, 0.0969, 0.4013]],
      [2.0, 0.5, [0.0007, 0.0158, 0.0669, 0.1700]],
      [2.0, 1.0, [0.0017, 0.0066, 0.0172, 0.0417]],
    ];
    const mus = [1, 0.7, 0.4, 0.1];
    for (const [ior, r, want] of rows) {
      mus.forEach((mu, i) => expect(Math.abs(S_ior(r, mu, zIor(ior)) - want[i]), `ior ${ior} r ${r} mu ${mu}`).toBeLessThanOrEqual(5.01e-5));
    }
    // E = F0 + (1 − F0)·S at IOR 1.5 (cycles-conventions §4.5): 0.1013 at r = 0.5, μ = 0.4
    const F0 = F0FromIor(1.5);
    expect(Math.abs(F0 + (1 - F0) * S_ior(0.5, 0.4, zIor(1.5)) - 0.1013)).toBeLessThan(5.01e-5);
    expect(Math.abs(F0 + (1 - F0) * S_ior(0.5, 0.1, zIor(1.5)) - 0.2037)).toBeLessThan(5.01e-5);
  });

  it('S5 (metal sample weight, §13.2), ggx_E / ggx_Eavg (§13.3)', () => {
    const s5: [number, number[]][] = [
      [0.0, [0.0000, 0.0034, 0.0805, 0.5974]], [0.25, [0.0000, 0.0044, 0.0790, 0.4330]],
      [0.5, [0.0001, 0.0068, 0.0540, 0.1718]], [1.0, [0.0002, 0.0030, 0.0119, 0.0354]],
    ];
    const mus = [1, 0.7, 0.4, 0.1];
    for (const [r, want] of s5) mus.forEach((mu, i) => expect(Math.abs(S_s(r, mu, 0.5) - want[i]), `S5 r ${r} mu ${mu}`).toBeLessThanOrEqual(5.01e-5));
    const E: [number, number[], number][] = [
      [0.25, [0.99560, 0.99302, 0.98244, 0.89780], 0.98710],
      [0.5, [0.91528, 0.88498, 0.84441, 0.89170], 0.88204],
      [1.0, [0.30685, 0.37894, 0.49905, 0.76056], 0.40914],
    ];
    for (const [r, want, eavg] of E) {
      mus.forEach((mu, i) => expect(Math.abs(ggxE(r, mu) - want[i]), `E r ${r} mu ${mu}`).toBeLessThanOrEqual(5.01e-6));
      expect(Math.abs(ggxEavg(r) - eavg)).toBeLessThanOrEqual(5.01e-6);
    }
  });

  it('lut.wgsl composes for every records element kind and with its own binding', () => {
    for (const recordsKind of ['f32', 'u32', 'vec4f', 'vec4u'] as const) {
      const decl = { f32: 'array<f32>', u32: 'array<u32>', vec4f: 'array<vec4f>', vec4u: 'array<vec4u>' }[recordsKind];
      const src = `@group(0) @binding(3) var<storage, read> records: ${decl};\n#include "material/lut.wgsl"\n`;
      const code = composeWgsl('t.wgsl', { sources: { ...shaderSources, 't.wgsl': src }, defines: lutDefines({ base: 100, recordsKind }) }).code;
      expect(code).toContain('lut_read_3d(rough, mu, z, 100u, 16u, 16u, 16u)');
      expect(code).toContain('lut_read_2d(rough, mu, 8292u, 32u, 32u)');
      expect(code.match(/var<storage, read> records/g)).toHaveLength(1);
    }
    const own = composeWgsl('material/lut.wgsl', { sources: shaderSources, defines: lutDefines({ declare: { group: 2, binding: 5 } }) }).code;
    expect(own).toContain('@group(2) @binding(5) var<storage, read> records: array<f32>;');
    expect(() => lutDefines({ base: -1 })).toThrow();
  });
});
