// P2 (docs/decisions/data-formats.md §B9): the env texture is uploaded in the smallest format in which EVERY texel
// round-trips bit-exactly (rgb9e5ufloat → rgba16float → rgba32float). CPU part: the exact encoders and the format
// choice for the validation HDRIs and the synthetic M3c envs; the GPU filtering check (ENV-F) is
// validation/gpu-tests/env-format.gpu.test.ts.
import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { f16BitsExact, packEnvTexels, rgb9e5Exact } from '../../src/core/render/env-gpu.ts';
import { decodeEnvironment } from '../../src/core/scene/env/load-env.ts';

const f16Value = (h: number) => {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  return s * (e === 0 ? m * 2 ** -24 : (1 + m / 1024) * 2 ** (e - 15));
};
const rgb9e5Value = (w: number) => {
  const E = w >>> 27, s = 2 ** (E - 24);
  return [(w & 511) * s, ((w >>> 9) & 511) * s, ((w >>> 18) & 511) * s];
};

describe('P2 env texture: smallest exact format', () => {
  it('f16 / rgb9e5 encoders are exact where they accept and refuse otherwise', () => {
    for (const x of [0, -0, 1, 0.5, 65504, 2 ** -24, 3 * 2 ** -24, 2 ** -14, 1023 * 2 ** -24, 1.5, 1000.5, -7.25]) {
      const h = f16BitsExact(x);
      expect(h, String(x)).not.toBeNull();
      expect(Object.is(f16Value(h!), x) || f16Value(h!) === x, String(x)).toBe(true);
    }
    for (const x of [65520, 1 + 2 ** -11, 2 ** -25, 0.1, Infinity, NaN]) expect(f16BitsExact(x), String(x)).toBeNull();
    for (const t of [[0, 0, 0], [1, 0.5, 0.25], [524, 2, 4], [61184, 0, 512], [511 * 2 ** -24, 0, 2 ** -24], [65408, 65408, 65408]]) {
      const w = rgb9e5Exact(t[0], t[1], t[2]);
      expect(w, t.join()).not.toBeNull();
      expect(rgb9e5Value(w!)).toEqual(t);
    }
    // 10 000 needs 10 mantissa bits next to a small component; 1 + 2^-9; negative; > 65408
    for (const t of [[10000, 1, 1], [1 + 2 ** -9, 0, 0], [-1, 0, 0], [65536, 0, 0], [1, 0.1, 0]]) expect(rgb9e5Exact(t[0], t[1], t[2]), t.join()).toBeNull();
  });

  it('validation HDRIs are exact in rgb9e5 (RGBE with exponents 112…144); the C0s sun texel falls back to f16; noise to f32', () => {
    const dir = 'validation/assets/downloaded/hdri';
    for (const id of ['studio_small_09_1k', 'overcast_soil_puresky_1k', 'kloofendal_48d_partly_cloudy_puresky_1k']) {
      const f = `${dir}/${id}.hdr`;
      if (!existsSync(f)) continue;
      const env = decodeEnvironment(new Uint8Array(readFileSync(f)), { mode: 'validation' }).env;
      const p = packEnvTexels(env.texels, ['rgb9e5ufloat', 'rgba16float', 'rgba32float']);
      expect(p.format, id).toBe('rgb9e5ufloat');
      const w = p.data as Uint32Array;
      for (let i = 0; i < env.width * env.height; i++) {
        const v = rgb9e5Value(w[i]);
        for (let k = 0; k < 3; k++) if (v[k] !== env.texels[4 * i + k]) throw new Error(`${id} texel ${i}.${k}: ${v[k]} != ${env.texels[4 * i + k]}`);
      }
      // …and in rgba16float when 9e5 is not allowed
      expect(packEnvTexels(env.texels, ['rgba16float', 'rgba32float']).format, id).toBe('rgba16float');
    }
    const W = 512, H = 256;
    const sun = new Float32Array(W * H * 4);
    for (let i = 0; i < W * H; i++) sun.set([0, 0, 0, 1], 4 * i);
    sun.set([1e4, 1e4, 1e4, 1], 4 * (100 * W + 7));
    expect(packEnvTexels(sun, ['rgb9e5ufloat', 'rgba16float', 'rgba32float']).format).toBe('rgba16float'); // 10 000 = 10 mantissa bits
    const cst = new Float32Array(64 * 32 * 4).fill(1);
    expect(packEnvTexels(cst, ['rgb9e5ufloat', 'rgba16float', 'rgba32float']).format).toBe('rgb9e5ufloat');
    const noise = Float32Array.from({ length: 64 * 4 }, (_, i) => Math.fround(Math.sin(i) * 3 + 3.3));
    expect(packEnvTexels(noise, ['rgb9e5ufloat', 'rgba16float', 'rgba32float']).format).toBe('rgba32float');
  });
});
