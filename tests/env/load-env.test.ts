// Loader policy: format sniffing, validation limits, interactive 2×2 downsampling, defaults.
import { describe, expect, it } from 'vitest';
import { decodeEnvironment, downsample2x2 } from '../../src/core/scene/env/load-env.ts';
import { encodeHdr } from './rgbe-synth.ts';

const flatHdr = (w: number, h: number, byte = 128, e = 129) => {
  const rgbe = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) rgbe.set([byte, byte, byte, e], i * 4);
  return encodeHdr(w, h, rgbe, w >= 8 ? 'rle' : 'flat');
};

describe('load-env', () => {
  it('sniffs .hdr, applies defaults and warns on non-2:1 aspect', () => {
    const r = decodeEnvironment(flatHdr(8, 8), { mode: 'validation' });
    expect(r.format).toBe('hdr');
    expect(r.env).toMatchObject({ width: 8, height: 8, strength: 1, tint: [1, 1, 1], rotationZ: 0, visibleToCamera: true });
    expect(r.env.texels[0]).toBe(128 * 2 ** -7);
    expect(r.warnings.join()).toMatch(/2:1/);
  });

  it('rejects unknown formats and over-size maps in validation', () => {
    expect(() => decodeEnvironment(new Uint8Array([1, 2, 3, 4]))).toThrow(/unknown environment format/);
    expect(() => decodeEnvironment(flatHdr(8200, 1), { mode: 'validation' })).toThrow(/8192/);
  });

  it('interactive mode 2×2-downsamples above 4096 (energy-preserving mean)', () => {
    const r = decodeEnvironment(flatHdr(8192, 2), { mode: 'interactive' });
    expect(r.env.width).toBe(4096);
    expect(r.env.height).toBe(1);
    expect(r.downsampleSteps).toBe(1);
    expect(r.env.texels[0]).toBe(128 * 2 ** -7);
    const d = downsample2x2({ width: 2, height: 2, texels: new Float32Array([1, 2, 3, 1, 3, 4, 5, 1, 5, 6, 7, 1, 7, 8, 9, 1]), warnings: [] });
    expect(Array.from(d.texels)).toEqual([4, 5, 6, 1]);
  });
});
