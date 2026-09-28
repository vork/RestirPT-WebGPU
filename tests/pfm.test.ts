import { describe, expect, it } from 'vitest';
import { decodePFM, encodePFM, orientationPattern } from '../src/core/io/pfm.ts';

describe('pfm codec', () => {
  it('round-trips and stores rows bottom-to-top', () => {
    const img = orientationPattern(7, 5);
    const bytes = encodePFM(img);
    const header = new TextDecoder().decode(bytes.slice(0, 12));
    expect(header.startsWith('PF\n7 5\n-1.0\n')).toBe(true);
    // first stored pixel is the bottom-left image pixel (c=0, r=4)
    const first = new DataView(bytes.buffer, 12).getFloat32(4, true);
    expect(first).toBe(4);
    const back = decodePFM(bytes);
    expect(back.width).toBe(7); expect(back.height).toBe(5);
    expect(Array.from(back.data)).toEqual(Array.from(img.data));
  });
});
