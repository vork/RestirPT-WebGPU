// T14-M4 (restir-api.md §6.1, §2.8): the M4 uniform-disk involution maps and the per-(t, r, m, s) transform.
import { describe, expect, it } from 'vitest';
import {
  Pcg32, dihedralApply, dihedralApplyT, generatePairLayer, pairLayer, pairPartner, pairTexData, pairTransform, radialKs,
  type PairLayer, type PairTransform,
} from '../../src/core/render/restir/pairing.ts';
import { PAIR_TEX_SIZES } from '../../src/core/render/restir/presets.ts';

const SIZES = PAIR_TEX_SIZES.filter((w) => w > 0);

describe('T14-M4: pairing maps', () => {
  it('PCG32 matches the reference stream (pcg32_srandom(42, 54))', () => {
    const r = new Pcg32(0, 42, 0, 54);
    expect(Array.from({ length: 6 }, () => r.next())).toEqual([0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e]);
  });

  it('dihedral Mᵀ·M = identity for all 8 codes', () => {
    for (let c = 0; c < 8; c++) for (const v of [[3, -7], [0, 5], [-2, 0], [11, 13]] as [number, number][]) {
      expect(dihedralApplyT(c, dihedralApply(c, v))).toEqual(v);
    }
  });

  for (const R of [10, 30]) {
    it(`R = ${R}: every layer is a torus involution, unmatched < 1%, radial KS (lattice-uniform) < 0.01`, () => {
      for (let s = 0; s < SIZES.length; s++) {
        const W = SIZES[s];
        const l = generatePairLayer(W, R, s);
        let bad = 0, n0 = 0;
        for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
          const i = 2 * (y * W + x), dx = l.delta[i], dy = l.delta[i + 1];
          if (dx === 0 && dy === 0) { n0++; continue; }
          if (dx * dx + dy * dy > R * R) bad++;
          const bx = (x + dx + W) % W, by = (y + dy + W) % W, j = 2 * (by * W + bx);
          if (l.delta[j] !== -dx || l.delta[j + 1] !== -dy) bad++;
        }
        const ks = radialKs(l);
        console.log(`[T14-M4] R=${R} W=${W} unmatched=${(100 * n0 / (W * W)).toFixed(3)}% KS lattice=${ks.lattice.toFixed(4)} continuous=${ks.continuous.toFixed(4)}`);
        expect(bad, `W=${W}`).toBe(0);
        expect(n0).toBe(l.unmatched);
        expect(n0 / (W * W)).toBeLessThan(0.01);
        expect(ks.lattice).toBeLessThan(0.01);
      }
    });
  }

  it('deterministic: regenerated layers and texture data are identical', () => {
    const a = generatePairLayer(254, 30, 0), b = generatePairLayer(254, 30, 0);
    expect(Buffer.from(a.delta.buffer).equals(Buffer.from(b.delta.buffer))).toBe(true);
    const t = pairTexData(10);
    expect(t.length).toBe(256 * 256 * 8 * 2);
    // texels outside W_s are zero
    const W = SIZES[5];
    let nz = 0;
    for (let y = 0; y < 256; y++) for (let x = W; x < 256; x++) nz += t[2 * (5 * 65536 + y * 256 + x)] !== 0 ? 1 : 0;
    expect(nz).toBe(0);
  });

  for (const [mw, mh] of [[960, 540], [1024, 1024]]) {
    it(`partner(partner(p)) = p for all 8 dihedral codes × random offsets on ${mw}×${mh}`, () => {
      const rng = new Pcg32(1, 2, 3, 4);
      for (const R of [10, 30]) {
        for (let s = 0; s < SIZES.length; s++) {
          const l: PairLayer = pairLayer(SIZES[s], R, s);
          for (let code = 0; code < 8; code++) {
            const tr: PairTransform = { code, ox: rng.bounded(l.W), oy: rng.bounded(l.W) };
            let bad = 0, paired = 0;
            for (let y = 0; y < mh; y += (R === 30 ? 1 : 3)) for (let x = 0; x < mw; x++) {
              const q = pairPartner(l, tr, [x, y], mw, mh);
              if (!q) continue;
              paired++;
              const pp = pairPartner(l, tr, q, mw, mh);
              if (!pp || pp[0] !== x || pp[1] !== y) bad++;
              if ((q[0] - x) ** 2 + (q[1] - y) ** 2 > R * R) bad++;
            }
            expect(bad, `R=${R} s=${s} code=${code}`).toBe(0);
            expect(paired).toBeGreaterThan(0);
          }
        }
      }
    });
  }

  it('transform: codes cover all 8 values and offsets stay in [0, W_s) (hash mirror)', () => {
    const seen = new Set<number>();
    for (let t = 0; t < 64; t++) for (let s = 0; s < 6; s++) {
      const tr = pairTransform(11, 0, t, 0, s, SIZES[s]);
      seen.add(tr.code);
      expect(tr.ox).toBeLessThan(SIZES[s]);
      expect(tr.oy).toBeLessThan(SIZES[s]);
    }
    expect(seen.size).toBe(8);
  });
});
