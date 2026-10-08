// T14 (PLAN §7.4 M6; restir-m6-api.md MD3, §4): the σ = 16 Gaussian reciprocal pairing maps and the corrected n_σ.
import { describe, expect, it } from 'vitest';
import {
  Pcg32, gaussLayer, generateGaussLayer, layerDeltaStats, layersTexData, gaussLayers, nSigma, pairPartner, pairTransform,
  type PairLayer,
} from '../../src/core/render/restir/pairing.ts';
import { GAUSS_PAIR_SIZES } from '../../src/core/render/restir/presets.ts';

/** Standard normal CDF (Abramowitz–Stegun 7.1.26 via erf, |err| < 1.5e-7). */
function Phi(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x) / Math.SQRT2);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x / 2);
  return x >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

/** KS distance of the integer deltas of one axis vs N(0, σ²) with continuity correction (F_emp(k) vs Φ((k + ½)/σ)). */
function ksAxis(l: PairLayer, axis: 0 | 1, sigma: number): number {
  const counts = new Map<number, number>();
  let n = 0;
  for (let i = 0; i < l.delta.length; i += 2) { const v = l.delta[i + axis]; counts.set(v, (counts.get(v) ?? 0) + 1); n++; }
  let acc = 0, ks = 0;
  for (let k = -127; k <= 127; k++) {
    acc += counts.get(k) ?? 0;
    ks = Math.max(ks, Math.abs(acc / n - Phi((k + 0.5) / sigma)));
  }
  return ks;
}

describe('T14: σ = 16 Gaussian pairing maps (MD3)', () => {
  it('n_σ (corrected Eq. 3): 128 at σ = 16, 1 at σ = 0.814, 14 at σ = 5.3', () => {
    expect(nSigma(16)).toBe(128);
    expect(nSigma(0.814)).toBe(1);
    expect(nSigma(5.3)).toBe(14);
  });

  it('every layer is a torus involution with a partner for every texel, |d| ≤ 127; σ per axis within 3 %; normal (KS < 0.01)', () => {
    for (let s = 0; s < GAUSS_PAIR_SIZES.length; s++) {
      const W = GAUSS_PAIR_SIZES[s];
      const l = gaussLayer(W, 16, s);
      let bad = 0;
      for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
        const i = 2 * (y * W + x), dx = l.delta[i], dy = l.delta[i + 1];
        if (dx === 0 && dy === 0) { bad++; continue; }
        const bx = (((x + dx) % W) + W) % W, by = (((y + dy) % W) + W) % W, j = 2 * (by * W + bx);
        if (l.delta[j] !== -dx || l.delta[j + 1] !== -dy) bad++;
      }
      const st = layerDeltaStats(l);
      const ksx = ksAxis(l, 0, st.sx), ksy = ksAxis(l, 1, st.sy);
      console.log(`[T14] W=${W} σx=${st.sx.toFixed(3)} σy=${st.sy.toFixed(3)} max|d|=${st.maxAbs} KS x=${ksx.toFixed(4)} y=${ksy.toFixed(4)}`);
      expect(bad, `W=${W}`).toBe(0);
      expect(l.unmatched).toBe(0);
      expect(st.zero).toBe(0);
      expect(st.maxAbs).toBeLessThanOrEqual(127);
      expect(Math.abs(st.sx / 16 - 1)).toBeLessThan(0.03);
      expect(Math.abs(st.sy / 16 - 1)).toBeLessThan(0.03);
      expect(ksx).toBeLessThan(0.01);
      expect(ksy).toBeLessThan(0.01);
    }
  });

  it('small σ: one shuffle gives per-axis σ ≈ 0.816 (enh-verify C1), n_σ = 1', () => {
    const l = generateGaussLayer(64, 0.814, 0);
    const st = layerDeltaStats(l);
    expect(Math.abs(st.sx - Math.sqrt(2 / 3)) / Math.sqrt(2 / 3)).toBeLessThan(0.05);
  });

  it('deterministic, distinct per layer, texture layout as the M4 maps (zeros outside W_s)', () => {
    const a = generateGaussLayer(230, 16, 1), b = generateGaussLayer(230, 16, 1), c = generateGaussLayer(230, 16, 2);
    expect(Buffer.from(a.delta.buffer).equals(Buffer.from(b.delta.buffer))).toBe(true);
    expect(Buffer.from(a.delta.buffer).equals(Buffer.from(c.delta.buffer))).toBe(false);
    const t = layersTexData(gaussLayers(16));
    expect(t.length).toBe(256 * 256 * 8 * 2);
    let nz = 0;
    for (let s = 0; s < 8; s++) {
      const W = s < GAUSS_PAIR_SIZES.length ? GAUSS_PAIR_SIZES[s] : 0;
      for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++) if (x >= W || y >= W) nz += t[2 * (s * 65536 + y * 256 + x)] !== 0 ? 1 : 0;
    }
    expect(nz).toBe(0);
  });

  for (const [mw, mh] of [[960, 540], [1024, 1024]]) {
    it(`partner(partner(p)) = p for all 8 dihedral codes × random offsets on ${mw}×${mh} (every layer)`, () => {
      const rng = new Pcg32(5, 6, 7, 8);
      for (let s = 0; s < GAUSS_PAIR_SIZES.length; s++) {
        const l = gaussLayer(GAUSS_PAIR_SIZES[s], 16, s);
        for (let rep = 0; rep < 3; rep++) {
          const tr = pairTransform(rng.next(), 0, rng.next(), rep, s, l.W);
          for (let code = 0; code < 8; code++) {
            const t2 = { ...tr, code };
            let n = 0, bad = 0;
            for (let y = 0; y < mh; y += 3) for (let x = 0; x < mw; x += 3) {
              const q = pairPartner(l, t2, [x, y], mw, mh);
              if (!q) continue;
              n++;
              const r = pairPartner(l, t2, q, mw, mh);
              if (!r || r[0] !== x || r[1] !== y) bad++;
            }
            expect(bad).toBe(0);
            expect(n).toBeGreaterThan(0.9 * (mw / 3) * (mh / 3));
          }
        }
      }
    });
  }
});
