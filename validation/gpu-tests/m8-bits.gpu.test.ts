// U-M8-BITS (docs/decisions/m8-perf.md §4; M8 Gate 0): optimisations that must not change results are bitwise the
// pre-M8 build (eda22be). VITE_M8_BITS_RECORD=1 prints the hashes instead of asserting (recorded once before any ReSTIR /
// PT shader change; see m8-bits.ts). PT batch on the same scenes (U-M8-PTBITS).
import { describe, expect, it } from 'vitest';
import { M8_BITS_CASES, m8BitsCase, m8PtBits, type M8BitsScene } from './m8-bits.ts';
import type { LightMode } from '../../src/core/render/lights-gpu.ts';

const RECORD = String(import.meta.env?.VITE_M8_BITS_RECORD ?? '') === '1';
// Recorded on m8-perf at b724d4c + the CWBVH WGSL (BVH_CWBVH absent: the composed text of every validation pipeline is
// the eda22be text, U-M7-BITS), Chrome 154 / Metal, before any further ReSTIR / PT shader change.
const GOLDEN: Record<string, string[]> = {
  'all-interactive-B': ["2a0b7fd2:ae7201c8:0,0,0,0:1966,194,315,0,0,0,0,0,0,0","692d4fa1:fc9adf81:0,0,0,0:990,107,144,2358,1402,1007,0,0,0,0","808a662e:7dcc0ee5:0,0,0,0:1012,96,122,2382,1717,705,3207,85,0,0","b2fc32cc:65670b08:0,0,0,0:1030,117,153,2382,1795,632,3140,87,0,0","ab212fb9:c2b1ea07:0,0,0,0:1020,105,155,2379,1813,611,3120,79,0,0"],
  'all-full-m6-B': ["52ac52cd:fef05b2e:0,0,0,0:984,114,162,0,0,0,0,0,0,0","69150d98:aab451bb:0,0,0,0:990,106,148,2361,1309,1098,0,0,0,0","9bba10da:9716b98c:0,0,0,0:1002,105,130,2388,1734,682,3183,77,0,0","d72d84a8:aeffc51b:0,0,0,0:1026,115,156,2381,1947,484,2975,66,0,0"],
  'all-offline-m6-B': ["efa108a0:4dbabe62:0,0,0,0:6004,629,777,0,0,0,0,0,0,0"],
  'nm-interactive-B': ["0b047d69:ca30fe5f:0,0,0,0:4268,83,669,0,0,0,0,0,0,0","602e96ff:b006f9cc:0,0,0,0:2086,50,369,3755,2321,1554,0,0,0,0","e89834e8:0ee6b840:0,0,0,0:2050,29,337,3788,2777,1073,5139,1804,0,0","d10910e5:050811f9:0,0,0,0:2078,43,341,3754,2836,994,5026,1843,0,0"],
  'glass-interactive-B': ["89aea652:d143812b:0,0,0,0:3068,222,572,0,0,0,0,0,0,0","7b8b1e02:ef60a0e0:0,0,0,0:1494,112,306,3224,2041,1415,0,0,0,0","423d872a:aee98644:0,0,0,0:1422,91,279,3480,2556,1028,0,0,0,0","0383b5d1:793455ee:0,0,0,0:1416,123,274,3522,2724,860,0,0,0,0"],
  'alpha-full-m6-A': ["fd77c17f:cf6e26e0:0,0,0,0:806,0,171,0,0,0,0,0,0,0","9559508f:89f053d8:0,0,0,0:720,0,162,2027,1150,929,0,0,0,0","eaf8147e:08f0df3f:0,0,0,0:702,0,160,2067,1507,586,0,0,0,0"],
};
const GOLDEN_PT: Record<string, string> = {
  'all-B': '61082f13:0,0,0,0',
  'all-A': 'e0f14980:0,0,0,0',
  'nm_smooth-B': '915c361c:0,0,0,0',
  'glass-B': '4d7cee66:0,0,0,0',
  'alpha-A': '135260d9:0,0,0,0',
};
const PT_CASES: [M8BitsScene, LightMode][] = [['all', 'B'], ['all', 'A'], ['nm_smooth', 'B'], ['glass', 'B'], ['alpha', 'A']];

describe('U-M8-BITS: ReSTIR chains (interactive and validation presets, light mode B) ≡ the pre-M8 build', () => {
  for (const c of M8_BITS_CASES) {
    it(c.name, async () => {
      const h = await m8BitsCase(c);
      console.log(`[U-M8-BITS] '${c.name}': ${JSON.stringify(h)},`);
      if (RECORD) return;
      expect(h).toEqual(GOLDEN[c.name]);
    }, 300_000);
  }
});

describe('U-M8-PTBITS: PT batch (validation configuration) ≡ the pre-M8 build', () => {
  for (const [s, m] of PT_CASES) {
    it(`${s} mode ${m}`, async () => {
      const h = await m8PtBits(s, m);
      console.log(`[U-M8-PTBITS] '${s}-${m}': '${h}',`);
      if (RECORD) return;
      expect(h).toBe(GOLDEN_PT[`${s}-${m}`]);
    }, 300_000);
  }
});
