// M6 Gate-0 GPU suite (restir-m6-api.md §7): U-M5-BITS (temporal chains with every M6 feature off ≡ the M5 build) and
// the M6 feature tests. VITE_M6_BITS_RECORD=1 prints the hashes instead of asserting (used once on main f5c23fd).
import { describe, expect, it } from 'vitest';
import { M5_BITS_CASES, m5BitsCase } from './restir-m6-bits.ts';

const RECORD = String(import.meta.env?.VITE_M6_BITS_RECORD ?? '') === '1';
// Recorded on the M5 build (main f5c23fd, Chrome 154 / Metal) with restir-m6-bits.ts.
const GOLDEN_M5: Record<string, string[]> = {
  'xq-full-static': [
    '82f3b457:c2357feb:0,0,0,0:844,66,200,0,0,0,0,0,0,0',
    '98c20044:e1381bf7:0,0,0,0:814,66,161,1968,1220,1050,0,0,0,0',
    'd77d39b7:cdf9dedd:0,0,0,0:844,76,143,2295,1667,729,0,0,0,0',
    'c46f25ef:e98a1a53:0,0,0,0:770,49,130,2379,1921,499,0,0,0,0',
    'f2912675:e2d3fe45:0,0,0,0:808,63,126,2397,1998,429,0,0,0,0',
  ],
  'all-full-motion': [
    'c6a2a4aa:7d950f3a:0,0,0,0:844,84,159,0,0,0,0,0,0,0',
    'e7a3fcc3:2f29b191:0,0,0,0:814,100,145,2270,1243,1148,0,0,0,0',
    '19b32a24:98bff02d:0,0,0,0:844,89,119,2385,1680,750,0,0,0,0',
    '5eee0b43:d16099ac:0,0,0,0:798,56,121,2390,1894,544,3036,318,0,0',
    'c16abd31:d3472bd9:0,0,0,0:806,64,131,2391,1977,462,2956,312,0,0',
    'e736179a:6132d53f:0,0,0,0:834,75,138,2364,2031,411,2896,294,0,0',
  ],
  'xq-interactive-M5': [
    '3bdcd165:a327a61e:0,0,0,0:1764,132,357,0,0,0,0,0,0,0',
    '1b3356be:175dff33:0,0,0,0:814,66,163,2030,1298,990,0,0,0,0',
    '67ea7d49:70fd61ac:0,0,0,0:844,79,150,2305,1697,705,3090,247,0,0',
    'cf73e88e:31059e64:0,0,0,0:770,52,125,2381,1883,539,2993,225,0,0',
    '0e3d4d0d:54a3a829:0,0,0,0:808,64,132,2397,1916,514,2982,220,0,0',
  ],
  'all-talbot': [
    '9cf71721:3b71bafc:0,0,0,0:844,71,170,0,0,0,0,0,0,0',
    '2bbe1aba:56324fee:0,0,0,0:814,98,150,2204,1234,2203,0,0,0,0',
    'f9e67d56:a8e2bfb6:0,0,0,0:844,90,120,2367,1666,2186,4716,139,0,0',
    '7d2e6d11:d2c278fe:0,0,0,0:770,48,117,2416,1900,2185,4733,143,0,0',
  ],
};

describe('U-M5-BITS: temporal chains with the M6 features off ≡ the M5 build (per frame: final reservoirs, image, counters)', () => {
  for (const c of M5_BITS_CASES) {
    it(c.name, async () => {
      const h = await m5BitsCase(c);
      console.log(`[U-M5-BITS] ${c.name} ${JSON.stringify(h)}`);
      if (RECORD) return;
      expect(h).toEqual(GOLDEN_M5[c.name]);
    });
  }
});
