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

// M8 P-7 (m8-perf.md §8): plane-major reservoirs (RS_RES_SOA, the interactive kernel) give the AoS hashes bit for bit
// (readReservoirs returns AoS words in both layouts).
describe('U-M8-BITS (SoA): plane-major reservoirs ≡ the pre-M8 build', () => {
  for (const c of M8_BITS_CASES) {
    it(c.name, async () => {
      const h = await m8BitsCase(c, { resLayout: 'soa' });
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

// m8-perf.md §5 (P-4): without rect / disk lights (nothing a BSDF ray can cross) light mode B and A are the same
// estimator; the kernel then compiles the Mode-A text (RestirKernelOptions.modeBNeedsAreaLights). Evidence: bitwise
// equal chains and PT images in both modes on the every-other-endpoint fixture.
describe('U-M8-MODEB: without rect / disk lights, light mode B ≡ light mode A (bitwise)', () => {
  for (const preset of ['interactive', 'full-m6'] as const) {
    it(`ReSTIR ${preset}`, async () => {
      const base = { name: `noarea-${preset}`, scene: 'noarea' as const, preset, settings: { maxBounces: 3 }, frames: 4, motion: { from: 2, dx: 0.03, dyaw: 0.01 } };
      const a = await m8BitsCase({ ...base, lightMode: 'A' });
      const b = await m8BitsCase({ ...base, lightMode: 'B' });
      console.log(`[U-M8-MODEB] ${preset} A ${JSON.stringify(a)} B ${JSON.stringify(b)}`);
      expect(b).toEqual(a);
    }, 300_000);
  }
  it('PT batch', async () => {
    expect(await m8PtBits('noarea', 'B')).toBe(await m8PtBits('noarea', 'A'));
  }, 300_000);
});

describe('M8 P-4: the interactive kernel compiles the Mode-A text while no rect / disk light exists', () => {
  it('variant follows the light list at frame boundaries (current, previous and pending lists), history resets on a switch', async () => {
    const { RestirKernel } = await import('../../src/core/render/restir/kernel.ts');
    const { restirSettings } = await import('../../src/core/render/restir/presets.ts');
    const { gpuScene, allLightsScene } = await import('./restir-fixtures.ts');
    const full = allLightsScene();
    const noArea = full.lights.filter((l) => l.type !== 'rect' && l.type !== 'disk');
    const g = await gpuScene({ ...full, lights: noArea });
    const k = await RestirKernel.create(g.device, g.gpu, g.env, { settings: restirSettings('interactive', { maxBounces: 2 }), lightMode: 'B', modeBNeedsAreaLights: true, features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
    expect(k.m6Defines().RS_MODE_B).toBe(0);
    k.setLights(full.lights);                              // a rect light appears (pending until the next commit)
    expect(k.syncLightModeVariant()).toBe(true);
    expect(k.m6Defines().RS_MODE_B).toBe(1);
    k.lights.commit();                                     // the rect light is now in the current slot
    expect(k.syncLightModeVariant()).toBe(false);
    k.setLights(noArea);                                   // removed: the previous slot still holds it → stays B
    k.lights.commit();
    expect(k.syncLightModeVariant()).toBe(false);
    expect(k.m6Defines().RS_MODE_B).toBe(1);
    k.lights.commit();                                     // unchanged commit: both slots without area lights → A again
    expect(k.syncLightModeVariant()).toBe(true);
    expect(k.m6Defines().RS_MODE_B).toBe(0);
    const off = await RestirKernel.create(g.device, g.gpu, g.env, { settings: restirSettings('interactive', { maxBounces: 2 }), lightMode: 'B', features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
    expect(off.m6Defines().RS_MODE_B).toBe(1);             // validation callers keep the requested text
    k.destroy(); off.destroy(); g.destroy();
  }, 300_000);
});
