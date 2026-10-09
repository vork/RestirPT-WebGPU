// U-M8-BITS (docs/decisions/m8-perf.md §4; M8 Gate 0): optimisations that must not change results are bitwise the
// pre-M8 build (eda22be). VITE_M8_BITS_RECORD=1 prints the hashes instead of asserting (recorded once before any ReSTIR /
// PT shader change; see m8-bits.ts). PT batch on the same scenes (U-M8-PTBITS).
// perf2 WP-0 (docs/decisions/perf2-api.md, perf2-plan.md §0 rule 3): two golden tiers.
//   Anchor   every results-changing perf flag off — bit-equal forever. Runs with the bitwise subset of VITE_PERF_FLAGS
//            (a bitwise package proves itself here: VITE_PERF_FLAGS=RS_VIS_MERGE …); a flag whose registry entry lists
//            an exception compares the P9-excluded hash ('P9') or the reservoir + image fields only ('counters').
//   Shipped  every release flag on (RELEASE_PERF_FLAGS ∪ VITE_PERF_FLAGS). Re-recorded only when a results-changing item
//            lands, with a written justification and a diagnostic; while the release set holds no results-changing flag
//            the tier is the Anchor table (checked structurally, no second GPU run).
// The perf2 cases (CWBVH-forced variants, Sponza-lite) and the P9-excluded hashes were recorded on the unmodified perf2
// text (6d46432), twice, identical.
import { describe, expect, it } from 'vitest';
import { M8_BITS_CASES, PERF2_BITS_CASES, m8BitsCase, m8BitsCaseHashes, m8PtBits, type M8BitsScene } from './m8-bits.ts';
import type { LightMode } from '../../src/core/render/lights-gpu.ts';
import { PERF_FLAGS, RELEASE_PERF_FLAGS, bitwiseSubset, normalizePerfFlags, perfFlagsKey, type PerfFlagDef, type PerfFlagName, type PerfFlags } from '../../src/core/render/restir/perf-flags.ts';
import { testPerfFlags } from './restir-fixtures.ts';

const RECORD = String(import.meta.env?.VITE_M8_BITS_RECORD ?? '') === '1';
const FORCED = normalizePerfFlags(testPerfFlags());
const ANCHOR_FLAGS = bitwiseSubset(FORCED);
const SHIPPED_FLAGS = normalizePerfFlags({ ...RELEASE_PERF_FLAGS, ...FORCED });
/** Hash projection of a flag set's documented exceptions: drop the counter fields ('counters'); 'P9' selects the
 *  P9-excluded table. */
function exceptionsOf(f: PerfFlags): { p9: boolean; counters: boolean } {
  const ex = (Object.keys(f) as PerfFlagName[]).flatMap((n) => (PERF_FLAGS[n] as PerfFlagDef).except ?? []);
  return { p9: ex.includes('P9'), counters: ex.includes('counters') };
}
const project = (h: string[], counters: boolean) => (counters ? h.map((x) => x.split(':').slice(0, 2).join(':')) : h);
// Recorded on m8-perf at b724d4c + the CWBVH WGSL (BVH_CWBVH absent: the composed text of every validation pipeline is
// the eda22be text, U-M7-BITS), Chrome 154 / Metal, before any further ReSTIR / PT shader change.
const ANCHOR: Record<string, string[]> = {
  'all-interactive-B': ["2a0b7fd2:ae7201c8:0,0,0,0:1966,194,315,0,0,0,0,0,0,0","692d4fa1:fc9adf81:0,0,0,0:990,107,144,2358,1402,1007,0,0,0,0","808a662e:7dcc0ee5:0,0,0,0:1012,96,122,2382,1717,705,3207,85,0,0","b2fc32cc:65670b08:0,0,0,0:1030,117,153,2382,1795,632,3140,87,0,0","ab212fb9:c2b1ea07:0,0,0,0:1020,105,155,2379,1813,611,3120,79,0,0"],
  'all-full-m6-B': ["52ac52cd:fef05b2e:0,0,0,0:984,114,162,0,0,0,0,0,0,0","69150d98:aab451bb:0,0,0,0:990,106,148,2361,1309,1098,0,0,0,0","9bba10da:9716b98c:0,0,0,0:1002,105,130,2388,1734,682,3183,77,0,0","d72d84a8:aeffc51b:0,0,0,0:1026,115,156,2381,1947,484,2975,66,0,0"],
  'all-offline-m6-B': ["efa108a0:4dbabe62:0,0,0,0:6004,629,777,0,0,0,0,0,0,0"],
  'nm-interactive-B': ["0b047d69:ca30fe5f:0,0,0,0:4268,83,669,0,0,0,0,0,0,0","602e96ff:b006f9cc:0,0,0,0:2086,50,369,3755,2321,1554,0,0,0,0","e89834e8:0ee6b840:0,0,0,0:2050,29,337,3788,2777,1073,5139,1804,0,0","d10910e5:050811f9:0,0,0,0:2078,43,341,3754,2836,994,5026,1843,0,0"],
  'glass-interactive-B': ["89aea652:d143812b:0,0,0,0:3068,222,572,0,0,0,0,0,0,0","7b8b1e02:ef60a0e0:0,0,0,0:1494,112,306,3224,2041,1415,0,0,0,0","423d872a:aee98644:0,0,0,0:1422,91,279,3480,2556,1028,0,0,0,0","0383b5d1:793455ee:0,0,0,0:1416,123,274,3522,2724,860,0,0,0,0"],
  'alpha-full-m6-A': ["fd77c17f:cf6e26e0:0,0,0,0:806,0,171,0,0,0,0,0,0,0","9559508f:89f053d8:0,0,0,0:720,0,162,2027,1150,929,0,0,0,0","eaf8147e:08f0df3f:0,0,0,0:702,0,160,2067,1507,586,0,0,0,0"],
  // perf2 WP-0 cases (recorded on 6d46432, twice, identical)
  'all-interactive-B-cwbvh': ["81fcfb5e:5a4279cc:0,0,0,0:1966,194,315,0,0,0,0,0,0,0","95b6920f:a210cef3:0,0,0,0:990,107,144,2358,1402,1007,0,0,0,0","8ea9a546:93549e14:0,0,0,0:1012,96,122,2382,1717,705,3207,85,0,0","2e7968fa:7ea22368:0,0,0,0:1030,117,153,2382,1795,632,3140,87,0,0","a17daee3:2986959f:0,0,0,0:1020,105,155,2379,1813,611,3120,79,0,0"],
  'nm-interactive-B-cwbvh': ["950856ce:a4d307a6:0,0,0,0:4268,83,669,0,0,0,0,0,0,0","93c44220:cbfbf0a1:0,0,0,0:2086,50,369,3755,2321,1554,0,0,0,0","d1597625:64a09d80:0,0,0,0:2050,29,337,3788,2777,1073,5139,1804,0,0","c54a9833:f3d0e73e:0,0,0,0:2078,43,341,3754,2836,994,5026,1843,0,0"],
  'alpha-full-m6-A-cwbvh': ["fd77c17f:cf6e26e0:0,0,0,0:806,0,171,0,0,0,0,0,0,0","9559508f:89f053d8:0,0,0,0:720,0,162,2027,1150,929,0,0,0,0","eaf8147e:08f0df3f:0,0,0,0:702,0,160,2067,1507,586,0,0,0,0"],
  'sponza-lite-interactive-B': ["d2713e23:c191e335:0,0,0,0:18354,80,2156,0,0,0,0,0,0,0","8c1d201c:5a7b2fe8:0,0,0,0:9120,42,1997,5911,4681,2797,0,0,0,0","6c25b4e9:ca757621:0,0,0,0:9146,41,2250,8629,7172,2250,11877,5620,0,0"],
};
/** P9-excluded hashes of the same runs (reservoir plane P9 zeroed; image and counters as in ANCHOR). */
const ANCHOR_NOP9: Record<string, string[]> = {
  'all-interactive-B': ["300df948:ae7201c8:0,0,0,0:1966,194,315,0,0,0,0,0,0,0","0ae2ff4a:fc9adf81:0,0,0,0:990,107,144,2358,1402,1007,0,0,0,0","6a0debd2:7dcc0ee5:0,0,0,0:1012,96,122,2382,1717,705,3207,85,0,0","fb9712e9:65670b08:0,0,0,0:1030,117,153,2382,1795,632,3140,87,0,0","6a0df71f:c2b1ea07:0,0,0,0:1020,105,155,2379,1813,611,3120,79,0,0"],
  'all-full-m6-B': ["5c705f36:fef05b2e:0,0,0,0:984,114,162,0,0,0,0,0,0,0","4920a36f:aab451bb:0,0,0,0:990,106,148,2361,1309,1098,0,0,0,0","970d09c5:9716b98c:0,0,0,0:1002,105,130,2388,1734,682,3183,77,0,0","1eda9285:aeffc51b:0,0,0,0:1026,115,156,2381,1947,484,2975,66,0,0"],
  'all-offline-m6-B': ["d07da6f7:4dbabe62:0,0,0,0:6004,629,777,0,0,0,0,0,0,0"],
  'nm-interactive-B': ["92c8c890:ca30fe5f:0,0,0,0:4268,83,669,0,0,0,0,0,0,0","f50a1e27:b006f9cc:0,0,0,0:2086,50,369,3755,2321,1554,0,0,0,0","4069d47f:0ee6b840:0,0,0,0:2050,29,337,3788,2777,1073,5139,1804,0,0","3ce2f185:050811f9:0,0,0,0:2078,43,341,3754,2836,994,5026,1843,0,0"],
  'glass-interactive-B': ["0464eae2:d143812b:0,0,0,0:3068,222,572,0,0,0,0,0,0,0","936a814e:ef60a0e0:0,0,0,0:1494,112,306,3224,2041,1415,0,0,0,0","31adc152:aee98644:0,0,0,0:1422,91,279,3480,2556,1028,0,0,0,0","2b6904ed:793455ee:0,0,0,0:1416,123,274,3522,2724,860,0,0,0,0"],
  'alpha-full-m6-A': ["4a1db6ba:cf6e26e0:0,0,0,0:806,0,171,0,0,0,0,0,0,0","10ca9dd0:89f053d8:0,0,0,0:720,0,162,2027,1150,929,0,0,0,0","2ed60b6f:08f0df3f:0,0,0,0:702,0,160,2067,1507,586,0,0,0,0"],
  'all-interactive-B-cwbvh': ["30d373fe:5a4279cc:0,0,0,0:1966,194,315,0,0,0,0,0,0,0","cb17e52b:a210cef3:0,0,0,0:990,107,144,2358,1402,1007,0,0,0,0","ae87b019:93549e14:0,0,0,0:1012,96,122,2382,1717,705,3207,85,0,0","68af05b1:7ea22368:0,0,0,0:1030,117,153,2382,1795,632,3140,87,0,0","0cb2cc00:2986959f:0,0,0,0:1020,105,155,2379,1813,611,3120,79,0,0"],
  'nm-interactive-B-cwbvh': ["56acff25:a4d307a6:0,0,0,0:4268,83,669,0,0,0,0,0,0,0","b83fe3df:cbfbf0a1:0,0,0,0:2086,50,369,3755,2321,1554,0,0,0,0","d1ef2e60:64a09d80:0,0,0,0:2050,29,337,3788,2777,1073,5139,1804,0,0","a7cec4df:f3d0e73e:0,0,0,0:2078,43,341,3754,2836,994,5026,1843,0,0"],
  'alpha-full-m6-A-cwbvh': ["4a1db6ba:cf6e26e0:0,0,0,0:806,0,171,0,0,0,0,0,0,0","10ca9dd0:89f053d8:0,0,0,0:720,0,162,2027,1150,929,0,0,0,0","2ed60b6f:08f0df3f:0,0,0,0:702,0,160,2067,1507,586,0,0,0,0"],
  'sponza-lite-interactive-B': ["1084a771:c191e335:0,0,0,0:18354,80,2156,0,0,0,0,0,0,0","a8e3fda9:5a7b2fe8:0,0,0,0:9120,42,1997,5911,4681,2797,0,0,0,0","c2320e01:ca757621:0,0,0,0:9146,41,2250,8629,7172,2250,11877,5620,0,0"],
};
/** Shipped tier: = Anchor while RELEASE_PERF_FLAGS holds no results-changing flag. A package releasing one replaces the
 *  affected entries here (re-recorded with its flag on), with the justification and diagnostic in its decision note. */
// perf2 WP-7e released RS_PRIMARY_EXT (unbiased, renderer-only): this rig has no M1 primary, so the kernel takes its
// rs_vtrace → rs_primary_ext path, which reproduced every Anchor hash (full and P9-excluded) bit for bit when the tier
// was re-recorded (2026-10-09, all 10 cases): the Shipped table stays the Anchor table, now checked by a second GPU run.
// The flag's results change is app-only (the M1 primary's intersection: U-WP7E-VBUF, wp7e-primary-ext.gpu.test.ts).
const SHIPPED: Record<string, string[]> = { ...ANCHOR };
const SHIPPED_NOP9: Record<string, string[]> = { ...ANCHOR_NOP9 };
const GOLDEN_PT: Record<string, string> = {
  'all-B': '61082f13:0,0,0,0',
  'all-A': 'e0f14980:0,0,0,0',
  'nm_smooth-B': '915c361c:0,0,0,0',
  'glass-B': '4d7cee66:0,0,0,0',
  'alpha-A': '135260d9:0,0,0,0',
};
const PT_CASES: [M8BitsScene, LightMode][] = [['all', 'B'], ['all', 'A'], ['nm_smooth', 'B'], ['glass', 'B'], ['alpha', 'A']];

function tierCase(name: string, flags: PerfFlags, full: Record<string, string[]>, noP9: Record<string, string[]>, c: (typeof M8_BITS_CASES)[number]) {
  it(c.name, async () => {
    const h = await m8BitsCaseHashes(c, { perfFlags: flags });
    console.log(`[${name}] '${c.name}': ${JSON.stringify(h.full)},`);
    console.log(`[${name}:noP9] '${c.name}': ${JSON.stringify(h.noP9)},`);
    if (RECORD) return;
    const ex = exceptionsOf(flags);
    expect(project(h.noP9, ex.counters)).toEqual(project(noP9[c.name], ex.counters));
    if (!ex.p9) expect(project(h.full, ex.counters)).toEqual(project(full[c.name], ex.counters));
  }, 300_000);
}

describe(`U-M8-BITS (Anchor tier, flags: ${perfFlagsKey(ANCHOR_FLAGS) || 'none'}): ReSTIR chains (interactive and validation presets, light mode B) ≡ the pre-M8 build`, () => {
  for (const c of M8_BITS_CASES) tierCase('U-M8-BITS', ANCHOR_FLAGS, ANCHOR, ANCHOR_NOP9, c);
});

// perf2 WP-0: CWBVH-forced variants (MT for the interactive cases) and Sponza-lite (160×90, CWBVH, textures, normal
// maps, alpha; the interactive kernel options), Anchor tier
describe(`U-M8-BITS perf2 cases (Anchor tier, flags: ${perfFlagsKey(ANCHOR_FLAGS) || 'none'}): CWBVH and Sponza-lite`, () => {
  for (const c of PERF2_BITS_CASES) tierCase('U-M8-BITS', ANCHOR_FLAGS, ANCHOR, ANCHOR_NOP9, c);
});

describe(`U-M8-BITS (Shipped tier, flags: ${perfFlagsKey(SHIPPED_FLAGS) || 'none'})`, () => {
  if (perfFlagsKey(SHIPPED_FLAGS) === perfFlagsKey(ANCHOR_FLAGS)) {
    it('release set adds nothing over the Anchor run: Shipped goldens = Anchor goldens', () => {
      expect(SHIPPED).toEqual(ANCHOR);
      expect(SHIPPED_NOP9).toEqual(ANCHOR_NOP9);
    });
  } else {
    for (const c of [...M8_BITS_CASES, ...PERF2_BITS_CASES]) tierCase('U-M8-BITS:shipped', SHIPPED_FLAGS, SHIPPED, SHIPPED_NOP9, c);
  }
});

// M8 P-7 (m8-perf.md §8): plane-major reservoirs (RS_RES_SOA, the interactive kernel) give the AoS hashes bit for bit
// (readReservoirs returns AoS words in both layouts).
describe('U-M8-BITS (SoA): plane-major reservoirs ≡ the pre-M8 build', () => {
  for (const c of M8_BITS_CASES) {
    it(c.name, async () => {
      const h = await m8BitsCase(c, { resLayout: 'soa', perfFlags: ANCHOR_FLAGS });
      if (RECORD) return;
      const ex = exceptionsOf(ANCHOR_FLAGS);
      if (!ex.p9) expect(project(h, ex.counters)).toEqual(project(ANCHOR[c.name], ex.counters));
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
      const a = await m8BitsCase({ ...base, lightMode: 'A' }, { perfFlags: ANCHOR_FLAGS });
      const b = await m8BitsCase({ ...base, lightMode: 'B' }, { perfFlags: ANCHOR_FLAGS });
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

describe('perf2 perf-flag plumbing (docs/decisions/perf2-api.md)', () => {
  it('validation kernels carry no flag; flags reach defines / customDefines / variantKey; setPerfFlags switches the variant', async () => {
    const { RestirKernel } = await import('../../src/core/render/restir/kernel.ts');
    const { restirSettings } = await import('../../src/core/render/restir/presets.ts');
    const { gpuScene, allLightsScene } = await import('./restir-fixtures.ts');
    const g = await gpuScene(allLightsScene());
    const o = { settings: restirSettings('interactive', { maxBounces: 2 }), lightMode: 'B' as const, features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures };
    const v = await RestirKernel.create(g.device, g.gpu, g.env, o);
    expect(v.perfDefines()).toEqual({});
    expect(v.variantKey()).not.toContain('|');
    for (const n of Object.keys(PERF_FLAGS)) expect(Object.keys(v.defines('rs_initial'))).not.toContain(n);
    const f = await RestirKernel.create(g.device, g.gpu, g.env, { ...o, perfFlags: 'RS_VIS_MERGE,CW_TRI_BUDGET=2' });
    expect(f.defines('rs_spatial_shift')).toMatchObject({ RS_VIS_MERGE: 1, CW_TRI_BUDGET: 2 });
    expect(f.customDefines({})).toMatchObject({ RS_VIS_MERGE: 1, CW_TRI_BUDGET: 2 });
    expect(f.variantKey()).toBe(`${v.variantKey()}|CW_TRI_BUDGET=2,RS_VIS_MERGE`);
    expect(f.isPrepared()).toBe(true);
    expect(f.setPerfFlags('CW_TRI_BUDGET=2,RS_VIS_MERGE')).toBe(false);   // same set, other order: no switch
    expect(f.setPerfFlags({})).toBe(true);
    expect(f.isPrepared()).toBe(false);                                     // a new variant: prepare() before the next frame
    expect(f.variantKey()).toBe(v.variantKey());
    await f.prepare();
    expect(f.isPrepared()).toBe(true);
    v.destroy(); f.destroy(); g.destroy();
  }, 300_000);
});
