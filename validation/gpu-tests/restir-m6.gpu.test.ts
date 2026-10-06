// M6 Gate-0 GPU suite (restir-m6-api.md §7): U-M5-BITS (temporal chains with every M6 feature off ≡ the M5 build) and
// the M6 feature tests. VITE_M6_BITS_RECORD=1 prints the hashes instead of asserting (used once on main f5c23fd).
import { describe, expect, it } from 'vitest';
import { M5_BITS_CASES, m5BitsCase } from './restir-m6-bits.ts';
import { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import { restirSettings, type RestirPresetName, type RestirSettings } from '../../src/core/render/restir/presets.ts';
import { TEMPORAL_PASSES } from '../../src/core/render/restir/resources.ts';
import { SPATIAL_PASSES } from '../../src/core/render/restir/stage-spatial.ts';
import { allLightsScene, boxCamera, gpuScene, ptImage, restirRig } from './restir-fixtures.ts';
import type { SceneData } from '../../src/core/scene/types.ts';
import type { RestirCounters } from '../../src/core/render/restir/kernel.ts';
import type { LightMode } from '../../src/core/render/lights-gpu.ts';

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

// ------------------------------------------------------------------------------------------------ variant compile smoke


describe('M6 pipeline variants compile (RS_RIS_NEE, RS_MODE_B, RS_DUAL_MV, RS_DUPMAP, RS_PLANT_T2)', () => {
  const cases: { name: string; preset: 'full' | 'offline'; s: Record<string, unknown>; mode: LightMode; dump?: boolean; extra?: Record<string, number> }[] = [
    { name: 'RIS', preset: 'full', s: { risNee: true }, mode: 'A' },
    { name: 'Mode B', preset: 'full', s: {}, mode: 'B', dump: true },
    { name: 'Mode A′', preset: 'offline', s: {}, mode: 'A′' },
    { name: 'RIS + Mode B + dual MV + dupmap', preset: 'full', s: { risNee: true, dualMv: true, dupmap: true, pairing: 'gauss' }, mode: 'B', dump: true },
    { name: 'U8-4 plant (RS_PLANT_T2)', preset: 'offline', s: { plant: { u8T2: true } }, mode: 'A' },
  ];
  for (const c of cases) {
    it(c.name, async () => {
      const g = await gpuScene(allLightsScene());
      const k = await RestirKernel.create(g.device, g.gpu, g.env, {
        settings: restirSettings(c.preset, c.s as never), lightMode: c.mode, features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures,
        instrumentation: { dumpCandidates: c.dump },
      });
      k.setView({ camera: boxCamera(), width: 16, height: 16, runSeed: 1 });
      await k.prepare();
      for (const n of [...SPATIAL_PASSES, ...(k.settings.temporal ? TEMPORAL_PASSES : [])]) await k.pipeline(n);
      k.destroy(); g.destroy();
    });
  }
});

// ------------------------------------------------------------------------------------------------ functional means (smoke)

/** Per-tile (16×16) and global luminance means of B batches; Welch z of ReSTIR vs PT per tile and globally. */
const ptCache = new Map<string, Float64Array[]>();
async function meanCompare(scene: SceneData, o: { preset: RestirPresetName; settings?: Partial<RestirSettings>; lightMode?: LightMode; frames: number; B: number; maxBounces: number; ptSpp: number; key: string }) {
  const W = 64, H = 64, T = 16;
  const nT = (W / T) * (H / T);
  const tiles = (img: Float32Array) => {
    const t = new Float64Array(nT + 1);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const l = 0.2126 * img[3 * i] + 0.7152 * img[3 * i + 1] + 0.0722 * img[3 * i + 2];
      t[((y / T) | 0) * (W / T) + ((x / T) | 0)] += l / (T * T);
      t[nT] += l / (W * H);
    }
    return t;
  };
  const rs: Float64Array[] = [], pt: Float64Array[] = [];
  const rig = await restirRig(scene, W, H, { preset: o.preset, settings: { maxBounces: o.maxBounces, ...(o.settings ?? {}) }, lightMode: o.lightMode, seed: 101 });
  for (let b = 0; b < o.B; b++) rs.push(tiles((await rig.frames(o.frames, b * o.frames, 4)).mean));
  const counters = await rig.kernel.readCounters(false);
  rig.destroy();
  const pk = `${o.key}:${o.lightMode ?? 'A'}:${o.maxBounces}:${o.ptSpp}`;
  if (!ptCache.has(pk)) {
    const a: Float64Array[] = [];
    for (let b = 0; b < 16; b++) a.push(tiles((await ptImage(scene, W, H, o.ptSpp, { maxBounces: o.maxBounces, lightMode: o.lightMode, seed: 7000 + b })).mean));
    ptCache.set(pk, a);
  }
  pt.push(...ptCache.get(pk)!);
  const z: number[] = [];
  for (let k = 0; k <= nT; k++) {
    const st = (a: Float64Array[]) => { const m = a.reduce((s, x) => s + x[k], 0) / a.length; const v = a.reduce((s, x) => s + (x[k] - m) ** 2, 0) / (a.length - 1); return { m, v }; };
    const A = st(rs), Bp = st(pt);
    z.push((A.m - Bp.m) / Math.sqrt(A.v / rs.length + Bp.v / pt.length + 1e-30));
  }
  const g = z[nT];
  const mean = (a: Float64Array[]) => a.reduce((s, x) => s + x[nT], 0) / a.length;
  const rel = (mean(rs) - mean(pt)) / mean(pt);
  return { zGlobal: g, zTileMax: Math.max(...z.slice(0, nT).map(Math.abs)), rel, counters };
}

function noErrors(c: RestirCounters): void {
  const r = c.rsc;
  expect(r.candNonFinite + r.shiftNonFinite + r.pendingLeft + r.slotMismatch + r.wNonFinite + r.bvhOverflow + r.bvhItercap).toBe(0);
  for (const q of c.queues) expect(q.overflow).toBe(0);
}

describe('M6 functional smoke: ReSTIR ≡ PT (global z ≤ 4, 16 tiles z ≤ 4.5) with RIS-NEE and Mode B', () => {
  const F = Number(import.meta.env?.VITE_M6_SMOKE_FRAMES ?? 256);
  const cases: { name: string; scene: () => SceneData; preset: RestirPresetName; settings?: Partial<RestirSettings>; mode?: LightMode; frames: number; mb: number }[] = [
    { name: 'control: M5 offline (features off)', scene: allLightsScene, preset: 'offline', settings: { trees: 4 }, frames: F / 8, mb: 2 },
    { name: 'RIS-NEE initial (all light types + env)', scene: allLightsScene, preset: 'initial', settings: { risNee: true }, frames: F, mb: 2 },
    { name: 'RIS-NEE offline (spatial, M(1) = 32 in the shifts)', scene: allLightsScene, preset: 'offline', settings: { risNee: true, trees: 4 }, frames: F / 8, mb: 2 },
    { name: 'Mode B initial (crossings, x_quads + env)', scene: allLightsScene, preset: 'initial', mode: 'B', frames: F, mb: 2 },
    { name: 'Mode B offline (crossing shifts)', scene: allLightsScene, preset: 'offline', settings: { trees: 4 }, mode: 'B', frames: F / 8, mb: 2 },
    { name: 'Mode B + RIS offline, gauss maps', scene: allLightsScene, preset: 'offline', settings: { trees: 4, risNee: true, pairing: 'gauss' }, mode: 'B', frames: F / 8, mb: 2 },
    { name: 'Mode A′ initial', scene: allLightsScene, preset: 'initial', mode: 'A′', frames: F, mb: 2 },
  ];
  for (const c of cases) {
    it(c.name, async () => {
      const r = await meanCompare(c.scene(), { preset: c.preset, settings: c.settings, lightMode: c.mode, frames: c.frames, B: 8, maxBounces: c.mb, ptSpp: 4 * F, key: 'all' });
      console.log(`[M6-smoke] ${c.name}: rel ${(r.rel * 100).toFixed(3)} %, z ${r.zGlobal.toFixed(2)}, tile |z|max ${r.zTileMax.toFixed(2)}`);
      noErrors(r.counters);
      expect(Math.abs(r.zGlobal)).toBeLessThan(4);
      expect(r.zTileMax).toBeLessThan(4.5);
    }, 600_000);
  }
});
