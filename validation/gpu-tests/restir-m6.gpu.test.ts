// M6 Gate-0 GPU suite (restir-m6-api.md §7): U-M5-BITS (temporal chains with every M6 feature off ≡ the M5 build) and
// the M6 feature tests. VITE_M6_BITS_RECORD=1 prints the hashes instead of asserting (used once on main f5c23fd).
import { describe, expect, it } from 'vitest';
import { M5_BITS_CASES, m5BitsCase } from './restir-m6-bits.ts';
import { TEMPORAL_PASSES } from '../../src/core/render/restir/resources.ts';
import { SPATIAL_PASSES } from '../../src/core/render/restir/stage-spatial.ts';
import { allLightsScene, boxCamera, gpuScene, ptImage, restirRig } from './restir-fixtures.ts';
import type { SceneData } from '../../src/core/scene/types.ts';
import type { RestirCounters } from '../../src/core/render/restir/kernel.ts';
import type { LightMode } from '../../src/core/render/lights-gpu.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { arenaM6Layout, RS_M6_CONSTS as K6 } from '../../src/core/render/restir/layout.ts';
import { JITTER_NONE } from '../../src/core/render/frame-uniforms.ts';
import { bitFixtureScene as bitScene } from './restir-fixtures.ts';
import { synthEnvData } from './env-fixtures.ts';
import { t3Scene } from '../scenes/make-m4.ts';
import { testPipeline, storageBuffer, readU32 } from './restir-shift-fixtures.ts';
import { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import { restirSettings, type RestirPresetName, type RestirSettings } from '../../src/core/render/restir/presets.ts';

const RECORD = String(import.meta.env?.VITE_M6_BITS_RECORD ?? '') === '1';
// Recorded on the M5 build (main f5c23fd, Chrome 154 / Metal) with restir-m6-bits.ts. Re-recorded for restir-m6-api.md
// Changelog M6-10 (incoming direction after a delta event): both fixtures contain the box's roughness-0 mirror, so the
// reservoir and image hashes change; every per-frame counter is unchanged.
const GOLDEN_M5: Record<string, string[]> = {
  'xq-full-static': [
    'df37a7aa:fb352eda:0,0,0,0:844,66,200,0,0,0,0,0,0,0',
    '4cb898d7:aaef6834:0,0,0,0:814,66,161,1968,1220,1050,0,0,0,0',
    'a44ebd6a:5b26ce4a:0,0,0,0:844,76,143,2295,1667,729,0,0,0,0',
    'd2a5ab45:c0dad779:0,0,0,0:770,49,130,2379,1921,499,0,0,0,0',
    '68612d3c:95ce0f4e:0,0,0,0:808,63,126,2397,1998,429,0,0,0,0',
  ],
  'all-full-motion': [
    '68bd296a:5725226b:0,0,0,0:844,84,159,0,0,0,0,0,0,0',
    'bd64b7bc:0d82871f:0,0,0,0:814,100,145,2270,1243,1148,0,0,0,0',
    'cd328e58:4193e146:0,0,0,0:844,89,119,2385,1680,750,0,0,0,0',
    '8765df8d:4f53767a:0,0,0,0:798,56,121,2390,1894,544,3036,318,0,0',
    '96119bfb:3d162dba:0,0,0,0:806,64,131,2391,1977,462,2956,312,0,0',
    'aeb85905:4c571419:0,0,0,0:834,75,138,2364,2031,411,2896,294,0,0',
  ],
  'xq-interactive-M5': [
    '4e17f659:88b02baf:0,0,0,0:1764,132,357,0,0,0,0,0,0,0',
    '98b7ac8e:bb9e13e4:0,0,0,0:814,66,163,2030,1298,990,0,0,0,0',
    'a17ea3e3:dacc530f:0,0,0,0:844,79,150,2305,1697,705,3090,247,0,0',
    'cbf9cd9f:1a73c7d8:0,0,0,0:770,52,125,2381,1883,539,2993,225,0,0',
    '15f43a94:8c9f4201:0,0,0,0:808,64,132,2397,1916,514,2982,220,0,0',
  ],
  'all-talbot': [
    'ec661ebf:0aeb3b32:0,0,0,0:844,71,170,0,0,0,0,0,0,0',
    '29ceb103:bac59c38:0,0,0,0:814,98,150,2204,1234,2203,0,0,0,0',
    '73262871:5bbcc620:0,0,0,0:844,90,120,2367,1666,2186,4716,139,0,0',
    '84a6e8e5:2a3fae3d:0,0,0,0:770,48,117,2416,1900,2185,4733,143,0,0',
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
async function meanCompare(scene: SceneData, o: { preset: RestirPresetName; settings?: Partial<RestirSettings>; lightMode?: LightMode; ptMode?: LightMode; frames: number; B: number; maxBounces: number; ptSpp: number; key: string }) {
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
  const ptMode = o.ptMode ?? o.lightMode;
  const pk = `${o.key}:${ptMode ?? 'A'}:${o.maxBounces}:${o.ptSpp}`;
  if (!ptCache.has(pk)) {
    const a: Float64Array[] = [];
    for (let b = 0; b < 16; b++) a.push(tiles((await ptImage(scene, W, H, o.ptSpp, { maxBounces: o.maxBounces, lightMode: ptMode, seed: 7000 + b })).mean));
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

/** C0e box with the roughness-0 mirror replaced by a Lambert face: no singular lobe, so A ≡ B ≡ A′ (U9, math.md#mis). */
function deltaFreeBox(): SceneData {
  const s = bitScene('c0e');
  s.materials = s.materials.map((m, i) => (i === 4 ? { ...m, model: 'v1', v1: { diffuse: [0.5, 0.5, 0.5], glossy: [0, 0, 0], roughness: 0.5, mix: 0 } } : m));
  return s;
}

function noErrors(c: RestirCounters): void {
  const r = c.rsc;
  expect(r.candNonFinite + r.shiftNonFinite + r.pendingLeft + r.slotMismatch + r.wNonFinite + r.bvhOverflow + r.bvhItercap).toBe(0);
  for (const q of c.queues) expect(q.overflow).toBe(0);
}

describe('M6 functional smoke: ReSTIR ≡ PT (global z ≤ 4, 16 tiles z ≤ 4.5) with RIS-NEE and Mode B', () => {
  const F = Number(import.meta.env?.VITE_M6_SMOKE_FRAMES ?? 256);
  const cases: { name: string; scene: () => SceneData; preset: RestirPresetName; settings?: Partial<RestirSettings>; mode?: LightMode; ptMode?: LightMode; key?: string; frames: number; mb: number }[] = [
    { name: 'U-RR-M6: RIS-NEE + RR (initial-rr, W_NEE / ∏q)', scene: allLightsScene, preset: 'initial-rr', settings: { risNee: true }, frames: F, mb: 2 },
    { name: 'U9-R: Mode B ReSTIR offline ≡ Mode A PT (delta-free c0e box)', scene: deltaFreeBox, preset: 'offline', settings: { trees: 4 }, mode: 'B', ptMode: 'A', key: 'c0e-df', frames: F / 8, mb: 2 },
    { name: 'U9-R: Mode A′ ReSTIR offline ≡ Mode A PT (delta-free c0e box)', scene: deltaFreeBox, preset: 'offline', settings: { trees: 4 }, mode: 'A′', ptMode: 'A', key: 'c0e-df', frames: F / 8, mb: 2 },
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
      const r = await meanCompare(c.scene(), { preset: c.preset, settings: c.settings, lightMode: c.mode, ptMode: c.ptMode, frames: c.frames, B: 8, maxBounces: c.mb, ptSpp: 4 * F, key: c.key ?? 'all' });
      console.log(`[M6-smoke] ${c.name}: rel ${(r.rel * 100).toFixed(3)} %, z ${r.zGlobal.toFixed(2)}, tile |z|max ${r.zTileMax.toFixed(2)}`);
      noErrors(r.counters);
      expect(Math.abs(r.zGlobal)).toBeLessThan(4);
      expect(r.zTileMax).toBeLessThan(4.5);
    }, 600_000);
  }
});

// ------------------------------------------------------------------------------------------------ U4 tiles, U1-M, U9-R, U10-B, U-DMV-1


/** Regularised upper incomplete gamma Q(a, x) (χ² survival with 2a dof at 2x): series / continued fraction. */
function gammaQ(a: number, x: number): number {
  const lg = (z: number) => { const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]; let y = z; const t = z + 5.5 - (z + 0.5) * Math.log(z + 5.5); let s = 1.000000000190015; for (const ci of c) s += ci / ++y; return -t + Math.log(2.5066282746310005 * s / z); };
  if (x < a + 1) { let sum = 1 / a, del = sum, ap = a; for (let n = 0; n < 1000; n++) { ap++; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-12) break; } return 1 - sum * Math.exp(-x + a * Math.log(x) - lg(a)); }
  let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (let i = 1; i < 1000; i++) { const an = -i * (i - a); b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300; c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-12) break; }
  return Math.exp(-x + a * Math.log(x) - lg(a)) * h;
}

describe('U4-tiles: light-tile entries are i.i.d. draws of the realized pmf (per member), differ across members and frames', () => {
  it('allLightsScene (5 analytic + emissive triangles + env), E = 4, χ² p > 1e-4 per member and pooled', async () => {
    const rig = await restirRig(allLightsScene(), 32, 32, { preset: 'initial', settings: { risNee: true }, members: 4, seed: 41 });
    const k = rig.kernel;
    const st = k.lights.state, slot = st.curSlot;
    const pmf = Array.from({ length: slot.nEntries }, (_, i) => new Float32Array(st.records.buffer, 4 * (slot.pmfOff + i), 1)[0]);
    const a = k.resources.alloc;
    const lay = arenaM6Layout(a.atlasW * a.atlasH, a.slots, !!a.temporal, a.m6!);
    const tilesWords = a.members * K6.RS_TILES * K6.RS_TILE_SIZE;
    const read = async () => new Uint32Array(await readBuffer(rig.g.device, k.resources.arena, 4 * tilesWords, 256 + 4 * lay.tiles));
    await rig.frames(1, 5);
    const f5 = await read();
    await rig.frames(1, 6);
    const f6 = await read();
    const chi = (w: Uint32Array) => {
      const n = new Float64Array(pmf.length);
      for (const e of w) n[e]++;
      let x2 = 0, dof = -1;
      for (let i = 0; i < pmf.length; i++) { if (pmf[i] <= 0) { expect(n[i]).toBe(0); continue; } const E = pmf[i] * w.length; x2 += (n[i] - E) ** 2 / E; dof++; }
      return gammaQ(dof / 2, x2 / 2);
    };
    const per = K6.RS_TILES * K6.RS_TILE_SIZE;
    for (let m = 0; m < 4; m++) {
      const p = chi(f5.subarray(m * per, (m + 1) * per));
      console.log(`[U4-tiles] member ${m}: χ² p = ${p.toExponential(3)}`);
      expect(p).toBeGreaterThan(1e-4);
    }
    expect(chi(f5)).toBeGreaterThan(1e-4);
    let same01 = 0, same56 = 0;
    for (let i = 0; i < per; i++) { if (f5[i] === f5[per + i]) same01++; if (f5[i] === f6[i]) same56++; }
    const pp = pmf.reduce((s, x) => s + x * x, 0);   // P(two independent draws coincide)
    console.log(`[U4-tiles] coincidence member 0 vs 1: ${(same01 / per).toFixed(4)}, frame 5 vs 6: ${(same56 / per).toFixed(4)} (independent: ${pp.toFixed(4)})`);
    expect(Math.abs(same01 / per - pp)).toBeLessThan(0.01);
    expect(Math.abs(same56 / per - pp)).toBeLessThan(0.01);
    rig.destroy();
  });
});

const U1M_WGSL = `
#include "restir/endpoint.wgsl"
#include "restir/m6-types.wgsl"
@group(2) @binding(0) var<storage, read_write> out: array<atomic<u32>>;
@compute @workgroup_size(64) fn u1m(@builtin(global_invocation_id) gid: vec3u) {
  let h = pcg4d(vec4u(gid.x, 0x77u, 0x31u, 0x9u));
  let p1 = exp2(u32_to_unit(h.x) * 30.0 - 15.0);
  let p2 = exp2(u32_to_unit(h.y) * 30.0 - 15.0);
  var ls = light_sample_none();
  ls.valid = true; ls.p1 = p1; ls.q = p1; ls.kind = LT_TRI; ls.analytic = false;
  for (var B = 1u; B <= 3u; B++) {
    let w1 = nee_mis_w1(ls, rs_p2m(p2, B), B);
    let w2 = mis_w2(p1, rs_p2m(p2, B), B);
    let M = rs_mis_M(B);
    let ref1 = M * p1 / (M * p1 + p2);
    if (!(abs(w1 + w2 - 1.0) < 1e-6)) { atomicAdd(&out[0], 1u); }
    if (!(abs(w1 - ref1) <= 1e-6 * max(ref1, 1e-30) + 1e-7)) { atomicAdd(&out[1], 1u); }
    if (B == 1u && M != 32.0) { atomicAdd(&out[2], 1u); }
    if (B != 1u && M != 1.0) { atomicAdd(&out[3], 1u); }
  }
  var ld = ls; ld.isDelta = true; ld.kind = LT_POINT; ld.analytic = true;
  if (nee_mis_w1(ld, rs_p2m(p2, 1u), 1u) != 1.0) { atomicAdd(&out[4], 1u); }
  atomicAdd(&out[5], 1u);
}`;

describe('U1-M: ω1 (NEE-time) + ω2 (hit-time) = 1 with M(1) = 32 realised as p2/M; ω1 ≡ 1 for delta lights', () => {
  it('2^20 random (p1, p2) over 30 decades, B ∈ {1, 2, 3}', async () => {
    const rig = await restirRig(allLightsScene(), 16, 16, { preset: 'initial', settings: { risNee: true } });
    const k = rig.kernel;
    const tp = await testPipeline(k, 'u1m', U1M_WGSL, 'u1m', 1, {});
    const out = storageBuffer(rig.g.device, 64);
    await tp.run([out], [16384, 1]);
    const w = await readU32(rig.g.device, out);
    console.log(`[U1-M] partition ${w[0]}, vs M p1/(M p1 + p2) ${w[1]}, M(1) ≠ 32 ${w[2]}, M(B≥2) ≠ 1 ${w[3]}, delta ω1 ≠ 1 ${w[4]} of ${w[5]}`);
    expect([w[0], w[1], w[2], w[3], w[4]]).toEqual([0, 0, 0, 0, 0]);
    expect(w[5]).toBe(1 << 20);
    rig.destroy();
  });
});

describe('U10-B: crossings consume nothing: the path tree visits the same vertices in Mode A and Mode B (and A′)', () => {
  it('x_quads 64², candidate dumps of tree 0: per pixel the base vertex chain x₁…x₈ is identical; Mode B adds BSDF_ANALYTIC candidates', async () => {
    const chains = async (mode: LightMode) => {
      const rig = await restirRig(bitScene('x_quads'), 64, 64, { preset: 'initial', settings: { maxBounces: 4 }, lightMode: mode, dumpCandidates: true, seed: 5 });
      await rig.frames(1, 3);
      const d = await rig.kernel.readCandidateDump();
      const P = 64 * 64;
      const chain: string[] = [], ana: number[] = [];
      for (let ai = 0; ai < P; ai++) {
        const n = d[P * 32 * 48 + ai];
        let longest = '', nAna = 0;
        for (let c = 0; c < n; c++) {
          const base = (ai * 32 + c) * 48;
          if (((d[base + 6] >> 8) & 3) === 2) nAna++;
          const prims = Array.from(d.subarray(base + 40, base + 48)).filter((x) => x !== 0xFFFFFFFF).join(',');
          if (prims.length > longest.length) longest = prims;
        }
        chain.push(longest); ana.push(nAna);
      }
      rig.destroy();
      return { chain, ana };
    };
    const A = await chains('A'), B = await chains('B'), Ap = await chains('A′');
    let diff = 0, diffAp = 0;
    for (let i = 0; i < A.chain.length; i++) { if (!A.chain[i].startsWith(B.chain[i]) && !B.chain[i].startsWith(A.chain[i])) diff++; if (A.chain[i] !== Ap.chain[i] && !A.chain[i].startsWith(Ap.chain[i]) && !Ap.chain[i].startsWith(A.chain[i])) diffAp++; }
    const nAnaA = A.ana.reduce((s, x) => s + x, 0), nAnaB = B.ana.reduce((s, x) => s + x, 0);
    console.log(`[U10-B] chain mismatches A/B ${diff}, A/A′ ${diffAp}; BSDF_ANALYTIC candidates A ${nAnaA}, B ${nAnaB}, A′ ${Ap.ana.reduce((s, x) => s + x, 0)}`);
    expect(diff).toBe(0);
    expect(diffAp).toBe(0);
    expect(nAnaA).toBe(0);
    expect(nAnaB).toBeGreaterThan(10);
  });
});

describe('U-DMV-1: dual motion vectors pick q′ from the G-buffers only (sample-independent) and find more history under motion', () => {
  it('ixs-like camera pan over x_quads: q′ identical for two light powers; RSC_T_DUAL > 0; disoccluded fraction lower than without', async () => {
    const env = synthEnvData(64, 32);
    const run = async (dual: boolean, powerScale: number) => {
      const t3 = t3Scene('t3_cases_256', env);
      const scene = t3.scene;
      scene.lights = scene.lights.map((l) => ({ ...l, power: l.power * powerScale }));
      const rig = await restirRig(scene, 96, 96, { preset: 'full', settings: { dualMv: dual, maxBounces: 2 }, jitterMode: JITTER_NONE, seed: 9, cam: { camToWorld: t3.camera.matrix, yfov: t3.camera.yfov } });
      const k = rig.kernel;
      await k.prepare();
      let qp: Uint32Array | undefined;
      let dualN = 0, disocc = 0;
      for (let t = 0; t < 6; t++) {
        const cam = { camToWorld: [...t3.camera.matrix], yfov: t3.camera.yfov };
        cam.camToWorld[12] += 0.08 * t;
        k.advance({ t, camera: cam, lights: scene.lights });
        k.beginSubmit();
        const enc = rig.g.device.createCommandEncoder();
        for (const u of k.frameUnits(t, { accum: rig.accum, counters: rig.counters })) u.encode(enc);
        rig.g.device.queue.submit([enc.finish()]);
        await rig.g.device.queue.onSubmittedWorkDone();
        const c = await k.readCounters(true);
        dualN += c.rsc.tDual; disocc += c.rsc.tDisocc;
        if (t === 5) qp = (await k.readTemporalState()).filter((_, i) => i % 20 === 8);
      }
      rig.destroy();
      return { qp: qp!, dualN, disocc };
    };
    const a = await run(true, 1), b = await run(true, 1.7), off = await run(false, 1);
    let same = 0;
    for (let i = 0; i < a.qp.length; i++) if (a.qp[i] === b.qp[i]) same++;
    console.log(`[U-DMV-1] q′ identical ${same}/${a.qp.length}; dual picks ${a.dualN}; disoccluded with dual ${a.disocc}, without ${off.disocc}`);
    expect(same).toBe(a.qp.length);
    expect(a.dualN).toBeGreaterThan(0);
    expect(a.disocc).toBeLessThan(off.disocc);
  });
});
