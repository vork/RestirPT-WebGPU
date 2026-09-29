// WP-E (restir-api.md §6.3/§6.4): the M4 gate configuration and its pure helpers — scene list of §6.3, seeds, suite
// FWER unit count, the sizing rule (PLAN §7.3: δ/(t + z), joint PT/ReSTIR allocation, 64² enlargement, never δ), the T16
// assertions, the submit packer of BatchAccumulator.runBatchUnits (§4.4), the chunking choice and the cache-key closures.
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { UnitPacker, unitKind, type BatchWorkUnit } from '../../src/core/render/batch-accumulator.ts';
import { chooseChunking } from '../../src/core/render/restir/batch-runner.ts';
import { RESTIR_PRESETS } from '../../src/core/render/restir/presets.ts';
import { M3C_G2 } from '../../validation/harness/gate-m3c.ts';
import {
  AA_PKG, CRIT2022_PKG, ENSEMBLE_UNIT, M4_RUNGS, M4_SCENES, PLANTS, SEEDS, aggregateSide, ensembleShape, niceCeil, normInv, nUnits,
  sizeScene, sizingTarget, t16Problems, tInv, tsClosure, uVector, wgslClosure, type PilotSide,
} from '../../validation/harness/gate-m4.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));

describe('scene list and units (restir-api.md §6.3)', () => {
  it('exactly the §6.3 packages; committed ones exist, env ones are make-m3c.ts packages', () => {
    expect(M4_SCENES.map((s) => s.pkg)).toEqual([
      'cornell_i_512', 'ii_cornell_point_512', 'iii_spot_grazing_512', 'iv_emissive_mesh_512', 'v_glossy_v1_sharp_512', 'v_glossy_v1_512',
      'v_glossy_v2_512', 'vi_glass_mirror_A_512', 'x_many_lights_512', 'xi_contact_512', 'xii_alpha_foliage_512', 'c0q_openbox_b13_256',
      'c0r_irradiance_256', 'c0r_mirror_256', 'xiii_spheres_512x256', 'xiv_overcast_b1_512', 'xiv_overcast_b3_512', 'xiv_overcast_b7_512',
      'xiv_overcast_rect_b3_512', 'xiv_kloof_b3_512', 'xiv_kloof_rect_b3_512',
    ]);
    const m3c = new Set(M3C_G2.map((s) => s.pkg));
    for (const s of M4_SCENES) {
      if (s.env) expect(m3c.has(s.pkg), s.pkg).toBe(true);
      else expect(existsSync(path.join(ROOT, 'validation/scenes', s.pkg, 'scene.json')), s.pkg).toBe(true);
    }
    // heavy-tail tier: (xiii) and the two kloofendal scenes (§6.3)
    expect(M4_SCENES.filter((s) => s.tier === 'heavy-tail').map((s) => s.pkg)).toEqual(['xiii_spheres_512x256', 'xiv_kloof_b3_512', 'xiv_kloof_rect_b3_512']);
  });

  it('rungs map to the presets of §4.6 (3.1 no RR, 3.1b RR from bounce 1, 3.2 offline S 32 / 3 rounds / 6 slots / R 10, no RR)', () => {
    expect(M4_RUNGS.map((r) => [r.id, r.preset])).toEqual([['3.1', 'initial'], ['3.1b', 'initial-rr'], ['3.2', 'offline']]);
    expect(RESTIR_PRESETS.initial).toMatchObject({ trees: 1, rounds: 0, rr: false });
    expect(RESTIR_PRESETS['initial-rr']).toMatchObject({ trees: 1, rounds: 0, rr: true, rrMinBounces: 1 });
    expect(RESTIR_PRESETS.offline).toMatchObject({ trees: 32, rounds: 3, slots: 6, diskRadius: 10, rr: false });
    expect(RESTIR_PRESETS.criteria2022).toMatchObject({ trees: 32, rounds: 3, criteria: '2022' });
  });

  it('special units, plants and A/A are on the §6.3 scenes', () => {
    expect(ENSEMBLE_UNIT).toEqual({ pkg: 'c0q_openbox_b13_256', members: 16 });
    expect(CRIT2022_PKG).toBe('cornell_i_512');
    expect(AA_PKG).toBe('cornell_i_512');
    expect(PLANTS.map((p) => [p.plant, p.pkg])).toEqual([['no-j', 'cornell_i_512'], ['no-j', 'v_glossy_v1_512'], ['marginal-j', 'v_glossy_v2_512']]);
  });

  it('seed sets are pairwise disjoint (confirmatory re-runs, plants and A/A need independent replicates)', () => {
    const all = [...Object.values(SEEDS), ...PLANTS.map((p) => p.seed)];
    expect(new Set(all).size).toBe(all.length - 1);   // plantBase appears twice (SEEDS.plantBase = PLANTS[0].seed)
    expect(SEEDS.ptRerun).not.toBe(SEEDS.pt);
    expect(SEEDS.restirRerun).not.toBe(SEEDS.restir);
  });

  it('suite FWER n_units = 4 × (21 scenes × 3 rungs + ensemble + 2022 + 3 rendered plants + W×1.003 + A/A)', () => {
    expect(nUnits()).toBe(4 * (21 * 3 + 2 + 3 + 1 + 1));
  });
});

describe('sizing rule (PLAN §7.3, stats.sizing_target)', () => {
  it('normal and t quantiles', () => {
    expect(normInv(0.99)).toBeCloseTo(2.326348, 5);
    expect(normInv(0.005)).toBeCloseTo(-2.575829, 5);
    expect(tInv(0.99, 15)).toBeCloseTo(2.6025, 2);
    expect(tInv(0.99, 31)).toBeCloseTo(2.4528, 2);
    expect(tInv(0.99, Infinity)).toBeCloseTo(2.326348, 5);
  });

  it('δ/4.90 globally and δ/6.44 at m = 256 (ν = ∞), as the stats README', () => {
    expect(0.01 / sizingTarget(0.01, 1, Infinity)).toBeCloseTo(4.90, 2);
    expect(0.01 / sizingTarget(0.01, 256, Infinity)).toBeCloseTo(6.44, 2);
  });

  it('niceCeil gives m·2^k sizes ≥ x', () => {
    expect([1, 3.2, 4, 4.1, 5, 7.5, 9, 100, 1000].map(niceCeil)).toEqual([1, 4, 4, 5, 5, 8, 10, 112, 1024]);
    for (let x = 1; x < 5000; x += 7.3) {
      const n = niceCeil(x);
      expect(n).toBeGreaterThanOrEqual(x);
      expect(n / x).toBeLessThan(1.26);
    }
  });

  /** Synthetic side: constant image `level` + Gaussian batch noise of per-sample SD `sd` (so batch SD = sd/√n). */
  function side(level: number, sd: number, n: number, B: number, msPerSample: number, W = 64, H = 64, seed = 1): PilotSide {
    let s = seed >>> 0;
    const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return (s + 0.5) / 2 ** 32; };
    const gauss = () => Math.sqrt(-2 * Math.log(rnd())) * Math.cos(2 * Math.PI * rnd());
    const batches = Array.from({ length: B }, () => {
      const img = new Float32Array(W * H * 3);
      // per-pixel independent noise → tile mean SD = sd/√(n·tilePixels)
      for (let i = 0; i < img.length; i++) img[i] = level + (sd / Math.sqrt(n)) * gauss();
      return img;
    });
    return { B, n, msPerSample, W, H, batches };
  }

  it('aggregateSide recovers per-tile replicate SD', () => {
    const sd = 0.5, n = 16;
    const a = aggregateSide(side(1, sd, n, 64, 1), 32);
    expect(a.nTiles).toBe(4);
    // R channel tile 1: expected SD of the batch-mean tile average = sd/√n/32
    const nA = 1 + a.nTiles;
    const got = a.sd[1 * nA + 1], want = sd / Math.sqrt(n) / 32;
    expect(got / want).toBeGreaterThan(0.7);
    expect(got / want).toBeLessThan(1.3);
  });

  it('the allocation satisfies SE_Δ ≤ target on every aggregate and favours the cheaper side', () => {
    const pt = side(1, 2, 64, 16, 1, 64, 64, 3), rs = side(1, 2, 64, 16, 4, 64, 64, 5);   // same variance, ReSTIR 4× dearer
    const z = sizeScene(pt, [{ id: '3.1', side: rs, tile: 32 }], { B: 16, margin: 1, minPtSpp: 1, minFrames: () => 1 });
    const ref = aggregateSide(pt, 32), s = aggregateSide(rs, 32);
    const uR = uVector({ sd: ref.sd, n: pt.n }, ref, { deltaGlobal: 0.002, deltaTile: 0.01, B: 16 });
    const uK = uVector({ sd: s.sd, n: rs.n }, ref, { deltaGlobal: 0.002, deltaTile: 0.01, B: 16 });
    let worst = 0;
    for (let a = 0; a < uR.length; a++) worst = Math.max(worst, uR[a] / z.ptSamples + uK[a] / z.rungs['3.1'].samples);
    expect(worst).toBeLessThanOrEqual(1.0001);
    // optimum n_i ∝ sd_i/√c_i: the PT (cost 1) gets ~2× the samples of ReSTIR (cost 4)
    expect(z.ptSamples / z.rungs['3.1'].samples).toBeGreaterThan(1.4);
    expect(z.ptSamples / z.rungs['3.1'].samples).toBeLessThan(2.9);
  });

  it('a side above the cap enlarges that rung to 64² tiles; δ is untouched', () => {
    const pt = side(1, 2, 64, 16, 0.001, 128, 128, 7), rs = side(1, 6, 64, 16, 1, 128, 128, 9);
    const z = sizeScene(pt, [{ id: '3.2', side: rs, tile: 32 }], { B: 16, capS: 1 });
    expect(z.rungs['3.2'].tile).toBe(64);
    expect(z.rungs['3.2'].enlarged).toBe(true);
    expect(z.notes.join(' ')).toMatch(/64²/);
    const zz = sizeScene(pt, [{ id: '3.2', side: rs, tile: 32 }], { B: 16, capS: 1e9 });
    expect(zz.rungs['3.2'].tile).toBe(32);
    expect(z.rungs['3.2'].samples).toBeLessThan(zz.rungs['3.2'].samples);
  });

  it('per-batch floors only raise sizes (PT 256 spp, 3.1 128 frames, 3.2 8 frames)', () => {
    const pt = side(1, 1e-3, 64, 16, 1, 64, 64, 11), rs = side(1, 1e-3, 64, 16, 1, 64, 64, 13);
    const z = sizeScene(pt, [{ id: '3.1', side: rs, tile: 32 }, { id: '3.2', side: rs, tile: 32 }], { B: 16 });
    expect(z.ptSpp).toBe(256);
    expect(z.rungs['3.1'].framesPerBatch).toBe(128);
    expect(z.rungs['3.2'].framesPerBatch).toBe(8);
  });

  it('ensemble shape holds the rung total in member-frames', () => {
    expect(ensembleShape(1000, 16)).toEqual({ framesPerBatch: 63, batches: 1 });
    expect(ensembleShape(5000, 16)).toEqual({ framesPerBatch: 64, batches: 5 });
    const e = ensembleShape(12345, 16);
    expect(e.framesPerBatch * e.batches * 16).toBeGreaterThanOrEqual(12345);
  });
});

describe('T16 assertions (restir-api.md §6.1)', () => {
  const pt = { ok: true, errors: [], width: 512, height: 512, config: { jitter: 'iid-per-run', maxBounces: 3, lightMode: 'A', rr: false, scene: 'abc', env: 'none' } };
  const rs = () => ({
    kernel: 'restir', ok: true, errors: [], width: 512, height: 512, members: 1, config: { jitter: 'iid-per-run', scene: 'abc', env: 'none' },
    t16: { validationModeUnbiased: true, plantsNamed: [], internalScale: 1, denoiser: 'none', upscaler: 'none', readback: 'linear accumulation buffer', jitterMode: 'iid-per-run', maxBounces: 3, lightMode: 'A', spatialRoundsExecuted: 3 },
  });
  it('a conforming pair passes', () => {
    expect(t16Problems(rs(), pt)).toEqual([]);
    expect(t16Problems(rs(), pt, { rounds: 3 })).toEqual([]);
  });
  it('each violation is reported', () => {
    const cases: [(m: any) => void, RegExp][] = [
      [(m) => { m.t16.maxBounces = 4; }, /maxBounces/],
      [(m) => { m.t16.validationModeUnbiased = false; }, /unbiased/],
      [(m) => { m.t16.jitterMode = 'r2'; }, /jitter/],
      [(m) => { m.t16.denoiser = 'oidn'; }, /denoiser/],
      [(m) => { m.t16.internalScale = 0.5; }, /internal scale/],
      [(m) => { m.t16.readback = 'tonemapped'; }, /linear/],
      [(m) => { m.config.scene = 'other'; }, /scene bytes/],
      [(m) => { m.ok = false; m.errors = ['3 NaN/Inf']; }, /NaN/],
      [(m) => { m.t16.spatialRoundsExecuted = 0; }, /spatial stage/],
      [(m) => { m.config.env = { nee: true }; }, /env NEE/],
    ];
    for (const [mut, re] of cases) {
      const m = rs();
      mut(m);
      expect(t16Problems(m, pt, { rounds: 3 }).join('; ')).toMatch(re);
    }
    expect(t16Problems(rs(), { ...pt, config: { ...pt.config, rr: true } }).join()).toMatch(/RR/);
  });
  it('plant runs must name their plant and are exempt from the unbiased check', () => {
    const m = rs();
    m.t16.validationModeUnbiased = false;
    m.t16.plantsNamed = ['noJ'] as never[];
    expect(t16Problems(m, pt, { plant: true })).toEqual([]);
    m.t16.plantsNamed = [];
    expect(t16Problems(m, pt, { plant: true }).join()).toMatch(/named plant/);
  });
});

describe('submit packing (BatchAccumulator.runBatchUnits, restir-api.md §4.4)', () => {
  const U = (label: string, costHint: number): BatchWorkUnit => ({ label, costHint, encode: () => {} });
  it('unit kinds', () => {
    expect(unitKind('rs_initial[0+32][0]')).toBe('rs_initial');
    expect(unitKind('rs_primary[128]')).toBe('rs_primary');
    expect(unitKind('ens_copy[5]')).toBe('ens_copy');
  });
  it('unknown kinds run alone until measured, then pack up to the target', () => {
    const p = new UnitPacker(50, 64, new Map());
    expect(p.push(U('a[0]', 10))).toBeUndefined();
    const f1 = p.push(U('a[1]', 10));             // estimate(unknown) = target → flush the first
    expect(f1?.map((u) => u.label)).toEqual(['a[0]']);
    p.learn(f1!, 5);                               // 0.5 ms per cost
    expect(p.estimate(U('a', 10))).toBeCloseTo(5);
    const f2 = p.push(U('a[2]', 10));
    expect(f2?.map((u) => u.label)).toEqual(['a[1]']);
    for (let i = 3; i < 12; i++) p.push(U(`a[${i}]`, 10));   // 5 ms each: 10 fit in 50 ms
    const rest = p.drain()!;
    expect(rest.length).toBe(10);
  });
  it('the unit cap bounds the RsDispatch ring use', () => {
    const p = new UnitPacker(1e9, 4, new Map([['a', 0]]));
    let flushed = 0;
    for (let i = 0; i < 10; i++) if (p.push(U('a', 1))) flushed++;
    expect(flushed).toBe(2);
  });
  it('mixed submits scale the known rates toward the measurement (damped)', () => {
    const rates = new Map([['a', 1], ['b', 1]]);
    const p = new UnitPacker(50, 64, rates);
    p.learn([U('a', 10), U('b', 10)], 80);         // est 20 → ×√4 = ×2
    expect(rates.get('a')).toBeCloseTo(2);
    expect(rates.get('b')).toBeCloseTo(2);
    p.learn([U('a', 10), U('c', 5)], 30);          // one unknown: (30 − 20)/5
    expect(rates.get('c')).toBeCloseTo(2);
  });
  it('chunking: whole atlas with several trees when cheap; row bands at 1 tree when a tree is too costly', () => {
    expect(chooseChunking({ rateInitial: 1e-6, atlasW: 512, atlasH: 512, maxBounces: 3, trees: 32, unitTargetMs: 25 })).toEqual({ rowBand: 0, treeChunk: 23 });
    expect(chooseChunking({ rateInitial: 1e-7, atlasW: 512, atlasH: 512, maxBounces: 3, trees: 32, unitTargetMs: 25 })).toEqual({ rowBand: 0, treeChunk: 0 });
    const c = chooseChunking({ rateInitial: 1e-4, atlasW: 1024, atlasH: 1024, maxBounces: 7, trees: 32, unitTargetMs: 25 });
    expect(c.treeChunk).toBe(1);
    expect(c.rowBand % 8).toBe(0);
    expect(c.rowBand).toBeGreaterThanOrEqual(8);
    expect(c.rowBand * 1e-4 * 1024 * 8).toBeLessThanOrEqual(25);
  });
});

describe('PT reference cache key closures', () => {
  it('the PT key covers the PT kernel, the batch accumulator and the pt.wgsl include closure, not ReSTIR code', () => {
    const ts = tsClosure(['validation/harness/batch-run.ts']);
    expect(ts).toContain('src/core/render/pt-kernel.ts');
    expect(ts).toContain('src/core/render/batch-accumulator.ts');
    expect(ts.some((f) => f.includes('/restir/'))).toBe(false);
    const w = wgslClosure(['passes/pt.wgsl']);
    for (const f of ['passes/pt.wgsl', 'path/length1.wgsl', 'lights/env-sample.wgsl', 'material/bsdf.wgsl', 'bvh/traverse.wgsl']) expect(w).toContain(`src/core/shaders/${f}`);
    expect(w.some((f) => f.includes('shaders/restir/'))).toBe(false);
  });
});
