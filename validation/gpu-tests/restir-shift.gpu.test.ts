// ReSTIR PT shift core (WP-B, restir-api.md §6.1–§6.2): U-RC-1 (rcPairTest ≡ f64 dual), T2, T3-0…T3-5, T3-D,
// T3-ENV, T4, T5/U7, U5, U-11…U-13, dense PSS sweeps. Chrome lane authoritative; dawn.node is a pre-check.
import { afterAll, describe, expect, it } from 'vitest';
import { restirSettings } from '../../src/core/render/restir/presets.ts';
import { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import { JITTER_IID } from '../../src/core/render/frame-uniforms.ts';
import { dualCheckRecord, dualMaterial, pairTestF64, isLogic, type DualScene, type RcEventD, type RcVertexD } from '../../tests/restir/rc-dual.ts';
import { recentrePositions } from '../../src/core/render/scene-gpu.ts';
import type { SceneData } from '../../src/core/scene/types.ts';
import { fetchScenePackage } from '../../src/core/scene/scene-package.ts';
import { releaseTestGpu } from './device-factory.ts';
import { loadHdri, synthEnvData } from './env-fixtures.ts';
import { T3_ENV_ID, t3Scene, type T3Variant } from '../scenes/make-m4.ts';
import { allLightsScene, boxCamera, boxScene, gpuScene, light, readTexture4, restirRig, type RestirRig } from './restir-fixtures.ts';
import {
  DENSE_INIT_WGSL, DENSE_OFF, T3S, T3_BIN_NAMES, T3_CASES, T3_DUAL_CAP, T3_DUAL_WORDS, T3_NBINS, T3_STATS_WORDS, T3_VIOL_CAP, T3_VIOL_WORDS, T3_WGSL, bitsToF32, decodeT3Stats, readU32, storageBuffer, testPipeline,
  type T3Stats, type TestPipeline,
} from './restir-shift-fixtures.ts';
import { RS_TECH_NAMES, SC_NAMES, rfUnpack } from '../../src/core/render/restir/layout.ts';

afterAll(releaseTestGpu);

// ------------------------------------------------------------------------------------------------ U-RC-1

const RC_WORDS = 28;
const RC_HARNESS = `
#include "lights/lights.wgsl"
#include "restir/rc.wgsl"
@group(2) @binding(0) var<storage, read_write> outW: array<u32>;
fn hu(i: u32, k: u32) -> u32 { return pcg4d(vec4u(i, k, 0x3c6ef372u, 0x1b873593u)).x; }
fn hf(i: u32, k: u32) -> f32 { return u32_to_unit(hu(i, k)); }
fn hdir(i: u32, k: u32) -> vec3f {
  let z = 1.0 - 2.0 * hf(i, k);
  let r = sqrt(max(0.0, 1.0 - z * z));
  let ph = TWO_PI * hf(i, k + 1u);
  return vec3f(r * cos(ph), r * sin(ph), z);
}
fn hevent(i: u32, k: u32) -> RcEvent {
  var e: RcEvent;
  e.lobe = hu(i, k) % 6u;
  e.delta = select(0u, 1u, hf(i, k + 1u) < 0.08);
  let ak = hu(i, k + 2u) % 8u;
  var a = hf(i, k + 3u);
  if (ak == 0u) { a = 0.0; } else if (ak == 1u) { a = 0.19; } else if (ak == 2u) { a = 0.2; } else if (ak == 3u) { a = 0.21; }
  else if (ak == 4u) { a = 1.0; } else if (ak == 5u) { a = FLT_MAX; }
  e.alpha = a;
  e.pMarg = exp2(mix(-12.0, 16.0, hf(i, k + 4u)));
  if (hf(i, k + 5u) < 0.03) { e.pMarg = 0.0; }
  return e;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.y * 64u * 256u + gid.x;
  var a = RcVertex(vec3f(0.0), hdir(i, 1u), RCK_SURFACE, 0u);
  let t = exp2(mix(-8.0, 5.0, hf(i, 3u)));
  var b = RcVertex(a.pos + t * hdir(i, 4u), hdir(i, 6u), hu(i, 8u) % 3u, select(0u, 1u, hf(i, 9u) < 0.25));
  a.pos = vec3f(mix(-3.0, 3.0, hf(i, 10u)), mix(-3.0, 3.0, hf(i, 11u)), mix(-3.0, 3.0, hf(i, 12u)));
  // opaque to the compiler (a runtime 0): relaxed math must not fold (a + t·h) − a back to t·h in rcPairTest
  b.pos = bitcast<vec3f>(bitcast<vec3u>(a.pos + t * hdir(i, 4u)) ^ vec3u(rsDispatch.rowBase));
  let ea = hevent(i, 20u);
  let eb = hevent(i, 30u);
  let thr = exp2(mix(-20.0, 8.0, hf(i, 40u)));
  let r = rcPairTest(a, ea, b, eb, thr);
  let o = i * ${RC_WORDS}u;
  outW[o] = bitcast<u32>(a.pos.x); outW[o + 1u] = bitcast<u32>(a.pos.y); outW[o + 2u] = bitcast<u32>(a.pos.z);
  outW[o + 3u] = bitcast<u32>(a.ng.x); outW[o + 4u] = bitcast<u32>(a.ng.y); outW[o + 5u] = bitcast<u32>(a.ng.z);
  outW[o + 6u] = bitcast<u32>(b.pos.x); outW[o + 7u] = bitcast<u32>(b.pos.y); outW[o + 8u] = bitcast<u32>(b.pos.z);
  outW[o + 9u] = bitcast<u32>(b.ng.x); outW[o + 10u] = bitcast<u32>(b.ng.y); outW[o + 11u] = bitcast<u32>(b.ng.z);
  outW[o + 12u] = b.kind | (b.diffuseOnly << 4u);
  outW[o + 13u] = ea.lobe | (ea.delta << 4u); outW[o + 14u] = bitcast<u32>(ea.alpha); outW[o + 15u] = bitcast<u32>(ea.pMarg);
  outW[o + 16u] = eb.lobe | (eb.delta << 4u); outW[o + 17u] = bitcast<u32>(eb.alpha); outW[o + 18u] = bitcast<u32>(eb.pMarg);
  outW[o + 19u] = bitcast<u32>(thr);
  outW[o + 20u] = select(0u, 1u, r.ok) | (r.term << 4u);
  outW[o + 21u] = bitcast<u32>(r.margin);
  outW[o + 22u] = bitcast<u32>(rsParams.crit2022MinDist);
  outW[o + 23u] = bitcast<u32>(rsParams.alphaMin);
}
`;

describe('U-RC-1: rcPairTest (Enhanced and 2022) ≡ the f64 dual on random pair configurations', () => {
  for (const crit of ['enhanced', '2022'] as const) {
    it(`${crit}: 2^20 random pairs, LOGIC disagreements = 0`, async () => {
      const g = await gpuScene(boxScene([light({ id: 1, type: 'point', power: 50 })]));
      const k = await RestirKernel.create(g.device, g.gpu, g.env, { settings: restirSettings('initial', { criteria: crit }), features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
      k.setView({ camera: boxCamera(), width: 8, height: 8, runSeed: 1, jitterMode: JITTER_IID });
      const tp = await testPipeline(k, 'rc-test', RC_HARNESS, 'main', 1);
      const N = 256 * 64 * 64;
      const out = storageBuffer(g.device, N * RC_WORDS * 4);
      await tp.run([out], [256, 64]);
      const w = await readU32(g.device, out);
      const dmin = bitsToF32(w[22]), alphaMin = bitsToF32(w[23]);
      let logic = 0, fp = 0, pass = 0;
      const termHist = [0, 0, 0, 0, 0, 0];
      const ex: string[] = [];
      for (let i = 0; i < N; i++) {
        const o = i * RC_WORDS;
        const f = (j: number) => bitsToF32(w[o + j]);
        const a: RcVertexD = { pos: [f(0), f(1), f(2)], ng: [f(3), f(4), f(5)], kind: 0, diffuseOnly: false };
        const b: RcVertexD = { pos: [f(6), f(7), f(8)], ng: [f(9), f(10), f(11)], kind: w[o + 12] & 15, diffuseOnly: (w[o + 12] >> 4) !== 0 };
        const ea: RcEventD = { lobe: w[o + 13] & 15, delta: (w[o + 13] >> 4) !== 0, alpha: f(14), pMarg: f(15) };
        const eb: RcEventD = { lobe: w[o + 16] & 15, delta: (w[o + 16] >> 4) !== 0, alpha: f(17), pMarg: f(18) };
        const gpuOk = (w[o + 20] & 1) !== 0, gpuTerm = w[o + 20] >> 4;
        const d = pairTestF64(a, ea, b, eb, f(19), { alphaMin, crit2022: crit === '2022', dmin });
        termHist[d.term]++;
        if (d.ok) pass++;
        if (d.ok !== gpuOk || (!d.ok && d.term !== gpuTerm)) {
          if (isLogic(d)) { logic++; if (ex.length < 8) ex.push(`i=${i} gpu(ok=${gpuOk},term=${gpuTerm},m=${f(21)}) f64(${JSON.stringify(d)}) in=${JSON.stringify({ a, ea, b, eb, thr: f(19) })}`); } else fp++;
        }
      }
      console.log(`[U-RC-1 ${crit}] N=${N} pass=${pass} terms=${termHist} LOGIC=${logic} FP=${fp}`);
      if (ex.length) console.log(ex.join('\n'));
      expect(logic).toBe(0);
      out.destroy(); k.destroy(); g.destroy();
    });
  }
});

// ------------------------------------------------------------------------------------------------ T3 rig

interface T3RunOptions {
  mode: 0 | 1 | 3; partners: number; frames: number; frameBase?: number; maxTrialsPerDispatch?: number;
  /** Initial bin skip mask (bit b = skip case bin b). */
  skipMask?: number;
  /** Stop a bin once it has this many forward-OK trials (round trips); stop the run when every bin with ≥ minSeen
   *  trials in the first frames reached it, or after `frames` frames / `maxMs`. */
  target?: number; maxMs?: number;
  /** T3-D: record every `dualStride`-th trial (both shifts) and re-derive its predicate decisions in f64. */
  dual?: DualScene; dualStride?: number;
  /** M6 keep mask: trials whose stored path has ℓ_{k−1} = G_R (bit 0) / ℓ_k = G_R (bit 1) / x_k on glass (bit 2, the
   *  side-flip candidates) run even in bins at their target; with `xTarget` the run stays open until every kept counter
   *  (T3_X_CATS grKm1 / grK / sideFlip) has that many round trips. */
  xKeep?: number; xTarget?: number;
}
interface DualSummary { records: number; pairs: number; logic: number; fp: number; tangentFp: number; jChecked: number; jBad: number; skipped: Record<string, number>; details: string[] }
interface T3Result { stats: T3Stats; viol: Uint32Array; nViol: number; frames: number; ms: number; dual?: DualSummary; mode?: number; extra?: Record<string, { trials: number; rtOk: number; logic: number }> }

/** Render frames with the candidate dump and run the T3 kernel over every dumped candidate × partners. */
async function runT3(rig: RestirRig, tp: TestPipeline, o: T3RunOptions): Promise<T3Result> {
  const dev = rig.g.device;
  const k = rig.kernel;
  const stats = storageBuffer(dev, T3_STATS_WORDS * 4);
  const viol = storageBuffer(dev, (4 + T3_VIOL_CAP * T3_VIOL_WORDS) * 4);
  const dualBuf = storageBuffer(dev, o.dual ? (4 + T3_DUAL_CAP * T3_DUAL_WORDS) * 4 : 16);
  const dualSum: DualSummary = { records: 0, pairs: 0, logic: 0, fp: 0, tangentFp: 0, jChecked: 0, jBad: 0, skipped: {}, details: [] };
  const drainDual = async () => {
    const w = await readU32(dev, dualBuf);
    const n = Math.min(w[0], T3_DUAL_CAP);
    for (let r = 0; r < n; r++) {
      const res = dualCheckRecord(o.dual!, w, 4 + r * T3_DUAL_WORDS);
      dualSum.records++;
      if (res.skipped) { dualSum.skipped[res.skipped] = (dualSum.skipped[res.skipped] ?? 0) + 1; continue; }
      dualSum.pairs += res.checked; dualSum.logic += res.logic; dualSum.fp += res.fp; dualSum.tangentFp += res.tangentFp; dualSum.jChecked += res.jChecked; dualSum.jBad += res.jBad;
      for (const d of res.details) if (dualSum.details.length < 12) dualSum.details.push(d);
    }
    const enc = dev.createCommandEncoder();
    enc.clearBuffer(dualBuf, 0, 16);
    dev.queue.submit([enc.finish()]);
  };
  const P = rig.W * rig.H;
  const total = P * 32 * o.partners;
  const chunk = o.maxTrialsPerDispatch ?? (1 << 20);
  const res = k.resources;
  const t0 = Date.now();
  let mask = o.skipMask ?? 0, maskHi = 0, frames = 0;     // bins 0–31 in passId, 32+ in RsDispatch.flags bits 16+ (M6)
  for (let f = 0; f < o.frames; f++) {
    const t = (o.frameBase ?? 0) + f;
    await rig.frames(1, t);
    for (let base = 0; base < total; base += chunk) {
      const n = Math.min(chunk, total - base);
      const g = Math.ceil(n / 64);
      await tp.run([res.candDump!, stats, viol, dualBuf], [Math.min(g, 65535), Math.ceil(g / 65535)],
        { t, treeBase: base, treeCount: n, round: o.mode, flags: (o.partners | (maskHi << 16) | ((o.xKeep ?? 0) << 20)) >>> 0, passId: mask >>> 0, rowBase: o.dual ? (o.dualStride ?? 101) : 0 },
        [res.views.vbuf, res.views.geo]);
    }
    if (o.dual) await drainDual();
    frames++;
    if (o.target) {
      const w = await readU32(dev, stats, T3_NBINS * T3S.words * 4);
      mask = o.skipMask ?? 0; maskHi = 0;
      let open = 0;
      for (let b = 0; b < T3_NBINS; b++) {
        const ok = w[b * T3S.words + T3S.fwdOk];
        if (ok >= o.target) { if (b < 32) mask |= 1 << b; else maskHi |= 1 << (b - 32); } else if (w[b * T3S.words + T3S.trials] > 0) open++;
      }
      if (o.xKeep && o.xTarget) {
        const x = decodeT3Extra(await readU32(dev, stats));
        T3_X_CATS.forEach((c, i) => { if (((o.xKeep! >> i) & 1) && x[c].rtOk < o.xTarget!) open++; });
      }
      if (open === 0) break;
    }
    if (o.maxMs && Date.now() - t0 > o.maxMs) break;
  }
  const sw = await readU32(dev, stats);
  const s = decodeT3Stats(sw);
  const v = await readU32(dev, viol);
  stats.destroy(); viol.destroy(); dualBuf.destroy();
  return { stats: s, viol: v, nViol: v[0], frames, ms: Date.now() - t0, dual: o.dual ? dualSum : undefined, mode: o.mode, extra: decodeT3Extra(sw) };
}

function t3Report(tag: string, r: T3Result): { logic: number; fp: number; rt: number; jBad: number; platform: number } {
  let logic = 0, fp = 0, rt = 0;
  const rows: string[] = [];
  for (const [n, b] of Object.entries(r.stats.bins)) {
    if (!b.trials) continue;
    logic += b.logic + b.sigLogic + b.jViol + b.visZero + b.fViol;
    fp += b.fp + b.sigFp;
    rt += b.fwdOk;
    rows.push(`  ${n.padEnd(14)} trials=${b.trials} fwdOk=${b.fwdOk} rtOk=${b.rtOk} LOGIC=${b.logic} FP=${b.fp} visZero=${b.visZero} sigL=${b.sigLogic} sigFP=${b.sigFp} J=${b.jViol} F=${b.fViol} Fcond=${b.fCond} fwd=[${r.stats.fwdCodes[n].map((c, i) => c ? `${SC_NAMES[i]}:${c}` : '').filter(Boolean).join(' ')}] inv=[${r.stats.invCodes[n].map((c, i) => c ? `${SC_NAMES[i]}:${c}` : '').filter(Boolean).join(' ')}]`);
  }
  console.log(`[${tag}] frames=${r.frames} ms=${r.ms.toFixed(0)} fwdOk=${rt} LOGIC=${logic} FP=${fp} violations=${r.nViol}\n${rows.join('\n')}`);
  const ex: string[] = [];
  for (let i = 0; i < Math.min(r.nViol, T3_VIOL_CAP, 12); i++) {
    const o = 4 + i * T3_VIOL_WORDS;
    const w = r.viol;
    const f = rfUnpack(w[o + 8]);
    ex.push(`  kind=${w[o] & 255} bin=${T3_BIN_NAMES[w[o] >> 8]} fwd=${SC_NAMES[w[o + 2] & 255]}(t${(w[o + 2] >> 8) & 15},p${(w[o + 2] >> 12) & 15}) inv=${SC_NAMES[w[o + 3] & 255]}(t${(w[o + 3] >> 8) & 15},p${(w[o + 3] >> 12) & 15}) sig=${w[o + 4].toString(16)} edge=${bitsToF32(w[o + 5]).toExponential(2)} J=${bitsToF32(w[o + 6]).toPrecision(6)}/${bitsToF32(w[o + 7]).toPrecision(6)} m|fr=${bitsToF32(w[o + 11]).toExponential(3)} ai=${w[o + 9]} q=${w[o + 10]} t=${w[o + 12]} trial=${w[o + 13]} J0bits=${w[o + 6].toString(16)} f=${JSON.stringify(f)}`);
  }
  if (ex.length) console.log(ex.join('\n'));
  if (r.dual) {
    console.log(`[${tag} T3-D] records=${r.dual.records} pairs=${r.dual.pairs} LOGIC=${r.dual.logic} FP=${r.dual.fp} (tangent ${r.dual.tangentFp}) U-11 J: ${r.dual.jChecked} checked, ${r.dual.jBad} bad; skipped=${JSON.stringify(r.dual.skipped)}${r.dual.details.length ? '\n  ' + r.dual.details.join('\n  ') : ''}`);
    logic += r.dual.logic;
  }
  // accounting invariants (modes 0/1): every trial has one forward code; self shifts have no inverse; round trips run
  // the inverse exactly for the forward-OK trials. A break is a PLATFORM fault (control flow), counted apart from LOGIC.
  let platform = r.stats.chain.platform ?? 0;
  if (r.mode === 0 || r.mode === 1) {
    for (const [n, b] of Object.entries(r.stats.bins)) {
      const fwd = r.stats.fwdCodes[n].reduce((a, x) => a + x, 0), inv = r.stats.invCodes[n].reduce((a, x) => a + x, 0);
      const bad = fwd !== b.trials || (r.mode === 0 ? inv !== 0 : inv !== b.fwdOk) || b.rtOk > b.fwdOk;
      if (bad) { platform++; console.log(`[${tag} PLATFORM] ${n}: trials=${b.trials} Σfwd=${fwd} fwdOk=${b.fwdOk} Σinv=${inv} rtOk=${b.rtOk}`); }
    }
  }
  if (platform) console.log(`[${tag} PLATFORM] total=${platform} (kernel records ${r.stats.chain.platform ?? 0})`);
  const c = r.stats.chain;
  if (c.trials) console.log(`[${tag} T3-5] chains=${c.trials} both=${c.both} J>1e-4=${c.jViol} F>1e-4=${c.fViol} definedness mismatch=${c.defMismatch}`);
  logic += c.jViol + c.fViol + c.defMismatch;
  const epN = ['tri', 'point', 'spot', 'rect', 'disk', 'sun', '-', 'env'];
  console.log(`[${tag} U5] ` + r.stats.ep.map((e, i) => (e.fwdOk ? `${epN[i]}: fwdOk=${e.fwdOk} rtOk=${e.rtOk} LOGIC=${e.logic}` : '')).filter(Boolean).join(' · '));
  return { logic, fp, rt, jBad: r.dual?.jBad ?? 0, platform };
}

describe('T3-0 / T3-1 smoke on the all-lights box (allLightsScene, 128², maxBounces 4)', () => {
  it('self shifts (T3-0) and round trips (T3-1/T4)', async () => {
    const rig = await restirRig(allLightsScene(), 128, 128, { dumpCandidates: true, settings: { maxBounces: 4 } });
    const tp = await testPipeline(rig.kernel, 't3', T3_WGSL, 't3_main', 4, { RS_REPLAY: 1, RS_SHIFT_TRACE: 1, RS_VBUF_BINDING: '4u', RS_GEO_BINDING: '5u' }, ['uint', 'unfilterable-float']);
    const self = t3Report('T3-0 smoke', await runT3(rig, tp, { mode: 0, partners: 1, frames: 2 }));
    const rt = t3Report('T3-1 smoke', await runT3(rig, tp, { mode: 1, partners: 4, frames: 2 }));
    expect(self.logic).toBe(0);
    expect(rt.logic).toBe(0);
    rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ T3 on the t3_cases fixtures

async function t3Rig(variant: T3Variant, W = 256, o: { envNee?: boolean; criteria?: 'enhanced' | '2022' } = {}): Promise<RestirRig & { scene: SceneData }> {
  const env = variant === 't3_cases_256_noenv' ? undefined : (await loadHdri(`${T3_ENV_ID}_1k.hdr`)) ?? synthEnvData(256, 128);
  const t = t3Scene(variant, env);
  const rig = await restirRig(t.scene, W, W, {
    dumpCandidates: true, cam: { camToWorld: t.camera.matrix, yfov: t.camera.yfov },
    settings: { maxBounces: t.maxBounces, criteria: o.criteria ?? 'enhanced' }, env: { nee: o.envNee ?? true },
  });
  return Object.assign(rig, { scene: t.scene });
}
function dualScene(rig: RestirRig, scene: SceneData, params: { alphaMin: number; crit2022?: boolean; dmin?: number } = { alphaMin: 0.2 }): DualScene {
  const g = scene.geometry;
  // textured / alpha-tested materials (M6 t3_alpha_256) have no f64 twin: records touching them are skipped ('material')
  const mats = scene.materials.map((m) => (m.baseColorTexture || m.alphaMode === 'MASK' ? undefined : dualMaterial(m)));
  return { positions: recentrePositions(g.positions, rig.g.gpu.origin), indices: g.indices, triMaterial: g.triMaterial, materials: mats as DualScene['materials'], tau: rig.kernel.settings.tau, params };
}
const T3_DEFINES = { RS_REPLAY: 1, RS_SHIFT_TRACE: 1, RS_VBUF_BINDING: '4u', RS_GEO_BINDING: '5u' };

/** PLATFORM events (a control-flow fault of the T3 kernel itself, docs/decisions/platform-lanes.md "Metal quirks") are
 *  reported apart from LOGIC; a rate above 1e-8 per shift means something systematic and fails the test. */
function expectPlatformRare(...rs: { platform: number; rt: number }[]): void {
  const p = rs.reduce((a, r) => a + r.platform, 0), n = rs.reduce((a, r) => a + r.rt, 0);
  if (p) console.warn(`[T3 PLATFORM] ${p} control-flow fault(s) of the test kernel in ${n} shifts (rate ${(p / Math.max(n, 1)).toExponential(2)})`);
  expect(p / Math.max(n, 1)).toBeLessThanOrEqual(1e-8);
}

// Gate-0 budgets (per variant and mode): stop a case bin at T3_TARGET forward-OK round trips; wall-clock cap per run.
const T3_TARGET = 10_000_000;
const T3_MS = Number(import.meta.env?.VITE_T3_MS ?? 8 * 60_000);

describe('T3-0 / T2 / T3-1 / T4 / T3-D / U5 / T3-ENV / U-12 on the t3 fixtures (≥ 10⁷ round trips per case)', () => {
  const runs: [T3Variant, { envNee?: boolean }][] = [
    ['t3_cases_256', {}], ['t3_rare_256', {}], ['t3_cases_256_noenv', {}], ['t3_cases_256_envonly', { envNee: true }],
    ['t3_cases_256_envonly', { envNee: false }], ['t3_cutoff_256', {}],
  ];
  for (const [variant, o] of runs) {
    it(`${variant}${o.envNee === false ? ' (env NEE off)' : ''}`, async () => {
      const rig = await t3Rig(variant, 256, o);
      const tp = await testPipeline(rig.kernel, 't3', T3_WGSL, 't3_main', 4, T3_DEFINES, ['uint', 'unfilterable-float']);
      const tag = `${variant}${o.envNee === false ? '/nee-off' : ''}`;
      const self = t3Report(`T3-0 ${tag}`, await runT3(rig, tp, { mode: 0, partners: 1, frames: 100000, target: T3_TARGET, maxMs: T3_MS / 3 }));
      const rt = t3Report(`T3-1 ${tag}`, await runT3(rig, tp, {
        mode: 1, partners: 16, frames: 100000, target: T3_TARGET, maxMs: T3_MS, dual: dualScene(rig, rig.scene), dualStride: 997,
      }));
      expect(self.logic).toBe(0);
      expect(rt.logic).toBe(0);
      expect(rt.jBad).toBe(0);
      expectPlatformRare(self, rt);
      rig.destroy();
    }, 3_600_000);
  }
});

describe('T3-5: chained shifts i → j → l (J_{i→l} = J_{i→j}·J_{j→l} after write-back)', () => {
  it('t3_cases_256 and t3_rare_256', async () => {
    for (const v of ['t3_cases_256', 't3_rare_256'] as T3Variant[]) {
      const rig = await t3Rig(v);
      const tp = await testPipeline(rig.kernel, 't3', T3_WGSL, 't3_main', 4, T3_DEFINES, ['uint', 'unfilterable-float']);
      const r = t3Report(`T3-5 ${v}`, await runT3(rig, tp, { mode: 3, partners: 8, frames: 40 }));
      expect(r.logic).toBe(0);
      rig.destroy();
    }
  }, 3_600_000);
});

describe('U-11: marginal pdfs in J (RSF_PLANT_MARGINAL_J) are detected by the joint-pdf dual', () => {
  it('t3_cases_256 with the plant: the f64 joint-pdf J disagrees; without: agrees', async () => {
    const env = (await loadHdri(`${T3_ENV_ID}_1k.hdr`)) ?? synthEnvData(256, 128);
    const t = t3Scene('t3_cases_256', env);
    const rig = await restirRig(t.scene, 256, 256, {
      dumpCandidates: true, cam: { camToWorld: t.camera.matrix, yfov: t.camera.yfov }, settings: { maxBounces: t.maxBounces, plant: { marginalJ: true } },
    });
    const tp = await testPipeline(rig.kernel, 't3', T3_WGSL, 't3_main', 4, T3_DEFINES, ['uint', 'unfilterable-float']);
    const r = await runT3(rig, tp, { mode: 1, partners: 8, frames: 2, dual: dualScene(rig, t.scene), dualStride: 31 });
    const rep = t3Report('U-11 plant', r);
    console.log(`[U-11] plant: J checked ${r.dual!.jChecked}, disagreeing ${r.dual!.jBad}`);
    expect(r.dual!.jBad).toBeGreaterThan(Math.max(50, r.dual!.jChecked * 0.01));
    void rep;
    rig.destroy();
  }, 1_800_000);
});

// ------------------------------------------------------------------------------------------------ T5 / U7

describe('T5 / U7: change-of-variables census E_i[J·h·1{T ok}] = E_j[h·1{T⁻¹ ok}]', () => {
  it('t3_cases_256, offsets δ ∈ {(7, 3), (−12, 5), (2, −21)}; h ≡ 1 and h = F/(1+F); per case and technique z ≤ 4', async () => {
    const rig = await t3Rig('t3_cases_256');
    const tp = await testPipeline(rig.kernel, 't3', T3_WGSL, 't3_main', 4, T3_DEFINES, ['uint', 'unfilterable-float']);
    const dev = rig.g.device, k = rig.kernel, res = k.resources;
    const P = 256 * 256, trials = P * 32;
    const stats = storageBuffer(dev, T3_STATS_WORDS * 4), viol = storageBuffer(dev, (4 + T3_VIOL_CAP * T3_VIOL_WORDS) * 4);
    const chunk = 1 << 20;
    const out = storageBuffer(dev, (4 + 3 * chunk) * 4);
    // acc[side][bin] = [n, Σv1, Σv1², Σvh, Σvh²] in f64 (side 0 = A: J·h from i, side 1 = B: h from j)
    const NBIN = T3_NBINS + 3;                // + technique domains NEE, BSDF_TRI, BSDF_ENV
    const acc = [0, 1].map(() => Array.from({ length: NBIN }, () => [0, 0, 0, 0, 0]));
    const techOf = (bin: number) => { const c = T3_CASES[bin >> 1]; return c.startsWith('none-tri') || c === 'c-tri' || c === 'd' || c === 'deep-bsdf' ? 1 : (c === 'e' || c === 'c-env' || c.startsWith('none-env')) ? 2 : 0; };
    const deltas: [number, number][] = [[7, 3], [-12, 5], [2, -21]];
    const FRAMES = Number(import.meta.env?.VITE_T5_FRAMES ?? 40);
    let n = 0;
    for (let f = 0; f < FRAMES; f++) {
      await rig.frames(1, 5000 + f);
      const [dx, dy] = deltas[f % deltas.length];
      const packed = ((dx & 0xFFFF) | ((dy & 0xFFFF) << 16)) >>> 0;
      for (const side of [0, 1]) {
        for (let base = 0; base < trials; base += chunk) {
          const cnt = Math.min(chunk, trials - base);
          const enc = dev.createCommandEncoder(); enc.clearBuffer(out); dev.queue.submit([enc.finish()]);
          const g = Math.ceil(cnt / 64);
          await tp.run([res.candDump!, stats, viol, out], [Math.min(g, 65535), Math.ceil(g / 65535)],
            { t: 5000 + f, treeBase: base, treeCount: cnt, round: 2, flags: (1 | (side ? 0x80000000 : 0)) >>> 0, rowEnd: packed }, [res.views.vbuf, res.views.geo]);
          const w = await readU32(dev, out);
          const fw = new Float32Array(w.buffer);
          // per-tree sums (trial = (pixel, candidate)): the variance of the census is that of per-tree totals
          const tree = new Map<number, [number, number]>();
          const flush = () => {
            for (const [key, [v1, vh]] of tree) {
              const a = acc[side][key & 0xff];
              a[0]++; a[1] += v1; a[2] += v1 * v1; a[3] += vh; a[4] += vh * vh;
            }
            tree.clear();
          };
          let curPix = -1;
          for (let i = 0; i < cnt; i++) {
            const pix = Math.floor((base + i) / 32);
            if (pix !== curPix) { flush(); curPix = pix; }
            const tag = w[4 + 3 * i + 2];
            if (!(tag & 0x200)) continue;
            const bin = tag & 0xff, v1 = fw[4 + 3 * i], vh = fw[4 + 3 * i + 1];
            for (const b of [bin, T3_NBINS + techOf(bin)]) {
              const e = tree.get(b) ?? [0, 0];
              e[0] += v1; e[1] += vh;
              tree.set(b, e);
            }
            n++;
          }
          flush();
        }
      }
    }
    const names = [...T3_BIN_NAMES, 'tech:NEE', 'tech:BSDF_TRI', 'tech:BSDF_ENV'];
    let worst = 0;
    const rows: string[] = [];
    for (let b = 0; b < NBIN; b++) {
      const A = acc[0][b], B = acc[1][b];
      if (A[0] + B[0] === 0) continue;
      // per-tree variance: the sums run over ALL trees of the pixel pairs (most contribute 0), so Var ≈ Σv² (no mean term)
      const z = (i: number) => { const d = A[i] - B[i]; const v = A[i + 1] + B[i + 1]; return v > 0 ? d / Math.sqrt(v) : 0; };
      const z1 = z(1), zh = z(3);
      worst = Math.max(worst, Math.abs(z1), Math.abs(zh));
      rows.push(`  ${names[b].padEnd(14)} nA=${A[0]} nB=${B[0]} ΣJ=${A[1].toPrecision(6)} vs Σ1=${B[1].toPrecision(6)} z=${z1.toFixed(2)} | ΣJh=${A[3].toPrecision(6)} vs Σh=${B[3].toPrecision(6)} z=${zh.toFixed(2)}`);
    }
    console.log(`[T5] frames=${FRAMES} samples=${n} max|z|=${worst.toFixed(2)}\n${rows.join('\n')}`);
    expect(worst).toBeLessThanOrEqual(4);
    stats.destroy(); viol.destroy(); out.destroy(); rig.destroy();
  }, 3_600_000);
});

// ------------------------------------------------------------------------------------------------ dense PSS sweep

describe('Dense PSS sweep (t3_cases_256_b2, d ≤ 4): 32 source pixels × 32 partners, 2²⁰ lattice ū per pair', () => {
  it('LOGIC = 0 on every round trip; path classes enumerated', async () => {
    const rig = await t3Rig('t3_cases_256_b2');
    const dev = rig.g.device, k = rig.kernel, res = k.resources;
    const W = 256, P = W * W;
    await rig.frames(1, 0);
    const vb = await readTexture4(dev, res.vbuf), geo = await readTexture4(dev, res.geo);
    const valid = (x: number, y: number) => x >= 0 && y >= 0 && x < W && y < W && vb[4 * (y * W + x)] !== 0xFFFFFFFF;
    // 32 sources on a jittered grid over hit pixels, 32 partners each at disk offsets 1–30 px
    const rnd = (() => { let a = 12345; return () => { a = (Math.imul(a, 1664525) + 1013904223) >>> 0; return a / 2 ** 32; }; })();
    const sources: [number, number][] = [];
    for (let gy = 0; gy < 8 && sources.length < 32; gy++) for (let gx = 0; gx < 8 && sources.length < 32; gx++) {
      for (let tries = 0; tries < 50; tries++) {
        const x = Math.floor((gx + rnd()) * W / 8), y = Math.floor((gy + rnd()) * W / 8);
        if (valid(x, y) && (gx + gy) % 2 === 0) { sources.push([x, y]); break; }
      }
    }
    const cDefs = { RS_RES_OUT_BINDING: '0u', RS_ARENA_BINDING: '1u', RS_ARENA_RW: true, RS_DUMP_CANDIDATES: true, RS_RNG_OVERRIDE: 1 };
    const c = GPUShaderStage.COMPUTE;
    const bgl = dev.createBindGroupLayout({ entries: [0, 1, 4].map((binding) => ({ binding, visibility: c, buffer: { type: 'storage' as const } })) });
    const initPl = await k.compile('dense-init.wgsl', 'dense_initial', k.customDefines(cDefs, true), k.customLayout(bgl, true), 'dense-init', { 'dense-init.wgsl': DENSE_INIT_WGSL });
    const initG2 = dev.createBindGroup({ layout: bgl, entries: [{ binding: 0, resource: { buffer: res.res[0] } }, { binding: 1, resource: { buffer: res.arena } }, { binding: 4, resource: { buffer: res.candDump! } }] });
    const tp = await testPipeline(k, 't3dense', T3_WGSL, 't3_main', 4, { RS_REPLAY: 1, RS_SHIFT_TRACE: 1, T3_DENSE: 1, RS_RNG_OVERRIDE: 1 });
    const stats = storageBuffer(dev, T3_STATS_WORDS * 4), viol = storageBuffer(dev, (4 + T3_VIOL_CAP * T3_VIOL_WORDS) * 4);
    const dual = storageBuffer(dev, (DENSE_OFF + 8 * 33) * 4);
    const CHUNKS = Number(import.meta.env?.VITE_DENSE_CHUNKS ?? 16);   // 16 × 65536 = 2²⁰ lattice points
    const t0 = Date.now();
    for (let si = 0; si < sources.length; si++) {
      const [sx, sy] = sources[si];
      const texel = (x: number, y: number) => { const i = 4 * (y * W + x); return [vb[i], vb[i + 1], vb[i + 2], vb[i + 3], geo[i], geo[i + 1], geo[i + 2], geo[i + 3]]; };
      const dst = [texel(sx, sy)];
      while (dst.length < 33) {
        const r = 1 + 29 * Math.sqrt(rnd()), ph = 2 * Math.PI * rnd();
        const x = sx + Math.round(r * Math.cos(ph)), y = sy + Math.round(r * Math.sin(ph));
        if (valid(x, y) && !(x === sx && y === sy)) dst.push(texel(x, y));
      }
      const pairSeed = 0x9e3779b9 ^ Math.imul(si + 1, 0x85ebca6b);
      dev.queue.writeBuffer(res.arena, 256, Uint32Array.from([...dst[0], pairSeed >>> 0]));
      dev.queue.writeBuffer(dual, DENSE_OFF * 4, Uint32Array.from(dst.flat()));
      for (let ch = 0; ch < CHUNKS; ch++) {
        dev.queue.writeBuffer(dual, (DENSE_OFF - 2) * 4, Uint32Array.from([ch * P, pairSeed >>> 0]));
        k.beginSubmit();
        const enc = dev.createCommandEncoder();
        enc.clearBuffer(res.candDump!);
        k.encodeCustom(enc, initPl, initG2, { t: ch, treeBase: ch * P, rowBase: 0, rowEnd: W }, [W / 8, W / 8], true);
        dev.queue.submit([enc.finish()]);
        await dev.queue.onSubmittedWorkDone();
        const total = P * 32 * 32, chunk = 1 << 22;
        for (let base = 0; base < total; base += chunk) {
          const g = Math.ceil(Math.min(chunk, total - base) / 64);
          await tp.run([res.candDump!, stats, viol, dual], [Math.min(g, 65535), Math.ceil(g / 65535)], { t: ch, treeBase: base, treeCount: Math.min(chunk, total - base), round: 1, flags: 32 });
        }
      }
    }
    const st = decodeT3Stats(await readU32(dev, stats));
    const v = await readU32(dev, viol);
    const r = t3Report('dense PSS', { stats: st, viol: v, nViol: v[0], frames: CHUNKS * sources.length, ms: Date.now() - t0 });
    const cls = st.classes.sort((a, b) => b.count - a.count);
    const lob = ['D', 'S', 'GR', 'GT', 'NEE', '-', '?', '?'];
    const name = (key: number) => { const f = rfUnpack(key & 0x3FFFFF); return `d${f.d}k${f.k}${RS_TECH_NAMES[f.tech]}/ep${f.ep}/${[0, 1, 2].map((i) => lob[(key >>> (22 + 3 * i)) & 7]).slice(0, f.d - 1).join('')}`; };
    console.log(`[dense PSS] sources=${sources.length} lattice=${CHUNKS * P} classes=${cls.length}\n  ${cls.map((c) => `${name(c.key)}:${c.count}`).join(' ')}`);
    expect(r.logic).toBe(0);
    stats.destroy(); viol.destroy(); dual.destroy(); rig.destroy();
  }, 3_600_000);
});

// ------------------------------------------------------------------------------------------------ U-13

const U13_WGSL = `
#include "lights/lights.wgsl"
#include "restir/rc.wgsl"
@group(2) @binding(0) var<storage, read_write> outW: array<u32>;
@compute @workgroup_size(1)
fn main() {
  // flat geometric normal +z, shading normal tilted toward +x (N ≠ Ng); V in the upper hemisphere of both
  let ng = vec3f(0.0, 0.0, 1.0);
  let ns = normalize(vec3f(0.45, 0.0, 0.89));
  let V = normalize(vec3f(-0.3, 0.1, 0.95));
  // L below the geometric plane but above the shading plane: Ng·L < 0 < N·L
  let L = normalize(vec3f(0.95, 0.0, -0.05));
  for (var c = 0u; c < 2u; c++) {
    var m: MatEval;
    m.model = BSDF_MODEL_V1;
    m.base_color = vec3f(0.7);
    m.specular_tint = vec3f(0.9);
    m.metallic = select(0.0, 1.0, c == 1u);        // 0: Lambert, 1: GGX glossy r 0.4
    m.roughness = 0.4;
    m.ior = 1.5;
    m.specular_level = 0.5;
    m.ns = ns;
    m.ng = ng;
    m.flags = bsdf_flags(m, V);
    let lobe = select(LOBE_D, LOBE_S, c == 1u);
    let qb = bsdf_query(m, V, L, lobe);
    let qn = bsdf_query(m, V, L, LOBE_NEE);
    let o = c * 8u;
    outW[o] = select(0u, 1u, qb.supp);
    outW[o + 1u] = bitcast<u32>(qb.p_joint);
    outW[o + 2u] = bitcast<u32>(luminance(qb.f_lobe));
    outW[o + 3u] = select(0u, 1u, qn.supp);
    outW[o + 4u] = bitcast<u32>(luminance(qn.f_all));
    outW[o + 5u] = bitcast<u32>(dot(ns, L));
    outW[o + 6u] = bitcast<u32>(dot(ng, L));
  }
}
`;

describe('U-13: sampler-support indicator with N ≠ Ng (math.md#support-indicator)', () => {
  it('BSDF-sampled segment with Ng·L < 0 < N·L: supp = 0 (the shift declares O0_SUPPORT / O0_LOBE: F = 0); NEE segment: no indicator, f > 0', async () => {
    const g = await gpuScene(boxScene([light({ id: 1, type: 'point', power: 50 })]));
    const k = await RestirKernel.create(g.device, g.gpu, g.env, { settings: restirSettings('initial'), features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
    k.setView({ camera: boxCamera(), width: 8, height: 8, runSeed: 1, jitterMode: JITTER_IID });
    const tp = await testPipeline(k, 'u13', U13_WGSL, 'main', 1);
    const out = storageBuffer(g.device, 64);
    await tp.run([out], [1, 1]);
    const w = await readU32(g.device, out);
    for (const [c, name] of [[0, 'D'], [1, 'S']] as const) {
      const o = c * 8;
      console.log(`[U-13] ${name}: N·L=${bitsToF32(w[o + 5]).toFixed(3)} Ng·L=${bitsToF32(w[o + 6]).toFixed(3)} supp=${w[o]} p_joint=${bitsToF32(w[o + 1])} f_lobe=${bitsToF32(w[o + 2])} | NEE supp=${w[o + 3]} f_all=${bitsToF32(w[o + 4])}`);
      expect(w[o]).toBe(0);                          // support indicator 0 on the BSDF-sampled segment
      expect(bitsToF32(w[o + 1])).toBe(0);           // and the true sampler density is 0 there
      expect(w[o + 3]).toBe(1);                      // NEE: no indicator
      expect(bitsToF32(w[o + 4])).toBeGreaterThan(0); // Cycles' NEE eval (shading normal only) is non-zero
    }
    out.destroy(); k.destroy(); g.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ Stage-B scenes

/** Stage-B scene packages (§6.2: T3/T4 on (i), (v) V1, (v) V2, (vi)-A, (xiii), (xiv) overcast+rect; T3-ENV: C0r, (xiii),
 *  (xiv)); env packages are generated by make-m3c.ts into validation/out/m3c/scenes. Rendered at half resolution. */
const STAGE_B: [string, string][] = [
  ['cornell_i_512', 'validation/scenes'], ['v_glossy_v1_512', 'validation/scenes'], ['v_glossy_v2_512', 'validation/scenes'],
  ['vi_glass_mirror_A_512', 'validation/scenes'], ['xiii_spheres_512x256', 'validation/out/m3c/scenes'],
  ['xiv_overcast_rect_b3_512', 'validation/out/m3c/scenes'], ['c0r_irradiance_256', 'validation/out/m3c/scenes'], ['c0r_mirror_256', 'validation/out/m3c/scenes'],
];

describe('T3-1 / T4 / T3-D / T3-ENV on the Stage-B scenes', () => {
  for (const [name, dir] of STAGE_B) {
    it(name, async () => {
      let pkg;
      try { pkg = await fetchScenePackage(`/${dir}/${name}/`); } catch (e) { console.warn(`[T3 ${name}] package missing (${(e as Error).message}); run make-m3c.ts`); return; }
      const W = Math.max(64, Math.round(pkg.render.width / 2)), H = Math.max(64, Math.round(pkg.render.height / 2));
      const rig = await restirRig(pkg.scene, W, H, {
        dumpCandidates: true, cam: { camToWorld: Array.from(pkg.camera.matrix), yfov: pkg.camera.yfov }, settings: { maxBounces: Math.min(pkg.render.maxBounces, 7) },
        env: { nee: (pkg.json.env as { sampling?: string } | undefined)?.sampling !== 'NONE' },
      });
      const tp = await testPipeline(rig.kernel, 't3', T3_WGSL, 't3_main', 4, T3_DEFINES, ['uint', 'unfilterable-float']);
      const rt = t3Report(`T3-1 ${name}`, await runT3(rig, tp, {
        mode: 1, partners: 16, frames: 100000, target: T3_TARGET, maxMs: Number(import.meta.env?.VITE_T3B_MS ?? 4 * 60_000),
        dual: dualScene(rig, pkg.scene), dualStride: 997,
      }));
      expect(rt.logic).toBe(0);
      expect(rt.jBad).toBe(0);
      expectPlatformRare(rt);
      rig.destroy();
    }, 3_600_000);
  }
});

// ------------------------------------------------------------------------------------------------ debugging aid

/** Replay one recorded T3 violation: VITE_T3_REPRO = "<variant>,<frame t>,<trial>,<mode>" (skipped otherwise). */
describe('T3 single-trial repro (debugging aid)', () => {
  it('re-runs one trial with the dual trace on every shift', async () => {
    const spec = String(import.meta.env?.VITE_T3_REPRO ?? '');
    if (!spec) return;
    const [variant, tStr, trialStr, modeStr] = spec.split(',');
    const t = Number(tStr), trial = Number(trialStr), mode = Number(modeStr ?? 1);
    const rig = await t3Rig(variant as T3Variant);
    const tp = await testPipeline(rig.kernel, 't3', T3_WGSL, 't3_main', 4, T3_DEFINES, ['uint', 'unfilterable-float']);
    const dev = rig.g.device, res = rig.kernel.resources;
    const stats = storageBuffer(dev, T3_STATS_WORDS * 4), viol = storageBuffer(dev, (4 + T3_VIOL_CAP * T3_VIOL_WORDS) * 4);
    const dual = storageBuffer(dev, (4 + 8 * T3_DUAL_WORDS) * 4);
    await rig.frames(1, t);
    for (let rep = 0; rep < 3; rep++) {
      const enc = dev.createCommandEncoder(); enc.clearBuffer(dual); enc.clearBuffer(viol); dev.queue.submit([enc.finish()]);
      await tp.run([res.candDump!, stats, viol, dual], [1, 1], { t, treeBase: trial, treeCount: 1, round: mode, flags: 16, rowBase: 1 }, [res.views.vbuf, res.views.geo]);
      const w = await readU32(dev, dual), v = await readU32(dev, viol);
      const n = w[0];
      for (let r = 0; r < n; r++) {
        const o = 4 + r * T3_DUAL_WORDS;
        console.log(`[repro ${rep}] shift ${r}: code=${SC_NAMES[w[o] & 255]}(${w[o].toString(16)}) flags=${JSON.stringify(rfUnpack(w[o + 1]))} J=${bitsToF32(w[o + 77])} jDen=${bitsToF32(w[o + 78])} dst=${w[o + 3]} light=${[16, 17, 18, 19, 20, 21, 22].map((i) => bitsToF32(w[o + i]).toPrecision(5))}`);
      }
      console.log(`[repro ${rep}] violations=${v[0]} kind=${v[4] & 255}`);
    }
    stats.destroy(); viol.destroy(); dual.destroy(); rig.destroy();
  }, 600_000);
});

// ------------------------------------------------------------------------------------------------ PLATFORM discriminator

/** VITE_DISC = "variant[,variant…]:frames:reps": T3-0 self shifts, a fresh kernel + pipelines per rep, fixed frame count
 *  (frame bases differ per rep); reports PLATFORM events (control-flow faults of the test kernel) per run. */
describe('T3 PLATFORM discriminator (VITE_DISC)', () => {
  it('repeated T3-0 runs', async () => {
    const spec = String(import.meta.env?.VITE_DISC ?? '');
    if (!spec) return;
    const [vs, fStr, rStr, build = 'new', maskSpec = 'all'] = spec.split(':');
    // 'forced': only the forced-NEE bins (a-*, f-env) stay active, as in the late phase of a targeted run
    const skipMask = maskSpec === 'forced' ? ((2 ** T3_NBINS - 1) & ~0x3FF) >>> 0 : 0;
    const frames = Number(fStr), reps = Number(rStr);
    const summary: string[] = [];
    // builds of the shift pipeline: 'new' (tree), 'legacy' = shift.wgsl before the compact-NEE-state restructure,
    // 'oldharness' (anywhere in the name) = the T3 kernel of 6e02a93 (no sentinels / PLATFORM counters),
    // '*-noguard' = + rc.wgsl before the |cos| guard (c807aac). Frozen copies: validation/gpu-tests/t3fault-*.
    const extra: Record<string, string> = {};
    if (build.startsWith('legacy')) extra['restir/shift.wgsl'] = (await import('./t3fault-shift-a54ac2d.wgsl.txt?raw')).default;
    if (build.endsWith('noguard')) extra['restir/rc.wgsl'] = (await import('./t3fault-rc-c807aac.wgsl.txt?raw')).default;
    for (const variant of vs.split(',') as T3Variant[]) {
      for (let rep = 0; rep < reps; rep++) {
        const rig = await t3Rig(variant);
        const kernelSrc = build.includes('oldharness') ? (await import('./t3fault-fixtures-6e02a93.ts')).T3_WGSL : T3_WGSL;
        const tp = await testPipeline(rig.kernel, `t3-${rep}`, kernelSrc + `\n// rep ${rep}\n`, 't3_main', 4, T3_DEFINES, ['uint', 'unfilterable-float'], extra);
        const r = t3Report(`DISC ${build} ${variant} #${rep}`, await runT3(rig, tp, { mode: 0, partners: 1, frames, frameBase: 100000 * rep, skipMask }));
        summary.push(`${build}/${maskSpec} ${variant} #${rep}: fwdOk=${r.rt} PLATFORM=${r.platform} LOGIC=${r.logic}`);
        rig.destroy();
      }
    }
    console.log(`[DISC summary]\n  ${summary.join('\n  ')}`);
  }, 3_600_000);
});

/** VITE_STRESS = "variant[,variant…]:frames": production passes (rs_spatial_replay / rs_spatial_shift / resample,
 *  3 rounds × 6 slots, 1 tree) on the T3 fixtures; the arena counters of a slot-index / control-flow fault
 *  (RSC_PENDING_LEFT, RSC_SLOT_MISMATCH, RSC_SHIFT_NONFINITE) must stay 0. */
describe('Production shift passes: platform-fault stress (VITE_STRESS)', () => {
  it('arena counters', async () => {
    const spec = String(import.meta.env?.VITE_STRESS ?? '');
    if (!spec) return;
    const [vs, fStr] = spec.split(':');
    const frames = Number(fStr);
    for (const variant of vs.split(',') as T3Variant[]) {
      const env = variant === 't3_cases_256_noenv' ? undefined : (await loadHdri(`${T3_ENV_ID}_1k.hdr`)) ?? synthEnvData(256, 128);
      const t = t3Scene(variant, env);
      const rig = await restirRig(t.scene, 256, 256, {
        preset: 'offline', cam: { camToWorld: t.camera.matrix, yfov: t.camera.yfov },
        settings: { maxBounces: t.maxBounces, trees: 1, rounds: 3, slots: 6, diskRadius: 10 },
      });
      const tot = { pendingLeft: 0, slotMismatch: 0, shiftNonFinite: 0, accepted: 0, queued: 0 };
      const codes = new Array(16).fill(0);
      for (let f = 0; f < frames; f += 20) {
        const r = await rig.frames(Math.min(20, frames - f), 7000 + f);
        for (const k of Object.keys(tot) as (keyof typeof tot)[]) tot[k] += r.arena.rsc[k];
        r.arena.codes.forEach((c, i) => { codes[i] += c; });
      }
      console.log(`[STRESS ${variant}] frames=${frames} ${JSON.stringify(tot)} codes=${codes.map((c, i) => (c ? `${SC_NAMES[i]}:${c}` : '')).filter(Boolean).join(' ')}`);
      expect(tot.pendingLeft + tot.slotMismatch + tot.shiftNonFinite).toBe(0);
      rig.destroy();
    }
  }, 3_600_000);
});

// ------------------------------------------------------------------------------------------------ M6: T3-M6 (gating)

import { t3M6Scene, type T3M6Variant } from '../scenes/m6-fixtures.ts';
import { T3_WGSL_BODY, T3_X_CATS, decodeT3Extra, type T3Card } from './restir-shift-fixtures.ts';
import type { LightMode } from '../../src/core/render/lights-gpu.ts';

/** M6 T3 variants (restir-m6-api.md §4 T3-M6): RIS-NEE on t3_cases_256, Mode B + RIS on t3_modeb_256, rough glass
 *  (G_R reconnection, side flips) on t3_glass_256 (gating in M6, D13 → rung 3.9), alpha cards on t3_alpha_256. */
const T3M6_RUNS: { name: string; base: T3Variant | T3M6Variant; mode: LightMode; settings: Record<string, unknown>; need: string[]; needX: string[] }[] = [
  { name: 't3_cases_256 + RIS-NEE', base: 't3_cases_256', mode: 'A', settings: { risNee: true }, need: ['a-delta', 'a-area', 'a-tri', 'f-env', 'd', 'e'], needX: [] },
  { name: 't3_modeb_256 (Mode B + RIS-NEE)', base: 't3_modeb_256', mode: 'B', settings: { risNee: true }, need: ['d-ana', 'c-ana'], needX: [] },
  { name: 't3_modeb_rare_256 (Mode B, frequent ∅ crossings)', base: 't3_modeb_rare_256', mode: 'B', settings: {}, need: ['none-ana'], needX: [] },
  { name: 't3_glass_256 (rough-glass G_R reconnection)', base: 't3_glass_256', mode: 'A', settings: {}, need: [], needX: ['grKm1', 'grK'] },
  { name: 't3_glass_pane_256 (rough-glass side flips)', base: 't3_glass_pane_256', mode: 'A', settings: {}, need: [], needX: ['sideFlip'] },
  { name: 't3_alpha_256 (cutouts on reconnection segments)', base: 't3_alpha_256', mode: 'A', settings: {}, need: [], needX: ['alphaCard'] },
];
const T3M6_MIN = Number(import.meta.env?.VITE_T3M6_MIN ?? 1_000_000);

describe('T3-M6: T3-0 / T3-1 / T4 / T3-D LOGIC = 0 on the M6 cases (≥ 10⁶ round trips per new case / counter)', () => {
  for (const run of T3M6_RUNS) {
    it(run.name, async () => {
      const env = (await loadHdri(`${T3_ENV_ID}_1k.hdr`)) ?? synthEnvData(256, 128);
      const m6 = (['t3_modeb_256', 't3_modeb_rare_256', 't3_alpha_256', 't3_glass_pane_256'] as string[]).includes(run.base);
      const t = m6 ? t3M6Scene(run.base as T3M6Variant, env) : { ...t3Scene(run.base as T3Variant, env), cards: [] as T3Card[] };
      const rig = await restirRig(t.scene, 256, 256, {
        dumpCandidates: true, cam: { camToWorld: t.camera.matrix, yfov: t.camera.yfov }, lightMode: run.mode,
        settings: { maxBounces: t.maxBounces, ...run.settings }, env: { nee: true },
      });
      const o = rig.g.gpu.origin;
      const cards: T3Card[] = t.cards.map((c) => ({ ...c, c: [c.c[0] - o[0], c.c[1] - o[1], c.c[2] - o[2]] as [number, number, number] }));
      const tp = await testPipeline(rig.kernel, 't3m6', T3_WGSL_BODY(cards), 't3_main', 4, { ...T3_DEFINES, RS_M6_TRACE: 1 }, ['uint', 'unfilterable-float']);
      const tag = run.base;
      // glass counters: trials whose stored path can produce the counted event keep running after their case bin reached
      // its target (grKm1 → ℓ_{k−1} = G_R, grK → ℓ_k = G_R, sideFlip → x_k on glass; restir-m6-api.md Changelog M6-14)
      const xKeep = (run.needX.includes('grKm1') ? 1 : 0) | (run.needX.includes('grK') ? 2 : 0) | (run.needX.includes('sideFlip') ? 4 : 0);
      const self = t3Report(`T3-0 ${tag}`, await runT3(rig, tp, { mode: 0, partners: 1, frames: 100000, target: T3_TARGET, maxMs: T3_MS / 3, xKeep }));
      const rtRes = await runT3(rig, tp, { mode: 1, partners: 16, frames: 100000, target: T3_TARGET, maxMs: T3_MS, dual: dualScene(rig, t.scene), dualStride: 997, xKeep, xTarget: T3_TARGET });
      const rt = t3Report(`T3-1 ${tag}`, rtRes);
      const x = rtRes.extra!;
      console.log(`[T3-M6 ${tag} extra] ${JSON.stringify(x)}`);
      expect(self.logic).toBe(0);
      expect(rt.logic).toBe(0);
      expect(rt.jBad).toBe(0);
      expectPlatformRare(self, rt);
      for (const b of run.need) {
        const n = (rtRes.stats.bins[`${b}/k2`]?.fwdOk ?? 0) + (rtRes.stats.bins[`${b}/k>2`]?.fwdOk ?? 0);
        console.log(`[T3-M6 ${tag}] ${b}: ${n} round trips`);
        expect(n, b).toBeGreaterThanOrEqual(T3M6_MIN);
      }
      for (const c of Object.keys(x)) expect(x[c].logic, c).toBe(0);
      for (const c of run.needX) expect(x[c].rtOk, c).toBeGreaterThanOrEqual(T3M6_MIN);
      rig.destroy();
    }, 3_600_000);
  }
});
