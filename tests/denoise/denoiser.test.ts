// M5.5 denoiser, CPU lane (docs/decisions/denoiser.md): mirrors (WGSL ↔ TS ↔ ReSTIR layout), uniform packing, the
// à-trous plan, the mode table (DN4/§8), formulas of the reference (§3–§7: demodulation, oct guides, the EMA moment
// identity, the gradient's λ), T16 helpers and the code-hash isolation (DN10). The passes themselves are compared with
// tests/denoise/dn-ref.ts on the GPU (validation/gpu-tests/denoiser.gpu.test.ts).
import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  DENOISER_DEFAULTS, DENOISER_VIEWS, DN_PARAMS_SIZE, DN_TSTATE, DN_VIEW, DNF, DNI, DNT, atrousPlan, denoiseModeKey, denoiserAllowed, denoiserDefault, denoiserAppSettings, dnTiles, packDnParams,
} from '../../src/core/render/denoise/layout.ts';
import { denoiserT16State, liveDenoisers, registerDenoiser, unregisterDenoiser } from '../../src/core/render/denoise/registry.ts';
import { BUILTIN_VIEWS } from '../../src/core/render/debug-views.ts';
import { RS_WGSL_CONSTS, TS_CONSTS, TSW, TS_WORDS } from '../../src/core/render/restir/layout.ts';
import { RESTIR_VIEWS } from '../../src/core/render/restir/debug.ts';
import { ENV_DEBUG_VIEWS } from '../../src/core/render/env-debug.ts';
import { EXTRA_VIEWS } from '../../src/core/render/renderer.ts';
import { denoiserT16Problems } from '../../validation/harness/t16.ts';
import { RECOVERY_RUNS, recoverySchedule } from '../../validation/harness/gate-m55.ts';
import { tsClosure } from '../../validation/harness/gate-m4.ts';
import { parsePkgFrames } from '../../validation/harness/run-denoise.ts';
import {
  SC, demodFactor, guideNormal, octDecode, octEncode, pack2x16snorm, refLambda, refPairs, unpack2x16snorm, type V3,
} from './dn-ref.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const SH = path.join(ROOT, 'src/core/render/denoise/shaders');
const wgsl = (f: string) => readFileSync(path.join(SH, f), 'utf8');
/** `const NAME: u32 = Nu;` constants of a WGSL source. */
function wgslConsts(src: string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const m of src.matchAll(/const\s+(\w+)\s*:\s*(?:u32|f32)\s*=\s*([0-9.eE+-]+|0x[0-9a-fA-F]+)u?\s*;/g)) out[m[1]] = Number(m[2]);
  return out;
}
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

describe('mirrors (WGSL ↔ layout.ts ↔ ReSTIR layout)', () => {
  const common = wgslConsts(wgsl('dn-common.wgsl'));
  const grad = wgslConsts(wgsl('dn-gradient.wgsl'));
  const filt = wgslConsts(wgsl('dn-filter.wgsl'));
  it('DNF flags and view ids', () => {
    for (const [k, v] of Object.entries(DNF)) expect(common[`DNF_${k}`], k).toBe(v);
    expect(common.DNV_VARIANCE).toBe(DN_VIEW.variance);
    expect(common.DNV_HISTORY).toBe(DN_VIEW.history);
    expect(common.DNV_ALPHA).toBe(DN_VIEW.alpha);
    expect(common.DNV_LAMBDA).toBe(DN_VIEW.lambda);
    expect(common.DNV_DEMOD).toBe(DN_VIEW.demod);
    expect(common.DNV_INTEGRATED).toBe(DN_VIEW.integrated);
    expect(common.DNV_REPROJ).toBe(DN_VIEW.reproj);
    expect(common.DNV_PAIRS).toBe(DN_VIEW.pairs);
    expect(common.DNV_LEVEL0).toBe(DN_VIEW.level0);
    expect(common.DNV_DEMOD_FACTOR).toBe(DN_VIEW.demodFactor); expect(common.DNV_ALB_ACCUM).toBe(DN_VIEW.albedoAccum); expect(common.DNV_DEMOD_CHECK).toBe(DN_VIEW.demodCheck);
    for (const [k, v] of Object.entries(DNI)) expect(filt[`DNI_${k}`], k).toBe(v);
  });
  it('the gradient pass reads the tState words, flags and slot codes of the M5 contract', () => {
    expect(grad.TSW_QPRIME).toBe(TSW.qPrime); expect(grad.TSW_CP).toBe(TSW.cP); expect(grad.TSW_FWDCODE).toBe(TSW.fwdCode);
    expect(grad.TSW_FLAGS).toBe(TSW.flags); expect(grad.TSW_WC).toBe(TSW.wc); expect(grad.TSW_WP).toBe(TSW.wp);
    expect(grad.TSW_INVCODE).toBe(TSW.invCode); expect(grad.TSW_PIRECOMP).toBe(TSW.piRecomp); expect(grad.TS_WORDS).toBe(TS_WORDS);
    for (const k of ['TS_QVALID', 'TS_SEL_C', 'TS_INV_DONE', 'TS_EMPTY_OUT'] as const) { expect(grad[k], k).toBe(TS_CONSTS[k]); expect(DN_TSTATE[k]).toBe(TS_CONSTS[k]); }
    for (const k of ['SC_OK', 'SC_O0_LIGHT', 'SC_OCCLUDED', 'SC_ZERO'] as const) { expect(grad[k], k).toBe(RS_WGSL_CONSTS[k]); expect(DN_TSTATE[k]).toBe(RS_WGSL_CONSTS[k]); }
    expect(SC.O0_LIGHT).toBe(RS_WGSL_CONSTS.SC_O0_LIGHT); expect(SC.OCCLUDED).toBe(RS_WGSL_CONSTS.SC_OCCLUDED); expect(SC.ZERO).toBe(RS_WGSL_CONSTS.SC_ZERO);
    expect(TSW.qPrime).toBe(DN_TSTATE.TSW_QPRIME);
  });
  it('views: unique ids and keys, no collision with any other registered view', () => {
    const others = [...BUILTIN_VIEWS, ...EXTRA_VIEWS, ...ENV_DEBUG_VIEWS, ...RESTIR_VIEWS];
    const ids = new Set(others.map((v) => v.id));
    const keys = new Set(others.map((v) => v.key));
    expect(new Set(DENOISER_VIEWS.map((v) => v.id)).size).toBe(DENOISER_VIEWS.length);
    for (const v of DENOISER_VIEWS) { expect(ids.has(v.id), `${v.id}`).toBe(false); expect(keys.has(v.key), v.key).toBe(false); }
    for (const id of [520, 521, 522]) expect(DENOISER_VIEWS.some((v) => v.id === id)).toBe(true);   // PLAN §6: variance, history, α
  });
});

describe('uniforms and the à-trous plan', () => {
  it('packDnParams matches DnParams (64 B)', () => {
    const b = packDnParams({ width: 961, height: 539, flags: DNF.RESET | DNF.FW, settings: { ...DENOISER_DEFAULTS, alphaMin: 0.25 }, tsBase: 12345, resPlanes: 10 });
    expect(b.byteLength).toBe(DN_PARAMS_SIZE);
    const u = new Uint32Array(b), f = new Float32Array(b);
    expect([u[0], u[1], u[2], u[3], u[4]]).toEqual([961, 539, 121, 68, 9]);
    expect([f[5], f[6], f[9], f[10], f[11]]).toEqual([64, Math.fround(0.25), 1, 128, 4]);
    expect([u[12], u[13]]).toEqual([12345, 10]);
    const b2 = packDnParams({ width: 8, height: 8, flags: 0, settings: { ...DENOISER_DEFAULTS, taaLightMax: 24, taaCamMax: 12, taaDilate: true, taaCubic: true }, tsBase: 0, resPlanes: 10 });
    expect([new Float32Array(b2)[21], new Float32Array(b2)[22], new Uint32Array(b2)[23]]).toEqual([24, 12, DNT.DILATE | DNT.CUBIC]);   // DN-16
    expect(dnTiles(960, 540)).toEqual([120, 68]);
  });
  it('atrousPlan: steps 2^i, feedback on the first, final on the last; N = 0 is one copy pass', () => {
    expect(atrousPlan(5)).toEqual([[0, 1, DNI.FEEDBACK], [1, 2, 0], [2, 4, 0], [3, 8, 0], [4, 16, DNI.FINAL]]);
    expect(atrousPlan(1)).toEqual([[0, 1, DNI.FEEDBACK | DNI.FINAL]]);
    expect(atrousPlan(0)).toEqual([[0, 0, DNI.COPY | DNI.FEEDBACK | DNI.FINAL]]);
  });
});

describe('modes (denoiser.md §8, DN4)', () => {
  it('forced off in ReSTIR-unbiased and albedo, allowed elsewhere; default on in ReSTIR-interactive and Potato', () => {
    expect(denoiserAllowed('restir', 'unbiased')).toBe(false);
    expect(denoiserAllowed('albedo', 'interactive')).toBe(false);
    for (const m of ['interactive', 'potato', 'criteria2022', 'offline', 'initial'] as const) expect(denoiserAllowed('restir', m)).toBe(true);
    expect(denoiserAllowed('pt', 'unbiased')).toBe(true);
    expect(denoiserDefault('restir', 'interactive')).toBe(true);
    expect(denoiserDefault('restir', 'potato')).toBe(true);
    expect(denoiserAppSettings('restir', 'potato').iterations).toBe(3);
    expect(denoiserAppSettings('restir', 'interactive')).toEqual(DENOISER_DEFAULTS);
    expect(denoiserAppSettings('pt', 'potato')).toEqual(DENOISER_DEFAULTS);
    expect(denoiserAppSettings('restir', 'potato', { iterations: 5 }).iterations).toBe(5);
    for (const m of ['unbiased', 'criteria2022', 'offline', 'initial'] as const) expect(denoiserDefault('restir', m)).toBe(false);
    expect(denoiserDefault('pt', 'interactive')).toBe(false);
    expect(denoiseModeKey('pt', 'offline')).toBe('pt');
    expect(denoiseModeKey('restir', 'offline')).toBe('restir:offline');
  });
});

describe('reference formulas', () => {
  it('demodulation factor: max(albedo, 0.02) + F0 (DN-3), black / glossy-only albedo → 1', () => {
    expect(demodFactor([0, 0, 0])).toEqual([1, 1, 1]);
    expect(demodFactor([0.019, 0.01, 0])).toEqual([1, 1, 1]);
    expect(demodFactor([0.6, 0.06, 0.01]).map((x) => Math.round(x * 1e9) / 1e9)).toEqual([0.64, 0.1, 0.06]);
  });
  it('oct 2×16 snorm guide normals: < 0.004° with round-to-nearest packing (data-formats.md §B3: 0.0025° with an optimised encoder)', () => {
    const r = rng(7);
    let worst = 0;
    for (let i = 0; i < 20000; i++) {
      const z = 2 * r() - 1, ph = 2 * Math.PI * r(), s = Math.sqrt(1 - z * z);
      const n: V3 = [s * Math.cos(ph), s * Math.sin(ph), z];
      const m = guideNormal(n);
      worst = Math.max(worst, Math.acos(Math.min(1, n[0] * m[0] + n[1] * m[1] + n[2] * m[2])) * 180 / Math.PI);
      const e = octEncode(n);
      expect(Math.abs(octDecode(e)[0] - n[0])).toBeLessThan(1e-9);
    }
    expect(worst).toBeLessThan(0.004);
    expect(unpack2x16snorm(pack2x16snorm(-1, 1))).toEqual([-1, 1]);
  });
  it('EMA moments: storing (μ, σ) with the West update equals m₂ − m₁² of the exponentially weighted raw moments', () => {
    const r = rng(11);
    let mu = 0, v = 0, m1 = 0, m2 = 0, n = 0;
    for (let k = 0; k < 200; k++) {
      const l = 1000 * r() ** 4;   // heavy-tailed luminance (would overflow fp16 as l²)
      n = Math.min(n + 1, 64);
      const a = Math.max(0.2, 1 / n);
      const d = l - mu; mu += a * d; v = (1 - a) * (v + a * d * d);
      m1 = (1 - a) * m1 + a * l; m2 = (1 - a) * m2 + a * l * l;
      expect(mu).toBeCloseTo(m1, 6);
      expect(Math.abs(v - (m2 - m1 * m1))).toBeLessThan(1e-6 * Math.max(1, m2));
    }
  });
  it('gradient λ: a light ×2 → 0.5, a removed light of share s → s, static → 0; geometric failures give no pair', () => {
    const W = 24, H = 24, P = W * H;
    const r = rng(3);
    const fw = new Float64Array(P).map(() => 0.5 + r());
    const mk = (f: (q: number) => number[]) => refLambda(W, H, Array.from({ length: P }, (_, q) => f(q)));
    const base = { flags: 1, qPrime: 0, cP: 5, fwdCode: SC.OK, wc: 0, wp: 0, invCode: 0, piRecomp: 0 };
    // light ×2: the forward re-evaluation doubles every history sample
    const x2 = mk((q) => refPairs({ ...base, qPrime: q, wp: 5 * 2 * fw[q] }, (i) => fw[i], 0, false, P));
    for (const l of x2.lambda) expect(l).toBeCloseTo(0.5, 12);
    // static lighting: a = b
    const st = mk((q) => refPairs({ ...base, qPrime: q, wp: 5 * fw[q] }, (i) => fw[i], 0, false, P));
    for (const l of st.lambda) expect(l).toBe(0);
    // removed light: 30 % of the history samples lost their light (SC_O0_LIGHT), weighted like their contribution
    const rm = mk((q) => (q % 10 < 3 ? refPairs({ ...base, qPrime: q, fwdCode: SC.O0_LIGHT }, () => 1, 0, false, P) : refPairs({ ...base, qPrime: q, wp: 5 }, () => 1, 0, false, P)));
    const mid = rm.lambda[Math.floor(dnTiles(W, H)[0] / 2) * dnTiles(W, H)[0] + 1];
    expect(mid).toBeCloseTo(0.3, 2);
    // camera-induced failure (O1): no pair
    expect(refPairs({ ...base, fwdCode: SC.O1 }, () => 1, 0, false, P)).toEqual([0, 0, 0, 0, 0]);
    // added light seen by the canonical sample only (s = c, inverse undefined): a = w̃_c, b = 0, weighted by 1/P(s = c)
    expect(refPairs({ ...base, fwdCode: SC.O1, flags: 1 | 32 | 128, wc: 2, wp: 6, invCode: SC.O0_LIGHT }, () => 1, 1, true, P)).toEqual([0, 0, 8, 8, 2]);
    // DN-2: an added light C with share s_C of the new radiance: forward pairs (old lights) unchanged, s = c on a fraction
    // P(s = c) of the pixels; the selection-weighted inverse family estimates s_C (λ = max of the two families)
    const sC = 0.4, pSel = 0.25;
    const add = mk((q) => (q % 4 === 0
      ? refPairs({ ...base, qPrime: q, wp: 5 * 1, flags: 1 | 32 | 128, wc: pSel / (1 - pSel) * 5, invCode: (q % 20 < 8 ? SC.O0_LIGHT : SC.OK), piRecomp: 1 }, () => 1, 1, true, P)
      : refPairs({ ...base, qPrime: q, wp: 5 * 1 }, () => 1, 0, true, P)));
    expect(add.lambda[Math.floor(dnTiles(W, H)[0] / 2) * dnTiles(W, H)[0] + 1]).toBeCloseTo(sC, 1);
    // firefly cap
    expect(refPairs({ ...base, wp: 5 * 1e6 }, () => 0, 0, false, P)).toEqual([1e4, 1e4, 0, 0, 1]);
  });
});

describe('T16, registry, schedules, code-hash isolation', () => {
  it('denoiserT16Problems', () => {
    expect(denoiserT16Problems({ t16: { denoiser: 'none' } }, 'x', true)).toEqual([]);
    expect(denoiserT16Problems({}, 'x', false)).toEqual([]);
    expect(denoiserT16Problems({}, 'x', true)[0]).toMatch(/no t16.denoiser/);
    expect(denoiserT16Problems({ t16: { denoiser: 'ACTIVE' } }, 'x', false)[0]).toMatch(/denoiser active/);
  });
  it('registry: none ⇔ no live denoiser object', () => {
    expect(denoiserT16State()).toBe('none');
    const d = {};
    registerDenoiser(d, () => 'test');
    expect(liveDenoisers()).toBe(1);
    expect(denoiserT16State()).toMatch(/^1 live: test/);
    unregisterDenoiser(d);
    expect(denoiserT16State()).toBe('none');
  });
  it('recovery schedules hold each state; step frames at warm + k·hold', () => {
    const s = recoverySchedule(RECOVERY_RUNS[0]);
    expect(s.steps).toEqual([32, 56, 80]);
    expect(s.pkgFrames.length).toBe(104);
    expect([s.pkgFrames[31], s.pkgFrames[32], s.pkgFrames[56], s.pkgFrames[80], s.pkgFrames[103]]).toEqual([7, 8, 14, 20, 20]);
    expect(recoverySchedule(RECOVERY_RUNS[1]).steps).toEqual([32]);
    expect(parsePkgFrames('13x3,14,-1x2')).toEqual([13, 13, 13, 14, -1, -1]);
  });
  it('DN10: no denoiser code in the validation runners closures or under src/core/shaders', () => {
    const closure = new Set([...tsClosure(['validation/harness/batch-run.ts']), ...tsClosure(['validation/harness/restir-batch-run.ts']), ...tsClosure(['validation/harness/restir-chain-run.ts'])]);
    for (const f of closure) expect(f.includes('render/denoise/'), f).toBe(false);
    const walk = (d: string): string[] => readdirSync(d, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]));
    expect(walk(path.join(ROOT, 'src/core/shaders')).filter((f) => /dn-|denois/.test(f))).toEqual([]);
  });
});
