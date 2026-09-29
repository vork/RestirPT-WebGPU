// M4 milestone gate (restir-api.md §6.3/§6.4, PLAN §5 M4 exit, §7.1, §7.3, §7.4 M4): `npm run validate -- --milestone M4`.
//   Gate 0  typecheck, cpu lane (tests/restir + regressions), python stats tests, package determinism (make-m4.ts
//           twice, byte-identical; make-m3c.ts generates the env scenes), the Chrome GPU suites restir-initial,
//           restir-shift, restir-spatial, restir-debug and the M3 regressions pt, bsdf, lights, env-sampling, glass,
//           pt-glass (the refactors are bit-identical), budget.json M4 rows, the M4 app smoke (m4-app-smoke.ts, WP-D).
//   Gate 3  Stage B: our ReSTIR ≡ our PT (compare.py stage B: TOST δ 0.2% global / 1% per 32² tile, Y/R/G/B, Šidák
//           tiles, χ²_red, mean-t, KS/AD under the suite FWER, num_eps 1e-4, min_replicates 16 / heavy-tail 32) per
//           scene on the ladder 3.1 initial → 3.1b + RR → 3.2 + spatial (offline); the ladder stops at the first failing
//           rung (after its disjoint-seed re-run) and the later rungs are "not run". Plus the ensemble unit (C0q(d)
//           rung 3.2, E = 16, ensemble.npz) and the 2022-criteria unit ((i), preset criteria2022).
//           PT references (--kernel pt, RR off, seed 4001 / re-run 104001) are cached in validation/out/m4/ptrefs,
//           keyed by package bytes + render config + a hash of the PT's WGSL include closure and TS import closure.
//           Sizing (PLAN §7.3): a pilot per scene (PT) and per rung (ReSTIR) gives per-sample SDs of every aggregate;
//           spp (PT, shared by the rungs) and frames per batch (ReSTIR) minimise the GPU time subject to
//           SE_Δ ≤ δ/(t_{1−α,B−1} + z_{1−0.005/m}) on every tile and the global mean of every channel, ×1.25 margin.
//           A side above 60 min enlarges that unit's tiles to 64² (recorded as aggregate_enlarged); δ is never loosened.
//   Plants  (1) omitted spatial Jacobian (RSF_PLANT_NO_J, rung 3.2) on (i) and (v) V1; (2) marginal pdfs in J
//           (RSF_PLANT_MARGINAL_J) on (v) V2: each detected in ≥ 9/10 half-size repeats (compare.py --calibrate
//           --planted; the PT A/A control ≥ 9/10) and failing the full comparison; (3) synthetic W × 1.003 (compare.py
//           stage-B default plant) on a (i) rung-3.2 ReSTIR run, whose --calibrate A/A re-splits must also pass.
//   A/A     two ReSTIR seed sets (5002 vs 5003) on (i) rung 3.2 must pass Stage B. Calibration sides that are split or
//           compared at equal size are rendered at CALIB_FACTOR (4) × the unit size: the A/A pair, and the PT
//           references of the rendered plants (seed 4201).
//   T15/T16 per run: NaN/Inf = 0, negatives = 0, BVH overflow = 0, RSC error counters = 0, queue overflow = 0, no submit
//           over the hard cap; config: unbiased preset (plants only where named), internal scale 1, no denoiser,
//           linear-accumulation readback, jitter iid-per-run, maxBounces = the PT reference's, Mode A, same scene
//           bytes / env NEE as the reference; rung 3.2 must execute all its spatial rounds.
// Output: validation/out/m4-gate-<time>/ (summary.json, summary.md, sizing.json, budget-m4.json, tests/, compare/,
// restir/ runs). Options: --only a,b (scene subset; skips Gate 0, plants, A/A, ensemble/2022 units unless their scene is
// selected), --pilot-only (pilots + sizing only), --write-budget (merge the M4 rows into validation/budget.json).
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodePFM } from '../../src/core/io/pfm.ts';
import { withGpuLockSync } from './gpu-lock.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PY = path.join(ROOT, 'validation/.venv/bin/python');
const ENV_SCENES = 'validation/out/m3c/scenes';
const M4_OUT = 'validation/out/m4';
const PTREFS = `${M4_OUT}/ptrefs`;
const PILOTS = `${M4_OUT}/pilots`;
export const SEEDS = { pt: 4001, ptRerun: 104001, restir: 4002, restirRerun: 104002, aa: 5002, aa2: 5003, ptCalib: 4201, ptPilot: 4011, restirPilot: 4012, plantBase: 4101 } as const;
/**
 * Calibration runs compare HALF-size replicate sets (compare.py --calibrate) or two ReSTIR sets of equal size (A/A):
 * the side(s) they split are rendered at CALIB_FACTOR × the unit's size so the half-size A/A control is powered like a
 * full Stage-B unit (a unit sized to SE_Δ ≈ target leaves its halves ~1.3–1.6× above target and TOST then fails at the
 * worst of 256 tiles). The planted runs themselves stay at 1× (P = a half of them).
 */
export const CALIB_FACTOR = 4;
export const NUM_EPS = 1e-4;
const NUM_EPS_NOTE = 'two f32 implementations (cycles-deviations.md D2; restir-api.md §6.3)';
/** Per-side GPU time above which a unit's tiles are enlarged to 64² (restir-api.md §6.3). */
export const SIDE_CAP_S = 3600;
export const SIZING_MARGIN = 1.25;
/**
 * Floors of the per-batch sample counts, so every batch mean averages enough i.i.d. samples for the replicate
 * statistics (Welch/TOST, mean-t, KS/AD) to see a near-normal statistic even on the smoothest scenes (a 1-frame
 * batch of c0r_mirror would be a single heavy-tailed sample): PT 256 spp, rungs 3.1/3.1b 128 frames, 3.2 8 frames
 * (8 × S = 256 path trees). They only ever raise a size.
 */
export const MIN_PT_SPP = 256;
export const minFramesPerBatch = (rung: string): number => (rung === '3.2' ? 8 : 128);
const TIMEOUT_MS = 12 * 3600_000;   // a run may wait hours for the shared GPU lock

export type RungId = '3.1' | '3.1b' | '3.2';
export type Preset = 'initial' | 'initial-rr' | 'offline' | 'criteria2022';
export const M4_RUNGS: { id: RungId; preset: Preset; label: string }[] = [
  { id: '3.1', preset: 'initial', label: 'initial RIS only' },
  { id: '3.1b', preset: 'initial-rr', label: '+ RR' },
  { id: '3.2', preset: 'offline', label: '+ spatial (offline: S 32, 3 rounds x 6 slots, R 10)' },
];

export interface M4Scene { pkg: string; label: string; env: boolean; tier: 'tight' | 'heavy-tail' }
const S = (pkg: string, label: string, o: Partial<M4Scene> = {}): M4Scene => ({ pkg, label, env: false, tier: 'tight', ...o });
const E = (pkg: string, label: string, o: Partial<M4Scene> = {}): M4Scene => S(pkg, label, { env: true, ...o });

/** Gate-3 scene list of restir-api.md §6.3 (PLAN §5 M4 exit: (i)–(vi), (x), (xi), (xii), C0q(d), C0r, (xiii), (xiv)). */
export const M4_SCENES: M4Scene[] = [
  S('cornell_i_512', '(i) diffuse Cornell, rect light'),
  S('ii_cornell_point_512', '(ii) Cornell, point light'),
  S('iii_spot_grazing_512', '(iii) spot grazing a wall'),
  S('iv_emissive_mesh_512', '(iv) Cornell + emissive icosphere + rect'),
  S('v_glossy_v1_sharp_512', '(v) V1 GGX r 0.05/0.1, emissive quads'),
  S('v_glossy_v1_512', '(v) V1 GGX r 0.2-0.8, rect lights'),
  S('v_glossy_v2_512', '(v) V2 Principled metal/dielectric'),
  S('vi_glass_mirror_A_512', '(vi) glass + mirror + glossy, Mode A'),
  S('x_many_lights_512', '(x) many lights'),
  S('xi_contact_512', '(xi) contact geometry'),
  S('xii_alpha_foliage_512', '(xii) alpha MASK foliage cards'),
  E('c0q_openbox_b13_256', 'C0q(d) rho=1 open box b=13, constant env'),
  E('c0r_irradiance_256', 'C0r Lambert irradiance sphere, overcast'),
  E('c0r_mirror_256', 'C0r mirror sphere, overcast'),
  E('xiii_spheres_512x256', '(xiii) GGX + V2 metal spheres, studio_small_09', { tier: 'heavy-tail' }),
  E('xiv_overcast_b1_512', '(xiv) open Cornell, overcast, b=1'),
  E('xiv_overcast_b3_512', '(xiv) open Cornell, overcast, b=3'),
  E('xiv_overcast_b7_512', '(xiv) open Cornell, overcast, b=7'),
  E('xiv_overcast_rect_b3_512', '(xiv) open Cornell, overcast + rect, b=3'),
  E('xiv_kloof_b3_512', '(xiv) open Cornell, kloofendal (sun), b=3', { tier: 'heavy-tail' }),
  E('xiv_kloof_rect_b3_512', '(xiv) open Cornell, kloofendal + rect, b=3', { tier: 'heavy-tail' }),
];
export const ENSEMBLE_UNIT = { pkg: 'c0q_openbox_b13_256', members: 16 } as const;
export const CRIT2022_PKG = 'cornell_i_512';
export const PLANTS: { name: string; tag: string; pkg: string; plant: 'no-j' | 'marginal-j'; seed: number }[] = [
  { name: 'omitted spatial Jacobian (RSF_PLANT_NO_J)', tag: 'noJ', pkg: 'cornell_i_512', plant: 'no-j', seed: SEEDS.plantBase },
  { name: 'omitted spatial Jacobian (RSF_PLANT_NO_J)', tag: 'noJ', pkg: 'v_glossy_v1_512', plant: 'no-j', seed: SEEDS.plantBase + 1 },
  { name: 'marginal pdfs in J (RSF_PLANT_MARGINAL_J)', tag: 'marginalJ', pkg: 'v_glossy_v2_512', plant: 'marginal-j', seed: SEEDS.plantBase + 2 },
];
export const AA_PKG = 'cornell_i_512';
export const GPU_SUITES: [string, string][] = [
  ['restir-initial', 'U-RES-1, U-EP-1, U-PT-BITS, U-RIS-1..4, U-SFX-1 (WP-A)'],
  ['restir-shift', 'T2, T3-0..T3-5, T3-D, T3-ENV, T4, T5/U7, U5, U-11..U-13, dense PSS, U-CASE (WP-B)'],
  ['restir-spatial', 'T3-3/M4, T6(a,b), T7, U-MIS-1, T17, U-ENS-1/2, U-OFF-1 (WP-C)'],
  ['restir-debug', 'U-DBG-1..3 (WP-D)'],
  ['pt', 'M3 regression: T10, U11, C0 probes, split dispatch, plants'],
  ['bsdf', 'M3 regression: T8'],
  ['lights', 'M3 regression: T9'],
  ['env-sampling', 'M3 regression: ENV-U3/U4/U6'],
  ['glass', 'M3 regression: U-G1..U-G10'],
  ['pt-glass', 'M3 regression: C0h/G1, U9, U10'],
];

/** Suite FWER units: every scene × rung + ensemble + 2022 + the rendered plants + the synthetic plant + A/A, × {Y,R,G,B}. */
export function nUnits(): number { return 4 * (M4_SCENES.length * M4_RUNGS.length + 2 + PLANTS.length + 1 + 1); }   // A/A and W×1.003 count once each

// ------------------------------------------------------------------------------------------------ statistics helpers

/** Standard normal quantile (Acklam, |rel err| < 1.2e-9). */
export function normInv(p: number): number {
  if (!(p > 0 && p < 1)) throw new Error(`normInv(${p})`);
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pl) return -normInv(1 - p);
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/** Student-t quantile by the Cornish–Fisher expansion (Abramowitz–Stegun 26.7.5); < 2e-3 abs error for ν ≥ 7. */
export function tInv(p: number, nu: number): number {
  const z = normInv(p);
  if (!Number.isFinite(nu)) return z;
  const z3 = z ** 3, z5 = z ** 5, z7 = z ** 7, z9 = z ** 9;
  const g1 = (z3 + z) / 4, g2 = (5 * z5 + 16 * z3 + 3 * z) / 96, g3 = (3 * z7 + 19 * z5 + 17 * z3 - 15 * z) / 384;
  const g4 = (79 * z9 + 776 * z7 + 1482 * z5 - 1920 * z3 - 945 * z) / 92160;
  return z + g1 / nu + g2 / nu ** 2 + g3 / nu ** 3 + g4 / nu ** 4;
}

/** stats.sizing_target: δ / (t_{1−α,ν} + z_{1−0.005/m}). */
export function sizingTarget(delta: number, m: number, nu: number, alpha = 0.01): number {
  return delta / (tInv(1 - alpha, nu) + normInv(1 - 0.005 / Math.max(m, 1)));
}

/** Smallest m·2^k (m ∈ {4,5,6,7}) or 1/2/3 that is ≥ x: stable sizes so cached references are reused. */
export function niceCeil(x: number): number {
  if (!(x > 0)) return 1;
  if (x <= 4) return Math.ceil(x);
  let k = 0;
  while (7 * 2 ** k < x) k++;
  for (const m of [4, 5, 6, 7]) if (m * 2 ** k >= x) return m * 2 ** k;
  return 8 * 2 ** k;
}

/** Replicate batch-mean images of one side (row 0 = top, RGB). */
export interface PilotSide { B: number; n: number; msPerSample: number; W: number; H: number; batches: Float32Array[] }

const LUMA = [0.2126, 0.7152, 0.0722];

/**
 * Per-aggregate replicate mean / SD (over batches) of the channels Y,R,G,B: layout [channel][0 = global, 1… = tiles
 * (row-major, edge tiles partial)]. SD is the sample SD of the batch means.
 */
export function aggregateSide(s: PilotSide, tile: number): { mean: Float64Array; sd: Float64Array; nTiles: number } {
  const th = Math.ceil(s.H / tile), tw = Math.ceil(s.W / tile), nT = th * tw, nA = 1 + nT;
  const sum = new Float64Array(4 * nA), sq = new Float64Array(4 * nA);
  const cnt = new Float64Array(nT);
  for (let y = 0; y < s.H; y++) for (let x = 0; x < s.W; x++) cnt[Math.floor(y / tile) * tw + Math.floor(x / tile)]++;
  const acc = new Float64Array(4 * nA);
  for (const img of s.batches) {
    acc.fill(0);
    for (let y = 0; y < s.H; y++) {
      const trow = Math.floor(y / tile) * tw;
      for (let x = 0; x < s.W; x++) {
        const i = 3 * (y * s.W + x), t = 1 + trow + Math.floor(x / tile);
        const r = img[i], g = img[i + 1], b = img[i + 2];
        const Y = LUMA[0] * r + LUMA[1] * g + LUMA[2] * b;
        const v = [Y, r, g, b];
        for (let c = 0; c < 4; c++) { acc[c * nA] += v[c]; acc[c * nA + t] += v[c]; }
      }
    }
    for (let c = 0; c < 4; c++) {
      acc[c * nA] /= s.W * s.H;
      for (let t = 0; t < nT; t++) acc[c * nA + 1 + t] /= cnt[t];
    }
    for (let k = 0; k < 4 * nA; k++) { sum[k] += acc[k]; sq[k] += acc[k] * acc[k]; }
  }
  const B = s.batches.length;
  const mean = new Float64Array(4 * nA), sd = new Float64Array(4 * nA);
  for (let k = 0; k < 4 * nA; k++) {
    mean[k] = sum[k] / B;
    sd[k] = B > 1 ? Math.sqrt(Math.max(0, (sq[k] - B * mean[k] * mean[k]) / (B - 1))) : 0;
  }
  return { mean, sd, nTiles: nT };
}

/**
 * u(a) = per-sample variance of aggregate a relative to its margin denominator, divided by the squared target:
 * SE_Δ(a) ≤ T(a)·D(a) ⇔ Σ_sides u_side(a)/N_side ≤ 1 (N = total samples of the side). D = max(R̄_a, 0.05·R̄_image)
 * from the reference (PT) pilot (compare.py dark-tile rule); aggregates with D = 0 are skipped (u = 0).
 */
export function uVector(side: { sd: Float64Array; n: number }, ref: { mean: Float64Array; nTiles: number }, o: { deltaGlobal: number; deltaTile: number; B: number }): Float64Array {
  const nA = 1 + ref.nTiles, u = new Float64Array(4 * nA);
  const Tg = sizingTarget(o.deltaGlobal, 1, o.B - 1), Tt = sizingTarget(o.deltaTile, ref.nTiles, o.B - 1);
  for (let c = 0; c < 4; c++) {
    const img = ref.mean[c * nA];
    for (let a = 0; a < nA; a++) {
      const k = c * nA + a;
      const D = Math.max(ref.mean[k], 0.05 * img);
      if (!(D > 0)) continue;
      const T = a === 0 ? Tg : Tt;
      u[k] = (side.sd[k] * side.sd[k] * side.n) / (D * D * T * T);
    }
  }
  return u;
}

export interface RungSizingInput { id: string; side: PilotSide; tile: 32 | 64 }
export interface SceneSizing {
  B: number;
  ptSamples: number; ptSpp: number; ptSeconds: number;
  rungs: Record<string, { tile: 32 | 64; samples: number; framesPerBatch: number; seconds: number; msPerFrame: number; enlarged: boolean }>;
  feasible: boolean; notes: string[];
}

/**
 * Joint allocation (PLAN §7.3 sizing rule): the PT reference is shared by the rungs; for each candidate total PT
 * sample count N_R the rung needs N_k = max_a u_k(a)/(1 − u_R(a)/N_R); pick N_R minimising c_R·N_R + Σ c_k·N_k, then
 * × margin and round per batch with niceCeil. A side above SIDE_CAP_S enlarges that rung's tiles to 64² (the PT side
 * above the cap enlarges every rung); δ is never loosened.
 */
export function sizeScene(pt: PilotSide, rungsIn: RungSizingInput[], o: { B: number; deltaGlobal?: number; deltaTile?: number; margin?: number; capS?: number; minPtSpp?: number; minFrames?: (id: string) => number }): SceneSizing {
  const dG = o.deltaGlobal ?? 0.002, dT = o.deltaTile ?? 0.01, margin = o.margin ?? SIZING_MARGIN, cap = o.capS ?? SIDE_CAP_S;
  const rungs = rungsIn.map((r) => ({ ...r }));
  const notes: string[] = [];
  const cache = new Map<number, { ref: ReturnType<typeof aggregateSide> }>();
  const refAgg = (tile: number) => { let c = cache.get(tile); if (!c) { c = { ref: aggregateSide(pt, tile) }; cache.set(tile, c); } return c.ref; };
  const solve = () => {
    const uR: Float64Array[] = [], uK: Float64Array[] = [];
    for (const r of rungs) {
      const ref = refAgg(r.tile);
      uR.push(uVector({ sd: ref.sd, n: pt.n }, ref, { deltaGlobal: dG, deltaTile: dT, B: o.B }));
      const side = aggregateSide(r.side, r.tile);
      uK.push(uVector({ sd: side.sd, n: r.side.n }, ref, { deltaGlobal: dG, deltaTile: dT, B: o.B }));
    }
    let uRmax = 0;
    for (const u of uR) for (const v of u) uRmax = Math.max(uRmax, v);
    const cR = pt.msPerSample;
    let best: { NR: number; NK: number[]; cost: number } | undefined;
    const lo = Math.max(uRmax * 1.02, 1);
    for (let i = 0; i <= 400; i++) {
      const NR = lo * 1000 ** (i / 400);
      const NK = rungs.map((_, j) => {
        let m = 0;
        for (let a = 0; a < uK[j].length; a++) {
          const rem = 1 - uR[j][a] / NR;
          if (uK[j][a] > 0) m = Math.max(m, uK[j][a] / rem);
        }
        return Math.max(m, 1);
      });
      const cost = cR * NR + NK.reduce((s, n, j) => s + rungs[j].side.msPerSample * n, 0);
      if (!best || cost < best.cost) best = { NR, NK, cost };
    }
    return best!;
  };
  let sol = solve();
  for (let iter = 0; iter < 3; iter++) {
    let changed = false;
    const ptS = (sol.NR * margin * pt.msPerSample) / 1000;
    if (ptS > cap && rungs.some((r) => r.tile === 32)) {
      for (const r of rungs) r.tile = 64;
      notes.push(`PT side ${(ptS / 60).toFixed(0)} min > ${cap / 60} min: every rung enlarged to 64² tiles`);
      changed = true;
    } else {
      rungs.forEach((r, j) => {
        const s = (sol.NK[j] * margin * r.side.msPerSample) / 1000;
        if (s > cap && r.tile === 32) { r.tile = 64; notes.push(`rung ${r.id}: ${(s / 60).toFixed(0)} min > ${cap / 60} min: 64² tiles`); changed = true; }
      });
    }
    if (!changed) break;
    sol = solve();
  }
  const ptSpp = Math.max(o.minPtSpp ?? MIN_PT_SPP, niceCeil((sol.NR * margin) / o.B));
  const out: SceneSizing = {
    B: o.B, ptSamples: ptSpp * o.B, ptSpp, ptSeconds: (ptSpp * o.B * pt.msPerSample) / 1000, rungs: {}, feasible: true, notes,
  };
  rungs.forEach((r, j) => {
    const f = Math.max((o.minFrames ?? minFramesPerBatch)(r.id), niceCeil((sol.NK[j] * margin) / o.B));
    const seconds = (f * o.B * r.side.msPerSample) / 1000;
    out.rungs[r.id] = { tile: r.tile, samples: f * o.B, framesPerBatch: f, seconds, msPerFrame: r.side.msPerSample, enlarged: r.tile !== rungsIn[j].tile };
    if (seconds > cap) { out.feasible = false; notes.push(`rung ${r.id}: ${(seconds / 60).toFixed(0)} min even at 64² tiles (weekly tier)`); }
  });
  if (out.ptSeconds > cap) notes.push(`PT reference ${(out.ptSeconds / 60).toFixed(0)} min (weekly tier)`);
  return out;
}

// ------------------------------------------------------------------------------------------------ T16

/** T16 config assertions of a ReSTIR run against its PT reference (restir-api.md §6.1 T15/T16). Pure. */
export function t16Problems(rs: Record<string, any>, pt: Record<string, any>, o: { plant?: boolean; rounds?: number; members?: number } = {}): string[] {
  const p: string[] = [];
  const t = rs.t16 ?? {};
  if (rs.kernel !== 'restir') p.push('ReSTIR meta: kernel is not restir');
  if (!rs.ok) p.push(`ReSTIR run errors: ${(rs.errors ?? []).join('; ')}`);
  if (!pt.ok) p.push(`PT reference errors: ${(pt.errors ?? []).join('; ')}`);
  if (!o.plant && t.validationModeUnbiased !== true) p.push('validation mode not unbiased (biased preset or an unnamed plant)');
  if (o.plant && !(t.plantsNamed?.length > 0)) p.push('plant run without a named plant');
  if (t.internalScale !== 1) p.push(`internal scale ${t.internalScale}`);
  if (t.denoiser !== 'none' || t.upscaler !== 'none') p.push('denoiser/upscaler active');
  if (!/^linear /.test(t.readback ?? '')) p.push('readback is not the linear radiance (accumulation buffer / ensemble sums)');
  if (t.jitterMode !== 'iid-per-run' || rs.config?.jitter !== 'iid-per-run') p.push('jitter is not iid-per-run');
  if (pt.config?.jitter !== 'iid-per-run') p.push('PT reference jitter is not iid-per-run');
  if (t.maxBounces !== pt.config?.maxBounces) p.push(`maxBounces ${t.maxBounces} != PT reference ${pt.config?.maxBounces}`);
  if (t.lightMode !== 'A' || pt.config?.lightMode !== 'A') p.push('not Mode A on both sides');
  if (pt.config?.rr !== false) p.push('PT reference has RR on');
  if (rs.config?.scene !== pt.config?.scene) p.push('scene bytes differ (package sha256)');
  if (rs.width !== pt.width || rs.height !== pt.height) p.push(`resolution ${rs.width}x${rs.height} != PT ${pt.width}x${pt.height}`);
  const rn = typeof rs.config?.env === 'object' ? rs.config.env.nee : 'none', pn = typeof pt.config?.env === 'object' ? pt.config.env.nee : 'none';
  if (rn !== pn) p.push(`env NEE ${rn} != PT ${pn}`);
  if ((o.members ?? 1) !== (rs.members ?? 1)) p.push(`members ${rs.members} != ${o.members ?? 1}`);
  if (o.rounds !== undefined && t.spatialRoundsExecuted !== o.rounds) p.push(`spatial stage executed ${t.spatialRoundsExecuted}/${o.rounds} rounds (stubbed stage)`);
  return p;
}

// ------------------------------------------------------------------------------------------------ code / package hashes

const sha = (b: string | Buffer) => createHash('sha256').update(b).digest('hex');

/** TS import closure (relative imports only) of the entry files; paths relative to ROOT, sorted. */
export function tsClosure(entries: string[]): string[] {
  const seen = new Set<string>();
  const stack = entries.map((e) => path.resolve(ROOT, e));
  while (stack.length) {
    const f = stack.pop()!;
    if (seen.has(f) || !existsSync(f)) continue;
    seen.add(f);
    const src = readFileSync(f, 'utf8');
    for (const m of src.matchAll(/(?:from\s+|import\s*\(\s*|import\s+)['"](\.{1,2}\/[^'"]+)['"]/g)) {
      const spec = m[1].replace(/\?.*$/, '');
      const r = path.resolve(path.dirname(f), spec);
      for (const c of [r, `${r}.ts`, path.join(r, 'index.ts')]) if (existsSync(c) && statSync(c).isFile()) { stack.push(c); break; }
    }
  }
  return [...seen].map((f) => path.relative(ROOT, f)).sort();
}

/** WGSL #include closure of entry modules under src/core/shaders (conditional includes are all followed). */
export function wgslClosure(entries: string[]): string[] {
  const base = path.join(ROOT, 'src/core/shaders');
  const seen = new Set<string>();
  const stack = [...entries];
  while (stack.length) {
    const f = stack.pop()!;
    if (seen.has(f)) continue;
    const p = path.join(base, f);
    if (!existsSync(p)) continue;
    seen.add(f);
    for (const m of readFileSync(p, 'utf8').matchAll(/^\s*#include\s+"([^"]+)"/gm)) stack.push(m[1]);
  }
  return [...seen].map((f) => `src/core/shaders/${f}`).sort();
}

function hashFiles(rel: string[]): string {
  return sha(rel.map((r) => `${r}\0${sha(readFileSync(path.join(ROOT, r)))}\n`).join(''));
}

function listDir(rel: string): string[] {
  const out: string[] = [];
  const walk = (d: string) => {
    for (const n of readdirSync(path.join(ROOT, d)).sort()) {
      const p = path.join(d, n);
      if (statSync(path.join(ROOT, p)).isDirectory()) walk(p); else out.push(p);
    }
  };
  walk(rel);
  return out;
}

let codeHashCache: { pt: string; restir: string; ptFiles: number; restirFiles: number } | undefined;
/** PT: TS closure of batch-run.ts + WGSL closure of passes/pt.wgsl. ReSTIR: TS closure of restir-batch-run.ts + every WGSL file. */
export function codeHashes(): { pt: string; restir: string; ptFiles: number; restirFiles: number } {
  if (codeHashCache) return codeHashCache;
  const ptFiles = [...tsClosure(['validation/harness/batch-run.ts']), ...wgslClosure(['passes/pt.wgsl'])];
  const rsFiles = [...tsClosure(['validation/harness/restir-batch-run.ts']), ...listDir('src/core/shaders').filter((f) => f.endsWith('.wgsl'))];
  codeHashCache = { pt: hashFiles(ptFiles), restir: hashFiles([...new Set(rsFiles)].sort()), ptFiles: ptFiles.length, restirFiles: rsFiles.length };
  return codeHashCache;
}

function packageHash(dir: string): string {
  return hashFiles(readdirSync(path.join(ROOT, dir)).filter((n) => statSync(path.join(ROOT, dir, n)).isFile()).sort().map((n) => path.join(dir, n)));
}

const stableJson = (v: unknown): string => JSON.stringify(v, (_k, x) => (x && typeof x === 'object' && !Array.isArray(x)
  ? Object.fromEntries(Object.entries(x as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : 1))) : x));

// ------------------------------------------------------------------------------------------------ process helpers

const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const r4 = (x: unknown) => (typeof x === 'number' ? Number(x.toPrecision(4)) : x);
const pct = (x: number | undefined | null, d = 3) => (typeof x === 'number' ? `${(x * 100).toFixed(d)}%` : 'n/a');
const readJson = <T = Record<string, any>>(p: string): T => JSON.parse(readFileSync(path.join(ROOT, p), 'utf8')) as T;
const tryJson = (p: string): Record<string, any> | undefined => { try { return readJson(p); } catch { return undefined; } };

function sh(cmd: string, argv: string[], echo: (l: string) => boolean = () => true): { code: number; out: string; seconds: number } {
  const t0 = performance.now();
  const r = spawnSync(cmd, argv, { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 28, timeout: TIMEOUT_MS });
  const out = `${r.stdout ?? ''}${r.stderr ?? ''}`;
  for (const l of out.split('\n')) if (l && echo(l)) console.log(`  ${l}`);
  return { code: r.status ?? 1, out, seconds: (performance.now() - t0) / 1000 };
}

/** Vitest config: in a git worktree whose node_modules is a symlink, the local config keeps Vite's cache in the worktree. */
function vitestConfigArgs(): string[] {
  try {
    if (lstatSync(path.join(ROOT, 'node_modules')).isSymbolicLink() && existsSync(path.join(ROOT, 'vitest.m4local.config.ts'))) return ['--config', 'vitest.m4local.config.ts'];
  } catch { /* default config */ }
  return [];
}

export const sceneDir = (s: M4Scene): string => (s.env ? `${ENV_SCENES}/${s.pkg}` : `validation/scenes/${s.pkg}`);

function batchImages(dir: string, meta: Record<string, any>): PilotSide {
  const files = readdirSync(path.join(ROOT, dir)).filter((n) => /^batch_\d{3}\.pfm$/.test(n)).sort();
  const imgs = files.map((f) => decodePFM(new Uint8Array(readFileSync(path.join(ROOT, dir, f)))));
  const n = meta.sppPerBatch as number, B = files.length;
  const ms = (meta.timings.batchMs as number[]).reduce((a, b) => a + b, 0);
  return { B, n, msPerSample: ms / (B * n), W: imgs[0].width, H: imgs[0].height, batches: imgs.map((i) => i.data) };
}

// ------------------------------------------------------------------------------------------------ the gate

type Add = (name: string, ok: boolean, seconds: number, data?: unknown, detail?: string) => void;
type Rec = (name: string, ok: boolean, detail?: string) => void;
interface Run { dir: string; meta: Record<string, any>; seconds: number; cacheHit?: boolean }

export interface M4Options { only?: Set<string>; pilotOnly?: boolean; writeBudget?: boolean }

export function milestoneM4(record: Rec, o: M4Options = {}): void {
  const t0 = performance.now();
  const runId = `m4-gate-${stamp()}`;
  const dir = path.join('validation/out', runId);
  mkdirSync(path.join(ROOT, dir), { recursive: true });
  const steps: { name: string; ok: boolean; seconds: number; data?: unknown; detail?: string }[] = [];
  const add: Add = (name, ok, seconds, data, detail) => {
    steps.push({ name, ok, seconds: Math.round(seconds * 10) / 10, data, detail });
    record(name, ok, `${detail ? `${detail}, ` : ''}${seconds.toFixed(1)} s`);
  };
  const runStep = (name: string, cmd: string, argv: string[], echo?: (l: string) => boolean) => {
    console.log(`\n--- ${name}: ${cmd} ${argv.join(' ')}`);
    const r = sh(cmd, argv, echo);
    add(name, r.code === 0, r.seconds, undefined, `exit ${r.code}`);
    return r;
  };
  const full = !o.only && !o.pilotOnly;
  const scenes = M4_SCENES.filter((s) => !o.only || o.only.has(s.pkg));
  const N_UNITS = nUnits();
  const hashes = codeHashes();
  console.log(`M4 gate ${runId}: ${scenes.length} scenes, n_units ${N_UNITS}, PT code hash ${hashes.pt.slice(0, 12)} (${hashes.ptFiles} files), ReSTIR ${hashes.restir.slice(0, 12)} (${hashes.restirFiles} files)`);

  // ---- Gate 0 -------------------------------------------------------------------------------------------------------
  add('M4 gate tools (venv python)', existsSync(PY), 0, undefined, existsSync(PY) ? undefined : `missing ${PY}`);
  envScenes(scenes, dir, add);
  if (full) {
    packagesM4Deterministic(dir, add);
    runStep('typecheck', 'npx', ['tsc', '--noEmit']);
    runStep('vitest cpu (tests/restir: layout, rc-dual, mis, pairing, queue, npz, gate-m4-config + regressions)', 'npx',
      ['vitest', 'run', ...vitestConfigArgs(), '--project', 'cpu'], (l) => /Test Files|Tests |FAIL|✗|×/.test(l));
    runStep('python stats tests (validation/tools/tests)', PY, ['-m', 'pytest', 'validation/tools/tests', '-q'], (l) => /passed|failed|error/i.test(l));
    withGpuLockSync('gate-m4', () => {
      for (const [file, what] of GPU_SUITES) {
        const rel = `validation/gpu-tests/${file}.gpu.test.ts`;
        if (!existsSync(path.join(ROOT, rel))) { add(`${file} (chrome): ${what}`, false, 0, undefined, `missing ${rel}`); continue; }
        runStep(`${file} (chrome): ${what}`, 'npx', ['vitest', 'run', ...vitestConfigArgs(), '--project', 'chrome', '--reporter=verbose', rel],
          (l) => /Tests |FAIL|✗|×|AssertionError|LOGIC|FP-BOUNDARY/.test(l));
      }
    });
    budgetRowsPresent(add);
    if (existsSync(path.join(ROOT, 'validation/harness/m4-app-smoke.ts'))) {
      runStep('M4 app smoke (ReSTIR mode on Cornell + HDRI, every M4 view, inspector dump, HUD f_r)', 'npx',
        ['tsx', 'validation/harness/m4-app-smoke.ts', '--run', `${runId}-app-smoke`], (l) => /^(PASS|FAIL)\s/.test(l));
    } else add('M4 app smoke (validation/harness/m4-app-smoke.ts, WP-D)', false, 0, undefined, 'missing');
  }

  // ---- pilots + sizing ------------------------------------------------------------------------------------------------
  const sizing: Record<string, SceneSizing & { pilot: Record<string, string> }> = {};
  for (const s of scenes) {
    const z = pilotAndSize(s, dir, add);
    if (z) sizing[s.pkg] = z;
  }
  writeFileSync(path.join(ROOT, dir, 'sizing.json'), `${JSON.stringify(sizing, null, 1)}\n`);
  const budget = budgetRows(scenes, sizing);
  writeFileSync(path.join(ROOT, dir, 'budget-m4.json'), `${JSON.stringify(budget, null, 1)}\n`);
  if (o.writeBudget) mergeBudget(budget, runId, add);

  const results: Record<string, any>[] = [];
  if (!o.pilotOnly) {
    // ---- Gate 3: per-scene ladders ------------------------------------------------------------------------------------
    const ladders: Record<string, { reached: RungId[]; failedAt?: RungId }> = {};
    const firstRuns: Record<string, Run> = {};
    for (const s of scenes) {
      const z = sizing[s.pkg];
      const lad: { reached: RungId[]; failedAt?: RungId } = { reached: [] };
      ladders[s.pkg] = lad;
      if (!z) { for (const r of M4_RUNGS) results.push(notRun(s, r.id, 'no sizing (pilot failed)')); continue; }
      const ref = ptRef(s, z, SEEDS.pt, add);
      for (const r of M4_RUNGS) {
        if (lad.failedAt || !ref) { results.push(notRun(s, r.id, lad.failedAt ? `ladder stopped at ${lad.failedAt}` : 'PT reference failed')); continue; }
        const res = stageB(s, r.id, r.preset, z, ref, dir, runId, N_UNITS, add, { rounds: r.id === '3.2' ? 3 : 0 }, firstRuns);
        results.push(res);
        if (res.ok) lad.reached.push(r.id); else lad.failedAt = r.id;
      }
    }
    // ---- ensemble unit and 2022 unit ------------------------------------------------------------------------------------
    const ens = M4_SCENES.find((x) => x.pkg === ENSEMBLE_UNIT.pkg)!;
    if (sizing[ens.pkg]) {
      if (ladders[ens.pkg]?.failedAt) results.push(notRun(ens, '3.2', `ensemble unit: ladder stopped at ${ladders[ens.pkg].failedAt}`, 'ensemble'));
      else {
        const ref = ptRef(ens, sizing[ens.pkg], SEEDS.pt, add);
        if (ref) results.push(stageB(ens, '3.2', 'offline', sizing[ens.pkg], ref, dir, runId, N_UNITS, add, { rounds: 3, members: ENSEMBLE_UNIT.members, unit: `${ens.pkg}@3.2-ensemble` }, firstRuns));
      }
    }
    const c22 = M4_SCENES.find((x) => x.pkg === CRIT2022_PKG)!;
    if (sizing[c22.pkg]) {
      if (ladders[c22.pkg]?.failedAt) results.push(notRun(c22, '3.2', `2022 unit: ladder stopped at ${ladders[c22.pkg].failedAt}`, 'crit2022'));
      else {
        const ref = ptRef(c22, sizing[c22.pkg], SEEDS.pt, add);
        if (ref) results.push(stageB(c22, '3.2', 'criteria2022', sizing[c22.pkg], ref, dir, runId, N_UNITS, add, { rounds: 3, unit: `${c22.pkg}@3.2-crit2022` }, firstRuns));
      }
    }
    // ---- plants, synthetic W × 1.003 + calibrate A/A, A/A ----------------------------------------------------------------
    if (full) {
      for (const p of PLANTS) {
        const s = M4_SCENES.find((x) => x.pkg === p.pkg)!;
        const z = sizing[p.pkg] ?? pilotAndSize(s, dir, add);
        const ref = z && ptRef(s, { ...z, ptSpp: z.ptSpp * CALIB_FACTOR }, SEEDS.ptCalib, add);
        if (z && ref) results.push(plantDetection(s, p, z, ref, dir, runId, N_UNITS, add));
      }
      const s = M4_SCENES.find((x) => x.pkg === AA_PKG)!;
      const z = sizing[AA_PKG] ?? pilotAndSize(s, dir, add);
      const ref = z && ptRef(s, z, SEEDS.pt, add);
      if (z && ref) {
        const aa = restirAA(s, z, ref, dir, runId, N_UNITS, add);
        results.push(aa.result);
        if (aa.base) results.push(syntheticPlant(s, aa.base, z, dir, N_UNITS, add));
        else add(`synthetic W x1.003 plant on ${AA_PKG}`, false, 0, undefined, 'the A/A ReSTIR run failed');
      } else add(`A/A and synthetic W x1.003 on ${AA_PKG} rung 3.2`, false, 0, undefined, 'no sizing / PT reference');
    }
  }

  // ---- summary --------------------------------------------------------------------------------------------------------
  const failed = steps.filter((x) => !x.ok).map((x) => x.name);
  const perRung = Object.fromEntries(['3.1', '3.1b', '3.2'].map((r) => {
    const u = results.filter((x) => x.rung === r && x.kind === 'scene');
    return [r, { pass: u.filter((x) => x.ok).length, fail: u.filter((x) => !x.ok && x.status !== 'not run').length, notRun: u.filter((x) => x.status === 'not run').length }];
  }));
  const summary = {
    milestone: 'M4', gate: 'Gate 0 + Gate 3 Stage B (rungs 3.1, 3.1b, 3.2) + ensemble + 2022 criteria + plants + A/A (restir-api.md §6.3/§6.4)',
    run: runId, created: new Date().toISOString(), ok: failed.length === 0, subset: o.only ? [...o.only] : undefined, pilotOnly: !!o.pilotOnly,
    total_s: Math.round((performance.now() - t0) / 100) / 10, n_units: N_UNITS, code_hashes: { pt: hashes.pt, restir: hashes.restir },
    per_rung: perRung, failed, units: results, steps,
  };
  writeFileSync(path.join(ROOT, dir, 'summary.json'), `${JSON.stringify(summary, null, 1)}\n`);
  writeFileSync(path.join(ROOT, dir, 'summary.md'), summaryMd(summary, results, sizing));
  console.log(`\n${table(results)}\nsummary: ${dir}/summary.json (${summary.total_s} s)`);
}

function notRun(s: M4Scene, rung: RungId, why: string, kind = 'scene'): Record<string, any> {
  const unit = kind === 'scene' ? `${s.pkg}@${rung}` : kind === 'ensemble' ? `${s.pkg}@3.2-ensemble` : `${s.pkg}@3.2-crit2022`;
  return { unit, kind, scene: s.pkg, rung, tier: s.tier, status: 'not run', ok: false, note: why };
}

/** make-m3c.ts must have generated the env packages the gate uses (validation/out/m3c/scenes; M3c gate: deterministic). */
function envScenes(scenes: M4Scene[], dir: string, add: Add): void {
  const need = [...new Set([...scenes, ...M4_SCENES.filter((s) => s.pkg === ENSEMBLE_UNIT.pkg)].filter((s) => s.env).map((s) => s.pkg))];
  const missing = need.filter((p) => !existsSync(path.join(ROOT, ENV_SCENES, p, 'scene.json')));
  if (!missing.length) { add(`env scene packages present (${need.length}, make-m3c.ts)`, true, 0); return; }
  const t0 = performance.now();
  const r = sh('npx', ['tsx', 'validation/scenes/make-m3c.ts', ENV_SCENES, '--only', missing.join(',')], () => false);
  const still = missing.filter((p) => !existsSync(path.join(ROOT, ENV_SCENES, p, 'scene.json')));
  add(`env scene packages generated (make-m3c.ts --only ${missing.join(',')})`, r.code === 0 && !still.length, (performance.now() - t0) / 1000, { missing: still },
    still.length ? `missing ${still.join(', ')} (${r.out.slice(-300)})` : undefined);
  void dir;
}

/** make-m4.ts (WP-B's T3 fixtures) writes byte-identical packages twice. */
function packagesM4Deterministic(dir: string, add: Add): void {
  const gen = 'validation/scenes/make-m4.ts';
  if (!existsSync(path.join(ROOT, gen))) { add('make-m4.ts package determinism (WP-B)', false, 0, undefined, `missing ${gen}`); return; }
  const t0 = performance.now();
  const a = path.join(dir, 'make-m4-a'), b = path.join(dir, 'make-m4-b');
  const ra = sh('npx', ['tsx', gen, a], () => false), rb = sh('npx', ['tsx', gen, b], () => false);
  const diffs: string[] = [];
  if (ra.code !== 0 || rb.code !== 0) diffs.push(`exit ${ra.code}/${rb.code}: ${(ra.out + rb.out).slice(-300)}`);
  const files = (d: string) => (existsSync(path.join(ROOT, d)) ? listDir(d).map((f) => path.relative(d, f)) : []);
  const fa = files(a), fb = files(b);
  for (const f of new Set([...fa, ...fb])) {
    if (!fa.includes(f) || !fb.includes(f) || !readFileSync(path.join(ROOT, a, f)).equals(readFileSync(path.join(ROOT, b, f)))) diffs.push(f);
  }
  if (!fa.length && !diffs.length) diffs.push('no files written');
  rmSync(path.join(ROOT, a), { recursive: true, force: true });
  rmSync(path.join(ROOT, b), { recursive: true, force: true });
  add('make-m4.ts package determinism (twice, byte-identical)', diffs.length === 0, (performance.now() - t0) / 1000, { files: fa.length, diffs: diffs.slice(0, 20) },
    diffs.length ? `diffs: ${diffs.slice(0, 6).join(', ')}` : `${fa.length} files`);
}

function budgetRowsPresent(add: Add): void {
  const b = readJson('validation/budget.json') as { m4_entries?: { scene: string; rung: string }[] };
  const want = M4_SCENES.flatMap((s) => M4_RUNGS.map((r) => `${s.pkg}@${r.id}`));
  const have = new Set((b.m4_entries ?? []).map((e) => `${e.scene}@${e.rung}`));
  const miss = want.filter((w) => !have.has(w));
  add('budget.json M4 rows (every scene × rung; gate-m4 --pilot-only --write-budget)', miss.length === 0, 0, { rows: b.m4_entries?.length ?? 0 },
    miss.length ? `missing ${miss.length}: ${miss.slice(0, 4).join(', ')}` : `${have.size} rows`);
}

/** run-batches.ts into validation/out/<run> (the GPU lock is taken inside), then moved to `dest`. */
function runBatches(argv: string[], run: string, dest: string): { dir?: string; meta?: Record<string, any>; code: number; out: string; seconds: number } {
  const r = sh('npx', ['tsx', 'validation/harness/run-batches.ts', ...argv, '--run', run], (l) => /^(FAIL)\s|errors:|Error|lock wait/.test(l));
  const from = path.join(ROOT, 'validation/out', run);
  if (!existsSync(from)) return { ...r };
  mkdirSync(path.dirname(path.join(ROOT, dest)), { recursive: true });
  rmSync(path.join(ROOT, dest), { recursive: true, force: true });
  renameSync(from, path.join(ROOT, dest));
  return { ...r, dir: dest, meta: tryJson(path.join(dest, 'meta.json')) };
}

function cachedRun(cacheRoot: string, keyObj: Record<string, unknown>, label: string, argv: string[], B: number, add: Add, stepName: string): Run | undefined {
  const key = sha(stableJson(keyObj)).slice(0, 16);
  const dest = path.join(cacheRoot, `${label}-${key}`);
  const meta = tryJson(path.join(dest, 'meta.json'));
  const nFiles = existsSync(path.join(ROOT, dest)) ? readdirSync(path.join(ROOT, dest)).filter((n) => /^batch_\d{3}\.pfm$/.test(n)).length : 0;
  if (meta?.ok && nFiles === B) {
    add(stepName, true, 0, { dir: dest, cache_hit: true, key }, 'cache hit');
    return { dir: dest, meta, seconds: 0, cacheHit: true };
  }
  console.log(`\n--- ${stepName}: ${argv.join(' ')}`);
  const r = runBatches(argv, `m4c-${label}-${key}`.slice(0, 120), dest);
  const ok = r.code === 0 && !!r.meta?.ok;
  if (r.dir) writeFileSync(path.join(ROOT, dest, 'cache-key.json'), `${JSON.stringify(keyObj, null, 1)}\n`);
  add(stepName, ok, r.seconds, { dir: dest, cache_hit: false, key, gpu_s: r.meta && r.meta.timings.totalMs / 1000 },
    ok ? `rendered in ${(r.meta!.timings.totalMs / 1000).toFixed(1)} s` : `exit ${r.code} ${(r.meta?.errors ?? []).join('; ')} ${r.out.slice(-300)}`);
  return ok ? { dir: dest, meta: r.meta!, seconds: r.seconds, cacheHit: false } : undefined;
}

const pilotB = (s: M4Scene) => (s.tier === 'heavy-tail' ? 32 : 16);
const PILOT_PT_SPP = 128;
const pilotFrames = (rung: RungId) => (rung === '3.2' ? 8 : 128);

function ptRunKey(s: M4Scene, spp: number, B: number, seed: number) {
  return { kind: 'pt', pkg: s.pkg, packageHash: packageHash(sceneDir(s)), spp, B, seed, rr: false, code: codeHashes().pt };
}

/** PT pilot + one ReSTIR pilot per rung (cached), then the joint sizing (PLAN §7.3). */
function pilotAndSize(s: M4Scene, _dir: string, add: Add): (SceneSizing & { pilot: Record<string, string> }) | undefined {
  const B = pilotB(s);
  const pt = cachedRun(PILOTS, ptRunKey(s, PILOT_PT_SPP, B, SEEDS.ptPilot), `${s.pkg}-pt`,
    ['--package', sceneDir(s), '--kernel', 'pt', '--spp', String(PILOT_PT_SPP), '--batches', String(B), '--seed', String(SEEDS.ptPilot)],
    B, add, `pilot PT ${s.pkg} (${PILOT_PT_SPP} spp x ${B})`);
  if (!pt) return undefined;
  const rungs: RungSizingInput[] = [];
  const pilot: Record<string, string> = { pt: pt.dir };
  for (const r of M4_RUNGS) {
    const f = pilotFrames(r.id);
    const run = cachedRun(PILOTS, { kind: 'restir', pkg: s.pkg, packageHash: packageHash(sceneDir(s)), preset: r.preset, frames: f, B, seed: SEEDS.restirPilot, code: codeHashes().restir },
      `${s.pkg}-rs${r.id}`, ['--package', sceneDir(s), '--kernel', 'restir', '--preset', r.preset, '--spp', String(f), '--batches', String(B), '--seed', String(SEEDS.restirPilot)],
      B, add, `pilot ReSTIR ${s.pkg} rung ${r.id} (${f} frames x ${B})`);
    if (!run) return undefined;
    pilot[r.id] = run.dir;
    rungs.push({ id: r.id, side: batchImages(run.dir, run.meta), tile: 32 });
  }
  const t0 = performance.now();
  const z = sizeScene(batchImages(pt.dir, pt.meta), rungs, { B: pilotB(s) });
  add(`sizing ${s.pkg} (B ${z.B}, x${SIZING_MARGIN} margin)`, true, (performance.now() - t0) / 1000, z,
    `PT ${z.ptSpp} spp x ${z.B} (${(z.ptSeconds / 60).toFixed(1)} min); ` + M4_RUNGS.map((r) => `${r.id}: ${z.rungs[r.id].framesPerBatch} fr x ${z.B}${z.rungs[r.id].tile === 64 ? ' @64²' : ''} (${(z.rungs[r.id].seconds / 60).toFixed(1)} min)`).join(', ')
    + (z.notes.length ? `; ${z.notes.join('; ')}` : ''));
  return { ...z, pilot };
}

function ptRef(s: M4Scene, z: SceneSizing, seed: number, add: Add): Run | undefined {
  return cachedRun(PTREFS, ptRunKey(s, z.ptSpp, z.B, seed), `${s.pkg}-s${seed}-${z.ptSpp}x${z.B}`,
    ['--package', sceneDir(s), '--kernel', 'pt', '--spp', String(z.ptSpp), '--batches', String(z.B), '--seed', String(seed)],
    z.B, add, `PT reference ${s.pkg} (${z.ptSpp} spp x ${z.B}, seed ${seed})`);
}

function writeTest(dir: string, name: string, nUnits: number, s: M4Scene, tile: 32 | 64, extra: Record<string, unknown> = {}): string {
  const p = path.join(dir, 'tests', `test-${name.replace(/[^\w.@-]+/g, '_')}.json`);
  mkdirSync(path.join(ROOT, dir, 'tests'), { recursive: true });
  writeFileSync(path.join(ROOT, p), JSON.stringify({
    name, stage: 'B', channels: ['Y', 'R', 'G', 'B'], n_units: nUnits, tier: s.tier, num_eps: NUM_EPS, num_eps_note: NUM_EPS_NOTE,
    min_replicates: s.tier === 'heavy-tail' ? 32 : 16,
    ...(tile !== 32 ? { tile, aggregate_note: `pilot sizing: a side needs > ${SIDE_CAP_S / 60} min at 32² tiles (restir-api.md §6.3, coordinator Q3)` } : {}),
    ...extra,
  }, null, 1));
  return p;
}

function compare(ours: string, ref: string, test: string, out: string, rerunOf?: string) {
  return sh(PY, ['validation/tools/compare.py', '--ours', ours, '--ref', ref, '--test', test, '--out', out, ...(rerunOf ? ['--rerun-of', rerunOf] : [])], () => false);
}

function summarize(rep: Record<string, any>) {
  const Y = rep.channels.Y;
  return {
    status: rep.status, failed_checks: rep.failed_checks, tier: rep.tier,
    global_rel_Y: r4(Y.global_.rel), mdb_global_Y: r4(Y.global_.mdb), worst_tile_rel_Y: r4(Y.tiles.worst_rel), worst_tile: Y.tiles.worst_tile ? `[${Y.tiles.worst_tile.join(',')}]` : undefined,
    mdb_tile_max_Y: r4(Y.tiles.mdb_max), mdb_tile_median_Y: r4(Y.tiles.mdb_median), tost_failed_tiles: Y.tiles.tost_failed, zero_var_tiles: Y.tiles.zero_var,
    multiplier_needed: r4(Math.max(...['Y', 'R', 'G', 'B'].map((c) => Math.max(rep.channels[c].global_.replicate_multiplier_needed ?? 1, rep.channels[c].tiles.replicate_multiplier_needed ?? 1)))),
    per_channel: Object.fromEntries(['R', 'G', 'B'].map((c) => [c, { global_rel: r4(rep.channels[c].global_.rel), worst_tile_rel: r4(rep.channels[c].tiles.worst_rel), tost_failed: rep.channels[c].tiles.tost_failed }])),
  };
}

function restirRun(s: M4Scene, preset: Preset, frames: number, B: number, seed: number, run: string, dest: string, extra: string[] = []) {
  console.log(`\n--- ReSTIR ${s.pkg} ${preset}: ${frames} frames x ${B}, seed ${seed}${extra.length ? ` ${extra.join(' ')}` : ''}`);
  return runBatches(['--package', sceneDir(s), '--kernel', 'restir', '--preset', preset, '--spp', String(frames), '--batches', String(B), '--seed', String(seed), ...extra], run, dest);
}

/** Frames per batch / batches of an ensemble run holding the rung's total member-frames (rows r = t·E + m). */
export function ensembleShape(totalFrames: number, members: number): { framesPerBatch: number; batches: number } {
  const T = Math.max(1, Math.ceil(totalFrames / members));
  const framesPerBatch = Math.min(T, 64);
  return { framesPerBatch, batches: Math.ceil(T / framesPerBatch) };
}

function stageB(s: M4Scene, rung: RungId, preset: Preset, z: SceneSizing, ref: Run, dir: string, runId: string, nU: number, add: Add,
  o: { rounds: number; members?: number; unit?: string }, firstRuns: Record<string, Run>): Record<string, any> {
  const unit = o.unit ?? `${s.pkg}@${rung}`;
  const kind = o.members ? 'ensemble' : preset === 'criteria2022' ? 'crit2022' : 'scene';
  const rz = z.rungs[rung];
  const shape = o.members ? ensembleShape(rz.samples, o.members) : { framesPerBatch: rz.framesPerBatch, batches: z.B };
  const extra = o.members ? ['--members', String(o.members)] : [];
  const t0 = performance.now();
  const first = restirRun(s, preset, shape.framesPerBatch, shape.batches, SEEDS.restir, `${runId}-${unit}`.replace(/[^\w.-]+/g, '_'), path.join(dir, 'restir', unit.replace(/[^\w.@-]+/g, '_')), extra);
  const test = writeTest(dir, unit, nU, s, rz.tile);
  const out = path.join(dir, 'compare', unit.replace(/[^\w.@-]+/g, '_'));
  const t16 = first.meta ? t16Problems(first.meta, ref.meta, { rounds: o.rounds, members: o.members }) : ['ReSTIR run produced no meta.json'];
  let rep: Record<string, any> | undefined;
  let c = { code: 1, out: first.out };
  if (first.dir && first.meta) {
    firstRuns[unit] = { dir: first.dir, meta: first.meta, seconds: first.seconds };
    c = compare(first.dir, ref.dir, test, out);
    rep = tryJson(path.join(out, 'report.json'));
  }
  let rerun: Record<string, unknown> | undefined;
  if (rep?.status === 'rerun_required') {
    console.log(`  ${unit}: rerun_required (${rep.failed_checks.join(', ')}) -> confirmatory re-run on disjoint seeds (PT ${SEEDS.ptRerun}, ReSTIR ${SEEDS.restirRerun})`);
    const ref2 = ptRef(s, z, SEEDS.ptRerun, add);
    const second = restirRun(s, preset, shape.framesPerBatch, shape.batches, SEEDS.restirRerun, `${runId}-${unit}-rerun`.replace(/[^\w.-]+/g, '_'), path.join(dir, 'restir', `${unit.replace(/[^\w.@-]+/g, '_')}-rerun`), extra);
    if (ref2 && second.dir && second.meta) {
      t16.push(...t16Problems(second.meta, ref2.meta, { rounds: o.rounds, members: o.members }).map((x) => `re-run: ${x}`));
      c = compare(second.dir, ref2.dir, test, `${out}-rerun`, path.join(out, 'report.json'));
      rerun = { first: summarize(rep), report: path.join(`${out}-rerun`, 'report.json') };
      rep = tryJson(path.join(`${out}-rerun`, 'report.json'));
    }
  }
  const sum = rep ? summarize(rep) : undefined;
  const statOk = !!rep && (rep.status === 'pass' || rep.status === 'pass_on_rerun');
  const ok = statOk && t16.length === 0;
  const rm = first.meta?.restir ?? {};
  const res = {
    unit, kind, scene: s.pkg, rung, preset, label: s.label, tier: s.tier, ok, ...(sum ?? { status: `error (exit ${c.code})` }),
    pt: `${z.ptSpp}x${z.B}`, restir: o.members ? `${shape.framesPerBatch}x${shape.batches}x E${o.members}` : `${shape.framesPerBatch}x${z.B}`, tile: rz.tile,
    aggregate_enlarged: rz.tile !== 32, fr: r4(rm.fr), restir_minutes: first.meta ? r4(first.meta.timings.totalMs / 60000) : undefined,
    pt_cache_hit: ref.cacheHit, t16, counters: first.meta?.counters, rsc_errors: rm.counters && Object.fromEntries(Object.entries(rm.counters as Record<string, number>).filter(([k, v]) => v && /NonFinite|pendingLeft|slotMismatch|bvh/.test(k))),
    submits: first.meta?.submits && { total: first.meta.submits.total, maxMs: r4(first.meta.submits.maxMs), overBudget: first.meta.submits.overBudget },
    report: path.join(out, 'report.json'), restir_dir: first.dir, pt_dir: ref.dir, ...(rerun ? { rerun } : {}), ...(rep?.notes && Object.keys(rep.notes).length ? { notes: rep.notes } : {}),
  };
  add(`Stage B ${unit} (${s.label}; rung ${rung} ${preset})`, ok, (performance.now() - t0) / 1000, res,
    sum ? `${sum.status}: Δ_Y ${pct(sum.global_rel_Y as number, 4)} (MDB ${pct(sum.mdb_global_Y as number, 3)}), worst tile ${pct(sum.worst_tile_rel_Y as number, 2)} ${sum.worst_tile ?? ''}, tile MDB max ${pct(sum.mdb_tile_max_Y as number, 2)}, mult ${sum.multiplier_needed}${sum.failed_checks?.length ? `, failed ${sum.failed_checks.join(' ')}` : ''}${t16.length ? `; T16: ${t16.join('; ')}` : ''}`
      : `compare exit ${c.code} ${c.out.slice(-300)}${t16.length ? `; T16: ${t16.join('; ')}` : ''}`);
  return res;
}

function plantDetection(s: M4Scene, p: (typeof PLANTS)[number], z: SceneSizing, ref: Run, dir: string, runId: string, nU: number, add: Add): Record<string, any> {
  const unit = `plant-${p.tag}-${s.pkg}`;
  const rz = z.rungs['3.2'];
  const t0 = performance.now();
  const run = restirRun(s, 'offline', rz.framesPerBatch, z.B, p.seed, `${runId}-${unit}`, path.join(dir, 'restir', unit), ['--plant', p.plant]);
  let ok = false, data: Record<string, any> = { unit, kind: 'plant', scene: s.pkg, rung: '3.2', plant: p.plant };
  let detail = `run exit ${run.code}`;
  if (run.dir && run.meta) {
    const t16 = t16Problems(run.meta, ref.meta, { plant: true, rounds: 3 });
    const test = writeTest(dir, `${unit}-calibrate`, nU, s, rz.tile, { calibration: { plants: [] } });
    const out = path.join(dir, 'calibrate', unit);
    const r = sh(PY, ['validation/tools/compare.py', '--calibrate', '--ref', ref.dir, '--planted', run.dir, '--plant-name', `ReSTIR ${p.name}`, '--test', test, '--out', out,
      '--splits', '20', '--repeats', '10'], () => false);
    const rep = tryJson(path.join(out, 'report.json'));
    const rp = rep?.calibration?.rendered_plant;
    const fullOut = `${out}-full`;
    compare(run.dir, ref.dir, writeTest(dir, `${unit}-full`, nU, s, rz.tile), fullOut);
    const fullRep = tryJson(path.join(fullOut, 'report.json'));
    ok = !!rp && rp.gate_fail_count >= 9 && rp.control_pass_count >= 9 && !!fullRep && fullRep.status !== 'pass' && t16.length === 0;
    data = {
      ...data, ok, status: ok ? 'detected' : 'not detected', report: path.join(out, 'report.json'), calibrate_exit: r.code, t16,
      detected: rp && `${rp.gate_fail_count}/${rp.n_repeats}`, equivalence_failed: rp && `${rp.equivalence_fail_count}/${rp.n_repeats}`,
      control_pass: rp && `${rp.control_pass_count}/${rp.n_repeats}`, planted_rel_Y: rp && r4(rp.channels.Y.planted_global_rel_median),
      mdb_global_Y: rp && r4(rp.channels.Y.mdb_global_median), failed_checks_Y: rp && Object.keys(rp.failed_checks_histogram).filter((k) => k.endsWith('[Y]')),
      full_compare: fullRep && { status: fullRep.status, global_rel_Y: r4(fullRep.channels.Y.global_.rel), worst_tile_rel_Y: r4(fullRep.channels.Y.tiles.worst_rel), failed_checks: fullRep.failed_checks },
      fr: r4(run.meta.restir?.fr),
    };
    detail = rp ? `${s.pkg}: detected ${rp.gate_fail_count}/10 (TOST ${rp.equivalence_fail_count}/10), PT A/A control ${rp.control_pass_count}/10, planted Δ_Y ${pct(rp.channels.Y.planted_global_rel_median, 3)}; full: ${fullRep?.status} (Δ_Y ${pct(fullRep?.channels?.Y?.global_?.rel, 3)})${t16.length ? `; T16: ${t16.join('; ')}` : ''}`
      : `calibrate exit ${r.code} ${r.out.slice(-300)}`;
  }
  add(`plant detected: ${p.name} on ${s.pkg} (rung 3.2)`, ok, (performance.now() - t0) / 1000, data, detail);
  return data;
}

/** compare.py --calibrate on the (i) rung-3.2 ReSTIR run: A/A re-splits at the nominal rate + synthetic W × 1.003 ≥ 9/10. */
function syntheticPlant(s: M4Scene, base: Run, z: SceneSizing, dir: string, nU: number, add: Add): Record<string, any> {
  const unit = `plant-W1.003-${s.pkg}`;
  const t0 = performance.now();
  const test = writeTest(dir, unit, nU, s, z.rungs['3.2'].tile);   // stage B default plants: W x1.003
  const out = path.join(dir, 'calibrate', unit);
  const r = sh(PY, ['validation/tools/compare.py', '--calibrate', '--ref', base.dir, '--test', test, '--out', out, '--splits', '20', '--repeats', '10'], () => false);
  const rep = tryJson(path.join(out, 'report.json'));
  const aa = rep?.calibration?.aa, pl = rep?.calibration?.plants?.[0];
  const ok = r.code === 0 && !!aa?.ok && !!pl?.calibrated;
  const data = {
    unit, kind: 'plant', scene: s.pkg, rung: '3.2', plant: 'W x1.003 (synthetic)', ok, status: ok ? 'detected' : 'not detected', report: path.join(out, 'report.json'),
    aa: aa && { ok: aa.ok, splits: aa.n_splits, fpr_tile: aa.per_tile, gate_pass_rate: aa.gate_pass_rate, tost_pass_rate: aa.tost_pass_rate },
    detected: pl && `${pl.gate_fail_count}/${pl.n_repeats}`, control_pass: pl && `${pl.control_pass_count}/${pl.n_repeats}`, mdb_global_median: pl && r4(pl.mdb_global_median),
  };
  add(`synthetic W x1.003 plant + calibrate A/A re-splits on the ReSTIR ${s.pkg} rung-3.2 run (x${CALIB_FACTOR}, seed ${SEEDS.aa})`, ok, (performance.now() - t0) / 1000, data,
    rep ? `A/A ${aa?.ok ? 'ok' : 'FAIL'} (gate pass ${Math.round((aa?.gate_pass_rate ?? 0) * 20)}/20); W x1.003 detected ${pl?.gate_fail_count}/10, control ${pl?.control_pass_count}/10, MDB ${pct(pl?.mdb_global_median, 3)}` : `exit ${r.code} ${r.out.slice(-300)}`);
  return data;
}

function restirAA(s: M4Scene, z: SceneSizing, ref: Run, dir: string, runId: string, nU: number, add: Add): { result: Record<string, any>; base?: Run } {
  const unit = `aa-restir-${s.pkg}`;
  const t0 = performance.now();
  const rz = z.rungs['3.2'];
  const frames = rz.framesPerBatch * CALIB_FACTOR;
  const runs = [SEEDS.aa, SEEDS.aa2].map((seed) => restirRun(s, 'offline', frames, z.B, seed, `${runId}-${unit}-${seed}`, path.join(dir, 'restir', `${unit}-${seed}`)));
  let data: Record<string, any> = { unit, kind: 'aa', scene: s.pkg, rung: '3.2', ok: false, status: `run exit ${runs.map((r) => r.code).join('/')}`, restir: `${frames}x${z.B} (x${CALIB_FACTOR})` };
  let base: Run | undefined;
  if (runs.every((r) => r.dir && r.meta)) {
    const t16 = runs.flatMap((r) => t16Problems(r.meta!, ref.meta, { rounds: 3 }));
    base = { dir: runs[0].dir!, meta: runs[0].meta!, seconds: runs[0].seconds };
    const out = path.join(dir, 'compare', unit);
    const c = compare(runs[1].dir!, runs[0].dir!, writeTest(dir, unit, nU, s, rz.tile), out);
    const rep = tryJson(path.join(out, 'report.json'));
    const sum = rep && summarize(rep);
    data = { ...data, ...(sum ?? { status: `compare exit ${c.code}` }), ok: rep?.status === 'pass' && t16.length === 0, t16, report: path.join(out, 'report.json'), seeds: `${SEEDS.aa2} vs ${SEEDS.aa}` };
  }
  add(`A/A: two ReSTIR seed sets on ${s.pkg} rung 3.2 (${SEEDS.aa} vs ${SEEDS.aa2}, ${frames} frames x ${z.B})`, data.ok, (performance.now() - t0) / 1000, data,
    `${data.status}: Δ_Y ${pct(data.global_rel_Y, 4)} (MDB ${pct(data.mdb_global_Y, 4)}), tile MDB max ${pct(data.mdb_tile_max_Y, 2)}`);
  return { result: data, base };
}

// ------------------------------------------------------------------------------------------------ budget + reports

function budgetRows(scenes: M4Scene[], sizing: Record<string, SceneSizing>): Record<string, unknown>[] {
  const rows: Record<string, unknown>[] = [];
  for (const s of scenes) {
    const z = sizing[s.pkg];
    if (!z) continue;
    const meta = tryJson(path.join(sceneDir(s), 'scene.json')) as { render?: { width: number; height: number; maxBounces: number } } | undefined;
    for (const r of M4_RUNGS) {
      const rz = z.rungs[r.id];
      rows.push({
        scene: s.pkg, rung: r.id, preset: r.preset, width: meta?.render?.width, height: meta?.render?.height, max_bounces: meta?.render?.maxBounces, tier: s.tier,
        B: z.B, pt_spp: z.ptSpp, pt_ms_per_spp: r4(z.ptSeconds * 1000 / z.ptSamples), pt_min: r4(z.ptSeconds / 60),
        frames_per_batch: rz.framesPerBatch, restir_ms_per_frame: r4(rz.msPerFrame), restir_min: r4(rz.seconds / 60), tile: rz.tile, aggregate_enlarged: rz.enlarged,
      });
    }
  }
  return rows;
}

function mergeBudget(rows: Record<string, unknown>[], runId: string, add: Add): void {
  const p = path.join(ROOT, 'validation/budget.json');
  const b = JSON.parse(readFileSync(p, 'utf8')) as Record<string, any>;
  const keep = (b.m4_entries ?? []).filter((e: any) => !rows.some((r) => r.scene === e.scene && r.rung === e.rung));
  b.m4_method = 'M4 Stage-B sizing (gate-m4.ts): per-scene PT pilot 128 spp x B and per-rung ReSTIR pilot (3.1/3.1b 128 frames, 3.2 8 frames) x B in headless Chrome; ms = batch wall time incl. readback / samples; sizes = PLAN §7.3 rule x1.25 margin, niceCeil per batch';
  b.m4_measured_at = new Date().toISOString();
  b.m4_run = runId;
  b.m4_entries = [...keep, ...rows];
  writeFileSync(p, `${JSON.stringify(b, null, 2)}\n`);
  add('budget.json M4 rows written (--write-budget)', true, 0, { rows: b.m4_entries.length });
}

function table(results: Record<string, any>[]): string {
  const head = '| unit | rung | tier | status | Δ_Y | MDB_Y | worst tile (rel) | tile MDB max | TOST-failed tiles | PT spp×B | ReSTIR frames×B | tile | mult. needed | f_r | min |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n';
  return head + results.map((r) => `| ${r.unit} | ${r.rung ?? ''} | ${r.tier ?? ''} | ${r.status}${r.t16?.length ? ' (T16)' : ''} | ${pct(r.global_rel_Y, 4)} | ${pct(r.mdb_global_Y, 4)} | ${pct(r.worst_tile_rel_Y, 2)} ${r.worst_tile ?? ''} | ${pct(r.mdb_tile_max_Y, 2)} | ${r.tost_failed_tiles ?? ''} | ${r.pt ?? ''} | ${r.restir ?? ''} | ${r.tile ?? ''} | ${r.multiplier_needed ?? ''} | ${r.fr ?? ''} | ${r.restir_minutes ?? ''} |`).join('\n') + '\n';
}

function summaryMd(summary: Record<string, any>, results: Record<string, any>[], sizing: Record<string, SceneSizing>): string {
  const lines = [
    `# M4 gate ${summary.run}`, '', `Result: **${summary.ok ? 'PASS' : 'FAIL'}** (${summary.total_s} s, n_units ${summary.n_units})`, '',
    '## Per rung', '', '| rung | pass | fail | not run |', '|---|---|---|---|',
    ...Object.entries(summary.per_rung as Record<string, any>).map(([k, v]) => `| ${k} | ${v.pass} | ${v.fail} | ${v.notRun} |`), '',
    '## Units', '', table(results),
    '## Sizing', '', '| scene | B | PT spp | PT min | 3.1 frames | 3.1b frames | 3.2 frames | tiles | notes |', '|---|---|---|---|---|---|---|---|---|',
    ...Object.entries(sizing).map(([k, z]) => `| ${k} | ${z.B} | ${z.ptSpp} | ${(z.ptSeconds / 60).toFixed(1)} | ${z.rungs['3.1']?.framesPerBatch} | ${z.rungs['3.1b']?.framesPerBatch} | ${z.rungs['3.2']?.framesPerBatch} | ${M4_RUNGS.map((r) => z.rungs[r.id]?.tile).join('/')} | ${z.notes.join('; ')} |`),
    '', '## Failed steps', '', ...(summary.failed as string[]).map((f) => `- ${f}`), '',
  ];
  return lines.join('\n');
}
