// M5 milestone gate (restir-temporal-api.md §6; PLAN §5 M5 exit, §7.1 rungs 3.3–3.6, §7.3 dynamic statistics, §7.4 M5):
// `npm run validate -- --milestone M5`. OWNER T-E.
//   Gate 0  typecheck, cpu lane (tests/restir incl. gate-m5-config), python tests (incl. dynamic / dyn_masks / plant_sign),
//           make-m5.ts determinism (twice, byte-identical), the Chrome GPU suites restir-tframe, restir-temporal,
//           restir-refresh, restir-debug, the M4 suites restir-initial/shift/spatial and the M3 regressions (one GPU-lock
//           hold per suite), the M5 app smoke (T-D), budget.json M5 rows.
//   Gate 3  static rungs at 256² on the 8-scene subset (TD28, Q2): 3.3 `temporal` and 3.4 `full`, per-frame ensembles at
//           t ∈ {1, 24} of 25-frame chains (stage B: 0.2 % global, 1 % per 32² tile), 3.5 = per-chain mean over frames
//           32…287 on three scenes (Q9, stage B); the ladder stops per scene at the first failing rung (after its re-run).
//           Dynamic rung 3.6: 18 units (13 sequences + Talbot / E2 on ixs_b and ixs_e (Q11) + boost on ixs_d) with
//           per-frame ensembles at every test frame (stage dyn: 0.2 % global, 2 % per 64² tile, 3 % per mask region) and
//           the sequence statistics of dynamic.py (drift, failing-tile binomial).
//   Plants  §6.5 (TD29): detected (half-size repeats ≥ 9/10 vs 4× PT, PT A/A ≥ 9/10, full comparison not `pass`) AND
//           the predicted sign (plant_sign.py); the synthetic W × 1.003 and the calibrate A/A re-splits on the A/A run;
//           the A/A test (seeds 7502 / 7503, 4× size, must pass).
//   U8      the U8 ladder (TD30, Q6): u8_* scenes through rungs 3.1, 3.1b, 3.2 (sequential, M4 harness) and 3.3, 3.4
//           (chains), all must pass; plus the U8 plants.
//   Sizing  pilots (PT 128 spp × 16 per frame; 64 chains per unit), the E7 method per test frame and aggregate (incl.
//           mask regions), × 1.25, R floor 256 (3.5: 64), multiples of E = 16; PT sizes frozen per PT code; per-side
//           unit cap 30 min: over the cap the TILE aggregate is enlarged one step (32→64, 64→128), the global never; a
//           unit whose global alone needs more than the cap gets 90 min once (Q4), beyond that it is `infeasible` and
//           escalates to the coordinator (never passed, δ never loosened). Plan > 14 h ⇒ two required parts (Q3):
//           --part core (Gate 0, 3.6, plants, A/A, app smoke) and --part static (3.3–3.5, U8 ladder).
//   T15/T16 every chain run: NaN/Inf/negatives 0, every RSC error counter incl. RSC_T_NONFINITE / RSC_T_PENDING_LEFT 0,
//           q0–q2 overflow 0, no submit over the hard cap; history valid on every frame except t = 0 and the package's
//           declared reset frames ("any config change resets history": ixs_k's map swap at 20), temporal units every
//           frame; preset as the rung, RR off, cCap 20, jitter iid, Mode A, maxBounces / scene bytes / env NEE / frame =
//           the PT reference's, spatialRoundsExecuted = rounds.
// Output: validation/out/m5-gate-<time>/ (summary.json/.md, sizing.json, budget-m5.json, tests/, compare/, chains/,
// plants/). Options: --only a,b (packages), --part core|static, --pilot-only, --prerender-ptrefs, --write-budget.
// Before `git worktree remove`, copy validation/out/m5* to the main checkout (M4 lesson).
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodePFM, encodePFM } from '../../src/core/io/pfm.ts';
import { readNpz } from '../../src/core/render/restir/npz.ts';
import { withGpuLockSync } from './gpu-lock.ts';
import {
  LOCK_CHUNK_S, NUM_EPS, chunkBatches, gpuSuiteHolds, codeHashes as m4CodeHashes, mergeChunkMetas, niceCeil, sizeScene, sizingTarget, t16Problems, tsClosure,
  type PilotSide,
} from './gate-m4.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PY = path.join(ROOT, 'validation/.venv/bin/python');
export const M5_OUT = 'validation/out/m5';
const ENV_SCENES = `${M5_OUT}/scenes`;
const PTREFS = `${M5_OUT}/ptrefs`;
const PILOTS = `${M5_OUT}/pilots`;
const MASKS = `${M5_OUT}/masks`;
const TIMEOUT_MS = 24 * 3600_000;

// ------------------------------------------------------------------------------------------------ configuration

/** Seeds of the M5 gate (restir-temporal-api.md §5), disjoint from M4's. */
export const SEEDS = {
  pt: 7001, ptRerun: 107001, chains: 7002, chainsRerun: 107002, aa: 7502, aa2: 7503, ptPilot: 7011, chainPilot: 7012,
  plantBase: 7101, ptCalib: 7201, ptMask: 7301,
} as const;
export const E_MEMBERS = 16;
/** Floor of the chains per unit (gap-temporal §9.1); rung 3.5's replicate is a 256-frame chain mean (Changelog E-4). */
export const R_FLOOR = 256;
export const R_FLOOR_AVG = 64;
export const PILOT_CHAINS = 64;
export const PILOT_PT_SPP = 128;
export const B_PT = 16;
export const MIN_PT_SPP = 256;
export const SIZING_MARGIN = 1.25;
/** Mask references (§6.4 masks: frozen PT means at t and t−1, disjoint seed 7301). */
export const MASK_PT = { spp: 512, B: 4 } as const;
/** Per-side unit cap (§6.6 control 5) and its one-time raise for a global-infeasible unit (Q4). */
export const UNIT_CAP_S = 30 * 60;
export const UNIT_CAP_RAISED_S = 90 * 60;
/** Plan above this ⇒ the two-part split (Q3). */
export const SPLIT_HOURS = 14;
export const CALIB_FACTOR = 4;
/** Chains of a plant on a package without a gating unit (ixs_d_glossy, ixs_e_half, ixs_n4; Changelog E-12). */
export const PLANT_ONLY_R = 4096;
/** Gate-0 wall estimate (M5 + M4 + M3 suites incl. the 18-min T3 runs and the three T3-2 rare-bin holds, app smoke). */
export const GATE0_EST_H = 1.5;
/** Wall seconds per run-batches invocation outside the GPU batches (Vite + Chrome start, compiles, probe), for the plan. */
export const INVOCATION_OVERHEAD_S = 30;
/** Samples per PT dispatch (one submit each) in the gate's PT runs (E-21). */
export const PT_MAX_SPP_PER_DISPATCH = 32;
export const STATIC = { T: 25, testFrames: [1, 24] } as const;
export const AVG = { T: 288, from: 32, to: 287 } as const;
export const DELTA = { B: { global: 0.002, tile: 0.01, mask: 0.01, tile0: 32 }, dyn: { global: 0.002, tile: 0.02, mask: 0.03, tile0: 64 } } as const;

export interface M5Scene { pkg: string; label: string; env: boolean; tier: 'tight' | 'heavy-tail' }
const S = (pkg: string, label: string, env = false): M5Scene => ({ pkg, label, env, tier: 'tight' });
/** §6.2 static subset (TD28). */
export const M5_STATIC: M5Scene[] = [
  S('m5s_cornell_i', '(i) diffuse Cornell, rect light'),
  S('m5s_spot_grazing', '(iii) spot grazing a wall'),
  S('m5s_glossy_v1', '(v) V1 GGX r 0.2-0.8, rect lights'),
  S('m5s_glass_mirror_A', '(vi) glass + mirror + glossy, Mode A'),
  S('m5s_many_lights', '(x) many lights'),
  S('m5s_alpha_foliage', '(xii) alpha MASK foliage cards'),
  S('m5s_c0q_openbox_b13', 'C0q(d) rho=1 open box b=13, constant env', true),
  S('m5s_overcast_rect_b3', '(xiv) open Cornell, overcast + rect, b=3', true),
];
export const M5_AVG_SCENES = ['m5s_cornell_i', 'm5s_glossy_v1', 'm5s_overcast_rect_b3'];
/** Reported, not gating (Q2): heavy-tail at 256² is unaffordable at δ; they gate again in M8's validate --all. */
export const M5_REPORT_ONLY = ['xiii_spheres_512x256', 'xiv_kloof_b3_512', 'xiv_kloof_rect_b3_512'];
export const STATIC_RUNGS = [
  { id: '3.3', preset: 'temporal', rounds: 0 },
  { id: '3.4', preset: 'full', rounds: 1 },
  { id: '3.5', preset: 'full', rounds: 1 },
] as const;

export interface SeqPkg { pkg: string; env: boolean; T: number; testFrames: number[]; label: string; plantOnly?: boolean }
const Q = (pkg: string, T: number, testFrames: number[], label: string, o: Partial<SeqPkg> = {}): SeqPkg => ({ pkg, env: false, T, testFrames, label, ...o });
/** §6.2 sequence packages (make-m5.ts; T and test frames also in each package's `sequence` block). */
export const M5_SEQUENCES: SeqPkg[] = [
  Q('ixs_a_point_256', 81, [1, 10, 25, 40, 80], 'ix-a point light path behind a box, stops at 40'),
  Q('ixs_b_area_256', 41, [15, 16, 17, 24, 25, 40], 'ix-b rect translate + spin, area x1.5 at 16, power x2 at 24'),
  Q('ixs_c_spot_b0_256', 41, [5, 20, 40], 'ix-c spot rotating 2 deg/frame, blend 0'),
  Q('ixs_c_spot_b03_256', 41, [5, 20, 40], 'ix-c spot rotating 2 deg/frame, blend 0.3'),
  Q('ixs_d_camera_256', 65, [8, 16, 24, 32, 40, 48, 56, 64], 'ix-d camera fly path'),
  Q('ixs_d0_jitter_256', 65, [8, 32, 64], 'ix-d0 static camera, jitter only'),
  Q('ixs_d_glossy_256', 33, [8, 16, 32], 'ix-d fly path, GGX r 0.3 floor (N6)', { plantOnly: true }),
  Q('ixs_e_addremove_256', 41, [8, 9, 14, 15, 20, 21, 40], 'ix-e add C at 8, A x2 at 14, remove B at 20'),
  Q('ixs_e_half_256', 22, [14, 15, 21], 'ix-e with A x0.5 at 14 (N1 positive sign)', { plantOnly: true }),
  Q('ixs_f_combined_256', 49, [8, 24, 25, 40, 48], 'ix-f camera + lights, teleport at 24, FOV ramp 32-48'),
  Q('ixs_g_sun_256', 49, [8, 24, 48], 'ix-g courtyard, rotating sun'),
  Q('ixs_h_envrot_256', 41, [1, 10, 25, 40], 'ix-h env rotating 1 deg/frame', { env: true }),
  Q('ixs_i_envradio_256', 32, [15, 16, 17, 24, 25, 30, 31], 'ix-i env strength x2 at 16, tint at 24, rect x2 at 30', { env: true }),
  Q('ixs_j_envcombo_256', 41, [8, 24, 40], 'ix-j env rotation + fly camera + moving rect', { env: true }),
  Q('ixs_k_envswap_256', 29, [19, 20, 21, 28], 'ix-k env map swap at 20 (history reset asserted)', { env: true }),
  Q('ixs_n4_twolights_256', 17, [8, 16], 'N4 two lights at distance ratio 3, refresh every frame', { plantOnly: true }),
];

export interface DynUnit { id: string; pkg: string; variant: 'base' | 'talbot' | 'e2' | 'boost'; args: string[] }
/** Rung 3.6 (§6.3): 13 base sequences + Talbot / E2 on ixs_b and ixs_e (Q11) + boost on ixs_d (Q7): 18 units. */
export const DYN_UNITS: DynUnit[] = [
  ...M5_SEQUENCES.filter((s) => !s.plantOnly).map((s): DynUnit => ({ id: `${s.pkg}@3.6`, pkg: s.pkg, variant: 'base', args: [] })),
  ...['ixs_b_area_256', 'ixs_e_addremove_256'].flatMap((pkg): DynUnit[] => [
    { id: `${pkg}@3.6-talbot`, pkg, variant: 'talbot', args: ['--temporal-mis', 'talbot'] },
    { id: `${pkg}@3.6-e2`, pkg, variant: 'e2', args: ['--refresh', 'e2'] },
  ]),
  { id: 'ixs_d_camera_256@3.6-boost', pkg: 'ixs_d_camera_256', variant: 'boost', args: ['--boost', '3'] },
];

export interface Prediction { frame: number; region: string; sign: '+' | '-' | '?' | 'detect' }
export interface PlantSpec {
  id: string; name: string; pkg: string; rung: '3.4' | '3.6';
  args: string[];
  predict: Prediction[];
  only?: '+' | '-';
  /** Single-emitter PT renders at these frames for M_light:<name> (lights by scene.json lightNames, or 'env'). */
  dominance?: { frames: number[]; names: string[] };
}
const P = (id: string, name: string, pkg: string, rung: PlantSpec['rung'], args: string[], predict: Prediction[], o: Partial<PlantSpec> = {}): PlantSpec => ({ id, name, pkg, rung, args, predict, ...o });
const pr = (frames: number[], region: string, sign: Prediction['sign']): Prediction[] => frames.map((frame) => ({ frame, region, sign }));
/** §6.5 rendered plants (19; U8-4 deferred to M6, B-9; U8-2t a Gate-0 activity test, E-22). Regions: mask names of dyn_masks.py; `global` = the image mean. */
export const M5_PLANTS: PlantSpec[] = [
  P('N1-mixed', 'N1 mixed E_{t-1} (TP_N1_MIXED)', 'ixs_e_addremove_256', '3.6', ['--tplant', 'n1Mixed'],
    [...pr([14, 15], 'M_light:A', '-'), ...pr([8], 'M_light:C', '-')], { dominance: { frames: [8, 14, 15], names: ['A', 'C'] } }),
  P('N1-mixed-half', 'N1 mixed, A x0.5 (TP_N1_MIXED)', 'ixs_e_half_256', '3.6', ['--tplant', 'n1Mixed'], pr([14, 15], 'M_light:A', '+'), { dominance: { frames: [14, 15], names: ['A'] } }),
  P('N1-consistent', 'N1 consistent (TP_N1_MIXED + TM_PP_RECOMPUTE)', 'ixs_a_point_256', '3.6', ['--tplant', 'n1Mixed', '--temporal-check', 'recompute'], [...pr([10, 25], 'M_new', '-'), ...pr([10, 25], 'M_up', '-')], { only: '-' }),
  P('N2', 'N2 omit J_P (TP_NO_JP)', 'ixs_e_addremove_256', '3.6', ['--tplant', 'noJP'], [...pr([14, 15], 'M_light:A', '-'), ...pr([14, 15], 'M_light:B', '+')], { dominance: { frames: [14, 15], names: ['A', 'B'] } }),
  P('N3', 'N3 stale suffix radiance (TP_N3_STALE)', 'ixs_a_point_256', '3.6', ['--tplant', 'n3Stale'], [...pr([40], 'M_down', '+'), ...pr([40], 'M_gone', '+')]),   // C-11: direct light exact under N3; no refresh frame after 40
  P('N4', 'N4 fresh RIS re-draw (TP_N4_RIS, synthetic)', 'ixs_n4_twolights_256', '3.6', ['--tplant', 'n4Ris'], pr([8, 16], 'global', '-')),   // C-11
  P('N5-d0', 'N5 pixel-centre prev primary (TP_N5_PIXEL_CENTRE)', 'ixs_d0_jitter_256', '3.6', ['--tplant', 'n5PixelCentre'], pr([32], 'M_sil', '+')),   // B-12
  P('N5-d', 'N5 pixel-centre prev primary (TP_N5_PIXEL_CENTRE)', 'ixs_d_camera_256', '3.6', ['--tplant', 'n5PixelCentre'], pr([16], 'M_edge', '+')),   // B-12
  P('N6', 'N6 current camera in E_{t-1} (TP_N6_CUR_CAM)', 'ixs_d_glossy_256', '3.6', ['--tplant', 'n6CurCam'], pr([16, 32], 'global', '?')),
  P('N7', 'N7 per-light refresh mask (TP_N7_PER_LIGHT)', 'ixs_e_addremove_256', '3.6', ['--tplant', 'n7PerLight'], pr([14, 15], 'M_light:B', '-'), { dominance: { frames: [14, 15], names: ['B'] } }),
  P('env-no-rot-vis', 'skip the rotation refresh (TP_ENV_NO_ROT_VIS)', 'ixs_h_envrot_256', '3.6', ['--tplant', 'envNoRotVis'], [...pr([10, 25], 'M_down', '+'), ...pr([10, 25], 'M_up', '-')]),   // C-11
  // M_new / M_gone are empty under a 1°/frame rotation (E-13): their sign-resolved edge cells M_up / M_down carry the prediction
  P('env-gamma-t', 'E_{t-1} with gamma_t (TP_ENV_GAMMA_T)', 'ixs_h_envrot_256', '3.6', ['--tplant', 'envGammaT'], [...pr([10, 25], 'M_new', '-'), ...pr([10, 25], 'M_up', '-'), ...pr([10, 25], 'M_down', '-')], { only: '-' }),   // B-12: darkening only
  P('no-jp-env', 'omit J_P on env (TP_NO_JP_ENV)', 'ixs_i_envradio_256', '3.6', ['--tplant', 'noJPEnv'], pr([16, 17], 'M_light:env', '-'), { dominance: { frames: [16, 17], names: ['env', 'R'] } }),   // C-11
  P('cp-plus1', 'c_p + 1 in the MIS denominator (TP_CP_PLUS1)', 'm5s_cornell_i', '3.4', ['--tplant', 'cpPlus1'], pr([24], 'global', '-')),
  P('U8-1', 'U8-1 w1 < 1 for delta (RSF_PLANT_U8_W1DELTA)', 'u8_c0c_point_b1', '3.4', ['--u8-plant', 'u8W1Delta'], pr([24], 'global', '+')),   // B-12
  P('U8-3', 'U8-3 no p_k ratio (RSF_PLANT_U8_NO_PK)', 'm5s_cornell_i', '3.4',   // B-12: needs case (c) / deep paths
    ['--u8-plant', 'u8NoPk'], pr([24], 'global', 'detect')),
  P('U8-6', 'U8-6 one-sided ignored (RSF_PLANT_U8_ONESIDED)', 'u8_c0e_rect_b1', '3.4', ['--u8-plant', 'u8OneSided'], pr([24], 'global', 'detect')),   // B-12: sign reported
  P('U8-9', 'U8-9 FAILED dropped from k (RSF_PLANT_U8_FAILED_K)', 'u8_c0e_rect_b1', '3.4', ['--u8-plant', 'u8FailedK'], pr([24], 'global', '+')),   // Δ > 0 (Changelog D-5)
  P('U8-5t', 'U8-5t spot profile of t-1 (TP_U8_SPOT_PREV_AXIS)', 'ixs_c_spot_b03_256', '3.6', ['--tplant', 'u8SpotPrevAxis'], pr([20], 'global', 'detect')),
];
/** U8 plants deferred to M6 (Q6; B-9: U8-4 J = t_x²/t_y² needs x_{d−1} of case (a) in the shift source). Listed in the
 *  summary as "deferred to M6", never silently absent. */
export const M5_DEFERRED_PLANTS = [
  { id: 'U8-4', name: 'U8-4 J = t_x^2/t_y^2 (RSF_PLANT_U8_T2)', why: 'coordinator B-9: the shift source carries no x_{d-1} for case (a)' },
  { id: 'U8-7', name: 'U8-7 (Mode B)', why: 'Q6: Mode B temporal is M6' },
  { id: 'U8-8', name: 'U8-8 (RIS-NEE tiles)', why: 'Q6: light tiles are M6' },
  { id: 'U8-10', name: 'U8-10 (RIS-NEE tiles)', why: 'Q6: light tiles are M6' },
];
/** Plants whose bias is structurally below δ: not Gate-3 controls; their activity is a required Gate-0 test (E-22). */
export const M5_GATE0_PLANTS = [
  { id: 'U8-2t', name: 'U8-2t stale aux across frames (TP_U8_STALE_AUX)', test: 'U8 plant activity',
    status: 'active, bias below δ (measured +0.006 % on emissive + env, below gate resolution) — verified by the Gate-0 activity test' },
];
export const AA_PKG = 'm5s_cornell_i';
export const U8_SCENES = ['u8_c0c_point_b0', 'u8_c0c_point_b1', 'u8_c0d_spot_b0', 'u8_c0d_spot_b1', 'u8_c0e_rect_b0', 'u8_c0e_rect_b1'];
export const U8_M4_RUNGS = [{ id: '3.1', preset: 'initial', rounds: 0 }, { id: '3.1b', preset: 'initial-rr', rounds: 0 }, { id: '3.2', preset: 'offline', rounds: 3 }] as const;
export const GPU_SUITES: [string, string][] = [
  ['restir-tframe', 'U-M4-BITS, U-BIND-1, U-TL-1..3, U-TV-1, U-TH-1, U-TP-1 (T-A)'],
  ['restir-temporal', 'T6(a-c), T3-2, T4-t, §9.3-5/6, U-TQ-1, U-TR-2, U-TE-1 (T-B)'],
  ['restir-refresh', '§9.3-2/3/4, T-ENV-temporal (T-C)'],
  ['restir-debug', 'U-DBG-1..3, U-TD-1..3 (T-D)'],
  ['restir-initial', 'M4 regression + U-SFX-2'],
  ['restir-shift', 'M4 regression: T2, T3-0..T3-5, T3-D, T3-ENV, T4, T5'],
  ['restir-spatial', 'M4 regression + T3-3-boost'],
  ['pt', 'M3 regression'], ['bsdf', 'M3 regression'], ['lights', 'M3 regression'], ['env-sampling', 'M3 regression'], ['glass', 'M3 regression'], ['pt-glass', 'M3 regression'],
];
export const GPU_SUITE_ENV: Record<string, Record<string, string>> = { 'restir-shift': { VITE_T3_MS: String(18 * 60_000) } };
/** T3-2 at ≥ 10⁶ round trips per bin (restir-temporal-api.md B-11): three separate lock holds of ≈ 2–2.3 min GPU each (B-11, bddc333). */
export const T32_RARE_ENV = { VITE_T32_RARE_PAIRS: '16000', VITE_T32_RARE_RES: '256', VITE_T32_MIN_BIN: '1000000' };
export const T32_RARE_CASES = ['rare bins: translate', 'rare bins: add / remove + intensity', 'rare bins: moving lights + env'];

/** Suite FWER units × {Y, R, G, B}: every (scene/sequence, rung, test frame) unit + plants + synthetic + A/A. */
export function nUnits(): number {
  const staticUnits = M5_STATIC.length * 2 * STATIC.testFrames.length + M5_AVG_SCENES.length;
  const dyn = DYN_UNITS.reduce((a, u) => a + M5_SEQUENCES.find((s) => s.pkg === u.pkg)!.testFrames.length, 0);
  const u8 = U8_SCENES.length * (U8_M4_RUNGS.length + 2 * STATIC.testFrames.length);
  return 4 * (staticUnits + dyn + u8 + M5_PLANTS.length + 2);
}

export const isEnvPkg = (pkg: string): boolean => M5_STATIC.some((s) => s.pkg === pkg && s.env) || M5_SEQUENCES.some((s) => s.pkg === pkg && s.env);
export const pkgDir = (pkg: string): string => (isEnvPkg(pkg) ? `${ENV_SCENES}/${pkg}` : `validation/scenes/${pkg}`);

// ------------------------------------------------------------------------------------------------ sizing (pure)

/** Per-aggregate replicate statistics of one side: layout [channel Y,R,G,B][global, tiles…, masks…]. */
export interface AggSide { mean: Float64Array; sd: Float64Array; nTiles: number; nMasks: number; replicates: number; samplesPerReplicate: number }
const LUMA = [0.2126, 0.7152, 0.0722];

function finishAgg(rows: Float64Array[], nTiles: number, nMasks: number, samplesPerReplicate: number): AggSide {
  const N = rows.length, K = rows[0].length;
  const mean = new Float64Array(K), sd = new Float64Array(K);
  for (const r of rows) for (let k = 0; k < K; k++) mean[k] += r[k] / N;
  if (N > 1) for (const r of rows) for (let k = 0; k < K; k++) sd[k] += (r[k] - mean[k]) ** 2 / (N - 1);
  for (let k = 0; k < K; k++) sd[k] = Math.sqrt(sd[k]);
  return { mean, sd, nTiles, nMasks, replicates: N, samplesPerReplicate };
}

/** PT side from batch-mean images (RGB, row 0 = top) with optional region masks (H·W bool). */
export function aggFromImages(images: Float32Array[], W: number, H: number, tile: number, masks: Uint8Array[], samplesPerReplicate: number): AggSide {
  const th = Math.ceil(H / tile), tw = Math.ceil(W / tile), nT = th * tw, nM = masks.length, A = 1 + nT + nM;
  const cnt = new Float64Array(nT), mcnt = masks.map((m) => m.reduce((a, v) => a + (v ? 1 : 0), 0));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) cnt[Math.floor(y / tile) * tw + Math.floor(x / tile)]++;
  const rows = images.map((img) => {
    const r = new Float64Array(4 * A);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, t = 1 + Math.floor(y / tile) * tw + Math.floor(x / tile);
      const v = [0, img[3 * i], img[3 * i + 1], img[3 * i + 2]];
      v[0] = LUMA[0] * v[1] + LUMA[1] * v[2] + LUMA[2] * v[3];
      for (let c = 0; c < 4; c++) {
        r[c * A] += v[c]; r[c * A + t] += v[c];
        for (let q = 0; q < nM; q++) if (masks[q][i]) r[c * A + 1 + nT + q] += v[c];
      }
    }
    for (let c = 0; c < 4; c++) {
      r[c * A] /= W * H;
      for (let t = 0; t < nT; t++) r[c * A + 1 + t] /= cnt[t];
      for (let q = 0; q < nM; q++) r[c * A + 1 + nT + q] /= Math.max(mcnt[q], 1);
    }
    return r;
  });
  return finishAgg(rows, nT, nM, samplesPerReplicate);
}

/** Chain side from an ensemble.npz (per-chain tile / global / mask SUMS; a chain is one sample). */
export function aggFromNpz(z: Map<string, { shape: number[]; data: unknown }>, tile: number): AggSide {
  const tk = z.get(`tiles${tile}`);
  const g = z.get('global')!;
  const W = Number((z.get('width')!.data as BigInt64Array)[0]), H = Number((z.get('height')!.data as BigInt64Array)[0]);
  const R = g.shape[0];
  let th: number, tw: number, tiles: Float64Array;
  if (tk) { [, th, tw] = tk.shape; tiles = tk.data as Float64Array; } else {
    // 128² (enlarged dyn tiles): sum the 64² tiles
    const t64 = z.get('tiles64')!;
    const [, h64, w64] = t64.shape;
    th = Math.ceil(h64 / 2); tw = Math.ceil(w64 / 2);
    tiles = new Float64Array(R * th * tw * 3);
    const d = t64.data as Float64Array;
    for (let r = 0; r < R; r++) for (let y = 0; y < h64; y++) for (let x = 0; x < w64; x++) for (let c = 0; c < 3; c++) {
      tiles[((r * th + (y >> 1)) * tw + (x >> 1)) * 3 + c] += d[((r * h64 + y) * w64 + x) * 3 + c];
    }
  }
  const nT = th * tw;
  const mk = z.get('masks'), mp = z.get('mask_pixels');
  const nM = mk ? mk.shape[1] : 0;
  const A = 1 + nT + nM;
  const pix = (ty: number, tx: number) => (Math.min(H, (ty + 1) * tile) - ty * tile) * (Math.min(W, (tx + 1) * tile) - tx * tile);
  const gd = g.data as Float64Array;
  const rows: Float64Array[] = [];
  for (let r = 0; r < R; r++) {
    const row = new Float64Array(4 * A);
    const put = (k: number, rgb: [number, number, number], n: number) => {
      const v = [LUMA[0] * rgb[0] + LUMA[1] * rgb[1] + LUMA[2] * rgb[2], ...rgb];
      for (let c = 0; c < 4; c++) row[c * A + k] = v[c] / n;
    };
    put(0, [gd[3 * r], gd[3 * r + 1], gd[3 * r + 2]], W * H);
    for (let ty = 0; ty < th; ty++) for (let tx = 0; tx < tw; tx++) {
      const o = ((r * th + ty) * tw + tx) * 3;
      put(1 + ty * tw + tx, [tiles[o], tiles[o + 1], tiles[o + 2]], pix(ty, tx));
    }
    for (let q = 0; q < nM; q++) {
      const md = mk!.data as Float64Array, o = (r * nM + q) * 3;
      put(1 + nT + q, [md[o], md[o + 1], md[o + 2]], (mp!.data as Float64Array)[q]);
    }
    rows.push(row);
  }
  return finishAgg(rows, nT, nM, 1);
}

/** u(a) = per-sample variance / (D·T)² (gate-m4 uVector generalised to masks; D from the reference means). */
export function uVec(side: AggSide, ref: AggSide, d: { global: number; tile: number; mask: number }, B = B_PT): Float64Array {
  const A = 1 + ref.nTiles + ref.nMasks;
  if (side.mean.length !== 4 * A) throw new Error(`uVec: aggregate layouts differ (${side.mean.length} vs ${4 * A})`);
  const Tg = sizingTarget(d.global, 1, B - 1), Tt = sizingTarget(d.tile, ref.nTiles, B - 1), Tm = sizingTarget(d.mask, Math.max(ref.nMasks, 1), B - 1);
  const u = new Float64Array(4 * A);
  for (let c = 0; c < 4; c++) {
    const img = ref.mean[c * A];
    for (let a = 0; a < A; a++) {
      const k = c * A + a;
      const D = Math.max(ref.mean[k], 0.05 * img);
      if (!(D > 0)) continue;
      const T = a === 0 ? Tg : a <= ref.nTiles ? Tt : Tm;
      u[k] = (side.sd[k] ** 2 * side.samplesPerReplicate) / (D * D * T * T);
    }
  }
  return u;
}

export interface GroupSizing {
  pt: Record<number, { samples: number; spp: number; seconds: number; fixed: boolean }>;
  chains: Record<string, { R: number; seconds: number; msPerChain: number }>;
  cost: number;
}

/**
 * Joint allocation of one PT-sharing group (PLAN §7.3; gate-m4 sizeScene generalised): PT frames f with u_R,f (their
 * N_f total samples shared by every chain variant k), chain variants with u_{k,f} on the same aggregates and a common R_k
 * over all their frames. N_f = s·max_a u_R,f(a) (a common scale s, or the frozen size), R_k = max_{f,a} u_{k,f}(a)/(1 −
 * u_R,f(a)/N_f); minimise Σ c_R N_f + Σ c_k R_k over s; then × margin, spp = niceCeil(N/B) ≥ 256, R = ⌈R/E⌉·E ≥ floor.
 */
export function sizeGroup(pt: { frame: number; u: Float64Array; msPerSample: number; fixedSpp?: number }[],
  variants: { id: string; u: Map<number, Float64Array>; msPerChain: number; rFloor: number }[],
  o: { margin?: number; B?: number; E?: number; minPtSpp?: number } = {}): GroupSizing {
  const margin = o.margin ?? SIZING_MARGIN, B = o.B ?? B_PT, E = o.E ?? E_MEMBERS, minSpp = o.minPtSpp ?? MIN_PT_SPP;
  const maxU = pt.map((p) => p.u.reduce((m, v) => Math.max(m, v), 0));
  let best: { N: number[]; R: number[]; cost: number } | undefined;
  for (let i = 0; i <= 400; i++) {
    const s = 1.02 * 1000 ** (i / 400);
    const N = pt.map((p, j) => (p.fixedSpp ? (p.fixedSpp * B) / margin : Math.max(s * maxU[j], 1)));
    const R = variants.map((v) => {
      let m = 1;
      pt.forEach((p, j) => {
        const uk = v.u.get(p.frame);
        if (!uk) return;
        for (let a = 0; a < uk.length; a++) if (uk[a] > 0) m = Math.max(m, uk[a] / Math.max(1 - p.u[a] / N[j], 0.02));
      });
      return m;
    });
    const cost = pt.reduce((a, p, j) => a + p.msPerSample * N[j], 0) + variants.reduce((a, v, k) => a + v.msPerChain * R[k], 0);
    if (!best || cost < best.cost) best = { N, R, cost };
    if (pt.every((p) => p.fixedSpp)) break;
  }
  const out: GroupSizing = { pt: {}, chains: {}, cost: best!.cost };
  pt.forEach((p, j) => {
    const spp = p.fixedSpp ?? Math.max(minSpp, niceCeil((best!.N[j] * margin) / B));
    out.pt[p.frame] = { samples: spp * B, spp, seconds: (spp * B * p.msPerSample) / 1000, fixed: !!p.fixedSpp };
  });
  variants.forEach((v, k) => {
    const R = Math.max(v.rFloor, Math.ceil((best!.R[k] * margin) / E) * E);
    out.chains[v.id] = { R, seconds: (R * v.msPerChain) / 1000, msPerChain: v.msPerChain };
  });
  return out;
}

/** Keep only the global aggregate of a u vector (the Q4 feasibility check: tiles/masks can be enlarged, the global never). */
export function globalOnly(u: Float64Array, side: { nTiles: number; nMasks: number }): Float64Array {
  const A = 1 + side.nTiles + side.nMasks, g = new Float64Array(u.length);
  for (let c = 0; c < 4; c++) g[c * A] = u[c * A];
  return g;
}

/** Q4 cap decision of a unit side (pure). */
export function capDecision(seconds: number, globalOnlySeconds: number, enlarged: boolean): { status: 'ok' | 'enlarge' | 'raised' | 'infeasible'; capS: number } {
  if (seconds <= UNIT_CAP_S) return { status: 'ok', capS: UNIT_CAP_S };
  if (!enlarged) return { status: 'enlarge', capS: UNIT_CAP_S };
  if (globalOnlySeconds <= UNIT_CAP_S && seconds <= UNIT_CAP_RAISED_S) return { status: 'raised', capS: UNIT_CAP_RAISED_S };
  if (seconds <= UNIT_CAP_RAISED_S) return { status: 'raised', capS: UNIT_CAP_RAISED_S };
  return { status: 'infeasible', capS: UNIT_CAP_RAISED_S };
}

/** Chain batches per GPU-lock chunk (whole batches, ≤ LOCK_CHUNK_S; chains are never split, TD25). */
export function chainChunks(R: number, E: number, estSeconds: number): number[] {
  const batches = R / E;
  const per = Math.max(1, estSeconds > LOCK_CHUNK_S ? Math.floor((batches * LOCK_CHUNK_S) / estSeconds) : batches);
  const out: number[] = [];
  for (let b = 0; b < batches; b += per) out.push(Math.min(per, batches - b));
  return out;
}

// ------------------------------------------------------------------------------------------------ T16 (pure)

/** T16 of a chain run vs its PT reference (restir-temporal-api.md §6.4 "T16 for M5 units"). */
export function t16ChainProblems(cm: Record<string, any>, pt: Record<string, any>, o: { plant?: boolean; rounds: number; frame?: number | 'avg'; staticScene?: boolean; members?: number }): string[] {
  const p: string[] = [];
  const t = cm.t16 ?? {};
  if (cm.kernel !== 'restir' || cm.kind !== 'chains') p.push('chain meta: not a restir chain run');
  if (!cm.ok) p.push(`chain run errors: ${(cm.errors ?? []).slice(0, 6).join('; ')}`);
  if (!pt.ok) p.push(`PT reference errors: ${(pt.errors ?? []).join('; ')}`);
  if (!o.plant && t.validationModeUnbiased !== true) p.push('validation mode not unbiased (a plant without --plant)');
  if (o.plant && !(t.plantsNamed?.length > 0)) p.push('plant run without a named plant');
  if (t.internalScale !== 1) p.push(`internal scale ${t.internalScale}`);
  if (t.denoiser !== 'none' || t.upscaler !== 'none') p.push('denoiser/upscaler active');
  if (!/^linear /.test(t.readback ?? '')) p.push('readback is not the linear radiance');
  if (t.jitterMode !== 'iid-per-run' || pt.config?.jitter !== 'iid-per-run') p.push('jitter is not iid-per-run on both sides');
  if (t.maxBounces !== pt.config?.maxBounces) p.push(`maxBounces ${t.maxBounces} != PT ${pt.config?.maxBounces}`);
  if (t.lightMode !== 'A' || pt.config?.lightMode !== 'A') p.push('not Mode A on both sides');
  if (t.rr !== false || pt.config?.rr !== false) p.push('RR on (math §25: RR off in 3.2–3.6)');
  if (t.temporal !== true) p.push('temporal off');
  if (t.cCap !== 20) p.push(`cCap ${t.cCap} != 20`);
  if (cm.config?.scene !== pt.config?.scene) p.push('scene bytes differ (package sha256)');
  if (cm.width !== pt.width || cm.height !== pt.height) p.push(`resolution ${cm.width}x${cm.height} != PT ${pt.width}x${pt.height}`);
  const rn = typeof cm.config?.env === 'object' ? cm.config.env.nee : 'none', pn = typeof pt.config?.env === 'object' ? pt.config.env.nee : 'none';
  if (rn !== pn) p.push(`env NEE ${rn} != PT ${pn}`);
  if ((o.members ?? E_MEMBERS) !== cm.members) p.push(`members ${cm.members} != ${o.members ?? E_MEMBERS}`);
  if (t.spatialRoundsExecuted !== o.rounds) p.push(`spatial rounds executed ${t.spatialRoundsExecuted} != ${o.rounds}`);
  // the PT reference renders the test frame's resolved state (static packages: the base state, frame null)
  if (typeof o.frame === 'number' && !o.staticScene && pt.config?.frame !== o.frame) p.push(`PT reference frame ${pt.config?.frame} != test frame ${o.frame}`);
  if (o.staticScene && pt.config?.frame !== null && pt.config?.frame !== undefined) p.push('static unit compared against a frame-override PT reference');
  const hp: boolean[] = t.historyPattern ?? [];
  const resets = new Set<number>(t.resetFrames ?? [0]);
  if (!hp.length) p.push('no per-frame history record');
  hp.forEach((v, i) => { if (v === resets.has(i)) p.push(`frame ${i}: history ${v ? 'valid' : 'invalid'} (resets at ${[...resets].join(',')})`); });
  return p;
}

// ------------------------------------------------------------------------------------------------ hashes, process helpers

const sha = (b: string | Buffer) => createHash('sha256').update(b).digest('hex');
const stableJson = (v: unknown): string => JSON.stringify(v, (_k, x) => (x && typeof x === 'object' && !Array.isArray(x)
  ? Object.fromEntries(Object.entries(x as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : 1))) : x));
function hashFiles(rel: string[]): string { return sha(rel.map((r) => `${r}\0${sha(readFileSync(path.join(ROOT, r)))}\n`).join('')); }
function listDir(rel: string): string[] {
  const out: string[] = [];
  const walk = (d: string) => { for (const n of readdirSync(path.join(ROOT, d)).sort()) { const p = path.join(d, n); if (statSync(path.join(ROOT, p)).isDirectory()) walk(p); else out.push(p); } };
  walk(rel);
  return out;
}
let chainHash: string | undefined;
/** Chain code hash: TS closure of restir-chain-run.ts + every WGSL file (pilot cache key). PT: the M4 PT closure. */
export function codeHashes(): { pt: string; chains: string } {
  chainHash ??= hashFiles([...new Set([...tsClosure(['validation/harness/restir-chain-run.ts']), ...listDir('src/core/shaders').filter((f) => f.endsWith('.wgsl'))])].sort());
  return { pt: m4CodeHashes().pt, chains: chainHash };
}
const packageHash = (dir: string) => hashFiles(readdirSync(path.join(ROOT, dir)).filter((n) => statSync(path.join(ROOT, dir, n)).isFile()).sort().map((n) => path.join(dir, n)));

const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const r4 = (x: unknown) => (typeof x === 'number' ? Number(x.toPrecision(4)) : x);
const pct = (x: number | undefined | null, d = 3) => (typeof x === 'number' ? `${(x * 100).toFixed(d)}%` : 'n/a');
const tryJson = (p: string): Record<string, any> | undefined => { try { return JSON.parse(readFileSync(path.join(ROOT, p), 'utf8')); } catch { return undefined; } };
const safe = (s: string) => s.replace(/[^\w.@-]+/g, '_');

function sh(cmd: string, argv: string[], echo: (l: string) => boolean = () => true, env?: Record<string, string>): { code: number; out: string; seconds: number } {
  const t0 = performance.now();
  const r = spawnSync(cmd, argv, { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 28, timeout: TIMEOUT_MS, ...(env ? { env: { ...process.env, ...env } } : {}) });
  const out = `${r.stdout ?? ''}${r.stderr ?? ''}`;
  for (const l of out.split('\n')) if (l && echo(l)) console.log(`  ${l}`);
  return { code: r.status ?? 1, out, seconds: (performance.now() - t0) / 1000 };
}
function vitestConfigArgs(): string[] {
  try { if (lstatSync(path.join(ROOT, 'node_modules')).isSymbolicLink() && existsSync(path.join(ROOT, 'vitest.m4local.config.ts'))) return ['--config', 'vitest.m4local.config.ts']; } catch { /* default */ }
  return [];
}

type Add = (name: string, ok: boolean, seconds: number, data?: unknown, detail?: string) => void;
type Rec = (name: string, ok: boolean, detail?: string) => void;
interface Run { dir: string; meta: Record<string, any>; seconds: number; cacheHit?: boolean }

// ------------------------------------------------------------------------------------------------ job collection (Changelog E-11)

/** A deferred run-batches invocation: collected in a dry pass, then run in `--jobs` groups (one Chrome, one GPU-lock hold
 *  per ≤ 10 min of GPU work), then post-processed into its cache directory by `finish`. */
interface Job { argv: string[]; est: number; label: string; finish(): void }
let collect: Job[] | undefined;
const DEFERRED = -999;
const JOB_GROUP_S = 600;
const JOB_OVERHEAD_S = 5;
const noAdd: Add = () => undefined;

/** Run `fn` as a dry pass collecting every small cacheable render, run them grouped, then return (the caller re-runs
 *  the phase for real: every collected render is now a cache hit). */
function prefetch(label: string, dir: string, add: Add, fn: (a: Add) => void): void {
  collect = [];
  try { fn(noAdd); } finally { /* keep what was collected */ }
  // the same render can be requested twice in one pass (frame t's reference is frame t+1's t−1 reference): once
  const jobs = [...new Map(collect.map((j) => [j.label, j])).values()];
  collect = undefined;
  if (!jobs.length) return;
  const groups: Job[][] = [];
  let cur: Job[] = [], est = 0;
  for (const j of jobs) {
    if (cur.length && est + j.est + JOB_OVERHEAD_S > JOB_GROUP_S) { groups.push(cur); cur = []; est = 0; }
    cur.push(j); est += j.est + JOB_OVERHEAD_S;
  }
  if (cur.length) groups.push(cur);
  groups.forEach((g, i) => {
    const file = path.join(dir, 'jobs', `${safe(label)}-${i}.json`);
    mkdirSync(path.dirname(path.join(ROOT, file)), { recursive: true });
    writeFileSync(path.join(ROOT, file), JSON.stringify(g.map((j) => j.argv), null, 1));
    console.log(`\n--- ${label}: job group ${i + 1}/${groups.length} (${g.length} runs, est ${g.reduce((a, j) => a + j.est + JOB_OVERHEAD_S, 0).toFixed(0)} s)`);
    const r = sh('npx', ['tsx', 'validation/harness/run-batches.ts', '--jobs', file], (l) => /^(FAIL)\s|errors:|Error|lock wait|JOBS:/.test(l));
    let failed = 0;
    for (const j of g) { try { j.finish(); } catch (e) { failed++; console.log(`  finish ${j.label}: ${e instanceof Error ? e.message : e}`); } }
    add(`${label}: job group ${i + 1}/${groups.length} (${g.length} runs under one GPU-lock hold)`, r.code === 0 && !failed, r.seconds, { file }, r.code === 0 ? undefined : r.out.slice(-300));
  });
}

// ------------------------------------------------------------------------------------------------ PT runs (cached, chunked)

const rbEcho = (l: string) => /^(FAIL)\s|errors:|Error|lock wait/.test(l);

/** One sequential PT run (run-batches --kernel pt [--frames t]), chunked into ≤ 12-min GPU-lock holds and merged. */
function ptRunInto(pdir: string, frame: number | undefined, spp: number, B: number, seed: number, dest: string, estSeconds: number, extra: string[] = []): { dir?: string; meta?: Record<string, any>; code: number; out: string; seconds: number } {
  const k = chunkBatches(B, estSeconds);
  mkdirSync(path.dirname(path.join(ROOT, dest)), { recursive: true });
  rmSync(path.join(ROOT, dest), { recursive: true, force: true });
  // ≤ 32 spp per dispatch/submit (≈ 20 ms nominal at 256²): PT references of the first ixs_i run had 0.9–1.2 s submits
  // (hard cap 200 ms) at the adaptive ≤ 256 (E-21); the result is the same up to f32 summation order
  const base = ['--package', pdir, '--kernel', 'pt', '--spp', String(spp), '--seed', String(seed), '--max-spp-per-dispatch', String(PT_MAX_SPP_PER_DISPATCH), ...(frame !== undefined ? ['--frames', String(frame)] : []), ...extra];
  const produced = (run: string) => path.join(ROOT, 'validation/out', frame !== undefined ? `${run}-f${frame}` : run);
  const metas: Record<string, any>[] = [];
  let seconds = 0, out = '';
  mkdirSync(path.join(ROOT, dest), { recursive: true });
  const finalize = () => {
    const meta = metas.length === 1 ? { ...metas[0], files: metas[0].files } : mergeChunkMetas(metas);
    const files = readdirSync(path.join(ROOT, dest)).filter((f) => /^batch_\d{3}\.pfm$/.test(f)).sort();
    const imgs = files.map((f) => decodePFM(new Uint8Array(readFileSync(path.join(ROOT, dest, f)))));
    const mean = new Float32Array(imgs[0].data.length);
    for (const im of imgs) for (let i = 0; i < mean.length; i++) mean[i] += im.data[i] / imgs.length;
    writeFileSync(path.join(ROOT, dest, 'mean.pfm'), encodePFM({ width: imgs[0].width, height: imgs[0].height, channels: 3, data: mean }));
    writeFileSync(path.join(ROOT, dest, 'meta.json'), JSON.stringify(meta, null, 1));
    return meta;
  };
  if (collect && k >= B) {
    const run = `m5pt-${sha(dest).slice(0, 10)}-c0`;
    collect.push({ argv: [...base, '--batches', String(B), '--run', run], est: estSeconds, label: dest, finish: () => {
      const from = produced(run);
      const meta = tryJson(path.relative(ROOT, path.join(from, 'meta.json')));
      if (!meta) throw new Error(`${run}: no meta.json`);
      for (const f of readdirSync(from)) if (/^batch_\d{3}\.pfm$/.test(f)) renameSync(path.join(from, f), path.join(ROOT, dest, f));
      rmSync(from, { recursive: true, force: true });
      metas.push(meta);
      finalize();
    } });
    return { code: DEFERRED, out: '', seconds: 0 };
  }
  for (let off = 0; off < B; off += k) {
    const n = Math.min(k, B - off);
    const run = `m5pt-${sha(dest).slice(0, 10)}-c${off}`;
    const r = sh('npx', ['tsx', 'validation/harness/run-batches.ts', ...base, '--batches', String(n), ...(off ? ['--batch-offset', String(off)] : []), '--run', run], rbEcho);
    seconds += r.seconds; out += r.out.slice(-2000);
    const from = produced(run);
    const meta = tryJson(path.relative(ROOT, path.join(from, 'meta.json')));
    if (r.code !== 0 || !meta) { rmSync(from, { recursive: true, force: true }); return { code: r.code || 1, out, seconds }; }
    for (const f of readdirSync(from)) if (/^batch_\d{3}\.pfm$/.test(f)) renameSync(path.join(from, f), path.join(ROOT, dest, f));
    rmSync(from, { recursive: true, force: true });
    metas.push(meta);
  }
  const meta = finalize();
  return { code: meta.ok ? 0 : 1, out, seconds, dir: dest, meta };
}

function ptRef(pkg: string, frame: number | undefined, spp: number, B: number, seed: number, add: Add, estSeconds: number, tag = 'ref', extra: string[] = [], dirOverride?: string): Run | undefined {
  const pdir = dirOverride ?? pkgDir(pkg);
  const keyObj = { kind: 'pt', pkg, dir: dirOverride ?? null, frame: frame ?? null, packageHash: packageHash(pdir), spp, B, seed, rr: false, code: codeHashes().pt, extra };
  const key = sha(stableJson(keyObj)).slice(0, 16);
  const dest = path.join(tag === 'pilot' ? PILOTS : PTREFS, `${dirOverride ? path.basename(path.dirname(path.dirname(dirOverride))) + '-only-' + path.basename(dirOverride) : pkg}-f${frame ?? 'base'}-s${seed}-${spp}x${B}-${key}`);
  const meta = tryJson(path.join(dest, 'meta.json'));
  const n = existsSync(path.join(ROOT, dest)) ? readdirSync(path.join(ROOT, dest)).filter((f) => /^batch_\d{3}\.pfm$/.test(f)).length : 0;
  const step = `PT ${tag} ${pkg}${dirOverride ? ` (${path.basename(dirOverride)} only)` : ''} f${frame ?? 'base'} (${spp} spp x ${B}, seed ${seed})`;
  if (meta?.ok && n === B) { add(step, true, 0, { dir: dest, cache_hit: true }, 'cache hit'); return { dir: dest, meta, seconds: 0, cacheHit: true }; }
  if (!collect) console.log(`\n--- ${step}`);
  const r = ptRunInto(pdir, frame, spp, B, seed, dest, estSeconds, extra);
  if (r.code === DEFERRED) return undefined;
  const ok = r.code === 0 && !!r.meta?.ok;
  if (r.dir) writeFileSync(path.join(ROOT, dest, 'cache-key.json'), `${JSON.stringify(keyObj, null, 1)}\n`);
  add(step, ok, r.seconds, { dir: dest, cache_hit: false }, ok ? `rendered in ${(r.meta!.timings.totalMs / 1000).toFixed(1)} s` : `exit ${r.code} ${r.out.slice(-300)}`);
  return ok ? { dir: dest, meta: r.meta!, seconds: r.seconds, cacheHit: false } : undefined;
}

// ------------------------------------------------------------------------------------------------ chain runs (chunked)

export interface ChainArgs { pkg: string; preset: 'temporal' | 'full' | 'initial' | 'initial-rr' | 'offline'; R: number; seed: number; frames?: number; testFrames: number[]; average?: { from: number; to: number }; masks?: string; extra?: string[] }

/** run-batches --chains in GPU-lock chunks of whole batches; per test frame the chunk npz files are merged (dynamic.py merge-npz). */
function chainRun(a: ChainArgs, dest: string, estSeconds: number): { dir?: string; meta?: Record<string, any>; code: number; out: string; seconds: number } {
  const chunks = chainChunks(a.R, E_MEMBERS, estSeconds);
  rmSync(path.join(ROOT, dest), { recursive: true, force: true });
  mkdirSync(path.join(ROOT, dest), { recursive: true });
  const base = ['--package', pkgDir(a.pkg), '--kernel', 'restir', '--preset', a.preset, '--seed', String(a.seed), '--members', String(E_MEMBERS),
    '--test-frames', a.testFrames.join(','), ...(a.frames ? ['--chain-frames', String(a.frames)] : []),
    ...(a.average ? ['--average', `${a.average.from}:${a.average.to}`] : []), ...(a.masks ? ['--masks', a.masks] : []), ...(a.extra ?? [])];
  let seconds = 0, out = '', b0 = 0;
  const parts: string[] = [];
  if (collect && chunks.length === 1) {
    const run = `m5ch-${sha(dest).slice(0, 10)}-b0`;
    collect.push({ argv: [...base, '--chains', String(a.R), '--batch-offset', '0', '--run', run], est: estSeconds, label: dest, finish: () => {
      const r = finishChainParts(a, [path.join('validation/out', run)], dest, 0, '');
      if (!r.dir) throw new Error(r.out.slice(-300));
    } });
    return { code: DEFERRED, out: '', seconds: 0 };
  }
  for (const nb of chunks) {
    const run = `m5ch-${sha(dest).slice(0, 10)}-b${b0}`;
    console.log(`  chains batches ${b0}..${b0 + nb - 1} of ${a.R / E_MEMBERS}`);
    const r = sh('npx', ['tsx', 'validation/harness/run-batches.ts', ...base, '--chains', String(nb * E_MEMBERS), '--batch-offset', String(b0), '--run', run], rbEcho);
    seconds += r.seconds; out += r.out.slice(-2000);
    const from = path.join('validation/out', run);
    if (r.code !== 0 && !tryJson(path.join(from, 'meta.json'))) return { code: r.code || 1, out, seconds };
    parts.push(from);
    b0 += nb;
  }
  return finishChainParts(a, parts, dest, seconds, out);
}

/** Merge the chunk runs of one chain unit into `dest` (per test frame: dynamic.py merge-npz; meta.json summed). */
function finishChainParts(a: ChainArgs, parts: string[], dest: string, seconds: number, out: string): { dir?: string; meta?: Record<string, any>; code: number; out: string; seconds: number } {
  if (parts.some((p) => !tryJson(path.join(p, 'meta.json')))) return { code: 1, out: `${out}\nmissing meta.json`, seconds };
  const subs = [...(a.testFrames.map((t) => `f${t}`)), ...(a.average ? ['avg'] : [])];
  for (const s of subs) {
    const inputs = parts.map((p) => path.join(p, s)).filter((p) => existsSync(path.join(ROOT, p, 'ensemble.npz')));
    if (inputs.length !== parts.length) return { code: 1, out: `${out}\nmissing ${s} in a chunk`, seconds };
    const m = sh(PY, ['validation/tools/dynamic.py', 'merge-npz', '--out', path.join(dest, s), ...inputs], () => false);
    if (m.code !== 0) return { code: 1, out: `${out}\nmerge ${s}: ${m.out.slice(-400)}`, seconds };
  }
  const metas = parts.map((p) => tryJson(path.join(p, 'meta.json'))!);
  const meta = { ...metas[0], chains: metas.reduce((x, m) => x + (m.chains ?? 0), 0), batches: metas.reduce((x, m) => x + (m.batches ?? 0), 0),
    ok: metas.every((m) => m.ok), errors: metas.flatMap((m) => m.errors ?? []), chunks: metas.map((m) => ({ batchOffset: m.batchOffset, chains: m.chains, totalMs: m.timings?.totalMs })),
    timings: { totalMs: metas.reduce((x, m) => x + (m.timings?.totalMs ?? 0), 0), batchMs: metas.flatMap((m) => m.timings?.batchMs ?? []) },
    counters: { nonFinite: metas.reduce((x, m) => x + (m.counters?.nonFinite ?? 0), 0), negative: metas.reduce((x, m) => x + (m.counters?.negative ?? 0), 0) } };
  writeFileSync(path.join(ROOT, dest, 'meta.json'), JSON.stringify(meta, null, 1));
  for (const p of parts) rmSync(path.join(ROOT, p), { recursive: true, force: true });
  return { code: meta.ok ? 0 : 1, out, seconds, dir: dest, meta };
}

function pilotChains(a: ChainArgs, add: Add): Run | undefined {
  const keyObj = { kind: 'chains', ...a, masks: a.masks ? packageHashMaybe(a.masks) : null, packageHash: packageHash(pkgDir(a.pkg)), code: codeHashes().chains };
  const key = sha(stableJson(keyObj)).slice(0, 16);
  const dest = path.join(PILOTS, `${a.pkg}-${a.preset}${a.extra?.length ? `-${safe(a.extra.join(''))}` : ''}-${key}`);
  const meta = tryJson(path.join(dest, 'meta.json'));
  const step = `pilot chains ${a.pkg} ${a.preset} ${a.extra?.join(' ') ?? ''} (${a.R} chains)`;
  if (meta?.ok !== undefined && a.testFrames.every((t) => existsSync(path.join(ROOT, dest, `f${t}`, 'ensemble.npz')))) { add(step, true, 0, { dir: dest, cache_hit: true }, 'cache hit'); return { dir: dest, meta: meta!, seconds: 0, cacheHit: true }; }
  if (!collect) console.log(`\n--- ${step}`);
  const r = chainRun(a, dest, a.frames ? (a.frames * (a.R / E_MEMBERS) * (a.preset === 'full' ? 0.04 : 0.025)) : 30);
  if (r.code === DEFERRED) return undefined;
  // pilots are sized even when T16 fails (e.g. P0 stubs); the unit runs gate T16
  const ok = !!r.dir && !!r.meta;
  if (ok) writeFileSync(path.join(ROOT, dest, 'cache-key.json'), `${JSON.stringify(keyObj, null, 1)}\n`);
  add(step, ok, r.seconds, { dir: dest, errors: r.meta?.errors?.slice(0, 4) }, ok ? `${(r.meta!.timings.totalMs / 1000).toFixed(1)} s${r.meta!.ok ? '' : ` (run errors: ${r.meta!.errors.slice(0, 2).join('; ')})`}` : r.out.slice(-300));
  return ok ? { dir: dest, meta: r.meta!, seconds: r.seconds } : undefined;
}
function packageHashMaybe(dir: string): string | null { return existsSync(path.join(ROOT, dir)) ? hashFiles(listDir(dir)) : null; }

// ------------------------------------------------------------------------------------------------ masks

interface MaskSet { dir: string; names: string[]; testMasks: { name: string; file: string }[]; bits: Map<number, Uint16Array> }

/** dyn_masks.py for one package: mask PT refs at t and t−1 (seed 7301), the disocc harness run, dominance renders. */
export function buildMasks(pkg: string, frames: number[], add: Add, o: { partition?: boolean; dominance?: { frames: number[]; names: string[] }; sil?: boolean; tag?: string } = {}): MaskSet | undefined {
  const tag = o.tag ?? (o.dominance || o.sil ? 'plant' : 'gate');
  const dir = path.join(MASKS, pkg, tag);
  const partition = o.partition ?? true;
  const est = MASK_PT.spp * MASK_PT.B * 0.0007 * 2;
  // disocclusion flags (production temporal_pixel, pixel centre)
  const disDir = path.join(MASKS, pkg, 'disocc');
  const need = frames.filter((t) => !existsSync(path.join(ROOT, disDir, `f${t}`, 'disocc.bin')));
  if (partition && need.length) {
    const run = `m5dis-${pkg}-${stamp()}`;
    const argv = ['--package', pkgDir(pkg), '--kernel', 'restir', '--preset', 'temporal', '--mode', 'disocc', '--test-frames', need.join(','), '--seed', String(SEEDS.ptMask), '--run', run];
    const absorb = () => {
      for (const t of need) {
        const src = path.join(ROOT, 'validation/out', run, `f${t}`, 'disocc.bin');
        if (existsSync(src)) { mkdirSync(path.join(ROOT, disDir, `f${t}`), { recursive: true }); renameSync(src, path.join(ROOT, disDir, `f${t}`, 'disocc.bin')); }
      }
      rmSync(path.join(ROOT, 'validation/out', run), { recursive: true, force: true });
    };
    if (collect) collect.push({ argv, est: 2 * need.length, label: `disocc ${pkg}`, finish: absorb });
    else {
      const r = sh('npx', ['tsx', 'validation/harness/run-batches.ts', ...argv], rbEcho);
      absorb();
      add(`M_disocc harness ${pkg} f${need.join(',')}`, r.code === 0, r.seconds, undefined, r.code === 0 ? undefined : r.out.slice(-300));
    }
  }
  const set: MaskSet = { dir, names: [], testMasks: [], bits: new Map() };
  for (const t of frames) {
    const refT = ptRef(pkg, t, MASK_PT.spp, MASK_PT.B, SEEDS.ptMask, add, est, 'mask');
    const refP = partition ? ptRef(pkg, t - 1, MASK_PT.spp, MASK_PT.B, SEEDS.ptMask, add, est, 'mask') : undefined;
    if (collect) {
      for (const n of o.dominance?.frames.includes(t) ? o.dominance.names : []) ptRef(pkg, t, MASK_PT.spp, MASK_PT.B, SEEDS.ptMask, add, est, 'mask', [], dominancePackage(pkg, n));
      continue;
    }
    if (!refT || (partition && !refP)) return undefined;
    const domArgs: string[] = [];
    if (o.dominance?.frames.includes(t)) {
      for (const n of o.dominance.names) {
        const d = ptRef(pkg, t, MASK_PT.spp, MASK_PT.B, SEEDS.ptMask, add, est, 'mask', [], dominancePackage(pkg, n));
        if (!d) return undefined;
        domArgs.push('--dominance', `${n}=${d.dir}`);
      }
    }
    const dis = path.join(disDir, `f${t}`, 'disocc.bin');
    const r = sh(PY, ['validation/tools/dyn_masks.py', '--ref-t', refT.dir, ...(partition ? ['--ref-prev', refP!.dir] : ['--no-partition']), '--frame', String(t), '--out', dir,
      ...(partition && existsSync(path.join(ROOT, dis)) ? ['--disocc', dis] : []), ...domArgs, ...(o.dominance || o.sil ? ['--sil'] : [])], () => false);
    const mj = tryJson(path.join(dir, `f${t}`, 'masks.json'));
    add(`masks ${pkg} f${t} (${tag})`, r.code === 0 && !!mj, r.seconds, mj && { names: mj.names, dropped: mj.dropped, pixels: mj.pixels }, r.code === 0 ? mj?.names.join(' ') : r.out.slice(-300));
    if (!mj) return undefined;
  }
  return collect ? undefined : set;
}

function masksOf(dir: string, t: number): { names: string[]; test: { name: string; file: string }[]; perMask: Uint8Array[] } {
  const mj = tryJson(path.join(dir, `f${t}`, 'masks.json'))!;
  const raw = readFileSync(path.join(ROOT, dir, `f${t}`, 'masks.bin'));
  const bits = new Uint16Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  const perMask = (mj.names as string[]).map((_, i) => Uint8Array.from(bits, (b) => (b >> i) & 1));
  return { names: mj.names, test: (mj.test_json_masks as { name: string; file: string }[]).map((m) => ({ name: m.name, file: path.join(ROOT, dir, `f${t}`, m.file) })), perMask };
}

/** Single-emitter copy of a sequence package (dominance renders, §6.4): every other light disabled in every frame, and
 *  the env at strength 0 unless the emitter is the env (then every analytic light is disabled). */
function dominancePackage(pkg: string, name: string): string {
  const src = pkgDir(pkg);
  const dst = path.join(MASKS, pkg, 'dominance-pkg', safe(name));
  const json = tryJson(path.join(src, 'scene.json'))!;
  const keep = name === 'env' ? undefined : (json.lightNames?.[name] ?? Number(name));
  if (name !== 'env' && !json.lights.some((l: any) => l.id === keep)) throw new Error(`${pkg}: no light ${name}`);
  const frames = (json.frames ?? []).map((f: any) => {
    const lights = { ...(f.lights ?? {}) };
    for (const l of json.lights) if (l.id !== keep) lights[String(l.id)] = { ...(lights[String(l.id)] ?? {}), enabled: false };
    return { ...f, lights, ...(json.env && name !== 'env' ? { env: { ...(f.env ?? {}), strength: 0 } } : {}) };
  });
  const out = { ...json, name: `${json.name}__only_${name}`, frames, ...(json.env && name !== 'env' ? { env: { ...json.env, strength: 0 } } : {}),
    lights: json.lights, dominance: { of: pkg, emitter: name } };
  rmSync(path.join(ROOT, dst), { recursive: true, force: true });
  mkdirSync(path.join(ROOT, dst), { recursive: true });
  for (const f of readdirSync(path.join(ROOT, src))) if (f !== 'scene.json') writeFileSync(path.join(ROOT, dst, f), readFileSync(path.join(ROOT, src, f)));
  writeFileSync(path.join(ROOT, dst, 'scene.json'), JSON.stringify(out, null, 1));
  return dst;
}

// ------------------------------------------------------------------------------------------------ compare helpers

function writeTest(dir: string, name: string, nU: number, stage: 'B' | 'dyn', tile: number, extra: Record<string, unknown> = {}): string {
  const p = path.join(dir, 'tests', `test-${safe(name)}.json`);
  mkdirSync(path.join(ROOT, dir, 'tests'), { recursive: true });
  const t0 = stage === 'B' ? 32 : 64;
  writeFileSync(path.join(ROOT, p), JSON.stringify({
    name, stage, channels: ['Y', 'R', 'G', 'B'], n_units: nU, tier: 'tight', num_eps: NUM_EPS, num_eps_note: 'two f32 implementations (cycles-deviations.md D2)',
    min_replicates: 16, ...(tile !== t0 ? { tile, aggregate_note: `unit cap ${UNIT_CAP_S / 60} min (restir-temporal-api.md §6.6 control 5)` } : {}), ...extra,
  }, null, 1));
  return p;
}
function compare(ours: string, ref: string, test: string, out: string, rerunOf?: string) {
  return sh(PY, ['validation/tools/compare.py', '--ours', ours, '--ref', ref, '--test', test, '--out', out, ...(rerunOf ? ['--rerun-of', rerunOf] : [])], () => false);
}
function summarize(rep: Record<string, any>) {
  const Y = rep.channels.Y;
  return {
    status: rep.status, failed_checks: rep.failed_checks,
    global_rel_Y: r4(Y.global_.rel), mdb_global_Y: r4(Y.global_.mdb), worst_tile_rel_Y: r4(Y.tiles.worst_rel), mdb_tile_max_Y: r4(Y.tiles.mdb_max),
    masks_Y: (Y.masks ?? []).map((m: any) => ({ name: m.name, rel: r4(m.rel), passed: m.passed })),
    multiplier_needed: r4(Math.max(...['Y', 'R', 'G', 'B'].map((c) => Math.max(rep.channels[c].global_.replicate_multiplier_needed ?? 1, rep.channels[c].tiles.replicate_multiplier_needed ?? 1)))),
  };
}

// ------------------------------------------------------------------------------------------------ the gate

export interface M5Options {
  only?: Set<string>; part?: 'core' | 'static'; pilotOnly?: boolean; writeBudget?: boolean; prerenderPtRefs?: boolean;
  /** Reuse the first-seed chain runs of an earlier gate directory (unit chains/<id> with the same R, frames and seed):
   *  re-evaluates a part after a harness-only fix without re-rendering. Re-runs on disjoint seeds still render. */
  reuseChains?: string;
  /** Run only these plants (ids; 'aa' = the A/A pair + synthetic W × 1.003), after the part's sizing; no Gate 0, no units. */
  plants?: Set<string>;
  /** Fresh, disjoint seeds for the plants (chains and their 4× PT references): plants whose prediction was revised after a
   *  measurement are confirmed out of sample (coordinator, B-12 / D-5). Default 0. */
  plantSeedOffset?: number;
}
let plantSeedOffset = 0;
let reuseChainsDir: string | undefined;

interface UnitPlan {
  id: string; kind: 'static' | 'avg' | 'dyn' | 'u8'; pkg: string; rung: string; preset: ChainArgs['preset']; variant?: string; extra: string[];
  frames: number; testFrames: number[]; average?: { from: number; to: number }; tile: number; stage: 'B' | 'dyn';
  R: number; chainSeconds: number; msPerChain: number; masks?: string; cap: string; notes: string[];
  /** Sized for another item (the A/A pair) but not run in this part. */
  sizeOnly?: boolean;
}
interface PtPlan { pkg: string; frame: number | undefined; spp: number; B: number; seconds: number }

export function milestoneM5(record: Rec, o: M5Options = {}): void {
  const t0 = performance.now();
  const runId = `m5-gate-${stamp()}`;
  const dir = path.join('validation/out', runId);
  mkdirSync(path.join(ROOT, dir), { recursive: true });
  const steps: { name: string; ok: boolean; seconds: number; data?: unknown; detail?: string }[] = [];
  const add: Add = (name, ok, seconds, data, detail) => {
    steps.push({ name, ok, seconds: Math.round(seconds * 10) / 10, data, detail });
    record(name, ok, `${detail ? `${detail}, ` : ''}${seconds.toFixed(1)} s`);
  };
  const runStep = (name: string, cmd: string, argv: string[], echo?: (l: string) => boolean, env?: Record<string, string>) => {
    console.log(`\n--- ${name}: ${cmd} ${argv.join(' ')}`);
    const r = sh(cmd, argv, echo, env);
    const log = path.join(dir, 'logs', `${safe(name).slice(0, 80)}.log`);
    mkdirSync(path.join(ROOT, dir, 'logs'), { recursive: true });
    writeFileSync(path.join(ROOT, log), r.out);
    // a vitest step whose filter (-t) matched no test exits 0 with every test skipped: that is a failure of the gate
    const noTests = argv[0] === 'vitest' && !/\d+ passed/.test(r.out);
    add(name, r.code === 0 && !noTests, r.seconds, { log, ...(env ? { env } : {}) }, `exit ${r.code}${noTests ? ', no test ran' : ''}`);
    return r;
  };
  const plantsOnly = !!o.plants;
  const full = !o.only && !o.pilotOnly && !o.prerenderPtRefs && !plantsOnly;
  reuseChainsDir = o.reuseChains;
  plantSeedOffset = o.plantSeedOffset ?? 0;
  const doCore = o.part !== 'static', doStatic = o.part !== 'core';
  const sel = (pkg: string) => !o.only || o.only.has(pkg);
  const nU = nUnits();
  const hashes = codeHashes();
  console.log(`M5 gate ${runId}: n_units ${nU}, PT code ${hashes.pt.slice(0, 12)}, chains code ${hashes.chains.slice(0, 12)}${o.part ? `, part ${o.part}` : ''}`);

  // ---- Gate 0 ---------------------------------------------------------------------------------------------------------
  add('M5 gate tools (venv python)', existsSync(PY), 0, undefined, existsSync(PY) ? undefined : `missing ${PY}`);
  ensurePackages(add);
  if (full && doCore) {
    packagesDeterministic(dir, add);
    runStep('typecheck', 'npx', ['tsc', '--noEmit']);
    runStep('vitest cpu (tests/restir incl. tmis, tqueue, refresh-ref, light-maps, config-hash, frame-state, gate-m5-config + regressions)', 'npx',
      ['vitest', 'run', ...vitestConfigArgs(), '--project', 'cpu'], (l) => /Test Files|Tests |FAIL|✗|×/.test(l));
    runStep('python tests (stats, compare, dynamic, dyn_masks, plant_sign)', PY, ['-m', 'pytest', 'validation/tools/tests', '-q'], (l) => /passed|failed|error/i.test(l));
    for (const [file, what] of GPU_SUITES) {
      const rel = `validation/gpu-tests/${file}.gpu.test.ts`;
      if (!existsSync(path.join(ROOT, rel))) { add(`${file} (chrome): ${what}`, false, 0, undefined, `missing ${rel}`); continue; }
      for (const h of gpuSuiteHolds(file)) {   // restir-shift: one hold per T3 variant (≤ 24 min, E-16), the rest ≤ 12 min
        withGpuLockSync(`gate-m5-${file}`, () => {
          runStep(`${file} (chrome)${h.label}: ${what}`, 'npx', ['vitest', 'run', ...vitestConfigArgs(), '--project', 'chrome', '--reporter=verbose', rel, ...h.args],
            (l) => /Tests |FAIL|✗|×|AssertionError|LOGIC|FP-BOUNDARY|violation/.test(l), GPU_SUITE_ENV[file]);
        });
      }
    }
    for (const c of T32_RARE_CASES) {   // -t is a regex: the names contain "+" (escaped below)
      withGpuLockSync(`gate-m5-t32-${safe(c)}`, () => {
        runStep(`restir-temporal T3-2 ${c} (>= 1e6 per bin; LOGIC 0, FP <= 1e-5, PLATFORM 0)`, 'npx', ['vitest', 'run', ...vitestConfigArgs(), '--project', 'chrome', '--reporter=verbose', '--testTimeout', '900000',
          'validation/gpu-tests/restir-temporal.gpu.test.ts', '-t', c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')], (l) => /Tests |FAIL|✗|×|AssertionError|LOGIC|FP-BOUNDARY|PLATFORM|bin/.test(l), T32_RARE_ENV);
      });
    }
    for (const g of M5_GATE0_PLANTS) {   // E-22: reservoirs change with the plant on emissive + env, bitwise identical with analytic lights only
      withGpuLockSync(`gate-m5-${safe(g.id)}`, () => {
        runStep(`${g.id} activity (Gate 0): ${g.test}`, 'npx', ['vitest', 'run', ...vitestConfigArgs(), '--project', 'chrome', '--reporter=verbose',
          'validation/gpu-tests/restir-temporal.gpu.test.ts', '-t', g.test.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')], (l) => /Tests |FAIL|✗|×|U8-/.test(l));
      });
    }
    uTr1(dir, add);
    if (existsSync(path.join(ROOT, 'validation/harness/m5-app-smoke.ts'))) {
      runStep('M5 app smoke (interactive temporal, animated camera/lights/env, HUD, map swap reset, pause)', 'npx', ['tsx', 'validation/harness/m5-app-smoke.ts', '--run', `${runId}-app-smoke`], (l) => /^(PASS|FAIL)\s/.test(l));
    } else add('M5 app smoke (validation/harness/m5-app-smoke.ts, T-D)', false, 0, undefined, 'missing');
  }

  // ---- unit list ------------------------------------------------------------------------------------------------------
  const units: UnitPlan[] = [];
  const mk = (u: Omit<UnitPlan, 'R' | 'chainSeconds' | 'msPerChain' | 'cap' | 'notes'>): UnitPlan => ({ ...u, R: 0, chainSeconds: 0, msPerChain: 0, cap: 'ok', notes: [] });
  if (doStatic) {
    for (const s of M5_STATIC.filter((x) => sel(x.pkg))) {
      units.push(mk({ id: `${s.pkg}@3.3`, kind: 'static', pkg: s.pkg, rung: '3.3', preset: 'temporal', extra: [], frames: STATIC.T, testFrames: [...STATIC.testFrames], tile: 32, stage: 'B' }));
      units.push(mk({ id: `${s.pkg}@3.4`, kind: 'static', pkg: s.pkg, rung: '3.4', preset: 'full', extra: [], frames: STATIC.T, testFrames: [...STATIC.testFrames], tile: 32, stage: 'B' }));
      if (M5_AVG_SCENES.includes(s.pkg)) units.push(mk({ id: `${s.pkg}@3.5`, kind: 'avg', pkg: s.pkg, rung: '3.5', preset: 'full', extra: [], frames: AVG.T, testFrames: [], average: { from: AVG.from, to: AVG.to }, tile: 32, stage: 'B' }));
    }
    for (const pkg of U8_SCENES.filter(sel)) {
      units.push(mk({ id: `${pkg}@3.3`, kind: 'u8', pkg, rung: '3.3', preset: 'temporal', extra: [], frames: STATIC.T, testFrames: [...STATIC.testFrames], tile: 32, stage: 'B' }));
      units.push(mk({ id: `${pkg}@3.4`, kind: 'u8', pkg, rung: '3.4', preset: 'full', extra: [], frames: STATIC.T, testFrames: [...STATIC.testFrames], tile: 32, stage: 'B' }));
    }
  }
  if (doCore && !doStatic && !o.only) {
    // the A/A pair is sized from its unit (m5s_cornell_i 3.4), which belongs to the static part: size it, never run it here
    units.push(mk({ id: `${AA_PKG}@3.4`, kind: 'static', pkg: AA_PKG, rung: '3.4', preset: 'full', extra: [], frames: STATIC.T, testFrames: [...STATIC.testFrames], tile: 32, stage: 'B', sizeOnly: true }));
  }
  if (doCore) {
    for (const d of DYN_UNITS.filter((x) => sel(x.pkg))) {
      const s = M5_SEQUENCES.find((x) => x.pkg === d.pkg)!;
      units.push(mk({ id: d.id, kind: 'dyn', pkg: d.pkg, rung: '3.6', preset: 'full', variant: d.variant, extra: d.args, frames: s.T, testFrames: s.testFrames, tile: 64, stage: 'dyn' }));
    }
  }

  // ---- masks (dyn units) + pilots + sizing -------------------------------------------------------------------------------
  const maskDirs = new Map<string, string>();
  const dynPkgs = [...new Set(units.filter((u) => u.kind === 'dyn').map((u) => u.pkg))];
  prefetch('mask references + disocclusion flags', dir, add, (a) => { for (const pkg of dynPkgs) buildMasks(pkg, M5_SEQUENCES.find((x) => x.pkg === pkg)!.testFrames, a); });
  for (const pkg of dynPkgs) {
    const s = M5_SEQUENCES.find((x) => x.pkg === pkg)!;
    const m = buildMasks(pkg, s.testFrames, add);
    if (m) maskDirs.set(pkg, m.dir);
  }
  for (const u of units) if (u.kind === 'dyn') u.masks = maskDirs.get(u.pkg);
  const ptPlans = new Map<string, PtPlan>();
  const pilotDirs: Record<string, string> = {};
  prefetch('pilots', dir, add, (a) => sizeAll(units.map((u) => ({ ...u, notes: [...u.notes] })), new Map(), {}, a));
  sizeAll(units, ptPlans, pilotDirs, add);
  // + a per-invocation overhead (Vite + Chrome start, pipeline compiles, chunking probe): one per lock chunk
  const runUnits = units.filter((u) => !u.sizeOnly);
  const invocations = runUnits.reduce((a, u) => a + chainChunks(u.R || E_MEMBERS, E_MEMBERS, u.chainSeconds * 1.2).length, 0) + [...ptPlans.values()].reduce((a, p) => a + Math.ceil(p.B / chunkBatches(p.B, p.seconds * 1.2)), 0);
  const planHours = (runUnits.reduce((a, u) => a + u.chainSeconds, 0) + [...ptPlans.values()].reduce((a, p) => a + p.seconds, 0) + INVOCATION_OVERHEAD_S * invocations) / 3600;
  const extra = planExtras(units, doCore, doStatic);
  const sizing = { units: units.map((u) => ({ id: u.id, R: u.R, frames: u.frames, testFrames: u.testFrames, tile: u.tile, minutes: r4(u.chainSeconds / 60), msPerChain: r4(u.msPerChain), cap: u.cap, notes: u.notes })),
    pt: [...ptPlans.values()].map((p) => ({ ...p, minutes: r4(p.seconds / 60) })), planHours: r4(planHours), extras: extra, totalHours: r4(planHours + extra.hours),
    split: planHours + extra.hours > SPLIT_HOURS, pilots: pilotDirs };
  writeFileSync(path.join(ROOT, dir, 'sizing.json'), `${JSON.stringify(sizing, null, 1)}\n`);
  add(`M5 plan: ${units.length} units, ${r4(planHours)} h of chains + PT references, + ${r4(extra.hours)} h (Gate 0, plants, A/A, U8 M4 rungs) = ${r4(planHours + extra.hours)} h${planHours + extra.hours > SPLIT_HOURS ? ` > ${SPLIT_HOURS} h: run as --part core + --part static (Q3)` : ''}`,
    !units.some((u) => u.cap === 'infeasible'), 0, sizing, units.filter((u) => u.cap === 'infeasible').map((u) => `${u.id} infeasible (Q4: escalate)`).join('; ') || undefined);
  const budget = budgetRows(units, ptPlans);
  writeFileSync(path.join(ROOT, dir, 'budget-m5.json'), `${JSON.stringify(budget, null, 1)}\n`);
  if (o.writeBudget) mergeBudget(budget, runId, planHours + extra.hours, add);
  if (full && doCore) budgetRowsPresent(add);

  const results: Record<string, any>[] = [];
  if (o.prerenderPtRefs) {
    for (const p of ptPlans.values()) ptRef(p.pkg, p.frame, p.spp, p.B, SEEDS.pt, add, p.seconds * 1.2);
  }
  if (!o.pilotOnly) {
    // every PT reference that fits one GPU-lock hold, grouped (the larger ones run chunked when their unit needs them)
    prefetch('PT references', dir, add, (a) => { for (const p of ptPlans.values()) ptRef(p.pkg, p.frame, p.spp, p.B, SEEDS.pt, a, p.seconds * 1.2); });
  }
  if (!o.pilotOnly && !o.prerenderPtRefs) {
    // ---- static ladders (3.3 → 3.4 → 3.5) and U8 chains ----------------------------------------------------------------------
    const stopped = new Map<string, string>();
    for (const u of units.filter((x) => x.kind !== 'dyn' && !x.sizeOnly && !plantsOnly)) {
      if (stopped.has(u.pkg)) { results.push(...u.testFrames.map((t) => ({ unit: `${u.id}-f${t}`, kind: u.kind, rung: u.rung, status: 'not run', ok: false, note: `ladder stopped at ${stopped.get(u.pkg)}` }))); continue; }
      const r = runUnit(u, ptPlans, dir, runId, nU, add);
      results.push(...r);
      if (r.some((x) => !x.ok)) stopped.set(u.pkg, u.rung);
    }
    if (doStatic && !plantsOnly) for (const pkg of U8_SCENES.filter(sel)) results.push(...u8M4Rungs(pkg, ptPlans, dir, runId, nU, add));
    // ---- dynamic units ---------------------------------------------------------------------------------------------------
    for (const u of units.filter((x) => x.kind === 'dyn' && !plantsOnly)) results.push(...runUnit(u, ptPlans, dir, runId, nU, add));
    // ---- plants, synthetic W × 1.003, A/A -----------------------------------------------------------------------------------
    if ((full && doCore) || plantsOnly) {
      for (const p of M5_PLANTS.filter((x) => !o.plants || o.plants.has(x.id))) results.push(runPlant(p, units, ptPlans, dir, runId, nU, add));
      if (!o.plants || o.plants.has('aa')) results.push(...aaAndSynthetic(units, ptPlans, dir, runId, nU, add));
    }
  }
  results.push(...M5_GATE0_PLANTS.map((p) => ({ unit: `plant-${p.id}`, kind: 'plant-gate0', status: p.status, ok: true, note: `${p.name}: not a Gate-3 control (E-22)` })));
  results.push(...M5_DEFERRED_PLANTS.map((p) => ({ unit: `plant-${p.id}`, kind: 'plant-deferred', status: 'deferred to M6', ok: true, note: `${p.name}: ${p.why}` })));
  results.push(...M5_REPORT_ONLY.map((pkg) => ({ unit: `${pkg}@3.3-3.5`, kind: 'report-only', status: 'not run', ok: true, note: 'reported, not gating (Q2); gated again by M8 validate --all' })));

  // ---- summary --------------------------------------------------------------------------------------------------------
  const failed = steps.filter((x) => !x.ok).map((x) => x.name);
  const byKind = Object.fromEntries(['static', 'avg', 'dyn', 'u8', 'u8-m4', 'plant', 'aa'].map((k) => {
    const u = results.filter((x) => x.kind === k);
    return [k, { pass: u.filter((x) => x.ok).length, fail: u.filter((x) => !x.ok && x.status !== 'not run').length, notRun: u.filter((x) => x.status === 'not run').length }];
  }));
  const summary = {
    milestone: 'M5', gate: 'Gate 0 + Gate 3 rungs 3.3-3.6 + plants with predicted signs + A/A + U8 ladder (restir-temporal-api.md §6)',
    run: runId, created: new Date().toISOString(), ok: failed.length === 0, part: o.part ?? 'all', subset: o.only ? [...o.only] : undefined,
    total_s: Math.round((performance.now() - t0) / 100) / 10, n_units: nU, code_hashes: hashes, plan_hours: r4(planHours), per_kind: byKind, failed, units: results, steps,
  };
  writeFileSync(path.join(ROOT, dir, 'summary.json'), `${JSON.stringify(summary, null, 1)}\n`);
  writeFileSync(path.join(ROOT, dir, 'summary.md'), summaryMd(summary, results, sizing));
  console.log(`\n${table(results)}\nsummary: ${dir}/summary.json (${summary.total_s} s)`);
}

function ensurePackages(add: Add): void {
  const need = [...M5_STATIC.map((s) => s.pkg), ...M5_SEQUENCES.map((s) => s.pkg), ...U8_SCENES].filter((p) => !existsSync(path.join(ROOT, pkgDir(p), 'scene.json')));
  if (!need.length) { add('M5 packages present (make-m5.ts)', true, 0); return; }
  const r = sh('npx', ['tsx', 'validation/scenes/make-m5.ts', '--only', need.join(',')], () => false);
  const still = need.filter((p) => !existsSync(path.join(ROOT, pkgDir(p), 'scene.json')));
  add(`M5 packages generated (make-m5.ts --only ${need.join(',')})`, r.code === 0 && !still.length, r.seconds, { missing: still }, still.length ? `missing ${still.join(', ')}` : undefined);
}

function packagesDeterministic(dir: string, add: Add): void {
  const a = path.join(dir, 'make-m5-a'), b = path.join(dir, 'make-m5-b');
  const t0 = performance.now();
  const ra = sh('npx', ['tsx', 'validation/scenes/make-m5.ts', a], () => false), rb = sh('npx', ['tsx', 'validation/scenes/make-m5.ts', b], () => false);
  const diffs: string[] = [];
  if (ra.code || rb.code) diffs.push(`exit ${ra.code}/${rb.code}: ${(ra.out + rb.out).slice(-300)}`);
  const files = (d: string) => (existsSync(path.join(ROOT, d)) ? listDir(d).map((f) => path.relative(d, f)) : []);
  const fa = files(a), fb = files(b);
  for (const f of new Set([...fa, ...fb])) if (!fa.includes(f) || !fb.includes(f) || !readFileSync(path.join(ROOT, a, f)).equals(readFileSync(path.join(ROOT, b, f)))) diffs.push(f);
  // the committed packages must be what the generator writes (no stale package in validation/scenes)
  for (const f of fa) {
    const pkg = f.split(path.sep)[0];
    if (isEnvPkg(pkg)) continue;
    const committed = path.join('validation/scenes', f);
    if (!existsSync(path.join(ROOT, committed)) || !readFileSync(path.join(ROOT, committed)).equals(readFileSync(path.join(ROOT, a, f)))) diffs.push(`stale ${committed}`);
  }
  rmSync(path.join(ROOT, a), { recursive: true, force: true });
  rmSync(path.join(ROOT, b), { recursive: true, force: true });
  add('make-m5.ts package determinism (twice, byte-identical; committed packages current)', diffs.length === 0, (performance.now() - t0) / 1000, { files: fa.length, diffs: diffs.slice(0, 20) },
    diffs.length ? `diffs: ${diffs.slice(0, 6).join(', ')}` : `${fa.length} files`);
}

/** U-TR-1 (§6.1): two chain runs with the same seed are bitwise identical (ensemble rows, counters) — 16 chains of
 *  ixs_e_addremove frames 0…15 (light add / intensity step: refresh frames), preset full. */
function uTr1(dir: string, add: Add): void {
  const t0 = performance.now();
  const runs = [0, 1].map((i) => chainRun({ pkg: 'ixs_e_addremove_256', preset: 'full', R: E_MEMBERS, seed: SEEDS.chains, frames: 16, testFrames: [8, 15] }, path.join(dir, 'u-tr-1', `run${i}`), 60));
  const diffs: string[] = [];
  if (runs.some((r) => !r.dir)) diffs.push('run failed');
  else for (const f of ['f8', 'f15']) {
    const a = readFileSync(path.join(ROOT, runs[0].dir!, f, 'ensemble.npz')), b = readFileSync(path.join(ROOT, runs[1].dir!, f, 'ensemble.npz'));
    if (!a.equals(b)) diffs.push(`${f}/ensemble.npz differs`);
  }
  if (runs.every((r) => r.meta) && stableJson(runs[0].meta!.restir?.counters) !== stableJson(runs[1].meta!.restir?.counters)) diffs.push('RSC counters differ');
  add('U-TR-1: two chain runs bitwise identical (ensemble rows, counters)', diffs.length === 0, (performance.now() - t0) / 1000, { diffs }, diffs.join('; ') || undefined);
}

// ---- sizing of every unit ---------------------------------------------------------------------------------------------------

function ptPilotSide(pkg: string, frame: number | undefined, add: Add): { ms: number; dir?: string } {
  const r = ptRef(pkg, frame, PILOT_PT_SPP, B_PT, SEEDS.ptPilot, add, 60, 'pilot');
  if (!r) return { ms: 0 };
  const n = readdirSync(path.join(ROOT, r.dir)).filter((f) => /^batch_\d{3}\.pfm$/.test(f)).length;
  return { dir: r.dir, ms: (r.meta.timings.batchMs as number[]).reduce((a, b) => a + b, 0) / (n * PILOT_PT_SPP) };
}

function sizeAll(units: UnitPlan[], ptPlans: Map<string, PtPlan>, pilotDirs: Record<string, string>, add: Add): void {
  // PT-sharing groups: static/avg/u8 units of one package share the base-state reference; dyn units of one package share the
  // per-frame references.
  const groups = new Map<string, UnitPlan[]>();
  for (const u of units) { const k = `${u.kind === 'dyn' ? 'dyn' : 'static'}:${u.pkg}`; groups.set(k, [...(groups.get(k) ?? []), u]); }
  for (const [key, us] of groups) {
    const pkg = us[0].pkg, dyn = key.startsWith('dyn');
    const frames = dyn ? M5_SEQUENCES.find((s) => s.pkg === pkg)!.testFrames : [-1];
    const d = dyn ? DELTA.dyn : DELTA.B;
    const tryTile = (tile: number) => {
      const ptSides = new Map<number, { side: AggSide; ms: number }>();
      for (const f of frames) {
        const pf = f < 0 ? undefined : f;
        const masks = dyn && us[0].masks ? { dir: us[0].masks!, t: f } : undefined;
        const s = ptPilotSideTile(pkg, pf, tile, add, masks, pilotDirs);
        if (!s) { if (collect) continue; return undefined; }
        ptSides.set(f, s);
      }
      const variants: { id: string; u: Map<number, Float64Array>; msPerChain: number; rFloor: number }[] = [];
      const gvariants: typeof variants = [];
      for (const u of us) {
        const pilot = pilotChains({ pkg, preset: u.preset, R: PILOT_CHAINS, seed: SEEDS.chainPilot, frames: u.frames, testFrames: u.average ? [] : u.testFrames, average: u.average, masks: u.masks, extra: u.extra }, add);
        if (!pilot || collect) { if (collect) continue; return undefined; }
        pilotDirs[u.id] = pilot.dir;
        const msPerChain = (pilot.meta.timings.batchMs as number[]).reduce((a2, b2) => a2 + b2, 0) / PILOT_CHAINS;   // GPU batches only (setup and the chunking probe are per invocation)
        const um = new Map<number, Float64Array>(), ug = new Map<number, Float64Array>();
        for (const f of frames) {
          const sub = u.average ? 'avg' : `f${f < 0 ? u.testFrames[0] : f}`;
          const fs = f < 0 ? (u.average ? ['avg'] : u.testFrames.map((t) => `f${t}`)) : [sub];
          for (const s of fs) {
            const z = readNpz(new Uint8Array(readFileSync(path.join(ROOT, pilot.dir, s, 'ensemble.npz')))) as unknown as Map<string, { shape: number[]; data: unknown }>;
            const side = aggFromNpz(z, tile);
            const uk = uVec(side, ptSides.get(f)!.side, d);
            const prev = um.get(f);
            um.set(f, prev ? prev.map((v, i) => Math.max(v, uk[i])) : uk);
            const g = globalOnly(uk, side);
            const pg = ug.get(f);
            ug.set(f, pg ? pg.map((v, i) => Math.max(v, g[i])) : g);
          }
        }
        variants.push({ id: u.id, u: um, msPerChain, rFloor: u.average ? R_FLOOR_AVG : R_FLOOR });
        gvariants.push({ id: u.id, u: ug, msPerChain, rFloor: u.average ? R_FLOOR_AVG : R_FLOOR });
      }
      if (collect) return undefined;
      const frozen = frames.map((f) => ({ f, z: tryJson(ptSizeFile(pkg, f < 0 ? undefined : f)) as { spp: number } | undefined }));
      const ptIn = frames.map((f, j) => ({ frame: f, u: uVec(ptSides.get(f)!.side, ptSides.get(f)!.side, d), msPerSample: ptSides.get(f)!.ms, fixedSpp: frozen[j].z?.spp }));
      const z = sizeGroup(ptIn, variants);
      const zg = sizeGroup(ptIn.map((p) => ({ ...p, u: globalOnly(p.u, ptSides.get(p.frame)!.side) })), gvariants);
      return { z, zg };
    };
    const t0 = performance.now();
    let tile = dyn ? 64 : 32;
    let res = tryTile(tile);
    if (!res) { add(`sizing ${key}`, false, 0, undefined, 'pilot failed'); continue; }
    const dec = (id: string) => capDecision(res!.z.chains[id].seconds, res!.zg.chains[id].seconds, tile !== (dyn ? 64 : 32));
    const res0 = res;
    if (us.some((u) => dec(u.id).status === 'enlarge')) {
      tile *= 2;
      const r2 = tryTile(tile);
      if (r2) res = r2;
      for (const u of us) u.notes.push(`tile aggregate enlarged to ${tile}² (unit cap ${UNIT_CAP_S / 60} min; at ${tile / 2}²: R ${res0.z.chains[u.id].R}, ${(res0.z.chains[u.id].seconds / 60).toFixed(0)} min)`);
    }
    for (const u of us) {
      const c = res.z.chains[u.id], dd = dec(u.id);
      u.R = c.R; u.chainSeconds = c.seconds; u.msPerChain = c.msPerChain; u.tile = tile;
      u.cap = dd.status === 'enlarge' ? 'ok' : dd.status;
      const gmin = (res.zg.chains[u.id].seconds / 60).toFixed(1);
      if (dd.status === 'raised') u.notes.push(`${(c.seconds / 60).toFixed(1)} min > ${UNIT_CAP_S / 60} min after the tile enlargement (global aggregate alone: ${gmin} min): cap raised once to ${UNIT_CAP_RAISED_S / 60} min (Q4)`);
      if (dd.status === 'infeasible') u.notes.push(`infeasible within ${UNIT_CAP_RAISED_S / 60} min: escalated to the coordinator (Q4)`);
    }
    for (const [f, p] of Object.entries(res.z.pt)) {
      const frame = Number(f) < 0 ? undefined : Number(f);
      ptPlans.set(`${pkg}:${frame ?? 'base'}`, { pkg, frame, spp: p.spp, B: B_PT, seconds: p.seconds });
      const sf = ptSizeFile(pkg, frame);
      if (!p.fixed) { mkdirSync(path.dirname(path.join(ROOT, sf)), { recursive: true }); writeFileSync(path.join(ROOT, sf), `${JSON.stringify({ pkg, frame: frame ?? null, spp: p.spp, B: B_PT, created: new Date().toISOString() }, null, 1)}\n`); }
    }
    add(`sizing ${key} (tile ${tile}², x${SIZING_MARGIN})`, true, (performance.now() - t0) / 1000, res.z,
      us.map((u) => `${u.id}: R ${u.R} (${(u.chainSeconds / 60).toFixed(1)} min${u.cap !== 'ok' ? `, ${u.cap}` : ''})`).join(', ') + `; PT ${Object.entries(res.z.pt).map(([f, p]) => `${Number(f) < 0 ? 'base' : `f${f}`}: ${p.spp}x${B_PT} (${(p.seconds / 60).toFixed(1)} min)`).join(' ')}`);
  }
}

function ptPilotSideTile(pkg: string, frame: number | undefined, tile: number, add: Add, masks: { dir: string; t: number } | undefined, pilotDirs: Record<string, string>): { side: AggSide; ms: number } | undefined {
  const r = ptPilotSide(pkg, frame, add);
  if (!r.dir) return undefined;
  pilotDirs[`pt:${pkg}:${frame ?? 'base'}`] = r.dir;
  const files = readdirSync(path.join(ROOT, r.dir)).filter((n) => /^batch_\d{3}\.pfm$/.test(n)).sort();
  const imgs = files.map((f) => decodePFM(new Uint8Array(readFileSync(path.join(ROOT, r.dir!, f)))));
  const mm = masks ? masksOf(masks.dir, masks.t) : undefined;
  return { side: aggFromImages(imgs.map((i) => i.data), imgs[0].width, imgs[0].height, tile, mm?.perMask ?? [], PILOT_PT_SPP), ms: r.ms };
}

const ptSizeFile = (pkg: string, frame: number | undefined) => path.join(M5_OUT, 'ptsize', `${pkg}-f${frame ?? 'base'}-${codeHashes().pt.slice(0, 16)}-${packageHash(pkgDir(pkg)).slice(0, 16)}.json`);

/** Plan estimate of the parts that are not sized units: Gate 0, plants (their base unit's chains for the frames up to
 *  the last predicted one; plant-only packages at PLANT_ONLY_R with the median ms per member-frame), A/A (2 × 4× the
 *  m5s_cornell_i 3.4 chains), U8 M4 rungs (≈ 5 min per scene). Hours. */
export function planExtras(units: { id: string; pkg: string; rung: string; kind: string; variant?: string; frames: number; msPerChain: number; chainSeconds: number }[], core = true, stat = true): { gate0: number; plants: number; aa: number; u8m4: number; hours: number } {
  const rates = units.filter((u) => u.msPerChain > 0).map((u) => u.msPerChain / u.frames).sort((a, b) => a - b);
  const msPerMemberFrame = rates.length ? rates[Math.floor(rates.length / 2)] : 3;
  let plants = 0;
  for (const p of M5_PLANTS) {
    const last = Math.max(...p.predict.map((x) => x.frame)) + 1;
    const base = units.find((u) => u.pkg === p.pkg && (p.rung === '3.4' ? u.rung === '3.4' : u.kind === 'dyn' && u.variant === 'base'));
    plants += base ? base.chainSeconds * Math.min(1, last / base.frames) : (PLANT_ONLY_R * last * msPerMemberFrame) / 1000;
  }
  const aaU = units.find((u) => u.pkg === AA_PKG && u.rung === '3.4');
  const aa = aaU ? 2 * CALIB_FACTOR * aaU.chainSeconds : 0;
  const gate0 = GATE0_EST_H * 3600, u8m4 = U8_SCENES.length * 300;
  const hours = ((core ? gate0 + plants + aa : 0) + (stat ? u8m4 : 0)) / 3600;
  return { gate0: r4(gate0 / 3600) as number, plants: r4(plants / 3600) as number, aa: r4(aa / 3600) as number, u8m4: r4(u8m4 / 3600) as number, hours };
}

/** --reuse-chains DIR[,DIR…]: copy an earlier gate's first-seed chain run of the same unit / plant into `dest` when it is
 *  the same run (R, seed, frames, package, plant / variant flags) and error-free (meta ok). Returns the source or undefined. */
function reusedChains(sub: 'chains' | 'plants', id: string, dest: string, want: { R: number; seed: number; frames: number; pkg: string; extra?: string[]; testFrames?: number[] }): string | undefined {
  for (const d of (reuseChainsDir ?? '').split(',').filter(Boolean)) {
    const prev = path.join(d, sub, safe(id));
    const pm = tryJson(path.join(prev, 'meta.json'));
    if (!pm?.ok || pm.chains !== want.R || pm.seed !== want.seed || pm.config?.frames !== want.frames || !String(pm.scene?.url ?? '').includes(want.pkg)) continue;
    const ex = want.extra ?? [];
    const flag = (k: string) => { const i = ex.indexOf(k); return i >= 0 ? ex[i + 1] : undefined; };
    const pl = pm.config?.plants ?? {}, st = pm.config?.settings ?? {};
    if ((flag('--tplant') ?? '') !== (pl.temporal ?? []).join(',') || (flag('--u8-plant') ?? '') !== (pl.u8 ?? []).join(',')) continue;
    if ((flag('--temporal-mis') ?? 'contribution') !== st.temporalMis || (flag('--refresh') ?? 'exact') !== st.refresh || Number(flag('--boost') ?? 0) !== (st.boostSlots ?? 0)) continue;
    if (want.testFrames && want.testFrames.join(',') !== (pm.config?.testFrames ?? []).join(',')) continue;
    rmSync(path.join(ROOT, dest), { recursive: true, force: true });
    mkdirSync(path.join(ROOT, dest), { recursive: true });
    cpSync(path.join(ROOT, prev), path.join(ROOT, dest), { recursive: true });
    return prev;
  }
  return undefined;
}

// ---- running and comparing a unit -------------------------------------------------------------------------------------------

function runUnit(u: UnitPlan, ptPlans: Map<string, PtPlan>, dir: string, runId: string, nU: number, add: Add): Record<string, any>[] {
  const t0 = performance.now();
  const out: Record<string, any>[] = [];
  if (!u.R) return [{ unit: u.id, kind: u.kind, rung: u.rung, status: 'not run', ok: false, note: 'no sizing' }];
  if (u.cap === 'infeasible') return [{ unit: u.id, kind: u.kind, rung: u.rung, status: 'infeasible', ok: false, note: u.notes.join('; ') }];
  const frames = u.average ? ['avg'] : u.testFrames.map((t) => `f${t}`);
  const refs = new Map<string, Run>();
  for (const f of frames) {
    const t = f === 'avg' || u.kind !== 'dyn' ? undefined : Number(f.slice(1));
    const p = ptPlans.get(`${u.pkg}:${t ?? 'base'}`);
    const r = p && ptRef(u.pkg, t, p.spp, p.B, SEEDS.pt, add, p.seconds * 1.2);
    if (!r) return [{ unit: u.id, kind: u.kind, rung: u.rung, status: 'not run', ok: false, note: 'PT reference failed' }];
    refs.set(f, r);
  }
  const rounds = u.preset === 'full' ? 1 : 0;
  const args: ChainArgs = { pkg: u.pkg, preset: u.preset, R: u.R, seed: SEEDS.chains, frames: u.frames, testFrames: u.average ? [] : u.testFrames, average: u.average, masks: u.masks, extra: u.extra };
  console.log(`\n--- chains ${u.id}: R ${u.R}, ${u.frames} frames`);
  const dest = path.join(dir, 'chains', safe(u.id));
  const prev = reusedChains('chains', u.id, dest, { R: u.R, seed: SEEDS.chains, frames: u.frames, pkg: u.pkg, extra: u.extra });
  if (prev) u.notes.push(`chains reused from ${prev} (harness-only re-evaluation)`);
  const run = prev ? { dir: dest, meta: tryJson(path.join(dest, 'meta.json')), code: 0, out: '', seconds: 0 } : chainRun(args, dest, u.chainSeconds * 1.2);
  const test = (f: string) => writeTest(dir, `${u.id}-${f}`, nU, u.stage, u.tile, u.masks && f !== 'avg' ? { masks: masksOf(u.masks, Number(f.slice(1))).test } : {});
  let seq: Record<string, any> | undefined;
  for (const f of frames) {
    const unit = `${u.id}-${f}`;
    const ref = refs.get(f)!;
    const cdir = run.dir && path.join(run.dir, f);
    const cm = cdir ? tryJson(path.join(cdir, 'meta.json')) : undefined;
    const t16 = cm ? t16ChainProblems(cm, ref.meta, { rounds, frame: f === 'avg' ? 'avg' : Number(f.slice(1)), staticScene: u.kind !== 'dyn' }) : ['chain run produced no meta.json'];
    const cout = path.join(dir, 'compare', safe(unit));
    let rep: Record<string, any> | undefined;
    if (cdir && cm) { compare(cdir, ref.dir, test(f), cout); rep = tryJson(path.join(cout, 'report.json')); }
    let rerun: Record<string, unknown> | undefined;
    if (rep?.status === 'rerun_required') {
      // confirmatory re-run of THIS test frame's unit on disjoint seeds (both sides)
      const t = f === 'avg' || u.kind !== 'dyn' ? undefined : Number(f.slice(1));
      const p = ptPlans.get(`${u.pkg}:${t ?? 'base'}`)!;
      const ref2 = ptRef(u.pkg, t, p.spp, p.B, SEEDS.ptRerun, add, p.seconds * 1.2);
      const run2 = chainRun({ ...args, seed: SEEDS.chainsRerun, testFrames: u.average ? [] : [Number(f.slice(1))], frames: u.average ? u.frames : Number(f.slice(1)) + 1 }, path.join(dir, 'chains', safe(`${unit}-rerun`)), u.chainSeconds * 1.2);
      if (ref2 && run2.dir) {
        const c2 = path.join(run2.dir, f);
        const cm2 = tryJson(path.join(c2, 'meta.json'));
        if (cm2) t16.push(...t16ChainProblems(cm2, ref2.meta, { rounds, frame: f === 'avg' ? 'avg' : Number(f.slice(1)), staticScene: u.kind !== 'dyn' }).map((x) => `re-run: ${x}`));
        compare(c2, ref2.dir, test(f), `${cout}-rerun`, path.join(cout, 'report.json'));
        rerun = { first: summarize(rep), report: path.join(`${cout}-rerun`, 'report.json') };
        rep = tryJson(path.join(`${cout}-rerun`, 'report.json'));
      }
    }
    const sum = rep ? summarize(rep) : undefined;
    const ok = !!rep && (rep.status === 'pass' || rep.status === 'pass_on_rerun') && t16.length === 0;
    const res = { unit, kind: u.kind, scene: u.pkg, rung: u.rung, variant: u.variant, frame: f, ok, ...(sum ?? { status: 'error' }), R: u.R, pt: `${ptPlans.get(`${u.pkg}:${f === 'avg' || u.kind !== 'dyn' ? 'base' : f.slice(1)}`)?.spp}x${B_PT}`, tile: u.tile,
      aggregate_enlarged: u.tile !== (u.stage === 'B' ? 32 : 64), cap: u.cap, notes: u.notes, t16, chain_minutes: run.meta ? r4(run.meta.timings.totalMs / 60000) : undefined, report: path.join(cout, 'report.json'), ...(rerun ? { rerun } : {}) };
    out.push(res);
    add(`${u.kind === 'dyn' ? 'dyn' : 'Stage B'} ${unit} (rung ${u.rung}${u.variant && u.variant !== 'base' ? ` ${u.variant}` : ''})`, ok, (performance.now() - t0) / 1000, res,
      sum ? `${sum.status}: Δ_Y ${pct(sum.global_rel_Y as number, 4)} (MDB ${pct(sum.mdb_global_Y as number, 3)}), worst tile ${pct(sum.worst_tile_rel_Y as number, 2)}, mult ${sum.multiplier_needed}${t16.length ? `; T16: ${t16.slice(0, 3).join('; ')}` : ''}` : `compare failed${t16.length ? `; T16: ${t16.slice(0, 3).join('; ')}` : ''}`);
  }
  if (u.kind === 'dyn' && run.dir) {
    const spec = path.join(dir, 'tests', `seq-${safe(u.id)}.json`);
    writeFileSync(path.join(ROOT, spec), JSON.stringify({ test: path.join(ROOT, test(frames[0])), n_units: nU,
      frames: u.testFrames.map((t) => ({ frame: t, ours: path.join(ROOT, run.dir!, `f${t}`), ref: path.join(ROOT, refs.get(`f${t}`)!.dir), test: path.join(ROOT, test(`f${t}`)) })) }, null, 1));
    const r = sh(PY, ['validation/tools/dynamic.py', 'sequence', '--spec', spec, '--out', path.join(dir, 'dynamic', safe(u.id))], () => false);
    seq = tryJson(path.join(dir, 'dynamic', safe(u.id), 'dynamic.json'));
    const res = { unit: `${u.id}-sequence`, kind: 'dyn', scene: u.pkg, rung: '3.6', variant: u.variant, ok: !!seq?.ok, status: seq ? (seq.ok ? 'pass' : 'fail') : `error (exit ${r.code})`,
      drift_z_Y: r4(seq?.channels?.Y?.drift?.z), drift_ok: seq?.drift_ok, tiles_ok: seq?.tiles_ok, series_z_Y: seq?.channels?.Y?.series_z?.map(r4) };
    out.push(res);
    add(`dyn sequence ${u.id}: drift + failing-tile binomial (dynamic.py)`, res.ok, r.seconds, res, `drift z_Y ${res.drift_z_Y}, tiles ${seq?.tiles_ok ? 'ok' : 'FAIL'}`);
  }
  return out;
}

function pilotSideOf(dir: string, meta: Record<string, any>): PilotSide {
  const files = readdirSync(path.join(ROOT, dir)).filter((n) => /^batch_\d{3}\.pfm$/.test(n)).sort();
  const imgs = files.map((f) => decodePFM(new Uint8Array(readFileSync(path.join(ROOT, dir, f)))));
  const n = meta.sppPerBatch as number, B = files.length;
  return { B, n, msPerSample: (meta.timings.batchMs as number[]).reduce((a, b) => a + b, 0) / (B * n), W: imgs[0].width, H: imgs[0].height, batches: imgs.map((i) => i.data) };
}

/** A sequential M4 ReSTIR run (run-batches --kernel restir), cached (pilots) or into `dest`. */
function restirSeq(pkg: string, preset: string, frames: number, B: number, seed: number, dest: string, add: Add, step: string, cache: boolean): Run | undefined {
  const keyObj = { kind: 'restir-seq', pkg, packageHash: packageHash(pkgDir(pkg)), preset, frames, B, seed, code: codeHashes().chains };
  const key = sha(stableJson(keyObj)).slice(0, 16);
  const d = cache ? path.join(PILOTS, `${pkg}-seq-${preset}-${key}`) : dest;
  const meta0 = cache ? tryJson(path.join(d, 'meta.json')) : undefined;
  if (meta0?.ok) { add(step, true, 0, { dir: d, cache_hit: true }, 'cache hit'); return { dir: d, meta: meta0, seconds: 0, cacheHit: true }; }
  const run = `m5seq-${key}-${stamp()}`;
  const r = sh('npx', ['tsx', 'validation/harness/run-batches.ts', '--package', pkgDir(pkg), '--kernel', 'restir', '--preset', preset, '--spp', String(frames), '--batches', String(B), '--seed', String(seed), '--run', run], rbEcho);
  rmSync(path.join(ROOT, d), { recursive: true, force: true });
  mkdirSync(path.dirname(path.join(ROOT, d)), { recursive: true });
  if (existsSync(path.join(ROOT, 'validation/out', run))) renameSync(path.join(ROOT, 'validation/out', run), path.join(ROOT, d));
  const meta = tryJson(path.join(d, 'meta.json'));
  add(step, !!meta?.ok, r.seconds, { dir: d }, meta?.ok ? `${(meta.timings.totalMs / 1000).toFixed(1)} s` : r.out.slice(-300));
  return meta ? { dir: d, meta, seconds: r.seconds } : undefined;
}

/** U8 ladder rungs 3.1 / 3.1b / 3.2 (M4 Stage B through the sequential harness), sized by gate-m4's sizeScene around the
 *  u8 package's frozen base-state PT reference (shared with its 3.3 / 3.4 chain units). */
function u8M4Rungs(pkg: string, ptPlans: Map<string, PtPlan>, dir: string, runId: string, nU: number, add: Add): Record<string, any>[] {
  const out: Record<string, any>[] = [];
  const plan = ptPlans.get(`${pkg}:base`);
  const ptPilot = ptRef(pkg, undefined, PILOT_PT_SPP, B_PT, SEEDS.ptPilot, add, 60, 'pilot');
  if (!plan || !ptPilot) return U8_M4_RUNGS.map((r) => ({ unit: `${pkg}@${r.id}`, kind: 'u8-m4', rung: r.id, status: 'not run', ok: false, note: 'no PT sizing / pilot' }));
  const rungs = [];
  for (const r of U8_M4_RUNGS) {
    const f = r.id === '3.2' ? 8 : 128;
    const p = restirSeq(pkg, r.preset, f, B_PT, SEEDS.chainPilot, '', add, `pilot ReSTIR ${pkg} rung ${r.id} (${f} frames x ${B_PT})`, true);
    if (!p) return U8_M4_RUNGS.map((x) => ({ unit: `${pkg}@${x.id}`, kind: 'u8-m4', rung: x.id, status: 'not run', ok: false, note: 'pilot failed' }));
    rungs.push({ id: r.id, side: pilotSideOf(p.dir, p.meta), tile: 32 as const });
  }
  const z = sizeScene(pilotSideOf(ptPilot.dir, ptPilot.meta), rungs, { B: B_PT, fixedPtSpp: plan.spp, capS: UNIT_CAP_S });
  const ref = ptRef(pkg, undefined, plan.spp, B_PT, SEEDS.pt, add, plan.seconds * 1.2);
  if (!ref) return U8_M4_RUNGS.map((r) => ({ unit: `${pkg}@${r.id}`, kind: 'u8-m4', rung: r.id, status: 'not run', ok: false, note: 'PT reference failed' }));
  for (const r of U8_M4_RUNGS) {
    const unit = `${pkg}@${r.id}`;
    const rz = z.rungs[r.id];
    const run = restirSeq(pkg, r.preset, rz.framesPerBatch, B_PT, SEEDS.chains, path.join(dir, 'restir', safe(unit)), add, `U8 ReSTIR ${unit} (${rz.framesPerBatch} frames x ${B_PT})`, false);
    const t16 = run ? t16Problems(run.meta, ref.meta, { rounds: r.rounds }) : ['no meta.json'];
    const cout = path.join(dir, 'compare', safe(unit));
    if (run) compare(run.dir, ref.dir, writeTest(dir, unit, nU, 'B', rz.tile), cout);
    const rep = tryJson(path.join(cout, 'report.json'));
    const sum = rep ? summarize(rep) : undefined;
    const ok = !!rep && rep.status === 'pass' && t16.length === 0;
    const res = { unit, kind: 'u8-m4', scene: pkg, rung: r.id, ok, ...(sum ?? { status: 'error' }), restir: `${rz.framesPerBatch}x${B_PT}`, pt: `${plan.spp}x${B_PT}`, t16 };
    out.push(res);
    add(`U8 ladder ${unit} (${r.preset})`, ok, run?.seconds ?? 0, res, sum ? `${sum.status}: Δ_Y ${pct(sum.global_rel_Y as number, 4)}` : 'no report');
    if (!ok) break;
  }
  void runId;
  return out;
}

// ---- plants -------------------------------------------------------------------------------------------------------------------

function runPlant(p: PlantSpec, units: UnitPlan[], ptPlans: Map<string, PtPlan>, dir: string, runId: string, nU: number, add: Add): Record<string, any> {
  const t0 = performance.now();
  const frames = [...new Set(p.predict.map((x) => x.frame))].sort((a, b) => a - b);
  const seq = M5_SEQUENCES.find((s) => s.pkg === p.pkg);
  const staticScene = !seq;
  const base = units.find((u) => u.pkg === p.pkg && (p.rung === '3.4' ? u.rung === '3.4' : u.kind === 'dyn' && u.variant === 'base'));
  const R = base?.R ?? PLANT_ONLY_R;
  const data: Record<string, any> = { unit: `plant-${p.id}`, kind: 'plant', scene: p.pkg, rung: p.rung, plant: p.name, ok: false };
  // masks: partition (dyn) + dominance regions
  let maskDir: string | undefined;
  if (!staticScene) {
    const m = buildMasks(p.pkg, frames, add, { dominance: p.dominance, sil: true, tag: `plant-${safe(p.id)}` });
    if (!m) { add(`plant ${p.id} masks`, false, 0); return { ...data, status: 'masks failed' }; }
    maskDir = m.dir;
  }
  const refs = new Map<number, Run>();
  for (const t of frames) {
    const plan = ptPlans.get(`${p.pkg}:${staticScene ? 'base' : t}`) ?? { spp: 1024, B: B_PT, seconds: 120 };
    const r = ptRef(p.pkg, staticScene ? undefined : t, plan.spp * CALIB_FACTOR, B_PT, SEEDS.ptCalib + plantSeedOffset, add, plan.seconds * CALIB_FACTOR * 1.2);
    if (!r) return { ...data, status: 'PT reference failed' };
    refs.set(t, r);
  }
  const lastFrame = Math.max(...frames);
  const pdest = path.join(dir, 'plants', safe(p.id));
  const pseed = SEEDS.plantBase + M5_PLANTS.indexOf(p) + plantSeedOffset, pframes = staticScene ? STATIC.T : lastFrame + 1;
  const reused = reusedChains('plants', p.id, pdest, { R, seed: pseed, frames: pframes, pkg: p.pkg, extra: p.args, testFrames: frames });
  const run = reused ? { dir: pdest, meta: tryJson(path.join(pdest, 'meta.json')), code: 0, out: '', seconds: 0 }
    : chainRun({ pkg: p.pkg, preset: 'full', R, seed: pseed, frames: pframes, testFrames: frames, masks: maskDir, extra: p.args }, pdest, (base?.chainSeconds ?? 300) * 1.2);
  if (reused) data.reused_chains = reused;
  if (plantSeedOffset) data.seeds = { chains: pseed, ptCalib: SEEDS.ptCalib + plantSeedOffset, note: 'fresh disjoint seeds (prediction revised after measurement, B-12 / D-5)' };
  if (!run.dir) { add(`plant ${p.id}: ${p.name} on ${p.pkg}`, false, run.seconds, undefined, run.out.slice(-300)); return { ...data, status: 'chain run failed' }; }
  const tests = new Map<number, string>();
  const full: Record<string, any> = {};
  let fullOk = false;
  for (const t of frames) {
    // the plant's 4× PT reference and chains are sized like its unit: compare on the unit's (possibly enlarged) tiles
    const test = writeTest(dir, `plant-${p.id}-f${t}`, nU, staticScene ? 'B' : 'dyn', base?.tile ?? (staticScene ? 32 : 64), maskDir ? { masks: masksOf(maskDir, t).test } : {});
    tests.set(t, test);
    const cout = path.join(dir, 'compare', safe(`plant-${p.id}-f${t}`));
    compare(path.join(run.dir, `f${t}`), refs.get(t)!.dir, test, cout);
    const rep = tryJson(path.join(cout, 'report.json'));
    full[t] = rep && { status: rep.status, global_rel_Y: r4(rep.channels.Y.global_.rel), failed: rep.failed_checks };
    fullOk ||= !!rep && rep.status !== 'pass';
  }
  const specPath = path.join(dir, 'plants', safe(p.id), 'plant.json');
  writeFileSync(path.join(ROOT, specPath), JSON.stringify({ name: p.name, test: path.join(ROOT, tests.get(frames[0])!), repeats: 10, seed: 0, only: p.only ?? null,
    frames: frames.map((t) => ({ frame: t, ours: path.join(ROOT, run.dir!, `f${t}`), ref: path.join(ROOT, refs.get(t)!.dir), test: path.join(ROOT, tests.get(t)!) })),
    predict: p.predict }, null, 1));
  const r = sh(PY, ['validation/tools/plant_sign.py', '--spec', specPath, '--out', path.join(dir, 'plants', safe(p.id), 'sign')], () => false);
  const rep = tryJson(path.join(dir, 'plants', safe(p.id), 'sign', 'report.json'));
  const t16 = run.meta ? [...refs.entries()].flatMap(([t, ref]) => t16ChainProblems(tryJson(path.join(run.dir!, `f${t}`, 'meta.json')) ?? {}, ref.meta, { plant: true, rounds: 1, frame: t, staticScene })) : ['no meta'];
  const ok = !!rep?.ok && fullOk && t16.length === 0;
  Object.assign(data, { ok, status: ok ? 'detected, sign holds' : rep ? `detected ${rep.detected}, sign ${rep.signs_hold}, only ${rep.only_ok}, full-not-pass ${fullOk}` : `plant_sign exit ${r.code}`,
    R, full_compare: full, predictions: rep?.frames?.flatMap((f: any) => f.predictions.map((x: any) => ({ frame: f.frame, region: x.region, sign: x.sign, rel: r4(x.rel), z: r4(x.z), holds: x.holds }))),
    detection: rep?.frames?.map((f: any) => ({ frame: f.frame, detected: `${f.detection.gate_fail_count}/${f.detection.n_repeats}`, control: `${f.detection.control_pass_count}/${f.detection.n_repeats}` })),
    only_hits: rep?.frames?.flatMap((f: any) => f.only_hits?.length ? [{ frame: f.frame, n: f.only_hits.length }] : []), t16, report: path.join(dir, 'plants', safe(p.id), 'sign', 'report.json') });
  add(`plant ${p.id}: ${p.name} on ${p.pkg} (${p.predict.map((x) => `f${x.frame} ${x.region} ${x.sign}`).join(', ')})`, ok, (performance.now() - t0) / 1000, data, data.status);
  return data;
}

function aaAndSynthetic(units: UnitPlan[], ptPlans: Map<string, PtPlan>, dir: string, runId: string, nU: number, add: Add): Record<string, any>[] {
  const out: Record<string, any>[] = [];
  const u = units.find((x) => x.pkg === AA_PKG && x.rung === '3.4');
  if (!u?.R) { add(`A/A on ${AA_PKG} 3.4`, false, 0, undefined, 'no sizing'); return out; }
  const R = u.R * CALIB_FACTOR;
  const runs = [SEEDS.aa, SEEDS.aa2].map((seed) => chainRun({ pkg: AA_PKG, preset: 'full', R, seed, frames: STATIC.T, testFrames: [24] }, path.join(dir, 'chains', `aa-${seed}`), u.chainSeconds * CALIB_FACTOR * 1.2));
  const t0 = performance.now();
  if (runs.every((r) => r.dir)) {
    const test = writeTest(dir, `aa-${AA_PKG}`, nU, 'B', u.tile);
    const cout = path.join(dir, 'compare', `aa-${AA_PKG}`);
    compare(path.join(runs[1].dir!, 'f24'), path.join(runs[0].dir!, 'f24'), test, cout);
    const rep = tryJson(path.join(cout, 'report.json'));
    const ok = rep?.status === 'pass' && runs.every((r) => r.meta?.ok);
    const res = { unit: `aa-${AA_PKG}`, kind: 'aa', ok, ...(rep ? summarize(rep) : { status: 'error' }), R: `${R} (x${CALIB_FACTOR})`, seeds: `${SEEDS.aa2} vs ${SEEDS.aa}` };
    out.push(res);
    add(`A/A: two chain seed sets on ${AA_PKG} rung 3.4 t=24 (${SEEDS.aa} vs ${SEEDS.aa2}, R ${R})`, ok, (performance.now() - t0) / 1000, res, rep ? `${rep.status}: Δ_Y ${pct(rep.channels.Y.global_.rel, 4)}` : 'no report');
    // synthetic W × 1.003 + calibrate A/A re-splits on the first A/A run (dynamic.py calibrate: rows of the npz)
    const c = sh(PY, ['validation/tools/dynamic.py', 'calibrate', '--ours', path.join(runs[0].dir!, 'f24'), '--test', test, '--out', path.join(dir, 'calibrate', `W1.003-${AA_PKG}`)], () => false);
    const cr = tryJson(path.join(dir, 'calibrate', `W1.003-${AA_PKG}`, 'report.json'));
    const ok2 = c.code === 0 && !!cr?.ok;
    const res2 = { unit: `plant-W1.003-${AA_PKG}`, kind: 'plant', ok: ok2, status: ok2 ? 'detected' : 'not detected', aa: cr?.aa && { ok: cr.aa.ok, gate_pass_rate: cr.aa.gate_pass_rate }, detected: cr && `${cr.plant.gate_fail_count}/${cr.plant.n_repeats}`, control: cr && `${cr.plant.control_pass_count}/${cr.plant.n_repeats}` };
    out.push(res2);
    add(`synthetic W x1.003 + calibrate A/A re-splits on the ${AA_PKG} 3.4 chains`, ok2, c.seconds, res2, cr ? `A/A ${cr.aa.ok ? 'ok' : 'FAIL'}, W x1.003 detected ${res2.detected}, control ${res2.control}` : c.out.slice(-300));
  } else add(`A/A on ${AA_PKG} 3.4`, false, 0, undefined, 'chain run failed');
  void ptPlans; void runId;
  return out;
}

// ---- budget + reports -----------------------------------------------------------------------------------------------------

function budgetRows(units: UnitPlan[], ptPlans: Map<string, PtPlan>): Record<string, unknown>[] {
  return [
    ...units.map((u) => ({ unit: u.id, scene: u.pkg, rung: u.rung, preset: u.preset, variant: u.variant ?? null, frames: u.frames, test_frames: u.testFrames, chains: u.R, members: E_MEMBERS,
      ms_per_chain: r4(u.msPerChain), ms_per_member_frame: r4(u.msPerChain / u.frames), chain_min: r4(u.chainSeconds / 60), tile: u.tile, cap: u.cap, notes: u.notes })),
    ...[...ptPlans.values()].map((p) => ({ unit: `pt:${p.pkg}:${p.frame ?? 'base'}`, scene: p.pkg, rung: 'pt', frame: p.frame ?? null, pt_spp: p.spp, B: p.B, pt_min: r4(p.seconds / 60) })),
  ];
}

function mergeBudget(rows: Record<string, unknown>[], runId: string, planHours: number, add: Add): void {
  const p = path.join(ROOT, 'validation/budget.json');
  const b = JSON.parse(readFileSync(p, 'utf8')) as Record<string, any>;
  b.m5_method = 'M5 gate sizing (gate-m5.ts): pilots PT 128 spp x 16 per (package, frame) and 64 chains per unit (E = 16, 256²) in headless Chrome; ms = wall time incl. readback; sizes = PLAN §7.3 rule per test frame and aggregate (incl. mask regions) x1.25, R >= 256 (3.5: 64), multiples of 16; PT spp niceCeil per batch >= 256';
  b.m5_measured_at = new Date().toISOString();
  b.m5_run = runId;
  b.m5_plan_hours = r4(planHours);
  b.m5_entries = rows;
  writeFileSync(p, `${JSON.stringify(b, null, 2)}\n`);
  add('budget.json M5 rows written (--write-budget)', true, 0, { rows: rows.length });
}

function budgetRowsPresent(add: Add): void {
  const b = tryJson('validation/budget.json') as { m5_entries?: { unit: string }[] } | undefined;
  const want = [...M5_STATIC.flatMap((s) => [`${s.pkg}@3.3`, `${s.pkg}@3.4`]), ...M5_AVG_SCENES.map((s) => `${s}@3.5`), ...DYN_UNITS.map((d) => d.id)];
  const have = new Set((b?.m5_entries ?? []).map((e) => e.unit));
  const miss = want.filter((w) => !have.has(w));
  add('budget.json M5 rows (every unit; gate-m5 --pilot-only --write-budget)', miss.length === 0, 0, { rows: have.size }, miss.length ? `missing ${miss.length}: ${miss.slice(0, 4).join(', ')}` : `${have.size} rows`);
}

function table(results: Record<string, any>[]): string {
  const head = '| unit | kind | rung | status | Δ_Y | MDB_Y | worst tile | tile MDB max | R | PT | tile | mult | min |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n';
  return head + results.map((r) => `| ${r.unit} | ${r.kind} | ${r.rung ?? ''} | ${r.status}${r.t16?.length ? ' (T16)' : ''} | ${pct(r.global_rel_Y, 4)} | ${pct(r.mdb_global_Y, 4)} | ${pct(r.worst_tile_rel_Y, 2)} | ${pct(r.mdb_tile_max_Y, 2)} | ${r.R ?? ''} | ${r.pt ?? ''} | ${r.tile ?? ''} | ${r.multiplier_needed ?? ''} | ${r.chain_minutes ?? ''} |`).join('\n') + '\n';
}

function summaryMd(summary: Record<string, any>, results: Record<string, any>[], sizing: Record<string, any>): string {
  return [
    `# M5 gate ${summary.run}`, '', `Result: **${summary.ok ? 'PASS' : 'FAIL'}** (${summary.total_s} s, n_units ${summary.n_units}, part ${summary.part}, plan ${summary.plan_hours} h)`, '',
    '## Per kind', '', '| kind | pass | fail | not run |', '|---|---|---|---|',
    ...Object.entries(summary.per_kind as Record<string, any>).map(([k, v]) => `| ${k} | ${v.pass} | ${v.fail} | ${v.notRun} |`), '',
    '## Units', '', table(results),
    '## Sizing', '', '| unit | R | frames | test frames | tile | min | cap | notes |', '|---|---|---|---|---|---|---|---|',
    ...(sizing.units as any[]).map((u) => `| ${u.id} | ${u.R} | ${u.frames} | ${u.testFrames.join(',')} | ${u.tile} | ${u.minutes} | ${u.cap} | ${u.notes.join('; ')} |`), '',
    '## Failed steps', '', ...(summary.failed as string[]).map((f) => `- ${f}`), '',
  ].join('\n');
}

export type { PilotSide };
