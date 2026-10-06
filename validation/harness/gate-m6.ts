// M6 milestone gate (restir-m6-api.md §4–§5; PLAN §5 M6 exit, §7.1 rungs 3.7 / 3.9 / 3.10 / 3.11 + Gate 5, §7.4 M6):
// `npm run validate -- --milestone M6 [--part core|r37|r39|r310|r311|gate5|plants] [--only id,pkg] [--pilot-only]`.
//   core    Gate 0: typecheck, cpu lane (incl. U-WGSL-BITS, T14, plant-m6 predictions), python tests (incl. dup_bias),
//           make-m6.ts determinism, the Chrome GPU suites of M6 (restir-m6: U-M5-BITS, compile smoke, U4-tiles, U1-M,
//           U10-B, U9-R, U-RR-M6, U-DMV-1, functional smokes; restir-debug: U-PAIR-VIEW / U-DUP-VIEW; restir-spatial:
//           T3-3/M6; restir-temporal: Mode-B T3-2, U8-2t with RIS) and the T3-M6 variants of restir-shift (one GPU-lock
//           hold per variant, ≤ 24 min, E-16), plus the M4 / M5 Gate-0 suites (regressions: restir-initial, restir-shift
//           incl. every M4 T3 variant, restir-spatial, restir-tframe (U-M4-BITS), restir-temporal, restir-refresh, the
//           M3 suites) and the T3-2 rare-bin holds; the M6 app smoke.
//   r37     rung 3.7: one unbiased feature toggled per unit on the M4 / M5 rung it extends, against the CACHED M4 / M5
//           PT references (MD14): pairing σ 16 ((i), (v) V1), RR ((i), (iv)), RIS-NEE ((x), (iv), (xiv) overcast + rect,
//           C0r irradiance), all ((i) offline-m6, (v) V2 offline-m6 + RR); chains full-m6 on m5s_cornell_i t ∈ {1, 24};
//           dual MVs (full + dualMv) on ixs_d_camera.
//   r39     rung 3.9 glass (offline-m6; new PT references, seed 6001; (vi) A reuses M4's).
//   r310    rung 3.10 alpha (offline-m6): (xii) (cached M4 PT), x10_foliage_256 (new).
//   r311    rung 3.11 Mode B (offline-m6 in Mode B): (i) B, m6_crossings_B_256, C0o, (xiv) overcast + rect B; chains
//           full-m6 Mode B on m6_crossings_B_256 t ∈ {1, 24}; dynamic ixs_b_area in Mode B (stage dyn).
//   gate5   MD15: chains of the interactive configuration (jitter iid, RR, σ 16, RIS, dual MV, boost 3) at cCap 5 and
//           20 with the duplication map ON (dup_bias.py: mean noise-debiased 16² tile |bias| ≤ 3.25 %, 99 % upper bound
//           of the global |bias| ≤ 3.25 %, noise floor ≤ 1 %) and its OFF twin (must pass Stage B / dyn) on
//           m5s_cornell_i t = 24, m5s_glossy_v1 t = 24 and ixs_d_camera.
//   plants  U8-4 / U8-7 / U8-8 / U8-10 (§5.3) with the predictions derived BEFORE the measurement (tests/restir/plant-m6
//           .test.ts): detected (plant_sign.py: half-size repeats ≥ 9/10 vs the 4× PT, PT A/A ≥ 9/10), the full comparison
//           not `pass`, the predicted sign (one-sided z ≥ 3) on the predicted region; synthetic W × 1.003 + calibrate A/A
//           re-splits and a ReSTIR A/A (6502 / 6503, 4× size) on (i) offline-m6.
// Statistics as M4 (compare.py stage B: TOST δ 0.2 % global / 1 % per 32² tile; dynamic 2 % per 64² tile, 3 % masks),
// pilot sizing (PLAN §7.3) × 1.25, one confirmatory re-run on disjoint seeds; δ is never loosened.
// T16 per run (§5.4): every M4 / M5 assertion, the light mode of the unit on both sides, the duplication map off outside
// Gate 5, plants only where named, the M6 features of the unit recorded in meta.json (restir.settings, t16.m6).
// Output: validation/out/m6-gate-<part>-<time>/ (summary.json/.md, sizing.json, tests/, compare/, restir/, chains/).
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodePFM, encodePFM } from '../../src/core/io/pfm.ts';
import { restirSettings, type RestirPresetName, type RestirSettings } from '../../src/core/render/restir/presets.ts';
import {
  cachedRun, codeHashes, GPU_SUITE_ENV, GPU_SUITES as M4_SUITES, gpuSuiteHolds, M4_SCENES, packageHash, runBatches, sceneDir, sizeScene,
  t16Problems, type PilotSide, type Run,
} from './gate-m4.ts';
import { EXT, GPU_SUITES as M5_SUITES, M5_GATE0_PLANTS, M5_SEQUENCES, runExternalChainUnits, T32_RARE_CASES, T32_RARE_ENV, type ExtChainUnit } from './gate-m5.ts';
import { withGpuLockSync } from './gpu-lock.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PY = path.join(ROOT, 'validation/.venv/bin/python');
export const M6_OUT = 'validation/out/m6';
export const M6_SCENE_DIR = `${M6_OUT}/scenes`;
const TIMEOUT_MS = 24 * 3600_000;
/** Seeds of the M6 gate (MD17). */
export const SEEDS = {
  pt: 6001, ptRerun: 106001, restir: 6002, restirRerun: 106002, aa: 6502, aa2: 6503, plantBase: 6101, ptCalib: 6201, ptPilot: 6011, restirPilot: 6012,
  chains: 6002, chainsRerun: 106002, chainPilot: 6012, revisedOffset: 700,
} as const;
const M4_PT = { root: 'validation/out/m4', pilot: 4011, ref: 4001, rerun: 104001 } as const;
const M6_PT = { root: M6_OUT, pilot: SEEDS.ptPilot, ref: SEEDS.pt, rerun: SEEDS.ptRerun } as const;
export const CALIB_FACTOR = 4;
export const NUM_EPS = 1e-4;
export const BUDGET_GATE5 = 0.0325;

/** Packages written by make-m6.ts (incl. its make-m3c.ts variants) into validation/out/m6/scenes. */
export const M6_PKGS = ['m6_crossings_B_256', 'x10_foliage_256', 'm6_tiles_glossy_256', 'cornell_i_B_512', 'ixs_b_area_B_256', 'xiv_overcast_rect_b3_512_glass', 'xiv_overcast_rect_b3_512_B'];
export function pkgDirM6(pkg: string): string {
  if (M6_PKGS.includes(pkg)) return `${M6_SCENE_DIR}/${pkg}`;
  const m4 = M4_SCENES.find((s) => s.pkg === pkg);
  return m4 ? sceneDir(m4) : `validation/scenes/${pkg}`;
}

// ------------------------------------------------------------------------------------------------ unit lists

export type Part = 'core' | 'r37' | 'r39' | 'r310' | 'r311' | 'gate5' | 'plants';
export const PARTS: Part[] = ['core', 'r37', 'r39', 'r310', 'r311', 'gate5', 'plants'];

export interface SeqUnit {
  id: string; part: Part; rung: string; pkg: string; label: string; preset: RestirPresetName; settings?: Partial<RestirSettings>;
  /** PT reference source: the cached M4 references (seed 4001) or new M6 ones (seed 6001). */
  pt: 'm4' | 'm6'; tier: 'tight' | 'heavy-tail'; lightMode: 'A' | 'B';
}
const U = (part: Part, rung: string, tag: string, pkg: string, label: string, preset: RestirPresetName, o: Partial<SeqUnit> = {}): SeqUnit =>
  ({ id: `${pkg}@${rung}-${tag}`, part, rung, pkg, label, preset, pt: 'm6', tier: 'tight', lightMode: 'A', ...o });
const GAUSS: Partial<RestirSettings> = { pairing: 'gauss', pairSigma: 16 };
const RR: Partial<RestirSettings> = { rr: true, rrMinBounces: 1 };
const RIS: Partial<RestirSettings> = { risNee: true, risM: 32 };
const G = (pkg: string, label: string, o: Partial<SeqUnit> = {}) => U('r39', '3.9', 'glass', pkg, label, 'offline-m6', o);

export const SEQ_UNITS: SeqUnit[] = [
  // ---- rung 3.7: one feature per unit, cached M4 PT references (MD14)
  U('r37', '3.7', 'gauss', 'cornell_i_512', '(i) offline + σ 16 Gaussian pairing', 'offline', { settings: GAUSS, pt: 'm4' }),
  U('r37', '3.7', 'gauss', 'v_glossy_v1_512', '(v) V1 offline + σ 16 Gaussian pairing', 'offline', { settings: GAUSS, pt: 'm4' }),
  U('r37', '3.7', 'rr', 'cornell_i_512', '(i) offline + RR (rrMinBounces 1)', 'offline', { settings: RR, pt: 'm4' }),
  U('r37', '3.7', 'rr', 'iv_emissive_mesh_512', '(iv) offline + RR (rrMinBounces 1)', 'offline', { settings: RR, pt: 'm4' }),
  U('r37', '3.7', 'ris', 'x_many_lights_512', '(x) offline + RIS-NEE (M 32, 128 × 1024 tiles)', 'offline', { settings: RIS, pt: 'm4' }),
  U('r37', '3.7', 'ris', 'iv_emissive_mesh_512', '(iv) offline + RIS-NEE', 'offline', { settings: RIS, pt: 'm4' }),
  U('r37', '3.7', 'ris', 'xiv_overcast_rect_b3_512', '(xiv) overcast + rect, offline + RIS-NEE (env in the tiles)', 'offline', { settings: RIS, pt: 'm4' }),
  U('r37', '3.7', 'ris', 'c0r_irradiance_256', 'C0r irradiance, offline + RIS-NEE (env-only tiles)', 'offline', { settings: RIS, pt: 'm4' }),
  U('r37', '3.7', 'all', 'cornell_i_512', '(i) offline-m6 (σ 16 + RIS-NEE)', 'offline-m6', { pt: 'm4' }),
  U('r37', '3.7', 'all', 'v_glossy_v2_512', '(v) V2 offline-m6 + RR', 'offline-m6', { settings: RR, pt: 'm4' }),
  // ---- rung 3.9: glass in ReSTIR (offline-m6)
  G('c0h_slab_transmission_256', 'C0h slab transmission'), G('c0i_immersed_emitter_256', 'C0i immersed emitter'),
  G('c0j_rough_glass_furnace_512x256', 'C0j rough glass furnace'), G('c0k_glass_shadow_256', 'C0k glass shadow'),
  G('g1_slab_furnace_glassnode_256', 'G1 slab furnace (Glass node)'), G('g1_slab_furnace_principled_256', 'G1 slab furnace (Principled)'),
  G('g3_rough_slab_reflection_512x256', 'G3 rough slab reflection'), G('g4_rough_slab_512x256', 'G4 rough slab'),
  G('g5_panes_point_256', 'G5 panes, point light'), G('g5b_panes_rect_B_256', 'G5b panes, rect light, Mode B', { lightMode: 'B' }),
  G('g6_caustic_B_256', 'G6 caustic, Mode B', { lightMode: 'B', tier: 'heavy-tail' }), G('g6neg_caustic_A_256', 'G6-neg caustic, Mode A', { tier: 'heavy-tail' }),
  G('g7_principled_mix_512', 'G7 Principled mix'), G('g7_principled_mix_furnace_512x256', 'G7 Principled mix furnace'),
  G('g8_cornell_glass_512', 'G8 Cornell glass'), G('g9_bubble_256', 'G9 bubble'), G('g10_colored_glass_refraction_256', 'G10 coloured glass refraction'),
  G('vi_glass_mirror_A_512', '(vi) glass + mirror + glossy, Mode A', { pt: 'm4' }),
  G('vi_glass_mirror_B_512', '(vi) glass + mirror + glossy, Mode B', { lightMode: 'B' }),
  G('vi_b_mirror_area_B_256', '(vi-B) mirror + area light, Mode B', { lightMode: 'B' }),
  G('xiv_overcast_rect_b3_512_glass', '(xiv) overcast + rect + glass sphere', { tier: 'heavy-tail' }),
  // ---- rung 3.10: alpha (cutouts on reconnection segments)
  U('r310', '3.10', 'alpha', 'xii_alpha_foliage_512', '(xii) alpha MASK foliage cards', 'offline-m6', { pt: 'm4' }),
  U('r310', '3.10', 'alpha', 'x10_foliage_256', 'large foliage: 480 alpha cards, sun + sky + rect', 'offline-m6'),
  // ---- rung 3.11: Mode B (offline-m6 in Mode B)
  U('r311', '3.11', 'modeB', 'cornell_i_B_512', '(i) Mode B', 'offline-m6', { lightMode: 'B' }),
  U('r311', '3.11', 'modeB', 'm6_crossings_B_256', 'crossing lights with geometry behind, Mode B', 'offline-m6', { lightMode: 'B' }),
  U('r311', '3.11', 'modeB', 'c0o_visible_camera_B_256', 'C0o lights visible to the camera, Mode B', 'offline-m6', { lightMode: 'B' }),
  U('r311', '3.11', 'modeB', 'xiv_overcast_rect_b3_512_B', '(xiv) overcast + rect, Mode B', 'offline-m6', { lightMode: 'B' }),
];

const json = (o: unknown) => JSON.stringify(o);
const ixsD = M5_SEQUENCES.find((s) => s.pkg === 'ixs_d_camera_256')!;
const ixsB = M5_SEQUENCES.find((s) => s.pkg === 'ixs_b_area_256')!;
export interface ChainUnitM6 extends ExtChainUnit { part: Part; label: string; lightMode: 'A' | 'B'; gate5?: { scene: string; cCap: number; dupmap: boolean; frame: number } }
const C = (part: Part, u: Omit<ChainUnitM6, 'part'>): ChainUnitM6 => ({ ...u, part });
const gate5Units = (): ChainUnitM6[] => {
  const out: ChainUnitM6[] = [];
  const scenes: { pkg: string; kind: 'static' | 'dyn'; frames: number; testFrames: number[]; frame: number }[] = [
    { pkg: 'm5s_cornell_i', kind: 'static', frames: 25, testFrames: [24], frame: 24 },
    { pkg: 'm5s_glossy_v1', kind: 'static', frames: 25, testFrames: [24], frame: 24 },
    { pkg: 'ixs_d_camera_256', kind: 'dyn', frames: ixsD.T, testFrames: ixsD.testFrames, frame: Math.max(...ixsD.testFrames) },
  ];
  for (const s of scenes) for (const cCap of [5, 20]) for (const dupmap of [true, false]) {
    out.push(C('gate5', { id: `${s.pkg}@G5-cap${cCap}-${dupmap ? 'dup' : 'nodup'}`, kind: s.kind, pkg: s.pkg, rung: 'G5', preset: 'interactive', variant: dupmap ? 'dupmap' : 'twin',
      extra: ['--restir-settings', json({ cCap, dupmap })], frames: s.frames, testFrames: s.testFrames, rounds: 1, lightMode: 'A',
      label: `interactive cCap ${cCap}, duplication map ${dupmap ? 'ON' : 'off (twin)'}`,
      t16: { cCap, rr: true, ...(dupmap ? { biased: 'dupmap' as const } : {}) }, gate5: { scene: s.pkg, cCap, dupmap, frame: s.frame } }));
  }
  return out;
};
export const CHAIN_UNITS: ChainUnitM6[] = [
  C('r37', { id: 'm5s_cornell_i@3.7-full-m6', kind: 'static', pkg: 'm5s_cornell_i', rung: '3.7', preset: 'full-m6', extra: [], frames: 25, testFrames: [1, 24], rounds: 1, lightMode: 'A', label: 'chains full-m6 (σ 16 + RIS)' }),
  C('r37', { id: 'ixs_d_camera_256@3.7-dualmv', kind: 'dyn', pkg: 'ixs_d_camera_256', rung: '3.7', preset: 'full', variant: 'dualmv', extra: ['--restir-settings', json({ dualMv: true })],
    frames: ixsD.T, testFrames: ixsD.testFrames, rounds: 1, lightMode: 'A', label: 'chains full + dual MVs (stage dyn)' }),
  C('r311', { id: 'm6_crossings_B_256@3.11-full-m6', kind: 'static', pkg: 'm6_crossings_B_256', rung: '3.11', preset: 'full-m6', extra: [], frames: 25, testFrames: [1, 24], rounds: 1, lightMode: 'B',
    label: 'chains full-m6 Mode B', t16: { lightMode: 'B' } }),
  C('r311', { id: 'ixs_b_area_B_256@3.11-full-m6', kind: 'dyn', pkg: 'ixs_b_area_B_256', rung: '3.11', preset: 'full-m6', extra: [], frames: ixsB.T, testFrames: ixsB.testFrames, rounds: 1, lightMode: 'B',
    label: 'dynamic area light, Mode B, full-m6 (stage dyn)', t16: { lightMode: 'B' } }),
  ...gate5Units(),
];

export interface PlantM6 {
  id: string; name: string; pkg: string; preset: RestirPresetName; base: Partial<RestirSettings>; plant: NonNullable<RestirSettings['plant']>;
  lightMode: 'A' | 'B'; predict: { region: string; sign: '+' | '-' | 'detect' }[]; masks: ('M_foot' | 'M_hl')[]; derivation: string;
}
export const PLANTS_M6: PlantM6[] = [
  { id: 'U8-4', name: 'U8-4 J = t_x²/t_y² (RSF_PLANT_U8_T2)', pkg: 'u8_c0c_point_b0', preset: 'offline', base: {}, plant: { u8T2: true }, lightMode: 'A',
    predict: [{ region: 'M_foot', sign: '+' }, { region: 'global', sign: 'detect' }], masks: ['M_foot'],
    derivation: 'pairwise-MIS toy on the package (plant-m6.test.ts): M_foot +2.6 %, global +1.0 % (global reported)' },
  { id: 'U8-7', name: 'U8-7 crossed area lights stop BSDF rays in the path tree (RSF_PLANT_U8_CROSS_OCC)', pkg: 'm6_crossings_B_256', preset: 'initial', base: {}, plant: { u8CrossOcc: true }, lightMode: 'B',
    predict: [{ region: 'global', sign: '-' }], masks: [], derivation: 'energy loss: light-ending paths whose BSDF segment passes a light are lost (gap-light §2.6)' },
  { id: 'U8-8', name: 'U8-8 RIS UCW in mixed measures (RSF_PLANT_U8_RIS_MIXED)', pkg: 'u8_c0e_rect_b0', preset: 'initial', base: RIS, plant: { u8RisMixed: true }, lightMode: 'A',
    predict: [{ region: 'global', sign: '+' }], masks: [], derivation: 'f64 quadrature (plant-m6.test.ts): contribution-weighted E[r²/cos θ_z] = 1.264 ⇒ +26 %' },
  { id: 'U8-10', name: 'U8-10 tile-conditional pmf in ω1 (RSF_PLANT_U8_TILE_PMF)', pkg: 'm6_tiles_glossy_256', preset: 'initial', base: RIS, plant: { u8TilePmf: true }, lightMode: 'A',
    predict: [{ region: 'global', sign: '+' }, { region: 'M_hl', sign: '+' }], masks: ['M_hl'],
    derivation: 'a selected light is present in its tile: tile frequency ≥ 1/1024 > pmf ⇒ ω1 inflated, ω1 + ω2 > 1 (strongest where p2 is large: the glossy highlight)' },
];
export const AA_PKG = 'cornell_i_512';

/** M6-only Chrome holds of Gate 0 (each ≤ 12 min; the rest of each file runs in the M4 / M5 regression holds). */
export const T32_MODEB_CASES = ['Mode B: camera translate', 'Mode B: moving + rotating crossing lights', 'Mode B: add / remove + intensity (crossing lights)'];
export const T32_MODEB_ENV = { VITE_T32_RARE_PAIRS: '24000', VITE_T32_RARE_RES: '256', VITE_T32_MIN_BIN: '1000000' };
const reEsc = (x: string) => x.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
export const M6_GPU_SUITES: { file: string; what: string; t?: string; env?: Record<string, string> }[] = [
  { file: 'restir-m6', what: 'U-M5-BITS, variant compile smoke, U4-tiles, U1-M, U10-B, U9-R, U-RR-M6, U-DMV-1, functional smokes vs the PT' },
  ...T32_MODEB_CASES.map((c) => ({ file: 'restir-temporal', what: `T3-2/M6 ${c} (every ana / deep bin >= 1e6, LOGIC 0)`, t: reEsc(c), env: T32_MODEB_ENV })),
];
/** The T3-M6 runs of restir-shift.gpu.test.ts (test names). */
export const T3M6_DESCRIBE = 'T3-M6: T3-0 / T3-1 / T4 / T3-D LOGIC = 0 on the M6 cases';
export const T3M6_RUNS = ['t3_cases_256 + RIS-NEE', 't3_modeb_256 (Mode B + RIS-NEE)', 't3_modeb_rare_256 (Mode B, frequent ∅ crossings)', 't3_glass_256 (rough-glass G_R reconnection)',
  't3_glass_pane_256 (rough-glass side flips)', 't3_alpha_256 (cutouts on reconnection segments)'];

export function nUnits(): number {
  const chains = CHAIN_UNITS.reduce((a, u) => a + u.testFrames.length, 0);
  return 4 * (SEQ_UNITS.length + chains + PLANTS_M6.length + 2);
}

// ------------------------------------------------------------------------------------------------ helpers

const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const r4 = (x: unknown) => (typeof x === 'number' ? Number(x.toPrecision(4)) : x);
const pct = (x: number | undefined | null, d = 3) => (typeof x === 'number' ? `${(x * 100).toFixed(d)}%` : 'n/a');
const tryJson = (p: string): Record<string, any> | undefined => { try { return JSON.parse(readFileSync(path.join(ROOT, p), 'utf8')); } catch { return undefined; } };
const safe = (s: string) => s.replace(/[^\w.@-]+/g, '_');
type Add = (name: string, ok: boolean, seconds: number, data?: unknown, detail?: string) => void;
type Rec = (name: string, ok: boolean, detail?: string) => void;

function sh(cmd: string, argv: string[], echo: (l: string) => boolean = () => true, env?: Record<string, string>): { code: number; out: string; seconds: number } {
  const t0 = performance.now();
  const r = spawnSync(cmd, argv, { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 28, timeout: TIMEOUT_MS, ...(env ? { env: { ...process.env, ...env } } : {}) });
  const out = `${r.stdout ?? ''}${r.stderr ?? ''}`;
  for (const l of out.split('\n')) if (l && echo(l)) console.log(`  ${l}`);
  return { code: r.status ?? 1, out, seconds: (performance.now() - t0) / 1000 };
}

function batchSide(dir: string, meta: Record<string, any>): PilotSide {
  const files = readdirSync(path.join(ROOT, dir)).filter((n) => /^batch_\d{3}\.pfm$/.test(n)).sort();
  const imgs = files.map((f) => decodePFM(new Uint8Array(readFileSync(path.join(ROOT, dir, f)))));
  const n = meta.sppPerBatch as number, B = files.length;
  return { B, n, msPerSample: (meta.timings.batchMs as number[]).reduce((a, b) => a + b, 0) / (B * n), W: imgs[0].width, H: imgs[0].height, batches: imgs.map((i) => i.data) };
}

function writeTest(dir: string, name: string, nU: number, tier: 'tight' | 'heavy-tail', tile: number, extra: Record<string, unknown> = {}): string {
  const p = path.join(dir, 'tests', `test-${safe(name)}.json`);
  mkdirSync(path.join(ROOT, dir, 'tests'), { recursive: true });
  writeFileSync(path.join(ROOT, p), JSON.stringify({
    name, stage: 'B', channels: ['Y', 'R', 'G', 'B'], n_units: nU, tier, num_eps: NUM_EPS, num_eps_note: 'two f32 implementations (cycles-deviations.md D2; restir-api.md §6.3)',
    min_replicates: tier === 'heavy-tail' ? 32 : 16, ...(tile !== 32 ? { tile, aggregate_note: 'pilot sizing: a side needs > 60 min at 32² tiles (restir-api.md §6.3)' } : {}), ...extra,
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
    mdb_tile_max_Y: r4(Y.tiles.mdb_max), tost_failed_tiles: Y.tiles.tost_failed,
    multiplier_needed: r4(Math.max(...['Y', 'R', 'G', 'B'].map((c) => Math.max(rep.channels[c].global_.replicate_multiplier_needed ?? 1, rep.channels[c].tiles.replicate_multiplier_needed ?? 1)))),
  };
}

/** T16 of a sequential M6 run (§5.4): the M4 assertions with the unit's light mode instead of "Mode A", + the M6 record. */
export function t16M6(rs: Record<string, any>, pt: Record<string, any>, o: { plant?: boolean; rounds: number; lightMode: 'A' | 'B'; expect: RestirSettings }): string[] {
  const p = t16Problems(rs, pt, { plant: o.plant, rounds: o.rounds }).filter((x) => x !== 'not Mode A on both sides');
  const t = rs.t16 ?? {};
  if (t.lightMode !== o.lightMode || (pt.config?.lightMode ?? 'A') !== o.lightMode) p.push(`light mode: ReSTIR ${t.lightMode}, PT ${pt.config?.lightMode}, unit ${o.lightMode}`);
  const m6 = t.m6 ?? {};
  for (const k of ['pairing', 'risNee', 'rr', 'dualMv', 'dupmap'] as const) if (m6[k] !== o.expect[k]) p.push(`setting ${k} = ${m6[k]} (unit: ${o.expect[k]})`);
  if (m6.dupmap) p.push('duplication map on outside Gate 5');
  if (o.expect.risNee && m6.risM !== 32) p.push(`risM ${m6.risM} != 32`);
  if (o.expect.pairing === 'gauss' && m6.pairSigma !== 16) p.push(`pairSigma ${m6.pairSigma} != 16`);
  return p;
}

const ptKey = (pkg: string, spp: number, B: number, seed: number) => ({ kind: 'pt', pkg, packageHash: packageHash(pkgDirM6(pkg)), spp, B, seed, rr: false, code: codeHashes().pt });
const ptsizeFile = (root: string, pkg: string) => path.join(root, 'ptsize', `${pkg}-${codeHashes().pt.slice(0, 16)}-${packageHash(pkgDirM6(pkg)).slice(0, 16)}.json`);
const pilotFrames = (preset: string) => (/^offline/.test(preset) || preset === 'criteria2022' ? 8 : 128);
const roundsOf = (preset: RestirPresetName, settings?: Partial<RestirSettings>) => restirSettings(preset, settings ?? {}).rounds;
function rsArgs(pkg: string, preset: string, settings?: Partial<RestirSettings>): string[] {
  return ['--package', pkgDirM6(pkg), '--kernel', 'restir', '--preset', preset, ...(settings && Object.keys(settings).length ? ['--restir-settings', json(settings)] : [])];
}

interface Sized { B: number; ptSpp: number; ptSeconds: number; frames: number; seconds: number; tile: 32 | 64; notes: string[]; ptSide: PilotSide; ptPilot: string }

/** PT pilot (cached) + ReSTIR pilot of the unit's configuration + PLAN §7.3 sizing around the frozen PT size. */
function sizeUnit(pkg: string, preset: RestirPresetName, settings: Partial<RestirSettings> | undefined, ptFrom: 'm4' | 'm6', tier: 'tight' | 'heavy-tail', add: Add, tag: string): Sized | undefined {
  const P = ptFrom === 'm4' ? M4_PT : M6_PT;
  const B = tier === 'heavy-tail' ? 32 : 16;
  const pdir = pkgDirM6(pkg);
  const pt = cachedRun(`${P.root}/pilots`, ptKey(pkg, 128, B, P.pilot), `${pkg}-pt`, ['--package', pdir, '--kernel', 'pt', '--spp', '128', '--batches', String(B), '--seed', String(P.pilot)],
    B, add, `pilot PT ${pkg} (128 spp x ${B}, seed ${P.pilot})`);
  if (!pt) return undefined;
  const f = pilotFrames(preset);
  const rk = { kind: 'restir', pkg, packageHash: packageHash(pdir), preset, settings: settings ?? null, frames: f, B, seed: SEEDS.restirPilot, code: codeHashes().restir };
  const rp = cachedRun(`${M6_OUT}/pilots`, rk, `${pkg}-${tag.replace(/[^\w.-]+/g, '_')}`, [...rsArgs(pkg, preset, settings), '--spp', String(f), '--batches', String(B), '--seed', String(SEEDS.restirPilot)],
    B, add, `pilot ReSTIR ${pkg} ${tag} (${f} frames x ${B})`);
  if (!rp) return undefined;
  const sf = ptsizeFile(P.root, pkg);
  const frozen = tryJson(sf) as { ptSpp: number; B: number } | undefined;
  const ptSide = batchSide(pt.dir, pt.meta);
  const z = sizeScene(ptSide, [{ id: tag, side: batchSide(rp.dir, rp.meta), tile: 32 }], { B, ...(frozen?.B === B ? { fixedPtSpp: frozen.ptSpp } : {}), minFrames: () => f });
  if (!frozen) {
    mkdirSync(path.dirname(path.join(ROOT, sf)), { recursive: true });
    writeFileSync(path.join(ROOT, sf), `${JSON.stringify({ pkg, ptSpp: z.ptSpp, B: z.B, ptSeconds: z.ptSeconds, created: new Date().toISOString() }, null, 1)}\n`);
  }
  const r = z.rungs[tag];
  return { B, ptSpp: z.ptSpp, ptSeconds: z.ptSeconds, frames: r.framesPerBatch, seconds: r.seconds, tile: r.tile, notes: z.notes, ptSide, ptPilot: pt.dir };
}

function ptRefRun(pkg: string, ptFrom: 'm4' | 'm6', spp: number, B: number, seed: number, estSeconds: number, add: Add): Run | undefined {
  const P = ptFrom === 'm4' ? M4_PT : M6_PT;
  return cachedRun(`${P.root}/ptrefs`, ptKey(pkg, spp, B, seed), `${pkg}-s${seed}-${spp}x${B}`,
    ['--package', pkgDirM6(pkg), '--kernel', 'pt', '--spp', String(spp), '--batches', String(B), '--seed', String(seed)], B, add, `PT reference ${pkg} (${spp} spp x ${B}, seed ${seed})`, estSeconds);
}

// ------------------------------------------------------------------------------------------------ sequential units

function seqUnit(u: SeqUnit, dir: string, runId: string, nU: number, add: Add, pilotOnly: boolean): Record<string, any> {
  const t0 = performance.now();
  const base = { unit: u.id, kind: 'seq', part: u.part, rung: u.rung, scene: u.pkg, label: u.label, preset: u.preset, settings: u.settings ?? {}, lightMode: u.lightMode, tier: u.tier };
  const z = sizeUnit(u.pkg, u.preset, u.settings, u.pt, u.tier, add, u.id);
  if (!z) return { ...base, ok: false, status: 'not run', note: 'pilot failed' };
  const sizing = { B: z.B, ptSpp: z.ptSpp, pt_min: r4(z.ptSeconds / 60), frames: z.frames, restir_min: r4(z.seconds / 60), tile: z.tile, notes: z.notes };
  add(`sizing ${u.id}`, true, 0, sizing, `PT ${z.ptSpp} spp x ${z.B} (${(z.ptSeconds / 60).toFixed(1)} min), ReSTIR ${z.frames} fr x ${z.B} (${(z.seconds / 60).toFixed(1)} min)${z.tile === 64 ? ' @64²' : ''}${z.notes.length ? `; ${z.notes.join('; ')}` : ''}`);
  if (pilotOnly) return { ...base, ok: true, status: 'sized', sizing };
  const P = u.pt === 'm4' ? M4_PT : M6_PT;
  const ref = ptRefRun(u.pkg, u.pt, z.ptSpp, z.B, P.ref, 1.2 * z.ptSeconds, add);
  if (!ref) return { ...base, ok: false, status: 'not run', note: 'PT reference failed', sizing };
  const expect = restirSettings(u.preset, u.settings ?? {});
  const rounds = expect.rounds;
  const run = (seed: number, sub: string) => {
    console.log(`\n--- ReSTIR ${u.id}: ${z.frames} frames x ${z.B}, seed ${seed}`);
    return runBatches([...rsArgs(u.pkg, u.preset, u.settings), '--spp', String(z.frames), '--batches', String(z.B), '--seed', String(seed)],
      `${runId}-${sub}`.replace(/[^\w.-]+/g, '_').slice(0, 120), path.join(dir, 'restir', safe(sub)), { B: z.B, estSeconds: z.seconds * 1.2 });
  };
  const first = run(SEEDS.restir, u.id);
  const test = writeTest(dir, u.id, nU, u.tier, z.tile);
  const out = path.join(dir, 'compare', safe(u.id));
  const t16 = first.meta ? t16M6(first.meta, ref.meta, { rounds, lightMode: u.lightMode, expect }) : ['ReSTIR run produced no meta.json'];
  let rep: Record<string, any> | undefined;
  if (first.dir && first.meta) { compare(first.dir, ref.dir, test, out); rep = tryJson(path.join(out, 'report.json')); }
  let rerun: Record<string, unknown> | undefined;
  if (rep?.status === 'rerun_required') {
    console.log(`  ${u.id}: rerun_required (${rep.failed_checks.join(', ')}) -> confirmatory re-run on disjoint seeds (PT ${P.rerun}, ReSTIR ${SEEDS.restirRerun})`);
    const ref2 = ptRefRun(u.pkg, u.pt, z.ptSpp, z.B, P.rerun, 1.2 * z.ptSeconds, add);
    const second = run(SEEDS.restirRerun, `${u.id}-rerun`);
    if (ref2 && second.dir && second.meta) {
      t16.push(...t16M6(second.meta, ref2.meta, { rounds, lightMode: u.lightMode, expect }).map((x) => `re-run: ${x}`));
      compare(second.dir, ref2.dir, test, `${out}-rerun`, path.join(out, 'report.json'));
      rerun = { first: summarize(rep), report: path.join(`${out}-rerun`, 'report.json') };
      rep = tryJson(path.join(`${out}-rerun`, 'report.json'));
    }
  }
  const sum = rep ? summarize(rep) : undefined;
  const ok = !!rep && (rep.status === 'pass' || rep.status === 'pass_on_rerun') && t16.length === 0;
  const rm = first.meta?.restir ?? {};
  const res = {
    ...base, ok, ...(sum ?? { status: 'error' }), sizing, pt: `${z.ptSpp}x${z.B}`, restir: `${z.frames}x${z.B}`, tile: z.tile, fr: r4(rm.fr),
    restir_minutes: first.meta ? r4(first.meta.timings.totalMs / 60000) : undefined, pt_cache_hit: ref.cacheHit, t16, report: path.join(out, 'report.json'),
    restir_dir: first.dir, pt_dir: ref.dir, ...(rerun ? { rerun } : {}),
  };
  add(`Stage B ${u.id} (${u.label}; ${u.preset}${u.settings ? ` ${json(u.settings)}` : ''}, Mode ${u.lightMode})`, ok, (performance.now() - t0) / 1000, res,
    sum ? `${sum.status}: Δ_Y ${pct(sum.global_rel_Y as number, 4)} (MDB ${pct(sum.mdb_global_Y as number, 3)}), worst tile ${pct(sum.worst_tile_rel_Y as number, 2)} ${sum.worst_tile ?? ''}, mult ${sum.multiplier_needed}${sum.failed_checks?.length ? `, failed ${sum.failed_checks.join(' ')}` : ''}${t16.length ? `; T16: ${t16.join('; ')}` : ''}`
      : `no report${t16.length ? `; T16: ${t16.join('; ')}` : ''}`);
  return res;
}

// ------------------------------------------------------------------------------------------------ chain units

function setupExt(): void {
  for (const p of ['m6_crossings_B_256', 'ixs_b_area_B_256']) EXT.pkgDirs.set(p, pkgDirM6(p));
  EXT.sequences = [{ ...ixsB, pkg: 'ixs_b_area_B_256', label: `${ixsB.label} (Mode B)` }];
  EXT.out = M6_OUT;
  EXT.chainSeeds = { chains: SEEDS.chains, chainsRerun: SEEDS.chainsRerun, chainPilot: SEEDS.chainPilot };
}

function chainUnits(units: ChainUnitM6[], dir: string, runId: string, nU: number, add: Add, pilotOnly: boolean): { results: Record<string, any>[]; sizing: Record<string, unknown> } {
  setupExt();
  const r = runExternalChainUnits(units, { dir, runId, nU, add, pilotOnly });
  const results = r.results.map((x) => {
    const u = units.find((v) => x.unit === v.id || String(x.unit).startsWith(`${v.id}-`))!;
    return { ...x, part: u.part, label: u.label, lightMode: u.lightMode, kind: u.gate5 ? (u.gate5.dupmap ? 'gate5-on' : 'gate5-twin') : `chain-${x.kind}` };
  });
  return { results, sizing: r.sizing };
}

/** Gate 5 metric on the dupmap-ON chains (dup_bias.py) of each scene × c_cap at its frame. */
function gate5Metrics(units: ChainUnitM6[], results: Record<string, any>[], dir: string, add: Add): Record<string, any>[] {
  const out: Record<string, any>[] = [];
  for (const u of units.filter((x) => x.gate5?.dupmap)) {
    const g = u.gate5!;
    const t0 = performance.now();
    const chainDir = path.join(dir, 'chains', safe(u.id), `f${g.frame}`);
    const unitRes = results.find((x) => x.unit === `${u.id}-f${g.frame}`);
    const ref: string | undefined = unitRes?.pt_dir;
    const twin = results.find((x) => x.unit === `${u.id.replace(/-dup$/, '-nodup')}-f${g.frame}`);
    const o = path.join(dir, 'gate5', safe(u.id));
    const r = ref && existsSync(path.join(ROOT, chainDir, 'ensemble.npz')) ? sh(PY, ['validation/tools/dup_bias.py', '--ours', chainDir, '--ref', ref, '--out', o, '--budget', String(BUDGET_GATE5)], () => false) : undefined;
    const rep = tryJson(path.join(o, 'report.json'));
    const twinOk = !!twin?.ok;
    const ok = !!rep?.ok && twinOk && (unitRes?.t16?.length ?? 1) === 0;
    const res = { unit: `gate5-${g.scene}-cap${g.cCap}`, kind: 'gate5', part: 'gate5', scene: g.scene, cCap: g.cCap, frame: g.frame, ok, status: ok ? 'pass' : 'fail',
      mean_tile_bias: r4(rep?.mean_tile_bias), global_rel: r4(rep?.global_rel), global_abs_upper99: r4(rep?.global_abs_upper99), noise_floor: r4(rep?.noise_floor),
      twin: twin ? { status: twin.status, ok: twin.ok, global_rel_Y: twin.global_rel_Y } : 'missing', t16: unitRes?.t16, report: path.join(o, 'report.json'), budget: BUDGET_GATE5 };
    out.push(res);
    add(`Gate 5 ${g.scene} cCap ${g.cCap} t=${g.frame}: duplication-map bias <= ${pct(BUDGET_GATE5, 2)} (twin unbiased)`, ok, (performance.now() - t0) / 1000, res,
      rep ? `mean tile |bias| ${pct(rep.mean_tile_bias, 2)}, global ${pct(rep.global_rel, 3)} (99% |.| <= ${pct(rep.global_abs_upper99, 3)}), noise floor ${pct(rep.noise_floor, 2)}; twin ${twin?.status ?? 'missing'}` : `dup_bias exit ${r?.code ?? 'not run'}`);
  }
  return out;
}
// ------------------------------------------------------------------------------------------------ plants

/** Planted-region masks (PFM, R = 1 inside) in the test directory: M_foot (24 px disc around the light's foot point) and
 *  M_hl (top 10 % PT radiance of the PT pilot, an image independent of both compared sides). */
function plantMasks(p: PlantM6, ptPilot: PilotSide, dir: string): { name: string; file: string }[] {
  const out: { name: string; file: string }[] = [];
  const W = ptPilot.W, H = ptPilot.H;
  const mdir = path.join(dir, 'tests', 'masks');
  mkdirSync(path.join(ROOT, mdir), { recursive: true });
  for (const m of p.masks) {
    const a = new Float32Array(W * H * 3);
    if (m === 'M_foot') {
      const j = tryJson(path.join(pkgDirM6(p.pkg), 'scene.json'))!;
      const cm = j.camera.matrix as number[], L = j.lights[0].matrix as number[];
      const q = [L[12], 0, L[14]], rel = [q[0] - cm[12], q[1] - cm[13], q[2] - cm[14]];
      const d3 = (v: number[], o: number) => rel[0] * v[o] + rel[1] * v[o + 1] + rel[2] * v[o + 2];
      const cx = d3(cm, 0), cy = d3(cm, 4), cz = -d3(cm, 8), t = Math.tan(j.camera.yfov / 2), asp = W / H;
      const fx = ((cx / cz) / (t * asp) + 1) / 2 * W - 0.5, fy = (1 - (cy / cz) / t) / 2 * H - 0.5;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if ((x - fx) ** 2 + (y - fy) ** 2 <= 24 * 24) a[3 * (y * W + x)] = 1;
    } else {
      const lum = new Float64Array(W * H);
      for (const img of ptPilot.batches) for (let i = 0; i < W * H; i++) lum[i] += 0.2126 * img[3 * i] + 0.7152 * img[3 * i + 1] + 0.0722 * img[3 * i + 2];
      const thr = [...lum].sort((x, y) => x - y)[Math.floor(0.9 * W * H)];
      for (let i = 0; i < W * H; i++) if (lum[i] > thr) a[3 * i] = 1;
    }
    writeFileSync(path.join(ROOT, mdir, `${m}-${p.id}.pfm`), encodePFM({ width: W, height: H, channels: 3, data: a }));
    out.push({ name: m, file: `masks/${m}-${p.id}.pfm` });
  }
  return out;
}

function plantUnit(p: PlantM6, i: number, seedOffset: number, dir: string, runId: string, nU: number, add: Add): Record<string, any> {
  const t0 = performance.now();
  const data: Record<string, any> = { unit: `plant-${p.id}`, kind: 'plant', part: 'plants', scene: p.pkg, plant: p.name, preset: p.preset, predict: p.predict, derivation: p.derivation, ok: false };
  const z = sizeUnit(p.pkg, p.preset, Object.keys(p.base).length ? p.base : undefined, 'm6', 'tight', add, `plantbase-${p.id}`);
  if (!z) return { ...data, status: 'sizing failed' };
  const calibSeed = SEEDS.ptCalib + i + seedOffset, pseed = SEEDS.plantBase + i + seedOffset;
  const ref = ptRefRun(p.pkg, 'm6', z.ptSpp * CALIB_FACTOR, z.B, calibSeed, 1.2 * CALIB_FACTOR * z.ptSeconds, add);
  if (!ref) return { ...data, status: 'PT reference failed' };
  const settings = { ...p.base, plant: p.plant };
  const run = runBatches([...rsArgs(p.pkg, p.preset, settings), '--spp', String(z.frames), '--batches', String(z.B), '--seed', String(pseed)],
    `${runId}-plant-${p.id}`.replace(/[^\w.-]+/g, '_'), path.join(dir, 'plants', safe(p.id), 'restir'), { B: z.B, estSeconds: z.seconds * 1.2 });
  if (!run.dir || !run.meta) { add(`plant ${p.id}`, false, run.seconds, undefined, run.out.slice(-300)); return { ...data, status: 'run failed' }; }
  const masks = plantMasks(p, z.ptSide, dir);
  const test = writeTest(dir, `plant-${p.id}`, nU, 'tight', z.tile, masks.length ? { masks } : {});
  const fullOut = path.join(dir, 'compare', safe(`plant-${p.id}-full`));
  compare(run.dir, ref.dir, test, fullOut);
  const full = tryJson(path.join(fullOut, 'report.json'));
  const spec = path.join(dir, 'plants', safe(p.id), 'plant.json');
  writeFileSync(path.join(ROOT, spec), JSON.stringify({ name: p.name, test: path.join(ROOT, test), repeats: 10, seed: 0, only: null,
    frames: [{ frame: 0, ours: path.join(ROOT, run.dir), ref: path.join(ROOT, ref.dir), test: path.join(ROOT, test) }],
    predict: p.predict.map((x) => ({ frame: 0, ...x })) }, null, 1));
  const r = sh(PY, ['validation/tools/plant_sign.py', '--spec', spec, '--out', path.join(dir, 'plants', safe(p.id), 'sign')], () => false);
  const rep = tryJson(path.join(dir, 'plants', safe(p.id), 'sign', 'report.json'));
  const expect = restirSettings(p.preset, settings);
  const t16 = t16M6(run.meta, ref.meta, { plant: true, rounds: expect.rounds, lightMode: p.lightMode, expect });
  const fullNotPass = !!full && full.status !== 'pass';
  const ok = !!rep?.ok && fullNotPass && t16.length === 0;
  const f0 = rep?.frames?.[0];
  Object.assign(data, {
    ok, status: ok ? 'detected, sign holds' : rep ? `detected ${rep.detected}, sign ${rep.signs_hold}, full-not-pass ${fullNotPass}` : `plant_sign exit ${r.code}`,
    seeds: { restir: pseed, ptCalib: calibSeed, ...(seedOffset ? { note: 'fresh disjoint seeds (revised prediction, E-18)' } : {}) },
    sizing: { frames: z.frames, B: z.B, ptSpp4x: z.ptSpp * CALIB_FACTOR, tile: z.tile },
    detection: f0 && `${f0.detection.gate_fail_count}/${f0.detection.n_repeats} (control ${f0.detection.control_pass_count}/${f0.detection.n_repeats})`,
    predictions: f0?.predictions?.map((x: any) => ({ region: x.region, sign: x.sign, rel: r4(x.rel), z: r4(x.z), holds: x.holds })),
    global_Y: f0 && { rel: r4(f0.global_Y.rel), z: r4(f0.global_Y.z) }, full_compare: full && { status: full.status, global_rel_Y: r4(full.channels.Y.global_.rel), failed: full.failed_checks },
    t16, report: path.join(dir, 'plants', safe(p.id), 'sign', 'report.json'),
  });
  add(`plant ${p.id}: ${p.name} on ${p.pkg} (${p.predict.map((x) => `${x.region} ${x.sign}`).join(', ')})`, ok, (performance.now() - t0) / 1000, data,
    `${data.status}; detection ${data.detection}; ${(data.predictions ?? []).map((x: any) => `${x.region} ${pct(x.rel, 3)} z ${x.z}`).join(', ')}${t16.length ? `; T16: ${t16.join('; ')}` : ''}`);
  return data;
}

/** A/A (6502 vs 6503 at 4× the unit size) + synthetic W × 1.003 with calibrate A/A re-splits on (i) offline-m6. */
function aaAndSynthetic(dir: string, runId: string, nU: number, add: Add): Record<string, any>[] {
  const t0 = performance.now();
  const u = SEQ_UNITS.find((x) => x.id === `${AA_PKG}@3.7-all`)!;
  const z = sizeUnit(u.pkg, u.preset, u.settings, 'm4', 'tight', add, u.id);
  if (!z) return [{ unit: 'aa-restir', kind: 'aa', ok: false, status: 'sizing failed' }];
  const ref = ptRefRun(u.pkg, 'm4', z.ptSpp, z.B, M4_PT.ref, 1.2 * z.ptSeconds, add);
  const frames = z.frames * CALIB_FACTOR;
  const runs = [SEEDS.aa, SEEDS.aa2].map((seed) => runBatches([...rsArgs(u.pkg, u.preset, u.settings), '--spp', String(frames), '--batches', String(z.B), '--seed', String(seed)],
    `${runId}-aa-${seed}`, path.join(dir, 'restir', `aa-${seed}`), { B: z.B, estSeconds: z.seconds * CALIB_FACTOR * 1.2 }));
  const out: Record<string, any>[] = [];
  const expect = restirSettings(u.preset, u.settings ?? {});
  if (runs.every((r) => r.dir && r.meta) && ref) {
    const t16 = runs.flatMap((r) => t16M6(r.meta!, ref.meta, { rounds: 3, lightMode: 'A', expect }));
    const test = writeTest(dir, 'aa-restir', nU, 'tight', z.tile);
    const o = path.join(dir, 'compare', 'aa-restir');
    compare(runs[1].dir!, runs[0].dir!, test, o);
    const rep = tryJson(path.join(o, 'report.json'));
    const ok = rep?.status === 'pass' && t16.length === 0;
    const res: Record<string, any> = { unit: `aa-restir-${u.pkg}`, kind: 'aa', part: 'plants', ok, ...(rep ? summarize(rep) : { status: 'error' }), seeds: `${SEEDS.aa2} vs ${SEEDS.aa}`, restir: `${frames}x${z.B} (x${CALIB_FACTOR})`, t16 };
    out.push(res);
    add(`A/A: two ReSTIR seed sets on ${u.pkg} offline-m6 (${SEEDS.aa} vs ${SEEDS.aa2}, ${frames} frames x ${z.B})`, ok, (performance.now() - t0) / 1000, res, `${res.status}: Δ_Y ${pct(res.global_rel_Y as number, 4)}`);
    const t1 = performance.now();
    const ctest = writeTest(dir, 'plant-W1.003', nU, 'tight', z.tile);
    const co = path.join(dir, 'calibrate', 'W1.003');
    const r = sh(PY, ['validation/tools/compare.py', '--calibrate', '--ref', runs[0].dir!, '--test', ctest, '--out', co, '--splits', '20', '--repeats', '10'], () => false);
    const crep = tryJson(path.join(co, 'report.json'));
    const aa = crep?.calibration?.aa, pl = crep?.calibration?.plants?.[0];
    const ok2 = r.code === 0 && !!aa?.ok && !!pl?.calibrated;
    const res2 = { unit: `plant-W1.003-${u.pkg}`, kind: 'plant', part: 'plants', plant: 'W x1.003 (synthetic)', ok: ok2, status: ok2 ? 'detected' : 'not detected',
      detected: pl && `${pl.gate_fail_count}/${pl.n_repeats}`, control_pass: pl && `${pl.control_pass_count}/${pl.n_repeats}`, aa_resplits: aa && { ok: aa.ok, gate_pass_rate: aa.gate_pass_rate } };
    out.push(res2);
    add(`synthetic W x1.003 + calibrate A/A re-splits on the ${u.pkg} offline-m6 A/A run`, ok2, (performance.now() - t1) / 1000, res2, `detected ${res2.detected}, control ${res2.control_pass}, A/A ${aa?.ok ? 'ok' : 'FAIL'}`);
  } else out.push({ unit: 'aa-restir', kind: 'aa', ok: false, status: 'run failed' });
  return out;
}

// ------------------------------------------------------------------------------------------------ Gate 0

function gate0(dir: string, runId: string, add: Add, runStep: (name: string, cmd: string, argv: string[], echo?: (l: string) => boolean, env?: Record<string, string>) => { code: number; out: string }): void {
  runStep('typecheck', 'npx', ['tsc', '--noEmit', '-p', '.']);
  runStep('vitest cpu (U-WGSL-BITS, T14, plant-m6 predictions, layout / inline budget, gate configs + regressions)', 'npx', ['vitest', 'run', '--project', 'cpu'], (l) => /Test Files|Tests |FAIL|✗|×/.test(l));
  runStep('python tests (stats, compare, dynamic, plant_sign, dup_bias)', PY, ['-m', 'pytest', 'validation/tools/tests', '-q'], (l) => /passed|failed|error/i.test(l));
  packagesDeterministic(dir, add);
  const vit = (file: string, extra: string[]) => ['vitest', 'run', '--project', 'chrome', '--reporter=verbose', `validation/gpu-tests/${file}.gpu.test.ts`, ...extra];
  const echo = (l: string) => /Tests |FAIL|✗|×|AssertionError|LOGIC|FP-BOUNDARY|violation|T3-M6/.test(l);
  // M6 suites
  for (const s of M6_GPU_SUITES) {
    withGpuLockSync(`gate-m6-${s.file}`, () => { runStep(`${s.file} (chrome) [M6]: ${s.what}`, 'npx', vit(s.file, s.t ? ['-t', s.t] : []), echo, s.env); });
  }
  for (const v of T3M6_RUNS) {   // one hold per T3-M6 variant (T3-0 at T3_MS/3 + T3-1 at T3_MS ≤ 24 min, E-16)
    withGpuLockSync(`gate-m6-t3m6-${v}`, () => {
      runStep(`restir-shift (chrome) [T3-M6 ${v}]: LOGIC 0, FP <= 1e-5, every new bin / counter >= 1e6`, 'npx',
        vit('restir-shift', ['--testTimeout', '3600000', '-t', `${reEsc(T3M6_DESCRIBE)}.* ${reEsc(v)}$`]), echo, { VITE_T3_MS: String(18 * 60_000) });
    });
  }
  // M4 / M5 Gate-0 suites (regressions; restir-shift split per M4 T3 variant, ≤ 24 min each, E-16)
  const seen = new Set<string>();
  for (const [file, what] of [...M4_SUITES, ...M5_SUITES]) {
    if (seen.has(file)) continue;
    seen.add(file);
    const rel = `validation/gpu-tests/${file}.gpu.test.ts`;
    if (!existsSync(path.join(ROOT, rel))) { add(`${file} (chrome): ${what}`, false, 0, undefined, `missing ${rel}`); continue; }
    for (const h of gpuSuiteHolds(file)) {
      // the M6 T3 runs above are not part of the M4 regression holds
      // the M6 T3 runs and the Mode-B T3-2 cases have their own holds above
      const args = file === 'restir-shift' && h.label === ' [all but the T3 variants]' ? ['-t', '^(?!.*(T3-0 / T2 / T3-1 / T4 / T3-D / U5 / T3-ENV / U-12 on the t3 fixtures|T3-M6))']
        : file === 'restir-temporal' ? ['-t', '^(?!.*Mode B: )'] : h.args;
      withGpuLockSync(`gate-m6-${file}`, () => { runStep(`${file} (chrome)${h.label} [M4/M5 regression]: ${what}`, 'npx', vit(file, args), echo, GPU_SUITE_ENV[file]); });
    }
  }
  for (const c of T32_RARE_CASES) {
    withGpuLockSync(`gate-m6-t32-${safe(c)}`, () => {
      runStep(`restir-temporal T3-2 ${c} [M5 regression] (>= 1e6 per bin)`, 'npx', vit('restir-temporal', ['--testTimeout', '900000', '-t', c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')]), echo, T32_RARE_ENV);
    });
  }
  for (const g of M5_GATE0_PLANTS) add(`${g.id} activity: covered by the restir-temporal suite above`, true, 0, undefined, g.status);
  if (existsSync(path.join(ROOT, 'validation/harness/m6-app-smoke.ts'))) {
    runStep('M6 app smoke (interactive M6 features compile, Mode B default, pairing / dup views 471-479, feature toggles)', 'npx', ['tsx', 'validation/harness/m6-app-smoke.ts', '--run', `${runId}-app-smoke`], (l) => /^(PASS|FAIL)\s/.test(l));
  } else add('M6 app smoke (validation/harness/m6-app-smoke.ts)', false, 0, undefined, 'missing');
}

/** make-m6.ts writes byte-identical packages twice (incl. its make-m3c variants). */
function packagesDeterministic(dir: string, add: Add): void {
  const t0 = performance.now();
  const a = path.join(dir, 'make-m6-a'), b = path.join(dir, 'make-m6-b');
  const ra = sh('npx', ['tsx', 'validation/scenes/make-m6.ts', a], () => false), rb = sh('npx', ['tsx', 'validation/scenes/make-m6.ts', b], () => false);
  const list = (d: string): string[] => {
    const out: string[] = [];
    const walk = (x: string) => { for (const n of readdirSync(path.join(ROOT, x)).sort()) { const q = path.join(x, n); if (statSync(path.join(ROOT, q)).isDirectory()) walk(q); else out.push(path.relative(d, q)); } };
    if (existsSync(path.join(ROOT, d))) walk(d);
    return out;
  };
  const fa = list(a), fb = list(b), diffs: string[] = [];
  if (ra.code !== 0 || rb.code !== 0) diffs.push(`exit ${ra.code}/${rb.code}: ${(ra.out + rb.out).slice(-300)}`);
  for (const f of new Set([...fa, ...fb])) if (!fa.includes(f) || !fb.includes(f) || !readFileSync(path.join(ROOT, a, f)).equals(readFileSync(path.join(ROOT, b, f)))) diffs.push(f);
  // the gate's packages must be the generated bytes
  for (const f of fa) if (!existsSync(path.join(ROOT, M6_SCENE_DIR, f)) || !readFileSync(path.join(ROOT, a, f)).equals(readFileSync(path.join(ROOT, M6_SCENE_DIR, f)))) diffs.push(`${M6_SCENE_DIR}/${f} differs from the generator`);
  if (!fa.length) diffs.push('no files written');
  rmSync(path.join(ROOT, a), { recursive: true, force: true });
  rmSync(path.join(ROOT, b), { recursive: true, force: true });
  add('make-m6.ts package determinism (twice, byte-identical, = validation/out/m6/scenes)', diffs.length === 0, (performance.now() - t0) / 1000, { files: fa.length, diffs: diffs.slice(0, 20) },
    diffs.length ? `diffs: ${diffs.slice(0, 6).join(', ')}` : `${fa.length} files`);
}

function ensurePackages(add: Add): void {
  const missing = M6_PKGS.filter((p) => !existsSync(path.join(ROOT, M6_SCENE_DIR, p, 'scene.json')));
  if (!missing.length) { add(`M6 scene packages present (${M6_PKGS.length}, make-m6.ts)`, true, 0); return; }
  const r = sh('npx', ['tsx', 'validation/scenes/make-m6.ts', M6_SCENE_DIR, '--only', missing.join(',')], () => false);
  const still = missing.filter((p) => !existsSync(path.join(ROOT, M6_SCENE_DIR, p, 'scene.json')));
  add(`M6 scene packages generated (${missing.join(', ')})`, r.code === 0 && !still.length, r.seconds, undefined, still.length ? `missing ${still.join(', ')}` : undefined);
}

// ------------------------------------------------------------------------------------------------ the gate

export interface M6Options { part?: Part; only?: Set<string>; pilotOnly?: boolean; plantSeedOffset?: number; writeBudget?: boolean }

export function milestoneM6(record: Rec, o: M6Options = {}): void {
  const t0 = performance.now();
  const parts = o.part ? [o.part] : PARTS;
  const runId = `m6-gate-${o.part ?? 'all'}-${stamp()}`;
  const dir = path.join('validation/out', runId);
  mkdirSync(path.join(ROOT, dir), { recursive: true });
  const steps: { name: string; ok: boolean; seconds: number; data?: unknown; detail?: string }[] = [];
  const add: Add = (name, ok, seconds, data, detail) => {
    steps.push({ name, ok, seconds: Math.round(seconds * 10) / 10, data, detail });
    record(name, ok, `${detail ? `${detail}, ` : ''}${seconds.toFixed(1)} s`);
  };
  const runStep = (name: string, cmd: string, argv: string[], echo?: (l: string) => boolean, env?: Record<string, string>) => {
    console.log(`\n--- ${name}: ${env ? `${Object.entries(env).map(([k, v]) => `${k}=${v}`).join(' ')} ` : ''}${cmd} ${argv.join(' ')}`);
    const r = sh(cmd, argv, echo, env);
    const log = path.join(dir, 'logs', `${safe(name).slice(0, 80)}.log`);
    mkdirSync(path.join(ROOT, dir, 'logs'), { recursive: true });
    writeFileSync(path.join(ROOT, log), r.out);
    const noTests = argv[0] === 'vitest' && !/\d+ passed/.test(r.out);
    add(name, r.code === 0 && !noTests, r.seconds, { log, ...(env ? { env } : {}) }, `exit ${r.code}${noTests ? ', no test ran' : ''}`);
    return r;
  };
  const sel = (id: string, pkg: string) => !o.only || o.only.has(id) || o.only.has(pkg);
  const nU = nUnits();
  const hashes = codeHashes();
  console.log(`M6 gate ${runId}: parts ${parts.join(',')}, n_units ${nU}, PT code ${hashes.pt.slice(0, 12)}, ReSTIR ${hashes.restir.slice(0, 12)}`);
  add('M6 gate tools (venv python)', existsSync(PY), 0, undefined, existsSync(PY) ? undefined : `missing ${PY}`);
  ensurePackages(add);
  const results: Record<string, any>[] = [];
  const sizing: Record<string, unknown> = {};
  if (parts.includes('core') && !o.only && !o.pilotOnly) gate0(dir, runId, add, runStep);
  for (const part of parts.filter((p) => p !== 'core')) {
    for (const u of SEQ_UNITS.filter((x) => x.part === part && sel(x.id, x.pkg))) results.push(seqUnit(u, dir, runId, nU, add, !!o.pilotOnly));
    const cu = CHAIN_UNITS.filter((x) => x.part === part && sel(x.id, x.pkg));
    if (cu.length) {
      const r = chainUnits(cu, dir, runId, nU, add, !!o.pilotOnly);
      results.push(...r.results);
      sizing[part] = r.sizing;
      if (part === 'gate5' && !o.pilotOnly) results.push(...gate5Metrics(cu, r.results, dir, add));
    }
    if (part === 'plants' && !o.pilotOnly) {
      PLANTS_M6.forEach((p, i) => { if (sel(p.id, p.pkg)) results.push(plantUnit(p, i, o.plantSeedOffset ?? 0, dir, runId, nU, add)); });
      if (!o.only || o.only.has('aa')) results.push(...aaAndSynthetic(dir, runId, nU, add));
    }
  }
  writeFileSync(path.join(ROOT, dir, 'sizing.json'), `${JSON.stringify({ seq: results.filter((r) => r.sizing).map((r) => ({ unit: r.unit, ...r.sizing })), chains: sizing }, null, 1)}\n`);
  const rows = budgetRows(results, sizing);
  writeFileSync(path.join(ROOT, dir, 'budget-m6.json'), `${JSON.stringify(rows, null, 1)}\n`);
  if (o.writeBudget && rows.length) mergeBudget(rows, runId, add);
  const failed = steps.filter((x) => !x.ok).map((x) => x.name);
  const kinds = ['seq', 'chain-static', 'chain-dyn', 'gate5', 'gate5-twin', 'plant', 'aa'];
  const perPart = Object.fromEntries(PARTS.filter((p) => p !== 'core').map((p) => {
    const u = results.filter((x) => x.part === p && x.kind !== 'gate5-on');
    return [p, { pass: u.filter((x) => x.ok).length, fail: u.filter((x) => !x.ok && x.status !== 'not run').length, notRun: u.filter((x) => x.status === 'not run').length }];
  }));
  const summary = {
    milestone: 'M6', gate: 'Gate 0 + rungs 3.7 / 3.9 / 3.10 / 3.11 + Gate 5 + M6 plants with predicted signs (restir-m6-api.md §4–§5)',
    run: runId, parts, created: new Date().toISOString(), ok: failed.length === 0, subset: o.only ? [...o.only] : undefined, pilotOnly: !!o.pilotOnly,
    total_s: Math.round((performance.now() - t0) / 100) / 10, n_units: nU, code_hashes: hashes, per_part: perPart, kinds, failed, units: results, steps,
  };
  writeFileSync(path.join(ROOT, dir, 'summary.json'), `${JSON.stringify(summary, null, 1)}\n`);
  writeFileSync(path.join(ROOT, dir, 'summary.md'), summaryMd(summary, results));
  console.log(`\n${table(results)}\nsummary: ${dir}/summary.json (${summary.total_s} s)`);
}

/** budget.json M6 rows: per sequential unit (PT / ReSTIR sizes and minutes) and per chain unit (R, minutes). */
function budgetRows(results: Record<string, any>[], chainSizing: Record<string, any>): Record<string, unknown>[] {
  const rows: Record<string, unknown>[] = [];
  for (const r of results.filter((x) => x.kind === 'seq' && x.sizing)) {
    const z = r.sizing;
    rows.push({ unit: r.unit, part: r.part, rung: r.rung, scene: r.scene, preset: r.preset, settings: r.settings, light_mode: r.lightMode, B: z.B, pt_spp: z.ptSpp, pt_min: z.pt_min,
      frames_per_batch: z.frames, restir_min: z.restir_min, tile: z.tile, measured_restir_min: r.restir_minutes });
  }
  for (const [part, zs] of Object.entries(chainSizing)) {
    for (const u of (zs as any).units ?? []) rows.push({ unit: u.id, part, kind: 'chains', R: u.R, frames: u.frames, test_frames: u.testFrames, tile: u.tile, chain_min: u.minutes, ms_per_chain: u.msPerChain, cap: u.cap });
  }
  return rows;
}
function mergeBudget(rows: Record<string, unknown>[], runId: string, add: Add): void {
  const p = path.join(ROOT, 'validation/budget.json');
  const b = JSON.parse(readFileSync(p, 'utf8')) as Record<string, any>;
  const keep = (b.m6_entries ?? []).filter((e: any) => !rows.some((r) => r.unit === e.unit));
  b.m6_method = 'M6 sizing (gate-m6.ts): PT pilot 128 spp x B and ReSTIR pilot of the unit configuration (offline presets 8 frames, initial 128) x B in headless Chrome; chain units by gate-m5 sizeGroup (64-chain pilots); PLAN §7.3 rule x1.25';
  b.m6_measured_at = new Date().toISOString();
  b.m6_runs = [...new Set([...(b.m6_runs ?? []), runId])];
  b.m6_entries = [...keep, ...rows];
  writeFileSync(p, `${JSON.stringify(b, null, 2)}\n`);
  add('budget.json M6 rows written (--write-budget)', true, 0, { rows: b.m6_entries.length });
}

function table(results: Record<string, any>[]): string {
  const head = '| unit | kind | rung | status | Δ_Y | MDB_Y | worst tile | PT | ReSTIR / R | tile | mult | min |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n';
  return head + results.map((r) => `| ${r.unit} | ${r.kind} | ${r.rung ?? ''} | ${r.status}${r.t16?.length ? ' (T16)' : ''} | ${pct(r.global_rel_Y ?? r.global_rel, 4)} | ${pct(r.mdb_global_Y, 4)} | ${pct(r.worst_tile_rel_Y, 2)} | ${r.pt ?? ''} | ${r.restir ?? r.R ?? ''} | ${r.tile ?? ''} | ${r.multiplier_needed ?? ''} | ${r.restir_minutes ?? r.chain_minutes ?? ''} |`).join('\n') + '\n';
}
function summaryMd(summary: Record<string, any>, results: Record<string, any>[]): string {
  return [
    `# M6 gate ${summary.run}`, '', `Result: **${summary.ok ? 'PASS' : 'FAIL'}** (${summary.total_s} s, parts ${summary.parts.join(', ')}, n_units ${summary.n_units})`, '',
    '## Per part', '', '| part | pass | fail | not run |', '|---|---|---|---|',
    ...Object.entries(summary.per_part as Record<string, any>).map(([k, v]) => `| ${k} | ${v.pass} | ${v.fail} | ${v.notRun} |`), '',
    '## Units', '', table(results), '## Failed steps', '', ...(summary.failed as string[]).map((f) => `- ${f}`), '',
  ].join('\n');
}

