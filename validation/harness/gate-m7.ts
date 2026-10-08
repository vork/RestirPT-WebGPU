// M7 milestone gate (docs/decisions/m7-api.md §6; PLAN §5 M7 exit, §7.1 rung 3.8, §7.2 (vii-N) / (vii-L) / (viii-L) /
// E2E-GLB / E2E-USD / E2E-HDR):
// `npm run validate -- --milestone M7 [--part core|loader|stageA|e2e|r38|plants] [--only id,pkg] [--pilot-only] [--write-budget]`.
//   core    Gate 0: typecheck, cpu lane (U-M7-BITS, U-M7-ARENA, tangents, plant-m7 predictions + every earlier CPU test),
//           python tests, make-m7.ts / make-m7-e2e.ts determinism (twice, byte-identical, = validation/out/m7), the Chrome
//           suites of M7 (normal-map: U-NM-1..3) and the T3-M7 runs of restir-shift (one GPU-lock hold each), the M6 / M5 /
//           M4 Gate-0 suites as regressions (the gate-m6 holds: restir-m6, T3-M6 variants, every M4 / M5 suite, T3-2
//           rare bins), the M7 perf probe (540p interactive with / without normal maps; recorded) and the M7 app smoke.
//   loader  (vii-L) our glTF loader vs Blender 5.2.2's stock importer, (viii-L) our USD loader vs an OpenUSD (pxr) dump
//           (m7-loader-fidelity.ts), U-TAN-B package tangents vs Blender's MikkTSpace (m7-tangent-check.ts).
//   stageA  (vii-N) our PT ≡ Cycles 5.2.2 (TOST δ 0.5 % global / 2 % per 32² tile, Y/R/G/B): m7_nm_flat_256 (tight),
//           the smooth / smooth normal-mapped scenes (model-approximate: TOST gates at the unchanged δ, the Δ = 0
//           rejection checks are reported, not gating; m7-api.md §6.1), E2E-HDR (xiv)-lite with Blender loading the
//           original Poly Haven .hdr / .exr (tight; the bridge records Blender's pixels = ours) and a PT A/A.
//   e2e     E2E-GLB (stock import_scene.gltf SPEC, FLAT and NORMALS) and E2E-USD (stock wm.usd_import; our loader in
//           Blender-compatible mode) on the 21 make-m7-e2e.ts packages, Stage A in the model-approximate tier.
//   r38     rung 3.8: ReSTIR ≡ our PT (Stage B, δ 0.2 % / 1 %) on smooth and normal-mapped scenes: initial + offline-m6
//           (Mode A and B, with env), chains full-m6 t ∈ {1, 24}; pilot sizing (PLAN §7.3) × 1.25 as gate-m6.
//   plants  predictions derived before measurement (validation/scenes/m7-plants.ts, tests/scene/plant-m7.test.ts):
//           Stage A (PT vs Cycles) and Stage B (ReSTIR vs 4× PT) Normal Map sign / strength plants on their panel masks,
//           the shading-normal shift-Jacobian plant (detect), synthetic W × 1.003 + calibrate A/A, a ReSTIR A/A.
// Statistics: compare.py; a failed unit is re-run once on disjoint seeds; δ is never loosened, nothing is clamped.
// Output: validation/out/m7-gate-<part>-<time>/ (summary.json/.md, sizing.json, budget-m7.json, tests/, compare/, pt/,
// restir/, plants/, loader/). Cycles references: validation/out/m7/refs (cached by package bytes + Blender settings).
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodePFM, encodePFM } from '../../src/core/io/pfm.ts';
import { restirSettings, type RestirPresetName, type RestirSettings } from '../../src/core/render/restir/presets.ts';
import { cachedRun, codeHashes, GPU_SUITE_ENV, GPU_SUITES as M4_SUITES, gpuSuiteHolds, packageHash, runBatches, sizeScene, type PilotSide, type Run } from './gate-m4.ts';
import { EXT, GPU_SUITES as M5_SUITES, runExternalChainUnits, T32_RARE_CASES, T32_RARE_ENV, type ExtChainUnit } from './gate-m5.ts';
import { M6_GPU_SUITES, T3M6_DESCRIBE, T3M6_RUNS, t16M6 } from './gate-m6.ts';
import { m7PlantPredictions } from '../scenes/m7-plants.ts';
import { withGpuLockSync } from './gpu-lock.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PY = path.join(ROOT, 'validation/.venv/bin/python');
const BLENDER = process.env.BLENDER ?? '/Applications/Blender.app/Contents/MacOS/Blender';
export const M7_OUT = 'validation/out/m7';
export const M7_SCENE_DIR = `${M7_OUT}/scenes`;
export const M7_E2E_DIR = `${M7_OUT}/e2e`;
const REFS = `${M7_OUT}/refs`;
const TIMEOUT_MS = 24 * 3600_000;
/** Seeds of the M7 gate (disjoint from M3–M6). Stage A: our PT 7007 / 107007, Cycles 0..K−1 / 100..100+K−1. */
export const SEEDS = {
  pt: 7001, ptRerun: 107001, restir: 7002, restirRerun: 107002, aa: 7502, aa2: 7503, plantBase: 7101, ptCalib: 7201, ptPilot: 7011, restirPilot: 7012,
  chains: 7002, chainsRerun: 107002, chainPilot: 7012, plantRevised: 7801, ours: 7007, oursRerun: 107007, oursAA: [7301, 7302], plantA: 7401, cyclesRerunBase: 100,
} as const;
const M7_PT = { root: M7_OUT, pilot: SEEDS.ptPilot, ref: SEEDS.pt, rerun: SEEDS.ptRerun } as const;
export const CALIB_FACTOR = 4;
export const NUM_EPS = 1e-4;
const NUM_EPS_NOTE = 'two f32 implementations (cycles-deviations.md D2; restir-api.md §6.3)';
/** The Δ = 0 rejection-type checks; informational in the model-approximate tier (m7-api.md §6.1). */
export const MODEL_APPROX_CHECKS = { sidak_tiles: false, chi2_red: false, mean_t: false, ks_ad: false, numeric_tiles: false } as const;

/** Packages written by make-m7.ts into validation/out/m7/scenes. */
export const M7_PKGS = ['m7_smooth_256', 'm7_smooth_lowpoly_256', 'm7_nm_flat_256', 'm7_nm_smooth_256', 'm7_nm_smooth_B_256', 'm7_nm_env_256', 'm7_xivlite_hdr_256', 'm7_xivlite_exr_256'];
/** Packages written by make-m7-e2e.ts into validation/out/m7/e2e (E2E_UNITS order). */
export const E2E_PKGS = [
  ...['cornell_point_spot', 'metalrough_spheres', 'texture_transform', 'normal_tangent_mirror', 'alpha_mask', 'emissive_strength', 'transmission', 'ior_grid']
    .flatMap((n) => [`e2e_glb_${n}_flat`, `e2e_glb_${n}_normals`]),
  ...['cornell', 'hand_yup', 'hand_zup', 'textured', 'instancing'].map((n) => `e2e_usd_${n}`),
];
export const pkgDirM7 = (pkg: string): string => (E2E_PKGS.includes(pkg) ? `${M7_E2E_DIR}/${pkg}` : `${M7_SCENE_DIR}/${pkg}`);
const pkgJson = (pkg: string): Record<string, any> => JSON.parse(readFileSync(path.join(ROOT, pkgDirM7(pkg), 'scene.json'), 'utf8'));
const pkgHasNormalMaps = (pkg: string): boolean => (pkgJson(pkg).materials as { normalTexture?: unknown }[]).some((m) => !!m.normalTexture);

// ------------------------------------------------------------------------------------------------ unit lists

export type Part = 'core' | 'loader' | 'stageA' | 'e2e' | 'r38' | 'plants';
export const PARTS: Part[] = ['core', 'loader', 'stageA', 'e2e', 'r38', 'plants'];
type Tier = 'tight' | 'model-approximate' | 'heavy-tail';

export interface StageAUnit { pkg: string; label: string; part: 'stageA' | 'e2e'; tier: Tier; cyclesSpp: number; K: number; ourSpp: number; B: number; hdr?: 'hdr' | 'exr' }
const A = (pkg: string, label: string, o: Partial<StageAUnit> = {}): StageAUnit => ({ pkg, label, part: 'stageA', tier: 'tight', cyclesSpp: 4096, K: 16, ourSpp: 4096, B: 16, ...o });
export const STAGE_A_UNITS: StageAUnit[] = [
  A('m7_nm_flat_256', '(vii-N) flat geometry + normal maps (tiles / bumps / waves; panels P1–P3)'),
  A('m7_smooth_256', 'smooth shading, no normal maps (open wavy sheet)', { tier: 'model-approximate' }),
  A('m7_smooth_lowpoly_256', 'low-poly smooth shading (Ns up to ~40° from Ng), no normal maps', { tier: 'model-approximate' }),
  A('m7_nm_smooth_256', '(vii-N) smooth + normal-mapped (bumps / waves / tiles torus / bumps sheet)', { tier: 'model-approximate' }),
  A('m7_nm_smooth_B_256', '(vii-N) smooth + normal-mapped, light mode B', { tier: 'model-approximate' }),
  A('m7_nm_env_256', '(vii-N) smooth normal-mapped spheres on a normal-mapped ground, overcast HDRI', { tier: 'model-approximate' }),
  A('m7_xivlite_hdr_256', 'E2E-HDR (xiv)-lite: Blender loads the original overcast_soil_puresky_1k.hdr', { hdr: 'hdr' }),
  A('m7_xivlite_exr_256', 'E2E-HDR (xiv)-lite: Blender loads the original overcast_soil_puresky_1k.exr', { hdr: 'exr' }),
  ...E2E_PKGS.map((p) => A(p, `E2E ${p.startsWith('e2e_usd') ? 'USD (stock wm.usd_import)' : `GLB (stock import_scene.gltf, ${p.endsWith('_flat') ? 'FLAT' : 'NORMALS'})`}`, { part: 'e2e', tier: 'model-approximate' })),
];
const AA_A_PKG = 'm7_nm_flat_256';

export interface SeqUnit { id: string; part: 'r38'; rung: string; pkg: string; label: string; preset: RestirPresetName; settings?: Partial<RestirSettings>; lightMode: 'A' | 'B' }
const U = (tag: string, pkg: string, label: string, preset: RestirPresetName, o: Partial<SeqUnit> = {}): SeqUnit =>
  ({ id: `${pkg}@3.8-${tag}`, part: 'r38', rung: '3.8', pkg, label, preset, lightMode: 'A', ...o });
export const SEQ_UNITS: SeqUnit[] = [
  U('initial', 'm7_smooth_256', 'smooth shading, initial candidates only', 'initial'),
  U('initial', 'm7_nm_smooth_256', 'smooth + normal maps, initial candidates only', 'initial'),
  U('offline', 'm7_smooth_256', 'smooth shading, offline-m6 (σ 16 + RIS-NEE, 3 spatial rounds)', 'offline-m6'),
  U('offline', 'm7_smooth_lowpoly_256', 'low-poly smooth shading (Ns up to ~40° from Ng), offline-m6', 'offline-m6'),
  U('offline', 'm7_nm_flat_256', 'flat + normal maps, offline-m6', 'offline-m6'),
  U('offline', 'm7_nm_smooth_256', 'smooth + normal maps, offline-m6', 'offline-m6'),
  U('offline', 'm7_nm_env_256', 'smooth + normal maps under the overcast HDRI, offline-m6', 'offline-m6'),
  U('offline-B', 'm7_nm_smooth_B_256', 'smooth + normal maps, offline-m6 in light mode B', 'offline-m6', { lightMode: 'B' }),
];
export interface ChainUnitM7 extends ExtChainUnit { label: string; lightMode: 'A' | 'B' }
export const CHAIN_UNITS: ChainUnitM7[] = [
  { id: 'm7_nm_smooth_256@3.8-full-m6', kind: 'static', pkg: 'm7_nm_smooth_256', rung: '3.8', preset: 'full-m6', extra: [], frames: 25, testFrames: [1, 24], rounds: 1, lightMode: 'A',
    label: 'chains full-m6 (temporal + spatial, σ 16 + RIS) on smooth + normal maps' },
];

/** Plants (m7-api.md §5). Stage A: our PT with NM_PLANT vs Cycles; Stage B: ReSTIR with the plant vs a 4× PT. */
export interface PlantM7 {
  id: string; name: string; stage: 'A' | 'B'; pkg: string; preset?: RestirPresetName; base?: Partial<RestirSettings>; plant: Record<string, unknown>; ptArgs?: string[];
  predict: { region: string; sign: '+' | '-' | 'detect' }[]; masks: ('M_P2' | 'M_P3')[]; derivation: string;
  /** Revised plants run on fresh disjoint seeds (E-18): ReSTIR seedBase, 4× PT seedBase + 100. */
  seedBase?: number;
}
export const PLANT_IDS = ['A-NM-sign', 'A-NM-strength', 'B-NM-sign', 'B-NM-strength', 'B-SM-J'] as const;
export function plantList(): PlantM7[] {
  const pred = m7Predictions();
  const d = (k: 'sign' | 'strength') => `${pred[k].derivation} (validation/scenes/m7-plants.ts)`;
  return [
    { id: 'A-NM-sign', name: 'Normal Map node: bitangent sign ignored (NM_PLANT 1), our PT', stage: 'A', pkg: 'm7_nm_flat_256', plant: { nm: 'sign' }, ptArgs: ['--plant-nm', 'sign'],
      predict: [{ region: 'M_P2', sign: pred.sign.sign }], masks: ['M_P2', 'M_P3'], derivation: d('sign') },
    { id: 'A-NM-strength', name: 'Normal Map node: glTF-style strength (c.z not mixed; NM_PLANT 2), our PT', stage: 'A', pkg: 'm7_nm_flat_256', plant: { nm: 'strength' }, ptArgs: ['--plant-nm', 'strength'],
      predict: [{ region: 'M_P3', sign: pred.strength.sign }], masks: ['M_P2', 'M_P3'], derivation: d('strength') },
    { id: 'B-NM-sign', name: 'Normal Map bitangent sign ignored in every ReSTIR pass but the path tree (m7NmSign)', stage: 'B', pkg: 'm7_nm_flat_256', preset: 'offline-m6', plant: { m7NmSign: true },
      predict: [{ region: 'M_P2', sign: pred.sign.sign }], masks: ['M_P2', 'M_P3'], derivation: `${d('sign')}; shifted paths through P2 evaluated with the flipped normal` },
    { id: 'B-NM-strength', name: 'glTF-style strength in every ReSTIR pass but the path tree (m7NmStrength)', stage: 'B', pkg: 'm7_nm_flat_256', preset: 'offline-m6', plant: { m7NmStrength: true },
      predict: [{ region: 'M_P3', sign: pred.strength.sign }], masks: ['M_P2', 'M_P3'], derivation: `${d('strength')}; shifted paths through P3 evaluated with the steeper normal` },
    // M7-11: the first B-SM-J ran on m7_smooth_256 (finely tessellated, Ns ≈ Ng) and was NOT detected (0/10, Δ −0.000 %);
    // revised before any run on the low-poly scene, measured on fresh disjoint seeds
    { id: 'B-SM-J', name: 'shading normal in the shift Jacobian geometry term (RS_PLANT_SMOOTH_J)', stage: 'B', pkg: 'm7_smooth_lowpoly_256', preset: 'offline-m6', plant: { m7SmoothJ: true },
      predict: [{ region: 'global', sign: 'detect' }], masks: [], seedBase: SEEDS.plantRevised,
      derivation: 'J uses |cos| at the reconnection vertex w.r.t. Ns instead of Ng: ratio cos_s/cos_g ≠ 1 wherever Ns ≠ Ng (up to ~40° on the low-poly meshes; sign varies with the side; detect only)' },
  ];
}
/** Read at use: the derivation reads validation/out/m7/scenes/m7_nm_flat_256/scene.json (ensurePackages first). */
const m7Predictions = () => m7PlantPredictions();

/** Gate-0 regression holds, (vii-L) / (viii-L) inputs, U-TAN-B packages. */
export const T3M7_DESCRIBE = 'T3-M7: T3-0 / T3-1 / T4 / T3-D LOGIC = 0 on smooth and normal-mapped reconnections';
export const T3M7_RUNS = ['t3_smooth_256 (smooth shading)', 't3_nm_256 (normal maps + smooth shading)'];
const K = 'validation/assets/downloaded/khronos';
export const GLB_FIDELITY = [
  'validation/assets/cornell/cornell_point_spot.glb', `${K}/MetalRoughSpheresNoTextures/MetalRoughSpheresNoTextures.glb`, `${K}/TextureTransformTest/TextureTransformTest.gltf`,
  `${K}/NormalTangentMirrorTest/NormalTangentMirrorTest.glb`, `${K}/AlphaBlendModeTest/AlphaBlendModeTest.glb`, `${K}/EmissiveStrengthTest/EmissiveStrengthTest.glb`,
  `${K}/TransmissionTest/TransmissionTest.glb`, `${K}/IORTestGrid/IORTestGrid.glb`, 'validation/assets/downloaded/sponza/Sponza.gltf',
];
export const USD_FIDELITY = ['validation/assets/cornell/cornell.usda', 'validation/assets/usd-m7/m7_textured.usda', 'validation/assets/usd-m7/m7_instancing.usda',
  'validation/assets/usd-m7/m7_e2e_yup.usda', 'validation/assets/usd-m7/m7_e2e_zup.usda', 'validation/assets/usd-spike/spike_hand.usda', 'validation/assets/usd-spike/spike_blender.usda'];
export const TANGENT_PKGS = ['m7_nm_flat_256', 'm7_nm_smooth_256', 'm7_nm_env_256', 'e2e_glb_normal_tangent_mirror_normals', 'e2e_usd_textured'];

export function nUnits(): number {
  const chains = CHAIN_UNITS.reduce((a, u) => a + u.testFrames.length, 0);
  return 4 * (STAGE_A_UNITS.length + 1 + SEQ_UNITS.length + chains + PLANT_IDS.length + 2);
}

// ------------------------------------------------------------------------------------------------ helpers

const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const r4 = (x: unknown) => (typeof x === 'number' ? Number(x.toPrecision(4)) : x);
const pct = (x: number | undefined | null, d = 3) => (typeof x === 'number' ? `${(x * 100).toFixed(d)}%` : 'n/a');
const tryJson = (p: string): Record<string, any> | undefined => { try { return JSON.parse(readFileSync(path.resolve(ROOT, p), 'utf8')); } catch { return undefined; } };
const safe = (s: string) => s.replace(/[^\w.@-]+/g, '_');
const json = (o: unknown) => JSON.stringify(o);
type Add = (name: string, ok: boolean, seconds: number, data?: unknown, detail?: string) => void;
type Rec = (name: string, ok: boolean, detail?: string) => void;

function sh(cmd: string, argv: string[], echo: (l: string) => boolean = () => true, env?: Record<string, string>): { code: number; out: string; seconds: number } {
  const t0 = performance.now();
  const r = spawnSync(cmd, argv, { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 28, timeout: TIMEOUT_MS, ...(env ? { env: { ...process.env, ...env } } : {}) });
  const out = `${r.stdout ?? ''}${r.stderr ?? ''}`;
  for (const l of out.split('\n')) if (l && echo(l)) console.log(`  ${l}`);
  return { code: r.status ?? 1, out, seconds: (performance.now() - t0) / 1000 };
}

function writeTest(dir: string, name: string, stage: 'A' | 'B', nU: number, tier: Tier, extra: Record<string, unknown> = {}): string {
  const p = path.join(dir, 'tests', `test-${safe(name)}.json`);
  mkdirSync(path.join(ROOT, dir, 'tests'), { recursive: true });
  writeFileSync(path.join(ROOT, p), JSON.stringify({
    name, stage, channels: ['Y', 'R', 'G', 'B'], n_units: nU, tier, num_eps: NUM_EPS, num_eps_note: NUM_EPS_NOTE,
    min_replicates: tier === 'heavy-tail' ? 32 : 16, ...(tier === 'model-approximate' ? { checks: MODEL_APPROX_CHECKS } : {}), ...extra,
  }, null, 1));
  return p;
}
function compare(ours: string, ref: string, test: string, out: string, rerunOf?: string) {
  return sh(PY, ['validation/tools/compare.py', '--ours', ours, '--ref', ref, '--test', test, '--out', out, ...(rerunOf ? ['--rerun-of', rerunOf] : [])], () => false);
}
function summarize(rep: Record<string, any>) {
  const Y = rep.channels.Y;
  // informational in the model-approximate tier: the rejection checks that did not pass (not gating)
  const info = (rep.checks as { name: string; channel: string; passed: boolean; gating: boolean }[] | undefined)?.filter((c) => !c.gating && !c.passed).map((c) => `${c.name}[${c.channel}]`) ?? [];
  return {
    status: rep.status, failed_checks: rep.failed_checks, tier: rep.tier, ...(info.length ? { informational_failed: info } : {}),
    global_rel_Y: r4(Y.global_.rel), mdb_global_Y: r4(Y.global_.mdb), worst_tile_rel_Y: r4(Y.tiles.worst_rel), worst_tile: Y.tiles.worst_tile ? `[${Y.tiles.worst_tile.join(',')}]` : undefined,
    mdb_tile_max_Y: r4(Y.tiles.mdb_max), tost_failed_tiles: Y.tiles.tost_failed,
    multiplier_needed: r4(Math.max(...['Y', 'R', 'G', 'B'].map((c) => Math.max(rep.channels[c].global_.replicate_multiplier_needed ?? 1, rep.channels[c].tiles.replicate_multiplier_needed ?? 1)))),
    per_channel: Object.fromEntries(['R', 'G', 'B'].map((c) => [c, { global_rel: r4(rep.channels[c].global_.rel), worst_tile_rel: r4(rep.channels[c].tiles.worst_rel) }])),
  };
}
const verdict = (sum: ReturnType<typeof summarize>) => `${sum.status}: Δ_Y ${pct(sum.global_rel_Y as number, 4)} (MDB ${pct(sum.mdb_global_Y as number, 3)}), worst tile ${pct(sum.worst_tile_rel_Y as number, 2)} ${sum.worst_tile ?? ''}, mult ${sum.multiplier_needed}`
  + `${sum.failed_checks?.length ? `, failed ${sum.failed_checks.join(' ')}` : ''}${sum.informational_failed?.length ? `; informational (model-approximate) ${sum.informational_failed.join(' ')}` : ''}`;

/** M7 T16 (m7-api.md §6.4): the normal-map state recorded in meta.json equals the package's. */
function t16M7(meta: Record<string, any> | undefined, pkg: string, side: string): string[] {
  if (!meta) return [`${side}: no meta.json`];
  const want = pkgHasNormalMaps(pkg);
  return meta.scene?.normalMaps === want ? [] : [`${side}: scene.normalMaps ${meta.scene?.normalMaps} (package: ${want})`];
}

// ------------------------------------------------------------------------------------------------ Stage A (PT vs Cycles)

function renderRef(pkg: string, spp: number, seeds: string): { code: number; out: string; seconds: number } {
  return sh(BLENDER, ['-b', '--factory-startup', '--python-exit-code', '1', '-P', 'validation/blender/render_reference.py', '--',
    '--package', pkgDirM7(pkg), '--out', REFS, '--spp', String(spp), '--seeds', seeds],
  (l) => l.startsWith('[render_reference]') && (l.includes('RESULT') || l.includes('cache') || l.includes('waited')) || /Error|Traceback/.test(l));
}
function cyclesRef(u: StageAUnit, base: number, add: Add): { dir: string; manifest: Record<string, any> } | undefined {
  const seeds = `${base}..${base + u.K - 1}`;
  console.log(`\n--- Cycles reference ${u.pkg}: ${u.cyclesSpp} spp × seeds ${seeds}`);
  let r = renderRef(u.pkg, u.cyclesSpp, seeds);
  if (r.code !== 0 && !r.out.includes('[render_reference] RESULT ')) { console.log('  Blender aborted without a result; retrying once'); r = renderRef(u.pkg, u.cyclesSpp, seeds); }
  const line = r.out.split('\n').reverse().find((l) => l.startsWith('[render_reference] RESULT '));
  const res = line ? JSON.parse(line.slice('[render_reference] RESULT '.length)) as { dir: string; cache_hit: boolean; renders: number; renders_total_s?: number } : undefined;
  add(`Cycles reference ${u.pkg} (${u.cyclesSpp} spp × ${seeds})`, r.code === 0 && !!res, r.seconds, res && { dir: path.relative(ROOT, res.dir), cache_hit: res.cache_hit, renders_total_s: res.renders_total_s ?? 0 },
    res ? (res.cache_hit ? 'cache hit' : `rendered ${res.renders} in ${res.renders_total_s} s`) : `exit ${r.code} ${r.out.slice(-400)}`);
  if (!res) return undefined;
  const dir = path.relative(ROOT, res.dir);
  return { dir, manifest: tryJson(path.join(dir, 'manifest.json')) ?? {} };
}

function ourPt(pkg: string, spp: number, B: number, seed: number, run: string, dir: string, add: Add, label: string, extra: string[] = []): { dir: string; meta: Record<string, any> } | undefined {
  console.log(`\n--- ${label} ${pkg}: ${spp} spp × ${B}, seed ${seed}${extra.length ? ` ${extra.join(' ')}` : ''}`);
  const r = runBatches(['--package', pkgDirM7(pkg), '--kernel', 'pt', '--spp', String(spp), '--batches', String(B), '--seed', String(seed), ...extra], run, path.join(dir, 'pt', safe(run.replace(/^m7-gate-[^-]+-\d+-\d+-/, ''))), { B, estSeconds: 60 });
  const ok = !!r.dir && !!r.meta;
  add(`${label} ${pkg} (${spp} spp × ${B}, seed ${seed})`, ok, r.seconds, ok ? { dir: r.dir, gpu_total_ms: Math.round(r.meta!.timings.totalMs) } : undefined, ok ? `${(r.meta!.timings.totalMs / 1000).toFixed(1)} s in the page` : r.out.slice(-300));
  return ok ? { dir: r.dir!, meta: r.meta! } : undefined;
}

/** E2E-HDR: Blender loaded the original file (not our decode) and its pixels equal ours (build_scene.py env.original). */
function hdrCheck(u: StageAUnit, manifest: Record<string, any>): string[] {
  const e = manifest.build_manifest?.env ?? {};
  const p: string[] = [];
  if (e.original !== true) p.push('Blender did not load the original HDRI');
  if (e.format !== u.hdr) p.push(`format ${e.format} != ${u.hdr}`);
  if (e.blender_equals_ours !== true) p.push('Blender pixels != our decode');
  return p;
}

function stageAUnit(u: StageAUnit, dir: string, runId: string, nU: number, add: Add): Record<string, any> {
  const t0 = performance.now();
  const base = { unit: u.pkg, kind: 'stageA', part: u.part, label: u.label, tier: u.tier, cycles: `${u.cyclesSpp}x${u.K}`, ours: `${u.ourSpp}x${u.B}` };
  const ref = cyclesRef(u, 0, add);
  if (!ref) return { ...base, ok: false, status: 'not run', note: 'Cycles reference failed' };
  const ours = ourPt(u.pkg, u.ourSpp, u.B, SEEDS.ours, `${runId}-${u.pkg}`, dir, add, 'PT batches');
  if (!ours) return { ...base, ok: false, status: 'not run', note: 'PT run failed' };
  const t16 = [...t16M7(ours.meta, u.pkg, 'PT'), ...(u.hdr ? hdrCheck(u, ref.manifest) : [])];
  const test = writeTest(dir, u.pkg, 'A', nU, u.tier);
  const out = path.join(dir, 'compare', safe(u.pkg));
  compare(ours.dir, ref.dir, test, out);
  let rep = tryJson(path.join(out, 'report.json'));
  let rerun: Record<string, unknown> | undefined;
  if (rep?.status === 'rerun_required') {
    console.log(`  ${u.pkg}: rerun_required (${rep.failed_checks.join(', ')}) -> confirmatory re-run on disjoint seeds`);
    const ref2 = cyclesRef(u, SEEDS.cyclesRerunBase, add);
    const o2 = ourPt(u.pkg, u.ourSpp, u.B, SEEDS.oursRerun, `${runId}-${u.pkg}-rerun`, dir, add, 'PT re-run batches');
    if (ref2 && o2) {
      t16.push(...t16M7(o2.meta, u.pkg, 'PT re-run'));
      compare(o2.dir, ref2.dir, test, `${out}-rerun`, path.join(out, 'report.json'));
      rerun = { first: summarize(rep), report: path.join(`${out}-rerun`, 'report.json') };
      rep = tryJson(path.join(`${out}-rerun`, 'report.json'));
    }
  }
  const sum = rep ? summarize(rep) : undefined;
  const ok = !!rep && (rep.status === 'pass' || rep.status === 'pass_on_rerun') && t16.length === 0;
  const res = { ...base, ok, ...(sum ?? { status: 'error' }), t16, report: path.join(out, 'report.json'), ref_dir: ref.dir, pt_dir: ours.dir, ...(rerun ? { rerun } : {}),
    ...(u.hdr ? { hdr: ref.manifest.build_manifest?.env } : {}) };
  add(`Stage A ${u.pkg} [${u.tier}] (${u.label})`, ok, (performance.now() - t0) / 1000, res, sum ? `${verdict(sum)}${t16.length ? `; T16: ${t16.join('; ')}` : ''}` : 'no report');
  return res;
}

function ptAA(dir: string, runId: string, nU: number, add: Add): Record<string, any> {
  const u = STAGE_A_UNITS.find((x) => x.pkg === AA_A_PKG)!;
  const [s1, s2] = SEEDS.oursAA;
  const a = ourPt(u.pkg, u.ourSpp, u.B, s1, `${runId}-aa-${s1}`, dir, add, `PT A/A seed set ${s1}`);
  const b = ourPt(u.pkg, u.ourSpp, u.B, s2, `${runId}-aa-${s2}`, dir, add, `PT A/A seed set ${s2}`);
  if (!a || !b) return { unit: `aa-pt-${u.pkg}`, kind: 'aa', part: 'stageA', ok: false, status: 'run failed' };
  const out = path.join(dir, 'compare', `aa-pt-${u.pkg}`);
  compare(a.dir, b.dir, writeTest(dir, `aa-pt-${u.pkg}`, 'A', nU, 'tight'), out);
  const rep = tryJson(path.join(out, 'report.json'));
  const sum = rep ? summarize(rep) : undefined;
  const ok = rep?.status === 'pass';
  const res = { unit: `aa-pt-${u.pkg}`, kind: 'aa', part: 'stageA', ok, ...(sum ?? { status: 'error' }), seeds: `${s1} vs ${s2}`, report: path.join(out, 'report.json') };
  add(`A/A: two of our PT seed sets on ${u.pkg} (${s1} vs ${s2})`, ok, 0, res, sum ? verdict(sum) : 'no report');
  return res;
}

// ------------------------------------------------------------------------------------------------ Stage B (ReSTIR vs PT)

const ptKey = (pkg: string, spp: number, B: number, seed: number) => ({ kind: 'pt', pkg, packageHash: packageHash(pkgDirM7(pkg)), spp, B, seed, rr: false, code: codeHashes().pt });
const ptsizeFile = (pkg: string) => path.join(M7_OUT, 'ptsize', `${pkg}-${codeHashes().pt.slice(0, 16)}-${packageHash(pkgDirM7(pkg)).slice(0, 16)}.json`);
const pilotFrames = (preset: string) => (/^offline/.test(preset) ? 8 : 128);
function rsArgs(pkg: string, preset: string, settings?: Partial<RestirSettings>): string[] {
  return ['--package', pkgDirM7(pkg), '--kernel', 'restir', '--preset', preset, ...(settings && Object.keys(settings).length ? ['--restir-settings', json(settings)] : [])];
}
function batchSide(dir: string, meta: Record<string, any>): PilotSide {
  const files = readdirSync(path.join(ROOT, dir)).filter((n) => /^batch_\d{3}\.pfm$/.test(n)).sort();
  const imgs = files.map((f) => decodePFM(new Uint8Array(readFileSync(path.join(ROOT, dir, f)))));
  const n = meta.sppPerBatch as number, B = files.length;
  return { B, n, msPerSample: (meta.timings.batchMs as number[]).reduce((x, y) => x + y, 0) / (B * n), W: imgs[0].width, H: imgs[0].height, batches: imgs.map((i) => i.data) };
}

interface Sized { B: number; ptSpp: number; ptSeconds: number; frames: number; seconds: number; tile: 32 | 64; notes: string[]; ptSide: PilotSide }
function sizeUnit(pkg: string, preset: RestirPresetName, settings: Partial<RestirSettings> | undefined, add: Add, tag: string): Sized | undefined {
  const B = 16;
  const pdir = pkgDirM7(pkg);
  const pt = cachedRun(`${M7_OUT}/pilots`, ptKey(pkg, 128, B, M7_PT.pilot), `${pkg}-pt`, ['--package', pdir, '--kernel', 'pt', '--spp', '128', '--batches', String(B), '--seed', String(M7_PT.pilot)],
    B, add, `pilot PT ${pkg} (128 spp x ${B}, seed ${M7_PT.pilot})`);
  if (!pt) return undefined;
  const f = pilotFrames(preset);
  const rk = { kind: 'restir', pkg, packageHash: packageHash(pdir), preset, settings: settings ?? null, frames: f, B, seed: SEEDS.restirPilot, code: codeHashes().restir };
  const rp = cachedRun(`${M7_OUT}/pilots`, rk, `${pkg}-${tag.replace(/[^\w.-]+/g, '_')}`, [...rsArgs(pkg, preset, settings), '--spp', String(f), '--batches', String(B), '--seed', String(SEEDS.restirPilot)],
    B, add, `pilot ReSTIR ${pkg} ${tag} (${f} frames x ${B})`);
  if (!rp) return undefined;
  const sf = ptsizeFile(pkg);
  const frozen = tryJson(sf) as { ptSpp: number; B: number } | undefined;
  const ptSide = batchSide(pt.dir, pt.meta);
  const z = sizeScene(ptSide, [{ id: tag, side: batchSide(rp.dir, rp.meta), tile: 32 }], { B, ...(frozen?.B === B ? { fixedPtSpp: frozen.ptSpp } : {}), minFrames: () => f });
  if (!frozen) {
    mkdirSync(path.dirname(path.join(ROOT, sf)), { recursive: true });
    writeFileSync(path.join(ROOT, sf), `${JSON.stringify({ pkg, ptSpp: z.ptSpp, B: z.B, ptSeconds: z.ptSeconds, created: new Date().toISOString() }, null, 1)}\n`);
  }
  const r = z.rungs[tag];
  return { B, ptSpp: z.ptSpp, ptSeconds: z.ptSeconds, frames: r.framesPerBatch, seconds: r.seconds, tile: r.tile, notes: z.notes, ptSide };
}
function ptRefRun(pkg: string, spp: number, B: number, seed: number, estSeconds: number, add: Add): Run | undefined {
  return cachedRun(`${M7_OUT}/ptrefs`, ptKey(pkg, spp, B, seed), `${pkg}-s${seed}-${spp}x${B}`,
    ['--package', pkgDirM7(pkg), '--kernel', 'pt', '--spp', String(spp), '--batches', String(B), '--seed', String(seed)], B, add, `PT reference ${pkg} (${spp} spp x ${B}, seed ${seed})`, estSeconds);
}

function seqUnit(u: SeqUnit, dir: string, runId: string, nU: number, add: Add, pilotOnly: boolean): Record<string, any> {
  const t0 = performance.now();
  const base = { unit: u.id, kind: 'seq', part: u.part, rung: u.rung, scene: u.pkg, label: u.label, preset: u.preset, settings: u.settings ?? {}, lightMode: u.lightMode, tier: 'tight' };
  const z = sizeUnit(u.pkg, u.preset, u.settings, add, u.id);
  if (!z) return { ...base, ok: false, status: 'not run', note: 'pilot failed' };
  const sizing = { B: z.B, ptSpp: z.ptSpp, pt_min: r4(z.ptSeconds / 60), frames: z.frames, restir_min: r4(z.seconds / 60), tile: z.tile, notes: z.notes };
  add(`sizing ${u.id}`, true, 0, sizing, `PT ${z.ptSpp} spp x ${z.B} (${(z.ptSeconds / 60).toFixed(1)} min), ReSTIR ${z.frames} fr x ${z.B} (${(z.seconds / 60).toFixed(1)} min)${z.tile === 64 ? ' @64²' : ''}${z.notes.length ? `; ${z.notes.join('; ')}` : ''}`);
  if (pilotOnly) return { ...base, ok: true, status: 'sized', sizing };
  const ref = ptRefRun(u.pkg, z.ptSpp, z.B, M7_PT.ref, 1.2 * z.ptSeconds, add);
  if (!ref) return { ...base, ok: false, status: 'not run', note: 'PT reference failed', sizing };
  const expect = restirSettings(u.preset, u.settings ?? {});
  const run = (seed: number, sub: string) => {
    console.log(`\n--- ReSTIR ${u.id}: ${z.frames} frames x ${z.B}, seed ${seed}`);
    return runBatches([...rsArgs(u.pkg, u.preset, u.settings), '--spp', String(z.frames), '--batches', String(z.B), '--seed', String(seed)],
      `${runId}-${sub}`.replace(/[^\w.-]+/g, '_').slice(0, 120), path.join(dir, 'restir', safe(sub)), { B: z.B, estSeconds: z.seconds * 1.2 });
  };
  const first = run(SEEDS.restir, u.id);
  const test = writeTest(dir, u.id, 'B', nU, 'tight', z.tile !== 32 ? { tile: z.tile, aggregate_note: 'pilot sizing: a side needs > 60 min at 32² tiles (restir-api.md §6.3)' } : {});
  const out = path.join(dir, 'compare', safe(u.id));
  const t16 = first.meta ? [...t16M6(first.meta, ref.meta, { rounds: expect.rounds, lightMode: u.lightMode, expect }), ...t16M7(first.meta, u.pkg, 'ReSTIR'), ...t16M7(ref.meta, u.pkg, 'PT')]
    : ['ReSTIR run produced no meta.json'];
  let rep: Record<string, any> | undefined;
  if (first.dir && first.meta) { compare(first.dir, ref.dir, test, out); rep = tryJson(path.join(out, 'report.json')); }
  let rerun: Record<string, unknown> | undefined;
  if (rep?.status === 'rerun_required') {
    console.log(`  ${u.id}: rerun_required (${rep.failed_checks.join(', ')}) -> confirmatory re-run on disjoint seeds (PT ${M7_PT.rerun}, ReSTIR ${SEEDS.restirRerun})`);
    const ref2 = ptRefRun(u.pkg, z.ptSpp, z.B, M7_PT.rerun, 1.2 * z.ptSeconds, add);
    const second = run(SEEDS.restirRerun, `${u.id}-rerun`);
    if (ref2 && second.dir && second.meta) {
      t16.push(...t16M6(second.meta, ref2.meta, { rounds: expect.rounds, lightMode: u.lightMode, expect }).map((x) => `re-run: ${x}`));
      compare(second.dir, ref2.dir, test, `${out}-rerun`, path.join(out, 'report.json'));
      rerun = { first: summarize(rep), report: path.join(`${out}-rerun`, 'report.json') };
      rep = tryJson(path.join(`${out}-rerun`, 'report.json'));
    }
  }
  const sum = rep ? summarize(rep) : undefined;
  const ok = !!rep && (rep.status === 'pass' || rep.status === 'pass_on_rerun') && t16.length === 0;
  const res = {
    ...base, ok, ...(sum ?? { status: 'error' }), sizing, pt: `${z.ptSpp}x${z.B}`, restir: `${z.frames}x${z.B}`, tile: z.tile, fr: r4(first.meta?.restir?.fr),
    restir_minutes: first.meta ? r4(first.meta.timings.totalMs / 60000) : undefined, pt_cache_hit: ref.cacheHit, t16, report: path.join(out, 'report.json'),
    restir_dir: first.dir, pt_dir: ref.dir, ...(rerun ? { rerun } : {}),
  };
  add(`Stage B ${u.id} (${u.label}; ${u.preset}${u.settings ? ` ${json(u.settings)}` : ''}, Mode ${u.lightMode})`, ok, (performance.now() - t0) / 1000, res,
    sum ? `${verdict(sum)}${t16.length ? `; T16: ${t16.join('; ')}` : ''}` : `no report${t16.length ? `; T16: ${t16.join('; ')}` : ''}`);
  return res;
}

function chainUnits(units: ChainUnitM7[], dir: string, runId: string, nU: number, add: Add, pilotOnly: boolean): { results: Record<string, any>[]; sizing: Record<string, unknown> } {
  for (const u of units) EXT.pkgDirs.set(u.pkg, pkgDirM7(u.pkg));
  EXT.sequences = [];
  EXT.out = M7_OUT;
  EXT.chainSeeds = { chains: SEEDS.chains, chainsRerun: SEEDS.chainsRerun, chainPilot: SEEDS.chainPilot };
  const r = runExternalChainUnits(units, { dir, runId, nU, add, pilotOnly });
  const results = r.results.map((x) => {
    const u = units.find((v) => x.unit === v.id || String(x.unit).startsWith(`${v.id}-`))!;
    return { ...x, part: 'r38', label: u.label, lightMode: u.lightMode, kind: `chain-${x.kind}` };
  });
  return { results, sizing: r.sizing };
}

// ------------------------------------------------------------------------------------------------ plants

/** Panel masks (PFM, R = 1 inside): the panel rectangle on the back wall (z = −0.497) inset by 15 %, per pixel by a ray
 *  through the pixel centre and the package camera. */
function panelMasks(pkg: string, names: ('M_P2' | 'M_P3')[], dir: string, tag: string): { name: string; file: string }[] {
  const j = pkgJson(pkg);
  const W = j.render.width as number, H = j.render.height as number;
  const cm = j.camera.matrix as number[], t = Math.tan(j.camera.yfov / 2), asp = W / H;
  const mdir = path.join(dir, 'tests', 'masks');
  mkdirSync(path.join(ROOT, mdir), { recursive: true });
  const out: { name: string; file: string }[] = [];
  for (const n of names) {
    const p = j.panels[n.slice(2)] as { x: [number, number]; y: [number, number] };
    const ix = 0.15 * (p.x[1] - p.x[0]), iy = 0.15 * (p.y[1] - p.y[0]);
    const a = new Float32Array(W * H * 3);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const u = ((x + 0.5) / W * 2 - 1) * t * asp, v = (1 - (y + 0.5) / H * 2) * t;
      const d = [0, 1, 2].map((k) => u * cm[k] + v * cm[4 + k] - cm[8 + k]);
      if (Math.abs(d[2]) < 1e-9) continue;
      const s = (-0.497 - cm[14]) / d[2];
      if (s <= 0) continue;
      const hx = cm[12] + s * d[0], hy = cm[13] + s * d[1];
      if (hx > p.x[0] + ix && hx < p.x[1] - ix && hy > p.y[0] + iy && hy < p.y[1] - iy) a[3 * (y * W + x)] = 1;
    }
    const file = `masks/${n}-${tag}.pfm`;
    writeFileSync(path.join(ROOT, mdir, `${n}-${tag}.pfm`), encodePFM({ width: W, height: H, channels: 3, data: a }));
    out.push({ name: n, file });
  }
  return out;
}

function plantSign(p: PlantM7, ours: string, ref: string, test: string, dir: string): Record<string, any> | undefined {
  const pdir = path.join(dir, 'plants', safe(p.id));
  mkdirSync(path.join(ROOT, pdir), { recursive: true });
  const spec = path.join(pdir, 'plant.json');
  writeFileSync(path.join(ROOT, spec), JSON.stringify({ name: p.name, test: path.join(ROOT, test), repeats: 10, seed: 0, only: null,
    frames: [{ frame: 0, ours: path.join(ROOT, ours), ref: path.join(ROOT, ref), test: path.join(ROOT, test) }],
    predict: p.predict.map((x) => ({ frame: 0, ...x })) }, null, 1));
  sh(PY, ['validation/tools/plant_sign.py', '--spec', spec, '--out', path.join(pdir, 'sign')], () => false);
  return tryJson(path.join(pdir, 'sign', 'report.json'));
}

function plantUnit(p: PlantM7, i: number, dir: string, runId: string, nU: number, add: Add): Record<string, any> {
  const t0 = performance.now();
  const data: Record<string, any> = { unit: `plant-${p.id}`, kind: 'plant', part: 'plants', stage: p.stage, scene: p.pkg, plant: p.name, predict: p.predict, derivation: p.derivation, ok: false };
  const masks = panelMasks(p.pkg, p.masks, dir, p.id);
  let ours: { dir: string; meta: Record<string, any> } | undefined, ref: { dir: string } | undefined, tile = 32;
  const t16: string[] = [];
  if (p.stage === 'A') {
    const u = STAGE_A_UNITS.find((x) => x.pkg === p.pkg)!;
    const cr = cyclesRef(u, 0, add);
    ours = ourPt(p.pkg, u.ourSpp, u.B, SEEDS.plantA + i, `${runId}-plant-${p.id}`, dir, add, `PT planted (${p.id})`, p.ptArgs);
    ref = cr;
    if (ours && !(ours.meta.config?.plant?.nm === (p.plant as { nm: string }).nm)) t16.push(`PT meta config.plant ${json(ours.meta.config?.plant)} (want nm ${(p.plant as { nm: string }).nm})`);
  } else {
    const z = sizeUnit(p.pkg, p.preset!, p.base, add, `plantbase-${p.id}`);
    if (!z) return { ...data, status: 'sizing failed' };
    tile = z.tile;
    const rSeed = (p.seedBase ?? SEEDS.plantBase) + i, cSeed = (p.seedBase ? p.seedBase + 100 : SEEDS.ptCalib) + i;
    const pr = ptRefRun(p.pkg, z.ptSpp * CALIB_FACTOR, z.B, cSeed, 1.2 * CALIB_FACTOR * z.ptSeconds, add);
    const settings = { ...(p.base ?? {}), plant: p.plant } as Partial<RestirSettings>;
    const run = runBatches([...rsArgs(p.pkg, p.preset!, settings), '--spp', String(z.frames), '--batches', String(z.B), '--seed', String(rSeed)],
      `${runId}-plant-${p.id}`.replace(/[^\w.-]+/g, '_'), path.join(dir, 'plants', safe(p.id), 'restir'), { B: z.B, estSeconds: z.seconds * 1.2 });
    if (run.dir && run.meta && pr) {
      ours = { dir: run.dir, meta: run.meta }; ref = pr;
      const expect = restirSettings(p.preset!, settings);
      t16.push(...t16M6(run.meta, pr.meta, { plant: true, rounds: expect.rounds, lightMode: 'A', expect }));
      const named = run.meta.t16?.plantsNamed as string[] | undefined;
      for (const k of Object.keys(p.plant)) if (!named?.includes(k)) t16.push(`plant ${k} not named in t16.plantsNamed (${json(named)})`);
    }
    Object.assign(data, { sizing: { frames: z.frames, B: z.B, ptSpp4x: z.ptSpp * CALIB_FACTOR, tile: z.tile }, seeds: { restir: rSeed, ptCalib: cSeed, ...(p.seedBase ? { note: 'revised plant: fresh disjoint seeds (E-18)' } : {}) } });
  }
  if (!ours || !ref) { add(`plant ${p.id}`, false, (performance.now() - t0) / 1000, data, 'run failed'); return { ...data, status: 'run failed' }; }
  const test = writeTest(dir, `plant-${p.id}`, p.stage, nU, 'tight', { ...(masks.length ? { masks } : {}), ...(tile !== 32 ? { tile } : {}) });
  const fullOut = path.join(dir, 'compare', safe(`plant-${p.id}-full`));
  compare(ours.dir, ref.dir, test, fullOut);
  const full = tryJson(path.join(fullOut, 'report.json'));
  const rep = plantSign(p, ours.dir, ref.dir, test, dir);
  const fullNotPass = !!full && full.status !== 'pass';
  const ok = !!rep?.ok && fullNotPass && t16.length === 0;
  const f0 = rep?.frames?.[0];
  Object.assign(data, {
    ok, status: ok ? 'detected, sign holds' : rep ? `detected ${rep.detected}, sign ${rep.signs_hold}, full-not-pass ${fullNotPass}` : 'plant_sign failed',
    detection: f0 && `${f0.detection.gate_fail_count}/${f0.detection.n_repeats} (control ${f0.detection.control_pass_count}/${f0.detection.n_repeats})`,
    predictions: f0?.predictions?.map((x: any) => ({ region: x.region, sign: x.sign, rel: r4(x.rel), z: r4(x.z), holds: x.holds })),
    global_Y: f0 && { rel: r4(f0.global_Y.rel), z: r4(f0.global_Y.z) }, full_compare: full && { status: full.status, global_rel_Y: r4(full.channels.Y.global_.rel), failed: full.failed_checks },
    t16, ours_dir: ours.dir, ref_dir: ref.dir, report: path.join(dir, 'plants', safe(p.id), 'sign', 'report.json'),
  });
  add(`plant ${p.id} (Stage ${p.stage}): ${p.name} on ${p.pkg} (${p.predict.map((x) => `${x.region} ${x.sign}`).join(', ')})`, ok, (performance.now() - t0) / 1000, data,
    `${data.status}; detection ${data.detection}; ${(data.predictions ?? []).map((x: any) => `${x.region} ${pct(x.rel, 3)} z ${x.z}`).join(', ')}${t16.length ? `; T16: ${t16.join('; ')}` : ''}`);
  return data;
}

/** ReSTIR A/A (7502 vs 7503 at 4× the unit size) + synthetic W × 1.003 with calibrate A/A re-splits on m7_nm_smooth_256. */
function aaAndSynthetic(dir: string, runId: string, nU: number, add: Add): Record<string, any>[] {
  const t0 = performance.now();
  const u = SEQ_UNITS.find((x) => x.id === 'm7_nm_smooth_256@3.8-offline')!;
  const z = sizeUnit(u.pkg, u.preset, u.settings, add, u.id);
  if (!z) return [{ unit: 'aa-restir', kind: 'aa', part: 'plants', ok: false, status: 'sizing failed' }];
  const ref = ptRefRun(u.pkg, z.ptSpp, z.B, M7_PT.ref, 1.2 * z.ptSeconds, add);
  const frames = z.frames * CALIB_FACTOR;
  const runs = [SEEDS.aa, SEEDS.aa2].map((seed) => runBatches([...rsArgs(u.pkg, u.preset, u.settings), '--spp', String(frames), '--batches', String(z.B), '--seed', String(seed)],
    `${runId}-aa-${seed}`, path.join(dir, 'restir', `aa-${seed}`), { B: z.B, estSeconds: z.seconds * CALIB_FACTOR * 1.2 }));
  const out: Record<string, any>[] = [];
  const expect = restirSettings(u.preset, u.settings ?? {});
  if (!(runs.every((r) => r.dir && r.meta) && ref)) return [{ unit: 'aa-restir', kind: 'aa', part: 'plants', ok: false, status: 'run failed' }];
  const t16 = runs.flatMap((r) => [...t16M6(r.meta!, ref.meta, { rounds: expect.rounds, lightMode: 'A', expect }), ...t16M7(r.meta, u.pkg, 'ReSTIR')]);
  const test = writeTest(dir, 'aa-restir', 'B', nU, 'tight', z.tile !== 32 ? { tile: z.tile } : {});
  const o = path.join(dir, 'compare', 'aa-restir');
  compare(runs[1].dir!, runs[0].dir!, test, o);
  const rep = tryJson(path.join(o, 'report.json'));
  const ok = rep?.status === 'pass' && t16.length === 0;
  const res: Record<string, any> = { unit: `aa-restir-${u.pkg}`, kind: 'aa', part: 'plants', ok, ...(rep ? summarize(rep) : { status: 'error' }), seeds: `${SEEDS.aa2} vs ${SEEDS.aa}`, restir: `${frames}x${z.B} (x${CALIB_FACTOR})`, t16 };
  out.push(res);
  add(`A/A: two ReSTIR seed sets on ${u.pkg} offline-m6 (${SEEDS.aa} vs ${SEEDS.aa2}, ${frames} frames x ${z.B})`, ok, (performance.now() - t0) / 1000, res, rep ? verdict(summarize(rep)) : 'no report');
  const t1 = performance.now();
  const ctest = writeTest(dir, 'plant-W1.003', 'B', nU, 'tight', z.tile !== 32 ? { tile: z.tile } : {});
  const co = path.join(dir, 'calibrate', 'W1.003');
  const r = sh(PY, ['validation/tools/compare.py', '--calibrate', '--ref', runs[0].dir!, '--test', ctest, '--out', co, '--splits', '20', '--repeats', '10'], () => false);
  const crep = tryJson(path.join(co, 'report.json'));
  const aa = crep?.calibration?.aa, pl = crep?.calibration?.plants?.[0];
  const ok2 = r.code === 0 && !!aa?.ok && !!pl?.calibrated;
  const res2 = { unit: `plant-W1.003-${u.pkg}`, kind: 'plant', part: 'plants', plant: 'W x1.003 (synthetic)', ok: ok2, status: ok2 ? 'detected' : 'not detected',
    detected: pl && `${pl.gate_fail_count}/${pl.n_repeats}`, control_pass: pl && `${pl.control_pass_count}/${pl.n_repeats}`, aa_resplits: aa && { ok: aa.ok, gate_pass_rate: aa.gate_pass_rate } };
  out.push(res2);
  add(`synthetic W x1.003 + calibrate A/A re-splits on the ${u.pkg} offline-m6 A/A run`, ok2, (performance.now() - t1) / 1000, res2, `detected ${res2.detected}, control ${res2.control_pass}, A/A ${aa?.ok ? 'ok' : 'FAIL'}`);
  return out;
}

// ------------------------------------------------------------------------------------------------ loader

function loader(dir: string, add: Add): Record<string, any>[] {
  const out: Record<string, any>[] = [];
  const ldir = path.join(dir, 'loader');
  mkdirSync(path.join(ROOT, ldir), { recursive: true });
  for (const [label, files] of [['(viii-L) USD loader vs OpenUSD (pxr)', USD_FIDELITY], ['(vii-L) glTF loader vs Blender 5.2.2 stock import', GLB_FIDELITY]] as const) {
    const r = sh('npx', ['tsx', 'validation/harness/m7-loader-fidelity.ts', '--out', ldir, ...files], (l) => /^(PASS|FAIL)|mismatch|Error/.test(l));
    const fails = r.out.split('\n').filter((l) => /^FAIL/.test(l));
    const ok = r.code === 0;
    out.push({ unit: label, kind: 'loader', part: 'loader', ok, status: ok ? 'pass' : 'fail', files: files.length, fails });
    add(`${label}: ${files.length} files`, ok, r.seconds, { files, fails, log: r.out.slice(-4000) }, ok ? `${files.length}/${files.length} files` : fails.slice(0, 4).join(' | ') || r.out.slice(-300));
  }
  const tdir = path.join(dir, 'loader', 'tangents');
  mkdirSync(path.join(ROOT, tdir), { recursive: true });
  const r = sh('npx', ['tsx', 'validation/harness/m7-tangent-check.ts', '--out', tdir, ...TANGENT_PKGS.map(pkgDirM7)], (l) => /^(PASS|FAIL)|Error/.test(l));
  out.push({ unit: 'U-TAN-B', kind: 'loader', part: 'loader', ok: r.code === 0, status: r.code === 0 ? 'pass' : 'fail', packages: TANGENT_PKGS });
  add(`U-TAN-B: package tangents vs Blender MikkTSpace (sign on every normal-mapped corner; angle p99.9 <= 0.01°, max <= 0.05°) on ${TANGENT_PKGS.length} packages`, r.code === 0, r.seconds,
    { log: r.out.slice(-3000) }, r.out.split('\n').filter((l) => /^(PASS|FAIL)/.test(l)).map((l) => l.slice(0, 160)).join(' | '));
  return out;
}

// ------------------------------------------------------------------------------------------------ Gate 0

type RunStep = (name: string, cmd: string, argv: string[], echo?: (l: string) => boolean, env?: Record<string, string>) => { code: number; out: string };
const reEsc = (x: string) => x.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function gate0(dir: string, runId: string, add: Add, runStep: RunStep): void {
  runStep('typecheck', 'npx', ['tsc', '--noEmit', '-p', '.']);
  runStep('vitest cpu (U-M7-BITS, U-M7-ARENA, tangents, plant-m7 predictions + regressions)', 'npx', ['vitest', 'run', '--project', 'cpu'], (l) => /Test Files|Tests |FAIL|✗|×/.test(l));
  runStep('python tests (stats, compare, dynamic, plant_sign, dup_bias)', PY, ['-m', 'pytest', 'validation/tools/tests', '-q'], (l) => /passed|failed|error/i.test(l));
  packagesDeterministic(dir, add, 'make-m7.ts', 'validation/scenes/make-m7.ts', M7_SCENE_DIR);
  packagesDeterministic(dir, add, 'make-m7-e2e.ts', 'validation/scenes/make-m7-e2e.ts', M7_E2E_DIR);
  const vit = (file: string, extra: string[]) => ['vitest', 'run', '--project', 'chrome', '--reporter=verbose', `validation/gpu-tests/${file}.gpu.test.ts`, ...extra];
  const echo = (l: string) => /Tests |FAIL|✗|×|AssertionError|LOGIC|FP-BOUNDARY|violation|T3-M[67]|U-NM|M7 perf/.test(l);
  withGpuLockSync('gate-m7-normal-map', () => { runStep('normal-map (chrome) [M7]: U-NM-1 decode vs the f64 Cycles node, U-NM-2 bump-shadowing term, U-NM-3 tilted-normal furnace', 'npx', vit('normal-map', []), echo); });
  for (const v of T3M7_RUNS) {
    withGpuLockSync(`gate-m7-t3m7-${v}`, () => {
      runStep(`restir-shift (chrome) [T3-M7 ${v}]: LOGIC 0, every bin / counter >= 1e6`, 'npx', vit('restir-shift', ['--testTimeout', '3600000', '-t', `${reEsc(T3M7_DESCRIBE)}.* ${reEsc(v)}$`]), echo, { VITE_T3_MS: String(18 * 60_000) });
    });
  }
  // M6 / M5 / M4 Gate-0 suites (regressions; the gate-m6 holds)
  for (const s of M6_GPU_SUITES) {
    withGpuLockSync(`gate-m7-${s.file}`, () => { runStep(`${s.file} (chrome) [M6 regression]: ${s.what}`, 'npx', vit(s.file, [...(s.env ? ['--testTimeout', '900000'] : []), ...(s.t ? ['-t', s.t] : [])]), echo, s.env); });
  }
  for (const v of T3M6_RUNS) {
    withGpuLockSync(`gate-m7-t3m6-${v}`, () => {
      runStep(`restir-shift (chrome) [T3-M6 ${v}, regression]`, 'npx', vit('restir-shift', ['--testTimeout', '3600000', '-t', `${reEsc(T3M6_DESCRIBE)}.* ${reEsc(v)}$`]), echo, { VITE_T3_MS: String(18 * 60_000) });
    });
  }
  const seen = new Set<string>();
  for (const [file, what] of [...M4_SUITES, ...M5_SUITES]) {
    if (seen.has(file)) continue;
    seen.add(file);
    if (!existsSync(path.join(ROOT, `validation/gpu-tests/${file}.gpu.test.ts`))) { add(`${file} (chrome): ${what}`, false, 0, undefined, 'missing'); continue; }
    for (const h of gpuSuiteHolds(file)) {
      const args = file === 'restir-shift' && h.label === ' [all but the T3 variants]' ? ['-t', '^(?!.*(T3-0 / T2 / T3-1 / T4 / T3-D / U5 / T3-ENV / U-12 on the t3 fixtures|T3-M6|T3-M7))']
        : file === 'restir-temporal' ? ['-t', '^(?!.*Mode B: )'] : h.args;
      withGpuLockSync(`gate-m7-${file}`, () => { runStep(`${file} (chrome)${h.label} [M4/M5 regression]: ${what}`, 'npx', vit(file, args), echo, GPU_SUITE_ENV[file]); });
    }
  }
  for (const c of T32_RARE_CASES) {
    withGpuLockSync(`gate-m7-t32-${safe(c)}`, () => {
      runStep(`restir-temporal T3-2 ${c} [M5 regression] (>= 1e6 per bin)`, 'npx', vit('restir-temporal', ['--testTimeout', '900000', '-t', reEsc(c)]), echo, T32_RARE_ENV);
    });
  }
  withGpuLockSync('gate-m7-perf', () => {
    const r = runStep('M7 perf probe (chrome; 540p interactive Mode B with / without normal maps; recorded)', 'npx', vit('m7-perf', ['--testTimeout', '900000']), echo);
    const i = r.out.indexOf('[M7 perf');
    const block = i >= 0 ? r.out.slice(i, i + 2500) : '';
    add('M7 perf: 540p interactive frame time with / without normal maps (recorded, not gating)', true, 0, { block }, block.replace(/\s+/g, ' ').slice(0, 400) || 'no perf block');
  });
  // Takes the GPU lock itself.
  runStep('M7 app smoke (Sponza / USD textured / Cornell: NORMAL_MAP on / off, PT + ReSTIR finite, views 320-327)', 'npx', ['tsx', 'validation/harness/m7-app-smoke.ts', '--run', `${runId}-app-smoke`], (l) => /^(PASS|FAIL)\s/.test(l));
}

/** A generator writes byte-identical packages twice, equal to the gate's packages. */
function packagesDeterministic(dir: string, add: Add, name: string, script: string, target: string): void {
  const t0 = performance.now();
  const a = path.join(dir, `${name}-a`), b = path.join(dir, `${name}-b`);
  const ra = sh('npx', ['tsx', script, a], () => false), rb = sh('npx', ['tsx', script, b], () => false);
  const list = (d: string): string[] => {
    const out: string[] = [];
    const walk = (x: string) => { for (const n of readdirSync(path.join(ROOT, x)).sort()) { const q = path.join(x, n); if (statSync(path.join(ROOT, q)).isDirectory()) walk(q); else out.push(path.relative(d, q)); } };
    if (existsSync(path.join(ROOT, d))) walk(d);
    return out;
  };
  const fa = list(a), fb = list(b), diffs: string[] = [];
  if (ra.code !== 0 || rb.code !== 0) diffs.push(`exit ${ra.code}/${rb.code}: ${(ra.out + rb.out).slice(-300)}`);
  for (const f of new Set([...fa, ...fb])) if (!fa.includes(f) || !fb.includes(f) || !readFileSync(path.join(ROOT, a, f)).equals(readFileSync(path.join(ROOT, b, f)))) diffs.push(f);
  for (const f of fa) if (!existsSync(path.join(ROOT, target, f)) || !readFileSync(path.join(ROOT, a, f)).equals(readFileSync(path.join(ROOT, target, f)))) diffs.push(`${target}/${f} differs from the generator`);
  if (!fa.length) diffs.push('no files written');
  rmSync(path.join(ROOT, a), { recursive: true, force: true });
  rmSync(path.join(ROOT, b), { recursive: true, force: true });
  add(`${name} package determinism (twice, byte-identical, = ${target})`, diffs.length === 0, (performance.now() - t0) / 1000, { files: fa.length, diffs: diffs.slice(0, 20) },
    diffs.length ? `diffs: ${diffs.slice(0, 6).join(', ')}` : `${fa.length} files`);
}

function ensurePackages(add: Add): void {
  for (const [list, script, target] of [[M7_PKGS, 'validation/scenes/make-m7.ts', M7_SCENE_DIR], [E2E_PKGS, 'validation/scenes/make-m7-e2e.ts', M7_E2E_DIR]] as const) {
    const missing = list.filter((p) => !existsSync(path.join(ROOT, target, p, 'scene.json')));
    if (!missing.length) { add(`M7 packages present (${list.length}, ${path.basename(script)})`, true, 0); continue; }
    const r = sh('npx', ['tsx', script, target, '--only', missing.join(',')], () => false);
    const still = missing.filter((p) => !existsSync(path.join(ROOT, target, p, 'scene.json')));
    add(`M7 packages generated (${missing.join(', ')})`, r.code === 0 && !still.length, r.seconds, undefined, still.length ? `missing ${still.join(', ')}` : undefined);
  }
}

// ------------------------------------------------------------------------------------------------ the gate

export interface M7Options { part?: Part; only?: Set<string>; pilotOnly?: boolean; writeBudget?: boolean }

export function milestoneM7(record: Rec, o: M7Options = {}): void {
  const t0 = performance.now();
  const parts = o.part ? [o.part] : PARTS;
  const runId = `m7-gate-${o.part ?? 'all'}-${stamp()}`;
  const dir = path.join('validation/out', runId);
  mkdirSync(path.join(ROOT, dir), { recursive: true });
  const steps: { name: string; ok: boolean; seconds: number; data?: unknown; detail?: string }[] = [];
  const add: Add = (name, ok, seconds, data, detail) => {
    steps.push({ name, ok, seconds: Math.round(seconds * 10) / 10, data, detail });
    record(name, ok, `${detail ? `${detail}, ` : ''}${seconds.toFixed(1)} s`);
  };
  const runStep: RunStep = (name, cmd, argv, echo, env) => {
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
  console.log(`M7 gate ${runId}: parts ${parts.join(',')}, n_units ${nU}, PT code ${hashes.pt.slice(0, 12)}, ReSTIR ${hashes.restir.slice(0, 12)}`);
  const missing = [PY, BLENDER].filter((p) => !existsSync(p));
  add('M7 gate tools (venv python, Blender)', missing.length === 0, 0, undefined, missing.length ? `missing ${missing.join(', ')}` : undefined);
  ensurePackages(add);
  const results: Record<string, any>[] = [];
  const sizing: Record<string, unknown> = {};
  for (const part of parts) {
    if (part === 'core') { if (!o.only && !o.pilotOnly) gate0(dir, runId, add, runStep); continue; }
    if (o.pilotOnly && part !== 'r38') continue;
    if (part === 'loader') { if (!o.only) results.push(...loader(dir, add)); continue; }
    if (part === 'stageA' || part === 'e2e') {
      for (const u of STAGE_A_UNITS.filter((x) => x.part === part && sel(x.pkg, x.pkg))) results.push(stageAUnit(u, dir, runId, nU, add));
      if (part === 'stageA' && (!o.only || o.only.has('aa'))) results.push(ptAA(dir, runId, nU, add));
      continue;
    }
    if (part === 'r38') {
      for (const u of SEQ_UNITS.filter((x) => sel(x.id, x.pkg))) results.push(seqUnit(u, dir, runId, nU, add, !!o.pilotOnly));
      const cu = CHAIN_UNITS.filter((x) => sel(x.id, x.pkg));
      if (cu.length) { const r = chainUnits(cu, dir, runId, nU, add, !!o.pilotOnly); results.push(...r.results); sizing.r38 = r.sizing; }
      continue;
    }
    if (part === 'plants') {
      plantList().forEach((p, i) => { if (sel(p.id, p.pkg)) results.push(plantUnit(p, i, dir, runId, nU, add)); });
      if (!o.only || o.only.has('aa')) results.push(...aaAndSynthetic(dir, runId, nU, add));
    }
  }
  writeFileSync(path.join(ROOT, dir, 'sizing.json'), `${JSON.stringify({ seq: results.filter((r) => r.sizing).map((r) => ({ unit: r.unit, ...r.sizing })), chains: sizing }, null, 1)}\n`);
  const rows = budgetRows(results, sizing);
  writeFileSync(path.join(ROOT, dir, 'budget-m7.json'), `${JSON.stringify(rows, null, 1)}\n`);
  if (o.writeBudget && rows.length) mergeBudget(rows, runId, add);
  const failed = steps.filter((x) => !x.ok).map((x) => x.name);
  const perPart = Object.fromEntries(PARTS.filter((p) => p !== 'core').map((p) => {
    const u = results.filter((x) => x.part === p);
    return [p, { pass: u.filter((x) => x.ok).length, fail: u.filter((x) => !x.ok && x.status !== 'not run').length, notRun: u.filter((x) => x.status === 'not run').length }];
  }));
  const summary = {
    milestone: 'M7', gate: 'Gate 0 + loader fidelity (vii-L / viii-L, U-TAN-B) + Stage A (vii-N, E2E-HDR, E2E-GLB / E2E-USD) + rung 3.8 + M7 plants (m7-api.md §6)',
    run: runId, parts, created: new Date().toISOString(), ok: failed.length === 0, subset: o.only ? [...o.only] : undefined, pilotOnly: !!o.pilotOnly,
    total_s: Math.round((performance.now() - t0) / 100) / 10, n_units: nU, code_hashes: hashes, per_part: perPart, failed, units: results, steps,
  };
  writeFileSync(path.join(ROOT, dir, 'summary.json'), `${JSON.stringify(summary, null, 1)}\n`);
  writeFileSync(path.join(ROOT, dir, 'summary.md'), summaryMd(summary, results));
  console.log(`\n${table(results)}\nsummary: ${dir}/summary.json (${summary.total_s} s)`);
}

function budgetRows(results: Record<string, any>[], chainSizing: Record<string, any>): Record<string, unknown>[] {
  const rows: Record<string, unknown>[] = [];
  for (const r of results.filter((x) => x.kind === 'seq' && x.sizing)) {
    const z = r.sizing;
    rows.push({ unit: r.unit, part: r.part, rung: r.rung, scene: r.scene, preset: r.preset, light_mode: r.lightMode, B: z.B, pt_spp: z.ptSpp, pt_min: z.pt_min,
      frames_per_batch: z.frames, restir_min: z.restir_min, tile: z.tile, measured_restir_min: r.restir_minutes });
  }
  for (const r of results.filter((x) => x.kind === 'stageA')) rows.push({ unit: r.unit, part: r.part, kind: 'stageA', tier: r.tier, cycles: r.cycles, ours: r.ours });
  for (const [part, zs] of Object.entries(chainSizing)) {
    for (const u of (zs as any).units ?? []) rows.push({ unit: u.id, part, kind: 'chains', R: u.R, frames: u.frames, test_frames: u.testFrames, tile: u.tile, chain_min: u.minutes, ms_per_chain: u.msPerChain, cap: u.cap });
  }
  return rows;
}
function mergeBudget(rows: Record<string, unknown>[], runId: string, add: Add): void {
  const p = path.join(ROOT, 'validation/budget.json');
  const b = JSON.parse(readFileSync(p, 'utf8')) as Record<string, any>;
  const keep = (b.m7_entries ?? []).filter((e: any) => !rows.some((r) => r.unit === e.unit));
  b.m7_method = 'M7 sizing (gate-m7.ts): Stage A fixed Cycles 4096 spp x 16 seeds vs our PT 4096 spp x 16 (D4: references >= 4096 spp); Stage B as gate-m6 (PT pilot 128 spp x 16, ReSTIR pilot 8 / 128 frames x 16, PLAN §7.3 x1.25); chains by gate-m5 sizeGroup';
  b.m7_measured_at = new Date().toISOString();
  b.m7_runs = [...new Set([...(b.m7_runs ?? []), runId])];
  b.m7_entries = [...keep, ...rows];
  writeFileSync(p, `${JSON.stringify(b, null, 2)}\n`);
  add('budget.json M7 rows written (--write-budget)', true, 0, { rows: b.m7_entries.length });
}

function table(results: Record<string, any>[]): string {
  const head = '| unit | kind | tier | status | Δ_Y | MDB_Y | worst tile | sizes | mult |\n|---|---|---|---|---|---|---|---|---|\n';
  return head + results.map((r) => `| ${r.unit} | ${r.kind} | ${r.tier ?? ''} | ${r.status}${r.t16?.length ? ' (T16)' : ''} | ${pct(r.global_rel_Y, 4)} | ${pct(r.mdb_global_Y, 4)} | ${pct(r.worst_tile_rel_Y, 2)} ${r.worst_tile ?? ''} | ${r.cycles ? `Cycles ${r.cycles} / PT ${r.ours}` : r.pt ? `PT ${r.pt} / ReSTIR ${r.restir}` : r.R ?? ''} | ${r.multiplier_needed ?? ''} |`).join('\n') + '\n';
}
function summaryMd(summary: Record<string, any>, results: Record<string, any>[]): string {
  return [
    `# M7 gate ${summary.run}`, '', `Result: **${summary.ok ? 'PASS' : 'FAIL'}** (${summary.total_s} s, parts ${summary.parts.join(', ')}, n_units ${summary.n_units})`, '',
    '## Per part', '', '| part | pass | fail | not run |', '|---|---|---|---|',
    ...Object.entries(summary.per_part as Record<string, any>).map(([k, v]) => `| ${k} | ${v.pass} | ${v.fail} | ${v.notRun} |`), '',
    '## Units', '', table(results), '## Failed steps', '', ...(summary.failed as string[]).map((f) => `- ${f}`), '',
  ].join('\n');
}

