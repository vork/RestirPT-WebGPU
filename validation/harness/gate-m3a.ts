// M3a milestone gate (plan §5 M3a exit, §7.1 Gates 0–2, §7.3, §7.4 M3a): `npm run validate -- --milestone M3a`.
//   Gate 0  typecheck, cpu lane, python stats tests, Chrome GPU tests T1/T13b (rng), T8 (bsdf), T9 (lights),
//           T10/C0f + U11 + C0 analytic + split-dispatch equality + plants (pt), primary, emission;
//           T13 camera registration: the PT kernel on C0a ×4 (+ C0b) through marker_check.py.
//   Gate 1  (M3a items) our-PT-vs-Cycles planted biases (every emitter ×1.01; drop 1% of the paths at vertex 2),
//           detected by compare.py --calibrate --planted (≥ 9/10 half-size repeats, control ≥ 9/10), and an A/A test
//           on two of OUR PT seed sets.
//   Gate 2  Stage A: our PT ≡ Cycles (TOST δ 0.5% global / 2% per 32² tile on Y,R,G,B, suite FWER over all units) on
//           C0a–C0g, C0l–C0n (C0o deferred: Mode B), (i)–(v), (vii), (viii), (x)–(xii) and the keyframes of ix-a…g,
//           with cached Cycles references (render_reference.py) and fresh PT batches (run-batches.ts). A failed unit
//           is re-run once on disjoint seeds (compare.py --rerun-of). Analytic expectations (C0c/C0e/C0m/C0f) are
//           checked on BOTH renderers (analytic_check.py).
// Scenes: validation/scenes/make-m3a.ts (+ make-cornell-i.ts, make-spot-c0d.ts, calib_scenes.py). Output:
// validation/out/m3a-gate-<time>/ (summary.json, summary.md, per-unit compare reports, filed PT runs).
// Options: --only a,b (scene subset, for debugging; the gate verdict then covers only those units).
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { withGpuLockSync } from './gpu-lock.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PY = path.join(ROOT, 'validation/.venv/bin/python');
const BLENDER = process.env.BLENDER ?? '/Applications/Blender.app/Contents/MacOS/Blender';
const REFS_M2 = 'validation/out/m2/refs'; // reused so the M2 references stay cache hits
const REFS = 'validation/out/m3a/refs';
const IX_FRAMES = [0, 8, 16, 24, 32, 40, 48];
const OUR_SEED = 7, OUR_RERUN_SEED = 100_007;
/** Numerical-equivalence floor (stats.py GateSpec.num_eps; docs/decisions/cycles-deviations.md D2): tiles whose SE_Δ is
 *  below 1e-4 of their mean (deterministic direct light, furnaces) must MATCH to 2e-4 (+ their own noise) instead of Δ = 0 tests
 *  that resolve f32 arithmetic differences (measured 3e-6 … 4e-5). TOST δ is unchanged. */
const NUM_EPS = 1e-4;
const NUM_EPS_NOTE = 'f32 arithmetic floor between two implementations (cycles-deviations.md D2)';

/** One Gate-2 unit set: a package, the Cycles reference (spp × K seeds) and our PT (spp × B batches). */
export interface G2 {
  pkg: string;
  label: string;
  cyclesSpp: number;
  K: number;
  ourSpp: number;
  B: number;
  frames?: number[];
  refs?: string;
  /** test.json overrides (δ, min_replicates, tile, checks) — every deviation from the stage default is recorded. */
  test?: Record<string, unknown>;
  analytic?: boolean;
  note?: string;
}

const C = (pkg: string, label: string, o: Partial<G2> = {}): G2 => ({ pkg, label, cyclesSpp: 1024, K: 16, ourSpp: 2048, B: 16, ...o });
const V = (pkg: string, label: string, o: Partial<G2> = {}): G2 => ({ pkg, label, cyclesSpp: 1024, K: 16, ourSpp: 1024, B: 16, ...o });
const IX = (pkg: string, label: string, o: Partial<G2> = {}): G2 => ({
  pkg, label, cyclesSpp: 2048, K: 8, ourSpp: 2048, B: 8, frames: IX_FRAMES,
  test: { min_replicates: 8, aggregate_note: 'keyframes: K = 8 Cycles seeds and B = 8 PT batches per frame (task spec)' }, ...o,
});

export const M3A_G2: G2[] = [
  C('c0a_512', 'C0a squares (camera registration)', { cyclesSpp: 256, ourSpp: 256 }),
  C('c0b_512', 'C0b emission units', { cyclesSpp: 16, ourSpp: 16, refs: REFS_M2 }),
  C('c0c_point_256', 'C0c point', { analytic: true }),
  C('spot_c0d_512', 'C0d spot', { ourSpp: 1024, refs: REFS_M2 }),
  C('c0e_rect_256', 'C0e rect (one-sided)', { ourSpp: 4096, analytic: true }),
  C('c0f_furnace_b0_256', 'C0f furnace b=0', { analytic: true }),
  C('c0f_furnace_b1_256', 'C0f furnace b=1', { analytic: true }),
  C('c0f_furnace_b3_256', 'C0f furnace b=3', { analytic: true }),
  C('c0f_furnace_b7_256', 'C0f furnace b=7', { analytic: true }),
  C('c0g_bsdf_furnace_512x256', 'C0g BSDF furnace'),
  C('c0l_sun_256', 'C0l sun'),
  C('c0m_disk_256', 'C0m disk', { ourSpp: 4096, analytic: true }),
  C('c0n_spread_256', 'C0n spread 30/90', { ourSpp: 8192 }),
  V('cornell_i_512', '(i) Cornell rect', { refs: REFS_M2 }),
  V('ii_cornell_point_512', '(ii) Cornell point'),
  V('iii_spot_grazing_512', '(iii) spot grazing', { cyclesSpp: 4096, ourSpp: 4096 }),
  V('iv_emissive_mesh_512', '(iv) emissive mesh + rect'),
  V('v_glossy_v1_sharp_512', '(v) V1 GGX r 0.05/0.1', { test: { delta: { tile: 0.03 } }, note: 'tile δ 3% for r ≤ 0.1 (plan §5 M3a (v))' }),
  V('v_glossy_v1_512', '(v) V1 GGX r 0.2–0.8'),
  V('v_glossy_v2_512', '(v) V2 Principled sweep'),
  V('vii_textured_512', '(vii) textures, flat'),
  V('viii_usd_cornell_512', '(viii) USD Cornell'),
  V('x_many_lights_512', '(x) many lights', { cyclesSpp: 4096, ourSpp: 4096 }),
  V('xi_contact_512', '(xi) contact geometry'),
  V('xii_alpha_foliage_512', '(xii) alpha MASK foliage'),
  IX('ix_a_point_256', 'ix-a moving point'),
  IX('ix_b_area_256', 'ix-b moving area'),
  IX('ix_c_spot_256', 'ix-c rotating spot', { cyclesSpp: 4096, ourSpp: 4096, K: 16, B: 16, test: {} }),
  IX('ix_d_camera_256', 'ix-d moving camera'),
  IX('ix_e_steps_256', 'ix-e add/remove/steps'),
  IX('ix_f_combined_256', 'ix-f camera+lights, teleport, FOV ramp'),
  IX('ix_g_sun_256', 'ix-g rotating sun'),
];

/** Gate-1 items of M3a (plan §5 M3a exit): planted biases in OUR PT vs Cycles (i), and our A/A. */
const CAL_PKG = 'cornell_i_512';
const PLANTS: { name: string; args: string[]; seed: number }[] = [
  { name: 'emit x1.01', args: ['--plant-emit-scale', '1.01'], seed: 3003 },
  { name: 'drop 1% of paths at vertex 2', args: ['--plant-drop', '0.01@2'], seed: 4004 },
];
const T13_PKGS = ['c0a_512', 'c0a_640x360', 'c0a_360x640', 'c0a_far_512', 'c0b_512'];

type Rec = (name: string, ok: boolean, detail?: string) => void;
interface Step { name: string; ok: boolean; seconds: number; data?: unknown; detail?: string }

const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const r4 = (x: unknown) => (typeof x === 'number' ? Number(x.toPrecision(4)) : x);
const pct = (x: number | undefined | null, d = 3) => (typeof x === 'number' ? `${(x * 100).toFixed(d)}%` : 'n/a');
const readJson = <T = Record<string, any>>(p: string): T => JSON.parse(readFileSync(path.join(ROOT, p), 'utf8')) as T;

function sh(cmd: string, argv: string[], echo: (l: string) => boolean = () => true): { code: number; out: string; seconds: number } {
  const t0 = performance.now();
  const r = spawnSync(cmd, argv, { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 28, timeout: 7_200_000 });
  const out = `${r.stdout ?? ''}${r.stderr ?? ''}`;
  for (const l of out.split('\n')) if (l && echo(l)) console.log(`  ${l}`);
  return { code: r.status ?? 1, out, seconds: (performance.now() - t0) / 1000 };
}

/** run-batches.ts writes validation/out/<run>/; file finished runs under the gate dir. */
function fileRun(run: string, dest: string): string {
  const to = path.join(dest, run);
  mkdirSync(path.join(ROOT, dest), { recursive: true });
  rmSync(path.join(ROOT, to), { recursive: true, force: true });
  if (existsSync(path.join(ROOT, 'validation/out', run))) renameSync(path.join(ROOT, 'validation/out', run), path.join(ROOT, to));
  return to;
}

export function milestoneM3a(record: Rec, only?: Set<string>): void {
  const t0 = performance.now();
  const runId = `m3a-gate-${stamp()}`;
  const dir = path.join('validation/out', runId);
  mkdirSync(path.join(ROOT, dir), { recursive: true });
  const steps: Step[] = [];
  const add = (name: string, ok: boolean, seconds: number, data?: unknown, detail?: string) => {
    steps.push({ name, ok, seconds: Math.round(seconds * 10) / 10, data, detail });
    record(name, ok, `${detail ? `${detail}, ` : ''}${seconds.toFixed(1)} s`);
  };
  const runStep = (name: string, cmd: string, argv: string[], echo?: (l: string) => boolean) => {
    console.log(`\n--- ${name}: ${cmd} ${argv.join(' ')}`);
    const r = sh(cmd, argv, echo);
    add(name, r.code === 0, r.seconds, undefined, `exit ${r.code}`);
    return r;
  };
  const scenes = M3A_G2.filter((s) => !only || only.has(s.pkg));
  const full = !only;

  // ---- Gate 0 ---------------------------------------------------------------------------------------------------
  const missing = [PY, BLENDER].filter((p) => !existsSync(p));
  add('M3a gate tools (venv python, Blender)', missing.length === 0, 0, { missing }, missing.length ? `missing: ${missing.join(', ')}` : undefined);
  if (full) {
    runStep('typecheck', 'npx', ['tsc', '--noEmit']);
    runStep('vitest cpu', 'npx', ['vitest', 'run', '--project', 'cpu'], (l) => /Test Files|Tests |FAIL|✗|×/.test(l));
    runStep('python stats tests (validation/tools/tests)', PY, ['-m', 'pytest', 'validation/tools/tests', '-q'], (l) => /passed|failed|error/i.test(l));
    packagesCurrent(dir, add);
    withGpuLockSync('gate-m3a', () => {
      for (const [name, file] of [
        ['T1 path RNG + T13b jitter i.i.d. (chrome)', 'rng'],
        ['T8 BSDF χ²/weights/LUT vectors = gap-bsdf U-1…U-10 (chrome)', 'bsdf'],
        ['T9 light pdf / selection / NEE-BSDF partition (chrome)', 'lights'],
        ['T10 furnaces, U11 primary emission, C0 analytic, split-dispatch equality, plants (chrome)', 'pt'],
        ['primary pass / MASK / jitter (chrome)', 'primary'],
        ['emission kernel (chrome)', 'emission'],
      ] as const) runStep(name, 'npx', ['vitest', 'run', '--project', 'chrome', `validation/gpu-tests/${file}.gpu.test.ts`], (l) => /Tests |FAIL|✗|×|AssertionError/.test(l));
    });
    t13(dir, runId, add);
  }

  // ---- Gate 2: references, PT batches, Stage-A comparisons --------------------------------------------------------
  const units = scenes.reduce((n, s) => n + (s.frames?.length ?? 1), 0) + (full ? 1 + PLANTS.length : 0);
  const nUnits = 4 * (full ? units : M3A_G2.reduce((n, s) => n + (s.frames?.length ?? 1), 0) + 1 + PLANTS.length);
  console.log(`\nGate 2: ${scenes.length} scenes, ${units} units; suite FWER over n_units = ${nUnits} (× {Y,R,G,B})`);
  const results: Record<string, unknown>[] = [];
  const refDirs: Record<string, string> = {};
  for (const s of scenes) {
    const ref = cyclesRef(s, s.refs ?? REFS, s.K === 16 && s.refs === REFS_M2 ? '0..15' : `0..${s.K - 1}`, add);
    if (!ref) continue;
    refDirs[s.pkg] = ref;
    const ours = ourRun(s, `${runId}-${s.pkg}`, OUR_SEED, dir, add);
    if (!ours) continue;
    for (const f of s.frames ?? [undefined]) {
      const oursDir = f === undefined ? ours.dirs[0] : ours.dirs[s.frames!.indexOf(f)];
      results.push(stageA(s, f, oursDir, ref, dir, runId, nUnits, add));
    }
    if (s.analytic) {
      for (const [side, d] of [['ours', ours.dirs[0]], ['cycles', ref]] as const) analytic(s, side, d, add);
    }
  }

  // ---- Gate 1 (M3a): planted biases in our PT vs Cycles, our A/A ----------------------------------------------------
  if (full) {
    const cal = M3A_G2.find((s) => s.pkg === CAL_PKG)!;
    const ref = refDirs[CAL_PKG] ?? cyclesRef(cal, cal.refs ?? REFS, '0..15', add);
    if (ref) {
      for (const p of PLANTS) plantDetection(cal, p, ref, dir, runId, nUnits, add);
      ourAA(cal, dir, runId, nUnits, add);
    }
  }

  // ---- summary --------------------------------------------------------------------------------------------------
  const failed = steps.filter((s) => !s.ok).map((s) => s.name);
  const summary = {
    milestone: 'M3a', gate: 'Gate 0 + Gate 1 (M3a items) + Gate 2 Stage A (plan §5 M3a exit)', run: runId, created: new Date().toISOString(),
    ok: failed.length === 0, subset: only ? [...only] : undefined, total_s: Math.round((performance.now() - t0) / 100) / 10, n_units: nUnits, failed,
    deferred: ['C0o visibleToCamera: Mode B only (Mode A forbids camera-visible area lights); the PT is Mode A until M3b'],
    units: results, steps,
  };
  writeFileSync(path.join(ROOT, dir, 'summary.json'), `${JSON.stringify(summary, null, 1)}\n`);
  writeFileSync(path.join(ROOT, dir, 'summary.md'), table(results));
  console.log(`\n${table(results)}\nsummary: ${dir}/summary.json (${summary.total_s} s)`);
}

function table(results: Record<string, any>[]): string {
  const head = '| unit | status | global Δ_Y | MDB_Y global | worst tile (rel) | tile MDB max | TOST-failed tiles | Cycles spp×K | PT spp×B | mult. needed |\n|---|---|---|---|---|---|---|---|---|---|\n';
  return head + results.map((r) => `| ${r.unit} | ${r.status} | ${pct(r.global_rel_Y, 4)} | ${pct(r.mdb_global_Y, 4)} | ${pct(r.worst_tile_rel_Y, 2)} ${r.worst_tile ?? ''} | ${pct(r.mdb_tile_max_Y, 2)} | ${r.tost_failed_tiles ?? ''} | ${r.cycles} | ${r.ours} | ${r4(r.multiplier_needed)} |`).join('\n') + '\n';
}

/** Generator-made packages must regenerate byte-identically (else the cached references show a stale scene). */
function packagesCurrent(dir: string, add: (n: string, ok: boolean, s: number, d?: unknown, det?: string) => void): void {
  const t0 = performance.now();
  const tmp = path.join(ROOT, dir, 'regen');
  const diffs: string[] = [];
  for (const gen of ['make-m3a.ts', 'make-cornell-i.ts', 'make-spot-c0d.ts']) {
    const r = sh('npx', ['tsx', `validation/scenes/${gen}`, tmp], () => false);
    if (r.code !== 0) diffs.push(`${gen}: exit ${r.code} ${r.out.slice(-300)}`);
  }
  for (const p of existsSync(tmp) ? readdirSync(tmp) : []) {
    const a = path.join(tmp, p), b = path.join(ROOT, 'validation/scenes', p);
    const names = new Set([...readdirSync(a), ...(existsSync(b) ? readdirSync(b) : [])]);
    for (const n of names) if (!existsSync(path.join(a, n)) || !existsSync(path.join(b, n)) || !readFileSync(path.join(a, n)).equals(readFileSync(path.join(b, n)))) diffs.push(`${p}/${n}`);
  }
  rmSync(tmp, { recursive: true, force: true });
  add('Gate-2 scene packages current (make-m3a.ts, make-cornell-i.ts, make-spot-c0d.ts)', diffs.length === 0, (performance.now() - t0) / 1000, { diffs }, diffs.length ? `stale: ${diffs.slice(0, 8).join(', ')}` : undefined);
}

/** T13 camera registration with the PT kernel: C0a markers (3 aspects + far/recentred) and C0b constants. */
function t13(dir: string, runId: string, add: (n: string, ok: boolean, s: number, d?: unknown, det?: string) => void): void {
  for (const pkg of T13_PKGS) {
    const run = `${runId}-t13-${pkg}`;
    const spp = pkg === 'c0b_512' ? 16 : 256;
    const r = sh('npx', ['tsx', 'validation/harness/run-batches.ts', '--package', `validation/scenes/${pkg}`, '--kernel', 'pt', '--spp', String(spp),
      '--batches', '8', '--seed', '11', '--run', run, '--check'], (l) => /^(OK|FAIL)\s|marker_check|max centroid|RESULT|FAIL/.test(l));
    const m = /max(?:imum)? centroid err(?:or)? ([\d.e-]+)/i.exec(r.out) ?? /max_err_px["=: ]+([\d.e-]+)/.exec(r.out);
    add(`T13 camera registration: PT kernel on ${pkg} (marker_check)`, r.code === 0, r.seconds, { run: fileRun(run, path.join(dir, 't13')) },
      m ? `max centroid err ${m[1]} px` : `exit ${r.code}`);
  }
}

function cyclesRef(s: G2, out: string, seeds: string, add: (n: string, ok: boolean, sec: number, d?: unknown, det?: string) => void): string | undefined {
  console.log(`\n--- Cycles reference ${s.pkg}: ${s.cyclesSpp} spp × seeds ${seeds}${s.frames ? ` × frames ${s.frames.join(',')}` : ''}`);
  const r = sh(BLENDER, ['-b', '--factory-startup', '--python-exit-code', '1', '-P', 'validation/blender/render_reference.py', '--',
    '--package', `validation/scenes/${s.pkg}`, '--out', out, '--spp', String(s.cyclesSpp), '--seeds', seeds, ...(s.frames ? ['--frames', s.frames.join(',')] : [])],
  (l) => l.startsWith('[render_reference]') && (l.includes('RESULT') || l.includes('cache') || l.includes('waited')) || /Error|Traceback/.test(l));
  const line = r.out.split('\n').reverse().find((l) => l.startsWith('[render_reference] RESULT '));
  const res = line ? JSON.parse(line.slice('[render_reference] RESULT '.length)) as { dir: string; cache_hit: boolean; renders: number; renders_total_s?: number } : undefined;
  add(`Cycles reference ${s.pkg} (${s.cyclesSpp} spp × ${seeds})`, r.code === 0 && !!res, r.seconds,
    res && { dir: path.relative(ROOT, res.dir), cache_hit: res.cache_hit, renders: res.renders, renders_total_s: res.renders_total_s ?? 0 },
    res ? (res.cache_hit ? 'cache hit' : `rendered ${res.renders} in ${res.renders_total_s} s`) : `exit ${r.code} ${r.out.slice(-400)}`);
  return res ? path.relative(ROOT, res.dir) : undefined;
}

function ourRun(s: G2, run: string, seed: number, dir: string, add: (n: string, ok: boolean, sec: number, d?: unknown, det?: string) => void, extra: string[] = [], label = 'PT batches'):
  { dirs: string[]; ms: number } | undefined {
  console.log(`\n--- ${label} ${s.pkg}: ${s.ourSpp} spp × ${s.B} batches, seed ${seed}${extra.length ? ` ${extra.join(' ')}` : ''}`);
  const r = sh('npx', ['tsx', 'validation/harness/run-batches.ts', '--package', `validation/scenes/${s.pkg}`, '--kernel', 'pt', '--spp', String(s.ourSpp),
    '--batches', String(s.B), '--seed', String(seed), '--run', run, ...(s.frames ? ['--frames', s.frames.join(',')] : []), ...extra],
  (l) => /^(FAIL)\s|errors:|Error/.test(l));
  const runs = s.frames ? s.frames.map((f) => `${run}-f${f}`) : [run];
  const dirs = runs.map((x) => fileRun(x, path.join(dir, 'pt')));
  const ms = dirs.reduce((t, d) => t + (existsSync(path.join(ROOT, d, 'meta.json')) ? readJson(path.join(d, 'meta.json')).timings.totalMs : 0), 0);
  const ok = r.code === 0 && dirs.every((d) => existsSync(path.join(ROOT, d, 'meta.json')));
  add(`${label} ${s.pkg} (${s.ourSpp} spp × ${s.B}${s.frames ? ` × ${s.frames.length} frames` : ''})`, ok, r.seconds, { dirs, gpu_total_ms: Math.round(ms) },
    ok ? `${(ms / 1000).toFixed(1)} s in the page` : `exit ${r.code} ${r.out.slice(-300)}`);
  return ok ? { dirs, ms } : undefined;
}

function writeTest(dir: string, name: string, nUnits: number, extra: Record<string, unknown> = {}): string {
  const p = path.join(dir, 'tests', `test-${name}.json`);
  mkdirSync(path.join(ROOT, dir, 'tests'), { recursive: true });
  writeFileSync(path.join(ROOT, p), JSON.stringify({ name, stage: 'A', channels: ['Y', 'R', 'G', 'B'], n_units: nUnits, tier: 'tight', num_eps: NUM_EPS, num_eps_note: NUM_EPS_NOTE, ...extra }, null, 1));
  return p;
}

function compare(ours: string, ref: string, test: string, out: string, frame?: number, rerunOf?: string) {
  return sh(PY, ['validation/tools/compare.py', '--ours', ours, '--ref', ref, '--test', test, '--out', out,
    ...(frame !== undefined ? ['--frame', String(frame)] : []), ...(rerunOf ? ['--rerun-of', rerunOf] : [])], () => false);
}

function summarize(rep: Record<string, any>) {
  const Y = rep.channels.Y;
  return {
    status: rep.status, failed_checks: rep.failed_checks,
    global_rel_Y: r4(Y.global_.rel), mdb_global_Y: r4(Y.global_.mdb), worst_tile_rel_Y: r4(Y.tiles.worst_rel), worst_tile: Y.tiles.worst_tile ? `[${Y.tiles.worst_tile.join(',')}]` : undefined,
    mdb_tile_max_Y: r4(Y.tiles.mdb_max), mdb_tile_median_Y: r4(Y.tiles.mdb_median), tost_failed_tiles: Y.tiles.tost_failed, zero_var_tiles: Y.tiles.zero_var,
    multiplier_needed: r4(Math.max(...['Y', 'R', 'G', 'B'].map((c) => Math.max(rep.channels[c].global_.replicate_multiplier_needed ?? 1, rep.channels[c].tiles.replicate_multiplier_needed ?? 1)))),
    per_channel: Object.fromEntries(['R', 'G', 'B'].map((c) => [c, { global_rel: r4(rep.channels[c].global_.rel), worst_tile_rel: r4(rep.channels[c].tiles.worst_rel), tost_failed: rep.channels[c].tiles.tost_failed }])),
  };
}

function stageA(s: G2, frame: number | undefined, ours: string, ref: string, dir: string, runId: string, nUnits: number,
  add: (n: string, ok: boolean, sec: number, d?: unknown, det?: string) => void): Record<string, unknown> {
  const unit = frame === undefined ? s.pkg : `${s.pkg}@f${frame}`;
  const test = writeTest(dir, unit, nUnits, { ...(frame !== undefined ? { frame } : {}), ...(s.test ?? {}) });
  const out = path.join(dir, 'compare', unit);
  const t0 = performance.now();
  let c = compare(ours, ref, test, out, frame);
  let rep = existsSync(path.join(ROOT, out, 'report.json')) ? readJson(path.join(out, 'report.json')) : undefined;
  let rerun: Record<string, unknown> | undefined;
  if (rep?.status === 'rerun_required') {
    // suite FWER rule (plan §7.3): re-run once on disjoint seeds, both sides
    console.log(`  ${unit}: rerun_required (${rep.failed_checks.join(', ')}) -> confirmatory re-run on disjoint seeds`);
    const ref2 = cyclesRef({ ...s, frames: frame !== undefined ? [frame] : undefined }, s.refs ?? REFS, `${100}..${100 + s.K - 1}`, add);
    const o2 = ourRun({ ...s, frames: frame !== undefined ? [frame] : undefined }, `${runId}-${unit.replace('@', '-')}-rerun`, OUR_RERUN_SEED, dir, add, [], 'PT re-run batches');
    if (ref2 && o2) {
      const out2 = `${out}-rerun`;
      c = compare(o2.dirs[0], ref2, test, out2, frame, path.join(out, 'report.json'));
      const rep2 = existsSync(path.join(ROOT, out2, 'report.json')) ? readJson(path.join(out2, 'report.json')) : undefined;
      rerun = { first: summarize(rep), report: path.join(out2, 'report.json') };
      rep = rep2;
    }
  }
  const sum = rep ? summarize(rep) : undefined;
  const ok = !!rep && (rep.status === 'pass' || rep.status === 'pass_on_rerun');
  const res = {
    unit, label: s.label, frame, ...(sum ?? { status: `error (exit ${c.code})` }), cycles: `${s.cyclesSpp}×${s.K}`, ours: `${s.ourSpp}×${s.B}`,
    report: path.join(out, 'report.json'), ...(rerun ? { rerun } : {}), ...(s.note ? { note: s.note } : {}), ...(rep?.notes && Object.keys(rep.notes).length ? { notes: rep.notes } : {}),
  };
  add(`Stage A ${unit} (${s.label})`, ok, (performance.now() - t0) / 1000, res,
    sum ? `${sum.status}: Δ_Y ${pct(sum.global_rel_Y as number, 4)} (MDB ${pct(sum.mdb_global_Y as number, 3)}), worst tile ${pct(sum.worst_tile_rel_Y as number, 2)}, tile MDB max ${pct(sum.mdb_tile_max_Y as number, 2)}${sum.failed_checks?.length ? `, failed ${sum.failed_checks.join(' ')}` : ''}` : `compare exit ${c.code} ${c.out.slice(-300)}`);
  return res;
}

function analytic(s: G2, side: string, d: string, add: (n: string, ok: boolean, sec: number, d?: unknown, det?: string) => void): void {
  const r = sh(PY, ['validation/tools/analytic_check.py', '--dir', d, '--package', `validation/scenes/${s.pkg}`, '--json'], () => false);
  let res: Record<string, any> = {};
  try { res = JSON.parse(r.out.trim().split('\n').pop()!); } catch { res = { ok: false, error: r.out.slice(-300) }; }
  if (res.ok === null) return; // not applicable
  const Y = res.channels?.Y;
  add(`analytic ${s.pkg} (${side})`, res.ok === true, r.seconds, res, Y ? `Δ_Y ${pct(Y.rel, 4)} vs closed form (z ${Y.z.toFixed(2)}), worst tile ${pct(Y.worst_tile_rel, 2)}` : res.error);
}

function plantDetection(s: G2, p: { name: string; args: string[]; seed: number }, ref: string, dir: string, runId: string, nUnits: number,
  add: (n: string, ok: boolean, sec: number, d?: unknown, det?: string) => void): void {
  const o = ourRun(s, `${runId}-plant-${p.seed}`, p.seed, dir, add, p.args, `PT planted (${p.name})`);
  if (!o) return;
  const test = writeTest(dir, `gate1-plant-${p.seed}`, nUnits, { calibration: { plants: [] } });
  const out = path.join(dir, 'calibrate', `plant-${p.seed}`);
  const r = sh(PY, ['validation/tools/compare.py', '--calibrate', '--ref', ref, '--planted', o.dirs[0], '--plant-name', `our PT ${p.name}`, '--test', test, '--out', out,
    '--splits', '20', '--repeats', '10'], () => false);
  let data: Record<string, unknown> | undefined;
  let detail = `exit ${r.code}`;
  let ok = false;
  if (existsSync(path.join(ROOT, out, 'report.json'))) {
    const rep = readJson(path.join(out, 'report.json'));
    const rp = rep.calibration.rendered_plant;
    const full = compare(o.dirs[0], ref, writeTest(dir, `gate1-plant-${p.seed}-full`, nUnits), `${out}-full`);
    const fullRep = existsSync(path.join(ROOT, `${out}-full`, 'report.json')) ? readJson(path.join(`${out}-full`, 'report.json')) : undefined;
    ok = !!rp && rp.gate_fail_count >= 9 && rp.control_pass_count >= 9 && !!fullRep && fullRep.status !== 'pass';
    data = {
      report: path.join(out, 'report.json'), detected: `${rp.gate_fail_count}/${rp.n_repeats}`, equivalence_failed: `${rp.equivalence_fail_count}/${rp.n_repeats}`,
      control_pass: `${rp.control_pass_count}/${rp.n_repeats}`, half_size: rp.half_size, planted_rel_Y: r4(rp.channels.Y.planted_global_rel_median),
      mdb_global_Y: r4(rp.channels.Y.mdb_global_median), failed_checks_Y: Object.keys(rp.failed_checks_histogram).filter((k) => k.endsWith('[Y]')),
      full_compare: fullRep && { status: fullRep.status, global_rel_Y: r4(fullRep.channels.Y.global_.rel), failed_checks: fullRep.failed_checks },
    };
    detail = `detected ${rp.gate_fail_count}/10 (TOST ${rp.equivalence_fail_count}/10), control ${rp.control_pass_count}/10, planted Δ_Y ${pct(rp.channels.Y.planted_global_rel_median, 3)}, MDB ${pct(rp.channels.Y.mdb_global_median, 4)}; full 16 vs 16: ${fullRep?.status}`;
  }
  add(`Gate 1 plant in our PT vs Cycles ${s.pkg}: ${p.name}`, ok, r.seconds, data, detail);
}

function ourAA(s: G2, dir: string, runId: string, nUnits: number, add: (n: string, ok: boolean, sec: number, d?: unknown, det?: string) => void): void {
  const a = ourRun(s, `${runId}-aa-1001`, 1001, dir, add, [], 'PT A/A seed set 1001');
  const b = ourRun(s, `${runId}-aa-2002`, 2002, dir, add, [], 'PT A/A seed set 2002');
  if (!a || !b) return;
  const out = path.join(dir, 'compare', `aa-ours-${s.pkg}`);
  const c = compare(a.dirs[0], b.dirs[0], writeTest(dir, `aa-ours-${s.pkg}`, nUnits), out);
  const rep = existsSync(path.join(ROOT, out, 'report.json')) ? readJson(path.join(out, 'report.json')) : undefined;
  const sum = rep ? summarize(rep) : undefined;
  add(`Gate 1 A/A on two of our PT seed sets (${s.pkg}, 1001 vs 2002)`, rep?.status === 'pass', 0, sum && { ...sum, report: path.join(out, 'report.json') },
    sum ? `${sum.status}: Δ_Y ${pct(sum.global_rel_Y as number, 4)} (MDB ${pct(sum.mdb_global_Y as number, 4)}), tile MDB max ${pct(sum.mdb_tile_max_Y as number, 2)}` : `exit ${c.code}`);
}
