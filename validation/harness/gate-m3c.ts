// M3c milestone gate (plan §5 M3c exit, §7.2 env scenes, §7.3, §7.4 M3c): `npm run validate -- --milestone M3c`.
//   Gate 0  typecheck, cpu lane (ENV-U3 CPU part, ENV-U5, ENV-U8, env as a light), python stats tests, scene-package
//           determinism (make-m3c.ts twice → byte-identical), Chrome GPU tests ENV-U3 (χ² + realized-pdf identity),
//           ENV-U4 (support), ENV-U6 (NEE/BSDF partition vs quadrature at pmf[ENV] ∈ {1, 0.3}) and the env/lights/pt
//           regressions (ENV-U2/U7, T9, T10/U11/C0 probes); the M3c app smoke (validation/harness/m3c-app-smoke.ts).
//   Gate 2  Stage A: our PT ≡ Cycles (TOST δ 0.5% global / 2% per 32² tile, Y/R/G/B, suite FWER) in Mode A on C0q (both
//           Blender world variants: constant texture with NEE, and constant Background = BSDF-only; the two Cycles
//           variants must also agree with each other), C0r, C0s (env NEE on/off), (xiii) and (xiv); analytic
//           expectations (validation/tools/env-expected.ts, f64 quadrature) are checked on BOTH renderers for C0q
//           (except the open box), C0r and C0s. A failed unit is re-run once on disjoint seeds. The tier (tight /
//           heavy-tail, K and B doubled) is recorded per unit.
//   Env plants (plan §7.4 M3c): strength ×1.0075, sinθ missing in pdf_σ, pdf from target weights, ω2 without
//           pmf[ENV], NEE and BSDF both weight 1 — each detected in ≥ 9/10 half-size repeats (compare.py --calibrate
//           --planted, Cycles A/A control ≥ 9/10) and failing the full comparison.
//   Negative controls (must pass Stage A): no floors, power heuristic, importance resolution 256.
//   A/A: two of our seed sets on C0r.
// Scenes: validation/scenes/make-m3c.ts → validation/out/m3c/scenes (generated; HDRIs from fetch_hdris.ts).
// Output: validation/out/m3c-gate-<time>/ (summary.json, summary.md, per-unit reports, expected images, PT runs).
// Options: --only a,b (scene subset; skips Gate 0, plants, controls).
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PY = path.join(ROOT, 'validation/.venv/bin/python');
const BLENDER = process.env.BLENDER ?? '/Applications/Blender.app/Contents/MacOS/Blender';
const GPU_LOCK = '/tmp/restirpt-gpu.lock';
const SCENES = 'validation/out/m3c/scenes';
const REFS = 'validation/out/m3c/refs';
const OUR_SEED = 7, OUR_RERUN_SEED = 100_007;
const NUM_EPS = 1e-4;
const NUM_EPS_NOTE = 'f32 arithmetic floor between two implementations (cycles-deviations.md D2)';

export interface G2E {
  pkg: string;
  label: string;
  cyclesSpp: number;
  K: number;
  ourSpp: number;
  B: number;
  tier: 'tight' | 'heavy-tail';
  /** Compare against validation/tools/env-expected.ts (both renderers). */
  analytic?: boolean;
  /** test.json overrides (recorded). */
  test?: Record<string, unknown>;
  /** Extra run-batches.ts arguments for our side. */
  ourArgs?: string[];
  /** First Cycles seed (default 0; the C0q "_bg" twins use 200.. so the two Cycles variants are independent). */
  seedBase?: number;
  note?: string;
}

const T = (pkg: string, label: string, o: Partial<G2E> = {}): G2E => ({ pkg, label, cyclesSpp: 1024, K: 16, ourSpp: 1024, B: 16, tier: 'tight', ...o });
/** Heavy-tail tier: K and B doubled (plan §7.2 "pilot-sized, with K doubled"). */
const HT = (pkg: string, label: string, o: Partial<G2E> = {}): G2E => T(pkg, label, { K: 32, B: 32, tier: 'heavy-tail', ...o });

export const M3C_G2: G2E[] = [
  ...[0, 1, 3].flatMap((b) => [
    T(`c0q_lambert_b${b}_256`, `C0q Lambert sphere b=${b} (constant texture, NEE)`, { analytic: true }),
    T(`c0q_lambert_b${b}_256_bg`, `C0q Lambert sphere b=${b} (Cycles constant Background, BSDF-only)`, { analytic: true, seedBase: 200 }),
  ]),
  T('c0q_quad_b1_256', 'C0q Lambert quad (texture)', { analytic: true }),
  T('c0q_quad_b1_256_bg', 'C0q Lambert quad (Background)', { analytic: true, seedBase: 200 }),
  T('c0q_ggx02_b1_256', 'C0q GGX alpha 0.2 sphere (texture)', { analytic: true }),
  T('c0q_ggx02_b1_256_bg', 'C0q GGX alpha 0.2 sphere (Background)', { analytic: true, seedBase: 200 }),
  T('c0q_ggx05_b1_256', 'C0q GGX alpha 0.5 sphere (texture)', { analytic: true }),
  T('c0q_ggx05_b1_256_bg', 'C0q GGX alpha 0.5 sphere (Background)', { analytic: true, seedBase: 200 }),
  T('c0q_openbox_b13_256', 'C0q rho=1 open box b=13 (texture)'),
  T('c0q_openbox_b13_256_bg', 'C0q rho=1 open box b=13 (Background)', { seedBase: 200 }),
  T('c0r_irradiance_256', 'C0r Lambert irradiance sphere, overcast', { analytic: true }),
  T('c0r_mirror_256', 'C0r mirror sphere, overcast', { analytic: true }),
  ...(['45', 'seam', 'top'] as const).flatMap((p) => [
    T(`c0s_${p}_nee_256`, `C0s sun texel ${p}, env NEE on`, { analytic: true }),
    // BSDF-only hits the 1e-4 sr texel with p ≈ 2.4e-5 per sample: 16× the spp of the NEE variant (pilot: tile MDB 2.0% and a
    // replicate multiplier of 2.7 at 4096 spp).
    T(`c0s_${p}_none_256`, `C0s sun texel ${p}, env NEE off (sampling_method NONE)`, { analytic: true, cyclesSpp: 16384, ourSpp: 16384 }),
  ]),
  HT('xiii_spheres_512x256', '(xiii) GGX r 0..0.5 + V2 metal spheres, studio_small_09'),
  T('xiv_overcast_b3_512', '(xiv) Cornell open, overcast, b=3'),
  T('xiv_overcast_rect_b3_512', '(xiv) Cornell open, overcast + rect light, b=3'),
  T('xiv_overcast_b1_512', '(xiv) Cornell open, overcast, b=1'),
  T('xiv_overcast_b7_512', '(xiv) Cornell open, overcast, b=7'),
  HT('xiv_kloof_b3_512', '(xiv) Cornell open, kloofendal (sun), b=3'),
  HT('xiv_kloof_rect_b3_512', '(xiv) Cornell open, kloofendal (sun) + rect light, b=3'),
];

/** The two Cycles C0q world variants must agree (constant texture with NEE vs constant Background, BSDF-only). */
const C0Q_PAIRS = M3C_G2.filter((s) => s.pkg.startsWith('c0q_') && s.pkg.endsWith('_bg')).map((s) => [s.pkg.replace(/_bg$/, ''), s.pkg] as const);

/** Env planted biases (env §5.3): scene + run-batches args. */
const PLANTS: { name: string; pkg: string; args: string[]; seed: number }[] = [
  { name: 'env strength x1.0075', pkg: 'c0r_irradiance_256', args: ['--env-strength-scale', '1.0075'], seed: 3101 },
  { name: 'sin(theta) missing in pdf_sigma', pkg: 'c0s_45_nee_256', args: ['--env-plant', 'missingSin'], seed: 3102 },
  { name: 'pdf from target weights', pkg: 'c0s_45_nee_256', args: ['--env-plant', 'pdfFromTargets'], seed: 3103 },
  { name: 'omega2 without pmf[ENV]', pkg: 'xiv_overcast_rect_b3_512', args: ['--env-plant', 'w2WithoutPmf'], seed: 3104 },
  { name: 'NEE and BSDF both weight 1', pkg: 'c0r_irradiance_256', args: ['--env-plant', 'doubleCount'], seed: 3105 },
];

/** Negative controls (must pass Stage A): still unbiased, only the variance changes. */
const CONTROLS: { name: string; args: string[] }[] = [
  { name: 'no floors', args: ['--env-no-floors'] },
  { name: 'power heuristic', args: ['--env-mis-power'] },
  { name: 'importance resolution 256', args: ['--env-cap', '256'] },
];
const CONTROL_PKGS = ['c0s_45_nee_256', 'xiv_overcast_rect_b3_512'];
const AA_PKG = 'c0r_irradiance_256';

type Add = (name: string, ok: boolean, seconds: number, data?: unknown, detail?: string) => void;
type Rec = (name: string, ok: boolean, detail?: string) => void;

const seedRange = (s: G2E) => `${s.seedBase ?? 0}..${(s.seedBase ?? 0) + s.K - 1}`;
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

function withGpuLock<T>(fn: () => T): T {
  const nap = new Int32Array(new SharedArrayBuffer(4));
  for (;;) { try { mkdirSync(GPU_LOCK); break; } catch { console.log('waiting for the GPU lock ...'); Atomics.wait(nap, 0, 0, 5000); } }
  const release = () => { try { rmSync(GPU_LOCK, { recursive: true, force: true }); } catch { /* gone */ } };
  process.once('exit', release);
  try { return fn(); } finally { release(); process.removeListener('exit', release); }
}

function fileRun(run: string, dest: string): string {
  const to = path.join(dest, run);
  mkdirSync(path.join(ROOT, dest), { recursive: true });
  rmSync(path.join(ROOT, to), { recursive: true, force: true });
  if (existsSync(path.join(ROOT, 'validation/out', run))) renameSync(path.join(ROOT, 'validation/out', run), path.join(ROOT, to));
  return to;
}

export function milestoneM3c(record: Rec, only?: Set<string>): void {
  const t0 = performance.now();
  const runId = `m3c-gate-${stamp()}`;
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
  const full = !only;
  const scenes = M3C_G2.filter((s) => !only || only.has(s.pkg));

  // ---- Gate 0 ---------------------------------------------------------------------------------------------------
  const missing = [PY, BLENDER].filter((p) => !existsSync(p));
  add('M3c gate tools (venv python, Blender)', missing.length === 0, 0, { missing }, missing.length ? `missing: ${missing.join(', ')}` : undefined);
  packagesDeterministic(dir, add, only);
  if (full) {
    runStep('typecheck', 'npx', ['tsc', '--noEmit']);
    runStep('vitest cpu (ENV-U3 CPU, ENV-U5, ENV-U8, env as a light)', 'npx', ['vitest', 'run', '--project', 'cpu'], (l) => /Test Files|Tests |FAIL|✗|×/.test(l));
    runStep('python stats tests (validation/tools/tests)', PY, ['-m', 'pytest', 'validation/tools/tests', '-q'], (l) => /passed|failed|error/i.test(l));
    withGpuLock(() => {
      for (const [name, file] of [
        ['ENV-U3 realized-pdf identity + GPU χ², ENV-U4 support, ENV-U6 NEE/BSDF partition (chrome)', 'env-sampling'],
        ['ENV-U2 mapping + ENV-U7 bilinear/pole wrap (chrome, regression)', 'env'],
        ['T9 light pdf / selection / NEE-BSDF partition (chrome, regression)', 'lights'],
        ['T10 furnaces, U11, C0 analytic, split dispatch, plants (chrome, regression)', 'pt'],
      ] as const) {
        runStep(name, 'npx', ['vitest', 'run', '--project', 'chrome', '--reporter=verbose', `validation/gpu-tests/${file}.gpu.test.ts`],
          (l) => /Tests |FAIL|✗|×|AssertionError|^ENV-U[346] /.test(l));
      }
    });
    {
      const b = readJson('validation/budget.json') as { env_entries?: { scene: string; cycles_s_per_4096spp: number }[] };
      const want = ['c0q_', 'c0r_', 'xiii_', 'xiv_'];
      const miss = want.filter((w) => !b.env_entries?.some((e) => e.scene.startsWith(w) && e.cycles_s_per_4096spp > 0));
      add('budget.json env rows (C0q, C0r, (xiii), (xiv); validation/tools/budget_env.py)', miss.length === 0, 0, { rows: b.env_entries?.length ?? 0 }, miss.length ? `missing ${miss.join(', ')}` : `${b.env_entries!.length} rows`);
    }
    // Takes the GPU lock itself: env panel (NEE toggle, importance resolution, P(env)), Worker tables, env debug views.
    runStep('M3c app smoke (env panel, Worker tables, env sampling debug views, splat χ²)', 'npx', ['tsx', 'validation/harness/m3c-app-smoke.ts', '--run', `${runId}-app-smoke`],
      (l) => /^(PASS|FAIL)\s/.test(l));
  }

  // ---- Gate 2 --------------------------------------------------------------------------------------------------
  const nUnits = 4 * (M3C_G2.length + C0Q_PAIRS.length + PLANTS.length + CONTROLS.length * CONTROL_PKGS.length + 1);
  console.log(`\nGate 2: ${scenes.length} scenes; suite FWER over n_units = ${nUnits} (× {Y,R,G,B})`);
  const results: Record<string, unknown>[] = [];
  const refDirs: Record<string, string> = {};
  const oursDirs: Record<string, string> = {};
  for (const s of scenes) {
    const ref = cyclesRef(s, REFS, seedRange(s), add);
    if (!ref) continue;
    refDirs[s.pkg] = ref;
    const ours = ourRun(s, `${runId}-${s.pkg}`, OUR_SEED, dir, add);
    if (!ours) continue;
    oursDirs[s.pkg] = ours.dirs[0];
    results.push(stageA(s, ours.dirs[0], ref, dir, runId, nUnits, add));
    if (s.analytic) analytic(s, { ours: ours.dirs[0], cycles: ref }, dir, add);
  }
  for (const [a, b] of C0Q_PAIRS) {
    if (!refDirs[a] || !refDirs[b]) continue;
    const s = M3C_G2.find((x) => x.pkg === a)!;
    results.push(stageA({ ...s, label: `C0q Cycles world variants agree (${a} texture+NEE vs Background BSDF-only)` }, refDirs[b], refDirs[a], dir, runId, nUnits, add,
      `cycles-variants-${a}`, false));
  }
  // C0s: NEE on vs off variance ratio (recorded; the four combinations are covered by the units + analytic checks).
  for (const p of ['45', 'seam', 'top']) {
    const on = results.find((r) => r.unit === `c0s_${p}_nee_256`), off = results.find((r) => r.unit === `c0s_${p}_none_256`);
    if (on && off && typeof on.mdb_global_Y === 'number' && typeof off.mdb_global_Y === 'number') {
      add(`C0s ${p}: env NEE on/off variance ratio (recorded)`, true, 0, { mdb_nee: on.mdb_global_Y, mdb_none: off.mdb_global_Y, spp: 'NEE 1024, NONE 16384 per replicate' },
        `SE(NONE, 16384 spp)/SE(NEE, 1024 spp) = ${((off.mdb_global_Y as number) / (on.mdb_global_Y as number)).toFixed(2)} → variance ratio per sample ≈ ${(16 * ((off.mdb_global_Y as number) / (on.mdb_global_Y as number)) ** 2).toFixed(0)}`);
    }
  }

  // ---- env plants, negative controls, A/A ------------------------------------------------------------------------
  if (full) {
    for (const p of PLANTS) {
      const s = M3C_G2.find((x) => x.pkg === p.pkg)!;
      const ref = refDirs[p.pkg] ?? cyclesRef(s, REFS, seedRange(s), add);
      if (ref) plantDetection(s, p, ref, dir, runId, nUnits, add);
    }
    for (const pkg of CONTROL_PKGS) {
      const s = M3C_G2.find((x) => x.pkg === pkg)!;
      const ref = refDirs[pkg] ?? cyclesRef(s, REFS, seedRange(s), add);
      if (!ref) continue;
      for (const c of CONTROLS) {
        const tag = c.name.replace(/\W+/g, '-');
        const o = ourRun({ ...s, ourArgs: c.args }, `${runId}-ctl-${tag}-${pkg}`, OUR_SEED + 17, dir, add, `PT negative control (${c.name})`);
        if (o) results.push(stageA({ ...s, label: `negative control: ${c.name} (${pkg})` }, o.dirs[0], ref, dir, runId, nUnits, add, `control-${tag}-${pkg}`, false));
      }
    }
    const s = M3C_G2.find((x) => x.pkg === AA_PKG)!;
    ourAA(s, dir, runId, nUnits, add);
  }

  // ---- summary ------------------------------------------------------------------------------------------------------
  const failed = steps.filter((x) => !x.ok).map((x) => x.name);
  const summary = {
    milestone: 'M3c', gate: 'Gate 0 + Gate 2 Stage A (env) + env plants + negative controls + A/A (plan §5 M3c exit)', run: runId, created: new Date().toISOString(),
    ok: failed.length === 0, subset: only ? [...only] : undefined, total_s: Math.round((performance.now() - t0) / 100) / 10, n_units: nUnits, failed,
    deferred: ['(xiv) Mode B and glass variants, (vi) env: after the M3b merge (make-m3c.ts --light-mode B / --glass)'],
    units: results, steps,
  };
  writeFileSync(path.join(ROOT, dir, 'summary.json'), `${JSON.stringify(summary, null, 1)}\n`);
  writeFileSync(path.join(ROOT, dir, 'summary.md'), table(results));
  console.log(`\n${table(results)}\nsummary: ${dir}/summary.json (${summary.total_s} s)`);
}

function table(results: Record<string, any>[]): string {
  const head = '| unit | tier | status | global Δ_Y | MDB_Y global | worst tile (rel) | tile MDB max | TOST-failed tiles | Cycles spp×K | PT spp×B | mult. needed |\n|---|---|---|---|---|---|---|---|---|---|---|\n';
  return head + results.map((r) => `| ${r.unit} | ${r.tier} | ${r.status} | ${pct(r.global_rel_Y, 4)} | ${pct(r.mdb_global_Y, 4)} | ${pct(r.worst_tile_rel_Y, 2)} ${r.worst_tile ?? ''} | ${pct(r.mdb_tile_max_Y, 2)} | ${r.tost_failed_tiles ?? ''} | ${r.cycles} | ${r.ours} | ${r4(r.multiplier_needed)} |`).join('\n') + '\n';
}

/** make-m3c.ts must write byte-identical packages (the Cycles cache keys on the bytes): generate into validation/out/m3c/
 *  scenes, then again into a temp dir, and compare. */
function packagesDeterministic(dir: string, add: Add, only?: Set<string>): void {
  const t0 = performance.now();
  const onlyArgs = only ? ['--only', [...only].join(',')] : [];
  const a = sh('npx', ['tsx', 'validation/scenes/make-m3c.ts', SCENES, ...onlyArgs], () => false);
  const tmp = path.join(dir, 'regen');
  const b = sh('npx', ['tsx', 'validation/scenes/make-m3c.ts', tmp, ...onlyArgs], () => false);
  const diffs: string[] = [];
  if (a.code !== 0) diffs.push(`make-m3c.ts: exit ${a.code} ${a.out.slice(-300)}`);
  if (b.code !== 0) diffs.push(`make-m3c.ts (regen): exit ${b.code}`);
  for (const p of existsSync(path.join(ROOT, tmp)) ? readdirSync(path.join(ROOT, tmp)) : []) {
    const x = path.join(ROOT, tmp, p), y = path.join(ROOT, SCENES, p);
    for (const n of new Set([...readdirSync(x), ...(existsSync(y) ? readdirSync(y) : [])])) {
      if (!existsSync(path.join(x, n)) || !existsSync(path.join(y, n)) || !readFileSync(path.join(x, n)).equals(readFileSync(path.join(y, n)))) diffs.push(`${p}/${n}`);
    }
  }
  rmSync(path.join(ROOT, tmp), { recursive: true, force: true });
  add('M3c scene packages generated + deterministic (make-m3c.ts twice, byte-identical)', diffs.length === 0, (performance.now() - t0) / 1000, { diffs, dir: SCENES },
    diffs.length ? `diffs: ${diffs.slice(0, 8).join(', ')}` : undefined);
}

function cyclesRef(s: G2E, out: string, seeds: string, add: Add): string | undefined {
  console.log(`\n--- Cycles reference ${s.pkg}: ${s.cyclesSpp} spp × seeds ${seeds}`);
  const r = sh(BLENDER, ['-b', '--factory-startup', '--python-exit-code', '1', '-P', 'validation/blender/render_reference.py', '--',
    '--package', `${SCENES}/${s.pkg}`, '--out', out, '--spp', String(s.cyclesSpp), '--seeds', seeds],
  (l) => l.startsWith('[render_reference]') && (l.includes('RESULT') || l.includes('cache') || l.includes('waited')) || /Error|Traceback/.test(l));
  const line = r.out.split('\n').reverse().find((l) => l.startsWith('[render_reference] RESULT '));
  const res = line ? JSON.parse(line.slice('[render_reference] RESULT '.length)) as { dir: string; cache_hit: boolean; renders: number; renders_total_s?: number } : undefined;
  add(`Cycles reference ${s.pkg} (${s.cyclesSpp} spp × ${seeds})`, r.code === 0 && !!res, r.seconds,
    res && { dir: path.relative(ROOT, res.dir), cache_hit: res.cache_hit, renders: res.renders, renders_total_s: res.renders_total_s ?? 0 },
    res ? (res.cache_hit ? 'cache hit' : `rendered ${res.renders} in ${res.renders_total_s} s`) : `exit ${r.code} ${r.out.slice(-400)}`);
  return res ? path.relative(ROOT, res.dir) : undefined;
}

function ourRun(s: G2E, run: string, seed: number, dir: string, add: Add, label = 'PT batches'): { dirs: string[]; ms: number } | undefined {
  const extra = s.ourArgs ?? [];
  console.log(`\n--- ${label} ${s.pkg}: ${s.ourSpp} spp × ${s.B} batches, seed ${seed}${extra.length ? ` ${extra.join(' ')}` : ''}`);
  const r = sh('npx', ['tsx', 'validation/harness/run-batches.ts', '--package', `${SCENES}/${s.pkg}`, '--kernel', 'pt', '--spp', String(s.ourSpp),
    '--batches', String(s.B), '--seed', String(seed), '--run', run, ...extra], (l) => /^(FAIL)\s|errors:|Error/.test(l));
  const d = fileRun(run, path.join(dir, 'pt'));
  const ok = r.code === 0 && existsSync(path.join(ROOT, d, 'meta.json'));
  const meta = ok ? readJson(path.join(d, 'meta.json')) : undefined;
  const ms = meta?.timings.totalMs ?? 0;
  add(`${label} ${s.pkg} (${s.ourSpp} spp × ${s.B})`, ok, r.seconds, { dir: d, gpu_total_ms: Math.round(ms), env: meta?.pt?.env },
    ok ? `${(ms / 1000).toFixed(1)} s in the page, P(env) ${meta?.pt?.env?.pEnv?.toFixed(4) ?? 'n/a'}` : `exit ${r.code} ${r.out.slice(-300)}`);
  return ok ? { dirs: [d], ms } : undefined;
}

function writeTest(dir: string, name: string, nUnits: number, extra: Record<string, unknown> = {}): string {
  const p = path.join(dir, 'tests', `test-${name}.json`);
  mkdirSync(path.join(ROOT, dir, 'tests'), { recursive: true });
  writeFileSync(path.join(ROOT, p), JSON.stringify({ name, stage: 'A', channels: ['Y', 'R', 'G', 'B'], n_units: nUnits, tier: 'tight', num_eps: NUM_EPS, num_eps_note: NUM_EPS_NOTE, ...extra }, null, 1));
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

function stageA(s: G2E, ours: string, ref: string, dir: string, runId: string, nUnits: number, add: Add, unitName?: string, rerunAllowed = true): Record<string, unknown> {
  const unit = unitName ?? s.pkg;
  const test = writeTest(dir, unit, nUnits, { tier: s.tier, ...(s.test ?? {}) });
  const out = path.join(dir, 'compare', unit);
  const t0 = performance.now();
  let c = compare(ours, ref, test, out);
  let rep = existsSync(path.join(ROOT, out, 'report.json')) ? readJson(path.join(out, 'report.json')) : undefined;
  let rerun: Record<string, unknown> | undefined;
  if (rep?.status === 'rerun_required' && rerunAllowed) {
    console.log(`  ${unit}: rerun_required (${rep.failed_checks.join(', ')}) -> confirmatory re-run on disjoint seeds`);
    const ref2 = cyclesRef(s, REFS, `${100}..${100 + s.K - 1}`, add);
    const o2 = ourRun(s, `${runId}-${unit}-rerun`, OUR_RERUN_SEED, dir, add, 'PT re-run batches');
    if (ref2 && o2) {
      const out2 = `${out}-rerun`;
      c = compare(o2.dirs[0], ref2, test, out2, path.join(out, 'report.json'));
      rerun = { first: summarize(rep), report: path.join(out2, 'report.json') };
      rep = existsSync(path.join(ROOT, out2, 'report.json')) ? readJson(path.join(out2, 'report.json')) : undefined;
    }
  }
  const sum = rep ? summarize(rep) : undefined;
  const ok = !!rep && (rep.status === 'pass' || rep.status === 'pass_on_rerun');
  const res = {
    unit, label: s.label, tier: s.tier, ...(sum ?? { status: `error (exit ${c.code})` }), cycles: `${s.cyclesSpp}×${s.K}`, ours: `${s.ourSpp}×${s.B}`,
    report: path.join(out, 'report.json'), ...(rerun ? { rerun } : {}), ...(s.note ? { note: s.note } : {}), ...(rep?.notes && Object.keys(rep.notes).length ? { notes: rep.notes } : {}),
  };
  add(`Stage A ${unit} (${s.label})`, ok, (performance.now() - t0) / 1000, res,
    sum ? `${sum.status}: Δ_Y ${pct(sum.global_rel_Y as number, 4)} (MDB ${pct(sum.mdb_global_Y as number, 3)}), worst tile ${pct(sum.worst_tile_rel_Y as number, 2)}, tile MDB max ${pct(sum.mdb_tile_max_Y as number, 2)}${sum.failed_checks?.length ? `, failed ${sum.failed_checks.join(' ')}` : ''}` : `compare exit ${c.code} ${c.out.slice(-300)}`);
  return res;
}

function analytic(s: G2E, sides: Record<string, string>, dir: string, add: Add): void {
  const exp = path.join(dir, 'expected', `${s.pkg}.pfm`);
  mkdirSync(path.join(ROOT, dir, 'expected'), { recursive: true });
  // 16×16 subsamples per pixel: the mirror sphere's facet edges alias at 4×4 (0.11% tiles); at 16×16 ≤ 1e-4.
  const g = sh('npx', ['tsx', 'validation/tools/env-expected.ts', '--package', `${SCENES}/${s.pkg}`, '--out', exp, '--ss', '16', '--json'], () => false);
  let info: Record<string, unknown> = {};
  try { info = JSON.parse(g.out.trim().split('\n').pop()!); } catch { info = { error: g.out.slice(-300) }; }
  if (g.code !== 0) { add(`analytic ${s.pkg}: expected image (env-expected.ts)`, false, g.seconds, info, `exit ${g.code}`); return; }
  for (const [side, d] of Object.entries(sides)) {
    const r = sh(PY, ['validation/tools/analytic_check.py', '--dir', d, '--package', `${SCENES}/${s.pkg}`, '--expected-image', exp, '--json'], () => false);
    let res: Record<string, any> = {};
    try { res = JSON.parse(r.out.trim().split('\n').pop()!); } catch { res = { ok: false, error: r.out.slice(-300) }; }
    const Y = res.channels?.Y;
    add(`analytic ${s.pkg} (${side}) vs f64 quadrature`, res.ok === true, r.seconds, { ...res, expected: info },
      Y ? `Δ_Y ${pct(Y.rel, 4)} (z ${Y.z.toFixed(2)}, SE ${pct(Y.se_rel, 4)}), tiles failed ${Y.tiles_failed}, worst tile ${pct(Y.worst_tile_rel, 2)}` : res.error);
  }
}

function plantDetection(s: G2E, p: { name: string; pkg: string; args: string[]; seed: number }, ref: string, dir: string, runId: string, nUnits: number, add: Add): void {
  const o = ourRun({ ...s, ourArgs: p.args }, `${runId}-plant-${p.seed}`, p.seed, dir, add, `PT planted (${p.name})`);
  if (!o) return;
  const test = writeTest(dir, `plant-${p.seed}`, nUnits, { tier: s.tier, calibration: { plants: [] } });
  const out = path.join(dir, 'calibrate', `plant-${p.seed}`);
  const r = sh(PY, ['validation/tools/compare.py', '--calibrate', '--ref', ref, '--planted', o.dirs[0], '--plant-name', `our PT ${p.name}`, '--test', test, '--out', out,
    '--splits', '20', '--repeats', '10'], () => false);
  let data: Record<string, unknown> | undefined;
  let detail = `exit ${r.code}`;
  let ok = false;
  if (existsSync(path.join(ROOT, out, 'report.json'))) {
    const rep = readJson(path.join(out, 'report.json'));
    const rp = rep.calibration.rendered_plant;
    const full = compare(o.dirs[0], ref, writeTest(dir, `plant-${p.seed}-full`, nUnits, { tier: s.tier }), `${out}-full`);
    const fullRep = existsSync(path.join(ROOT, `${out}-full`, 'report.json')) ? readJson(path.join(`${out}-full`, 'report.json')) : undefined;
    ok = !!rp && rp.gate_fail_count >= 9 && rp.control_pass_count >= 9 && !!fullRep && fullRep.status !== 'pass';
    data = {
      scene: p.pkg, report: path.join(out, 'report.json'), detected: `${rp.gate_fail_count}/${rp.n_repeats}`, equivalence_failed: `${rp.equivalence_fail_count}/${rp.n_repeats}`,
      control_pass: `${rp.control_pass_count}/${rp.n_repeats}`, half_size: rp.half_size, planted_rel_Y: r4(rp.channels.Y.planted_global_rel_median),
      mdb_global_Y: r4(rp.channels.Y.mdb_global_median), failed_checks_Y: Object.keys(rp.failed_checks_histogram).filter((k) => k.endsWith('[Y]')),
      full_compare: fullRep && { status: fullRep.status, global_rel_Y: r4(fullRep.channels.Y.global_.rel), failed_checks: fullRep.failed_checks }, full_exit: full.code,
    };
    detail = `${p.pkg}: detected ${rp.gate_fail_count}/10 (TOST ${rp.equivalence_fail_count}/10), control ${rp.control_pass_count}/10, planted Δ_Y ${pct(rp.channels.Y.planted_global_rel_median, 3)}; full: ${fullRep?.status}`;
  }
  add(`env plant detected: ${p.name}`, ok, r.seconds, data, detail);
}

function ourAA(s: G2E, dir: string, runId: string, nUnits: number, add: Add): void {
  const a = ourRun(s, `${runId}-aa-1001`, 1001, dir, add, 'PT A/A seed set 1001');
  const b = ourRun(s, `${runId}-aa-2002`, 2002, dir, add, 'PT A/A seed set 2002');
  if (!a || !b) return;
  const out = path.join(dir, 'compare', `aa-ours-${s.pkg}`);
  const c = compare(a.dirs[0], b.dirs[0], writeTest(dir, `aa-ours-${s.pkg}`, nUnits), out);
  const rep = existsSync(path.join(ROOT, out, 'report.json')) ? readJson(path.join(out, 'report.json')) : undefined;
  const sum = rep ? summarize(rep) : undefined;
  add(`A/A on two of our PT seed sets (${s.pkg}, 1001 vs 2002)`, rep?.status === 'pass', 0, sum && { ...sum, report: path.join(out, 'report.json') },
    sum ? `${sum.status}: Δ_Y ${pct(sum.global_rel_Y as number, 4)} (MDB ${pct(sum.mdb_global_Y as number, 4)}), tile MDB max ${pct(sum.mdb_tile_max_Y as number, 2)}` : `exit ${c.code}`);
}
