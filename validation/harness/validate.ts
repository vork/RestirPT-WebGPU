// Milestone gate (plan §5: "each milestone ends with `npm run validate -- --milestone Mx` green").
// M0: typecheck, cpu + node-dawn lanes, Chrome lane, Chrome smoke CLI, and the M0 exit artifacts.
// M1: typecheck, cpu lane (ENV-U1, scene/BVH/layout tests), node-dawn pre-check, Chrome lane for T12 (bvh),
//     ENV-U2/U7 (env), textures and the primary pass, the app-shell e2e and the M1 app smoke (Sponza + HDRI).
// M2: typecheck, cpu lane, python stats tests, the Blender bridge suite (run_m2_tests.py --quick), Gate-1 package
//     freshness, cached Cycles references (render only when cold), Gate 1-lite on real data (compare.py --calibrate
//     A/A + synthetic plants on C0b / (i) / spot, rendered Cycles plants (i) power ×1.0075 and spot blend 0.16 on
//     disjoint seeds), emission-kernel marker checks (C0a ×4, C0b, C0p 3 aspects × 12 frames) and our A/A (C0a, C0b).
//     Writes validation/out/m2-gate-<time>/summary.json.
//   npm run validate -- --milestone M0|M1|M2
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const { values: args } = parseArgs({ options: { milestone: { type: 'string', default: 'M0' } } });

interface Step { name: string; ok: boolean; detail?: string }
const steps: Step[] = [];
function record(name: string, ok: boolean, detail?: string): void {
  steps.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
}

function run(name: string, cmd: string, cmdArgs: string[]): void {
  console.log(`\n--- ${name}: ${cmd} ${cmdArgs.join(' ')}`);
  const t0 = performance.now();
  const r = spawnSync(cmd, cmdArgs, { cwd: ROOT, stdio: 'inherit' });
  record(name, r.status === 0, `exit ${r.status ?? r.signal}, ${((performance.now() - t0) / 1000).toFixed(1)} s`);
}

function file(rel: string, test: (text: string) => string | null = () => null): void {
  const p = path.join(ROOT, rel);
  if (!existsSync(p)) return record(rel, false, 'missing');
  let err: string | null;
  try { err = test(readFileSync(p, 'utf8')); } catch (e) { err = String(e); }
  record(rel, err === null, err ?? undefined);
}

function budgetPopulated(text: string): string | null {
  const b = JSON.parse(text) as { entries?: { scene: string; max_bounces: number; res: number; s_per_4096spp: number }[] };
  const want = [['cornell', 3], ['cornell', 7], ['sponza', 3]] as const;
  const missing = want.filter(([s, k]) => !b.entries?.some(
    (e) => e.scene === s && e.max_bounces === k && e.res === 512 && Number.isFinite(e.s_per_4096spp) && e.s_per_4096spp > 0));
  return missing.length ? `no s/4096spp @512² for ${missing.map(([s, k]) => `${s} b=${k}`).join(', ')}` : null;
}

function milestoneM0(): void {
  run('typecheck', 'npx', ['tsc', '--noEmit']);
  run('vitest cpu + node-dawn', 'npx', ['vitest', 'run', '--project', 'cpu', '--project', 'node-dawn']);
  run('vitest chrome', 'npx', ['vitest', 'run', '--project', 'chrome']);
  // Takes /tmp/restirpt-gpu.lock itself around the allocation probes.
  run('chrome smoke', 'npx', ['tsx', 'validation/harness/run-chrome.ts', '--smoke']);
  file('validation/budget.json', budgetPopulated);
  file('docs/decisions/usd.md', (t) => (/^## Decision:/m.test(t) ? null : 'no "## Decision:" heading'));
  file('docs/decisions/platform-lanes.md');
  file('docs/math.md');
}

// Shared GPU lock (plan §1.8: GPU-heavy jobs never overlap). Scripts that take the lock themselves run outside it.
const GPU_LOCK = '/tmp/restirpt-gpu.lock';
function withGpuLock(fn: () => void): void {
  const nap = new Int32Array(new SharedArrayBuffer(4));
  for (;;) { try { mkdirSync(GPU_LOCK); break; } catch { console.log('waiting for the GPU lock ...'); Atomics.wait(nap, 0, 0, 5000); } }
  const release = () => { try { rmdirSync(GPU_LOCK); } catch { /* gone */ } };
  process.once('exit', release);
  try { fn(); } finally { release(); process.removeListener('exit', release); }
}

const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');

/** M1 gate inputs. Their tests skip (ENV-U1 vs OIIO) or fall back (smoke: Cornell instead of Sponza) when these are
 *  missing, which would turn the gate green without testing what it claims, so the gate itself requires them. */
function m1Assets(): void {
  const need = ['validation/.venv/bin/python', 'validation/assets/cornell/cornell.glb', 'validation/assets/downloaded/sponza/Sponza.gltf'];
  const hdris = JSON.parse(readFileSync(path.join(ROOT, 'validation/assets/hdris.json'), 'utf8')) as { files: { file: string }[] };
  need.push(...hdris.files.map((f) => `validation/assets/downloaded/hdri/${f.file}`));
  const missing = need.filter((p) => !existsSync(path.join(ROOT, p)));
  record('M1 gate assets (OIIO venv, Sponza, HDRIs)', missing.length === 0,
    missing.length ? `missing: ${missing.join(', ')} (validation/blender/fetch_sponza.py, validation/assets/fetch_hdris.ts)` : undefined);
}

function milestoneM1(): void {
  const runId = `m1-${stamp()}`;
  m1Assets();
  run('typecheck', 'npx', ['tsc', '--noEmit']);
  run('vitest cpu (ENV-U1 RGBE/EXR vs OIIO, loader, BVH, layouts)', 'npx', ['vitest', 'run', '--project', 'cpu']);
  withGpuLock(() => {
    run('vitest node-dawn (pre-check)', 'npx', ['vitest', 'run', '--project', 'node-dawn']);
    run('T12 BVH brute force / watertight / offsets / overflow (chrome)', 'npx', ['vitest', 'run', '--project', 'chrome', 'validation/gpu-tests/bvh.gpu.test.ts']);
    run('ENV-U2 mapping + ENV-U7 bilinear/pole-wrap (chrome)', 'npx', ['vitest', 'run', '--project', 'chrome', 'validation/gpu-tests/env.gpu.test.ts']);
    run('textures validation/interactive paths (chrome)', 'npx', ['vitest', 'run', '--project', 'chrome', 'validation/gpu-tests/textures.gpu.test.ts']);
    run('primary pass V-buffer / MASK / env miss (chrome)', 'npx', ['vitest', 'run', '--project', 'chrome', 'validation/gpu-tests/primary.gpu.test.ts']);
  });
  // These take the GPU lock themselves.
  run('app shell e2e (test pattern)', 'npx', ['tsx', 'tests/app/e2e-app.ts', `validation/out/${runId}/app-e2e`]);
  run('M1 app smoke (Sponza + HDRI, fly camera, views, timings)', 'npx', ['tsx', 'validation/harness/m1-app-smoke.ts', '--run', runId]);
  const rep = path.join(ROOT, 'validation/out', runId, 'report.json');
  if (existsSync(rep)) {
    const r = JSON.parse(readFileSync(rep, 'utf8')) as { timing?: Record<string, { isolatedMedianMs?: number; hudPrimary?: { ms: number } }>; warnings?: string[] };
    for (const [k, v] of Object.entries(r.timing ?? {})) console.log(`timing  primary ${k}: ${v.isolatedMedianMs?.toFixed(3)} ms isolated, ${v.hudPrimary?.ms.toFixed(3)} ms in-app avg`);
    for (const w of r.warnings ?? []) console.log(`WARN    ${w}`);
    console.log(`screenshots: validation/out/${runId}/`);
  }
}

// ---- M2: validation harness + Cycles references, Gate 1-lite on real data (plan §5 M2 exit, §7.3) ----------------
// Nothing here holds the GPU lock: render_reference.py, run-batches.ts and run_m2_tests.py take it themselves.

const PY = path.join(ROOT, 'validation/.venv/bin/python');
const BLENDER = process.env.BLENDER ?? '/Applications/Blender.app/Contents/MacOS/Blender';
const BLENDER_PY = process.env.BLENDER_PY ?? '/Applications/Blender.app/Contents/Resources/5.1/python/bin/python3.13';
const REFS = 'validation/out/m2/refs'; // shared Cycles cache (render_reference.py keys on package + args + scripts + Blender)
const PLANT_SEEDS = '500..515'; // disjoint from the reference seeds 0..15 (compare.py --planted refuses overlaps)
/** Cycles references: [package, spp, seeds]. (i) 1024 spp ≈ 3.5 s/seed on Metal; the spot scene is b = 0. */
const M2_REFS: [string, number, string][] = [
  ['c0b_512', 16, '0..15'],
  ['cornell_i_512', 1024, '0..15'],
  ['cornell_i_power1.0075_512', 1024, PLANT_SEEDS],
  ['spot_c0d_512', 1024, '0..15'],
  ['spot_c0d_blend0.16_512', 1024, PLANT_SEEDS],
];
/** Emission-kernel marker runs: [package, spp per batch, batches, frames]. */
const M2_EMISSION: [string, number, number, number[] | undefined][] = [
  ['c0a_512', 256, 8, undefined], ['c0a_640x360', 256, 8, undefined], ['c0a_360x640', 256, 8, undefined], ['c0a_far_512', 256, 8, undefined],
  ['c0b_512', 64, 4, undefined],
  ...(['c0p_512', 'c0p_640x360', 'c0p_360x640'] as const).map((p) => [p, 256, 8, [...Array(12).keys()]] as [string, number, number, number[]]),
];
/** Our A/A (two seed sets of the emission kernel): [package, spp per batch]; 16 batches per side. */
const M2_OUR_AA: [string, number][] = [['c0a_512', 256], ['c0b_512', 16]];
// suite FWER units: {C0b, (i), spot} calibrations + (i)/spot rendered plants + our C0a/C0b A/A = 7 units × {Y,R,G,B}
const M2_N_UNITS = 28;

interface M2Step { name: string; ok: boolean; seconds: number; data?: unknown; detail?: string }
const m2Steps: M2Step[] = [];

function sh(cmd: string, argv: string[], opts: { echo?: (line: string) => boolean; timeoutMs?: number } = {}): { code: number; out: string; seconds: number } {
  const t0 = performance.now();
  const r = spawnSync(cmd, argv, { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 28, timeout: opts.timeoutMs ?? 3_600_000 });
  const out = `${r.stdout ?? ''}${r.stderr ?? ''}`;
  const echo = opts.echo ?? (() => true);
  for (const l of out.split('\n')) if (l && echo(l)) console.log(`  ${l}`);
  return { code: r.status ?? 1, out, seconds: (performance.now() - t0) / 1000 };
}

function m2Record(name: string, ok: boolean, seconds: number, data?: unknown, detail?: string): void {
  m2Steps.push({ name, ok, seconds: Math.round(seconds * 10) / 10, data, detail });
  record(name, ok, `${detail ? `${detail}, ` : ''}${seconds.toFixed(1)} s`);
}

function m2Run(name: string, cmd: string, argv: string[], echo?: (l: string) => boolean): { code: number; out: string } {
  console.log(`\n--- ${name}: ${cmd} ${argv.join(' ')}`);
  const r = sh(cmd, argv, { echo });
  m2Record(name, r.code === 0, r.seconds, undefined, `exit ${r.code}`);
  return r;
}

const readJson = <T = Record<string, any>>(p: string): T => JSON.parse(readFileSync(path.join(ROOT, p), 'utf8')) as T;

/** The generator-made packages must be current (byte-identical regeneration), else the refs render a stale scene. */
function m2PackagesCurrent(dir: string): void {
  const t0 = performance.now();
  const diffs: string[] = [];
  for (const [gen, pkgs] of [['make-cornell-i.ts', ['cornell_i_512', 'cornell_i_power1.0075_512']], ['make-spot-c0d.ts', ['spot_c0d_512', 'spot_c0d_blend0.16_512']]] as const) {
    const tmp = path.join(ROOT, dir, 'regen');
    const r = sh('npx', ['tsx', `validation/scenes/${gen}`, tmp]);
    if (r.code !== 0) { diffs.push(`${gen}: exit ${r.code}`); continue; }
    for (const p of pkgs) {
      const a = path.join(tmp, p), b = path.join(ROOT, 'validation/scenes', p);
      const names = new Set([...(existsSync(a) ? readdirSync(a) : []), ...(existsSync(b) ? readdirSync(b) : [])]);
      for (const n of names) {
        if (!existsSync(path.join(a, n)) || !existsSync(path.join(b, n)) || !readFileSync(path.join(a, n)).equals(readFileSync(path.join(b, n)))) diffs.push(`${p}/${n}`);
      }
    }
  }
  rmSync(path.join(ROOT, dir, 'regen'), { recursive: true, force: true });
  m2Record('Gate-1 scene packages current (make-cornell-i.ts, make-spot-c0d.ts)', diffs.length === 0, (performance.now() - t0) / 1000, { diffs }, diffs.length ? `stale: ${diffs.join(', ')}` : undefined);
}

/** render_reference.py for every Gate-1 reference; renders only when the cache is cold. Returns package -> ref dir. */
function m2References(): Record<string, string> {
  const dirs: Record<string, string> = {};
  for (const [pkg, spp, seeds] of M2_REFS) {
    console.log(`\n--- Cycles reference ${pkg}: ${spp} spp × seeds ${seeds}`);
    const r = sh(BLENDER, ['-b', '--factory-startup', '--python-exit-code', '1', '-P', 'validation/blender/render_reference.py', '--',
      '--package', `validation/scenes/${pkg}`, '--out', REFS, '--spp', String(spp), '--seeds', seeds],
    { echo: (l) => l.startsWith('[render_reference]') && (l.includes('RESULT') || l.includes('cache') || l.includes('waited')) });
    const line = r.out.split('\n').reverse().find((l) => l.startsWith('[render_reference] RESULT '));
    const res = line ? JSON.parse(line.slice('[render_reference] RESULT '.length)) as { dir: string; cache_hit: boolean; renders: number; renders_total_s?: number } : undefined;
    if (res) dirs[pkg] = path.relative(ROOT, res.dir);
    m2Record(`Cycles reference ${pkg} (${spp} spp × ${seeds})`, r.code === 0 && !!res, r.seconds,
      res && { dir: dirs[pkg], cache_hit: res.cache_hit, renders: res.renders, renders_total_s: res.renders_total_s ?? 0 },
      res ? (res.cache_hit ? 'cache hit' : `rendered ${res.renders} in ${res.renders_total_s} s`) : `exit ${r.code}${r.out.slice(-400)}`);
  }
  return dirs;
}

function writeTest(dir: string, name: string, extra: Record<string, unknown> = {}): string {
  const p = path.join(dir, `test-${name}.json`);
  writeFileSync(path.join(ROOT, p), JSON.stringify({ name, stage: 'A', channels: ['Y', 'R', 'G', 'B'], n_units: M2_N_UNITS, tier: 'tight', ...extra }, null, 1));
  return p;
}

const r4 = (x: unknown) => (typeof x === 'number' ? Number(x.toPrecision(4)) : x);

/** compare.py --calibrate (A/A ≥ 20 re-splits + synthetic δ-scale plants) [+ --planted rendered plant]. */
function m2Calibrate(dir: string, label: string, ref: string, planted?: string): void {
  const out = path.join(dir, `calibrate-${label}`);
  const test = writeTest(dir, `gate1-${label}`);
  console.log(`\n--- Gate 1-lite calibration ${label}${planted ? ` + rendered plant ${path.basename(planted)}` : ''}`);
  const r = sh(PY, ['validation/tools/compare.py', '--calibrate', '--ref', ref, '--test', test, '--out', out, '--splits', '20', '--repeats', '10',
    ...(planted ? ['--planted', planted] : [])]);
  let data: Record<string, unknown> | undefined;
  let detail = `exit ${r.code}`;
  if (existsSync(path.join(ROOT, out, 'report.json'))) {
    const rep = readJson(path.join(out, 'report.json'));
    const aa = rep.calibration.aa;
    const plants = (rep.calibration.plants as Record<string, any>[]).map((p) => ({
      name: p.name, detected: `${p.gate_fail_count}/${p.n_repeats}`, control_pass: `${p.control_pass_count}/${p.n_repeats}`, mdb_global_median: r4(p.mdb_global_median), calibrated: p.calibrated,
    }));
    const rp = rep.calibration.rendered_plant;
    data = {
      report: path.join(out, 'report.json'), status: rep.status, failed_checks: rep.failed_checks,
      aa: {
        ok: aa.ok, splits: aa.n_splits, half_size: aa.half_size, alpha_u: r4(aa.alpha_u),
        fpr_tile: Object.fromEntries(Object.entries(aa.per_tile as Record<string, { rate: number; n: number; p_excess: number }>).map(([k, v]) => [k, { rate: r4(v.rate), n: v.n, p_excess: r4(v.p_excess) }])),
        rejection_rates: Object.fromEntries(Object.entries(aa.rejection_checks as Record<string, { rate: number; n: number }>).map(([k, v]) => [k, `${r4(v.rate)} (n ${v.n})`])),
        tost_pass_rate: aa.tost_pass_rate, gate_pass_rate: aa.gate_pass_rate,
      },
      synthetic_plants: plants,
      rendered_plant: rp && {
        name: rp.name, detected: `${rp.gate_fail_count}/${rp.n_repeats}`, equivalence_failed: `${rp.equivalence_fail_count}/${rp.n_repeats}`,
        control_pass: `${rp.control_pass_count}/${rp.n_repeats}`, half_size: rp.half_size, calibrated: rp.calibrated,
        planted_rel_Y: r4(rp.channels.Y.planted_global_rel_median), mdb_global_Y: r4(rp.channels.Y.mdb_global_median),
        mdb_tile_max_Y: r4(rp.channels.Y.mdb_tile_max_median), mdb_tile_median_Y: r4(rp.channels.Y.mdb_tile_median_median),
        failed_checks: Object.keys(rp.failed_checks_histogram).filter((k) => k.endsWith('[Y]')),
      },
    };
    const fpr = aa.per_tile as Record<string, { rate: number; n: number }>;
    detail = `A/A ${aa.ok ? 'ok' : 'FAIL'}: tile FPR ${fpr['0.05'].n ? `${(fpr['0.05'].rate * 100).toFixed(2)}% @5%, ${(fpr['0.01'].rate * 100).toFixed(2)}% @1%` : 'n/a (every tile zero-variance: exact 1e-6 rule)'}, gate pass ${aa.gate_pass_rate * 20}/20`
      + (rp ? `; ${rp.name}: detected ${rp.gate_fail_count}/10 (TOST ${rp.equivalence_fail_count}/10), control ${rp.control_pass_count}/10, Δ ${(rp.channels.Y.planted_global_rel_median * 100).toFixed(3)}%, MDB ${(rp.channels.Y.mdb_global_median * 100).toExponential(2)}%` : '');
  }
  m2Record(`Gate 1-lite calibrate ${label}${planted ? ' + rendered plant' : ''}`, r.code === 0, r.seconds, data, detail);
}

/** run-batches.ts writes validation/out/<run>/ (one path component); file the finished runs under the gate dir. */
function fileRun(run: string, dest: string): string {
  const to = path.join(dest, run);
  mkdirSync(path.join(ROOT, dest), { recursive: true });
  if (existsSync(path.join(ROOT, 'validation/out', run))) renameSync(path.join(ROOT, 'validation/out', run), path.join(ROOT, to));
  return to;
}

/** Emission kernel runs (run-batches.ts, GPU lock inside) + marker_check.py --batch-dir per frame. */
function m2Emission(dir: string, runPrefix: string): void {
  for (const [pkg, spp, batches, frames] of M2_EMISSION) {
    console.log(`\n--- emission kernel ${pkg}: ${spp} spp × ${batches} batches${frames ? `, frames 0..${frames.length - 1}` : ''}`);
    const run = `${runPrefix}-${pkg}`;
    const r = sh('npx', ['tsx', 'validation/harness/run-batches.ts', '--package', `validation/scenes/${pkg}`, '--spp', String(spp), '--batches', String(batches),
      '--seed', '7', '--run', run, ...(frames ? ['--frames', frames.join(',')] : [])],
    { echo: (l) => /^(OK|FAIL)\s|lock wait|RESULT|error|Error/.test(l) });
    const per: Record<string, unknown>[] = [];
    let ok = r.code === 0;
    for (const f of frames ?? [undefined]) {
      const out = path.join('validation/out', f === undefined ? run : `${run}-f${f}`);
      const c = sh(PY, ['validation/tools/marker_check.py', '--batch-dir', out, '--package', `validation/scenes/${pkg}`, '--frame', String(f ?? 0), '--json'], { echo: () => false });
      let res: Record<string, any> = {};
      try { res = JSON.parse(c.out.trim().split('\n').pop()!); } catch { res = { ok: false, failures: [c.out.slice(-300)] }; }
      ok &&= c.code === 0 && res.ok === true;
      const markers = (res.markers ?? []) as { name: string; err_px: number; mass_ratio: number; mass_se?: number; centroid_se_px?: number }[];
      per.push({
        frame: f ?? 0, ok: res.ok, max_err_px: r4(res.max_err_px), max_const_err: r4(res.max_abs_err), probe_max_err: r4(res.probe_max_err),
        stray_mass_ratio: r4(res.stray_mass_ratio), failures: res.failures,
        worst_mass: markers.length ? markers.map((m) => ({ name: m.name, ratio: r4(m.mass_ratio), se: r4(m.mass_se) }))
          .sort((a, b) => Math.abs((b.ratio as number) - 1) - Math.abs((a.ratio as number) - 1))[0] : undefined,
      });
      if (!res.ok) console.log(`  FAIL ${pkg} frame ${f ?? 0}: ${(res.failures ?? []).join('; ')}`);
      per[per.length - 1].dir = fileRun(path.basename(out), path.join(dir, 'emission'));
    }
    const errs = per.map((p) => p.max_err_px).filter((x): x is number => typeof x === 'number');
    const cerr = per.map((p) => p.max_const_err).filter((x): x is number => typeof x === 'number');
    const detail = errs.length ? `max centroid err ${Math.max(...errs).toFixed(4)} px over ${per.length} frame(s)` : `max |px - L_e| ${Math.max(...cerr).toExponential(2)}`;
    m2Record(`emission kernel ${pkg} markers (${spp * batches} spp)`, ok, r.seconds, { run, spp_per_batch: spp, batches, frames: per }, detail);
  }
}

/** A/A between two of OUR seed sets (emission kernel, 16 batches each) through compare.py compare mode. */
function m2OurAA(dir: string, runPrefix: string): void {
  for (const [pkg, spp] of M2_OUR_AA) {
    console.log(`\n--- our A/A ${pkg}: two seed sets × 16 batches × ${spp} spp`);
    const t0 = performance.now();
    const runs = [1001, 2002].map((seed) => {
      const run = `${runPrefix}-aa-${pkg}-s${seed}`;
      const r = sh('npx', ['tsx', 'validation/harness/run-batches.ts', '--package', `validation/scenes/${pkg}`, '--spp', String(spp), '--batches', '16', '--seed', String(seed), '--run', run],
        { echo: (l) => /^(OK|FAIL)\s|RESULT/.test(l) });
      return { run, code: r.code };
    });
    const out = path.join(dir, `aa-ours-${pkg}`);
    const c = sh(PY, ['validation/tools/compare.py', '--ours', `validation/out/${runs[0].run}`, '--ref', `validation/out/${runs[1].run}`,
      '--test', writeTest(dir, `aa-ours-${pkg}`), '--out', out], { echo: (l) => !l.includes('RuntimeWarning') && !l.includes('nanmean') });
    let data: Record<string, unknown> | undefined;
    let detail = `compare exit ${c.code}`;
    if (existsSync(path.join(ROOT, out, 'report.json'))) {
      const rep = readJson(path.join(out, 'report.json'));
      const Y = rep.channels.Y;
      data = {
        report: path.join(out, 'report.json'), status: rep.status, failed_checks: rep.failed_checks,
        Y: { rel: r4(Y.global_.rel), mdb_global: r4(Y.global_.mdb), tost_failed_tiles: Y.tiles.tost_failed, zero_var_tiles: Y.tiles.zero_var, mdb_tile_max: r4(Y.tiles.mdb_max), sidak_rejected: Y.tiles.sidak_rejected },
      };
      detail = `${rep.status}: Δ_Y ${(Y.global_.rel * 100).toFixed(4)}%, MDB_Y ${(Y.global_.mdb * 100).toFixed(4)}%, tile MDB max ${(Y.tiles.mdb_max * 100).toFixed(3)}%, ${Y.tiles.zero_var}/256 zero-variance tiles`;
    }
    if (data) data.runs = runs.map((x) => fileRun(x.run, path.join(dir, 'aa-runs')));
    m2Record(`our A/A ${pkg} (emission kernel, seeds 1001 vs 2002)`, runs.every((x) => x.code === 0) && c.code === 0, (performance.now() - t0) / 1000, data, detail);
  }
}

function milestoneM2(): void {
  const t0 = performance.now();
  const runId = `m2-gate-${stamp()}`;
  const dir = path.join('validation/out', runId);
  mkdirSync(path.join(ROOT, dir), { recursive: true });
  const need = [PY, BLENDER, BLENDER_PY];
  const missing = need.filter((p) => !existsSync(p));
  m2Record('M2 gate tools (venv python, Blender, Blender python)', missing.length === 0, 0, { missing }, missing.length ? `missing: ${missing.join(', ')}` : undefined);

  m2Run('typecheck', 'npx', ['tsc', '--noEmit']);
  m2Run('vitest cpu', 'npx', ['vitest', 'run', '--project', 'cpu'], (l) => /Test Files|Tests |FAIL|✗|×/.test(l));
  m2Run('python stats tests (validation/tools/tests)', PY, ['-m', 'pytest', 'validation/tools/tests', '-q'], (l) => /passed|failed|error/i.test(l));
  m2Run('Blender bridge suite (run_m2_tests.py --quick)', BLENDER_PY, ['validation/blender/tests/run_m2_tests.py', '--quick'], (l) => l.startsWith('[m2-tests]'));
  if (existsSync(path.join(ROOT, 'validation/out/m2/summary.json'))) m2Steps[m2Steps.length - 1].data = { summary: 'validation/out/m2/summary.json', ok: readJson('validation/out/m2/summary.json').ok };
  m2PackagesCurrent(dir);

  const refs = m2References();
  const have = (...p: string[]) => p.every((k) => refs[k]);
  if (have('c0b_512')) m2Calibrate(dir, 'c0b', refs.c0b_512);
  if (have('cornell_i_512', 'cornell_i_power1.0075_512')) m2Calibrate(dir, 'cornell_i', refs.cornell_i_512, refs['cornell_i_power1.0075_512']);
  if (have('spot_c0d_512', 'spot_c0d_blend0.16_512')) m2Calibrate(dir, 'spot_c0d', refs.spot_c0d_512, refs['spot_c0d_blend0.16_512']);

  m2Emission(dir, runId);
  m2OurAA(dir, runId);

  const failed = m2Steps.filter((s) => !s.ok).map((s) => s.name);
  const summary = { milestone: 'M2', gate: 'Gate 1-lite on real data (plan §5 M2 exit)', run: runId, created: new Date().toISOString(), ok: failed.length === 0 && steps.every((s) => s.ok),
    total_s: Math.round((performance.now() - t0) / 100) / 10, failed, n_units: M2_N_UNITS, steps: m2Steps };
  writeFileSync(path.join(ROOT, dir, 'summary.json'), `${JSON.stringify(summary, null, 1)}\n`);
  console.log(`\nsummary: ${dir}/summary.json (${summary.total_s} s)`);
}

const gates: Record<string, () => void> = { M0: milestoneM0, M1: milestoneM1, M2: milestoneM2 };
const gate = gates[args.milestone!.toUpperCase()];
if (!gate) {
  console.error(`unknown milestone ${args.milestone}; known: ${Object.keys(gates).join(', ')}`);
  process.exit(2);
}
gate();
const failed = steps.filter((s) => !s.ok);
console.log(`\n=== validate ${args.milestone}: ${steps.length - failed.length}/${steps.length} passed ===`);
for (const s of failed) console.log(`FAIL  ${s.name}${s.detail ? `  (${s.detail})` : ''}`);
process.exit(failed.length ? 1 : 0);
