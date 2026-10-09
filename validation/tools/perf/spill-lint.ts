// perf2 report-only "spill bytes per pipeline + MSL lint" (perf2-plan.md §0 rule 4 / rule 5, V-BIT item 5; WP-0 step 4;
// docs/decisions/perf2-api.md §4). Per scene:
//   1. MSL: prof-driver.ts --mode dump (Chrome + Dawn dump_shaders) on a short interactive job → split-dump.py →
//      msl_lint.py: per kernel MSL bytes, robustness clamps, tint_loop_idx loops, tint_volatile_zero guards, zero-inits,
//      Tint integer div/mod helpers, bvh_trace copies after inlining.
//   2. Spill: prof-driver.ts --mode xct (Metal System Trace attached to Chrome's GPU process during the one-submit-per-
//      pass phase) → xct-export.sh (encoders + graphics-compiler-spill-events) → xct_spill.py: spilled bytes per thread
//      per pipeline. The job animates a light so the refresh kernels run too.
//   3. <out>/<tag>/report.{json,md}; --compare BASE/report.json adds deltas and marks growth in hot kernels with '!'.
// Takes the GPU lock around each Chrome run itself (do NOT wrap it in with-gpu-lock.ts). Not gating: exit 0 unless a
// step failed.
//   npx tsx validation/tools/perf/spill-lint.ts [--scene sponza,cornell] [--res 480x270] [--kernel-flags A,B] [--tag T]
//        [--out validation/out/perf2-spill] [--no-dump] [--no-trace] [--compare validation/out/perf2-spill/<T0>/report.json]
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { acquireGpuLock } from '../../harness/gpu-lock.ts';
import { PERF_SCENES, pinnedJob } from '../../harness/run-perf.ts';
import { normalizePerfFlags, perfFlagsKey } from '../../../src/core/render/restir/perf-flags.ts';

const ROOT = fileURLToPath(new URL('../../../', import.meta.url));
const HERE = path.dirname(fileURLToPath(import.meta.url));
/** Kernels whose spill / lint must not grow (perf2-plan.md V-BIT item 5): the traced ReSTIR kernels and the primaries. */
export const HOT_KERNELS = ['rs_initial', 'rs_spatial_shift', 'rs_spatial_replay', 'rs_t_classify', 'rs_t_forward', 'rs_t_inverse',
  'rs_refresh_fwd', 'rs_refresh_inv', 'rs_t_select', 'rs_primary', 'primary', 'rs_spatial_resample', 'rs_pair_accept', 'rs_dupmap'];
const LINT_COLS = ['bytes', 'clamps', 'loop_idx', 'volatile', 'zero_init', 'divmod', 'bvh_copies'] as const;
type Lint = Record<(typeof LINT_COLS)[number], number>;
interface SceneReport { scene: string; perfFlags: string; spill: Record<string, number[]>; lint: Record<string, Lint>; errors: string[] }

const { values: a } = parseArgs({ options: {
  scene: { type: 'string', default: 'sponza' }, res: { type: 'string', default: '480x270' }, 'kernel-flags': { type: 'string' },
  tag: { type: 'string' }, out: { type: 'string', default: 'validation/out/perf2-spill' }, 'no-dump': { type: 'boolean', default: false },
  'no-trace': { type: 'boolean', default: false }, compare: { type: 'string' }, 'xct-secs': { type: 'string', default: '4' },
} });
const flags = normalizePerfFlags(a['kernel-flags']);
const tag = a.tag ?? `spill-${new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-')}${perfFlagsKey(flags) ? `-${perfFlagsKey(flags).replace(/[^\w]+/g, '_')}` : ''}`;
const dir = path.resolve(ROOT, a.out!, tag);
mkdirSync(dir, { recursive: true });
const [W, H] = a.res!.split('x').map(Number);
const rel = (p: string) => path.relative(ROOT, p);

/** A CPU-side step (no GPU lock). */
function run(name: string, cmd: string, argv: string[]): boolean {
  console.log(`\n--- ${name}: ${cmd} ${argv.join(' ')}`);
  const r = spawnSync(cmd, argv, { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 28, stdio: ['ignore', 'pipe', 'pipe'] });
  writeFileSync(path.join(dir, `${name}.log`), `${r.stdout ?? ''}${r.stderr ?? ''}`);
  for (const l of `${r.stdout ?? ''}`.split('\n').filter((x) => /^\[drv\]|rows=|^kernel|spill\/thread/.test(x)).slice(0, 60)) console.log(`  ${l}`);
  if (r.status !== 0) console.log(`  exit ${r.status}: ${(r.stderr ?? '').slice(-800)}`);
  return r.status === 0;
}

async function locked(name: string, cmd: string, argv: string[]): Promise<boolean> {
  const release = await acquireGpuLock(`spill-lint-${name}`.slice(0, 60));
  try {
    console.log(`\n--- ${name} (GPU lock, waited ${release.waitedMs.toFixed(0)} ms): ${cmd} ${argv.join(' ')}`);
    const r = spawnSync(cmd, argv, { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 28, stdio: ['ignore', 'pipe', 'pipe'] });
    writeFileSync(path.join(dir, `${name}.log`), `${r.stdout ?? ''}${r.stderr ?? ''}`);
    for (const l of `${r.stdout ?? ''}`.split('\n').filter((x) => /^\[drv\]/.test(x)).slice(0, 30)) console.log(`  ${l}`);
    if (r.status !== 0) console.log(`  exit ${r.status}: ${(r.stderr ?? '').slice(-800)}`);
    return r.status === 0;
  } finally { release(); }
}

async function scene(sc: string): Promise<SceneReport> {
  if (!PERF_SCENES[sc]) throw new Error(`unknown scene ${sc} (${Object.keys(PERF_SCENES).join(', ')})`);
  const rep: SceneReport = { scene: sc, perfFlags: perfFlagsKey(flags), spill: {}, lint: {}, errors: [] };
  const base = { ...pinnedJob(sc), width: W, height: H, lightAnim: PERF_SCENES[sc].lightAnim, ...(perfFlagsKey(flags) ? { perfFlags: flags } : {}) };
  if (!a['no-dump']) {
    const jobs = path.join(dir, `${sc}-dump-jobs.json`);
    writeFileSync(jobs, JSON.stringify([{ ...base, frames: 32, warmup: 2, block: 8, latencyFrames: 0, passFrames: 0, label: `${sc} dump` }]));
    const dump = path.join(dir, `${sc}-dump.jsonl`);
    rmSync(dump, { force: true });
    const msl = path.join(dir, `${sc}-msl`);
    if (await locked(`${sc}-dump`, 'npx', ['tsx', path.join(HERE, 'prof-driver.ts'), '--mode', 'dump', '--jobs', rel(jobs), '--dump-file', dump, '--out', rel(dir)])
      && run(`${sc}-split-dump`, 'python3', [path.join(HERE, 'split-dump.py'), dump, msl])
      && run(`${sc}-lint`, 'python3', [path.join(HERE, 'msl_lint.py'), msl, '--json', path.join(dir, `${sc}-lint.json`)])) {
      rep.lint = JSON.parse(readFileSync(path.join(dir, `${sc}-lint.json`), 'utf8'));
      rmSync(dump, { force: true });   // ~10 MB of console JSON; the split MSL / WGSL stay
    } else rep.errors.push('MSL dump / lint failed (see logs)');
  }
  if (!a['no-trace']) {
    const jobs = path.join(dir, `${sc}-xct-jobs.json`);
    writeFileSync(jobs, JSON.stringify([{ ...base, frames: 32, warmup: 8, latencyFrames: 0, passFrames: 400, label: `${sc} split xct` }]));
    const trace = path.join(dir, 'xct', `${sc}-split.trace`);
    const pre = path.join(dir, `${sc}-xct-`);
    if (await locked(`${sc}-xct`, 'npx', ['tsx', path.join(HERE, 'prof-driver.ts'), '--mode', 'xct', '--jobs', rel(jobs), '--tag', `${sc}-split`, '--phase', 'split', '--after', '40',
      '--xct-secs', a['xct-secs']!, '--out', rel(dir)])
      && existsSync(trace)
      && run(`${sc}-xct-export`, path.join(HERE, 'xct-export.sh'), [trace, pre, 'metal-application-encoders-list', 'graphics-compiler-spill-events'])
      && run(`${sc}-spill`, 'python3', [path.join(HERE, 'xct_spill.py'), pre, '--json', path.join(dir, `${sc}-spill.json`)])) {
      rep.spill = JSON.parse(readFileSync(path.join(dir, `${sc}-spill.json`), 'utf8'));
      rmSync(trace, { recursive: true, force: true });   // ~100 MB+; the exported tables stay
    } else rep.errors.push('Metal System Trace / spill export failed (see logs)');
  }
  return rep;
}

function table(r: SceneReport, base?: SceneReport): string[] {
  const names = [...new Set([...Object.keys(r.lint), ...Object.keys(r.spill)])].filter((n) => !/^Blit|^\?/.test(n))
    .sort((x, y) => (Math.max(0, ...(r.spill[y] ?? [])) - Math.max(0, ...(r.spill[x] ?? []))) || ((r.lint[y]?.bytes ?? 0) - (r.lint[x]?.bytes ?? 0)));
  const fmt = (v: number | undefined, b: number | undefined, hot: boolean) => {
    if (v === undefined) return '–';
    if (b === undefined || b === v) return String(v);
    return `${v} (${v > b ? '+' : ''}${v - b}${hot && v > b ? ' !' : ''})`;
  };
  const sp = (x?: number[]) => (x === undefined ? undefined : Math.max(0, ...x));
  const out = [`| kernel | hot | spill B/thread | ${LINT_COLS.join(' | ')} |`, `|---|---|---|${LINT_COLS.map(() => '---').join('|')}|`];
  for (const n of names) {
    const hot = HOT_KERNELS.includes(n);
    out.push(`| ${n} | ${hot ? '●' : ''} | ${fmt(sp(r.spill[n]), sp(base?.spill[n]), hot)} | ${LINT_COLS.map((c) => fmt(r.lint[n]?.[c], base?.lint[n]?.[c], hot && c !== 'bytes')).join(' | ')} |`);
  }
  return out;
}

const reports: SceneReport[] = [];
for (const sc of a.scene!.split(',')) reports.push(await scene(sc));
const base = a.compare ? (JSON.parse(readFileSync(path.resolve(ROOT, a.compare), 'utf8')) as { reports: SceneReport[] }).reports : undefined;
writeFileSync(path.join(dir, 'report.json'), `${JSON.stringify({ tag, created: new Date().toISOString(), res: [W, H], perfFlags: perfFlagsKey(flags), compare: a.compare ?? null, reports }, null, 1)}\n`);
const md = [`# Spill + MSL lint ${tag}`, '', `Resolution ${W}×${H}, perf flags: ${perfFlagsKey(flags) || 'none'}${a.compare ? `, deltas vs ${a.compare} ('!' = growth in a hot kernel)` : ''}.`,
  'Spill: graphics-compiler-spill-events of a Metal System Trace (bytes per thread). Lint: msl_lint.py on the Dawn dump_shaders MSL.', ''];
for (const r of reports) md.push(`## ${r.scene}`, '', ...table(r, base?.find((x) => x.scene === r.scene)), '', ...(r.errors.length ? [`Errors: ${r.errors.join('; ')}`, ''] : []));
writeFileSync(path.join(dir, 'report.md'), md.join('\n'));
console.log(`\n${md.join('\n')}\nwrote ${rel(path.join(dir, 'report.md'))}`);
process.exit(reports.some((r) => r.errors.length) ? 1 : 0);
