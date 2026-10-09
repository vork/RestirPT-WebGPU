// perf2 profiling driver (promoted from validation/out/perf2-profile/tools/prof-driver.ts + counters/prof-driver-ctr.ts;
// docs/decisions/perf2-api.md §4). Headless Chrome on the harness page, window.__harness.renderPerf per job.
// Does NOT take the GPU lock: run it under validation/harness/with-gpu-lock.ts (spill-lint.ts takes the lock around it).
//   modes:
//     dump  Chrome with Dawn dump_shaders (+ disable_symbol_renaming); every console message (all levels) is appended to
//           --dump-file as JSON lines {t, x}; runs the jobs (keep them short). Split with split-dump.py.
//     xct   runs ONE renderPerf job; a page-side hook counts createCommandEncoder() calls by label ('perf-split' = the
//           one-submit-per-pass phase, 'perf-<i>' = frames). When the chosen --phase has created --after encoders,
//           `xcrun xctrace record` is attached to Chrome's GPU process (--target attach) or records all processes
//           (--target all; GPU counters then start only when some process creates a Metal device: --trigger CMD runs
//           CMD --trigger-delay ms after recording started) for --xct-secs; then the browser is closed. The trace goes to
//           <out>/xct/<tag>.trace (export with xct-export.sh, read with xct_split.py / xct_spill.py).
//     perf  plain jobs with custom Chrome args / Dawn toggles (sensitivity runs); reports printed and written to
//           <out>/<tag>.json (abba.py reads it).
//   npx tsx validation/tools/perf/prof-driver.ts --mode xct --jobs FILE --tag T [--phase split|frame] [--after 40]
//        [--template 'Metal System Trace'] [--instrument I ...] [--xct-secs 4] [--chrome-args "a b"] [--dawn-features a,b]
//        [--disable-dawn-features a,b] [--kernel-flags A,B] [--out DIR] [--target attach|all] [--trigger CMD] [--pre-delay s]
// Jobs: a JSON array of PerfOptions (validation/harness/perf-run.ts; perfFlags per job, or --kernel-flags for all).
import { execSync, spawn } from 'node:child_process';
import { appendFileSync, existsSync, lstatSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer as createNetServer } from 'node:net';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium } from 'playwright';
import { createServer } from 'vite';
import { normalizePerfFlags } from '../../../src/core/render/restir/perf-flags.ts';

const ROOT = fileURLToPath(new URL('../../../', import.meta.url));
const { values: a } = parseArgs({ options: {
  mode: { type: 'string', default: 'perf' }, jobs: { type: 'string' }, out: { type: 'string', default: 'validation/out/perf2-profile' },
  tag: { type: 'string', default: 'drv' }, template: { type: 'string', default: 'Metal System Trace' },
  instrument: { type: 'string', multiple: true }, 'xct-secs': { type: 'string', default: '4' }, labels: { type: 'boolean', default: true },
  'chrome-args': { type: 'string', default: '' }, 'dawn-features': { type: 'string', default: '' }, 'disable-dawn-features': { type: 'string', default: '' },
  phase: { type: 'string', default: 'split' }, after: { type: 'string', default: '40' }, 'dump-file': { type: 'string' },
  target: { type: 'string', default: 'attach' }, trigger: { type: 'string' }, 'trigger-delay': { type: 'string', default: '700' }, 'pre-delay': { type: 'string', default: '0' },
  'kernel-flags': { type: 'string' },
} });
if (!a.jobs) { console.error('--jobs FILE required'); process.exit(2); }
if (a.mode === 'dump' && !a['dump-file']) { console.error('--mode dump needs --dump-file'); process.exit(2); }

function freePort(): Promise<number> {
  return new Promise((res, rej) => { const s = createNetServer(); s.once('error', rej); s.listen(0, '127.0.0.1', () => { const { port } = s.address() as { port: number }; s.close(() => res(port)); }); });
}
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const outDir = path.resolve(ROOT, a.out!);
mkdirSync(outDir, { recursive: true });
let jobs = JSON.parse(readFileSync(path.resolve(ROOT, a.jobs), 'utf8')) as Record<string, unknown>[];
if (a['kernel-flags'] !== undefined) { const f = normalizePerfFlags(a['kernel-flags']); jobs = jobs.map((j) => ({ ...j, perfFlags: f })); }
const port = await freePort();
let cacheDir = path.join(ROOT, '.vite-cache-perf-tools');
try { if (lstatSync(path.join(ROOT, 'node_modules')).isSymbolicLink()) cacheDir = path.join(ROOT, `.vite-cache-${path.basename(ROOT).replace(/^WebGPURestirPT-?/, '') || 'wt'}-perf-tools`); } catch { /* default */ }
const vite = await createServer({ root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), server: { port, strictPort: true, host: '127.0.0.1', hmr: false, watch: null }, logLevel: 'warn', cacheDir });
await vite.listen();
const features: string[] = a['dawn-features'] ? a['dawn-features'].split(',').filter(Boolean) : [];
if (a.mode === 'dump') features.push('dump_shaders', 'disable_symbol_renaming');
if (a.labels || a.mode === 'xct') features.push('use_user_defined_labels_in_backend');
const args = ['--enable-webgpu-developer-features'];
if (features.length) args.push(`--enable-dawn-features=${[...new Set(features)].join(',')}`);
if (a['disable-dawn-features']) args.push(`--disable-dawn-features=${a['disable-dawn-features']}`);
if (a['chrome-args']) args.push(...a['chrome-args'].split(' ').filter(Boolean));
const udd = path.join(tmpdir(), `restirpt-prof-chrome-${process.pid}`);
console.log(`[drv] chrome args: ${args.join(' ')}`);
const ctx = await chromium.launchPersistentContext(udd, { channel: 'chrome', headless: true, args });
const reports: unknown[] = [];
let nConsole = 0;
let exitCode = 0;
try {
  const page = ctx.pages()[0] ?? await ctx.newPage();
  page.on('console', (m) => {
    if (a.mode === 'dump') { nConsole++; appendFileSync(a['dump-file']!, `${JSON.stringify({ t: m.type(), x: m.text() })}\n`); }
    else if (m.type() === 'error' || m.type() === 'warning') console.log(`[page:${m.type()}] ${m.text().slice(0, 400)}`);
  });
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
  await page.addInitScript(() => {
    const w = window as unknown as { __perf2: Record<string, number> };
    w.__perf2 = { split: 0, frame: 0, other: 0 };
    const orig = GPUDevice.prototype.createCommandEncoder;
    GPUDevice.prototype.createCommandEncoder = function (d?: GPUCommandEncoderDescriptor) {
      const l = d?.label ?? '';
      if (l === 'perf-split') w.__perf2.split++; else if (/^perf-\d+$/.test(l)) w.__perf2.frame++; else w.__perf2.other++;
      return orig.call(this, d);
    };
  });
  await page.goto(`http://127.0.0.1:${port}/validation/harness/harness.html`);
  await page.waitForFunction(() => (window as unknown as { __harness?: unknown }).__harness !== undefined, undefined, { timeout: 120_000 });
  const ps = execSync('ps -axo pid=,ppid=,command=', { encoding: 'utf8', maxBuffer: 64 << 20 }).split('\n').map((l) => l.trim()).filter(Boolean)
    .map((l) => { const m = /^(\d+)\s+(\d+)\s+(.*)$/.exec(l)!; return { pid: Number(m[1]), ppid: Number(m[2]), cmd: m[3] }; });
  const main = ps.find((p) => p.cmd.includes(`--user-data-dir=${udd}`) && !p.cmd.includes('--type='));
  const gpu = main ? ps.find((p) => p.ppid === main.pid && p.cmd.includes('--type=gpu-process')) : undefined;
  console.log(`[drv] browser pid ${main?.pid}, gpu pid ${gpu?.pid}`);
  if (a.mode === 'xct') {
    if (!gpu) throw new Error('GPU process not found');
    const job = jobs[0];
    let finished = false;
    const run = page.evaluate((x) => (window as unknown as { __harness: { renderPerf(o: unknown): Promise<unknown> } }).__harness.renderPerf(x), job)
      .then((r) => { reports.push(r); finished = true; return r; }, (e: unknown) => { finished = true; console.log(`[drv] job error ${String(e).slice(0, 300)}`); });
    const t0 = Date.now(); const want = Number(a.after); let c: Record<string, number> = {};
    while (Date.now() - t0 < 300_000 && !finished) {
      c = await page.evaluate(() => (window as unknown as { __perf2: Record<string, number> }).__perf2);
      if ((c[a.phase!] ?? 0) >= want) break;
      await sleep(200);
    }
    console.log(`[drv] phase ${a.phase} reached after ${((Date.now() - t0) / 1000).toFixed(1)} s: ${JSON.stringify(c)}`);
    if (finished) throw new Error('job finished before the phase threshold (raise passFrames / frames)');
    if (Number(a['pre-delay']) > 0) await sleep(Number(a['pre-delay']) * 1000);
    const trace = path.join(outDir, 'xct', `${a.tag}.trace`);
    mkdirSync(path.dirname(trace), { recursive: true });
    rmSync(trace, { recursive: true, force: true });
    const xargs = ['xctrace', 'record', '--template', a.template!, ...(a.instrument ?? []).flatMap((i) => ['--instrument', i]),
      ...(a.target === 'all' ? ['--all-processes'] : ['--attach', String(gpu.pid)]), '--time-limit', `${a['xct-secs']}s`, '--no-prompt', '--output', trace];
    console.log(`[drv] xcrun ${xargs.join(' ')}`);
    const xt = spawn('xcrun', xargs, { stdio: ['ignore', 'pipe', 'pipe'] });
    let trig = false;
    xt.stdout.on('data', (d) => {
      process.stdout.write(`[xctrace] ${d}`);
      if (a.trigger && !trig && String(d).includes('Ctrl-C')) {
        trig = true;
        setTimeout(() => { console.log(`[drv] trigger: ${a.trigger}`); spawn('/bin/zsh', ['-c', a.trigger!], { stdio: 'inherit' }); }, Number(a['trigger-delay']));
      }
    });
    xt.stderr.on('data', (d) => process.stdout.write(`[xctrace!] ${d}`));
    const done = new Promise<number | null>((r) => xt.on('exit', (code) => r(code)));
    const code = await Promise.race([done, sleep(240_000).then(() => 'timeout' as const)]);
    if (code === 'timeout') { console.log('[drv] xctrace timeout: SIGINT'); xt.kill('SIGINT'); await Promise.race([done, sleep(30_000)]); exitCode = 1; }
    // exit 2 = "run issues detected" (e.g. 'Fatal logging system error: the log archive is corrupt'): the Metal tables
    // are still recorded and exportable, so it is a warning; a missing trace is an error
    else { console.log(`[drv] xctrace exit ${code}`); if (code !== 0 && !(code === 2 && existsSync(trace))) exitCode = 1; }
    c = await page.evaluate(() => (window as unknown as { __perf2: Record<string, number> }).__perf2).catch(() => c);
    console.log(`[drv] after recording: ${JSON.stringify(c)}; job finished=${finished}; trace ${path.relative(ROOT, trace)}`);
    await Promise.race([run, sleep(500)]);
  } else {
    for (const j of jobs) {
      const r = await page.evaluate((x) => (window as unknown as { __harness: { renderPerf(o: unknown): Promise<unknown> } }).__harness.renderPerf(x), j) as Record<string, any>;
      reports.push(r);
      if (!r.ok) exitCode = 1;
      const p = (r.passes ?? []).map((x: any) => `${x.name} ${x.meanMs.toFixed(2)}`).join(', ');
      const dn = r.denoiser ? ` | dn ${r.denoiser.totalMs.toFixed(2)}` : '';
      console.log(`${r.ok ? 'OK  ' : 'FAIL'} ${String(r.label).padEnd(34)} frame ${r.frame.meanMs.toFixed(2)} ms (median ${r.frame.medianMs.toFixed(2)}) Σpasses ${r.passTotalMs?.toFixed(2) ?? '-'}${dn}${r.ok ? '' : ' ' + r.errors.join('; ').slice(0, 500)}\n     ${p}`);
      if (a.mode === 'perf') writeFileSync(path.join(outDir, `${a.tag}.json`), `${JSON.stringify({ tag: a.tag, args, reports }, null, 1)}\n`);
    }
  }
} catch (e) {
  console.error(`[drv] ${e instanceof Error ? e.message : String(e)}`);
  exitCode = 1;
} finally {
  await ctx.close().catch(() => {});
  await vite.close();
  rmSync(udd, { recursive: true, force: true });
}
if (a.mode === 'dump') console.log(`[drv] console messages captured: ${nConsole}`);
console.log('[drv] done');
process.exit(exitCode);
