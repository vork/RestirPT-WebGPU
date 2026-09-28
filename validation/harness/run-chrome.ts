// Chrome 154 harness driver (plan §1.1, §5 M0). Starts Vite, launches headless Chrome through Playwright and runs
// the M0 smoke: hardware Metal adapter + profile limits, IO orientation upload (+ Python check), allocation probe.
//   npx tsx validation/harness/run-chrome.ts --smoke [--run <id>] [--members 16,64] [--skip-alloc]
import { execFile } from 'node:child_process';
import { existsSync, mkdirSync, rmdirSync } from 'node:fs';
import { mkdir, stat, writeFile } from 'node:fs/promises';
import { createServer as createNetServer } from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium, type Browser, type Page } from 'playwright';
import { createServer, type ViteDevServer } from 'vite';
import { PROFILE_CHROME154_M5PRO } from '../../src/core/gpu/profile.ts';
import type { AllocProbeReport } from './alloc-probe.ts';
import type { SmokeReport } from './harness.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const OUT = path.join(ROOT, 'validation/out');
const PYTHON = path.join(ROOT, 'validation/.venv/bin/python');
const ORIENT_CHECK = path.join(ROOT, 'validation/tools/orientation_check.py');
const GPU_LOCK = '/tmp/restirpt-gpu.lock';

const { values: args } = parseArgs({
  options: {
    smoke: { type: 'boolean', default: false },
    run: { type: 'string' },
    members: { type: 'string', default: '16,64' },
    'skip-alloc': { type: 'boolean', default: false },
    'python-wait': { type: 'string', default: '300' },
  },
});

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
const now = () => performance.now();
const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');

interface Check { name: string; ok: boolean; detail?: string }
const checks: Check[] = [];
function check(name: string, ok: boolean, detail?: string): boolean {
  checks.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
  return ok;
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const s = createNetServer();
    s.once('error', reject);
    s.listen(0, '127.0.0.1', () => {
      const { port } = s.address() as { port: number };
      s.close(() => resolve(port));
    });
  });
}

// GPU lock shared with the Blender renders and other GPU-heavy jobs (plan §1.8: never overlap).
let lockHeld = false;
async function acquireGpuLock(): Promise<number> {
  const t0 = now();
  for (;;) {
    try { mkdirSync(GPU_LOCK); lockHeld = true; return now() - t0; } catch { await sleep(5000); }
  }
}
function releaseGpuLock(): void {
  if (!lockHeld) return;
  try { rmdirSync(GPU_LOCK); } catch { /* already gone */ }
  lockHeld = false;
}
for (const sig of ['SIGINT', 'SIGTERM'] as const) process.on(sig, () => { releaseGpuLock(); process.exit(130); });
process.on('exit', releaseGpuLock);

function withTimeout<T>(p: Promise<T>, ms: number, what: string): Promise<T> {
  return Promise.race([p, sleep(ms).then(() => { throw new Error(`${what} timed out after ${ms} ms`); })]);
}

async function launch(unsafeFlag: boolean): Promise<Browser> {
  return chromium.launch({
    channel: 'chrome',
    headless: true,
    args: unsafeFlag ? ['--enable-unsafe-webgpu'] : [],
  });
}

async function openHarness(browser: Browser, url: string): Promise<Page> {
  const page = await browser.newPage();
  page.on('console', (m) => console.log(`[page:${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
  await page.goto(url);
  await page.waitForFunction(() => window.__harness !== undefined, undefined, { timeout: 30_000 });
  return page;
}

function run(cmd: string, argv: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    execFile(cmd, argv, { cwd: ROOT, timeout: 120_000 }, (err, stdout, stderr) => {
      const code = err ? (typeof err.code === 'number' ? err.code : 1) : 0;
      resolve({ code, stdout: stdout.toString(), stderr: stderr.toString() });
    });
  });
}

async function main(): Promise<number> {
  if (!args.smoke) {
    console.error('usage: run-chrome.ts --smoke [--run <id>] [--members 16,64] [--skip-alloc]');
    return 2;
  }
  const runId = args.run ?? `m0-smoke-${stamp()}`;
  if (!/^[A-Za-z0-9._-]+$/.test(runId)) throw new Error(`bad run id ${runId}`);
  const runDir = path.join(OUT, runId);
  await mkdir(runDir, { recursive: true });
  const timings: Record<string, number> = {};
  const report: Record<string, unknown> = { runId, startedAt: new Date().toISOString(), checks, timings };

  let t = now();
  const port = await freePort();
  const vite: ViteDevServer = await createServer({
    root: ROOT,
    configFile: path.join(ROOT, 'vite.config.ts'),
    server: { port, strictPort: true, host: '127.0.0.1' },
    logLevel: 'warn',
  });
  await vite.listen();
  const url = `http://127.0.0.1:${port}/validation/harness/harness.html`;
  timings.viteStartMs = now() - t;
  console.log(`vite dev server on :${port}, run ${runId}`);

  let browser: Browser | undefined;
  try {
    // 1) Try WITHOUT --enable-unsafe-webgpu first (plan §5 M0); fall back to the flag only if no adapter.
    t = now();
    browser = await launch(false);
    let page = await openHarness(browser, url);
    let smoke = await page.evaluate(() => window.__harness!.smoke());
    const withoutFlag = { adapterAvailable: smoke.adapterAvailable, hasNavigatorGpu: smoke.hasNavigatorGpu, error: smoke.error };
    let usedUnsafeFlag = false;
    if (!smoke.adapterAvailable) {
      console.log(`no adapter without --enable-unsafe-webgpu (${smoke.error}); relaunching with the flag`);
      await browser.close();
      browser = await launch(true);
      page = await openHarness(browser, url);
      smoke = await page.evaluate(() => window.__harness!.smoke());
      usedUnsafeFlag = true;
    }
    timings.smokeMs = now() - t;
    const chromeVersion = browser.version();
    report.chrome = { version: chromeVersion, userAgent: smoke.userAgent, headless: true, usedUnsafeFlag, withoutFlag };
    report.smoke = smoke;

    check('webgpu adapter available', smoke.adapterAvailable, usedUnsafeFlag ? 'needed --enable-unsafe-webgpu' : 'no unsafe flag');
    check('chrome major 154', /^154\./.test(chromeVersion), chromeVersion);
    check('hardware Metal context', smoke.ok, smoke.error);
    const ctx = smoke.context as { vendor?: string; architecture?: string; limits?: Record<string, number> } | undefined;
    if (ctx) {
      check('vendor apple', ctx.vendor === 'apple', ctx.vendor);
      check('architecture metal*', (ctx.architecture ?? '').startsWith('metal'), ctx.architecture);
      check('not fallback adapter', smoke.isFallbackAdapter === false, String(smoke.isFallbackAdapter));
      const short: string[] = [];
      for (const [k, want] of Object.entries(PROFILE_CHROME154_M5PRO)) {
        const have = ctx.limits?.[k];
        if (have === undefined || have < want) short.push(`${k}=${have ?? 'missing'}<${want}`);
      }
      check('device limits >= CHROME154_M5PRO profile', short.length === 0, short.join(', ') || undefined);
    }

    // 2) IO orientation: page encodes (c, r, c+r) as PFM and uploads through the dev middleware.
    t = now();
    try {
      await page.evaluate((r) => window.__harness!.orientationUpload(r), runId);
      const st = await stat(path.join(runDir, 'orientation.pfm'));
      check('orientation.pfm uploaded', st.size === 'PF\n64 48\n-1.0\n'.length + 64 * 48 * 12, `${st.size} B`);
    } catch (e) {
      check('orientation.pfm uploaded', false, String(e));
    }
    await page.evaluate(([r, s]) => window.__harness!.log(r, { kind: 'smoke', ok: s }), [runId, smoke.ok] as const)
      .then(() => check('log endpoint', existsSync(path.join(runDir, 'log.jsonl'))))
      .catch((e: unknown) => check('log endpoint', false, String(e)));
    timings.orientationUploadMs = now() - t;

    // 3) Allocation probe for the ensemble atlas, under the shared GPU lock.
    const allocs: AllocProbeReport[] = [];
    if (smoke.ok && !args['skip-alloc']) {
      const members = args.members!.split(',').map(Number).filter((n) => n > 0);
      console.log('acquiring GPU lock for the allocation probe...');
      timings.gpuLockWaitMs = await acquireGpuLock();
      try {
        for (const m of members) {
          t = now();
          try {
            const rep = await withTimeout(page.evaluate((mm) => window.__harness!.allocProbe({ members: mm }), m), 180_000, `allocProbe E=${m}`);
            allocs.push(rep);
            const bad = rep.buffers.filter((b) => !b.ok).map((b) => `${b.name}: ${b.error}`);
            check(`alloc probe E=${m}`, rep.ok,
              `${(rep.allocatedBytes / 2 ** 30).toFixed(3)} / ${(rep.requestedBytes / 2 ** 30).toFixed(3)} GiB in ${rep.buffers.length} buffers, ` +
              `alloc ${rep.allocMs.toFixed(0)} ms, verify ${rep.verifyMs.toFixed(0)} ms${bad.length ? `; ${bad.join('; ')}` : ''}${rep.deviceLost ? `; lost ${rep.deviceLost}` : ''}`);
          } catch (e) {
            check(`alloc probe E=${m}`, false, String(e));
          }
          timings[`allocProbeE${m}Ms`] = now() - t;
        }
      } finally {
        releaseGpuLock();
      }
    }
    report.allocProbe = allocs;
  } finally {
    await browser?.close();
    await vite.close();
  }

  // 4) Python orientation check (validation/tools/orientation_check.py, created by another M0 worker).
  const pfm = path.join(runDir, 'orientation.pfm');
  if (existsSync(pfm)) {
    t = now();
    const waitMs = Number(args['python-wait']) * 1000;
    while (!(existsSync(PYTHON) && existsSync(ORIENT_CHECK)) && now() - t < waitMs) {
      console.log('waiting for validation/.venv/bin/python and validation/tools/orientation_check.py ...');
      await sleep(10_000);
    }
    if (existsSync(PYTHON) && existsSync(ORIENT_CHECK)) {
      const r = await run(PYTHON, [ORIENT_CHECK, 'pfm', pfm]);
      report.orientationCheck = { status: r.code === 0 ? 'pass' : 'fail', ...r };
      check('orientation_check.py pfm', r.code === 0, (r.stdout + r.stderr).trim().split('\n').slice(-3).join(' | '));
    } else {
      report.orientationCheck = { status: 'skipped', reason: 'venv python or orientation_check.py missing after wait' };
      console.log('SKIP  orientation_check.py pfm (tool missing)');
    }
    timings.orientationCheckMs = now() - t;
  }

  const ok = checks.every((c) => c.ok);
  report.ok = ok;
  report.finishedAt = new Date().toISOString();
  await writeFile(path.join(runDir, 'm0-smoke.json'), JSON.stringify(report, null, 2));

  const c = report.chrome as { version: string; usedUnsafeFlag: boolean } | undefined;
  console.log('\n=== M0 Chrome smoke summary ===');
  console.log(`run          ${runId}`);
  console.log(`chrome       ${c?.version} (headless, unsafe flag ${c?.usedUnsafeFlag ? 'REQUIRED' : 'not needed'})`);
  console.log(`checks       ${checks.filter((x) => x.ok).length}/${checks.length} passed`);
  for (const [k, v] of Object.entries(timings)) console.log(`${k.padEnd(20)} ${v.toFixed(0)} ms`);
  console.log(`report       ${path.relative(ROOT, path.join(runDir, 'm0-smoke.json'))}`);
  console.log(ok ? 'RESULT: PASS' : 'RESULT: FAIL');
  return ok ? 0 : 1;
}

main().then(
  (code) => { releaseGpuLock(); process.exit(code); },
  (e: unknown) => { releaseGpuLock(); console.error(e); process.exit(1); },
);
