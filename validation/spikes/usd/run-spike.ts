// Drive the LightUSD spike page headless in Chrome and diff every loader against a pxr dump.
//   npx tsx validation/spikes/usd/run-spike.ts [--port 5190] [--files a.usda,b.usdc] [--no-pxr] [--no-stock]
// Pass A: `npx vite validation/spikes/usd` with Vite's default config, modes direct + three.
// Pass B: same root with --config vite.lightusd-stock.config.ts, modes next + legacy (stock LightUSDWorkerLoader).
// Writes validation/out/usd-spike/{pxr,lightusd-*,three}.<file>.json, compare.json, summary.json.
import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium } from 'playwright';
import { compareDumps, type Comparison } from './compare.ts';
import type { SceneDump } from './dump-types.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '../../..');
const ASSETS = join(ROOT, 'validation/assets/usd-spike');
const OUT = join(ROOT, 'validation/out/usd-spike');
const PXR_PYTHON = '/Applications/Blender.app/Contents/Resources/5.1/python/bin/python3.13';

const { values: args } = parseArgs({
  options: {
    port: { type: 'string', default: '5190' },
    files: { type: 'string' },
    'no-pxr': { type: 'boolean', default: false },
    'no-stock': { type: 'boolean', default: false },
    headed: { type: 'boolean', default: false },
  },
});
const PERF = join(OUT, 'perf');
const files = args.files?.split(',') ?? [
  ...readdirSync(ASSETS).filter((f) => /\.usd[acz]$/.test(f)).sort(),
  ...(existsSync(join(PERF, 'perf.usdc')) ? ['perf.usdc'] : []),
];
const assetPath = (f: string): string => (existsSync(join(ASSETS, f)) ? join(ASSETS, f) : join(PERF, f));
mkdirSync(OUT, { recursive: true });

function runPxr(): void {
  const r = spawnSync(PXR_PYTHON, [join(HERE, 'pxr_dump.py'), ...files.map(assetPath), '--out', OUT], {
    cwd: ROOT, encoding: 'utf8',
  });
  process.stdout.write(r.stdout);
  if (r.status !== 0) throw new Error(`pxr_dump failed:\n${r.stderr}`);
}

async function startVite(port: number, config?: string): Promise<ChildProcess> {
  const extra = config ? ['--config', config] : [];
  const vite = spawn('npx', ['vite', 'validation/spikes/usd', '--port', String(port), '--strictPort', ...extra], {
    cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'], detached: true,
  });
  let log = '';
  vite.stdout!.on('data', (b) => (log += b));
  vite.stderr!.on('data', (b) => (log += b));
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    try {
      const r = await fetch(`http://localhost:${port}/`);
      if (r.ok) return vite;
    } catch {
      /* not up yet */
    }
    if (vite.exitCode !== null) break;
    await new Promise((r) => setTimeout(r, 250));
  }
  vite.kill();
  throw new Error(`vite did not start:\n${log}`);
}

const stopVite = (vite: ChildProcess): void => {
  try {
    process.kill(-vite.pid!, 'SIGTERM');
  } catch {
    vite.kill('SIGTERM');
  }
};

interface SpikeResult {
  done: boolean;
  userAgent: string;
  crossOriginIsolated: boolean;
  sharedArrayBuffer: boolean;
  init: Record<string, { ms: number; error?: string }>;
  dumps: SceneDump[];
}

async function runPage(port: number, modes: string): Promise<{ result: SpikeResult; wasm: Record<string, number>; workers: string[]; console: string[] }> {
  const browser = await chromium.launch({ channel: 'chrome', headless: !args.headed });
  const page = await browser.newPage();
  const wasm: Record<string, number> = {};
  const workers: string[] = [];
  const consoleLines: string[] = [];
  page.on('worker', (w) => workers.push(w.url()));
  page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') consoleLines.push(`${m.type()}: ${m.text()}`); });
  page.on('pageerror', (e) => consoleLines.push(`pageerror: ${e.message}`));
  // Worker fetches surface on the browser context in Chromium.
  page.context().on('response', async (r) => {
    if (!/\.wasm(\?|$)/.test(r.url())) return;
    try {
      wasm[r.url().split('/').pop()!.split('?')[0]] = (await r.body()).length;
    } catch {
      wasm[r.url()] = -1;
    }
  });
  const q = new URLSearchParams({ files: files.join(','), modes });
  await page.goto(`http://localhost:${port}/?${q}`);
  await page.waitForFunction(() => (window as unknown as { __USD_SPIKE__?: { done: boolean } }).__USD_SPIKE__?.done === true, null, {
    timeout: 10 * 60_000,
  });
  const result = (await page.evaluate(() => (window as unknown as { __USD_SPIKE__: unknown }).__USD_SPIKE__)) as SpikeResult;
  await browser.close();
  return { result, wasm, workers, console: consoleLines };
}

function fmtTally(t: { ok: number; mismatch: number; missing: number }): string {
  return `${t.ok}/${t.ok + t.mismatch + t.missing}` + (t.mismatch ? ` ${t.mismatch}x` : '') + (t.missing ? ` ${t.missing}?` : '');
}

async function main(): Promise<void> {
  if (!args['no-pxr']) runPxr();
  const port = Number(args.port);
  const passes: { modes: string; config?: string }[] = [{ modes: 'direct,three' }];
  if (!args['no-stock']) passes.push({ modes: 'next,legacy', config: 'validation/spikes/usd/vite.lightusd-stock.config.ts' });
  const pages = [];
  for (const pass of passes) {
    const vite = await startVite(port, pass.config);
    try {
      pages.push({ ...(await runPage(port, pass.modes)), pass });
    } finally {
      stopVite(vite);
      await new Promise((r) => setTimeout(r, 500));
    }
  }
  const result: SpikeResult = { ...pages[0].result, init: {}, dumps: [] };
  for (const p of pages) {
    Object.assign(result.init, p.result.init);
    result.dumps.push(...p.result.dumps);
  }
  const page = {
    wasm: Object.fromEntries(pages.map((p) => [p.pass.modes, p.wasm])),
    workers: pages.flatMap((p) => p.workers),
    console: pages.flatMap((p) => p.console.map((l) => `[${p.pass.modes}] ${l}`)),
  };
  for (const d of result.dumps) writeFileSync(join(OUT, `${d.source}.${d.file}.json`), JSON.stringify(d, null, 1));

  const comparisons: Comparison[] = [];
  for (const d of result.dumps) {
    const ref = JSON.parse(readFileSync(join(OUT, `pxr.${d.file}.json`), 'utf8')) as SceneDump;
    comparisons.push(compareDumps(ref, d));
  }
  writeFileSync(join(OUT, 'compare.json'), JSON.stringify(comparisons, null, 1));

  const pkg = join(ROOT, 'node_modules/lightusd');
  const sizes = Object.fromEntries(
    readdirSync(pkg).filter((f) => /\.(wasm|zst|js)$/.test(f)).map((f) => [f, statSync(join(pkg, f)).size]),
  );
  const summary = {
    date: new Date().toISOString(),
    userAgent: result.userAgent,
    crossOriginIsolated: result.crossOriginIsolated,
    sharedArrayBufferDefined: result.sharedArrayBuffer,
    workers: page.workers,
    init: result.init,
    wasmFetchedBytes: page.wasm,
    packageFileBytes: sizes,
    consoleErrors: page.console,
    timings: result.dumps.map((d) => ({ source: d.source, file: d.file, ok: d.ok, ...d.timings })),
    fileBytes: Object.fromEntries(files.map((f) => [f, statSync(assetPath(f)).size])),
  };
  writeFileSync(join(OUT, 'summary.json'), JSON.stringify(summary, null, 1));

  console.log('\nsource            file                     load ms  stage lights(m/ref) lightFields        draws(m/ref/got) tris   world  mats   PI world');
  for (const c of comparisons) {
    const lf = Object.values(c.lights.fields).reduce((a, t) => ({ ok: a.ok + t.ok, mismatch: a.mismatch + t.mismatch, missing: a.missing + t.missing }), { ok: 0, mismatch: 0, missing: 0 });
    const t = result.dumps.find((d) => d.source === c.source && d.file === c.file)?.timings?.loadMs ?? NaN;
    console.log(
      [
        c.source.padEnd(17), c.file.padEnd(24), t.toFixed(1).padStart(7), (c.loaded ? Object.values(c.stage).every((s) => s === 'ok') ? 'ok' : 'DIFF' : 'FAIL').padEnd(5),
        `${c.lights.matched}/${c.lights.ref}`.padEnd(14), fmtTally(lf).padEnd(18),
        `${c.draws.matched}/${c.draws.ref}/${c.draws.got}`.padEnd(16), fmtTally(c.draws.triangles).padEnd(6), fmtTally(c.draws.world).padEnd(6),
        fmtTally(c.draws.materialAssignment).padEnd(6), `${fmtTally(c.pointInstancers.world)} (${c.pointInstancers.gotInstances}/${c.pointInstancers.refInstances})`,
      ].join(' '),
    );
  }
  console.log(`\nwasm fetched: ${JSON.stringify(page.wasm)}\ninit: ${JSON.stringify(result.init)}\nworkers: ${page.workers.length}`);
  if (page.console.length) console.log(`console:\n  ${page.console.slice(0, 20).join('\n  ')}`);
  console.log(`\nwrote ${OUT}/{compare,summary}.json`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
