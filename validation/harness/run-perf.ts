// M8 performance runner (docs/decisions/m8-perf.md §1): Vite (no HMR) + headless Chrome with
// --enable-webgpu-developer-features (unquantised timestamps for the denoiser re-run) on the harness page;
// window.__harness.renderPerf per job, each job under its own GPU-lock hold (a job takes ~0.5–2 min).
//   npx tsx validation/harness/run-perf.ts --suite baseline [--res 540p,720p] [--only cornell,sponza] [--out DIR] [--tag TAG]
//   npx tsx validation/harness/run-perf.ts --jobs FILE   (a JSON array of PerfOptions)
// Writes <out>/<tag>.json (all reports) and prints a table. Default out: validation/out/m8-perf.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createServer as createNetServer } from 'node:net';
import { lstatSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium, type Browser, type Page } from 'playwright';
import { createServer, type ViteDevServer } from 'vite';
import { acquireGpuLock } from './gpu-lock.ts';
import type { PerfOptions, PerfReport } from './perf-run.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const HDRI = '/validation/assets/downloaded/hdri/kloofendal_48d_partly_cloudy_puresky_1k.hdr';
export const PERF_SCENES: Record<string, Pick<PerfOptions, 'scene' | 'env' | 'lightAnim'>> = {
  cornell: { scene: '/validation/scenes/cornell_i_512/', lightAnim: { index: 0, amp: 0.05 } },
  sponza: { scene: '/validation/assets/downloaded/sponza/Sponza.gltf', env: HDRI, lightAnim: { index: 0, amp: 1.0 } },
  crossings: { scene: '/validation/out/m6/scenes/m6_crossings_B_256/', lightAnim: { index: 0, amp: 0.05 } },
  nm_smooth: { scene: '/validation/out/m7/scenes/m7_nm_smooth_256/', lightAnim: { index: 0, amp: 0.05 } },
};
export const RES: Record<string, [number, number]> = { '540p': [960, 540], '720p': [1280, 720] };

/** The M8 target configurations (PLAN §5 M8 exit): N=3 static at 540p and 720p, N=1 + moving lights at 540p. */
export function suiteJobs(name: string, scenes: string[], res: string[], extra: Partial<PerfOptions> = {}): PerfOptions[] {
  const jobs: PerfOptions[] = [];
  for (const sc of scenes) {
    const s = PERF_SCENES[sc];
    if (!s) throw new Error(`unknown scene ${sc}`);
    for (const r of res) {
      const [w, h] = RES[r];
      jobs.push({ scene: s.scene, env: s.env, width: w, height: h, label: `${sc}@${r} N3`, ...extra });
      if (name === 'baseline' && r === '540p') jobs.push({ scene: s.scene, env: s.env, width: w, height: h, restir: { slots: 1 }, lightAnim: s.lightAnim, label: `${sc}@${r} N1+moving`, ...extra });
    }
  }
  return jobs;
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const s = createNetServer();
    s.once('error', reject);
    s.listen(0, '127.0.0.1', () => { const { port } = s.address() as { port: number }; s.close(() => resolve(port)); });
  });
}

async function openHarness(): Promise<{ page: Page; chromeVersion: string; close(): Promise<void> }> {
  const port = await freePort();
  let cacheDir: string | undefined;
  try { if (lstatSync(path.join(ROOT, 'node_modules')).isSymbolicLink()) cacheDir = path.join(ROOT, `.vite-cache-${path.basename(ROOT).replace(/^WebGPURestirPT-?/, '') || 'wt'}-harness`); } catch { /* default */ }
  const vite: ViteDevServer = await createServer({
    root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), server: { port, strictPort: true, host: '127.0.0.1', hmr: false, watch: null }, logLevel: 'warn',
    ...(cacheDir ? { cacheDir } : {}),
  });
  await vite.listen();
  let browser: Browser | undefined;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-webgpu-developer-features'] });
    const chromeVersion = browser.version();
    const page = await browser.newPage();
    page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log(`[page:${m.type()}] ${m.text()}`); });
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
    await page.goto(`http://127.0.0.1:${port}/validation/harness/harness.html`);
    await page.waitForFunction(() => window.__harness !== undefined, undefined, { timeout: 120_000 });
    return { page, chromeVersion, close: async () => { await browser?.close(); await vite.close(); } };
  } catch (e) {
    await browser?.close(); await vite.close();
    throw e;
  }
}

export function fmtReport(r: PerfReport): string {
  const p = (r.passes ?? []).map((x) => `${x.name} ${x.meanMs.toFixed(2)}`).join(', ');
  const dn = r.denoiser ? ` | dn ${r.denoiser.totalMs.toFixed(2)} (${r.denoiser.passes.map((x) => `${x.name.replace(/^dn_/, '')} ${x.ms.toFixed(2)}`).join(', ')})` : '';
  return `${r.ok ? 'OK  ' : 'FAIL'} ${r.label.padEnd(24)} frame ${r.frame.meanMs.toFixed(2)} ms (median ${r.frame.medianMs.toFixed(2)}, ${r.frame.frames} fr), latency ${r.latency?.meanMs.toFixed(2) ?? '-'}, Σpasses ${r.passTotalMs?.toFixed(2) ?? '-'}, f_r ${r.counters?.fr.toFixed(3) ?? '-'}\n     ${p}${dn}${r.ok ? '' : `\n     ${r.errors.join('; ')}`}`;
}

async function main(): Promise<number> {
  const { values: a } = parseArgs({ options: {
    suite: { type: 'string' }, jobs: { type: 'string' }, res: { type: 'string', default: '540p,720p' }, only: { type: 'string' },
    out: { type: 'string', default: 'validation/out/m8-perf' }, tag: { type: 'string' }, frames: { type: 'string' }, extra: { type: 'string' },
  } });
  const extra = a.extra ? JSON.parse(a.extra) as Partial<PerfOptions> : {};
  if (a.frames) extra.frames = Number(a.frames);
  const jobs: PerfOptions[] = a.jobs ? (JSON.parse(readFileSync(path.resolve(ROOT, a.jobs), 'utf8')) as PerfOptions[]).map((j) => ({ ...j, ...extra }))
    : suiteJobs(a.suite ?? 'baseline', (a.only ?? Object.keys(PERF_SCENES).join(',')).split(','), a.res.split(','), extra);
  const tag = a.tag ?? `perf-${new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-')}`;
  const outDir = path.resolve(ROOT, a.out);
  mkdirSync(outDir, { recursive: true });
  const h = await openHarness();
  const reports: PerfReport[] = [];
  let fails = 0;
  try {
    for (const j of jobs) {
      const release = await acquireGpuLock(`m8-perf-${j.label ?? 'job'}`.replace(/[^\w.@+-]/g, '_').slice(0, 60));
      let rep: PerfReport;
      try { rep = await h.page.evaluate((x) => window.__harness!.renderPerf(x), j); } finally { release(); }
      reports.push(rep);
      if (!rep.ok) fails++;
      console.log(fmtReport(rep));
      writeFileSync(path.join(outDir, `${tag}.json`), `${JSON.stringify({ tag, chrome: h.chromeVersion, reports }, null, 1)}\n`);
    }
  } finally {
    await h.close();
  }
  console.log(`wrote ${path.relative(ROOT, path.join(outDir, `${tag}.json`))}; ${fails ? `RESULT: FAIL (${fails}/${jobs.length})` : `RESULT: PASS (${jobs.length})`}`);
  return fails ? 1 : 0;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  main().then((c) => process.exit(c), (e: unknown) => { console.error(e); process.exit(1); });
}
