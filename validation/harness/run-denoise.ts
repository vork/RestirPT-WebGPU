// M5.5 denoiser evaluation driver (docs/decisions/denoiser.md §11): Vite (no HMR) + headless Chrome on the harness page,
// window.__harness.renderDenoise under the shared GPU lock. Output: validation/out/<run>/ (see denoise-run.ts).
//   npx tsx validation/harness/run-denoise.ts --package validation/scenes/cornell_i_512 --mode flip --frames 64 --eval-frames 16,32,48,63 --seed 1
//   npx tsx validation/harness/run-denoise.ts --package validation/scenes/ixs_e_half_256 --mode recovery --pkg-frames 13x32,14x24 --seed 1
//   npx tsx validation/harness/run-denoise.ts --package validation/scenes/cornell_i_512 --mode timing --width 960 --height 540
//   npx tsx validation/harness/run-denoise.ts --jobs FILE   (a JSON array of argument lists, one page and ONE lock hold)
// --pkg-frames: comma list of package frames, `fxN` repeats f N times. --iterations / --alpha-min / --lambda0 / --lambda1
// override the denoiser settings. Timing runs launch Chrome with --enable-webgpu-developer-features (unquantised
// timestamps) unless --no-dev-features.
// perf2 WP-Q (equal-quality harness, validation/harness/run-eq.ts): --restir JSON (Partial<RestirSettings> over the app
// mode's preset, e.g. '{"risM":16}'), --renderer JSON (Partial<RendererOptions>, e.g. '{"lightMode":"B","bvhKind":"auto"}'),
// --perf-flags A,B (opaque names for WP-0's perf-flag registry), --pan-osc AMP / --light-osc INDEX:AMP with --osc-knots
// 0,16,32,48,63 (oscillating motion, the knots on the base state), --lock-per-job (with --jobs: one GPU-lock hold per
// job instead of one for the list, so long lists stay inside the 12-min hold rule).
import { lstatSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createServer as createNetServer } from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium, type Browser, type Page } from 'playwright';
import { createServer, type ViteDevServer } from 'vite';
import { acquireGpuLock, GPU_LOCK } from './gpu-lock.ts';
import type { RenderDenoiseOptions } from './denoise-run.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const OPTIONS = {
  'experimental-builder': { type: 'string' }, 'capture-vbuffer': { type: 'boolean', default: false },
  package: { type: 'string' }, mode: { type: 'string', default: 'flip' }, frames: { type: 'string', default: '64' }, 'eval-frames': { type: 'string' },
  'pkg-frames': { type: 'string' }, seed: { type: 'string', default: '1' }, run: { type: 'string' }, width: { type: 'string' }, height: { type: 'string' },
  iterations: { type: 'string' }, 'alpha-min': { type: 'string' }, 'sigma-l': { type: 'string' }, 'sigma-a': { type: 'string' }, 'var-corr': { type: 'string' }, 'no-resolve': { type: 'boolean', default: false }, 'no-guide': { type: 'boolean', default: false }, 'inv-radius': { type: 'string' }, 'lum-min-n': { type: 'string' }, 'lum-pre': { type: 'string' }, 'debug-views': { type: 'string' }, 'debug-frames': { type: 'string' }, jitter: { type: 'string' }, 'no-denoise': { type: 'boolean', default: false }, accumulate: { type: 'boolean', default: false }, pan: { type: 'string' }, 'light-anim': { type: 'string' }, 'dn-json': { type: 'string' }, lambda0: { type: 'string' }, lambda1: { type: 'string' },
  restir: { type: 'string' }, renderer: { type: 'string' }, 'perf-flags': { type: 'string' }, 'pan-osc': { type: 'string' }, 'light-osc': { type: 'string' }, 'osc-knots': { type: 'string', default: '0,16,32,48,63' },
  warmup: { type: 'string' }, 'timing-submits': { type: 'string' }, 'timing-runs': { type: 'string' }, 'no-dev-features': { type: 'boolean', default: false },
} as const;
const parse = (argv?: string[]) => parseArgs({ options: { ...OPTIONS, jobs: { type: 'string' }, 'lock-per-job': { type: 'boolean', default: false } }, ...(argv ? { args: argv } : {}) }).values;
const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');

/** `13x32,14x24` → [13 ×32, 14 ×24]; plain numbers once. */
export function parsePkgFrames(s: string): number[] {
  const out: number[] = [];
  for (const part of s.split(',').filter(Boolean)) {
    const m = /^(-?\d+)(?:x(\d+))?$/.exec(part.trim());
    if (!m) throw new Error(`--pkg-frames: bad item '${part}'`);
    for (let k = 0; k < Number(m[2] ?? 1); k++) out.push(Number(m[1]));
  }
  return out;
}

function knots(s: string): number[] {
  const k = s.split(',').map(Number);
  if (k.length < 2 || k.some((x, i) => !Number.isInteger(x) || (i > 0 && x <= k[i - 1]))) throw new Error(`--osc-knots: increasing integers, got '${s}'`);
  return k;
}

function optionsOf(a: ReturnType<typeof parse>, chromeVersion: string): RenderDenoiseOptions {
  if (!a.package) throw new Error('--package is required');
  const pkgUrl = `/${path.relative(ROOT, path.resolve(ROOT, a.package)).split(path.sep).join('/')}/`;
  const pkgFrames = a['pkg-frames'] ? parsePkgFrames(a['pkg-frames']) : undefined;
  const mode = a.mode as RenderDenoiseOptions['mode'];
  if (!['flip', 'recovery', 'timing'].includes(mode)) throw new Error('--mode flip|recovery|timing');
  const num = (x?: string) => (x !== undefined ? Number(x) : undefined);
  const dn = Object.fromEntries(Object.entries({ iterations: num(a.iterations), alphaMin: num(a['alpha-min']), lambda0: num(a.lambda0), lambda1: num(a.lambda1), sigmaL: num(a['sigma-l']), sigmaA: num(a['sigma-a']), varCorr: num(a['var-corr']), resolve: a['no-resolve'] ? false : undefined, guide: a['no-guide'] ? false : undefined, invRadius: num(a['inv-radius']), lumMinN: num(a['lum-min-n']), lumPre: num(a['lum-pre']), ...(a['dn-json'] ? JSON.parse(a['dn-json']) as Record<string, unknown> : {}) }).filter(([, v]) => v !== undefined));
  return {
    run: a.run ?? `denoise-${path.basename(a.package)}-${mode}-${stamp()}`, package: pkgUrl, seed: Number(a.seed), mode,
    frames: pkgFrames?.length ?? Number(a.frames), pkgFrames, evalFrames: a['eval-frames']?.split(',').map(Number),
    experimentalBuilderUrl: a['experimental-builder'], captureVbuffer: a['capture-vbuffer'],
    width: num(a.width), height: num(a.height), denoiser: dn, jitter: a.jitter as RenderDenoiseOptions['jitter'], denoise: !a['no-denoise'], accumulate: a.accumulate,
    debugViews: a['debug-views'] ? { ids: a['debug-views'].split(',').map(Number), frames: (a['debug-frames'] ?? '').split(',').map(Number) } : undefined,
    pan: a.pan ? (([dx, from, to]) => ({ dx, from, to }))(a.pan.split(':').map(Number)) : undefined,
    lightAnim: a['light-anim'] ? (([index, amp, move]) => ({ index, amp, move }))(a['light-anim'].split(':').map(Number)) : undefined,
    restir: a.restir ? JSON.parse(a.restir) as RenderDenoiseOptions['restir'] : undefined, renderer: a.renderer ? JSON.parse(a.renderer) as RenderDenoiseOptions['renderer'] : undefined,
    perfFlags: a['perf-flags'] ? a['perf-flags'].split(',').filter(Boolean) : undefined,
    panOsc: a['pan-osc'] ? { amp: Number(a['pan-osc']), knots: knots(a['osc-knots']!) } : undefined,
    lightOsc: a['light-osc'] ? (([index, amp]) => ({ index, amp, knots: knots(a['osc-knots']!) }))(a['light-osc'].split(':').map(Number)) : undefined, warmup: num(a.warmup), timingSubmits: num(a['timing-submits']), timingRuns: num(a['timing-runs']), chromeVersion,
  };
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const s = createNetServer();
    s.once('error', reject);
    s.listen(0, '127.0.0.1', () => { const { port } = s.address() as { port: number }; s.close(() => resolve(port)); });
  });
}

/** Vite (no HMR) + headless Chrome on the harness page (as run-batches.ts). */
async function openHarness(devFeatures: boolean): Promise<{ page: Page; chromeVersion: string; close(): Promise<void> }> {
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
    browser = await chromium.launch({ channel: 'chrome', headless: true, args: devFeatures ? ['--enable-webgpu-developer-features'] : [] });
    const chromeVersion = browser.version();
    const page = await browser.newPage();
    page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log(`[page:${m.type()}] ${m.text()}`); });
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
    await page.goto(`http://127.0.0.1:${port}/validation/harness/harness.html`);
    await page.waitForFunction(() => window.__harness !== undefined, undefined, { timeout: 60_000 });
    return { page, chromeVersion, close: async () => { await browser?.close(); await vite.close(); } };
  } catch (e) {
    await browser?.close(); await vite.close();
    throw e;
  }
}

async function runOne(page: Page, o: RenderDenoiseOptions): Promise<boolean> {
  const t0 = performance.now();
  const rep = await page.evaluate((x) => window.__harness!.renderDenoise(x), o);
  const m = rep.meta as { timing?: { meanTotalMs: number; medianTotalMs: number; passes: { name: string; ms: number }[] } };
  console.log(`${rep.ok ? 'OK  ' : 'FAIL'} ${o.run}: ${rep.files.length} files, ${((performance.now() - t0) / 1000).toFixed(1)} s${rep.ok ? '' : `; ${rep.errors.join('; ')}`}`);
  if (m.timing) console.log(`     denoiser ${m.timing.meanTotalMs.toFixed(3)} ms mean (median ${m.timing.medianTotalMs.toFixed(3)}): ${m.timing.passes.map((p) => `${p.name} ${p.ms.toFixed(3)}`).join(', ')}`);
  return rep.ok;
}

async function main(): Promise<number> {
  const a = parse();
  const jobs: string[][] = a.jobs ? JSON.parse(await readFile(path.resolve(ROOT, a.jobs), 'utf8')) as string[][] : [process.argv.slice(2)];
  const dev = !a['no-dev-features'] && jobs.some((j) => parse(j).mode === 'timing');
  const h = await openHarness(dev);
  let failures = 0;
  const perJob = a['lock-per-job'];
  const lock = async () => {
    const r = await acquireGpuLock('run-denoise');
    console.log(`     lock wait ${r.waitedMs.toFixed(0)} ms`);
    return r;
  };
  console.log(`acquiring GPU lock (${GPU_LOCK}) for ${jobs.length} job(s)${perJob ? ', one hold per job' : ''} ...`);
  const release = perJob ? undefined : await lock();
  try {
    for (const j of jobs) {
      const rj = perJob ? await lock() : undefined;
      try { if (!await runOne(h.page, optionsOf(parse(j), h.chromeVersion))) failures++; } catch (e) { failures++; console.error(e); } finally { rj?.(); }
    }
  } finally {
    release?.();
    await h.close();
  }
  console.log(failures ? `RESULT: FAIL (${failures}/${jobs.length})` : `RESULT: PASS (${jobs.length})`);
  return failures ? 1 : 0;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  main().then((c) => process.exit(c), (e: unknown) => { console.error(e); process.exit(1); });
}
