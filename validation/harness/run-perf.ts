// M8 performance runner (docs/decisions/m8-perf.md §1): Vite (no HMR) + headless Chrome with
// --enable-webgpu-developer-features (unquantised timestamps for the denoiser re-run) on the harness page;
// window.__harness.renderPerf per job, each job under its own GPU-lock hold (a job takes ~0.5–2 min).
//   npx tsx validation/harness/run-perf.ts --suite baseline [--res 540p,720p] [--only cornell,sponza] [--out DIR] [--tag TAG]
//   npx tsx validation/harness/run-perf.ts --jobs FILE   (a JSON array of PerfOptions)
// perf2 (docs/decisions/perf2-api.md):
//   --kernel-flags A,B=2     perf flags for every job (PerfOptions.perfFlags; default the app's release set)
//   --decisions              baseline suite + one labelled 540p N3 row per user-decision configuration (DECISION_CONFIGS)
//   --abba FLAGS [--blocks 2] same-session ABBA per scene / resolution: (base, FLAGS, FLAGS, base) × 2 per block, labels
//                            '<scene>@<res> N3 base #i' / '<scene>@<res> N3 <FLAGS> #i' (validation/tools/perf/abba.py)
//   --abba-restir app-defaults|JSON   ABBA variant = the pinned job + these ReSTIR knobs (with or without --abba FLAGS);
//                            'app-defaults' = INTERACTIVE_APP_DEFAULTS (role 'appDefaults'), JSON = role 'restir'
// Every baseline / ABBA row pins the interactive knobs (INTERACTIVE_PINNED + the scene's maxBounces), so app-default
// decisions never move them; decision rows are the only rows that change knobs.
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
import { INTERACTIVE_PINNED, type RestirSettings } from '../../src/core/render/restir/presets.ts';
import { normalizePerfFlags, perfFlagsKey, unlandedFlags, type PerfFlags } from '../../src/core/render/restir/perf-flags.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const HDRI = '/validation/assets/downloaded/hdri/kloofendal_48d_partly_cloudy_puresky_1k.hdr';
/** perf2 WP-0: maxBounces is pinned per scene (= what perf-run derived for each: the package's render.maxBounces, 3 for
 *  the glTF; crossings has glass). */
export const PERF_SCENES: Record<string, Pick<PerfOptions, 'scene' | 'env' | 'lightAnim'> & { maxBounces: number }> = {
  cornell: { scene: '/validation/scenes/cornell_i_512/', lightAnim: { index: 0, amp: 0.05 }, maxBounces: 3 },
  sponza: { scene: '/validation/assets/downloaded/sponza/Sponza.gltf', env: HDRI, lightAnim: { index: 0, amp: 1.0 }, maxBounces: 3 },
  crossings: { scene: '/validation/out/m6/scenes/m6_crossings_B_256/', lightAnim: { index: 0, amp: 0.05 }, maxBounces: 4 },
  nm_smooth: { scene: '/validation/out/m7/scenes/m7_nm_smooth_256/', lightAnim: { index: 0, amp: 0.05 }, maxBounces: 3 },
};

/** perf2 WP-0: a job of `scene` with the interactive knobs pinned (INTERACTIVE_PINNED, the scene's maxBounces) and the
 *  given ReSTIR overrides on top. */
export function pinnedJob(sceneKey: string, restir: Partial<RestirSettings> = {}): Pick<PerfOptions, 'scene' | 'env' | 'restir' | 'renderer'> {
  const s = PERF_SCENES[sceneKey];
  if (!s) throw new Error(`unknown scene ${sceneKey}`);
  return { scene: s.scene, ...(s.env ? { env: s.env } : {}), restir: { ...INTERACTIVE_PINNED, ...restir }, renderer: { maxBounces: s.maxBounces } };
}

/** `extra` over a job: top-level keys replace, `restir` / `renderer` merge (the pins survive `--extra`). */
export function withExtra(j: PerfOptions, e: Partial<PerfOptions>): PerfOptions {
  return { ...j, ...e, restir: { ...j.restir, ...e.restir }, renderer: { ...j.renderer, ...e.renderer } };
}

/** perf2 user decisions (perf2-plan.md §3 / §5) as separately labelled rows: knob changes over the pinned baseline, or
 *  a perf flag (rows whose flags are reserved names without WGSL yet are skipped with a note). */
export const DECISION_CONFIGS: { id: string; restir?: Partial<RestirSettings>; perfFlags?: PerfFlags }[] = [
  { id: 'D1 rrMin2', restir: { rrMinBounces: 2 } },
  { id: 'D2 risM16', restir: { risM: 16 } },
  { id: 'D3 dupmapOff', restir: { dupmap: false } },
  { id: 'D4 slots2', restir: { slots: 2 } },
  { id: 'D6 halfRate', perfFlags: { RS_HALF_RATE: 1 } },
  // the shipped app defaults (INTERACTIVE_APP_DEFAULTS in renderer.ts: D1 + D3; tests/restir/perf-flags.test.ts checks equality)
  { id: 'D1+D3 appDefaults', restir: { rrMinBounces: 2, dupmap: false } },
];
/** `--abba-restir app-defaults`: the knob changes of INTERACTIVE_APP_DEFAULTS (renderer.ts cannot be imported in node). */
export const APP_DEFAULTS_RESTIR: Partial<RestirSettings> = { rrMinBounces: 2, dupmap: false };
export const RES: Record<string, [number, number]> = { '540p': [960, 540], '720p': [1280, 720] };

/** The M8 target configurations (PLAN §5 M8 exit): N=3 static at 540p and 720p, N=1 + moving lights at 540p. */
export function suiteJobs(name: string, scenes: string[], res: string[], extra: Partial<PerfOptions> = {}, o: { decisions?: boolean } = {}): PerfOptions[] {
  const jobs: PerfOptions[] = [];
  for (const sc of scenes) {
    const s = PERF_SCENES[sc];
    if (!s) throw new Error(`unknown scene ${sc}`);
    for (const r of res) {
      const [w, h] = RES[r];
      jobs.push(withExtra({ ...pinnedJob(sc), width: w, height: h, label: `${sc}@${r} N3` }, extra));
      if (name === 'baseline' && r === '540p') jobs.push(withExtra({ ...pinnedJob(sc, { slots: 1 }), width: w, height: h, lightAnim: s.lightAnim, label: `${sc}@${r} N1+moving` }, extra));
      if (o.decisions && r === '540p') {
        for (const d of DECISION_CONFIGS) {
          const un = unlandedFlags(d.perfFlags);
          if (un.length) { console.log(`[run-perf] skipping decision row '${d.id}': ${un.join(', ')} not landed yet (reserved name only)`); continue; }
          const j = withExtra({ ...pinnedJob(sc), width: w, height: h, label: `${sc}@${r} N3 [${d.id}]` }, extra);
          jobs.push({ ...j, restir: { ...j.restir, ...d.restir }, ...(d.perfFlags ? { perfFlags: { ...normalizePerfFlags(extra.perfFlags), ...d.perfFlags } } : {}) });
        }
      }
    }
  }
  return jobs;
}

/** perf2 V-PERF: same-session ABBA jobs (base = the pinned N3 job with `extra`; variant = + `flags`), per scene and
 *  resolution `blocks` × (base, variant, variant, base, base, variant, variant, base). */
export function abbaJobs(flags: string, scenes: string[], res: string[], blocks: number, extra: Partial<PerfOptions> = {}, restir?: { role: string; settings: Partial<RestirSettings> }): PerfOptions[] {
  const vf = normalizePerfFlags(flags);
  const role = [perfFlagsKey(vf), restir?.role].filter(Boolean).join('+');
  if (!role) throw new Error('--abba: no flags and no --abba-restir');
  const jobs: PerfOptions[] = [];
  let i = 0;
  for (let b = 0; b < blocks; b++) for (const sc of scenes) for (const r of res) {
    const [w, h] = RES[r];
    const base: PerfOptions = withExtra({ ...pinnedJob(sc), width: w, height: h }, extra);
    const variant: PerfOptions = { ...base, restir: { ...base.restir, ...restir?.settings }, perfFlags: { ...normalizePerfFlags(extra.perfFlags), ...vf } };
    for (const v of [false, true, true, false, false, true, true, false]) jobs.push({ ...(v ? variant : base), label: `${sc}@${r} N3 ${v ? role : 'base'} #${i++}` });
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
    'kernel-flags': { type: 'string' }, decisions: { type: 'boolean', default: false }, abba: { type: 'string' }, blocks: { type: 'string', default: '2' },
    'abba-restir': { type: 'string' },
  } });
  const abbaRestir = a['abba-restir'] === undefined ? undefined : a['abba-restir'] === 'app-defaults'
    ? { role: 'appDefaults', settings: APP_DEFAULTS_RESTIR } : { role: 'restir', settings: JSON.parse(a['abba-restir']) as Partial<RestirSettings> };
  const extra = a.extra ? JSON.parse(a.extra) as Partial<PerfOptions> : {};
  if (a.frames) extra.frames = Number(a.frames);
  if (a['kernel-flags'] !== undefined) extra.perfFlags = normalizePerfFlags(a['kernel-flags']);
  const scenes = (a.only ?? Object.keys(PERF_SCENES).join(',')).split(',');
  const jobs: PerfOptions[] = a.jobs ? (JSON.parse(readFileSync(path.resolve(ROOT, a.jobs), 'utf8')) as PerfOptions[]).map((j) => ({ ...j, ...extra }))
    : a.abba !== undefined || abbaRestir ? abbaJobs(a.abba ?? '', scenes, a.res.split(','), Number(a.blocks), extra, abbaRestir)
      : suiteJobs(a.suite ?? 'baseline', scenes, a.res.split(','), extra, { decisions: a.decisions });
  const un = [...new Set(jobs.flatMap((j) => unlandedFlags(j.perfFlags)))];
  if (un.length) console.log(`[run-perf] note: ${un.join(', ')} are reserved names without WGSL yet (no effect)`);
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
