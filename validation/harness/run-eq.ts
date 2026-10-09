// perf2 WP-Q equal-quality harness driver (docs/decisions/perf2-plan.md §2 WP-Q, §3, §5). Node side; every GPU job goes
// through a harness that takes the shared GPU lock itself (run-batches.ts per chunk, run-denoise.ts --lock-per-job,
// run-perf.ts per job), so no hold exceeds the 12-min rule.
//
//   export   the run-perf.ts PERF_SCENES Sponza setup (glTF + HDRI, autoSetup camera / point light) as a scene package
//            validation/out/wpq-sponza_perf_540/ (960×540, maxBounces 3, light mode B). No GPU.
//              npx tsx validation/harness/run-eq.ts export
//   ref      the PT reference (run-batches.ts --kernel pt, 960×540, maxBounces 3, light mode B, seed 7777) in GPU-lock
//            chunks wpq-ref-<scene>-c<k> (--batch-offset), merged by eq_eval.py ref-merge into wpq-ref-<scene>/{ref.pfm,
//            ref_var.pfm, ref.json} (noise floor = mean Var/(ref² + 0.01)); batch files deleted after the merge.
//            Resumable: --from-batch continues a reference (merge then covers every chunk dir present).
//              npx tsx validation/harness/run-eq.ts ref --scene cornell --spp 256 --batches 32 --chunk 8 [--from-batch 0]
//   run      one configuration on the WP-Q protocol: scenes × sequences {static, pan, light} × seeds {1,2,3,4}, 64 frames
//            from a reset, eval frames 16,32,48,63 plus the temporal window 40:64 (run-denoise.ts flip mode, app
//            renderer: interactive, Mode B, CWBVH auto, denoiser on). Pan / light: oscillating motion with phase knots
//            0,16,32,48,63 (eval frames on the base state at peak speed: one PT reference per scene). Each run is
//            evaluated at once (eq_eval.py run → wpq-<config>-<scene>-<seq>-s<seed>/metrics.json; eq_eval.py seq adds the
//            cross-seed metrics) and its PFMs deleted (seed 1 keeps dn_f63 / raw_f63); finished groups are skipped. Then eq_eval.py summary → wpq-eq/<config>.json.
//              npx tsx validation/harness/run-eq.ts run --config baseline [--scenes cornell,sponza] [--seqs static,pan,light]
//              npx tsx validation/harness/run-eq.ts run --config my-cand --restir '{"risM":16}' [--perf-flags RS_VIS_MERGE,…]
//   perf     run-perf.ts timing of configurations, same session, ABBA order (block r runs the configs forward for even r,
//            reversed for odd r; each block runs every scene), 540p N3, → wpq-eq/perf-<tag>.json and a summary
//            wpq-eq/perf.json (per config / scene: mean frame ms over the reps, and the paired ratio to the baseline).
//              npx tsx validation/harness/run-eq.ts perf --configs baseline,d1-rr2 --scenes cornell,sponza --reps 4
//   decide   eq_eval.py decide of candidates against the baseline (+ Pareto plots) → wpq-eq/decision.json
//              npx tsx validation/harness/run-eq.ts decide --candidates d1-rr2,d2-m16
//
// Perf flags: --perf-flags A,B is an opaque list. run: passed to denoise-run.ts (renderer.restirKernel.perfFlags);
// perf: passed as PerfOptions.renderer.restirKernel.perfFlags. Both are WP-0's RestirKernelOptions.perfFlags registry
// hook (perf2-plan.md WP-0 step 1); until that registry is merged the kernel ignores the field.
import { spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { PERF_SCENES } from './run-perf.ts';
import type { PerfOptions } from './perf-run.ts';
import type { RestirSettings } from '../../src/core/render/restir/presets.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const OUT = path.join(ROOT, 'validation/out');
const PY = path.join(ROOT, 'validation/.venv/bin/python');
const EQ_EVAL = path.join(ROOT, 'validation/tools/eq_eval.py');
const EQ_DIR = path.join(OUT, 'wpq-eq');
const W = 960, H = 540;
const SEEDS = [1, 2, 3, 4];
const FRAMES = 64;
const EVAL = [16, 32, 48, 63];
const WINDOW: [number, number] = [40, 64];
const KNOTS = [0, 16, 32, 48, 63];
const REF_SEED = 7777;
/** The app renderer of the quality runs: the interactive defaults (Mode B, CWBVH from 64k triangles). Validation
 *  textures and Woop intersection stay (denoise-run.ts defaults), as in the PT reference. */
const RENDERER = { lightMode: 'B', bvhKind: 'auto' };

export const SPONZA_PKG = 'wpq-sponza_perf_540';
/** Scenes: package dir (relative to the repo), the run-perf.ts PERF_SCENES key, pan amplitude (metres along the
 *  camera's right axis) and light motion (light index, metres of world x). Peak speed = 2π·amp/16 per frame. */
export const EQ_SCENES: Record<string, { pkg: string; perf: string; pan: number; light: { index: number; amp: number } }> = {
  cornell: { pkg: 'validation/scenes/cornell_i_512', perf: 'cornell', pan: 0.03, light: { index: 0, amp: 0.03 } },
  sponza: { pkg: `validation/out/${SPONZA_PKG}`, perf: 'sponza', pan: 0.25, light: { index: 0, amp: 0.5 } },
};
export const SEQS = ['static', 'pan', 'light'] as const;
/** Configurations: ReSTIR settings over the interactive preset (§3 decisions; knobs pinned explicitly where relevant). */
export const EQ_CONFIGS: Record<string, Partial<RestirSettings>> = {
  baseline: {},
  'd1-rr2': { rrMinBounces: 2 }, 'd1-rr1': { rrMinBounces: 1 },
  'd2-m16': { risM: 16 }, 'd2-m8': { risM: 8 },
  'd4-slots2': { slots: 2 },
  'd3-nodup': { dupmap: false },
};

function sh(cmd: string, args: string[], log = true): Promise<number> {
  return new Promise((resolve) => {
    const p = spawn(cmd, args, { cwd: ROOT, stdio: log ? 'inherit' : 'ignore' });
    p.on('exit', (c, s) => resolve(c ?? (s ? 128 : 1)));
  });
}
const tsx = (script: string, args: string[]) => sh('npx', ['tsx', path.join('validation/harness', script), ...args]);
const py = (args: string[]) => sh(PY, [EQ_EVAL, ...args]);

async function pool<T>(items: T[], n: number, f: (x: T) => Promise<void>): Promise<void> {
  let k = 0;
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => { while (k < items.length) await f(items[k++]); }));
}

// ------------------------------------------------------------------------------------------------ export
async function cmdExport(): Promise<number> {
  // the harness page (Vite + headless Chrome) without the GPU lock: loading and exporting need no GPU
  const { chromium } = await import('playwright');
  const { createServer } = await import('vite');
  const net = await import('node:net');
  const port = await new Promise<number>((res) => { const s = net.createServer(); s.listen(0, '127.0.0.1', () => { const p = (s.address() as { port: number }).port; s.close(() => res(p)); }); });
  const cacheDir = path.join(ROOT, `.vite-cache-${path.basename(ROOT).replace(/^WebGPURestirPT-?/, '') || 'wt'}-harness`);
  const vite = await createServer({ root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), server: { port, strictPort: true, host: '127.0.0.1', hmr: false, watch: null }, logLevel: 'warn', cacheDir });
  await vite.listen();
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const page = await browser.newPage();
    page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log(`[page:${m.type()}] ${m.text()}`); });
    await page.goto(`http://127.0.0.1:${port}/validation/harness/harness.html`);
    await page.waitForFunction(() => window.__harness !== undefined, undefined, { timeout: 120_000 });
    const s = PERF_SCENES.sponza;
    const r = await page.evaluate((o) => window.__harness!.exportPerfScene(o), { run: SPONZA_PKG, scene: s.scene, env: s.env, width: W, height: H, maxBounces: 3, lightMode: 'B' as const, name: SPONZA_PKG });
    console.log(`exported validation/out/${SPONZA_PKG}: ${r.files.length} files, ${(r.bytes / 2 ** 20).toFixed(1)} MiB, ${r.triangles} triangles, ${r.lights} lights, sha256 ${r.sha256.slice(0, 16)}`);
    console.log(`camera ${JSON.stringify(r.camera.map((x) => +x.toFixed(4)))}`);
    return 0;
  } finally {
    await browser.close(); await vite.close();
  }
}

// ------------------------------------------------------------------------------------------------ ref
async function cmdRef(a: Record<string, string | boolean | undefined>): Promise<number> {
  const sc = EQ_SCENES[String(a.scene)];
  if (!sc) throw new Error(`--scene ${Object.keys(EQ_SCENES).join('|')}`);
  const spp = Number(a.spp ?? 256), batches = Number(a.batches ?? 32), chunk = Number(a.chunk ?? 8), from = Number(a['from-batch'] ?? 0);
  const base = `wpq-ref-${a.scene}`;
  for (let b = from; b < batches; b += chunk) {
    const n = Math.min(chunk, batches - b), run = `${base}-c${String(b).padStart(3, '0')}`;
    const dir = path.join(OUT, run);
    if (existsSync(path.join(dir, 'meta.json')) && readdirSync(dir).filter((f) => f.startsWith('batch_')).length === n) { console.log(`skip ${run} (complete)`); continue; }
    const c = await tsx('run-batches.ts', ['--package', sc.pkg, '--kernel', 'pt', '--spp', String(spp), '--batches', String(n), '--batch-offset', String(b),
      '--width', String(W), '--height', String(H), '--max-bounces', '3', '--light-mode', 'B', '--seed', String(REF_SEED), '--run', run]);
    if (c) return c;
  }
  if (a['no-merge']) return 0;
  const dirs = readdirSync(OUT).filter((d) => d.startsWith(`${base}-c`)).sort().map((d) => path.join(OUT, d));
  return py(['ref-merge', '--dirs', dirs.join(','), '--out-dir', path.join(OUT, base), ...(a.keep ? [] : ['--delete-batches'])]);
}

// ------------------------------------------------------------------------------------------------ run
function jobArgs(config: string, scene: string, seq: string, seed: number, restir: Partial<RestirSettings>, perfFlags?: string): string[] {
  const sc = EQ_SCENES[scene];
  const evalFrames = [...new Set([...EVAL, ...Array.from({ length: WINDOW[1] - WINDOW[0] }, (_, i) => WINDOW[0] + i)])].sort((x, y) => x - y);
  return [
    '--package', sc.pkg, '--mode', 'flip', '--frames', String(FRAMES), '--eval-frames', evalFrames.join(','), '--seed', String(seed),
    '--run', runName(config, scene, seq, seed), '--width', String(W), '--height', String(H), '--renderer', JSON.stringify(RENDERER),
    '--restir', JSON.stringify(restir), '--osc-knots', KNOTS.join(','),
    ...(seq === 'pan' ? ['--pan-osc', String(sc.pan)] : seq === 'light' ? ['--light-osc', `${sc.light.index}:${sc.light.amp}`] : []),
    ...(perfFlags ? ['--perf-flags', perfFlags] : []),
  ];
}
const runName = (config: string, scene: string, seq: string, seed: number) => `wpq-${config}-${scene}-${seq}-s${seed}`;

async function cmdRun(a: Record<string, string | boolean | undefined>): Promise<number> {
  const config = String(a.config ?? 'baseline');
  if (!/^[\w.+-]+$/.test(config)) throw new Error('--config: [\\w.+-]+');
  const restir = a.restir ? JSON.parse(String(a.restir)) as Partial<RestirSettings> : EQ_CONFIGS[config];
  if (!restir) throw new Error(`--config ${config}: not in EQ_CONFIGS (${Object.keys(EQ_CONFIGS).join(', ')}); give --restir JSON`);
  const scenes = String(a.scenes ?? 'cornell,sponza').split(','), seqs = String(a.seqs ?? SEQS.join(',')).split(','), seeds = a.seeds ? String(a.seeds).split(',').map(Number) : SEEDS;
  const perfFlags = a['perf-flags'] ? String(a['perf-flags']) : undefined;
  mkdirSync(EQ_DIR, { recursive: true });
  let fails = 0;
  for (const scene of scenes) {
    const refDir = path.join(OUT, `wpq-ref-${scene}`);
    if (!existsSync(path.join(refDir, 'ref.pfm'))) throw new Error(`${refDir}/ref.pfm missing: run 'run-eq.ts ref --scene ${scene}' first`);
    for (const seq of seqs) {
      const todo = seeds.filter((s) => !existsSync(path.join(OUT, runName(config, scene, seq, s), 'metrics.json')));
      if (!todo.length) { console.log(`skip ${config} ${scene} ${seq} (complete)`); continue; }
      const jobsFile = path.join(EQ_DIR, `jobs-${config}-${scene}-${seq}.json`);
      writeFileSync(jobsFile, JSON.stringify(todo.map((s) => jobArgs(config, scene, seq, s, restir, perfFlags))));
      const c = await tsx('run-denoise.ts', ['--jobs', path.relative(ROOT, jobsFile), '--lock-per-job']);
      if (c) { console.log(`run-denoise exit ${c} (${config} ${scene} ${seq})`); fails++; }
      await pool(todo, 4, async (s) => {
        const d = path.join(OUT, runName(config, scene, seq, s));
        if (!existsSync(path.join(d, 'meta.json'))) { fails++; return; }
        const m = JSON.parse(readFileSync(path.join(d, 'meta.json'), 'utf8')) as { ok: boolean; errors: string[] };
        if (!m.ok) console.log(`WARN ${d}: ${m.errors.join('; ')}`);
        const rc = await py(['run', '--ref-dir', refDir, '--dir', d, '--eval', EVAL.join(','), '--window', WINDOW.join(':'), '--out', path.join(d, 'metrics.json'),
          '--label', JSON.stringify({ config, scene, seq, seed: s, restir, perfFlags: perfFlags ?? null })]);
        if (rc) fails++;
      });
      // cross-seed metrics (motion-compensated temporal std, spikes of moving sequences) need every seed's frames: a
      // (scene, sequence) is rendered in one go, so a partial resume re-renders the whole group
      const dirs = seeds.map((s) => path.join(OUT, runName(config, scene, seq, s)));
      if (todo.length !== seeds.length) throw new Error(`${config} ${scene} ${seq}: seeds ${seeds.filter((s) => !todo.includes(s)).join(',')} finished earlier without their frames; delete their metrics.json to re-render the group`);
      if (await py(['seq', '--ref-dir', refDir, '--dirs', dirs.join(','), '--window', WINDOW.join(':'), ...(seq === 'static' ? [] : ['--motion']), '--delete', '--keep', 'dn_f63.pfm,raw_f63.pfm'])) fails++;
    }
  }
  const rc = await py(['summary', '--config', config, '--metrics', path.join(OUT, `wpq-${config}-*-s*`, 'metrics.json'), '--out', path.join(EQ_DIR, `${config}.json`)]);
  console.log(fails ? `RESULT: FAIL (${fails})` : 'RESULT: PASS');
  return fails || rc;
}

// ------------------------------------------------------------------------------------------------ perf
interface PerfFile { tag: string; reports: { ok: boolean; label: string; frame: { meanMs: number } }[] }

function summarizePerf(files: string[], baseline: string): Record<string, unknown> {
  const rows: { cfg: string; scene: string; rep: number; ms: number }[] = [];
  for (const f of files) {
    const j = JSON.parse(readFileSync(f, 'utf8')) as PerfFile;
    for (const r of j.reports) {
      const m = /^(.+)\|(\w+)\|r(\d+)$/.exec(r.label);
      if (m && r.ok) rows.push({ cfg: m[1], scene: m[2], rep: Number(m[3]) + 1000 * files.indexOf(f), ms: r.frame.meanMs });
    }
  }
  const configs: Record<string, Record<string, { frame_ms: number; reps: number[]; ratio_vs_baseline?: number; ratio_reps?: number[] }>> = {};
  for (const r of rows) {
    const c = (configs[r.cfg] ??= {});
    const s = (c[r.scene] ??= { frame_ms: 0, reps: [] });
    s.reps.push(r.ms);
  }
  for (const c of Object.values(configs)) for (const s of Object.values(c)) s.frame_ms = s.reps.reduce((x, y) => x + y, 0) / s.reps.length;
  // paired ratios: candidate rep k vs the baseline of the same rep (same block)
  for (const [cfg, c] of Object.entries(configs)) {
    for (const [scene, s] of Object.entries(c)) {
      const mine = rows.filter((r) => r.cfg === cfg && r.scene === scene);
      const ratios = mine.map((r) => { const b = rows.find((q) => q.cfg === baseline && q.scene === scene && q.rep === r.rep); return b ? r.ms / b.ms : NaN; }).filter(Number.isFinite);
      s.ratio_reps = ratios;
      s.ratio_vs_baseline = ratios.length ? ratios.reduce((x, y) => x + y, 0) / ratios.length : undefined;
    }
  }
  return { baseline, files: files.map((f) => path.relative(ROOT, f)), configs };
}

async function cmdPerf(a: Record<string, string | boolean | undefined>): Promise<number> {
  const cfgs = String(a.configs ?? Object.keys(EQ_CONFIGS).join(',')).split(',');
  const scenes = String(a.scenes ?? 'cornell,sponza').split(','), reps = Number(a.reps ?? 4);
  const perfFlags = a['perf-flags'] ? String(a['perf-flags']).split(',') : undefined;
  const extra = a.restir ? JSON.parse(String(a.restir)) as Partial<RestirSettings> : undefined;
  const jobs: PerfOptions[] = [];
  for (let r = 0; r < reps; r++) {
    const order = r % 2 === 0 ? cfgs : [...cfgs].reverse();
    for (const cfg of order) {
      const restir = extra && cfg !== 'baseline' ? extra : EQ_CONFIGS[cfg];
      if (!restir) throw new Error(`unknown config ${cfg}`);
      for (const scene of scenes) {
        const s = PERF_SCENES[EQ_SCENES[scene].perf];
        // frame time only (the per-pass split and isolated latency are not used by the rule)
        jobs.push({ scene: s.scene, env: s.env, width: W, height: H, restir, label: `${cfg}|${scene}|r${r}`, latencyFrames: 0, passFrames: 0,
          ...(perfFlags && cfg !== 'baseline' ? { renderer: { restirKernel: { perfFlags } } as PerfOptions['renderer'] } : {}),
          ...(a.frames ? { frames: Number(a.frames) } : {}) });
      }
    }
  }
  mkdirSync(EQ_DIR, { recursive: true });
  const tag = String(a.tag ?? `perf-${new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-')}`);
  const jobsFile = path.join(EQ_DIR, `${tag}-jobs.json`);
  writeFileSync(jobsFile, JSON.stringify(jobs));
  const c = await tsx('run-perf.ts', ['--jobs', path.relative(ROOT, jobsFile), '--out', path.relative(ROOT, EQ_DIR), '--tag', tag]);
  const files = readdirSync(EQ_DIR).filter((f) => /^perf-.*\.json$/.test(f) && !f.endsWith('-jobs.json') && f !== 'perf.json').map((f) => path.join(EQ_DIR, f));
  const sum = summarizePerf(String(a.merge ?? '') === 'none' ? [path.join(EQ_DIR, `${tag}.json`)] : files, 'baseline');
  writeFileSync(path.join(EQ_DIR, 'perf.json'), `${JSON.stringify(sum, null, 1)}\n`);
  for (const [cfg, cs] of Object.entries(sum.configs as Record<string, Record<string, { frame_ms: number; ratio_vs_baseline?: number }>>)) {
    console.log(`${cfg.padStart(12)}: ${Object.entries(cs).map(([s, v]) => `${s} ${v.frame_ms.toFixed(2)} ms (×${v.ratio_vs_baseline?.toFixed(3) ?? '-'})`).join(', ')}`);
  }
  return c;
}

// ------------------------------------------------------------------------------------------------ decide
async function cmdDecide(a: Record<string, string | boolean | undefined>): Promise<number> {
  const cands = String(a.candidates ?? '').split(',').filter(Boolean).map((c) => path.join(EQ_DIR, `${c}.json`));
  return py(['decide', '--baseline', path.join(EQ_DIR, `${a.baseline ?? 'baseline'}.json`), '--candidates', cands.join(','),
    '--perf', path.join(EQ_DIR, String(a.perf ?? 'perf.json')), '--out', path.join(EQ_DIR, String(a.out ?? 'decision.json')), '--plot-dir', EQ_DIR]);
}

async function main(): Promise<number> {
  const cmd = process.argv[2];
  const { values: a } = parseArgs({ args: process.argv.slice(3), options: {
    scene: { type: 'string' }, scenes: { type: 'string' }, spp: { type: 'string' }, batches: { type: 'string' }, chunk: { type: 'string' }, 'from-batch': { type: 'string' },
    'no-merge': { type: 'boolean' }, keep: { type: 'boolean' }, config: { type: 'string' }, configs: { type: 'string' }, restir: { type: 'string' }, seqs: { type: 'string' },
    seeds: { type: 'string' }, 'perf-flags': { type: 'string' }, reps: { type: 'string' }, tag: { type: 'string' }, frames: { type: 'string' }, merge: { type: 'string' },
    candidates: { type: 'string' }, baseline: { type: 'string' }, perf: { type: 'string' }, out: { type: 'string' },
  } });
  switch (cmd) {
    case 'export': return cmdExport();
    case 'ref': return cmdRef(a);
    case 'run': return cmdRun(a);
    case 'perf': return cmdPerf(a);
    case 'decide': return cmdDecide(a);
    default: console.error('usage: run-eq.ts export | ref | run | perf | decide (see the header)'); return 2;
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1])) {
  main().then((c) => process.exit(c), (e: unknown) => { console.error(e); process.exit(1); });
}
