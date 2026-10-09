// M8 milestone gate (docs/decisions/m8-perf.md §13; PLAN §5 M8):
// `npm run validate -- --milestone M8 [--part core|stageB|perf] [--only id,pkg]`.
//   core    Gate 0: typecheck; cpu lane (U-M7-BITS: every validation pipeline composes to the M6 / M7 text with the M8
//           code, cwbvh encoder, dynamic resolution, env-compact static check + every earlier CPU test); python tests;
//           Chrome: U-M8-BITS / U-M8-BITS (SoA) / U-M8-PTBITS / U-M8-MODEB / P-4 switching (m8-bits), T12 on BVH2 and
//           CWBVH incl. the ray-by-ray hit equivalence (bvh), the denoiser suite (U-DN-3b / 3c: tiled à-trous), the env
//           suites (ENV-U7c: compact env formats bit-identical; ENV-F report-only), the M7 normal-map suite and the
//           M5.5 / M7 app smokes (the app: CWBVH auto, P-4, SoA, bicubic, dynamic resolution compile and render finite).
//   stageB  CWBVH end to end (Stage B, δ 0.2 % / 1 %): ReSTIR offline-m6 on the CWBVH (Woop) vs our PT on BVH2 on the
//           normal-mapped smooth scenes (Mode A, Mode B, HDRI); gate-m7's sequential-unit machinery (pilot sizing ×1.25,
//           one disjoint-seed re-run per failed unit); T16 also requires meta.config.bvh = 'cwbvh'.
//   perf    the PLAN §5 M8 targets (540p N=3, 540p N=1 + moving lights, 720p N=3; ≥ 32-frame pipelined averages,
//           Chrome --enable-webgpu-developer-features) on Cornell, Sponza + HDRI, m6_crossings, m7_nm_smooth, with the
//           pre-M8-equivalent configuration (BVH2, AoS reservoirs, no P-4, texture-path à-trous) interleaved in the same
//           session for the before / after; recorded, misses are documented deviations (not gating).
// Statistics: compare.py through gate-m7; δ is never loosened, nothing is clamped.
// Output: validation/out/m8-gate-<part>-<time>/ (summary.json/.md, logs/, compare/, pt/, restir/, perf.json).
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { codeHashes } from './gate-m4.ts';
import { M7_PKGS, seqUnit, type SeqUnit } from './gate-m7.ts';
import { withGpuLockSync } from './gpu-lock.ts';
import { pinnedJob, withExtra } from './run-perf.ts';
import type { PerfOptions } from './perf-run.ts';
import type { RestirSettings } from '../../src/core/render/restir/presets.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PY = path.join(ROOT, 'validation/.venv/bin/python');
const TIMEOUT_MS = 24 * 3600_000;

export type Part = 'core' | 'stageB' | 'perf';
export const PARTS: Part[] = ['core', 'stageB', 'perf'];

const U = (tag: string, pkg: string, label: string, lightMode: 'A' | 'B'): SeqUnit =>
  ({ id: `${pkg}@M8-cwbvh-${tag}`, part: 'stageB', rung: 'M8', pkg, label, preset: 'offline-m6', lightMode, extraArgs: ['--bvh', 'cwbvh'] });
export const STAGE_B_UNITS: SeqUnit[] = [
  U('offline', 'm7_nm_smooth_256', 'CWBVH: smooth + normal maps, offline-m6, Mode A', 'A'),
  U('offline-B', 'm7_nm_smooth_B_256', 'CWBVH: smooth + normal maps, offline-m6, light mode B', 'B'),
  U('offline-env', 'm7_nm_env_256', 'CWBVH: normal-mapped spheres under the overcast HDRI, offline-m6', 'A'),
];
export const nUnits = (): number => 4 * STAGE_B_UNITS.length;

const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const safe = (s: string) => s.replace(/[^\w.@-]+/g, '_');
const tryJson = (p: string): Record<string, any> | undefined => { try { return JSON.parse(readFileSync(path.resolve(ROOT, p), 'utf8')); } catch { return undefined; } };
type Add = (name: string, ok: boolean, seconds: number, data?: unknown, detail?: string) => void;
type Rec = (name: string, ok: boolean, detail?: string) => void;
type RunStep = (name: string, cmd: string, argv: string[], echo?: (l: string) => boolean, env?: Record<string, string>) => { code: number; out: string };

function sh(cmd: string, argv: string[], echo: (l: string) => boolean = () => true, env?: Record<string, string>): { code: number; out: string; seconds: number } {
  const t0 = performance.now();
  const r = spawnSync(cmd, argv, { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 28, timeout: TIMEOUT_MS, ...(env ? { env: { ...process.env, ...env } } : {}) });
  const out = `${r.stdout ?? ''}${r.stderr ?? ''}`;
  for (const l of out.split('\n')) if (l && echo(l)) console.log(`  ${l}`);
  return { code: r.status ?? 1, out, seconds: (performance.now() - t0) / 1000 };
}

// ------------------------------------------------------------------------------------------------ Gate 0

export const M8_GPU_SUITES: { file: string; what: string; timeout?: number }[] = [
  { file: 'm8-bits', what: 'U-M8-BITS (AoS / SoA = the pre-M8 hashes), U-M8-PTBITS, U-M8-MODEB, P-4 variant switching' },
  { file: 'bvh', what: 'T12 on BVH2 and CWBVH (MT, Woop): f64 reference, hit equivalence, watertight, self-hits, alpha, seams, throughput', timeout: 900_000 },
  { file: 'denoiser', what: 'U-DN-1…7 incl. U-DN-3b (N = 4) and U-DN-3c (tiled step-1 à-trous bit-identical)' },
  { file: 'env', what: 'ENV-U2 / U7 / U7b / U7c (compact env formats bit-identical in envRadiance / envBackground)' },
  { file: 'env-format', what: 'ENV-F hardware-filter precision (report only)' },
  { file: 'env-sampling', what: 'ENV-U3 / U4 / U6 (env sampling with the validation env formats)' },
  { file: 'normal-map', what: 'U-NM-1…3 (M7; the normal-mapped scenes of the stageB part)' },
];

function gate0(dir: string, runId: string, add: Add, runStep: RunStep): void {
  runStep('typecheck', 'npx', ['tsc', '--noEmit', '-p', '.']);
  runStep('vitest cpu (U-M7-BITS validation text, cwbvh, dynres, env-compact + every earlier CPU test)', 'npx', ['vitest', 'run', '--project', 'cpu'], (l) => /Test Files|Tests |FAIL|✗|×/.test(l));
  runStep('python tests', PY, ['-m', 'pytest', 'validation/tools/tests', '-q'], (l) => /passed|failed|error/i.test(l));
  const vit = (file: string, extra: string[]) => ['vitest', 'run', '--project', 'chrome', '--reporter=verbose', `validation/gpu-tests/${file}.gpu.test.ts`, ...extra];
  const echo = (l: string) => /Tests |FAIL|✗|×|AssertionError|U-M8|ENV-U7c|equiv/.test(l);
  for (const s of M8_GPU_SUITES) {
    withGpuLockSync(`gate-m8-${s.file}`, () => { runStep(`${s.file} (chrome): ${s.what}`, 'npx', vit(s.file, s.timeout ? ['--testTimeout', String(s.timeout)] : []), echo); });
  }
  // Take the GPU lock themselves.
  runStep('M5.5 app smoke (Cornell + editor lights, Sponza + HDRI: ReSTIR-interactive with the denoiser; Q3)', 'npx', ['tsx', 'validation/harness/m55-app-smoke.ts', '--run', `${runId}-m55-smoke`], (l) => /^(PASS|FAIL)\s/.test(l));
  runStep('M7 app smoke (Sponza / USD textured / Cornell: PT + ReSTIR finite, views 320-327)', 'npx', ['tsx', 'validation/harness/m7-app-smoke.ts', '--run', `${runId}-m7-smoke`], (l) => /^(PASS|FAIL)\s/.test(l));
}

// ------------------------------------------------------------------------------------------------ perf

/** The pre-M8-equivalent interactive configuration (every M8 interactive option off) for the same-session before / after. */
export const PRE_M8: Record<string, unknown> = { renderer: { bvhKind: 'bvh2', restirKernel: { resLayout: 'aos', modeBNeedsAreaLights: false } }, denoiser: { atrousTile: false } };

function perf(dir: string, add: Add): Record<string, any> {
  const scenes: Record<string, { scene: string; env?: string; lightAnim: { index: number; amp: number } }> = {
    cornell: { scene: '/validation/scenes/cornell_i_512/', lightAnim: { index: 0, amp: 0.05 } },
    sponza: { scene: '/validation/assets/downloaded/sponza/Sponza.gltf', env: '/validation/assets/downloaded/hdri/kloofendal_48d_partly_cloudy_puresky_1k.hdr', lightAnim: { index: 0, amp: 1.0 } },
    crossings: { scene: '/validation/out/m6/scenes/m6_crossings_B_256/', lightAnim: { index: 0, amp: 0.05 } },
    nm_smooth: { scene: '/validation/out/m7/scenes/m7_nm_smooth_256/', lightAnim: { index: 0, amp: 0.05 } },
  };
  if (!existsSync(path.join(ROOT, 'validation/out/m6/scenes/m6_crossings_B_256/scene.json'))) sh('npx', ['tsx', 'validation/scenes/make-m6.ts', '--only', 'm6_crossings_B_256'], () => false);
  // perf2 WP-0: every job pins the interactive knobs (run-perf.ts pinnedJob: INTERACTIVE_PINNED + the scene's maxBounces)
  const cfgs: { id: string; w: number; h: number; restir?: Partial<RestirSettings>; target: number }[] = [
    { id: '540p N3', w: 960, h: 540, target: 35 },
    { id: '540p N1+moving', w: 960, h: 540, restir: { slots: 1 }, target: 32 },
    { id: '720p N3', w: 1280, h: 720, target: 51 },
    // the dynamic-resolution controller's 0.625 level of 540p (m8-perf.md §9), its 33 ms target
    { id: '336p N3 (dyn-res 0.625)', w: 600, h: 336, target: 33 },
  ];
  const jobs: PerfOptions[] = [];
  // ABBA order per (scene, config): a sustained run slows down over time (thermal / drift, m8-perf.md §2), so each side
  // is measured once early and once late and the two are averaged
  for (const [name, s] of Object.entries(scenes)) for (const c of cfgs) for (const [tag, base, rep] of [['M8', {}, 0], ['pre-M8', PRE_M8, 0], ['pre-M8', PRE_M8, 1], ['M8', {}, 1]] as const) {
    const moving = c.id.includes('moving');
    jobs.push(withExtra({ ...pinnedJob(name, c.restir), width: c.w, height: c.h, frames: 64, label: `${name} ${c.id} ${tag} r${rep}`,
      ...(moving ? { lightAnim: s.lightAnim } : {}) }, base as Partial<PerfOptions>));
  }
  const jf = path.join(dir, 'perf-jobs.json');
  writeFileSync(path.join(ROOT, jf), JSON.stringify(jobs, null, 1));
  const r = sh('npx', ['tsx', 'validation/harness/run-perf.ts', '--jobs', jf, '--out', dir, '--tag', 'perf'], (l) => /^(OK|FAIL)\s/.test(l));
  const rep = tryJson(path.join(dir, 'perf.json'));
  const rows: Record<string, any>[] = [];
  for (const [name] of Object.entries(scenes)) for (const c of cfgs) {
    const get = (tag: string) => [0, 1].map((r) => rep?.reports?.find((x: any) => x.label === `${name} ${c.id} ${tag} r${r}`));
    const avg = (xs: any[], f: (x: any) => number) => (xs.every(Boolean) ? xs.reduce((t, x) => t + f(x), 0) / xs.length : undefined);
    const a = get('M8'), b = get('pre-M8');
    const m8 = avg(a, (x) => x.frame.meanMs), pre = avg(b, (x) => x.frame.meanMs);
    rows.push({ scene: name, config: c.id, target_ms: c.target, m8_ms: m8, pre_m8_ms: pre, m8_runs: a.map((x) => x?.frame?.meanMs), pre_m8_runs: b.map((x) => x?.frame?.meanMs),
      met: m8 !== undefined ? m8 <= c.target : undefined, denoiser_ms: avg(a, (x) => x.denoiser?.totalMs ?? NaN), pre_denoiser_ms: avg(b, (x) => x.denoiser?.totalMs ?? NaN),
      passes_m8: a[0]?.passes, passes_pre: b[0]?.passes, ok: [...a, ...b].every((x) => !!x?.ok) });
  }
  add('M8 perf (recorded, not gating): frame ms M8 vs the pre-M8-equivalent configuration, same session', r.code === 0 && rows.every((x) => x.ok), r.seconds, { rows },
    rows.map((x) => `${x.scene} ${x.config}: ${x.m8_ms?.toFixed(1)} (pre ${x.pre_m8_ms?.toFixed(1)}) ${x.met ? '≤' : '>'} ${x.target_ms}`).join(' | '));
  return { rows };
}

// ------------------------------------------------------------------------------------------------ the gate

export interface M8Options { part?: Part; only?: Set<string> }

export function milestoneM8(record: Rec, o: M8Options = {}): boolean {
  const t0 = performance.now();
  const parts = o.part ? [o.part] : PARTS;
  const runId = `m8-gate-${o.part ?? 'all'}-${stamp()}`;
  const dir = path.join('validation/out', runId);
  mkdirSync(path.join(ROOT, dir), { recursive: true });
  const steps: { name: string; ok: boolean; seconds: number; data?: unknown; detail?: string }[] = [];
  const add: Add = (name, ok, seconds, data, detail) => {
    steps.push({ name, ok, seconds: Math.round(seconds * 10) / 10, data, detail });
    record(name, ok, `${detail ? `${detail}, ` : ''}${seconds.toFixed(1)} s`);
  };
  const runStep: RunStep = (name, cmd, argv, echo, env) => {
    console.log(`\n--- ${name}: ${cmd} ${argv.join(' ')}`);
    const r = sh(cmd, argv, echo, env);
    const log = path.join(dir, 'logs', `${safe(name).slice(0, 80)}.log`);
    mkdirSync(path.join(ROOT, dir, 'logs'), { recursive: true });
    writeFileSync(path.join(ROOT, log), r.out);
    const noTests = argv[0] === 'vitest' && !/\d+ passed/.test(r.out);
    add(name, r.code === 0 && !noTests, r.seconds, { log }, `exit ${r.code}${noTests ? ', no test ran' : ''}`);
    return r;
  };
  const hashes = codeHashes();
  console.log(`M8 gate ${runId}: parts ${parts.join(',')}, PT code ${hashes.pt.slice(0, 12)}, ReSTIR ${hashes.restir.slice(0, 12)}`);
  const missingPkgs = M7_PKGS.filter((p) => !existsSync(path.join(ROOT, 'validation/out/m7/scenes', p, 'scene.json')));
  if (missingPkgs.length) sh('npx', ['tsx', 'validation/scenes/make-m7.ts', 'validation/out/m7/scenes', '--only', missingPkgs.join(',')], () => false);
  const results: Record<string, any>[] = [];
  let perfOut: Record<string, any> | undefined;
  for (const part of parts) {
    if (part === 'core') { if (!o.only) gate0(dir, runId, add, runStep); continue; }
    if (part === 'stageB') {
      for (const u of STAGE_B_UNITS.filter((x) => !o.only || o.only.has(x.id) || o.only.has(x.pkg))) {
        const res = seqUnit(u, dir, runId, nUnits(), add, false);
        // T16 (M8): the ReSTIR side really ran on the CWBVH
        const meta = res.restir_dir ? tryJson(path.join(res.restir_dir, 'meta.json')) : undefined;
        const bvhOk = meta?.config?.bvh === 'cwbvh';
        if (!bvhOk) { res.ok = false; (res.t16 ??= []).push(`ReSTIR meta.config.bvh ${meta?.config?.bvh} (want cwbvh)`); }
        add(`T16 ${u.id}: ReSTIR ran on the CWBVH`, bvhOk, 0, undefined, `config.bvh ${meta?.config?.bvh}`);
        results.push({ ...res, part: 'stageB' });
      }
      continue;
    }
    if (part === 'perf') perfOut = perf(dir, add);
  }
  const failed = steps.filter((x) => !x.ok).map((x) => x.name);
  const summary = {
    milestone: 'M8', gate: 'Gate 0 (bitwise / equivalence evidence) + Stage B CWBVH end to end + the PLAN targets (recorded)',
    run: runId, parts, created: new Date().toISOString(), ok: failed.length === 0, total_s: Math.round((performance.now() - t0) / 100) / 10,
    code_hashes: hashes, failed, units: results, perf: perfOut, steps,
  };
  writeFileSync(path.join(ROOT, dir, 'summary.json'), `${JSON.stringify(summary, null, 1)}\n`);
  const md = [`# M8 gate ${runId}`, '', `Result: **${summary.ok ? 'PASS' : 'FAIL'}** (${summary.total_s} s, parts ${parts.join(', ')})`, '',
    '## Stage B', '', ...results.map((r) => `- ${r.unit}: ${r.status}, Δ_Y ${typeof r.global_rel_Y === 'number' ? `${(r.global_rel_Y * 100).toFixed(4)} %` : 'n/a'}, worst tile ${typeof r.worst_tile_rel_Y === 'number' ? `${(r.worst_tile_rel_Y * 100).toFixed(2)} %` : 'n/a'}`), '',
    ...(perfOut ? ['## Perf (ms)', '', '| scene | config | target | M8 | pre-M8 |', '|---|---|---|---|---|',
      ...perfOut.rows.map((x: any) => `| ${x.scene} | ${x.config} | ${x.target_ms} | ${x.m8_ms?.toFixed(2)} | ${x.pre_m8_ms?.toFixed(2)} |`), ''] : []),
    '## Failed steps', '', ...failed.map((f) => `- ${f}`), ''].join('\n');
  writeFileSync(path.join(ROOT, dir, 'summary.md'), md);
  console.log(`\nsummary: ${dir}/summary.json (${summary.total_s} s)`);
  return summary.ok;
}
