// M5.5 gate: the A-SVGF-lite denoiser (PLAN §5 M5.5; docs/decisions/denoiser.md §11).
//   Gate 0   typecheck, CPU unit tests (tests/denoise), the Chrome denoiser suite (validation/gpu-tests/denoiser.gpu.test.ts),
//            the M5.5 app smoke (toggle, views, mode defaults, ReSTIR-unbiased forced off, held frames, screenshots).
//   FLIP     (i) cornell_i_512, (v) v_glossy_v1_512, (vii) vii_textured_512, ix-d ixs_d_camera_256: ReSTIR-interactive 1-frame
//            output (rsFrame) vs the denoised output of the same run against our 65 536-spp PT reference (16 × 4096 spp,
//            cached in validation/out/m55/ptrefs, keyed by package hash + PT code hash + spp + batches + seed + frame);
//            pass iff mean LDR-FLIP(raw) / mean LDR-FLIP(denoised) ≥ 2 per scene (HDR-FLIP reported).
//   Recovery ixs_e_addremove_256 (C added, A ×2, B removed) and ixs_e_half_256 (A ×0.5), each state held 24 frames (32
//            before the first step), 8 seeds: frames until the masked regional mean is within 5 % of the step
//            (denoiser.md §11.3); pass iff ≤ 8 for every step.
//   Timing   960×540 (cornell_i_512, vii_textured_512 re-rendered at 960×540), default settings, separate timing
//            submits (Q3): pass iff the mean total ≤ 3 ms.
//   T16      every PT reference used carries t16.denoiser === 'none' (harness.ts wrapper).
// Every render takes the shared GPU lock itself (run-batches.ts, run-denoise.ts, the app smoke; Chrome GPU tests through
// withGpuLockSync), each hold ≤ 12 min. Output: validation/out/m55-gate-<time>/{summary.json, summary.md}.
//   npm run validate -- --milestone M5.5 [--only flip,recovery,timing,gate0] [--prerender-ptrefs]
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { decodePFM, encodePFM } from '../../src/core/io/pfm.ts';
import { LOCK_CHUNK_S, chunkBatches, codeHashes as m4CodeHashes, mergeChunkMetas, packageHash, tsClosure } from './gate-m4.ts';
import { withGpuLockSync } from './gpu-lock.ts';
import { denoiserT16Problems } from './t16.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PY = path.join(ROOT, 'validation/.venv/bin/python');
export const M55_OUT = 'validation/out/m55';
const PTREFS = `${M55_OUT}/ptrefs`;
const DNRUNS = `${M55_OUT}/dnruns`;
const TIMEOUT_MS = 24 * 3600_000;

// ------------------------------------------------------------------------------------------------ configuration

export const SEEDS = { ptRef: 55001, ptMask: 55101, dn: 55201 } as const;
/** 65 536 spp = 16 batches × 4096 spp (PLAN §5 M5.5 "64k-spp PT"). */
export const REF = { spp: 4096, B: 16 } as const;
/** Recovery masks (tile means of the before / after states): 4096 spp = 4 × 1024. */
export const MASK_REF = { spp: 1024, B: 4 } as const;
/** Measured PT cost per 4096 spp at 512² (budget.json, cornell b = 3: 12.8 s); others scale by pixels, × 1.5 margin. */
const PT_S_PER_4096_512 = 12.8 * 1.5;

export interface FlipScene { pkg: string; label: string; frames: number[]; dynamic: boolean }
/** Evaluation frames: static scenes after a reset; ix-d at its package test frames (frame-override PT references). */
export const FLIP_SCENES: FlipScene[] = [
  { pkg: 'cornell_i_512', label: '(i) diffuse Cornell', frames: [16, 32, 48, 63], dynamic: false },
  { pkg: 'v_glossy_v1_512', label: '(v) glossy sweep', frames: [16, 32, 48, 63], dynamic: false },
  { pkg: 'vii_textured_512', label: '(vii) textures, flat', frames: [16, 32, 48, 63], dynamic: false },
  { pkg: 'ixs_d_camera_256', label: 'ix-d moving camera', frames: [16, 24, 32, 40, 48, 56, 64], dynamic: true },
];
export const FLIP_SEEDS = 4;
export const FLIP_RATIO_MIN = 2;

/** ix-e steps: package states held for `hold` frames; the step frames of the run are the first frame of each state. */
export interface RecoveryRun { pkg: string; label: string; states: number[]; names: string[]; warm: number; hold: number }
export const RECOVERY_RUNS: RecoveryRun[] = [
  { pkg: 'ixs_e_addremove_256', label: 'ix-e add / ×2 / remove', states: [7, 8, 14, 20], names: ['C added', 'A ×2', 'B removed'], warm: 32, hold: 24 },
  { pkg: 'ixs_e_half_256', label: 'ix-e ×0.5', states: [13, 14], names: ['A ×0.5'], warm: 32, hold: 24 },
];
export const RECOVERY_SEEDS = 8;
export const RECOVERY_MAX_FRAMES = 8;
export const TIMING = { width: 960, height: 540, warmup: 32, iterations: 64, scenes: ['cornell_i_512', 'vii_textured_512'], maxMs: 3 } as const;

/** Frame schedule of a recovery run: run frame → package frame. Step k starts at warm + (k−1)·hold. */
export function recoverySchedule(r: RecoveryRun): { pkgFrames: number[]; steps: number[] } {
  const pkgFrames: number[] = [];
  const steps: number[] = [];
  r.states.forEach((s, i) => {
    if (i > 0) steps.push(pkgFrames.length);
    const n = i === 0 ? r.warm : r.hold;
    for (let k = 0; k < n; k++) pkgFrames.push(s);
  });
  return { pkgFrames, steps };
}

// ------------------------------------------------------------------------------------------------ helpers

const sha = (b: string | Buffer) => createHash('sha256').update(b).digest('hex');
const stableJson = (v: unknown): string => JSON.stringify(v, (_k, x) => (x && typeof x === 'object' && !Array.isArray(x)
  ? Object.fromEntries(Object.entries(x as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : 1))) : x));
const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const readJson = <T = Record<string, any>>(p: string): T => JSON.parse(readFileSync(path.join(ROOT, p), 'utf8')) as T;
const tryJson = (p: string): Record<string, any> | undefined => { try { return readJson(p); } catch { return undefined; } };
export const pkgDir = (pkg: string): string => `validation/scenes/${pkg}`;

function sh(cmd: string, argv: string[], echo: (l: string) => boolean = () => true, env?: Record<string, string>): { code: number; out: string; seconds: number } {
  const t0 = performance.now();
  const r = spawnSync(cmd, argv, { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 28, timeout: TIMEOUT_MS, ...(env ? { env: { ...process.env, ...env } } : {}) });
  const out = `${r.stdout ?? ''}${r.stderr ?? ''}`;
  for (const l of out.split('\n')) if (l && echo(l)) console.log(`  ${l}`);
  return { code: r.status ?? 1, out, seconds: (performance.now() - t0) / 1000 };
}

type Add = (name: string, ok: boolean, seconds: number, data?: unknown, detail?: string) => void;
type Rec = (name: string, ok: boolean, detail?: string) => void;
interface Run { dir: string; meta: Record<string, any>; seconds: number; cacheHit?: boolean }

/** Denoiser code hash (the renderer side and the denoise harness): cache key of the denoise runs. */
let dnHash: string | undefined;
export function denoiseCodeHash(): string {
  if (!dnHash) {
    const files = [...tsClosure(['validation/harness/denoise-run.ts']),
      ...readdirSync(path.join(ROOT, 'src/core/render/denoise/shaders')).filter((f) => f.endsWith('.wgsl')).map((f) => `src/core/render/denoise/shaders/${f}`)].sort();
    dnHash = sha(files.map((r) => `${r}\0${sha(readFileSync(path.join(ROOT, r)))}\n`).join(''));
  }
  return dnHash;
}

// ------------------------------------------------------------------------------------------------ PT references

const rbEcho = (l: string) => /^(FAIL)\s|errors:|Error|lock wait/.test(l);

/** run-batches --kernel pt [--frames t] into `dest`, chunked into ≤ 12-min GPU-lock holds and merged (mean.pfm). */
function ptRunInto(pdir: string, frame: number | undefined, spp: number, B: number, seed: number, dest: string, estSeconds: number): { dir?: string; meta?: Record<string, any>; code: number; out: string; seconds: number } {
  const k = chunkBatches(B, estSeconds);
  rmSync(path.join(ROOT, dest), { recursive: true, force: true });
  mkdirSync(path.join(ROOT, dest), { recursive: true });
  const base = ['--package', pdir, '--kernel', 'pt', '--spp', String(spp), '--seed', String(seed), ...(frame !== undefined ? ['--frames', String(frame)] : [])];
  const produced = (run: string) => path.join(ROOT, 'validation/out', frame !== undefined ? `${run}-f${frame}` : run);
  const metas: Record<string, any>[] = [];
  let seconds = 0, out = '';
  for (let off = 0; off < B; off += k) {
    const n = Math.min(k, B - off);
    const run = `m55pt-${sha(dest).slice(0, 10)}-c${off}`;
    if (B > k) console.log(`  chunk batches ${off}..${off + n - 1} of ${B}`);
    const r = sh('npx', ['tsx', 'validation/harness/run-batches.ts', ...base, '--batches', String(n), ...(off ? ['--batch-offset', String(off)] : []), '--run', run], rbEcho);
    seconds += r.seconds; out += r.out.slice(-2000);
    const from = produced(run);
    const meta = tryJson(path.relative(ROOT, path.join(from, 'meta.json')));
    if (r.code !== 0 || !meta) { rmSync(from, { recursive: true, force: true }); return { code: r.code || 1, out, seconds }; }
    for (const f of readdirSync(from)) if (/^batch_\d{3}\.pfm$/.test(f)) renameSync(path.join(from, f), path.join(ROOT, dest, f));
    rmSync(from, { recursive: true, force: true });
    metas.push(meta);
  }
  const meta = metas.length === 1 ? metas[0] : mergeChunkMetas(metas);
  const files = readdirSync(path.join(ROOT, dest)).filter((f) => /^batch_\d{3}\.pfm$/.test(f)).sort();
  const imgs = files.map((f) => decodePFM(new Uint8Array(readFileSync(path.join(ROOT, dest, f)))));
  const mean = new Float32Array(imgs[0].data.length);
  for (const im of imgs) for (let i = 0; i < mean.length; i++) mean[i] += im.data[i] / imgs.length;
  writeFileSync(path.join(ROOT, dest, 'mean.pfm'), encodePFM({ width: imgs[0].width, height: imgs[0].height, channels: 3, data: mean }));
  writeFileSync(path.join(ROOT, dest, 'meta.json'), JSON.stringify(meta, null, 1));
  return { code: meta.ok ? 0 : 1, out, seconds, dir: dest, meta };
}

function pkgPixels(pkg: string): number {
  const j = readJson(path.join(pkgDir(pkg), 'scene.json'));
  return j.render.width * j.render.height;
}

/** Cached PT reference (key: package hash, PT code hash, spp, batches, seed, frame). */
export function ptRef(pkg: string, frame: number | undefined, spp: number, B: number, seed: number, add: Add, tag: string): Run | undefined {
  const pdir = pkgDir(pkg);
  const keyObj = { kind: 'pt', gate: 'm55', pkg, frame: frame ?? null, packageHash: packageHash(pdir), spp, B, seed, rr: false, code: m4CodeHashes().pt };
  const key = sha(stableJson(keyObj)).slice(0, 16);
  const dest = path.join(PTREFS, `${pkg}-f${frame ?? 'base'}-s${seed}-${spp}x${B}-${key}`);
  const meta = tryJson(path.join(dest, 'meta.json'));
  const n = existsSync(path.join(ROOT, dest)) ? readdirSync(path.join(ROOT, dest)).filter((f) => /^batch_\d{3}\.pfm$/.test(f)).length : 0;
  const step = `PT ${tag} ${pkg} f${frame ?? 'base'} (${spp} spp × ${B}, seed ${seed})`;
  if (meta?.ok && n === B && existsSync(path.join(ROOT, dest, 'mean.pfm'))) { add(step, true, 0, { dir: dest, cache_hit: true }, 'cache hit'); return { dir: dest, meta, seconds: 0, cacheHit: true }; }
  console.log(`\n--- ${step}`);
  const est = (spp * B / 4096) * PT_S_PER_4096_512 * pkgPixels(pkg) / (512 * 512);
  const r = ptRunInto(pdir, frame, spp, B, seed, dest, est);
  const ok = r.code === 0 && !!r.meta?.ok;
  if (r.dir) writeFileSync(path.join(ROOT, dest, 'cache-key.json'), `${JSON.stringify(keyObj, null, 1)}\n`);
  add(step, ok, r.seconds, { dir: dest, cache_hit: false }, ok ? `rendered in ${(r.meta!.timings.totalMs / 1000).toFixed(1)} s` : `exit ${r.code} ${r.out.slice(-300)}`);
  return ok ? { dir: dest, meta: r.meta!, seconds: r.seconds, cacheHit: false } : undefined;
}

/** Every PT reference of the gate: [pkg, frame, spp, B, seed, tag]. */
export function ptRefList(): [string, number | undefined, number, number, number, string][] {
  const out: [string, number | undefined, number, number, number, string][] = [];
  for (const s of FLIP_SCENES) {
    if (s.dynamic) for (const f of s.frames) out.push([s.pkg, f, REF.spp, REF.B, SEEDS.ptRef, 'reference']);
    else out.push([s.pkg, undefined, REF.spp, REF.B, SEEDS.ptRef, 'reference']);
  }
  for (const r of RECOVERY_RUNS) for (const st of r.states) out.push([r.pkg, st, MASK_REF.spp, MASK_REF.B, SEEDS.ptMask, 'mask reference']);
  return out;
}

// ------------------------------------------------------------------------------------------------ the gate

export interface M55Options { only?: Set<string>; prerenderPtRefs?: boolean }

export function milestoneM55(record: Rec, o: M55Options = {}): void {
  const t0 = performance.now();
  const runId = `m55-gate-${stamp()}`;
  const dir = path.join('validation/out', runId);
  mkdirSync(path.join(ROOT, dir), { recursive: true });
  const steps: { name: string; ok: boolean; seconds: number; data?: unknown; detail?: string }[] = [];
  const add: Add = (name, ok, seconds, data, detail) => {
    steps.push({ name, ok, seconds: Math.round(seconds * 10) / 10, data, detail });
    record(name, ok, `${detail ? `${detail}, ` : ''}${seconds.toFixed(1)} s`);
  };
  const want = (k: string) => !o.only || o.only.has(k);

  if (o.prerenderPtRefs) {
    for (const [pkg, f, spp, B, seed, tag] of ptRefList()) ptRef(pkg, f, spp, B, seed, add, tag);
  } else {
    gateBody(dir, add, want);
  }
  const failed = steps.filter((s) => !s.ok).map((s) => s.name);
  const summary = { milestone: 'M5.5', run: runId, created: new Date().toISOString(), ok: failed.length === 0, total_s: Math.round((performance.now() - t0) / 100) / 10, failed, steps };
  writeFileSync(path.join(ROOT, dir, 'summary.json'), `${JSON.stringify(summary, null, 1)}\n`);
  const md = [`# M5.5 gate ${runId}: ${summary.ok ? 'PASS' : 'FAIL'}`, '', '| step | result | detail |', '|---|---|---|',
    ...steps.map((x) => `| ${x.name} | ${x.ok ? 'pass' : '**FAIL**'} | ${(x.detail ?? '').replace(/\|/g, '/')} |`)].join('\n');
  writeFileSync(path.join(ROOT, dir, 'summary.md'), `${md}\n`);
  console.log(`\nsummary: ${dir}/summary.json, summary.md`);
}

const dnEcho = (l: string) => /^(OK|FAIL)\s|RESULT|lock wait|denoiser .* ms|Error/.test(l);

/** Vitest config: in a git worktree whose node_modules is a symlink, a local config keeps Vite's cache here. */
function vitestConfigArgs(): string[] {
  try {
    if (lstatSync(path.join(ROOT, 'node_modules')).isSymbolicLink() && existsSync(path.join(ROOT, 'vitest.m4local.config.ts'))) return ['--config', 'vitest.m4local.config.ts'];
  } catch { /* default config */ }
  return [];
}

interface DnJob { argv: string[]; run: string; dest: string; key: Record<string, unknown> }

/** Cached denoise runs: the jobs whose dest has no ok meta.json run through run-denoise.ts --jobs in one lock hold. */
function denoiseRuns(jobs: DnJob[], label: string, add: Add): boolean {
  const todo = jobs.filter((j) => !tryJson(path.join(j.dest, 'meta.json'))?.ok);
  if (!todo.length) { add(`${label}: ${jobs.length} denoise runs`, true, 0, undefined, 'cache hit'); return true; }
  const file = path.join(M55_OUT, `jobs-${sha(label + JSON.stringify(todo.map((j) => j.argv))).slice(0, 10)}.json`);
  mkdirSync(path.join(ROOT, M55_OUT), { recursive: true });
  writeFileSync(path.join(ROOT, file), JSON.stringify(todo.map((j) => [...j.argv, '--run', j.run])));
  console.log(`\n--- ${label}: ${todo.length} denoise runs (one GPU-lock hold)`);
  const r = sh('npx', ['tsx', 'validation/harness/run-denoise.ts', '--jobs', file], dnEcho);
  let ok = r.code === 0;
  for (const j of todo) {
    const from = path.join(ROOT, 'validation/out', j.run);
    if (!existsSync(from)) { ok = false; continue; }
    rmSync(path.join(ROOT, j.dest), { recursive: true, force: true });
    mkdirSync(path.dirname(path.join(ROOT, j.dest)), { recursive: true });
    renameSync(from, path.join(ROOT, j.dest));
    writeFileSync(path.join(ROOT, j.dest, 'cache-key.json'), `${JSON.stringify(j.key, null, 1)}\n`);
    ok &&= !!tryJson(path.join(j.dest, 'meta.json'))?.ok;
  }
  add(`${label}: ${todo.length} denoise runs`, ok, r.seconds, { jobs: file }, ok ? undefined : r.out.slice(-600));
  return ok;
}

function dnJob(pkg: string, mode: string, seed: number, extra: string[]): DnJob {
  const argv = ['--package', pkgDir(pkg), '--mode', mode, '--seed', String(seed), ...extra];
  const key = { kind: 'denoise', pkg, packageHash: packageHash(pkgDir(pkg)), argv, code: denoiseCodeHash(), restir: m4CodeHashes().restir };
  const k = sha(stableJson(key)).slice(0, 12);
  return { argv, run: `m55dn-${pkg}-${mode}-s${seed}-${k}`, dest: path.join(DNRUNS, `${pkg}-${mode}-s${seed}-${k}`), key };
}

function gateBody(dir: string, add: Add, want: (k: string) => boolean): void {
  const results: Record<string, unknown> = {};
  if (want('gate0')) {
    const t0 = performance.now();
    const tc = sh('npx', ['tsc', '--noEmit'], (l) => /error/.test(l));
    add('typecheck', tc.code === 0, tc.seconds);
    const cpu = sh('npx', ['vitest', 'run', ...vitestConfigArgs(), '--project', 'cpu', 'tests/denoise', 'tests/restir/gate-m4-config.test.ts', 'tests/restir/gate-m5-config.test.ts'], (l) => /Test Files|Tests |FAIL|✗|×/.test(l));
    add('CPU: tests/denoise (mirrors, modes, reference formulas, T16 helpers, DN10) + gate config tests', cpu.code === 0, cpu.seconds);
    let gpu = { code: 1, out: '', seconds: 0 };
    withGpuLockSync('validate-m55', () => {
      gpu = sh('npx', ['vitest', 'run', ...vitestConfigArgs(), '--project', 'chrome', 'validation/gpu-tests/denoiser.gpu.test.ts'], (l) => /Test Files|Tests |FAIL|✗|×/.test(l));
    });
    add('Chrome: denoiser.gpu.test.ts (U-DN-1…5: every pass vs the f64 reference, held frames, timing re-run, no timestamps)', gpu.code === 0, gpu.seconds);
    const smoke = sh('npx', ['tsx', 'validation/harness/m55-app-smoke.ts', '--run', `${path.basename(dir)}-app`], (l) => /^(PASS|FAIL)\s|RESULT/.test(l));
    add('M5.5 app smoke (toggle, modes, views, held frames, HUD timing, screenshots)', smoke.code === 0, smoke.seconds, { dir: `validation/out/${path.basename(dir)}-app` });
    results.gate0_s = (performance.now() - t0) / 1000;
  }
  // T16 on the PT references the gate uses (rendered or cached; harness.ts T16 wrapper)
  const refs = new Map<string, Run>();
  for (const [pkg, f, spp, B, seed, tag] of ptRefList()) {
    if ((tag === 'reference' && !want('flip')) || (tag === 'mask reference' && !want('recovery'))) continue;
    const r = ptRef(pkg, f, spp, B, seed, add, tag);
    if (!r) continue;
    const t16 = denoiserT16Problems(r.meta, r.dir, true);
    if (t16.length) add(`T16 ${r.dir}`, false, 0, undefined, t16.join('; '));
    refs.set(`${pkg}@${f ?? 'base'}`, r);
  }
  if (refs.size) add(`T16: denoiser off in every PT reference readback (${refs.size} references)`, [...refs.values()].every((r) => r.meta.t16?.denoiser === 'none'), 0);

  if (want('flip')) {
    const jobs: DnJob[] = [];
    const per = new Map<string, DnJob[]>();
    for (const s of FLIP_SCENES) {
      const T = Math.max(...s.frames) + 1;
      const js = Array.from({ length: FLIP_SEEDS }, (_, k) => dnJob(s.pkg, 'flip', SEEDS.dn + k, ['--frames', String(T), '--eval-frames', s.frames.join(',')]));
      per.set(s.pkg, js); jobs.push(...js);
    }
    const ran = denoiseRuns(jobs, 'FLIP renders (raw and denoised 1-frame ReSTIR-interactive)', add);
    const flip: Record<string, unknown> = {};
    for (const s of FLIP_SCENES) {
      const js = per.get(s.pkg)!;
      if (!ran && !js.every((j) => tryJson(path.join(j.dest, 'meta.json'))?.ok)) { add(`FLIP ${s.label}`, false, 0, undefined, 'renders missing'); continue; }
      const outs: Record<string, any>[] = [];
      for (const f of s.frames) {
        const ref = refs.get(`${s.pkg}@${s.dynamic ? f : 'base'}`);
        if (!ref) continue;
        const pairs = js.map((j) => `${path.join(ROOT, j.dest, `raw_f${f}.pfm`)}:${path.join(ROOT, j.dest, `dn_f${f}.pfm`)}`).join(',');
        const out = path.join(dir, `flip-${s.pkg}-f${f}.json`);
        const png = f === s.frames[s.frames.length - 1] ? ['--png', path.join(ROOT, dir, `png-${s.pkg}-f${f}`)] : [];
        const r = sh(PY, ['validation/tools/denoise_eval.py', 'flip', '--ref', path.join(ROOT, ref.dir, 'mean.pfm'), '--pairs', pairs, '--out', path.join(ROOT, out), ...png], (l) => l.startsWith('FLIP'));
        if (r.code === 0) outs.push({ frame: f, ...readJson(out) });
      }
      if (!outs.length) { add(`FLIP ${s.label}`, false, 0, undefined, 'no evaluation'); continue; }
      const m = (k: string) => outs.reduce((a, o) => a + o[k], 0) / outs.length;
      const ldrRaw = m('ldr_raw'), ldrDn = m('ldr_dn'), hdrRaw = m('hdr_raw'), hdrDn = m('hdr_dn');
      const ratio = ldrRaw / ldrDn;
      flip[s.pkg] = { label: s.label, frames: s.frames, seeds: FLIP_SEEDS, ldr_raw: ldrRaw, ldr_dn: ldrDn, ratio_ldr: ratio, hdr_raw: hdrRaw, hdr_dn: hdrDn, ratio_hdr: hdrRaw / hdrDn,
        rmse_raw: m('rmse_raw'), rmse_dn: m('rmse_dn'), flip_evaluator: outs[0].flip_evaluator, per_frame: outs.map((o) => ({ frame: o.frame, ldr_raw: o.ldr_raw, ldr_dn: o.ldr_dn, hdr_raw: o.hdr_raw, hdr_dn: o.hdr_dn })) };
      add(`FLIP ${s.label}: mean LDR-FLIP raw / denoised ≥ ${FLIP_RATIO_MIN}`, ratio >= FLIP_RATIO_MIN, 0, flip[s.pkg],
        `LDR ${ldrRaw.toFixed(4)} → ${ldrDn.toFixed(4)} (×${ratio.toFixed(2)}), HDR ${hdrRaw.toFixed(4)} → ${hdrDn.toFixed(4)} (×${(hdrRaw / hdrDn).toFixed(2)}), ${outs.length} frames × ${FLIP_SEEDS} seeds`);
    }
    results.flip = flip;
  }

  if (want('recovery')) {
    const rec: Record<string, unknown> = {};
    const jobs: DnJob[] = [];
    const per = new Map<string, DnJob[]>();
    for (const r of RECOVERY_RUNS) {
      const sch = recoverySchedule(r);
      const spec = r.states.map((st, i) => `${st}x${i === 0 ? r.warm : r.hold}`).join(',');
      const js = Array.from({ length: RECOVERY_SEEDS }, (_, k) => dnJob(r.pkg, 'recovery', SEEDS.dn + 100 + k, ['--pkg-frames', spec]));
      void sch;
      per.set(r.pkg, js); jobs.push(...js);
    }
    denoiseRuns(jobs, 'recovery renders (ix-e steps held 24 frames)', add);
    for (const r of RECOVERY_RUNS) {
      const sch = recoverySchedule(r);
      const js = per.get(r.pkg)!.filter((j) => tryJson(path.join(j.dest, 'meta.json'))?.ok);
      const pairs = r.states.slice(1).map((st, i) => {
        const b = refs.get(`${r.pkg}@${r.states[i]}`), a = refs.get(`${r.pkg}@${st}`);
        return b && a ? `${path.join(ROOT, b.dir, 'mean.pfm')}:${path.join(ROOT, a.dir, 'mean.pfm')}` : undefined;
      });
      if (js.length < RECOVERY_SEEDS || pairs.some((x) => !x)) { add(`recovery ${r.label}`, false, 0, undefined, 'renders or mask references missing'); continue; }
      const out = path.join(dir, `recovery-${r.pkg}.json`);
      const c = sh(PY, ['validation/tools/denoise_eval.py', 'recovery', '--runs', js.map((j) => path.join(ROOT, j.dest)).join(','), '--steps', sch.steps.join(','),
        '--hold', String(r.hold), '--refs', pairs.join(','), '--names', r.names.join(','), '--max-frames', String(RECOVERY_MAX_FRAMES), '--out', path.join(ROOT, out)], (l) => / @ /.test(l));
      const rep = tryJson(out);
      rec[r.pkg] = rep;
      for (const st of (rep?.steps ?? []) as Record<string, any>[]) {
        add(`recovery ${r.pkg} ${st.step} (frame ${st.frame}): 95 % within ≤ ${RECOVERY_MAX_FRAMES} frames`, !!st.pass, 0, st,
          st.error ?? `denoised ${st.dn.frames_to_95 ?? '> 16'} frames (raw ${st.raw.frames_to_95}), mask ${st.mask_tiles} tiles, PT ${Number(st.pt_before).toPrecision(4)} → ${Number(st.pt_after).toPrecision(4)}, `
            + `steady state vs PT: denoised ${(100 * st.dn.steady_vs_pt_after).toFixed(2)} %, raw ${(100 * st.raw.steady_vs_pt_after).toFixed(2)} %`);
      }
      if (c.code !== 0 && !rep) add(`recovery ${r.label}`, false, c.seconds, undefined, c.out.slice(-400));
    }
    results.recovery = rec;
  }

  if (want('timing')) {
    const jobs = TIMING.scenes.map((pkg) => dnJob(pkg, 'timing', SEEDS.dn + 900, ['--width', String(TIMING.width), '--height', String(TIMING.height), '--warmup', String(TIMING.warmup),
      '--timing-submits', String(TIMING.iterations / 8), '--timing-runs', '8']));
    // timing is never cached: it measures this machine now
    for (const j of jobs) rmSync(path.join(ROOT, j.dest), { recursive: true, force: true });
    denoiseRuns(jobs, `timing at ${TIMING.width}×${TIMING.height}`, add);
    const tm: Record<string, unknown> = {};
    for (const j of jobs) {
      const meta = tryJson(path.join(j.dest, 'meta.json'));
      const t = meta?.timing;
      const pkg = String(j.key.pkg);
      tm[pkg] = { ...t, adapter: meta?.adapterInfo, chrome: meta?.chromeVersion };
      add(`timing ${pkg} at ${TIMING.width}×${TIMING.height}: denoiser ≤ ${TIMING.maxMs} ms`, !!t && t.meanTotalMs <= TIMING.maxMs, 0, tm[pkg],
        t ? `${t.meanTotalMs.toFixed(3)} ms mean (median ${t.medianTotalMs.toFixed(3)}, ${t.submits}×${t.runsPerSubmit} re-runs): ${t.passes.map((p: { name: string; ms: number }) => `${p.name} ${p.ms.toFixed(3)}`).join(', ')}; ${meta?.adapterInfo?.description ?? ''}` : 'no timing');
    }
    results.timing = tm;
  }
  writeFileSync(path.join(ROOT, dir, 'results.json'), `${JSON.stringify(results, null, 1)}\n`);
}

export { LOCK_CHUNK_S };
