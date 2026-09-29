// M3b milestone gate (plan §5 M3b exit, §7.1 Gates 0–2, §7.3, §7.4 M3b): `npm run validate -- --milestone M3b`.
//   Gate 0  typecheck, cpu lane (tests/material/glass-ref.test.ts: U-G1…U-G10 CPU parts incl. the Tier-2 tables),
//           python stats tests, M3b package freshness, Chrome GPU tests: glass.gpu.test.ts (U-G1…U-G10: Fresnel,
//           eval/pdf vectors, full-sphere χ² vs the valid-only pdf with B-spur / η / r controls, albedo tables, weight /
//           support consistency, q(·) vectors, delta per sub-event, backfacing η, Λ(−c) = Λ(c), BTDF non-reciprocity),
//           pt-glass.gpu.test.ts (C0h/G1 closed forms, G2 η² detector, U9 A ≡ B ≡ A′ on a delta-free scene, the mirror
//           positive control A < B with A′ ≡ B, U10 pass-through replay determinism), and the M3a GPU suites (bsdf, pt,
//           lights) as regression.
//   Gate 1  (M3b items) the gap-glass §7 glass plants in OUR PT vs Cycles, each on a scene that detects it: B-η (1/η²
//           BTDF scaling; C0i immersed emitter), B-tint (C for √C), B-side (η not inverted on backfaces) and an
//           un-compensated P_R = 0.5 (G1 Principled slab), B-shadow (glass does not occlude; C0k); each detected in
//           ≥ 9/10 half-size repeats with the Cycles A/A control passing ≥ 9/10, and a full 16-vs-16 compare must fail;
//           our PT A/A (two seed sets). B-spur is a Gate-0 control (glass.gpu.test.ts χ² must reject the Cycles pdf).
//   Gate 2  Stage A (our PT ≡ Cycles; TOST δ 0.5% global / 2% per 32² tile on Y,R,G,B, suite FWER): C0h–C0k, G1–G10,
//           (vi) glass/mirror in Mode A (vs Cycles MIS off) and Mode B (vs Cycles MIS on), (vi-B), C0o; closed forms
//           checked on BOTH renderers (analytic_check.py glass-slab / glass-shadow). A failed unit is re-run once on
//           disjoint seeds. Rough-glass Mode-B scenes run in the documented approximate tier (report.json "tier").
// Scenes: validation/scenes/make-m3b.ts. Output: validation/out/m3b-gate-<time>/ (summary.json, summary.md, reports).
// Options: --only a,b (scene subset for debugging; the verdict then covers only those units).
// Helpers mirror gate-m3a.ts (kept separate so the M3a gate is untouched).
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { withGpuLockSync } from './gpu-lock.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const PY = path.join(ROOT, 'validation/.venv/bin/python');
const BLENDER = process.env.BLENDER ?? '/Applications/Blender.app/Contents/MacOS/Blender';
const REFS = 'validation/out/m3b/refs';
const OUR_SEED = 7, OUR_RERUN_SEED = 100_007;
/** stats.py num_eps (docs/decisions/cycles-deviations.md D2): f32 arithmetic floor of near-deterministic tiles. */
const NUM_EPS = 1e-4;
const NUM_EPS_NOTE = 'f32 arithmetic floor between two implementations (cycles-deviations.md D2)';

/** One Gate-2 unit: a package, the Cycles reference (spp × K seeds) and our PT (spp × B batches). */
export interface G2B {
  pkg: string;
  label: string;
  cyclesSpp: number;
  K: number;
  ourSpp: number;
  B: number;
  /** 'tight' (default) or 'approximate' (rough glass under Mode-B MIS: gap-glass §5.3; validation.md). */
  tier?: 'tight' | 'approximate';
  test?: Record<string, unknown>;
  analytic?: boolean;
  note?: string;
}

// Budgets (plan §7.3 sizing rule, SE_Δ ≤ δ/(t + z_Šidák)): defaults from the pilots; the pilot run
// validation/out/m3b-gate-20260929-025921 (1024×16 Cycles, 2048×16 ours, 1024 for the 512² scenes) measured the
// replicate multiplier needed by the worst 32² tile: g5 5.67, g8 6.99, g9 9.39, g10 11.34 (per-sample variance of
// the two renderers within 0.8–1.05× of each other, so both sides are scaled): the budgets below leave a margin of
// ×1.3–1.4 over the need (g5 ×1.32, g8 ×1.43, g9 ×1.31, g10 ×1.32).
// Cycles references use ≥ 4096 spp (more than 2048): with Sobol-Burley and next_pow2(spp) ≤ 2048, Cycles shows a
// seed-independent error of ~3e-4 on multi-bounce rough glass (G4) and structured tile errors on (vi) Mode B, which
// vanish from spp 2560 on (docs/decisions/cycles-deviations.md D4).
const S = (pkg: string, label: string, o: Partial<G2B> = {}): G2B => ({ pkg, label, cyclesSpp: 4096, K: 16, ourSpp: 2048, B: 16, ...o });

export const M3B_G2: G2B[] = [
  S('c0h_slab_transmission_256', 'C0h smooth slab over an emitter (closed form)', { analytic: true }),
  S('g1_slab_furnace_principled_256', 'G1 slab furnace, Principled C 0.5, N 3 (closed form)', { analytic: true }),
  S('g1_slab_furnace_glassnode_256', 'G1 slab furnace, Glass node c 0.8, N 2 (closed form)', { analytic: true }),
  S('c0i_immersed_emitter_256', 'C0i / G2 emitter inside smooth glass (η² detector, closed form)', { analytic: true }),
  S('c0j_rough_glass_furnace_512x256', 'C0j rough glass furnace, N 8'),
  S('c0k_glass_shadow_256', 'C0k glass shadow = 0', { analytic: true }),
  S('g3_rough_slab_reflection_512x256', 'G3 rough slabs, reflection only (N 0)'),
  S('g4_rough_slab_512x256', 'G4 rough slabs, N 3'),
  S('g5_panes_point_256', 'G5 point light over smooth / rough slab / rough pane (Mode A)', { cyclesSpp: 10240, ourSpp: 10240 }),
  S('g5b_panes_rect_B_256', 'G5b rect light over the panes, Mode B (rough glass + MIS)', { tier: 'approximate',
    note: 'rough transmission NEE of an MIS light: Cycles weights the spurious region by its power heuristic, we by 1 (gap-glass §5.3)' }),
  S('g6_caustic_B_256', 'G6 caustic through a smooth glass sphere, Mode B', { ourSpp: 8192 }),
  S('g6neg_caustic_A_256', 'G6-neg the same in Mode A (no caustic)'),
  S('g7_principled_mix_512', 'G7 Principled mixtures in a room (Mode A, N 4)', { ourSpp: 1024 }),
  S('g7_principled_mix_furnace_512x256', 'G7 Principled mixtures in a furnace (N 0)'),
  S('g8_cornell_glass_512', 'G8 Cornell + rough icosphere + smooth cube (N 8)', { cyclesSpp: 10240, ourSpp: 10240 }),
  S('g9_bubble_256', 'G9 bubbles (IOR 0.75), smooth and rough', { cyclesSpp: 16384, ourSpp: 16384 }),
  S('g10_colored_glass_refraction_256', 'G10 coloured Glass node + Refraction node', { cyclesSpp: 20480, ourSpp: 20480 }),
  S('vi_glass_mirror_A_512', '(vi) glass / mirror, Mode A (Cycles MIS off)', { ourSpp: 1024 }),
  S('vi_glass_mirror_B_512', '(vi) glass / mirror, Mode B (Cycles MIS on)', { ourSpp: 1024 }),
  S('vi_b_mirror_area_B_256', '(vi-B) rect light in a roughness-0 mirror, Mode B', { ourSpp: 8192 }),
  S('c0o_visible_camera_B_256', 'C0o visibleToCamera rect + disk, Mode B'),
];

/** Gate-1 items of M3b: the gap-glass §7 glass plants (+ an un-compensated P_R) in our PT vs Cycles, each on the scene
 *  that detects it (a 1/η² scaling cancels across the two interfaces of a slab, so B-η runs on the immersed emitter
 *  G2/C0i), and our A/A on the G1 Principled slab. */
const CAL_PKG = 'g1_slab_furnace_principled_256';
const PLANTS: { name: string; pkg: string; args: string[]; seed: number }[] = [
  { name: 'B-η: 1/η² BTDF radiance scaling', pkg: 'c0i_immersed_emitter_256', args: ['--plant-glass', 'eta2'], seed: 5005 },
  { name: 'P_R replaced by 0.5 without pdf compensation', pkg: CAL_PKG, args: ['--plant-glass', 'pr-half'], seed: 6006 },
  { name: 'B-tint: C instead of √C per interface', pkg: CAL_PKG, args: ['--plant-glass', 'tint'], seed: 7007 },
  { name: 'B-side: η not inverted on backfaces', pkg: CAL_PKG, args: ['--plant-glass', 'side'], seed: 8008 },
  { name: 'B-shadow: shadow rays pass through glass', pkg: 'c0k_glass_shadow_256', args: ['--plant-glass', 'shadow'], seed: 9009 },
];

type Rec = (name: string, ok: boolean, detail?: string) => void;
type Add = (n: string, ok: boolean, sec: number, d?: unknown, det?: string) => void;
interface Step { name: string; ok: boolean; seconds: number; data?: unknown; detail?: string }

const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const r4 = (x: unknown) => (typeof x === 'number' ? Number(x.toPrecision(4)) : x);
const pct = (x: number | undefined | null, d = 3) => (typeof x === 'number' ? `${(x * 100).toFixed(d)}%` : 'n/a');
const readJson = <T = Record<string, any>>(p: string): T => JSON.parse(readFileSync(path.join(ROOT, p), 'utf8')) as T;

function sh(cmd: string, argv: string[], echo: (l: string) => boolean = () => true): { code: number; out: string; seconds: number } {
  const t0 = performance.now();
  const r = spawnSync(cmd, argv, { cwd: ROOT, encoding: 'utf8', maxBuffer: 1 << 28, timeout: 7_200_000 });
  const out = `${r.stdout ?? ''}${r.stderr ?? ''}`;
  for (const l of out.split('\n')) if (l && echo(l)) console.log(`  ${l}`);
  return { code: r.status ?? 1, out, seconds: (performance.now() - t0) / 1000 };
}

function fileRun(run: string, dest: string): string {
  const to = path.join(dest, run);
  mkdirSync(path.join(ROOT, dest), { recursive: true });
  rmSync(path.join(ROOT, to), { recursive: true, force: true });
  if (existsSync(path.join(ROOT, 'validation/out', run))) renameSync(path.join(ROOT, 'validation/out', run), path.join(ROOT, to));
  return to;
}

export function milestoneM3b(record: Rec, only?: Set<string>): void {
  const t0 = performance.now();
  const runId = `m3b-gate-${stamp()}`;
  const dir = path.join('validation/out', runId);
  mkdirSync(path.join(ROOT, dir), { recursive: true });
  const steps: Step[] = [];
  const add: Add = (name, ok, seconds, data, detail) => {
    steps.push({ name, ok, seconds: Math.round(seconds * 10) / 10, data, detail });
    record(name, ok, `${detail ? `${detail}, ` : ''}${seconds.toFixed(1)} s`);
  };
  const runStep = (name: string, cmd: string, argv: string[], echo?: (l: string) => boolean) => {
    console.log(`\n--- ${name}: ${cmd} ${argv.join(' ')}`);
    const r = sh(cmd, argv, echo);
    add(name, r.code === 0, r.seconds, undefined, `exit ${r.code}`);
    return r;
  };
  const scenes = M3B_G2.filter((s) => !only || only.has(s.pkg));
  const full = !only;

  // ---- Gate 0 ---------------------------------------------------------------------------------------------------
  const missing = [PY, BLENDER].filter((p) => !existsSync(p));
  add('M3b gate tools (venv python, Blender)', missing.length === 0, 0, { missing }, missing.length ? `missing: ${missing.join(', ')}` : undefined);
  if (full) {
    runStep('typecheck', 'npx', ['tsc', '--noEmit']);
    runStep('vitest cpu (incl. glass-ref: U-G1…U-G10 CPU parts, Tier-2 tables)', 'npx', ['vitest', 'run', '--project', 'cpu'], (l) => /Test Files|Tests |FAIL|✗|×/.test(l));
    runStep('python stats tests (validation/tools/tests)', PY, ['-m', 'pytest', 'validation/tools/tests', '-q'], (l) => /passed|failed|error/i.test(l));
    packagesCurrent(dir, add);
    withGpuLockSync('gate-m3b', () => {
      for (const [name, file] of [
        ['U-G1…U-G10 glass lobe G: Fresnel, vectors, full-sphere χ² (+controls), albedo tables, consistency, q(·), delta, backfacing, Λ (chrome)', 'glass'],
        ['C0h/G1 closed forms, G2 η², U9 A ≡ B ≡ A′, mirror control A < B, U10 pass-through determinism (chrome)', 'pt-glass'],
        ['M3a regression: T8 BSDF (chrome)', 'bsdf'],
        ['M3a regression: T10/U11/C0 PT (chrome)', 'pt'],
        ['M3a regression: T9 lights (chrome)', 'lights'],
      ] as const) runStep(name, 'npx', ['vitest', 'run', '--project', 'chrome', `validation/gpu-tests/${file}.gpu.test.ts`], (l) => /Tests |FAIL|✗|×|AssertionError|GLASS_|U-G/.test(l));
    });
  }

  // ---- Gate 2 ----------------------------------------------------------------------------------------------------
  const nUnits = 4 * (M3B_G2.length + 1 + PLANTS.length);
  console.log(`\nGate 2: ${scenes.length} scenes; suite FWER over n_units = ${nUnits} (× {Y,R,G,B})`);
  const results: Record<string, unknown>[] = [];
  const refDirs: Record<string, string | undefined> = {};
  for (const s of scenes) {
    const ref = cyclesRef(s, REFS, `0..${s.K - 1}`, add);
    if (!ref) continue;
    refDirs[s.pkg] = ref;
    const ours = ourRun(s, `${runId}-${s.pkg}`, OUR_SEED, dir, add);
    if (!ours) continue;
    results.push(stageA(s, ours, ref, dir, runId, nUnits, add));
    if (s.analytic) for (const [side, d] of [['ours', ours], ['cycles', ref]] as const) analytic(s, side, d, add);
  }

  // ---- Gate 1 (M3b): glass plants in our PT vs Cycles, our A/A --------------------------------------------------------
  if (full) {
    const refFor = (pkg: string): [G2B, string | undefined] => {
      const u = M3B_G2.find((s) => s.pkg === pkg)!;
      return [u, refDirs[pkg] ?? (refDirs[pkg] = cyclesRef(u, REFS, `0..${u.K - 1}`, add))];
    };
    for (const p of PLANTS) {
      const [u, ref] = refFor(p.pkg);
      if (ref) plantDetection(u, p, ref, dir, runId, nUnits, add);
    }
    const [cal, calRef] = refFor(CAL_PKG);
    if (calRef) ourAA(cal, dir, runId, nUnits, add);
  }

  const failed = steps.filter((s) => !s.ok).map((s) => s.name);
  const summary = {
    milestone: 'M3b', gate: 'Gate 0 + Gate 1 (M3b items) + Gate 2 Stage A (plan §5 M3b exit)', run: runId, created: new Date().toISOString(),
    ok: failed.length === 0, subset: only ? [...only] : undefined, total_s: Math.round((performance.now() - t0) / 100) / 10, n_units: nUnits, failed,
    units: results, steps,
  };
  writeFileSync(path.join(ROOT, dir, 'summary.json'), `${JSON.stringify(summary, null, 1)}\n`);
  writeFileSync(path.join(ROOT, dir, 'summary.md'), table(results));
  console.log(`\n${table(results)}\nsummary: ${dir}/summary.json (${summary.total_s} s)`);
}

function table(results: Record<string, any>[]): string {
  const head = '| unit | tier | status | global Δ_Y | MDB_Y global | worst tile (rel) | tile MDB max | TOST-failed tiles | Cycles spp×K | PT spp×B | mult. needed |\n|---|---|---|---|---|---|---|---|---|---|---|\n';
  return head + results.map((r) => `| ${r.unit} | ${r.tier} | ${r.status} | ${pct(r.global_rel_Y, 4)} | ${pct(r.mdb_global_Y, 4)} | ${pct(r.worst_tile_rel_Y, 2)} ${r.worst_tile ?? ''} | ${pct(r.mdb_tile_max_Y, 2)} | ${r.tost_failed_tiles ?? ''} | ${r.cycles} | ${r.ours} | ${r4(r.multiplier_needed)} |`).join('\n') + '\n';
}

function packagesCurrent(dir: string, add: Add): void {
  const t0 = performance.now();
  const tmp = path.join(ROOT, dir, 'regen');
  const diffs: string[] = [];
  const r = sh('npx', ['tsx', 'validation/scenes/make-m3b.ts', tmp], () => false);
  if (r.code !== 0) diffs.push(`make-m3b.ts: exit ${r.code} ${r.out.slice(-300)}`);
  for (const p of existsSync(tmp) ? readdirSync(tmp) : []) {
    const a = path.join(tmp, p), b = path.join(ROOT, 'validation/scenes', p);
    const names = new Set([...readdirSync(a), ...(existsSync(b) ? readdirSync(b) : [])]);
    for (const n of names) if (!existsSync(path.join(a, n)) || !existsSync(path.join(b, n)) || !readFileSync(path.join(a, n)).equals(readFileSync(path.join(b, n)))) diffs.push(`${p}/${n}`);
  }
  rmSync(tmp, { recursive: true, force: true });
  add('Gate-2 scene packages current (make-m3b.ts)', diffs.length === 0, (performance.now() - t0) / 1000, { diffs }, diffs.length ? `stale: ${diffs.slice(0, 8).join(', ')}` : undefined);
}

function cyclesRef(s: G2B, out: string, seeds: string, add: Add): string | undefined {
  console.log(`\n--- Cycles reference ${s.pkg}: ${s.cyclesSpp} spp × seeds ${seeds}`);
  const r = sh(BLENDER, ['-b', '--factory-startup', '--python-exit-code', '1', '-P', 'validation/blender/render_reference.py', '--',
    '--package', `validation/scenes/${s.pkg}`, '--out', out, '--spp', String(s.cyclesSpp), '--seeds', seeds],
  (l) => l.startsWith('[render_reference]') && (l.includes('RESULT') || l.includes('cache') || l.includes('waited')) || /Error|Traceback/.test(l));
  const line = r.out.split('\n').reverse().find((l) => l.startsWith('[render_reference] RESULT '));
  const res = line ? JSON.parse(line.slice('[render_reference] RESULT '.length)) as { dir: string; cache_hit: boolean; renders: number; renders_total_s?: number } : undefined;
  add(`Cycles reference ${s.pkg} (${s.cyclesSpp} spp × ${seeds})`, r.code === 0 && !!res, r.seconds,
    res && { dir: path.relative(ROOT, res.dir), cache_hit: res.cache_hit, renders: res.renders, renders_total_s: res.renders_total_s ?? 0 },
    res ? (res.cache_hit ? 'cache hit' : `rendered ${res.renders} in ${res.renders_total_s} s`) : `exit ${r.code} ${r.out.slice(-400)}`);
  return res ? path.relative(ROOT, res.dir) : undefined;
}

function ourRun(s: G2B, run: string, seed: number, dir: string, add: Add, extra: string[] = [], label = 'PT batches'): string | undefined {
  console.log(`\n--- ${label} ${s.pkg}: ${s.ourSpp} spp × ${s.B} batches, seed ${seed}${extra.length ? ` ${extra.join(' ')}` : ''}`);
  const r = sh('npx', ['tsx', 'validation/harness/run-batches.ts', '--package', `validation/scenes/${s.pkg}`, '--kernel', 'pt', '--spp', String(s.ourSpp),
    '--batches', String(s.B), '--seed', String(seed), '--run', run, ...extra], (l) => /^(FAIL)\s|errors:|Error/.test(l));
  const d = fileRun(run, path.join(dir, 'pt'));
  const ok = r.code === 0 && existsSync(path.join(ROOT, d, 'meta.json'));
  const ms = ok ? readJson(path.join(d, 'meta.json')).timings.totalMs : 0;
  add(`${label} ${s.pkg} (${s.ourSpp} spp × ${s.B})`, ok, r.seconds, { dir: d, gpu_total_ms: Math.round(ms) }, ok ? `${(ms / 1000).toFixed(1)} s in the page` : `exit ${r.code} ${r.out.slice(-300)}`);
  return ok ? d : undefined;
}

function writeTest(dir: string, name: string, nUnits: number, extra: Record<string, unknown> = {}): string {
  const p = path.join(dir, 'tests', `test-${name}.json`);
  mkdirSync(path.join(ROOT, dir, 'tests'), { recursive: true });
  writeFileSync(path.join(ROOT, p), JSON.stringify({ name, stage: 'A', channels: ['Y', 'R', 'G', 'B'], n_units: nUnits, tier: 'tight', num_eps: NUM_EPS, num_eps_note: NUM_EPS_NOTE, ...extra }, null, 1));
  return p;
}

function compare(ours: string, ref: string, test: string, out: string, rerunOf?: string) {
  return sh(PY, ['validation/tools/compare.py', '--ours', ours, '--ref', ref, '--test', test, '--out', out, ...(rerunOf ? ['--rerun-of', rerunOf] : [])], () => false);
}

function summarize(rep: Record<string, any>) {
  const Y = rep.channels.Y;
  return {
    status: rep.status, failed_checks: rep.failed_checks, tier: rep.tier,
    global_rel_Y: r4(Y.global_.rel), mdb_global_Y: r4(Y.global_.mdb), worst_tile_rel_Y: r4(Y.tiles.worst_rel), worst_tile: Y.tiles.worst_tile ? `[${Y.tiles.worst_tile.join(',')}]` : undefined,
    mdb_tile_max_Y: r4(Y.tiles.mdb_max), mdb_tile_median_Y: r4(Y.tiles.mdb_median), tost_failed_tiles: Y.tiles.tost_failed, zero_var_tiles: Y.tiles.zero_var,
    multiplier_needed: r4(Math.max(...['Y', 'R', 'G', 'B'].map((c) => Math.max(rep.channels[c].global_.replicate_multiplier_needed ?? 1, rep.channels[c].tiles.replicate_multiplier_needed ?? 1)))),
    per_channel: Object.fromEntries(['R', 'G', 'B'].map((c) => [c, { global_rel: r4(rep.channels[c].global_.rel), worst_tile_rel: r4(rep.channels[c].tiles.worst_rel), tost_failed: rep.channels[c].tiles.tost_failed }])),
  };
}

function stageA(s: G2B, ours: string, ref: string, dir: string, runId: string, nUnits: number, add: Add): Record<string, unknown> {
  const unit = s.pkg;
  const test = writeTest(dir, unit, nUnits, { tier: s.tier ?? 'tight', ...(s.note ? { tier_note: s.note } : {}), ...(s.test ?? {}) });
  const out = path.join(dir, 'compare', unit);
  const t0 = performance.now();
  let c = compare(ours, ref, test, out);
  let rep = existsSync(path.join(ROOT, out, 'report.json')) ? readJson(path.join(out, 'report.json')) : undefined;
  let rerun: Record<string, unknown> | undefined;
  if (rep?.status === 'rerun_required') {
    console.log(`  ${unit}: rerun_required (${rep.failed_checks.join(', ')}) -> confirmatory re-run on disjoint seeds`);
    const ref2 = cyclesRef(s, REFS, `${100}..${100 + s.K - 1}`, add);
    const o2 = ourRun(s, `${runId}-${unit}-rerun`, OUR_RERUN_SEED, dir, add, [], 'PT re-run batches');
    if (ref2 && o2) {
      const out2 = `${out}-rerun`;
      c = compare(o2, ref2, test, out2, path.join(out, 'report.json'));
      const rep2 = existsSync(path.join(ROOT, out2, 'report.json')) ? readJson(path.join(out2, 'report.json')) : undefined;
      rerun = { first: summarize(rep), report: path.join(out2, 'report.json') };
      rep = rep2;
    }
  }
  const sum = rep ? summarize(rep) : undefined;
  const ok = !!rep && (rep.status === 'pass' || rep.status === 'pass_on_rerun');
  const res = {
    unit, label: s.label, tier: s.tier ?? 'tight', ...(sum ?? { status: `error (exit ${c.code})` }), cycles: `${s.cyclesSpp}×${s.K}`, ours: `${s.ourSpp}×${s.B}`,
    report: path.join(out, 'report.json'), ...(rerun ? { rerun } : {}), ...(s.note ? { note: s.note } : {}), ...(rep?.notes && Object.keys(rep.notes).length ? { notes: rep.notes } : {}),
  };
  add(`Stage A ${unit} (${s.label})`, ok, (performance.now() - t0) / 1000, res,
    sum ? `${sum.status}${s.tier === 'approximate' ? ' [approximate tier]' : ''}: Δ_Y ${pct(sum.global_rel_Y as number, 4)} (MDB ${pct(sum.mdb_global_Y as number, 3)}), worst tile ${pct(sum.worst_tile_rel_Y as number, 2)}, tile MDB max ${pct(sum.mdb_tile_max_Y as number, 2)}${sum.failed_checks?.length ? `, failed ${sum.failed_checks.join(' ')}` : ''}` : `compare exit ${c.code} ${c.out.slice(-300)}`);
  return res;
}

function analytic(s: G2B, side: string, d: string, add: Add): void {
  const r = sh(PY, ['validation/tools/analytic_check.py', '--dir', d, '--package', `validation/scenes/${s.pkg}`, '--json'], () => false);
  let res: Record<string, any> = {};
  try { res = JSON.parse(r.out.trim().split('\n').pop()!); } catch { res = { ok: false, error: r.out.slice(-300) }; }
  if (res.ok === null) return;
  const Y = res.channels?.Y;
  const detail = res.shadow_pixels !== undefined
    ? `${res.shadow_pixels} shadow pixels, max |value| ${res.max_abs} over ${res.n} replicates`
    : Y ? `Δ_Y ${pct(Y.rel, 4)} vs closed form (z ${Y.z.toFixed(2)}), worst tile ${pct(Y.worst_tile_rel, 2)}, ${res.valid_pixels ?? 'all'} px` : res.error;
  add(`analytic ${s.pkg} (${side})`, res.ok === true, r.seconds, res, detail);
}

function plantDetection(s: G2B, p: { name: string; args: string[]; seed: number }, ref: string, dir: string, runId: string, nUnits: number, add: Add): void {
  const o = ourRun(s, `${runId}-plant-${p.seed}`, p.seed, dir, add, p.args, `PT planted (${p.name})`);
  if (!o) return;
  const test = writeTest(dir, `gate1-plant-${p.seed}`, nUnits, { calibration: { plants: [] } });
  const out = path.join(dir, 'calibrate', `plant-${p.seed}`);
  const r = sh(PY, ['validation/tools/compare.py', '--calibrate', '--ref', ref, '--planted', o, '--plant-name', `our PT ${p.name}`, '--test', test, '--out', out,
    '--splits', '20', '--repeats', '10'], () => false);
  let data: Record<string, unknown> | undefined;
  let detail = `exit ${r.code}`;
  let ok = false;
  if (existsSync(path.join(ROOT, out, 'report.json'))) {
    const rep = readJson(path.join(out, 'report.json'));
    const rp = rep.calibration.rendered_plant;
    const full = compare(o, ref, writeTest(dir, `gate1-plant-${p.seed}-full`, nUnits), `${out}-full`);
    const fullRep = existsSync(path.join(ROOT, `${out}-full`, 'report.json')) ? readJson(path.join(`${out}-full`, 'report.json')) : undefined;
    void full;
    ok = !!rp && rp.gate_fail_count >= 9 && rp.control_pass_count >= 9 && !!fullRep && fullRep.status !== 'pass';
    data = {
      report: path.join(out, 'report.json'), detected: `${rp.gate_fail_count}/${rp.n_repeats}`, equivalence_failed: `${rp.equivalence_fail_count}/${rp.n_repeats}`,
      control_pass: `${rp.control_pass_count}/${rp.n_repeats}`, half_size: rp.half_size, planted_rel_Y: r4(rp.channels.Y.planted_global_rel_median),
      mdb_global_Y: r4(rp.channels.Y.mdb_global_median),
      full_compare: fullRep && { status: fullRep.status, global_rel_Y: r4(fullRep.channels.Y.global_.rel), failed_checks: fullRep.failed_checks },
    };
    detail = `detected ${rp.gate_fail_count}/10 (TOST ${rp.equivalence_fail_count}/10), control ${rp.control_pass_count}/10, planted Δ_Y ${pct(rp.channels.Y.planted_global_rel_median, 3)}; full 16 vs 16: ${fullRep?.status}`;
  }
  add(`Gate 1 plant in our PT vs Cycles ${s.pkg}: ${p.name}`, ok, r.seconds, data, detail);
}

function ourAA(s: G2B, dir: string, runId: string, nUnits: number, add: Add): void {
  const a = ourRun(s, `${runId}-aa-1001`, 1001, dir, add, [], 'PT A/A seed set 1001');
  const b = ourRun(s, `${runId}-aa-2002`, 2002, dir, add, [], 'PT A/A seed set 2002');
  if (!a || !b) return;
  const out = path.join(dir, 'compare', `aa-ours-${s.pkg}`);
  const c = compare(a, b, writeTest(dir, `aa-ours-${s.pkg}`, nUnits), out);
  const rep = existsSync(path.join(ROOT, out, 'report.json')) ? readJson(path.join(out, 'report.json')) : undefined;
  const sum = rep ? summarize(rep) : undefined;
  add(`Gate 1 A/A on two of our PT seed sets (${s.pkg}, 1001 vs 2002)`, rep?.status === 'pass', 0, sum && { ...sum, report: path.join(out, 'report.json') },
    sum ? `${sum.status}: Δ_Y ${pct(sum.global_rel_Y as number, 4)} (MDB ${pct(sum.mdb_global_Y as number, 4)}), tile MDB max ${pct(sum.mdb_tile_max_Y as number, 2)}` : `exit ${c.code}`);
}
