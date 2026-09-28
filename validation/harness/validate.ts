// Milestone gate (plan §5: "each milestone ends with `npm run validate -- --milestone Mx` green").
// M0: typecheck, cpu + node-dawn lanes, Chrome lane, Chrome smoke CLI, and the M0 exit artifacts.
// M1: typecheck, cpu lane (ENV-U1, scene/BVH/layout tests), node-dawn pre-check, Chrome lane for T12 (bvh),
//     ENV-U2/U7 (env), textures and the primary pass, the app-shell e2e and the M1 app smoke (Sponza + HDRI).
//   npm run validate -- --milestone M0|M1
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const { values: args } = parseArgs({ options: { milestone: { type: 'string', default: 'M0' } } });

interface Step { name: string; ok: boolean; detail?: string }
const steps: Step[] = [];
function record(name: string, ok: boolean, detail?: string): void {
  steps.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
}

function run(name: string, cmd: string, cmdArgs: string[]): void {
  console.log(`\n--- ${name}: ${cmd} ${cmdArgs.join(' ')}`);
  const t0 = performance.now();
  const r = spawnSync(cmd, cmdArgs, { cwd: ROOT, stdio: 'inherit' });
  record(name, r.status === 0, `exit ${r.status ?? r.signal}, ${((performance.now() - t0) / 1000).toFixed(1)} s`);
}

function file(rel: string, test: (text: string) => string | null = () => null): void {
  const p = path.join(ROOT, rel);
  if (!existsSync(p)) return record(rel, false, 'missing');
  let err: string | null;
  try { err = test(readFileSync(p, 'utf8')); } catch (e) { err = String(e); }
  record(rel, err === null, err ?? undefined);
}

function budgetPopulated(text: string): string | null {
  const b = JSON.parse(text) as { entries?: { scene: string; max_bounces: number; res: number; s_per_4096spp: number }[] };
  const want = [['cornell', 3], ['cornell', 7], ['sponza', 3]] as const;
  const missing = want.filter(([s, k]) => !b.entries?.some(
    (e) => e.scene === s && e.max_bounces === k && e.res === 512 && Number.isFinite(e.s_per_4096spp) && e.s_per_4096spp > 0));
  return missing.length ? `no s/4096spp @512² for ${missing.map(([s, k]) => `${s} b=${k}`).join(', ')}` : null;
}

function milestoneM0(): void {
  run('typecheck', 'npx', ['tsc', '--noEmit']);
  run('vitest cpu + node-dawn', 'npx', ['vitest', 'run', '--project', 'cpu', '--project', 'node-dawn']);
  run('vitest chrome', 'npx', ['vitest', 'run', '--project', 'chrome']);
  // Takes /tmp/restirpt-gpu.lock itself around the allocation probes.
  run('chrome smoke', 'npx', ['tsx', 'validation/harness/run-chrome.ts', '--smoke']);
  file('validation/budget.json', budgetPopulated);
  file('docs/decisions/usd.md', (t) => (/^## Decision:/m.test(t) ? null : 'no "## Decision:" heading'));
  file('docs/decisions/platform-lanes.md');
  file('docs/math.md');
}

// Shared GPU lock (plan §1.8: GPU-heavy jobs never overlap). Scripts that take the lock themselves run outside it.
const GPU_LOCK = '/tmp/restirpt-gpu.lock';
function withGpuLock(fn: () => void): void {
  const nap = new Int32Array(new SharedArrayBuffer(4));
  for (;;) { try { mkdirSync(GPU_LOCK); break; } catch { console.log('waiting for the GPU lock ...'); Atomics.wait(nap, 0, 0, 5000); } }
  const release = () => { try { rmdirSync(GPU_LOCK); } catch { /* gone */ } };
  process.once('exit', release);
  try { fn(); } finally { release(); process.removeListener('exit', release); }
}

const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');

/** M1 gate inputs. Their tests skip (ENV-U1 vs OIIO) or fall back (smoke: Cornell instead of Sponza) when these are
 *  missing, which would turn the gate green without testing what it claims, so the gate itself requires them. */
function m1Assets(): void {
  const need = ['validation/.venv/bin/python', 'validation/assets/cornell/cornell.glb', 'validation/assets/downloaded/sponza/Sponza.gltf'];
  const hdris = JSON.parse(readFileSync(path.join(ROOT, 'validation/assets/hdris.json'), 'utf8')) as { files: { file: string }[] };
  need.push(...hdris.files.map((f) => `validation/assets/downloaded/hdri/${f.file}`));
  const missing = need.filter((p) => !existsSync(path.join(ROOT, p)));
  record('M1 gate assets (OIIO venv, Sponza, HDRIs)', missing.length === 0,
    missing.length ? `missing: ${missing.join(', ')} (validation/blender/fetch_sponza.py, validation/assets/fetch_hdris.ts)` : undefined);
}

function milestoneM1(): void {
  const runId = `m1-${stamp()}`;
  m1Assets();
  run('typecheck', 'npx', ['tsc', '--noEmit']);
  run('vitest cpu (ENV-U1 RGBE/EXR vs OIIO, loader, BVH, layouts)', 'npx', ['vitest', 'run', '--project', 'cpu']);
  withGpuLock(() => {
    run('vitest node-dawn (pre-check)', 'npx', ['vitest', 'run', '--project', 'node-dawn']);
    run('T12 BVH brute force / watertight / offsets / overflow (chrome)', 'npx', ['vitest', 'run', '--project', 'chrome', 'validation/gpu-tests/bvh.gpu.test.ts']);
    run('ENV-U2 mapping + ENV-U7 bilinear/pole-wrap (chrome)', 'npx', ['vitest', 'run', '--project', 'chrome', 'validation/gpu-tests/env.gpu.test.ts']);
    run('textures validation/interactive paths (chrome)', 'npx', ['vitest', 'run', '--project', 'chrome', 'validation/gpu-tests/textures.gpu.test.ts']);
    run('primary pass V-buffer / MASK / env miss (chrome)', 'npx', ['vitest', 'run', '--project', 'chrome', 'validation/gpu-tests/primary.gpu.test.ts']);
  });
  // These take the GPU lock themselves.
  run('app shell e2e (test pattern)', 'npx', ['tsx', 'tests/app/e2e-app.ts', `validation/out/${runId}/app-e2e`]);
  run('M1 app smoke (Sponza + HDRI, fly camera, views, timings)', 'npx', ['tsx', 'validation/harness/m1-app-smoke.ts', '--run', runId]);
  const rep = path.join(ROOT, 'validation/out', runId, 'report.json');
  if (existsSync(rep)) {
    const r = JSON.parse(readFileSync(rep, 'utf8')) as { timing?: Record<string, { isolatedMedianMs?: number; hudPrimary?: { ms: number } }>; warnings?: string[] };
    for (const [k, v] of Object.entries(r.timing ?? {})) console.log(`timing  primary ${k}: ${v.isolatedMedianMs?.toFixed(3)} ms isolated, ${v.hudPrimary?.ms.toFixed(3)} ms in-app avg`);
    for (const w of r.warnings ?? []) console.log(`WARN    ${w}`);
    console.log(`screenshots: validation/out/${runId}/`);
  }
}

const gates: Record<string, () => void> = { M0: milestoneM0, M1: milestoneM1 };
const gate = gates[args.milestone!.toUpperCase()];
if (!gate) {
  console.error(`unknown milestone ${args.milestone}; known: ${Object.keys(gates).join(', ')}`);
  process.exit(2);
}
gate();
const failed = steps.filter((s) => !s.ok);
console.log(`\n=== validate ${args.milestone}: ${steps.length - failed.length}/${steps.length} passed ===`);
for (const s of failed) console.log(`FAIL  ${s.name}${s.detail ? `  (${s.detail})` : ''}`);
process.exit(failed.length ? 1 : 0);
