// Milestone gate (plan §5: "each milestone ends with `npm run validate -- --milestone Mx` green").
// M0: typecheck, cpu + node-dawn lanes, Chrome lane, Chrome smoke CLI, and the M0 exit artifacts.
//   npm run validate -- --milestone M0
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
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

const gates: Record<string, () => void> = { M0: milestoneM0 };
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
