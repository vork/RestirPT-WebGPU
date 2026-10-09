// perf2 V-UNB Stage-B units (docs/decisions/perf2-plan.md §2 "Standard verification sets"): a results-changing perf flag
// forced on in the trees = 1 chain configuration it affects, against our PT at the unchanged Stage-B δ (0.2 % global /
// 1 % per 32² tile), through the M5 / M6 gates' own chain machinery (pilots + joint sizing, PT references, chains,
// compare.py, one confirmatory re-run on disjoint seeds per failed test frame, T16).
//   npx tsx validation/harness/v-unb.ts --kernel-flags RS_RIS_PREPASS [--only id,…] [--pilot-only]
// Units: full-m6 Mode A (m5s_cornell_i, the rung-3.7 chain unit), full-m6 Mode B (m6_crossings_B_256, the rung-3.11
// chain unit) and an HDRI package (m7_nm_env_256: normal maps under the overcast HDRI, full-m6). Each unit's run-batches
// call gets `--kernel-flags`; the flags are recorded in every chain meta (meta.config.perfFlags). The statistics use the
// M6 gate's n_units (its Bonferroni split), so a pass here is a pass at the gate's per-unit level.
// Output: validation/out/v-unb-<flags>-<time>/ (summary.json / .md, chains/, compare/, tests/). Takes the GPU lock per
// run-batches call (do not wrap).
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { EXT, runExternalChainUnits } from './gate-m5.ts';
import { CHAIN_UNITS, M6_OUT, SEEDS, nUnits, pkgDirM6, type ChainUnitM6 } from './gate-m6.ts';
import { codeHashes } from './gate-m4.ts';
import { checkValidationPerfFlags } from '../../src/core/render/restir/perf-flags.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const { values: a } = parseArgs({ options: { 'kernel-flags': { type: 'string' }, only: { type: 'string' }, 'pilot-only': { type: 'boolean', default: false } } });
if (!a['kernel-flags']) { console.error('usage: v-unb.ts --kernel-flags A,B [--only id,…] [--pilot-only]'); process.exit(2); }
const { key: flags } = checkValidationPerfFlags(a['kernel-flags'], false);

const M7_ENV = 'm7_nm_env_256';
const byId = (id: string) => { const u = CHAIN_UNITS.find((x) => x.id === id); if (!u) throw new Error(`no chain unit ${id}`); return u; };
const UNITS: ChainUnitM6[] = [
  byId('m5s_cornell_i@3.7-full-m6'),
  byId('m6_crossings_B_256@3.11-full-m6'),
  { id: `${M7_ENV}@V-UNB-full-m6`, part: 'r37', kind: 'static', pkg: M7_ENV, rung: 'V-UNB', preset: 'full-m6', extra: [], frames: 25, testFrames: [1, 24], rounds: 1,
    lightMode: 'A', label: 'chains full-m6, normal maps under the overcast HDRI' },
];
const only = a.only ? new Set(a.only.split(',')) : undefined;
const units = UNITS.filter((u) => !only || only.has(u.id) || only.has(u.pkg)).map((u) => ({ ...u, extra: [...u.extra, '--kernel-flags', flags] }));

const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const runId = `v-unb-${flags.replace(/[^\w]+/g, '_')}-${stamp()}`;
const dir = path.join('validation/out', runId);
mkdirSync(path.join(ROOT, dir), { recursive: true });
const steps: { name: string; ok: boolean; seconds: number; data?: unknown; detail?: string }[] = [];
const add = (name: string, ok: boolean, seconds: number, data?: unknown, detail?: string) => {
  steps.push({ name, ok, seconds: Math.round(seconds * 10) / 10, data, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail}, ${seconds.toFixed(1)} s)` : ''}`);
};

// the M6 gate's chain setup (gate-m6.ts setupExt) + the M7 HDRI package
for (const p of ['m6_crossings_B_256', 'ixs_b_area_B_256']) EXT.pkgDirs.set(p, pkgDirM6(p));
EXT.pkgDirs.set(M7_ENV, `validation/out/m7/scenes/${M7_ENV}`);
EXT.out = M6_OUT;
EXT.chainSeeds = { chains: SEEDS.chains, chainsRerun: SEEDS.chainsRerun, chainPilot: SEEDS.chainPilot };

const nU = nUnits();
const h = codeHashes();
console.log(`V-UNB ${runId}: flags ${flags}, units ${units.map((u) => u.id).join(', ')}, n_units ${nU} (M6 gate), PT code ${h.pt.slice(0, 12)}, ReSTIR ${h.restir.slice(0, 12)}`);
const t0 = performance.now();
const r = runExternalChainUnits(units, { dir, runId, nU, add, pilotOnly: a['pilot-only'] });
const failed = steps.filter((x) => !x.ok).map((x) => x.name);
const summary = { runId, flags, nUnits: nU, codeHashes: h, ok: failed.length === 0 && r.results.length > 0, failed, seconds: Math.round((performance.now() - t0) / 1000), results: r.results, sizing: r.sizing, steps };
writeFileSync(path.join(ROOT, dir, 'summary.json'), `${JSON.stringify(summary, null, 1)}\n`);
const pct = (x: unknown, d = 4) => (typeof x === 'number' ? `${(x * 100).toFixed(d)}%` : 'n/a');
const md = [`# V-UNB ${runId}`, '', `Flags: \`${flags}\`. Result: **${summary.ok ? 'PASS' : 'FAIL'}** (${summary.seconds} s, n_units ${nU})`, '',
  '| unit | status | Δ_Y | MDB_Y | worst tile | R | PT | T16 |', '|---|---|---|---|---|---|---|---|',
  ...r.results.map((x) => `| ${x.unit} | ${x.status} | ${pct(x.global_rel_Y)} | ${pct(x.mdb_global_Y, 3)} | ${pct(x.worst_tile_rel_Y, 2)} | ${x.R ?? ''} | ${x.pt ?? ''} | ${(x.t16 ?? []).join('; ')} |`),
  '', ...(failed.length ? ['Failed steps:', ...failed.map((f) => `- ${f}`)] : [])].join('\n');
writeFileSync(path.join(ROOT, dir, 'summary.md'), `${md}\n`);
console.log(`\n${md}`);
process.exit(summary.ok ? 0 : 1);
