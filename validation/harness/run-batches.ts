// Batch validation renders in headless Chrome 154 (plan §5 M2): starts Vite, opens the harness page, and calls
// window.__harness.renderBatches under the shared GPU lock. Output: validation/out/<run>/batch_###.pfm, mean.pfm,
// meta.json. With --check, runs validation/tools/marker_check.py on mean.pfm against the package's "expected".
//   npx tsx validation/harness/run-batches.ts --package validation/scenes/c0b_512 --spp 64 --batches 4 --check
//   npx tsx validation/harness/run-batches.ts --package validation/scenes/c0p_512 --frames 0,1,2 --check
//   npx tsx validation/harness/run-batches.ts --scene /validation/assets/cornell/cornell.usda --spp 16 --batches 2
//   npx tsx validation/harness/run-batches.ts --package validation/scenes/cornell_i_512 --kernel pt --spp 256 --batches 16
//     (M3a reference PT; --max-bounces N overrides the package's render.maxBounces, --rr, --technique mis|nee|bsdf,
//      --light-mode A|B|A' overrides the package's light mode (plan §1.4; U9 A ≡ B), --plant-glass eta2|pr-half|tint|side|shadow
//      (M3b glass plants, pt-kernel.ts GlassPlant: B-η 1/η² BTDF scaling, R/T chosen with 0.5 instead of P_R without
//      pdf compensation, B-tint C instead of √C, B-side η not inverted on backfaces, B-shadow glass does not occlude),
//      M7 Stage-A Normal Map plants: --plant-nm sign|strength (the bitangent sign ignored / glTF-style strength; NM_PLANT),
//      Gate-1 planted biases: --plant-emit-scale 1.01 (every emitter ×s), --plant-drop 0.01@2 (terminate 1% of the
//      paths at vertex 2, no compensation); M3c env options: --env-nee on|off (default: the package's env.sampling),
//      --env-cap N (importance resolution), --env-no-floors, --env-mis-power, --env-plant
//      missingSin|w2WithoutPmf|doubleCount|pdfFromTargets, --env-strength-scale 1.0075)
//   npx tsx validation/harness/run-batches.ts --package validation/scenes/cornell_i_512 --kernel restir --preset offline --spp 64 --batches 16
//     (M4 Stage-B ReSTIR, restir-api.md §6.4: --spp = frames per batch (alias --frames-per-batch), --preset
//      initial|initial-rr|offline|criteria2022, --members E (ensemble atlas, writes ensemble.npz), --plant no-j|marginal-j,
//      --w-scale s (W × s plant), --max-bounces N; --env-nee on|off as for the PT; M6: --preset offline-m6, --light-mode
//      A|B|A' (default: the package's), --restir-settings JSON (Partial<RestirSettings>: pairing, risNee, rr, dualMv,
//      dupmap, plant {u8T2, u8RisMixed, u8TilePmf, u8CrossOcc}, …; recorded in meta.json))
//   --batch-offset N (pt / restir sequential): render batches N … N+batches−1 of a longer run (the same samples; the
//      gate splits long references into GPU-lock chunks and merges the batch files)
//   npx tsx validation/harness/run-batches.ts --package validation/scenes/ixs_d_camera_256 --kernel restir --preset full --chains 256
//     (M5 temporal chains, restir-temporal-api.md §6.3–§6.7: --chains R (multiple of --members E, default 16), --batch-offset b
//      (first chain batch; GPU-lock chunks), --chain-base c, --chain-frames T / --test-frames a,b (default: the package's
//      sequence), --average from:to (rung 3.5), --masks DIR (f<t>/masks.json + masks.bin), --temporal-mis contribution|talbot,
//      --temporal-check none|recompute|robust, --refresh exact|e2, --boost NB, --tplant n1Mixed,noJP,… (TP_* plants),
//      --u8-plant u8W1Delta,… (RSF U8 plants), --w-scale s, --mode disocc (M_disocc flags of test frame t; E = 1, jitter
//      off). Output validation/out/<run>/f<t>/{ensemble.npz, meta.json} (+ avg/) + meta.json)
// If the package directory is missing and --make-c0b is given, an equivalent C0b package (calib_scenes.py make_c0b:
// 100 m emissive quad at z = −2, L_e = (0.5, 0.25, 0.125)·2, vfov 40°, 512²) is written with exportScenePackage to
// validation/out/tmp-c0b/ and rendered instead.
import { quantizeScene } from '../../src/core/scene/quantize.ts';
import { execFile } from 'node:child_process';
import { existsSync, lstatSync } from 'node:fs';
import { mkdir, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises';
import { createServer as createNetServer } from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium, type Browser } from 'playwright';
import { createServer, type ViteDevServer } from 'vite';
import { decodePFM } from '../../src/core/io/pfm.ts';
import { exportScenePackage } from '../../src/core/scene/scene-package.ts';
import type { SceneData } from '../../src/core/scene/types.ts';
import type { RenderBatchesReport, ValidationKernel } from './batch-run.ts';
import type { RestirPlantName } from './restir-batch-run.ts';
import type { RenderRestirChainsOptions, TPlantName, U8PlantName } from './restir-chain-run.ts';
import type { RestirPresetName, RestirSettings } from '../../src/core/render/restir/presets.ts';
import { acquireGpuLock, GPU_LOCK } from './gpu-lock.ts';
import type { GlassPlant, PtEnvOptions, PtEnvPlant, PtPlant, PtTechnique } from '../../src/core/render/pt-kernel.ts';

const GLASS_PLANTS: readonly GlassPlant[] = ['eta2', 'pr-half', 'tint', 'side', 'shadow'];   // pt-kernel.ts order

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const OUT = path.join(ROOT, 'validation/out');
const PYTHON = path.join(ROOT, 'validation/.venv/bin/python');
const MARKER_CHECK = path.join(ROOT, 'validation/tools/marker_check.py');

const OPTIONS = {
  package: { type: 'string' },
  scene: { type: 'string' },
  kernel: { type: 'string', default: 'emission' },
  spp: { type: 'string', default: '64' },
  batches: { type: 'string', default: '4' },
  width: { type: 'string' },
  height: { type: 'string' },
  seed: { type: 'string' },
  run: { type: 'string' },
  frames: { type: 'string' },
  check: { type: 'boolean', default: false },
  'make-c0b': { type: 'boolean', default: false },
  'max-bounces': { type: 'string' },
  rr: { type: 'boolean', default: false },
  technique: { type: 'string' },
  'plant-emit-scale': { type: 'string' },
  'plant-drop': { type: 'string' },
  'light-mode': { type: 'string' },
  'plant-glass': { type: 'string' },
  'plant-nm': { type: 'string' },
  'env-nee': { type: 'string' },
  'env-cap': { type: 'string' },
  'env-no-floors': { type: 'boolean', default: false },
  'env-mis-power': { type: 'boolean', default: false },
  'env-plant': { type: 'string' },
  'env-strength-scale': { type: 'string' },
  preset: { type: 'string', default: 'initial' },
  'batch-offset': { type: 'string' },
  'frames-per-batch': { type: 'string' },
  members: { type: 'string' },
  plant: { type: 'string' },
  'w-scale': { type: 'string' },
  chains: { type: 'string' },
  'chain-base': { type: 'string' },
  'chain-frames': { type: 'string' },
  'test-frames': { type: 'string' },
  average: { type: 'string' },
  masks: { type: 'string' },
  'temporal-mis': { type: 'string' },
  'temporal-check': { type: 'string' },
  refresh: { type: 'string' },
  boost: { type: 'string' },
  tplant: { type: 'string' },
  'u8-plant': { type: 'string' },
  mode: { type: 'string' },
  'max-spp-per-dispatch': { type: 'string' },
  'restir-settings': { type: 'string' },
} as const;
const parse = (argv?: string[]) => parseArgs({ options: { ...OPTIONS, jobs: { type: 'string' } }, ...(argv ? { args: argv } : {}) }).values;
let args = parse();

const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');

/** `<root>/.vite-cache-<dir>` when node_modules is a symlink (git worktree sharing another checkout's modules). */
function worktreeCacheDir(): string | undefined {
  try {
    return lstatSync(path.join(ROOT, 'node_modules')).isSymbolicLink() ? path.join(ROOT, `.vite-cache-${path.basename(ROOT).replace(/^WebGPURestirPT-?/, '') || 'wt'}-harness`) : undefined;
  } catch { return undefined; }
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const s = createNetServer();
    s.once('error', reject);
    s.listen(0, '127.0.0.1', () => { const { port } = s.address() as { port: number }; s.close(() => resolve(port)); });
  });
}

function run(cmd: string, argv: string[]): Promise<{ code: number; out: string }> {
  return new Promise((resolve) => {
    execFile(cmd, argv, { cwd: ROOT, timeout: 300_000, maxBuffer: 1 << 26 }, (err, stdout, stderr) => {
      resolve({ code: err ? (typeof err.code === 'number' ? err.code : 1) : 0, out: `${stdout}${stderr}` });
    });
  });
}

/** C0b per validation/blender/calib_scenes.py make_c0b (used only when the Blender-side package is absent). */
async function makeC0b(dir: string): Promise<void> {
  const S = 50, D = 2;
  const positions = Float32Array.from([-S, -S, -D, S, -S, -D, S, S, -D, -S, S, -D]);
  const scene: SceneData = {
    name: 'c0b_512',
    geometry: {
      positions, normals: Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]), tangents: new Float32Array(16),
      uv0: new Float32Array(8), indices: Uint32Array.from([0, 1, 2, 0, 2, 3]), triMaterial: new Uint32Array(2), triFlags: Uint32Array.from([2, 2]),
    },
    materials: [{
      name: 'emitter', model: 'v1', v1: { diffuse: [0, 0, 0], glossy: [0, 0, 0], roughness: 0.5, mix: 0 },
      baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 0.5, emissiveFactor: [0.5, 0.25, 0.125], emissiveStrength: 2,
      ior: 1.5, specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true,
    }],
    textures: [], lights: [], cameras: [], bounds: { min: [-S, -S, -D], max: [S, S, -D] }, warnings: [],
  };
  const pkg = await exportScenePackage(quantizeScene(scene).scene, { // package v2 (data-formats.md §B0)
    camera: { matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], yfov: 40 * Math.PI / 180 },
    render: { width: 512, height: 512, maxBounces: 0 }, lightMode: 'A', flatShaded: true, name: 'c0b_512',
    source: { uri: 'validation/harness/run-batches.ts makeC0b' },
  });
  const json = { ...pkg.json, expected: { kind: 'constant', value: [1, 0.5, 0.25], tolerance: 1e-4, frames: { 0: {} } } };
  pkg.files.set('scene.json', new TextEncoder().encode(JSON.stringify(json, null, 1)));
  await mkdir(dir, { recursive: true });
  for (const [k, v] of pkg.files) await writeFile(path.join(dir, k), v);
}

/** A shared page when running a --jobs list (one Vite/Chrome, one GPU-lock hold for the whole list). */
interface SharedPage { page: import('playwright').Page; chromeVersion: string }

async function main(shared?: SharedPage): Promise<number> {
  if (!args.package && !args.scene) {
    console.error('usage: run-batches.ts (--package DIR | --scene URL) [--spp 64] [--batches 4] [--frames 0,1] [--check] [--make-c0b]');
    return 2;
  }
  let pkgDir = args.package ? path.resolve(ROOT, args.package) : undefined;
  if (pkgDir && !existsSync(path.join(pkgDir, 'scene.json'))) {
    if (!args['make-c0b']) { console.error(`${args.package}: no scene.json (use --make-c0b for the C0b fallback)`); return 2; }
    pkgDir = path.join(OUT, 'tmp-c0b');
    await makeC0b(pkgDir);
    console.log(`package missing: wrote an equivalent C0b package to ${path.relative(ROOT, pkgDir)}`);
  }
  const pkgUrl = pkgDir ? `/${path.relative(ROOT, pkgDir).split(path.sep).join('/')}/` : undefined;
  const frames = args.frames ? args.frames.split(',').map(Number) : [undefined];
  const seed = args.seed !== undefined ? Number(args.seed) >>> 0 : (Math.random() * 2 ** 32) >>> 0;
  const base = args.run ?? `batches-${path.basename(pkgDir ?? args.scene!).replace(/[^\w.-]+/g, '_')}-${stamp()}`;

  let plant: PtPlant | undefined;
  if (args['plant-emit-scale'] || args['plant-drop'] || args['plant-glass'] || args['plant-nm']) {
    plant = {};
    if (args['plant-nm']) {
      if (!['sign', 'strength'].includes(args['plant-nm'])) { console.error('--plant-nm sign|strength'); return 2; }
      plant.nm = args['plant-nm'] as 'sign' | 'strength';
    }
    if (args['plant-glass']) {
      if (!(GLASS_PLANTS as readonly string[]).includes(args['plant-glass'])) { console.error(`--plant-glass ${GLASS_PLANTS.join('|')}`); return 2; }
      plant.glass = args['plant-glass'] as GlassPlant;
    }
    if (args['plant-emit-scale']) plant.emitScale = Number(args['plant-emit-scale']);
    if (args['plant-drop']) {
      const m = /^([\d.eE+-]+)@(\d+)$/.exec(args['plant-drop']);
      if (!m) { console.error('--plant-drop PROB@BOUNCE, e.g. 0.01@2'); return 2; }
      plant.dropProb = Number(m[1]); plant.dropBounce = Number(m[2]);
    }
  }
  let env: (PtEnvOptions & { strengthScale?: number }) | undefined;
  if (args['env-nee'] || args['env-cap'] || args['env-no-floors'] || args['env-mis-power'] || args['env-plant'] || args['env-strength-scale']) {
    env = {};
    if (args['env-nee']) {
      if (!['on', 'off'].includes(args['env-nee'])) { console.error('--env-nee on|off'); return 2; }
      env.nee = args['env-nee'] === 'on';
    }
    if (args['env-cap']) env.importanceCap = Number(args['env-cap']);
    if (args['env-no-floors']) env.floors = false;
    if (args['env-mis-power']) env.misPower = true;
    if (args['env-plant']) {
      if (!['missingSin', 'w2WithoutPmf', 'doubleCount', 'pdfFromTargets'].includes(args['env-plant'])) { console.error(`--env-plant ${args['env-plant']}?`); return 2; }
      env.plant = args['env-plant'] as PtEnvPlant;
    }
    if (args['env-strength-scale']) env.strengthScale = Number(args['env-strength-scale']);
  }
  const restir = args.kernel === 'restir';
  // M6 (restir-m6-api.md §5): --restir-settings '{"pairing":"gauss","risNee":true,…}' overrides the preset's settings
  let rsSettings: Partial<RestirSettings> | undefined;
  if (args['restir-settings']) {
    try { rsSettings = JSON.parse(args['restir-settings']) as Partial<RestirSettings>; } catch { console.error('--restir-settings: JSON object'); return 2; }
  }
  const chains = restir && (args.chains !== undefined || args.mode === 'disocc');
  if (chains) {
    if (!['temporal', 'full', 'initial', 'initial-rr', 'offline', 'criteria2022', 'interactive', 'offline-m6', 'full-m6'].includes(args.preset!)) { console.error('--preset temporal|full|…'); return 2; }
    if (args.frames || args.scene || args.check || args.plant) { console.error('--chains: --frames/--scene/--check/--plant are not supported (use --chain-frames, --tplant, --u8-plant)'); return 2; }
  } else if (restir) {
    if (!['initial', 'initial-rr', 'offline', 'criteria2022', 'offline-m6'].includes(args.preset!)) { console.error('--preset initial|initial-rr|offline|criteria2022|offline-m6'); return 2; }
    if (args.plant && !['no-j', 'marginal-j'].includes(args.plant)) { console.error('--plant no-j|marginal-j'); return 2; }
    if (args.frames || args.scene || args.check) { console.error('--kernel restir: --frames/--scene/--check are not supported'); return 2; }
  }
  const own = shared ? undefined : await openHarness();
  let failures = 0;
  try {
    const { page, chromeVersion } = shared ?? own!;
    for (const frame of frames) {
      const runId = frame === undefined ? base : `${base}-f${frame}`;
      if (!shared) console.log(`acquiring GPU lock (${GPU_LOCK}) ...`);
      const releaseGpuLock = shared ? Object.assign(() => undefined, { waitedMs: 0, holder: 'jobs' }) : await acquireGpuLock('run-batches');
      const waited = releaseGpuLock.waitedMs;
      let rep: RenderBatchesReport;
      try {
        if (chains) {
          if (!pkgUrl) throw new Error('--chains needs --package');
          const list = (x?: string) => (x ? x.split(',').filter(Boolean) : undefined);
          const avg = args.average ? args.average.split(':').map(Number) : undefined;
          const co: RenderRestirChainsOptions = {
            run: runId, package: pkgUrl, preset: args.preset as RestirPresetName, chains: Number(args.chains ?? 1), seed, chromeVersion,
            batchOffset: args['batch-offset'] ? Number(args['batch-offset']) : undefined, chainBase: args['chain-base'] ? Number(args['chain-base']) : undefined,
            members: args.members ? Number(args.members) : undefined, frames: args['chain-frames'] ? Number(args['chain-frames']) : undefined,
            testFrames: list(args['test-frames'])?.map(Number), average: avg ? { from: avg[0], to: avg[1] } : undefined,
            masks: args.masks ? `/${path.relative(ROOT, path.resolve(ROOT, args.masks)).split(path.sep).join('/')}/` : undefined,
            temporalMis: args['temporal-mis'] as RenderRestirChainsOptions['temporalMis'], temporalCheck: args['temporal-check'] as RenderRestirChainsOptions['temporalCheck'],
            refresh: args.refresh as RenderRestirChainsOptions['refresh'], boostSlots: args.boost !== undefined ? Number(args.boost) : undefined,
            tPlants: list(args.tplant) as TPlantName[] | undefined, u8Plants: list(args['u8-plant']) as U8PlantName[] | undefined,
            wScale: args['w-scale'] !== undefined ? Number(args['w-scale']) : undefined,
            maxBounces: args['max-bounces'] !== undefined ? Number(args['max-bounces']) : undefined, env: env && env.nee !== undefined ? { nee: env.nee } : undefined,
            mode: args.mode as RenderRestirChainsOptions['mode'],
            settings: rsSettings, lightMode: args['light-mode'],
          };
          rep = await page.evaluate((x) => window.__harness!.renderRestirChains(x), co) as unknown as RenderBatchesReport;
        } else if (restir) {
          if (!pkgUrl) throw new Error('--kernel restir needs --package');
          rep = await page.evaluate((o) => window.__harness!.renderRestirBatches(o), {
            run: runId, package: pkgUrl, preset: args.preset as RestirPresetName, framesPerBatch: Number(args['frames-per-batch'] ?? args.spp),
            batches: Number(args.batches), batchOffset: args['batch-offset'] ? Number(args['batch-offset']) : undefined, seed, chromeVersion, members: args.members ? Number(args.members) : undefined,
            plant: args.plant as RestirPlantName | undefined, wScale: args['w-scale'] !== undefined ? Number(args['w-scale']) : undefined,
            maxBounces: args['max-bounces'] !== undefined ? Number(args['max-bounces']) : undefined, env: env && env.nee !== undefined ? { nee: env.nee } : undefined,
            settings: rsSettings, lightMode: args['light-mode'],
          });
        } else rep = await page.evaluate((o) => window.__harness!.renderBatches(o), {
          run: runId, package: pkgUrl, sceneUrl: args.scene, kernel: args.kernel as ValidationKernel, spp: Number(args.spp), batches: Number(args.batches),
          batchOffset: args['batch-offset'] ? Number(args['batch-offset']) : undefined, maxBounces: args['max-bounces'] !== undefined ? Number(args['max-bounces']) : undefined, rr: args.rr,
          technique: args.technique as PtTechnique | undefined, plant, lightMode: args['light-mode'], env,
          width: args.width ? Number(args.width) : undefined, height: args.height ? Number(args.height) : undefined, seed, chromeVersion, frame,
          // PT: cap the samples per dispatch (each dispatch is one submit) so a slowed-down GPU stays under the 200 ms hard cap
          ...(args['max-spp-per-dispatch'] ? { budget: { maxSamplesPerDispatch: Number(args['max-spp-per-dispatch']) } } : {}),
        });
      } finally {
        releaseGpuLock();
      }
      const m = rep.meta as { submits: { total: number; maxMs: number }; timings: { totalMs: number; batchMs: number[] }; counters: Record<string, number>; configHash: string };
      console.log(`${rep.ok ? 'OK  ' : 'FAIL'} ${runId}: ${rep.files.join(' ')}`);
      console.log(`     lock wait ${waited.toFixed(0)} ms, total ${m.timings.totalMs.toFixed(0)} ms, batches ${m.timings.batchMs.map((x) => x.toFixed(0)).join('/')} ms, ` +
        `${m.submits.total} submits (max ${m.submits.maxMs.toFixed(1)} ms), counters ${JSON.stringify(m.counters)}, configHash ${m.configHash.slice(0, 12)}`);
      if (!rep.ok) { failures++; console.log(`     errors: ${rep.errors.join('; ')}`); }
      const dir = path.join(OUT, runId);
      if (chains) {   // chain runs upload "<sub>__<file>" (single path components): move them to <sub>/<file>
        // large files arrive as "<name>.part###" (restir-chain-run.ts uploadParts): concatenate in order
        const parts = new Map<string, string[]>();
        for (const f of (await readdir(dir)).sort()) { const m = /^(.+)\.part\d{3}$/.exec(f); if (m) parts.set(m[1], [...(parts.get(m[1]) ?? []), f]); }
        for (const [name, ps] of parts) {
          await writeFile(path.join(dir, name), Buffer.concat(await Promise.all(ps.map((p) => readFile(path.join(dir, p))))));
          for (const p of ps) await rm(path.join(dir, p));
        }
        for (const f of await readdir(dir)) {
          const m = /^([\w.-]+)__(.+)$/.exec(f);
          if (!m) continue;
          await mkdir(path.join(dir, m[1]), { recursive: true });
          await rename(path.join(dir, f), path.join(dir, m[1], m[2]));
        }
        rep.files = rep.files.map((f) => f.replace('__', '/'));
      }
      for (const f of rep.files) if (!existsSync(path.join(dir, f))) { failures++; console.log(`     missing ${f}`); }
      // quick numeric summary of the mean image (ensemble runs write ensemble.npz instead)
      if (!existsSync(path.join(dir, 'mean.pfm'))) continue;
      const mean = decodePFM(new Uint8Array(await readFile(path.join(dir, 'mean.pfm'))));
      const ch = [0, 1, 2].map((c) => { let s = 0; for (let i = c; i < mean.data.length; i += 3) s += mean.data[i]; return s / (mean.data.length / 3); });
      console.log(`     mean image ${mean.width}x${mean.height}, channel means ${ch.map((x) => x.toPrecision(7)).join(', ')}`);
      if (args.check && pkgDir) {
        if (!existsSync(PYTHON) || !existsSync(MARKER_CHECK)) { console.log('     SKIP marker_check.py (venv or tool missing)'); continue; }
        // mean.pfm is gated (centroids, probes, stray mass, and mass ratios with a noise-aware tolerance from the
        // per-batch spread); every batch must match the exact parts (constant images, probes). One batch alone is
        // too noisy for the thresholded pole-marker mass (marker_check.py check_batch_dir).
        const r = await run(PYTHON, [MARKER_CHECK, '--batch-dir', dir, '--package', pkgDir, '--frame', String(frame ?? 0)]);
        console.log(`     marker_check mean.pfm + ${args.batches} batches: ${r.code === 0 ? 'PASS' : 'FAIL'}\n${r.out.trim().split('\n').map((l) => `       ${l}`).join('\n')}`);
        if (r.code !== 0) failures++;
      }
    }
  } finally {
    await own?.close();
  }
  console.log(failures ? `RESULT: FAIL (${failures})` : 'RESULT: PASS');
  return failures ? 1 : 0;
}

/** Vite (no HMR / watching) + headless Chrome on the harness page. */
async function openHarness(): Promise<SharedPage & { close(): Promise<void> }> {
  const port = await freePort();
  const vite: ViteDevServer = await createServer({
    // no HMR / file watching: a source edit elsewhere in the tree (another agent, an editor) must never reload the
    // harness page in the middle of a run ("Execution context was destroyed"); the page loads its modules once
    root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), server: { port, strictPort: true, host: '127.0.0.1', hmr: false, watch: null }, logLevel: 'warn',
    // a git worktree whose node_modules is a symlink to another checkout keeps its own dep-optimizer cache (never
    // rewrite the other checkout's node_modules/.vite while its jobs run)
    ...(worktreeCacheDir() ? { cacheDir: worktreeCacheDir() } : {}),
  });
  await vite.listen();
  let browser: Browser | undefined;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
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

/**
 * --jobs FILE (M5 gate, restir-temporal-api.md Changelog E-11): a JSON array of argument lists, each one ordinary
 * run-batches invocation, run on ONE harness page under ONE GPU-lock hold (the caller keeps the list's GPU time ≤ 12
 * min). Many small renders (mask references, pilots, disocclusion flags) otherwise pay one lock wait each.
 */
async function jobsMain(file: string): Promise<number> {
  const jobs = JSON.parse(await readFile(path.resolve(ROOT, file), 'utf8')) as string[][];
  const h = await openHarness();
  let failures = 0;
  console.log(`acquiring GPU lock (${GPU_LOCK}) for ${jobs.length} jobs ...`);
  const release = await acquireGpuLock('run-batches-jobs');
  console.log(`     lock wait ${release.waitedMs.toFixed(0)} ms`);
  try {
    for (const j of jobs) {
      args = parse(j);
      console.log(`--- job ${j.join(' ')}`);
      const c = await main(h).catch((e: unknown) => { console.error(e); return 1; });
      if (c) failures++;
    }
  } finally {
    release();
    await h.close();
  }
  console.log(failures ? `JOBS: FAIL (${failures}/${jobs.length})` : `JOBS: PASS (${jobs.length})`);
  return failures ? 1 : 0;
}

(args.jobs ? jobsMain(args.jobs) : main()).then((c) => process.exit(c), (e: unknown) => { console.error(e); process.exit(1); });
