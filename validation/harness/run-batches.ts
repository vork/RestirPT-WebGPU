// Batch validation renders in headless Chrome 154 (plan §5 M2): starts Vite, opens the harness page, and calls
// window.__harness.renderBatches under the shared GPU lock. Output: validation/out/<run>/batch_###.pfm, mean.pfm,
// meta.json. With --check, runs validation/tools/marker_check.py on mean.pfm against the package's "expected".
//   npx tsx validation/harness/run-batches.ts --package validation/scenes/c0b_512 --spp 64 --batches 4 --check
//   npx tsx validation/harness/run-batches.ts --package validation/scenes/c0p_512 --frames 0,1,2 --check
//   npx tsx validation/harness/run-batches.ts --scene /validation/assets/cornell/cornell.usda --spp 16 --batches 2
//   npx tsx validation/harness/run-batches.ts --package validation/scenes/cornell_i_512 --kernel pt --spp 256 --batches 16
//     (M3a reference PT; --max-bounces N overrides the package's render.maxBounces, --rr, --technique mis|nee|bsdf,
//      Gate-1 planted biases: --plant-emit-scale 1.01 (every emitter ×s), --plant-drop 0.01@2 (terminate 1% of the
//      paths at vertex 2, no compensation))
// If the package directory is missing and --make-c0b is given, an equivalent C0b package (calib_scenes.py make_c0b:
// 100 m emissive quad at z = −2, L_e = (0.5, 0.25, 0.125)·2, vfov 40°, 512²) is written with exportScenePackage to
// validation/out/tmp-c0b/ and rendered instead.
import { execFile } from 'node:child_process';
import { existsSync, mkdirSync, rmdirSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
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
import type { PtPlant, PtTechnique } from '../../src/core/render/pt-kernel.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const OUT = path.join(ROOT, 'validation/out');
const PYTHON = path.join(ROOT, 'validation/.venv/bin/python');
const MARKER_CHECK = path.join(ROOT, 'validation/tools/marker_check.py');
const GPU_LOCK = '/tmp/restirpt-gpu.lock';

const { values: args } = parseArgs({
  options: {
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
  },
});

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');

let lockHeld = false;
async function acquireGpuLock(): Promise<number> {
  const t0 = performance.now();
  for (;;) {
    try { mkdirSync(GPU_LOCK); lockHeld = true; return performance.now() - t0; } catch { await sleep(5000); }
  }
}
function releaseGpuLock(): void {
  if (!lockHeld) return;
  try { rmdirSync(GPU_LOCK); } catch { /* gone */ }
  lockHeld = false;
}
for (const sig of ['SIGINT', 'SIGTERM'] as const) process.on(sig, () => { releaseGpuLock(); process.exit(130); });
process.on('exit', releaseGpuLock);

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
  const pkg = await exportScenePackage(scene, {
    camera: { matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], yfov: 40 * Math.PI / 180 },
    render: { width: 512, height: 512, maxBounces: 0 }, lightMode: 'A', flatShaded: true, name: 'c0b_512',
    source: { uri: 'validation/harness/run-batches.ts makeC0b' },
  });
  const json = { ...pkg.json, expected: { kind: 'constant', value: [1, 0.5, 0.25], tolerance: 1e-4, frames: { 0: {} } } };
  pkg.files.set('scene.json', new TextEncoder().encode(JSON.stringify(json, null, 1)));
  await mkdir(dir, { recursive: true });
  for (const [k, v] of pkg.files) await writeFile(path.join(dir, k), v);
}

async function main(): Promise<number> {
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
  if (args['plant-emit-scale'] || args['plant-drop']) {
    plant = {};
    if (args['plant-emit-scale']) plant.emitScale = Number(args['plant-emit-scale']);
    if (args['plant-drop']) {
      const m = /^([\d.eE+-]+)@(\d+)$/.exec(args['plant-drop']);
      if (!m) { console.error('--plant-drop PROB@BOUNCE, e.g. 0.01@2'); return 2; }
      plant.dropProb = Number(m[1]); plant.dropBounce = Number(m[2]);
    }
  }
  const port = await freePort();
  const vite: ViteDevServer = await createServer({
    root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), server: { port, strictPort: true, host: '127.0.0.1' }, logLevel: 'warn',
  });
  await vite.listen();
  let browser: Browser | undefined;
  let failures = 0;
  try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const chromeVersion = browser.version();
    const page = await browser.newPage();
    page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log(`[page:${m.type()}] ${m.text()}`); });
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
    await page.goto(`http://127.0.0.1:${port}/validation/harness/harness.html`);
    await page.waitForFunction(() => window.__harness !== undefined, undefined, { timeout: 60_000 });
    for (const frame of frames) {
      const runId = frame === undefined ? base : `${base}-f${frame}`;
      console.log(`acquiring GPU lock (${GPU_LOCK}) ...`);
      const waited = await acquireGpuLock();
      let rep: RenderBatchesReport;
      try {
        rep = await page.evaluate((o) => window.__harness!.renderBatches(o), {
          run: runId, package: pkgUrl, sceneUrl: args.scene, kernel: args.kernel as ValidationKernel, spp: Number(args.spp), batches: Number(args.batches),
          maxBounces: args['max-bounces'] !== undefined ? Number(args['max-bounces']) : undefined, rr: args.rr,
          technique: args.technique as PtTechnique | undefined, plant,
          width: args.width ? Number(args.width) : undefined, height: args.height ? Number(args.height) : undefined, seed, chromeVersion, frame,
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
      for (const f of rep.files) if (!existsSync(path.join(dir, f))) { failures++; console.log(`     missing ${f}`); }
      // quick numeric summary of the mean image
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
    await browser?.close();
    await vite.close();
  }
  console.log(failures ? `RESULT: FAIL (${failures})` : 'RESULT: PASS');
  return failures ? 1 : 0;
}

main().then((c) => { releaseGpuLock(); process.exit(c); }, (e: unknown) => { releaseGpuLock(); console.error(e); process.exit(1); });
