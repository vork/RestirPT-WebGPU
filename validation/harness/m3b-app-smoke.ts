// M3b app smoke (plan §5 M3b; task "App: Mode A/B/A′ selector and glass materials render interactively"): writes a
// small GLB with KHR_materials_transmission / KHR_materials_ior glass (smooth sphere, rough cube) and a roughness-0
// metal mirror floor tile, opens the real app in headless Chrome, adds a rect light above the glass through the light
// editor API, and renders the interactive PT (PtFramePass, Möller–Trumbore) in light modes A, B and A′ (the Renderer
// panel's "light mode" option). Checks: no renderer / WebGPU errors, NaN/Inf and BVH counters 0, max bounces raised to
// ≥ 4 for a glass scene, the image changes between modes (B/A′ show the light in the mirror and the caustic, A does
// not: mean(B) > mean(A)), and A′ ≈ B. Saves screenshots to validation/out/<run>/.
//   npx tsx validation/harness/m3b-app-smoke.ts [--run <id>]
import { mkdirSync, rmdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { Document, NodeIO } from '@gltf-transform/core';
import { KHRMaterialsIOR, KHRMaterialsTransmission } from '@gltf-transform/extensions';
import { chromium, type Page } from 'playwright';
import { createServer } from 'vite';
import type { App } from '../../src/app/app.ts';
import type { EditorHandle } from '../../src/app/editor/index.ts';
import type { Integration } from '../../src/app/integration.ts';
import { decodePng } from '../../src/core/io/png.ts';

declare global { interface Window { __app?: App; __integration?: Integration; __editor?: EditorHandle } }

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const LOCK = '/tmp/restirpt-gpu.lock';
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const { values: args } = parseArgs({ options: { run: { type: 'string' } } });
const runId = args.run ?? `m3b-app-smoke-${stamp()}`;
const OUT = path.join(ROOT, 'validation/out', runId);

const checks: { name: string; ok: boolean; detail?: string }[] = [];
const check = (name: string, ok: boolean, detail?: string) => { checks.push({ name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`); return ok; };

/** Flat-shaded mesh from quads / triangles. */
async function writeGlassGlb(file: string): Promise<void> {
  const doc = new Document();
  const buf = doc.createBuffer();
  const tr = doc.createExtension(KHRMaterialsTransmission);
  const ior = doc.createExtension(KHRMaterialsIOR);
  const scene = doc.createScene('glass-smoke');
  const mesh = (name: string, tris: number[][][], mat: ReturnType<Document['createMaterial']>) => {
    const pos: number[] = [], nrm: number[] = [];
    for (const [a, b, c] of tris) {
      const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
      const l = Math.hypot(...n);
      for (const p of [a, b, c]) { pos.push(...p); nrm.push(n[0] / l, n[1] / l, n[2] / l); }
    }
    const prim = doc.createPrimitive().setMaterial(mat)
      .setAttribute('POSITION', doc.createAccessor().setType('VEC3').setArray(new Float32Array(pos)).setBuffer(buf))
      .setAttribute('NORMAL', doc.createAccessor().setType('VEC3').setArray(new Float32Array(nrm)).setBuffer(buf));
    scene.addChild(doc.createNode(name).setMesh(doc.createMesh(name).addPrimitive(prim)));
  };
  const quad = (p0: number[], p1: number[], p2: number[], p3: number[]) => [[p0, p1, p2], [p0, p2, p3]];
  const box = (a: number[], b: number[]) => {
    const [x0, y0, z0] = a, [x1, y1, z1] = b;
    return [
      ...quad([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]), ...quad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]),
      ...quad([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]), ...quad([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]),
      ...quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]), ...quad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]),
    ];
  };
  const floor = doc.createMaterial('floor').setBaseColorFactor([0.7, 0.7, 0.7, 1]).setMetallicFactor(0).setRoughnessFactor(1);
  const mirror = doc.createMaterial('mirror').setBaseColorFactor([0.95, 0.95, 0.95, 1]).setMetallicFactor(1).setRoughnessFactor(0);
  const glass = doc.createMaterial('glass').setBaseColorFactor([1, 1, 1, 1]).setMetallicFactor(0).setRoughnessFactor(0)
    .setExtension('KHR_materials_transmission', tr.createTransmission().setTransmissionFactor(1))
    .setExtension('KHR_materials_ior', ior.createIOR().setIOR(1.5));
  const rough = doc.createMaterial('rough_glass').setBaseColorFactor([0.9, 0.95, 1, 1]).setMetallicFactor(0).setRoughnessFactor(0.3)
    .setExtension('KHR_materials_transmission', tr.createTransmission().setTransmissionFactor(1))
    .setExtension('KHR_materials_ior', ior.createIOR().setIOR(1.45));
  mesh('floor', [...quad([-3, 0, 3], [3, 0, 3], [3, 0, -3], [-3, 0, -3]), ...quad([-3, 0, -3], [3, 0, -3], [3, 2.5, -3], [-3, 2.5, -3])], floor);
  mesh('mirror', quad([-2.5, 0.002, 2], [-0.5, 0.002, 2], [-0.5, 0.002, 0], [-2.5, 0.002, 0]), mirror);
  // flat icosahedron-ish "sphere": an octahedron subdivided twice is enough for a smoke test
  const octa = (c: number[], r: number) => {
    let faces: number[][][] = [];
    const v = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
    for (const [a, b, d] of [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]) faces.push([v[a], v[b], v[d]]);
    const nz = (p: number[]) => { const l = Math.hypot(...p); return p.map((x) => x / l); };
    for (let s = 0; s < 3; s++) {
      const next: number[][][] = [];
      for (const [a, b, d] of faces) {
        const ab = nz(a.map((x, i) => x + b[i])), bd = nz(b.map((x, i) => x + d[i])), da = nz(d.map((x, i) => x + a[i]));
        next.push([a, ab, da], [ab, b, bd], [da, bd, d], [ab, bd, da]);
      }
      faces = next;
    }
    return faces.map((f) => f.map((p) => p.map((x, i) => c[i] + r * x)));
  };
  mesh('glass_sphere', octa([0.4, 0.55, -0.4], 0.5), glass);
  mesh('rough_cube', box([1.3, 0.001, 0.2], [2.1, 0.8, 1.0]), rough);
  await new NodeIO().registerExtensions([KHRMaterialsTransmission, KHRMaterialsIOR]).write(file, doc);
}

async function frames(page: Page, n: number): Promise<void> {
  const f0 = await page.evaluate(() => window.__app!.frameCounter);
  await page.waitForFunction((t) => window.__app!.frameCounter >= t, f0 + n, { timeout: 120_000, polling: 30 });
}

/** Mean sRGB luminance of the viewport canvas (screenshot of #view). */
async function shotMean(page: Page, file: string): Promise<number> {
  const png = await page.locator('#view').screenshot({ path: file });
  const img = await decodePng(new Uint8Array(png));
  let s = 0;
  for (let i = 0; i < img.pixels.length; i += 4) s += 0.2126 * img.pixels[i] + 0.7152 * img.pixels[i + 1] + 0.0722 * img.pixels[i + 2];
  return s / (img.pixels.length / 4);
}

async function main(): Promise<number> {
  mkdirSync(OUT, { recursive: true });
  const glb = path.join(OUT, 'glass-smoke.glb');
  await writeGlassGlb(glb);
  let locked = false;
  const release = () => { if (locked) { try { rmdirSync(LOCK); } catch { /* gone */ } locked = false; } };
  process.on('exit', release);
  for (;;) { try { mkdirSync(LOCK); locked = true; break; } catch { await sleep(3000); } }
  const vite = await createServer({ root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), server: { port: 0, host: '127.0.0.1' }, logLevel: 'warn' });
  await vite.listen();
  const port = (vite.httpServer!.address() as { port: number }).port;
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const errors: string[] = [];
  const report: Record<string, unknown> = { runId, checks };
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
    page.on('console', (m) => { if (m.type() === 'error') { errors.push(m.text()); console.log(`[page:error] ${m.text()}`); } });
    page.on('pageerror', (e) => { errors.push(e.message); console.log(`[pageerror] ${e.message}`); });
    await page.addInitScript(() => { (globalThis as unknown as { __name: (f: unknown) => unknown }).__name = (f) => f; });
    const sceneUrl = `/${path.relative(ROOT, glb).split(path.sep).join('/')}`;
    await page.goto(`http://127.0.0.1:${port}/?seed=1&hud=0&res=540p&scene=${encodeURIComponent(sceneUrl)}`);
    await page.waitForFunction(() => !!window.__integration && !!window.__app && !!window.__editor && !!window.__app.scene, undefined, { timeout: 120_000 });
    await page.evaluate(() => window.__integration!.sceneReady);
    await page.waitForFunction(() => { const r = window.__integration!.renderer(); return !!r && !r.loading && r.ready; }, undefined, { timeout: 120_000 });
    const info = await page.evaluate(() => {
      const a = window.__app!, r = window.__integration!.renderer()!;
      return { scene: a.scene!.name, mats: a.scene!.materials.map((m) => [m.name, m.transmissionFactor, m.ior]), maxBounces: r.options.maxBounces, err: r.lastError };
    });
    report.scene = info;
    check('glass GLB loaded (KHR_materials_transmission / ior)', info.mats.some((m) => m[1] === 1), JSON.stringify(info.mats));
    check('max bounces raised to ≥ 4 for a glass scene (plan §1.10)', info.maxBounces >= 4, `${info.maxBounces}`);
    // camera + a rect light above the glass sphere (emitting down) through the light-editor API
    await page.evaluate(() => {
      const a = window.__app!;
      a.camera.lookAt([0.2, 2.3, 4.2], [0.2, 0.3, -0.3]);
      a.camera.initialPose = a.camera.pose();
      const m = new Float32Array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0.3, 2.2, -0.4, 1]);   // local −Z = world −Y
      window.__editor!.editor.addLight({ type: 'rect', matrix: m, name: 'Rect.smoke', power: 60, sizeX: 0.6, sizeY: 0.6 });
      window.__editor!.editor.select(undefined);
      const r = window.__integration!.renderer()!;
      r.options.renderMode = 'pt';
    });
    const means: Record<string, number> = {};
    for (const mode of ['A', 'B', 'A′'] as const) {
      await page.evaluate(async (m) => { const r = window.__integration!.renderer()!; await r.setOptions({ lightMode: m }); window.__app!.resetHistory(); }, mode);
      await frames(page, 160);
      const st = await page.evaluate(() => {
        const a = window.__app!, r = window.__integration!.renderer()!;
        return { totals: { ...a.runTotals }, err: r.lastError, mode: (r as unknown as { state?: { pt?: { lights: { lightMode: string } } } }).state?.pt?.lights.lightMode, hud: r.hudLines() };
      });
      means[mode] = await shotMean(page, path.join(OUT, `mode-${mode === 'A′' ? 'Aprime' : mode}.png`));
      check(`mode ${mode}: renderer error-free, PT light mode applied`, !st.err && st.mode === mode, `${st.err ?? ''} ${st.mode}`);
      check(`mode ${mode}: NaN/Inf 0, BVH overflow/itercap 0`, st.totals.nan === 0 && st.totals.inf === 0 && st.totals.bvhOverflow === 0 && st.totals.bvhItercap === 0, JSON.stringify(st.totals));
    }
    report.means = means;
    check('Mode B shows the area light in the mirror / through smooth glass: mean(B) > mean(A)', means.B > means.A * 1.01, `A ${means.A.toFixed(2)}, B ${means.B.toFixed(2)}`);
    check('Mode A′ ≈ Mode B (same expectation)', Math.abs(means['A′'] - means.B) < 0.03 * means.B, `A′ ${means['A′'].toFixed(2)}, B ${means.B.toFixed(2)}`);
    check('no page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  } finally {
    await browser.close();
    await vite.close();
    release();
  }
  report.ok = checks.every((c) => c.ok);
  writeFileSync(path.join(OUT, 'report.json'), JSON.stringify(report, null, 1));
  console.log(`${report.ok ? 'RESULT: PASS' : 'RESULT: FAIL'} (${OUT})`);
  return report.ok ? 0 : 1;
}

main().then((c) => process.exit(c), (e: unknown) => { console.error(e); process.exit(1); });
