// M2 app smoke: the real app in headless Chrome loads the Cornell USD (LightUSD worker) + an HDRI (validation env
// mode), renders frames without WebGPU errors, then "Export for Cycles" writes a scene package to
// validation/out/export-<id>/, which is read back in Node (readScenePackage verifies the env pixel hash).
//   npx tsx validation/harness/m2-app-export.ts [--scene /validation/assets/cornell/cornell.usda] [--env <url>]
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium } from 'playwright';
import { createServer } from 'vite';
import type { App } from '../../src/app/app.ts';
import type { Integration } from '../../src/app/integration.ts';
import { readScenePackage } from '../../src/core/scene/scene-package.ts';
import { acquireGpuLock } from './gpu-lock.ts';

declare global { interface Window { __app?: App; __integration?: Integration; __webgpuErrors?: string[] } }

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const HDRI = 'validation/assets/downloaded/hdri/kloofendal_48d_partly_cloudy_puresky_1k.hdr';
const { values: args } = parseArgs({ options: { scene: { type: 'string', default: '/validation/assets/cornell/cornell.usda' }, env: { type: 'string' } } });

const checks: { name: string; ok: boolean }[] = [];
const check = (name: string, ok: boolean, detail?: string) => { checks.push({ name, ok }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`); return ok; };

async function main(): Promise<number> {
  const env = args.env ?? (existsSync(path.join(ROOT, HDRI)) ? `/${HDRI}` : undefined);
  const vite = await createServer({ root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), server: { port: 0, host: '127.0.0.1' }, logLevel: 'warn' });
  await vite.listen();
  const port = (vite.httpServer!.address() as { port: number }).port;
  const release = await acquireGpuLock('m2-app-export');
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  let exported: string | undefined;
  try {
    const page = await browser.newPage({ viewport: { width: 800, height: 600 } });
    const logs: string[] = [];
    page.on('console', (m) => { logs.push(`[${m.type()}] ${m.text()}`); if (m.type() === 'error') console.log(`[page:error] ${m.text()}`); });
    page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
    const q = new URLSearchParams({ scene: args.scene!, ...(env ? { env } : {}), seed: '1' });
    await page.goto(`http://127.0.0.1:${port}/?${q}`);
    await page.waitForFunction(() => !!window.__integration && !!window.__app, undefined, { timeout: 60_000 });
    await page.evaluate(() => window.__integration!.sceneReady);
    if (env) await page.waitForFunction(() => !!window.__app!.env, undefined, { timeout: 60_000 });
    const f0 = await page.evaluate(() => window.__app!.frameCounter);
    await page.waitForFunction((t) => window.__app!.frameCounter >= t, f0 + 30, { timeout: 60_000 });
    const st = await page.evaluate(() => ({
      name: window.__app!.scene?.name, tris: window.__app!.scene?.geometry.indices.length, lights: window.__app!.scene?.lights.length,
      env: window.__app!.env ? `${window.__app!.env.width}x${window.__app!.env.height}` : null, errors: window.__webgpuErrors ?? [], totals: window.__app!.runTotals,
    }));
    check('USD scene loaded in the app', (st.tris ?? 0) > 0, `${st.name}: ${(st.tris ?? 0) / 3} tris, ${st.lights} lights`);
    check('USD loaded through the worker', logs.some((l) => /\[scene\].*upAxis/.test(l)));
    if (env) check('env loaded in validation mode', logs.some((l) => /\[env\].*validation mode/.test(l)), st.env ?? '');
    check('no uncaptured WebGPU errors', st.errors.length === 0, st.errors.slice(0, 3).join(' | '));
    check('NaN/Inf = 0, BVH overflow = 0', st.totals.nan + st.totals.inf + st.totals.bvhOverflow + st.totals.bvhItercap === 0, JSON.stringify(st.totals));
    await page.screenshot({ path: path.join(ROOT, 'validation/out/m2-app-export.png') });
    exported = await page.evaluate(() => window.__integration!.exportForCycles(window.__app!, { width: 256, height: 256, maxBounces: 3, lightMode: 'A' }));
    check('Export for Cycles wrote a package', !!exported, exported);
  } finally {
    await browser.close();
    release();
    await vite.close();
  }
  if (exported) {
    const dir = path.join(ROOT, exported);
    const files = Object.fromEntries(readdirSync(dir).map((f) => [f, new Uint8Array(readFileSync(path.join(dir, f)))]));
    try {
      const p = await readScenePackage(files);
      check('exported package reads back (env hash verified)', true,
        `${Object.keys(files).join(', ')}; ${p.scene.geometry.indices.length / 3} tris, ${p.scene.lights.length} lights, env ${p.json.env ? `${p.json.env.width}x${p.json.env.height} sha ${p.json.env.sha256.slice(0, 12)}` : 'none'}, flatShaded ${p.flatShaded}`);
    } catch (e) {
      check('exported package reads back (env hash verified)', false, String(e));
    }
  }
  const ok = checks.every((c) => c.ok);
  console.log(ok ? 'RESULT: PASS' : 'RESULT: FAIL');
  return ok ? 0 : 1;
}

main().then((c) => process.exit(c), (e: unknown) => { console.error(e); process.exit(1); });
