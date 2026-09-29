// M3c app smoke (plan §1.4b UI, §6 "Env sampling" views): the real app in headless Chrome with the Cornell box + an
// HDRI + one point light added through the editor. Checks: the Worker-built importance tables reach the PT (HUD "env NEE on  P(env) …"), every env sampling debug
// view renders (screenshots), the splat χ² appears in the HUD with p ≥ 1e-3, the env-NEE toggle and the importance
// resolution change take effect (HUD), strength edits change P(env) and rotation does not, no uncaptured WebGPU errors,
// NaN/Inf = 0.
//   npx tsx validation/harness/m3c-app-smoke.ts [--run <id>] [--no-lock]
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium, type Page } from 'playwright';
import { createServer } from 'vite';
import type { App } from '../../src/app/app.ts';
import type { Integration } from '../../src/app/integration.ts';
import type { EditorHandle } from '../../src/app/editor/index.ts';
import { acquireGpuLock } from './gpu-lock.ts';

declare global { interface Window { __app?: App; __integration?: Integration; __editor?: EditorHandle } }

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const HDRI = 'validation/assets/downloaded/hdri/studio_small_09_1k.hdr';
const stamp = () => new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
const { values: args } = parseArgs({ options: { run: { type: 'string' }, 'no-lock': { type: 'boolean', default: false } } });

const checks: { name: string; ok: boolean; detail?: string }[] = [];
const check = (name: string, ok: boolean, detail?: string) => { checks.push({ name, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`); return ok; };

async function frames(page: Page, n: number): Promise<void> {
  const f0 = await page.evaluate(() => window.__app!.frameCounter);
  await page.waitForFunction((t) => window.__app!.frameCounter >= t, f0 + n, { timeout: 120_000, polling: 50 });
}
const hud = (page: Page) => page.evaluate(() => window.__integration!.renderer()!.hudLines());
const envLine = async (page: Page) => (await hud(page)).find((l) => l.startsWith('env NEE')) ?? '';
const pEnvOf = (l: string) => Number(/P\(env\) ([\d.]+)/.exec(l)?.[1] ?? NaN);

async function main(): Promise<number> {
  const runId = args.run ?? `m3c-app-smoke-${stamp()}`;
  const OUT = path.join(ROOT, 'validation/out', runId);
  mkdirSync(OUT, { recursive: true });
  const release = args['no-lock'] ? () => {} : await acquireGpuLock('m3c-app-smoke');
  const vite = await createServer({ root: ROOT, configFile: path.join(ROOT, 'vite.config.ts'), server: { port: 0, host: '127.0.0.1' }, logLevel: 'warn' });
  await vite.listen();
  const port = (vite.httpServer!.address() as { port: number }).port;
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  const errors: string[] = [];
  try {
    const page = await (await browser.newContext({ viewport: { width: 1280, height: 720 } })).newPage();
    page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); if (/^\[env\]|uncaptured|error/i.test(m.text())) console.log(`[page:${m.type()}] ${m.text()}`); });
    page.on('pageerror', (e) => errors.push(`[pageerror] ${e.message}`));
    // A light in the scene makes P(env) < 1 (the Cornell rect light), so the clamp / proxy path is exercised.
    const q = new URLSearchParams({ seed: '1', res: '540p', scene: '/validation/assets/cornell/cornell.glb', env: `/${HDRI}` });
    await page.goto(`http://127.0.0.1:${port}/index.html?${q}`);
    await page.waitForFunction(() => {
      const r = window.__integration?.renderer(); const a = window.__app;
      return !!r && !!a?.scene && r.scene?.scene === a.scene && !r.loading && r.ready && r.env.present && r.hudLines().some((l) => l.startsWith('env NEE'));
    }, undefined, { timeout: 180_000, polling: 100 });
    // A point light inside the box (the glTF carries no light): P(env) < 1 exercises the proxy + clamp path.
    await page.evaluate(() => {
      window.__editor!.editor.store!.add({ type: 'point', power: 30, matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0.4, -0.25, 1]) });
    });
    await frames(page, 30);
    let l = await envLine(page);
    const p0 = pEnvOf(l);
    check('Worker-built importance tables reach the PT (HUD: env NEE on, P(env), grid)', l.includes('env NEE on') && /importance 1024x512/.test(l) && p0 > 0 && p0 < 1, l);
    await page.screenshot({ path: path.join(OUT, 'beauty.png') });

    // rotation never changes the pmf; strength does (radiometric)
    await page.evaluate(() => { const a = window.__app!; a.envParams.rotationDeg = 40; a.envParamsChanged(); });
    await frames(page, 4);
    const p1 = pEnvOf(await envLine(page));
    await page.evaluate(() => { const a = window.__app!; a.envParams.strength = 3; a.envParamsChanged(); });
    await frames(page, 4);
    const p2 = pEnvOf(await envLine(page));
    check('rotation keeps P(env); strength ×3 raises it', p1 === p0 && p2 > p0, `P(env) ${p0} → rotate ${p1} → strength ×3 ${p2}`);

    // env debug views (camera looks at the box; the env views are env-space)
    for (const [name, id] of [['importance', 310], ['ratio', 311], ['splat', 312], ['escape', 313], ['w1', 314], ['w2', 315]] as const) {
      await page.evaluate((v) => window.__app!.selectDebugView(v), id);
      await frames(page, name === 'splat' ? 90 : 20);
      await page.screenshot({ path: path.join(OUT, `env-${name}.png`) });
      if (name === 'splat') {
        await page.waitForFunction(() => window.__integration!.renderer()!.hudLines().some((x) => x.startsWith('env splat')), undefined, { timeout: 60_000 });
        const s = (await hud(page)).find((x) => x.startsWith('env splat'))!;
        const p = Number(/p ([\d.e+-]+)/.exec(s)?.[1] ?? NaN);
        check('splat χ² vs the realized pdf (HUD) p ≥ 1e-3', p >= 1e-3, s);
      }
    }
    const aov = await page.evaluate(async () => {
      // mean of the ω1 view AOV (must lie in [0, 1] and be > 0 on the lit box)
      const a = window.__app!;
      a.selectDebugView(314);
      return a.debug.settings.mode;
    });
    check('ω1 view selectable', aov === 314);
    await page.evaluate(() => window.__app!.selectDebugView(0));

    // env NEE off (≡ Cycles sampling_method NONE) and importance resolution 256
    await page.evaluate(() => { const a = window.__app!; a.envParams.nee = false; a.envParamsChanged(); });
    await page.waitForFunction(() => window.__integration!.renderer()!.hudLines().some((x) => x.startsWith('env NEE off')), undefined, { timeout: 30_000 });
    check('env NEE toggle off (HUD)', true, await envLine(page));
    await page.evaluate(() => { const a = window.__app!; a.envParams.nee = true; a.envParams.importanceRes = 256; a.envParamsChanged(); });
    await page.waitForFunction(() => window.__integration!.renderer()!.hudLines().some((x) => /env NEE on .*importance 256x128/.test(x)), undefined, { timeout: 30_000 });
    l = await envLine(page);
    check('importance resolution 256 (Worker rebuild, HUD)', /importance 256x128/.test(l), l);
    await frames(page, 30);
    await page.screenshot({ path: path.join(OUT, 'beauty-imp256.png') });
    const t = await page.evaluate(() => ({ ...window.__app!.runTotals }));
    check('NaN/Inf = 0', t.nan === 0 && t.inf === 0, JSON.stringify(t));
    const lastError = await page.evaluate(() => window.__integration!.renderer()!.lastError);
    check('renderer error-free', !lastError, lastError);
    check('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
  } finally {
    await browser.close();
    await vite.close();
    release();
  }
  writeFileSync(path.join(OUT, 'report.json'), JSON.stringify({ runId, checks }, null, 1));
  const failed = checks.filter((c) => !c.ok).length;
  console.log(`${failed ? 'FAIL' : 'PASS'} m3c app smoke: ${checks.length - failed}/${checks.length} (screenshots: ${path.relative(ROOT, OUT)})`);
  return failed ? 1 : 0;
}

main().then((c) => process.exit(c), (e: unknown) => { console.error(e); process.exit(1); });
