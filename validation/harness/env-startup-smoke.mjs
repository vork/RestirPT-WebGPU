// Production-bundle startup and live HDRI replacement regression.
// npm run build && node scripts/pages-assets.mjs
// npx tsx validation/harness/with-gpu-lock.ts env-startup -- node validation/harness/env-startup-smoke.mjs
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { preview } from 'vite';
import { chromium } from 'playwright';

const out = 'validation/out/env-startup';
mkdirSync(out, { recursive: true });
const server = await preview({ preview: { port: 0, host: '127.0.0.1' }, logLevel: 'warn' });
const browser = await chromium.launch({ channel: 'chrome', headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const errors = [], results = [];
page.on('pageerror', e => errors.push(String(e)));
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
const root = server.resolvedUrls.local[0];
const env = new URL('validation/assets/downloaded/hdri/kloofendal_48d_partly_cloudy_puresky_1k.hdr', root).href;
async function check(label) {
  await page.waitForFunction(() => {
    const r = window.__integration?.renderer();
    return r?.ready && r.env.present && r.restir?.ready && r.denoisedLastFrame;
  }, undefined, { timeout: 120000 });
  const first = await page.evaluate(() => window.__app.frameCounter);
  await page.waitForFunction(f => window.__app.frameCounter > f + 12, first, { timeout: 60000 });
  const state = await page.evaluate(async () => {
    const a = window.__app, r = window.__integration.renderer(), d = a.device, t = a.targets;
    const bytesPerRow = Math.ceil(t.width * 16 / 256) * 256;
    const b = d.createBuffer({ size: bytesPerRow * t.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = d.createCommandEncoder();
    e.copyTextureToBuffer({ texture: t.color }, { buffer: b, bytesPerRow }, [t.width, t.height]);
    d.queue.submit([e.finish()]); await b.mapAsync(GPUMapMode.READ);
    const pixels = new Float32Array(b.getMappedRange());
    let sum = 0, nonFinite = 0;
    for (let y = 0; y < t.height; y++) for (let x = 0; x < t.width; x++) for (let c = 0; c < 3; c++) {
      const v = pixels[y * bytesPerRow / 4 + x * 4 + c];
      if (Number.isFinite(v)) sum += v; else nonFinite++;
    }
    b.unmap(); b.destroy();
    return { mean: sum / (t.width * t.height * 3), nonFinite, errors: window.__webgpuErrors,
      rendererError: r.lastError, restirError: r.restirError, currentEnvironment: r.restir.kernel.envResources === r.env,
      mode: r.options.restirMode, filter: a.present.filter };
  });
  assert.deepEqual(state.errors, []); assert.deepEqual(errors, []);
  assert.equal(state.rendererError, undefined); assert.equal(state.restirError, undefined);
  assert.equal(state.nonFinite, 0); assert.ok(state.mean > 0); assert.equal(state.currentEnvironment, true);
  results.push({ label, ...state });
  await page.locator('#view').screenshot({ path: `${out}/${label}.png` });
  console.log(`PASS ${label}: mean ${state.mean.toFixed(4)}, no GPU errors`);
}
try {
  for (const [label, scene] of [
    ['cornell', 'validation/assets/cornell/cornell_point_spot.glb'],
    ['sponza', 'validation/assets/downloaded/sponza/Sponza.gltf'],
  ]) {
    const url = new URL(root);
    url.search = new URLSearchParams({ scene: new URL(scene, root).href, env, seed: '7' }).toString();
    await page.goto(url.href);
    await check(`${label}-startup`);
    assert.equal(results.at(-1).mode, 'interactive'); assert.equal(results.at(-1).filter, 'fsr1');
    await page.evaluate(async env => {
      const a = window.__app, r = window.__integration.renderer();
      await r.setOptions({ restirMode: 'potato', restirFeatures: { indirectCache: 2 } });
      await a.loadEnvironment({ kind: 'url', url: env, name: 'replacement.hdr' });
      a.resetHistory();
    }, env);
    await check(`${label}-replacement`);
  }
  writeFileSync(`${out}/results.json`, JSON.stringify({ ok: true, chrome: browser.version(), results, errors }, null, 2));
} finally {
  await browser.close(); await server.httpServer.close();
}
