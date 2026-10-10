import { afterAll, expect, it } from 'vitest';
import { getTestGpu, releaseTestGpu } from './device-factory.ts';
import { DebugResources, DebugViewRegistry, defaultDebugSettings } from '../../src/core/render/debug-views.ts';
import { FRAME_RESET_HISTORY, FrameUniformBuffer, JITTER_IID, boundsDiagonal, computeRenderOrigin } from '../../src/core/render/frame-uniforms.ts';
import { Renderer, type RestirAppMode } from '../../src/core/render/renderer.ts';
import { allLightsScene, boxCamera, hashF32 } from './restir-fixtures.ts';

afterAll(async () => { await releaseTestGpu(); });

// Exercise actual resource reallocation, shader variants, denoiser mode defaults and history resets, rather than
// only merging settings. Returning to a mode must reproduce a fresh renderer at the same seeds, pixel for pixel.
async function run(modes: RestirAppMode[]): Promise<string[]> {
  const ctx = await getTestGpu(), { device } = ctx;
  const W = 32, H = 24, scene = allLightsScene(), origin = computeRenderOrigin(scene.bounds, scene.quant);
  const debug = new DebugResources(device, new DebugViewRegistry());
  await debug.init(); debug.resize(W, H);
  const r = await Renderer.create({ device, debugLayout: debug.layout, debug, features: ctx.features, wgslLanguageFeatures: ctx.wgslLanguageFeatures },
    { renderMode: 'restir', restirMode: modes[0], watertight: false, lightMode: 'B', bvhKind: 'auto', maxBounces: 3, accumulate: false });
  const fu = new FrameUniformBuffer(device);
  const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
  const color = device.createTexture({ size: [W, H], format: 'rgba32float', usage });
  const depth = device.createTexture({ size: [W, H], format: 'r32float', usage });
  const readback = device.createBuffer({ size: W * H * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const hashes: string[] = [], errors: string[] = [];
  const onError = (e: GPUUncapturedErrorEvent) => errors.push(e.error.message);
  device.addEventListener('uncapturederror', onError);
  try {
    if (scene.env) await r.setEnvironment(scene.env);
    expect(await r.setScene(scene, origin)).toBeTruthy();
    r.resize({ width: W, height: H, color, colorFormat: 'rgba32float', depth, frameUniforms: fu.buffer });
    const cam = boxCamera();
    for (const mode of modes) {
      await r.setOptions({ restirMode: mode });
      expect(r.options.maxBounces).toBe(3);
      expect(r.options.temporal).toBe(true);
      const denoise = mode === 'interactive' || mode === 'potato';
      expect(r.denoiseWanted()).toBe(denoise);
      expect(await r.prepareRestir()).toBeTruthy();
      if (denoise) {
        expect(await r.prepareDenoiser()).toBeTruthy();
        expect(r.denoiser!.settings.iterations).toBe(mode === 'potato' ? 3 : 4);
      }
      await r.warmup(false);
      for (let frame = 1; frame <= 4; frame++) {
        const reset = frame === 1;
        fu.write({ camera: cam, prevCamera: cam, width: W, height: H, frameIndex: frame, seedIndex: frame, runSeed: 7,
          flags: reset ? FRAME_RESET_HISTORY : 0, jitterMode: JITTER_IID, jitter: [0.5, 0.5], origin, exposure: 1,
          time: frame / 60, dt: 1 / 60, sceneDiag: boundsDiagonal(scene.bounds) });
        debug.update(defaultDebugSettings(), frame);
        const e = device.createCommandEncoder(); debug.beginFrame(e);
        expect(r.encode(e, { advanced: true, debugMode: 0, debugGroup: debug.bindGroup, resetTemporal: reset, resetHistory: reset })).toBe(true);
        expect(r.denoisedLastFrame).toBe(denoise);
        device.queue.submit([e.finish()]);
      }
      const e = device.createCommandEncoder();
      e.copyTextureToBuffer({ texture: color }, { buffer: readback, bytesPerRow: W * 16 }, [W, H]);
      device.queue.submit([e.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(readback.getMappedRange().slice(0)); readback.unmap();
      expect(values.every(Number.isFinite)).toBe(true);
      expect(values.some((v, i) => i % 4 !== 3 && v > 0)).toBe(true);
      hashes.push(hashF32(values));
    }
    expect(errors).toEqual([]);
  } finally {
    device.removeEventListener('uncapturederror', onError);
    r.destroy(); debug.destroy(); fu.destroy(); color.destroy(); depth.destroy(); readback.destroy();
  }
  return hashes;
}

it('Potato → offline → Potato → interactive restores fresh rendering and denoiser defaults', async () => {
  const interactive = (await run(['interactive']))[0];
  const potato = (await run(['potato']))[0];
  const sequence = await run(['interactive', 'potato', 'offline', 'potato', 'interactive']);
  expect(sequence[0]).toBe(interactive);
  expect(sequence[1]).toBe(potato);
  expect(sequence[3]).toBe(potato);
  expect(sequence[4]).toBe(interactive);
  expect(potato).not.toBe(interactive);
}, 180_000);
