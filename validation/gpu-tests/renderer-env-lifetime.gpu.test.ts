import { afterAll, expect, it, vi } from 'vitest';
import { getTestGpu, releaseTestGpu } from './device-factory.ts';
import { DebugResources, DebugViewRegistry, defaultDebugSettings } from '../../src/core/render/debug-views.ts';
import { FRAME_RESET_HISTORY, FrameUniformBuffer, JITTER_IID, boundsDiagonal, computeRenderOrigin } from '../../src/core/render/frame-uniforms.ts';
import { Renderer } from '../../src/core/render/renderer.ts';
import { PtFramePass } from '../../src/core/render/pt-kernel.ts';
import { RestirFramePass } from '../../src/core/render/restir/kernel.ts';
import { allLightsScene, boxCamera } from './restir-fixtures.ts';

afterAll(async () => { await releaseTestGpu(); });

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>(r => { resolve = r; });
  return { promise, resolve };
}

// Hold a real compiled pass just before publication. Loading an environment while compilation is
// pending must not leave the pass bound to the destroyed placeholder, regardless of GPU timing.
for (const phase of ['pt-create', 'scene-ready', 'restir-ready'] as const) {
  it(`${phase}: environment replacement during compilation renders with live resources`, async () => {
    const mode = phase === 'restir-ready' ? 'restir' : 'pt';
    const ctx = await getTestGpu(), { device } = ctx;
    const W = 32, H = 24, scene = allLightsScene(), origin = computeRenderOrigin(scene.bounds, scene.quant);
    const debug = new DebugResources(device, new DebugViewRegistry());
    await debug.init(); debug.resize(W, H);
    const r = await Renderer.create({ device, debugLayout: debug.layout, debug, features: ctx.features, wgslLanguageFeatures: ctx.wgslLanguageFeatures },
      { renderMode: mode, restirMode: 'interactive', accumulate: false, watertight: false });
    r.setDenoise(false);
    const fu = new FrameUniformBuffer(device);
    const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
    const color = device.createTexture({ size: [W, H], format: 'rgba32float', usage });
    const depth = device.createTexture({ size: [W, H], format: 'r32float', usage });
    const readback = device.createBuffer({ size: W * H * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const entered = deferred(), resume = deferred(), ptReady = deferred();
    const errors: string[] = [];
    const onError = (e: GPUUncapturedErrorEvent) => errors.push(e.error.message);
    device.addEventListener('uncapturederror', onError);
    const ptCreate = PtFramePass.create, rsPrepare = RestirFramePass.prototype.prepare;
    const spy = mode === 'pt'
      ? vi.spyOn(PtFramePass, 'create').mockImplementationOnce(async (...args) => {
        const pass = await ptCreate(...args);
        ptReady.resolve();
        if (phase === 'pt-create') { entered.resolve(); await resume.promise; }
        return pass;
      })
      : vi.spyOn(RestirFramePass.prototype, 'prepare').mockImplementationOnce(async function (this: RestirFramePass) {
        await rsPrepare.call(this); entered.resolve(); await resume.promise;
      });
    const createPipeline = device.createComputePipelineAsync;
    const sceneSpy = phase === 'scene-ready' ? vi.spyOn(device, 'createComputePipelineAsync').mockImplementation(async desc => {
      const pipeline = await createPipeline.call(device, desc);
      if (desc.compute.entryPoint === 'primary') {
        await ptReady.promise; entered.resolve(); await resume.promise;
      }
      return pipeline;
    }) : undefined;
    try {
      r.resize({ width: W, height: H, color, colorFormat: 'rgba32float', depth, frameUniforms: fu.buffer });
      const loading = mode === 'pt' ? r.setScene(scene, origin) : (async () => {
        await r.setScene(scene, origin); return r.prepareRestir();
      })();
      await entered.promise;
      await r.setEnvironment(scene.env);
      resume.resolve();
      expect(await loading).toBeTruthy();
      const cam = boxCamera();
      for (let frame = 0; frame < 3; frame++) {
        // A later map removal exercises the normal, already-published replacement path as well.
        if (frame === 2) await r.setEnvironment(undefined);
        fu.write({ camera: cam, prevCamera: cam, width: W, height: H, frameIndex: frame, seedIndex: frame,
          runSeed: 7, flags: FRAME_RESET_HISTORY, jitterMode: JITTER_IID, jitter: [0.5, 0.5], origin,
          exposure: 1, time: frame / 60, dt: 1 / 60, sceneDiag: boundsDiagonal(scene.bounds) });
        debug.update(defaultDebugSettings(), frame);
        device.pushErrorScope('validation');
        const e = device.createCommandEncoder({ label: 'env-lifetime-frame' }); debug.beginFrame(e);
        expect(r.encode(e, { advanced: true, debugMode: 0, debugGroup: debug.bindGroup, resetTemporal: true, resetHistory: true })).toBe(true);
        e.copyTextureToBuffer({ texture: color }, { buffer: readback, bytesPerRow: W * 16 }, [W, H]);
        device.queue.submit([e.finish()]);
        const validation = await device.popErrorScope();
        expect(validation?.message).toBeUndefined();
        await readback.mapAsync(GPUMapMode.READ);
        const values = new Float32Array(readback.getMappedRange().slice(0)); readback.unmap();
        expect(values.every(Number.isFinite)).toBe(true);
        expect(values.some((v, i) => i % 4 !== 3 && v > 0)).toBe(true);
      }
      if (mode === 'restir') expect(r.restir!.kernel.envResources).toBe(r.env);
      expect(errors).toEqual([]);
    } finally {
      resume.resolve(); spy.mockRestore(); sceneSpy?.mockRestore();
      device.removeEventListener('uncapturederror', onError);
      r.destroy(); debug.destroy(); fu.destroy(); color.destroy(); depth.destroy(); readback.destroy();
    }
  }, 120_000);
}
