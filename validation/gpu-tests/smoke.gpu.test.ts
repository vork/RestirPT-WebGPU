import { afterAll, describe, expect, it } from 'vitest';
import { getTestGpu, lane, releaseTestGpu } from './device-factory.ts';
import { composeWgsl, createCheckedShaderModule } from '../../src/core/gpu/wgsl-composer.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { describeContext } from '../../src/core/gpu/device.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';

const N = 4096;

describe(`smoke (${lane()})`, () => {
  afterAll(releaseTestGpu);
  it('device satisfies the M5 Pro profile and reports lane info', async () => {
    const ctx = await getTestGpu();
    console.log('LANE_REPORT', lane(), JSON.stringify(describeContext(ctx)));
    expect(ctx.adapterInfo.vendor).toBe('apple');
    expect(ctx.limits.maxStorageBuffersPerShaderStage).toBe(10);
    expect(ctx.limits.maxStorageBufferBindingSize).toBeGreaterThanOrEqual(2 ** 31);
  });

  it('composed shader runs: RNG determinism, bit-test NaN/Inf, relaxed-math probes', async () => {
    const { device, features, wgslLanguageFeatures } = await getTestGpu();
    const shader = composeWgsl('tests/smoke.wgsl', { sources: shaderSources, defines: { N: `${N}u`, WG: 64 }, features, wgslLanguageFeatures });
    const module = await createCheckedShaderModule(device, shader, 'smoke');
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // a=b=1.0000001 (f32), c=-1.0000002: exact product-sum differs between FMA and separate rounding
    const a = Math.fround(1.0000001), c = Math.fround(-1.0000002);
    device.queue.writeBuffer(params, 0, new Float32Array([a, a, c, 0]));
    const outU = device.createBuffer({ size: (N + 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const outF = device.createBuffer({ size: (N + 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: outU } }, { binding: 2, resource: { buffer: outF } }] });
    const run = async () => {
      const enc = device.createCommandEncoder();
      const p = enc.beginComputePass(); p.setPipeline(pipeline); p.setBindGroup(0, bg); p.dispatchWorkgroups(Math.ceil(N / 64)); p.end();
      device.queue.submit([enc.finish()]);
      return [new Uint32Array(await readBuffer(device, outU, (N + 4) * 4)), new Float32Array(await readBuffer(device, outF, (N + 4) * 4))] as const;
    };
    const [u1, f1] = await run();
    const [u2, f2] = await run();
    expect(Array.from(u1.subarray(0, N))).toEqual(Array.from(u2.subarray(0, N)));
    expect(Array.from(f1.subarray(0, N))).toEqual(Array.from(f2.subarray(0, N)));
    // uniform-ish sanity of rand1: mean ≈ 0.5, all in [0,1)
    const fs = Array.from(f1.subarray(0, N));
    expect(Math.min(...fs)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...fs)).toBeLessThan(1);
    expect(Math.abs(fs.reduce((s, x) => s + x, 0) / N - 0.5)).toBeLessThan(0.02);
    expect(u1[N + 0]).toBe(1); // bit-test is_nan works
    expect(u1[N + 1]).toBe(1); // bit-test is_inf works
    expect(u1[N + 3]).toBe(1); // FLT_MAX finite
    const sep = Math.fround(Math.fround(a * a) + c);
    console.log('PROBES', lane(), JSON.stringify({ nanSelfCompare: u1[N + 2], fmaResult: f1[N], separateRounding: sep, fmaContracted: f1[N] !== sep }));
  });
});
