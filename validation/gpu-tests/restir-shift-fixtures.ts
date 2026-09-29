// WP-B helpers for the shift GPU tests (restir-shift.gpu.test.ts; restir-api.md §1.2, §6.1–§6.2): custom test
// pipelines on top of RestirKernel (G0/G1 of the kernel, a test G2 of storage buffers), f32 bit helpers.
import { readBuffer } from '../../src/core/gpu/readback.ts';
import type { Defines } from '../../src/core/gpu/wgsl-composer.ts';
import type { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import type { RsDispatchCpu } from '../../src/core/render/restir/layout.ts';
import { storageBuffer } from './restir-fixtures.ts';

const f32b = new Float32Array(1);
const u32b = new Uint32Array(f32b.buffer);
export const bitsToF32 = (u: number): number => { u32b[0] = u; return f32b[0]; };
export const f32ToBits = (x: number): number => { f32b[0] = x; return u32b[0]; };
export const toF32 = (x: number): number => Math.fround(x);

export interface TestPipeline {
  pipeline: GPUComputePipeline;
  layout: GPUBindGroupLayout;
  /** Bind `buffers` (G2 bindings 0…n−1, storage) and dispatch [x, y] workgroups with the kernel's G0 / G1. */
  run(buffers: GPUBuffer[], work: [number, number], d?: Partial<RsDispatchCpu>, textures?: GPUTextureView[]): Promise<void>;
}

/**
 * Compile a test pipeline from an in-memory WGSL source: G2 = `nBuffers` read-write storage buffers (bindings 0…),
 * then `textures` sampled textures (uint | unfilterable-float) at the following bindings. Uses the kernel's G0 (frame,
 * env, RestirParams, lights, records, RsDispatch) and G1 (scene). No debug group (DEBUG_NO_BINDINGS).
 */
export async function testPipeline(k: RestirKernel, name: string, src: string, entry: string, nBuffers: number,
  defines: Defines = {}, textures: GPUTextureSampleType[] = []): Promise<TestPipeline> {
  const c = GPUShaderStage.COMPUTE;
  const entries: GPUBindGroupLayoutEntry[] = [];
  for (let i = 0; i < nBuffers; i++) entries.push({ binding: i, visibility: c, buffer: { type: 'storage' } });
  textures.forEach((t, j) => entries.push({ binding: nBuffers + j, visibility: c, texture: { sampleType: t } }));
  const layout = k.device.createBindGroupLayout({ label: `${name}-g2`, entries });
  const file = `${name}.wgsl`;
  const pipeline = await k.compile(file, entry, k.customDefines(defines, true), k.customLayout(layout, true), name, { [file]: src });
  return {
    pipeline, layout,
    async run(buffers, work, d = {}, texViews = []) {
      const g2 = k.device.createBindGroup({
        layout,
        entries: [...buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })), ...texViews.map((v, j) => ({ binding: nBuffers + j, resource: v }))],
      });
      k.beginSubmit();
      const enc = k.device.createCommandEncoder({ label: name });
      k.encodeCustom(enc, pipeline, g2, d, work, true);
      k.device.queue.submit([enc.finish()]);
      await k.device.queue.onSubmittedWorkDone();
    },
  };
}

export async function readU32(device: GPUDevice, b: GPUBuffer, bytes = b.size): Promise<Uint32Array> {
  return new Uint32Array(await readBuffer(device, b, bytes));
}
export { storageBuffer };
