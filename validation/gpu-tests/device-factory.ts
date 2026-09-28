// Returns a GPUContext for whichever lane runs the test: navigator.gpu in Chrome, dawn.node in Node.
import { createGpuContext, type GpuContext } from '../../src/core/gpu/device.ts';

let cached: Promise<GpuContext> | undefined;
// Keep the dawn.node GPU object alive for the whole worker lifetime; destroying it mid-run crashes Dawn.
let dawnGpu: unknown;

export function getTestGpu(): Promise<GpuContext> {
  cached ??= (async () => {
    if (typeof navigator !== 'undefined' && (navigator as Navigator).gpu) {
      return createGpuContext((navigator as Navigator).gpu, { requireHardwareMetal: true, label: 'chrome-test' });
    }
    const webgpu = await import('webgpu');
    Object.assign(globalThis, webgpu.globals);
    dawnGpu = webgpu.create(['backend=metal']);
    return createGpuContext(dawnGpu as GPU, { requireHardwareMetal: true, label: 'dawn-test' });
  })();
  return cached;
}

/** Call from afterAll: finish GPU work and destroy the device before the worker exits. */
export async function releaseTestGpu(): Promise<void> {
  if (!cached) return;
  const ctx = await cached;
  await ctx.device.queue.onSubmittedWorkDone();
  ctx.device.destroy();
}

export const lane = (): 'chrome' | 'node-dawn' => (typeof navigator !== 'undefined' && (navigator as Navigator).gpu ? 'chrome' : 'node-dawn');
