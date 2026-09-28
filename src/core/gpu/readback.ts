// Buffer readback helpers (DOM-free).
export async function readBuffer(device: GPUDevice, src: GPUBuffer, byteLength: number, srcOffset = 0): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size: alignTo(byteLength, 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder({ label: 'readback' });
  enc.copyBufferToBuffer(src, srcOffset, staging, 0, alignTo(byteLength, 4));
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange(0, alignTo(byteLength, 4)).slice(0, byteLength);
  staging.unmap();
  staging.destroy();
  return out;
}

export function alignTo(n: number, a: number): number { return Math.ceil(n / a) * a; }
