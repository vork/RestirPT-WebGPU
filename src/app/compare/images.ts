// Image sources for the compare view: Cycles EXRs (three.js EXRLoader used purely as a PARSER, plan §1.1) and the
// app's current accumulated image (the linear color target, rgba32float or rgba16float, read back from the GPU).
import { FloatType, RGBAFormat } from 'three';
import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';
import { halfToFloat, type RgbaImage } from './compare-math.ts';

/** Parse an OpenEXR file into RGBA float32 rows TOP-DOWN (EXRLoader emits bottom-up rows; missing A = 1). */
export function parseExrRgba(bytes: ArrayBuffer | Uint8Array): RgbaImage {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const buf = u8.byteOffset === 0 && u8.byteLength === u8.buffer.byteLength ? (u8.buffer as ArrayBuffer) : (u8.slice().buffer as ArrayBuffer);
  const parsed = new EXRLoader().setDataType(FloatType).setOutputFormat(RGBAFormat).parse(buf) as { width: number; height: number; data: Float32Array };
  const { width, height } = parsed;
  const src = parsed.data;
  if (src.length !== width * height * 4) throw new Error(`EXR: unexpected output size ${src.length} for ${width}x${height}`);
  const data = new Float32Array(src.length);
  const row = width * 4;
  for (let y = 0; y < height; y++) data.set(src.subarray((height - 1 - y) * row, (height - y) * row), y * row);
  return { width, height, data };
}

export async function fetchExr(url: string): Promise<RgbaImage> {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return parseExrRgba(new Uint8Array(await r.arrayBuffer()));
}

/** Read a 2D rgba32float / rgba16float texture (COPY_SRC) into RGBA float32, rows top-down. */
export async function readColorTexture(device: GPUDevice, tex: GPUTexture): Promise<RgbaImage> {
  const fmt = tex.format;
  if (fmt !== 'rgba32float' && fmt !== 'rgba16float') throw new Error(`readColorTexture: unsupported format ${fmt}`);
  const bpp = fmt === 'rgba32float' ? 16 : 8;
  const width = tex.width, height = tex.height;
  const bytesPerRow = Math.ceil((width * bpp) / 256) * 256;
  const buf = device.createBuffer({ label: 'compare-readback', size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const enc = device.createCommandEncoder({ label: 'compare-readback' });
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow }, { width, height });
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const raw = buf.getMappedRange().slice(0);
    buf.unmap();
    const data = new Float32Array(width * height * 4);
    for (let y = 0; y < height; y++) {
      if (bpp === 16) data.set(new Float32Array(raw, y * bytesPerRow, width * 4), y * width * 4);
      else {
        const h = new Uint16Array(raw, y * bytesPerRow, width * 4);
        for (let i = 0; i < width * 4; i++) data[y * width * 4 + i] = halfToFloat(h[i]);
      }
    }
    return { width, height, data };
  } finally {
    buf.destroy();
  }
}
