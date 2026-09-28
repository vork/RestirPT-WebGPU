// Minimal OpenEXR writer: single-part scanline image, FLOAT (32-bit) channels, NO_COMPRESSION or ZIP_COMPRESSION
// (16-line zlib blocks with the OpenEXR byte-interleave + delta predictor). Used for the scene package env.exr
// (docs/decisions/scene-bridge.md) so Blender/OIIO read back the exact float32 texels our GPU samples.
// Input rows are TOP-DOWN (EXR convention, INCREASING_Y); channels are written in the sorted order EXR requires.
// Checked by reading back with OpenImageIO (tests/io/exr-writer.test.ts).
import { zlibDeflate } from './zlib.ts';

export type ExrCompression = 'none' | 'zip';

export interface ExrImage {
  width: number;
  height: number;
  /** Interleaved float32 pixels, rows top-down, `channels.length` values per pixel. */
  data: Float32Array;
  /** Channel names in the order of `data` (e.g. ['R','G','B','A']). */
  channels: string[];
}

const COMPRESSION_CODE: Record<ExrCompression, number> = { none: 0, zip: 3 };
const LINES_PER_BLOCK: Record<ExrCompression, number> = { none: 1, zip: 16 };
const PIXEL_FLOAT = 2;

class ByteWriter {
  private buf = new Uint8Array(1024);
  private dv = new DataView(this.buf.buffer);
  length = 0;
  private ensure(n: number): void {
    if (this.length + n <= this.buf.length) return;
    let cap = this.buf.length * 2;
    while (cap < this.length + n) cap *= 2;
    const nb = new Uint8Array(cap);
    nb.set(this.buf.subarray(0, this.length));
    this.buf = nb;
    this.dv = new DataView(nb.buffer);
  }
  u8(v: number): void { this.ensure(1); this.buf[this.length++] = v; }
  i32(v: number): void { this.ensure(4); this.dv.setInt32(this.length, v, true); this.length += 4; }
  u32(v: number): void { this.ensure(4); this.dv.setUint32(this.length, v, true); this.length += 4; }
  f32(v: number): void { this.ensure(4); this.dv.setFloat32(this.length, v, true); this.length += 4; }
  u64(v: number): void { this.ensure(8); this.dv.setBigUint64(this.length, BigInt(v), true); this.length += 8; }
  str(s: string): void { for (let i = 0; i < s.length; i++) this.u8(s.charCodeAt(i)); this.u8(0); }
  bytes(b: Uint8Array): void { this.ensure(b.length); this.buf.set(b, this.length); this.length += b.length; }
  patchU64(at: number, v: number): void { this.dv.setBigUint64(at, BigInt(v), true); }
  result(): Uint8Array { return this.buf.slice(0, this.length); }
}

function attr(w: ByteWriter, name: string, type: string, size: number, body: () => void): void {
  w.str(name); w.str(type); w.i32(size);
  const start = w.length;
  body();
  if (w.length - start !== size) throw new Error(`EXR attribute ${name}: wrote ${w.length - start} bytes, declared ${size}`);
}

/** OpenEXR ZIP pre-processing (ImfZip.cpp): split even/odd bytes into two halves, then a byte-delta predictor. */
export function exrZipPredict(raw: Uint8Array): Uint8Array {
  const n = raw.length;
  const t = new Uint8Array(n);
  let t1 = 0, t2 = (n + 1) >> 1, i = 0;
  for (;;) {
    if (i < n) t[t1++] = raw[i++]; else break;
    if (i < n) t[t2++] = raw[i++]; else break;
  }
  let p = t[0];
  for (let k = 1; k < n; k++) {
    const d = (t[k] - p + (128 + 256)) & 0xff;
    p = t[k];
    t[k] = d;
  }
  return t;
}

/** Encode an EXR file. */
export async function encodeExr(img: ExrImage, compression: ExrCompression = 'zip'): Promise<Uint8Array> {
  const { width: W, height: H, data, channels } = img;
  const nc = channels.length;
  if (!(W > 0 && H > 0) || data.length !== W * H * nc) throw new Error(`EXR: data length ${data.length} != ${W}x${H}x${nc}`);
  if (new Set(channels).size !== nc || nc === 0) throw new Error(`EXR: bad channel list ${channels.join(',')}`);
  // EXR requires the channel list sorted by name (byte order); pixel data follows the same order.
  const order = channels.map((name, src) => ({ name, src })).sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));

  const w = new ByteWriter();
  w.u32(20000630); // magic 0x762f3101
  w.u32(2);        // version 2, single-part scanline, short names
  const chlistSize = order.reduce((s, c) => s + c.name.length + 1 + 16, 0) + 1;
  attr(w, 'channels', 'chlist', chlistSize, () => {
    for (const c of order) { w.str(c.name); w.i32(PIXEL_FLOAT); w.u8(0); w.u8(0); w.u8(0); w.u8(0); w.i32(1); w.i32(1); }
    w.u8(0);
  });
  attr(w, 'compression', 'compression', 1, () => w.u8(COMPRESSION_CODE[compression]));
  attr(w, 'dataWindow', 'box2i', 16, () => { w.i32(0); w.i32(0); w.i32(W - 1); w.i32(H - 1); });
  attr(w, 'displayWindow', 'box2i', 16, () => { w.i32(0); w.i32(0); w.i32(W - 1); w.i32(H - 1); });
  attr(w, 'lineOrder', 'lineOrder', 1, () => w.u8(0)); // INCREASING_Y
  attr(w, 'pixelAspectRatio', 'float', 4, () => w.f32(1));
  attr(w, 'screenWindowCenter', 'v2f', 8, () => { w.f32(0); w.f32(0); });
  attr(w, 'screenWindowWidth', 'float', 4, () => w.f32(1));
  w.u8(0); // end of header

  const lpb = LINES_PER_BLOCK[compression];
  const nBlocks = Math.ceil(H / lpb);
  const tableAt = w.length;
  for (let b = 0; b < nBlocks; b++) w.u64(0);

  const lineBytes = W * nc * 4;
  for (let b = 0; b < nBlocks; b++) {
    const y0 = b * lpb, lines = Math.min(lpb, H - y0);
    const raw = new Uint8Array(lines * lineBytes);
    const rdv = new DataView(raw.buffer);
    for (let l = 0; l < lines; l++) {
      const y = y0 + l;
      let o = l * lineBytes;
      for (const c of order) {
        for (let x = 0; x < W; x++) { rdv.setFloat32(o, data[(y * W + x) * nc + c.src], true); o += 4; }
      }
    }
    let payload: Uint8Array = raw;
    if (compression === 'zip') {
      const z = await zlibDeflate(exrZipPredict(raw));
      if (z.length < raw.length) payload = z; // else stored raw (readers detect it by size)
    }
    w.patchU64(tableAt + 8 * b, w.length);
    w.i32(y0);
    w.i32(payload.length);
    w.bytes(payload);
  }
  return w.result();
}

/** RGBA float32 rows BOTTOM-UP (EnvironmentData.texels, GPU upload order) → rows TOP-DOWN (EXR/image order). */
export function flipRows(data: Float32Array, width: number, height: number, channels = 4): Float32Array {
  const out = new Float32Array(data.length);
  const row = width * channels;
  for (let y = 0; y < height; y++) out.set(data.subarray(y * row, (y + 1) * row), (height - 1 - y) * row);
  return out;
}
