// Minimal lossless PNG codec for RGBA8 textures in scene packages (docs/decisions/scene-bridge.md: tex_<i>.png).
// Encoder: 8-bit RGBA, non-interlaced, per-row filter chosen by the minimum-sum-of-absolute-differences heuristic,
// zlib via CompressionStream. No colour chunks (no gAMA/sRGB/iCCP): the bytes are the texture bytes, and colour
// interpretation (sRGB vs linear) is decided by the material slot, exactly as in the renderer.
// Decoder: 8-bit grey / grey+alpha / RGB / RGBA, non-interlaced (enough for our own files and typical textures);
// the output is always RGBA8, rows top-first.
import { zlibDeflate, zlibInflate } from './zlib.ts';

const SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(bytes: Uint8Array, start = 0, end = bytes.length): number {
  let c = 0xffffffff;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function paeth(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

export interface Rgba8Image { width: number; height: number; /** RGBA8, rows top-first. */ pixels: Uint8Array }

export async function encodePng(img: Rgba8Image): Promise<Uint8Array> {
  const { width: w, height: h, pixels } = img;
  if (!(w > 0 && h > 0) || pixels.length !== w * h * 4) throw new Error(`PNG: need ${w}x${h}x4 bytes, got ${pixels.length}`);
  const stride = w * 4;
  const raw = new Uint8Array((stride + 1) * h);
  const cand = new Uint8Array(stride);
  for (let y = 0; y < h; y++) {
    const cur = pixels.subarray(y * stride, (y + 1) * stride);
    const prev = y > 0 ? pixels.subarray((y - 1) * stride, y * stride) : null;
    let best = -1, bestSum = Infinity;
    for (let f = 0; f < 5; f++) {
      if (f >= 2 && !prev) break;
      let sum = 0;
      for (let i = 0; i < stride; i++) {
        const a = i >= 4 ? cur[i - 4] : 0, b = prev ? prev[i] : 0, c = prev && i >= 4 ? prev[i - 4] : 0;
        const pred = f === 0 ? 0 : f === 1 ? a : f === 2 ? b : f === 3 ? (a + b) >> 1 : paeth(a, b, c);
        const v = (cur[i] - pred) & 0xff;
        cand[i] = v;
        sum += v < 128 ? v : 256 - v;
      }
      if (sum < bestSum) {
        bestSum = sum; best = f;
        raw[y * (stride + 1)] = f;
        raw.set(cand, y * (stride + 1) + 1);
      }
    }
    if (best < 0) throw new Error('PNG: filter selection failed');
  }
  const idat = await zlibDeflate(raw);
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, w); dv.setUint32(4, h);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  const chunks = [chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', new Uint8Array(0))];
  const out = new Uint8Array(8 + chunks.reduce((s, c) => s + c.length, 0));
  out.set(SIG, 0);
  let o = 8;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out, 4, 8 + data.length));
  return out;
}

export function isPng(bytes: Uint8Array): boolean {
  return bytes.length >= 8 && SIG.every((b, i) => bytes[i] === b);
}

export async function decodePng(bytes: Uint8Array): Promise<Rgba8Image> {
  if (!isPng(bytes)) throw new Error('PNG: bad signature');
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let o = 8, w = 0, h = 0, depth = 0, ctype = -1, interlace = 0;
  const idat: Uint8Array[] = [];
  while (o + 12 <= bytes.length) {
    const len = dv.getUint32(o);
    const type = String.fromCharCode(bytes[o + 4], bytes[o + 5], bytes[o + 6], bytes[o + 7]);
    const data = bytes.subarray(o + 8, o + 8 + len);
    if (crc32(bytes, o + 4, o + 8 + len) !== dv.getUint32(o + 8 + len)) throw new Error(`PNG: CRC mismatch in ${type}`);
    if (type === 'IHDR') {
      const d = new DataView(data.buffer, data.byteOffset, data.byteLength);
      w = d.getUint32(0); h = d.getUint32(4); depth = data[8]; ctype = data[9]; interlace = data[12];
    } else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    o += 12 + len;
  }
  const channels = ({ 0: 1, 2: 3, 4: 2, 6: 4 } as Record<number, number>)[ctype];
  if (depth !== 8 || !channels || interlace !== 0) throw new Error(`PNG: unsupported format (depth ${depth}, colour type ${ctype}, interlace ${interlace})`);
  const z = new Uint8Array(idat.reduce((s, c) => s + c.length, 0));
  let zo = 0;
  for (const c of idat) { z.set(c, zo); zo += c.length; }
  const raw = await zlibInflate(z);
  const stride = w * channels;
  if (raw.length < (stride + 1) * h) throw new Error('PNG: truncated image data');
  const img = new Uint8Array(stride * h);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)];
    const src = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const cur = img.subarray(y * stride, (y + 1) * stride);
    const prev = y > 0 ? img.subarray((y - 1) * stride, y * stride) : null;
    for (let i = 0; i < stride; i++) {
      const a = i >= channels ? cur[i - channels] : 0, b = prev ? prev[i] : 0, c = prev && i >= channels ? prev[i - channels] : 0;
      const pred = f === 0 ? 0 : f === 1 ? a : f === 2 ? b : f === 3 ? (a + b) >> 1 : f === 4 ? paeth(a, b, c) : -1;
      if (pred < 0) throw new Error(`PNG: bad filter ${f}`);
      cur[i] = (src[i] + pred) & 0xff;
    }
  }
  if (channels === 4) return { width: w, height: h, pixels: img };
  const px = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const s = i * channels;
    if (channels === 1) { px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = img[s]; px[4 * i + 3] = 255; }
    else if (channels === 2) { px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = img[s]; px[4 * i + 3] = img[s + 1]; }
    else { px[4 * i] = img[s]; px[4 * i + 1] = img[s + 1]; px[4 * i + 2] = img[s + 2]; px[4 * i + 3] = 255; }
  }
  return { width: w, height: h, pixels: px };
}
