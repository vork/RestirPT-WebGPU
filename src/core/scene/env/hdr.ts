// Radiance RGBE (.hdr) decoder, bit-identical to OpenImageIO/Blender (plan §1.4b; math.md#env-mapping).
//
// Pixel = 0 if e == 0, else f32(byte · T[e]) with T[e] = 2^(e−136) (OIIO rgbe2float; NOT Ward's (byte+0.5) and NOT
// three.js HDRLoader, whose 2^(e−128)/255 is +0.39% bright). OIIO's T is a table of f32 decimal literals, and three
// entries are 1 ulp BELOW the exact power of two (e = 40, 223, 226; measured with OIIO 3.1 via
// validation/tools/env_ref.py, ENV-U1 synthetic 'exponent table' case). We replicate the table, so the result is
// bit-identical to OIIO/Blender for every byte pattern. byte·T[e] is exact in f64; Math.fround = f32 multiply.
// Scanlines: new-style RLE ([2,2,hi,lo] + 4 run-length channel planes) or flat RGBE. As in Ward's reader (which
// OIIO follows), the first scanline that is not new-style RLE switches the REST of the image to flat pixels.
// Old-style RLE (1,1,1,n repeat pixels) is not interpreted, like OIIO (they decode as ordinary flat pixels).

export interface HdrHeader {
  format: string;
  /** EXPOSURE= values present in the header; ignored like OIIO/Blender (logged as a warning). */
  exposure?: number;
  lines: string[];
}

export interface DecodedEnvImage {
  width: number;
  height: number;
  /** RGBA float32 (alpha 1), rows BOTTOM-UP (row 0 = nadir, v = 0), as uploaded to the GPU. */
  texels: Float32Array;
  warnings: string[];
}

export class HdrDecodeError extends Error {}

/** OIIO rgbe2float table as f32: 2^(e−136) for e ≥ 1 (with OIIO's three 1-ulp-low literals), 0 for e = 0. */
export const RGBE_EXP_TABLE = (() => {
  const t = new Float32Array(256);
  const u = new Uint32Array(t.buffer);
  for (let e = 1; e < 256; e++) t[e] = Math.pow(2, e - 136);
  for (const e of [40, 223, 226]) u[e] -= 1; // OIIO literal rounding (0x0f7fffff, 0x6affffff, 0x6c7fffff)
  return t;
})();

export function isHdr(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === 0x23 && bytes[1] === 0x3f; // "#?"
}

/** Decode a .hdr file. Rejects anything but `FORMAT=32-bit_rle_rgbe` with a `-Y H +X W` resolution line. */
export function decodeHdr(input: ArrayBuffer | Uint8Array): DecodedEnvImage & { header: HdrHeader } {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const warnings: string[] = [];
  let pos = 0;
  const readLine = (): string => {
    const start = pos;
    while (pos < bytes.length && bytes[pos] !== 0x0a) pos++;
    if (pos >= bytes.length) throw new HdrDecodeError('truncated header');
    if (pos - start > 4096) throw new HdrDecodeError('header line too long');
    const s = String.fromCharCode(...bytes.subarray(start, pos)).replace(/\r$/, '');
    pos++;
    return s;
  };

  const magic = readLine();
  if (!/^#\?(RADIANCE|RGBE)/.test(magic)) throw new HdrDecodeError(`not a Radiance file (magic '${magic.slice(0, 16)}')`);
  const header: HdrHeader = { format: '', lines: [magic] };
  for (;;) {
    const line = readLine();
    if (line.trim() === '') break;
    header.lines.push(line);
    const kv = /^\s*([A-Za-z_-]+)\s*=\s*(.*?)\s*$/.exec(line);
    if (!kv) continue; // comments / program lines
    if (kv[1] === 'FORMAT') header.format = kv[2];
    else if (kv[1] === 'EXPOSURE') {
      header.exposure = (header.exposure ?? 1) * Number(kv[2]);
      warnings.push(`EXPOSURE=${kv[2]} ignored (as OIIO/Blender)`);
    }
  }
  if (header.format !== '32-bit_rle_rgbe') throw new HdrDecodeError(`unsupported FORMAT '${header.format || '(missing)'}' (only 32-bit_rle_rgbe)`);
  const res = /^\s*-Y\s+(\d+)\s+\+X\s+(\d+)\s*$/.exec(readLine());
  if (!res) throw new HdrDecodeError('unsupported resolution line (only "-Y H +X W")');
  const height = Number(res[1]);
  const width = Number(res[2]);
  if (!(width > 0 && height > 0)) throw new HdrDecodeError(`bad resolution ${width}x${height}`);

  // RGBE bytes, rows in file order (top-down).
  const rgbe = new Uint8Array(width * height * 4);
  const scan = new Uint8Array(width * 4);
  let flatFrom = -1; // pixel index where the flat remainder starts
  if (width < 8 || width > 0x7fff) flatFrom = 0;
  for (let y = 0; y < height && flatFrom < 0; y++) {
    if (pos + 4 > bytes.length) throw new HdrDecodeError(`truncated at scanline ${y}`);
    if (bytes[pos] !== 2 || bytes[pos + 1] !== 2 || (bytes[pos + 2] & 0x80) !== 0) { flatFrom = y * width; break; }
    if (((bytes[pos + 2] << 8) | bytes[pos + 3]) !== width) throw new HdrDecodeError(`wrong scanline width at row ${y}`);
    pos += 4;
    for (let c = 0; c < 4; c++) {
      let x = 0;
      while (x < width) {
        if (pos >= bytes.length) throw new HdrDecodeError(`truncated RLE at row ${y}`);
        let count = bytes[pos++];
        if (count > 128) {
          count -= 128;
          if (count > width - x) throw new HdrDecodeError(`bad RLE run at row ${y}`);
          const v = bytes[pos++];
          for (let k = 0; k < count; k++) scan[(x++) * 4 + c] = v;
        } else {
          if (count === 0 || count > width - x) throw new HdrDecodeError(`bad RLE literal at row ${y}`);
          if (pos + count > bytes.length) throw new HdrDecodeError(`truncated RLE literal at row ${y}`);
          for (let k = 0; k < count; k++) scan[(x++) * 4 + c] = bytes[pos++];
        }
      }
    }
    rgbe.set(scan, y * width * 4);
  }
  if (flatFrom >= 0) {
    const need = (width * height - flatFrom) * 4;
    if (pos + need > bytes.length) throw new HdrDecodeError(`truncated flat data (${bytes.length - pos} of ${need} bytes)`);
    rgbe.set(bytes.subarray(pos, pos + need), flatFrom * 4);
    pos += need;
  }

  // Decode and flip to bottom-up rows.
  const texels = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    const src = y * width * 4;
    const dst = (height - 1 - y) * width * 4;
    for (let x = 0; x < width * 4; x += 4) {
      const s = RGBE_EXP_TABLE[rgbe[src + x + 3]];
      texels[dst + x] = rgbe[src + x] * s; // exact in f64, rounded to f32 on store (= OIIO's f32 multiply)
      texels[dst + x + 1] = rgbe[src + x + 1] * s;
      texels[dst + x + 2] = rgbe[src + x + 2] * s;
      texels[dst + x + 3] = 1;
    }
  }
  return { width, height, texels, warnings, header };
}
