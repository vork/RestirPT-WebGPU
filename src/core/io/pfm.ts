// Portable Float Map (PFM) codec. In-memory images are row 0 = TOP (image convention, plan §1.2);
// PFM stores rows bottom-to-top with a negative scale for little-endian (see validation/tools/imageio_util.py).
export interface FloatImage { width: number; height: number; channels: 1 | 3; data: Float32Array } // top row first

export function encodePFM(img: FloatImage): Uint8Array {
  const { width: w, height: h, channels: c, data } = img;
  if (data.length !== w * h * c) throw new Error(`PFM: data length ${data.length} != ${w}x${h}x${c}`);
  const header = new TextEncoder().encode(`${c === 3 ? 'PF' : 'Pf'}\n${w} ${h}\n-1.0\n`);
  const out = new Uint8Array(header.length + w * h * c * 4);
  out.set(header, 0);
  const body = new DataView(out.buffer, header.length);
  for (let row = 0; row < h; row++) {
    const srcRow = h - 1 - row; // bottom-to-top
    for (let i = 0; i < w * c; i++) body.setFloat32((row * w * c + i) * 4, data[srcRow * w * c + i], true);
  }
  return out;
}

export function decodePFM(bytes: Uint8Array): FloatImage {
  let pos = 0;
  const readToken = (): string => {
    while (pos < bytes.length && /\s/.test(String.fromCharCode(bytes[pos]))) pos++;
    let s = '';
    while (pos < bytes.length && !/\s/.test(String.fromCharCode(bytes[pos]))) s += String.fromCharCode(bytes[pos++]);
    return s;
  };
  const magic = readToken();
  const channels = magic === 'PF' ? 3 : magic === 'Pf' ? 1 : 0;
  if (!channels) throw new Error(`PFM: bad magic '${magic}'`);
  const w = Number(readToken()), h = Number(readToken()), scale = Number(readToken());
  pos++; // single whitespace after scale
  const little = scale < 0;
  const view = new DataView(bytes.buffer, bytes.byteOffset + pos);
  const data = new Float32Array(w * h * channels);
  for (let row = 0; row < h; row++) {
    const dstRow = h - 1 - row;
    for (let i = 0; i < w * channels; i++) data[dstRow * w * channels + i] = view.getFloat32((row * w * channels + i) * 4, little);
  }
  return { width: w, height: h, channels: channels as 1 | 3, data };
}

/** Test pattern used by the M0 IO-orientation round trip: I(c, r) = (c, r, c + r), row 0 = top. */
export function orientationPattern(width: number, height: number): FloatImage {
  const data = new Float32Array(width * height * 3);
  for (let r = 0; r < height; r++) for (let c = 0; c < width; c++) {
    const o = (r * width + c) * 3; data[o] = c; data[o + 1] = r; data[o + 2] = c + r;
  }
  return { width, height, channels: 3, data };
}
