// ensemble.npz writer (restir-api.md §2.10; validation/tools/README.md "Ensemble .npz format"). OWNER WP-C.
// An .npz is a zip of .npy files; np.load(allow_pickle=False) reads it. This writer produces a stored (method 0,
// uncompressed) zip with NPY format 1.0 members (little-endian, C order, header padded to a multiple of 64 bytes).
// Supported dtypes: <f8, <f4, <u4, <i8 (numeric data) and '<U' (fixed-width unicode strings, UTF-32LE; the width is
// the longest string). readNpz parses what writeNpz writes (tests, tools); no compressed or zip64 archives.

export type NpzNumericDtype = '<f8' | '<f4' | '<u4' | '<i8';
export interface NpzArray {
  name: string;
  shape: number[];
  /** Numeric dtype, or '<U' for string arrays. */
  dtype: NpzNumericDtype | '<U';
  data: ArrayBufferView | string[];
}

const ITEM: Record<NpzNumericDtype, number> = { '<f8': 8, '<f4': 4, '<u4': 4, '<i8': 8 };

/** CRC-32 (IEEE 802.3, reflected, as zip). */
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
export function crc32(b: Uint8Array): number {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < b.length; i++) c = CRC_TABLE[(c ^ b[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}

const count = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

/** One .npy file (format 1.0). */
export function encodeNpy(a: NpzArray): Uint8Array {
  const n = count(a.shape);
  let descr: string;
  let body: Uint8Array;
  if (a.dtype === '<U') {
    const strs = a.data as string[];
    if (strs.length !== n) throw new Error(`npz ${a.name}: ${strs.length} strings for shape ${a.shape}`);
    const cps = strs.map((s) => Array.from(s, (ch) => ch.codePointAt(0)!));
    const w = Math.max(1, ...cps.map((c) => c.length));
    descr = `<U${w}`;
    const u = new Uint32Array(n * w);
    cps.forEach((c, i) => c.forEach((cp, j) => { u[i * w + j] = cp; }));
    body = new Uint8Array(u.buffer);
  } else {
    const v = a.data as ArrayBufferView;
    if (v.byteLength !== n * ITEM[a.dtype]) throw new Error(`npz ${a.name}: ${v.byteLength} bytes for shape ${a.shape} ${a.dtype}`);
    descr = a.dtype;
    body = new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
  }
  const shape = a.shape.length === 0 ? '()' : a.shape.length === 1 ? `(${a.shape[0]},)` : `(${a.shape.join(', ')})`;
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': ${shape}, }`;
  const total = 10 + header.length + 1;
  header += ' '.repeat((64 - (total % 64)) % 64) + '\n';
  const out = new Uint8Array(10 + header.length + body.length);
  out.set([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 1, 0], 0);   // \x93NUMPY v1.0
  new DataView(out.buffer).setUint16(8, header.length, true);
  for (let i = 0; i < header.length; i++) out[10 + i] = header.charCodeAt(i);
  out.set(body, 10 + header.length);
  return out;
}

/** Stored (uncompressed) zip of .npy arrays (`name` → `name.npy`). */
export function writeNpz(arrays: NpzArray[]): Uint8Array {
  const files = arrays.map((a) => ({ name: new TextEncoder().encode(`${a.name}.npy`), data: encodeNpy(a) }));
  const locals: Uint8Array[] = [];
  const centrals: Uint8Array[] = [];
  let offset = 0;
  for (const f of files) {
    const crc = crc32(f.data);
    const lh = new Uint8Array(30 + f.name.length);
    const l = new DataView(lh.buffer);
    l.setUint32(0, 0x04034b50, true); l.setUint16(4, 20, true); l.setUint16(6, 0, true); l.setUint16(8, 0, true);
    l.setUint16(10, 0, true); l.setUint16(12, 0x21, true);    // DOS time 0, date 1980-01-01
    l.setUint32(14, crc, true); l.setUint32(18, f.data.length, true); l.setUint32(22, f.data.length, true);
    l.setUint16(26, f.name.length, true); l.setUint16(28, 0, true);
    lh.set(f.name, 30);
    const ch = new Uint8Array(46 + f.name.length);
    const c = new DataView(ch.buffer);
    c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0, true);
    c.setUint16(10, 0, true); c.setUint16(12, 0, true); c.setUint16(14, 0x21, true);
    c.setUint32(16, crc, true); c.setUint32(20, f.data.length, true); c.setUint32(24, f.data.length, true);
    c.setUint16(28, f.name.length, true); c.setUint16(30, 0, true); c.setUint16(32, 0, true); c.setUint16(34, 0, true);
    c.setUint16(36, 0, true); c.setUint32(38, 0, true); c.setUint32(42, offset, true);
    ch.set(f.name, 46);
    locals.push(lh, f.data);
    centrals.push(ch);
    offset += lh.length + f.data.length;
    if (offset > 0xFFFFFFFF) throw new Error('npz: > 4 GB (zip64 not supported)');
  }
  const cdSize = centrals.reduce((s, b) => s + b.length, 0);
  const eocd = new Uint8Array(22);
  const e = new DataView(eocd.buffer);
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true);
  e.setUint32(12, cdSize, true); e.setUint32(16, offset, true);
  const out = new Uint8Array(offset + cdSize + 22);
  let o = 0;
  for (const b of [...locals, ...centrals, eocd]) { out.set(b, o); o += b.length; }
  return out;
}

export interface NpyParsed { name: string; descr: string; shape: number[]; data: Float64Array | Float32Array | Uint32Array | BigInt64Array | string[] }

/** Parse a zip written by writeNpz (stored members only). */
export function readNpz(zip: Uint8Array): Map<string, NpyParsed> {
  const dv = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
  const out = new Map<string, NpyParsed>();
  let o = 0;
  while (o + 4 <= zip.length && dv.getUint32(o, true) === 0x04034b50) {
    if (dv.getUint16(o + 8, true) !== 0) throw new Error('readNpz: compressed member');
    const size = dv.getUint32(o + 18, true);
    const nlen = dv.getUint16(o + 26, true), xlen = dv.getUint16(o + 28, true);
    const name = new TextDecoder().decode(zip.subarray(o + 30, o + 30 + nlen)).replace(/\.npy$/, '');
    const data = zip.slice(o + 30 + nlen + xlen, o + 30 + nlen + xlen + size);
    if (crc32(data) !== dv.getUint32(o + 14, true)) throw new Error(`readNpz: CRC mismatch in ${name}`);
    out.set(name, parseNpy(name, data));
    o += 30 + nlen + xlen + size;
  }
  return out;
}

function parseNpy(name: string, b: Uint8Array): NpyParsed {
  const hl = new DataView(b.buffer, b.byteOffset).getUint16(8, true);
  const header = new TextDecoder().decode(b.subarray(10, 10 + hl));
  const descr = /'descr': '([^']+)'/.exec(header)![1];
  const shape = /'shape': \(([^)]*)\)/.exec(header)![1].split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  const body = b.slice(10 + hl);
  const n = count(shape);
  if (descr.startsWith('<U')) {
    const w = Number(descr.slice(2));
    const u = new Uint32Array(body.buffer, body.byteOffset, n * w);
    const strs: string[] = [];
    for (let i = 0; i < n; i++) {
      let s = '';
      for (let j = 0; j < w; j++) { const cp = u[i * w + j]; if (cp) s += String.fromCodePoint(cp); }
      strs.push(s);
    }
    return { name, descr, shape, data: strs };
  }
  const buf = body.buffer;
  const data = descr === '<f8' ? new Float64Array(buf, 0, n) : descr === '<f4' ? new Float32Array(buf, 0, n)
    : descr === '<u4' ? new Uint32Array(buf, 0, n) : new BigInt64Array(buf, 0, n);
  return { name, descr, shape, data };
}
