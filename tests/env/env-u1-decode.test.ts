// ENV-U1 (plan §7.4 M1; env §5): our RGBE decoder (and the EXRLoader path) are bit-identical to OpenImageIO,
// the library Blender uses. Also documents that three.js HDRLoader is +0.39% (256/255) bright.
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { FloatType } from 'three';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';
import { describe, expect, it } from 'vitest';
import { decodeHdr } from '../../src/core/scene/env/hdr.ts';
import { decodeExr } from '../../src/core/scene/env/exr.ts';
import { decodeEnvironment } from '../../src/core/scene/env/load-env.ts';
import { HDRI_IDS, hdriPath, verifyLocal } from '../../validation/assets/fetch_hdris.ts';
import { encodeHdr, prng } from './rgbe-synth.ts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const python = join(root, 'validation/.venv/bin/python');
const script = join(root, 'validation/tools/env_ref.py');
const outDir = join(root, 'validation/out/test/env');
const hasVenv = existsSync(python);
if (!hasVenv) console.warn(`[env-u1] OIIO comparisons skipped: ${python} missing`);

interface OiioDump { width: number; height: number; channels: number; channelnames: string[]; compression: string; data: Float32Array }

function oiioDump(path: string): OiioDump {
  mkdirSync(outDir, { recursive: true });
  const raw = join(outDir, `${path.split('/').pop()}.f32`);
  const res = spawnSync(python, [script, 'dump', path, raw], { encoding: 'utf8' });
  if (res.status !== 0) throw new Error(`env_ref.py failed: ${res.stdout}${res.stderr}`);
  const meta = JSON.parse(res.stdout.trim().split('\n').pop()!);
  const b = readFileSync(raw);
  return { ...meta, data: new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4) };
}

/** Compare our RGBA bottom-up texels with OIIO's top-down C-channel dump. Returns mismatch count and max |Δbits|. */
function compareBits(w: number, h: number, ours: Float32Array, ref: OiioDump) {
  expect(ref.width).toBe(w);
  expect(ref.height).toBe(h);
  const ob = new Uint32Array(ours.buffer, ours.byteOffset, ours.length);
  const rb = new Uint32Array(ref.data.buffer, ref.data.byteOffset, ref.data.length);
  let mismatches = 0, maxUlp = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const o = (((h - 1 - y) * w) + x) * 4; // ours: bottom-up
      const r = (y * w + x) * ref.channels;  // OIIO: top-down
      for (let c = 0; c < 3; c++) {
        const a = ob[o + c], b = rb[r + (ref.channels === 1 ? 0 : c)];
        if (a !== b) { mismatches++; maxUlp = Math.max(maxUlp, Math.abs((a | 0) - (b | 0))); }
      }
    }
  }
  return { mismatches, maxUlp };
}

describe('ENV-U1 RGBE decoder', () => {
  it('decodes known RGBE bytes as byte*2^(e-136), e=0 -> 0, rows bottom-up', () => {
    // 2x1 image would be flat; use width 1, height 2 (flat): top pixel then bottom pixel.
    const rgbe = new Uint8Array([128, 64, 1, 136, /* bottom row */ 255, 7, 9, 0]);
    const d = decodeHdr(encodeHdr(1, 2, rgbe, 'flat'));
    expect(Array.from(d.texels)).toEqual([0, 0, 0, 1, 128, 64, 1, 1]); // row 0 = bottom (e = 0 -> 0 despite bytes)
    const ext = decodeHdr(encodeHdr(1, 2, new Uint8Array([255, 1, 0, 255, 1, 2, 255, 1]), 'flat'));
    expect(ext.texels[4]).toBe(255 * 2 ** 119);   // top row (file pixel 0), e = 255
    expect(ext.texels[0]).toBe(2 ** -135);        // bottom row (file pixel 1), e = 1: f32 subnormals, exact
    expect(ext.texels[1]).toBe(2 * 2 ** -135);
    expect(ext.texels[2]).toBe(255 * 2 ** -135);
  });

  it('RLE and flat encodings of the same data decode identically; rejects unsupported headers', () => {
    const w = 37, h = 11, next = prng(7);
    const rgbe = new Uint8Array(w * h * 4);
    for (let i = 0; i < rgbe.length; i++) rgbe[i] = (i % 17 < 6) ? 200 : next() & 0xff; // mix of runs and literals
    const a = decodeHdr(encodeHdr(w, h, rgbe, 'rle'));
    const b = decodeHdr(encodeHdr(w, h, rgbe, 'flat'));
    expect(Array.from(a.texels)).toEqual(Array.from(b.texels));
    const bad = new TextEncoder().encode('#?RADIANCE\nFORMAT=32-bit_rle_xyze\n\n-Y 1 +X 1\n\0\0\0\0');
    expect(() => decodeHdr(bad)).toThrow(/FORMAT/);
    const flipped = new TextEncoder().encode('#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n+Y 1 +X 1\n\0\0\0\0');
    expect(() => decodeHdr(flipped)).toThrow(/resolution/);
  });

  describe.skipIf(!hasVenv)('synthetic files vs OIIO (bitwise)', () => {
    const cases: [string, number, number, 'flat' | 'rle', (r: Uint8Array) => void][] = [
      ['rle-random', 64, 9, 'rle', () => {}],
      ['flat-narrow', 5, 7, 'flat', () => {}],                 // width < 8: always flat
      ['flat-wide', 40, 6, 'flat', (r) => { r[0] = 9; }],      // first scanline not RLE -> flat remainder
      ['flat-oldstyle-run', 12, 3, 'flat', (r) => { r[0] = 9; r.set([1, 1, 1, 3], 16); }], // Ward old-style run marker
      // every exponent 0..255 with mantissas 1, 255 and random: pins OIIO's f32 exponent table (e = 40, 223, 226)
      ['exp-table', 256, 3, 'rle', (r) => { for (let x = 0; x < 256; x++) { r[x * 4 + 3] = r[(256 + x) * 4 + 3] = r[(512 + x) * 4 + 3] = x; r.fill(1, x * 4, x * 4 + 3); r.fill(255, (256 + x) * 4, (256 + x) * 4 + 3); } }],
    ];
    for (const [name, w, h, mode, tweak] of cases) {
      it(name, () => {
        const next = prng(name.length * 977);
        const rgbe = new Uint8Array(w * h * 4);
        for (let i = 0; i < rgbe.length; i++) rgbe[i] = next() & 0xff;
        for (let i = 3; i < rgbe.length; i += 32) rgbe[i] = 0;   // some e = 0 with non-zero mantissas
        for (let i = 7; i < rgbe.length; i += 40) rgbe[i] = 1;   // subnormals
        tweak(rgbe);
        const file = join(outDir, `synth-${name}.hdr`);
        mkdirSync(outDir, { recursive: true });
        writeFileSync(file, encodeHdr(w, h, rgbe, mode));
        const ours = decodeHdr(readFileSync(file));
        const { mismatches, maxUlp } = compareBits(w, h, ours.texels, oiioDump(file));
        console.log(`ENV-U1 synth ${name}: mismatches=${mismatches} maxUlp=${maxUlp}`);
        expect(mismatches).toBe(0);
      });
    }
  });

  for (const id of HDRI_IDS) {
    const have = hasVenv && verifyLocal(id, 'hdr');
    if (!have) console.warn(`[env-u1] ${id}.hdr skipped (run: npx tsx validation/assets/fetch_hdris.ts)`);
    it.skipIf(!have)(`${id}_1k.hdr: ours == OIIO bitwise; three.js HDRLoader is +0.39%`, () => {
      const path = hdriPath(id, 'hdr');
      const bytes = readFileSync(path);
      const ours = decodeHdr(bytes);
      const ref = oiioDump(path);
      const { mismatches, maxUlp } = compareBits(ours.width, ours.height, ours.texels, ref);
      // three.js HDRLoader (top-down rows, RGBA): ratio over pixels with e != 0.
      const three = new HDRLoader().setDataType(FloatType).parse(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
      const td = three.data as Float32Array;
      const { width: w, height: h } = ours;
      let sOurs = 0, sThree = 0, maxRel = 0;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const o = ((h - 1 - y) * w + x) * 4, t = (y * w + x) * 4;
        for (let c = 0; c < 3; c++) {
          const a = ours.texels[o + c], b = td[t + c];
          if (a > 0) { sOurs += a; sThree += b; maxRel = Math.max(maxRel, Math.abs(b / a - 256 / 255)); }
        }
      }
      const ratio = sThree / sOurs;
      console.log(`ENV-U1 ${id}.hdr: ${w}x${h} mismatches=${mismatches} maxUlp=${maxUlp} oiioSha=${(ref as unknown as { sha256: string }).sha256.slice(0, 16)} three/ours=${ratio.toFixed(7)} (+${((ratio - 1) * 100).toFixed(4)}%) maxDev(256/255)=${maxRel.toExponential(2)}`);
      expect(mismatches).toBe(0);
      expect(maxUlp).toBe(0);
      expect(Math.abs(ratio - 256 / 255)).toBeLessThan(1e-6);
    });
  }

  for (const id of HDRI_IDS) {
    const have = hasVenv && verifyLocal(id, 'exr');
    it.skipIf(!have)(`${id}_1k.exr: EXRLoader path == OIIO bitwise (validation mode)`, () => {
      const path = hdriPath(id, 'exr');
      const ours = decodeExr(readFileSync(path), 'validation');
      const ref = oiioDump(path);
      const { mismatches, maxUlp } = compareBits(ours.width, ours.height, ours.texels, ref);
      console.log(`ENV-U1 ${id}.exr: ${ours.width}x${ours.height} ${ours.info.compression} channels=${ours.info.channels.join('')} pixelTypes=${ours.info.pixelTypes.join('')} mismatches=${mismatches} maxUlp=${maxUlp}`);
      expect(mismatches).toBe(0);
      // .hdr and .exr of the same asset are both published; record how close they are (informational).
      if (verifyLocal(id, 'hdr')) {
        const hdr = decodeEnvironment(readFileSync(hdriPath(id, 'hdr')), { mode: 'validation' }).env.texels;
        let s0 = 0, s1 = 0;
        for (let i = 0; i < hdr.length; i += 4) for (let c = 0; c < 3; c++) { s0 += hdr[i + c]; s1 += ours.texels[i + c]; }
        console.log(`ENV-U1 ${id}: mean(exr)/mean(hdr) = ${(s1 / s0).toFixed(6)}`);
      }
    });
  }
});

describe('EXR policy', () => {
  it('rejects negative texels in validation and clamps them interactively', async () => {
    // Build a tiny uncompressed float EXR by hand: 2x1, channels B,G,R (alphabetical), FLOAT.
    const exr = tinyExr(2, 1, [[-1, 0.5, 2], [0.25, 3, 4]]);
    expect(() => decodeExr(exr, 'validation')).toThrow(/negative/);
    const r = decodeExr(exr, 'interactive');
    expect(r.warnings.join()).toMatch(/negative/);
    expect(Array.from(r.texels)).toEqual([0, 0.5, 2, 1, 0.25, 3, 4, 1]);
  });
});

/** Minimal scanline, NO_COMPRESSION, FLOAT RGB EXR. pixels[x] = [r,g,b], single row (y = 0 top). */
function tinyExr(w: number, h: number, pixels: [number, number, number][]): Uint8Array {
  const parts: number[] = [];
  const u8 = (...b: number[]) => parts.push(...b);
  const i32 = (v: number) => { const a = new DataView(new ArrayBuffer(4)); a.setInt32(0, v, true); u8(...new Uint8Array(a.buffer)); };
  const f32 = (v: number) => { const a = new DataView(new ArrayBuffer(4)); a.setFloat32(0, v, true); u8(...new Uint8Array(a.buffer)); };
  const str = (s: string) => u8(...new TextEncoder().encode(s), 0);
  const attr = (name: string, type: string, body: number[]) => { str(name); str(type); i32(body.length); u8(...body); };
  const bytesOf = (fn: () => void) => { const start = parts.length; fn(); return parts.splice(start); };
  u8(0x76, 0x2f, 0x31, 0x01, 2, 0, 0, 0);
  attr('channels', 'chlist', bytesOf(() => { for (const c of ['B', 'G', 'R']) { str(c); i32(2); u8(0, 0, 0, 0); i32(1); i32(1); } u8(0); }));
  attr('compression', 'compression', [0]);
  const box = bytesOf(() => { i32(0); i32(0); i32(w - 1); i32(h - 1); });
  attr('dataWindow', 'box2i', box);
  attr('displayWindow', 'box2i', box);
  attr('lineOrder', 'lineOrder', [0]);
  attr('pixelAspectRatio', 'float', bytesOf(() => f32(1)));
  attr('screenWindowCenter', 'v2f', bytesOf(() => { f32(0); f32(0); }));
  attr('screenWindowWidth', 'float', bytesOf(() => f32(1)));
  u8(0);
  const tableAt = parts.length;
  for (let y = 0; y < h; y++) { i32(0); i32(0); } // offsets (u64), patched below
  const offsets: number[] = [];
  for (let y = 0; y < h; y++) {
    offsets.push(parts.length);
    i32(y); i32(w * 3 * 4);
    for (const c of [2, 1, 0]) for (let x = 0; x < w; x++) f32(pixels[y * w + x][c]);
  }
  const out = new Uint8Array(parts);
  const dv = new DataView(out.buffer);
  offsets.forEach((o, y) => { dv.setUint32(tableAt + y * 8, o, true); dv.setUint32(tableAt + y * 8 + 4, 0, true); });
  return out;
}
