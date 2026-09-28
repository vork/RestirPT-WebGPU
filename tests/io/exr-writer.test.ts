// EXR writer (src/core/io/exr.ts) and PNG codec (src/core/io/png.ts): read back through OpenImageIO in the
// validation venv (skipped when the venv is missing), and through our own decoders (three EXRLoader, decodePng).
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { encodeExr, exrZipPredict, flipRows } from '../../src/core/io/exr.ts';
import { decodePng, encodePng } from '../../src/core/io/png.ts';
import { decodeExr } from '../../src/core/scene/env/exr.ts';

const PY = path.resolve('validation/.venv/bin/python');
const hasOiio = existsSync(PY) && spawnSync(PY, ['-c', 'import OpenImageIO'], { encoding: 'utf8' }).status === 0;
const dir = mkdtempSync(path.join(tmpdir(), 'exr-writer-'));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** Deterministic RGBA test image with awkward values (denormals, huge, 1 ulp steps), rows top-down. */
function pattern(W: number, H: number): Float32Array {
  const d = new Float32Array(W * H * 4);
  const u = new Uint32Array(d.buffer);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const o = (y * W + x) * 4;
    d[o] = x + 0.001 * y;          // encodes (x, y) for orientation
    d[o + 1] = y;
    u[o + 2] = (0x3f800000 + x * 977 + y * 131) >>> 0; // 1 + k ulp
    d[o + 3] = (x + y) % 7 === 0 ? 1e-40 : (x * y) % 5 === 0 ? 3e38 : 1;
  }
  return d;
}

function oiioRead(file: string): { w: number; h: number; ch: string[]; comp: string; data: Float32Array } {
  const code = [
    'import sys, OpenImageIO as oiio, numpy as np',
    'i = oiio.ImageInput.open(sys.argv[1]); s = i.spec()',
    'a = i.read_image(0, 0, 0, s.nchannels, "float"); i.close()',
    'sys.stdout.write(f"{s.width} {s.height} {s.get_string_attribute(\'compression\')} {\',\'.join(s.channelnames)}\\n")',
    'sys.stdout.flush(); sys.stdout.buffer.write(np.ascontiguousarray(a, dtype="<f4").tobytes())',
  ].join('\n');
  const r = spawnSync(PY, ['-c', code, file], { maxBuffer: 1 << 28 });
  if (r.status !== 0) throw new Error(r.stderr.toString());
  const out = r.stdout;
  const nl = out.indexOf(10);
  const [w, h, comp, ch] = out.subarray(0, nl).toString().split(' ');
  const body = out.subarray(nl + 1);
  const data = new Float32Array(body.buffer.slice(body.byteOffset, body.byteOffset + body.byteLength));
  return { w: Number(w), h: Number(h), comp, ch: ch.split(','), data };
}

describe('EXR writer', () => {
  it('ZIP predictor matches the OpenEXR reference on a tiny vector', () => {
    // bytes 0..5 → interleave [0,2,4 | 1,3,5] → deltas (+128 mod 256)
    expect(Array.from(exrZipPredict(Uint8Array.from([0, 1, 2, 3, 4, 5])))).toEqual([0, 130, 130, 125, 130, 130]);
  });

  for (const compression of ['none', 'zip'] as const) {
    for (const [W, H] of [[37, 21], [64, 32]] as const) {
      it(`${compression} ${W}x${H}: three EXRLoader reads the exact bits (rows bottom-up after decode)`, async () => {
        const src = pattern(W, H);
        // decodeExr (validation) rejects alpha != 1: write alpha 1 for this check
        const rgba1 = src.slice(); for (let i = 3; i < rgba1.length; i += 4) rgba1[i] = 1;
        const bytes = await encodeExr({ width: W, height: H, data: rgba1, channels: ['R', 'G', 'B', 'A'] }, compression);
        const dec = decodeExr(bytes, 'validation');
        expect(dec.info.compression).toBe(compression === 'zip' ? 'ZIP_COMPRESSION' : 'NO_COMPRESSION');
        expect([dec.width, dec.height]).toEqual([W, H]);
        expect(new Uint32Array(flipRows(dec.texels, W, H).buffer)).toEqual(new Uint32Array(rgba1.buffer));
      });

      it.skipIf(!hasOiio)(`${compression} ${W}x${H}: OpenImageIO reads the exact bits, rows top-down, RGBA`, async () => {
        const src = pattern(W, H);
        const bytes = await encodeExr({ width: W, height: H, data: src, channels: ['R', 'G', 'B', 'A'] }, compression);
        const f = path.join(dir, `p-${compression}-${W}.exr`);
        writeFileSync(f, bytes);
        const r = oiioRead(f);
        expect([r.w, r.h]).toEqual([W, H]);
        expect(r.comp).toBe(compression === 'zip' ? 'zip' : 'none');
        expect(r.ch).toEqual(['R', 'G', 'B', 'A']);
        expect(new Uint32Array(r.data.buffer)).toEqual(new Uint32Array(src.buffer));
        expect(r.data[(3 * W + 5) * 4]).toBeCloseTo(5.003, 6); // row 3 (from the top), column 5
      });
    }
  }
});

describe('PNG codec', () => {
  it('round-trips RGBA8 exactly through decodePng', async () => {
    const W = 33, H = 17;
    const px = new Uint8Array(W * H * 4);
    for (let i = 0; i < px.length; i++) px[i] = (i * 37 + (i >> 7) * 11) & 0xff;
    const png = await encodePng({ width: W, height: H, pixels: px });
    const back = await decodePng(png);
    expect([back.width, back.height]).toEqual([W, H]);
    expect(back.pixels).toEqual(px);
  });

  it.skipIf(!hasOiio)('OpenImageIO reads our PNG bytes unchanged', async () => {
    const W = 20, H = 9;
    const px = new Uint8Array(W * H * 4);
    for (let i = 0; i < px.length; i++) px[i] = (i * 53) & 0xff;
    const f = path.join(dir, 't.png');
    writeFileSync(f, await encodePng({ width: W, height: H, pixels: px }));
    // UnassociatedAlpha: OIIO premultiplies PNG alpha on read by default
    const code = 'import sys, OpenImageIO as oiio, numpy as np\nc = oiio.ImageSpec(); c.attribute("oiio:UnassociatedAlpha", 1)\ni = oiio.ImageInput.open(sys.argv[1], c); a = i.read_image(0,0,0,4,"uint8"); i.close()\nsys.stdout.buffer.write(np.ascontiguousarray(a).tobytes())';
    const r = spawnSync(PY, ['-c', code, f]);
    expect(r.status, r.stderr?.toString()).toBe(0);
    expect(new Uint8Array(r.stdout)).toEqual(px);
  });
});
