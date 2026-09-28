// OpenEXR env loading via three.js EXRLoader used purely as a PARSER (FloatType; half→float is exact)
// (plan §1.1, §1.4b; math.md#env-mapping). EXRLoader writes rows bottom-up (outLineOffset = (H−1−y)·W), which is
// already our canonical order (row 0 = nadir). Policy (env §2.1, env-verify):
//   validation:  lossless codecs only; alpha ≡ 1 (or absent); NaN/Inf/negative texels rejected.
//   interactive: lossy codecs allowed (warning); NaN/Inf → 0 and negatives → 0 with a warning.
// Always rejected: multi-part, deep, mip/rip-tiled, data window ≠ display window, luminance-only chroma (YC).
import { FloatType, RGBAFormat } from 'three';
import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';
import type { DecodedEnvImage } from './hdr.ts';

export type EnvLoadMode = 'validation' | 'interactive';

export const EXR_LOSSLESS = new Set(['NO_COMPRESSION', 'RLE_COMPRESSION', 'ZIPS_COMPRESSION', 'ZIP_COMPRESSION', 'PIZ_COMPRESSION']);

export interface ExrInfo { compression: string; channels: string[]; pixelTypes: number[]; lossy: boolean }

export class ExrDecodeError extends Error {}

export function isExr(bytes: Uint8Array): boolean {
  return bytes.length >= 4 && bytes[0] === 0x76 && bytes[1] === 0x2f && bytes[2] === 0x31 && bytes[3] === 0x01;
}

interface Box2i { xMin: number; yMin: number; xMax: number; yMax: number }
interface ExrHeaderLike {
  compression: string;
  channels: { name: string; pixelType: number }[];
  dataWindow: Box2i;
  displayWindow: Box2i;
  spec: { singleTile: boolean; deepFormat: boolean; multiPart: boolean };
  tiles?: { levelMode: string };
  chromaticities?: { redX: number; redY: number; greenX: number; greenY: number; blueX: number; blueY: number; whiteX: number; whiteY: number };
}

const REC709 = [0.64, 0.33, 0.3, 0.6, 0.15, 0.06, 0.3127, 0.329];

export function decodeExr(input: ArrayBuffer | Uint8Array, mode: EnvLoadMode): DecodedEnvImage & { info: ExrInfo } {
  const u8 = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (!isExr(u8)) throw new ExrDecodeError('not an OpenEXR file');
  const buffer = u8.byteOffset === 0 && u8.byteLength === u8.buffer.byteLength ? (u8.buffer as ArrayBuffer) : (u8.slice().buffer as ArrayBuffer);
  const warnings: string[] = [];
  const loader = new EXRLoader().setDataType(FloatType).setOutputFormat(RGBAFormat);
  let parsed;
  try { parsed = loader.parse(buffer) as ReturnType<EXRLoader['parse']> & { header: ExrHeaderLike }; }
  catch (e) { throw new ExrDecodeError(`EXRLoader: ${(e as Error).message}`); }
  const h = parsed.header;
  const width = parsed.width!, height = parsed.height!;
  const data = parsed.data as Float32Array;
  const channels = h.channels.map((c) => c.name);
  const info: ExrInfo = { compression: h.compression, channels, pixelTypes: h.channels.map((c) => c.pixelType), lossy: !EXR_LOSSLESS.has(h.compression) };

  if (h.spec.multiPart || h.spec.deepFormat) throw new ExrDecodeError('multi-part/deep EXR not supported');
  if (h.spec.singleTile && h.tiles && h.tiles.levelMode !== 'ONE_LEVEL') throw new ExrDecodeError(`tiled EXR with ${h.tiles.levelMode} not supported`);
  const dw = h.dataWindow, dsp = h.displayWindow;
  if (dw.xMin !== dsp.xMin || dw.yMin !== dsp.yMin || dw.xMax !== dsp.xMax || dw.yMax !== dsp.yMax) {
    throw new ExrDecodeError('data window != display window');
  }
  if (!(channels.includes('R') && channels.includes('G') && channels.includes('B')) && !(channels.length === 1 && channels[0] === 'Y')) {
    throw new ExrDecodeError(`unsupported channels [${channels.join(',')}] (need R,G,B or Y)`);
  }
  if (info.lossy) {
    if (mode === 'validation') throw new ExrDecodeError(`lossy codec ${h.compression} not allowed in validation`);
    warnings.push(`lossy EXR codec ${h.compression}`);
  }
  if (h.chromaticities) {
    const c = h.chromaticities;
    const got = [c.redX, c.redY, c.greenX, c.greenY, c.blueX, c.blueY, c.whiteX, c.whiteY];
    if (got.some((v, i) => Math.abs(v - REC709[i]) > 1e-3)) warnings.push(`EXR chromaticities are not Rec.709 (${got.map((v) => v.toFixed(4)).join(',')}); used as-is`);
  }
  if (width * height * 4 !== data.length) throw new ExrDecodeError('unexpected EXRLoader output size');

  let nonFinite = 0, negative = 0, alphaNot1 = 0;
  for (let i = 0; i < data.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const v = data[i + c];
      if (!Number.isFinite(v)) { nonFinite++; data[i + c] = 0; }
      else if (v < 0) { negative++; data[i + c] = 0; }
    }
    if (data[i + 3] !== 1) { alphaNot1++; data[i + 3] = 1; }
  }
  if (mode === 'validation') {
    if (nonFinite) throw new ExrDecodeError(`${nonFinite} NaN/Inf texel channels (rejected in validation)`);
    if (negative) throw new ExrDecodeError(`${negative} negative texel channels (rejected in validation)`);
    if (alphaNot1) throw new ExrDecodeError(`${alphaNot1} texels with alpha != 1 (validation requires opaque maps)`);
  } else {
    if (nonFinite) warnings.push(`${nonFinite} NaN/Inf texel channels replaced by 0`);
    if (negative) warnings.push(`${negative} negative texel channels clamped to 0`);
    if (alphaNot1) warnings.push(`${alphaNot1} texels with alpha != 1; alpha ignored`);
  }
  return { width, height, texels: data, warnings, info };
}
