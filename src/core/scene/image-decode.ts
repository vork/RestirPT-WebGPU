// Browser/Worker image decode to raw RGBA8 (plan §1.6; gap-bsdf §9 item 7).
// Bytes are uploaded unmodified: no colour-space conversion, no premultiplication, rows top-first.
// Works on the main thread and in Workers (createImageBitmap + OffscreenCanvas). Not available in Node:
// Node callers pass no decoder and the loader keeps texture sizes only (see gltf-loader.ts).

export interface DecodedImage {
  width: number;
  height: number;
  /** RGBA8, rows top-first, straight (non-premultiplied) alpha. */
  pixels: Uint8Array;
}

/** Returns null (and the loader logs a warning) for formats we cannot decode. */
export type ImageDecoder = (bytes: Uint8Array, mimeType: string, name: string) => Promise<DecodedImage | null>;

/** MIME types createImageBitmap decodes in Chrome. KTX2/Basis is not supported in v1 (warned by the loader). */
export const DECODABLE_MIME = new Set(['image/png', 'image/jpeg', 'image/webp', 'image/avif', 'image/gif', 'image/bmp']);

export function canDecodeInThisRuntime(): boolean {
  return typeof createImageBitmap === 'function' && typeof OffscreenCanvas === 'function';
}

/**
 * createImageBitmap(blob, {colorSpaceConversion:'none', premultiplyAlpha:'none'}) → OffscreenCanvas 2D →
 * getImageData({colorSpace:'srgb'}). The 2D canvas stores premultiplied pixels internally, so texels with
 * 0 < α < 255 may lose colour precision; validation textures have α ≡ 1 (plan §1.6), so parity is unaffected.
 */
export const browserImageDecoder: ImageDecoder = async (bytes, mimeType) => {
  if (!DECODABLE_MIME.has(mimeType)) return null;
  const blob = new Blob([bytes as Uint8Array<ArrayBuffer>], { type: mimeType });
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  try {
    const { width, height } = bmp;
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true, colorSpace: 'srgb' });
    if (!ctx) throw new Error('OffscreenCanvas 2D context unavailable');
    ctx.globalCompositeOperation = 'copy';
    ctx.drawImage(bmp, 0, 0);
    const data = ctx.getImageData(0, 0, width, height, { colorSpace: 'srgb' }).data;
    return { width, height, pixels: new Uint8Array(data.buffer, data.byteOffset, data.byteLength) };
  } finally {
    bmp.close();
  }
};

/** True when every alpha byte is 255 (used for the TRI_ALPHA_MASK "alpha not constant 1" test). */
export function isAlphaOpaque(pixels: Uint8Array): boolean {
  for (let i = 3; i < pixels.length; i += 4) if (pixels[i] !== 255) return false;
  return true;
}
