// Test helpers: tiny PNG encoder (Node zlib) and synthetic glTF documents written to GLB.
import { crc32, deflateSync } from 'node:zlib';
import { Document, WebIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS } from '@gltf-transform/extensions';

/** Minimal RGBA8 PNG (no filtering). */
export function encodePng(width: number, height: number, rgba: Uint8Array): Uint8Array<ArrayBuffer> {
  const chunk = (type: string, data: Uint8Array) => {
    const out = Buffer.alloc(12 + data.length);
    out.writeUInt32BE(data.length, 0);
    out.write(type, 4, 'ascii');
    Buffer.from(data).copy(out, 8);
    out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)) >>> 0, 8 + data.length);
    return out;
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; ihdr[9] = 6; // 8-bit RGBA
  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; y++) Buffer.from(rgba.subarray(y * width * 4, (y + 1) * width * 4)).copy(raw, y * (width * 4 + 1) + 1);
  const sig = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  return new Uint8Array(Buffer.concat([sig, chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', new Uint8Array(0))]));
}

export async function toGlb(doc: Document, deps: Record<string, unknown> = {}): Promise<Uint8Array> {
  return new WebIO().registerExtensions(ALL_EXTENSIONS).registerDependencies(deps).writeBinary(doc);
}

/** Document with one buffer and a helper to add float/uint accessors. */
export function newDoc() {
  const doc = new Document();
  const buffer = doc.createBuffer();
  const f32 = (type: 'VEC2' | 'VEC3' | 'VEC4', data: number[]) => doc.createAccessor().setType(type).setArray(new Float32Array(data)).setBuffer(buffer);
  const u32 = (data: number[]) => doc.createAccessor().setType('SCALAR').setArray(new Uint32Array(data)).setBuffer(buffer);
  const scene = doc.createScene('s');
  doc.getRoot().setDefaultScene(scene);
  return { doc, buffer, f32, u32, scene };
}
