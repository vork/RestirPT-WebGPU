// zlib (RFC 1950) deflate/inflate through the Compression Streams API, available in Chrome, Workers and Node ≥ 18.
// Used by the PNG codec (IDAT) and the EXR writer (ZIP_COMPRESSION blocks are zlib `compress()` streams).

async function pump(data: Uint8Array, stream: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const body = new Blob([data as Uint8Array<ArrayBuffer>]).stream().pipeThrough(stream);
  return new Uint8Array(await new Response(body).arrayBuffer());
}

/** zlib-wrapped deflate (what zlib `compress2` produces). */
export function zlibDeflate(data: Uint8Array): Promise<Uint8Array> {
  return pump(data, new CompressionStream('deflate'));
}

/** Inverse of zlibDeflate. */
export function zlibInflate(data: Uint8Array): Promise<Uint8Array> {
  return pump(data, new DecompressionStream('deflate'));
}

/** SHA-256 (hex) via WebCrypto (browser, Worker, Node ≥ 19). */
export async function sha256Hex(data: Uint8Array | ArrayBufferView): Promise<string> {
  const u8 = data instanceof Uint8Array ? data : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const digest = await crypto.subtle.digest('SHA-256', u8 as Uint8Array<ArrayBuffer>);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}
