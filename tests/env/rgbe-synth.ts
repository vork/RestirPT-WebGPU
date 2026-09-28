// Test helper: write Radiance .hdr files from raw RGBE bytes (top-down rows), flat or new-style RLE.
export function encodeHdr(width: number, height: number, rgbe: Uint8Array, mode: 'flat' | 'rle', extraHeader: string[] = []): Uint8Array {
  const head = new TextEncoder().encode(['#?RADIANCE', 'FORMAT=32-bit_rle_rgbe', ...extraHeader, '', `-Y ${height} +X ${width}`, ''].join('\n'));
  const body: number[] = [];
  if (mode === 'flat') body.push(...rgbe);
  else {
    if (width < 8 || width > 0x7fff) throw new Error('RLE needs 8 <= width <= 32767');
    for (let y = 0; y < height; y++) {
      body.push(2, 2, width >> 8, width & 0xff);
      for (let c = 0; c < 4; c++) {
        const ch = Array.from({ length: width }, (_, x) => rgbe[(y * width + x) * 4 + c]);
        let x = 0;
        while (x < width) {
          let run = 1;
          while (x + run < width && run < 127 && ch[x + run] === ch[x]) run++;
          if (run >= 3) { body.push(128 + run, ch[x]); x += run; continue; }
          // literal until the next run of >= 3 (max 128)
          let n = 0;
          while (x + n < width && n < 128) {
            if (x + n + 2 < width && ch[x + n] === ch[x + n + 1] && ch[x + n] === ch[x + n + 2]) break;
            n++;
          }
          if (n === 0) n = 1;
          body.push(n, ...ch.slice(x, x + n));
          x += n;
        }
      }
    }
  }
  const out = new Uint8Array(head.length + body.length);
  out.set(head);
  out.set(body, head.length);
  return out;
}

/** Deterministic xorshift32 byte stream. */
export function prng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s; };
}
