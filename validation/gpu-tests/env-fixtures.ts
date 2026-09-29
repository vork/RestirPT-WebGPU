// Shared f64 references for the env-lighting tests (M3c): Cycles-identical mapping, the GPU's bilinear repeat lookup
// (Apple GPUs quantize the bilinear fraction to 8 bits, round to nearest — ENV-U7; Cycles on Metal uses the same
// sampler), synthetic maps and HDRI loading for both GPU lanes.
import type { EnvironmentData } from '../../src/core/scene/types.ts';
import { decodeHdr } from '../../src/core/scene/env/hdr.ts';

export type V3 = [number, number, number];

/** b = R_z(γ)·C·d (math.md#env-mapping). */
export function envToBlender(d: V3, g: number): V3 {
  const cg = Math.cos(g), sg = Math.sin(g);
  return [cg * d[0] + sg * d[2], sg * d[0] - cg * d[2], d[1]];
}

/** Cycles direction_to_equirectangular after R_z(γ)·C (f64). */
export function envUV(d: V3, g: number): [number, number] {
  const b = envToBlender(d, g);
  const phi = b[0] === 0 && b[1] === 0 ? 0 : Math.atan2(b[1], b[0]);
  const theta = Math.atan2(Math.hypot(b[0], b[1]), b[2]);
  return [(phi - Math.PI) / (-2 * Math.PI), (theta - Math.PI) / -Math.PI];
}

/** Inverse mapping (math.md#env-mapping envDir). */
export function envDir(u: number, v: number, g: number): V3 {
  const phi = -2 * Math.PI * u + Math.PI, theta = -Math.PI * v + Math.PI;
  const b: V3 = [Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)];
  const cg = Math.cos(g), sg = Math.sin(g);
  const rx = cg * b[0] + sg * b[1], ry = -sg * b[0] + cg * b[1];
  return [rx, b[2], -ry];
}

/** GPU-equivalent bilinear lookup, repeat/repeat, rows bottom-up, fraction quantized to `bits` (8 on Apple GPUs). */
export function envLookup(env: Pick<EnvironmentData, 'texels' | 'width' | 'height'>, u: number, v: number, bits: number | undefined = 8): V3 {
  const W = env.width, H = env.height, t = env.texels;
  const x = u * W - 0.5, y = v * H - 0.5;
  const x0 = Math.floor(x), y0 = Math.floor(y);
  let fx = x - x0, fy = y - y0;
  if (bits !== undefined) { const s = 2 ** bits; fx = Math.round(fx * s) / s; fy = Math.round(fy * s) / s; }
  const at = (c: number, r: number, k: number) => t[4 * ((((r % H) + H) % H) * W + (((c % W) + W) % W)) + k];
  const out: V3 = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    out[k] = (at(x0, y0, k) * (1 - fx) + at(x0 + 1, y0, k) * fx) * (1 - fy) + (at(x0, y0 + 1, k) * (1 - fx) + at(x0 + 1, y0 + 1, k) * fx) * fy;
  }
  return out;
}

/** Synthetic HDR map (rows bottom-up): sky gradient + dim ground + very bright texels (incl. the seam and pole rows). */
export function synthEnvData(W = 128, H = 64, name = 'synth'): EnvironmentData {
  const t = new Float32Array(W * H * 4);
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const v = (r + 0.5) / H, u = (c + 0.5) / W;
      const sky = v > 0.5 ? 0.5 + 2 * (v - 0.5) + 0.3 * Math.sin(6 * Math.PI * u) ** 2 : 0.05 + 0.02 * u;
      t.set([sky, 0.8 * sky + 0.01, 0.6 * sky + 0.05 * u, 1], 4 * (r * W + c));
    }
  }
  const hot = (c: number, r: number, val: number) => t.set([val, 0.9 * val, 0.7 * val, 1], 4 * (r * W + c));
  hot(Math.round(W * 0.3), Math.round(H * 0.75), 2000);
  hot(0, Math.round(H * 0.6), 500);
  hot(W - 1, Math.round(H * 0.6), 500);
  hot(Math.round(W * 0.7), H - 1, 300);
  hot(Math.round(W * 0.1), 0, 100);
  return { name, width: W, height: H, texels: t, strength: 1, tint: [1, 1, 1], rotationZ: 0, visibleToCamera: true };
}

/** C0s-style map: upper hemisphere 1, lower 0, one texel `val` at (col, row) (rows bottom-up). */
export function sunTexelEnv(W: number, H: number, col: number, row: number, val = 1e4): EnvironmentData {
  const t = new Float32Array(W * H * 4);
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) { const up = r >= H / 2 ? 1 : 0; t.set([up, up, up, 1], 4 * (r * W + c)); }
  t.set([val, val, val, 1], 4 * (row * W + col));
  return { name: 'sun-texel', width: W, height: H, texels: t, strength: 1, tint: [1, 1, 1], rotationZ: 0, visibleToCamera: true };
}

/** A Poly Haven 1k .hdr from validation/assets/downloaded/hdri (fetch in Chrome, fs in Node); undefined if absent. */
export async function loadHdri(file: string): Promise<EnvironmentData | undefined> {
  let bytes: Uint8Array | undefined;
  if (typeof window !== 'undefined') {
    const r = await fetch(`/validation/assets/downloaded/hdri/${file}`);
    if (r.ok) bytes = new Uint8Array(await r.arrayBuffer());
  } else {
    const fs = await import('node:fs');
    try { bytes = new Uint8Array(fs.readFileSync(`validation/assets/downloaded/hdri/${file}`)); } catch { bytes = undefined; }
  }
  if (!bytes || bytes.length < 16) return undefined;
  const img = decodeHdr(bytes);
  return { name: file, width: img.width, height: img.height, texels: img.texels, strength: 1, tint: [1, 1, 1], rotationZ: 0, visibleToCamera: true };
}
