// Environment map loading: bytes → EnvironmentData (plan §1.4b; math.md#env-mapping).
// Format is sniffed from the magic bytes (.hdr "#?", .exr 0x762f3101). Decoding runs in a Worker when one is
// available (browser) and inline otherwise (Node tests, or when the Worker fails); texels come back via transfer.
// Size policy (env §2.2): validation keeps the native resolution up to W ≤ 8192; interactive 2×2 box-downsamples
// (linear, energy-preserving mean) until W ≤ 4096. A downsampled map is a different environment (warning).
import type { EnvironmentData } from '../types.ts';
import { decodeHdr, isHdr, type DecodedEnvImage } from './hdr.ts';
import { decodeExr, isExr, type EnvLoadMode, type ExrInfo } from './exr.ts';

export type { EnvLoadMode } from './exr.ts';

export const ENV_MAX_WIDTH_VALIDATION = 8192;
export const ENV_MAX_WIDTH_INTERACTIVE = 4096;
const TEXTURE_LIMIT = 16384;

export interface LoadEnvOptions {
  mode?: EnvLoadMode;             // default 'interactive'
  name?: string;
  strength?: number;
  tint?: [number, number, number];
  rotationZ?: number;
  visibleToCamera?: boolean;
}

export interface LoadedEnv {
  env: EnvironmentData;
  format: 'hdr' | 'exr';
  /** Source resolution before any interactive downsampling. */
  sourceWidth: number;
  sourceHeight: number;
  /** Number of 2×2 downsampling steps applied (0 in validation). */
  downsampleSteps: number;
  exr?: ExrInfo;
  warnings: string[];
  /** Where decoding ran (diagnostics). */
  decodedIn: 'worker' | 'inline';
}

export class EnvLoadError extends Error {}

/** Decode synchronously (used by the Worker, by tests, and as the no-Worker fallback). */
export function decodeEnvironment(input: ArrayBuffer | Uint8Array, opts: LoadEnvOptions = {}): LoadedEnv {
  const mode = opts.mode ?? 'interactive';
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  let img: DecodedEnvImage;
  let format: 'hdr' | 'exr';
  let exr: ExrInfo | undefined;
  if (isHdr(bytes)) { img = decodeHdr(bytes); format = 'hdr'; }
  else if (isExr(bytes)) { const r = decodeExr(bytes, mode); img = r; exr = r.info; format = 'exr'; }
  else throw new EnvLoadError('unknown environment format (expected Radiance .hdr or OpenEXR)');
  const warnings = [...img.warnings];
  const sourceWidth = img.width, sourceHeight = img.height;
  if (img.width !== 2 * img.height) warnings.push(`equirect aspect is ${img.width}x${img.height}, expected 2:1`);
  if (img.width > TEXTURE_LIMIT || img.height > TEXTURE_LIMIT) {
    if (mode === 'validation') throw new EnvLoadError(`env ${img.width}x${img.height} exceeds the 16384 texture limit`);
  }
  let steps = 0;
  if (mode === 'validation') {
    if (img.width > ENV_MAX_WIDTH_VALIDATION) throw new EnvLoadError(`validation env width ${img.width} > ${ENV_MAX_WIDTH_VALIDATION}`);
  } else {
    while (img.width > ENV_MAX_WIDTH_INTERACTIVE) { img = downsample2x2(img); steps++; }
    if (steps) warnings.push(`downsampled ${sourceWidth}x${sourceHeight} -> ${img.width}x${img.height} (interactive; not the source environment)`);
  }
  const env: EnvironmentData = {
    name: opts.name ?? `env-${format}`,
    width: img.width,
    height: img.height,
    texels: img.texels,
    strength: opts.strength ?? 1,
    tint: opts.tint ?? [1, 1, 1],
    rotationZ: opts.rotationZ ?? 0,
    visibleToCamera: opts.visibleToCamera ?? true,
  };
  return { env, format, sourceWidth, sourceHeight, downsampleSteps: steps, exr, warnings, decodedIn: 'inline' };
}

/** 2×2 box filter in f64 (rows stay bottom-up; odd trailing row/column dropped). */
export function downsample2x2(img: DecodedEnvImage): DecodedEnvImage {
  const w = Math.max(1, img.width >> 1), h = Math.max(1, img.height >> 1);
  const src = img.texels, W = img.width;
  const out = new Float32Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const a = ((2 * y) * W + 2 * x) * 4, b = a + W * 4;
      const o = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) out[o + c] = (src[a + c] + src[a + 4 + c] + src[b + c] + src[b + 4 + c]) * 0.25;
      out[o + 3] = 1;
    }
  }
  return { width: w, height: h, texels: out, warnings: img.warnings };
}

type WorkerReply = { ok: true; result: LoadedEnv } | { ok: false; error: string };

/** Load from bytes, a Blob/File (drag-and-drop) or a URL. Uses a module Worker when available. */
export async function loadEnvironment(source: ArrayBuffer | Uint8Array | Blob | string | URL, opts: LoadEnvOptions = {}): Promise<LoadedEnv> {
  let bytes: ArrayBuffer;
  let name = opts.name;
  if (typeof source === 'string' || source instanceof URL) {
    const url = String(source);
    const res = await fetch(url);
    if (!res.ok) throw new EnvLoadError(`${url}: HTTP ${res.status}`);
    bytes = await res.arrayBuffer();
    name ??= url.split(/[?#]/)[0].split('/').pop();
  } else if (typeof Blob !== 'undefined' && source instanceof Blob) {
    bytes = await source.arrayBuffer();
    name ??= (source as Blob & { name?: string }).name;
  } else if (source instanceof Uint8Array) {
    bytes = source.slice().buffer as ArrayBuffer;
  } else {
    bytes = source as ArrayBuffer;
  }
  const o = { ...opts, name };
  if (typeof Worker === 'undefined' || typeof window === 'undefined') return decodeEnvironment(bytes, o);
  let worker: Worker;
  try { worker = new Worker(new URL('./env-worker.ts', import.meta.url), { type: 'module', name: 'env-loader' }); }
  catch { return decodeEnvironment(bytes, o); }
  try {
    const reply = await new Promise<WorkerReply>((resolve, reject) => {
      worker.onmessage = (ev: MessageEvent<WorkerReply>) => resolve(ev.data);
      worker.onerror = (ev) => reject(new EnvLoadError(`env worker failed: ${ev.message}`));
      worker.postMessage({ bytes, opts: o }, [bytes]);
    });
    if (!reply.ok) throw new EnvLoadError(reply.error);
    return reply.result;
  } finally {
    worker.terminate();
  }
}
