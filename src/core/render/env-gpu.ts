// GPU resources for the environment map (plan §1.4b, §1.8 "texEnv + sEnv bound in G0"; math.md#env-mapping).
// - texEnv: W×H, 1 mip, rows uploaded in EnvironmentData order = BOTTOM-UP (row 0 = nadir). WebGPU's
//   texture coordinate y = 0 is the first uploaded row, so shaders sample at Cycles' (u, v) with no flip (env.wgsl).
//   Format (data-formats.md §B9, P2): the smallest one in which EVERY texel round-trips bit-exactly — rgb9e5ufloat
//   (4 B; env.wgsl samples .rgb only), else rgba16float (8 B), else rgba32float (16 B). The env is never quantized:
//   the CPU texels (importance tables, env.exr export, ENV-U9 hash) are unchanged. Validation mode uses a compact
//   format only while ENV_COMPACT_IN_VALIDATION holds (ENV-F: hardware filtering of the compact texels = filtering of
//   the same texels stored as f32, validation/gpu-tests/env-format.gpu.test.ts).
// - sEnv: repeat/repeat, linear/linear (Cycles EXTENSION_REPEAT incl. the pole blend). rgba32float needs
//   float32-filterable; rgb9e5ufloat and rgba16float are filterable in core WebGPU.
// - envParams uniform (32 B): {cg, sg, strength, visibleToCamera, tint, flags}, layout = EnvParams in env.wgsl.
// Without an env map a 1×1 black placeholder is bound and flags.PRESENT = 0.
import type { EnvironmentData } from '../scene/types.ts';
import type { Defines } from '../gpu/wgsl-composer.ts';
import { buildEnvImportance, type EnvImportance, type EnvImportanceOptions } from '../scene/env/env-importance.ts';

export const ENV_UNIFORM_SIZE = 32;
export const ENV_FLAG_PRESENT = 1;
/** Upload band limit (plan §1.1: writeBuffer/writeTexture in chunks ≤ 64 MB). */
const UPLOAD_BAND_BYTES = 64 * 1024 * 1024;

export interface EnvParamsCpu {
  rotationZ: number;           // γ (radians), Blender Mapping rotation Z
  strength: number;
  tint: [number, number, number];
  visibleToCamera: boolean;
}

export type EnvTexFormat = 'rgb9e5ufloat' | 'rgba16float' | 'rgba32float';
export const ENV_FORMAT_BYTES: Record<EnvTexFormat, number> = { rgb9e5ufloat: 4, rgba16float: 8, rgba32float: 16 };
/** ENV-F result (env-format.gpu.test.ts, Chrome/Metal): compact formats are allowed in validation mode. */
export const ENV_COMPACT_IN_VALIDATION = false;

export interface EnvGpuOptions {
  /** 'validation' (default: harness, tests) or 'interactive' (the app's interactive texture mode). */
  mode?: 'validation' | 'interactive';
  /** Force a format (tests); 'auto' = the smallest exact one allowed in `mode`. A forced compact format throws if a
   *  texel does not round-trip. */
  format?: EnvTexFormat | 'auto';
}

export interface EnvGpuResources {
  texture: GPUTexture;
  format: EnvTexFormat;
  view: GPUTextureView;
  sampler: GPUSampler;
  uniform: GPUBuffer;
  width: number;
  height: number;
  present: boolean;
  params: EnvParamsCpu;
  /** The CPU texels this texture was made from (M3c: importance tables are built from them). */
  source?: EnvironmentData;
  /** Importance tables for `importanceKey` (built by envImportanceFor, or attached by the app's Worker build). */
  importance?: EnvImportance;
  importanceKey?: string;
}

export interface EnvMemoryReport { width: number; height: number; format: EnvTexFormat; textureBytes: number; uniformBytes: number; totalBytes: number; text: string }

export class EnvGpuError extends Error {}

// ---- exact compact encodings ----------------------------------------------------------------------------------

const F32 = new Float32Array(1), U32 = new Uint32Array(F32.buffer);

/** f16 bits of x if x is exactly representable as a finite half (incl. subnormals, ±0), else null. */
export function f16BitsExact(x: number): number | null {
  F32[0] = x;
  const u = U32[0];
  const sign = (u >>> 16) & 0x8000;
  const e = (u >>> 23) & 0xff, m = u & 0x7fffff;
  if (e === 0xff) return null;                                     // Inf / NaN
  if (e === 0) return m === 0 ? sign : null;                       // ±0; f32 subnormals are far below f16's range
  const ue = e - 127;                                              // unbiased
  if (ue > 15) return null;
  if (ue >= -14) {                                                 // f16 normal: 10 mantissa bits
    if (m & 0x1fff) return null;
    return sign | ((ue + 15) << 10) | (m >>> 13);
  }
  if (ue < -24) return null;                                       // below the smallest f16 subnormal (2^-24)
  const full = m | 0x800000;                                       // 24-bit significand, value = full·2^(ue−23)
  const sh = -ue - 1;                                              // f16 subnormal q·2^-24 ⇒ q = full·2^(ue+1) = full >> sh
  if (full & ((1 << sh) - 1)) return null;
  return sign | (full >>> sh);
}

/** rgb9e5 word of (r, g, b) if exactly representable (shared exponent, 9-bit mantissas), else null. */
export function rgb9e5Exact(r: number, g: number, b: number): number | null {
  if (!(r >= 0 && g >= 0 && b >= 0) || !Number.isFinite(r + g + b)) return null;
  const mx = Math.max(r, g, b);
  if (mx === 0) return 0;
  // smallest exponent E (bias 15, value = m·2^(E−24)) with m_max ≤ 511; only it can make every mantissa integral
  let E = Math.max(0, Math.ceil(Math.log2(mx / 511)) + 24);
  while (E > 0 && mx / 2 ** (E - 1 - 24) <= 511) E--;
  while (mx / 2 ** (E - 24) > 511) E++;
  if (E > 31) return null;
  const s = 2 ** (E - 24);
  const mr = r / s, mg = g / s, mb = b / s;
  if (!Number.isInteger(mr) || !Number.isInteger(mg) || !Number.isInteger(mb)) return null;
  return (mr | (mg << 9) | (mb << 18) | (E << 27)) >>> 0;
}

/** Texels in the smallest allowed format in which every texel is exact (rgba32float always is). */
export function packEnvTexels(texels: Float32Array, allowed: readonly EnvTexFormat[]): { format: EnvTexFormat; data: Float32Array | Uint16Array | Uint32Array } {
  const n = texels.length / 4;
  if (allowed.includes('rgb9e5ufloat')) {
    const out = new Uint32Array(n);
    let ok = true;
    for (let i = 0; i < n && ok; i++) {
      const w = rgb9e5Exact(texels[4 * i], texels[4 * i + 1], texels[4 * i + 2]);
      if (w === null) ok = false; else out[i] = w;
    }
    if (ok) return { format: 'rgb9e5ufloat', data: out };
  }
  if (allowed.includes('rgba16float')) {
    const out = new Uint16Array(4 * n);
    let ok = true;
    for (let i = 0; i < 4 * n && ok; i++) {
      const h = f16BitsExact(texels[i]);
      if (h === null) ok = false; else out[i] = h;
    }
    if (ok) return { format: 'rgba16float', data: out };
  }
  if (!allowed.includes('rgba32float')) throw new EnvGpuError(`env texels are not exact in ${allowed.join(' / ')}`);
  return { format: 'rgba32float', data: texels };
}

/** Create texture, sampler and uniform; upload the texels (row bands inside an out-of-memory error scope). */
export async function createEnvResources(device: GPUDevice, env?: EnvironmentData, label = 'env', opts: EnvGpuOptions = {}): Promise<EnvGpuResources> {
  const width = env?.width ?? 1, height = env?.height ?? 1;
  const max = device.limits.maxTextureDimension2D;
  if (width > max || height > max) throw new EnvGpuError(`env ${width}x${height} exceeds maxTextureDimension2D ${max}`);
  if (env && env.texels.length !== width * height * 4) throw new EnvGpuError(`env texels length ${env.texels.length} != ${width}x${height}x4`);
  const mode = opts.mode ?? 'validation';
  const allowed: EnvTexFormat[] = opts.format && opts.format !== 'auto' ? [opts.format]
    : mode === 'validation' && !ENV_COMPACT_IN_VALIDATION ? ['rgba32float'] : ['rgb9e5ufloat', 'rgba16float', 'rgba32float'];
  const packed = packEnvTexels(env?.texels ?? new Float32Array(4), allowed);
  const format = packed.format;
  if (format === 'rgba32float' && !device.features.has('float32-filterable')) {
    throw new EnvGpuError('float32-filterable is required for linear filtering of the rgba32float env map');
  }

  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  const texture = device.createTexture({
    label: `${label}.tex`, size: { width, height }, format, mipLevelCount: 1,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
  });
  const bpt = ENV_FORMAT_BYTES[format];
  const perTexel = packed.data.length / (width * height);
  const rowBytes = width * bpt;
  const rowsPerBand = Math.max(1, Math.floor(UPLOAD_BAND_BYTES / rowBytes));
  for (let y0 = 0; y0 < height; y0 += rowsPerBand) {
    const rows = Math.min(rowsPerBand, height - y0);
    const band = packed.data.subarray(y0 * width * perTexel, (y0 + rows) * width * perTexel);
    device.queue.writeTexture({ texture, origin: { x: 0, y: y0 } }, band, { bytesPerRow: rowBytes, rowsPerImage: rows }, { width, height: rows });
  }
  const uniform = device.createBuffer({ label: `${label}.params`, size: ENV_UNIFORM_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const validationErr = await device.popErrorScope();
  const oomErr = await device.popErrorScope();
  if (oomErr || validationErr) {
    texture.destroy(); uniform.destroy();
    throw new EnvGpuError(`env upload failed (${width}x${height}): ${(oomErr ?? validationErr)!.message}`);
  }
  const sampler = device.createSampler({
    label: `${label}.sampler`, addressModeU: 'repeat', addressModeV: 'repeat',
    magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'nearest',
  });
  const res: EnvGpuResources = {
    texture, format, view: texture.createView({ label: `${label}.view` }), sampler, uniform, width, height, present: !!env, source: env,
    params: env
      ? { rotationZ: env.rotationZ, strength: env.strength, tint: [...env.tint], visibleToCamera: env.visibleToCamera }
      : { rotationZ: 0, strength: 0, tint: [0, 0, 0], visibleToCamera: false },
  };
  writeEnvParams(device, res, res.params);
  return res;
}

/** Pack EnvParams (cg, sg from γ in f64, then f32). */
export function packEnvParams(p: EnvParamsCpu, present: boolean): ArrayBuffer {
  const buf = new ArrayBuffer(ENV_UNIFORM_SIZE);
  const f = new Float32Array(buf), u = new Uint32Array(buf);
  f[0] = Math.cos(p.rotationZ);
  f[1] = Math.sin(p.rotationZ);
  f[2] = p.strength;
  u[3] = p.visibleToCamera ? 1 : 0;
  f[4] = p.tint[0]; f[5] = p.tint[1]; f[6] = p.tint[2];
  u[7] = present ? ENV_FLAG_PRESENT : 0;
  return buf;
}

/** Update rotation/strength/tint/visibility (UI or timeline tracks). No texture work. */
export function writeEnvParams(device: GPUDevice, res: EnvGpuResources, p: Partial<EnvParamsCpu>): void {
  res.params = { ...res.params, ...p, tint: [...(p.tint ?? res.params.tint)] as [number, number, number] };
  device.queue.writeBuffer(res.uniform, 0, packEnvParams(res.params, res.present));
}

/** Cache key of the importance-table options (defaults filled in). */
export function envImportanceKey(o: EnvImportanceOptions = {}): string {
  return JSON.stringify({ cap: o.cap ?? 4096, floors: o.floors ?? true, plantPdfFromTargets: o.plantPdfFromTargets ?? false });
}

/** The importance tables of `res` for options `o`, built synchronously if not cached (validation / tests). */
export function envImportanceFor(res: EnvGpuResources, o: EnvImportanceOptions = {}): EnvImportance | undefined {
  if (!res.present || !res.source) return undefined;
  const key = envImportanceKey(o);
  if (res.importance && res.importanceKey === key) return res.importance;
  const s = res.source;
  res.importance = buildEnvImportance(s.texels, s.width, s.height, o);
  res.importanceKey = key;
  return res.importance;
}

/** Defines for env.wgsl bindings: uniform at `base`, texture at base+1, sampler at base+2. */
export function envDefines(group: number, base: number): Defines {
  return { ENV_GROUP: group, ENV_BINDING: base };
}

export function envBindGroupLayoutEntries(base: number, visibility: GPUShaderStageFlags = GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT): GPUBindGroupLayoutEntry[] {
  return [
    { binding: base, visibility, buffer: { type: 'uniform', minBindingSize: ENV_UNIFORM_SIZE } },
    { binding: base + 1, visibility, texture: { sampleType: 'float', viewDimension: '2d' } },
    { binding: base + 2, visibility, sampler: { type: 'filtering' } },
  ];
}

export function envBindGroupEntries(res: EnvGpuResources, base: number): GPUBindGroupEntry[] {
  return [
    { binding: base, resource: { buffer: res.uniform } },
    { binding: base + 1, resource: res.view },
    { binding: base + 2, resource: res.sampler },
  ];
}

export function envMemoryReport(res: EnvGpuResources): EnvMemoryReport {
  const textureBytes = res.width * res.height * ENV_FORMAT_BYTES[res.format];
  const totalBytes = textureBytes + ENV_UNIFORM_SIZE;
  const mib = (b: number) => (b / (1024 * 1024)).toFixed(2);
  return {
    width: res.width, height: res.height, format: res.format, textureBytes, uniformBytes: ENV_UNIFORM_SIZE, totalBytes,
    text: res.present ? `env ${res.width}x${res.height} ${res.format} ${mib(textureBytes)} MiB` : 'env: none',
  };
}

export function destroyEnvResources(res: EnvGpuResources): void {
  res.texture.destroy();
  res.uniform.destroy();
}
