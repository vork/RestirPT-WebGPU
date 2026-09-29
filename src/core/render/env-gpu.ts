// GPU resources for the environment map (plan §1.4b, §1.8 "texEnv + sEnv bound in G0"; math.md#env-mapping).
// - texEnv: rgba32float, W×H, 1 mip, rows uploaded in EnvironmentData order = BOTTOM-UP (row 0 = nadir). WebGPU's
//   texture coordinate y = 0 is the first uploaded row, so shaders sample at Cycles' (u, v) with no flip (env.wgsl).
// - sEnv: repeat/repeat, linear/linear (Cycles EXTENSION_REPEAT incl. the pole blend). Needs float32-filterable.
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

export interface EnvGpuResources {
  texture: GPUTexture;
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

export interface EnvMemoryReport { width: number; height: number; format: 'rgba32float'; textureBytes: number; uniformBytes: number; totalBytes: number; text: string }

export class EnvGpuError extends Error {}

/** Create texture, sampler and uniform; upload the texels (row bands inside an out-of-memory error scope). */
export async function createEnvResources(device: GPUDevice, env?: EnvironmentData, label = 'env'): Promise<EnvGpuResources> {
  if (!device.features.has('float32-filterable')) {
    throw new EnvGpuError('float32-filterable is required for linear filtering of the rgba32float env map');
  }
  const width = env?.width ?? 1, height = env?.height ?? 1;
  const max = device.limits.maxTextureDimension2D;
  if (width > max || height > max) throw new EnvGpuError(`env ${width}x${height} exceeds maxTextureDimension2D ${max}`);
  if (env && env.texels.length !== width * height * 4) throw new EnvGpuError(`env texels length ${env.texels.length} != ${width}x${height}x4`);

  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  const texture = device.createTexture({
    label: `${label}.tex`, size: { width, height }, format: 'rgba32float', mipLevelCount: 1,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
  });
  const texels = env?.texels ?? new Float32Array(4);
  const rowBytes = width * 16;
  const rowsPerBand = Math.max(1, Math.floor(UPLOAD_BAND_BYTES / rowBytes));
  for (let y0 = 0; y0 < height; y0 += rowsPerBand) {
    const rows = Math.min(rowsPerBand, height - y0);
    const band = texels.subarray(y0 * width * 4, (y0 + rows) * width * 4);
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
    texture, view: texture.createView({ label: `${label}.view` }), sampler, uniform, width, height, present: !!env, source: env,
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
  const textureBytes = res.width * res.height * 16;
  const totalBytes = textureBytes + ENV_UNIFORM_SIZE;
  const mib = (b: number) => (b / (1024 * 1024)).toFixed(2);
  return {
    width: res.width, height: res.height, format: 'rgba32float', textureBytes, uniformBytes: ENV_UNIFORM_SIZE, totalBytes,
    text: res.present ? `env ${res.width}x${res.height} rgba32float ${mib(textureBytes)} MiB` : 'env: none',
  };
}

export function destroyEnvResources(res: EnvGpuResources): void {
  res.texture.destroy();
  res.uniform.destroy();
}
