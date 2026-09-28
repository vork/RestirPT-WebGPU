// Scene textures on the GPU (plan §1.6; gap-bsdf §9). WGSL side: shaders/material/textures.wgsl.
//
// VALIDATION path (every Cycles comparison) never resamples: one rgba8unorm texture_2d_array per distinct (w, h)
// (≤ 16, error otherwise), each image a layer, mipLevelCount 1, hardware bilinear/nearest with the glTF address
// modes, sampled at LOD 0, sRGB decoded after filtering in the shader.
//
// INTERACTIVE path: square arrays bucketed by size (256, 512, 1024, 2048), images resampled into their bucket on
// the GPU and given LINEAR-space mips (sRGB decode → average → re-encode; storage stays rgba8unorm so the shader
// decode is identical to validation). A memory budget (default 1 GiB) halves the largest images until the set
// fits and reports every downscale as a warning.
//
// Bindings (textures.wgsl): arrays at base+0..base+15, samplers at base+16..base+23; only used entries exist.
// Samplers are the distinct (wrapS, wrapT, filter) combinations used by the scene, ≤ 8.
import { composeWgsl, createCheckedShaderModule } from '../gpu/wgsl-composer.ts';
import type { Defines } from '../gpu/wgsl-composer.ts';
import { shaderSources } from '../shaders/index.ts';
import type { MaterialData, SceneData, TextureData, TextureRef } from '../scene/types.ts';

export const TEX_MAX_ARRAYS = 16;
export const TEX_MAX_SAMPLERS = 8;
export const TEX_SAMPLER_BINDING_OFFSET = 16;
/** Size of one packed TexSlot (WGSL struct TexSlot). */
export const TEX_SLOT_BYTES = 32;
export const INTERACTIVE_BUCKETS = [256, 512, 1024, 2048] as const;
export const DEFAULT_TEXTURE_BUDGET_BYTES = 1 << 30;
/** Material texture fields that hold sRGB-encoded colour (decoded after filtering); all others are linear data. */
export const SRGB_MATERIAL_TEXTURES = ['baseColorTexture', 'emissiveTexture', 'specularColorTexture'] as const;
const LINEAR_MATERIAL_TEXTURES = ['metallicRoughnessTexture', 'normalTexture', 'specularTexture', 'transmissionTexture'] as const;

const MAX_WRITE_BYTES = 64 << 20; // plan §1.1: writeBuffer/writeTexture chunks ≤ 64 MB
const TEX_VALID = 0x80000000, TEX_SRGB = 0x40000, TEX_UVSET1 = 0x80000;

export type TexturePathMode = 'validation' | 'interactive';

/** Resolved material texture slot (packed into material records with packTexSlot). */
export interface TexSlot {
  arrayIndex: number;
  layer: number;
  sampler: number;
  uvSet: number;
  srgb: boolean;
  /** Row-major 2x3 affine [a b c; d e f] (identity when the ref has no KHR_texture_transform). */
  xform: [number, number, number, number, number, number];
}

export interface GpuTextureOptions {
  mode: TexturePathMode;
  /** Interactive only (default 1 GiB). */
  budgetBytes?: number;
  /** For composing the interactive resample kernel. */
  features?: Set<string>;
  wgslLanguageFeatures?: Set<string>;
}

interface Placement { arrayIndex: number; layer: number }

export class GpuTextures {
  constructor(
    readonly mode: TexturePathMode,
    readonly arrays: GPUTexture[],
    readonly views: GPUTextureView[],
    readonly samplers: GPUSampler[],
    /** 'wrapS|wrapT|filter' per sampler index. */
    readonly samplerKeys: string[],
    /** Per SceneData.textures index: array/layer, or null when the texture has no usable pixels. */
    readonly placement: (Placement | null)[],
    /** Per SceneData.textures index: sampler index. */
    readonly samplerOf: number[],
    readonly bytes: number,
    readonly warnings: string[],
  ) {}

  /** Slot for a material texture reference; null → the shader treats it as absent (tex_sample returns 1). */
  slot(ref: TextureRef | undefined, srgb: boolean): TexSlot | null {
    if (!ref) return null;
    const p = this.placement[ref.texture];
    if (!p) return null;
    return { arrayIndex: p.arrayIndex, layer: p.layer, sampler: this.samplerOf[ref.texture], uvSet: ref.texCoord === 1 ? 1 : 0, srgb, xform: ref.transform ?? [1, 0, 0, 0, 1, 0] };
  }

  /** Composer defines for textures.wgsl. */
  defines(group: number, bindingBase: number): Defines {
    return { TEX_GROUP: group, TEX_BINDING_BASE: bindingBase, TEX_ARRAYS: this.arrays.length, TEX_SAMPLERS: this.samplers.length };
  }

  layoutEntries(bindingBase: number, visibility: GPUShaderStageFlags = GPUShaderStage.COMPUTE): GPUBindGroupLayoutEntry[] {
    return [
      ...this.views.map((_, i) => ({ binding: bindingBase + i, visibility, texture: { sampleType: 'float', viewDimension: '2d-array' } }) as GPUBindGroupLayoutEntry),
      ...this.samplers.map((_, i) => ({ binding: bindingBase + TEX_SAMPLER_BINDING_OFFSET + i, visibility, sampler: { type: 'filtering' } }) as GPUBindGroupLayoutEntry),
    ];
  }

  bindGroupEntries(bindingBase: number): GPUBindGroupEntry[] {
    return [
      ...this.views.map((v, i) => ({ binding: bindingBase + i, resource: v })),
      ...this.samplers.map((s, i) => ({ binding: bindingBase + TEX_SAMPLER_BINDING_OFFSET + i, resource: s })),
    ];
  }

  destroy(): void { for (const t of this.arrays) t.destroy(); }
}

/** Write one TexSlot (32 B) at `byteOffset`; null writes an invalid slot. */
export function packTexSlot(slot: TexSlot | null, view: DataView, byteOffset: number): void {
  const x = slot?.xform ?? [1, 0, 0, 0, 1, 0];
  let info = 0;
  if (slot) {
    info = (slot.arrayIndex & 0xf) | ((slot.layer & 0x7ff) << 4) | ((slot.sampler & 0x7) << 15) | TEX_VALID;
    if (slot.srgb) info |= TEX_SRGB;
    if (slot.uvSet === 1) info |= TEX_UVSET1;
  }
  for (let i = 0; i < 3; i++) view.setFloat32(byteOffset + 4 * i, x[i], true);
  view.setUint32(byteOffset + 12, info >>> 0, true);
  for (let i = 0; i < 3; i++) view.setFloat32(byteOffset + 16 + 4 * i, x[3 + i], true);
  view.setUint32(byteOffset + 28, 0, true);
}

/** Texture indices used as sRGB colour / linear data by the materials. */
export function textureColorSpaces(materials: MaterialData[]): { srgb: Set<number>; linear: Set<number> } {
  const srgb = new Set<number>(), linear = new Set<number>();
  for (const m of materials) {
    for (const k of SRGB_MATERIAL_TEXTURES) { const r = m[k]; if (r) srgb.add(r.texture); }
    for (const k of LINEAR_MATERIAL_TEXTURES) { const r = m[k]; if (r) linear.add(r.texture); }
  }
  return { srgb, linear };
}

export async function createGpuTextures(device: GPUDevice, scene: Pick<SceneData, 'textures' | 'materials'>, opts: GpuTextureOptions): Promise<GpuTextures> {
  const warnings: string[] = [];
  const textures = scene.textures;

  // Distinct images (TextureData entries that share a pixel buffer share a layer).
  const images: TextureData[] = [];
  const imageOf: number[] = [];
  const byPixels = new Map<Uint8Array, number>();
  let skipped = 0;
  for (const t of textures) {
    if (t.pixels.length === 0 || t.pixels.length !== t.width * t.height * 4 || t.width < 1 || t.height < 1) { imageOf.push(-1); skipped++; continue; }
    let i = byPixels.get(t.pixels);
    if (i === undefined) { i = images.length; images.push(t); byPixels.set(t.pixels, i); }
    imageOf.push(i);
  }
  if (skipped) warnings.push(`${skipped} texture(s) without usable pixels are treated as absent`);

  // Samplers: distinct (wrapS, wrapT, filter) of usable textures, in texture order.
  const samplerKeys: string[] = [];
  const samplerOf = textures.map((t, i) => {
    if (imageOf[i] < 0) return 0;
    const key = `${t.wrapS}|${t.wrapT}|${t.filter}`;
    let s = samplerKeys.indexOf(key);
    if (s < 0) { s = samplerKeys.length; samplerKeys.push(key); }
    return s;
  });
  if (samplerKeys.length > TEX_MAX_SAMPLERS) throw new Error(`scene needs ${samplerKeys.length} texture samplers; the binding budget is ${TEX_MAX_SAMPLERS}`);
  const interactive = opts.mode === 'interactive';
  const samplers = samplerKeys.map((k) => {
    const [wrapS, wrapT, filter] = k.split('|') as [GPUAddressMode, GPUAddressMode, GPUFilterMode];
    return device.createSampler({ label: `tex-${k}`, addressModeU: wrapS, addressModeV: wrapT, magFilter: filter, minFilter: filter, mipmapFilter: interactive ? 'linear' : 'nearest' });
  });

  const maxLayers = device.limits.maxTextureArrayLayers;
  const maxDim = device.limits.maxTextureDimension2D;
  const result = interactive
    ? await buildInteractive(device, images, scene.materials, textures, imageOf, opts, warnings)
    : buildValidation(device, images, maxLayers, maxDim);
  const placement = imageOf.map((i) => (i < 0 ? null : result.placement[i]));
  if (result.arrays.length > TEX_MAX_ARRAYS) throw new Error(`scene needs ${result.arrays.length} texture arrays; the binding budget is ${TEX_MAX_ARRAYS}`);
  return new GpuTextures(opts.mode, result.arrays, result.arrays.map((a) => a.createView({ dimension: '2d-array' })), samplers,
    samplerKeys, placement, samplerOf, result.bytes, warnings);
}

// ---------------------------------------------------------------------------------------------------------------

function uploadLayer(device: GPUDevice, tex: GPUTexture, layer: number, img: { width: number; height: number; pixels: Uint8Array }): void {
  const rowBytes = img.width * 4;
  const rowsPerChunk = Math.max(1, Math.floor(MAX_WRITE_BYTES / rowBytes));
  for (let y = 0; y < img.height; y += rowsPerChunk) {
    const rows = Math.min(rowsPerChunk, img.height - y);
    const data = img.pixels.subarray(y * rowBytes, (y + rows) * rowBytes) as Uint8Array<ArrayBuffer>;
    device.queue.writeTexture({ texture: tex, origin: { x: 0, y, z: layer } }, data, { bytesPerRow: rowBytes, rowsPerImage: rows }, { width: img.width, height: rows, depthOrArrayLayers: 1 });
  }
}

function buildValidation(device: GPUDevice, images: TextureData[], maxLayers: number, maxDim: number) {
  const groups = new Map<string, number[]>();
  images.forEach((img, i) => {
    const k = `${img.width}x${img.height}`;
    const g = groups.get(k);
    if (g) g.push(i); else groups.set(k, [i]);
  });
  if (groups.size > TEX_MAX_ARRAYS) {
    throw new Error(`validation textures have ${groups.size} distinct sizes (${[...groups.keys()].join(', ')}); at most ${TEX_MAX_ARRAYS} are supported (plan §1.6)`);
  }
  // Check every group before allocating anything (a throw must not leak already-created arrays).
  for (const [key, members] of groups) {
    const { width, height } = images[members[0]];
    if (width > maxDim || height > maxDim) throw new Error(`texture ${key} exceeds maxTextureDimension2D ${maxDim}`);
    if (members.length > maxLayers) throw new Error(`${members.length} textures of size ${key} exceed maxTextureArrayLayers ${maxLayers}`);
  }
  const placement: Placement[] = new Array(images.length);
  const arrays: GPUTexture[] = [];
  let bytes = 0;
  for (const [key, members] of groups) {
    const { width, height } = images[members[0]];
    const tex = device.createTexture({
      label: `tex-val-${key}`, size: { width, height, depthOrArrayLayers: members.length }, format: 'rgba8unorm', mipLevelCount: 1,
      dimension: '2d', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    const arrayIndex = arrays.length;
    arrays.push(tex);
    members.forEach((img, layer) => { placement[img] = { arrayIndex, layer }; uploadLayer(device, tex, layer, images[img]); });
    bytes += width * height * 4 * members.length;
  }
  return { arrays, placement, bytes };
}

const nextPow2 = (x: number) => 2 ** Math.ceil(Math.log2(Math.max(1, x)));
const mipBytes = (b: number) => { let s = 0; for (let d = b; d >= 1; d >>= 1) s += d * d * 4; return s; };

async function buildInteractive(device: GPUDevice, images: TextureData[], materials: MaterialData[], textures: TextureData[],
  imageOf: number[], opts: GpuTextureOptions, warnings: string[]) {
  const [minB, maxB] = [INTERACTIVE_BUCKETS[0], INTERACTIVE_BUCKETS[INTERACTIVE_BUCKETS.length - 1]];
  const budget = opts.budgetBytes ?? DEFAULT_TEXTURE_BUDGET_BYTES;

  // sRGB flag per image (mips average in linear space for colour images).
  const cs = textureColorSpaces(materials);
  const imgSrgb = images.map(() => false);
  const imgLinear = images.map(() => false);
  textures.forEach((_, ti) => {
    const i = imageOf[ti];
    if (i < 0) return;
    if (cs.srgb.has(ti)) imgSrgb[i] = true;
    if (cs.linear.has(ti)) imgLinear[i] = true;
  });
  const mixed = images.filter((_, i) => imgSrgb[i] && imgLinear[i]).length;
  if (mixed) warnings.push(`${mixed} image(s) used both as sRGB colour and linear data; their mips are built in linear-light (sRGB) space`);

  // Bucket assignment and budget.
  const bucket = images.map((img) => Math.min(maxB, Math.max(minB, nextPow2(Math.max(img.width, img.height)))));
  const oversize = images.filter((img) => Math.max(img.width, img.height) > maxB).length;
  if (oversize) warnings.push(`${oversize} texture(s) larger than ${maxB} px downscaled to ${maxB} (interactive path)`);
  let total = bucket.reduce((s, b) => s + mipBytes(b), 0);
  let budgetDownscales = 0;
  while (total > budget) {
    let pick = -1;
    for (let i = 0; i < images.length; i++) if (bucket[i] > minB && (pick < 0 || bucket[i] > bucket[pick])) pick = i;
    if (pick < 0) break;
    total -= mipBytes(bucket[pick]) - mipBytes(bucket[pick] / 2);
    bucket[pick] /= 2;
    budgetDownscales++;
  }
  if (budgetDownscales) warnings.push(`texture budget ${(budget / 2 ** 20).toFixed(0)} MiB: ${budgetDownscales} downscale step(s) applied (interactive path)`);
  if (total > budget) warnings.push(`textures need ${(total / 2 ** 20).toFixed(0)} MiB even at ${minB} px; budget ${(budget / 2 ** 20).toFixed(0)} MiB exceeded`);

  // One array per used bucket size, in ascending size.
  const sizes = [...new Set(bucket)].sort((a, b) => a - b);
  const placement: Placement[] = new Array(images.length);
  const arrays: GPUTexture[] = [];
  const members: number[][] = sizes.map(() => []);
  images.forEach((_, i) => {
    const a = sizes.indexOf(bucket[i]);
    placement[i] = { arrayIndex: a, layer: members[a].length };
    members[a].push(i);
  });
  device.pushErrorScope('out-of-memory');
  sizes.forEach((b, a) => {
    arrays.push(device.createTexture({
      label: `tex-int-${b}`, size: { width: b, height: b, depthOrArrayLayers: members[a].length }, format: 'rgba8unorm',
      mipLevelCount: Math.log2(b) + 1, dimension: '2d',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING,
    }));
  });
  const oom = await device.popErrorScope();
  if (oom) { arrays.forEach((t) => t.destroy()); throw new Error(`texture allocation failed: ${oom.message}`); }
  if (!images.length) return { arrays, placement, bytes: 0 };

  // Resample pipeline (textures.wgsl, TEX_RESAMPLE).
  const shader = composeWgsl('material/textures.wgsl', {
    sources: shaderSources, defines: { TEX_RESAMPLE: true, TEX_ARRAYS: 0, TEX_SAMPLERS: 0 },
    features: opts.features, wgslLanguageFeatures: opts.wgslLanguageFeatures,
  });
  const module = await createCheckedShaderModule(device, shader, 'tex-resample');
  const bgl = device.createBindGroupLayout({
    label: 'tex-resample', entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d-array' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 32 } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', minBindingSize: 256 } },
    ],
  });
  const pipeline = await device.createComputePipelineAsync({
    label: 'tex-resample', layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module, entryPoint: 'tex_resample' },
  });

  interface Job { src: GPUTextureView; dst: GPUTextureView; srgbBuf: GPUBuffer; srcSize: number[]; dstSize: number[]; srcLayer: number; dstLayer: number; layers: number }
  const jobs: Job[] = [];
  const temps: GPUTexture[] = [];
  const srgbBufs: GPUBuffer[] = [];
  sizes.forEach((b, a) => {
    const arr = arrays[a];
    const mask = new Uint32Array(64);
    members[a].forEach((img, layer) => { if (imgSrgb[img]) mask[layer >> 5] |= 1 << (layer & 31); });
    const srgbBuf = device.createBuffer({ label: `tex-srgb-${b}`, size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(srgbBuf, 0, mask);
    srgbBufs.push(srgbBuf);
    const level = (l: number) => arr.createView({ dimension: '2d-array', baseMipLevel: l, mipLevelCount: 1 });
    // Level 0: direct upload when the image already has the bucket size, else GPU resample from a temp texture.
    members[a].forEach((img, layer) => {
      const im = images[img];
      if (im.width === b && im.height === b) { uploadLayer(device, arr, layer, im); return; }
      const tmp = device.createTexture({ label: `tex-src-${img}`, size: { width: im.width, height: im.height, depthOrArrayLayers: 1 }, format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      uploadLayer(device, tmp, 0, im);
      temps.push(tmp);
      jobs.push({ src: tmp.createView({ dimension: '2d-array' }), dst: level(0), srgbBuf, srcSize: [im.width, im.height], dstSize: [b, b], srcLayer: 0, dstLayer: layer, layers: 1 });
    });
    for (let l = 1, d = b / 2; d >= 1; l++, d /= 2) {
      jobs.push({ src: level(l - 1), dst: level(l), srgbBuf, srcSize: [d * 2, d * 2], dstSize: [d, d], srcLayer: 0, dstLayer: 0, layers: members[a].length });
    }
  });

  const STRIDE = 256;
  const params = device.createBuffer({ label: 'tex-resample-params', size: Math.max(1, jobs.length) * STRIDE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const pdata = new Uint32Array((jobs.length * STRIDE) / 4);
  jobs.forEach((j, i) => {
    const o = (i * STRIDE) / 4;
    const taps = (s: number, d: number) => Math.min(8, Math.max(1, Math.ceil(s / d)));
    pdata.set([j.srcSize[0], j.srcSize[1], j.dstSize[0], j.dstSize[1], j.srcLayer, j.dstLayer, taps(j.srcSize[0], j.dstSize[0]), taps(j.srcSize[1], j.dstSize[1])], o);
  });
  device.queue.writeBuffer(params, 0, pdata);
  const enc = device.createCommandEncoder({ label: 'tex-resample' });
  const pass = enc.beginComputePass({ label: 'tex-resample' });
  pass.setPipeline(pipeline);
  jobs.forEach((j, i) => {
    // Each job reads the previous job's output (mip chain): separate dispatches in one pass are ordered with
    // implicit storage→sampled synchronization between dispatches (WebGPU usage scopes are per dispatch).
    pass.setBindGroup(0, device.createBindGroup({ layout: bgl, entries: [
      { binding: 0, resource: j.src }, { binding: 1, resource: j.dst },
      { binding: 2, resource: { buffer: params, size: 32 } }, { binding: 3, resource: { buffer: j.srgbBuf } },
    ] }), [i * STRIDE]);
    pass.dispatchWorkgroups(Math.ceil(j.dstSize[0] / 8), Math.ceil(j.dstSize[1] / 8), j.layers);
  });
  pass.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  temps.forEach((t) => t.destroy());
  srgbBufs.forEach((b) => b.destroy());
  params.destroy();
  return { arrays, placement, bytes: total };
}
