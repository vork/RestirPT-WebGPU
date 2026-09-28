// SceneData -> GPU (plan §1.3, §1.6, §1.8). WGSL mirror: src/core/shaders/scene/scene-data.wgsl.
// - Positions are recentred in f64 (p_int = p_world − O, O = bounds centre; plan §1.2) BEFORE the BVH build, so
//   the BVH, the vertex buffer and the camera all live in the same f32 internal frame. `origin` is kept for exports.
// - The SAH BVH is built in its Worker (browser) or inline (Node); both MT and Woop layouts are kept on the CPU so
//   the watertight toggle only re-uploads the triangle buffer.
// - Storage buffers of the scene group (5 of the 10 per stage): bvh nodes, bvh tris, vertices (position, normal,
//   uv0 and COLOR_0 merged, 48 B), triangles (indices + material + flags merged, 16 B), materials (336 B).
//   Tangents are uploaded as their own buffer for later passes (normal maps, M7) but are not bound in M1.
// - Textures: validation path by default (plan §1.6), interactive on request.
import { buildBvh } from '../bvh/sah-builder.ts';
import { uploadBvh, type BvhData, type BvhGpuBuffers } from '../bvh/layout.ts';
import type { Defines } from '../gpu/wgsl-composer.ts';
import type { MaterialData, SceneData, SceneGeometry } from '../scene/types.ts';
import { createGpuTextures, packTexSlot, type GpuTextures, type TexturePathMode } from './textures-gpu.ts';

export const SCENE_GROUP_DEFAULT = 1;
export const SCENE_BINDING = { bvhNodes: 0, bvhTris: 1, vertices: 2, tris: 3, materials: 4, textureBase: 8 } as const;
export const VERTEX_BYTES = 48;
export const TRI_BYTES = 16;
/** Byte offsets of MaterialGpu (scene-data.wgsl); tests/render/scene-gpu.test.ts checks them against the WGSL. */
export const MATERIAL_LAYOUT = {
  baseColor: 0, emission: 16, alphaCutoff: 28, metallic: 32, roughness: 36, ior: 40, flags: 44,
  specularColor: 48, specularFactor: 60, v1Diffuse: 64, transmission: 76, v1Glossy: 80, v1Roughness: 92,
  v1Mix: 96, normalScale: 100,
  texBaseColor: 112, texMetalRough: 144, texNormal: 176, texEmissive: 208, texTransmission: 240, texSpecular: 272,
  texSpecularColor: 304,
  size: 336,
} as const;
export const MAT_ALPHA_MASK = 1;
export const MAT_DOUBLE_SIDED = 2;
export const MAT_V1 = 4;
export const TRI_MAT_MASK = 0xffffff;
export const TRI_FLAGS_SHIFT = 24;

export type BvhBuilder = (positions: Float32Array, indices: Uint32Array) => Promise<BvhData>;

export interface SceneGpuOptions {
  textureMode: TexturePathMode;
  /** Woop watertight intersection (T12, glass, contact scenes) instead of Möller–Trumbore. */
  watertight: boolean;
  /** Default: the Worker build in browsers, inline elsewhere. */
  buildBvh?: BvhBuilder;
  textureBudgetBytes?: number;
  features?: Set<string>;
  wgslLanguageFeatures?: Set<string>;
  label?: string;
}

export interface SceneGpuStats {
  triangles: number;
  vertices: number;
  materials: number;
  bvhMs: number;
  uploadMs: number;
  textureMs: number;
  geometryBytes: number;
  textureBytes: number;
}

/** Recentre positions in f64: out = f32(p − origin). */
export function recentrePositions(positions: Float32Array, origin: readonly number[]): Float32Array {
  const out = new Float32Array(positions.length);
  for (let i = 0; i < positions.length; i += 3) {
    out[i] = positions[i] - origin[0];
    out[i + 1] = positions[i + 1] - origin[1];
    out[i + 2] = positions[i + 2] - origin[2];
  }
  return out;
}

/** SceneVertex records (48 B): p (recentred), uv.x, n, uv.y, COLOR_0 (1 when absent). */
export function packVertices(g: SceneGeometry, recentred: Float32Array): Float32Array {
  const n = recentred.length / 3;
  const out = new Float32Array(Math.max(1, n) * (VERTEX_BYTES / 4));
  for (let i = 0; i < n; i++) {
    const o = i * 12;
    out[o] = recentred[3 * i]; out[o + 1] = recentred[3 * i + 1]; out[o + 2] = recentred[3 * i + 2];
    out[o + 3] = g.uv0[2 * i] ?? 0;
    out[o + 4] = g.normals[3 * i]; out[o + 5] = g.normals[3 * i + 1]; out[o + 6] = g.normals[3 * i + 2];
    out[o + 7] = g.uv0[2 * i + 1] ?? 0;
    if (g.color0) { for (let c = 0; c < 4; c++) out[o + 8 + c] = g.color0[4 * i + c]; }
    else { out[o + 8] = 1; out[o + 9] = 1; out[o + 10] = 1; out[o + 11] = 1; }
  }
  return out;
}

/** Triangle records (vec4u): i0, i1, i2, material | triFlags << 24. */
export function packTris(g: SceneGeometry): Uint32Array {
  const n = g.indices.length / 3;
  const out = new Uint32Array(Math.max(1, n) * 4);
  for (let t = 0; t < n; t++) {
    const m = g.triMaterial[t];
    if (m > TRI_MAT_MASK) throw new Error(`material index ${m} exceeds 2^24`);
    out[4 * t] = g.indices[3 * t]; out[4 * t + 1] = g.indices[3 * t + 1]; out[4 * t + 2] = g.indices[3 * t + 2];
    out[4 * t + 3] = (m | ((g.triFlags[t] & 0xff) << TRI_FLAGS_SHIFT)) >>> 0;
  }
  return out;
}

/** MaterialGpu records. `textures` null → every slot invalid (tex_sample returns 1). */
export function packMaterials(materials: MaterialData[], textures: Pick<GpuTextures, 'slot'> | null): ArrayBuffer {
  const L = MATERIAL_LAYOUT;
  const buf = new ArrayBuffer(Math.max(1, materials.length) * L.size);
  const dv = new DataView(buf);
  const f = (o: number, v: number) => dv.setFloat32(o, v, true);
  const v3 = (o: number, v: readonly number[]) => { f(o, v[0]); f(o + 4, v[1]); f(o + 8, v[2]); };
  materials.forEach((m, i) => {
    const b = i * L.size;
    v3(b + L.baseColor, m.baseColorFactor); f(b + L.baseColor + 12, m.baseColorFactor[3]);
    v3(b + L.emission, m.emissiveFactor.map((c) => c * m.emissiveStrength));
    f(b + L.alphaCutoff, m.alphaMode === 'OPAQUE' ? 0 : m.alphaCutoff);
    f(b + L.metallic, m.metallicFactor);
    f(b + L.roughness, m.roughnessFactor);
    f(b + L.ior, m.ior);
    let flags = 0;
    if (m.alphaMode !== 'OPAQUE') flags |= MAT_ALPHA_MASK;
    if (m.doubleSided) flags |= MAT_DOUBLE_SIDED;
    if (m.model === 'v1') flags |= MAT_V1;
    dv.setUint32(b + L.flags, flags, true);
    v3(b + L.specularColor, m.specularColorFactor);
    f(b + L.specularFactor, m.specularFactor);
    v3(b + L.v1Diffuse, m.v1?.diffuse ?? [0, 0, 0]);
    f(b + L.transmission, m.transmissionFactor);
    v3(b + L.v1Glossy, m.v1?.glossy ?? [0, 0, 0]);
    f(b + L.v1Roughness, m.v1?.roughness ?? 0);
    f(b + L.v1Mix, m.v1?.mix ?? 0);
    f(b + L.normalScale, m.normalTexture?.scale ?? 1);
    const slot = (ref: MaterialData['baseColorTexture'], srgb: boolean) => (textures ? textures.slot(ref, srgb) : null);
    packTexSlot(slot(m.baseColorTexture, true), dv, b + L.texBaseColor);
    packTexSlot(slot(m.metallicRoughnessTexture, false), dv, b + L.texMetalRough);
    packTexSlot(slot(m.normalTexture, false), dv, b + L.texNormal);
    packTexSlot(slot(m.emissiveTexture, true), dv, b + L.texEmissive);
    packTexSlot(slot(m.transmissionTexture, false), dv, b + L.texTransmission);
    packTexSlot(slot(m.specularTexture, false), dv, b + L.texSpecular);
    packTexSlot(slot(m.specularColorTexture, true), dv, b + L.texSpecularColor);
  });
  return buf;
}

async function defaultBuilder(positions: Float32Array, indices: Uint32Array): Promise<BvhData> {
  if (typeof Worker !== 'undefined' && typeof window !== 'undefined') {
    const { buildBvhInWorker } = await import('../bvh/build-in-worker.ts');
    return buildBvhInWorker(positions, indices, { mt: true, woop: true });
  }
  return buildBvh(positions, indices, { mt: true, woop: true });
}

function storageBuffer(device: GPUDevice, data: ArrayBufferView | ArrayBuffer, label: string, minBytes: number): GPUBuffer {
  const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const size = Math.max(minBytes, Math.ceil(bytes.byteLength / 4) * 4);
  const buf = device.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
  new Uint8Array(buf.getMappedRange()).set(bytes);
  buf.unmap();
  return buf;
}

export class SceneGpu {
  private constructor(
    readonly device: GPUDevice,
    readonly scene: SceneData,
    readonly origin: [number, number, number],
    readonly bvh: BvhData,
    public bvhBuffers: BvhGpuBuffers,
    readonly vertices: GPUBuffer,
    readonly tris: GPUBuffer,
    readonly materials: GPUBuffer,
    readonly tangents: GPUBuffer,
    readonly textures: GpuTextures,
    readonly stats: SceneGpuStats,
    readonly warnings: string[],
  ) {}

  /** Upload a scene recentred by `origin` (normally the bounds centre, computeRenderOrigin). */
  static async create(device: GPUDevice, scene: SceneData, origin: [number, number, number], opts: SceneGpuOptions): Promise<SceneGpu> {
    const label = opts.label ?? 'scene';
    const g = scene.geometry;
    const pos = recentrePositions(g.positions, origin);
    let t0 = performance.now();
    const bvhP = (opts.buildBvh ?? defaultBuilder)(pos, g.indices);
    bvhP.catch(() => undefined); // observed below; avoids an unhandled rejection if the texture upload throws first
    // Textures upload while the BVH builds (Worker).
    const tt = performance.now();
    const textures = await createGpuTextures(device, scene, {
      mode: opts.textureMode, budgetBytes: opts.textureBudgetBytes, features: opts.features, wgslLanguageFeatures: opts.wgslLanguageFeatures,
    });
    const textureMs = performance.now() - tt;
    let bvh: BvhData;
    try { bvh = await bvhP; } catch (e) { textures.destroy(); throw e; }
    const bvhMs = performance.now() - t0;
    t0 = performance.now();
    // Everything created here is destroyed again on any failure (no GPU memory leaks across scene reloads).
    const created: GPUBuffer[] = [];
    const keep = (b: GPUBuffer): GPUBuffer => { created.push(b); return b; };
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    let up: { bvhBuffers: BvhGpuBuffers; vertices: GPUBuffer; tris: GPUBuffer; materials: GPUBuffer; tangents: GPUBuffer } | undefined;
    let thrown: unknown;
    try {
      const bvhBuffers = uploadBvh(device, bvh, { watertight: opts.watertight, label });
      keep(bvhBuffers.nodes); keep(bvhBuffers.tris);
      up = {
        bvhBuffers,
        vertices: keep(storageBuffer(device, packVertices(g, pos), `${label}.vertices`, VERTEX_BYTES)),
        tris: keep(storageBuffer(device, packTris(g), `${label}.tris`, TRI_BYTES)),
        materials: keep(storageBuffer(device, packMaterials(scene.materials, textures), `${label}.materials`, MATERIAL_LAYOUT.size)),
        tangents: keep(storageBuffer(device, g.tangents, `${label}.tangents`, 16)),
      };
    } catch (e) { thrown = e ?? new Error('scene upload failed'); }
    const valErr = await device.popErrorScope();
    const oom = await device.popErrorScope();
    if (!up || valErr || oom) {
      for (const b of created) b.destroy();
      textures.destroy();
      if (thrown !== undefined) throw thrown;
      throw new Error(`scene upload failed: ${(oom ?? valErr)!.message}`);
    }
    const { bvhBuffers, vertices, tris, materials, tangents } = up;
    const uploadMs = performance.now() - t0;
    const geometryBytes = bvhBuffers.nodes.size + bvhBuffers.tris.size + vertices.size + tris.size + materials.size + tangents.size;
    const warnings = [...textures.warnings];
    if (bvh.stats.skippedNonFinite) warnings.push(`${bvh.stats.skippedNonFinite} non-finite triangle(s) left out of the BVH`);
    return new SceneGpu(device, scene, origin, bvh, bvhBuffers, vertices, tris, materials, tangents, textures, {
      triangles: g.indices.length / 3, vertices: g.positions.length / 3, materials: scene.materials.length,
      bvhMs, uploadMs, textureMs, geometryBytes, textureBytes: textures.bytes,
    }, warnings);
  }

  get watertight(): boolean { return this.bvhBuffers.watertight; }

  /** Switch MT <-> Woop (re-uploads only the triangle layout; the caller recompiles with the new defines). */
  setWatertight(on: boolean): void {
    if (on === this.watertight) return;
    const old = this.bvhBuffers;
    const next = uploadBvh(this.device, this.bvh, { watertight: on, label: 'scene' });
    old.tris.destroy();
    next.nodes.destroy();
    this.bvhBuffers = { ...next, nodes: old.nodes };
  }

  /** Composer defines for scene-data.wgsl + traverse.wgsl + textures.wgsl in group `group`. */
  defines(group = SCENE_GROUP_DEFAULT): Defines {
    return {
      SCENE_GROUP: group,
      BVH_DECLARE_BINDINGS: true, BVH_GROUP: group, BVH_BINDING_NODES: SCENE_BINDING.bvhNodes, BVH_BINDING_TRIS: SCENE_BINDING.bvhTris,
      WATERTIGHT: this.watertight,
      CUSTOM_ALPHA: true,
      ...this.textures.defines(group, SCENE_BINDING.textureBase),
    };
  }

  layoutEntries(visibility: GPUShaderStageFlags = GPUShaderStage.COMPUTE): GPUBindGroupLayoutEntry[] {
    const ro = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer: { type: 'read-only-storage' } });
    return [
      ro(SCENE_BINDING.bvhNodes), ro(SCENE_BINDING.bvhTris), ro(SCENE_BINDING.vertices), ro(SCENE_BINDING.tris), ro(SCENE_BINDING.materials),
      ...this.textures.layoutEntries(SCENE_BINDING.textureBase, visibility),
    ];
  }

  bindGroupEntries(): GPUBindGroupEntry[] {
    return [
      { binding: SCENE_BINDING.bvhNodes, resource: { buffer: this.bvhBuffers.nodes } },
      { binding: SCENE_BINDING.bvhTris, resource: { buffer: this.bvhBuffers.tris } },
      { binding: SCENE_BINDING.vertices, resource: { buffer: this.vertices } },
      { binding: SCENE_BINDING.tris, resource: { buffer: this.tris } },
      { binding: SCENE_BINDING.materials, resource: { buffer: this.materials } },
      ...this.textures.bindGroupEntries(SCENE_BINDING.textureBase),
    ];
  }

  destroy(): void {
    for (const b of [this.bvhBuffers.nodes, this.bvhBuffers.tris, this.vertices, this.tris, this.materials, this.tangents]) b.destroy();
    this.textures.destroy();
  }
}

/** A scene without triangles (env-only rendering). */
export function emptyScene(name = 'empty'): SceneData {
  return {
    name,
    geometry: {
      positions: new Float32Array(0), normals: new Float32Array(0), tangents: new Float32Array(0), uv0: new Float32Array(0),
      indices: new Uint32Array(0), triMaterial: new Uint32Array(0), triFlags: new Uint32Array(0),
    },
    materials: [], textures: [], lights: [], cameras: [],
    bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
    warnings: [],
  };
}
