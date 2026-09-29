// Scene package v1 (docs/decisions/scene-bridge.md — THE contract with validation/blender/build_scene.py).
// exportScenePackage: SceneData (+ camera, render settings, light mode, frames) → { scene.json, geometry.bin,
// tex_<i>.png, env.exr }. readScenePackage: the inverse (round-trip tested in tests/scene/scene-package.test.ts).
// Works in the browser, in Workers and in Node (CompressionStream + WebCrypto only).
//
// - Frame/units: glTF canonical, un-recentred world coordinates (plan §1.2); Blender units for lights (plan §1.4).
// - geometry.bin: the SceneGeometry arrays verbatim (little-endian), primId order preserved. `triFlags` is written as
//   an extra buffer (additive to the contract) so the round trip is exact; readers that ignore it lose nothing.
// - env.exr: the exact float32 GPU texels (RGBA) flipped to TOP-DOWN rows, ZIP compressed. scene.json env.sha256 =
//   SHA-256 of that top-down little-endian float32 RGBA array (ENV-U9 pixel-hash definition).
// - Anything the bridge cannot represent exactly is a hard error (contract "Rules").
import { encodeExr, flipRows } from '../io/exr.ts';
import { decodePng, encodePng } from '../io/png.ts';
import { sha256Hex } from '../io/zlib.ts';
import { decodeExr } from './env/exr.ts';
import {
  TRI_ALPHA_MASK, TRI_EMISSIVE, type CameraData, type EnvironmentData, type LightData, type MaterialData, type SceneData,
  type SceneGeometry, type TextureData, type TextureRef,
} from './types.ts';

export const SCENE_PACKAGE_FORMAT = 'restir-scene-package';
export const SCENE_PACKAGE_VERSION = 1;

/** plan §1.4 Modes (A′ = pass-through only after delta lobes; its Cycles reference is Mode B: per-light MIS on). */
export type LightMode = 'A' | 'B' | 'A′';
type V3 = [number, number, number];

export interface PackageRender { width: number; height: number; maxBounces: number }
export interface PackageCamera { matrix: ArrayLike<number>; yfov: number; znear?: number }
export interface PackageFrame {
  frame: number;
  label?: string;
  camera?: { matrix: ArrayLike<number>; yfov: number };
  lights?: Record<string, { matrix?: ArrayLike<number>; power?: number }>;
  env?: { rotationZ?: number; strength?: number };
}

export interface ExportScenePackageOptions {
  camera: PackageCamera;
  render: PackageRender;
  lightMode: LightMode;
  frames?: PackageFrame[];
  /** Default: detected (every vertex normal equals its face's geometric normal). */
  flatShaded?: boolean;
  name?: string;
  source?: { uri: string; sha256?: string };
  /** World importance sampling in Cycles (default AUTOMATIC; NONE = env NEE off). */
  envSampling?: 'AUTOMATIC' | 'NONE';
}

interface BufferView { offset: number; length: number; dtype: 'f32' | 'u32'; components: number }

export interface TextureRefJson { texture: number; texCoord: number; transform?: number[] }

export interface MaterialJson {
  name: string;
  model: 'v1' | 'principled' | 'glass' | 'refraction';
  v1?: { diffuse: V3; glossy: V3; roughness: number; mix: number };
  baseColorFactor: [number, number, number, number];
  baseColorTexture?: TextureRefJson;
  metallicFactor: number;
  roughnessFactor: number;
  metallicRoughnessTexture?: TextureRefJson;
  normalTexture?: TextureRefJson & { scale: number };
  emissiveFactor: V3;
  emissiveStrength: number;
  emissiveTexture?: TextureRefJson;
  ior: number;
  specularFactor: number;
  specularColorFactor: V3;
  specularTexture?: TextureRefJson;
  specularColorTexture?: TextureRefJson;
  transmissionFactor: number;
  transmissionTexture?: TextureRefJson;
  alphaMode: 'OPAQUE' | 'MASK';
  alphaCutoff: number;
  doubleSided: boolean;
  emission: { color: V3; strength: number };
  emissionSampling: 'FRONT_BACK' | 'NONE';
}

export interface LightJson {
  id: number; name: string; type: LightData['type']; color: V3; power: number; exposure: number; matrix: number[];
  spotSize?: number; spotBlend?: number; sizeX?: number; sizeY?: number; spread?: number; visibleToCamera: boolean;
  simplified?: string;
}

export interface EnvJson {
  file: string; strength: number; tint: V3; rotationZ: number; visibleToCamera: boolean; sampling: 'AUTOMATIC' | 'NONE';
  /** SHA-256 of env.exr's pixels as written: little-endian float32 RGBA, rows top-down. */
  sha256: string;
  width: number; height: number; name?: string;
}

export interface SceneJson {
  format: typeof SCENE_PACKAGE_FORMAT;
  version: number;
  name: string;
  source?: { uri: string; sha256?: string };
  buffers: Record<string, BufferView>;
  flatShaded: boolean;
  materials: MaterialJson[];
  textures: { file: string; wrapS: string; wrapT: string; filter: string; name?: string; width?: number; height?: number }[];
  lights: LightJson[];
  lightMode: LightMode;
  camera: { matrix: number[]; yfov: number; znear: number };
  env: EnvJson | null;
  render: PackageRender;
  frames?: { frame: number; label?: string; camera?: { matrix: number[]; yfov: number }; lights?: Record<string, { matrix?: number[]; power?: number }>; env?: { rotationZ?: number; strength?: number } }[];
  warnings?: string[];
}

export interface ScenePackage { files: Map<string, Uint8Array>; json: SceneJson }

export class ScenePackageError extends Error {}

const arr = (a: ArrayLike<number>): number[] => Array.from(a, (x) => x);
const texName = (i: number) => `tex_${i}.png`;

/** Build the package files. Throws ScenePackageError for anything that cannot be represented exactly. */
export async function exportScenePackage(scene: SceneData, opts: ExportScenePackageOptions): Promise<ScenePackage> {
  const fail = (m: string): never => { throw new ScenePackageError(m); };
  const g = scene.geometry;
  const nTris = g.indices.length / 3;
  const nVerts = g.positions.length / 3;
  if (!(opts.render.width > 0 && opts.render.height > 0)) fail(`bad render size ${opts.render.width}x${opts.render.height}`);
  if (!Number.isInteger(opts.render.maxBounces) || opts.render.maxBounces < 0) fail(`bad maxBounces ${opts.render.maxBounces}`);
  if (opts.camera.matrix.length !== 16 || !(opts.camera.yfov > 0 && opts.camera.yfov < Math.PI)) fail('bad camera');
  if (!isRigid(opts.camera.matrix)) fail('camera matrix has scale/shear/mirror');
  for (const f of opts.frames ?? []) {
    if (f.camera && !isRigid(f.camera.matrix)) fail(`frame ${f.frame}: camera matrix has scale/shear/mirror`);
    for (const [id, l] of Object.entries(f.lights ?? {})) if (l.matrix && !isRigid(l.matrix)) fail(`frame ${f.frame}: light ${id} matrix has scale/shear/mirror`);
  }
  if (g.normals.length !== nVerts * 3 || g.uv0.length !== nVerts * 2 || g.triMaterial.length !== nTris) fail('inconsistent SceneGeometry lengths');
  for (let i = 0; i < g.triMaterial.length; i++) if (g.triMaterial[i] >= scene.materials.length) fail(`triangle ${i} references material ${g.triMaterial[i]}`);

  // ---- geometry.bin ----
  const parts: { key: string; data: Float32Array | Uint32Array; components: number }[] = [
    { key: 'positions', data: g.positions, components: 3 },
    { key: 'normals', data: g.normals, components: 3 },
    { key: 'uv0', data: g.uv0, components: 2 },
  ];
  if (g.color0) parts.push({ key: 'color0', data: g.color0, components: 4 });
  parts.push({ key: 'indices', data: g.indices, components: 3 }, { key: 'triMaterial', data: g.triMaterial, components: 1 },
    { key: 'triFlags', data: g.triFlags, components: 1 });
  const total = parts.reduce((s, p) => s + p.data.byteLength, 0);
  const bin = new Uint8Array(total);
  const bdv = new DataView(bin.buffer);
  const buffers: Record<string, BufferView> = {};
  let off = 0;
  for (const p of parts) {
    const isF = p.data instanceof Float32Array;
    for (let i = 0; i < p.data.length; i++) {
      if (isF) bdv.setFloat32(off + 4 * i, p.data[i], true); else bdv.setUint32(off + 4 * i, p.data[i], true);
    }
    buffers[p.key] = { offset: off, length: p.data.byteLength, dtype: isF ? 'f32' : 'u32', components: p.components };
    off += p.data.byteLength;
  }

  // ---- textures ----
  const files = new Map<string, Uint8Array>();
  files.set('geometry.bin', bin);
  const textures: SceneJson['textures'] = [];
  for (let i = 0; i < scene.textures.length; i++) {
    const t = scene.textures[i];
    if (t.pixels.length !== t.width * t.height * 4 || t.pixels.length === 0) {
      fail(`texture ${i} '${t.name}' has no decoded RGBA8 pixels (${t.pixels.length} bytes for ${t.width}x${t.height})`);
    }
    files.set(texName(i), await encodePng({ width: t.width, height: t.height, pixels: t.pixels }));
    textures.push({ file: texName(i), wrapS: t.wrapS, wrapT: t.wrapT, filter: t.filter, name: t.name, width: t.width, height: t.height });
  }

  // ---- materials ----
  const texRef = (r: TextureRef | undefined, what: string): TextureRefJson | undefined => {
    if (!r) return undefined;
    if (r.texture < 0 || r.texture >= scene.textures.length) fail(`${what}: texture index ${r.texture} out of range`);
    if (r.texCoord !== 0) fail(`${what}: texCoord ${r.texCoord} (only TEXCOORD_0 is bridged)`);
    if (r.transform && !isTrsTransform(r.transform)) fail(`${what}: skewed texture transform cannot be represented by a Mapping node`);
    return r.transform ? { texture: r.texture, texCoord: r.texCoord, transform: [...r.transform] } : { texture: r.texture, texCoord: r.texCoord };
  };
  const materials: MaterialJson[] = scene.materials.map((m, i) => {
    const w = `material ${i} '${m.name}'`;
    if ((m.alphaMode as string) === 'BLEND') fail(`${w}: alphaMode BLEND cannot be represented (convert to MASK first)`);
    if (m.model === 'v1' && !m.v1) fail(`${w}: model 'v1' without v1 parameters`);
    if (m.model === 'glass' || m.model === 'refraction') {
      // Cycles Glass / Refraction BSDF nodes (M3b): constant Color / Roughness / IOR only (build_scene.py
      // build_material_glass); emission, textures, alpha and transmission are not part of these node graphs.
      const extra = [m.baseColorTexture, m.metallicRoughnessTexture, m.normalTexture, m.emissiveTexture, m.transmissionTexture,
        m.specularTexture, m.specularColorTexture].some((t) => t !== undefined);
      if (extra) fail(`${w}: model '${m.model}' takes no textures`);
      if (Math.max(...m.emissiveFactor) * m.emissiveStrength > 0) fail(`${w}: model '${m.model}' cannot emit`);
      if (m.alphaMode !== 'OPAQUE') fail(`${w}: model '${m.model}' must be OPAQUE`);
    }
    const mj: MaterialJson = {
      name: m.name, model: m.model,
      baseColorFactor: [...m.baseColorFactor], baseColorTexture: texRef(m.baseColorTexture, w),
      metallicFactor: m.metallicFactor, roughnessFactor: m.roughnessFactor,
      metallicRoughnessTexture: texRef(m.metallicRoughnessTexture, w),
      normalTexture: m.normalTexture ? { ...texRef(m.normalTexture, w)!, scale: m.normalTexture.scale } : undefined,
      emissiveFactor: [...m.emissiveFactor], emissiveStrength: m.emissiveStrength, emissiveTexture: texRef(m.emissiveTexture, w),
      ior: m.ior, specularFactor: m.specularFactor, specularColorFactor: [...m.specularColorFactor],
      specularTexture: texRef(m.specularTexture, w), specularColorTexture: texRef(m.specularColorTexture, w),
      transmissionFactor: m.transmissionFactor, transmissionTexture: texRef(m.transmissionTexture, w),
      alphaMode: m.alphaMode as 'OPAQUE' | 'MASK', alphaCutoff: m.alphaCutoff, doubleSided: m.doubleSided,
      emission: { color: [...m.emissiveFactor], strength: m.emissiveStrength },
      emissionSampling: m.emissionSampling ?? 'FRONT_BACK',
    };
    if (m.v1) mj.v1 = { diffuse: [...m.v1.diffuse], glossy: [...m.v1.glossy], roughness: m.v1.roughness, mix: m.v1.mix };
    return stripUndefined(mj);
  });

  // ---- lights ----
  const ids = new Set<number>();
  const lights: LightJson[] = scene.lights.map((l) => {
    if (ids.has(l.id)) fail(`duplicate light id ${l.id}`);
    ids.add(l.id);
    const area = l.type === 'rect' || l.type === 'disk';
    if (area && !(l.sizeX! > 0 && (l.type === 'disk' || l.sizeY! > 0))) fail(`light '${l.name}': area light without a size`);
    if (l.type === 'disk' && l.sizeY !== undefined && Math.abs(l.sizeY - l.sizeX!) > 1e-6 * l.sizeX!) fail(`light '${l.name}': elliptical disk (sizeY ${l.sizeY} != sizeX ${l.sizeX})`);
    if (!isRigid(l.matrix)) fail(`light '${l.name}': matrix has scale/shear/mirror (lights are unscaled, plan §1.4)`);
    if (area && l.visibleToCamera && opts.lightMode === 'A') {
      fail(`light '${l.name}': camera-visible area lights are not allowed in Mode A (plan §1.4; Cycles 5.1.2 hides them without MIS)`);
    }
    const lj: LightJson = {
      id: l.id, name: l.name, type: l.type, color: [...l.color], power: l.power, exposure: l.exposure, matrix: arr(l.matrix),
      spotSize: l.spotSize, spotBlend: l.spotBlend, sizeX: l.sizeX, sizeY: l.type === 'disk' ? undefined : l.sizeY,
      spread: area ? (l.spread ?? Math.PI) : undefined, visibleToCamera: l.visibleToCamera, simplified: l.simplified,
    };
    return stripUndefined(lj);
  });

  // ---- env ----
  let env: EnvJson | null = null;
  if (scene.env) {
    const e = scene.env;
    const { exr, sha256 } = await encodeEnvExr(e);
    files.set('env.exr', exr);
    env = {
      file: 'env.exr', strength: e.strength, tint: [...e.tint], rotationZ: e.rotationZ, visibleToCamera: e.visibleToCamera,
      sampling: opts.envSampling ?? 'AUTOMATIC', sha256, width: e.width, height: e.height, name: e.name,
    };
  }

  const json: SceneJson = {
    format: SCENE_PACKAGE_FORMAT, version: SCENE_PACKAGE_VERSION,
    name: opts.name ?? scene.name,
    ...(opts.source ? { source: opts.source } : {}),
    buffers,
    flatShaded: opts.flatShaded ?? detectFlatShaded(g),
    materials,
    textures,
    lights,
    lightMode: opts.lightMode,
    camera: { matrix: arr(opts.camera.matrix), yfov: opts.camera.yfov, znear: opts.camera.znear ?? 1e-4 },
    env,
    render: { width: opts.render.width, height: opts.render.height, maxBounces: opts.render.maxBounces },
    ...(opts.frames ? {
      frames: opts.frames.map((f) => stripUndefined({
        frame: f.frame,
        label: f.label,
        camera: f.camera ? { matrix: arr(f.camera.matrix), yfov: f.camera.yfov } : undefined,
        lights: f.lights ? Object.fromEntries(Object.entries(f.lights).map(([k, v]) => [k, stripUndefined({ matrix: v.matrix ? arr(v.matrix) : undefined, power: v.power })])) : undefined,
        env: f.env,
      })),
    } : {}),
    ...(scene.warnings.length ? { warnings: [...scene.warnings] } : {}),
  };
  files.set('scene.json', new TextEncoder().encode(JSON.stringify(json, null, 1)));
  return { files, json };
}

/** env.exr bytes (ZIP, RGBA float32, rows top-down) and the ENV-U9 hash of exactly those pixels. */
export async function encodeEnvExr(e: EnvironmentData): Promise<{ exr: Uint8Array; sha256: string; topDown: Float32Array }> {
  if (e.texels.length !== e.width * e.height * 4) throw new ScenePackageError(`env texels ${e.texels.length} != ${e.width}x${e.height}x4`);
  const topDown = flipRows(e.texels, e.width, e.height, 4);
  const exr = await encodeExr({ width: e.width, height: e.height, data: topDown, channels: ['R', 'G', 'B', 'A'] }, 'zip');
  return { exr, sha256: await envPixelHash(topDown), topDown };
}

/** ENV-U9 hash: SHA-256 over the little-endian float32 RGBA array, rows top-down (as written to env.exr). */
export async function envPixelHash(topDownRgba: Float32Array): Promise<string> {
  const bytes = new Uint8Array(topDownRgba.length * 4);
  const dv = new DataView(bytes.buffer);
  for (let i = 0; i < topDownRgba.length; i++) dv.setFloat32(4 * i, topDownRgba[i], true);
  return sha256Hex(bytes);
}

/** Rotation + translation only (orthonormal columns, det +1, last row 0 0 0 1) to f32-level tolerance. */
export function isRigid(m: ArrayLike<number>, tol = 1e-5): boolean {
  const c = (i: number) => [m[4 * i], m[4 * i + 1], m[4 * i + 2]];
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const [x, y, z] = [c(0), c(1), c(2)];
  if ([dot(x, x), dot(y, y), dot(z, z)].some((v) => Math.abs(v - 1) > tol)) return false;
  if ([dot(x, y), dot(y, z), dot(x, z)].some((v) => Math.abs(v) > tol)) return false;
  const det = x[0] * (y[1] * z[2] - y[2] * z[1]) - y[0] * (x[1] * z[2] - x[2] * z[1]) + z[0] * (x[1] * y[2] - x[2] * y[1]);
  return det > 0 && m[3] === 0 && m[7] === 0 && m[11] === 0 && m[15] === 1;
}

/** A KHR_texture_transform-style [a b c; d e f] = T·R·S (no skew): columns of the 2x2 are orthogonal. */
export function isTrsTransform(t: readonly number[], tol = 1e-6): boolean {
  const [a, b, , d, e] = t;
  const n = Math.hypot(a, d) * Math.hypot(b, e);
  return Math.abs(a * b + d * e) <= tol * Math.max(n, 1e-30);
}

/** True when every corner normal equals its triangle's geometric normal (winding order) to 1e-6. */
export function detectFlatShaded(g: SceneGeometry): boolean {
  const p = g.positions, n = g.normals, idx = g.indices;
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t], b = idx[t + 1], c = idx[t + 2];
    const ex = p[3 * b] - p[3 * a], ey = p[3 * b + 1] - p[3 * a + 1], ez = p[3 * b + 2] - p[3 * a + 2];
    const fx = p[3 * c] - p[3 * a], fy = p[3 * c + 1] - p[3 * a + 1], fz = p[3 * c + 2] - p[3 * a + 2];
    let nx = ey * fz - ez * fy, ny = ez * fx - ex * fz, nz = ex * fy - ey * fx;
    const l = Math.hypot(nx, ny, nz);
    if (!(l > 0)) continue;
    nx /= l; ny /= l; nz /= l;
    for (const v of [a, b, c]) if (n[3 * v] * nx + n[3 * v + 1] * ny + n[3 * v + 2] * nz < 1 - 1e-6) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------------------------------------------
// Reading

export interface LoadedScenePackage {
  scene: SceneData;
  /** The package camera with the f64 matrix exactly as written (scene.cameras[0] holds an f32 copy). */
  camera: Omit<CameraData, 'matrix'> & { matrix: Float64Array };
  render: PackageRender;
  lightMode: LightMode;
  flatShaded: boolean;
  frames?: SceneJson['frames'];
  json: SceneJson;
}

export type PackageFiles = Map<string, Uint8Array> | Record<string, Uint8Array>;

/** Parse a package (files by name, as written by exportScenePackage). Verifies the env pixel hash. */
export async function readScenePackage(input: PackageFiles): Promise<LoadedScenePackage> {
  const files = input instanceof Map ? input : new Map(Object.entries(input));
  const get = (name: string): Uint8Array => {
    const f = files.get(name);
    if (!f) throw new ScenePackageError(`package file '${name}' missing`);
    return f;
  };
  const json = JSON.parse(new TextDecoder().decode(get('scene.json'))) as SceneJson;
  if (json.format !== SCENE_PACKAGE_FORMAT) throw new ScenePackageError(`not a scene package (format '${json.format}')`);
  if (json.version !== SCENE_PACKAGE_VERSION) throw new ScenePackageError(`unsupported package version ${json.version}`);
  const bin = get('geometry.bin');
  const bdv = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
  const view = (key: string, required: boolean): Float32Array | Uint32Array | undefined => {
    const b = json.buffers[key];
    if (!b) { if (required) throw new ScenePackageError(`buffer '${key}' missing`); return undefined; }
    if (b.offset + b.length > bin.byteLength || b.length % 4) throw new ScenePackageError(`buffer '${key}' out of range`);
    const n = b.length / 4;
    const out = b.dtype === 'f32' ? new Float32Array(n) : new Uint32Array(n);
    for (let i = 0; i < n; i++) out[i] = b.dtype === 'f32' ? bdv.getFloat32(b.offset + 4 * i, true) : bdv.getUint32(b.offset + 4 * i, true);
    return out;
  };
  const positions = view('positions', true) as Float32Array;
  const indices = view('indices', true) as Uint32Array;
  const triMaterial = view('triMaterial', true) as Uint32Array;
  const nVerts = positions.length / 3;
  const materials: MaterialData[] = json.materials.map(materialFromJson);
  const color0 = view('color0', false) as Float32Array | undefined;
  const triFlags = (view('triFlags', false) as Uint32Array | undefined) ?? computeTriFlags(triMaterial, materials, color0, indices);
  const geometry: SceneGeometry = {
    positions,
    normals: view('normals', true) as Float32Array,
    tangents: new Float32Array(nVerts * 4),
    uv0: (view('uv0', false) as Float32Array | undefined) ?? new Float32Array(nVerts * 2),
    color0,
    indices,
    triMaterial,
    triFlags,
  };
  const textures: TextureData[] = [];
  for (const t of json.textures ?? []) {
    const img = await decodePng(get(t.file));
    textures.push({
      name: t.name ?? t.file, width: img.width, height: img.height, pixels: img.pixels,
      wrapS: t.wrapS as TextureData['wrapS'], wrapT: t.wrapT as TextureData['wrapT'], filter: t.filter as TextureData['filter'],
    });
  }
  const lights: LightData[] = (json.lights ?? []).map((l) => stripUndefined({
    id: l.id, name: l.name ?? `light${l.id}`, type: l.type, color: [...(l.color ?? [1, 1, 1])] as V3, power: l.power, exposure: l.exposure ?? 0,
    matrix: new Float32Array(l.matrix), spotSize: l.spotSize, spotBlend: l.spotBlend, sizeX: l.sizeX, sizeY: l.sizeY,
    spread: l.spread, visibleToCamera: l.visibleToCamera ?? false, simplified: l.simplified,
  }));
  const camera = { name: 'package', matrix: new Float64Array(json.camera.matrix), yfov: json.camera.yfov, znear: json.camera.znear ?? 1e-4 };
  let env: EnvironmentData | undefined;
  if (json.env) {
    const e = json.env;
    const dec = decodeExr(get(e.file), 'validation'); // rows bottom-up (GPU order)
    const hash = await envPixelHash(flipRows(dec.texels, dec.width, dec.height, 4));
    if (e.sha256 && hash !== e.sha256) throw new ScenePackageError(`env pixel hash mismatch: file ${hash}, scene.json ${e.sha256}`);
    env = {
      name: e.name ?? e.file, width: dec.width, height: dec.height, texels: dec.texels, strength: e.strength,
      tint: [...(e.tint ?? [1, 1, 1])] as V3, rotationZ: e.rotationZ ?? 0, visibleToCamera: e.visibleToCamera ?? true,
    };
  }
  const scene: SceneData = {
    name: json.name, geometry, materials, textures, lights, cameras: [{ ...camera, matrix: new Float32Array(camera.matrix) }], env,
    bounds: boundsOf(positions, indices), warnings: [...(json.warnings ?? [])],
  };
  if (!scene.env) delete scene.env;
  const lightMode: LightMode = (json.lightMode as string) === "A'" ? 'A′' : json.lightMode;   // ASCII spelling accepted
  return { scene, camera, render: json.render, lightMode, flatShaded: json.flatShaded, frames: json.frames, json };
}

/** Fetch a package directory (URL ending in '/', or the scene.json URL) and read it. */
export async function fetchScenePackage(url: string, fetcher: (u: string) => Promise<Uint8Array> = fetchBytes): Promise<LoadedScenePackage & { files: Map<string, Uint8Array> }> {
  const base = url.endsWith('scene.json') ? url.slice(0, -'scene.json'.length) : url.endsWith('/') ? url : `${url}/`;
  const files = new Map<string, Uint8Array>();
  const sj = await fetcher(`${base}scene.json`);
  files.set('scene.json', sj);
  const json = JSON.parse(new TextDecoder().decode(sj)) as SceneJson;
  const names = ['geometry.bin', ...(json.textures ?? []).map((t) => t.file), ...(json.env ? [json.env.file] : [])];
  await Promise.all(names.map(async (n) => { files.set(n, await fetcher(`${base}${n}`)); }));
  return { ...(await readScenePackage(files)), files };
}

async function fetchBytes(u: string): Promise<Uint8Array> {
  const r = await fetch(u);
  if (!r.ok) throw new ScenePackageError(`${u}: HTTP ${r.status}`);
  if ((r.headers.get('content-type') ?? '').includes('text/html') && !u.endsWith('.html')) throw new ScenePackageError(`${u}: not found (dev server returned HTML)`);
  return new Uint8Array(await r.arrayBuffer());
}

/** Material from JSON. Missing principled fields take the glTF defaults (as build_scene.py does), so minimal
 *  packages written by other tools (e.g. validation/blender/calib_scenes.py v1 emitters) read too. */
function materialFromJson(mj: Partial<MaterialJson> & { name: string; model: MaterialJson['model']; principled?: Partial<MaterialJson> }): MaterialData {
  const m = { ...mj, ...(mj.principled ?? {}) }; // principled fields may be top level (our writer) or nested
  const ref = (r: TextureRefJson | undefined): TextureRef | undefined => (r
    ? (r.transform ? { texture: r.texture, texCoord: r.texCoord ?? 0, transform: r.transform as TextureRef['transform'] } : { texture: r.texture, texCoord: r.texCoord ?? 0 })
    : undefined);
  const md: MaterialData = {
    name: m.name,
    baseColorFactor: [...(m.baseColorFactor ?? [1, 1, 1, 1])] as MaterialData['baseColorFactor'],
    baseColorTexture: ref(m.baseColorTexture),
    metallicFactor: m.metallicFactor ?? 1,
    roughnessFactor: m.roughnessFactor ?? 1,
    metallicRoughnessTexture: ref(m.metallicRoughnessTexture),
    normalTexture: m.normalTexture ? { ...ref(m.normalTexture)!, scale: m.normalTexture.scale ?? 1 } : undefined,
    emissiveFactor: [...(m.emissiveFactor ?? m.emission?.color ?? [0, 0, 0])] as V3,
    emissiveStrength: m.emissiveStrength ?? m.emission?.strength ?? 1,
    emissiveTexture: ref(m.emissiveTexture),
    ior: m.ior ?? 1.5,
    specularFactor: m.specularFactor ?? 1,
    specularColorFactor: [...(m.specularColorFactor ?? [1, 1, 1])] as V3,
    specularTexture: ref(m.specularTexture),
    specularColorTexture: ref(m.specularColorTexture),
    transmissionFactor: m.transmissionFactor ?? 0,
    transmissionTexture: ref(m.transmissionTexture),
    alphaMode: m.alphaMode ?? 'OPAQUE',
    alphaCutoff: m.alphaCutoff ?? 0.5,
    doubleSided: m.doubleSided ?? true,
    model: m.model,
    ...(m.emissionSampling === 'NONE' ? { emissionSampling: 'NONE' as const } : {}),
    v1: m.v1 ? { diffuse: [...m.v1.diffuse] as V3, glossy: [...m.v1.glossy] as V3, roughness: m.v1.roughness, mix: m.v1.mix } : undefined,
  };
  if ((md.alphaMode as string) === 'BLEND') throw new ScenePackageError(`material '${m.name}': alphaMode BLEND`);
  if (md.model === 'v1' && !md.v1) throw new ScenePackageError(`material '${m.name}': model 'v1' without v1 parameters`);
  return stripUndefined(md);
}

/** triFlags when the package has none: emissive materials, and MASK materials (conservatively always tested). */
function computeTriFlags(triMaterial: Uint32Array, materials: MaterialData[], _color0: Float32Array | undefined, _indices: Uint32Array): Uint32Array {
  const out = new Uint32Array(triMaterial.length);
  for (let t = 0; t < out.length; t++) {
    const m = materials[triMaterial[t]];
    let f = 0;
    if (Math.max(...m.emissiveFactor) * m.emissiveStrength > 0) f |= TRI_EMISSIVE;
    if (m.alphaMode === 'MASK') f |= TRI_ALPHA_MASK;
    out[t] = f;
  }
  return out;
}

function boundsOf(p: Float32Array, idx: Uint32Array): SceneData['bounds'] {
  const mn: V3 = [Infinity, Infinity, Infinity], mx: V3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < idx.length; i++) {
    const v = idx[i];
    for (let k = 0; k < 3; k++) { const x = p[3 * v + k]; if (x < mn[k]) mn[k] = x; if (x > mx[k]) mx[k] = x; }
  }
  return idx.length ? { min: mn, max: mx } : { min: [0, 0, 0], max: [0, 0, 0] };
}

function stripUndefined<T extends object>(o: T): T {
  for (const k of Object.keys(o) as (keyof T)[]) if (o[k] === undefined) delete o[k];
  return o;
}
