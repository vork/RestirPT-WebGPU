// glTF/GLB → SceneData (types.ts). Runs in a Web Worker (gltf-loader.worker.ts) and in Node (tests).
// Parser: @gltf-transform/core + ALL_EXTENSIONS; decoders (draco3dgltf, meshoptimizer) are loaded lazily, only
// when the file uses them; then uninstance() + dequantize(). Semantics follow the Blender importer
// (plan §1.2, §1.4, §1.5; math.md#units-lights; gap-bsdf §9-§10; scene-io §3-§4).
import { type Document, type GLTF, type JSONDocument, type Material, type Texture, TextureInfo, WebIO } from '@gltf-transform/core';
import {
  ALL_EXTENSIONS, type EmissiveStrength, type IOR, type Light, type Specular, type Transform as TextureTransform, type Transmission,
} from '@gltf-transform/extensions';
import { dequantize, uninstance } from '@gltf-transform/functions';
import { flattenScene, type FlattenMaterialInfo, type FlattenStats, type TangentGenerator } from './flatten.ts';
import { isAlphaOpaque, type ImageDecoder } from './image-decode.ts';
import type { CameraData, LightData, MaterialData, SceneData, TextureData, TextureRef, WrapMode } from './types.ts';

// ---------------------------------------------------------------------------------------------------------------
// Public API

export type GltfSource =
  | { kind: 'glb'; bytes: Uint8Array; name?: string }
  /** `.gltf` JSON plus external resources keyed by their URI exactly as written in the JSON. */
  | { kind: 'gltf'; json: GLTF.IGLTF | string; resources?: Record<string, Uint8Array>; name?: string }
  /** Fetched (GLB or glTF with relative resources). */
  | { kind: 'url'; url: string; name?: string };

export interface GltfLoadOptions {
  /** Image decoder (browserImageDecoder in the browser/Worker). Omit in Node: textures keep sizes, pixels empty. */
  decodeImage?: ImageDecoder;
  /** Generate MikkTSpace tangents for normal-mapped primitives (default true). */
  tangents?: boolean;
  /** Override the MikkTSpace generator (tests); default loads the `mikktspace` wasm. */
  generateTangents?: TangentGenerator;
}

export interface LoadStats {
  ms: { parse: number; images: number; flatten: number; total: number };
  flatten: FlattenStats;
  extensionsUsed: string[];
}

export interface GltfLoadResult { scene: SceneData; stats: LoadStats }

export async function loadGltf(source: GltfSource, opts: GltfLoadOptions = {}): Promise<GltfLoadResult> {
  const t0 = now();
  const warnings: string[] = [];
  const jsonDoc = await toJSONDocument(source);
  const extensionsUsed = jsonDoc.json.extensionsUsed ?? [];
  const io = await createIO(extensionsUsed);
  const doc = await io.readJSON(jsonDoc);
  await doc.transform(uninstance(), dequantize());
  const t1 = now();

  const tex = await buildTextures(doc, opts.decodeImage, warnings);
  const t2 = now();
  const mats = buildMaterials(doc, tex, warnings);
  const gen = opts.tangents === false ? undefined : (opts.generateTangents ?? (await loadMikkTSpace()));
  const flat = flattenScene(doc, { materialInfo: mats.info, generateTangents: gen, warnings });
  const { lights, cameras } = collectLightsAndCameras(doc, warnings);
  const t3 = now();

  const scene: SceneData = {
    name: source.name ?? (source.kind === 'url' ? source.url.split('/').pop() ?? 'scene' : 'scene'),
    geometry: flat.geometry,
    materials: mats.materials,
    textures: tex.textures,
    lights,
    cameras,
    bounds: flat.bounds,
    warnings,
  };
  return {
    scene,
    stats: { ms: { parse: t1 - t0, images: t2 - t1, flatten: t3 - t2, total: t3 - t0 }, flatten: flat.stats, extensionsUsed },
  };
}

/** Every ArrayBuffer referenced by the scene, deduplicated (a duplicate in a transfer list is a DataCloneError). */
export function sceneTransferList(scene: SceneData): ArrayBuffer[] {
  const set = new Set<ArrayBuffer>();
  const add = (a: ArrayBufferView | undefined) => { if (a && a.buffer instanceof ArrayBuffer) set.add(a.buffer); };
  const g = scene.geometry;
  [g.positions, g.normals, g.tangents, g.uv0, g.color0, g.indices, g.triMaterial, g.triFlags].forEach(add);
  scene.textures.forEach((t) => add(t.pixels));
  scene.lights.forEach((l) => add(l.matrix));
  scene.cameras.forEach((c) => add(c.matrix));
  add(scene.env?.texels);
  return [...set];
}

// Worker protocol (gltf-loader.worker.ts ⇄ load-scene.ts).
export interface GltfWorkerRequest { id: number; source: GltfSource; tangents?: boolean }
export type GltfWorkerResponse =
  | { id: number; ok: true; result: GltfLoadResult }
  | { id: number; ok: false; error: string };

// ---------------------------------------------------------------------------------------------------------------
// Light / camera conversion (pure; unit-tested)

export interface PunctualLightInput {
  type: 'point' | 'spot' | 'directional';
  color: [number, number, number];
  intensity: number;           // cd (point/spot) or lux (directional)
  innerConeAngle?: number;     // default 0
  outerConeAngle?: number;     // default π/4
}

/**
 * KHR_lights_punctual → Blender light (importer `blender/imp/light.py`, SPEC mode; math.md#units-lights):
 * point/spot P = cd·4π/683 W, sun E = lux/683 W/m², spot_size = 2·outer, spot_blend = 1 − inner/outer.
 * Direction is local −Z; the matrix is the node's world matrix with scale removed. Never camera-visible.
 */
export function convertPunctualLight(l: PunctualLightInput, world: ArrayLike<number>, id: number, name: string): LightData {
  const base = { id, name, color: [...l.color] as [number, number, number], exposure: 0, matrix: rigidMatrix(world), visibleToCamera: false };
  if (l.type === 'directional') return { ...base, type: 'sun', power: l.intensity / 683 };
  const power = (l.intensity * 4 * Math.PI) / 683;
  if (l.type === 'point') return { ...base, type: 'point', power };
  const outer = l.outerConeAngle ?? Math.PI / 4;
  const inner = l.innerConeAngle ?? 0;
  return { ...base, type: 'spot', power, spotSize: 2 * outer, spotBlend: outer > 0 ? 1 - inner / outer : 0 };
}

/**
 * Column-major world matrix → rotation + translation (scale and shear removed). The −Z axis is kept exactly
 * (emission / view direction), +Y is Gram–Schmidt-orthogonalized against it, +X = Y × Z (right-handed).
 */
export function rigidMatrix(m: ArrayLike<number>): Float32Array {
  const z = norm3([m[8], m[9], m[10]]);
  let y = [m[4], m[5], m[6]];
  const d = y[0] * z[0] + y[1] * z[1] + y[2] * z[2];
  y = norm3([y[0] - d * z[0], y[1] - d * z[1], y[2] - d * z[2]]);
  const x = [y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0]];
  return new Float32Array([x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, z[0], z[1], z[2], 0, m[12], m[13], m[14], 1]);
}

/** KHR_texture_transform → row-major 2x3 [a b c; d e f] with uv' = T·R·S·[uv,1] (glTF uv space, origin top-left):
 *  R has rows [c s; −s c] (GLSL column-major mat3(c, −s, 0, s, c, 0, 0, 0, 1)), so
 *  u' = c·sx·u + s·sy·v + ox,  v' = −s·sx·u + c·sy·v + oy.
 *  Derived from Blender's `texture_transform_gltf_to_blender` (io_scene_gltf2/blender/com/conversion.py, Mapping node
 *  in v-flipped space) and identical to three.js GLTFLoader's "per glTF spec" override matrix. */
export function textureTransformMatrix(offset: [number, number], rotation: number, scale: [number, number]):
  [number, number, number, number, number, number] {
  const c = Math.cos(rotation), s = Math.sin(rotation);
  return [c * scale[0], s * scale[1], offset[0], -s * scale[0], c * scale[1], offset[1]];
}

// ---------------------------------------------------------------------------------------------------------------
// Parsing and decoders

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
const isNode = () => typeof process !== 'undefined' && !!process.versions?.node && typeof window === 'undefined'
  && typeof (globalThis as { WorkerGlobalScope?: unknown }).WorkerGlobalScope === 'undefined';

async function toJSONDocument(source: GltfSource): Promise<JSONDocument> {
  const io = new WebIO(); // only its pure GLB container parsing is used here
  if (source.kind === 'glb') return io.binaryToJSON(source.bytes);
  if (source.kind === 'gltf') {
    const json = typeof source.json === 'string' ? (JSON.parse(source.json) as GLTF.IGLTF) : source.json;
    return { json, resources: { ...(source.resources ?? {}) } as Record<string, Uint8Array<ArrayBuffer>> };
  }
  const res = await fetch(source.url);
  if (!res.ok) throw new Error(`fetch ${source.url}: HTTP ${res.status}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  if (isGlb(bytes)) return io.binaryToJSON(bytes);
  const json = JSON.parse(new TextDecoder().decode(bytes)) as GLTF.IGLTF;
  const resources: Record<string, Uint8Array<ArrayBuffer>> = {};
  const uris = [...(json.buffers ?? []), ...(json.images ?? [])].map((r) => r.uri).filter((u): u is string => !!u && !u.startsWith('data:'));
  await Promise.all([...new Set(uris)].map(async (uri) => {
    const r = await fetch(new URL(uri, new URL(source.url, globalThis.location?.href)).href);
    if (!r.ok) throw new Error(`fetch ${uri}: HTTP ${r.status}`);
    resources[uri] = new Uint8Array(await r.arrayBuffer());
  }));
  return { json, resources };
}

export function isGlb(bytes: Uint8Array): boolean {
  return bytes.length >= 12 && bytes[0] === 0x67 && bytes[1] === 0x6c && bytes[2] === 0x54 && bytes[3] === 0x46; // 'glTF'
}

let meshoptPromise: Promise<unknown> | undefined;
let dracoPromise: Promise<unknown> | undefined;

async function createIO(extensionsUsed: string[]): Promise<WebIO> {
  const io = new WebIO().registerExtensions(ALL_EXTENSIONS);
  const deps: Record<string, unknown> = {};
  if (extensionsUsed.includes('EXT_meshopt_compression') || extensionsUsed.includes('KHR_meshopt_compression')) {
    meshoptPromise ??= import('meshoptimizer/decoder').then(async ({ MeshoptDecoder }) => { await MeshoptDecoder.ready; return MeshoptDecoder; });
    deps['meshopt.decoder'] = await meshoptPromise;
  }
  if (extensionsUsed.includes('KHR_draco_mesh_compression')) {
    dracoPromise ??= loadDraco();
    deps['draco3d.decoder'] = await dracoPromise;
  }
  return io.registerDependencies(deps);
}

async function loadDraco(): Promise<unknown> {
  // @ts-ignore draco3dgltf ships no type declarations
  const mod = (await import('draco3dgltf')) as { default?: DracoFactory } & DracoFactory;
  const draco = mod.default ?? mod;
  if (isNode()) return draco.createDecoderModule({});
  // Browser/Worker: hand emscripten the wasm bytes so it never uses its own (Node/script-relative) lookup.
  const url = (await import('draco3dgltf/draco_decoder_gltf.wasm?url')).default;
  const wasmBinary = await (await fetch(url)).arrayBuffer();
  return draco.createDecoderModule({ wasmBinary });
}
interface DracoFactory { createDecoderModule(opts: { wasmBinary?: ArrayBuffer }): Promise<unknown> }

// MikkTSpace: the package's ESM entry imports its .wasm as an ES module (needs a bundler plugin), so we
// instantiate the same wasm ourselves with a minimal wasm-bindgen glue (2 imports, 1 export).
let mikkPromise: Promise<TangentGenerator> | undefined;

export function loadMikkTSpace(): Promise<TangentGenerator> {
  mikkPromise ??= (async () => instantiateMikkTSpace(await mikkWasmBytes()))();
  return mikkPromise;
}

async function mikkWasmBytes(): Promise<ArrayBuffer | Uint8Array<ArrayBuffer>> {
  const rel = 'mikktspace/dist/module/mikktspace_module_bg.wasm';
  if (isNode()) {
    const fs = 'node:fs/promises', mod = 'node:module';
    const { readFile } = (await import(/* @vite-ignore */ fs)) as typeof import('node:fs/promises');
    const { createRequire } = (await import(/* @vite-ignore */ mod)) as typeof import('node:module');
    return new Uint8Array(await readFile(createRequire(import.meta.url).resolve(rel)));
  }
  const url = (await import('mikktspace/dist/module/mikktspace_module_bg.wasm?url')).default;
  return (await fetch(url)).arrayBuffer();
}

interface MikkExports {
  memory: WebAssembly.Memory;
  generateTangents(ret: number, p0: number, l0: number, p1: number, l1: number, p2: number, l2: number): void;
  __wbindgen_add_to_stack_pointer(d: number): number;
  __wbindgen_malloc(n: number): number;
  __wbindgen_free(p: number, n: number): void;
}

async function instantiateMikkTSpace(bytes: ArrayBuffer | Uint8Array<ArrayBuffer>): Promise<TangentGenerator> {
  const heap: unknown[] = [];
  let ex!: MikkExports;
  const imports = {
    './mikktspace_module_bg.js': {
      __wbindgen_string_new: (ptr: number, len: number) => heap.push(new TextDecoder().decode(new Uint8Array(ex.memory.buffer, ptr, len))) - 1,
      __wbindgen_rethrow: (idx: number) => { throw new Error(String(heap[idx])); },
    },
  };
  const { instance } = await WebAssembly.instantiate(bytes, imports);
  ex = instance.exports as unknown as MikkExports;
  const pass = (a: Float32Array) => {
    const ptr = ex.__wbindgen_malloc(a.length * 4);
    new Float32Array(ex.memory.buffer, ptr, a.length).set(a);
    return ptr; // ownership moves to Rust (Vec<f32>)
  };
  return (position, normal, texcoord) => {
    const ret = ex.__wbindgen_add_to_stack_pointer(-16);
    try {
      const p0 = pass(position), p1 = pass(normal), p2 = pass(texcoord);
      ex.generateTangents(ret, p0, position.length, p1, normal.length, p2, texcoord.length);
      const i32 = new Int32Array(ex.memory.buffer);
      const r0 = i32[ret / 4], r1 = i32[ret / 4 + 1];
      const out = new Float32Array(ex.memory.buffer, r0, r1).slice();
      ex.__wbindgen_free(r0, r1 * 4);
      return out;
    } finally {
      ex.__wbindgen_add_to_stack_pointer(16);
    }
  };
}

// ---------------------------------------------------------------------------------------------------------------
// Textures

const WRAP: Record<number, WrapMode> = {
  [TextureInfo.WrapMode.REPEAT]: 'repeat',
  [TextureInfo.WrapMode.CLAMP_TO_EDGE]: 'clamp-to-edge',
  [TextureInfo.WrapMode.MIRRORED_REPEAT]: 'mirror-repeat',
};

interface TextureTable {
  textures: TextureData[];
  /** TextureRef for a (texture, textureInfo) use, or undefined when the image is unusable (KTX2, decode error). */
  ref(tex: Texture | null, info: TextureInfo | null): TextureRef | undefined;
  /** Alpha is 255 everywhere (false if unknown, e.g. not decoded). */
  opaque(tex: Texture): boolean;
  /** Number of references to TEXCOORD_n, n > 0 (warned once). */
  otherUvRefs(): number;
}

async function buildTextures(doc: Document, decode: ImageDecoder | undefined, warnings: string[]): Promise<TextureTable> {
  // Distinct images in first-use order over materials (deterministic texture indices).
  const used: Texture[] = [];
  for (const m of doc.getRoot().listMaterials()) {
    for (const t of materialTextures(m)) if (t.tex && !used.includes(t.tex)) used.push(t.tex);
  }
  interface Img { width: number; height: number; pixels: Uint8Array; opaque: boolean; ok: boolean }
  const images = new Map<Texture, Img>();
  let undecoded = 0;
  await Promise.all(used.map(async (tex, i) => {
    const mime = tex.getMimeType() || 'application/octet-stream';
    const name = tex.getName() || tex.getURI() || `image${i}`;
    const bytes = tex.getImage();
    if (!bytes) { warnings.push(`image '${name}' has no data; texture ignored`); images.set(tex, { width: 0, height: 0, pixels: new Uint8Array(0), opaque: false, ok: false }); return; }
    if (mime === 'image/ktx2') { warnings.push(`image '${name}': KTX2/Basis is not supported in v1; texture ignored`); images.set(tex, { width: 0, height: 0, pixels: new Uint8Array(0), opaque: false, ok: false }); return; }
    if (!decode) {
      const size = tex.getSize() ?? [0, 0];
      undecoded++;
      images.set(tex, { width: size[0], height: size[1], pixels: new Uint8Array(0), opaque: false, ok: true });
      return;
    }
    try {
      const d = await decode(bytes, mime, name);
      if (!d) { warnings.push(`image '${name}' (${mime}) cannot be decoded; texture ignored`); images.set(tex, { width: 0, height: 0, pixels: new Uint8Array(0), opaque: false, ok: false }); return; }
      images.set(tex, { ...d, opaque: isAlphaOpaque(d.pixels), ok: true });
    } catch (e) {
      warnings.push(`image '${name}' decode failed (${e instanceof Error ? e.message : String(e)}); texture ignored`);
      images.set(tex, { width: 0, height: 0, pixels: new Uint8Array(0), opaque: false, ok: false });
    }
  }));
  if (undecoded) warnings.push(`${undecoded} image(s) not decoded (no image decoder in this runtime); pixels are empty`);

  const textures: TextureData[] = [];
  const index = new Map<string, number>();
  const texId = new Map<Texture, number>();
  used.forEach((t, i) => texId.set(t, i));
  let otherUv = 0;
  const table: TextureTable = {
    textures,
    opaque: (tex) => images.get(tex)?.opaque ?? false,
    otherUvRefs: () => otherUv,
    ref(tex, info) {
      if (!tex || !info) return undefined;
      const img = images.get(tex);
      if (!img?.ok) return undefined;
      const wrapS = WRAP[info.getWrapS()] ?? 'repeat', wrapT = WRAP[info.getWrapT()] ?? 'repeat';
      const filter = info.getMagFilter() === TextureInfo.MagFilter.NEAREST ? 'nearest' : 'linear';
      const key = `${texId.get(tex)}|${wrapS}|${wrapT}|${filter}`;
      let ti = index.get(key);
      if (ti === undefined) {
        ti = textures.length;
        index.set(key, ti);
        // Same image with another sampler → a second TextureData sharing the same pixel buffer.
        textures.push({ name: tex.getName() || tex.getURI() || `texture${ti}`, width: img.width, height: img.height, pixels: img.pixels, wrapS, wrapT, filter });
      }
      const xf = info.getExtension<TextureTransform>('KHR_texture_transform');
      const texCoord = xf?.getTexCoord() ?? info.getTexCoord();
      if (texCoord !== 0) otherUv++;
      const r: TextureRef = { texture: ti, texCoord };
      if (xf) r.transform = textureTransformMatrix(xf.getOffset() as [number, number], xf.getRotation(), xf.getScale() as [number, number]);
      return r;
    },
  };
  return table;
}

function materialTextures(m: Material): { tex: Texture | null; info: TextureInfo | null }[] {
  const spec = m.getExtension<Specular>('KHR_materials_specular');
  const trans = m.getExtension<Transmission>('KHR_materials_transmission');
  return [
    { tex: m.getBaseColorTexture(), info: m.getBaseColorTextureInfo() },
    { tex: m.getMetallicRoughnessTexture(), info: m.getMetallicRoughnessTextureInfo() },
    { tex: m.getNormalTexture(), info: m.getNormalTextureInfo() },
    { tex: m.getEmissiveTexture(), info: m.getEmissiveTextureInfo() },
    { tex: spec?.getSpecularTexture() ?? null, info: spec?.getSpecularTextureInfo() ?? null },
    { tex: spec?.getSpecularColorTexture() ?? null, info: spec?.getSpecularColorTextureInfo() ?? null },
    { tex: trans?.getTransmissionTexture() ?? null, info: trans?.getTransmissionTextureInfo() ?? null },
  ];
}

// ---------------------------------------------------------------------------------------------------------------
// Materials (gap-bsdf §10.1 table; missing material → §10.3 Blender/Cycles defaults)

const SUPPORTED_MATERIAL_EXT = new Set([
  'KHR_materials_emissive_strength', 'KHR_materials_ior', 'KHR_materials_specular', 'KHR_materials_transmission',
]);

function buildMaterials(doc: Document, tex: TextureTable, warnings: string[]) {
  const materials: MaterialData[] = [];
  const infoOf = new Map<Material, FlattenMaterialInfo>();
  const ignoredExt = new Map<string, number>();
  let blendCount = 0;

  for (const m of doc.getRoot().listMaterials()) {
    for (const e of m.listExtensions()) {
      if (!SUPPORTED_MATERIAL_EXT.has(e.extensionName)) ignoredExt.set(e.extensionName, (ignoredExt.get(e.extensionName) ?? 0) + 1);
    }
    const es = m.getExtension<EmissiveStrength>('KHR_materials_emissive_strength');
    const ior = m.getExtension<IOR>('KHR_materials_ior');
    const spec = m.getExtension<Specular>('KHR_materials_specular');
    const trans = m.getExtension<Transmission>('KHR_materials_transmission');
    let alphaMode = m.getAlphaMode();
    let alphaCutoff = m.getAlphaCutoff();
    if (alphaMode === 'BLEND') { blendCount++; alphaMode = 'MASK'; alphaCutoff = 0.5; } // plan §0: BLEND → MASK 0.5
    const baseTex = tex.ref(m.getBaseColorTexture(), m.getBaseColorTextureInfo());
    const normalRef = tex.ref(m.getNormalTexture(), m.getNormalTextureInfo());
    const md: MaterialData = {
      name: m.getName() || `material${materials.length}`,
      baseColorFactor: [...m.getBaseColorFactor()] as MaterialData['baseColorFactor'],
      baseColorTexture: baseTex,
      metallicFactor: m.getMetallicFactor(),
      roughnessFactor: m.getRoughnessFactor(),
      metallicRoughnessTexture: tex.ref(m.getMetallicRoughnessTexture(), m.getMetallicRoughnessTextureInfo()),
      normalTexture: normalRef ? { ...normalRef, scale: m.getNormalScale() } : undefined,
      emissiveFactor: [...m.getEmissiveFactor()] as MaterialData['emissiveFactor'],
      emissiveStrength: es?.getEmissiveStrength() ?? 1,
      emissiveTexture: tex.ref(m.getEmissiveTexture(), m.getEmissiveTextureInfo()),
      ior: ior?.getIOR() ?? 1.5,
      specularFactor: spec?.getSpecularFactor() ?? 1,
      specularColorFactor: spec ? ([...spec.getSpecularColorFactor()] as [number, number, number]) : [1, 1, 1],
      specularTexture: spec ? tex.ref(spec.getSpecularTexture(), spec.getSpecularTextureInfo()) : undefined,
      specularColorTexture: spec ? tex.ref(spec.getSpecularColorTexture(), spec.getSpecularColorTextureInfo()) : undefined,
      transmissionFactor: trans?.getTransmissionFactor() ?? 0,
      transmissionTexture: trans ? tex.ref(trans.getTransmissionTexture(), trans.getTransmissionTextureInfo()) : undefined,
      alphaMode,
      alphaCutoff,
      doubleSided: m.getDoubleSided(),
      model: 'principled',
    };
    stripUndefined(md);
    const baseImg = m.getBaseColorTexture();
    infoOf.set(m, {
      index: materials.length,
      masked: alphaMode === 'MASK',
      alphaConstOne: md.baseColorFactor[3] === 1 && (!baseTex || (!!baseImg && tex.opaque(baseImg))),
      emissive: Math.max(...md.emissiveFactor) * md.emissiveStrength > 0,
      wantsTangents: !!md.normalTexture,
    });
    materials.push(md);
  }
  if (tex.otherUvRefs()) warnings.push(`${tex.otherUvRefs()} texture reference(s) use TEXCOORD_n with n > 0; v1 samples TEXCOORD_0 only`);
  if (blendCount) warnings.push(`${blendCount} BLEND material(s) rendered as MASK with alphaCutoff 0.5 (v1)`);
  for (const [name, n] of ignoredExt) warnings.push(`material extension ${name} ignored (${n} material(s))`);

  // Primitives without a material: Blender DefaultMaterial with COLOR_0 (base = vertex colour), else Cycles'
  // default surface (Principled base 0.8, metallic 0, roughness 0.5, IOR 1.5) — gap-bsdf §10.3.
  const defaults = new Map<boolean, FlattenMaterialInfo>();
  const info = (mat: Material | null, hasColor0: boolean): FlattenMaterialInfo => {
    if (mat) return infoOf.get(mat)!;
    let d = defaults.get(hasColor0);
    if (!d) {
      const b = hasColor0 ? 1 : 0.8;
      materials.push({
        name: hasColor0 ? '__default_vertex_color' : '__default', baseColorFactor: [b, b, b, 1], metallicFactor: 0, roughnessFactor: 0.5,
        emissiveFactor: [0, 0, 0], emissiveStrength: 1, ior: 1.5, specularFactor: 1, specularColorFactor: [1, 1, 1],
        transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true, model: 'principled',
      });
      d = { index: materials.length - 1, masked: false, alphaConstOne: true, emissive: false, wantsTangents: false };
      defaults.set(hasColor0, d);
    }
    return d;
  };
  return { materials, info };
}

function stripUndefined(o: object): void {
  for (const k of Object.keys(o) as (keyof typeof o)[]) if (o[k] === undefined) delete o[k];
}

// ---------------------------------------------------------------------------------------------------------------
// Lights and cameras

function collectLightsAndCameras(doc: Document, warnings: string[]): { lights: LightData[]; cameras: CameraData[] } {
  const lights: LightData[] = [];
  const cameras: CameraData[] = [];
  const root = doc.getRoot();
  const scene = root.getDefaultScene() ?? root.listScenes()[0];
  let ranged = 0, ortho = 0;
  scene?.traverse((node) => {
    const light = node.getExtension<Light>('KHR_lights_punctual');
    if (light) {
      if (light.getRange() !== null) ranged++;
      lights.push(convertPunctualLight({
        type: light.getType() as PunctualLightInput['type'],
        color: [...light.getColor()] as [number, number, number],
        intensity: light.getIntensity(),
        innerConeAngle: light.getInnerConeAngle(),
        outerConeAngle: light.getOuterConeAngle(),
      }, node.getWorldMatrix(), lights.length, light.getName() || node.getName() || `light${lights.length}`));
    }
    const cam = node.getCamera();
    if (cam) {
      if (cam.getType() !== 'perspective') { ortho++; return; }
      cameras.push({ name: cam.getName() || node.getName() || `camera${cameras.length}`, matrix: rigidMatrix(node.getWorldMatrix()), yfov: cam.getYFov(), znear: cam.getZNear() });
    }
  });
  if (ranged) warnings.push(`${ranged} punctual light(s) define 'range'; ignored like the Blender importer`);
  if (ortho) warnings.push(`${ortho} orthographic camera(s) skipped (v1 supports perspective only)`);
  return { lights, cameras };
}

function norm3(v: number[]): number[] {
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > 0 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0];
}
