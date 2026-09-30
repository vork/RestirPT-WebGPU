// CPU-side scene representation shared by loaders, BVH builder, renderer and the Cycles scene bridge.
// Frame: glTF canonical (right-handed, +Y up, metres, UN-recentred world coordinates) — plan §1.2.
// Recentring is render-internal and applied identically to everything (see RenderScene.origin).

/** Per-triangle flag bits (triFlags). */
export const TRI_ALPHA_MASK = 1 << 0;   // material alphaMode MASK with a textured/vertex alpha (needs any-hit test)
export const TRI_EMISSIVE = 1 << 1;     // material emission > 0 (emissive-triangle light)
export const TRI_FLIPPED = 1 << 2;      // world transform had negative determinant (winding flipped at flatten)
/** Flat face (every corner normal equals the face normal; set by quantizeScene): shading uses ns = ng exactly, like
 *  Cycles on flat faces, and the stored vertex normal is never read (docs/decisions/data-formats.md §B3). */
export const TRI_FLAT = 1 << 3;

export interface SceneGeometry {
  /** World-space vertex positions (xyz per vertex), float32. Quantized scenes: on the global 2^posLog2 lattice. */
  positions: Float32Array;
  /** World-space shading normals (xyz per vertex, normalized). Flat-shaded faces are unwelded. */
  normals: Float32Array;
  /** MikkTSpace tangents (xyzw per vertex, w = bitangent sign); zeros if the mesh has no UVs. */
  tangents: Float32Array;
  /** TEXCOORD_0 in glTF convention (origin top-left of the image), 2 per vertex. */
  uv0: Float32Array;
  /** Vertex colour COLOR_0 (rgba per vertex) or undefined (= white). */
  color0?: Float32Array;
  /** 3 vertex indices per triangle. primId = triangle index in this array and is STABLE (plan §1.3). */
  indices: Uint32Array;
  /** Material index per triangle. */
  triMaterial: Uint32Array;
  /** TRI_* flags per triangle. */
  triFlags: Uint32Array;
}

export type WrapMode = 'repeat' | 'clamp-to-edge' | 'mirror-repeat';

export interface TextureRef {
  /** Index into SceneData.textures. */
  texture: number;
  /** UV set (0 or 1; v1 supports 0 only, others logged). */
  texCoord: number;
  /** KHR_texture_transform as a 2x3 affine (row-major [a b c; d e f]) applied to uv, or undefined. */
  transform?: [number, number, number, number, number, number];
}

export interface TextureData {
  name: string;
  width: number;
  height: number;
  /** RGBA8, rows top-first, NOT premultiplied, NO colour conversion (createImageBitmap colorSpaceConversion none).
   *  (glTF loader) Length 0 when loaded without an image decoder (Node tests); width/height then come from the header. */
  pixels: Uint8Array;
  wrapS: WrapMode;
  wrapT: WrapMode;
  /** 'nearest' for glTF NEAREST mag filters (Blender 'Closest'), else 'linear'. */
  filter: 'linear' | 'nearest';
}

export type AlphaMode = 'OPAQUE' | 'MASK' | 'BLEND';

/** glTF metallic-roughness (+ extensions) → Cycles Principled V2 inputs (plan §1.5; math.md#bsdf-v2). */
export interface MaterialData {
  name: string;
  baseColorFactor: [number, number, number, number];
  baseColorTexture?: TextureRef;      // sRGB
  metallicFactor: number;
  roughnessFactor: number;
  metallicRoughnessTexture?: TextureRef; // linear; B = metallic, G = roughness
  normalTexture?: TextureRef & { scale: number };
  emissiveFactor: [number, number, number];
  emissiveStrength: number;           // KHR_materials_emissive_strength (default 1)
  emissiveTexture?: TextureRef;       // sRGB
  ior: number;                        // KHR_materials_ior (default 1.5)
  specularFactor: number;             // KHR_materials_specular (default 1) → Specular IOR Level = 0.5*factor
  specularColorFactor: [number, number, number];
  /** (added by the glTF loader) KHR_materials_specular specularTexture: linear, A = factor multiplier. */
  specularTexture?: TextureRef;
  /** (added by the glTF loader) KHR_materials_specular specularColorTexture: sRGB, RGB multiplies specularColorFactor. */
  specularColorTexture?: TextureRef;
  transmissionFactor: number;         // KHR_materials_transmission (default 0)
  transmissionTexture?: TextureRef;
  alphaMode: AlphaMode;
  alphaCutoff: number;                // default 0.5 (BLEND is rendered as MASK with 0.5, plan §0)
  doubleSided: boolean;               // informational: Cycles renders everything two-sided
  /** Which BSDF model the renderer uses for this material (V1 validation materials come from scene.json).
   *  'glass' = Cycles Glass BSDF node, 'refraction' = Cycles Refraction BSDF node (validation, M3b; math.md#glass):
   *  Color = baseColorFactor.rgb, Roughness = roughnessFactor, IOR = ior; no other field is used. */
  model: 'principled' | 'v1' | 'glass' | 'refraction';
  /** Cycles material emission_sampling (default FRONT_BACK). 'NONE': the emissive triangles are not NEE entries
   *  (BSDF-only, ω2 := 1; glass §7.1 furnace enclosures). */
  emissionSampling?: 'FRONT_BACK' | 'NONE';
  /** V1 parameters (validation): Lambert albedo, GGX glossy colour/roughness, mix factor. */
  v1?: { diffuse: [number, number, number]; glossy: [number, number, number]; roughness: number; mix: number };
}

export type LightType = 'point' | 'spot' | 'rect' | 'disk' | 'sun';

/** Analytic light in Blender units (plan §1.4; math.md#units-lights). Emits along local −Z. No scale. */
export interface LightData {
  id: number;                  // stable id (editor + temporal id maps)
  name: string;
  type: LightType;
  color: [number, number, number];
  power: number;               // W (point/spot/area) or W/m^2 irradiance (sun)
  exposure: number;
  /** World transform (column-major 4x4, rotation+translation only). */
  matrix: Float32Array;
  spotSize?: number;           // full cone angle, radians
  spotBlend?: number;
  sizeX?: number;              // rect: full width (m); disk: diameter
  sizeY?: number;              // rect: full height (m)
  spread?: number;             // radians, (0, π]
  visibleToCamera: boolean;
  /** Set when a file light was simplified (USD sphere radius → 0, distant angle → 0). */
  simplified?: string;
}

export interface CameraData {
  name: string;
  /** Camera-to-world, column-major 4x4 (looks down local −Z, +Y up). */
  matrix: Float32Array;
  yfov: number;                // radians
  znear: number;
}

/** Equirect HDRI (plan §1.4b; math.md#env-mapping). */
export interface EnvironmentData {
  name: string;
  width: number;
  height: number;
  /** RGBA float32, rows BOTTOM-UP (row 0 = nadir, v = 0), as uploaded to the GPU. */
  texels: Float32Array;
  strength: number;
  tint: [number, number, number];
  rotationZ: number;           // Blender Mapping-node rotation Z (radians)
  visibleToCamera: boolean;
}

export interface Bounds { min: [number, number, number]; max: [number, number, number] }

/** Per-material UV lattice (data-formats.md §B5): uv = (q + base)·2^k per axis, q ∈ [0, 65535]; `wide` = f32 UVs. */
export interface UvLattice { ku: number; kv: number; baseU: number; baseV: number; wide: boolean }

/** Quantization parameters of a scene (quantizeScene, scene package v2 `quant`; data-formats.md §B0). */
export interface SceneQuant {
  /** 'quantized': every SceneGeometry value is on the lattices below; 'lossless': f32 everything (identity). */
  mode: 'quantized' | 'lossless';
  /** Global position lattice exponent k (step 2^k m, P21: 3 × 21 bit offsets). */
  posLog2: number;
  /** One lattice per material (index = material index). */
  uv: UvLattice[];
  /** τ (texels) the lattices were chosen with. */
  uvTolerance: number;
  normal: 'oct16' | 'f32';
  tangent: 'oct15' | 'f32';
  color: 'rgba8' | 'rgba16' | 'f32' | 'none';
}

export interface SceneData {
  name: string;
  geometry: SceneGeometry;
  materials: MaterialData[];
  textures: TextureData[];
  lights: LightData[];
  cameras: CameraData[];
  env?: EnvironmentData;
  bounds: Bounds;
  /** Loader warnings (unsupported extensions, simplified lights, dropped degenerate triangles...). */
  warnings: string[];
  /** Set by quantizeScene (every loader runs it). Absent = never quantized (ad-hoc test scenes): f32 GPU format. */
  quant?: SceneQuant;
}
