// Browser-safe core of the validation scene kit (no node: imports): vector helpers, matrices, materials, the flat
// MeshBuilder, sceneOf (quantizeScene), lights and procedural textures. scene-kit.ts re-exports everything and adds the
// Node-only parts (Cornell base from disk, the package writer). Split out in M7 so GPU tests can build kit scenes.
import { quantizeScene } from '../../src/core/scene/quantize.ts';
import {
  TRI_ALPHA_MASK, TRI_EMISSIVE, type LightData, type MaterialData, type SceneData, type SceneGeometry, type TextureData,
} from '../../src/core/scene/types.ts';

export const deg = Math.PI / 180;
export type V3 = [number, number, number];

export const sub = (a: readonly number[], b: readonly number[]): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a: readonly number[], b: readonly number[]): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a: readonly number[], s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const cross = (a: readonly number[], b: readonly number[]): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const dot = (a: readonly number[], b: readonly number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const norm = (a: readonly number[]): V3 => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };

/** Deterministic PRNG (mulberry32) for generator randomness (scene files must regenerate byte-identically). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---- matrices ------------------------------------------------------------------------------------------------------

/** Column-major rigid matrix from unit right-handed axes and a position. */
export const frame = (X: V3, Y: V3, Z: V3, p: V3): number[] => [...X, 0, ...Y, 0, ...Z, 0, ...p, 1];

/** Camera-to-world looking from `eye` at `target` (camera looks down local −Z, +Y up). */
export function lookAt(eye: V3, target: V3, up: V3 = [0, 1, 0]): number[] {
  const z = norm(sub(eye, target));
  const x = norm(cross(up, z));
  return frame(x, cross(z, x), z, eye);
}

/** Light matrix at p whose emission axis a_L = −Z_obj points along `dir` (sun: dir = direction the light travels). */
export function lightToward(dir: readonly number[], p: V3, up: V3 = [0, 1, 0]): Float32Array {
  const Z = norm(scale(dir, -1));
  const helper: V3 = Math.abs(dot(Z, up)) < 0.95 ? up : [1, 0, 0];
  const X = norm(cross(helper, Z));
  return new Float32Array(frame(X, cross(Z, X), Z, p));
}

/** Unit direction from elevation (above the XZ plane) and azimuth (from +X toward +Z), both radians. */
export const dirFromAngles = (elev: number, azim: number): V3 => [Math.cos(elev) * Math.cos(azim), Math.sin(elev), Math.cos(elev) * Math.sin(azim)];

// ---- materials -----------------------------------------------------------------------------------------------------

const BASE_MAT: Omit<MaterialData, 'name' | 'model'> = {
  baseColorFactor: [0.8, 0.8, 0.8, 1], metallicFactor: 0, roughnessFactor: 0.5, emissiveFactor: [0, 0, 0], emissiveStrength: 1,
  ior: 1.5, specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true,
};

/** V1 validation material: (1−mix)·Lambert(diffuse) + mix·GGX(glossy, roughness, F ≡ 1) [+ two-sided emission]. */
export function v1(name: string, o: { diffuse?: V3; glossy?: V3; roughness?: number; mix?: number; emission?: V3 } = {}): MaterialData {
  const diffuse = o.diffuse ?? [0.8, 0.8, 0.8];
  return {
    ...BASE_MAT, name, model: 'v1', baseColorFactor: [...diffuse, 1],
    emissiveFactor: o.emission ?? [0, 0, 0], emissiveStrength: 1,
    v1: { diffuse, glossy: o.glossy ?? [0, 0, 0], roughness: o.roughness ?? 0.5, mix: o.mix ?? 0 },
  };
}

/** Principled V2 (glTF metallic-roughness semantics, math.md#bsdf-v2). specularLevel = 0.5·specularFactor. */
export function principled(name: string, o: Partial<Omit<MaterialData, 'name' | 'model'>> & { specularLevel?: number } = {}): MaterialData {
  const { specularLevel, ...rest } = o;
  return { ...BASE_MAT, ...rest, ...(specularLevel !== undefined ? { specularFactor: specularLevel / 0.5 } : {}), name, model: 'principled' };
}

/** Cycles Glass BSDF node (M3b): Color, Roughness, IOR (distribution GGX; math.md#glass). */
export function glassNode(name: string, color: V3 = [1, 1, 1], roughness = 0, ior = 1.5): MaterialData {
  return { ...BASE_MAT, name, model: 'glass', baseColorFactor: [...color, 1], roughnessFactor: roughness, ior, metallicFactor: 0 };
}

/** Cycles Refraction BSDF node (M3b): Color, Roughness, IOR (distribution GGX; no reflection, TIR kills the path). */
export function refractionNode(name: string, color: V3 = [1, 1, 1], roughness = 0, ior = 1.5): MaterialData {
  return { ...BASE_MAT, name, model: 'refraction', baseColorFactor: [...color, 1], roughnessFactor: roughness, ior, metallicFactor: 0 };
}

/** Two-sided black emitter with Cycles emission_sampling NONE (BSDF-only: furnace enclosures, glass §7.1). */
export function emitterNone(name: string, Le: V3 = [1, 1, 1]): MaterialData {
  return { ...v1(name, { diffuse: [0, 0, 0], emission: Le }), emissionSampling: 'NONE' };
}

// ---- mesh builder (flat shaded: every face has its own vertices with the face normal) -------------------------------

export class MeshBuilder {
  readonly pos: number[] = [];
  readonly nrm: number[] = [];
  readonly uv: number[] = [];
  readonly idx: number[] = [];
  readonly mat: number[] = [];

  get triangles(): number { return this.mat.length; }

  /** Triangle a, b, c (counter-clockwise seen from the normal side) with optional glTF UVs. */
  tri(a: readonly number[], b: readonly number[], c: readonly number[], mat: number, uvs?: [number, number][]): this {
    const n = norm(cross(sub(b, a), sub(c, a)));
    const base = this.pos.length / 3;
    [a, b, c].forEach((p, k) => { this.pos.push(p[0], p[1], p[2]); this.nrm.push(...n); this.uv.push(...(uvs?.[k] ?? [0, 0])); });
    this.idx.push(base, base + 1, base + 2);
    this.mat.push(mat);
    return this;
  }

  /** Quad p0 p1 p2 p3 (counter-clockwise from the normal side; normal = (p1−p0)×(p3−p0)); UV (0,1) (1,1) (1,0) (0,0) × uvScale. */
  quad(p0: readonly number[], p1: readonly number[], p2: readonly number[], p3: readonly number[], mat: number, uvScale: [number, number] = [1, 1]): this {
    const n = norm(cross(sub(p1, p0), sub(p3, p0)));
    const base = this.pos.length / 3;
    const uvs: [number, number][] = [[0, uvScale[1]], [uvScale[0], uvScale[1]], [uvScale[0], 0], [0, 0]];
    [p0, p1, p2, p3].forEach((p, k) => { this.pos.push(p[0], p[1], p[2]); this.nrm.push(...n); this.uv.push(...uvs[k]); });
    this.idx.push(base, base + 1, base + 2, base, base + 2, base + 3);
    this.mat.push(mat, mat);
    return this;
  }

  /** Horizontal quad y = h over [x0, x1] × [z0, z1] facing +Y (or −Y with down). */
  floor(x0: number, x1: number, z0: number, z1: number, h: number, mat: number, o: { down?: boolean; uvScale?: [number, number] } = {}): this {
    return o.down
      ? this.quad([x0, h, z0], [x1, h, z0], [x1, h, z1], [x0, h, z1], mat, o.uvScale)
      : this.quad([x0, h, z1], [x1, h, z1], [x1, h, z0], [x0, h, z0], mat, o.uvScale);
  }

  /** Axis-aligned box (outward faces), rotated by `rotY` about its vertical centre axis; `omit` drops faces (e.g. '-y'). */
  box(min: V3, max: V3, mat: number, o: { rotY?: number; omit?: string[]; inward?: boolean } = {}): this {
    const c: V3 = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
    const cr = Math.cos(o.rotY ?? 0), sr = Math.sin(o.rotY ?? 0);
    const P = (x: number, y: number, z: number): V3 => {
      const dx = x - c[0], dz = z - c[2];
      return [c[0] + cr * dx + sr * dz, y, c[2] - sr * dx + cr * dz];
    };
    const [x0, y0, z0] = min, [x1, y1, z1] = max;
    const faces: [string, V3[]][] = [
      ['+x', [P(x1, y0, z1), P(x1, y0, z0), P(x1, y1, z0), P(x1, y1, z1)]],
      ['-x', [P(x0, y0, z0), P(x0, y0, z1), P(x0, y1, z1), P(x0, y1, z0)]],
      ['+y', [P(x0, y1, z1), P(x1, y1, z1), P(x1, y1, z0), P(x0, y1, z0)]],
      ['-y', [P(x0, y0, z0), P(x1, y0, z0), P(x1, y0, z1), P(x0, y0, z1)]],
      ['+z', [P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)]],
      ['-z', [P(x1, y0, z0), P(x0, y0, z0), P(x0, y1, z0), P(x1, y1, z0)]],
    ];
    for (const [k, q] of faces) {
      if (o.omit?.includes(k)) continue;
      if (o.inward) this.quad(q[0], q[3], q[2], q[1], mat); else this.quad(q[0], q[1], q[2], q[3], mat);
    }
    return this;
  }

  /** Flat-shaded icosphere (outward, or inward for enclosures), 20·4^subdiv triangles. */
  icosphere(center: V3, r: number, subdiv: number, mat: number, o: { inward?: boolean } = {}): this {
    const t = (1 + Math.sqrt(5)) / 2;
    let verts: V3[] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]].map((v) => norm(v));
    let faces: [number, number, number][] = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
    for (let s = 0; s < subdiv; s++) {
      const cache = new Map<string, number>();
      const mid = (a: number, b: number): number => {
        const k = a < b ? `${a},${b}` : `${b},${a}`;
        let m = cache.get(k);
        if (m === undefined) { m = verts.length; verts.push(norm(scale(add(verts[a], verts[b]), 0.5))); cache.set(k, m); }
        return m;
      };
      const next: [number, number, number][] = [];
      for (const [a, b, c] of faces) {
        const ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
        next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
      }
      faces = next;
      verts = [...verts];
    }
    const P = (i: number): V3 => add(center, scale(verts[i], r));
    for (const [a, b, c] of faces) {
      if (o.inward) this.tri(P(a), P(c), P(b), mat); else this.tri(P(a), P(b), P(c), mat);
    }
    return this;
  }

  /** Append an existing (flat) geometry, offsetting its material indices. */
  append(g: SceneGeometry, matOffset = 0): this {
    const base = this.pos.length / 3;
    this.pos.push(...g.positions);
    this.nrm.push(...g.normals);
    this.uv.push(...g.uv0);
    for (const i of g.indices) this.idx.push(base + i);
    for (const m of g.triMaterial) this.mat.push(m + matOffset);
    return this;
  }

  /** SceneGeometry with TRI_EMISSIVE / TRI_ALPHA_MASK flags from the materials. */
  build(materials: readonly MaterialData[]): SceneGeometry {
    const flags = this.mat.map((m) => {
      const md = materials[m];
      if (!md) throw new Error(`material ${m} missing`);
      let f = 0;
      if (Math.max(...md.emissiveFactor) * md.emissiveStrength > 0) f |= TRI_EMISSIVE;
      if (md.alphaMode === 'MASK' && (md.baseColorTexture || md.baseColorFactor[3] < 1)) f |= TRI_ALPHA_MASK;
      return f;
    });
    const nV = this.pos.length / 3;
    return {
      positions: Float32Array.from(this.pos), normals: Float32Array.from(this.nrm), tangents: new Float32Array(nV * 4),
      uv0: Float32Array.from(this.uv), indices: Uint32Array.from(this.idx), triMaterial: Uint32Array.from(this.mat), triFlags: Uint32Array.from(flags),
    };
  }
}

export function boundsOf(g: SceneGeometry): SceneData['bounds'] {
  const mn: V3 = [Infinity, Infinity, Infinity], mx: V3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < g.positions.length; i += 3) {
    for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], g.positions[i + k]); mx[k] = Math.max(mx[k], g.positions[i + k]); }
  }
  return { min: mn, max: mx };
}

/** The scene of `mb`, quantized (data-formats.md §B0: global position lattice, TRI_FLAT, UV lattices, re-weld). */
export function sceneOf(name: string, mb: MeshBuilder, materials: MaterialData[], lights: LightData[], textures: TextureData[] = []): SceneData {
  const geometry = mb.build(materials);
  return quantizeScene({ name, geometry, materials, textures, lights, cameras: [], bounds: boundsOf(geometry), warnings: [] }).scene;
}

// ---- lights --------------------------------------------------------------------------------------------------------

let nextLightId = 0;
export function resetLightIds(): void { nextLightId = 0; }

export function light(type: LightData['type'], matrix: Float32Array, power: number, o: Partial<LightData> = {}): LightData {
  const area = type === 'rect' || type === 'disk';
  return {
    id: nextLightId++, name: `${type}${nextLightId - 1}`, type, color: [1, 1, 1], power, exposure: 0, matrix, visibleToCamera: false,
    ...(area ? { spread: Math.PI } : {}), ...o,
  };
}

// ---- textures ------------------------------------------------------------------------------------------------------

export function texture(name: string, width: number, height: number, texel: (x: number, y: number) => [number, number, number, number],
  o: { wrapS?: TextureData['wrapS']; wrapT?: TextureData['wrapT']; filter?: TextureData['filter'] } = {}): TextureData {
  const pixels = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const c = texel(x, y);
      for (let k = 0; k < 4; k++) pixels[4 * (y * width + x) + k] = Math.max(0, Math.min(255, Math.round(c[k])));
    }
  }
  return { name, width, height, pixels, wrapS: o.wrapS ?? 'repeat', wrapT: o.wrapT ?? 'repeat', filter: o.filter ?? 'linear' };
}

/** KHR_texture_transform 2x3 [a b c; d e f] = [c·sx, s·sy, ox; −s·sx, c·sy, oy] (gltf-loader.ts textureTransformMatrix). */
export function texTransform(offset: [number, number], rotation: number, sc: [number, number]): [number, number, number, number, number, number] {
  const c = Math.cos(rotation), s = Math.sin(rotation);
  return [c * sc[0], s * sc[1], offset[0], -s * sc[0], c * sc[1], offset[1]];
}

