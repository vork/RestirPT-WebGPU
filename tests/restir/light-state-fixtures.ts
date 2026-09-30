// Shared CPU fixtures of the M5 light-state tests (T-A): a tiny scene with two emissive triangles (one NEE entry,
// one emission_sampling NONE), analytic lights of every type and a small synthetic env importance table.
import { LightsState, type EnvLightInput } from '../../src/core/render/lights-gpu.ts';
import { buildEnvImportance } from '../../src/core/scene/env/env-importance.ts';
import type { LightData, MaterialData, SceneData } from '../../src/core/scene/types.ts';
import { TRI_EMISSIVE } from '../../src/core/scene/types.ts';

export const M = (p: number[]): Float32Array => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...p, 1]);

export function L(o: Partial<LightData> & Pick<LightData, 'type' | 'id'>): LightData {
  return { name: 'l', color: [1, 0.5, 0.25], power: 10, exposure: 0, matrix: M([1, 2, 3]), visibleToCamera: false, sizeX: 0.5, sizeY: 0.4, ...o };
}

function mat(o: Partial<MaterialData> = {}): MaterialData {
  return {
    name: 'm', baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 1, emissiveFactor: [0, 0, 0], emissiveStrength: 1, ior: 1.5,
    specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true, model: 'v1',
    v1: { diffuse: [0.5, 0.5, 0.5], glossy: [0, 0, 0], roughness: 0.5, mix: 0 }, ...o,
  };
}

export function scene(lights: LightData[] = []): SceneData {
  const positions = Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 1, 0, 2, 1, 5, 5, 5, 6, 5, 5, 5, 6, 5, 3, 3, 3, 4, 3, 3, 3, 4, 3]);
  return {
    name: 's',
    geometry: {
      positions, normals: new Float32Array(36), tangents: new Float32Array(48), uv0: new Float32Array(24),
      indices: Uint32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), triMaterial: Uint32Array.from([0, 1, 2, 1]),
      triFlags: Uint32Array.from([0, TRI_EMISSIVE, TRI_EMISSIVE, TRI_EMISSIVE]),
    },
    materials: [mat(), mat({ emissiveFactor: [2, 1, 0.5], emissiveStrength: 3 }), { ...mat({ emissiveFactor: [1, 1, 1] }), emissionSampling: 'NONE' } as MaterialData],
    textures: [], lights, cameras: [], bounds: { min: [0, 0, 0], max: [6, 6, 5] }, warnings: [],
  };
}

export function lightState(lights: LightData[] = []): LightsState {
  const s = scene(lights);
  return new LightsState(s, [0, 0, 0], s.geometry.positions, 'A');
}

let envCache: EnvLightInput['table'] | undefined;
export function envInput(strength = 1, tint: [number, number, number] = [1, 1, 1]): EnvLightInput {
  if (!envCache) {
    const W = 16, H = 8, t = new Float32Array(W * H * 4);
    for (let i = 0; i < W * H; i++) { t[4 * i] = 1 + (i % 7); t[4 * i + 1] = 1 + (i % 5); t[4 * i + 2] = 1 + (i % 3); t[4 * i + 3] = 1; }
    envCache = buildEnvImportance(t, W, H);
  }
  return { table: envCache, strength, tint, nee: true };
}

/** Deterministic PRNG (mulberry32). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
