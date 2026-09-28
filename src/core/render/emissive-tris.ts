// Emissive-triangle light list (plan §1.4 "Storage": emissive triangles stay in the BVH, two-sided, opaque,
// area-sampled; math.md#light-selection "Entries" and "Power proxies"; math.md#units-lights "Emissive triangle").
// - Every triangle flagged TRI_EMISSIVE whose material does not opt out of light sampling
//   (emission_sampling NONE → BSDF-only, ω2 := 1; glass §7.1) becomes one entry, in primId order (stable).
// - Area A from the RECENTRED f32 positions (the ones the GPU samples), in f64. It is stored as f32 in the record and
//   used for BOTH q = P/A at NEE time and p1 at BSDF-hit time, so the MIS partition is exact.
// - Power proxy Φ̃ = lum(avg L_e)·A·π·2 (two-sided). Textured emission uses the mean of the decoded (sRGB → linear)
//   texture over the WHOLE image (a variance-only choice: any positive proxy is unbiased as long as the same realized
//   pmf is used for sampling and for every pdf evaluation).
import type { MaterialData, SceneData, TextureData } from '../scene/types.ts';
import { TRI_EMISSIVE } from '../scene/types.ts';

export interface EmissiveTriangles {
  /** primId per emissive entry (ascending). */
  primIds: Uint32Array;
  /** Triangle area (m²) per entry, f32 as uploaded. */
  areas: Float32Array;
  /** Average emitted radiance (rgb) per entry, used only for the power proxy. */
  avgRadiance: Float32Array;
  /** Power proxy Φ̃ per entry (f64). */
  power: Float64Array;
  /** primId → entry index, or 0xffffffff (length = triangle count). */
  primToEntry: Uint32Array;
}

export const NO_ENTRY = 0xffffffff;

const lum = (r: number, g: number, b: number): number => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/** Exact sRGB EOTF (Cycles util/color.h color_srgb_to_linear). */
export function srgbToLinear(c: number): number {
  return c < 0.04045 ? (c < 0 ? 0 : c * (1 / 12.92)) : ((c + 0.055) / 1.055) ** 2.4;
}

/** Mean decoded RGB of an sRGB texture (1,1,1 when the pixels were not decoded, e.g. Node without an image decoder). */
export function meanTextureRgb(t: TextureData | undefined): [number, number, number] {
  if (!t || t.pixels.length < 4) return [1, 1, 1];
  const lut = new Float64Array(256);
  for (let i = 0; i < 256; i++) lut[i] = srgbToLinear(i / 255);
  let r = 0, g = 0, b = 0;
  const n = t.pixels.length >> 2;
  for (let i = 0; i < n; i++) { r += lut[t.pixels[4 * i]]; g += lut[t.pixels[4 * i + 1]]; b += lut[t.pixels[4 * i + 2]]; }
  return [r / n, g / n, b / n];
}

/** Emission radiance (rgb, before texture) of a material: emissiveFactor · emissiveStrength. */
export function materialEmission(m: MaterialData): [number, number, number] {
  return [m.emissiveFactor[0] * m.emissiveStrength, m.emissiveFactor[1] * m.emissiveStrength, m.emissiveFactor[2] * m.emissiveStrength];
}

/** True when the material opts out of NEE (Cycles emission_sampling NONE). Optional field, absent in v1 scenes. */
export function emissionSamplingNone(m: MaterialData): boolean {
  return (m as MaterialData & { emissionSampling?: string }).emissionSampling === 'NONE';
}

/** Collect the emissive-triangle entries. `positions` = the recentred f32 positions uploaded to the GPU. */
export function collectEmissiveTriangles(scene: SceneData, positions: Float32Array = scene.geometry.positions): EmissiveTriangles {
  const g = scene.geometry;
  const nTris = g.indices.length / 3;
  const texMean = new Map<number, [number, number, number]>();
  const matAvg = scene.materials.map((m) => {
    const e = materialEmission(m);
    if (m.emissiveTexture) {
      const ti = m.emissiveTexture.texture;
      if (!texMean.has(ti)) texMean.set(ti, meanTextureRgb(scene.textures[ti]));
      const t = texMean.get(ti)!;
      return [e[0] * t[0], e[1] * t[1], e[2] * t[2]] as [number, number, number];
    }
    return e;
  });
  const prims: number[] = [];
  for (let t = 0; t < nTris; t++) {
    if ((g.triFlags[t] & TRI_EMISSIVE) === 0) continue;
    const m = scene.materials[g.triMaterial[t]];
    if (!m || emissionSamplingNone(m)) continue;
    prims.push(t);
  }
  const n = prims.length;
  const primIds = Uint32Array.from(prims);
  const areas = new Float32Array(n);
  const avgRadiance = new Float32Array(3 * n);
  const power = new Float64Array(n);
  const primToEntry = new Uint32Array(nTris).fill(NO_ENTRY);
  for (let i = 0; i < n; i++) {
    const t = primIds[i];
    const a = 3 * g.indices[3 * t], b = 3 * g.indices[3 * t + 1], c = 3 * g.indices[3 * t + 2];
    const e1 = [positions[b] - positions[a], positions[b + 1] - positions[a + 1], positions[b + 2] - positions[a + 2]];
    const e2 = [positions[c] - positions[a], positions[c + 1] - positions[a + 1], positions[c + 2] - positions[a + 2]];
    const cx = e1[1] * e2[2] - e1[2] * e2[1], cy = e1[2] * e2[0] - e1[0] * e2[2], cz = e1[0] * e2[1] - e1[1] * e2[0];
    const A = 0.5 * Math.hypot(cx, cy, cz);
    areas[i] = A;
    const L = matAvg[g.triMaterial[t]];
    avgRadiance.set(L, 3 * i);
    power[i] = Math.max(0, lum(L[0], L[1], L[2])) * Math.fround(A) * Math.PI * 2;
    primToEntry[t] = i;
  }
  return { primIds, areas, avgRadiance, power, primToEntry };
}
