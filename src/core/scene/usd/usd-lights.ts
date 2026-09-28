// UsdLux → LightData (plan §1.2 "USD lights, v1 rules"; math.md#units-lights; docs/decisions/usd.md adapter rule 3)
// and the USD stage → glTF-canonical frame transform. Pure functions (unit-tested in tests/scene/usd-lights.test.ts).
//
// Units: LightData.power is Blender power without the exposure (Φ = color ⊙ power · 2^exposure; exposure kept as
// authored). With i = inputs:intensity:
//   Rect/Disk  normalize true : L_e = i·2^e/A  ⇒ power = π·i            normalize false: L_e = i·2^e ⇒ power = π·i·A
//   Sphere     normalize true : I = i·2^e/4    ⇒ power = π·i            (r := 0, radiant intensity preserved)
//              false, r > 0   : I = i·2^e·πr² ⇒ power = 4π²r²·i
//              false, r = 0 or treatAsPoint: I = i·2^e ⇒ power = 4π·i
//   Sphere+ShapingAPI → spot: spot_size = 2·coneAngle, spot_blend = coneSoftness, same I rule.
//   Distant    E = i·2^e, ×4 for Blender-authored files (Blender exports intensity = energy/4); angle := 0.
// Sizes and radii are in metres: authored size × the light's world scale × metersPerUnit (A = sizeX·sizeY for rect,
// π/4·sizeX·sizeY for disk). Colour temperature (enableColorTemperature) multiplies the colour by Blender's blackbody
// colour (Blender 5.1 Light.temperature_color, tests/scene/blender-blackbody.json): Blender's own USD importer
// honours it, and Blender writes enableColorTemperature = 1 on every light (usd.md Blender quirk 1).
import { rigidMatrix } from '../gltf-loader.ts';
import type { LightData } from '../types.ts';

export type Mat4 = number[]; // column-major 4x4 (USD row-vector matrices have the same memory layout)

export interface UsdLightInput {
  primPath: string;
  type: 'sphere' | 'rect' | 'disk' | 'distant' | string;
  intensity: number;
  exposure: number;
  color: [number, number, number];
  normalize: boolean;
  enableColorTemperature: boolean;
  colorTemperature: number;
  radius?: number;             // sphere/disk (authored units)
  width?: number;              // rect
  height?: number;             // rect
  angle?: number;              // distant: angular diameter, degrees
  treatAsPoint?: boolean;
  shaping?: { coneAngle: number; coneSoftness: number; focus: number } | null; // coneAngle in degrees
}

export interface LightConvertOptions {
  /** Distant-light ×4 quirk: 'auto' = only for Blender-authored stages (root-layer doc "Blender v…"). */
  blenderAuthored: boolean;
}

/** Blender's blackbody colour (intern/cycles blackbody fit, rec709 = scene linear, clamped ≥ 0). */
export function blenderBlackbody(t: number): [number, number, number] {
  if (t >= 12000) return [0.8262954810464208, 0.9945080501520986, 1.566307710274283];
  if (t < 800) return [5.413294490189271, 0, 0]; // (−0.203, −0.082) clamped to 0
  const i = t >= 6365 ? 6 : t >= 3315 ? 5 : t >= 1902 ? 4 : t >= 1449 ? 3 : t >= 1167 ? 2 : t >= 965 ? 1 : 0;
  const r = BB_R[i], g = BB_G[i], b = BB_B[i];
  const ti = 1 / t;
  const f = Math.fround;
  return [
    f(Math.max(0, r[0] * ti + r[1] * t + r[2])),
    f(Math.max(0, g[0] * ti + g[1] * t + g[2])),
    f(Math.max(0, ((b[0] * t + b[1]) * t + b[2]) * t + b[3])),
  ];
}
const BB_R = [
  [1.61919106e+03, -2.05010916e-03, 5.02995757e+00], [2.48845471e+03, -1.11330907e-03, 3.22621544e+00],
  [3.34143193e+03, -4.86551192e-04, 1.76486769e+00], [4.09461742e+03, -1.27446582e-04, 7.25731635e-01],
  [4.67028036e+03, 2.91258199e-05, 1.26703442e-01], [4.59509185e+03, 2.87495649e-05, 1.50345020e-01],
  [3.78717450e+03, 9.35907826e-06, 3.99075871e-01],
];
const BB_G = [
  [-4.88999748e+02, 6.04330754e-04, -7.55807526e-02], [-7.55994277e+02, 3.16730098e-04, 4.78306139e-01],
  [-1.02363977e+03, 1.20223470e-04, 9.36662319e-01], [-1.26571316e+03, 4.87340896e-06, 1.27054498e+00],
  [-1.42529332e+03, -4.01150431e-05, 1.43972784e+00], [-1.17554822e+03, -2.16378048e-05, 1.30408023e+00],
  [-5.00799571e+02, -4.59832026e-06, 1.09098763e+00],
];
const BB_B = [
  [5.96945309e-11, -4.85742887e-08, -9.70622247e-05, -4.07936148e-03], [2.40430366e-11, 5.55021075e-08, -1.98503712e-04, 2.89312858e-02],
  [-1.40949732e-11, 1.89878968e-07, -3.56632824e-04, 9.10767778e-02], [-3.61460868e-11, 2.84822009e-07, -4.93211319e-04, 1.56723440e-01],
  [-1.97075738e-11, 1.75359352e-07, -2.50542825e-04, -2.22783266e-02], [-1.61997957e-13, -1.64216008e-08, 3.86216271e-04, -7.38077418e-01],
  [6.72650283e-13, -2.73078809e-08, 4.24098264e-04, -7.52335691e-01],
];

/** Stage → glTF canonical: W = metersPerUnit · (upAxis Z ? R_x(−90°) : I). Column-major. */
export function stageMatrix(upAxis: string | null | undefined, metersPerUnit: number | null | undefined): Mat4 {
  const s = metersPerUnit && metersPerUnit > 0 && Number.isFinite(metersPerUnit) ? metersPerUnit : 1;
  // R_x(−90°): (x, y, z) → (x, z, −y); columns are the images of the basis vectors.
  if ((upAxis ?? 'Y').toUpperCase() === 'Z') return [s, 0, 0, 0, 0, 0, -s, 0, 0, s, 0, 0, 0, 0, 0, 1];
  return [s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1];
}

/** Column-major a·b. */
export function mul4(a: ArrayLike<number>, b: ArrayLike<number>): Mat4 {
  const r = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c++) for (let row = 0; row < 4; row++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + row] * b[c * 4 + k];
    r[c * 4 + row] = s;
  }
  return r;
}

const colLen = (m: ArrayLike<number>, c: number) => Math.hypot(m[c * 4], m[c * 4 + 1], m[c * 4 + 2]);

export interface LightConvertResult { light: LightData | null; warnings: string[] }

/**
 * One UsdLux light (fields as normalized from LightUSD + the layer scan) with its world matrix ALREADY in the glTF
 * canonical frame (stageMatrix · usdWorld) → LightData. Returns null (with a warning) for unsupported types.
 */
export function convertUsdLight(l: UsdLightInput, world: Mat4, id: number, opts: LightConvertOptions): LightConvertResult {
  const warnings: string[] = [];
  const name = l.primPath.split('/').pop() || `light${id}`;
  const i = l.intensity;
  let color: [number, number, number] = [...l.color];
  if (l.enableColorTemperature) {
    const bb = blenderBlackbody(l.colorTemperature || 6500);
    color = [color[0] * bb[0], color[1] * bb[1], color[2] * bb[2]];
  }
  const sx = colLen(world, 0), sy = colLen(world, 1), sz = colLen(world, 2);
  const base = { id, name, color, exposure: l.exposure || 0, matrix: rigidMatrix(world), visibleToCamera: false };
  const tag = `light ${l.primPath}`;
  switch (l.type) {
    case 'rect': {
      const sizeX = (l.width ?? 1) * sx, sizeY = (l.height ?? 1) * sy;
      const power = l.normalize ? Math.PI * i : Math.PI * i * sizeX * sizeY;
      return { light: { ...base, type: 'rect', power, sizeX, sizeY, spread: Math.PI }, warnings };
    }
    case 'disk': {
      const r = l.radius ?? 0.5;
      const sizeX = 2 * r * sx;
      let sizeY = 2 * r * sy;
      if (Math.abs(sizeY - sizeX) <= 1e-6 * sizeX) sizeY = sizeX; // circular (f32 scale noise)
      else warnings.push(`${tag}: non-uniformly scaled DiskLight (ellipse ${sizeX}×${sizeY}); the scene bridge rejects elliptical disks`);
      const power = l.normalize ? Math.PI * i : Math.PI * i * (Math.PI / 4) * sizeX * sizeY;
      const disk: LightData = { ...base, type: 'disk', power, sizeX, spread: Math.PI };
      if (sizeY !== sizeX) disk.sizeY = sizeY;
      return { light: disk, warnings };
    }
    case 'sphere': {
      if (Math.abs(sx - sy) > 1e-6 * sx || Math.abs(sx - sz) > 1e-6 * sx) warnings.push(`${tag}: non-uniform scale; radius uses the X scale`);
      const r = (l.radius ?? 0.5) * sx;
      const point = !!l.treatAsPoint || !(r > 0);
      const power = l.normalize ? Math.PI * i : point ? 4 * Math.PI * i : 4 * Math.PI * Math.PI * r * r * i;
      const simplified = !point ? `sphere radius ${r.toPrecision(4)} m → point (r = 0, radiant intensity preserved)` : undefined;
      if (l.shaping) {
        const cone = Math.min(180, Math.max(1, 2 * l.shaping.coneAngle)) * Math.PI / 180;
        if (l.shaping.focus) warnings.push(`${tag}: shaping:focus ${l.shaping.focus} ignored (Blender has no equivalent)`);
        const spot: LightData = { ...base, type: 'spot', power, spotSize: cone, spotBlend: Math.min(1, Math.max(0, l.shaping.coneSoftness || 0)) };
        if (simplified) spot.simplified = simplified;
        return { light: spot, warnings };
      }
      const pt: LightData = { ...base, type: 'point', power };
      if (simplified) pt.simplified = simplified;
      return { light: pt, warnings };
    }
    case 'distant': {
      const power = (opts.blenderAuthored ? 4 : 1) * i;
      const sun: LightData = { ...base, type: 'sun', power };
      if ((l.angle ?? 0) > 0) sun.simplified = `sun angle ${l.angle}° → 0`;
      return { light: sun, warnings };
    }
    default:
      warnings.push(`${tag}: UsdLux ${l.type} light not supported in v1; skipped`);
      return { light: null, warnings };
  }
}
