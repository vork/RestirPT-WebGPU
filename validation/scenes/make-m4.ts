// M4 T3 / case-coverage fixture scenes (restir-api.md §6.2, WP-B): the shift-invertibility fixtures of Gate 0.
//   npx tsx validation/scenes/make-m4.ts [outDir] [--only name,name]
// Default outDir: validation/out/m4/scenes (gitignored: the env variants embed the downloaded Poly Haven texels,
// validation/assets/fetch_hdris.ts). Deterministic: re-running writes byte-identical packages (Gate 0 checks it).
//
// The scene builders below are BROWSER-SAFE (no Node imports at module level): the Chrome GPU test
// (validation/gpu-tests/restir-shift.gpu.test.ts) imports t3Scene() directly and passes the HDRI it fetched; only the
// CLI (main) loads Node modules, dynamically.
//
// t3_cases_256 (Mode A, flat shaded, b = 4): an open-front Cornell box with Lambert walls and a hole in the ceiling (the
// env reaches the interior), V1 GGX spheres r ∈ {0.1, 0.19, 0.21, 0.3, 0.5} (never r = 0.2 exactly, math.md#rc-predicate),
// a V1 constant-mix plastic (Lambert + GGX r 0.3), V2 Principled metal r 0.25 and dielectric r 0.4, a roughness-0 mirror
// (delta S), a two-sided emissive triangle mesh, point, spot, rect, disk and sun lights and studio_small_09 (P(env)
// clamped by the analytic lights). Variants: _noenv, _envonly (env only; env NEE on and off at run time), _b2 (b = 2,
// d ≤ 4: dense PSS sweep), t3_glass_256 (+ smooth glass sphere and rough glass r 0.3 sphere: reported only, D13).
// WP-B additions (restir-api.md Changelog B-3): t3_rare_256 = t3_cases_256 with glossy r 0.1 floor / back wall (pair 2
// fails R, so k > 2, ∅-TRI and ∅-ENV paths are frequent) and a large emissive panel; t3_cutoff_256 (U-12) = the
// spheres replaced by V2 / V1 materials whose lobe weights sit at the closure cutoff 1e-5 (metallic 1 − 1.5e-5,
// 1 − 0.8e-5, mix 1 − 1.2e-5, specular level ~1e-5).
import type { EnvironmentData, LightData, MaterialData, SceneData, SceneGeometry } from '../../src/core/scene/types.ts';
import { TRI_EMISSIVE } from '../../src/core/scene/types.ts';
import { quantizeScene } from '../../src/core/scene/quantize.ts';

export type V3 = [number, number, number];
export type T3Variant = 't3_cases_256' | 't3_cases_256_noenv' | 't3_cases_256_envonly' | 't3_cases_256_b2' | 't3_glass_256' | 't3_rare_256' | 't3_cutoff_256';
export const T3_VARIANTS: T3Variant[] = ['t3_cases_256', 't3_cases_256_noenv', 't3_cases_256_envonly', 't3_cases_256_b2', 't3_glass_256', 't3_rare_256', 't3_cutoff_256'];
export const T3_ENV_ID = 'studio_small_09';

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scl = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const nrm = (a: V3): V3 => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };

// ---- materials (the same field conventions as validation/scenes/scene-kit.ts) ------------------------------------

const BASE: Omit<MaterialData, 'name' | 'model'> = {
  baseColorFactor: [0.8, 0.8, 0.8, 1], metallicFactor: 0, roughnessFactor: 0.5, emissiveFactor: [0, 0, 0], emissiveStrength: 1,
  ior: 1.5, specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true,
};
function v1(name: string, o: { diffuse?: V3; glossy?: V3; roughness?: number; mix?: number; emission?: V3 } = {}): MaterialData {
  const diffuse = o.diffuse ?? [0.8, 0.8, 0.8];
  return { ...BASE, name, model: 'v1', baseColorFactor: [...diffuse, 1], emissiveFactor: o.emission ?? [0, 0, 0],
    v1: { diffuse, glossy: o.glossy ?? [0, 0, 0], roughness: o.roughness ?? 0.5, mix: o.mix ?? 0 } };
}
function principled(name: string, o: Partial<MaterialData>): MaterialData { return { ...BASE, ...o, name, model: 'principled' }; }

// ---- flat-shaded mesh builder (browser-safe twin of scene-kit's MeshBuilder: quads and icospheres) ----------------

class Mesh {
  pos: number[] = []; nrm: number[] = []; uv: number[] = []; idx: number[] = []; mat: number[] = [];
  tri(a: V3, b: V3, c: V3, m: number): this {
    const n = nrm(cross(sub(b, a), sub(c, a)));
    const base = this.pos.length / 3;
    for (const p of [a, b, c]) { this.pos.push(...p); this.nrm.push(...n); this.uv.push(0, 0); }
    this.idx.push(base, base + 1, base + 2);
    this.mat.push(m);
    return this;
  }
  /** Quad with its normal oriented toward `inside` (walls of an enclosure face the interior). */
  quad(p0: V3, p1: V3, p2: V3, p3: V3, m: number, inside?: V3): this {
    let q = [p0, p1, p2, p3];
    if (inside && dot(cross(sub(p1, p0), sub(p3, p0)), sub(inside, p0)) < 0) q = [p0, p3, p2, p1];
    return this.tri(q[0], q[1], q[2], m).tri(q[0], q[2], q[3], m);
  }
  icosphere(c: V3, r: number, subdiv: number, m: number): this {
    const t = (1 + Math.sqrt(5)) / 2;
    const verts: V3[] = ([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]] as V3[]).map(nrm);
    let faces: [number, number, number][] = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
    for (let s = 0; s < subdiv; s++) {
      const cache = new Map<string, number>();
      const mid = (a: number, b: number) => {
        const key = a < b ? `${a},${b}` : `${b},${a}`;
        let i = cache.get(key);
        if (i === undefined) { i = verts.length; verts.push(nrm(scl(add(verts[a], verts[b]), 0.5))); cache.set(key, i); }
        return i;
      };
      const next: [number, number, number][] = [];
      for (const [a, b, cc] of faces) { const ab = mid(a, b), bc = mid(b, cc), ca = mid(cc, a); next.push([a, ab, ca], [b, bc, ab], [cc, ca, bc], [ab, bc, ca]); }
      faces = next;
    }
    for (const [a, b, cc] of faces) this.tri(add(c, scl(verts[a], r)), add(c, scl(verts[b], r)), add(c, scl(verts[cc], r)), m);
    return this;
  }
  build(materials: MaterialData[]): SceneGeometry {
    const flags = this.mat.map((m) => (Math.max(...materials[m].emissiveFactor) * materials[m].emissiveStrength > 0 ? TRI_EMISSIVE : 0));
    const nV = this.pos.length / 3;
    return { positions: Float32Array.from(this.pos), normals: Float32Array.from(this.nrm), tangents: new Float32Array(nV * 4),
      uv0: Float32Array.from(this.uv), indices: Uint32Array.from(this.idx), triMaterial: Uint32Array.from(this.mat), triFlags: Uint32Array.from(flags) };
  }
}

/** Column-major rigid frame from unit axes and a position. */
const frameM = (X: V3, Y: V3, Z: V3, p: V3): number[] => [...X, 0, ...Y, 0, ...Z, 0, ...p, 1];
/** Camera-to-world looking from eye at target (camera looks down local −Z). */
export function lookAt(eye: V3, target: V3, up: V3 = [0, 1, 0]): number[] {
  const z = nrm(sub(eye, target));
  const x = nrm(cross(up, z));
  return frameM(x, cross(z, x), z, eye);
}
/** Light matrix at p whose emission axis −Z_obj points along dir. */
function lightToward(dir: V3, p: V3): Float32Array {
  const Z = nrm(scl(dir, -1));
  const helper: V3 = Math.abs(Z[1]) < 0.95 ? [0, 1, 0] : [1, 0, 0];
  const X = nrm(cross(helper, Z));
  return new Float32Array(frameM(X, cross(Z, X), Z, p));
}
function light(id: number, type: LightData['type'], matrix: Float32Array, power: number, o: Partial<LightData> = {}): LightData {
  return { id, name: `${type}${id}`, type, color: [1, 1, 1], power, exposure: 0, matrix, visibleToCamera: false, ...o };
}

export interface T3Scene { scene: SceneData; camera: { matrix: number[]; yfov: number }; maxBounces: number; width: number; height: number; notes: string }

/** Material indices of t3_cases (the dual and the tests look materials up by index). */
export const T3_MAT = {
  floor: 0, back: 1, left: 2, right: 3, ceiling: 4, ggx010: 5, ggx019: 6, ggx021: 7, ggx030: 8, ggx050: 9, plastic: 10, metal: 11,
  dielectric: 12, mirror: 13, emissive: 14, glass: 15, roughGlass: 16,
} as const;

/**
 * Build a T3 fixture. `env` = the studio_small_09 1k map (required by the env variants; the test passes the map it
 * fetched, the CLI decodes the downloaded file).
 */
export function t3Scene(variant: T3Variant, env?: EnvironmentData): T3Scene {
  const withEnv = variant !== 't3_cases_256_noenv';
  const lightsOn = variant !== 't3_cases_256_envonly';
  const glass = variant === 't3_glass_256';
  if (withEnv && !env) throw new Error(`${variant}: the ${T3_ENV_ID} map is required`);
  const materials: MaterialData[] = [
    v1('floor', { diffuse: [0.7, 0.7, 0.7] }), v1('back', { diffuse: [0.6, 0.6, 0.5] }), v1('left', { diffuse: [0.6, 0.15, 0.1] }),
    v1('right', { diffuse: [0.15, 0.5, 0.15] }), v1('ceiling', { diffuse: [0.7, 0.7, 0.7] }),
    ...[0.1, 0.19, 0.21, 0.3, 0.5].map((r) => v1(`ggx_r${r}`, { diffuse: [0, 0, 0], glossy: [0.9, 0.9, 0.9], roughness: r, mix: 1 })),
    v1('plastic', { diffuse: [0.2, 0.3, 0.7], glossy: [0.8, 0.8, 0.8], roughness: 0.3, mix: 0.5 }),
    principled('metal', { baseColorFactor: [1, 0.78, 0.34, 1], metallicFactor: 1, roughnessFactor: 0.25 }),
    principled('dielectric', { baseColorFactor: [0.8, 0.2, 0.2, 1], metallicFactor: 0, roughnessFactor: 0.4 }),
    v1('mirror', { diffuse: [0, 0, 0], glossy: [0.95, 0.95, 0.95], roughness: 0, mix: 1 }),
    v1('emissive', { diffuse: [0, 0, 0], emission: [6, 5, 4] }),
  ];
  if (variant === 't3_rare_256') {
    materials[T3_MAT.floor] = v1('floor_glossy', { diffuse: [0.1, 0.1, 0.1], glossy: [0.8, 0.8, 0.8], roughness: 0.1, mix: 0.9 });
    materials[T3_MAT.back] = v1('back_glossy', { diffuse: [0, 0, 0], glossy: [0.85, 0.85, 0.85], roughness: 0.12, mix: 1 });
  }
  if (variant === 't3_cutoff_256') {
    materials[T3_MAT.ggx010] = principled('m_cut_a', { baseColorFactor: [0.9, 0.6, 0.3, 1], metallicFactor: 1 - 1.5e-5, roughnessFactor: 0.3 });
    materials[T3_MAT.ggx019] = principled('m_cut_b', { baseColorFactor: [0.9, 0.6, 0.3, 1], metallicFactor: 1 - 0.8e-5, roughnessFactor: 0.3 });
    materials[T3_MAT.ggx021] = v1('mix_cut', { diffuse: [0.8, 0.8, 0.8], glossy: [0.9, 0.9, 0.9], roughness: 0.3, mix: 1 - 1.2e-5 });
    materials[T3_MAT.ggx030] = principled('spec_cut', { baseColorFactor: [0.7, 0.7, 0.7, 1], metallicFactor: 0, roughnessFactor: 0.35, specularFactor: 2e-5 });
    materials[T3_MAT.ggx050] = principled('diff_cut', { baseColorFactor: [1.2e-5, 1.2e-5, 1.2e-5, 1], metallicFactor: 0, roughnessFactor: 0.45 });
  }
  if (glass) {
    materials.push(principled('glass', { baseColorFactor: [1, 1, 1, 1], roughnessFactor: 0, transmissionFactor: 1, ior: 1.5 }));
    materials.push(principled('rough_glass', { baseColorFactor: [1, 1, 1, 1], roughnessFactor: 0.3, transmissionFactor: 1, ior: 1.5 }));
  }
  const M = T3_MAT;
  const mb = new Mesh();
  const X0 = -1.5, X1 = 1.5, Y1 = 2, Z0 = -1.5, Z1 = 1.0;
  const inside: V3 = [0, 1, -0.25];
  mb.quad([X0, 0, Z1], [X1, 0, Z1], [X1, 0, Z0], [X0, 0, Z0], M.floor, inside);
  mb.quad([X0, 0, Z0], [X1, 0, Z0], [X1, Y1, Z0], [X0, Y1, Z0], M.back, inside);
  mb.quad([X0, 0, Z1], [X0, 0, Z0], [X0, Y1, Z0], [X0, Y1, Z1], M.left, inside);
  mb.quad([X1, 0, Z0], [X1, 0, Z1], [X1, Y1, Z1], [X1, Y1, Z0], M.right, inside);
  // ceiling with a hole z ∈ [−0.6, 0.2] (the env and the sun reach the interior through it)
  mb.quad([X0, Y1, Z0], [X1, Y1, Z0], [X1, Y1, -0.6], [X0, Y1, -0.6], M.ceiling, inside);
  mb.quad([X0, Y1, 0.2], [X1, Y1, 0.2], [X1, Y1, Z1], [X0, Y1, Z1], M.ceiling, inside);
  // GGX spheres along the back, plastic / metal / dielectric in front
  [M.ggx010, M.ggx019, M.ggx021, M.ggx030, M.ggx050].forEach((m, i) => mb.icosphere([-1.0 + 0.5 * i, 0.22, -1.0], 0.2, 2, m));
  mb.icosphere([-0.8, 0.25, -0.2], 0.25, 2, M.plastic);
  mb.icosphere([0.0, 0.25, -0.1], 0.25, 2, M.metal);
  mb.icosphere([0.85, 0.3, -0.3], 0.3, 2, M.dielectric);
  // mirror on the left wall (delta S), a two-sided emissive mesh hanging in the right back corner
  mb.quad([X0 + 0.02, 0.3, -0.4], [X0 + 0.02, 0.3, -1.2], [X0 + 0.02, 1.3, -1.2], [X0 + 0.02, 1.3, -0.4], M.mirror, inside);
  mb.icosphere([0.95, 1.45, -1.05], 0.13, 0, M.emissive);
  if (variant === 't3_rare_256') mb.quad([1.45, 0.3, 0.6], [1.45, 0.3, -0.2], [1.45, 1.2, -0.2], [1.45, 1.2, 0.6], M.emissive, inside);
  if (glass) {
    mb.icosphere([-0.35, 0.8, 0.3], 0.18, 2, M.glass);
    mb.icosphere([0.45, 0.8, 0.35], 0.18, 2, M.roughGlass);
  }
  const lights: LightData[] = lightsOn ? [
    light(1, 'point', lightToward([0, -1, 0], [-0.6, 1.7, 0.3]), 40),
    light(2, 'spot', lightToward(nrm([-0.5, -1.8, -1.0]), [0.8, 1.8, 0.5]), 60, { spotSize: 1.2, spotBlend: 0.25 }),
    light(3, 'rect', lightToward([0, -1, 0], [-0.8, 1.99, -1.0]), 30, { sizeX: 0.5, sizeY: 0.3 }),
    light(4, 'disk', lightToward(nrm([0, -0.3, 1]), [0.9, 1.2, -1.45]), 20, { sizeX: 0.3 }),
    light(5, 'sun', lightToward(nrm([0.3, -1, -0.2]), [0, 5, 0]), 3),
  ] : [];
  const geometry = mb.build(materials);
  const mn: V3 = [Infinity, Infinity, Infinity], mx: V3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < geometry.positions.length; i += 3) for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], geometry.positions[i + k]); mx[k] = Math.max(mx[k], geometry.positions[i + k]); }
  // package v2 / GPU: the one lossy step (data-formats.md §B0), before the env is attached
  const scene: SceneData = quantizeScene({ name: variant, geometry, materials, textures: [], lights, cameras: [], bounds: { min: mn, max: mx }, warnings: [] }).scene;
  if (withEnv) scene.env = { ...env!, name: `${T3_ENV_ID}_1k`, strength: 1, tint: [1, 1, 1], rotationZ: 0.6, visibleToCamera: true };
  return {
    scene, camera: { matrix: lookAt([0, 1.1, 3.6], [0, 0.8, -0.4]), yfov: 50 * Math.PI / 180 },
    maxBounces: variant === 't3_cases_256_b2' ? 2 : 4, width: 256, height: 256,
    notes: `${variant}: T3 case-coverage fixture (restir-api.md §6.2)${glass ? '; glass: reported only (D13)' : ''}`,
  };
}

// ---- CLI --------------------------------------------------------------------------------------------------------------

async function main(): Promise<void> {
  const [{ writePackage, ROOT }, path, fs, { decodeHdr }] = await Promise.all([
    import('./scene-kit.ts'), import('node:path'), import('node:fs'), import('../../src/core/scene/env/hdr.ts'),
  ]);
  const argv = process.argv.slice(2);
  const i = argv.indexOf('--only');
  const only = i >= 0 ? new Set(argv[i + 1].split(',')) : undefined;
  const out = argv[0] && !argv[0].startsWith('--') ? path.resolve(argv[0]) : path.join(ROOT, 'validation/out/m4/scenes');
  const file = path.join(ROOT, `validation/assets/downloaded/hdri/${T3_ENV_ID}_1k.hdr`);
  let env: EnvironmentData | undefined;
  try {
    const img = decodeHdr(new Uint8Array(fs.readFileSync(file)));
    env = { name: `${T3_ENV_ID}_1k`, width: img.width, height: img.height, texels: img.texels, strength: 1, tint: [1, 1, 1], rotationZ: 0, visibleToCamera: true };
  } catch { throw new Error(`${file} missing: run npx tsx validation/assets/fetch_hdris.ts`); }
  fs.mkdirSync(out, { recursive: true });
  for (const v of T3_VARIANTS) {
    if (only && !only.has(v)) continue;
    const t = t3Scene(v, env);
    const dir = await writePackage(out, t.scene, {
      name: v, camera: { matrix: t.camera.matrix, yfov: t.camera.yfov }, render: { width: t.width, height: t.height, maxBounces: t.maxBounces },
      source: { uri: `validation/scenes/make-m4.ts (${v}) + ${T3_ENV_ID} 1k` }, extra: { notes: t.notes, tier: v === 't3_glass_256' ? 'reported' : 'gate0' },
    });
    console.log(`wrote ${path.relative(ROOT, dir)}`);
  }
}

if (typeof process !== 'undefined' && process.argv?.[1]?.endsWith('make-m4.ts')) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
