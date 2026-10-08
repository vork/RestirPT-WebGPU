// M7 scene packages (docs/decisions/m7-api.md §4; PLAN §5 M7, §7.2 (vii-N), (xiv)-lite E2E-HDR):
//   npx tsx validation/scenes/make-m7.ts [outDir] [--only name,name]
// Default outDir: validation/out/m7/scenes (gitignored: generated, deterministic — the M7 gate checks byte identity; the
// env scenes embed the downloaded Poly Haven texels, validation/assets/fetch_hdris.ts).
//
// Room (all but the env scenes): 1 m box (x, z ∈ [−0.5, 0.5], y ∈ [0, 1]) open toward the camera, Principled walls (flat),
// a 0.3 × 0.3 m 12 W ceiling rect light; 256², b = 3; Mode A unless noted. Smooth meshes have shared vertices with
// analytic normals (m7-kit.ts), so quantizeScene leaves them smooth (no TRI_FLAT) and the packages are exported smooth
// (Blender: bit-exact custom normals). Normal maps are procedural RGBA8 PNGs (Non-Color) on Principled materials; Blender
// recomputes MikkTSpace on the package mesh, ours does the same on read (tangents.ts).
//
//   m7_smooth_256        smooth shading, no normal map: a Principled dielectric sphere, a gold metal sphere, a smooth torus
//                        and an OPEN wavy sheet (one-sided: Ng·L ≤ 0 < Ns·L is not self-occluded there); rect + point.
//   m7_nm_flat_256       flat geometry, normal maps (vii-N tight tier): tiles floor (s 1), bumps back wall (s 0.6), waves
//                        left wall (s 1.7), a bumps box (s 1); three panels on the back wall for the Stage-A plants: P1 tilt
//                        35° toward +B (s 1), P2 the same with mirrored u (MikkTSpace w = −1), P3 tilt 60° toward −B at s 0.5;
//                        rect + a spot on the panels.
//   m7_nm_smooth_256     smooth + normal-mapped (vii-N model-approximate): bumps sphere (diffuse), waves gold sphere
//                        (r 0.3), tiles torus, bumps open sheet; rect + point.
//   m7_nm_smooth_B_256   m7_nm_smooth_256 in light mode B (Cycles per-light MIS on).
//   m7_nm_env_256        smooth normal-mapped spheres (diffuse / glossy dielectric / metal) on a tiles ground under
//                        overcast_soil_puresky (γ 0.9), b = 3 (env NEE + BSDF_ENV with mapped normals).
//   m7_xivlite_hdr_256, m7_xivlite_exr_256   E2E-HDR (xiv)-lite: the Cornell box without its ceiling under the ORIGINAL
//                        overcast_soil_puresky_1k .hdr / .exr, b = 2; scene.json env.original names the downloaded file, which
//                        build_scene.py loads directly (Blender's own decoder) instead of env.exr; our side decodes the same
//                        file (load-env.ts).
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { exportScenePackage } from '../../src/core/scene/scene-package.ts';
import { decodeEnvironment } from '../../src/core/scene/env/load-env.ts';
import type { EnvironmentData, LightData, MaterialData, SceneData } from '../../src/core/scene/types.ts';
import { MeshBuilder, ROOT, cornellBase, deg, light, lightToward, lookAt, principled, resetLightIds, sceneOf, type V3 } from './scene-kit.ts';
import { bumpsMap, smoothSheet, smoothSphere, smoothTorus, tilesMap, tiltMap, wavesMap } from './m7-kit.ts';

const argv = process.argv.slice(2);
const flag = (k: string) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : undefined; };
const ONLY = flag('--only') ? new Set(flag('--only')!.split(',')) : undefined;
export const M7_OUT = 'validation/out/m7/scenes';
const OUT = argv[0] && !argv[0].startsWith('--') ? path.resolve(argv[0]) : path.join(ROOT, M7_OUT);
const HDRI_DIR = 'validation/assets/downloaded/hdri';

export const M7_SCENES = ['m7_smooth_256', 'm7_smooth_lowpoly_256', 'm7_nm_flat_256', 'm7_nm_smooth_256', 'm7_nm_smooth_B_256', 'm7_nm_env_256', 'm7_xivlite_hdr_256', 'm7_xivlite_exr_256'] as const;

/** Plant regions of m7_nm_flat_256 (pixel rectangles in the 256² image, from the panel corners; m7-api.md §5.1). */
export const M7_PANELS = {
  P1: { x: [-0.42, -0.16] as [number, number], y: [0.56, 0.86] as [number, number] },
  P2: { x: [-0.13, 0.13] as [number, number], y: [0.56, 0.86] as [number, number] },
  P3: { x: [0.16, 0.42] as [number, number], y: [0.56, 0.86] as [number, number] },
};
export const ROOM_CAMERA = { matrix: lookAt([0, 0.5, 1.9], [0, 0.5, 0]), yfov: 39 * deg };

// ---- room -------------------------------------------------------------------------------------------------------------

const RECT_SIZE = 0.3;
const rectLight = (): LightData => light('rect', new Float32Array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0.995, 0, 1]), 12, { sizeX: RECT_SIZE, sizeY: RECT_SIZE });

interface RoomMats { floor: number; back: number; left: number; right: number; ceiling: number }
/** Inward room faces (no front): floor, ceiling, back, left (−x), right (+x). */
function room(mb: MeshBuilder, m: RoomMats, uv: { floor?: [number, number]; back?: [number, number]; left?: [number, number] } = {}): void {
  mb.quad([-0.5, 0, 0.5], [0.5, 0, 0.5], [0.5, 0, -0.5], [-0.5, 0, -0.5], m.floor, uv.floor);           // +y
  mb.quad([-0.5, 1, -0.5], [0.5, 1, -0.5], [0.5, 1, 0.5], [-0.5, 1, 0.5], m.ceiling);                    // −y
  mb.quad([-0.5, 0, -0.5], [0.5, 0, -0.5], [0.5, 1, -0.5], [-0.5, 1, -0.5], m.back, uv.back);            // +z
  mb.quad([-0.5, 0, 0.5], [-0.5, 0, -0.5], [-0.5, 1, -0.5], [-0.5, 1, 0.5], m.left, uv.left);            // +x
  mb.quad([0.5, 0, -0.5], [0.5, 0, 0.5], [0.5, 1, 0.5], [0.5, 1, -0.5], m.right);                        // −x
}
const wallMats = (): MaterialData[] => [
  principled('floor', { baseColorFactor: [0.72, 0.7, 0.66, 1], roughnessFactor: 0.45 }),
  principled('back', { baseColorFactor: [0.7, 0.7, 0.7, 1], roughnessFactor: 0.6 }),
  principled('left', { baseColorFactor: [0.63, 0.12, 0.1, 1], roughnessFactor: 0.6 }),
  principled('right', { baseColorFactor: [0.14, 0.45, 0.1, 1], roughnessFactor: 0.6 }),
  principled('ceiling', { baseColorFactor: [0.75, 0.75, 0.75, 1], roughnessFactor: 0.8 }),
];
const ROOM: RoomMats = { floor: 0, back: 1, left: 2, right: 3, ceiling: 4 };

// ---- package writer ----------------------------------------------------------------------------------------------------

interface W { name: string; scene: SceneData; flatShaded: boolean; lightMode?: 'A' | 'B'; camera?: { matrix: number[]; yfov: number }; b?: number; extra?: Record<string, unknown>; envExtra?: Record<string, unknown> }
async function write(o: W): Promise<void> {
  const pkg = await exportScenePackage(o.scene, {
    camera: o.camera ?? ROOM_CAMERA, render: { width: 256, height: 256, maxBounces: o.b ?? 3 }, lightMode: o.lightMode ?? 'A', flatShaded: o.flatShaded,
    name: o.name, source: { uri: `validation/scenes/make-m7.ts (${o.name})` },
  });
  const json = { ...pkg.json, ...(o.envExtra && pkg.json.env ? { env: { ...pkg.json.env, ...o.envExtra } } : {}), ...o.extra };
  pkg.files.set('scene.json', new TextEncoder().encode(JSON.stringify(json, null, 1)));
  const dir = path.join(OUT, o.name);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  for (const [k, b] of pkg.files) writeFileSync(path.join(dir, k), b);
  console.log(`wrote ${path.relative(ROOT, dir)}${pkg.json.warnings?.length ? ` (warnings: ${pkg.json.warnings.join('; ')})` : ''}`);
}

// ---- scenes -------------------------------------------------------------------------------------------------------------

async function m7Smooth(): Promise<void> {
  resetLightIds();
  const mats = [...wallMats(),
    principled('dielectric', { baseColorFactor: [0.75, 0.75, 0.8, 1], roughnessFactor: 0.35 }),
    principled('gold', { baseColorFactor: [0.95, 0.72, 0.4, 1], metallicFactor: 1, roughnessFactor: 0.25 }),
    principled('torus', { baseColorFactor: [0.35, 0.55, 0.75, 1], roughnessFactor: 0.4 }),
    principled('sheet', { baseColorFactor: [0.6, 0.6, 0.8, 1], roughnessFactor: 0.3 })];
  const mb = new MeshBuilder();
  room(mb, ROOM);
  smoothSphere(mb, [-0.24, 0.18, -0.12], 0.18, 24, 48, 5);
  smoothSphere(mb, [0.24, 0.14, 0.1], 0.14, 20, 40, 6);
  smoothTorus(mb, [0.0, 0.07, 0.25], 0.12, 0.05, 40, 20, 7);
  smoothSheet(mb, 0.04, 0.46, -0.46, -0.12, 0.45, 0.03, 18, 15, 24, 8);
  const lights = [rectLight(), light('point', new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.3, 0.78, 0.25, 1]), 3)];
  await write({ name: 'm7_smooth_256', scene: sceneOf('m7_smooth_256', mb, mats, lights), flatShaded: false,
    extra: { tier: 'model-approximate', notes: 'smooth shading, no normal maps; open wavy sheet (Cycles NEE/BSDF MIS non-partition where Ng·L ≤ 0 < Ns·L, gap-bsdf §8.2)' } });
}

/** M7-11: low-poly smooth shading (Ns up to ~35–45° from Ng): the B-SM-J plant needs reconnection vertices where the
 *  shading and geometric cosines differ (m7_smooth_256 is finely tessellated: undetectable, measured). */
async function m7SmoothLowpoly(): Promise<void> {
  resetLightIds();
  const mats = [...wallMats(),
    principled('dielectric', { baseColorFactor: [0.75, 0.75, 0.8, 1], roughnessFactor: 0.35 }),
    principled('gold', { baseColorFactor: [0.95, 0.72, 0.4, 1], metallicFactor: 1, roughnessFactor: 0.3 }),
    principled('torus', { baseColorFactor: [0.35, 0.55, 0.75, 1], roughnessFactor: 0.4 }),
    principled('sheet', { baseColorFactor: [0.6, 0.6, 0.8, 1], roughnessFactor: 0.3 })];
  const mb = new MeshBuilder();
  room(mb, ROOM);
  smoothSphere(mb, [-0.24, 0.18, -0.12], 0.18, 4, 6, 5);
  smoothSphere(mb, [0.24, 0.14, 0.1], 0.14, 4, 7, 6);
  smoothTorus(mb, [0.0, 0.07, 0.25], 0.12, 0.05, 8, 5, 7);
  smoothSheet(mb, 0.04, 0.46, -0.46, -0.12, 0.45, 0.06, 18, 15, 4, 8);
  const lights = [rectLight(), light('point', new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.3, 0.78, 0.25, 1]), 3)];
  await write({ name: 'm7_smooth_lowpoly_256', scene: sceneOf('m7_smooth_lowpoly_256', mb, mats, lights), flatShaded: false,
    extra: { tier: 'model-approximate', notes: 'm7_smooth_256 with low-poly meshes (sphere 4×6 / 4×7, torus 8×5, sheet 4×4): Ns up to ~40° from Ng' } });
}

async function m7NmFlat(): Promise<void> {
  resetLightIds();
  const tex = [tilesMap('tiles', 256, 4), bumpsMap('bumps', 256, 6, 0.8), wavesMap('waves', 256, 5), tiltMap('tilt35', 35 * deg, 90 * deg), tiltMap('tilt60', 60 * deg, 270 * deg)];
  const nm = (i: number, s: number) => ({ normalTexture: { texture: i, texCoord: 0, scale: s } });
  const mats = [
    principled('floor', { baseColorFactor: [0.72, 0.7, 0.66, 1], roughnessFactor: 0.35, ...nm(0, 1) }),
    principled('back', { baseColorFactor: [0.7, 0.7, 0.7, 1], roughnessFactor: 0.6, ...nm(1, 0.6) }),
    principled('left', { baseColorFactor: [0.63, 0.12, 0.1, 1], roughnessFactor: 0.5, ...nm(2, 1.7) }),
    principled('right', { baseColorFactor: [0.14, 0.45, 0.1, 1], roughnessFactor: 0.6 }),
    principled('ceiling', { baseColorFactor: [0.75, 0.75, 0.75, 1], roughnessFactor: 0.8 }),
    principled('box', { baseColorFactor: [0.8, 0.8, 0.78, 1], roughnessFactor: 0.4, ...nm(1, 1) }),
    principled('panel_tilt', { baseColorFactor: [0.8, 0.8, 0.8, 1], roughnessFactor: 0.7, ...nm(3, 1) }),
    principled('panel_s05', { baseColorFactor: [0.8, 0.8, 0.8, 1], roughnessFactor: 0.7, ...nm(4, 0.5) }),
  ];
  const mb = new MeshBuilder();
  room(mb, ROOM, { floor: [2, 2], back: [2, 2], left: [1.5, 1.5] });
  const bx = mb.pos.length / 3;
  mb.box([-0.36, 0, 0.0], [-0.12, 0.42, 0.24], 5, { rotY: 0.35, omit: ['-y'] });
  // box UVs: one unit per face from the face-local corner order (MeshBuilder.quad writes (0,1)(1,1)(1,0)(0,0))
  void bx;
  const z = -0.497;
  const P = M7_PANELS;
  mb.quad([P.P1.x[0], P.P1.y[0], z], [P.P1.x[1], P.P1.y[0], z], [P.P1.x[1], P.P1.y[1], z], [P.P1.x[0], P.P1.y[1], z], 6);
  const b2 = mb.pos.length / 3;
  mb.quad([P.P2.x[0], P.P2.y[0], z], [P.P2.x[1], P.P2.y[0], z], [P.P2.x[1], P.P2.y[1], z], [P.P2.x[0], P.P2.y[1], z], 6);
  for (let k = 0; k < 4; k++) mb.uv[2 * (b2 + k)] = 1 - mb.uv[2 * (b2 + k)];   // mirrored u ⇒ w = −1, same B
  mb.quad([P.P3.x[0], P.P3.y[0], z], [P.P3.x[1], P.P3.y[0], z], [P.P3.x[1], P.P3.y[1], z], [P.P3.x[0], P.P3.y[1], z], 7);
  const lights = [rectLight(), light('spot', lightToward([0.15, 0.05, -1], [-0.1, 0.62, 0.45]), 6, { spotSize: 70 * deg, spotBlend: 0.2 })];
  await write({ name: 'm7_nm_flat_256', scene: sceneOf('m7_nm_flat_256', mb, mats, lights, tex), flatShaded: true,
    extra: { tier: 'tight', panels: P, notes: 'flat geometry + normal maps (tiles s 1, bumps s 0.6 / 1, waves s 1.7); panels P1 tilt 35° +B s 1, P2 = P1 with mirrored u, P3 tilt 60° −B (down) s 0.5' } });
}

function nmSmoothBuild(name: string): { scene: SceneData } {
  resetLightIds();
  const tex = [bumpsMap('bumps', 256, 8, 0.8), wavesMap('waves', 256, 6, 0.3), tilesMap('tiles', 256, 6, 0.1, 0.5)];
  const nm = (i: number, s: number) => ({ normalTexture: { texture: i, texCoord: 0, scale: s } });
  const mats = [...wallMats(),
    principled('bumps_diffuse', { baseColorFactor: [0.78, 0.76, 0.74, 1], roughnessFactor: 0.5, ...nm(0, 1) }),
    principled('waves_gold', { baseColorFactor: [0.95, 0.72, 0.4, 1], metallicFactor: 1, roughnessFactor: 0.3, ...nm(1, 0.8) }),
    principled('tiles_torus', { baseColorFactor: [0.35, 0.6, 0.4, 1], roughnessFactor: 0.45, ...nm(2, 1) }),
    principled('bumps_sheet', { baseColorFactor: [0.6, 0.6, 0.8, 1], roughnessFactor: 0.3, ...nm(0, 1.3) })];
  const mb = new MeshBuilder();
  room(mb, ROOM);
  smoothSphere(mb, [-0.24, 0.18, -0.12], 0.18, 24, 48, 5, { uvScale: [3, 1.5] });
  smoothSphere(mb, [0.24, 0.14, 0.1], 0.14, 20, 40, 6, { uvScale: [2, 1] });
  smoothTorus(mb, [0.0, 0.07, 0.25], 0.12, 0.05, 40, 20, 7, { uvScale: [4, 1] });
  smoothSheet(mb, 0.04, 0.46, -0.46, -0.12, 0.45, 0.03, 18, 15, 24, 8, { uvScale: [2, 1.6] });
  const lights = [rectLight(), light('point', new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.3, 0.78, 0.25, 1]), 3)];
  return { scene: sceneOf(name, mb, mats, lights, tex) };
}
async function m7NmSmooth(): Promise<void> {
  await write({ name: 'm7_nm_smooth_256', scene: nmSmoothBuild('m7_nm_smooth_256').scene, flatShaded: false,
    extra: { tier: 'model-approximate', notes: 'smooth + normal-mapped: bumps sphere, waves gold sphere, tiles torus, bumps open sheet (s 1.3)' } });
}
async function m7NmSmoothB(): Promise<void> {
  await write({ name: 'm7_nm_smooth_B_256', scene: nmSmoothBuild('m7_nm_smooth_B_256').scene, flatShaded: false, lightMode: 'B',
    extra: { tier: 'model-approximate', notes: 'm7_nm_smooth_256 in light mode B' } });
}

// ---- env scenes ---------------------------------------------------------------------------------------------------------

function hdriFile(id: string, ext: 'hdr' | 'exr'): { rel: string; bytes: Uint8Array; sha256: string } {
  const rel = `${HDRI_DIR}/${id}_1k.${ext}`;
  let bytes: Uint8Array;
  try { bytes = new Uint8Array(readFileSync(path.join(ROOT, rel))); } catch { throw new Error(`${rel} missing: run npx tsx validation/assets/fetch_hdris.ts`); }
  return { rel, bytes, sha256: createHash('sha256').update(bytes).digest('hex') };
}
function envFrom(id: string, ext: 'hdr' | 'exr', rotationZ: number): EnvironmentData {
  const f = hdriFile(id, ext);
  const e = decodeEnvironment(f.bytes, { mode: 'validation', name: `${id}_1k.${ext}`, rotationZ }).env;
  return { ...e, rotationZ };
}

async function m7NmEnv(): Promise<void> {
  resetLightIds();
  const tex = [bumpsMap('bumps', 256, 8, 0.8), wavesMap('waves', 256, 6, 0.3), tilesMap('tiles', 256, 8)];
  const nm = (i: number, s: number) => ({ normalTexture: { texture: i, texCoord: 0, scale: s } });
  const mats = [
    principled('ground', { baseColorFactor: [0.5, 0.48, 0.45, 1], roughnessFactor: 0.5, ...nm(2, 1) }),
    principled('bumps_diffuse', { baseColorFactor: [0.78, 0.76, 0.74, 1], roughnessFactor: 0.6, ...nm(0, 1) }),
    principled('waves_glossy', { baseColorFactor: [0.2, 0.25, 0.6, 1], roughnessFactor: 0.15, ...nm(1, 1) }),
    principled('bumps_metal', { baseColorFactor: [0.9, 0.9, 0.92, 1], metallicFactor: 1, roughnessFactor: 0.25, ...nm(0, 0.7) }),
  ];
  const mb = new MeshBuilder();
  mb.quad([-1.5, 0, 1.5], [1.5, 0, 1.5], [1.5, 0, -1.5], [-1.5, 0, -1.5], 0, [3, 3]);
  smoothSphere(mb, [-0.62, 0.3, 0], 0.3, 24, 48, 1, { uvScale: [3, 1.5] });
  smoothSphere(mb, [0, 0.3, 0], 0.3, 24, 48, 2, { uvScale: [2, 1] });
  smoothSphere(mb, [0.62, 0.3, 0], 0.3, 24, 48, 3, { uvScale: [3, 1.5] });
  const scene = { ...sceneOf('m7_nm_env_256', mb, mats, [], tex), env: envFrom('overcast_soil_puresky', 'hdr', 0.9) };
  await write({ name: 'm7_nm_env_256', scene, flatShaded: false, camera: { matrix: lookAt([0, 0.9, 2.3], [0, 0.25, 0]), yfov: 40 * deg },
    extra: { tier: 'model-approximate', notes: 'smooth normal-mapped spheres on a normal-mapped ground under overcast_soil_puresky 1k (γ 0.9)' } });
}

/** (xiv)-lite for E2E-HDR: Cornell without its ceiling, the ORIGINAL file loaded by both renderers. */
async function xivLite(ext: 'hdr' | 'exr'): Promise<void> {
  resetLightIds();
  const c = await cornellBase();
  const g = c.geometry;
  let ymax = -Infinity;
  for (let i = 1; i < g.positions.length; i += 3) ymax = Math.max(ymax, g.positions[i]);
  const mb = new MeshBuilder();
  for (let t = 0; t < g.indices.length / 3; t++) {
    const P = (k: number): V3 => { const i = g.indices[3 * t + k]; return [g.positions[3 * i], g.positions[3 * i + 1], g.positions[3 * i + 2]]; };
    if ([0, 1, 2].every((k) => Math.abs(P(k)[1] - ymax) < 1e-6)) continue;
    mb.tri(P(0), P(1), P(2), g.triMaterial[t]);
  }
  const f = hdriFile('overcast_soil_puresky', ext);
  const name = `m7_xivlite_${ext}_256`;
  const scene = { ...sceneOf(name, mb, c.materials, []), env: envFrom('overcast_soil_puresky', ext, 0.9) };
  await write({ name, scene, flatShaded: true, camera: c.camera, b: 2,
    envExtra: { original: { file: f.rel, sha256: f.sha256, bytes: f.bytes.length, format: ext } },
    extra: { tier: 'tight', notes: `E2E-HDR (xiv)-lite: Blender loads ${f.rel} directly (env.original); b = 2`, source_glb_sha256: c.glbSha } });
}

const GENS: Record<string, () => Promise<void>> = {
  m7_smooth_256: m7Smooth, m7_smooth_lowpoly_256: m7SmoothLowpoly, m7_nm_flat_256: m7NmFlat, m7_nm_smooth_256: m7NmSmooth, m7_nm_smooth_B_256: m7NmSmoothB, m7_nm_env_256: m7NmEnv,
  m7_xivlite_hdr_256: () => xivLite('hdr'), m7_xivlite_exr_256: () => xivLite('exr'),
};

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });
  for (const n of M7_SCENES) if (!ONLY || ONLY.has(n)) await GENS[n]();
}

if (process.argv[1] && import.meta.url === new URL(`file://${path.resolve(process.argv[1])}`).href) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
