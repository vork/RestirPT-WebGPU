// M7 E2E-GLB / E2E-USD stock-import packages (PLAN §7.2; docs/decisions/m7-api.md §3.4):
//   npx tsx validation/scenes/make-m7-e2e.ts [outDir] [--only name,name]
// Default outDir: validation/out/m7/e2e (generated, deterministic). Each package is OUR loader's scene in
// Blender-compatible mode (lossless: no lattice snap, data-formats.md E-13; glTF FLAT variants shade every face flat like
// Blender's import_shading FLAT) with the test's camera, added lights / env, and a scene.json "stock" block naming the
// ORIGINAL asset (repo path + sha256) and the importer options: our PT renders the package, the Cycles reference is
// Blender's stock importer on the asset (validation/blender/build_scene.py build_from_stock) + the §7.5 settings.
//   glTF (bpy.ops.import_scene.gltf, lighting SPEC): cornell_point_spot (its own point + spot; Mode B), the Khronos samples
//        MetalRoughSpheresNoTextures, TextureTransformTest, NormalTangentMirrorTest, AlphaBlendModeTest (camera on the
//        MASK panels: BLEND is not bridged, we render it as MASK 0.5), EmissiveStrengthTest, TransmissionTest, IORTestGrid
//        under overcast_soil_puresky + an added 0.6 × 0.6 m rect light; each with import_shading FLAT and NORMALS.
//   USD (bpy.ops.wm.usd_import, defaults): the Blender Cornell (validation/assets/cornell/cornell.usda), the hand files
//        m7_e2e_yup (Y-up, cm) / m7_e2e_zup (Z-up, m), m7_textured (UsdUVTexture), m7_instancing (instanceable +
//        PointInstancer); the asset's camera and lights.
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { exportScenePackage } from '../../src/core/scene/scene-package.ts';
import { loadGltf } from '../../src/core/scene/gltf-loader.ts';
import { loadUsdInline } from '../../src/core/scene/usd/load-usd.ts';
import { decodeEnvironment } from '../../src/core/scene/env/load-env.ts';
import { decodePng, isPng } from '../../src/core/io/png.ts';
import type { LightData, SceneData, SceneGeometry } from '../../src/core/scene/types.ts';
import { ROOT, deg, light, lookAt, resetLightIds, type V3 } from './scene-kit.ts';

const argv = process.argv.slice(2);
const flag = (k: string) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : undefined; };
const ONLY = flag('--only') ? new Set(flag('--only')!.split(',')) : undefined;
export const M7_E2E_OUT = 'validation/out/m7/e2e';
const OUT = argv[0] && !argv[0].startsWith('--') ? path.resolve(argv[0]) : path.join(ROOT, M7_E2E_OUT);
const K = 'validation/assets/downloaded/khronos';

export interface E2EUnit {
  name: string; importer: 'gltf' | 'usd'; file: string; shading?: 'FLAT' | 'NORMALS'; lightMode: 'A' | 'B'; env: boolean; addRect: boolean;
  maxBounces: number; camera?: { eye: V3; target: V3; yfov: number }; notes: string;
}
const GLB: [string, string, Partial<E2EUnit>][] = [
  ['cornell_point_spot', 'validation/assets/cornell/cornell_point_spot.glb', { env: false, addRect: false, lightMode: 'B', notes: 'Cornell + KHR_lights_punctual point and spot (inner 0.3 ≠ 0, outer 0.6)' }],
  ['metalrough_spheres', `${K}/MetalRoughSpheresNoTextures/MetalRoughSpheresNoTextures.glb`, { notes: 'metallic × roughness sphere grid' }],
  ['texture_transform', `${K}/TextureTransformTest/TextureTransformTest.gltf`, { notes: 'KHR_texture_transform offset / rotation / scale, clamp / repeat' }],
  ['normal_tangent_mirror', `${K}/NormalTangentMirrorTest/NormalTangentMirrorTest.glb`, { notes: 'normal maps with mirrored UVs (MikkTSpace sign)' }],
  ['alpha_mask', `${K}/AlphaBlendModeTest/AlphaBlendModeTest.glb`, { camera: { eye: [1.5, 1.0, 5.2], target: [1.5, 1.0, 0], yfov: 32 * deg }, notes: 'MASK panels (cutoff 0.25 / 0.5 / 0.75); the BLEND panel is outside the view' }],
  ['emissive_strength', `${K}/EmissiveStrengthTest/EmissiveStrengthTest.glb`, { notes: 'KHR_materials_emissive_strength' }],
  ['transmission', `${K}/TransmissionTest/TransmissionTest.glb`, { maxBounces: 6, notes: 'KHR_materials_transmission (rough / smooth, textured)' }],
  ['ior_grid', `${K}/IORTestGrid/IORTestGrid.glb`, { maxBounces: 6, notes: 'KHR_materials_ior + transmission + specular (KHR_materials_volume not modelled)' }],
];
export const E2E_UNITS: E2EUnit[] = [
  ...GLB.flatMap(([n, f, o]) => (['FLAT', 'NORMALS'] as const).map((sh) => ({
    name: `e2e_glb_${n}_${sh.toLowerCase()}`, importer: 'gltf' as const, file: f, shading: sh, lightMode: 'A' as const, env: true, addRect: true, maxBounces: 3, notes: '', ...o,
  }))),
  ...([['cornell', 'validation/assets/cornell/cornell.usda', 'the Blender Cornell export (Z-up, rect light)'],
    ['hand_yup', 'validation/assets/usd-m7/m7_e2e_yup.usda', 'hand file: upAxis Y, metersPerUnit 0.01; rect normalize 1, disk normalize 0, sphere treatAsPoint, sphere r 0.5, shaping, distant; UV flip'],
    ['hand_zup', 'validation/assets/usd-m7/m7_e2e_zup.usda', 'hand file: upAxis Z, metersPerUnit 1 (same content)'],
    ['textured', 'validation/assets/usd-m7/m7_textured.usda', 'UsdUVTexture: diffuse / roughness-metallic / normal / emissive / opacity'],
    ['instancing', 'validation/assets/usd-m7/m7_instancing.usda', 'instanceable references + PointInstancer']] as const).map(([n, f, notes]) => ({
    name: `e2e_usd_${n}`, importer: 'usd' as const, file: f, lightMode: 'B' as const, env: false, addRect: false, maxBounces: 3, notes,
  })),
];

const sha = (b: Uint8Array) => createHash('sha256').update(b).digest('hex');

/** Node image decoder for the glTF loader: PNG (io/png.ts), JPEG through sharp (libjpeg-turbo, raw RGBA8). */
async function decodeImage(bytes: Uint8Array, mime: string): Promise<{ width: number; height: number; pixels: Uint8Array } | null> {
  if (isPng(bytes)) {
    try { return await decodePng(bytes); } catch { /* palette / interlaced PNG (io/png.ts subset): decoded by sharp below */ }
  }
  if (isPng(bytes) || mime === 'image/jpeg') {
    const { default: sharp } = await import('sharp');
    const { data, info } = await sharp(bytes, { failOn: 'none' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
    return { width: info.width, height: info.height, pixels: new Uint8Array(data.buffer, data.byteOffset, data.byteLength) };
  }
  return null;
}

/** Blender import_shading FLAT: every face its own vertices with the face normal. */
function flatShade(g: SceneGeometry): SceneGeometry {
  const nT = g.indices.length / 3;
  const P = new Float32Array(nT * 9), N = new Float32Array(nT * 9), U = new Float32Array(nT * 6), T = new Float32Array(nT * 12);
  const C = g.color0 ? new Float32Array(nT * 12) : undefined;
  for (let t = 0; t < nT; t++) {
    const v = [0, 1, 2].map((k) => g.indices[3 * t + k]);
    const p = v.map((i) => [g.positions[3 * i], g.positions[3 * i + 1], g.positions[3 * i + 2]]);
    const e1 = [0, 1, 2].map((k) => p[1][k] - p[0][k]), e2 = [0, 1, 2].map((k) => p[2][k] - p[0][k]);
    const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const l = Math.hypot(n[0], n[1], n[2]) || 1;
    v.forEach((i, k) => {
      const c = 3 * t + k;
      for (let j = 0; j < 3; j++) { P[3 * c + j] = g.positions[3 * i + j]; N[3 * c + j] = n[j] / l; }
      U[2 * c] = g.uv0[2 * i]; U[2 * c + 1] = g.uv0[2 * i + 1];
      for (let j = 0; j < 4; j++) T[4 * c + j] = g.tangents[4 * i + j];
      if (C) for (let j = 0; j < 4; j++) C[4 * c + j] = g.color0![4 * i + j];
    });
  }
  return { positions: P, normals: N, uv0: U, tangents: T, ...(C ? { color0: C } : {}), indices: Uint32Array.from({ length: nT * 3 }, (_, i) => i), triMaterial: g.triMaterial, triFlags: g.triFlags };
}

function frameCamera(s: SceneData): { matrix: number[]; yfov: number } {
  const c: V3 = [0, 1, 2].map((k) => 0.5 * (s.bounds.min[k] + s.bounds.max[k])) as V3;
  const r = 0.5 * Math.hypot(...[0, 1, 2].map((k) => s.bounds.max[k] - s.bounds.min[k]));
  const yfov = 40 * deg, d = 1.15 * r / Math.sin(yfov / 2);
  const dir: V3 = [0.35, 0.45, 1];
  const l = Math.hypot(...dir);
  return { matrix: lookAt([c[0] + d * dir[0] / l, c[1] + d * dir[1] / l, c[2] + d * dir[2] / l], c), yfov };
}

async function build(u: E2EUnit): Promise<void> {
  resetLightIds();
  const abs = path.join(ROOT, u.file);
  const bytes = new Uint8Array(readFileSync(abs));
  let scene: SceneData;
  if (u.importer === 'gltf') {
    let source;
    if (/\.glb$/i.test(abs)) source = { kind: 'glb' as const, bytes, name: path.basename(abs) };
    else {
      const json = JSON.parse(new TextDecoder().decode(bytes)) as { buffers?: { uri?: string }[]; images?: { uri?: string }[] };
      const resources: Record<string, Uint8Array> = {};
      for (const r of [...(json.buffers ?? []), ...(json.images ?? [])]) if (r.uri && !r.uri.startsWith('data:')) resources[r.uri] = new Uint8Array(readFileSync(path.join(path.dirname(abs), decodeURIComponent(r.uri))));
      source = { kind: 'gltf' as const, json: json as never, resources, name: path.basename(abs) };
    }
    scene = (await loadGltf(source, { quantize: 'lossless', decodeImage })).scene;
    if (u.shading === 'FLAT') scene = { ...scene, geometry: flatShade(scene.geometry) };
  } else {
    scene = (await loadUsdInline(bytes, path.basename(abs), { assetBase: path.dirname(abs), quantize: 'lossless', blenderCompat: true })).scene;
  }
  const lights: LightData[] = scene.lights.map((l) => ({ ...l, origin: 'asset' as const }));
  let id = Math.max(-1, ...lights.map((l) => l.id)) + 1;
  if (u.addRect) {
    const b = scene.bounds, c = [0, 1, 2].map((k) => 0.5 * (b.min[k] + b.max[k]));
    const r = 0.5 * Math.hypot(...[0, 1, 2].map((k) => b.max[k] - b.min[k]));
    const p: V3 = [c[0] + 0.3 * r, b.max[1] + 1.2 * r, c[2] + 0.6 * r];
    const m = new Float32Array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, ...p, 1]);   // facing −Y (down)
    lights.push({ ...light('rect', m, 40 * r * r, { sizeX: 0.6 * r, sizeY: 0.6 * r }), id: id++, name: 'added_rect', origin: 'added' });
  }
  const cam = u.camera ? { matrix: lookAt(u.camera.eye, u.camera.target), yfov: u.camera.yfov } : scene.cameras.length && u.importer === 'usd'
    ? { matrix: Array.from(scene.cameras[0].matrix), yfov: scene.cameras[0].yfov } : frameCamera(scene);
  let env: SceneData['env'];
  if (u.env) {
    const f = 'validation/assets/downloaded/hdri/overcast_soil_puresky_1k.hdr';
    env = { ...decodeEnvironment(new Uint8Array(readFileSync(path.join(ROOT, f))), { mode: 'validation', name: 'overcast_soil_puresky_1k.hdr' }).env, rotationZ: 0.9 };
  }
  const s: SceneData = { ...scene, lights, cameras: [], ...(env ? { env } : {}) };
  if (!env) delete s.env;
  const pkg = await exportScenePackage(s, { camera: cam, render: { width: 256, height: 256, maxBounces: u.maxBounces }, lightMode: u.lightMode, name: u.name, source: { uri: u.file, sha256: sha(bytes) } });
  const opts = u.importer === 'gltf' ? { export_import_convert_lighting_mode: 'SPEC', import_shading: u.shading } : {};
  const json = {
    ...pkg.json,
    stock: { importer: u.importer, file: u.file, sha256: sha(bytes), options: opts },
    ...(u.env && u.addRect ? { cycles: { use_light_tree: false } } : {}),   // env + analytic light: D6
    tier: 'model-approximate', notes: u.notes,
  };
  pkg.files.set('scene.json', new TextEncoder().encode(JSON.stringify(json, null, 1)));
  const dir = path.join(OUT, u.name);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  for (const [k, b] of pkg.files) writeFileSync(path.join(dir, k), b);
  console.log(`wrote ${path.relative(ROOT, dir)} (${s.geometry.indices.length / 3} tris, ${lights.length} lights${scene.warnings.length ? `; ${scene.warnings.length} loader warnings` : ''})`);
}

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });
  for (const u of E2E_UNITS) if (!ONLY || ONLY.has(u.name)) await build(u);
}

if (process.argv[1] && import.meta.url === new URL(`file://${path.resolve(process.argv[1])}`).href) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
