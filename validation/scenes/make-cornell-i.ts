// Scene (i) diffuse Cornell package for Gate 1-lite (plan §5 M2 exit, §7.2 (i)) and its planted twin.
//   npx tsx validation/scenes/make-cornell-i.ts [outDir]
// Source: validation/assets/cornell/cornell.glb (geometry, primId order) + cornell.meta.json (rect light, camera,
// reflectances; the glTF exporter drops the area light). V1 pure-diffuse materials with the exact Cornell
// reflectances, one-sided rect light 0.13 × 0.105 m, 4 W, spread 180°, visibleToCamera false (Mode A), b = 3, flat.
// Writes validation/scenes/cornell_i_512/ and validation/scenes/cornell_i_power1.0075_512/ (light power × 1.0075,
// the Stage-A δ-scale plant, rendered on disjoint seeds for the Cycles-vs-Cycles detection check).
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadGltf } from '../../src/core/scene/gltf-loader.ts';
import { exportScenePackage } from '../../src/core/scene/scene-package.ts';
import type { LightData, SceneData } from '../../src/core/scene/types.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const ASSET = path.join(ROOT, 'validation/assets/cornell');
const OUT = process.argv[2] ? path.resolve(process.argv[2]) : path.join(ROOT, 'validation/scenes'); // [out dir] (the M2 gate regenerates into a temp dir and diffs)
const PLANT_FACTOR = 1.0075;

type V3 = [number, number, number];
interface CornellMeta {
  camera: { position: V3; forward: V3; up: V3; vfov_deg: number };
  lights: { name: string; type: string; power_W: number; color: V3; position: V3; normal: V3; size_x: number; size_along_minus_z_gltf: number; spread_deg: number }[];
  reflectance: Record<string, V3>;
}

async function main(): Promise<void> {
  const meta = JSON.parse(readFileSync(path.join(ASSET, 'cornell.meta.json'), 'utf8')) as CornellMeta;
  const glb = new Uint8Array(readFileSync(path.join(ASSET, 'cornell.glb')));
  const base = (await loadGltf({ kind: 'glb', bytes: glb, name: 'cornell.glb' }, { tangents: false })).scene;

  const materials = base.materials.map((m) => {
    const rho = meta.reflectance[m.name];
    if (!rho) throw new Error(`cornell.glb material '${m.name}' has no reflectance in cornell.meta.json`);
    return { ...m, model: 'v1' as const, baseColorFactor: [...rho, 1] as [number, number, number, number], v1: { diffuse: [...rho] as V3, glossy: [0, 0, 0] as V3, roughness: 0.5, mix: 0 } };
  });

  const L = meta.lights[0];
  if (meta.lights.length !== 1 || L.type !== 'rect' || L.normal[1] !== -1) throw new Error('expected one downward rect light in cornell.meta.json');
  // local -Z (emission) = world -Y, local X = world X, local Y = world -Z (size_along_minus_z_gltf)
  const lightMatrix = new Float32Array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, L.position[0], L.position[1], L.position[2], 1]);
  const light = (power: number): LightData => ({
    id: 0, name: L.name, type: 'rect', color: [...L.color], power, exposure: 0, matrix: lightMatrix,
    sizeX: L.size_x, sizeY: L.size_along_minus_z_gltf, spread: L.spread_deg * Math.PI / 180, visibleToCamera: false,
  });

  const c = meta.camera;
  if (c.forward.join() !== '0,0,-1' || c.up.join() !== '0,1,0') throw new Error('camera is expected to look down -Z with +Y up');
  const camera = { matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...c.position, 1], yfov: c.vfov_deg * Math.PI / 180 };

  const glbSha = Buffer.from(await crypto.subtle.digest('SHA-256', glb)).toString('hex');
  const variants = [
    { name: 'cornell_i_512', power: L.power_W, plant: undefined },
    { name: `cornell_i_power${PLANT_FACTOR}_512`, power: L.power_W * PLANT_FACTOR, plant: { of: 'cornell_i_512', what: `light power x${PLANT_FACTOR}` } },
  ];
  for (const v of variants) {
    const scene: SceneData = { ...base, name: v.name, materials, lights: [light(v.power)], cameras: [] };
    const pkg = await exportScenePackage(scene, {
      camera, render: { width: 512, height: 512, maxBounces: 3 }, lightMode: 'A', flatShaded: true, name: v.name,
      source: { uri: 'validation/assets/cornell/cornell.glb + cornell.meta.json (validation/scenes/make-cornell-i.ts)', sha256: glbSha },
    });
    const json = v.plant ? { ...pkg.json, plant: v.plant } : pkg.json;
    pkg.files.set('scene.json', new TextEncoder().encode(JSON.stringify(json, null, 1)));
    const dir = path.join(OUT, v.name);
    rmSync(dir, { recursive: true, force: true });
    mkdirSync(dir, { recursive: true });
    for (const [k, b] of pkg.files) writeFileSync(path.join(dir, k), b);
    console.log(`wrote ${path.relative(ROOT, dir)} (${scene.geometry.indices.length / 3} tris, light ${v.power} W)`);
  }
}

await main();
