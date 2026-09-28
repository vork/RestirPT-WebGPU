// Spot calibration scene for Gate 1-lite (plan §5 M2 exit: "spot blend 0.15 → 0.16" planted bias; §7.2 C0d).
//   npx tsx validation/scenes/make-spot-c0d.ts [outDir]
// A 10 × 10 m diffuse floor (ρ = 0.5, y = 0, normal +Y) lit by one spot light 1 m above the origin, cone axis tilted
// 20° from straight down toward +X; spotSize 60°, power 100 W, maxBounces 0 (direct light only), camera looking
// down at the lit footprint. Writes validation/scenes/spot_c0d_512/ (spotBlend 0.15) and
// validation/scenes/spot_c0d_blend0.16_512/ (the plant: spotBlend 0.16, rendered on disjoint seeds).
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { exportScenePackage } from '../../src/core/scene/scene-package.ts';
import type { SceneData } from '../../src/core/scene/types.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const OUT = process.argv[2] ? path.resolve(process.argv[2]) : path.join(ROOT, 'validation/scenes'); // [out dir] (the M2 gate regenerates into a temp dir and diffs)
const deg = Math.PI / 180;
type V3 = [number, number, number];

const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3): V3 => { const l = Math.hypot(...a); return [a[0] / l, a[1] / l, a[2] / l]; };

/** Camera-to-world (column-major) looking from `eye` at `target` (camera looks down local −Z, +Y up). */
function lookAt(eye: V3, target: V3, up: V3): number[] {
  const z = norm(sub(eye, target));
  const x = norm(cross(up, z));
  const y = cross(z, x);
  return [...x, 0, ...y, 0, ...z, 0, ...eye, 1];
}

async function main(): Promise<void> {
  const S = 5;
  const positions = Float32Array.from([-S, 0, S, S, 0, S, S, 0, -S, -S, 0, -S]);
  const tilt = 20 * deg, c = Math.cos(tilt), s = Math.sin(tilt);
  // R_z(tilt)·R_x(−90°): local −Z (spot axis) = (sin t, −cos t, 0); position (0, 1, 0)
  const spotMatrix = new Float32Array([c, s, 0, 0, 0, 0, -1, 0, -s, c, 0, 0, 0, 1, 0, 1]);
  const scene = (blend: number): SceneData => ({
    name: 'spot_c0d',
    geometry: {
      positions, normals: Float32Array.from([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]), tangents: new Float32Array(16),
      uv0: Float32Array.from([0, 1, 1, 1, 1, 0, 0, 0]), indices: Uint32Array.from([0, 1, 2, 0, 2, 3]), triMaterial: new Uint32Array(2), triFlags: new Uint32Array(2),
    },
    materials: [{
      name: 'floor', model: 'v1', v1: { diffuse: [0.5, 0.5, 0.5], glossy: [0, 0, 0], roughness: 0.5, mix: 0 },
      baseColorFactor: [0.5, 0.5, 0.5, 1], metallicFactor: 0, roughnessFactor: 1, emissiveFactor: [0, 0, 0], emissiveStrength: 1,
      ior: 1.5, specularFactor: 0, specularColorFactor: [1, 1, 1], transmissionFactor: 0, alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true,
    }],
    textures: [],
    lights: [{ id: 0, name: 'spot', type: 'spot', color: [1, 1, 1], power: 100, exposure: 0, matrix: spotMatrix, spotSize: 60 * deg, spotBlend: blend, visibleToCamera: false }],
    cameras: [], bounds: { min: [-S, 0, -S], max: [S, 0, S] }, warnings: [],
  });
  const camera = { matrix: lookAt([0.45, 2.2, 1.6], [0.45, 0, 0], [0, 1, 0]), yfov: 45 * deg };
  const variants = [
    { name: 'spot_c0d_512', blend: 0.15, plant: undefined },
    { name: 'spot_c0d_blend0.16_512', blend: 0.16, plant: { of: 'spot_c0d_512', what: 'spotBlend 0.15 -> 0.16' } },
  ];
  for (const v of variants) {
    const pkg = await exportScenePackage(scene(v.blend), {
      camera, render: { width: 512, height: 512, maxBounces: 0 }, lightMode: 'A', flatShaded: true, name: v.name,
      source: { uri: 'validation/scenes/make-spot-c0d.ts' },
    });
    const json = v.plant ? { ...pkg.json, plant: v.plant } : pkg.json;
    pkg.files.set('scene.json', new TextEncoder().encode(JSON.stringify(json, null, 1)));
    const dir = path.join(OUT, v.name);
    rmSync(dir, { recursive: true, force: true });
    mkdirSync(dir, { recursive: true });
    for (const [k, b] of pkg.files) writeFileSync(path.join(dir, k), b);
    console.log(`wrote ${path.relative(ROOT, dir)} (spotBlend ${v.blend})`);
  }
}

await main();
