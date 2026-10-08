// Small scene-building kit for the validation scene generators (validation/scenes/make-*.ts): flat-shaded meshes
// (quads, boxes, icospheres), V1 / Principled materials, light and camera matrices, procedural RGBA8 textures, the
// Cornell base (cornell.glb + cornell.meta.json) and the package writer (exportScenePackage + extra scene.json keys).
// Frame: glTF canonical (+Y up, metres, un-recentred), docs/decisions/scene-bridge.md.
// Every scene is quantized ONCE, in sceneOf (docs/decisions/data-formats.md §B0): assets are loaded lossless here
// (cornellBase) so the only lossy step is the final scene's lattice snap; the packages are v2.
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadGltf } from '../../src/core/scene/gltf-loader.ts';
import { exportScenePackage, type ExportScenePackageOptions } from '../../src/core/scene/scene-package.ts';
import type { MaterialData, SceneData, SceneGeometry } from '../../src/core/scene/types.ts';

export const ROOT = fileURLToPath(new URL('../../', import.meta.url));
export * from './kit-core.ts';
import { deg, v1, type V3 } from './kit-core.ts';

// ---- Cornell base ---------------------------------------------------------------------------------------------------

export interface CornellBase {
  geometry: SceneGeometry;
  materials: MaterialData[];
  camera: { matrix: number[]; yfov: number };
  /** The file's rect light (4 W, 0.13 × 0.105 m, one-sided, facing down). */
  rect: { position: V3; sizeX: number; sizeY: number; power: number; matrix: Float32Array };
}

export async function cornellBase(): Promise<CornellBase & { glbSha: string }> {
  const dir = path.join(ROOT, 'validation/assets/cornell');
  const meta = JSON.parse(readFileSync(path.join(dir, 'cornell.meta.json'), 'utf8')) as {
    camera: { position: V3; vfov_deg: number }; lights: { position: V3; size_x: number; size_along_minus_z_gltf: number; power_W: number }[]; reflectance: Record<string, V3>;
  };
  const glb = new Uint8Array(readFileSync(path.join(dir, 'cornell.glb')));
  const base = (await loadGltf({ kind: 'glb', bytes: glb, name: 'cornell.glb' }, { tangents: false, quantize: 'lossless' })).scene;
  const materials = base.materials.map((m) => v1(m.name, { diffuse: meta.reflectance[m.name] }));
  const L = meta.lights[0];
  return {
    geometry: base.geometry, materials,
    camera: { matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...meta.camera.position, 1], yfov: meta.camera.vfov_deg * deg },
    rect: { position: L.position, sizeX: L.size_x, sizeY: L.size_along_minus_z_gltf, power: L.power_W, matrix: new Float32Array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, ...L.position, 1]) },
    glbSha: Buffer.from(await crypto.subtle.digest('SHA-256', glb)).toString('hex'),
  };
}

// ---- package writer --------------------------------------------------------------------------------------------------

export interface WriteOptions extends Omit<ExportScenePackageOptions, 'lightMode'> {
  /** Default 'A' (M3a: analytic lights NEE-only); 'B' / 'A′' (M3b) render with Cycles per-light MIS on. */
  lightMode?: ExportScenePackageOptions['lightMode'];
  /** Extra top-level scene.json keys (e.g. "expected", "notes", "plant"). */
  extra?: Record<string, unknown>;
}

export async function writePackage(outRoot: string, scene: SceneData, o: WriteOptions): Promise<string> {
  const { extra, ...opts } = o;
  const pkg = await exportScenePackage(scene, { flatShaded: true, ...opts, lightMode: opts.lightMode ?? 'A' });
  if (extra) pkg.files.set('scene.json', new TextEncoder().encode(JSON.stringify({ ...pkg.json, ...extra }, null, 1)));
  const dir = path.join(outRoot, o.name ?? scene.name);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  for (const [k, b] of pkg.files) writeFileSync(path.join(dir, k), b);
  return dir;
}
