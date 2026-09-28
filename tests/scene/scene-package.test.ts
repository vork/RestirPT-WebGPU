// Scene package v1 (docs/decisions/scene-bridge.md): export ↔ read round trip on the Cornell GLB plus synthetic
// lights / textures / env, env pixel hash (ENV-U9 definition), contract checks (hard errors).
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { flipRows } from '../../src/core/io/exr.ts';
import { loadGltf } from '../../src/core/scene/gltf-loader.ts';
import {
  ScenePackageError, detectFlatShaded, envPixelHash, exportScenePackage, readScenePackage, type SceneJson,
} from '../../src/core/scene/scene-package.ts';
import type { EnvironmentData, LightData, SceneData } from '../../src/core/scene/types.ts';

const cam = { matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0.273, 1.0775, 1], yfov: 39.3 * Math.PI / 180 };
const render = { width: 512, height: 512, maxBounces: 3 };

async function cornell(): Promise<SceneData> {
  const bytes = new Uint8Array(readFileSync('validation/assets/cornell/cornell.glb'));
  return (await loadGltf({ kind: 'glb', bytes, name: 'cornell.glb' }, { tangents: false })).scene;
}

function env(W = 16, H = 8): EnvironmentData {
  const t = new Float32Array(W * H * 4);
  for (let i = 0; i < W * H; i++) { t[4 * i] = i * 0.37; t[4 * i + 1] = (i % W) / W; t[4 * i + 2] = 1e-38 * (i + 1); t[4 * i + 3] = 1; }
  return { name: 'synthetic', width: W, height: H, texels: t, strength: 1.7, tint: [1, 0.9, 0.8], rotationZ: 0.52, visibleToCamera: true };
}

const rect = (o: Partial<LightData> = {}): LightData => ({
  id: 7, name: 'ceiling', type: 'rect', color: [1, 1, 1], power: 4, exposure: 0,
  matrix: new Float32Array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0.554, 0, 1]), sizeX: 0.13, sizeY: 0.105, spread: Math.PI,
  visibleToCamera: false, ...o,
});

describe('scene package', () => {
  it('round-trips the Cornell GLB (geometry bit-exact, materials, lights, camera, env with hash)', async () => {
    const scene = await cornell();
    scene.lights.push(rect(), { id: 9, name: 'spot', type: 'spot', color: [1, 0.5, 0.25], power: 100, exposure: 1, matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1]), spotSize: 1, spotBlend: 0.15, visibleToCamera: false });
    scene.env = env();
    scene.materials[0] = { ...scene.materials[0], model: 'v1', v1: { diffuse: [0.725, 0.71, 0.68], glossy: [0, 0, 0], roughness: 0.5, mix: 0 } };
    const pkg = await exportScenePackage(scene, { camera: cam, render, lightMode: 'A' });
    expect([...pkg.files.keys()].sort()).toEqual(['env.exr', 'geometry.bin', 'scene.json']);
    const json = JSON.parse(new TextDecoder().decode(pkg.files.get('scene.json'))) as SceneJson;
    expect(json).toMatchObject({ format: 'restir-scene-package', version: 1, lightMode: 'A', flatShaded: true, render });
    expect(json.materials[0]).toMatchObject({ model: 'v1', v1: { diffuse: [0.725, 0.71, 0.68], mix: 0 }, emissionSampling: 'FRONT_BACK' });
    expect(json.lights[0]).toMatchObject({ id: 7, type: 'rect', spread: Math.PI, visibleToCamera: false });
    expect(json.env).toMatchObject({ file: 'env.exr', strength: 1.7, rotationZ: 0.52, sampling: 'AUTOMATIC', width: 16, height: 8 });
    expect(json.env!.sha256).toBe(await envPixelHash(flipRows(scene.env.texels, 16, 8)));
    for (const [k, b] of Object.entries(json.buffers)) expect(b.offset % 4, k).toBe(0);

    const back = await readScenePackage(pkg.files);
    const g0 = scene.geometry, g1 = back.scene.geometry;
    for (const k of ['positions', 'normals', 'uv0', 'indices', 'triMaterial', 'triFlags'] as const) {
      expect(new Uint32Array(g1[k].buffer), k).toEqual(new Uint32Array(g0[k].buffer.slice(g0[k].byteOffset, g0[k].byteOffset + g0[k].byteLength)));
    }
    expect(back.scene.materials).toEqual(scene.materials);
    expect(back.scene.lights.map((l) => ({ ...l, matrix: Array.from(l.matrix) }))).toEqual(scene.lights.map((l) => ({ ...l, matrix: Array.from(l.matrix) })));
    expect(back.scene.bounds).toEqual(scene.bounds);
    expect(back.camera.yfov).toBeCloseTo(cam.yfov, 12);
    expect(Array.from(back.camera.matrix)).toEqual(cam.matrix);
    expect(back.render).toEqual(render);
    expect(back.lightMode).toBe('A');
    expect(back.scene.env!.width).toBe(16);
    expect(new Uint32Array(back.scene.env!.texels.buffer)).toEqual(new Uint32Array(scene.env.texels.buffer));
    expect(back.scene.env).toMatchObject({ strength: 1.7, tint: [1, 0.9, 0.8], rotationZ: 0.52, visibleToCamera: true });

    // A second export of the read-back scene is byte-identical (stable writer).
    const again = await exportScenePackage(back.scene, { camera: back.camera, render, lightMode: 'A', name: json.name });
    for (const [k, v] of pkg.files) expect(again.files.get(k), k).toEqual(v);
  });

  it('writes and reads textures losslessly and keeps texture refs', async () => {
    const scene = await cornell();
    const px = new Uint8Array(8 * 4 * 4);
    for (let i = 0; i < px.length; i++) px[i] = (i * 29) & 0xff;
    scene.textures.push({ name: 'checker', width: 8, height: 4, pixels: px, wrapS: 'repeat', wrapT: 'clamp-to-edge', filter: 'nearest' });
    scene.materials[1] = { ...scene.materials[1], baseColorTexture: { texture: 0, texCoord: 0, transform: [1, 0, 0.5, 0, 2, 0] } };
    const pkg = await exportScenePackage(scene, { camera: cam, render, lightMode: 'B' });
    expect(pkg.json.textures).toEqual([{ file: 'tex_0.png', wrapS: 'repeat', wrapT: 'clamp-to-edge', filter: 'nearest', name: 'checker', width: 8, height: 4 }]);
    const back = await readScenePackage(Object.fromEntries(pkg.files));
    expect(back.scene.textures[0].pixels).toEqual(px);
    expect(back.scene.textures[0]).toMatchObject({ wrapS: 'repeat', wrapT: 'clamp-to-edge', filter: 'nearest', width: 8, height: 4 });
    expect(back.scene.materials[1].baseColorTexture).toEqual({ texture: 0, texCoord: 0, transform: [1, 0, 0.5, 0, 2, 0] });
  });

  it('hard errors: camera-visible area light in Mode A, undecoded texture, env hash mismatch', async () => {
    const scene = await cornell();
    await expect(exportScenePackage({ ...scene, lights: [rect({ visibleToCamera: true })] }, { camera: cam, render, lightMode: 'A' })).rejects.toThrow(ScenePackageError);
    await expect(exportScenePackage({ ...scene, lights: [rect({ visibleToCamera: true })] }, { camera: cam, render, lightMode: 'B' })).resolves.toBeDefined();
    await expect(exportScenePackage({ ...scene, textures: [{ name: 't', width: 4, height: 4, pixels: new Uint8Array(0), wrapS: 'repeat', wrapT: 'repeat', filter: 'linear' }] },
      { camera: cam, render, lightMode: 'A' })).rejects.toThrow(/no decoded RGBA8 pixels/);
    const pkg = await exportScenePackage({ ...scene, env: env() }, { camera: cam, render, lightMode: 'A' });
    const j = JSON.parse(new TextDecoder().decode(pkg.files.get('scene.json'))) as SceneJson;
    j.env!.sha256 = '0'.repeat(64);
    pkg.files.set('scene.json', new TextEncoder().encode(JSON.stringify(j)));
    await expect(readScenePackage(pkg.files)).rejects.toThrow(/hash mismatch/);
  });

  const SCENES = 'validation/scenes';
  const calib = existsSync(SCENES) ? readdirSync(SCENES).filter((d) => existsSync(`${SCENES}/${d}/scene.json`)) : [];
  it.skipIf(!calib.length)('reads every package written by validation/blender/calib_scenes.py and re-exports it losslessly', async () => {
    for (const d of calib) {
      const files = Object.fromEntries(readdirSync(`${SCENES}/${d}`).map((f) => [f, new Uint8Array(readFileSync(`${SCENES}/${d}/${f}`))]));
      const p = await readScenePackage(files); // verifies env.sha256 (ENV-U9 definition shared with build_scene.py)
      const again = await readScenePackage((await exportScenePackage(p.scene, {
        camera: p.camera, render: p.render, lightMode: p.lightMode, flatShaded: p.flatShaded, frames: p.frames, name: p.json.name,
      })).files);
      expect(again.scene.geometry.positions, d).toEqual(p.scene.geometry.positions);
      expect(again.scene.geometry.indices, d).toEqual(p.scene.geometry.indices);
      expect(again.scene.materials, d).toEqual(p.scene.materials);
      expect(again.json.env?.sha256, d).toBe(p.json.env?.sha256);
      expect(JSON.parse(JSON.stringify(again.json.frames ?? null)), d).toEqual(JSON.parse(JSON.stringify(p.json.frames ?? null)));
    }
  });

  it('detects smooth shading', async () => {
    const scene = await cornell();
    expect(detectFlatShaded(scene.geometry)).toBe(true);
    const n = scene.geometry.normals.slice();
    n[0] = 0.6; n[1] = 0.8; n[2] = 0;
    expect(detectFlatShaded({ ...scene.geometry, normals: n })).toBe(false);
  });
});
