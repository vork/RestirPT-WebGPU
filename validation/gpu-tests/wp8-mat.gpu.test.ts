// perf2 WP-8 (MAT_VARIANTS; docs/decisions/perf2-plan.md WP-8, perf2-api.md §1): a U-M8-MODEB-style A/B — the same short
// interactive ReSTIR chain with and without MAT_VARIANTS (alone and on top of the release set) must give identical
// reservoirs, images and counters on scenes the m8-bits Anchor cases do not cover: a V1-only package (Cornell), Glass +
// Refraction nodes, Principled textures with KHR_texture_transform, every material texture slot kind + Principled
// transmission (model 2), and the all-lights fixture with a Principled glass material on CWBVH. Plus the variant keys:
// a scene that gains a glass material is a different SceneGpu, so its kernel compiles other text (no stale variant).
import { describe, expect, it } from 'vitest';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import type { LightData, MaterialData, SceneData } from '../../src/core/scene/types.ts';
import { fetchScenePackage } from '../../src/core/scene/scene-package.ts';
import { INTERACTIVE_PINNED } from '../../src/core/render/restir/presets.ts';
import { KERNEL_RELEASE_PERF_FLAGS, normalizePerfFlags, type PerfFlagsInput } from '../../src/core/render/restir/perf-flags.ts';
import { allLightsScene, boxCamera, hashF32, restirRig, type GpuSceneOptions } from './restir-fixtures.ts';

type Cam = { camToWorld: number[]; yfov: number };

async function pkg(path: string): Promise<{ scene: SceneData; cam: Cam }> {
  const p = await fetchScenePackage(path);
  return { scene: p.scene, cam: { camToWorld: Array.from(p.camera.matrix), yfov: p.camera.yfov } };
}

/** Per frame `reservoirs:image:finalize counters:arena counters` of a 4-frame interactive chain (light / camera motion). */
async function chain(scene: SceneData, cam0: Cam, perfFlags: PerfFlagsInput, gpu?: GpuSceneOptions, maxBounces = 3): Promise<string[]> {
  const W = 64, H = 64;
  const rig = await restirRig(scene, W, H, {
    preset: 'interactive', settings: { maxBounces, ...INTERACTIVE_PINNED }, seed: 31, lightMode: 'B', cam: cam0, resLayout: 'soa',
    modeBNeedsAreaLights: true, gpu, perfFlags,
  });
  const k = rig.kernel, device = rig.g.device;
  await k.prepare();
  const out: string[] = [];
  let lights: LightData[] = scene.lights;
  let cam = cam0;
  for (let f = 0; f < 4; f++) {
    if (f >= 2) {
      lights = lights.map((l, i) => { if (i !== 0) return l; const m = new Float32Array(l.matrix); m[12] += 0.03; return { ...l, matrix: m }; });
      const m = cam.camToWorld.slice(), c = Math.cos(0.01), s = Math.sin(0.01);
      for (const col of [0, 4, 8]) { const x = m[col], z = m[col + 2]; m[col] = c * x + s * z; m[col + 2] = -s * x + c * z; }
      cam = { camToWorld: m, yfov: cam.yfov };
    }
    const clear = device.createCommandEncoder();
    clear.clearBuffer(rig.accum); clear.clearBuffer(rig.counters);
    device.queue.submit([clear.finish()]);
    k.advance({ t: 3 + f, camera: cam, lights });
    k.beginSubmit();
    const enc = device.createCommandEncoder();
    for (const u of k.frameUnits(3 + f, { accum: rig.accum, counters: rig.counters })) u.encode(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const img = new Float32Array(await readBuffer(device, rig.accum, W * H * 16));
    const fin = await k.readReservoirs('final');
    const cnt = Array.from(new Uint32Array(await readBuffer(device, rig.counters, 16)));
    const ar = await k.readCounters(true);
    out.push(`${hashF32(new Float32Array(fin.buffer))}:${hashF32(img)}:${cnt.join(',')}:${Object.values(ar.rsc).join(',')}`);
  }
  rig.destroy();
  return out;
}

async function ab(scene: SceneData, cam: Cam, gpu?: GpuSceneOptions, maxBounces = 3): Promise<void> {
  const base = await chain(scene, cam, {}, gpu, maxBounces);
  expect(await chain(scene, cam, { MAT_VARIANTS: 1 }, gpu, maxBounces)).toEqual(base);
  const baseFlags = normalizePerfFlags({ ...KERNEL_RELEASE_PERF_FLAGS, MAT_VARIANTS: 0 });
  const rel = await chain(scene, cam, baseFlags, gpu, maxBounces);
  expect(await chain(scene, cam, { ...KERNEL_RELEASE_PERF_FLAGS, MAT_VARIANTS: 1 }, gpu, maxBounces)).toEqual(rel);
  // the frames differ (a degenerate chain would prove nothing)
  expect(new Set(base.map((h) => h.split(':')[1])).size).toBeGreaterThan(1);
}

/** vii_textured with every texture slot kind populated (the base texture reused) and one Principled transmission
 *  material (model 2, Principled glass closure). */
function allSlots(s: SceneData): SceneData {
  const mats: MaterialData[] = s.materials.map((m, i) => {
    const ref = m.baseColorTexture ?? m.metallicRoughnessTexture ?? m.emissiveTexture;
    if (!ref) return m;
    return {
      ...m, specularTexture: ref, specularColorTexture: ref, transmissionTexture: ref, metallicRoughnessTexture: m.metallicRoughnessTexture ?? ref,
      ...(i === 0 ? { transmissionFactor: 0.6 } : {}),
    };
  });
  return { ...s, materials: mats };
}

describe('perf2 WP-8 MAT_VARIANTS A/B (bitwise)', () => {
  it('V1-only package (cornell_i_512)', async () => {
    const { scene, cam } = await pkg('/validation/scenes/cornell_i_512/');
    await ab(scene, cam);
  }, 600_000);

  it('Glass + Refraction nodes (g10_colored_glass_refraction_256)', async () => {
    const { scene, cam } = await pkg('/validation/scenes/g10_colored_glass_refraction_256/');
    await ab(scene, cam, undefined, 4);
  }, 600_000);

  it('textures with KHR_texture_transform (vii_textured_512)', async () => {
    const { scene, cam } = await pkg('/validation/scenes/vii_textured_512/');
    await ab(scene, cam);
  }, 600_000);

  it('every texture slot kind + Principled transmission (vii_textured_512, modified)', async () => {
    const { scene, cam } = await pkg('/validation/scenes/vii_textured_512/');
    await ab(allSlots(scene), cam, undefined, 4);
  }, 600_000);

  it('all-lights fixture with a Principled glass material, CWBVH + MT', async () => {
    const s = allLightsScene();
    s.materials = s.materials.map((m, i) => (i === 6 ? { ...m, transmissionFactor: 0.8 } : m));
    await ab(s, boxCamera(), { bvhKind: 'cwbvh', watertight: false }, 4);
  }, 600_000);

  it('variant keys follow the material table: a scene with a glass material compiles the glass text', async () => {
    const plain = allLightsScene();
    const glass = allLightsScene();
    glass.materials = glass.materials.map((m, i) => (i === 6 ? { ...m, transmissionFactor: 0.8 } : m));
    const a = await restirRig(plain, 16, 16, { preset: 'interactive', perfFlags: { MAT_VARIANTS: 1 } });
    const b = await restirRig(glass, 16, 16, { preset: 'interactive', perfFlags: { MAT_VARIANTS: 1 } });
    const v = await restirRig(plain, 16, 16, { preset: 'interactive', perfFlags: {} });
    expect(a.kernel.defines('rs_initial')).toMatchObject({ MAT_VARIANTS: 1, MAT_NO_G: 1, MAT_NO_PGLASS: 1 });
    expect(b.kernel.defines('rs_initial').MAT_NO_G).toBeUndefined();
    expect(b.kernel.defines('rs_initial').MAT_NO_PGLASS).toBeUndefined();
    for (const key of Object.keys(v.kernel.defines('rs_initial'))) expect(key).not.toMatch(/^(MAT_NO_|MAT_ONLY_|TEX_NO_|MAT_VARIANTS)/);
    a.destroy(); b.destroy(); v.destroy();
  }, 600_000);
});
