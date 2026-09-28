// Minimal UsdSceneSource (plan §5 M2; docs/decisions/usd.md): LightUSD next backend + adapter, run inline in Node.
// Checks counts, frames/units, the plan §1.2 USD light rules (independent re-derivation from the pxr dumps in
// validation/out/usd-spike when present) and Blender's truth.json, and the Cornell USD against the Cornell GLB.
import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { loadGltf } from '../../src/core/scene/gltf-loader.ts';
import { loadUsdInline, type UsdLoadResult } from '../../src/core/scene/usd/load-usd.ts';
import { blenderBlackbody, convertUsdLight, mul4, stageMatrix } from '../../src/core/scene/usd/usd-lights.ts';
import type { LightData } from '../../src/core/scene/types.ts';

const SPIKE = 'validation/assets/usd-spike';
const PXR = 'validation/out/usd-spike';
const load = async (f: string): Promise<UsdLoadResult> => loadUsdInline(new Uint8Array(readFileSync(f)), f.split('/').pop()!);
const pos = (l: LightData) => [l.matrix[12], l.matrix[13], l.matrix[14]];

interface PxrLight {
  path: string; type: string; intensity: number; exposure: number; color: number[]; normalize: boolean;
  enableColorTemperature: boolean; colorTemperature: number; radius: number | null; width: number | null; height: number | null;
  angle: number | null; treatAsPoint: boolean | null; shaping: { coneAngle: number; coneSoftness: number } | null; world: number[];
}

/** Independent re-derivation of plan §1.2 / math.md#units-lights for one pxr light (metres = authored · scale · mpu). */
function expectedLight(l: PxrLight, mpu: number, blender: boolean) {
  const scale = (c: number) => Math.hypot(l.world[4 * c], l.world[4 * c + 1], l.world[4 * c + 2]) * mpu;
  const bb = l.enableColorTemperature ? blenderBlackbody(l.colorTemperature) : [1, 1, 1];
  const color = l.color.map((c, k) => c * bb[k]);
  const i = l.intensity;
  if (l.type === 'rect') {
    const A = l.width! * scale(0) * l.height! * scale(1);
    return { type: 'rect', power: l.normalize ? Math.PI * i : i * Math.PI * A, color, sizeX: l.width! * scale(0), sizeY: l.height! * scale(1) };
  }
  if (l.type === 'disk') {
    const d = 2 * l.radius! * scale(0);
    return { type: 'disk', power: l.normalize ? Math.PI * i : i * Math.PI * (Math.PI / 4) * d * d, color, sizeX: d };
  }
  if (l.type === 'distant') return { type: 'sun', power: (blender ? 4 : 1) * i, color };
  const r = (l.radius ?? 0.5) * scale(0);
  const I = l.normalize ? i / 4 : l.treatAsPoint || r === 0 ? i : i * Math.PI * r * r;
  return { type: l.shaping ? 'spot' : 'point', power: 4 * Math.PI * I, color, spotSize: l.shaping ? 2 * l.shaping.coneAngle * Math.PI / 180 : undefined, spotBlend: l.shaping?.coneSoftness };
}

describe('USD loader (LightUSD next + adapter)', () => {
  it('blackbody matches Blender 5.1.2 Light.temperature_color', () => {
    const ref = JSON.parse(readFileSync('tests/scene/blender-blackbody.json', 'utf8')).values as Record<string, number[]>;
    for (const [t, v] of Object.entries(ref)) {
      const c = blenderBlackbody(Number(t));
      for (let k = 0; k < 3; k++) expect(Math.abs(c[k] - v[k]), `${t} K ch${k}`).toBeLessThan(1e-5 * Math.max(1, Math.abs(v[k])));
    }
  });

  it('stage matrix: Z-up → R_x(−90°), metersPerUnit scale', () => {
    const W = stageMatrix('Z', 0.01);
    const p = mul4(W, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 100, 200, 300, 1]);
    expect([p[12], p[13], p[14]].map((x) => +x.toFixed(12))).toEqual([1, 3, -2]); // (x, z, −y)·0.01
  });

  it('light rules: sphere normalize true/false, treatAsPoint, spot cone, distant quirk, disk/rect areas', () => {
    const I = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const base = { primPath: '/L', intensity: 10, exposure: 0, color: [1, 1, 1] as [number, number, number], enableColorTemperature: false, colorTemperature: 6500 };
    const conv = (o: object, blender = false) => convertUsdLight({ ...base, type: 'sphere', normalize: false, ...o }, I, 0, { blenderAuthored: blender }).light!;
    expect(conv({ normalize: true, radius: 0.5 }).power).toBeCloseTo(Math.PI * 10, 12);             // I = i/4
    expect(conv({ radius: 0.5 }).power).toBeCloseTo(4 * Math.PI * Math.PI * 0.25 * 10, 10);        // I = i·πr²
    expect(conv({ radius: 0.5, treatAsPoint: true }).power).toBeCloseTo(4 * Math.PI * 10, 10);     // I = i
    expect(conv({ radius: 0 }).power).toBeCloseTo(4 * Math.PI * 10, 10);
    expect(conv({ radius: 0.5 }).simplified).toMatch(/→ point/);
    const spot = conv({ radius: 0, shaping: { coneAngle: 30, coneSoftness: 0.2, focus: 0 } });
    expect(spot).toMatchObject({ type: 'spot', spotBlend: 0.2 });
    expect(spot.spotSize).toBeCloseTo(Math.PI / 3, 12);
    expect(conv({ type: 'distant', angle: 0.53 }, true)).toMatchObject({ type: 'sun', power: 40 });
    expect(conv({ type: 'distant', angle: 0 }, false)).toMatchObject({ type: 'sun', power: 10 });
    expect(conv({ type: 'rect', width: 2, height: 3 }).power).toBeCloseTo(Math.PI * 10 * 6, 10); // L_e = i
    expect(conv({ type: 'rect', width: 2, height: 3, normalize: true }).power).toBeCloseTo(Math.PI * 10, 10);
    expect(conv({ type: 'disk', radius: 1 }).power).toBeCloseTo(Math.PI * 10 * Math.PI, 10);
    expect(convertUsdLight({ ...base, type: 'cylinder', normalize: false }, I, 0, { blenderAuthored: false }).light).toBeNull();
  });

  const files = ['spike_hand.usda', 'spike_hand.usdc', 'spike_hand.usdz', 'spike_blender.usda', 'spike_blender.usdc', 'spike_blender.usdz', 'spike_blender_yup.usda'];
  for (const f of files) {
    const pxrFile = `${PXR}/pxr.${f}.json`;
    it.skipIf(!existsSync(pxrFile))(`${f}: lights and stage match the pxr dump under the plan rules`, async () => {
      const pxr = JSON.parse(readFileSync(pxrFile, 'utf8')) as { stage: { upAxis: string; metersPerUnit: number; doc?: string }; lights: PxrLight[]; draws: { triangles: number }[] };
      const r = await load(`${SPIKE}/${f}`);
      const blender = /^Blender v/.test(pxr.stage.doc ?? '');
      expect(r.stats).toMatchObject({ upAxis: pxr.stage.upAxis, metersPerUnit: pxr.stage.metersPerUnit, blenderAuthored: blender });
      const W = stageMatrix(pxr.stage.upAxis, pxr.stage.metersPerUnit);
      expect(r.scene.lights.length).toBe(pxr.lights.length);
      for (const pl of pxr.lights) {
        const name = pl.path.split('/').pop()!;
        const ours = r.scene.lights.find((l) => l.name === name)!;
        expect(ours, name).toBeDefined();
        const e = expectedLight(pl, pxr.stage.metersPerUnit, blender);
        expect(ours.type, name).toBe(e.type);
        expect(ours.power / e.power, `${name} power`).toBeCloseTo(1, 5);
        expect(ours.exposure, name).toBe(pl.exposure);
        ours.color.forEach((c, k) => expect(c, `${name} color`).toBeCloseTo(e.color[k], 5));
        if (e.sizeX !== undefined) expect(ours.sizeX!, `${name} sizeX`).toBeCloseTo(e.sizeX, 5);
        if (e.sizeY !== undefined) expect(ours.sizeY!, `${name} sizeY`).toBeCloseTo(e.sizeY, 5);
        if (e.spotSize !== undefined) { expect(ours.spotSize!).toBeCloseTo(e.spotSize, 5); expect(ours.spotBlend!).toBeCloseTo(e.spotBlend!, 6); }
        const t = mul4(W, pl.world);
        pos(ours).forEach((x, k) => expect(x, `${name} position`).toBeCloseTo(t[12 + k], 5));
        // emission axis −Z_obj survives the frame change (rigid part only)
        const z = [t[8], t[9], t[10]], zl = Math.hypot(...z);
        [8, 9, 10].forEach((k, j) => expect(ours.matrix[k], `${name} axis`).toBeCloseTo(z[j] / zl, 5));
      }
      if (!f.startsWith('spike_hand')) { // no PointInstancer: triangle count = pxr drawn triangles
        expect(r.scene.geometry.indices.length / 3).toBe(pxr.draws.reduce((s, d) => s + d.triangles, 0));
      }
    });
  }

  it('spike_hand: PointInstancer draws, units (cm → m), UV flip, GeomSubset materials, warnings', async () => {
    const r = await load(`${SPIKE}/spike_hand.usda`);
    expect(r.stats).toMatchObject({ pointInstanceDraws: 3, upAxis: 'Y', metersPerUnit: 0.01 });
    const s = r.scene;
    expect(new Set(s.materials.map((m) => m.name))).toEqual(new Set(['Red', 'Steel', 'Glow']));
    const glow = s.materials.find((m) => m.name === 'Glow')!;
    expect(glow).toMatchObject({ emissiveFactor: [2, 1.5, 0.5], alphaMode: 'MASK' });
    expect(glow.alphaCutoff).toBeCloseTo(0.25, 6);
    expect(glow.ior).toBeCloseTo(1.45, 6);
    expect(s.warnings.join('\n')).toMatch(/clearcoat ignored/);
    expect(s.warnings.join('\n')).toMatch(/shaping:focus/);
    for (let k = 0; k < 3; k++) expect(s.bounds.max[k] - s.bounds.min[k]).toBeLessThan(10); // metres, not cm
    const all = [...s.geometry.uv0];
    expect(all.every((v) => v >= -1e-6 && v <= 1 + 1e-6)).toBe(true);
  });

  it('Cornell USD (Blender Z-up) matches the Cornell GLB geometry, camera and meta light', async () => {
    const meta = JSON.parse(readFileSync('validation/assets/cornell/cornell.meta.json', 'utf8'));
    const glb = (await loadGltf({ kind: 'glb', bytes: new Uint8Array(readFileSync('validation/assets/cornell/cornell.glb')) }, { tangents: false })).scene;
    for (const f of ['cornell.usda', 'cornell.usdc']) {
      const { scene: u } = await load(`validation/assets/cornell/${f}`);
      expect(u.geometry.indices.length, f).toBe(glb.geometry.indices.length);
      for (let k = 0; k < 3; k++) {
        expect(u.bounds.min[k]).toBeCloseTo(glb.bounds.min[k], 5);
        expect(u.bounds.max[k]).toBeCloseTo(glb.bounds.max[k], 5);
      }
      // total area per material colour is identical
      const areaBy = (s: typeof u) => {
        const acc = new Map<string, number>();
        const g = s.geometry;
        for (let t = 0; t < g.indices.length / 3; t++) {
          const [a, b, c] = [g.indices[3 * t], g.indices[3 * t + 1], g.indices[3 * t + 2]].map((i) => [0, 1, 2].map((k) => g.positions[3 * i + k]));
          const e = [0, 1, 2].map((k) => b[k] - a[k]), h = [0, 1, 2].map((k) => c[k] - a[k]);
          const area = 0.5 * Math.hypot(e[1] * h[2] - e[2] * h[1], e[2] * h[0] - e[0] * h[2], e[0] * h[1] - e[1] * h[0]);
          const key = s.materials[g.triMaterial[t]].baseColorFactor.slice(0, 3).map((x) => x.toFixed(3)).join(',');
          acc.set(key, (acc.get(key) ?? 0) + area);
        }
        return [...acc.entries()].sort().map(([k, v]) => [k, +v.toFixed(6)]);
      };
      expect(areaBy(u)).toEqual(areaBy(glb));
      const cam = u.cameras[0];
      expect(cam.yfov * 180 / Math.PI).toBeCloseTo(meta.camera.vfov_deg, 4);
      [12, 13, 14].forEach((k, j) => expect(cam.matrix[k]).toBeCloseTo(meta.camera.position[j], 5));
      [8, 9, 10].forEach((k, j) => expect(-cam.matrix[k]).toBeCloseTo(meta.camera.forward[j], 5));
      const L = meta.lights[0];
      const l = u.lights[0];
      expect(l).toMatchObject({ type: 'rect', name: 'ceiling_light', visibleToCamera: false });
      expect(l.power).toBeCloseTo(L.power_W, 5);
      expect(l.sizeX!).toBeCloseTo(L.size_x, 6);
      expect(l.sizeY!).toBeCloseTo(L.size_along_minus_z_gltf, 6);
      [12, 13, 14].forEach((k, j) => expect(l.matrix[k]).toBeCloseTo(L.position[j], 6));
      [8, 9, 10].forEach((k, j) => expect(-l.matrix[k]).toBeCloseTo(L.normal[j], 6)); // emits along −Z_obj
      // Blender writes enableColorTemperature = 1 (6500 K default) on every light: applied like Blender's importer.
      l.color.forEach((c, k) => expect(c).toBeCloseTo(blenderBlackbody(6500)[k], 6));
    }
  });
});
