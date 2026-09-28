import { describe, expect, it } from 'vitest';
import { convertPunctualLight, loadGltf, rigidMatrix } from '../../src/core/scene/gltf-loader.ts';
import { newDoc, toGlb } from './helpers.ts';

const I4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

// math.md#units-lights: point/spot Φ = cd·4π/683 W, sun E = lux/683 W/m², spot_size = 2·outer, blend = 1 − inner/outer.
describe('KHR_lights_punctual → Blender units', () => {
  it('point: cd → W (I = cd/683 W/sr)', () => {
    const l = convertPunctualLight({ type: 'point', color: [1, 0.5, 0.25], intensity: 100 }, I4, 7, 'p');
    expect(l.type).toBe('point');
    expect(l.power).toBeCloseTo((100 * 4 * Math.PI) / 683, 12);
    expect(l.power / (4 * Math.PI)).toBeCloseTo(100 / 683, 12); // radiant intensity
    expect(l.color).toEqual([1, 0.5, 0.25]);
    expect(l.id).toBe(7);
    expect(l.exposure).toBe(0);
    expect(l.visibleToCamera).toBe(false);
  });
  it('directional: lux → W/m² sun', () => {
    const l = convertPunctualLight({ type: 'directional', color: [1, 1, 1], intensity: 683 }, I4, 0, 's');
    expect(l.type).toBe('sun');
    expect(l.power).toBeCloseTo(1, 12);
  });
  it('spot: cone mapping and glTF defaults', () => {
    const l = convertPunctualLight({ type: 'spot', color: [1, 1, 1], intensity: 54.35, innerConeAngle: Math.PI / 8, outerConeAngle: Math.PI / 4 }, I4, 0, 's');
    expect(l.type).toBe('spot');
    expect(l.power).toBeCloseTo((54.35 * 4 * Math.PI) / 683, 12);
    expect(l.spotSize).toBeCloseTo(Math.PI / 2, 12);
    expect(l.spotBlend).toBeCloseTo(0.5, 12);
    // Blender export: spot_size 45°, blend 0.15 → inner 19.125°, outer 22.5° (scene-io §3.5) → round trip
    const rt = convertPunctualLight({ type: 'spot', color: [1, 1, 1], intensity: 1, innerConeAngle: (19.125 * Math.PI) / 180, outerConeAngle: (22.5 * Math.PI) / 180 }, I4, 0, 's');
    expect((rt.spotSize! * 180) / Math.PI).toBeCloseTo(45, 10);
    expect(rt.spotBlend).toBeCloseTo(0.15, 10);
    const d = convertPunctualLight({ type: 'spot', color: [1, 1, 1], intensity: 1 }, I4, 0, 's');
    expect(d.spotSize).toBeCloseTo(Math.PI / 2, 12); // outer default π/4 → 90°
    expect(d.spotBlend).toBe(1);                      // inner default 0
  });
  it('matrix drops scale and keeps the −Z emission axis and position', () => {
    // R_x(+90°) maps local z to world −y, so the emission axis −Z_local points to world +Y; scale 3; translation (1,2,3)
    const c = 0, s = 1;
    const m = [3, 0, 0, 0, 0, 3 * c, 3 * s, 0, 0, -3 * s, 3 * c, 0, 1, 2, 3, 1]; // R_x(+90°)·S(3), column-major
    const r = rigidMatrix(m);
    expect([...r.subarray(12, 15)]).toEqual([1, 2, 3]);
    expect([-r[8], -r[9], -r[10]]).toEqual([0, 1, 0].map((x) => expect.closeTo(x, 6))); // −Z_local → +Y world
    for (let col = 0; col < 3; col++) expect(Math.hypot(r[col * 4], r[col * 4 + 1], r[col * 4 + 2])).toBeCloseTo(1, 6);
    // right-handed: x × y = z
    const x = [r[0], r[1], r[2]], y = [r[4], r[5], r[6]];
    expect([x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]]).toEqual([r[8], r[9], r[10]].map((v) => expect.closeTo(v, 6)));
  });
  it('loader: lights inherit the node world transform (scaled parent), range warned', async () => {
    const { doc, scene } = newDoc();
    const { KHRLightsPunctual } = await import('@gltf-transform/extensions');
    const ext = doc.createExtension(KHRLightsPunctual);
    const spot = ext.createLight('spot1').setType('spot').setIntensity(683 / (4 * Math.PI)).setInnerConeAngle(0.2).setOuterConeAngle(0.4).setRange(10);
    const sun = ext.createLight('sun1').setType('directional').setIntensity(2 * 683);
    const parent = doc.createNode('parent').setScale([2, 2, 2]).setTranslation([0, 1, 0]);
    const child = doc.createNode('child').setTranslation([1, 0, 0]).setRotation([Math.SQRT1_2, 0, 0, Math.SQRT1_2]).setExtension('KHR_lights_punctual', spot);
    parent.addChild(child);
    scene.addChild(parent);
    scene.addChild(doc.createNode('sunNode').setExtension('KHR_lights_punctual', sun));
    const { scene: s } = await loadGltf({ kind: 'glb', bytes: await toGlb(doc) });
    expect(s.lights.map((l) => [l.name, l.type, l.id])).toEqual([['spot1', 'spot', 0], ['sun1', 'sun', 1]]);
    const L = s.lights[0];
    expect(L.power).toBeCloseTo(1, 6);
    expect(L.spotSize).toBeCloseTo(0.8, 6);
    expect(L.spotBlend).toBeCloseTo(0.5, 6);
    expect([...L.matrix.subarray(12, 15)]).toEqual([2, 1, 0].map((x) => expect.closeTo(x, 6)));
    expect([-L.matrix[8], -L.matrix[9], -L.matrix[10]]).toEqual([0, 1, 0].map((x) => expect.closeTo(x, 6)));
    expect(s.lights[1].power).toBeCloseTo(2, 6);
    expect(s.warnings.some((w) => /range/.test(w))).toBe(true);
  });
});
