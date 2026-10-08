// M7 E2E-GLB asset (PLAN §7.2 "Cornell with point + spot (inner ≠ 0)"; docs/decisions/m7-api.md §3.4): cornell.glb (no
// lights: glTF has no area lights; its rect lives in cornell.meta.json) plus two KHR_lights_punctual lights: a point light (intensity in candela, glTF / SPEC import convention) and a spot
// with innerConeAngle 0.3 ≠ 0 and outerConeAngle 0.6. Deterministic (gltf-transform writes the same bytes).
//   npx tsx validation/assets/cornell/make_cornell_point_spot.ts
import { readFileSync, writeFileSync } from 'node:fs';
import { WebIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, KHRLightsPunctual } from '@gltf-transform/extensions';

const io = new WebIO().registerExtensions(ALL_EXTENSIONS);
const doc = await io.readBinary(new Uint8Array(readFileSync(new URL('./cornell.glb', import.meta.url))));
const ext = doc.createExtension(KHRLightsPunctual);
const scene = doc.getRoot().getDefaultScene() ?? doc.getRoot().listScenes()[0];
const point = ext.createLight('point_key').setType('point').setIntensity(0.6).setColor([1, 0.95, 0.85]);
scene.addChild(doc.createNode('PointKey').setTranslation([0.12, 0.42, 0.05]).setExtension('KHR_lights_punctual', point));
const spot = ext.createLight('spot_fill').setType('spot').setIntensity(2.0).setColor([0.85, 0.9, 1]).setInnerConeAngle(0.3).setOuterConeAngle(0.6);
// glTF spot points along −Z of its node: aim from the front-left down toward the back-right corner
const q = (() => {   // rotation taking −Z to d
  const d = [0.5, -0.55, -0.67], l = Math.hypot(...d);
  const v = d.map((x) => x / l), z = [0, 0, -1];
  const ax = [z[1] * v[2] - z[2] * v[1], z[2] * v[0] - z[0] * v[2], z[0] * v[1] - z[1] * v[0]];
  const s = Math.hypot(...ax), c = z[0] * v[0] + z[1] * v[1] + z[2] * v[2];
  const ang = Math.atan2(s, c), h = Math.sin(ang / 2) / s;
  return [ax[0] * h, ax[1] * h, ax[2] * h, Math.cos(ang / 2)] as [number, number, number, number];
})();
scene.addChild(doc.createNode('SpotFill').setTranslation([-0.18, 0.48, 0.2]).setRotation(q).setExtension('KHR_lights_punctual', spot));
writeFileSync(new URL('./cornell_point_spot.glb', import.meta.url), await io.writeBinary(doc));
console.log('wrote validation/assets/cornell/cornell_point_spot.glb');
