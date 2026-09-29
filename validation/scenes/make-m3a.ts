// M3a Gate-2 scene packages (plan §5 M3a exit, §7.2): calibration C0c/C0e/C0f/C0g/C0l/C0m/C0n, validation scenes
// (ii)–(v), (vii), (viii), (x)–(xii) and the keyframed dynamic sequences ix-a…g. C0d and (i) come from
// make-spot-c0d.ts / make-cornell-i.ts; C0a/C0b/C0p from validation/blender/calib_scenes.py; C0o (visibleToCamera) is
// Mode B only and is deferred to M3b (see docs/decisions/validation-m3a.md).
//   npx tsx validation/scenes/make-m3a.ts [outDir] [--only name,name]
// Every scene is flat shaded, Mode A (analytic lights NEE-only, not camera-visible), no env. Analytic answers are
// written to scene.json "expected" where a closed form exists (furnaces: constant; direct lighting on a plane:
// formula + parameters). Deterministic: re-running writes byte-identical packages (the gate checks this).
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { Animation, exportFrames, lightTarget, matrixToPoseQ, presetSweep, quatAxisAngle, quatLookDir, quatMul, quatNormalize, type Quat } from '../../src/core/scene/animation.ts';
import { loadUsdInline } from '../../src/core/scene/usd/load-usd.ts';
import type { LightData, MaterialData, SceneData, TextureData } from '../../src/core/scene/types.ts';
import {
  MeshBuilder, ROOT, cornellBase, deg, dirFromAngles, light, lightToward, lookAt, norm, principled, resetLightIds, rng,
  sceneOf, scale, texTransform, texture, v1, writePackage, type V3,
} from './scene-kit.ts';

const argv = process.argv.slice(2);
const onlyArg = argv.indexOf('--only');
const ONLY = onlyArg >= 0 ? new Set(argv[onlyArg + 1].split(',')) : undefined;
const OUT = argv[0] && !argv[0].startsWith('--') ? path.resolve(argv[0]) : path.join(ROOT, 'validation/scenes');
const SRC = (what: string) => ({ uri: `validation/scenes/make-m3a.ts (${what})` });

/** Test frames of the ix sequences: every 8th frame of 0..48 at 24 fps (plan §5 M3a exit: "the keyframes of ix-a…g"). */
export const IX_FRAMES = [0, 8, 16, 24, 32, 40, 48];
const IX_FPS = 24;

type Gen = () => Promise<void>;
const gens: [string, Gen][] = [];
const def = (name: string, g: Gen) => gens.push([name, g]);

/** Half size of the analytic scenes' floor: its edge lies beyond the horizon region, so the analytic image has no
 *  coverage discontinuity (analytic_check.py supersamples smooth radiance only). */
const FLOOR = 50;
const floorCam = (res: number) => ({ camera: { matrix: lookAt([0, 2.2, 2.4], [0, 0, -0.3]), yfov: 50 * deg }, render: { width: res, height: res, maxBounces: 0 } });

// ================================================ calibration =======================================================

def('c0c_point_256', async () => {
  resetLightIds();
  const rho = 0.5;
  const mb = new MeshBuilder().floor(-FLOOR, FLOOR, -FLOOR, FLOOR, 0, 0);
  const P = 60, pos: V3 = [0.3, 1.0, -0.2];
  const L = light('point', new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...pos, 1]), P, { color: [1, 0.9, 0.8] });
  const scene = sceneOf('c0c_point_256', mb, [v1('floor', { diffuse: [rho, rho, rho] })], [L]);
  await writePackage(OUT, scene, {
    ...floorCam(256), source: SRC('C0c point light over a Lambert floor'),
    extra: { expected: { kind: 'direct-plane', light: 'point', formula: 'L(x) = rho/pi * color*P/(4 pi) * h/d^3 (h = light height, d = |x - c|)', rho, power: P, color: L.color, position: pos, plane: 'y = 0', plane_half_size: FLOOR } },
  });
});

def('c0e_rect_256', async () => {
  resetLightIds();
  const rho = 0.5;
  const mb = new MeshBuilder().floor(-FLOOR, FLOOR, -FLOOR, FLOOR, 0, 0);
  // emission axis tilted 55° from straight down toward +Z: floor points with (x − c)·a_L ≤ 0 (z ≲ −0.35) are behind
  // the one-sided emitter and must be exactly black at b = 0.
  const aL = norm([0, -Math.cos(55 * deg), Math.sin(55 * deg)]);
  const c: V3 = [0, 0.5, 0];
  const L = light('rect', lightToward(aL, c), 30, { sizeX: 0.6, sizeY: 0.3, color: [0.9, 1, 1] });
  const scene = sceneOf('c0e_rect_256', mb, [v1('floor', { diffuse: [rho, rho, rho] })], [L]);
  await writePackage(OUT, scene, {
    ...floorCam(256), source: SRC('C0e one-sided rect light, tilted 55 deg'),
    extra: { expected: { kind: 'direct-plane', light: 'rect', formula: 'L(x) = rho/pi * L_e * E_polygon(x) for (x - c).a_L > 0, else 0; L_e = color*P/(pi*sx*sy)', rho, power: 30, center: c, axis: aL, one_sided_black_region: 'z < -0.35 (behind the emitter plane)', plane_half_size: FLOOR } },
  });
});

for (const b of [0, 1, 3, 7]) {
  def(`c0f_furnace_b${b}_256`, async () => {
    resetLightIds();
    const rho: V3 = [0.7, 0.5, 0.3], Le: V3 = [0.5, 0.5, 0.5];
    const mb = new MeshBuilder().box([-1, -1, -1], [1, 1, 1], 0, { inward: true });
    const scene = sceneOf(`c0f_furnace_b${b}_256`, mb, [v1('furnace', { diffuse: rho, emission: Le })], []);
    const value = rho.map((r, k) => Le[k] * (1 - r ** (b + 2)) / (1 - r));
    await writePackage(OUT, scene, {
      camera: { matrix: lookAt([0.3, 0.1, 0.4], [-1, -0.5, -1]), yfov: 75 * deg }, render: { width: 256, height: 256, maxBounces: b },
      source: SRC(`C0f closed emissive Lambert cube, b = ${b}`),
      extra: { expected: { kind: 'constant', value, formula: 'L_e (1 - rho^(b+2)) / (1 - rho)', rho, Le, maxBounces: b, statistic: 'every pixel (and the image mean)' } },
    });
  });
}

def('c0g_bsdf_furnace_512x256', async () => {
  resetLightIds();
  const mats: MaterialData[] = [v1('enclosure', { diffuse: [0, 0, 0], emission: [1, 1, 1] })];
  const mb = new MeshBuilder().icosphere([0, 0, 0], 20, 3, 0, { inward: true });
  const rough = [0.05, 0.2, 0.5, 1.0];
  rough.forEach((r, i) => {
    mats.push(v1(`ggx_r${r}`, { diffuse: [0, 0, 0], glossy: [1, 1, 1], roughness: r, mix: 1 }));
    mb.icosphere([-2 + i, 0, 0], 0.42, 4, mats.length - 1);
  });
  mats.push(v1('lambert', { diffuse: [0.8, 0.8, 0.8] }));
  mb.icosphere([2, 0, 0], 0.42, 4, mats.length - 1);
  const scene = sceneOf('c0g_bsdf_furnace_512x256', mb, mats, []);
  await writePackage(OUT, scene, {
    camera: { matrix: lookAt([0, 0, 5.5], [0, 0, 0]), yfov: 28 * deg }, render: { width: 512, height: 256, maxBounces: 0 },
    source: SRC('C0g BSDF furnace: V1 GGX r 0.05/0.2/0.5/1.0 + Lambert 0.8 inside an L = 1 emissive sphere'),
    extra: { expected: { kind: 'bsdf-furnace', formula: 'pixel = directional albedo of the sphere BSDF at the view angle (minus mutual occlusion); background = 1 exactly', enclosure_Le: 1, roughness: rough, lambert: 0.8 } },
  });
});

def('c0l_sun_256', async () => {
  resetLightIds();
  const rho: V3 = [0.6, 0.5, 0.4];
  const mb = new MeshBuilder().floor(-3, 3, -3, 3, 0, 0).box([-0.25, 0, -0.55], [0.25, 0.5, -0.05], 1, { rotY: 20 * deg, omit: ['-y'] });
  const toSun = dirFromAngles(40 * deg, 30 * deg);
  const L = light('sun', lightToward(scale(toSun, -1), [0, 5, 0]), 3, { color: [1, 0.95, 0.9] });
  const scene = sceneOf('c0l_sun_256', mb, [v1('floor', { diffuse: rho }), v1('box', { diffuse: [0.3, 0.4, 0.6] })], [L]);
  await writePackage(OUT, scene, {
    ...floorCam(256), source: SRC('C0l sun, elevation 40 deg, over a floor with a shadow-casting box'),
    extra: { expected: { kind: 'direct-plane', light: 'sun', formula: 'unshadowed floor: L = rho/pi * color*E * sin(elevation)', rho, irradiance: 3, color: L.color, elevation_deg: 40, value_unshadowed: rho.map((r, k) => r / Math.PI * L.color[k] * 3 * Math.sin(40 * deg)) } },
  });
});

def('c0m_disk_256', async () => {
  resetLightIds();
  const rho = 0.5;
  const mb = new MeshBuilder().floor(-FLOOR, FLOOR, -FLOOR, FLOOR, 0, 0);
  const aL = norm([Math.sin(25 * deg), -Math.cos(25 * deg), 0]);
  const L = light('disk', lightToward(aL, [0, 0.6, 0]), 20, { sizeX: 0.5, color: [1, 1, 0.9] });
  const scene = sceneOf('c0m_disk_256', mb, [v1('floor', { diffuse: [rho, rho, rho] })], [L]);
  await writePackage(OUT, scene, {
    ...floorCam(256), source: SRC('C0m disk light, diameter 0.5 m, tilted 25 deg'),
    extra: { expected: { kind: 'direct-plane', light: 'disk', formula: 'L(x) = rho/pi * L_e * E_disk(x), L_e = color*P/(pi * pi/4 * d^2), one-sided', rho, power: 20, diameter: 0.5, plane_half_size: FLOOR } },
  });
});

def('c0n_spread_256', async () => {
  resetLightIds();
  const rho = 0.5;
  const mb = new MeshBuilder().floor(-3, 3, -3, 3, 0, 0);
  const tilt = (s: number): V3 => norm([s * Math.sin(20 * deg), -Math.cos(20 * deg), 0]);
  const lights = [
    light('rect', lightToward(tilt(-1), [-0.8, 0.5, 0]), 20, { sizeX: 0.4, sizeY: 0.4, spread: 30 * deg }),
    light('rect', lightToward(tilt(1), [0.8, 0.5, 0]), 20, { sizeX: 0.4, sizeY: 0.4, spread: 90 * deg, color: [1, 0.9, 0.8] }),
  ];
  const scene = sceneOf('c0n_spread_256', mb, [v1('floor', { diffuse: [rho, rho, rho] })], lights);
  await writePackage(OUT, scene, {
    ...floorCam(256), source: SRC('C0n rect lights with spread 30 and 90 deg'),
    extra: { expected: { kind: 'direct-plane', light: 'rect+spread', formula: 'L(x) = rho/pi * int L_e spread(theta) cos_z cos_x / r^2 dA (math.md#units-lights spread)', rho, spreads_deg: [30, 90] } },
  });
});

// ================================================ validation scenes ==================================================

def('ii_cornell_point_512', async () => {
  resetLightIds();
  const c = await cornellBase();
  const L = light('point', new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0.45, 0.02, 1]), 4, { color: [1, 0.92, 0.85] });
  const mb = new MeshBuilder().append(c.geometry);
  const scene = sceneOf('ii_cornell_point_512', mb, c.materials, [L]);
  await writePackage(OUT, scene, { camera: c.camera, render: { width: 512, height: 512, maxBounces: 3 }, source: { uri: 'validation/assets/cornell/cornell.glb (validation/scenes/make-m3a.ts (ii))', sha256: c.glbSha } });
});

def('iii_spot_grazing_512', async () => {
  resetLightIds();
  const mats = [v1('floor', { diffuse: [0.6, 0.6, 0.6] }), v1('wall', { diffuse: [0.7, 0.6, 0.5] }), v1('box', { diffuse: [0.4, 0.5, 0.7] })];
  const mb = new MeshBuilder().floor(-2, 2, -1, 1.2, 0, 0)
    .quad([-2, 0, -1], [2, 0, -1], [2, 2, -1], [-2, 2, -1], 1)
    .box([0.2, 0, -0.6], [0.8, 0.5, 0], 2, { rotY: 20 * deg, omit: ['-y'] });
  const pos: V3 = [-1.7, 0.22, -0.88];
  const L = light('spot', lightToward(norm([3.3, -0.17, 0.1]), pos), 150, { spotSize: 70 * deg, spotBlend: 0.25, color: [1, 0.95, 0.85] });
  const scene = sceneOf('iii_spot_grazing_512', mb, mats, [L]);
  await writePackage(OUT, scene, { camera: { matrix: lookAt([0.3, 1.4, 2.4], [0, 0.2, -0.6]), yfov: 50 * deg }, render: { width: 512, height: 512, maxBounces: 3 }, source: SRC('(iii) spot grazing wall + floor + box') });
});

def('iv_emissive_mesh_512', async () => {
  resetLightIds();
  const c = await cornellBase();
  const mats = [...c.materials, v1('emitter', { diffuse: [0.2, 0.2, 0.2], emission: [6, 4.5, 3] })];
  const mb = new MeshBuilder().append(c.geometry).icosphere([0.1, 0.36, -0.1], 0.05, 2, mats.length - 1);
  const L = light('rect', c.rect.matrix, 1, { sizeX: c.rect.sizeX, sizeY: c.rect.sizeY });
  const scene = sceneOf('iv_emissive_mesh_512', mb, mats, [L]);
  await writePackage(OUT, scene, { camera: c.camera, render: { width: 512, height: 512, maxBounces: 3 }, source: { uri: 'cornell.glb + emissive icosphere (validation/scenes/make-m3a.ts (iv))', sha256: c.glbSha } });
});

/** (v) room: floor, back and side walls, a key light and a small side light. Analytic rect lights (Mode A: NEE only),
 *  or — for the near-specular r ≤ 0.1 spheres, whose highlights NEE alone cannot resolve at any affordable spp — the
 *  same two rectangles as one-sided-facing emissive quads (two-sided emitters, MIS with BSDF sampling). */
function glossyRoom(mb: MeshBuilder, mats: MaterialData[], emissive = false): LightData[] {
  const m0 = mats.length;
  mats.push(v1('floor', { diffuse: [0.5, 0.5, 0.5] }), v1('back', { diffuse: [0.6, 0.6, 0.7] }), v1('left', { diffuse: [0.6, 0.3, 0.25] }), v1('right', { diffuse: [0.25, 0.5, 0.3] }));
  mb.floor(-2.5, 2.5, -1.5, 1.5, 0, m0)
    .quad([-2.5, 0, -1.5], [2.5, 0, -1.5], [2.5, 2.5, -1.5], [-2.5, 2.5, -1.5], m0 + 1)
    .quad([-2.5, 0, 1.5], [-2.5, 0, -1.5], [-2.5, 2.5, -1.5], [-2.5, 2.5, 1.5], m0 + 2)
    .quad([2.5, 0, -1.5], [2.5, 0, 1.5], [2.5, 2.5, 1.5], [2.5, 2.5, -1.5], m0 + 3);
  const key = { m: lightToward(norm([0, -Math.cos(20 * deg), -Math.sin(20 * deg)]), [0, 2.0, 0.6]), P: 80, sx: 1.2, sy: 0.6, color: [1, 1, 1] as V3 };
  const side = { m: lightToward(norm([-1, -0.3, -1]), [1.6, 1.0, 1.0]), P: 15, sx: 0.3, sy: 0.3, color: [1, 0.85, 0.7] as V3 };
  if (!emissive) {
    return [light('rect', key.m, key.P, { sizeX: key.sx, sizeY: key.sy }), light('rect', side.m, side.P, { sizeX: side.sx, sizeY: side.sy, color: side.color })];
  }
  for (const l of [key, side]) {
    // same rectangle and front-face radiance as the analytic light: L_e = color·P/(π·A)
    const Le = l.P / (Math.PI * l.sx * l.sy);
    mats.push(v1(`emitter_${mats.length}`, { diffuse: [0, 0, 0], emission: [l.color[0] * Le, l.color[1] * Le, l.color[2] * Le] }));
    const X: V3 = [l.m[0], l.m[1], l.m[2]], Y: V3 = [l.m[4], l.m[5], l.m[6]], c: V3 = [l.m[12], l.m[13], l.m[14]];
    const p = (u: number, v: number): V3 => [c[0] + u * X[0] + v * Y[0], c[1] + u * X[1] + v * Y[1], c[2] + u * X[2] + v * Y[2]];
    // winding so the face normal is the emission axis −Z_obj (irrelevant for two-sided emission; keeps orientation)
    mb.quad(p(-l.sx / 2, -l.sy / 2), p(-l.sx / 2, l.sy / 2), p(l.sx / 2, l.sy / 2), p(l.sx / 2, -l.sy / 2), mats.length - 1);
  }
  return [];
}

const V1_GLOSSY = (r: number) => v1(`v1_ggx_r${r}`, { diffuse: [0.2, 0.3, 0.5], glossy: [0.95, 0.9, 0.8], roughness: r, mix: 0.7 });

for (const [name, rough, note, emissive] of [
  ['v_glossy_v1_sharp_512', [0.05, 0.1], 'tile delta 3% (plan: r <= 0.1); emissive-quad lights (MIS)', true],
  ['v_glossy_v1_512', [0.2, 0.3, 0.5, 0.8], 'tile delta 2%', false],
] as const) {
  def(name, async () => {
    resetLightIds();
    const mats: MaterialData[] = [];
    const mb = new MeshBuilder();
    const lights = glossyRoom(mb, mats, emissive);
    const n = rough.length;
    rough.forEach((r, i) => { mats.push(V1_GLOSSY(r)); mb.icosphere([(i - (n - 1) / 2) * 0.95, 0.35, -0.2], 0.35, 4, mats.length - 1); });
    const scene = sceneOf(name, mb, mats, lights);
    await writePackage(OUT, scene, {
      camera: { matrix: lookAt([0, 1.1, 2.6], [0, 0.35, -0.2]), yfov: (n === 2 ? 32 : 45) * deg }, render: { width: 512, height: 512, maxBounces: 3 },
      source: SRC(`(v) glossy sweep V1 GGX r = ${rough.join('/')}; ${note}`),
    });
  });
}

def('v_glossy_v2_512', async () => {
  resetLightIds();
  const mats: MaterialData[] = [];
  const mb = new MeshBuilder();
  const lights = glossyRoom(mb, mats);
  const dielectric: V3[] = [[0.8, 0.2, 0.2], [0.2, 0.6, 0.3], [0.2, 0.3, 0.8], [0.9, 0.8, 0.3], [0.7, 0.7, 0.7], [0.5, 0.3, 0.7]];
  const metal: V3[] = [[1.0, 0.78, 0.34], [0.95, 0.64, 0.54], [0.95, 0.93, 0.88], [0.56, 0.57, 0.58], [0.91, 0.92, 0.92], [0.8, 0.5, 0.9]];
  const combos: [number, number][] = [];
  for (const r of [0.2, 0.5, 0.8]) for (const lvl of [0.5, 0.8]) combos.push([r, lvl]);
  for (const metallic of [0, 1]) {
    combos.forEach(([r, lvl], i) => {
      const tint: V3 = lvl === 0.8 ? [1, 0.85, 0.7] : [1, 1, 1];
      const base = (metallic ? metal : dielectric)[i];
      mats.push(principled(`p_m${metallic}_r${r}_s${lvl}`, { baseColorFactor: [...base, 1], metallicFactor: metallic, roughnessFactor: r, ior: 1.5, specularLevel: lvl, specularColorFactor: tint }));
      mb.icosphere([(i - 2.5) * 0.72, 0.27, metallic ? -0.75 : 0.05], 0.27, 4, mats.length - 1);
    });
  }
  const scene = sceneOf('v_glossy_v2_512', mb, mats, lights);
  await writePackage(OUT, scene, {
    camera: { matrix: lookAt([0, 1.3, 2.8], [0, 0.25, -0.35]), yfov: 45 * deg }, render: { width: 512, height: 512, maxBounces: 3 },
    source: SRC('(v) V2 Principled sweep: metallic 0/1 x roughness 0.2/0.5/0.8 x specular level 0.5/0.8 (tint (1,.85,.7) at 0.8), IOR 1.5'),
  });
});

def('vii_textured_512', async () => {
  resetLightIds();
  const srgbChecker = texture('checker', 64, 64, (x, y) => (((x >> 3) + (y >> 3)) & 1 ? [230, 200, 120, 255] : [40, 90, 160, 255]));
  const clampGrad = texture('grad_grid', 64, 64, (x, y) => ((x & 15) === 0 || (y & 15) === 0 ? [250, 250, 250, 255] : [x * 4, 60 + y * 2, 200 - x * 2, 255]), { wrapS: 'clamp-to-edge', wrapT: 'clamp-to-edge' });
  const mirrorStripes = texture('stripes', 32, 32, (x, y) => [x < 12 ? 220 : 60, 100 + 4 * y, x > 24 ? 200 : 40, 255], { wrapS: 'mirror-repeat', wrapT: 'mirror-repeat' });
  const mr = texture('metal_rough', 32, 32, (x, y) => [0, 40 + 6 * x, ((x >> 3) + (y >> 3)) & 1 ? 255 : 0, 255]);
  const pixelArt = texture('pixel_art', 16, 16, (x, y) => [(x * 37) & 255, (y * 53) & 255, ((x ^ y) * 29) & 255, 255], { filter: 'nearest' });
  const emis = texture('emissive', 16, 16, (x, y) => ((x >> 2) + (y >> 2)) & 1 ? [255, 200, 120, 255] : [60, 120, 255, 255]);
  const textures: TextureData[] = [srgbChecker, clampGrad, mirrorStripes, mr, pixelArt, emis];
  const mats: MaterialData[] = [
    principled('floor_checker', { baseColorTexture: { texture: 0, texCoord: 0, transform: texTransform([0.1, 0.2], 0.3, [3, 3]) }, roughnessFactor: 0.8, specularLevel: 0.3 }),
    principled('back_clamp', { baseColorTexture: { texture: 1, texCoord: 0, transform: texTransform([-0.25, -0.25], 0, [1.5, 1.5]) }, roughnessFactor: 1, specularLevel: 0.5 }),
    principled('left_mirror', { baseColorTexture: { texture: 2, texCoord: 0 }, roughnessFactor: 0.6 }),
    principled('box_mr', { baseColorFactor: [0.9, 0.7, 0.5, 1], metallicRoughnessTexture: { texture: 3, texCoord: 0 }, metallicFactor: 1, roughnessFactor: 1 }),
    principled('right_nearest', { baseColorTexture: { texture: 4, texCoord: 0 }, roughnessFactor: 0.9 }),
    principled('emissive_tex', { baseColorFactor: [0.1, 0.1, 0.1, 1], emissiveFactor: [1, 1, 1], emissiveStrength: 3, emissiveTexture: { texture: 5, texCoord: 0 }, roughnessFactor: 1 }),
  ];
  const mb = new MeshBuilder()
    .floor(-2, 2, -1.5, 1.5, 0, 0)
    .quad([-2, 0, -1.5], [2, 0, -1.5], [2, 2.2, -1.5], [-2, 2.2, -1.5], 1)
    .quad([-2, 0, 1.5], [-2, 0, -1.5], [-2, 2.2, -1.5], [-2, 2.2, 1.5], 2, [3, 2])
    .quad([2, 0, -1.5], [2, 0, 1.5], [2, 2.2, 1.5], [2, 2.2, -1.5], 4, [2, 1.5])
    .box([-0.3, 0, -0.9], [0.4, 0.6, -0.2], 3, { rotY: 25 * deg, omit: ['-y'] })
    .quad([0.9, 0.5, -1.49], [1.5, 0.5, -1.49], [1.5, 1.1, -1.49], [0.9, 1.1, -1.49], 5);
  const L = light('rect', lightToward(norm([0, -1, -0.25]), [0, 2.1, 0.3]), 60, { sizeX: 1, sizeY: 0.6 });
  const scene = sceneOf('vii_textured_512', mb, mats, [L], textures);
  await writePackage(OUT, scene, {
    camera: { matrix: lookAt([0.2, 1.3, 2.9], [0, 0.6, -0.6]), yfov: 55 * deg }, render: { width: 512, height: 512, maxBounces: 3 },
    source: SRC('(vii) flat textured: sRGB base colour (repeat + KHR_texture_transform), clamp, mirror, nearest, metal/rough, emissive texture; alpha = 1'),
  });
});

def('viii_usd_cornell_512', async () => {
  const file = path.join(ROOT, 'validation/assets/cornell/cornell.usda');
  const { scene: s } = await loadUsdInline(new Uint8Array(readFileSync(file)), 'cornell.usda');
  const cam = s.cameras[0];
  if (!cam) throw new Error('cornell.usda: no camera');
  // Mode A (M3a): analytic area lights are not camera-visible (the USD light is; Cycles hides it without MIS anyway).
  const scene: SceneData = { ...s, name: 'viii_usd_cornell_512', lights: s.lights.map((l) => ({ ...l, visibleToCamera: false })) };
  const sha = Buffer.from(await crypto.subtle.digest('SHA-256', new Uint8Array(readFileSync(file)))).toString('hex');
  await writePackage(OUT, scene, {
    camera: { matrix: Array.from(cam.matrix), yfov: cam.yfov }, render: { width: 512, height: 512, maxBounces: 3 }, flatShaded: undefined,
    source: { uri: 'validation/assets/cornell/cornell.usda via loadUsdInline (validation/scenes/make-m3a.ts (viii))', sha256: sha },
    extra: { notes: ['(viii) minimal USD: the Cornell USD through our USD loader (LightUSD) -> SceneData -> package; lights forced visibleToCamera=false (Mode A)'] },
  });
});

def('x_many_lights_512', async () => {
  resetLightIds();
  const R = rng(1234);
  const mats = [v1('floor', { diffuse: [0.55, 0.55, 0.55] }), v1('back', { diffuse: [0.6, 0.55, 0.5] }), v1('box', { diffuse: [0.35, 0.45, 0.6] }), v1('emitter', { diffuse: [0.1, 0.1, 0.1], emission: [3, 5, 4] })];
  const mb = new MeshBuilder().floor(-4, 4, -4, 3, 0, 0)
    .quad([-4, 0, -4], [4, 0, -4], [4, 3, -4], [-4, 3, -4], 1)
    .box([-2.2, 0, -2.5], [-1.2, 1.2, -1.5], 2, { rotY: 15 * deg, omit: ['-y'] })
    .box([1.0, 0, -1.8], [1.8, 0.6, -1.0], 2, { rotY: -30 * deg, omit: ['-y'] })
    .box([-0.4, 0, 0.2], [0.3, 0.35, 0.9], 2, { omit: ['-y'] })
    .icosphere([0.3, 0.9, -2.6], 0.25, 2, 3);
  const lights: LightData[] = [];
  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      const p: V3 = [-3.5 + i + 0.3 * (R() - 0.5), 0.4 + 0.8 * R(), -3.5 + j * 0.85 + 0.3 * (R() - 0.5)];
      lights.push(light('point', new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...p, 1]), 0.5 + 3 * R(), { color: [0.4 + 0.6 * R(), 0.4 + 0.6 * R(), 0.4 + 0.6 * R()] }));
    }
  }
  for (let k = 0; k < 8; k++) {
    const p: V3 = [-3 + 6 * R(), 1.6 + 0.8 * R(), -3.2 + 5 * R()];
    const d = norm([0.6 * (R() - 0.5), -1, 0.6 * (R() - 0.5)]);
    lights.push(light('rect', lightToward(d, p), 5 + 10 * R(), { sizeX: 0.2 + 0.3 * R(), sizeY: 0.2 + 0.3 * R(), color: [0.6 + 0.4 * R(), 0.6 + 0.4 * R(), 0.6 + 0.4 * R()] }));
  }
  for (let k = 0; k < 4; k++) {
    const p: V3 = [-3 + 6 * R(), 1.2 + 0.8 * R(), -3 + 4 * R()];
    lights.push(light('disk', lightToward(norm([0.5 * (R() - 0.5), -1, 0.5 * (R() - 0.5)]), p), 5 + 5 * R(), { sizeX: 0.2 + 0.2 * R(), spread: 60 * deg }));
  }
  for (let k = 0; k < 4; k++) {
    const p: V3 = [-3 + 6 * R(), 1.5 + R(), -1 + 3 * R()];
    const target: V3 = [-2 + 4 * R(), 0, -3 + 3 * R()];
    lights.push(light('spot', lightToward(norm([target[0] - p[0], target[1] - p[1], target[2] - p[2]]), p), 20 + 30 * R(), { spotSize: (30 + 40 * R()) * deg, spotBlend: 0.1 + 0.5 * R() }));
  }
  lights.push(light('sun', lightToward(scale(dirFromAngles(50 * deg, 200 * deg), -1), [0, 5, 0]), 0.3, { color: [1, 0.9, 0.8] }));
  // editor property coverage: Blender exposure (Φ = color·power·2^exposure) on a point, a rect and a spot
  lights[5].exposure = 1; lights[66].exposure = -0.5; lights[77].exposure = 0.75;
  const scene = sceneOf('x_many_lights_512', mb, mats, lights);
  await writePackage(OUT, scene, {
    camera: { matrix: lookAt([0.5, 3.2, 4.6], [0, 0.3, -1.2]), yfov: 55 * deg }, render: { width: 512, height: 512, maxBounces: 2 },
    source: SRC('(x) many lights: 64 point + 8 rect + 4 disk (spread 60 deg) + 4 spot + 1 sun + 1 emissive mesh; exposure != 0 on 3 lights'),
  });
});

def('xi_contact_512', async () => {
  resetLightIds();
  const mats = [v1('floor', { diffuse: [0.6, 0.6, 0.6] }), v1('back', { diffuse: [0.7, 0.65, 0.6] }), v1('left', { diffuse: [0.55, 0.6, 0.7] }), v1('box', { diffuse: [0.75, 0.5, 0.35] }), v1('slab', { diffuse: [0.4, 0.7, 0.45] })];
  const mb = new MeshBuilder().floor(-1, 1.5, -1, 1, 0, 0)
    .quad([-1, 0, -1], [1.5, 0, -1], [1.5, 1.5, -1], [-1, 1.5, -1], 1)       // back wall z = −1
    .quad([-1, 0, 1], [-1, 0, -1], [-1, 1.5, -1], [-1, 1.5, 1], 2)           // left wall x = −1 (corner with the back wall)
    .box([-1, 0, -1], [-0.6, 0.4, -0.6], 3, { omit: ['-y', '-x', '-z'] })     // box pushed into the corner (touches floor + both walls)
    .box([-0.1, 0, -0.35], [0.3, 0.25, 0.05], 3, { rotY: 30 * deg, omit: ['-y'] }) // box resting on the floor
    .box([0.5, 0.002, -0.9], [1.1, 0.012, -0.5], 4)                          // 1 cm slab, 2 mm above the floor
    .quad([0.2, 0.3, -0.997], [0.6, 0.3, -0.997], [0.6, 0.8, -0.997], [0.2, 0.8, -0.997], 4); // panel 3 mm off the back wall
  const lights = [
    light('rect', lightToward(norm([0.2, -1, -0.1]), [0, 1.45, -0.2]), 40, { sizeX: 0.8, sizeY: 0.5 }),
    light('point', new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1.4, 0.05, -0.2, 1]), 2, { color: [1, 0.8, 0.6] }),
  ];
  const scene = sceneOf('xi_contact_512', mb, mats, lights);
  await writePackage(OUT, scene, {
    camera: { matrix: lookAt([0.45, 1.0, 1.5], [-0.05, 0.2, -0.6]), yfov: 50 * deg }, render: { width: 512, height: 512, maxBounces: 3 },
    source: SRC('(xi) contact geometry: wall corner, box in the corner and on the floor, 2 mm slab gap, 3 mm wall panel'),
  });
});

def('xii_alpha_foliage_512', async () => {
  resetLightIds();
  const R = rng(99);
  const blobs = Array.from({ length: 14 }, () => ({ x: 20 + 88 * R(), y: 16 + 96 * R(), rx: 8 + 14 * R(), ry: 5 + 9 * R(), a: Math.PI * R() }));
  const leaf = texture('leaves', 128, 128, (x, y) => {
    let inside = Math.abs(x - 64) < 2 && y > 20; // stem
    for (const b of blobs) {
      const dx = x + 0.5 - b.x, dy = y + 0.5 - b.y, c = Math.cos(b.a), s = Math.sin(b.a);
      const u = (c * dx + s * dy) / b.rx, v = (-s * dx + c * dy) / b.ry;
      if (u * u + v * v <= 1) inside = true;
    }
    return [40 + (x & 31), 110 + (y & 63), 40, inside ? 255 : 0];
  });
  const mats: MaterialData[] = [
    v1('floor', { diffuse: [0.5, 0.5, 0.5] }),
    principled('leaf_card', { baseColorTexture: { texture: 0, texCoord: 0 }, alphaMode: 'MASK', alphaCutoff: 0.5, roughnessFactor: 0.7, specularLevel: 0.3 }),
  ];
  const card = (cx: number, cz: number, rot: number, w: number, h: number, mb: MeshBuilder) => {
    const c = Math.cos(rot), s = Math.sin(rot);
    const p = (u: number, y: number): V3 => [cx + c * u, y, cz - s * u];
    mb.quad(p(-w / 2, 0.02), p(w / 2, 0.02), p(w / 2, 0.02 + h), p(-w / 2, 0.02 + h), 1);
  };
  const mb = new MeshBuilder().floor(-2, 2, -2, 2, 0, 0);
  card(-0.3, -0.2, 20 * deg, 1.0, 1.2, mb);
  card(0.45, -0.6, -35 * deg, 0.9, 1.1, mb);
  const L = light('rect', lightToward(norm([-0.3, -1, -0.6]), [0.4, 1.8, 0.9]), 50, { sizeX: 0.8, sizeY: 0.8 });
  const scene = sceneOf('xii_alpha_foliage_512', mb, mats, [L], [leaf]);
  await writePackage(OUT, scene, {
    camera: { matrix: lookAt([0.3, 1.1, 2.4], [0, 0.5, -0.4]), yfov: 50 * deg }, render: { width: 512, height: 512, maxBounces: 3 },
    source: SRC('(xii) alpha MASK foliage cards (texture alpha in {0, 1}) over a diffuse floor under a rect light'),
  });
});

// ================================================ ix keyframes =========================================================

interface IxSpec {
  name: string;
  note: string;
  build(c: Awaited<ReturnType<typeof cornellBase>>): { scene: SceneData; camera: { matrix: number[]; yfov: number }; anim: Animation };
}

const pointMatrix = (p: V3) => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...p, 1]);
const poseQ = (m: ArrayLike<number>) => matrixToPoseQ(m);

function cornellScene(c: Awaited<ReturnType<typeof cornellBase>>, name: string, lights: LightData[]): SceneData {
  return sceneOf(name, new MeshBuilder().append(c.geometry), c.materials, lights);
}

function newAnim(): Animation {
  const a = new Animation();
  a.fps = IX_FPS;
  a.duration = IX_FRAMES[IX_FRAMES.length - 1] / IX_FPS;
  a.loop = false;
  return a;
}

const IX: IxSpec[] = [
  {
    name: 'ix_a_point_256', note: 'ix-a moving point light',
    build(c) {
      const L = light('point', pointMatrix([-0.15, 0.4, 0.1]), 3, { color: [1, 0.9, 0.8] });
      const anim = newAnim();
      anim.setKey(lightTarget(L.id), 'position', 0, [-0.15, 0.4, 0.1]);
      anim.setKey(lightTarget(L.id), 'position', 1, [0.15, 0.28, -0.1]);
      anim.setKey(lightTarget(L.id), 'position', 2, [0.0, 0.47, 0.15]);
      return { scene: cornellScene(c, this.name, [L]), camera: c.camera, anim };
    },
  },
  {
    name: 'ix_b_area_256', note: 'ix-b moving + tilting rect light',
    build(c) {
      const L = light('rect', c.rect.matrix, 4, { sizeX: c.rect.sizeX, sizeY: c.rect.sizeY });
      const q0 = poseQ(c.rect.matrix).quaternion;
      const anim = newAnim();
      const tg = lightTarget(L.id);
      // stays ≥ 3 cm below the ceiling (y = 0.555) at the full 20° tilt: an emitter touching a surface makes that
      // surface's irradiance singular (infinite-variance estimators on both sides)
      anim.setKey(tg, 'position', 0, [-0.15, 0.5, -0.05]);
      anim.setKey(tg, 'position', 2, [0.12, 0.5, 0.08]);
      anim.setKey(tg, 'rotation', 0, q0);
      anim.setKey(tg, 'rotation', 2, quatNormalize(quatMul(quatAxisAngle([0, 0, 1], 20 * deg), q0)));
      return { scene: cornellScene(c, this.name, [L]), camera: c.camera, anim };
    },
  },
  {
    name: 'ix_c_spot_256', note: 'ix-c rotating (sweeping) spot light',
    build(c) {
      const m = lightToward(norm([0, -1, -0.8]), [0, 0.5, 0.2]);
      const L = light('spot', m, 6, { spotSize: 45 * deg, spotBlend: 0.2, color: [1, 0.95, 0.9] });
      const anim = newAnim();
      const sw = presetSweep({ start: poseQ(m).quaternion, angle: 50 * deg, period: 2, samplesPerPeriod: 16 });
      anim.replaceChannel(lightTarget(L.id), 'rotation', sw.rotation!);
      return { scene: cornellScene(c, this.name, [L]), camera: c.camera, anim };
    },
  },
  {
    name: 'ix_d_camera_256', note: 'ix-d moving camera (static rect light)',
    build(c) {
      const L = light('rect', c.rect.matrix, 4, { sizeX: c.rect.sizeX, sizeY: c.rect.sizeY });
      const anim = newAnim();
      const target: V3 = [0, 0.25, -0.1];
      const keys: [number, V3][] = [[0, [0, 0.273, 1.0775]], [0.8, [0.18, 0.36, 0.8]], [1.4, [-0.12, 0.2, 0.7]], [2, [0.05, 0.3, 0.95]]];
      for (const [t, p] of keys) {
        anim.setKey('camera', 'position', t, p);
        anim.setKey('camera', 'rotation', t, quatLookDir([target[0] - p[0], target[1] - p[1], target[2] - p[2]]));
      }
      return { scene: cornellScene(c, this.name, [L]), camera: c.camera, anim };
    },
  },
  {
    name: 'ix_e_steps_256', note: 'ix-e add / remove / intensity steps (step keys; removed = power 0)',
    build(c) {
      const A = light('point', pointMatrix([-0.12, 0.42, 0.05]), 0, { color: [1, 0.8, 0.6] });
      const B = light('rect', c.rect.matrix, 4, { sizeX: c.rect.sizeX, sizeY: c.rect.sizeY });
      const C = light('spot', lightToward(norm([0.3, -1, -0.4]), [-0.1, 0.5, 0.15]), 2, { spotSize: 60 * deg, spotBlend: 0.3, color: [0.7, 0.85, 1] });
      const anim = newAnim();
      anim.setKey(lightTarget(A.id), 'power', 0, [0], 'step');
      anim.setKey(lightTarget(A.id), 'power', 0.6, [3], 'step');      // added
      anim.setKey(lightTarget(B.id), 'power', 0, [4], 'step');
      anim.setKey(lightTarget(B.id), 'power', 1.0, [0], 'step');      // removed
      anim.setKey(lightTarget(B.id), 'power', 1.6, [2], 'step');      // re-added at half power
      anim.setKey(lightTarget(C.id), 'power', 0, [2], 'step');
      anim.setKey(lightTarget(C.id), 'power', 0.3, [6], 'step');      // intensity step up
      anim.setKey(lightTarget(C.id), 'power', 1.4, [1], 'step');      // and down
      return { scene: cornellScene(c, this.name, [A, B, C]), camera: c.camera, anim };
    },
  },
  {
    name: 'ix_f_combined_256', note: 'ix-f camera + lights together, a camera teleport (step) and an FOV ramp',
    build(c) {
      const P = light('point', pointMatrix([0.1, 0.45, -0.1]), 2, { color: [1, 0.85, 0.7] });
      const Rr = light('rect', c.rect.matrix, 3, { sizeX: c.rect.sizeX, sizeY: c.rect.sizeY });
      const anim = newAnim();
      anim.setKey(lightTarget(P.id), 'position', 0, [0.1, 0.45, -0.1]);
      anim.setKey(lightTarget(P.id), 'position', 2, [-0.15, 0.3, 0.12]);
      anim.setKey(lightTarget(Rr.id), 'power', 0, [3]);
      anim.setKey(lightTarget(Rr.id), 'power', 2, [1]);
      const target: V3 = [0, 0.25, -0.1];
      const front: V3 = [0, 0.273, 1.0775], side: V3 = [0.2, 0.4, 0.6];
      anim.setKey('camera', 'position', 0, front, 'step');
      anim.setKey('camera', 'rotation', 0, quatLookDir([target[0] - front[0], target[1] - front[1], target[2] - front[2]]), 'step');
      anim.setKey('camera', 'position', 1, side, 'step');       // teleport at t = 1 s (frame 24)
      anim.setKey('camera', 'rotation', 1, quatLookDir([target[0] - side[0], target[1] - side[1], target[2] - side[2]]), 'step');
      anim.setKey('camera', 'yfov', 0, [30 * deg]);
      anim.setKey('camera', 'yfov', 2, [60 * deg]);
      return { scene: cornellScene(c, this.name, [P, Rr]), camera: c.camera, anim };
    },
  },
  {
    name: 'ix_g_sun_256', note: 'ix-g rotating sun over an open courtyard',
    build() {
      const mats = [v1('ground', { diffuse: [0.5, 0.5, 0.45] }), v1('wall', { diffuse: [0.7, 0.6, 0.5] }), v1('box', { diffuse: [0.3, 0.45, 0.65] })];
      const mb = new MeshBuilder().floor(-3, 3, -3, 3, 0, 0)
        .quad([-3, 0, -2], [3, 0, -2], [3, 1.5, -2], [-3, 1.5, -2], 1)
        .box([-1.2, 0, -1.2], [-0.4, 0.8, -0.4], 2, { rotY: 10 * deg, omit: ['-y'] })
        .box([0.5, 0, -0.5], [1.1, 0.4, 0.1], 2, { rotY: -25 * deg, omit: ['-y'] });
      const toSun0 = dirFromAngles(35 * deg, 20 * deg);
      const m = lightToward(scale(toSun0, -1), [0, 5, 0]);
      const S = light('sun', m, 3, { color: [1, 0.95, 0.85] });
      const anim = newAnim();
      const q0 = poseQ(m).quaternion;
      for (let k = 0; k <= 8; k++) {
        const t = k * 0.25;
        anim.setKey(lightTarget(S.id), 'rotation', t, quatNormalize(quatMul(quatAxisAngle([0, 1, 0], -k * 20 * deg), q0)));
      }
      return { scene: sceneOf(this.name, mb, mats, [S]), camera: { matrix: lookAt([0.5, 2.6, 3.6], [0, 0.3, -0.6]), yfov: 50 * deg }, anim };
    },
  },
];

for (const spec of IX) {
  def(spec.name, async () => {
    resetLightIds();
    const c = await cornellBase();
    const { scene, camera, anim } = spec.build(c);
    const cp = matrixToPoseQ(camera.matrix);
    const frames = exportFrames(anim, { fps: IX_FPS, frames: IX_FRAMES, base: { camera: { position: cp.position, quaternion: cp.quaternion as Quat, yfov: camera.yfov }, lights: scene.lights } });
    await writePackage(OUT, scene, {
      camera, render: { width: 256, height: 256, maxBounces: 3 }, frames,
      source: SRC(`${spec.note}; frames ${IX_FRAMES.join(',')} of 0..48 @ ${IX_FPS} fps resolved with src/core/scene/animation.ts`),
      extra: { animation: { fps: IX_FPS, frames: IX_FRAMES, tracks: anim.targets } },
    });
  });
}

// ====================================================================================================================

async function main(): Promise<void> {
  const want = gens.filter(([n]) => !ONLY || ONLY.has(n));
  if (ONLY && want.length !== ONLY.size) throw new Error(`unknown scene(s): ${[...ONLY].filter((n) => !gens.some(([g]) => g === n)).join(', ')}`);
  for (const [name, g] of want) {
    await g();
    console.log(`wrote ${path.relative(ROOT, path.join(OUT, name))}`);
  }
}

export const M3A_SCENES = gens.map(([n]) => n);
if (process.argv[1]?.endsWith('make-m3a.ts')) await main();
