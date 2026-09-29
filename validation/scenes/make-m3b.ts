// M3b Gate-2 scene packages (plan §5 M3b exit, §7.2; gap-glass §7.2 G1–G10; gap-light U9 / §6): glass calibration
// C0h–C0k, the glass spec scenes G1–G10, (vi) glass/mirror in Mode A and Mode B, (vi-B) an area light in a
// roughness-0 mirror, and C0o (visibleToCamera, Mode B; deferred from M3a).
//   npx tsx validation/scenes/make-m3b.ts [outDir] [--only name,name]
// Every scene is flat shaded (Blender-identical winding ⇒ identical backfacing / η_side), Tier-1 glass (GGX), no env.
// Furnace enclosures are emission_sampling NONE (BSDF-only in both renderers: the rough-glass parity rule of glass §5.3).
// Closed forms are written to scene.json "expected" (validation/tools/analytic_check.py kinds glass-slab, glass-shadow).
// Deterministic: re-running writes byte-identical packages (the gate checks this).
import path from 'node:path';
import type { LightData, MaterialData } from '../../src/core/scene/types.ts';
import {
  MeshBuilder, ROOT, cornellBase, deg, emitterNone, glassNode, light, lightToward, lookAt, norm, principled, refractionNode,
  resetLightIds, sceneOf, v1, writePackage, type V3,
} from './scene-kit.ts';

const argv = process.argv.slice(2);
const onlyArg = argv.indexOf('--only');
const ONLY = onlyArg >= 0 ? new Set(argv[onlyArg + 1].split(',')) : undefined;
const OUT = argv[0] && !argv[0].startsWith('--') ? path.resolve(argv[0]) : path.join(ROOT, 'validation/scenes');
const SRC = (what: string) => ({ uri: `validation/scenes/make-m3b.ts (${what})` });

type Gen = () => Promise<void>;
const gens: [string, Gen][] = [];
const def = (name: string, g: Gen) => gens.push([name, g]);

/** Principled glass (Transmission Weight t, metallic m, Specular IOR Level 0.5 unless given). */
const pglass = (name: string, o: { C?: V3; r?: number; ior?: number; t?: number; m?: number; level?: number; tint?: V3 } = {}): MaterialData =>
  principled(name, {
    baseColorFactor: [...(o.C ?? [1, 1, 1]), 1], roughnessFactor: o.r ?? 0, ior: o.ior ?? 1.5, transmissionFactor: o.t ?? 1,
    metallicFactor: o.m ?? 0, specularLevel: o.level ?? 0.5, specularColorFactor: o.tint ?? [1, 1, 1],
  });

/** Camera looking down at the plane y = 0 over the region around the origin: incidence ≈ 0°…70° across the image. */
const slabCam = { matrix: lookAt([0, 1.0, 1.35], [0, 0, -0.65]), yfov: 62 * deg };

// ================================================ calibration: closed forms =========================================

def('c0h_slab_transmission_256', async () => {
  resetLightIds();
  // colourless smooth Glass-node slab over an L_e = 1 emissive plane (BSDF-only), black above: every pixel sees the
  // plane through the slab: E = L_e (1−F)² Σ_{j=0}^{J} F^{2j}, J = ⌊(N−1)/2⌋ internal round trips (N = max_bounces)
  const mats = [emitterNone('backlight', [1, 1, 1]), glassNode('slab', [1, 1, 1], 0, 1.5)];
  const mb = new MeshBuilder().floor(-60, 60, -60, 60, -1, 0).box([-12, -0.01, -12], [12, 0.01, 12], 1);
  await writePackage(OUT, sceneOf('c0h_slab_transmission_256', mb, mats, []), {
    camera: slabCam, render: { width: 256, height: 256, maxBounces: 4 }, source: SRC('C0h smooth slab over an emissive plane'),
    extra: { expected: { kind: 'glass-slab', formula: 'transmission', ior: 1.5, N: 4, Le: 1, plane_y: 0.01, half_size: 12, margin: 0.5,
      text: 'L_e (1-F)^2 sum_{j<=(N-1)/2} F^{2j}, F = F_diel(cos theta_i, 1.5)' } },
  });
});

for (const [name, mat, N, formula, extra] of [
  ['g1_slab_furnace_principled_256', pglass('slab', { C: [0.5, 0.5, 0.5] }), 3, 'furnace-principled', { C: 0.5 }],
  ['g1_slab_furnace_glassnode_256', glassNode('slab', [0.8, 0.8, 0.8], 0, 1.5), 2, 'furnace-glassnode', { c: 0.8 }],
] as const) {
  def(name, async () => {
    resetLightIds();
    // smooth slab in an L_e = 1 enclosure (emission_sampling NONE); glass §7.2 G1
    const mats = [emitterNone('enclosure', [1, 1, 1]), mat];
    const mb = new MeshBuilder().box([-25, -25, -25], [25, 25, 25], 0, { inward: true }).box([-10, -0.01, -10], [10, 0.01, 10], 1);
    await writePackage(OUT, sceneOf(name, mb, mats, []), {
      camera: slabCam, render: { width: 256, height: 256, maxBounces: N }, source: SRC(`G1 smooth slab in a furnace, ${formula}, N = ${N}`),
      extra: { expected: { kind: 'glass-slab', formula, ior: 1.5, N, Le: 1, plane_y: 0.01, half_size: 10, margin: 0.5, ...extra,
        text: formula === 'furnace-principled' ? 'F + (1-F) C (1-F^N)' : 'cF + c^2 (1-F)^2 (1-(cF)^N)/(1-cF)' } },
    });
  });
}

def('c0i_immersed_emitter_256', async () => {
  resetLightIds();
  // glass §7.2 G2 / plan C0i: an emissive quad (L_e = 2, two-sided) inside a smooth Principled glass box (C = 0.36),
  // black world, max_bounces 1: E = (1 − F)·√C·L_e where the refracted ray hits the quad (1/η² would give 0.444×)
  const mats = [v1('emitter', { diffuse: [0, 0, 0], emission: [2, 2, 2] }), pglass('box', { C: [0.36, 0.36, 0.36] })];
  const mb = new MeshBuilder().box([-1, -0.5, -1], [1, 0.5, 1], 1).floor(-0.9, 0.9, -0.9, 0.9, 0, 0);
  await writePackage(OUT, sceneOf('c0i_immersed_emitter_256', mb, mats, []), {
    camera: { matrix: lookAt([0, 3, 1.0], [0, 0, 0]), yfov: 28 * deg }, render: { width: 256, height: 256, maxBounces: 1 },
    source: SRC('C0i / G2 emitter inside a smooth glass box'),
    extra: { expected: { kind: 'glass-slab', formula: 'immersed', ior: 1.5, C: 0.36, Le: 2, N: 1, plane_y: 0.5, half_size: 1, margin: 0.05,
      quad_y: 0, quad_half: 0.9, text: '(1-F) sqrt(C) L_e for rays refracted onto the quad' } },
  });
});

def('c0j_rough_glass_furnace_512x256', async () => {
  resetLightIds();
  // rough glass in an L = 1 furnace (BSDF-only enclosure), max_bounces 8: Glass node r 0.1 / 0.25 / 0.5 / 1, Principled
  // t = 1 (C = (0.9, 0.7, 0.5), r = 0.3). Tier-1 GGX loses energy (no multiscatter): Cycles parity, not a constant.
  const mats: MaterialData[] = [emitterNone('enclosure', [1, 1, 1])];
  const mb = new MeshBuilder().icosphere([0, 0, 0], 20, 3, 0, { inward: true });
  [0.1, 0.25, 0.5, 1.0].forEach((r, i) => { mats.push(glassNode(`glass_r${r}`, [1, 1, 1], r, 1.5)); mb.icosphere([-2 + i, 0, 0], 0.42, 3, mats.length - 1); });
  mats.push(pglass('principled_r0.3', { C: [0.9, 0.7, 0.5], r: 0.3, ior: 1.45 }));
  mb.icosphere([2, 0, 0], 0.42, 3, mats.length - 1);
  await writePackage(OUT, sceneOf('c0j_rough_glass_furnace_512x256', mb, mats, []), {
    camera: { matrix: lookAt([0, 0, 5.5], [0, 0, 0]), yfov: 28 * deg }, render: { width: 512, height: 256, maxBounces: 8 },
    source: SRC('C0j rough glass furnace'),
  });
});

def('c0k_glass_shadow_256', async () => {
  resetLightIds();
  // plan C0k: glass occludes every shadow ray: a point light over a floor with a smooth glass cube in between;
  // max_bounces 0 ⇒ the floor inside the cube's shadow is exactly 0 in both renderers
  const mats = [v1('floor', { diffuse: [0.5, 0.5, 0.5] }), glassNode('cube', [1, 1, 1], 0, 1.5)];
  const mb = new MeshBuilder().floor(-4, 4, -4, 4, 0, 0).box([-0.35, 0.55, -0.35], [0.35, 0.85, 0.35], 1);
  const L = light('point', new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1, 1.8, -0.05, 1]), 80);
  await writePackage(OUT, sceneOf('c0k_glass_shadow_256', mb, mats, [L]), {
    camera: { matrix: lookAt([0, 2.6, 2.3], [0, 0, -0.1]), yfov: 50 * deg }, render: { width: 256, height: 256, maxBounces: 0 },
    source: SRC('C0k glass shadow = 0'),
    extra: { expected: { kind: 'glass-shadow', light: [0.1, 1.8, -0.05], box: [[-0.35, 0.55, -0.35], [0.35, 0.85, 0.35]], margin: 0.02,
      text: 'floor points whose segment to the point light crosses the glass cube: exactly 0' } },
  });
});

// ================================================ glass spec scenes G3–G10 ===========================================

/** Four rough slab tiles (Glass node, r 0.1 / 0.25 / 0.5 / 1) in an L = 1 BSDF-only furnace, viewed at ~35–65°. */
function roughTiles(): { mb: MeshBuilder; mats: MaterialData[] } {
  const mats: MaterialData[] = [emitterNone('enclosure', [1, 1, 1])];
  const mb = new MeshBuilder().box([-25, -25, -25], [25, 25, 25], 0, { inward: true });
  [0.1, 0.25, 0.5, 1.0].forEach((r, i) => {
    mats.push(glassNode(`slab_r${r}`, [1, 1, 1], r, 1.5));
    mb.box([-2 + i, -0.02, -1.5], [-1.02 + i, 0.02, 1.5], mats.length - 1);
  });
  return { mb, mats };
}
const tileCam = { matrix: lookAt([0, 1.6, 2.4], [0, 0, -0.2]), yfov: 55 * deg };

def('g3_rough_slab_reflection_512x256', async () => {
  resetLightIds();
  const { mb, mats } = roughTiles();
  await writePackage(OUT, sceneOf('g3_rough_slab_reflection_512x256', mb, mats, []), {
    camera: tileCam, render: { width: 512, height: 256, maxBounces: 0 }, source: SRC('G3 rough slabs r 0.1/0.25/0.5/1, reflection part only (N = 0)'),
  });
});

def('g4_rough_slab_512x256', async () => {
  resetLightIds();
  const { mb, mats } = roughTiles();
  await writePackage(OUT, sceneOf('g4_rough_slab_512x256', mb, mats, []), {
    camera: tileCam, render: { width: 512, height: 256, maxBounces: 3 }, source: SRC('G4 rough slabs r 0.1/0.25/0.5/1, N = 3 (inside η = 1/ior, TIR, Tier-1 energy loss)'),
  });
});

/** Floor + point light over three panes: (a) smooth closed slab, (b) rough closed slab, (c) rough single-quad pane. */
function panes(roughLight: 'point' | 'rect'): { mb: MeshBuilder; mats: MaterialData[]; lights: LightData[] } {
  const mats = [v1('floor', { diffuse: [0.55, 0.55, 0.55] }), v1('wall', { diffuse: [0.6, 0.55, 0.5] }), glassNode('smooth', [1, 1, 1], 0, 1.5), glassNode('rough', [1, 1, 1], 0.3, 1.5)];
  const mb = new MeshBuilder().floor(-3, 3, -2, 2, 0, 0).quad([-3, 0, -2], [3, 0, -2], [3, 2.5, -2], [-3, 2.5, -2], 1)
    .box([-1.6, 0.9, -0.4], [-0.8, 0.96, 0.4], 2)                              // (a) smooth closed slab
    .box([-0.4, 0.9, -0.4], [0.4, 0.96, 0.4], 3)                               // (b) rough closed slab
    .quad([0.8, 0.93, 0.4], [1.6, 0.93, 0.4], [1.6, 0.93, -0.4], [0.8, 0.93, -0.4], 3);   // (c) rough single-quad pane (up)
  const lights = roughLight === 'point'
    ? [light('point', new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2.2, 0, 1]), 90)]
    : [light('rect', lightToward([0, -1, 0], [0, 2.2, 0]), 90, { sizeX: 0.5, sizeY: 0.5 })];
  return { mb, mats, lights };
}
const paneCam = { matrix: lookAt([0, 2.8, 3.0], [0, 0.3, -0.3]), yfov: 50 * deg };

def('g5_panes_point_256', async () => {
  resetLightIds();
  const { mb, mats, lights } = panes('point');
  await writePackage(OUT, sceneOf('g5_panes_point_256', mb, mats, lights), {
    camera: paneCam, render: { width: 256, height: 256, maxBounces: 3 },
    source: SRC('G5 point light over (a) smooth slab (b) rough slab (c) rough pane, Mode A: exit-face / pane NEE through rough glass'),
  });
});

def('g5b_panes_rect_B_256', async () => {
  resetLightIds();
  const { mb, mats, lights } = panes('rect');
  await writePackage(OUT, sceneOf('g5b_panes_rect_B_256', mb, mats, lights), {
    camera: paneCam, render: { width: 256, height: 256, maxBounces: 3 }, lightMode: 'B',
    source: SRC('G5b rect light (Mode B, MIS on) over smooth/rough glass: NEE through rough transmission meets Cycles power-heuristic MIS on the spurious region (glass §5.3: approximate tier)'),
  });
});

/** Rect light over a smooth flat glass icosphere on a floor (glass §7.2 G6): caustic only by BSDF sampling. */
function caustic(): { mb: MeshBuilder; mats: MaterialData[]; lights: LightData[] } {
  const mats = [v1('floor', { diffuse: [0.6, 0.6, 0.6] }), v1('wall', { diffuse: [0.5, 0.55, 0.6] }), glassNode('sphere', [1, 1, 1], 0, 1.5)];
  const mb = new MeshBuilder().floor(-3, 3, -2, 2, 0, 0).quad([-3, 0, -2], [3, 0, -2], [3, 2.5, -2], [-3, 2.5, -2], 1)
    .icosphere([0, 0.55, 0], 0.45, 2, 2);
  const lights = [light('rect', lightToward([0, -1, 0], [0, 2.4, 0]), 40, { sizeX: 0.6, sizeY: 0.6 })];
  return { mb, mats, lights };
}
const causticCam = { matrix: lookAt([0, 2.0, 2.6], [0, 0.35, 0]), yfov: 45 * deg };

def('g6_caustic_B_256', async () => {
  resetLightIds();
  const { mb, mats, lights } = caustic();
  await writePackage(OUT, sceneOf('g6_caustic_B_256', mb, mats, lights), {
    camera: causticCam, render: { width: 256, height: 256, maxBounces: 4 }, lightMode: 'B',
    source: SRC('G6 rect light (Mode B) behind a smooth glass icosphere: caustic through delta glass'),
  });
});

def('g6neg_caustic_A_256', async () => {
  resetLightIds();
  const { mb, mats, lights } = caustic();
  await writePackage(OUT, sceneOf('g6neg_caustic_A_256', mb, mats, lights), {
    camera: causticCam, render: { width: 256, height: 256, maxBounces: 4 }, lightMode: 'A',
    source: SRC('G6-neg: G6 in Mode A (no caustic: A < B, glass §4.3)'),
  });
});

/** A small room (floor, back and side walls) with a key rect light (Mode A). */
function room(mb: MeshBuilder, mats: MaterialData[]): LightData[] {
  const m0 = mats.length;
  mats.push(v1('floor', { diffuse: [0.5, 0.5, 0.5] }), v1('back', { diffuse: [0.6, 0.6, 0.7] }), v1('left', { diffuse: [0.6, 0.3, 0.25] }), v1('right', { diffuse: [0.25, 0.5, 0.3] }));
  mb.floor(-2.5, 2.5, -1.5, 1.5, 0, m0)
    .quad([-2.5, 0, -1.5], [2.5, 0, -1.5], [2.5, 2.5, -1.5], [-2.5, 2.5, -1.5], m0 + 1)
    .quad([-2.5, 0, 1.5], [-2.5, 0, -1.5], [-2.5, 2.5, -1.5], [-2.5, 2.5, 1.5], m0 + 2)
    .quad([2.5, 0, -1.5], [2.5, 0, 1.5], [2.5, 2.5, 1.5], [2.5, 2.5, -1.5], m0 + 3);
  return [light('rect', lightToward(norm([0, -Math.cos(20 * deg), -Math.sin(20 * deg)]), [0, 2.0, 0.6]), 80, { sizeX: 1.2, sizeY: 0.6 }),
    light('rect', lightToward(norm([-1, -0.3, -1]), [1.6, 1.0, 1.0]), 15, { sizeX: 0.3, sizeY: 0.3, color: [1, 0.85, 0.7] })];
}
const roomCam = { matrix: lookAt([0, 1.2, 2.7], [0, 0.35, -0.3]), yfov: 45 * deg };

const G7_MATS = (): MaterialData[] => [
  pglass('t0.25_m0', { C: [0.9, 0.5, 0.3], r: 0.3, t: 0.25, level: 0.5 }),
  pglass('t0.5_m0', { C: [0.3, 0.7, 0.9], r: 0.3, t: 0.5, level: 0.25, tint: [1, 0.8, 0.6] }),
  pglass('t0.25_m0.25', { C: [0.8, 0.8, 0.4], r: 0.45, t: 0.25, m: 0.25, level: 0.5, tint: [0.7, 0.9, 1] }),
  pglass('t0.5_m0.25', { C: [0.6, 0.4, 0.8], r: 0.25, t: 0.5, m: 0.25, level: 0.25 }),
];

def('g7_principled_mix_512', async () => {
  resetLightIds();
  const mats: MaterialData[] = [];
  const mb = new MeshBuilder();
  const lights = room(mb, mats);
  G7_MATS().forEach((m, i) => { mats.push(m); mb.icosphere([(i - 1.5) * 0.9, 0.35, -0.3], 0.35, 3, mats.length - 1); });
  await writePackage(OUT, sceneOf('g7_principled_mix_512', mb, mats, lights), {
    camera: roomCam, render: { width: 512, height: 512, maxBounces: 4 },
    source: SRC('G7 Principled mixtures t 0.25/0.5 x m 0/0.25, coloured C, Specular IOR Level 0.25/0.5, coloured Specular Tint; Mode A'),
  });
});

def('g7_principled_mix_furnace_512x256', async () => {
  resetLightIds();
  const mats: MaterialData[] = [emitterNone('enclosure', [1, 1, 1])];
  const mb = new MeshBuilder().icosphere([0, 0, 0], 20, 3, 0, { inward: true });
  G7_MATS().forEach((m, i) => { mats.push(m); mb.icosphere([(i - 1.5) * 1.1, 0, 0], 0.45, 3, mats.length - 1); });
  await writePackage(OUT, sceneOf('g7_principled_mix_furnace_512x256', mb, mats, []), {
    camera: { matrix: lookAt([0, 0, 5.5], [0, 0, 0]), yfov: 25 * deg }, render: { width: 512, height: 256, maxBounces: 0 },
    source: SRC('G7 Principled mixtures in an L = 1 furnace, N = 0 (single-scatter albedo sum of the closures)'),
  });
});

def('g8_cornell_glass_512', async () => {
  resetLightIds();
  const c = await cornellBase();
  const mats = [...c.materials, glassNode('rough_ico', [1, 1, 1], 0.3, 1.5), glassNode('smooth_cube', [1, 1, 1], 0, 1.5)];
  // rough icosphere resting 2 mm above the short box (top y = 0.165 at (0.0925, 0.1085)); smooth cube 1 mm above the
  // floor in the front-left corner (no coplanar faces: the two renderers would disagree on coincident surfaces)
  const mb = new MeshBuilder().append(c.geometry)
    .icosphere([0.0925, 0.237, 0.1085], 0.07, 2, mats.length - 2)
    .box([-0.225, 0.001, 0.12], [-0.125, 0.101, 0.22], mats.length - 1, { rotY: 20 * deg });
  const L = light('rect', c.rect.matrix, c.rect.power, { sizeX: c.rect.sizeX, sizeY: c.rect.sizeY });
  await writePackage(OUT, sceneOf('g8_cornell_glass_512', mb, mats, [L]), {
    camera: c.camera, render: { width: 512, height: 512, maxBounces: 8 },
    source: { uri: 'cornell.glb + rough glass icosphere (r 0.3) + smooth glass cube (validation/scenes/make-m3b.ts G8)', sha256: c.glbSha },
  });
});

def('g9_bubble_256', async () => {
  resetLightIds();
  const mats: MaterialData[] = [];
  const mb = new MeshBuilder();
  const lights = room(mb, mats);
  mats.push(pglass('bubble', { C: [1, 1, 1], r: 0, ior: 0.75 }), pglass('bubble_rough', { C: [0.9, 0.95, 1], r: 0.25, ior: 0.75 }));
  mb.icosphere([-0.5, 0.45, -0.3], 0.4, 3, mats.length - 2).icosphere([0.5, 0.45, -0.3], 0.4, 3, mats.length - 1);
  await writePackage(OUT, sceneOf('g9_bubble_256', mb, mats, lights), {
    camera: roomCam, render: { width: 256, height: 256, maxBounces: 6 },
    source: SRC('G9 bubbles: Principled IOR 0.75 in air (η_side < 1 from the front, TIR from outside), smooth and r = 0.25'),
  });
});

def('g10_colored_glass_refraction_256', async () => {
  resetLightIds();
  const mats: MaterialData[] = [];
  const mb = new MeshBuilder();
  const lights = room(mb, mats);
  mats.push(glassNode('glass_coloured', [0.9, 0.6, 0.3], 0.15, 1.5), refractionNode('refraction', [0.8, 0.9, 1.0], 0.2, 1.45));
  mb.icosphere([-0.5, 0.45, -0.3], 0.4, 3, mats.length - 2).icosphere([0.5, 0.45, -0.3], 0.4, 3, mats.length - 1);
  await writePackage(OUT, sceneOf('g10_colored_glass_refraction_256', mb, mats, lights), {
    camera: roomCam, render: { width: 256, height: 256, maxBounces: 6 },
    source: SRC('G10 coloured Glass node (colour on R and T) + Refraction node GGX (TIR energy loss)'),
  });
});

// ================================================ (vi), (vi-B), C0o =================================================

/** (vi) room with a roughness-0 metal mirror panel, a smooth glass sphere, a glossy sphere; rect + disk lights. */
function glassMirror(): { mb: MeshBuilder; mats: MaterialData[]; lights: LightData[] } {
  const mats: MaterialData[] = [];
  const mb = new MeshBuilder();
  const lights = room(mb, mats);
  mats.push(principled('mirror', { baseColorFactor: [0.95, 0.95, 0.95, 1], metallicFactor: 1, roughnessFactor: 0 }));
  mb.quad([-2.4, 0.2, -1.2], [-2.4, 0.2, 0.6], [-2.4, 1.8, 0.6], [-2.4, 1.8, -1.2], mats.length - 1);   // on the left wall, facing +x
  mats.push(glassNode('glass', [1, 1, 1], 0, 1.5), v1('glossy', { diffuse: [0.3, 0.4, 0.6], glossy: [0.9, 0.9, 0.9], roughness: 0.3, mix: 0.6 }));
  mb.icosphere([-0.2, 0.4, -0.2], 0.4, 3, mats.length - 2).icosphere([0.95, 0.3, -0.6], 0.3, 3, mats.length - 1);
  lights.push(light('disk', lightToward(norm([0.3, -1, -0.2]), [-1.2, 2.3, 0.3]), 25, { sizeX: 0.4, spread: 100 * deg, color: [0.85, 0.9, 1] }));
  return { mb, mats, lights };
}
const viCam = { matrix: lookAt([0.6, 1.3, 2.7], [-0.4, 0.5, -0.4]), yfov: 50 * deg };

for (const mode of ['A', 'B'] as const) {
  def(`vi_glass_mirror_${mode}_512`, async () => {
    resetLightIds();
    const { mb, mats, lights } = glassMirror();
    await writePackage(OUT, sceneOf(`vi_glass_mirror_${mode}_512`, mb, mats, lights), {
      camera: viCam, render: { width: 512, height: 512, maxBounces: 6 }, lightMode: mode,
      source: SRC(`(vi) glass / mirror, Mode ${mode} (Cycles per-light MIS ${mode === 'B' ? 'on' : 'off'})`),
    });
  });
}

def('vi_b_mirror_area_B_256', async () => {
  resetLightIds();
  // plan (vi-B) / §1.4 positive control: an area light seen (and reflected onto the wall) by a roughness-0 mirror floor
  const mats = [principled('mirror', { baseColorFactor: [0.9, 0.9, 0.9, 1], metallicFactor: 1, roughnessFactor: 0 }), v1('wall', { diffuse: [0.6, 0.6, 0.6] })];
  const mb = new MeshBuilder().floor(-2, 2, -2, 2, 0, 0).quad([-2, 0, -2], [2, 0, -2], [2, 2.5, -2], [-2, 2.5, -2], 1)
    .quad([-2, 0, 2], [-2, 0, -2], [-2, 2.5, -2], [-2, 2.5, 2], 1);
  const L = light('rect', lightToward([0, -1, 0], [0.2, 1.8, -0.8]), 50, { sizeX: 1.0, sizeY: 0.7 });
  await writePackage(OUT, sceneOf('vi_b_mirror_area_B_256', mb, mats, [L]), {
    camera: { matrix: lookAt([0, 1.3, 2.6], [0, 0.4, -0.9]), yfov: 50 * deg }, render: { width: 256, height: 256, maxBounces: 2 }, lightMode: 'B',
    source: SRC('(vi-B) rect light in a roughness-0 mirror floor, Mode B (Cycles MIS on)'),
  });
});

def('c0o_visible_camera_B_256', async () => {
  resetLightIds();
  // plan C0o: camera-visible rect + disk lights (length-1 terms, weight 1, pass-through), a hidden rect light, Mode B
  const mats = [v1('floor', { diffuse: [0.5, 0.5, 0.5] }), v1('wall', { diffuse: [0.6, 0.55, 0.5] })];
  const mb = new MeshBuilder().floor(-3, 3, -3, 3, 0, 0).quad([-3, 0, -2], [3, 0, -2], [3, 2.5, -2], [-3, 2.5, -2], 1);
  const lights = [
    light('rect', lightToward(norm([0, -0.3, 1]), [-0.7, 1.0, -1.2]), 12, { sizeX: 0.5, sizeY: 0.35, visibleToCamera: true, color: [1, 0.8, 0.6] }),
    light('disk', lightToward(norm([0.2, -0.2, 1]), [0.8, 0.9, -1.0]), 10, { sizeX: 0.4, visibleToCamera: true, spread: 120 * deg, color: [0.6, 0.8, 1] }),
    light('rect', lightToward([0, -1, 0], [0, 2.2, 0]), 30, { sizeX: 0.8, sizeY: 0.8, visibleToCamera: false }),
  ];
  await writePackage(OUT, sceneOf('c0o_visible_camera_B_256', mb, mats, lights), {
    camera: { matrix: lookAt([0, 1.2, 2.5], [0, 0.8, -1]), yfov: 50 * deg }, render: { width: 256, height: 256, maxBounces: 2 }, lightMode: 'B',
    source: SRC('C0o visibleToCamera: camera-visible rect and disk (spread 120 deg) facing the camera + a hidden rect light, Mode B'),
  });
});

for (const [name, g] of gens) {
  if (ONLY && !ONLY.has(name)) continue;
  await g();
  console.log(`wrote ${path.join(OUT, name)}`);
}
