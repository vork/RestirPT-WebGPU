// M6 Gate-3 scene packages (restir-m6-api.md §5.1, §5.3; MD14, MD16, Q2): our PT vs our ReSTIR (Stage B), no Cycles.
//   npx tsx validation/scenes/make-m6.ts [outDir] [--only name,name]
// Default outDir: validation/out/m6/scenes (gitignored: generated, deterministic — re-running writes byte-identical
// packages; Gate 0 checks it).
//
//   m6_crossings_B_256   t3_cases_256 without the env (point, spot, rect, disk, sun, emissive mesh, mirror, GGX / V2
//                        spheres, b = 4) + the three crossing lights of the T3 Mode-B fixture (m6-fixtures.ts) standing in
//                        the room with geometry behind them: BSDF rays cross them front-facing (rung 3.11, U8-7).
//   x10_foliage_256      large-foliage alpha scene (Q2: generated instead of Sponza): 480 alpha-MASK leaf cards (the
//                        procedural leaf texture of t3_alpha_256) in 12 clumps over a Lambert ground with a back wall, sun +
//                        a constant sky (env in the light tiles) + one rect light; b = 3 (rung 3.10).
//   m6_tiles_glossy_256  4096 small emissive triangles (64 × 64 grid, 8 emission levels: a non-uniform power pmf) in a
//                        panel 1.2 m above a GGX r 0.3 floor; the camera sees the panel's glossy highlight; b = 1 (U8-10).
//   cornell_i_B_512, ixs_b_area_B_256   byte copies of validation/scenes/{cornell_i_512, ixs_b_area_256} with
//                        scene.json lightMode "B" (rung 3.11 Mode-B units; the PT and ReSTIR read the package's mode).
// The (xiv) glass and Mode-B variants come from make-m3c.ts (--glass, --light-mode B) into the same directory.
import type { EnvironmentData, LightData, MaterialData, SceneData } from '../../src/core/scene/types.ts';
import { quantizeScene } from '../../src/core/scene/quantize.ts';
import { light, lightToward, Mesh, principled, t3Scene, v1, type V3 } from './make-m4.ts';
import { crossingLights, leafTexture } from './m6-fixtures.ts';

export const M6_SCENES = ['m6_crossings_B_256', 'x10_foliage_256', 'm6_tiles_glossy_256', 'cornell_i_B_512', 'ixs_b_area_B_256'] as const;
/** Packages made by make-m3c.ts flags into the M6 scene directory: [name, flags, --only]. */
export const M6_M3C_VARIANTS: [string, string[], string][] = [
  ['xiv_overcast_rect_b3_512_glass', ['--glass'], 'xiv_overcast_rect_b3_512'],
  ['xiv_overcast_rect_b3_512_B', ['--light-mode', 'B'], 'xiv_overcast_rect_b3_512'],
];
/** Mode-B copies: [new name, source package under validation/scenes]. */
const MODE_B_COPIES: [string, string][] = [['cornell_i_B_512', 'cornell_i_512'], ['ixs_b_area_B_256', 'ixs_b_area_256']];

const nrm = (a: V3): V3 => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
function lookAt(eye: V3, target: V3): number[] {
  const z = nrm([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
  const x = nrm([z[2], 0, -z[0]]);
  const y: V3 = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  return [...x, 0, ...y, 0, ...z, 0, ...eye, 1];
}
/** Deterministic LCG in [0, 1). */
function lcg(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a = (Math.imul(a, 1664525) + 1013904223) >>> 0; return a / 4294967296; };
}

interface Built { scene: SceneData; camera: { matrix: number[]; yfov: number }; maxBounces: number; width: number; height: number; notes: string; lightMode: 'A' | 'B' }

function finish(name: string, mb: Mesh, materials: MaterialData[], lights: LightData[], textures: SceneData['textures'], env?: EnvironmentData): SceneData {
  const geometry = mb.build(materials);
  const mn: V3 = [Infinity, Infinity, Infinity], mx: V3 = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < geometry.positions.length; i += 3) for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], geometry.positions[i + k]); mx[k] = Math.max(mx[k], geometry.positions[i + k]); }
  const scene = quantizeScene({ name, geometry, materials, textures, lights, cameras: [], bounds: { min: mn, max: mx }, warnings: [] }).scene;
  if (env) scene.env = env;
  return scene;
}

export function m6Crossings(): Built {
  const t = t3Scene('t3_cases_256_noenv', undefined, (b) => { b.lights.push(...crossingLights()); });
  t.scene.name = 'm6_crossings_B_256';
  return { ...t, notes: 'm6_crossings_B_256: t3_cases_256 (no env) + 3 crossing lights, Mode B (restir-m6-api.md §5.1 rung 3.11, U8-7)', lightMode: 'B' };
}

export function x10Foliage(): Built {
  const R = lcg(0xF011A6E);
  const materials: MaterialData[] = [v1('ground', { diffuse: [0.45, 0.4, 0.3] }), v1('wall', { diffuse: [0.6, 0.6, 0.6] }),
    principled('leaf_card', { baseColorTexture: { texture: 0, texCoord: 0 }, alphaMode: 'MASK', alphaCutoff: 0.5, roughnessFactor: 0.6 } as Partial<MaterialData>)];
  const mb = new Mesh();
  mb.quad([-4, 0, 3], [4, 0, 3], [4, 0, -4], [-4, 0, -4], 0, [0, 1, 0]);
  mb.quad([-4, 0, -4], [4, 0, -4], [4, 3, -4], [-4, 3, -4], 1, [0, 1, 0]);
  const uv: [number, number][] = [[0, 1], [1, 1], [1, 0], [0, 0]];
  for (let c = 0; c < 12; c++) {
    const cx = -2.6 + 5.2 * R(), cz = -3.2 + 4 * R(), cy = 0.4 + 1.2 * R();
    for (let k = 0; k < 40; k++) {
      const p: V3 = [cx + 0.5 * (R() - 0.5), cy + 0.6 * (R() - 0.5), cz + 0.5 * (R() - 0.5)];
      const n = nrm([R() - 0.5, R() - 0.3, R() - 0.5]);
      const u = nrm([n[2], 0, -n[0]]), v: V3 = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
      const h = 0.12 + 0.1 * R();
      const q = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => [p[0] + h * (a * u[0] + b * v[0]), p[1] + h * (a * u[1] + b * v[1]), p[2] + h * (a * u[2] + b * v[2])] as V3);
      mb.tri(q[0], q[1], q[2], 2, [uv[0], uv[1], uv[2]]).tri(q[0], q[2], q[3], 2, [uv[0], uv[2], uv[3]]);
    }
  }
  const lights: LightData[] = [
    light(1, 'sun', lightToward(nrm([-0.4, -1, -0.5]), [0, 6, 0]), 2.5),
    light(2, 'rect', lightToward([0, -1, 0], [0.8, 2.6, -1.2]), 40, { sizeX: 0.8, sizeY: 0.5 }),
  ];
  const W = 64, H = 32, tex = new Float32Array(W * H * 4);
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) { const up = r >= H / 2 ? 1 : 0.15; tex.set([0.35 * up, 0.45 * up, 0.6 * up, 1], 4 * (r * W + c)); }
  const env: EnvironmentData = { name: 'sky-constant', width: W, height: H, texels: tex, strength: 1, tint: [1, 1, 1], rotationZ: 0, visibleToCamera: true };
  const scene = finish('x10_foliage_256', mb, materials, lights, [leafTexture()], env);
  return { scene, camera: { matrix: lookAt([0, 1.4, 4.2], [0, 0.9, -1.2]), yfov: 50 * Math.PI / 180 }, maxBounces: 3, width: 256, height: 256,
    notes: 'x10_foliage_256: 480 alpha-MASK leaf cards in 12 clumps, sun + constant sky + rect, b = 3 (restir-m6-api.md Q2, rung 3.10)', lightMode: 'A' };
}

export function tilesGlossy(): Built {
  const levels = [0.5, 1, 2, 4, 8, 16, 32, 64];
  const materials: MaterialData[] = [v1('floor_ggx', { diffuse: [0.05, 0.05, 0.05], glossy: [0.9, 0.9, 0.9], roughness: 0.3, mix: 0.9 }), v1('wall', { diffuse: [0.5, 0.5, 0.5] }),
    ...levels.map((e) => v1(`emit_${e}`, { diffuse: [0, 0, 0], emission: [e * 0.06, e * 0.055, e * 0.05] }))];
  const mb = new Mesh();
  mb.quad([-3, 0, 3], [3, 0, 3], [3, 0, -3], [-3, 0, -3], 0, [0, 1, 0]);
  mb.quad([-3, 0, -3], [3, 0, -3], [3, 3, -3], [-3, 3, -3], 1, [0, 1, 0]);
  const R = lcg(0x711E5);
  const N = 64, s = 1.6 / N, y = 1.2;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const x0 = -0.8 + i * s, z0 = -1.6 + j * s, m = 2 + Math.floor(R() * levels.length);
    // one small triangle per cell, facing down (−y): the floor below sees its emitting side
    mb.tri([x0 + 0.1 * s, y, z0 + 0.1 * s], [x0 + 0.9 * s, y, z0 + 0.1 * s], [x0 + 0.5 * s, y, z0 + 0.9 * s], m);
  }
  const scene = finish('m6_tiles_glossy_256', mb, materials, [], []);
  return { scene, camera: { matrix: lookAt([0, 0.9, 2.6], [0, 0.2, -1.0]), yfov: 50 * Math.PI / 180 }, maxBounces: 1, width: 256, height: 256,
    notes: 'm6_tiles_glossy_256: 4096 emissive triangles (8 levels) 1.2 m above a GGX r 0.3 floor, b = 1 (U8-10)', lightMode: 'A' };
}

async function main(): Promise<void> {
  const [{ writePackage, ROOT }, path, fs, cp] = await Promise.all([import('./scene-kit.ts'), import('node:path'), import('node:fs'), import('node:child_process')]);
  const argv = process.argv.slice(2);
  const i = argv.indexOf('--only');
  const only = i >= 0 ? new Set(argv[i + 1].split(',')) : undefined;
  const out = argv[0] && !argv[0].startsWith('--') ? path.resolve(argv[0]) : path.join(ROOT, 'validation/out/m6/scenes');
  fs.mkdirSync(out, { recursive: true });
  const gens: [string, () => Built][] = [['m6_crossings_B_256', m6Crossings], ['x10_foliage_256', x10Foliage], ['m6_tiles_glossy_256', tilesGlossy]];
  for (const [name, g] of gens) {
    if (only && !only.has(name)) continue;
    const t = g();
    const dir = await writePackage(out, t.scene, {
      name, camera: { matrix: t.camera.matrix, yfov: t.camera.yfov }, render: { width: t.width, height: t.height, maxBounces: t.maxBounces },
      lightMode: t.lightMode, source: { uri: `validation/scenes/make-m6.ts (${name})` }, extra: { notes: t.notes, tier: 'tight' },
    });
    console.log(`wrote ${path.relative(ROOT, dir)}`);
  }
  for (const [name, srcName] of MODE_B_COPIES) {
    if (only && !only.has(name)) continue;
    const src = path.join(ROOT, 'validation/scenes', srcName), dst = path.join(out, name);
    fs.rmSync(dst, { recursive: true, force: true });
    fs.cpSync(src, dst, { recursive: true });
    const sj = path.join(dst, 'scene.json');
    const j = JSON.parse(fs.readFileSync(sj, 'utf8'));
    if (j.lightMode !== 'A') throw new Error(`${srcName}: light mode ${j.lightMode}`);
    j.lightMode = 'B';
    j.name = name;
    j.notes = `${srcName} in light mode B (make-m6.ts; restir-m6-api.md §5.1 rung 3.11)`;
    fs.writeFileSync(sj, JSON.stringify(j, null, 1));
    console.log(`wrote ${path.relative(ROOT, dst)}`);
  }
  for (const [name, flags, onlyName] of M6_M3C_VARIANTS) {
    if (only && !only.has(name)) continue;
    const r = cp.spawnSync('npx', ['tsx', 'validation/scenes/make-m3c.ts', out, '--only', onlyName, ...flags], { cwd: ROOT, encoding: 'utf8' });
    if (r.status !== 0 || !fs.existsSync(path.join(out, name, 'scene.json'))) throw new Error(`make-m3c ${flags.join(' ')}: ${r.stdout}${r.stderr}`);
    console.log(`wrote ${path.relative(ROOT, path.join(out, name))}`);
  }
}

if (typeof process !== 'undefined' && process.argv?.[1]?.endsWith('make-m6.ts')) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
