// M3c Gate-2 scene packages: environment lighting (plan §5 M3c exit, §7.2 C0q/C0r/C0s/(xiii)/(xiv); env §5.2).
//   npx tsx validation/scenes/make-m3c.ts [outDir] [--only name,name] [--light-mode A|B] [--glass]
// Default outDir: validation/out/m3c/scenes (gitignored): the HDRI scenes embed the downloaded Poly Haven texels
// (validation/assets/fetch_hdris.ts, pinned SHA-256), so the packages are generated, not committed. Deterministic:
// re-running writes byte-identical packages (the gate checks this; the Cycles cache keys on the package bytes).
//
// Variants by flag (merge hygiene with M3b): --light-mode B writes Mode-B packages (analytic lights with per-light MIS
// in Cycles; names get a "_B" suffix); --glass adds the (xiv) glass-sphere variant (heavy-tail; needs the M3b glass
// lobe). The M3c gate runs the Mode-A, glass-free set.
//
// Scenes (all flat shaded; env visible to the camera):
//   C0q  constant env L = 1 (64×32 constant texture): Lambert sphere ρ 0.8 (b 0/1/3), Lambert quad from above, V1 GGX
//        spheres F ≡ 1 at α 0.2 / 0.5 (E_ss by quadrature), ρ = 1 open box b = 13. Each has a "_bg" twin whose Blender
//        world is a constant Background node (Cycles: no background light, BSDF-only, weight 1) — both Cycles variants
//        must agree with ours and with the closed form.
//   C0r  overcast_soil_puresky 1k (γ 0.6): Lambert irradiance sphere and mirror sphere (V1 GGX r = 0, F ≡ 1),
//        checked by f64 quadrature (validation/tools/env-expected.ts), independent of Cycles.
//   C0s  512×256 map: upper hemisphere 1, lower 0, one texel 1e4 at 45° elevation / on the u seam / in the top row
//        (pole wrap); Lambert plane ρ 0.5; env NEE on (AUTOMATIC) and off (NONE).
//   xiii V1 GGX spheres r ∈ {0, 0.05, 0.15, 0.19, 0.21, 0.3, 0.5} + a Principled metal under studio_small_09 (heavy tail).
//   xiv  Cornell box without its ceiling under overcast_soil_puresky (tight) and kloofendal_48d_partly_cloudy_puresky
//        (heavy tail, sun), with and without the interior rect light; b = 3 (+ b = 1, 7 overcast).
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { exportScenePackage, type ExportScenePackageOptions } from '../../src/core/scene/scene-package.ts';
import { decodeHdr } from '../../src/core/scene/env/hdr.ts';
import type { EnvironmentData, LightData, MaterialData, SceneData } from '../../src/core/scene/types.ts';
import { MeshBuilder, ROOT, cornellBase, deg, light, lookAt, principled, resetLightIds, sceneOf, v1, type V3 } from './scene-kit.ts';

const argv = process.argv.slice(2);
const flag = (k: string) => { const i = argv.indexOf(k); return i >= 0 ? argv[i + 1] : undefined; };
const ONLY = flag('--only') ? new Set(flag('--only')!.split(',')) : undefined;
const LIGHT_MODE = (flag('--light-mode') ?? 'A') as 'A' | 'B';
const GLASS = argv.includes('--glass');
const OUT = argv[0] && !argv[0].startsWith('--') ? path.resolve(argv[0]) : path.join(ROOT, 'validation/out/m3c/scenes');
const SUFFIX = LIGHT_MODE === 'B' ? '_B' : '';
const SRC = (what: string) => ({ uri: `validation/scenes/make-m3c.ts (${what})` });
const HDRI_DIR = path.join(ROOT, 'validation/assets/downloaded/hdri');

type Gen = () => Promise<void>;
const gens: [string, Gen][] = [];
const def = (name: string, g: Gen) => gens.push([name + SUFFIX, g]);

// ---- env maps -------------------------------------------------------------------------------------------------------

const envBase = { strength: 1, tint: [1, 1, 1] as [number, number, number], rotationZ: 0, visibleToCamera: true };

function constantEnv(L: number, W = 64, H = 32): EnvironmentData {
  const t = new Float32Array(W * H * 4);
  for (let k = 0; k < W * H; k++) t.set([L, L, L, 1], 4 * k);
  return { ...envBase, name: `constant-${L}`, width: W, height: H, texels: t };
}

/** C0s map: upper hemisphere 1, lower 0, one texel `val` at (col, row), rows bottom-up (row r covers v ∈ [r/H, (r+1)/H)). */
function sunTexelEnv(col: number, row: number, val = 1e4, W = 512, H = 256): EnvironmentData {
  const t = new Float32Array(W * H * 4);
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) { const up = r >= H / 2 ? 1 : 0; t.set([up, up, up, 1], 4 * (r * W + c)); }
  t.set([val, val, val, 1], 4 * (row * W + col));
  return { ...envBase, name: `sun-texel-${col}-${row}`, width: W, height: H, texels: t };
}

const hdriCache = new Map<string, EnvironmentData>();
function hdri(id: string, rotationZ = 0): EnvironmentData {
  let e = hdriCache.get(id);
  if (!e) {
    const file = path.join(HDRI_DIR, `${id}_1k.hdr`);
    let bytes: Uint8Array;
    try { bytes = new Uint8Array(readFileSync(file)); } catch { throw new Error(`${file} missing: run npx tsx validation/assets/fetch_hdris.ts`); }
    const img = decodeHdr(bytes);
    e = { ...envBase, name: `${id}_1k`, width: img.width, height: img.height, texels: img.texels };
    hdriCache.set(id, e);
  }
  return { ...e, rotationZ };
}

/** (u, v) of the brightest texel (luminance). */
function brightestUV(e: EnvironmentData): [number, number] {
  let best = -1, bi = 0;
  for (let k = 0; k < e.width * e.height; k++) {
    const y = 0.2126 * e.texels[4 * k] + 0.7152 * e.texels[4 * k + 1] + 0.0722 * e.texels[4 * k + 2];
    if (y > best) { best = y; bi = k; }
  }
  return [((bi % e.width) + 0.5) / e.width, (Math.floor(bi / e.width) + 0.5) / e.height];
}

/** glTF world direction of env texture coordinate (u, v) at rotation γ (math.md#env-mapping envDir). */
function envDirAt(u: number, v: number, g: number): V3 {
  const phi = -2 * Math.PI * u + Math.PI, theta = -Math.PI * v + Math.PI;
  const b = [Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)];
  const cg = Math.cos(g), sg = Math.sin(g);
  return [cg * b[0] + sg * b[1], b[2], -(-sg * b[0] + cg * b[1])];
}

// ---- package writer (own copy of scene-kit writePackage with an env-JSON patch hook) -----------------------------------

interface WriteOpts extends Omit<ExportScenePackageOptions, 'lightMode'> {
  extra?: Record<string, unknown>;
  /** Extra keys merged into scene.json "env" (e.g. blenderWorld: 'constant'). */
  envExtra?: Record<string, unknown>;
}

async function write(scene: SceneData, o: WriteOpts): Promise<void> {
  const { extra, envExtra, ...opts } = o;
  const pkg = await exportScenePackage(scene, { flatShaded: true, ...opts, lightMode: LIGHT_MODE });
  const json = { ...pkg.json, ...(envExtra && pkg.json.env ? { env: { ...pkg.json.env, ...envExtra } } : {}), ...extra };
  pkg.files.set('scene.json', new TextEncoder().encode(JSON.stringify(json, null, 1)));
  const dir = path.join(OUT, o.name ?? scene.name);
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  for (const [k, b] of pkg.files) writeFileSync(path.join(dir, k), b);
}

const withEnv = (s: SceneData, e: EnvironmentData): SceneData => ({ ...s, env: e });

// ---- C0q: constant-env furnace ------------------------------------------------------------------------------------

const SPHERE_CAM = { matrix: lookAt([0, 0.25, 3.5], [0, 0, 0]), yfov: 40 * deg };
const ANALYTIC = (what: string) => ({ kind: 'env-analytic', what, tool: 'validation/tools/env-expected.ts', supersampling: '4x4 per pixel', lookup: 'bilinear repeat, 8-bit fraction (Metal sampler, ENV-U7)' });

function c0q(name: string, b: number, build: () => { mb: MeshBuilder; mats: MaterialData[]; camera: { matrix: number[]; yfov: number }; analytic: boolean; note: string }): void {
  for (const bg of [false, true]) {
    def(`${name}${bg ? '_bg' : ''}`, async () => {
      resetLightIds();
      const s = build();
      const scene = withEnv(sceneOf(name, s.mb, s.mats, []), constantEnv(1));
      await write(scene, {
        name: `${name}${bg ? '_bg' : ''}${SUFFIX}`, camera: s.camera, render: { width: 256, height: 256, maxBounces: b },
        source: SRC(`C0q ${s.note}${bg ? ', Blender constant Background' : ''}`),
        ...(bg ? { envExtra: { blenderWorld: 'constant' } } : {}),
        extra: {
          ...(s.analytic ? { expected: { ...ANALYTIC(`C0q: ${s.note}`), formula: 'Lambert: rho*L; V1 GGX (F = 1): L*glossy*E_ss(mu_o, alpha); background: L' } } : {}),
          notes: bg ? 'Cycles world = constant Background (no background light: BSDF-only); ours samples the constant texture with env NEE'
            : 'Cycles world = the constant 64x32 texture (AUTOMATIC: background light with NEE + MIS)',
        },
      });
    });
  }
}

for (const b of [0, 1, 3]) {
  c0q(`c0q_lambert_b${b}_256`, b, () => ({
    mb: new MeshBuilder().icosphere([0, 0, 0], 1, 3, 0), mats: [v1('lambert', { diffuse: [0.8, 0.8, 0.8] })], camera: SPHERE_CAM, analytic: true,
    note: `Lambert sphere rho 0.8, b = ${b} (convex: every pixel rho*L)`,
  }));
}
c0q('c0q_quad_b1_256', 1, () => ({
  mb: new MeshBuilder().floor(-1, 1, -1, 1, 0, 0), mats: [v1('lambert', { diffuse: [0.8, 0.7, 0.6] })],
  camera: { matrix: lookAt([0, 2.2, 2.2], [0, 0, 0]), yfov: 50 * deg }, analytic: true, note: 'Lambert quad seen from above (rho*L; Cycles BSDF-only has zero variance)',
}));
for (const a of [0.2, 0.5]) {
  c0q(`c0q_ggx${String(a).replace('.', '')}_b1_256`, 1, () => ({
    mb: new MeshBuilder().icosphere([0, 0, 0], 1, 3, 0), mats: [v1(`ggx_alpha_${a}`, { diffuse: [0, 0, 0], glossy: [1, 1, 1], roughness: Math.sqrt(a), mix: 1 })],
    camera: SPHERE_CAM, analytic: true, note: `V1 GGX sphere F = 1, alpha ${a} (r = sqrt(alpha))`,
  }));
}
c0q('c0q_openbox_b13_256', 13, () => ({
  mb: new MeshBuilder().box([-0.5, 0, -0.5], [0.5, 1, 0.5], 0, { inward: true, omit: ['+y'] }), mats: [v1('white', { diffuse: [1, 1, 1] })],
  camera: { matrix: lookAt([0.9, 1.9, 1.4], [0, 0.3, 0]), yfov: 45 * deg }, analytic: false, note: 'rho = 1 open box, b = 13 (d <= 15; no closed form)',
}));

// ---- C0r: HDRI irradiance and mirror spheres ----------------------------------------------------------------------

const C0R_GAMMA = 0.6;
def('c0r_irradiance_256', async () => {
  resetLightIds();
  const scene = withEnv(sceneOf('c0r_irradiance_256', new MeshBuilder().icosphere([0, 0, 0], 1, 2, 0), [v1('lambert', { diffuse: [0.8, 0.8, 0.8] })], []), hdri('overcast_soil_puresky', C0R_GAMMA));
  await write(scene, {
    camera: SPHERE_CAM, render: { width: 256, height: 256, maxBounces: 1 }, source: SRC('C0r Lambert irradiance sphere, overcast_soil_puresky, gamma 0.6'),
    extra: { expected: { ...ANALYTIC('C0r irradiance sphere'), formula: 'rho/pi * integral L_env(w) max(0, n_facet.w) dw (4x4 per texel midpoint quadrature)' } },
  });
});
def('c0r_mirror_256', async () => {
  resetLightIds();
  const scene = withEnv(sceneOf('c0r_mirror_256', new MeshBuilder().icosphere([0, 0, 0], 1, 2, 0), [v1('mirror', { diffuse: [0, 0, 0], glossy: [1, 1, 1], roughness: 0, mix: 1 })], []), hdri('overcast_soil_puresky', C0R_GAMMA));
  await write(scene, {
    camera: SPHERE_CAM, render: { width: 256, height: 256, maxBounces: 1 }, source: SRC('C0r mirror sphere (V1 GGX r = 0, F = 1), overcast_soil_puresky, gamma 0.6'),
    extra: { expected: { ...ANALYTIC('C0r mirror sphere'), formula: 'L_env(reflect(d, n_facet))' } },
  });
});

// ---- C0s: single bright texel ----------------------------------------------------------------------------------------

const PLANE_CAM = { matrix: [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 1, 0, 1], yfov: 60 * deg };   // straight down
for (const [label, col, row] of [['45', 128, 191], ['seam', 0, 191], ['top', 300, 255]] as const) {
  for (const sampling of ['AUTOMATIC', 'NONE'] as const) {
    const name = `c0s_${label}_${sampling === 'NONE' ? 'none' : 'nee'}_256`;
    def(name, async () => {
      resetLightIds();
      const scene = withEnv(sceneOf(name, new MeshBuilder().floor(-50, 50, -50, 50, 0, 0), [v1('plane', { diffuse: [0.5, 0.5, 0.5] })], []), sunTexelEnv(col, row));
      await write(scene, {
        camera: PLANE_CAM, render: { width: 256, height: 256, maxBounces: 1 }, envSampling: sampling,
        source: SRC(`C0s texel 1e4 at (col ${col}, row ${row}) of 512x256, env NEE ${sampling === 'NONE' ? 'off' : 'on'}`),
        extra: { expected: { ...ANALYTIC(`C0s ${label}`), formula: 'rho/pi * E(+Y), E = integral of the bilinear map x cos+ (quadrature)' } },
      });
    });
  }
}

// ---- (xiii) glossy / mirror spheres under studio_small_09 ------------------------------------------------------------

def('xiii_spheres_512x256', async () => {
  resetLightIds();
  const rs = [0, 0.05, 0.15, 0.19, 0.21, 0.3, 0.5];
  const mb = new MeshBuilder();
  const mats: MaterialData[] = rs.map((r) => v1(`ggx_r${r}`, { diffuse: [0, 0, 0], glossy: [0.9, 0.9, 0.9], roughness: r, mix: 1 }));
  mats.push(principled('metal', { baseColorFactor: [1.0, 0.78, 0.34, 1], metallicFactor: 1, roughnessFactor: 0.25 }));
  for (let i = 0; i < 8; i++) mb.icosphere([-3.5 + i, 0, 0], 0.42, 3, i);
  const scene = withEnv(sceneOf('xiii_spheres_512x256', mb, mats, []), hdri('studio_small_09', 0.3));
  await write(scene, {
    camera: { matrix: lookAt([0, 0.35, 7.5], [0, 0, 0]), yfov: 32 * deg }, render: { width: 512, height: 256, maxBounces: 3 },
    source: SRC('(xiii) V1 GGX spheres r 0/0.05/0.15/0.19/0.21/0.3/0.5 + Principled gold metal r 0.25, studio_small_09 gamma 0.3'),
    extra: { tier: 'heavy-tail', notes: 'studio_small_09: 18 EV (verify: tight tier is <= 18 EV without sun/caustic chains; recorded as heavy-tail per the plan scene list)' },
  });
});

// ---- (xiv) Cornell open to the sky ---------------------------------------------------------------------------------------

async function cornellOpen(): Promise<{ geo: SceneData['geometry']; mats: MaterialData[]; camera: { matrix: number[]; yfov: number }; rect: Awaited<ReturnType<typeof cornellBase>>['rect']; glbSha: string }> {
  const c = await cornellBase();
  const g = c.geometry;
  let ymax = -Infinity;
  for (let i = 1; i < g.positions.length; i += 3) ymax = Math.max(ymax, g.positions[i]);
  // Drop the ceiling: triangles whose three vertices lie at the top of the box.
  const keep: number[] = [];
  for (let t = 0; t < g.indices.length / 3; t++) {
    const top = [0, 1, 2].every((k) => Math.abs(g.positions[3 * g.indices[3 * t + k] + 1] - ymax) < 1e-6);
    if (!top) keep.push(t);
  }
  const mb = new MeshBuilder();
  for (const t of keep) {
    const P = (k: number): V3 => { const i = g.indices[3 * t + k]; return [g.positions[3 * i], g.positions[3 * i + 1], g.positions[3 * i + 2]]; };
    mb.tri(P(0), P(1), P(2), g.triMaterial[t]);
  }
  return { geo: mb.build(c.materials), mats: c.materials, camera: c.camera, rect: c.rect, glbSha: c.glbSha };
}

function sunTowardCamera(id: string): number {
  // γ (0.1° grid, rounded so the package bytes are stable) that puts the env's brightest texel (the sun) behind-above
  // the camera (azimuth toward +Z, slightly +X): the sunlight enters through the open top and hits the back wall/floor.
  const [u, v] = brightestUV(hdri(id, 0));
  const want = [0.3, 0, 1].map((x) => x / Math.hypot(0.3, 1));
  let best = -Infinity, bestG = 0;
  for (let k = 0; k < 3600; k++) {
    const g = (k / 3600) * 2 * Math.PI;
    const d = envDirAt(u, v, g);
    const h = Math.hypot(d[0], d[2]);
    const score = (d[0] * want[0] + d[2] * want[2]) / h;
    if (score > best) { best = score; bestG = g; }
  }
  return Number(bestG.toFixed(6));
}

const XIV: { name: string; id: string; tier: 'tight' | 'heavy-tail'; rect: boolean; b: number }[] = [
  { name: 'xiv_overcast_b3_512', id: 'overcast_soil_puresky', tier: 'tight', rect: false, b: 3 },
  { name: 'xiv_overcast_rect_b3_512', id: 'overcast_soil_puresky', tier: 'tight', rect: true, b: 3 },
  { name: 'xiv_overcast_b1_512', id: 'overcast_soil_puresky', tier: 'tight', rect: false, b: 1 },
  { name: 'xiv_overcast_b7_512', id: 'overcast_soil_puresky', tier: 'tight', rect: false, b: 7 },
  { name: 'xiv_kloof_b3_512', id: 'kloofendal_48d_partly_cloudy_puresky', tier: 'heavy-tail', rect: false, b: 3 },
  { name: 'xiv_kloof_rect_b3_512', id: 'kloofendal_48d_partly_cloudy_puresky', tier: 'heavy-tail', rect: true, b: 3 },
];
for (const x of XIV) {
  def(x.name, async () => {
    resetLightIds();
    const c = await cornellOpen();
    const mats = [...c.mats];
    const mb = new MeshBuilder().append(c.geo);
    if (GLASS) {
      mats.push(principled('glass', { baseColorFactor: [1, 1, 1, 1], roughnessFactor: 0, transmissionFactor: 1, ior: 1.5 }));
      mb.icosphere([0.15, 0.09, -0.15], 0.09, 3, mats.length - 1);
    }
    const lights: LightData[] = x.rect ? [light('rect', c.rect.matrix, c.rect.power, { sizeX: c.rect.sizeX, sizeY: c.rect.sizeY })] : [];
    const gamma = x.id === 'overcast_soil_puresky' ? 0.9 : sunTowardCamera(x.id);
    const name = `${x.name}${GLASS ? '_glass' : ''}`;
    const scene = withEnv(sceneOf(name, mb, mats, lights), hdri(x.id, gamma));
    await write(scene, {
      name: `${name}${SUFFIX}`, camera: c.camera, render: { width: 512, height: 512, maxBounces: x.b },
      source: { uri: `validation/assets/cornell/cornell.glb without ceiling + ${x.id} 1k (validation/scenes/make-m3c.ts (xiv))`, sha256: c.glbSha },
      extra: { tier: GLASS ? 'heavy-tail' : x.tier, notes: `${x.id} gamma ${gamma.toFixed(6)}${x.rect ? ' + interior rect light 4 W (P(env) clamp)' : ''}` },
    });
  });
}

// ---- main -----------------------------------------------------------------------------------------------------------

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });
  const todo = gens.filter(([n]) => !ONLY || ONLY.has(n) || ONLY.has(n.replace(/_B$/, '')));
  for (const [name, g] of todo) {
    await g();
    console.log(`wrote ${path.relative(ROOT, path.join(OUT, name))}`);
  }
}

export const M3C_SCENES = (): string[] => gens.map(([n]) => n);

if (process.argv[1] && import.meta.url === new URL(`file://${path.resolve(process.argv[1])}`).href) {
  main().catch((e) => { console.error(e); process.exit(1); });
}
