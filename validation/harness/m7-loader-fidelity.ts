// M7 loader fidelity (docs/decisions/m7-api.md §3.2; PLAN §7.2 (vii-L), (viii-L)).
//   (viii-L) USD: our loader (LightUSD next + adapter + M7 textures, lossless) vs an OpenUSD (pxr) dump
//            (validation/tools/usd_pxr_dump.py, Blender's bundled pxr): per draw (instance proxies, PointInstancer
//            instances "<pi>[i]") the triangle count and the canonical-frame world bbox of its triangles; per material
//            the UsdPreviewSurface constants and the UsdUVTexture bindings (file, channel, wrap); per light the
//            converted LightData vs convertUsdLight on the pxr fields + pxr's world matrix; per camera the vertical FOV
//            and the matrix.
//   (vii-L)  GLB: our glTF loader (lossless) vs Blender 5.2.2's stock importer (validation/blender/import_dump.py):
//            total triangles, triangles and Principled inputs per material, the scene bbox, lights (type, power,
//            colour, spot cone / blend, world matrix).
// Tolerances: positions 1e-5·(scene extent), matrices 1e-5, scalars 1e-5 relative (f32 vs f64), counts exact.
//   npx tsx validation/harness/m7-loader-fidelity.ts [--out DIR] FILE...
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadUsdInline } from '../../src/core/scene/usd/load-usd.ts';
import { convertUsdLight, stageMatrix, mul4 } from '../../src/core/scene/usd/usd-lights.ts';
import { loadGltf } from '../../src/core/scene/gltf-loader.ts';
import type { LightData, SceneData } from '../../src/core/scene/types.ts';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const BLENDER_PY = process.env.BLENDER_PY ?? '/Applications/Blender.app/Contents/Resources/5.2/python/bin/python3.13';
const BLENDER = process.env.BLENDER ?? '/Applications/Blender.app/Contents/MacOS/Blender';

export interface FidelityCheck { item: string; ok: boolean; detail?: string }
export interface FidelityReport { file: string; kind: 'usd' | 'glb'; ok: boolean; checks: FidelityCheck[]; counts: Record<string, number>; warnings: string[] }

const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(1, Math.abs(a), Math.abs(b));
const close = (a: number, b: number, tol = 1e-5) => rel(a, b) <= tol;
const vclose = (a: readonly number[], b: readonly number[], tol = 1e-5) => a.length === b.length && a.every((x, i) => close(x, b[i], tol));
const fmt = (v: unknown) => JSON.stringify(v, (_k, x) => (typeof x === 'number' ? Number(x.toPrecision(6)) : x));

function triBBox(s: SceneData, t0: number, n: number): number[] | null {
  if (!n) return null;
  const g = s.geometry, mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
  for (let t = t0; t < t0 + n; t++) for (let k = 0; k < 3; k++) {
    const v = g.indices[3 * t + k];
    for (let c = 0; c < 3; c++) { const x = g.positions[3 * v + c]; if (x < mn[c]) mn[c] = x; if (x > mx[c]) mx[c] = x; }
  }
  return [...mn, ...mx];
}

// ------------------------------------------------------------------------------------------------ (viii-L)

export async function usdFidelity(file: string, outDir: string): Promise<FidelityReport> {
  const abs = path.resolve(ROOT, file);
  const dumpFile = path.join(outDir, `${path.basename(file)}.pxr.json`);
  const r = spawnSync(BLENDER_PY, [path.join(ROOT, 'validation/tools/usd_pxr_dump.py'), abs, '--out', dumpFile], { encoding: 'utf8' });
  if (r.status !== 0) throw new Error(`usd_pxr_dump.py: ${(r.stdout ?? '') + (r.stderr ?? '')}`.slice(-600));
  const ref = JSON.parse(readFileSync(dumpFile, 'utf8'));
  const res = await loadUsdInline(new Uint8Array(readFileSync(abs)), path.basename(abs), { assetBase: path.dirname(abs), quantize: 'lossless' });
  const s = res.scene, st = res.stats;
  const checks: FidelityCheck[] = [];
  const add = (item: string, ok: boolean, detail?: string) => { checks.push({ item, ok, detail }); };
  const ext = Math.max(...[0, 1, 2].map((k) => s.bounds.max[k] - s.bounds.min[k]), 1e-6);
  // stage
  add('stage upAxis / metersPerUnit', st.upAxis === ref.stage.upAxis && close(st.metersPerUnit, ref.stage.metersPerUnit), `ours ${st.upAxis} ${st.metersPerUnit}, pxr ${ref.stage.upAxis} ${ref.stage.metersPerUnit}`);
  // draws
  const ours = new Map(st.drawRecords.map((d) => [d.path, d]));
  const refDraws = (ref.draws as Record<string, any>[]);
  const missing = refDraws.filter((d) => !ours.has(d.path)).map((d) => d.path);
  const extra = [...ours.keys()].filter((p) => !refDraws.some((d) => d.path === p));
  add(`draws: every pxr draw present (${refDraws.length})`, missing.length === 0, missing.slice(0, 5).join(', '));
  add('draws: no extra draws', extra.length === 0, extra.slice(0, 5).join(', '));
  let triBad = 0, boxBad = 0, boxWorst = 0;
  const ex: string[] = [];
  for (const d of refDraws) {
    const o = ours.get(d.path);
    if (!o) continue;
    if (o.triCount !== d.triangles) { triBad++; if (ex.length < 4) ex.push(`${d.path}: ${o.triCount} vs ${d.triangles} tris`); }
    const bb = triBBox(s, o.triStart, o.triCount);
    if (bb && d.bbox) {
      const e = Math.max(...bb.map((x, i) => Math.abs(x - d.bbox[i]))) / ext;
      boxWorst = Math.max(boxWorst, e);
      if (e > 1e-5) { boxBad++; if (ex.length < 6) ex.push(`${d.path}: bbox ${fmt(bb)} vs ${fmt(d.bbox)}`); }
    }
  }
  add('draws: triangle counts', triBad === 0, ex.filter((x) => /tris/.test(x)).join('; '));
  add('draws: canonical world bbox (|Δ| ≤ 1e-5·extent)', boxBad === 0, `worst ${boxWorst.toExponential(2)}${boxBad ? `; ${ex.filter((x) => /bbox/.test(x)).join('; ')}` : ''}`);
  // materials
  let matBad = 0;
  const mex: string[] = [];
  for (const m of ref.materials as Record<string, any>[]) {
    // a material inside an instance prototype is reported by pxr under each instance proxy's path; ours carries one of
    // them (the instance whose mesh LightUSD emitted): match the prototype-relative tail (3 path components)
    const tail = (p: string | null) => (p ?? '').split('/').slice(-3).join('/');
    let i = st.materialPaths.indexOf(m.path);
    if (i < 0) i = st.materialPaths.findIndex((p) => tail(p) === tail(m.path) && tail(m.path).includes('/'));
    if (i < 0) { matBad++; mex.push(`${m.path} missing`); continue; }
    const md = s.materials[i];
    const T = m.textures as Record<string, any>;
    const want: [string, boolean][] = [];
    const inp = m.inputs;
    if (T.diffuseColor) want.push(['diffuseColor → baseColorTexture', !!md.baseColorTexture && s.textures[md.baseColorTexture.texture].name === T.diffuseColor.file.replace(/^\.\//, '').replace(/^\.\//, '') || (!!md.baseColorTexture && s.textures[md.baseColorTexture.texture].name.endsWith(T.diffuseColor.file.replace(/^\.\//, '')))]);
    else want.push(['diffuseColor', vclose(md.baseColorFactor.slice(0, 3), inp.diffuseColor)]);
    if (T.roughness || T.metallic) want.push(['roughness / metallic → metallicRoughnessTexture', !!md.metallicRoughnessTexture]);
    if (!T.roughness) want.push(['roughness', close(md.roughnessFactor, inp.roughness)]);
    if (!T.metallic) want.push(['metallic', close(md.metallicFactor, inp.metallic)]);
    if (T.normal) want.push(['normal → normalTexture', !!md.normalTexture]);
    if (T.emissiveColor) want.push(['emissiveColor → emissiveTexture', !!md.emissiveTexture]);
    else want.push(['emissiveColor', vclose(md.emissiveFactor.map((c) => c * md.emissiveStrength), inp.emissiveColor)]);
    want.push(['ior', close(md.ior, inp.ior)]);
    const thr = inp.opacityThreshold as number;
    want.push(['opacityThreshold → alphaMode', thr > 0 ? md.alphaMode === 'MASK' && close(md.alphaCutoff, thr) : md.alphaMode === (inp.opacity < 1 ? 'MASK' : 'OPAQUE')]);
    for (const [k, v] of Object.entries(T)) {
      if (!v.file) continue;
      const wrap = ({ repeat: 'repeat', mirror: 'mirror-repeat', clamp: 'clamp-to-edge' } as Record<string, string>)[v.wrapS ?? 'repeat'] ?? 'repeat';
      const ref2 = k === 'diffuseColor' || k === 'opacity' ? md.baseColorTexture : k === 'normal' ? md.normalTexture : k === 'emissiveColor' ? md.emissiveTexture : md.metallicRoughnessTexture;
      if (ref2) want.push([`${k} wrap ${v.wrapS ?? '(unauthored → repeat)'}`, s.textures[ref2.texture].wrapS === wrap]);
    }
    const bad = want.filter(([, ok]) => !ok).map(([n]) => n);
    if (bad.length) { matBad++; mex.push(`${m.path}: ${bad.join(', ')}`); }
  }
  add(`materials: PreviewSurface constants and UsdUVTexture bindings (${(ref.materials as unknown[]).length})`, matBad === 0, mex.slice(0, 4).join('; '));
  // lights: convertUsdLight on the pxr fields + pxr's canonical world matrix
  let lBad = 0;
  const lex: string[] = [];
  const blender = !!ref.stage.doc && /^Blender v/.test(ref.stage.doc);
  const refLights = ref.lights as Record<string, any>[];
  if (refLights.length !== s.lights.length) { lBad++; lex.push(`${s.lights.length} lights vs ${refLights.length}`); }
  refLights.forEach((l, i) => {
    const o = s.lights.find((x) => x.name === l.path || x.name === l.path.split('/').pop()) ?? s.lights[i];
    const isSpot = l.type === 'sphere' && !!l.shaping;
    const input = {
      primPath: l.path, type: l.type, intensity: l.intensity ?? 1, exposure: l.exposure ?? 0, color: (l.color ?? [1, 1, 1]) as [number, number, number],
      normalize: !!l.normalize, enableColorTemperature: !!l.enableColorTemperature, colorTemperature: l.colorTemperature ?? 6500,
      radius: l.radius ?? undefined, width: l.width ?? undefined, height: l.height ?? undefined, angle: l.angle ?? undefined, treatAsPoint: l.treatAsPoint ?? undefined,
      shaping: isSpot || l.shaping ? { coneAngle: l.shaping?.coneAngle ?? 90, coneSoftness: l.shaping?.coneSoftness ?? 0, focus: l.shaping?.focus ?? 0 } : null,
    };
    const c = convertUsdLight(input as never, l.world, i, { blenderAuthored: blender }).light as LightData | undefined;
    if (!c || !o) { lBad++; lex.push(`${l.path}: ${!c ? 'not convertible' : 'missing'}`); return; }
    const okL = c.type === o.type && close(c.power, o.power) && vclose(c.color, o.color) && vclose(Array.from(c.matrix), Array.from(o.matrix), 2e-5)
      && (c.sizeX === undefined || close(c.sizeX, o.sizeX!)) && (c.sizeY === undefined || close(c.sizeY, o.sizeY!))
      && (c.spotSize === undefined || close(c.spotSize, o.spotSize!)) && (c.spotBlend === undefined || close(c.spotBlend, o.spotBlend!));
    if (!okL) { lBad++; lex.push(`${l.path}: ours ${fmt({ t: o.type, P: o.power, c: o.color, m: Array.from(o.matrix).slice(12, 15), sx: o.sizeX })} pxr→ ${fmt({ t: c.type, P: c.power, c: c.color, m: Array.from(c.matrix).slice(12, 15), sx: c.sizeX })}`); }
  });
  add(`lights: type, power, colour, size, cone, matrix (${refLights.length})`, lBad === 0, lex.slice(0, 3).join('; '));
  // cameras
  const refCams = ref.cameras as Record<string, any>[];
  let cBad = 0;
  refCams.forEach((c, i) => {
    const o = s.cameras[i];
    if (!o || !close(o.yfov, c.yfov) || !vclose(Array.from(o.matrix).slice(12, 15), c.world.slice(12, 15), 2e-5)) cBad++;
  });
  add(`cameras: vertical FOV and position (${refCams.length})`, cBad === 0 && s.cameras.length === refCams.length);
  // PointInstancers
  const piRef = (ref.pointInstancers as Record<string, any>[]).reduce((a, p) => a + p.instances, 0);
  add('PointInstancer instances', st.pointInstanceDraws >= piRef, `ours ${st.pointInstanceDraws} draws, pxr ${piRef} instances`);
  return { file, kind: 'usd', ok: checks.every((c) => c.ok), checks, warnings: s.warnings,
    counts: { draws: refDraws.length, materials: (ref.materials as unknown[]).length, lights: refLights.length, cameras: refCams.length, pointInstances: piRef, textures: s.textures.length } };
}

// ------------------------------------------------------------------------------------------------ (vii-L)

export async function glbFidelity(file: string, outDir: string, shading: 'NORMALS' | 'FLAT' = 'NORMALS'): Promise<FidelityReport> {
  const abs = path.resolve(ROOT, file);
  const dumpFile = path.join(outDir, `${path.basename(file)}.blender.json`);
  const r = spawnSync(BLENDER, ['-b', '--factory-startup', '--python-exit-code', '1', '-P', path.join(ROOT, 'validation/blender/import_dump.py'), '--', '--file', abs, '--out', dumpFile, '--shading', shading], { encoding: 'utf8', maxBuffer: 1 << 26 });
  if (r.status !== 0) throw new Error(`import_dump.py: ${(r.stdout ?? '') + (r.stderr ?? '')}`.slice(-800));
  const ref = JSON.parse(readFileSync(dumpFile, 'utf8'));
  const bytes = new Uint8Array(readFileSync(abs));
  const source = /\.glb$/i.test(abs) ? { kind: 'glb' as const, bytes, name: path.basename(abs) } : await gltfSource(abs);
  const s = (await loadGltf(source, { quantize: 'lossless' })).scene;
  const checks: FidelityCheck[] = [];
  const add = (item: string, ok: boolean, detail?: string) => { checks.push({ item, ok, detail }); };
  const nT = s.geometry.indices.length / 3;
  add('total triangles', nT === ref.triangles, `ours ${nT}, Blender ${ref.triangles}`);
  const ext = Math.max(...[0, 1, 2].map((k) => s.bounds.max[k] - s.bounds.min[k]), 1e-6);
  const bb = [...s.bounds.min, ...s.bounds.max];
  const e = Math.max(...bb.map((x, i) => Math.abs(x - ref.bbox[i]))) / ext;
  add('scene bbox (canonical frame, |Δ| ≤ 1e-5·extent)', e <= 1e-5, `worst ${e.toExponential(2)}`);
  // triangles per material (by name; Blender may suffix duplicates ".001", names unnamed glTF materials "Material_<i>" where
  // we use "material<i>", and calls unbound faces "<none>" where we use the default material "__default")
  const key = (n: string) => n.replace(/\.\d{3}$/, '').replace(/^material_?(\d+)$/i, 'material#$1').replace(/^<none>$/, '__default');
  const ours: Record<string, number> = {};
  for (let t = 0; t < nT; t++) { const n = key(s.materials[s.geometry.triMaterial[t]].name); ours[n] = (ours[n] ?? 0) + 1; }
  const theirs: Record<string, number> = {};
  for (const [k, v] of Object.entries(ref.trianglesByMaterial as Record<string, number>)) { const n = key(k); theirs[n] = (theirs[n] ?? 0) + v; }
  const tmBad = [...new Set([...Object.keys(ours), ...Object.keys(theirs)])].filter((k) => (ours[k] ?? 0) !== (theirs[k] ?? 0));
  add('triangles per material', tmBad.length === 0, tmBad.slice(0, 5).map((k) => `${k}: ${ours[k] ?? 0} vs ${theirs[k] ?? 0}`).join('; '));
  // Principled inputs (constants) per material name
  let mBad = 0;
  const mex: string[] = [];
  for (const m of ref.materials as Record<string, any>[]) {
    // glTF material names need not be unique (Blender suffixes ".001"): any of ours with the base name that matches
    const cands = s.materials.filter((x) => key(x.name) === key(m.name));
    if (!cands.length) { mBad++; mex.push(`${m.name} missing`); continue; }
    const wantOf = (md: (typeof cands)[number]): [string, boolean][] => [
      ['metallic', m.metallic.linked ? !!md.metallicRoughnessTexture : close(md.metallicFactor, m.metallic.value)],
      ['roughness', m.roughness.linked ? !!md.metallicRoughnessTexture : close(md.roughnessFactor, m.roughness.value)],
      ['ior', close(md.ior, m.ior.value)],
      ['transmission', m.transmission.linked ? !!md.transmissionTexture : close(md.transmissionFactor, m.transmission.value)],
      ['emission strength', close(md.emissiveStrength * Math.max(...md.emissiveFactor, md.emissiveTexture ? 1 : 0), m.emissionStrength.value * (m.emissionColor.linked ? 1 : Math.max(...m.emissionColor.value.slice(0, 3))))],
      ['base colour', m.baseColor.linked ? !!md.baseColorTexture || !!s.geometry.color0 : vclose(md.baseColorFactor.slice(0, 3), m.baseColor.value.slice(0, 3))],
      ['normal map', !!m.normalMap === !!md.normalTexture && (!m.normalMap || close(m.normalMap.strength, md.normalTexture!.scale))],
    ];
    const best = cands.map(wantOf).sort((x, y) => x.filter(([, ok]) => !ok).length - y.filter(([, ok]) => !ok).length)[0];
    const bad = best.filter(([, ok]) => !ok).map(([n]) => n);
    if (bad.length) { mBad++; mex.push(`${m.name}: ${bad.join(', ')}`); }
  }
  add(`Principled inputs per material (${(ref.materials as unknown[]).length})`, mBad === 0, mex.slice(0, 4).join('; '));
  // lights
  let lBad = 0;
  const lex: string[] = [];
  const refL = ref.lights as Record<string, any>[];
  if (refL.length !== s.lights.length) { lBad++; lex.push(`${s.lights.length} lights vs ${refL.length}`); }
  for (const l of refL) {
    const T = ({ POINT: 'point', SPOT: 'spot', SUN: 'sun', AREA: 'rect' } as Record<string, string>)[l.type];
    // Blender names the light OBJECT after the glTF node, we name it after the light: match by type and position
    const d2 = (x: LightData) => [12, 13, 14].reduce((a, k, j) => a + (x.matrix[k] - l.matrix[12 + j]) ** 2, 0);
    const o = s.lights.filter((x) => x.type === T).sort((a, b) => d2(a) - d2(b))[0];
    if (!o) { lBad++; lex.push(`${l.name} missing`); continue; }
    const ok = o.type === T && close(o.power, l.energy) && vclose(o.color, l.color) && vclose(Array.from(o.matrix).slice(12, 15), l.matrix.slice(12, 15), 2e-5)
      && (T !== 'spot' || (close(o.spotSize!, l.spotSize) && close(o.spotBlend!, l.spotBlend, 1e-4)));
    if (!ok) { lBad++; lex.push(`${l.name}: ours ${fmt({ t: o.type, P: o.power, c: o.color, ss: o.spotSize, sb: o.spotBlend })} Blender ${fmt({ t: T, P: l.energy, c: l.color, ss: l.spotSize, sb: l.spotBlend })}`); }
  }
  add(`lights (${refL.length})`, lBad === 0, lex.slice(0, 3).join('; '));
  return { file, kind: 'glb', ok: checks.every((c) => c.ok), checks, warnings: s.warnings, counts: { triangles: nT, materials: s.materials.length, lights: s.lights.length, textures: s.textures.length } };
}

async function gltfSource(abs: string) {
  const json = JSON.parse(readFileSync(abs, 'utf8')) as { buffers?: { uri?: string }[]; images?: { uri?: string }[] };
  const resources: Record<string, Uint8Array> = {};
  for (const r of [...(json.buffers ?? []), ...(json.images ?? [])]) if (r.uri && !r.uri.startsWith('data:')) resources[r.uri] = new Uint8Array(readFileSync(path.join(path.dirname(abs), decodeURIComponent(r.uri))));
  return { kind: 'gltf' as const, json: json as never, resources, name: path.basename(abs) };
}

if (process.argv[1] && import.meta.url === new URL(`file://${path.resolve(process.argv[1])}`).href) {
  const args = process.argv.slice(2);
  const oi = args.indexOf('--out');
  const out = oi >= 0 ? path.resolve(args[oi + 1]) : path.join(ROOT, 'validation/out/m7/fidelity');
  const files = args.filter((x, i) => !(i === oi || i === oi + 1));
  mkdirSync(out, { recursive: true });
  (async () => {
    const reps: FidelityReport[] = [];
    for (const f of files) {
      const rep = /\.usd[acz]?$/i.test(f) ? await usdFidelity(f, out) : await glbFidelity(f, out);
      reps.push(rep);
      console.log(`${rep.ok ? 'PASS' : 'FAIL'}  ${rep.kind === 'usd' ? '(viii-L)' : '(vii-L)'} ${f} ${JSON.stringify(rep.counts)}`);
      for (const c of rep.checks) console.log(`   ${c.ok ? 'ok  ' : 'FAIL'} ${c.item}${c.detail ? `  (${c.detail})` : ''}`);
    }
    writeFileSync(path.join(out, 'report.json'), `${JSON.stringify(reps, null, 1)}\n`);
    process.exit(reps.every((r) => r.ok) ? 0 : 1);
  })().catch((e) => { console.error(e); process.exit(2); });
}
void mul4; void stageMatrix;
