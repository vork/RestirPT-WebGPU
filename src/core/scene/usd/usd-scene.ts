// Minimal UsdSceneSource (plan §5 M2; docs/decisions/usd.md adapter rules 1–6): LightUSD raw data → SceneData.
// - UsdGeomMesh with composed xformOps (LightUSD worldMatrix), GeomSubset materials, instance proxies of
//   instanceable prims, PointInstancer draws (rc4 transform quirk); class/prototype subtrees are not drawn.
// - Stage → glTF canonical frame: upAxis Z → R_x(−90°), metersPerUnit scaling (stageMatrix), applied once.
// - UV v flip (USD st origin bottom-left → glTF top-left). Winding preserved; det < 0 flips it (TRI_FLIPPED).
// - UsdPreviewSurface constants → MaterialData (principled); textures (UsdUVTexture) are not supported yet (warned).
// - UsdLux Rect/Disk/Sphere(→ point, r = 0)/Sphere+Shaping(→ spot)/Distant(angle 0, Blender ×4) → usd-lights.ts.
// - Degenerate / non-finite triangles are dropped (dense primIds), like the glTF flatten (plan §1.3).
import { rigidMatrix } from '../gltf-loader.ts';
import { MAX_TRIANGLES } from '../flatten.ts';
import {
  TRI_ALPHA_MASK, TRI_EMISSIVE, TRI_FLIPPED, type Bounds, type CameraData, type LightData, type MaterialData, type SceneData,
} from '../types.ts';
import { parseUsdaValue, type UsdaScan } from './usda-scan.ts';
import { convertUsdLight, mul4, stageMatrix, type Mat4, type UsdLightInput } from './usd-lights.ts';
import type { Rec, UsdRaw } from './usd-native.ts';

export interface UsdConvertOptions {
  /** DistantLight ×4 (Blender-author quirk): 'auto' = when the root-layer doc starts with "Blender v". */
  distantQuirk?: 'auto' | 'always' | 'never';
}

export interface UsdSceneStats {
  draws: number;
  instanceProxyDraws: number;
  pointInstanceDraws: number;
  triangles: number;
  droppedDegenerate: number;
  droppedNonFinite: number;
  flippedTriangles: number;
  upAxis: string;
  metersPerUnit: number;
  doc: string | null;
  blenderAuthored: boolean;
}

interface Draw { path: string; mesh: Rec; world: Mat4; materialOverride?: string; }

const num = (v: unknown, d: number): number => (typeof v === 'number' && Number.isFinite(v) ? v : d);
const matOf = (x: Rec | undefined): Rec | undefined => (x && typeof x.material === 'object' ? x.material : x);
const isUnder = (path: string, roots: Iterable<string>): boolean => {
  for (const r of roots) if (path === r || path.startsWith(r + '/')) return true;
  return false;
};

function transpose3(m: number[]): number[] {
  const r = m.slice();
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) r[i * 4 + j] = m[j * 4 + i];
  return r;
}
/** Row-vector product a·b (USD convention) = column-major b·a. */
const mulRow = (a: number[], b: number[]): number[] => mul4(b, a);

export function usdToScene(raw: UsdRaw, opts: UsdConvertOptions = {}): { scene: SceneData; stats: UsdSceneStats } {
  const warnings: string[] = [];
  const warn = (m: string) => { if (!warnings.includes(m)) warnings.push(m); };
  const meta = raw.metadata ?? {};
  const scan: UsdaScan | undefined = raw.layerScan;
  if (!scan) warn(`USD root-layer scan failed (${raw.layerScanError ?? 'unknown'}): treatAsPoint, spot radius, ior, clearcoat and the Blender doc are unavailable (schema defaults used)`);
  if (scan?.externalLayers || raw.extraLayers.length) warn('USD file composes other layers; fields missing from LightUSD rc4 are only patched from the root layer');
  const upAxis = String(meta.upAxis ?? 'Y');
  const mpu = num(meta.metersPerUnit, 1);
  const W = stageMatrix(upAxis, mpu);
  const doc = scan?.doc ?? null;
  const q = opts.distantQuirk ?? 'auto';
  const blenderAuthored = q === 'always' || (q === 'auto' && !!doc && /^Blender v/.test(doc));

  // ---- renderables (adapter rules 1, 2) ----
  const nodes = (raw.nodes ?? []).filter(Boolean) as Rec[];
  const pis = raw.pointInstancers ?? [];
  const nodeByPath = new Map(nodes.map((n) => [n.primPath as string, n]));
  const meshByPath = new Map((raw.meshes ?? []).map((m) => [m.primPath as string, m]));
  const skipRoots = new Set<string>([...(scan?.abstract ?? []), ...pis.flatMap((p) => (p.prototypePaths ?? []) as string[])]);
  const instRoots = Object.entries(scan?.instanceable ?? {});
  const draws: Draw[] = [];
  let proxies = 0, hidden = 0, unresolved = 0;
  for (const n of nodes) {
    if (n.type !== 'mesh' || isUnder(n.primPath, skipRoots)) continue;
    if (n.visible === false) { hidden++; continue; }
    let m = meshByPath.get(n.primPath);
    if (!m) {
      const own = instRoots.find(([p]) => n.primPath.startsWith(p + '/'));
      const suffix = own ? n.primPath.slice(own[0].length) : null;
      const peer = own && instRoots.find(([p, ref]) => p !== own[0] && ref === own[1] && meshByPath.has(p + suffix));
      m = peer ? meshByPath.get(peer[0] + suffix) : undefined;
      if (m) proxies++;
    }
    if (!m) { unresolved++; continue; }
    draws.push({ path: n.primPath, mesh: m, world: Array.from(n.worldMatrix as ArrayLike<number>) });
  }
  if (hidden) warn(`${hidden} invisible mesh prim(s) skipped`);
  if (unresolved) warn(`${unresolved} mesh prim(s) without geometry (unsupported instancing) skipped`);
  let piDraws = 0;
  for (const pi of pis) {
    const piWorld = nodeByPath.get(pi.primPath)?.worldMatrix as number[] | undefined;
    if (!piWorld) { warn(`PointInstancer ${pi.primPath}: no node; skipped`); continue; }
    for (const dr of (raw.pointInstanceDraws ?? []).filter((x) => x.pointInstancerId === pi.index)) {
      const m = meshByPath.get(dr.meshPath);
      if (!m || !dr.transform) { warn(`PointInstancer ${pi.primPath}: prototype mesh ${dr.meshPath} missing; instance skipped`); continue; }
      // rc4: transform is instancer-relative with a transposed 3×3 (usd.md finding 5).
      draws.push({ path: `${pi.primPath}[${dr.instanceIndex}]`, mesh: m, world: mulRow(transpose3(Array.from(dr.transform)), Array.from(piWorld)), materialOverride: dr.materialPath || undefined });
      piDraws++;
    }
  }
  if ((raw.unsupportedRenderables ?? []).length) warn(`${raw.unsupportedRenderables.length} unsupported renderable(s) (points/curves/...) skipped`);

  // ---- materials (adapter rule 4) ----
  const materials: MaterialData[] = [];
  const matIndex = new Map<string, number>();
  const matRecs = new Map<string, Rec>();
  for (const m of raw.meshes ?? []) for (const s of [m.material, ...((m.materials ?? []) as Rec[])]) {
    const mm = matOf(s);
    if (mm?.primPath && !matRecs.has(mm.primPath)) matRecs.set(mm.primPath, mm);
  }
  const materialFor = (path: string | null): number => {
    const key = path ?? '__default';
    let i = matIndex.get(key);
    if (i === undefined) {
      i = materials.length;
      matIndex.set(key, i);
      materials.push(path ? previewSurfaceMaterial(path, matRecs.get(path), scan, warn) : defaultMaterial());
    }
    return i;
  };

  // ---- geometry ----
  let totalTris = 0, totalVerts = 0;
  for (const d of draws) {
    const idx = d.mesh.indices as Uint32Array | null;
    const nPts = (d.mesh.points?.length ?? 0) / 3;
    totalTris += idx ? Math.floor(idx.length / 3) : Math.floor(nPts / 3);
    totalVerts += nPts;
  }
  if (totalTris > MAX_TRIANGLES) throw new Error(`USD scene has ${totalTris} triangles; the limit is ${MAX_TRIANGLES}`);
  const positions = new Float32Array(totalVerts * 3), normals = new Float32Array(totalVerts * 3), uv0 = new Float32Array(totalVerts * 2);
  let indices = new Uint32Array(totalTris * 3), triMaterial = new Uint32Array(totalTris), triFlags = new Uint32Array(totalTris);
  const bmin = [Infinity, Infinity, Infinity], bmax = [-Infinity, -Infinity, -Infinity];
  let vBase = 0, tOut = 0, degenerate = 0, nonFinite = 0, flippedTris = 0, noNormals = 0, noUv = 0;
  for (const d of draws) {
    const m = d.mesh;
    const P = m.points as Float32Array | null;
    if (!P || P.length < 9) continue;
    const nv = P.length / 3;
    const N = m.normals && (m.normals as Float32Array).length === nv * 3 ? (m.normals as Float32Array) : null;
    const T = m.uv0 && (m.uv0 as Float32Array).length === nv * 2 ? (m.uv0 as Float32Array) : null;
    if (!N) noNormals++;
    if (!T) noUv++;
    const M = mul4(W, d.world);
    const a00 = M[0], a10 = M[1], a20 = M[2], a01 = M[4], a11 = M[5], a21 = M[6], a02 = M[8], a12 = M[9], a22 = M[10];
    const c00 = a11 * a22 - a12 * a21, c01 = a12 * a20 - a10 * a22, c02 = a10 * a21 - a11 * a20;
    const c10 = a02 * a21 - a01 * a22, c11 = a00 * a22 - a02 * a20, c12 = a01 * a20 - a00 * a21;
    const c20 = a01 * a12 - a02 * a11, c21 = a02 * a10 - a00 * a12, c22 = a00 * a11 - a01 * a10;
    const det = a00 * c00 + a01 * c01 + a02 * c02;
    const flipped = det < 0;
    for (let v = 0; v < nv; v++) {
      const x = P[3 * v], y = P[3 * v + 1], z = P[3 * v + 2];
      const o = (vBase + v) * 3;
      positions[o] = a00 * x + a01 * y + a02 * z + M[12];
      positions[o + 1] = a10 * x + a11 * y + a12 * z + M[13];
      positions[o + 2] = a20 * x + a21 * y + a22 * z + M[14];
      if (N) { // inverse-transpose = cofactor / det; the 1/det scale is removed by the normalization (sign kept)
        const nx = N[3 * v], ny = N[3 * v + 1], nz = N[3 * v + 2];
        const s = det < 0 ? -1 : 1;
        const wx = s * (c00 * nx + c01 * ny + c02 * nz), wy = s * (c10 * nx + c11 * ny + c12 * nz), wz = s * (c20 * nx + c21 * ny + c22 * nz);
        const l = Math.hypot(wx, wy, wz);
        if (l > 0 && Number.isFinite(l)) { normals[o] = wx / l; normals[o + 1] = wy / l; normals[o + 2] = wz / l; }
      }
      if (T) { uv0[(vBase + v) * 2] = T[2 * v]; uv0[(vBase + v) * 2 + 1] = 1 - T[2 * v + 1]; }
    }
    const idx = (m.indices as Uint32Array | null) ?? Uint32Array.from({ length: nv - (nv % 3) }, (_, i) => i);
    const nt = Math.floor(idx.length / 3);
    // per-triangle material: GeomSubset ranges (start/count in index units), base material elsewhere
    const baseMat = d.materialOverride ?? (matOf(m.material)?.primPath as string | undefined) ?? null;
    const triMat = new Int32Array(nt).fill(-1);
    for (const s of (m.submeshes ?? []) as Rec[]) {
      const p = matOf(((m.materials ?? []) as Rec[])[s.materialIndex])?.primPath as string | undefined;
      const mi = materialFor(p ?? baseMat);
      for (let t = Math.floor(s.start / 3); t < Math.min(nt, Math.floor((s.start + s.count) / 3)); t++) triMat[t] = mi;
    }
    const baseIdx = materialFor(baseMat);
    const k1 = flipped ? 2 : 1, k2 = flipped ? 1 : 2;
    for (let t = 0; t < nt; t++) {
      const i0 = vBase + idx[3 * t], i1 = vBase + idx[3 * t + k1], i2 = vBase + idx[3 * t + k2];
      const q = triQuality(positions, i0, i1, i2);
      if (q === 'nonfinite') { nonFinite++; continue; }
      if (q === 'degenerate') { degenerate++; continue; }
      indices[3 * tOut] = i0; indices[3 * tOut + 1] = i1; indices[3 * tOut + 2] = i2;
      const mi = triMat[t] >= 0 ? triMat[t] : baseIdx;
      triMaterial[tOut] = mi;
      const mat = materials[mi];
      let f = flipped ? TRI_FLIPPED : 0;
      if (Math.max(...mat.emissiveFactor) * mat.emissiveStrength > 0) f |= TRI_EMISSIVE;
      if (mat.alphaMode === 'MASK' && mat.baseColorFactor[3] !== 1) f |= TRI_ALPHA_MASK;
      triFlags[tOut] = f;
      if (flipped) flippedTris++;
      for (const vi of [i0, i1, i2]) for (let k = 0; k < 3; k++) {
        const p = positions[vi * 3 + k];
        if (p < bmin[k]) bmin[k] = p;
        if (p > bmax[k]) bmax[k] = p;
      }
      // zero/missing normals → face normal (flat)
      tOut++;
    }
    vBase += nv;
  }
  repairNormals(positions, normals, indices.subarray(0, tOut * 3));
  if (noNormals) warn(`${noNormals} USD mesh(es) without normals: face normals used`);
  if (noUv) warn(`${noUv} USD mesh(es) without st/uv: UVs are 0`);
  if (degenerate) warn(`dropped ${degenerate} zero-area triangle(s); primIds are dense over the kept triangles`);
  if (nonFinite) warn(`dropped ${nonFinite} triangle(s) with non-finite positions`);
  if (tOut < totalTris) { indices = indices.slice(0, tOut * 3); triMaterial = triMaterial.slice(0, tOut); triFlags = triFlags.slice(0, tOut); }
  const bounds: Bounds = tOut ? { min: [bmin[0], bmin[1], bmin[2]], max: [bmax[0], bmax[1], bmax[2]] } : { min: [0, 0, 0], max: [0, 0, 0] };

  // ---- lights (adapter rule 3) ----
  const lights: LightData[] = [];
  for (const l of (raw.lights ?? []) as Rec[]) {
    if (!l) continue;
    const input = lightInput(l, scan);
    if (!input.world) { warn(`light ${l.primPath}: no transform; skipped`); continue; }
    const r = convertUsdLight(input.light, mul4(W, input.world), lights.length, { blenderAuthored });
    for (const w of r.warnings) warn(w);
    if (r.light) {
      if (r.light.simplified) warn(`light ${l.primPath}: simplified (${r.light.simplified})`);
      lights.push(r.light);
    }
  }

  // ---- cameras ----
  const cameras: CameraData[] = [];
  for (const c of (raw.cameras ?? []) as Rec[]) {
    if (!c) continue;
    if (c.type && c.type !== 'perspective') { warn(`camera ${c.primPath}: ${c.type} projection skipped (perspective only)`); continue; }
    const t = c.transform as number[] | undefined;
    if (!t) continue;
    const va = num(c.verticalAperture, 0), fl = num(c.focalLength, 0);
    const yfov = va > 0 && fl > 0 ? 2 * Math.atan(va / (2 * fl)) : num(c.fovY, Math.PI / 4);
    cameras.push({ name: c.name ?? c.primPath ?? `camera${cameras.length}`, matrix: rigidMatrix(mul4(W, Array.from(t))), yfov, znear: num(c.nearClip, 1e-4) * mpu });
  }

  const scene: SceneData = {
    name: raw.name,
    geometry: { positions, normals, tangents: new Float32Array(totalVerts * 4), uv0, indices, triMaterial, triFlags },
    materials, textures: [], lights, cameras, bounds, warnings,
  };
  return {
    scene,
    stats: {
      draws: draws.length, instanceProxyDraws: proxies, pointInstanceDraws: piDraws, triangles: tOut,
      droppedDegenerate: degenerate, droppedNonFinite: nonFinite, flippedTriangles: flippedTris,
      upAxis, metersPerUnit: mpu, doc, blenderAuthored,
    },
  };
}

// ---------------------------------------------------------------------------------------------------------------

function lightInput(l: Rec, scan: UsdaScan | undefined): { light: UsdLightInput; world: number[] | null } {
  const attrs = scan?.attrs[l.primPath] ?? {};
  const t = String(l.type);
  const isSpot = t === 'spot';
  const type = isSpot || t === 'point' || t === 'sphere' ? 'sphere' : t === 'directional' || t === 'distant' ? 'distant' : t;
  const shaped = isSpot || num(l.shapingFocus, 0) > 0 || num(l.shapingConeSoftness, 0) > 0;
  const bool = (v: unknown): boolean | undefined => (v === null || v === undefined ? undefined : v === true || v === 1);
  const radius = typeof l.radius === 'number' ? l.radius : (parseUsdaValue(attrs['inputs:radius']) as number | null) ?? undefined;
  return {
    world: Array.isArray(l.transform) || ArrayBuffer.isView(l.transform) ? Array.from(l.transform as ArrayLike<number>) : null,
    light: {
      primPath: l.primPath, type,
      intensity: num(l.intensity, 1), exposure: num(l.exposure, 0),
      color: (l.color ? [l.color[0], l.color[1], l.color[2]] : [1, 1, 1]) as [number, number, number],
      normalize: l.normalize === true, // USD fallback false
      enableColorTemperature: l.enableColorTemperature === true,
      colorTemperature: num(l.colorTemperature, 6500),
      radius: type === 'sphere' ? (radius ?? 0.5) : radius,
      width: typeof l.width === 'number' ? l.width : undefined,
      height: typeof l.height === 'number' ? l.height : undefined,
      angle: type === 'distant' ? num(l.angle, 0.53) : undefined, // next: DistantLight angle in DEGREES
      treatAsPoint: type === 'sphere' ? (bool(l.treatAsPoint) ?? bool(parseUsdaValue(attrs.treatAsPoint)) ?? false) : undefined,
      // next reports the spot cone as `angle` in RADIANS (usd.md finding 6)
      shaping: shaped ? {
        coneAngle: isSpot && typeof l.angle === 'number' ? l.angle * 180 / Math.PI : num(l.shapingConeAngle, 90),
        coneSoftness: num(l.shapingConeSoftness, 0), focus: num(l.shapingFocus, 0),
      } : null,
    },
  };
}

function previewInputs(m: Rec | undefined): Record<string, unknown> {
  const inputs: Record<string, unknown> = {};
  if (!m) return inputs;
  try {
    const ps = JSON.parse(m.materialXJson || '{}').previewSurface as Rec | undefined;
    for (const [k, v] of Object.entries(ps ?? {})) {
      const value = (v as Rec)?.value;
      if ((v as Rec)?.texture || (v as Rec)?.connection) inputs[`${k}:texture`] = true;
      if (!Array.isArray(value)) inputs[k] = value ?? v;
      else inputs[k] = /Color$/.test(k) || k === 'normal' ? value.slice(0, 3) : value[0];
    }
  } catch { /* fall through to the flattened fields */ }
  inputs.diffuseColor ??= m.baseColor;
  inputs.roughness ??= m.roughness;
  inputs.metallic ??= m.metallic;
  inputs.opacity ??= m.opacity;
  inputs.emissiveColor ??= m.emissive;
  if (typeof m.opacityThreshold === 'number' && m.opacityThreshold >= 0) inputs.opacityThreshold = m.opacityThreshold;
  const tm = m.textureMetadata;
  if (tm && typeof tm === 'object' && Object.keys(tm).length) inputs.__textures = Object.keys(tm);
  return inputs;
}

const v3 = (v: unknown, d: [number, number, number]): [number, number, number] =>
  Array.isArray(v) && v.length >= 3 && v.slice(0, 3).every((x) => typeof x === 'number') ? [v[0], v[1], v[2]] : d;

/** UsdPreviewSurface constants → MaterialData (principled). Unsupported inputs are warned, never silently used. */
function previewSurfaceMaterial(path: string, rec: Rec | undefined, scan: UsdaScan | undefined, warn: (m: string) => void): MaterialData {
  const inp = previewInputs(rec);
  // Shader prims sit directly under their Material (Blender and hand-authored files): patch fields rc4 drops.
  const patched: Record<string, unknown> = {};
  if (scan) for (const [p, attrs] of Object.entries(scan.attrs)) {
    if (p.slice(0, p.lastIndexOf('/')) !== path) continue;
    for (const k of ['ior', 'clearcoat', 'clearcoatRoughness', 'useSpecularWorkflow', 'specularColor', 'specular', 'opacityThreshold']) {
      const v = parseUsdaValue(attrs[`inputs:${k}`]);
      if (v !== null) patched[k] = v;
    }
  }
  const get = (k: string) => patched[k] ?? inp[k];
  if (!rec) warn(`material ${path}: not reported by LightUSD; Cycles default surface used`);
  if (rec && rec.shaderType && rec.shaderType !== 'PreviewSurface') warn(`material ${path}: ${rec.shaderType} shader; only UsdPreviewSurface constants are read`);
  if (inp.__textures || Object.keys(inp).some((k) => k.endsWith(':texture'))) warn(`material ${path}: UsdUVTexture inputs are not supported yet (M7); constants used`);
  if (num(get('clearcoat'), 0) > 0) warn(`material ${path}: clearcoat ignored (v1)`);
  if (num(get('useSpecularWorkflow'), 0) === 1) warn(`material ${path}: specular workflow ignored (metallic workflow used)`);
  const opacity = num(get('opacity'), 1);
  const threshold = num(get('opacityThreshold'), 0);
  let alphaMode: MaterialData['alphaMode'] = 'OPAQUE', alphaCutoff = 0.5;
  if (threshold > 0) { alphaMode = 'MASK'; alphaCutoff = threshold; }
  else if (opacity < 1) { alphaMode = 'MASK'; alphaCutoff = 0.5; warn(`material ${path}: opacity ${opacity} without opacityThreshold rendered as MASK 0.5 (v1: no blending)`); }
  const blenderSpecular = get('specular');
  return {
    name: path.split('/').pop() || path,
    baseColorFactor: [...v3(get('diffuseColor'), [0.18, 0.18, 0.18]), opacity] as MaterialData['baseColorFactor'],
    metallicFactor: num(get('metallic'), 0),
    roughnessFactor: num(get('roughness'), 0.5),
    emissiveFactor: v3(get('emissiveColor'), [0, 0, 0]),
    emissiveStrength: 1,
    ior: num(get('ior'), 1.5),
    // Blender writes its Specular IOR Level as the non-standard `specular` (0.5 = default) = 0.5·KHR specularFactor.
    specularFactor: typeof blenderSpecular === 'number' ? 2 * blenderSpecular : 1,
    specularColorFactor: [1, 1, 1],
    transmissionFactor: 0,
    alphaMode,
    alphaCutoff,
    doubleSided: true,
    model: 'principled',
  };
}

/** Unbound geometry: Cycles' default surface (Principled base 0.8, roughness 0.5), as in the glTF loader. */
function defaultMaterial(): MaterialData {
  return {
    name: '__default', baseColorFactor: [0.8, 0.8, 0.8, 1], metallicFactor: 0, roughnessFactor: 0.5, emissiveFactor: [0, 0, 0],
    emissiveStrength: 1, ior: 1.5, specularFactor: 1, specularColorFactor: [1, 1, 1], transmissionFactor: 0,
    alphaMode: 'OPAQUE', alphaCutoff: 0.5, doubleSided: true, model: 'principled',
  };
}

function triQuality(p: Float32Array, i0: number, i1: number, i2: number): 'ok' | 'degenerate' | 'nonfinite' {
  for (const i of [i0, i1, i2]) if (!Number.isFinite(p[i * 3]) || !Number.isFinite(p[i * 3 + 1]) || !Number.isFinite(p[i * 3 + 2])) return 'nonfinite';
  const ax = p[i1 * 3] - p[i0 * 3], ay = p[i1 * 3 + 1] - p[i0 * 3 + 1], az = p[i1 * 3 + 2] - p[i0 * 3 + 2];
  const bx = p[i2 * 3] - p[i0 * 3], by = p[i2 * 3 + 1] - p[i0 * 3 + 1], bz = p[i2 * 3 + 2] - p[i0 * 3 + 2];
  const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
  const a2 = nx * nx + ny * ny + nz * nz;
  if (!Number.isFinite(a2)) return 'nonfinite';
  return a2 > 0 ? 'ok' : 'degenerate';
}

/** Vertices whose normal is still zero (missing/invalid) get the normalized sum of adjacent face normals. */
function repairNormals(pos: Float32Array, nrm: Float32Array, idx: Uint32Array): void {
  const nv = pos.length / 3;
  let bad = false;
  for (let v = 0; v < nv && !bad; v++) bad = nrm[3 * v] === 0 && nrm[3 * v + 1] === 0 && nrm[3 * v + 2] === 0;
  if (!bad) return;
  const acc = new Float64Array(nv * 3);
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t], b = idx[t + 1], c = idx[t + 2];
    const ex = pos[3 * b] - pos[3 * a], ey = pos[3 * b + 1] - pos[3 * a + 1], ez = pos[3 * b + 2] - pos[3 * a + 2];
    const fx = pos[3 * c] - pos[3 * a], fy = pos[3 * c + 1] - pos[3 * a + 1], fz = pos[3 * c + 2] - pos[3 * a + 2];
    const nx = ey * fz - ez * fy, ny = ez * fx - ex * fz, nz = ex * fy - ey * fx;
    const l = Math.hypot(nx, ny, nz) || 1;
    for (const v of [a, b, c]) { acc[3 * v] += nx / l; acc[3 * v + 1] += ny / l; acc[3 * v + 2] += nz / l; }
  }
  for (let v = 0; v < nv; v++) {
    if (nrm[3 * v] !== 0 || nrm[3 * v + 1] !== 0 || nrm[3 * v + 2] !== 0) continue;
    const l = Math.hypot(acc[3 * v], acc[3 * v + 1], acc[3 * v + 2]);
    if (l > 0) { nrm[3 * v] = acc[3 * v] / l; nrm[3 * v + 1] = acc[3 * v + 1] / l; nrm[3 * v + 2] = acc[3 * v + 2] / l; }
    else nrm[3 * v + 1] = 1;
  }
}
