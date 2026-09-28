// Page-side normalizers: LightUSD worker results (next / legacy backends) and three.js USDLoader groups
// -> SceneDump (dump-types.ts), so they can be diffed against the pxr reference.
import * as THREE from 'three';
import type { CameraDump, DrawDump, InstancerDump, LightDump, MaterialDump, Mat16, SceneDump } from './dump-types.ts';
import { parseUsdaValue } from './usda-scan.ts';

type Rec = Record<string, any>; // loader output is untyped

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const bool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);
const arr = (v: unknown, n?: number): number[] | null =>
  Array.isArray(v) || ArrayBuffer.isView(v) ? Array.from(v as ArrayLike<number>).slice(0, n ?? undefined) : null;
const mat16 = (v: unknown): Mat16 | null => {
  const a = arr(v);
  return a && a.length >= 16 ? a.slice(0, 16) : null;
};
const DEG = 180 / Math.PI;

/** Replace typed arrays / long numeric arrays so raw objects stay small in the JSON dump. */
export function slim(v: unknown, depth = 0): unknown {
  if (ArrayBuffer.isView(v)) return `${v.constructor.name}(${(v as unknown as ArrayLike<number>).length})`;
  if (Array.isArray(v)) {
    if (v.length > 32 && typeof v[0] === 'number') return `Array(${v.length})`;
    return depth > 6 ? '[...]' : v.map((x) => slim(x, depth + 1));
  }
  if (v && typeof v === 'object') {
    if (depth > 6) return '{...}';
    const o: Rec = {};
    for (const [k, x] of Object.entries(v)) {
      if (k === 'materialXJson' || k === 'openPBRNodeGraphJson') o[k] = typeof x === 'string' ? `${x.length} chars` : x;
      else o[k] = slim(x, depth + 1);
    }
    return o;
  }
  return v;
}

function emptyDump(source: string, file: string): SceneDump {
  return {
    source, file, ok: true, stage: { upAxis: null, metersPerUnit: null },
    draws: [], lights: [], materials: [], pointInstancers: [], cameras: [], notes: [], raw: {},
  };
}

function trisByMaterial(ranges: { start: number; count: number; mat: string | null }[], totalTris: number, base: string | null) {
  const out: Record<string, number> = {};
  let covered = 0;
  for (const r of ranges) {
    const key = r.mat ?? '<none>';
    out[key] = (out[key] ?? 0) + r.count / 3;
    covered += r.count / 3;
  }
  if (totalTris - covered > 0) out[base ?? '<none>'] = (out[base ?? '<none>'] ?? 0) + (totalTris - covered);
  return out;
}

const LIGHT_TYPE: Record<string, string> = {
  point: 'sphere', sphere: 'sphere', spot: 'sphere', directional: 'distant', distant: 'distant',
  rect: 'rect', disk: 'disk', cylinder: 'cylinder', dome: 'dome', geometry: 'geometry',
};

function cameraFrom(c: Rec, path: string, world: unknown): CameraDump {
  return {
    path, focalLength: num(c.focalLength), horizontalAperture: num(c.horizontalAperture),
    verticalAperture: num(c.verticalAperture), world: mat16(world),
  };
}

// ---------------------------------------------------------------- next backend (RenderStream / tydra-next)

function nextMaterialInputs(m: Rec): Record<string, unknown> {
  const inputs: Record<string, unknown> = {};
  try {
    const ps = JSON.parse(m.materialXJson || '{}').previewSurface as Rec | undefined;
    for (const [k, v] of Object.entries(ps ?? {})) {
      const value = (v as Rec)?.value;
      if (!Array.isArray(value)) inputs[k] = v;
      else inputs[k] = /Color$/.test(k) || k === 'normal' ? value.slice(0, 3) : value[0];
    }
  } catch {
    /* fall through to flattened fields */
  }
  inputs.diffuseColor ??= m.baseColor;
  inputs.roughness ??= m.roughness;
  inputs.metallic ??= m.metallic;
  inputs.opacity ??= m.opacity;
  inputs.emissiveColor ??= m.emissive;
  if (num(m.opacityThreshold) !== null && m.opacityThreshold >= 0) inputs.opacityThreshold = m.opacityThreshold;
  return inputs;
}

// Stock worker wraps subset materials as {material: {...}}; raw RenderStream.getMesh() does not.
const matOf = (x: Rec | undefined): Rec | undefined => (x && typeof x.material === 'object' ? x.material : x);

function bboxOf(points: ArrayLike<number> | null | undefined): number[] | null {
  if (!points || points.length < 3) return null;
  const b = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (let i = 0; i < points.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      b[k] = Math.min(b[k], points[i + k]);
      b[k + 3] = Math.max(b[k + 3], points[i + k]);
    }
  }
  return b;
}

const PS_DEFAULTS: Record<string, unknown> = {
  useSpecularWorkflow: 0, specularColor: [0, 0, 0], clearcoat: 0, clearcoatRoughness: 0.01, opacityThreshold: 0, ior: 1.5,
};
const toBool = (v: unknown): boolean | null => (v === null || v === undefined ? null : v === true || v === 1);
const isUnder = (path: string, roots: Iterable<string>): boolean => {
  for (const r of roots) if (path === r || path.startsWith(r + '/')) return true;
  return false;
};

/**
 * next backend (RenderStream / tydra-next) -> SceneDump.
 * adapter=false: what the rc4 API returns as-is (meshes by their own primPath/worldMatrix).
 * adapter=true : what our UsdSceneSource does on top (see docs/decisions/usd.md): renderables matched to mesh
 *   nodes by prim path (never node.dataId), instance proxies of instanceable prims resolved to the shared mesh,
 *   class and PointInstancer-prototype subtrees skipped, instance transforms composed with the instancer's world,
 *   and fields rc4 drops patched from the root-layer USDA scan (treatAsPoint, spot radius, ior, clearcoat, doc).
 */
export function normalizeNext(d: Rec, file: string, source = 'lightusd-next', adapter = false): SceneDump {
  const out = emptyDump(source, file);
  const meta = (d.metadata ?? {}) as Rec;
  const scan = (adapter ? d.layerScan : null) as UsdaScanLike | null;
  out.stage = { upAxis: meta.upAxis ?? d.upAxis ?? null, metersPerUnit: num(meta.metersPerUnit), doc: scan?.doc ?? null };
  const meshes = (d.meshes ?? []) as Rec[];
  const nodes = (d.nodes ?? []) as Rec[];
  const pis = (d.pointInstancers ?? []) as Rec[];
  const nodeByPath = new Map(nodes.filter(Boolean).map((n) => [n.primPath as string, n]));
  const meshByPath = new Map(meshes.map((m) => [m.primPath as string, m]));

  const drawOf = (path: string, m: Rec, world: unknown): DrawDump => {
    const nPts = (m.points?.length ?? 0) / 3;
    const tris = m.indices?.length ? m.indices.length / 3 : nPts / 3;
    const subs = (m.materials ?? []) as Rec[];
    const base = m.material?.primPath || null;
    const ranges = ((m.submeshes ?? []) as Rec[]).map((s) => ({
      start: s.start, count: s.count, mat: matOf(subs[s.materialIndex])?.primPath ?? null,
    }));
    return {
      path, points: nPts, triangles: tris, material: base, trianglesByMaterial: trisByMaterial(ranges, tris, base),
      world: mat16(world), bbox: bboxOf(m.points),
    };
  };

  let unresolved = 0;
  if (!adapter) {
    for (const m of meshes) out.draws.push(drawOf(m.primPath, m, m.worldMatrix));
  } else {
    const skipRoots = new Set<string>([...(scan?.abstract ?? []), ...pis.flatMap((p) => (p.prototypePaths ?? []) as string[])]);
    const instRoots = Object.entries(scan?.instanceable ?? {});
    for (const n of nodes) {
      if (!n || n.type !== 'mesh' || isUnder(n.primPath, skipRoots)) continue;
      let m = meshByPath.get(n.primPath);
      if (!m) {
        // Instance proxy: rc4 emits the shared mesh once, under the first instance's path.
        const own = instRoots.find(([p]) => n.primPath.startsWith(p + '/'));
        const suffix = own ? n.primPath.slice(own[0].length) : null;
        const peer = own && instRoots.find(([p, ref]) => p !== own[0] && ref === own[1] && meshByPath.has(p + suffix));
        m = peer ? meshByPath.get(peer[0] + suffix) : undefined;
      }
      if (!m) { unresolved++; out.notes!.push(`mesh node ${n.primPath} has no renderable`); continue; }
      out.draws.push(drawOf(n.primPath, m, n.worldMatrix));
    }
  }

  const mats = new Map<string, MaterialDump>();
  for (const m of (d.materials ?? []) as Rec[]) mats.set(m.primPath, { path: m.primPath, inputs: nextMaterialInputs(m) });
  // Subset materials are not always in the deduplicated table; pick them up from the meshes too.
  for (const m of meshes) {
    for (const s of [m.material, ...((m.materials ?? []) as Rec[])]) {
      const mm = matOf(s);
      if (mm?.primPath && !mats.has(mm.primPath)) mats.set(mm.primPath, { path: mm.primPath, inputs: nextMaterialInputs(mm) });
    }
  }
  if (scan) {
    // Shader prims sit directly under their Material in Blender and hand-authored files.
    for (const [path, attrs] of Object.entries(scan.attrs)) {
      const mat = mats.get(path.slice(0, path.lastIndexOf('/')));
      if (!mat) continue;
      for (const [k, def] of Object.entries({ ...PS_DEFAULTS, specular: undefined })) {
        const v = parseUsdaValue(attrs[`inputs:${k}`]);
        if (v !== null) mat.inputs[k] = v;
        else if (def !== undefined && mat.inputs[k] === undefined) mat.inputs[k] = def;
      }
    }
  }
  out.materials = [...mats.values()];

  for (const l of (d.lights ?? []) as Rec[]) {
    if (!l) continue;
    const isSpot = l.type === 'spot';
    const shaped = isSpot || (num(l.shapingFocus) ?? 0) > 0 || (num(l.shapingConeSoftness) ?? 0) > 0;
    const type = LIGHT_TYPE[l.type] ?? String(l.type);
    const attrs = scan?.attrs[l.primPath] ?? {};
    const lt: LightDump = {
      path: l.primPath, type,
      intensity: num(l.intensity), exposure: num(l.exposure), color: arr(l.color, 3),
      normalize: bool(l.normalize), enableColorTemperature: bool(l.enableColorTemperature),
      colorTemperature: num(l.colorTemperature),
      radius: num(l.radius), width: num(l.width), height: num(l.height),
      angle: l.type === 'directional' || l.type === 'distant' ? num(l.angle) : null,
      treatAsPoint: bool(l.treatAsPoint),
      shaping: shaped
        ? {
            // next reports the spot cone as `angle` in RADIANS (distant `angle` stays in degrees).
            coneAngle: isSpot && num(l.angle) !== null ? l.angle * DEG : num(l.shapingConeAngle),
            coneSoftness: num(l.shapingConeSoftness), focus: num(l.shapingFocus), focusTint: arr(l.shapingFocusTint, 3),
          }
        : null,
      world: mat16(l.transform),
    };
    if (scan && type === 'sphere') {
      lt.treatAsPoint ??= toBool(parseUsdaValue(attrs.treatAsPoint)) ?? false; // schema default false
      lt.radius ??= (parseUsdaValue(attrs['inputs:radius']) as number | null) ?? 0.5; // schema default 0.5
    }
    out.lights.push(lt);
  }

  const draws = (d.pointInstanceDraws ?? []) as Rec[];
  for (const pi of pis) {
    const inst: InstancerDump = { path: pi.primPath, prototypes: (pi.prototypePaths ?? []) as string[], instances: [] };
    const piWorld = mat16(nodeByPath.get(pi.primPath)?.worldMatrix);
    for (const dr of draws.filter((x) => x.pointInstancerId === pi.index)) {
      // rc4 draw.transform is relative to the instancer prim and its 3x3 (orientation*scale) is TRANSPOSED
      // (translation is in the right place). Adapter: transpose the 3x3, then world = local * instancer world.
      const t = mat16(dr.transform);
      inst.instances.push({ proto: dr.meshPath ?? null, world: adapter && t && piWorld ? mulRow(transpose3(t), piWorld) : t });
    }
    out.pointInstancers.push(inst);
  }
  for (const c of (d.cameras ?? []) as Rec[]) if (c) out.cameras.push(cameraFrom(c, c.primPath, c.transform));
  out.counts = {
    meshes: meshes.length, nodes: nodes.length, unresolvedMeshNodes: unresolved, lights: (d.lights ?? []).length,
    materials: out.materials.length, pointInstancers: pis.length, pointInstanceDraws: draws.length,
    cameras: (d.cameras ?? []).length, unsupportedRenderables: (d.unsupportedRenderables ?? []).length,
  };
  out.raw = {
    metadata: slim(meta), stats: slim(d.stats), lights: slim(d.lights), firstMesh: slim(meshes[0]), pointInstancers: slim(pis),
    pointInstanceDraws: slim(draws), cameras: slim(d.cameras), unsupportedRenderables: slim(d.unsupportedRenderables),
    layerScan: adapter ? slim(d.layerScan) : undefined, layerScanError: d.layerScanError,
  };
  return out;
}

interface UsdaScanLike { doc: string | null; abstract: string[]; instanceable: Record<string, string>; attrs: Record<string, Record<string, string>> }

function transpose3(m: Mat16): Mat16 {
  const r = m.slice();
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) r[i * 4 + j] = m[j * 4 + i];
  return r;
}

function mulRow(a: Mat16, b: Mat16): Mat16 {
  const r = new Array<number>(16).fill(0);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 4; k++) r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
  return r;
}

// ---------------------------------------------------------------- legacy backend (LightUSDLoaderNative / Tydra)

export function normalizeLegacy(d: Rec, file: string): SceneDump {
  const out = emptyDump('lightusd-legacy', file);
  const meta = (d.metadata ?? {}) as Rec;
  out.stage = { upAxis: meta.upAxis ?? d.upAxis ?? null, metersPerUnit: num(meta.metersPerUnit) };
  const meshes = (d.meshes ?? []) as Rec[];
  const mats = (d.materials ?? []) as Rec[];
  const matPath = (id: number): string | null => mats[id]?.abs_path ?? null;
  const walk = (n: Rec | null): void => {
    if (!n) return;
    if (n.nodeType === 'mesh') {
      const m = meshes[n.contentId];
      if (m) {
        const idx = m.faceVertexIndices ?? m.indices;
        const tris = idx ? idx.length / 3 : 0;
        const base = matPath(m.materialId);
        const ranges = ((m.submeshes ?? []) as Rec[]).map((s) => ({ start: s.start, count: s.count, mat: matPath(s.materialId) }));
        out.draws.push({
          path: n.absPath, points: (m.points?.length ?? 0) / 3, triangles: tris, material: base,
          trianglesByMaterial: trisByMaterial(ranges, tris, base), world: mat16(n.globalMatrix ?? n.worldMatrix),
          bbox: bboxOf(m.points),
        });
      } else out.notes!.push(`mesh node ${n.absPath} contentId ${n.contentId} has no mesh`);
    }
    for (const c of (n.children ?? []) as Rec[]) walk(c);
  };
  walk(d.rootNode as Rec);

  for (const m of mats) {
    const ss = (m.surfaceShader ?? {}) as Rec;
    const inputs: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(ss)) if (k !== 'type') inputs[k] = v;
    out.materials.push({ path: m.abs_path, inputs });
  }

  for (const l of (d.lights ?? []) as Rec[]) {
    if (!l) continue;
    const t = LIGHT_TYPE[l.type] ?? String(l.type);
    // Legacy always reports shaping fields with defaults, so "ShapingAPI applied" is inferred from non-default values.
    const shaped = (num(l.shapingConeAngle) ?? 90) < 90 || (num(l.shapingConeSoftness) ?? 0) > 0 || (num(l.shapingFocus) ?? 0) > 0;
    out.lights.push({
      path: l.absPath, type: t, intensity: num(l.intensity), exposure: num(l.exposure), color: arr(l.color, 3),
      normalize: bool(l.normalize), enableColorTemperature: bool(l.enableColorTemperature),
      colorTemperature: num(l.colorTemperature),
      radius: t === 'sphere' || t === 'disk' || t === 'cylinder' ? num(l.radius) : null,
      width: t === 'rect' ? num(l.width) : null, height: t === 'rect' ? num(l.height) : null,
      angle: t === 'distant' ? num(l.angle) : null, treatAsPoint: bool(l.treatAsPoint),
      shaping: shaped
        ? { coneAngle: num(l.shapingConeAngle), coneSoftness: num(l.shapingConeSoftness), focus: num(l.shapingFocus), focusTint: arr(l.shapingFocusTint, 3) }
        : null,
      world: mat16(l.transform),
    });
  }
  for (const c of (d.cameras ?? []) as Rec[]) if (c) out.cameras.push(cameraFrom(c, c.absPath ?? c.primPath ?? c.name, c.transform ?? c.globalMatrix));
  out.counts = {
    meshes: meshes.length, lights: (d.lights ?? []).length, materials: mats.length, cameras: (d.cameras ?? []).length,
    pointInstancers: num(d.numPointInstancers) ?? 0, pointInstanceDraws: num(d.numPointInstanceDraws) ?? 0,
  };
  out.raw = {
    metadata: slim(meta), upAxis: d.upAxis, lights: slim(d.lights), firstMesh: slim(meshes[0]), materials: slim(mats),
    cameras: slim(d.cameras), dataKeys: Object.keys(d),
  };
  return out;
}

// ---------------------------------------------------------------- three.js USDLoader (fallback-1)

export function normalizeThree(group: THREE.Group, file: string): SceneDump {
  const out = emptyDump('three', file);
  group.updateMatrixWorld(true);
  out.stage = {
    upAxis: Math.abs(group.rotation.x + Math.PI / 2) < 1e-6 ? 'Z' : 'Y',
    metersPerUnit: group.scale.x,
  };
  const inv = group.matrixWorld.clone().invert();
  const rel = (o: THREE.Object3D): Mat16 => new THREE.Matrix4().multiplyMatrices(inv, o.matrixWorld).toArray();
  const pathOf = (o: THREE.Object3D): string => {
    const names: string[] = [];
    for (let p: THREE.Object3D | null = o; p && p !== group; p = p.parent) names.unshift(p.name);
    return '/' + names.join('/');
  };
  const mats = new Map<THREE.Material, MaterialDump>();
  const matName = (m: THREE.Material): string => {
    if (!mats.has(m)) {
      const s = m as THREE.MeshPhysicalMaterial;
      mats.set(m, {
        path: m.name || `three:${m.uuid.slice(0, 8)}`,
        inputs: {
          diffuseColor: s.color?.toArray(), roughness: s.roughness, metallic: s.metalness,
          emissiveColor: s.emissive ? s.emissive.clone().multiplyScalar(s.emissiveIntensity ?? 1).toArray() : undefined,
          opacity: s.opacity, opacityThreshold: s.alphaTest, ior: s.ior, clearcoat: s.clearcoat, clearcoatRoughness: s.clearcoatRoughness,
          threeType: m.type,
        },
      });
    }
    return mats.get(m)!.path;
  };
  group.traverse((o) => {
    const anyO = o as Rec;
    if (anyO.isMesh) {
      const mesh = o as THREE.Mesh;
      const g = mesh.geometry;
      const tris = (g.index ? g.index.count : g.attributes.position.count) / 3;
      const mlist = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      const ranges = Array.isArray(mesh.material)
        ? g.groups.map((gr) => ({ start: gr.start, count: gr.count, mat: matName(mlist[gr.materialIndex ?? 0]) }))
        : [];
      const base = matName(mlist[0]);
      const count = anyO.isInstancedMesh ? (mesh as THREE.InstancedMesh).count : 1;
      out.draws.push({
        path: pathOf(o), points: g.attributes.position.count, triangles: tris, material: base,
        trianglesByMaterial: trisByMaterial(ranges, tris, base), world: rel(o),
        bbox: bboxOf(g.attributes.position.array),
      });
      if (count > 1) out.notes!.push(`${pathOf(o)} is an InstancedMesh with ${count} instances`);
    } else if (anyO.isLight) {
      const l = o as THREE.Light;
      const spot = anyO.isSpotLight ? (o as THREE.SpotLight) : null;
      const rect = anyO.isRectAreaLight ? (o as THREE.RectAreaLight) : null;
      const type = anyO.isDirectionalLight ? 'distant' : anyO.isPointLight || spot ? 'sphere' : rect ? 'rect' : `three:${o.type}`;
      out.lights.push({
        path: pathOf(o), type, intensity: l.intensity, exposure: null, color: l.color.toArray(), normalize: null,
        enableColorTemperature: null, colorTemperature: null, radius: null,
        width: rect ? rect.width : null, height: rect ? rect.height : null, angle: null, treatAsPoint: null,
        shaping: spot ? { coneAngle: spot.angle * DEG, coneSoftness: spot.penumbra, focus: null, focusTint: null } : null,
        world: rel(o),
      });
    } else if (anyO.isCamera) {
      out.cameras.push({ path: pathOf(o), focalLength: null, horizontalAperture: null, verticalAperture: null, world: rel(o) });
    }
  });
  out.materials = [...mats.values()];
  out.counts = { draws: out.draws.length, lights: out.lights.length, materials: out.materials.length };
  return out;
}
