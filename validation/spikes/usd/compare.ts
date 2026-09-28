// Diff a loader SceneDump against the pxr reference dump (same file). Used by run-spike.ts in Node.
import type { DrawDump, LightDump, Mat16, MaterialDump, SceneDump } from './dump-types.ts';

export type Status = 'ok' | 'mismatch' | 'missing';
export interface Mismatch { path: string; field: string; ref: unknown; got: unknown; status: Status }
export interface FieldTally { ok: number; mismatch: number; missing: number }

export interface Comparison {
  file: string;
  source: string;
  loaded: boolean;
  error?: string;
  stage: Record<string, Status>;
  lights: { ref: number; matched: number; unmatched: string[]; extra: string[]; fields: Record<string, FieldTally> };
  draws: {
    ref: number; got: number; matched: number; unmatched: string[]; extra: string[];
    triangles: FieldTally; world: FieldTally; bbox: FieldTally; materialAssignment: FieldTally;
  };
  materials: { ref: number; matched: number; unmatched: string[]; fields: Record<string, FieldTally> };
  pointInstancers: { refInstances: number; gotInstances: number; world: FieldTally; relativeInsteadOfWorld: number };
  mismatches: Mismatch[];
}

const TOL = 1e-4;
const close = (a: number, b: number, tol = TOL): boolean => Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));

export function sameValue(ref: unknown, got: unknown, tol = TOL): boolean {
  if (typeof ref === 'number' && typeof got === 'number') return close(ref, got, tol);
  if (typeof ref === 'boolean' || typeof got === 'boolean') return Number(ref) === Number(got);
  if (Array.isArray(ref) && Array.isArray(got)) {
    const n = Math.min(ref.length, got.length);
    if (n === 0 || Math.abs(ref.length - got.length) > 1) return false; // allow rgb vs rgba
    for (let i = 0; i < n; i++) if (!sameValue(ref[i], got[i], tol)) return false;
    return true;
  }
  return ref === got;
}

// Row-vector 4x4 helpers (USD layout).
function mul(a: Mat16, b: Mat16): Mat16 {
  const r = new Array<number>(16).fill(0);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 4; k++) r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
  return r;
}
function inv(m: Mat16): Mat16 {
  // Gauss-Jordan; matrices here are well-conditioned affine transforms.
  const a = m.slice();
  const r = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  for (let c = 0; c < 4; c++) {
    let p = c;
    for (let i = c + 1; i < 4; i++) if (Math.abs(a[i * 4 + c]) > Math.abs(a[p * 4 + c])) p = i;
    for (let j = 0; j < 4; j++) {
      [a[c * 4 + j], a[p * 4 + j]] = [a[p * 4 + j], a[c * 4 + j]];
      [r[c * 4 + j], r[p * 4 + j]] = [r[p * 4 + j], r[c * 4 + j]];
    }
    const d = a[c * 4 + c];
    for (let j = 0; j < 4; j++) { a[c * 4 + j] /= d; r[c * 4 + j] /= d; }
    for (let i = 0; i < 4; i++) {
      if (i === c) continue;
      const f = a[i * 4 + c];
      for (let j = 0; j < 4; j++) { a[i * 4 + j] -= f * a[c * 4 + j]; r[i * 4 + j] -= f * r[c * 4 + j]; }
    }
  }
  return r;
}

const tally = (): FieldTally => ({ ok: 0, mismatch: 0, missing: 0 });
const leaf = (p: string): string => p.split('/').pop() ?? p;

const LIGHT_FIELDS = [
  'type', 'intensity', 'exposure', 'color', 'normalize', 'enableColorTemperature', 'colorTemperature',
  'radius', 'width', 'height', 'angle', 'treatAsPoint',
  'shaping.coneAngle', 'shaping.coneSoftness', 'shaping.focus', 'shaping.focusTint', 'world',
] as const;
const MATERIAL_FIELDS = [
  'diffuseColor', 'emissiveColor', 'metallic', 'roughness', 'opacity', 'opacityThreshold', 'ior',
  'clearcoat', 'clearcoatRoughness', 'useSpecularWorkflow', 'specularColor', 'specular',
] as const;

function get(o: LightDump, f: string): unknown {
  if (f.startsWith('shaping.')) return o.shaping ? (o.shaping as unknown as Record<string, unknown>)[f.slice(8)] : null;
  return (o as unknown as Record<string, unknown>)[f];
}

function materialSig(dump: SceneDump, path: string): string {
  const m = dump.materials.find((x) => x.path === path);
  if (!m) return `path:${path}`;
  const f = (v: unknown): string => (Array.isArray(v) ? v.slice(0, 3).map((x) => Number(x).toFixed(3)).join(',') : Number(v).toFixed(3));
  return `d=${f(m.inputs.diffuseColor)} r=${f(m.inputs.roughness)} m=${f(m.inputs.metallic)}`;
}
function sigTris(dump: SceneDump, d: DrawDump): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [p, n] of Object.entries(d.trianglesByMaterial)) {
    const s = materialSig(dump, p);
    out[s] = (out[s] ?? 0) + n;
  }
  return out;
}

export function compareDumps(ref: SceneDump, got: SceneDump): Comparison {
  // three rebuilds paths from Object3D names; its materials only carry the prim name -> leaf fallback.
  const byLeaf = got.source === 'three';
  const key = (p: string): string => p;
  const c: Comparison = {
    file: got.file, source: got.source, loaded: got.ok, error: got.error, stage: {},
    lights: { ref: ref.lights.length, matched: 0, unmatched: [], extra: [], fields: {} },
    draws: {
      ref: ref.draws.length, got: got.draws.length, matched: 0, unmatched: [], extra: [],
      triangles: tally(), world: tally(), bbox: tally(), materialAssignment: tally(),
    },
    materials: { ref: ref.materials.length, matched: 0, unmatched: [], fields: {} },
    pointInstancers: { refInstances: 0, gotInstances: 0, world: tally(), relativeInsteadOfWorld: 0 },
    mismatches: [],
  };
  if (!got.ok) return c;
  const note = (t: FieldTally, s: Status, path: string, field: string, r: unknown, g: unknown): void => {
    t[s]++;
    if (s !== 'ok') c.mismatches.push({ path, field, ref: r, got: g, status: s });
  };
  const status = (r: unknown, g: unknown): Status => (g === null || g === undefined ? (r === null || r === undefined ? 'ok' : 'missing') : r === null || r === undefined ? 'ok' : sameValue(r, g) ? 'ok' : 'mismatch');

  for (const f of ['upAxis', 'metersPerUnit'] as const) {
    const s = status(ref.stage[f], got.stage[f]);
    c.stage[f] = s;
    if (s !== 'ok') c.mismatches.push({ path: '<stage>', field: f, ref: ref.stage[f], got: got.stage[f], status: s });
  }

  // Lights
  for (const f of LIGHT_FIELDS) c.lights.fields[f] = tally();
  const gotLights = new Map(got.lights.map((l) => [key(l.path), l]));
  for (const rl of ref.lights) {
    const gl = gotLights.get(key(rl.path));
    if (!gl) { c.lights.unmatched.push(rl.path); continue; }
    gotLights.delete(key(rl.path));
    c.lights.matched++;
    for (const f of LIGHT_FIELDS) {
      const r = get(rl, f);
      const g = get(gl, f);
      let s: Status;
      if (f.startsWith('shaping.') && !rl.shaping) s = gl.shaping ? 'mismatch' : 'ok';
      else s = status(r, g);
      note(c.lights.fields[f], s, rl.path, f, r, g);
    }
  }
  c.lights.extra = [...gotLights.keys()];

  // Materials
  for (const f of MATERIAL_FIELDS) c.materials.fields[f] = tally();
  const gotMats = new Map<string, MaterialDump>(got.materials.map((m) => [key(m.path), m]));
  for (const rm of ref.materials) {
    const gm = gotMats.get(key(rm.path)) ?? (byLeaf ? got.materials.find((m) => leaf(m.path) === leaf(rm.path)) : undefined);
    if (!gm) { c.materials.unmatched.push(rm.path); continue; }
    c.materials.matched++;
    for (const f of MATERIAL_FIELDS) {
      const r = rm.inputs[f];
      if (r === undefined) continue;
      note(c.materials.fields[f], status(r, gm.inputs[f]), rm.path, `material.${f}`, r, gm.inputs[f]);
    }
  }

  // Draws
  const gotDraws = new Map<string, DrawDump>(got.draws.map((d) => [key(d.path), d]));
  for (const rd of ref.draws) {
    const gd = gotDraws.get(key(rd.path));
    if (!gd) { c.draws.unmatched.push(rd.path); continue; }
    gotDraws.delete(key(rd.path));
    c.draws.matched++;
    note(c.draws.triangles, rd.triangles === gd.triangles ? 'ok' : 'mismatch', rd.path, 'triangles', rd.triangles, gd.triangles);
    note(c.draws.world, gd.world ? (sameValue(rd.world, gd.world) ? 'ok' : 'mismatch') : 'missing', rd.path, 'world', rd.world, gd.world);
    if (rd.bbox) note(c.draws.bbox, gd.bbox ? (sameValue(rd.bbox, gd.bbox) ? 'ok' : 'mismatch') : 'missing', rd.path, 'bbox', rd.bbox, gd.bbox);
    const rs = sigTris(ref, rd);
    const gs = sigTris(got, gd);
    const same = Object.keys(rs).length === Object.keys(gs).length && Object.entries(rs).every(([k, v]) => gs[k] === v);
    note(c.draws.materialAssignment, same ? 'ok' : 'mismatch', rd.path, 'trianglesByMaterial', rs, gs);
  }
  c.draws.extra = [...gotDraws.keys()];

  // PointInstancers
  for (const rp of ref.pointInstancers) {
    c.pointInstancers.refInstances += rp.instances.length;
    const gp = got.pointInstancers.find((p) => key(p.path) === key(rp.path));
    if (!gp) { c.mismatches.push({ path: rp.path, field: 'pointInstancer', ref: rp.instances.length, got: null, status: 'missing' }); continue; }
    c.pointInstancers.gotInstances += gp.instances.length;
    rp.instances.forEach((ri, i) => {
      const gi = gp.instances[i];
      if (!gi?.world || !ri.world) { note(c.pointInstancers.world, 'missing', `${rp.path}[${i}]`, 'instance.world', ri.world, gi?.world ?? null); return; }
      // orientations are quath: pxr uses the raw half values, loaders may renormalize -> 1e-3 tolerance.
      const ok = sameValue(ri.world, gi.world, 1e-3);
      if (!ok && rp.world && sameValue(mul(ri.world, inv(rp.world)), gi.world, 1e-3)) c.pointInstancers.relativeInsteadOfWorld++;
      note(c.pointInstancers.world, ok ? 'ok' : 'mismatch', `${rp.path}[${i}]`, 'instance.world', ri.world, gi.world);
    });
  }
  return c;
}
