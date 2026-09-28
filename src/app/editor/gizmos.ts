// Translate (axis + plane handles) and rotate (ring) gizmos for the selected light, and the light wireframes
// ("light gizmos (spot cones, area frames)", plan §6 M3a). No scale gizmo: lights are rigid (plan §1.4).
// Pure math + line-list builders; the editor uploads the lines to the raster overlay (overlay.ts): light wireframes
// depth-tested against the V-buffer depth, gizmo handles always on top. Unit-tested in tests/editor/gizmo.test.ts.
//
// Handles use WORLD axes (Blender's default "Global" orientation). Sizes are screen-constant (GIZMO_PX).
// Drags:
//   axis    closest point between the mouse ray and the axis line through the gizmo centre
//   plane   mouse ray ∩ the plane through the centre spanned by the two axes
//   ring    signed angle between the start/current ray ∩ ring plane (fallback: screen-space angle when edge-on)
import type { LightData } from '../../core/scene/types.ts';
import {
  add, colOf, cross, dot, emitDirOf, len, normalize, pointSegmentDist, positionOf, projectPoint, rayPlane, scale,
  screenRay, spotConeGeometry, sub, sunScreenPosition, worldPerPixel, type Ray, type Vec3, type ViewInfo,
} from './picking.ts';

export type GizmoMode = 'translate' | 'rotate';
export type AxisHandle = 'x' | 'y' | 'z';
export type PlaneHandle = 'xy' | 'yz' | 'xz';
export type RingHandle = 'rx' | 'ry' | 'rz';
export type HandleId = AxisHandle | PlaneHandle | RingHandle;

export const GIZMO_PX = 90;
export const HANDLE_TOL_PX = 8;
export const RING_SEGMENTS = 64;
/** Ring segments closer than this fraction of the ring's screen radius to the centre are not grabbable. */
export const RING_INNER_FRAC = 0.45;
export const AXES: Record<AxisHandle, Vec3> = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
const AXIS_COLOR: Record<AxisHandle, [number, number, number, number]> = { x: [1, 0.25, 0.25, 1], y: [0.35, 1, 0.35, 1], z: [0.35, 0.55, 1, 1] };
const PLANE_AXES: Record<PlaneHandle, [AxisHandle, AxisHandle]> = { xy: ['x', 'y'], yz: ['y', 'z'], xz: ['x', 'z'] };
const PLANE_NORMAL: Record<PlaneHandle, AxisHandle> = { xy: 'z', yz: 'x', xz: 'y' };
const RING_AXIS: Record<RingHandle, AxisHandle> = { rx: 'x', ry: 'y', rz: 'z' };
const HILITE: [number, number, number, number] = [1, 0.85, 0.2, 1];

export interface GizmoLayout { center: Vec3; size: number }

export function gizmoLayout(center: Vec3, v: ViewInfo, px = GIZMO_PX): GizmoLayout {
  return { center, size: px * worldPerPixel(v, center) };
}

/** Plane-handle square: [s0, s1] along both axes, as fractions of the axis length. */
const PLANE_RANGE: [number, number] = [0.25, 0.45];

function ringPoints(c: Vec3, axis: Vec3, r: number, n = RING_SEGMENTS): Vec3[] {
  const a = normalize(axis);
  const helper: Vec3 = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const u = normalize(cross(a, helper));
  const w = cross(a, u);
  const pts: Vec3[] = [];
  for (let i = 0; i <= n; i++) {
    const th = (2 * Math.PI * i) / n;
    pts.push(add(c, add(scale(u, r * Math.cos(th)), scale(w, r * Math.sin(th)))));
  }
  return pts;
}

/** Which handle is under CSS pixel (x, y), if any. Axis tips and planes win over rings' overlap by distance. */
export function pickHandle(g: GizmoLayout, mode: GizmoMode, v: ViewInfo, x: number, y: number, tolPx = HANDLE_TOL_PX): HandleId | undefined {
  const c2 = projectPoint(v, g.center);
  if (!c2) return undefined;
  let best: { h: HandleId; d: number } | undefined;
  const consider = (h: HandleId, d: number) => { if (d <= tolPx && (!best || d < best.d)) best = { h, d }; };
  if (mode === 'translate') {
    // planes first (inside the square = distance 0)
    for (const ph of Object.keys(PLANE_AXES) as PlaneHandle[]) {
      const [a, b] = PLANE_AXES[ph];
      const corners = [[PLANE_RANGE[0], PLANE_RANGE[0]], [PLANE_RANGE[1], PLANE_RANGE[0]], [PLANE_RANGE[1], PLANE_RANGE[1]], [PLANE_RANGE[0], PLANE_RANGE[1]]]
        .map(([s, t]) => projectPoint(v, add(g.center, add(scale(AXES[a], s * g.size), scale(AXES[b], t * g.size)))));
      if (corners.some((q) => !q)) continue;
      const qd = corners as { x: number; y: number }[];
      if (quadArea(qd) < 16) continue; // edge-on plane: not grabbable (the axis handles cover it)
      if (pointInQuad(x, y, qd)) consider(ph, 0);
    }
    for (const ah of Object.keys(AXES) as AxisHandle[]) {
      const tip = projectPoint(v, add(g.center, scale(AXES[ah], g.size)));
      if (!tip) continue;
      // skip the inner part so the centre stays free for the plane handles / picking through
      const from = { x: c2.x + (tip.x - c2.x) * 0.2, y: c2.y + (tip.y - c2.y) * 0.2 };
      consider(ah, pointSegmentDist(x, y, from.x, from.y, tip.x, tip.y));
    }
  } else {
    const innerPx = RING_INNER_FRAC * (g.size / worldPerPixel(v, g.center));
    for (const rh of Object.keys(RING_AXIS) as RingHandle[]) {
      const pts = ringPoints(g.center, AXES[RING_AXIS[rh]], g.size).map((p) => projectPoint(v, p));
      let d = Infinity;
      for (let i = 0; i + 1 < pts.length; i++) {
        const a = pts[i], b = pts[i + 1];
        if (!a || !b) continue;
        // edge-on rings project to a line through the centre: keep the inner area free (picking through / other rings)
        if (Math.hypot(a.x - c2.x, a.y - c2.y) < innerPx || Math.hypot(b.x - c2.x, b.y - c2.y) < innerPx) continue;
        d = Math.min(d, pointSegmentDist(x, y, a.x, a.y, b.x, b.y));
      }
      consider(rh, d);
    }
  }
  return best?.h;
}

function quadArea(q: { x: number; y: number }[]): number {
  let a = 0;
  for (let i = 0; i < q.length; i++) { const p = q[i], n = q[(i + 1) % q.length]; a += p.x * n.y - n.x * p.y; }
  return Math.abs(a) / 2;
}

function pointInQuad(x: number, y: number, q: { x: number; y: number }[]): boolean {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const a = q[i], b = q[(i + 1) % 4];
    const c = (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
    const s = Math.sign(c);
    if (s === 0) continue;
    if (sign === 0) sign = s; else if (s !== sign) return false;
  }
  return true;
}

// ---- drags --------------------------------------------------------------------------------------------------------

export interface DragState {
  handle: HandleId;
  center: Vec3;
  startMatrix: Float64Array;
  /** axis: start parameter along the axis; plane: start hit; ring: start in-plane vector (or screen angle). */
  s0?: number;
  p0?: Vec3;
  v0?: Vec3;
  screenAngle0?: number;
  axis?: Vec3;
}

export const isRing = (h: HandleId): h is RingHandle => h === 'rx' || h === 'ry' || h === 'rz';
export const isPlane = (h: HandleId): h is PlaneHandle => h === 'xy' || h === 'yz' || h === 'xz';

/** Parameter s of the point on line (c + s·a, unit a) closest to the ray; undefined when (nearly) parallel. */
export function closestOnAxis(c: Vec3, a: Vec3, r: Ray): number | undefined {
  const b = dot(a, r.d);
  const den = 1 - b * b;
  if (den < 1e-6) return undefined;
  const w0 = sub(c, r.o);
  const dd = dot(a, w0), ee = dot(r.d, w0);
  return (b * ee - dd) / den;
}

const EDGE_ON = 0.12;

export function beginDrag(handle: HandleId, center: Vec3, matrix: ArrayLike<number>, r: Ray, v?: ViewInfo, sx?: number, sy?: number): DragState | undefined {
  const st: DragState = { handle, center: [...center], startMatrix: Float64Array.from(matrix) };
  if (isRing(handle)) {
    const axis = AXES[RING_AXIS[handle]];
    st.axis = axis;
    const t = Math.abs(dot(axis, r.d)) > EDGE_ON ? rayPlane(r, center, axis) : undefined;
    if (t !== undefined) {
      const h = sub(add(r.o, scale(r.d, t)), center);
      st.v0 = sub(h, scale(axis, dot(h, axis)));
      if (len(st.v0) < 1e-12) return undefined;
    } else if (v && sx !== undefined && sy !== undefined) {
      const c = projectPoint(v, center);
      if (!c) return undefined;
      st.screenAngle0 = Math.atan2(sy - c.y, sx - c.x);
    } else return undefined;
    return st;
  }
  if (isPlane(handle)) {
    const n = AXES[PLANE_NORMAL[handle]];
    const t = rayPlane(r, center, n);
    if (t === undefined) return undefined;
    st.p0 = add(r.o, scale(r.d, t));
    return st;
  }
  const a = AXES[handle];
  const s = closestOnAxis(center, a, r);
  if (s === undefined) return undefined;
  st.axis = a;
  st.s0 = s;
  return st;
}

/** Rotation about a unit axis as a 3x3 (column-major) applied to the start matrix's rotation columns. */
function rotateColumns(m: Float64Array, axis: Vec3, angle: number): Float64Array {
  const out = Float64Array.from(m);
  const c = Math.cos(angle), s = Math.sin(angle), k = 1 - c;
  const [x, y, z] = axis;
  const R = [
    c + x * x * k, y * x * k + z * s, z * x * k - y * s,
    x * y * k - z * s, c + y * y * k, z * y * k + x * s,
    x * z * k + y * s, y * z * k - x * s, c + z * z * k,
  ];
  for (let j = 0; j < 3; j++) {
    const col = [m[4 * j], m[4 * j + 1], m[4 * j + 2]];
    for (let i = 0; i < 3; i++) out[4 * j + i] = R[i] * col[0] + R[3 + i] * col[1] + R[6 + i] * col[2];
  }
  return orthonormalize(out);
}

/** Gram–Schmidt on the rotation columns keeping −Z (the emission axis) exact; keeps drags rigid under f64 drift. */
export function orthonormalize(m: Float64Array): Float64Array {
  const z = normalize(colOf(m, 2));
  let y = colOf(m, 1);
  y = normalize(sub(y, scale(z, dot(y, z))));
  const x = cross(y, z);
  const out = Float64Array.from(m);
  out.set([x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, z[0], z[1], z[2], 0], 0);
  out[15] = 1;
  return out;
}

export interface DragOptions { snap?: number; angleSnap?: number }

/** New light matrix for the current mouse ray (rotation about the gizmo centre keeps the position). */
export function dragMatrix(st: DragState, r: Ray, o: DragOptions = {}, v?: ViewInfo, sx?: number, sy?: number): Float64Array {
  const m = Float64Array.from(st.startMatrix);
  const snapT = (d: number) => (o.snap ? Math.round(d / o.snap) * o.snap : d);
  if (isRing(st.handle)) {
    let angle: number | undefined;
    if (st.v0) {
      const t = rayPlane(r, st.center, st.axis!);
      if (t === undefined) return m;
      const h = sub(add(r.o, scale(r.d, t)), st.center);
      const v1 = sub(h, scale(st.axis!, dot(h, st.axis!)));
      if (len(v1) < 1e-12) return m;
      angle = Math.atan2(dot(st.axis!, cross(st.v0, v1)), dot(st.v0, v1));
    } else if (st.screenAngle0 !== undefined && v && sx !== undefined && sy !== undefined) {
      const c = projectPoint(v, st.center);
      if (!c) return m;
      const a1 = Math.atan2(sy - c.y, sx - c.x);
      // screen y points down: a clockwise-looking screen motion is +atan2; the axis sign toward the viewer decides
      const toCam = normalize(sub([v.camToWorld[12], v.camToWorld[13], v.camToWorld[14]], st.center));
      angle = -(a1 - st.screenAngle0) * Math.sign(dot(st.axis!, toCam) || 1);
    }
    if (angle === undefined) return m;
    if (o.angleSnap) angle = Math.round(angle / o.angleSnap) * o.angleSnap;
    return rotateColumns(m, st.axis!, angle);
  }
  let delta: Vec3 = [0, 0, 0];
  if (isPlane(st.handle)) {
    const n = AXES[PLANE_NORMAL[st.handle]];
    const t = rayPlane(r, st.center, n);
    if (t === undefined) return m;
    const d = sub(add(r.o, scale(r.d, t)), st.p0!);
    delta = [snapT(d[0]), snapT(d[1]), snapT(d[2])];
    delta = sub(delta, scale(n, dot(delta, n)));
  } else {
    const s = closestOnAxis(st.center, st.axis!, r);
    if (s === undefined) return m;
    delta = scale(st.axis!, snapT(s - st.s0!));
  }
  m[12] += delta[0]; m[13] += delta[1]; m[14] += delta[2];
  return m;
}

/** Convenience for tests / automation: drag from screen (x0, y0) to (x1, y1). */
export function dragScreen(handle: HandleId, center: Vec3, matrix: ArrayLike<number>, v: ViewInfo, x0: number, y0: number, x1: number, y1: number, o: DragOptions = {}): Float64Array | undefined {
  const st = beginDrag(handle, center, matrix, screenRay(v, x0, y0), v, x0, y0);
  if (!st) return undefined;
  return dragMatrix(st, screenRay(v, x1, y1), o, v, x1, y1);
}

// ---- line lists ---------------------------------------------------------------------------------------------------

export interface Lines { points: number[]; colors: number[] }
const push = (L: Lines, a: readonly number[], b: readonly number[], c: readonly number[]) => { L.points.push(a[0], a[1], a[2], b[0], b[1], b[2]); L.colors.push(...c, ...c); };
const polyline = (L: Lines, pts: readonly Vec3[], c: readonly number[]) => { for (let i = 0; i + 1 < pts.length; i++) push(L, pts[i], pts[i + 1], c); };

/** Gizmo handles (always on top). */
export function gizmoLines(g: GizmoLayout, mode: GizmoMode, hot?: HandleId, active?: HandleId): Lines {
  const L: Lines = { points: [], colors: [] };
  const colFor = (h: HandleId, base: readonly number[]) => (h === active || h === hot ? HILITE : base);
  if (mode === 'translate') {
    for (const ah of Object.keys(AXES) as AxisHandle[]) {
      const a = AXES[ah];
      const tip = add(g.center, scale(a, g.size));
      const c = colFor(ah, AXIS_COLOR[ah]);
      push(L, add(g.center, scale(a, 0.2 * g.size)), tip, c);
      // arrow head: small cross-lines
      const helper: Vec3 = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
      const u = normalize(cross(a, helper)), w = cross(a, u);
      const back = add(g.center, scale(a, 0.85 * g.size));
      for (const q of [u, w, scale(u, -1), scale(w, -1)]) push(L, tip, add(back, scale(q, 0.06 * g.size)), c);
    }
    for (const ph of Object.keys(PLANE_AXES) as PlaneHandle[]) {
      const [a, b] = PLANE_AXES[ph];
      const n = PLANE_NORMAL[ph];
      const c = colFor(ph, AXIS_COLOR[n].map((x, i) => (i === 3 ? 0.9 : x * 0.8)));
      const q = [[PLANE_RANGE[0], PLANE_RANGE[0]], [PLANE_RANGE[1], PLANE_RANGE[0]], [PLANE_RANGE[1], PLANE_RANGE[1]], [PLANE_RANGE[0], PLANE_RANGE[1]], [PLANE_RANGE[0], PLANE_RANGE[0]]]
        .map(([s, t]) => add(g.center, add(scale(AXES[a], s * g.size), scale(AXES[b], t * g.size))));
      polyline(L, q, c);
      push(L, q[0], q[2], c); // diagonal to make the square read as a handle
    }
  } else {
    for (const rh of Object.keys(RING_AXIS) as RingHandle[]) {
      polyline(L, ringPoints(g.center, AXES[RING_AXIS[rh]], g.size), colFor(rh, AXIS_COLOR[RING_AXIS[rh]]));
    }
  }
  return L;
}

/** World anchor of the i-th sun's screen icon (on the view ray through sunScreenPosition, at depth 1). */
export function sunIconAnchor(i: number, v: ViewInfo): Vec3 {
  const [sx, sy] = sunScreenPosition(i, v);
  const ray = screenRay(v, sx, sy);
  return add(ray.o, ray.d);
}
export const SUN_GIZMO_PX = 38;

const LIGHT_COL: [number, number, number, number] = [1, 0.95, 0.6, 1];
const LIGHT_SEL: [number, number, number, number] = [1, 0.55, 0.1, 1];

/** Wireframe of one light (depth-tested): point star, spot cone, rect/disk frame + normal, sun arrow (screen-fixed). */
export function lightWireframe(l: LightData, v: ViewInfo, selected: boolean, sunIndex = 0): Lines {
  const L: Lines = { points: [], colors: [] };
  const c = selected ? LIGHT_SEL : LIGHT_COL;
  const p = positionOf(l.matrix);
  const wpp = worldPerPixel(v, p);
  const X = normalize(colOf(l.matrix, 0)), Y = normalize(colOf(l.matrix, 1)), D = emitDirOf(l.matrix);
  const star = (r: number) => {
    for (const a of [X, Y, D]) push(L, add(p, scale(a, -r)), add(p, scale(a, r)), c);
    polyline(L, ringPoints(p, D, r * 0.8, 24), c);
  };
  switch (l.type) {
    case 'point': star(10 * wpp); break;
    case 'spot': {
      star(6 * wpp);
      const g = spotConeGeometry(l, v);
      const rim = ringPoints(g.end, D, g.radius, 32);
      polyline(L, rim, c);
      for (let i = 0; i < 4; i++) push(L, p, rim[i * 8], c);
      // blend: inner cone
      const blend = l.spotBlend ?? 0;
      if (blend > 0) {
        const half = Math.min((l.spotSize ?? Math.PI / 4) / 2, 80 * Math.PI / 180);
        const cosH = Math.cos(half);
        const cosI = Math.min(1, cosH + (1 - cosH) * blend);
        const rin = g.len * Math.tan(Math.acos(cosI));
        polyline(L, ringPoints(g.end, D, rin, 32), c.map((x, i) => (i === 3 ? 0.5 : x)));
      }
      break;
    }
    case 'rect': {
      const hx = (l.sizeX ?? 1) / 2, hy = (l.sizeY ?? l.sizeX ?? 1) / 2;
      const q = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]].map(([a, b]) => add(p, add(scale(X, a * hx), scale(Y, b * hy))));
      polyline(L, q, c);
      push(L, p, add(p, scale(D, Math.max(0.5 * Math.min(hx, hy), 30 * wpp))), c);
      break;
    }
    case 'disk': {
      const r = (l.sizeX ?? 1) / 2;
      polyline(L, ringPoints(p, D, r, 48), c);
      push(L, p, add(p, scale(D, Math.max(0.5 * r, 30 * wpp))), c);
      break;
    }
    case 'sun': {
      // A screen-fixed icon: a small circle with an arrow along the sun's emission direction, placed on the view ray
      // through sunScreenPosition at a fixed depth.
      const [sx, sy] = sunScreenPosition(sunIndex, v);
      const ray = screenRay(v, sx, sy);
      const at = sunIconAnchor(sunIndex, v);
      const w = worldPerPixel(v, at);
      polyline(L, ringPoints(at, ray.d, 10 * w, 24), c);
      push(L, at, add(at, scale(D, 30 * w)), c);
      for (let i = 0; i < 8; i++) {
        const th = (i / 8) * 2 * Math.PI;
        const u = normalize(cross(ray.d, [0, 1, 0]));
        const ww = cross(ray.d, u);
        const dir = add(scale(u, Math.cos(th)), scale(ww, Math.sin(th)));
        push(L, add(at, scale(dir, 13 * w)), add(at, scale(dir, 17 * w)), c);
      }
      break;
    }
  }
  return L;
}
