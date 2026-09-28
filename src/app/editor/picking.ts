// CPU proxy picking for analytic lights (plan §5 M3a: analytic lights are NOT in the BVH, so the V-buffer cannot
// see them). Pure f64 math, unit-tested in tests/editor/picking.test.ts.
//
// Screen convention: CSS pixels relative to the canvas, origin top-left, y down. The camera ray through (x, y)
// matches plan §1.2's raster mapping with (c+u)/W = x/W and (H−1−r+v)/H = 1 − y/H:
//   d_cam = ((2x/W − 1)·tan(vfov/2)·W/H, (1 − 2y/H)·tan(vfov/2), −1)
// Proxies (sizes in screen pixels are converted with worldPerPixel so they stay a constant size on screen):
//   point  sphere (radius PICK_PX)                    spot  apex sphere + cone-end disk (cone drawn SPOT_LEN_PX long)
//   rect   the light's quad (+ a centre sphere)       disk  the light's disk (+ a centre sphere)
//   sun    a fixed screen-space icon (sunScreenPosition), tested in 2D before any 3D proxy
import type { LightData } from '../../core/scene/types.ts';

export type Vec3 = [number, number, number];
export interface Ray { o: Vec3; d: Vec3 }

/** Camera + canvas description for picking (CSS pixels). */
export interface ViewInfo { camToWorld: ArrayLike<number>; yfov: number; width: number; height: number }

export const PICK_PX = 10;
export const SPOT_LEN_PX = 70;
export const SUN_ICON_PX = 18;

export const dot = (a: readonly number[], b: readonly number[]): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const sub = (a: readonly number[], b: readonly number[]): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const add = (a: readonly number[], b: readonly number[]): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const scale = (a: readonly number[], s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const cross = (a: readonly number[], b: readonly number[]): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const len = (a: readonly number[]): number => Math.hypot(a[0], a[1], a[2]);
export const normalize = (a: readonly number[]): Vec3 => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

export const colOf = (m: ArrayLike<number>, j: number): Vec3 => [m[4 * j], m[4 * j + 1], m[4 * j + 2]];
export const positionOf = (m: ArrayLike<number>): Vec3 => [m[12], m[13], m[14]];
/** Emission axis a_L = −Z_obj (math.md#units-lights). */
export const emitDirOf = (m: ArrayLike<number>): Vec3 => normalize(scale(colOf(m, 2), -1));

/** World ray through CSS pixel (x, y). */
export function screenRay(v: ViewInfo, x: number, y: number): Ray {
  const t = Math.tan(v.yfov / 2);
  const aspect = v.width / v.height;
  const dc: Vec3 = [(2 * x / v.width - 1) * t * aspect, (1 - 2 * y / v.height) * t, -1];
  const m = v.camToWorld;
  const d = normalize([
    m[0] * dc[0] + m[4] * dc[1] + m[8] * dc[2],
    m[1] * dc[0] + m[5] * dc[1] + m[9] * dc[2],
    m[2] * dc[0] + m[6] * dc[1] + m[10] * dc[2],
  ]);
  return { o: [m[12], m[13], m[14]], d };
}

/** CSS pixel of a world point and its view depth (−z_cam > 0 in front); undefined behind the camera. */
export function projectPoint(v: ViewInfo, p: readonly number[]): { x: number; y: number; depth: number } | undefined {
  const m = v.camToWorld;
  const r = sub(p, [m[12], m[13], m[14]]);
  const xc = dot(r, colOf(m, 0)), yc = dot(r, colOf(m, 1)), zc = dot(r, colOf(m, 2));
  if (zc > -1e-9) return undefined;
  const t = Math.tan(v.yfov / 2);
  const aspect = v.width / v.height;
  const nx = xc / (-zc * t * aspect), ny = yc / (-zc * t);
  return { x: (nx + 1) * 0.5 * v.width, y: (1 - ny) * 0.5 * v.height, depth: -zc };
}

/** World length of one CSS pixel at point p (perspective, measured at p's view depth). */
export function worldPerPixel(v: ViewInfo, p: readonly number[]): number {
  const m = v.camToWorld;
  const depth = Math.max(1e-6, -dot(sub(p, [m[12], m[13], m[14]]), colOf(m, 2)));
  return (2 * depth * Math.tan(v.yfov / 2)) / v.height;
}

// ---- primitives -------------------------------------------------------------------------------------------------

/** Nearest t > 0 of a ray vs sphere, or undefined. */
export function raySphere(r: Ray, c: readonly number[], radius: number): number | undefined {
  const oc = sub(r.o, c);
  const b = dot(oc, r.d);
  const cc = dot(oc, oc) - radius * radius;
  const disc = b * b - cc;
  if (disc < 0) return undefined;
  const s = Math.sqrt(disc);
  const t0 = -b - s, t1 = -b + s;
  if (t0 > 0) return t0;
  if (t1 > 0) return t1;
  return undefined;
}

/** Ray vs plane (point p, normal n): t or undefined when parallel/behind. */
export function rayPlane(r: Ray, p: readonly number[], n: readonly number[]): number | undefined {
  const den = dot(n, r.d);
  if (Math.abs(den) < 1e-12) return undefined;
  const t = dot(sub(p, r.o), n) / den;
  return t > 0 ? t : undefined;
}

/** Two-sided disk (centre c, normal n, radius). */
export function rayDisk(r: Ray, c: readonly number[], n: readonly number[], radius: number): number | undefined {
  const t = rayPlane(r, c, n);
  if (t === undefined) return undefined;
  const h = sub(add(r.o, scale(r.d, t)), c);
  return dot(h, h) <= radius * radius ? t : undefined;
}

/** Two-sided rectangle: centre c, unit axes u, v, half extents hu, hv. */
export function rayQuad(r: Ray, c: readonly number[], u: readonly number[], v: readonly number[], hu: number, hv: number): number | undefined {
  const n = cross(u, v);
  const t = rayPlane(r, c, n);
  if (t === undefined) return undefined;
  const h = sub(add(r.o, scale(r.d, t)), c);
  return Math.abs(dot(h, u)) <= hu && Math.abs(dot(h, v)) <= hv ? t : undefined;
}

/** Distance (px) from point to a 2D segment. */
export function pointSegmentDist(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const vx = bx - ax, vy = by - ay;
  const l2 = vx * vx + vy * vy;
  const u = l2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * vx + (py - ay) * vy) / l2)) : 0;
  return Math.hypot(px - (ax + u * vx), py - (ay + u * vy));
}

// ---- light proxies ----------------------------------------------------------------------------------------------

export type LightProxy =
  | { id: number; kind: 'sphere'; c: Vec3; r: number }
  | { id: number; kind: 'disk'; c: Vec3; n: Vec3; r: number }
  | { id: number; kind: 'quad'; c: Vec3; u: Vec3; v: Vec3; hu: number; hv: number };

/** Length of the drawn spot cone (world units) and its end-disk radius. */
export function spotConeGeometry(l: Pick<LightData, 'matrix' | 'spotSize'>, v: ViewInfo): { len: number; radius: number; end: Vec3 } {
  const p = positionOf(l.matrix);
  const L = SPOT_LEN_PX * worldPerPixel(v, p);
  const half = Math.min((l.spotSize ?? Math.PI / 4) / 2, 80 * Math.PI / 180);
  return { len: L, radius: L * Math.tan(half), end: add(p, scale(emitDirOf(l.matrix), L)) };
}

export function lightProxies(l: LightData, v: ViewInfo): LightProxy[] {
  const p = positionOf(l.matrix);
  const r = PICK_PX * worldPerPixel(v, p);
  switch (l.type) {
    case 'point': return [{ id: l.id, kind: 'sphere', c: p, r }];
    case 'spot': {
      const g = spotConeGeometry(l, v);
      return [{ id: l.id, kind: 'sphere', c: p, r }, { id: l.id, kind: 'disk', c: g.end, n: emitDirOf(l.matrix), r: Math.max(g.radius, r) }];
    }
    case 'rect': return [
      { id: l.id, kind: 'quad', c: p, u: normalize(colOf(l.matrix, 0)), v: normalize(colOf(l.matrix, 1)), hu: (l.sizeX ?? 1) / 2, hv: (l.sizeY ?? l.sizeX ?? 1) / 2 },
      { id: l.id, kind: 'sphere', c: p, r },
    ];
    case 'disk': return [
      { id: l.id, kind: 'disk', c: p, n: emitDirOf(l.matrix), r: (l.sizeX ?? 1) / 2 },
      { id: l.id, kind: 'sphere', c: p, r },
    ];
    case 'sun': return [];
  }
}

export function intersectProxy(r: Ray, p: LightProxy): number | undefined {
  if (p.kind === 'sphere') return raySphere(r, p.c, p.r);
  if (p.kind === 'disk') return rayDisk(r, p.c, p.n, p.r);
  return rayQuad(r, p.c, p.u, p.v, p.hu, p.hv);
}

/** Fixed screen position (CSS px) of the i-th sun gizmo: a row along the bottom-left of the viewport. */
export function sunScreenPosition(i: number, v: Pick<ViewInfo, 'width' | 'height'>): [number, number] {
  return [48 + i * 56, v.height - 56];
}

export interface LightPick { id: number; t: number; screen?: boolean }

/** Pick the nearest light under CSS pixel (x, y). Suns (screen icons) win over 3D proxies. */
export function pickLight(lights: readonly LightData[], v: ViewInfo, x: number, y: number): LightPick | undefined {
  const suns = lights.filter((l) => l.type === 'sun');
  for (let i = 0; i < suns.length; i++) {
    const [sx, sy] = sunScreenPosition(i, v);
    if (Math.hypot(x - sx, y - sy) <= SUN_ICON_PX) return { id: suns[i].id, t: 0, screen: true };
  }
  const ray = screenRay(v, x, y);
  let best: LightPick | undefined;
  for (const l of lights) {
    for (const p of lightProxies(l, v)) {
      const t = intersectProxy(ray, p);
      if (t !== undefined && (!best || t < best.t)) best = { id: l.id, t };
    }
  }
  return best;
}
