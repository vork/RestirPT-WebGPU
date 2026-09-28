// Picking and gizmo math (src/app/editor/{picking,gizmos,placement}.ts): camera rays vs the plan §1.2 raster
// mapping, projection round trips, light proxies (point/spot/rect/disk/sun), handle picking, axis/plane/ring drags
// (exact, rigid), and surface placement from a V-buffer texel.
import { describe, expect, it } from 'vitest';
import { poseToMatrix, quatLookDir } from '../../src/core/scene/animation.ts';
import { isRigid } from '../../src/core/scene/scene-package.ts';
import type { LightData, SceneGeometry } from '../../src/core/scene/types.ts';
import {
  beginDrag, closestOnAxis, dragMatrix, dragScreen, gizmoLayout, gizmoLines, lightWireframe, pickHandle,
} from '../../src/app/editor/gizmos.ts';
import {
  emitDirOf, lightProxies, pickLight, projectPoint, raySphere, screenRay, sunScreenPosition, worldPerPixel, type ViewInfo,
} from '../../src/app/editor/picking.ts';
import { BVH_MISS, frameFromEmission, placementMatrix, surfaceFromTexel } from '../../src/app/editor/placement.ts';

const close = (a: ArrayLike<number>, b: ArrayLike<number>, eps = 1e-9) => { for (let i = 0; i < b.length; i++) expect(Math.abs(a[i] - b[i]), `index ${i}: ${a[i]} vs ${b[i]}`).toBeLessThan(eps); };

/** Camera at (0, 1, 5) looking down −Z, 60° vfov, 800×600 CSS px. */
const view = (): ViewInfo => ({ camToWorld: poseToMatrix([0, 1, 5], [0, 0, 0, 1]), yfov: Math.PI / 3, width: 800, height: 600 });
const T = (x: number, y: number, z: number) => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
const light = (o: Partial<LightData>): LightData => ({ id: 1, name: 'l', type: 'point', color: [1, 1, 1], power: 1, exposure: 0, matrix: T(0, 1, 0), visibleToCamera: false, ...o });

describe('screen rays and projection', () => {
  it('matches the plan §1.2 raster mapping and round-trips through projectPoint', () => {
    const v = view();
    const c = screenRay(v, 400, 300);
    close(c.o, [0, 1, 5]);
    close(c.d, [0, 0, -1]);
    // top-right corner: d_cam = (tan(30°)·4/3, tan(30°), −1)
    const t = Math.tan(Math.PI / 6);
    const r = screenRay(v, 800, 0);
    const n = Math.hypot(t * 4 / 3, t, 1);
    close(r.d, [t * 4 / 3 / n, t / n, -1 / n], 1e-12);
    for (const [x, y] of [[10, 20], [400, 300], [799, 599], [123.5, 456.25]]) {
      const ray = screenRay(v, x, y);
      const p = [ray.o[0] + 3.7 * ray.d[0], ray.o[1] + 3.7 * ray.d[1], ray.o[2] + 3.7 * ray.d[2]];
      const q = projectPoint(v, p)!;
      expect(Math.abs(q.x - x)).toBeLessThan(1e-9);
      expect(Math.abs(q.y - y)).toBeLessThan(1e-9);
    }
    expect(projectPoint(v, [0, 1, 6])).toBeUndefined(); // behind the camera
    expect(worldPerPixel(v, [0, 1, 0])).toBeCloseTo((2 * 5 * t) / 600, 12);
  });

  it('ray/sphere returns the nearest positive hit', () => {
    expect(raySphere({ o: [0, 0, 0], d: [0, 0, -1] }, [0, 0, -5], 1)).toBeCloseTo(4, 12);
    expect(raySphere({ o: [0, 0, -5], d: [0, 0, -1] }, [0, 0, -5], 1)).toBeCloseTo(1, 12); // inside
    expect(raySphere({ o: [0, 0, 0], d: [0, 0, 1] }, [0, 0, -5], 1)).toBeUndefined();
  });
});

describe('light proxies and picking', () => {
  it('picks point, spot (apex and cone end), rect, disk and screen-space suns; nearest wins', () => {
    const v = view();
    const pt = light({ id: 10, type: 'point', matrix: T(0, 1, 0) });
    // centre pixel hits the point light's screen-constant sphere
    expect(pickLight([pt], v, 400, 300)?.id).toBe(10);
    // PICK_PX = 10 → a click 8 px away still hits, 14 px misses
    expect(pickLight([pt], v, 408, 300)?.id).toBe(10);
    expect(pickLight([pt], v, 414, 300)).toBeUndefined();

    // rect facing the camera at z = −1: hit anywhere inside its 2 m × 1 m quad
    const rect = light({ id: 11, type: 'rect', sizeX: 2, sizeY: 1, spread: Math.PI, matrix: T(0, 1, -1) });
    const inside = projectPoint(v, [0.9, 1.45, -1])!;
    const outside = projectPoint(v, [1.1, 1.0, -1])!;
    expect(pickLight([rect], v, inside.x, inside.y)?.id).toBe(11);
    expect(pickLight([rect], v, outside.x, outside.y)).toBeUndefined();

    // disk: circle of diameter 1 at z = −2 (emits toward the camera: local −Z = +Z world → rotate 180° about Y)
    const disk = light({ id: 12, type: 'disk', sizeX: 1, spread: Math.PI, matrix: new Float32Array([-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 2, 1, -2, 1]) });
    const dIn = projectPoint(v, [2.45, 1, -2])!, dOut = projectPoint(v, [2.4, 1.4, -2])!; // 0.45 in, 0.566 out
    expect(pickLight([disk], v, dIn.x, dIn.y)?.id).toBe(12);
    expect(pickLight([disk], v, dOut.x, dOut.y)).toBeUndefined();

    // spot pointing down from (−2, 2, 0): the cone-end disk (SPOT_LEN_PX long) is pickable below the apex
    const down = frameFromEmission([0, -1, 0], [-2, 2, 0]);
    const spot = light({ id: 13, type: 'spot', spotSize: Math.PI / 2, spotBlend: 0.1, matrix: down });
    const pr = lightProxies(spot, v);
    expect(pr.map((p) => p.kind)).toEqual(['sphere', 'disk']);
    const end = (pr[1] as { c: number[] }).c;
    const e2 = projectPoint(v, end)!;
    expect(pickLight([spot], v, e2.x, e2.y)?.id).toBe(13);
    const apex = projectPoint(v, [-2, 2, 0])!;
    expect(pickLight([spot], v, apex.x, apex.y)?.id).toBe(13);

    // nearest: a point light in front of the rect wins at the same pixel
    const front = light({ id: 14, type: 'point', matrix: T(0.5, 1.2, 0) });
    const pf = projectPoint(v, [0.5, 1.2, 0])!;
    expect(pickLight([rect, front], v, pf.x, pf.y)?.id).toBe(14);

    // suns: fixed screen icons, tested before 3D proxies
    const sun1 = light({ id: 20, type: 'sun', matrix: frameFromEmission([0.3, -1, 0.2], [0, 0, 0]) });
    const sun2 = light({ id: 21, type: 'sun', matrix: frameFromEmission([0, -1, 0], [0, 0, 0]) });
    const [sx, sy] = sunScreenPosition(1, v);
    expect(pickLight([sun1, rect, sun2], v, sx + 5, sy - 5)).toMatchObject({ id: 21, screen: true });
    expect(lightProxies(sun1, v)).toEqual([]);
  });

  it('wireframes are line lists (pairs of points, rgba per vertex)', () => {
    const v = view();
    for (const type of ['point', 'spot', 'rect', 'disk', 'sun'] as const) {
      const w = lightWireframe(light({ type, spotSize: 1, spotBlend: 0.2, sizeX: 1, sizeY: 0.5, spread: Math.PI }), v, type === 'spot');
      expect(w.points.length % 6).toBe(0);
      expect(w.colors.length).toBe((w.points.length / 3) * 4);
      expect(w.points.length).toBeGreaterThan(0);
    }
  });
});

describe('gizmo handles and drags', () => {
  it('picks axis tips, plane squares and rings under the cursor', () => {
    const v = view();
    const c: [number, number, number] = [0, 1, 0];
    const g = gizmoLayout(c, v);
    expect(g.size).toBeCloseTo(90 * worldPerPixel(v, c), 12);
    const tipX = projectPoint(v, [g.size, 1, 0])!;
    expect(pickHandle(g, 'translate', v, tipX.x - 2, tipX.y + 1)).toBe('x');
    const tipY = projectPoint(v, [0, 1 + g.size * 0.9, 0])!;
    expect(pickHandle(g, 'translate', v, tipY.x, tipY.y)).toBe('y');
    const xy = projectPoint(v, [0.35 * g.size, 1 + 0.35 * g.size, 0])!;
    expect(pickHandle(g, 'translate', v, xy.x, xy.y)).toBe('xy');
    expect(pickHandle(g, 'translate', v, 400 - 60, 300 + 60)).toBeUndefined();
    // rotate: the Z ring faces the camera (a circle of radius 90 px on screen)
    expect(pickHandle(g, 'rotate', v, 400 + 90 * Math.cos(1), 300 - 90 * Math.sin(1))).toBe('rz');
    expect(pickHandle(g, 'rotate', v, 400, 300)).toBeUndefined();
    const lines = gizmoLines(g, 'translate', 'x');
    expect(lines.points.length % 6).toBe(0);
  });

  it('axis drag moves exactly along the world axis under the cursor', () => {
    const v = view();
    const m = T(0, 1, 0);
    const a = projectPoint(v, [0, 1, 0])!, b = projectPoint(v, [0.75, 1, 0])!;
    const out = dragScreen('x', [0, 1, 0], m, v, a.x, a.y, b.x, b.y)!;
    close(out, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.75, 1, 0, 1], 1e-9);
    // off-axis mouse motion only moves along the axis (closest point between ray and axis line)
    const noisy = dragScreen('x', [0, 1, 0], m, v, a.x, a.y, b.x, b.y + 25)!;
    close([noisy[13], noisy[14]], [1, 0], 1e-12);
    expect(noisy[12]).toBeCloseTo(0.75, 2);
    // snapping
    const s = dragScreen('x', [0, 1, 0], m, v, a.x, a.y, b.x, b.y, { snap: 0.5 })!;
    expect(s[12]).toBeCloseTo(1, 12);
    // closestOnAxis on a ray parallel to the axis is undefined
    expect(closestOnAxis([0, 0, 0], [0, 0, 1], { o: [0, 0, 5], d: [0, 0, -1] })).toBeUndefined();
  });

  it('plane drag follows the cursor inside the plane; out-of-plane component is zero', () => {
    const v = view();
    const m = T(0, 1, 0);
    const a = projectPoint(v, [0, 1, 0])!, b = projectPoint(v, [-0.4, 1.3, 0])!;
    const out = dragScreen('xy', [0, 1, 0], m, v, a.x, a.y, b.x, b.y)!;
    close([out[12], out[13], out[14]], [-0.4, 1.3, 0], 1e-9);
  });

  it('ring drag rotates about the world axis by the swept angle; the matrix stays rigid; position fixed', () => {
    const v = view();
    const m = frameFromEmission([0, 0, -1], [0, 1, 0]);
    const c: [number, number, number] = [0, 1, 0];
    const R = 0.5;
    const p0 = projectPoint(v, [R, 1, 0])!;
    const ang = 0.7;
    const p1 = projectPoint(v, [R * Math.cos(ang), 1 + R * Math.sin(ang), 0])!;
    const out = dragScreen('rz', c, m, v, p0.x, p0.y, p1.x, p1.y)!;
    expect(isRigid(out, 1e-9)).toBe(true);
    close([out[12], out[13], out[14]], [0, 1, 0]);
    // X axis rotated by +0.7 rad about +Z
    close([out[0], out[1], out[2]], [Math.cos(ang), Math.sin(ang), 0], 1e-9);
    close(emitDirOf(out), [0, 0, -1], 1e-9);
    // edge-on ring falls back to a screen-space angle but still yields a rigid rotation about the axis
    const st = beginDrag('rx', c, m, screenRay(v, p0.x, p0.y), v, p0.x, p0.y)!;
    const o2 = dragMatrix(st, screenRay(v, p1.x, p1.y), {}, v, p1.x, p1.y);
    expect(isRigid(o2, 1e-9)).toBe(true);
    close([o2[0], o2[1], o2[2]], [1, 0, 0], 1e-9); // rotation about X keeps X
    // angle snapping to 15°
    const sn = dragMatrix(beginDrag('rz', c, m, screenRay(v, p0.x, p0.y))!, screenRay(v, p1.x, p1.y), { angleSnap: Math.PI / 12 });
    close([sn[0], sn[1]], [Math.cos(Math.PI / 4), Math.sin(Math.PI / 4)], 1e-9);
  });
});

describe('surface placement', () => {
  const g: SceneGeometry = {
    // floor quad at y = 0 (normal +Y by winding), and a wall at z = −1 facing +Z
    positions: new Float32Array([-1, 0, 1, 1, 0, 1, 1, 0, -1, -1, 0, 0, 1, 0, 0, 0, 1, 0]).map((x, i) => (i >= 9 ? [-1, 0, -1, 1, 0, -1, 0, 1, -1][i - 9] : x)),
    normals: new Float32Array(18), tangents: new Float32Array(24), uv0: new Float32Array(12),
    indices: new Uint32Array([0, 1, 2, 3, 4, 5]), triMaterial: new Uint32Array(2), triFlags: new Uint32Array(2),
  };

  it('rebuilds the hit point and geometric normal from (primId, u, v)', () => {
    const h = surfaceFromTexel(g, { primId: 0, u: 0.25, v: 0.5 })!;
    // p = (1−u−v)·a + u·b + v·c
    close(h.position, [0.25 * -1 + 0.25 * 1 + 0.5 * 1, 0, 0.25 + 0.25 - 0.5]);
    close(h.ng, [0, 1, 0], 1e-12);
    expect(surfaceFromTexel(g, { primId: BVH_MISS, u: 0, v: 0 })).toBeUndefined();
    expect(surfaceFromTexel(g, { primId: 9, u: 0, v: 0 })).toBeUndefined();
    close(surfaceFromTexel(g, { primId: 1, u: 0, v: 0 })!.ng, [0, 0, 1], 1e-12);
  });

  it('places at hit + ε·Ng (viewer side), emitting away from the surface or toward the camera', () => {
    const hit = { position: [0.2, 0, 0.1] as [number, number, number], ng: [0, -1, 0] as [number, number, number] }; // flipped normal
    const eye = [0, 2, 3];
    const m = placementMatrix(hit, eye, { epsilon: 0.01, facing: 'surface' });
    expect(isRigid(m)).toBe(true);
    close([m[12], m[13], m[14]], [0.2, 0.01, 0.1], 1e-7); // oriented toward the viewer's side (+Y)
    close(emitDirOf(m), [0, 1, 0], 1e-7);                  // emits along +Ng: local +Z = −Ng
    const mc = placementMatrix(hit, eye, { epsilon: 0.01, facing: 'camera' });
    const toEye = [0 - 0.2, 2 - 0.01, 3 - 0.1];
    const n = Math.hypot(...toEye);
    close(emitDirOf(mc), toEye.map((x) => x / n), 1e-6);
    // frames for axis-aligned emission directions stay rigid (up-vector degeneracy handled)
    for (const d of [[0, 1, 0], [0, -1, 0], [1, 0, 0], [0, 0, -1]]) expect(isRigid(frameFromEmission(d, [0, 0, 0]))).toBe(true);
    // quatLookDir and frameFromEmission agree on the emission axis
    const q = quatLookDir([0.3, -0.8, 0.2]);
    const n2 = Math.hypot(0.3, -0.8, 0.2);
    close(emitDirOf(poseToMatrix([0, 0, 0], q)), [0.3 / n2, -0.8 / n2, 0.2 / n2], 1e-12);
  });
});
