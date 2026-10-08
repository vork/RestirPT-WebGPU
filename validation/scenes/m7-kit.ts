// M7 scene kit (docs/decisions/m7-api.md §4): smooth meshes with shared vertices and analytic normals, procedural
// tangent-space normal maps (RGBA8, Non-Color), and the CPU mirror of the Cycles Normal Map node used by the tests and
// the analytic expectations. Builds on scene-kit.ts (its MeshBuilder arrays are appended to directly: smooth vertices
// are shared, so quantizeScene does NOT flag their triangles TRI_FLAT and the package is exported smooth).
import { MeshBuilder, norm, texture, type V3 } from './kit-core.ts';
import type { TextureData } from '../../src/core/scene/types.ts';

/** Cycles svm_node_normal_map, tangent space (f64 mirror of scene-data.wgsl normal_map_cycles); null = Cycles' fallback. */
export function normalMapCycles(rgb: readonly number[], s: number, T: readonly number[], sgn: number, nU: readonly number[]): V3 | null {
  const c = rgb.map((x) => 2 * (x - 0.5));
  c[0] *= s; c[1] *= s;
  const sat = Math.min(1, Math.max(0, s));
  c[2] = 1 + (c[2] - 1) * sat;                       // mix(1, c.z, saturate(s))
  const B = [sgn * (nU[1] * T[2] - nU[2] * T[1]), sgn * (nU[2] * T[0] - nU[0] * T[2]), sgn * (nU[0] * T[1] - nU[1] * T[0])];
  const N = [0, 1, 2].map((k) => c[0] * T[k] + c[1] * B[k] + c[2] * nU[k]);
  const l = Math.hypot(N[0], N[1], N[2]);
  if (!(l > 0) || !Number.isFinite(l)) return null;
  return [N[0] / l, N[1] / l, N[2] / l];
}

const pushV = (mb: MeshBuilder, p: readonly number[], n: readonly number[], uv: readonly number[]): number => {
  mb.pos.push(p[0], p[1], p[2]); mb.nrm.push(n[0], n[1], n[2]); mb.uv.push(uv[0], uv[1]);
  return mb.pos.length / 3 - 1;
};
const pushT = (mb: MeshBuilder, a: number, b: number, c: number, mat: number) => { mb.idx.push(a, b, c); mb.mat.push(mat); };

/** Smooth UV sphere: shared vertices, analytic normals (outward), glTF UVs (u = φ/2π·su, v = θ/π·sv), seam duplicated,
 *  polar caps cut at θ0 (closed by a flat disk each, so the mesh is closed and has no degenerate fans). */
export function smoothSphere(mb: MeshBuilder, c: V3, r: number, nLat: number, nLon: number, mat: number, o: { uvScale?: [number, number]; capMat?: number } = {}): void {
  const [su, sv] = o.uvScale ?? [1, 1];
  const th0 = 0.06, th1 = Math.PI - 0.06;
  const at = (i: number, j: number) => base + i * (nLon + 1) + j;
  const base = mb.pos.length / 3;
  for (let i = 0; i <= nLat; i++) {
    const th = th0 + (th1 - th0) * i / nLat;
    for (let j = 0; j <= nLon; j++) {
      const ph = 2 * Math.PI * j / nLon;
      const n: V3 = [Math.sin(th) * Math.cos(ph), Math.cos(th), -Math.sin(th) * Math.sin(ph)];
      pushV(mb, [c[0] + r * n[0], c[1] + r * n[1], c[2] + r * n[2]], n, [su * j / nLon, sv * i / nLat]);
    }
  }
  for (let i = 0; i < nLat; i++) for (let j = 0; j < nLon; j++) {
    pushT(mb, at(i, j), at(i + 1, j), at(i + 1, j + 1), mat);
    pushT(mb, at(i, j), at(i + 1, j + 1), at(i, j + 1), mat);
  }
  // flat caps (own vertices with the cap normal: TRI_FLAT after quantizeScene)
  const capMat = o.capMat ?? mat;
  for (const top of [true, false]) {
    const i = top ? 0 : nLat;
    const th = top ? th0 : th1;
    const y = c[1] + r * Math.cos(th);
    const n: V3 = [0, top ? 1 : -1, 0];
    // planar UV projection (x, z) of the cap: non-degenerate, so MikkTSpace's bitangent sign is well defined
    const capUv = (p: readonly number[]): [number, number] => [0.5 + (p[0] - c[0]) / (2 * r), 0.5 + (p[2] - c[2]) / (2 * r)];
    const ctr = pushV(mb, [c[0], y, c[2]], n, capUv(c));
    const ring: number[] = [];
    for (let j = 0; j < nLon; j++) {
      const p = mb.pos.slice(3 * at(i, j), 3 * at(i, j) + 3);
      ring.push(pushV(mb, p, n, capUv(p)));
    }
    ring.push(ring[0]);
    for (let j = 0; j < nLon; j++) {
      if (top) pushT(mb, ctr, ring[j], ring[j + 1], capMat); else pushT(mb, ctr, ring[j + 1], ring[j], capMat);   // outward (CCW seen from outside)
    }
  }
}

/** Open smooth height-field sheet y = h0 + A·sin(kx·x)·sin(kz·z) over [x0,x1]×[z0,z1] (n×n quads), analytic normals,
 *  UV = (x, z) scaled into [0, su]×[0, sv]. One-sided geometry: the place where Ng·L ≤ 0 < Ns·L is not self-occluded. */
export function smoothSheet(mb: MeshBuilder, x0: number, x1: number, z0: number, z1: number, h0: number, A: number, kx: number, kz: number,
  n: number, mat: number, o: { uvScale?: [number, number] } = {}): void {
  const [su, sv] = o.uvScale ?? [1, 1];
  const base = mb.pos.length / 3;
  for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) {
    const x = x0 + (x1 - x0) * j / n, z = z0 + (z1 - z0) * i / n;
    const y = h0 + A * Math.sin(kx * x) * Math.sin(kz * z);
    const dydx = A * kx * Math.cos(kx * x) * Math.sin(kz * z), dydz = A * kz * Math.sin(kx * x) * Math.cos(kz * z);
    pushV(mb, [x, y, z], norm([-dydx, 1, -dydz]), [su * j / n, sv * i / n]);
  }
  const at = (i: number, j: number) => base + i * (n + 1) + j;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    // counter-clockwise seen from +y: (i, j) → (i+1, j) → (i+1, j+1) has normal +y for z increasing with i
    pushT(mb, at(i, j), at(i + 1, j), at(i + 1, j + 1), mat);
    pushT(mb, at(i, j), at(i + 1, j + 1), at(i, j + 1), mat);
  }
}

/** Smooth torus around +Y (major R, minor r), shared vertices, analytic normals, UV (φ/2π·su, ψ/2π·sv). Closed. */
export function smoothTorus(mb: MeshBuilder, c: V3, R: number, r: number, nU: number, nV: number, mat: number, o: { uvScale?: [number, number] } = {}): void {
  const [su, sv] = o.uvScale ?? [1, 1];
  const base = mb.pos.length / 3;
  for (let i = 0; i <= nU; i++) for (let j = 0; j <= nV; j++) {
    const ph = 2 * Math.PI * i / nU, ps = 2 * Math.PI * j / nV;
    const n: V3 = [Math.cos(ps) * Math.cos(ph), Math.sin(ps), -Math.cos(ps) * Math.sin(ph)];
    const p: V3 = [c[0] + (R + r * Math.cos(ps)) * Math.cos(ph), c[1] + r * Math.sin(ps), c[2] - (R + r * Math.cos(ps)) * Math.sin(ph)];
    pushV(mb, p, n, [su * i / nU, sv * j / nV]);
  }
  const at = (i: number, j: number) => base + i * (nV + 1) + j;
  for (let i = 0; i < nU; i++) for (let j = 0; j < nV; j++) {
    pushT(mb, at(i, j), at(i + 1, j), at(i + 1, j + 1), mat);
    pushT(mb, at(i, j), at(i + 1, j + 1), at(i, j + 1), mat);
  }
}

// ---- procedural tangent-space normal maps (RGBA8, encoded (n + 1)/2, Non-Color) -------------------------------------

const enc = (n: readonly number[]): [number, number, number, number] => [n[0], n[1], n[2]].map((x) => (x + 1) / 2 * 255).concat(255) as [number, number, number, number];

/** Constant tilt: every texel is the unit vector at polar angle θ toward azimuth φ in tangent space (x = T, y = B). */
export function tiltMap(name: string, theta: number, phi: number, size = 4): TextureData {
  const n = [Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)];
  return texture(name, size, size, () => enc(n), { filter: 'nearest' });
}

/** Hemispherical bumps on a k×k grid (height field h = √(R² − d²)), normals from the analytic gradient. */
export function bumpsMap(name: string, size: number, k: number, depth = 1): TextureData {
  return texture(name, size, size, (x, y) => {
    const u = ((x + 0.5) / size) * k, v = ((y + 0.5) / size) * k;
    const dx = (u % 1) - 0.5, dy = (v % 1) - 0.5;
    const d2 = dx * dx + dy * dy, R2 = 0.16;
    if (d2 >= R2) return enc([0, 0, 1]);
    const h = Math.sqrt(R2 - d2);
    // ∂h/∂u = −dx/h; image y runs DOWN (glTF v), and tangent-space +y is +v_Blender = up the image ⇒ ∂/∂y_ts = −∂/∂v
    return enc(norm([depth * dx / h, -depth * dy / h, 1]));
  });
}

/** Bevelled tiles: k×k tiles with a bevel of width w (fraction of a tile) sloping at angle a. */
export function tilesMap(name: string, size: number, k: number, w = 0.12, a = 0.6): TextureData {
  return texture(name, size, size, (x, y) => {
    const u = ((x + 0.5) / size) * k % 1, v = ((y + 0.5) / size) * k % 1;
    let nx = 0, ny = 0;
    if (u < w) nx = -Math.sin(a); else if (u > 1 - w) nx = Math.sin(a);
    if (v < w) ny = Math.sin(a); else if (v > 1 - w) ny = -Math.sin(a);
    return enc(norm([nx, ny, 1]));
  });
}

/** Smooth wave ripples: n = normalize(−A·k·cos(k·u·2π), −A·k·cos(k·v·2π)·0.5, 1). */
export function wavesMap(name: string, size: number, k: number, A = 0.35): TextureData {
  return texture(name, size, size, (x, y) => {
    const u = (x + 0.5) / size, v = (y + 0.5) / size;
    return enc(norm([-A * Math.cos(2 * Math.PI * k * u), 0.5 * A * Math.cos(2 * Math.PI * k * v), 1]));
  });
}
