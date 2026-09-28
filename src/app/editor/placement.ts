// Place a light on a surface (plan §5 M3a "place on a surface via V-buffer readback"): read the V-buffer texel
// (primId, u, v bits) under the clicked pixel, rebuild the hit point and geometric normal on the CPU from the ORIGINAL
// SceneGeometry (un-recentred world coordinates, primId is the stable triangle index; plan §1.3), and build a rigid
// light frame:
//   position    = hit + ε·Ng   (Ng oriented toward the viewer, so the light sits on the visible side)
//   orientation = emission axis (local −Z) along +Ng ("facing along −Ng": local +Z = −Ng) — area/spot lights shine
//                 away from the surface — or toward the camera (option).
import type { SceneGeometry } from '../../core/scene/types.ts';
import { add, cross, dot, normalize, scale, sub, type Vec3 } from './picking.ts';

export const BVH_MISS = 0xffffffff;

export interface VBufferTexel { primId: number; u: number; v: number; matId: number }
export interface SurfaceHit { primId: number; position: Vec3; ng: Vec3; u: number; v: number }

/** Read one texel of the rgba32uint V-buffer (primId, bitcast u, bitcast v, matId). */
export async function readVBufferTexel(device: GPUDevice, vbuf: GPUTexture, x: number, y: number): Promise<VBufferTexel> {
  const buf = device.createBuffer({ label: 'vbuffer-pick', size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const enc = device.createCommandEncoder({ label: 'vbuffer-pick' });
    enc.copyTextureToBuffer({ texture: vbuf, origin: { x, y } }, { buffer: buf, bytesPerRow: 256 }, { width: 1, height: 1 });
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const u32 = new Uint32Array(buf.getMappedRange().slice(0, 16));
    buf.unmap();
    const f = new Float32Array(u32.buffer);
    return { primId: u32[0], u: f[1], v: f[2], matId: u32[3] };
  } finally {
    buf.destroy();
  }
}

/** Hit point + geometric normal (winding order, unit) of triangle `primId` at barycentrics (u, v). */
export function surfaceFromTexel(g: SceneGeometry, t: Pick<VBufferTexel, 'primId' | 'u' | 'v'>): SurfaceHit | undefined {
  const n = g.indices.length / 3;
  if (t.primId === BVH_MISS || t.primId >= n || !Number.isFinite(t.u) || !Number.isFinite(t.v)) return undefined;
  const P = g.positions;
  const ia = g.indices[3 * t.primId] * 3, ib = g.indices[3 * t.primId + 1] * 3, ic = g.indices[3 * t.primId + 2] * 3;
  const a: Vec3 = [P[ia], P[ia + 1], P[ia + 2]], b: Vec3 = [P[ib], P[ib + 1], P[ib + 2]], c: Vec3 = [P[ic], P[ic + 1], P[ic + 2]];
  const w = 1 - t.u - t.v;
  const position: Vec3 = [w * a[0] + t.u * b[0] + t.v * c[0], w * a[1] + t.u * b[1] + t.v * c[1], w * a[2] + t.u * b[2] + t.v * c[2]];
  const ng = cross(sub(b, a), sub(c, a));
  if (!(Math.hypot(...ng) > 0)) return undefined;
  return { primId: t.primId, position, ng: normalize(ng), u: t.u, v: t.v };
}

/** Rigid column-major frame whose local −Z is `emit` (unit), +Y as close to world +Y as possible (else world −Z). */
export function frameFromEmission(emit: readonly number[], position: readonly number[]): Float32Array {
  const z = normalize(scale(emit, -1));
  let up: Vec3 = [0, 1, 0];
  if (Math.abs(dot(z, up)) > 0.999) up = [0, 0, -1];
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, z[0], z[1], z[2], 0, position[0], position[1], position[2], 1]);
}

export type PlacementFacing = 'surface' | 'camera';

export interface PlacementOptions {
  /** Offset along the normal (m). */
  epsilon: number;
  facing: PlacementFacing;
}

/** Light matrix for a surface hit seen from `eye` (plan §5 M3a placement rule, see the header). */
export function placementMatrix(hit: Pick<SurfaceHit, 'position' | 'ng'>, eye: readonly number[], o: PlacementOptions): Float32Array {
  let n = normalize(hit.ng);
  if (dot(n, sub(eye, hit.position)) < 0) n = scale(n, -1);
  const pos = add(hit.position, scale(n, o.epsilon));
  const emit = o.facing === 'camera' ? normalize(sub(eye, pos)) : n;
  return frameFromEmission(emit, pos);
}

/** Default placement offset: area lights hug the surface, point/spot lights stand off a little. */
export function defaultPlacementEpsilon(type: string, sceneDiag: number): number {
  const d = Math.max(sceneDiag, 1e-3);
  return type === 'rect' || type === 'disk' ? 1e-3 * d : 0.03 * d;
}
