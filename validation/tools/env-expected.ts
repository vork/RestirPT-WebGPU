// Expected images of the M3c analytic env scenes (plan §7.2 C0q/C0r/C0s; env §5.2), independent of Cycles and of our
// WGSL: f64 ray casting of the package geometry (validation/scenes/make-m3c.ts, "expected.kind": "env-analytic"),
// the env as the GPU/Cycles sampler reconstructs it (bilinear, repeat/repeat, 8-bit fraction: ENV-U7 — Cycles on Metal
// uses the same hardware sampler) and midpoint quadrature over the env's (u, v) domain (dω = 2π² sin(πv) du dv).
//   npx tsx validation/tools/env-expected.ts --package DIR --out expected.pfm [--ss 4] [--json]
// Per camera subsample (ss×ss per pixel, box filter = Cycles BOX 1 px):
//   miss                         visibleToCamera · L_env(d)
//   V1 Lambert (mix 0)           ρ/π · E(n),  E(n) = ∫ L_env(ω) max(0, n·ω) dω   (n = flat facet normal, facing the ray)
//   V1 GGX r = 0, F ≡ 1 (mirror) glossy · L_env(reflect(d, n))
//   V1 GGX r > 0, F ≡ 1          constant env only: glossy · L · E_ss(n·V, α = r²) (single-scatter GGX albedo, f64 quadrature)
// Valid only for scenes whose surfaces never see each other (one convex object, or a plane): the package declares it.
import { readdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { bvhTrace64 } from '../../src/core/bvh/cpu-trace.ts';
import { BVH_MISS } from '../../src/core/bvh/layout.ts';
import { buildBvh } from '../../src/core/bvh/sah-builder.ts';
import { encodePFM } from '../../src/core/io/pfm.ts';
import { readScenePackage } from '../../src/core/scene/scene-package.ts';
import type { EnvironmentData } from '../../src/core/scene/types.ts';

type V3 = [number, number, number];

/** Cycles direction_to_equirectangular after b = R_z(γ)·C·d (math.md#env-mapping), f64. */
export function envUV(d: V3, g: number): [number, number] {
  const cg = Math.cos(g), sg = Math.sin(g);
  const bx = cg * d[0] + sg * d[2], by = sg * d[0] - cg * d[2], bz = d[1];
  const phi = bx === 0 && by === 0 ? 0 : Math.atan2(by, bx);
  return [(phi - Math.PI) / (-2 * Math.PI), (Math.atan2(Math.hypot(bx, by), bz) - Math.PI) / -Math.PI];
}

export function envDir(u: number, v: number, g: number): V3 {
  const phi = -2 * Math.PI * u + Math.PI, theta = -Math.PI * v + Math.PI;
  const st = Math.sin(theta);
  const bx = st * Math.cos(phi), by = st * Math.sin(phi), bz = Math.cos(theta);
  const cg = Math.cos(g), sg = Math.sin(g);
  return [cg * bx + sg * by, bz, -(-sg * bx + cg * by)];
}

/** Bilinear repeat/repeat lookup (rows bottom-up) with the 8-bit fraction of the Metal sampler, times strength·tint. */
export function envLookup(e: EnvironmentData, u: number, v: number, out: Float64Array): void {
  const W = e.width, H = e.height, t = e.texels;
  const x = u * W - 0.5, y = v * H - 0.5;
  const x0 = Math.floor(x), y0 = Math.floor(y);
  const fx = Math.round((x - x0) * 256) / 256, fy = Math.round((y - y0) * 256) / 256;
  const c0 = ((x0 % W) + W) % W, c1 = (c0 + 1) % W, r0 = ((y0 % H) + H) % H, r1 = (r0 + 1) % H;
  const a = 4 * (r0 * W + c0), b = 4 * (r0 * W + c1), c = 4 * (r1 * W + c0), d = 4 * (r1 * W + c1);
  for (let k = 0; k < 3; k++) {
    const s = e.strength * e.tint[k];
    out[k] = s * ((t[a + k] * (1 - fx) + t[b + k] * fx) * (1 - fy) + (t[c + k] * (1 - fx) + t[d + k] * fx) * fy);
  }
}

/** E(n) = ∫ L_env(ω) max(0, n·ω) dω for every normal, one pass over a (W·q)×(H·q) midpoint grid (f64). */
export function irradiance(e: EnvironmentData, normals: V3[], minCols = 2048): Float64Array {
  const q = Math.max(4, Math.ceil(minCols / e.width));
  const Wf = e.width * q, Hf = e.height * q;
  const out = new Float64Array(3 * normals.length);
  const L = new Float64Array(3);
  const nx = Float64Array.from(normals, (n) => n[0]), ny = Float64Array.from(normals, (n) => n[1]), nz = Float64Array.from(normals, (n) => n[2]);
  for (let r = 0; r < Hf; r++) {
    const v = (r + 0.5) / Hf;
    const dOmega = (2 * Math.PI * Math.PI * Math.sin(Math.PI * v)) / (Wf * Hf);
    for (let c = 0; c < Wf; c++) {
      const u = (c + 0.5) / Wf;
      const w = envDir(u, v, e.rotationZ);
      envLookup(e, u, v, L);
      const l0 = L[0] * dOmega, l1 = L[1] * dOmega, l2 = L[2] * dOmega;
      for (let i = 0; i < normals.length; i++) {
        const cs = nx[i] * w[0] + ny[i] * w[1] + nz[i] * w[2];
        if (cs > 0) { out[3 * i] += l0 * cs; out[3 * i + 1] += l1 * cs; out[3 * i + 2] += l2 * cs; }
      }
    }
  }
  return out;
}

/** Single-scatter GGX albedo E_ss(μ) (F ≡ 1) on a μ grid, midpoint quadrature in NDF space (tests/material/quadrature.ts). */
export function ggxAlbedoTable(alpha: number, nMu = 1025, n = 512): Float64Array {
  const a2 = alpha * alpha;
  const lam = (c: number) => { const cc = Math.max(Math.abs(c), 1e-7); return 0.5 * (Math.sqrt(1 + a2 * Math.max(1 / (cc * cc) - 1, 0)) - 1); };
  const tab = new Float64Array(nMu);
  for (let m = 0; m < nMu; m++) {
    const mu = Math.max(m / (nMu - 1), 1e-4);
    const V = [Math.sqrt(1 - mu * mu), 0, mu];
    const lI = lam(mu);
    let acc = 0;
    for (let i = 0; i < n; i++) {
      const sv = ((i + 0.5) / n) * (Math.PI / 2);
      const dw = Math.sin(2 * sv) * (Math.PI / 2);
      const t2 = alpha * alpha * Math.tan(sv) ** 2;
      const ch = 1 / Math.sqrt(1 + t2), sh = Math.sqrt(t2) * ch;
      for (let j = 0; j < n; j++) {
        const phi = ((j + 0.5) / n) * 2 * Math.PI;
        const vh = V[0] * sh * Math.cos(phi) + V[2] * ch;
        if (!(vh > 0)) continue;
        const Lz = 2 * vh * ch - mu;
        if (Lz < 0) continue;
        acc += (vh * dw) / ((1 + lI + lam(Lz)) * mu * ch);
      }
    }
    tab[m] = acc / (n * n);
  }
  return tab;
}

const lerpTab = (tab: Float64Array, mu: number) => {
  const x = Math.min(Math.max(mu, 0), 1) * (tab.length - 1);
  const i = Math.min(Math.floor(x), tab.length - 2), f = x - i;
  return tab[i] * (1 - f) + tab[i + 1] * f;
};

export async function expectedImage(pkgDir: string, ss = 4): Promise<{ width: number; height: number; data: Float32Array; info: Record<string, unknown> }> {
  const files = new Map<string, Uint8Array>();
  for (const f of readdirSync(pkgDir)) files.set(f, new Uint8Array(readFileSync(path.join(pkgDir, f))));
  const p = await readScenePackage(files);
  const exp = (p.json as unknown as { expected?: { kind?: string } }).expected;
  if (exp?.kind !== 'env-analytic') throw new Error(`${pkgDir}: expected.kind ${exp?.kind} is not env-analytic`);
  const s = p.scene, env = s.env;
  if (!env) throw new Error(`${pkgDir}: no env`);
  const g = s.geometry;
  const bvh = buildBvh(g.positions, g.indices, { mt: true, woop: false });
  const W = p.render.width, H = p.render.height;
  const M = Array.from(p.camera.matrix);
  const ty = Math.tan(p.camera.yfov / 2), tx = ty * W / H;
  // env constant? (C0q: GGX closed form)
  let constant = true;
  for (let k = 4; k < env.texels.length && constant; k += 4) for (let c = 0; c < 3; c++) if (env.texels[k + c] !== env.texels[c]) constant = false;
  // unique facet normals of Lambert triangles → irradiance
  const nrm = (t: number): V3 => {
    const I = g.indices, P = g.positions;
    const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
    const e1 = [P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2]], e2 = [P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]];
    const n: V3 = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
    const l = Math.hypot(...n);
    return [n[0] / l, n[1] / l, n[2] / l];
  };
  const nTri = g.indices.length / 3;
  const kind = s.materials.map((m) => {
    if (m.model !== 'v1' || !m.v1) throw new Error(`material ${m.name}: only V1 is supported by env-expected`);
    if (m.v1.mix === 0) return 'lambert';
    if (m.v1.mix === 1 && m.v1.roughness === 0) return 'mirror';
    if (m.v1.mix === 1) { if (!constant) throw new Error(`${m.name}: rough GGX needs a constant env`); return 'ggx'; }
    throw new Error(`${m.name}: mixed V1 not supported`);
  });
  // Pass 1: trace every subsample; pass 2 shades. Lambert irradiance is computed only for the facet sides actually seen.
  const nSub = W * H * ss * ss;
  const hitPrim = new Int32Array(nSub).fill(-1);
  const hitSide = new Int8Array(nSub);
  const dirs = new Float64Array(3 * nSub);
  let k = 0;
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      for (let sy = 0; sy < ss; sy++) {
        for (let sx = 0; sx < ss; sx++, k++) {
          const u = (sx + 0.5) / ss, v = (sy + 0.5) / ss;
          const dc = [(2 * (c + u) / W - 1) * tx, (2 * (H - 1 - r + v) / H - 1) * ty, -1];
          const d: V3 = [M[0] * dc[0] + M[4] * dc[1] + M[8] * dc[2], M[1] * dc[0] + M[5] * dc[1] + M[9] * dc[2], M[2] * dc[0] + M[6] * dc[1] + M[10] * dc[2]];
          const dl = Math.hypot(...d);
          dirs.set([d[0] / dl, d[1] / dl, d[2] / dl], 3 * k);
          const hit = bvhTrace64(bvh, g.positions, g.indices, [M[12], M[13], M[14]], [d[0] / dl, d[1] / dl, d[2] / dl]);
          if (hit.primId === BVH_MISS) continue;
          hitPrim[k] = hit.primId;
          const n = nrm(hit.primId);
          hitSide[k] = n[0] * d[0] + n[1] * d[1] + n[2] * d[2] > 0 ? -1 : 1;   // facet side facing the ray
        }
      }
    }
  }
  const normIndex = new Map<number, number>();
  const normals: V3[] = [];
  for (let i = 0; i < nSub; i++) {
    const t = hitPrim[i];
    if (t < 0 || kind[g.triMaterial[t]] !== 'lambert') continue;
    const key = hitSide[i] * (t + 1);
    if (normIndex.has(key)) continue;
    const n = nrm(t);
    normIndex.set(key, normals.length);
    normals.push([hitSide[i] * n[0], hitSide[i] * n[1], hitSide[i] * n[2]]);
  }
  const t0 = performance.now();
  const E = normals.length ? irradiance(env, normals) : new Float64Array(0);
  const tE = performance.now() - t0;
  const tabs = new Map<number, Float64Array>();
  for (const m of s.materials) if (m.v1 && m.v1.mix === 1 && m.v1.roughness > 0) tabs.set(m.v1.roughness, ggxAlbedoTable(m.v1.roughness ** 2));
  const data = new Float32Array(W * H * 3);
  const L = new Float64Array(3);
  const acc = new Float64Array(3);
  k = 0;
  for (let p = 0; p < W * H; p++) {
    acc.fill(0);
    for (let j = 0; j < ss * ss; j++, k++) {
      const d: V3 = [dirs[3 * k], dirs[3 * k + 1], dirs[3 * k + 2]];
      const t = hitPrim[k];
      if (t < 0) {
        if (env.visibleToCamera) { const [eu, ev] = envUV(d, env.rotationZ); envLookup(env, eu, ev, L); for (let q = 0; q < 3; q++) acc[q] += L[q]; }
        continue;
      }
      const mat = s.materials[g.triMaterial[t]];
      const n0 = nrm(t);
      const n: V3 = [hitSide[k] * n0[0], hitSide[k] * n0[1], hitSide[k] * n0[2]];
      const k0 = kind[g.triMaterial[t]];
      if (k0 === 'lambert') {
        const i = normIndex.get(hitSide[k] * (t + 1))!;
        for (let q = 0; q < 3; q++) acc[q] += (mat.v1!.diffuse[q] / Math.PI) * E[3 * i + q];
      } else if (k0 === 'mirror') {
        const dn = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
        const rf: V3 = [d[0] - 2 * dn * n[0], d[1] - 2 * dn * n[1], d[2] - 2 * dn * n[2]];
        const [eu, ev] = envUV(rf, env.rotationZ);
        envLookup(env, eu, ev, L);
        for (let q = 0; q < 3; q++) acc[q] += mat.v1!.glossy[q] * L[q];
      } else {
        const Ess = lerpTab(tabs.get(mat.v1!.roughness)!, -(d[0] * n[0] + d[1] * n[1] + d[2] * n[2]));
        for (let q = 0; q < 3; q++) acc[q] += mat.v1!.glossy[q] * env.strength * env.tint[q] * env.texels[q] * Ess;
      }
    }
    for (let q = 0; q < 3; q++) data[3 * p + q] = acc[q] / (ss * ss);
  }
  return { width: W, height: H, data, info: { normals: normals.length, irradianceMs: Math.round(tE), totalMs: Math.round(performance.now() - t0), ss, constantEnv: constant, kinds: [...new Set(kind)] } };
}

async function main(): Promise<void> {
  const { values: a } = parseArgs({ options: { package: { type: 'string' }, out: { type: 'string' }, ss: { type: 'string', default: '4' }, json: { type: 'boolean', default: false } } });
  if (!a.package || !a.out) { console.error('usage: env-expected.ts --package DIR --out FILE.pfm [--ss 4] [--json]'); process.exit(2); }
  const img = await expectedImage(a.package, Number(a.ss));
  writeFileSync(a.out, encodePFM({ width: img.width, height: img.height, channels: 3, data: img.data }));
  let m = [0, 0, 0];
  for (let i = 0; i < img.data.length; i += 3) m = m.map((x, k) => x + img.data[i + k]);
  m = m.map((x) => x / (img.width * img.height));
  const res = { ok: true, out: a.out, mean: m, ...img.info };
  console.log(a.json ? JSON.stringify(res) : `expected image ${a.out}: mean ${m.map((x) => x.toPrecision(7)).join(', ')} (${JSON.stringify(img.info)})`);
}

if (process.argv[1] && import.meta.url === new URL(`file://${path.resolve(process.argv[1])}`).href) main().catch((e) => { console.error(e); process.exit(1); });
