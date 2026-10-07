// f64 CPU reference of the A-SVGF-lite passes (docs/decisions/denoiser.md §3–§7; mirror of
// src/core/render/denoise/shaders/*.wgsl). Storage points round like the GPU: rgba16float (Math.f16round of the
// clamped value), 2×16 snorm octahedral normals, f32 distances. Used by tests/denoise (formulas) and by
// validation/gpu-tests/denoiser.gpu.test.ts (each pass against this reference).
import { DNF, DNI, DN_REPROJ, atrousPlan, dnTiles, type DenoiserSettings } from '../../src/core/render/denoise/layout.ts';

export type V3 = [number, number, number];
export const lum = (c: ArrayLike<number>): number => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
const FP16_MAX = 65504;
export const fp16 = (x: number): number => (Number.isFinite(x) ? Math.min(Math.max(x, 0), FP16_MAX) : 0);
/** Value after a store to rgba16float. */
export const st16 = (x: number): number => Math.f16round(fp16(x));
export const dot = (a: ArrayLike<number>, b: ArrayLike<number>): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export function demodFactor(a: V3): V3 {
  if (!(Math.max(a[0], a[1], a[2]) >= 0.02)) return [1, 1, 1];
  return [Math.max(a[0], 0.02) + 0.04, Math.max(a[1], 0.02) + 0.04, Math.max(a[2], 0.02) + 0.04];
}

// ------------------------------------------------------------------------------------------------ octahedral normals

export function octEncode(n: V3): [number, number] {
  const a = Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2]);
  let p: [number, number] = [n[0] / Math.max(a, 1e-20), n[1] / Math.max(a, 1e-20)];
  if (n[2] < 0) p = [(1 - Math.abs(p[1])) * (p[0] >= 0 ? 1 : -1), (1 - Math.abs(p[0])) * (p[1] >= 0 ? 1 : -1)];
  return p;
}
export function octDecode(e: [number, number]): V3 {
  const v: V3 = [e[0], e[1], 1 - Math.abs(e[0]) - Math.abs(e[1])];
  const t = Math.max(-v[2], 0);
  v[0] += v[0] >= 0 ? -t : t;
  v[1] += v[1] >= 0 ? -t : t;
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
}
const snorm = (x: number) => Math.floor(0.5 + 32767 * Math.min(Math.max(x, -1), 1));
/** WGSL pack2x16snorm. */
export function pack2x16snorm(a: number, b: number): number { return ((snorm(a) & 0xffff) | ((snorm(b) & 0xffff) << 16)) >>> 0; }
export function unpack2x16snorm(u: number): [number, number] {
  const s = (x: number) => { const v = x & 0x8000 ? x - 0x10000 : x; return Math.max(v / 32767, -1); };
  return [s(u & 0xffff), s(u >>> 16)];
}
/** The normal after the guide round trip (oct 2×16 snorm). */
export function guideNormal(n: V3): V3 { const e = octEncode(n); return octDecode(unpack2x16snorm(pack2x16snorm(e[0], e[1]))); }

// ------------------------------------------------------------------------------------------------ camera (frame.wgsl)

export interface RefCamera { camToWorld: ArrayLike<number>; yfov: number }
function inv(m: ArrayLike<number>): number[] {
  const o = new Array<number>(16).fill(0);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[c * 4 + r] = m[r * 4 + c];
  for (let r = 0; r < 3; r++) o[12 + r] = -(o[r] * m[12] + o[4 + r] * m[13] + o[8 + r] * m[14]);
  o[15] = 1;
  return o;
}
function toCam(cam: RefCamera, p: V3): V3 {
  const w = inv(cam.camToWorld);
  return [0, 1, 2].map((r) => w[r] * p[0] + w[4 + r] * p[1] + w[8 + r] * p[2] + w[12 + r]) as V3;
}
export const viewDepth = (cam: RefCamera, p: V3): number => -toCam(cam, p)[2];
export function project(cam: RefCamera, p: V3, W: number, H: number): [number, number] {
  const pc = toCam(cam, p);
  if (pc[2] > -1e-12) return [-1, -1];
  const t = Math.tan(cam.yfov / 2), a = W / H;
  const nx = pc[0] / (-pc[2] * t * a), ny = pc[1] / (-pc[2] * t);
  return [(nx + 1) * 0.5 * W, H - (ny + 1) * 0.5 * H];
}
export const camPos = (cam: RefCamera): V3 => [cam.camToWorld[12], cam.camToWorld[13], cam.camToWorld[14]];
/** World point hit by the pixel-centre ray of (c, r) on the plane z = zPlane (camera looking down −Z). */
export function rayDir(cam: RefCamera, c: number, r: number, W: number, H: number, jit: [number, number] = [0.5, 0.5]): V3 {
  const t = Math.tan(cam.yfov / 2);
  const x = (2 * (c + jit[0]) / W - 1) * t * W / H, y = (2 * (H - 1 - r + jit[1]) / H - 1) * t;
  const m = cam.camToWorld;
  const d: V3 = [m[0] * x + m[4] * y - m[8], m[1] * x + m[5] * y - m[9], m[2] * x + m[6] * y - m[10]];
  const l = Math.hypot(...d);
  return [d[0] / l, d[1] / l, d[2] / l];
}

// ------------------------------------------------------------------------------------------------ frame state

export interface RefPixel { hit: boolean; pos: V3; ns: V3; albedo: V3 }
export interface RefState {
  W: number; H: number;
  /** rgba16float values as stored. */
  hist: Float64Array;   // W·H·3
  mom: Float64Array;    // W·H·4 (μ, σ, n, FW)
  alb: Float64Array;    // W·H·4 accumulated demodulation factor ā and its history length n_a (Changelog DN-1, DN-2)
  l1: Float64Array;     // W·H·3 accumulated L1 (DN-2)
  dist: Float64Array;   // W·H (f32)
  n: V3[];              // decoded guide normals
}
export function emptyState(W: number, H: number): RefState {
  return { W, H, hist: new Float64Array(W * H * 3), mom: new Float64Array(W * H * 4), alb: new Float64Array(W * H * 4), l1: new Float64Array(W * H * 3), dist: new Float64Array(W * H), n: Array.from({ length: W * H }, () => [0, 0, 1] as V3) };
}

export interface RefTemporalIn {
  W: number; H: number; px: RefPixel[]; cam: RefCamera; prevCam: RefCamera;
  radiance: ArrayLike<number>; l1?: ArrayLike<number>;
  flags: number; settings: DenoiserSettings;
  /** lum(F·W) per pixel (DNF.FW). */
  fw?: ArrayLike<number>;
  /** λ per tile (DNF.GRADIENT). */
  lambdaTiles?: ArrayLike<number>;
  cameraMoved?: boolean;
}
export interface RefTemporalOut { state: RefState; atrous0: Float64Array; code: Uint32Array; alpha: Float64Array; lambda: Float64Array; demod: Float64Array }

function tapValid(prev: RefState, c: [number, number], n: V3, z: number): boolean {
  if (c[0] < 0 || c[1] < 0 || c[0] >= prev.W || c[1] >= prev.H) return false;
  const i = c[1] * prev.W + c[0];
  const zp = prev.dist[i];
  if (!(zp > 0)) return false;
  if (!(dot(n, prev.n[i]) >= 0.5)) return false;
  return Math.abs(z - zp) <= 0.1 * zp;
}

export function refLambdaAt(inp: RefTemporalIn, x: number, y: number): number {
  if (!(inp.flags & DNF.GRADIENT) || !inp.lambdaTiles) return 0;
  if (!((inp.flags & DNF.LAMBDA) || ((inp.flags & DNF.LAMBDA_CAM) && inp.cameraMoved))) return 0;
  const [tx, ty] = dnTiles(inp.W, inp.H);
  const tc = [(x + 0.5) / 8 - 0.5, (y + 0.5) / 8 - 0.5];
  const b = tc.map(Math.floor), f = [tc[0] - b[0], tc[1] - b[1]];
  const cl = (v: number, m: number) => Math.min(Math.max(v, 0), m);
  const L = (cx: number, cy: number) => inp.lambdaTiles![cl(cy, ty - 1) * tx + cl(cx, tx - 1)];
  const a = L(b[0], b[1]) * (1 - f[0]) + L(b[0] + 1, b[1]) * f[0];
  const c = L(b[0], b[1] + 1) * (1 - f[0]) + L(b[0] + 1, b[1] + 1) * f[0];
  return a * (1 - f[1]) + c * f[1];
}

/** dn_temporal. */
export function refTemporal(inp: RefTemporalIn, prev: RefState): RefTemporalOut {
  const { W, H, settings: s } = inp;
  const moved = inp.cam.yfov !== inp.prevCam.yfov || Array.from({ length: 16 }, (_, k) => inp.cam.camToWorld[k] !== inp.prevCam.camToWorld[k]).some(Boolean);
  const st = emptyState(W, H);
  const atrous0 = new Float64Array(W * H * 4), code = new Uint32Array(W * H), alpha = new Float64Array(W * H), lambdaOut = new Float64Array(W * H), demod = new Float64Array(W * H * 3);
  const pc = camPos(inp.prevCam), cc = camPos(inp.cam);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, p = inp.px[i];
    const fw = inp.flags & DNF.FW && inp.fw ? st16(inp.fw[i]) : 0;
    if (!p.hit) { st.mom[4 * i + 3] = fw; code[i] = DN_REPROJ.BG; continue; }
    const a = demodFactor(p.albedo);
    const l1 = inp.flags & DNF.HAS_L1 && inp.l1 ? [inp.l1[3 * i], inp.l1[3 * i + 1], inp.l1[3 * i + 2]] : [0, 0, 0];
    const c = [0, 1, 2].map((k) => fp16((inp.radiance[3 * i + k] - l1[k]) / a[k]));
    for (let k = 0; k < 3; k++) demod[3 * i + k] = c[k];
    const l = lum(c);
    // reprojection
    let hw = 0, hc = [0, 0, 0], ha = [0, 0, 0, 0], hl = [0, 0, 0], hmu = 0, hm2 = 0, hn = 0, cd: number = DN_REPROJ.NONE;
    const acc = (cx: number, cy: number, w: number) => {
      const j = cy * W + cx;
      for (let k = 0; k < 3; k++) { hc[k] += w * prev.hist[3 * j + k]; hl[k] += w * prev.l1[3 * j + k]; }
      for (let k = 0; k < 4; k++) ha[k] += w * prev.alb[4 * j + k];
      const mu = prev.mom[4 * j], sg = prev.mom[4 * j + 1];
      hmu += w * mu; hm2 += w * (sg * sg + mu * mu); hn += w * prev.mom[4 * j + 2]; hw += w;
    };
    if (inp.flags & DNF.RESET) cd = DN_REPROJ.RESET;
    else if (!moved) { if (prev.dist[i] > 0) { acc(x, y, 1); cd = DN_REPROJ.FULL; } }
    else if (viewDepth(inp.prevCam, p.pos) >= 1e-12) {
      const z = Math.hypot(p.pos[0] - pc[0], p.pos[1] - pc[1], p.pos[2] - pc[2]);
      const pr = project(inp.prevCam, p.pos, W, H);
      const sp = [pr[0] - 0.5, pr[1] - 0.5];
      if (sp[0] >= -2 && sp[1] >= -2 && sp[0] <= W + 2 && sp[1] <= H + 2) {
        const b = [Math.floor(sp[0]), Math.floor(sp[1])], f = [sp[0] - b[0], sp[1] - b[1]];
        let valid = 0;
        for (let j = 0; j < 4; j++) {
          const o = [j & 1, j >> 1];
          const w = (o[0] ? f[0] : 1 - f[0]) * (o[1] ? f[1] : 1 - f[1]);
          const cxy: [number, number] = [b[0] + o[0], b[1] + o[1]];
          if (tapValid(prev, cxy, p.ns, z)) { acc(cxy[0], cxy[1], w); valid++; }
        }
        if (hw >= 1e-3) cd = valid === 4 ? DN_REPROJ.FULL : DN_REPROJ.PARTIAL;
        else {
          hw = 0; hc = [0, 0, 0]; ha = [0, 0, 0, 0]; hl = [0, 0, 0]; hmu = 0; hm2 = 0; hn = 0;
          const cr = [Math.floor(sp[0] + 0.5), Math.floor(sp[1] + 0.5)];
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            const cxy: [number, number] = [cr[0] + dx, cr[1] + dy];
            if (tapValid(prev, cxy, p.ns, z)) acc(cxy[0], cxy[1], 1);
          }
          if (hw > 0) cd = DN_REPROJ.RING;
        }
      }
    }
    if (hw > 0) { hc = hc.map((v) => v / hw); ha = ha.map((v) => v / hw); hl = hl.map((v) => v / hw); hmu /= hw; hm2 /= hw; hn /= hw; }
    const lambda = refLambdaAt(inp, x, y);
    const lp = Math.min(Math.max((lambda - s.lambda0) / Math.max(s.lambda1 - s.lambda0, 1e-6), 0), 1);
    const nIn = hw > 0 ? hn : 0;
    const nNew = Math.min(1 + (1 - lp) * (lp > 0 ? Math.min(nIn, 1 / s.alphaMin) : nIn), s.nMax);   // DN-8
    const al = Math.max(s.alphaMin, 1 / nNew, lp);
    const nA = Math.min(1 + (hw > 0 ? ha[3] : 0), s.nMax);
    const alA = moved ? Math.max(1 / nA, 0.1) : 1 / nA;
    let col = c, alb: number[] = a, l1o: number[] = l1, mu = l, vr = 0;
    if (hw > 0) {
      col = [0, 1, 2].map((k) => hc[k] + al * (c[k] - hc[k]));
      alb = [0, 1, 2].map((k) => ha[k] + alA * (a[k] - ha[k]));
      const all = Math.max(alA, lp);
      l1o = [0, 1, 2].map((k) => hl[k] + all * (l1[k] - hl[k]));
      const v0 = Math.max(hm2 - hmu * hmu, 0);
      const d = l - hmu;
      mu = hmu + al * d;
      vr = (1 - al) * (v0 + al * d * d);
    }
    for (let k = 0; k < 3; k++) { st.hist[3 * i + k] = st16(col[k]); atrous0[4 * i + k] = st16(col[k]); st.alb[4 * i + k] = Math.fround(alb[k]); st.l1[3 * i + k] = Math.fround(Math.max(l1o[k], 0)); }   // f32 (DN-7)
    st.alb[4 * i + 3] = nA;
    atrous0[4 * i + 3] = st16(vr * (s.varCorr > 0 ? Math.min(1, s.varCorr * al / (2 - al)) : 1));   // DN-4
    st.mom.set([st16(mu), st16(Math.sqrt(Math.max(vr, 0))), Math.f16round(nNew), fw], 4 * i);
    st.dist[i] = Math.fround(Math.hypot(p.pos[0] - cc[0], p.pos[1] - cc[1], p.pos[2] - cc[2]));
    st.n[i] = guideNormal(p.ns);
    code[i] = cd; alpha[i] = al; lambdaOut[i] = lambda;
  }
  return { state: st, atrous0, code, alpha, lambda: lambdaOut, demod };
}

// ------------------------------------------------------------------------------------------------ edge stops (§7)

export function zgrad(st: RefState, x: number, y: number): [number, number] {
  const z = st.dist[y * st.W + x];
  const g: [number, number] = [0, 0];
  for (let a = 0; a < 2; a++) {
    let best = 1e30;
    for (const s of [-1, 1]) {
      const cx = x + (a === 0 ? s : 0), cy = y + (a === 1 ? s : 0);
      if (cx < 0 || cy < 0 || cx >= st.W || cy >= st.H) continue;
      const zq = st.dist[cy * st.W + cx];
      if (zq > 0) best = Math.min(best, Math.abs(zq - z));
    }
    g[a] = best < 1e29 ? best : 0;
  }
  return g;
}
export function wGeo(s: DenoiserSettings, zc: number, zg: [number, number], nc: V3, zq: number, nq: V3, d: [number, number]): number {
  const phi = s.sigmaZ * (Math.abs(zg[0] * d[0]) + Math.abs(zg[1] * d[1])) + 1e-3 * zc;
  return Math.exp(-Math.abs(zc - zq) / phi) * Math.max(dot(nc, nq), 0) ** s.sigmaN;
}

/** DN-13: the binomial prefilter (radius s.lumPre) of the luminance over geometrically compatible neighbours (r32float). */
export function refLumGuide(s: DenoiserSettings, st: RefState, atrous0: Float64Array): Float64Array {
  const { W, H } = st;
  const g = new Float64Array(W * H);
  const R = s.lumPre;
  const k = R === 2 ? [0.0625, 0.25, 0.375, 0.25, 0.0625] : [0.25, 0.5, 0.25];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, zc = st.dist[i];
    if (!(zc > 0)) continue;
    if (R === 0) { g[i] = lum([atrous0[4 * i], atrous0[4 * i + 1], atrous0[4 * i + 2]]); continue; }
    const zg = zgrad(st, x, y);
    let s0 = 0, ws = 0;
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      const qx = x + dx, qy = y + dy;
      if (qx < 0 || qy < 0 || qx >= W || qy >= H) continue;
      const j = qy * W + qx, zq = st.dist[j];
      if (!(zq > 0)) continue;
      const w = k[dx + R] * k[dy + R] * wGeo(s, zc, zg, st.n[i], zq, st.n[j], [dx, dy]);
      s0 += w * lum([atrous0[4 * j], atrous0[4 * j + 1], atrous0[4 * j + 2]]); ws += w;
    }
    g[i] = s0 / Math.max(ws, 1e-12);
  }
  return g;
}

/** dn_variance: rgba (colour, variance) after the spatial estimate for n < 4. */
export function refVariance(s: DenoiserSettings, st: RefState, atrous0: Float64Array): Float64Array {
  const { W, H } = st;
  const out = new Float64Array(atrous0);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, zc = st.dist[i];
    if (!(zc > 0)) continue;
    const n = st.mom[4 * i + 2];
    let v = atrous0[4 * i + 3];
    if (n < 4) {
      const zg = zgrad(st, x, y);
      let sw = 0, sl = 0, sl2 = 0;
      for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
        const qx = x + dx, qy = y + dy;
        if (qx < 0 || qy < 0 || qx >= W || qy >= H) continue;
        const j = qy * W + qx, zq = st.dist[j];
        if (!(zq > 0)) continue;
        const w = wGeo(s, zc, zg, st.n[i], zq, st.n[j], [dx, dy]);
        const l = lum([atrous0[4 * j], atrous0[4 * j + 1], atrous0[4 * j + 2]]);
        sw += w; sl += w * l; sl2 += w * l * l;
      }
      const m = sl / sw;
      v = Math.max(sl2 / sw - m * m, 0) * 4 / Math.max(n, 1);
    }
    out[4 * i + 3] = st16(v);
  }
  return out;
}

/** One dn_atrous iteration (step 0 = copy) on rgba (colour, variance); returns the stored rgba16float values. */
export function refAtrous(s: DenoiserSettings, st: RefState, src: Float64Array, step: number, lumG?: Float64Array): Float64Array {
  const L = (j: number) => (lumG ? lumG[j] : lum([src[4 * j], src[4 * j + 1], src[4 * j + 2]]));   // DN-13 (iteration 0 only)
  const { W, H } = st;
  const out = new Float64Array(src.length);
  const b3 = [0.0625, 0.25, 0.375, 0.25, 0.0625];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x, zc = st.dist[i];
    if (!(zc > 0)) { out.set(src.subarray(4 * i, 4 * i + 4), 4 * i); continue; }
    let res = [src[4 * i], src[4 * i + 1], src[4 * i + 2], src[4 * i + 3]];
    if (step > 0) {
      const zg = zgrad(st, x, y);
      const lc = L(i);
      const ac = [st16(st.alb[4 * i]), st16(st.alb[4 * i + 1]), st16(st.alb[4 * i + 2])];   // DN-14: ā via dnTap (rgba16float)
      const invA = s.sigmaA > 0 ? 1 / s.sigmaA : 0;
      let v3 = 0, w3 = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const qx = x + dx, qy = y + dy;
        if (qx < 0 || qy < 0 || qx >= W || qy >= H) continue;
        const w = (dx ? 0.25 : 0.5) * (dy ? 0.25 : 0.5);
        v3 += w * src[4 * (qy * W + qx) + 3]; w3 += w;
      }
      const phiL = st.mom[4 * i + 2] >= s.lumMinN ? s.sigmaL * Math.sqrt(Math.max(v3 / w3, 0)) + 1e-6 : 1e30;   // DN-12
      let sc = [0, 0, 0], sv = 0, sw = 0;
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
        const h = b3[dx + 2] * b3[dy + 2];
        if (dx === 0 && dy === 0) { sc = sc.map((v, k) => v + h * src[4 * i + k]); sv += h * h * src[4 * i + 3]; sw += h; continue; }
        const qx = x + dx * step, qy = y + dy * step;
        if (qx < 0 || qy < 0 || qx >= W || qy >= H) continue;
        const j = qy * W + qx, zq = st.dist[j];
        if (!(zq > 0)) continue;
        const wg = wGeo(s, zc, zg, st.n[i], zq, st.n[j], [dx * step, dy * step]);
        if (wg < 1e-6) continue;   // DN-15
        const cq = [src[4 * j], src[4 * j + 1], src[4 * j + 2]];
        const da = [0, 1, 2].reduce((acc, k) => acc + Math.abs(st16(st.alb[4 * j + k]) - ac[k]), 0);
        const wl = Math.exp(-Math.abs(lc - L(j)) / phiL - da * (invA / 3));   // DN-5
        const w = h * wg * wl;
        sc = sc.map((v, k) => v + w * cq[k]); sv += w * w * src[4 * j + 3]; sw += w;
      }
      res = [sc[0] / sw, sc[1] / sw, sc[2] / sw, sv / (sw * sw)];
    }
    out.set(res.map(st16), 4 * i);
  }
  return out;
}

/** The whole filter chain after dn_temporal: dn_variance, the à-trous iterations, the remodulated output. */
export function refFilter(s: DenoiserSettings, st: RefState, atrous0: Float64Array, _px: RefPixel[], radiance: ArrayLike<number>, _l1: ArrayLike<number> | undefined):
  { variance: Float64Array; levels: Float64Array[]; feedback: Float64Array; colour: Float64Array } {
  const variance = refVariance(s, st, atrous0);
  let cur: Float64Array = variance;
  const levels: Float64Array[] = [];
  let feedback: Float64Array = new Float64Array(0);
  const lumG = s.lumPre > 0 ? refLumGuide(s, st, atrous0) : undefined;
  for (const [iter, step, flags] of atrousPlan(s.iterations)) {
    cur = refAtrous(s, st, cur, flags & DNI.COPY ? 0 : step, iter === 0 ? lumG : undefined);
    levels.push(cur);
    if (flags & DNI.FEEDBACK) feedback = cur;
  }
  const colour = new Float64Array(st.W * st.H * 3);
  for (let i = 0; i < st.W * st.H; i++) {
    if (!(st.dist[i] > 0)) { for (let k = 0; k < 3; k++) colour[3 * i + k] = radiance[3 * i + k]; continue; }
    for (let k = 0; k < 3; k++) colour[3 * i + k] = cur[4 * i + k] * st.alb[4 * i + k] + st.l1[3 * i + k];
  }
  return { variance, levels, feedback, colour };
}

// ------------------------------------------------------------------------------------------------ gradient (§5)

export interface RefTState { flags: number; qPrime: number; cP: number; fwdCode: number; wc: number; wp: number; invCode: number; piRecomp: number }
export const SC = { OK: 0, O0_LIGHT: 6, O1: 7, OCCLUDED: 10, ZERO: 11 } as const;
const lightZero = (c: number) => { const sc = c & 0xff; return sc === SC.O0_LIGHT || sc === SC.OCCLUDED || sc === SC.ZERO; };

/** (Δ_f, M_f, Δ_i, M_i, bits) of one pixel (dn_pairs; Changelog DN-2: the inverse pairs weighted by 1/P(s = c)).
 *  `fwPrev` = dnMom[prev].a, `piC` = lum F of res[w][q]. */
export function refPairs(t: RefTState, fwPrev: (q: number) => number, piC: number, inverse: boolean, P: number): [number, number, number, number, number] {
  let df = 0, mf = 0, di = 0, mi = 0, bits = 0;
  if (!(t.flags & 1) || t.qPrime === 0xffffffff || t.qPrime >= P) return [0, 0, 0, 0, 0];
  const pos = (x: number) => (Number.isFinite(x) && x > 0 ? x : 0);
  const b = pos(fwPrev(t.qPrime));
  const wp = pos(t.wp);
  if ((t.fwdCode & 0xff) === SC.OK && t.cP > 0) { const a = pos(wp / t.cP); df = a - b; mf = Math.max(a, b); bits |= 1; }
  else if (lightZero(t.fwdCode)) { df = -b; mf = b; bits |= 1; }
  if (inverse && (t.flags & (32 | 128)) === (32 | 128) && !(t.flags & 256)) {
    const a = pos(t.wc);
    let bi = -1;
    if ((t.invCode & 0xff) === SC.OK) { if (piC > 0) bi = pos(a * t.piRecomp / piC); } else if (lightZero(t.invCode)) bi = 0;
    if (bi >= 0 && a > 0) { const w = (a + wp) / a; di = w * (a - bi); mi = w * Math.max(a, bi); bits |= 2; }
  }
  if (mf > 1e4) { df *= 1e4 / mf; mf = 1e4; }
  if (mi > 1e4) { di *= 1e4 / mi; mi = 1e4; }
  return [df, mf, di, mi, bits];
}
/** Tile sums (8×8; Δ_f, M_f, Δ_i, M_i) and λ = max(|ΣΔ_f|/ΣM_f, |ΣΔ_i|/ΣM_i) of the 3×3-tile windows. */
export function refLambda(W: number, H: number, pairs: number[][], invRadius = 1): { tiles: Float64Array; lambda: Float64Array } {
  const [tx, ty] = dnTiles(W, H);
  const tiles = new Float64Array(tx * ty * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const t = Math.floor(y / 8) * tx + Math.floor(x / 8);
    for (let k = 0; k < 4; k++) tiles[4 * t + k] += pairs[y * W + x][k];
  }
  const lambda = new Float64Array(tx * ty);
  for (let y = 0; y < ty; y++) for (let x = 0; x < tx; x++) {
    const sum = [0, 0, 0, 0];
    const R = Math.max(invRadius, 1);
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      const cx = x + dx, cy = y + dy;
      if (cx < 0 || cy < 0 || cx >= tx || cy >= ty) continue;
      for (let k = 0; k < 4; k++) if (k >= 2 || (Math.abs(dx) <= 1 && Math.abs(dy) <= 1)) sum[k] += tiles[4 * (cy * tx + cx) + k];
    }
    const lf = sum[1] > 1e-8 ? Math.min(Math.abs(sum[0]) / sum[1], 1) : 0;
    const li = sum[3] > 1e-8 ? Math.min(Math.abs(sum[2]) / sum[3], 1) : 0;
    lambda[y * tx + x] = Math.max(lf, li);
  }
  return { tiles, lambda };
}

/** dn_resolve with a static camera and no lighting change in the last 8 frames (Changelog DN-6): every pixel (hit or
 *  background) α_t = 1/n_t, n_t ≤ nMaxT. `prev` = dnTaa[prev] (rgb, n_t), `out` = the remodulated output (fp16-stored).
 *  `dynamic`: the n_t ≤ taaLightMax cap (the variance clipping is not modelled: tests use a smooth input there). */
export function refResolveStatic(s: DenoiserSettings, out: ArrayLike<number>, prev: Float64Array, reset: boolean, P: number, dynamic = false): Float64Array {
  const res = new Float64Array(P * 4);
  for (let i = 0; i < P; i++) {
    const h = !reset ? prev.subarray(4 * i, 4 * i + 4) : new Float64Array(4);
    const nMax = dynamic ? Math.min(s.nMaxT, s.taaLightMax) : s.nMaxT;   // static camera: the lighting window's cap (DN-16)
    const nT = s.resolve ? Math.min(1 + h[3], Math.max(nMax, 1)) : 1;
    for (let k = 0; k < 3; k++) res[4 * i + k] = Math.fround(Math.max(h[3] > 0 && nT > 1 ? h[k] + (out[3 * i + k] - h[k]) / nT : out[3 * i + k], 0));   // f32 (DN-7)
    res[4 * i + 3] = nT;
  }
  return res;
}
