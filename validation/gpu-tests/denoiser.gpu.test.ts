// M5.5 denoiser passes against the f64 reference (tests/denoise/dn-ref.ts; docs/decisions/denoiser.md §3–§7, §11):
//   U-DN-1  dn_temporal on a reset frame: demodulated input, guides (f32 distance, oct 2×16 normal), moments, codes
//   U-DN-2  dn_temporal with a fractional camera motion: bilinear reprojection with TD12's predicate (depth step,
//           background, image border), ring fallback / disocclusion codes, α and n, EMA moments
//   U-DN-3  dn_variance (7×7 spatial estimate for n < 4) and dn_atrous (N = 2: the feedback level and the remodulated
//           output; background pass-through, L1 added back)
//   U-DN-4  dn_gradient / dn_grad_filter: tile sums and λ from synthetic tState pairs (forward OK / light-zero /
//           geometric, inverse OK / light-zero) with FW written by the previous frame; λ drives α and n in dn_temporal
//   U-DN-5  DN9: a held frame and the timing re-run reproduce the frame's outputs bit for bit; the frame's passes carry
//           no timestampWrites (Q3), the timing submit does
// Chrome lane authoritative; dawn.node is a pre-check.
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { DebugResources, DebugViewRegistry } from '../../src/core/render/debug-views.ts';
import { Denoiser, type DenoiseFrame } from '../../src/core/render/denoise/denoiser.ts';
import { DENOISER_DEFAULTS, DN_REPROJ, DNF, dnTiles, type DenoiserSettings } from '../../src/core/render/denoise/layout.ts';
import { FrameUniformBuffer, JITTER_NONE, type CameraState } from '../../src/core/render/frame-uniforms.ts';
import { liveDenoisers } from '../../src/core/render/denoise/registry.ts';
import {
  emptyState, refFilter, refLambda, refPairs, refTemporal, st16, rayDir, type RefPixel, type RefState, type RefTState, type V3,
} from '../../tests/denoise/dn-ref.ts';
import { getTestGpu, releaseTestGpu } from './device-factory.ts';

afterAll(releaseTestGpu);

const W = 40, H = 32, P = W * H;
const GB_HIT = 1;
const YFOV = 50 * Math.PI / 180;
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const camAt = (x: number): CameraState => ({ camToWorld: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1], yfov: YFOV, znear: 1e-4 });

/** Synthetic scene: top 4 rows background, a plane at z = −2, a box face at z = −1.4 for columns 24…31 (depth step), a
 *  black-albedo patch (demodulation factor 1); pixel-centre rays of camera `cam` (the jittered hit = the centre). */
function scene(cam: CameraState): RefPixel[] {
  const px: RefPixel[] = [];
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
    if (r < 4) { px.push({ hit: false, pos: [0, 0, 0], ns: [0, 0, 1], albedo: [0, 0, 0] }); continue; }
    const d = rayDir(cam, c, r, W, H);
    const o = [cam.camToWorld[12], cam.camToWorld[13], cam.camToWorld[14]];
    const zPlane = c >= 24 && c < 32 ? -1.4 : -2;
    const t = (zPlane - o[2]) / d[2];
    const pos: V3 = [Math.fround(o[0] + t * d[0]), Math.fround(o[1] + t * d[1]), Math.fround(o[2] + t * d[2])];
    const albedo: V3 = c < 6 && r > 20 ? [0, 0, 0] : c < 20 ? [0.8, 0.3, 0.2] : [0.2, 0.6, 0.9];
    px.push({ hit: true, pos, ns: [0, 0, 1], albedo });
  }
  return px;
}
/** GBufTexel array (80 B; passes/gbuffer.wgsl). */
function gbufBytes(px: RefPixel[]): ArrayBuffer {
  const b = new ArrayBuffer(P * 80), f = new Float32Array(b), u = new Uint32Array(b);
  px.forEach((p, i) => {
    const o = i * 20;
    f.set([0, 0, 1], o); f[o + 3] = 0; f.set(p.ns, o + 4); f[o + 7] = p.hit ? -p.pos[2] : 0;
    f.set(p.pos, o + 8); u[o + 11] = p.hit ? 0 : 0xffffffff; f.set(p.albedo, o + 12); u[o + 15] = p.hit ? GB_HIT : 0;
  });
  return b;
}

interface Rig {
  device: GPUDevice; dn: Denoiser; debug: DebugResources; fu: FrameUniformBuffer;
  gbuf: GPUBuffer; radiance: GPUTexture; l1: GPUTexture; colour: GPUTexture; arena: GPUBuffer; resW: GPUBuffer; resFinal: GPUBuffer;
}
let rig: Rig;

beforeAll(async () => {
  const { device, features, wgslLanguageFeatures } = await getTestGpu();
  const debug = new DebugResources(device, new DebugViewRegistry());
  await debug.init();
  debug.resize(W, H);
  debug.update({ ...debug.settings, mode: 0 }, 0);
  const dn = await Denoiser.create(device, { debugLayout: debug.layout, colorFormat: 'rgba32float', features, wgslLanguageFeatures });
  dn.resize(W, H);
  const tex = (label: string, usage: number) => device.createTexture({ label, size: [W, H], format: 'rgba32float', usage });
  const T = GPUTextureUsage;
  rig = {
    device, dn, debug, fu: new FrameUniformBuffer(device),
    gbuf: device.createBuffer({ size: P * 80, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
    radiance: tex('radiance', T.TEXTURE_BINDING | T.COPY_DST | T.COPY_SRC), l1: tex('l1', T.TEXTURE_BINDING | T.COPY_DST),
    colour: tex('colour', T.STORAGE_BINDING | T.TEXTURE_BINDING | T.COPY_SRC | T.COPY_DST),
    arena: device.createBuffer({ size: (64 + 20 * P) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
    resW: device.createBuffer({ size: P * 160, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
    resFinal: device.createBuffer({ size: P * 160, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
  };
});

const writeTex = (t: GPUTexture, rgb: ArrayLike<number>) => {
  const a = new Float32Array(P * 4);
  for (let i = 0; i < P; i++) { a[4 * i] = rgb[3 * i]; a[4 * i + 1] = rgb[3 * i + 1]; a[4 * i + 2] = rgb[3 * i + 2]; a[4 * i + 3] = 1; }
  rig.device.queue.writeTexture({ texture: t }, a, { bytesPerRow: W * 16 }, [W, H]);
};
async function readTex(t: GPUTexture): Promise<ArrayBuffer> {
  const bpp = t.format === 'rgba32float' ? 16 : t.format === 'rgba16float' || t.format === 'rg32uint' || t.format === 'rg32float' ? 8 : 4;
  const bpr = Math.ceil((t.width * bpp) / 256) * 256;
  const b = rig.device.createBuffer({ size: bpr * t.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const e = rig.device.createCommandEncoder();
  e.copyTextureToBuffer({ texture: t }, { buffer: b, bytesPerRow: bpr }, [t.width, t.height]);
  rig.device.queue.submit([e.finish()]);
  await b.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(b.getMappedRange());
  const out = new Uint8Array(t.width * t.height * bpp);
  for (let y = 0; y < t.height; y++) out.set(src.subarray(y * bpr, y * bpr + t.width * bpp), y * t.width * bpp);
  b.unmap(); b.destroy();
  return out.buffer;
}
const f16 = async (t: GPUTexture) => Array.from(new Float16Array(await readTex(t)));
const f32 = async (t: GPUTexture) => Array.from(new Float32Array(await readTex(t)));

/** Max |gpu − ref| / (|ref| + abs) over the entries. */
function maxErr(gpu: ArrayLike<number>, ref: ArrayLike<number>, abs = 1e-4): { err: number; at: number } {
  let err = 0, at = -1;
  for (let i = 0; i < ref.length; i++) {
    const e = Math.abs(gpu[i] - ref[i]) / (Math.abs(ref[i]) + abs);
    if (!(e <= err)) { err = e; at = i; }
  }
  return { err, at };
}

interface FrameIn {
  cam: CameraState; prev: CameraState; px: RefPixel[]; radiance: Float64Array; l1: Float64Array; reset: boolean;
  settings: DenoiserSettings; kind: 'restir' | 'pt'; restir?: DenoiseFrame['restir']; advanced?: boolean;
}
let frameNo = 0;
function encodeFrame(f: FrameIn, record?: GPUComputePassDescriptor[]): void {
  rig.fu.write({
    camera: f.cam, prevCamera: f.prev, width: W, height: H, frameIndex: frameNo, seedIndex: frameNo, runSeed: 1, flags: 0, jitterMode: JITTER_NONE,
    jitter: [0.5, 0.5], origin: [0, 0, 0], exposure: 1, time: 0, dt: 0, sceneDiag: 10,
  });
  frameNo++;
  rig.device.queue.writeBuffer(rig.gbuf, 0, gbufBytes(f.px));
  writeTex(rig.radiance, f.radiance);
  writeTex(rig.l1, f.l1);
  rig.dn.setSettings(f.settings);
  const enc = rig.device.createCommandEncoder();
  if (record) {
    const orig = enc.beginComputePass.bind(enc);
    enc.beginComputePass = (d?: GPUComputePassDescriptor) => { record.push(d ?? {}); return orig(d); };
  }
  rig.debug.beginFrame(enc);
  if (f.kind === 'pt') {
    const e2 = rig.device.createCommandEncoder();
    e2.copyTextureToTexture({ texture: rig.radiance }, { texture: rig.colour }, [W, H]);
    rig.device.queue.submit([e2.finish()]);
  }
  const ok = rig.dn.encode(enc, {
    kind: f.kind, advanced: f.advanced ?? true, reset: f.reset, frameUniforms: rig.fu.buffer, gbuf: rig.gbuf,
    radiance: f.kind === 'restir' ? rig.radiance : undefined, l1: f.kind === 'restir' ? rig.l1 : undefined, colour: rig.colour, restir: f.restir, debugGroup: rig.debug.bindGroup,
  });
  expect(ok).toBe(true);
  rig.device.queue.submit([enc.finish()]);
}

function noisy(px: RefPixel[], seed: number, scale = 1): Float64Array {
  const r = rng(seed);
  const out = new Float64Array(P * 3);
  px.forEach((p, i) => { for (let k = 0; k < 3; k++) out[3 * i + k] = p.hit ? scale * (0.2 + 0.8 * r()) * (k === 0 ? 1.2 : 1) * (p.pos[0] > 0 ? 2 : 1) : 0.3 + 0.1 * k; });
  return out;
}
const l1Of = (px: RefPixel[]) => Float64Array.from({ length: P * 3 }, (_, j) => (px[Math.floor(j / 3)].hit ? 0.05 * (j % 3) : 0.3 + 0.1 * (j % 3)));

/** GPU state of the current parity → RefState (as stored). */
async function gpuState(): Promise<{ st: RefState; mom: number[]; atrous: [number[], number[]]; hist: number[]; colour: number[] }> {
  const t = rig.dn.textures!;
  const cur = rig.dn.parity;
  const mom = await f16(t.mom[cur]);
  const hist4 = await f16(t.hist[cur]);
  const alb4 = await f16(t.alb[cur]);
  const geo = new Uint32Array(await readTex(t.geo[cur]));
  const st = emptyState(W, H);
  for (let i = 0; i < P; i++) {
    st.mom.set(mom.slice(4 * i, 4 * i + 4), 4 * i);
    st.hist.set(hist4.slice(4 * i, 4 * i + 3), 3 * i);
    st.alb.set(alb4.slice(4 * i, 4 * i + 3), 3 * i);
    st.dist[i] = new Float32Array(Uint32Array.of(geo[2 * i]).buffer)[0];
  }
  return { st, mom, atrous: [await f16(t.atrous[0]), await f16(t.atrous[1])], hist: hist4, colour: await f32(rig.colour) };
}

describe('denoiser passes vs the f64 reference', () => {
  const S0: DenoiserSettings = { ...DENOISER_DEFAULTS, iterations: 0 };
  let prevRef: RefState;
  let pxA: RefPixel[];

  it('U-DN-1 dn_temporal on a reset frame (+ variance, copy output)', async () => {
    const cam = camAt(0);
    pxA = scene(cam);
    const radiance = noisy(pxA, 1), l1 = l1Of(pxA);
    encodeFrame({ cam, prev: cam, px: pxA, radiance, l1, reset: true, settings: S0, kind: 'restir' });
    const ref = refTemporal({ W, H, px: pxA, cam, prevCam: cam, radiance, l1, flags: DNF.RESET | DNF.HAS_L1, settings: S0 }, emptyState(W, H));
    const g = await gpuState();
    expect(maxErr(g.atrous[0], ref.atrous0, 1e-3).err).toBeLessThan(2e-3);
    expect(maxErr(g.mom, ref.state.mom, 1e-3).err).toBeLessThan(2e-3);
    expect(maxErr(g.st.dist, ref.state.dist, 1e-6).err).toBeLessThan(1e-6);
    const fl = refFilter(S0, ref.state, ref.atrous0, pxA, radiance, l1);
    expect(maxErr(g.atrous[1], fl.variance, 1e-3).err).toBeLessThan(3e-3);
    expect(maxErr(g.colour.filter((_, j) => j % 4 !== 3), fl.colour, 1e-3).err).toBeLessThan(3e-3);
    // demodulation: colour = c·a′ + L1 = the input up to fp16 storage
    prevRef = ref.state;
    prevRef.n = ref.state.n;
  });

  it('U-DN-2 dn_temporal: fractional camera motion, bilinear reprojection with the TD12 predicate', async () => {
    const prev = camAt(0), cam = camAt(0.013);   // ≈ 0.31 px at z = −2, 0.44 px on the box face
    const px = scene(cam);
    const radiance = noisy(px, 2), l1 = l1Of(px);
    // the reference reads the GPU's stored previous state (rgba16float values, f32 distances)
    const g0 = await gpuState();
    const prevSt: RefState = { ...g0.st, n: prevRef.n };
    encodeFrame({ cam, prev, px, radiance, l1, reset: false, settings: S0, kind: 'restir' });
    const ref = refTemporal({ W, H, px, cam, prevCam: prev, radiance, l1, flags: DNF.HAS_L1, settings: S0 }, prevSt);
    const g = await gpuState();
    expect(maxErr(g.atrous[0], ref.atrous0, 1e-3).err).toBeLessThan(3e-3);
    expect(maxErr(g.mom, ref.state.mom, 1e-3).err).toBeLessThan(3e-3);
    const codes = new Set(ref.code);
    expect(codes.has(DN_REPROJ.FULL) && codes.has(DN_REPROJ.PARTIAL) && codes.has(DN_REPROJ.BG)).toBe(true);
    // n = 2 everywhere a history was found
    let n2 = 0;
    for (let i = 0; i < P; i++) if (ref.code[i] === DN_REPROJ.FULL) { expect(g.mom[4 * i + 2]).toBe(2); n2++; }
    expect(n2).toBeGreaterThan(P / 2);
    prevRef = ref.state;
  });

  it('U-DN-3 dn_variance and dn_atrous (N = 2): feedback level and remodulated output', async () => {
    const S2: DenoiserSettings = { ...DENOISER_DEFAULTS, iterations: 2 };
    const cam = camAt(0.013);
    const px = scene(cam);
    const radiance = noisy(px, 3), l1 = l1Of(px);
    const g0 = await gpuState();
    const prevSt: RefState = { ...g0.st, n: prevRef.n };
    encodeFrame({ cam, prev: cam, px, radiance, l1, reset: false, settings: S2, kind: 'restir' });
    const ref = refTemporal({ W, H, px, cam, prevCam: cam, radiance, l1, flags: DNF.HAS_L1, settings: S2 }, prevSt);
    const fl = refFilter(S2, ref.state, ref.atrous0, px, radiance, l1);
    const g = await gpuState();
    // feedback = output of iteration 0 (dnHist[cur]); atrous[0] = iteration 0 too (iteration 1 is the final pass)
    const fb = Array.from({ length: P * 3 }, (_, j) => fl.feedback[4 * Math.floor(j / 3) + (j % 3)]);
    expect(maxErr(Array.from({ length: P * 3 }, (_, j) => g.hist[4 * Math.floor(j / 3) + (j % 3)]), fb, 1e-3).err).toBeLessThan(5e-3);
    expect(maxErr(g.atrous[1], fl.variance, 1e-3).err).toBeLessThan(5e-3);
    expect(maxErr(g.colour.filter((_, j) => j % 4 !== 3), fl.colour, 1e-3).err).toBeLessThan(5e-3);
    // the filter smooths: the variance of the demodulated colour drops on the plane
    const lumOf = (a: ArrayLike<number>, stride: number, i: number) => 0.2126 * a[stride * i] + 0.7152 * a[stride * i + 1] + 0.0722 * a[stride * i + 2];
    const sd = (f: (i: number) => number) => { const v: number[] = []; for (let r = 10; r < 30; r++) for (let c = 8; c < 18; c++) v.push(f(r * W + c)); const m = v.reduce((a, b) => a + b) / v.length; return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length); };
    expect(sd((i) => lumOf(fl.colour, 3, i))).toBeLessThan(0.5 * sd((i) => lumOf(radiance, 3, i)));
    prevRef = ref.state;
  });

  it('U-DN-4 dn_gradient / dn_grad_filter: tile sums, λ, and λ driving α', async () => {
    const S: DenoiserSettings = { ...DENOISER_DEFAULTS, iterations: 1 };
    const cam = camAt(0.013);
    const px = scene(cam);
    const r = rng(9);
    // frame A: FW = lum(F)·W of res[final] is written into dnMom[cur].a
    const res = new Uint32Array(P * 40);
    const fwRef = new Float64Array(P);
    for (let i = 0; i < P; i++) {
      const Wv = Math.fround(0.5 + r()), F = [Math.fround(r()), Math.fround(r()), Math.fround(r())];
      res.set([new Uint32Array(Float32Array.of(Wv).buffer)[0], ...new Uint32Array(Float32Array.from(F).buffer)], i * 40);
      fwRef[i] = st16(Math.fround((0.2126 * F[0] + 0.7152 * F[1] + 0.0722 * F[2])) * Wv);
    }
    rig.device.queue.writeBuffer(rig.resFinal, 0, res);
    const radA = noisy(px, 4), l1 = l1Of(px);
    const restirA = { arena: rig.arena, resW: rig.resW, resFinal: rig.resFinal, tsBase: 64, gradient: false, lightingChanged: false, inverse: true };
    encodeFrame({ cam, prev: cam, px, radiance: radA, l1, reset: false, settings: S, kind: 'restir', restir: restirA });
    const gA = await gpuState();
    for (let i = 0; i < P; i++) expect(Math.abs(gA.mom[4 * i + 3] - fwRef[i])).toBeLessThanOrEqual(1e-3 * fwRef[i]);   // FW for every pixel (background too)
    // frame B: synthetic tState (forward OK ×2 in the left half, light-zero and geometric failures, inverse pairs)
    const words = new Uint32Array(64 + 20 * P);
    const resWv = new Uint32Array(P * 40);
    const ts: RefTState[] = [];
    const bitsF = (x: number) => new Uint32Array(Float32Array.of(x).buffer)[0];
    for (let i = 0; i < P; i++) {
      const x = i % W, y = Math.floor(i / W);
      const t: RefTState = { flags: 1, qPrime: i, cP: 5, fwdCode: 0, wc: 0, wp: 0, invCode: 1, piRecomp: 0 };
      if (y < 4) t.flags = 0;
      else if (x < 20) t.wp = Math.fround(5 * 2 * fwRef[i]);                    // light ×2
      else if (x < 28) { t.fwdCode = 6; }                                          // removed light
      else if (x < 34) { t.fwdCode = 7; t.flags |= 32 | 128; t.wc = Math.fround(r()); t.invCode = 0; t.piRecomp = Math.fround(0.5 * r()); }   // O1 + inverse OK
      else { t.wp = Math.fround(5 * fwRef[i]); t.flags |= 32 | 128; t.wc = Math.fround(r()); t.invCode = 10; }   // static fwd + inverse occluded
      ts.push(t);
      const o = 64 + 20 * i;
      words[o + 8] = t.qPrime; words[o + 9] = bitsF(t.cP); words[o + 10] = t.fwdCode; words[o + 11] = t.flags;
      words[o + 13] = bitsF(t.wc); words[o + 14] = bitsF(t.wp); words[o + 15] = t.invCode; words[o + 17] = bitsF(t.piRecomp);
      const Fc = [Math.fround(0.3 + r()), Math.fround(0.3 + r()), Math.fround(0.3 + r())];
      resWv.set(new Uint32Array(Float32Array.from([1, ...Fc]).buffer), i * 40);
    }
    rig.device.queue.writeBuffer(rig.arena, 0, words);
    rig.device.queue.writeBuffer(rig.resW, 0, resWv);
    const piC = (i: number) => { const F = new Float32Array(resWv.buffer, i * 160 + 4, 3); return 0.2126 * F[0] + 0.7152 * F[1] + 0.0722 * F[2]; };
    const fwGpu = (q: number) => gA.mom[4 * q + 3];   // the stored FW (the gradient reads dnMom[prev].a)
    const pairs = ts.map((t, i) => refPairs(t, fwGpu, piC(i), true, P));
    const refL = refLambda(W, H, pairs);
    const prevSt: RefState = { ...gA.st, n: prevRef.n };
    const radB = noisy(px, 5);
    encodeFrame({ cam, prev: cam, px, radiance: radB, l1, reset: false, settings: S, kind: 'restir', restir: { ...restirA, gradient: true, lightingChanged: true } });
    const t = rig.dn.textures!;
    const tiles = await f32(t.gradTile);
    const lam = await f32(t.lambda);
    const [tx, ty] = dnTiles(W, H);
    for (let k = 0; k < tx * ty; k++) {
      expect(Math.abs(tiles[2 * k] - refL.tiles[2 * k])).toBeLessThan(1e-4 * (1 + Math.abs(refL.tiles[2 * k + 1])));
      expect(Math.abs(tiles[2 * k + 1] - refL.tiles[2 * k + 1])).toBeLessThan(1e-4 * (1 + Math.abs(refL.tiles[2 * k + 1])));
      expect(Math.abs(lam[k] - refL.lambda[k])).toBeLessThan(1e-4);
    }
    expect(Math.max(...lam)).toBeGreaterThan(0.3);
    // λ drives the temporal pass (α, n) exactly as in the reference
    const ref = refTemporal({ W, H, px, cam, prevCam: cam, radiance: radB, l1, flags: DNF.HAS_L1 | DNF.FW | DNF.GRADIENT | DNF.LAMBDA | DNF.INVERSE, settings: S, fw: Float64Array.from({ length: P }, (_, q) => fwGpu(q)), lambdaTiles: Float64Array.from(lam) }, prevSt);
    const g = await gpuState();
    expect(maxErr(g.mom.filter((_, j) => j % 4 !== 3), Array.from(ref.state.mom).filter((_, j) => j % 4 !== 3), 1e-3).err).toBeLessThan(3e-3);
    let reset = 0;
    for (let i = 0; i < P; i++) if (px[i].hit && ref.lambda[i] >= S.lambda1) { expect(g.mom[4 * i + 2]).toBe(1); reset++; }
    expect(reset).toBeGreaterThan(50);
    // gate closed (static lighting): λ is ignored
    encodeFrame({ cam, prev: cam, px, radiance: radB, l1, reset: false, settings: S, kind: 'restir', restir: { ...restirA, gradient: true, lightingChanged: false } });
    const g2 = await gpuState();
    for (let i = 0; i < P; i++) if (px[i].hit) expect(g2.mom[4 * i + 2]).toBeGreaterThan(1);
  });

  it('U-DN-5 held frame and timing re-run are bit-identical; no timestamp writes in the frame', async () => {
    const S: DenoiserSettings = { ...DENOISER_DEFAULTS };
    const cam = camAt(0.02);
    const px = scene(cam);
    const radiance = noisy(px, 6), l1 = l1Of(px);
    const rec: GPUComputePassDescriptor[] = [];
    encodeFrame({ cam, prev: camAt(0.013), px, radiance, l1, reset: false, settings: S, kind: 'restir' }, rec);
    expect(rec.length).toBe(rig.dn.passNames.length);
    expect(rec.every((d) => !d.timestampWrites)).toBe(true);
    const a = await gpuState();
    // held frame: same parity, same plan
    const enc = rig.device.createCommandEncoder();
    expect(rig.dn.encode(enc, { kind: 'restir', advanced: false, reset: false, frameUniforms: rig.fu.buffer, gbuf: rig.gbuf, radiance: rig.radiance, l1: rig.l1, colour: rig.colour, debugGroup: rig.debug.bindGroup })).toBe(true);
    rig.device.queue.submit([enc.finish()]);
    const b = await gpuState();
    expect(b.colour).toEqual(a.colour);
    expect(b.hist).toEqual(a.hist);
    expect(b.mom).toEqual(a.mom);
    if (rig.device.features.has('timestamp-query')) {
      const t = await rig.dn.time(3);
      expect(t && t.totalMs > 0 && t.passes.length === rig.dn.passNames.length).toBe(true);
      const c = await gpuState();
      expect(c.colour).toEqual(a.colour);
      expect(c.mom).toEqual(a.mom);
    }
    // PT input: the colour target holds the sample; the denoiser copies it before overwriting
    encodeFrame({ cam, prev: cam, px, radiance, l1, reset: true, settings: { ...S, iterations: 0 }, kind: 'pt' });
    const ref = refTemporal({ W, H, px, cam, prevCam: cam, radiance, flags: DNF.RESET, settings: { ...S, iterations: 0 } }, emptyState(W, H));
    const g = await gpuState();
    expect(maxErr(g.atrous[0], ref.atrous0, 1e-3).err).toBeLessThan(2e-3);
  });

  it('registry: the live denoiser is visible to the T16 check and gone after destroy()', async () => {
    expect(liveDenoisers()).toBe(1);
    const { device, features, wgslLanguageFeatures } = await getTestGpu();
    const d2 = await Denoiser.create(device, { debugLayout: rig.debug.layout, colorFormat: 'rgba16float', features, wgslLanguageFeatures });
    expect(liveDenoisers()).toBe(2);
    d2.destroy();
    rig.dn.destroy();
    expect(liveDenoisers()).toBe(0);
  });
});
