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
  emptyState, lum, refFilter, refLambda, refPairs, refResolveStatic, refTemporal, st16, rayDir, type RefPixel, type RefState, type RefTState, type V3,
} from '../../tests/denoise/dn-ref.ts';
import { getTestGpu, releaseTestGpu } from './device-factory.ts';
import { testPerfFlags } from './restir-fixtures.ts';
import { normalizePerfFlags } from '../../src/core/render/restir/perf-flags.ts';

/** perf2 (perf2-api.md): VITE_PERF_FLAGS reaches the denoiser's pipelines (WP-7: GBUF_48, DN_ZGRAD_TEX, …). */
const PERF = normalizePerfFlags(testPerfFlags());
const GB48 = !!PERF.GBUF_48;

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
/** GBufTexel array (80 B; passes/gbuffer.wgsl). `motion`: per-pixel motion vectors (GB_MOTION_VALID set). With GBUF_48
 *  the stored GBufStore (48 B: pos, flags, ns, motion.x, albedo, motion.y). */
function gbufBytes(px: RefPixel[], motion?: [number, number][]): ArrayBuffer {
  if (GB48) {
    const b = new ArrayBuffer(P * 48), f = new Float32Array(b), u = new Uint32Array(b);
    px.forEach((p, i) => {
      const o = i * 12;
      f.set(p.pos, o); u[o + 3] = p.hit ? GB_HIT : 0; f.set(p.ns, o + 4); f.set(p.albedo, o + 8);
      if (motion) { f[o + 7] = motion[i][0]; f[o + 11] = motion[i][1]; u[o + 3] |= 4; }   // GB_MOTION_VALID
    });
    return b;
  }
  const b = new ArrayBuffer(P * 80), f = new Float32Array(b), u = new Uint32Array(b);
  px.forEach((p, i) => {
    const o = i * 20;
    f.set([0, 0, 1], o); f[o + 3] = 0; f.set(p.ns, o + 4); f[o + 7] = p.hit ? -p.pos[2] : 0;
    f.set(p.pos, o + 8); u[o + 11] = p.hit ? 0 : 0xffffffff; f.set(p.albedo, o + 12); u[o + 15] = p.hit ? GB_HIT : 0;
    if (motion) { f.set(motion[i], o + 16); u[o + 15] |= 4; }   // GB_MOTION_VALID
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
  const dn = await Denoiser.create(device, { debugLayout: debug.layout, colorFormat: 'rgba32float', features, wgslLanguageFeatures, perfFlags: PERF });
  dn.resize(W, H);
  const tex = (label: string, usage: number) => device.createTexture({ label, size: [W, H], format: 'rgba32float', usage });
  const T = GPUTextureUsage;
  rig = {
    device, dn, debug, fu: new FrameUniformBuffer(device),
    gbuf: device.createBuffer({ size: P * (GB48 ? 48 : 80), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
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

/** Moments (μ, σ, n, FW): μ / n / FW at fp16 precision; σ (√(m₂ − μ²) of fp16-stored moments: cancellation) as σ² relative to μ². */
function expectMoments(gpu: ArrayLike<number>, ref: ArrayLike<number>, skipFw = false): void {
  const strict = (a: ArrayLike<number>) => Array.from(a).filter((_, j) => (j & 3) !== 1 && (!skipFw || (j & 3) !== 3));
  expect(maxErr(strict(gpu), strict(ref), 1e-3).err).toBeLessThan(3e-3);
  for (let i = 0; i < ref.length / 4; i++) {
    const mu = ref[4 * i], sg = ref[4 * i + 1], sg2 = gpu[4 * i + 1];
    expect(Math.abs(sg2 * sg2 - sg * sg), `σ of pixel ${i}`).toBeLessThanOrEqual(4e-3 * (sg * sg + mu * mu) + 1e-7);
  }
}

interface FrameIn {
  cam: CameraState; prev: CameraState; px: RefPixel[]; radiance: Float64Array; l1: Float64Array; reset: boolean;
  settings: DenoiserSettings; kind: 'restir' | 'pt'; restir?: DenoiseFrame['restir']; advanced?: boolean; motion?: [number, number][];
}
let frameNo = 0;
function encodeFrame(f: FrameIn, record?: GPUComputePassDescriptor[]): void {
  rig.fu.write({
    camera: f.cam, prevCamera: f.prev, width: W, height: H, frameIndex: frameNo, seedIndex: frameNo, runSeed: 1, flags: 0, jitterMode: JITTER_NONE,
    jitter: [0.5, 0.5], origin: [0, 0, 0], exposure: 1, time: 0, dt: 0, sceneDiag: 10,
  });
  frameNo++;
  rig.device.queue.writeBuffer(rig.gbuf, 0, gbufBytes(f.px, f.motion));
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
  const alb4 = await f32(t.alb[cur]);
  const l14 = await f32(t.l1[cur]);
  const geo = new Uint32Array(await readTex(t.geo[cur]));
  const st = emptyState(W, H);
  for (let i = 0; i < P; i++) {
    st.mom.set(mom.slice(4 * i, 4 * i + 4), 4 * i);
    st.hist.set(hist4.slice(4 * i, 4 * i + 3), 3 * i);
    st.alb.set(alb4.slice(4 * i, 4 * i + 4), 4 * i);
    st.l1.set(l14.slice(4 * i, 4 * i + 3), 3 * i);
    st.dist[i] = new Float32Array(Uint32Array.of(geo[2 * i]).buffer)[0];
  }
  return { st, mom, atrous: [await f16(t.atrous[0]), await f16(t.atrous[1])], hist: hist4, colour: await f32(rig.colour) };
}

describe('denoiser passes vs the f64 reference', () => {
  const S0: DenoiserSettings = { ...DENOISER_DEFAULTS, iterations: 0, resolve: false };
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
    expectMoments(g.mom, ref.state.mom);
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
    // colour at fp16 precision; the variance (m₂ − μ² of fp16-stored moments: cancellation) relative to μ²
    const col = (a: ArrayLike<number>) => Array.from(a).filter((_, j) => (j & 3) !== 3);
    const e0 = maxErr(col(g.atrous[0]), col(ref.atrous0), 1e-3);
    expect(e0.err, `colour[${e0.at}]`).toBeLessThan(3e-3);
    for (let i = 0; i < P; i++) {
      const mu = ref.state.mom[4 * i];
      expect(Math.abs(g.atrous[0][4 * i + 3] - ref.atrous0[4 * i + 3]), `variance of pixel ${i}`).toBeLessThanOrEqual(4e-3 * (ref.atrous0[4 * i + 3] + mu * mu) + 1e-7);
    }
    expectMoments(g.mom, ref.state.mom);
    const codes = new Set(ref.code);
    expect(codes.has(DN_REPROJ.FULL) && codes.has(DN_REPROJ.PARTIAL) && codes.has(DN_REPROJ.BG)).toBe(true);
    // n = 2 everywhere a history was found
    let n2 = 0;
    for (let i = 0; i < P; i++) if (ref.code[i] === DN_REPROJ.FULL) { expect(g.mom[4 * i + 2]).toBe(2); n2++; }
    expect(n2).toBeGreaterThan(P / 2);
    prevRef = ref.state;
  });

  it('U-DN-3 dn_variance and dn_atrous (N = 2): feedback level and remodulated output', async () => {
    const S2: DenoiserSettings = { ...DENOISER_DEFAULTS, iterations: 2, resolve: false };
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

  // M8 P-6 (m8-perf.md §7): the lattice-tiled à-trous at steps 1, 2, 4, 8 (super-tiles 8 … 64 px on the 40 × 32 image:
  // partial super-tiles, residue classes, the image border) against the same f64 reference.
  it('U-DN-3b dn_atrous (N = 4, lattice tiles at steps 1 / 2 / 4 / 8) vs the f64 reference', async () => {
    const S4: DenoiserSettings = { ...DENOISER_DEFAULTS, iterations: 4, resolve: false };
    const cam = camAt(0.013);
    const px = scene(cam);
    const radiance = noisy(px, 5), l1 = l1Of(px);
    const g0 = await gpuState();
    const prevSt: RefState = { ...g0.st, n: prevRef.n };
    encodeFrame({ cam, prev: cam, px, radiance, l1, reset: false, settings: S4, kind: 'restir' });
    const ref = refTemporal({ W, H, px, cam, prevCam: cam, radiance, l1, flags: DNF.HAS_L1, settings: S4 }, prevSt);
    const fl = refFilter(S4, ref.state, ref.atrous0, px, radiance, l1);
    const g = await gpuState();
    const fb = Array.from({ length: P * 3 }, (_, j) => fl.feedback[4 * Math.floor(j / 3) + (j % 3)]);
    expect(maxErr(Array.from({ length: P * 3 }, (_, j) => g.hist[4 * Math.floor(j / 3) + (j % 3)]), fb, 1e-3).err).toBeLessThan(5e-3);
    expect(maxErr(g.colour.filter((_, j) => j % 4 !== 3), fl.colour, 1e-3).err).toBeLessThan(5e-3);
    prevRef = ref.state;
  });

  // M8 P-6: the tiled step-1 level is bitwise the texture-path dn_atrous (same arithmetic, same order): the frame above
  // (tiled) is re-encoded as a held frame (DN9: same inputs, same outputs) with dn_atrous at the step-1 level.
  it('U-DN-3c the tiled step-1 à-trous level ≡ dn_atrous bit for bit', async () => {
    const a = await gpuState();
    const dn = rig.dn as unknown as { plan: { name: string; pipeline: GPUComputePipeline }[]; pipelines: Map<string, GPUComputePipeline> };
    const lvl = dn.plan.find((x) => x.name === 'dn_atrous0')!;
    expect(lvl.pipeline).toBe(dn.pipelines.get('dn_atrous_tile1'));
    lvl.pipeline = dn.pipelines.get('dn_atrous')!;
    const enc = rig.device.createCommandEncoder();
    expect(rig.dn.encode(enc, { kind: 'restir', advanced: false, reset: false, frameUniforms: rig.fu.buffer, gbuf: rig.gbuf, radiance: rig.radiance, l1: rig.l1, colour: rig.colour, debugGroup: rig.debug.bindGroup })).toBe(true);
    rig.device.queue.submit([enc.finish()]);
    const b = await gpuState();
    lvl.pipeline = dn.pipelines.get('dn_atrous_tile1')!;
    expect(b.hist).toEqual(a.hist);
    expect(b.atrous).toEqual(a.atrous);
    expect(b.colour).toEqual(a.colour);
  });

  it('U-DN-4 dn_gradient / dn_grad_filter: tile sums, λ, and λ driving α', async () => {
    const S: DenoiserSettings = { ...DENOISER_DEFAULTS, iterations: 1, resolve: false };
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
    const radB = noisy(px, 5);
    const refL0 = refLambda(W, H, pairs, S.invRadius);
    // DN-11 colour family from the GPU's own history (frame A): cur = lum(L − L1), old = lum(hist·ā) at q′ (= q here)
    const [tx0, ty0] = dnTiles(W, H);
    const t2 = new Float64Array(tx0 * ty0 * 4);
    for (let i = 0; i < P; i++) {
      if (!(ts[i].flags & 1) || !(gA.st.alb[4 * i + 3] > 0)) continue;
      const cur = Math.max(lum([radB[3 * i] - l1[3 * i], radB[3 * i + 1] - l1[3 * i + 1], radB[3 * i + 2] - l1[3 * i + 2]]), 0);
      const old = lum([gA.hist[4 * i] * gA.st.alb[4 * i], gA.hist[4 * i + 1] * gA.st.alb[4 * i + 1], gA.hist[4 * i + 2] * gA.st.alb[4 * i + 2]]);
      const k = Math.floor(Math.floor(i / W) / 8) * tx0 + Math.floor((i % W) / 8);
      t2[4 * k] += cur; t2[4 * k + 1] += old; t2[4 * k + 2] += cur * cur; t2[4 * k + 3] += 1;
    }
    const lambdaC = (x: number, y: number) => {
      const sm = [0, 0, 0, 0];
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const cx = x + dx, cy = y + dy;
        if (cx < 0 || cy < 0 || cx >= tx0 || cy >= ty0) continue;
        for (let c = 0; c < 4; c++) sm[c] += t2[4 * (cy * tx0 + cx) + c];
      }
      if (sm[3] < 16) return 0;
      const mc = sm[0] / sm[3], mo = sm[1] / sm[3], se = Math.sqrt(Math.max(sm[2] / sm[3] - mc * mc, 0) / sm[3]);
      return Math.max(mc, mo) > 1e-8 ? Math.min(Math.max(Math.abs(mc - mo) - 3 * se, 0) / Math.max(mc, mo), 1) : 0;
    };
    const refL = { tiles: refL0.tiles, lambda: refL0.lambda.map((l, k) => Math.max(l, lambdaC(k % tx0, Math.floor(k / tx0)))) };
    const prevSt: RefState = { ...gA.st, n: prevRef.n };
    encodeFrame({ cam, prev: cam, px, radiance: radB, l1, reset: false, settings: S, kind: 'restir', restir: { ...restirA, gradient: true, lightingChanged: true } });
    const t = rig.dn.textures!;
    const tiles = await f32(t.gradTile);
    const lam = await f32(t.lambda);
    const [tx, ty] = dnTiles(W, H);
    for (let k = 0; k < tx * ty; k++) {
      for (const c of [0, 1, 2, 3]) expect(Math.abs(tiles[4 * k + c] - refL.tiles[4 * k + c])).toBeLessThan(1e-4 * (1 + Math.abs(refL.tiles[4 * k + (c | 1)])));
      expect(Math.abs(lam[k] - refL.lambda[k])).toBeLessThan(1e-3);
    }
    expect(Math.max(...lam)).toBeGreaterThan(0.3);
    // λ drives the temporal pass (α, n) exactly as in the reference
    const ref = refTemporal({ W, H, px, cam, prevCam: cam, radiance: radB, l1, flags: DNF.HAS_L1 | DNF.FW | DNF.GRADIENT | DNF.LAMBDA | DNF.INVERSE, settings: S, fw: Float64Array.from({ length: P }, (_, q) => fwGpu(q)), lambdaTiles: Float64Array.from(lam) }, prevSt);
    const g = await gpuState();
    expectMoments(g.mom, ref.state.mom, true);
    let reset = 0;
    for (let i = 0; i < P; i++) if (px[i].hit && ref.lambda[i] >= S.lambda1) { expect(g.mom[4 * i + 2]).toBe(1); reset++; }
    expect(reset).toBeGreaterThan(50);
    // gate closed (static lighting): λ is ignored
    encodeFrame({ cam, prev: cam, px, radiance: radB, l1, reset: false, settings: S, kind: 'restir', restir: { ...restirA, gradient: true, lightingChanged: false } });
    const g2 = await gpuState();
    for (let i = 0; i < P; i++) if (px[i].hit) expect(g2.mom[4 * i + 2]).toBeGreaterThan(1);
  });

  it('U-DN-6 dn_resolve: static camera, no lighting change for 8 frames ⇒ the progressive mean of the denoised outputs', async () => {
    const S: DenoiserSettings = { ...DENOISER_DEFAULTS, iterations: 1, resolve: true };
    const cam = camAt(0.05);
    const px = scene(cam);
    const l1 = l1Of(px);
    let prevTaa = new Float64Array(P * 4);
    for (let k = 0; k < 12; k++) {
      const radiance = noisy(px, 40 + k);
      encodeFrame({ cam, prev: k === 0 ? camAt(0.02) : cam, px, radiance, l1, reset: k === 0, settings: S, kind: 'restir' });
      const t = rig.dn.textures!;
      const taa = await f32(t.taa[rig.dn.parity]);
      if (k >= 8) {                                      // past the dynamic window after the reset (n_t ≤ 8 + clipping)
        const out = await f16(t.out);
        const out3 = Array.from({ length: P * 3 }, (_, j) => out[4 * Math.floor(j / 3) + (j % 3)]);
        const ref = refResolveStatic(S, out3, prevTaa, false, P);
        expect(maxErr(taa, ref, 1e-3).err).toBeLessThan(3e-3);
        const colour = await f32(rig.colour);
        expect(maxErr(colour.filter((_, j) => j % 4 !== 3), Array.from(ref).filter((_, j) => j % 4 !== 3), 1e-3).err).toBeLessThan(1e-3);
      }
      prevTaa = Float64Array.from(taa);
    }
    // every pixel (background too) accumulates: n_t = 8 + 4 after the 4 static frames
    for (let i = 0; i < P; i++) expect(prevTaa[4 * i + 3]).toBe(12);
  });

  it('U-DN-7 dn_resolve in motion: closest-hit (3×3) motion vector and Catmull-Rom history fetch (Changelog DN-17)', async () => {
    // γ = 0 (no clipping) and n_t ≤ 2: the resolved value is ½(history + current), so the fetched history is 2·T − out.
    const S: DenoiserSettings = { ...DENOISER_DEFAULTS, iterations: 1, taaCamMax: 2, taaGammaCam: 0, taaDilate: true, taaCubic: true };
    const camA = camAt(0.02), camB = camAt(0.05);
    const pxA = scene(camA), pxB = scene(camB);
    const l1A = l1Of(pxA), l1B = l1Of(pxB);
    encodeFrame({ cam: camA, prev: camA, px: pxA, radiance: noisy(pxA, 70), l1: l1A, reset: true, settings: S, kind: 'restir' });
    const t = rig.dn.textures!;
    const taaA = await f32(t.taa[rig.dn.parity]);
    // plane (z = −2) and box face (z = −1.4) move differently; background by its own vector
    const motion = pxB.map((p, i): [number, number] => (!p.hit ? [0.1, 0] : (i % W) >= 24 && (i % W) < 32 ? [1.7, -0.4] : [0.3, 0.2]));
    encodeFrame({ cam: camB, prev: camA, px: pxB, radiance: noisy(pxB, 71), l1: l1B, reset: false, settings: S, kind: 'restir', motion });
    const taaB = await f32(t.taa[rig.dn.parity]);
    const out = await f16(t.out);
    const camPos = [camB.camToWorld[12], camB.camToWorld[13], camB.camToWorld[14]];
    const dist = pxB.map((p) => (p.hit ? Math.fround(Math.hypot(p.pos[0] - camPos[0], p.pos[1] - camPos[1], p.pos[2] - camPos[2])) : 0));
    const at = (x: number, y: number, k: number) => taaA[4 * (Math.min(Math.max(y, 0), H - 1) * W + Math.min(Math.max(x, 0), W - 1)) + k];
    const cr = (f: number): number[] => [f * (-0.5 + f * (1 - 0.5 * f)), 1 + f * f * (-2.5 + 1.5 * f), f * (0.5 + f * (2 - 1.5 * f)), f * f * (-0.5 + 0.5 * f)];
    let checked = 0, dilated = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let best = 1e30, mi = y * W + x;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const qx = x + dx, qy = y + dy;
        if (qx < 0 || qy < 0 || qx >= W || qy >= H) continue;
        const d = dist[qy * W + qx];
        if (d > 0 && d < best) { best = d; mi = qy * W + qx; }
      }
      if (motion[mi] !== motion[y * W + x]) dilated++;
      const sx = x + motion[mi][0], sy = y + motion[mi][1], bx = Math.floor(sx), by = Math.floor(sy), fx = sx - bx, fy = sy - by;
      if (bx < 0 || by < 0 || bx + 1 >= W || by + 1 >= H) continue;   // the bilinear fallback at the border: not modelled
      const wx = cr(fx), wy = cr(fy);
      for (let k = 0; k < 3; k++) {
        let h = 0;
        for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) h += wx[i] * wy[j] * at(bx + i - 1, by + j - 1, k);
        h = Math.max(h, 0);
        const i4 = 4 * (y * W + x);
        expect(taaB[i4 + 3]).toBe(2);
        const got = 2 * taaB[i4 + k] - out[i4 + k];
        expect(Math.abs(got - h), `pixel (${x}, ${y}) channel ${k}`).toBeLessThanOrEqual(2e-3 * (Math.abs(h) + Math.abs(out[i4 + k])) + 1e-5);
      }
      checked++;
    }
    expect(checked).toBeGreaterThan(P / 2);
    expect(dilated).toBeGreaterThan(20);   // the box's motion reaches the plane and background pixels next to it
  });

  it('U-DN-5 held frame and timing re-run are bit-identical; no timestamp writes in the frame', async () => {
    const S: DenoiserSettings = { ...DENOISER_DEFAULTS, resolve: false };
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
