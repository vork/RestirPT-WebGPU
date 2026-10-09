// perf2 WP-7 (docs/decisions/perf2-plan.md §2 WP-7, perf2-api.md): the frame / primary / denoiser items behind bitwise
// perf flags reproduce the flag-free app frame bit for bit.
//   U-WP7-APP  the interactive Renderer (ReSTIR-interactive, light mode B, MT, denoiser on, accumulation on: the app's
//              configuration) on the all-lights fixture, once without flags and once with the WP-7 set: per frame the
//              colour target, the denoiser's current histories (dnHist, dnMom, dnTaa, dnAlb, dnL1) and, on the
//              pairs-view frames, the debug AOV are identical. The schedule covers 18 static frames (gradient gated:
//              DN_GRAD_SKIP skips it), a light change (λ read), camera motion, held frames (DN9), view 527 switched on
//              while held and advanced, the denoiser toggled off and on (RS_SKIP_DISPLAY restarts the ReSTIR accumulation:
//              the flag-free run gets an explicit FRAME_RESET_HISTORY on that frame, so the restart is shown to be the only
//              difference), and gradientOnCamera with camera motion (λ read on camera-only frames). The stored G-buffer
//              (GBUF_48) equals the 80 B texel's pos / flags / ns / motion / albedo.
//   U-WP7-DUP  rs_dupmap with RS_DUPMAP_S64 on a planted seed image (few distinct seeds: collisions everywhere, empty and
//              background reservoirs, the sentinel seed (2³²−1, 2³²−1) on hits, so the fallback runs): the count region
//              equals the flag-free kernel's and an f64 CPU count.
import { afterAll, describe, expect, it } from 'vitest';
import { getTestGpu, releaseTestGpu } from './device-factory.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { DEBUG_BUFFER_LAYOUT, DebugResources, DebugViewRegistry, defaultDebugSettings } from '../../src/core/render/debug-views.ts';
import { FRAME_RESET_HISTORY, FrameUniformBuffer, JITTER_IID, boundsDiagonal, computeRenderOrigin, type CameraState } from '../../src/core/render/frame-uniforms.ts';
import { GBUF48_TEXEL_BYTES, GBUF_TEXEL_BYTES, Renderer } from '../../src/core/render/renderer.ts';
import { DENOISER_DEFAULTS, DN_VIEW } from '../../src/core/render/denoise/layout.ts';
import { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import { restirSettings } from '../../src/core/render/restir/presets.ts';
import { RS_M6_CONSTS as K6 } from '../../src/core/render/restir/layout.ts';
import type { PerfFlagsInput } from '../../src/core/render/restir/perf-flags.ts';
import type { LightData } from '../../src/core/scene/types.ts';
import { allLightsScene, bitFixtureScene, boxCamera, gpuScene, hashF32 } from './restir-fixtures.ts';

afterAll(async () => { await releaseTestGpu(); });

const WP7_FLAGS = 'DN_GRAD_SKIP,PRIM_SKIP_BEAUTY,RS_SKIP_DISPLAY,GBUF_48,DN_ZGRAD_TEX,DN_COLOUR_EARLY,RS_DUPMAP_S64';
const W = 96, H = 64;

const hashBytes = (b: ArrayBuffer): string => hashF32(new Float32Array(b.slice(0, b.byteLength & ~3)));

async function readTex(device: GPUDevice, t: GPUTexture): Promise<ArrayBuffer> {
  const bpp = { rgba32float: 16, rgba16float: 8, rg32uint: 8, rgba32uint: 16, r32float: 4, rg32float: 8 }[t.format as string] ?? 4;
  const bpr = Math.ceil((t.width * bpp) / 256) * 256;
  const b = device.createBuffer({ size: bpr * t.height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const e = device.createCommandEncoder();
  e.copyTextureToBuffer({ texture: t }, { buffer: b, bytesPerRow: bpr }, [t.width, t.height]);
  device.queue.submit([e.finish()]);
  await b.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(b.getMappedRange());
  const out = new Uint8Array(t.width * t.height * bpp);
  for (let y = 0; y < t.height; y++) out.set(src.subarray(y * bpr, y * bpr + t.width * bpp), y * t.width * bpp);
  b.unmap(); b.destroy();
  return out.buffer;
}

/** One step of the schedule. */
interface Step {
  advanced: boolean; cam: number; lightDx?: number; debugMode?: number; denoise?: boolean; gradientOnCamera?: boolean;
  /** Only the flag-free run: FRAME_RESET_HISTORY (the restart RS_SKIP_DISPLAY performs on the first undenoised frame). */
  baseRestart?: boolean; reset?: boolean;
}
const SCHEDULE: Step[] = [
  { advanced: true, cam: 0, reset: true },
  ...Array.from({ length: 17 }, (): Step => ({ advanced: true, cam: 0 })),                  // static: gradient gated
  { advanced: true, cam: 0, lightDx: 0.06 },                                                 // light change: λ read
  { advanced: true, cam: 0, lightDx: 0.06 },
  { advanced: true, cam: 1, lightDx: 0.06 }, { advanced: true, cam: 2, lightDx: 0.06 }, { advanced: true, cam: 3, lightDx: 0.06 },
  { advanced: false, cam: 3, lightDx: 0.06 },                                                // held (DN9)
  { advanced: false, cam: 3, lightDx: 0.06, debugMode: DN_VIEW.pairs },                      // view 527 on while held
  { advanced: true, cam: 3, lightDx: 0.06, debugMode: DN_VIEW.pairs },                       // view 527, static, advanced
  { advanced: true, cam: 3, lightDx: 0.06, denoise: false, baseRestart: true },              // denoiser off: restart
  { advanced: true, cam: 3, lightDx: 0.06, denoise: false },
  { advanced: false, cam: 3, lightDx: 0.06, denoise: false },
  { advanced: true, cam: 3, lightDx: 0.06 },                                                 // denoiser on again
  { advanced: true, cam: 4, lightDx: 0.06, gradientOnCamera: true },                         // λ on camera-only frames
  { advanced: true, cam: 5, lightDx: 0.06, gradientOnCamera: true },
  { advanced: true, cam: 5, lightDx: 0.06, gradientOnCamera: true },
];

interface FrameHashes { colour: string; dn: string; dbg: string; gbuf: string; passes: string }

async function runApp(flags: PerfFlagsInput, base: boolean): Promise<FrameHashes[]> {
  const ctx = await getTestGpu();
  const { device } = ctx;
  const scene = allLightsScene();
  const debug = new DebugResources(device, new DebugViewRegistry());
  await debug.init();
  debug.resize(W, H);
  const r = await Renderer.create({ device, debugLayout: debug.layout, debug, features: ctx.features, wgslLanguageFeatures: ctx.wgslLanguageFeatures }, {
    watertight: false, renderMode: 'restir', restirMode: 'interactive', temporal: true, accumulate: true, lightMode: 'B', bvhKind: 'auto', maxBounces: 3,
    restirKernel: { perfFlags: flags },
  });
  const origin = computeRenderOrigin(scene.bounds, scene.quant);
  const fu = new FrameUniformBuffer(device);
  const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
  const color = device.createTexture({ label: 'wp7-colour', size: [W, H], format: 'rgba16float', usage });
  const depth = device.createTexture({ label: 'wp7-depth', size: [W, H], format: 'r32float', usage });
  const out: FrameHashes[] = [];
  try {
    if (scene.env) await r.setEnvironment(scene.env);
    expect(await r.setScene(scene, origin)).toBeTruthy();
    r.resize({ width: W, height: H, color, colorFormat: 'rgba16float', depth, frameUniforms: fu.buffer });
    r.setDenoise(true);
    r.setDenoiserSettings({ ...DENOISER_DEFAULTS });
    expect(await r.prepareRestir()).toBeTruthy();
    expect(await r.prepareDenoiser()).toBeTruthy();
    await r.warmup(false);
    const lights0 = scene.lights.map((l) => ({ ...l, matrix: new Float32Array(l.matrix) }));
    const sceneDiag = boundsDiagonal(scene.bounds);
    const camOf = (k: number): CameraState => { const c = boxCamera(); c.camToWorld[12] += 0.02 * k; c.camToWorld[13] += 0.01 * k; return { ...c, znear: 1e-4 }; };
    let prev: CameraState | undefined;
    let frame = 0;
    for (const [i, s] of SCHEDULE.entries()) {
      const cam = camOf(s.cam);
      r.setLights(lights0.map((l, k) => (k === 0 && s.lightDx ? { ...l, matrix: Object.assign(new Float32Array(l.matrix), { 12: l.matrix[12] + s.lightDx }) } : l)) as LightData[]);
      if (s.advanced) frame++;
      fu.write({
        camera: cam, prevCamera: s.advanced ? (prev ?? cam) : cam, width: W, height: H, frameIndex: frame, seedIndex: frame, runSeed: 7,
        flags: (base && s.baseRestart) || s.reset ? FRAME_RESET_HISTORY : 0, jitterMode: JITTER_IID, jitter: [0.5, 0.5], origin, exposure: 1,
        time: frame / 24, dt: 1 / 24, sceneDiag,
      });
      if (s.advanced) prev = cam;
      r.setDenoise(s.denoise ?? true);
      r.setDenoiserSettings({ gradientOnCamera: !!s.gradientOnCamera });
      const mode = s.debugMode ?? 0;
      debug.update({ ...defaultDebugSettings(), mode }, frame);
      const enc = device.createCommandEncoder({ label: `wp7-${i}` });
      debug.beginFrame(enc);
      expect(r.encode(enc, { advanced: s.advanced, debugMode: mode, debugGroup: debug.bindGroup, resetTemporal: !!s.reset, resetHistory: !!s.reset }), `frame ${i}`).toBe(true);
      expect(r.denoisedLastFrame, `frame ${i} denoised`).toBe(s.denoise ?? true);
      device.queue.submit([enc.finish()]);
      const d = r.denoiser!;
      const t = d.textures!;
      const c = d.parity;
      const dn = s.denoise ?? true
        ? (await Promise.all([t.hist[c], t.mom[c], t.taa[c], t.alb[c], t.l1[c]].map(async (x) => hashBytes(await readTex(device, x))))).join(',')
        : '-';
      const dbg = mode ? hashBytes(await readBuffer(device, debug.buffer, DEBUG_BUFFER_LAYOUT.aovOffset + W * H * DEBUG_BUFFER_LAYOUT.aovStride)) : '-';
      // the G-buffer projected onto the 48 B fields (pos, flags, ns, motion.x, albedo, motion.y)
      const gb = new Uint32Array(await readBuffer(device, r.gbuffer!, r.gbuffer!.size));
      const stride = gb.byteLength / (W * H) / 4;
      expect(stride * 4).toBe(base ? GBUF_TEXEL_BYTES : GBUF48_TEXEL_BYTES);
      const proj = new Uint32Array(W * H * 12);
      for (let p = 0; p < W * H; p++) {
        const o = p * stride, q = p * 12;
        if (stride === 20) {   // 80 B: ng thr | ns viewZ | pos matId | albedo flags | motion pad
          proj.set(gb.subarray(o + 8, o + 11), q); proj[q + 3] = gb[o + 15]; proj.set(gb.subarray(o + 4, o + 7), q + 4); proj[q + 7] = gb[o + 16];
          proj.set(gb.subarray(o + 12, o + 15), q + 8); proj[q + 11] = gb[o + 17];
        } else proj.set(gb.subarray(o, o + 12), q);
      }
      out.push({
        colour: hashBytes(await readTex(device, color)), dn, dbg, gbuf: hashF32(new Float32Array(proj.buffer)),
        passes: d.passNames.join(' '),
      });
    }
  } finally {
    r.destroy(); debug.destroy(); fu.destroy(); color.destroy(); depth.destroy();
  }
  return out;
}

describe('U-WP7-APP: the WP-7 bitwise flags reproduce the flag-free app frame (perf2-plan.md WP-7 a–d, f)', () => {
  it(`colour target, denoiser histories, pairs view and G-buffer fields bit-identical with ${WP7_FLAGS}`, async () => {
    const a = await runApp({}, true);
    const b = await runApp(WP7_FLAGS, false);
    for (let i = 0; i < SCHEDULE.length; i++) {
      console.log(`[U-WP7-APP] frame ${i}: base [${a[i].passes}] flags [${b[i].passes}]`);
      expect(b[i].colour, `colour ${i}`).toBe(a[i].colour);
      expect(b[i].dn, `denoiser histories ${i}`).toBe(a[i].dn);
      expect(b[i].dbg, `debug AOV ${i}`).toBe(a[i].dbg);
      expect(b[i].gbuf, `G-buffer ${i}`).toBe(a[i].gbuf);
    }
    // non-trivial frames: the colour and the histories change over the schedule
    expect(new Set(a.map((x) => x.colour)).size).toBeGreaterThan(SCHEDULE.length / 2);
    expect(new Set(a.map((x) => x.dn)).size).toBeGreaterThan(SCHEDULE.length / 2);
    // the skip happened on the static frames and did not on the λ / view-527 / gradientOnCamera frames
    expect(b[10].passes).not.toContain('dn_gradient');
    expect(a[10].passes).toContain('dn_gradient');
    expect(b[18].passes).toContain('dn_gradient');
    expect(b[25].passes).toContain('dn_gradient');
    expect(b[30].passes).toContain('dn_gradient');
    // the run is deterministic (Q2)
    const b2 = await runApp(WP7_FLAGS, false);
    expect(b2.map((x) => x.colour + x.dn)).toEqual(b.map((x) => x.colour + x.dn));
  });
});

describe('U-WP7-DUP: rs_dupmap with RS_DUPMAP_S64 on a planted seed image (perf2-plan.md WP-7b)', () => {
  const DW = 61, DH = 45;   // not a multiple of 16: partial workgroups and out-of-atlas tile texels
  async function dupCounts(flags: PerfFlagsInput, seeds: Uint32Array): Promise<Uint32Array> {
    const g = await gpuScene(bitFixtureScene('x_quads'));
    const device = g.device;
    const k = await RestirKernel.create(device, g.gpu, g.env, {
      settings: restirSettings('full', { dupmap: true }), features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures, perfFlags: flags,
    });
    k.setView({ camera: boxCamera(), width: DW, height: DH, runSeed: 3, jitterMode: JITTER_IID });
    await k.prepare();
    k.advance({ t: 0, camera: boxCamera(), lights: g.gpu.scene.lights });
    const res = k.resources;
    const fin = k.finalResIndex();
    device.queue.writeBuffer(res.res[fin], 0, seeds);
    const enc = device.createCommandEncoder();
    k.beginSubmit();
    k.encodePass(enc, 'rs_dupmap', k.pipelineSync('rs_dupmap'), res.g2('rs_dupmap', fin), { t: 0, passId: K6.RS_PASS_DUPMAP }, [Math.ceil(DW / 16), Math.ceil(DH / 16)]);
    device.queue.submit([enc.finish()]);
    const arena = new Uint32Array(await readBuffer(device, res.arena, res.arena.size));
    k.destroy();
    return arena;
  }

  it('count region ≡ the flag-free kernel ≡ an f64 CPU count (collisions, empty, background, sentinel seeds)', async () => {
    const P = DW * DH;
    const recs = new Uint32Array(P * 40);   // AoS: 10 planes × vec4u; plane 1 = (seed.x, seed.y, flags, ·)
    let s = 12345;
    const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return s / 2 ** 32; };
    const SEEDS: [number, number][] = [[1, 2], [1, 3], [7, 2], [0xFFFFFFFF, 0xFFFFFFFF], [0, 0], [0xFFFFFFFF, 0]];
    for (let i = 0; i < P; i++) {
      const u = rnd();
      const sd = SEEDS[Math.floor(rnd() * SEEDS.length)];
      const flagsWord = u < 0.1 ? 0 : u < 0.18 ? (0x1 | 0x2000000) : 0x1;   // empty | background | a sample
      recs.set([sd[0], sd[1], flagsWord, 0], i * 40 + 4);
    }
    const off = await dupCounts({}, recs);
    const on = await dupCounts('RS_DUPMAP_S64', recs);
    expect(hashF32(new Float32Array(on.buffer))).toBe(hashF32(new Float32Array(off.buffer)));   // the whole arena
    // f64 CPU count (restir-m6-api.md MD10): locate the count region by matching the CPU counts
    const ok = (i: number) => (recs[i * 40 + 6] & 0xF) !== 0 && (recs[i * 40 + 6] & 0x2000000) === 0;
    const cpu = new Uint32Array(P);
    for (let y = 0; y < DH; y++) for (let x = 0; x < DW; x++) {
      const i = y * DW + x;
      if (!ok(i)) continue;
      for (let dy = -8; dy <= 8; dy++) for (let dx = -8; dx <= 8; dx++) {
        if (!dx && !dy) continue;
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= DW || yy >= DH) continue;
        const j = yy * DW + xx;
        if (ok(j) && recs[j * 40 + 4] === recs[i * 40 + 4] && recs[j * 40 + 5] === recs[i * 40 + 5]) cpu[i]++;
      }
    }
    let found = -1;
    for (let b = 0; b + P <= on.length && found < 0; b++) {
      let eq = true;
      for (let i = 0; i < P && eq; i++) eq = on[b + i] === cpu[i];
      if (eq) found = b;
    }
    expect(found, 'the CPU counts appear in the arena').toBeGreaterThan(0);
    const sentinelHits = Array.from({ length: P }, (_, i) => i).filter((i) => ok(i) && recs[i * 40 + 4] === 0xFFFFFFFF && recs[i * 40 + 5] === 0xFFFFFFFF).length;
    console.log(`[U-WP7-DUP] count region at word ${found}; Σcount ${cpu.reduce((a, x) => a + x, 0)}; sentinel-seed hits ${sentinelHits}`);
    expect(sentinelHits).toBeGreaterThan(0);
  });
});
