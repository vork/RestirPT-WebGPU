// Browser side of the M5.5 denoiser evaluation (docs/decisions/denoiser.md §11). NOT a validation readback: it drives
// the interactive Renderer (ReSTIR-interactive with the denoiser on) on a scene package, frame by frame, and uploads
// to validation/out/<run>/:
//   mode 'flip'      raw_f<t>.pfm (rsFrame: the 1-frame ReSTIR output, no accumulation) and dn_f<t>.pfm (the denoised
//                    colour target of the SAME frame) at the evaluation frames
//   mode 'recovery'  tiles.bin: per frame, 8×8-tile means of the luminance of the denoised and of the raw output
//                    (f32 [frame][2][tile]; frames follow `pkgFrames`, a package-frame schedule holding states)
//   mode 'timing'    the denoiser GPU time from separate timing submits (Q3) after `warmup` frames, at any resolution
// plus meta.json (settings, per-frame denoiser flags, finalize counters, adapter). The renderer (and its denoiser) is
// destroyed at the end, so a validation run on the same page afterwards sees no live denoiser (T16, harness.ts).
import { describeContext, type GpuContext } from '../../src/core/gpu/device.ts';
import { encodePFM } from '../../src/core/io/pfm.ts';
import { DebugResources, DebugViewRegistry } from '../../src/core/render/debug-views.ts';
import { DENOISER_DEFAULTS, type DenoiserSettings } from '../../src/core/render/denoise/layout.ts';
import { FrameUniformBuffer, JITTER_IID, JITTER_NONE, JITTER_R2, boundsDiagonal, computeRenderOrigin, r2Jitter, type CameraState } from '../../src/core/render/frame-uniforms.ts';
import { Renderer, type RestirAppMode } from '../../src/core/render/renderer.ts';
import { fetchScenePackage, resolvePackageFrame } from '../../src/core/scene/scene-package.ts';
import { packageSha256, uploadFile } from './export-package.ts';

export interface RenderDenoiseOptions {
  run: string;
  package: string;
  seed: number;
  mode: 'flip' | 'recovery' | 'timing';
  /** Frames to render (from a reset). */
  frames: number;
  /** Run frame → package frame (sequences); default: frame i (sequence packages) or the base state (static). */
  pkgFrames?: number[];
  /** 'flip': frames whose raw and denoised images are uploaded. */
  evalFrames?: number[];
  /** Resolution override (timing at 960×540). */
  width?: number;
  height?: number;
  restirMode?: RestirAppMode;
  denoiser?: Partial<DenoiserSettings>;
  /** Primary jitter (default 'iid'; the interactive app offers R2 and pixel centre). */
  jitter?: 'iid' | 'r2' | 'none';
  /** Denoiser on (default) or off; `accumulate` turns the progressive mean on (the app's denoiser-off display). */
  denoise?: boolean;
  accumulate?: boolean;
  /** Slow camera motion: translate along the camera's right axis by `pan` metres per frame on frames [from, to). */
  pan?: { dx: number; from: number; to: number };
  /** 'timing': warm-up frames, timing submits and re-runs per submit. */
  warmup?: number;
  timingSubmits?: number;
  timingRuns?: number;
  chromeVersion?: string;
}
export interface RenderDenoiseReport { ok: boolean; run: string; files: string[]; meta: Record<string, unknown>; errors: string[] }

const TILE = 8;
const TILE_WGSL = /* wgsl */ `
struct P { size: vec2u, tiles: vec2u, base: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var dnTex: texture_2d<f32>;
@group(0) @binding(1) var rawTex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outBuf: array<f32>;
@group(0) @binding(3) var<uniform> prm: P;
@compute @workgroup_size(8, 8, 1)
fn tiles(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid.xy >= prm.tiles)) { return; }
  var a = 0.0; var b = 0.0; var n = 0.0;
  for (var y = 0u; y < ${TILE}u; y++) {
    for (var x = 0u; x < ${TILE}u; x++) {
      let p = gid.xy * ${TILE}u + vec2u(x, y);
      if (any(p >= prm.size)) { continue; }
      let w = vec3f(0.2126, 0.7152, 0.0722);
      a += dot(textureLoad(dnTex, p, 0).rgb, w);
      b += dot(textureLoad(rawTex, p, 0).rgb, w);
      n += 1.0;
    }
  }
  let i = prm.base + 2u * (gid.y * prm.tiles.x + gid.x);
  outBuf[i] = a / n;
  outBuf[i + 1u] = b / n;
}`;

async function readTexture(device: GPUDevice, tex: GPUTexture): Promise<Float32Array> {
  const u = await readTexture4(device, tex);
  const f = new Float32Array(u.buffer);
  const out = new Float32Array(tex.width * tex.height * 3);
  for (let k = 0; k < tex.width * tex.height; k++) for (let c = 0; c < 3; c++) out[3 * k + c] = f[4 * k + c];
  return out;
}
/** The 4 × 32-bit texels of an rgba32float / rgba32uint texture, bit-exact (row padding removed). */
async function readTexture4(device: GPUDevice, tex: GPUTexture): Promise<Uint32Array> {
  const W = tex.width, H = tex.height, bpr = Math.ceil((W * 16) / 256) * 256;
  const buf = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const e = device.createCommandEncoder({ label: 'dn-readback' });
  e.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [W, H]);
  device.queue.submit([e.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const src = new Uint32Array(buf.getMappedRange().slice(0));
  buf.unmap(); buf.destroy();
  const out = new Uint32Array(W * H * 4);
  for (let y = 0; y < H; y++) out.set(src.subarray((y * bpr) / 4, (y * bpr) / 4 + W * 4), y * W * 4);
  return out;
}

export async function renderDenoise(ctx: GpuContext, o: RenderDenoiseOptions): Promise<RenderDenoiseReport> {
  const t0 = performance.now();
  const errors: string[] = [];
  const files: string[] = [];
  const { device } = ctx;
  const p = await fetchScenePackage(o.package);
  const hash = await packageSha256(p.files);
  const W = o.width ?? p.render.width, H = o.height ?? p.render.height;
  const seqFrames = p.sequence?.frameCount;
  const pkgFrame = (i: number): number => o.pkgFrames?.[i] ?? (seqFrames ? Math.min(i, seqFrames - 1) : -1);
  if ((p.lightMode ?? 'A') !== 'A') throw new Error(`${o.package}: light mode ${p.lightMode}; the interactive ReSTIR is Mode A only`);

  const debug = new DebugResources(device, new DebugViewRegistry());
  await debug.init();
  debug.resize(W, H);
  const settings0 = { ...DENOISER_DEFAULTS, ...o.denoiser };
  const r = await Renderer.create({ device, debugLayout: debug.layout, debug, features: ctx.features, wgslLanguageFeatures: ctx.wgslLanguageFeatures }, {
    textureMode: 'validation', watertight: true, renderMode: 'restir', restirMode: o.restirMode ?? 'interactive', temporal: true, accumulate: !!o.accumulate,
    maxBounces: p.render.maxBounces ?? 3, lightMode: 'A', envNee: (p.json.env?.sampling ?? 'AUTOMATIC') !== 'NONE',
  });
  const origin = computeRenderOrigin(p.scene.bounds, p.scene.quant);
  const fu = new FrameUniformBuffer(device);
  const color = device.createTexture({ label: 'dn-eval-colour', size: [W, H], format: 'rgba32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
  const depth = device.createTexture({ label: 'dn-eval-depth', size: [W, H], format: 'r32float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
  const tilesX = Math.ceil(W / TILE), tilesY = Math.ceil(H / TILE), nTiles = tilesX * tilesY;
  let tileBuf: GPUBuffer | undefined, tilePipe: GPUComputePipeline | undefined, tileParams: GPUBuffer | undefined;
  const perFrame: { flags: number; lambdaGate: boolean; reset: boolean }[] = [];
  let timing: Record<string, unknown> | undefined;
  try {
    if (p.scene.env) await r.setEnvironment(p.scene.env);
    const g = await r.setScene(p.scene, origin);
    if (!g) throw new Error('setScene superseded');
    r.resize({ width: W, height: H, color, colorFormat: 'rgba32float', depth, frameUniforms: fu.buffer });
    const denoise = o.denoise ?? true;
    r.setDenoise(denoise);
    r.setDenoiserSettings(settings0);
    if (!await r.prepareRestir()) throw new Error(`ReSTIR compile failed: ${r.restirError}`);
    if (denoise && !await r.prepareDenoiser()) throw new Error(`denoiser compile failed: ${r.denoiserError}`);
    await r.warmup(false);
    const sceneDiag = boundsDiagonal(p.scene.bounds);
    if (o.mode === 'recovery') {
      tileBuf = device.createBuffer({ label: 'dn-eval-tiles', size: o.frames * nTiles * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      tileParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const module = device.createShaderModule({ code: TILE_WGSL });
      tilePipe = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'tiles' } });
    }
    let prev: CameraState | undefined;
    const jm = o.jitter === 'r2' ? JITTER_R2 : o.jitter === 'none' ? JITTER_NONE : JITTER_IID;
    let vbufMismatch = -1, vbufMaxBary = 0;
    const total = o.mode === 'timing' ? (o.warmup ?? 32) : o.frames;
    for (let i = 0; i < total; i++) {
      const st = resolvePackageFrame(p, pkgFrame(i));
      const cam: CameraState = { camToWorld: Array.from(st.camera.camToWorld), yfov: st.camera.yfov, znear: 1e-4 };
      if (o.pan) {   // slow pan along the camera's right axis (column 0)
        const k = Math.min(Math.max(i, o.pan.from), o.pan.to) - o.pan.from;
        for (let a = 0; a < 3; a++) cam.camToWorld[12 + a] += k * o.pan.dx * cam.camToWorld[a];
      }
      r.setLights(st.lights);
      if (st.env) r.setEnvParams(st.env.params);
      fu.write({
        camera: cam, prevCamera: prev ?? cam, width: W, height: H, frameIndex: i, seedIndex: i, runSeed: o.seed >>> 0, flags: 0,
        jitterMode: jm, jitter: o.jitter === 'r2' ? r2Jitter(i, o.seed >>> 0) : [0.5, 0.5], origin, exposure: 1, time: i / 24, dt: 1 / 24, sceneDiag,
      });
      prev = cam;
      debug.update({ ...debug.settings, mode: 0 }, i);
      const enc = device.createCommandEncoder({ label: `dn-eval-${i}` });
      debug.beginFrame(enc);
      const ok = r.encode(enc, { advanced: true, debugMode: 0, debugGroup: debug.bindGroup, resetTemporal: i === 0, resetHistory: i === 0 });
      if (!ok || r.denoisedLastFrame !== denoise) throw new Error(`frame ${i}: renderer not ready (encode ${ok}, denoised ${r.denoisedLastFrame}; ${r.restirError ?? ''} ${r.denoiserError ?? ''} ${r.lastError ?? ''})`);
      const d = r.denoiser;
      if (d && denoise) perFrame.push({ flags: d.flags, lambdaGate: (d.flags & 2) !== 0, reset: (d.flags & 1) !== 0 });
      const rs = r.restir!.kernel.resources;
      if (o.mode === 'recovery') {
        device.queue.writeBuffer(tileParams!, 0, new Uint32Array([W, H, tilesX, tilesY, i * nTiles * 2, 0, 0, 0]));
        const bg = device.createBindGroup({ layout: tilePipe!.getBindGroupLayout(0), entries: [
          { binding: 0, resource: color.createView() }, { binding: 1, resource: rs.frameTex.createView() }, { binding: 2, resource: { buffer: tileBuf! } }, { binding: 3, resource: { buffer: tileParams! } }] });
        const pass = enc.beginComputePass({ label: 'dn-eval-tiles' });
        pass.setPipeline(tilePipe!); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(tilesX / 8), Math.ceil(tilesY / 8)); pass.end();
      }
      device.queue.submit([enc.finish()]);
      if (i === 1) {
        // the denoiser's albedo is the M1 G-buffer's: the same jittered primary hit as rs_primary (V-buffer ids equal)
        const a = await readTexture4(device, r.vbuffer!), b = await readTexture4(device, rs.vbuf);
        // primId must match exactly; the f32 barycentrics of the two pipelines may differ by contraction (FMA) ulps
        vbufMismatch = 0;
        const fa = new Float32Array(a.buffer), fb = new Float32Array(b.buffer);
        for (let k = 0; k < W * H; k++) {
          if (a[4 * k] !== b[4 * k]) { vbufMismatch++; continue; }
          if (a[4 * k] !== 0xFFFFFFFF) vbufMaxBary = Math.max(vbufMaxBary, Math.abs(fa[4 * k + 1] - fb[4 * k + 1]), Math.abs(fa[4 * k + 2] - fb[4 * k + 2]));
        }
        // two separately compiled intersection pipelines: FMA contraction moves barycentrics by ulps of the edge functions
        // (≤ 1e-3 on large triangles) and can flip an exact edge tie to the neighbouring triangle (a handful of pixels)
        if (vbufMismatch > 1e-4 * W * H || vbufMaxBary > 1e-2) errors.push(`M1 V-buffer vs rsVbuf: ${vbufMismatch} primId mismatches, max |Δbary| ${vbufMaxBary} (denoiser albedo from another sample)`);
      }
      if (o.mode === 'flip' && o.evalFrames?.includes(i)) {
        const dn = await readTexture(device, color);
        const raw = await readTexture(device, rs.frameTex);
        for (const [name, img] of [[`dn_f${i}.pfm`, dn], [`raw_f${i}.pfm`, raw]] as const) {
          let bad = 0;
          for (const x of img) if (!Number.isFinite(x)) bad++;
          if (bad) errors.push(`${name}: ${bad} non-finite values`);
          await uploadFile(o.run, name, encodePFM({ width: W, height: H, channels: 3, data: img }));
          files.push(name);
        }
      }
      if (i % 16 === 15) await device.queue.onSubmittedWorkDone();
    }
    if (o.mode === 'timing') {
      // The last frame's denoiser passes, re-run in separate submits with timestamp writes (DN9, Q3).
      const runs: { totalMs: number; passes: { name: string; ms: number }[] }[] = [];
      for (let k = 0; k < (o.timingSubmits ?? 8); k++) {
        await device.queue.onSubmittedWorkDone();
        const t = await r.denoiser!.time(o.timingRuns ?? 8);
        if (t) runs.push(t);
      }
      if (!runs.length) errors.push('timing: no timestamp-query support');
      const totals = runs.map((x) => x.totalMs).sort((a, b) => a - b);
      const names = runs[0]?.passes.map((x) => x.name) ?? [];
      timing = {
        width: W, height: H, submits: runs.length, runsPerSubmit: o.timingRuns ?? 8, meanTotalMs: totals.reduce((a, b) => a + b, 0) / Math.max(1, totals.length),
        medianTotalMs: totals[Math.floor(totals.length / 2)], minTotalMs: totals[0], maxTotalMs: totals[totals.length - 1],
        passes: names.map((name) => ({ name, ms: runs.reduce((a, x) => a + (x.passes.find((q) => q.name === name)?.ms ?? 0), 0) / runs.length })),
        passNames: r.denoiser!.passNames,
      };
    }
    if (o.mode === 'recovery') {
      const rb = device.createBuffer({ size: tileBuf!.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = device.createCommandEncoder();
      e.copyBufferToBuffer(tileBuf!, 0, rb, 0, tileBuf!.size);
      device.queue.submit([e.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const data = new Uint8Array(rb.getMappedRange().slice(0));
      rb.unmap(); rb.destroy();
      const f = new Float32Array(data.buffer);
      let bad = 0;
      for (const x of f) if (!Number.isFinite(x)) bad++;
      if (bad) errors.push(`tiles.bin: ${bad} non-finite tile means`);
      await uploadFile(o.run, 'tiles.bin', data);
      files.push('tiles.bin');
    }
    const fin = r.restir ? Array.from(new Uint32Array(await (async () => {
      const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const e = device.createCommandEncoder(); e.copyBufferToBuffer(r.restir!.counters, 0, b, 0, 16); device.queue.submit([e.finish()]);
      await b.mapAsync(GPUMapMode.READ); const x = b.getMappedRange().slice(0); b.unmap(); b.destroy(); return x;
    })())) : [];
    if (fin[0]) errors.push(`finalize: ${fin[0]} non-finite pixels`);
    const info = describeContext(ctx) as { vendor?: string; architecture?: string; description?: string };
    const meta = {
      kind: 'denoise-eval', mode: o.mode, run: o.run, package: o.package, packageSha256: hash.sha256, seed: o.seed, width: W, height: H, frames: total,
      pkgFrames: Array.from({ length: total }, (_, i) => pkgFrame(i)), evalFrames: o.evalFrames ?? [], tiles: o.mode === 'recovery' ? { tile: TILE, x: tilesX, y: tilesY, layout: 'f32 [frame][tile][dn, raw]' } : undefined,
      renderer: { renderMode: 'restir', restirMode: o.restirMode ?? 'interactive', settings: r.restir?.settings, textureMode: 'validation', intersector: 'woop-watertight', jitter: 'iid', accumulate: false },
      denoiser: { on: denoise, settings: r.denoiser?.settings, perFrame }, jitter: o.jitter ?? 'iid', accumulate: !!o.accumulate, pan: o.pan, vbuf: { primIdMismatch: vbufMismatch, maxBaryDiff: vbufMaxBary }, timing, finalizeCounters: fin,
      adapterInfo: { vendor: info.vendor, architecture: info.architecture, description: info.description }, chromeVersion: o.chromeVersion, userAgent: navigator.userAgent,
      files, ok: errors.length === 0, errors, timings: { totalMs: performance.now() - t0 }, createdAt: new Date().toISOString(),
    };
    await uploadFile(o.run, 'meta.json', new TextEncoder().encode(JSON.stringify(meta, null, 1)));
    files.push('meta.json');
    return { ok: errors.length === 0, run: o.run, files, meta, errors };
  } finally {
    r.destroy();
    debug.destroy();
    fu.destroy();
    color.destroy(); depth.destroy();
    tileBuf?.destroy(); tileParams?.destroy();
  }
}
