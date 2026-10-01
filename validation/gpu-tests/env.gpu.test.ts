// ENV-U2 and ENV-U7 (plan §7.4 M1; env §5), both GPU lanes.
// ENV-U2: env.wgsl envUV/envDir vs an f64 port of Cycles direction_to_equirectangular (+ R_z(γ)·C).
// ENV-U7: the env bilinear + repeat/repeat (incl. the pole wrap) through env-gpu.ts + envRadiance vs an f64 reference.
// M5 (restir-temporal-api.md Changelog C-10): envRadiance is an explicit f32 bilinear (exact weights), no longer the
// hardware sampler (8-bit fraction on Apple GPUs); ENV-U7b checks it on a 1k HDRI and across texel formats (Q4).
import { afterAll, describe, expect, it } from 'vitest';
import { getTestGpu, lane, releaseTestGpu } from './device-factory.ts';
import { composeWgsl, createCheckedShaderModule } from '../../src/core/gpu/wgsl-composer.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { createEnvResources, destroyEnvResources, envBindGroupEntries, envDefines, envMemoryReport, writeEnvParams } from '../../src/core/render/env-gpu.ts';
import type { EnvironmentData } from '../../src/core/scene/types.ts';
import { loadEnvironment } from '../../src/core/scene/env/load-env.ts';
import { decodeHdr } from '../../src/core/scene/env/hdr.ts';
import { encodeHdr, prng } from '../../tests/env/rgbe-synth.ts';

// ---- f64 reference: literal port of Cycles kernel/camera/projection.h + util/projection.h ----
type V3 = [number, number, number];
/** b = R_z(γ)·C·d with C = R_x(+90°): (x,y,z)_g → (x, −z, y)_b, then Mapping POINT rotation about +Z. */
function glTFToBlender(d: V3, g: number): V3 {
  const c: V3 = [d[0], -d[2], d[1]];
  const cg = Math.cos(g), sg = Math.sin(g);
  return [cg * c[0] - sg * c[1], sg * c[0] + cg * c[1], c[2]];
}
function directionToEquirectangular(dir: V3): [number, number] {
  const range = [-2 * Math.PI, Math.PI, -Math.PI, Math.PI];
  const len = Math.hypot(dir[0], dir[1], dir[2]);
  if (len === 0) return [0, 0];
  return [(Math.atan2(dir[1], dir[0]) - range[1]) / range[0], (Math.acos(dir[2] / len) - range[3]) / range[2]];
}
function equirectangularToDirectionGltf(u: number, v: number, g: number): V3 {
  const phi = -2 * Math.PI * u + Math.PI, theta = -Math.PI * v + Math.PI;
  const b: V3 = [Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)];
  const cg = Math.cos(g), sg = Math.sin(g);
  const r: V3 = [cg * b[0] + sg * b[1], -sg * b[0] + cg * b[1], b[2]]; // R_z(−γ)
  return [r[0], r[2], -r[1]];                                          // C⁻¹
}

function xorshift(seed: number) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

const U2_WGSL = `
#include "lights/env.wgsl"
struct P { cg: f32, sg: f32, n: u32, pad: u32 }
@group(0) @binding(0) var<uniform> prm: P;
@group(0) @binding(1) var<storage, read> dirs: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> outUV: array<vec4f>;    // (envUV(d), envUV(envDir(uvIn)))
@group(0) @binding(3) var<storage, read_write> outDir: array<vec4f>;   // envDir(envUV(d))
@group(0) @binding(4) var<storage, read_write> outDir2: array<vec4f>;  // envDir(uvIn)
@group(0) @binding(5) var<storage, read> uvIn: array<vec2f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= prm.n) { return; }
  let d = dirs[i].xyz;
  let uv = envUV(d, prm.cg, prm.sg);
  outUV[i] = vec4f(uv, envUV(envDir(uvIn[i], prm.cg, prm.sg), prm.cg, prm.sg));
  outDir[i] = vec4f(envDir(uv, prm.cg, prm.sg), 0.0);
  outDir2[i] = vec4f(envDir(uvIn[i], prm.cg, prm.sg), 0.0);
}`;

const U7_WGSL = `
#include "lights/env.wgsl"
@group(0) @binding(3) var<storage, read> uvs: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> outRad: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> outBg: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&uvs)) { return; }
  outRad[i] = vec4f(envRadiance(uvs[i]), 0.0);
  outBg[i] = vec4f(envBackground(envDir(uvs[i], envParams.cg, envParams.sg)), 0.0); // camera-miss path
}`;

const GRID_WGSL = `
#include "debug/env-grid.wgsl"
struct P { cg: f32, sg: f32 }
@group(0) @binding(0) var<uniform> prm: P;
@group(0) @binding(1) var<storage, read> dirs: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> outC: array<vec4f>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&dirs)) { return; }
  outC[i] = env_grid_overlay(dirs[i].xyz, prm.cg, prm.sg);
}`;

describe(`env (${lane()})`, () => {
  afterAll(releaseTestGpu);

  it('loadEnvironment (Worker in Chrome, inline in Node) == decodeHdr, texels transferred intact', async () => {
    const w = 64, h = 32, next = prng(99);
    const rgbe = new Uint8Array(w * h * 4).map(() => next() & 0xff);
    const bytes = encodeHdr(w, h, rgbe, 'rle');
    const ref = decodeHdr(bytes);
    const got = await loadEnvironment(bytes.slice().buffer, { mode: 'validation', name: 'synth', strength: 3, rotationZ: 0.25 });
    expect(got.format).toBe('hdr');
    expect(got.decodedIn).toBe(lane() === 'chrome' ? 'worker' : 'inline');
    expect(got.env.width).toBe(w);
    expect(got.env.strength).toBe(3);
    expect(got.env.rotationZ).toBe(0.25);
    expect(Array.from(got.env.texels)).toEqual(Array.from(ref.texels));
  });

  it('ENV-U2: envUV/envDir == f64 Cycles port (1e5 dirs, γ ∈ {0, 0.5, −1.57})', async () => {
    const { device, features, wgslLanguageFeatures } = await getTestGpu();
    const shader = composeWgsl('tests/env-u2.wgsl', { sources: { ...shaderSources, 'tests/env-u2.wgsl': U2_WGSL }, defines: { ENV_NO_BINDINGS: true }, features, wgslLanguageFeatures });
    const module = await createCheckedShaderModule(device, shader, 'env-u2');
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });

    const N = 100_000;
    const rnd = xorshift(12345);
    const dirs = new Float32Array(N * 4);
    const uvIn = new Float32Array(N * 2);
    // exact axes and seam directions first, then uniform random directions (f64, rounded to f32)
    const special: V3[] = [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [-1, 0, 1e-7], [-1, 0, -1e-7], [0.6, 0.8, 0]];
    for (let i = 0; i < N; i++) {
      let d: V3;
      if (i < special.length) d = special[i];
      else {
        const z = 2 * rnd() - 1, p = 2 * Math.PI * rnd(), s = Math.sqrt(1 - z * z);
        d = [s * Math.cos(p), z, s * Math.sin(p)];
      }
      const l = Math.hypot(...d);
      dirs.set([d[0] / l, d[1] / l, d[2] / l, 0], i * 4);
      uvIn.set([rnd(), rnd()], i * 2);
    }
    const mk = (data: Float32Array | null, size: number, usage: number) => {
      const b = device.createBuffer({ size, usage, mappedAtCreation: !!data });
      if (data) { new Float32Array(b.getMappedRange()).set(data); b.unmap(); }
      return b;
    };
    const S = GPUBufferUsage.STORAGE;
    const dirBuf = mk(dirs, dirs.byteLength, S);
    const uvInBuf = mk(uvIn, uvIn.byteLength, S);
    const outUV = mk(null, N * 16, S | GPUBufferUsage.COPY_SRC);
    const outDir = mk(null, N * 16, S | GPUBufferUsage.COPY_SRC);
    const outDir2 = mk(null, N * 16, S | GPUBufferUsage.COPY_SRC);
    const prm = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const report: Record<string, unknown> = {};
    const checks: (() => void)[] = [];
    for (const g of [0, 0.5, -1.57]) {
      const pbuf = new ArrayBuffer(16);
      new Float32Array(pbuf, 0, 2).set([Math.cos(g), Math.sin(g)]);
      new Uint32Array(pbuf, 8, 1)[0] = N;
      device.queue.writeBuffer(prm, 0, pbuf);
      const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: prm } }, { binding: 1, resource: { buffer: dirBuf } }, { binding: 2, resource: { buffer: outUV } },
        { binding: 3, resource: { buffer: outDir } }, { binding: 4, resource: { buffer: outDir2 } }, { binding: 5, resource: { buffer: uvInBuf } }] });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(N / 64)); pass.end();
      device.queue.submit([enc.finish()]);
      const guv = new Float32Array(await readBuffer(device, outUV, N * 16));
      const gdir = new Float32Array(await readBuffer(device, outDir, N * 16));
      const gdir2 = new Float32Array(await readBuffer(device, outDir2, N * 16));

      // Metrics. "Pole" = within ~0.06° of the map zenith/nadir (sinθ < 1e-3), where atan2/acos are ill-conditioned.
      let maxDu = 0, maxDv = 0, maxDuAll = 0, maxDvAll = 0, maxRt = 0, maxDir = 0, maxUvRt = 0, nPole = 0, worstDvBz = 0;
      const dvHist = [0, 0, 0, 0]; // dv error buckets by |b.z|: <0.9, <0.99, <0.9999, rest
      for (let i = 0; i < N; i++) {
        const d: V3 = [dirs[i * 4], dirs[i * 4 + 1], dirs[i * 4 + 2]];
        const b = glTFToBlender(d, g);
        const [u, v] = directionToEquirectangular(b);
        let du = Math.abs(guv[i * 4] - u);
        du = Math.min(du, Math.abs(1 - du)); // atan2 seam: u ∈ {0, 1} are the same meridian
        const dv = Math.abs(guv[i * 4 + 1] - v);
        const pole = Math.hypot(b[0], b[1]) < 1e-3;
        if (pole) nPole++;
        maxDuAll = Math.max(maxDuAll, du); maxDvAll = Math.max(maxDvAll, dv);
        if (!pole) { maxDu = Math.max(maxDu, du); if (dv > maxDv) { maxDv = dv; worstDvBz = b[2]; } }
        const az = Math.abs(b[2]), bk = az < 0.9 ? 0 : az < 0.99 ? 1 : az < 0.9999 ? 2 : 3;
        dvHist[bk] = Math.max(dvHist[bk], dv);
        maxRt = Math.max(maxRt, Math.hypot(gdir[i * 4] - d[0], gdir[i * 4 + 1] - d[1], gdir[i * 4 + 2] - d[2]));
        const ref = equirectangularToDirectionGltf(uvIn[i * 2], uvIn[i * 2 + 1], g);
        maxDir = Math.max(maxDir, Math.hypot(gdir2[i * 4] - ref[0], gdir2[i * 4 + 1] - ref[1], gdir2[i * 4 + 2] - ref[2]));
        // uv round trip away from the poles and the seam
        const vIn = uvIn[i * 2 + 1];
        if (vIn > 1e-3 && vIn < 1 - 1e-3) {
          let e = Math.abs(guv[i * 4 + 2] - uvIn[i * 2]); e = Math.min(e, Math.abs(1 - e));
          maxUvRt = Math.max(maxUvRt, e, Math.abs(guv[i * 4 + 3] - vIn));
        }
      }
      // Orientation (math.md#env-mapping): at γ = 0, +X → u = 0.5, −Z → 0.25, +Z → 0.75, +Y → v = 1.
      if (g === 0) {
        expect(guv[0]).toBeCloseTo(0.5, 6); expect(guv[1]).toBeCloseTo(0.5, 6);
        expect(guv[3 * 4]).toBeCloseTo(0.25, 6); expect(guv[2 * 4]).toBeCloseTo(0.75, 6);
        expect(guv[4 * 4 + 1]).toBeCloseTo(1, 6); expect(guv[5 * 4 + 1]).toBeCloseTo(0, 6);
        expect(guv[4 * 4]).toBe(0.5); expect(guv[5 * 4]).toBe(0.5); // exact poles: Cycles atan2f(0,0) = 0 → u = 0.5
      }
      const r = { maxDu, maxDv, worstDvBz, maxDuAll, maxDvAll, nPole, roundTripDir: maxRt, envDirVsRef: maxDir, uvRoundTrip: maxUvRt };
      report[`gamma=${g}`] = { ...Object.fromEntries(Object.entries(r).map(([k, x]) => [k, k !== 'nPole' ? Number(x.toExponential(3)) : x])), dvByAbsBz: dvHist.map((x) => Number(x.toExponential(2))) };
      checks.push(() => {
        expect(maxDu).toBeLessThan(1e-6);
        expect(maxDv).toBeLessThan(1e-6);
        expect(maxRt).toBeLessThan(2e-6);
        expect(maxDir).toBeLessThan(2e-6);
        expect(maxUvRt).toBeLessThan(2e-6);
      });
    }
    console.log('ENV-U2', lane(), JSON.stringify(report));
    checks.forEach((c) => c());
    for (const b of [dirBuf, uvInBuf, outUV, outDir, outDir2, prm]) b.destroy();
  });

  it('ENV-U7: explicit f32 bilinear + repeat/repeat incl. pole wrap == f64 reference (exact weights)', async () => {
    const { device, features, wgslLanguageFeatures } = await getTestGpu();
    const W = 8, H = 4;
    // Distinct texels, rows bottom-up (row 0 = nadir). r: 1 + c + 8r; g: 100 + 17c − 9r; b: 2^(c%4)·(r+1).
    const texels = new Float32Array(W * H * 4);
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) texels.set([1 + c + 8 * r, 100 + 17 * c - 9 * r, 2 ** (c % 4) * (r + 1), 1], (r * W + c) * 4);
    const env: EnvironmentData = { name: 'u7', width: W, height: H, texels, strength: 1, tint: [1, 1, 1], rotationZ: 0, visibleToCamera: true };
    const res = await createEnvResources(device, env, 'env-u7');
    expect(envMemoryReport(res).textureBytes).toBe(W * H * 16);

    const shader = composeWgsl('tests/env-u7.wgsl', { sources: { ...shaderSources, 'tests/env-u7.wgsl': U7_WGSL }, defines: envDefines(0, 0), features, wgslLanguageFeatures });
    const module = await createCheckedShaderModule(device, shader, 'env-u7');
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });

    const texel = (c: number, r: number) => { const i = ((((r % H) + H) % H) * W + (((c % W) + W) % W)) * 4; return [texels[i], texels[i + 1], texels[i + 2]]; };
    /** CPU bilinear, repeat/repeat; `bits` quantizes the fractional weights (undefined = exact f64). */
    const bilinear = (u: number, v: number, bits?: number) => {
      const x = u * W - 0.5, y = v * H - 0.5;
      const x0 = Math.floor(x), y0 = Math.floor(y);
      let fx = x - x0, fy = y - y0;
      if (bits !== undefined) { const s = 2 ** bits; fx = Math.round(fx * s) / s; fy = Math.round(fy * s) / s; }
      const a = texel(x0, y0), b = texel(x0 + 1, y0), c = texel(x0, y0 + 1), d = texel(x0 + 1, y0 + 1);
      return [0, 1, 2].map((k) => (a[k] * (1 - fx) + b[k] * fx) * (1 - fy) + (c[k] * (1 - fx) + d[k] * fx) * fy);
    };

    // Crafted uv: texel centres, texel edges, the u seam, v within half a texel of both poles (wrap).
    const crafted: [number, number][] = [];
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) crafted.push([(c + 0.5) / W, (r + 0.5) / H]);
    for (let c = 0; c <= W; c++) crafted.push([c / W, 0.5 / H], [c / W, 2 / H]);
    crafted.push([0, 0], [1, 1], [0, 1], [1, 0], [0.5, 0], [0.5, 1], [0.03125, 0.5], [0.96875, 0.5], [0.5 / W, 0.125 / H], [0.5, 1 - 0.125 / H],
      [0.25 / W, 0.25 / H], [1 - 0.25 / W, 1 - 0.25 / H], [0.5, 0.25 / H], [0.5, 1 - 0.25 / H], [-0.25 / W, 0.5], [1 + 0.25 / W, 0.5], [3.5 / W, 0]);
    const rnd = xorshift(777);
    const NR = 100_000;
    const uvs = new Float32Array((crafted.length + NR) * 2);
    crafted.forEach(([u, v], i) => uvs.set([u, v], i * 2));
    for (let i = 0; i < NR; i++) {
      // half the random points concentrated within half a texel of the poles and the seam
      const u = i % 2 ? rnd() : (rnd() - 0.5) / W;
      const v = i % 4 === 1 ? rnd() * 0.5 / H : i % 4 === 3 ? 1 - rnd() * 0.5 / H : rnd();
      uvs.set([u, v], (crafted.length + i) * 2);
    }
    const n = uvs.length / 2;
    const uvBuf = device.createBuffer({ size: uvs.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Float32Array(uvBuf.getMappedRange()).set(uvs); uvBuf.unmap();
    const out = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const outBg = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    const run = async () => {
      const layout = pipeline.getBindGroupLayout(0);
      const bg = device.createBindGroup({ layout, entries: [...envBindGroupEntries(res, 0), { binding: 3, resource: { buffer: uvBuf } }, { binding: 4, resource: { buffer: out } }, { binding: 5, resource: { buffer: outBg } }] });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(n / 64)); pass.end();
      device.queue.submit([enc.finish()]);
      return [new Float32Array(await readBuffer(device, out, n * 16)), new Float32Array(await readBuffer(device, outBg, n * 16))] as const;
    };
    const [g, bg0] = await run();
    const scale = 1 + 8 * H + 100 + 17 * W; // ~ max texel magnitude, for relative errors
    const errOf = (i: number, bits?: number) => { const ref = bilinear(uvs[i * 2], uvs[i * 2 + 1], bits); return Math.max(...[0, 1, 2].map((k) => Math.abs(g[i * 4 + k] - ref[k]))) / scale; };
    let craftedMax = 0;
    for (let i = 0; i < crafted.length; i++) craftedMax = Math.max(craftedMax, errOf(i));
    // Random points: exact-weight error and the best-fitting fractional-weight precision.
    let randMax = 0;
    for (let i = crafted.length; i < n; i++) randMax = Math.max(randMax, errOf(i));
    const fit: Record<number, number> = {};
    for (const bits of [6, 8, 10, 12, 14, 16, 20]) { let m = 0; for (let i = crafted.length; i < n; i++) m = Math.max(m, errOf(i, bits)); fit[bits] = Number(m.toExponential(2)); }
    // Pole wrap is really a wrap (not clamp): at v = 0, u = centre of col 3 → 50/50 of row 0 and row H−1.
    const iWrap = crafted.findIndex(([u, v]) => u === 3.5 / W && v === 0);
    const wrapExpect = [0, 1, 2].map((k) => 0.5 * (texel(3, 0)[k] + texel(3, H - 1)[k]));
    const wrapGot = [0, 1, 2].map((k) => g[iWrap * 4 + k]);
    // Row 0 (first uploaded, bottom-up) sits at v = 0.5/H: WebGPU y = 0 is the first uploaded row, no flip.
    const iRow0 = 3; // crafted[3] = centre of (c=3, r=0)
    expect([g[iRow0 * 4], g[iRow0 * 4 + 1], g[iRow0 * 4 + 2]]).toEqual(texel(3, 0));
    for (let k = 0; k < 3; k++) expect(Math.abs(wrapGot[k] - wrapExpect[k]) / scale).toBeLessThan(1e-6);

    // strength·tint scaling goes through the same lookup.
    writeEnvParams(device, res, { strength: 2, tint: [1, 0.5, 0.25] });
    const [g2] = await run();
    for (let k = 0; k < 3; k++) expect(g2[iRow0 * 4 + k]).toBe(texel(3, 0)[k] * 2 * [1, 0.5, 0.25][k]);
    // Camera-miss term: envBackground(envDir(uv)) == envRadiance(uv) at texel centres (γ = 0.7); 0 when hidden.
    writeEnvParams(device, res, { strength: 1, tint: [1, 1, 1], rotationZ: 0.7 });
    const [g3, bg3] = await run();
    let bgMax = 0;
    for (let i = 0; i < W * H; i++) for (let k = 0; k < 3; k++) bgMax = Math.max(bgMax, Math.abs(bg3[i * 4 + k] - g3[i * 4 + k]) / scale);
    expect(bgMax).toBeLessThan(1e-4);
    expect(bg0[iRow0 * 4]).toBeCloseTo(texel(3, 0)[0], 3);
    writeEnvParams(device, res, { visibleToCamera: false });
    const [, bg4] = await run();
    expect(Math.max(...bg4.subarray(0, W * H * 4))).toBe(0);

    console.log('ENV-U7', lane(), JSON.stringify({ crafted: crafted.length, craftedMaxRel: Number(craftedMax.toExponential(3)), random: NR,
      randomMaxRelExactWeights: Number(randMax.toExponential(3)), randomMaxRelQuantizedWeightsByBits: fit, poleWrap: { got: wrapGot, expect: wrapExpect },
      envBackgroundVsRadiance: Number(bgMax.toExponential(2)) }));
    expect(craftedMax).toBeLessThan(1e-6);
    // C-10: exact weights (the M3 hardware sampler matched the 8-bit emulation instead; Cycles on Metal still does,
    // a ≤ 2^-9·Δtexel per-lookup difference that averages out over any pixel footprint).
    expect(randMax).toBeLessThan(1e-6);
    uvBuf.destroy(); out.destroy(); outBg.destroy(); destroyEnvResources(res);
  });
  it('ENV-U7b: envRadiance on studio_small_09 1k ≡ f64 bilinear at the f32 texel coordinate (seam, poles); compact formats bit-identical (Q4)', async () => {
    const { device, features, wgslLanguageFeatures } = await getTestGpu();
    const { loadHdri } = await import('./env-fixtures.ts');
    const env = await loadHdri('studio_small_09_1k.hdr');
    if (!env) { console.warn('ENV-U7b: studio_small_09_1k.hdr missing; skipped'); return; }
    const W = env.width, H = env.height, t = env.texels;
    const shader = composeWgsl('tests/env-u7.wgsl', { sources: { ...shaderSources, 'tests/env-u7.wgsl': U7_WGSL }, defines: envDefines(0, 0), features, wgslLanguageFeatures });
    const module = await createCheckedShaderModule(device, shader, 'env-u7b');
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const rnd = xorshift(4242);
    const N = 200_000;
    const uvs = new Float32Array(2 * N);
    for (let i = 0; i < N; i++) {
      const k = i % 4;
      const u = k === 0 ? (rnd() - 0.5) * 4 / W : rnd();
      const v = k === 1 ? rnd() * 1.5 / H : k === 2 ? 1 - rnd() * 1.5 / H : rnd();
      uvs.set([u, v], 2 * i);
    }
    const uvBuf = device.createBuffer({ size: uvs.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Float32Array(uvBuf.getMappedRange()).set(uvs); uvBuf.unmap();
    const out = device.createBuffer({ size: N * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const outBg = device.createBuffer({ size: N * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const run = async (res: Awaited<ReturnType<typeof createEnvResources>>) => {
      const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [...envBindGroupEntries(res, 0), { binding: 3, resource: { buffer: uvBuf } }, { binding: 4, resource: { buffer: out } }, { binding: 5, resource: { buffer: outBg } }] });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(N / 64)); pass.end();
      device.queue.submit([enc.finish()]);
      return new Float32Array(await readBuffer(device, out, N * 16));
    };
    const f32res = await createEnvResources(device, env, 'env-u7b-f32', { format: 'rgba32float' });
    const g = await run(f32res);
    const at = (c: number, r: number, k: number) => t[4 * ((((r % H) + H) % H) * W + (((c % W) + W) % W)) + k];
    let maxRel = 0;
    for (let i = 0; i < N; i++) {
      // reference: the f32 texel coordinate x = fl(fl(u·W) − 0.5) (as the shader forms it), f64 weights and lerps
      const x = Math.fround(Math.fround(uvs[2 * i] * W) - 0.5), y = Math.fround(Math.fround(uvs[2 * i + 1] * H) - 0.5);
      const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
      for (let k = 0; k < 3; k++) {
        const a = at(x0, y0, k), b = at(x0 + 1, y0, k), c = at(x0, y0 + 1, k), d = at(x0 + 1, y0 + 1, k);
        const ref = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
        const m = Math.max(a, b, c, d, 1e-30);
        maxRel = Math.max(maxRel, Math.abs(g[4 * i + k] - ref) / m);
      }
    }
    const formats: Record<string, number> = {};
    const cmp = await createEnvResources(device, env, 'env-u7b-compact', { mode: 'interactive' });   // smallest exact format
    const gc = await run(cmp);
    let diff = 0;
    for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) if (gc[4 * i + k] !== g[4 * i + k]) diff++;
    formats[cmp.format] = diff;
    console.log('ENV-U7b', lane(), JSON.stringify({ W, H, N, maxRelVsF64AtF32Coord: Number(maxRel.toExponential(3)), compactFormatMismatches: formats }));
    expect(maxRel).toBeLessThan(1e-6);
    expect(diff).toBe(0);
    uvBuf.destroy(); out.destroy(); outBg.destroy(); destroyEnvResources(f32res); destroyEnvResources(cmp);
  });
  it('env-grid overlay: axis discs, rings and horizon', async () => {
    const { device, features, wgslLanguageFeatures } = await getTestGpu();
    const shader = composeWgsl('tests/env-grid.wgsl', { sources: { ...shaderSources, 'tests/env-grid.wgsl': GRID_WGSL }, defines: { ENV_NO_BINDINGS: true }, features, wgslLanguageFeatures });
    const module = await createCheckedShaderModule(device, shader, 'env-grid');
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const n2 = (x: number, y: number, z: number) => { const l = Math.hypot(x, y, z); return [x / l, y / l, z / l, 0]; };
    const ring = 0.0873; // 5° in rad: on the env ring around the u = 0.5 map point
    const dirs = new Float32Array([
      ...n2(1, 0, 0), ...n2(0, 0, -1), ...n2(0, 0, 1), ...n2(-1, 0, 0), ...n2(0, 1, 0),   // discs
      ...n2(Math.cos(0.3), 0, Math.sin(0.3)),                                          // horizon line (17° from +X)
      ...n2(Math.cos(0.26) * Math.cos(0.4), Math.sin(0.26), Math.cos(0.26) * Math.sin(0.4)), // 15° up, 23° azimuth: empty
      ...n2(Math.cos(ring), Math.sin(ring), 0),                                        // on the +X ring (γ = 0)
    ]);
    const dirBuf = device.createBuffer({ size: dirs.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Float32Array(dirBuf.getMappedRange()).set(dirs); dirBuf.unmap();
    const n = dirs.length / 4;
    const out = device.createBuffer({ size: n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const prm = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const run = async (g: number) => {
      device.queue.writeBuffer(prm, 0, new Float32Array([Math.cos(g), Math.sin(g)]));
      const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: prm } }, { binding: 1, resource: { buffer: dirBuf } }, { binding: 2, resource: { buffer: out } }] });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(1); pass.end();
      device.queue.submit([enc.finish()]);
      return new Float32Array(await readBuffer(device, out, n * 16));
    };
    const c = await run(0);
    const px = (i: number) => Array.from(c.subarray(i * 4, i * 4 + 4)).map((x) => Number(x.toFixed(2)));
    expect(px(0)).toEqual([1, 0.1, 0.1, 1]);   // +X red
    expect(px(1)).toEqual([0.1, 1, 0.1, 1]);   // −Z green
    expect(px(2)).toEqual([0.2, 0.4, 1, 1]);   // +Z blue
    expect(px(3)).toEqual([1, 0.9, 0.1, 1]);   // −X yellow
    expect(px(4)).toEqual([1, 1, 1, 1]);       // zenith
    expect(c[5 * 4 + 3]).toBeGreaterThan(0.8); // horizon
    expect(c[6 * 4 + 3]).toBe(0);              // off-grid
    expect(px(7)).toEqual([1, 0.1, 0.1, 1]);   // env ring for u = 0.5 around +X at γ = 0
    const r = await run(0.5);                  // rotated env: the ring moves away (world discs stay)
    expect(r[7 * 4 + 3]).toBeLessThan(1);
    expect(Array.from(r.subarray(0, 4)).map((x) => Number(x.toFixed(2)))).toEqual([1, 0.1, 0.1, 1]);
    dirBuf.destroy(); out.destroy(); prm.destroy();
  });
});
