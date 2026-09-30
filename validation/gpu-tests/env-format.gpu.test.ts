// ENV-F (docs/decisions/data-formats.md §B9, E-3, P2): hardware bilinear filtering (repeat, linear — the env sampler)
// of the compact env formats (rgb9e5ufloat / rgba16float) vs the SAME texels stored as rgba32float, at 10^6 random uv
// (incl. the u seam and the poles). Criterion: max |Δ| / L ≤ 2^-20 (L = the f32 result's largest component).
// Validation mode may use the compact formats only if this passes (env-gpu.ts ENV_COMPACT_IN_VALIDATION).
import { afterAll, describe, expect, it } from 'vitest';
import { getTestGpu, lane, releaseTestGpu } from './device-factory.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { ENV_COMPACT_IN_VALIDATION, createEnvResources, destroyEnvResources, type EnvTexFormat } from '../../src/core/render/env-gpu.ts';
import { decodeEnvironment } from '../../src/core/scene/env/load-env.ts';
import type { EnvironmentData } from '../../src/core/scene/types.ts';
import { readBytes } from './quant-fixtures.ts';
import { rng } from '../../tests/bvh/fixtures.ts';

const KERNEL = /* wgsl */ `
@group(0) @binding(0) var texA: texture_2d<f32>;
@group(0) @binding(1) var texB: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read> uvs: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> outp: array<vec4f>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x + gid.y * 65535u * 64u;
  if (i >= arrayLength(&uvs)) { return; }
  let a = textureSampleLevel(texA, samp, uvs[i], 0.0).rgb;
  let b = textureSampleLevel(texB, samp, uvs[i], 0.0).rgb;
  outp[2u * i] = vec4f(a, 0.0);
  outp[2u * i + 1u] = vec4f(b, 0.0);
}`;

const THRESH = 2 ** -20;
const N = 1_000_000;

function env(name: string, W: number, H: number, texel: (x: number, y: number) => number[]): EnvironmentData {
  const t = new Float32Array(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) t.set([...texel(x, y), 1], 4 * (y * W + x));
  return { name, width: W, height: H, texels: t, strength: 1, tint: [1, 1, 1], rotationZ: 0, visibleToCamera: true };
}

describe(`ENV-F compact env formats: hardware filtering = f32 filtering (${lane()})`, () => {
  afterAll(releaseTestGpu);
  const report: Record<string, unknown> = { lane: lane(), threshold: THRESH, ENV_COMPACT_IN_VALIDATION };
  afterAll(() => console.log('ENVF_REPORT', JSON.stringify(report)));

  const cases: { name: string; make: () => Promise<EnvironmentData | undefined> }[] = [
    ...['studio_small_09_1k', 'overcast_soil_puresky_1k', 'kloofendal_48d_partly_cloudy_puresky_1k'].map((id) => ({
      name: id, make: async () => { const b = await readBytes(`validation/assets/downloaded/hdri/${id}.hdr`); return b ? decodeEnvironment(b, { mode: 'validation' }).env : undefined; },
    })),
    { name: 'c0s-sun', make: async () => env('sun', 512, 256, (x, y) => (x === 7 && y === 100 ? [1e4, 1e4, 1e4] : [0, 0, 0])) },
    // RGBE-like random texels (9e5 exact: 9-bit mantissas at a shared exponent over 30 stops) with sharp contrasts
    { name: 'random-rgbe', make: async () => { const R = rng(5); return env('rgbe', 256, 128, () => { const e = Math.floor(R() * 30) - 15; return [0, 0, 0].map(() => Math.floor(R() * 512) * 2 ** (e - 8)); }); } },
  ];

  let allPass = true;
  for (const c of cases) {
    it(`${c.name}: compact vs rgba32float, ${N} bilinear lookups`, async () => {
      const e = await c.make();
      if (!e) { console.warn(`${c.name} not present; skipping`); report[`${c.name}.skipped`] = true; return; }
      const { device } = await getTestGpu();
      const f32 = await createEnvResources(device, e, 'envf-f32', { format: 'rgba32float' });
      const cmp = await createEnvResources(device, e, 'envf-compact', { mode: 'interactive' }); // smallest exact format
      const R = rng(9);
      const uv = new Float32Array(2 * N);
      for (let i = 0; i < N; i++) {
        const k = i % 8;
        let u = R(), v = R();
        if (k === 0) u = (R() - 0.5) * 4 / e.width;                 // across the u = 0 / 1 seam
        if (k === 1) v = R() * 2 / e.height;                        // nadir pole rows
        if (k === 2) v = 1 - R() * 2 / e.height;                    // zenith pole rows
        if (k === 3) { u = (Math.floor(R() * e.width) + 0.5) / e.width; v = (Math.floor(R() * e.height) + 0.5) / e.height; } // texel centres
        uv[2 * i] = u; uv[2 * i + 1] = v;
      }
      const uvBuf = device.createBuffer({ size: uv.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
      new Float32Array(uvBuf.getMappedRange()).set(uv); uvBuf.unmap();
      const outBuf = device.createBuffer({ size: N * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const module = device.createShaderModule({ code: KERNEL });
      const pipe = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
      const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: f32.view }, { binding: 1, resource: cmp.view }, { binding: 2, resource: f32.sampler },
        { binding: 3, resource: { buffer: uvBuf } }, { binding: 4, resource: { buffer: outBuf } }] });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe); pass.setBindGroup(0, bg);
      const wg = Math.ceil(N / 64);
      pass.dispatchWorkgroups(Math.min(wg, 65535), Math.ceil(wg / 65535));
      pass.end();
      device.queue.submit([enc.finish()]);
      const out = new Float32Array(await readBuffer(device, outBuf, N * 32));
      let worst = 0, worstAbs = 0, bad = 0, identical = 0;
      for (let i = 0; i < N; i++) {
        const a = out.subarray(8 * i, 8 * i + 3), b = out.subarray(8 * i + 4, 8 * i + 7);
        const L = Math.max(Math.abs(a[0]), Math.abs(a[1]), Math.abs(a[2]));
        const d = Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1]), Math.abs(a[2] - b[2]));
        if (d === 0) identical++;
        const rel = L > 0 ? d / L : d > 0 ? Infinity : 0;
        worst = Math.max(worst, rel); worstAbs = Math.max(worstAbs, d);
        if (rel > THRESH) bad++;
      }
      const pass_ = worst <= THRESH;
      allPass &&= pass_;
      const r = { format: cmp.format as EnvTexFormat, bytes: [e.width * e.height * 16, e.width * e.height * { rgb9e5ufloat: 4, rgba16float: 8, rgba32float: 16 }[cmp.format]], lookups: N,
        identical, worstRel: worst, worstRelLog2: worst > 0 ? +Math.log2(worst).toFixed(2) : -Infinity, worstAbs, overThreshold: bad, pass: pass_ };
      report[c.name] = r;
      console.log('ENV-F', lane(), c.name, JSON.stringify(r));
      uvBuf.destroy(); outBuf.destroy(); destroyEnvResources(f32); destroyEnvResources(cmp);
      // Validation mode may only use the compact formats when they filter like f32.
      if (ENV_COMPACT_IN_VALIDATION) expect(pass_).toBe(true);
    }, 300_000);
  }
  afterAll(() => { report.allPass = allPass; });
});
