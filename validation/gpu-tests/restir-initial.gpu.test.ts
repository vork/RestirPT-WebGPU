// ReSTIR PT initial sampling (WP-A, restir-api.md §6.1): U-PT-BITS, U-RES-1 (GPU part), U-EP-1, U-RIS-1…4, U-SFX-1,
// plus the compile smoke of every M4 pipeline. Chrome lane authoritative; dawn.node is a pre-check.
import { afterAll, describe, expect, it } from 'vitest';
import { composeWgsl, createCheckedShaderModule } from '../../src/core/gpu/wgsl-composer.ts';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import { createLutBuffer, lutDefines } from '../../src/core/render/luts/lut-layout.ts';
import { RS_WGSL_CONSTS, rfPack, rfUnpack } from '../../src/core/render/restir/layout.ts';
import { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import { restirSettings } from '../../src/core/render/restir/presets.ts';
import { RS_PASSES, type RsPassName } from '../../src/core/render/restir/resources.ts';
import { getTestGpu, releaseTestGpu } from './device-factory.ts';
import { material, quadScene } from './pt-fixtures.ts';
import {
  allLightsScene, bitFixtureScene, boxCamera, gpuScene, hashF32, ptImage, restirRig, storageBuffer, type BitFixture,
} from './restir-fixtures.ts';

afterAll(releaseTestGpu);

// U-PT-BITS (1): hashes of 64-spp PT images recorded on the pre-refactor tree (commit 6efd0bd, Chrome lane / Metal).
const PT_BITS: Record<BitFixture, string> = { c0c: '10d55063', c0e: '87ffe26f', c0m: '2aa85291', x_quads: '194ab49b', c0s: '378997d3' };

describe('U-PT-BITS: the PT is bit-identical after the env-sample / length1 / bsdf_query refactors', () => {
  it('(1) image hashes of 64 spp on C0c, C0e, C0m, (x) quads, C0s', async () => {
    const got: Record<string, string> = {};
    for (const name of Object.keys(PT_BITS) as BitFixture[]) {
      const img = await ptImage(bitFixtureScene(name), 32, 24, 64);
      expect(img.counters.every((c) => c === 0), `${name} counters ${img.counters}`).toBe(true);
      got[name] = hashF32(img.mean);
    }
    console.log(`[U-PT-BITS] ${JSON.stringify(got)}`);
    for (const [k, v] of Object.entries(PT_BITS)) if (v) expect(got[k], k).toBe(v);
  });
});

// ------------------------------------------------------------------------------------------------ U-PT-BITS (2)

const QUERY_HARNESS = `
#include "material/bsdf.wgsl"
#include "common/rng.wgsl"
@group(0) @binding(1) var<storage, read_write> res: array<atomic<u32>, 64>;
struct Prm { seed: u32, nx: u32, pad0: u32, pad1: u32 }
@group(0) @binding(2) var<uniform> prm: Prm;

fn rnd(i: u32, k: u32) -> f32 { return u32_to_unit(pcg4d(vec4u(i, k, prm.seed, 0x51ed27u)).x); }
fn rdir(i: u32, k: u32) -> vec3f {
  let z = 1.0 - 2.0 * rnd(i, k);
  let r = sqrt(max(0.0, 1.0 - z * z));
  let ph = TWO_PI * rnd(i, k + 1u);
  return vec3f(r * cos(ph), r * sin(ph), z);
}
fn rel(a: f32, b: f32) -> f32 {
  if (a == b) { return 0.0; }
  let e = abs(a - b) / max(abs(b), 1e-30);
  return select(3e38, e, e <= 3e38);
}
fn rel3(a: vec3f, b: vec3f) -> f32 { return max(rel(a.x, b.x), max(rel(a.y, b.y), rel(a.z, b.z))); }
fn amax(slot: u32, e: f32) { atomicMax(&res[slot], bitcast<u32>(e)); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.y * prm.nx * 64u + gid.x;
  var m: MatEval;
  m.model = i % 5u;
  let rk = (i / 5u) % 7u;
  var r = rnd(i, 1u);
  if (rk == 0u) { r = 0.0; } else if (rk == 1u) { r = 0.0037; } else if (rk == 2u) { r = 0.19; } else if (rk == 3u) { r = 0.21; } else if (rk == 4u) { r = 1.0; }
  m.roughness = r;
  m.base_color = vec3f(rnd(i, 2u), rnd(i, 3u), rnd(i, 4u));
  m.metallic = select(rnd(i, 5u), select(0.0, 1.0, rnd(i, 5u) < 0.5), rnd(i, 6u) < 0.3);
  m.ior = select(1.0 + 1.5 * rnd(i, 7u), 1.0, rnd(i, 8u) < 0.05);
  m.specular_level = rnd(i, 9u);
  m.specular_tint = vec3f(rnd(i, 10u), rnd(i, 11u), rnd(i, 12u));
  m.transmission = select(0.0, max(rnd(i, 13u), 0.01), m.model == 2u);
  let ng = rdir(i, 20u);
  var ns = normalize(ng + 0.35 * rdir(i, 22u));
  if (rnd(i, 24u) < 0.3) { ns = ng; }
  m.ng = ng;
  m.ns = ns;
  let V = rdir(i, 30u);
  let L = select(rdir(i, 32u), reflect(-V, ns), rnd(i, 34u) < 0.1);
  m.flags = select(0u, MATEVAL_BACKFACING, rnd(i, 35u) < 0.5);
  m.flags = bsdf_flags(m, V);
  let lobe = (i / 35u) % 6u;
  let q = bsdf_query(m, V, L, lobe);
  let el = bsdf_eval_lobe(m, V, L, lobe);
  let ev = bsdf_eval(m, V, L);
  let pm = bsdf_pdf_marginal(m, V, L);
  let sp = bsdf_sample_support(m, V, L, lobe);
  atomicAdd(&res[0], 1u);
  amax(1u, rel3(q.f_lobe, el.rgb));
  amax(2u, rel(q.p_joint, el.a));
  amax(3u, rel3(q.f_all, ev.f_cos));
  amax(4u, rel(q.p_marg, ev.pdf_marginal));
  amax(5u, rel(q.p_marg, pm));
  if (q.supp != sp) { atomicAdd(&res[6], 1u); }
  if (q.p_joint > 0.0 && lobe < 4u) { atomicAdd(&res[8u + lobe], 1u); }
  if (sp) { atomicAdd(&res[16u + lobe], 1u); }
  if (any(q.f_lobe > vec3f(0.0))) { atomicAdd(&res[24u + lobe], 1u); }
}
`;

describe('U-PT-BITS (2): bsdf_query ≡ the public BSDF API', () => {
  it('f_lobe, p_joint ≡ bsdf_eval_lobe; f_all, p_marg ≡ bsdf_eval / bsdf_pdf_marginal (1e-6 rel); supp ≡ bsdf_sample_support', async () => {
    const { device, features, wgslLanguageFeatures } = await getTestGpu();
    const shader = composeWgsl('restir-query-test.wgsl', {
      sources: { ...shaderSources, 'restir-query-test.wgsl': QUERY_HARNESS }, defines: lutDefines({ declare: { group: 0, binding: 0 } }), features, wgslLanguageFeatures,
    });
    const module = await createCheckedShaderModule(device, shader, 'bsdf-query-test');
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const lut = createLutBuffer(device);
    const out = storageBuffer(device, 256);
    const prm = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const NX = 1024, NY = 64;   // 4,194,304 queries
    device.queue.writeBuffer(prm, 0, Uint32Array.from([0x1234567, NX, 0, 0]));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [0, 1, 2].map((b) => ({ binding: b, resource: { buffer: [lut, out, prm][b] } })) }));
    pass.dispatchWorkgroups(NX, NY);
    pass.end();
    device.queue.submit([enc.finish()]);
    const r = new Uint32Array(await readBuffer(device, out, 256));
    const f = new Float32Array(r.buffer);
    const report = { n: r[0], errLobe: f[1], errJoint: f[2], errAll: f[3], errMarg: f[4], errMarg2: f[5], suppMismatch: r[6], jointPos: [...r.subarray(8, 12)], supp: [...r.subarray(16, 22)], fPos: [...r.subarray(24, 30)] };
    console.log(`[U-PT-BITS 2] ${JSON.stringify(report)}`);
    expect(r[0]).toBe(NX * NY * 64);
    for (const k of ['errLobe', 'errJoint', 'errAll', 'errMarg', 'errMarg2'] as const) expect(report[k], k).toBeLessThanOrEqual(1e-6);
    expect(report.suppMismatch).toBe(0);
    for (const c of report.jointPos) expect(c).toBeGreaterThan(1000);   // coverage: every lobe class has positive joint pdfs
    lut.destroy(); out.destroy(); prm.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ U-RES-1 (GPU part)

const RES1_WGSL = `
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"
@group(2) @binding(0) var<storage, read> inp: array<u32>;
@group(2) @binding(1) var<storage, read_write> outp: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i == 0u) {
    // uniforms echo (RestirParams, RsDispatch)
    outp[0] = rsParams.atlasSize.x; outp[1] = rsParams.atlasSize.y; outp[2] = rsParams.memberSize.x; outp[3] = rsParams.memberSize.y;
    outp[4] = rsParams.memberCols; outp[5] = rsParams.memberCount; outp[6] = rsParams.maxBounces; outp[7] = rsParams.flags;
    outp[8] = rsParams.numTrees; outp[9] = rsParams.numSlots; outp[10] = rsParams.numRounds; outp[11] = rsParams.rrMinBounces;
    outp[12] = bitcast<u32>(rsParams.tau); outp[13] = bitcast<u32>(rsParams.alphaMin); outp[14] = bitcast<u32>(rsParams.wScale);
    outp[15] = bitcast<u32>(rsParams.crit2022MinDist); outp[16] = rsParams.pairTexSize.x; outp[17] = rsParams.pairTexSize.w;
    outp[18] = rsParams.pairTexSize2.x; outp[19] = rsParams.lightMode; outp[20] = rsParams.memberBase;
    outp[21] = rsDispatch.t; outp[22] = rsDispatch.passId; outp[23] = rsDispatch.round; outp[24] = rsDispatch.treeBase;
    outp[25] = rsDispatch.treeCount; outp[26] = rsDispatch.flags; outp[27] = rsDispatch.rowBase; outp[28] = rsDispatch.rowEnd;
  }
  if (i >= 4096u) { return; }
  let b = i * 10u;
  let f = rf_pack(inp[b], inp[b + 1u], inp[b + 2u], inp[b + 3u], inp[b + 4u] != 0u, inp[b + 5u], inp[b + 6u] != 0u, inp[b + 7u], inp[b + 8u] != 0u, inp[b + 9u] != 0u);
  outp[64u + i] = f;
  outp[64u + 4096u + i] = rf_d(f) | (rf_k(f) << 4u) | (rf_tech(f) << 8u) | (rf_ep(f) << 10u) | (rf_lkm1(f) << 13u) | (rf_lk(f) << 16u);
}
`;

describe('U-RES-1 (GPU part): WGSL rf_pack / accessors and uniform layouts ≡ layout.ts', () => {
  it('rf_pack of 4096 random field sets equals rfPack; RestirParams / RsDispatch read back as packed', async () => {
    const rig = await restirRig(bitFixtureScene('c0c'), 16, 8, { settings: { maxBounces: 5, rr: true, rrMinBounces: 2, trees: 7, rounds: 2, slots: 5 }, memberBase: 3 });
    const k = rig.kernel, device = rig.g.device;
    let s = 99;
    const rnd = (n: number) => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return (s >>> 8) % n; };
    const inp = new Uint32Array(4096 * 10);
    const want = new Uint32Array(4096);
    for (let i = 0; i < 4096; i++) {
      const f = { d: rnd(16), k: rnd(16), tech: rnd(4), ep: rnd(8), isDelta: !!rnd(2), lkm1: rnd(8), dkm1: !!rnd(2), lk: rnd(8), dk: !!rnd(2), forced: !!rnd(2) };
      inp.set([f.d, f.k, f.tech, f.ep, +f.isDelta, f.lkm1, +f.dkm1, f.lk, +f.dk, +f.forced], i * 10);
      want[i] = rfPack(f);
    }
    const bin = storageBuffer(device, inp), bout = storageBuffer(device, (64 + 8192) * 4);
    const g2l = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ] });
    const pl = await k.compile('res1.wgsl', 'main', k.customDefines({}, false), k.customLayout(g2l, false), 'u-res-1', { 'res1.wgsl': RES1_WGSL });
    const g2 = device.createBindGroup({ layout: g2l, entries: [{ binding: 0, resource: { buffer: bin } }, { binding: 1, resource: { buffer: bout } }] });
    k.beginSubmit();
    const enc = device.createCommandEncoder();
    k.encodeCustom(enc, pl, g2, { t: 1234567, passId: 17, round: 2, treeBase: 5, treeCount: 3, flags: 6, rowBase: 8, rowEnd: 16 }, [64, 1], false);
    device.queue.submit([enc.finish()]);
    const o = new Uint32Array(await readBuffer(device, bout, (64 + 8192) * 4));
    const of = new Float32Array(o.buffer);
    expect(Array.from(o.subarray(0, 12))).toEqual([16, 8, 16, 8, 1, 1, 5, RS_WGSL_CONSTS.RSF_RR, 7, 5, 2, 2]);
    expect([of[12], of[13], of[14]]).toEqual([Math.fround(2e-4), Math.fround(0.2), 1]);
    expect(of[15]).toBeGreaterThan(0);
    expect(Array.from(o.subarray(16, 21))).toEqual([254, 230, 222, 0, 3]);
    expect(Array.from(o.subarray(21, 29))).toEqual([1234567, 17, 2, 5, 3, 6, 8, 16]);
    expect(Array.from(o.subarray(64, 64 + 4096))).toEqual(Array.from(want));
    for (let i = 0; i < 4096; i++) {
      const u = rfUnpack(want[i]);
      expect(o[64 + 4096 + i]).toBe(u.d | (u.k << 4) | (u.tech << 8) | (u.ep << 10) | (u.lkm1 << 13) | (u.lk << 16));
    }
    bin.destroy(); bout.destroy(); rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ U-EP-1

const EP1_WGSL = `
#include "restir/frame.wgsl"
#include "restir/endpoint.wgsl"
@group(2) @binding(0) var<storage, read_write> cnt: array<atomic<u32>, 32>;
fn same(a: LightSample, b: LightSample) -> bool {
  return a.valid == b.valid && a.entry == b.entry && a.kind == b.kind && all(bitcast<vec3u>(a.pos) == bitcast<vec3u>(b.pos))
    && all(bitcast<vec3u>(a.nz) == bitcast<vec3u>(b.nz)) && a.prim == b.prim && all(bitcast<vec3u>(a.dir) == bitcast<vec3u>(b.dir))
    && bitcast<u32>(a.dist) == bitcast<u32>(b.dist) && bitcast<u32>(a.cosZ) == bitcast<u32>(b.cosZ)
    && all(bitcast<vec3u>(a.Lambda) == bitcast<vec3u>(b.Lambda)) && bitcast<u32>(a.q) == bitcast<u32>(b.q)
    && bitcast<u32>(a.p1) == bitcast<u32>(b.p1) && a.isDelta == b.isDelta && a.isInf == b.isInf && a.analytic == b.analytic;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.y * 65536u * 64u + gid.x + rsDispatch.treeBase * 0x100000u;
  let h = pcg4d(vec4u(i, rsDispatch.t, 0x3c6ef372u, 11u));
  let h2 = pcg4d(vec4u(i, rsDispatch.t, 0xa54ff53au, 13u));
  let x = vec3f(-1.45, 0.02, -1.95) + vec3f(2.9, 1.9, 2.9) * vec3f(u32_to_unit(h2.x), u32_to_unit(h2.y), u32_to_unit(h2.z));
  let hL = vec3u(h.z, h.w, h2.w);
  let a = nee_sample(x, h.x, h.y, hL);
  let ep = nee_draw(lightsParams.cur, h.x, h.y, hL);
  let b = nee_eval(x, ep);
  let k = min(a.kind, 7u);
  atomicAdd(&cnt[k], 1u);
  if (!same(a, b)) { atomicAdd(&cnt[8u + k], 1u); }
  if (a.valid) { atomicAdd(&cnt[16u + k], 1u); }
  let w = nee_endpoint_words(ep);
  let back = nee_endpoint_from_words(w);
  if (back.entry != ep.entry || back.a != ep.a || back.b != ep.b || (w.x & RC_TAG_MASK) != RC_TAG_NEE) { atomicAdd(&cnt[24], 1u); }
}
`;

describe('U-EP-1: nee_eval(x, nee_draw(h)) ≡ nee_sample(x, h) bitwise, every endpoint type', () => {
  it('≥ 10⁶ random (x, hashes) per light type incl. env and emissive triangles', async () => {
    const rig = await restirRig(allLightsScene(), 8, 8);
    const k = rig.kernel, device = rig.g.device;
    const cnt = storageBuffer(device, 128);
    const g2l = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }] });
    const pl = await k.compile('ep1.wgsl', 'main', k.customDefines({}), k.customLayout(g2l), 'u-ep-1', { 'ep1.wgsl': EP1_WGSL });
    const g2 = device.createBindGroup({ layout: g2l, entries: [{ binding: 0, resource: { buffer: cnt } }] });
    const types = [0, 1, 2, 3, 4, 5, 7];
    let c = new Uint32Array(32);
    for (let chunk = 0; chunk < 400; chunk++) {
      k.beginSubmit();
      const enc = device.createCommandEncoder();
      k.encodeCustom(enc, pl, g2, { t: 5, treeBase: chunk }, [16384, 1]);   // 2^20 per chunk
      device.queue.submit([enc.finish()]);
      c = new Uint32Array(await readBuffer(device, cnt, 128));
      if (types.every((t) => c[t] >= 1e6)) break;
    }
    const report = Object.fromEntries(types.map((t) => [t, { n: c[t], valid: c[16 + t], mismatch: c[8 + t] }]));
    console.log(`[U-EP-1] ${JSON.stringify(report)} wordsMismatch=${c[24]}`);
    for (const t of types) {
      expect(c[t], `samples of type ${t}`).toBeGreaterThanOrEqual(1e6);
      expect(c[16 + t], `valid samples of type ${t}`).toBeGreaterThan(0);
      expect(c[8 + t], `bitwise mismatches of type ${t}`).toBe(0);
    }
    expect(c[24]).toBe(0);
    cnt.destroy(); rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ compile smoke

const ALL_MODULES_WGSL = `
#include "restir/frame.wgsl"
#include "restir/reservoir.wgsl"
#include "restir/queue.wgsl"
#include "restir/endpoint.wgsl"
#include "restir/rc.wgsl"
#include "restir/shift.wgsl"
#include "restir/mis.wgsl"
#include "restir/pairing.wgsl"
#include "path/replay.wgsl"
#include "path/pathtree.wgsl"
#include "path/length1.wgsl"
#include "debug/restir-views.wgsl"
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let p = rs_pix(gid.xy);
  if (!p.valid) { return; }
  let src = shift_src_load(p.ai);
  let dst = shift_dst_load(p.px);
  let o = shift_hybrid(src, dst);
  let pr = pair_partner(p.local, p.member, rs_t(), 0u, 0u);
  let a0 = pair_A0(rs_vbuf(p.px), rs_geo(p.px), rs_vbuf(p.px), rs_geo(p.px));
  let d = dihedral_apply_t(3u, dihedral_apply(3u, vec2i(1, 2)));
  let w = mis_partner_weight(0.5, 1.0, 1.0, 1.0, 2u) + mis_canonical_term(0.5, 1.0, 1.0, 1.0);
  if (pr.valid || a0 || d.x != 1 || w < 0.0 || res_needs_replay(src.flags)) { queue_append(0u, (p.ai << 3u) | 0u); }
  rs_count(RSC_CODE_BASE + rs_slot_code_sc(o.code), 1u);
  spatial_resample(p, 0u, true);
  let r = replay_prefix(src.seed, vertex_from_ids(dst.prim, dst.bary.x, dst.bary.y, dst.camPos), dst.prim, dst.camPos, dst.thr, 3u, 4u, 0u);
  rs_count(RSC_CODE_BASE + rs_slot_code_sc(r.code), 1u);
  pathtree_run(p, rs_frame_key(p.member, rs_t(), p.localIdx), 0u, 1u, true, true);
  rsdbg_slot(p.px, 0u, o.code, o.J, false);
}
`;

describe('P0 compile smoke: every M4 pipeline and every shared module', () => {
  it('RestirKernel (primary, initial, dump variant, finalize batch + interactive), all-module smoke, WP-C passes present', async () => {
    const g = await gpuScene(allLightsScene());
    const k = await RestirKernel.create(g.device, g.gpu, g.env, { settings: restirSettings('initial'), features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures, instrumentation: { dumpCandidates: true } });
    await k.pipeline('rs_finalize_frame', {}, 'rgba16float');
    k.setView({ camera: boxCamera(), width: 16, height: 16, runSeed: 1 });
    const device = g.device;
    const entries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'uint' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float' } },
    ];
    const g2l = device.createBindGroupLayout({ entries });
    const defs = k.customDefines({ RS_RES_IN_BINDING: '0u', RS_ARENA_BINDING: '1u', RS_ARENA_RW: true, RS_VBUF_BINDING: '2u', RS_GEO_BINDING: '3u', RS_RES_OUT_BINDING: '4u', RS_SHADE_W_BINDING: '5u', RS_REPLAY: 1 });
    // candDump is declared at binding 4 under RS_DUMP_CANDIDATES only: the smoke module does not set it.
    await k.compile('all.wgsl', 'main', defs, k.customLayout(g2l), 'restir-all-modules', { 'all.wgsl': ALL_MODULES_WGSL });
    const present = (['rs_pair_accept', 'rs_args', 'rs_spatial_replay', 'rs_spatial_shift', 'rs_spatial_resample', 'rs_ensemble_stats'] as RsPassName[]).filter((n) => RS_PASSES[n].file in shaderSources);
    for (const n of present) await k.pipeline(n);
    console.log(`[P0 smoke] WP-C passes compiled: ${present.join(', ') || '(none yet)'}`);
    k.destroy(); g.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ rung 3.1 frames

describe('rung 3.1 frames through RestirKernel', () => {
  it('camera facing an emissive quad: pixel = L_e exactly (length-1 term), counters 0; background pixels = env', async () => {
    const S = 50, D = 2;
    const scene = quadScene([{ p: [[-S, -S, -D], [-S, S, -D], [S, S, -D], [S, -S, -D]], mat: 0 }],
      [material({ emissiveFactor: [0.5, 0.25, 0.125], emissiveStrength: 2, v1: { diffuse: [0, 0, 0], glossy: [0, 0, 0], roughness: 0.5, mix: 0 } })]);
    const I4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const rig = await restirRig(scene, 32, 24, { cam: { camToWorld: I4, yfov: 40 * Math.PI / 180 }, settings: { maxBounces: 3 } });
    const r = await rig.frames(4);
    expect(r.counters).toEqual([0, 0, 0, 0]);
    let bad = 0;
    for (let i = 0; i < 32 * 24; i++) if (r.mean[3 * i] !== 1 || r.mean[3 * i + 1] !== 0.5 || r.mean[3 * i + 2] !== 0.25) bad++;
    expect(bad).toBe(0);
    expect(r.arena.rsc.candNonFinite + r.arena.rsc.bvhOverflow + r.arena.rsc.bvhItercap).toBe(0);
    rig.destroy();
  });
});
