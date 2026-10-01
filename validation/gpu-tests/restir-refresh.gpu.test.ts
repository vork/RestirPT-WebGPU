// Suffix refresh and the light/env change taxonomy (T-C, restir-temporal-api.md §3.5, §6.1; math.md#light-changes
// [M5 addition]; gap-temporal §9.3): canary (Q1), U-SFX-2, §9.3-3 refresh idempotence, §9.3-4 refresh vs a full
// re-trace on light-change frames (forward and inverse), §9.3-2 (a) FD area ratio of the light-local map and (b) the PSS
// change of variables with J_P (z-test, 10⁸ samples; negative control N2), the refresh part of T-ENV-temporal (env
// rotation / strength / tint in every case above), E2 and the refresh plants (N3, N4, N7, NO_JP, NO_JP_ENV,
// ENV_NO_ROT_VIS). U-SFX-2 lives here instead of restir-initial.gpu.test.ts (Changelog C-4). Chrome lane authoritative.
// No test kernel loops over shifts (the Q2 shape); records are read plane by plane (Q1).
import { afterAll, describe, expect, it } from 'vitest';
import { readBuffer } from '../../src/core/gpu/readback.ts';
import type { EnvParamsCpu } from '../../src/core/render/env-gpu.ts';
import type { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import {
  RES_WORDS, RS_DUMP_CAP, RS_WGSL_CONSTS as K, RW, TS_CONSTS as S, dumpCountWord, dumpRecordWord, lobeHistAt, pathClass, rfUnpack, PATH_CLASS_NAMES,
} from '../../src/core/render/restir/layout.ts';
import type { RestirSettings } from '../../src/core/render/restir/presets.ts';
import type { LightData, SceneData } from '../../src/core/scene/types.ts';
import { releaseTestGpu } from './device-factory.ts';
import { lightMatrixToward, type V3 } from './pt-fixtures.ts';
import { allLightsScene, boxCamera, boxScene, light, restirRig, storageBuffer, TEST_RC_DR, type RestirRig } from './restir-fixtures.ts';

afterAll(releaseTestGpu);

const C = { L: 0, N1: 1, B1: 2, E: 3, DNEE: 4, DBSDF: 5, R: 6 } as const;
const f32 = (u: number) => new Float32Array(new Uint32Array([u]).buffer)[0];
const rel = (a: number, b: number) => (a === b ? 0 : Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1e-30));
const rel3 = (a: ArrayLike<number>, b: ArrayLike<number>, ao = 0, bo = 0) => Math.max(rel(a[ao], b[bo]), rel(a[ao + 1], b[bo + 1]), rel(a[ao + 2], b[bo + 2]));

// ------------------------------------------------------------------------------------------------ records

interface Recs { rec: Uint32Array; ai: Uint32Array; n: number }

/** Every streamed candidate of tree 0 (the candidate dump) as a plain record array (40 words each) + its pixel. */
function harvest(dump: Uint32Array, P: number): Recs {
  let n = 0;
  for (let ai = 0; ai < P; ai++) n += Math.min(dump[dumpCountWord(P, ai)], RS_DUMP_CAP);
  const rec = new Uint32Array(Math.max(n, 1) * RES_WORDS), aiA = new Uint32Array(Math.max(n, 1));
  let i = 0;
  for (let ai = 0; ai < P; ai++) {
    const c = Math.min(dump[dumpCountWord(P, ai)], RS_DUMP_CAP);
    for (let j = 0; j < c; j++, i++) {
      const b = dumpRecordWord(ai, j);
      rec.set(dump.subarray(b, b + RES_WORDS), i * RES_WORDS);
      aiA[i] = ai;
    }
  }
  return { rec, ai: aiA, n };
}
const recFlags = (r: Recs, i: number) => r.rec[i * RES_WORDS + RW.flags];
const recClass = (r: Recs, i: number) => pathClass(rfUnpack(recFlags(r, i)));
const recF = (r: Recs, i: number, w: number) => f32(r.rec[i * RES_WORDS + w]);

// ------------------------------------------------------------------------------------------------ test kernels

const OUT_WORDS = 8;
/** refresh_record over a record array: i < treeCount, fsFrom = RsDispatch.round, fsTo = RsDispatch.flags. */
const RF_WGSL = `
#include "restir/refresh.wgsl"
@group(2) @binding(2) var<storage, read_write> outR: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.y * 65535u * 64u + gid.x;
  if (i >= rsDispatch.treeCount) { return; }
  let r = refresh_record(i, rsDispatch.round, rsDispatch.flags);
  let b = 8u * i;
  outR[b] = bitcast<u32>(r.rad.x); outR[b + 1u] = bitcast<u32>(r.rad.y); outR[b + 2u] = bitcast<u32>(r.rad.z);
  outR[b + 3u] = bitcast<u32>(r.aux); outR[b + 4u] = r.status; outR[b + 5u] = r.entryTo;
  outR[b + 6u] = bitcast<u32>(r.jp); outR[b + 7u] = r.gen;
}`;

/** Full re-trace (§9.3-4): replay the record's BSDF dimensions from x₁ (V-buffer of its frame, camera of fsFrom) to
 *  x_{d−1} with the path tree's own operations, recompute β_s, then the end under frame fsTo: NEE to the stored light
 *  point (entry translated; the shadow ray ALWAYS traced) or the final BSDF bounce (seed dims of B = d−1). Output per
 *  record: (rad.rgb, aux, bits, entryTo, 0, 0); bits 1 evaluated, 2 visible, 4 x_{d−1} ≠ suffix cache ids, 8 undefined
 *  entry, 16 replay failed, 32 BSDF end ≠ stored end. */
const RETRACE_WGSL = `
#include "restir/refresh.wgsl"
#include "path/path-weight.wgsl"
@group(2) @binding(1) var<storage, read> recAi: array<u32>;
@group(2) @binding(2) var<storage, read_write> outR: array<u32>;
@group(2) @binding(3) var vbufT: texture_2d<u32>;
fn put(i: u32, rad: vec3f, aux: f32, st: u32, e: u32) {
  let b = 8u * i;
  outR[b] = bitcast<u32>(rad.x); outR[b + 1u] = bitcast<u32>(rad.y); outR[b + 2u] = bitcast<u32>(rad.z);
  outR[b + 3u] = bitcast<u32>(aux); outR[b + 4u] = st; outR[b + 5u] = e; outR[b + 6u] = 0u; outR[b + 7u] = 0u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.y * 65535u * 64u + gid.x;
  if (i >= rsDispatch.treeCount) { return; }
  let fsFrom = rsDispatch.round;
  let fsTo = rsDispatch.flags;
  let p1 = resin_plane(i, RP_SEED);
  let d = rf_d(p1.z);
  let k = rf_k(p1.z);
  let tech = rf_tech(p1.z);
  if (d == 0u || k == 0u || k + 1u > d) { put(i, vec3f(0.0), 0.0, 0u, RC_NONE); return; }
  let seed = p1.xy;
  let ai = recAi[i];
  let vb = textureLoad(vbufT, vec2u(ai % rsParams.atlasSize.x, ai / rsParams.atlasSize.x), 0);
  let camPos = lf_cam_pos(fsFrom);
  var cur = vertex_from_ids(vb.x, bitcast<f32>(vb.y), bitcast<f32>(vb.z), camPos);
  var V = normalize(camPos - cur.pos);
  var curPrim = vb.x;
  var beta = vec3f(1.0);
  for (var B = 1u; B + 1u < d; B++) {
    let m = material_eval(cur, V);
    let bs = bsdf_sample(m, V, rs_path_bsdf_u4(seed, B));
    if (!bs.valid) { put(i, vec3f(0.0), 0.0, 16u, RC_NONE); return; }
    let org = offset_ray(cur.pos, select(-cur.ng, cur.ng, dot(cur.ng, bs.L) >= 0.0));
    let h = trace_closest_ex(org, bs.L, FLT_MAX, curPrim, BVH_MISS);
    if (h.primId == BVH_MISS) { put(i, vec3f(0.0), 0.0, 16u, RC_NONE); return; }
    let nxt = vertex_from_ids(h.primId, h.u, h.v, cur.pos);
    let wOut = normalize(nxt.pos - cur.pos);
    let qb = bsdf_query(m, V, wOut, bs.lobe);
    if (B > k) { beta *= rs_path_weight(qb, bs.weight, bs.is_delta); }
    cur = nxt;
    curPrim = h.primId;
    V = -wOut;
  }
  var st = 1u;
  let sfx = resin_plane(i, RP_SFX0);
  let ids = select(sfx.xyz, resin_plane(i, RP_RC).xyz, k + 1u == d && tech != RS_TECH_NEE);
  if (curPrim != ids.x || bitcast<u32>(cur.pos.x) != bitcast<u32>(scene_surface(ids.x, bitcast<f32>(ids.y), bitcast<f32>(ids.z), vec3f(0.0)).pos.x)) { st |= 4u; }
  let slot = lf_slot(fsTo);
  let er = lf_env(fsTo);
  let m = material_eval(cur, V);
  let end = resin_plane(i, RP_END).xyz;
  if (tech == RS_TECH_NEE) {
    let eTo = lt_translate(end.x & RC_ENTRY_MASK, fsFrom, fsTo);
    if (eTo == LIGHT_NONE || !(lt_pmf(eTo, fsTo) > 0.0)) { put(i, vec3f(0.0), 0.0, st | 8u, eTo); return; }
    let ls = nee_eval_s(cur.pos, NeeEndpoint(eTo, end.y, end.z), slot, er);
    let live = ls.valid && any(ls.Lambda > vec3f(0.0));
    var vis = false;
    if (live) { vis = nee_visible(cur, curPrim, ls); }
    if (vis) { st |= 2u; }
    if (k + 1u == d) { put(i, ls.Lambda, ls.p1, st, eTo); return; }
    let qn = bsdf_query(m, V, ls.dir, LOBE_NEE);
    let w1 = nee_mis_w1(ls, qn.p_marg, d - 1u);
    let ok = live && vis && ls.prim != curPrim;
    put(i, select(vec3f(0.0), beta * (w1 / ls.q) * qn.f_all * ls.Lambda, ok), 0.0, st, eTo);
    return;
  }
  let bs = bsdf_sample(m, V, rs_path_bsdf_u4(seed, d - 1u));
  if (!bs.valid) { put(i, vec3f(0.0), 0.0, 16u, RC_NONE); return; }
  let org = offset_ray(cur.pos, select(-cur.ng, cur.ng, dot(cur.ng, bs.L) >= 0.0));
  let h = trace_closest_ex(org, bs.L, FLT_MAX, curPrim, BVH_MISS);
  var L = vec3f(0.0);
  var p1e = 0.0;
  var w2 = 1.0;
  var qbEnd: BsdfQuery;
  if (h.primId == BVH_MISS) {
    if (tech != RS_TECH_BSDF_ENV) { st |= 32u; }
    let qb = bsdf_query(m, V, bs.L, bs.lobe);
    qbEnd = qb;
    L = envRadiance_s(envUV(bs.L, er.cg, er.sg), er);
    p1e = p1Env_s(bs.L, slot, er);
    w2 = env_bsdf_mis_weight_s(bs.L, qb.p_marg, d - 1u, bs.is_delta, slot, er);
  } else {
    if (tech != RS_TECH_BSDF_TRI || h.primId != end.x) { st |= 32u; }
    let z = vertex_from_ids(h.primId, h.u, h.v, cur.pos);
    let qb = bsdf_query(m, V, normalize(z.pos - cur.pos), bs.lobe);
    qbEnd = qb;
    L = tri_emission(h.primId, h.u, h.v);
    p1e = tri_light_p1_s(cur.pos, z.pos, z.ng, h.primId, slot);
    if (!bs.is_delta) { w2 = mis_w2(p1e, qb.p_marg, d - 1u); }
  }
  // the path tree multiplies the last bounce's factor into β_s before a BSDF ending (pathtree.wgsl step 3 → step 5)
  if (d - 1u > k) { beta *= rs_path_weight(qbEnd, bs.weight, bs.is_delta); }
  if (k + 1u == d) { put(i, L, p1e, st, RC_NONE); } else { put(i, beta * w2 * L, 0.0, st, RC_NONE); }
}`;

const layoutCache = new WeakMap<RestirKernel, Map<string, { pl: GPUComputePipeline; l: GPUBindGroupLayout }>>();
async function testPipeline(k: RestirKernel, name: string, src: string, entries: GPUBindGroupLayoutEntry[], defines: Record<string, string | number | boolean>, sources: Record<string, string>) {
  let m = layoutCache.get(k);
  if (!m) { m = new Map(); layoutCache.set(k, m); }
  const key = `${name}:${JSON.stringify(defines)}:${Object.keys(sources).join(',')}`;
  let e = m.get(key);
  if (!e) {
    const l = k.device.createBindGroupLayout({ entries });
    const pl = await k.compile(`${name}.wgsl`, 'main', k.customDefines(defines), k.customLayout(l), name, { ...sources, [`${name}.wgsl`]: src });
    e = { pl, l };
    m.set(key, e);
  }
  return e;
}
const CS = GPUShaderStage?.COMPUTE ?? 4;
const roE = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: CS, buffer: { type: 'read-only-storage' } });
const rwE = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: CS, buffer: { type: 'storage' } });

async function dispatchRecords(k: RestirKernel, pl: GPUComputePipeline, bg: GPUBindGroup, n: number, fsFrom: number, fsTo: number): Promise<void> {
  k.beginSubmit();
  const enc = k.device.createCommandEncoder();
  k.encodeCustom(enc, pl, bg, { treeCount: n, round: fsFrom, flags: fsTo, t: 1 }, [Math.min(Math.ceil(n / 64), 65535), Math.ceil(Math.ceil(n / 64) / 65535)]);
  k.device.queue.submit([enc.finish()]);
  await k.device.queue.onSubmittedWorkDone();
}

/** refresh_record(i, fsFrom, fsTo) of every record → OUT_WORDS words per record. */
async function runRefresh(k: RestirKernel, r: Recs, fsFrom: number, fsTo: number, o: { defines?: Record<string, string | number | boolean>; sources?: Record<string, string> } = {}): Promise<Uint32Array> {
  const { pl, l } = await testPipeline(k, 'rf-test', RF_WGSL, [roE(0), rwE(1), rwE(2)],
    { RS_RES_IN_BINDING: '0u', RS_ARENA_BINDING: '1u', RS_ARENA_RW: true, ...(o.defines ?? {}) }, o.sources ?? {});
  const recB = storageBuffer(k.device, r.rec), out = storageBuffer(k.device, Math.max(r.n, 1) * OUT_WORDS * 4);
  const bg = k.device.createBindGroup({ layout: l, entries: [{ binding: 0, resource: { buffer: recB } }, { binding: 1, resource: { buffer: k.resources.arena } }, { binding: 2, resource: { buffer: out } }] });
  await dispatchRecords(k, pl, bg, r.n, fsFrom, fsTo);
  const u = new Uint32Array(await readBuffer(k.device, out, Math.max(r.n, 1) * OUT_WORDS * 4));
  recB.destroy(); out.destroy();
  return u;
}

/** The full re-trace of every record (vbuf = the V-buffer of the records' own frame). */
async function runRetrace(k: RestirKernel, r: Recs, vbuf: GPUTextureView, fsFrom: number, fsTo: number, sources: Record<string, string> = {}): Promise<Uint32Array> {
  const { pl, l } = await testPipeline(k, 'rf-retrace', RETRACE_WGSL,
    [roE(0), roE(1), rwE(2), { binding: 3, visibility: CS, texture: { sampleType: 'uint' } }], { RS_RES_IN_BINDING: '0u' }, sources);
  const recB = storageBuffer(k.device, r.rec), aiB = storageBuffer(k.device, r.ai), out = storageBuffer(k.device, Math.max(r.n, 1) * OUT_WORDS * 4);
  const bg = k.device.createBindGroup({ layout: l, entries: [
    { binding: 0, resource: { buffer: recB } }, { binding: 1, resource: { buffer: aiB } }, { binding: 2, resource: { buffer: out } }, { binding: 3, resource: vbuf },
  ] });
  await dispatchRecords(k, pl, bg, r.n, fsFrom, fsTo);
  const u = new Uint32Array(await readBuffer(k.device, out, Math.max(r.n, 1) * OUT_WORDS * 4));
  recB.destroy(); aiB.destroy(); out.destroy();
  return u;
}

// ------------------------------------------------------------------------------------------------ frames

/** advance() + only the canonical units of the frame (rs_primary, rs_initial): the refresh tests do not depend on the
 *  temporal stage (T-B). Returns the RestirAdvance. */
async function canonicalFrame(rig: RestirRig, t: number, lights: LightData[], env?: EnvParamsCpu) {
  const k = rig.kernel, device = rig.g.device;
  const adv = k.advance({ t, camera: boxCamera(), lights, env: env ? { params: env, mapId: 'env' } : undefined });
  k.beginSubmit();
  const enc = device.createCommandEncoder();
  for (const u of k.frameUnits(t, { accum: rig.accum, counters: rig.counters })) {
    if (u.label.startsWith('rs_primary') || u.label.startsWith('rs_initial')) u.encode(enc);
  }
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  return adv;
}

const move = (m: Float32Array, dp: V3): Float32Array => { const o = new Float32Array(m); o[12] += dp[0]; o[13] += dp[1]; o[14] += dp[2]; return o; };
const pos = (m: Float32Array): V3 => [m[12], m[13], m[14]];

/** allLightsScene lights of frame 0 and a frame-1 set exercising every change class: point moved, spot rotated (spot
 *  axis: radiometric), rect moved + resized, disk ×2 power, sun removed, a new point light added. */
function changeSets(scene: SceneData): { L0: LightData[]; L1: LightData[]; L1radio: LightData[] } {
  const L0 = scene.lights.map((l) => ({ ...l, matrix: new Float32Array(l.matrix) }));
  const by = (id: number) => L0.find((l) => l.id === id)!;
  const L1: LightData[] = [
    { ...by(1), matrix: move(by(1).matrix, [0.12, -0.05, 0.1]) },
    { ...by(2), matrix: lightMatrixToward([0.4, -1, -0.1], pos(by(2).matrix)) },
    { ...by(3), matrix: move(by(3).matrix, [0.05, 0.1, 0]), sizeX: (by(3).sizeX ?? 1) * 1.5 },
    { ...by(4), power: by(4).power * 2 },
    light({ id: 6, type: 'point', power: 25, matrix: lightMatrixToward([0, -1, 0], [-0.6, 1.5, -0.4]) }),
  ];
  const L1radio: LightData[] = L0.map((l) => (l.id === 3 ? { ...l, power: l.power * 3, color: [1, 0.6, 0.3] as V3 } : l));
  return { L0, L1, L1radio };
}
const ENV0: EnvParamsCpu = { rotationZ: 0.2, strength: 1, tint: [1, 1, 1], visibleToCamera: true };
const ENV1: EnvParamsCpu = { rotationZ: 0.55, strength: 1.5, tint: [1, 0.9, 0.8], visibleToCamera: true };

// ------------------------------------------------------------------------------------------------ canary (Q1)

describe('canary: a known-answer kernel including restir/refresh.wgsl', () => {
  it('refresh_class, refresh_entry within one frame (identity, J_P = 1; LIGHT_NONE undefined), SfxRec round trip', async () => {
    const rig = await restirRig(allLightsScene(), 8, 8, { preset: 'temporal', settings: { maxBounces: 3 } });
    const k = rig.kernel, device = rig.g.device;
    const src = `
#include "restir/refresh.wgsl"
@group(2) @binding(1) var<storage, read_write> outv: array<u32, 32>;
@compute @workgroup_size(1) fn main() {
  outv[0] = 0xC0FFEEu;
  outv[1] = refresh_class(3u, 3u, RS_TECH_NEE);          // L
  outv[2] = refresh_class(3u, 2u, RS_TECH_NEE);          // N1
  outv[3] = refresh_class(3u, 2u, RS_TECH_BSDF_ENV);     // B1
  outv[4] = refresh_class(4u, 4u, RS_TECH_BSDF_TRI);     // E
  outv[5] = refresh_class(5u, 2u, RS_TECH_NEE);          // D-NEE
  outv[6] = refresh_class(5u, 3u, RS_TECH_BSDF_TRI);     // D-BSDF
  outv[7] = refresh_class(5u, 0u, RS_TECH_BSDF_ENV);     // R
  let e = refresh_entry(1u, RS_FS_CUR, RS_FS_CUR);
  outv[8] = e.eTo;
  outv[9] = bitcast<u32>(e.jp);
  outv[10] = select(0u, 1u, e.undef);
  let n = refresh_entry(LIGHT_NONE, RS_FS_CUR, RS_FS_CUR);
  outv[11] = select(0u, 1u, n.undef);
  var r: SfxRec;
  r.rad = vec3f(1.0, 2.0, 3.0); r.aux = 4.0; r.status = SXS_DONE | SXS_VIS; r.entryTo = 7u; r.jp = 0.5; r.gen = 9u;
  sfx_store(SFX_INV, 5u, r);
  let q = sfx_load(SFX_INV, 5u);
  outv[12] = select(0u, 1u, all(q.rad == r.rad) && q.aux == r.aux && q.status == r.status && q.entryTo == 7u && q.jp == 0.5 && q.gen == 9u);
}`;
    // resIn (binding 2) is declared by refresh.wgsl but not read here
    const { pl, l } = await testPipeline(k, 'rf-canary', src, [rwE(0), rwE(1), roE(2)], { RS_RES_IN_BINDING: '2u', RS_ARENA_BINDING: '0u', RS_ARENA_RW: true }, {});
    const out = storageBuffer(device, 128);
    const bg = device.createBindGroup({ layout: l, entries: [
      { binding: 0, resource: { buffer: k.resources.arena } }, { binding: 1, resource: { buffer: out } }, { binding: 2, resource: { buffer: k.resources.res[0] } },
    ] });
    k.beginSubmit();
    const enc = device.createCommandEncoder();
    k.encodeCustom(enc, pl, bg, {}, [1, 1]);
    device.queue.submit([enc.finish()]);
    const u = new Uint32Array(await readBuffer(device, out, 128));
    expect(u[0]).toBe(0xC0FFEE);
    expect(Array.from(u.subarray(1, 8))).toEqual([C.L, C.N1, C.B1, C.E, C.DNEE, C.DBSDF, C.R]);
    expect(u[8]).toBe(1);
    expect(f32(u[9])).toBe(1);
    expect(u[10]).toBe(0);
    expect(u[11]).toBe(1);
    expect(u[12]).toBe(1);
    out.destroy(); rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ U-SFX-2

describe('U-SFX-2: SFX_DELTA_END is set iff the final BSDF event was delta (temporal on); never with temporal off', () => {
  it('allLightsScene 32², maxBounces 5, test rc, 3 frames: every candidate vs lobeHist (d − 1 ≤ 8)', async () => {
    const W = 32, H = 32, P = W * H;
    const rep: Record<string, number> = {};
    for (const temporal of [true, false]) {
      const rig = await restirRig(allLightsScene(), W, H, {
        preset: temporal ? 'temporal' : 'initial', settings: { maxBounces: 5 }, dumpCandidates: true, extraSources: { 'restir/rc.wgsl': TEST_RC_DR },
      });
      let nee = 0, bsdf = 0, deltaEnds = 0, bad = 0;
      for (let f = 0; f < 3; f++) {
        await rig.frames(1, f);
        const r = harvest(await rig.kernel.readCandidateDump(), P);
        for (let i = 0; i < r.n; i++) {
          const fl = rfUnpack(recFlags(r, i));
          if (fl.d === 0) continue;
          const set = (r.rec[i * RES_WORDS + RW.sfxFlags] & K.SFX_DELTA_END) !== 0;
          if (fl.tech === K.RS_TECH_NEE) { nee++; if (set) bad++; continue; }
          const Bm = fl.d - 1;
          if (Bm > 8) continue;
          bsdf++;
          const deltaLast = (lobeHistAt(r.rec[i * RES_WORDS + RW.lobeHist], Bm) & 8) !== 0;
          if (deltaLast) deltaEnds++;
          if (set !== (temporal && deltaLast)) bad++;
        }
      }
      Object.assign(rep, { [`${temporal ? 'on' : 'off'}.nee`]: nee, [`${temporal ? 'on' : 'off'}.bsdf`]: bsdf, [`${temporal ? 'on' : 'off'}.deltaEnds`]: deltaEnds, [`${temporal ? 'on' : 'off'}.bad`]: bad });
      expect(bad).toBe(0);
      expect(deltaEnds).toBeGreaterThan(20);
      rig.destroy();
    }
    console.log(`[U-SFX-2] ${JSON.stringify(rep)}`);
  });
});

// ------------------------------------------------------------------------------------------------ §9.3-3

interface ClassStats { n: number[]; maxRel: number[]; bad: number[] }
const newStats = (): ClassStats => ({ n: Array(7).fill(0), maxRel: Array(7).fill(0), bad: Array(7).fill(0) });
const statsRep = (s: ClassStats) => Object.fromEntries(PATH_CLASS_NAMES.map((c, i) => [c, `${s.n[i]}/${s.maxRel[i].toExponential(1)}/${s.bad[i]}`]));

describe('§9.3-3 / T-ENV-temporal: refresh idempotence on an unchanged frame — rad / aux ≡ the stored cache (analytic, triangles, env)', () => {
  for (const rcMode of ['test-rc (D∧R)', 'production rc'] as const) {
    it(`${rcMode}: allLightsScene 32², maxBounces 5, 3 frames; with and without forced rays (visibility rule)`, async () => {
      const W = 32, H = 32, P = W * H;
      const sources: Record<string, string> = rcMode === 'test-rc (D∧R)' ? { 'restir/rc.wgsl': TEST_RC_DR } : {};
      const rig = await restirRig(allLightsScene(), W, H, { preset: 'temporal', settings: { maxBounces: 5 }, dumpCandidates: true, extraSources: sources });
      const k = rig.kernel;
      const st = newStats();
      let entryBad = 0, jpBad = 0, undef = 0, rays = 0, forcedRays = 0, visFail = 0, statusBad = 0;
      for (let f = 0; f < 3; f++) {
        await rig.frames(1, 7 + f);
        const r = harvest(await k.readCandidateDump(), P);
        for (const forced of [false, true]) {
          const o = await runRefresh(k, r, K.RS_FS_CUR, K.RS_FS_CUR, { defines: forced ? { RS_REFRESH_FORCE_MOVED: true } : {}, sources });
          const of = new Float32Array(o.buffer);
          for (let i = 0; i < r.n; i++) {
            const cls = recClass(r, i);
            if (cls < 0) continue;
            const b = OUT_WORDS * i, status = o[b + 4], base = i * RES_WORDS;
            if ((status & S.SXS_DONE) === 0) statusBad++;
            if (status & S.SXS_UNDEF) { undef++; continue; }
            if (status & S.SXS_RAY) { if (forced) forcedRays++; else rays++; }
            const fl = rfUnpack(recFlags(r, i));
            if (fl.tech === K.RS_TECH_NEE) {
              if (o[b + 5] !== (r.rec[base + RW.end] & K.RC_ENTRY_MASK)) entryBad++;
              if (of[b + 6] !== 1) jpBad++;
            } else if (o[b + 5] !== K.RC_NONE || of[b + 6] !== 1) jpBad++;
            if (cls === C.DNEE || cls === C.DBSDF) {
              const e = rel3(of, new Float32Array(r.rec.buffer), b, base + RW.rcRad);
              st.n[cls]++; st.maxRel[cls] = Math.max(st.maxRel[cls], e); if (e > 1e-6) st.bad[cls]++;
            } else if (cls === C.N1 || cls === C.B1) {
              const e = Math.max(rel3(of, new Float32Array(r.rec.buffer), b, base + RW.rcRad), rel(of[b + 3], recF(r, i, RW.aux)));
              st.n[cls]++; st.maxRel[cls] = Math.max(st.maxRel[cls], e); if (e > 1e-6) st.bad[cls]++;
              if (cls === C.N1 && (status & S.SXS_VIS) === 0) visFail++;
            } else if (of[b] !== 0 || of[b + 1] !== 0 || of[b + 2] !== 0) st.bad[cls]++;
          }
        }
      }
      const rep = { classes: statsRep(st), entryBad, jpBad, undef, rays, forcedRays, visFail, statusBad };
      console.log(`[§9.3-3 ${rcMode}] ${JSON.stringify(rep)}`);
      expect(st.bad.every((x) => x === 0)).toBe(true);
      expect(entryBad + jpBad + undef + rays + visFail + statusBad).toBe(0);
      expect(forcedRays).toBeGreaterThan(0);
      if (rcMode === 'test-rc (D∧R)') for (const c of [C.N1, C.B1, C.DNEE, C.DBSDF]) expect(st.n[c], PATH_CLASS_NAMES[c]).toBeGreaterThan(100);
      rig.destroy();
    });
  }
});

// ------------------------------------------------------------------------------------------------ §9.3-4

interface CmpRep {
  classes: Record<string, string>; undefRefresh: number; undefMismatch: number; entryBad: number; visMismatch: number; replayFail: number;
  idsMismatch: number; endMismatch: number; rays: number; bad: number[]; n: number[];
}

/** Compare the refresh output with the full re-trace, per class (tolerance 1e-4 relative). */
function compareRetrace(r: Recs, o: Uint32Array, ref: Uint32Array): CmpRep {
  const of = new Float32Array(o.buffer), rf = new Float32Array(ref.buffer);
  const st = newStats();
  const rep = { undefRefresh: 0, undefMismatch: 0, entryBad: 0, visMismatch: 0, replayFail: 0, idsMismatch: 0, endMismatch: 0, rays: 0 };
  for (let i = 0; i < r.n; i++) {
    const cls = recClass(r, i);
    if (!(cls === C.N1 || cls === C.B1 || cls === C.DNEE || cls === C.DBSDF)) continue;
    const b = OUT_WORDS * i, status = o[b + 4], bits = ref[b + 4];
    if (status & S.SXS_RAY) rep.rays++;
    if (bits & 16) { rep.replayFail++; continue; }
    if (bits & 4) rep.idsMismatch++;
    if (bits & 32) rep.endMismatch++;
    const u = (status & S.SXS_UNDEF) !== 0;
    if (u) rep.undefRefresh++;
    if (u !== ((bits & 8) !== 0)) { rep.undefMismatch++; continue; }
    if (u) continue;
    const nee = rfUnpack(recFlags(r, i)).tech === K.RS_TECH_NEE;
    if (nee && o[b + 5] !== ref[b + 5]) rep.entryBad++;
    let e = rel3(of, rf, b, b);
    if (cls === C.N1 || cls === C.B1) e = Math.max(e, rel(of[b + 3], rf[b + 3]));
    if (cls === C.N1 && ((status & S.SXS_VIS) !== 0) !== ((bits & 2) !== 0)) rep.visMismatch++;
    st.n[cls]++; st.maxRel[cls] = Math.max(st.maxRel[cls], e); if (e > 1e-4) st.bad[cls]++;
  }
  return { classes: statsRep(st), ...rep, bad: st.bad, n: st.n };
}

describe('§9.3-4 / T-ENV-temporal: cached refresh ≡ full re-trace on light- and env-change frames, forward and inverse (≤ 1e-4)', () => {
  for (const variant of ['every change class + env rotation/strength/tint', 'radiometric only (visibility rule: no rays)', 'point light moved only (analytic rays)'] as const) {
    it(variant, async () => {
      const W = 32, H = 32, P = W * H;
      const scene = allLightsScene();
      const sources = { 'restir/rc.wgsl': TEST_RC_DR };
      const rig = await restirRig(scene, W, H, { preset: 'temporal', settings: { maxBounces: 5 }, dumpCandidates: true, extraSources: sources, seed: 31 });
      const k = rig.kernel;
      const { L0, L1, L1radio } = changeSets(scene);
      const all = variant.startsWith('every'), pointOnly = variant.startsWith('point');
      const L1point = L0.map((l) => (l.id === 1 ? { ...l, matrix: move(l.matrix, [0.12, -0.05, 0.1]) } : l));
      const a0 = await canonicalFrame(rig, 0, L0, ENV0);
      expect(a0.histValid).toBe(false);
      const r0 = harvest(await k.readCandidateDump(), P);
      const a1 = await canonicalFrame(rig, 1, all ? L1 : pointOnly ? L1point : L1radio, all ? ENV1 : ENV0);
      if (pointOnly) expect(a1.flags & K.TF_LIGHT_MOVED).not.toBe(0);
      expect(a1.histValid).toBe(true);
      expect(a1.flags & K.TF_REFRESH).not.toBe(0);
      expect(a1.flags & K.TF_LIGHTS_SAME).toBe(0);
      if (all) expect(a1.flags & (K.TF_ENV_MOVED | K.TF_ENV_RADIO)).toBe(K.TF_ENV_MOVED | K.TF_ENV_RADIO);
      const r1 = harvest(await k.readCandidateDump(), P);
      const fwd = compareRetrace(r0, await runRefresh(k, r0, K.RS_FS_PREV, K.RS_FS_CUR, { sources }), await runRetrace(k, r0, k.resources.views.vbufPrev, K.RS_FS_PREV, K.RS_FS_CUR, sources));
      const inv = compareRetrace(r1, await runRefresh(k, r1, K.RS_FS_CUR, K.RS_FS_PREV, { sources }), await runRetrace(k, r1, k.resources.views.vbuf, K.RS_FS_CUR, K.RS_FS_PREV, sources));
      const strip = ({ bad, n, ...rest }: CmpRep) => { void bad; void n; return rest; };
      console.log(`[§9.3-4 ${variant}] fwd ${JSON.stringify(strip(fwd))} inv ${JSON.stringify(strip(inv))}`);
      for (const x of [fwd, inv]) {
        expect(x.bad.every((b) => b === 0)).toBe(true);
        expect(x.undefMismatch + x.entryBad + x.visMismatch + x.replayFail + x.idsMismatch + x.endMismatch).toBe(0);
        for (const c of [C.N1, C.B1, C.DNEE, C.DBSDF]) expect(x.n[c], PATH_CLASS_NAMES[c]).toBeGreaterThan(50);
        if (all) { expect(x.rays).toBeGreaterThan(0); expect(x.undefRefresh).toBeGreaterThan(0); } else if (pointOnly) expect(x.rays).toBeGreaterThan(0); else expect(x.rays).toBe(0);
      }
      rig.destroy();
    });
  }
});

// ------------------------------------------------------------------------------------------------ passes

describe('passes: rs_refresh_fwd / rs_args(q2) + rs_refresh_inv store refresh_record of res[h] / Q_i into sfxOut', () => {
  it('change frame, 3 row bands (3 item chunks for Q_i); non-refresh frame writes nothing', async () => {
    const W = 32, H = 24, P = W * H;
    const scene = allLightsScene();
    const rig = await restirRig(scene, W, H, { preset: 'temporal', settings: { maxBounces: 4 }, seed: 5 });
    const k = rig.kernel, device = rig.g.device;
    k.rowBand = 8;
    const { L0, L1 } = changeSets(scene);
    await canonicalFrame(rig, 0, L0, ENV0);
    const a1 = await canonicalFrame(rig, 1, L1, ENV1);
    const h = k.historyIndex(), w = k.resBase();
    expect(h).toBeGreaterThanOrEqual(0);
    const NS = k.resources.alloc.slots;
    // Q_i = every pixel (items (ai << 3) at the q2 item base; counter = P)
    const items = new Uint32Array(P).map((_, i) => i << 3);
    device.queue.writeBuffer(k.resources.arena, 4 * (64 + 5 * P * NS + P), items);
    device.queue.writeBuffer(k.resources.arena, 4 * 4 * K.RS_Q_INV, new Uint32Array([P, 0]));
    const { refreshFwdUnits, refreshInvUnits } = await import('../../src/core/render/restir/refresh.ts');
    const units = [...refreshFwdUnits(k, 1, a1.flags), ...refreshInvUnits(k, 1, a1.flags)];
    expect(units.length).toBe(2 * 3);
    k.beginSubmit();
    const enc = device.createCommandEncoder();
    for (const u of units) u.encode(enc);
    device.queue.submit([enc.finish()]);
    const ts = await k.readTemporalState();
    const { decodeSfxLocal } = await import('../../src/core/render/restir/layout.ts');
    const recs = async (idx: number): Promise<Recs> => ({ rec: await k.readReservoirs(idx as 0 | 1), ai: new Uint32Array(P).map((_, i) => i), n: P });
    const expFwd = await runRefresh(k, await recs(h), K.RS_FS_PREV, K.RS_FS_CUR);
    const expInv = await runRefresh(k, await recs(w), K.RS_FS_CUR, K.RS_FS_PREV);
    let bad = 0, refreshed = 0;
    const badDir = [0, 0], firstBad: string[] = [];
    for (let ai = 0; ai < P; ai++) {
      for (const [dir, exp] of [[S.SFX_FWD, expFwd], [S.SFX_INV, expInv]] as const) {
        const s = decodeSfxLocal(ts, P, NS, dir, ai);
        const e = exp.subarray(OUT_WORDS * ai, OUT_WORDS * ai + OUT_WORDS);
        const ef = new Float32Array(e.slice().buffer);
        // words exact; floats ≤ 1e-6 relative (the production pass and the test kernel are separate pipelines: FMA contraction)
        if (s.status !== e[4] || s.entryTo !== e[5] || s.gen !== e[7] || rel(s.jp, ef[6]) > 1e-6 || rel(s.aux, ef[3]) > 1e-6 || s.rad.some((x, c) => rel(x, ef[c]) > 1e-6)) {
          bad++; badDir[dir]++;
          if (firstBad.filter((x) => x.startsWith(`${dir}:`)).length < 3) firstBad.push(`${dir}:${ai} got ${JSON.stringify(s)} exp ${Array.from(e).join(',')}`);
        }
        if (s.status) refreshed++;
      }
    }
    console.log(`[passes] refreshed ${refreshed} / ${2 * P}, mismatches ${bad} (fwd ${badDir[0]}, inv ${badDir[1]}), gen ${a1.temporal.frameGen}; ${firstBad.join(' | ')}`);
    expect(bad).toBe(0);
    expect(refreshed).toBeGreaterThan(P);
    // a frame without TF_REFRESH (nothing changed) leaves sfxOut untouched even when the units are encoded
    const a2 = k.advance({ t: 2, camera: boxCamera(), lights: L1, env: { params: ENV1, mapId: 'env' } });
    expect(a2.flags & K.TF_REFRESH).toBe(0);
    expect(refreshFwdUnits(k, 2, a2.flags)).toEqual([]);
    expect(refreshInvUnits(k, 2, a2.flags)).toEqual([]);
    k.beginSubmit();
    const enc2 = device.createCommandEncoder();
    const force = K.TF_REFRESH | K.TF_HIST_VALID;          // encode anyway: the passes check RsTemporal.flags themselves
    for (const u of refreshFwdUnits(k, 2, force).concat(refreshInvUnits(k, 2, force))) u.encode(enc2);
    device.queue.submit([enc2.finish()]);
    const ts2 = await k.readTemporalState();
    expect(ts2).toEqual(ts);
    rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ E2 and plants

describe('E2 (TD23) and the refresh plants on a change frame', () => {
  it('E2, N3, N7, NO_JP, NO_JP_ENV, ENV_NO_ROT_VIS, N4 against the exact refresh', async () => {
    const W = 32, H = 32, P = W * H;
    const scene = allLightsScene();
    const sources = { 'restir/rc.wgsl': TEST_RC_DR };
    const rig = await restirRig(scene, W, H, { preset: 'temporal', settings: { maxBounces: 5 }, dumpCandidates: true, extraSources: sources, seed: 41 });
    const k = rig.kernel;
    const { L0, L1 } = changeSets(scene);
    await canonicalFrame(rig, 0, L0, ENV0);
    const r0 = harvest(await k.readCandidateDump(), P);
    await canonicalFrame(rig, 1, L1, ENV1);
    const run = async (s: Partial<RestirSettings>, fsFrom: number = K.RS_FS_PREV, fsTo: number = K.RS_FS_CUR) => {
      k.setSettings({ refresh: 'exact', temporalCheck: 'none', tPlant: {}, ...s });
      return runRefresh(k, r0, fsFrom, fsTo, { sources });
    };
    const base = await run({});
    const recs = new Float32Array(r0.rec.buffer);
    const word = (o: Uint32Array, i: number, w: number) => o[OUT_WORDS * i + w];
    const same = (a: Uint32Array, b: Uint32Array, i: number) => a.subarray(OUT_WORDS * i, OUT_WORDS * i + 7).every((x, j) => x === b[OUT_WORDS * i + j]);
    const cnt = { e2: 0, e2Bad: 0, n3: 0, n3Bad: 0, n7Stored: 0, n7Refreshed: 0, n7Bad: 0, noJpBad: 0, noJpEnvBad: 0, envNee: 0, rotVisBad: 0, n4: 0, n4Diff: 0, n4Bad: 0 };
    const cls = (i: number) => recClass(r0, i);
    const deep = (c: number) => c === C.DNEE || c === C.DBSDF;
    const cached = (c: number) => deep(c) || c === C.N1 || c === C.B1;
    const fl = (i: number) => rfUnpack(recFlags(r0, i));
    const undef = (o: Uint32Array, i: number) => (word(o, i, 4) & S.SXS_UNDEF) !== 0;

    const e2 = await run({ refresh: 'e2' });
    for (let i = 0; i < r0.n; i++) {
      if (cls(i) < 0) continue;
      if (deep(cls(i)) && !undef(base, i)) { cnt.e2++; if (!(word(e2, i, 4) & S.SXS_E2) || !undef(e2, i) || word(e2, i, 5) !== word(base, i, 5)) cnt.e2Bad++; }
      else if (!same(e2, base, i)) cnt.e2Bad++;
    }
    const n3 = await run({ tPlant: { n3Stale: true } });
    for (let i = 0; i < r0.n; i++) {
      const c = cls(i);
      if (c < 0 || undef(base, i)) continue;
      if (!cached(c)) { if (!same(n3, base, i)) cnt.n3Bad++; continue; }
      cnt.n3++;
      const b = OUT_WORDS * i, rb = i * RES_WORDS;
      if ([0, 1, 2].some((j) => n3[b + j] !== r0.rec[rb + RW.rcRad + j]) || (!deep(c) && n3[b + 3] !== r0.rec[rb + RW.aux])) cnt.n3Bad++;
      if (word(n3, i, 4) & S.SXS_RAY) cnt.n3Bad++;
      if (word(n3, i, 5) !== word(base, i, 5) || word(n3, i, 6) !== word(base, i, 6)) cnt.n3Bad++;
    }
    // N7: in L1 every surviving analytic light and the env changed; emissive triangles did not (endpoint type LT_TRI)
    const n7 = await run({ tPlant: { n7PerLight: true } });
    for (let i = 0; i < r0.n; i++) {
      const c = cls(i);
      if (c < 0 || undef(base, i) || !cached(c)) continue;
      const tri = fl(i).ep === 0;
      const b = OUT_WORDS * i, rb = i * RES_WORDS;
      if (tri) {
        cnt.n7Stored++;
        if ([0, 1, 2].some((j) => n7[b + j] !== r0.rec[rb + RW.rcRad + j])) cnt.n7Bad++;
      } else {
        cnt.n7Refreshed++;
        if (!same(n7, base, i)) cnt.n7Bad++;
      }
    }
    const noJp = await run({ tPlant: { noJP: true } });
    const noJpEnv = await run({ tPlant: { noJPEnv: true } });
    for (let i = 0; i < r0.n; i++) {
      if (cls(i) < 0 || fl(i).tech !== K.RS_TECH_NEE) continue;
      if (f32(word(noJp, i, 6)) !== 1) cnt.noJpBad++;
      const env = fl(i).ep === 7;
      if (env ? f32(word(noJpEnv, i, 6)) !== 1 : word(noJpEnv, i, 6) !== word(base, i, 6)) cnt.noJpEnvBad++;
    }
    const rotVis = await run({ tPlant: { envNoRotVis: true } });
    for (let i = 0; i < r0.n; i++) {
      const c = cls(i);
      if (c < 0 || undef(base, i)) continue;
      const envNee = fl(i).tech === K.RS_TECH_NEE && fl(i).ep === 7 && (c === C.N1 || c === C.DNEE);
      if (envNee) { cnt.envNee++; if (word(rotVis, i, 4) & S.SXS_RAY) cnt.rotVisBad++; if (c === C.N1 && !(word(rotVis, i, 4) & S.SXS_VIS)) cnt.rotVisBad++; }
      else if (!same(rotVis, base, i)) cnt.rotVisBad++;
    }
    const n4a = await run({ tPlant: { n4Ris: true } });
    const n4b = await run({ tPlant: { n4Ris: true } });
    const n4inv = await run({ tPlant: { n4Ris: true } }, K.RS_FS_CUR, K.RS_FS_PREV);
    const inv0 = await run({}, K.RS_FS_CUR, K.RS_FS_PREV);
    for (let i = 0; i < r0.n; i++) {
      const c = cls(i);
      if (c < 0) continue;
      if (!same(n4a, n4b, i)) cnt.n4Bad++;                      // deterministic (frame-t hashes of the pixel)
      if (!same(n4inv, inv0, i)) cnt.n4Bad++;                   // forward only
      if (c === C.DNEE && !undef(base, i)) {
        cnt.n4++;
        if (!(word(n4a, i, 4) & S.SXS_PLANT)) cnt.n4Bad++;
        if ([0, 1, 2].some((j) => word(n4a, i, j) !== word(base, i, j))) cnt.n4Diff++;
      } else if (!same(n4a, base, i)) cnt.n4Bad++;
    }
    // C-8: the robust check's inverse (TM_ROBUST, CUR → PREV) is always fresh; the forward stays stale
    let robustBad = 0;
    const robN3inv = await run({ temporalCheck: 'robust', tPlant: { n3Stale: true } }, K.RS_FS_CUR, K.RS_FS_PREV);
    const robN7inv = await run({ temporalCheck: 'robust', tPlant: { n7PerLight: true } }, K.RS_FS_CUR, K.RS_FS_PREV);
    const robN3fwd = await run({ temporalCheck: 'robust', tPlant: { n3Stale: true } });
    for (let i = 0; i < r0.n; i++) {
      if (cls(i) < 0) continue;
      if (!same(robN3inv, inv0, i) || !same(robN7inv, inv0, i) || !same(robN3fwd, n3, i)) robustBad++;
    }
    void recs;
    console.log(`[plants] ${JSON.stringify({ ...cnt, robustBad })}`);
    expect(robustBad).toBe(0);
    expect(cnt.e2Bad + cnt.n3Bad + cnt.n7Bad + cnt.noJpBad + cnt.noJpEnvBad + cnt.rotVisBad + cnt.n4Bad).toBe(0);
    for (const x of [cnt.e2, cnt.n3, cnt.n7Stored, cnt.n7Refreshed, cnt.envNee, cnt.n4, cnt.n4Diff]) expect(x).toBeGreaterThan(10);
    rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ §9.3-2 (a)

/** Light-local map Φ_s(e, u, v) at 8 small (u, v) triangles, for every analytic / triangle entry of frame t−1 and its
 *  translated entry in frame t: positions (f32) → the CPU forms the triangle areas in f64. */
const FD_WGSL = `
#include "restir/tframe.wgsl"
#include "lights/measure.wgsl"
@group(2) @binding(0) var<storage, read_write> outF: array<f32>;
@group(2) @binding(1) var<storage, read_write> outK: array<u32>;
const FD_EPS: f32 = 0.2;
@compute @workgroup_size(1) fn main() {
  let sp = lf_slot(RS_FS_PREV);
  let sc = lf_slot(RS_FS_CUR);
  let n = min(sp.nAnalytic + lightsParams.triCount, 16u);
  outK[0] = n;
  let x0 = vec3f(0.1, 0.4, -0.3);
  for (var e = 0u; e < n; e++) {
    let ec = lt_translate(e, RS_FS_PREV, RS_FS_CUR);
    var kind = LT_TRI;
    if (e < sp.nAnalytic) { kind = light_load(sp, e).kind; }
    outK[1u + 2u * e] = kind;
    outK[2u + 2u * e] = ec;
    for (var j = 0u; j < 8u; j++) {
      let u0 = vec2f(0.05 + 0.09 * f32(j), 0.7 - 0.08 * f32(j));
      for (var c = 0u; c < 3u; c++) {
        let uv = u0 + select(select(vec2f(0.0), vec2f(0.0, FD_EPS), c == 2u), vec2f(FD_EPS, 0.0), c == 1u);
        let a = light_sample_entry(x0, sp, e, uv).pos;
        var b = vec3f(0.0);
        if (ec != LIGHT_NONE) { b = light_sample_entry(x0, sc, ec, uv).pos; }
        let o = ((e * 8u + j) * 3u + c) * 6u;
        outF[o] = a.x; outF[o + 1u] = a.y; outF[o + 2u] = a.z; outF[o + 3u] = b.x; outF[o + 4u] = b.y; outF[o + 5u] = b.z;
      }
    }
  }
}`;

describe('§9.3-2 (a): FD area ratio of the light-local map Φ_t ∘ Φ_{t−1}⁻¹ = A_t/A_{t−1} (rect, disk, emissive triangle)', () => {
  it('rect ×1.5 in X + moved + rotated, disk ⌀ ×1.2 + moved, triangles static: |ratio − A_t/A_{t−1}| ≤ 1e-5', async () => {
    const scene = allLightsScene();
    const rig = await restirRig(scene, 8, 8, { preset: 'temporal', settings: { maxBounces: 3 } });
    const k = rig.kernel, device = rig.g.device;
    const L0 = scene.lights;
    const by = (id: number) => L0.find((l) => l.id === id)!;
    const L1 = L0.map((l) => {
      if (l.id === 3) return { ...l, sizeX: (l.sizeX ?? 1) * 1.5, matrix: lightMatrixToward([0.2, -0.4, 1], [0.35, 1.25, -1.85]) };
      if (l.id === 4) return { ...l, sizeX: (l.sizeX ?? 1) * 1.2, matrix: move(l.matrix, [0.1, 0, 0.05]) };
      return l;
    });
    void by;
    await canonicalFrame(rig, 0, L0);
    const adv = await canonicalFrame(rig, 1, L1);
    expect(adv.histValid).toBe(true);
    const outF = storageBuffer(device, 16 * 8 * 3 * 6 * 4), outK = storageBuffer(device, 64 * 4);
    const l = device.createBindGroupLayout({ entries: [rwE(0), rwE(1)] });
    const pl = await k.compile('rf-fd.wgsl', 'main', k.customDefines({}), k.customLayout(l), 'rf-fd', { 'rf-fd.wgsl': FD_WGSL });
    const bg = device.createBindGroup({ layout: l, entries: [{ binding: 0, resource: { buffer: outF } }, { binding: 1, resource: { buffer: outK } }] });
    k.beginSubmit();
    const enc = device.createCommandEncoder();
    k.encodeCustom(enc, pl, bg, {}, [1, 1]);
    device.queue.submit([enc.finish()]);
    const F = new Float32Array(await readBuffer(device, outF, 16 * 8 * 3 * 6 * 4));
    const Ku = new Uint32Array(await readBuffer(device, outK, 64 * 4));
    const area = (o: number, off: number) => {
      const p = [0, 1, 2].map((c) => [F[o + c * 6 + off], F[o + c * 6 + off + 1], F[o + c * 6 + off + 2]]);
      const a = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]], b = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
      return 0.5 * Math.hypot(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
    };
    const expected: Record<number, number> = { 3: 1.5, 4: 1.44, 0: 1 };
    const seen: Record<number, { n: number; maxErr: number }> = {};
    for (let e = 0; e < Ku[0]; e++) {
      const kind = Ku[1 + 2 * e];
      if (!(kind in expected)) continue;
      expect(Ku[2 + 2 * e]).not.toBe(0xFFFFFFFF);
      for (let j = 0; j < 8; j++) {
        const o = (e * 8 + j) * 3 * 6;
        const r = area(o, 3) / area(o, 0);
        const s = (seen[kind] ??= { n: 0, maxErr: 0 });
        s.n++; s.maxErr = Math.max(s.maxErr, Math.abs(r - expected[kind]) / expected[kind]);
      }
    }
    console.log(`[§9.3-2a] ${JSON.stringify(seen)}`);
    for (const kind of [0, 3, 4]) { expect(seen[kind]?.n ?? 0).toBeGreaterThan(0); expect(seen[kind].maxErr).toBeLessThanOrEqual(1e-5); }
    outF.destroy(); outK.destroy(); rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ §9.3-2 (b)

/** PSS change of variables of the NEE light dimensions (gap-temporal §9.3-2, §6): side A draws x ∼ the frame-(t−1)
 *  sampler and scores h(T x)·J_P·1{x ∈ D}; side B draws y ∼ the frame-t sampler and scores h(y)·1{y ∈ I}. h is a bounded
 *  function of the light point under frame t. Per thread: Kahan sums of x and x² for both sides. */
const PSS_WGSL = `
#include "restir/refresh.wgsl"
@group(2) @binding(2) var<storage, read_write> outS: array<f32>;
fn htest(e: u32, a: u32, b: u32) -> f32 {
  let slot = lf_slot(RS_FS_CUR);
  let ep = NeeEndpoint(e, a, b);
  let ls = nee_eval_s(vec3f(0.1, 0.5, -0.4), ep, slot, lf_env(RS_FS_CUR));
  var t = 0.0;
  if (ls.valid) { t = luminance(ls.Lambda); }
  return (1.0 + t / (1.0 + t)) * (1.0 + 0.1 * f32(nee_endpoint_id_s(ep, slot) % 7u));
}
fn kahan(s: ptr<function, vec2f>, x: f32) { let y = x - (*s).y; let t = (*s).x + y; (*s).y = (t - (*s).x) - y; (*s).x = t; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3u) {
  let tid = gid.x;
  if (tid >= rsDispatch.treeCount) { return; }
  var sa = vec2f(0.0); var sa2 = vec2f(0.0); var sb = vec2f(0.0); var sb2 = vec2f(0.0);
  for (var s = 0u; s < rsDispatch.round; s++) {
    let hA = pcg4d(vec4u(tid, s, 0xA11CE5u, rsDispatch.t));
    let hA2 = pcg4d(vec4u(tid, s, 0xA11CE6u, rsDispatch.t));
    let x = nee_draw(lf_slot(RS_FS_PREV), hA.x, hA.y, vec3u(hA.z, hA.w, hA2.x));
    var va = 0.0;
    if (x.entry != LIGHT_NONE) {
      let te = refresh_entry(x.entry, RS_FS_PREV, RS_FS_CUR);
      if (!te.undef) { va = htest(te.eTo, x.a, x.b) * te.jp; }
    }
    let hB = pcg4d(vec4u(tid, s, 0xB0B5u, rsDispatch.t));
    let hB2 = pcg4d(vec4u(tid, s, 0xB0B6u, rsDispatch.t));
    let y = nee_draw(lf_slot(RS_FS_CUR), hB.x, hB.y, vec3u(hB.z, hB.w, hB2.x));
    var vb = 0.0;
    if (y.entry != LIGHT_NONE && !refresh_entry(y.entry, RS_FS_CUR, RS_FS_PREV).undef) { vb = htest(y.entry, y.a, y.b); }
    kahan(&sa, va); kahan(&sa2, va * va); kahan(&sb, vb); kahan(&sb2, vb * vb);
  }
  outS[4u * tid] = sa.x; outS[4u * tid + 1u] = sa2.x; outS[4u * tid + 2u] = sb.x; outS[4u * tid + 3u] = sb2.x;
}`;

async function pssZ(k: RestirKernel, samples: number): Promise<{ z: number; mA: number; mB: number; n: number }> {
  const device = k.device;
  const T = 24576, N = 256, batches = Math.ceil(samples / (T * N));
  const { pl, l } = await testPipeline(k, 'rf-pss', PSS_WGSL, [roE(0), rwE(1), rwE(2)], { RS_RES_IN_BINDING: '0u', RS_ARENA_BINDING: '1u', RS_ARENA_RW: true }, {});
  const out = storageBuffer(device, T * 16);
  const bg = device.createBindGroup({ layout: l, entries: [
    { binding: 0, resource: { buffer: k.resources.res[0] } }, { binding: 1, resource: { buffer: k.resources.arena } }, { binding: 2, resource: { buffer: out } },
  ] });
  let a = 0, a2 = 0, b = 0, b2 = 0;
  for (let bi = 0; bi < batches; bi++) {
    k.beginSubmit();
    const enc = device.createCommandEncoder();
    k.encodeCustom(enc, pl, bg, { t: 1000 + bi, round: N, treeCount: T }, [T / 64, 1]);
    device.queue.submit([enc.finish()]);
    const f = new Float32Array(await readBuffer(device, out, T * 16));
    for (let i = 0; i < T; i++) { a += f[4 * i]; a2 += f[4 * i + 1]; b += f[4 * i + 2]; b2 += f[4 * i + 3]; }
  }
  out.destroy();
  const n = batches * T * N;
  const mA = a / n, mB = b / n;
  const vA = a2 / n - mA * mA, vB = b2 / n - mB * mB;
  return { z: (mA - mB) / Math.sqrt(vA / n + vB / n), mA, mB, n };
}

describe('§9.3-2 (b) / T-ENV-temporal: PSS change of variables E_prev[h(Tx)·J_P·1_D] = E_cur[h(y)·1_I] (10⁸ samples, |z| ≤ 4)', () => {
  const twoLights = (pA: number) => [
    light({ id: 1, type: 'point', power: pA, matrix: lightMatrixToward([0, -1, 0], [0.4, 1.7, -0.6]) }),
    light({ id: 2, type: 'rect', power: 60, sizeX: 0.5, sizeY: 0.3, matrix: lightMatrixToward([0, -1, 0], [-0.4, 1.9, -1.0]) }),
  ];
  const cases: { name: string; scene: () => SceneData; f0: (s: SceneData) => [LightData[], EnvParamsCpu | undefined]; f1: (s: SceneData) => [LightData[], EnvParamsCpu | undefined] }[] = [
    { name: 'two lights, intensity ×2 on A', scene: () => boxScene(twoLights(60)), f0: () => [twoLights(60), undefined], f1: () => [twoLights(120), undefined] },
    {
      name: 'all light types + env: moves, add, remove, ×2 power, env strength ×1.5 / tint / rotation', scene: allLightsScene,
      f0: (s) => [changeSets(s).L0, ENV0], f1: (s) => [changeSets(s).L1, ENV1],
    },
    { name: 'env strength only (allLightsScene)', scene: allLightsScene, f0: (s) => [s.lights, ENV0], f1: (s) => [s.lights, { ...ENV0, strength: 2.5 }] },
  ];
  for (const c of cases) {
    it(c.name, async () => {
      const scene = c.scene();
      const rig = await restirRig(scene, 8, 8, { preset: 'temporal', settings: { maxBounces: 3 } });
      const k = rig.kernel;
      const [l0, e0] = c.f0(scene), [l1, e1] = c.f1(scene);
      await canonicalFrame(rig, 0, l0, e0);
      const adv = await canonicalFrame(rig, 1, l1, e1);
      expect(adv.histValid).toBe(true);
      expect(adv.flags & K.TF_PMF_CHANGED).not.toBe(0);
      const ok = await pssZ(k, 1e8);
      k.setSettings({ tPlant: { noJP: true } });                // N2: J_P omitted must be detected
      const n2 = await pssZ(k, 2.5e7);
      console.log(`[§9.3-2b ${c.name}] exact z=${ok.z.toFixed(2)} mA=${ok.mA.toFixed(6)} mB=${ok.mB.toFixed(6)} n=${ok.n}; N2 z=${n2.z.toFixed(1)}`);
      expect(Math.abs(ok.z)).toBeLessThanOrEqual(4);
      expect(Math.abs(n2.z)).toBeGreaterThan(8);
      rig.destroy();
    });
  }
});

// ------------------------------------------------------------------------------------------------ interactive path

describe('interactive RestirFramePass: a moving point light is refreshed with shadow rays (m5 app smoke follow-up)', () => {
  it('c0c-like box, point light moved every frame through setLights + advanceInteractive: refresh rays > 0 on history frames', async () => {
    const pointAt = (p: V3) => [light({ id: 1, type: 'point', power: 60, matrix: lightMatrixToward([0, -1, 0], p) }),
      light({ id: 2, type: 'rect', power: 40, sizeX: 0.5, sizeY: 0.3, matrix: lightMatrixToward([0, -1, 0], [-0.4, 1.9, -1.0]) })];
    const scene = boxScene(pointAt([0.2, 1.7, -0.5]));
    const { gpuScene } = await import('./restir-fixtures.ts');
    const g = await gpuScene(scene);
    const dev = g.device;
    const { FrameUniformBuffer, JITTER_IID } = await import('../../src/core/render/frame-uniforms.ts');
    const { RestirKernel } = await import('../../src/core/render/restir/kernel.ts');
    const { restirSettings } = await import('../../src/core/render/restir/presets.ts');
    const W = 64, H = 48;
    const pass = await RestirKernel.interactive(dev, g.gpu, g.env, 'rgba16float', { settings: restirSettings('interactive', { maxBounces: 3 }), features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
    const color = dev.createTexture({ size: [W, H], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
    const fu = new FrameUniformBuffer(dev);
    pass.setTargets({ width: W, height: H, color, frameUniforms: fu.buffer });
    const k = pass.kernel;
    const rows: string[] = [];
    let histFrames = 0, raysOnHist = 0, framesWithRays = 0;
    for (let f = 0; f < 10; f++) {
      pass.setLights(pointAt([0.2 + 0.03 * f, 1.7, -0.5 + 0.02 * f]));
      fu.write({ camera: boxCamera(), prevCamera: boxCamera(), width: W, height: H, frameIndex: f, seedIndex: f, runSeed: 7, flags: 0,
        jitterMode: JITTER_IID, jitter: [0.5, 0.5], origin: g.gpu.origin, exposure: 1, time: 0, dt: 0, sceneDiag: 1 });
      await k.readCounters(true);
      const enc = dev.createCommandEncoder();
      const adv = k.advanceInteractive(fu.buffer, { reset: false });
      pass.encode(enc, { advanced: true, accumulate: true });
      dev.queue.submit([enc.finish()]);
      await dev.queue.onSubmittedWorkDone();
      const c = await k.readCounters(true);
      rows.push(`f${f} flags ${adv.flags} recs ${c.rsc.tRefreshRecs} rays ${c.rsc.tRefreshRays}`);
      if (adv.histValid && (adv.flags & K.TF_LIGHT_MOVED)) { histFrames++; raysOnHist += c.rsc.tRefreshRays; if (c.rsc.tRefreshRays > 0) framesWithRays++; }
    }
    console.log(`[interactive moving point] ${rows.join(' | ')}`);
    expect(histFrames).toBeGreaterThan(5);
    expect(framesWithRays).toBe(histFrames);
    pass.destroy(); color.destroy(); fu.destroy(); g.destroy();
  });
});

describe('interactive RestirFramePass on the smoke set-up (cornell_i + point 30 W + rect 20 W, point on the smoke loop)', () => {
  it('refresh rays > 0 on every history frame with LIGHT_MOVED', async () => {
    const { fetchScenePackage } = await import('../../src/core/scene/scene-package.ts');
    const pkg = await fetchScenePackage('/validation/scenes/cornell_i_512/');
    const scene = pkg.scene;
    const pt = (s: number) => light({ id: 901, type: 'point', power: 30, matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.1 + 0.12 * Math.sin(0.09 * s), 0.35, -0.2 + 0.1 * Math.cos(0.09 * s), 1]) });
    const rect = light({ id: 902, type: 'rect', power: 20, sizeX: 0.15, sizeY: 0.1, visibleToCamera: false, matrix: new Float32Array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0.5, -0.28, 1]) });
    const lightsAt = (s: number) => [...scene.lights, pt(s), rect];
    scene.lights = lightsAt(0);
    const { gpuScene } = await import('./restir-fixtures.ts');
    const g = await gpuScene(scene);
    const dev = g.device;
    const { FrameUniformBuffer, JITTER_IID } = await import('../../src/core/render/frame-uniforms.ts');
    const { RestirKernel } = await import('../../src/core/render/restir/kernel.ts');
    const { restirSettings } = await import('../../src/core/render/restir/presets.ts');
    const W = 128, H = 128;
    const pass = await RestirKernel.interactive(dev, g.gpu, g.env, 'rgba16float', { settings: restirSettings('interactive', { maxBounces: 3 }), features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
    const color = dev.createTexture({ size: [W, H], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
    const fu = new FrameUniformBuffer(dev);
    pass.setTargets({ width: W, height: H, color, frameUniforms: fu.buffer });
    const k = pass.kernel;
    const cam = { camToWorld: Array.from(pkg.camera.matrix), yfov: pkg.camera.yfov };
    const rows: string[] = [];
    let hist = 0, withRays = 0;
    for (let f = 0; f < 12; f++) {
      pass.setLights(lightsAt(f));
      fu.write({ camera: cam, prevCamera: cam, width: W, height: H, frameIndex: f, seedIndex: f, runSeed: 7, flags: 0,
        jitterMode: JITTER_IID, jitter: [0.5, 0.5], origin: g.gpu.origin, exposure: 1, time: 0, dt: 0, sceneDiag: 1 });
      await k.readCounters(true);
      const enc = dev.createCommandEncoder();
      const adv = k.advanceInteractive(fu.buffer, { reset: false });
      pass.encode(enc, { advanced: true, accumulate: true });
      dev.queue.submit([enc.finish()]);
      await dev.queue.onSubmittedWorkDone();
      const c = await k.readCounters(true);
      rows.push(`f${f} ${adv.flags} ${c.rsc.tRefreshRecs}/${c.rsc.tRefreshRays}`);
      if (adv.histValid && (adv.flags & K.TF_LIGHT_MOVED)) { hist++; if (c.rsc.tRefreshRays > 0) withRays++; }
    }
    console.log(`[interactive smoke set-up] ${rows.join(' | ')}`);
    expect(hist).toBeGreaterThan(5);
    expect(withRays).toBe(hist);
    pass.destroy(); color.destroy(); fu.destroy(); g.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ T6(b) with N3 / N7

describe('T6(b) robust mode detects the stale-suffix plants N3 and N7 (Changelog C-8 / B-10)', () => {
  it('box with point + rect, maxBounces 6, 12 chain frames: exact ≈ 0 mismatches; N3 (point moving + rect power steps) and N7 (rect power steps) ≫ 0', async () => {
    const W = 48, H = 48;
    const lights = (f: number, move: boolean, radio: boolean) => [
      light({ id: 1, type: 'point', power: 60, matrix: lightMatrixToward([0, -1, 0], [0.2 + (move ? 0.08 * f : 0), 1.7, -0.5]) }),
      light({ id: 2, type: 'rect', power: radio ? 40 * (1 + 0.5 * (f % 2)) : 40, sizeX: 0.5, sizeY: 0.3, matrix: lightMatrixToward([0, -1, 0], [-0.4, 1.9, -1.0]) }),
    ];
    const run = async (tPlant: RestirSettings['tPlant'], move: boolean, radio: boolean) => {
      const scene = boxScene(lights(0, move, radio));
      // Until T-B drops the N3 check of tsrc_load (B-10 loader side: since C-5 the refresh serves the stale values, and
      // under TM_ROBUST the inverse must stay fresh, C-8), the test runs the loader without it; a no-op afterwards.
      const { shaderSources } = await import('../../src/core/shaders/index.ts');
      const OLD = 'if (tcls_deep(f) && !rs_tplant(TP_N3_STALE)) {';
      const ts = shaderSources['restir/tshift.wgsl'];
      const extraSources: Record<string, string> = ts.includes(OLD) ? { 'restir/tshift.wgsl': ts.replace(OLD, 'if (tcls_deep(f)) {') } : {};
      const rig = await restirRig(scene, W, H, { preset: 'temporal', settings: { maxBounces: 6, temporalCheck: 'robust', tPlant }, seed: 77, extraSources });
      const k = rig.kernel, device = rig.g.device;
      let mismatch = 0, invOk = 0, selP = 0;
      for (let t = 0; t < 12; t++) {
        await k.readCounters(true);
        k.advance({ t, camera: boxCamera(), lights: lights(t, move, radio) });
        k.beginSubmit();
        const enc = device.createCommandEncoder();
        for (const u of k.frameUnits(t, { accum: rig.accum, counters: rig.counters })) u.encode(enc);
        device.queue.submit([enc.finish()]);
        await device.queue.onSubmittedWorkDone();
        const c = await k.readCounters(true);
        if (t > 0) { mismatch += c.rsc.tRobustMismatch; invOk += c.rsc.tInvOk; selP += c.rsc.tSelP; }
      }
      rig.destroy();
      return { mismatch, invOk, selP };
    };
    const exactMove = await run({}, true, true);
    const n3 = await run({ n3Stale: true }, true, true);
    const exactRadio = await run({}, false, true);
    const n7 = await run({ n7PerLight: true }, false, true);
    console.log(`[T6(b) plants] exact/move ${JSON.stringify(exactMove)} N3 ${JSON.stringify(n3)} exact/radio ${JSON.stringify(exactRadio)} N7 ${JSON.stringify(n7)}`);
    for (const e of [exactMove, exactRadio]) expect(e.mismatch).toBeLessThanOrEqual(Math.max(2, 1e-3 * e.selP));
    expect(n3.mismatch).toBeGreaterThan(Math.max(100, 20 * exactMove.mismatch));
    expect(n7.mismatch).toBeGreaterThan(Math.max(50, 20 * exactRadio.mismatch));
  });
});

// ------------------------------------------------------------------------------------------------ high-frequency HDRI

describe('§9.3-3 / §9.3-4 on t3_rare_256 with the studio_small_09 HDRI: env escapes ≤ 1e-6 (C-9 / C-10)', () => {
  it('idempotence (same frame) and refresh vs re-trace (lights moved + env rotation/strength) per technique', async () => {
    const { loadHdri } = await import('./env-fixtures.ts');
    const { T3_ENV_ID, t3Scene } = await import('../scenes/make-m4.ts');
    const env = await loadHdri(`${T3_ENV_ID}_1k.hdr`);
    if (!env) { console.warn('studio_small_09_1k.hdr missing: skipped'); return; }
    const t = t3Scene('t3_rare_256', env);
    const W = 128, P = W * W;
    const sources = { 'restir/rc.wgsl': TEST_RC_DR };
    const rig = await restirRig(t.scene, W, W, {
      preset: 'temporal', dumpCandidates: true, cam: { camToWorld: t.camera.matrix, yfov: t.camera.yfov }, extraSources: sources,
      settings: { maxBounces: t.maxBounces }, env: { nee: true }, seed: 19,
    });
    const k = rig.kernel;
    const byTech = (r: Recs, o: Uint32Array, ref: (i: number) => Float32Array | undefined) => {
      const of = new Float32Array(o.buffer);
      const st: Record<string, { n: number; max: number; bad: number; worst?: string }> = {};
      for (let i = 0; i < r.n; i++) {
        const c = recClass(r, i);
        if (!(c === C.DNEE || c === C.DBSDF || c === C.N1 || c === C.B1) || (o[OUT_WORDS * i + 4] & S.SXS_UNDEF)) continue;
        const exp = ref(i);
        if (!exp) continue;
        const key = `${PATH_CLASS_NAMES[c]}/${rfUnpack(recFlags(r, i)).tech}`;
        const e = rel3(of, exp, OUT_WORDS * i, 0);
        const s = (st[key] ??= { n: 0, max: 0, bad: 0 });
        s.n++;
        if (e > s.max) { s.max = e; s.worst = `${of[OUT_WORDS * i].toExponential(5)} vs ${exp[0].toExponential(5)}`; }
        if (e > 1e-6) s.bad++;
      }
      return st;
    };
    // (1) idempotence: the same frame's refresh vs the stored cache (fsFrom = fsTo = CUR)
    const idem: Record<string, unknown>[] = [];
    let idemBad = 0;
    for (let f = 0; f < 3; f++) {
      await rig.frames(1, 3 + f);
      const r = harvest(await k.readCandidateDump(), P);
      const o = await runRefresh(k, r, K.RS_FS_CUR, K.RS_FS_CUR, { sources });
      const recF32 = new Float32Array(r.rec.buffer);
      const st = byTech(r, o, (i) => recF32.subarray(i * RES_WORDS + RW.rcRad, i * RES_WORDS + RW.rcRad + 3));
      idem.push(st);
      idemBad += Object.values(st).reduce((a, s) => a + s.bad, 0);
    }
    // (2) change frame: lights moved / power steps + env rotation / strength; refresh vs re-trace, forward and inverse
    const L0 = t.scene.lights.map((l) => ({ ...l, matrix: new Float32Array(l.matrix) }));
    const L1 = L0.map((l, j) => (j % 2 === 0 ? { ...l, matrix: move(l.matrix, [0.03, 0.01, -0.02]) } : { ...l, power: l.power * 1.7 }));
    const E0: EnvParamsCpu = { rotationZ: 0.1, strength: 1, tint: [1, 1, 1], visibleToCamera: true };
    const E1: EnvParamsCpu = { rotationZ: 0.13, strength: 1.3, tint: [1, 1, 1], visibleToCamera: true };
    await canonicalFrame(rig, 10, L0, E0);
    const r0 = harvest(await k.readCandidateDump(), P);
    const a1 = await canonicalFrame(rig, 11, L1, E1);
    expect(a1.histValid).toBe(true);
    const r1 = harvest(await k.readCandidateDump(), P);
    const fwd = compareRetrace(r0, await runRefresh(k, r0, K.RS_FS_PREV, K.RS_FS_CUR, { sources }), await runRetrace(k, r0, k.resources.views.vbufPrev, K.RS_FS_PREV, K.RS_FS_CUR, sources));
    const inv = compareRetrace(r1, await runRefresh(k, r1, K.RS_FS_CUR, K.RS_FS_PREV, { sources }), await runRetrace(k, r1, k.resources.views.vbuf, K.RS_FS_CUR, K.RS_FS_PREV, sources));
    // (3) round trip of the cache: frame-0 records refreshed to CUR, then back to PREV ≡ the stored cache
    const fw = await runRefresh(k, r0, K.RS_FS_PREV, K.RS_FS_CUR, { sources });
    const y = { rec: r0.rec.slice(), ai: r0.ai, n: r0.n };
    for (let i = 0; i < r0.n; i++) {
      const b = i * RES_WORDS, o = OUT_WORDS * i;
      if (fw[o + 4] & S.SXS_UNDEF) continue;
      const c = recClass(r0, i);
      if (c === C.DNEE || c === C.DBSDF || c === C.N1 || c === C.B1) for (let j = 0; j < 3; j++) y.rec[b + RW.rcRad + j] = fw[o + j];
      if (rfUnpack(recFlags(r0, i)).tech === K.RS_TECH_NEE) {
        const e = (K.RC_TAG_NEE | (fw[o + 5] & K.RC_ENTRY_MASK)) >>> 0;
        y.rec[b + RW.end] = e;
        if (rfUnpack(recFlags(r0, i)).k === rfUnpack(recFlags(r0, i)).d) y.rec[b + RW.rc] = e;
      }
    }
    const back = await runRefresh(k, y, K.RS_FS_CUR, K.RS_FS_PREV, { sources });
    const rec0 = new Float32Array(r0.rec.buffer);
    const rt = byTech(r0, back, (i) => ((fw[OUT_WORDS * i + 4] & S.SXS_UNDEF) ? undefined : rec0.subarray(i * RES_WORDS + RW.rcRad, i * RES_WORDS + RW.rcRad + 3)));
    const rtBad = Object.values(rt).reduce((a, s) => a + s.bad, 0);
    const strip = ({ bad, n, ...rest }: CmpRep) => { void bad; void n; return rest; };
    console.log(`[t3_rare HDRI] idempotence ${JSON.stringify(idem)}\n fwd ${JSON.stringify(strip(fwd))}\n inv ${JSON.stringify(strip(inv))}\n round trip ${JSON.stringify(rt)}`);
    // C-9 / C-10: env escapes (technique 3) evaluate L_env(envUV(ω)) from a stored direction; with the explicit f32
    // bilinear every technique, env escapes included, is within 1e-6 of the stored cache (no allowance).
    expect(idemBad).toBe(0);
    expect(rtBad).toBe(0);
    for (const x of [fwd, inv]) {
      expect(x.bad.every((b) => b === 0)).toBe(true);
      expect(x.undefMismatch + x.entryBad + x.visMismatch + x.replayFail + x.idsMismatch + x.endMismatch).toBe(0);
    }
    rig.destroy();
  });
});

// ------------------------------------------------------------------------------------------------ material textures (measurement)

describe('measurement: textured vertices re-evaluated in another pipeline (vii_textured_512; coordinator C-10 item 3)', () => {
  it('stored rcRad (path-tree pipeline) vs refresh (x_{d−1} material) and vs full re-trace (every suffix material): report', async () => {
    const { fetchScenePackage } = await import('../../src/core/scene/scene-package.ts');
    const W = 128, P = W * W;
    const sources = { 'restir/rc.wgsl': TEST_RC_DR };
    const rep: Record<string, unknown> = {};
    for (const name of ['vii_textured_512', 'cornell_i_512']) {
      const pkg = await fetchScenePackage(`/validation/scenes/${name}/`);
      const rig = await restirRig(pkg.scene, W, W, {
        preset: 'temporal', dumpCandidates: true, cam: { camToWorld: Array.from(pkg.camera.matrix), yfov: pkg.camera.yfov }, extraSources: sources,
        settings: { maxBounces: 4 }, seed: 23,
      });
      const k = rig.kernel;
      const hist = { n: 0, gt1e6: 0, gt1e5: 0, gt1e4: 0, gt1e3: 0, max: 0 };
      const histRt = { n: 0, gt1e6: 0, gt1e5: 0, gt1e4: 0, gt1e3: 0, max: 0 };
      const add = (h: typeof hist, e: number) => { h.n++; h.max = Math.max(h.max, e); if (e > 1e-6) h.gt1e6++; if (e > 1e-5) h.gt1e5++; if (e > 1e-4) h.gt1e4++; if (e > 1e-3) h.gt1e3++; };
      for (let f = 0; f < 2; f++) {
        const adv = await canonicalFrame(rig, f, pkg.scene.lights);
        void adv;
        const r = harvest(await k.readCandidateDump(), P);
        const o = await runRefresh(k, r, K.RS_FS_CUR, K.RS_FS_CUR, { sources });
        const rt = await runRetrace(k, r, k.resources.views.vbuf, K.RS_FS_CUR, K.RS_FS_CUR, sources);
        const of = new Float32Array(o.buffer), rf = new Float32Array(rt.buffer), recF32 = new Float32Array(r.rec.buffer);
        for (let i = 0; i < r.n; i++) {
          const c = recClass(r, i);
          if (c !== C.DNEE && c !== C.DBSDF) continue;
          const b = i * RES_WORDS + RW.rcRad;
          add(hist, rel3(of, recF32, OUT_WORDS * i, b));
          if ((rt[OUT_WORDS * i + 4] & (16 | 4 | 32)) === 0) add(histRt, rel3(rf, recF32, OUT_WORDS * i, b));
        }
      }
      rep[name] = { refreshVsStored: hist, retraceVsStored: histRt };
      rig.destroy();
    }
    console.log(`[material textures] ${JSON.stringify(rep)}`);
    expect((rep['vii_textured_512'] as { refreshVsStored: { n: number } }).refreshVsStored.n).toBeGreaterThan(100);
  });
});
