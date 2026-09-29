// ReSTIR PT shift core (WP-B, restir-api.md §6.1–§6.2): U-RC-1 (rcPairTest ≡ f64 dual), T2, T3-0…T3-5, T3-D,
// T3-ENV, T4, T5/U7, U5, U-11…U-13, dense PSS sweeps. Chrome lane authoritative; dawn.node is a pre-check.
import { afterAll, describe, expect, it } from 'vitest';
import { restirSettings } from '../../src/core/render/restir/presets.ts';
import { RestirKernel } from '../../src/core/render/restir/kernel.ts';
import { JITTER_IID } from '../../src/core/render/frame-uniforms.ts';
import { pairTestF64, isLogic, type RcEventD, type RcVertexD } from '../../tests/restir/rc-dual.ts';
import { releaseTestGpu } from './device-factory.ts';
import { boxCamera, boxScene, gpuScene, light } from './restir-fixtures.ts';
import { bitsToF32, readU32, storageBuffer, testPipeline } from './restir-shift-fixtures.ts';

afterAll(releaseTestGpu);

// ------------------------------------------------------------------------------------------------ U-RC-1

const RC_WORDS = 28;
const RC_HARNESS = `
#include "lights/lights.wgsl"
#include "restir/rc.wgsl"
@group(2) @binding(0) var<storage, read_write> outW: array<u32>;
fn hu(i: u32, k: u32) -> u32 { return pcg4d(vec4u(i, k, 0x3c6ef372u, 0x1b873593u)).x; }
fn hf(i: u32, k: u32) -> f32 { return u32_to_unit(hu(i, k)); }
fn hdir(i: u32, k: u32) -> vec3f {
  let z = 1.0 - 2.0 * hf(i, k);
  let r = sqrt(max(0.0, 1.0 - z * z));
  let ph = TWO_PI * hf(i, k + 1u);
  return vec3f(r * cos(ph), r * sin(ph), z);
}
fn hevent(i: u32, k: u32) -> RcEvent {
  var e: RcEvent;
  e.lobe = hu(i, k) % 6u;
  e.delta = select(0u, 1u, hf(i, k + 1u) < 0.08);
  let ak = hu(i, k + 2u) % 8u;
  var a = hf(i, k + 3u);
  if (ak == 0u) { a = 0.0; } else if (ak == 1u) { a = 0.19; } else if (ak == 2u) { a = 0.2; } else if (ak == 3u) { a = 0.21; }
  else if (ak == 4u) { a = 1.0; } else if (ak == 5u) { a = FLT_MAX; }
  e.alpha = a;
  e.pMarg = exp2(mix(-12.0, 16.0, hf(i, k + 4u)));
  if (hf(i, k + 5u) < 0.03) { e.pMarg = 0.0; }
  return e;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.y * 64u * 256u + gid.x;
  var a = RcVertex(vec3f(0.0), hdir(i, 1u), RCK_SURFACE, 0u);
  let t = exp2(mix(-8.0, 5.0, hf(i, 3u)));
  var b = RcVertex(a.pos + t * hdir(i, 4u), hdir(i, 6u), hu(i, 8u) % 3u, select(0u, 1u, hf(i, 9u) < 0.25));
  a.pos = vec3f(mix(-3.0, 3.0, hf(i, 10u)), mix(-3.0, 3.0, hf(i, 11u)), mix(-3.0, 3.0, hf(i, 12u)));
  // opaque to the compiler (a runtime 0): relaxed math must not fold (a + t·h) − a back to t·h in rcPairTest
  b.pos = bitcast<vec3f>(bitcast<vec3u>(a.pos + t * hdir(i, 4u)) ^ vec3u(rsDispatch.rowBase));
  let ea = hevent(i, 20u);
  let eb = hevent(i, 30u);
  let thr = exp2(mix(-20.0, 8.0, hf(i, 40u)));
  let r = rcPairTest(a, ea, b, eb, thr);
  let o = i * ${RC_WORDS}u;
  outW[o] = bitcast<u32>(a.pos.x); outW[o + 1u] = bitcast<u32>(a.pos.y); outW[o + 2u] = bitcast<u32>(a.pos.z);
  outW[o + 3u] = bitcast<u32>(a.ng.x); outW[o + 4u] = bitcast<u32>(a.ng.y); outW[o + 5u] = bitcast<u32>(a.ng.z);
  outW[o + 6u] = bitcast<u32>(b.pos.x); outW[o + 7u] = bitcast<u32>(b.pos.y); outW[o + 8u] = bitcast<u32>(b.pos.z);
  outW[o + 9u] = bitcast<u32>(b.ng.x); outW[o + 10u] = bitcast<u32>(b.ng.y); outW[o + 11u] = bitcast<u32>(b.ng.z);
  outW[o + 12u] = b.kind | (b.diffuseOnly << 4u);
  outW[o + 13u] = ea.lobe | (ea.delta << 4u); outW[o + 14u] = bitcast<u32>(ea.alpha); outW[o + 15u] = bitcast<u32>(ea.pMarg);
  outW[o + 16u] = eb.lobe | (eb.delta << 4u); outW[o + 17u] = bitcast<u32>(eb.alpha); outW[o + 18u] = bitcast<u32>(eb.pMarg);
  outW[o + 19u] = bitcast<u32>(thr);
  outW[o + 20u] = select(0u, 1u, r.ok) | (r.term << 4u);
  outW[o + 21u] = bitcast<u32>(r.margin);
  outW[o + 22u] = bitcast<u32>(rsParams.crit2022MinDist);
  outW[o + 23u] = bitcast<u32>(rsParams.alphaMin);
}
`;

describe('U-RC-1: rcPairTest (Enhanced and 2022) ≡ the f64 dual on random pair configurations', () => {
  for (const crit of ['enhanced', '2022'] as const) {
    it(`${crit}: 2^20 random pairs, LOGIC disagreements = 0`, async () => {
      const g = await gpuScene(boxScene([light({ id: 1, type: 'point', power: 50 })]));
      const k = await RestirKernel.create(g.device, g.gpu, g.env, { settings: restirSettings('initial', { criteria: crit }), features: g.features, wgslLanguageFeatures: g.wgslLanguageFeatures });
      k.setView({ camera: boxCamera(), width: 8, height: 8, runSeed: 1, jitterMode: JITTER_IID });
      const tp = await testPipeline(k, 'rc-test', RC_HARNESS, 'main', 1);
      const N = 256 * 64 * 64;
      const out = storageBuffer(g.device, N * RC_WORDS * 4);
      await tp.run([out], [256, 64]);
      const w = await readU32(g.device, out);
      const dmin = bitsToF32(w[22]), alphaMin = bitsToF32(w[23]);
      let logic = 0, fp = 0, pass = 0;
      const termHist = [0, 0, 0, 0, 0, 0];
      const ex: string[] = [];
      for (let i = 0; i < N; i++) {
        const o = i * RC_WORDS;
        const f = (j: number) => bitsToF32(w[o + j]);
        const a: RcVertexD = { pos: [f(0), f(1), f(2)], ng: [f(3), f(4), f(5)], kind: 0, diffuseOnly: false };
        const b: RcVertexD = { pos: [f(6), f(7), f(8)], ng: [f(9), f(10), f(11)], kind: w[o + 12] & 15, diffuseOnly: (w[o + 12] >> 4) !== 0 };
        const ea: RcEventD = { lobe: w[o + 13] & 15, delta: (w[o + 13] >> 4) !== 0, alpha: f(14), pMarg: f(15) };
        const eb: RcEventD = { lobe: w[o + 16] & 15, delta: (w[o + 16] >> 4) !== 0, alpha: f(17), pMarg: f(18) };
        const gpuOk = (w[o + 20] & 1) !== 0, gpuTerm = w[o + 20] >> 4;
        const d = pairTestF64(a, ea, b, eb, f(19), { alphaMin, crit2022: crit === '2022', dmin });
        termHist[d.term]++;
        if (d.ok) pass++;
        if (d.ok !== gpuOk || (!d.ok && d.term !== gpuTerm)) {
          if (isLogic(d)) { logic++; if (ex.length < 8) ex.push(`i=${i} gpu(ok=${gpuOk},term=${gpuTerm},m=${f(21)}) f64(${JSON.stringify(d)}) in=${JSON.stringify({ a, ea, b, eb, thr: f(19) })}`); } else fp++;
        }
      }
      console.log(`[U-RC-1 ${crit}] N=${N} pass=${pass} terms=${termHist} LOGIC=${logic} FP=${fp}`);
      if (ex.length) console.log(ex.join('\n'));
      expect(logic).toBe(0);
      out.destroy(); k.destroy(); g.destroy();
    });
  }
});
