// U-RES-1 (CPU part, restir-api.md §6.1): the WGSL data contract (types.wgsl constants, reservoir planes, RestirParams /
// RsDispatch layouts) equals the TS mirror layout.ts; rf pack/unpack round trip. Plus U-PT-BITS part 3: the inline
// budget of every composed ReSTIR entry point (§4.5: ≤ 1 bsdf_sample, ≤ 2 bsdf_query, ≤ 2 material_eval call sites,
// counted with inlining multiplicity through the call graph), and that every pass stays at ≤ 9 storage buffers.
import { describe, expect, it } from 'vitest';
import { composeWgsl, type Defines } from '../../src/core/gpu/wgsl-composer.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import {
  RESTIR_PARAMS_LAYOUT, RESTIR_PARAMS_SIZE, RP, RS_DISPATCH_SIZE, RS_WGSL_CONSTS, RW, rfPack, rfUnpack, decodeReservoir, RES_WORDS,
} from '../../src/core/render/restir/layout.ts';
import { RS_PASSES, restirDefines, storageBufferCount, type RsPassName } from '../../src/core/render/restir/resources.ts';

const stripComments = (s: string) => s.replace(/\/\/.*$/gm, '');

function wgslConsts(src: string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const m of stripComments(src).matchAll(/const\s+(\w+)\s*:\s*u32\s*=\s*(0x[0-9a-fA-F]+|\d+)u\s*;/g)) out[m[1]] = Number(m[2]);
  return out;
}

/** WGSL host-shareable struct layout (u32/f32/vec2u/vec4u only). */
function structOffsets(src: string, name: string): { fields: [string, number][]; size: number } {
  const m = new RegExp(`struct\\s+${name}\\s*\\{([^}]*)\\}`).exec(stripComments(src));
  if (!m) throw new Error(`struct ${name} not found`);
  const T: Record<string, [number, number]> = { u32: [4, 4], f32: [4, 4], vec2u: [8, 8], vec2f: [8, 8], vec4u: [16, 16], vec4f: [16, 16] };
  const fields: [string, number][] = [];
  let off = 0, maxAlign = 4;
  for (const f of m[1].matchAll(/(\w+)\s*:\s*(\w+)/g)) {
    const [size, align] = T[f[2]] ?? (() => { throw new Error(`type ${f[2]}`); })();
    off = Math.ceil(off / align) * align;
    fields.push([f[1], off]);
    off += size;
    maxAlign = Math.max(maxAlign, align);
  }
  return { fields, size: Math.ceil(off / maxAlign) * maxAlign };
}

describe('U-RES-1: WGSL data contract ≡ layout.ts', () => {
  it('every types.wgsl constant equals RS_WGSL_CONSTS (both directions)', () => {
    const w = wgslConsts(shaderSources['restir/types.wgsl']);
    expect(Object.keys(w).length).toBeGreaterThan(90);
    expect(w).toEqual({ ...RS_WGSL_CONSTS });
  });

  it('reservoir plane indices and word offsets', () => {
    const w = wgslConsts(shaderSources['restir/reservoir.wgsl']);
    expect({ WF: w.RP_WF, SEED: w.RP_SEED, RC: w.RP_RC, WI: w.RP_WI, RAD: w.RP_RAD, END: w.RP_END, SFX0: w.RP_SFX0, SFX1: w.RP_SFX1, SFX2: w.RP_SFX2, DIAG: w.RP_DIAG }).toEqual({ ...RP });
    // word offsets are 4·plane + component (§2.2)
    expect(RW.W).toBe(4 * RP.WF); expect(RW.seed).toBe(4 * RP.SEED); expect(RW.flags).toBe(4 * RP.SEED + 2); expect(RW.c).toBe(4 * RP.SEED + 3);
    expect(RW.rc).toBe(4 * RP.RC); expect(RW.jDen).toBe(4 * RP.RC + 3); expect(RW.rcWi).toBe(4 * RP.WI); expect(RW.aux).toBe(4 * RP.WI + 3);
    expect(RW.rcRad).toBe(4 * RP.RAD); expect(RW.wSum).toBe(4 * RP.RAD + 3); expect(RW.end).toBe(4 * RP.END); expect(RW.lobeHist).toBe(4 * RP.END + 3);
    expect(RW.sfx).toBe(4 * RP.SFX0); expect(RW.sfxFlags).toBe(4 * RP.SFX0 + 3); expect(RW.sfxDir).toBe(4 * RP.SFX1); expect(RW.sfxT).toBe(4 * RP.SFX1 + 3);
    expect(RW.betaS).toBe(4 * RP.SFX2); expect(RW.sfxP2).toBe(4 * RP.SFX2 + 3); expect(RW.nCand).toBe(4 * RP.DIAG); expect(RW.endpointId).toBe(4 * RP.DIAG + 3);
    expect(RES_WORDS).toBe(4 * RS_WGSL_CONSTS.RS_RES_PLANES);
  });

  it('RestirParams (128 B) and RsDispatch (32 B) layouts', () => {
    const src = shaderSources['restir/frame.wgsl'];
    const p = structOffsets(src, 'RestirParams');
    expect(p.size).toBe(RESTIR_PARAMS_SIZE);
    const named = Object.fromEntries(p.fields);
    for (const [k, off] of Object.entries(RESTIR_PARAMS_LAYOUT)) expect(named[k], k).toBe(off);
    const d = structOffsets(src, 'RsDispatch');
    expect(d.size).toBe(RS_DISPATCH_SIZE);
    expect(d.fields.map((f) => f[0])).toEqual(['t', 'passId', 'round', 'treeBase', 'treeCount', 'flags', 'rowBase', 'rowEnd']);
  });

  it('rf pack / unpack round trip (random fields) and decodeReservoir', () => {
    let s = 12345;
    const r = (n: number) => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s % n; };
    for (let i = 0; i < 20000; i++) {
      const f = { d: r(16), k: r(16), tech: r(4), ep: r(8), isDelta: !!r(2), lkm1: r(8), dkm1: !!r(2), lk: r(8), dk: !!r(2), mode: r(4), forced: !!r(2), bg: !!r(2) };
      const v = rfPack(f);
      expect(rfUnpack(v)).toEqual(f);
      expect(v & 0xFC000000).toBe(0);   // bits 26–31 reserved
    }
    const words = new Uint32Array(RES_WORDS * 2);
    const fl = new Float32Array(words.buffer);
    words[RES_WORDS + RW.flags] = rfPack({ d: 4, k: 2, tech: 1 });
    fl[RES_WORDS + RW.W] = 0.5; fl[RES_WORDS + RW.F + 2] = 3; words[RES_WORDS + RW.endpointId] = 77;
    const rec = decodeReservoir(words, 1);
    expect([rec.d, rec.k, rec.tech, rec.W, rec.F[2], rec.endpointId]).toEqual([4, 2, 1, 0.5, 3, 77]);
  });
});

// ------------------------------------------------------------------------------------------------ inline budget

const SCENE_DEFINES: Defines = {
  SCENE_GROUP: 1, BVH_DECLARE_BINDINGS: true, BVH_GROUP: 1, BVH_BINDING_NODES: 0, BVH_BINDING_TRIS: 1, WATERTIGHT: true, CUSTOM_ALPHA: true, VERTEX_FORMAT: 1,
  TEX_GROUP: 1, TEX_BINDING_BASE: 8, TEX_ARRAYS: 1, TEX_SAMPLERS: 1,
};

/** Function bodies of a composed module. */
function functions(code: string): Map<string, string> {
  const src = stripComments(code);
  const out = new Map<string, string>();
  const re = /\bfn\s+(\w+)\s*\(/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(src))) {
    let i = src.indexOf('{', m.index);
    const start = i;
    let depth = 0;
    for (; i < src.length; i++) {
      if (src[i] === '{') depth++;
      else if (src[i] === '}' && --depth === 0) break;
    }
    out.set(m[1], src.slice(start, i + 1));
  }
  return out;
}

/** Inlined call count of `target` from `entry` (Σ over call paths). */
export function inlineCount(code: string, entry: string, target: string): number {
  const fns = functions(code);
  const memo = new Map<string, number>();
  const count = (f: string, stack: string[]): number => {
    if (f === target) return 1;
    if (memo.has(f)) return memo.get(f)!;
    if (stack.includes(f)) throw new Error(`recursion ${stack.join('→')}→${f}`);
    const body = fns.get(f) ?? '';
    let n = 0;
    for (const c of body.matchAll(/\b(\w+)\s*\(/g)) if (fns.has(c[1]) && c[1] !== f) n += count(c[1], [...stack, f]);
    memo.set(f, n);
    return n;
  };
  return count(entry, []);
}

const PASSES_TO_CHECK: { name: RsPassName; extra?: Defines }[] = [
  { name: 'rs_primary' }, { name: 'rs_initial' }, { name: 'rs_initial_dump' }, { name: 'rs_finalize' },
  { name: 'rs_finalize_frame', extra: { COLOR_FORMAT: 'rgba16float' } },
  { name: 'rs_pair_accept' }, { name: 'rs_args' }, { name: 'rs_spatial_replay' }, { name: 'rs_spatial_shift' }, { name: 'rs_spatial_resample' },
  { name: 'rs_ensemble_stats' },
];

describe('U-PT-BITS (3): inline budget of every composed ReSTIR entry point (§4.5)', () => {
  for (const { name, extra } of PASSES_TO_CHECK) {
    const def = RS_PASSES[name];
    it.skipIf(!(def.file in shaderSources))(`${name}: ≤ 1 bsdf_sample, ≤ 2 bsdf_query, ≤ 2 material_eval; ≤ 9 storage buffers`, () => {
      const code = composeWgsl(def.file, { sources: shaderSources, defines: restirDefines(name, { sceneDefines: SCENE_DEFINES, debug: true, extra }) }).code;
      expect(code).toContain(`fn ${def.entry}(`);
      const n = { sample: inlineCount(code, def.entry, 'bsdf_sample'), query: inlineCount(code, def.entry, 'bsdf_query'), mat: inlineCount(code, def.entry, 'material_eval') };
      expect(n.sample, 'bsdf_sample').toBeLessThanOrEqual(1);
      expect(n.query, 'bsdf_query').toBeLessThanOrEqual(2);
      expect(n.mat, 'material_eval').toBeLessThanOrEqual(2);
      expect(storageBufferCount(name, true)).toBeLessThanOrEqual(9);
    });
  }

  it('the counter sees inlining multiplicity (PT: exactly one bsdf_sample, bsdf_eval once through pt_trace)', () => {
    const code = composeWgsl('passes/pt.wgsl', {
      sources: shaderSources,
      defines: { ...SCENE_DEFINES, ENV_GROUP: 0, ENV_BINDING: 1, LIGHTS_GROUP: 0, LIGHTS_BINDING: 5, LUT_BASE: '0u', LUT_IOR_S: '0u', LUT_S: '0u', LUT_E: '0u', LUT_EAVG: '0u', LUT_RECORDS_KIND: 1, PT_INTERACTIVE: false, PT_PROBE: false, GLASS_PLANT: 0, ENV_PLANT: 0, ENV_MIS_POWER: false },
    }).code;
    expect(inlineCount(code, 'pt_batch', 'bsdf_sample')).toBe(1);
    expect(inlineCount(code, 'pt_batch', 'material_eval')).toBe(1);
    const toy = 'fn a() { b(); b(); }\nfn b() { c(); }\nfn c() { }\nfn e() { a(); c(); }';
    expect(inlineCount(toy, 'e', 'c')).toBe(3);
  });
});

// ------------------------------------------------------------------------------------------------ M6 (restir-m6-api.md)

import { RS_M6_CONSTS } from '../../src/core/render/restir/layout.ts';

describe('U-RES-1 (M6): m6-types.wgsl ≡ RS_M6_CONSTS; inline budget of the M6 variants (MD6, R2)', () => {
  it('every u32 constant of m6-types.wgsl equals RS_M6_CONSTS (both directions; DUP_DENOM is f32 288)', () => {
    const w = wgslConsts(shaderSources['restir/m6-types.wgsl']);
    const { DUP_DENOM, ...u32s } = RS_M6_CONSTS;
    expect(w).toEqual(u32s);
    expect(stripComments(shaderSources['restir/m6-types.wgsl'])).toMatch(new RegExp(`const DUP_DENOM: f32 = ${DUP_DENOM}\\.0;`));
  });

  const M6: Defines = { RS_RIS_NEE: 1, RS_MODE_B: 1, RS_DUAL_MV: 1, RS_DUPMAP: 1 };
  const budget: { name: RsPassName; query: number }[] = [
    { name: 'rs_initial', query: 4 },            // NEE, continuation, RIS candidates, crossing candidates
    { name: 'rs_spatial_replay', query: 2 }, { name: 'rs_spatial_shift', query: 2 },
    { name: 'rs_t_forward', query: 2 }, { name: 'rs_t_inverse', query: 2 }, { name: 'rs_refresh_fwd', query: 2 },
    { name: 'rs_light_tiles', query: 0 }, { name: 'rs_dupmap', query: 0 },
  ];
  for (const { name, query } of budget) {
    it(`${name} (all M6 variants): ≤ 1 bsdf_sample, ≤ ${query} bsdf_query, ≤ 2 material_eval; ≤ 9 storage buffers`, () => {
      const def = RS_PASSES[name];
      const code = composeWgsl(def.file, { sources: shaderSources, defines: restirDefines(name, { sceneDefines: SCENE_DEFINES, debug: true, extra: M6 }) }).code;
      const n = { sample: inlineCount(code, def.entry, 'bsdf_sample'), query: inlineCount(code, def.entry, 'bsdf_query'), mat: inlineCount(code, def.entry, 'material_eval') };
      console.log(`[M6 inline] ${name} ${JSON.stringify(n)}`);
      expect(n.sample).toBeLessThanOrEqual(1);
      expect(n.query).toBeLessThanOrEqual(query);
      expect(n.mat).toBeLessThanOrEqual(2);
      expect(storageBufferCount(name, true)).toBeLessThanOrEqual(9);
    });
  }

  // perf2 WP-2a (RS_RIS_HOIST): the RIS candidate loop evaluates bsdf_f_all_ctx from one hoisted bsdf_prepare instead
  // of bsdf_query; the new entry counts against the same query budget, and appears exactly once
  it('rs_initial (all M6 variants + RS_RIS_HOIST): bsdf_query + bsdf_f_all_ctx ≤ 4, bsdf_f_all_ctx = 1, ≤ 1 bsdf_sample, ≤ 2 material_eval', () => {
    const def = RS_PASSES.rs_initial;
    const code = composeWgsl(def.file, { sources: shaderSources, defines: restirDefines('rs_initial', { sceneDefines: SCENE_DEFINES, debug: true, extra: { ...M6, RS_RIS_HOIST: 1 } }) }).code;
    const n = { sample: inlineCount(code, def.entry, 'bsdf_sample'), query: inlineCount(code, def.entry, 'bsdf_query'),
      fAll: inlineCount(code, def.entry, 'bsdf_f_all_ctx'), mat: inlineCount(code, def.entry, 'material_eval') };
    console.log(`[M6 inline] rs_initial+RS_RIS_HOIST ${JSON.stringify(n)}`);
    expect(n.fAll).toBe(1);
    expect(n.query + n.fAll).toBeLessThanOrEqual(4);
    expect(n.sample).toBeLessThanOrEqual(1);
    expect(n.mat).toBeLessThanOrEqual(2);
  });

  // perf2 WP-2b (RS_NEE_SITE): rs_initial's path tree traces visibility from two sites (NEE shadow + k=B retest; tree-pair
  // + emitter-rc retest) instead of five; Mode B's crossings keep their own two
  for (const modeB of [false, true]) {
    it(`rs_initial (M6${modeB ? '' : ' Mode-A text'}) + RS_NEE_SITE: trace_any_ex sites 5 → 2 in the path tree`, () => {
      const def = RS_PASSES.rs_initial;
      const ex: Defines = { ...M6, RS_MODE_B: modeB ? 1 : 0 };
      const cnt = (extra: Defines) => {
        const code = composeWgsl(def.file, { sources: shaderSources, defines: restirDefines('rs_initial', { sceneDefines: SCENE_DEFINES, debug: true, extra }) }).code;
        return { any: inlineCount(code, def.entry, 'trace_any_ex'), closest: inlineCount(code, def.entry, 'trace_closest_ex'), vr: inlineCount(code, def.entry, 'vis_ray') };
      };
      const off = cnt(ex);
      const on = cnt({ ...ex, RS_NEE_SITE: 1 });
      console.log(`[RS_NEE_SITE] modeB=${modeB} off ${JSON.stringify(off)} on ${JSON.stringify(on)}`);
      expect(on.vr).toBe(2);
      expect(off.any - on.any).toBe(3);
      expect(on.closest).toBe(off.closest);
    });
  }
});
