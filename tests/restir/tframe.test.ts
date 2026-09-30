// M5 temporal data contract, CPU part (restir-temporal-api.md §2, §4.2, §6.1; T-A):
//   U-RES-1 (M5 part)  tframe.wgsl TS_* / SXS_* / SFX_* constants, RsTemporal and RestirParams layouts ≡ layout.ts
//   U-BIND-1           every pass of §4.2 ≤ 9 storage buffers (declared in the composed module and in the pass table),
//                      ≤ 5 of them in the scene group; the temporal passes stay within the inline budget (§4.5)
//   arena / presets    arena extension offsets, NS_alloc, packRsTemporal, the temporal presets, temporal-off flags = M4
import { describe, expect, it } from 'vitest';
import { composeWgsl, type Defines } from '../../src/core/gpu/wgsl-composer.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import {
  ARENA_HDR_BYTES, RESTIR_PARAMS_LAYOUT, RS_TEMPORAL_LAYOUT, RS_TEMPORAL_SIZE, RS_WGSL_CONSTS as K, TS_CONSTS, TSW, arenaBytes, arenaWords,
  decodeSfx, decodeTState, nsAlloc, packRestirParams, packRsTemporal, queueCapacityQ, queueItemBase, sfxWord, tsWord, unpackRsTemporal,
} from '../../src/core/render/restir/layout.ts';
import { numSlotsOf, restirFlags, restirSettings, tModeOf, tPlantsOf, validateSettings } from '../../src/core/render/restir/presets.ts';
import { RS_PASSES, TEMPORAL_PASSES, restirDefines, storageBufferCount, type RsPassName } from '../../src/core/render/restir/resources.ts';
import { inlineCount } from './layout.test.ts';

const stripComments = (s: string) => s.replace(/\/\/.*$/gm, '');
function wgslConsts(src: string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const m of stripComments(src).matchAll(/const\s+(\w+)\s*:\s*u32\s*=\s*(0x[0-9a-fA-F]+|\d+)u\s*;/g)) out[m[1]] = Number(m[2]);
  return out;
}

/** WGSL struct layout with the types used by the M5 uniforms (EnvParams = 32 B, align 16). */
function structOffsets(src: string, name: string): { fields: [string, number][]; size: number } {
  const m = new RegExp(`struct\\s+${name}\\s*\\{([^}]*)\\}`).exec(stripComments(src));
  if (!m) throw new Error(`struct ${name} not found`);
  const T: Record<string, [number, number]> = { u32: [4, 4], f32: [4, 4], vec2u: [8, 8], vec2f: [8, 8], vec3f: [12, 16], vec4u: [16, 16], vec4f: [16, 16], EnvParams: [32, 16] };
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

describe('U-RES-1 (M5): tframe.wgsl ≡ layout.ts', () => {
  it('TS_* / SXS_* / SFX_* constants (both directions)', () => {
    const w = wgslConsts(shaderSources['restir/tframe.wgsl']);
    const mirrored = Object.fromEntries(Object.entries(w).filter(([k]) => /^(TS_|SXS_|SFX_)/.test(k) && k !== 'TS_QPRIME_NONE'));
    expect(mirrored).toEqual({ ...TS_CONSTS });
    const tsw = Object.fromEntries(Object.entries(w).filter(([k]) => k.startsWith('TSW_')).map(([k, v]) => [k.slice(4), v]));
    const expected = Object.fromEntries(Object.entries(TSW).map(([k, v]) => [k.toUpperCase(), v]));
    expect(tsw).toEqual(expected);
  });

  it('RsTemporal (128 B) and the RestirParams M5 words', () => {
    const t = structOffsets(shaderSources['restir/tframe.wgsl'], 'RsTemporal');
    expect(t.size).toBe(RS_TEMPORAL_SIZE);
    const named = Object.fromEntries(t.fields);
    for (const [k, off] of Object.entries(RS_TEMPORAL_LAYOUT)) expect(named[k], k).toBe(off);
    const env = structOffsets(shaderSources['lights/env.wgsl'], 'EnvParams');
    expect(env.size).toBe(32);
    const p = Object.fromEntries(structOffsets(shaderSources['restir/frame.wgsl'], 'RestirParams').fields);
    expect([p.boostSlots, p.tMode, p.cCap, p.tPlants, p.pad4, p.pad5]).toEqual([104, 108, 112, 116, 120, 124]);
    expect(RESTIR_PARAMS_LAYOUT.cCap).toBe(112);
    const f = new Float32Array(packRestirParams({
      atlasSize: [8, 8], memberSize: [8, 8], memberCols: 1, memberCount: 1, maxBounces: 3, flags: 0, numTrees: 1, numSlots: 3, numRounds: 0,
      rrMinBounces: 3, tau: 2e-4, alphaMin: 0.2, wScale: 1, crit2022MinDist: 0, pairTexSize: [], lightMode: 0, memberBase: 0,
      boostSlots: 2, tMode: 5, cCap: 20, tPlants: 1024,
    }));
    const u = new Uint32Array(f.buffer);
    expect([u[26], u[27], f[28], u[29], u[30], u[31]]).toEqual([2, 5, 20, 1024, 0, 0]);
  });

  it('packRsTemporal round trip; envPrev is copied verbatim', () => {
    const env = Uint32Array.from([1, 2, 3, 4, 5, 6, 7, 8]);
    const t = { flags: K.TF_HIST_VALID | K.TF_REFRESH, histFrames: 7, frameGen: 9, prevGen: 8, envPrev: env, gens: [1, 2, 3, 4] as [number, number, number, number], gensPrev: [5, 6, 7, 8] as [number, number, number, number], configHash: 0xdeadbeef };
    const u = new Uint32Array(packRsTemporal(t));
    expect(u.length * 4).toBe(RS_TEMPORAL_SIZE);
    expect(Array.from(u.subarray(4, 12))).toEqual(Array.from(env));
    const back = unpackRsTemporal(u);
    expect({ ...back, envPrev: Array.from(back.envPrev as Uint32Array) }).toEqual({ ...t, envPrev: Array.from(env) });
  });
});

describe('arena extension (TD16/TD17, §2.8)', () => {
  it('NS_alloc, sizes, offsets and queue regions', () => {
    expect(nsAlloc(1, false)).toBe(1); expect(nsAlloc(1, true)).toBe(2); expect(nsAlloc(3, true)).toBe(3);
    const P = 1000, NS = 3;
    expect(arenaBytes(P, NS)).toBe(ARENA_HDR_BYTES + 24 * P * NS);            // M4 unchanged
    expect(arenaBytes(P, NS, true)).toBe(4 * (64 + 6 * P * NS + 36 * P));     // §2.8 total words
    const a = arenaWords(P, NS);
    expect([a.slots, a.codes, a.items, a.tState, a.sfxOut, a.end]).toEqual([0, 4 * P * NS, 5 * P * NS, 6 * P * NS, 6 * P * NS + 20 * P, 6 * P * NS + 36 * P]);
    expect(4 * a.end + ARENA_HDR_BYTES).toBe(arenaBytes(P, NS, true));
    expect(tsWord(P, NS, 5, 19)).toBe(6 * P * NS + 100 + 19);
    expect(sfxWord(P, NS, 1, 5, 7)).toBe(6 * P * NS + 20 * P + 80 + 8 + 7);
    expect(sfxWord(P, NS, 1, P - 1, 7) + 1).toBe(a.end);
    expect([queueItemBase(P, 0), queueItemBase(P, 1), queueItemBase(P, 2)]).toEqual([0, 0, P]);
    expect([queueCapacityQ(P, NS, 0), queueCapacityQ(P, NS, 1), queueCapacityQ(P, NS, 2)]).toEqual([P * NS, P, P]);
    // Q_f + Q_i fit into the item region of NS_alloc slots
    expect(queueItemBase(P, 2) + queueCapacityQ(P, NS, 2)).toBeLessThanOrEqual(P * nsAlloc(1, true));
  });

  it('decodeTState / decodeSfx read the documented words', () => {
    const P = 4, NS = 2;
    const words = new Uint32Array(arenaWords(P, NS).end);
    const f = new Float32Array(words.buffer);
    f[tsWord(P, NS, 3, TSW.fwdF + 1)] = 2.5; words[tsWord(P, NS, 3, TSW.flags)] = TS_CONSTS.TS_QVALID; words[tsWord(P, NS, 3, TSW.qPrime)] = 2;
    f[tsWord(P, NS, 3, TSW.cPrev)] = 33;
    const t = decodeTState(words, P, NS, 3);
    expect([t.fwdF[1], t.flags, t.qPrime, t.cPrev]).toEqual([2.5, TS_CONSTS.TS_QVALID, 2, 33]);
    f[sfxWord(P, NS, 1, 2, 6)] = 0.25; words[sfxWord(P, NS, 1, 2, 4)] = TS_CONSTS.SXS_DONE;
    const r = decodeSfx(words, P, NS, 1, 2);
    expect([r.jp, r.status]).toEqual([0.25, TS_CONSTS.SXS_DONE]);
  });
});

describe('presets and flags (TD24)', () => {
  it('temporal off: flags, numSlots and tMode are the M4 values', () => {
    for (const p of ['initial', 'initial-rr', 'offline', 'criteria2022'] as const) {
      const s = restirSettings(p);
      expect(s.temporal, p).toBe(false);
      expect(restirFlags(s) & K.RSF_TEMPORAL).toBe(0);
      expect(numSlotsOf(s)).toBe(s.slots);
      expect(tModeOf(s)).toBe(0);
      expect(tPlantsOf(s)).toBe(0);
    }
    expect(restirFlags(restirSettings('initial'))).toBe(0);
  });
  it('temporal / full / interactive presets', () => {
    const t = restirSettings('temporal'), f = restirSettings('full'), i = restirSettings('interactive');
    expect([t.trees, t.rounds, t.rr, t.temporal, t.cCap]).toEqual([1, 0, false, true, 20]);
    expect([f.trees, f.rounds, f.slots, f.diskRadius, f.rr, f.temporal]).toEqual([1, 1, 3, 30, false, true]);
    expect([i.temporal, i.rr]).toEqual([true, true]);
    expect(restirFlags(t) & K.RSF_TEMPORAL).toBe(K.RSF_TEMPORAL);
    expect(numSlotsOf({ slots: 3, boostSlots: 3, temporal: true })).toBe(6);
    expect(numSlotsOf({ slots: 3, boostSlots: 3, temporal: false })).toBe(3);
    expect(() => validateSettings({ ...f, slots: 4, boostSlots: 3 })).toThrow();
    expect(tModeOf({ ...f, temporalMis: 'talbot', temporalCheck: 'robust', refresh: 'e2' })).toBe(K.TM_TALBOT | K.TM_ROBUST | K.TM_E2);
    expect(tPlantsOf({ ...f, tPlant: { n1Mixed: true, cpPlus1: true } })).toBe(K.TP_N1_MIXED | K.TP_CP_PLUS1);
  });
});

// ------------------------------------------------------------------------------------------------ U-BIND-1

/** Scene defines of the merged dataformats scene group (SceneGpu.defines(1) of a textured, quantized scene). */
const SCENE_DEFINES: Defines = {
  SCENE_GROUP: 1, BVH_DECLARE_BINDINGS: true, BVH_GROUP: 1, BVH_BINDING_NODES: 0, BVH_BINDING_TRIS: 1, WATERTIGHT: true, CUSTOM_ALPHA: true, VERTEX_FORMAT: 1,
  TEX_GROUP: 1, TEX_BINDING_BASE: 8, TEX_ARRAYS: 1, TEX_SAMPLERS: 1,
};

/** Storage-buffer declarations of a composed module: total and per group. */
function storageDecls(code: string): { total: number; byGroup: Map<number, number> } {
  const byGroup = new Map<number, number>();
  let total = 0;
  for (const m of stripComments(code).matchAll(/@group\((\d+)u?\)\s*@binding\([^)]*\)\s*var<storage/g)) {
    total++;
    const g = Number(m[1]);
    byGroup.set(g, (byGroup.get(g) ?? 0) + 1);
  }
  return { total, byGroup };
}

describe('U-BIND-1: every ReSTIR pass ≤ 9 storage buffers, scene group ≤ 5 (§4.2)', () => {
  const names = Object.keys(RS_PASSES) as RsPassName[];
  for (const name of names) {
    const def = RS_PASSES[name];
    it.skipIf(!(def.file in shaderSources))(name, () => {
      const extra: Defines = name === 'rs_finalize_frame' ? { COLOR_FORMAT: 'rgba16float' } : {};
      const code = composeWgsl(def.file, { sources: shaderSources, defines: restirDefines(name, { sceneDefines: SCENE_DEFINES, debug: true, extra }) }).code;
      expect(code).toContain(`fn ${def.entry}(`);
      const d = storageDecls(code);
      expect(d.byGroup.get(1) ?? 0, 'scene group').toBeLessThanOrEqual(5);
      expect(d.total, 'declared storage buffers').toBeLessThanOrEqual(9);
      expect(storageBufferCount(name, true), 'pass table count').toBeLessThanOrEqual(9);
      // the pass table and the composed module agree on the G2 storage buffers
      const g2 = def.g2.filter((g) => g.k === 'ro' || g.k === 'rw').length;
      expect(d.byGroup.get(2) ?? 0, 'G2 storage').toBeLessThanOrEqual(g2);
    });
  }
});

describe('temporal passes: inline budget (§4.5) and bindings', () => {
  for (const name of TEMPORAL_PASSES) {
    it(name, () => {
      const def = RS_PASSES[name];
      const code = composeWgsl(def.file, { sources: shaderSources, defines: restirDefines(name, { sceneDefines: SCENE_DEFINES, debug: true }) }).code;
      expect(inlineCount(code, def.entry, 'bsdf_sample')).toBeLessThanOrEqual(1);
      expect(inlineCount(code, def.entry, 'bsdf_query')).toBeLessThanOrEqual(2);
      expect(inlineCount(code, def.entry, 'material_eval')).toBeLessThanOrEqual(2);
      expect(code).toContain('@group(0) @binding(8) var<uniform> rsTemporal');
      expect(def.defines.RS_TEMPORAL).toBe(1);
    });
  }
  it('M4 passes do not declare the temporal uniform or the prev G-buffer (bitwise M4 modules)', () => {
    for (const name of Object.keys(RS_PASSES) as RsPassName[]) {
      if (TEMPORAL_PASSES.includes(name) || !(RS_PASSES[name].file in shaderSources)) continue;
      const extra: Defines = name === 'rs_finalize_frame' ? { COLOR_FORMAT: 'rgba16float' } : {};
      const code = composeWgsl(RS_PASSES[name].file, { sources: shaderSources, defines: restirDefines(name, { sceneDefines: SCENE_DEFINES, debug: true, extra }) }).code;
      const c = stripComments(code);
      expect(c, name).not.toContain('rsTemporal');
      expect(c, name).not.toContain('rsVbufPrev');
    }
  });
});
