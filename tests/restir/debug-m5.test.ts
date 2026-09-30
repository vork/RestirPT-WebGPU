// M5 T-D, CPU part (restir-temporal-api.md §2.11, §3.7, TD21; Changelog D-1):
//   boost mirror      pairing.wgsl reads the tState flag word without tframe.wgsl: PAIR_TS_* ≡ layout.ts (TS_WORDS,
//                     TSW.flags, TS_DISOCC) and the index formula ≡ tsWord; rs_pair_accept stays free of the temporal
//                     uniform; the interactive preset has boost 3 and every boost slot has a pairing layer
//   view registry     views 480–497 and probe tags 73–79 ≡ debug/restir-views.wgsl (both directions), legends
//   probe decoding    tags 73–78 → RsProbeTemporal; temporal overlay paths 23 / 31
//   HUD               temporal lines from the arena header and the frame's RsTemporal flags
import { describe, expect, it } from 'vitest';
import { composeWgsl } from '../../src/core/gpu/wgsl-composer.ts';
import { shaderSources } from '../../src/core/shaders/index.ts';
import type { ProbeRecord } from '../../src/core/render/probe.ts';
import {
  RESTIR_PROBE_TAGS, RESTIR_VIEWS, RS_PROBE_TAG_T, RS_VIEW_T, codeName, decodeRestirProbe, isRestirCodeView, legendCodes, parseArenaHeader,
  shiftedPolylines, temporalHudLines, tfFlagNames,
} from '../../src/core/render/restir/debug.ts';
import { ARENA_HDR_BYTES, RS_WGSL_CONSTS as K, TS_CONSTS, TS_WORDS, TSW, arenaWords, queueHdr, tsWord } from '../../src/core/render/restir/layout.ts';
import { PAIR_TEX_SIZES, RESTIR_PRESETS, numSlotsOf, restirSettings } from '../../src/core/render/restir/presets.ts';
import { restirDefines } from '../../src/core/render/restir/resources.ts';

const stripComments = (s: string) => s.replace(/\/\/.*$/gm, '');
function wgslConsts(src: string): Record<string, number> {
  const out: Record<string, number> = {};
  for (const m of stripComments(src).matchAll(/const\s+(\w+)\s*:\s*u32\s*=\s*(0x[0-9a-fA-F]+|\d+)u\s*;/g)) out[m[1]] = Number(m[2]);
  return out;
}
const SCENE_DEFINES = { SCENE_GROUP: 1, SCENE_BVH_NODES_BINDING: 0, SCENE_BVH_TRIS_BINDING: 1, SCENE_VERTS_BINDING: 2, SCENE_TRIS_BINDING: 3, SCENE_MATS_BINDING: 4 };

describe('boost (TD21): pairing.wgsl tState mirror, presets', () => {
  it('PAIR_TS_* ≡ layout.ts and the flag-word index ≡ tsWord(P, NS, ai, TSW.flags)', () => {
    const c = wgslConsts(shaderSources['restir/pairing.wgsl']);
    expect(c.PAIR_TS_WORDS).toBe(TS_WORDS);
    expect(c.PAIR_TSW_FLAGS).toBe(TSW.flags);
    expect(c.PAIR_TS_DISOCC).toBe(TS_CONSTS.TS_DISOCC);
    expect(stripComments(shaderSources['restir/pairing.wgsl'])).toContain('6u * rs_atlas_pixels() * rs_ns_alloc() + PAIR_TS_WORDS * ai + PAIR_TSW_FLAGS');
    for (const [P, NS, ai] of [[16, 2, 0], [1024, 6, 777], [540 * 960, 6, 540 * 960 - 1]]) {
      expect(6 * P * NS + c.PAIR_TS_WORDS * ai + c.PAIR_TSW_FLAGS).toBe(tsWord(P, NS, ai, TSW.flags));
      expect(arenaWords(P, NS).tState).toBe(6 * P * NS);
    }
  });

  it('rs_pair_accept stays an M4 module (no temporal uniform) and calls the boost helper', () => {
    const code = composeWgsl('passes/restir/pair-accept.wgsl', { sources: shaderSources, defines: restirDefines('rs_pair_accept', { debug: true }) }).code;
    expect(stripComments(code)).not.toContain('rsTemporal');
    expect(code).toContain('pair_boost_accept(a, pair_disoccluded(p.ai), pair_disoccluded(qai))');
    void SCENE_DEFINES;
  });

  it('interactive preset: temporal + boost 3 (Q7); unbiased presets have no boost; every slot has a pairing layer', () => {
    expect(RESTIR_PRESETS.interactive.boostSlots).toBe(3);
    const s = restirSettings('interactive');
    expect(numSlotsOf(s)).toBe(6);
    expect(numSlotsOf(s)).toBeLessThanOrEqual(K.RS_MAX_SLOTS);
    for (let l = 0; l < numSlotsOf(s); l++) expect(PAIR_TEX_SIZES[l], `layer ${l}`).toBeGreaterThan(0);
    for (const p of ['temporal', 'full'] as const) expect(numSlotsOf(restirSettings(p))).toBe(restirSettings(p).slots);
    expect(numSlotsOf({ ...s, temporal: false })).toBe(s.slots);     // boost only with temporal on
  });
});

describe('M5 view registry and probe tags ≡ restir-views.wgsl', () => {
  const w = wgslConsts(shaderSources['debug/restir-views.wgsl']);
  it('views 480–497 (both directions) and code views', () => {
    const wgslViews = Object.entries(w).filter(([k]) => k.startsWith('RSV_T_') || k === 'RSV_S_BOOST').map(([, v]) => v).sort((a, b) => a - b);
    expect(wgslViews).toEqual(Object.values(RS_VIEW_T).sort((a, b) => a - b));
    for (const id of Object.values(RS_VIEW_T)) {
      const v = RESTIR_VIEWS.filter((x) => x.id === id);
      expect(v.length, `view ${id}`).toBe(1);
    }
    const code = new Set(RESTIR_VIEWS.filter((v) => v.id >= 480 && v.kind === 'code').map((v) => v.id));
    expect([...code].sort()).toEqual([480, 482, 483, 484, 485, 492, 496, 497]);
    for (const id of code) {
      expect(isRestirCodeView(id)).toBe(true);
      for (const c of legendCodes(id) ?? []) expect(codeName(id, c)).not.toBe('');
    }
    // rs_debug_fill treats exactly these as code views (debug.wgsl rsdbg_is_code_view)
    const dbg = shaderSources['passes/restir/debug.wgsl'];
    for (const n of ['RSV_T_QVALID', 'RSV_T_RF_FWD', 'RSV_T_INVCODE', 'RSV_T_SEL', 'RSV_T_LCHG', 'RSV_S_BOOST']) expect(dbg).toContain(n);
    expect(codeName(RS_VIEW_T.qvalid, 3)).toMatch(/disoccluded/);
    expect(codeName(RS_VIEW_T.lightsChanged, 5)).toBe('moved + undefined');
    const keys = RESTIR_VIEWS.map((v) => v.key);
    expect(new Set(keys).size).toBe(keys.length);
  });
  it('probe tags 73–79', () => {
    const tags = Object.entries(w).filter(([k]) => k.startsWith('RSP_T_')).map(([, v]) => v).sort((a, b) => a - b);
    expect(tags).toEqual(Object.values(RS_PROBE_TAG_T).sort((a, b) => a - b));
    for (const t of tags) expect(RESTIR_PROBE_TAGS.some(([x]) => x === t), `tag ${t} registered`).toBe(true);
  });
});

const rec = (tag: number, value: number[], asBits = [false, false, false, false], seq = 0): ProbeRecord => {
  const bits = new Uint32Array(4), f = new Float32Array(bits.buffer);
  value.forEach((v, i) => { if (asBits[i]) bits[i] = v >>> 0; else f[i] = v; });
  return { pixel: [3, 4], tag, seq, value: Array.from(f), bits: Array.from(bits) } as unknown as ProbeRecord;
};

describe('decodeRestirProbe (M5 tags)', () => {
  it('73–78 → RsProbeTemporal; paths 23 / 31', () => {
    const flags = TS_CONSTS.TS_QVALID | TS_CONSTS.TS_INV_QUEUED | TS_CONSTS.TS_SEL_C;
    const recs = [
      rec(78, [3.25, 4.5, 2, 1], [false, false, true, true]),
      rec(73, [100, 77, 12, flags], [true, true, false, true]),
      rec(74, [K.SC_OK, 1.25, 0.5, 0.3], [true, false, false, false]),
      rec(77, [TS_CONSTS.SXS_DONE | TS_CONSTS.SXS_RAY, 0.2, 0.1, 5], [true, false, false, true]),
      rec(75, [K.SC_O1 | (3 << 12), 0, 0.4, 0.25], [true, false, false, false]),
      rec(77, [TS_CONSTS.SXS_DONE | (1 << 16) | (1 << 17), 0, 0, 0xFFFFFFFF], [true, false, false, true]),
      rec(76, [1, 2, 0.3, 1 | (1 << 8)], [false, false, false, true]),
      rec(67, [1, 2, 3, (23 << 8) | (1 << 4) | 0xF], [false, false, false, true]),
      rec(67, [4, 5, 6, (23 << 8) | (2 << 4) | 0xF], [false, false, false, true]),
      rec(67, [0, 0, 9, (31 << 8) | (0 << 4) | 0xF], [false, false, false, true]),
      rec(67, [7, 8, 9, (31 << 8) | (1 << 4) | 0xF], [false, false, false, true]),
    ];
    const d = decodeRestirProbe(recs);
    const t = d.temporal!;
    expect(t.pick).toEqual({ sp: [3.25, 4.5], tap: 2, valid: true });
    expect([t.ai, t.qPrime, t.cPrev, t.flags]).toEqual([100, 77, 12, flags]);
    expect(t.flagNames).toEqual(['QVALID', 'SEL_C', 'INV_QUEUED']);
    expect(t.forward).toMatchObject({ code: { sc: K.SC_OK, name: 'OK' }, Jp: 1.25, JP: 0.5 });
    expect(t.inverse).toMatchObject({ code: { sc: K.SC_O1, pair: 3 }, Jinv: 0, piP: 0.25 });
    expect(t.select).toMatchObject({ wc: 1, wp: 2, sel: 1, selName: 'canonical selected', phase: 1 });
    expect(t.refresh.map((r) => [r.dir, r.fromPass, r.statusNames.join('|')])).toEqual([['fwd', false, 'DONE|RAY'], ['inv', true, 'DONE']]);
    const lines = shiftedPolylines(d, [0, 0, 10]);
    const fwd = lines.find((l) => l.path === 23)!, inv = lines.find((l) => l.path === 31)!;
    expect(fwd.pts).toEqual([[0, 0, 10], [1, 2, 3], [4, 5, 6]]);
    expect(fwd.ok).toBe(true);
    expect(inv.pts).toEqual([[0, 0, 9], [7, 8, 9]]);           // starts at the previous camera (b = 0)
    expect(inv.ok).toBe(false);                                 // inverse O1
  });
});

describe('temporal HUD lines', () => {
  it('from the arena header (words 48–63, Q_f / Q_i) and the frame RsTemporal', () => {
    const raw = new Uint32Array(ARENA_HDR_BYTES / 4);
    raw[K.RSC_T_QVALID] = 900; raw[K.RSC_T_DISOCC] = 100; raw[K.RSC_T_SEL_P] = 450; raw[K.RSC_T_FWD_QUEUED] = 90;
    raw[K.RSC_T_REFRESH_RECS] = 1000; raw[K.RSC_T_REFRESH_RAYS] = 40; raw[K.RSC_T_CLASS_UNDEF] = 2;
    const q1 = queueHdr(K.RS_Q_FWD), q2 = queueHdr(K.RS_Q_INV);
    raw[q1.n] = 90; raw[q1.capacity] = 1000; raw[q2.n] = 450; raw[q2.capacity] = 1000; raw[q2.overflow] = 1;
    const f = parseArenaHeader(raw);
    expect(temporalHudLines(f)).toEqual([]);
    f.temporal = { flags: K.TF_HIST_VALID | K.TF_REFRESH | K.TF_LIGHT_MOVED, histFrames: 17, reasons: [] };
    const l = temporalHudLines(f);
    expect(l[0]).toBe('ReSTIR temporal: hist 17 frames  TF HIST_VALID|REFRESH|LIGHT_MOVED');
    expect(l[1]).toContain('q′ valid 900  disocc 100 (10.0%)  P(s=p) 50.0%  fwd replay 10.0%  Q_f 90/1000  Q_i 450/1000 OVERFLOW');
    expect(l[2]).toContain('records 1000  rays 40');
    expect(l[2]).toContain('class-change 2');
    f.temporal = { flags: K.TF_RESET, histFrames: 0, reasons: ['config'] };
    expect(temporalHudLines(f)[0]).toBe('ReSTIR temporal: hist 0 frames RESET (config)  TF RESET');
    expect(tfFlagNames(K.TF_ENV_MOVED | K.TF_CAM_SAME)).toEqual(['ENV_MOVED', 'CAM_SAME']);
  });
});
