// ReSTIR debug views (ids 400–499), probe decoding for the pixel inspector, the rs_debug passes and the per-frame HUD
// readback of the shift-arena header (restir-api.md §2.11 + Changelog D1–D4; PLAN §6 M4 rows). OWNER WP-D.
//   RESTIR_VIEWS / RESTIR_PROBE_TAGS   registry entries (renderer mode 'restir'; app/integration.ts registers them)
//   decodeRestirProbe(records)        probe records (tags 64–72) → reservoirs, candidates, paths, slots, MIS
//   RestirDebugPass                   rs_debug_fill (code views start as DBG_CODE_NONE) + rs_debug_views (shift views
//                                     and the probe's final slot records from the arena); dispatched only while a
//                                     ReSTIR view or the probe is active (debug off ⇒ nothing is encoded)
//   RestirHud                         arena header (queues, RSC_* counters, SC histogram) → HUD lines, f_r
// WGSL: shaders/debug/restir-views.wgsl (hooks), shaders/passes/restir/debug.wgsl (passes).
// M5 (T-D; restir-temporal-api.md §2.11, Changelog D-1): temporal views 480–496 (T1 / T3 hooks) and 497 (boost mask,
// rs_debug_views), the reservoir views' tap "after temporal", probe tags 73–79 (decodeRestirProbe → `temporal`), the
// forward / inverse temporal paths of the inspector overlay (paths 23 / 31), and the temporal HUD lines (header words
// 48–63, the Q_f / Q_i headers and the frame's RsTemporal flags).
import type { DebugResources, DebugViewDef } from '../debug-views.ts';
import type { ProbeRecord } from '../probe.ts';
import type { RestirKernel } from './kernel.ts';
import {
  ARENA_HDR_BYTES, EP_TYPE, PATH_CLASS_NAMES, RES_WORDS, RS_PROBE_TAG, RS_TECH_NAMES, RS_VIEW, RS_WGSL_CONSTS as K, RSC, SC_NAMES,
  TS_CONSTS as TS, decodeReservoir, queueHdr, type ReservoirRecord, type RscName,
} from './layout.ts';
import type { RestirAdvance } from './frame-state.ts';
import { restirCommonDefines } from './resources.ts';

// ------------------------------------------------------------------------------------------------ views

/** Added by WP-D (Changelog D2): primary threshold thr of the pixel (rsGeo.w), the reference of the margin views. */
export const RS_VIEW_THR = 446;
const G_RES = 'ReSTIR reservoir', G_SHIFT = 'ReSTIR shift', G_MIS = 'ReSTIR MIS';
const TAP_NOTE = 'stage tap: after initial (rs_initial) | after spatial (final-round resample) | final (the last stage that ran)';

function slotViews(base: number, key: string, label: string, def: Partial<DebugViewDef>): DebugViewDef[] {
  return Array.from({ length: 6 }, (_, s) => ({
    id: base + s, key: `${key}[${s}]`, label: `${label} [slot ${s}]`, group: G_SHIFT, source: 'rs_debug_views', kind: 'scalar' as const, ...def,
  }));
}

/** M5 temporal view ids (restir-temporal-api.md §2.11; debug/restir-views.wgsl RSV_T_*, RSV_S_BOOST). */
export const RS_VIEW_T = {
  qvalid: 480, motion: 481, refreshFwd: 482, refreshInv: 483, fwdCode: 484, invCode: 485, logJ: 486, logJP: 487, pic: 488, pip: 489,
  cprev: 490, cout: 491, sel: 492, phatRel: 493, robust: 494, wp: 495, lightsChanged: 496, boost: 497,
} as const;
/** M5 probe tags (§2.11; 79 = internal anchor ids, Changelog D-1). */
export const RS_PROBE_TAG_T = { header: 73, forward: 74, inverse: 75, select: 76, refresh: 77, pick: 78, anchor: 79 } as const;
const G_T = 'ReSTIR temporal';
const T3 = 'rsdbg_temporal (T3 rs_t_select)';

export const RESTIR_VIEWS: DebugViewDef[] = [
  { id: RS_VIEW.c, key: 'rs.c', label: 'c (confidence)', group: G_RES, source: 'rsdbg_reservoir', kind: 'scalar', range: [0.5, 64], log: true, colormap: 'turbo', tapped: true, description: TAP_NOTE },
  { id: RS_VIEW.W, key: 'rs.W', label: 'W (UCW)', group: G_RES, source: 'rsdbg_reservoir', kind: 'scalar', range: [1e-3, 1e3], log: true, colormap: 'turbo', tapped: true },
  { id: RS_VIEW.phat, key: 'rs.phat', label: 'p̂ = lum(F)', group: G_RES, source: 'rsdbg_reservoir', kind: 'scalar', range: [1e-4, 10], log: true, colormap: 'turbo', tapped: true },
  { id: RS_VIEW.FW, key: 'rs.FW', label: 'F·W (estimate)', group: G_RES, source: 'rsdbg_reservoir', kind: 'vec3', range: [0, 1], tapped: true },
  { id: RS_VIEW.d, key: 'rs.d', label: 'd (path length, 0 empty)', group: G_RES, source: 'rsdbg_reservoir', kind: 'code', tapped: true },
  { id: RS_VIEW.k, key: 'rs.k', label: 'k (rc vertex, 0 = ∅)', group: G_RES, source: 'rsdbg_reservoir', kind: 'code', tapped: true },
  { id: RS_VIEW.tech, key: 'rs.tech', label: 'technique', group: G_RES, source: 'rsdbg_reservoir', kind: 'code', tapped: true },
  { id: RS_VIEW.endpoint, key: 'rs.endpoint', label: 'endpoint type', group: G_RES, source: 'rsdbg_reservoir', kind: 'code', tapped: true },
  { id: RS_VIEW.lobes, key: 'rs.lobes', label: 'lobes ℓk−1·16 + ℓk (+δ·256)', group: G_RES, source: 'rsdbg_reservoir', kind: 'code', tapped: true },
  { id: RS_VIEW.class, key: 'rs.class', label: 'path class (L N1 B1 E D-NEE D-BSDF R)', group: G_RES, source: 'rsdbg_reservoir', kind: 'code', tapped: true },
  { id: RS_VIEW.kMargin, key: 'rs.kMargin', label: 'k* margin (log2 footprint/thr)', group: G_RES, source: 'rsdbg_reservoir', kind: 'scalar', range: [-8, 8], colormap: 'signed', tapped: true },
  ...slotViews(RS_VIEW.shiftCode, 'shift.code', 'outcome code', { kind: 'code' }),
  ...slotViews(RS_VIEW.shiftLogJ, 'shift.logJ', 'log2 J (valid)', { range: [-8, 8], colormap: 'signed' }),
  ...slotViews(RS_VIEW.shiftTerm, 'shift.term', 'failing term | pair<<4', { kind: 'code' }),
  { id: RS_VIEW.replayMask, key: 'shift.replayMask', label: 'replay mask (queued slots)', group: G_SHIFT, source: 'rs_debug_views', kind: 'code' },
  { id: RS_VIEW.acceptMask, key: 'shift.acceptMask', label: 'accept mask', group: G_SHIFT, source: 'rs_debug_views', kind: 'code' },
  ...slotViews(RS_VIEW.shiftMargin, 'shift.margin', 'predicate margin (log2 footprint/thr)', { range: [-8, 8], colormap: 'signed' }),
  { id: RS_VIEW_THR, key: 'shift.thr', label: 'thr (primary footprint threshold)', group: G_SHIFT, source: 'rs_debug_views', kind: 'scalar', range: [1e-6, 1], log: true },
  { id: RS_VIEW.misMc, key: 'mis.mc', label: 'm_c (canonical MIS weight)', group: G_MIS, source: 'rsdbg_mis', kind: 'scalar', range: [0, 1] },
  { id: RS_VIEW.misSumM, key: 'mis.sumM', label: 'Σm − 1 at X_c (must be 0)', group: G_MIS, source: 'rsdbg_mis', kind: 'scalar', range: [-1e-5, 1e-5], colormap: 'signed' },
  { id: RS_VIEW.misLumL, key: 'mis.lumL', label: '(lum L − Σw)/Σw (must be 0)', group: G_MIS, source: 'rsdbg_mis', kind: 'scalar', range: [-1e-5, 1e-5], colormap: 'signed' },
  ...Array.from({ length: 6 }, (_, s): DebugViewDef => ({ id: RS_VIEW.misMj + s, key: `mis.mj[${s}]`, label: `m_j [slot ${s}]`, group: G_MIS, source: 'rsdbg_mis', kind: 'scalar', range: [0, 1] })),
  { id: RS_VIEW.misK, key: 'mis.k', label: 'k = |S_c|', group: G_MIS, source: 'rsdbg_mis', kind: 'code' },
  { id: RS_VIEW.misSel, key: 'mis.sel', label: 'selected (0 canonical, 1+s partner)', group: G_MIS, source: 'rsdbg_mis', kind: 'code' },
  // ---- M5 temporal (§2.11): written only on frames whose temporal stage ran (ReSTIR modes with temporal on)
  { id: RS_VIEW_T.qvalid, key: 't.qvalid', label: 'q′ validity (bg / centre / ring / disoccluded / no history)', group: G_T, source: T3, kind: 'code' },
  { id: RS_VIEW_T.motion, key: 't.motion', label: 'motion s′ − q (px; grey = 0)', group: G_T, source: 'rsdbg_tpick (T1)', kind: 'vec3', range: [-4, 4] },
  { id: RS_VIEW_T.refreshFwd, key: 't.refreshFwd', label: 'refresh of X_p (fwd): none / analytic / ray / undef / E2 / zero', group: G_T, source: T3, kind: 'code' },
  { id: RS_VIEW_T.refreshInv, key: 't.refreshInv', label: 'refresh of X_c (inv, Q_i pixels)', group: G_T, source: T3, kind: 'code' },
  { id: RS_VIEW_T.fwdCode, key: 't.fwdCode', label: 'forward shift outcome T(X_p)', group: G_T, source: T3, kind: 'code' },
  { id: RS_VIEW_T.invCode, key: 't.invCode', label: 'inverse shift outcome T⁻¹(X_c)', group: G_T, source: T3, kind: 'code' },
  { id: RS_VIEW_T.logJ, key: 't.logJ', label: 'log2 J_p = J_rc·J_P (forward)', group: G_T, source: T3, kind: 'scalar', range: [-4, 4], colormap: 'signed' },
  { id: RS_VIEW_T.logJP, key: 't.logJP', label: 'log2 J_P (light pmf ratio)', group: G_T, source: T3, kind: 'scalar', range: [-2, 2], colormap: 'signed' },
  { id: RS_VIEW_T.pic, key: 't.pic', label: 'π_c = p̂_t of the output sample', group: G_T, source: T3, kind: 'scalar', range: [1e-4, 10], log: true, colormap: 'turbo' },
  { id: RS_VIEW_T.pip, key: 't.pip', label: 'π_p (stored for s = p, recomputed for s = c)', group: G_T, source: T3, kind: 'scalar', range: [1e-4, 10], log: true, colormap: 'turbo' },
  { id: RS_VIEW_T.cprev, key: 't.cprev', label: 'c_prev (uncapped)', group: G_T, source: T3, kind: 'scalar', range: [0.5, 256], log: true, colormap: 'turbo' },
  { id: RS_VIEW_T.cout, key: 't.cout', label: 'c_out = 1 + c_p (temporal output)', group: G_T, source: T3, kind: 'scalar', range: [0.5, 32], log: true, colormap: 'turbo' },
  { id: RS_VIEW_T.sel, key: 't.sel', label: 'selection (kept / canonical / temporal / empty)', group: G_T, source: T3, kind: 'code' },
  { id: RS_VIEW_T.phatRel, key: 't.phatRel', label: 'forward vs stored p̂: (lum F_t(Y_p) − lum F_p)/max', group: G_T, source: T3, kind: 'scalar', range: [-1, 1], colormap: 'signed' },
  { id: RS_VIEW_T.robust, key: 't.robust', label: 'robust: (π_p recomputed − stored)/max', group: G_T, source: T3, kind: 'scalar', range: [-1e-3, 1e-3], colormap: 'signed' },
  { id: RS_VIEW_T.wp, key: 't.wp', label: 'w̃_p / (w̃_c + w̃_p)', group: G_T, source: T3, kind: 'scalar', range: [0, 1] },
  { id: RS_VIEW_T.lightsChanged, key: 't.lightsChanged', label: 'X_p light changed: moved | radiometric | undefined', group: G_T, source: T3, kind: 'code' },
  { id: RS_VIEW_T.boost, key: 's.boost', label: 'accepted boost slots (bit s − slots)', group: G_SHIFT, source: 'rs_debug_views', kind: 'code' },
];

export const isRestirView = (mode: number): boolean => mode >= 400 && mode < 500;
export const isRestirShiftView = (mode: number): boolean => mode >= RS_VIEW.shiftCode && mode <= RS_VIEW_THR;
/** Views written by rs_debug_views (shift views and the boost mask). */
export const isRestirDebugPassView = (mode: number): boolean => isRestirShiftView(mode) || mode === RS_VIEW_T.boost;
export const isRestirTemporalView = (mode: number): boolean => mode >= RS_VIEW_T.qvalid && mode <= RS_VIEW_T.boost;
export const isRestirCodeView = (mode: number): boolean => RESTIR_VIEWS.some((v) => v.id === mode && v.kind === 'code');

export const RESTIR_PROBE_TAGS: [number, string][] = [
  [RS_PROBE_TAG.plane, 'rs.plane'], [RS_PROBE_TAG.header, 'rs.reservoir'], [RS_PROBE_TAG.candidate, 'rs.candidate'],
  [RS_PROBE_TAG.vertex, 'rs.vertex'], [RS_PROBE_TAG.slot, 'rs.slotEvent'], [RS_PROBE_TAG.misPartner, 'rs.mis'],
  [RS_PROBE_TAG.misCanonical, 'rs.misCanon'], [71, 'rs.slotOut'], [72, 'rs.slotIn'],
  [73, 't.header'], [74, 't.forward'], [75, 't.inverse'], [76, 't.select'], [77, 't.refresh'], [78, 't.pick'], [79, 't.anchor'],
];

// ------------------------------------------------------------------------------------------------ code legends

const LT_NAMES: Record<number, string> = Object.fromEntries(Object.entries(EP_TYPE).map(([k, v]) => [v, k]));
export const LOBE_NAMES = ['D', 'S', 'G_R', 'G_T', 'NEE', 'NONE', '?', '?'] as const;
export const RCT_NAMES = ['none', 'D', 'R', 'F', 'I', 'guard'] as const;
export const QVALID_NAMES = ['background', 'centre tap', 'ring tap', 'no q′ (disoccluded)', 'no history (reset)'] as const;
export const REFRESH_CLASS_NAMES = ['not refreshed', 'analytic (no ray)', 'ray traced', 'undefined', 'E2 zeroed', 'zero'] as const;
export const TSEL_NAMES = ['canonical kept (no q′)', 'canonical selected', 'temporal selected', 'empty'] as const;
const LCHG_BITS = ['moved', 'radiometric', 'undefined'] as const;

/** Human-readable meaning of a code value of view `id` (legend, inspector). */
export function codeName(id: number, code: number): string {
  if (code === 0xFFFFFFFF) return 'none';
  if (id === RS_VIEW.d || id === RS_VIEW.k) return code === 0 ? (id === RS_VIEW.d ? 'empty' : '∅') : String(code);
  if (id === RS_VIEW.tech) return RS_TECH_NAMES[code] ?? String(code);
  if (id === RS_VIEW.endpoint) return LT_NAMES[code] ?? String(code);
  if (id === RS_VIEW.lobes) {
    const d = code >> 8;
    return `${LOBE_NAMES[(code >> 4) & 7]}${d & 1 ? 'δ' : ''} → ${LOBE_NAMES[code & 7]}${d & 2 ? 'δ' : ''}`;
  }
  if (id === RS_VIEW.class) return PATH_CLASS_NAMES[code] ?? String(code);
  if (id >= RS_VIEW.shiftCode && id < RS_VIEW.shiftCode + 6) return SC_NAMES[code] ?? String(code);
  if (id >= RS_VIEW.shiftTerm && id < RS_VIEW.shiftTerm + 6) return `${RCT_NAMES[code & 0xF] ?? code & 0xF}${code >> 4 ? ` pair ${code >> 4}` : ''}`;
  if (id === RS_VIEW.replayMask || id === RS_VIEW.acceptMask) return code.toString(2).padStart(6, '0').split('').reverse().join('') + ' (slot 0 first)';
  if (id === RS_VIEW.misSel) return code === 0 ? 'canonical' : `partner slot ${code - 1}`;
  if (id === RS_VIEW_T.qvalid) return QVALID_NAMES[code] ?? String(code);
  if (id === RS_VIEW_T.refreshFwd || id === RS_VIEW_T.refreshInv) return REFRESH_CLASS_NAMES[code] ?? String(code);
  if (id === RS_VIEW_T.fwdCode || id === RS_VIEW_T.invCode) return SC_NAMES[code] ?? String(code);
  if (id === RS_VIEW_T.sel) return TSEL_NAMES[code] ?? String(code);
  if (id === RS_VIEW_T.lightsChanged) return code === 0 ? 'unchanged' : LCHG_BITS.filter((_, b) => code & (1 << b)).join(' + ');
  if (id === RS_VIEW_T.boost) return code.toString(2).padStart(3, '0').split('').reverse().join('') + ' (boost slot 0 first)';
  return String(code);
}

/** The code values a legend lists for view `id` (undefined: open-ended numbers). */
export function legendCodes(id: number): number[] | undefined {
  if (id === RS_VIEW.tech) return [0, 1, 2, 3];
  if (id === RS_VIEW.endpoint) return Object.values(EP_TYPE);
  if (id === RS_VIEW.class) return [0, 1, 2, 3, 4, 5, 6];
  if (id >= RS_VIEW.shiftCode && id < RS_VIEW.shiftCode + 6) return SC_NAMES.map((_, i) => i);
  if (id === RS_VIEW.d || id === RS_VIEW.k) return Array.from({ length: 10 }, (_, i) => i);
  if (id === RS_VIEW_T.qvalid) return [0, 1, 2, 3, 4];
  if (id === RS_VIEW_T.refreshFwd || id === RS_VIEW_T.refreshInv) return [0, 1, 2, 3, 4, 5];
  if (id === RS_VIEW_T.fwdCode || id === RS_VIEW_T.invCode) return SC_NAMES.map((_, i) => i);
  if (id === RS_VIEW_T.sel) return [0, 1, 2, 3];
  if (id === RS_VIEW_T.lightsChanged || id === RS_VIEW_T.boost) return [0, 1, 2, 3, 4, 5, 6, 7];
  return undefined;
}

/** TS port of debug-common.wgsl cmap_code (sRGB-encoded [0,1]); DBG_CODE_NONE → black. */
export function cmapCode(code: number): [number, number, number] {
  if (code >>> 0 === 0xFFFFFFFF) return [0, 0, 0];
  let x = (Math.imul(code >>> 0, 1664525) + 1013904223) >>> 0;
  let y = (Math.imul(0x2545f491, 1664525) + 1013904223) >>> 0;
  let z = (Math.imul(0x9e3779b9, 1664525) + 1013904223) >>> 0;
  x = (x + Math.imul(y, z)) >>> 0; y = (y + Math.imul(z, x)) >>> 0; z = (z + Math.imul(x, y)) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0; y = (y ^ (y >>> 16)) >>> 0; z = (z ^ (z >>> 16)) >>> 0;
  x = (x + Math.imul(y, z)) >>> 0; y = (y + Math.imul(z, x)) >>> 0; z = (z + Math.imul(x, y)) >>> 0;
  return [0.15 + 0.85 * (x & 255) / 255, 0.15 + 0.85 * (y & 255) / 255, 0.15 + 0.85 * (z & 255) / 255];
}

// ------------------------------------------------------------------------------------------------ probe decoding

export interface SlotCode { sc: number; name: string; term: number; pair: number; margin: number }
export function decodeSlotCode(code: number): SlotCode {
  const sc = code & 0xFF;
  return { sc, name: SC_NAMES[sc] ?? `?${sc}`, term: (code >>> 8) & 0xF, pair: (code >>> 12) & 0xF, margin: f16ToF32(code >>> 16) };
}
/** J word (D7) → status or J. */
export function decodeJWord(w: number): { status: 'VALID' | 'FAILED' | 'PENDING' | 'NOT_ACCEPTED'; J: number } {
  if (w >>> 0 === K.JW_NOT_ACCEPTED) return { status: 'NOT_ACCEPTED', J: 0 };
  if (w >>> 0 === K.JW_PENDING) return { status: 'PENDING', J: 0 };
  if (w === 0) return { status: 'FAILED', J: 0 };
  return { status: 'VALID', J: new Float32Array(Uint32Array.of(w).buffer)[0] };
}
export function f16ToF32(h: number): number {
  const s = h & 0x8000 ? -1 : 1, e = (h >>> 10) & 0x1F, m = h & 0x3FF;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

export interface RsProbeReservoir { tap: number; pass: number; round: number; ai: number; words: Uint32Array; rec: ReservoirRecord }
export interface RsProbeCandidate { d: number; tech: number; k: number; selected: boolean; w: number; lumF: number; counter: number; tree: number; B: number; slotInVertex: number }
export interface RsProbeVertex { path: number; b: number; lobe: number; delta: boolean; pos: [number, number, number] }
export interface RsProbeSlotEvent { s: number; code: SlotCode; J: number; replayed: boolean }
export interface RsProbeSlot {
  s: number; code: SlotCode; j: ReturnType<typeof decodeJWord>; queued: boolean; partnerAi: number | undefined;
  /** The partner's path shifted into the probe pixel (slot s of the partner). */
  incoming?: { code: SlotCode; j: ReturnType<typeof decodeJWord>; lumFJ: number };
}
export interface RsProbeMis {
  canonical?: { k: number; sel: number; mc: number; sumM: number; lumRel: number; wc: number };
  partners: { s: number; m: number; w: number }[];
}
/** M5 temporal probe records of the probe pixel (tags 73–78). */
export interface RsProbeTemporal {
  ai?: number; qPrime?: number; cPrev?: number; flags?: number; flagNames: string[];
  forward?: { code: SlotCode; Jp: number; JP: number; lumF: number };
  inverse?: { code: SlotCode; Jinv: number; lumFPrev: number; piP: number };
  select?: { wc: number; wp: number; piC: number; sel: number; selName: string; phase: number };
  refresh: { dir: 'fwd' | 'inv'; fromPass: boolean; status: number; statusNames: string[]; lumRad: number; aux: number; entryTo: number }[];
  pick?: { sp: [number, number]; tap: number; valid: boolean };
}
const TS_FLAG_NAMES = Object.entries(TS).filter(([k]) => k.startsWith('TS_')).map(([k, v]) => [k.slice(3), v] as const);
const SXS_NAMES = Object.entries(TS).filter(([k]) => k.startsWith('SXS_')).map(([k, v]) => [k.slice(4), v] as const);
export const tsFlagNames = (f: number): string[] => TS_FLAG_NAMES.filter(([, v]) => f & v).map(([k]) => k);
export const sxsNames = (f: number): string[] => SXS_NAMES.filter(([, v]) => f & v).map(([k]) => k);

export interface RestirProbeDecoded {
  /** M5: the temporal records (undefined when no temporal stage ran for the probe pixel). */
  temporal?: RsProbeTemporal;
  reservoirs: RsProbeReservoir[];
  candidates: RsProbeCandidate[];
  vertices: RsProbeVertex[];
  /** Vertices grouped by path id (0 base tree, 1+s shifted into slot s's partner, 8+s partner → probe), sorted by b. */
  paths: Map<number, RsProbeVertex[]>;
  slotEvents: RsProbeSlotEvent[];
  slots: RsProbeSlot[];
  mis: RsProbeMis;
}

const u32Of = (r: ProbeRecord, i: number) => r.bits[i] >>> 0;

/** Decode the ReSTIR probe records of one frame (tags 64–79; other tags are ignored). Records must be in seq order. */
export function decodeRestirProbe(records: readonly ProbeRecord[]): RestirProbeDecoded {
  const out: RestirProbeDecoded = { reservoirs: [], candidates: [], vertices: [], paths: new Map(), slotEvents: [], slots: [], mis: { partners: [] } };
  const T = RS_PROBE_TAG;
  const TT = RS_PROBE_TAG_T;
  const tmp = (): RsProbeTemporal => (out.temporal ??= { flagNames: [], refresh: [] });
  for (let i = 0; i < records.length; i++) {
    const r = records[i];
    switch (r.tag) {
      case TT.header: {
        const t = tmp(), q = u32Of(r, 1), f = u32Of(r, 3);
        t.ai = u32Of(r, 0); t.qPrime = q === 0xFFFFFFFF ? undefined : q; t.cPrev = r.value[2]; t.flags = f; t.flagNames = tsFlagNames(f);
        break;
      }
      case TT.forward: tmp().forward = { code: decodeSlotCode(u32Of(r, 0)), Jp: r.value[1], JP: r.value[2], lumF: r.value[3] }; break;
      case TT.inverse: tmp().inverse = { code: decodeSlotCode(u32Of(r, 0)), Jinv: r.value[1], lumFPrev: r.value[2], piP: r.value[3] }; break;
      case TT.select: {
        const c = u32Of(r, 3);
        tmp().select = { wc: r.value[0], wp: r.value[1], piC: r.value[2], sel: c & 0xFF, selName: TSEL_NAMES[c & 0xFF] ?? String(c & 0xFF), phase: (c >>> 8) & 0xFF };
        break;
      }
      case TT.refresh: {
        const st = u32Of(r, 0);
        tmp().refresh.push({ dir: (st >>> 16) & 1 ? 'inv' : 'fwd', fromPass: ((st >>> 17) & 1) === 1, status: st & 0xFFFF, statusNames: sxsNames(st & 0xFFFF), lumRad: r.value[1], aux: r.value[2], entryTo: u32Of(r, 3) });
        break;
      }
      case TT.pick: tmp().pick = { sp: [r.value[0], r.value[1]], tap: u32Of(r, 2), valid: u32Of(r, 3) !== 0 }; break;
      case T.header: {
        const words = new Uint32Array(RES_WORDS);
        let n = 0;
        for (; n < 10 && i + 1 + n < records.length && records[i + 1 + n].tag === T.plane; n++) words.set(records[i + 1 + n].bits.map((x) => x >>> 0), 4 * n);
        if (n === 10) out.reservoirs.push({ pass: u32Of(r, 0), round: u32Of(r, 1), ai: u32Of(r, 2), tap: u32Of(r, 3), words, rec: decodeReservoir(words, 0) });
        i += n;
        break;
      }
      case T.candidate: {
        const c = u32Of(r, 0), counter = u32Of(r, 3);
        out.candidates.push({
          d: c & 0xF, tech: (c >>> 4) & 0xF, k: (c >>> 8) & 0xF, selected: ((c >>> 12) & 1) === 1, w: r.value[1], lumF: r.value[2], counter,
          tree: counter >>> 20, B: (counter >>> 12) & 0xFF, slotInVertex: counter & 0xFFF,
        });
        break;
      }
      case T.vertex: {
        const c = u32Of(r, 3);
        const v: RsProbeVertex = { path: c >>> 8, b: (c >>> 4) & 0xF, lobe: c & 7, delta: (c & 8) !== 0, pos: [r.value[0], r.value[1], r.value[2]] };
        out.vertices.push(v);
        if (!out.paths.has(v.path)) out.paths.set(v.path, []);
        out.paths.get(v.path)!.push(v);
        break;
      }
      case T.slot: out.slotEvents.push({ code: decodeSlotCode(u32Of(r, 0)), J: r.value[1], s: u32Of(r, 2), replayed: u32Of(r, 3) !== 0 }); break;
      case T.misPartner: {
        const s = u32Of(r, 0);
        if (s === 0xFF) { out.mis.canonical ??= { k: 0, sel: 0, mc: r.value[1], sumM: 0, lumRel: 0, wc: 0 }; out.mis.canonical.wc = r.value[2]; }
        else out.mis.partners.push({ s, m: r.value[1], w: r.value[2] });
        break;
      }
      case T.misCanonical: {
        const c = u32Of(r, 0);
        out.mis.canonical = { wc: out.mis.canonical?.wc ?? 0, k: c & 0xFF, sel: (c >>> 8) & 0xFF, mc: r.value[1], sumM: r.value[2], lumRel: r.value[3] };
        break;
      }
      case 71: {
        const f = u32Of(r, 2);
        out.slots.push({ s: f & 0xFF, code: decodeSlotCode(u32Of(r, 0)), j: decodeJWord(u32Of(r, 1)), queued: (f & 0x100) !== 0, partnerAi: f & 0x200 ? u32Of(r, 3) : undefined });
        break;
      }
      case 72: {
        const s = u32Of(r, 2);
        const slot = out.slots.find((x) => x.s === s && !x.incoming);
        const inc = { code: decodeSlotCode(u32Of(r, 0)), j: decodeJWord(u32Of(r, 1)), lumFJ: r.value[3] };
        if (slot) slot.incoming = inc;
        break;
      }
      default: break;
    }
  }
  for (const p of out.paths.values()) p.sort((a, b) => a.b - b.b);
  return out;
}

/** Temporal overlay paths (M5): forward T(X_p) into the probe pixel, inverse into q′ at t−1 (from the previous camera). */
export const PATH_T_FWD = 23;
export const PATH_T_INV = 31;

/** Path colour of the inspector overlay: base tree white, p→partner slots warm, partner→p slots cool; temporal forward
 *  magenta, temporal inverse lime. */
export function pathColour(path: number): [number, number, number, number] {
  if (path === 0) return [1, 1, 1, 1];
  if (path === PATH_T_FWD) return [1, 0.2, 1, 1];
  if (path === PATH_T_INV) return [0.6, 1, 0.1, 1];
  const incoming = path >= 24 || (path >= 8 && path < 16);
  const s = path >= 24 ? path - 24 : path >= 16 ? path - 16 : path >= 8 ? path - 8 : path - 1;
  const warm: [number, number, number][] = [[1, 0.35, 0.2], [1, 0.7, 0.1], [0.95, 0.9, 0.2], [1, 0.4, 0.7], [0.9, 0.55, 0.3], [0.8, 0.3, 0.3]];
  const cool: [number, number, number][] = [[0.2, 0.6, 1], [0.2, 0.9, 0.9], [0.4, 1, 0.5], [0.6, 0.5, 1], [0.3, 0.8, 0.7], [0.5, 0.7, 1]];
  const c = (incoming ? cool : warm)[s % 6];
  return [c[0], c[1], c[2], 1];
}

/**
 * Polylines of the inspector overlay (render-frame positions, camera prepended): the base path (path 0 vertices, up
 * to the probe reservoir's d), every p→partner shift (anchor y₁ of the partner (path 16+s), then the base path from
 * x_k on; k = ∅: y₁ only; the replayed prefix y₂…y_{k−1} of k > 2 is not recorded and drawn as the straight y₁→x_k)
 * and every partner→p shift (the probe's y₁, the partner's x_k and surface endpoint (path 24+s)). Paths recorded by
 * rsdbg_vertex for 1+s / 8+s are drawn as recorded. M5: path 23 = the temporal forward shift T(X_p) into the probe pixel
 * (camera, y₁, X_p's x_k and surface endpoint), path 31 = the inverse shift into q′ at t−1 (previous camera, y₁′,
 * x_k, surface endpoint).
 */
export function shiftedPolylines(d: RestirProbeDecoded, cam: [number, number, number]): { path: number; pts: [number, number, number][]; ok: boolean }[] {
  const out: { path: number; pts: [number, number, number][]; ok: boolean }[] = [];
  const src = d.reservoirs.find((r) => r.tap === 1)?.rec;
  const base = (d.paths.get(0) ?? []).filter((v) => !src || src.d === 0 || v.b <= src.d);
  if (base.length) out.push({ path: 0, pts: [cam, ...base.map((v) => v.pos)], ok: true });
  for (const [path, vs] of d.paths) {
    if (path === 0 || !vs.length) continue;
    if (path === PATH_T_FWD) { out.push({ path, pts: [cam, ...vs.map((v) => v.pos)], ok: d.temporal?.forward?.code.sc === K.SC_OK && d.temporal.forward.Jp > 0 }); continue; }
    if (path === PATH_T_INV) {
      // b = 0 is the previous camera (recorded by rs_debug_views); without it the current camera stands in.
      const pts = vs[0].b === 0 ? vs.map((v) => v.pos) : [cam, ...vs.map((v) => v.pos)];
      out.push({ path, pts, ok: d.temporal?.inverse?.code.sc === K.SC_OK && d.temporal.inverse.Jinv > 0 });
      continue;
    }
    if (path < 16) { out.push({ path, pts: [cam, ...vs.map((v) => v.pos)], ok: true }); continue; }
    const s = path >= 24 ? path - 24 : path - 16;
    const slot = d.slots.find((x) => x.s === s);
    if (path < 24) {
      const k = src?.k ?? 0;
      const tail = k >= 2 ? base.filter((v) => v.b >= k).map((v) => v.pos) : [];
      out.push({ path, pts: [cam, ...vs.map((v) => v.pos), ...tail], ok: slot?.j.status === 'VALID' });
    } else {
      out.push({ path, pts: [cam, ...vs.map((v) => v.pos)], ok: slot?.incoming?.j.status === 'VALID' });
    }
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ rs_debug passes

/** Encoded through kernel.encodePass with this pass-table entry's group shape (G1 scene, G3 debug). */
const HOST_PASS = 'rs_spatial_shift' as const;

export class RestirDebugPass {
  private readonly g2 = new Map<number, GPUBindGroup>();
  private g2Key: unknown;
  private readonly layout: GPUBindGroupLayout;

  private constructor(readonly kernel: RestirKernel, readonly debug: DebugResources, private readonly views: GPUComputePipeline, private readonly fill: GPUComputePipeline, layout: GPUBindGroupLayout) {
    this.layout = layout;
  }

  static async create(kernel: RestirKernel, debug: DebugResources): Promise<RestirDebugPass> {
    if (kernel.layouts.g3 !== debug.layout) throw new Error('RestirDebugPass: the kernel must be created with these debug resources');
    const c = GPUShaderStage.COMPUTE;
    const g2 = kernel.device.createBindGroupLayout({
      label: 'rs-debug-g2',
      entries: [
        { binding: 0, visibility: c, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: c, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: c, texture: { sampleType: 'uint' } },
        { binding: 3, visibility: c, texture: { sampleType: 'unfilterable-float' } },
        { binding: 4, visibility: c, texture: { sampleType: 'sint', viewDimension: '2d-array' } },
        { binding: 5, visibility: c, texture: { sampleType: 'uint' } },
      ],
    });
    const layout = kernel.device.createPipelineLayout({ label: 'rs-debug', bindGroupLayouts: [kernel.layouts.g0, kernel.layouts.g1Scene, g2, kernel.layouts.g3] });
    const defines = {
      ...restirCommonDefines(kernel.scene.defines(1), true),
      RS_ARENA_BINDING: '1u', RS_ARENA_RW: false, RS_VBUF_BINDING: '2u', RS_GEO_BINDING: '3u', RS_PAIRTEX_BINDING: '4u',
      RS_VBUF_PREV_BINDING: '5u',
    };
    const [views, fill] = await Promise.all([
      kernel.compile('passes/restir/debug.wgsl', 'rs_debug_views', defines, layout, 'rs_debug_views'),
      kernel.compile('passes/restir/debug.wgsl', 'rs_debug_fill', defines, layout, 'rs_debug_fill'),
    ]);
    return new RestirDebugPass(kernel, debug, views, fill, g2);
  }

  /** G2 with the reservoir buffer the last spatial round read (res[(w + rounds − 1) % 2], the slots' source paths;
   *  w = kernel.resBase(), M5 TD2), the current-parity G-buffer and the previous V-buffer (M5 §2.6). */
  private group(rounds: number): GPUBindGroup {
    const r = this.kernel.resources;
    if (this.g2Key !== r) { this.g2.clear(); this.g2Key = r; }
    const src = (this.kernel.resBase() + Math.max(rounds - 1, 0)) % 2;
    const key = src + 2 * r.parity;
    let g = this.g2.get(key);
    if (!g) {
      g = this.kernel.device.createBindGroup({
        label: `rs-debug-g2-${src}`, layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: r.res[src] } }, { binding: 1, resource: { buffer: r.arena } },
          { binding: 2, resource: r.views.vbuf }, { binding: 3, resource: r.views.geo }, { binding: 4, resource: r.views.pair },
          { binding: 5, resource: r.views.vbufPrev },
        ],
      });
      this.g2.set(key, g);
    }
    return g;
  }

  private work(): [number, number] {
    const a = this.kernel.resources.alloc;
    return this.kernel.perPixelWorkgroups(0, a.atlasH);
  }

  /** Before the ReSTIR passes of a frame: AOV := DBG_CODE_NONE while a ReSTIR code view is active. */
  encodeBegin(enc: GPUCommandEncoder): void {
    if (!isRestirCodeView(this.debug.settings.mode)) return;
    this.kernel.encodePass(enc, HOST_PASS, this.fill, this.group(0), { rowBase: 0, rowEnd: this.kernel.resources.alloc.atlasH }, this.work());
  }

  /** After the frame's ReSTIR passes (same submit): shift views + probe slot records. `rounds` = executed spatial rounds
   *  (kernel.lastRounds), `t` = the frame's seed index for sequential kernels (interactive kernels use frame.seedIndex). */
  encodeViews(enc: GPUCommandEncoder, o: { rounds: number; t?: number }): void {
    const s = this.debug.settings;
    if (!isRestirDebugPassView(s.mode) && !s.probeEnabled) return;
    const a = this.kernel.resources.alloc;
    this.kernel.encodePass(enc, HOST_PASS, this.views, this.group(o.rounds), { t: o.t ?? 0, round: o.rounds, rowBase: 0, rowEnd: a.atlasH }, this.work());
  }
}

// ------------------------------------------------------------------------------------------------ HUD readback

export interface RestirHudFrame {
  queues: { counter: number; n: number; capacity: number; overflow: number }[]; rsc: Record<RscName, number>; codes: number[]; fr: number;
  /** M5: the frame's temporal state (undefined: no temporal stage ran in that frame). */
  temporal?: RestirHudTemporal;
}
/** Temporal state of one HUD frame (RsTemporal of the frame, restir-temporal-api.md §2.7). */
export interface RestirHudTemporal { flags: number; histFrames: number; reasons: string[] }

export function parseArenaHeader(raw: Uint32Array): RestirHudFrame {
  const queues = [0, 1, 2, 3].map((q) => { const h = queueHdr(q); return { counter: raw[h.counter], n: raw[h.n], capacity: raw[h.capacity], overflow: raw[h.overflow] }; });
  const rsc = Object.fromEntries(Object.entries(RSC).map(([k, w]) => [k, raw[w]])) as Record<RscName, number>;
  const codes = Array.from(raw.subarray(K.RSC_CODE_BASE, K.RSC_CODE_BASE + 16));
  return { queues, rsc, codes, fr: rsc.accepted > 0 ? rsc.queued / rsc.accepted : 0 };
}

const TF_NAMES = Object.entries(K).filter(([k]) => k.startsWith('TF_')).map(([k, v]) => [k.slice(3), v as number] as const);
/** Names of the set RsTemporal.flags bits (TF_*). */
export const tfFlagNames = (f: number): string[] => TF_NAMES.filter(([, v]) => f & v).map(([k]) => k);

/** Temporal HUD lines of one frame (restir-temporal-api.md §2.11 "HUD"). */
export function temporalHudLines(f: RestirHudFrame): string[] {
  const t = f.temporal;
  if (!t) return [];
  const r = f.rsc, q1 = f.queues[K.RS_Q_FWD], q2 = f.queues[K.RS_Q_INV];
  const pix = r.tQvalid + r.tDisocc;
  const pct = (a: number, b: number) => (b > 0 ? `${(100 * a / b).toFixed(1)}%` : '-');
  return [
    `ReSTIR temporal: hist ${t.histFrames} frames${t.flags & K.TF_HIST_VALID ? '' : ` RESET (${t.reasons.join(', ') || 'reset'})`}  TF ${tfFlagNames(t.flags).join('|') || '-'}`,
    `ReSTIR temporal: q′ valid ${r.tQvalid}  disocc ${r.tDisocc} (${pct(r.tDisocc, pix)})  P(s=p) ${pct(r.tSelP, r.tQvalid)}  fwd replay ${pct(r.tFwdQueued, r.tQvalid)}`
      + `  Q_f ${q1.n}/${q1.capacity}${q1.overflow ? ' OVERFLOW' : ''}  Q_i ${q2.n}/${q2.capacity}${q2.overflow ? ' OVERFLOW' : ''}  fwd OK ${r.tFwdOk}  inv OK ${r.tInvOk}/${r.tInvQueued}  empty ${r.tEmptyOut}`,
    `ReSTIR refresh: records ${r.tRefreshRecs}  rays ${r.tRefreshRays}  light-undef ${r.tLightUndef}  class-change ${r.tClassUndef}  E2 zeroed ${r.tE2Zeroed}  robust mismatch ${r.tRobustMismatch}`,
  ];
}

type HudSlot = { buf: GPUBuffer; state: 'free' | 'copied' | 'mapping'; temporal?: RestirHudTemporal };

/**
 * Per-frame arena header readback (queue occupancy of the last round, RSC_* counters, SC histogram, f_r; M5: the
 * temporal counters 48–63, Q_f / Q_i and the frame's RsTemporal flags). The counter words 16–63 are cleared at the
 * start of every frame (interactive kernels only; batch runs read them with RestirKernel.readCounters). `totals`
 * accumulates the error counters since the last reset.
 */
export class RestirHud {
  private readonly ring: HudSlot[] = [];
  latest: RestirHudFrame | undefined;
  readonly totals = {
    candNonFinite: 0, shiftNonFinite: 0, wNonFinite: 0, pendingLeft: 0, slotMismatch: 0, bvhOverflow: 0, bvhItercap: 0, queueOverflow: 0,
    tNonFinite: 0, tPendingLeft: 0, tQueueOverflow: 0, frames: 0, temporalFrames: 0, tQvalid: 0, tFwdOk: 0, tSelP: 0,
  };

  constructor(private readonly device: GPUDevice) {}

  resetTotals(): void { for (const k of Object.keys(this.totals) as (keyof RestirHud['totals'])[]) this.totals[k] = 0; }

  /** Clear the per-frame counters (before the frame's ReSTIR passes). */
  encodeBegin(enc: GPUCommandEncoder, arena: GPUBuffer): void {
    enc.clearBuffer(arena, 64, ARENA_HDR_BYTES - 64);
  }

  /** Copy the header after the frame's ReSTIR passes; maps copies of earlier (already submitted) frames. `adv` = the
   *  frame's RestirAdvance when its temporal stage ran. */
  encodeEnd(enc: GPUCommandEncoder, arena: GPUBuffer, adv?: Pick<RestirAdvance, 'flags' | 'reasons'> & { temporal: { histFrames: number } }): void {
    for (const s of this.ring) if (s.state === 'copied') this.map(s);
    let s = this.ring.find((x) => x.state === 'free');
    if (!s && this.ring.length < 4) {
      s = { buf: this.device.createBuffer({ label: `rs-hud${this.ring.length}`, size: ARENA_HDR_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), state: 'free' };
      this.ring.push(s);
    }
    if (!s) return;
    enc.copyBufferToBuffer(arena, 0, s.buf, 0, ARENA_HDR_BYTES);
    s.state = 'copied';
    s.temporal = adv ? { flags: adv.flags, histFrames: adv.temporal.histFrames, reasons: [...adv.reasons] } : undefined;
  }

  private map(s: HudSlot): void {
    s.state = 'mapping';
    const temporal = s.temporal;
    s.buf.mapAsync(GPUMapMode.READ).then(() => {
      const raw = new Uint32Array(s.buf.getMappedRange().slice(0));
      s.buf.unmap();
      s.state = 'free';
      const f = parseArenaHeader(raw);
      f.temporal = temporal;
      this.latest = f;
      const t = this.totals;
      t.candNonFinite += f.rsc.candNonFinite; t.shiftNonFinite += f.rsc.shiftNonFinite; t.wNonFinite += f.rsc.wNonFinite;
      t.pendingLeft += f.rsc.pendingLeft; t.slotMismatch += f.rsc.slotMismatch; t.bvhOverflow += f.rsc.bvhOverflow; t.bvhItercap += f.rsc.bvhItercap;
      t.queueOverflow += f.queues[0].overflow; t.frames++;
      if (temporal) {
        t.temporalFrames++;
        t.tNonFinite += f.rsc.tNonFinite; t.tPendingLeft += f.rsc.tPendingLeft;
        t.tQueueOverflow += f.queues[K.RS_Q_FWD].overflow + f.queues[K.RS_Q_INV].overflow;
        t.tQvalid += f.rsc.tQvalid; t.tFwdOk += f.rsc.tFwdOk; t.tSelP += f.rsc.tSelP;
      }
    }).catch(() => { s.state = 'free'; });
  }

  /** Error counters since the last reset (M4 + M5 temporal). */
  errorCount(): number {
    const t = this.totals;
    return t.candNonFinite + t.shiftNonFinite + t.wNonFinite + t.pendingLeft + t.slotMismatch + t.bvhOverflow + t.bvhItercap + t.queueOverflow
      + t.tNonFinite + t.tPendingLeft + t.tQueueOverflow;
  }

  /** HUD lines (queue occupancy and f_r of the last frame read back, SC histogram, temporal state, error counters). */
  lines(): string[] {
    const f = this.latest;
    if (!f) return ['ReSTIR counters: waiting for readback'];
    const q = f.queues[0];
    const hist = f.codes.map((n, sc) => [sc, n] as const).filter(([, n]) => n > 0).sort((a, b) => b[1] - a[1]).slice(0, 6)
      .map(([sc, n]) => `${SC_NAMES[sc]} ${n}`).join('  ');
    const t = this.totals;
    const errs = this.errorCount();
    return [
      `ReSTIR f_r ${f.fr.toFixed(3)}  accepted ${f.rsc.accepted}  queued ${f.rsc.queued}  q0 ${q.n}/${q.capacity} (last round)${q.overflow ? ' OVERFLOW' : ''}  shifted-selected ${f.rsc.selectedShifted}`,
      `ReSTIR SC ${hist || '(no slots)'}`,
      ...temporalHudLines(f),
      `ReSTIR errors ${errs}${errs ? ' !' : ''} (cand/shift/W non-finite ${t.candNonFinite}/${t.shiftNonFinite}/${t.wNonFinite}, pending ${t.pendingLeft}, mismatch ${t.slotMismatch}, BVH ${t.bvhOverflow}/${t.bvhItercap}, q overflow ${t.queueOverflow}`
        + `; temporal non-finite ${t.tNonFinite}, pending ${t.tPendingLeft}, Q_f/Q_i overflow ${t.tQueueOverflow}; ${t.frames} frames, ${t.temporalFrames} temporal)`,
    ];
  }

  destroy(): void { for (const s of this.ring) s.buf.destroy(); }
}
