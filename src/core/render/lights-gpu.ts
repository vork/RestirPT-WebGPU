// Lights on the GPU (plan §1.4, §1.8 "records", §3 step 0; math.md#units-lights, #light-selection, #measure).
// WGSL mirror: src/core/shaders/lights/lights.wgsl (LightRec, LightSlot, LightsParams; keep offsets in sync).
//
// One read-only storage buffer `records` (array<u32>) holds, at word offsets given by the LightsParams uniform:
//   luts     : the Cycles BSDF LUT block (luts/lut-layout.ts, f32 bits) at word LUT_RECORDS_BASE (plan §1.8)
//   tris     : per emissive-triangle entry (primId, bits(area))                         static per scene
//   primMap  : primId → emissive entry index or 0xffffffff                              static per scene
//   slot 0/1 : cur/prev double buffer, each { lights (LIGHT_REC_WORDS per light, stable-id order) | alias (q, alias)
//              pairs | realized pmf (f32 per entry) | curToPrev | prevToCur }
//   env      : the env importance tables (M3c, static per env map + importance resolution): rowAlias (H_m u32),
//              colAlias (H_m·W_m u32), pdfUV (H_m·W_m f32 bits); offsets + log2 W_m in LightsParams (plan §1.8)
// Global alias entries (math.md#light-selection): analytic lights in stable-id order, then emissive triangles in primId
// order, then ENV (entry nA + nT, when an env with env NEE on is set). Table size n = 2^m (power-of-two padding with
// zero-weight entries that always alias to real ones, math.md open item 5); the realized pmf is computed from the
// stored u16 thresholds (alias.ts). With the env present and other emitters: P(env) = clamp(Φ_env/ΣΦ̃, 0.1, 0.9), the
// others share 1 − P(env) in proportion to Φ̃; env alone: P(env) = 1. Env rotation never changes the pmf; strength,
// tint or the env-NEE toggle rebuild it (radiometric / config change).
// update(lights) writes the next state into the non-current slot and flips: `prev` is the last frame's slot, so the
// maps and pmf_{t−1} stay available for temporal reuse (M5). The alias table and pmf are rebuilt ONLY when a light's
// power proxy or the set of lights changes (deterministic: identical inputs → identical bits); otherwise they are
// copied bitwise (pmf_t ≡ pmf_{t−1}, J_P = 1 exactly). Rigid motion never rebuilds.
import type { LightData, SceneData } from '../scene/types.ts';
import { buildAliasTable, packAliasEntries, type AliasTable } from './alias.ts';
import { envImportanceBytes, envPowerProxy, type EnvImportance } from '../scene/env/env-importance.ts';
import { spreadNormalization } from './emission-kernel.ts';
import { collectEmissiveTriangles, NO_ENTRY, type EmissiveTriangles } from './emissive-tris.ts';
import { LUT_LAYOUT, lutFloats } from './luts/lut-layout.ts';

/** Word offset of the BSDF LUT block inside `records` (material/lut.wgsl lutDefines({ base, recordsKind: 'u32' })). */
export const LUT_RECORDS_BASE = 0;

export const LIGHT_REC_WORDS = 28; // 112 B
/** Word offsets inside a light record (lights.wgsl light_load). Word 26 = M5 change bits (LCB_*, cur slot, relative
 *  to the slot's own predecessor; restir-temporal-api.md TD5); word 27 = 0. */
export const LIGHT_REC = {
  pos: 0, type: 3, axisU: 4, halfU: 7, axisV: 8, halfV: 11, normal: 12, area: 15, emit: 16, flags: 19,
  cosHalf: 20, spotSmooth: 21, spreadNorm: 22, tanHalfSpread: 23, stableId: 24, invArea: 25, changeBits: 26,
} as const;
/** Change bits of word 26 (restir-temporal-api.md §2.1, TD5). */
export const LCB = { moved: 1, radio: 2, added: 4 } as const;
/** Record words that enter the light point / direction Φ (MOVED), per type code (TD5, math §24 [M5]). */
export function movedWords(typeCode: number): number[] {
  const pos = [0, 1, 2];
  switch (typeCode) {
    case LT.point: case LT.spot: return pos;
    case LT.sun: return [12, 13, 14];
    default: return [...pos, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];     // rect / disk: pos, axisU, halfU, axisV, halfV, normal
  }
}
/** LCB bits of a light record vs its predecessor record (both LIGHT_REC_WORDS words, predecessor undefined = added). */
export function lightChangeBits(cur: ArrayLike<number>, prev: ArrayLike<number> | undefined): number {
  if (!prev) return LCB.added;
  const type = cur[LIGHT_REC.type];
  if (prev[LIGHT_REC.type] !== type) return LCB.moved | LCB.radio;
  const moved = new Set(movedWords(type));
  let bits = 0;
  for (let w = 0; w < LIGHT_REC.changeBits; w++) {
    if (cur[w] === prev[w]) continue;
    bits |= moved.has(w) ? LCB.moved : LCB.radio;
  }
  return bits;
}
/** math.md#path-tree endpoint types. */
export const LT = { tri: 0, point: 1, spot: 2, rect: 3, disk: 4, sun: 5, env: 7 } as const;
export const LF_VISIBLE_CAMERA = 1;
export const LF_DELTA = 2;
/** LightsParams.flags. */
export const LP_MODE_A = 1;   // analytic area lights are NEE-only (ω1 ≡ 1), never hit by BSDF rays
export const LP_ENV_NEE = 2;  // the env is an alias entry (env NEE on); escapes use MIS with p1Env
export const LP_CROSS_ALL = 4;    // Mode B: every BSDF ray crosses the analytic area lights (pass-through, MIS)
export const LP_CROSS_DELTA = 8;  // Mode A′: only rays leaving a delta lobe cross them (weight 1)
export const LIGHT_SLOT_WORDS = 12;
export const LIGHTS_PARAMS_SIZE = 128; // 2 × 48 B slots + 16 B + 16 B env tables
/** P(env) clamp when other emitters exist (math.md#light-selection). */
export const P_ENV_MIN = 0.1;
export const P_ENV_MAX = 0.9;
export const NO_ENV_ENTRY = 0xffffffff;

const lum = (c: readonly number[]): number => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

/** plan §1.4 Modes: A (analytic lights NEE-only), B (pass-through + MIS), A′ (pass-through after delta lobes only). */
export type LightMode = 'A' | 'B' | 'A′';

/** LightsParams.flags of a light mode (path/crossings.wgsl). */
export function lightModeFlags(m: LightMode): number {
  switch (m) {
    case 'A': return LP_MODE_A;
    case 'B': return LP_CROSS_ALL;
    case 'A′': return LP_MODE_A | LP_CROSS_DELTA;
    default: throw new Error(`unknown light mode ${m as string}`);
  }
}

/** Accepts the ASCII spelling "A'" for A′ (CLI / hand-written JSON). */
export function parseLightMode(s: string | undefined, fallback: LightMode = 'A'): LightMode {
  if (s === undefined) return fallback;
  if (s === 'A' || s === 'B' || s === 'A′') return s;
  if (s === "A'" || s === 'Aprime') return 'A′';
  throw new Error(`unknown light mode ${s}`);
}

/** Radiometric emission of one analytic light (math.md#units-lights). */
export function lightEmission(l: LightData): { emit: [number, number, number]; area: number } {
  const P = l.power * 2 ** l.exposure;
  const c = l.color;
  switch (l.type) {
    case 'point': case 'spot': {
      const s = P / (4 * Math.PI); // I = Φ/(4π) W/sr (spot NOT cone-normalised)
      return { emit: [c[0] * s, c[1] * s, c[2] * s], area: 0 };
    }
    case 'rect': case 'disk': {
      const sx = l.sizeX ?? 1, sy = l.sizeY ?? sx;
      const A = l.type === 'disk' ? (Math.PI / 4) * sx * sy : sx * sy;
      const s = P / (Math.PI * A); // L_e = Φ/(πA), one-sided
      return { emit: [c[0] * s, c[1] * s, c[2] * s], area: A };
    }
    case 'sun':
      return { emit: [c[0] * P, c[1] * P, c[2] * P], area: 0 }; // E = Φ (W/m²)
  }
}

/** Spot profile parameters: cos(spot_size/2) and 1/((1−cosH)·blend); blend = 0 → −1 (hard step guard). */
export function spotParams(l: LightData): { cosHalf: number; smooth: number } {
  const cosHalf = Math.cos((l.spotSize ?? Math.PI / 4) / 2);
  const blend = l.spotBlend ?? 0.15;
  const den = (1 - cosHalf) * blend;
  return { cosHalf, smooth: den > 0 ? 1 / den : -1 };
}

/** Spot profile S(t) (math.md#units-lights), CPU mirror of spot.wgsl. */
export function spotProfile(cosT: number, cosHalf: number, smooth: number): number {
  if (smooth < 0) return cosT > cosHalf ? 1 : 0;
  const f = (cosT - cosHalf) * smooth;
  return f <= 0 ? 0 : f >= 1 ? 1 : f * f * (3 - 2 * f);
}

/**
 * Power proxy Φ̃ (math.md#light-selection "Power proxies"). Variance-only; rect/disk use lum(L_e)·π·A (the
 * cosine-weighted mean of the spread factor is 1 because Cycles' spread normalisation preserves power).
 */
export function lightPowerProxy(l: LightData, sceneRadius: number): number {
  const { emit, area } = lightEmission(l);
  const Y = Math.max(0, lum(emit));
  switch (l.type) {
    case 'point': return 4 * Math.PI * Y;
    case 'spot': {
      const cOuter = Math.cos((l.spotSize ?? Math.PI / 4) / 2);
      const cFull = cOuter + (1 - cOuter) * (l.spotBlend ?? 0.15);
      return Y * 2 * Math.PI * ((1 - cFull) + (cFull - cOuter) / 2);
    }
    case 'rect': case 'disk': return Y * Math.PI * area;
    case 'sun': return Y * Math.PI * sceneRadius * sceneRadius;
  }
}

const TYPE_CODE: Record<LightData['type'], number> = { point: LT.point, spot: LT.spot, rect: LT.rect, disk: LT.disk, sun: LT.sun };

/** Pack one analytic light record (recentred by `origin`). math.md#units-lights frame: a_u = X, a_v = Y, a_L = −Z. */
export function packLightRecord(l: LightData, origin: readonly number[], out: DataView, byteOffset: number): void {
  const m = l.matrix;
  const unit = (c: number): number[] => { const v = [m[4 * c], m[4 * c + 1], m[4 * c + 2]]; const n = Math.hypot(v[0], v[1], v[2]) || 1; return v.map((x) => x / n); };
  const X = unit(0), Y = unit(1), Z = unit(2);
  const f = (w: number, v: number) => out.setFloat32(byteOffset + 4 * w, v, true);
  const u = (w: number, v: number) => out.setUint32(byteOffset + 4 * w, v >>> 0, true);
  const R = LIGHT_REC;
  f(R.pos, m[12] - origin[0]); f(R.pos + 1, m[13] - origin[1]); f(R.pos + 2, m[14] - origin[2]);
  u(R.type, TYPE_CODE[l.type]);
  const sx = l.sizeX ?? 1, sy = l.type === 'disk' ? (l.sizeY ?? sx) : (l.sizeY ?? sx);
  const area = l.type === 'rect' || l.type === 'disk';
  X.forEach((x, i) => f(R.axisU + i, x)); f(R.halfU, area ? sx / 2 : 0);
  Y.forEach((x, i) => f(R.axisV + i, x)); f(R.halfV, area ? sy / 2 : 0);
  f(R.normal, -Z[0]); f(R.normal + 1, -Z[1]); f(R.normal + 2, -Z[2]);
  const em = lightEmission(l);
  f(R.area, em.area);
  em.emit.forEach((x, i) => f(R.emit + i, x));
  const delta = l.type === 'point' || l.type === 'spot' || l.type === 'sun';
  u(R.flags, (l.visibleToCamera && area ? LF_VISIBLE_CAMERA : 0) | (delta ? LF_DELTA : 0));
  const sp = l.type === 'spot' ? spotParams(l) : { cosHalf: -2, smooth: -1 };
  f(R.cosHalf, sp.cosHalf); f(R.spotSmooth, sp.smooth);
  const spr = area ? spreadNormalization(l.spread ?? Math.PI) : { tanHalf: 0, norm: -1 };
  f(R.spreadNorm, spr.norm); f(R.tanHalfSpread, spr.tanHalf);
  u(R.stableId, l.id);
  f(R.invArea, em.area > 0 ? 1 / em.area : 0);
}

export interface LightSlotCpu {
  lightOff: number; lightCount: number; aliasOff: number; aliasLog2: number; pmfOff: number;
  nAnalytic: number; nEntries: number; envEntry: number; curToPrevOff: number; prevToCurOff: number;
}

interface SlotState {
  cpu: LightSlotCpu;
  ids: number[];
  weights: Float64Array;
  table: AliasTable | undefined;
}

export interface LightsUpdate {
  /** The set or any light record changed. */
  lightsChanged: boolean;
  /** The alias table / realized pmf changed (power proxy or set change). */
  pmfChanged: boolean;
  /** The records buffer was reallocated: rebuild bind groups. */
  reallocated: boolean;
}

/** M5 (restir-temporal-api.md §2.4): what the ONE light commit of a frame changed. */
export interface LightsCommit {
  /** Nothing changed: the slot did not flip, prev ≡ cur (TF_LIGHTS_SAME). */
  same: boolean;
  /** The alias table / realized pmf changed (TF_PMF_CHANGED). */
  pmfChanged: boolean;
  /** Some analytic light has LCB_MOVED / LCB_RADIO in its change word (word 26). */
  anyMoved: boolean;
  anyRadio: boolean;
  /** Stable ids of added / removed analytic lights. */
  added: number[];
  removed: number[];
  /** The records buffer was re-laid out (capacity growth, env table change): a config-hash input (reset). */
  reallocated: boolean;
}

/** The environment as a light (M3c): importance tables + the radiometric parameters that enter Φ_env. */
export interface EnvLightInput {
  table: EnvImportance;
  strength: number;
  tint: readonly [number, number, number] | readonly number[];
  /** Env NEE on (≡ Cycles world sampling_method AUTOMATIC); off = BSDF-only env (sampling_method NONE). */
  nee: boolean;
}

export interface LightsGpuOptions {
  lightMode?: LightMode;
  label?: string;
}

/** CPU packing of the whole light state (separated from the GPU object for unit tests). */
export class LightsState {
  readonly tris: EmissiveTriangles;
  readonly sceneRadius: number;
  lightMode: LightMode;
  /** Capacities (analytic lights per slot, alias entries per slot). */
  capLights = 0;
  capLog2 = 1;
  triOff = 0;
  primMapOff = 0;
  slotBase: [number, number] = [0, 0];
  slotWords = 0;
  totalWords = 0;
  cur = 0;
  slots: [SlotState | undefined, SlotState | undefined] = [undefined, undefined];
  records = new Uint32Array(0);
  /** Env tables in `records` (static section; 0 when no env table is set). */
  envRowOff = 0;
  envColOff = 0;
  envPdfOff = 0;
  envLog2W = 0;
  private envTable: EnvImportance | undefined;
  private dirtyAll = true;
  /** Word ranges written by the last update (for partial uploads). */
  dirty: [number, number][] = [];

  constructor(readonly scene: SceneData, readonly origin: readonly number[], recentredPositions: Float32Array, lightMode: LightMode = 'A') {
    this.tris = collectEmissiveTriangles(scene, recentredPositions);
    const b = scene.bounds;
    this.sceneRadius = 0.5 * Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
    this.lightMode = lightMode;
  }

  get curSlot(): LightSlotCpu { return this.slots[this.cur]!.cpu; }
  get prevSlotIndex(): number { return this.cur ^ 1; }
  get table(): AliasTable | undefined { return this.slots[this.cur]?.table; }
  /** Alias entry index of emissive-triangle entry i. */
  triEntry(i: number): number { return this.curSlot.nAnalytic + i; }

  private layout(nLights: number, nEntries: number, envTable: EnvImportance | undefined): boolean {
    let capLights = Math.max(8, this.capLights);
    while (capLights < nLights) capLights *= 2;
    let capLog2 = Math.max(this.capLog2, 1);
    while (2 ** capLog2 < Math.max(2, nEntries + 1)) capLog2++; // +1: the env entry
    if (capLights === this.capLights && capLog2 === this.capLog2 && envTable === this.envTable && this.records.length) return false;
    this.capLights = capLights;
    this.capLog2 = capLog2;
    const nTri = this.tris.primIds.length;
    const nPrim = this.tris.primToEntry.length;
    this.triOff = LUT_RECORDS_BASE + LUT_LAYOUT.floats;
    this.primMapOff = this.triOff + 2 * nTri;
    const capN = 2 ** capLog2;
    this.slotWords = capLights * LIGHT_REC_WORDS + 2 * capN + capN + 2 * capLights;
    // Env tables (static): rowAlias | colAlias | pdfUV.
    this.envTable = envTable;
    const envBase = this.primMapOff + Math.max(1, nPrim);
    const cells = envTable ? envTable.Wm * envTable.Hm : 0;
    this.envRowOff = envBase;
    this.envColOff = envBase + (envTable?.Hm ?? 0);
    this.envPdfOff = this.envColOff + cells;
    this.envLog2W = envTable?.log2W ?? 0;
    const s0 = this.envPdfOff + cells;
    this.slotBase = [s0, s0 + this.slotWords];
    this.totalWords = s0 + 2 * this.slotWords;
    this.records = new Uint32Array(this.totalWords);
    const f = new Float32Array(this.records.buffer);
    f.set(lutFloats(), LUT_RECORDS_BASE);
    for (let i = 0; i < nTri; i++) { this.records[this.triOff + 2 * i] = this.tris.primIds[i]; f[this.triOff + 2 * i + 1] = this.tris.areas[i]; }
    this.records.set(this.tris.primToEntry, this.primMapOff);
    if (envTable) {
      this.records.set(envTable.rowAlias, this.envRowOff);
      this.records.set(envTable.colAlias, this.envColOff);
      f.set(envTable.pdfUV, this.envPdfOff);
    }
    this.slots = [undefined, undefined];
    this.dirtyAll = true;
    return true;
  }

  /** Pack `lights` (and the env, when given) into the next slot and flip. Returns what changed. */
  update(lightsIn: readonly LightData[], env?: EnvLightInput): LightsUpdate {
    return this.pack(lightsIn, env, true).update;
  }

  /**
   * M5 (restir-temporal-api.md §2.4, TD4): the ONE light commit of a frame. Packs the state into the non-current slot;
   * flips only if something changed (records other than word 26, the id set, the weights or the layout). Otherwise the
   * non-current slot now holds a bitwise copy of cur and `same` is set (TF_LIGHTS_SAME: prev accessors read cur).
   */
  commit(lightsIn: readonly LightData[], env?: EnvLightInput): LightsCommit {
    return this.pack(lightsIn, env, false).commit;
  }

  private pack(lightsIn: readonly LightData[], env: EnvLightInput | undefined, alwaysFlip: boolean): { update: LightsUpdate; commit: LightsCommit } {
    const lights = [...lightsIn].sort((a, b) => a.id - b.id);
    const nA = lights.length, nT = this.tris.primIds.length;
    const phiEnv = env?.nee ? envPowerProxy(env.table, env.strength, env.tint, this.sceneRadius) : 0;
    const hasEnvEntry = phiEnv > 0;
    const reallocated = this.layout(nA, nA + nT + (hasEnvEntry ? 1 : 0), env?.table);
    const prev = this.slots[this.cur];
    const first = !prev;
    const next = first ? this.cur : this.cur ^ 1;
    const base = this.slotBase[next];
    const capN = 2 ** this.capLog2;
    const lightOff = base;
    const aliasOff = lightOff + this.capLights * LIGHT_REC_WORDS;
    const pmfOff = aliasOff + 2 * capN;
    const curToPrevOff = pmfOff + capN;
    const prevToCurOff = curToPrevOff + this.capLights;
    const dv = new DataView(this.records.buffer);
    const f32 = new Float32Array(this.records.buffer);

    // Light records (stable-id order).
    for (let i = 0; i < nA; i++) packLightRecord(lights[i], this.origin, dv, 4 * (lightOff + i * LIGHT_REC_WORDS));
    const ids = lights.map((l) => l.id);

    // Selection weights: analytic (stable-id order), emissive triangles, then the env (P(env) clamp, f64).
    const nE = nA + nT + (hasEnvEntry ? 1 : 0);
    const weights = new Float64Array(nE);
    for (let i = 0; i < nA; i++) weights[i] = lightPowerProxy(lights[i], this.sceneRadius);
    weights.set(this.tris.power, nA);
    if (hasEnvEntry) {
      let others = 0;
      for (let i = 0; i < nA + nT; i++) others += weights[i];
      if (others > 0) {
        const pEnv = Math.min(P_ENV_MAX, Math.max(P_ENV_MIN, phiEnv / (others + phiEnv)));
        for (let i = 0; i < nA + nT; i++) weights[i] = ((1 - pEnv) * weights[i]) / others;
        weights[nA + nT] = pEnv;
      } else {
        weights[nA + nT] = 1;
      }
    }
    const sameSet = !!prev && prev.ids.length === nA && prev.ids.every((x, i) => x === ids[i]);
    const sameWeights = sameSet && prev.weights.length === weights.length && prev.weights.every((x, i) => Object.is(x, weights[i]));
    const table = sameWeights ? prev.table : buildAliasTable(weights, 1);
    const log2 = table?.log2n ?? 1;
    if (table) {
      this.records.set(packAliasEntries(table), aliasOff);
      for (let e = 0; e < table.count; e++) f32[pmfOff + e] = table.pmf[e];
    }
    // id maps (prev slot = last frame's state; the first frame maps onto itself).
    const prevIds = prev?.ids ?? ids;
    const prevIndex = new Map(prevIds.map((id, i) => [id, i]));
    const curIndex = new Map(ids.map((id, i) => [id, i]));
    for (let i = 0; i < nA; i++) this.records[curToPrevOff + i] = prevIndex.get(ids[i]) ?? NO_ENTRY;
    for (let i = 0; i < prevIds.length; i++) this.records[prevToCurOff + i] = curIndex.get(prevIds[i]) ?? NO_ENTRY;

    const cpu: LightSlotCpu = {
      lightOff, lightCount: nA, aliasOff, aliasLog2: log2, pmfOff, nAnalytic: nA, nEntries: table ? nE : 0,
      envEntry: table && hasEnvEntry ? nA + nT : NO_ENV_ENTRY, curToPrevOff, prevToCurOff,
    };
    // Change bits (word 26) against the predecessor record of the same stable id (M5 TD5); word 27 = 0. Words 26/27 are
    // excluded from lightsChanged (they are derived).
    const W = LIGHT_REC_WORDS;
    let lightsChanged = first || !sameSet;
    let anyMoved = false, anyRadio = false;
    const added: number[] = [];
    for (let i = 0; i < nA; i++) {
      const o = lightOff + i * W;
      const j = this.records[curToPrevOff + i];
      const prevRec = prev && j !== NO_ENTRY ? this.records.subarray(prev.cpu.lightOff + j * W, prev.cpu.lightOff + (j + 1) * W) : undefined;
      const bits = first ? 0 : lightChangeBits(this.records.subarray(o, o + W), prevRec);
      this.records[o + LIGHT_REC.changeBits] = bits;
      this.records[o + 27] = 0;
      if (bits & LCB.added) added.push(ids[i]);
      if (bits & LCB.moved) anyMoved = true;
      if (bits & LCB.radio) anyRadio = true;
      if (bits) lightsChanged = true;
    }
    const removed = first ? [] : prevIds.filter((id) => !curIndex.has(id));
    if (!lightsChanged && prev) {
      for (let i = 0; i < nA && !lightsChanged; i++) {
        for (let w = 0; w < LIGHT_REC.changeBits; w++) {
          if (this.records[prev.cpu.lightOff + i * W + w] !== this.records[lightOff + i * W + w]) { lightsChanged = true; break; }
        }
      }
    }
    const pmfChanged = first || !sameWeights;
    const same = !alwaysFlip && !first && !lightsChanged && !pmfChanged && !reallocated;
    this.slots[next] = { cpu, ids, weights, table };
    if (!same) this.cur = next;                          // M5 commit: an unchanged state does not flip (TD4)
    this.dirty = this.dirtyAll ? [[0, this.totalWords]] : [[base, base + this.slotWords]];
    this.dirtyAll = false;
    return {
      update: { lightsChanged, pmfChanged, reallocated },
      commit: { same, pmfChanged, anyMoved, anyRadio, added, removed, reallocated },
    };
  }

  /** LightsParams uniform (lights.wgsl): cur slot, prev slot, tri/primMap offsets, flags, env table offsets. */
  paramsBytes(flags = lightModeFlags(this.lightMode) | (this.curSlot.envEntry !== NO_ENV_ENTRY ? LP_ENV_NEE : 0)): ArrayBuffer {
    const buf = new ArrayBuffer(LIGHTS_PARAMS_SIZE);
    const u = new Uint32Array(buf);
    const put = (o: number, s: LightSlotCpu) => {
      u.set([s.lightOff, s.lightCount, s.aliasOff, s.aliasLog2, s.pmfOff, s.nAnalytic, s.nEntries, s.envEntry, s.curToPrevOff, s.prevToCurOff, 0, 0], o);
    };
    const cur = this.slots[this.cur]!.cpu;
    const prev = this.slots[this.cur ^ 1]?.cpu ?? cur;
    put(0, cur);
    put(LIGHT_SLOT_WORDS, prev);
    u.set([this.triOff, this.tris.primIds.length, this.primMapOff, flags], 2 * LIGHT_SLOT_WORDS);
    u.set([this.envRowOff, this.envColOff, this.envPdfOff, this.envLog2W], 2 * LIGHT_SLOT_WORDS + 4);
    return buf;
  }

  /** The slot the GPU sees as `prev` (LightsParams.prev: the non-current slot, or cur before the first flip). */
  get prevSlot(): LightSlotCpu { return (this.slots[this.cur ^ 1] ?? this.slots[this.cur]!).cpu; }

  /**
   * CPU mirror of tframe.wgsl lt_translate (restir-temporal-api.md §3.2, TD7): alias entry `entry` of frame `from` in the
   * numbering of frame `to` ('prev' / 'cur'), NO_ENTRY when absent. `same` = TF_LIGHTS_SAME (identity).
   */
  translate(entry: number, from: 'prev' | 'cur', to: 'prev' | 'cur', same = false): number {
    if (entry === NO_ENTRY) return NO_ENTRY;
    if (from === to || same) return entry;
    const sf = from === 'prev' ? this.prevSlot : this.curSlot;
    const st = to === 'prev' ? this.prevSlot : this.curSlot;
    const cur = this.curSlot;
    if (entry === sf.envEntry) return st.envEntry;
    if (entry < sf.nAnalytic) return this.records[(from === 'prev' ? cur.prevToCurOff : cur.curToPrevOff) + entry];
    const tri = entry - sf.nAnalytic;
    if (tri >= this.tris.primIds.length) return NO_ENTRY;
    return st.nAnalytic + tri;
  }

  /** Realized pmf of alias entry e of a slot (f32 as stored). */
  pmfOf(slot: LightSlotCpu, e: number): number { return new Float32Array(this.records.buffer)[slot.pmfOff + e]; }

  /** Realized pmf of alias entry e in the current slot (f32 as stored). */
  pmf(e: number): number { return new Float32Array(this.records.buffer)[this.curSlot.pmfOff + e]; }

  /** Realized P(env) of the current slot (0 without an env entry). */
  envPmf(): number { const e = this.curSlot.envEntry; return e === NO_ENV_ENTRY ? 0 : this.pmf(e); }

  /** The env table currently stored in `records`. */
  get env(): EnvImportance | undefined { return this.envTable; }
}

/** GPU side: the `records` storage buffer + the LightsParams uniform. */
export class LightsGpu {
  records: GPUBuffer;
  readonly params: GPUBuffer;
  /** Bumps when `records` is reallocated (bind groups must be rebuilt). */
  version = 0;
  readonly state: LightsState;
  /** The env as a light (M3c), applied on every update(). */
  private envInput: EnvLightInput | undefined;
  private lastLights: readonly LightData[];

  constructor(readonly device: GPUDevice, scene: SceneData, origin: readonly number[], recentredPositions: Float32Array, private readonly opts: LightsGpuOptions = {}) {
    this.state = new LightsState(scene, origin, recentredPositions, opts.lightMode ?? 'A');
    this.lastLights = scene.lights;
    this.state.update(scene.lights);
    this.params = device.createBuffer({ label: `${opts.label ?? 'lights'}.params`, size: LIGHTS_PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.records = this.createRecords();
    this.upload(true);
  }

  private createRecords(): GPUBuffer {
    return this.device.createBuffer({
      label: `${this.opts.label ?? 'lights'}.records`, size: Math.max(16, this.state.records.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
  }

  private upload(all: boolean): void {
    const r = this.state.records;
    const ranges = all ? [[0, r.length] as [number, number]] : this.state.dirty;
    for (const [a, b] of ranges) if (b > a) this.device.queue.writeBuffer(this.records, 4 * a, r.buffer, r.byteOffset + 4 * a, 4 * (b - a));
    this.device.queue.writeBuffer(this.params, 0, this.state.paramsBytes());
  }

  get lightMode(): LightMode { return this.state.lightMode; }
  setLightMode(m: LightMode): void { this.state.lightMode = m; this.device.queue.writeBuffer(this.params, 0, this.state.paramsBytes()); }

  /**
   * M5 (restir-temporal-api.md §2.4, TD4): with `deferred`, update() / setEnvironment() only STAGE the edit (they return
   * a provisional LightsUpdate; bind groups follow `version`) and commit() applies all staged edits of the frame once.
   * The PT kernel's LightsGpu keeps the M3 behaviour (deferred = false: every call commits and flips).
   */
  deferred = false;
  private pending = false;

  /** Staged edits waiting for commit(). */
  get hasPending(): boolean { return this.pending; }

  /** Apply the staged state once (no flip when nothing changed). */
  commit(): LightsCommit {
    this.pending = false;
    const res = this.state.commit(this.lastLights, this.envInput);
    this.afterPack(res.reallocated);
    return res;
  }

  private afterPack(reallocated: boolean): void {
    if (reallocated) {
      this.records.destroy();
      this.records = this.createRecords();
      this.version++;
      this.upload(true);
    } else {
      this.upload(false);
    }
  }

  /** New light list for this frame (sorted by stable id internally). */
  update(lights: readonly LightData[]): LightsUpdate {
    this.lastLights = lights;
    if (this.deferred) { this.pending = true; return { lightsChanged: true, pmfChanged: true, reallocated: false }; }
    const res = this.state.update(lights, this.envInput);
    if (res.reallocated) {
      this.records.destroy();
      this.records = this.createRecords();
      this.version++;
      this.upload(true);
    } else {
      this.upload(false);
    }
    return res;
  }

  /** Set (or clear) the env as a light and re-pack the current light list (table change → reallocation). */
  setEnvironment(env: EnvLightInput | undefined): LightsUpdate {
    this.envInput = env;
    return this.update(this.lastLights);
  }

  get environment(): EnvLightInput | undefined { return this.envInput; }

  /** HUD / meta summary. */
  summary(): { analytic: number; emissiveTriangles: number; aliasEntries: number; aliasLog2: number; recordsBytes: number; env: boolean; pEnv: number; envGrid?: [number, number]; envTableBytes?: number } {
    const s = this.state.curSlot;
    const t = this.state.env;
    return {
      analytic: s.nAnalytic, emissiveTriangles: this.state.tris.primIds.length, aliasEntries: s.nEntries, aliasLog2: s.aliasLog2, recordsBytes: this.records.size,
      env: s.envEntry !== NO_ENV_ENTRY, pEnv: this.state.envPmf(), ...(t ? { envGrid: [t.Wm, t.Hm] as [number, number], envTableBytes: envImportanceBytes(t) } : {}),
    };
  }

  destroy(): void { this.records.destroy(); this.params.destroy(); }
}
