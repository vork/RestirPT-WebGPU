// ReSTIR settings and the rung / mode presets (restir-api.md §4.6, §6.3; PLAN §3 modes, §7.1 rungs 3.1/3.1b/3.2).
// Criteria mode, plants, RR, S, NS and rounds are UNIFORMS (no recompiles, §4.5).
import { RS_WGSL_CONSTS as K } from './layout.ts';

export interface RestirSettings {
  /** Cycles max_bounces N (≤ N+1 scattering vertices, the PT's semantics). */
  maxBounces: number;
  /** Russian roulette at initial sampling (D11: the PT's q = min(sqrt(max_c β_c), 1) for B > rrMinBounces). */
  rr: boolean; rrMinBounces: number;
  /** S path trees per pixel streamed into one reservoir (offline 32). */
  trees: number;
  /** Paired spatial rounds and slots per round (NS ≤ 6), pairing disk radius R in px. */
  rounds: number; slots: number; diskRadius: number;
  criteria: 'enhanced' | '2022';
  /** rc threshold τ (thr = τ·R²_pri) and α_min (perceptual roughness). */
  tau: number; alphaMin: number;
  /** Planted controls (validation only): omitted spatial J, marginal pdfs in J, W × wScale; M5: U8 plants (RSF bits). */
  plant?: {
    noJ?: boolean; marginalJ?: boolean; wScale?: number;
    u8W1Delta?: boolean; u8NoPk?: boolean; u8T2?: boolean; u8OneSided?: boolean; u8FailedK?: boolean;
  };
  // ---- M5 temporal (restir-temporal-api.md §3.8, TD1, TD14, TD21, TD23, TD24) ----
  /** Temporal reuse on (TD15: temporal before spatial). Off ⇒ every M4 output bitwise unchanged. */
  temporal: boolean;
  /** Confidence cap of the temporal candidate, c_p = min(cCap, c_prev) (TD14, math §26). */
  cCap: number;
  /** Temporal estimator: contribution MIS (default) or Talbot-exact (TD1). */
  temporalMis: 'contribution' | 'talbot';
  /** Validation variants of π_p(Y_p): recomputed by E_{t−1} (TM_PP_RECOMPUTE) or both + compare (TM_ROBUST). */
  temporalCheck?: 'none' | 'recompute' | 'robust';
  /** Suffix refresh: exact, or the E2 class-zeroing fallback (TD23). */
  refresh: 'exact' | 'e2';
  /** Reciprocal disocclusion boost slots NB (TD21); numSlots = slots + NB with temporal on. */
  boostSlots: number;
  /** Temporal plants (validation only; TP_* bits, §6.5). */
  tPlant?: {
    n1Mixed?: boolean; noJP?: boolean; noJPEnv?: boolean; n3Stale?: boolean; n4Ris?: boolean; n5PixelCentre?: boolean;
    n6CurCam?: boolean; n7PerLight?: boolean; envNoRotVis?: boolean; envGammaT?: boolean; cpPlus1?: boolean; u8StaleAux?: boolean;
    u8SpotPrevAxis?: boolean;
  };
}

export type RestirPresetName = 'initial' | 'initial-rr' | 'offline' | 'interactive' | 'criteria2022' | 'temporal' | 'full';

export const DEFAULT_RESTIR_SETTINGS: RestirSettings = {
  maxBounces: 3, rr: false, rrMinBounces: 3, trees: 1, rounds: 0, slots: 3, diskRadius: 30,
  criteria: 'enhanced', tau: 2e-4, alphaMin: 0.2,
  temporal: false, cCap: 20, temporalMis: 'contribution', temporalCheck: 'none', refresh: 'exact', boostSlots: 0,
};

const OFFLINE: Partial<RestirSettings> = { trees: 32, rounds: 3, slots: 6, diskRadius: 10, rr: false };

export const RESTIR_PRESETS: Record<RestirPresetName, Partial<RestirSettings>> = {
  initial: { trees: 1, rounds: 0, rr: false },
  'initial-rr': { trees: 1, rounds: 0, rr: true, rrMinBounces: 1 },
  offline: OFFLINE,
  // TD24: interactive gains temporal on and the reciprocal disocclusion boost (3 slots, TD21; Q7: on interactively,
  // off in validation except the gating ixs_d unit). The interactive temporal units run on frames the renderer
  // prepares with advanceInteractive() (renderer.ts, TD20).
  // Coordinator decision TD-I1 (2026-10-01, user-approved): interactive c_cap 5, not 20. Contribution MIS gives an
  // undefined inverse shift (π_p = 0) the whole Σw̃ including the history weight, i.e. a ~(1+c_p)× spike that then
  // persists ~c_p frames (diag 2026-10-01: births 3.7× over-represented at edges). c_cap 5: edge RMSE 0.041 → 0.021,
  // spike events −62%, still unbiased (constant cap). Validation presets ('temporal', 'full') keep the spec's 20.
  interactive: { trees: 1, rounds: 1, slots: 3, diskRadius: 30, rr: true, rrMinBounces: 3, temporal: true, boostSlots: 3, cCap: 5 },
  criteria2022: { ...OFFLINE, criteria: '2022' },
  // M5 rungs (TD24; math §25: RR off in 3.2–3.6)
  temporal: { trees: 1, rounds: 0, rr: false, temporal: true, boostSlots: 0 },
  full: { trees: 1, rounds: 1, slots: 3, diskRadius: 30, rr: false, temporal: true, boostSlots: 0 },
};

/** Default settings ⊕ preset ⊕ overrides (maxBounces etc. come from the scene package). */
export function restirSettings(preset?: RestirPresetName, overrides: Partial<RestirSettings> = {}): RestirSettings {
  const s = { ...DEFAULT_RESTIR_SETTINGS, ...(preset ? RESTIR_PRESETS[preset] : {}), ...overrides };
  validateSettings(s);
  return s;
}

export function validateSettings(s: RestirSettings): void {
  const int = (x: number, lo: number, hi: number, n: string) => {
    if (!(Number.isInteger(x) && x >= lo && x <= hi)) throw new Error(`RestirSettings.${n} = ${x} not in [${lo}, ${hi}]`);
  };
  int(s.maxBounces, 0, K.RS_MAX_D - 2, 'maxBounces');
  int(s.trees, 1, K.RS_MAX_TREES, 'trees');
  int(s.rounds, 0, K.RS_MAX_ROUNDS, 'rounds');
  int(s.slots, 1, K.RS_MAX_SLOTS, 'slots');
  int(s.rrMinBounces, 0, 64, 'rrMinBounces');
  if (!(s.tau > 0) || !(s.alphaMin >= 0)) throw new Error('RestirSettings: tau > 0 and alphaMin ≥ 0 required');
  int(s.boostSlots ?? 0, 0, K.RS_MAX_SLOTS, 'boostSlots');
  if (numSlotsOf(s) > K.RS_MAX_SLOTS) throw new Error(`RestirSettings: slots + boostSlots = ${numSlotsOf(s)} > ${K.RS_MAX_SLOTS}`);
  if (!(s.cCap >= 0) || !Number.isFinite(s.cCap)) throw new Error(`RestirSettings.cCap = ${s.cCap}: finite ≥ 0 required`);
}

/** RestirParams.numSlots: slots + boost slots (boost only with temporal on, TD21). */
export function numSlotsOf(s: Pick<RestirSettings, 'slots' | 'boostSlots' | 'temporal'>): number {
  return s.slots + (s.temporal ? (s.boostSlots ?? 0) : 0);
}

/** RestirParams.tMode (TM_*). */
export function tModeOf(s: RestirSettings): number {
  let m = 0;
  if (s.temporalMis === 'talbot') m |= K.TM_TALBOT;
  if (s.temporalCheck === 'recompute') m |= K.TM_PP_RECOMPUTE;
  if (s.temporalCheck === 'robust') m |= K.TM_ROBUST;
  if (s.refresh === 'e2') m |= K.TM_E2;
  return m;
}

/** RestirParams.tPlants (TP_*). */
export function tPlantsOf(s: RestirSettings): number {
  const p = s.tPlant ?? {};
  const bits: [boolean | undefined, number][] = [
    [p.n1Mixed, K.TP_N1_MIXED], [p.noJP, K.TP_NO_JP], [p.noJPEnv, K.TP_NO_JP_ENV], [p.n3Stale, K.TP_N3_STALE], [p.n4Ris, K.TP_N4_RIS],
    [p.n5PixelCentre, K.TP_N5_PIXEL_CENTRE], [p.n6CurCam, K.TP_N6_CUR_CAM], [p.n7PerLight, K.TP_N7_PER_LIGHT],
    [p.envNoRotVis, K.TP_ENV_NO_ROT_VIS], [p.envGammaT, K.TP_ENV_GAMMA_T], [p.cpPlus1, K.TP_CP_PLUS1],
    [p.u8StaleAux, K.TP_U8_STALE_AUX], [p.u8SpotPrevAxis, K.TP_U8_SPOT_PREV_AXIS],
  ];
  return bits.reduce((m, [on, b]) => (on ? m | b : m), 0);
}

/** RestirParams.flags (RSF_*) of the settings (ensemble / interactive bits are added by the kernel). */
export function restirFlags(s: RestirSettings): number {
  let f = 0;
  if (s.rr) f |= K.RSF_RR;
  if (s.criteria === '2022') f |= K.RSF_CRIT_2022;
  if (s.plant?.noJ) f |= K.RSF_PLANT_NO_J;
  if (s.plant?.marginalJ) f |= K.RSF_PLANT_MARGINAL_J;
  if (s.temporal) f |= K.RSF_TEMPORAL;
  if (s.plant?.u8W1Delta) f |= K.RSF_PLANT_U8_W1DELTA;
  if (s.plant?.u8NoPk) f |= K.RSF_PLANT_U8_NO_PK;
  if (s.plant?.u8T2) f |= K.RSF_PLANT_U8_T2;
  if (s.plant?.u8OneSided) f |= K.RSF_PLANT_U8_ONESIDED;
  if (s.plant?.u8FailedK) f |= K.RSF_PLANT_U8_FAILED_K;
  return f;
}

/** Logical sizes W_s of the 8 pairing-map layers (§2.8; M4 defaults for 6 slots, layers 6–7 unused). */
export const PAIR_TEX_SIZES = [254, 246, 238, 230, 222, 210, 0, 0];
