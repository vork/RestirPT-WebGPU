// ReSTIR settings and the rung / mode presets (restir-api.md §4.6, §6.3; PLAN §3 modes, §7.1 rungs 3.1/3.1b/3.2).
// Criteria mode, plants, RR, S, NS and rounds are UNIFORMS (no recompiles, §4.5).
import { RS_WGSL_CONSTS as K, RS_M6_CONSTS as K6 } from './layout.ts';

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
    /** M6 (restir-m6-api.md §5.3): U8-8 RIS UCW in mixed measures, U8-10 tile-conditional pmf in ω1, U8-7 crossed lights
     *  stop BSDF rays in the path tree (Mode B). U8-4 is u8T2 (J = t_x²/t_y², spatial shifts). */
    u8RisMixed?: boolean; u8TilePmf?: boolean; u8CrossOcc?: boolean;
    /** M7 (m7-api.md §5.2): in every ReSTIR pass but the path tree, the MikkTSpace bitangent sign is ignored (B-NM-sign)
     *  / the Normal Map strength is applied glTF-style (B-NM-strength); the shift's geometry term at x_k uses the shading
     *  normal (B-SM-J). Pipeline variants (defines NM_PLANT, RS_PLANT_SMOOTH_J). */
    m7NmSign?: boolean; m7NmStrength?: boolean; m7SmoothJ?: boolean;
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
  // ---- M6 (restir-m6-api.md §2.5; every feature off by default ⇒ the M5 pipelines, MD1) ----
  /** Pairing maps: the M4 uniform-disk involutions of radius diskRadius, or the M6 Gaussian maps of std pairSigma (MD3). */
  pairing: 'disk' | 'gauss';
  pairSigma: number;
  /** RIS-NEE light tiles at x₁ with risM candidates (MD4; M(1) = risM, a power of two). Pipeline variant RS_RIS_NEE. */
  risNee: boolean;
  risM: number;
  /** Dual motion vectors for disoccluded pixels (MD11; temporal only, unbiased). Variant RS_DUAL_MV. */
  dualMv: boolean;
  /** Duplication-map adaptive temporal cap (MD10; BIASED, temporal only). Variant RS_DUPMAP + pass rs_dupmap. */
  dupmap: boolean;
}

export type RestirPresetName = 'initial' | 'initial-rr' | 'offline' | 'interactive' | 'criteria2022' | 'temporal' | 'full' | 'offline-m6' | 'full-m6';

export const DEFAULT_RESTIR_SETTINGS: RestirSettings = {
  maxBounces: 3, rr: false, rrMinBounces: 3, trees: 1, rounds: 0, slots: 3, diskRadius: 30,
  criteria: 'enhanced', tau: 2e-4, alphaMin: 0.2,
  temporal: false, cCap: 20, temporalMis: 'contribution', temporalCheck: 'none', refresh: 'exact', boostSlots: 0,
  pairing: 'disk', pairSigma: 16, risNee: false, risM: 32, dualMv: false, dupmap: false,
};

/** The unbiased Enhanced features of M6 (rungs 3.9–3.11 configuration, restir-m6-api.md MD13). */
const M6_UNBIASED: Partial<RestirSettings> = { pairing: 'gauss', pairSigma: 16, risNee: true, risM: 32 };

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
  // M6 (restir-m6-api.md MD13): + the σ = 16 Gaussian maps, RIS-NEE at x₁, dual MVs and the (biased) duplication map.
  interactive: {
    trees: 1, rounds: 1, slots: 3, diskRadius: 30, rr: true, rrMinBounces: 3, temporal: true, boostSlots: 3, cCap: 5,
    ...M6_UNBIASED, dualMv: true, dupmap: true,
  },
  criteria2022: { ...OFFLINE, criteria: '2022' },
  // M5 rungs (TD24; math §25: RR off in 3.2–3.6)
  temporal: { trees: 1, rounds: 0, rr: false, temporal: true, boostSlots: 0 },
  full: { trees: 1, rounds: 1, slots: 3, diskRadius: 30, rr: false, temporal: true, boostSlots: 0 },
  // M6 rungs (restir-m6-api.md MD13, MD14): every unbiased Enhanced feature on (no RR: toggled in rung 3.7).
  'offline-m6': { ...OFFLINE, ...M6_UNBIASED },
  'full-m6': { trees: 1, rounds: 1, slots: 3, diskRadius: 30, rr: false, temporal: true, boostSlots: 0, ...M6_UNBIASED },
};

/** M6 features off (the M5 configuration of a preset): U-M4-BITS / U-M5-BITS pin the interactive cases with it. */
export const M6_OFF: Partial<RestirSettings> = { pairing: 'disk', risNee: false, dualMv: false, dupmap: false };

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
  if (s.pairing !== 'disk' && s.pairing !== 'gauss') throw new Error(`RestirSettings.pairing = ${String(s.pairing)}`);
  if (!(s.pairSigma >= 0.8 && s.pairSigma <= 40)) throw new Error(`RestirSettings.pairSigma = ${s.pairSigma} not in [0.8, 40]`);
  int(s.risM, 1, K6.RS_RIS_M_MAX, 'risM');
  if ((s.risM & (s.risM - 1)) !== 0) throw new Error(`RestirSettings.risM = ${s.risM}: a power of two (MD5: p2/M is exact)`);
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
  // M6 (restir-m6-api.md §2.1): the feature bits mirror the defines; plants act only inside their variants
  if (s.risNee) f |= K6.RSF_RIS_NEE;
  if (s.dupmap && s.temporal) f |= K6.RSF_DUPMAP;
  if (s.dualMv && s.temporal) f |= K6.RSF_DUAL_MV;
  if (s.plant?.u8RisMixed) f |= K6.RSF_PLANT_U8_RIS_MIXED;
  if (s.plant?.u8TilePmf) f |= K6.RSF_PLANT_U8_TILE_PMF;
  if (s.plant?.u8CrossOcc) f |= K6.RSF_PLANT_U8_CROSS_OCC;
  return f;
}

/** Composer defines of the M6 pipeline variants (MD1; 0 / false ⇒ the M5 text). */
export function m6Defines(s: RestirSettings, lightMode: 'A' | 'B' | 'A′' = 'A'): Record<string, number> {
  return {
    RS_RIS_NEE: s.risNee ? 1 : 0,
    RS_MODE_B: lightMode === 'A' ? 0 : 1,
    RS_DUAL_MV: s.dualMv && s.temporal ? 1 : 0,
    RS_DUPMAP: s.dupmap && s.temporal ? 1 : 0,
    // U8-4 plant (restir-m6-api.md §5.3): validation-only variant, the spatial shifts know their source record
    RS_PLANT_T2: s.plant?.u8T2 ? 1 : 0,
    // M7 plants (m7-api.md §5.2): validation-only variants (0 ⇒ the unplanted text)
    RS_PLANT_SMOOTH_J: s.plant?.m7SmoothJ ? 1 : 0,
  };
}

/** M7 Normal Map plant define of a pass (m7-api.md §5.2): every ReSTIR pass but the path tree (rs_initial[_dump]). */
export function m7NmPlantDefine(s: RestirSettings, pass: string): number {
  if (pass === 'rs_initial' || pass === 'rs_initial_dump') return 0;
  return s.plant?.m7NmSign ? 1 : s.plant?.m7NmStrength ? 2 : 0;
}

/** Logical layer sizes W_s of the pairing maps of the settings (RestirParams.pairTexSize). */
export function pairTexSizes(s: Pick<RestirSettings, 'pairing'>): number[] {
  return s.pairing === 'gauss' ? [...GAUSS_PAIR_SIZES, 0, 0] : PAIR_TEX_SIZES;
}

/** Logical sizes W_s of the 8 pairing-map layers (§2.8; M4 defaults for 6 slots, layers 6–7 unused). */
export const PAIR_TEX_SIZES = [254, 246, 238, 230, 222, 210, 0, 0];
/** M6 Gaussian maps (MD3): the plan's 254 / 230 / 210 first, then the 6-slot / boost layers. */
export const GAUSS_PAIR_SIZES = [254, 230, 210, 246, 238, 222];
