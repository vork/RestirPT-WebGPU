// TS mirror of the ReSTIR WGSL data contract (docs/decisions/restir-api.md §2 and appendix B; checked by U-RES-1 in
// tests/restir/layout.test.ts and restir-initial.gpu.test.ts). Keep in sync with shaders/restir/types.wgsl,
// reservoir.wgsl and frame.wgsl: adding a constant is a contract amendment (append-only numbering).

/** Every constant of shaders/restir/types.wgsl, by its WGSL name (U-RES-1 compares both directions). */
export const RS_WGSL_CONSTS = {
  RS_TECH_NEE: 0, RS_TECH_BSDF_TRI: 1, RS_TECH_BSDF_ANALYTIC: 2, RS_TECH_BSDF_ENV: 3,
  RCK_SURFACE: 0, RCK_LIGHT: 1, RCK_ENV: 2,
  RC_TAG_NEE: 0x80000000, RC_TAG_CROSS: 0xC0000000, RC_TAG_MASK: 0xC0000000,
  RC_ENTRY_MASK: 0x3FFFFFFF, RC_ENV_DIR: 0xFFFFFFF1, RC_NONE: 0xFFFFFFFF,
  RS_ENV_ID: 0xFFFFFFFE,
  RF_D_SHIFT: 0, RF_K_SHIFT: 4, RF_TECH_SHIFT: 8, RF_EP_SHIFT: 10,
  RF_ISDELTA: 0x2000, RF_LKM1_SHIFT: 14, RF_DKM1: 0x20000,
  RF_LK_SHIFT: 18, RF_DK: 0x200000, RF_MODE_SHIFT: 22,
  RF_FORCED: 0x1000000, RF_BG: 0x2000000,
  JW_FAILED: 0, JW_PENDING: 0xFFFFFFFE, JW_NOT_ACCEPTED: 0xFFFFFFFF,
  SC_OK: 0, SC_NOT_ACCEPTED: 1, SC_EMPTY_SRC: 2, SC_O0_MISS: 3,
  SC_O0_LOBE: 4, SC_O0_TECH: 5, SC_O0_LIGHT: 6, SC_O1: 7,
  SC_O2: 8, SC_O3: 9, SC_OCCLUDED: 10, SC_ZERO: 11,
  SC_J_INVALID: 12, SC_O0_SUPPORT: 13, SC_PENDING: 14, SC_NONFINITE: 15,
  RCT_NONE: 0, RCT_D: 1, RCT_R: 2, RCT_F: 3, RCT_I: 4, RCT_GUARD: 5,
  RSC_CAND_NONFINITE: 16, RSC_BVH_OVERFLOW: 17, RSC_BVH_ITERCAP: 18,
  RSC_SHIFT_NONFINITE: 19, RSC_PENDING_LEFT: 20, RSC_SLOT_MISMATCH: 21,
  RSC_W_NONFINITE: 22, RSC_BASE_JDEN_INVALID: 23, RSC_ACCEPTED: 24,
  RSC_QUEUED: 25, RSC_SELECTED_SHIFTED: 26, RSC_EMPTY_CANON: 27, RSC_CODE_BASE: 32,
  RSF_RR: 1, RSF_CRIT_2022: 2, RSF_PLANT_NO_J: 4, RSF_PLANT_MARGINAL_J: 8,
  RSF_ENSEMBLE: 16, RSF_INTERACTIVE: 32, RSF_J_REJECT: 128,
  RSD_FIRST_CHUNK: 1, RSD_FINAL_CHUNK: 2, RSD_FINAL_ROUND: 4,
  RS_PASS_PRIMARY: 0, RS_PASS_INITIAL: 1, RS_PASS_TEMPORAL: 8, RS_PASS_SPATIAL: 16,
  STREAM_TREE: 0x27d4eb2f, STREAM_PAIRING: 0x165667b1,
  RS_MAX_D: 15, RS_MAX_SLOTS: 6, RS_MAX_TREES: 64, RS_MAX_ROUNDS: 4,
  RS_RES_PLANES: 10, RS_ARENA_HDR_WORDS: 64, RS_WG: 64,
  RS_DUMP_CAP: 32, RS_DUMP_WORDS: 48,
  // appended by contract amendments (restir-api.md Changelog)
  RSD_ACCUMULATE: 8, RSD_ADVANCED: 16,
  SFX_BSDF_END: 1, SFX_ESCAPE: 2, SFX_VALID: 4,
  RS_HIST_NONE: 0xFFFFFFFF,
  // M5 (restir-temporal-api.md §2.1, appendix B.1)
  RS_FS_CUR: 0, RS_FS_PREV: 1,
  TF_HIST_VALID: 1, TF_LIGHTS_SAME: 2, TF_ENV_SAME: 4, TF_REFRESH: 8, TF_PMF_CHANGED: 16, TF_ENV_MOVED: 32,
  TF_ENV_RADIO: 64, TF_LIGHT_MOVED: 128, TF_RESET: 256, TF_CAM_SAME: 512,
  TM_TALBOT: 1, TM_PP_RECOMPUTE: 2, TM_ROBUST: 4, TM_E2: 8,
  TP_N1_MIXED: 1, TP_NO_JP: 2, TP_NO_JP_ENV: 4, TP_N3_STALE: 8, TP_N4_RIS: 16, TP_N5_PIXEL_CENTRE: 32, TP_N6_CUR_CAM: 64,
  TP_N7_PER_LIGHT: 128, TP_ENV_NO_ROT_VIS: 256, TP_ENV_GAMMA_T: 512, TP_CP_PLUS1: 1024, TP_U8_STALE_AUX: 2048,
  TP_U8_SPOT_PREV_AXIS: 4096,
  RSF_TEMPORAL: 64,
  RSF_PLANT_U8_W1DELTA: 256, RSF_PLANT_U8_NO_PK: 512, RSF_PLANT_U8_T2: 1024, RSF_PLANT_U8_ONESIDED: 2048, RSF_PLANT_U8_FAILED_K: 4096,
  RSD_QUEUE_SHIFT: 8, RSD_PHASE_B: 32,
  LCB_MOVED: 1, LCB_RADIO: 2, LCB_ADDED: 4,
  RS_PASS_T_REFRESH_FWD: 9, RS_PASS_T_CLASSIFY: 10, RS_PASS_T_FWD: 11, RS_PASS_T_REFRESH_INV: 12, RS_PASS_T_INV: 13, RS_PASS_T_PLANT: 14,
  STREAM_TEMPORAL_PICK: 0x2c1b3c6d,
  SFX_DELTA_END: 8,
  RSC_T_QVALID: 48, RSC_T_DISOCC: 49, RSC_T_FWD_QUEUED: 50, RSC_T_FWD_OK: 51, RSC_T_SEL_P: 52, RSC_T_INV_QUEUED: 53,
  RSC_T_INV_OK: 54, RSC_T_EMPTY_OUT: 55, RSC_T_LIGHT_UNDEF: 56, RSC_T_CLASS_UNDEF: 57, RSC_T_REFRESH_RECS: 58,
  RSC_T_REFRESH_RAYS: 59, RSC_T_E2_ZEROED: 60, RSC_T_ROBUST_MISMATCH: 61, RSC_T_NONFINITE: 62, RSC_T_PENDING_LEFT: 63,
  RS_Q_SPATIAL: 0, RS_Q_FWD: 1, RS_Q_INV: 2,
  // restir-temporal-api.md Changelog B-9 (T-B)
  RSC_T_LIGHT_CLASS: 28,
} as const;
const K = RS_WGSL_CONSTS;

/** Every constant of shaders/restir/m6-types.wgsl (restir-m6-api.md §2.1; included only by M6 code, MD1). */
export const RS_M6_CONSTS = {
  RSF_RIS_NEE: 8192, RSF_DUPMAP: 16384, RSF_DUAL_MV: 32768,
  RSF_PLANT_U8_RIS_MIXED: 65536, RSF_PLANT_U8_TILE_PMF: 131072, RSF_PLANT_U8_CROSS_OCC: 262144,
  RS_PASS_RIS_NEE: 2, RS_PASS_LIGHT_TILES: 3, RS_PASS_DUPMAP: 4,
  STREAM_LIGHT_TILE: 0x3c6ef372, STREAM_RIS_NEE: 0x1b873593,
  RS_TILES: 128, RS_TILE_SIZE: 1024, RS_SCREEN_TILE: 8, RS_RIS_M_MAX: 32,
  SFX_CROSS: 16, RSC_T_DUAL: 29,
  DUP_HALF: 8, DUP_DENOM: 288,
} as const;
const K6 = RS_M6_CONSTS;

/** perf2 WP-5 constants (shaders/restir/queue.wgsl under RS_DENSE_SLOTS / RS_BOOST_GATE; tests/restir/queue.test.ts):
 *  q3 = the dense non-replay spatial slot items (top end of q0's item region); header word 30 = the boost gate. */
export const WP5_CONSTS = { RS_Q_DENSE: 3, RS_HDR_BOOST_GATE: 30, RSD_BOOST_OPEN: 4096 } as const;
/** perf2 WP-6 constant (shaders/restir/tframe.wgsl under RS_TSEL_FOLD): RsDispatch flag RSD_TFOLD = T3 phase B folded
 *  into T4 (contribution MIS without a check mode). */
export const WP6_CONSTS = { RSD_TFOLD: 1048576 } as const;

export const RS_TECH = { nee: K.RS_TECH_NEE, bsdfTri: K.RS_TECH_BSDF_TRI, bsdfAnalytic: K.RS_TECH_BSDF_ANALYTIC, bsdfEnv: K.RS_TECH_BSDF_ENV } as const;
export const RS_TECH_NAMES = ['NEE', 'BSDF_TRI', 'BSDF_ANALYTIC', 'BSDF_ENV'] as const;
/** Lobe codes (bsdf.wgsl LOBE_*). */
export const LOBE = { D: 0, S: 1, GR: 2, GT: 3, NEE: 4, NONE: 5 } as const;
/** Endpoint types = lights.wgsl LT_*. */
export const EP_TYPE = { tri: 0, point: 1, spot: 2, rect: 3, disk: 4, sun: 5, env: 7 } as const;

export const SC_NAMES = ['OK', 'NOT_ACCEPTED', 'EMPTY_SRC', 'O0_MISS', 'O0_LOBE', 'O0_TECH', 'O0_LIGHT', 'O1', 'O2', 'O3',
  'OCCLUDED', 'ZERO', 'J_INVALID', 'O0_SUPPORT', 'PENDING', 'NONFINITE'] as const;

/** Arena counters (header word index) by name. */
export const RSC = {
  candNonFinite: K.RSC_CAND_NONFINITE, bvhOverflow: K.RSC_BVH_OVERFLOW, bvhItercap: K.RSC_BVH_ITERCAP,
  shiftNonFinite: K.RSC_SHIFT_NONFINITE, pendingLeft: K.RSC_PENDING_LEFT, slotMismatch: K.RSC_SLOT_MISMATCH,
  wNonFinite: K.RSC_W_NONFINITE, baseJdenInvalid: K.RSC_BASE_JDEN_INVALID, accepted: K.RSC_ACCEPTED,
  queued: K.RSC_QUEUED, selectedShifted: K.RSC_SELECTED_SHIFTED, emptyCanon: K.RSC_EMPTY_CANON,
  // M5 temporal counters (header words 48–63, restir-temporal-api.md §2.1)
  tQvalid: K.RSC_T_QVALID, tDisocc: K.RSC_T_DISOCC, tFwdQueued: K.RSC_T_FWD_QUEUED, tFwdOk: K.RSC_T_FWD_OK, tSelP: K.RSC_T_SEL_P,
  tInvQueued: K.RSC_T_INV_QUEUED, tInvOk: K.RSC_T_INV_OK, tEmptyOut: K.RSC_T_EMPTY_OUT, tLightUndef: K.RSC_T_LIGHT_UNDEF,
  tClassUndef: K.RSC_T_CLASS_UNDEF, tRefreshRecs: K.RSC_T_REFRESH_RECS, tRefreshRays: K.RSC_T_REFRESH_RAYS, tE2Zeroed: K.RSC_T_E2_ZEROED,
  tRobustMismatch: K.RSC_T_ROBUST_MISMATCH, tNonFinite: K.RSC_T_NONFINITE, tPendingLeft: K.RSC_T_PENDING_LEFT,
  tLightClass: K.RSC_T_LIGHT_CLASS,
  // M6 (restir-m6-api.md MD11)
  tDual: K6.RSC_T_DUAL,
} as const;
export type RscName = keyof typeof RSC;

// ------------------------------------------------------------------------------------------------ reservoir record

export const RES_PLANES = 10;
export const RES_WORDS = 40;
export const RES_BYTES = 160;
/** Plane indices (reservoir.wgsl RP_*). */
export const RP = { WF: 0, SEED: 1, RC: 2, WI: 3, RAD: 4, END: 5, SFX0: 6, SFX1: 7, SFX2: 8, DIAG: 9 } as const;
/** Word offsets inside a record (§2.2). */
export const RW = {
  W: 0, F: 1, seed: 4, flags: 6, c: 7, rc: 8, jDen: 11, rcWi: 12, aux: 15, rcRad: 16, wSum: 19,
  end: 20, lobeHist: 23, sfx: 24, sfxFlags: 27, sfxDir: 28, sfxT: 31, betaS: 32, sfxP2: 35,
  nCand: 36, selId: 37, kMargin: 38, endpointId: 39,
} as const;

export interface RfFields {
  d: number; k: number; tech: number; ep: number; isDelta: boolean;
  lkm1: number; dkm1: boolean; lk: number; dk: boolean; mode: number; forced: boolean; bg: boolean;
}

/** rf_pack (reservoir.wgsl) + the mode and bg bits. */
export function rfPack(f: Partial<RfFields>): number {
  let v = ((f.d ?? 0) & 0xF) | (((f.k ?? 0) & 0xF) << K.RF_K_SHIFT) | (((f.tech ?? 0) & 3) << K.RF_TECH_SHIFT) | (((f.ep ?? 0) & 7) << K.RF_EP_SHIFT)
    | (((f.lkm1 ?? 0) & 7) << K.RF_LKM1_SHIFT) | (((f.lk ?? 0) & 7) << K.RF_LK_SHIFT) | (((f.mode ?? 0) & 3) << K.RF_MODE_SHIFT);
  if (f.isDelta) v |= K.RF_ISDELTA;
  if (f.dkm1) v |= K.RF_DKM1;
  if (f.dk) v |= K.RF_DK;
  if (f.forced) v |= K.RF_FORCED;
  if (f.bg) v |= K.RF_BG;
  return v >>> 0;
}

export function rfUnpack(v: number): RfFields {
  return {
    d: v & 0xF, k: (v >>> K.RF_K_SHIFT) & 0xF, tech: (v >>> K.RF_TECH_SHIFT) & 3, ep: (v >>> K.RF_EP_SHIFT) & 7,
    isDelta: (v & K.RF_ISDELTA) !== 0, lkm1: (v >>> K.RF_LKM1_SHIFT) & 7, dkm1: (v & K.RF_DKM1) !== 0,
    lk: (v >>> K.RF_LK_SHIFT) & 7, dk: (v & K.RF_DK) !== 0, mode: (v >>> K.RF_MODE_SHIFT) & 3,
    forced: (v & K.RF_FORCED) !== 0, bg: (v & K.RF_BG) !== 0,
  };
}

export interface ReservoirRecord extends RfFields {
  W: number; F: [number, number, number]; seed: [number, number]; flags: number; c: number;
  rc: [number, number, number]; jDen: number; rcWi: [number, number, number]; aux: number;
  rcRad: [number, number, number]; wSum: number; end: [number, number, number]; lobeHist: number;
  sfx: [number, number, number]; sfxFlags: number; sfxDir: [number, number, number]; sfxT: number;
  betaS: [number, number, number]; sfxP2: number; nCand: number; selId: number; kMargin: number; endpointId: number;
}

/** Decode record `i` of a reservoir buffer read back as u32 words (`words` = RES_WORDS per record, from `base`). */
export function decodeReservoir(u: Uint32Array, i: number, base = 0): ReservoirRecord {
  const o = base + i * RES_WORDS;
  const f = new Float32Array(u.buffer, u.byteOffset, u.length);
  const f3 = (w: number): [number, number, number] => [f[o + w], f[o + w + 1], f[o + w + 2]];
  const u3 = (w: number): [number, number, number] => [u[o + w], u[o + w + 1], u[o + w + 2]];
  const flags = u[o + RW.flags];
  return {
    ...rfUnpack(flags),
    W: f[o + RW.W], F: f3(RW.F), seed: [u[o + RW.seed], u[o + RW.seed + 1]], flags, c: f[o + RW.c],
    rc: u3(RW.rc), jDen: f[o + RW.jDen], rcWi: f3(RW.rcWi), aux: f[o + RW.aux], rcRad: f3(RW.rcRad), wSum: f[o + RW.wSum],
    end: u3(RW.end), lobeHist: u[o + RW.lobeHist], sfx: u3(RW.sfx), sfxFlags: u[o + RW.sfxFlags], sfxDir: f3(RW.sfxDir),
    sfxT: f[o + RW.sfxT], betaS: f3(RW.betaS), sfxP2: f[o + RW.sfxP2], nCand: u[o + RW.nCand], selId: u[o + RW.selId],
    kMargin: f[o + RW.kMargin], endpointId: u[o + RW.endpointId],
  };
}

/** Path class (math.md#reservoir-fields; view 409 codes 0…6: L, N1, B1, E, D-NEE, D-BSDF, R); -1 for empty. */
export function pathClass(r: Pick<RfFields, 'd' | 'k' | 'tech'>): number {
  if (r.d === 0) return -1;
  if (r.k === 0) return 6;
  const nee = r.tech === RS_TECH.nee;
  if (r.k === r.d) return nee ? 0 : 3;
  if (r.k === r.d - 1) return nee ? 1 : 2;
  return nee ? 4 : 5;
}
export const PATH_CLASS_NAMES = ['L', 'N1', 'B1', 'E', 'D-NEE', 'D-BSDF', 'R'] as const;

// ------------------------------------------------------------------------------------------------ uniforms

export const RESTIR_PARAMS_SIZE = 128;
/** Byte offsets of RestirParams (frame.wgsl, appendix B.3). */
export const RESTIR_PARAMS_LAYOUT = {
  atlasSize: 0, memberSize: 8, memberCols: 16, memberCount: 20, maxBounces: 24, flags: 28, numTrees: 32, numSlots: 36,
  numRounds: 40, rrMinBounces: 44, tau: 48, alphaMin: 52, wScale: 56, crit2022MinDist: 60, pairTexSize: 64,
  pairTexSize2: 80, lightMode: 96, memberBase: 100,
  // M5 (restir-temporal-api.md §2.7, appendix B.2; were pad1 … pad3)
  boostSlots: 104, tMode: 108, cCap: 112, tPlants: 116, pad4: 120, pad5: 124,
} as const;

export interface RestirParamsCpu {
  atlasSize: [number, number]; memberSize: [number, number]; memberCols: number; memberCount: number;
  maxBounces: number; flags: number; numTrees: number; numSlots: number; numRounds: number; rrMinBounces: number;
  tau: number; alphaMin: number; wScale: number; crit2022MinDist: number; pairTexSize: number[]; lightMode: number;
  memberBase: number;
  /** M5 (appendix B.2): boost slots NB (numSlots already includes them), TM_* word, c cap (20), TP_* word. */
  boostSlots?: number; tMode?: number; cCap?: number; tPlants?: number;
  /** M6 (restir-m6-api.md §2.2; words 120/124, the WGSL names stay pad4 / pad5): words[] index of the M6 arena region
   *  and the RIS-NEE candidate count. 0 when no M6 feature needs them (the M5 uniform bytes). */
  m6Base?: number; risM?: number;
}

export function packRestirParams(p: RestirParamsCpu, out = new ArrayBuffer(RESTIR_PARAMS_SIZE)): ArrayBuffer {
  const u = new Uint32Array(out), f = new Float32Array(out), L = RESTIR_PARAMS_LAYOUT;
  u.fill(0);
  u[L.atlasSize / 4] = p.atlasSize[0]; u[L.atlasSize / 4 + 1] = p.atlasSize[1];
  u[L.memberSize / 4] = p.memberSize[0]; u[L.memberSize / 4 + 1] = p.memberSize[1];
  u[L.memberCols / 4] = p.memberCols; u[L.memberCount / 4] = p.memberCount; u[L.maxBounces / 4] = p.maxBounces;
  u[L.flags / 4] = p.flags >>> 0; u[L.numTrees / 4] = p.numTrees; u[L.numSlots / 4] = p.numSlots;
  u[L.numRounds / 4] = p.numRounds; u[L.rrMinBounces / 4] = p.rrMinBounces;
  f[L.tau / 4] = p.tau; f[L.alphaMin / 4] = p.alphaMin; f[L.wScale / 4] = p.wScale; f[L.crit2022MinDist / 4] = p.crit2022MinDist;
  for (let i = 0; i < 8; i++) u[L.pairTexSize / 4 + i] = p.pairTexSize[i] ?? 0;
  u[L.lightMode / 4] = p.lightMode; u[L.memberBase / 4] = p.memberBase;
  u[L.boostSlots / 4] = p.boostSlots ?? 0; u[L.tMode / 4] = (p.tMode ?? 0) >>> 0; f[L.cCap / 4] = p.cCap ?? 20;
  u[L.tPlants / 4] = (p.tPlants ?? 0) >>> 0;
  u[L.pad4 / 4] = (p.m6Base ?? 0) >>> 0; u[L.pad5 / 4] = (p.risM ?? 0) >>> 0;
  return out;
}

export const RS_DISPATCH_SIZE = 32;
export const RS_DISPATCH_STRIDE = 256;
export const RS_DISPATCH_RING = 512;
export interface RsDispatchCpu { t: number; passId: number; round: number; treeBase: number; treeCount: number; flags: number; rowBase: number; rowEnd: number }
export function packRsDispatch(d: RsDispatchCpu): Uint32Array {
  return Uint32Array.from([d.t >>> 0, d.passId, d.round, d.treeBase, d.treeCount, d.flags, d.rowBase, d.rowEnd]);
}

// ------------------------------------------------------------------------------------------------ shift arena, dump

export const ARENA_HDR_BYTES = 256;
/** NS_alloc (restir-temporal-api.md TD16, B.3): numSlots, or max(numSlots, 2) with temporal on (Q_f + Q_i need 2P items). */
export const nsAlloc = (numSlots: number, temporal = false): number => (temporal ? Math.max(numSlots, 2) : numSlots);
/** Arena size for P atlas pixels and NS (= NS_alloc) slots (§2.6; M5 §2.8: + 36·P words of tState / sfxOut). */
export const arenaBytes = (P: number, NS: number, temporal = false): number => ARENA_HDR_BYTES + 24 * P * NS + (temporal ? 144 * P : 0);
/** Word offsets inside the arena's words[] (after the 64-word header). M5: tState at `tState`, sfxOut at `sfxOut`
 *  (words[] indices = global word − 64, as every WGSL arena accessor; restir-temporal-api.md Changelog A-1). */
export const arenaWords = (P: number, NS: number) => ({
  slots: 0, codes: 4 * P * NS, items: 5 * P * NS, tState: 6 * P * NS, sfxOut: 6 * P * NS + 20 * P, end: 6 * P * NS + 36 * P,
});
/** M6 arena region (restir-m6-api.md §2.3), words[] indices: the duplication counts (P words, dupmap) and the light
 *  tiles (E·128·1024 words, RIS-NEE), appended after the M4 / M5 regions. */
export interface ArenaM6 { dup: boolean; tileMembers: number }
export const arenaM6Base = (P: number, NS: number, temporal: boolean): number => 6 * P * NS + (temporal ? 36 * P : 0);
export const arenaM6Words = (P: number, m6: ArenaM6): number => (m6.dup ? P : 0) + m6.tileMembers * K6.RS_TILES * K6.RS_TILE_SIZE;
export const arenaM6Layout = (P: number, NS: number, temporal: boolean, m6: ArenaM6) => {
  const base = arenaM6Base(P, NS, temporal);
  return { base, dup: base, tiles: base + (m6.dup ? P : 0), end: base + arenaM6Words(P, m6) };
};
/** Arena bytes including the M6 region. */
export const arenaBytesM6 = (P: number, NS: number, temporal: boolean, m6: ArenaM6): number => arenaBytes(P, NS, temporal) + 4 * arenaM6Words(P, m6);
/** Queue q header words {counter, n, capacity, overflow}. */
export const queueHdr = (q: number) => ({ counter: 4 * q, n: 4 * q + 1, capacity: 4 * q + 2, overflow: 4 * q + 3 });

export const RS_DUMP_CAP = K.RS_DUMP_CAP;
export const RS_DUMP_WORDS = K.RS_DUMP_WORDS;
/** Candidate dump bytes for P atlas pixels (§2.12): records, then one u32 count per pixel. */
export const dumpBytes = (P: number): number => (P * RS_DUMP_CAP * RS_DUMP_WORDS + P) * 4;
export const dumpCountWord = (P: number, ai: number): number => P * RS_DUMP_CAP * RS_DUMP_WORDS + ai;
export const dumpRecordWord = (ai: number, n: number): number => (ai * RS_DUMP_CAP + n) * RS_DUMP_WORDS;

// ------------------------------------------------------------------------------------------------ debug ids (§2.11)

export const RS_VIEW = {
  c: 400, W: 401, phat: 402, FW: 403, d: 404, k: 405, tech: 406, endpoint: 407, lobes: 408, class: 409, kMargin: 410,
  shiftCode: 420, shiftLogJ: 426, shiftTerm: 432, replayMask: 438, acceptMask: 439, shiftMargin: 440,
  misMc: 460, misSumM: 461, misLumL: 462, misMj: 463, misK: 469, misSel: 470,
} as const;
export const RS_PROBE_TAG = { plane: 64, header: 65, candidate: 66, vertex: 67, slot: 68, misPartner: 69, misCanonical: 70 } as const;

/** 4-bit lobe history nibble of vertex b (1…8) of a record's lobeHist (0xF = none). */
export const lobeHistAt = (h: number, b: number): number => (h >>> (4 * (b - 1))) & 0xF;

// ------------------------------------------------------------------------------------------------ M5 temporal (§2.7–§2.8)

/** RsTemporal uniform (tframe.wgsl, G0 binding 8, appendix B.2). */
export const RS_TEMPORAL_SIZE = 128;
export const RS_TEMPORAL_LAYOUT = { flags: 0, histFrames: 4, frameGen: 8, prevGen: 12, envPrev: 16, gens: 48, gensPrev: 64, configHash: 80 } as const;
export interface RsTemporalCpu {
  flags: number; histFrames: number; frameGen: number; prevGen: number;
  /** Verbatim packed EnvParams bytes of frame t−1 (env-gpu.ts packEnvParams, 32 B). */
  envPrev: ArrayBuffer | Uint32Array;
  gens: [number, number, number, number]; gensPrev: [number, number, number, number]; configHash: number;
}
export function packRsTemporal(t: RsTemporalCpu, out = new ArrayBuffer(RS_TEMPORAL_SIZE)): ArrayBuffer {
  const u = new Uint32Array(out), L = RS_TEMPORAL_LAYOUT;
  u.fill(0);
  u[L.flags / 4] = t.flags >>> 0; u[L.histFrames / 4] = t.histFrames >>> 0; u[L.frameGen / 4] = t.frameGen >>> 0; u[L.prevGen / 4] = t.prevGen >>> 0;
  const env = t.envPrev instanceof Uint32Array ? t.envPrev : new Uint32Array(t.envPrev);
  if (env.length !== 8) throw new Error(`packRsTemporal: envPrev must be 8 words (got ${env.length})`);
  u.set(env, L.envPrev / 4);
  u.set(t.gens.map((x) => x >>> 0), L.gens / 4); u.set(t.gensPrev.map((x) => x >>> 0), L.gensPrev / 4);
  u[L.configHash / 4] = t.configHash >>> 0;
  return out;
}
export function unpackRsTemporal(u: Uint32Array): RsTemporalCpu {
  const L = RS_TEMPORAL_LAYOUT;
  return {
    flags: u[L.flags / 4], histFrames: u[L.histFrames / 4], frameGen: u[L.frameGen / 4], prevGen: u[L.prevGen / 4],
    envPrev: u.slice(L.envPrev / 4, L.envPrev / 4 + 8), gens: Array.from(u.subarray(L.gens / 4, L.gens / 4 + 4)) as RsTemporalCpu['gens'],
    gensPrev: Array.from(u.subarray(L.gensPrev / 4, L.gensPrev / 4 + 4)) as RsTemporalCpu['gensPrev'], configHash: u[L.configHash / 4],
  };
}

/** tState flags TS_* and sfxOut status SXS_* (tframe.wgsl, appendix B.3). */
export const TS_CONSTS = {
  TS_QVALID: 1, TS_DISOCC: 2, TS_FWD_QUEUED: 4, TS_FWD_DONE: 8, TS_SEL_P: 16, TS_SEL_C: 32, TS_INV_QUEUED: 64, TS_INV_DONE: 128,
  TS_EMPTY_OUT: 256, TS_NO_HIST: 512, TS_PICK_RING: 1024, TS_ROBUST: 2048, TS_E2_ZERO: 4096, TS_FINAL: 8192, TS_BG: 16384,
  TS_DUAL_PICK: 65536,   // RS_DUAL_MV variant only (q′ found by the dual MV; T3 resamples it with Talbot MIS, DMV-1)
  SXS_DONE: 1, SXS_UNDEF: 2, SXS_VIS: 4, SXS_RAY: 8, SXS_DEEP: 16, SXS_N1: 32, SXS_B1: 64, SXS_ZERO: 128, SXS_E2: 256, SXS_PLANT: 512,
  SFX_FWD: 0, SFX_INV: 1,
} as const;
export const TS_WORDS = 20;
export const SFX_REC_WORDS = 8;
/** tState word names (§2.8; word index inside the 20-word per-pixel block). */
export const TSW = {
  fwdF: 0, fwdJ: 3, invF: 4, invJ: 7, qPrime: 8, cP: 9, fwdCode: 10, flags: 11, jP: 12, wc: 13, wp: 14, invCode: 15,
  piStored: 16, piRecomp: 17, xpEntry: 18, cPrev: 19,
} as const;
/** words[] index of tState[ai] word w / sfxOut[dir][ai] word w (mirror of tframe.wgsl ts_word / sfx_word). */
export const tsWord = (P: number, NS: number, ai: number, w: number): number => arenaWords(P, NS).tState + TS_WORDS * ai + w;
export const sfxWord = (P: number, NS: number, dir: number, ai: number, w: number): number => arenaWords(P, NS).sfxOut + 16 * ai + 8 * dir + w;
/** perf2 WP-6 (RS_TSTATE_SOA): the GPU tState region is word-major (word w of pixel ai at w·P + ai). These
 *  convert the first TS_WORDS·P words of a readTemporalState()-shaped array between that and the record-major layout
 *  every CPU decoder uses (sfxOut and the rest are unchanged). */
export function tStateSoaToAos(words: Uint32Array, P: number): Uint32Array {
  const out = words.slice();
  for (let w = 0; w < TS_WORDS; w++) for (let ai = 0; ai < P; ai++) out[TS_WORDS * ai + w] = words[w * P + ai];
  return out;
}
export function tStateAosToSoa(words: Uint32Array, P: number): Uint32Array {
  const out = words.slice();
  for (let w = 0; w < TS_WORDS; w++) for (let ai = 0; ai < P; ai++) out[w * P + ai] = words[TS_WORDS * ai + w];
  return out;
}
/** Queue item base (within the item region) and capacity of queue q (tframe.wgsl queue_item_base / queue_capacity_q). */
export const queueItemBase = (P: number, q: number): number => (q === 2 ? P : 0);
export const queueCapacityQ = (P: number, NS: number, q: number): number => (q === 0 ? P * NS : P);

export interface TStateRecord {
  fwdF: [number, number, number]; fwdJ: number; invF: [number, number, number]; invJ: number; qPrime: number; cP: number;
  fwdCode: number; flags: number; jP: number; wc: number; wp: number; invCode: number; piStored: number; piRecomp: number;
  xpEntry: number; cPrev: number;
}
export interface SfxRecord { rad: [number, number, number]; aux: number; status: number; entryTo: number; jp: number; gen: number }

/** Decode tState[ai] from the arena's words[] (read back as u32, header excluded; `base` = words[] offset of 0). */
export function decodeTState(words: Uint32Array, P: number, NS: number, ai: number, base = 0): TStateRecord {
  const o = base + tsWord(P, NS, ai, 0);
  const f = new Float32Array(words.buffer, words.byteOffset, words.length);
  return {
    fwdF: [f[o], f[o + 1], f[o + 2]], fwdJ: words[o + 3], invF: [f[o + 4], f[o + 5], f[o + 6]], invJ: words[o + 7],
    qPrime: words[o + 8], cP: f[o + 9], fwdCode: words[o + 10], flags: words[o + 11], jP: f[o + 12], wc: f[o + 13], wp: f[o + 14],
    invCode: words[o + 15], piStored: f[o + 16], piRecomp: f[o + 17], xpEntry: words[o + 18], cPrev: f[o + 19],
  };
}
export function decodeSfx(words: Uint32Array, P: number, NS: number, dir: number, ai: number, base = 0): SfxRecord {
  const o = base + sfxWord(P, NS, dir, ai, 0);
  const f = new Float32Array(words.buffer, words.byteOffset, words.length);
  return { rad: [f[o], f[o + 1], f[o + 2]], aux: f[o + 3], status: words[o + 4], entryTo: words[o + 5], jp: f[o + 6], gen: words[o + 7] };
}
/** Decoders for RestirKernel.readTemporalState() (words start at tState). */
export const decodeTStateLocal = (words: Uint32Array, P: number, NS: number, ai: number): TStateRecord => decodeTState(words, P, NS, ai, -arenaWords(P, NS).tState);
export const decodeSfxLocal = (words: Uint32Array, P: number, NS: number, dir: number, ai: number): SfxRecord => decodeSfx(words, P, NS, dir, ai, -arenaWords(P, NS).tState);
