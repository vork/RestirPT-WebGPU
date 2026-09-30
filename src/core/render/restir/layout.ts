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
} as const;
const K = RS_WGSL_CONSTS;

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
} as const;

export interface RestirParamsCpu {
  atlasSize: [number, number]; memberSize: [number, number]; memberCols: number; memberCount: number;
  maxBounces: number; flags: number; numTrees: number; numSlots: number; numRounds: number; rrMinBounces: number;
  tau: number; alphaMin: number; wScale: number; crit2022MinDist: number; pairTexSize: number[]; lightMode: number;
  memberBase: number;
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
/** Arena size for P atlas pixels and NS slots (§2.6). */
export const arenaBytes = (P: number, NS: number): number => ARENA_HDR_BYTES + 24 * P * NS;
/** Word offsets inside the arena's words[] (after the 64-word header). */
export const arenaWords = (P: number, NS: number) => ({ slots: 0, codes: 4 * P * NS, items: 5 * P * NS });
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
