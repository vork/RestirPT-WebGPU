// ReSTIR PT shared constants and tiny helpers (restir-api.md appendix B.1, normative; mirror: render/restir/layout.ts,
// checked by U-RES-1). Adding a constant is a contract amendment (append-only numbering, never renumber).
// math.md#path-tree (techniques, endpoint types), #reservoir-fields (flags), #offset-checks / #paired-mis (statuses).
#include "common/nan.wgsl"

// Techniques (math.md#path-tree)
const RS_TECH_NEE: u32 = 0u;  const RS_TECH_BSDF_TRI: u32 = 1u;  const RS_TECH_BSDF_ANALYTIC: u32 = 2u;  const RS_TECH_BSDF_ENV: u32 = 3u;
// rc vertex kinds (rcPairTest)
const RCK_SURFACE: u32 = 0u;  const RCK_LIGHT: u32 = 1u;  const RCK_ENV: u32 = 2u;
// rc / endpoint word-A tags (§2.3)
const RC_TAG_NEE: u32 = 0x80000000u;  const RC_TAG_CROSS: u32 = 0xC0000000u;  const RC_TAG_MASK: u32 = 0xC0000000u;
const RC_ENTRY_MASK: u32 = 0x3FFFFFFFu;  const RC_ENV_DIR: u32 = 0xFFFFFFF1u;  const RC_NONE: u32 = 0xFFFFFFFFu;
const RS_ENV_ID: u32 = 0xFFFFFFFEu;
// Reservoir flags (§2.2)
const RF_D_SHIFT: u32 = 0u;   const RF_K_SHIFT: u32 = 4u;   const RF_TECH_SHIFT: u32 = 8u;  const RF_EP_SHIFT: u32 = 10u;
const RF_ISDELTA: u32 = 0x2000u;  const RF_LKM1_SHIFT: u32 = 14u;  const RF_DKM1: u32 = 0x20000u;
const RF_LK_SHIFT: u32 = 18u;  const RF_DK: u32 = 0x200000u;  const RF_MODE_SHIFT: u32 = 22u;
const RF_FORCED: u32 = 0x1000000u;  const RF_BG: u32 = 0x2000000u;
// J-word statuses (§2.6)
const JW_FAILED: u32 = 0u;  const JW_PENDING: u32 = 0xFFFFFFFEu;  const JW_NOT_ACCEPTED: u32 = 0xFFFFFFFFu;
// Slot outcome codes
const SC_OK: u32 = 0u;  const SC_NOT_ACCEPTED: u32 = 1u;  const SC_EMPTY_SRC: u32 = 2u;  const SC_O0_MISS: u32 = 3u;
const SC_O0_LOBE: u32 = 4u;  const SC_O0_TECH: u32 = 5u;  const SC_O0_LIGHT: u32 = 6u;  const SC_O1: u32 = 7u;
const SC_O2: u32 = 8u;  const SC_O3: u32 = 9u;  const SC_OCCLUDED: u32 = 10u;  const SC_ZERO: u32 = 11u;
const SC_J_INVALID: u32 = 12u;  const SC_O0_SUPPORT: u32 = 13u;  const SC_PENDING: u32 = 14u;  const SC_NONFINITE: u32 = 15u;
// Predicate sub-tests (RcResult.term, slot code bits 8–11)
const RCT_NONE: u32 = 0u;  const RCT_D: u32 = 1u;  const RCT_R: u32 = 2u;  const RCT_F: u32 = 3u;  const RCT_I: u32 = 4u;  const RCT_GUARD: u32 = 5u;
// Arena counters (header word index)
const RSC_CAND_NONFINITE: u32 = 16u;  const RSC_BVH_OVERFLOW: u32 = 17u;  const RSC_BVH_ITERCAP: u32 = 18u;
const RSC_SHIFT_NONFINITE: u32 = 19u;  const RSC_PENDING_LEFT: u32 = 20u;  const RSC_SLOT_MISMATCH: u32 = 21u;
const RSC_W_NONFINITE: u32 = 22u;  const RSC_BASE_JDEN_INVALID: u32 = 23u;  const RSC_ACCEPTED: u32 = 24u;
const RSC_QUEUED: u32 = 25u;  const RSC_SELECTED_SHIFTED: u32 = 26u;  const RSC_EMPTY_CANON: u32 = 27u;  const RSC_CODE_BASE: u32 = 32u;
// RestirParams.flags / RsDispatch.flags
const RSF_RR: u32 = 1u;  const RSF_CRIT_2022: u32 = 2u;  const RSF_PLANT_NO_J: u32 = 4u;  const RSF_PLANT_MARGINAL_J: u32 = 8u;
const RSF_ENSEMBLE: u32 = 16u;  const RSF_INTERACTIVE: u32 = 32u;  const RSF_J_REJECT: u32 = 128u;
const RSD_FIRST_CHUNK: u32 = 1u;  const RSD_FINAL_CHUNK: u32 = 2u;  const RSD_FINAL_ROUND: u32 = 4u;
// Pass ids (RNG stream separation, §5) and stream constants
const RS_PASS_PRIMARY: u32 = 0u;  const RS_PASS_INITIAL: u32 = 1u;  const RS_PASS_TEMPORAL: u32 = 8u;  const RS_PASS_SPATIAL: u32 = 16u;
const STREAM_TREE: u32 = 0x27d4eb2fu;  const STREAM_PAIRING: u32 = 0x165667b1u;
// Limits
const RS_MAX_D: u32 = 15u;  const RS_MAX_SLOTS: u32 = 6u;  const RS_MAX_TREES: u32 = 64u;  const RS_MAX_ROUNDS: u32 = 4u;
const RS_RES_PLANES: u32 = 10u;  const RS_ARENA_HDR_WORDS: u32 = 64u;  const RS_WG: u32 = 64u;
const RS_DUMP_CAP: u32 = 32u;  const RS_DUMP_WORDS: u32 = 48u;

// ---- appended by contract amendments (restir-api.md Changelog) --------------------------------------------------
// A1: interactive progressive-mean bits of the rs_finalize dispatch (RsDispatch.flags)
const RSD_ACCUMULATE: u32 = 8u;  const RSD_ADVANCED: u32 = 16u;
// A2: suffix-cache flag bits (reservoir word 27, §2.2 sfxFlags)
const SFX_BSDF_END: u32 = 1u;  const SFX_ESCAPE: u32 = 2u;  const SFX_VALID: u32 = 4u;
// A3: lobeHist "no event" nibble and word (§2.2 word 23)
const RS_HIST_NONE: u32 = 0xFFFFFFFFu;

// ---- M5 (restir-temporal-api.md §2.1 and appendix B.1, normative; append-only) ----------------------------------
const RS_FS_CUR: u32 = 0u;  const RS_FS_PREV: u32 = 1u;
const TF_HIST_VALID: u32 = 1u;  const TF_LIGHTS_SAME: u32 = 2u;  const TF_ENV_SAME: u32 = 4u;  const TF_REFRESH: u32 = 8u;
const TF_PMF_CHANGED: u32 = 16u;  const TF_ENV_MOVED: u32 = 32u;  const TF_ENV_RADIO: u32 = 64u;  const TF_LIGHT_MOVED: u32 = 128u;
const TF_RESET: u32 = 256u;  const TF_CAM_SAME: u32 = 512u;
const TM_TALBOT: u32 = 1u;  const TM_PP_RECOMPUTE: u32 = 2u;  const TM_ROBUST: u32 = 4u;  const TM_E2: u32 = 8u;
const TP_N1_MIXED: u32 = 1u;  const TP_NO_JP: u32 = 2u;  const TP_NO_JP_ENV: u32 = 4u;  const TP_N3_STALE: u32 = 8u;
const TP_N4_RIS: u32 = 16u;  const TP_N5_PIXEL_CENTRE: u32 = 32u;  const TP_N6_CUR_CAM: u32 = 64u;  const TP_N7_PER_LIGHT: u32 = 128u;
const TP_ENV_NO_ROT_VIS: u32 = 256u;  const TP_ENV_GAMMA_T: u32 = 512u;  const TP_CP_PLUS1: u32 = 1024u;
const TP_U8_STALE_AUX: u32 = 2048u;  const TP_U8_SPOT_PREV_AXIS: u32 = 4096u;
const RSF_TEMPORAL: u32 = 64u;
const RSF_PLANT_U8_W1DELTA: u32 = 256u;  const RSF_PLANT_U8_NO_PK: u32 = 512u;  const RSF_PLANT_U8_T2: u32 = 1024u;
const RSF_PLANT_U8_ONESIDED: u32 = 2048u;  const RSF_PLANT_U8_FAILED_K: u32 = 4096u;
const RSD_QUEUE_SHIFT: u32 = 8u;  const RSD_PHASE_B: u32 = 32u;
const LCB_MOVED: u32 = 1u;  const LCB_RADIO: u32 = 2u;  const LCB_ADDED: u32 = 4u;
const RS_PASS_T_REFRESH_FWD: u32 = 9u;  const RS_PASS_T_CLASSIFY: u32 = 10u;  const RS_PASS_T_FWD: u32 = 11u;
const RS_PASS_T_REFRESH_INV: u32 = 12u;  const RS_PASS_T_INV: u32 = 13u;  const RS_PASS_T_PLANT: u32 = 14u;
const STREAM_TEMPORAL_PICK: u32 = 0x2c1b3c6du;
const SFX_DELTA_END: u32 = 8u;
const RSC_T_QVALID: u32 = 48u;  const RSC_T_DISOCC: u32 = 49u;  const RSC_T_FWD_QUEUED: u32 = 50u;  const RSC_T_FWD_OK: u32 = 51u;
const RSC_T_SEL_P: u32 = 52u;  const RSC_T_INV_QUEUED: u32 = 53u;  const RSC_T_INV_OK: u32 = 54u;  const RSC_T_EMPTY_OUT: u32 = 55u;
const RSC_T_LIGHT_UNDEF: u32 = 56u;  const RSC_T_CLASS_UNDEF: u32 = 57u;  const RSC_T_REFRESH_RECS: u32 = 58u;
const RSC_T_REFRESH_RAYS: u32 = 59u;  const RSC_T_E2_ZEROED: u32 = 60u;  const RSC_T_ROBUST_MISMATCH: u32 = 61u;
const RSC_T_NONFINITE: u32 = 62u;  const RSC_T_PENDING_LEFT: u32 = 63u;
const RS_Q_SPATIAL: u32 = 0u;  const RS_Q_FWD: u32 = 1u;  const RS_Q_INV: u32 = 2u;
// restir-temporal-api.md Changelog B-9: light-driven undefined temporal shifts (free header word 28, both directions)
const RSC_T_LIGHT_CLASS: u32 = 28u;

/// Packed slot code word (§2.6): sc | term << 8 | pair << 12 | f16(margin) << 16.
fn rs_slot_code(sc: u32, term: u32, pair: u32, margin: f32) -> u32 {
  return (sc & 0xFFu) | ((term & 0xFu) << 8u) | ((pair & 0xFu) << 12u) | ((pack2x16float(vec2f(margin, 0.0)) & 0xFFFFu) << 16u);
}
fn rs_slot_code_sc(code: u32) -> u32 { return code & 0xFFu; }

// ---- integer float predicates (gap-rc §3.6; relaxed Metal math folds x != x, so these are bit tests) -------------
/// x is finite and > 0 (+0 and every negative, NaN and Inf fail).
fn rs_pos_finite(x: f32) -> bool {
  let b = bitcast<u32>(x);
  return b != 0u && b < 0x7f800000u;
}
/// a ≥ b for non-negative finite a, b, compared on the bit patterns (monotone for IEEE non-negative floats).
fn rs_geq_pos(a: f32, b: f32) -> bool { return bitcast<u32>(a) >= bitcast<u32>(b); }
