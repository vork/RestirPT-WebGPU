// M6 constants and RestirParams accessors (restir-m6-api.md §2.1–§2.3, normative; mirror: render/restir/layout.ts
// RS_M6_CONSTS, checked by U-RES-1). Included ONLY by M6 code (pipeline variants RS_RIS_NEE / RS_MODE_B / RS_DUAL_MV /
// RS_DUPMAP and the passes rs_light_tiles / rs_dupmap), so the M5 pipelines keep their text (MD1).
#include "restir/frame.wgsl"

// RestirParams.flags (feature bits mirror the defines; plants act only inside their variants)
const RSF_RIS_NEE: u32 = 8192u;  const RSF_DUPMAP: u32 = 16384u;  const RSF_DUAL_MV: u32 = 32768u;
const RSF_PLANT_U8_RIS_MIXED: u32 = 65536u;  const RSF_PLANT_U8_TILE_PMF: u32 = 131072u;  const RSF_PLANT_U8_CROSS_OCC: u32 = 262144u;
// Pass ids (resampling-stream separation, restir-api §5) and streams
const RS_PASS_RIS_NEE: u32 = 2u;  const RS_PASS_LIGHT_TILES: u32 = 3u;  const RS_PASS_DUPMAP: u32 = 4u;
const STREAM_LIGHT_TILE: u32 = 0x3c6ef372u;  const STREAM_RIS_NEE: u32 = 0x1b873593u;
// Light tiles (MD4)
const RS_TILES: u32 = 128u;  const RS_TILE_SIZE: u32 = 1024u;  const RS_SCREEN_TILE: u32 = 8u;  const RS_RIS_M_MAX: u32 = 32u;
// Suffix flag of an analytic crossing end (word 27 bit 4, MD8) and the dual-MV counter (MD11)
const SFX_CROSS: u32 = 16u;  const RSC_T_DUAL: u32 = 29u;
// Duplication map (MD10): 17×17 window (half width 8), D = count / 288
const DUP_HALF: u32 = 8u;  const DUP_DENOM: f32 = 288.0;

/// words[] index of the M6 arena region (RestirParams word 120, `pad4` in the M5 struct text).
fn rs_m6_base() -> u32 { return rsParams.pad4; }
/// RIS-NEE candidates at x₁ (RestirParams word 124, `pad5`).
fn rs_ris_m() -> u32 { return min(max(rsParams.pad5, 1u), RS_RIS_M_MAX); }
fn rs_m6_flag(bit: u32) -> bool { return (rsParams.flags & bit) != 0u; }
/// words[] index of the duplication counts (first M6 sub-region, present iff RSF_DUPMAP).
fn rs_dup_base() -> u32 { return rs_m6_base(); }
/// words[] index of the light tiles (after the duplication counts when those exist).
fn rs_tiles_base() -> u32 {
  return rs_m6_base() + select(0u, rsParams.atlasSize.x * rsParams.atlasSize.y, rs_m6_flag(RSF_DUPMAP));
}

/// M(B) of the ReSTIR MIS weights (math.md#mis [M6 addition]): risM at B = 1 with RIS-NEE, else 1. Realised as p2/M
/// (MD5): ω1 = p1/(p1 + p2/M) = M p1/(M p1 + p2) exactly for M a power of two.
fn rs_mis_M(B: u32) -> f32 { return select(1.0, f32(rs_ris_m()), B == 1u && rs_m6_flag(RSF_RIS_NEE)); }
fn rs_p2m(p2: f32, B: u32) -> f32 { return p2 / rs_mis_M(B); }
