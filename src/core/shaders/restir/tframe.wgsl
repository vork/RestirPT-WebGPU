// ReSTIR temporal frame state (restir-temporal-api.md §2.1–§2.8, §3.1–§3.2 and appendix B.2/B.3, normative; math.md
// #temporal, #light-changes [M5 additions]). OWNER T-A (P0). Mirror: render/restir/layout.ts (RS_TEMPORAL_LAYOUT,
// TS_CONSTS, tsWord / sfxWord / queueItemBase / queueCapacityQ).
//   G0 binding 8   RsTemporal (128 B, per frame): flags TF_*, generations, the verbatim EnvParams words of frame t−1
//   frame selector fs ∈ {RS_FS_CUR, RS_FS_PREV}: lf_slot / lf_env / lf_cam_pos read the uniforms on demand (Metal Q1:
//                  no frame struct is kept live)
//   lt_*           alias-entry translation t−1 ↔ t (every entry kind, TD7), realized pmf of a frame, per-light change bits
//   tState/sfxOut  arena extension after the M4 item region (TD17); every accessor returns a words[] index (after the
//                  64-word header, like queue.wgsl's arena_*_word; restir-temporal-api.md Changelog A-1)
#include "restir/frame.wgsl"
#include "restir/queue.wgsl"
#include "lights/select.wgsl"
#include "lights/env.wgsl"

struct RsTemporal {            // 128 B, G0 binding 8 (layout.ts RS_TEMPORAL_SIZE)
  flags: u32,                  //   0  TF_*
  histFrames: u32,             //   4  frames since the last reset (0 on a reset frame)
  frameGen: u32,               //   8  advance() counter of frame t
  prevGen: u32,                //  12  advance() counter of the E_{t−1} snapshot
  envPrev: EnvParams,          //  16  verbatim packed EnvParams words of frame t−1 (32 B)
  gens: vec4u,                 //  48  (camGen, vbufGen, lightGen, pmfGen) of frame t
  gensPrev: vec4u,             //  64  same of frame t−1
  configHash: u32,             //  80
  pad0: u32, pad1: u32, pad2: u32,   // 84..95
  pad3: vec4u, pad4: vec4u,    //  96..127
}
@group(0) @binding(8) var<uniform> rsTemporal: RsTemporal;

// tState flags (TS3/TS2.w) and sfxOut status (appendix B.3)
const TS_QVALID: u32 = 1u;  const TS_DISOCC: u32 = 2u;  const TS_FWD_QUEUED: u32 = 4u;  const TS_FWD_DONE: u32 = 8u;
const TS_SEL_P: u32 = 16u;  const TS_SEL_C: u32 = 32u;  const TS_INV_QUEUED: u32 = 64u;  const TS_INV_DONE: u32 = 128u;
const TS_EMPTY_OUT: u32 = 256u;  const TS_NO_HIST: u32 = 512u;  const TS_PICK_RING: u32 = 1024u;  const TS_ROBUST: u32 = 2048u;
const TS_E2_ZERO: u32 = 4096u;  const TS_FINAL: u32 = 8192u;  const TS_BG: u32 = 16384u;
#if RS_DUAL_MV
/// Dual-MV variant only: T1 found q′ through the dual motion vector (tpick tap ≥ 10; its c_p is DMV_C_CAP-capped,
/// restir-m6-api.md MD11 amendment DMV-1). Diagnostics / debug.
const TS_DUAL_PICK: u32 = 65536u;
#endif
const SXS_DONE: u32 = 1u;  const SXS_UNDEF: u32 = 2u;  const SXS_VIS: u32 = 4u;  const SXS_RAY: u32 = 8u;  const SXS_DEEP: u32 = 16u;
const SXS_N1: u32 = 32u;  const SXS_B1: u32 = 64u;  const SXS_ZERO: u32 = 128u;  const SXS_E2: u32 = 256u;  const SXS_PLANT: u32 = 512u;
const SFX_FWD: u32 = 0u;  const SFX_INV: u32 = 1u;
// tState word indices (§2.8)
const TSW_FWDF: u32 = 0u;  const TSW_FWDJ: u32 = 3u;  const TSW_INVF: u32 = 4u;  const TSW_INVJ: u32 = 7u;
const TSW_QPRIME: u32 = 8u;  const TSW_CP: u32 = 9u;  const TSW_FWDCODE: u32 = 10u;  const TSW_FLAGS: u32 = 11u;
const TSW_JP: u32 = 12u;  const TSW_WC: u32 = 13u;  const TSW_WP: u32 = 14u;  const TSW_INVCODE: u32 = 15u;
const TSW_PISTORED: u32 = 16u;  const TSW_PIRECOMP: u32 = 17u;  const TSW_XPENTRY: u32 = 18u;  const TSW_CPREV: u32 = 19u;
const TS_QPRIME_NONE: u32 = 0xFFFFFFFFu;

/// Model-agnostic record of the refresh (restir-temporal-api.md §3.5, §2.8 sfxOut).
struct SfxRec { rad: vec3f, aux: f32, status: u32, entryTo: u32, jp: f32, gen: u32 }

fn rs_tf(bit: u32) -> bool { return (rsTemporal.flags & bit) != 0u; }
fn rs_tplant(bit: u32) -> bool { return (rsParams.tPlants & bit) != 0u; }
fn rs_tmode(bit: u32) -> bool { return (rsParams.tMode & bit) != 0u; }

// ---- frame selectors (§3.1) ----------------------------------------------------------------------------------------
/// Light slot of frame fs: CUR = lightsParams.cur; PREV = cur under TF_LIGHTS_SAME (or the N1-mixed plant), else prev.
fn lf_slot(fs: u32) -> LightSlot {
  if (fs == RS_FS_CUR || rs_tf(TF_LIGHTS_SAME) || rs_tplant(TP_N1_MIXED)) { return lightsParams.cur; }
  return lightsParams.prev;
}
/// Env record of frame fs: CUR = envParams; PREV = RsTemporal.envPrev (N1-mixed plant: envParams; ENV_GAMMA_T plant:
/// envPrev with the rotation of frame t).
fn lf_env(fs: u32) -> EnvParams {
  if (fs == RS_FS_CUR || rs_tplant(TP_N1_MIXED)) { return envParams; }
  var e = rsTemporal.envPrev;
  if (rs_tplant(TP_ENV_GAMMA_T)) { e.cg = envParams.cg; e.sg = envParams.sg; }
  return e;
}
fn lf_cam_pos(fs: u32) -> vec3f {
  if (fs == RS_FS_PREV) { return frame.prevCam.camToWorld[3].xyz; }
  return frame.cam.camToWorld[3].xyz;
}

// ---- entry translation, pmf, change bits (§3.2) --------------------------------------------------------------------
/// Alias entry `entry` of frame fsFrom in the numbering of frame fsTo; LIGHT_NONE if absent there (TD7).
fn lt_translate(entry: u32, fsFrom: u32, fsTo: u32) -> u32 {
  if (entry == LIGHT_NONE) { return LIGHT_NONE; }
  if (fsFrom == fsTo || rs_tf(TF_LIGHTS_SAME) || rs_tplant(TP_N1_MIXED)) { return entry; }
  let sf = lf_slot(fsFrom);
  let st = lf_slot(fsTo);
  if (entry == sf.envEntry) { return st.envEntry; }
  if (entry < sf.nAnalytic) {
    let cur = lightsParams.cur;
    return records[select(cur.curToPrevOff, cur.prevToCurOff, fsFrom == RS_FS_PREV) + entry];
  }
  let tri = entry - sf.nAnalytic;
  if (tri >= lightsParams.triCount) { return LIGHT_NONE; }
  return st.nAnalytic + tri;
}
/// Realized pmf of entry in frame fs (0 for LIGHT_NONE).
fn lt_pmf(entry: u32, fs: u32) -> f32 {
  if (entry == LIGHT_NONE) { return 0.0; }
  return light_pmf(lf_slot(fs), entry);
}
/// Change bits (LCB_*) of the light behind a frame-t entry: analytic = word 26 of its cur record (relative to the
/// slot's own predecessor; 0 under TF_LIGHTS_SAME), triangles 0, env from TF_ENV_MOVED / TF_ENV_RADIO.
fn lt_change_bits(entryCur: u32) -> u32 {
  let cur = lightsParams.cur;
  if (entryCur == LIGHT_NONE) { return 0u; }
  if (entryCur == cur.envEntry) {
    return select(0u, LCB_MOVED, rs_tf(TF_ENV_MOVED)) | select(0u, LCB_RADIO, rs_tf(TF_ENV_RADIO));
  }
  if (entryCur < cur.nAnalytic) {
    if (rs_tf(TF_LIGHTS_SAME)) { return 0u; }
    return records[cur.lightOff + entryCur * LIGHT_REC_WORDS + 26u];
  }
  return 0u;
}
fn lt_moved(entryCur: u32) -> bool { return (lt_change_bits(entryCur) & LCB_MOVED) != 0u; }

// ---- arena extension (TD16/TD17, §2.8) -----------------------------------------------------------------------------
/// words[] index of the temporal region (= global arena word 64 + 6·P·NS_alloc).
fn arena_tbase() -> u32 { return 6u * rs_atlas_pixels() * rs_ns_alloc(); }
#if RS_TSTATE_SOA
/// perf2 WP-6 (RS_TSTATE_SOA): word-major tState, word w of pixel ai at w·P + ai (same region, same 20·P words).
fn ts_word(ai: u32, w: u32) -> u32 { return arena_tbase() + w * rs_atlas_pixels() + ai; }
#else
fn ts_word(ai: u32, w: u32) -> u32 { return arena_tbase() + 20u * ai + w; }
#endif
fn sfx_word(dir: u32, ai: u32, w: u32) -> u32 { return arena_tbase() + 20u * rs_atlas_pixels() + 16u * ai + 8u * dir + w; }
/// Item base of queue q inside the item region: 0 (q0, q1), P (q2).
fn queue_item_base(q: u32) -> u32 { return select(0u, rs_atlas_pixels(), q == RS_Q_INV); }
/// Capacity of queue q: P·NS_alloc (q0), P (q1, q2).
fn queue_capacity_q(q: u32) -> u32 { return select(rs_atlas_pixels(), queue_capacity(), q == RS_Q_SPATIAL); }

#if RS_TSEL_FOLD
// perf2 WP-6 (RS_TSEL_FOLD; TS mirror layout.ts WP6_CONSTS). RSD_TFOLD (RsDispatch.flags of T4 and T3 phase B):
// contribution MIS without a check mode, T4 finishes the s = c pixel itself (phase B folded into T4) and phase B only
// records the debug views. RS_TSEL_FOLD = 2 also splits Q_i by the replay predicate of its record res[w][q]
// (shift.wgsl res_needs_replay): replay items stay on q2 (bottom of the Q_i region [P, 2P)), non-replay items go on
// q3 (RS_Q_INV_NR, header words 12–15, free during the temporal stage: the spatial stage clears q3 before it uses it)
// from the TOP of the same region (item i at P + P − 1 − i); together at most P items. Both queues run the SAME T4
// pipeline (an RS_REPLAY = 0 T4 for q3 changed bits: the Metal compiler contracts the shift differently there);
// rs_refresh_inv covers both.
const RS_Q_INV_NR: u32 = 3u;
const RSD_TFOLD: u32 = 1048576u;
/// Item region index of item i of Q_i (q2) or its non-replay part (q3).
fn tinv_item_index(q: u32, i: u32) -> u32 {
  let P = rs_atlas_pixels();
  return select(P + i, P + P - 1u - i, q == RS_Q_INV_NR);
}
/// Queue of the T4 dispatch / refresh_inv dispatch (RsDispatch.flags queue field; q2 or q3).
fn tinv_queue() -> u32 { return select(RS_Q_INV, RS_Q_INV_NR, ((rsDispatch.flags >> RSD_QUEUE_SHIFT) & 3u) == RS_Q_INV_NR); }
#endif

#if RS_ARENA_BINDING
fn ts_load(ai: u32, w: u32) -> u32 { return rsArena.words[ts_word(ai, w)]; }
fn ts_loadf(ai: u32, w: u32) -> f32 { return bitcast<f32>(rsArena.words[ts_word(ai, w)]); }
fn sfx_load(dir: u32, ai: u32) -> SfxRec {
  let b = sfx_word(dir, ai, 0u);
  var r: SfxRec;
  r.rad = vec3f(bitcast<f32>(rsArena.words[b]), bitcast<f32>(rsArena.words[b + 1u]), bitcast<f32>(rsArena.words[b + 2u]));
  r.aux = bitcast<f32>(rsArena.words[b + 3u]);
  r.status = rsArena.words[b + 4u];
  r.entryTo = rsArena.words[b + 5u];
  r.jp = bitcast<f32>(rsArena.words[b + 6u]);
  r.gen = rsArena.words[b + 7u];
  return r;
}
#if RS_ARENA_RW
fn ts_store(ai: u32, w: u32, v: u32) { rsArena.words[ts_word(ai, w)] = v; }
fn ts_storef(ai: u32, w: u32, v: f32) { rsArena.words[ts_word(ai, w)] = bitcast<u32>(v); }
#if RS_TSTATE_SOA
/// perf2 WP-6 (RS_TSTATE_SOA): = ts_clear below, each word written once (same final words).
fn ts_clear(ai: u32, flags: u32) {
  let na = rs_slot_code(SC_NOT_ACCEPTED, RCT_NONE, 0u, 0.0);
  ts_store(ai, TSW_FWDF, 0u);  ts_store(ai, TSW_FWDF + 1u, 0u);  ts_store(ai, TSW_FWDF + 2u, 0u);
  ts_store(ai, TSW_FWDJ, JW_FAILED);
  ts_clear_inv(ai);
  ts_store(ai, TSW_QPRIME, TS_QPRIME_NONE);
  ts_store(ai, TSW_CP, 0u);
  ts_store(ai, TSW_FWDCODE, na);
  ts_store(ai, TSW_FLAGS, flags);
  ts_store(ai, TSW_JP, 0u);
  ts_store(ai, TSW_XPENTRY, RC_NONE);
  ts_store(ai, TSW_CPREV, 0u);
}
/// The words T1 never sets beyond their cleared value (inverse block, w̃, π words): INVF 0, INVJ FAILED, WC / WP 0,
/// INVCODE SC_NOT_ACCEPTED, PISTORED / PIRECOMP 0.
fn ts_clear_inv(ai: u32) {
  ts_store(ai, TSW_INVF, 0u);  ts_store(ai, TSW_INVF + 1u, 0u);  ts_store(ai, TSW_INVF + 2u, 0u);
  ts_store(ai, TSW_INVJ, JW_FAILED);
  ts_store(ai, TSW_WC, 0u);
  ts_store(ai, TSW_WP, 0u);
  ts_store(ai, TSW_INVCODE, rs_slot_code(SC_NOT_ACCEPTED, RCT_NONE, 0u, 0.0));
  ts_store(ai, TSW_PISTORED, 0u);
  ts_store(ai, TSW_PIRECOMP, 0u);
}
#else
/// Every tState word of pixel ai to its cleared value (T1): qPrime NONE, codes SC_NOT_ACCEPTED, J words FAILED,
/// xpEntry RC_NONE, everything else 0; then flags.
fn ts_clear(ai: u32, flags: u32) {
  let na = rs_slot_code(SC_NOT_ACCEPTED, RCT_NONE, 0u, 0.0);
  for (var w = 0u; w < 20u; w++) { rsArena.words[ts_word(ai, w)] = 0u; }
  rsArena.words[ts_word(ai, TSW_FWDJ)] = JW_FAILED;
  rsArena.words[ts_word(ai, TSW_INVJ)] = JW_FAILED;
  rsArena.words[ts_word(ai, TSW_QPRIME)] = TS_QPRIME_NONE;
  rsArena.words[ts_word(ai, TSW_FWDCODE)] = na;
  rsArena.words[ts_word(ai, TSW_INVCODE)] = na;
  rsArena.words[ts_word(ai, TSW_XPENTRY)] = RC_NONE;
  rsArena.words[ts_word(ai, TSW_FLAGS)] = flags;
}
#endif
fn sfx_store(dir: u32, ai: u32, r: SfxRec) {
  let b = sfx_word(dir, ai, 0u);
  rsArena.words[b] = bitcast<u32>(r.rad.x);
  rsArena.words[b + 1u] = bitcast<u32>(r.rad.y);
  rsArena.words[b + 2u] = bitcast<u32>(r.rad.z);
  rsArena.words[b + 3u] = bitcast<u32>(r.aux);
  rsArena.words[b + 4u] = r.status;
  rsArena.words[b + 5u] = r.entryTo;
  rsArena.words[b + 6u] = bitcast<u32>(r.jp);
  rsArena.words[b + 7u] = r.gen;
}
#endif
#endif
