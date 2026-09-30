// Suffix-refresh unit builders (restir-temporal-api.md §3.5, §4.1, §4.4, TD9). OWNER T-C. Called by stage-temporal.ts:
//   refreshFwdUnits  rs_refresh_fwd per row band over the atlas, record res[h] → sfxOut.fwd (before T1)
//   refreshInvUnits  per item chunk: rs_args(q2) + rs_refresh_inv (2D indirect over Q_i), record res[w] → sfxOut.inv
//                    (after T3 phase A, before T4); not emitted under the N1-mixed plant (the loader then uses stored
//                    values, §3.5)
// Both are emitted only on TF_REFRESH frames when the frame's flags are known (`flags`, default: the kernel's current
// advance); the WGSL entry points also return early without TF_REFRESH (and fwd without TF_HIST_VALID), so emitting
// them on any temporal frame is correct, only slower. Bind groups are built with the parity and roles of the frame
// being built (units of a frame are encoded before the next advance(), TD26).
import type { RestirKernel, WorkUnit } from './kernel.ts';
import { RS_WGSL_CONSTS as K } from './layout.ts';

/** Indirect-args byte offset of queue q (rs_args writes rsArgs[4q … 4q+3]). */
const argsOffset = (q: number): number => 16 * q;

function refreshFrame(k: RestirKernel, flags: number | undefined, needHistory: boolean): boolean {
  const f = flags ?? k.currentAdvance?.flags;
  if (f === undefined) return true;                       // unknown: emit (the passes check TF_REFRESH themselves)
  if ((f & K.TF_REFRESH) === 0) return false;
  return !needHistory || (f & K.TF_HIST_VALID) !== 0;
}

/** rs_refresh_fwd over the history res[h] (row bands; costHint atlasW·rows·2, §4.4). */
export function refreshFwdUnits(k: RestirKernel, t: number, flags?: number): WorkUnit[] {
  const h = k.historyIndex();
  if (h < 0 || !refreshFrame(k, flags, true)) return [];
  const res = k.resources;
  const pl = k.pipelineSync('rs_refresh_fwd');
  const g2 = res.g2('rs_refresh_fwd', h);
  return k.rowBands().map(([r0, r1]) => ({
    label: `rs_refresh_fwd[${r0}]`, costHint: res.alloc.atlasW * (r1 - r0) * 2,
    encode: (enc: GPUCommandEncoder) => k.encodePass(enc, 'rs_refresh_fwd', pl, g2,
      { t, passId: K.RS_PASS_T_REFRESH_FWD, rowBase: r0, rowEnd: r1 }, k.perPixelWorkgroups(r0, r1)),
  }));
}

/** rs_args(q2) + rs_refresh_inv over Q_i, one item chunk per row band (Changelog C7 of restir-api.md). */
export function refreshInvUnits(k: RestirKernel, t: number, flags?: number): WorkUnit[] {
  if (!refreshFrame(k, flags, false) || k.settings.tPlant?.n1Mixed) return [];
  const res = k.resources;
  const a = res.alloc;
  const args = k.pipelineSync('rs_args');
  const pl = k.pipelineSync('rs_refresh_inv');
  const w = k.resBase();
  const g2Args = res.g2('rs_args');
  const g2 = res.g2('rs_refresh_inv', w);
  const bands = k.rowBands();
  const P = a.atlasW * a.atlasH;
  const chunk = bands.length > 1 ? Math.max(1, Math.ceil(P / bands.length)) : 0;
  const qFlags = K.RS_Q_INV << K.RSD_QUEUE_SHIFT;
  return bands.map(([r0, r1], ci) => ({
    label: `rs_refresh_inv[${ci}]`, costHint: a.atlasW * (r1 - r0) * 2,
    encode: (enc: GPUCommandEncoder) => {
      const d = { t, passId: K.RS_PASS_T_REFRESH_INV, treeBase: ci * chunk, treeCount: chunk, flags: qFlags };
      k.encodePass(enc, 'rs_args', args, g2Args, d, [1, 1]);
      k.encodePass(enc, 'rs_refresh_inv', pl, g2, d, { indirect: res.args, offset: argsOffset(K.RS_Q_INV) });
    },
  }));
}

// ------------------------------------------------------------------------------------------------ CPU mirrors (tests)

/** Path class of a record (mirror of refresh.wgsl refresh_class; equals layout.ts pathClass for non-empty records):
 *  0 L, 1 N1, 2 B1, 3 E, 4 D-NEE, 5 D-BSDF, 6 R. */
export function refreshClass(d: number, k: number, tech: number): number {
  if (k === 0) return 6;
  const nee = tech === K.RS_TECH_NEE;
  if (k === d) return nee ? 0 : 3;
  if (k + 1 === d) return nee ? 1 : 2;
  return nee ? 4 : 5;
}

const LIGHT_NONE = 0xFFFFFFFF;
const posFinite = (x: number): boolean => x > 0 && Number.isFinite(x);

/** Mirror of refresh.wgsl refresh_entry (f32 J_P): entry of frame `from` translated to frame `to`, J_P =
 *  pmf_to(e_to)/pmf_from(e), undefined iff the entry is missing or a pmf is ≤ 0 (restir-temporal-api.md §3.2). */
export function refreshEntry(entry: number, translate: (e: number) => number, pmfFrom: (e: number) => number, pmfTo: (e: number) => number,
  o: { isEnv?: boolean; noJP?: boolean; noJPEnv?: boolean } = {}): { eTo: number; jp: number; undef: boolean } {
  const eTo = entry === LIGHT_NONE ? LIGHT_NONE : translate(entry);
  const pTo = eTo === LIGHT_NONE ? 0 : Math.fround(pmfTo(eTo));
  const pFrom = entry === LIGHT_NONE ? 0 : Math.fround(pmfFrom(entry));
  const jp = Math.fround(pTo / pFrom);
  const undef = !posFinite(pTo) || !posFinite(pFrom) || !posFinite(jp);
  let j = undef ? 1 : jp;
  if (o.noJP || (o.isEnv && o.noJPEnv)) j = 1;
  return { eTo, jp: j, undef };
}
