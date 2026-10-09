// Temporal stage (restir-temporal-api.md §3.6, §4.1, §4.4; PLAN §3 steps 2–4; math.md#temporal). OWNER T-B.
// Per frame with temporal on (after rs_initial, before spatial, TD15):
//   clear q1/q2 {counter, n} (overflow sticky, C1) → [rs_refresh_fwd (T-C)] → rs_t_classify (T1, row bands) →
//   per item chunk rs_args(q1) + rs_t_forward (T2) → rs_t_select phase A (row bands) → [rs_args(q2) + rs_refresh_inv
//   (T-C)] → per item chunk rs_args(q2) + rs_t_inverse (T4) → rs_t_select phase B (row bands).
// Reservoir roles from the kernel (TD2): h = k.historyIndex() (read-only), w = k.resBase() (written in place); after a
// reset (h = −1) only T1 and T3-A run (flags, taps; the canonical is untouched, §4.1) and the history slot 1 − w is bound
// but never read. Every stage covers the whole atlas before the next starts (units in order). Item chunks follow
// restir-api.md Changelog C7 (one chunk per row band; RsDispatch.treeBase / treeCount = item base / count), so the
// result is bitwise independent of row bands and chunks (U-TR-2). Unit labels start with the pass kind (E1).
import type { RestirKernel, RestirStage, WorkUnit } from './kernel.ts';
import { RS_WGSL_CONSTS as K, WP5_CONSTS } from './layout.ts';
import { refreshFwdUnits, refreshInvUnits } from './refresh.ts';
import { TEMPORAL_PASSES, type RsPassName } from './resources.ts';

export class TemporalStage implements RestirStage {
  async prepare(k: RestirKernel): Promise<void> {
    await Promise.all([...TEMPORAL_PASSES, 'rs_args' as const].map((n) => k.pipeline(n)));
  }

  frameUnits(k: RestirKernel, t: number): WorkUnit[] {
    const res = k.resources, a = res.alloc, s = k.settings;
    const pl = (n: RsPassName) => k.pipelineSync(n);
    const classify = pl('rs_t_classify'), forward = pl('rs_t_forward'), select = pl('rs_t_select'), inverse = pl('rs_t_inverse'), args = pl('rs_args');
    const w = k.resBase();
    const hIdx = 1 - w;                                   // = h with history; the unread other buffer after a reset
    const hist = k.historyIndex() >= 0;
    const bands = k.rowBands();
    const P = a.atlasW * a.atlasH;
    const chunk = bands.length > 1 ? Math.max(1, Math.ceil(P / bands.length)) : 0;
    const nChunks = bands.length;
    const rows = (r0: number, r1: number) => a.atlasW * (r1 - r0);
    // bind groups resolved now (the parity and roles belong to this frame)
    const gClassify = res.g2('rs_t_classify', hIdx), gForward = res.g2('rs_t_forward', hIdx), gSelect = res.g2('rs_t_select', hIdx);
    const gInverse = res.g2('rs_t_inverse', w), gArgs = res.g2('rs_args');
    const units: WorkUnit[] = [];
    const perPixel = (label: string, name: RsPassName, pipe: GPUComputePipeline, g2: GPUBindGroup, d: { passId: number; round?: number; flags?: number }, first?: (enc: GPUCommandEncoder) => void) => {
      bands.forEach(([r0, r1], bi) => units.push({
        label: `${label}[${r0}]`, costHint: rows(r0, r1),
        encode: (enc) => {
          if (bi === 0) first?.(enc);
          k.encodePass(enc, name, pipe, g2, { t, ...d, rowBase: r0, rowEnd: r1 }, k.perPixelWorkgroups(r0, r1));
        },
      }));
    };
    const indirect = (label: string, name: RsPassName, pipe: GPUComputePipeline, g2: GPUBindGroup, q: number, passId: number) => {
      for (let ci = 0; ci < nChunks; ci++) {
        const items = chunk > 0 ? chunk : P;
        units.push({
          label: `${label}[${ci}]`, costHint: items * (s.maxBounces + 1),
          encode: (enc) => {
            const d = { t, passId, treeBase: ci * chunk, treeCount: chunk };
            k.encodePass(enc, 'rs_args', args, gArgs, { ...d, flags: q << K.RSD_QUEUE_SHIFT }, [1, 1]);
            k.encodePass(enc, name, pipe, g2, d, { indirect: res.args, offset: 16 * q });
          },
        });
      }
    };
    const boostGate = !!k.perfFlags.RS_BOOST_GATE;
    const clearQueues = (enc: GPUCommandEncoder) => {
      enc.clearBuffer(res.arena, 16 * K.RS_Q_FWD, 8);
      enc.clearBuffer(res.arena, 16 * K.RS_Q_INV, 8);
      // perf2 WP-5 (RS_BOOST_GATE): T1 sets the any-disocclusion word; it is cleared here, before T1, every frame
      if (boostGate) enc.clearBuffer(res.arena, 4 * WP5_CONSTS.RS_HDR_BOOST_GATE, 4);
    };
    // The queue clear rides on the first unit of the stage (a unit of its own would be submitted alone until measured, E1).
    const pre = hist ? refreshFwdUnits(k, t) : [];
    if (pre.length > 0) {
      const u0 = pre[0], enc0 = u0.encode;
      pre[0] = { ...u0, encode: (enc) => { clearQueues(enc); enc0(enc); } };
    }
    units.push(...pre);
    perPixel('rs_t_classify', 'rs_t_classify', classify, gClassify, { passId: K.RS_PASS_T_CLASSIFY }, pre.length > 0 ? undefined : clearQueues);
    if (hist) indirect('rs_t_forward', 'rs_t_forward', forward, gForward, K.RS_Q_FWD, K.RS_PASS_T_FWD);
    perPixel('rs_t_select_a', 'rs_t_select', select, gSelect, { passId: K.RS_PASS_TEMPORAL, round: 0 });
    if (!hist) return units;
    units.push(...refreshInvUnits(k, t));
    indirect('rs_t_inverse', 'rs_t_inverse', inverse, gInverse, K.RS_Q_INV, K.RS_PASS_T_INV);
    perPixel('rs_t_select_b', 'rs_t_select', select, gSelect, { passId: K.RS_PASS_TEMPORAL, round: 1, flags: K.RSD_PHASE_B });
    return units;
  }

  destroy(): void { }
}
