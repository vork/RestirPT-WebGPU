// U-TQ-1 CPU part (restir-temporal-api.md §2.8, TD16, §6.1): the temporal queues Q_f (q1) and Q_i (q2) occupy disjoint
// item ranges [0, P) and [P, 2P) inside the M4 item region of NS_alloc = max(NS, 2) slots, never overlap the tState /
// sfxOut extension, and the 2D indirect args cover n ∈ {0, 1, 64, 65535·64, 65536·64 + 1}. The GPU part (every item
// processed once, overflow exactly when counter > P, rs_args queue selection) is in restir-temporal.gpu.test.ts.
// OWNER T-B.
import { describe, expect, it } from 'vitest';
import { RS_WGSL_CONSTS as K, arenaWords, nsAlloc, queueCapacityQ, queueItemBase } from '../../src/core/render/restir/layout.ts';
import { queueArgs } from '../../src/core/render/restir/stage-spatial.ts';

describe('U-TQ-1 CPU: temporal queue regions and args', () => {
  it('Q_f / Q_i regions are disjoint, inside the item region, before tState, for every NS and P', () => {
    for (const P of [1, 64, 256 * 256, 1024 * 1024]) for (let ns = 1; ns <= 6; ns++) {
      const NS = nsAlloc(ns, true);
      const w = arenaWords(P, NS);
      const itemsEnd = w.items + P * NS;
      expect(itemsEnd).toBe(w.tState);
      const f0 = w.items + queueItemBase(P, K.RS_Q_FWD), f1 = f0 + queueCapacityQ(P, NS, K.RS_Q_FWD);
      const i0 = w.items + queueItemBase(P, K.RS_Q_INV), i1 = i0 + queueCapacityQ(P, NS, K.RS_Q_INV);
      expect(f1).toBeLessThanOrEqual(i0);
      expect(i1).toBeLessThanOrEqual(itemsEnd);
      expect(queueCapacityQ(P, NS, K.RS_Q_SPATIAL)).toBe(P * NS);
      expect(queueItemBase(P, K.RS_Q_SPATIAL)).toBe(0);
    }
  });

  it('args for n ∈ {0, 1, 64, 65535·64, 65536·64 + 1}; the queue index field of RsDispatch.flags', () => {
    expect(queueArgs(0)).toEqual([0, 1, 1]);
    expect(queueArgs(1)).toEqual([1, 1, 1]);
    expect(queueArgs(64)).toEqual([1, 1, 1]);
    expect(queueArgs(65535 * 64)).toEqual([65535, 1, 1]);
    expect(queueArgs(65536 * 64 + 1)).toEqual([65535, 2, 1]);
    for (const q of [0, 1, 2]) expect(((q << K.RSD_QUEUE_SHIFT) >>> K.RSD_QUEUE_SHIFT) & 3).toBe(q);
    // no M4 RsDispatch flag bit falls into the queue field (bits 8–9)
    for (const b of [K.RSD_FIRST_CHUNK, K.RSD_FINAL_CHUNK, K.RSD_FINAL_ROUND, K.RSD_ACCUMULATE, K.RSD_ADVANCED, K.RSD_PHASE_B]) {
      expect((b >>> K.RSD_QUEUE_SHIFT) & 3).toBe(0);
    }
  });
});
