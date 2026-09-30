// Temporal stage (restir-temporal-api.md §3.6, §4.1, §4.4; PLAN §3 steps 2–4). OWNER T-B.
// Per frame with temporal on (after rs_initial, before spatial): clear q1/q2 → [rs_refresh_fwd] → rs_t_classify (T1) →
// rs_args(q1) + rs_t_forward (T2) → rs_t_select phase A → [rs_args(q2) + rs_refresh_inv] → rs_args(q2) + rs_t_inverse
// (T4) → rs_t_select phase B. Reservoir roles from the kernel: res[h] = k.historyIndex() (read-only), res[w] =
// k.resBase() (written in place). Unit labels start with the pass kind (E1).
// P0 STUB (§1.4): frameUnits returns [] (temporal off, w = 0, bitwise M4); prepare compiles every temporal pipeline.
import type { RestirKernel, RestirStage, WorkUnit } from './kernel.ts';
import { TEMPORAL_PASSES } from './resources.ts';

export class TemporalStage implements RestirStage {
  async prepare(k: RestirKernel): Promise<void> {
    await Promise.all([...TEMPORAL_PASSES, 'rs_args' as const].map((n) => k.pipeline(n)));
  }

  frameUnits(_k: RestirKernel, _t: number): WorkUnit[] {
    return [];
  }

  destroy(): void { }
}
