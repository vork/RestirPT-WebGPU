// Suffix-refresh unit builders (restir-temporal-api.md §3.5, §4.1, TD9). OWNER T-C. Called by stage-temporal.ts on
// TF_REFRESH frames: rs_refresh_fwd (per pixel over the history, before T1) and rs_args(q2) + rs_refresh_inv (Q_i, before
// T4).
// P0 STUB: the builders return [] (the WGSL refresh stub marks every record undefined on TF_REFRESH frames).
import type { RestirKernel, WorkUnit } from './kernel.ts';

export function refreshFwdUnits(_k: RestirKernel, _t: number): WorkUnit[] { return []; }
export function refreshInvUnits(_k: RestirKernel, _t: number): WorkUnit[] { return []; }
