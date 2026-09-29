// Paired spatial reuse stage: pair_accept → args → spatial_replay → spatial_shift → spatial_resample per round
// (restir-api.md §4.1, §3.8–§3.9). OWNER WP-C. P0 STUB (restir-api.md §1.4): frameUnits returns [] — the kernel then
// finalizes with F·W of the initial reservoir (canonical-only output: unbiased).
// Contract for the real stage: prepare(k) compiles its pipelines through k.pipeline(name); frameUnits(k, t) returns
// one unit per round (or per stage × row band); round r reads res[r % 2] and writes res[(r + 1) % 2]
// (k.resources.g2(name, r % 2)); the final round (RSD_FINAL_ROUND) writes rsShade. The kernel finalizes with rsShade
// whenever this returns a non-empty list.
import type { RestirKernel, RestirStage, WorkUnit } from './kernel.ts';

export class SpatialStage implements RestirStage {
  async prepare(_k: RestirKernel): Promise<void> { }
  frameUnits(_k: RestirKernel, _t: number): WorkUnit[] { return []; }
  destroy(): void { }
}
