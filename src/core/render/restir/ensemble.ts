// Ensemble atlas statistics stage (restir-api.md §2.10, D14/D15). OWNER WP-C. P0 STUB: frameUnits returns [].
// Contract for the real stage: prepare(k) compiles rs_ensemble_stats through k.pipeline('rs_ensemble_stats');
// frameUnits(k, t) encodes the per-(member, 16² tile) reduction of rsFrame into k.resources.ensStats / ensPixel.
import type { RestirKernel, RestirStage, WorkUnit } from './kernel.ts';

export class EnsembleStage implements RestirStage {
  async prepare(_k: RestirKernel): Promise<void> { }
  frameUnits(_k: RestirKernel, _t: number): WorkUnit[] { return []; }
  destroy(): void { }
}
