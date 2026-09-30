// M5 per-frame state of the ReSTIR kernel (restir-temporal-api.md §2.3–§2.5, §2.10, §3.8, TD4, TD19, TD20). OWNER T-A.
// FrameStateTracker.advance(kernel, state) runs once per rendered frame (RestirKernel.advance): one light commit, the
// env record of frame t−1 (verbatim packed words), prevCam, the config hash and the history-validity decision, and
// the RsTemporal uniform contents.
// P0 STUB (§1.4): lights are committed once per frame, RsTemporal is written with TF_HIST_VALID = 0 (every frame is a
// reset frame). A1 replaces the body (change bits, env diff, config hash, generations).
import type { LightData } from '../../scene/types.ts';
import type { EnvParamsCpu } from '../env-gpu.ts';
import type { CameraState } from '../frame-uniforms.ts';
import type { LightsCommit } from '../lights-gpu.ts';
import { RS_WGSL_CONSTS as K, type RsTemporalCpu } from './layout.ts';
import type { RestirKernel } from './kernel.ts';

/** State of one frame of a validation run / chain (§3.8). */
export interface RestirFrameState {
  /** Animation frame counted from the chain's reset (RsDispatch.t of the frame). */
  t: number;
  camera: CameraState;
  lights: LightData[];
  /** Env parameters (rotation, strength, tint, visibleToCamera) and the env map id (a map change is a reset). */
  env?: { params: EnvParamsCpu; mapId: string };
  /** Force a history reset on this frame. */
  reset?: boolean;
}
export interface RestirInteractiveAdvance { lights?: LightData[]; env?: { params: EnvParamsCpu; mapId: string }; reset: boolean }
export interface RestirAdvance {
  histValid: boolean;
  /** RsTemporal.flags (TF_*). */
  flags: number;
  /** Why history is invalid (empty when valid). */
  reasons: string[];
  commit: LightsCommit;
  configHash: number;
  /** The RsTemporal uniform written for this frame. */
  temporal: RsTemporalCpu;
}

const NO_COMMIT: LightsCommit = { same: true, pmfChanged: false, anyMoved: false, anyRadio: false, added: [], removed: [], reallocated: false };

export class FrameStateTracker {
  private frameGen = 0;
  private pendingReasons: string[] = ['first-frame'];

  /** The next advance() is a reset (allocation, scene or settings change outside advance). */
  invalidate(reason: string): void { this.pendingReasons.push(reason); }

  advance(k: RestirKernel, state: RestirFrameState): RestirAdvance {
    const commit = this.commitLights(k, state.lights);
    const last = k.frame.last;
    if (!last) throw new Error('FrameStateTracker.advance: frame uniforms not written (setView first)');
    k.frame.write({ ...last, camera: state.camera, prevCamera: state.camera });
    return this.finish(k, commit, [...this.pendingReasons.splice(0), 'p0-stub']);
  }

  advanceInteractive(k: RestirKernel, o: RestirInteractiveAdvance): RestirAdvance {
    const commit = o.lights ? this.commitLights(k, o.lights) : NO_COMMIT;
    return this.finish(k, commit, [...this.pendingReasons.splice(0), 'p0-stub']);
  }

  private commitLights(k: RestirKernel, lights: LightData[]): LightsCommit {
    const u = k.lights.update(lights);
    return { same: !u.lightsChanged && !u.pmfChanged, pmfChanged: u.pmfChanged, anyMoved: u.lightsChanged, anyRadio: u.lightsChanged, added: [], removed: [], reallocated: u.reallocated };
  }

  private finish(k: RestirKernel, commit: LightsCommit, reasons: string[]): RestirAdvance {
    const gen = ++this.frameGen;
    const flags = K.TF_RESET;
    const env = new Uint32Array(k.envParamsWords());
    const temporal: RsTemporalCpu = { flags, histFrames: 0, frameGen: gen, prevGen: gen, envPrev: env, gens: [gen, gen, gen, gen], gensPrev: [gen, gen, gen, gen], configHash: 0 };
    return { histValid: false, flags, reasons, commit, configHash: 0, temporal };
  }
}
