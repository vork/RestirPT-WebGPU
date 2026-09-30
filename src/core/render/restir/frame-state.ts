// M5 per-frame state of the ReSTIR kernel (restir-temporal-api.md §2.3–§2.5, §2.10, §3.8, TD4, TD19, TD20). OWNER T-A.
// FrameStateTracker.advance(kernel, state) runs once per rendered frame (RestirKernel.advance):
//   1. env: the frame's env parameters are written (writeEnvParams) and staged into the light state (Φ_env enters the
//      pmf); a new env map id is a reset.
//   2. lights: exactly ONE light commit (LightsGpu.commit, no flip when nothing changed ⇒ TF_LIGHTS_SAME).
//   3. camera: frame.cam = the frame's camera, frame.prevCam = the previous frame's (= cam after a reset).
//   4. env record: RsTemporal.envPrev = the verbatim packed EnvParams words of frame t−1 (= t after a reset);
//      TF_ENV_MOVED ⇔ cg/sg words differ, TF_ENV_RADIO ⇔ strength/tint words differ (§2.5).
//   5. config hash (§2.10): any change of the hashed configuration, state.reset, a reallocation, a previous frame that
//      was not advanced, or the first frame ⇒ TF_HIST_VALID = 0 and TF_RESET for exactly this frame.
//   6. RsTemporal: flags, histFrames, frameGen / prevGen, generations (camGen, vbufGen, lightGen, pmfGen) of t and t−1.
// The pure parts (configHash, envDiff, temporalFlags) are exported for the CPU tests (U-TL-2, U-TL-3).
import type { LightData } from '../../scene/types.ts';
import { writeEnvParams, type EnvParamsCpu } from '../env-gpu.ts';
import { camerasEqual, type CameraState } from '../frame-uniforms.ts';
import type { LightsCommit } from '../lights-gpu.ts';
import { RS_WGSL_CONSTS as K, type RsTemporalCpu } from './layout.ts';
import type { RestirKernel } from './kernel.ts';

/** State of one frame of a validation run / chain (§3.8). */
export interface RestirFrameState {
  /** Animation frame counted from the chain's reset (RsDispatch.t of the frame). */
  t: number;
  camera: CameraState;
  /** The enabled analytic lights of the frame (disabled / removed lights are absent). */
  lights: LightData[];
  /** Env parameters (rotation, strength, tint, visibleToCamera) and the env map id (a map change is a reset; the map
   *  itself is swapped with RestirKernel.setEnvironment before advance). */
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

// ------------------------------------------------------------------------------------------------ pure helpers

/** Every input of the config hash (§2.10). Anything that changes here resets history for one frame. */
export interface ConfigHashInput {
  /** Scene generation (SceneGpu identity / package hash): geometry, materials, BSDF tier, textures. */
  sceneGen: number | string;
  atlas: [number, number]; member: [number, number]; members: number; memberBase: number;
  lightMode: string;
  /** LightsGpu.version: bumps on every records reallocation (LightsCommit.reallocated). */
  lightsLayout: number;
  /** Env map generation (bumps on setEnvironment) and the package map id. */
  envMapGen: number; envMapId: string;
  importanceKey: string; envNee: boolean;
  /** The ReSTIR settings (maxBounces, criteria, τ, α_min, cCap, temporal modes, plants, wScale, slots, …). */
  settings: unknown;
  /** RestirParams.flags and tMode / tPlants words (plant bits, RR, criteria, temporal). */
  flags: number; tMode: number; tPlants: number;
  jitterMode: number;
  /** M(B) is a constant 1 in M5 (lights/measure.wgsl mis_M); hashed so a future change resets. */
  misM: number;
}

const stable = (v: unknown): string => JSON.stringify(v, (_k, x) => {
  if (ArrayBuffer.isView(x)) return Array.from(x as unknown as ArrayLike<number>);
  if (x && typeof x === 'object' && !Array.isArray(x)) return Object.fromEntries(Object.entries(x as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : 1)));
  return x;
});

/** FNV-1a (32 bit) over the UTF-16 code units of the stable JSON of the input. */
export function configHash(i: ConfigHashInput): number {
  const s = stable(i);
  let h = 0x811c9dc5;
  for (let k = 0; k < s.length; k++) {
    const c = s.charCodeAt(k);
    h ^= c & 0xff; h = Math.imul(h, 0x01000193) >>> 0;
    h ^= c >>> 8; h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** Env record diff on the packed EnvParams words (§2.5): moved ⇔ cg/sg words, radio ⇔ strength/tint words. */
export function envDiff(prev: Uint32Array, cur: Uint32Array): { same: boolean; moved: boolean; radio: boolean } {
  const d = (w: number) => prev[w] !== cur[w];
  const moved = d(0) || d(1);
  const radio = d(2) || d(4) || d(5) || d(6);
  let same = true;
  for (let w = 0; w < 8; w++) if (d(w)) same = false;
  return { same, moved, radio };
}

/** RsTemporal.flags of a frame (§2.1). */
export function temporalFlags(o: { histValid: boolean; commit: LightsCommit; env: { same: boolean; moved: boolean; radio: boolean }; camSame: boolean }): number {
  let f = 0;
  if (o.histValid) f |= K.TF_HIST_VALID; else f |= K.TF_RESET;
  if (o.commit.same) f |= K.TF_LIGHTS_SAME;
  if (o.env.same) f |= K.TF_ENV_SAME;
  if (o.commit.pmfChanged) f |= K.TF_PMF_CHANGED;
  if (o.env.moved) f |= K.TF_ENV_MOVED;
  if (o.env.radio) f |= K.TF_ENV_RADIO;
  if (o.commit.anyMoved) f |= K.TF_LIGHT_MOVED;
  if (o.camSame) f |= K.TF_CAM_SAME;
  // Refresh scope per frame (math §24): any light, pmf or env-light change between t−1 and t.
  if (o.histValid && (!o.commit.same || o.env.moved || o.env.radio)) f |= K.TF_REFRESH;
  return f >>> 0;
}

// ------------------------------------------------------------------------------------------------ tracker

const NO_COMMIT: LightsCommit = { same: true, pmfChanged: false, anyMoved: false, anyRadio: false, added: [], removed: [], reallocated: false };
type Gens = [number, number, number, number];

export class FrameStateTracker {
  private frameGen = 0;
  private pendingReasons: string[] = ['first-frame'];
  private prevEnv: Uint32Array | undefined;
  private prevCamera: CameraState | undefined;
  private prevHash: number | undefined;
  private prevMapId: string | undefined;
  private histFrames = 0;
  private gens: Gens = [0, 0, 0, 0];

  /** The next advance() is a reset (allocation, view, env map or scene change outside advance). */
  invalidate(reason: string): void { this.pendingReasons.push(reason); }

  advance(k: RestirKernel, state: RestirFrameState): RestirAdvance {
    const last = k.frame.last;
    if (!last) throw new Error('FrameStateTracker.advance: frame uniforms not written (setView first)');
    const reasons = this.takeReasons(k, state.reset);
    if (state.env) this.applyEnv(k, state.env, reasons);
    k.lights.update(state.lights);                      // staged (LightsGpu.deferred) …
    const commit = k.lights.commit();                    // … and committed exactly once (TD4)
    const cam = state.camera;
    const camSame = !!this.prevCamera && camerasEqual(this.prevCamera, cam);
    const r = this.finish(k, commit, reasons, camSame, last.jitterMode);
    k.frame.write({ ...last, camera: cam, prevCamera: r.histValid && this.prevCamera ? this.prevCamera : cam });
    this.prevCamera = { camToWorld: Array.from(cam.camToWorld), yfov: cam.yfov, znear: cam.znear };
    return r;
  }

  advanceInteractive(k: RestirKernel, o: RestirInteractiveAdvance): RestirAdvance {
    const reasons = this.takeReasons(k, o.reset);
    if (o.env) this.applyEnv(k, o.env, reasons);
    if (o.lights) k.lights.update(o.lights);
    const commit = k.lights.hasPending ? k.lights.commit() : NO_COMMIT;
    // The renderer owns the camera (prevCam in its frame uniforms): camGen bumps every frame.
    return this.finish(k, commit, reasons, false, -1);
  }

  private takeReasons(k: RestirKernel, reset?: boolean): string[] {
    const reasons = this.pendingReasons.splice(0);
    if (reset) reasons.push('state.reset');
    if (!k.previousFrameAdvanced && !reasons.length) reasons.push('previous-frame-not-advanced');
    return reasons;
  }

  private applyEnv(k: RestirKernel, env: { params: EnvParamsCpu; mapId: string }, reasons: string[]): void {
    if (this.prevMapId !== undefined && env.mapId !== this.prevMapId) reasons.push('env-map');
    this.prevMapId = env.mapId;
    writeEnvParams(k.device, k.envResources, env.params);
    k.envParamsChanged();                                // Φ_env (strength, tint) staged into the light state
  }

  private finish(k: RestirKernel, commit: LightsCommit, reasons: string[], camSame: boolean, jitterMode: number): RestirAdvance {
    const hash = configHash(k.configHashInput(jitterMode, this.prevMapId ?? ''));
    if (this.prevHash !== undefined && hash !== this.prevHash) reasons.push('config');
    if (commit.reallocated) reasons.push('lights-reallocated');
    this.prevHash = hash;
    const histValid = reasons.length === 0;
    const curEnv = new Uint32Array(k.envParamsWords());
    const prevEnv = histValid && this.prevEnv ? this.prevEnv : curEnv;
    const env = histValid ? envDiff(prevEnv, curEnv) : { same: true, moved: false, radio: false };
    const flags = temporalFlags({ histValid, commit, env, camSame: histValid && camSame });
    const gensPrev: Gens = [...this.gens];
    const gen = ++this.frameGen;
    if (!(histValid && camSame)) this.gens[0]++;
    this.gens[1]++;                                      // a new V-buffer every frame
    if (!(histValid && commit.same)) this.gens[2]++;
    if (!histValid || commit.pmfChanged) this.gens[3]++;
    this.histFrames = histValid ? this.histFrames + 1 : 0;
    const temporal: RsTemporalCpu = {
      flags, histFrames: this.histFrames, frameGen: gen, prevGen: gen - 1, envPrev: prevEnv.slice(), gens: [...this.gens], gensPrev, configHash: hash,
    };
    this.prevEnv = curEnv;
    return { histValid, flags, reasons, commit, configHash: hash, temporal };
  }
}
