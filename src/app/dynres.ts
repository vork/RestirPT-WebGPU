// M8 dynamic resolution (PLAN §1.10: "a dynamic-resolution controller targeting 33 ms"; docs/decisions/m8-perf.md §9).
// Discrete scale levels of the internal-resolution preset's height. The input is the GPU time of each frame (measured
// from queue.onSubmittedWorkDone(): busy interval = done − max(submit, previous done)); an EMA of it drives the level
// with hysteresis: down one level when the EMA exceeds target·(1 + downMargin), up one level when the time PREDICTED at
// the next level (∝ pixel count) stays below target·(1 − upMargin); at most one change per `minFrames` frames and
// never on the frames right after a change (they include reallocation and history resets). A level change resizes the
// internal targets (the ReSTIR and denoiser histories reset once, as on any resize).

export const DYNRES_LEVELS = [1, 0.875, 0.75, 0.625, 0.5] as const;

export interface DynResOptions {
  /** Target GPU frame time (ms). */
  targetMs: number;
  /** EMA weight of a new sample. */
  alpha: number;
  /** Frames between decisions (and ignored after a change). */
  minFrames: number;
  downMargin: number;
  upMargin: number;
}

export const DYNRES_DEFAULTS: DynResOptions = { targetMs: 33, alpha: 0.15, minFrames: 30, downMargin: 0.08, upMargin: 0.1 };

export class DynamicResolution {
  readonly o: DynResOptions;
  level = 0;
  ema = 0;
  private samples = 0;
  private sinceChange = 0;

  constructor(o: Partial<DynResOptions> = {}) { this.o = { ...DYNRES_DEFAULTS, ...o }; }

  get scale(): number { return DYNRES_LEVELS[this.level]; }

  reset(): void { this.level = 0; this.ema = 0; this.samples = 0; this.sinceChange = 0; }

  /** Feed one frame's GPU time (ms). Returns true when the level changed (the caller resizes). */
  update(gpuMs: number): boolean {
    if (!(gpuMs > 0) || !Number.isFinite(gpuMs)) return false;
    this.sinceChange++;
    if (this.sinceChange <= 3) return false;              // the frames right after a change: reallocation, resets
    this.ema = this.samples === 0 ? gpuMs : this.ema + this.o.alpha * (gpuMs - this.ema);
    this.samples++;
    if (this.sinceChange < this.o.minFrames) return false;
    const t = this.o.targetMs;
    if (this.ema > t * (1 + this.o.downMargin) && this.level < DYNRES_LEVELS.length - 1) return this.change(this.level + 1);
    if (this.level > 0) {
      const up = DYNRES_LEVELS[this.level - 1], cur = DYNRES_LEVELS[this.level];
      if (this.ema * (up * up) / (cur * cur) < t * (1 - this.o.upMargin)) return this.change(this.level - 1);
    }
    return false;
  }

  private change(level: number): boolean {
    this.level = level;
    this.sinceChange = 0;
    this.samples = 0;
    return true;
  }
}

/** GPU busy time of a submitted frame from completion times: done − max(submit, previous done). */
export function gpuBusyMs(submitMs: number, doneMs: number, prevDoneMs: number | undefined): number {
  return doneMs - Math.max(submitMs, prevDoneMs ?? -Infinity);
}
