// Timeline clock (plan §5 M3a: "interactive mode runs on wall-clock time; validation uses frame/fps").
//   interactive  t advances by wall-clock seconds while playing (loops per Animation.loop)
//   validation   an integer frame counter advances by exactly 1 per rendered (advanced) app frame; t = frame / fps,
//                the same definition exportFrames() uses, so our frame k and Cycles' frame k resolve identically.
import { frameTime, type Animation } from '../../core/scene/animation.ts';

export type TimeMode = 'interactive' | 'validation';

export class TimelinePlayer {
  playing = false;
  mode: TimeMode = 'interactive';
  private t = 0;
  private f = 0;
  /** Set whenever the evaluated time changed (consumed by the editor, which then applies the animation). */
  dirty = true;
  readonly listeners = new Set<() => void>();

  constructor(readonly anim: Animation) {}

  get frame(): number { return this.f; }
  get lastFrame(): number { return Math.round(this.anim.duration * this.anim.fps); }

  /** Time at which the animation is evaluated. */
  get time(): number { return this.mode === 'validation' ? frameTime(this.f, this.anim.fps) : this.t; }

  setMode(m: TimeMode): void {
    if (m === this.mode) return;
    if (m === 'validation') this.f = Math.max(0, Math.round(this.t * this.anim.fps));
    else this.t = frameTime(this.f, this.anim.fps);
    this.mode = m;
    this.touch();
  }

  play(): void { this.playing = true; this.touch(); }
  pause(): void { this.playing = false; this.touch(); }
  toggle(): void { if (this.playing) this.pause(); else this.play(); }

  seek(t: number): void {
    const T = this.anim.duration;
    const c = Math.max(0, Math.min(T, t));
    if (this.mode === 'validation') this.f = Math.round(c * this.anim.fps);
    else this.t = c;
    this.touch();
  }

  seekFrame(k: number): void {
    this.f = Math.max(0, Math.min(this.lastFrame, Math.round(k)));
    this.t = frameTime(this.f, this.anim.fps);
    this.touch();
  }

  /** Advance the clock for one app frame. Returns true when the evaluated time changed. */
  tick(wallDt: number, advanced: boolean): boolean {
    if (!this.playing) return false;
    const T = this.anim.duration;
    if (this.mode === 'validation') {
      if (!advanced) return false;
      let k = this.f + 1;
      if (k > this.lastFrame) { if (this.anim.loop) k = 0; else { k = this.lastFrame; this.playing = false; } }
      if (k === this.f) return false;
      this.f = k;
    } else {
      if (!(wallDt > 0)) return false;
      let t = this.t + Math.min(wallDt, 1);
      if (t > T) { if (this.anim.loop && T > 0) t %= T; else { t = T; this.playing = false; } }
      this.t = t;
    }
    this.touch(false);
    return true;
  }

  private touch(notify = true): void {
    this.dirty = true;
    if (notify) for (const cb of this.listeners) cb();
  }
}
