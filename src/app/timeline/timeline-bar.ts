// Timeline bar (DOM, bottom centre): play/pause, scrubber with key markers, time/frame readout and mode badge.
// Scrubbing pauses playback and seeks the TimelinePlayer; the editor applies the animation on the next frame.
import type { TargetId } from '../../core/scene/animation.ts';
import type { TimelinePlayer } from './player.ts';

export class TimelineBar {
  readonly root: HTMLDivElement;
  private readonly play: HTMLButtonElement;
  private readonly range: HTMLInputElement;
  private readonly marks: HTMLDivElement;
  private readonly readout: HTMLSpanElement;
  private keyTargets: () => TargetId[] = () => [];
  private marksKey = '';

  constructor(readonly player: TimelinePlayer, parent: HTMLElement = document.body) {
    this.root = document.createElement('div');
    this.root.className = 'timeline-bar';
    Object.assign(this.root.style, {
      position: 'fixed', left: '50%', bottom: '8px', transform: 'translateX(-50%)', width: 'min(640px, calc(100% - 360px))', minWidth: '280px',
      zIndex: '3', display: 'flex', alignItems: 'center', gap: '8px', padding: '4px 8px', borderRadius: '6px',
      background: 'rgba(20,20,20,0.85)', font: '11px ui-monospace, Menlo, monospace', color: '#ddd',
    } satisfies Partial<CSSStyleDeclaration>);
    this.play = document.createElement('button');
    this.play.type = 'button';
    this.play.title = 'Play / pause (Space)';
    Object.assign(this.play.style, { width: '28px', height: '22px', cursor: 'pointer', background: '#333', color: '#eee', border: '1px solid #555', borderRadius: '3px' });
    this.play.addEventListener('click', () => { player.toggle(); this.refresh(); });
    const track = document.createElement('div');
    Object.assign(track.style, { position: 'relative', flex: '1', height: '22px' });
    this.range = document.createElement('input');
    this.range.type = 'range';
    this.range.min = '0';
    this.range.step = 'any';
    this.range.setAttribute('aria-label', 'timeline');
    Object.assign(this.range.style, { position: 'absolute', inset: '0', width: '100%', margin: '0' });
    this.range.addEventListener('input', () => {
      player.pause();
      const t = Number(this.range.value);
      if (player.mode === 'validation') player.seekFrame(Math.round(t * player.anim.fps)); else player.seek(t);
      this.refresh();
    });
    this.marks = document.createElement('div');
    Object.assign(this.marks.style, { position: 'absolute', left: '8px', right: '8px', top: '0', height: '5px', pointerEvents: 'none' });
    track.append(this.range, this.marks);
    this.readout = document.createElement('span');
    Object.assign(this.readout.style, { minWidth: '150px', textAlign: 'right', whiteSpace: 'pre' });
    this.root.append(this.play, track, this.readout);
    parent.append(this.root);
    player.listeners.add(() => this.refresh());
    player.anim.onChange(() => this.refresh());
    this.refresh();
  }

  /** Which targets' keys to mark (e.g. the selected light + camera + env). */
  setKeyTargets(fn: () => TargetId[]): void { this.keyTargets = fn; this.refresh(); }

  refresh(): void {
    const p = this.player;
    const a = p.anim;
    this.play.textContent = p.playing ? '❚❚' : '▶';
    this.range.max = String(a.duration);
    this.range.value = String(Math.min(a.duration, p.time));
    const frame = p.mode === 'validation' ? p.frame : Math.round(p.time * a.fps);
    this.readout.textContent = `${p.time.toFixed(2)}/${a.duration.toFixed(1)} s  f${frame}${p.mode === 'validation' ? ' [val]' : ''}`;
    const times = new Set<number>();
    for (const t of this.keyTargets()) for (const k of a.keyTimes(t)) times.add(k);
    const mk = `${a.duration}|${[...times].join(',')}`;
    if (mk === this.marksKey) return;
    this.marksKey = mk;
    this.marks.replaceChildren(...[...times].map((t) => {
      const m = document.createElement('div');
      Object.assign(m.style, { position: 'absolute', left: `${(100 * t) / Math.max(1e-9, a.duration)}%`, width: '3px', marginLeft: '-1px', height: '5px', background: '#fc4' });
      return m;
    }));
  }
}
