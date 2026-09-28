// Reference compare view (plan §6 M3a "Compare: Cycles EXR split/flip; relative error; t-map", §5 M3a
// "auto-loaded in the compare view"): a 2D canvas over the render showing our current accumulated image against a
// Cycles reference (mean of K seeds):
//   split   ours left of the divider, Cycles right (drag the divider handle)
//   flip    alternates ours / Cycles every `flipMs`
//   relerr  |Y_ours − Y_ref| / (Y_ref + ε), sequential map on [0, relErrMax]
//   tmap    per-tile Welch t (ours batches vs Cycles seeds), diverging map on [−4, 4]
//   ours / ref  one side only
// "Ours" is re-read from the app's color target every `liveMs` while visible; "capture batch" stores the current
// image as an independent replicate for the t-map and restarts accumulation.
import type { App } from '../app.ts';
import {
  diverging, heat, meanImage, relativeError, resampleNearest, sameSize, summarize, tileTMap, toneMap, type CompareSummary, type RgbaImage,
} from './compare-math.ts';
import { readColorTexture } from './images.ts';

export type CompareMode = 'off' | 'split' | 'flip' | 'relerr' | 'tmap' | 'ours' | 'ref';

export interface CompareSettings {
  mode: CompareMode;
  split: number;
  exposureEV: number;
  relErrMax: number;
  tile: number;
  flipMs: number;
  live: boolean;
}

export class CompareView {
  readonly settings: CompareSettings = { mode: 'off', split: 0.5, exposureEV: 0, relErrMax: 0.5, tile: 16, flipMs: 600, live: true };
  ref: RgbaImage[] = [];
  refLabel = '';
  ours: RgbaImage | undefined;
  batches: RgbaImage[] = [];
  summary: CompareSummary | undefined;
  status = '';
  readonly listeners = new Set<() => void>();
  private readonly root: HTMLDivElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly handle: HTMLDivElement;
  private readonly label: HTMLDivElement;
  private timer: number | undefined;
  private reading = false;
  private lastDraw = '';

  constructor(readonly app: App, parent: HTMLElement = document.body) {
    this.root = document.createElement('div');
    this.root.className = 'compare-view';
    Object.assign(this.root.style, { position: 'fixed', inset: '0', zIndex: '1', pointerEvents: 'none', display: 'none' } satisfies Partial<CSSStyleDeclaration>);
    this.canvas = document.createElement('canvas');
    Object.assign(this.canvas.style, { position: 'absolute', inset: '0', width: '100%', height: '100%', objectFit: 'contain', imageRendering: 'pixelated' } satisfies Partial<CSSStyleDeclaration>);
    this.handle = document.createElement('div');
    Object.assign(this.handle.style, {
      position: 'absolute', top: '0', bottom: '0', width: '12px', marginLeft: '-6px', cursor: 'ew-resize', pointerEvents: 'auto',
      background: 'linear-gradient(90deg, transparent 5px, rgba(255,210,60,0.9) 5px, rgba(255,210,60,0.9) 7px, transparent 7px)',
    } satisfies Partial<CSSStyleDeclaration>);
    this.label = document.createElement('div');
    Object.assign(this.label.style, {
      position: 'absolute', left: '50%', bottom: '8px', transform: 'translateX(-50%)', font: '11px ui-monospace, Menlo, monospace',
      background: 'rgba(0,0,0,0.65)', padding: '3px 8px', borderRadius: '4px', whiteSpace: 'pre', color: '#eee',
    } satisfies Partial<CSSStyleDeclaration>);
    this.root.append(this.canvas, this.handle, this.label);
    parent.append(this.root);
    this.installDivider();
  }

  private changed(): void { for (const cb of this.listeners) cb(); }

  private installDivider(): void {
    let dragging = false;
    const move = (e: PointerEvent) => {
      if (!dragging) return;
      const r = this.displayRect();
      this.settings.split = Math.max(0, Math.min(1, (e.clientX - r.left) / Math.max(1, r.width)));
      this.draw(true);
      this.changed();
    };
    this.handle.addEventListener('pointerdown', (e) => { dragging = true; this.handle.setPointerCapture(e.pointerId); e.preventDefault(); e.stopPropagation(); });
    this.handle.addEventListener('pointermove', move);
    this.handle.addEventListener('pointerup', (e) => { dragging = false; if (this.handle.hasPointerCapture(e.pointerId)) this.handle.releasePointerCapture(e.pointerId); });
  }

  /** On-screen rectangle of the (object-fit: contain) image. */
  private displayRect(): { left: number; top: number; width: number; height: number } {
    const r = this.root.getBoundingClientRect();
    const iw = this.canvas.width || 1, ih = this.canvas.height || 1;
    const s = Math.min(r.width / iw, r.height / ih);
    const w = iw * s, h = ih * s;
    return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2, width: w, height: h };
  }

  setReference(imgs: RgbaImage[], label: string): void {
    if (!imgs.length) throw new Error('no reference images');
    for (const im of imgs) if (!sameSize(im, imgs[0])) throw new Error('reference seeds differ in size');
    this.ref = imgs;
    this.refLabel = label;
    this.batches = [];
    this.summary = undefined;
    this.status = `reference: ${label} (${imgs.length} seed${imgs.length > 1 ? 's' : ''}, ${imgs[0].width}x${imgs[0].height})`;
    if (this.settings.mode === 'off') this.settings.mode = 'split';
    this.apply();
    this.changed();
  }

  setMode(m: CompareMode): void { this.settings.mode = m; this.apply(); this.changed(); }

  /** Start/stop the live refresh and show/hide the overlay after a settings change. */
  apply(): void {
    const on = this.settings.mode !== 'off';
    this.root.style.display = on ? 'block' : 'none';
    this.handle.style.display = on && this.settings.mode === 'split' ? 'block' : 'none';
    if (on && this.timer === undefined) {
      const tick = () => {
        this.timer = window.setTimeout(tick, Math.min(250, this.settings.flipMs));
        if (this.settings.live && !this.app.suspended) void this.refreshOurs();
        this.draw();
      };
      tick();
    } else if (!on && this.timer !== undefined) { clearTimeout(this.timer); this.timer = undefined; }
    this.draw(true);
  }

  /** Read our current accumulated image (async GPU readback). */
  async refreshOurs(): Promise<RgbaImage | undefined> {
    if (this.reading || !this.app.targets) return this.ours;
    this.reading = true;
    try {
      this.ours = await readColorTexture(this.app.device, this.app.targets.color);
      this.draw(true);
      return this.ours;
    } finally { this.reading = false; }
  }

  /** Store the current image as an independent replicate (t-map) and restart accumulation. */
  async captureBatch(): Promise<number> {
    const im = await this.refreshOurs();
    if (!im) return this.batches.length;
    this.batches.push({ width: im.width, height: im.height, data: im.data.slice() });
    this.app.resetHistory();
    this.computeSummary();
    this.changed();
    return this.batches.length;
  }

  clearBatches(): void { this.batches = []; this.computeSummary(); this.changed(); }

  /** Our replicates for statistics: captured batches, or the live image alone. */
  oursReplicates(): RgbaImage[] { return this.batches.length ? this.batches : this.ours ? [this.ours] : []; }

  computeSummary(): CompareSummary | undefined {
    const o = this.oursReplicates();
    if (!o.length || !this.ref.length || !sameSize(o[0], this.ref[0])) { this.summary = undefined; return undefined; }
    this.summary = summarize(o, this.ref, this.settings.tile);
    return this.summary;
  }

  summaryText(): string {
    const s = this.computeSummary();
    if (!this.ref.length) return 'no reference loaded';
    const o = this.oursReplicates()[0];
    if (!o) return 'waiting for our image';
    if (!sameSize(o, this.ref[0])) return `size mismatch: ours ${o.width}x${o.height}, Cycles ${this.ref[0].width}x${this.ref[0].height} (statistics disabled)`;
    if (!s) return '';
    const pct = (x: number) => `${(100 * x).toFixed(2)}%`;
    return `Y ours/ref − 1: ${pct(s.relDiff[3])} (R ${pct(s.relDiff[0])} G ${pct(s.relDiff[1])} B ${pct(s.relDiff[2])})\n` +
      `global t ${s.globalT.t.toFixed(2)} (df ${Number.isFinite(s.globalT.df) ? s.globalT.df.toFixed(1) : '∞'})  mean rel.err ${s.meanRelErr.toFixed(3)}\n` +
      `tiles |t|>3.3: ${Number.isNaN(s.tilesOver) ? 'n/a' : pct(s.tilesOver)}  (${this.oursReplicates().length} ours × ${this.ref.length} seeds)`;
  }

  /** Render the overlay canvas for the current mode. */
  draw(force = false): void {
    const mode = this.settings.mode;
    if (mode === 'off') return;
    const ref = this.ref.length ? meanImageCached(this.ref) : undefined;
    const ours = this.ours;
    const base = ref ?? ours;
    if (!base) { this.label.textContent = 'compare: no images yet'; return; }
    const phase = mode === 'flip' ? Math.floor(performance.now() / this.settings.flipMs) % 2 : 0;
    const key = `${mode}|${phase}|${this.settings.split}|${this.settings.exposureEV}|${this.settings.relErrMax}|${this.settings.tile}|${this.ours?.data.byteOffset}|${this.oursStamp}|${this.ref.length}|${this.batches.length}`;
    if (!force && key === this.lastDraw) return;
    this.lastDraw = key;
    const W = base.width, H = base.height;
    if (this.canvas.width !== W || this.canvas.height !== H) { this.canvas.width = W; this.canvas.height = H; }
    const g = this.canvas.getContext('2d')!;
    const img = g.createImageData(W, H);
    const o = ours ? resampleNearest(ours, W, H) : undefined;
    const r = ref;
    let text = '';
    if (mode === 'split' || mode === 'ours' || mode === 'ref' || mode === 'flip') {
      const showOursAll = mode === 'ours' || (mode === 'flip' && phase === 0);
      const showRefAll = mode === 'ref' || (mode === 'flip' && phase === 1);
      const to = o ? toneMap(o, this.settings.exposureEV) : undefined;
      const tr = r ? toneMap(r, this.settings.exposureEV) : undefined;
      const cut = Math.round(this.settings.split * W);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const useOurs = showOursAll || (!showRefAll && x < cut);
        const src = useOurs ? to ?? tr : tr ?? to;
        const i = (y * W + x) * 4;
        img.data.set(src!.subarray(i, i + 4), i);
      }
      text = mode === 'split' ? `ours  |  Cycles ${this.refLabel}` : mode === 'flip' ? (phase === 0 ? 'ours' : `Cycles ${this.refLabel}`) : mode === 'ours' ? 'ours' : `Cycles ${this.refLabel}`;
      const r0 = this.displayRect();
      this.handle.style.left = `${r0.left + this.settings.split * r0.width}px`;
    } else if (mode === 'relerr') {
      if (!o || !r) { this.label.textContent = 'relative error needs both images'; return; }
      const e = relativeError(o, r);
      for (let i = 0; i < e.length; i++) { const c = heat(e[i], this.settings.relErrMax); img.data.set([c[0], c[1], c[2], 255], 4 * i); }
      text = `|ΔY|/Y_ref  (0 … ${this.settings.relErrMax})`;
    } else if (mode === 'tmap') {
      const reps = this.batches.length ? this.batches : o ? [o] : [];
      if (!reps.length || !r || !sameSize(reps[0], r)) { this.label.textContent = 't-map needs equal-size images'; return; }
      const tm = tileTMap(reps, this.ref, this.settings.tile);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const k = Math.floor(y / tm.tile) * tm.tx + Math.floor(x / tm.tile);
        const c = tm.valid ? diverging(tm.t[k]) : [128, 128, 128];
        img.data.set([c[0], c[1], c[2], 255], (y * W + x) * 4);
      }
      text = `tile t (${tm.tile}px): ${tm.note}  [blue −4 … red +4: ours brighter]`;
    }
    g.putImageData(img, 0, 0);
    this.label.textContent = text;
  }

  private get oursStamp(): number { return this.ours ? this.ours.data.length + (this.ours.data[0] ?? 0) + (this.ours.data[this.ours.data.length >> 1] ?? 0) : 0; }

  dispose(): void { if (this.timer !== undefined) clearTimeout(this.timer); this.root.remove(); }
}

const meanCache = new WeakMap<RgbaImage[], RgbaImage>();
function meanImageCached(imgs: RgbaImage[]): RgbaImage {
  let m = meanCache.get(imgs);
  if (!m) { m = meanImage(imgs); meanCache.set(imgs, m); }
  return m;
}
