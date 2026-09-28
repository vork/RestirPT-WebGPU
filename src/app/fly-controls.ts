// DOM input for the fly camera: RMB-drag look (pointer capture), F = pointer-lock fly mode, WASD/E/Q by
// KeyboardEvent.code, wheel = speed, Shift/Ctrl modifiers, Home, bookmarks (Ctrl+1..9 save, 1..9 recall).
// LMB is reserved for picking/gizmos: it only emits 'pick' events.
import { keyToAction, type FlyCamera } from './fly-camera.ts';
import { loadPref, savePref } from './storage.ts';

export interface PickEvent {
  /** CSS pixels relative to the canvas. */
  x: number;
  y: number;
  /** Canvas CSS size at the time of the click. */
  width: number;
  height: number;
  alt: boolean;
  shift: boolean;
  ctrl: boolean;
  button: number;
}

export interface FlyPrefs { sensitivity: number; invertY: boolean }
const PREFS_KEY = 'fly-camera';

export function isTextInput(t: EventTarget | null): boolean {
  if (!(t instanceof HTMLElement)) return false;
  return t.isContentEditable || t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement;
}

export class FlyControls {
  private looking = false;
  private readonly abort = new AbortController();
  readonly onPick = new Set<(e: PickEvent) => void>();
  /** Called after Home/bookmark/file-camera jumps and while the camera moves (UI refresh). */
  readonly onChange = new Set<() => void>();
  /** Other app-level keys (pause, step, HUD...) not consumed by the camera. Return true if handled. */
  extraKeys: ((e: KeyboardEvent) => boolean) | undefined;

  constructor(private readonly canvas: HTMLCanvasElement, readonly camera: FlyCamera) {
    const prefs = loadPref<FlyPrefs>(PREFS_KEY, { sensitivity: camera.sensitivity, invertY: camera.invertY });
    if (Number.isFinite(prefs.sensitivity) && prefs.sensitivity > 0) camera.sensitivity = prefs.sensitivity;
    camera.invertY = !!prefs.invertY;
    const o = { signal: this.abort.signal };
    canvas.addEventListener('pointerdown', this.pointerDown, o);
    canvas.addEventListener('pointermove', this.pointerMove, o);
    canvas.addEventListener('pointerup', this.pointerUp, o);
    canvas.addEventListener('pointercancel', this.pointerUp, o);
    canvas.addEventListener('contextmenu', (e) => e.preventDefault(), o);
    canvas.addEventListener('wheel', this.wheel, { signal: this.abort.signal, passive: false });
    window.addEventListener('keydown', this.keyDown, o);
    window.addEventListener('keyup', this.keyUp, o);
    window.addEventListener('blur', this.releaseAll, o);
    document.addEventListener('visibilitychange', () => { if (document.hidden) this.releaseAll(); }, o);
    document.addEventListener('pointerlockchange', () => this.emitChange(), o);
  }

  get pointerLocked(): boolean { return document.pointerLockElement === this.canvas; }

  savePrefs(): void { savePref(PREFS_KEY, { sensitivity: this.camera.sensitivity, invertY: this.camera.invertY } satisfies FlyPrefs); }

  dispose(): void { this.abort.abort(); }

  private emitChange(): void { for (const cb of this.onChange) cb(); }

  private readonly releaseAll = (): void => {
    this.camera.held.clear();
    this.camera.shift = false;
    this.camera.ctrl = false;
    this.looking = false;
  };

  private readonly pointerDown = (e: PointerEvent): void => {
    if (e.button === 2) {
      this.looking = true;
      this.canvas.setPointerCapture(e.pointerId);
      e.preventDefault();
    } else if (e.button === 0 && !this.pointerLocked) {
      const r = this.canvas.getBoundingClientRect();
      const ev: PickEvent = {
        x: e.clientX - r.left, y: e.clientY - r.top, width: r.width, height: r.height,
        alt: e.altKey, shift: e.shiftKey, ctrl: e.ctrlKey, button: e.button,
      };
      for (const cb of this.onPick) cb(ev);
    }
    this.canvas.focus({ preventScroll: true });
  };

  private readonly pointerMove = (e: PointerEvent): void => {
    if (!this.looking && !this.pointerLocked) return;
    this.camera.look(e.movementX, e.movementY);
  };

  private readonly pointerUp = (e: PointerEvent): void => {
    if (e.button === 2 || e.type === 'pointercancel') {
      this.looking = false;
      if (this.canvas.hasPointerCapture(e.pointerId)) this.canvas.releasePointerCapture(e.pointerId);
    }
  };

  private readonly wheel = (e: WheelEvent): void => {
    e.preventDefault();
    this.camera.wheel(e.deltaY);
    this.emitChange();
  };

  private togglePointerLock(): void {
    if (this.pointerLocked) document.exitPointerLock();
    else {
      // requestPointerLock returns a promise in Chrome; failures (no user gesture) are non-fatal
      const r = this.canvas.requestPointerLock() as unknown as Promise<void> | undefined;
      r?.catch?.(() => undefined);
    }
  }

  private readonly keyDown = (e: KeyboardEvent): void => {
    if (isTextInput(e.target) || e.metaKey) return;
    this.camera.shift = e.shiftKey;
    this.camera.ctrl = e.ctrlKey;
    const a = keyToAction(e.code);
    if (a && !e.altKey) {
      this.camera.held.add(a);
      if (this.camera.playing) this.camera.stopPlayback();
      e.preventDefault();
      return;
    }
    const digit = /^Digit([1-9])$/.exec(e.code);
    if (digit && !e.altKey) {
      const slot = Number(digit[1]);
      if (e.ctrlKey) this.camera.saveBookmark(slot);
      else this.camera.recallBookmark(slot);
      e.preventDefault();
      this.emitChange();
      return;
    }
    if (e.code === 'KeyF' && !e.ctrlKey && !e.repeat) { this.togglePointerLock(); e.preventDefault(); return; }
    if (e.code === 'Home') { this.camera.reset(); e.preventDefault(); this.emitChange(); return; }
    if (this.extraKeys?.(e)) e.preventDefault();
  };

  private readonly keyUp = (e: KeyboardEvent): void => {
    this.camera.shift = e.shiftKey;
    this.camera.ctrl = e.ctrlKey;
    const a = keyToAction(e.code);
    if (a) this.camera.held.delete(a);
  };
}
