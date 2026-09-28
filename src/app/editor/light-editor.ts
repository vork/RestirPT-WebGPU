// Interactive light editor (plan §5 M3a "Light editor", §6 "light gizmos"): add point/spot/rect/disk/sun, place on
// a surface through the V-buffer, CPU proxy picking, translate/rotate gizmos on the raster overlay, delete,
// duplicate, undo/redo and keyboard shortcuts. It also drives the timeline: the animation is applied to the
// LightStore / camera / env at the start of every app frame (App.beforeFrame).
//
// Input: LMB = pick / gizmo / place (RMB stays camera look in fly-controls.ts; Alt+LMB and "click = probe" stay the
// probe). Keys (ignored in text fields): Delete/Backspace delete · Ctrl/Cmd+D duplicate · Ctrl/Cmd+Z undo ·
// Shift+Ctrl/Cmd+Z or Ctrl+Y redo · G translate · R rotate · Esc cancel/deselect · Space play/pause.
// The keydown listener runs in the capture phase so Ctrl+D is not also seen as the fly camera's D.
import {
  Animation, lightTarget, matrixToPoseQ, type BaseState, type ChannelKind, type Interp, type TargetId,
} from '../../core/scene/animation.ts';
import { cloneLight, ensureLightStore, LightStore, type LightChangeBatch, type LightInit, type LightPatch } from '../../core/scene/light-store.ts';
import type { LightData, LightType, SceneData } from '../../core/scene/types.ts';
import type { App } from '../app.ts';
import { DEG, type Quat, type Vec3 as CamVec3 } from '../camera-math.ts';
import { isTextInput } from '../fly-controls.ts';
import { TimelinePlayer } from '../timeline/player.ts';
import {
  SUN_GIZMO_PX, beginDrag, dragMatrix, gizmoLayout, gizmoLines, lightWireframe, pickHandle, sunIconAnchor, type DragState, type GizmoLayout,
  type GizmoMode, type HandleId, type Lines,
} from './gizmos.ts';
import { pickLight, positionOf, screenRay, type ViewInfo } from './picking.ts';
import { defaultPlacementEpsilon, placementMatrix, readVBufferTexel, surfaceFromTexel, type PlacementFacing } from './placement.ts';
import {
  UndoStack, addLightCommand, changeTypeCommand, compositeCommand, removeLightCommand, trackCommand, updateLightCommand, type Command,
} from './undo.ts';
import { pickPixel } from '../../core/render/probe.ts';

export interface EditorHost {
  /** Current V-buffer (rgba32uint) at internal resolution, if the renderer has one. */
  vbuffer(): GPUTexture | undefined;
}

export interface PlacementSettings {
  facing: PlacementFacing;
  /** Surface offset ε (m); 0 = per-type default (defaultPlacementEpsilon). */
  epsilon: number;
}

export type EditorEvent = 'selection' | 'lights' | 'mode' | 'placement' | 'undo' | 'time' | 'scene' | 'message';

export class LightEditor {
  readonly undo = new UndoStack();
  readonly anim = new Animation();
  readonly player = new TimelinePlayer(this.anim);
  store: LightStore | undefined;
  scene: SceneData | undefined;
  selected: number | undefined;
  mode: GizmoMode = 'translate';
  /** Type waiting for a surface click, if any. */
  placing: LightType | undefined;
  readonly placement: PlacementSettings = { facing: 'surface', epsilon: 0 };
  /** Keys new keyframes use. */
  keyInterp: Interp = 'linear';
  /** Edits of animated lights set keys at the current time. */
  autoKey = false;
  snap = 0;
  message = '';
  readonly listeners = new Set<(e: EditorEvent) => void>();

  private drag: { st: DragState; id: number; before: Float32Array } | undefined;
  private hot: HandleId | undefined;
  private overlayIds: number[] = [];
  private overlayKey = '';
  private unsubStore: (() => void) | undefined;
  private readonly abort = new AbortController();

  constructor(readonly app: App, readonly host: EditorHost) {
    const c = app.canvas;
    const o = { signal: this.abort.signal };
    c.addEventListener('pointerdown', this.onPointerDown, o);
    c.addEventListener('pointermove', this.onPointerMove, o);
    c.addEventListener('pointerup', this.onPointerUp, o);
    c.addEventListener('pointercancel', this.onPointerCancel, o);
    window.addEventListener('keydown', this.onKeyDown, { signal: this.abort.signal, capture: true });
    app.beforeFrame.add(this.beforeFrame);
    this.anim.onChange(() => { this.player.dirty = true; this.emit('time'); });
    this.player.listeners.add(() => this.emit('time'));
    this.undo.listeners.add(() => this.emit('undo'));
  }

  dispose(): void {
    this.abort.abort();
    this.app.beforeFrame.delete(this.beforeFrame);
    this.unsubStore?.();
    for (const id of this.overlayIds) this.app.overlay.remove(id);
  }

  private emit(e: EditorEvent): void { for (const cb of [...this.listeners]) cb(e); }
  private say(msg: string): void { this.message = msg; this.emit('message'); }

  // ---- scene binding ----------------------------------------------------------------------------------------------

  /** Adopt the app's current scene (file lights become editable entries of its LightStore). */
  bindScene(scene: SceneData | undefined): void {
    if (scene === this.scene) return;
    this.unsubStore?.();
    this.scene = scene;
    this.store = scene ? ensureLightStore(scene) : undefined;
    this.unsubStore = this.store?.onChange((b) => this.onLightsChanged(b));
    this.selected = undefined;
    this.placing = undefined;
    this.drag = undefined;
    this.undo.clear();
    this.anim.load({ version: 1, duration: this.anim.duration, fps: this.anim.fps, loop: this.anim.loop, tracks: [] });
    this.player.seek(0);
    this.overlayKey = '';
    this.emit('scene');
    this.emit('lights');
    this.emit('selection');
  }

  private onLightsChanged(b: LightChangeBatch): void {
    if (this.selected !== undefined && !this.store?.has(this.selected)) {
      const tc = b.events.find((e) => e.kind === 'typeChanged' && e.prevId === this.selected);
      this.selected = tc ? tc.id : undefined;
      this.emit('selection');
    }
    this.overlayKey = '';
    if (b.source !== 'animation') this.emit('lights');
  }

  get lights(): readonly LightData[] { return this.store?.list() ?? []; }
  get selectedLight(): Readonly<LightData> | undefined { return this.selected === undefined ? undefined : this.store?.get(this.selected); }

  select(id: number | undefined): void {
    if (id !== undefined && !this.store?.has(id)) id = undefined;
    if (id === this.selected) return;
    this.selected = id;
    this.overlayKey = '';
    this.emit('selection');
  }

  setMode(m: GizmoMode): void { this.mode = m; this.overlayKey = ''; this.emit('mode'); }

  // ---- view -------------------------------------------------------------------------------------------------------

  view(): ViewInfo {
    const r = this.app.canvas.getBoundingClientRect();
    return { camToWorld: this.app.camera.camToWorld(), yfov: this.app.camera.yfov, width: Math.max(1, r.width), height: Math.max(1, r.height) };
  }

  private sceneDiag(): number { return this.app.sceneDiag || 10; }

  // ---- commands ---------------------------------------------------------------------------------------------------

  private requireStore(): LightStore {
    if (!this.store) throw new Error('no scene loaded');
    return this.store;
  }

  /** Add a light with an explicit init (API / automation). Returns the new id. */
  addLight(init: LightInit): number {
    const store = this.requireStore();
    let id = -1;
    this.undo.push(addLightCommand(store, init, (i) => { id = i; }));
    this.select(id);
    return id;
  }

  /** Add a light in front of the camera (no surface). */
  addInFront(type: LightType): number {
    const cam = this.app.camera;
    const m = cam.camToWorld();
    const d = 0.25 * this.sceneDiag();
    const pos: CamVec3 = [m[12] - m[8] * d, m[13] - m[9] * d, m[14] - m[10] * d];
    const mat = Float32Array.from(m);
    mat[12] = pos[0]; mat[13] = pos[1]; mat[14] = pos[2];
    // camera-looking frame: the light emits along the view direction (local −Z of the camera)
    return this.addLight(this.initFor(type, mat));
  }

  private initFor(type: LightType, matrix: Float32Array): LightInit {
    const diag = this.sceneDiag();
    const s = Math.max(0.05, 0.1 * diag);
    const n = this.lights.filter((l) => l.type === type).length;
    const base: LightInit = { type, matrix, name: `${type[0].toUpperCase()}${type.slice(1)}${n ? `.${String(n).padStart(3, '0')}` : ''}` };
    if (type === 'rect') { base.sizeX = s; base.sizeY = s; }
    if (type === 'disk') base.sizeX = s;
    return base;
  }

  /** Start "click a surface to place". */
  beginPlacement(type: LightType): void {
    if (!this.store) { this.say('load a scene first'); return; }
    if (type === 'sun') { this.addInFront('sun'); return; } // direction only: place in front of the camera
    this.placing = type;
    this.say(`click a surface to place the ${type} light (Esc cancels)`);
    this.emit('placement');
  }

  cancelPlacement(): void { if (this.placing) { this.placing = undefined; this.say(''); this.emit('placement'); } }

  /** Place `type` at internal-resolution pixel (px, py) through the V-buffer. Returns the new id, or undefined on a miss. */
  async placeAtPixel(type: LightType, px: number, py: number): Promise<number | undefined> {
    const store = this.requireStore();
    const vb = this.host.vbuffer();
    if (!vb || !this.scene) { this.say('no V-buffer yet'); return undefined; }
    const x = Math.min(vb.width - 1, Math.max(0, Math.floor(px))), y = Math.min(vb.height - 1, Math.max(0, Math.floor(py)));
    const texel = await readVBufferTexel(this.app.device, vb, x, y);
    if (this.store !== store) return undefined; // scene changed meanwhile
    const hit = surfaceFromTexel(this.scene.geometry, texel);
    if (!hit) { this.say('nothing under the cursor (background): light not placed'); return undefined; }
    const eye = this.app.camera.position;
    const eps = this.placement.epsilon > 0 ? this.placement.epsilon : defaultPlacementEpsilon(type, this.sceneDiag());
    const m = placementMatrix(hit, eye, { epsilon: eps, facing: this.placement.facing });
    const id = this.addLight(this.initFor(type, m));
    this.say(`placed ${type} on primitive ${hit.primId}`);
    return id;
  }

  deleteSelected(): void {
    const id = this.selected;
    if (id === undefined || !this.store) return;
    this.undo.push(removeLightCommand(this.store, this.anim, id));
    this.select(undefined);
  }

  duplicateSelected(): number | undefined {
    const l = this.selectedLight;
    if (!l || !this.store) return undefined;
    const c = cloneLight(l);
    const off = 0.03 * this.sceneDiag();
    const cam = this.app.camera.camToWorld();
    c.matrix[12] += cam[0] * off; c.matrix[13] += cam[1] * off; c.matrix[14] += cam[2] * off; // along camera right
    const { id: _drop, ...init } = c;
    return this.addLight({ ...init, name: `${l.name} copy` });
  }

  /** Property edit of the selected light (one undo step). `type` changes go through setType. */
  updateSelected(patch: LightPatch, label = 'edit light'): void {
    const l = this.selectedLight;
    if (!l || !this.store) return;
    const before: LightPatch = {};
    for (const k of Object.keys(patch) as (keyof LightPatch)[]) (before as Record<string, unknown>)[k] = k === 'matrix' ? Float32Array.from(l.matrix) : k === 'color' ? [...l.color] : l[k];
    try {
      const cmds: Command[] = [updateLightCommand(this.store, l.id, before, patch, label)];
      cmds[0].do();
      const auto = this.autoKeyCommand(l.id, Object.keys(patch) as (keyof LightPatch)[]);
      if (auto) { auto.do(); cmds.push(auto); }
      this.undo.push(cmds.length > 1 ? compositeCommand(label, cmds) : cmds[0], true);
    } catch (e) {
      this.say(e instanceof Error ? e.message : String(e));
      this.emit('lights');
    }
  }

  private live: { id: number; before: LightPatch } | undefined;

  /** Slider-style edit: applies immediately, records ONE undo step when `commit` (end of the slider drag). */
  editLive(patch: LightPatch, commit: boolean, label = 'edit light'): void {
    const l = this.selectedLight;
    if (!l || !this.store) return;
    if (this.live && this.live.id !== l.id) this.live = undefined;
    if (!this.live) this.live = { id: l.id, before: {} };
    const before = this.live.before as Record<string, unknown>;
    for (const k of Object.keys(patch) as (keyof LightPatch)[]) {
      if (!(k in before)) before[k] = k === 'matrix' ? Float32Array.from(l.matrix) : k === 'color' ? [...l.color] : l[k];
    }
    try {
      this.store.update(l.id, patch, 'editor');
    } catch (e) {
      this.say(e instanceof Error ? e.message : String(e));
      this.emit('lights');
      return;
    }
    if (!commit) return;
    const b = this.live.before;
    this.live = undefined;
    const cur = this.store.get(l.id)!;
    const after: LightPatch = {};
    for (const k of Object.keys(b) as (keyof LightPatch)[]) (after as Record<string, unknown>)[k] = k === 'matrix' ? Float32Array.from(cur.matrix) : k === 'color' ? [...cur.color] : cur[k];
    const cmds: Command[] = [updateLightCommand(this.store, l.id, b, after, label)];
    const auto = this.autoKeyCommand(l.id, Object.keys(b) as (keyof LightPatch)[]);
    if (auto) { auto.do(); cmds.push(auto); }
    this.undo.push(cmds.length > 1 ? compositeCommand(label, cmds) : cmds[0], true);
  }

  setType(type: LightType): void {
    const l = this.selectedLight;
    if (!l || !this.store || l.type === type) return;
    this.undo.push(changeTypeCommand(this.store, this.anim, l.id, type, (id) => this.select(id)));
  }

  undoLast(): void { this.undo.undo(); this.overlayKey = ''; this.emit('lights'); }
  redoLast(): void { this.undo.redo(); this.overlayKey = ''; this.emit('lights'); }

  // ---- animation --------------------------------------------------------------------------------------------------

  baseState(): BaseState {
    const cam = this.app.camera;
    const env = this.app.env ? { rotationZ: this.app.envParams.rotationDeg * DEG, strength: this.app.envParams.strength } : undefined;
    return { camera: { position: [...cam.position], quaternion: [...cam.quaternion], yfov: cam.yfov }, lights: this.lights, env };
  }

  private channelValues(target: TargetId): Partial<Record<ChannelKind, number[]>> {
    if (target === 'camera') {
      const c = this.app.camera;
      return { position: [...c.position], rotation: [...c.quaternion], yfov: [c.yfov] };
    }
    if (target === 'env') return { rotationZ: [this.app.envParams.rotationDeg * DEG], strength: [this.app.envParams.strength] };
    const id = Number(target.slice(6));
    const l = this.store?.get(id);
    if (!l) return {};
    const p = matrixToPoseQ(l.matrix);
    return { position: p.position, rotation: p.quaternion, power: [l.power], color: [...l.color] };
  }

  /** Key the given channels (default: all) of a target at the current time, as one undo step. */
  keyTarget(target: TargetId, channels?: ChannelKind[]): void {
    const vals = this.channelValues(target);
    const t = this.player.time;
    const chs = (channels ?? (Object.keys(vals) as ChannelKind[])).filter((c) => vals[c]);
    if (!chs.length) return;
    this.undo.push(trackCommand(this.anim, target, `key ${target} @ ${t.toFixed(3)} s`, () => {
      for (const ch of chs) this.anim.setKey(target, ch, t, vals[ch]!, this.keyInterp, false);
      this.anim.touch(target);
    }));
  }

  removeKeyAt(target: TargetId): void {
    const t = this.player.time;
    const has = this.anim.keyTimes(target).some((k) => Math.abs(k - t) < 1e-6);
    if (!has) { this.say(`no ${target} key at ${t.toFixed(3)} s`); return; }
    this.undo.push(trackCommand(this.anim, target, `remove key ${target}`, () => {
      for (const k of this.anim.keyTimes(target)) if (Math.abs(k - t) < 1e-6) this.anim.removeKey(target, k);
    }));
  }

  /** Apply a preset's keys (replacing those channels) as one undo step. */
  applyPreset(target: TargetId, keys: Partial<Record<ChannelKind, { t: number; v: number[]; interp: Interp }[]>>, label: string): void {
    this.undo.push(trackCommand(this.anim, target, label, () => {
      for (const [ch, ks] of Object.entries(keys) as [ChannelKind, { t: number; v: number[]; interp: Interp }[]][]) this.anim.replaceChannel(target, ch, ks);
    }));
  }

  private autoKeyCommand(id: number, fields: (keyof LightPatch)[]): Command | undefined {
    const target = lightTarget(id);
    if (!this.autoKey || !this.anim.hasTrack(target)) return undefined;
    const map: Partial<Record<keyof LightPatch, ChannelKind[]>> = { matrix: ['position', 'rotation'], power: ['power'], color: ['color'] };
    const chs = [...new Set(fields.flatMap((f) => map[f] ?? []))];
    if (!chs.length) return undefined;
    const t = this.player.time;
    return trackCommand(this.anim, target, 'auto key', () => {
      const vals = this.channelValues(target);
      for (const ch of chs) this.anim.setKey(target, ch, t, vals[ch]!, this.keyInterp, false);
      this.anim.touch(target);
    });
  }

  /** Apply the animation at the player's time to the store (one 'animation' batch), camera and env. */
  applyAnimation(): void {
    if (!this.store || this.anim.isEmpty) return;
    const t = this.player.mode === 'validation' ? this.player.time : this.anim.wrapTime(this.player.time);
    const s = this.anim.evaluate(t, this.baseState());
    const store = this.store;
    store.batch('animation', () => {
      for (const r of s.lights.values()) {
        if (this.drag?.id === r.id) continue; // the user is dragging it
        store.update(r.id, { matrix: r.matrix, power: r.power, color: r.color }, 'animation');
      }
    });
    if (s.camera) this.app.camera.setPose({ position: s.camera.position, quaternion: s.camera.quaternion as Quat, yfov: s.camera.yfov });
    if (s.env && this.app.env) {
      this.app.envParams.rotationDeg = s.env.rotationZ / DEG;
      this.app.envParams.strength = s.env.strength;
      this.app.envParamsChanged();
    }
  }

  // ---- frame hook -------------------------------------------------------------------------------------------------

  private readonly beforeFrame = (f: { rawDt: number; advance: boolean }): void => {
    if (this.app.scene !== this.scene) this.bindScene(this.app.scene);
    if (this.player.tick(f.rawDt, f.advance) || this.player.dirty) {
      this.player.dirty = false;
      this.applyAnimation();
      if (this.player.playing) this.emit('time');
    }
    this.updateOverlay();
  };

  /** Rebuild the overlay line lists when the camera, lights, selection or canvas changed. */
  updateOverlay(): void {
    const ov = this.app.overlay;
    const v = this.view();
    const key = `${this.app.camera.version}|${this.store?.version ?? -1}|${this.selected}|${this.mode}|${this.hot}|${this.drag?.st.handle}|${v.width}x${v.height}|${this.app.camera.yfov}`;
    if (key === this.overlayKey) return;
    this.overlayKey = key;
    for (const id of this.overlayIds) ov.remove(id);
    this.overlayIds = [];
    const lights = this.lights;
    if (!lights.length) return;
    const depth: Lines = { points: [], colors: [] };
    const top: Lines = { points: [], colors: [] };
    let sunIdx = 0;
    for (const l of lights) {
      const w = lightWireframe(l, v, l.id === this.selected, l.type === 'sun' ? sunIdx : 0);
      const dst = l.type === 'sun' ? top : depth;
      if (l.type === 'sun') sunIdx++;
      dst.points.push(...w.points); dst.colors.push(...w.colors);
    }
    const sel = this.selectedLight;
    if (sel) {
      const gl = gizmoLines(this.layoutFor(sel, v), this.modeFor(sel), this.hot, this.drag?.st.handle);
      top.points.push(...gl.points); top.colors.push(...gl.colors);
    }
    if (depth.points.length) this.overlayIds.push(ov.addLines(depth.points, depth.colors, false));
    if (top.points.length) this.overlayIds.push(ov.addLines(top.points, top.colors, true));
  }

  /** Suns have a direction only: always the rotate rings, drawn around their screen icon. */
  private modeFor(l: Readonly<LightData>): GizmoMode { return l.type === 'sun' ? 'rotate' : this.mode; }

  private sunIndex(id: number): number { return this.lights.filter((l) => l.type === 'sun').findIndex((l) => l.id === id); }

  /** Gizmo placement for a light: at its position, or (suns) around the screen-fixed icon with smaller rings. */
  layoutFor(l: Readonly<LightData>, v: ViewInfo): GizmoLayout {
    if (l.type === 'sun') return gizmoLayout(sunIconAnchor(Math.max(0, this.sunIndex(l.id)), v), v, SUN_GIZMO_PX);
    return gizmoLayout(positionOf(l.matrix), v);
  }

  // ---- pointer ----------------------------------------------------------------------------------------------------

  private localXY(e: PointerEvent): [number, number] {
    const r = this.app.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  private ignorePointer(e: PointerEvent): boolean {
    return e.button !== 0 || e.altKey || !!this.app.panel?.probeMode() || document.pointerLockElement === this.app.canvas;
  }

  /** Handle a LMB press at CSS (x, y). Exposed for automation (Playwright drives real events too). */
  async pressAt(x: number, y: number): Promise<'placed' | 'drag' | 'picked' | 'none'> {
    if (!this.store) return 'none';
    const v = this.view();
    if (this.placing) {
      const type = this.placing;
      this.placing = undefined;
      this.emit('placement');
      const t = this.app.targets;
      const [px, py] = pickPixel(x, y, v.width, v.height, t.width, t.height);
      const id = await this.placeAtPixel(type, px, py);
      return id === undefined ? 'none' : 'placed';
    }
    const sel = this.selectedLight;
    if (sel) {
      const g = this.layoutFor(sel, v);
      const h = pickHandle(g, this.modeFor(sel), v, x, y);
      if (h) {
        const st = beginDrag(h, g.center, sel.matrix, screenRay(v, x, y), v, x, y);
        if (st) {
          this.drag = { st, id: sel.id, before: Float32Array.from(sel.matrix) };
          this.overlayKey = '';
          return 'drag';
        }
      }
    }
    const hit = pickLight(this.lights, v, x, y);
    this.select(hit?.id);
    return hit ? 'picked' : 'none';
  }

  /** Continue a drag at CSS (x, y). */
  moveTo(x: number, y: number, snap = false): void {
    if (!this.drag || !this.store) {
      const sel = this.selectedLight;
      if (sel) {
        const v = this.view();
        this.hot = pickHandle(this.layoutFor(sel, v), this.modeFor(sel), v, x, y);
      }
      return;
    }
    const v = this.view();
    const m = dragMatrix(this.drag.st, screenRay(v, x, y), snap ? { snap: this.snap || 0.1 * Math.pow(10, Math.floor(Math.log10(this.sceneDiag() / 10))), angleSnap: 15 * DEG } : {}, v, x, y);
    try { this.store.update(this.drag.id, { matrix: Float32Array.from(m) }, 'drag'); } catch (e) { this.say(String(e)); }
  }

  /** Finish a drag: one undo step from the start matrix to the final one. */
  release(): void {
    const d = this.drag;
    this.drag = undefined;
    this.overlayKey = '';
    if (!d || !this.store?.has(d.id)) return;
    const after = Float32Array.from(this.store.get(d.id)!.matrix);
    if (after.every((x, i) => x === d.before[i])) return;
    const cmds: Command[] = [updateLightCommand(this.store, d.id, { matrix: d.before }, { matrix: after }, d.st.handle.startsWith('r') ? 'rotate light' : 'move light')];
    const auto = this.autoKeyCommand(d.id, ['matrix']);
    if (auto) { auto.do(); cmds.push(auto); }
    this.undo.push(cmds.length > 1 ? compositeCommand(cmds[0].label, cmds) : cmds[0], true);
    this.emit('lights');
  }

  cancelDrag(): void {
    const d = this.drag;
    this.drag = undefined;
    this.overlayKey = '';
    if (d && this.store?.has(d.id)) this.store.update(d.id, { matrix: d.before }, 'drag');
  }

  get dragging(): boolean { return !!this.drag; }

  private readonly onPointerDown = (e: PointerEvent): void => {
    if (this.ignorePointer(e)) return;
    const [x, y] = this.localXY(e);
    void this.pressAt(x, y).then((r) => {
      if (r === 'drag') { try { this.app.canvas.setPointerCapture(e.pointerId); } catch { /* released already */ } }
    });
  };

  private readonly onPointerMove = (e: PointerEvent): void => {
    if (document.pointerLockElement === this.app.canvas) return;
    if (!this.drag && e.buttons & 2) return; // RMB look in progress
    const [x, y] = this.localXY(e);
    this.moveTo(x, y, e.ctrlKey || e.metaKey);
  };

  private readonly onPointerUp = (e: PointerEvent): void => {
    if (e.button !== 0 || !this.drag) return;
    const [x, y] = this.localXY(e);
    this.moveTo(x, y, e.ctrlKey || e.metaKey);
    this.release();
    if (this.app.canvas.hasPointerCapture(e.pointerId)) this.app.canvas.releasePointerCapture(e.pointerId);
  };

  private readonly onPointerCancel = (): void => { this.cancelDrag(); };

  // ---- keyboard ---------------------------------------------------------------------------------------------------

  private readonly onKeyDown = (e: KeyboardEvent): void => {
    if (isTextInput(e.target) || e.altKey) return;
    const mod = e.ctrlKey || e.metaKey;
    let handled = true;
    if (mod && e.code === 'KeyZ') { if (e.shiftKey) this.redoLast(); else this.undoLast(); }
    else if (mod && e.code === 'KeyY') this.redoLast();
    else if (mod && e.code === 'KeyD') { if (this.selected !== undefined) this.duplicateSelected(); }
    else if (!mod && (e.code === 'Delete' || e.code === 'Backspace') && this.selected !== undefined) this.deleteSelected();
    else if (!mod && !e.shiftKey && e.code === 'KeyG' && this.selected !== undefined) this.setMode('translate');
    else if (!mod && !e.shiftKey && e.code === 'KeyR' && this.selected !== undefined) this.setMode('rotate');
    else if (!mod && e.code === 'Space' && !e.repeat && !(e.target instanceof HTMLButtonElement)) this.player.toggle();
    else if (e.code === 'Escape') {
      if (this.drag) this.cancelDrag();
      else if (this.placing) this.cancelPlacement();
      else this.select(undefined);
    } else handled = false;
    if (handled) { e.preventDefault(); e.stopImmediatePropagation(); }
  };
}
