// Wires the M3a editor into the app shell: LightEditor (+ LightStore per scene, animation, undo), the timeline bar,
// the compare view, and the Lights / Animation / Validation folders of the panel. Called once from main.ts.
//   window.__editor  exposes the handle for automation (tests/editor/e2e-editor.ts).
import { lightTarget } from '../../core/scene/animation.ts';
import type { App } from '../app.ts';
import { DEG } from '../camera-math.ts';
import { CompareView } from '../compare/compare-view.ts';
import { defaultReferenceConfig, exportAndRenderReference, type ReferenceConfig, type ReferenceProgress } from '../compare/reference-flow.ts';
import type { Integration } from '../integration.ts';
import { pickFiles } from '../loader.ts';
import { TimelineBar } from '../timeline/timeline-bar.ts';
import { addAnimationPanel, download } from '../ui/panels/animation-panel.ts';
import { addComparePanel } from '../ui/panels/compare-panel.ts';
import { addLightsPanel } from '../ui/panels/lights-panel.ts';
import { parseEditorState, serializeEditorState, type EditorState } from './editor-state.ts';
import { LightEditor } from './light-editor.ts';

export interface EditorHandle {
  editor: LightEditor;
  compare: CompareView;
  timeline: TimelineBar;
  referenceConfig: ReferenceConfig;
  /** Dev only: export + Blender + auto-load into the compare view. */
  runReference?(cfg?: Partial<ReferenceConfig>, onProgress?: (p: ReferenceProgress) => void): Promise<unknown>;
  collectState(): EditorState;
  applyState(s: EditorState): void;
  saveState(): string;
  loadState(text: string): void;
}

declare global { interface Window { __editor?: EditorHandle } }

export function installEditor(app: App, integration?: Integration): EditorHandle {
  const editor = new LightEditor(app, { vbuffer: () => integration?.renderer()?.vbuffer });
  const compare = new CompareView(app);
  const timeline = new TimelineBar(editor.player);
  timeline.setKeyTargets(() => ['camera', 'env', ...(editor.selected !== undefined ? [lightTarget(editor.selected)] : [])]);
  editor.listeners.add((e) => { if (e === 'time' || e === 'selection' || e === 'undo' || e === 'scene') timeline.refresh(); });
  const referenceConfig = defaultReferenceConfig();

  const collectState = (): EditorState => {
    const cam = app.camera;
    const p = app.envParams;
    return {
      source: { name: app.scene?.name },
      lights: editor.store ? [...editor.store.list()] : [],
      lightIds: editor.store?.idState(),
      camera: { position: [...cam.position], quaternion: [...cam.quaternion], yfov: cam.yfov },
      env: app.env ? { url: p.url, strength: p.strength, rotationZ: p.rotationDeg * DEG, tint: [p.tint.r, p.tint.g, p.tint.b], visibleToCamera: p.visibleToCamera } : undefined,
      timeline: { time: editor.player.time, mode: editor.player.mode, frame: editor.player.frame },
      animation: editor.anim.toJSON(),
    };
  };

  const applyState = (s: EditorState): void => {
    if (!editor.store) throw new Error('load a scene before loading an editor scene.json');
    if (s.source?.name && app.scene && s.source.name !== app.scene.name) console.warn(`[editor] scene.json was saved for '${s.source.name}', the loaded scene is '${app.scene.name}'`);
    editor.store.replaceAll(s.lights, s.lightIds, 'load');
    editor.anim.load(s.animation);
    editor.undo.clear();
    app.camera.setPose({ position: [...s.camera.position], quaternion: [...s.camera.quaternion], yfov: s.camera.yfov });
    if (s.env && app.env) {
      app.envParams.strength = s.env.strength;
      app.envParams.rotationDeg = s.env.rotationZ / DEG;
      app.envParams.tint = { r: s.env.tint[0], g: s.env.tint[1], b: s.env.tint[2] };
      app.envParams.visibleToCamera = s.env.visibleToCamera;
      app.envParamsChanged();
    }
    editor.player.setMode(s.timeline.mode);
    if (s.timeline.mode === 'validation') editor.player.seekFrame(s.timeline.frame); else editor.player.seek(s.timeline.time);
    editor.select(undefined);
    app.panel?.refresh();
  };

  const handle: EditorHandle = {
    editor, compare, timeline, referenceConfig, collectState, applyState,
    saveState: () => serializeEditorState(collectState()),
    loadState: (text) => applyState(parseEditorState(text)),
  };
  if (import.meta.env.DEV) {
    handle.runReference = (cfg = {}, onProgress = () => undefined) => {
      if (!editor.store) return Promise.reject(new Error('no scene loaded'));
      return exportAndRenderReference({ app, store: editor.store, anim: editor.anim, player: editor.player, compare }, { ...referenceConfig, ...cfg }, onProgress);
    };
  }

  const folders = app.panel?.folders;
  if (folders) {
    addLightsPanel(folders.lights, editor);
    addAnimationPanel(folders.animation, editor, {
      save: () => download(`${(app.scene?.name ?? 'scene').replace(/[^\w.-]+/g, '_')}.scene.json`, handle.saveState()),
      load: () => {
        void pickFiles('.json').then(async (f) => {
          if (!f[0]) return;
          try { handle.loadState(await f[0].text()); } catch (e) { editor.message = e instanceof Error ? e.message : String(e); console.error(e); }
        });
      },
    });
    addComparePanel(folders.validation, compare, { config: referenceConfig, runReference: handle.runReference });
  }
  window.__editor = handle;
  return handle;
}
