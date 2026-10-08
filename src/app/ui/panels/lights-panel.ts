// 'Lights' panel (plan §1.4 "Light-editor properties", §5 M3a): add/place, selection, gizmo mode, delete/duplicate,
// undo/redo, and the selected light's properties in Blender units (power W — sun W/m² —, colour, exposure, spot
// size + blend, area shape (rect/disk = type) + size/sizeY, spread, visibleToCamera). No radius field (v1 lights are
// r = 0, plan §1.4). Changing the type is remove + add with a new id. Emissive meshes are listed read-only.
import { LightStore } from '../../../core/scene/light-store.ts';
import type { LightType } from '../../../core/scene/types.ts';
import { DEG } from '../../camera-math.ts';
import type { LightEditor } from '../../editor/light-editor.ts';
import { tip, type TpFolder } from '../tweakpane.ts';

const ADD_TITLES: Record<LightType, string> = {
  point: 'Add point light', spot: 'Add spot light', rect: 'Add area light (rectangle)', disk: 'Add area light (disk)', sun: 'Add sun (direction only)',
};

export const MODE_A_WARNING = 'Mode A exports only: Cycles shows area lights to camera rays only with MIS, so a Mode A export of a camera-visible area light fails. Mode B (the default) is fine.';

/** Fills the (pre-created, hidden) Lights folder of the panel and shows it. */
export function addLightsPanel(f: TpFolder, ed: LightEditor): { folder: TpFolder; refresh(): void } {
  f.hidden = false;
  // Tweakpane's refresh() re-emits 'change' for values it pulls from the object: ignore those.
  let quiet = false;
  const q = <T>(fn: (e: T) => void) => (e: T) => { if (!quiet) fn(e); };
  const refreshQuiet = (x: { refresh(): void } | undefined) => { quiet = true; try { x?.refresh(); } finally { quiet = false; } };

  const ui = {
    facing: ed.placement.facing as string,
    epsilon: ed.placement.epsilon,
    selected: -1,
    mode: ed.mode as string,
    status: '',
    emissive: '',
  };
  const add = f.addFolder({ title: 'Add a light (then click a surface)', expanded: true });
  for (const t of ['point', 'spot', 'rect', 'disk', 'sun'] as LightType[]) {
    const b = add.addButton({ title: ADD_TITLES[t] }).on('click', () => ed.beginPlacement(t));
    if (t === 'sun') tip(b, 'No surface click: the sun is added in front of the camera; aim it with the rotate gizmo (R).');
  }
  add.addBinding(ui, 'facing', { label: 'orientation', options: { 'away from the surface': 'surface', 'toward the camera': 'camera' } })
    .on('change', q((e) => { ed.placement.facing = e.value as 'surface' | 'camera'; }));
  tip(add.addBinding(ui, 'epsilon', { label: 'surface offset (m)', min: 0, max: 10, step: 0.001 }), 'Distance of a placed light from the clicked surface; 0 = automatic.').on('change', q((e) => { ed.placement.epsilon = e.value; }));

  let selBinding = f.addBinding(ui, 'selected', { label: 'selected', options: { '(none)': -1 } });
  let selKey = '';
  const rebuildSelect = () => {
    const key = ed.lights.map((l) => `${l.id}:${l.name}:${l.type}`).join('|');
    ui.selected = ed.selected ?? -1;
    if (key === selKey) return;
    selKey = key;
    const opts: Record<string, number> = { '(none)': -1 };
    for (const l of ed.lights) opts[`${l.name} [${l.type} #${l.id}]`] = l.id;
    const idx = f.children.indexOf(selBinding);
    selBinding.dispose();
    selBinding = f.addBinding(ui, 'selected', { label: 'selected', options: opts, index: idx });
    selBinding.on('change', q((e) => ed.select(e.value >= 0 ? e.value : undefined)));
  };
  f.addBinding(ui, 'mode', { label: 'gizmo (G / R)', options: { translate: 'translate', rotate: 'rotate' } }).on('change', q((e) => ed.setMode(e.value as 'translate' | 'rotate')));
  f.addButton({ title: 'Duplicate (Ctrl+D)' }).on('click', () => ed.duplicateSelected());
  f.addButton({ title: 'Delete (Del)' }).on('click', () => ed.deleteSelected());
  const undoBtn = f.addButton({ title: 'Undo (Ctrl+Z)' }).on('click', () => ed.undoLast());
  const redoBtn = f.addButton({ title: 'Redo (Shift+Ctrl+Z)' }).on('click', () => ed.redoLast());
  f.addBinding(ui, 'status', { readonly: true, multiline: true, rows: 2, label: 'status' });

  // ---- selected light properties (rebuilt on selection / type change) ----
  let props: TpFolder | undefined;
  const p = {
    name: '', type: 'point' as string, power: 0, color: { r: 1, g: 1, b: 1 }, exposure: 0, spotSizeDeg: 45, spotBlend: 0.15,
    sizeX: 1, sizeY: 1, spreadDeg: 180, visibleToCamera: false, warning: '', simplified: '',
  };
  const load = () => {
    const l = ed.selectedLight;
    if (!l) return;
    p.name = l.name; p.type = l.type; p.power = l.power; p.color = { r: l.color[0], g: l.color[1], b: l.color[2] };
    p.exposure = l.exposure; p.spotSizeDeg = (l.spotSize ?? 0) / DEG; p.spotBlend = l.spotBlend ?? 0;
    p.sizeX = l.sizeX ?? 0; p.sizeY = l.sizeY ?? l.sizeX ?? 0; p.spreadDeg = (l.spread ?? Math.PI) / DEG; p.visibleToCamera = l.visibleToCamera;
    p.warning = (l.type === 'rect' || l.type === 'disk') && l.visibleToCamera ? MODE_A_WARNING : '';
    p.simplified = l.simplified ?? '';
  };
  let builtFor = '';
  const buildProps = () => {
    const l = ed.selectedLight;
    const key = l ? `${l.id}|${l.type}` : '';
    if (key === builtFor) { load(); refreshQuiet(props); return; }
    builtFor = key;
    props?.dispose();
    props = undefined;
    if (!l) return;
    load();
    const pf = f.addFolder({ title: `Properties: ${l.name}`, expanded: true, index: f.children.indexOf(emF) });
    props = pf;
    const commit = (ev: { last?: boolean }) => ev.last !== false;
    pf.addBinding(p, 'name', { label: 'name' }).on('change', q((e) => ed.updateSelected({ name: e.value }, 'rename light')));
    pf.addBinding(p, 'type', { label: 'type / shape', options: { point: 'point', spot: 'spot', 'area: rect': 'rect', 'area: disk': 'disk', sun: 'sun' } })
      .on('change', q((e) => ed.setType(e.value as LightType)));
    pf.addBinding(p, 'power', { label: l.type === 'sun' ? 'strength (W/m²)' : 'power (W)', min: 0, max: l.type === 'sun' ? 100 : 5000, step: 0.01 })
      .on('change', q((e) => ed.editLive({ power: Math.max(0, e.value) }, commit(e), 'power')));
    pf.addBinding(p, 'color', { label: 'colour', color: { type: 'float' } })
      .on('change', q((e) => ed.editLive({ color: [Math.max(0, e.value.r), Math.max(0, e.value.g), Math.max(0, e.value.b)] }, commit(e), 'colour')));
    pf.addBinding(p, 'exposure', { label: 'exposure (EV)', min: -10, max: 10, step: 0.01 }).on('change', q((e) => ed.editLive({ exposure: e.value }, commit(e), 'exposure')));
    if (l.type === 'spot') {
      pf.addBinding(p, 'spotSizeDeg', { label: 'spot size (°)', min: 1, max: 180, step: 0.1 })
        .on('change', q((e) => ed.editLive({ spotSize: Math.min(180, Math.max(1, e.value)) * DEG }, commit(e), 'spot size')));
      pf.addBinding(p, 'spotBlend', { label: 'spot blend', min: 0, max: 1, step: 0.001 })
        .on('change', q((e) => ed.editLive({ spotBlend: Math.min(1, Math.max(0, e.value)) }, commit(e), 'spot blend')));
    }
    if (l.type === 'rect' || l.type === 'disk') {
      pf.addBinding(p, 'sizeX', { label: l.type === 'disk' ? 'size (diameter, m)' : 'size X (m)', min: 1e-4, max: 100, step: 0.001 })
        .on('change', q((e) => ed.editLive({ sizeX: Math.max(1e-4, e.value) }, commit(e), 'size')));
      if (l.type === 'rect') {
        pf.addBinding(p, 'sizeY', { label: 'size Y (m)', min: 1e-4, max: 100, step: 0.001 })
          .on('change', q((e) => ed.editLive({ sizeY: Math.max(1e-4, e.value) }, commit(e), 'size Y')));
      }
      pf.addBinding(p, 'spreadDeg', { label: 'spread (°)', min: 0.1, max: 180, step: 0.1 })
        .on('change', q((e) => ed.editLive({ spread: Math.min(180, Math.max(0.1, e.value)) * DEG }, commit(e), 'spread')));
      pf.addBinding(p, 'visibleToCamera', { label: 'visible to camera' }).on('change', q((e) => {
        if (e.value) { console.warn(`[lights] ${MODE_A_WARNING}`); ed.message = MODE_A_WARNING; }
        ed.updateSelected({ visibleToCamera: e.value }, 'visible to camera');
      }));
      pf.addBinding(p, 'warning', { readonly: true, multiline: true, rows: 3, label: 'note' });
    }
    if (l.simplified) pf.addBinding(p, 'simplified', { readonly: true, label: 'simplified (file)' });
  };

  const refresh = () => {
    ui.mode = ed.mode;
    ui.status = ed.message || (ed.placing ? `placing ${ed.placing}: click a surface` : `${ed.lights.length} lights`);
    const s = ed.scene;
    const em = s ? LightStore.emissiveMeshes(s) : [];
    ui.emissive = em.length ? em.map((m) => `${m.name}: ${m.triangles} tris, ${m.area.toFixed(3)} m², ~${m.approxPower.toFixed(1)} W${m.textured ? ' (textured)' : ''}`).join('\n') : '(none)';
    undoBtn.title = ed.undo.canUndo ? `Undo ${ed.undo.undoLabel} (Ctrl+Z)` : 'Undo (Ctrl+Z)';
    redoBtn.title = ed.undo.canRedo ? `Redo ${ed.undo.redoLabel} (Shift+Ctrl+Z)` : 'Redo (Shift+Ctrl+Z)';
    buildProps();
    refreshQuiet(f);
  };
  const emF = f.addFolder({ title: 'Emissive meshes (static, read-only)', expanded: false });
  emF.addBinding(ui, 'emissive', { readonly: true, multiline: true, rows: 4, label: 'meshes' });

  let pendingSelect = true;
  ed.listeners.add((e) => {
    if (e === 'selection' || e === 'lights' || e === 'scene') pendingSelect = true;
    if (e === 'time') return; // high-frequency during playback; lights edits by the animation are not shown live
    if (pendingSelect) { rebuildSelect(); pendingSelect = false; }
    refresh();
  });
  rebuildSelect();
  refresh();
  return { folder: f, refresh };
}
