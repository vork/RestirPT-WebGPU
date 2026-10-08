// Tweakpane UI shell (plan §5 M1). The top-level folders are created here in one fixed order; the folders whose
// controls need the renderer or the editor (Lights, Animation, ReSTIR, Denoiser, Validation) start hidden and are
// filled and shown by integration.ts / editor/index.ts (PanelHandle.folders):
//   Scene · Camera · Environment · Lights · Animation · Render (+ Output, Sampling, Advanced) · ReSTIR · Denoiser ·
//   Debug views (+ View mapping, Pixel probe) · Validation · Help
// Expert folders start collapsed. Debug views are picked by category, then view (registry groups → CATEGORIES).
import { DEBUG_TAPS, type DebugViewDef } from '../../core/render/debug-views.ts';
import { cmapCode, codeName, legendCodes } from '../../core/render/restir/debug.ts';
import type { App, ColorFormat } from '../app.ts';
import { DEG } from '../camera-math.ts';
import { pickFiles } from '../loader.ts';
import { addHtmlBlock, createPane, tip, type TpBinding, type TpFolder, type TpPane } from './tweakpane.ts';

export interface PanelFolders {
  scene: TpFolder;
  camera: TpFolder;
  environment: TpFolder;
  lights: TpFolder;
  animation: TpFolder;
  /** Render: the integrator controls are inserted at the top by integration.ts. */
  render: TpFolder;
  renderAdvanced: TpFolder;
  restir: TpFolder;
  denoiser: TpFolder;
  debug: TpFolder;
  /** Debug › Pixel probe (integration.ts adds the ReSTIR pixel inspector toggle). */
  probe: TpFolder;
  validation: TpFolder;
  help: TpFolder;
}

export interface PanelHandle {
  pane: TpPane;
  folders: PanelFolders;
  refresh(): void;
  refreshMonitors(): void;
  /** 'Click to probe' mode: plain LMB selects the probe pixel. */
  probeMode(): boolean;
}

/** Debug view categories: registry group → UI name, in display order (unknown groups follow, alphabetically). */
const CATEGORIES: [group: string, name: string][] = [
  ['G-buffer', 'G-buffer (primary hit)'],
  ['Shading normals', 'Shading normals / normal maps'],
  ['Checks', 'Checks (NaN/Inf)'],
  ['BVH', 'BVH traversal'],
  ['Env', 'Environment (primary)'],
  ['Env sampling', 'Environment sampling'],
  ['ReSTIR reservoir', 'ReSTIR: reservoir'],
  ['ReSTIR temporal', 'ReSTIR: temporal reuse'],
  ['ReSTIR shift', 'ReSTIR: spatial shifts'],
  ['ReSTIR MIS', 'ReSTIR: MIS weights'],
  ['ReSTIR Enhanced (M6)', 'ReSTIR: pairing / duplication map'],
  ['Denoiser', 'Denoiser'],
  ['Test', 'Test pattern'],
];
const categoryName = (group: string) => CATEGORIES.find(([g]) => g === group)?.[1] ?? group;
const categoryRank = (group: string) => { const i = CATEGORIES.findIndex(([g]) => g === group); return i < 0 ? CATEGORIES.length : i; };

const KEYS_HELP = [
  ['RMB drag', 'look around'],
  ['F', 'fly mode (pointer lock)'],
  ['W A S D', 'move'],
  ['E / Q', 'up / down'],
  ['wheel', 'fly speed (Shift ×4, Ctrl ×0.25)'],
  ['Home', 'reset the view'],
  ['Ctrl+1..9', 'save a bookmark'],
  ['1..9', 'recall a bookmark'],
  ['LMB', 'select / place a light, drag a gizmo'],
  ['G / R', 'translate / rotate gizmo'],
  ['Del', 'delete the selected light'],
  ['Ctrl+D', 'duplicate the selected light'],
  ['Ctrl+Z', 'undo (Shift+Ctrl+Z, Ctrl+Y redo)'],
  ['Esc', 'cancel / deselect'],
  ['Space', 'play / pause the animation'],
  ['P / .', 'pause / step one frame'],
  ['Shift+R', 'reset history (incl. ReSTIR temporal)'],
  ['Alt+click', 'probe a pixel'],
  ['H', 'show / hide the HUD'],
  ['drop files', '.glb .gltf (+ .bin, textures) .usd* .hdr .exr'],
];

export function buildPanel(app: App): PanelHandle {
  const host = document.createElement('div');
  host.className = 'panel';
  document.body.append(host);
  const pane = createPane(host, 'ReSTIR PT');
  // Tweakpane's refresh() re-emits 'change' for values it pulls from the objects: the handlers below ignore those.
  let quiet = false;
  const q = <T>(fn: (e: T) => void) => (e: T) => { if (!quiet) fn(e); };

  // ---- top-level folders, fixed order ----
  const top = (title: string, expanded: boolean, hidden = false) => {
    const f = pane.addFolder({ title, expanded });
    f.hidden = hidden;
    return f;
  };
  const scene = top('Scene', true);
  const camF = top('Camera', false);
  const eF = top('Environment', false);
  const lights = top('Lights', true, true);
  const animation = top('Animation', false, true);
  const rF = top('Render', true);
  const restir = top('ReSTIR', false, true);
  const denoiser = top('Denoiser', false, true);
  const dF = top('Debug views', false);
  const validation = top('Validation', false, true);
  const help = top('Help: keys and mouse', false);

  // ---- Scene ----
  const sceneUi = { url: '', info: 'no scene (test pattern)' };
  tip(scene.addButton({ title: 'Open files...' }), 'glTF (.glb, or .gltf with its .bin and textures), USD (.usd/.usda/.usdc/.usdz), or an HDRI (.hdr/.exr). Drag and drop works too.')
    .on('click', () => { void pickFiles().then((f) => (f.length ? app.loadFiles(f) : undefined)); });
  scene.addBinding(sceneUi, 'url', { label: 'scene URL' });
  scene.addButton({ title: 'Load URL' }).on('click', () => {
    const u = sceneUi.url.trim();
    if (!u) return;
    try {
      const url = new URL(u, location.href).href;
      void app.loadScene({ kind: 'url', url, name: url.split('/').pop() || url });
    } catch { app.loading.start('Load URL'); app.loading.error(`bad URL: ${u}`); }
  });
  scene.addBinding(sceneUi, 'info', { readonly: true, multiline: true, rows: 4, label: 'info' });

  // ---- Camera ----
  const cam = app.camera;
  const camUi = {
    get fovDeg() { return cam.yfov / DEG; },
    set fovDeg(v: number) { cam.yfov = v * DEG; cam.version++; },
    get sensitivity() { return cam.sensitivity * 1000; },
    set sensitivity(v: number) { cam.sensitivity = v / 1000; app.controls.savePrefs(); },
    get invertY() { return cam.invertY; },
    set invertY(v: boolean) { cam.invertY = v; app.controls.savePrefs(); },
    speed: '',
    fileCamera: 0,
    loop: false,
    bookmarks: '',
  };
  camF.addBinding(camUi, 'fovDeg', { label: 'vertical FOV (°)', min: 5, max: 120, step: 0.1 });
  tip(camF.addBinding(camUi, 'speed', { readonly: true, label: 'fly speed' }), 'Mouse wheel changes the speed; Shift ×4, Ctrl ×0.25 while held.');
  camF.addBinding(camUi, 'sensitivity', { label: 'mouse sensitivity', min: 0.2, max: 10, step: 0.1 });
  camF.addBinding(camUi, 'invertY', { label: 'invert Y' });
  camF.addButton({ title: 'Reset view (Home)' }).on('click', () => cam.reset());
  const views = camF.addFolder({ title: 'File cameras and bookmarks', expanded: false });
  views.addBinding(camUi, 'fileCamera', { label: 'file camera #', min: 0, max: 16, step: 1 });
  views.addButton({ title: 'Use file camera' }).on('click', () => {
    if (!app.useFileCamera(camUi.fileCamera)) { app.loading.start('Camera'); app.loading.error('The scene has no camera with that index.'); }
  });
  tip(views.addBinding(camUi, 'bookmarks', { readonly: true, label: 'bookmarks' }), 'Ctrl+1..9 saves the view into a slot, 1..9 recalls it.');
  const track = camF.addFolder({ title: 'Camera track', expanded: false });
  const recBtn = track.addButton({ title: 'Record track' });
  recBtn.on('click', () => app.toggleRecording());
  track.addBinding(camUi, 'loop', { label: 'loop playback' });
  track.addButton({ title: 'Play track' }).on('click', () => app.playTrack(camUi.loop));
  track.addButton({ title: 'Stop playback' }).on('click', () => cam.stopPlayback());
  track.addButton({ title: 'Export track JSON' }).on('click', () => app.exportTrack());
  track.addButton({ title: 'Import track JSON...' }).on('click', () => {
    void pickFiles('.json').then(async (f) => { if (f[0]) await app.importTrack(f[0]).catch((e: unknown) => { app.loading.start('Track'); app.loading.error(String(e)); }); });
  });

  // ---- Environment (the integrator consumes onEnvironmentParams) ----
  const env = app.envParams;
  eF.addButton({ title: 'Load HDRI (.hdr/.exr)...' }).on('click', () => { void pickFiles('.hdr,.exr').then((f) => (f.length ? app.loadFiles(f) : undefined)); });
  eF.addBinding(env, 'url', { label: 'HDRI URL' });
  eF.addButton({ title: 'Load HDRI URL' }).on('click', () => {
    try {
      const url = new URL(env.url.trim(), location.href).href;
      void app.loadEnvironment({ kind: 'url', url, name: url.split('/').pop() || url });
    } catch { app.loading.start('HDRI'); app.loading.error(`bad URL: ${env.url}`); }
  });
  const envChanged = () => app.envParamsChanged();
  eF.addBinding(env, 'strength', { label: 'strength', min: 0, max: 20, step: 0.01 }).on('change', envChanged);
  eF.addBinding(env, 'rotationDeg', { label: 'rotation Z (°)', min: -180, max: 180, step: 0.1 }).on('change', envChanged);
  eF.addBinding(env, 'tint', { label: 'tint', color: { type: 'float' } }).on('change', envChanged);
  eF.addBinding(env, 'visibleToCamera', { label: 'visible to camera' }).on('change', envChanged);
  // M3c env sampling: NEE on/off (≡ Cycles world sampling_method AUTOMATIC / NONE) and the importance resolution.
  const envS = eF.addFolder({ title: 'Sampling', expanded: false });
  tip(envS.addBinding(env, 'nee', { label: 'env NEE' }), 'Next-event estimation of the environment (Cycles world sampling AUTOMATIC); off = BSDF sampling only (NONE).')
    .on('change', envChanged);
  tip(envS.addBinding(env, 'importanceRes', { label: 'importance res', options: { 256: 256, 512: 512, 1024: 1024, 2048: 2048, 4096: 4096 } }),
    'Cap of the importance-map width (largest power of two ≤ min(map width, cap)).').on('change', envChanged);
  eF.addBinding(env, 'info', { readonly: true, multiline: true, rows: 3, label: 'info' });

  // ---- Render: integrator (inserted by integration.ts), frame controls, Output, Sampling, Advanced ----
  const r = app.render;
  const p = app.present;
  rF.addBinding(r, 'paused', { label: 'pause (P)' }).on('change', (e) => app.setPaused(e.value));
  rF.addButton({ title: 'Step one frame (.)' }).on('click', () => app.step());
  tip(rF.addButton({ title: 'Restart accumulation' }), 'Restarts the progressive accumulation; the ReSTIR temporal history is kept (Shift+R resets both).')
    .on('click', () => app.resetHistory());
  const out = rF.addFolder({ title: 'Output', expanded: true });
  out.addBinding(r, 'resolution', { label: 'internal res', options: { '540p': '540p', '720p': '720p', '1080p': '1080p', native: 'native' } })
    .on('change', (e) => app.setResolution(e.value));
  out.addBinding(p, 'exposureEV', { label: 'exposure (EV)', min: -10, max: 10, step: 0.1 });
  tip(out.addBinding(p, 'tonemap', { label: 'view transform', options: { 'Standard (Blender, exact)': 'standard', 'AgX (approx.)': 'agx', 'ACES fit (approx.)': 'aces', 'Raw (linear)': 'raw' } }),
    'Display transform. Standard matches Blender exactly; AgX and ACES are approximations.');
  out.addBinding(p, 'filter', { label: 'upscale filter', options: { 'bicubic (Catmull-Rom)': 'bicubic', bilinear: 'bilinear', nearest: 'nearest' } });
  // M8 dynamic resolution (docs/decisions/m8-perf.md §9)
  out.addBinding(r, 'dynamicResolution', { label: 'dynamic res' }).on('change', (e) => app.setDynamicResolution(e.value));
  out.addBinding(r, 'targetMs', { label: 'target (ms)', min: 8, max: 100, step: 1 }).on('change', (e) => app.setDynamicResolution(r.dynamicResolution, e.value));
  const hudUi = {
    get hud() { return app.hudVisible; },
    set hud(v: boolean) { app.hudVisible = v; app.hud.setVisible(v); },
  };
  out.addBinding(hudUi, 'hud', { label: 'HUD (H)' });
  const samp = rF.addFolder({ title: 'Sampling', expanded: false });
  // The app runs with i.i.d. jitter (integration.ts, plan §1.2); the binding shows the active value.
  tip(samp.addBinding(r, 'jitter', { label: 'pixel jitter', options: { 'i.i.d. random': 'iid', 'R2 sequence': 'r2', 'none (pixel centre)': 'none' } }),
    'Sub-pixel jitter of the camera rays. i.i.d. is the app default (and what validation uses).').on('change', () => app.resetHistory());
  tip(samp.addBinding(r, 'freezeSeed', { label: 'freeze seed' }), 'Reuse the same random seed every frame.');
  tip(samp.addBinding(r, 'freezeFrame', { label: 'freeze frame index' }), 'Keep the frame index (and with it the jitter) fixed.');
  const adv = rF.addFolder({ title: 'Advanced', expanded: false });
  adv.addBinding(r, 'colorFormat', { label: 'colour format', options: { rgba32float: 'rgba32float', rgba16float: 'rgba16float' } })
    .on('change', (e) => app.setColorFormat(e.value as ColorFormat));
  adv.addBinding(p, 'highlightNonFinite', { label: 'NaN/Inf in magenta' });
  tip(adv.addBinding(r, 'overlay', { label: 'overlay' }), 'Line overlay: axes, light gizmos, inspector paths.');

  // ---- Debug views ----
  const ds = app.debugSettings;
  const dbgUi = { category: '', probeClick: false, probe: '' };
  let catBinding: TpBinding | undefined;
  let viewBinding: TpBinding | undefined;
  const grouped = (): Map<string, DebugViewDef[]> => {
    const m = new Map<string, DebugViewDef[]>();
    for (const v of app.debug.registry.list()) { if (!m.has(v.group)) m.set(v.group, []); m.get(v.group)!.push(v); }
    return new Map([...m].sort(([a], [b]) => categoryRank(a) - categoryRank(b) || a.localeCompare(b)));
  };
  const buildViewList = () => {
    viewBinding?.dispose();
    const o: Record<string, number> = { 'off (beauty)': 0 };
    for (const v of grouped().get(dbgUi.category) ?? []) o[o[v.label] === undefined ? v.label : `${v.label} (${v.id})`] = v.id;
    viewBinding = dF.addBinding(ds, 'mode', { label: 'view', options: o, index: 1 });
    viewBinding.on('change', q((e) => app.selectDebugView(e.value as number)));
  };
  const buildLists = () => {
    const g = grouped();
    if (!g.has(dbgUi.category)) dbgUi.category = app.debug.activeView()?.group ?? g.keys().next().value ?? '';
    catBinding?.dispose();
    const o: Record<string, string> = {};
    for (const k of g.keys()) o[categoryName(k)] = k;
    catBinding = dF.addBinding(dbgUi, 'category', { label: 'category', options: o, index: 0 });
    catBinding.on('change', q((e) => {
      const first = grouped().get(e.value as string)?.[0];
      buildViewList();
      app.selectDebugView(first?.id ?? 0); // show the category at once
    }));
    buildViewList();
  };
  buildLists();
  app.debug.registry.onChange(buildLists);
  const legend = document.createElement('div');
  legend.style.cssText = 'font: 11px/1.5 ui-monospace, Menlo, monospace; padding: 2px 4px 4px; white-space: normal; color: var(--tp-monitor-foreground-color, #bbb);';
  addHtmlBlock(dF, legend);
  const tapOpts: Record<string, number> = {};
  for (const t of DEBUG_TAPS) tapOpts[t.label] = t.id;
  tip(dF.addBinding(ds, 'tap', { label: 'stage tap', options: tapOpts }), 'For tapped views (ReSTIR reservoir): which pipeline stage the view reads.');
  const map = dF.addFolder({ title: 'View mapping', expanded: false });
  map.addBinding(ds, 'rangeMin', { label: 'min' });
  map.addBinding(ds, 'rangeMax', { label: 'max' });
  map.addBinding(ds, 'log', { label: 'log scale' });
  map.addBinding(ds, 'colormap', { label: 'colormap', options: { viridis: 'viridis', turbo: 'turbo', signed: 'signed', grey: 'grey' } });
  map.addBinding(ds, 'abs', { label: 'absolute |v|' });
  map.addBinding(ds, 'highlightNonFinite', { label: 'NaN/Inf colour' });
  tip(map.addBinding(ds, 'split', { label: 'split with beauty' }), 'Left of the split: the view; right: the beauty image.');
  map.addBinding(ds, 'splitPos', { label: 'split position', min: 0, max: 1, step: 0.01 });
  const probe = dF.addFolder({ title: 'Pixel probe', expanded: false });
  tip(probe.addBinding(dbgUi, 'probeClick', { label: 'click = probe' }), 'Plain left click picks the probe pixel (Alt+click always does).');
  probe.addBinding(ds, 'probeEnabled', { label: 'probe panel' }).on('change', (e) => app.probePanel.setVisible(e.value));
  probe.addBinding(dbgUi, 'probe', { readonly: true, label: 'probe pixel' });

  // ---- Help ----
  const keys = document.createElement('div');
  keys.style.cssText = 'display: grid; grid-template-columns: max-content 1fr; gap: 1px 10px; padding: 4px; font: 11px/1.45 ui-monospace, Menlo, monospace; color: var(--tp-label-foreground-color, #bbb);';
  for (const [k, what] of KEYS_HELP) { const a = document.createElement('span'); a.textContent = k; const b = document.createElement('span'); b.textContent = what; keys.append(a, b); }
  addHtmlBlock(help, keys);

  let legendKey = '-';
  const updateLegend = () => {
    const v = app.debug.activeView();
    const key = v ? `${v.id}|${ds.tap}` : '';
    if (key === legendKey) return;
    legendKey = key;
    legend.replaceChildren();
    if (!v) { legend.append('beauty image (no debug view)'); return; }
    const head = document.createElement('div');
    head.textContent = `#${v.id} ${v.key}: ${v.kind}${v.tapped ? ` (stage tap: ${DEBUG_TAPS.find((t) => t.id === ds.tap)?.label ?? ds.tap})` : ''}, written by ${v.source}`;
    legend.append(head);
    if (v.description) { const d = document.createElement('div'); d.textContent = v.description; legend.append(d); }
    const codes = v.kind === 'code' ? legendCodes(v.id) : undefined;
    for (const c of codes ?? []) {
      const row = document.createElement('div');
      const sw = document.createElement('span');
      const [cr, cg, cb] = cmapCode(c).map((x) => Math.round(x * 255));
      sw.style.cssText = `display:inline-block;width:10px;height:10px;margin-right:6px;background:rgb(${cr},${cg},${cb});`;
      row.append(sw, `${c}  ${codeName(v.id, c)}`);
      legend.append(row);
    }
    if (v.kind === 'code' && !codes) legend.append('categorical colours (a hash of the code); the probe panel prints the value');
    if (v.kind !== 'code') legend.append(`default range [${v.range?.[0] ?? 0}, ${v.range?.[1] ?? 1}]${v.log ? ' log' : ''}, ${v.colormap ?? 'viridis'}`);
  };

  const refreshMonitors = () => {
    const s = app.scene;
    sceneUi.info = s
      ? `${s.name}\n${s.geometry.indices.length / 3} tris, ${s.materials.length} mats, ${s.textures.length} tex\n${s.lights.length} lights, ${s.cameras.length} cameras\ndiag ${app.sceneDiag.toFixed(2)} m, ${s.warnings.length} warnings`
      : 'no scene (test pattern)\ndrop a .glb/.gltf/.usd* or .hdr/.exr\nor use ?scene=<url>&env=<url>';
    camUi.speed = `${cam.speed.toFixed(3)} m/s (×1.25^${cam.wheelSteps})`;
    camUi.bookmarks = cam.bookmarks.map((b, i) => (b ? String(i) : '')).filter(Boolean).join(' ') || '(none)';
    recBtn.title = cam.recording ? `Stop recording (${cam.recording.keys.length} keys)` : 'Record track';
    const v = app.debug.activeView();
    if (v && v.group !== dbgUi.category) { dbgUi.category = v.group; buildViewList(); } // a view selected elsewhere
    updateLegend();
    dbgUi.probe = ds.probeEnabled ? `${ds.probePixel[0]}, ${ds.probePixel[1]}` : 'off';
  };
  refreshMonitors();
  return {
    pane,
    folders: { scene, camera: camF, environment: eF, lights, animation, render: rF, renderAdvanced: adv, restir, denoiser, debug: dF, probe, validation, help },
    refresh: () => {
      refreshMonitors();
      quiet = true;
      try { pane.refresh(); } finally { quiet = false; }
    },
    refreshMonitors,
    probeMode: () => dbgUi.probeClick,
  };
}
