// Tweakpane UI: Scene, Camera, Render, Environment, Debug folders + HUD toggle (plan §5 M1).
import { DEBUG_TAPS } from '../../core/render/debug-views.ts';
import type { App, ColorFormat } from '../app.ts';
import { DEG } from '../camera-math.ts';
import { pickFiles } from '../loader.ts';
import { createPane, type TpBinding, type TpPane } from './tweakpane.ts';

export interface PanelHandle {
  pane: TpPane;
  refresh(): void;
  refreshMonitors(): void;
  /** 'Click to probe' mode: plain LMB selects the probe pixel. */
  probeMode(): boolean;
}

export function buildPanel(app: App): PanelHandle {
  const host = document.createElement('div');
  host.className = 'panel';
  document.body.append(host);
  const pane = createPane(host, 'ReSTIR PT');

  // ---- Scene ----
  const scene = pane.addFolder({ title: 'Scene' });
  const sceneUi = { url: '', info: 'no scene (test pattern)' };
  scene.addButton({ title: 'Open files...' }).on('click', () => { void pickFiles().then((f) => (f.length ? app.loadFiles(f) : undefined)); });
  scene.addBinding(sceneUi, 'url', { label: 'URL' });
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
  const camF = pane.addFolder({ title: 'Camera' });
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
  camF.addBinding(camUi, 'fovDeg', { label: 'vfov (deg)', min: 5, max: 120, step: 0.1 });
  camF.addBinding(camUi, 'speed', { readonly: true, label: 'speed' });
  camF.addBinding(camUi, 'sensitivity', { label: 'sensitivity', min: 0.2, max: 10, step: 0.1 });
  camF.addBinding(camUi, 'invertY', { label: 'invert Y' });
  camF.addButton({ title: 'Home (reset)' }).on('click', () => cam.reset());
  camF.addBinding(camUi, 'fileCamera', { label: 'file camera', min: 0, max: 16, step: 1 });
  camF.addButton({ title: 'Use file camera' }).on('click', () => {
    if (!app.useFileCamera(camUi.fileCamera)) { app.loading.start('Camera'); app.loading.error('The scene has no camera with that index.'); }
  });
  camF.addBinding(camUi, 'bookmarks', { readonly: true, label: 'bookmarks' });
  const recBtn = camF.addButton({ title: 'Record track' });
  recBtn.on('click', () => app.toggleRecording());
  camF.addBinding(camUi, 'loop', { label: 'loop playback' });
  camF.addButton({ title: 'Play track' }).on('click', () => app.playTrack(camUi.loop));
  camF.addButton({ title: 'Stop playback' }).on('click', () => cam.stopPlayback());
  camF.addButton({ title: 'Export track JSON' }).on('click', () => app.exportTrack());
  camF.addButton({ title: 'Import track JSON...' }).on('click', () => {
    void pickFiles('.json').then(async (f) => { if (f[0]) await app.importTrack(f[0]).catch((e: unknown) => { app.loading.start('Track'); app.loading.error(String(e)); }); });
  });
  const help = { keys: 'RMB drag: look · F: fly mode · WASD, E up, Q down\nwheel: speed · Shift x4 · Ctrl x0.25 · Home\nCtrl+1..9 save, 1..9 recall · P pause · . step\nAlt+click: probe pixel · H: HUD' };
  camF.addBinding(help, 'keys', { readonly: true, multiline: true, rows: 4, label: 'keys' });

  // ---- Render ----
  const rF = pane.addFolder({ title: 'Render' });
  const r = app.render;
  const p = app.present;
  rF.addBinding(r, 'resolution', { label: 'internal res', options: { '540p': '540p', '720p': '720p', '1080p': '1080p', native: 'native' } })
    .on('change', (e) => app.setResolution(e.value));
  rF.addBinding(r, 'colorFormat', { label: 'color format', options: { rgba32float: 'rgba32float', rgba16float: 'rgba16float' } })
    .on('change', (e) => app.setColorFormat(e.value as ColorFormat));
  rF.addBinding(p, 'filter', { label: 'upscale', options: { bilinear: 'bilinear', nearest: 'nearest' } });
  rF.addBinding(p, 'exposureEV', { label: 'exposure (EV)', min: -10, max: 10, step: 0.1 });
  rF.addBinding(p, 'tonemap', { label: 'view transform', options: { 'Standard (Blender, exact)': 'standard', 'AgX (approx)': 'agx', 'ACES fit (approx)': 'aces', 'Raw (linear)': 'raw' } });
  rF.addBinding(p, 'highlightNonFinite', { label: 'NaN/Inf magenta' });
  rF.addBinding(r, 'jitter', { label: 'jitter', options: { 'R2 (interactive)': 'r2', 'i.i.d. (validation)': 'iid', 'none (centre)': 'none' } })
    .on('change', () => app.resetHistory());
  rF.addBinding(r, 'paused', { label: 'pause (P)' }).on('change', (e) => app.setPaused(e.value));
  rF.addButton({ title: 'Step (.)' }).on('click', () => app.step());
  rF.addBinding(r, 'freezeSeed', { label: 'freeze seed' });
  rF.addBinding(r, 'freezeFrame', { label: 'freeze frame' });
  rF.addButton({ title: 'Reset history (Shift+R)' }).on('click', () => app.resetHistory());
  rF.addBinding(r, 'overlay', { label: 'overlay' });

  // ---- Environment (the integrator consumes onEnvironmentParams) ----
  const eF = pane.addFolder({ title: 'Environment', expanded: false });
  const env = app.envParams;
  eF.addButton({ title: 'Load HDRI (.hdr/.exr)...' }).on('click', () => { void pickFiles('.hdr,.exr').then((f) => (f.length ? app.loadFiles(f) : undefined)); });
  eF.addBinding(env, 'url', { label: 'URL' });
  eF.addButton({ title: 'Load HDRI URL' }).on('click', () => {
    try {
      const url = new URL(env.url.trim(), location.href).href;
      void app.loadEnvironment({ kind: 'url', url, name: url.split('/').pop() || url });
    } catch { app.loading.start('HDRI'); app.loading.error(`bad URL: ${env.url}`); }
  });
  const envChanged = () => app.envParamsChanged();
  eF.addBinding(env, 'strength', { min: 0, max: 20, step: 0.01 }).on('change', envChanged);
  eF.addBinding(env, 'rotationDeg', { label: 'rotation Z (deg)', min: -180, max: 180, step: 0.1 }).on('change', envChanged);
  eF.addBinding(env, 'tint', { color: { type: 'float' } }).on('change', envChanged);
  eF.addBinding(env, 'visibleToCamera', { label: 'visible to camera' }).on('change', envChanged);
  // M3c env sampling: NEE on/off (≡ Cycles world sampling_method AUTOMATIC / NONE) and the importance resolution.
  eF.addBinding(env, 'nee', { label: 'env NEE' }).on('change', envChanged);
  eF.addBinding(env, 'importanceRes', { label: 'importance res', options: { 256: 256, 512: 512, 1024: 1024, 2048: 2048, 4096: 4096 } }).on('change', envChanged);
  eF.addBinding(env, 'info', { readonly: true, multiline: true, rows: 3, label: 'info' });

  // ---- Debug ----
  const dF = pane.addFolder({ title: 'Debug' });
  const ds = app.debugSettings;
  const dbgUi = { probeClick: false, info: '', probe: '' };
  let viewBinding: TpBinding | undefined;
  const viewOptions = (): Record<string, number> => {
    const o: Record<string, number> = { 'beauty (off)': 0 };
    for (const v of app.debug.registry.list()) o[`${v.group}: ${v.label}`] = v.id;
    return o;
  };
  const buildViewList = () => {
    viewBinding?.dispose();
    viewBinding = dF.addBinding(ds, 'mode', { label: 'view', options: viewOptions(), index: 0 });
    viewBinding.on('change', (e) => app.selectDebugView(e.value as number));
  };
  buildViewList();
  app.debug.registry.onChange(buildViewList);
  const tapOpts: Record<string, number> = {};
  for (const t of DEBUG_TAPS) tapOpts[t.label] = t.id;
  dF.addBinding(ds, 'tap', { label: 'stage tap', options: tapOpts });
  dF.addBinding(dbgUi, 'info', { readonly: true, label: 'view info' });
  dF.addBinding(ds, 'rangeMin', { label: 'min' });
  dF.addBinding(ds, 'rangeMax', { label: 'max' });
  dF.addBinding(ds, 'log', { label: 'log scale' });
  dF.addBinding(ds, 'colormap', { label: 'colormap', options: { viridis: 'viridis', turbo: 'turbo', signed: 'signed', grey: 'grey' } });
  dF.addBinding(ds, 'abs', { label: '|v|' });
  dF.addBinding(ds, 'highlightNonFinite', { label: 'NaN/Inf colour' });
  dF.addBinding(ds, 'split', { label: 'A|B split' });
  dF.addBinding(ds, 'splitPos', { label: 'split pos', min: 0, max: 1, step: 0.01 });
  dF.addBinding(dbgUi, 'probeClick', { label: 'click = probe' });
  dF.addBinding(ds, 'probeEnabled', { label: 'probe on' }).on('change', (e) => app.probePanel.setVisible(e.value));
  dF.addBinding(dbgUi, 'probe', { readonly: true, label: 'probe pixel' });

  // ---- HUD ----
  const hudUi = {
    get hud() { return app.hudVisible; },
    set hud(v: boolean) { app.hudVisible = v; app.hud.setVisible(v); },
  };
  pane.addBinding(hudUi, 'hud', { label: 'HUD (H)' });

  const refreshMonitors = () => {
    const s = app.scene;
    sceneUi.info = s
      ? `${s.name}\n${s.geometry.indices.length / 3} tris, ${s.materials.length} mats, ${s.textures.length} tex\n${s.lights.length} lights, ${s.cameras.length} cameras\ndiag ${app.sceneDiag.toFixed(2)} m, ${s.warnings.length} warnings`
      : 'no scene (test pattern)\ndrop a .glb/.gltf/.usd* or .hdr/.exr\nor use ?scene=<url>&env=<url>';
    camUi.speed = `${cam.speed.toFixed(3)} m/s (x1.25^${cam.wheelSteps})`;
    camUi.bookmarks = cam.bookmarks.map((b, i) => (b ? String(i) : '')).filter(Boolean).join(' ') || '(none)';
    recBtn.title = cam.recording ? `Stop recording (${cam.recording.keys.length} keys)` : 'Record track';
    const v = app.debug.activeView();
    dbgUi.info = v ? `${v.kind} from ${v.source}${v.tapped ? ' (tapped)' : ''}` : '-';
    dbgUi.probe = ds.probeEnabled ? `${ds.probePixel[0]}, ${ds.probePixel[1]}` : 'off';
  };
  refreshMonitors();
  return {
    pane,
    refresh: () => { refreshMonitors(); pane.refresh(); },
    refreshMonitors,
    probeMode: () => dbgUi.probeClick,
  };
}

