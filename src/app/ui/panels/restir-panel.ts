// 'ReSTIR' panel (restir-api.md §1.2 WP-D; PLAN §3 modes, §6 M4 rows): PT ↔ ReSTIR toggle, the ReSTIR mode
// (ReSTIR-unbiased / ReSTIR-2022-criteria / Offline / initial only), the stage tap, a ReSTIR view picker with a colour
// legend for code views, the arena statistics (f_r, queue occupancy, SC histogram) and the pixel-inspector toggle.
// The full view list stays in the Debug folder; this panel only offers the ReSTIR subset.
// M5 (T-D; restir-temporal-api.md §2.11, TD19–TD21, Changelog D-2): ReSTIR-interactive / ReSTIR-unbiased modes, the
// temporal on/off switch, "reset temporal history" and "freeze history" (temporal reuse suspended: every frame resets),
// the "after temporal" tap and the temporal views 480–497 with their legends; the arena box adds the temporal lines.
import { RESTIR_VIEWS, cmapCode, codeName, legendCodes } from '../../../core/render/restir/debug.ts';
import { RESTIR_APP_MODES, type Renderer, type RestirAppMode, type RestirFeatureOverrides } from '../../../core/render/renderer.ts';
import { restirSettings } from '../../../core/render/restir/presets.ts';
import type { App } from '../../app.ts';
import type { TpFolder } from '../tweakpane.ts';
import type { RestirInspector } from './restir-inspector.ts';

export interface RestirPanelHandle { folder: TpFolder; refresh(): void }

/** Stage taps offered for the ReSTIR views (debug-views.ts DEBUG_TAPS ids). */
const TAPS: Record<string, number> = { final: 0, 'after initial': 1, 'after temporal': 2, 'after spatial': 3 };

export function addRestirPanel(app: App, r: Renderer, inspector: RestirInspector | undefined, index?: number): RestirPanelHandle | undefined {
  const pane = app.panel?.pane;
  if (!pane) return undefined;
  const f = pane.addFolder({ title: 'ReSTIR', expanded: false, index });
  let quiet = false;
  const q = <T>(fn: (e: T) => void) => (e: T) => { if (!quiet) fn(e); };
  const ui = {
    get enabled() { return r.options.renderMode === 'restir'; },
    set enabled(v: boolean) { r.options.renderMode = v ? 'restir' : 'pt'; },
    mode: r.options.restirMode as string,
    temporal: r.options.temporal,
    view: 0,
    stats: '',
    inspector: false,
  };
  f.addBinding(ui, 'enabled', { label: 'ReSTIR (off = PT)' }).on('change', q(() => {
    if (ui.enabled) void r.prepareRestir();
    app.resetHistory();
    app.panel?.refresh();
  }));
  const modes: Record<string, string> = {};
  for (const [k, label] of Object.entries(RESTIR_APP_MODES)) modes[label] = k;
  f.addBinding(ui, 'mode', { label: 'mode', options: modes }).on('change', q((e) => {
    void r.setOptions({ restirMode: e.value as RestirAppMode }).then(() => app.resetHistory());
  }));
  // Temporal reuse (M5): a settings change, i.e. a config-hash reset (and a reallocation of the temporal buffers).
  f.addBinding(ui, 'temporal', { label: 'temporal reuse' }).on('change', q((e) => {
    void r.setOptions({ temporal: e.value }).then(() => app.resetHistory());
  }));
  // M6 (restir-m6-api.md MD13): the Enhanced features as toggles over the mode's preset (pipeline variants: the first
  // frame after a change recompiles; the config hash changes, so the history resets)
  const feat = r.options.restirFeatures;
  const eff = () => restirSettings(undefined, r.restirSettings());
  const fx = { gauss: eff().pairing === 'gauss', ris: eff().risNee, dualMv: eff().dualMv, dupmap: eff().dupmap };
  const setF = (o: RestirFeatureOverrides) => { Object.assign(feat, o); void r.setOptions({ restirFeatures: feat }).then(() => app.resetHistory()); };
  f.addBinding(fx, 'gauss', { label: 'σ 16 pairing maps' }).on('change', q((e) => setF({ pairing: e.value ? 'gauss' : 'disk' })));
  f.addBinding(fx, 'ris', { label: 'RIS-NEE light tiles' }).on('change', q((e) => setF({ risNee: e.value })));
  f.addBinding(fx, 'dualMv', { label: 'dual motion vectors' }).on('change', q((e) => setF({ dualMv: e.value })));
  f.addBinding(fx, 'dupmap', { label: 'duplication map (biased)' }).on('change', q((e) => setF({ dupmap: e.value })));
  f.addBinding(app.render, 'freezeHistory', { label: 'freeze history' });
  f.addButton({ title: 'Reset temporal history (Shift+R)' }).on('click', () => app.resetTemporalHistory());
  f.addBinding(app.debugSettings, 'tap', { label: 'stage tap', options: TAPS });
  const views: Record<string, number> = { 'beauty (off)': 0 };
  for (const v of RESTIR_VIEWS) views[`${v.group.replace('ReSTIR ', '')}: ${v.label}`] = v.id;
  f.addBinding(ui, 'view', { label: 'view', options: views }).on('change', q((e) => app.selectDebugView(e.value)));
  const legend = document.createElement('div');
  legend.className = 'restir-legend';
  legend.style.cssText = 'font: 11px ui-monospace, monospace; padding: 2px 8px 6px; line-height: 1.5;';
  f.element.append(legend);
  f.addBinding(ui, 'stats', { readonly: true, multiline: true, rows: 7, label: 'arena' });
  f.addBinding(ui, 'inspector', { label: 'pixel inspector' }).on('change', q((e) => {
    inspector?.setVisible(e.value);
    if (e.value) app.debugSettings.probeEnabled = true;
    app.panel?.refresh();
  }));
  f.addBlade({ view: 'separator' });

  let legendKey = -1;
  const updateLegend = () => {
    const id = app.debugSettings.mode;
    if (id === legendKey) return;
    legendKey = id;
    legend.replaceChildren();
    const v = RESTIR_VIEWS.find((x) => x.id === id);
    if (!v) return;
    const head = document.createElement('div');
    head.textContent = `${v.key}: ${v.kind}${v.tapped ? ' (stage tap)' : ''} from ${v.source}`;
    legend.append(head);
    const codes = v.kind === 'code' ? legendCodes(id) : undefined;
    for (const c of codes ?? []) {
      const row = document.createElement('div');
      const sw = document.createElement('span');
      const [cr, cg, cb] = cmapCode(c).map((x) => Math.round(x * 255));
      sw.style.cssText = `display:inline-block;width:10px;height:10px;margin-right:6px;background:rgb(${cr},${cg},${cb});`;
      row.append(sw, `${c}  ${codeName(id, c)}`);
      legend.append(row);
    }
    if (v.kind === 'code' && !codes) legend.append('categorical colours (hash of the code); the probe panel prints the value');
    if (v.kind !== 'code') legend.append(`range [${v.range?.[0] ?? 0}, ${v.range?.[1] ?? 1}]${v.log ? ' log' : ''} ${v.colormap ?? 'viridis'}`);
  };

  const refresh = () => {
    quiet = true;
    try {
      ui.mode = r.options.restirMode;
      ui.temporal = r.options.temporal;
      { const e = eff(); fx.gauss = e.pairing === 'gauss'; fx.ris = e.risNee; fx.dualMv = e.dualMv; fx.dupmap = e.dupmap; }
      ui.view = RESTIR_VIEWS.some((v) => v.id === app.debugSettings.mode) ? app.debugSettings.mode : 0;
      ui.stats = r.options.renderMode === 'restir' ? (r.restirHud?.lines().join('\n') ?? r.restirError ?? 'compiling ...') : 'ReSTIR off';
      updateLegend();
      f.refresh();
    } finally { quiet = false; }
  };
  refresh();
  return { folder: f, refresh };
}
