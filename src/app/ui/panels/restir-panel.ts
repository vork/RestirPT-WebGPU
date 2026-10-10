// 'ReSTIR' folder (restir-api.md §1.2 WP-D; PLAN §3 modes, §6 M4 rows): the ReSTIR preset (interactive / unbiased /
// 2022 criteria / offline / initial only), temporal reuse, freeze / reset of the temporal history (M5; restir-temporal-
// api.md TD19–TD21), the M6 Enhanced feature toggles, the arena statistics (f_r, queue occupancy, SC histogram) and the
// pixel-inspector toggle. PT ↔ ReSTIR is the Render folder's integrator; the ReSTIR debug views (with their colour
// legends) and the stage tap live in the Debug views folder.
import { RESTIR_APP_MODES, type Renderer, type RestirAppMode, type RestirFeatureOverrides } from '../../../core/render/renderer.ts';
import { restirSettings } from '../../../core/render/restir/presets.ts';
import type { App } from '../../app.ts';
import { tip, type TpFolder } from '../tweakpane.ts';
import type { RestirInspector } from './restir-inspector.ts';

export interface RestirPanelHandle { folder: TpFolder; refresh(): void }

/** Short names of the app modes; the full preset description (RESTIR_APP_MODES) is the tooltip. */
const MODE_NAMES: Record<RestirAppMode, string> = {
  interactive: 'interactive', potato: 'Potato ReSTIR', unbiased: 'unbiased', criteria2022: '2022 criteria', offline: 'offline (S 32)', initial: 'initial candidates only',
};

export function addRestirPanel(app: App, r: Renderer, inspector: RestirInspector | undefined): RestirPanelHandle | undefined {
  const folders = app.panel?.folders;
  if (!folders) return undefined;
  const f = folders.restir;
  f.hidden = false;
  let quiet = false;
  const q = <T>(fn: (e: T) => void) => (e: T) => { if (!quiet) fn(e); };
  const ui = {
    mode: r.options.restirMode as string,
    temporal: r.options.temporal,
    stats: '',
    inspector: false,
  };
  const modes: Record<string, string> = {};
  for (const k of Object.keys(RESTIR_APP_MODES) as RestirAppMode[]) modes[MODE_NAMES[k] ?? k] = k;
  tip(f.addBinding(ui, 'mode', { label: 'preset', options: modes }), Object.values(RESTIR_APP_MODES).join('\n'))
    .on('change', q((e) => { void r.setOptions({ restirMode: e.value as RestirAppMode }).then(() => app.resetHistory()); }));
  // Temporal reuse (M5): a settings change, i.e. a config-hash reset (and a reallocation of the temporal buffers).
  const temporalToggle = f.addBinding(ui, 'temporal', { label: 'temporal reuse' }).on('change', q((e) => {
    void r.setOptions({ temporal: e.value }).then(() => app.resetHistory());
  }));
  tip(f.addBinding(app.render, 'freezeHistory', { label: 'freeze history' }), 'Suspend temporal reuse: every frame resets the temporal history.');
  f.addButton({ title: 'Reset temporal history (Shift+R)' }).on('click', () => app.resetTemporalHistory());
  // M6 (restir-m6-api.md MD13): the Enhanced features as toggles over the preset (pipeline variants: the first frame
  // after a change recompiles; the config hash changes, so the history resets)
  const feat = r.options.restirFeatures;
  const eff = () => restirSettings(undefined, r.restirSettings());
  const fx = { gauss: eff().pairing === 'gauss', ris: eff().risNee, dualMv: eff().dualMv, dupmap: eff().dupmap, rrMin: eff().rrMinBounces };
  const setF = (o: RestirFeatureOverrides) => { Object.assign(feat, o); void r.setOptions({ restirFeatures: feat }).then(() => app.resetHistory()); };
  const ef = f.addFolder({ title: 'Enhanced features', expanded: false });
  tip(ef.addBinding(fx, 'gauss', { label: 'σ 16 pairing maps' }), 'Spatial partners from reciprocal pairing maps with Gaussian (σ 16 px) offsets; off = disk partners.')
    .on('change', q((e) => setF({ pairing: e.value ? 'gauss' : 'disk' })));
  tip(ef.addBinding(fx, 'ris', { label: 'RIS-NEE light tiles' }), 'Light selection for NEE by RIS over light tiles.').on('change', q((e) => setF({ risNee: e.value })));
  tip(ef.addBinding(fx, 'dualMv', { label: 'dual motion vectors' }), 'On disocclusion, retry the temporal reprojection with the occluder\'s motion (needs temporal reuse).').on('change', q((e) => setF({ dualMv: e.value })));
  tip(ef.addBinding(fx, 'dupmap', { label: 'duplication map (biased)' }), 'Caps the temporal confidence where neighbouring pixels share a sample (biased; off by default since perf2 decision D3, so ReSTIR-interactive is unbiased; needs temporal reuse).')
    .on('change', q((e) => setF({ dupmap: e.value })));
  tip(ef.addBinding(fx, 'rrMin', { label: 'RR after bounce', options: { 1: 1, 2: 2, 3: 3 } }), 'Russian roulette at initial sampling only at vertices past this bounce (unbiased; 2 by default in ReSTIR-interactive, perf2 decision D1; lower = faster, noisier). Needs the mode\'s RR on.')
    .on('change', q((e) => setF({ rrMinBounces: Number(e.value) })));
  tip(f.addBinding(ui, 'stats', { readonly: true, multiline: true, rows: 7, label: 'status' }), 'Shift arena: replay fraction, queue occupancy, shift outcome histogram, error counters.');
  tip(f.addBinding(ui, 'inspector', { label: 'pixel inspector' }), 'Reservoir / shift / MIS dump and a 3D path overlay for the probe pixel (Alt+click a pixel).')
    .on('change', q((e) => {
      inspector?.setVisible(e.value);
      if (e.value) app.debugSettings.probeEnabled = true;
      app.panel?.refresh();
    }));

  const refresh = () => {
    quiet = true;
    try {
      ui.mode = r.options.restirMode;
      ui.temporal = eff().temporal;
      temporalToggle.disabled = r.options.restirMode === 'potato';
      { const e = eff(); fx.gauss = e.pairing === 'gauss'; fx.ris = e.risNee; fx.dualMv = e.dualMv; fx.dupmap = e.dupmap; fx.rrMin = e.rrMinBounces; }
      ui.inspector = !!inspector?.visible;
      ui.stats = r.options.renderMode === 'restir' ? (r.restirHud?.lines().join('\n') ?? r.restirError ?? 'compiling ...') : 'off (Render › integrator is not ReSTIR PT)';
      f.refresh();
    } finally { quiet = false; }
  };
  refresh();
  return { folder: f, refresh };
}
