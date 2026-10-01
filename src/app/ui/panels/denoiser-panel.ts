// 'Denoiser' panel (M5.5, docs/decisions/denoiser.md §8–§10; PLAN §6 "Denoiser: variance; history; α"): the toggle
// (remembered per mode: default on in ReSTIR-interactive, off elsewhere, unavailable in ReSTIR-unbiased), the à-trous
// iteration count, α_min, the gradient ramp λ₀/λ₁, σ_l, the camera-gradient option, a picker for the denoiser views
// (variance, history length, α, λ, demodulated input, integrated colour, reprojection, gradient pairs, à-trous levels)
// and the GPU time of the separate timing submits.
import { DENOISER_DEFAULTS, DENOISER_VIEWS, DN_MAX_ITERATIONS } from '../../../core/render/denoise/layout.ts';
import type { Renderer } from '../../../core/render/renderer.ts';
import type { App } from '../../app.ts';
import type { TpBinding, TpFolder } from '../tweakpane.ts';

export interface DenoiserPanelHandle { folder: TpFolder; refresh(): void }

export function addDenoiserPanel(app: App, r: Renderer, index?: number): DenoiserPanelHandle | undefined {
  const pane = app.panel?.pane;
  if (!pane) return undefined;
  const f = pane.addFolder({ title: 'Denoiser', expanded: false, index });
  let quiet = false;
  const q = <T>(fn: (e: T) => void) => (e: T) => { if (!quiet) fn(e); };
  const s = { ...DENOISER_DEFAULTS, ...r.pendingDenoiserSettings };
  const ui = { on: r.options.denoise, view: 0, status: '' };
  const toggle: TpBinding<boolean> = f.addBinding(ui, 'on', { label: 'denoise (A-SVGF-lite)' }).on('change', q((e) => { r.setDenoise(e.value); app.panel?.refresh(); }));
  const apply = () => r.setDenoiserSettings({ ...s });
  f.addBinding(s, 'iterations', { label: 'à-trous iterations', min: 0, max: DN_MAX_ITERATIONS, step: 1 }).on('change', q(apply));
  f.addBinding(s, 'alphaMin', { label: 'α min', min: 0.02, max: 1, step: 0.01 }).on('change', q(apply));
  f.addBinding(s, 'lambda0', { label: 'gradient λ₀', min: 0, max: 0.5, step: 0.005 }).on('change', q(apply));
  f.addBinding(s, 'lambda1', { label: 'gradient λ₁', min: 0.01, max: 1, step: 0.005 }).on('change', q(apply));
  f.addBinding(s, 'sigmaL', { label: 'σ luminance', min: 0.5, max: 16, step: 0.5 }).on('change', q(apply));
  f.addBinding(s, 'gradientOnCamera', { label: 'gradient on camera motion' }).on('change', q(apply));
  f.addButton({ title: 'Reset denoiser history' }).on('click', () => r.denoiser?.reset());
  const views: Record<string, number> = { 'beauty (off)': 0 };
  for (const v of DENOISER_VIEWS) views[v.label] = v.id;
  f.addBinding(ui, 'view', { label: 'view', options: views }).on('change', q((e) => app.selectDebugView(e.value)));
  f.addBinding(ui, 'status', { readonly: true, multiline: true, rows: 4, label: 'status' });

  const refresh = () => {
    quiet = true;
    try {
      r.denoiseWanted();                                  // follow mode switches (per-mode toggle)
      ui.on = r.options.denoise;
      toggle.disabled = !r.denoiseAllowed;
      ui.view = DENOISER_VIEWS.some((v) => v.id === app.debugSettings.mode) ? app.debugSettings.mode : 0;
      ui.status = r.denoiserHudLine().replace(/ {2}GPU/, '\nGPU').replace(/\), /g, '),\n');
      f.refresh();
    } finally { quiet = false; }
  };
  refresh();
  return { folder: f, refresh };
}
