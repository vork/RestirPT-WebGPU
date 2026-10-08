// 'Denoiser' folder (M5.5, docs/decisions/denoiser.md §8–§10; PLAN §6 "Denoiser: variance; history; α"): the toggle
// (remembered per mode: default on in ReSTIR-interactive, off elsewhere, unavailable in ReSTIR-unbiased), the à-trous
// iteration count, α_min, σ_l, σ_a, the temporal resolve (DN-6), the gradient ramp λ₀/λ₁ and the camera-gradient
// option, and the GPU time of the separate timing submits. The denoiser debug views are in the Debug views folder.
import { DENOISER_DEFAULTS, DN_MAX_ITERATIONS } from '../../../core/render/denoise/layout.ts';
import type { Renderer } from '../../../core/render/renderer.ts';
import type { App } from '../../app.ts';
import { tip, type TpBinding, type TpFolder } from '../tweakpane.ts';

export interface DenoiserPanelHandle { folder: TpFolder; refresh(): void }

export function addDenoiserPanel(app: App, r: Renderer): DenoiserPanelHandle | undefined {
  const folders = app.panel?.folders;
  if (!folders) return undefined;
  const f = folders.denoiser;
  f.hidden = false;
  let quiet = false;
  const q = <T>(fn: (e: T) => void) => (e: T) => { if (!quiet) fn(e); };
  const s = { ...DENOISER_DEFAULTS, ...r.pendingDenoiserSettings };
  const ui = { on: r.options.denoise, status: '' };
  const toggle: TpBinding<boolean> = tip(f.addBinding(ui, 'on', { label: 'denoise (A-SVGF-lite)' }),
    'Remembered per mode: on by default in ReSTIR interactive, off elsewhere; unavailable in ReSTIR unbiased.')
    .on('change', q((e) => { r.setDenoise(e.value); app.panel?.refresh(); }));
  const apply = () => r.setDenoiserSettings({ ...s });
  f.addBinding(s, 'iterations', { label: 'à-trous iterations', min: 0, max: DN_MAX_ITERATIONS, step: 1 }).on('change', q(apply));
  tip(f.addBinding(s, 'alphaMin', { label: 'α min', min: 0.02, max: 1, step: 0.01 }), 'Lower bound of the temporal blend factor α (the weight of the new frame).').on('change', q(apply));
  f.addBinding(s, 'sigmaL', { label: 'σ luminance', min: 0.5, max: 16, step: 0.5 }).on('change', q(apply));
  f.addBinding(s, 'sigmaA', { label: 'σ albedo', min: 0, max: 0.5, step: 0.01 }).on('change', q(apply));
  tip(f.addBinding(s, 'resolve', { label: 'temporal resolve (TAA)' }), 'Final temporal resolve of the denoised output.').on('change', q(apply));
  const grad = f.addFolder({ title: 'Temporal gradient', expanded: false });
  tip(grad.addBinding(s, 'lambda0', { label: 'λ₀', min: 0, max: 0.5, step: 0.005 }), 'Gradient ramp start: below λ₀ a change is not detected.').on('change', q(apply));
  tip(grad.addBinding(s, 'lambda1', { label: 'λ₁', min: 0.01, max: 1, step: 0.005 }), 'Gradient ramp end: at λ₁ the history is dropped (α = 1).').on('change', q(apply));
  grad.addBinding(s, 'gradientOnCamera', { label: 'on camera motion' }).on('change', q(apply));
  f.addButton({ title: 'Reset denoiser history' }).on('click', () => r.denoiser?.reset());
  f.addBinding(ui, 'status', { readonly: true, multiline: true, rows: 4, label: 'status' });

  const refresh = () => {
    quiet = true;
    try {
      r.denoiseWanted();                                  // follow mode switches (per-mode toggle)
      ui.on = r.options.denoise;
      toggle.disabled = !r.denoiseAllowed;
      ui.status = r.denoiserHudLine().replace(/ {2}GPU/, '\nGPU').replace(/\), /g, '),\n');
      f.refresh();
    } finally { quiet = false; }
  };
  refresh();
  return { folder: f, refresh };
}
