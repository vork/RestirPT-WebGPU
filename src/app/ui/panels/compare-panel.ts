// Validation folder: 'Compare with Cycles' and 'Cycles reference render (dev)' (plan §5 M3a "Export for Cycles",
// §6 M3a "Compare"): dev-only export + headless Blender render through /api/reference with streamed progress, then the
// compare view modes (split, flip, relative error, t-map), batch capture for replicate statistics, and manual EXR loading.
import type { CompareMode, CompareView } from '../../compare/compare-view.ts';
import { parseExrRgba } from '../../compare/images.ts';
import type { ReferenceConfig, ReferenceProgress } from '../../compare/reference-flow.ts';
import { pickFiles } from '../../loader.ts';
import { tip, type TpFolder } from '../tweakpane.ts';

export interface ComparePanelHooks {
  /** Undefined in production builds (no Blender endpoint). */
  runReference?(cfg: ReferenceConfig, onProgress: (p: ReferenceProgress) => void): Promise<unknown>;
  config: ReferenceConfig;
}

/** Fills the (pre-created, hidden) Validation folder: "Compare with Cycles" (view modes, batches, EXR loading) and, in
 *  dev builds, "Cycles reference render (dev)". integration.ts appends "Export scene package (dev)". */
export function addComparePanel(parent: TpFolder, view: CompareView, hooks: ComparePanelHooks): { folder: TpFolder; refresh(): void } {
  parent.hidden = false;
  const f = parent.addFolder({ title: 'Compare with Cycles', expanded: true });
  let quiet = false;
  const q = <T>(fn: (e: T) => void) => (e: T) => { if (!quiet) fn(e); };
  const s = view.settings;
  const ui = { progress: '', stats: '' };
  let running = false;
  if (hooks.runReference) {
    const cfg = hooks.config;
    const ex = parent.addFolder({ title: 'Cycles reference render (dev)', expanded: false });
    ex.addBinding(cfg, 'spp', { label: 'spp', min: 1, max: 65536, step: 1 });
    ex.addBinding(cfg, 'seeds', { label: 'seeds (e.g. 0..3)' });
    ex.addBinding(cfg, 'maxBounces', { label: 'max bounces', min: 0, max: 64, step: 1 });
    tip(ex.addBinding(cfg, 'lightMode', { label: 'package light mode', options: { 'A: NEE only': 'A', 'B: MIS': 'B', 'A′ (Cycles: MIS)': 'A′' } }),
      'Light mode of the exported package / reference (independent of the render light mode).');
    ex.addBinding(cfg, 'frames', { label: 'frames', options: { 'current frame': 'current', all: 'all' } });
    ex.addBinding(cfg, 'width', { label: 'width (0 = internal)', min: 0, max: 8192, step: 1 });
    ex.addBinding(cfg, 'height', { label: 'height (0 = internal)', min: 0, max: 8192, step: 1 });
    ex.addBinding(cfg, 'device', { label: 'Cycles device', options: { GPU: 'GPU', CPU: 'CPU' } });
    const btn = tip(ex.addButton({ title: 'Export + render reference' }), 'Exports the scene package, renders it with headless Blender (/api/reference) and loads the EXRs into the compare view.');
    btn.on('click', () => {
      if (running) return;
      running = true;
      btn.disabled = true;
      void hooks.runReference!(cfg, (p) => {
        ui.progress = `${p.stage}${p.total ? ` ${p.done ?? 0}/${p.total}` : ''}: ${p.message}`;
        refresh();
      }).catch((e: unknown) => { ui.progress = `error: ${e instanceof Error ? e.message : String(e)}`; })
        .finally(() => { running = false; btn.disabled = false; refresh(); });
    });
    ex.addBinding(ui, 'progress', { readonly: true, multiline: true, rows: 3, label: 'progress' });
  }
  f.addBinding(s, 'mode', { label: 'view', options: { off: 'off', split: 'split', flip: 'flip', 'relative error': 'relerr', 't-map': 'tmap', 'ours only': 'ours', 'Cycles only': 'ref' } })
    .on('change', q((e) => view.setMode(e.value as CompareMode)));
  f.addBinding(s, 'split', { label: 'split position', min: 0, max: 1, step: 0.001 }).on('change', q(() => view.draw(true)));
  f.addBinding(s, 'exposureEV', { label: 'exposure (EV)', min: -10, max: 10, step: 0.1 }).on('change', q(() => view.draw(true)));
  f.addBinding(s, 'relErrMax', { label: 'rel. error max', min: 0.01, max: 4, step: 0.01 }).on('change', q(() => view.draw(true)));
  f.addBinding(s, 'tile', { label: 't-map tile (px)', min: 4, max: 128, step: 1 }).on('change', q(() => view.draw(true)));
  f.addBinding(s, 'flipMs', { label: 'flip period (ms)', min: 100, max: 5000, step: 10 });
  tip(f.addBinding(s, 'live', { label: 'live ours' }), 'Keep re-reading our image while the compare view is shown (off: keep the last read).');
  f.addButton({ title: 'Capture batch (restarts accumulation)' }).on('click', () => { void view.captureBatch().then(refresh); });
  f.addButton({ title: 'Clear batches' }).on('click', () => { view.clearBatches(); refresh(); });
  f.addButton({ title: 'Load reference EXR(s)...' }).on('click', () => {
    void pickFiles('.exr').then(async (files) => {
      if (!files.length) return;
      try {
        const imgs = await Promise.all(files.map(async (fl) => parseExrRgba(new Uint8Array(await fl.arrayBuffer()))));
        view.setReference(imgs, files.map((fl) => fl.name).join(', '));
      } catch (e) { ui.progress = `EXR load failed: ${e instanceof Error ? e.message : String(e)}`; }
      refresh();
    });
  });
  f.addBinding(ui, 'stats', { readonly: true, multiline: true, rows: 4, label: 'stats' });
  const refresh = () => {
    ui.stats = view.status ? `${view.status}\n${view.summaryText()}` : view.summaryText();
    quiet = true;
    try { f.refresh(); } finally { quiet = false; }
  };
  view.listeners.add(refresh);
  let t = 0;
  window.setInterval(() => { if (view.settings.mode !== 'off' && performance.now() - t > 1000) { t = performance.now(); refresh(); } }, 1000);
  refresh();
  return { folder: f, refresh };
}
