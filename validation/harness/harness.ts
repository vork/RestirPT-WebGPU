// Browser side of the Chrome validation harness. run-chrome.ts drives it through window.__harness.
import { createGpuContext, describeContext, type GpuContext } from '../../src/core/gpu/device.ts';
import { encodePFM, orientationPattern } from '../../src/core/io/pfm.ts';
import { allocProbe, type AllocProbeOptions, type AllocProbeReport } from './alloc-probe.ts';
import { renderBatches, type RenderBatchesOptions, type RenderBatchesReport } from './batch-run.ts';
import { renderRestirBatches, type RenderRestirBatchesOptions, type RenderRestirBatchesReport } from './restir-batch-run.ts';
import { renderRestirChains, type RenderRestirChainsOptions, type RenderRestirChainsReport } from './restir-chain-run.ts';
import { exportAndUpload } from './export-package.ts';
import { renderDenoise, type RenderDenoiseOptions, type RenderDenoiseReport } from './denoise-run.ts';
import { renderPerf, type PerfOptions, type PerfReport } from './perf-run.ts';
import { fetchScenePackage, type ExportScenePackageOptions } from '../../src/core/scene/scene-package.ts';
import { denoiserT16State } from '../../src/core/render/denoise/registry.ts';

export interface SmokeReport {
  ok: boolean;
  userAgent: string;
  hasNavigatorGpu: boolean;
  /** requestAdapter() returned an adapter (independent of the hardware-Metal check). */
  adapterAvailable: boolean;
  isFallbackAdapter?: boolean;
  context?: Record<string, unknown>;
  error?: string;
  ms: number;
}

export interface Harness {
  smoke(): Promise<SmokeReport>;
  allocProbe(opts: AllocProbeOptions): Promise<AllocProbeReport>;
  orientationUpload(run: string): Promise<void>;
  log(run: string, entry: Record<string, unknown>): Promise<void>;
  /** M2: render `batches` × `spp` with a validation kernel; uploads batch_###.pfm + meta.json to validation/out/<run>/. */
  renderBatches(opts: RenderBatchesOptions): Promise<RenderBatchesReport>;
  /** M4 (restir-api.md §6.4): Stage-B ReSTIR batches (rung preset, frames per batch, ensemble members, plants). */
  renderRestirBatches(opts: RenderRestirBatchesOptions): Promise<RenderRestirBatchesReport>;
  /** M5 (restir-temporal-api.md §6.3–§6.5): temporal chains (ensemble atlas, per-test-frame ensemble.npz). */
  renderRestirChains(opts: RenderRestirChainsOptions): Promise<RenderRestirChainsReport>;
  /** M5.5 (docs/decisions/denoiser.md §11): the denoiser evaluation on the interactive renderer (not a validation
   *  readback: the renderer and its denoiser are destroyed before it returns). */
  renderDenoise(opts: RenderDenoiseOptions): Promise<RenderDenoiseReport>;
  /** M8 (docs/decisions/m8-perf.md §1): interactive frame / per-pass / denoiser timing (not a validation readback). */
  renderPerf(opts: PerfOptions): Promise<PerfReport>;
  /** M2: re-export a scene package (read from `packageUrl`) to validation/out/<run>/ (bridge round trip). */
  reexportPackage(packageUrl: string, run: string, overrides?: Partial<ExportScenePackageOptions>): Promise<{ files: string[]; sha256: string }>;
}

declare global {
  interface Window { __harness?: Harness }
}

let ctxPromise: Promise<GpuContext> | undefined;
function getContext(): Promise<GpuContext> {
  ctxPromise ??= createGpuContext(navigator.gpu, {
    requireHardwareMetal: true,
    label: 'harness',
    onLost: (i) => { console.error(`[harness] device lost (${i.reason}): ${i.message}`); ctxPromise = undefined; },
    onUncapturedError: (m) => console.error(`[harness] uncaptured: ${m}`),
  });
  ctxPromise.catch(() => { ctxPromise = undefined; });
  return ctxPromise;
}

async function post(path: string, body: BodyInit, contentType: string): Promise<void> {
  const r = await fetch(path, { method: 'POST', body, headers: { 'content-type': contentType } });
  if (r.status !== 204) throw new Error(`${path}: HTTP ${r.status} ${await r.text()}`);
}

export async function upload(run: string, name: string, bytes: Uint8Array): Promise<void> {
  const q = new URLSearchParams({ run, name });
  await post(`/__harness/upload?${q}`, bytes as Uint8Array<ArrayBuffer>, 'application/octet-stream');
}

/**
 * T16 "no denoiser in validation readbacks" (PLAN §7.4; docs/decisions/denoiser.md DN4, §11): the denoiser exists only
 * inside the interactive Renderer, which no validation runner builds. Every validation entry point runs through this
 * wrapper: a live denoiser on the page before or after the run fails it, and the measured state replaces the run's
 * `t16.denoiser` in meta.json (PT references gain a t16 block). The gates require 'none'.
 */
async function t16Denoiser<R extends { ok: boolean; run: string; meta: Record<string, unknown>; errors: string[] }>(run: string, f: () => Promise<R>): Promise<R> {
  const before = denoiserT16State();
  const r = await f();
  const after = denoiserT16State();
  const state = before === 'none' && after === 'none' ? 'none' : `ACTIVE (before: ${before}; after: ${after})`;
  const t16 = (r.meta.t16 ?? {}) as Record<string, unknown>;
  r.meta.t16 = { ...t16, denoiser: state, denoiserCheck: 'harness.ts: live Denoiser objects on the validation page before and after the run (render/denoise/registry.ts)' };
  if (state !== 'none') {
    r.errors.push(`T16: a denoiser is live on the validation page (${state})`);
    r.ok = false;
    r.meta.ok = false;
    r.meta.errors = r.errors;
  }
  await upload(run, 'meta.json', new TextEncoder().encode(JSON.stringify(r.meta, null, 1)));
  return r;
}

const harness: Harness = {
  async smoke() {
    const t0 = performance.now();
    const rep: SmokeReport = {
      ok: false,
      userAgent: navigator.userAgent,
      hasNavigatorGpu: !!navigator.gpu,
      adapterAvailable: false,
      ms: 0,
    };
    try {
      if (!navigator.gpu) throw new Error('navigator.gpu missing');
      const probe = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      rep.adapterAvailable = probe !== null;
      if (!probe) throw new Error('requestAdapter() returned null');
      rep.isFallbackAdapter = (probe.info as GPUAdapterInfo & { isFallbackAdapter?: boolean }).isFallbackAdapter === true;
      const ctx = await getContext();
      rep.context = describeContext(ctx);
      rep.ok = true;
    } catch (e) {
      rep.error = e instanceof Error ? e.message : String(e);
    }
    rep.ms = performance.now() - t0;
    return rep;
  },

  async allocProbe(opts) {
    const ctx = await getContext();
    return allocProbe(ctx.device, opts);
  },

  async orientationUpload(run) {
    await upload(run, 'orientation.pfm', encodePFM(orientationPattern(64, 48)));
  },

  async renderBatches(opts) {
    const ctx = await getContext();
    return t16Denoiser(opts.run, () => renderBatches(ctx, opts));
  },

  async renderRestirBatches(opts) {
    const ctx = await getContext();
    return t16Denoiser(opts.run, () => renderRestirBatches(ctx, opts));
  },

  async renderRestirChains(opts) {
    const ctx = await getContext();
    return t16Denoiser(opts.run, () => renderRestirChains(ctx, opts));
  },

  async renderDenoise(opts) {
    const ctx = await getContext();
    return renderDenoise(ctx, opts);
  },

  async renderPerf(opts) {
    const ctx = await getContext();
    return renderPerf(ctx, opts);
  },

  async reexportPackage(packageUrl, run, overrides = {}) {
    const p = await fetchScenePackage(packageUrl);
    const r = await exportAndUpload(p.scene, {
      camera: p.camera, render: p.render, lightMode: p.lightMode, flatShaded: p.flatShaded, name: p.json.name,
      frames: p.frames, envSampling: p.json.env?.sampling, ...overrides,
    }, run);
    return { files: r.files, sha256: r.sha256 };
  },

  async log(run, entry) {
    await post(`/__harness/log?${new URLSearchParams({ run })}`, JSON.stringify(entry), 'application/json');
  },
};

window.__harness = harness;
document.getElementById('status')!.textContent = 'harness ready';
