// Env sampling debug views (plan §6 "M3c Env sampling"; WGSL: shaders/passes/env-debug.wgsl). Drawn through the debug AOV
// plane so the debug resolve pass does the false colour; the splat histogram's χ² is computed here for the HUD.
// Storage buffers (≤ 10): scene 5 + records + envDbg + targetUV + debug buffer = 9.
import { composeWgsl, createCheckedShaderModule } from '../gpu/wgsl-composer.ts';
import { shaderSources } from '../shaders/index.ts';
import { chi2Merged } from './chi2.ts';
import type { DebugViewDef } from './debug-views.ts';
import { envBindGroupEntries, envBindGroupLayoutEntries, envDefines, type EnvGpuResources } from './env-gpu.ts';
import { LUT_RECORDS_BASE, type LightsGpu } from './lights-gpu.ts';
import { lutDefines } from './luts/lut-layout.ts';
import type { SceneGpu } from './scene-gpu.ts';

export const ENV_DBG = { IMPORTANCE: 310, RATIO: 311, SPLAT: 312, ESCAPE: 313, W1: 314, W2: 315 } as const;
export const isEnvDebugView = (mode: number): boolean => mode >= 310 && mode <= 315;
const PIXEL_WORDS = 6;
const PARAMS_SIZE = 32;
const SPLAT_THREADS = 4096, SPLAT_K = 256;

export const ENV_DEBUG_VIEWS: DebugViewDef[] = [
  { id: ENV_DBG.IMPORTANCE, key: 'env.pdfUV', label: 'Env importance: pdf_uv (log)', group: 'Env sampling', source: 'env-debug', kind: 'scalar', range: [1e-2, 1e2], log: true, colormap: 'turbo',
    description: 'realized density in (u, v) of the env cell seen along the camera ray (env space; geometry ignored)' },
  { id: ENV_DBG.RATIO, key: 'env.realizedOverTarget', label: 'Env importance: realized / target − 1', group: 'Env sampling', source: 'env-debug', kind: 'scalar', range: [-0.01, 0.01], colormap: 'signed',
    description: 'u16 alias quantization of the floored kernel weights (≈ 0)' },
  { id: ENV_DBG.SPLAT, key: 'env.splat', label: 'Env NEE splat / expected − 1 (χ² in HUD)', group: 'Env sampling', source: 'env-debug', kind: 'scalar', range: [-0.2, 0.2], colormap: 'signed',
    description: 'histogram of NEE_ENV samples over the env cells divided by the realized-pdf expectation, minus 1' },
  { id: ENV_DBG.ESCAPE, key: 'env.escape', label: 'Env escape fraction (primary hit)', group: 'Env sampling', source: 'env-debug', kind: 'scalar', range: [0, 1], colormap: 'viridis' },
  { id: ENV_DBG.W1, key: 'env.w1', label: 'NEE-env ω1 (primary hit)', group: 'Env sampling', source: 'env-debug', kind: 'scalar', range: [0, 1], colormap: 'viridis' },
  { id: ENV_DBG.W2, key: 'env.w2', label: 'BSDF-env ω2 (primary hit)', group: 'Env sampling', source: 'env-debug', kind: 'scalar', range: [0, 1], colormap: 'viridis' },
];

export interface EnvDebugFrame {
  mode: number;
  env: EnvGpuResources;
  lights: LightsGpu;
  frameUniforms: GPUBuffer;
  width: number;
  height: number;
  debugGroup: GPUBindGroup;
  /** Restart the per-pixel running means (camera moved, history reset, view switched). */
  reset: boolean;
}

export class EnvDebugPass {
  private readonly params: GPUBuffer;
  private buf: GPUBuffer | undefined;
  private target: GPUBuffer | undefined;
  private bufKey = '';
  private bufTable: unknown;
  private g0: GPUBindGroup | undefined;
  private g0Key = '';
  private g0Env: EnvGpuResources | undefined;
  private g0Frame: GPUBuffer | undefined;
  private g0Lights: LightsGpu | undefined;
  private g2: GPUBindGroup | undefined;
  private readonly g1: GPUBindGroup;
  private splatTotal = 0;
  private splatSeed = 1;
  private cells = 0;
  private lastMode = -1;
  /** Last χ² of the splat histogram (HUD). */
  chi2: { chi2: number; dof: number; p: number; cells: number; total: number } | undefined;
  private chi2Busy = false;

  private constructor(
    readonly device: GPUDevice,
    private readonly layouts: GPUBindGroupLayout[],
    private readonly splat: GPUComputePipeline,
    private readonly view: GPUComputePipeline,
    scene: SceneGpu,
  ) {
    this.params = device.createBuffer({ label: 'env-debug.params', size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.g1 = device.createBindGroup({ label: 'env-debug-g1', layout: layouts[1], entries: scene.bindGroupEntries() });
  }

  static async create(device: GPUDevice, scene: SceneGpu, debugLayout: GPUBindGroupLayout, opts: { features?: Set<string>; wgslLanguageFeatures?: Set<string> } = {}): Promise<EnvDebugPass> {
    const c = GPUShaderStage.COMPUTE;
    const g0 = device.createBindGroupLayout({ label: 'env-debug-g0', entries: [
      { binding: 0, visibility: c, buffer: { type: 'uniform' } },
      ...envBindGroupLayoutEntries(1, c),
      { binding: 4, visibility: c, buffer: { type: 'uniform', minBindingSize: PARAMS_SIZE } },
      { binding: 5, visibility: c, buffer: { type: 'uniform' } },
      { binding: 6, visibility: c, buffer: { type: 'read-only-storage' } },
    ] });
    const g1 = device.createBindGroupLayout({ label: 'env-debug-g1', entries: scene.layoutEntries(c) });
    const g2 = device.createBindGroupLayout({ label: 'env-debug-g2', entries: [
      { binding: 0, visibility: c, buffer: { type: 'storage' } }, { binding: 1, visibility: c, buffer: { type: 'read-only-storage' } }] });
    const shader = composeWgsl('passes/env-debug.wgsl', {
      sources: shaderSources,
      defines: { ...scene.defines(1), ...envDefines(0, 1), LIGHTS_GROUP: 0, LIGHTS_BINDING: 5, ...lutDefines({ base: LUT_RECORDS_BASE, recordsKind: 'u32' }) },
      features: opts.features, wgslLanguageFeatures: opts.wgslLanguageFeatures,
    });
    const module = await createCheckedShaderModule(device, shader, 'env-debug');
    const layout = device.createPipelineLayout({ label: 'env-debug', bindGroupLayouts: [g0, g1, g2, debugLayout] });
    const [splat, view] = await Promise.all(['env_splat', 'env_view'].map((e) => device.createComputePipelineAsync({ label: `env-debug-${e}`, layout, compute: { module, entryPoint: e } })));
    return new EnvDebugPass(device, [g0, g1, g2], splat, view, scene);
  }

  private ensureBuffers(f: EnvDebugFrame): void {
    const t = f.lights.state.env;
    const cells = t ? t.Wm * t.Hm : 0;
    const key = `${cells}|${f.width}x${f.height}|${f.lights.version}`;
    if (key === this.bufKey && this.buf && this.bufTable === t) return;
    this.bufTable = t;
    this.buf?.destroy(); this.target?.destroy();
    this.cells = cells;
    this.buf = this.device.createBuffer({ label: 'env-debug.buf', size: 4 * Math.max(4, cells + f.width * f.height * PIXEL_WORDS), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.target = this.device.createBuffer({ label: 'env-debug.targetUV', size: 4 * Math.max(4, cells), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    if (t) this.device.queue.writeBuffer(this.target, 0, t.targetUV);
    this.g2 = this.device.createBindGroup({ label: 'env-debug-g2', layout: this.layouts[2], entries: [{ binding: 0, resource: { buffer: this.buf } }, { binding: 1, resource: { buffer: this.target } }] });
    this.splatTotal = 0;
    this.chi2 = undefined;
    this.bufKey = key;
  }

  encode(encoder: GPUCommandEncoder, f: EnvDebugFrame): void {
    if (!isEnvDebugView(f.mode)) return;
    this.ensureBuffers(f);
    const g0Key = `${f.lights.version}`;
    if (!this.g0 || this.g0Key !== g0Key || this.g0Env !== f.env || this.g0Frame !== f.frameUniforms || this.g0Lights !== f.lights) {
      this.g0Env = f.env; this.g0Frame = f.frameUniforms; this.g0Lights = f.lights;
      this.g0 = this.device.createBindGroup({ label: 'env-debug-g0', layout: this.layouts[0], entries: [
        { binding: 0, resource: { buffer: f.frameUniforms } }, ...envBindGroupEntries(f.env, 1),
        { binding: 4, resource: { buffer: this.params } },
        { binding: 5, resource: { buffer: f.lights.params } }, { binding: 6, resource: { buffer: f.lights.records } }] });
      this.g0Key = g0Key;
    }
    const reset = f.reset || this.lastMode !== f.mode;
    this.lastMode = f.mode;
    const splat = f.mode === ENV_DBG.SPLAT && this.cells > 0;
    if (splat) this.splatTotal += SPLAT_THREADS * SPLAT_K;
    const buf = new ArrayBuffer(PARAMS_SIZE);
    const u = new Uint32Array(buf), fl = new Float32Array(buf);
    u.set([SPLAT_THREADS, SPLAT_K, this.splatSeed++, reset ? 1 : 0]);
    fl[4] = this.splatTotal;
    this.device.queue.writeBuffer(this.params, 0, buf);
    const pass = encoder.beginComputePass({ label: 'env-debug' });
    pass.setBindGroup(0, this.g0);
    pass.setBindGroup(1, this.g1);
    pass.setBindGroup(2, this.g2!);
    pass.setBindGroup(3, f.debugGroup);
    if (splat) { pass.setPipeline(this.splat); pass.dispatchWorkgroups(SPLAT_THREADS / 256); }
    pass.setPipeline(this.view);
    pass.dispatchWorkgroups(Math.ceil(f.width / 8), Math.ceil(f.height / 8));
    pass.end();
  }

  /** Read the splat histogram back and compute its χ² against the realized pdf (cells merged to ≥ 50 expected). */
  async updateChi2(lights: LightsGpu): Promise<void> {
    const t = lights.state.env;
    if (this.chi2Busy || !t || !this.buf || !this.splatTotal || this.cells !== t.Wm * t.Hm) return;
    this.chi2Busy = true;
    try {
      const total = this.splatTotal;
      const rb = this.device.createBuffer({ size: 4 * this.cells, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(this.buf, 0, rb, 0, 4 * this.cells);
      this.device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const obs = new Uint32Array(rb.getMappedRange().slice(0));
      rb.unmap(); rb.destroy();
      const exp = new Float64Array(this.cells);
      for (let k = 0; k < this.cells; k++) exp[k] = (total * t.pdfUV[k]) / this.cells;
      this.chi2 = { ...chi2Merged(obs, exp, 50), total };
    } finally {
      this.chi2Busy = false;
    }
  }

  destroy(): void { this.params.destroy(); this.buf?.destroy(); this.target?.destroy(); }
}
