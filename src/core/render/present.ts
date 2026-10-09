// Present (plan §1.8, §1.10): fullscreen render pass that resolves the internal-resolution linear 'color' texture
// (exposure 2^EV + view transform) or the debug view and upscales it to the canvas (nearest | bilinear), then draws
// the raster overlay in the same pass. The canvas is configured non-sRGB (bgra8unorm): encoding happens in WGSL.
// perf2 WP-7g (PresentSettings.pretone, default on): a compute pass (post/tonemap.wgsl) applies the view transform once
// per internal texel into an rgba16float display texture, which the blit upscales (9 bilinear taps for bicubic, one
// texel at 1:1). Off: the M8 blit (the transform on each of the 16 bicubic taps of every canvas pixel).
import { composeWgsl, createCheckedShaderModule } from '../gpu/wgsl-composer.ts';
import { shaderSources } from '../shaders/index.ts';
import type { CameraState } from './frame-uniforms.ts';
import type { Overlay } from './overlay.ts';

export type Tonemap = 'standard' | 'agx' | 'aces' | 'raw';
export const TONEMAP_CODE: Record<Tonemap, number> = { standard: 0, agx: 1, aces: 2, raw: 3 };
export type UpscaleFilter = 'nearest' | 'bilinear' | 'bicubic';

export interface PresentSettings {
  exposureEV: number;
  tonemap: Tonemap;
  filter: UpscaleFilter;
  highlightNonFinite: boolean;
  /** perf2 WP-7g: tonemap once per internal texel (default true; false = the M8 per-tap transform, for A/B). */
  pretone?: boolean;
}

export interface PresentDebug {
  active: boolean;
  split: boolean;
  splitPos: number;
  probe: boolean;
  probePixel: [number, number];
}

const BLIT = { BILINEAR: 1, DEBUG: 2, SPLIT: 4, NONFINITE: 8, PROBE: 16, BICUBIC: 32, PRETONED: 64 } as const;
const PARAMS_SIZE = 48;
const TONE_PARAMS_SIZE = 32;

/** Internal resolution for a preset at a given canvas aspect: height H, width = round(H·aspect/8)·8. M8 dynamic
 *  resolution (src/app/dynres.ts): `scale` < 1 scales H to a multiple of 8 (native: both canvas axes). */
export type ResolutionPreset = '540p' | '720p' | '1080p' | 'native';
export function internalResolution(preset: ResolutionPreset, canvasW: number, canvasH: number, scale = 1): [number, number] {
  const cw = Math.max(1, Math.round(canvasW));
  const ch = Math.max(1, Math.round(canvasH));
  const sc = (x: number) => (scale === 1 ? x : Math.max(8, Math.round((x * scale) / 8) * 8));
  if (preset === 'native') return scale === 1 ? [cw, ch] : [sc(cw), sc(ch)];
  const h = sc(preset === '540p' ? 540 : preset === '720p' ? 720 : 1080);
  const w = Math.max(8, Math.round((h * (cw / ch)) / 8) * 8);
  return [w, h];
}

export class Presenter {
  readonly context: GPUCanvasContext;
  readonly format: GPUTextureFormat;
  private pipeline: GPURenderPipeline | undefined;
  private readonly layout: GPUBindGroupLayout;
  private readonly params: GPUBuffer;
  private bindGroup: GPUBindGroup | undefined;
  private boundColor: GPUTexture | undefined;
  private boundDebug: GPUTexture | undefined;
  private readonly scratch = new ArrayBuffer(PARAMS_SIZE);
  // perf2 WP-7g: tonemap-once pass
  private tonePipeline: GPUComputePipeline | undefined;
  private readonly toneLayout: GPUBindGroupLayout;
  private readonly toneParams: GPUBuffer;
  private toneGroup: GPUBindGroup | undefined;
  private disp: GPUTexture | undefined;
  private readonly sampler: GPUSampler;
  private readonly toneScratch = new ArrayBuffer(TONE_PARAMS_SIZE);

  constructor(private readonly device: GPUDevice, readonly canvas: HTMLCanvasElement, gpu: GPU = navigator.gpu) {
    const ctx = canvas.getContext('webgpu');
    if (!ctx) throw new Error('canvas.getContext("webgpu") failed');
    this.context = ctx;
    // Preferred is bgra8unorm on macOS. Never an -srgb view: the OETF is applied in blit.wgsl.
    this.format = gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format: this.format, alphaMode: 'opaque' });
    this.layout = device.createBindGroupLayout({
      label: 'blit',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    this.params = device.createBuffer({ label: 'blit-params', size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.toneLayout = device.createBindGroupLayout({
      label: 'tonemap',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    this.toneParams = device.createBuffer({ label: 'tonemap-params', size: TONE_PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.sampler = device.createSampler({ label: 'blit-disp', magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
  }

  async init(): Promise<void> {
    const shader = composeWgsl('post/blit.wgsl', { sources: shaderSources });
    const module = await createCheckedShaderModule(this.device, shader, 'blit');
    this.pipeline = await this.device.createRenderPipelineAsync({
      label: 'blit',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module, entryPoint: 'vs_fullscreen' },
      fragment: { module, entryPoint: 'fs_blit', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
    const tone = composeWgsl('post/tonemap.wgsl', { sources: shaderSources });
    const toneModule = await createCheckedShaderModule(this.device, tone, 'tonemap');
    this.tonePipeline = await this.device.createComputePipelineAsync({
      label: 'tonemap', layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.toneLayout] }), compute: { module: toneModule, entryPoint: 'tonemap' },
    });
  }

  /**
   * Encode the present pass into the current canvas texture. `color` is rgba32float/rgba16float linear radiance,
   * `debugOut` the display-encoded debug view (both internal resolution).
   */
  encode(
    encoder: GPUCommandEncoder,
    src: { color: GPUTexture; debugOut: GPUTexture },
    s: PresentSettings,
    dbg: PresentDebug,
    overlay?: { overlay: Overlay; camera: CameraState },
    /** The blit's timestamps; a function is called only when the pass is encoded, after the tonemap's (encode order). */
    timestampWrites?: GPURenderPassTimestampWrites | (() => GPURenderPassTimestampWrites | undefined),
    /** WP-7g: the tonemap pass's timestamps (a function: called only when the pass is encoded); `target`: render into
     *  this texture (tests) instead of the canvas. */
    o: { toneTimestampWrites?: GPUComputePassTimestampWrites | (() => GPUComputePassTimestampWrites | undefined); target?: GPUTexture } = {},
  ): void {
    if (!this.pipeline || !this.tonePipeline) return;
    const target = o.target ?? this.context.getCurrentTexture();
    if (src.color !== this.boundColor || src.debugOut !== this.boundDebug) {
      this.boundColor = src.color;
      this.boundDebug = src.debugOut;
      this.disp?.destroy();
      this.disp = this.device.createTexture({
        label: 'present-disp', size: [src.color.width, src.color.height], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.bindGroup = this.device.createBindGroup({
        label: 'blit',
        layout: this.layout,
        entries: [
          { binding: 0, resource: src.color.createView() },
          { binding: 1, resource: src.debugOut.createView() },
          { binding: 2, resource: { buffer: this.params } },
          { binding: 3, resource: this.disp.createView() },
          { binding: 4, resource: this.sampler },
        ],
      });
      this.toneGroup = this.device.createBindGroup({
        label: 'tonemap',
        layout: this.toneLayout,
        entries: [
          { binding: 0, resource: src.color.createView() },
          { binding: 1, resource: this.disp.createView() },
          { binding: 2, resource: { buffer: this.toneParams } },
        ],
      });
    }
    const pretone = s.pretone !== false;
    // the beauty is needed unless the debug view covers the whole canvas
    const beauty = !(dbg.active && !dbg.split);
    if (pretone && beauty) {
      const tu = new Uint32Array(this.toneScratch), tf = new Float32Array(this.toneScratch);
      tu[0] = src.color.width; tu[1] = src.color.height; tf[2] = 2 ** s.exposureEV; tu[3] = TONEMAP_CODE[s.tonemap]; tu[4] = s.highlightNonFinite ? 1 : 0;
      this.device.queue.writeBuffer(this.toneParams, 0, this.toneScratch);
      const tw = typeof o.toneTimestampWrites === 'function' ? o.toneTimestampWrites() : o.toneTimestampWrites;
      const cp = encoder.beginComputePass({ label: 'tonemap', timestampWrites: tw });
      cp.setPipeline(this.tonePipeline);
      cp.setBindGroup(0, this.toneGroup!);
      cp.dispatchWorkgroups(Math.ceil(src.color.width / 8), Math.ceil(src.color.height / 8));
      cp.end();
    }
    const u32 = new Uint32Array(this.scratch);
    const f32 = new Float32Array(this.scratch);
    u32[0] = src.color.width; u32[1] = src.color.height;
    u32[2] = target.width; u32[3] = target.height;
    f32[4] = 2 ** s.exposureEV;
    u32[5] = TONEMAP_CODE[s.tonemap];
    let flags = 0;
    if (s.filter === 'bilinear') flags |= BLIT.BILINEAR;
    if (s.filter === 'bicubic') flags |= BLIT.BICUBIC;
    if (dbg.active) flags |= BLIT.DEBUG;
    if (dbg.active && dbg.split) flags |= BLIT.SPLIT;
    if (s.highlightNonFinite) flags |= BLIT.NONFINITE;
    if (dbg.probe) flags |= BLIT.PROBE;
    if (pretone) flags |= BLIT.PRETONED;
    u32[6] = flags;
    f32[7] = dbg.splitPos;
    u32[8] = dbg.probePixel[0]; u32[9] = dbg.probePixel[1];
    u32[10] = 0; u32[11] = 0;
    this.device.queue.writeBuffer(this.params, 0, this.scratch);

    const pass = encoder.beginRenderPass({
      label: 'present',
      colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      timestampWrites: typeof timestampWrites === 'function' ? timestampWrites() : timestampWrites,
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup!);
    pass.draw(3);
    overlay?.overlay.draw(pass, overlay.camera, [src.color.width, src.color.height], [target.width, target.height]);
    pass.end();
  }

  destroy(): void { this.params.destroy(); this.toneParams.destroy(); this.disp?.destroy(); }
}
