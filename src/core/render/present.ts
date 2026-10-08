// Present (plan §1.8, §1.10): fullscreen render pass that resolves the internal-resolution linear 'color' texture
// (exposure 2^EV + view transform) or the debug view and upscales it to the canvas (nearest | bilinear), then draws
// the raster overlay in the same pass. The canvas is configured non-sRGB (bgra8unorm): encoding happens in WGSL.
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
}

export interface PresentDebug {
  active: boolean;
  split: boolean;
  splitPos: number;
  probe: boolean;
  probePixel: [number, number];
}

const BLIT = { BILINEAR: 1, DEBUG: 2, SPLIT: 4, NONFINITE: 8, PROBE: 16, BICUBIC: 32 } as const;
const PARAMS_SIZE = 48;

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
      ],
    });
    this.params = device.createBuffer({ label: 'blit-params', size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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
    timestampWrites?: GPURenderPassTimestampWrites,
  ): void {
    if (!this.pipeline) return;
    const target = this.context.getCurrentTexture();
    if (src.color !== this.boundColor || src.debugOut !== this.boundDebug) {
      this.boundColor = src.color;
      this.boundDebug = src.debugOut;
      this.bindGroup = this.device.createBindGroup({
        label: 'blit',
        layout: this.layout,
        entries: [
          { binding: 0, resource: src.color.createView() },
          { binding: 1, resource: src.debugOut.createView() },
          { binding: 2, resource: { buffer: this.params } },
        ],
      });
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
    u32[6] = flags;
    f32[7] = dbg.splitPos;
    u32[8] = dbg.probePixel[0]; u32[9] = dbg.probePixel[1];
    u32[10] = 0; u32[11] = 0;
    this.device.queue.writeBuffer(this.params, 0, this.scratch);

    const pass = encoder.beginRenderPass({
      label: 'present',
      colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
      timestampWrites,
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup!);
    pass.draw(3);
    overlay?.overlay.draw(pass, overlay.camera, [src.color.width, src.color.height], [target.width, target.height]);
    pass.end();
  }

  destroy(): void { this.params.destroy(); }
}
