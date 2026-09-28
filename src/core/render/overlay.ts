// Raster overlay (plan §3 step 7, §6): world-space line lists drawn on the canvas after the blit, depth-tested in
// the fragment shader against the primary pass's linear depth (internal resolution), or always on top.
// Points are UN-recentred world coordinates; the origin is subtracted in f64 on upload (plan §1.2).
import { composeWgsl, createCheckedShaderModule } from '../gpu/wgsl-composer.ts';
import { shaderSources } from '../shaders/index.ts';
import { recentre, rigidInverse, type CameraState } from './frame-uniforms.ts';

export type OverlayDepthKind = 'view-z' | 'distance';

interface LineBatch { points: Float64Array; colors: Uint8Array; onTop: boolean }

const VERTEX_STRIDE = 20; // f32x3 position, unorm8x4 colour, u32 flags
const PARAMS_SIZE = 160;

function toUnorm8(c: number): number { return Math.max(0, Math.min(255, Math.round(c * 255))); }

/** Reversed infinite perspective (clip z = near, w = -z_view) times view, column-major, f64. */
export function overlayViewProj(worldToCam: ArrayLike<number>, yfov: number, aspect: number, near: number): Float64Array {
  const t = Math.tan(yfov / 2);
  const P = new Float64Array(16);
  P[0] = 1 / (t * aspect);
  P[5] = 1 / t;
  P[11] = -1;
  P[14] = near;
  const out = new Float64Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += P[k * 4 + r] * worldToCam[c * 4 + k];
    out[c * 4 + r] = s;
  }
  return out;
}

export class Overlay {
  private readonly batches = new Map<number, LineBatch>();
  private nextId = 1;
  private dirty = true;
  private vbuf: GPUBuffer | undefined;
  private vcount = 0;
  private pipeline: GPURenderPipeline | undefined;
  private readonly layout: GPUBindGroupLayout;
  private readonly params: GPUBuffer;
  private bindGroup: GPUBindGroup | undefined;
  private depthTex: GPUTexture | undefined;
  private readonly dummyDepth: GPUTexture;
  private origin: [number, number, number] = [0, 0, 0];
  depthKind: OverlayDepthKind = 'view-z';
  /** Alpha multiplier for occluded fragments (0 = hide). */
  hiddenAlpha = 0.25;
  depthEps = 2e-3;
  enabled = true;

  constructor(private readonly device: GPUDevice, private readonly format: GPUTextureFormat) {
    this.layout = device.createBindGroupLayout({
      label: 'overlay',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    });
    this.params = device.createBuffer({ label: 'overlay-params', size: PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dummyDepth = device.createTexture({ label: 'overlay-no-depth', size: [1, 1], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING });
  }

  async init(): Promise<void> {
    const shader = composeWgsl('debug/overlay.wgsl', { sources: shaderSources });
    const module = await createCheckedShaderModule(this.device, shader, 'overlay');
    this.pipeline = await this.device.createRenderPipelineAsync({
      label: 'overlay',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: {
        module,
        entryPoint: 'vs_overlay',
        buffers: [{
          arrayStride: VERTEX_STRIDE,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' },
            { shaderLocation: 1, offset: 12, format: 'unorm8x4' },
            { shaderLocation: 2, offset: 16, format: 'uint32' },
          ],
        }],
      },
      fragment: {
        module,
        entryPoint: 'fs_overlay',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'line-list' },
    });
  }

  /**
   * Add a line list: `points` holds xyz pairs (2 vertices per segment) in UN-recentred world space. `colors` is
   * either one rgba for all vertices or rgba per vertex (0..1). Returns a handle for remove().
   */
  addLines(points: ArrayLike<number>, colors: ArrayLike<number> = [1, 1, 0, 1], onTop = false): number {
    const nv = Math.floor(points.length / 3) & ~1;
    const perVertex = colors.length >= nv * 4 && nv > 0 && colors.length !== 4;
    const col = new Uint8Array(nv * 4);
    for (let i = 0; i < nv; i++) for (let k = 0; k < 4; k++) col[i * 4 + k] = toUnorm8(perVertex ? colors[i * 4 + k] : colors[k] ?? 1);
    const id = this.nextId++;
    this.batches.set(id, { points: Float64Array.from({ length: nv * 3 }, (_, i) => points[i]), colors: col, onTop });
    this.dirty = true;
    return id;
  }

  remove(id: number): void { if (this.batches.delete(id)) this.dirty = true; }
  clear(): void { this.batches.clear(); this.dirty = true; }
  get lineCount(): number { let n = 0; for (const b of this.batches.values()) n += b.points.length / 6; return n; }

  setOrigin(o: [number, number, number]): void { this.origin = [...o]; this.dirty = true; }

  /** Linear depth texture (r32float, internal resolution) from the primary pass; undefined = no depth test. */
  setDepth(tex: GPUTexture | undefined, kind: OverlayDepthKind = this.depthKind): void {
    if (tex !== this.depthTex) this.bindGroup = undefined;
    this.depthTex = tex;
    this.depthKind = kind;
  }

  private upload(): void {
    this.dirty = false;
    let nv = 0;
    for (const b of this.batches.values()) nv += b.points.length / 3;
    this.vcount = nv;
    if (nv === 0) return;
    const bytes = nv * VERTEX_STRIDE;
    if (!this.vbuf || this.vbuf.size < bytes) {
      this.vbuf?.destroy();
      this.vbuf = this.device.createBuffer({ label: 'overlay-vertices', size: Math.max(bytes, 4096) * 2, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    }
    const data = new ArrayBuffer(bytes);
    const f32 = new Float32Array(data);
    const u8 = new Uint8Array(data);
    const u32 = new Uint32Array(data);
    let v = 0;
    for (const b of this.batches.values()) {
      for (let i = 0; i < b.points.length / 3; i++, v++) {
        const w = (v * VERTEX_STRIDE) / 4;
        for (let k = 0; k < 3; k++) f32[w + k] = b.points[i * 3 + k] - this.origin[k];
        u8.set(b.colors.subarray(i * 4, i * 4 + 4), v * VERTEX_STRIDE + 12);
        u32[w + 4] = b.onTop ? 1 : 0;
      }
    }
    this.device.queue.writeBuffer(this.vbuf, 0, data);
  }

  /** Update uniforms for this frame's camera (UN-recentred world matrix) and draw into an open render pass. */
  draw(pass: GPURenderPassEncoder, camera: CameraState, internal: [number, number], canvas: [number, number]): void {
    if (!this.enabled || !this.pipeline || this.batches.size === 0) return;
    if (this.dirty) this.upload();
    if (this.vcount === 0 || !this.vbuf) return;
    const w2c = rigidInverse(recentre(camera.camToWorld, this.origin));
    const vp = overlayViewProj(w2c, camera.yfov, internal[0] / internal[1], camera.znear ?? 1e-3);
    const buf = new ArrayBuffer(PARAMS_SIZE);
    const f32 = new Float32Array(buf);
    const u32 = new Uint32Array(buf);
    f32.set(vp, 0);
    f32.set(w2c, 16);
    u32[32] = internal[0]; u32[33] = internal[1];
    u32[34] = canvas[0]; u32[35] = canvas[1];
    u32[36] = this.depthKind === 'view-z' ? 0 : 1;
    f32[37] = this.hiddenAlpha;
    f32[38] = this.depthEps;
    u32[39] = this.depthTex ? 1 : 0;
    this.device.queue.writeBuffer(this.params, 0, buf);
    this.bindGroup ??= this.device.createBindGroup({
      label: 'overlay',
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 1, resource: (this.depthTex ?? this.dummyDepth).createView() },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vbuf);
    pass.draw(this.vcount);
  }

  destroy(): void {
    this.vbuf?.destroy();
    this.params.destroy();
    this.dummyDepth.destroy();
  }
}
