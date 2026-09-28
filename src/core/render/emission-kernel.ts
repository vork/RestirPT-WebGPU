// Emission-only kernel (plan §5 M2): primary emission + camera-visible area lights + the length-1 env term, for the
// harness calibration scenes (C0a/C0b/C0p) before the path tracer exists. WGSL: shaders/passes/emission.wgsl.
//
// Bind groups (storage buffers 7 of 10):
//   G0  0 FrameUniforms | 1..3 env (uniform, texEnv, sEnv) | 4 EmissionParams | 5 camLights (read)
//   G1  scene (SceneGpu: bvh nodes/tris, vertices, tris, materials, textures)
//   G2  0 accum (rgb sum, vec4f per pixel) | 1 counters (NaN/Inf samples, BVH overflow, BVH iteration cap)
// One dispatch = `sampleCount` samples for rows [rowBase, rowBase + rows); BatchAccumulator sizes them to the submit
// budget (plan §1.8).
import { composeWgsl, createCheckedShaderModule } from '../gpu/wgsl-composer.ts';
import { shaderSources } from '../shaders/index.ts';
import type { LightData } from '../scene/types.ts';
import { envBindGroupEntries, envBindGroupLayoutEntries, envDefines, type EnvGpuResources } from './env-gpu.ts';
import {
  FrameUniformBuffer, JITTER_IID, JITTER_NONE, boundsDiagonal, type CameraState, type JitterMode,
} from './frame-uniforms.ts';
import type { SceneGpu } from './scene-gpu.ts';

export const EMISSION_PARAMS_SIZE = 32;
export const CAM_LIGHT_BYTES = 80;
export const EMISSION_COUNTERS = { nonFinite: 0, bvhOverflow: 1, bvhItercap: 2 } as const;
export const EMISSION_COUNTER_BYTES = 16;
const ENV_BINDING_BASE = 1;
const SCENE_GROUP = 1;

export interface EmissionView {
  camera: CameraState;
  width: number;
  height: number;
  runSeed: number;
  /** Default JITTER_IID (validation: i.i.d. per run and sample). JITTER_NONE uses `jitter` (default pixel centre). */
  jitterMode?: JitterMode;
  jitter?: [number, number];
}

export interface EmissionDispatch { sampleBase: number; sampleCount: number; rowBase: number; rows: number }

/** Spread normalisation N_s (math.md#units-lights; Cycles scene/light.cpp:1351-1355). Negative = spread π (factor 1). */
export function spreadNormalization(spread: number): { tanHalf: number; norm: number } {
  if (!(spread < Math.PI)) return { tanHalf: 0, norm: -1 };
  const a = spread / 2;
  return { tanHalf: Math.tan(a), norm: a > 0.05 ? 1 / (Math.tan(a) - a) : 3 / (a * a * a) };
}

/** Radiance of an area light, L = color·power·2^exposure/(π A) (math.md#units-lights). */
export function areaLightRadiance(l: LightData): [number, number, number] {
  const sx = l.sizeX ?? 1, sy = l.type === 'disk' ? (l.sizeY ?? sx) : (l.sizeY ?? sx);
  const A = l.type === 'disk' ? (Math.PI / 4) * sx * sy : sx * sy;
  const s = (l.power * 2 ** l.exposure) / (Math.PI * A);
  return [l.color[0] * s, l.color[1] * s, l.color[2] * s];
}

/** Camera-visible rect/disk lights → CamLight records (recentred by `origin`). */
export function packCamLights(lights: LightData[], origin: readonly number[]): { data: ArrayBuffer; count: number } {
  const vis = lights.filter((l) => l.visibleToCamera && (l.type === 'rect' || l.type === 'disk'));
  const buf = new ArrayBuffer(Math.max(1, vis.length) * CAM_LIGHT_BYTES);
  const f = new Float32Array(buf), u = new Uint32Array(buf);
  vis.forEach((l, i) => {
    const o = (i * CAM_LIGHT_BYTES) / 4;
    const m = l.matrix;
    const unit = (c: number) => { const v = [m[4 * c], m[4 * c + 1], m[4 * c + 2]]; const n = Math.hypot(...v) || 1; return v.map((x) => x / n); };
    const X = unit(0), Y = unit(1), Z = unit(2);
    const sx = l.sizeX ?? 1, sy = l.sizeY ?? sx;
    const L = areaLightRadiance(l);
    const sp = spreadNormalization(l.spread ?? Math.PI);
    f[o] = m[12] - origin[0]; f[o + 1] = m[13] - origin[1]; f[o + 2] = m[14] - origin[2]; u[o + 3] = l.type === 'disk' ? 1 : 0;
    f.set(X, o + 4); f[o + 7] = sx / 2;
    f.set(Y, o + 8); f[o + 11] = sy / 2;
    f.set(L, o + 12); f[o + 15] = sp.norm;
    f[o + 16] = -Z[0]; f[o + 17] = -Z[1]; f[o + 18] = -Z[2]; f[o + 19] = sp.tanHalf;
  });
  return { data: buf, count: vis.length };
}

export interface EmissionKernelOptions {
  features?: Set<string>;
  wgslLanguageFeatures?: Set<string>;
}

export class EmissionKernel {
  readonly frame: FrameUniformBuffer;
  private readonly params: GPUBuffer;
  private camLights: GPUBuffer;
  private lightCount = 0;
  private g0!: GPUBindGroup;
  private readonly g1: GPUBindGroup;
  private g2: GPUBindGroup | undefined;
  private g2Key: [GPUBuffer, GPUBuffer] | undefined;
  private view: EmissionView | undefined;

  private constructor(
    readonly device: GPUDevice,
    readonly scene: SceneGpu,
    private env: EnvGpuResources,
    private readonly pipeline: GPUComputePipeline,
    private readonly layouts: GPUBindGroupLayout[],
  ) {
    this.frame = new FrameUniformBuffer(device);
    this.params = device.createBuffer({ label: 'emission-params', size: EMISSION_PARAMS_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.camLights = device.createBuffer({ label: 'emission-camlights', size: CAM_LIGHT_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.g1 = device.createBindGroup({ label: 'emission-g1', layout: layouts[1], entries: scene.bindGroupEntries() });
    this.setLights(scene.scene.lights);
  }

  static async create(device: GPUDevice, scene: SceneGpu, env: EnvGpuResources, opts: EmissionKernelOptions = {}): Promise<EmissionKernel> {
    const c = GPUShaderStage.COMPUTE;
    const g0 = device.createBindGroupLayout({
      label: 'emission-g0',
      entries: [
        { binding: 0, visibility: c, buffer: { type: 'uniform' } },
        ...envBindGroupLayoutEntries(ENV_BINDING_BASE, c),
        { binding: 4, visibility: c, buffer: { type: 'uniform', minBindingSize: EMISSION_PARAMS_SIZE } },
        { binding: 5, visibility: c, buffer: { type: 'read-only-storage' } },
      ],
    });
    const g1 = device.createBindGroupLayout({ label: 'emission-g1', entries: scene.layoutEntries(c) });
    const g2 = device.createBindGroupLayout({
      label: 'emission-g2',
      entries: [
        { binding: 0, visibility: c, buffer: { type: 'storage' } },
        { binding: 1, visibility: c, buffer: { type: 'storage', minBindingSize: EMISSION_COUNTER_BYTES } },
      ],
    });
    const shader = composeWgsl('passes/emission.wgsl', {
      sources: shaderSources,
      defines: { ...scene.defines(SCENE_GROUP), ...envDefines(0, ENV_BINDING_BASE) },
      features: opts.features, wgslLanguageFeatures: opts.wgslLanguageFeatures,
    });
    const module = await createCheckedShaderModule(device, shader, 'emission');
    const pipeline = await device.createComputePipelineAsync({
      label: 'emission',
      layout: device.createPipelineLayout({ label: 'emission', bindGroupLayouts: [g0, g1, g2] }),
      compute: { module, entryPoint: 'emission' },
    });
    const k = new EmissionKernel(device, scene, env, pipeline, [g0, g1, g2]);
    k.rebuildG0();
    return k;
  }

  private rebuildG0(): void {
    this.g0 = this.device.createBindGroup({
      label: 'emission-g0',
      layout: this.layouts[0],
      entries: [
        { binding: 0, resource: { buffer: this.frame.buffer } },
        ...envBindGroupEntries(this.env, ENV_BINDING_BASE),
        { binding: 4, resource: { buffer: this.params } },
        { binding: 5, resource: { buffer: this.camLights } },
      ],
    });
  }

  /** Camera-visible rect/disk lights (others are never hit by camera rays). */
  setLights(lights: LightData[]): void {
    const { data, count } = packCamLights(lights, this.scene.origin);
    if (data.byteLength > this.camLights.size) {
      this.camLights.destroy();
      this.camLights = this.device.createBuffer({ label: 'emission-camlights', size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      if (this.g0) this.rebuildG0();
    }
    this.device.queue.writeBuffer(this.camLights, 0, data);
    this.lightCount = count;
  }

  get cameraVisibleLights(): number { return this.lightCount; }

  setEnvironment(env: EnvGpuResources): void { this.env = env; this.rebuildG0(); }

  /** Camera, resolution, seed and jitter for the following dispatches. */
  setView(v: EmissionView): void {
    this.view = v;
    const jm = v.jitterMode ?? JITTER_IID;
    this.frame.write({
      camera: v.camera, prevCamera: v.camera, width: v.width, height: v.height,
      frameIndex: 0, seedIndex: 0, runSeed: v.runSeed >>> 0, flags: 0, jitterMode: jm,
      jitter: jm === JITTER_NONE ? (v.jitter ?? [0.5, 0.5]) : [0.5, 0.5],
      origin: this.scene.origin, exposure: 1, time: 0, dt: 0, sceneDiag: boundsDiagonal(this.scene.scene.bounds),
    });
  }

  /**
   * Encode one dispatch. Writes the params with queue.writeBuffer, so submit the encoder before encoding the next
   * dispatch (one dispatch per submit; BatchAccumulator does this).
   */
  encode(encoder: GPUCommandEncoder, d: EmissionDispatch, accum: GPUBuffer, counters: GPUBuffer): void {
    const v = this.view;
    if (!v) throw new Error('EmissionKernel.setView() first');
    if (this.g2Key?.[0] !== accum || this.g2Key?.[1] !== counters) {
      this.g2 = this.device.createBindGroup({
        label: 'emission-g2', layout: this.layouts[2],
        entries: [{ binding: 0, resource: { buffer: accum } }, { binding: 1, resource: { buffer: counters, size: EMISSION_COUNTER_BYTES } }],
      });
      this.g2Key = [accum, counters];
    }
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([d.sampleBase >>> 0, d.sampleCount, this.lightCount, d.rowBase, d.rowBase + d.rows, 0, 0, 0]));
    const pass = encoder.beginComputePass({ label: 'emission' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.g0);
    pass.setBindGroup(1, this.g1);
    pass.setBindGroup(2, this.g2!);
    pass.dispatchWorkgroups(Math.ceil(v.width / 8), Math.ceil(d.rows / 8));
    pass.end();
  }

  destroy(): void {
    this.frame.destroy();
    this.params.destroy();
    this.camLights.destroy();
  }
}

export { JITTER_IID, JITTER_NONE };
