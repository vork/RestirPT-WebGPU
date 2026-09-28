// Fallback renderer (no scene / no integrator renderFrame): analytic test scene + calibration strip.
import { composeWgsl, createCheckedShaderModule } from '../core/gpu/wgsl-composer.ts';
import { probeTags, registerProbeTag } from '../core/render/probe.ts';
import { shaderSources } from '../core/shaders/index.ts';
import type { RenderTargets } from './app.ts';
import testPatternWgsl from './test-pattern.wgsl?raw';

export class TestPattern {
  private pipeline: GPUComputePipeline | undefined;
  private readonly layout0: GPUBindGroupLayout;
  private readonly empty: GPUBindGroupLayout;
  private readonly emptyGroup: GPUBindGroup;
  private group0: GPUBindGroup | undefined;
  private boundTargets: RenderTargets | undefined;

  constructor(private readonly device: GPUDevice, private readonly debugLayout: GPUBindGroupLayout, private readonly colorFormat: GPUTextureFormat) {
    this.layout0 = device.createBindGroupLayout({
      label: 'test-pattern-g0',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: colorFormat } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float' } },
      ],
    });
    this.empty = device.createBindGroupLayout({ label: 'empty', entries: [] });
    this.emptyGroup = device.createBindGroup({ layout: this.empty, entries: [] });
  }

  async init(): Promise<void> {
    if (!probeTags.has(16)) registerProbeTag(16, 'test.rgb|t');
    if (!probeTags.has(17)) registerProbeTag(17, 'test.dir|frame');
    const shader = composeWgsl('app/test-pattern.wgsl', {
      sources: { ...shaderSources, 'app/test-pattern.wgsl': testPatternWgsl },
      defines: { COLOR_FORMAT: this.colorFormat },
    });
    const module = await createCheckedShaderModule(this.device, shader, 'test-pattern');
    this.pipeline = await this.device.createComputePipelineAsync({
      label: 'test-pattern',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout0, this.empty, this.empty, this.debugLayout] }),
      compute: { module, entryPoint: 'test_pattern' },
    });
  }

  encode(encoder: GPUCommandEncoder, t: RenderTargets, timestampWrites?: GPUComputePassTimestampWrites): void {
    if (!this.pipeline) return;
    if (t !== this.boundTargets) {
      this.boundTargets = t;
      this.group0 = this.device.createBindGroup({
        label: 'test-pattern-g0',
        layout: this.layout0,
        entries: [
          { binding: 0, resource: { buffer: t.frameUniforms } },
          { binding: 1, resource: t.color.createView() },
          { binding: 2, resource: t.depth.createView() },
        ],
      });
    }
    const pass = encoder.beginComputePass({ label: 'test-pattern', timestampWrites });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.group0!);
    pass.setBindGroup(1, this.emptyGroup);
    pass.setBindGroup(2, this.emptyGroup);
    pass.setBindGroup(3, t.debug.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(t.width / 8), Math.ceil(t.height / 8));
    pass.end();
  }
}
