// M7 shading-normal debug views (plan §6; docs/decisions/m7-api.md §7; WGSL: shaders/passes/shading-debug.wgsl): the
// closure normal N, the unmapped shading normal Ns, the MikkTSpace tangent frame (T, B, sign), the normal-map decode
// and the angles N∠Ns, Ns∠Ng, at the primary hit through the pixel centre. A separate pass (compiled on first use), so
// the primary pass and every validation pipeline keep their text (U-M7-BITS).
import { composeWgsl, createCheckedShaderModule } from '../gpu/wgsl-composer.ts';
import { shaderSources } from '../shaders/index.ts';
import type { DebugViewDef } from './debug-views.ts';
import type { SceneGpu } from './scene-gpu.ts';

export const SH_DBG = { N: 320, NS: 321, T: 322, B: 323, NM: 324, NM_ANGLE: 325, NS_ANGLE: 326, SIGN: 327 } as const;
export const isShadingDebugView = (mode: number): boolean => mode >= 320 && mode <= 327;

export const SHADING_DEBUG_VIEWS: DebugViewDef[] = [
  { id: SH_DBG.N, key: 'shading.n', label: 'Closure normal N (normal-mapped)', group: 'Shading normals', source: 'shading-debug', kind: 'vec3', range: [-1, 1],
    description: 'the normal every BSDF lobe uses: the Cycles Normal Map node output where the material has a normal texture, else Ns' },
  { id: SH_DBG.NS, key: 'shading.ns', label: 'Unmapped shading normal Ns', group: 'Shading normals', source: 'shading-debug', kind: 'vec3', range: [-1, 1],
    description: 'interpolated vertex normal (Ng on flat faces); the bump-shadowing reference normal' },
  { id: SH_DBG.T, key: 'shading.tangent', label: 'Tangent T (MikkTSpace)', group: 'Shading normals', source: 'shading-debug', kind: 'vec3', range: [-1, 1],
    description: 'interpolated MikkTSpace tangent (black: no tangent / no normal-mapped material in the scene)' },
  { id: SH_DBG.B, key: 'shading.bitangent', label: 'Bitangent B = sign·(Ns × T)', group: 'Shading normals', source: 'shading-debug', kind: 'vec3', range: [-1, 1] },
  { id: SH_DBG.NM, key: 'shading.nmDecode', label: 'Normal-map decode c (strength applied)', group: 'Shading normals', source: 'shading-debug', kind: 'vec3', range: [-1, 1],
    description: 'c = 2(rgb − ½), c.xy·s, c.z = mix(1, c.z, saturate(s)) (tangent space; (0, 0, 1) without a normal texture)' },
  { id: SH_DBG.NM_ANGLE, key: 'shading.nmAngle', label: 'angle(N, Ns) [°]', group: 'Shading normals', source: 'shading-debug', kind: 'scalar', range: [0, 60], colormap: 'turbo' },
  { id: SH_DBG.NS_ANGLE, key: 'shading.nsNgAngle', label: 'angle(Ns, Ng) [°] (smooth shading)', group: 'Shading normals', source: 'shading-debug', kind: 'scalar', range: [0, 10], colormap: 'turbo' },
  { id: SH_DBG.SIGN, key: 'shading.tangentSign', label: 'Bitangent sign (1: +1, 2: −1)', group: 'Shading normals', source: 'shading-debug', kind: 'code' },
];

export class ShadingDebugPass {
  private g0: GPUBindGroup | undefined;
  private g0Frame: GPUBuffer | undefined;
  private readonly g1: GPUBindGroup;

  private constructor(readonly device: GPUDevice, private readonly layouts: GPUBindGroupLayout[], private readonly pipeline: GPUComputePipeline, scene: SceneGpu) {
    this.g1 = device.createBindGroup({ label: 'shading-debug-g1', layout: layouts[1], entries: scene.bindGroupEntries() });
  }

  static async create(device: GPUDevice, scene: SceneGpu, debugLayout: GPUBindGroupLayout, opts: { features?: Set<string>; wgslLanguageFeatures?: Set<string> } = {}): Promise<ShadingDebugPass> {
    const c = GPUShaderStage.COMPUTE;
    const g0 = device.createBindGroupLayout({ label: 'shading-debug-g0', entries: [{ binding: 0, visibility: c, buffer: { type: 'uniform' } }] });
    const g1 = device.createBindGroupLayout({ label: 'shading-debug-g1', entries: scene.layoutEntries(c) });
    const g2 = device.createBindGroupLayout({ label: 'shading-debug-g2', entries: [] });
    const shader = composeWgsl('passes/shading-debug.wgsl', { sources: shaderSources, defines: { ...scene.defines(1) }, features: opts.features, wgslLanguageFeatures: opts.wgslLanguageFeatures });
    const module = await createCheckedShaderModule(device, shader, 'shading-debug');
    const layout = device.createPipelineLayout({ label: 'shading-debug', bindGroupLayouts: [g0, g1, g2, debugLayout] });
    const pipeline = await device.createComputePipelineAsync({ label: 'shading-debug', layout, compute: { module, entryPoint: 'shading_view' } });
    return new ShadingDebugPass(device, [g0, g1, g2], pipeline, scene);
  }

  encode(encoder: GPUCommandEncoder, f: { mode: number; frameUniforms: GPUBuffer; width: number; height: number; debugGroup: GPUBindGroup }): void {
    if (!isShadingDebugView(f.mode)) return;
    if (!this.g0 || this.g0Frame !== f.frameUniforms) {
      this.g0Frame = f.frameUniforms;
      this.g0 = this.device.createBindGroup({ label: 'shading-debug-g0', layout: this.layouts[0], entries: [{ binding: 0, resource: { buffer: f.frameUniforms } }] });
    }
    const pass = encoder.beginComputePass({ label: 'shading-debug' });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.g0);
    pass.setBindGroup(1, this.g1);
    pass.setBindGroup(2, this.device.createBindGroup({ layout: this.layouts[2], entries: [] }));
    pass.setBindGroup(3, f.debugGroup);
    pass.dispatchWorkgroups(Math.ceil(f.width / 8), Math.ceil(f.height / 8));
    pass.end();
  }

  destroy(): void { /* no owned buffers */ }
}
