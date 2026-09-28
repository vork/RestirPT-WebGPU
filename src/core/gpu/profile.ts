// Device limit profile and feature policy.
//
// Every lane (Chrome 154, dawn.node) requests min(adapter, PROFILE) for each max* limit so that no lane can
// accept a layout that Chrome on the target Mac rejects (plan §1.1, review WGPU-8). Values were measured on the
// M5 Pro in Chromium 152/154 (docs/research/webgpu-platform.md §2.1) and cross-checked in dawn.node (M0).

export const PROFILE_CHROME154_M5PRO = {
  maxTextureDimension1D: 16384,
  maxTextureDimension2D: 16384,
  maxTextureDimension3D: 2048,
  maxTextureArrayLayers: 2048,
  maxBindGroups: 4,
  maxBindingsPerBindGroup: 1000,
  maxDynamicUniformBuffersPerPipelineLayout: 10,
  maxDynamicStorageBuffersPerPipelineLayout: 8,
  maxSampledTexturesPerShaderStage: 48,
  maxSamplersPerShaderStage: 16,
  maxStorageBuffersPerShaderStage: 10,
  maxStorageTexturesPerShaderStage: 8,
  maxUniformBuffersPerShaderStage: 12,
  maxUniformBufferBindingSize: 65536,
  maxStorageBufferBindingSize: 4294967292,
  maxBufferSize: 4294967292,
  maxComputeWorkgroupStorageSize: 32768,
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupSizeX: 1024,
  maxComputeWorkgroupSizeY: 1024,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65535,
  maxImmediateSize: 64,
} as const satisfies Record<string, number>;

export type ProfileLimitName = keyof typeof PROFILE_CHROME154_M5PRO;

/** Optional features we use when present. Nothing here is required for correctness; the composer exposes
 *  HAS_* defines so shaders pick fallbacks (plan §1.1). */
export const WANTED_FEATURES = [
  'timestamp-query',
  'subgroups',
  'shader-f16',
  'float32-filterable',
  'texture-formats-tier2',
  'bgra8unorm-storage',
  'texture-compression-bc',
] as const;

/** WGSL `requires`/`enable` directives that shipped in Chrome (not unsafe-experimental). Anything else is rejected
 *  by the composer so --enable-unsafe-webgpu can never leak experimental WGSL into the product (review WGPU-9). */
export const SHIPPED_WGSL_DIRECTIVES = new Set<string>([
  // `enable` extensions (require the matching device feature)
  'f16',
  'subgroups',
  // `requires` language features
  'immediate_address_space',
  'linear_indexing',
  'unrestricted_pointer_parameters',
  'pointer_composite_access',
  'readonly_and_readwrite_storage_textures',
  'packed_4x8_integer_dot_product',
  'subgroup_uniformity',
  'subgroup_id',
  'uniform_buffer_standard_layout',
]);

/** Map from `enable` extension name to the device feature it needs. */
export const ENABLE_TO_FEATURE: Record<string, GPUFeatureName> = {
  f16: 'shader-f16',
  subgroups: 'subgroups',
};
