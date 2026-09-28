# Platform lanes (M0 lane diff)

Generated at M0 from `vitest --project chrome` and `--project node-dawn` (`validation/gpu-tests/smoke.gpu.test.ts`).
The Chrome lane is authoritative for every gate. dawn.node is a fast pre-check only (plan §1.1).

| Item | Chrome 154 (headless, Playwright) | dawn.node (webgpu@0.6.1, Node 26) |
|---|---|---|
| adapter | apple / metal-3 | apple / metal-3 |
| features (device, after profile request) | bgra8unorm-storage, core-features-and-limits, float32-filterable, rg11b10ufloat-renderable, shader-f16, subgroups, texture-compression-bc, texture-formats-tier1, texture-formats-tier2, timestamp-query | bgra8unorm-storage, core-features-and-limits, float32-filterable, rg11b10ufloat-renderable, shader-f16, subgroups, texture-compression-bc, texture-formats-tier1, texture-formats-tier2, timestamp-query |
| features only in one lane | — | — |
| WGSL language features only in one lane | — | — |
| device limits differing | — (identical after profile clamp) | — |
| limits below profile | — | — |
| `x != x` NaN self-compare | FOLDED (false) — use bit tests | FOLDED (false) — use bit tests |
| FMA contraction of a*b+c | True | True |
| WGSL language features (both lanes) | buffer_view, immediate_address_space, linear_indexing, packed_4x8_integer_dot_product, pointer_composite_access, readonly_and_readwrite_storage_textures, subgroup_id, subgroup_uniformity, swizzle_assignment, texture_and_sampler_let, texture_formats_tier1, uniform_buffer_standard_layout, unrestricted_pointer_parameters | identical to Chrome |

Headless Chrome 154 exposes the hardware Metal adapter without `--enable-unsafe-webgpu` (`npm run smoke:chrome`), so the vitest Chrome lane runs without the flag.
