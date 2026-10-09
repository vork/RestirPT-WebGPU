// Cycles LUT reader: exact port of kernel/util/lookup_table.h (math.md#bsdf-v2 "LUT reader"; gap-bsdf §3.5).
// The tables live in the `records` storage buffer (plan §1.8), never in a filtered texture: r32float is not
// filterable without float32-filterable and hardware weights are quantised (cycles-verify omission 6).
//
// Defines (src/core/render/luts/lut-layout.ts lutDefines()):
//   LUT_BASE, LUT_IOR_S, LUT_S, LUT_E, LUT_EAVG   absolute word offsets in `records` (u32 literals, e.g. "4096u")
//   LUT_RECORDS_KIND      element type of `records`: 0 f32, 1 u32 (bitcast), 2 vec4f, 3 vec4u (bitcast)
//   LUT_DECLARE_BINDING   declare `records` here as array<f32> at @group(LUT_GROUP) @binding(LUT_BINDING);
//                         otherwise the includer declares `records` itself (with the element type above).
//
// Tables (v5.1.2 shader.tables, generated into cycles-luts.ts by validation/tools/extract_luts.py):
//   ggx_gen_schlick_ior_s 16³  x = rough, y = mu, z = sqrt|(eta-1)/(eta+1)|   dielectric layering + sample weight
//   ggx_gen_schlick_s     16³  x = rough, y = mu, z = 1/(0.2·exponent+1)       metal sample weight (z = 0.5)
//   ggx_E                 32²  x = rough, y = mu                                Tier 2
//   ggx_Eavg              32   x = rough                                        Tier 2

#if LUT_DECLARE_BINDING
@group($LUT_GROUP) @binding($LUT_BINDING) var<storage, read> records: array<f32>;
#endif

fn lut_fetch(i: u32) -> f32 {
#if LUT_DECLARE_BINDING
  return records[i];
#elif LUT_RECORDS_KIND == 1 && RS_LIGHT_REC4
  return bitcast<f32>(records[i >> 2u][i & 3u]);      // perf2 WP-4a: lights.wgsl declares records as array<vec4u>
#elif LUT_RECORDS_KIND == 1
  return bitcast<f32>(records[i]);
#elif LUT_RECORDS_KIND == 2
  return records[i >> 2u][i & 3u];
#elif LUT_RECORDS_KIND == 3
  return bitcast<f32>(records[i >> 2u][i & 3u]);
#else
  return records[i];
#endif
}

// lookup_table_read: x' = saturate(x)·(n−1); i = min(trunc(x'), n−1); j = min(i+1, n−1); t = x' − i.
fn lut_read(x_in: f32, off: u32, n: u32) -> f32 {
  let x = saturate(x_in) * f32(n - 1u);
  let i = min(u32(x), n - 1u);              // float_to_int truncation (x ≥ 0 after saturate)
  let j = min(i + 1u, n - 1u);
  let t = x - f32(i);
  let d0 = lut_fetch(off + i);
  if (t == 0.0) { return d0; }
  let d1 = lut_fetch(off + j);
  return (1.0 - t) * d0 + t * d1;
}

// lookup_table_read_2D: rows y (outer), layout d[off + y·nx + x].
fn lut_read_2d(x: f32, y_in: f32, off: u32, nx: u32, ny: u32) -> f32 {
  let y = saturate(y_in) * f32(ny - 1u);
  let i = min(u32(y), ny - 1u);
  let j = min(i + 1u, ny - 1u);
  let t = y - f32(i);
  let d0 = lut_read(x, off + nx * i, nx);
  if (t == 0.0) { return d0; }
  let d1 = lut_read(x, off + nx * j, nx);
  return (1.0 - t) * d0 + t * d1;
}

// lookup_table_read_3D: slices z (outer) → rows y → x (inner), layout d[off + z·nx·ny + y·nx + x].
fn lut_read_3d(x: f32, y: f32, z_in: f32, off: u32, nx: u32, ny: u32, nz: u32) -> f32 {
  let z = saturate(z_in) * f32(nz - 1u);
  let i = min(u32(z), nz - 1u);
  let j = min(i + 1u, nz - 1u);
  let t = z - f32(i);
  let d0 = lut_read_2d(x, y, off + nx * ny * i, nx, ny);
  if (t == 0.0) { return d0; }
  let d1 = lut_read_2d(x, y, off + nx * ny * j, nx, ny);
  return (1.0 - t) * d0 + t * d1;
}

/// S_ior(rough, mu, z): dielectric generalized-Schlick albedo factor (bsdf_microfacet.h:441-451).
fn lut_ggx_gen_schlick_ior_s(rough: f32, mu: f32, z: f32) -> f32 { return lut_read_3d(rough, mu, z, $LUT_IOR_S, 16u, 16u, 16u); }
/// S(rough, mu, z) for exponent ≥ 0; the F82 metal estimate reads it at z = 0.5 (bsdf_microfacet.h:465-479).
fn lut_ggx_gen_schlick_s(rough: f32, mu: f32, z: f32) -> f32 { return lut_read_3d(rough, mu, z, $LUT_S, 16u, 16u, 16u); }
/// E(rough, mu): single-scatter GGX albedo with F = 1 (Tier 2; also the V1 glossy albedo).
fn lut_ggx_E(rough: f32, mu: f32) -> f32 { return lut_read_2d(rough, mu, $LUT_E, 32u, 32u); }
/// Eavg(rough): cosine-weighted average of E (Tier 2).
fn lut_ggx_Eavg(rough: f32) -> f32 { return lut_read(rough, $LUT_EAVG, 32u); }
