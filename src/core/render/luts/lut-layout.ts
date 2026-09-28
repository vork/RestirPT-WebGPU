// Cycles BSDF LUT block (plan §1.5, §1.8 "records"; math.md#bsdf-v2 LUT table; gap-bsdf §3.5).
// The four tables are concatenated into ONE f32 block that the owner of the `records` storage buffer appends at a
// float offset `base`; material/lut.wgsl reads them with an exact port of Cycles' lookup_table_read* (never through
// a filtered texture). WGSL side: `lutDefines(base)` supplies $LUT_BASE / $LUT_IOR_S / $LUT_S / $LUT_E / $LUT_EAVG.
import type { Defines } from '../../gpu/wgsl-composer.ts';
import { TABLE_GGX_E, TABLE_GGX_EAVG, TABLE_GGX_GEN_SCHLICK_IOR_S, TABLE_GGX_GEN_SCHLICK_S } from './cycles-luts.ts';

/** Float offsets inside the LUT block (math.md#bsdf-v2 table; identical to gap-bsdf §11). */
export const LUT_LAYOUT = {
  ggxGenSchlickIorS: 0,   // 16^3, x = rough, y = mu, z = sqrt|(eta-1)/(eta+1)|
  ggxGenSchlickS: 4096,   // 16^3, z = 1/(0.2*exponent+1) (metal sample weight: z = 0.5)
  ggxE: 8192,             // 32^2, Tier 2
  ggxEavg: 9216,          // 32,   Tier 2
  floats: 9248,
} as const;

let block: Float32Array | undefined;

/** The LUT block (9248 f32 = 36,992 B), built once. */
export function lutFloats(): Float32Array {
  if (!block) {
    block = new Float32Array(LUT_LAYOUT.floats);
    block.set(TABLE_GGX_GEN_SCHLICK_IOR_S, LUT_LAYOUT.ggxGenSchlickIorS);
    block.set(TABLE_GGX_GEN_SCHLICK_S, LUT_LAYOUT.ggxGenSchlickS);
    block.set(TABLE_GGX_E, LUT_LAYOUT.ggxE);
    block.set(TABLE_GGX_EAVG, LUT_LAYOUT.ggxEavg);
  }
  return block;
}

/** Element type of the `records` array the LUT block lives in (material/lut.wgsl LUT_RECORDS_KIND). */
export type LutRecordsKind = 'f32' | 'u32' | 'vec4f' | 'vec4u';
const KIND_CODE: Record<LutRecordsKind, number> = { f32: 0, u32: 1, vec4f: 2, vec4u: 3 };

export interface LutDefineOptions {
  /** Float (4-byte word) offset of the LUT block inside `records`. Default 0. */
  base?: number;
  /** Element type of `records` (default f32). u32/vec4u words are bitcast. */
  recordsKind?: LutRecordsKind;
  /**
   * Declare `records` inside material/lut.wgsl (standalone use: tests, or a pass whose only records user is the BSDF).
   * When false (default) the includer declares `var<storage, read> records: array<...>` itself.
   */
  declare?: { group: number; binding: number };
}

/** Composer defines for material/lut.wgsl (and everything that includes it: bsdf.wgsl, material-eval.wgsl). */
export function lutDefines(opts: LutDefineOptions = {}): Defines {
  const base = opts.base ?? 0;
  if (!Number.isInteger(base) || base < 0) throw new Error(`LUT base must be a non-negative integer word offset (got ${base})`);
  return {
    LUT_BASE: `${base}u`,
    LUT_IOR_S: `${base + LUT_LAYOUT.ggxGenSchlickIorS}u`,
    LUT_S: `${base + LUT_LAYOUT.ggxGenSchlickS}u`,
    LUT_E: `${base + LUT_LAYOUT.ggxE}u`,
    LUT_EAVG: `${base + LUT_LAYOUT.ggxEavg}u`,
    LUT_RECORDS_KIND: KIND_CODE[opts.recordsKind ?? 'f32'],
    LUT_DECLARE_BINDING: opts.declare !== undefined,
    LUT_GROUP: opts.declare?.group ?? 0,
    LUT_BINDING: opts.declare?.binding ?? 0,
  };
}

/** Standalone LUT storage buffer (tests; passes that bind the LUT block on its own). */
export function createLutBuffer(device: GPUDevice, label = 'cycles-luts'): GPUBuffer {
  const data = lutFloats();
  const buf = device.createBuffer({ label, size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
  new Float32Array(buf.getMappedRange()).set(data);
  buf.unmap();
  return buf;
}
