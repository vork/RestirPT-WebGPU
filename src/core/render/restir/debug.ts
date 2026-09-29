// ReSTIR debug views (ids 400–499) and probe decoding for the inspector (restir-api.md §2.11). OWNER WP-D. P0 STUB:
// no views registered; probe decoding returns nothing.
import type { DebugViewDef } from '../debug-views.ts';

/** View definitions registered by the renderer in ReSTIR mode (P0: none). */
export const RESTIR_VIEWS: DebugViewDef[] = [];

export interface RestirProbeDecoded { reservoirs: unknown[]; candidates: unknown[]; vertices: unknown[]; slots: unknown[] }
export function decodeRestirProbe(_records: ArrayBuffer): RestirProbeDecoded {
  return { reservoirs: [], candidates: [], vertices: [], slots: [] };
}
