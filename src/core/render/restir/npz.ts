// ensemble.npz writer (restir-api.md §2.10; compare.py README "Ensemble .npz format"). OWNER WP-C. P0 STUB.
export interface NpzArray { name: string; shape: number[]; dtype: '<f8' | '<f4' | '<u4' | '<i8'; data: ArrayBufferView }

/** Stored (uncompressed) zip of .npy arrays. STUB (P0): not implemented yet. */
export function writeNpz(_arrays: NpzArray[]): Uint8Array {
  throw new Error('npz.ts: writeNpz is not implemented yet (WP-C)');
}
