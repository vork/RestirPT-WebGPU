// T16 "no denoiser in validation readbacks" (PLAN §7.4; docs/decisions/denoiser.md DN4, §11). harness.ts writes the
// measured page state into every validation run's meta.json (`t16.denoiser`, 'none' = no live Denoiser object on the
// page before and after the run). Pure helpers for the gates.

/**
 * Problems of one validation run's meta.json. `requirePresent`: runs rendered by this gate invocation must carry the
 * field; cached references from before M5.5 have none (they were rendered by harness code without a denoiser) and are
 * accepted when `requirePresent` is false.
 */
export function denoiserT16Problems(meta: Record<string, any> | undefined, label: string, requirePresent: boolean): string[] {
  const d = meta?.t16?.denoiser;
  if (d === undefined) return requirePresent ? [`${label}: no t16.denoiser record (harness.ts T16 wrapper not applied)`] : [];
  return d === 'none' ? [] : [`${label}: denoiser active in a validation readback (${String(d)})`];
}
