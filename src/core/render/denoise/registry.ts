// Page-global registry of live denoisers (docs/decisions/denoiser.md DN4, §11 T16). Every `Denoiser` registers itself
// on construction and unregisters on destroy(). The validation harness (validation/harness/harness.ts) checks it before
// and after every validation readback and records the result as `t16.denoiser` in the run's meta.json; the gates
// require 'none'. No GPU code here, so the harness can import it without pulling in the renderer.

const live = new Map<object, () => string>();

/** Register a live denoiser; `describe` returns a one-line state (mode, enabled) for the T16 record. */
export function registerDenoiser(d: object, describe: () => string): void { live.set(d, describe); }
export function unregisterDenoiser(d: object): void { live.delete(d); }
/** Number of live denoiser objects on this page (enabled or not). */
export function liveDenoisers(): number { return live.size; }

/** T16 value: 'none' when no denoiser object exists on the page, else a description of the live ones. */
export function denoiserT16State(): string {
  if (live.size === 0) return 'none';
  return `${live.size} live: ${[...live.values()].map((f) => f()).join('; ')}`;
}
