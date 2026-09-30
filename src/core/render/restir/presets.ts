// ReSTIR settings and the rung / mode presets (restir-api.md §4.6, §6.3; PLAN §3 modes, §7.1 rungs 3.1/3.1b/3.2).
// Criteria mode, plants, RR, S, NS and rounds are UNIFORMS (no recompiles, §4.5).
import { RS_WGSL_CONSTS as K } from './layout.ts';

export interface RestirSettings {
  /** Cycles max_bounces N (≤ N+1 scattering vertices, the PT's semantics). */
  maxBounces: number;
  /** Russian roulette at initial sampling (D11: the PT's q = min(sqrt(max_c β_c), 1) for B > rrMinBounces). */
  rr: boolean; rrMinBounces: number;
  /** S path trees per pixel streamed into one reservoir (offline 32). */
  trees: number;
  /** Paired spatial rounds and slots per round (NS ≤ 6), pairing disk radius R in px. */
  rounds: number; slots: number; diskRadius: number;
  criteria: 'enhanced' | '2022';
  /** rc threshold τ (thr = τ·R²_pri) and α_min (perceptual roughness). */
  tau: number; alphaMin: number;
  /** Planted controls (validation only): omitted spatial J, marginal pdfs in J, W × wScale. */
  plant?: { noJ?: boolean; marginalJ?: boolean; wScale?: number };
}

export type RestirPresetName = 'initial' | 'initial-rr' | 'offline' | 'interactive' | 'criteria2022';

export const DEFAULT_RESTIR_SETTINGS: RestirSettings = {
  maxBounces: 3, rr: false, rrMinBounces: 3, trees: 1, rounds: 0, slots: 3, diskRadius: 30,
  criteria: 'enhanced', tau: 2e-4, alphaMin: 0.2,
};

const OFFLINE: Partial<RestirSettings> = { trees: 32, rounds: 3, slots: 6, diskRadius: 10, rr: false };

export const RESTIR_PRESETS: Record<RestirPresetName, Partial<RestirSettings>> = {
  initial: { trees: 1, rounds: 0, rr: false },
  'initial-rr': { trees: 1, rounds: 0, rr: true, rrMinBounces: 1 },
  offline: OFFLINE,
  interactive: { trees: 1, rounds: 1, slots: 3, diskRadius: 30, rr: true, rrMinBounces: 3 },
  criteria2022: { ...OFFLINE, criteria: '2022' },
};

/** Default settings ⊕ preset ⊕ overrides (maxBounces etc. come from the scene package). */
export function restirSettings(preset?: RestirPresetName, overrides: Partial<RestirSettings> = {}): RestirSettings {
  const s = { ...DEFAULT_RESTIR_SETTINGS, ...(preset ? RESTIR_PRESETS[preset] : {}), ...overrides };
  validateSettings(s);
  return s;
}

export function validateSettings(s: RestirSettings): void {
  const int = (x: number, lo: number, hi: number, n: string) => {
    if (!(Number.isInteger(x) && x >= lo && x <= hi)) throw new Error(`RestirSettings.${n} = ${x} not in [${lo}, ${hi}]`);
  };
  int(s.maxBounces, 0, K.RS_MAX_D - 2, 'maxBounces');
  int(s.trees, 1, K.RS_MAX_TREES, 'trees');
  int(s.rounds, 0, K.RS_MAX_ROUNDS, 'rounds');
  int(s.slots, 1, K.RS_MAX_SLOTS, 'slots');
  int(s.rrMinBounces, 0, 64, 'rrMinBounces');
  if (!(s.tau > 0) || !(s.alphaMin >= 0)) throw new Error('RestirSettings: tau > 0 and alphaMin ≥ 0 required');
}

/** RestirParams.flags (RSF_*) of the settings (ensemble / interactive bits are added by the kernel). */
export function restirFlags(s: RestirSettings): number {
  let f = 0;
  if (s.rr) f |= K.RSF_RR;
  if (s.criteria === '2022') f |= K.RSF_CRIT_2022;
  if (s.plant?.noJ) f |= K.RSF_PLANT_NO_J;
  if (s.plant?.marginalJ) f |= K.RSF_PLANT_MARGINAL_J;
  return f;
}

/** Logical sizes W_s of the 8 pairing-map layers (§2.8; M4 defaults for 6 slots, layers 6–7 unused). */
export const PAIR_TEX_SIZES = [254, 246, 238, 230, 222, 210, 0, 0];
