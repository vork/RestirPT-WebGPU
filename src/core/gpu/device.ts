// DOM-free device creation shared by the app, the Chrome harness and the dawn.node test lane.
import { PROFILE_CHROME154_M5PRO, WANTED_FEATURES, type ProfileLimitName } from './profile.ts';

export interface GpuContext {
  device: GPUDevice;
  adapterInfo: GPUAdapterInfo;
  features: Set<string>;
  limits: Record<ProfileLimitName, number>;
  wgslLanguageFeatures: Set<string>;
  /** Limits where the adapter offered less than the profile (lane diff report). */
  belowProfile: Partial<Record<ProfileLimitName, number>>;
}

export interface CreateDeviceOptions {
  /** Throw if the adapter is a software/fallback adapter or not Apple Metal (headless validation runs). */
  requireHardwareMetal?: boolean;
  onLost?: (info: GPUDeviceLostInfo) => void;
  onUncapturedError?: (msg: string) => void;
  label?: string;
}

export class GpuUnavailableError extends Error {}

/** Request an adapter + device clamped to the Chrome154/M5 Pro profile. `gpu` is navigator.gpu or dawn.node's GPU. */
export async function createGpuContext(gpu: GPU, opts: CreateDeviceOptions = {}): Promise<GpuContext> {
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new GpuUnavailableError('WebGPU adapter unavailable');
  const info = adapter.info;
  if (opts.requireHardwareMetal) {
    const isFallback = (info as GPUAdapterInfo & { isFallbackAdapter?: boolean }).isFallbackAdapter === true;
    if (isFallback || info.vendor !== 'apple' || !info.architecture.startsWith('metal')) {
      throw new GpuUnavailableError(
        `Refusing non-hardware adapter: vendor=${info.vendor} arch=${info.architecture} fallback=${isFallback}`,
      );
    }
  }

  const requiredLimits: Record<string, number> = {};
  const belowProfile: Partial<Record<ProfileLimitName, number>> = {};
  const adapterLimits = adapter.limits as unknown as Record<string, number | undefined>;
  for (const [name, want] of Object.entries(PROFILE_CHROME154_M5PRO) as [ProfileLimitName, number][]) {
    const have = adapterLimits[name];
    if (have === undefined) continue; // limit unknown to this implementation (e.g. immediates on older Dawn)
    const v = Math.min(have, want);
    if (have < want) belowProfile[name] = have;
    requiredLimits[name] = v;
  }

  const requiredFeatures = WANTED_FEATURES.filter((f) => adapter.features.has(f)) as GPUFeatureName[];
  const device = await adapter.requestDevice({ requiredLimits, requiredFeatures, label: opts.label ?? 'restir-pt' });

  device.lost.then((lostInfo) => {
    // Chrome 140+: a lost device needs a fresh requestAdapter(); callers rebuild from CPU-side scene copies.
    opts.onLost?.(lostInfo);
  });
  device.addEventListener('uncapturederror', (ev) => {
    const msg = (ev as GPUUncapturedErrorEvent).error.message;
    if (opts.onUncapturedError) opts.onUncapturedError(msg);
    else console.error('[webgpu uncaptured]', msg);
  });

  const limits = {} as Record<ProfileLimitName, number>;
  const deviceLimits = device.limits as unknown as Record<string, number>;
  for (const name of Object.keys(PROFILE_CHROME154_M5PRO) as ProfileLimitName[]) {
    if (deviceLimits[name] !== undefined) limits[name] = deviceLimits[name];
  }
  const wgslLF = (gpu as GPU & { wgslLanguageFeatures?: Set<string> }).wgslLanguageFeatures;
  return {
    device,
    adapterInfo: info,
    features: new Set([...device.features] as string[]),
    limits,
    wgslLanguageFeatures: new Set(wgslLF ? [...wgslLF] : []),
    belowProfile,
  };
}

/** Summary used by the M0 lane diff (docs/decisions/platform-lanes.md). */
export function describeContext(ctx: GpuContext): Record<string, unknown> {
  return {
    vendor: ctx.adapterInfo.vendor,
    architecture: ctx.adapterInfo.architecture,
    description: ctx.adapterInfo.description,
    features: [...ctx.features].sort(),
    wgslLanguageFeatures: [...ctx.wgslLanguageFeatures].sort(),
    limits: ctx.limits,
    belowProfile: ctx.belowProfile,
  };
}
