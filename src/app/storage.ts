// localStorage wrappers: per-viewer conveniences only; every access may throw (private mode, blocked storage).
const PREFIX = 'restirpt:';

export function loadPref<T>(key: string, fallback: T): T {
  try {
    const s = globalThis.localStorage?.getItem(PREFIX + key);
    return s == null ? fallback : { ...fallback, ...(JSON.parse(s) as T) };
  } catch {
    return fallback;
  }
}

export function savePref(key: string, value: unknown): void {
  try { globalThis.localStorage?.setItem(PREFIX + key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}
