// All WGSL sources keyed by path relative to this directory (e.g. 'common/rng.wgsl').
// Works under Vite, Vitest (node + browser) because import.meta.glob is a Vite transform.
const modules = import.meta.glob('./**/*.wgsl', { query: '?raw', import: 'default', eager: true }) as Record<string, string>;

export const shaderSources: Record<string, string> = Object.fromEntries(
  Object.entries(modules).map(([k, v]) => [k.replace(/^\.\//, ''), v]),
);
