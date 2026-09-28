// Scene/env loading plumbing: drag-and-drop, file picker and URL parameters (?scene=<url>&env=<url>) resolved into
// SceneSource objects and handed to an injected SceneLoader (the glTF/USD/HDRI loaders are wired by the integrator).
import type { EnvironmentData, SceneData } from '../core/scene/types.ts';

export type SceneSource =
  | { kind: 'url'; url: string; name: string }
  /** `main` is the scene/env file; `files` holds every dropped file (glTF .bin/textures resolve by name). */
  | { kind: 'files'; main: File; files: File[]; name: string };

export interface LoadProgressEvent {
  /** 0..1, or undefined for indeterminate stages. */
  fraction?: number;
  stage: string;
}
export type LoadProgress = (e: LoadProgressEvent) => void;

export interface SceneLoader {
  loadScene(src: SceneSource, progress: LoadProgress, signal: AbortSignal): Promise<SceneData>;
  loadEnvironment?(src: SceneSource, progress: LoadProgress, signal: AbortSignal): Promise<EnvironmentData>;
}

export type FileRole = 'scene' | 'env' | 'aux' | 'unknown';

const SCENE_EXT = ['glb', 'gltf', 'usd', 'usda', 'usdc', 'usdz'];
const ENV_EXT = ['hdr', 'exr'];
const AUX_EXT = ['bin', 'png', 'jpg', 'jpeg', 'webp', 'ktx2', 'avif'];

export function extensionOf(name: string): string {
  const clean = name.split(/[?#]/)[0];
  const i = clean.lastIndexOf('.');
  return i >= 0 ? clean.slice(i + 1).toLowerCase() : '';
}

export function classifyFile(name: string): FileRole {
  const e = extensionOf(name);
  if (SCENE_EXT.includes(e)) return 'scene';
  if (ENV_EXT.includes(e)) return 'env';
  if (AUX_EXT.includes(e)) return 'aux';
  return 'unknown';
}

export interface ResolvedSources { scene?: SceneSource; env?: SceneSource; ignored: string[] }

/** Pick the scene and env files out of a dropped/picked set; auxiliary files travel with the scene. */
export function sourcesFromFiles(files: File[]): ResolvedSources {
  const out: ResolvedSources = { ignored: [] };
  for (const f of files) {
    const role = classifyFile(f.name);
    if (role === 'scene' && !out.scene) out.scene = { kind: 'files', main: f, files, name: f.name };
    else if (role === 'env' && !out.env) out.env = { kind: 'files', main: f, files: [f], name: f.name };
    else if (role !== 'aux') out.ignored.push(f.name); // unknown, or a second scene/env file
  }
  return out;
}

/** ?scene=<url>&env=<url>, resolved against the page URL. Only http(s)/blob URLs are accepted. */
export function sourcesFromQuery(search: string, base: string): ResolvedSources {
  const q = new URLSearchParams(search);
  const out: ResolvedSources = { ignored: [] };
  for (const key of ['scene', 'env'] as const) {
    const raw = q.get(key);
    if (!raw) continue;
    let u: URL;
    try { u = new URL(raw, base); } catch { out.ignored.push(`${key}=${raw} (bad URL)`); continue; }
    if (!['http:', 'https:', 'blob:'].includes(u.protocol)) { out.ignored.push(`${key}=${raw} (protocol)`); continue; }
    const name = decodeURIComponent(u.pathname.split('/').pop() || u.href);
    out[key] = { kind: 'url', url: u.href, name };
  }
  return out;
}

/** Window-level drag-and-drop. `onFiles` receives every dropped file. Returns a disposer. */
export function installDropTarget(target: HTMLElement, onFiles: (files: File[]) => void, onHover?: (on: boolean) => void): () => void {
  const ac = new AbortController();
  const o = { signal: ac.signal };
  let depth = 0;
  const hasFiles = (e: DragEvent) => !!e.dataTransfer && [...e.dataTransfer.types].includes('Files');
  target.addEventListener('dragenter', (e) => { if (!hasFiles(e)) return; e.preventDefault(); depth++; onHover?.(true); }, o);
  target.addEventListener('dragover', (e) => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer!.dropEffect = 'copy'; }, o);
  target.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (depth === 0) onHover?.(false); }, o);
  target.addEventListener('drop', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    onHover?.(false);
    const files = [...(e.dataTransfer?.files ?? [])];
    if (files.length) onFiles(files);
  }, o);
  return () => ac.abort();
}

/** Open a native file picker (multiple files so .gltf + .bin + textures can be selected together). */
export function pickFiles(accept = [...SCENE_EXT, ...ENV_EXT, ...AUX_EXT].map((e) => `.${e}`).join(',')): Promise<File[]> {
  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = accept;
    input.addEventListener('change', () => resolve([...(input.files ?? [])]), { once: true });
    input.addEventListener('cancel', () => resolve([]), { once: true });
    input.click();
  });
}
