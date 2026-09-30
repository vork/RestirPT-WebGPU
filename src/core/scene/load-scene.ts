// Main-thread scene loading: URL, drag-and-drop files, or raw bytes → SceneData via the glTF Worker.
// Falls back to loading on the calling thread when Workers are unavailable (or useWorker: false).
// USD goes through its own path (docs/decisions/usd.md); this module handles .glb/.gltf only.
import { isGlb, loadGltf, type GltfLoadResult, type GltfSource, type GltfWorkerRequest, type GltfWorkerResponse } from './gltf-loader.ts';
import { browserImageDecoder, canDecodeInThisRuntime } from './image-decode.ts';

export type SceneInput = string | URL | Blob | readonly File[] | FileList | ArrayBuffer | Uint8Array;

export interface LoadSceneOptions {
  /** Default true when `Worker` exists. */
  useWorker?: boolean;
  /** MikkTSpace tangents for normal-mapped primitives (default true). */
  tangents?: boolean;
  /** quantizeScene mode (default 'quantized'; data-formats.md §B0). */
  quantize?: 'quantized' | 'lossless';
}

export async function loadScene(input: SceneInput, opts: LoadSceneOptions = {}): Promise<GltfLoadResult> {
  const { source, transfer } = await toSource(input);
  const useWorker = opts.useWorker ?? typeof Worker !== 'undefined';
  if (!useWorker) {
    return loadGltf(source, { decodeImage: canDecodeInThisRuntime() ? browserImageDecoder : undefined, tangents: opts.tangents, quantize: opts.quantize });
  }
  return runInWorker({ id: nextId++, source, tangents: opts.tangents, quantize: opts.quantize }, transfer);
}

let nextId = 1;

function runInWorker(req: GltfWorkerRequest, transfer: ArrayBuffer[]): Promise<GltfLoadResult> {
  const worker = new Worker(new URL('./gltf-loader.worker.ts', import.meta.url), { type: 'module', name: 'gltf-loader' });
  return new Promise<GltfLoadResult>((resolve, reject) => {
    worker.onmessage = (ev: MessageEvent<GltfWorkerResponse>) => {
      if (ev.data.id !== req.id) return;
      worker.terminate();
      if (ev.data.ok) resolve(ev.data.result); else reject(new Error(`glTF load failed: ${ev.data.error}`));
    };
    worker.onerror = (ev) => { worker.terminate(); reject(new Error(`glTF worker error: ${ev.message}`)); };
    worker.postMessage(req, transfer);
  });
}

/** Normalize any supported input into a GltfSource (+ buffers we own and may transfer to the Worker). */
export async function toSource(input: SceneInput): Promise<{ source: GltfSource; transfer: ArrayBuffer[] }> {
  if (typeof input === 'string' || input instanceof URL) {
    const url = new URL(String(input), globalThis.location?.href).href;
    return { source: { kind: 'url', url, name: decodeURIComponent(url.split('/').pop() ?? 'scene') }, transfer: [] };
  }
  if (input instanceof ArrayBuffer || input instanceof Uint8Array) {
    const bytes = input instanceof Uint8Array ? input.slice() : new Uint8Array(input.slice(0)); // private copy → transferable
    if (!isGlb(bytes)) throw new Error('raw bytes must be a GLB (use files or a URL for .gltf with resources)');
    return { source: { kind: 'glb', bytes, name: 'scene.glb' }, transfer: [bytes.buffer] };
  }
  const files = input instanceof Blob ? [input as File] : Array.from(input as ArrayLike<File>);
  return filesToSource(files);
}

/** Drag-and-drop: one .glb, or one .gltf plus its buffers/images (matched by file name, ignoring directories). */
export async function filesToSource(files: File[]): Promise<{ source: GltfSource; transfer: ArrayBuffer[] }> {
  const nameOf = (f: File) => f.name || 'scene';
  const glb = files.find((f) => /\.glb$/i.test(nameOf(f)));
  const gltf = files.find((f) => /\.gltf$/i.test(nameOf(f)));
  if (glb && !gltf) {
    const bytes = new Uint8Array(await glb.arrayBuffer());
    return { source: { kind: 'glb', bytes, name: nameOf(glb) }, transfer: [bytes.buffer] };
  }
  if (!gltf) {
    if (files.length === 1) { // unknown extension: sniff GLB magic
      const bytes = new Uint8Array(await files[0].arrayBuffer());
      if (isGlb(bytes)) return { source: { kind: 'glb', bytes, name: nameOf(files[0]) }, transfer: [bytes.buffer] };
    }
    throw new Error(`no .glb/.gltf among dropped files: ${files.map(nameOf).join(', ')}`);
  }
  const json = JSON.parse(await gltf.text()) as { buffers?: { uri?: string }[]; images?: { uri?: string }[] };
  const byName = new Map(files.map((f) => [nameOf(f), f]));
  const resources: Record<string, Uint8Array> = {};
  const transfer: ArrayBuffer[] = [];
  const missing: string[] = [];
  for (const r of [...(json.buffers ?? []), ...(json.images ?? [])]) {
    const uri = r.uri;
    if (!uri || uri.startsWith('data:') || uri in resources) continue;
    const base = decodeURIComponent(uri).split('/').pop()!;
    const f = byName.get(base) ?? byName.get(uri);
    if (!f) { missing.push(uri); continue; }
    const bytes = new Uint8Array(await f.arrayBuffer());
    resources[uri] = bytes;
    transfer.push(bytes.buffer);
  }
  if (missing.length) throw new Error(`glTF resources missing from the drop: ${missing.join(', ')}`);
  return { source: { kind: 'gltf', json: json as never, resources, name: nameOf(gltf) }, transfer };
}
