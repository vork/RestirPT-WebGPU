// Web Worker entry for glTF loading (plan §1.1: workers hand data back with transfer lists, no SAB).
// Spawned by load-scene.ts: new Worker(new URL('./gltf-loader.worker.ts', import.meta.url), { type: 'module' }).
import { loadGltf, sceneTransferList, type GltfWorkerRequest, type GltfWorkerResponse } from './gltf-loader.ts';
import { browserImageDecoder, canDecodeInThisRuntime } from './image-decode.ts';

const scope = self as unknown as DedicatedWorkerGlobalScope;

scope.onmessage = async (ev: MessageEvent<GltfWorkerRequest>) => {
  const { id, source, tangents, quantize } = ev.data;
  try {
    const result = await loadGltf(source, { decodeImage: canDecodeInThisRuntime() ? browserImageDecoder : undefined, tangents, quantize });
    const msg: GltfWorkerResponse = { id, ok: true, result };
    scope.postMessage(msg, sceneTransferList(result.scene));
  } catch (e) {
    const msg: GltfWorkerResponse = { id, ok: false, error: e instanceof Error ? `${e.message}\n${e.stack ?? ''}` : String(e) };
    scope.postMessage(msg);
  }
};
