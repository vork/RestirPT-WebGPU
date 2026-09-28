// Module Worker: USD bytes → SceneData (LightUSD next backend + adapter), transferred back (plan §1.1).
import { sceneTransferList } from '../gltf-loader.ts';
import { loadUsdInline, type UsdWorkerRequest, type UsdWorkerResponse } from './load-usd.ts';

const ctx = self as unknown as { onmessage: ((ev: MessageEvent<UsdWorkerRequest>) => void) | null; postMessage(msg: UsdWorkerResponse, transfer?: Transferable[]): void };

ctx.onmessage = async (ev) => {
  const { id, bytes, name, opts } = ev.data;
  try {
    const result = await loadUsdInline(new Uint8Array(bytes), name, opts);
    ctx.postMessage({ id, ok: true, result }, sceneTransferList(result.scene));
  } catch (e) {
    ctx.postMessage({ id, ok: false, error: e instanceof Error ? e.message : String(e) });
  }
};
