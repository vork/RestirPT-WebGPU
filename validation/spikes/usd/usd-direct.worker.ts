// 'direct' mode: our own module Worker over lightusd_next.js (RenderStream / tydra-next), the shape the
// UsdSceneSource adapter should take. Needs no Vite config: the wasm URL comes from `?url`, not import.meta.url.
import createLightUSDNext from 'lightusd/lightusd_next.js';
import wasmUrl from 'lightusd/lightusd_next.wasm?url';
import { scanUsda } from './usda-scan.ts';

type Rec = Record<string, any>; // loader output is untyped

export interface DirectRequest { id: number; name: string; bytes: ArrayBuffer }
export interface DirectResponse {
  id: number;
  ok: boolean;
  error?: string;
  data?: Rec;
  timings?: Record<string, number>;
}

let nativePromise: Promise<Rec> | null = null;
let initMs = 0;

function init(): Promise<Rec> {
  nativePromise ??= (async () => {
    const t0 = performance.now();
    const native = await createLightUSDNext({ locateFile: (p: string) => (p.endsWith('.wasm') ? wasmUrl : p) });
    initMs = performance.now() - t0;
    return native;
  })();
  return nativePromise;
}

/** USDZ = uncompressed zip; RenderStream.begin() wants the root layer, other layers via provideAsset(). */
function unzipStored(u8: Uint8Array): { name: string; data: Uint8Array }[] {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const out: { name: string; data: Uint8Array }[] = [];
  let o = 0;
  while (o + 30 <= u8.length && dv.getUint32(o, true) === 0x04034b50) {
    const method = dv.getUint16(o + 8, true);
    const size = dv.getUint32(o + 18, true);
    const nameLen = dv.getUint16(o + 26, true);
    const extraLen = dv.getUint16(o + 28, true);
    const name = new TextDecoder().decode(u8.subarray(o + 30, o + 30 + nameLen));
    const start = o + 30 + nameLen + extraLen;
    if (method !== 0) throw new Error(`usdz entry ${name} is compressed (method ${method}); USDZ requires stored entries`);
    out.push({ name, data: u8.subarray(start, start + size) });
    o = start + size;
  }
  return out;
}

const VIEW: Record<string, new (b: ArrayBuffer, o: number, n: number) => ArrayBufferView> = {
  f32: Float32Array, u32: Uint32Array, i32: Int32Array, u16: Uint16Array, u8: Uint8Array,
};

/** Copy a {ptr,length,dtype} wasm heap view out before the next RenderStream call reuses it. */
function copyView(native: Rec, desc: Rec | undefined, transfer: ArrayBuffer[]): ArrayBufferView | null {
  if (!desc || !Number.isFinite(desc.ptr) || !desc.length) return null;
  const Ctor = VIEW[desc.dtype];
  if (!Ctor) return null;
  const heap = (native.HEAPU8 as Uint8Array).buffer as ArrayBuffer;
  const src = new Ctor(heap, desc.ptr, desc.length) as unknown as { slice(): ArrayBufferView };
  const copy = src.slice();
  transfer.push(copy.buffer as ArrayBuffer);
  return copy;
}

function extract(native: Rec, rs: Rec, transfer: ArrayBuffer[]): Rec {
  const list = (n: number, get: (i: number) => Rec): Rec[] => Array.from({ length: n }, (_, i) => get(i));
  const meshes = list(rs.meshCount(), (i) => {
    const m = rs.getMesh(i) as Rec;
    const out: Rec = { ...m };
    for (const k of ['points', 'indices', 'normals', 'uv0', 'tangents']) out[k] = copyView(native, m[k], transfer);
    return out;
  });
  return {
    metadata: rs.getSceneMetadata(),
    meshes,
    nodes: list(rs.nodeCount(), (i) => rs.getNode(i)),
    lights: list(rs.lightCount(), (i) => rs.getLight(i)),
    cameras: list(rs.cameraCount(), (i) => rs.getCamera(i)),
    pointInstancers: list(rs.pointInstancerCount(), (i) => rs.getPointInstancer(i)),
    pointInstanceDraws: list(rs.pointInstanceDrawCount(), (i) => rs.getPointInstanceDraw(i)),
    unsupportedRenderables: rs.getUnsupportedRenderables(),
    stats: rs.getStats(),
  };
}

self.onmessage = async (e: MessageEvent<DirectRequest>) => {
  const { id, name, bytes } = e.data;
  const t0 = performance.now();
  let rs: Rec | null = null;
  try {
    const native = await init();
    const tInit = performance.now();
    let root: Uint8Array = new Uint8Array(bytes);
    rs = new native.RenderStream() as Rec;
    if (/\.usdz$/i.test(name)) {
      const entries = unzipStored(root);
      const rootEntry = entries.find((x) => /\.usd[ac]?$/i.test(x.name));
      if (!rootEntry) throw new Error('usdz has no USD root layer');
      for (const x of entries) if (x !== rootEntry && /\.usd[ac]?$/i.test(x.name)) rs.provideAsset(x.name, x.data);
      root = rootEntry.data;
    }
    const begin = rs.begin(root) as Rec;
    if (!begin?.success) throw new Error(begin?.error || rs.error?.() || 'RenderStream.begin failed');
    const tBegin = performance.now();
    const transfer: ArrayBuffer[] = [];
    const data = extract(native, rs, transfer);
    const tExtract = performance.now();
    // Workaround probe: root layer -> USDA text -> scan fields the next backend drops (see usda-scan.ts).
    const conv = new native.NextUSDZConverterNative() as Rec;
    try {
      if (conv.loadFromBinary(new Uint8Array(bytes), name)) data.layerScan = scanUsda(conv.exportAsUSDA());
      else data.layerScanError = conv.error?.() || 'loadFromBinary failed';
    } finally {
      conv.delete?.();
    }
    const tScan = performance.now();
    const resp: DirectResponse = {
      id, ok: true, data,
      timings: {
        wasmInitMs: initMs, beginMs: tBegin - tInit, extractMs: tExtract - tBegin, layerScanMs: tScan - tExtract,
        workerTotalMs: tScan - t0,
      },
    };
    (self as unknown as Worker).postMessage(resp, transfer);
  } catch (err) {
    (self as unknown as Worker).postMessage({ id, ok: false, error: String((err as Error)?.stack ?? err) } satisfies DirectResponse);
  } finally {
    try { rs?.end(); rs?.delete(); } catch { /* already released */ }
  }
};
