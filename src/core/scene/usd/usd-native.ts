// LightUSD rc4 `next` backend (lightusd_next.wasm, RenderStream) → plain raw data (docs/decisions/usd.md "Adapter
// contract"). Runs in a module Worker (browser) or inline (Node tests). Wasm heap views are copied out immediately.
// Ported from validation/spikes/usd/usd-direct.worker.ts.
import { scanUsda, type UsdaScan } from './usda-scan.ts';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Rec = Record<string, any>; // LightUSD output is untyped

export interface UsdRaw {
  name: string;
  metadata: Rec;
  meshes: Rec[];            // getMesh(i) with points/indices/normals/uv0 replaced by typed-array copies
  nodes: Rec[];
  lights: Rec[];
  cameras: Rec[];
  pointInstancers: Rec[];
  pointInstanceDraws: Rec[];
  unsupportedRenderables: unknown[];
  layerScan?: UsdaScan;
  layerScanError?: string;
  /** Other USD layers inside a .usdz (the scan only covers the root layer). */
  extraLayers: string[];
  timings: Record<string, number>;
}

const isNode = () => typeof process !== 'undefined' && !!process.versions?.node && typeof window === 'undefined'
  && typeof (globalThis as { WorkerGlobalScope?: unknown }).WorkerGlobalScope === 'undefined';

let nativePromise: Promise<Rec> | undefined;
let initMs = 0;

export function loadLightUsdNext(): Promise<Rec> {
  nativePromise ??= (async () => {
    const t0 = performance.now();
    const { default: create } = await import('lightusd/lightusd_next.js');
    let native: Rec;
    if (isNode()) {
      native = await create({}); // emscripten resolves the .wasm next to the module through fs
    } else {
      const wasmUrl = (await import('lightusd/lightusd_next.wasm?url')).default;
      native = await create({ locateFile: (p: string) => (p.endsWith('.wasm') ? wasmUrl : p) });
    }
    initMs = performance.now() - t0;
    return native;
  })();
  nativePromise.catch(() => { nativePromise = undefined; });
  return nativePromise;
}

/** USDZ = uncompressed zip. */
export function unzipStored(u8: Uint8Array): { name: string; data: Uint8Array }[] {
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

function copyView(native: Rec, desc: Rec | undefined | null): ArrayBufferView | null {
  if (!desc || !Number.isFinite(desc.ptr) || !desc.length) return null;
  const Ctor = VIEW[desc.dtype];
  if (!Ctor) return null;
  const heap = (native.HEAPU8 as Uint8Array).buffer as ArrayBuffer;
  return (new Ctor(heap, desc.ptr, desc.length) as unknown as { slice(): ArrayBufferView }).slice();
}

const isUsdaText = (b: Uint8Array) => b.length >= 8 && new TextDecoder().decode(b.subarray(0, 8)) === '#usda 1.';

/** Parse one USD file (.usda/.usdc/.usdz bytes) into raw LightUSD data plus the root-layer scan. */
export async function extractUsd(bytes: Uint8Array, name: string): Promise<UsdRaw> {
  const t0 = performance.now();
  const native = await loadLightUsdNext();
  const tInit = performance.now();
  const rs = new native.RenderStream() as Rec;
  const extraLayers: string[] = [];
  try {
    rs.setBuildVertexIndices(true);
    let root = bytes;
    if (/\.usdz$/i.test(name) || (bytes[0] === 0x50 && bytes[1] === 0x4b)) {
      const entries = unzipStored(bytes);
      const rootEntry = entries.find((x) => /\.usd[ac]?$/i.test(x.name));
      if (!rootEntry) throw new Error('usdz has no USD root layer');
      for (const x of entries) if (x !== rootEntry && /\.usd[ac]?$/i.test(x.name)) { rs.provideAsset(x.name, x.data); extraLayers.push(x.name); }
      root = rootEntry.data;
    }
    const begin = rs.begin(root) as Rec;
    if (!begin?.success) throw new Error(begin?.error || rs.error?.() || 'RenderStream.begin failed');
    const tBegin = performance.now();
    const list = (n: number, get: (i: number) => Rec): Rec[] => Array.from({ length: n }, (_, i) => get(i));
    const meshes = list(rs.meshCount(), (i) => {
      const m = rs.getMesh(i) as Rec;
      const out: Rec = { ...m };
      for (const k of ['points', 'indices', 'normals', 'uv0']) out[k] = copyView(native, m[k]);
      delete out.tangents;
      return out;
    });
    const raw: UsdRaw = {
      name,
      metadata: rs.getSceneMetadata() as Rec,
      meshes,
      nodes: list(rs.nodeCount(), (i) => rs.getNode(i)),
      lights: list(rs.lightCount(), (i) => rs.getLight(i)),
      cameras: list(rs.cameraCount(), (i) => rs.getCamera(i)),
      pointInstancers: list(rs.pointInstancerCount(), (i) => rs.getPointInstancer(i)),
      pointInstanceDraws: list(rs.pointInstanceDrawCount(), (i) => rs.getPointInstanceDraw(i)),
      unsupportedRenderables: rs.getUnsupportedRenderables() ?? [],
      extraLayers,
      timings: {},
    };
    const tExtract = performance.now();
    // Root-layer scan for the fields rc4 drops (treatAsPoint, spot radius, ior, clearcoat, specular, doc, instancing).
    try {
      if (isUsdaText(root)) raw.layerScan = scanUsda(new TextDecoder().decode(root));
      else {
        const conv = new native.NextUSDZConverterNative() as Rec;
        try {
          if (conv.loadFromBinary(root, name.replace(/\.usdz$/i, '.usdc'))) raw.layerScan = scanUsda(conv.exportAsUSDA());
          else raw.layerScanError = conv.error?.() || 'loadFromBinary failed';
        } finally { conv.delete?.(); }
      }
    } catch (e) {
      raw.layerScanError = e instanceof Error ? e.message : String(e);
    }
    const tScan = performance.now();
    raw.timings = { wasmInitMs: initMs, beginMs: tBegin - tInit, extractMs: tExtract - tBegin, layerScanMs: tScan - tExtract, totalMs: tScan - t0 };
    return raw;
  } finally {
    try { rs.end(); rs.delete(); } catch { /* already released */ }
  }
}
