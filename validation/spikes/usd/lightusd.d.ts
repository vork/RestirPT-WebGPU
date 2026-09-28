// Minimal types for the parts of lightusd@1.0.0-rc4 the spike touches (the package ships no .d.ts).
declare module 'lightusd' {
  export interface LightUSDInitOptions {
    backend?: 'next' | 'legacy' | 'auto';
    useZstdCompressedWasm?: boolean;
    useMemory64?: boolean;
  }

  /** Plain-object scene handed back by LightUSDWorker.js (extractSceneData); fields vary by backend. */
  export interface WorkerLoadResult {
    _data: Record<string, unknown>;
    numMeshes(): number;
    numLights(): number;
    numMaterials(): number;
    getUpAxis(): string;
    getSceneMetadata(): Record<string, unknown>;
  }

  export class LightUSDWorkerLoader {
    constructor(options?: { workerUrl?: URL | string; onProgress?: (p: unknown) => void });
    init(options?: LightUSDInitOptions): Promise<void>;
    load(url: string, options?: LightUSDInitOptions): Promise<WorkerLoadResult>;
    parse(binary: Uint8Array, filename?: string, options?: LightUSDInitOptions): Promise<WorkerLoadResult>;
    dispose(): void;
  }

  export class LightUSDLoader {
    constructor(manager?: unknown, options?: LightUSDInitOptions);
    init(options?: LightUSDInitOptions): Promise<void>;
    parseAsync(binary: ArrayBuffer | Uint8Array, filePath?: string, options?: LightUSDInitOptions): Promise<unknown>;
  }
}

// Emscripten factory of the next-only module (RenderStream). Deep import allowed by the package exports map.
declare module 'lightusd/lightusd_next.js' {
  const createLightUSDNext: (moduleArg?: { locateFile?: (path: string, prefix: string) => string }) => Promise<Record<string, any>>;
  export default createLightUSDNext;
}
