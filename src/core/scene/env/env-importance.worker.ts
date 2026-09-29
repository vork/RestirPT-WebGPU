// Worker entry for the env importance tables (plan §1.4b "Tables are built in a Worker"; §1.1: transfer lists, no
// SharedArrayBuffer). Input texels are copied by the client; the tables come back transferred.
/// <reference lib="webworker" />
import { buildEnvImportance, type EnvImportanceOptions } from './env-importance.ts';

export interface EnvImportanceRequest { id: number; texels: Float32Array; width: number; height: number; options?: EnvImportanceOptions }

self.onmessage = (ev: MessageEvent<EnvImportanceRequest>) => {
  const { id, texels, width, height, options } = ev.data;
  const post = (self as unknown as DedicatedWorkerGlobalScope).postMessage.bind(self);
  try {
    const t = buildEnvImportance(texels, width, height, options);
    post({ id, table: t }, [t.rowAlias.buffer, t.colAlias.buffer, t.pdfUV.buffer, t.targetUV.buffer] as ArrayBuffer[]);
  } catch (e) {
    post({ id, error: e instanceof Error ? e.message : String(e) });
  }
};
