// Module Worker: decodes an environment map off the main thread and transfers the texels back (plan §1.1).
import { decodeEnvironment, type LoadEnvOptions } from './load-env.ts';

const ctx = self as unknown as { onmessage: ((ev: MessageEvent) => void) | null; postMessage(msg: unknown, transfer?: Transferable[]): void };

ctx.onmessage = (ev: MessageEvent<{ bytes: ArrayBuffer; opts: LoadEnvOptions }>) => {
  try {
    const result = { ...decodeEnvironment(ev.data.bytes, ev.data.opts), decodedIn: 'worker' as const };
    ctx.postMessage({ ok: true, result }, [result.env.texels.buffer as ArrayBuffer]);
  } catch (e) {
    ctx.postMessage({ ok: false, error: (e as Error).message });
  }
};
