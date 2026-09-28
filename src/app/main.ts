// M0 placeholder app: create the GPU context and print adapter info + limits. Replaced by the renderer in M1.
import { createGpuContext, describeContext } from '../core/gpu/device.ts';

const status = document.getElementById('status')!;
const info = document.getElementById('info')!;

async function main(): Promise<void> {
  if (!navigator.gpu) throw new Error('navigator.gpu is missing: this browser does not expose WebGPU');
  const ctx = await createGpuContext(navigator.gpu, {
    label: 'app',
    onLost: (i) => { status.textContent = `Device lost (${i.reason}): ${i.message}`; status.className = 'err'; },
  });
  const d = describeContext(ctx);
  status.textContent = `WebGPU ready: ${ctx.adapterInfo.vendor} / ${ctx.adapterInfo.architecture}`;
  info.textContent = JSON.stringify({ userAgent: navigator.userAgent, ...d }, null, 2);
}

main().catch((e: unknown) => {
  status.textContent = `WebGPU init failed: ${e instanceof Error ? e.message : String(e)}`;
  status.className = 'err';
  console.error(e);
});
