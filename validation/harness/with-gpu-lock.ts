// Run any command under the shared GPU lock (protocol: gpu-lock.ts).
//   npx tsx validation/harness/with-gpu-lock.ts <tag> -- <cmd> [args...]
// Waits for the lock (holder file <tag>-<pid>), runs the command with inherited stdio, forwards SIGINT / SIGTERM /
// SIGHUP to it, releases the lock when it exits and exits with its code (128 + signo if it died from a signal).
// RESTIRPT_GPU_LOCK=<dir> overrides the lock path (tests only).
// Do not wrap commands that take the lock themselves (render_reference.py, run-batches.ts, the app smokes, ...): nested
// acquisition from a child process would wait forever.
import { spawn, type ChildProcess } from 'node:child_process';
import { constants as osConstants } from 'node:os';
import { acquireGpuLock, type GpuLockRelease } from './gpu-lock.ts';

const argv = process.argv.slice(2);
const sep = argv.indexOf('--');
const tag = argv[0];
const cmd = sep >= 0 ? argv.slice(sep + 1) : [];
if (!tag || sep !== 1 || cmd.length === 0) {
  console.error('usage: npx tsx validation/harness/with-gpu-lock.ts <tag> -- <cmd> [args...]');
  process.exit(2);
}

const SIGNALS = ['SIGINT', 'SIGTERM', 'SIGHUP'] as const;
const signo = (s: NodeJS.Signals) => osConstants.signals[s] ?? 1;
let child: ChildProcess | undefined;
let release: GpuLockRelease | undefined;
for (const sig of SIGNALS) {
  process.on(sig, () => {
    if (child && child.exitCode === null && child.signalCode === null) { child.kill(sig); return; } // exit follows the child
    release?.();
    process.exit(128 + signo(sig));
  });
}

release = await acquireGpuLock(tag, { signals: false, lockPath: process.env.RESTIRPT_GPU_LOCK || undefined });
child = spawn(cmd[0], cmd.slice(1), { stdio: 'inherit' });
child.on('error', (e) => { console.error(`[with-gpu-lock] ${cmd[0]}: ${e.message}`); release?.(); process.exit(127); });
child.on('exit', (code, sig) => {
  release?.();
  process.exit(code ?? (sig ? 128 + signo(sig) : 1));
});
