// Child process for tests/harness/gpu-lock.test.ts (not a test file):
//   node --import tsx tests/harness/gpu-lock-child.ts <lockPath> <logFile> <holdMs> <sync|async>
// Takes the lock, appends "start <pid> <holder files>" to the log, holds it for holdMs, appends "end <pid>", releases.
import { appendFileSync } from 'node:fs';
import { acquireGpuLock, acquireGpuLockSync, gpuLockHolders } from '../../validation/harness/gpu-lock.ts';

const [lockPath, logFile, holdMs, mode] = process.argv.slice(2);
const opts = { lockPath, pollMs: 10, log: (m: string) => appendFileSync(logFile, `log ${process.pid} ${m}\n`), signals: false };
const release = mode === 'async' ? await acquireGpuLock('child', opts) : acquireGpuLockSync('child', opts);
appendFileSync(logFile, `start ${process.pid} ${(gpuLockHolders(lockPath) ?? []).map((h) => h.name).join(',')}\n`);
Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, Number(holdMs));
appendFileSync(logFile, `end ${process.pid}\n`);
release();
