// Shared GPU lock protocol (validation/harness/gpu-lock.ts). Every test uses its own temp lock path; the real
// /tmp/restirpt-gpu.lock is never touched.
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, utimesSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  acquireGpuLock, acquireGpuLockSync, GPU_LOCK, gpuLockHolders, GpuLockNestedError, GpuLockTimeoutError, pidAlive, withGpuLock,
  withGpuLockSync, type GpuLockOptions,
} from '../../validation/harness/gpu-lock.ts';

const CHILD = fileURLToPath(new URL('./gpu-lock-child.ts', import.meta.url));
const PY = fileURLToPath(new URL('../../validation/.venv/bin/python', import.meta.url));
const PY_LOCK_DIR = fileURLToPath(new URL('../../validation/blender/', import.meta.url));
// Python twin of gpu-lock-child.ts (validation/blender/gpu_lock.py), for the mixed-language race.
const PY_CHILD = `
import os, sys, time
sys.path.insert(0, sys.argv[1])
import gpu_lock as gl
lock, log = sys.argv[2], sys.argv[3]
def out(s):
    with open(log, "a") as f:
        f.write(s + "\\n")
with gl.gpu_lock("pychild", lock, poll_s=0.01, signals=False, log=lambda m: out(f"log {os.getpid()} {m}")):
    out(f"start {os.getpid()} " + ",".join(n for n, _ in gl.lock_holders(lock) or []))
    time.sleep(0.06)
    out(f"end {os.getpid()}")
`;

let dir: string;
let lock: string;
let logs: string[];
const opts = (o: GpuLockOptions = {}): GpuLockOptions => ({ lockPath: lock, pollMs: 20, log: (m) => logs.push(m), signals: false, ...o });

/** A pid that certainly belonged to a process which has exited (and was reaped). */
function deadPid(): number {
  const r = spawnSync(process.execPath, ['-e', '']);
  expect(r.pid).toBeGreaterThan(0);
  expect(pidAlive(r.pid!)).toBe(false);
  return r.pid!;
}

function staleLock(...holders: string[]): void {
  mkdirSync(lock);
  for (const h of holders) writeFileSync(path.join(lock, h), '');
}

beforeEach(() => {
  dir = mkdtempSync(path.join(os.tmpdir(), 'gpu-lock-test-'));
  lock = path.join(dir, 'test-gpu.lock');
  logs = [];
});
afterEach(() => { rmSync(dir, { recursive: true, force: true }); });

describe('gpu-lock', () => {
  it('uses the shared path by default and never the real lock in these tests', () => {
    expect(GPU_LOCK).toBe('/tmp/restirpt-gpu.lock');
    expect(lock).not.toBe(GPU_LOCK);
  });

  it('acquires with a holder file and releases (idempotent)', () => {
    const release = acquireGpuLockSync('unit', opts());
    const holder = `unit-${process.pid}`;
    expect(release.holder).toBe(holder);
    expect(readdirSync(lock)).toEqual([holder]);
    expect(JSON.parse(readFileSync(path.join(lock, holder), 'utf8'))).toMatchObject({ tag: 'unit', pid: process.pid });
    expect(gpuLockHolders(lock)).toEqual([{ name: holder, pid: process.pid }]);
    release();
    expect(existsSync(lock)).toBe(false);
    release();
    expect(existsSync(lock)).toBe(false);
    expect(gpuLockHolders(lock)).toBeNull();
  });

  it('with* helpers release after the body, also when it throws', async () => {
    expect(withGpuLockSync('sync', () => { expect(existsSync(path.join(lock, `sync-${process.pid}`))).toBe(true); return 7; }, opts())).toBe(7);
    expect(existsSync(lock)).toBe(false);
    expect(() => withGpuLockSync('sync', () => { throw new Error('boom'); }, opts())).toThrow('boom');
    expect(existsSync(lock)).toBe(false);
    expect(await withGpuLock('async', async () => 'x', opts())).toBe('x');
    await expect(withGpuLock('async', async () => { throw new Error('bang'); }, opts())).rejects.toThrow('bang');
    expect(existsSync(lock)).toBe(false);
  });

  it('reclaims a stale lock whose holder files all name dead pids', async () => {
    const a = deadPid(), b = deadPid();
    staleLock(`render_reference-${a}`, `other-${b}`);
    const release = acquireGpuLockSync('after', opts());
    expect(logs.some((m) => m.startsWith('reclaimed stale GPU lock from ') && m.includes(`render_reference-${a}`) && m.includes(`other-${b}`))).toBe(true);
    expect(readdirSync(lock)).toEqual([`after-${process.pid}`]);
    release();
    expect(readdirSync(dir)).toEqual([]); // no .stale-* directory left behind

    staleLock(`render_reference-${deadPid()}`);
    const r2 = await acquireGpuLock('async-after', opts());
    expect(readdirSync(lock)).toEqual([`async-after-${process.pid}`]);
    r2();
  });

  it('does not reclaim while any holder is alive (including EPERM pids)', () => {
    staleLock(`dead-${deadPid()}`, `parent-${process.ppid}`);
    expect(() => acquireGpuLockSync('w', opts({ timeoutMs: 200 }))).toThrow(GpuLockTimeoutError);
    expect(existsSync(path.join(lock, `parent-${process.ppid}`))).toBe(true);
    rmSync(lock, { recursive: true });

    expect(pidAlive(1)).toBe(true); // launchd / init: kill(1, 0) fails with EPERM for a normal user = alive
    staleLock('init-1');
    expect(() => acquireGpuLockSync('w', opts({ timeoutMs: 200 }))).toThrow(GpuLockTimeoutError);
    expect(readdirSync(lock)).toEqual(['init-1']);
    expect(logs.some((m) => m.startsWith('reclaimed'))).toBe(false);
  });

  it('never reclaims a lock without holder files, and says how to clear it', async () => {
    staleLock('README'); // not a <tag>-<pid> name: no holder
    const old = (Date.now() - 25 * 60_000) / 1000;
    utimesSync(lock, old, old);
    await expect(acquireGpuLock('w', opts({ timeoutMs: 200 }))).rejects.toThrow(GpuLockTimeoutError);
    expect(readdirSync(lock)).toEqual(['README']);
    const warn = logs.filter((m) => m.includes('with no holder file'));
    expect(warn).toEqual([`GPU lock held 25 min with no holder file; if no GPU job is running, remove ${lock}`]); // once per 10 min
    rmSync(lock, { recursive: true });

    mkdirSync(lock); // fresh and empty (e.g. old code, or between another job's mkdir and holder file): wait quietly
    expect(() => acquireGpuLockSync('w', opts({ timeoutMs: 150 }))).toThrow(GpuLockTimeoutError);
    expect(existsSync(lock)).toBe(true);
  });

  it('throws a clear error on nested acquisition by the same process', async () => {
    const release = acquireGpuLockSync('outer', opts());
    expect(() => acquireGpuLockSync('inner', opts())).toThrow(GpuLockNestedError);
    expect(() => acquireGpuLockSync('inner', opts())).toThrow(/already held by this process \(outer-\d+\).*would deadlock/);
    await expect(acquireGpuLock('inner', opts())).rejects.toThrow(GpuLockNestedError);
    expect(readdirSync(lock)).toEqual([`outer-${process.pid}`]); // the outer hold is untouched
    release();
    expect(existsSync(lock)).toBe(false);

    staleLock(`elsewhere-${process.pid}`); // a holder file naming our pid (e.g. another module instance)
    expect(() => acquireGpuLockSync('again', opts({ timeoutMs: 1000 }))).toThrow(GpuLockNestedError);
  });

  it('does not remove a lock that was reclaimed from under its holder', () => {
    const release = acquireGpuLockSync('victim', opts());
    rmSync(lock, { recursive: true });
    staleLock(`newowner-${process.ppid}`);
    release();
    expect(readdirSync(lock)).toEqual([`newowner-${process.ppid}`]);
    expect(logs.some((m) => m.includes('vanished'))).toBe(true);
  });

  const race = async (n: number, pre?: () => void, withPython = false) => {
    pre?.();
    const log = path.join(dir, 'events.log');
    writeFileSync(log, '');
    const kids = Array.from({ length: n }, (_, i) => (withPython && i % 3 === 2
      ? spawn(PY, ['-c', PY_CHILD, PY_LOCK_DIR, lock, log], { stdio: 'inherit' })
      : spawn(process.execPath, ['--import', 'tsx', CHILD, lock, log, '60', i % 2 ? 'async' : 'sync'], { stdio: 'inherit' })));
    const codes = await Promise.all(kids.map((k) => new Promise<number | null>((r) => k.on('exit', r))));
    expect(codes).toEqual(Array(n).fill(0));
    const ev = readFileSync(log, 'utf8').split('\n').filter((l) => /^(start|end) /.test(l)).map((l) => l.split(' '));
    expect(ev.length).toBe(2 * n);
    for (let i = 0; i < ev.length; i += 2) { // strictly start/end of the same pid: never two holders at once
      expect(ev[i][0]).toBe('start');
      expect(ev[i + 1]).toEqual(['end', ev[i][1]]);
      expect(ev[i][2]).toMatch(new RegExp(`^(child|pychild)-${ev[i][1]}$`)); // exactly one holder file while held
    }
    expect(new Set(ev.map((e) => e[1])).size).toBe(n);
    expect(existsSync(lock)).toBe(false);
    return readFileSync(log, 'utf8');
  };

  it('serializes concurrent waiters in separate processes (one winner at a time)', async () => {
    await race(6);
  }, 30_000);

  it.skipIf(!existsSync(PY))('interoperates with the Python implementation (mixed race over a stale lock)', async () => {
    const dead = deadPid();
    const out = await race(9, () => staleLock(`render_reference-${dead}`), true);
    expect(out).toMatch(/^start \d+ pychild-\d+$/m);
    expect(out.split('\n').filter((l) => l.includes(`reclaimed stale GPU lock from render_reference-${dead}`)).length).toBe(1);
  }, 60_000);

  it('reclaims a lock whose reclaimer died mid-claim', () => {
    staleLock(`render_reference-${deadPid()}.reclaim-${deadPid()}`);
    acquireGpuLockSync('after', opts())();
    expect(readdirSync(dir)).toEqual([]);
  });

  it('lets exactly one of several concurrent waiters reclaim a stale lock', async () => {
    const dead = deadPid();
    const out = await race(6, () => staleLock(`render_reference-${dead}`));
    expect(out.split('\n').filter((l) => l.includes(`reclaimed stale GPU lock from render_reference-${dead}`)).length).toBe(1);
    expect(readdirSync(dir).filter((f) => f.includes('.stale-'))).toEqual([]);
  }, 30_000);
});

describe('with-gpu-lock CLI', () => {
  const CLI = fileURLToPath(new URL('../../validation/harness/with-gpu-lock.ts', import.meta.url));
  const cli = (args: string[]) => spawnSync(process.execPath, ['--import', 'tsx', CLI, ...args],
    { env: { ...process.env, RESTIRPT_GPU_LOCK: lock }, encoding: 'utf8' });

  it('runs the command under the lock and forwards its exit code', () => {
    const r = cli(['cli', '--', process.execPath, '-e', `console.log(require('fs').readdirSync(${JSON.stringify(lock)}).join(',')); process.exit(3)`]);
    expect(r.status).toBe(3);
    expect(r.stdout.trim()).toMatch(/^cli-\d+$/);
    expect(existsSync(lock)).toBe(false);
  }, 30_000);

  it('reports a child killed by a signal as 128 + signo and releases', () => {
    const r = cli(['cli', '--', process.execPath, '-e', "process.kill(process.pid, 'SIGTERM')"]);
    expect(r.status).toBe(128 + os.constants.signals.SIGTERM);
    expect(existsSync(lock)).toBe(false);
  }, 30_000);

  it('rejects a missing -- separator', () => {
    expect(cli(['cli', 'true']).status).toBe(2);
  }, 30_000);
});
