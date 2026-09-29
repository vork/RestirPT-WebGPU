// Shared GPU lock (plan §1.8, §7.5 Orchestration: GPU-heavy jobs never overlap). The one TypeScript implementation;
// validation/blender/gpu_lock.py is the Python twin (same protocol, stdlib only, runs inside Blender's Python).
//
// Protocol (lock = /tmp/restirpt-gpu.lock, a directory):
//   Acquire   mkdir(lock) (atomic). On success create the holder file lock/<tag>-<pid> at once (tag = short job name,
//             content = a JSON line for humans). Release deletes the holder file, then rmdir's the lock; release runs
//             in finally blocks, on process exit and on SIGINT / SIGTERM. If the holder file is already gone at release
//             (someone reclaimed the lock from under us) the lock directory is left alone: it belongs to someone else.
//   Wait      On EEXIST poll every 5 s. Each poll reads the holder files and takes the pid from the trailing
//             `-<digits>` of each name. Liveness: kill(pid, 0); ESRCH = dead, EPERM (or success) = alive.
//   Reclaim   If there is at least one holder file and EVERY holder pid is dead (a job killed by SIGKILL, a tool timeout
//             or a Blender abort that skipped atexit), the lock is stale:
//             1. claim it: rename(lock/<first dead holder>, lock/<first dead holder>.reclaim-<mypid>). Atomic, so one
//                waiter wins; the others now see a live holder (the token's trailing pid) and keep waiting. Without
//                this step a waiter that read the dead holders just before another waiter reclaimed could rename a
//                FRESH lock (mkdir done, holder file not yet written) away, and two jobs would run at once.
//             2. rename(lock, lock.stale-<mypid>-<ms>) and re-check inside the renamed directory: our token present
//                and every other holder dead → remove it, log "reclaimed stale GPU lock from <holders>" and retry
//                mkdir at once. Otherwise (a race with code outside this protocol) restore the token name, rename the
//                directory back if lock does not exist, and keep waiting.
//   No holder A lock without holder files (old code, e.g. the ../WebGPURestirPT-m3c worktree, or the instant between
//             another job's mkdir and its holder file) is NEVER reclaimed automatically. Every 10 min a waiter logs
//             "GPU lock held N min with no holder file; if no GPU job is running, remove /tmp/restirpt-gpu.lock".
//   Nesting   Acquiring a lock this process already holds (in-process registry, or a holder file naming our pid)
//             throws GpuLockNestedError instead of deadlocking.
//
// API: acquireGpuLockSync / withGpuLockSync (block with Atomics.wait; for spawnSync-driven gates), acquireGpuLock /
// withGpuLock (async; Playwright drivers). CLI: validation/harness/with-gpu-lock.ts <tag> -- <cmd> [args...].
import { existsSync, mkdirSync, readdirSync, renameSync, rmdirSync, rmSync, statSync, unlinkSync, writeFileSync } from 'node:fs';
import { constants as osConstants } from 'node:os';
import path from 'node:path';

export const GPU_LOCK = '/tmp/restirpt-gpu.lock';

export interface GpuLockOptions {
  /** Lock directory (tests inject a temp path; default GPU_LOCK). */
  lockPath?: string;
  /** Poll interval while waiting (default 5000 ms). */
  pollMs?: number;
  /** Interval of the "held N min" messages (default 10 min). */
  warnMs?: number;
  /** Give up with GpuLockTimeoutError after this long (default: wait forever). */
  timeoutMs?: number;
  /** Logger (default console.log with a "[gpu-lock]" prefix). */
  log?: (msg: string) => void;
  /** Release on SIGINT / SIGTERM and exit 128+signo (default true; the CLI forwards signals itself). */
  signals?: boolean;
}

export interface GpuLockHolder { name: string; pid: number }

/** Idempotent release function; `waitedMs` is the time spent waiting for the lock. */
export type GpuLockRelease = (() => void) & { readonly waitedMs: number; readonly holder: string };

export class GpuLockNestedError extends Error { override name = 'GpuLockNestedError'; }
export class GpuLockTimeoutError extends Error { override name = 'GpuLockTimeoutError'; }

interface Resolved { lockPath: string; pollMs: number; warnMs: number; timeoutMs: number; log: (msg: string) => void; signals: boolean }
const resolve = (o: GpuLockOptions = {}): Resolved => ({
  lockPath: o.lockPath ?? GPU_LOCK,
  pollMs: o.pollMs ?? 5000,
  warnMs: o.warnMs ?? 600_000,
  timeoutMs: o.timeoutMs ?? Infinity,
  log: o.log ?? ((m) => console.log(`[gpu-lock] ${m}`)),
  signals: o.signals ?? true,
});

const errCode = (e: unknown) => (e as NodeJS.ErrnoException | undefined)?.code;

/** kill(pid, 0): ESRCH → dead; success or EPERM (another user's process) → alive. */
export function pidAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true; } catch (e) { return errCode(e) !== 'ESRCH'; }
}

/** Holder files of a lock directory (entries named <tag>-<pid>); null when the lock does not exist. */
export function gpuLockHolders(lockPath: string = GPU_LOCK): GpuLockHolder[] | null {
  let names: string[];
  try { names = readdirSync(lockPath); } catch (e) { if (errCode(e) === 'ENOENT') return null; throw e; }
  const out: GpuLockHolder[] = [];
  for (const name of names) {
    const m = /-(\d+)$/.exec(name);
    const pid = m ? Number(m[1]) : 0;
    if (pid > 0) out.push({ name, pid });
  }
  return out;
}

const describe = (h: GpuLockHolder[]) => (h.length ? h.map((x) => x.name).join(', ') : 'no holder file');

// ---- in-process registry + exit / signal release ---------------------------------------------------------------------

interface Held { holder: string; signals: boolean; release: () => void }
const held = new Map<string, Held>();
const releaseAll = () => { for (const h of [...held.values()]) h.release(); };
const onSignal = (sig: 'SIGINT' | 'SIGTERM') => { releaseAll(); process.exit(128 + (osConstants.signals[sig] ?? 2)); };
const sigHandlers = { SIGINT: () => onSignal('SIGINT'), SIGTERM: () => onSignal('SIGTERM') } as const;
let exitHooked = false, sigHooked = false;
function syncHandlers(): void {
  if (!exitHooked) { process.on('exit', releaseAll); exitHooked = true; }
  const want = [...held.values()].some((h) => h.signals);
  if (want === sigHooked) return;
  for (const [sig, fn] of Object.entries(sigHandlers)) {
    if (want) process.on(sig, fn); else process.removeListener(sig, fn);
  }
  sigHooked = want;
}

// ---- one acquisition attempt, shared by the sync and async waiters ---------------------------------------------------

class Attempt {
  readonly t0 = Date.now();
  private lastDesc: string | undefined;
  private lastWarn = -Infinity;
  readonly tag: string;
  readonly o: Resolved;

  constructor(tag: string, o: Resolved) {
    const t = tag.replace(/[^A-Za-z0-9_.-]/g, '_').replace(/-+$/, '');
    if (!t) throw new Error(`bad GPU lock tag ${JSON.stringify(tag)}`);
    this.tag = t;
    this.o = o;
  }

  /** The release function once acquired, else the ms to wait before the next attempt (0 = retry at once). */
  step(): GpuLockRelease | number {
    const { lockPath } = this.o;
    const mine = held.get(lockPath);
    if (mine) throw this.nested(mine.holder);
    try {
      mkdirSync(lockPath);
    } catch (e) {
      if (errCode(e) !== 'EEXIST') throw e;
      return this.contended();
    }
    return this.take();
  }

  /** Throws GpuLockTimeoutError when the next wait would exceed timeoutMs. */
  checkTimeout(waitMs: number): void {
    if (Date.now() - this.t0 + waitMs > this.o.timeoutMs) {
      throw new GpuLockTimeoutError(`timed out after ${Date.now() - this.t0} ms waiting for the GPU lock ${this.o.lockPath} (${this.lastDesc ?? 'held'})`);
    }
  }

  private nested(holder: string): GpuLockNestedError {
    return new GpuLockNestedError(`GPU lock ${this.o.lockPath} is already held by this process (${holder}); acquiring it again as ` +
      `"${this.tag}" would deadlock. Release it first, or run the inner step outside the lock.`);
  }

  private take(): GpuLockRelease {
    const { lockPath, log } = this.o;
    const holder = `${this.tag}-${process.pid}`;
    let holderPath: string | undefined = path.join(lockPath, holder);
    try {
      writeFileSync(holderPath, `${JSON.stringify({ tag: this.tag, pid: process.pid, started: new Date().toISOString(), cwd: process.cwd() })}\n`, { flag: 'wx' });
    } catch (e) {
      log(`warning: could not create the holder file ${holderPath} (${errCode(e) ?? e}); holding the lock without one`);
      holderPath = undefined;
    }
    let released = false;
    const release = () => {
      if (released) return;
      released = true;
      held.delete(lockPath);
      syncHandlers();
      if (holderPath) {
        try { unlinkSync(holderPath); } catch (e) {
          // Our holder file is gone: the lock was reclaimed from under us and may now belong to another job.
          if (errCode(e) === 'ENOENT') { log(`warning: holder file ${holderPath} vanished; leaving ${lockPath} alone`); return; }
        }
      }
      try { rmdirSync(lockPath); } catch { /* gone */ }
    };
    held.set(lockPath, { holder, signals: this.o.signals, release });
    syncHandlers();
    const waitedMs = Date.now() - this.t0;
    if (waitedMs > 1000) log(`acquired the GPU lock as ${holder} after ${(waitedMs / 1000).toFixed(0)} s`);
    return Object.assign(release, { waitedMs, holder });
  }

  private contended(): number {
    const { lockPath, pollMs, warnMs, log } = this.o;
    const holders = gpuLockHolders(lockPath);
    if (holders === null) return 0; // released in the meantime
    const self = holders.find((h) => h.pid === process.pid);
    if (self) throw this.nested(self.name);
    if (holders.length > 0 && holders.every((h) => !pidAlive(h.pid))) {
      if (reclaimStale(lockPath, holders, log)) return 0;
    }
    const desc = describe(holders);
    if (desc !== this.lastDesc) { log(`waiting for the GPU lock ${lockPath} (held by ${desc}) ...`); this.lastDesc = desc; }
    const now = Date.now();
    let since = this.t0;
    try { since = Math.min(since, statSync(lockPath).mtimeMs); } catch { /* gone */ }
    const heldMs = now - since;
    if (heldMs >= warnMs && now - this.lastWarn >= warnMs) {
      this.lastWarn = now;
      const min = Math.round(heldMs / 60_000);
      log(holders.length === 0
        ? `GPU lock held ${min} min with no holder file; if no GPU job is running, remove ${lockPath}`
        : `GPU lock held ${min} min by ${desc} (alive); still waiting`);
    }
    return pollMs;
  }
}

/** Move a lock whose holders (all dead) were just read out of the way. True when the caller should retry mkdir now. */
function reclaimStale(lockPath: string, dead: GpuLockHolder[], log: (m: string) => void): boolean {
  // Claim first: rename the (sorted-first) dead holder file to a token naming our pid. Only one waiter can win this, the
  // token reads as a live holder to everyone else, and only the stale lock contains that file, so the directory we
  // rename next is guaranteed to be the stale lock and never a fresh one taken between our readdir and the rename.
  const first = [...dead].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))[0];
  const token = `${first.name}.reclaim-${process.pid}`;
  try { renameSync(path.join(lockPath, first.name), path.join(lockPath, token)); } catch (e) {
    return errCode(e) === 'ENOENT'; // lost the claim or the lock is gone: look again now; anything else: wait
  }
  const stale = `${lockPath}.stale-${process.pid}-${Date.now()}`;
  try { renameSync(lockPath, stale); } catch (e) {
    try { renameSync(path.join(lockPath, token), path.join(lockPath, first.name)); } catch { /* gone */ }
    return errCode(e) === 'ENOENT';
  }
  const inside = gpuLockHolders(stale) ?? [];
  const ours = inside.some((h) => h.name === token);
  const others = inside.filter((h) => h.name !== token);
  if (ours && others.every((h) => !pidAlive(h.pid))) {
    rmSync(stale, { recursive: true, force: true });
    log(`reclaimed stale GPU lock from ${describe([first, ...others])}`);
    return true;
  }
  // Not the lock we claimed, or it gained a live holder (old code interfering): restore it and keep waiting.
  if (ours) { try { renameSync(path.join(stale, token), path.join(stale, first.name)); } catch { /* keep going */ } }
  if (!existsSync(lockPath)) {
    try { renameSync(stale, lockPath); return false; } catch { /* fall through */ }
  }
  log(`warning: moved a live GPU lock (${describe(others)}) to ${stale} and could not put it back; ${lockPath} was re-created meanwhile`);
  return false;
}

// ---- public API ------------------------------------------------------------------------------------------------------

const napCell = new Int32Array(new SharedArrayBuffer(4));

/** Block (Atomics.wait) until the lock is ours; returns the release function. */
export function acquireGpuLockSync(tag: string, opts?: GpuLockOptions): GpuLockRelease {
  const a = new Attempt(tag, resolve(opts));
  for (;;) {
    const r = a.step();
    if (typeof r === 'function') return r;
    if (r > 0) { a.checkTimeout(r); Atomics.wait(napCell, 0, 0, r); }
  }
}

/** Wait (timers) until the lock is ours; returns the release function. */
export async function acquireGpuLock(tag: string, opts?: GpuLockOptions): Promise<GpuLockRelease> {
  const a = new Attempt(tag, resolve(opts));
  for (;;) {
    const r = a.step();
    if (typeof r === 'function') return r;
    if (r > 0) { a.checkTimeout(r); await new Promise((res) => setTimeout(res, r)); }
  }
}

export function withGpuLockSync<T>(tag: string, fn: () => T, opts?: GpuLockOptions): T {
  const release = acquireGpuLockSync(tag, opts);
  try { return fn(); } finally { release(); }
}

export async function withGpuLock<T>(tag: string, fn: () => Promise<T>, opts?: GpuLockOptions): Promise<T> {
  const release = await acquireGpuLock(tag, opts);
  try { return await fn(); } finally { release(); }
}
