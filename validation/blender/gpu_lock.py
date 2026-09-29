"""Shared GPU lock, Python twin of validation/harness/gpu-lock.ts (stdlib only: runs inside Blender's Python).

The protocol is documented once, in the header of validation/harness/gpu-lock.ts. In short: mkdir(lock) + holder file
lock/<tag>-<pid>; release removes the holder file, then rmdir's the lock (finally, atexit, SIGINT / SIGTERM); waiters poll
every 5 s and reclaim the lock (claim a dead holder file by renaming it to <name>.reclaim-<mypid>, then atomic rename
of the lock, re-check, remove) only when it has holder files and every holder pid is dead; a lock without holder files
is never reclaimed (a message every 10 min says how to clear it by hand); acquiring a lock this process already holds
raises GpuLockNestedError.

    with gpu_lock("render_reference") as waited_s:
        ...
"""
from __future__ import annotations

import atexit
import contextlib
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

GPU_LOCK = Path("/tmp/restirpt-gpu.lock")

_HOLDER_RE = re.compile(r"-(\d+)$")


class GpuLockNestedError(RuntimeError):
    pass


class GpuLockTimeoutError(TimeoutError):
    pass


def _default_log(msg: str) -> None:
    print(f"[gpu-lock] {msg}", flush=True)


def pid_alive(pid: int) -> bool:
    """kill(pid, 0): ESRCH -> dead; success or EPERM (another user's process) -> alive."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def lock_holders(lock_path: Path = GPU_LOCK) -> list[tuple[str, int]] | None:
    """Holder files (<tag>-<pid>) as (name, pid); None when the lock does not exist."""
    try:
        names = os.listdir(lock_path)
    except FileNotFoundError:
        return None
    out = []
    for name in sorted(names):
        m = _HOLDER_RE.search(name)
        if m and int(m.group(1)) > 0:
            out.append((name, int(m.group(1))))
    return out


def _describe(holders: list[tuple[str, int]]) -> str:
    return ", ".join(n for n, _ in holders) if holders else "no holder file"


# --- in-process registry + atexit / signal release ------------------------------------------------------------------

_held: dict[str, "Release"] = {}
_prev_handlers: dict[int, object] = {}


def _release_all() -> None:
    for r in list(_held.values()):
        r()


def _on_signal(signum: int, _frame: object) -> None:
    _release_all()
    sys.exit(128 + signum)


def _sync_handlers() -> None:
    want = any(r.signals for r in _held.values())
    if threading.current_thread() is not threading.main_thread():
        return  # signal.signal only works on the main thread; atexit + finally still release
    if want and not _prev_handlers:
        for s in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(ValueError, OSError):
                _prev_handlers[s] = signal.signal(s, _on_signal)
    elif not want and _prev_handlers:
        for s, h in _prev_handlers.items():
            with contextlib.suppress(ValueError, OSError, TypeError):
                signal.signal(s, h)  # type: ignore[arg-type]
        _prev_handlers.clear()


atexit.register(_release_all)


class Release:
    """Idempotent release of a held lock; `waited` = seconds spent waiting for it."""

    def __init__(self, lock_path: Path, holder: str, holder_path: Path | None, waited: float, signals: bool,
                 log: Callable[[str], None]) -> None:
        self.lock_path, self.holder, self.holder_path = lock_path, holder, holder_path
        self.waited, self.signals, self._log, self._done = waited, signals, log, False

    def __call__(self) -> None:
        if self._done:
            return
        self._done = True
        _held.pop(str(self.lock_path), None)
        _sync_handlers()
        if self.holder_path is not None:
            try:
                self.holder_path.unlink()
            except FileNotFoundError:
                # Our holder file is gone: the lock was reclaimed from under us and may belong to another job now.
                self._log(f"warning: holder file {self.holder_path} vanished; leaving {self.lock_path} alone")
                return
            except OSError:
                pass
        with contextlib.suppress(OSError):
            self.lock_path.rmdir()


def _reclaim_stale(lock_path: Path, dead: list[tuple[str, int]], log: Callable[[str], None]) -> bool:
    """Move a lock whose holders (all dead) were just read out of the way. True: retry mkdir at once.

    Claim first (rename the sorted-first dead holder file to <name>.reclaim-<mypid>), so the directory renamed next is
    guaranteed to be the stale lock, never a fresh one; see the protocol header in validation/harness/gpu-lock.ts.
    """
    first = sorted(dead)[0][0]
    token = f"{first}.reclaim-{os.getpid()}"
    try:
        os.rename(lock_path / first, lock_path / token)
    except FileNotFoundError:
        return True  # lost the claim, or the lock is gone: look again now
    except OSError:
        return False
    stale = Path(f"{lock_path}.stale-{os.getpid()}-{int(time.time() * 1000)}")
    try:
        os.rename(lock_path, stale)
    except OSError as e:
        with contextlib.suppress(OSError):
            os.rename(lock_path / token, lock_path / first)
        return isinstance(e, FileNotFoundError)
    inside = lock_holders(stale) or []
    ours = any(n == token for n, _ in inside)
    others = [(n, p) for n, p in inside if n != token]
    if ours and not any(pid_alive(p) for _, p in others):
        shutil.rmtree(stale, ignore_errors=True)
        log(f"reclaimed stale GPU lock from {_describe([(first, 0), *others])}")
        return True
    # Not the lock we claimed, or it gained a live holder (old code interfering): restore it and keep waiting.
    if ours:
        with contextlib.suppress(OSError):
            os.rename(stale / token, stale / first)
    if not lock_path.exists():
        try:
            os.rename(stale, lock_path)
            return False
        except OSError:
            pass
    log(f"warning: moved a live GPU lock ({_describe(others)}) to {stale} and could not put it back; "
        f"{lock_path} was re-created meanwhile")
    return False


def acquire(tag: str, lock_path: Path | str = GPU_LOCK, poll_s: float = 5.0, warn_s: float = 600.0,
            timeout_s: float | None = None, log: Callable[[str], None] = _default_log, signals: bool = True) -> Release:
    """Block until the lock is ours; returns the release callable."""
    lock_path = Path(lock_path)
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", tag).rstrip("-")
    if not tag:
        raise ValueError("empty GPU lock tag")
    t0 = time.monotonic()
    wall0 = time.time()
    last_desc: str | None = None
    last_warn = float("-inf")

    def nested(holder: str) -> GpuLockNestedError:
        return GpuLockNestedError(
            f"GPU lock {lock_path} is already held by this process ({holder}); acquiring it again as {tag!r} would "
            "deadlock. Release it first, or run the inner step outside the lock.")

    while True:
        mine = _held.get(str(lock_path))
        if mine is not None:
            raise nested(mine.holder)
        try:
            os.mkdir(lock_path)
        except FileExistsError:
            pass
        else:
            break
        holders = lock_holders(lock_path)
        if holders is None:
            continue  # released in the meantime
        for name, pid in holders:
            if pid == os.getpid():
                raise nested(name)
        if holders and not any(pid_alive(p) for _, p in holders):
            if _reclaim_stale(lock_path, holders, log):
                continue
        desc = _describe(holders)
        if desc != last_desc:
            log(f"waiting for the GPU lock {lock_path} (held by {desc}) ...")
            last_desc = desc
        now = time.time()
        since = wall0
        with contextlib.suppress(OSError):
            since = min(since, lock_path.stat().st_mtime)
        held_s = now - since
        if held_s >= warn_s and now - last_warn >= warn_s:
            last_warn = now
            minutes = round(held_s / 60)
            log(f"GPU lock held {minutes} min with no holder file; if no GPU job is running, remove {lock_path}"
                if not holders else f"GPU lock held {minutes} min by {desc} (alive); still waiting")
        if timeout_s is not None and time.monotonic() - t0 + poll_s > timeout_s:
            raise GpuLockTimeoutError(f"timed out after {time.monotonic() - t0:.1f} s waiting for the GPU lock {lock_path} ({desc})")
        time.sleep(poll_s)

    holder = f"{tag}-{os.getpid()}"
    holder_path: Path | None = lock_path / holder
    try:
        fd = os.open(holder_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"tag": tag, "pid": os.getpid(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                "cwd": os.getcwd()}) + "\n")
    except OSError as e:
        log(f"warning: could not create the holder file {holder_path} ({e}); holding the lock without one")
        holder_path = None
    waited = time.monotonic() - t0
    rel = Release(lock_path, holder, holder_path, waited, signals, log)
    _held[str(lock_path)] = rel
    _sync_handlers()
    return rel


@contextlib.contextmanager
def gpu_lock(tag: str, lock_path: Path | str = GPU_LOCK, **kw) -> Iterator[float]:
    """Hold the lock for the body; yields the seconds waited."""
    rel = acquire(tag, lock_path, **kw)
    try:
        yield rel.waited
    finally:
        rel()
