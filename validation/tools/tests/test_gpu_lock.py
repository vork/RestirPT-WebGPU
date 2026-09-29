"""Shared GPU lock, Python side (validation/blender/gpu_lock.py; protocol in validation/harness/gpu-lock.ts).

Every test uses a temp lock path; the real /tmp/restirpt-gpu.lock is never touched.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

BLENDER_DIR = Path(__file__).resolve().parents[2] / "blender"
sys.path.insert(0, str(BLENDER_DIR))
import gpu_lock as gl  # noqa: E402


@pytest.fixture
def lock(tmp_path: Path) -> Path:
    return tmp_path / "test-gpu.lock"


@pytest.fixture
def logs() -> list[str]:
    return []


def kw(logs: list[str], **extra):
    return {"poll_s": 0.02, "log": logs.append, "signals": False, **extra}


def dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert not gl.pid_alive(p.pid)
    return p.pid


def stale_lock(lock: Path, *holders: str) -> None:
    lock.mkdir()
    for h in holders:
        (lock / h).write_text("")


def test_default_path_is_shared_and_not_used_here(lock: Path) -> None:
    assert gl.GPU_LOCK == Path("/tmp/restirpt-gpu.lock")
    assert lock != gl.GPU_LOCK


def test_acquire_release(lock: Path, logs: list[str]) -> None:
    with gl.gpu_lock("unit", lock, **kw(logs)) as waited:
        assert waited < 1
        assert sorted(os.listdir(lock)) == [f"unit-{os.getpid()}"]
        assert gl.lock_holders(lock) == [(f"unit-{os.getpid()}", os.getpid())]
    assert not lock.exists()
    rel = gl.acquire("unit", lock, **kw(logs))
    rel()
    rel()  # idempotent
    assert not lock.exists()
    with pytest.raises(RuntimeError, match="boom"):
        with gl.gpu_lock("unit", lock, **kw(logs)):
            raise RuntimeError("boom")
    assert not lock.exists()


def test_stale_reclaim(lock: Path, tmp_path: Path, logs: list[str]) -> None:
    a, b = dead_pid(), dead_pid()
    stale_lock(lock, f"render_reference-{a}", f"other-{b}")
    with gl.gpu_lock("after", lock, **kw(logs)):
        assert os.listdir(lock) == [f"after-{os.getpid()}"]
    assert any(m.startswith("reclaimed stale GPU lock from ") and f"render_reference-{a}" in m and f"other-{b}" in m for m in logs)
    assert os.listdir(tmp_path) == []  # no .stale-* directory left behind


def test_no_reclaim_when_holder_alive(lock: Path, logs: list[str]) -> None:
    stale_lock(lock, f"dead-{dead_pid()}", f"parent-{os.getppid()}")
    with pytest.raises(gl.GpuLockTimeoutError):
        gl.acquire("w", lock, **kw(logs, timeout_s=0.2))
    assert (lock / f"parent-{os.getppid()}").exists()
    assert gl.pid_alive(1)  # EPERM for a normal user = alive
    for f in lock.iterdir():
        f.unlink()
    (lock / "init-1").write_text("")
    with pytest.raises(gl.GpuLockTimeoutError):
        gl.acquire("w", lock, **kw(logs, timeout_s=0.2))
    assert os.listdir(lock) == ["init-1"]
    assert not any(m.startswith("reclaimed") for m in logs)


def test_no_reclaim_without_holder_file(lock: Path, logs: list[str]) -> None:
    stale_lock(lock, "README")
    old = time.time() - 25 * 60
    os.utime(lock, (old, old))
    with pytest.raises(gl.GpuLockTimeoutError):
        gl.acquire("w", lock, **kw(logs, timeout_s=0.2))
    assert os.listdir(lock) == ["README"]
    assert [m for m in logs if "with no holder file" in m] == [
        f"GPU lock held 25 min with no holder file; if no GPU job is running, remove {lock}"]


def test_nested_acquire_raises(lock: Path, logs: list[str]) -> None:
    with gl.gpu_lock("outer", lock, **kw(logs)):
        with pytest.raises(gl.GpuLockNestedError, match=r"already held by this process \(outer-\d+\).*would deadlock"):
            gl.acquire("inner", lock, **kw(logs))
        assert os.listdir(lock) == [f"outer-{os.getpid()}"]
    assert not lock.exists()
    stale_lock(lock, f"elsewhere-{os.getpid()}")
    with pytest.raises(gl.GpuLockNestedError):
        gl.acquire("again", lock, **kw(logs, timeout_s=1))


def test_release_leaves_a_reclaimed_lock_alone(lock: Path, logs: list[str]) -> None:
    rel = gl.acquire("victim", lock, **kw(logs))
    for f in lock.iterdir():
        f.unlink()
    (lock / f"newowner-{os.getppid()}").write_text("")
    rel()
    assert os.listdir(lock) == [f"newowner-{os.getppid()}"]


CHILD = textwrap.dedent("""
    import os, sys, time
    sys.path.insert(0, sys.argv[1])
    import gpu_lock as gl
    lock, log = sys.argv[2], sys.argv[3]
    def out(s):
        with open(log, "a") as f:
            f.write(s + "\\n")
    with gl.gpu_lock("child", lock, poll_s=0.01, signals=False, log=lambda m: out(f"log {os.getpid()} {m}")):
        out(f"start {os.getpid()} " + ",".join(n for n, _ in gl.lock_holders(lock) or []))
        time.sleep(0.06)
        out(f"end {os.getpid()}")
""")


@pytest.mark.parametrize("stale", [False, True])
def test_concurrent_waiters_one_winner(lock: Path, tmp_path: Path, stale: bool) -> None:
    dead = dead_pid()
    if stale:
        stale_lock(lock, f"render_reference-{dead}")
    log = tmp_path / "events.log"
    log.write_text("")
    n = 6
    kids = [subprocess.Popen([sys.executable, "-c", CHILD, str(BLENDER_DIR), str(lock), str(log)]) for _ in range(n)]
    assert [k.wait(timeout=60) for k in kids] == [0] * n
    text = log.read_text()
    ev = [ln.split(" ") for ln in text.splitlines() if ln.startswith(("start ", "end "))]
    assert len(ev) == 2 * n
    for i in range(0, len(ev), 2):
        assert ev[i][0] == "start" and ev[i + 1] == ["end", ev[i][1]]
        assert ev[i][2] == f"child-{ev[i][1]}"
    assert len({e[1] for e in ev}) == n
    assert not lock.exists()
    if stale:
        assert sum(f"reclaimed stale GPU lock from render_reference-{dead}" in ln for ln in text.splitlines()) == 1
    assert [f for f in os.listdir(tmp_path) if ".stale-" in f] == []
