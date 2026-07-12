"""
Keep the Mac awake (display + idle sleep, and thus screen lock) while the
router is actively proxying requests, plus a short grace period afterward.

macOS's screen lock is driven by the same idle timer as display sleep and
the screensaver (HIDIdleTime). `caffeinate -d -i` holds that timer off for
as long as the assertion process is alive; it does not override an explicit
lock (Cmd+Ctrl+Q) or lid-close sleep, only the idle-timeout-driven lock that
was killing long-running remote/agent sessions (fans don't spin up while
locked, and the M-series core overheats — see CLAUDE.md known bugs).

A background thread watches a `touch()`-updated timestamp and starts/stops
the caffeinate subprocess accordingly, so the Mac reverts to normal lock
behavior GRACE_SECONDS after the last request activity.
"""
from __future__ import annotations

import atexit
import logging
import subprocess
import threading
import time

logger = logging.getLogger("mlx_server.keepawake")

GRACE_SECONDS = 90.0
_POLL_INTERVAL = 5.0

_lock = threading.Lock()
_last_activity = 0.0
_proc: subprocess.Popen | None = None
_thread: threading.Thread | None = None


def touch() -> None:
    """Record activity. Call on every request and on every streamed chunk."""
    global _last_activity, _thread
    with _lock:
        _last_activity = time.monotonic()
        if _thread is None:
            _thread = threading.Thread(target=_watch_loop, daemon=True)
            _thread.start()


def _start_assertion() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        return
    try:
        _proc = subprocess.Popen(["caffeinate", "-d", "-i"])
        logger.info("keepawake: caffeinate started (pid %d)", _proc.pid)
    except FileNotFoundError:
        logger.warning("keepawake: `caffeinate` binary not found; skipping")


def _stop_assertion() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.kill()
        logger.info("keepawake: caffeinate stopped")
    _proc = None


def _watch_loop() -> None:
    while True:
        time.sleep(_POLL_INTERVAL)
        with _lock:
            idle_for = time.monotonic() - _last_activity
            if idle_for <= GRACE_SECONDS:
                _start_assertion()
            else:
                _stop_assertion()


atexit.register(_stop_assertion)
