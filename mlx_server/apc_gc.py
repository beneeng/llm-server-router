"""
Time-based expiry for the APC disk-cache tier (see CLAUDE.md "KV caching").

mlx-vlm's own APC eviction (`APC_DISK_MAX_GB`) is purely size-triggered — it
has no notion of age. Cache entries are only a re-prefill time saver, so
anything unused for 48h is swept regardless of total cache size. Each cache
file is self-describing (mlx-vlm rebuilds its hash->path index by scanning
shards at startup); there's no on-disk manifest to desync, so deleting old
files externally is safe. mlx-vlm also calls os.utime() on a shard when it's
read, so mtime already means "last used" — a hot entry keeps resetting its
own clock instead of being swept out from under active use.

Mirrors keepawake.py's daemon-thread + plain-sleep-loop pattern rather than
adding new asyncio-task infra.
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("mlx_server.apc_gc")

APC_TTL_SECONDS = 48 * 3600
_SWEEP_INTERVAL_SECONDS = 3600  # hourly checks are plenty for a 48h TTL

_thread: threading.Thread | None = None
_lock = threading.Lock()


def sweep_once(ttl_seconds: float = APC_TTL_SECONDS, dry_run: bool = False) -> tuple[int, int]:
    """Delete APC shard files older (by mtime) than ttl_seconds.

    Returns (files_removed, bytes_freed).
    """
    from .manager import APC_DIR

    if not APC_DIR.exists():
        return 0, 0

    cutoff = time.time() - ttl_seconds
    removed = 0
    freed = 0
    for path in APC_DIR.rglob("*.safetensors"):
        try:
            st = path.stat()
        except FileNotFoundError:
            continue
        if st.st_mtime >= cutoff:
            continue
        if not dry_run:
            try:
                path.unlink()
            except FileNotFoundError:
                # Raced with mlx-vlm's own size-based eviction deleting the
                # same file concurrently — not an error.
                continue
        removed += 1
        freed += st.st_size

    if removed:
        logger.info(
            "APC GC: %s %d shard file(s), %.2f GB",
            "would remove" if dry_run else "removed",
            removed,
            freed / 1e9,
        )
    return removed, freed


def _loop() -> None:
    while True:
        time.sleep(_SWEEP_INTERVAL_SECONDS)
        try:
            sweep_once()
        except Exception:
            logger.exception("APC GC sweep failed")


def start_background_gc() -> None:
    """Lazily start the daemon sweep thread. Idempotent."""
    global _thread
    with _lock:
        if _thread is None:
            _thread = threading.Thread(target=_loop, daemon=True)
            _thread.start()
            logger.info("APC GC thread started (ttl=%.0fh, interval=%.0fm)", APC_TTL_SECONDS / 3600, _SWEEP_INTERVAL_SECONDS / 60)
