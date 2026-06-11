from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict

import psutil

from .config import ServerConfig
from .manager import _is_alive, _load_pids, _port_bound, start_model, stop_process

log = logging.getLogger(__name__)

_STARTUP_GRACE_SECONDS = 60.0


def _available_memory_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def _total_model_rss_gb(cfg: ServerConfig) -> float:
    pids = _load_pids()
    total = 0
    for model in cfg.models:
        pid = pids.get(model.name)
        if pid and _is_alive(pid):
            try:
                total += psutil.Process(pid).memory_info().rss
            except psutil.NoSuchProcess:
                pass
    return total / (1024 ** 3)


class MemoryManager:
    def __init__(self, cfg: ServerConfig) -> None:
        self._cfg = cfg
        self._lru: OrderedDict[str, float] = OrderedDict()
        self._started_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def ensure_model_ready(self, model_name: str) -> None:
        async with self._lock:
            self._touch(model_name)

            model_cfg = self._cfg.model_by_name(model_name)
            pids = _load_pids()
            pid = pids.get(model_name)
            already_running = bool(pid and _is_alive(pid))

            # How much additional RAM will loading this model consume?
            # Use the annotated memory_gb hint; fall back to 0 (conservative —
            # means we only evict based on current usage, not projected).
            projected_gb = 0.0 if already_running else (model_cfg.memory_gb or 0.0)

            # Evict BEFORE starting the model.  Previously this ran after
            # _ensure_running, which meant the new model was already mid-load
            # (and had claimed its RAM) before eviction fired.
            await self._enforce_memory_cap(model_name, projected_additional_gb=projected_gb)

            # Check free-memory floor BEFORE starting.  This is the last line of
            # defence against swap/OOM when eviction couldn't free enough room
            # (e.g. all other models are protected or within the grace window).
            if not already_running:
                loop = asyncio.get_running_loop()
                available = await loop.run_in_executor(None, _available_memory_gb)
                needed = projected_gb or 0.0
                if available - needed < self._cfg.min_free_memory_gb:
                    raise RuntimeError(
                        f"Not enough free memory to start '{model_name}': "
                        f"{available:.1f} GB available, need {needed:.1f} GB + "
                        f"{self._cfg.min_free_memory_gb:.1f} GB headroom"
                    )

            await self._ensure_running(model_name)

    def _touch(self, name: str) -> None:
        self._lru[name] = time.monotonic()
        self._lru.move_to_end(name)

    async def _ensure_running(self, name: str) -> None:
        pids = _load_pids()
        pid = pids.get(name)
        if pid and _is_alive(pid):
            return

        model_cfg = self._cfg.model_by_name(name)
        if model_cfg is None:
            return

        log.info("Starting model '%s' on demand", name)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, start_model, model_cfg)
        self._started_at[name] = time.monotonic()

        if not await self._async_wait_for_port(model_cfg.port, timeout=120.0):
            raise RuntimeError(f"Model '{name}' did not become ready within 120s")

    async def _enforce_memory_cap(
        self,
        requesting_model: str,
        projected_additional_gb: float = 0.0,
    ) -> None:
        no_cap = self._cfg.max_total_memory_gb is None
        no_floor = self._cfg.min_free_memory_gb <= 0
        if no_cap and no_floor:
            return

        loop = asyncio.get_running_loop()
        while True:
            total_gb = await loop.run_in_executor(None, _total_model_rss_gb, self._cfg)
            available_gb = await loop.run_in_executor(None, _available_memory_gb)

            over_cap = (
                self._cfg.max_total_memory_gb is not None
                and total_gb + projected_additional_gb > self._cfg.max_total_memory_gb
            )
            under_floor = (
                available_gb - projected_additional_gb < self._cfg.min_free_memory_gb
            )

            if not over_cap and not under_floor:
                break

            victim = self._pick_lru_victim(requesting_model)
            if victim is None:
                log.warning(
                    "Memory pressure (%.1f GB used + %.1f GB projected, %.1f GB free) "
                    "but no evictable model found",
                    total_gb,
                    projected_additional_gb,
                    available_gb,
                )
                break

            log.info(
                "Evicting '%s' (LRU) — used=%.1f GB projected=%.1f GB free=%.1f GB cap=%s",
                victim,
                total_gb,
                projected_additional_gb,
                available_gb,
                f"{self._cfg.max_total_memory_gb:.1f} GB" if self._cfg.max_total_memory_gb else "none",
            )
            await loop.run_in_executor(None, stop_process, victim)
            self._lru.pop(victim, None)
            self._started_at.pop(victim, None)

    def _pick_lru_victim(self, skip: str) -> str | None:
        pids = _load_pids()
        now = time.monotonic()
        for name in self._lru:  # leftmost = least recently used
            if name == skip:
                continue
            model_cfg = self._cfg.model_by_name(name)
            if model_cfg is None or model_cfg.protected:
                continue
            if now - self._started_at.get(name, 0) < _STARTUP_GRACE_SECONDS:
                continue
            pid = pids.get(name)
            if pid and _is_alive(pid):
                return name
        return None

    def memory_report(self) -> list[dict]:
        pids = _load_pids()
        rows = []
        total_bytes = 0
        for model in self._cfg.models:
            pid = pids.get(model.name)
            alive = bool(pid and _is_alive(pid))
            rss_gb = 0.0
            if alive and pid:
                try:
                    rss_bytes = psutil.Process(pid).memory_info().rss
                    rss_gb = rss_bytes / (1024 ** 3)
                    total_bytes += rss_bytes
                except psutil.NoSuchProcess:
                    alive = False
            rows.append({
                "name": model.name,
                "pid": pid,
                "alive": alive,
                "rss_gb": rss_gb,
                "protected": model.protected,
                "last_used": self._lru.get(model.name),
            })
        rows.append({
            "name": "__total__",
            "rss_gb": total_bytes / (1024 ** 3),
            "cap_gb": self._cfg.max_total_memory_gb,
            "free_gb": _available_memory_gb(),
        })
        return rows

    @staticmethod
    async def _async_wait_for_port(port: int, timeout: float = 120.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _port_bound(port):
                return True
            await asyncio.sleep(0.5)
        return False