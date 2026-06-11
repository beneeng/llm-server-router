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


class MemoryManager:
    def __init__(self, cfg: ServerConfig) -> None:
        self._cfg = cfg
        self._lru: OrderedDict[str, float] = OrderedDict()
        self._started_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def ensure_model_ready(self, model_name: str) -> None:
        async with self._lock:
            self._touch(model_name)
            await self._ensure_running(model_name)
            await self._enforce_memory_cap(model_name)

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

    async def _enforce_memory_cap(self, requesting_model: str) -> None:
        if self._cfg.max_total_memory_gb is None:
            return

        loop = asyncio.get_running_loop()
        while True:
            total_gb = await loop.run_in_executor(None, self._measure_total_rss_gb)
            if total_gb <= self._cfg.max_total_memory_gb:
                break
            victim = self._pick_lru_victim(requesting_model)
            if victim is None:
                log.warning(
                    "Memory cap %.1f GB exceeded (%.1f GB used) but no evictable model found",
                    self._cfg.max_total_memory_gb,
                    total_gb,
                )
                break
            log.info(
                "Evicting '%s' (LRU) to stay under %.1f GB cap (currently %.1f GB)",
                victim,
                self._cfg.max_total_memory_gb,
                total_gb,
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

    def _measure_total_rss_gb(self) -> float:
        pids = _load_pids()
        total = 0
        for model in self._cfg.models:
            pid = pids.get(model.name)
            if pid and _is_alive(pid):
                try:
                    total += psutil.Process(pid).memory_info().rss
                except psutil.NoSuchProcess:
                    pass
        return total / (1024 ** 3)

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
