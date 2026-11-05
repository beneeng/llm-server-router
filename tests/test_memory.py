"""Tests for graceful eviction in mlx_server/memory.py.

MemoryManager.ensure_model_ready() evicts LRU models under memory pressure.
These tests verify: (1) an idle model is preferred over a busy one, (2) a
busy model is waited on rather than killed immediately, (3) waiting is bounded
by evict_wait_timeout_seconds and falls back to a force-stop.

No real subprocess/psutil calls are made — manager.py and the RSS/available
memory functions are monkeypatched with in-memory fakes.

Run with:
    ~/.local/share/uv/tools/mlx-server/bin/python -m pytest tests/test_memory.py -v
"""
from __future__ import annotations

import asyncio

from mlx_server import memory as memory_mod
from mlx_server.config import ModelConfig, ServerConfig
from mlx_server.memory import MemoryManager


def _cfg(*names: str, evict_wait_timeout_seconds: float = 5.0) -> ServerConfig:
    models = [ModelConfig(name=n, source=f"src-{n}", backend="mlx-lm") for n in names]
    return ServerConfig(
        max_total_memory_gb=10.0,
        min_free_memory_gb=0.0,
        evict_wait_timeout_seconds=evict_wait_timeout_seconds,
        models=models,
    )


def _patch_pressure(
    monkeypatch,
    alive: dict[str, int],
    stopped: list[str],
    pressure_names: set[str],
) -> None:
    """Fake out manager.py + RSS lookups. "Over cap" stays true for as long as
    any model in `pressure_names` is still alive; goes away once they're gone."""

    def fake_load_pids() -> dict[str, int]:
        return dict(alive)

    def fake_is_alive(pid: int) -> bool:
        return pid in alive.values()

    def fake_stop_process(name: str) -> bool:
        stopped.append(name)
        alive.pop(name, None)
        return True

    def fake_total_rss(cfg: ServerConfig) -> float:
        return 999.0 if (pressure_names & alive.keys()) else 0.0

    def fake_available() -> float:
        return 0.0 if (pressure_names & alive.keys()) else 999.0

    monkeypatch.setattr(memory_mod, "_load_pids", fake_load_pids)
    monkeypatch.setattr(memory_mod, "_is_alive", fake_is_alive)
    monkeypatch.setattr(memory_mod, "stop_process", fake_stop_process)
    monkeypatch.setattr(memory_mod, "_total_model_rss_gb", fake_total_rss)
    monkeypatch.setattr(memory_mod, "_available_memory_gb", fake_available)


async def _run_enforce(mgr: MemoryManager, requesting: str = "requester") -> None:
    async with mgr._lock:
        await mgr._enforce_memory_cap(requesting, projected_additional_gb=0.0)


def test_prefers_idle_victim_over_busy(monkeypatch):
    """Both an idle and a busy model are evictable candidates: the idle one is
    killed and the busy one is never touched or waited on."""
    alive = {"a": 1001, "b": 1002}
    stopped: list[str] = []
    _patch_pressure(monkeypatch, alive, stopped, pressure_names={"b"})

    cfg = _cfg("a", "b")
    mgr = MemoryManager(cfg)
    mgr._touch("a")
    mgr._touch("b")
    mgr.mark_busy("a")  # busy for the whole test

    asyncio.run(_run_enforce(mgr))

    assert stopped == ["b"]
    assert "a" in alive


def test_waits_for_busy_model_then_evicts(monkeypatch):
    """The sole evictable candidate is busy: eviction waits until it goes idle
    and only stops it afterwards, not before."""
    alive = {"a": 1001}
    stopped: list[str] = []
    _patch_pressure(monkeypatch, alive, stopped, pressure_names={"a"})

    cfg = _cfg("a", evict_wait_timeout_seconds=5.0)
    mgr = MemoryManager(cfg)
    mgr._touch("a")
    mgr.mark_busy("a")

    async def scenario() -> None:
        task = asyncio.create_task(_run_enforce(mgr))
        await asyncio.sleep(0.05)
        assert stopped == [], "should still be waiting, not force-stopped early"
        mgr.mark_idle("a")
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(scenario())

    assert stopped == ["a"]


def test_timeout_falls_back_to_force_stop(monkeypatch):
    """If the busy model never finishes, it's force-stopped once
    evict_wait_timeout_seconds elapses rather than waiting forever."""
    alive = {"a": 1001}
    stopped: list[str] = []
    _patch_pressure(monkeypatch, alive, stopped, pressure_names={"a"})

    cfg = _cfg("a", evict_wait_timeout_seconds=0.05)
    mgr = MemoryManager(cfg)
    mgr._touch("a")
    mgr.mark_busy("a")  # never marked idle

    asyncio.run(asyncio.wait_for(_run_enforce(mgr), timeout=2.0))

    assert stopped == ["a"]


def test_evicts_running_model_the_router_never_saw(monkeypatch):
    """A model started outside the router's request path (`mlx start --model X`,
    or still running from before a router restart) is not in the in-memory LRU.
    It must still be evictable, otherwise a second model can never be loaded and
    the request fails with "Not enough free memory"."""
    alive = {"a": 1001}
    stopped: list[str] = []
    _patch_pressure(monkeypatch, alive, stopped, pressure_names={"a"})

    cfg = _cfg("a", "requester")
    mgr = MemoryManager(cfg)
    # "a" is deliberately never _touch()ed: only the requester went through the router.
    mgr._touch("requester")

    asyncio.run(_run_enforce(mgr))

    assert stopped == ["a"]


def test_untracked_model_is_evicted_before_a_tracked_one(monkeypatch):
    """A model with no recorded router use is older than any that has one."""
    alive = {"seen": 1001, "unseen": 1002}
    stopped: list[str] = []
    _patch_pressure(monkeypatch, alive, stopped, pressure_names={"seen", "unseen"})

    cfg = _cfg("seen", "unseen", "requester")
    mgr = MemoryManager(cfg)
    mgr._touch("seen")
    mgr._touch("requester")

    asyncio.run(_run_enforce(mgr))

    assert stopped[0] == "unseen"
