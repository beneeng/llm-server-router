"""The proxy tallies prompts per model for the dashboard, without touching bodies.

A fake backend answers via httpx.MockTransport; the memory manager is a stub so
no model is ever started.

Run with:
    ~/.local/share/uv/tools/mlx-server/bin/python -m pytest tests/test_router_counting.py -v
"""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from mlx_server import router
from mlx_server.config import ModelConfig, ServerConfig
from mlx_server.metrics_store import MetricsStore


class _StubMemory:
    before_stop = None

    async def ensure_model_ready(self, name: str) -> None: ...
    def mark_busy(self, name: str) -> None: ...
    def mark_idle(self, name: str) -> None: ...


def _setup(monkeypatch, tmp_path, handler) -> MetricsStore:
    store = MetricsStore(tmp_path / "m.db")
    cfg = ServerConfig(dashboard_enabled=True, models=[ModelConfig(name="small", source="org/small", backend="mlx-lm")])
    monkeypatch.setattr(router, "_cfg", cfg)
    monkeypatch.setattr(router, "_store", store)
    monkeypatch.setattr(router, "_mem", _StubMemory())
    real = httpx.AsyncClient
    monkeypatch.setattr(
        router.httpx, "AsyncClient",
        lambda *a, **k: real(transport=httpx.MockTransport(handler), **k),
    )
    return store


def _counts(store: MetricsStore) -> dict:
    return {c["model"]: c for c in store.router_counts()}


def test_post_is_counted_and_response_passes_through(monkeypatch, tmp_path):
    store = _setup(monkeypatch, tmp_path, lambda req: httpx.Response(200, json={"ok": True}))
    client = TestClient(router.app)

    resp = client.post("/v1/chat/completions", json={"model": "small", "messages": []})

    assert resp.status_code == 200 and resp.json() == {"ok": True}
    assert _counts(store)["small"]["requests"] == 1
    assert _counts(store)["small"]["errors"] == 0


def test_backend_error_status_counts_as_error(monkeypatch, tmp_path):
    store = _setup(monkeypatch, tmp_path, lambda req: httpx.Response(500, json={"error": "x"}))
    TestClient(router.app).post("/v1/chat/completions", json={"model": "small", "messages": []})

    assert _counts(store)["small"] == {"model": "small", "requests": 1, "errors": 1}


def test_unreachable_backend_counts_as_error(monkeypatch, tmp_path):
    def refuse(req):
        raise httpx.ConnectError("down")

    store = _setup(monkeypatch, tmp_path, refuse)
    resp = TestClient(router.app).post("/v1/chat/completions", json={"model": "small", "messages": []})

    assert resp.status_code == 503
    assert _counts(store)["small"] == {"model": "small", "requests": 1, "errors": 1}


def test_get_requests_are_not_counted(monkeypatch, tmp_path):
    store = _setup(monkeypatch, tmp_path, lambda req: httpx.Response(200, json={}))
    TestClient(router.app).get("/v1/models/info?model=small")

    assert "small" not in _counts(store)


def test_counting_failure_never_breaks_the_proxy(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, lambda req: httpx.Response(200, json={"ok": True}))

    def broken():
        raise RuntimeError("disk full")

    monkeypatch.setattr(router, "_metrics_store", broken)
    resp = TestClient(router.app).post("/v1/chat/completions", json={"model": "small", "messages": []})

    assert resp.status_code == 200
