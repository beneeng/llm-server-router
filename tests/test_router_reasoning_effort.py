"""The router rewrites reasoning_effort per model before proxying (end to end).

Run with:
    ~/.local/share/uv/tools/mlx-server/bin/python -m pytest tests/test_router_reasoning_effort.py -v
"""
from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from mlx_server import router
from mlx_server.config import ModelConfig, ServerConfig


class _StubMemory:
    before_stop = None

    async def ensure_model_ready(self, name: str) -> None: ...
    def mark_busy(self, name: str) -> None: ...
    def mark_idle(self, name: str) -> None: ...


def _post(monkeypatch, model: ModelConfig, payload: dict) -> dict:
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(router, "_cfg", ServerConfig(models=[model]))
    monkeypatch.setattr(router, "_mem", _StubMemory())
    real = httpx.AsyncClient
    monkeypatch.setattr(
        router.httpx, "AsyncClient",
        lambda *a, **k: real(transport=httpx.MockTransport(handler), **k),
    )
    TestClient(router.app).post("/v1/chat/completions", json={"model": model.name, **payload})
    return seen


def test_high_reaches_backend_as_xhigh(monkeypatch):
    m = ModelConfig(name="q", source="org/q", backend="mlx-vlm", reasoning_effort_map={"high": "xhigh"})
    seen = _post(monkeypatch, m, {"messages": [], "reasoning_effort": "high"})
    assert seen["reasoning_effort"] == "xhigh"


def test_high_untouched_without_map(monkeypatch):
    m = ModelConfig(name="q", source="org/q", backend="mlx-vlm")
    seen = _post(monkeypatch, m, {"messages": [], "reasoning_effort": "high"})
    assert seen["reasoning_effort"] == "high"
