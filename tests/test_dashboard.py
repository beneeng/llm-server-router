"""Tests for the read-only dashboard API (mlx_server/dashboard.py).

The dashboard may show settings, model status and numbers, never prompts or
processed data, and it must not offer any way to change anything.

Run with:
    ~/.local/share/uv/tools/mlx-server/bin/python -m pytest tests/test_dashboard.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlx_server import dashboard, memory as memory_mod, router
from mlx_server.config import ModelConfig, ServerConfig
from mlx_server.metrics_store import MetricsStore

MARKER = "ZXQ-PRIVACY-MARKER-7731"
HOME = str(Path.home())


def _cfg(**over) -> ServerConfig:
    return ServerConfig(
        max_total_memory_gb=100.0,
        models=[
            ModelConfig(name="big", source=f"{HOME}/.cache/mlx-vlm/big-checkpoint",
                        backend="mlx-vlm", mtp=True, memory_gb=80.0,
                        extra_args=["--lora-paths", f"{HOME}/.local/share/mlx/loras/x.safetensors"]),
            ModelConfig(name="small", source="org/small", backend="mlx-lm"),
        ],
        **over,
    )


def _env(ts: float) -> dict:
    return {"timestamp_unix": ts, "endpoint": "/chat/completions", "prompt_tokens": 3000,
            "completion_tokens": 50, "prefill_tok_s": 400.0, "decode_tok_s": 45.0,
            "ttft_s": 7.5, "finish_reason": "stop", "prompt_text": MARKER, "completion": MARKER}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    cfg = _cfg(dashboard_enabled=True)
    store = MetricsStore(tmp_path / "m.db")
    store.insert_many("big", [_env(1.0), _env(2.0)])
    store.count_request("small")
    monkeypatch.setattr(router, "_cfg", cfg)
    monkeypatch.setattr(router, "_store", store)
    monkeypatch.setattr(router, "_mem", None)
    monkeypatch.setattr(memory_mod, "_load_pids", lambda: {"big": 4242})
    monkeypatch.setattr(memory_mod, "_is_alive", lambda pid: pid == 4242)
    monkeypatch.setattr(memory_mod, "_proc_rss_bytes", lambda pid: 50 * 1024**3)
    monkeypatch.setattr(memory_mod, "_available_memory_gb", lambda: 40.0)
    monkeypatch.setattr(dashboard, "_load_pids", lambda: {"big": 4242})
    monkeypatch.setattr(dashboard, "_is_alive", lambda pid: pid == 4242)
    monkeypatch.setattr(dashboard, "APC_DIR", tmp_path / "apc")
    return TestClient(router.app)


GET_PATHS = [
    "/_dashboard/",
    "/_dashboard/api/overview",
    "/_dashboard/api/config",
    "/_dashboard/api/requests",
    "/_dashboard/api/stats",
]


@pytest.mark.parametrize("path", GET_PATHS)
def test_endpoints_respond(client, path):
    assert client.get(path).status_code == 200


def test_no_dashboard_route_accepts_writes():
    methods = {
        m for route in router.app.routes
        if getattr(route, "path", "").startswith("/_dashboard")
        for m in getattr(route, "methods", set())
    }
    assert methods <= {"GET", "HEAD"}


def test_no_endpoint_leaks_prompt_or_completion_text(client):
    """The store was fed envelopes containing a marker in text fields."""
    for path in [*GET_PATHS, "/_dashboard/api/cache/big", "/_dashboard/api/cache/small"]:
        assert MARKER not in client.get(path).text, path


def test_config_hides_the_home_directory(client):
    text = client.get("/_dashboard/api/config").text
    assert HOME not in text
    assert "~/.cache/mlx-vlm/big-checkpoint" in text


def test_config_exposes_settings_and_models(client):
    data = client.get("/_dashboard/api/config").json()
    assert data["server"]["max_total_memory_gb"] == 100.0
    big = next(m for m in data["models"] if m["name"] == "big")
    assert big["backend"] == "mlx-vlm" and big["mtp"] is True and big["memory_gb"] == 80.0


def test_overview_reports_status_and_memory(client):
    data = client.get("/_dashboard/api/overview").json()
    models = {m["name"]: m for m in data["models"]}
    assert models["big"]["alive"] is True and models["big"]["rss_gb"] == 50.0
    assert models["small"]["alive"] is False
    assert data["memory"]["cap_gb"] == 100.0
    assert {c["model"]: c["requests"] for c in data["router_counts"]} == {"small": 1}


def test_requests_returns_only_numbers_and_enums(client):
    rows = client.get("/_dashboard/api/requests?limit=1").json()["requests"]
    assert len(rows) == 1
    assert rows[0]["prefill_tok_s"] == 400.0
    assert set(rows[0]) <= {
        "model", "drafter", "ts", "endpoint", "stream", "prompt_tokens", "completion_tokens",
        "generated_tokens", "reasoning_tokens", "prefill_tok_s", "decode_tok_s", "ttft_s",
        "request_elapsed_s", "decode_elapsed_s", "peak_memory_gb", "finish_reason", "thinking",
        "tool_calls", "apc_enabled",
    }


def test_requests_limit_is_clamped(client):
    assert client.get("/_dashboard/api/requests?limit=999999999").status_code == 200
    assert client.get("/_dashboard/api/requests?limit=abc").status_code == 422


def test_stats_buckets_by_prompt_size(client):
    data = client.get("/_dashboard/api/stats?model=big").json()
    bucket = next(b for b in data["buckets"] if b["bucket"] == "2k-8k")
    assert bucket["requests"] == 2 and bucket["decode_p50"] == 45.0


def test_cache_endpoint_for_stopped_and_unknown_models(client):
    assert client.get("/_dashboard/api/cache/small").json() == {"running": False}
    assert client.get("/_dashboard/api/cache/nope").status_code == 404


def test_cache_stats_pass_through_numbers_only(client, monkeypatch):
    monkeypatch.setattr(dashboard, "_fetch_json", lambda url: {
        "enabled": True, "hits": 3, "stored_tokens": 1000, "note": MARKER, "nested": {"a": 1},
    })
    data = client.get("/_dashboard/api/cache/big").json()
    assert data == {"running": True, "stats": {"enabled": True, "hits": 3, "stored_tokens": 1000}}


def test_disabled_dashboard_is_not_served(tmp_path, monkeypatch):
    monkeypatch.setattr(router, "_cfg", _cfg(dashboard_enabled=False))
    monkeypatch.setattr(router, "_store", MetricsStore(tmp_path / "m.db"))
    c = TestClient(router.app)
    assert c.get("/_dashboard/api/overview").status_code == 404
    assert c.get("/_dashboard/").status_code == 404


def test_responses_are_valid_json(client):
    for path in GET_PATHS[1:]:
        json.loads(client.get(path).text)


def test_dashboard_is_opt_in_by_default():
    assert ServerConfig().dashboard_enabled is False
