"""Tests for mlx_server/metrics_poller.py.

No sockets, no processes: the pid table, liveness and the HTTP fetch are faked.

Run with:
    ~/.local/share/uv/tools/mlx-server/bin/python -m pytest tests/test_metrics_poller.py -v
"""
from __future__ import annotations

from mlx_server import metrics_poller as poller
from mlx_server.config import ModelConfig, ServerConfig
from mlx_server.metrics_store import MetricsStore


def _cfg() -> ServerConfig:
    return ServerConfig(models=[
        ModelConfig(name="vlm-running", source="org/a", backend="mlx-vlm",
                    extra_args=["--draft-model", "org/d", "--draft-kind", "dflash"]),
        ModelConfig(name="vlm-stopped", source="org/b", backend="mlx-vlm"),
        ModelConfig(name="lm-running", source="org/c", backend="mlx-lm"),
    ])


def _envelope(ts: float) -> dict:
    return {"timestamp_unix": ts, "endpoint": "/chat/completions", "model": "org/a",
            "prompt_tokens": 100, "completion_tokens": 10, "prefill_tok_s": 300.0,
            "decode_tok_s": 50.0, "finish_reason": "stop"}


def _patch(monkeypatch, alive: set[str], fetched: list[str], payload: dict | None):
    pids = {name: 1000 + i for i, name in enumerate(sorted(alive))}
    monkeypatch.setattr(poller, "_load_pids", lambda: pids)
    monkeypatch.setattr(poller, "_is_alive", lambda pid: pid in pids.values())

    def fake_get(url: str) -> dict | None:
        fetched.append(url)
        return payload

    monkeypatch.setattr(poller, "_fetch_metrics", fake_get)


def test_polls_only_running_mlx_vlm_models(tmp_path, monkeypatch):
    cfg = _cfg()
    fetched: list[str] = []
    _patch(monkeypatch, {"vlm-running", "lm-running"}, fetched, {"recent": [_envelope(1.0)]})
    store = MetricsStore(tmp_path / "m.db")

    poller.poll_once(cfg, store)

    port = cfg.model_by_name("vlm-running").port
    assert fetched == [f"http://127.0.0.1:{port}/v1/metrics"]  # not the stopped one, not mlx-lm


def test_rows_use_the_router_alias_and_config_drafter(tmp_path, monkeypatch):
    cfg = _cfg()
    _patch(monkeypatch, {"vlm-running"}, [], {"recent": [_envelope(1.0)]})
    store = MetricsStore(tmp_path / "m.db")

    poller.poll_once(cfg, store)

    row = store.recent()[0]
    assert row["model"] == "vlm-running"
    assert row["drafter"] == "dflash"


def test_polling_twice_does_not_duplicate(tmp_path, monkeypatch):
    cfg = _cfg()
    _patch(monkeypatch, {"vlm-running"}, [], {"recent": [_envelope(1.0), _envelope(2.0)]})
    store = MetricsStore(tmp_path / "m.db")

    assert poller.poll_once(cfg, store) == 2
    assert poller.poll_once(cfg, store) == 0


def test_unreachable_backend_is_skipped_not_fatal(tmp_path, monkeypatch):
    cfg = _cfg()
    _patch(monkeypatch, {"vlm-running"}, [], None)  # fetch failed
    store = MetricsStore(tmp_path / "m.db")

    assert poller.poll_once(cfg, store) == 0


def test_malformed_payload_is_ignored(tmp_path, monkeypatch):
    cfg = _cfg()
    _patch(monkeypatch, {"vlm-running"}, [], {"recent": "garbage"})
    store = MetricsStore(tmp_path / "m.db")

    assert poller.poll_once(cfg, store) == 0


def test_poll_model_reads_one_model_before_eviction(tmp_path, monkeypatch):
    """Used right before a model is stopped, so its last requests are not lost."""
    cfg = _cfg()
    fetched: list[str] = []
    _patch(monkeypatch, {"vlm-running"}, fetched, {"recent": [_envelope(5.0)]})
    store = MetricsStore(tmp_path / "m.db")

    assert poller.poll_model(cfg, store, "vlm-running") == 1
    assert poller.poll_model(cfg, store, "lm-running") == 0  # not an mlx-vlm backend


def test_mtplx_records_are_translated_and_leak_no_content(tmp_path, monkeypatch):
    cfg = ServerConfig(models=[ModelConfig(name="mt", source="org/m", backend="mtplx")])
    record = {
        "completed_at_s": 1794562893.7, "prompt_tokens": 46704, "completion_tokens": 173,
        "prefill_tok_s": 168.3, "decode_tok_s": 63.8, "ttft_s": 6.1, "request_elapsed_s": 8.8,
        "decode_elapsed_s": 2.7, "peak_memory_bytes": 94_701_949_770, "finish_reason": "tool_calls",
        "request_enable_thinking": True, "tool_call_count": 1,
        "request_last_user_preview": "SECRET-PROMPT-MARKER",
    }
    fetched: list[str] = []
    _patch(monkeypatch, {"mt"}, fetched, {"latest": record, "recent": [record]})
    store = MetricsStore(tmp_path / "m.db")

    assert poller.poll_once(cfg, store) == 1

    assert fetched == [f"http://127.0.0.1:{cfg.model_by_name('mt').port}/metrics"]
    row = store.recent()[0]
    assert (row["model"], row["prompt_tokens"], row["decode_tok_s"]) == ("mt", 46704, 63.8)
    assert row["peak_memory_gb"] == 94.701949770
    assert (row["thinking"], row["tool_calls"], row["finish_reason"]) == (1, 1, "tool_calls")
    assert "SECRET-PROMPT-MARKER" not in str(dict(row))
