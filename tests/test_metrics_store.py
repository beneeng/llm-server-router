"""Tests for mlx_server/metrics_store.py.

The store keeps one row of numbers per finished request. The privacy rule is
structural: only allow-listed keys are copied from a backend's metrics envelope,
and text-typed columns only accept short enum-like tokens, so no prompt or
completion text can reach the database even if a backend starts emitting it.

Run with:
    ~/.local/share/uv/tools/mlx-server/bin/python -m pytest tests/test_metrics_store.py -v
"""
from __future__ import annotations

import sqlite3

from mlx_server.metrics_store import MetricsStore


def _env(ts: float, **over) -> dict:
    base = {
        "timestamp_unix": ts,
        "endpoint": "/chat/completions",
        "model": "org/some-hf-id",
        "stream": True,
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "generated_tokens": 120,
        "reasoning_tokens": 20,
        "prefill_tok_s": 500.0,
        "decode_tok_s": 50.0,
        "ttft_s": 2.0,
        "request_elapsed_s": 4.0,
        "decode_elapsed_s": 2.0,
        "peak_memory_gb": 80.5,
        "finish_reason": "stop",
        "thinking_enabled": True,
        "tool_calls": False,
        "apc_enabled": True,
    }
    return base | over


def test_insert_is_idempotent_for_repeated_polls(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    envs = [_env(1.0), _env(2.0)]
    assert store.insert_many("alias", envs) == 2
    assert store.insert_many("alias", envs) == 0  # same 32 entries polled again
    assert len(store.recent(limit=100)) == 2


def test_uses_router_alias_not_backend_model_id(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    store.insert_many("my-alias", [_env(1.0)])
    assert store.recent()[0]["model"] == "my-alias"


def test_unknown_keys_never_reach_the_database(tmp_path):
    """A future mlx-vlm field, including a text one, must not be stored."""
    marker = "ZXQ-PRIVACY-MARKER-7731"
    store = MetricsStore(tmp_path / "m.db")
    store.insert_many("alias", [_env(1.0, prompt_text=marker, completion=marker, debug=marker)])

    raw = sqlite3.connect(tmp_path / "m.db")
    dump = "\n".join(raw.iterdump())
    assert marker not in dump
    assert marker not in repr(store.recent())


def test_text_columns_reject_free_text(tmp_path):
    """Enum-like columns only take short tokens; a sentence is dropped to NULL."""
    marker = "please summarise my secret document"
    store = MetricsStore(tmp_path / "m.db")
    store.insert_many("alias", [_env(1.0, finish_reason=marker, endpoint=marker)])

    row = store.recent()[0]
    assert row["finish_reason"] is None
    assert row["endpoint"] is None
    assert marker not in "\n".join(sqlite3.connect(tmp_path / "m.db").iterdump())


def test_bad_numeric_values_become_null(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    store.insert_many("alias", [_env(1.0, prefill_tok_s="not a number", prompt_tokens={"x": 1})])
    row = store.recent()[0]
    assert row["prefill_tok_s"] is None
    assert row["prompt_tokens"] is None


def test_history_survives_reopening(tmp_path):
    path = tmp_path / "m.db"
    MetricsStore(path).insert_many("alias", [_env(1.0), _env(2.0)])
    assert len(MetricsStore(path).recent()) == 2


def test_prune_keeps_the_newest_rows(tmp_path):
    store = MetricsStore(tmp_path / "m.db", retention_rows=3)
    store.insert_many("alias", [_env(float(i)) for i in range(1, 8)])
    store.prune()
    rows = store.recent(limit=100)
    assert [r["ts"] for r in rows] == [7.0, 6.0, 5.0]  # newest first


def test_recent_filters_by_model_and_caps_limit(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    store.insert_many("a", [_env(1.0), _env(2.0)])
    store.insert_many("b", [_env(3.0)])
    assert [r["model"] for r in store.recent(model="a")] == ["a", "a"]
    assert len(store.recent(limit=1)) == 1
    assert len(store.recent(limit=10**9)) == 3  # limit is clamped, not trusted


def test_stats_bucket_by_prompt_size(tmp_path):
    store = MetricsStore(tmp_path / "m.db")
    store.insert_many("a", [
        _env(1.0, prompt_tokens=500, decode_tok_s=60.0),
        _env(2.0, prompt_tokens=600, decode_tok_s=58.0),
        _env(3.0, prompt_tokens=10_000, decode_tok_s=45.0),
        _env(4.0, prompt_tokens=40_000, decode_tok_s=30.0),
    ])
    buckets = {b["bucket"]: b for b in store.stats(model="a")["buckets"]}
    assert buckets["<2k"]["requests"] == 2
    assert buckets["2k-8k"]["requests"] == 0
    assert buckets["8k-32k"]["decode_p50"] == 45.0
    assert buckets["32k+"]["decode_p50"] == 30.0


def test_router_counts_cover_models_without_envelopes(tmp_path):
    """mlx-lm / audio / image backends write no envelopes; the router still counts them."""
    store = MetricsStore(tmp_path / "m.db")
    store.count_request("gemma4-e4b-qat")
    store.count_request("gemma4-e4b-qat")
    store.count_error("gemma4-e4b-qat")
    store.flush()
    counts = {c["model"]: c for c in store.router_counts()}
    assert counts["gemma4-e4b-qat"]["requests"] == 2
    assert counts["gemma4-e4b-qat"]["errors"] == 1
