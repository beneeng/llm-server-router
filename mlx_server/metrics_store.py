"""Persistent, content-free request metrics.

One row of numbers per finished request, in a SQLite file that survives router
and model restarts. Rows come from the metrics envelopes that mlx-vlm backends
already record (`GET /v1/metrics`); see metrics_poller.py.

Privacy is structural, not a promise: only the keys in `_COLUMNS` are copied out
of an envelope, numeric columns are coerced to numbers, and text columns accept
only short enum-like tokens. A prompt, a completion, or any field a future
mlx-vlm adds therefore cannot reach the database.
"""
from __future__ import annotations

import math
import re
import sqlite3
import threading
from datetime import date
from pathlib import Path

DEFAULT_RETENTION_ROWS = 50_000
MAX_RECENT = 5000

# column, envelope key, kind
_COLUMNS: list[tuple[str, str, str]] = [
    ("ts", "timestamp_unix", "real"),
    ("endpoint", "endpoint", "token"),
    ("stream", "stream", "bool"),
    ("prompt_tokens", "prompt_tokens", "int"),
    ("completion_tokens", "completion_tokens", "int"),
    ("generated_tokens", "generated_tokens", "int"),
    ("reasoning_tokens", "reasoning_tokens", "int"),
    ("prefill_tok_s", "prefill_tok_s", "real"),
    ("decode_tok_s", "decode_tok_s", "real"),
    ("ttft_s", "ttft_s", "real"),
    ("request_elapsed_s", "request_elapsed_s", "real"),
    ("decode_elapsed_s", "decode_elapsed_s", "real"),
    ("peak_memory_gb", "peak_memory_gb", "real"),
    ("finish_reason", "finish_reason", "token"),
    ("thinking", "thinking_enabled", "bool"),
    ("tool_calls", "tool_calls", "bool"),
    ("apc_enabled", "apc_enabled", "bool"),
]

_SQL_TYPES = {"real": "REAL", "int": "INTEGER", "bool": "INTEGER", "token": "TEXT"}
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_./:\-]{1,64}$")

# (label, inclusive lower bound, exclusive upper bound) on prompt tokens
_BUCKETS: list[tuple[str, int, int | None]] = [
    ("<2k", 0, 2_000),
    ("2k-8k", 2_000, 8_000),
    ("8k-32k", 8_000, 32_000),
    ("32k+", 32_000, None),
]


def _coerce(value, kind: str):
    """Return a safe value for the column, or None when it does not fit."""
    if value is None:
        return None
    if kind == "token":
        return value if isinstance(value, str) and _TOKEN_RE.match(value) else None
    if kind == "bool":
        return int(value) if isinstance(value, (bool, int)) and value in (0, 1) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return float(value) if kind == "real" else int(value)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


class MetricsStore:
    def __init__(self, path: Path, retention_rows: int = DEFAULT_RETENTION_ROWS) -> None:
        self.path = Path(path)
        self.retention_rows = retention_rows
        self._pending: dict[tuple[str, str], list[int]] = {}
        self._pending_lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            cols = ",\n".join(f"{c} {_SQL_TYPES[k]}" for c, _, k in _COLUMNS)
            db.execute(
                "CREATE TABLE IF NOT EXISTS requests ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT NOT NULL, "
                f"drafter TEXT, {cols})"
            )
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS requests_dedup "
                "ON requests(model, ts, prompt_tokens, completion_tokens)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS router_counts ("
                "model TEXT NOT NULL, day TEXT NOT NULL, requests INTEGER NOT NULL, "
                "errors INTEGER NOT NULL, PRIMARY KEY (model, day))"
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    # ── writing ──────────────────────────────────────────────────────────────

    def insert_many(self, model: str, envelopes: list[dict], drafter: str | None = None) -> int:
        """Store allow-listed fields of each envelope; return how many were new."""
        drafter = _coerce(drafter, "token")
        names = [c for c, _, _ in _COLUMNS]
        sql = (
            f"INSERT OR IGNORE INTO requests (model, drafter, {', '.join(names)}) "
            f"VALUES (?, ?, {', '.join('?' * len(names))})"
        )
        inserted = 0
        with self._connect() as db:
            for env in envelopes:
                if not isinstance(env, dict):
                    continue
                values = [_coerce(env.get(key), kind) for _, key, kind in _COLUMNS]
                if values[0] is None:  # no usable timestamp -> cannot dedupe or order
                    continue
                inserted += db.execute(sql, [model, drafter, *values]).rowcount
        return inserted

    def prune(self) -> int:
        """Delete everything but the newest `retention_rows` rows."""
        with self._connect() as db:
            return db.execute(
                "DELETE FROM requests WHERE id NOT IN "
                "(SELECT id FROM requests ORDER BY ts DESC, id DESC LIMIT ?)",
                (self.retention_rows,),
            ).rowcount

    def count_request(self, model: str) -> None:
        """Cheap in-memory tally for the proxy hot path; persisted by flush()."""
        self._tally(model, requests=1, errors=0)

    def count_error(self, model: str) -> None:
        """Count a failed request (backend down, HTTP >= 400); call besides count_request."""
        self._tally(model, requests=0, errors=1)

    def _tally(self, model: str, requests: int, errors: int) -> None:
        key = (model, date.today().isoformat())
        with self._pending_lock:
            counts = self._pending.setdefault(key, [0, 0])
            counts[0] += requests
            counts[1] += errors

    def flush(self) -> None:
        with self._pending_lock:
            pending, self._pending = self._pending, {}
        if not pending:
            return
        with self._connect() as db:
            for (model, day), (requests, errors) in pending.items():
                db.execute(
                    "INSERT INTO router_counts (model, day, requests, errors) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(model, day) DO UPDATE SET "
                    "requests = requests + excluded.requests, errors = errors + excluded.errors",
                    (model, day, requests, errors),
                )

    # ── reading ──────────────────────────────────────────────────────────────

    def recent(self, model: str | None = None, limit: int = 100) -> list[dict]:
        limit = max(1, min(int(limit), MAX_RECENT))
        where, args = ("WHERE model = ?", [model]) if model else ("", [])
        with self._connect() as db:
            rows = db.execute(
                f"SELECT * FROM requests {where} ORDER BY ts DESC, id DESC LIMIT ?",
                [*args, limit],
            ).fetchall()
        return [{k: row[k] for k in row.keys() if k != "id"} for row in rows]

    def router_counts(self) -> list[dict]:
        self.flush()
        with self._connect() as db:
            rows = db.execute(
                "SELECT model, SUM(requests) AS requests, SUM(errors) AS errors "
                "FROM router_counts GROUP BY model ORDER BY model"
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self, model: str | None = None, since: float | None = None) -> dict:
        """Totals plus p50/p90 of prefill/decode tok/s per prompt-size bucket.

        Decode speed falls with context, so one overall average hides the signal.
        """
        clauses, args = [], []
        if model:
            clauses.append("model = ?")
            args.append(model)
        if since is not None:
            clauses.append("ts >= ?")
            args.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as db:
            rows = db.execute(
                "SELECT prompt_tokens, prefill_tok_s, decode_tok_s, ttft_s "
                f"FROM requests {where}",
                args,
            ).fetchall()

        buckets = []
        for label, low, high in _BUCKETS:
            inside = [
                r for r in rows
                if r["prompt_tokens"] is not None
                and r["prompt_tokens"] >= low
                and (high is None or r["prompt_tokens"] < high)
            ]
            prefill = [r["prefill_tok_s"] for r in inside if r["prefill_tok_s"] is not None]
            decode = [r["decode_tok_s"] for r in inside if r["decode_tok_s"] is not None]
            ttft = [r["ttft_s"] for r in inside if r["ttft_s"] is not None]
            buckets.append({
                "bucket": label,
                "requests": len(inside),
                "prefill_p50": _percentile(prefill, 0.5),
                "prefill_p90": _percentile(prefill, 0.9),
                "decode_p50": _percentile(decode, 0.5),
                "decode_p90": _percentile(decode, 0.9),
                "ttft_p50": _percentile(ttft, 0.5),
            })
        return {"requests": len(rows), "buckets": buckets}
