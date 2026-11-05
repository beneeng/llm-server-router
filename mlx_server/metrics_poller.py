"""Collects per-request metrics from running mlx-vlm backends into the MetricsStore.

mlx-vlm already records a content-free metrics envelope per finished request and
serves the last 32 at `GET /v1/metrics`. The router never reads response bytes
for this: it polls that endpoint, so client traffic is untouched, streaming and
non-streaming requests are covered alike, and there is no way for prompt or
completion text to pass through this module.

Polling talks to the backend port directly. It never goes through the proxy, so
it cannot trigger a lazy model start, mark a model busy, or keep the Mac awake.
"""
from __future__ import annotations

import logging
import threading
import time

import httpx

from .config import ModelConfig, ServerConfig
from .manager import _is_alive, _load_pids
from .metrics_store import MetricsStore

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5.0
_PRUNE_INTERVAL_SECONDS = 3600.0
_FETCH_TIMEOUT_SECONDS = 3.0

_thread: threading.Thread | None = None
_lock = threading.Lock()


def _fetch_metrics(url: str) -> dict | None:
    try:
        resp = httpx.get(url, timeout=_FETCH_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _drafter_of(model: ModelConfig) -> str | None:
    """Name of the speculative drafter from config; the envelope does not carry it."""
    args = model.extra_args or []
    if "--draft-kind" in args:
        index = args.index("--draft-kind") + 1
        if index < len(args):
            return args[index]
    return "draft" if "--draft-model" in args else None


def _has_metrics_endpoint(model: ModelConfig) -> bool:
    return model.resolve_backend().name.startswith("mlx-vlm")


def _poll(model: ModelConfig, store: MetricsStore) -> int:
    payload = _fetch_metrics(f"http://127.0.0.1:{model.port}/v1/metrics")
    if payload is None:
        return 0
    recent = payload.get("recent")
    if not isinstance(recent, list):
        return 0
    return store.insert_many(model.name, recent, drafter=_drafter_of(model))


def poll_once(cfg: ServerConfig, store: MetricsStore) -> int:
    """Poll every running mlx-vlm model once. Returns the number of new rows."""
    pids = _load_pids()
    total = 0
    for model in cfg.models:
        pid = pids.get(model.name)
        if not (pid and _is_alive(pid)) or not _has_metrics_endpoint(model):
            continue
        total += _poll(model, store)
    return total


def poll_model(cfg: ServerConfig, store: MetricsStore, name: str) -> int:
    """Poll a single model, e.g. right before it is evicted."""
    model = cfg.model_by_name(name)
    pid = _load_pids().get(name)
    if model is None or not (pid and _is_alive(pid)) or not _has_metrics_endpoint(model):
        return 0
    return _poll(model, store)


def _loop(cfg: ServerConfig, store: MetricsStore) -> None:
    last_prune = 0.0
    while True:
        try:
            poll_once(cfg, store)
            store.flush()
            if time.monotonic() - last_prune > _PRUNE_INTERVAL_SECONDS:
                store.prune()
                last_prune = time.monotonic()
        except Exception:
            logger.exception("metrics poll failed")
        time.sleep(POLL_INTERVAL_SECONDS)


def start_background_poller(cfg: ServerConfig, store: MetricsStore) -> None:
    """Lazily start the daemon polling thread. Idempotent."""
    global _thread
    with _lock:
        if _thread is None:
            _thread = threading.Thread(target=_loop, args=(cfg, store), daemon=True)
            _thread.start()
            logger.info("metrics poller started (interval=%.0fs)", POLL_INTERVAL_SECONDS)
