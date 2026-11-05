"""Read-only technical dashboard: settings, models, statistics, recent request speeds.

Everything here is GET-only and exposes numbers and configuration, never prompts
or responses (the metrics store cannot hold them, see metrics_store.py). The
router mounts this before its catch-all proxy route so `/_dashboard/*` is not
mistaken for a model request.

No authentication: the router listens on 127.0.0.1 by default. If you bind it to
a reachable address, anyone who can reach the port can read these pages.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from .config import ModelConfig, ServerConfig
from .manager import APC_DIR, _is_alive, _load_pids
from .memory import MemoryManager
from .metrics_store import MetricsStore

_STARTED_AT = time.time()
_STATIC_DIR = Path(__file__).parent / "dashboard"
_APC_SIZE_TTL_SECONDS = 60.0
_apc_size_cache: dict[str, tuple[float, int]] = {}


def _fetch_json(url: str) -> dict | None:
    try:
        resp = httpx.get(url, timeout=3.0)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _home_to_tilde(value):
    """Replace the user's home directory with ~ in every string, so a local path
    does not reveal the account name."""
    home = str(Path.home())
    if isinstance(value, str):
        return value.replace(home, "~")
    if isinstance(value, list):
        return [_home_to_tilde(v) for v in value]
    if isinstance(value, dict):
        return {k: _home_to_tilde(v) for k, v in value.items()}
    return value


def _model_view(model: ModelConfig) -> dict:
    return _home_to_tilde({
        "name": model.name,
        "source": model.source,
        "backend": model.backend,
        "modality": model.modality,
        "accepts_image_input": model.accepts_image_input,
        "thinking": model.thinking,
        "mtp": model.mtp,
        "lazy": model.lazy,
        "protected": model.protected,
        "memory_gb": model.memory_gb,
        "port": model.port,
        "defaults": model.defaults.as_request_fields(),
        "extra_args": list(model.extra_args),
    })


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for f in path.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total


def _apc_bytes(name: str) -> int:
    now = time.monotonic()
    cached = _apc_size_cache.get(name)
    if cached and now - cached[0] < _APC_SIZE_TTL_SECONDS:
        return cached[1]
    size = _dir_size(APC_DIR / name)
    _apc_size_cache[name] = (now, size)
    return size


def build_router(
    get_cfg: Callable[[], ServerConfig],
    get_store: Callable[[], MetricsStore],
    get_mem: Callable[[], MemoryManager],
) -> APIRouter:
    def enabled() -> None:
        if not get_cfg().dashboard_enabled:
            raise HTTPException(status_code=404)

    api = APIRouter(prefix="/_dashboard", dependencies=[Depends(enabled)])

    @api.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @api.get("/api/overview")
    def overview() -> dict:
        cfg = get_cfg()
        report = get_mem().memory_report()
        total = next(r for r in report if r["name"] == "__total__")
        live = {r["name"]: r for r in report if r["name"] != "__total__"}
        models = []
        for model in cfg.models:
            row = live.get(model.name, {})
            models.append({
                "name": model.name,
                "backend": model.backend,
                "modality": model.modality,
                "alive": bool(row.get("alive")),
                "rss_gb": row.get("rss_gb", 0.0),
                "memory_gb": model.memory_gb,
                "port": model.port,
                "lazy": model.lazy,
                "protected": model.protected,
                "apc_bytes": _apc_bytes(model.name),
            })
        return {
            "uptime_s": time.time() - _STARTED_AT,
            "models": models,
            "memory": {
                "used_gb": total["rss_gb"],
                "cap_gb": total["cap_gb"],
                "free_gb": total["free_gb"],
            },
            "router_counts": get_store().router_counts(),
        }

    @api.get("/api/config")
    def config() -> dict:
        cfg = get_cfg()
        return {
            "server": {
                "host": cfg.host,
                "port": cfg.port,
                "base_port": cfg.base_port,
                "max_total_memory_gb": cfg.max_total_memory_gb,
                "min_free_memory_gb": cfg.min_free_memory_gb,
                "evict_wait_timeout_seconds": cfg.evict_wait_timeout_seconds,
                "metrics_retention_rows": cfg.metrics_retention_rows,
            },
            "models": [_model_view(m) for m in cfg.models],
        }

    @api.get("/api/requests")
    def requests(model: str | None = None, limit: int = Query(100, ge=1)) -> dict:
        return {"requests": get_store().recent(model=model, limit=limit)}

    @api.get("/api/stats")
    def stats(model: str | None = None, hours: float | None = Query(None, gt=0)) -> dict:
        since = time.time() - hours * 3600 if hours else None
        return get_store().stats(model=model, since=since)

    @api.get("/api/cache/{name}")
    def cache(name: str) -> dict:
        model = get_cfg().model_by_name(name)
        if model is None:
            raise HTTPException(status_code=404, detail="unknown model")
        pid = _load_pids().get(name)
        if not (pid and _is_alive(pid)):
            return {"running": False}
        raw = _fetch_json(f"http://127.0.0.1:{model.port}/v1/cache/stats")
        if raw is None:
            return {"running": True, "stats": None}
        # Counters only: drop strings and nested structures wholesale.
        return {"running": True, "stats": {
            k: v for k, v in raw.items() if isinstance(v, (bool, int, float))
        }}

    return api
