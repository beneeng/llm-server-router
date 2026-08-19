"""
FastAPI reverse proxy on port 6300.

Routing logic:
  1. Extract `model` from the JSON body (POST) or `model` query param (GET).
  2. Look up the ModelConfig; inject its `defaults` into the body for any key
     the caller did not supply (caller values always win).
  3. Forward the (possibly enriched) request to the model's mlx_lm.server backend.
  4. Stream the response back verbatim.

/v1/models is handled specially: it aggregates the model list from the YAML
config rather than forwarding, so it works even when backends are offline.
"""
from __future__ import annotations

import json
import re
from typing import AsyncGenerator

import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

def _has_image_content(body: dict) -> bool:
    """Return True if any message in the request contains a non-text content block."""
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") not in ("text",):
                    return True
    return False


def _backend_unavailable(model_name: str, port: int) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"error": {
            "message": f"Model '{model_name}' backend is not running (port {port}). Run: mlx start --model {model_name}",
            "type": "service_unavailable",
        }},
    )

from .apc_gc import start_background_gc
from .config import ServerConfig, load_config
from .keepawake import touch as _touch_keepawake
from .memory import MemoryManager

# Loaded once at startup; re-read the file on SIGHUP if you want hot reload.
_cfg: ServerConfig | None = None
_mem: MemoryManager | None = None


def _config() -> ServerConfig:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def _memory_manager() -> MemoryManager:
    global _mem
    if _mem is None:
        _mem = MemoryManager(_config())
    return _mem


@asynccontextmanager
async def lifespan(app: FastAPI):
    _memory_manager()  # create asyncio.Lock on the running event loop
    start_background_gc()  # hourly sweep of APC shards older than 48h
    yield


app = FastAPI(lifespan=lifespan)


# ── /v1/models ────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models() -> JSONResponse:
    cfg = _config()
    data = [
        {"id": m.name, "object": "model", "owned_by": "mlx-community"}
        for m in cfg.models
    ]
    return JSONResponse({"object": "list", "data": data})


# ── catch-all proxy ───────────────────────────────────────────────────────────

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def proxy(path: str, request: Request) -> Response:
    _touch_keepawake()
    cfg = _config()

    # --- determine which model is requested ---
    model_name: str | None = request.query_params.get("model")
    body_bytes = await request.body()
    body: dict = {}

    if body_bytes:
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            pass
        if not model_name:
            model_name = body.get("model")

    # For multipart/form-data (e.g. audio transcriptions), extract `model` form field
    if not model_name and body_bytes:
        ct = request.headers.get("content-type", "")
        if "multipart/form-data" in ct:
            m = re.search(rb'name="model"\r\n\r\n([^\r\n-]+)', body_bytes)
            if m:
                model_name = m.group(1).decode(errors="replace").strip()

    if not model_name:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Missing `model` field", "type": "invalid_request_error"}},
        )

    model_cfg = cfg.model_by_name(model_name)
    if model_cfg is None:
        available = [m.name for m in cfg.models]
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": f"Unknown model '{model_name}'. Available: {available}",
                    "type": "invalid_request_error",
                }
            },
        )

    # --- on-demand start + LRU eviction ---
    try:
        await _memory_manager().ensure_model_ready(model_name)
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"error": {
                "message": f"Failed to start model '{model_name}': {exc}",
                "type": "service_unavailable",
            }},
        )

    # --- reject image content for text-only models before touching the backend ---
    if body and _has_image_content(body) and not model_cfg.accepts_image_input:
        multimodal_models = [m.name for m in cfg.models if m.accepts_image_input]
        return JSONResponse(
            status_code=400,
            content={"error": {
                "message": (
                    f"Model '{model_name}' is text-only and does not support image content. "
                    f"Use a multimodal model instead: {multimodal_models}"
                ),
                "type": "invalid_request_error",
            }},
        )

    # --- inject default hyperparams (caller values win) + rewrite model alias ---
    if body:
        defaults = model_cfg.defaults.as_request_fields()
        for key, val in defaults.items():
            if key not in body:
                body[key] = val
        # Rewrite the friendly alias to the full source ID the backend expects.
        body["model"] = model_cfg.source

        # Backend-specific request rewriting (thinking-budget handling, etc.).
        model_cfg.resolve_backend().transform_request(body, model_cfg)

        body_bytes = json.dumps(body).encode()

    # --- forward ---
    backend = model_cfg.backend_url()
    url = f"{backend}/{path}"

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }
    if body_bytes:
        headers["content-length"] = str(len(body_bytes))

    is_stream = body.get("stream", False) or request.query_params.get("sse") == "true"
    fwd_params = {k: v for k, v in request.query_params.items() if k != "model"}

    # Mark the model busy for the duration of the actual backend call, so the
    # eviction logic in MemoryManager can wait for it to finish instead of
    # killing it mid-request. Guard against a double decrement across the
    # early-return / exception / streaming-generator exit paths below.
    _memory_manager().mark_busy(model_name)
    marked_idle = False

    def _mark_idle_once() -> None:
        nonlocal marked_idle
        if not marked_idle:
            marked_idle = True
            _memory_manager().mark_idle(model_name)

    if is_stream:
        # Pre-flight: a downed backend must return a real HTTP 503 rather than
        # yielding raw JSON through the SSE stream.  Clients like Pi see a
        # non-SSE body as "Stream ended without finish_reason" otherwise.
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as probe:
                await probe.head(backend)
        except (httpx.ConnectError, httpx.TimeoutException):
            _mark_idle_once()
            return _backend_unavailable(model_name, model_cfg.port)

        # Client must live inside the generator — it would be closed by the time
        # Starlette starts iterating if we opened it in the outer scope.
        async def _stream_gen() -> AsyncGenerator[bytes, None]:
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    try:
                        async with client.stream(
                            request.method, url,
                            content=body_bytes, headers=headers, params=fwd_params,
                        ) as resp:
                            if resp.status_code >= 400:
                                # Backend returned an error (e.g. 404 from mlx_lm.server
                                # when it rejects a request). Emit a proper SSE error so
                                # the client sees it rather than "Stream ended without
                                # finish_reason".
                                raw = await resp.aread()
                                try:
                                    detail = json.loads(raw).get("error", raw.decode())
                                except Exception:
                                    detail = raw.decode()
                                err = json.dumps({"error": {"message": str(detail), "type": "backend_error", "code": resp.status_code}})
                                yield f"data: {err}\n\ndata: [DONE]\n\n".encode()
                                return
                            async for chunk in resp.aiter_bytes():
                                _touch_keepawake()
                                yield chunk
                    except httpx.ConnectError:
                        err = json.dumps({"error": {"message": f"Backend for '{model_name}' died mid-stream", "type": "service_unavailable"}})
                        yield f"data: {err}\n\ndata: [DONE]\n\n".encode()
            finally:
                _mark_idle_once()

        return StreamingResponse(_stream_gen(), media_type="text/event-stream")
    else:
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.request(
                    request.method, url,
                    content=body_bytes, headers=headers, params=fwd_params,
                )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )
        except (httpx.ConnectError, httpx.TimeoutException):
            return _backend_unavailable(model_name, model_cfg.port)
        finally:
            _mark_idle_once()
