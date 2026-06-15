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
from typing import AsyncGenerator

import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

def _backend_unavailable(model_name: str, port: int) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"error": {
            "message": f"Model '{model_name}' backend is not running (port {port}). Run: mlx start --model {model_name}",
            "type": "service_unavailable",
        }},
    )

from .config import ServerConfig, load_config
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

    # --- inject default hyperparams (caller values win) + rewrite model alias ---
    if body:
        defaults = model_cfg.defaults.as_request_fields()
        for key, val in defaults.items():
            if key not in body:
                body[key] = val
        # mlx_lm.server ignores the model field for routing (model is fixed at
        # server startup) but uses it for response metadata — pass the source ID.
        body["model"] = model_cfg.source

        # Pi sends enable_thinking as a top-level field (thinkingFormat="qwen"),
        # which mlx_lm.server ignores. For False we must move it into
        # chat_template_kwargs so the tokenizer actually suppresses thinking.
        # For True we leave it top-level — moving it would trigger prefill.
        if body.get("enable_thinking") is False:
            chat_kwargs = body.get("chat_template_kwargs") or {}
            chat_kwargs["enable_thinking"] = False
            body["chat_template_kwargs"] = chat_kwargs
            del body["enable_thinking"]

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

    is_stream = body.get("stream", False)
    fwd_params = {k: v for k, v in request.query_params.items() if k != "model"}

    if is_stream:
        # Pre-flight: a downed backend must return a real HTTP 503 rather than
        # yielding raw JSON through the SSE stream.  Clients like Pi see a
        # non-SSE body as "Stream ended without finish_reason" otherwise.
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as probe:
                await probe.head(backend)
        except (httpx.ConnectError, httpx.TimeoutException):
            return _backend_unavailable(model_name, model_cfg.port)

        # Client must live inside the generator — it would be closed by the time
        # Starlette starts iterating if we opened it in the outer scope.
        async def _stream_gen() -> AsyncGenerator[bytes, None]:
            async with httpx.AsyncClient(timeout=None) as client:
                try:
                    async with client.stream(
                        request.method, url,
                        content=body_bytes, headers=headers, params=fwd_params,
                    ) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                except httpx.ConnectError:
                    err = json.dumps({"error": {"message": f"Backend for '{model_name}' died mid-stream", "type": "service_unavailable"}})
                    yield f"data: {err}\n\ndata: [DONE]\n\n".encode()

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
