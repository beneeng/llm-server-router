"""
FastAPI reverse proxy on port 6300.

Routing logic:
  1. Extract `model` from the JSON body (POST) or `model` query param (GET).
  2. Look up the ModelConfig; inject its `defaults` into the body for any key
     the caller did not supply (caller values always win).
  3. Forward the (possibly enriched) request to the model's vllm-mlx backend.
  4. Stream the response back verbatim.

/v1/models is handled specially: it aggregates the model list from the YAML
config rather than forwarding, so it works even when backends are offline.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator

import httpx
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

app = FastAPI()

# Loaded once at startup; re-read the file on SIGHUP if you want hot reload.
_cfg: ServerConfig | None = None


def _config() -> ServerConfig:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


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

    # --- inject default hyperparams (caller values win) + rewrite model alias ---
    if body:
        defaults = model_cfg.defaults.as_request_fields()
        for key, val in defaults.items():
            if key not in body:
                body[key] = val
        # vllm-mlx expects the HuggingFace model ID, not our friendly alias
        body["model"] = model_cfg.source
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
                    yield _backend_unavailable(model_name, model_cfg.port).body

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
        except httpx.ConnectError:
            return _backend_unavailable(model_name, model_cfg.port)
