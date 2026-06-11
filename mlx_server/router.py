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

# Gemma4 special token delimiters (single vocab tokens in the Gemma4 tokenizer)
_G4_QUOTE = '<|"|>'


def _prerender_gemma4_tool_history(messages: list[dict]) -> list[dict]:
    """Convert tool-call history to pre-rendered Gemma4 token format.

    vllm-mlx's MLLM.stream_chat silently drops assistant messages that have
    content=null (even when tool_calls is present), so the Gemma4 chat template
    never gets a chance to forward-scan for the tool responses.  By embedding
    the tool call + response tokens directly into the assistant content string
    we bypass the bug: the MLLM path sees non-empty content and includes the
    message, and the tokenizer encodes <|tool_call> / <|tool_response> / <|"|>
    as the correct special-token IDs.

    Tool messages that are consumed by this pass are removed from the output.
    """
    result: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            parts: list[str] = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "")
                # Embed arguments: JSON string goes in verbatim; dict is
                # formatted as Gemma4 key:<|"|>val<|"|> pairs.
                if isinstance(args, str):
                    args_inner = args
                else:
                    kv: list[str] = []
                    for k, v in (args or {}).items():
                        kv.append(f"{k}:{_G4_QUOTE}{v}{_G4_QUOTE}" if isinstance(v, str)
                                  else f"{k}:{json.dumps(v, ensure_ascii=False)}")
                    args_inner = ",".join(kv)
                parts.append(f"<|tool_call>call:{name}{{{args_inner}}}<tool_call|>")

                # Forward-scan: find the matching tool response
                tc_id = tc.get("id")
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    tool_msg = messages[j]
                    if not tc_id or tool_msg.get("tool_call_id") == tc_id:
                        response_text = tool_msg.get("content") or ""
                        parts.append(
                            f"<|tool_response>response:{name}"
                            f"{{value:{_G4_QUOTE}{response_text}{_G4_QUOTE}}}"
                            f"<tool_response|>"
                        )
                        break
                    j += 1

            result.append({"role": "assistant", "content": "".join(parts)})
            # Consume all following tool messages for this turn
            i += 1
            while i < len(messages) and messages[i].get("role") == "tool":
                i += 1
        else:
            result.append(msg)
            i += 1
    return result

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

        if "messages" in body:
            if model_cfg.multimodal:
                # vllm-mlx's MLLM.stream_chat drops assistant messages with
                # content=null even when tool_calls is present, so the Gemma4
                # chat template never sees the tool history and the model loops.
                # Pre-render tool call + response tokens into the content string
                # so the MLLM path includes the message and the tokenizer encodes
                # the special tokens correctly.
                body["messages"] = _prerender_gemma4_tool_history(body["messages"])
            else:
                # For LLM (non-multimodal) models: strip content from assistant
                # messages that also carry tool_calls. Some clients store both
                # the raw streamed text and tool_calls on the same entry; the
                # chat template would then emit the tool call twice.
                for msg in body["messages"]:
                    if (
                        msg.get("role") == "assistant"
                        and msg.get("tool_calls")
                        and msg.get("content") is not None
                    ):
                        msg["content"] = None

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
        except httpx.ConnectError:
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
