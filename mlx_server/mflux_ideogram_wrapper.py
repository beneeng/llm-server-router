"""
Ideogram 4 image generation wrapper.
Exposes /v1/images/generations (txt2img only — no img2img support in Ideogram 4).

Presets map to mflux's Ideogram4Scheduler:
  V4_TURBO_12  — 12 steps, variable guidance schedule (fastest)
  V4_DEFAULT_20 — 20 steps, variable guidance schedule (default)
  V4_QUALITY_48 — 48 steps, variable guidance schedule (highest quality)

When steps matches a known preset (12, 20, 48), the full preset schedule is used.
Any other step count passes num_inference_steps directly (flat guidance=7.0).
Explicit preset= in the request always wins.

SSE streaming via ?sse=true: emits only done/error events (no per-step progress —
Ideogram4 has no CallbackRegistry). Useful for async response delivery without polling.

Batch (n>1): sequential calls to generate_image (no tensor-batched batch API).
"""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import io
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import json

from _wrapper_base import add_health, base_parser, serve


_model = None
_model_source: str = ""
_model_init_kwargs: dict = {}
_mlx_executor: concurrent.futures.ThreadPoolExecutor | None = None

STEP_TO_PRESET: dict[int, str] = {
    12: "V4_TURBO_12",
    20: "V4_DEFAULT_20",
    48: "V4_QUALITY_48",
}


class ImageGenRequest(BaseModel):
    model: str = ""
    prompt: str
    negative_prompt: Optional[str] = None  # accepted for API compatibility, ignored
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    preset: Optional[str] = None  # V4_TURBO_12 | V4_DEFAULT_20 | V4_QUALITY_48


def _resolve_preset_and_steps(req: ImageGenRequest) -> tuple[str | None, int | None]:
    """Return (preset, num_inference_steps) to pass to generate_image.

    preset=X, num_inference_steps=None → mflux uses the preset's variable guidance schedule.
    preset=None, num_inference_steps=N → flat guidance=7.0 for N steps.
    preset=None, num_inference_steps=None → mflux defaults to V4_DEFAULT_20.
    """
    if req.preset:
        return req.preset, None
    if req.steps in STEP_TO_PRESET:
        return STEP_TO_PRESET[req.steps], None
    if req.steps:
        return None, req.steps
    return None, None


def _resolve_dims(req: ImageGenRequest) -> tuple[int, int]:
    if req.width is not None and req.height is not None:
        return req.width, req.height
    try:
        w_str, h_str = req.size.split("x")
        return int(w_str), int(h_str)
    except (ValueError, AttributeError):
        return 1024, 1024


app = FastAPI()
add_health(app, lambda: {"model": _model_source, "status": "ok"})


async def _run_generate(request: ImageGenRequest, http_req: Request) -> JSONResponse:
    width, height = _resolve_dims(request)
    preset, num_steps = _resolve_preset_and_steps(request)
    guidance = request.guidance_scale
    base_seed = request.seed if request.seed is not None else int(time.time())

    if request.negative_prompt:
        print(
            f"[ideogram4] negative_prompt ignored (not supported by Ideogram4 API): {request.negative_prompt!r:.60}",
            flush=True,
        )

    print(
        f"generate: prompt={request.prompt!r:.80}"
        f" n={request.n} preset={preset!r} steps={num_steps}"
        f" guidance={guidance} size={width}x{height} seed={base_seed}",
        flush=True,
    )

    loop = asyncio.get_running_loop()
    disconnected = False

    def _generate_all():
        results = []
        for i in range(request.n):
            seed = base_seed + i
            img = _model.generate_image(
                seed=seed,
                prompt=request.prompt,
                num_inference_steps=num_steps,
                height=height,
                width=width,
                guidance=guidance,
                preset=preset,
            )
            results.append(img)
        return results

    async def _watch():
        nonlocal disconnected
        while True:
            if await http_req.is_disconnected():
                disconnected = True
                print("[ideogram4] client disconnected (generation cannot be aborted)", flush=True)
                return
            await asyncio.sleep(0.3)

    watcher = asyncio.create_task(_watch())
    try:
        results = await loop.run_in_executor(_mlx_executor, _generate_all)
    finally:
        watcher.cancel()

    if disconnected:
        return JSONResponse({"created": int(time.time()), "data": []})

    images_data = []
    for result in results:
        buf = io.BytesIO()
        result.image.save(buf, format="PNG")
        b64_out = base64.b64encode(buf.getvalue()).decode()
        images_data.append({"b64_json": b64_out, "revised_prompt": request.prompt})
    return JSONResponse({"created": int(time.time()), "data": images_data})


async def _run_generate_sse(request: ImageGenRequest, http_req: Request) -> StreamingResponse:
    """SSE path — emits done/error only (no per-step progress; Ideogram4 has no callbacks)."""
    width, height = _resolve_dims(request)
    preset, num_steps = _resolve_preset_and_steps(request)
    guidance = request.guidance_scale
    base_seed = request.seed if request.seed is not None else int(time.time())

    print(
        f"generate(sse): prompt={request.prompt!r:.60}"
        f" n={request.n} preset={preset!r} steps={num_steps}",
        flush=True,
    )

    loop = asyncio.get_running_loop()

    def _generate_all():
        results = []
        for i in range(request.n):
            seed = base_seed + i
            img = _model.generate_image(
                seed=seed,
                prompt=request.prompt,
                num_inference_steps=num_steps,
                height=height,
                width=width,
                guidance=guidance,
                preset=preset,
            )
            results.append(img)
        return results

    gen_task = loop.run_in_executor(_mlx_executor, _generate_all)

    async def _watch():
        while not gen_task.done():
            if await http_req.is_disconnected():
                print("[ideogram4] client disconnected during SSE (cannot abort)", flush=True)
                return
            await asyncio.sleep(0.3)

    watcher = asyncio.create_task(_watch())

    async def event_stream():
        try:
            try:
                results = await gen_task
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                return

            images_data = []
            for result in results:
                buf = io.BytesIO()
                result.image.save(buf, format="PNG")
                b64_out = base64.b64encode(buf.getvalue()).decode()
                images_data.append({"b64_json": b64_out, "revised_prompt": request.prompt})
            yield f"data: {json.dumps({'type': 'done', 'created': int(time.time()), 'data': images_data})}\n\n"
        finally:
            watcher.cancel()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/images/generations")
async def generate_images(request: ImageGenRequest, http_req: Request, sse: bool = False):
    if sse:
        return await _run_generate_sse(request, http_req)
    return await _run_generate(request, http_req)


def main():
    global _model, _model_source, _model_init_kwargs, _mlx_executor

    parser = base_parser("mflux Ideogram 4 image generation server")
    parser.add_argument(
        "--quantize", type=int, default=None, choices=[4, 8],
        help="Weight quantization bits (None = fp8/BF16 as stored in weights)",
    )
    parser.add_argument(
        "--lora-paths", nargs="+", default=None,
        help="LoRA .safetensors paths (loaded at startup)",
    )
    parser.add_argument(
        "--lora-scales", nargs="+", type=float, default=None,
        help="LoRA scale weights (one per --lora-paths entry, default 1.0)",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Local model directory override (use with pre-quantized community models)",
    )
    args = parser.parse_args()

    _model_source = args.model

    lora_paths = None
    if args.lora_paths:
        lora_paths = [str(Path(p).expanduser()) for p in args.lora_paths]

    model_path = str(Path(args.model_path).expanduser()) if args.model_path else None

    extra = []
    if lora_paths:
        extra.append(f"lora={lora_paths}")
    if model_path:
        extra.append(f"model_path={model_path}")
    desc = ", ".join(extra) if extra else "no extras"
    print(
        f"Loading Ideogram 4 model: {args.model} "
        f"(quantize={args.quantize}, {desc})",
        flush=True,
    )

    _mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _init_model():
        global _model
        from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4
        from mflux.models.common.config.model_config import ModelConfig
        model_config = ModelConfig.from_name(args.model)
        _model = Ideogram4(
            quantize=args.quantize,
            model_path=model_path,
            model_config=model_config,
            lora_paths=lora_paths,
            lora_scales=args.lora_scales,
        )

    _mlx_executor.submit(_init_model).result()

    print(f"Model loaded. Starting server on {args.host}:{args.port}", flush=True)
    serve(app, args.host, args.port)


if __name__ == "__main__":
    main()
