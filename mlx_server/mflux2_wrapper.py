"""
FLUX.2-klein unified wrapper — text-to-image and image-to-image.
Exposes /v1/images/generations and /v1/images/edits.

Uses Flux2Klein, which supports:
  - txt2img: no images field required
  - img2img: first image in `images` used as reference; `image_strength` (0-1):
    higher = more faithful to original (starts later in the denoising schedule,
    so less noise is added). Lower = more change. With the default 4 steps only
    3 distinct levels exist; use 8–12 steps for finer control.

SSE progress streaming via ?sse=true:
  progress: {"type":"progress","step":N,"total":N,"percent":N,"eta_seconds":N}
  done:     {"type":"done","created":N,"data":[{"b64_json":"...","revised_prompt":"..."}]}
  error:    {"type":"error","message":"generation aborted"}

All mflux2 model entries (flux2-klein-fast, flux2-klein-hifi, etc.)
can be used for both txt2img and img2img without any model-ID change.
"""
from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import io
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from _wrapper_base import add_health, base_parser, serve


_model = None
_model_mode: str = "txt2img"   # "txt2img" | "edit"
_model_source: str = ""
_model_init_kwargs: dict = {}
_mlx_executor: concurrent.futures.ThreadPoolExecutor | None = None


class GenerationAborted(Exception):
    pass


class ImageGenRequest(BaseModel):
    model: str = ""
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    # img2img fields — first image used as reference; image_strength controls blend
    images: Optional[list[str]] = None
    image_strength: Optional[float] = None
    edit_mode: Optional[bool] = None
    # Extended fields merged from config defaults by the router
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None


class ProgressCallback:
    """Registered on _model.callbacks for each request.

    When queue=None (non-SSE path), only checks abort_event each step.
    When queue is provided (SSE path), also pushes progress events to it.
    """

    def __init__(self, queue: asyncio.Queue | None, loop: asyncio.AbstractEventLoop):
        self._queue = queue
        self._loop = loop
        self._step = 0
        self._total: int = 4
        self._t0: float = 0.0
        self.abort_event = threading.Event()

    def _push(self, item) -> None:
        if self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def call_before_loop(self, seed, prompt, latents, config, **kw):
        self._total = config.num_inference_steps
        self._t0 = time.time()

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps, **kw):
        if self.abort_event.is_set():
            raise GenerationAborted()
        self._step += 1
        elapsed = time.time() - self._t0
        per_step = elapsed / self._step
        eta = round(per_step * (self._total - self._step), 1)
        pct = int(self._step / self._total * 100)
        self._push({
            "type": "progress", "step": self._step,
            "total": self._total, "percent": pct, "eta_seconds": eta,
        })

    def call_after_loop(self, seed, prompt, latents, config, **kw):
        self._push(None)  # sentinel — generation done


app = FastAPI()
add_health(app, lambda: {"model": _model_source, "status": "ok"})


def _ensure_mode(mode: str) -> None:
    global _model, _model_mode
    if _model_mode == mode:
        return
    print(f"[mflux2] switching model mode: {_model_mode!r} → {mode!r}", flush=True)
    from mflux.models.common.config.model_config import ModelConfig
    model_config = ModelConfig.from_name(_model_source)
    kwargs = {**_model_init_kwargs, "model_config": model_config}
    if mode == "edit":
        from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
        _model = Flux2KleinEdit(**kwargs)
    else:
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        _model = Flux2Klein(**kwargs)
    _model_mode = mode


def _resolve_dims(request: ImageGenRequest) -> tuple[int, int]:
    if request.width is not None and request.height is not None:
        return request.width, request.height
    try:
        w_str, h_str = request.size.split("x")
        return int(w_str), int(h_str)
    except (ValueError, AttributeError):
        return 1024, 1024


def _write_tmp_files(images: list[str] | None) -> list[str]:
    tmp: list[str] = []
    for b64 in (images or []):
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.write(base64.b64decode(b64))
        f.close()
        tmp.append(f.name)
    return tmp


def _cleanup(tmp_files: list[str]) -> None:
    for p in tmp_files:
        Path(p).unlink(missing_ok=True)


def _reset_callbacks() -> None:
    from mflux.callbacks.callback_registry import CallbackRegistry
    _model.callbacks = CallbackRegistry()


async def _run_edit_generate(request: ImageGenRequest, http_req: Request) -> JSONResponse:
    width, height = _resolve_dims(request)
    steps = request.steps or 4
    guidance = request.guidance_scale if request.guidance_scale is not None else 1.0
    base_seed = request.seed if request.seed is not None else int(time.time())
    n_ref = len(request.images) if request.images else 0
    print(f"edit: prompt={request.prompt!r:.60} ref_images={n_ref}", flush=True)

    tmp_files = _write_tmp_files(request.images)
    loop = asyncio.get_running_loop()
    cb = ProgressCallback(queue=None, loop=loop)

    def _generate():
        _ensure_mode("edit")
        _model.callbacks.register(cb)
        try:
            return _model.generate_image(
                seed=base_seed,
                prompt=request.prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
                image_paths=tmp_files if tmp_files else None,
            )
        except GenerationAborted:
            return None
        finally:
            _reset_callbacks()
            _cleanup(tmp_files)

    async def _watch():
        while True:
            if await http_req.is_disconnected():
                cb.abort_event.set()
                return
            await asyncio.sleep(0.3)

    watcher = asyncio.create_task(_watch())
    try:
        result = await loop.run_in_executor(_mlx_executor, _generate)
    finally:
        watcher.cancel()

    if result is None:
        print("[mflux2] edit generation aborted (client disconnected)", flush=True)
        return JSONResponse({"created": int(time.time()), "data": []})

    buf = io.BytesIO()
    result.image.save(buf, format="PNG")
    b64_out = base64.b64encode(buf.getvalue()).decode()
    return JSONResponse({
        "created": int(time.time()),
        "data": [{"b64_json": b64_out, "revised_prompt": request.prompt}],
    })


async def _run_generate(request: ImageGenRequest, http_req: Request) -> JSONResponse:
    if request.edit_mode and request.images:
        return await _run_edit_generate(request, http_req)

    n_ref = len(request.images) if request.images else 0
    print(f"generate: prompt={request.prompt!r:.60} n={request.n} ref_images={n_ref}", flush=True)

    width, height = _resolve_dims(request)
    steps = request.steps or 4
    guidance = request.guidance_scale if request.guidance_scale is not None else 1.0
    base_seed = request.seed if request.seed is not None else int(time.time())
    tmp_files = _write_tmp_files(request.images)
    image_path = tmp_files[0] if tmp_files else None  # Flux2Klein takes a single path
    loop = asyncio.get_running_loop()

    if request.n > 1:
        # Batch path: generate_images has no step callbacks so cannot abort mid-generation.
        # Run on MLX thread so the event loop stays free; discard result if client disconnected.
        disconnected = False

        def _batch():
            _ensure_mode("txt2img")
            try:
                seeds = [base_seed + i for i in range(request.n)]
                return _model.generate_images(
                    seeds=seeds,
                    prompt=request.prompt,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance,
                    image_path=image_path,
                    image_strength=request.image_strength,
                )
            finally:
                _cleanup(tmp_files)

        async def _watch_batch():
            nonlocal disconnected
            while True:
                if await http_req.is_disconnected():
                    disconnected = True
                    print("[mflux2] client disconnected (batch cannot be aborted, discarding result)", flush=True)
                    return
                await asyncio.sleep(0.3)

        watcher = asyncio.create_task(_watch_batch())
        try:
            results = await loop.run_in_executor(_mlx_executor, _batch)
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

    else:
        cb = ProgressCallback(queue=None, loop=loop)

        def _single():
            _ensure_mode("txt2img")
            _model.callbacks.register(cb)
            try:
                return _model.generate_image(
                    seed=base_seed,
                    prompt=request.prompt,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance,
                    image_path=image_path,
                    image_strength=request.image_strength,
                )
            except GenerationAborted:
                return None
            finally:
                _reset_callbacks()
                _cleanup(tmp_files)

        async def _watch_single():
            while True:
                if await http_req.is_disconnected():
                    cb.abort_event.set()
                    return
                await asyncio.sleep(0.3)

        watcher = asyncio.create_task(_watch_single())
        try:
            result = await loop.run_in_executor(_mlx_executor, _single)
        finally:
            watcher.cancel()

        if result is None:
            print("[mflux2] generation aborted (client disconnected)", flush=True)
            return JSONResponse({"created": int(time.time()), "data": []})

        buf = io.BytesIO()
        result.image.save(buf, format="PNG")
        b64_out = base64.b64encode(buf.getvalue()).decode()
        return JSONResponse({
            "created": int(time.time()),
            "data": [{"b64_json": b64_out, "revised_prompt": request.prompt}],
        })


async def _run_generate_sse(request: ImageGenRequest, http_req: Request) -> StreamingResponse:
    """SSE path: always single-image (generate_images batch has no step callbacks)."""
    use_edit = bool(request.edit_mode and request.images)

    width, height = _resolve_dims(request)
    steps = request.steps or 4
    guidance = request.guidance_scale if request.guidance_scale is not None else 1.0
    base_seed = request.seed if request.seed is not None else int(time.time())
    n_ref = len(request.images) if request.images else 0
    print(f"{'edit' if use_edit else 'generate'}(sse): prompt={request.prompt!r:.60} ref_images={n_ref}", flush=True)

    tmp_files = _write_tmp_files(request.images)
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    cb = ProgressCallback(queue=queue, loop=loop)

    def _generate():
        _ensure_mode("edit" if use_edit else "txt2img")
        _model.callbacks.register(cb)
        try:
            if use_edit:
                return _model.generate_image(
                    seed=base_seed,
                    prompt=request.prompt,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance,
                    image_paths=tmp_files if tmp_files else None,
                )
            else:
                return _model.generate_image(
                    seed=base_seed,
                    prompt=request.prompt,
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance,
                    image_path=tmp_files[0] if tmp_files else None,
                    image_strength=request.image_strength,
                )
        except GenerationAborted:
            return None
        finally:
            _reset_callbacks()
            _cleanup(tmp_files)
            # Push sentinel even on abort so event_stream doesn't hang.
            # (call_after_loop does this for normal completion; this covers abort.)
            loop.call_soon_threadsafe(queue.put_nowait, None)

    gen_task = loop.run_in_executor(_mlx_executor, _generate)

    async def _watch():
        while not gen_task.done():
            if await http_req.is_disconnected():
                cb.abort_event.set()
                return
            await asyncio.sleep(0.3)

    watcher = asyncio.create_task(_watch())

    async def event_stream():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if event is None:  # sentinel
                    break
                yield f"data: {json.dumps(event)}\n\n"

            result = await gen_task
            if result is None:
                print("[mflux2] SSE generation aborted (client disconnected)", flush=True)
                yield f"data: {json.dumps({'type': 'error', 'message': 'generation aborted'})}\n\n"
            else:
                buf = io.BytesIO()
                result.image.save(buf, format="PNG")
                b64_out = base64.b64encode(buf.getvalue()).decode()
                yield f"data: {json.dumps({'type': 'done', 'created': int(time.time()), 'data': [{'b64_json': b64_out, 'revised_prompt': request.prompt}]})}\n\n"
        finally:
            watcher.cancel()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/images/generations")
async def generate_images(request: ImageGenRequest, http_req: Request, sse: bool = False):
    if sse:
        return await _run_generate_sse(request, http_req)
    return await _run_generate(request, http_req)


@app.post("/v1/images/edits")
async def edit_images(request: ImageGenRequest, http_req: Request, sse: bool = False):
    if sse:
        return await _run_generate_sse(request, http_req)
    return await _run_generate(request, http_req)


def main():
    global _model, _model_source, _model_init_kwargs, _mlx_executor

    parser = base_parser("mflux FLUX.2-klein unified image generation server")
    parser.add_argument(
        "--quantize", type=int, default=None, choices=[4, 8],
        help="Weight quantization bits (None = BF16)",
    )
    parser.add_argument(
        "--lora-paths", nargs="+", default=None,
        help="Klein-native LoRA .safetensors paths (loaded at startup)",
    )
    parser.add_argument(
        "--lora-scales", nargs="+", type=float, default=None,
        help="LoRA scale weights (one per --lora-paths entry, default 1.0)",
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Local model directory override",
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
        f"Loading FLUX.2-klein model: {args.model} "
        f"(quantize={args.quantize}bit, {desc})",
        flush=True,
    )

    _model_init_kwargs = {
        "quantize": args.quantize,
        "lora_paths": lora_paths,
        "lora_scales": args.lora_scales,
        "model_path": model_path,
    }

    # Create the dedicated MLX thread and initialise the model on it so the GPU stream
    # is owned by that thread for the lifetime of the process.
    _mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _init_model():
        global _model
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        from mflux.models.common.config.model_config import ModelConfig
        model_config = ModelConfig.from_name(args.model)
        _model = Flux2Klein(
            model_config=model_config,
            model_path=model_path,
            quantize=args.quantize,
            lora_paths=lora_paths,
            lora_scales=args.lora_scales,
        )

    _mlx_executor.submit(_init_model).result()
    print(f"Model loaded. Starting server on {args.host}:{args.port}", flush=True)

    serve(app, args.host, args.port)


if __name__ == "__main__":
    main()
