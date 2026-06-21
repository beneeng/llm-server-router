"""
FLUX.2-klein unified wrapper — text-to-image and image-to-image.
Exposes /v1/images/generations and /v1/images/edits.

Uses Flux2Klein, which supports:
  - txt2img: no images field required
  - img2img: first image in `images` used as reference; `image_strength` (0-1):
    higher = more faithful to original (starts later in the denoising schedule,
    so less noise is added). Lower = more change. With the default 4 steps only
    3 distinct levels exist; use 8–12 steps for finer control.

All mflux2 model entries (flux2-klein-fast, flux2-klein-hifi, etc.)
can be used for both txt2img and img2img without any model-ID change.
"""
from __future__ import annotations

import base64
import io
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from _wrapper_base import add_health, base_parser, serve


_model = None
_model_mode: str = "txt2img"   # "txt2img" | "edit"
_model_source: str = ""
_model_init_kwargs: dict = {}


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


async def _run_edit_generate(request: ImageGenRequest) -> JSONResponse:
    _ensure_mode("edit")

    width = request.width
    height = request.height
    if width is None or height is None:
        try:
            w_str, h_str = request.size.split("x")
            width, height = int(w_str), int(h_str)
        except (ValueError, AttributeError):
            width, height = 1024, 1024

    steps = request.steps or 4
    guidance = request.guidance_scale if request.guidance_scale is not None else 1.0
    base_seed = request.seed if request.seed is not None else int(time.time())
    n_ref = len(request.images) if request.images else 0
    print(
        f"edit: prompt={request.prompt!r:.60} n={request.n} ref_images={n_ref}",
        flush=True,
    )

    tmp_files: list[str] = []
    try:
        for b64 in (request.images or []):
            f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            f.write(base64.b64decode(b64))
            f.close()
            tmp_files.append(f.name)

        images_data = []
        for i in range(max(1, request.n)):
            result = _model.generate_image(
                seed=base_seed + i,
                prompt=request.prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
                image_paths=tmp_files if tmp_files else None,
            )
            buf = io.BytesIO()
            result.image.save(buf, format="PNG")
            b64_out = base64.b64encode(buf.getvalue()).decode()
            images_data.append({"b64_json": b64_out, "revised_prompt": request.prompt})

    finally:
        for p in tmp_files:
            Path(p).unlink(missing_ok=True)

    return JSONResponse({"created": int(time.time()), "data": images_data})


async def _run_generate(request: ImageGenRequest) -> JSONResponse:
    if request.edit_mode and request.images:
        return await _run_edit_generate(request)
    _ensure_mode("txt2img")
    n_ref = len(request.images) if request.images else 0
    print(
        f"generate: prompt={request.prompt!r:.60} n={request.n} ref_images={n_ref}",
        flush=True,
    )

    width = request.width
    height = request.height
    if width is None or height is None:
        try:
            w_str, h_str = request.size.split("x")
            width, height = int(w_str), int(h_str)
        except (ValueError, AttributeError):
            width, height = 1024, 1024

    steps = request.steps or 4
    guidance = request.guidance_scale if request.guidance_scale is not None else 1.0

    tmp_files: list[str] = []
    try:
        if request.images:
            for b64 in request.images:
                f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                f.write(base64.b64decode(b64))
                f.close()
                tmp_files.append(f.name)

        # Flux2Klein takes a single image_path (uses first reference image)
        image_path = tmp_files[0] if tmp_files else None
        images_data = []
        base_seed = request.seed if request.seed is not None else int(time.time())

        if request.n > 1:
            # Batch path: text encoder runs once, one transformer pass per denoising step for all N
            seeds = [base_seed + i for i in range(request.n)]
            results = _model.generate_images(
                seeds=seeds,
                prompt=request.prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
                image_path=image_path,
                image_strength=request.image_strength,
            )
            for result in results:
                buf = io.BytesIO()
                result.image.save(buf, format="PNG")
                b64_out = base64.b64encode(buf.getvalue()).decode()
                images_data.append({"b64_json": b64_out, "revised_prompt": request.prompt})
        else:
            result = _model.generate_image(
                seed=base_seed,
                prompt=request.prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
                image_path=image_path,
                image_strength=request.image_strength,
            )
            buf = io.BytesIO()
            result.image.save(buf, format="PNG")
            b64_out = base64.b64encode(buf.getvalue()).decode()
            images_data.append({"b64_json": b64_out, "revised_prompt": request.prompt})

    finally:
        for p in tmp_files:
            Path(p).unlink(missing_ok=True)

    return JSONResponse({"created": int(time.time()), "data": images_data})


@app.post("/v1/images/generations")
async def generate_images(request: ImageGenRequest):
    return await _run_generate(request)


@app.post("/v1/images/edits")
async def edit_images(request: ImageGenRequest):
    return await _run_generate(request)


def main():
    global _model, _model_source, _model_init_kwargs

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
    _model_init_kwargs = {
        "quantize": args.quantize,
        "lora_paths": lora_paths,
        "lora_scales": args.lora_scales,
        "model_path": model_path,
    }
    print(f"Model loaded. Starting server on {args.host}:{args.port}", flush=True)

    serve(app, args.host, args.port)


if __name__ == "__main__":
    main()
