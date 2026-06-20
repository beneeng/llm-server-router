"""
FastAPI wrapper around mflux's Flux2KleinEdit for image-to-image and multi-image generation.
Exposes /v1/images/generations and /v1/images/edits in OpenAI-compatible format.

Differences from mflux2_wrapper.py (text-to-image):
  - Uses Flux2KleinEdit (not Flux2Klein) — same 4B/9B weights, different inference path
  - Accepts `images` (list of base64 strings) as reference images for conditioning
  - `image_strength` controls denoising (0.0 = full regen, 1.0 = stay close to reference)
  - `use_kv_cache` caches reference image KV states — speeds up n>1 batch with same reference
  - Works without images too (falls back to text-to-image behaviour)

Usage in models.yaml:
  - name: flux2-klein-edit-fast
    source: flux2-klein-4b
    backend: mflux2-edit
    memory_gb: 10
    lazy: true
    extra_args: ["--quantize", "8"]
    defaults:
      steps: 4
      width: 1024
      height: 1024

  - name: flux2-klein-edit-hifi
    source: flux2-klein-9b
    backend: mflux2-edit
    memory_gb: 22
    lazy: true
    extra_args:
      - "--quantize"
      - "8"
      - "--lora-paths"
      - "~/.local/share/mlx/loras/ultra-real-klein-v4.safetensors"
      - "--lora-scales"
      - "0.8"
    defaults:
      steps: 4
      width: 1024
      height: 1024
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
_model_source: str = ""


class ImageEditRequest(BaseModel):
    model: str = ""
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    # Reference images: base64-encoded PNG/JPEG strings
    images: Optional[list[str]] = None
    # Denoising strength (0.0 = fully regenerate, 1.0 = stay close to reference).
    # None = full conditioning without partial denoising.
    image_strength: Optional[float] = None
    # None defers to model_config.supports_kv_cache. True speeds up n>1 with same reference.
    use_kv_cache: Optional[bool] = None
    # Extended fields merged from config defaults by the router
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None


app = FastAPI()
add_health(app, lambda: {"model": _model_source, "status": "ok"})


async def _run_generate(request: ImageEditRequest) -> JSONResponse:
    n_images = len(request.images) if request.images else 0
    print(
        f"generate: prompt={request.prompt!r:.60} n={request.n} ref_images={n_images}",
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

        image_paths = tmp_files or None
        images_data = []
        base_seed = request.seed if request.seed is not None else int(time.time())

        for i in range(max(1, request.n)):
            result = _model.generate_image(
                seed=base_seed + i,
                prompt=request.prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
                image_paths=image_paths,
                image_strength=request.image_strength,
                use_kv_cache=request.use_kv_cache,
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
async def generate_images(request: ImageEditRequest):
    return await _run_generate(request)


@app.post("/v1/images/edits")
async def edit_images(request: ImageEditRequest):
    return await _run_generate(request)


def main():
    global _model, _model_source

    parser = base_parser("mflux FLUX.2-klein image editing server")
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
        help="Local model directory override — use for merged uncensored text encoder snapshots",
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
    print(f"Loading FLUX.2-klein-edit model: {args.model} (quantize={args.quantize}bit, {desc})", flush=True)

    from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
    from mflux.models.common.config.model_config import ModelConfig

    model_config = ModelConfig.from_name(args.model)

    _model = Flux2KleinEdit(
        model_config=model_config,
        model_path=model_path,
        quantize=args.quantize,
        lora_paths=lora_paths,
        lora_scales=args.lora_scales,
    )
    print(f"Model loaded. Starting server on {args.host}:{args.port}", flush=True)

    serve(app, args.host, args.port)


if __name__ == "__main__":
    main()
