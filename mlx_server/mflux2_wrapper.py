"""
Thin FastAPI wrapper around mflux's Flux2Klein for FLUX.2-klein models.
Exposes /v1/images/generations in OpenAI-compatible format.

Key differences from mflux_wrapper.py (FLUX.1):
  - Uses Flux2Klein (not Flux1.from_alias) — different mflux class hierarchy
  - Default guidance=1.0 (distilled models; do NOT pass 3.5)
  - Default steps=4
  - Supports --lora-paths / --lora-scales (loaded at startup, Klein-native LoRAs only)
  - Supports --model-path for loading a local model dir (e.g. merged uncensored encoder)

Usage in models.yaml:
  - name: flux2-klein-fast
    source: flux2-klein-4b
    backend: mflux2
    memory_gb: 10
    lazy: true
    extra_args: ["--quantize", "8"]
    defaults:
      steps: 4
      width: 1024
      height: 1024

  - name: flux2-klein-hifi
    source: flux2-klein-9b
    backend: mflux2
    memory_gb: 22
    lazy: true
    extra_args:
      - "--quantize"
      - "8"
      - "--lora-paths"
      - "~/.cache/mflux/loras/ultra-real-klein-v4.safetensors"
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
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from _wrapper_base import add_health, base_parser, serve


_model = None
_model_source: str = ""


class ImageGenRequest(BaseModel):
    model: str = ""
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    # Extended fields (merged from config defaults by the router)
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None


app = FastAPI()
add_health(app, lambda: {"model": _model_source, "status": "ok"})


@app.post("/v1/images/generations")
async def generate_images(request: ImageGenRequest):
    print(f"generate: prompt={request.prompt!r:.60} n={request.n}", flush=True)
    width = request.width
    height = request.height
    if width is None or height is None:
        try:
            w_str, h_str = request.size.split("x")
            width, height = int(w_str), int(h_str)
        except (ValueError, AttributeError):
            width, height = 1024, 1024

    steps = request.steps or 4
    # FLUX.2-klein (distilled) uses guidance=1.0; only the base variants use CFG > 1.
    guidance = request.guidance_scale if request.guidance_scale is not None else 1.0

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
        )
        buf = io.BytesIO()
        result.image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        images_data.append({"b64_json": b64, "revised_prompt": request.prompt})

    return JSONResponse({"created": int(time.time()), "data": images_data})


def main():
    global _model, _model_source

    parser = base_parser("mflux FLUX.2-klein image generation server")
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

    # Expand ~ in LoRA paths so shell-unescaped paths work from models.yaml extra_args
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
    print(f"Loading FLUX.2-klein model: {args.model} (quantize={args.quantize}bit, {desc})", flush=True)

    from mflux.models.flux2.variants import Flux2Klein
    from mflux.models.common.config.model_config import ModelConfig

    model_config = ModelConfig.from_name(args.model)

    _model = Flux2Klein(
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
