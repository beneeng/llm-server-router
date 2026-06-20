"""
Thin FastAPI wrapper around mflux that exposes /v1/images/generations
in OpenAI-compatible format.

The model is loaded at startup so the port becomes available only after
weights are loaded (~5-30 s depending on quantization). The MemoryManager's
120 s poll window accommodates this.

Usage in models.yaml:
  - name: flux-schnell
    source: schnell            # mflux alias: "schnell" or "dev"
    backend: mflux
    memory_gb: 12
    lazy: true
    extra_args: ["--quantize", "8"]
    defaults:
      steps: 4
      width: 1024
      height: 1024
"""
from __future__ import annotations

import base64
import io
import time
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from _wrapper_base import add_health, base_parser, serve


_flux = None
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
    from mflux import Config

    # width/height from request body (set by router from config defaults),
    # falling back to parsing the size string, then 1024x1024.
    width = request.width
    height = request.height
    if width is None or height is None:
        try:
            w_str, h_str = request.size.split("x")
            width, height = int(w_str), int(h_str)
        except (ValueError, AttributeError):
            width, height = 1024, 1024

    steps = request.steps or 4
    # guidance is only meaningful for FLUX-dev; schnell ignores it
    guidance = request.guidance_scale or 3.5

    images_data = []
    base_seed = request.seed if request.seed is not None else int(time.time())
    for i in range(max(1, request.n)):
        result = _flux.generate_image(
            seed=base_seed + i,
            prompt=request.prompt,
            config=Config(
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=guidance,
            ),
        )
        buf = io.BytesIO()
        result.image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        images_data.append({"b64_json": b64, "revised_prompt": request.prompt})

    return JSONResponse({"created": int(time.time()), "data": images_data})


def main():
    global _flux, _model_source

    parser = base_parser("mflux image generation server")
    parser.add_argument("--quantize", type=int, default=8, choices=[4, 8], help="Weight quantization bits")
    args = parser.parse_args()

    _model_source = args.model

    print(f"Loading image model: {args.model} (quantize={args.quantize}bit)", flush=True)
    from mflux import Flux1
    _flux = Flux1.from_alias(alias=args.model, quantize=args.quantize)
    print(f"Model loaded. Starting server on {args.host}:{args.port}", flush=True)

    serve(app, args.host, args.port)


if __name__ == "__main__":
    main()
