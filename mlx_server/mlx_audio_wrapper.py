"""
Thin wrapper that pre-loads a single mlx_audio model (TTS or STT) and exposes
OpenAI-compatible /v1/audio/speech and /v1/audio/transcriptions endpoints.

Deliberately does NOT import mlx_audio.server (which requires webrtcvad for its
WebSocket realtime endpoint). Only mlx_audio.utils and mlx_audio.audio_io are used.

Auto-detects model type: TTS models expose a `sample_rate` attribute; STT models do not.
Override with --type tts|stt if auto-detection is wrong.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, Response, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from _wrapper_base import add_health, base_parser, serve


_model = None
_model_source: str = ""
_is_tts: bool = True


class SpeechRequest(BaseModel):
    model: str = ""
    input: str
    voice: Optional[str] = None
    speed: Optional[float] = 1.0
    response_format: Optional[str] = "mp3"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    lang_code: Optional[str] = None
    instruct: Optional[str] = None


app = FastAPI()
add_health(app, lambda: {"model": _model_source, "status": "ok", "type": "tts" if _is_tts else "stt"})


@app.post("/v1/audio/speech")
def tts_speech(request: SpeechRequest):
    if not _is_tts:
        return JSONResponse(
            status_code=400,
            content={"error": f"Model {_model_source!r} is an STT model; use /v1/audio/transcriptions"},
        )

    generate_kwargs: dict = {}
    for key, val in {
        "voice": request.voice,
        "speed": request.speed,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "lang_code": request.lang_code,
        "instruct": request.instruct,
    }.items():
        if val is not None:
            generate_kwargs[key] = val

    audio_chunks = []
    sample_rate = None
    for result in _model.generate(request.input, **generate_kwargs):
        audio_chunks.append(result.audio)
        if sample_rate is None:
            sample_rate = result.sample_rate

    if not audio_chunks:
        return JSONResponse(status_code=400, content={"error": "No audio generated"})

    from mlx_audio.audio_io import write as audio_write
    buf = io.BytesIO()
    fmt = request.response_format or "mp3"
    audio_write(buf, np.concatenate(audio_chunks), sample_rate, format=fmt)

    media_type = "audio/mpeg" if fmt == "mp3" else f"audio/{fmt}"
    return Response(content=buf.getvalue(), media_type=media_type)


@app.post("/v1/audio/transcriptions")
async def stt_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    response_format: str = Form("json"),
):
    if _is_tts:
        return JSONResponse(
            status_code=400,
            content={"error": f"Model {_model_source!r} is a TTS model; use /v1/audio/speech"},
        )

    data = await file.read()
    suffix = Path(file.filename or "audio.mp3").suffix or ".mp3"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = _model.generate(tmp_path, language=language, verbose=False)
        # Whisper returns an object with .text; streaming models return a generator
        if hasattr(result, "__next__") or hasattr(result, "__iter__") and not hasattr(result, "text"):
            text = "".join(
                (chunk if isinstance(chunk, str) else getattr(chunk, "text", ""))
                for chunk in result
            )
        else:
            text = result.text if hasattr(result, "text") else str(result)
    finally:
        os.unlink(tmp_path)

    text = text.strip()
    if response_format == "text":
        return Response(content=text, media_type="text/plain")
    return JSONResponse({"text": text})


def main():
    global _model, _model_source, _is_tts

    parser = base_parser("MLX Audio wrapper server")
    parser.add_argument(
        "--type", choices=["tts", "stt"], default=None,
        help="Force model type (auto-detected via sample_rate attribute if omitted)",
    )
    args = parser.parse_args()

    _model_source = args.model

    from mlx_audio.utils import load_model
    print(f"Loading audio model: {args.model}", flush=True)
    _model = load_model(args.model)

    if args.type:
        _is_tts = args.type == "tts"
    else:
        _is_tts = hasattr(_model, "sample_rate")

    print(
        f"Model loaded (type={'tts' if _is_tts else 'stt'}). "
        f"Starting server on {args.host}:{args.port}",
        flush=True,
    )
    serve(app, args.host, args.port)


if __name__ == "__main__":
    main()
