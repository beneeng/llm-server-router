"""Shared boilerplate for self-hosted backend wrappers (audio, image, video, ...).

These wrappers run as standalone scripts under *isolated* backend venvs, so this
module must import ONLY the standard library plus fastapi/uvicorn — never
mlx_lm/mlx_vlm or other mlx_server internals (they are not installed in those
venvs). It is importable from a sibling wrapper because Python puts the script's
own directory on sys.path, so `from _wrapper_base import ...` resolves.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable

import uvicorn
from fastapi import FastAPI


def base_parser(description: str) -> argparse.ArgumentParser:
    """ArgumentParser pre-seeded with the args every wrapper takes.

    Wrappers add their own modality-specific args (e.g. --type, --quantize).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=True, help="HF model ID, alias, or local path")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    return parser


def add_health(app: FastAPI, info: Callable[[], dict]) -> None:
    """Register a `GET /` health endpoint returning `info()`."""

    @app.get("/")
    def health() -> dict:
        return info()


def serve(app: FastAPI, host: str, port: int) -> None:
    """Run the app with the standard single-worker uvicorn config."""
    uvicorn.run(app, host=host, port=port, log_level="error", workers=1)
