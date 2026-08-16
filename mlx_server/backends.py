"""Backend registry — the single source of truth for serving engines.

A *backend* bundles everything that makes one serving engine distinct:
  - the Python environment it runs in (`env`: the main tool venv, or an isolated
    venv whose name doubles as the `mlx install <env>` target)
  - the pip packages that env needs (`packages`)
  - the wrapper script launched as a subprocess (`wrapper`)
  - what it can do (`modalities`, `accepts_image_input`)
  - how its subprocess argv is built (`build_args` / `extra_flags`)
  - any router-side request rewriting it requires (`transform_request`)

Models reference a backend by name (`backend: mlx-vlm` in models.yaml). Adding a
new modality is a new `Backend` subclass + a wrapper file — no central dispatch
to edit. `config.py`, `manager.py`, `cli.py`, and `router.py` all read from
`BACKENDS` rather than branching on modality themselves.

This module imports `_bin`/`_backend_python` from `manager` lazily (inside
`python()`) to avoid an import cycle (config → backends → manager → config).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ModelConfig


class Backend:
    """Base serving engine. Subclasses set the class attrs and override hooks."""

    name: str = ""
    # "main" → the tool venv (installed by `uv tool install`). Otherwise the name
    # of an isolated venv under ~/.local/share/mlx/venvs/<env>/, created on demand
    # or via `mlx install <env>`. The same string is the `mlx install` target.
    env: str = "main"
    # pip packages installed into the isolated env. Empty when env == "main".
    packages: tuple[str, ...] = ()
    # Wrapper script filename, resolved relative to this package directory.
    wrapper: str = ""
    # Output modalities this engine produces.
    modalities: frozenset[str] = frozenset()
    # True if the engine accepts image_url content blocks as input.
    accepts_image_input: bool = False

    def python(self) -> str:
        """The interpreter that runs this backend's wrapper."""
        from .manager import _backend_python, _bin

        if self.env == "main":
            return _bin("python3")
        return _backend_python(self.env)

    def extra_flags(self, model: "ModelConfig") -> list[str]:
        """Per-model CLI flags appended after --host (before model.extra_args)."""
        return []

    def env_vars(self, model: "ModelConfig") -> dict[str, str]:
        """Per-model env vars merged into the subprocess environment."""
        return {}

    def build_args(self, model: "ModelConfig", port: int) -> list[str]:
        wrapper = str(Path(__file__).parent / self.wrapper)
        return [
            self.python(), wrapper,
            "--model", model.source,
            "--port", str(port),
            "--host", "127.0.0.1",
            *self.extra_flags(model),
            *model.extra_args,
        ]

    def transform_request(self, body: dict, model: "ModelConfig") -> None:
        """Rewrite the request body in place for backend-specific quirks.

        Default: no-op. Generic default-injection and alias rewriting happen in
        the router before this is called.
        """
        return None

    def primary_modality(self) -> str:
        """A stable single label for display (e.g. the `mlx models` table)."""
        return sorted(self.modalities)[0] if self.modalities else "text"


class MlxLm(Backend):
    """Text-only LLMs via mlx_lm.server (+ ThinkingBudgetProcessor patch)."""

    name = "mlx-lm"
    env = "main"
    wrapper = "mlx_lm_wrapper.py"
    modalities = frozenset({"text"})
    accepts_image_input = False

    def extra_flags(self, model: "ModelConfig") -> list[str]:
        return ["--trust-remote-code"]

    def transform_request(self, body: dict, model: "ModelConfig") -> None:
        # Pi sends enable_thinking as a top-level field (thinkingFormat="qwen"),
        # which mlx_lm.server ignores. For False we must move it into
        # chat_template_kwargs so the tokenizer actually suppresses thinking.
        # For True we leave it top-level — moving it would trigger prefill.
        if body.get("enable_thinking") is False:
            chat_kwargs = body.get("chat_template_kwargs") or {}
            chat_kwargs["enable_thinking"] = False
            body["chat_template_kwargs"] = chat_kwargs
            del body["enable_thinking"]


class MlxVlm(Backend):
    """Vision-language / thinking text models via mlx_vlm.server."""

    name = "mlx-vlm"
    env = "main"
    wrapper = "mlx_vlm_wrapper.py"
    modalities = frozenset({"text"})
    accepts_image_input = True

    def extra_flags(self, model: "ModelConfig") -> list[str]:
        flags = ["--trust-remote-code"]
        if model.thinking:
            flags.append("--enable-thinking")
        return flags

    def env_vars(self, model: "ModelConfig") -> dict[str, str]:
        # Automatic Prefix Caching (mlx-vlm >=0.5): hash-based, block-level KV
        # reuse across requests with LRU+refcount eviction. APC_DISK_PATH adds
        # an SSD tier so a warm cache survives this model's process being
        # stopped/evicted, not just concurrent requests within one run.
        from .manager import STATE_DIR

        return {
            "APC_ENABLED": "1",
            "APC_DISK_PATH": str(STATE_DIR / "apc" / model.name),
        }

    def transform_request(self, body: dict, model: "ModelConfig") -> None:
        # mlx_vlm.server understands enable_thinking and thinking_budget natively.
        # Rename thinking_token_budget (our internal field) to thinking_budget,
        # unless MTP is active — mlx_vlm rejects thinking_budget with a draft model.
        if "thinking_token_budget" in body:
            if model.mtp:
                body.pop("thinking_token_budget")
            else:
                body["thinking_budget"] = body.pop("thinking_token_budget")
        # mlx_vlm does not use chat_template_kwargs — strip it so the server
        # doesn't choke on an unrecognised field.
        body.pop("chat_template_kwargs", None)


class MlxAudio(Backend):
    """TTS / STT via mlx-audio (isolated venv)."""

    name = "mlx-audio"
    env = "audio"
    packages = ("mlx-audio", "fastapi", "uvicorn[standard]", "pydantic", "numpy")
    wrapper = "mlx_audio_wrapper.py"
    modalities = frozenset({"audio"})


class Mflux(Backend):
    """Image generation via mflux (isolated venv)."""

    name = "mflux"
    env = "image"
    packages = ("mflux", "fastapi", "uvicorn[standard]", "pydantic")
    wrapper = "mflux_wrapper.py"
    modalities = frozenset({"image"})


class Mflux2(Backend):
    """FLUX.2-klein unified image generation via mflux (isolated image venv).

    Uses Flux2KleinEdit — supports both text-to-image and image-to-image:
    - images: list[str]  base64 reference images (img2img / multi-image conditioning)
    - image_strength: float 0-1  denoising strength (None = full conditioning)
    - use_kv_cache: bool  cache reference KV states for faster n>1 batches
    Exposes /v1/images/generations and /v1/images/edits.
    Supports LoRA via --lora-paths and abliterated encoder via --model-path.
    """

    name = "mflux2"
    env = "image"
    packages = ("mflux", "fastapi", "uvicorn[standard]", "pydantic")
    wrapper = "mflux2_wrapper.py"
    modalities = frozenset({"image"})


class MfluxIdeogram(Backend):
    """Ideogram 4 image generation via mflux (isolated image venv).

    Text-to-image only (no img2img). Supports three quality presets:
    - V4_TURBO_12  — 12 steps, variable guidance schedule
    - V4_DEFAULT_20 — 20 steps, variable guidance schedule (default)
    - V4_QUALITY_48 — 48 steps, variable guidance schedule
    steps=12/20/48 in model defaults auto-maps to the matching preset schedule.
    Supports LoRA via --lora-paths and pre-quantized models via --model-path.
    """

    name = "mflux-ideogram"
    env = "image"
    packages = ("mflux", "fastapi", "uvicorn[standard]", "pydantic")
    wrapper = "mflux_ideogram_wrapper.py"
    modalities = frozenset({"image"})


BACKENDS: dict[str, Backend] = {
    b.name: b for b in (MlxLm(), MlxVlm(), MlxAudio(), Mflux(), Mflux2(), MfluxIdeogram())
}
