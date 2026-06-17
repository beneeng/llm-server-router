from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, model_validator

CONFIG_PATH = Path(os.environ.get("MLX_CONFIG", "~/.config/mlx/models.yaml")).expanduser()


class ModelDefaults(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    repetition_context_size: int | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    thinking_token_budget: int | None = None

    def as_request_fields(self) -> dict[str, Any]:
        """Return only the fields that have non-None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ModelConfig(BaseModel):
    name: str
    source: str
    # Arbitrary extra flags forwarded verbatim to mlx_lm.server, e.g.:
    #   ["--chat-template-args", '{"enable_thinking": false}']
    #   ["--decode-concurrency", "16"]
    extra_args: list[str] = []
    defaults: ModelDefaults = ModelDefaults()
    protected: bool = False
    # True for models that accept image_url content blocks (Gemma4, qwen3_5_moe VL, etc.).
    multimodal: bool = False
    # True → start via mlx_vlm.server (full vision pipeline, per-request thinking_budget).
    # False → start via mlx_lm_wrapper.py (text-only, ThinkingBudgetProcessor patch).
    vlm: bool = False
    # True → MTP speculative decoding is active; thinking_token_budget is incompatible
    # and is stripped by the router rather than forwarded as thinking_budget.
    mtp: bool = False
    # True → pass --enable-thinking to mlx_vlm.server, allowing per-request thinking
    # control. Set only for models that support thinking mode (Qwen3). Gemma4 emits
    # <|channel>thought tokens by default when this flag is present.
    thinking: bool = False
    # When True, mlx start skips this model; it is started on the first request
    # via MemoryManager.ensure_model_ready().  mlx start --model NAME ignores
    # this flag and starts the model explicitly regardless.
    lazy: bool = True
    # Expected total footprint (weights + KV cache) in GB. Used for pre-eviction
    # budgeting: we clear this much room BEFORE starting the model. Set conservatively
    # (i.e. larger than actual weights) to account for KV-cache pre-allocation.
    memory_gb: float | None = None
    # Assigned at load time by ServerConfig
    port: int = 0

    def backend_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def server_args(self) -> list[str]:
        from .manager import _bin
        from pathlib import Path as _Path
        if self.vlm:
            vlm_wrapper = str(_Path(__file__).parent / "mlx_vlm_wrapper.py")
            return [
                _bin("python3"), vlm_wrapper,
                "--model", self.source,
                "--port", str(self.port),
                "--host", "127.0.0.1",
                "--trust-remote-code",
                *( ["--enable-thinking"] if self.thinking else [] ),
            ] + self.extra_args
        wrapper = str(_Path(__file__).parent / "mlx_lm_wrapper.py")
        return [
            _bin("python3"), wrapper,
            "--model", self.source,
            "--port", str(self.port),
            "--trust-remote-code",
        ] + self.extra_args


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 6300
    base_port: int = 6301
    max_total_memory_gb: float | None = None
    # Hard floor: refuse to start a model (503) when free system RAM would drop
    # below this after loading. Backstop against swap/OOM when cap alone is not
    # enough (e.g. KV-cache spikes during the first inference pass).
    min_free_memory_gb: float = 8.0
    models: list[ModelConfig] = []

    @model_validator(mode="after")
    def assign_ports(self) -> "ServerConfig":
        for i, model in enumerate(self.models):
            if model.port == 0:
                model.port = self.base_port + i
        return self

    def model_by_name(self, name: str) -> ModelConfig | None:
        return next((m for m in self.models if m.name == name), None)


def load_config(path: Path = CONFIG_PATH) -> ServerConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. "
            "Run: mkdir -p ~/.config/mlx && cp config/models.yaml ~/.config/mlx/models.yaml"
        )
    raw = yaml.safe_load(path.read_text())
    server_raw = raw.get("server", {})
    models_raw = raw.get("models", [])
    return ServerConfig(
        **server_raw,
        models=[ModelConfig(**m) for m in models_raw],
    )
