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
    max_tokens: int | None = None
    thinking_token_budget: int | None = None

    def as_request_fields(self) -> dict[str, Any]:
        """Return only the fields that have non-None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ModelConfig(BaseModel):
    name: str
    source: str
    multimodal: bool = False
    # vllm-mlx 0.4.0rc1 continuous-batching patch for Gemma4 MoE is broken
    # (patch_gemma4_attention_for_batching doesn't forward shared_kv).
    # Set to False for any mllm model until a fix lands upstream.
    continuous_batching: bool = True
    reasoning_parser: str | None = None
    # Arbitrary extra flags forwarded verbatim to vllm-mlx serve, e.g.:
    #   ["--default-chat-template-kwargs", '{"enable_thinking": false}']
    extra_args: list[str] = []
    defaults: ModelDefaults = ModelDefaults()
    protected: bool = False
    # When True, mlx start skips this model; it is started on the first request
    # via MemoryManager.ensure_model_ready().  mlx start --model NAME ignores
    # this flag and starts the model explicitly regardless.
    lazy: bool = False
    # Expected total footprint (weights + KV cache) in GB. Used for pre-eviction
    # budgeting: we clear this much room BEFORE starting the model. Set conservatively
    # (i.e. larger than actual weights) to account for KV-cache pre-allocation.
    memory_gb: float | None = None
    # Assigned at load time by ServerConfig
    port: int = 0

    def backend_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def vllm_args(self) -> list[str]:
        from .manager import _bin
        args = [_bin("vllm-mlx"), "serve", self.source, "--port", str(self.port)]
        if self.continuous_batching:
            args.append("--continuous-batching")
        if self.multimodal:
            args.append("--mllm")
        if self.reasoning_parser:
            args += ["--reasoning-parser", self.reasoning_parser]
        args += self.extra_args
        return args


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
