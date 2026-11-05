from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, field_validator, model_validator

from .backends import BACKENDS

if TYPE_CHECKING:
    from .backends import Backend

CONFIG_PATH = Path(os.environ.get("MLX_CONFIG", "~/.config/mlx/models.yaml")).expanduser()


class ModelDefaults(BaseModel):
    # LLM generation
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    repetition_context_size: int | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    thinking_token_budget: int | None = None
    # Image generation
    steps: int | None = None
    guidance_scale: float | None = None
    width: int | None = None
    height: int | None = None
    seed: int | None = None

    def as_request_fields(self) -> dict[str, Any]:
        """Return only the fields that have non-None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ModelConfig(BaseModel):
    name: str
    source: str
    # Arbitrary extra flags forwarded verbatim to the backend subprocess, e.g.:
    #   ["--chat-template-args", '{"enable_thinking": false}']
    #   ["--decode-concurrency", "16"]
    #   ["--quantize", "8"]  (for mflux image models)
    extra_args: list[str] = []
    defaults: ModelDefaults = ModelDefaults()
    protected: bool = False
    # Which serving engine runs this model. Must be a key in backends.BACKENDS
    # (e.g. "mlx-lm", "mlx-vlm", "mlx-audio", "mflux"). The backend determines the
    # wrapper, the Python env, capabilities, and request transforms.
    backend: str = "mlx-lm"
    # Override for image-url input support. None → use the backend's default
    # (e.g. a model on the mlx-lm backend that should still accept image content).
    multimodal: bool | None = None
    # True → MTP speculative decoding is active; thinking_token_budget is incompatible
    # and is stripped by the router rather than forwarded as thinking_budget.
    mtp: bool = False
    # True → pass --enable-thinking to mlx_vlm.server, allowing per-request thinking
    # control. Set only for models that support thinking mode (Qwen3). Gemma4 emits
    # <|channel>thought tokens by default when this flag is present.
    thinking: bool = False
    # Rewrites the request's `reasoning_effort` per model, e.g. {high: xhigh}.
    # For chat templates that accept only a fixed set of effort names (Qwen3.8
    # takes xhigh/medium/low and raises on the OpenAI-style "high" that clients
    # such as opencode send). Values not in the map pass through unchanged.
    reasoning_effort_map: dict[str, str] = {}
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

    @field_validator("source")
    @classmethod
    def _expand_source(cls, v: str) -> str:
        # Local checkpoints may be written as ~/...; HF repo ids never start with ~.
        return os.path.expanduser(v) if v.startswith("~") else v

    @model_validator(mode="after")
    def _check_backend(self) -> "ModelConfig":
        if self.backend not in BACKENDS:
            raise ValueError(
                f"Unknown backend {self.backend!r} for model {self.name!r}. "
                f"Valid backends: {sorted(BACKENDS)}"
            )
        return self

    def resolve_backend(self) -> "Backend":
        return BACKENDS[self.backend]

    @property
    def modality(self) -> str:
        """Output modality, derived from the backend (for display/routing)."""
        return self.resolve_backend().primary_modality()

    @property
    def accepts_image_input(self) -> bool:
        if self.multimodal is not None:
            return self.multimodal
        return self.resolve_backend().accepts_image_input

    def backend_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def server_args(self) -> list[str]:
        return self.resolve_backend().build_args(self, self.port)


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 6300
    base_port: int = 6301
    max_total_memory_gb: float | None = None
    # Hard floor: refuse to start a model (503) when free system RAM would drop
    # below this after loading. Backstop against swap/OOM when cap alone is not
    # enough (e.g. KV-cache spikes during the first inference pass).
    min_free_memory_gb: float = 8.0
    # Before force-killing an in-flight (busy) LRU victim during eviction, wait
    # up to this long for it to finish its current request(s). If it doesn't
    # finish in time, it is force-stopped anyway so memory pressure can't
    # deadlock the system indefinitely.
    evict_wait_timeout_seconds: float = 300.0
    # Opt-in (default off). Read-only technical dashboard at /_dashboard/ (numbers only, never prompts
    # or responses). Per-request metrics are kept in a SQLite file under
    # ~/.local/share/mlx/ and survive restarts; the newest N rows are retained.
    dashboard_enabled: bool = False
    metrics_retention_rows: int = 50_000
    models: list[ModelConfig] = []

    @model_validator(mode="after")
    def assign_ports(self) -> "ServerConfig":
        for i, model in enumerate(self.models):
            if model.port == 0:
                model.port = self.base_port + i
        return self

    def model_by_name(self, name: str) -> ModelConfig | None:
        return next((m for m in self.models if m.name == name), None)


def parse_listen_address(address: str) -> tuple[str, int | None]:
    """Parse 'host:port' or 'host' into (host, optional_port).

    Examples:
        "0.0.0.0:6300"  → ("0.0.0.0", 6300)
        "0.0.0.0"       → ("0.0.0.0", None)
        "127.0.0.1"     → ("127.0.0.1", None)
    """
    if ":" in address:
        parts = address.rsplit(":", 1)
        if parts[1].isdigit():
            return parts[0], int(parts[1])
    return address, None


def load_config(path: Path = CONFIG_PATH) -> ServerConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path}. "
            "Run: mkdir -p ~/.config/mlx && cp config/models.example.yaml ~/.config/mlx/models.yaml"
        )
    raw = yaml.safe_load(path.read_text())
    server_raw = raw.get("server", {})
    models_raw = raw.get("models", [])
    return ServerConfig(
        **server_raw,
        models=[ModelConfig(**m) for m in models_raw],
    )
