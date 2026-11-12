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
    # Extra environment for the pip install of `packages` (e.g. a build-time flag).
    build_env: dict[str, str] = {}
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

    # Per-model cap for the on-disk APC tier. mlx-vlm defaults APC_DISK_MAX_GB
    # to 0 (unbounded) when unset, which let the cache fill the entire disk
    # (observed: 1.2TB across 7 models). LRU-evicted once a model's shard
    # directory exceeds this.
    APC_DISK_MAX_GB = 100

    def env_vars(self, model: "ModelConfig") -> dict[str, str]:
        # Automatic Prefix Caching (mlx-vlm >=0.5): hash-based, block-level KV
        # reuse across requests with LRU+refcount eviction. APC_DISK_PATH adds
        # an SSD tier so a warm cache survives this model's process being
        # stopped/evicted, not just concurrent requests within one run.
        from .manager import STATE_DIR

        return {
            "APC_ENABLED": "1",
            "APC_DISK_PATH": str(STATE_DIR / "apc" / model.name),
            "APC_DISK_MAX_GB": str(self.APC_DISK_MAX_GB),
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
        # Per-model remap of the reasoning effort for templates with a fixed value
        # set. mlx-vlm reads it from `reasoning_effort` or `reasoning.effort` and
        # normalizes with strip().lower(), so match the same way.
        if model.reasoning_effort_map:
            remap = {k.strip().lower(): v for k, v in model.reasoning_effort_map.items()}
            holders = [(body, "reasoning_effort")]
            if isinstance(body.get("reasoning"), dict):
                holders.append((body["reasoning"], "effort"))
            for holder, key in holders:
                val = holder.get(key)
                if isinstance(val, str) and val.strip().lower() in remap:
                    holder[key] = remap[val.strip().lower()]
        # mlx_vlm does not use chat_template_kwargs — strip it so the server
        # doesn't choke on an unrecognised field.
        body.pop("chat_template_kwargs", None)


class MlxVlmDsv4(MlxVlm):
    """DeepSeek-V4 via mlx_vlm.server with DeepSeek's own prompt encoder.

    The checkpoint's chat template is a stub without tools; the wrapper swaps in
    the reference encoder and a DSML tool parser (see mlx_vlm_dsv4_wrapper.py).
    Text-only.
    """

    name = "mlx-vlm-dsv4"
    wrapper = "mlx_vlm_dsv4_wrapper.py"
    accepts_image_input = False

    def transform_request(self, body: dict, model: "ModelConfig") -> None:
        super().transform_request(body, model)
        # The stub chat template carries no tool markers, so the parser cannot
        # be inferred from it; select the DSML parser explicitly.
        if body.get("tools"):
            body.setdefault("tool_parser", "deepseek_v4_dsml")


class MlxVlmZoo(MlxVlm):
    """mlx_vlm.server with unsloth-zoo's qwen4_exp fusions (PR #1871) enabled.

    Opt-in per model. Runs in its own env because unsloth-zoo pins
    mlx==0.32.4 and mlx-vlm<=0.7.12, which must not hold back the main venv.
    Keep the pins here in sync with unsloth-zoo's own requirements.
    """

    name = "mlx-vlm-zoo"
    env = "zoo"
    packages = ("unsloth-zoo==2026.11.4", "mlx-vlm==0.7.12", "mlx==0.32.4")
    wrapper = "mlx_vlm_zoo_wrapper.py"


class Mtplx(Backend):
    """Qwen3.8-Flash-Next via MTPLX (native MTP speculative decoding, isolated env).

    MTPLX is its own OpenAI-compatible server, so there is no wrapper script: the
    subprocess is `mtplx serve`, which execs into the server (the pid we track is
    the server). The 29.8 GiB n-gram table is streamed from SSD by MTPLX itself,
    so resident weights stay at ~77 GB. The model must already be downloaded
    (`hf download <repo>`): a first-start download would blow the port timeout.
    Pinned because the CLI flags and the pack contract change between releases.
    """

    name = "mtplx"
    env = "mtplx"
    packages = ("mtplx==2.12.3",)
    modalities = frozenset({"text"})
    accepts_image_input = True

    # Soft cap for the persistent session cache (MTPLX's default is "auto", up to
    # 100 GB per model, written under ~/.mtplx). Same reasoning as APC_DISK_MAX_GB.
    SESSION_CACHE_MAX = "32GB"

    def build_args(self, model: "ModelConfig", port: int) -> list[str]:
        from .manager import STATE_DIR

        mtplx_bin = str(Path(self.python()).parent / "mtplx")
        return [
            mtplx_bin, "serve",
            "--model", model.source,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--no-auth", "--yes",
            "--ssd-session-cache-dir", str(STATE_DIR / "mtplx-cache" / model.name),
            "--ssd-session-cache-max-size", self.SESSION_CACHE_MAX,
            *model.extra_args,
        ]

    def transform_request(self, body: dict, model: "ModelConfig") -> None:
        # MTPLX reads enable_thinking / chat_template_kwargs natively (verified:
        # both forms switch reasoning off). thinking_token_budget is our internal
        # field and means nothing to it.
        body.pop("thinking_token_budget", None)


class Omlx(Backend):
    """Qwen3.8-Flash-Next via oMLX (Lightning MTP, automatic PLE mmap offload).

    oMLX is its own OpenAI-compatible server; `omlx serve` is the tracked process.
    It serves every subdirectory of a `--model-dir` and addresses a model by its
    directory name, so each router model gets a private model dir holding one
    symlink named after the alias, plus a private `--base-path` (settings, caches,
    logs). `source` is a local checkpoint dir or an HF repo id that must already be
    in the HF cache (`hf download <repo>`): a first-start download would blow the
    port timeout. The checkpoint must carry an embedded MTP head (oQ*e-mtp repos).

    oMLX is not on PyPI, so the env installs a pinned git commit. The custom Metal
    kernels (much faster than the generic fallbacks for this model family) are only
    built when OMLX_WITH_CUSTOM_KERNEL=1, which needs the Xcode Metal Toolchain
    (`xcodebuild -downloadComponent MetalToolchain`); without it the install fails
    loudly rather than silently serving slow.
    """

    name = "omlx"
    env = "omlx"
    COMMIT = "dbbfa6195cf77a6dc421fe9e3f2c573e9788447d"
    packages = (f"omlx @ git+https://github.com/jundot/omlx@{COMMIT}",)
    build_env = {"OMLX_WITH_CUSTOM_KERNEL": "1"}
    modalities = frozenset({"text"})
    accepts_image_input = True

    # Cap for the paged SSD KV cache (oMLX's default is unbounded). Same reasoning
    # as APC_DISK_MAX_GB.
    SSD_CACHE_MAX = "32GB"

    @staticmethod
    def checkpoint_dir(model: "ModelConfig") -> Path:
        """Resolve `source` to a local directory without touching the network."""
        local = Path(model.source).expanduser()
        if local.is_dir():
            return local.resolve()
        from huggingface_hub import snapshot_download

        try:
            return Path(snapshot_download(model.source, local_files_only=True))
        except Exception as exc:
            raise FileNotFoundError(
                f"{model.source} is not in the HF cache; run `hf download {model.source}` first"
            ) from exc

    def build_args(self, model: "ModelConfig", port: int) -> list[str]:
        from .manager import STATE_DIR

        root = STATE_DIR / "omlx" / model.name
        models_dir = root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        link = models_dir / model.name
        target = self.checkpoint_dir(model)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
        omlx_bin = str(Path(self.python()).parent / "omlx")
        return [
            omlx_bin, "serve",
            "--model-dir", str(models_dir),
            "--base-path", str(root / "home"),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--headless",
            "--memory-guard", "safe",
            "--paged-ssd-cache-dir", str(root / "ssd-cache"),
            "--paged-ssd-cache-max-size", self.SSD_CACHE_MAX,
            *model.extra_args,
        ]

    def transform_request(self, body: dict, model: "ModelConfig") -> None:
        # oMLX addresses the model by its directory name (= the alias), not by source.
        body["model"] = model.name
        body.pop("thinking_token_budget", None)


class MlxAudio(Backend):
    """TTS / STT via mlx-audio (isolated venv)."""

    name = "mlx-audio"
    env = "audio"
    packages = ("mlx-audio", "fastapi", "uvicorn[standard]", "pydantic", "numpy", "python-multipart")
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
    b.name: b
    for b in (MlxLm(), MlxVlm(), MlxVlmDsv4(), MlxVlmZoo(), Mtplx(), Omlx(), MlxAudio(), Mflux(), Mflux2(), MfluxIdeogram())
}
