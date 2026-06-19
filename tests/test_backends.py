"""Tests for the backend registry (mlx_server/backends.py) and how ModelConfig
resolves to it.

These exercise the pieces that used to be inline in config.server_args() and
router.proxy() — argv construction and request transforms — without starting any
subprocess or model.

Run with:
    ~/.local/share/uv/tools/mlx-server/bin/python -m pytest tests/test_backends.py -v
"""

import pytest

from mlx_server.backends import BACKENDS, Mflux, MlxAudio, MlxLm, MlxVlm
from mlx_server.config import ModelConfig


# ── registry ────────────────────────────────────────────────────────────────

def test_registry_keys():
    assert set(BACKENDS) == {"mlx-lm", "mlx-vlm", "mlx-audio", "mflux"}


def test_envs_and_packages():
    assert BACKENDS["mlx-lm"].env == "main"
    assert BACKENDS["mlx-vlm"].env == "main"
    assert BACKENDS["mlx-audio"].env == "audio"
    assert BACKENDS["mflux"].env == "image"
    # main backends declare no install packages; optional ones do.
    assert BACKENDS["mlx-lm"].packages == ()
    assert "mflux" in BACKENDS["mflux"].packages
    assert "mlx-audio" in BACKENDS["mlx-audio"].packages


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="Unknown backend"):
        ModelConfig(name="x", source="y", backend="nope")


# ── modality / capability resolution ──────────────────────────────────────────

def test_modality_derived_from_backend():
    assert ModelConfig(name="t", source="s", backend="mlx-lm").modality == "text"
    assert ModelConfig(name="a", source="s", backend="mlx-audio").modality == "audio"
    assert ModelConfig(name="i", source="s", backend="mflux").modality == "image"


def test_accepts_image_input_default_and_override():
    # mlx-vlm accepts images by default; mlx-lm does not.
    assert ModelConfig(name="v", source="s", backend="mlx-vlm").accepts_image_input is True
    assert ModelConfig(name="l", source="s", backend="mlx-lm").accepts_image_input is False
    # per-model override wins (the gemma4-e4b-qat case).
    assert ModelConfig(name="o", source="s", backend="mlx-lm", multimodal=True).accepts_image_input is True


# ── argv construction ─────────────────────────────────────────────────────────

def _args_for(model: ModelConfig) -> list[str]:
    """build_args without resolving the real interpreter path."""
    backend = model.resolve_backend()
    backend.python = lambda: "PY"  # type: ignore[method-assign]
    try:
        return backend.build_args(model, model.port or 6301)
    finally:
        del backend.python  # restore the class method


def test_mlx_lm_argv():
    m = ModelConfig(name="m", source="repo/x", backend="mlx-lm", port=6301)
    args = _args_for(m)
    assert args[:1] == ["PY"]
    assert args[1].endswith("mlx_lm_wrapper.py")
    assert "--trust-remote-code" in args
    assert ["--model", "repo/x"] == args[2:4]
    assert "--host" in args and "127.0.0.1" in args


def test_mlx_vlm_argv_thinking_flag():
    on = ModelConfig(name="m", source="s", backend="mlx-vlm", thinking=True, port=6301)
    off = ModelConfig(name="m", source="s", backend="mlx-vlm", thinking=False, port=6301)
    assert "--enable-thinking" in _args_for(on)
    assert "--enable-thinking" not in _args_for(off)


def test_extra_args_appended_last():
    m = ModelConfig(name="m", source="s", backend="mflux", extra_args=["--quantize", "8"], port=6301)
    args = _args_for(m)
    assert args[-2:] == ["--quantize", "8"]
    assert args[1].endswith("mflux_wrapper.py")


# ── request transforms ────────────────────────────────────────────────────────

def test_mlx_vlm_transform_renames_thinking_budget():
    m = ModelConfig(name="m", source="s", backend="mlx-vlm")
    body = {"thinking_token_budget": 2048, "chat_template_kwargs": {"x": 1}}
    m.resolve_backend().transform_request(body, m)
    assert body["thinking_budget"] == 2048
    assert "thinking_token_budget" not in body
    assert "chat_template_kwargs" not in body  # stripped for mlx_vlm


def test_mlx_vlm_transform_drops_budget_under_mtp():
    m = ModelConfig(name="m", source="s", backend="mlx-vlm", mtp=True)
    body = {"thinking_token_budget": 2048}
    m.resolve_backend().transform_request(body, m)
    assert "thinking_budget" not in body
    assert "thinking_token_budget" not in body


def test_mlx_lm_transform_moves_enable_thinking_false():
    m = ModelConfig(name="m", source="s", backend="mlx-lm")
    body = {"enable_thinking": False}
    m.resolve_backend().transform_request(body, m)
    assert body["chat_template_kwargs"]["enable_thinking"] is False
    assert "enable_thinking" not in body


def test_mlx_lm_transform_leaves_enable_thinking_true():
    m = ModelConfig(name="m", source="s", backend="mlx-lm")
    body = {"enable_thinking": True}
    m.resolve_backend().transform_request(body, m)
    assert body["enable_thinking"] is True
