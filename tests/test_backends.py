"""Tests for the backend registry (mlx_server/backends.py) and how ModelConfig
resolves to it.

These exercise the pieces that used to be inline in config.server_args() and
router.proxy() — argv construction and request transforms — without starting any
subprocess or model.

Run with:
    ~/.local/share/uv/tools/mlx-server/bin/python -m pytest tests/test_backends.py -v
"""

from pathlib import Path

import pytest

from mlx_server.backends import BACKENDS, Mflux, MlxAudio, MlxLm, MlxVlm
from mlx_server.config import ModelConfig


# ── registry ────────────────────────────────────────────────────────────────

def test_registry_keys():
    assert set(BACKENDS) == {
        "mlx-lm", "mlx-vlm", "mlx-vlm-dsv4", "mlx-audio", "mflux", "mflux2", "mflux-ideogram",
    }


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

# ── env vars (APC) ────────────────────────────────────────────────────────────

def test_env_vars_default_empty():
    m = ModelConfig(name="m", source="s", backend="mlx-lm")
    assert m.resolve_backend().env_vars(m) == {}
    for name in ("mlx-audio", "mflux"):
        assert BACKENDS[name].env_vars(m) == {}


def test_mlx_vlm_env_vars_enable_apc():
    m = ModelConfig(name="qwen3.6-27b", source="s", backend="mlx-vlm")
    env = m.resolve_backend().env_vars(m)
    assert env["APC_ENABLED"] == "1"
    assert env["APC_DISK_PATH"].endswith("apc/qwen3.6-27b")


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


# ── mlx-vlm-dsv4 (DeepSeek-V4) ────────────────────────────────────────────────

def _dsv4(**kw) -> ModelConfig:
    return ModelConfig(name="ds", source="mlx-community/DeepSeek-V4-Flash-4bit",
                       backend="mlx-vlm-dsv4", **kw)


def test_dsv4_is_text_only_mlx_vlm_variant():
    b = BACKENDS["mlx-vlm-dsv4"]
    assert isinstance(b, MlxVlm) and b.env == "main"
    assert b.wrapper == "mlx_vlm_dsv4_wrapper.py"
    assert _dsv4().accepts_image_input is False


def test_dsv4_argv_uses_dsv4_wrapper_and_thinking_flag():
    args = _dsv4(thinking=True, port=6301).server_args()
    assert args[1].endswith("mlx_vlm_dsv4_wrapper.py")
    assert "--enable-thinking" in args and "--trust-remote-code" in args


def test_dsv4_selects_dsml_parser_only_with_tools():
    m = _dsv4()
    with_tools = {"messages": [], "tools": [{"type": "function"}]}
    m.resolve_backend().transform_request(with_tools, m)
    assert with_tools["tool_parser"] == "deepseek_v4_dsml"

    plain = {"messages": []}
    m.resolve_backend().transform_request(plain, m)
    assert "tool_parser" not in plain


def test_dsv4_respects_caller_tool_parser():
    m = _dsv4()
    body = {"tools": [{}], "tool_parser": "json_tools"}
    m.resolve_backend().transform_request(body, m)
    assert body["tool_parser"] == "json_tools"


def test_dsv4_strips_chat_template_kwargs_like_mlx_vlm():
    m = _dsv4()
    body = {"chat_template_kwargs": {"x": 1}}
    m.resolve_backend().transform_request(body, m)
    assert "chat_template_kwargs" not in body


def test_source_tilde_is_expanded():
    cfg = ModelConfig(name="x", source="~/.cache/m", backend="mlx-vlm")
    assert cfg.source == str(Path("~/.cache/m").expanduser())
    assert not cfg.source.startswith("~")


def test_source_hf_repo_id_untouched():
    cfg = ModelConfig(name="x", source="org/model", backend="mlx-vlm")
    assert cfg.source == "org/model"


def test_mlx_vlm_reasoning_effort_map_rewrites_listed_value():
    m = ModelConfig(name="m", source="s", backend="mlx-vlm", reasoning_effort_map={"high": "xhigh"})
    body = {"reasoning_effort": "high"}
    m.resolve_backend().transform_request(body, m)
    assert body["reasoning_effort"] == "xhigh"


def test_mlx_vlm_reasoning_effort_passthrough_without_map():
    m = ModelConfig(name="m", source="s", backend="mlx-vlm")
    body = {"reasoning_effort": "high"}
    m.resolve_backend().transform_request(body, m)
    assert body["reasoning_effort"] == "high"


def test_mlx_vlm_reasoning_effort_map_leaves_unlisted_value():
    m = ModelConfig(name="m", source="s", backend="mlx-vlm", reasoning_effort_map={"high": "xhigh"})
    body = {"reasoning_effort": "low"}
    m.resolve_backend().transform_request(body, m)
    assert body["reasoning_effort"] == "low"


def test_mlx_vlm_reasoning_effort_map_nested_and_case_insensitive():
    m = ModelConfig(name="m", source="s", backend="mlx-vlm", reasoning_effort_map={"high": "xhigh"})
    body = {"reasoning": {"effort": " High "}}
    m.resolve_backend().transform_request(body, m)
    assert body["reasoning"]["effort"] == "xhigh"
