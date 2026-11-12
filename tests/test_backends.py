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
        "mlx-lm", "mlx-vlm", "mlx-vlm-dsv4", "mlx-vlm-zoo", "mtplx", "omlx", "mlx-audio", "mflux", "mflux2", "mflux-ideogram",
    }


def test_envs_and_packages():
    assert BACKENDS["mlx-lm"].env == "main"
    assert BACKENDS["mlx-vlm"].env == "main"
    assert BACKENDS["mlx-audio"].env == "audio"
    assert BACKENDS["mflux"].env == "image"
    assert BACKENDS["mlx-vlm-zoo"].env == "zoo"
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


# ── mlx-vlm-zoo (unsloth-zoo qwen4_exp fusions, isolated env) ─────────────────

def _zoo(**kw) -> ModelConfig:
    return ModelConfig(name="fn", source="/tmp/ckpt", backend="mlx-vlm-zoo", **kw)


def test_zoo_is_isolated_mlx_vlm_variant():
    b = BACKENDS["mlx-vlm-zoo"]
    assert isinstance(b, MlxVlm)
    # Own env: unsloth-zoo pins mlx and mlx-vlm and must not hold back the main venv.
    assert b.env == "zoo"
    assert b.wrapper == "mlx_vlm_zoo_wrapper.py"
    assert "unsloth-zoo==2026.11.4" in b.packages
    assert "mlx-vlm==0.7.12" in b.packages and "mlx==0.32.4" in b.packages
    # Keeps image input and the dashboard poller's "mlx-vlm" prefix match.
    assert _zoo().accepts_image_input is True
    assert b.name.startswith("mlx-vlm")


def test_zoo_env_does_not_share_packages_with_other_envs():
    # cli._installable_envs() unions packages per env name.
    others = {b.env for n, b in BACKENDS.items() if n != "mlx-vlm-zoo"}
    assert "zoo" not in others


def test_zoo_argv_and_inherited_apc_env():
    argv = _args_for(_zoo(thinking=True, port=6306))
    assert argv[0] == "PY" and argv[1].endswith("mlx_vlm_zoo_wrapper.py")
    assert "--enable-thinking" in argv and "--trust-remote-code" in argv
    env = _zoo().resolve_backend().env_vars(_zoo())
    assert env["APC_ENABLED"] == "1" and env["APC_DISK_PATH"].endswith("apc/fn")


# ── mtplx ─────────────────────────────────────────────────────────────────────

def test_mtplx_env_and_packages():
    b = BACKENDS["mtplx"]
    assert b.env == "mtplx"
    # pinned: MTPLX's server flags and the on-disk pack contract change between releases
    assert any(p.startswith("mtplx==") for p in b.packages)
    assert b.modalities == frozenset({"text"})
    assert b.accepts_image_input is True


def test_mtplx_argv_launches_serve_from_the_env_bin_dir():
    m = ModelConfig(name="flash", source="Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed",
                    backend="mtplx", port=6306)
    backend = m.resolve_backend()
    backend.python = lambda: "/venv/bin/python3"  # type: ignore[method-assign]
    try:
        args = backend.build_args(m, 6306)
    finally:
        del backend.python
    assert args[:2] == ["/venv/bin/mtplx", "serve"]
    assert ["--model", "Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"] == args[2:4]
    assert args[args.index("--port") + 1] == "6306"
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert "--no-auth" in args and "--yes" in args
    # the session cache is bounded and per model, never the default ~/.mtplx location
    assert args[args.index("--ssd-session-cache-dir") + 1].endswith("mtplx-cache/flash")
    assert args[args.index("--ssd-session-cache-max-size") + 1] == "32GB"


def test_mtplx_extra_args_come_last():
    m = ModelConfig(name="f", source="s", backend="mtplx", port=6306, extra_args=["--depth", "4"])
    backend = m.resolve_backend()
    backend.python = lambda: "/venv/bin/python3"  # type: ignore[method-assign]
    try:
        args = backend.build_args(m, 6306)
    finally:
        del backend.python
    assert args[-2:] == ["--depth", "4"]


def test_mtplx_transform_keeps_thinking_controls_and_drops_internal_budget():
    m = ModelConfig(name="f", source="s", backend="mtplx")
    body = {"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False},
            "thinking_token_budget": 512, "messages": []}
    m.resolve_backend().transform_request(body, m)
    assert body["enable_thinking"] is False
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "thinking_token_budget" not in body


# ── omlx ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def omlx_setup(tmp_path, monkeypatch):
    import mlx_server.manager as manager

    monkeypatch.setattr(manager, "STATE_DIR", tmp_path / "state")
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    m = ModelConfig(name="unc", source=str(ckpt), backend="omlx", port=6307)
    backend = m.resolve_backend()
    monkeypatch.setattr(backend, "python", lambda: "/venv/bin/python3", raising=False)
    return m, backend, ckpt, tmp_path / "state" / "omlx" / "unc"


def test_omlx_env_pinned_commit_and_kernel_build():
    b = BACKENDS["omlx"]
    assert b.env == "omlx"
    # not on PyPI: pinned to a commit, and the custom Metal kernels must be built
    assert any(p.startswith("omlx @ git+") and p.endswith(b.COMMIT) for p in b.packages)
    assert b.build_env == {"OMLX_WITH_CUSTOM_KERNEL": "1"}
    assert b.modalities == frozenset({"text"})


def test_omlx_argv_serves_a_private_model_dir_named_after_the_alias(omlx_setup):
    m, backend, ckpt, root = omlx_setup
    args = backend.build_args(m, 6307)
    assert args[:2] == ["/venv/bin/omlx", "serve"]
    assert args[args.index("--model-dir") + 1] == str(root / "models")
    assert args[args.index("--base-path") + 1] == str(root / "home")
    assert args[args.index("--port") + 1] == "6307"
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert "--headless" in args
    # the SSD KV cache is bounded and per model
    assert args[args.index("--paged-ssd-cache-dir") + 1] == str(root / "ssd-cache")
    assert args[args.index("--paged-ssd-cache-max-size") + 1] == BACKENDS["omlx"].SSD_CACHE_MAX
    link = root / "models" / "unc"
    assert link.is_symlink() and link.resolve() == ckpt.resolve()


def test_omlx_relinks_when_the_checkpoint_moves(omlx_setup, tmp_path):
    m, backend, ckpt, root = omlx_setup
    backend.build_args(m, 6307)
    other = tmp_path / "other"
    other.mkdir()
    m.source = str(other)
    backend.build_args(m, 6307)
    assert (root / "models" / "unc").resolve() == other.resolve()


def test_omlx_extra_args_come_last(omlx_setup):
    m, backend, _, _ = omlx_setup
    m.extra_args = ["--log-level", "debug"]
    assert backend.build_args(m, 6307)[-2:] == ["--log-level", "debug"]


def test_omlx_missing_hf_checkpoint_fails_with_download_hint(tmp_path, monkeypatch):
    import mlx_server.manager as manager

    monkeypatch.setattr(manager, "STATE_DIR", tmp_path / "state")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hf"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    m = ModelConfig(name="unc", source="nobody/not-downloaded", backend="omlx", port=6307)
    with pytest.raises(FileNotFoundError, match="hf download nobody/not-downloaded"):
        BACKENDS["omlx"].checkpoint_dir(m)


def test_omlx_transform_addresses_model_by_alias_and_drops_internal_budget():
    m = ModelConfig(name="unc", source="some/repo", backend="omlx")
    body = {"model": "some/repo", "thinking_token_budget": 512, "enable_thinking": False, "messages": []}
    m.resolve_backend().transform_request(body, m)
    assert body["model"] == "unc"
    assert body["enable_thinking"] is False
    assert "thinking_token_budget" not in body


def test_provision_env_passes_the_backend_build_env_to_pip(tmp_path, monkeypatch):
    import mlx_server.manager as manager

    monkeypatch.setattr(manager, "VENV_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(manager.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    manager.provision_env("omlx", ["omlx"])
    pip_call = [kw for cmd, kw in calls if cmd[-1] == "omlx" and "install" in cmd][0]
    assert pip_call["env"]["OMLX_WITH_CUSTOM_KERNEL"] == "1"
    calls.clear()
    manager.provision_env("audio", ["mlx-audio"])
    assert "OMLX_WITH_CUSTOM_KERNEL" not in calls[-1][1]["env"]
