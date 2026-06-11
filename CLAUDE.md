# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Install / reinstall

```bash
uv tool install -e . --python 3.12 --prerelease allow
```

This installs the `mlx` CLI globally into a uv-managed venv at `~/.local/share/uv/tools/mlx-server/`. The `--prerelease allow` flag is required because `vllm-mlx` is at 0.4.0rc1.

One-time config setup:
```bash
mkdir -p ~/.config/mlx && cp config/models.yaml ~/.config/mlx/models.yaml
```

## Running / testing manually

```bash
mlx start                        # start all model backends + router
mlx start --model qwen3.6-27b   # start a single model
mlx status                       # check live/dead pids
mlx logs [MODEL]                 # tail -f log (default: router)
mlx stop                         # stop router first, then all models

curl http://localhost:6300/v1/models
curl -X POST http://localhost:6300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

There are no automated tests. Verification is manual via `curl` + `mlx logs`.

## Architecture

Two processes per run: one `vllm-mlx serve` subprocess per model (ports 6301+) and one FastAPI uvicorn process as the router (port 6300).

**Request flow through the router (`mlx_server/router.py`):**
1. Extract `model` from JSON body or query param
2. Look up `ModelConfig` by alias name
3. Merge `ModelConfig.defaults` into the body (caller values take precedence)
4. Rewrite `body["model"]` from the friendly alias to the full HuggingFace ID (`model_cfg.source`) — vllm-mlx rejects aliases
5. Proxy to `http://127.0.0.1:<port>/<path>` via `httpx`

For streaming requests, the `httpx.AsyncClient` **must** be opened inside `_stream_gen()`, not in the outer `proxy()` scope — the outer async context exits before Starlette iterates the generator, which closes the client prematurely.

**Config loading (`mlx_server/config.py`):**
- `load_config()` reads `~/.config/mlx/models.yaml` (overridable via `MLX_CONFIG` env var)
- `ServerConfig.assign_ports` auto-assigns sequential ports starting at `base_port` (6301) to models that have `port == 0`
- `ModelConfig.vllm_args()` builds the full `vllm-mlx serve` command; calls `_bin()` from `manager.py` to resolve binaries from the venv

**Process management (`mlx_server/manager.py`):**
- PIDs at `~/.local/share/mlx/pids.json`; logs at `~/.local/share/mlx/logs/<name>.log`
- `_bin(name)` checks `Path(sys.executable).parent / name` first (venv bin dir) before `shutil.which` — required because `vllm-mlx` is not on the system PATH
- Stop sequence: SIGTERM → 5s poll → SIGKILL

## Known bugs and constraints

**Gemma 4 `continuous_batching: false`** — `vllm-mlx` 0.4.0rc1's `patch_gemma4_attention_for_batching` doesn't forward the `shared_kv` kwarg from `mlx-vlm` 0.6.2, crashing on any batched request. All Gemma4 models in config have `continuous_batching: false`. Do not re-enable until a vllm-mlx upstream fix is confirmed.

**Gemma 4 tool-call format** — Gemma4 emits tool calls in a custom non-JSON format (`<|tool_call>` / `<tool_call|>` delimiters, `<|"|>` string escaping, unquoted keys). Without `--enable-auto-tool-choice --tool-call-parser gemma4` (set in `extra_args` for all Gemma4 models), vllm-mlx does not parse these as tool calls — the model gets no result back and retries forever.

**Gemma 4 MLLM tool history bug (router workaround)** — vllm-mlx 0.4.0rc1's `MLLM.stream_chat` silently drops assistant messages where `content=null`, even when `tool_calls` is present. Gemma4 models are always loaded as MLLM. The result is the model sees only the original user message on every turn and loops. **Fix**: `router.py:_prerender_gemma4_tool_history()` pre-renders tool call + response tokens directly into the assistant message `content` string using native Gemma4 special tokens (`<|tool_call>`, `<|tool_response>`, `<|"|>`), which the tokenizer correctly encodes. Applied for all `multimodal: true` models before forwarding.

**Gemma 4 repetition collapse** — separate structural bug triggered by large system prompts (>10K chars) or many tool definitions (>5-6). Not fixable with `repetition_penalty`. Gemma is excluded from opencode; use via Pi harness or direct curl.

**Qwen3 temperature=0 prohibition** — vendor docs prohibit `temperature: 0` in thinking mode; minimum 0.6 for `qwen3.6-moe`, 0.2 for `qwen3.6-27b`.

**Router config is loaded once at startup** — `_cfg` in `router.py` is a module-level singleton. Config changes require restarting the router process (`mlx stop` then `mlx start`). There is no hot-reload.

## External integrations

- **opencode**: `~/.config/opencode/opencode_local.json` — `mlx-router` provider (port 6300), Qwen models only
- **Pi harness**: `~/.pi/agent/models.json` — `mlx-router` provider, all 5 models including Gemma4
