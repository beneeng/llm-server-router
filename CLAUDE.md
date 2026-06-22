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

**Optional backends** (audio, image, …) run in separate isolated venvs because they
have conflicting `mlx` version pins with `mlx-vlm`. They are **auto-provisioned on
first model start** — no manual step is required. To pre-warm them explicitly:
```bash
mlx install audio   # creates ~/.local/share/mlx/venvs/audio/ + installs mlx-audio
mlx install image   # creates ~/.local/share/mlx/venvs/image/ + installs mflux
mlx install --all   # provisions every optional backend env
```
The install target is a backend **env** name (`audio`, `image`); the package
lists are derived from `backends.BACKENDS`, not hardcoded in the CLI.

Do NOT add `mflux` or `mlx-audio` to `pyproject.toml` — they will break `uv sync`.

## Automated tests

```bash
~/.local/share/uv/tools/mlx-server/bin/python -m pytest tests/test_patches.py -v
```

Tests must run under the tool venv python (not system python) because `vllm_mlx` and `mlx_lm` are installed there. `pytest` is installed in that venv; if missing run `~/.local/share/uv/tools/mlx-server/bin/python -m pip install pytest`.

`tests/test_patches.py` covers both monkey-patches in `mlx_lm_wrapper.py`: for each patch there is a "without patch" test that reproduces the known failure mode, and a "with patch" test that confirms it is gone.

## Running / testing manually

```bash
mlx start                        # start all non-lazy model backends + router
mlx start --model qwen3.6-27b   # start a single model
mlx status                       # check live/dead pids
mlx logs [MODEL]                 # tail -f log (default: router)
mlx stop                         # stop router first, then all models
mlx memory                       # per-model RSS + system RAM pressure

# Text
curl http://localhost:6300/v1/models
curl -X POST http://localhost:6300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}],"stream":true}'

# Audio (requires a kokoro entry in config; audio backend auto-provisions on first call)
curl -X POST http://localhost:6300/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello world","voice":"af_heart"}' --output speech.mp3

# Image (requires a flux-schnell entry in config; image backend auto-provisions on first call)
curl -X POST http://localhost:6300/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"flux-schnell","prompt":"a red cat"}' \
  | python3 -c "import sys,json,base64; open('out.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"
```

There are no automated integration tests. Verification is manual via `curl` + `mlx logs`.

## Architecture

One subprocess per model (ports 6301+) and one FastAPI uvicorn router (port 6300).

**Backend registry (`mlx_server/backends.py`) — the single source of truth.** A
*backend* is a serving engine that bundles: its Python env, pip packages, wrapper
script, capabilities (`modalities`, `accepts_image_input`), subprocess argv
(`build_args`/`extra_flags`), and router-side request rewriting (`transform_request`).
Models reference a backend by name (`backend:` in models.yaml). To add a modality
(e.g. video): add one `Backend` subclass + one wrapper file + register it in
`BACKENDS` — no central dispatch to edit.

| backend (`name`) | wrapper | env (`mlx install <env>`) |
|---|---|---|
| `mlx-lm` | `mlx_lm_wrapper.py` | `main` (tool venv) |
| `mlx-vlm` | `mlx_vlm_wrapper.py` | `main` (tool venv) |
| `mlx-audio` | `mlx_audio_wrapper.py` | `audio` → `~/.local/share/mlx/venvs/audio/` |
| `mflux` | `mflux_wrapper.py` | `image` → `~/.local/share/mlx/venvs/image/` |

Text wrappers (`mlx-lm`/`mlx-vlm`) monkeypatch + delegate to the upstream
`mlx_lm.server` / `mlx_vlm.server`. Self-hosted wrappers (`mlx-audio`/`mflux`,
and future video) define their own FastAPI app and share boilerplate via
`_wrapper_base.py` (argparse + health route + uvicorn launch). `_wrapper_base.py`
must stay import-light (stdlib + fastapi/uvicorn only) so it loads under the
isolated venvs; wrappers import it by plain name (`from _wrapper_base import …`)
since the script's own dir is on `sys.path`.

**Request flow through the router (`mlx_server/router.py`):**
1. Extract `model` from JSON body, query param, or multipart form field (for audio transcriptions)
2. Look up `ModelConfig` by alias name
3. Call `MemoryManager.ensure_model_ready()` — triggers lazy start and LRU eviction if needed
4. Merge `ModelConfig.defaults` into the body (caller values take precedence)
5. Rewrite `body["model"]` from alias to `model_cfg.source`
6. Apply `model_cfg.resolve_backend().transform_request(body, model_cfg)` (e.g. mlx-vlm renames thinking_budget + strips chat_template_kwargs; mlx-lm rewrites enable_thinking; audio/image are no-ops)
7. Proxy to `http://127.0.0.1:<port>/<path>` via `httpx`

Image-content requests to a model whose `accepts_image_input` is False are
rejected with 400 before the backend is touched.

For streaming requests, the `httpx.AsyncClient` **must** be opened inside `_stream_gen()`, not in the outer `proxy()` scope — the outer async context exits before Starlette iterates the generator, which closes the client prematurely.

**Config loading (`mlx_server/config.py`):**
- `load_config()` reads `~/.config/mlx/models.yaml` (overridable via `MLX_CONFIG` env var)
- `ServerConfig.assign_ports` auto-assigns sequential ports starting at `base_port` (6301) to models that have `port == 0`
- `ModelConfig.backend` names a key in `backends.BACKENDS` (validated at load — unknown backend fails fast). `modality` and `accepts_image_input` are **derived** from the backend; `multimodal` is an optional per-model override of image-input support
- `ModelConfig.server_args()` is a one-liner delegating to `resolve_backend().build_args(self, port)`
- `ModelDefaults` carries both LLM params and image-gen params (`steps`, `guidance_scale`, `width`, `height`, `seed`); `as_request_fields()` returns only non-None values. Adding video defaults later means new optional fields here

**Process management (`mlx_server/manager.py`):**
- PIDs at `~/.local/share/mlx/pids.json`; logs at `~/.local/share/mlx/logs/<name>.log`
- `_bin(name)` resolves executables from the tool venv bin dir before falling back to PATH
- `_backend_python(env)` resolves the Python for an isolated backend env from its venv; raises `FileNotFoundError` with `mlx install <env>` hint if missing
- `provision_env(env, packages)` creates a venv + pip-installs; `ensure_backend_env(backend)` calls it on first model start (logged to the model's log) for any non-`main` backend whose venv is absent — this is the auto-provisioning. Slow first start is expected; provisioning blocks `start_model` *before* the 120 s port poll, so it doesn't count against that timeout
- `VENV_DIR = ~/.local/share/mlx/venvs/`
- Stop sequence: SIGTERM → 5s poll → SIGKILL

**Memory management (`mlx_server/memory.py`):**
- `MemoryManager.ensure_model_ready(name)` is called on every proxied request
- Evicts LRU (least-recently-used, non-protected, not started in last 60 s) models until memory is within budget
- Polls the backend port for up to 120 s after starting (accommodates mflux's pre-startup model load)

## API specification — keep in sync

`docs/api/openapi.yaml` is the authoritative API specification.  `docs/api/APISUMMARY.md`
is the quick-reference for consumer apps (2-3 file read to onboard).

**Any change that adds, removes, or modifies an API endpoint, request schema,
response schema, SSE event format, or router behaviour MUST also update both
files.**  This includes:

- Adding a new wrapper with new endpoints
- Changing request or response fields in any wrapper
- Adding a new backend or modality
- Changing router request transforms or error responses
- Adding/changing SSE event types or formats
- Adding new config defaults that affect proxied requests

Run a quick sanity check after changes: verify the OpenAPI spec at
https://editor.swagger.io or with `python3 -c "import yaml; yaml.safe_load(open('docs/api/openapi.yaml'))"`.

## Known bugs and constraints

**Gemma 4 continuous batching** — `vllm-mlx` 0.4.0rc1's `patch_gemma4_attention_for_batching` doesn't forward the `shared_kv` kwarg from `mlx-vlm` 0.6.2, crashing on any batched request. All Gemma4 models in config have `continuous_batching: false` via their `extra_args`. Do not re-enable until a vllm-mlx upstream fix is confirmed.

**Gemma 4 tool-call format** — Gemma4 emits tool calls in a custom non-JSON format (`<|tool_call>` / `<tool_call|>` delimiters, `<|"|>` string escaping, unquoted keys). Without `--enable-auto-tool-choice --tool-call-parser gemma4` in `extra_args`, the parser does not fire and the model loops forever.

**Gemma 4 MLLM tool history bug (mlx_vlm_wrapper workaround)** — mlx-vlm's `MLLM.stream_chat` silently drops assistant messages where `content=null`, even when `tool_calls` is present. The result is the model sees only the original user message on every turn and loops. **Fix**: `mlx_vlm_wrapper.py` pre-renders tool call + response tokens directly into the assistant message `content` string using native Gemma4 special tokens before forwarding.

**Gemma 4 repetition collapse** — separate structural bug triggered by large system prompts (>10K chars) or many tool definitions (>5-6). Not fixable with `repetition_penalty`. Gemma is excluded from opencode; use via Pi harness or direct curl.

**Qwen3 tool-call parser (AutoToolParser cross-contamination)** — Without `--enable-auto-tool-choice --tool-call-parser qwen`, vllm-mlx falls back to `AutoToolParser("auto")` which tries the Gemma4 format (`<|tool_call>`) *first*. When qwen3.6-moe occasionally generates `<|tool_call>` format (4-bit quantization artifact), the Gemma4 parser fires and fails silently — the tool call content is suppressed, Pi sees the stream end without a `finish_reason`, and reports "output stopped". **Fix**: all Qwen models have `--enable-auto-tool-choice --tool-call-parser qwen` in `extra_args`, which uses `QwenToolParser` directly. The router also normalizes `<|tool_call>` → `<tool_call>` / `<tool_call|>` → `</tool_call>` in assistant message history for non-multimodal models, breaking the feedback loop where Pi stores the malformed content and the model imitates the wrong format.

**Qwen3 temperature=0 prohibition** — vendor docs prohibit `temperature: 0` in thinking mode; minimum 0.6 for `qwen3.6-moe`, 0.2 for `qwen3.6-27b`.

**Router config is loaded once at startup** — `_cfg` in `router.py` is a module-level singleton. Config changes require restarting the router process (`mlx stop` then `mlx start`). There is no hot-reload.

## External integrations

- **opencode**: `~/.config/opencode/opencode_local.json` — `mlx-router` provider (port 6300), Qwen models only
- **Pi harness**: `~/.pi/agent/models.json` — `mlx-router` provider, all models including Gemma4
