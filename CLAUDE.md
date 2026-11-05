# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Install / reinstall

```bash
uv tool install -e . --python 3.12
```

This installs the `mlx` CLI globally into a uv-managed venv at `~/.local/share/uv/tools/mlx-server/`. `--prerelease allow` is no longer required (dropped as of the mlx-vlm 0.6.13 upgrade — re-check with `uv lock --dry-run` after future dependency bumps in case a transitive dep goes prerelease-only again).

**This project also has a local `.venv/` (created by `uv sync`, tracking the same `uv.lock`) and a `.envrc` that activates it.** If direnv is active, `.venv/bin` is prepended to `PATH` ahead of the `uv tool install` shim at `~/.local/bin/mlx` — so inside this directory, `mlx` actually resolves to `.venv/bin/mlx`, not the globally tool-installed one. **Both must be kept in sync**: after bumping deps, run both `uv tool install -e . --python 3.12 --reinstall` (for shells without direnv / outside this directory) and `uv sync` (for the local `.venv` direnv picks up). Check `which mlx` if version drift is suspected.

One-time config setup:
```bash
mkdir -p ~/.config/mlx && cp config/models.example.yaml ~/.config/mlx/models.yaml
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

## Speculative-decoding benchmarks

```bash
scripts/bench_draft_variants.py --model qwen3.8-27b \
    --variant "baseline:" --variant "dflash2:--draft-model incoai/Qwen3.8-27B-DFlash2 --draft-kind dflash"
scripts/bench_speculative.py --compare /tmp/bench_speculative.jsonl
```

Each variant gets a cold backend on port 6399 (the router and the model's own
port are untouched) because mlx-vlm reads drafter settings from the environment
at startup, not per request. Base argv comes from `ModelConfig.server_args()`;
a variant is only the extra flags, and mlx-vlm's argparse takes the last
occurrence, so `--draft-model` can be overridden by appending it.

Judge results by **tokens/round and acceptance**, not tok/s alone: acceptance is
a model property and therefore independent of machine load. mlx-vlm returns both
in the non-streaming response's `timings` object (llama.cpp field names).
Raising `--draft-block-size` usually *loses*: deeper drafts cost extra drafter
forwards that a falling acceptance rate can't pay for.

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
| `mlx-vlm-dsv4` | `mlx_vlm_dsv4_wrapper.py` | `main` (tool venv) |
| `mlx-audio` | `mlx_audio_wrapper.py` | `audio` → `~/.local/share/mlx/venvs/audio/` |
| `mflux` | `mflux_wrapper.py` | `image` → `~/.local/share/mlx/venvs/image/` |
| `mflux2` | `mflux2_wrapper.py` | `image` → `~/.local/share/mlx/venvs/image/` |
| `mflux-ideogram` | `mflux_ideogram_wrapper.py` | `image` → `~/.local/share/mlx/venvs/image/` |

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
rejected with 400 before the backend is touched. This same gate covers video:
the router's non-text-content check does not distinguish `image_url` from
`video_url`/`video`/`input_video` blocks, and there is no separate
`accepts_video_input` flag — any `mlx-vlm` model can receive a `video_url`
block today. Video handling itself lives entirely in the vendored `mlx-vlm`
server (untouched by `mlx_vlm_wrapper.py`): `resolve_video_inputs()` checks
whether the model's processor has a native `video_processor`; if not, it
decodes the video with OpenCV and falls back to sending up to 16
evenly-spaced frames (default 2 fps) as ordered images instead. This is not
visible anywhere in this repo's own code — check `mlx_vlm/generate/video.py`
and `mlx_vlm/server/openai.py` in the installed package if debugging video
requests.

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

**Dashboard and request metrics (`dashboard.py`, `metrics_store.py`, `metrics_poller.py`):** a read-only,
GET-only web UI at `/_dashboard/` (static `mlx_server/dashboard/index.html`, no external requests). Privacy is
structural: the router never reads response bytes for metrics. A daemon thread polls each running mlx-vlm backend's own
`GET /v1/metrics` (content-free per-request envelopes, last 32) every 5 s plus once in `MemoryManager.before_stop`
right before an eviction, and `MetricsStore` copies only allow-listed numeric/enum columns into SQLite
(`~/.local/share/mlx/metrics.db`, dedup on model+ts+tokens, newest `metrics_retention_rows` kept, default 50 000). Text
columns only accept short enum-like tokens, so a future mlx-vlm text field cannot leak. mlx-lm/audio/image backends have
no such endpoint; the proxy only tallies POSTs and errors per model (`count_request`/`count_error`, flushed by the
poller). The dashboard router is included BEFORE the catch-all proxy route; adding a write route under `/_dashboard`
is forbidden (a test enforces GET/HEAD only). `prefill_tok_s` is inflated on APC hits (envelope has no cached-token
count). The feature is opt-in: `dashboard_enabled` defaults to false and gates UI, API, poller and counting. Tests: `test_metrics_store.py`,
`test_metrics_poller.py`, `test_dashboard.py` (includes a prompt-marker privacy test), `test_router_counting.py`.

**Keep-awake (`mlx_server/keepawake.py`):** every proxied request (and every
streamed chunk) calls `touch()`, which holds a `caffeinate -d -i` assertion
for `GRACE_SECONDS` (90s) past the last activity, then releases it. Prevents
macOS's idle-timer-driven screen lock during/just-after remote/agent
sessions — a locked screen has been triggering a fan-control bug that
overheats the M-series core mid-generation (force-shutdown required). Does
not override an explicit lock or lid-close sleep.

**Memory management (`mlx_server/memory.py`):**
- `MemoryManager.ensure_model_ready(name)` is called on every proxied request
- Evicts LRU (least-recently-used, non-protected, not started in last 60 s) models until memory is within budget
- Graceful eviction: the router calls `mark_busy`/`mark_idle` around each proxied request (including for the duration of a streaming response). An LRU victim that's currently busy is not killed immediately — eviction releases `MemoryManager._lock` and waits on that model's idle `asyncio.Event` for up to `ServerConfig.evict_wait_timeout_seconds` (default 300 s), then force-stops it anyway if it still hasn't gone idle. Idle candidates are always preferred over busy ones.
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

## KV caching (mlx-vlm Automatic Prefix Caching)

All `mlx-vlm`-backed models get Automatic Prefix Caching (APC) enabled via
`MlxVlm.env_vars()` in `backends.py` (`APC_ENABLED=1`,
`APC_DISK_PATH=~/.local/share/mlx/apc/<model-name>/`,
`APC_DISK_MAX_GB=100`), merged into the subprocess env by `manager.py`'s
`start_model()`. APC is hash-based, block-level KV-cache reuse across
requests (LRU + refcount eviction), with the disk tier letting a warm cache
survive that model's process being stopped/evicted — not just concurrent
requests within one run. `mlx-lm`-only models don't get this (`MlxLm` stays
on the base-class no-op `env_vars`); `mlx-lm`'s own server has a separate,
simpler multi-slot `LRUPromptCache` (`--prompt-cache-size`,
`--prompt-cache-bytes`), currently left on defaults.

Notes:
- **`APC_DISK_MAX_GB` must stay set.** mlx-vlm defaults this to `0` (unbounded)
  when unset — the disk tier then never evicts and grows until the disk
  fills (this happened once: 1.2TB across 7 models on a 1.8TB disk). The
  100GB/model cap in `MlxVlm.APC_DISK_MAX_GB` is a class constant, not
  per-model config; bump it there if a wider cache is needed, but check free
  disk space against `# models × cap` first.
- **Time-based expiry is separate from the size cap** (`mlx_server/apc_gc.py`).
  mlx-vlm's own eviction is purely size-triggered — it has no TTL. A daemon
  thread started from `router.py`'s `lifespan` sweeps `~/.local/share/mlx/apc/`
  hourly and deletes shard files whose `mtime` is older than 48h
  (`apc_gc.APC_TTL_SECONDS`). mlx-vlm bumps a shard's `mtime` on every read,
  so this is really a "last used," not "created," cutoff — a hot entry keeps
  resetting its own clock. Only runs while the router process is up (`mlx
  start --model X` alone won't trigger it). Manual/dry-run entry point:
  `mlx gc [--dry-run] [--ttl-hours N]`.
- `/v1/cache/stats` and `/v1/cache/reset` exist per-model on the backend's
  own port (`http://127.0.0.1:<model-port>/...`, look up the port via `mlx
  status`) — not proxied through the router.
- Warm-started completions can diverge slightly from cold ones at near-tied
  logits (flash-attention batch/tile-shape non-invariance, not a bug —
  warm-to-warm repeats are deterministic).
- KV-cache quantization (`--kv-bits`, `--kv-quant-scheme`) is a separate,
  independently-toggleable `extra_args` flag. Upstream's APC v1 had a caveat
  that KV-quantized cache paths skip APC — verify this still applies (or
  doesn't) on the installed `mlx-vlm` version before combining the two on a
  given model.

## Known bugs and constraints

**Gemma 4 tool-call format** — Gemma4 emits tool calls in a custom non-JSON format (`<|tool_call>` / `<tool_call|>` delimiters, `<|"|>` string escaping, unquoted keys). Without `--enable-auto-tool-choice --tool-call-parser gemma4` in `extra_args`, the parser does not fire and the model loops forever.

**Gemma 4 MLLM tool history bug (mlx_vlm_wrapper workaround)** — mlx-vlm's `MLLM.stream_chat` silently drops assistant messages where `content=null`, even when `tool_calls` is present. The result is the model sees only the original user message on every turn and loops. **Fix**: `mlx_vlm_wrapper.py` pre-renders tool call + response tokens directly into the assistant message `content` string using native Gemma4 special tokens before forwarding.

**DeepSeek-V4 (`mlx-vlm-dsv4`)** - three separate traps, all hit on the M3 deployment:
- *Garbage output after a silent load.* mlx-vlm 0.7.0 does not remap `embed_tokens.scales/biases` and `hc_head.*` for the mlx-community 4-bit checkpoint (fixed in 0.7.11, hence the pin). `mlx_vlm_wrapper.py` loads non-strictly and hid the error, so the server answered with noise. `mlx_vlm_dsv4_wrapper.py` deliberately keeps strict loading. If this model ever prints noise again, run `python -m mlx_vlm.server` without our wrapper first: a `Received N parameters not in model` error means a version problem.
- *Stub chat template.* The repo's `chat_template.jinja` has no tools, no `tool` role, no `tool_calls` (a request with `tools` is only ~11 prompt tokens). The real format is DeepSeek's `encoding_dsv4.py`, vendored unchanged in `mlx_server/dsv4/` (MIT, see `THIRD_PARTY_NOTICES.md`; re-vendor instead of editing). The wrapper swaps `DeepseekV4Processor.apply_chat_template` for it.
- *No DSML parser upstream.* mlx-vlm cannot infer a parser from the stub template, so the backend's `transform_request` sets `tool_parser: deepseek_v4_dsml`, registered by the wrapper in `mlx_vlm.tools.registry`. These hooks are tied to mlx-vlm 0.7.11 (`mlx_vlm.tools.*`); `tests/test_dsv4.py` fails loudly on a bump that moves them. Tokenizer adds no BOS, so the encoder's own BOS is the only one.

**Gemma 4 repetition collapse** — separate structural bug triggered by large system prompts (>10K chars) or many tool definitions (>5-6). Not fixable with `repetition_penalty`. Gemma is excluded from opencode; use via Pi harness or direct curl.

**Qwen3 tool-call parser (AutoToolParser cross-contamination)** — Without `--enable-auto-tool-choice --tool-call-parser qwen`, vllm-mlx falls back to `AutoToolParser("auto")` which tries the Gemma4 format (`<|tool_call>`) *first*. When qwen3.6-moe occasionally generates `<|tool_call>` format (4-bit quantization artifact), the Gemma4 parser fires and fails silently — the tool call content is suppressed, Pi sees the stream end without a `finish_reason`, and reports "output stopped". **Fix**: all Qwen models have `--enable-auto-tool-choice --tool-call-parser qwen` in `extra_args`, which uses `QwenToolParser` directly. The router also normalizes `<|tool_call>` → `<tool_call>` / `<tool_call|>` → `</tool_call>` in assistant message history for non-multimodal models, breaking the feedback loop where Pi stores the malformed content and the model imitates the wrong format.

**Qwen3 temperature=0 prohibition** — vendor docs prohibit `temperature: 0` in thinking mode; minimum 0.6 for `qwen3.6-moe`, 0.2 for `qwen3.6-27b`.

**Router config is loaded once at startup** — `_cfg` in `router.py` is a module-level singleton. Config changes require restarting the router process (`mlx stop` then `mlx start`). There is no hot-reload.

## External integrations

- **opencode**: `~/.config/opencode/opencode_local.json` — `mlx-router` provider (port 6300), Qwen models only
- **Pi harness**: `~/.pi/agent/models.json` — `mlx-router` provider, all models including Gemma4
