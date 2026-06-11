# local-llm

Personal MLX LLM serving stack for Apple Silicon. Runs multiple models as isolated `vllm-mlx` processes behind a single OpenAI-compatible router on port 6300, with per-model hyperparameter defaults baked in.

## Architecture

```
opencode / Pi harness / Open WebUI / curl
              │
              ▼
    ┌─────────────────────┐
    │  FastAPI router      │  :6300  (mlx_server/router.py)
    │  • model alias→ID    │
    │  • injects defaults  │
    └──────┬──────┬───────┘
           │      │  ...
    :6301  ▼  :6302 ▼  :6303 ▼  ...
  vllm-mlx  vllm-mlx  vllm-mlx
  gemma4-   qwen3.6-  gemma4-
  moe       moe       12b
```

Each model runs in its own `vllm-mlx serve` subprocess — full process isolation, independent Metal contexts, individual logs. The router on :6300 translates friendly aliases (`gemma4-moe`) to full HuggingFace IDs before proxying, and merges per-model default hyperparameters into every request that omits them.

## Tech stack

| Layer | Tool |
|---|---|
| Inference backend | [vllm-mlx](https://github.com/waybarrios/vllm-mlx) 0.4.0rc1 — continuous batching, OpenAI + Anthropic API, native MLX/Metal |
| Router | FastAPI + httpx async proxy |
| CLI | Typer + Rich |
| Install | `uv tool install -e .` → global `mlx` command |
| Config | `~/.config/mlx/models.yaml` |
| State | PIDs at `~/.local/share/mlx/pids.json`, logs at `~/.local/share/mlx/logs/` |

## Models

| Alias | HuggingFace ID | Port | Notes |
|---|---|---|---|
| `gemma4-moe` | `mlx-community/gemma-4-26b-a4b-it-nvfp4` | 6301 | 26B MoE, multimodal, thinking |
| `qwen3.6-moe` | `mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit` | 6302 | 35B MoE, coding, thinking + MTP head |
| `gemma4-12b` | `mlx-community/gemma-4-12B-it-nvfp4` | 6303 | 12B dense, multimodal, thinking |
| `qwen3.6-27b` | `mlx-community/Qwen3.6-27B-nvfp4` | 6304 | 27B dense, direct coding |
| `gemma4-moe-8bit` | `mlx-community/gemma-4-26b-a4b-it-8bit` | 6305 | 8-bit experiment, loop stability |

## Installation

```bash
# 1. Install
cd ~/workspace/local_llm
uv tool install -e . --python 3.12 --prerelease allow

# 2. Install config
mkdir -p ~/.config/mlx
cp config/models.yaml ~/.config/mlx/models.yaml

# 3. Start
mlx start
```

Models are downloaded from HuggingFace on first start (~13–26 GB each). Monitor progress:

```bash
mlx logs qwen3.6-moe
```

## CLI

```bash
mlx start                    # start all models + router (waits for ports)
mlx start --model gemma4-12b # start one model
mlx stop                     # stop router first, then all models
mlx stop --model qwen3.6-27b # stop one model
mlx status                   # table: name / pid / port / alive / log path
mlx logs [MODEL]             # tail -f the log (default: router)
mlx models                   # list models from config with their defaults
```

## Usage

Any OpenAI-compatible client pointed at `http://localhost:6300/v1`:

```bash
# list models
curl http://localhost:6300/v1/models

# chat (hyperparameter defaults injected automatically)
curl -X POST http://localhost:6300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-moe","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

## Hyperparameter defaults

Defaults from the repetition-loop research study, applied per-model at the router layer. Caller-supplied values always override.

| Model | temp | top_p | top_k | rep_pen | max_tokens |
|---|---|---|---|---|---|
| gemma4-* | 0.9 | 0.95 | 64 | 1.0 | 4096 |
| qwen3.6-moe | 0.6 | 0.95 | 20 | 1.05 | 8192 |
| qwen3.6-27b | 0.2 | 0.9 | 20 | 1.05 | 4096 |

## Config reference

```yaml
server:
  host: 127.0.0.1
  port: 6300          # router port
  base_port: 6301     # models get sequential ports from here

models:
  - name: my-model            # alias used by clients
    source: org/repo-id       # HuggingFace model ID
    multimodal: false         # adds --mllm flag (mlx-vlm backend)
    continuous_batching: true # set false for Gemma4 (vllm-mlx bug)
    reasoning_parser: qwen3   # gemma4 | qwen3 | deepseek_r1 | ...
    extra_args: []            # verbatim flags appended to vllm-mlx serve
    defaults:
      temperature: 0.7
      top_p: 0.95
      top_k: 40
      repetition_penalty: 1.05
      repetition_context_size: 64
      max_tokens: 4096
```

## Caveats

### Gemma 4 repetition collapse
All Gemma 4 variants have a structural repetition-collapse bug in the MoE attention. It is **not fixable via hyperparameters** — `repetition_penalty` has no effect. Triggers:
- Large system prompts (>10K chars)
- Many tool definitions in context (>5–6 tools)
- Consecutive user messages without interleaved assistant turns

**Consequence:** Gemma 4 is excluded from opencode and any agentic harness that injects tool schemas. Use it via Pi harness or Open WebUI with a short system prompt and no tools.

### vllm-mlx 0.4.0rc1 MLLM tool history bug (router workaround active)
`MLLM.stream_chat` drops assistant messages with `content=null` even when `tool_calls` is present, so tool-call history is invisible to the model and it loops forever. The router pre-renders tool call + response tokens into the assistant content string as native Gemma4 special tokens before forwarding. See [`TODO.md`](TODO.md) for revert instructions once the upstream fix lands.

### vllm-mlx 0.4.0rc1 continuous batching bug
`patch_gemma4_attention_for_batching` doesn't forward the `shared_kv` argument introduced in mlx-vlm 0.6.2. All Gemma 4 models run with `continuous_batching: false` until this is fixed upstream.

### Qwen3 thinking mode
Never use `temperature: 0` with Qwen3 thinking mode — explicitly prohibited in vendor docs and causes deterministic repetition loops. Minimum recommended: 0.6 for thinking mode, 0.2 for direct generation.

### Models load lazily
`vllm-mlx` accepts requests immediately but returns empty/error responses while the model is loading into memory. Watch `mlx logs <name>` for the ready signal.

### Partial downloads after model switches
When switching quantization variants, incomplete downloads for the old model remain in `~/.cache/huggingface/hub/`. Safe to delete once the new variant is confirmed working.
