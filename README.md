# local-llm

Personal MLX serving stack for Apple Silicon. Runs LLM, image generation, and audio (TTS/STT) models as isolated subprocesses behind a single OpenAI-compatible router on port 6300, with on-demand lazy loading and LRU memory eviction.

## Architecture

```
opencode / Pi harness / Open WebUI / curl
                    │
                    ▼
        ┌─────────────────────┐
        │   FastAPI router    │  :6300  (mlx_server/router.py)
        │  • model alias→ID  │
        │  • injects defaults │
        │  • LRU eviction     │
        └──┬──────┬──────┬───┘
           │      │      │
     :6301 ▼ :6302 ▼ :6303 ▼  :6304 ▼  ...   (sequential, by config order)
    mlx-vlm mlx-lm  mlx-audio  mflux
    (text)  (text)  (audio)    (image)
```

Each model runs in its own subprocess — full process isolation, individual logs. Models are started lazily on the first request; the router's LRU memory manager evicts the least-recently-used model when RAM is tight, so you can freely switch between a large LLM and an image generator without manual unloading.

Each serving engine is a **backend** (`mlx_server/backends.py`) — the single
source of truth for its Python env, packages, wrapper, capabilities, and request
transforms. A model just declares `backend:`. Optional backends (audio, image)
live in **isolated venvs** (`~/.local/share/mlx/venvs/<env>/`) because their
dependencies conflict with the LLM stack; these are auto-provisioned on first use.

## Installation

```bash
# 1. Install the main package (LLM backends)
uv tool install -e . --python 3.12 --prerelease allow

# 2. Copy config template
mkdir -p ~/.config/mlx && cp config/models.yaml ~/.config/mlx/models.yaml

# 3. Start
mlx start
```

Optional audio/image backends are **auto-provisioned on first use** — just add a
model with `backend: mlx-audio` / `backend: mflux` and send a request. To
pre-warm them instead: `mlx install audio`, `mlx install image`, or
`mlx install --all`.

Models are downloaded from HuggingFace on first use. Watch progress with `mlx logs <model>`.

## CLI

```bash
mlx start                     # start all non-lazy models + router
mlx start --model gemma4-12b  # start one specific model
mlx stop                      # stop router, then all models
mlx stop --model qwen3.6-27b  # stop one model
mlx status                    # table: name / pid / port / alive / log path
mlx logs [MODEL]              # tail -f log (default: router)
mlx memory                    # per-model RSS, system RAM, LRU cap status
mlx models                    # list all models from config with defaults
mlx install audio             # pre-warm the audio backend venv (mlx-audio)
mlx install image             # pre-warm the image backend venv (mflux)
mlx install --all             # pre-warm every optional backend venv
```

## Usage

Any OpenAI-compatible client pointed at `http://localhost:6300/v1`. The router translates the friendly model alias to the full HuggingFace ID and injects per-model default hyperparameters (caller values always win).

```bash
# List available models
curl http://localhost:6300/v1/models

# Text generation (hyperparameter defaults injected automatically)
curl -X POST http://localhost:6300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hello"}],"stream":true}'

# Text-to-speech (requires a kokoro entry in models.yaml; audio backend auto-provisions)
curl -X POST http://localhost:6300/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello world","voice":"af_heart"}' \
  --output speech.mp3

# Speech-to-text
curl -X POST http://localhost:6300/v1/audio/transcriptions \
  -F "model=whisper-large" -F "file=@audio.mp3"

# Image generation (requires a flux-schnell entry in models.yaml; image backend auto-provisions)
curl -X POST http://localhost:6300/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"flux-schnell","prompt":"a tabby cat on a red sofa"}' \
  | python3 -c "
import sys, json, base64
d = json.load(sys.stdin)['data'][0]
open('out.png','wb').write(base64.b64decode(d['b64_json']))"
```

## Config reference

```yaml
server:
  host: 127.0.0.1
  port: 6300           # router port
  base_port: 6301      # models assigned sequential ports from here
  max_total_memory_gb: 88.0   # LRU eviction threshold (set to ~70% of RAM)
  min_free_memory_gb: 12.0    # hard floor; refuse to start if RAM would drop below this

models:
  - name: my-llm               # alias used by clients
    source: org/repo-id        # HuggingFace model ID
    backend: mlx-vlm           # mlx-lm | mlx-vlm | mlx-audio | mflux  (default: mlx-lm)
    # multimodal: true         # optional: override image-input support; omit to use
                               #   the backend's own capability (mlx-vlm already accepts images)
    mtp: false                 # true → MTP speculative decoding active
    thinking: false            # true → pass --enable-thinking to mlx-vlm
    lazy: true                 # true → skip on `mlx start`; start on first request
    protected: false           # true → never evicted by LRU manager
    memory_gb: 25              # expected footprint; used for pre-eviction budgeting
    extra_args: []             # verbatim flags appended to the subprocess command
    defaults:
      # LLM generation
      temperature: 0.7
      top_p: 0.95
      top_k: 40
      repetition_penalty: 1.05
      repetition_context_size: 64
      min_p: 0.1
      max_tokens: 16384
      thinking_token_budget: 2048
      # Image generation (backend: mflux)
      steps: 4
      guidance_scale: 3.5
      width: 1024
      height: 1024
      seed: null

  - name: kokoro
    source: mlx-community/Kokoro-82M-bf16
    backend: mlx-audio         # mlx_audio_wrapper.py in the isolated audio venv
    memory_gb: 1
    lazy: true

  - name: flux-schnell
    source: schnell            # mflux alias; use "dev" for FLUX.1-dev
    backend: mflux             # mflux_wrapper.py in the isolated image venv
    memory_gb: 12
    lazy: true
    extra_args: ["--quantize", "8"]
    defaults:
      steps: 4
      width: 1024
      height: 1024
```

## Memory management

The router's `MemoryManager` handles model lifecycle automatically:

- **Lazy start**: a model is started the first time it receives a request
- **LRU eviction**: when total RSS + the new model's projected footprint would exceed `max_total_memory_gb`, the least-recently-used non-protected model is stopped first
- **Hard floor**: if free system RAM after loading would fall below `min_free_memory_gb`, the start is refused with 503

This means you can freely switch between a 41 GB LLM and a 12 GB image generator — the router evicts the LLM when you send your first image request, and reloads it on the next chat request.

```bash
mlx memory   # live RSS per model, cap/floor status, macOS memory pressure
```

## Isolated venvs for audio/image

`mflux` and `mlx-audio` pin different versions of `mlx` than `mlx-vlm` requires, so they cannot share the main tool venv. Each non-`main` backend gets a separate venv, auto-provisioned on first model start (or pre-warmed with `mlx install <env>` / `mlx install --all`):

```
~/.local/share/mlx/
├── pids.json
├── logs/
└── venvs/
    ├── audio/    ← mlx-audio + fastapi/uvicorn  (env of the mlx-audio backend)
    └── image/    ← mflux + fastapi/uvicorn      (env of the mflux backend)
```

The package lists come from each `Backend` in `mlx_server/backends.py`. The wrapper scripts (`mlx_audio_wrapper.py`, `mflux_wrapper.py`) run under the isolated venv's Python and share boilerplate via `_wrapper_base.py`; the main router and all LLM backends continue using the tool venv.

### Adding a new modality (e.g. video)

1. Add a `Backend` subclass to `mlx_server/backends.py` (its `env`, `packages`, `wrapper`, `modalities`) and register it in `BACKENDS`.
2. Write the wrapper script (use `_wrapper_base.py` for the FastAPI/uvicorn boilerplate).
3. Add models with `backend: <name>` to `models.yaml`.

No changes to the router, manager, CLI install, or config dispatch are needed.

## Caveats

### Gemma 4 repetition collapse
All Gemma 4 variants have a structural repetition-collapse bug in the MoE attention, triggered by large system prompts (>10K chars) or many tool definitions (>5–6). Not fixable via `repetition_penalty`. Gemma 4 is excluded from opencode; use via Pi harness or direct curl.

### Qwen3 temperature=0 prohibition
Never use `temperature: 0` with Qwen3 thinking mode — explicitly prohibited in vendor docs. Minimum: 0.6 for MoE thinking, 0.2 for 27B direct generation.

### Router config is loaded once at startup
`_cfg` in `router.py` is a module-level singleton. Config changes require restarting the router (`mlx stop --router && mlx start --router`).

### FLUX.2 model family — what's available and what isn't

Black Forest Labs released FLUX.2 on November 25, 2025. Four variants exist:

| variant | weights | HuggingFace | notes |
|---|---|---|---|
| [pro] | API only | — | managed API, ~$0.03/MP via [bfl.ai](https://bfl.ai/models/flux-2) |
| [flex] | API only | — | developer API tier, ~$0.06/MP |
| [dev] | **open weights** | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) | 32B, text-to-image + editing; ComfyUI day-0 support |
| [klein] | open weights (Apache 2.0) | [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) / [9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | fast, 4-step distilled; **what this stack uses** |

**FLUX.2 [dev] is not supported by mflux** (as of 0.18.0, the latest). It is a fundamentally different architecture — a 32B flow transformer coupled with a Mistral-3 24B vision-language model — and requires its own `Flux2Dev` class that mflux has not yet implemented. Watch [mflux releases](https://github.com/filipstrand/mflux/releases) for when support lands. In the meantime FLUX.2 [dev] runs well in ComfyUI on a 24 GB CUDA GPU.

### mflux loads before port is ready
The image wrapper loads weights synchronously before starting uvicorn, so the port becomes available only after model weights are in memory (~5–30 s). The MemoryManager polls for up to 120 s, which is enough for FLUX 8-bit on any supported Mac.

### Large model downloads stalling mid-way (hf-transfer / hf-xet)
HuggingFace's accelerated downloaders (`hf-transfer` and `hf-xet`) use parallel chunk workers that can silently stall at arbitrary byte offsets for large files. If `mlx logs <model>` shows the download frozen for more than a few minutes, kill the process and retry with both disabled:

```bash
# Kill the stalled model process
mlx stop --model <name>

# Re-download with plain sequential HTTP (resumes from partial .incomplete blob)
HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 \
  ~/.local/share/mlx/venvs/image/bin/python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='<org/repo>', filename='<path/to/file.safetensors>', force_download=True)
"

mlx start --model <name>
```

Plain HTTP uses a single stream with proper Range-header resuming and does not exhibit the chunk-stall behaviour.
