# MLX Router API — Quick Reference

**Single endpoint:** `http://localhost:6300`

The router proxies requests to model backends (LLM, VLM, TTS, STT, image gen).
Every request must include a `model` field identifying which model to use.

## Model discovery

```bash
curl http://localhost:6300/v1/models
# → {"object":"list","data":[{"id":"qwen3.6-27b","object":"model","owned_by":"mlx-community"},...]}
```

## Chat (LLM / VLM)

```bash
curl -X POST http://localhost:6300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

| Parameter | Notes |
|-----------|-------|
| `model` | Alias from `models.yaml` |
| `messages` | Standard OpenAI format; image URLs for VLMs |
| `stream` | `true` → SSE `data: ...\n\n` |
| `thinking_token_budget` | Router renames to `thinking_budget` for mlx-vlm; stripped for MTP models |
| `enable_thinking` | Router moves into `chat_template_kwargs` for mlx-lm; native for mlx-vlm and `mlx-vlm-dsv4` (selects `<think>` vs `</think>` mode) |
| `tools` | OpenAI function tools. For `mlx-vlm-dsv4` (DeepSeek-V4) the router adds `tool_parser: deepseek_v4_dsml` unless the caller sets one |

**DeepSeek-V4** (`backend: mlx-vlm-dsv4`, text-only): prompts are built with DeepSeek's reference encoder (the checkpoint's own chat template is a stub without tools). Tool calls come back as standard OpenAI `tool_calls` (parsed from DSML, also when streaming); reasoning is returned in `reasoning_content`. `enable_thinking: false` switches to non-thinking mode; `reasoning_effort: "high"|"max"` is forwarded to the encoder.

**Image input** (VLM only): Use `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` content blocks. Text-only models return 400.

**Video input** (mlx-vlm only, same gate as image input): Use `{"type":"video_url","video_url":{"url":"file:///path/to/clip.mp4"}}` content blocks. mlx-vlm decodes the video itself — native processors use it directly, others get up to 16 frames auto-sampled (2 fps default) and sent as images.

## Text-to-Speech

```bash
curl -X POST http://localhost:6300/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello world","voice":"af_heart"}' \
  --output speech.mp3
```

Returns raw audio bytes (`audio/mpeg` by default). Supports `response_format`: mp3, wav, flac, pcm.

## Speech-to-Text

```bash
curl -X POST http://localhost:6300/v1/audio/transcriptions \
  -F "model=whisper-large" -F "file=@audio.mp3"
```

Returns `{"text":"..."}`. Use `response_format=text` for plain text.

Two models available: `whisper-large` (whisper-large-v3, best accuracy) and `whisper-turbo` (whisper-large-v3-turbo, pruned decoder — faster, marginally less accurate).

## Image Generation

### Synchronous (all backends)

```bash
curl -X POST http://localhost:6300/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"flux2-klein-fast","prompt":"a red cat"}' \
  | python3 -c "import sys,json,base64; open('out.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"
```

### SSE streaming (mflux2 + mflux-ideogram — append `?sse=true`)

```bash
curl -X POST "http://localhost:6300/v1/images/generations?sse=true" \
  -H "Content-Type: application/json" \
  -d '{"model":"flux2-klein-fast","prompt":"a red cat"}'
```

SSE events (each line: `data: <json>\n\n`):

| Event | Format | Backends |
|-------|--------|---------|
| progress | `{"type":"progress","step":N,"total":N,"percent":N,"eta_seconds":N}` | mflux2 only |
| done | `{"type":"done","created":N,"data":[{"b64_json":"...","revised_prompt":"..."}]}` | all |
| error | `{"type":"error","message":"..."}` | all |

Note: mflux-ideogram emits only `done`/`error` (no per-step `progress` — Ideogram 4 has no step callbacks).

### Ideogram 4 (mflux-ideogram backend)

```bash
curl -X POST http://localhost:6300/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"ideogram-4","prompt":"a photorealistic red cat on a rooftop at sunset"}'
  | python3 -c "import sys,json,base64; open('out.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"
```

| Parameter | Notes |
|-----------|-------|
| `preset` | `V4_TURBO_12` / `V4_DEFAULT_20` (default) / `V4_QUALITY_48` — takes precedence over `steps` |
| `steps` | 12/20/48 auto-map to the matching preset schedule; other values use flat guidance=7.0 |
| `negative_prompt` | Ignored (not supported by Ideogram 4 API) |

No img2img or edit support. Batch (`n>1`) generates images sequentially.

### Image-to-image editing (mflux2 only)

```bash
curl -X POST http://localhost:6300/v1/images/edits \
  -H "Content-Type: application/json" \
  -d '{"model":"flux2-klein-fast","prompt":"as an oil painting","images":["<base64>"],"image_strength":0.6}'
```

`/v1/images/edits` is an alias for `/v1/images/generations` with `edit_mode=true`.

## Batch mode (mflux2: tensor-batched; mflux-ideogram: sequential)

```bash
curl -X POST http://localhost:6300/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"flux2-klein-fast","prompt":"a red cat","n":4}'
```

All images generated in a single tensor-batched denoising loop. With `?sse=true`, batch progress events reflect shared steps; the done event contains all N images. **Cannot be aborted mid-generation.**

## Input fields injected by router (per-model defaults)

The router merges `ModelConfig.defaults` from `models.yaml` on every request.
Caller values take precedence.

**LLM defaults:** temperature, top_p, top_k, repetition_penalty, repetition_context_size, min_p, max_tokens, thinking_token_budget
**Image defaults:** steps, guidance_scale, width, height, seed

## Error responses

| Status | Meaning |
|--------|---------|
| 400 | Missing model, image on text-only model, bad request |
| 404 | Unknown model alias |
| 503 | Backend failed to start or disconnected mid-stream |

## For the complete spec

Read `docs/api/openapi.yaml` — it covers every endpoint, schema, and edge case.
