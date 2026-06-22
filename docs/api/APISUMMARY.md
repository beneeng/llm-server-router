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
| `enable_thinking` | Router moves into `chat_template_kwargs` for mlx-lm |

**Image input** (VLM only): Use `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` content blocks. Text-only models return 400.

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

## Image Generation

### Synchronous (all backends)

```bash
curl -X POST http://localhost:6300/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"flux2-klein-fast","prompt":"a red cat"}' \
  | python3 -c "import sys,json,base64; open('out.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"
```

### SSE streaming (mflux2 only — append `?sse=true`)

```bash
curl -X POST "http://localhost:6300/v1/images/generations?sse=true" \
  -H "Content-Type: application/json" \
  -d '{"model":"flux2-klein-fast","prompt":"a red cat"}'
```

SSE events (each line: `data: <json>\n\n`):

| Event | Format |
|-------|--------|
| progress | `{"type":"progress","step":N,"total":N,"percent":N,"eta_seconds":N}` |
| done | `{"type":"done","created":N,"data":[{"b64_json":"...","revised_prompt":"..."}]}` |
| error | `{"type":"error","message":"..."}` |

### Image-to-image editing (mflux2)

```bash
curl -X POST http://localhost:6300/v1/images/edits \
  -H "Content-Type: application/json" \
  -d '{"model":"flux2-klein-fast","prompt":"as an oil painting","images":["<base64>"],"image_strength":0.6}'
```

`/v1/images/edits` is an alias for `/v1/images/generations` with `edit_mode=true`.

## Batch mode (mflux2, n>1)

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
