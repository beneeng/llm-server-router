# Local LLM on Apple Silicon: MLX Setup Guide

> Researched May 2026. Personal use — production readiness not a criterion.

---

## The 50K Crash

The `BlockingIOError: [Errno 35] write could not complete without blocking` crash when
using extended context windows (50K tokens, ~17K input) was in LM Studio's **mlx-engine**
Python wrapper layer, not core mlx-lm. It was fixed in mlx-engine PR #212 and the issue
is closed. If you use mlx-lm directly, you never hit this.

The related **long-context performance regression** (mlx-lm Issue #763, opened Jan 2026)
is a separate problem: at 100K+ token contexts, mlx-lm generates tokens at roughly half
the speed of llama.cpp with flash attention. Root cause undiagnosed, issue still open.
For very long contexts (>50K), keep this in mind.

---

## Setup Options

### 1. mlx-lm server

The direct option. Ships with Apple's MLX framework, OpenAI-compatible API.

```bash
pip install mlx-lm
```

**Basic server:**

```bash
mlx_lm.server --model mlx-community/Qwen3.6-27B-MLX-4bit
```

**With MTP (Qwen3.5/3.6 only):**

```bash
mlx_lm.server \
  --model mlx-community/Qwen3.6-27B-MLX-4bit \
  --mtp
```

**With concurrency:**

```bash
mlx_lm.server \
  --model mlx-community/Qwen3.6-27B-MLX-4bit \
  --mtp \
  --max-concurrency 4 \
  --queue-size 20 \
  --queue-timeout 300 \
  --port 8080
```

API at `http://localhost:8080/v1`. Default `--max-concurrency` is 1 — raise it explicitly.

**Active bug with MTP in server mode:** When the system prompt stays the same across
requests but the user prompt changes (warm prefix cache, different suffix), the server
produces 1–9 token completions instead of a full response. PR #1306 is open but not
merged as of May 2026. Workaround: use the non-MTP model variant for server mode, or
use MTP only via `mlx_lm.generate` (CLI), which is not affected.

---

### 2. oMLX

Best option for concurrent agent workloads. The only MLX-based server that genuinely
batches concurrent requests, with SSD-backed KV cache offload for long-context stability.

**Install:**

```bash
# Homebrew
brew tap jundot/omlx https://github.com/jundot/omlx
brew install omlx

# Or: download .dmg from releases page (includes auto-update)
# Or: from source
git clone https://github.com/jundot/omlx.git
cd omlx
pip install -e .
pip install -e ".[mcp]"   # with MCP tool support
```

Requires macOS 15+, Apple Silicon, Python 3.10+.

**Run:**

```bash
# Basic
omlx serve --model-dir ~/models

# With SSD KV cache (recommended for agent workloads with large system prompts)
omlx serve \
  --model-dir ~/models \
  --paged-ssd-cache-dir ~/.omlx/kvcache \
  --hot-cache-max-size 20% \
  --max-concurrent-requests 8

# As a persistent service
brew services start omlx
```

Web admin UI at `http://localhost:PORT/admin` — model management, real-time monitoring,
per-model settings, chat, benchmark.

**The SSD KV cache** is the key differentiator: hot KV blocks live in unified memory,
overflow blocks are offloaded to SSD in safetensors format. On the next request with a
matching prefix (e.g. same large system prompt with tool definitions), blocks are restored
from disk instead of recomputed — even across server restarts. Maintains flat ~20 tok/s
at 32K context where other backends roughly halve.

Concurrency benchmark: 4.14× throughput gain at 8× concurrency vs single-request baseline.

---

### 3. vllm-mlx

Best for high-throughput concurrent serving. Community project built specifically for
Apple Silicon with the MLX backend, not official vLLM (which has CPU-only experimental
Mac support).

```bash
pip install vllm-mlx
pip install vllm-mlx[audio]   # add audio support
brew install espeak-ng          # if using audio
```

**Serve with continuous batching:**

```bash
vllm-mlx serve mlx-community/Qwen3-30B-A3B-Instruct-4bit \
  --port 8000 \
  --continuous-batching
```

**With MTP (Qwen3-Next variants):**

```bash
vllm-mlx serve mlx-community/Qwen3.6-27B-Next-4bit \
  --port 8000 \
  --continuous-batching \
  --mtp
```

**Benchmark your concurrency:**

```bash
vllm-mlx bench-serve --url http://localhost:8000 --concurrency 5
```

OpenAI and Anthropic compatible. Supports vision models (Qwen3-VL, Gemma 3/4, Pixtral,
Llama vision). Also serves embedding models via `--embedding-model`. Works with Claude
Code, Cursor, Continue.

Aggregate throughput on M4: 400+ tok/s at high concurrency. Better suited for parallel
agent workloads than mlx-lm's built-in server.

---

## MTP (Multi-Token Prediction)

Merged in mlx-lm PR #990. Adds `--mtp` to both `mlx_lm.generate` and `mlx_lm.server`.

**Works with models that ship MTP heads:** Qwen3.5, Qwen3.6, Gemma 4.
Does not help with Llama 3, Mistral, or any model not trained with MTP heads.

**CLI generation (no server bug here):**

```bash
mlx_lm.generate \
  --model mlx-community/Qwen3.6-27B-MLX-4bit \
  --mtp \
  --prompt "Write a Python function that..."

# On M1/M2: add --dtype float16 to avoid BF16 software emulation overhead
mlx_lm.generate --model ... --mtp --dtype float16
```

**Server (with caveat):**

```bash
mlx_lm.server --model mlx-community/Qwen3.6-27B-MLX-4bit --mtp
```

See the server-mode bug note under Setup 1 above.

**Measured speedups on M4 Pro, Qwen3.6-27B 4-bit:**

| Config | Speedup |
|---|---|
| MTP depth-1, temp=0 | 1.52× |
| MTP depth-1, temp=0.6 | 1.51× |
| MTP depth-3 (aggressive) | up to 2.6× (~18 tok/s from 7 tok/s baseline) |
| M1/M2 (GPU scheduling constraint) | ~1.1× |

~85% draft token acceptance rate.

**Model conversion note:** The model must have MTP weights preserved during conversion.
If converting from HuggingFace yourself, use the current mlx-lm branch — older versions
stripped MTP layers. Pre-converted models from `mlx-community` on HuggingFace should
already include them.

---

## TurboQuant (PolarQuant + QJL): MLX Status

TurboQuant is Google's ICLR 2026 KV cache compression method. It works in two stages:

1. **PolarQuant** — applies a random orthogonal rotation to key/value vectors so variance
   is distributed uniformly across coordinates, allowing accurate scalar quantization
2. **QJL** — a 1-bit Johnson-Lindenstrauss transform that compresses the residual
   quantization error as a correction signal

Result: ~6× KV cache memory reduction vs FP16, near-lossless, no calibration data needed.

**Current MLX status: not production-ready yet.**

| Repo | What it is | Inference-ready? |
|---|---|---|
| `lingengyuan/qjl-mlx` | Research port, runs validation tests | No — research only |
| `TheTom/turboquant_plus` | Walsh-Hadamard rotation variant | llama.cpp fork, not MLX |
| `AtomicBot-ai/atomic-llama-cpp-turboquant` | llama.cpp fork with `-ctk turbo3` | llama.cpp/Metal only |

The algorithm is validated on Apple Silicon (`qjl-mlx` confirms correctness), but no one
has wired it into the mlx-lm or oMLX inference path as a usable flag yet. Watch
[ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) — it will likely land as a
`--kv-cache-quant` style option.

**If you want TurboQuant + MTP today** and are willing to step outside pure MLX, the
`atomic-llama-cpp-turboquant` fork runs on Metal and combines both:

```bash
# Build with Metal support, then:
llama-server -m Qwen3.6-27B-MTP.gguf \
  -c 32768 -ngl 99 \
  -ctk turbo3 -ctv turbo3 \
  -fa on \
  --spec-type nextn
```

`-ctk turbo3 -ctv turbo3` enables TurboQuant on the KV cache. `--spec-type nextn`
activates Qwen3.6 MTP heads without a separate model file. This is the only way to have
both features at once right now.

---

## Which Setup to Use

| Goal | Setup |
|---|---|
| Get started fast, single-user, MTP | `mlx_lm.server --mtp` |
| Multi-turn agents, large system prompts, SSD KV offload | oMLX |
| 5–10 parallel requests, throughput-oriented | vllm-mlx with `--continuous-batching` |
| TurboQuant + MTP today | `atomic-llama-cpp-turboquant` (Metal, not MLX) |
| TurboQuant in pure MLX | Not ready yet |

---

## Known Active Issues (May 2026)

| Issue | Affects | Status |
|---|---|---|
| MTP 1-token completion in server mode | mlx-lm + Qwen3.6 --mtp | PR #1306 open |
| 50% performance drop at 100K+ context | mlx-lm | Issue #763 open |
| Prefix cache broken for hybrid models (SSM/Mamba) | mlx-lm | Open |
| 50K crash (BlockingIOError) | mlx-engine / LM Studio | Fixed (PR #212) |

---

## Sources

- [mlx-engine Issue #168 — 50K crash, fixed](https://github.com/lmstudio-ai/mlx-engine/issues/168)
- [mlx-lm Issue #763 — Long context perf regression](https://github.com/ml-explore/mlx-lm/issues/763)
- [mlx-lm PR #990 — Native MTP merged](https://github.com/ml-explore/mlx-lm/pull/990)
- [mlx-lm Issue #1292 — MTP server bug](https://github.com/ml-explore/mlx-lm/issues/1292)
- [mlx-lm SERVER.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)
- [oMLX GitHub](https://github.com/jundot/omlx)
- [vllm-mlx GitHub](https://github.com/waybarrios/vllm-mlx)
- [qjl-mlx — MLX TurboQuant research port](https://github.com/lingengyuan/qjl-mlx)
- [AtomicBot TurboQuant + MTP fork](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant)
- [TurboQuant paper — arXiv 2504.19874](https://arxiv.org/html/2504.19874v1)
- [Medium — 18 tok/s from 27B model with MTP](https://medium.com/@vinoth.lingam333/i-got-18-tok-s-from-a-27b-model-on-a-macbook-mlx-native-mtp-speculative-decoding-on-apple-5e1211a06d0c)
