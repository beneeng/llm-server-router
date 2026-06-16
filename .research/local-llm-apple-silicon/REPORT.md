# Local LLM Runtime on Apple Silicon: Research Report (May 2026)

**Depth:** Medium | **Sources:** ~25 verified | **Bias classification:** Product-comparison (lightweight bias flags applied)

---

## Executive Summary

The Apple Silicon local LLM landscape has changed dramatically in early 2026. MLX has won the short-context speed race and is now the backend underlying Ollama itself. However, for your specific use case — research and coding agents, moderate concurrency (2–10 parallel requests), and very long contexts — **llama-server (llama.cpp) is still the most mature and controllable backend**, with MLX catching up fast. The right answer depends on what you optimize for: raw short-context speed → MLX. Long context, concurrency, stability → llama.cpp. Both together, minimal config → LM Studio 0.4+. Maximum concurrency with batching → oMLX or vLLM-MLX.

The "polar coordinate KV cache turbo quant" you mentioned is **Google's TurboQuant** (ICLR 2026) — a real, significant technique. Community implementations exist for both MLX and llama.cpp, but it is not yet in any mainline tool.

---

## 1. MLX and mlx-lm: Current State and Bugs

### The "70k Token Bug" — What It Actually Is

There is no single GitHub issue titled "70k context window bug." What exists is a cluster of related long-context problems in MLX:

**1a. BlockingIOError at large contexts (the bug you likely heard about)**¹

The mlx-engine (used by LM Studio's MLX backend) throws `BlockingIOError: [Errno 35] write could not complete without blocking` when combining a large context window (~50,000 tokens) with a long input (~17,000 tokens) on Mac Studio M3 Ultra. The app crashes mid-generation. A fix PR (#212) was submitted, but final resolution status is unclear. This is specific to the mlx-engine Python layer, not core MLX itself.

**1b. Performance cliff at long context (open, Jan 2026)**²

MLX token generation is **consistently ~50% slower than llama.cpp** at very long contexts. Measured at 146K context: MLX = 5.95 tok/s vs llama.cpp = 12.12 tok/s. The root cause is undiagnosed — the issue notes llama.cpp benefits from flash attention at long context but MLX does not appear to recover the same way. This issue was opened January 2026 and **remains open with no assignee**.

**1c. Gemma 3 context cap at 4096 (model conversion bug)**³

Models that support 128K context officially get converted with a 4096 cap in mlx_lm conversion. Model-specific; may be fixed for some architectures.

**Bottom line on the 70k bug:** It exists in multiple forms — a hard crash in the mlx-engine at 50K+ context, and an open performance regression at very long contexts. At a 70K token window, you are likely to hit both. **For long-context workloads, use llama.cpp with flash attention instead.**

### MLX-LM: Current Capabilities (2026)

| Feature | Status |
|---|---|
| Quantization formats | 4-bit, 8-bit, mixed precision, MLX-native formats |
| Flash attention | Supported (prefill acceleration) |
| Speculative decoding | Supported — MTP for Qwen3.6 and Gemma 4; EAGLE-3 prototype in community |
| Prefix caching | Broken for hybrid architectures (sliding window, SSM/Mamba)⁴ |
| Server mode | OpenAI-compatible HTTP API, continuous batching |
| MTP heads | Qwen3.6, Gemma 4 supported |
| Fine-tuning | LoRA on-device supported |

### Known Active Bugs (as of May 2026)

- **KV cache cross-contamination (mlx_lm.server, fixed in 0.31.x)**⁵: At 16+ concurrent requests, responses from one prompt leaked into another's output. Agent Score dropped from 0.727 to 0.456 under concurrent load. The issue was closed (PR #976), but concurrent server reliability should be tested before relying on it for agent workloads.
- **MTP server bug with Qwen3.6**⁶: mlx_lm.server returns 1-token completions on Qwen3.6 MTP variants when system prompt is reused with a different user prompt.
- **Long-context performance regression**²: 50% slower than llama.cpp at 100K+ context. Open issue.
- **Prefix cache broken for hybrid models**⁴: SSM/Mamba-based models get their entire KV cache invalidated on every turn. Painful for multi-turn agent workloads.

---

## 2. llama.cpp Metal Backend

### Current State

llama.cpp treats Apple Silicon as a first-class platform via the Metal backend. As of 2026:⁷

- **Flash attention on Metal**: Enabled with `-fa` / `--flash-attn`. Required for long-context inference — without it, performance and memory usage degrade badly at 32K+ context.
- **KV cache quantization**: `--cache-type-k q8_0 --cache-type-v q8_0` is the recommended Apple Silicon config. Halves KV cache memory. **Requires flash attention to be enabled** — without `-fa`, dequantization overhead makes it slower. Typical memory saving: a 70B model at 32K context goes from needing 40GB+ KV cache to ~8GB.⁸
- **Concurrent requests**: llama-server supports `--parallel N` for continuous batching. The three knobs: `--parallel`, `--batch-size`, `--ubatch-size`. Set `-b 2048 -ub 2048` on Apple Silicon for best prefill via Metal matrix parallelism.⁹
- **Speculative decoding**: Supported on Metal. Prompt lookup decoding has zero memory overhead. Full draft-model speculative decoding works but requires fitting two models in RAM.
- **OpenAI-compatible API**: Yes — `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, streaming, tool calling.

### Performance vs MLX

| Metric | llama.cpp Metal | MLX |
|---|---|---|
| Short context tok/s | ~150 tok/s | ~230 tok/s |
| Long context (100K+) | **~12 tok/s** | **~6 tok/s** (50% slower) |
| Concurrency handling | Mature (continuous batching) | Buggy history, improved |
| Memory efficiency | Excellent (KV quantization) | Good |
| Model format support | GGUF (broadest) | MLX native |
| Stability | Production-grade | Still maturing |

⁷'¹⁰

**Recommended flags for Apple Silicon:**
```
llama-server -m model.gguf \
  -ngl 99 \          # full GPU offload
  -fa \              # flash attention (REQUIRED for long context + KV quant)
  -c 65536 \         # context window
  -b 2048 -ub 2048 \ # batch sizes for Metal parallelism
  --cache-type-k q8_0 --cache-type-v q8_0 \  # KV quantization
  --parallel 4 \     # concurrent request slots
  --host 0.0.0.0 --port 8080
```

**Known Metal bug**: Issue #21450 — Metal fails mixed quantized KV (e.g., q4_0 keys + q8_0 values) when flash attention is unavailable for that architecture. Stick to symmetric q8_0/q8_0 to avoid this.

---

## 3. Advanced Features

### 3.1 TurboQuant / PolarQuant (= what you called "polar coordinate KV cache")

[VENDOR-CLAIM note: primary source is Google Research]

**What it is**: TurboQuant is a Google Research algorithm published at ICLR 2026.¹¹ It is a two-stage KV cache quantization method:

1. **PolarQuant**: Applies a random orthogonal rotation to each key/value vector. This redistributes variance uniformly across all coordinates so each can be quantized accurately with a simple scalar quantizer. This is the "polar coordinate" part you heard about.
2. **QJL** (Quantized Johnson-Lindenstrauss): A 1-bit transform that compresses the residual quantization error as a correction signal.

**Result**: ~6× KV cache memory reduction vs FP16 with near-lossless quality. Llama 3.1 70B at 128K context drops from ~40 GB to ~7.5 GB KV cache.

**Data-oblivious**: No calibration data, no training, no model-specific tuning — apply at inference time to any transformer.

**Apple Silicon support status**:

| Implementation | Maturity | Apple Silicon |
|---|---|---|
| `TheTom/llama-cpp-turboquant` | Community fork | Yes, Metal flag `DGGML_METAL=ON` |
| `AtomicBot-ai/atomic-llama-cpp-turboquant` | Fork, adds MTP + TurboQuant | Yes |
| `lingengyuan/qjl-mlx` | Community, first native MLX port | Yes, Apple Silicon native |
| Mainline llama.cpp | Discussion open (#20969) | Not yet |
| Mainline mlx-lm | Not tracked | Not yet |

¹²

**Practical note**: On a Mac Mini M4 Pro, TurboQuant-compressed Llama 70B Q4_K_M (40GB weights) fits in 64GB with 32K context at 6.3 tok/s. This enables models that wouldn't otherwise fit. The community llama.cpp fork is the most production-tested option right now.¹³

### 3.2 Multi-Token Prediction (MTP)

MTP allows the model to draft multiple candidate tokens in one forward pass using auxiliary "MTP heads" trained alongside the main model. Unlike speculative decoding with a separate draft model, MTP heads add no memory overhead beyond the main model's weights.

**Key facts**:
- **Only works on models trained with MTP heads**: Qwen3.6, DeepSeek V3/R1, Gemma 4 (via drafter). Won't help with Llama 3, Mistral, etc.
- **llama.cpp**: MTP support merged, currently in beta.¹⁴
- **MLX**: Native MTP support for Qwen3.6 and Gemma 4.¹⁵

**Measured speedups on Apple Silicon**:

| Config | Speedup |
|---|---|
| MTP depth-1 on M4 Pro, Qwen3.5-4B | 1.20× |
| MTP depth-1 on M4 Pro, Qwen3.6-27B | 1.40× |
| MTP depth-3 on M4 Pro, 27B model | 2.6× (18.3 tok/s from 7 tok/s baseline) |

¹⁶

For coding and research use where you interact with 27B-class models, MTP depth-3 on Qwen3.6-27B is a significant practical win.

### 3.3 Speculative Decoding

Standard speculative decoding (separate draft model) is supported in both llama.cpp and mlx-lm. Practical options:

- **Prompt lookup decoding**: Works with any model, zero memory overhead, ~1.2× speedup for completion-style prompts with repetition (code, data). Available in llama.cpp.
- **EAGLE-3 on MLX**: Community prototype, fastest speculative approach when draft model is available.¹⁷
- **ReDrafter on Apple Silicon**: Up to 2.3× speedup measured on M2 Ultra.
- **Without MTP heads**: Typical speedup is 1.2–1.5× with a good draft model.

Speculative decoding and MTP are complementary — MTP uses the model's own internal heads while speculative decoding uses a separate smaller model.

### 3.4 Flash Attention on Apple Silicon

Both backends support it:
- **llama.cpp**: `-fa` flag. Practically required for contexts >32K. Enables KV cache quantization. Reduces attention memory from O(n²) to O(n).
- **MLX**: Supported for prefill acceleration. The performance gap vs llama.cpp at very long context suggests MLX's flash attention implementation or the attention path has a bottleneck that remains unresolved as of January 2026.

---

## 4. Server Frontends and Concurrency

### For Your Use Case: 2–10 Parallel Requests, Coding + Research Agents

| Runtime | Backend | Concurrency | Long Context | Control | Recommended For |
|---|---|---|---|---|---|
| **llama-server** | llama.cpp | Continuous batching via `--parallel` | Best (flash attn + KV quant) | Full | Power users, coding, long context |
| **oMLX** | MLX | True continuous batching + SSD KV offload | Excellent (SSD-backed) | Medium | Multi-agent, max concurrency |
| **vLLM-MLX** | MLX | Best multi-user throughput (400+ tok/s aggregate) | Good | Medium | High-concurrency serving |
| **LM Studio 0.4+** | MLX + llama.cpp | Parallel requests (v0.4.2+) | Good | Low (GUI) | Ease of use + API |
| **Ollama 0.19+** | MLX (as of March 2026) | Basic | Good | Medium | Simplicity, now fast |
| **Jan.ai** | MLX/llama.cpp | Limited | OK | Low | Privacy-strict, offline |
| **mlx-openai-server** | MLX | FastAPI-based | OK | High (Python) | Custom MLX serving |

### Detailed Notes

**Ollama (now on MLX)**: Version 0.19 (March 2026) switched to MLX backend on Apple Silicon. Benchmark: on M5 Max with Qwen3.5-35B-A3B — prefill went from 1,154 to 1,810 tok/s (+57%), decode from 58 to 112 tok/s (+93%).¹⁸ [VENDOR-CLAIM] Still not designed for high concurrency; best for single-user interactive use.

**oMLX** (highly recommended for your concurrency needs): The only backend that **actually batches** under concurrency as of early 2026.¹⁹ Features:
- Continuous batching with 4.14× throughput gain at 8× concurrency
- SSD-backed KV cache: hot cache on unified memory, overflow blocks offloaded to SSD in safetensors format, restored on next matching prefix — even across server restarts
- Stable at 32K+ context (~20 tok/s flat vs competitors halving at 32K)
- Web admin UI at `/admin`
- Requires macOS 15+, M1 or later, 16GB+ RAM (64GB+ recommended)
- OpenAI-compatible API

**vLLM on Apple Silicon**: Official vLLM has only experimental CPU-mode support on Apple Silicon (no GPU). Two community alternatives:²⁰
- `vllm-metal` (community plugin via Docker Model Runner): v0.2.0 April 2026, 83× TTFT improvement, Metal GPU support
- `vllm-mlx` (waybarrios): MLX backend, 400+ tok/s aggregate on M4, continuous batching, OpenAI-compatible, supports MCP tool calling. Works with Claude Code, Cursor.

**vLLM-MLX is the best option if you want a vLLM-like API with true Apple Silicon GPU performance.** The main vLLM project is still GPU-cloud-centric.

**LM Studio 0.4** (January 2026): Brought parallel requests with continuous batching. 0.4.2 added MLX parallel requests. MLX inference is 30–50% faster than llama.cpp for most models in LM Studio. OpenAI-compatible API. Good middle ground if you want a GUI + API without deep configuration. [VENDOR-CLAIM]²¹

**llama-server concurrency**: `--parallel 4` enables 4 simultaneous request slots. For a Mac with 64GB unified memory running a 27B model at Q4, 4 parallel slots is a practical ceiling before memory bandwidth saturates. At 2–10 parallel requests for coding/research agents, this is sufficient.⁹

**vLLM vs Ollama throughput gap**: At high concurrency (32 simultaneous users), vLLM-MLX achieves ~1,150 tok/s aggregate vs Ollama's single-user 58 tok/s. For your 2–10 concurrent request scenario, this gap is less dramatic — Ollama and llama-server both handle it reasonably.²²

**lmdeploy**: No confirmed Apple Silicon GPU support found. Likely CPU-only or not supported.

**TabbyML**: Backend on Apple Silicon is llama.cpp or MLX depending on version; documentation on high-concurrency handling is sparse. Better suited as a coding assistant plugin layer on top of another server.

**mistral.rs**: Works on Apple Silicon, MLX backend support. Good OpenAI-compatible API. Downside: 8-second cold start, 48MB idle RAM, heavier than Ollama.²³

---

## 5. Recommendations for Your Setup

**Your profile**: Strong Apple Silicon Mac, want advanced features (TurboQuant, MTP), research + coding agents, 2–10 parallel requests, not just Ollama.

### Stack A: Maximum Control + Best Long-Context (Recommended for daily use)

**llama-server** as the core runtime with:
```
-ngl 99 -fa -c 65536 -b 2048 -ub 2048
--cache-type-k q8_0 --cache-type-v q8_0
--parallel 4
```
- Best stability at 70K+ context
- Flash attention + KV quantization = fit 70B models in 64GB with 32K context
- OpenAI-compatible API for any client (Claude Code, Cursor, Continue, etc.)
- Broadest model format support (GGUF)
- Add MTP via `AtomicBot-ai/atomic-llama-cpp-turboquant` fork for Qwen3.6

### Stack B: Best Concurrency for Agent Workloads

**oMLX** as the server. It's the only option that genuinely batches concurrent requests and maintains long-context performance via SSD KV offload. Pair with Qwen3.6 or DeepSeek V4 models.

### Stack C: Cutting-Edge MLX Features (Short-Context, High Speed)

**mlx-lm server** directly for single-user or low-concurrency use:
- 230+ tok/s at short context
- Native MTP with Qwen3.6 (1.4–2.6× speedup)
- Beware: long-context performance regression, avoid for 70K+ contexts until Issue #763 is resolved

### Model Recommendations (May 2026)

| Use Case | Model | Format |
|---|---|---|
| Coding (64GB Mac) | Qwen3.6-27B | Q6_K or Q8_0 GGUF |
| Coding (32GB Mac) | Qwen3.5-14B | Q6_K GGUF |
| Research/long doc | DeepSeek V4 | Q4_K_M GGUF |
| Speed-priority | Qwen3.5-4B MTP | MLX 8-bit |
| MoE efficiency | Qwen3.5-35B-A3B | Q4_K_M GGUF |

### TurboQuant: Worth Using Now?

If you have a 64GB Mac and want to run 70B models with 32K+ context: **yes**, the llama.cpp fork (`TheTom/llama-cpp-turboquant`) is functional on Metal. 6.3 tok/s on Mac Mini M4 Pro with a 70B model at 32K context. If you have 128GB, you probably don't need it yet. Wait for mainline integration for stability — it's coming.

---

## 6. Summary of Active Bugs to Watch

| Bug | Runtime | Severity | Status |
|---|---|---|---|
| Long-context performance 50% slower | mlx-lm | High | **Open** (Jan 2026) |
| BlockingIOError at 50K+ context | mlx-engine/LM Studio | High | Partial fix PR |
| Prefix cache broken for hybrid models | mlx-lm | Medium | Open |
| MTP 1-token completion bug in server | mlx-lm 0.31+ | Medium | Open (#1292) |
| KV cache cross-contamination at 16+ concurrent | mlx-lm server | Critical | Closed (#965, fixed) |
| Mixed KV quant without flash attn on Metal | llama.cpp Metal | Low | Open (#21450) |

---

## Sources

1. [LM Studio Bug Tracker — BlockingIOError at 50K context](https://github.com/lmstudio-ai/mlx-engine/issues/168)
2. [mlx-lm Issue #763 — Token generation 50% slower than llama.cpp at long context](https://github.com/ml-explore/mlx-lm/issues/763)
3. [mlx-lm Issue #48 — Gemma 3 context capped at 4096](https://github.com/ml-explore/mlx-lm/issues/48)
4. [mlx-lm Issue #980 — Prefix cache broken for hybrid architectures](https://github.com/ml-explore/mlx-lm/issues/980)
5. [mlx-lm Issue #965 — KV cache cross-contamination at 16+ concurrent requests](https://github.com/ml-explore/mlx-lm/issues/965)
6. [mlx-lm Issue #1292 — MTP 1-token completion bug in server mode](https://github.com/ml-explore/mlx-lm/issues/1292)
7. [Medium — Tuning llama.cpp on Apple Silicon: 7 Flags](https://medium.com/@michael.hannecke/tuning-llama-cpp-on-apple-silicon-843f37a6c3dc)
8. [Hacker News — KVSplit: 2-3x longer contexts on Apple Silicon](https://news.ycombinator.com/item?id=44009321)
9. [Medium — Tuning llama-server on Apple Silicon](https://medium.com/@michael.hannecke/tuning-llama-server-on-apple-silicon-9b3e778ab100)
10. [Comparative Study: MLX, MLC-LLM, Ollama, llama.cpp (arXiv 2511.05502)](https://arxiv.org/pdf/2511.05502)
11. [TurboQuant ICLR 2026 paper](https://arxiv.org/html/2504.19874v1)
12. [GitHub — lingengyuan/qjl-mlx: First MLX native TurboQuant](https://github.com/lingengyuan/qjl-mlx)
13. [TurboQuant benchmark on Apple Silicon](https://asiai.dev/turboquant/)
14. [llama.cpp MTP beta support — Startup Fortune](https://startupfortune.com/llamacpp-now-supports-multi-token-prediction-in-beta-and-the-implications-for-local-ai-tooling-are-bigger-than-the-pr-suggests/)
15. [Gemma 4 MTP speculative decoding with MLX — Google Blog](https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/)
16. [Medium — 18 tok/s from 27B model: MLX + MTP on Apple Silicon](https://medium.com/@vinoth.lingam333/i-got-18-tok-s-from-a-27b-model-on-a-macbook-mlx-native-mtp-speculative-decoding-on-apple-5e1211a06d0c)
17. [GitHub Discussion #890 — EAGLE-3 speculative decoding on Apple Silicon](https://github.com/ml-explore/mlx-lm/discussions/890)
18. [Will It Run AI — MLX vs Ollama benchmarks 2026](https://willitrunai.com/blog/mlx-vs-ollama-apple-silicon-benchmarks)
19. [GitHub — jundot/omlx](https://github.com/jundot/omlx)
20. [GitHub — vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal)
21. [LM Studio — Parallel Requests docs](https://lmstudio.ai/docs/app/advanced/parallel-requests)
22. [vLLM vs Ollama 2026: 9x throughput gap tested](https://tech-insider.org/vllm-vs-ollama-2026/)
23. [Ollama vs Mistral — Markaicode 2026](https://markaicode.com/vs/ollama-vs-mistral/)

---

## Research Metadata
- Depth mode: Medium
- Sources analyzed: ~25 verified
- Research date: 2026-05-30
- Key search terms: mlx-lm context window bug, TurboQuant PolarQuant apple silicon, MTP speculative decoding apple silicon, llama.cpp metal flash attention KV cache, omlx continuous batching, vLLM apple silicon 2026, llama-server concurrent requests
- Citation style: Compact (numbered footnotes)
- Bias classification: Product comparison — vendor claims flagged with [VENDOR-CLAIM]
