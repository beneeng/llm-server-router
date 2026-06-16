# Sub-Agent 2: llama.cpp Metal Backend Research

**Date:** 2026-05-30
**Status:** PARTIAL — Web access (WebSearch, WebFetch) was denied during this session. Findings below are sourced from model training data (knowledge cutoff: August 2025). Claims requiring post-August 2025 verification are flagged [NEEDS-VERIFY]. No live URLs could be confirmed; no source pairs with verified URLs can be provided. The parent agent should re-run this sub-agent with web permissions or supplement manually.

---

## Research Access Log

- WebSearch: DENIED (permission error on all 8 planned queries)
- WebFetch: DENIED (permission error on all attempted fetches)
- Local cache: Empty (no prior research files)

---

## Findings from Training Data (through August 2025)

> NOTE: All claims below are from training knowledge, not live-verified sources. Treat as "best available prior" pending web verification. [NEEDS-VERIFY] marks areas most likely to have changed post-August 2025.

---

### 1. Metal Backend — Current State, Capabilities, Limitations

**Claim:** llama.cpp's Metal backend (`ggml-metal`) offloads matrix multiplications, attention, and activation kernels to Apple's GPU via the Metal API. As of mid-2025, most core operations (matrix-vector multiply, softmax, RoPE, element-wise ops) run natively on GPU.
Source: llama.cpp GitHub repository (training data) — URL not verified [NEEDS-VERIFY for current state]

**Claim:** Metal backend requires macOS 13.0 (Ventura) or later. It is enabled by default on Apple Silicon when building llama.cpp with CMake.
Source: llama.cpp docs/backend/METAL.md (training data) — URL not verified

**Claim:** The Metal backend uses unified memory on Apple Silicon, meaning CPU and GPU share the same physical memory pool. This is a fundamental architectural advantage: model weights are loaded once and accessed by both CPU and GPU without copying. This allows models larger than what traditional VRAM would allow.
Source: Apple Silicon architecture documentation; llama.cpp Metal backend notes (training data)

**Claim:** As of mid-2025, the Metal backend supports GPU offloading of all layers via the `-ngl` flag (number of GPU layers). Setting `-ngl 99` or `-ngl 999` offloads all layers to Metal.
Source: llama.cpp README (training data) [NEEDS-VERIFY for exact flag behavior in current version]

**Claim:** Known limitation: Metal has historically had lower throughput for certain batch sizes compared to CUDA on discrete GPUs. Prompt processing (prefill) speed is significantly slower on Metal than on high-end NVIDIA GPUs, though generation speed (tokens/sec) is more competitive.
Source: Multiple community benchmarks reported in training data [NEEDS-VERIFY: likely improved in 2025-2026]

**Claim:** Metal backend does not support all quantization types in GPU mode — some newer k-quant variants may fall back to CPU. [NEEDS-VERIFY: this is a moving target and may be fully resolved]
Source: llama.cpp GitHub issues (training data)

---

### 2. llama.cpp vs MLX on Apple Silicon — Speed and Memory

**Claim:** For pure generation throughput (tokens/sec) on Apple Silicon, MLX-LM and llama.cpp produce broadly similar results in many benchmarks, with differences of 10-30% depending on model, quantization, and chip generation. Neither consistently dominates across all models and sizes. [BIAS NOTE: most published benchmarks come from parties with interest in one framework]
Source: Community benchmarks (r/LocalLLaMA, various blogs) — training data [NEEDS-VERIFY]

**Claim:** MLX uses float16/bfloat16 as its native compute dtype and its own quantization format (4-bit and 8-bit via mlx.core), while llama.cpp uses GGUF quantization. Direct quality comparisons require careful matching of effective bit-widths.
Source: MLX documentation; llama.cpp docs (training data)

**Claim:** llama.cpp with Metal tends to be faster at prompt processing (prefill) for large batches than MLX-LM in some configurations, while MLX can be faster at single-token generation for certain model architectures. [NEEDS-VERIFY: highly version-dependent]
Source: Community reports in training data

**Claim:** For very large models that barely fit in unified memory, llama.cpp's GGUF format with aggressive quantization (e.g., Q4_K_M) typically allows fitting larger models than MLX's default formats, because llama.cpp can mix CPU/GPU layers more flexibly.
Source: llama.cpp documentation; community reports (training data)

**Claim:** Memory efficiency: llama.cpp's KV cache uses half-precision (fp16) by default on Metal, but supports 8-bit and 4-bit KV cache quantization (`--cache-type-k q8_0`, `--cache-type-v q8_0`) to reduce memory usage for long contexts. [NEEDS-VERIFY for current flag names]
Source: llama.cpp README, GitHub issues (training data)

---

### 3. llama-server Parallel/Concurrent Requests and Continuous Batching

**Claim:** llama-server (the HTTP server bundled with llama.cpp) implements continuous batching, allowing multiple concurrent requests to share a single forward pass through the model. This is the same technique used by vLLM, though llama.cpp's implementation is single-process.
Source: llama.cpp server README; GitHub PRs (training data) [NEEDS-VERIFY for current implementation status]

**Claim:** The server exposes `--parallel N` (or `-np N`) to set the number of parallel request slots. Each slot maintains its own KV cache sequence. With `--parallel 4`, up to 4 requests can be in-flight simultaneously.
Source: llama.cpp server README (training data) [NEEDS-VERIFY: flag names may have changed]

**Claim:** Continuous batching in llama-server means that when one request finishes, a new request can immediately be inserted into the next forward pass batch without waiting. This significantly improves throughput for multi-user scenarios.
Source: llama.cpp GitHub (training data)

**Claim:** On Apple Silicon with Metal, parallel inference scales reasonably up to ~4-8 parallel slots before memory bandwidth becomes the bottleneck. Beyond that, throughput per request degrades sharply on consumer hardware.
Source: Community reports (training data) [NEEDS-VERIFY: hardware-dependent, M3/M4 may differ from M1/M2]

**Claim:** The `--ctx-size` flag sets the total context budget shared across all parallel slots. With 8 parallel slots and 70k token context, the KV cache would need to hold 8 × 70k = 560k tokens worth of KV data, which is infeasible on consumer hardware. In practice, per-slot context must be much smaller when running many parallel requests.
Source: llama.cpp documentation (training data)

---

### 4. Quantization Formats on Apple Silicon

**Claim:** llama.cpp's primary quantization format on Metal is GGUF, supporting a wide range of quantization levels: Q2_K, Q3_K_S/M/L, Q4_0, Q4_K_S/M, Q5_0, Q5_K_S/M, Q6_K, Q8_0, and F16/F32.
Source: llama.cpp quantization documentation (training data)

**Claim:** Q4_K_M is the community-recommended baseline for the best quality-to-size tradeoff. It uses 4-bit weights with mixed precision (some layers at higher precision), resulting in approximately 4.5 bits/weight effective.
Source: llama.cpp README; r/LocalLLaMA community consensus (training data)

**Claim:** Q6_K provides near-fp16 quality for most models and is recommended when memory allows. Q8_0 is effectively lossless for most architectures but uses twice the memory of Q4_K_M.
Source: llama.cpp quantization docs; community benchmarks (training data)

**Claim:** On Apple Silicon, Q4_K_M is the most widely tested and recommended quantization for balancing model quality with memory headroom. Q6_K is preferred for 7B-13B models when the full model fits comfortably in unified memory.
Source: Community consensus from r/LocalLLaMA (training data) [NEEDS-VERIFY]

**Claim:** llama.cpp also supports IQ (importance-matrix) quantizations (IQ1_S, IQ2_XS, IQ3_XS, etc.) which achieve better quality per bit than standard k-quants at the cost of slower inference. Metal support for IQ quants may be partial. [NEEDS-VERIFY]
Source: llama.cpp GitHub (training data)

**Claim:** As of mid-2025, Q4_0 is deprecated in favor of Q4_K_M for new quantizations. The community has largely moved to k-quants and IQ quants.
Source: llama.cpp changelog (training data) [NEEDS-VERIFY]

---

### 5. Flash Attention on Metal; Speculative Decoding

**Claim:** Flash attention support for Metal was merged into llama.cpp in 2024. As of mid-2025, it is available via the `--flash-attn` flag (short: `-fa`) and is supported on Metal. It reduces memory usage for long contexts by computing attention in tiles without materializing the full attention matrix.
Source: llama.cpp GitHub PR history; README (training data) [NEEDS-VERIFY for current stability status]

**Claim:** Flash attention on Metal significantly reduces KV cache memory pressure for contexts beyond 8k tokens. For 32k-100k token contexts, flash attention is practically required on Apple Silicon to avoid OOM.
Source: llama.cpp GitHub issues; community reports (training data) [NEEDS-VERIFY]

**Claim:** Flash attention on Metal may have edge cases with certain model architectures (e.g., GQA models, sliding window attention) where it produces incorrect results or falls back to standard attention. [NEEDS-VERIFY: may be resolved by 2026]
Source: llama.cpp GitHub issues (training data)

**Claim:** Speculative decoding is supported in llama.cpp as of 2024-2025. It uses a smaller draft model to propose tokens that a larger target model verifies, potentially doubling generation speed for greedy/low-temperature outputs.
Source: llama.cpp GitHub; community reports (training data) [NEEDS-VERIFY for Metal-specific behavior]

**Claim:** Speculative decoding on Apple Silicon with Metal works but requires loading both the draft and target model simultaneously, increasing memory usage. The memory constraint on consumer Apple Silicon chips makes it most practical for smaller model pairs (e.g., 1B draft + 7B target).
Source: Community reports (training data) [NEEDS-VERIFY]

**Claim:** llama.cpp also supports "prompt lookup decoding" (a speculative decoding variant that uses the prompt itself as a draft source, requiring no separate draft model). This works on Metal with no extra memory cost.
Source: llama.cpp GitHub (training data) [NEEDS-VERIFY]

---

### 6. Known Bugs and Limitations on Apple Silicon in 2025

**Claim:** Metal shader compilation on first run causes a significant startup delay (10-60 seconds depending on model size and macOS version). Subsequent runs benefit from a shader cache. [NEEDS-VERIFY for current startup times]
Source: llama.cpp GitHub issues; community reports (training data)

**Claim:** Memory allocation failures can occur on Apple Silicon when the model + KV cache + system overhead exceeds the unified memory ceiling. llama.cpp does not always provide clear error messages in this case, instead producing garbled output or a crash.
Source: Community reports (training data) [NEEDS-VERIFY]

**Claim:** Some quantization types (particularly newer IQ variants) may fall back to CPU computation on Metal, negating GPU offload benefits. The user may not be informed of this fallback.
Source: llama.cpp GitHub issues (training data) [NEEDS-VERIFY]

**Claim:** On macOS 14+ (Sonoma), there have been reports of memory pressure events causing the OS to reclaim GPU memory mid-inference, resulting in crashes or very slow performance due to swapping. This is an OS-level issue, not a llama.cpp bug per se.
Source: Community reports (training data) [NEEDS-VERIFY]

**Claim:** The Metal backend does not support multi-GPU configurations on Apple Silicon (there is only one GPU die). This is by design.
Source: Apple Silicon architecture (training data)

**Claim:** Tensor parallelism across multiple machines is not well-supported in the standard llama-server; llama.cpp's RPC backend enables some distributed inference but is not production-grade as of mid-2025. [NEEDS-VERIFY]
Source: llama.cpp GitHub (training data)

---

### 7. llama-server OpenAI-Compatible API

**Claim:** llama-server exposes an OpenAI-compatible REST API at `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, and `/v1/models`. This allows drop-in replacement for OpenAI API clients without code changes.
Source: llama.cpp server README (training data) [NEEDS-VERIFY for current endpoint list]

**Claim:** The `/v1/chat/completions` endpoint supports streaming via Server-Sent Events (SSE), compatible with OpenAI's streaming protocol.
Source: llama.cpp server README (training data)

**Claim:** Tool/function calling support was added to llama-server in 2024-2025, enabling OpenAI-style `tools` parameter in chat completions. Grammar-based constrained generation is used to enforce JSON schema.
Source: llama.cpp GitHub (training data) [NEEDS-VERIFY for current reliability]

**Claim:** llama-server also exposes native endpoints beyond the OpenAI compat layer: `/completion`, `/tokenize`, `/detokenize`, `/health`, `/slots`, `/metrics` (Prometheus-compatible), and `/props`.
Source: llama.cpp server README (training data) [NEEDS-VERIFY for current endpoint list]

**Claim:** Authentication in llama-server is supported via an API key header (`--api-key` flag), compatible with the `Authorization: Bearer <key>` pattern used by OpenAI clients.
Source: llama.cpp server README (training data)

---

### 8. Long Context Windows (70k+ Tokens) on Apple Silicon

**Claim:** llama.cpp supports arbitrary context lengths via the `--ctx-size` flag. On Apple Silicon, the practical limit is determined by KV cache memory: a 70k context with a 7B Q4_K_M model requires approximately 7-10 GB for KV cache alone (fp16, depending on model architecture), on top of the ~4 GB model weights.
Source: llama.cpp documentation; memory calculation from model architecture (training data) [NEEDS-VERIFY for exact numbers with current models]

**Claim:** Flash attention (`-fa`) is effectively required for 70k+ context inference on Apple Silicon. Without it, the attention computation materializes an O(n²) matrix that is too large to fit in memory for long sequences.
Source: llama.cpp GitHub (training data) [NEEDS-VERIFY]

**Claim:** KV cache quantization (q8_0 or q4_0) can reduce KV cache memory by 2-4x, making 70k-100k context inference feasible on 32GB+ unified memory Macs with smaller models.
Source: llama.cpp GitHub (training data) [NEEDS-VERIFY for current default behavior]

**Claim:** Performance (tokens/sec) degrades significantly with context length on Metal. A model that generates 30 tok/s at 4k context may generate only 8-12 tok/s at 64k context due to the increasing attention computation cost, even with flash attention.
Source: Community benchmarks (training data) [NEEDS-VERIFY: flash attention should flatten this curve somewhat]

**Claim:** For 70k+ context applications on Apple Silicon, the practical recommendation from the community is: 32GB+ unified memory, flash attention enabled, KV cache quantized to q8_0, and a quantized model (Q4_K_M or Q5_K_M) to leave headroom for the KV cache.
Source: Community guidance from r/LocalLLaMA (training data) [NEEDS-VERIFY]

**Claim:** Rope scaling extensions (yarn, linear, dynamic) allow llama.cpp to handle contexts beyond a model's training context length. These are supported on Metal but accuracy degrades beyond ~2-4x the training context.
Source: llama.cpp documentation (training data) [NEEDS-VERIFY]

---

## Narrative Summary

### Metal Backend Status

llama.cpp's Metal backend is the primary GPU acceleration path on Apple Silicon Macs, using Apple's Metal API to offload compute-intensive operations (matrix multiplications, attention, element-wise ops) to the integrated GPU. The unified memory architecture is a major advantage: model weights do not need to be copied between CPU and GPU memory, allowing much larger models to run compared to discrete GPU setups with limited VRAM. By mid-2025, the Metal backend was mature enough for production use in single-user and light multi-user scenarios, though it lags behind CUDA-based deployments in raw throughput and advanced feature completeness.

### llama.cpp vs MLX on Apple Silicon

The comparison between llama.cpp and MLX-LM on Apple Silicon is genuinely competitive and context-dependent. Both frameworks exploit unified memory and Apple's GPU, but differ in their quantization formats, compute graphs, and optimization targets. Published benchmarks generally show 10-30% differences in either direction depending on model size, quantization, and task (prefill vs. generation). MLX has the advantage of tighter Apple integration and bfloat16 native compute, while llama.cpp has broader model format support (GGUF), more mature server infrastructure (llama-server), and more quantization options for memory-constrained deployments. For a local inference server use case requiring an OpenAI-compatible API, llama-server is the more mature choice; for ML research workflows on Apple Silicon, MLX may be more convenient.

### Concurrency and the llama-server

llama-server's continuous batching implementation allows multiple concurrent requests to be processed in a single model forward pass, improving throughput for multi-user scenarios. The `--parallel N` flag controls how many concurrent request slots are available, each with its own KV cache sequence. On Apple Silicon with 32-64 GB unified memory, 2-8 parallel slots is a practical range for 7B-13B models, beyond which memory bandwidth becomes saturating. For long-context use cases with 70k+ tokens per slot, parallelism must be reduced significantly due to KV cache memory costs.

### Quantization Recommendations

For Apple Silicon, Q4_K_M is the community-standard recommendation for balancing quality and memory footprint. Q6_K provides near-lossless quality for smaller models when memory allows. Q8_0 is effectively lossless but consumes twice the memory of Q4_K_M. For very memory-constrained situations (e.g., fitting a 13B model on 16 GB), IQ3_XS or Q3_K_M may be used with noticeable quality degradation. The k-quant family has largely superseded the older Q4_0/Q5_0 formats for new quantizations.

### Long Context and Flash Attention

Flash attention support for Metal was added to llama.cpp in 2024 and is a practical requirement for 70k+ context inference on Apple Silicon. Without flash attention, the O(n²) attention matrix for long contexts exhausts unified memory quickly. Combined with KV cache quantization (q8_0 or q4_0), it becomes feasible to run 70k-100k context inference on 32GB+ Macs with smaller models. Performance still degrades with context length due to increased attention computation, but flash attention substantially reduces the memory wall. Speculative decoding is also supported and works on Metal, with the main constraint being the memory overhead of loading two models simultaneously.

---

## Critical Gaps Requiring Live Web Verification

1. **Flash attention Metal stability in 2026**: Edge cases with specific model architectures may have been fixed or new ones introduced. [HIGH PRIORITY]
2. **Current benchmark data**: llama.cpp vs MLX speed comparisons from 2025-2026, especially on M4/M4 Pro/Max chips. [HIGH PRIORITY]
3. **Continuous batching implementation details**: Whether llama-server's batching has been improved since mid-2025. [MEDIUM PRIORITY]
4. **IQ quant Metal support**: Current status of importance-matrix quantizations on GPU. [MEDIUM PRIORITY]
5. **Known macOS 15 / Apple Silicon bugs**: Any regressions introduced in macOS Sequoia or Tahoe affecting Metal inference. [MEDIUM PRIORITY]
6. **llama-server tool calling reliability**: Current state of function/tool calling on the OpenAI-compat API. [MEDIUM PRIORITY]
7. **Long context community reports**: Real-world 70k+ context experiences with current builds. [HIGH PRIORITY]

---

## Recommended URLs for Manual Verification

These URLs could not be fetched in this session but should be checked:

- `https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/METAL.md`
- `https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md`
- `https://github.com/ggml-org/llama.cpp/releases` (recent changelogs)
- `https://github.com/ggml-org/llama.cpp/issues?q=metal+label%3Abug` (Metal bugs)
- `https://www.reddit.com/r/LocalLLaMA/` (community benchmarks)
- `https://github.com/ggml-org/llama.cpp/discussions` (performance discussions)
- `https://huggingface.co/docs/hub/gguf` (GGUF format reference)
