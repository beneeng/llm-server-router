# Research Plan: MLX and mlx-lm for Apple Silicon (2025-2026)

## Bias Classification
**BIAS-SENSITIVE TOPIC**: YES
- Product/performance comparisons (MLX vs llama.cpp vs Ollama)
- Benchmark claims from vendor-affiliated sources
- Apple and MLX maintainers have commercial/reputational interest in MLX performance numbers
- Fast-moving field (<12 months)

**ARRB Mode**: Lightweight bias check
**Vendor claim flag**: [VENDOR-CLAIM] for Apple Inc., MLX project maintainers, and Hugging Face claims about their own MLX-compatible models

## Sub-Questions
1. Context window bug ~70k tokens: status, cause, fix
2. New features 2025: flash attention, speculative decoding, prompt caching, quantization
3. Current known limitations and bugs
4. Quantization formats supported
5. Throughput benchmarks vs alternatives

## Search Strategy
- GitHub releases/changelog: https://github.com/ml-explore/mlx-lm/releases
- GitHub issues (context window): search issues
- MLX community discussions: Reddit r/LocalLLaMA, HuggingFace forums
- Independent benchmarks: third-party blog posts, academic papers
- PyPI changelog: https://pypi.org/project/mlx-lm/#history

## Source Diversity Targets
- Official docs/GitHub: ≤40%
- Independent benchmarks/blogs: ≥30%
- Community reports: ≥20%
- Academic: remainder

## Stopping Criteria
- 3 consecutive sources adding <5% new info
- All 5 sub-questions answered with confidence
- Max ~20 sources (medium depth)
