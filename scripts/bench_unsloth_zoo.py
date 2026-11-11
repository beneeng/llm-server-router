#!/usr/bin/env python3
"""A/B decode speed: stock mlx-vlm vs. the same call inside unsloth-zoo's generation_mode().

unsloth-zoo PR #1871 routes qwen4_exp (Qwen3.8-Flash-Next) decode steps onto fused
kernels. The fusions are inference scopes opened by `unsloth_zoo.mlx.generate.generation_mode`;
they are NOT applied by `mlx_vlm.server`. So this benchmark runs in-process, loads the model
once, and times `mlx_vlm.stream_generate` with and without that context manager.

Run it under the isolated env, never the main tool venv (unsloth-zoo pins mlx-vlm<=0.7.12):

    ~/.local/share/mlx/venvs/zoo-bench/bin/python scripts/bench_unsloth_zoo.py \
        --model ~/.cache/mlx-vlm/qwen3.8-flash-next-ple-offload --contexts 0 8000

Stop every model first (`mlx stop`): the checkpoint peaks near 83 GB. The script refuses to
load if available RAM is too low and sets a hard MLX memory limit so it fails instead of
swapping the machine to death.

Greedy (temperature 0) so the two modes can be compared token by token: the PR claims
bit-identical output.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

GB = 1024**3


def long_text(root: Path, approx_tokens: int) -> str:
    """Real source code as filler (~3.5 chars/token); the exact count is measured later."""
    chunks, size, budget = [], 0, int(approx_tokens * 3.5)
    for path in sorted(root.glob("mlx_server/**/*.py")) + sorted(root.glob("scripts/*.py")):
        text = path.read_text(errors="ignore")
        chunks.append(f"### {path.name}\n{text}")
        size += len(text)
        if size >= budget:
            break
    return "\n\n".join(chunks)[:budget] + "\n\nSummarize what the code above does in four sentences."


SHORT = "Explain to a senior backend engineer why speculative decoding does not change the output distribution, in four paragraphs."


def render(processor, user_text: str) -> str:
    tok = getattr(processor, "tokenizer", processor)
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def run_once(model, processor, prompt: str, max_tokens: int, zoo: bool):
    import mlx.core as mx
    from mlx_vlm import stream_generate

    mx.clear_cache()
    kwargs = dict(max_tokens=max_tokens, temperature=0.0)
    tokens, last = [], None

    def drive():
        nonlocal last
        for resp in stream_generate(model, processor, prompt, **kwargs):
            if resp.token is not None:
                tokens.append(int(resp.token))
            last = resp

    t0 = time.time()
    if zoo:
        from unsloth_zoo.mlx.generate import generation_mode
        with generation_mode(model):
            drive()
    else:
        drive()
    return {
        "tokens": tokens,
        "prompt_tokens": last.prompt_tokens,
        "gen_tokens": last.generation_tokens,
        "prompt_tps": last.prompt_tps,
        "decode_tps": last.generation_tps,
        "peak_gb": last.peak_memory,
        "wall_s": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--contexts", type=int, nargs="+", default=[0, 8000],
                    help="approx prompt tokens; 0 = short prompt")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2, help="measured repetitions per mode (after 1 warmup each)")
    ap.add_argument("--min-available-gb", type=float, default=95.0)
    ap.add_argument("--memory-limit-gb", type=float, default=100.0)
    ap.add_argument("--out", type=Path, default=Path("/tmp/bench_unsloth_zoo.jsonl"))
    args = ap.parse_args()

    import mlx.core as mx
    import psutil

    avail = psutil.virtual_memory().available / GB
    if avail < args.min_available_gb:
        print(f"ABORT: only {avail:.1f} GB RAM available, need {args.min_available_gb:.0f}. Run `mlx stop` / close apps.")
        return 2
    mx.set_memory_limit(int(args.memory_limit_gb * GB))
    print(f"available RAM {avail:.1f} GB, MLX memory limit {args.memory_limit_gb:.0f} GB", flush=True)

    from mlx_vlm import load
    model, processor = load(str(args.model.expanduser()))
    root = Path(__file__).resolve().parent.parent
    rows = []

    for ctx in args.contexts:
        prompt = render(processor, SHORT if ctx == 0 else long_text(root, ctx))
        # Warm each mode once on the exact prompt shape: the zoo admits its fused kernels per
        # shape only after a bit-for-bit check on the first call, so that call is not steady-state.
        for zoo in (False, True):
            run_once(model, processor, prompt, 16, zoo)
        runs = {False: [], True: []}
        for rep in range(args.reps):
            for zoo in ((False, True) if rep % 2 == 0 else (True, False)):  # ABBA against drift/thermals
                r = run_once(model, processor, prompt, args.max_tokens, zoo)
                runs[zoo].append(r)
                print(f"ctx~{ctx:>6} {'zoo ' if zoo else 'stock'} rep{rep}: prompt={r['prompt_tokens']} "
                      f"decode={r['decode_tps']:.1f} tok/s prefill={r['prompt_tps']:.0f} peak={r['peak_gb']:.1f}GB", flush=True)
        def first_diff(a, b):
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    return i
            return None if len(a) == len(b) else min(len(a), len(b))

        s0, s1 = runs[False][0]["tokens"], runs[False][-1]["tokens"]
        z0, z1 = runs[True][0]["tokens"], runs[True][-1]["tokens"]
        diffs = {"stock_vs_stock": first_diff(s0, s1), "zoo_vs_zoo": first_diff(z0, z1),
                 "stock_vs_zoo": first_diff(s0, z0)}
        identical = all(v is None for v in diffs.values())
        print(f"ctx~{ctx} first diverging token index (None = identical): {diffs}", flush=True)
        dump = args.out.with_suffix(f".tokens-{ctx}.json")
        dump.write_text(json.dumps({"stock": [r["tokens"] for r in runs[False]],
                                    "zoo": [r["tokens"] for r in runs[True]]}))
        mean = lambda z, k: sum(r[k] for r in runs[z]) / len(runs[z])
        row = {"ctx": ctx, "prompt_tokens": runs[False][0]["prompt_tokens"],
               "stock_tps": mean(False, "decode_tps"), "zoo_tps": mean(True, "decode_tps"),
               "stock_prefill": mean(False, "prompt_tps"), "zoo_prefill": mean(True, "prompt_tps"),
               "peak_gb": max(r["peak_gb"] for z in runs for r in runs[z]), "tokens_identical": identical, "first_diff": diffs}
        row["speedup"] = row["zoo_tps"] / row["stock_tps"]
        rows.append(row)
        with args.out.open("a") as f:
            f.write(json.dumps({**row, "model": str(args.model), "ts": time.time()}) + "\n")

    print("\nctx(prompt tok)  stock tok/s  zoo tok/s  speedup  prefill stock/zoo  peak GB  identical")
    for r in rows:
        print(f"{r['prompt_tokens']:>14}  {r['stock_tps']:>11.1f}  {r['zoo_tps']:>9.1f}  {r['speedup']:>6.2f}x  "
              f"{r['stock_prefill']:>7.0f}/{r['zoo_prefill']:<7.0f}  {r['peak_gb']:>7.1f}  {r['tokens_identical']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
