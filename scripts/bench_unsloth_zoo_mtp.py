#!/usr/bin/env python3
"""Four-way decode benchmark: {stock mlx-vlm, unsloth-zoo} x {no drafter, MTP drafter}.

Companion to bench_unsloth_zoo.py (same helpers, same isolated-venv rules, same memory guards).

  stock       mlx_vlm.stream_generate, no drafter
  zoo         same, inside unsloth_zoo generation_mode() (PR #1871 fusions)
  stock+mtp   mlx-vlm's own speculative rounds with the MTP drafter (load_drafter)
  zoo+mtp     unsloth-zoo's SpeculativeEngine (adaptive DraftController, fused verify) with the
              same drafter, reached through install_speculative_seam()

Judge MTP by acceptance and tokens per verify round, not tok/s alone. mlx-vlm's GenerationResult
carries no acceptance count, so acceptance is only reported for zoo+mtp (draft_n_accepted/draft_n).
install_speculative_seam() patches mlx-vlm globally for the rest of the process; it is installed
lazily on the first zoo+mtp call, so later stock+mtp runs go through its wrappers (they delegate to
upstream for non-zoo drafters).

    ~/.local/share/mlx/venvs/zoo-bench/bin/python scripts/bench_unsloth_zoo_mtp.py \
        --model ~/.cache/mlx-vlm/qwen3.8-flash-next-ple-offload \
        --drafter sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_unsloth_zoo import GB, SHORT, long_text, render  # noqa: E402

MODES = ("stock", "zoo", "stock+mtp", "zoo+mtp")


class Ctx:
    """Everything the four modes share: the target, its processor, and the two drafters."""

    def __init__(self, model, processor, drafter_path: str, block_size: int | None):
        from mlx_vlm.speculative.drafters import load_drafter
        self.model, self.processor, self.block_size = model, processor, block_size
        self.stock_drafter, self.stock_kind = load_drafter(drafter_path)
        self.drafter_path = drafter_path
        self.zoo_draft = None  # built lazily, with the seam

    def zoo_speculative(self):
        if self.zoo_draft is None:
            from unsloth_zoo.mlx.speculative import (
                DraftController, SpeculativeDraft, companion_drafter, install_speculative_seam)
            install_speculative_seam()
            drafter = companion_drafter(self.drafter_path, self.model)
            self.zoo_draft = SpeculativeDraft(DraftController(max_depth=getattr(drafter, "max_depth", None) or int(drafter.model.config.block_size) - 1), drafter)
        return self.zoo_draft


def run_once(c: Ctx, prompt: str, max_tokens: int, mode: str):
    import mlx.core as mx
    from mlx_vlm import stream_generate

    mx.clear_cache()
    kwargs = dict(max_tokens=max_tokens, temperature=0.0)
    draft = None
    if mode == "stock+mtp":
        kwargs.update(draft_model=c.stock_drafter, draft_kind=c.stock_kind)
        if c.block_size:
            kwargs["draft_block_size"] = c.block_size
    elif mode == "zoo+mtp":
        from unsloth_zoo.mlx.generate import SamplingParams
        draft = c.zoo_speculative()
        tok = getattr(c.processor, "tokenizer", c.processor)
        draft.prepare(tok.encode(prompt, add_special_tokens=False), SamplingParams(temperature=0.0), [])
        kwargs.update(draft_model=draft, draft_kind=draft.draft_kind)

    tokens, last = [], None

    def drive():
        nonlocal last
        for resp in stream_generate(c.model, c.processor, prompt, **kwargs):
            if resp.token is not None:
                tokens.append(int(resp.token))
            last = resp

    t0 = time.time()
    if mode.startswith("zoo"):
        from unsloth_zoo.mlx.generate import generation_mode
        with generation_mode(c.model):
            drive()
    else:
        drive()
    return {
        "tokens": tokens, "prompt_tokens": last.prompt_tokens, "gen_tokens": last.generation_tokens,
        "prompt_tps": last.prompt_tps, "decode_tps": last.generation_tps, "peak_gb": last.peak_memory,
        "wall_s": time.time() - t0,
        "draft_n": getattr(draft, "draft_n", None), "draft_acc": getattr(draft, "draft_n_accepted", None),
    }


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--drafter", required=True, help="MTP drafter repo id or path")
    ap.add_argument("--block-size", type=int, default=None, help="stock+mtp draft_block_size (default: drafter's own)")
    ap.add_argument("--contexts", type=int, nargs="+", default=[0, 8000, 24000])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--min-available-gb", type=float, default=95.0)
    ap.add_argument("--memory-limit-gb", type=float, default=100.0)
    ap.add_argument("--out", type=Path, default=Path("/tmp/bench_unsloth_zoo_mtp.jsonl"))
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
    c = Ctx(model, processor, args.drafter, args.block_size)
    root = Path(__file__).resolve().parent.parent
    rows = []

    for ctx in args.contexts:
        prompt = render(processor, SHORT if ctx == 0 else long_text(root, ctx))
        for mode in MODES:  # warm each mode on this exact prompt shape (zoo admits kernels per shape)
            run_once(c, prompt, 16, mode)
        runs = {m: [] for m in MODES}
        for rep in range(args.reps):
            order = MODES if rep % 2 == 0 else MODES[::-1]  # alternate order against drift/thermals
            for mode in order:
                r = run_once(c, prompt, args.max_tokens, mode)
                runs[mode].append(r)
                acc = f" accept={r['draft_acc']}/{r['draft_n']}" if r["draft_n"] else ""
                print(f"ctx~{ctx:>6} {mode:<9} rep{rep}: prompt={r['prompt_tokens']} decode={r['decode_tps']:.1f} tok/s "
                      f"prefill={r['prompt_tps']:.0f} peak={r['peak_gb']:.1f}GB{acc}", flush=True)
        mean = lambda m, k: sum(r[k] for r in runs[m]) / len(runs[m])
        row = {"ctx": ctx, "prompt_tokens": runs["stock"][0]["prompt_tokens"],
               **{f"{m}_tps": mean(m, "decode_tps") for m in MODES},
               "peak_gb": max(r["peak_gb"] for m in MODES for r in runs[m]),
               "zoo_mtp_accept": (sum(r["draft_acc"] for r in runs["zoo+mtp"]) / max(1, sum(r["draft_n"] for r in runs["zoo+mtp"]))),
               "first_diff_vs_stock": {m: first_diff(runs["stock"][0]["tokens"], runs[m][0]["tokens"]) for m in MODES[1:]},
               "self_consistent": {m: first_diff(runs[m][0]["tokens"], runs[m][-1]["tokens"]) for m in MODES}}
        rows.append(row)
        print(f"ctx~{ctx} first diverging token vs stock: {row['first_diff_vs_stock']}; run-to-run: {row['self_consistent']}", flush=True)
        args.out.with_suffix(f".tokens-{ctx}.json").write_text(json.dumps({m: [r["tokens"] for r in runs[m]] for m in MODES}))
        with args.out.open("a") as f:
            f.write(json.dumps({**row, "model": str(args.model), "drafter": args.drafter, "ts": time.time()}) + "\n")

    print("\nprompt tok   stock   zoo   stock+mtp   zoo+mtp   zoo-mtp accept   peak GB")
    for r in rows:
        print(f"{r['prompt_tokens']:>10}  {r['stock_tps']:>6.1f} {r['zoo_tps']:>5.1f} {r['stock+mtp_tps']:>10.1f} "
              f"{r['zoo+mtp_tps']:>9.1f}   {r['zoo_mtp_accept']:>13.0%}   {r['peak_gb']:>7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
