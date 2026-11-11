#!/usr/bin/env python3
"""Benchmark one Qwen3.8-Flash-Next checkpoint at several context lengths (stock mlx-vlm, no zoo).

Used to compare the PLE-offloaded view against the fully resident checkpoint. One process per
checkpoint, so the weights of the previous run are really gone from RAM:

    PY=~/.local/share/uv/tools/mlx-server/bin/python
    $PY scripts/bench_ple_resident.py --label offload  --model ~/.cache/mlx-vlm/qwen3.8-flash-next-ple-offload
    $PY scripts/bench_ple_resident.py --label resident --model ~/.cache/mlx-vlm/qwen3.8-flash-next-4bit-source
    $PY scripts/bench_ple_resident.py --compare /tmp/bench_ple_resident.jsonl

Run `mlx stop` first. Greedy decoding. "cold" is the first request after load (page cache and
kernel shapes not warmed), "warm" is the mean of the measured repetitions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_unsloth_zoo import GB, SHORT, long_text, render, run_once  # noqa: E402


def compare(path: Path) -> int:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(f"{'label':<10} {'prompt':>7} {'cold pf':>8} {'pf tok/s':>9} {'dec tok/s':>10} {'peak GB':>8} {'swap MB':>8}")
    for r in rows:
        print(f"{r['label']:<10} {r['prompt_tokens']:>7} {r['cold_prefill']:>8.0f} {r['prefill']:>9.0f} "
              f"{r['decode']:>10.1f} {r['peak_gb']:>8.1f} {r['swap_used_mb']:>8.0f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--label", default="run")
    ap.add_argument("--contexts", type=int, nargs="+", default=[0, 2000, 8000, 24000])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--memory-limit-gb", type=float, default=107.0)
    ap.add_argument("--out", type=Path, default=Path("/tmp/bench_ple_resident.jsonl"))
    ap.add_argument("--compare", type=Path)
    args = ap.parse_args()
    if args.compare:
        return compare(args.compare)
    if not args.model:
        ap.error("--model required")

    import mlx.core as mx
    import psutil

    mx.set_memory_limit(int(args.memory_limit_gb * GB))
    from mlx_vlm import load

    t0 = time.time()
    model, processor = load(str(args.model.expanduser()))
    print(f"[{args.label}] loaded in {time.time() - t0:.0f}s, active={mx.get_active_memory() / GB:.1f} GB", flush=True)
    root = Path(__file__).resolve().parent.parent

    for ctx in args.contexts:
        prompt = render(processor, SHORT if ctx == 0 else long_text(root, ctx))
        mx.reset_peak_memory()
        cold = run_once(model, processor, prompt, 16, False)
        runs = [run_once(model, processor, prompt, args.max_tokens, False) for _ in range(args.reps)]
        mean = lambda k: sum(r[k] for r in runs) / len(runs)
        row = {
            "label": args.label, "model": str(args.model), "ctx": ctx, "prompt_tokens": runs[0]["prompt_tokens"],
            "cold_prefill": cold["prompt_tps"], "prefill": mean("prompt_tps"), "decode": mean("decode_tps"),
            "peak_gb": max(r["peak_gb"] for r in runs + [cold]), "swap_used_mb": psutil.swap_memory().used / 2**20,
            "ts": time.time(),
        }
        print(f"[{args.label}] ctx={row['prompt_tokens']:>6} cold_pf={row['cold_prefill']:.0f} pf={row['prefill']:.0f} "
              f"dec={row['decode']:.1f} tok/s peak={row['peak_gb']:.1f}GB swap={row['swap_used_mb']:.0f}MB", flush=True)
        with args.out.open("a") as f:
            f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
