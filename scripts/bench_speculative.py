#!/usr/bin/env python3
"""Benchmark speculative decoding against one running mlx-vlm backend.

Measures decode throughput *and* drafter acceptance, because tok/s alone can't
tell whether a change helped the drafter or just caught a cooler machine.
mlx-vlm returns both in the non-streaming response's `timings` object
(llama.cpp field names): `predicted_per_second`, `draft_rounds`, `draft_n`,
`draft_n_accepted`.

Talks to a model port directly (6301+), not the router — the router injects
per-model defaults and rewrites the body, which would make runs incomparable.

    # measure one variant
    scripts/bench_speculative.py --port 6399 --model mlx-community/Qwen3.8-27B-8bit \
        --label "mtp-block3" --out /tmp/bench.jsonl

    # compare everything recorded so far
    scripts/bench_speculative.py --compare /tmp/bench.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

# Fixed suite. Acceptance is workload-dependent: boilerplate-heavy code drafts
# far better than free prose, so a single prompt type would flatter or punish a
# drafter unfairly. Keep these stable across runs or comparisons are worthless.
PROMPTS: list[tuple[str, str]] = [
    (
        "code-boilerplate",
        "Write a Python dataclass `ServerConfig` with fields host, port, "
        "workers, timeout_seconds and log_level, including type annotations, "
        "sensible defaults, a `from_env` classmethod and a `__post_init__` "
        "that validates the port range. Code only.",
    ),
    (
        "code-algorithm",
        "Implement an LRU cache in Python with O(1) get and put using a dict "
        "plus a doubly linked list. Include the node class and short docstrings.",
    ),
    (
        "math-reasoning",
        "A train leaves station A at 14:20 travelling 82 km/h. A second train "
        "leaves station B, 451 km away, at 15:05 travelling 96 km/h toward A. "
        "At what clock time do they meet, and how far from A? Show each step.",
    ),
    (
        "structured-output",
        "Produce a JSON array of 8 objects describing European capitals, each "
        "with keys city, country, population_millions, founded_year, "
        "river. JSON only, no prose.",
    ),
    (
        "prose",
        "Explain to a senior backend engineer why speculative decoding does not "
        "change the output distribution of the target model, in four paragraphs.",
    ),
]

# Held fixed across variants. Thinking is off so output lengths stay comparable
# between runs; decode throughput itself is unaffected by it.
SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 20,
    "repetition_penalty": 1.05,
    "enable_thinking": False,
    "stream": False,
}


def long_prompt(approx_tokens: int) -> str:
    """Build a prompt of roughly `approx_tokens` tokens from this repo's own source.

    Real code (not repeated filler) so the drafter faces realistic text. ~3.5
    chars per token is an estimate; the actual count is in the response usage.
    """
    root = Path(__file__).resolve().parent.parent
    chunks: list[str] = []
    size = 0
    budget = int(approx_tokens * 3.5)
    for path in sorted(root.glob("mlx_server/**/*.py")) + sorted(root.glob("scripts/*.py")):
        text = path.read_text(errors="ignore")
        chunks.append(f"### {path.relative_to(root)}\n{text}")
        size += len(text)
        if size >= budget:
            break
    body = "\n\n".join(chunks)[:budget]
    return body + "\n\nSummarize what the code above does in four sentences."


def _post(client: httpx.Client, base: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        **SAMPLING,
    }
    started = time.perf_counter()
    resp = client.post(f"{base}/chat/completions", json=body)
    wall = time.perf_counter() - started
    resp.raise_for_status()
    data = resp.json()
    data["_wall_seconds"] = wall
    return data


def _row(prompt_id: str, data: dict) -> dict:
    timings = data.get("timings") or {}
    usage = data.get("usage") or {}
    rounds = timings.get("draft_rounds")
    accepted = timings.get("draft_n_accepted")
    drafted = timings.get("draft_n")
    return {
        "prompt": prompt_id,
        "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens") or 0,
        "decode_tps": timings.get("predicted_per_second"),
        "prefill_tps": timings.get("prompt_per_second"),
        "prefill_ms": timings.get("prompt_ms"),
        "draft_kind": timings.get("draft_kind"),
        "draft_rounds": rounds,
        "draft_n": drafted,
        "draft_n_accepted": accepted,
        # Tokens committed per target forward pass. 1.0 means the drafter never
        # helped; the theoretical ceiling is the block size.
        "tokens_per_round": ((accepted + rounds) / rounds) if rounds else None,
        "accept_ratio": (accepted / drafted) if drafted else None,
        "peak_memory_gb": timings.get("peak_memory"),
        "wall_seconds": data["_wall_seconds"],
    }


def _summarise(label: str, rows: list[dict]) -> dict:
    def mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.fmean(vals) if vals else None

    rounds = sum(r["draft_rounds"] or 0 for r in rows)
    accepted = sum(r["draft_n_accepted"] or 0 for r in rows)
    drafted = sum(r["draft_n"] or 0 for r in rows)
    return {
        "label": label,
        "prompts": len(rows),
        "decode_tps": mean("decode_tps"),
        "prefill_tps": mean("prefill_tps"),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "draft_kind": next((r["draft_kind"] for r in rows if r["draft_kind"]), None),
        "tokens_per_round": ((accepted + rounds) / rounds) if rounds else None,
        "accept_ratio": (accepted / drafted) if drafted else None,
        "peak_memory_gb": mean("peak_memory_gb"),
    }


def _fmt(value, spec: str = ".1f") -> str:
    return "-" if value is None else format(value, spec)


def _print_summaries(summaries: list[dict]) -> None:
    header = f"{'label':<22} {'dec tok/s':>10} {'tok/round':>10} {'accept':>8} {'pre tok/s':>10} {'kind':>7} {'peak GB':>8}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['label']:<22} {_fmt(s['decode_tps']):>10} {_fmt(s['tokens_per_round'], '.2f'):>10} "
            f"{_fmt(s['accept_ratio'], '.0%'):>8} {_fmt(s['prefill_tps']):>10} "
            f"{(s['draft_kind'] or '-'):>7} {_fmt(s['peak_memory_gb'], '.1f'):>8}"
        )
    if len(summaries) > 1:
        base = summaries[0]
        if base["decode_tps"]:
            print()
            for s in summaries[1:]:
                if s["decode_tps"]:
                    print(f"  {s['label']} vs {base['label']}: "
                          f"{s['decode_tps'] / base['decode_tps']:.2f}x decode")


def run(args: argparse.Namespace) -> int:
    base = f"http://127.0.0.1:{args.port}/v1"
    rows: list[dict] = []
    with httpx.Client(timeout=args.timeout) as client:
        # Warmup: first request pays Metal kernel compilation and weight
        # page-in, which would otherwise land entirely on the first prompt.
        print(f"[{args.label}] warmup...", file=sys.stderr)
        _post(client, base, args.model, "Say OK.", 16)

        prompts = list(PROMPTS) + [
            (f"long-{n}", long_prompt(n)) for n in (getattr(args, "long_context", None) or [])
        ]
        for repeat in range(args.repeat):
            for prompt_id, prompt in prompts:
                data = _post(client, base, args.model, prompt, args.max_tokens)
                row = _row(prompt_id, data)
                row["repeat"] = repeat
                rows.append(row)
                print(
                    f"[{args.label}] {prompt_id:<20} "
                    f"{_fmt(row['decode_tps']):>6} tok/s  "
                    f"{_fmt(row['tokens_per_round'], '.2f'):>5} tok/round  "
                    f"{row['output_tokens']:>4} tok",
                    file=sys.stderr,
                )

    summary = _summarise(args.label, rows)
    print()
    _print_summaries([summary])

    if args.out:
        record = {"summary": summary, "rows": rows, "recorded_at": time.time()}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"\nappended to {args.out}", file=sys.stderr)
    return 0


def compare(path: Path) -> int:
    summaries = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                summaries.append(json.loads(line)["summary"])
    if not summaries:
        print(f"no records in {path}", file=sys.stderr)
        return 1
    _print_summaries(summaries)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=6399, help="backend port (not the router)")
    parser.add_argument("--model", help="model id as the backend knows it (its `source`)")
    parser.add_argument("--label", default="run", help="name for this variant in the output")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--repeat", type=int, default=1, help="passes over the prompt suite")
    parser.add_argument("--long-context", type=int, action="append", metavar="TOKENS",
                        help="also run a prompt of ~TOKENS tokens built from this repo's source; repeatable")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--out", help="append the result as JSON to this file")
    parser.add_argument("--compare", type=Path, help="print a table of everything in this file and exit")
    args = parser.parse_args()

    if args.compare:
        return compare(args.compare)
    if not args.model:
        parser.error("--model is required unless --compare is given")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
