#!/usr/bin/env python3
"""Run bench_speculative.py across several drafter configurations.

Each variant gets a cold backend process on a dedicated bench port, so the
drafter settings (which mlx-vlm reads from the environment at startup, not per
request) actually differ between runs. The model's normal port and the router
are left alone.

Base argv comes from this repo's own `ModelConfig.server_args()`, so a variant
is only the extra flags on top. mlx-vlm's argparse takes the last occurrence of
a flag, so appending `--draft-model X` overrides the configured drafter.

    scripts/bench_draft_variants.py --model qwen3.8-27b \
        --variant "mtp-block3:" \
        --variant "mtp-block5:--draft-block-size 5" \
        --variant "dflash2:--draft-model incoai/Qwen3.8-27B-DFlash2 --draft-kind dflash"

APC is disabled for every variant: prefix-cache hits would make prefill numbers
depend on run order instead of on the drafter.
"""
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import bench_speculative as bench  # noqa: E402

from mlx_server.config import load_config  # noqa: E402

LOG_DIR = Path("~/.local/share/mlx/logs").expanduser()


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"backend exited with code {proc.returncode} before serving")
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"backend did not answer on port {port} within {timeout:.0f}s")


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


MAX_SWAP_USED_GB = 3.0
GUARD_WAIT_SECONDS = 90.0  # macOS releases a stopped backend's memory with a delay


def memory_guard(model_cfg, min_free_gb: float) -> str | None:
    """Return a reason to refuse starting the backend, or None when it is safe.

    This script bypasses the router's MemoryManager, so it needs its own floor:
    a model that does not fit pushes the machine into swap and freezes it.
    """
    available = psutil.virtual_memory().available / 2**30
    swap_used = psutil.swap_memory().used / 2**30
    if swap_used > MAX_SWAP_USED_GB:
        return f"swap already {swap_used:.1f} GB used (limit {MAX_SWAP_USED_GB:.0f}); reboot or free memory first"
    if available - model_cfg.memory_gb < min_free_gb:
        return (f"{available:.1f} GB available - {model_cfg.memory_gb} GB model "
                f"would leave less than {min_free_gb:.0f} GB free")
    return None


def run_variant(model_cfg, label: str, extra: list[str], args: argparse.Namespace) -> dict | None:
    deadline = time.time() + GUARD_WAIT_SECONDS
    refusal = memory_guard(model_cfg, args.min_free_gb)
    while refusal and time.time() < deadline:
        time.sleep(5.0)
        refusal = memory_guard(model_cfg, args.min_free_gb)
    if refusal:
        print(f"\n=== {label} === REFUSED: {refusal}", file=sys.stderr)
        return None
    argv = model_cfg.server_args()
    port_idx = argv.index("--port")
    argv[port_idx + 1] = str(args.port)
    if args.python:
        argv[0] = args.python  # benchmark another mlx-vlm version without touching the tool venv
    argv += extra

    env = os.environ | model_cfg.resolve_backend().env_vars(model_cfg)
    env["APC_ENABLED"] = "0"
    env.pop("APC_DISK_PATH", None)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"bench-{label}.log"
    print(f"\n=== {label} ===")
    print(f"  argv: {shlex.join(argv[1:])}")
    print(f"  log:  {log_path}")

    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(argv, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        try:
            print(f"  waiting for port {args.port} (up to {args.startup_timeout:.0f}s)...")
            _wait_for_port(args.port, proc, args.startup_timeout)
            bench_args = argparse.Namespace(
                port=args.port,
                model=model_cfg.source,
                label=label,
                max_tokens=args.max_tokens,
                repeat=args.repeat,
                timeout=args.request_timeout,
                long_context=args.long_context,
                out=args.out,
            )
            bench.run(bench_args)
        except Exception as exc:  # keep going: one broken variant shouldn't lose the rest
            print(f"  FAILED: {exc}", file=sys.stderr)
            print(f"  see {log_path}", file=sys.stderr)
            return None
        finally:
            _stop(proc)
    return {"label": label}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model name from models.yaml")
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="LABEL:FLAGS",
        help="repeatable; extra flags appended to the configured argv",
    )
    parser.add_argument("--port", type=int, default=6399, help="bench port, kept clear of the 6301+ block")
    parser.add_argument("--python", help="interpreter for the backend (e.g. an isolated venv with another mlx-vlm)")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--long-context", type=int, action="append", metavar="TOKENS",
                        help="also run a ~TOKENS-token prompt built from this repo's source; repeatable")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=900.0,
                        help="raise this when a variant has to download a drafter first")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--out", default="/tmp/bench_speculative.jsonl")
    args = parser.parse_args()

    cfg = load_config()
    args.min_free_gb = cfg.min_free_memory_gb
    matches = [m for m in cfg.models if m.name == args.model]
    if not matches:
        print(f"unknown model {args.model!r}; known: {', '.join(m.name for m in cfg.models)}", file=sys.stderr)
        return 1
    model_cfg = matches[0]

    variants = []
    for raw in args.variant:
        label, _, flags = raw.partition(":")
        variants.append((label.strip(), shlex.split(flags)))

    for label, extra in variants:
        run_variant(model_cfg, label, extra, args)

    print("\n=== summary ===")
    bench.compare(Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
