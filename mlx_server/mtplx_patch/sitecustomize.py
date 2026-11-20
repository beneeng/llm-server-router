"""Loaded automatically by the MTPLX server process (via PYTHONPATH).

MTPLX's memory guard calls a prefill "death_signature" when free pages are under
the abort floor while swap grows >= 64 MiB/s. Two things are wrong for this Mac:

1. The rate is a constant, not configurable. A healthy prefill that pushed
   ~65 MiB/s to swap was aborted with 30 GB "available".
   -> MLX_MTPLX_SWAP_RATE_MIB_S raises that line.
2. Swap growth is counted gross. macOS often moves already-compressed pages into
   swap, so the compressor shrinks by as much as swap grows (seen: compressor
   -0.59 GiB, swap +0.57 GiB in 3.2 s). RAM is not getting scarcer there, it is
   getting freer. -> MLX_MTPLX_NET_SWAP=1 counts swap growth net of compressor
   shrinkage. Compressor growth, free-page floors, the runaway and full-compressor
   rules are untouched, so a real death pattern (compressor AND swap growing) still fires.

Tied to mtplx 2.12.x private names (tests/test_backends.py checks they exist).
"""
import dataclasses
import os
import sys


def _net_swap(sm) -> None:
    original = sm.thrashing_base

    def thrashing_base(reading, previous):
        if reading is None or previous is None or reading.compressor_bytes is None:
            return original(reading, previous)
        single = isinstance(previous, sm.SystemMemory)
        adjusted = []
        for base in ([previous] if single else list(previous)):
            if (
                base is not None
                and base is not reading
                and base.compressor_bytes is not None
                and base.swap_used_bytes is not None
            ):
                shrink = max(0, int(base.compressor_bytes) - int(reading.compressor_bytes))
                if shrink:
                    # Raising the base swap lowers the measured growth by `shrink`
                    # (clone: SystemMemory is frozen, and `is` identity of the
                    # untouched readings must stay intact for the callers).
                    base = dataclasses.replace(base, swap_used_bytes=int(base.swap_used_bytes) + shrink)
            adjusted.append(base)
        return original(reading, adjusted[0] if single else adjusted)

    sm.thrashing_base = thrashing_base  # memory_thrashing() resolves it via module globals


def _apply() -> None:
    rate = os.environ.get("MLX_MTPLX_SWAP_RATE_MIB_S")
    net = os.environ.get("MLX_MTPLX_NET_SWAP") == "1"
    if not (rate or net):
        return
    try:
        from mtplx import system_memory as sm

        for name in ("_THRASH_SWAP_BYTES_PER_S", "thrashing_base", "SystemMemory"):
            if not hasattr(sm, name):
                raise AttributeError(f"{name} missing")
        if rate:
            sm._THRASH_SWAP_BYTES_PER_S = int(float(rate) * 1024 * 1024)
        if net:
            _net_swap(sm)
        print(f"[mlx-server] mtplx guard patch applied: swap_rate={rate or 'default'} MiB/s, net_swap={net}", file=sys.stderr)
    except Exception as exc:  # never block the server start, but say so loudly
        print(f"[mlx-server] WARNING: mtplx guard patch NOT applied: {exc!r}", file=sys.stderr)


_apply()
