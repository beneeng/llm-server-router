"""
mlx_vlm.server with unsloth-zoo's qwen4_exp decode/verify fusions switched on.

unsloth-zoo PR #1871 routes Qwen3.8-Flash-Next decode steps onto fused Metal
kernels. The fusions are inference scopes opened by
`unsloth_zoo.mlx.generate.generation_mode(model)`; mlx_vlm.server never opens
one. Measured in-process: ~+16 % decode and ~1.8x prefill at 8k+ context.
Output is NOT bit-identical to stock (tokens diverge after a few dozen tokens),
see CLAUDE.md.

mlx-vlm's continuous-batching server runs every generation on one dedicated
thread: `ResponseGenerator._run` -> `_run_impl` -> `_initialize_model`. The
model only exists after `_initialize_model`, so the scope is entered right
after it and held until the thread exits. Entering once matters: the scope walks
the module graph on entry.

This wrapper runs in the isolated `zoo` env. It may import only stdlib,
mlx_vlm, unsloth_zoo and sibling wrappers by plain name (no `mlx_server`
package there).
"""
from __future__ import annotations

import contextlib
import functools


def patch_response_generator(cls, generation_mode) -> None:
    """Hold `generation_mode(self.model)` open for the lifetime of the worker thread.

    `cls` is mlx_vlm's `ResponseGenerator`; `generation_mode` is unsloth-zoo's
    context manager factory. Both are parameters so the patch is testable
    without either library.
    """
    if getattr(cls, "_zoo_patched", False):
        return
    original_init = cls._initialize_model
    original_run = cls._run

    @functools.wraps(original_init)
    def _initialize_model(self, *args, **kwargs):
        result = original_init(self, *args, **kwargs)
        stack = contextlib.ExitStack()
        stack.enter_context(generation_mode(self.model))
        self._zoo_scope = stack
        return result

    @functools.wraps(original_run)
    def _run(self, *args, **kwargs):
        try:
            return original_run(self, *args, **kwargs)
        finally:
            stack = getattr(self, "_zoo_scope", None)
            if stack is not None:
                self._zoo_scope = None
                stack.close()

    cls._initialize_model = _initialize_model
    cls._run = _run
    cls._zoo_patched = True


def main() -> None:
    # Plain-name sibling import: the script's own dir is on sys.path.
    from mlx_vlm_wrapper import _apply_patches

    _apply_patches()

    # Fail loudly instead of silently serving stock speed.
    from unsloth_zoo.mlx.generate import generation_mode
    from mlx_vlm.server.generation import ResponseGenerator

    patch_response_generator(ResponseGenerator, generation_mode)

    from mlx_vlm.server.cli import main as _server_main
    _server_main()


if __name__ == "__main__":
    main()
