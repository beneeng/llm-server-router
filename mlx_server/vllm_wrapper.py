"""
Drop-in wrapper around vllm-mlx that injects two patches before startup:

1. repetition_context_size support
   vllm-mlx exposes repetition_penalty as a request parameter but always calls
   make_logits_processors(...) without repetition_context_size, so mlx_lm's
   hardcoded default of 20 is used. This wrapper patches the function before
   any vllm_mlx module imports it, so every call site picks up the new default.

2. ThinkingAwareLogitsProcessor prompt-skip fix
   Root cause of infinite thinking loops in multi-turn conversations:
   The processor receives (prompt_tokens + generated_tokens) as context and
   processes all tokens through its THINKING-phase state machine. For multi-turn
   chats the prompt includes </think> from prior assistant turns; when the
   processor sees that token it transitions to CONTENT and then actively masks
   </think>, preventing it from ever being generated in the current turn.

   Fix: on the first call, skip prompt tokens by pre-initialising the processor's
   position counter to len(prompt_tokens). Only generated tokens are then fed
   through the state machine, so the thinking budget correctly counts only tokens
   the model actually generates inside the current <think> block.

Usage (via ModelConfig.vllm_args):
  python -m mlx_server.vllm_wrapper --repetition-context-size 512 serve MODEL --port PORT ...
"""
import argparse
import sys


def _patch_repetition_context_size(context_size: int) -> None:
    """Patch mlx_lm.sample_utils so every call uses the given repetition_context_size."""
    import mlx_lm.sample_utils as _su
    _orig = _su.make_logits_processors

    def _patched(*args, repetition_context_size: int = context_size, **kwargs):
        return _orig(*args, repetition_context_size=repetition_context_size, **kwargs)

    _su.make_logits_processors = _patched
    import logging as _logging
    _logging.getLogger("vllm_mlx.vllm_wrapper").info(
        "Patched make_logits_processors: repetition_context_size default -> %d", context_size
    )


def _patch_thinking_processor() -> None:
    """Fix multi-turn thinking loop bug in ThinkingAwareLogitsProcessor.

    On the first __call__ the processor receives all_prompt_tokens (the context
    accumulated by mlx_lm's TokenBuffer).  Any </think> token from a prior
    assistant turn that sits inside those prompt tokens causes a premature
    THINKING→CONTENT transition, after which the processor masks </think> and
    the model loops forever inside <think>.

    Fix: on the first call, pre-initialise _processed_len / _processed_token_ids
    to the full prompt length.  Subsequent calls then only advance the state
    machine over newly generated tokens, so the budget counts generated thinking
    tokens exclusively.
    """
    from vllm_mlx.constrained.thinking_processor import (
        ThinkingAwareLogitsProcessor,
        Phase,
    )
    import logging as _logging

    _log = _logging.getLogger("vllm_mlx.vllm_wrapper")
    _original_call = ThinkingAwareLogitsProcessor.__call__

    def _prompt_skipping_call(self, tokens, logits):  # noqa: ANN001
        # _processed_len == 0 means __init__ just ran and no tokens have been
        # fed through the state machine yet.  Use this as the "first call"
        # signal.  id()-based sets are unreliable because Python reuses object
        # IDs after GC: a recycled ID leaves a stale entry in the set, causing
        # the first call for a new request to skip straight to _original_call,
        # which then processes all prompt tokens (including prior </think>)
        # through the state machine and causes the CONTENT-phase </think> mask.
        if self._processed_len == 0:
            n = int(tokens.size)
            if n > 0:
                # Skip prompt tokens: pre-set the processed-position counter so
                # _sync_to_tokens only processes tokens beyond position n.
                # _processed_token_ids must match the actual token IDs so that
                # the common-prefix walk in _sync_to_tokens succeeds on the next call.
                self._processed_len = n
                self._processed_token_ids = tokens.tolist()
            # Apply the current phase (usually THINKING → pass through).
            # Budget=0 starts in TRANSITIONING and must force </think> immediately.
            state = self._state
            if state is Phase.TRANSITIONING:
                return self._force_transition(logits)
            if state is Phase.CONTENT:
                return self._call_inner(tokens, logits)
            return logits  # THINKING or IDLE: no constraint on first token

        return _original_call(self, tokens, logits)

    ThinkingAwareLogitsProcessor.__call__ = _prompt_skipping_call
    _log.info(
        "Patched ThinkingAwareLogitsProcessor: prompt tokens skipped on first call"
    )


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repetition-context-size", type=int, default=None)
    our_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    context_size = our_args.repetition_context_size
    if context_size is not None:
        # Import and patch mlx_lm.sample_utils BEFORE any vllm_mlx module runs.
        # vllm_mlx.scheduler and vllm_mlx.engine.simple both do:
        #   from mlx_lm.sample_utils import make_logits_processors
        # at module level. Since Python resolves "from X import Y" by looking up
        # Y on the already-cached sys.modules["X"], patching the attribute here
        # means every subsequent import of make_logits_processors gets our version.
        _patch_repetition_context_size(context_size)

    # Patch ThinkingAwareLogitsProcessor before the server starts.
    # server.py imports it lazily (inside _build_thinking_processor) on the first
    # request, so importing and patching the class here ensures the patched version
    # is used for every request.
    _patch_thinking_processor()

    from vllm_mlx.cli import main as _vllm_main
    sys.exit(_vllm_main())


if __name__ == "__main__":
    main()
