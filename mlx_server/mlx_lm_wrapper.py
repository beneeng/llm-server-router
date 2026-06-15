"""
Drop-in wrapper around mlx_lm.server that adds per-request thinking-budget
enforcement via a logits processor.

Usage (via ModelConfig.server_args):
    python mlx_server/mlx_lm_wrapper.py --model MODEL --port PORT [extra mlx_lm.server args]

Per-request API:
    Include "thinking_token_budget": N in the chat completion request body.
    When the model has generated N tokens inside a <think>…</think> span the
    processor forces </think> by masking every other token to -inf, then hands
    off to content generation normally.

Patches applied (4 total):
    1. APIHandler.handle_completion   — stash budget from request body (HTTP thread)
    2. ResponseGenerator.generate     — forward budget onto GenerationArguments (HTTP thread)
    3. ResponseGenerator._tokenize    — stash tokenizer + initial_state (generation thread)
    4. mlx_lm.server._make_logits_processors — inject ThinkingBudgetProcessor (generation thread)

Multi-turn note: mlx_lm's BatchGenerator initialises each request's TokenBuffer
with the cached prompt tokens, so the first logits-processor call receives
[prompt_tokens…, first_generated_token].  A </think> from a prior assistant turn
in those prompt tokens would transition the state machine to CONTENT and block
future </think> tokens.  ThinkingBudgetProcessor.__call__ detects the first
non-empty call via _processed_len == 0 and skips all tokens in that call,
mirroring the prompt-skip patch in the old vllm_wrapper.py.
"""
from __future__ import annotations

import enum
import threading
from collections import deque
from typing import Callable

import mlx.core as mx


# ── ThinkingBudgetProcessor ───────────────────────────────────────────────────
# Ported from vllm_mlx/constrained/thinking_processor.py (pure mlx + Python).
# Renamed; no other behavioural changes.

class _BoundedSuffixMatcher:
    """Rolling suffix buffer that fires when its tail equals a target sequence."""

    __slots__ = ("target", "_buf", "_max_len")

    def __init__(self, target_ids: list[int]) -> None:
        self.target = tuple(target_ids)
        self._max_len = len(target_ids)
        self._buf: deque[int] = deque(maxlen=self._max_len)

    def feed(self, token_id: int) -> bool:
        self._buf.append(token_id)
        return len(self._buf) == self._max_len and tuple(self._buf) == self.target

    def reset(self) -> None:
        self._buf.clear()

    def snapshot(self) -> tuple[int, ...]:
        return tuple(self._buf)

    def restore(self, state: tuple[int, ...]) -> None:
        self._buf.clear()
        self._buf.extend(state)


class _Phase(enum.Enum):
    IDLE         = "idle"
    THINKING     = "thinking"
    TRANSITIONING = "transitioning"
    CONTENT      = "content"


class ThinkingBudgetProcessor:
    """Force </think> after *budget* generated thinking tokens.

    State machine: IDLE → THINKING → TRANSITIONING → CONTENT.

    IDLE          Pass through until the <think> start sequence is seen.
    THINKING      Count generated tokens; at budget → TRANSITIONING.
    TRANSITIONING Mask all tokens to -inf except the next </think> token ID,
                  stepping through the end sequence one token at a time.
    CONTENT       After </think>: mask <think>/<think> starts, delegate to inner.

    Args:
        start_token_ids:   Token IDs that form the thinking-start sequence.
        end_token_ids:     Token IDs that form the thinking-end sequence.
        budget:            Max thinking tokens before forcing </think>.
        inner:             Optional logits processor to delegate to in CONTENT.
        prompt_has_think_tag: True when the chat template already injected
                           <think> into the prompt (Qwen3 prefill behaviour).
                           Starts in THINKING instead of IDLE.
    """

    __slots__ = (
        "_start_matcher",
        "_end_matcher",
        "_end_token_ids",
        "_content_phase_mask_ids",
        "_budget",
        "_inner",
        "_state",
        "_thinking_tokens",
        "_transition_index",
        "_prompt_has_think_tag",
        "_processed_len",
        "_processed_token_ids",
        "_snapshots",
    )

    def __init__(
        self,
        start_token_ids: list[int],
        end_token_ids: list[int],
        budget: int,
        inner: Callable[[mx.array, mx.array], mx.array] | None = None,
        prompt_has_think_tag: bool = False,
    ) -> None:
        self._start_matcher = _BoundedSuffixMatcher(start_token_ids)
        self._end_matcher = _BoundedSuffixMatcher(end_token_ids)
        self._end_token_ids = list(end_token_ids)
        self._content_phase_mask_ids = tuple(
            dict.fromkeys([start_token_ids[0], end_token_ids[0]])
        )
        self._budget = budget
        self._inner = inner
        self._thinking_tokens = 0
        self._transition_index = 0
        if prompt_has_think_tag:
            self._state = _Phase.TRANSITIONING if budget == 0 else _Phase.THINKING
        else:
            self._state = _Phase.IDLE
        self._prompt_has_think_tag = prompt_has_think_tag
        self._processed_len = 0
        self._processed_token_ids: list[int] = []
        self._snapshots = [self._snapshot()]

    @property
    def state(self) -> _Phase:
        return self._state

    @property
    def thinking_tokens(self) -> int:
        return self._thinking_tokens

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if tokens.size == 0:
            if self._state is _Phase.TRANSITIONING:
                return self._force_transition(logits)
            if self._state is _Phase.CONTENT:
                return self._call_inner(tokens, logits)
            return logits

        # First non-empty call when the prompt already prefilled <think>:
        # mlx_lm's BatchGenerator initialises TokenBuffer with cached prompt
        # tokens, so `tokens` here is [prompt_cache_tokens…, first_generated].
        # Any </think> from a prior assistant turn in those prompt tokens would
        # drive the state machine into CONTENT and permanently block future
        # </think> tokens.  When the prompt started in "reasoning" mode
        # (prompt_has_think_tag=True) we skip all tokens in this first call,
        # mirroring the prompt-skip patch in the old vllm_wrapper.py.
        # When prompt_has_think_tag=False the model generates <think> itself as
        # its first token, so we must NOT skip or the state machine never sees it.
        if self._processed_len == 0 and self._prompt_has_think_tag:
            n = int(tokens.size)
            self._processed_len = n
            self._processed_token_ids = tokens.tolist()
            # Fill snapshot slots for all skipped positions so that _restore(k)
            # for k ≤ n correctly returns thinking=0.  Without this, the next
            # _sync call would save snapshots starting at index 1 (not n+1),
            # shifting the restore index by n-1 and making rollback land wrong.
            snap = self._snapshot()
            for _ in range(n):
                self._snapshots.append(snap)
            if self._state is _Phase.TRANSITIONING:
                return self._force_transition(logits)
            if self._state is _Phase.CONTENT:
                return self._call_inner(tokens, logits)
            return logits  # THINKING: no constraint on first generated token

        self._sync(tokens)

        if self._state is _Phase.TRANSITIONING:
            return self._force_transition(logits)
        if self._state is _Phase.CONTENT:
            return self._call_inner(tokens, logits)
        return logits

    def _force_transition(self, logits: mx.array) -> mx.array:
        target_id = self._end_token_ids[self._transition_index]
        masked = mx.full(logits.shape, float("-inf"))
        if masked.ndim == 1:
            masked[target_id] = 0.0
        else:
            masked[..., target_id] = 0.0
        return masked

    def _call_inner(self, tokens: mx.array, logits: mx.array) -> mx.array:
        if self._inner is not None:
            logits = self._inner(tokens, logits)
        return self._mask_content_phase_tokens(logits)

    def _mask_content_phase_tokens(self, logits: mx.array) -> mx.array:
        for tid in self._content_phase_mask_ids:
            if logits.ndim == 1:
                logits[tid] = float("-inf")
            else:
                logits[..., tid] = float("-inf")
        return logits

    def _snapshot(self) -> tuple:
        return (
            self._state,
            self._thinking_tokens,
            self._transition_index,
            self._start_matcher.snapshot(),
            self._end_matcher.snapshot(),
        )

    def _restore(self, processed_len: int) -> None:
        idx = min(processed_len, len(self._snapshots) - 1)
        (
            self._state,
            self._thinking_tokens,
            self._transition_index,
            start_snap,
            end_snap,
        ) = self._snapshots[idx]
        self._start_matcher.restore(start_snap)
        self._end_matcher.restore(end_snap)
        self._processed_len = processed_len
        self._processed_token_ids = self._processed_token_ids[:processed_len]
        self._snapshots = self._snapshots[: idx + 1]

    def _sync(self, tokens: mx.array) -> None:
        target_len = int(tokens.size)
        token_ids = tokens.tolist()
        common = 0
        max_common = min(target_len, self._processed_len)
        while common < max_common and self._processed_token_ids[common] == token_ids[common]:
            common += 1
        if common < self._processed_len:
            self._restore(common)
        for tid in token_ids[self._processed_len:]:
            self._advance(tid)
            self._processed_token_ids.append(tid)
            self._processed_len += 1
            if self._state is not _Phase.CONTENT:
                self._snapshots.append(self._snapshot())

    def _advance(self, token_id: int) -> None:
        if self._state is _Phase.IDLE:
            if self._start_matcher.feed(token_id):
                self._state = _Phase.THINKING
                if self._budget == 0:
                    self._state = _Phase.TRANSITIONING
                    self._transition_index = 0
            return

        if self._state is _Phase.THINKING:
            if self._end_matcher.feed(token_id):
                self._state = _Phase.CONTENT
                return
            self._thinking_tokens += 1
            if self._thinking_tokens >= self._budget:
                self._state = _Phase.TRANSITIONING
                self._transition_index = 0
            return

        if self._state is _Phase.TRANSITIONING:
            expected = self._end_token_ids[self._transition_index]
            if token_id == expected:
                self._transition_index += 1
                if self._transition_index >= len(self._end_token_ids):
                    self._state = _Phase.CONTENT
                    self._end_matcher.reset()
            return


# ── Thread-locals ─────────────────────────────────────────────────────────────

_http_ctx = threading.local()  # HTTP handler thread: carries budget to queue
_gen_ctx  = threading.local()  # Generation thread: carries tokenizer + initial_state


# ── Patches ───────────────────────────────────────────────────────────────────

def _apply_patches() -> None:
    import mlx_lm.server as _srv
    from mlx_lm.server import APIHandler, ResponseGenerator

    # Patch 1 — stash thinking_token_budget from body (HTTP thread).
    # handle_completion is called after do_POST sets self.body, so self.body is
    # available.  We stash in _http_ctx so Patch 2 can read it moments later.
    _orig_handle_completion = APIHandler.handle_completion

    def _handle_completion(self, request, stop_words):
        _http_ctx.thinking_token_budget = self.body.get("thinking_token_budget")
        return _orig_handle_completion(self, request, stop_words)

    APIHandler.handle_completion = _handle_completion

    # Patch 2 — forward budget onto GenerationArguments before it enters the Queue
    # (still HTTP thread).  GenerationArguments is a non-frozen dataclass so
    # extra attributes can be set on the instance.
    _orig_generate = ResponseGenerator.generate

    def _generate(self, request, generation_args, progress_callback=None):
        budget = getattr(_http_ctx, "thinking_token_budget", None)
        if budget is not None:
            try:
                generation_args.thinking_token_budget = int(budget)
            except (TypeError, ValueError):
                pass
        return _orig_generate(self, request, generation_args, progress_callback)

    ResponseGenerator.generate = _generate

    # Patch 3 — stash tokenizer + initial_state in generation thread-local.
    # _tokenize is called in the generation thread immediately before
    # _make_logits_processors, so _gen_ctx is coherent between the two.
    _orig_tokenize = ResponseGenerator._tokenize

    def _tokenize(self, tokenizer, request, args):
        result = _orig_tokenize(self, tokenizer, request, args)
        _gen_ctx.tokenizer = tokenizer
        _gen_ctx.initial_state = result[3]  # "normal" or "reasoning"
        return result

    ResponseGenerator._tokenize = _tokenize

    # Patch 4 — inject ThinkingBudgetProcessor (generation thread).
    # mlx_lm.server resolves _make_logits_processors from its module globals at
    # call time, so replacing the module attribute is sufficient.
    _orig_make_lp = _srv._make_logits_processors

    def _make_logits_processors(args):
        processors = _orig_make_lp(args)
        budget = getattr(args, "thinking_token_budget", None)
        tokenizer = getattr(_gen_ctx, "tokenizer", None)
        initial_state = getattr(_gen_ctx, "initial_state", "normal")
        if budget and tokenizer and getattr(tokenizer, "has_thinking", False):
            processors.append(ThinkingBudgetProcessor(
                list(tokenizer.think_start_tokens),
                list(tokenizer.think_end_tokens),
                budget,
                prompt_has_think_tag=(initial_state == "reasoning"),
            ))
        return processors

    _srv._make_logits_processors = _make_logits_processors


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _apply_patches()
    from mlx_lm.server import main as _server_main
    _server_main()


if __name__ == "__main__":
    main()
