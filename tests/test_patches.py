"""Tests for ThinkingBudgetProcessor in mlx_server/mlx_lm_wrapper.py.

The processor is tested directly (no server, no model) using synthetic token IDs
and uniform logits.  Framework code (mlx_lm) is not imported here.

Run with:
    ~/.local/share/uv/tools/mlx-server/bin/python -m pytest tests/test_patches.py -v
"""

import mlx.core as mx
import pytest

from mlx_server.mlx_lm_wrapper import ThinkingBudgetProcessor, _Phase


# ── Constants ─────────────────────────────────────────────────────────────────

THINK_START = 100   # synthetic <think> token ID
THINK_END   = 101   # synthetic </think> token ID
VOCAB_SIZE  = 1000  # small vocab for fast tests
OTHER       = 200   # an arbitrary normal token


# ── Helpers ───────────────────────────────────────────────────────────────────

def logits() -> mx.array:
    return mx.zeros([VOCAB_SIZE])


def make_proc(budget: int = 10, prompt_has_think_tag: bool = False) -> ThinkingBudgetProcessor:
    return ThinkingBudgetProcessor(
        start_token_ids=[THINK_START],
        end_token_ids=[THINK_END],
        budget=budget,
        prompt_has_think_tag=prompt_has_think_tag,
    )


def feed(proc: ThinkingBudgetProcessor, token_ids: list[int]) -> mx.array:
    """Feed a sequence of tokens, returning logits after the last one."""
    result = logits()
    for i, tid in enumerate(token_ids):
        result = proc(mx.array(token_ids[: i + 1]), logits())
    return result


# ── IDLE → THINKING → TRANSITIONING → CONTENT ────────────────────────────────

class TestBudgetEnforcement:
    def test_starts_idle(self):
        proc = make_proc(budget=5)
        assert proc.state is _Phase.IDLE

    def test_enters_thinking_on_start_token(self):
        proc = make_proc(budget=5)
        feed(proc, [THINK_START])
        assert proc.state is _Phase.THINKING

    def test_stays_thinking_below_budget(self):
        proc = make_proc(budget=5)
        feed(proc, [THINK_START, OTHER, OTHER, OTHER])
        assert proc.state is _Phase.THINKING
        assert proc.thinking_tokens == 3

    def test_transitions_at_budget(self):
        budget = 5
        proc = make_proc(budget=budget)
        feed(proc, [THINK_START] + [OTHER] * budget)
        assert proc.state is _Phase.TRANSITIONING

    def test_force_transition_masks_all_except_end(self):
        proc = make_proc(budget=2)
        result = feed(proc, [THINK_START, OTHER, OTHER])
        assert proc.state is _Phase.TRANSITIONING
        # Only THINK_END should survive
        assert float(result[THINK_END]) == 0.0
        assert float(result[OTHER]) == float("-inf")
        assert float(result[0]) == float("-inf")

    def test_enters_content_after_end_token(self):
        proc = make_proc(budget=2)
        feed(proc, [THINK_START, OTHER, OTHER])   # budget hit → TRANSITIONING
        feed(proc, [THINK_START, OTHER, OTHER, THINK_END])  # force complete
        assert proc.state is _Phase.CONTENT

    def test_natural_end_before_budget_enters_content(self):
        proc = make_proc(budget=10)
        feed(proc, [THINK_START, OTHER, THINK_END])
        assert proc.state is _Phase.CONTENT

    def test_content_phase_masks_think_start(self):
        proc = make_proc(budget=10)
        result = feed(proc, [THINK_START, OTHER, THINK_END, OTHER])
        assert proc.state is _Phase.CONTENT
        # <think> and </think> starts are masked to prevent re-entry
        assert float(result[THINK_START]) == float("-inf")
        assert float(result[THINK_END]) == float("-inf")


# ── prompt_has_think_tag ──────────────────────────────────────────────────────
# When the prompt prefills <think> (Qwen3 default), the processor starts in
# THINKING and the first non-empty call skips all tokens (they are prompt cache
# tokens, not generated tokens).  Budget counting begins with the second call.

class TestPromptHasThinkTag:
    def test_starts_in_thinking(self):
        proc = make_proc(budget=5, prompt_has_think_tag=True)
        assert proc.state is _Phase.THINKING

    def test_first_call_skips_prompt_tokens(self):
        # Simulates BatchGenerator: first call includes prompt cache tokens
        proc = make_proc(budget=3, prompt_has_think_tag=True)
        # First call with "prompt" tokens — should be a no-op
        proc(mx.array([OTHER, OTHER, OTHER]), logits())
        assert proc.thinking_tokens == 0
        assert proc.state is _Phase.THINKING

    def test_counts_from_second_call(self):
        proc = make_proc(budget=3, prompt_has_think_tag=True)
        proc(mx.array([OTHER]), logits())           # first call = prompt skip
        proc(mx.array([OTHER, OTHER]), logits())    # second call: 1 new token
        assert proc.thinking_tokens == 1
        assert proc.state is _Phase.THINKING

    def test_transitions_at_budget_after_skip(self):
        budget = 3
        proc = make_proc(budget=budget, prompt_has_think_tag=True)
        proc(mx.array([OTHER]), logits())                        # prompt skip
        proc(mx.array([OTHER] * (1 + budget)), logits())        # budget tokens
        assert proc.state is _Phase.TRANSITIONING

    def test_prompt_end_token_does_not_trigger_content(self):
        # Core regression: </think> sitting in prompt cache must NOT push the
        # processor to CONTENT and block future </think> generation.
        proc = make_proc(budget=10, prompt_has_think_tag=True)
        # Simulate prompt containing a prior turn's </think>
        proc(mx.array([THINK_END, OTHER]), logits())  # first call = skip
        assert proc.state is _Phase.THINKING          # must still be THINKING


# ── budget=0 ──────────────────────────────────────────────────────────────────

class TestBudgetZero:
    def test_prompt_tag_starts_transitioning(self):
        proc = make_proc(budget=0, prompt_has_think_tag=True)
        assert proc.state is _Phase.TRANSITIONING

    def test_forces_end_on_first_call(self):
        proc = make_proc(budget=0, prompt_has_think_tag=True)
        result = proc(mx.array([]), logits())  # empty tokens (pre-first-token call)
        assert float(result[THINK_END]) == 0.0
        assert float(result[OTHER]) == float("-inf")

    def test_forces_end_on_first_nonempty_call(self):
        proc = make_proc(budget=0, prompt_has_think_tag=True)
        result = proc(mx.array([OTHER]), logits())  # first non-empty = skip but TRANSITIONING
        assert float(result[THINK_END]) == 0.0
        assert float(result[OTHER]) == float("-inf")

    def test_idle_budget_zero_transitions_after_start(self):
        proc = make_proc(budget=0, prompt_has_think_tag=False)
        # IDLE until <think> seen, then immediately TRANSITIONING
        feed(proc, [OTHER, THINK_START])
        assert proc.state is _Phase.TRANSITIONING


# ── Rollback via snapshot/restore ─────────────────────────────────────────────

class TestSnapshotRestore:
    def test_rollback_resets_thinking_count(self):
        proc = make_proc(budget=10, prompt_has_think_tag=True)
        # Simulate: prompt skip on first call, then grow, then roll back
        proc(mx.array([OTHER]), logits())                           # prompt skip
        proc(mx.array([OTHER, OTHER, OTHER, OTHER, OTHER]), logits())  # 4 new tokens
        assert proc.thinking_tokens == 4

        proc(mx.array([OTHER, OTHER]), logits())  # roll back to 1 new token
        assert proc.thinking_tokens == 1

    def test_rollback_stays_in_thinking(self):
        proc = make_proc(budget=10, prompt_has_think_tag=True)
        proc(mx.array([OTHER]), logits())           # prompt skip
        proc(mx.array([OTHER] * 9), logits())       # 8 new tokens — near budget
        proc(mx.array([OTHER] * 4), logits())       # roll back to 3 new tokens
        assert proc.state is _Phase.THINKING
        assert proc.thinking_tokens == 3
