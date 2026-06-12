"""Integration tests for the two monkey-patches in mlx_server/vllm_wrapper.py.

Each test class covers one patch and has a pair of tests:
  - test_*_without_patch: confirms the known failure mode exists in upstream code
  - test_*_with_patch:    confirms the patch eliminates the failure

The only mock is the LLM: logits are uniform float arrays.  Framework code
(mlx_lm, vllm_mlx) runs un-mocked so these tests catch API changes on upgrade.

Run with:
    ~/.local/share/uv/tools/mlx-server/bin/python -m pytest tests/test_patches.py -v
"""

import gc
import inspect

import mlx.core as mx
import pytest

from vllm_mlx.constrained.thinking_processor import (
    Phase,
    ThinkingAwareLogitsProcessor,
)

# ── Token IDs for Qwen3 vocabulary (248320 tokens) ────────────────────────────
THINK_START = 248068  # <think>
THINK_END = 248069    # </think>
VOCAB_SIZE = 248320


# ── Helpers ───────────────────────────────────────────────────────────────────

def uniform_logits() -> mx.array:
    return mx.zeros([VOCAB_SIZE])


def make_processor(budget: int = 10, prompt_has_think: bool = True) -> ThinkingAwareLogitsProcessor:
    return ThinkingAwareLogitsProcessor(
        start_token_ids=[THINK_START],
        end_token_ids=[THINK_END],
        thinking_token_budget=budget,
        vocab_size=VOCAB_SIZE,
        prompt_has_think_tag=prompt_has_think,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def clean_make_logits_processors():
    """Save and restore mlx_lm.sample_utils.make_logits_processors."""
    import mlx_lm.sample_utils as su
    original = su.make_logits_processors
    yield
    su.make_logits_processors = original


@pytest.fixture()
def clean_thinking_processor_call():
    """Save and restore ThinkingAwareLogitsProcessor.__call__."""
    original = ThinkingAwareLogitsProcessor.__call__
    yield
    ThinkingAwareLogitsProcessor.__call__ = original


# ── Patch 1: repetition_context_size ─────────────────────────────────────────

class TestRepetitionContextSizePatch:
    """
    vllm-mlx always calls make_logits_processors() without repetition_context_size,
    so mlx_lm's hardcoded default of 20 is used regardless of the model config.
    _patch_repetition_context_size() overrides the default by wrapping the function.
    """

    def test_default_context_size_is_20_without_patch(self, clean_make_logits_processors):
        """Upstream mlx_lm hardcodes context_size=20 — the bug is real."""
        import mlx_lm.sample_utils as su

        procs = su.make_logits_processors(repetition_penalty=1.1)

        assert len(procs) == 1
        cvars = inspect.getclosurevars(procs[0])
        assert cvars.nonlocals["context_size"] == 20

    def test_patch_overrides_default_context_size(self, clean_make_logits_processors):
        """After applying the patch, calls without explicit context_size use the new default."""
        from mlx_server.vllm_wrapper import _patch_repetition_context_size
        _patch_repetition_context_size(512)

        import mlx_lm.sample_utils as su
        procs = su.make_logits_processors(repetition_penalty=1.1)

        assert len(procs) == 1
        cvars = inspect.getclosurevars(procs[0])
        assert cvars.nonlocals["context_size"] == 512

    def test_caller_explicit_value_still_wins(self, clean_make_logits_processors):
        """The patch only changes the default; an explicit caller value overrides it."""
        from mlx_server.vllm_wrapper import _patch_repetition_context_size
        _patch_repetition_context_size(512)

        import mlx_lm.sample_utils as su
        procs = su.make_logits_processors(repetition_penalty=1.1, repetition_context_size=64)

        cvars = inspect.getclosurevars(procs[0])
        assert cvars.nonlocals["context_size"] == 64


# ── Patch 2: ThinkingAwareLogitsProcessor multi-turn fix ─────────────────────

class TestThinkingProcessorPatch:
    """
    In a multi-turn conversation the prompt fed to the processor includes a
    prior </think> token (THINK_END = 248069).  The unpatched processor starts
    in THINKING phase (because prompt_has_think_tag=True) and processes ALL
    prompt tokens through its state machine; when it hits the prior </think> it
    transitions to CONTENT phase and permanently masks </think> at -inf.  The
    model can then never close its thinking block → infinite loop.

    The patch pre-sets _processed_len to the full first-call token count so the
    state machine only sees tokens generated after the prompt.
    """

    # Simulated multi-turn prompt:
    #   user turn 1 tokens  +  assistant turn 1 (<think>…</think>…content)
    #   + user turn 2 tokens  +  template-injected <think> for current turn
    PROMPT = (
        [1, 2, 3, 4, 5]                                      # user turn 1
        + [THINK_START, 100, 101, THINK_END, 200, 201]        # assistant turn 1
        + [6, 7, 8]                                           # user turn 2
        + [THINK_START]                                       # current-turn <think> from template
    )
    FIRST_GEN = 1234  # first token the model emits

    def _first_call_tokens(self) -> mx.array:
        # mlx_lm TokenBuffer: returns prompt + [generated_token] on first call
        return mx.array(self.PROMPT + [self.FIRST_GEN])

    # ── Necessity: confirm the failure mode ───────────────────────────────────

    def test_without_patch_prior_end_token_corrupts_phase(self, clean_thinking_processor_call):
        """
        Failure mode: unpatched processor enters CONTENT on the first call
        because it sees THINK_END in the multi-turn prompt.
        """
        assert THINK_END in self.PROMPT, "test setup error: THINK_END must be in prompt"

        proc = make_processor(budget=50)
        assert proc.state == Phase.THINKING  # starts correctly

        result = proc(self._first_call_tokens(), uniform_logits())

        # Processor hit THINK_END in prompt → jumped to CONTENT
        assert proc.state == Phase.CONTENT
        # CONTENT phase permanently masks </think>
        assert float(result[THINK_END]) == float("-inf")

    def test_without_patch_budget_never_triggers(self, clean_thinking_processor_call):
        """
        Consequence: because the processor is already in CONTENT, the thinking
        budget enforcement is bypassed forever.  </think> stays masked on every
        subsequent call — the infinite loop condition.
        """
        budget = 5
        proc = make_processor(budget=budget)
        proc(self._first_call_tokens(), uniform_logits())
        assert proc.state == Phase.CONTENT

        tokens = list(self.PROMPT) + [self.FIRST_GEN]
        for _ in range(budget + 10):
            tokens.append(2000)
            result = proc(mx.array(tokens), uniform_logits())
            assert float(result[THINK_END]) == float("-inf"), "THINK_END must stay masked"
            assert proc.state == Phase.CONTENT, "state must stay CONTENT forever"

    # ── Correctness: confirm the patch fixes it ───────────────────────────────

    def test_with_patch_first_call_skips_prompt_tokens(self, clean_thinking_processor_call):
        """
        With the patch, the first call skips all prompt tokens.
        The processor stays in THINKING and does not mask </think>.
        """
        from mlx_server.vllm_wrapper import _patch_thinking_processor
        _patch_thinking_processor()

        proc = make_processor(budget=50)
        result = proc(self._first_call_tokens(), uniform_logits())

        assert proc.state == Phase.THINKING
        assert float(result[THINK_END]) != float("-inf")

    def test_with_patch_budget_enforced_on_generated_tokens_only(self, clean_thinking_processor_call):
        """
        After the first-call skip, subsequent calls count only generated tokens
        against the budget.  TRANSITIONING triggers at exactly 'budget' generated tokens.
        """
        from mlx_server.vllm_wrapper import _patch_thinking_processor
        _patch_thinking_processor()

        budget = 5
        proc = make_processor(budget=budget)

        # First call: skip prompt
        tokens = list(self.PROMPT) + [self.FIRST_GEN]
        proc(mx.array(tokens), uniform_logits())
        assert proc.state == Phase.THINKING

        # Generate 'budget - 1' more tokens; each increments thinking_tokens by 1
        for i in range(budget - 1):
            tokens.append(2000 + i)
            proc(mx.array(tokens), uniform_logits())
            assert proc.state == Phase.THINKING, f"still thinking after {i + 1} gen tokens"

        # One more pushes thinking_tokens to budget → TRANSITIONING
        tokens.append(3000)
        proc(mx.array(tokens), uniform_logits())
        assert proc.state == Phase.TRANSITIONING

    def test_with_patch_transitioning_forces_end_token(self, clean_thinking_processor_call):
        """
        Once TRANSITIONING, _force_transition masks everything except THINK_END.
        The model is forced to emit </think>, ending the thinking block.
        """
        from mlx_server.vllm_wrapper import _patch_thinking_processor
        _patch_thinking_processor()

        budget = 3
        proc = make_processor(budget=budget)

        tokens = list(self.PROMPT) + [self.FIRST_GEN]
        proc(mx.array(tokens), uniform_logits())

        for i in range(budget):
            tokens.append(2000 + i)
            result = proc(mx.array(tokens), uniform_logits())

        assert proc.state == Phase.TRANSITIONING
        # Only THINK_END should be non -inf
        assert float(result[THINK_END]) != float("-inf")
        # A random non-special token should be -inf
        assert float(result[42]) == float("-inf")


# ── Regression: v1 patch's id()-reuse bug ────────────────────────────────────

class TestIdReuseRegression:
    """
    The v1 patch used id(self) in a set to track first-call state.  Python
    reuses object ids after GC, so a recycled id caused the first call for a
    new processor to be routed directly to the original __call__, feeding all
    prompt tokens through the state machine and triggering the bug.

    The v2 patch uses self._processed_len == 0 (always 0 after __init__) as
    the first-call signal, which is immune to id reuse.
    """

    def test_new_processor_is_not_affected_by_gc_of_old_processors(
        self, clean_thinking_processor_call
    ):
        """
        Even after many processors are GC'd (freeing their ids for reuse),
        a freshly created processor must correctly skip prompt tokens.
        """
        from mlx_server.vllm_wrapper import _patch_thinking_processor
        _patch_thinking_processor()

        # Create and discard many processors to maximize id-reuse opportunity
        for _ in range(200):
            p = make_processor()
            del p
        gc.collect()

        # Fresh processor must have _processed_len == 0 and skip prompt correctly
        prompt_with_prior_end = [1, 2, THINK_START, 100, THINK_END, 3, THINK_START]
        proc = make_processor(budget=50)

        assert proc._processed_len == 0, "_processed_len must be 0 after __init__"

        first_tokens = mx.array(prompt_with_prior_end + [999])
        proc(first_tokens, uniform_logits())

        assert proc.state == Phase.THINKING, (
            "ID-reuse regression: new processor processed all prompt tokens "
            "including prior THINK_END and entered CONTENT phase"
        )
