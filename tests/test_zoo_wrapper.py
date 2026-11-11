"""mlx_vlm_zoo_wrapper.patch_response_generator, tested with fakes.

unsloth_zoo is not installed in the main tool venv, so the scope factory is
injected. The fake ResponseGenerator mirrors the call order in
mlx_vlm/server/generation.py: _run -> _run_impl -> _initialize_model.
"""
import pytest

from mlx_server.mlx_vlm_zoo_wrapper import patch_response_generator


class Recorder:
    def __init__(self):
        self.events = []

    def mode(self, model):
        rec = self

        class Scope:
            def __enter__(self_):
                rec.events.append(("enter", model))
                return model

            def __exit__(self_, *exc):
                rec.events.append(("exit", model))
                return False

        return Scope()


def make_cls(rec, fail_in_run=False):
    class FakeResponseGenerator:
        def _initialize_model(self):
            rec.events.append(("load", None))
            self.model = "the-model"

        def _run_impl(self):
            self._initialize_model()
            rec.events.append(("serve", None))
            if fail_in_run:
                raise RuntimeError("boom")

        def _run(self):
            self._run_impl()

    return FakeResponseGenerator


def test_scope_opens_after_load_and_closes_after_serving():
    rec = Recorder()
    cls = make_cls(rec)
    patch_response_generator(cls, rec.mode)
    cls()._run()
    assert rec.events == [("load", None), ("enter", "the-model"), ("serve", None), ("exit", "the-model")]


def test_scope_closes_when_the_worker_crashes():
    rec = Recorder()
    cls = make_cls(rec, fail_in_run=True)
    patch_response_generator(cls, rec.mode)
    with pytest.raises(RuntimeError, match="boom"):
        cls()._run()
    assert rec.events[-1] == ("exit", "the-model")


def test_patch_is_idempotent_and_enters_once():
    rec = Recorder()
    cls = make_cls(rec)
    patch_response_generator(cls, rec.mode)
    patch_response_generator(cls, rec.mode)
    cls()._run()
    assert [e for e in rec.events if e[0] == "enter"] == [("enter", "the-model")]


def test_unpatched_class_never_opens_a_scope():
    """Documents the bug the wrapper fixes: stock mlx-vlm opens no zoo scope."""
    rec = Recorder()
    make_cls(rec)()._run()
    assert not [e for e in rec.events if e[0] == "enter"]
