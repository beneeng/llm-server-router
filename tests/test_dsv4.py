"""DeepSeek-V4 encoding, adapter and DSML parser tests.

Pure-Python (no mlx, no model weights). The fixtures under tests/fixtures/dsv4
are DeepSeek's official encoder test vectors (MIT, see THIRD_PARTY_NOTICES.md).
"""
import json
from pathlib import Path

import pytest

from mlx_server.dsv4 import dsml_parser
from mlx_server.dsv4.adapter import build_prompt, normalise_messages
from mlx_server.dsv4.encoding_dsv4 import (
    encode_messages,
    parse_message_from_completion_text,
)

FIX = Path(__file__).parent / "fixtures" / "dsv4"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
}]


# ── official vectors through the vendored encoder ────────────────────────────

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_official_vectors(n):
    data = json.loads((FIX / f"test_input_{n}.json").read_text())
    gold = (FIX / f"test_output_{n}.txt").read_text()
    if isinstance(data, dict):
        messages = data["messages"]
        if "tools" in data:
            messages[0]["tools"] = data["tools"]
    else:
        messages = data
    mode = "chat" if n == 4 else "thinking"  # mirrors upstream test_encoding_dsv4.py
    assert encode_messages(messages, thinking_mode=mode) == gold


# ── adapter: what the stub template got wrong ────────────────────────────────

def test_stub_template_drops_tools_but_adapter_renders_them():
    prompt = build_prompt([{"role": "user", "content": "weather in Berlin?"}], tools=TOOLS)
    assert "## Tools" in prompt
    assert "get_weather" in prompt
    assert prompt.startswith("<｜begin▁of▁sentence｜>")
    assert prompt.count("<｜begin▁of▁sentence｜>") == 1
    assert prompt.endswith("<｜Assistant｜><think>")


def test_chat_mode_closes_thinking():
    prompt = build_prompt([{"role": "user", "content": "hi"}], enable_thinking=False)
    assert prompt.endswith("<｜Assistant｜></think>")


def test_tools_attach_to_existing_system_message():
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    out = normalise_messages(msgs, TOOLS)
    assert out[0]["tools"] == TOOLS and len(out) == 2
    assert build_prompt(msgs, TOOLS).startswith("<｜begin▁of▁sentence｜>SYS")


def test_developer_role_is_treated_as_system():
    out = normalise_messages([{"role": "developer", "content": "D"}, {"role": "user", "content": "x"}])
    assert out[0]["role"] == "system"


def test_content_blocks_and_none_are_flattened():
    out = normalise_messages([
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "assistant", "content": None},
    ])
    assert out[0]["content"] == "ab" and out[1]["content"] == ""


def test_tool_round_trip_with_dict_arguments():
    """mlx-vlm hands over arguments as dicts; the encoder needs JSON strings."""
    msgs = [
        {"role": "user", "content": "weather in Berlin?"},
        {"role": "assistant", "content": "", "reasoning_content": "need the tool",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "get_weather", "arguments": {"city": "Berlin", "days": 2}}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    ]
    prompt = build_prompt(msgs, tools=TOOLS)
    assert '<｜DSML｜invoke name="get_weather">' in prompt
    assert '<｜DSML｜parameter name="city" string="true">Berlin</｜DSML｜parameter>' in prompt
    assert '<｜DSML｜parameter name="days" string="false">2</｜DSML｜parameter>' in prompt
    assert "sunny" in prompt
    assert prompt.endswith("<｜Assistant｜><think>")


def test_empty_tool_calls_list_is_ignored():
    out = normalise_messages([{"role": "assistant", "content": "x", "tool_calls": []}])
    assert "tool_calls" not in out[0]


# ── DSML parser (mlx-vlm tool-parser contract) ───────────────────────────────

def _block(*invokes: str) -> str:
    return "\n".join(invokes)


def _invoke(name: str, *params: str) -> str:
    return f'<｜DSML｜invoke name="{name}">\n' + "\n".join(params) + "\n</｜DSML｜invoke>"


def _param(name: str, value: str, is_str: bool) -> str:
    return (f'<｜DSML｜parameter name="{name}" string="{str(is_str).lower()}">'
            f"{value}</｜DSML｜parameter>")


def test_parser_markers_match_dsml():
    assert dsml_parser.tool_call_start == "<｜DSML｜tool_calls>"
    assert dsml_parser.tool_call_end == "</｜DSML｜tool_calls>"


def test_parse_single_call():
    body = _block(_invoke("get_weather", _param("city", "Berlin", True), _param("days", "2", False)))
    calls = dsml_parser.parse_tool_call(f"\n{body}\n")
    assert [c["name"] for c in calls] == ["get_weather"]
    assert json.loads(calls[0]["arguments"]) == {"city": "Berlin", "days": 2}


def test_parse_multiple_calls_and_multiline_value():
    body = _block(
        _invoke("write", _param("path", "a.py", True), _param("text", "x = 1\ny = 2", True)),
        _invoke("noop"),
    )
    calls = dsml_parser.parse_tool_call(body)
    assert [c["name"] for c in calls] == ["write", "noop"]
    assert json.loads(calls[0]["arguments"])["text"] == "x = 1\ny = 2"
    assert json.loads(calls[1]["arguments"]) == {}


def test_parse_json_typed_parameter():
    body = _invoke("f", _param("opts", '{"a": [1, 2]}', False))
    assert json.loads(dsml_parser.parse_tool_call(body)[0]["arguments"]) == {"opts": {"a": [1, 2]}}


def test_encode_then_parse_roundtrip():
    """What the encoder renders for an assistant call must parse back identically."""
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function",
                         "function": {"name": "get_weather",
                                      "arguments": json.dumps({"city": "Köln", "days": 3})}}]},
    ]
    prompt = build_prompt(msgs, tools=TOOLS)
    # The tools preamble also contains the markers as an example; use the last block.
    start = prompt.rindex(dsml_parser.tool_call_start) + len(dsml_parser.tool_call_start)
    end = prompt.rindex(dsml_parser.tool_call_end)
    calls = dsml_parser.parse_tool_call(prompt[start:end])
    assert calls[0]["name"] == "get_weather"
    assert json.loads(calls[0]["arguments"]) == {"city": "Köln", "days": 3}


def test_full_completion_parses_with_reference_parser():
    text = ("reasoning</think>\n\n<｜DSML｜tool_calls>\n"
            + _block(_invoke("get_weather", _param("city", "Rome", True)))
            + "\n</｜DSML｜tool_calls><｜end▁of▁sentence｜>")
    msg = parse_message_from_completion_text(text, thinking_mode="thinking")
    assert msg["reasoning_content"] == "reasoning"
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"


def test_malformed_block_raises():
    with pytest.raises(Exception):
        dsml_parser.parse_tool_call("not dsml at all")


# ── wrapper patches (need mlx-vlm >= 0.7.11: mlx_vlm.tools registry) ─────────

class TestWrapperPatches:
    @pytest.fixture(autouse=True)
    def _need_new_mlx_vlm(self):
        pytest.importorskip("mlx_vlm.tools.registry")
        pytest.importorskip("mlx_vlm.models.deepseek_v4.processing_deepseek_v4")

    def test_stub_template_loses_tools_patched_processor_keeps_them(self):
        """Reproduces the bug (stub drops tools) and shows the patch fixes it."""
        from mlx_server import mlx_vlm_dsv4_wrapper as w

        class FakeTokenizer:
            """Mimics the mlx-community stub template: no tools, no tool role."""
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
                return "".join(f"<{m['role']}>{m['content']}" for m in messages)

        class Proc:
            tokenizer = FakeTokenizer()
            def apply_chat_template(self, conversation, *a, **kw):
                return self.tokenizer.apply_chat_template(conversation, *a, **kw)
            def __call__(self, text, *a, **kw):
                return kw

        msgs = [{"role": "user", "content": "weather?"}]
        assert "get_weather" not in Proc().apply_chat_template(msgs, tools=TOOLS)

        w.patch_processor(Proc)
        out = Proc().apply_chat_template(msgs, tools=TOOLS, enable_thinking=True)
        assert "## Tools" in out and "get_weather" in out
        assert out.endswith("<｜Assistant｜><think>")
        chat = Proc().apply_chat_template(msgs, tools=None, enable_thinking=False)
        assert chat.endswith("<｜Assistant｜></think>")

    def test_parser_registered_and_loadable(self):
        from mlx_vlm.tools import registry
        from mlx_server import mlx_vlm_dsv4_wrapper as w

        w.register_tool_parser(registry)
        assert registry._infer_tool_parser(None, override="deepseek_v4_dsml") == "deepseek_v4_dsml"
        module = registry.load_tool_module("deepseek_v4_dsml")
        assert module.tool_call_start == "<｜DSML｜tool_calls>"

    def test_extractor_turns_dsml_into_openai_tool_calls(self):
        from mlx_vlm.tools import registry
        from mlx_vlm.tools.extractor import process_tool_calls
        from mlx_server import mlx_vlm_dsv4_wrapper as w

        w.register_tool_parser(registry)
        module = registry.load_tool_module("deepseek_v4_dsml")
        output = ("Let me check.\n\n<｜DSML｜tool_calls>\n"
                  + _invoke("get_weather", _param("city", "Berlin", True), _param("days", "2", False))
                  + "\n</｜DSML｜tool_calls>")
        result = process_tool_calls(output, module, TOOLS)
        assert result.remaining_text == "Let me check."
        call = result.calls[0]["function"]
        assert call["name"] == "get_weather"
        assert json.loads(call["arguments"]) == {"city": "Berlin", "days": 2}
