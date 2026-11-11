"""patch_qwen_tool_parser: malformed array/object tool args must not raise SyntaxError.

Runs against the real qwen3_coder parser when mlx_vlm is importable (the
regression fixture is the argument that crashed a streaming request), and
against a fake module otherwise.
"""
import ast
import json
import types

import pytest

from mlx_server.mlx_vlm_wrapper import patch_qwen_tool_parser

# Real failure: nested array whose final bracket is missing.
MALFORMED = '[{"question": "Weg?", "type": "single", "options": [{"label": "A"}, {"label": "B"}]}'


def fake_parser():
    def _convert_param_value(param_value, param_name, param_config):
        try:
            return json.loads(param_value)
        except json.JSONDecodeError:
            return ast.literal_eval(param_value)

    return types.SimpleNamespace(_convert_param_value=_convert_param_value)


def test_unpatched_fake_reproduces_the_crash():
    with pytest.raises(SyntaxError):
        fake_parser()._convert_param_value(MALFORMED, "questions", {})


def test_patched_returns_raw_string_on_malformed_value():
    m = fake_parser()
    patch_qwen_tool_parser(m)
    assert m._convert_param_value(MALFORMED, "questions", {}) == MALFORMED


def test_patched_still_parses_valid_json():
    m = fake_parser()
    patch_qwen_tool_parser(m)
    assert m._convert_param_value('[{"a": 1}]', "questions", {}) == [{"a": 1}]


def test_patch_is_idempotent():
    m = fake_parser()
    patch_qwen_tool_parser(m)
    first = m._convert_param_value
    patch_qwen_tool_parser(m)
    assert m._convert_param_value is first


def test_real_parser_end_to_end():
    qwen3_coder = pytest.importorskip("mlx_vlm.tools.parsers.qwen3_coder")
    patch_qwen_tool_parser(qwen3_coder)
    tools = [{"type": "function", "function": {"name": "ask_user_question",
              "parameters": {"type": "object", "properties": {"questions": {"type": "array"}}}}}]
    call = f"<function=ask_user_question>\n<parameter=questions>\n{MALFORMED}\n</parameter>\n</function>"
    parsed = qwen3_coder.parse_tool_call(call, tools)
    assert parsed["name"] == "ask_user_question"
    assert parsed["arguments"]["questions"] == MALFORMED
