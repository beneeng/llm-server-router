"""mlx-vlm tool-parser module for DeepSeek-V4's DSML tool-call format.

Follows the contract in ``mlx_vlm.tools.base.ToolParser``: two marker strings and
a ``parse_tool_call`` callable. The extractor slices the text between the
markers (exclusive) and hands it over; this module re-wraps it so DeepSeek's
reference parser (``parse_tool_calls``) can consume it unchanged.

Pure Python so it can be unit-tested without mlx.
"""
from __future__ import annotations

from typing import Any, Optional

from .encoding_dsv4 import dsml_token, parse_tool_calls, tool_calls_block_name

tool_call_start = f"<{dsml_token}{tool_calls_block_name}>"
tool_call_end = f"</{dsml_token}{tool_calls_block_name}>"


def parse_tool_call(text: str, tools: Optional[Any] = None) -> list[dict]:
    """Parse the body of one ``tool_calls`` block into ``[{name, arguments}]``.

    ``arguments`` is a JSON string, which the extractor passes through as is.
    """
    # parse_tool_calls expects to start right after "<｜DSML｜tool_calls" and to
    # see ">\n" before the first invoke and the closing tag at the end.
    wrapped = ">\n" + text.strip() + "\n" + tool_call_end
    _, _, calls = parse_tool_calls(0, wrapped)
    return calls
