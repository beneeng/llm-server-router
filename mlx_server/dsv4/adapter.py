"""OpenAI-style chat messages -> DeepSeek-V4 prompt string.

The mlx-community DeepSeek-V4 checkpoints ship a stub ``chat_template.jinja``
that ignores tools, the ``tool`` role and ``tool_calls``. The real format lives
in DeepSeek's reference encoder (``encoding_dsv4.py``, vendored unchanged); this
module only normalises what mlx-vlm hands the processor so the encoder can be
fed directly.

Pure Python (stdlib only) so it imports without mlx and can be unit-tested.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from .encoding_dsv4 import encode_messages


def _content_text(content: Any) -> str:
    """Flatten OpenAI content (str, None or a list of blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in ("text", "input_text"):
                    parts.append(block.get("text", block.get("content", "")))
        return "".join(parts)
    return str(content)


def _normalise_tool_call(tool_call: dict) -> dict:
    """Ensure ``function.arguments`` is a JSON string (the encoder's input).

    mlx-vlm's ``_normalize_tool_message`` turns string arguments into dicts
    before the processor sees them; the encoder wants the OpenAI string form.
    """
    call = dict(tool_call)
    function = dict(call.get("function") or {})
    arguments = function.get("arguments", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    function["arguments"] = arguments
    call["function"] = function
    return call


def normalise_messages(messages: list[dict], tools: Optional[list[dict]] = None) -> list[dict]:
    """Return encoder-ready messages. Tools are attached to the system message."""
    out: list[dict] = []
    for message in messages:
        if isinstance(message, str):
            message = {"role": "user", "content": message}
        msg = dict(message)
        role = msg.get("role")
        if role == "developer":
            # OpenAI's newer name for the system role. DeepSeek's `developer`
            # role is reserved for its internal search pipeline.
            role = msg["role"] = "system"
        msg["content"] = _content_text(msg.get("content"))
        if msg.get("tool_calls"):
            msg["tool_calls"] = [_normalise_tool_call(tc) for tc in msg["tool_calls"]]
        else:
            msg.pop("tool_calls", None)
        out.append(msg)

    if tools:
        if out and out[0].get("role") == "system":
            out[0]["tools"] = tools
        else:
            out.insert(0, {"role": "system", "content": "", "tools": tools})
    return out


def build_prompt(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    enable_thinking: bool = True,
    reasoning_effort: Optional[str] = None,
) -> str:
    """Encode a conversation exactly like DeepSeek's reference implementation.

    For a conversation ending on a user (or tool) message the encoder already
    appends the assistant opener: ``<｜Assistant｜><think>`` in thinking mode,
    ``<｜Assistant｜></think>`` in chat mode. mlx-vlm always asks for a
    generation prompt, so no separate flag is needed.
    """
    normalised = normalise_messages(messages, tools)
    mode = "thinking" if enable_thinking else "chat"
    effort = reasoning_effort if reasoning_effort in ("max", "high") else None
    return encode_messages(normalised, thinking_mode=mode, reasoning_effort=effort)
