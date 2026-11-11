"""
Thin wrapper around mlx_vlm.server that patches weight loading to use
strict=False, silently ignoring unrecognised weights (e.g. the MTP head in
Qwen3.6-35B-A3B-OptiQ-4bit) instead of raising ValueError.

Without the patch mlx_vlm.utils.load_model calls model.load_weights with the
default strict=True.  Qwen3.6 checkpoints include mtp.* (Multi-Token
Prediction) weights that the mlx_vlm model class does not define, so loading
fails.  The MTP head is only used for speculative decoding; single-step
inference works fine without it.
"""
from __future__ import annotations


def patch_qwen_tool_parser(parser_module) -> None:
    """Keep a malformed object/array tool argument from killing the whole stream.

    qwen3_coder's `_convert_param_value` falls back to `ast.literal_eval` when
    `json.loads` fails. A model that drops a closing bracket in a long nested
    argument makes that raise `SyntaxError`, which the streaming path
    (`mlx_vlm/tools/streaming.py`, `except (ValueError, KeyError, TypeError)`)
    does not catch: the request fails and all generated tokens, including text
    written before the tool call, are lost.

    The patch returns the raw string instead. The client then reports an
    argument validation error to the model, which can retry the call.
    `parser_module` is a parameter so the patch is testable without mlx_vlm.
    """
    if getattr(parser_module, "_malformed_args_patched", False):
        return
    original = parser_module._convert_param_value

    def _convert_param_value(param_value, param_name, param_config, *args, **kwargs):
        try:
            return original(param_value, param_name, param_config, *args, **kwargs)
        except SyntaxError:
            return param_value

    parser_module._convert_param_value = _convert_param_value
    parser_module._malformed_args_patched = True


def _apply_patches() -> None:
    import mlx.nn as nn
    import mlx_vlm.utils as _utils

    _orig_load_model = _utils.load_model
    _orig_load_weights = nn.Module.load_weights

    def _non_strict_load_weights(self, weights, strict: bool = True):
        return _orig_load_weights(self, weights, strict=False)

    def _patched_load_model(model_path, lazy: bool = False, **kwargs):
        # Temporarily make weight loading non-strict so extra checkpoint keys
        # (mtp.*) are silently discarded instead of raising ValueError.
        nn.Module.load_weights = _non_strict_load_weights
        try:
            return _orig_load_model(model_path, lazy, **kwargs)
        finally:
            nn.Module.load_weights = _orig_load_weights

    _utils.load_model = _patched_load_model

    # Patch ThinkingStreamState._strip_open_marker to handle compound open markers
    # that the tokenizer splits across multiple tokens (e.g. Gemma4's
    # "<|channel>thought" arrives as two tokens: "<|channel>" then "thought").
    # The original does a substring search for the full marker string, which never
    # matches when the marker is split. We accumulate fragments across calls so the
    # full marker can be detected and stripped, including the trailing newline.
    from mlx_vlm.server.responses_state import ThinkingStreamState

    def _patched_strip_open_marker(self, text: str) -> str:
        if getattr(self, "_just_stripped_marker", False):
            self._just_stripped_marker = False
            text = text.lstrip("\n")
            if not text:
                return ""

        combined = getattr(self, "_pending_open_fragment", "") + text
        for marker in self.open_markers:
            if marker in combined:
                before, after = combined.split(marker, 1)
                self._pending_open_fragment = ""
                self._just_stripped_marker = True
                return (before + after).lstrip("\n")
            if marker.startswith(combined):
                self._pending_open_fragment = combined
                return ""
        self._pending_open_fragment = ""
        return combined

    ThinkingStreamState._strip_open_marker = _patched_strip_open_marker

    from mlx_vlm.tools.parsers import qwen3_coder
    patch_qwen_tool_parser(qwen3_coder)


def main() -> None:
    _apply_patches()
    from mlx_vlm.server.cli import main as _server_main
    _server_main()


if __name__ == "__main__":
    main()
