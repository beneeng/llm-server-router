"""
Wrapper around mlx_vlm.server for DeepSeek-V4 checkpoints.

The mlx-community DeepSeek-V4 repos ship a stub chat template that ignores tools,
the `tool` role and `tool_calls`, and mlx-vlm has no parser for DeepSeek's DSML
tool-call format. This wrapper fixes both without touching site-packages:

1. `DeepseekV4Processor.apply_chat_template` is replaced by DeepSeek's reference
   encoder (vendored in `mlx_server/dsv4/`), so tools, tool results and the
   thinking mode reach the model in the format it was trained on.
2. A DSML tool-parser module is registered in mlx-vlm's tool registry
   (`deepseek_v4_dsml`); the router selects it per request via `tool_parser`.

The encoder emits `<｜begin▁of▁sentence｜>` itself. The checkpoint's tokenizer does
not add a second one (verified: exactly one BOS with and without
`add_special_tokens`), so no tokenization patch is needed.

Deliberately NOT applied here: `mlx_vlm_wrapper`'s non-strict weight loading.
Strict loading is what exposed the mlx-vlm 0.7.0 key-remap bug for this
checkpoint (garbage output instead of a load error); keep it loud.
"""
from __future__ import annotations

import sys

PARSER_NAME = "deepseek_v4_dsml"


def patch_processor(processor_cls) -> None:
    """Route chat templating through DeepSeek's encoder."""
    from mlx_server.dsv4.adapter import build_prompt

    def apply_chat_template(
        self,
        conversation,
        *args,
        tools=None,
        enable_thinking=True,
        reasoning_effort=None,
        tokenize=False,
        **kwargs,
    ):
        if isinstance(conversation, dict):
            conversation = [conversation]
        prompt = build_prompt(
            list(conversation),
            tools=tools or None,
            enable_thinking=bool(enable_thinking),
            reasoning_effort=reasoning_effort,
        )
        if tokenize:
            return self.tokenizer.encode(prompt, add_special_tokens=False)
        return prompt

    processor_cls.apply_chat_template = apply_chat_template


def register_tool_parser(registry) -> None:
    """Make `deepseek_v4_dsml` resolvable by mlx_vlm.tools.registry."""
    from mlx_server.dsv4 import dsml_parser

    sys.modules[f"mlx_vlm.tools.parsers.{PARSER_NAME}"] = dsml_parser
    spec = registry.ParserSpec(PARSER_NAME, ((dsml_parser.tool_call_start,),))
    specs = tuple(s for s in registry.SPECS if s.name != PARSER_NAME) + (spec,)
    registry.SPECS = specs
    registry._BY_NAME[PARSER_NAME] = spec
    # The request schema validates `tool_parser` against `mlx_vlm.tools.SPECS`,
    # which is a separate binding of the same tuple (`from .registry import SPECS`).
    import mlx_vlm.tools as tools_pkg

    tools_pkg.SPECS = specs


def _apply_patches() -> None:
    from mlx_vlm.models.deepseek_v4.processing_deepseek_v4 import DeepseekV4Processor
    from mlx_vlm.tools import registry

    patch_processor(DeepseekV4Processor)
    register_tool_parser(registry)


def main() -> None:
    _apply_patches()
    from mlx_vlm.server.cli import main as _server_main
    _server_main()


if __name__ == "__main__":
    main()
