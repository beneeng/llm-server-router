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


def main() -> None:
    _apply_patches()
    from mlx_vlm.server.cli import main as _server_main
    _server_main()


if __name__ == "__main__":
    main()
