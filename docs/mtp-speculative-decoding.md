# Handover: MTP Speculative Decoding for Qwen3.6 Models

## Task

Wire up the native MTP (Multi-Token Prediction) speculative decoding head that is already
present in all three Qwen3.6 checkpoints. The infrastructure is in place; only the split
step and a config change remain.

## Background: what MTP is

Qwen3.6 checkpoints ship with a bundled MTP head — a single extra transformer layer whose
weights live in `mtp.safetensors` alongside the main model. At generation time the MTP head
proposes one draft token; the main model verifies it in the same forward pass. Accepted
drafts are free — effective throughput rises ~1.5–2× with no quality change. `mlx_vlm`
0.6.3 has full support for this via its speculative-decoding subsystem.

## Current state

All three Qwen models in `~/.config/mlx/models.yaml` are `vlm: true` and started via
`mlx_server/mlx_vlm_wrapper.py` → `mlx_vlm.server`. None currently use the MTP head.

| Model name | Source HF repo | MTP drafter |
|---|---|---|
| `qwen3.6-moe-8bit` | `mlx-community/Qwen3.6-35B-A3B-8bit` | `mlx-community/Qwen3.6-35B-A3B-MTP-5bit` (separate repo) |
| `qwen3.6-moe-mxfp4` | `mlx-community/Qwen3.6-35B-A3B-mxfp4` | `mtp.safetensors` embedded in checkpoint |
| `qwen3.6-27b` | `mlx-community/Qwen3.6-27B-nvfp4` | check (likely has embedded MTP) |

The 8-bit checkpoint does NOT embed MTP — the drafter is a standalone HF repo
(`mlx-community/Qwen3.6-35B-A3B-MTP-5bit`). Download is automatic when mlx_vlm
starts with `--draft-model mlx-community/Qwen3.6-35B-A3B-MTP-5bit`.

## How to activate — step by step

### Step 1 — Extract the MTP head (one-time per model)

`mlx_vlm` provides a split tool that reads `mtp.*` tensors from the existing HF cache and
writes a standalone drafter directory. No re-download is needed.

```bash
PYTHON=~/.local/share/uv/tools/mlx-server/bin/python3

# qwen3.6-moe-8bit — uses standalone drafter, no split step needed
# Just add --draft-model mlx-community/Qwen3.6-35B-A3B-MTP-5bit directly

# qwen3.6-moe-mxfp4
$PYTHON -m mlx_vlm.speculative.drafters.qwen3_5_mtp.split \
  --model mlx-community/Qwen3.6-35B-A3B-mxfp4 \
  --output ~/.cache/mlx-mtp/qwen3.6-moe-mxfp4

# qwen3.6-27b (if mtp.safetensors is present in its cache — verify first)
$PYTHON -m mlx_vlm.speculative.drafters.qwen3_5_mtp.split \
  --model mlx-community/Qwen3.6-27B-nvfp4 \
  --output ~/.cache/mlx-mtp/qwen3.6-27b
```

Each output directory will contain `model.safetensors` + `config.json` with
`"model_type": "qwen3_5_mtp"`. `mlx_vlm` auto-detects `--draft-kind mtp` from that type.

### Step 2 — Add `extra_args` to `~/.config/mlx/models.yaml`

For each model, add the `--draft-model` flag. Example for `qwen3.6-moe-8bit`:

```yaml
- name: qwen3.6-moe-8bit
  source: mlx-community/Qwen3.6-35B-A3B-8bit
  vlm: true
  multimodal: true
  mtp: true        # signals router to strip thinking_budget (see Step 3)
  memory_gb: 40
  extra_args:
    - "--draft-model"
    - "mlx-community/Qwen3.6-35B-A3B-MTP-5bit"
  defaults:
    temperature: 0.7
    ...
```

For `qwen3.6-moe-mxfp4` the drafter is embedded in the checkpoint; use the split tool
path from `~/.cache/mlx-mtp/qwen3.6-moe-mxfp4` instead.
`--draft-kind mtp` is auto-resolved from the drafter's `model_type` — do not set it
manually unless auto-detection fails.

Also mirror these changes into the repo copy at
`~/workspace/local_llm/config/models.yaml`.

### Step 3 — Restart

```bash
~/.local/bin/mlx stop && ~/.local/bin/mlx start
```

The mlx_vlm_wrapper passes `extra_args` verbatim to `mlx_vlm.server`. The server picks up
`MLX_VLM_DRAFT_MODEL` and `MLX_VLM_DRAFT_KIND` env vars set by its own CLI and loads the
drafter at startup alongside the main model.

## Critical constraint: thinking_budget + MTP are mutually exclusive

`mlx_vlm` 0.6.3 explicitly rejects `thinking_budget` when a draft model is active
(`generation.py:987`: `"thinking_budget is not supported with speculative decoding"`). This
matters because the router injects `thinking_token_budget` from `ModelDefaults` into every
Qwen request, which the router then renames to `thinking_budget` for vlm models
(`router.py` vlm branch).

**Consequence:** if MTP is enabled server-side, every Qwen request that carries
`thinking_budget` will fail with a 500.

### Why this conflict is architectural, not fixable per-request

The conflict is not an implementation detail — it is structural. In normal generation, tokens
are sampled one at a time: a logits processor (our `ThinkingBudgetProcessor` in
`mlx_lm_wrapper.py`) or mlx_vlm's native `ThinkingBudgetCriteria` can intervene at every
step. In speculative decoding the draft model proposes a batch of N tokens independently,
without any logits processors from the main model. The main model then verifies the batch
atomically. If token 3 of a 5-token draft batch crosses the budget boundary, there is no
clean insertion point for `</think>` — splitting the batch would negate the entire throughput
benefit. Porting our logits-processor implementation to vlm has the same problem: draft
tokens bypass it entirely.

There is no implementation path that gives both MTP and a hard thinking-token cap with the
current mlx_vlm architecture. Per-request dynamic switching (e.g. disabling the draft model
when `thinking_budget` is present) would require patching mlx_vlm internals in a fragile
way and was explicitly ruled out as too hacky.

### Decided approach: split by quantisation

The two MoE variants serve different roles and get different trade-offs:

| Model | Quantisation | Strategy | Rationale |
|---|---|---|---|
| `qwen3.6-moe-mxfp4` | mxfp4 (4-bit) | `thinking_token_budget`, no MTP | 4-bit MoE router has low precision (16 values/weight); expert selection degenerates under load → thinking loops are common → hard cap needed |
| `qwen3.6-moe-8bit` | 8-bit | MTP (`--draft-model`), no budget | 8-bit router has 256 values/weight → far better expert diversity → loops are rare; MTP compensates for 8-bit's extra memory bandwidth cost, keeping t/s comparable to 4-bit |

**The MoE router precision argument in detail:** The MoE router is a small linear layer that
scores all experts and selects the top-K. At 4-bit, quantisation noise collapses score
differences between experts — the router loses fine-grained discrimination and repeatedly
activates the same subset. That feedback loop (same experts → similar activations → same
router decision) drives repetition and thinking loops. At 8-bit the discrimination recovers
and expert selection stays diverse across turns, breaking the loop structurally.

**Last-resort safety net for 8-bit:** with no thinking budget, a runaway thinking span on
the 8-bit model is bounded only by `max_tokens`. The current default is 16384. This is the
backstop — it will terminate any generation, including a stuck thinking loop, before the
context window fills. If loops appear in practice despite 8-bit quality, lower `max_tokens`
in the model defaults as the first mitigation before reconsidering MTP removal.

**Required router fix before enabling MTP:** in `mlx_server/router.py`, inside the
`if model_cfg.vlm:` block, add a check:

```python
if model_cfg.vlm:
    if "thinking_token_budget" in body:
        if model_cfg.mtp:          # new flag — see below
            body.pop("thinking_token_budget")
        else:
            body["thinking_budget"] = body.pop("thinking_token_budget")
    body.pop("chat_template_kwargs", None)
```

Add `mtp: bool = False` to `ModelConfig` in `mlx_server/config.py` and set `mtp: true` in
the yaml for `qwen3.6-moe-8bit`. The mxfp4 model does NOT get `mtp: true` — it keeps
`thinking_token_budget` in its defaults and that value is forwarded as `thinking_budget` to
mlx_vlm's native enforcement.

## Key files

| Path | Relevance |
|---|---|
| `~/.config/mlx/models.yaml` | Live config — edit here, mirror to repo |
| `mlx_server/config.py` | `ModelConfig.server_args()` passes `extra_args` to mlx_vlm |
| `mlx_server/mlx_vlm_wrapper.py` | Patches `strict=False` for MTP weight loading, then calls `mlx_vlm.server.cli.main` |
| `mlx_server/router.py` (vlm branch) | Renames `thinking_token_budget→thinking_budget`; must be updated for MTP conflict |
| `mlx_vlm/speculative/drafters/qwen3_5_mtp/split.py` | The split tool — entry point is `__main__` |
| `mlx_vlm/speculative/drafters/__init__.py` | `DRAFTER_KIND_BY_MODEL_TYPE` maps `qwen3_5_mtp → mtp` |
| `mlx_vlm/server/generation.py:930–950` | Draft model loading; `987` is the thinking_budget guard |
| `mlx_vlm/server/cli.py:207–213` | `--draft-model` / `--draft-kind` → env var wiring |

