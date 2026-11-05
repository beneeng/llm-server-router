# TODO / Upstream workarounds

Items that should be revisited once a dependency fix lands.

---

## Remove `_prerender_gemma4_tool_history` once vllm-mlx fixes MLLM tool history

**File:** `mlx_server/router.py` — function `_prerender_gemma4_tool_history` and its call-site in `proxy()`

**Condition to revert:** vllm-mlx releases a version in which `MLLM.stream_chat` (and `MLLM.chat`) correctly preserve `tool_calls` on assistant messages and include them in the `chat_messages` list passed to `get_chat_template`. The bug is in the message-building loop of both methods — the gate `if msg_text or msg_image_count or msg_audio_count` silently drops assistant messages where `content=null` (i.e. `msg_text == ""`), even when `tool_calls` is present. Even when `msg_text` is non-empty, `tool_calls` is never carried through to the built `chat_messages` entry.

**What the workaround does:** For every `multimodal: true` model, the router rewrites conversation history before forwarding: each `(assistant with tool_calls, role:tool)` pair is collapsed into a single assistant message whose `content` field contains the pre-rendered Gemma4 special-token sequence:
```
<|tool_call>call:name{args}<tool_call|><|tool_response>response:name{value:<|"|>result<|"|>}<tool_response|>
```
All of `<|tool_call>`, `<tool_call|>`, `<|tool_response>`, `<tool_response|>`, and `<|"|>` are single vocabulary tokens in the Gemma4 tokenizer, so they are encoded correctly when embedded in a text content field.

**How to revert:**
1. Verify the vllm-mlx fix: run the two-step curl test below and confirm `prompt_tokens` increases after step 2 and the model gives a different response referencing the tool result.
2. Delete `_prerender_gemma4_tool_history` from `router.py`.
3. Replace the call-site in `proxy()` with the old content-nulling fix (or nothing, if the upstream fix also handles the duplicate-content case from Pi harness):
   ```python
   for msg in body["messages"]:
       if (msg.get("role") == "assistant"
               and msg.get("tool_calls")
               and msg.get("content") is not None):
           msg["content"] = None
   ```

**Two-step smoke test:**
```bash
# Step 1 — trigger tool call
curl -s -X POST http://localhost:6300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-12b","stream":false,"max_tokens":256,
       "messages":[{"role":"user","content":"Search the web for murals in Pforzheim"}],
       "tools":[{"type":"function","function":{"name":"web_search","description":"Search","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]}'

# Capture the tool_call id from step 1, then:
# Step 2 — send back a fake result; model must answer differently (not re-issue same call)
curl -s -X POST http://localhost:6300/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-12b","stream":false,"max_tokens":256,
       "messages":[
         {"role":"user","content":"Search the web for murals in Pforzheim"},
         {"role":"assistant","content":null,"tool_calls":[{"id":"CALL_ID","type":"function","function":{"name":"web_search","arguments":"{\"query\":\"murals in Pforzheim Germany\"}"}}]},
         {"role":"tool","tool_call_id":"CALL_ID","content":"Found: murals at Karl-Friedrich-Strasse, annual street art festival."}
       ],
       "tools":[{"type":"function","function":{"name":"web_search","description":"Search","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]}'
```
Expected after the fix: step 2 `prompt_tokens` > step 1 `prompt_tokens`, and the model either asks a follow-up question or synthesises an answer referencing the result text.

---

## Flash-Next (`qwen3.8-flash-next-ple`): speed levers still open

**Where we are (measured 2026-11-05, mlx-vlm 0.7.12, mlx 0.32.4, APC off):** decode ~54 tok/s on short prompts, ~47 at 8k context, ~44 at 24k, peak 82.9 GB. No drafter on purpose: the MTP drafter (`sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16`) is +23% on short prompts but slower from ~8k up (41 vs 47 at 8k, 34 vs 44 at 24k), and agent sessions live far above 8k. The reason is the generic batch-invariant verify path in stock mlx-vlm. Details are in the comment on the model entry in `models.yaml`.

**Re-measure with:**
```bash
scripts/bench_draft_variants.py --model qwen3.8-flash-next-ple --long-context 8000 --long-context 24000 --repeat 2 \
    --variant "baseline:" --variant "mtp:--draft-model sh0wie/Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16 --draft-kind mtp"
```
Close other apps and run `mlx stop` first; the script refuses to start when free RAM or swap is short. Use `--python <venv>/bin/python` to test another mlx-vlm version in an isolated venv. Judge by the long-context rows, not the short-prompt mean.

**1. Re-test MTP after each mlx-vlm bump.** Cheap: one benchmark run. Enable `--draft-model ... --draft-kind mtp` + `mtp: true` only if it wins at 8k and 24k, not just on short prompts. Also check whether `thinking_budget` still gets rejected with a drafter active (0.7.12 added "speculative thinking budgets", unverified here), since the `mtp: true` strip in `MlxVlm.transform_request` may then be unnecessary.

**2. unsloth-zoo PR #1871 fusions** (https://github.com/unslothai/unsloth-zoo/pull/1871, merged 2026-10-19). Exact fused kernels for qwen4_exp decode and verify; the author reports verify of a 4-token block going from 1541 ms to 108 ms (M3 Max). They are context managers in `unsloth_zoo.mlx.inference` (`exact_batch_invariant_decode`), not applied by mlx-vlm itself. Using them means wrapping generation in `mlx_vlm_wrapper.py`. Blockers: `unsloth-zoo` pins `mlx-vlm<=0.7.11` and `mlx==0.32.4` (we are on 0.7.12), it depends on internal hooks that fail closed on drift, and it was only benchmarked on M3 Max. Check first whether a newer `unsloth-zoo` lifts the pin; then prototype in an isolated venv before touching the wrapper.

**3. MTPLX** (https://github.com/youssofal/MTPLX, Apache-2.0). Separate MLX runtime with its own OpenAI-compatible server and its own checkpoints (`Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed`: 115 GB download, ~83 GB resident, n-gram table streamed from SSD). Vendor numbers on an M5 Max: 79 tok/s on a 9k prompt, 62 tok/s at 109k, 50 tok/s at 200k. Those are the author's own measurements (best case: warm prompt cache, fans pinned), so verify on this box before believing them. Integration would be one new `Backend` subclass in `backends.py` plus a wrapper; the 115 GB download needs disk and a free-RAM check first.

**4. (low priority) REAP-288** (`sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit`): experts pruned 512 -> 288, 68 GB, HumanEval -2.4 points, vision untested. Only worth it if RAM headroom for a second model matters more than quality.

**Ruled out:** Unsloth `UD-Q4_K_XL` GGUF. 111 GB (larger than the MLX 4-bit, so not a stronger quantization), llama.cpp is roughly half the speed of MLX on this Mac, and its MTP build lists no Metal prebuilt.
