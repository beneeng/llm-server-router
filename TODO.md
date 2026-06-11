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
