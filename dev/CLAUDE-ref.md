# CLAUDE-ref.md — ov_server reference detail

> Load only when the user explicitly asks about a topic in this file.
> Not read on session re-entry.

---

## § Tool-Call Gap

`openvino_genai` runs raw text generation — it does **not** handle OpenAI tool-call semantics. Everything must be wired manually in `ov_server.py`.

### What is missing

1. **`ChatRequest` has no `tools` / `tool_choice` fields.**
   The Pydantic model only accepts `messages`, `model`, `max_tokens`, `temperature`, `stream`, `thinking`.

2. **`Message` has no `tool_calls` or `tool_call_id` fields.**
   Tool result turns (`role: "tool"`) and assistant turns with tool invocations (`role: "assistant", tool_calls: [...]`) cannot be represented.

3. **`build_chatml()` does not inject tool schemas.**
   Tool definitions must be serialised into the system/user prompt (Qwen supports a specific JSON schema format) before the model can know what tools exist.

4. **No tool-call output parser.**
   Qwen models emit tool calls as a JSON block inside `<tool_call>…</tool_call>` tags. `ov_server.py` has no code to detect or extract these; the raw text is returned verbatim.

5. **No `finish_reason: "tool_calls"` in responses.**
   Callers (AnythingLLM, LangChain, etc.) rely on this field to know whether to parse `tool_calls` or treat the response as a final answer.

6. **Streaming does not accumulate tool call fragments.**
   A tool-call JSON block may arrive across multiple tokens; streaming must buffer and detect the complete block before emitting it as a `tool_calls` delta.

### Implementation order

1. Extend `Message` to accept `tool_calls: Optional[List[ToolCall]]` and `tool_call_id: Optional[str]`.
2. Extend `ChatRequest` to accept `tools: Optional[List[Tool]]` and `tool_choice`.
3. Add `format_tools_for_chatml(tools)` — serialise tool schemas into the Qwen tool-call system prompt block.
4. Extend `build_chatml()` to handle `role: "tool"` turns (tool results) and assistant turns that contain `tool_calls`.
5. Add `extract_tool_calls(raw_text) -> (tool_calls, answer)` — detect `<tool_call>` blocks, parse JSON, return structured list.
6. Non-streaming path: populate `message.tool_calls` and set `finish_reason: "tool_calls"` when tool calls are detected.
7. Streaming path: buffer tokens until `<tool_call>` blocks are complete, then emit as `tool_calls` deltas.

### Qwen tool call format

**System prompt injection:**
```
# Tools

You may call one or more functions to assist with the user query.

<tools>
[{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]
</tools>
```

**Model output when calling a tool:**
```
<tool_call>
{"name": "function_name", "arguments": {"param": "value"}}
</tool_call>
```

**Tool result turn (injected into next prompt):**
```
<|im_start|>tool
<tool_response>
{"result": "..."}
</tool_response>
<|im_end|>
```
