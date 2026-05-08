# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 24 wrapped cleanly. Embedder on GPU.0 (vision_device/embedding_device config keys). Assessor disabled — qwen3-8b can't coexist with 14b+VLM on B60 (18.7/22.7GB). VLM fixed: AUTO device breaks Qwen2.5-VL shape computation; vision_device=GPU.1 required. VLM streaming hang fixed (queue.put_nowait on exception). Web search working end-to-end via SearxNG (had to enable json format in settings.yml). Open WebUI web search uses RAG injection, not tool_calls.
