# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 29: ModelFamilyAdapter Protocol (DefaultAdapter, MistralAdapter, InternVLAdapter) in prompt_builder.py — max_context_tokens, sampling_defaults, validate_messages(), build_prompt(), parse_tool_calls(). ChatRequest.temperature=None (adapter defaults). _chat_vlm updated for multi-VLM routing. test_internvl.py (7 tests). InternVL2.5-26B downloading (PID 345796, 3.8/52GB). Dynamic KV cache sizing in server_config.py: compute_kv_cache_gb() reads config.json architecture params + tokenizer_config.json for family → Qwen3-14B 7GB, Qwen2.5-VL 3GB, InternVL will get 8192-token budget.
