# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Session 2026-05-04 (Session 3) summary

Session completed Phases 2–4 in full. Phase 2: Backend ABC + LocalBackend (Step 6), _pick_backend() config router with routing.model_map (Step 7), /health router key (Step 8). Phase 3: OVH API key stored in .env, OpenAICompatBackend with extract_thinking + parse_tool_calls (Step 9), KV cache U8 precision + DYNAMIC_QUANTIZATION_GROUP_SIZE in CONFIG (Step 11), get_scheduler_config() + LLMPipeline kwarg (Step 12). Phase 4: optional bearer auth via verify_token() on /v1/messages routes only (Step 13), CORSMiddleware allow all origins (Step 14). Server restarted and verified: 36.1 tok/s on qwen3-14b-int4-ov, OVH backend registered, CORS header confirmed live. Only Step 15 (request ID logging) and Step 10 (AnthropicBackend, needs API key) remain.
