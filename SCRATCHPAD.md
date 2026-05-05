# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 10 (2026-05-05) summary

Session 10 wired Claude Code integration end-to-end. Fixed auth conflict (claude.ai token + API key), patched `AnthropicThinking` to accept `{"type":"adaptive"}` with optional `budget_tokens`, changed "adaptive" → thinking disabled. Added `claude_code` config block with wildcard model_map (haiku→coder, sonnet→14b, opus→OVH) replacing all claude-* model_aliases. Added `_resolve_claude_code()` + `_maximize_context_for()` — the latter sets max KV for the selected model leaving `vram_reserve_pct` VRAM free (default 20%, gives 9GB KV for qwen3-14b). Fixed VRAM soft-cap check to include KV size + added retry-after-eviction for OpenVINO OOM. **Unresolved:** "hello" from Claude Code → model loads OK, GPU runs at 100W, but no tokens reach Claude Code. Root cause unknown — next session must debug the streaming path.

## Streaming hang — facts gathered

- Request `claude-sonnet-4-6` → `qwen3-14b-int4-ov` via claude_code mode
- `_maximize_context_for` evicts qwen3-8b, sets KV=9GB (after fix; was 12GB which also hung)
- `qwen3-14b` loads successfully, VRAM allocated shown in log
- GPU power: 100W sustained — scheduler running, possibly idle CB polling or actual generation
- No completion log, no error log — suggests generation started but tokens not arriving at consumer
- `/v1/chat/completions` path was NOT tested — unknown if it works for same model
- `_infer_lock` not held by any prior request (first request after restart)
- `AsyncTokenStreamer` uses `loop.call_soon_threadsafe` — could fail if loop reference is stale after `_maximize_context_for` context switch

## Next session — investigation plan

- Test: `curl -s http://localhost:11435/v1/chat/completions` with qwen3-14b — does the non-Anthropic path work?
- Test: `curl -s http://localhost:11435/v1/messages` directly — bypasses Claude Code, isolates server vs client issue
- Add `log.info("[anthropic stream] lock acquired, gen_task created")` before generation start
- Check: does `AsyncTokenStreamer` receive any tokens? Add log in `put()` method
- Check: is the issue specific to streaming (`stream=True`)? Try non-streaming via `/v1/messages`
- Check: does CB mode require a different generate() call signature vs non-CB?
