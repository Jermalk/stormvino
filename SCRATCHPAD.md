# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 12 (2026-05-06) summary

Session 12 fixed the 3-minute hang and reduced latency to ~40-55s first-response. Root fix (session 11): circular import in anthropic_layer._anthropic_to_messages caused module re-init during inference. This session: (1) stripped tool JSON schemas from CC requests (53K→26K tokens), (2) unified all claude-* CC routes to qwen3-14b-int4-ov eliminating haiku/sonnet eviction cycle, (3) enabled prefix caching (proven 39.5s→0.4s on warm requests), (4) capped max_new_tokens at 8192 for CC. CC is now functional: directory listing, file read, code explanation all work. Hard floor is ~40s first turn (26K token system prompt on 14B model). Prefix cache helps internal haiku calls (0.4s) but seems to miss after long generations (881-token response followed by full 47s prefill on next turn — likely KV block pressure).
