# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 2026-05-04 (Session 6) summary

Session 6 diagnosed and fixed AnythingLLM agentic mode. Root cause: qwen3-8b generates
prose instead of strict JSON (unlike original qwen2.5-3b). Three fixes: (1) agent streaming
now buffers internally and strips `<think>` before emitting; (2) `_extract_agent_json()`
scans prose for embedded JSON, returns `""` on miss for fast fallback; (3) `_record_stats()`
extracted from 7 inline duplicate blocks. Agent pipeline confirmed working: 30 tokens/3.2s
per tool-selection call, full web-search + 14b summarization chain running.

## Hashtag routing — implementation plan (deferred from Session 5)

- Server patch: top of `_pick_backend()` in ov_server.py — read `/tmp/ov_routing_override.json`;
  check `expires > time.time()`; get `backend` + `fallback` keys; return `_backends.get(name) or _backends.get(fallback) or _backends["local"]`; log at INFO; catch all exceptions silently.
- Hook script: `~/.claude/hooks/route-selector.sh` — reads stdin JSON, extracts `prompt`,
  detects `#use-local-box` / `#use-ovh` / `#use-uncle-a`, writes override file with TTL=300s, exits 0.
- Registration: `~/.claude/settings.json` → `hooks.UserPromptSubmit` → command pointing to hook script.
- Full code for both pieces is in `CLAUDE_CODE_INTEGRATION.md` §3a and §3b — implement from there verbatim.
- `#use-uncle-a` needs Step 10 (AnthropicBackend) to do anything beyond fallback to local.

## Misc facts

- Test venv: `/home/jerzy/ov_env` — `source /home/jerzy/ov_env/bin/activate && python -m pytest /opt/ov_server/tests/ -q`
- Debug logging currently ON — disable with `kill -USR1 $(systemctl show ov-server --property=MainPID --value)`
