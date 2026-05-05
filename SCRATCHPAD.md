# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 2026-05-05 (Session 9) summary

Session 9 fixed and polished profile switching. (1) `_apply_profile()` now evicts VLMs always but evicts LLMs only when `kv_cache_size_gb` changes — avoids unnecessary reload on speed→ovh switch. (2) ov_monitor server panel gained VLM loaded row. (3) Segmented VRAM bar fixed for VLM models (no KV segment subtracted; legend shows "X.XGB VLM"). (4) `kv_cache_size_gb` added to `/health` and shown in monitor KV cache row. Both repos committed, 32/32 tests pass.

## Hashtag routing — implementation plan (carried from Session 5, still pending)

- Server patch: top of `_pick_backend()` in ov_server.py — read `/tmp/ov_routing_override.json`;
  check `expires > time.time()`; get `backend` + `fallback` keys; return `_backends.get(name) or _backends.get(fallback) or _backends["local"]`; log at INFO; catch all exceptions silently.
- Hook script: `~/.claude/hooks/route-selector.sh` — reads stdin JSON, extracts `prompt`,
  detects `#use-local-box` / `#use-ovh` / `#use-uncle-a`, writes override file with TTL=300s, exits 0.
- Registration: `~/.claude/settings.json` → `hooks.UserPromptSubmit` → command pointing to hook script.
- Full code for both pieces is in `CLAUDE_CODE_INTEGRATION.md` §3a and §3b — implement from there verbatim.
- `#use-uncle-a` needs Step 10 (AnthropicBackend) to do anything beyond fallback to local.

## Misc facts

- Test venv: `/home/jerzy/ov_env` — `source /home/jerzy/ov_env/bin/activate && python -m pytest /opt/ov_server/tests/ -q`
- Debug logging still ON — disable with `kill -USR1 $(systemctl show ov-server --property=MainPID --value)`
- on_event("startup") deprecation warning is harmless — FastAPI still honours it
