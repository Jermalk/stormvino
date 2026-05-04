# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 2026-05-04 (Session 4) summary + deferred plan

Session 4 completed Step 15 (RequestIDMiddleware + _RequestIDFilter + ContextVar log
correlation) and wrote CLAUDE_CODE_INTEGRATION.md. The hashtag routing feature was
designed but not yet implemented — plan is saved below for next session.

## Hashtag routing — implementation plan (deferred)

- Server patch: top of `_pick_backend()` in ov_server.py — read `/tmp/ov_routing_override.json`;
  check `expires > time.time()`; get `backend` + `fallback` keys; return `_backends.get(name) or _backends.get(fallback) or _backends["local"]`; log at INFO; catch all exceptions silently.
- Hook script: `~/.claude/hooks/route-selector.sh` — reads stdin JSON, extracts `prompt`,
  detects `#use-local-box` / `#use-ovh` / `#use-uncle-a`, writes override file with TTL=300s, exits 0.
- Registration: `~/.claude/settings.json` → `hooks.UserPromptSubmit` → command pointing to hook script.
- Full code for both pieces is in `CLAUDE_CODE_INTEGRATION.md` §3a and §3b — implement from there verbatim.
- `#use-uncle-a` needs Step 10 (AnthropicBackend) to do anything beyond fallback to local.

## Misc facts

- Test venv: `/home/jerzy/ov_env` — `source /home/jerzy/ov_env/bin/activate && python -m pytest /opt/ov_server/tests/ -q`
