# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 2026-05-05 (Session 7) summary

Session 7 fixed three layers of VRAM management bugs and added model preloading. (1) `_evict_lru()` and VLM inline eviction now call `gc.collect()` so the C++ destructor runs before the next VRAM query. (2) Soft VRAM cap changed from `if` to `while` with re-query after each eviction. (3) `vram_free_gb()` was fundamentally broken — a fresh `ov.Core()` always shows zero usage; replaced with internal `_vram_allocated` tracking + `_TOTAL_VRAM_GB` queried once at startup. (4) `kv_cache_size_gb` reduced 8→3 in config.json (two loaded models were consuming 29.6 GB against 22.71 GB total). (5) Startup preload of qwen3-8b via `@app.on_event("startup")`; speculative preload of qwen3-14b fired when agent returns tool_calls. VRAM state now visible in `/health` response. Tests: 32/32.

## Session 8 summary (2026-05-05)

Profile switching and ov_monitor control panel shipped and user-confirmed working, including graceful wait during active inference. `POST /admin/profile` → 202, `_apply_profile()` drains in-flight requests then evicts/reloads. `/health` gained `active_profile`, `profile_switching`, `kv_cache_size_gb`. ov_monitor gained Profiles panel (1/2/3 keypresses via KeypressThread setcbreak) and segmented VRAM bar (█=weights, ▓=KV, ░=free, colored per model). Both repos committed.

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
