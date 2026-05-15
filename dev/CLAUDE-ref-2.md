# CLAUDE-ref-2.md — File Conventions

> Extracted from CLAUDE.md to stay under the 320-line budget.
> Load only when you need to know which file owns what.

| File | Purpose |
|---|---|
| `ov_server.py` | FastAPI app wiring: middleware, router includes, embeddings endpoint, startup/shutdown |
| `app_state.py` | Shared mutable state: ServerStats, active_profile, ig_router, debug_logging |
| `chat_handler.py` | Full chat path: ChatRequest, VLM, OVH proxy, `/v1/chat/completions` |
| `admin_routes.py` | health, version, metrics, admin, catalogue, monitor endpoints; `_apply_profile` |
| `media_routes.py` | `/v1/images/generations`, `/v1/audio/transcriptions` |
| `server_config.py` | Config loading, model discovery, startup constants, resolved-model helpers |
| `model_manager.py` | Model lifecycle state, VRAM tracking, LRU eviction, AsyncTokenStreamer, assessor |
| `catalogue.py` | Model catalogue: local discovery + OVH remote fetch with TTL cache |
| `router.py` | Routing state + embedding centroid loader; all routing logic is in infergate |
| `prompt_builder.py` | Prompt building, tool-call parsing, streaming think-block handler |
| `config.json` | Runtime config: models_dir, device, model names, limits. Falls back to defaults if absent. |
| `dev/coding_standards_python.json` | Python typing and clean-code standards. 1st-order rules are inlined in CLAUDE.md; load this file only for 2nd-order techniques (TypedDict, TypeAlias, Protocol, TypeVar). |
| `dev/CONVENTIONS.md` | Machine-facing coding conventions for AI tools (Aider, Qwen, etc.). Module map, import rules, how-to recipes. Keep in sync with CLAUDE.md when modules change. |
| `README.md` | User-facing commands — **keep in sync** with any endpoint/startup/network changes |
| `MODELS.md` | Model conversion guide, directory layout, VRAM sizing, adding/removing models |
| `dev/PROGRESS.md` | Build progress — read NOW section only on re-entry |
| `dev/DECISIONS.md` | Append-only architectural decisions log |
| `dev/SCRATCHPAD.md` | In-session working memory |
| `dev/SESSION.md` | Crash-recovery snapshot — empty = clean close; non-empty = broken session |
| `dev/CLAUDE-ref.md` | Reference detail (Tool-Call Gap, Qwen format) — load only on explicit request |
| `dev/CLAUDE-ref-2.md` | This file — File Conventions table |
| `gpu_monitor.py` | GPU hardware metrics poller (temperature, power, VRAM via sysfs/XE driver); imported by `admin_routes.py`, started by `ov_server.py` |
| `monitor_sidecar.py` | Standalone HTTP server on `:11436` — GPU metrics + health proxy for SVP monitor; survives server restarts |
| `dev/Docker.md` | Docker run commands for Open WebUI + SearxNG containers |
| `autotest/YYYYMMdd_<hash>.md` | PND recovery artifacts — live test and debug session logs |
| `dev/plans/YYYYMMdd_PLAN_<subject>.md` | Development plans — all plans live here, inside the repo |
| `dev/plans/YYYYMMdd_<subject>.sql` | SQL attachments referenced by plans |
| `infergate/INFERGATE_USAGE.md` | infergate integration track — full story, adapter code, decisions, status |
| `infergate/ov_backend.py` | `OVServerBackend` — `Backend` Protocol impl (routing-only, reads ov_server globals) |
| `infergate/ov_embedding_provider.py` | `OVEmbeddingProvider` — `EmbeddingProvider` Protocol wrapping `emb_model` |
| `infergate/config.yaml` | infergate routing config (mirrors `config.json` task_classes, infergate field names) |
| `infergate/feedback/` (infergate repo) | Cross-session feedback loop — see INFERGATE_USAGE.md § Feedback loop |
