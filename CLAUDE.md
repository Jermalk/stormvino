# CLAUDE.md — ov_server (OpenVINO OpenAI-Compatible API Server)

> **CLAUDE.md file budget:** This file must stay under 320 lines (hard cap).
> At 290+ lines, extract the largest section to `CLAUDE-ref-N.md` and replace with a one-line pointer.
> Never load a `CLAUDE-ref` file unless the user explicitly asks about its topic.

---

## Re-entry protocol — read this first, every session

**Bootstrap guard — check before anything else:**
- `PROGRESS.md` missing → create with empty NOW section, continue.
- `SCRATCHPAD.md` missing → create it empty, continue.
- `SESSION.md` missing → create empty, continue.
- `SESSION.md` **non-empty** → **BROKEN SESSION DETECTED.** Read it aloud to user, ask "Continue from this state?" before proceeding.

**Normal re-entry — in this order:**
1. Read `PROGRESS.md` — **NOW section only** (skip history).
2. Read `SCRATCHPAD.md` — summarise in one paragraph. Write back as `## Carried over:` (first entry), then clear the rest.
3. Read only files named in PROGRESS.md "Next action". If "Next action" is empty or absent → stop and ask.
4. Stop. Do not open other files speculatively.

If task is clear from steps 1–3, start coding. If not, ask — do not explore to resolve ambiguity.

---

## Framework rules

| ID | Rule | One-line trigger |
|---|---|---|
| KYE | Know Your Enemy | Read the terrain before forming a hypothesis |
| SBS | Step By Step | Each step explicit, verified, proven before the next |
| AEC | Always Embrace Change | Evaluate rule spirit vs letter — break consciously when cost > benefit |
| OMK | Overconfidence May Kill | Step back mid-implementation — what else could this break? |
| YNC | You're Not Chrome | Surface irreversible actions; Jerzy decides, Claude executes |
| PND | Post-Nuke Discovery | Create a log file first; write each finding as it's made |

**KYE — Know Your Enemy** *(Sun Tzu)*
The "enemy" is the problem, the codebase, the constraint, or the bug. Understand it before fighting it. Never hypothesise before reconnaissance. Firing condition: before writing any code, read the relevant files, logs, and constraints first. A wrong mental model costs more than the time spent reading.

**SBS — Step By Step** *(with proof in hand)*
Small steps alone are not enough — each step must be verified before the next begins. Write the test, run it, see it green. Run curl, read the response. State what you expect, then confirm it. Rushing past verification is where bugs hide for days. The proof is not optional — it is the step.

**AEC — Always Embrace Change**
No rule foresees every situation. When a rule costs more than it saves, evaluate the spirit of the rule, make the judgment explicit, and decide consciously. Example: file is 3 lines over the length limit — refactoring is waste; act when it reaches 10% over. Never break a rule silently — state that you are doing it and why. This rule authorises judgment, not carelessness.

**OMK — Overconfidence May Kill**
Tunnel vision on a target stops you seeing the board. The chess beginner loses not because they played badly but because they stopped watching what the opponent was doing. After any non-trivial change: run the full test suite, check `/health`, ask *"what else could this break?"* Especially dangerous during refactoring and wiring steps where side-effects are invisible until production.

**YNC — You're Not Chrome**
Claude is a powerful assistant but not the decision-maker. Responsibility stays with Jerzy. Propose architecture and approaches; surface irreversible actions before taking them; never unilaterally decide on design tradeoffs. If uncertain whether an action is reversible, ask. This is not timidity — it is correct role definition.

**PND — Post-Nuke Discovery**
The session is not the unit of work — the log is. During any multi-step discovery, debugging, or live-testing session, create a dedicated log file *before* starting and write each significant finding to it immediately after it's confirmed. This makes the log the recovery artifact: if context compacts or the session is interrupted, the next session reads the log first and resumes without re-running discovery from scratch. Firing condition: any investigation expected to span >5 steps or >15 minutes. Format: `~/autotest/YYYYMMdd_<commitHash>.md` for live tests; free-form file in `/tmp/` for ad-hoc debugging. The log must be self-contained — a cold reader with no session history must be able to resume from it.

---

## Domain

FastAPI server exposing an OpenAI-compatible REST API (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`) backed by `openvino_genai.LLMPipeline`. Runs on Linux Mint, Intel GPU (`GPU.1`), accessed from the local network on port `11435`.

---

## Architecture at a Glance

| Layer | Component | Notes |
|---|---|---|
| HTTP server | FastAPI + Uvicorn | Single-worker, `asyncio` loop |
| LLM inference | `openvino_genai.LLMPipeline` | Blocking; offloaded to executor |
| Chat path | `chat_handler.py` | ChatRequest, VLM path, OVH proxy, `/v1/chat/completions` |
| Prompt building | `prompt_builder.py` | `build_prompt()`, `build_vlm_prompt()`, tool-call parser, streaming handler |
| Streaming | `AsyncTokenStreamer` (model_manager) | Subclass of `ov_genai.StreamerBase`; event loop captured at construction |
| Embeddings | `OVModelForFeatureExtraction` (optimum-intel) | Mean-pooled, L2-normalised; lives in `model_manager.py` |
| Routing | `router.py` | Signal detection → embedding similarity → model selection |
| Model catalogue | `catalogue.py` | Local discovery + OVH remote fetch with TTL cache |
| Model lifecycle | `model_manager.py` | Loaded models, VRAM tracking, LRU eviction, assessor |
| Config/discovery | `server_config.py` | Config loading, model discovery, startup constants |
| Shared state | `app_state.py` | ServerStats, active_profile, ig_router, debug_logging — leaf module, no cycles |
| Admin/ops | `admin_routes.py` | health, version, metrics, admin, catalogue, monitor endpoints; `_apply_profile` |
| Media | `media_routes.py` | `/v1/images/generations`, `/v1/audio/transcriptions` |
| Models | `qwen3-8b-int4-ov`, `qwen3-14b-int4-ov`, `qwen2.5-vl-7b-int4-ov` | Up to 2 LLMs loaded; LRU eviction with VRAM check |

**Entry point:** `/opt/ov_server/ov_server.py`
**Config file:** `/opt/ov_server/config.json`
**Models dir:** `~/ov_models/` (configured in `config.json`, auto-discovered at startup)
**Cache dir:** `/tmp/ov_cache_b60`
**Device:** `GPU.1` with `PERFORMANCE_HINT=LATENCY`

---

## Tool-Call Gap

→ Full gap analysis, implementation order, and Qwen format spec in `CLAUDE-ref.md § Tool-Call Gap`.

**Summary:** `openvino_genai` does not handle OpenAI tool-call semantics. `ChatRequest` and `Message` have no `tools`/`tool_calls` fields, `build_chatml()` does not inject tool schemas, and there is no `<tool_call>` parser or `finish_reason: "tool_calls"` emission.

---

## Known Bugs / Sharp Edges

*(All previously listed bugs fixed in b647cfb — section retained for future entries.)*

---

## Context load discipline

| Situation | Load | Do not load |
|---|---|---|
| Session start | `PROGRESS.md` NOW, `SCRATCHPAD.md` | Everything else until needed |

Never load speculatively. Test files only when writing or fixing that test.

**Two separate limits — do not confuse them:**

| Limit | Threshold | What to do |
|---|---|---|
| CLAUDE.md file budget | 290 lines (hard cap 320) | Extract largest section to `CLAUDE-ref-N.md` |
| Context load budget | 800 lines of actively-loaded files | Flush to SCRATCHPAD, finish atomic unit, commit, recommend new session |

*Context load* counts only files explicitly Read or written this session — not tool output, not PROGRESS.md already closed.

---

## PROGRESS.md — NOW section format

```
## NOW

**Working on:** <one sentence>
**Last commit:** <hash> — <message>
**Next action:** <specific file and function name>
**Blocked on:** <decision needed, or "nothing">
**Open questions:** <brief list, or "none">
**Tests:** <"pass" | "fail — N failing" | "not run">
```

File has two parts: history (append-only, skip on re-entry) and NOW (overwritten each session, always last). **During session-wrap: copy current NOW into History first, then overwrite NOW.** Never skip — it is the only audit trail.

---

## DECISIONS.md — entry format

```
### YYYY-MM-DD — <topic>
**Decision:** <one sentence>
**Rationale:** <one to three sentences>
**Rejected alternative:** <one sentence, or "none considered">
**Affects:** <file or component name>
```

**Write immediately** when an architectural decision is made during a session — do not defer to session-wrap. One entry per decision, appended in real time.

Read `DECISIONS.md` only when the user explicitly asks about a past decision.

---

## SCRATCHPAD.md discipline

In-session working memory. Write to it when:
- You have analysed a file — write extracted facts, not the filename.
- You are mid-way through a multi-step change and context is filling.
- You have made a decision not yet in `DECISIONS.md`.

Format: bullet points, max 5 lines per topic, no prose. Cleared at start of every session (carry-over paragraph replaces it).

---

## SESSION.md — broken-session recovery

Live crash snapshot. Overwritten on every commit during a session; cleared (emptied to zero bytes) by `#session-wrap`.

**Format:**
```
## BROKEN SESSION — <YYYY-MM-DD HH:MM>

**Last commit:** <hash> — <message>
**Mid-step:** <what was in progress — one sentence>
**Next action:** <exact file + function>
**Tests:** <pass N/N | fail — N failing | not run>
**Notes:** <anything else needed to resume cleanly>
```

On re-entry: if non-empty, read aloud and ask user before proceeding (bootstrap guard handles this).

---

## `#session-wrap`

1. Run tests if available.
2. Copy current NOW block verbatim into `PROGRESS.md` History (append as `### YYYY-MM-DD — Session N (<hash>)`), then overwrite NOW with updated fields including **Tests** result.
3. Append to `DECISIONS.md` — one entry per architectural decision made this session.
4. Clear `SCRATCHPAD.md`, write one-paragraph session summary.
5. Clear `SESSION.md` (write empty file — signals clean close).
6. Commit: `docs: session wrap — <summary>`.
7. Report: committed files, what NOW says, what next session opens first.

---

## Hard Rules

- **Never use `sudo pip install`.** All dependencies live in the venv at `/home/jerzy/ov_env`.
- **PEP8 compliant Python.** Use `black` if available.
- **Type hints on all function signatures.**
- **No bare `except:`.** Catch specific exceptions.
- **OpenVINO device check before any inference change:**
  ```python
  import openvino as ov
  core = ov.Core()
  assert "GPU.1" in core.available_devices, "GPU.1 not available"
  ```
- **Do not break `/health`, `/v1/models`, `/v1/embeddings`** when modifying the chat path.
- **Test both streaming and non-streaming** after any change to `chat()`.

---

## Diagnostic Protocol

1. **Snapshot** — `hostnamectl`, `python3 --version`, check venv active, `lscpu | grep "Model name"`.
2. **Logs** — `journalctl -u ov-server` or process stdout. Read the traceback before hypothesising.
3. **Hypothesis** — State explicitly what is wrong and why.
4. **Targeted fix** — Minimal change that resolves the root cause.
5. **Verification** — `curl -s http://localhost:11435/health | python3 -m json.tool`; follow with a minimal chat request.

---

## Python Code Standards

- `pathlib.Path` over `os.path`.
- Environment variables via `os.environ.get()` with defaults — never hardcoded paths except `MODELS_DIR` (already parameterised).
- Async blocking work via `loop.run_in_executor(None, ...)` — never `await` a CPU-bound call directly.
- Use `asyncio.get_running_loop()` — not deprecated `get_event_loop()`.
- **Typing — 1st order (always apply):**
  - `Literal` for categorical sentinels: `finish_reason`, device names, profile names, `PERFORMANCE_HINT` values.
  - Modern generics: `X | None` not `Optional[X]`; `list[str]` not `List[str]`; remove all legacy `typing` imports.
  - `# type: ignore` at `openvino_genai` boundaries — no published stubs; annotate and move on.
  - Domain-specific names: `stream_chunk`, `token_count`, `raw_payload` — not `data`, `output`, `result`.
- **Typing — 2nd order:** See `coding_standards_python.json` (TypedDict, TypeAlias, Protocol, TypeVar). Apply only when the stated `apply_when` condition is met — not by default.
- **Conventions for AI coding tools:** See `CONVENTIONS.md` — update it whenever a module is added or ownership changes.

---

## File Conventions

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
| `router.py` | Routing logic: signal detection, embedding similarity, model selection |
| `prompt_builder.py` | Prompt building, tool-call parsing, streaming think-block handler |
| `config.json` | Runtime config: models_dir, device, model names, limits. Falls back to defaults if absent. |
| `coding_standards_python.json` | Python typing and clean-code standards. 1st-order rules are inlined above; load this file only for 2nd-order techniques (TypedDict, TypeAlias, Protocol, TypeVar). |
| `CONVENTIONS.md` | Machine-facing coding conventions for AI tools (Aider, Qwen, etc.). Module map, import rules, how-to recipes. Keep in sync with CLAUDE.md when modules change. |
| `.aider.conf.yml` | Aider persistent config: default model (Qwen3-30b via ov_server), conventions context, architect-mode comments. |
| `README.md` | User-facing commands — **keep in sync** with any endpoint/startup/network changes |
| `MODELS.md` | Model conversion guide, directory layout, VRAM sizing, adding/removing models |
| `PROGRESS.md` | Build progress — read NOW section only on re-entry |
| `DECISIONS.md` | Append-only architectural decisions log |
| `SCRATCHPAD.md` | In-session working memory |
| `SESSION.md` | Crash-recovery snapshot — empty = clean close; non-empty = broken session |
| `CLAUDE-ref.md` | Reference detail (Tool-Call Gap, Qwen format) — load only on explicit request |
| `gpu_monitor.py` | GPU hardware metrics poller (temperature, power, VRAM via sysfs/XE driver); imported by `admin_routes.py`, started by `ov_server.py` |
| `monitor_sidecar.py` | Standalone HTTP server on `:11436` — GPU metrics + health proxy for SVP monitor; survives server restarts |
| `Docker.md` | Docker run commands for Open WebUI + SearxNG containers |
| `autotest/YYYYMMdd_<hash>.md` | PND recovery artifacts — live test and debug session logs |
| `plans/YYYYMMdd_PLAN_<subject>.md` | Development plans — all plans live here, inside the repo |
| `plans/YYYYMMdd_<subject>.sql` | SQL attachments referenced by plans |
| `infergate/INFERGATE_USAGE.md` | infergate integration track — full story, adapter code, decisions, status |
| `infergate/ov_backend.py` | `OVServerBackend` — `Backend` Protocol impl (routing-only, reads ov_server globals) |
| `infergate/ov_embedding_provider.py` | `OVEmbeddingProvider` — `EmbeddingProvider` Protocol wrapping `emb_model` |
| `infergate/config.yaml` | infergate routing config (mirrors `config.json` task_classes, infergate field names) |
| `infergate/feedback/` (infergate repo) | Cross-session feedback loop — see INFERGATE_USAGE.md § Feedback loop |

---

## Tone & Output Style

- Technical and precise. No filler phrases.
- When uncertain, say so explicitly.
- Provide commands ready to copy-paste with no unresolved placeholders.

---

## Dev notes

- **EnvyStorm** is the dev machine: OpenVINO + Arc B60. Local inference, zero cloud.
