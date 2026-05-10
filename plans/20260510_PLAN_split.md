# PLAN — Split ov_server.py into modules

**Date:** 2026-05-10
**Status:** Draft — not started
**Goal:** Reduce ov_server.py from 2090 lines to ~500 by extracting four coherent modules.
**Driver:** Readability for human operator; enable clean application of Python typing standards.

---

## Why split

The single-file rule in CLAUDE.md says "keep it unless a module exceeds ~200 lines of distinct concern."
The file is already 2090 lines — 10× that threshold. Three distinct concerns are already separate
(`prompt_builder.py`, `db.py`). Four more are clearly visible and can be extracted without redesign.

Applying coding standards (Literal, modern generics, domain naming) adds perhaps 30 lines.
That is not the bloat problem. The problem is navigability: finding the model-loader
means scrolling past ~600 lines of unrelated catalogue and routing code.

---

## The shared mutable state problem

This is the key design challenge. Python module-level names are local bindings —
doing `from server_config import DEFAULT_MODEL` and then reassigning `DEFAULT_MODEL`
in `_apply_profile()` (inside ov_server.py) does NOT update the binding in other modules.

**Three names are mutated at runtime** by `_apply_profile()`:
- `DEFAULT_MODEL` — fallback model selection
- `AGENT_MODEL` — warm-on-startup model
- `MAX_LOADED_MODELS` — hard cap for LRU eviction

**Solution (Step 0):** Store these in `_cfg` (already a module-level dict imported by reference).
Because Python dicts are passed by reference and module imports are cached singletons,
all modules that `import _cfg from server_config` share the same dict object.
Mutations to `_cfg["max_loaded_models"]` are immediately visible everywhere.

Replace module-level constants with dict reads:
```python
# Before
MAX_LOADED_MODELS = _cfg["max_loaded_models"]   # stale copy after profile switch

# After (read live every time)
_cfg["max_loaded_models"]                        # always current
```

Expose two helpers in `server_config.py`:
```python
def get_default_model() -> str:
    return _cfg.get("_resolved_default_model", "")

def get_agent_model() -> str:
    return _cfg.get("_resolved_agent_model", "")
```

`_apply_profile()` updates `_cfg["_resolved_default_model"]` and `_cfg["_resolved_agent_model"]`
instead of reassigning module globals. All modules call `get_default_model()` — no stale copies.

---

## Target module layout

```
server_config.py    ~180 lines   — config, discovery, no ov_server imports
model_manager.py    ~500 lines   — model lifecycle, VRAM, AsyncTokenStreamer
catalogue.py        ~130 lines   — model catalogue, remote fetch
router.py           ~380 lines   — signal detection, embedding routing, model selection
ov_server.py        ~500 lines   — FastAPI app, endpoints, middleware, profile switching
prompt_builder.py   +10 lines    — add _has_images() (needed by router.py)
```

Total: ~1700 lines across 6 files vs 2090 in one. Same code, better navigation.

---

## Import dependency graph (no cycles)

```
db.py              ← standard libs only
server_config.py   ← json, pathlib, subprocess, logging
prompt_builder.py  ← standard libs + pydantic
model_manager.py   ← server_config, db
catalogue.py       ← server_config, model_manager, httpx
router.py          ← server_config, model_manager, db, prompt_builder
ov_server.py       ← server_config, model_manager, catalogue, router, prompt_builder, db
```

No cycles. Each arrow is a one-way dependency.

---

## Module responsibilities

### `server_config.py`
What moves here from ov_server.py:
- `_load_config()`, `_validate_config()`, `_KNOWN_CONFIG_KEYS`
- `_read_git_commit()`
- `_discover_models()`, `_discover_vlm_models()`
- `_pick()`, `_model_kv_gb()`, `get_scheduler_config()`
- All startup constants derived from `_cfg`:
  `MODELS_DIR`, `DEVICE`, `CONFIG`, `AVAILABLE_MODELS`, `AVAILABLE_VLM_MODELS`,
  `SERVER_VERSION`, `_GIT_COMMIT`, `MODEL_ALIASES`, `EMBEDDING_MODEL_ID`,
  `EMBEDDING_MODEL_PATH`, `VISION_MODEL`, `VLM_MAX_IMAGE_TURNS`, `VLM_MAX_IMAGE_SIDE_PX`
- New helpers: `get_default_model()`, `get_agent_model()`

What does NOT move: nothing from the server — this module has zero ov_server imports.

### `model_manager.py`
What moves here from ov_server.py:
- All shared state: `loaded_models`, `loaded_tokenizers`, `model_last_used`,
  `loaded_vlm_models`, `loaded_vlm_tokenizers`, `_model_lock`, `_vlm_lock`,
  `_infer_locks`, `_vlm_infer_locks`, `_TOTAL_VRAM_GB`, `_vram_allocated`,
  `emb_model`, `emb_tokenizer`, `_emb_lock`
- `AsyncTokenStreamer` class
- `check_memory()`
- `model_size_gb()`, `vram_free_gb()`, `_init_vram()`, `_evict_lru()`
- `get_model()`, `get_embedding_model()`, `get_vlm()`
- `_warm_model()`, `_warm_vlm()`
- `_assessor_pipe`, `_assessor_tokenizer`, `_assessor_lock`, `_load_assessor()`
- New helpers for _apply_profile(): `evict_all_models()`, `evict_all_vlms()`

### `catalogue.py`
What moves here from ov_server.py:
- `_catalogue_cache`, `_AUTO_ENTRY`
- `_tier_map_for_provider()`, `_local_catalogue()`
- `_fetch_ovh_catalogue()`, `_scope_includes()`
- `_build_catalogue()`, `_refresh_catalogue()`

Note: `_local_catalogue()` reads `loaded_models` and `loaded_vlm_models` from model_manager —
import by reference, no copy.

### `router.py`
What moves here from ov_server.py:
- `COMPLEXITY_SIGNALS`, `SIMPLE_Q_RE`
- `_task_class_embeddings`, `_routing_prompt_cache`, `_last_routing_decision`
- `_detect_signal()` — calls `_has_images()` (now in prompt_builder) and `_text_content()`
- `_compute_task_class_centroids()`, `_load_embedding_centroids()`
- `_route_by_embedding()`, `complexity_score()`, `_select_model()`

Note: `_route_by_embedding()` reads `emb_model` and `emb_tokenizer` from model_manager.
Import the module, access via `model_manager.emb_model` — avoids stale binding.

### `prompt_builder.py` (small addition)
Add `has_images(messages)` — currently called `_has_images()` in ov_server.py.
Used by `_detect_signal()` in router.py. Cannot import from ov_server.py without a cycle.
Function is 4 lines of message inspection — belongs in prompt_builder alongside `_text_content`.

### `ov_server.py` (what stays)
- FastAPI `app`, middleware classes, startup/shutdown handlers
- Pydantic models: `ChatRequest`, `EmbeddingRequest`, `ProfileRequest`, `ScopeRequest`
- `ServerStats`, `stats`, `_record_stats()`
- `_active_profile`, `_profile_switching`, `_profile_lock`, `_last_routing_decision`
- `_apply_profile()` — orchestrates model_manager + server_config; stays here
- `_system_snapshot_loop()`
- Image helpers: `_decode_image()`, `_pil_to_ov_tensor()`, `_extract_images()`, `_limit_image_history()`
  (used only in the VLM path of chat(); keeping here avoids PIL imports in other modules)
- All route handlers: `chat()`, `embeddings()`, `health()`, `/v1/models`, profile/scope/control endpoints

---

## Migration steps

Each step leaves the server in a running state. Verify after each.

### Step 0 — Consolidate mutable globals into `_cfg`
**File:** `ov_server.py` (pre-split, in-place)
**What:** Replace `MAX_LOADED_MODELS`, `DEFAULT_MODEL`, `AGENT_MODEL` module constants with
reads from `_cfg`. Update `_apply_profile()` to write to `_cfg` keys instead of reassigning globals.
**Verify:** restart server, confirm `/health` and a streaming chat request still work.
**Risk:** Low — behaviour-identical refactor, no new files.

### Step 1 — Extract `server_config.py`
**What:** Move all config/discovery code. Add `get_default_model()`, `get_agent_model()`.
In `ov_server.py`, replace direct definitions with `from server_config import ...`.
**Verify:** server starts, `/health` returns correct model list.
**Risk:** Low — no logic change, pure relocation of startup code.

### Step 2 — Extract `model_manager.py`
**What:** Move all state, locks, VRAM helpers, loaders, AsyncTokenStreamer, assessor.
Add `evict_all_models()` and `evict_all_vlms()` helpers.
Update `_apply_profile()` in ov_server.py to call these instead of directly manipulating dicts.
**Verify:** restart, `/health`, streaming chat, non-streaming chat, `/v1/embeddings`.
**Risk:** Medium — the `_apply_profile()` wiring is the fragile point. Test it explicitly
by switching profile via API after the change.

### Step 3 — Add `has_images()` to `prompt_builder.py`
**What:** Move `_has_images()` body, rename to `has_images()` (drop underscore — it is
exported now). Keep the private `_has_images` alias in ov_server.py for one commit
to ensure no missed callsites.
**Verify:** no import errors on startup.
**Risk:** Low — 4 lines.

### Step 4 — Extract `catalogue.py`
**What:** Move catalogue functions and state. Import `model_manager` module (not its names)
so `_local_catalogue()` accesses `model_manager.loaded_models` live.
**Verify:** `/v1/models` returns correct list including `loaded: true` for warm models.
**Risk:** Low — catalogue is read-only relative to model state.

### Step 5 — Extract `router.py`
**What:** Move routing functions and state. Access `emb_model`/`emb_tokenizer` via
`model_manager.emb_model` (module attribute access, not import binding) so the embedding
model loaded after startup is visible.
**Verify:** restart, send a chat request that triggers routing (model="Auto"),
check logs for routing decision.
**Risk:** Medium — `emb_model` and `emb_tokenizer` start as None and are set by
`get_embedding_model()` on first call. Module-attribute access (`model_manager.emb_model`)
reads the current value correctly; a local import binding (`from model_manager import emb_model`)
would be stale (always None). Must use the module-attribute form.

### Step 6 — Final tidy
- Remove the `_has_images` alias from ov_server.py
- Remove any `# noqa` / `# type: ignore` scaffolding added during migration
- Run black on all new files
- Apply 1st-order typing standards (Literal for finish_reason etc., `|` syntax)
- Update CLAUDE.md Architecture table with new module list
- Update File Conventions table

---

## Verification checklist (after Step 5)

```bash
curl -s http://localhost:11435/health | python3 -m json.tool
curl -s http://localhost:11435/v1/models | python3 -m json.tool
# streaming
curl -s -N -X POST http://localhost:11435/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Auto","messages":[{"role":"user","content":"hi"}],"stream":true}' | head -20
# non-streaming
curl -s -X POST http://localhost:11435/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Auto","messages":[{"role":"user","content":"hi"}],"stream":false}' | python3 -m json.tool
# embeddings
curl -s -X POST http://localhost:11435/v1/embeddings -H "Content-Type: application/json" -d '{"model":"","input":["test"]}' | python3 -m json.tool
```

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Stale binding for `emb_model` / `emb_tokenizer` in router.py | High if done wrong | Use `model_manager.emb_model`, not `from model_manager import emb_model` |
| `_apply_profile()` misses a dict reference after model_manager split | Medium | Add `evict_all_models()` / `evict_all_vlms()` helpers; test profile switch explicitly |
| Circular import introduced accidentally | Low if steps are followed in order | server_config has zero ov_server imports; work outward from there |
| `_has_images` callsite missed during rename | Low | grep before removing alias |

---

### Step 6 — Final tidy
- Remove the `_has_images` alias from ov_server.py
- Remove any `# noqa` / `# type: ignore` scaffolding added during migration
- Run black on all new files
- Apply 1st-order typing standards (Literal for finish_reason etc., `|` syntax)
- Update CLAUDE.md Architecture table with new module list
- Update File Conventions table

### Step 7 — Hybrid workflow artifacts
Three files that make the split codebase usable for smaller coding models and Aider.
None change server behaviour. All committed together at the end of Step 7.

#### 7a — Module docstrings
Add a 4-line docstring at the top of every new module (before imports).
Format:
```
"""
<One sentence: what this module owns.>
<One sentence: what it must never contain.>
Imports: <list of ov_server modules it depends on>.
To add <most common task>: <one-sentence recipe>.
"""
```
Example for model_manager.py:
```python
"""
Owns all model lifecycle state: loaded_models, VRAM tracking, locks, AsyncTokenStreamer.
Never import from ov_server.py, router.py, or catalogue.py.
Imports: server_config, db.
To add a new loader: follow the get_model() pattern (lock → check → evict → load → register state).
"""
```
Write docstrings for all five modules during their respective extraction steps,
not deferred to Step 7. Step 7a just audits and finalises them.

#### 7b — `CONVENTIONS.md`
A machine-facing coding conventions file at the repo root.
Target reader: any coding model (Qwen, Deepseek-Coder, etc.) asked to make a change.
NOT a copy of CLAUDE.md — no session protocol, no framework philosophy.
Content: module ownership map, import rules, config access pattern, state access pattern,
how to add an endpoint, 1st-order typing rules, hard rules, verification commands.
~100 lines. See draft at repo root.

Add one line to CLAUDE.md Python Code Standards section:
`- **Coding conventions for AI tools:** see `CONVENTIONS.md` — keep in sync when adding modules.`

#### 7c — `.aider.conf.yml`
Aider persistent config at repo root. Sets:
- Default model: `qwen3-30b-a3b-int4-ov` via `http://localhost:11435/v1` (dogfooding)
- Read-only context: `CONVENTIONS.md` injected into every Aider session
- Auto-commits: true, matching repo commit style
- Repo map size: 1024 tokens (enough to see all module signatures)
- Comments showing how to switch to architect mode (Claude reasoning + Qwen editing)
See draft at repo root.

#### Sync rule
If a new module is added or a module's ownership changes:
1. Update its docstring
2. Update the Module Map section in CONVENTIONS.md
3. Update the File Conventions table in CLAUDE.md
These three must always agree.

---

## Hybrid workflow — task taxonomy

Which tool to reach for, once the split is complete:

| Task | Tool | Reason |
|---|---|---|
| Multi-file architectural change | Claude Code | Cross-file reasoning, session state |
| Async / streaming bug | Claude Code | Implicit state, thread/loop interaction |
| Add Literal types to a module | Aider + Qwen3-30b-coder | Single file, pattern-matching |
| Rename function across codebase | Aider + Qwen3-30b-coder | Mechanical, grep + replace |
| Add a simple new endpoint | Aider + Qwen3-30b-coder | Template-driven, one file |
| Write a validation/unit test | Aider + Qwen3-30b-coder | Isolated, no cross-module state |
| Routing / async bug | Claude Code | Needs tracing across 3+ files |
| Session wrap / planning | Claude Code | Session protocol is Claude-specific |

Aider can also run in **architect mode**: Claude API as the reasoning model,
Qwen3-30b-coder via ov_server as the editing model. Useful for changes that need
strong reasoning but have mechanical execution. See `.aider.conf.yml` for the
commented-out architect config.

---

## Explicitly NOT in scope

- Changing any server behaviour
- Adding tests (separate task — requires Protocol abstraction first per coding_standards_python.json)
- Splitting `prompt_builder.py` further
- Moving image helpers out of ov_server.py (low value, adds PIL imports to more modules)
