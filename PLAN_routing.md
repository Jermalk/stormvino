# PLAN_routing.md — Intelligent Routing Implementation

> Step-by-step (SBS) plan. Each step is the smallest independently verifiable unit.
> Reference ADR: `ADR_20260507_routing.md`
> Start each session by reading PROGRESS.md NOW, then this file from the current step.

---

## Phase 1 — Config schema + model catalogue

**Goal:** New config.json format loaded and understood by the server. `GET /v1/models`
returns a merged, cached, annotated catalogue. Provider scope switchable at runtime.
Existing routing behaviour preserved via compatibility shim during transition.

---

### Step 1.1 — New config.json + loader

**Files:** `config.json`, `ov_server.py` (`_load_config`)

Write target `config.json` using the schema from ADR § Decision.
Key changes from current:
- Add `provider_scope`, `providers`, `assessor`, `router`, `task_classes` blocks
- Replace `profiles` shape (behavioral presets only — no `kv_cache_size_gb` etc.)
- Remove `default_model`, `agent_model`, `max_new_tokens_agent` (replaced by routing)
- Keep hardware keys: `max_loaded_models`, `kv_cache_size_gb`, `vram_headroom_gb`,
  `max_ram_percent`, `vlm_*`
- Read `active_profile` from config at startup (currently hardcoded `"speed"`)

Update `_load_config()` defaults to match new schema.
Add `_validate_config()` that logs warnings for unrecognised keys — no hard fail.

**Test:** Server starts, logs show new config fields parsed. `/health` shows
`active_profile` matching config value.

---

### Step 1.2 — Model catalogue builder

**Files:** `ov_server.py` (new `_build_catalogue()`)

```python
def _build_catalogue(scope: str) -> list[dict]:
    """Return merged model list for given provider_scope."""
```

- Always includes local discovered models (annotated `provider: "loc"`, `loaded: bool`)
- If scope includes `ovh`: fetch OVH `/v1/models`, cache result with TTL from config
- Each entry: `{id, provider, tier, context_length, pricing, loaded}`
- On OVH fetch failure: log warning, use cached result; if no cache → skip provider
- Cache stored in module-level `_catalogue_cache: dict` keyed by scope

**Test:** `python -c "from ov_server import _build_catalogue; print(_build_catalogue('local'))"` (no GPU needed — test in isolation).

---

### Step 1.3 — Extended `GET /v1/models`

**Files:** `ov_server.py` (`/v1/models` route)

Replace current simple model list with catalogue output.
Response shape (OpenAI-compatible + extensions):

```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen3-14b-int4-ov",
      "object": "model",
      "provider": "loc",
      "tier": "best",
      "context_length": 32768,
      "pricing": null,
      "loaded": true
    },
    {
      "id": "Qwen3-32B",
      "object": "model",
      "provider": "ovh",
      "tier": "best",
      "context_length": 32768,
      "pricing": { "prompt": "0.00000009", "completion": "0.00000025", "currency_unit": "USD" },
      "loaded": false
    }
  ]
}
```

**Test:** `curl http://localhost:11435/v1/models | jq '.data[] | {id, provider, loaded}'`

---

### Step 1.4 — `POST /admin/scope`

**Files:** `ov_server.py`

```python
class ScopeRequest(BaseModel):
    scope: str   # "local" | "local+ovh" | "all"

@app.post("/admin/scope")
async def set_scope(req: ScopeRequest) -> JSONResponse:
```

Validates scope value. Updates `_cfg["provider_scope"]`. Invalidates catalogue cache
(forces re-fetch on next `/v1/models` call). Returns 200 with new scope.
No model eviction — scope only affects routing decisions for new requests.

Add `provider_scope` to `/health` response.

**Test:** `curl -X POST .../admin/scope -d '{"scope":"local+ovh"}' && curl .../health | jq .provider_scope`

---

### Step 1.5 — Scope exposed in `/health` + ov-monitor

**Files:** `ov_server.py` (`health()`), `~/ov_monitor/ov_monitor.py`

`/health` response additions:
```json
"provider_scope": "local",
"last_routing_decision": null
```

ov-monitor: add scope display row to server panel. Add keyboard shortcut `s` to cycle
`local → local+ovh → all → local` (calls `/admin/scope`).

**Test:** ov-monitor shows scope; pressing `s` cycles it.

**Phase 1 complete when:** server starts with new config, catalogue merges providers,
`/v1/models` shows annotated list, scope switchable at runtime.

---

## Phase 2 — Rule-based routing + profile behavioral settings

**Goal:** Incoming requests are automatically routed to the right task class and model.
Profile behavioral settings (thinking, max_tokens) applied. Routing decision visible
in `/health`. Assessor not yet wired.

---

### Step 2.1 — Signal detector

**Files:** `ov_server.py` (new `_detect_signal(req) -> str | None`)

```python
def _detect_signal(req: ChatRequest) -> str | None:
    """Return task_class name if a fast-path signal fires, else None."""
```

Signals (checked in order):
1. `has_image` → `"vision"` (reuse existing `_has_images()`)
2. `has_tools` → `None` (tools in req.tools → bypass task-class routing, dispatch to assessor)
3. `long_context` → `"document"` (prompt token count > threshold from config)
4. `keyword` → `"web_search"` (any keyword from task_class config matches last user message)

Signal check is O(1) or O(n_keywords) — always <1 ms.

**Test:** Unit test with synthetic ChatRequest objects covering each signal case.

---

### Step 2.2 — Embedding similarity router

**Files:** `ov_server.py` (new `_route_by_embedding(query: str) -> tuple[str, float]`)

At startup: compute centroid embedding for each task class description using the
already-loaded e5-large model. Store as `_task_class_embeddings: dict[str, np.ndarray]`.

On each ambiguous request:
1. Embed the last user message (mean-pool, L2-normalise — same as embeddings endpoint)
2. Cosine similarity against all task class centroids
3. Return `(task_class_name, similarity_score)`

If `similarity_score >= router.embedding_threshold` → use this task class.
Else → fall through to assessor (Phase 3) or `"general"` (Phase 2 fallback).

**Note:** This reuses the embedding model already in VRAM. No extra memory cost.

**Test:** A few representative sentences for each task class should score ≥ 0.72 to their
correct class. Tune threshold if needed on real query data.

---

### Step 2.3 — Model selector

**Files:** `ov_server.py` (new `_select_model(task_class: str, profile: dict) -> dict`)

```python
def _select_model(task_class: str, profile: dict) -> dict:
    """Return {id, provider} for the best model given preference + active scope."""
```

Algorithm:
1. Get model list for `task_class` from config
2. Filter by `provider_scope`: skip entries whose provider is not in active scope
3. Apply `model_preference`:
   - `fastest` → first entry with `tier: fast` and `provider: loc`
   - `balanced` → last entry with `provider: loc`
   - `best` → last entry overall (may be `ovh` if scope allows)
4. Escalate if no match: `fastest → balanced → best → any available`
5. If list is empty after filtering → return assessor model as fallback + log warning

**Test:** Unit tests covering all preference × scope combinations for each task class.

---

### Step 2.4 — Wire routing into `chat()`

**Files:** `ov_server.py` (`chat()`)

Replace the `backend_name = _pick_backend_name(req.model)` block with:

```python
# 1. Explicit model override — bypass routing
if req.model and req.model in AVAILABLE_MODELS:
    model_id = req.model
    routing_decision = {"strategy": "explicit", "task_class": None}
else:
    # 2. Signal detection
    task_class = _detect_signal(req)
    strategy = "rule"
    if task_class is None:
        # 3. Embedding similarity
        task_class, score = await loop.run_in_executor(None, _route_by_embedding, last_user_msg)
        strategy = "embedding" if score >= threshold else "general_fallback"
        if score < threshold:
            task_class = "general"
    model_entry = _select_model(task_class, active_profile_cfg)
    model_id = model_entry["id"]
    routing_decision = {"task_class": task_class, "model": model_id, "strategy": strategy}

_last_routing_decision = routing_decision   # read by /health
```

Apply profile behavioral settings:
- `thinking = active_profile_cfg["thinking"]`
- `max_new_tokens = active_profile_cfg["max_new_tokens"]`

**Test:** Streaming + non-streaming requests; check `/health` shows `last_routing_decision`.

---

### Step 2.5 — Routing decision in `/health` + ov-monitor

**Files:** `ov_server.py`, `ov_monitor.py`

`/health` response addition:
```json
"last_routing_decision": {
  "task_class": "document",
  "model": "qwen3-14b-int4-ov",
  "strategy": "embedding",
  "confidence": 0.84,
  "latency_ms": 11
}
```

ov-monitor: add "Last route" row to server panel showing `task_class → model (strategy)`.

**Phase 2 complete when:** all requests are routed automatically, routing decision visible
in monitor, profile behavioral settings applied.

---

## Phase 3 — Assessor

**Goal:** Qwen3-8B permanently loaded as a separate pipeline. Fires for ambiguous queries
in `precise`/`laborious` profiles. Routing quality for complex/compound queries improves
significantly.

---

### Step 3.1 — Assessor pipeline bootstrap

**Files:** `ov_server.py`

Add module-level:
```python
_assessor_pipe: ov_genai.LLMPipeline | None = None
_assessor_lock = asyncio.Lock()
```

In `@app.on_event("startup")`:
```python
asyncio.create_task(_load_assessor())
```

`_load_assessor()`:
- Loads `assessor.model` from config using same LLMPipeline init as task models
- Uses `assessor.kv_cache_size_gb` — separate KV budget
- Sets `_assessor_pipe`
- Logs load time and VRAM allocated
- Does NOT add to `loaded_models` dict — excluded from LRU and max_loaded_models

**VRAM accounting:** `_vram_allocated` must include assessor's allocation so
`vram_free_gb()` and the VRAM bar in ov-monitor remain accurate.

**Test:** Server starts, logs show `[assessor] loaded qwen3-8b in X.Xs`. `/health` shows
assessor model under new `assessor_loaded: true` field.

---

### Step 3.2 — Assessor routing prompt

**Files:** `ov_server.py` (new `_build_routing_prompt(req, task_classes, profile) -> str`)

Prompt structure (prefix-cacheable — static block first):

```
<|im_start|>system
You are a routing agent. Given a user query, select the best task class and model.
Output only valid JSON. Do not explain.

Task classes:
{json: task_class descriptions + available models for current scope/preference}

Active profile: {name} — {description}
<|im_end|>
<|im_start|>user
{last user message, truncated to 512 tokens}
<|im_end|>
<|im_start|>assistant
```

Expected output (validated with `json.loads`, fallback to `"general"` on parse failure):
```json
{
  "task_class": "document",
  "steps": [{ "model": "qwen3-14b-int4-ov", "provider": "loc", "purpose": "summarise" }],
  "confidence": 0.91,
  "reasoning": "long pasted text, no image"
}
```

**Test:** Send routing prompt to assessor manually via curl; verify JSON output.

---

### Step 3.3 — Wire assessor into routing pipeline

**Files:** `ov_server.py` (`chat()`, `_route_by_embedding`)

Extend Phase 2 routing logic:

```python
if score < threshold and active_profile_cfg.get("use_assessor") and _assessor_pipe:
    routing_json = await _run_assessor_routing(req)
    task_class = routing_json["task_class"]
    strategy = "assessor"
    confidence = routing_json.get("confidence", 0.0)
```

`_run_assessor_routing()`:
- Acquires `_assessor_lock` (routing decisions serialised)
- Builds routing prompt
- Calls `_assessor_pipe.generate()` in executor with `max_new_tokens=256`
- Parses JSON; falls back to `("general", 0.0)` on any error

When task resolves to assessor model (`qwen3-8b-int4-ov`), reuse `_assessor_pipe`
for task execution — no second pipeline loaded.

**Test:** Ambiguous query ("tell me something interesting about the news") should
route via assessor and show `strategy: "assessor"` in `/health`.

**Phase 3 complete when:** ambiguous queries route through assessor for
precise/laborious profiles; VRAM bar accounts for assessor; monitor shows strategy.

---

## Phase 4 — Pipeline executor v1 (sequential)

**Goal:** Assessor can emit multi-step plans; server executes steps sequentially,
passing output of step N as context to step N+1. Enables web-search → summarise.

---

### Step 4.1 — Task graph executor

**Files:** `ov_server.py` (new `_execute_task_graph(graph: dict, req: ChatRequest)`)

```python
async def _execute_task_graph(graph: dict, req: ChatRequest) -> AsyncGenerator[str, None]:
    context = ""
    for i, step in enumerate(graph["steps"]):
        model_id = step["model"]
        step_req = _build_step_request(req, step, context, i)
        context = await _run_step(model_id, step_req)
    # stream context (final step output) to client
```

For Phase 4, only linear dependencies (`depends_on: [i-1]` implicitly). Parallel steps
with explicit `depends_on` lists are Phase 5.

Step result is the full generated text (non-streaming internally; final step streams to
client).

**Test:** Manually craft a 2-step graph JSON; verify second step receives first step
output in its context.

---

### Step 4.2 — Web-search → summarise scenario

**Files:** `ov_server.py` (assessor prompt update), test

Update assessor prompt to include the `web_search` multi-step template:
```json
{
  "task_class": "web_search",
  "steps": [
    { "model": "qwen3-8b-int4-ov",  "provider": "loc", "purpose": "extract search terms and scrape" },
    { "model": "qwen3-14b-int4-ov", "provider": "loc", "purpose": "summarise results" }
  ]
}
```

Test end-to-end: "search for recent OpenVINO release notes" → assessor emits 2-step
graph → step 1 extracts/scrapes → step 2 summarises → streamed to client.

**Phase 4 complete when:** at least the web-search scenario works end-to-end.

---

## Phase 5 — Pipeline executor v2 (parallel + synthesis) — FUTURE

> Do not implement until Phase 4 is stable and a real use case demands it.

- Parallel step execution with `asyncio.gather`
- Dependency resolution via `depends_on` lists
- Result synthesiser: final step receives outputs of all parallel predecessors
- Conditional branching: step result determines next step selection

---

## Migration notes

| Current symbol | Replaced by |
|---|---|
| `DEFAULT_MODEL` | `_select_model("general", profile)` |
| `AGENT_MODEL` | `assessor.model` from config |
| `MAX_NEW_TOKENS_AGENT` | `profile.max_new_tokens` (fast profile) |
| `_pick_backend_name()` | `_detect_signal()` + `_route_by_embedding()` + `_run_assessor_routing()` |
| `profiles` config block | new `profiles` shape (behavioral presets only) |
| `routing.default` per profile | `provider_scope` (separate control) |

Old `_proxy_chat()` and `_build_backends()` remain for OVH forwarding — they are called
by `_select_model()` when provider is `ovh`, not by the profile system.

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Embedding threshold wrong for real queries | Medium | Medium | Make threshold configurable; log all routing decisions for offline tuning |
| Assessor parse failure on routing JSON | Low | Low | Fallback to `"general"` class; log parse error |
| Assessor VRAM + 2 task models > 24 GB | Low | High | Assessor reuses pipeline when task = qwen3-8b; VLM evicts task models |
| OVH catalogue stale during provider outage | Low | Low | Cache retained; affected models skipped silently |
| Phase 4 step context blows KV budget | Medium | Medium | Truncate step context to 75% of KV budget before feeding next step |
