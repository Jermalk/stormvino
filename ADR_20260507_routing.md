# ADR 2026-05-07 — Intelligent Routing Architecture

**Status:** Accepted  
**Author:** Jerzy Majchrzak  
**Supersedes:** implicit routing via `_pick_backend_name()` + profile `routing_default`

---

## Context

The server started as a single-model OpenAI-compatible endpoint. Over time it grew
profiles, backend routing, and model aliases to handle different use cases. By May 2026
the profile system conflated three orthogonal concerns:

- **Which provider** serves the request (local GPU / OVH cloud / other)
- **What hardware settings** to apply (KV budget, number of loaded models)
- **Which model to pick** for the task

This produced fragile combinations: switching to the `ovh` profile evicted local models
even though they were not the reason for the switch. Model choice was manual — the user
always had to know which model fit the task. There was no automatic routing, no
awareness of OVH pricing, and no extensible path toward multi-step task execution.

---

## Decision

Replace the monolithic profile system with three independent axes and an intelligent
routing pipeline:

```
provider_scope  →  what models exist          (local | local+ovh | all)
profile         →  how the task should run    (fast | precise | laborious)
routing         →  which model executes it    (rules → embedding → assessor)
```

### 1. Provider scope

A separate runtime control, switchable via `POST /admin/scope`.  
Values: `"local"` | `"local+ovh"` | `"all"`

Switching scope rebuilds the active model catalogue. It does not evict loaded models.

Remote catalogues (OVH) are fetched once and cached for `catalogue_ttl_sec` (default
300 s). On fetch failure the cached catalogue is retained until TTL expires, then the
provider is treated as unavailable without interrupting service.

### 2. Profiles — behavioral presets only

Profiles control *how* a request is executed, not *which model* executes it.

| Profile | Thinking | Max tokens | Model preference | Use assessor |
|---|---|---|---|---|
| `fast` | off | 512 | fastest | no — rules + embedding only |
| `precise` | on | 4 096 | balanced (best local) | yes |
| `laborious` | on | 16 384 | best (OVH if scope allows) | yes |

`model_preference` maps to position in the task-class model list:
- `fastest` → first entry with `provider: loc`
- `balanced` → last entry with `provider: loc`
- `best` → last entry overall, respecting active `provider_scope`

If no model satisfies the preference within the current scope, the system escalates:
fastest → balanced → best → first available (never returns 503 due to preference alone).

### 3. Task classes

Each task class defines a description, optional fast-path signal, and an ordered model
list. Models are annotated with `provider` and `tier`:

- `provider`: `"loc"` (local GPU), `"ovh"` (OVH AI Endpoints), `"ext"` (other)
- `tier`: `"fast"` | `"best"` — maps to `model_preference` in profiles

```jsonc
"document": {
  "description": "Summarise or analyse a long document or pasted text",
  "signal":      "long_context",
  "context_threshold_tokens": 4000,
  "models": [
    { "id": "qwen3-14b-int4-ov",         "provider": "loc", "tier": "fast" },
    { "id": "qwen3-30b-a3b-int4-ov",     "provider": "loc", "tier": "best" },
    { "id": "Qwen3-32B",                 "provider": "ovh", "tier": "fast" },
    { "id": "gpt-oss-120b",              "provider": "ovh", "tier": "best" }
  ]
}
```

Task classes defined at launch: `vision`, `web_search`, `document`, `code`, `general`.

### 4. Routing pipeline

```
Request
  │
  ├─ explicit model in req.model? ──────────────────────────── bypass, dispatch directly
  │
  ├─ STAGE 1 — signal rules  (<1 ms)
  │    has_image           → vision
  │    has_tools           → dispatch to assessor directly (tool-capable model needed)
  │    long_context        → document
  │    keyword match       → web_search
  │    → match? dispatch.  → no match? next stage.
  │
  ├─ STAGE 2 — embedding similarity  (~10 ms, e5-large already loaded)
  │    Embed query → cosine sim against task-class description centroids
  │    → sim ≥ threshold (default 0.72)? dispatch.
  │    → sim < threshold? next stage.
  │
  └─ STAGE 3 — assessor  (~1–2 s, precise/laborious only)
       Hidden pre-turn to qwen3-8b with routing prompt.
       Output: JSON task graph (single step now; multi-step in Phase 4).
       fast profile: stages 1–2 only, no assessor.
```

Routing decision is logged and exposed in `GET /health` as `last_routing_decision`.

### 5. Assessor model

`qwen3-8b-int4-ov` — permanently loaded in a dedicated LLMPipeline with 2 GB KV
budget. This pipeline is **outside** the task model pool (`max_loaded_models` does not
count it).

VRAM envelope on B60 (24 GB visible):

| Component | Weights | KV | Total |
|---|---|---|---|
| Assessor (qwen3-8b) | ~5 GB | 2 GB | 7 GB |
| Task model A (14b) | ~9 GB | 3 GB | 12 GB |
| Embeddings (e5-large) | ~1 GB | — | 1 GB |
| Headroom | | | ~4 GB |
| **Total** | | | **~24 GB** |

When the routing decision selects `qwen3-8b` as the task model, the assessor pipeline
is **reused** for the task execution — no second pipeline is loaded, no extra VRAM
consumed.

Assessor routing prompt is prefix-cacheable: the system block (task class descriptions
+ model catalogue) is static per scope/profile combination and will cache on warm turns.

### 6. Routing task graph (serialised JSON)

The assessor (and embedding stage) always emit a task graph, even in Phase 1 when it
contains exactly one step. This ensures the pipeline executor interface never changes.

```jsonc
// Phase 1 — single step
{
  "task_class": "document",
  "steps": [
    { "model": "qwen3-14b-int4-ov", "provider": "loc", "purpose": "summarise" }
  ],
  "confidence": 0.91,
  "strategy": "embedding"   // "rule" | "embedding" | "assessor"
}

// Phase 4 — multi-step (future)
{
  "task_class": "web_search",
  "steps": [
    { "model": "qwen3-8b-int4-ov",  "provider": "loc", "purpose": "scrape" },
    { "model": "qwen3-14b-int4-ov", "provider": "loc", "purpose": "summarise",
      "depends_on": [0] }
  ],
  "confidence": 0.85,
  "strategy": "assessor"
}
```

### 7. Model catalogue endpoint

`GET /v1/models` returns merged local + remote models respecting active `provider_scope`.
Each entry includes:

- `id` — model identifier
- `provider` — `loc` / `ovh` / `ext`
- `context_length` — from remote API or local config
- `pricing` — `null` for local; `{prompt, completion, currency_unit}` for remote
- `loaded` — `true` if currently in VRAM (local only)

### 8. New endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/admin/scope` | Switch provider scope at runtime |
| `GET` | `/v1/models` | Merged model catalogue (extended from current) |
| `GET` | `/health` | Existing; adds `last_routing_decision`, `provider_scope` |

---

## Gaps identified and mitigated

| Gap | Mitigation |
|---|---|
| Tool-call requests need tool-capable model | `has_tools` signal → dispatch to assessor pipeline directly |
| Assessor not ready at startup | `@app.on_event("startup")` preloads assessor; `precise`/`laborious` requests queue behind preload |
| OVH unavailable mid-session | Cached catalogue retained; affected models silently skipped; fallback to local |
| Assessor adds TTFT latency | `fast` profile skips assessor entirely; rule/embedding path is <15 ms |
| qwen3-8b in task pool AND assessor = double VRAM | Assessor pipeline reused when task model resolves to qwen3-8b |
| `active_profile` hardcoded in server | Read from `config.json` at startup; runtime switch via `/admin/profile` as before |
| Embedding threshold not tunable | `router.embedding_threshold` in config (default 0.72); operator-adjustable |
| VLM + assessor + 2 task models > 24 GB | VLM load still evicts LRU task models; assessor is never evicted |
| Pipeline executor interface not future-proof | Task graph JSON from day 1, even for single-step routing |
| No cost tracking | `GET /health` will include `estimated_session_cost_usd` (sum of OVH token costs) |

---

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Keep profiles as-is, add model override per profile | Mixing behavioral and model concerns; explosion of profile count |
| 1.5B LLM as sole router | Unreliable for compound intents; no tool use; 1.5B not currently on disk |
| Always-on embedding routing, no assessor | Embedding cosine sim struggles with domain-specific or code-heavy queries |
| FrugalGPT cascade (try small, escalate on low confidence) | Increases latency on every ambiguous query; assessor pre-classification is faster for clear cases |
| Remote routing service | External dependency; privacy concern; latency; overkill for personal server |

---

## Consequences

**Positive**
- Model selection is automatic; users interact at intent level, not model level
- Provider scope and profile are independently switchable without evicting models
- Routing decisions are visible (`/health`) — debuggable, not a black box
- Pipeline executor scaffolding is in place from Phase 1
- OVH pricing visible in model catalogue — cost-aware routing becomes possible

**Negative**
- Assessor adds ~1–2 s to TTFT for `precise`/`laborious` first turn (acceptable trade-off)
- More configuration surface in `config.json` — mitigated by good defaults
- Embedding threshold (0.72) will need tuning on real query data

---

## Implementation reference

See `PLAN_routing.md` for the phase-by-phase step-by-step implementation plan.
