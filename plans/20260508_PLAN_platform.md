# PLAN_platform.md — ov_server as a Governed AI Platform

> Proposal — nothing here is scheduled. User decides before any phase begins.
> Each phase is independently shippable and testable.
> Builds entirely on existing infrastructure — no new external services required.

---

## Vision

Transform ov_server from a single-user local inference server into a **governed, multi-user AI platform** running fully on-premises:

- Users access AI services through policy-enforced API keys
- Local LLM inference (Arc B60) is the default; cloud (OVH) is opt-in per policy
- Internal documents are searchable via fully local RAG (pgvector + local embeddings)
- Admins manage users, keys, collections, and monitor usage through a Svelte web UI
- Everything runs on one machine — no SaaS dependencies, no data leaves the building

```
┌─────────────────────────────────────────────────┐
│              Svelte Monitor / Admin UI           │
│   live stats · user keys · RAG · usage graphs   │
└─────────────────────┬───────────────────────────┘
                      │  served by ov_server
┌─────────────────────▼───────────────────────────┐
│                   ov_server                     │
│   routing · inference · RAG injection · auth    │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│                 PostgreSQL                      │
│   ov_metrics · api_keys · documents · vectors   │
│   + pgvector extension                          │
└─────────────────────────────────────────────────┘
```

---

## Phase 1 — API Key Auth + User Policies

### Goal
Every request is authenticated. Each key carries a policy that governs what the user can do.

### Data model (Postgres)

```sql
CREATE TABLE api_keys (
    id          SERIAL PRIMARY KEY,
    key_hash    TEXT UNIQUE NOT NULL,    -- SHA-256 of the raw key
    label       TEXT NOT NULL,           -- "alice", "team-dev", etc.
    is_admin    BOOLEAN DEFAULT FALSE,
    scope       TEXT DEFAULT 'local',    -- 'local' | 'local+ovh'
    profiles    TEXT[] DEFAULT '{fast}', -- ['fast','precise','laborious']
    allowed_models TEXT[],              -- NULL = all in scope
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    last_used   TIMESTAMPTZ,
    expires_at  TIMESTAMPTZ             -- NULL = never
);
```

### Enforcement points in ov_server

| Location | What is enforced |
|---|---|
| FastAPI middleware | Validate `Authorization: Bearer <key>`, attach policy to request state |
| `chat()` routing | `scope` overrides `_cfg["provider_scope"]` per-request |
| `chat()` routing | `profiles` caps the effective profile (request can't exceed user's max) |
| `chat()` model selection | `allowed_models` rejects disallowed explicit model requests (403) |
| `/admin/*` endpoints | Require `is_admin=True` |
| `/monitor/*` (Svelte UI) | Require `is_admin=True` |

### Config additions

```json
"auth": {
    "enabled": true,
    "anonymous_scope": "local",
    "anonymous_profiles": ["fast"]
}
```

`anonymous_*` defines what unauthenticated requests get — allows gradual rollout.

### Key issuance

Keys are 32-byte random hex strings (`sk-ov-<hex>`), stored as SHA-256 hashes only.
Admin creates keys via Svelte UI or `POST /admin/keys`.

### Implementation files
- `ov_server.py` — `AuthMiddleware`, policy injection into `request.state`
- SQL migration — `api_keys` table
- `POST /admin/keys`, `GET /admin/keys`, `DELETE /admin/keys/{id}`

### Estimated effort: 1–2 sessions

---

## Phase 2 — pgvector RAG

### Goal
Users can upload documents into policy-gated collections. Relevant chunks are automatically
injected into chat context when a collection is referenced.

### Infrastructure

```sql
-- One-time setup
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE rag_collections (
    id          SERIAL PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,
    description TEXT,
    allowed_keys TEXT[],   -- NULL = all authenticated users
    created_by  TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE rag_documents (
    id            SERIAL PRIMARY KEY,
    collection_id INT REFERENCES rag_collections(id) ON DELETE CASCADE,
    source        TEXT,        -- filename, URL, etc.
    chunk_index   INT,
    content       TEXT,
    embedding     vector(1024), -- multilingual-e5-large output dim
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON rag_documents USING ivfflat (embedding vector_cosine_ops);
```

### Ingestion pipeline

```
POST /v1/rag/upload
  file (PDF | TXT | MD | DOCX)  +  collection_id

ov_server:
  1. Extract text (pdfplumber / python-docx / plain)
  2. Chunk: 512-token windows, 64-token overlap
  3. Batch-embed chunks → multilingual-e5-large (GPU.0, already loaded)
  4. INSERT INTO rag_documents (content, embedding, ...)
```

### Retrieval + injection

Triggered by `@collection-name` in user message, or by a system prompt directive.

```
chat() pre-processing:
  1. Detect @collection reference in last user message
  2. Embed user query (reuse embedding model — already warm)
  3. SELECT content FROM rag_documents
     WHERE collection_id = ?
     ORDER BY embedding <=> $query_vec LIMIT 5
  4. Prepend retrieved chunks to system prompt as:
     ### Context from [collection]:
     [chunk 1]
     [chunk 2]
     ...
  5. Continue normal routing + inference
```

### Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/rag/collections` | Create collection (admin or scoped) |
| `GET /v1/rag/collections` | List accessible collections |
| `POST /v1/rag/upload` | Ingest document into collection |
| `DELETE /v1/rag/documents/{id}` | Remove document |
| `POST /v1/rag/query` | Direct similarity search (debug/test) |

### Policy integration
- Collection `allowed_keys` is checked at retrieval time
- Users only see and query collections they have access to
- Upload permission can be gated separately (e.g. only power users)

### Implementation files
- `ov_server.py` — ingestion endpoint, retrieval function, chat() injection
- `rag.py` — chunking, embedding batch, similarity query (extracted module)
- SQL migration — `rag_collections`, `rag_documents` tables + ivfflat index

### Estimated effort: 2–3 sessions

---

## Phase 3 — Svelte Monitor / Admin UI

### Goal
Replace the curses-based `ov_monitor.py` with a web dashboard served directly by ov_server.
Admin-only. All data comes from `/health`, Postgres, and new admin endpoints.

### Repository integration
```
/opt/ov_server/
  monitor/
    src/
      App.svelte
      lib/
        LiveStats.svelte      ← /health polling
        ModelPanel.svelte
        VramBar.svelte
        UsersPanel.svelte     ← api_keys CRUD
        CollectionsPanel.svelte ← RAG management
        Charts.svelte         ← Postgres historical data
    package.json
    vite.config.js
    dist/                     ← compiled output, served by ov_server
```

### Served by ov_server

```python
# Static mount — built Svelte bundle
app.mount("/monitor", StaticFiles(directory="monitor/dist", html=True))
# All /monitor/* requests require is_admin=True (middleware)
```

### Panels

| Panel | Data source | Update |
|---|---|---|
| Live Stats | `GET /health` | Poll 2s |
| VRAM Bar | `GET /health` → `vram_allocated_gb` | Poll 2s |
| Loaded Models | `GET /health` → `loaded_models` | Poll 2s |
| Requests/min | `GET /admin/stats` | Poll 5s |
| Users & Keys | `GET /admin/keys` | On demand |
| RAG Collections | `GET /v1/rag/collections` | On demand |
| Throughput graph | Postgres `ov_requests` | Poll 30s |
| Model usage pie | Postgres `ov_requests` | Poll 30s |
| Token histogram | Postgres `ov_requests` | Poll 30s |
| Latency timeline | Postgres `ov_requests` | Poll 30s |

### Tech stack
- **Svelte 5** — minimal bundle, reactive by default
- **Chart.js** — lightweight charts (no D3 complexity for this use case)
- **Vite** — build tool, dev server with HMR
- No backend framework needed — Svelte talks directly to ov_server REST endpoints

### Build workflow
```bash
cd monitor && npm run build   # outputs to monitor/dist/
# ov_server serves monitor/dist/ at /monitor
```

### Implementation files
- `monitor/` — Svelte project (new directory in repo)
- `ov_server.py` — `StaticFiles` mount, admin auth on `/monitor`
- `GET /admin/stats` — aggregated metrics endpoint for dashboard

### Estimated effort: 3–4 sessions

---

## Phase 4 — Profile Selection in UI

### Goal
Users can select their profile from Open WebUI without manual API calls.
Profiles exposed as model variants in the catalogue.

### Implementation

Extend `_build_catalogue()` to emit profile-tagged Auto entries:

```json
{ "id": "Auto",           "description": "Automatic routing · fast profile" },
{ "id": "Auto [precise]", "description": "Automatic routing · precise profile (thinking)" },
{ "id": "Auto [laborious]","description": "Automatic routing · laborious profile (best model)" }
```

`chat()` parses the model name, applies the profile for that request only (no global state change).
Policy enforcement: user's `profiles` list gates which variants are visible in `/v1/models`.

### Estimated effort: 0.5 sessions

---

## Implementation Order

```
Phase 1 (Auth)      →  Phase 2 (RAG)  →  Phase 3 (Monitor)  →  Phase 4 (Profiles)
  1–2 sessions         2–3 sessions        3–4 sessions           0.5 sessions
```

Phases 1 and 2 are pure backend — no UI required, testable with curl.
Phase 3 builds on Phase 1 (needs auth to gate the monitor).
Phase 4 is small and can slot in anywhere after Phase 1.

---

## What is NOT needed

- No new external services (auth server, vector DB service, object storage)
- No Docker for ov_server itself
- No change to the OpenAI-compatible API contract — existing clients keep working
- No cloud dependency — OVH is already optional and policy-gated

---

## Open Questions

1. **Anonymous access**: keep it for LAN use (just `fast` + `local`) or require keys for everyone from day one?
2. **Key distribution**: how do users get their keys? Admin hands them out, or self-registration with approval?
3. **RAG chunking strategy**: fixed 512 tokens vs semantic chunking (slower but more coherent)?
4. **DOCX/PDF extraction**: `pdfplumber` + `python-docx` sufficient, or need OCR for scanned PDFs?
5. **Monitor auth**: is_admin flag on the key, or a separate admin password for the UI?
6. **STT integration** (from PLAN_future.md): fits naturally as Phase 5 — upload audio → transcribe → ingest as RAG document.
