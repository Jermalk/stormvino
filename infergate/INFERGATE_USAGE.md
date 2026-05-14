# infergate in ov_server — Integration Story

**Purpose:** Prove infergate's usefulness as a routing library by wiring it into ov_server.
**Parallel track:** infergate is developed independently; this folder is the ov_server side only.
**Hard rule:** ov_server-specific concepts (OpenVINO pipeline, VRAM eviction, openvino_genai,
GPU.1 device, AnythingLLM/n8n quirks) stay here. Nothing ov_server-specific leaks into infergate.

---

## Folder layout

```
/opt/ov_server/infergate/
  INFERGATE_USAGE.md        — this file; top-level story + links
  ov_backend.py             — OVServerBackend (Backend Protocol impl)
  ov_embedding_provider.py  — OVEmbeddingProvider (EmbeddingProvider Protocol impl)
  config.yaml               — infergate-style routing config (mirrors config.json task_classes)
  DECISIONS.md              — decisions made during this integration track (append-only)
  PROGRESS.md               — NOW section for this track (overwritten each session)
```

---

## Integration approach: routing-only mode

infergate's `Router.decide()` replaces the three-function chain in `router.py`:
- `_detect_signal()` → `signals.task_class_directive()` + `signals.detect_signal()`
- `route_by_embedding()` → `embeddings.route_by_embedding()` (via `Router.decide`)
- `_select_model()` → `selector.select_model()` (via `Router.decide`)

The inference pipeline after routing is untouched:
`_build_prompt()` → `pipeline.generate()` → streaming/non-streaming response.

---

## Adapter pieces needed

### 1. `OVServerBackend` — `infergate/ov_backend.py`
Implements `infergate.protocols.Backend`. No HTTP — reads from ov_server's live globals.

```python
from infergate.protocols import Backend
from infergate.types import InferRequest
import model_manager
from server_config import AVAILABLE_MODELS, AVAILABLE_VLM_MODELS

class OVServerBackend:
    is_local = True

    def name(self) -> str:
        return "ov_server"

    def available_models(self) -> list[str]:
        return list(AVAILABLE_MODELS | AVAILABLE_VLM_MODELS)

    def loaded_model_ids(self) -> list[str]:
        return (
            list(model_manager.loaded_models.keys())
            + list(model_manager.loaded_vlm_models.keys())
        )

    async def chat(self, request: InferRequest, model_id: str) -> dict:
        raise NotImplementedError("routing-only mode — ov_server handles execution")
```

### 2. `OVEmbeddingProvider` — `infergate/ov_embedding_provider.py`
Wraps the existing `model_manager.emb_model` / `emb_tokenizer` as `EmbeddingProvider`.
The forward pass already runs in executor in `router.py` — same pattern here.

### 3. Config — `infergate/config.yaml`
Mirrors the `task_classes` block from `config.json`, translated to infergate field names:
- `provider: "loc"` → `backend: "ov_server"` (matches `OVServerBackend.name()`)
- `provider: "ovh"` → `backend: "ovh"` (for future OVH backend)
- `max_context_tokens` → `ctx_limit`
- `signal: "has_image"` → `signal_only: true`

### 4. Wiring — `ov_server.py` startup
```python
from infergate import Router, RouterConfig
from infergate.infergate_dir.ov_backend import OVServerBackend
from infergate.infergate_dir.ov_embedding_provider import OVEmbeddingProvider

# In lifespan startup, after embedding model is loaded:
_ov_backend = OVServerBackend()
_ov_emb_provider = OVEmbeddingProvider()
_router = Router.from_config(
    config=yaml.safe_load(Path("infergate/config.yaml").read_text()),
    backends={"ov_server": _ov_backend},
    embedding_provider=_ov_emb_provider,
)
await _router.load_embeddings()
```

In `chat()` endpoint, replace the routing block with:
```python
infer_req = InferRequest(messages=req.messages, tools=req.tools, force_tier=...)
decision = await _router.decide(infer_req)
model_id = decision.model_id
```

---

## What stays in ov_server (must not go into infergate)

| Concept | Why it stays |
|---|---|
| `openvino_genai.LLMPipeline` | OpenVINO-specific, no abstraction value for the library |
| VRAM eviction / LRU logic | Hardware-specific scheduling for Intel Arc GPU.1 |
| `AsyncTokenStreamer` | openvino_genai streaming subclass |
| `build_prompt()` / ChatML building | Qwen3/VLM prompt format, not routing |
| `catalogue.py` OVH remote fetch | ov_server's multi-source model catalogue |
| AnythingLLM / n8n quirks | Deployment-specific signal handling |
| GPU.1 device assertion | EnvyStorm-specific hardware check |

---

## Implementation order

1. `infergate/config.yaml` — translate task_classes from config.json
2. `infergate/ov_backend.py` — OVServerBackend (20-30 lines)
3. `infergate/ov_embedding_provider.py` — OVEmbeddingProvider (20-25 lines)
4. Wire into `ov_server.py` lifespan + `chat()` endpoint
5. Delete `router.py` routing functions that are now replaced
6. Smoke test: `/health`, `/v1/models`, streaming and non-streaming chat, vision request

---

## Status

| Step | Status |
|---|---|
| Gap analysis | Done (20260514_infergate_gaps.md) |
| All gaps fixed in infergate 0.1.1 | Confirmed (code review 2026-05-14) |
| config.yaml | Done ✓ — verified against 0.1.2 wheel |
| OVServerBackend | Done ✓ — Backend protocol satisfied |
| OVEmbeddingProvider | Done ✓ — EmbeddingProvider protocol satisfied |
| Wiring | Done ✓ — startup + chat() routing live |
| Smoke test | Done ✓ — health, non-streaming, streaming, #code directive |

---

## Feedback loop

After each integration round, the ov_server session writes a developer letter to the infergate session.
Full protocol is in `feedback/SIGNAL.md` (infergate repo). Summary:

| Who | Action | File written |
|---|---|---|
| ov_server session | completes a round, hits friction | `feedback/round_NN_vX.Y.Z.md` (copy ROUND_TEMPLATE.md) |
| ov_server session | updates handoff flag | `feedback/SIGNAL.md` → Direction: FEEDBACK READY |
| infergate session | reads round file, ships PyPI release | `feedback/addressed_NN_vX.Y.Z.md` (copy RESPONSE_TEMPLATE.md) |
| infergate session | updates handoff flag | `feedback/SIGNAL.md` → Direction: RELEASE READY |
| ov_server session | upgrades venv, starts next round | `source /home/jerzy/ov_env/bin/activate && pip install --upgrade infergate` |

**Hard rule:** feedback files describe ov_server integration experience only.
Do not propose features driven by ov_server internals (OpenVINO pipeline, VRAM eviction,
openvino_genai). Those stay in ov_server. Only propose library-level API gaps.

---

## Open questions

- **Embedding model:** Keep `OVModelForFeatureExtraction` (OpenVINO-accelerated) or switch to
  `SentenceTransformerProvider`? Keeping OV preserves GPU acceleration, requires the adapter.
  Switching is simpler but adds a new dependency and loses GPU embedding.
- **OVH backend:** `OpenAICompatBackend` from infergate can wrap the OVH endpoint. Decide when
  OVH scope is needed.
- **Config sync:** `config.json` and `infergate/config.yaml` will have overlapping data. Plan
  for a single source of truth once integration is stable.
