# CONVENTIONS.md — ov_server coding conventions

> Machine-facing reference for coding models (Qwen, Deepseek-Coder, Aider, etc.).
> No session protocol. No philosophy. Actionable facts only.
> Human/Claude session rules live in CLAUDE.md. Keep both in sync when adding modules.

---

## Module ownership map

| File | Owns | Never put here |
|---|---|---|
| `server_config.py` | Config loading, model discovery, startup constants, `_cfg` dict | Any model loading, FastAPI, state mutation |
| `model_manager.py` | Loaded model state, VRAM tracking, locks, loaders, AsyncTokenStreamer | Route handlers, catalogue, routing logic |
| `catalogue.py` | Model catalogue (local + remote), TTL cache | Model loading, route handlers |
| `router.py` | Signal detection, embedding routing, model selection, assessor | Model loading, catalogue fetch, route handlers |
| `prompt_builder.py` | Message types, prompt construction, tool-call parsing, `has_images()` | Any I/O, model loading, FastAPI |
| `db.py` | Postgres writes and queries | Model logic, routing, FastAPI |
| `ov_server.py` | FastAPI app, endpoints, middleware, profile switching, image helpers | Business logic that belongs in the modules above |

---

## Import rules — the dependency graph

```
db              ← no ov_server imports
server_config   ← no ov_server imports
prompt_builder  ← no ov_server imports
model_manager   ← server_config, db
catalogue       ← server_config, model_manager
router          ← server_config, model_manager, db, prompt_builder
ov_server       ← all of the above
```

**Rule:** never introduce an import that creates a cycle.
If module A already imports module B, module B must not import module A.

---

## Config access

All runtime config lives in `_cfg` — a plain dict imported from `server_config`.
Because Python dict imports are by reference, all modules share the same object.
Mutations (e.g. from profile switching) are immediately visible everywhere.

```python
from server_config import _cfg

# Read a value
kv_gb = _cfg.get("kv_cache_size_gb", 8)

# Read the current default model (resolved at startup, may change on profile switch)
from server_config import get_default_model
model_id = get_default_model()
```

Do not copy config values into module-level constants — they become stale after a profile switch.
Exception: immutable values set once at startup (`MODELS_DIR`, `DEVICE`, `CONFIG`) are safe as constants.

---

## State access across modules

Module-level variables in `model_manager.py` (e.g. `emb_model`, `loaded_models`) start as
`None` or empty and are populated at runtime. Import the **module**, not the **name**:

```python
# WRONG — stale: captures None at import time, never sees the loaded model
from model_manager import emb_model

# CORRECT — live: reads the current value every time
import model_manager
model = model_manager.emb_model
```

---

## Adding a new endpoint

1. Open `ov_server.py` only.
2. Add the Pydantic request model (if needed) near the other models in ov_server.py (around `ChatRequest`).
3. Add the route handler using `@app.get` or `@app.post`.
4. Call into the relevant module (`model_manager`, `catalogue`, `router`) — do not
   duplicate logic that already exists there.
5. Verify: `curl -s http://localhost:11435/<your-path> | python3 -m json.tool`

---

## Adding a new config key

1. Add it with a sensible default in `_load_config()` defaults dict in `server_config.py`.
2. Add the key name to `_KNOWN_CONFIG_KEYS` in the same file (prevents spurious warnings).
3. Read it via `_cfg.get("your_key", default)` wherever needed.
4. Document it in `config.json` with a comment.

---

## 1st-order typing rules (always apply)

```python
# Literal for categorical sentinels
from typing import Literal
FinishReason = Literal["stop", "length", "tool_calls"]

# Modern union syntax (Python 3.12+)
def foo(x: int | None) -> list[str]: ...   # not Optional[int], not List[str]

# openvino_genai has no type stubs — annotate the boundary and move on
pipe: ov_genai.LLMPipeline = ...  # type: ignore[no-untyped-call]

# Domain-specific names
stream_chunk: str  # not: token, output, data, result
completion_tokens: int  # not: count, n, tokens
```

Replace legacy imports whenever you touch a file:
`from typing import Optional, Dict, List, Tuple, Union` → delete; use built-in generics.

---

## Hard rules

- **`python3`** — the `python` binary does not exist on this machine.
- **No `sudo pip install`** — venv is at `/home/jerzy/ov_env`. Activate it first.
- **No bare `except:`** — catch specific exceptions (`Exception`, `HTTPException`, `OSError`…).
- **Type hints on every function signature** — parameters and return type.
- **`pathlib.Path`** over `os.path`.
- **`asyncio.get_running_loop()`** — never `asyncio.get_event_loop()` (deprecated).
- **Blocking work in executor** — `await loop.run_in_executor(None, fn)`, never `await` CPU-bound calls.
- **Do not break `/health`, `/v1/models`, `/v1/embeddings`** when editing the chat path.
- **Test streaming AND non-streaming** after any change to `chat()`.

---

## Verification commands

```bash
curl -s http://localhost:11435/health | python3 -m json.tool
curl -s http://localhost:11435/v1/models | python3 -m json.tool
curl -s -N -X POST http://localhost:11435/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Auto","messages":[{"role":"user","content":"hi"}],"stream":true}' | head -20
curl -s -X POST http://localhost:11435/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Auto","messages":[{"role":"user","content":"hi"}],"stream":false}' | python3 -m json.tool
curl -s -X POST http://localhost:11435/v1/embeddings -H "Content-Type: application/json" -d '{"model":"","input":["test"]}' | python3 -m json.tool
```

Server is at `http://localhost:11435`. Service name: `ov-server` (hyphen).
Restart: `sudo systemctl restart ov-server`
Logs: `journalctl -u ov-server -f`
