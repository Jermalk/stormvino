# ov_server — Technical Capabilities & Limitations
**Version:** 0.9.0 | **Date:** 2026-05-13 | **Host:** EnvyStorm

---

## 1. Hardware Platform

| Component | Specification |
|---|---|
| Machine | EnvyStorm (Linux Mint) |
| CPU | 12th Gen Intel Core i7-12700K |
| System RAM | 78 GB DDR |
| GPU.0 | Intel UHD Graphics 770 (iGPU) — 78.5 GB shared from system RAM |
| GPU.1 | Intel Arc B60 (dGPU) — **24 GB dedicated VRAM** |
| Disk | 938 GB SSD, ~489 GB free |
| Network | LAN, accessible as `EnvyStorm.local:11435` / `192.168.0.136:11435` |

**GPU assignment:**
- `GPU.0` runs the embedding model (multilingual-e5-large) — frees GPU.1 headroom
- `GPU.1` runs all LLMs, VLMs, SDXL image generation, and Whisper STT
- No cloud, no internet dependency for inference

---

## 2. Software Stack

| Layer | Package | Version |
|---|---|---|
| Inference runtime | OpenVINO | 2026.1.0 |
| LLM/VLM/SDXL/STT pipeline | openvino-genai | 2026.1.0.0 |
| Model conversion / embedding loader | optimum-intel | 1.27.0 |
| HTTP server | FastAPI + Uvicorn | 0.128.8 / 0.45.0 |
| Python | CPython | 3.12 |
| Metrics storage | PostgreSQL | `ov_metrics` database |
| Web search backend | SearxNG | self-hosted (Docker) |
| Venv | `/home/jerzy/ov_env` | isolated from system Python |

---

## 3. API Endpoints

The server speaks the **OpenAI REST API dialect** — any OpenAI-compatible client can point at it without modification.

### Core inference
| Method | Endpoint | Description |
|---|---|---|
| POST | `/v1/chat/completions` | Chat, streaming and non-streaming, tool calls, vision |
| POST | `/v1/embeddings` | Text embeddings, OpenAI-compatible |
| POST | `/v1/images/generations` | Image generation (SDXL), returns base64 PNG |
| POST | `/v1/audio/transcriptions` | Speech-to-text (Whisper), multipart form |
| GET | `/v1/models` | Catalogue of available models (local + OVH remote) |

### Administration
| Method | Endpoint | Description |
|---|---|---|
| POST | `/admin/profile` | Switch inference profile (fast / precise / laborious) |
| GET / POST | `/admin/profile-models` | VRAM profiler status and trigger |
| POST | `/admin/scope` | Set routing scope: local / ovh / all |
| POST | `/maintenance/restart` | Graceful server restart |

### Observability
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Full server health, loaded models, VRAM, routing stats |
| GET | `/version` | Server version + git commit |
| GET | `/metrics/events` | Per-request event log (last N, since timestamp) |
| GET | `/metrics/summary` | Aggregate stats from PostgreSQL |
| GET | `/monitor/api/system` | Live GPU temperature, power, fan, CPU, RAM |
| GET | `/monitor/api/metrics` | Stub — Postgres time-series for charts (Phase 4) |

### Web monitor
Served as a Svelte SPA at `http://EnvyStorm.local:11435/monitor` — shows VRAM bar, GPU/system stats, profile switcher, model profiler, catalogue view, and routing decision detail.

---

## 4. Installed Models

| Model | Type | Disk | VRAM (approx) | Role |
|---|---|---|---|---|
| `qwen3-14b-int4-ov` | LLM INT4 | 7.9 GB | 9.1 GB | Default general / reasoning / document |
| `qwen3-8b-int4-ov` | LLM INT4 | 4.6 GB | 4.6 GB | Agent turns / tool selection |
| `qwen3-coder-30b-a3b-int4-ov` | LLM MoE INT4 | 16 GB | ~7 GB (3B active) | Best local code model |
| `qwen2.5-coder-14b-int4` | LLM INT4 | 7.9 GB | ~9 GB | Code / SQL (fast tier) |
| `mistral-small-3.2-24b-int4-ov` | LLM INT4 | 12 GB | ~14 GB | Balanced: code + general + tool calls |
| `phi-4-int4-ov` | LLM INT4 | 7.6 GB | ~8 GB | Available, not in default routing |
| `qwen2.5-vl-7b-int4-ov` | VLM INT4 | 4.9 GB | 5–6 GB | Vision — image understanding (fast) |
| `internvl2.5-8b-int4-ov` | VLM INT4 | 4.5 GB | ~5 GB | Vision — image understanding (best) |
| `multilingual-e5-large-int8` | Embedding INT8 | 1.7 GB | 563 MB (GPU.0) | Semantic embeddings / routing |
| `whisper-large-v3-int8-ov` | STT INT8 | 1.5 GB | ~1.5 GB (GPU.1) | Speech-to-text, multilingual |
| `sdxl-fp16-ov` | Image gen FP16 | 6.5 GB | ~6 GB (GPU.1) | SDXL text-to-image |

**VRAM budget (GPU.1, 24 GB):**
- Max 2 LLMs loaded simultaneously; LRU eviction when a third is needed
- Typical warm state: qwen3-14b + qwen3-8b ≈ 13.7 GB → leaves ~10 GB headroom
- Image gen and Whisper load on demand, unloaded when idle
- 1.5 GB headroom reserved (`vram_headroom_gb`) as guard

---

## 5. Inference Capabilities

### 5.1 Text completion
- OpenAI `/v1/chat/completions` — full message history, system prompt, user/assistant/tool roles
- Streaming (SSE `text/event-stream`) and non-streaming JSON
- `max_tokens`, `temperature`, `stream` parameters honoured
- Prefix caching enabled — warm context prefix reuse across turns

### 5.2 Extended thinking
Available in **precise** and **laborious** profiles. Models emit `<think>...</think>` blocks; the server strips them from the final response content and surfaces `thinking` separately in the response object.

### 5.3 Tool calling / function calling
- Qwen3 format: `<tool_call>{"name":…,"arguments":…}</tool_call>` — parsed and emitted as OpenAI `tool_calls` array
- Mistral format: `[TOOL_CALLS]` + JSON array — dual-parser handles both
- `finish_reason: "tool_calls"` returned when tool call detected
- Tool schemas injected into system prompt (not native ov_genai support — handled via prompt engineering)
- **Limitation:** VLMs (Qwen2.5-VL, InternVL) do not support tool calls yet

### 5.4 Vision / multimodal
- Accepts OpenAI vision API format: `image_url` content parts (base64 `data:` URIs or HTTP URLs)
- Images resized to max 1280px longest side before encoding
- KV-cache growth bounded: 1280px → ~2090 image tokens vs ~6760 at 2560px
- Image history limited to 1 turn (configurable) to prevent VRAM growth across multi-turn chats
- Two VLMs available: Qwen2.5-VL-7B (fast) and InternVL2.5-8B (best)

### 5.5 Embeddings
- `POST /v1/embeddings` — mean-pooled, L2-normalised 1024-dim vectors
- Model: multilingual-e5-large-int8 (supports 100+ languages)
- Runs on GPU.0 — does not compete with GPU.1 LLM inference

### 5.6 Image generation
- `POST /v1/images/generations` — OpenAI-compatible request
- Model: SDXL FP16, 1024×1024 default, configurable size, 20 denoising steps
- Returns base64 PNG in `b64_json` format
- Loads on first request, unloads when idle to free GPU.1 VRAM

### 5.7 Speech-to-text
- `POST /v1/audio/transcriptions` — OpenAI-compatible multipart form
- Model: Whisper large-v3 INT8, supports ~100 languages
- Input: WAV, MP3, FLAC (16 kHz target; server resamples if needed)
- Returns `{"text": "..."}` JSON
- Resident in VRAM after first load (~1.5 GB)

---

## 6. Intelligent Routing

The routing layer selects the best model for each request automatically. Priority order:

1. **Explicit directive** — user adds `#code`, `#document`, or `#general` to the message → forces that task class
2. **Binary signal — image** — message contains `image_url` → routed to VLM
3. **Binary signal — tools** — request includes tool definitions → web_search class
4. **Keyword detection** — "search for", "latest news", "wyszukaj" etc. → web_search class
5. **Long context** — estimated prompt > 4000 tokens → document class
6. **Embedding similarity** — prompt semantically matched against task-class examples using multilingual-e5-large (min confidence: 0.72)
7. **Default** — general class

Within a task class, model tier is selected by active profile (fast → fastest, precise → balanced, laborious → best).

### Task classes and their model pools

| Class | Trigger | Local models | Cloud fallback |
|---|---|---|---|
| `vision` | image in request | Qwen2.5-VL-7B (fast), InternVL2.5-8B (best) | — |
| `code` | #code directive or routing | Qwen2.5-Coder-14B (fast), Mistral-24B (balanced), Qwen3-Coder-30B-MoE (best) | Qwen3-Coder-30B-A3B (OVH) |
| `document` | long context / #document | Qwen3-8B (fast), Qwen3-14B (balanced), Mistral-24B (best) | Qwen3-32B, gpt-oss-120B (OVH) |
| `general` | default / #general | Qwen3-8B (fast), Qwen3-14B (balanced), Mistral-24B (best) | Qwen3-32B (OVH) |
| `web_search` | tool definitions / keywords | Qwen3-8B, Qwen3-14B, Mistral-24B | Qwen3-32B (OVH) |

### Web search
SearxNG instance (Docker) integrated via tool-call loop. When the model generates a `search_web` tool call, the server executes it against SearxNG, injects results, and continues generation.

### OVH cloud backend
- Provider: OVH AI Endpoints (OpenAI-compatible proxy)
- Available models: Qwen3-32B, Qwen3-Coder-30B-A3B, gpt-oss-120B
- Activated when `provider_scope` is `ovh` or `all`, or when `laborious` profile routes to `best` tier and local capacity is insufficient
- Requires `OVH_API_KEY` environment variable
- Catalogue cached with 5-minute TTL

---

## 7. Inference Profiles

| Profile | Thinking | Max new tokens | Model tier | Use case |
|---|---|---|---|---|
| `fast` | off | 2048 | fastest | Quick answers, agent turns |
| `precise` | on | 4096 | balanced | Reasoned answers, analysis |
| `laborious` | on | 16384 | best | Deep research, long documents |

Profiles are switched via `POST /admin/profile` or persisted in `config.json`. Profile `max_new_tokens` is a **floor** — client cannot request fewer tokens than the profile sets, but can request more.

---

## 8. Observability & Operations

- **Per-request logging** with `X-Request-ID` correlation header
- **PostgreSQL** records every request: model, tokens, tok/sec, TTFB, elapsed, task class
- **SIGUSR1** toggles debug request body logging without restart
- **VRAM profiler** — `POST /admin/profile-models` measures live VRAM per model
- **Monitor sidecar** (`monitor_sidecar.py`) on `:11436` — reads GPU metrics from `/proc/fdinfo` and sysfs independently of the main server; also proxies `/health`
- **systemd service** `ov-server` — `Restart=always`, no sudo needed for restart (`kill -SIGTERM`)
- **Svelte web monitor** at `/monitor` — live VRAM bar, GPU/system stats, profiles, catalogue, routing detail

---

## 9. Current Limitations

### Hard constraints
| Limitation | Detail |
|---|---|
| **Single worker** | FastAPI runs single-process, single asyncio loop. One inference at a time; concurrent requests queue |
| **Max 2 LLMs in VRAM** | Third model triggers LRU eviction — cold load takes 30–90 s depending on model size |
| **GPU.1 shared** | LLMs, VLMs, SDXL, and Whisper all compete for the same 24 GB. Concurrent load of all four is not possible |
| **No batch inference** | `openvino_genai.LLMPipeline` processes one request at a time; `max_num_batched_tokens` is 4096 |
| **Context window cap** | Effective KV cache budget = 8 GB → ~28,000 tokens at INT4 precision for 14B model |
| **VLM image history** | Only the most recent image turn is re-encoded; older turns lose images (VRAM guard) |
| **Max image size** | 1280 px longest side (server resizes automatically) |
| **No TTS** | `POST /v1/audio/speech` not implemented yet |
| **VLM tool calls** | InternVL and Qwen2.5-VL adapters do not parse tool-call output — deferred |
| **No native tool-call schema injection** | Tools injected via system prompt text, not via openvino_genai API (no such API exists); works but is less robust than native handling |
| **Embedding routing threshold** | 0.72 similarity minimum — needs live traffic tuning; under-threshold falls to keyword/default |
| **No streaming for image gen / STT** | Both return synchronous responses only |
| **OVH gating** | Cloud fallback requires `OVH_API_KEY` env var; absent → local only, no automatic failover |

### Known minor issues
- VRAM bar in web monitor can read slightly over 100% (disk-size-based VRAM estimate, not live measurement — sidecar `/proc/fdinfo` path resolves this partially)
- `monitor/api/metrics` and `monitor/api/model-usage` return stub data — Postgres chart wiring (SVP Phase 4) not yet implemented

---

## 10. Planned Extensions

| Feature | Status | Notes |
|---|---|---|
| **SVP Phase 4** — Postgres time-series charts | Next up | uPlot charts wired to `request_log` — tok/sec, latency, TTFB history |
| **TTS endpoint** (`/v1/audio/speech`) | Planned | Kokoro-82M (ONNX→OV) or Piper (CPU); OpenAI-compatible |
| **Full voice agent** | Planned | Mic → Whisper → LLM → TTS loop; morning briefing mode with RSS news injection |
| **n8n AI Agent tool-call validation** | Pending | Tool-call loop end-to-end test with n8n AI Agent node |
| **InternVL tool calling** | Deferred | InternLM2 `<|action_start|><|plugin|>` format parser |
| **Embedding threshold tuning** | Pending | Requires traffic analysis against `request_log` |
| **Monitor VRAM bar fix** | Pending | Use sidecar `/proc/fdinfo` live reading instead of disk-size estimate |

---

## 11. Integration Notes

- **Drop-in OpenAI replacement:** Set `base_url` to `http://EnvyStorm.local:11435/v1`, `api_key` to any string — no auth
- **Model aliases:** Any client model name can be remapped in `config.json` (e.g., `"gpt-4o": "qwen3-14b-int4-ov"`)
- **Claude Code:** Configured to use this server as Anthropic proxy (`/v1/messages` layer removed — now via direct CC config)
- **AnythingLLM:** Uses `qwen2.5-coder:14b` alias, tool-call loop verified working
- **n8n:** AI Agent node integration tested (web search loop functional)
- **CORS:** `allow_origins=["*"]` — accessible from any LAN client without preflight issues

---

*Generated from live server state, config.json, MODELS.md, PROGRESS.md, and plans/ on 2026-05-13.*
