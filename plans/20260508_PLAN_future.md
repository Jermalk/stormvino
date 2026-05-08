# FUTURE_PLAN.md — Local Voice Conversation Agent

> Proposal only — nothing here is scheduled. User decides before any step is executed.
> Architecture builds on the existing ov_server stack; all new features are additive.

---

## Goal

A fully local, interactive voice conversation agent:

```
Microphone → STT → LLM (ov_server) → TTS → Speaker
                         ↑
                   News context (RSS + scraper)
```

Morning briefing mode: scrape headlines → summarize → inject into context →
warm prefix cache → enter voice Q&A loop. No cloud. No latency spikes.

---

## Component Map

| Component | Technology | Device | Endpoint |
|---|---|---|---|
| STT | Whisper (small or large-v3-turbo), OpenVINO | GPU.1 | `POST /v1/audio/transcriptions` |
| LLM | qwen3-14b-int4-ov (existing) | GPU.1 | `POST /v1/messages` |
| TTS | Kokoro-82M or Piper | CPU | `POST /v1/audio/speech` |
| News | RSS + trafilatura | CPU | `POST /v1/news/refresh`, `GET /v1/news/context` |
| Client | Python script (sounddevice + requests) | local | — |

---

## Phase 1 — STT: `/v1/audio/transcriptions`

**Model:** `openai/whisper-small` or `openai/whisper-large-v3-turbo` converted to OpenVINO IR via `optimum-intel`.

**Inference class:** `optimum.intel.OVModelForSpeechSeq2Seq`

**Convert:**
```bash
optimum-cli export openvino \
  --model openai/whisper-large-v3-turbo \
  --task automatic-speech-recognition \
  /opt/ov_server/models/whisper-large-v3-turbo-ov
```

**Request format** (OpenAI-compatible multipart):
```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
  file=<audio.wav>
  model=whisper-large-v3-turbo
  language=en          (optional)
  response_format=json (default)
```

**Response:**
```json
{ "text": "transcribed text here" }
```

**Implementation notes:**
- Load model at startup alongside LLM; fits on GPU.1 (Whisper large-v3-turbo ≈ 1.5 GB IR)
- Use `processor.feature_extractor` → float32 mel features → `model.generate()`
- Accept 16 kHz WAV/MP3/FLAC via `soundfile` or `ffmpeg` pipe
- Keep it blocking (short audio ≤ 30 s); no streaming needed for STT

**Config additions to `config.json`:**
```json
"stt_model": "whisper-large-v3-turbo-ov",
"stt_device": "GPU.1"
```

---

## Phase 2 — TTS: `/v1/audio/speech`

**Model options (in preference order):**

| Model | Size | Quality | License | Notes |
|---|---|---|---|---|
| Kokoro-82M | 82 M params | Very good | Apache 2.0 | ONNX available; needs ONNX→OV or run via onnxruntime |
| Piper | ~30 MB per voice | Good | MIT | C++ binary + Python bindings; fast CPU inference |
| MMS-TTS | ~300 MB | Decent | CC-BY-NC | Fairseq; harder to port |

**Recommendation:** Start with Piper on CPU (2–4× realtime, zero GPU contention). Add Kokoro-82M as upgrade path if Piper quality is insufficient.

**Request format** (OpenAI-compatible):
```
POST /v1/audio/speech
Content-Type: application/json
{
  "model": "piper",
  "input": "text to speak",
  "voice": "en_US-lessac-medium",
  "response_format": "wav"    (wav | mp3 | opus)
}
```

**Response:** binary audio stream (`Content-Type: audio/wav`)

**Implementation notes:**
- Piper: subprocess call to `piper --model <voice.onnx> --output_file -` piped to stdout
- Kokoro: load ONNX model via `onnxruntime`; phonemize with `espeak-ng`
- Run on CPU to leave GPU free for simultaneous LLM generation
- Response as `StreamingResponse` so client can start playing before full synthesis

**Config additions:**
```json
"tts_model": "piper",
"tts_voice": "en_US-lessac-medium",
"tts_device": "CPU"
```

---

## Phase 3 — News Scraper

**Stack:** `feedparser` (RSS) + `trafilatura` (article extraction) + APScheduler or asyncio background task.

**Endpoints:**
```
POST /v1/news/refresh          — trigger immediate scrape
GET  /v1/news/context          — return current news digest as plain text
```

**Feed list:** configurable in `config.json`:
```json
"news_feeds": [
  "https://feeds.bbci.co.uk/news/rss.xml",
  "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"
]
```

**Flow:**
1. Background task fires every N minutes (configurable, default 60)
2. `feedparser.parse(url)` → list of entries
3. `trafilatura.fetch_url(entry.link)` + `trafilatura.extract()` → clean article text
4. Store in-memory: `List[Dict]` with `{title, url, published, body, summary}`
5. On `/v1/news/context`: join last N articles into a text block, truncate to `max_news_tokens`
6. On `/v1/messages` with `inject_news=true` header or model hint: prepend news context to system prompt before sending to LLM

**Morning briefing sequence (client-driven):**
```
POST /v1/news/refresh          → wait for scrape
GET  /v1/news/context          → get digest text
POST /v1/messages (summarize)  → get LLM summary
POST /v1/audio/speech          → convert summary to audio
play audio                     → user hears briefing
→ enter voice Q&A loop
```

---

## Phase 4 — Python Client

**File:** `voice_client.py` (standalone script, not part of ov_server)

**Dependencies:** `sounddevice`, `soundfile`, `requests`, `numpy`

**Loop:**
```python
while True:
    audio = record_until_silence()          # VAD via RMS threshold
    text  = transcribe(audio)               # POST /v1/audio/transcriptions
    reply = chat(text, history)             # POST /v1/messages (streaming)
    speech = synthesize(reply)              # POST /v1/audio/speech
    play(speech)
    history.append(...)                     # rolling window, respect context limit
```

**VAD:** simple energy-based (RMS < threshold for 1.5 s → stop recording). Upgrade to Silero VAD if false triggers are a problem.

**Config file:** `~/.voice_agent.json`
```json
{
  "server": "http://localhost:11435",
  "model": "claude-sonnet-4-6",
  "history_turns": 10,
  "silence_threshold": 0.01,
  "silence_duration": 1.5
}
```

---

## Implementation Order

```
Phase 1 (STT)   →  Phase 2 (TTS)  →  Phase 3 (news)  →  Phase 4 (client)
   ~1 session       ~1 session         ~1 session           ~1 session
```

Each phase is independently testable via curl before the next begins.

**Phase 1 smoke test:**
```bash
curl -s -X POST http://localhost:11435/v1/audio/transcriptions \
  -F "file=@test.wav" -F "model=whisper-large-v3-turbo" | python3 -m json.tool
```

**Phase 2 smoke test:**
```bash
curl -s -X POST http://localhost:11435/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"piper","input":"Hello, this is a test.","voice":"en_US-lessac-medium"}' \
  --output test_out.wav && aplay test_out.wav
```

---

## VRAM / Resource Budget (B60, 24 GB)

| Component | VRAM | RAM | CPU |
|---|---|---|---|
| qwen3-14b-int4-ov | ~9 GB | ~2 GB | 0% (GPU) |
| KV cache | 3–12 GB (profile) | — | — |
| Whisper large-v3-turbo | ~1.5 GB | ~0.5 GB | 0% (GPU) |
| Piper TTS | 0 GB | ~200 MB | <5% |
| News scraper | 0 GB | ~100 MB | <1% (idle) |
| **Total (speed profile)** | ~14 GB | ~3 GB | — |
| **Total (document profile)** | ~23 GB | ~3 GB | — |

Whisper and LLM can share GPU.1 as long as requests are serialized — the existing executor handles this naturally.

---

## Open Questions

1. **Kokoro-82M on OpenVINO**: ONNX export path untested on this stack. Fallback is Piper.
2. **Whisper device conflict**: if STT and LLM requests arrive simultaneously, one blocks. Accept or add a second queue?
3. **News inject strategy**: header-based or always-on for a "morning briefing" system prompt profile?
4. **Wake word**: always-listening requires a wake-word detector (e.g. openWakeWord). Out of scope for v1 — push-to-talk is simpler.
5. **Streaming TTS**: synthesize in chunks as LLM streams tokens? Reduces end-to-end latency but complicates client buffering.
