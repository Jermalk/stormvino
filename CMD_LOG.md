# CMD_LOG.md — command log

> Append-only. One entry per significant command.
> Format: `### YYYY-MM-DD — <description>` followed by fenced shell block.
> Do not read top-to-bottom — last entry is always the most recent.

---

### 2026-05-04 — Health check (baseline)

```bash
curl -s http://localhost:11435/health | python3 -m json.tool
```

### 2026-05-04 — List available models

```bash
curl -s http://localhost:11435/v1/models | python3 -m json.tool
```

### 2026-05-04 — Run all tests (from project root, ov_env active)

```bash
source /home/jerzy/ov_env/bin/activate && python3 -m pytest tests/ -v
```

### 2026-05-04 — Restart server after Steps 3-5 (run as user, then continue)

```bash
sudo systemctl restart ov_server && sleep 3 && curl -s http://localhost:11435/health | python3 -m json.tool
```

### 2026-05-04 — Step 4: Non-streaming /v1/messages smoke test

```bash
curl -s http://localhost:11435/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b-int4-ov",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "Reply with one word: hello"}]
  }' | python3 -m json.tool
# Expect: {"id":"msg_...","type":"message","role":"assistant","stop_reason":"end_turn","content":[{"type":"text","text":"..."}],"usage":{...}}
```

### 2026-05-04 — Step 4: Verify active_requests returns to 0 after non-streaming call

```bash
curl -s http://localhost:11435/health | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['active_requests']==0, f'LEAK: {d[\"active_requests\"]}'; print('OK — active_requests=0')"
```

### 2026-05-04 — Step 4: Streaming /v1/messages smoke test (SSE event sequence)

```bash
curl -sN http://localhost:11435/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b-int4-ov",
    "max_tokens": 64,
    "stream": true,
    "messages": [{"role": "user", "content": "Reply with one word: hello"}]
  }'
# Expect SSE lines: event: message_start / event: content_block_start / event: ping /
#   event: content_block_delta (×N) / event: content_block_stop /
#   event: message_delta (stop_reason: end_turn) / event: message_stop
```

### 2026-05-04 — Step 5: count_tokens smoke test

```bash
curl -s http://localhost:11435/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b-int4-ov",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello world"}]
  }' | python3 -m json.tool
# Expect: {"input_tokens": N} where N > 0
```

### 2026-05-04 — Phase 1 gate: point Claude Code at local server

```bash
export ANTHROPIC_BASE_URL=http://localhost:11435
export ANTHROPIC_API_KEY=local
claude --version   # confirm it starts; run one prompt to verify round-trip
```

### 2026-05-04 — F1: Test scheduler_config kwarg vs config-dict forms

```bash
source /home/jerzy/ov_env/bin/activate && python3 -c "
import openvino_genai as ov_genai
sc = ov_genai.SchedulerConfig(); sc.cache_size = 4
try: ov_genai.LLMPipeline('/nonexistent', 'CPU', scheduler_config=sc)
except Exception as e: print('kwarg:', type(e).__name__, str(e)[:80])
try: ov_genai.LLMPipeline('/nonexistent', 'CPU', {'scheduler_config': sc})
except Exception as e: print('dict:', type(e).__name__, str(e)[:80])
"
# Result: kwarg form is correct (no deprecation). Dict form deprecated.
```
