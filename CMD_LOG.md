# CMD_LOG.md — command log

> Append-only. One entry per significant command.
> Format: `### YYYY-MM-DD — <description>` followed by fenced shell block.
> Do not read top-to-bottom — last entry is always the most recent.
> All curl commands are single-line bash.

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

### 2026-05-04 — Restart server

```bash
sudo systemctl restart ov-server && sleep 8 && curl -s http://localhost:11435/health | python3 -m json.tool
```

### 2026-05-04 — F1: Test scheduler_config kwarg vs config-dict forms

```bash
source /home/jerzy/ov_env/bin/activate && python3 -c "import openvino_genai as ov_genai; sc = ov_genai.SchedulerConfig(); sc.cache_size = 4; [print(f, type(e).__name__, str(e)[:80]) for f, e in [('kwarg', __import__('contextlib').suppress(None))] ]"
```

### 2026-05-04 — Step 4: Non-streaming /v1/messages smoke test

```bash
curl -s http://localhost:11435/v1/messages -H "Content-Type: application/json" -d '{"model":"qwen3-14b-int4-ov","max_tokens":64,"messages":[{"role":"user","content":"Reply with one word: hello"}]}' | python3 -m json.tool
```

### 2026-05-04 — Step 4: Verify active_requests returns to 0

```bash
curl -s http://localhost:11435/health | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['active_requests']==0, f'LEAK: {d[\"active_requests\"]}'; print('OK — active_requests=0')"
```

### 2026-05-04 — Step 4: Streaming /v1/messages smoke test

```bash
curl -sN http://localhost:11435/v1/messages -H "Content-Type: application/json" -d '{"model":"qwen3-14b-int4-ov","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"Reply with one word: hello"}]}'
```

### 2026-05-04 — Step 5: count_tokens smoke test

```bash
curl -s http://localhost:11435/v1/messages/count_tokens -H "Content-Type: application/json" -d '{"model":"qwen3-14b-int4-ov","max_tokens":1024,"messages":[{"role":"user","content":"Hello world"}]}' | python3 -m json.tool
```

### 2026-05-04 — Restart after streaming bug fix (create_task coroutine fix)

```bash
sudo systemctl restart ov-server && sleep 8 && curl -s http://localhost:11435/health | python3 -m json.tool
```

### 2026-05-04 — Streaming retest after fix

```bash
curl -sN --max-time 60 http://localhost:11435/v1/messages -H "Content-Type: application/json" -d '{"model":"qwen3-8b-int4-ov","max_tokens":32,"stream":true,"thinking":false,"messages":[{"role":"user","content":"Say: hi"}]}'
```

### 2026-05-04 — Phase 1 gate: point Claude Code at local server

```bash
ANTHROPIC_BASE_URL=http://localhost:11435 ANTHROPIC_API_KEY=local claude
```

### 2026-05-04 — Check debug_logging state
```sh
grep -n "debug_logging" /opt/ov_server/ov_server.py | head -5
```

### 2026-05-04 — Enable debug logging via SIGUSR1
```sh
kill -USR1 $(systemctl show ov-server --property=MainPID --value)
```

### 2026-05-04 — Watch live server log (agentic request capture)
```sh
journalctl -u ov-server -f --output=cat
```

### 2026-05-04 — Run tests after agent-stream fix
```sh
source /home/jerzy/ov_env/bin/activate && python -m pytest tests/ -q
```

### 2026-05-04 — Restart service + health check
```sh
systemctl restart ov-server
sleep 5 && curl -s http://localhost:11435/health | python3 -m json.tool
```

### 2026-05-04 — Restart + re-enable debug logging
```sh
systemctl restart ov-server
sleep 4 && kill -USR1 $(systemctl show ov-server --property=MainPID --value)
```

### 2026-05-04 — Restart + re-enable debug after _extract_agent_json fix
```sh
systemctl restart ov-server
sleep 4 && kill -USR1 $(systemctl show ov-server --property=MainPID --value)
```
