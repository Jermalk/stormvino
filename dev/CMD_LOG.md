# CMD_LOG.md — command log

> Append-only. One entry per significant command.
> Format: `### YYYY-MM-DD — <description>` followed by fenced shell block.
> Do not read top-to-bottom — last entry is always the most recent.
> All curl commands are single-line bash.

---
### 2026-05-06 — model download

```  source /home/jerzy/ov_env/bin/activate 
     && python3 -c "from huggingface_hub import snapshot_download; snapshot_download('OpenVINO/Qwen3-30B-A3B-Instruct-2507-int4-ov', local_dir='/opt/ov_server/models/qwen3-30b-a3b-int4-ov')"  
```

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

## 2026-05-05 — VRAM eviction bug fixes
```
# Bugs fixed:
# 1. _evict_lru(): added gc.collect() so LLMPipeline destructor runs before next vram_free_gb() query
# 2. get_model() soft cap: if → while + re-query after each eviction
# 3. get_vlm() inline VLM eviction: added gc.collect()
# 4. get_vlm() LLM eviction: if → while + re-query after each eviction
python -m py_compile /opt/ov_server/ov_server.py  # OK
systemctl restart ov-server
curl -s http://localhost:11435/health | python3 -m json.tool
```

## 2026-05-05 — Model preloading
```
# Added: startup preload of AGENT_MODEL (qwen3-8b) + speculative preload of
# DEFAULT_MODEL (qwen3-14b) on tool_calls detection in both streaming/non-streaming paths.
# _warm_model() helper is fire-and-forget; exceptions logged, never raised.
python -m py_compile /opt/ov_server/ov_server.py  # OK
systemctl restart ov-server
# Watch startup preload in logs:
journalctl -u ov-server -f
```

## 2026-05-05 — Session 11: fix /v1/messages hang

```bash
# Confirm /v1/chat/completions works (baseline)
curl -s -N http://localhost:11435/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3-14b-int4-ov","messages":[{"role":"user","content":"Say hello."}],"stream":true,"max_tokens":20}'

# Reproduce hang — non-streaming /v1/messages
timeout 90 curl -s http://localhost:11435/v1/messages -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Say hello."}],"max_tokens":20,"stream":false}'

# After fix — verify both paths
curl -s http://localhost:11435/v1/messages -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Reply with exactly one word: hi."}],"max_tokens":10,"stream":false}'
curl -s -N http://localhost:11435/v1/messages -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"Reply with exactly one word: hi."}],"max_tokens":10,"stream":true}'
```

## 2026-05-06 — Session 12: CC latency reduction

```bash
# Measure actual input token count from real CC request
journalctl -u ov-server -n 20 --no-pager | grep "input tokens"
# Result: 53881 tokens with full schemas, 26861 after schema stripping

# Confirm prefix caching works (controlled test, same 3K-token prefix)
time curl -s http://localhost:11435/v1/messages \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3-14b-int4-ov\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"system\":\"$SYSTEM\",\"max_tokens\":10,\"stream\":false}"
# Cold: 39.5s, Warm (identical): 0.4s → 100x speedup confirmed
```

### 2026-05-07 — Test suite bootstrap (65 tests, all pass)
```
source /home/jerzy/ov_env/bin/activate && python -m pytest
# 65 passed, 2 warnings in 0.30s

pip install pytest-watch --quiet

make test       # run once
make watch      # auto-rerun on .py changes (ptw)
```

### 2026-05-07 — Step 1.1 commit
```
git add config.json ov_server.py tests/conftest.py tests/test_pure.py pytest.ini Makefile
git commit  # cd8d32a — feat: Step 1.1 — new config schema, _validate_config, 79-test suite
```

### 2026-05-08 — Step 2.4: tests + commit
```
make test          # 170/170 pass
git add ov_server.py SESSION.md
git commit         # b0e6792
```

### 2026-05-08 — Step 2.5: routing confidence/latency + ov-monitor
```
make test          # 170/170 pass
git commit         # 15db100 (ov_server), 8e4d961 (ov_monitor)
```

### 2026-05-08 — Step 2.6: streaming fixes
```
python3 -m pytest tests/ -q    # 176/176 pass
git add ov_server.py tests/test_pure.py
git commit                      # 76b31ff
```

### 2026-05-08 — Step 3.1: assessor bootstrap
```
python3 -m pytest tests/ -q    # 176/176 pass
git add ov_server.py
git commit                      # 1fcc1c4
```

### 2026-05-08 — Step 3.2: assessor routing prompt
```
python3 -m pytest tests/ -q    # 186/186 pass
git add ov_server.py tests/test_pure.py
git commit                      # d9780f5
```

### 2026-05-08 — Step 3.3: assessor wired into routing
```
python3 -m pytest tests/ -q    # 186/186 pass
git add ov_server.py
git commit                      # 01bc6bd
```

## 2026-05-08 12:19 — Drop 30B local models, fix assessor
```
# Removed from config.json: qwen3-8b-int4-ov (broken), qwen3-30b-a3b-int4-ov (broken), qwen3-coder-30b-a3b-int4-ov
# assessor.model: qwen3-14b-int4-ov (confirmed working, pipe reused for general/web_search/document)
```

## 2026-05-14 — responsive polish + GPU bars
```
npm run build
```
