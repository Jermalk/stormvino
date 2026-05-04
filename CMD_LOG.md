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
