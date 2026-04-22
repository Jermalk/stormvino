# Model Management

How to convert HuggingFace models to OpenVINO format and add them to ov_server.

---

## Directory layout

```
/opt/ov_server/
├── models/
│   ├── qwen3-14b-int4/          ← LLM (has generation_config.json)
│   ├── qwen3-8b-int4/
│   ├── qwen2.5-3b-int4/
│   └── multilingual-e5-large-int8/   ← embedding (no generation_config.json)
├── config.json
└── ov_server.py
```

`models/` is excluded from git — model binaries are large and machine-specific.

The server discovers LLMs at startup by scanning `models_dir` (set in `config.json`).
A directory is classified as an LLM if it contains **both**:
- `openvino_model.xml`
- `generation_config.json`

Directories with `openvino_model.xml` but **without** `generation_config.json` are
treated as embedding models and are not auto-discovered (they are loaded on demand
via the `embedding_model` config key).

---

## Environment requirements

| Package | Version used | Install |
|---|---|---|
| `openvino` | 2026.1.0 | `pip install openvino` |
| `openvino-genai` | 2026.1.0.0 | `pip install openvino-genai` |
| `optimum-intel` | 1.27.0 | `pip install optimum-intel[openvino]` |
| `transformers` | latest stable | pulled by optimum-intel |

`optimum-cli` lands in `~/.local/bin/` — add to PATH or call with full path:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

HuggingFace models download to `~/.cache/huggingface/hub/` by default.
Set `HF_HOME` to redirect if disk is limited:

```bash
export HF_HOME=/mnt/data/hf_cache
```

---

## Conversion: LLM (INT4)

INT4 is the sweet spot for this GPU (24 GB VRAM). Smaller than INT8, quality close
enough for chat and agent use. Use INT8 only for tasks where accuracy matters more
than VRAM footprint (see tradeoffs below).

```bash
~/.local/bin/optimum-cli export openvino \
  --model Qwen/Qwen3-14B \
  --task text-generation-with-past \
  --weight-format int4 \
  --group-size 128 \
  --ratio 1.0 \
  --sym \
  /opt/ov_server/models/qwen3-14b-int4
```

**Key flags:**

| Flag | What it does |
|---|---|
| `--task text-generation-with-past` | Exports with KV-cache (required for autoregressive generation) |
| `--weight-format int4` | 4-bit weight quantisation |
| `--group-size 128` | Quantisation granularity — 128 is standard; smaller = more accurate, larger = smaller file |
| `--ratio 1.0` | Fraction of layers quantised to INT4 (1.0 = all; lower = fewer sensitive layers quantised) |
| `--sym` | Symmetric quantisation — slightly faster on Intel hardware |
| `--backup-precision int8_asym` | Optional: fall back INT8 for layers that degrade badly at INT4 |

**With accuracy-recovery (slower, better quality):**

```bash
~/.local/bin/optimum-cli export openvino \
  --model Qwen/Qwen3-14B \
  --task text-generation-with-past \
  --weight-format int4 \
  --group-size 128 \
  --ratio 0.8 \
  --sym \
  --backup-precision int8_asym \
  --dataset wikitext2 \
  --num-samples 128 \
  /opt/ov_server/models/qwen3-14b-int4-accurate
```

This takes significantly longer (calibration dataset pass) but recovers quality on
perplexity-sensitive tasks. Not needed for typical chat workloads.

---

## Conversion: LLM (INT8)

Use INT8 when the model will run quantitative tasks (code, math, structured JSON)
where INT4 rounding artifacts are noticeable, and VRAM budget allows.

```bash
~/.local/bin/optimum-cli export openvino \
  --model Qwen/Qwen3-8B \
  --task text-generation-with-past \
  --weight-format int8 \
  /opt/ov_server/models/qwen3-8b-int8
```

INT8 approximately doubles the VRAM requirement vs INT4 for the same parameter count.
Rule of thumb: INT4 14B ≈ 9 GB, INT8 14B ≈ 16 GB.

---

## Conversion: embedding model

Embedding models use `--task feature-extraction` and are loaded via
`OVModelForFeatureExtraction` (not `LLMPipeline`).

```bash
~/.local/bin/optimum-cli export openvino \
  --model intfloat/multilingual-e5-large \
  --task feature-extraction \
  --weight-format int8 \
  /opt/ov_server/models/multilingual-e5-large-int8
```

INT8 is appropriate for embedding models — the output vectors are robust to weight
quantisation, and INT4 offers no meaningful VRAM saving at this scale.

---

## INT4 vs INT8 — decision guide

| | INT4 | INT8 |
|---|---|---|
| VRAM (14B model) | ~9 GB | ~16 GB |
| Quality (chat) | Good | Excellent |
| Quality (math/code) | Acceptable | Better |
| Conversion time | Fast | Fast |
| With calibration | Slower, closes gap | N/A |
| Recommended for | Agent/chat on 24 GB GPU | Precision-critical tasks |

Current GPU (Intel Arc B60) has 24 GB. Comfortable budget:
- Two INT4 models loaded simultaneously: ~9 + ~5 = 14 GB → fits with headroom
- One INT8 14B + one INT4 3B: ~16 + ~2 = 18 GB → fits, less headroom

---

## Adding a converted model to the server

1. Drop the converted directory into `/opt/ov_server/models/`:

   ```bash
   mv /path/to/conversion/output /opt/ov_server/models/new-model-name
   ```

2. Verify discovery conditions are met:

   ```bash
   ls /opt/ov_server/models/new-model-name/openvino_model.xml
   ls /opt/ov_server/models/new-model-name/generation_config.json
   ```

3. Optionally update `config.json` to set it as default or agent model:

   ```json
   "default_model": "new-model-name"
   ```

4. Restart the server:

   ```bash
   sudo systemctl restart ov-server
   ```

5. Confirm the model appears in `/v1/models`:

   ```bash
   curl -s http://localhost:11435/v1/models | python3 -m json.tool
   ```

No code changes required — discovery is automatic.

---

## Adding a model alias

If a client sends a name that doesn't match the directory name (e.g., AnythingLLM
sends `qwen2.5-coder:14b` regardless of local config), add an alias to `config.json`:

```json
"model_aliases": {
    "qwen2.5-coder:14b": "qwen3-14b-int4",
    "gpt-4o":            "qwen3-14b-int4"
}
```

Aliases are resolved silently — the server logs the resolved name at `INFO` level.

---

## Currently installed models

| Directory | Type | VRAM (approx) | Role |
|---|---|---|---|
| `qwen2.5-coder-14b-int4` | LLM INT4 | ~9 GB | default — code/SQL/database analysis |
| `qwen3-14b-int4` | LLM INT4 | 9.1 GB | general chat / reasoning |
| `qwen3-8b-int4` | LLM INT4 | 4.6 GB | agent / tool selection |
| `qwen2.5-3b-int4` | LLM INT4 | 1.7 GB | lightweight fallback |
| `multilingual-e5-large-int8` | embedding INT8 | 563 MB | `/v1/embeddings` |

---

## Removing a model

Stop using it in `config.json` (remove references from `default_model`, `agent_model`,
`model_aliases`), then delete the directory:

```bash
rm -rf /opt/ov_server/models/old-model-name
```

The server will no longer discover it on next restart. No code changes needed.
