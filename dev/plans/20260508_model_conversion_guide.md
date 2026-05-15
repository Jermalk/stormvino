# Model Conversion Guide — OV 2026.1.0

## What we learned the hard way (2026-05-08)

### Mistake 1: Wrong `--task` flag → stateless model

`--task text-generation` exports a stateless model. `openvino_genai.LLMPipeline` uses
paged attention (stateful KV cache), so it rejects the model at load time:

```
Check '!model->get_variables().empty()' failed at sdpa_to_paged_attention.cpp:51:
Model is supposed to be stateful, cannot perform the SDPAToPagedAttention transformation.
```

**Always use `--task text-generation-with-past`.**

### Mistake 2: Old IR + `KV_CACHE_PRECISION=u8` → crash on fresh compile

Locally-exported or internally-sourced models (e.g. from `/nfs/ov-share-01/...`) may have
IR that was generated with an older OV toolchain. OV 2026.1.0 added stricter static-type
validation for KV cache tensors. Fresh compilation fails:

```
Check 'm_element_type.is_static()' failed at src/inference/src/dev/make_tensor.cpp:58
```

This is silent if a compiled blob is already cached — the blob is loaded without
recompiling the IR. The bug surfaces only when:
- The blob is invalidated (KV size change, device driver update, etc.)
- The server is restarted on a clean cache

**Fix:** Either re-export with the current OV toolchain, or remove `KV_CACHE_PRECISION=u8`
from the server config. Default KV precision (f16) is compatible with all IR versions.

### Mistake 3: Dependency hell in production venv

`optimum-cli` in the production venv (`/home/jerzy/ov_env`) may have version conflicts
(e.g. `huggingface-hub==1.4.1` vs `transformers` requiring `<1.0`). Never run conversion
inside the production venv — always use a throw-away environment.

---

## Correct conversion command (OV 2026.1.0)

```bash
# Create a clean conversion venv (once per machine, reusable)
python3 -m venv /tmp/convert_env
source /tmp/convert_env/bin/activate
pip install -q "optimum[openvino,nncf]" transformers torch

# Convert (weights cached after first run — subsequent conversions reuse cache)
HF_HUB_OFFLINE=1 optimum-cli export openvino \
    -m <HuggingFace-model-id> \
    --task text-generation-with-past \
    --weight-format int4 \
    --trust-remote-code \
    /opt/ov_server/models/<output-dir>

deactivate
```

If HF Hub is unreachable mid-run (connection drop during task inference), add
`--task text-generation-with-past` explicitly and `HF_HUB_OFFLINE=1` to use cached weights.

---

## Checklist before adding a new model to routing

- [ ] Exported with `--task text-generation-with-past`
- [ ] Output dir contains `openvino_model.xml`, `openvino_model.bin`, `openvino_tokenizer.xml`
- [ ] Test load: `curl .../v1/chat/completions` with model name — no error in response
- [ ] Check journalctl for `m_element_type.is_static()` — indicates IR incompatibility
- [ ] Add to `config.json` task_classes with correct `tier` and optional `max_context_tokens`
- [ ] If VRAM is tight, add to `model_kv_overrides` to cap KV allocation

---

## OV HuggingFace org availability (as of 2026-05-08)

| Model          | int4 | int8 | fp16 |
|----------------|------|------|------|
| Qwen3-8B       | ✓    | ✓    | ✓    |
| Qwen3-14B      | ✗    | ✓    | ✓    |
| Qwen3-32B      | ?    | ?    | ?    |
| Phi-4          | ?    | ?    | ✓    |

Qwen3-14B int4 must be self-converted. Use the command above.

---

## Context capacity without u8 KV (default f16)

With `KV_CACHE_PRECISION=u8` removed, each KV GB holds ~half the tokens vs u8.
Approximate capacity at configured budgets:

| Model         | KV budget | ~tokens (f16) | Training ctx |
|---------------|-----------|---------------|--------------|
| qwen3-8b      | 4 GB      | ~37k          | 32k ✓        |
| phi-4         | 5 GB      | ~40k          | 16k ✓        |
| qwen3-14b     | 6 GB      | ~38k          | 32k ✓        |

All models remain within their training context limits at f16 precision.
