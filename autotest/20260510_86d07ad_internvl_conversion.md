# InternVL2.5-26B Conversion & Integration Log
## Session: 2026-05-10 | Commit: 86d07ad

### Environment
- optimum-intel: 1.27.0 | openvino-genai: 2026.1.0
- internvl_chat → _OVInternVLForCausalLM confirmed in MODEL_TYPE_TO_CLS_MAPPING
- transformers import fix: hf_shim at /tmp/hf_shim (huggingface_hub 0.36.2 symlinked)
- Disk free: 536 GB on /dev/sda2 — sufficient for 52 GB download + ~15 GB output

### Conversion command
```
PYTHONPATH=/tmp/hf_shim /home/jerzy/ov_env/bin/optimum-cli export openvino \
  --model OpenGVLab/InternVL2_5-26B \
  --trust-remote-code \
  --weight-format int4 \
  --sym \
  --ratio 1.0 \
  /opt/ov_server/models/internvl2.5-26b-int4-ov
```
Log: /tmp/internvl_convert.log

### Dependencies found during conversion
- `einops` missing → installed 0.8.2
- `timm` missing → installed 1.0.27
- Both now in `/home/jerzy/ov_env`
- Model type confirmed: `internvl_chat` (maps to `_OVInternVLForCausalLM` in optimum-intel)
- LLM backbone: `internlm2`

### Findings as they arrive

- [x] Conversion started (PID 345793/345796) — downloading ~52 GB weights (3.8 GB done at session end)
- [ ] Model downloaded to HF cache (~/.cache/huggingface/hub)
- [ ] Export completed — check output dir size (expect ~14-16 GB INT4)
- [ ] VLMPipeline loads the model → run: `python3 autotest/test_internvl.py --load-only`
- [x] Server config.json updated — internvl2.5-26b-int4-ov added to vision task class (tier: best)
- [x] _chat_vlm updated — multi-VLM model selection by req.model
- [x] Dynamic KV cache sizing implemented (server_config.py compute_kv_cache_gb)
- [ ] Prompt building compatible (apply_chat_template with image tokens) — verify in test 3
- [ ] Test script passes → `python3 autotest/test_internvl.py`

### Dynamic KV cache — session 2 notes
- `compute_kv_cache_gb(model_dir, max_context_tokens, headroom=1.25, floor_gb=1.0)` in server_config.py
- Reads model's config.json: num_hidden_layers × num_kv_heads × head_dim × 2(K+V) × 2bytes(FP16) × ctx
- Family detection via tokenizer_config.json: `<IMG_CONTEXT>` → 8192; default → 32768
- `_model_kv_gb(model_id)`: overrides > formula > global default
- Verified: Qwen3-14B → 7GB (was 8), Qwen2.5-VL-7B → 3GB, Mistral-24B → 7GB
- InternVL will get 8192-context KV once model dir exists

### Next session resume point
Check conversion: `du -sh ~/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-26B`
If complete: `ls -lh /opt/ov_server/models/internvl2.5-26b-int4-ov/`
Then run: `python3 autotest/test_internvl.py --load-only` before starting the server.
