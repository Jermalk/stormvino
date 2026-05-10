# InternVL2.5 Conversion & Integration Log
## Session: 2026-05-10 | Commit: 86d07ad

### Environment
- optimum-intel: 1.27.0 | openvino-genai: 2026.1.0
- internvl_chat → _OVInternVLForCausalLM confirmed in MODEL_TYPE_TO_CLS_MAPPING
- transformers import fix: hf_shim at /tmp/hf_shim (huggingface_hub 0.36.2 symlinked)

### Model choice
- **26B abandoned** — download repeatedly stuck at the same 4.6 GB shard (3 attempts, ~48 GB each time)
- **8B chosen** — ~17 GB download, ~5 GB INT4 output, fits easily on GPU.1

### Conversion command (8B)
```
PYTHONPATH=/tmp/hf_shim /home/jerzy/ov_env/bin/optimum-cli export openvino \
  --model OpenGVLab/InternVL2_5-8B \
  --trust-remote-code \
  --weight-format int4 \
  --sym \
  --ratio 1.0 \
  /opt/ov_server/models/internvl2.5-8b-int4-ov
```
Log: /tmp/internvl_convert.log

### Dependencies (already installed from 26B attempts)
- `einops` 0.8.2 — in `/home/jerzy/ov_env`
- `timm` 1.0.27 — in `/home/jerzy/ov_env`

### Findings

- [x] 26B HF cache deleted (~43 GB freed)
- [x] config.json updated — internvl2.5-8b-int4-ov in vision task class (tier: best)
- [x] test_internvl.py updated — all model refs point to 8B
- [x] Conversion started — downloading ~17 GB
- [ ] Export completed — check output dir size (expect ~5 GB INT4)
- [ ] VLMPipeline loads → run: `python3 autotest/test_internvl.py --load-only`
- [ ] Full test suite passes → `python3 autotest/test_internvl.py`

### Next session resume point
Check: `du -sh ~/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-8B`
If conversion done: `ls -lh /opt/ov_server/models/internvl2.5-8b-int4-ov/`
Then: `python3 autotest/test_internvl.py --load-only`
