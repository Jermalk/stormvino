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

- [x] Conversion started (PID 345793) — downloading 11 metadata files first, then ~52 GB weights
- [ ] Model downloaded to HF cache (~/.cache/huggingface/hub)
- [ ] Export completed — check output dir size (expect ~14-16 GB INT4)
- [ ] VLMPipeline loads the model → run: python3 autotest/test_internvl.py --load-only
- [x] Server config.json updated — internvl2.5-26b-int4-ov added to vision task class (tier: best)
- [x] _chat_vlm updated — multi-VLM model selection by req.model
- [ ] Prompt building compatible (apply_chat_template with image tokens) — verify in test 3
- [ ] Test script passes → python3 autotest/test_internvl.py

### Next session resume point
If conversion is still running: `tail -f /tmp/internvl_convert.log`
If complete: `ls -lh /opt/ov_server/models/internvl2.5-26b-int4-ov/`
Then run: `python3 autotest/test_internvl.py --load-only` before starting the server.
