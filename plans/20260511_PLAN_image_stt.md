# PLAN: Image Generation (SDXL) + STT (Whisper) — 2026-05-11

## Environment facts (KYE)

| Item | Value |
|---|---|
| ov_genai version | 2026.1.0.0 |
| `Text2ImagePipeline` | native, `generate(prompt, **kwargs) → Tensor` |
| `WhisperPipeline` | native, `generate(float_list_16khz) → WhisperDecodedResults` |
| SDXL source | `OpenVINO/stable-diffusion-xl-base-1.0-int8-ov` (36 files, pre-converted) |
| Whisper source | `OpenVINO/whisper-large-v3-int8-ov` (21 files, pre-converted) |
| GPU.0 | iGPU UHD 770, 78.5 GB shared RAM — embedding lives here |
| GPU.1 | Arc B60, 24.39 GB dGPU — LLMs + VLMs |
| Audio decode | `soundfile` available, target: float32 list at 16kHz |
| Restart | `kill $(systemctl show ov-server --property=MainPID --value)` — no sudo, Restart=always |
| Disk free | 513 GB |
| optimum-intel | BROKEN (hf-hub 1.4.1 vs transformers <1.0) — not needed |

## Device strategy

SDXL: **GPU.1** — faster denoising on dGPU. VRAM is tight (~22.71 GB limit;
LLM ~15 GB + SDXL ~6 GB = ~21 GB — within limit). Image gen loads/unloads
on demand via `_image_lock`, never occupies VRAM when idle.

Whisper: **GPU.1** — model is ~1.5 GB, negligible. Kept resident after first load.

Both: configurable via `image_device` / `stt_device` in config.json.

---

## Phase 1 — SDXL Image Generation

### Step 1.1 — Download SDXL model

```bash
source /home/jerzy/ov_env/bin/activate
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'OpenVINO/stable-diffusion-xl-base-1.0-int8-ov',
    local_dir='/opt/ov_server/models/sdxl-int8-ov',
    ignore_patterns=['*.msgpack','flax_model*','tf_model*'],
)
print('SDXL download complete')
"
```

### Step 1.2 — Smoke-test pipeline load

```bash
python3 -c "
import openvino_genai as ov_genai
pipe = ov_genai.Text2ImagePipeline('/opt/ov_server/models/sdxl-int8-ov', 'GPU.1')
result = pipe.generate('a red circle on white background', num_inference_steps=1, width=256, height=256)
print('SDXL smoke test OK, result type:', type(result))
"
```

### Step 1.3 — Write `image_pipeline.py`

New module: `/opt/ov_server/image_pipeline.py`

Responsibilities:
- `load_image_pipeline(model_dir, device)` — load `Text2ImagePipeline`, return it
- `generate_image(pipe, prompt, negative_prompt, width, height, steps, seed)` → bytes (PNG)
- `tensor_to_png(tensor)` → bytes — convert OV Tensor to PIL → PNG bytes → base64

### Step 1.4 — Add endpoint to `ov_server.py`

New request model `ImageGenerationRequest`:
```python
class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    n: int = 1  # number of images (support 1 for now)
    size: str = "1024x1024"  # "WxH" string
    response_format: str = "b64_json"  # "b64_json" | "url" (only b64_json supported)
    model: str = "sdxl-int8-ov"
    quality: str = "standard"  # ignored, OpenAI compat
    style: str = "vivid"       # ignored, OpenAI compat
    num_inference_steps: int = 20
    seed: int | None = None
```

Endpoint `POST /v1/images/generations`:
- Parse size → width, height
- Load pipeline (lazy, cached in `_image_pipe`)
- Run in executor (blocking GPU call)
- Return OpenAI-compatible response:
  ```json
  {"created": <unix_ts>, "data": [{"b64_json": "<base64_png>"}]}
  ```

### Step 1.5 — Config additions

```json
"image_model": "sdxl-int8-ov",
"image_device": "GPU.1",
"image_num_steps": 20
```

### Step 1.6 — Health additions

```json
"image_model_loaded": true/false,
"image_model_id": "sdxl-int8-ov"
```

### Step 1.7 — Auto-test: `autotest/test_image_gen.py`

Tests:
1. Model files present (xml files in model dir)
2. Pipeline loads on GPU.1
3. Generate 256×256 image, 1 step — returns non-empty bytes, valid PNG header
4. Server API: `POST /v1/images/generations` → `data[0].b64_json` is valid base64 PNG
5. Size parsing: `512x512`, `1024x1024` — both accepted
6. Negative prompt accepted without error

Run with: `python3 autotest/test_image_gen.py --api-only` (server must be up)

---

## Phase 2 — STT (Whisper)

### Step 2.1 — Download Whisper model

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'OpenVINO/whisper-large-v3-int8-ov',
    local_dir='/opt/ov_server/models/whisper-large-v3-int8-ov',
)
print('Whisper download complete')
"
```

### Step 2.2 — Smoke-test Whisper load

```bash
python3 -c "
import openvino_genai as ov_genai, numpy as np
pipe = ov_genai.WhisperPipeline('/opt/ov_server/models/whisper-large-v3-int8-ov', 'GPU.1')
# Silent 1-second audio at 16kHz
silence = [0.0] * 16000
result = pipe.generate(silence)
print('Whisper smoke test OK:', result.texts)
"
```

### Step 2.3 — Write `stt_pipeline.py`

New module: `/opt/ov_server/stt_pipeline.py`

Responsibilities:
- `load_stt_pipeline(model_dir, device)` → `WhisperPipeline`
- `decode_audio(data: bytes, content_type: str) → list[float]`
  - Uses `soundfile` to decode audio bytes → float32 array resampled to 16kHz
  - Supports: wav, mp3 (via soundfile), ogg, flac, m4a (where soundfile can handle)
- `transcribe(pipe, audio_floats, language, task) → str`

### Step 2.4 — Add endpoint to `ov_server.py`

`POST /v1/audio/transcriptions` (multipart/form-data):
- `file`: audio file upload
- `model`: str (default "whisper-large-v3-int8-ov")
- `language`: optional ISO-639-1 code
- `response_format`: "json" | "text" | "verbose_json" (default "json")
- `task`: "transcribe" | "translate" (default "transcribe")

Returns:
```json
{"text": "transcribed text here"}
```

### Step 2.5 — Config additions

```json
"stt_model": "whisper-large-v3-int8-ov",
"stt_device": "GPU.1"
```

### Step 2.6 — Health additions

```json
"stt_model_loaded": true/false
```

### Step 2.7 — Auto-test: `autotest/test_stt.py`

Tests:
1. Model files present
2. Pipeline loads on GPU.1
3. Silence (1s, 16kHz) transcribes without crash
4. Synthetic 440 Hz sine (1s) — returns string result
5. Server API: POST `/v1/audio/transcriptions` with minimal WAV → `{"text": ...}`
6. `response_format=text` returns plain string
7. `language=en` accepted without error

---

## Execution order and restart points

```
Download SDXL (~30-60 min)   [Step 1.1]
Smoke-test SDXL load          [Step 1.2]
Write image_pipeline.py       [Step 1.3]
Edit ov_server.py             [Step 1.4 + 1.5 + 1.6]
Write autotest                [Step 1.7]
Restart server
Run autotest

Download Whisper (~15-30 min) [Step 2.1]
Smoke-test Whisper load       [Step 2.2]
Write stt_pipeline.py         [Step 2.3]
Edit ov_server.py             [Step 2.4 + 2.5 + 2.6]
Write autotest                [Step 2.7]
Restart server
Run autotest

Commit: "feat: SDXL image generation + Whisper STT endpoints"
Update PROGRESS.md
Session wrap
```

## Failure modes + recovery

| Failure | Recovery |
|---|---|
| SDXL VRAM OOM on GPU.1 | Switch `image_device` to GPU.0 (iGPU, 78.5 GB shared) |
| SDXL download fails | Retry with `resume_download=True` |
| Whisper OOM | Already small (~1.5 GB), shouldn't happen |
| Server restart fails | Check `journalctl -u ov-server -n 20` |
| Autotest API failure | Check server log, isolate to pipeline vs endpoint |

## Files affected

| File | Change |
|---|---|
| `image_pipeline.py` | New |
| `stt_pipeline.py` | New |
| `ov_server.py` | +2 endpoints, +2 lazy-load blocks, +2 health fields |
| `config.json` | +image_model, image_device, image_num_steps, stt_model, stt_device |
| `CONVENTIONS.md` | +image_pipeline.py, stt_pipeline.py entries |
| `autotest/test_image_gen.py` | New |
| `autotest/test_stt.py` | New |
| `PROGRESS.md` | Updated |
