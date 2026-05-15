"""Owns /v1/images/generations, /v1/audio/transcriptions, /v1/audio/speech endpoints.

Never import from ov_server.py or chat_handler.py.
Imports: server_config, image_pipeline, stt_pipeline, tts_pipeline.
To add a new media type: add a new endpoint function and register its router in ov_server.py.
"""
import logging
import re
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

# Piper voice names look like: pl_PL-gosia-medium, en_US-amy-medium
_PIPER_VOICE_RE = re.compile(r"^[a-z]{2}_[A-Z]{2}-")

import image_pipeline
import stt_pipeline
import tts_pipeline
from server_config import DEVICE, MODELS_DIR, _cfg

log = logging.getLogger("ov_server")

media_router = APIRouter()


class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    model: str = ""
    quality: str = "standard"
    style: str = "vivid"
    num_inference_steps: int | None = None
    seed: int | None = None


class AudioTranscriptionRequest(BaseModel):
    model: str = ""
    language: str | None = None
    response_format: str = "json"
    task: str = "transcribe"
    temperature: float | None = None


class TTSSpeechRequest(BaseModel):
    model: str = "piper"
    input: str
    voice: str = ""
    language: str = "auto"   # "auto" | "en" | "pl" | …
    response_format: str = "wav"
    speed: float = 1.0


# Polish diacritics that cannot appear in plain English text.
_PL_RE = re.compile(r"[ąęóśźżćńłĄĘÓŚŹŻĆŃŁ]")


def _auto_voice(text: str, cfg: dict) -> str:
    """Return the best TTS voice for *text* based on diacritic detection."""
    if _PL_RE.search(text):
        return cfg.get("tts_voice_pl", "pl_PL-gosia-medium")
    return cfg.get("tts_voice", "af_kore")


@media_router.post("/v1/images/generations")
async def images_generations(req: ImageGenerationRequest):
    model_id = req.model or _cfg.get("image_model", "")
    if not model_id:
        raise HTTPException(
            status_code=400, detail="No image_model configured and no model in request"
        )
    model_dir = str(Path(MODELS_DIR) / model_id)
    if not Path(model_dir).exists():
        raise HTTPException(
            status_code=400, detail=f"Image model '{model_id}' not found at {model_dir}"
        )
    device = _cfg.get("image_device", DEVICE)
    steps = (
        req.num_inference_steps
        if req.num_inference_steps is not None
        else _cfg.get("image_num_steps", 20)
    )
    width, height = image_pipeline._parse_size(req.size)
    n = max(1, min(req.n, 4))

    try:
        b64_images = await image_pipeline.generate_images(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            seed=req.seed,
            num_images=n,
            model_dir=model_dir,
            device=device,
        )
    except Exception as exc:
        log.error(f"Image generation error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "created": int(time.time()),
        "data": [{"b64_json": b64} for b64 in b64_images],
    }


@media_router.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    task: str = Form(default="transcribe"),
):
    model_id = _cfg.get("stt_model", "")
    if not model_id:
        raise HTTPException(status_code=400, detail="No stt_model configured")
    model_dir = str(Path(MODELS_DIR) / model_id)
    if not Path(model_dir).exists():
        raise HTTPException(
            status_code=400, detail=f"STT model '{model_id}' not found at {model_dir}"
        )
    device = _cfg.get("stt_device", DEVICE)

    audio_bytes = await file.read()
    try:
        text = await stt_pipeline.transcribe(
            audio_data=audio_bytes,
            filename=file.filename or "audio",
            language=language,
            task=task,
            model_dir=model_dir,
            device=device,
        )
    except Exception as exc:
        log.error(f"STT transcription error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    if response_format == "text":
        return PlainTextResponse(text)
    return {"text": text}


@media_router.post("/v1/audio/speech")
async def audio_speech(req: TTSSpeechRequest):
    if req.voice:
        voice_name = req.voice                        # explicit — honour it
    elif req.language == "auto":
        voice_name = _auto_voice(req.input, _cfg)     # detect from text
    elif req.language == "pl":
        voice_name = _cfg.get("tts_voice_pl", "pl_PL-gosia-medium")
    else:
        voice_name = _cfg.get("tts_voice", "af_kore")
    if _PIPER_VOICE_RE.match(voice_name):
        model_dir = str(Path(MODELS_DIR) / "piper")
    else:
        model_dir = str(Path(MODELS_DIR) / _cfg.get("tts_model_dir", "kokoro"))

    if not Path(model_dir).exists():
        raise HTTPException(
            status_code=400,
            detail=f"TTS model dir '{model_dir}' not found",
        )
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' must not be empty")

    try:
        wav_bytes = await tts_pipeline.synthesize(
            text=req.input,
            model_dir=model_dir,
            voice_name=voice_name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        log.error(f"TTS synthesis error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return Response(content=wav_bytes, media_type="audio/wav")
