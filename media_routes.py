"""Owns /v1/images/generations and /v1/audio/transcriptions endpoints.

Never import from ov_server.py or chat_handler.py.
Imports: server_config, image_pipeline, stt_pipeline.
To add a new media type: add a new endpoint function and register its router in ov_server.py.
"""
import logging
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

import image_pipeline
import stt_pipeline
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
