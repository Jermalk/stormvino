"""
Speech-to-text pipeline wrapper.

Wraps ov_genai.WhisperPipeline for use in ov_server.py.
Audio decoding uses soundfile; resampling via numpy (linear, adequate for STT).
Never import from ov_server.py.
"""
import asyncio
import io
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("ov_server")

_stt_pipe = None
_stt_model_id: str = ""
_stt_lock = asyncio.Lock()


def is_loaded() -> bool:
    return _stt_pipe is not None


def loaded_model_id() -> str:
    return _stt_model_id


async def get_stt_pipeline(model_dir: str, device: str):
    """Return loaded WhisperPipeline, loading it on first call."""
    global _stt_pipe, _stt_model_id
    async with _stt_lock:
        if _stt_pipe is not None:
            return _stt_pipe
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        log.info(f"Loading STT pipeline from {model_dir} on {device}...")
        try:
            import openvino_genai as ov_genai
            _stt_pipe = await loop.run_in_executor(
                None,
                lambda: ov_genai.WhisperPipeline(model_dir, device)
            )
            _stt_model_id = Path(model_dir).name
            log.info(f"STT pipeline loaded in {time.perf_counter() - t0:.1f}s")
        except Exception as exc:
            log.error(f"Failed to load STT pipeline: {exc}")
            raise
        return _stt_pipe


def _decode_audio_bytes(data: bytes, filename: str = "") -> np.ndarray:
    """Decode audio bytes → float32 numpy array at 16 kHz, mono.

    Uses soundfile for the actual decoding. Linear resampling to 16kHz.
    Supports WAV, FLAC, OGG, AIFF natively via soundfile (libsndfile).
    MP3/M4A require ffmpeg installed; soundfile will raise if unavailable.
    """
    import soundfile as sf
    with sf.SoundFile(io.BytesIO(data)) as f:
        samples = f.read(dtype="float32", always_2d=True)  # (frames, channels)
        src_rate = f.samplerate

    # Mix to mono
    mono = samples.mean(axis=1)

    # Resample to 16kHz via linear interpolation if necessary
    target_rate = 16_000
    if src_rate != target_rate:
        old_len = len(mono)
        new_len = int(old_len * target_rate / src_rate)
        mono = np.interp(
            np.linspace(0, old_len - 1, new_len),
            np.arange(old_len),
            mono,
        ).astype(np.float32)

    # Clamp to [-1, 1] as required by WhisperPipeline
    mono = np.clip(mono, -1.0, 1.0)
    return mono


async def transcribe(
    audio_data: bytes,
    filename: str = "audio",
    language: Optional[str] = None,
    task: str = "transcribe",
    model_dir: str = "",
    device: str = "GPU.1",
) -> str:
    """Transcribe audio bytes; return transcribed text string."""
    pipe = await get_stt_pipeline(model_dir, device)
    loop = asyncio.get_running_loop()

    def _run() -> str:
        audio_floats = _decode_audio_bytes(audio_data, filename)
        cfg = pipe.get_generation_config()
        if language:
            cfg.language = f"<|{language}|>"
        cfg.task = task
        cfg.return_timestamps = False
        t0 = time.perf_counter()
        result = pipe.generate(audio_floats.tolist(), cfg)
        elapsed = time.perf_counter() - t0
        text = "".join(result.texts).strip()
        log.info(f"STT: {len(audio_floats)/16000:.1f}s audio → {len(text)} chars in {elapsed:.1f}s")
        return text

    return await loop.run_in_executor(None, _run)
