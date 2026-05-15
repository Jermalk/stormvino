"""
Text-to-speech pipeline wrapper — supports kokoro-onnx and piper-tts.

Backend is auto-detected from model_dir:
  - contains kokoro-v1.0.onnx → kokoro-onnx (24 kHz, higher quality)
  - otherwise → piper-tts (22050 Hz)

Lazy-loads on first request and keeps the model in memory.
Synthesis runs in a thread executor (CPU-bound).
Never import from ov_server.py or chat_handler.py.
"""
import asyncio
import io
import logging
import time
import wave
from pathlib import Path

log = logging.getLogger("ov_server")

_engine = None          # Kokoro instance or PiperVoice instance
_engine_lock = asyncio.Lock()
_backend: str = ""      # "kokoro" | "piper"
_voice_name: str = ""


def is_loaded() -> bool:
    return _engine is not None


def loaded_voice() -> str:
    return _voice_name


def _detect_backend(model_dir: str) -> str:
    return "kokoro" if (Path(model_dir) / "kokoro-v1.0.onnx").exists() else "piper"


async def _get_engine(model_dir: str, voice_name: str):
    global _engine, _backend, _voice_name
    async with _engine_lock:
        if _engine is not None:
            return _engine
        backend = _detect_backend(model_dir)
        loop = asyncio.get_running_loop()
        log.info(f"Loading TTS engine '{backend}' voice='{voice_name}' from {model_dir}...")
        t0 = time.perf_counter()

        if backend == "kokoro":
            onnx_path = str(Path(model_dir) / "kokoro-v1.0.onnx")
            voices_path = str(Path(model_dir) / "voices-v1.0.bin")

            def _load_kokoro():
                from kokoro_onnx import Kokoro
                return Kokoro(onnx_path, voices_path)

            _engine = await loop.run_in_executor(None, _load_kokoro)
        else:
            onnx_path = Path(model_dir) / f"{voice_name}.onnx"
            json_path = Path(model_dir) / f"{voice_name}.onnx.json"
            if not onnx_path.exists():
                raise FileNotFoundError(f"Piper voice model not found: {onnx_path}")

            def _load_piper():
                from piper import PiperVoice
                return PiperVoice.load(str(onnx_path), config_path=str(json_path), use_cuda=False)

            _engine = await loop.run_in_executor(None, _load_piper)

        _backend = backend
        _voice_name = voice_name
        log.info(f"TTS engine loaded in {time.perf_counter() - t0:.1f}s")
        return _engine


async def synthesize(text: str, model_dir: str, voice_name: str) -> bytes:
    """Synthesize text → WAV bytes."""
    engine = await _get_engine(model_dir, voice_name)
    loop = asyncio.get_running_loop()

    if _backend == "kokoro":
        def _run() -> bytes:
            import numpy as np
            t0 = time.perf_counter()
            samples, rate = engine.create(text, voice=voice_name, speed=1.0, lang="en-us")
            pcm = (np.array(samples) * 32767).clip(-32768, 32767).astype("<i2")
            buf = io.BytesIO()
            wf = wave.open(buf, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(pcm.tobytes())
            wf.close()
            wav_bytes = buf.getvalue()
            log.info(f"TTS kokoro: {len(text)} chars → {len(wav_bytes)} bytes in {time.perf_counter()-t0:.3f}s")
            return wav_bytes
    else:
        def _run() -> bytes:
            buf = io.BytesIO()
            wf = wave.open(buf, "wb")
            t0 = time.perf_counter()
            engine.synthesize_wav(text, wf)
            wf.close()
            wav_bytes = buf.getvalue()
            log.info(f"TTS piper: {len(text)} chars → {len(wav_bytes)} bytes in {time.perf_counter()-t0:.3f}s")
            return wav_bytes

    return await loop.run_in_executor(None, _run)
