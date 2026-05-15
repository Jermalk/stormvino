"""
Text-to-speech pipeline wrapper using piper-tts.

Lazy-loads the PiperVoice on first request and keeps it in memory.
Synthesis runs in a thread executor (CPU-bound, ~60 ms for short text).
Never import from ov_server.py or chat_handler.py.
"""
import asyncio
import io
import logging
import time
import wave
from pathlib import Path

log = logging.getLogger("ov_server")

_voice = None
_voice_name: str = ""
_voice_lock = asyncio.Lock()


def is_loaded() -> bool:
    return _voice is not None


def loaded_voice() -> str:
    return _voice_name


async def get_voice(model_dir: str, voice_name: str):
    """Return loaded PiperVoice, loading it on first call."""
    global _voice, _voice_name
    async with _voice_lock:
        if _voice is not None:
            return _voice
        loop = asyncio.get_running_loop()
        onnx_path = Path(model_dir) / f"{voice_name}.onnx"
        json_path = Path(model_dir) / f"{voice_name}.onnx.json"
        if not onnx_path.exists():
            raise FileNotFoundError(f"TTS voice model not found: {onnx_path}")
        log.info(f"Loading TTS voice '{voice_name}' from {model_dir}...")
        t0 = time.perf_counter()

        def _load():
            from piper import PiperVoice
            return PiperVoice.load(str(onnx_path), config_path=str(json_path), use_cuda=False)

        _voice = await loop.run_in_executor(None, _load)
        _voice_name = voice_name
        log.info(f"TTS voice loaded in {time.perf_counter() - t0:.1f}s")
        return _voice


async def synthesize(
    text: str,
    model_dir: str,
    voice_name: str,
) -> bytes:
    """Synthesize text → WAV bytes (16-bit PCM, mono, 22050 Hz)."""
    voice = await get_voice(model_dir, voice_name)
    loop = asyncio.get_running_loop()

    def _run() -> bytes:
        buf = io.BytesIO()
        wf = wave.open(buf, "wb")
        t0 = time.perf_counter()
        voice.synthesize_wav(text, wf)
        wf.close()
        elapsed = time.perf_counter() - t0
        wav_bytes = buf.getvalue()
        log.info(f"TTS: {len(text)} chars → {len(wav_bytes)} bytes in {elapsed:.3f}s")
        return wav_bytes

    return await loop.run_in_executor(None, _run)
