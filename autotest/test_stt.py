"""
Automated tests for Whisper STT (speech-to-text).

Tests:
  1. Model files present — whisper-large-v3-int8-ov dir has expected .xml files
  2. Pipeline loads on GPU.1
  3. Silence (1s, 16kHz) transcribes without crash, returns string
  4. Synthetic 440 Hz sine (1s) — transcribes without crash
  5. Server API: POST /v1/audio/transcriptions with minimal WAV → {"text": ...}
  6. response_format=text returns plain string
  7. language=en accepted without error
  8. Health endpoint shows stt_model_loaded=true after first call

Usage:
  python3 autotest/test_stt.py            # all tests (requires server)
  python3 autotest/test_stt.py --load-only  # tests 1-4 only (no server)
  python3 autotest/test_stt.py --api-only   # tests 5-8 only (server required)
"""
import argparse
import io
import math
import struct
import sys
import time
from pathlib import Path

import httpx

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "whisper-large-v3-int8-ov"
DEVICE = "GPU.1"
BASE = "http://localhost:11435"
TIMEOUT = 180.0


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def _get(path: str) -> dict:
    with httpx.Client(timeout=10) as c:
        r = c.get(f"{BASE}{path}")
        r.raise_for_status()
        return r.json()


def _make_wav_bytes(samples: list[float], sample_rate: int = 16000) -> bytes:
    """Build a minimal PCM WAV from a float list (values in [-1, 1])."""
    n_samples = len(samples)
    # Convert float32 → int16
    pcm = b"".join(
        struct.pack("<h", max(-32768, min(32767, int(s * 32767))))
        for s in samples
    )
    data_size = len(pcm)
    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))        # chunk size
    buf.write(struct.pack("<H", 1))         # PCM
    buf.write(struct.pack("<H", 1))         # mono
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))  # byte rate
    buf.write(struct.pack("<H", 2))         # block align
    buf.write(struct.pack("<H", 16))        # bits per sample
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm)
    return buf.getvalue()


def _silence_wav(duration_sec: float = 1.0, rate: int = 16000) -> bytes:
    return _make_wav_bytes([0.0] * int(duration_sec * rate), rate)


def _sine_wav(freq: float = 440.0, duration_sec: float = 1.0, rate: int = 16000) -> bytes:
    n = int(duration_sec * rate)
    samples = [0.3 * math.sin(2 * math.pi * freq * i / rate) for i in range(n)]
    return _make_wav_bytes(samples, rate)


# ---------------------------------------------------------------------------
# Test 1 — model files present
# ---------------------------------------------------------------------------

def test_files_present() -> bool:
    print("\n[1] Model files present")
    ok = True
    ok &= _check("model directory exists", MODEL_DIR.exists(), str(MODEL_DIR))
    if not MODEL_DIR.exists():
        return False
    xml_files = list(MODEL_DIR.glob("*.xml"))
    ok &= _check("at least 2 .xml IR files (encoder+decoder)", len(xml_files) >= 2,
                 f"found {len(xml_files)}: {[f.name for f in xml_files]}")
    total_gb = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file()) / 1e9
    ok &= _check("total size > 0.5 GB", total_gb > 0.5, f"{total_gb:.2f} GB")
    return ok


# ---------------------------------------------------------------------------
# Test 2 — pipeline loads
# ---------------------------------------------------------------------------

def test_pipeline_loads() -> bool:
    print(f"\n[2] WhisperPipeline loads on {DEVICE}")
    try:
        import openvino_genai as ov_genai
    except ImportError:
        return _check("openvino_genai importable", False)

    if not MODEL_DIR.exists():
        return _check("model dir present", False)

    t0 = time.perf_counter()
    try:
        pipe = ov_genai.WhisperPipeline(str(MODEL_DIR), DEVICE)
        elapsed = time.perf_counter() - t0
        return _check("WhisperPipeline loads", True, f"{elapsed:.0f}s")
    except Exception as exc:
        return _check("pipeline load", False, str(exc)[:200])


# ---------------------------------------------------------------------------
# Test 3 — silence transcription
# ---------------------------------------------------------------------------

def test_silence() -> bool:
    print("\n[3] Silence (1s, 16kHz) transcribes without crash")
    try:
        import openvino_genai as ov_genai
        import numpy as np
        pipe = ov_genai.WhisperPipeline(str(MODEL_DIR), DEVICE)
        silence = [0.0] * 16000
        t0 = time.perf_counter()
        result = pipe.generate(silence)
        elapsed = time.perf_counter() - t0
        text = "".join(result.texts)
        ok = isinstance(text, str)
        return _check("silence returns str", ok, f"{text!r} ({elapsed:.1f}s)")
    except Exception as exc:
        return _check("silence transcription", False, str(exc)[:200])


# ---------------------------------------------------------------------------
# Test 4 — sine wave transcription
# ---------------------------------------------------------------------------

def test_sine() -> bool:
    print("\n[4] 440 Hz sine (1s) transcribes without crash")
    try:
        import openvino_genai as ov_genai
        import math
        pipe = ov_genai.WhisperPipeline(str(MODEL_DIR), DEVICE)
        rate = 16000
        sine = [0.3 * math.sin(2 * math.pi * 440 * i / rate) for i in range(rate)]
        t0 = time.perf_counter()
        result = pipe.generate(sine)
        elapsed = time.perf_counter() - t0
        text = "".join(result.texts)
        ok = isinstance(text, str)
        return _check("sine returns str", ok, f"{text!r} ({elapsed:.1f}s)")
    except Exception as exc:
        return _check("sine transcription", False, str(exc)[:200])


# ---------------------------------------------------------------------------
# Test 5 — server API: POST /v1/audio/transcriptions
# ---------------------------------------------------------------------------

def test_api_transcribe() -> bool:
    print("\n[5] Server API: POST /v1/audio/transcriptions (silence WAV)")
    wav = _silence_wav(1.0)
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            t0 = time.perf_counter()
            r = c.post(
                f"{BASE}/v1/audio/transcriptions",
                files={"file": ("silence.wav", wav, "audio/wav")},
                data={"model": "whisper-large-v3-int8-ov", "response_format": "json"},
            )
            r.raise_for_status()
            elapsed = time.perf_counter() - t0
        resp = r.json()
        ok = True
        ok &= _check("response has 'text'", "text" in resp, str(resp)[:80])
        ok &= _check("text is str", isinstance(resp.get("text"), str),
                     f"{resp.get('text')!r} ({elapsed:.1f}s)")
        return ok
    except Exception as exc:
        return _check("API call", False, str(exc)[:200])


# ---------------------------------------------------------------------------
# Test 6 — response_format=text
# ---------------------------------------------------------------------------

def test_api_text_format() -> bool:
    print("\n[6] response_format=text returns plain string")
    wav = _silence_wav(0.5)
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.post(
                f"{BASE}/v1/audio/transcriptions",
                files={"file": ("silence.wav", wav, "audio/wav")},
                data={"response_format": "text"},
            )
            r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        ok = _check("content-type is text/plain", "text/plain" in content_type,
                    content_type)
        return ok
    except Exception as exc:
        return _check("text format request", False, str(exc)[:200])


# ---------------------------------------------------------------------------
# Test 7 — language=en accepted
# ---------------------------------------------------------------------------

def test_api_language() -> bool:
    print("\n[7] language=en accepted")
    wav = _silence_wav(0.5)
    try:
        with httpx.Client(timeout=TIMEOUT) as c:
            r = c.post(
                f"{BASE}/v1/audio/transcriptions",
                files={"file": ("silence.wav", wav, "audio/wav")},
                data={"language": "en"},
            )
            r.raise_for_status()
        return _check("language=en accepted (no 400/500)", True)
    except Exception as exc:
        return _check("language=en request", False, str(exc)[:200])


# ---------------------------------------------------------------------------
# Test 8 — health shows stt_model_loaded
# ---------------------------------------------------------------------------

def test_health_stt() -> bool:
    print("\n[8] Health: stt_model_loaded=true")
    try:
        h = _get("/health")
        return _check("stt_model_loaded=true", h.get("stt_model_loaded") is True,
                      str(h.get("stt_model_loaded")))
    except Exception as exc:
        return _check("health check", False, str(exc)[:200])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-only", action="store_true", help="Tests 1-4 only")
    parser.add_argument("--api-only", action="store_true", help="Tests 5-8 only (server required)")
    args = parser.parse_args()

    passed = failed = 0

    def run(fn) -> bool:
        nonlocal passed, failed
        ok = fn()
        if ok:
            passed += 1
        else:
            failed += 1
        return ok

    if args.api_only:
        try:
            _get("/health")
        except Exception as e:
            print(f"Server unreachable: {e}")
            sys.exit(1)
        run(test_api_transcribe)
        run(test_api_text_format)
        run(test_api_language)
        run(test_health_stt)
    elif args.load_only:
        run(test_files_present)
        run(test_pipeline_loads)
        run(test_silence)
        run(test_sine)
    else:
        run(test_files_present)
        run(test_pipeline_loads)
        run(test_silence)
        run(test_sine)
        try:
            _get("/health")
            run(test_api_transcribe)
            run(test_api_text_format)
            run(test_api_language)
            run(test_health_stt)
        except Exception:
            print("\n  (server not reachable — skipping API tests 5-8)")

    total = passed + failed
    print(f"\n{'='*60}")
    print(f"Result: {passed}/{total} passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
