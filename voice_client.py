#!/usr/bin/env python3
"""
voice_client.py — local voice conversation agent (PoC)

Loop:
  record (VAD) → transcribe (Whisper) → chat (LLM) → synthesize (TTS) → play → repeat

Usage:
  python3 voice_client.py
  python3 voice_client.py --briefing          # fetch + read news on startup
  python3 voice_client.py --model qwen3-14b-int4-ov
  python3 voice_client.py --lang pl           # Polish voice + system prompt hint

Config (optional): ~/.voice_agent.json
  {
    "server": "http://localhost:11435",
    "model": "Auto",
    "tts_voice": "af_kore",
    "tts_lang": "en",
    "history_turns": 6,
    "silence_threshold": 0.02,
    "silence_duration": 1.5,
    "max_record_sec": 30
  }
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import sounddevice as sd
import soundfile as sf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = {
    "server":            "http://localhost:11435",
    "model":             "Auto",
    "tts_voice":         "af_kore",
    "tts_lang":          "en",
    "history_turns":     6,
    "silence_threshold": 0.02,
    "silence_duration":  1.5,
    "max_record_sec":    30,
}

def load_config(overrides: dict) -> dict:
    cfg = dict(DEFAULTS)
    cfg_path = Path.home() / ".voice_agent.json"
    if cfg_path.exists():
        with open(cfg_path) as fh:
            cfg.update(json.load(fh))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text

def info(msg: str)  -> None: print(_c("36", f"  {msg}"))
def user(msg: str)  -> None: print(_c("32", f"\nYou: {msg}"))
def agent(msg: str) -> None: print(_c("33", f"Agent: {msg}"))
def warn(msg: str)  -> None: print(_c("31", f"  ⚠ {msg}"), file=sys.stderr)

# ---------------------------------------------------------------------------
# Recording with energy-based VAD
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000   # Whisper expects 16 kHz
CHANNELS    = 1
CHUNK_SEC   = 0.05     # 50 ms chunks

def record_until_silence(cfg: dict) -> np.ndarray | None:
    """Record mic until silence_duration seconds of silence after speech starts.

    Returns float32 mono array at SAMPLE_RATE, or None if nothing was said.
    """
    thresh          = cfg["silence_threshold"]
    silence_dur     = cfg["silence_duration"]
    max_dur         = cfg["max_record_sec"]
    chunk_samples   = int(SAMPLE_RATE * CHUNK_SEC)
    silence_needed  = int(silence_dur / CHUNK_SEC)
    max_chunks      = int(max_dur / CHUNK_SEC)

    frames:        list[np.ndarray] = []
    silence_count: int = 0
    speaking:      bool = False

    print(_c("36", "\n[Listening…]"), end="", flush=True)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_samples)
                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if rms > thresh:
                    if not speaking:
                        print(_c("32", " [Recording]"), end="", flush=True)
                        speaking = True
                    silence_count = 0
                    frames.append(chunk.copy())
                elif speaking:
                    frames.append(chunk.copy())
                    silence_count += 1
                    if silence_count >= silence_needed:
                        break
    except sd.PortAudioError as exc:
        warn(f"Audio device error: {exc}")
        return None

    print()
    if not frames:
        return None
    return np.concatenate(frames, axis=0).flatten()

# ---------------------------------------------------------------------------
# STT — POST /v1/audio/transcriptions
# ---------------------------------------------------------------------------

def transcribe(audio: np.ndarray, server: str) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    try:
        r = httpx.post(
            f"{server}/v1/audio/transcriptions",
            files={"file": ("utterance.wav", buf, "audio/wav")},
            data={"model": "whisper"},
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json().get("text", "").strip()
    except Exception as exc:
        warn(f"Transcription failed: {exc}")
        return ""

# ---------------------------------------------------------------------------
# Chat — POST /v1/chat/completions (streaming)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Respond concisely — your reply will be read aloud. "
    "Avoid markdown, bullet lists, and code blocks unless the user explicitly asks. "
    "Keep answers under 3 sentences when possible."
)

SYSTEM_PROMPT_PL = (
    "Jesteś pomocnym asystentem głosowym. "
    "Odpowiadaj zwięźle — Twoja odpowiedź zostanie odczytana na głos. "
    "Unikaj markdown, list i bloków kodu, chyba że użytkownik o to prosi. "
    "Trzymaj odpowiedzi do 3 zdań gdy to możliwe."
)

def chat_stream(messages: list[dict], model: str, server: str) -> str:
    """Stream chat completions; return full reply text."""
    try:
        full = []
        print(_c("33", "Agent: "), end="", flush=True)
        with httpx.stream(
            "POST",
            f"{server}/v1/chat/completions",
            json={"model": model, "messages": messages, "stream": True, "thinking": False},
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        full.append(delta)
                except (json.JSONDecodeError, KeyError):
                    pass
        print()
        return "".join(full).strip()
    except Exception as exc:
        warn(f"Chat failed: {exc}")
        return ""

# ---------------------------------------------------------------------------
# TTS — POST /v1/audio/speech
# ---------------------------------------------------------------------------

def synthesize(text: str, cfg: dict) -> np.ndarray | None:
    lang  = cfg.get("tts_lang", "en")
    voice = cfg.get("tts_voice", "af_kore")
    try:
        r = httpx.post(
            f"{cfg['server']}/v1/audio/speech",
            json={"model": "tts", "input": text, "voice": voice, "language": lang},
            timeout=60.0,
        )
        r.raise_for_status()
        audio, sr = sf.read(io.BytesIO(r.content), dtype="float32")
        # Resample to output device rate if needed (sounddevice handles it via sd.play)
        return audio, sr
    except Exception as exc:
        warn(f"TTS failed: {exc}")
        return None, None

def play(audio: np.ndarray, sr: int) -> None:
    sd.play(audio, samplerate=sr)
    sd.wait()

# ---------------------------------------------------------------------------
# News briefing
# ---------------------------------------------------------------------------

def morning_briefing(cfg: dict, messages: list[dict]) -> None:
    info("Fetching news…")
    try:
        r = httpx.get(f"{cfg['server']}/v1/news/context", timeout=10.0)
        context = r.text if r.status_code == 200 else ""
    except Exception:
        context = ""

    if not context or "no news" in context:
        info("No news available — skipping briefing")
        return

    briefing_prompt = (
        "Here is the current news digest:\n\n"
        f"{context}\n\n"
        "Please give me a 3-sentence spoken morning briefing covering the most important stories."
    )
    messages.append({"role": "user", "content": briefing_prompt})
    reply = chat_stream(messages, cfg["model"], cfg["server"])
    messages.append({"role": "assistant", "content": reply})

    if reply:
        info("Synthesizing briefing…")
        audio, sr = synthesize(reply, cfg)
        if audio is not None:
            play(audio, sr)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def build_history(history: list[dict], max_turns: int) -> list[dict]:
    """Return system message + last max_turns pairs (user+assistant)."""
    pairs = []
    i = 0
    while i < len(history) - 1:
        if history[i]["role"] == "user" and history[i+1]["role"] == "assistant":
            pairs.append((history[i], history[i+1]))
            i += 2
        else:
            i += 1
    kept = pairs[-max_turns:]
    return [msg for pair in kept for msg in pair]

def main() -> None:
    parser = argparse.ArgumentParser(description="Voice conversation agent")
    parser.add_argument("--briefing", action="store_true", help="Read news on startup")
    parser.add_argument("--model",    default=None, help="Override model")
    parser.add_argument("--lang",     default=None, help="Language hint: en | pl")
    args = parser.parse_args()

    overrides = {
        "model":    args.model,
        "tts_lang": args.lang,
    }
    if args.lang == "pl":
        overrides["tts_voice"] = "pl_PL-gosia-medium"

    cfg = load_config(overrides)

    sys_prompt = SYSTEM_PROMPT_PL if cfg.get("tts_lang") == "pl" else SYSTEM_PROMPT
    system_msg = {"role": "system", "content": sys_prompt}

    conversation: list[dict] = []   # user/assistant turns only (no system)

    print(_c("36;1", "\n=== Voice Agent ==="))
    info(f"Server : {cfg['server']}")
    info(f"Model  : {cfg['model']}")
    info(f"Voice  : {cfg['tts_voice']} ({cfg['tts_lang']})")
    info("Ctrl+C to quit\n")

    # Verify server
    try:
        httpx.get(f"{cfg['server']}/health", timeout=3.0).raise_for_status()
    except Exception:
        warn(f"Server not reachable at {cfg['server']} — exiting")
        sys.exit(1)

    if args.briefing:
        morning_briefing(cfg, conversation)

    while True:
        try:
            audio = record_until_silence(cfg)
        except KeyboardInterrupt:
            print("\nBye.")
            break

        if audio is None or len(audio) < SAMPLE_RATE * 0.3:
            info("(no speech detected)")
            continue

        info("Transcribing…")
        text = transcribe(audio, cfg["server"])
        if not text:
            info("(could not transcribe)")
            continue
        user(text)

        # Trim conversation to rolling window
        history_window = build_history(conversation, cfg["history_turns"])
        messages = [system_msg] + history_window + [{"role": "user", "content": text}]

        reply = chat_stream(messages, cfg["model"], cfg["server"])
        if not reply:
            continue

        conversation.append({"role": "user",      "content": text})
        conversation.append({"role": "assistant",  "content": reply})

        info("Synthesizing…")
        audio_out, sr = synthesize(reply, cfg)
        if audio_out is not None:
            play(audio_out, sr)


if __name__ == "__main__":
    main()
