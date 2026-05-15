# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 55: TTS Phase 1 complete. tts_pipeline.py (lazy PiperVoice, executor-offloaded), POST /v1/audio/speech in media_routes.py, config keys tts_model_dir/tts_voice added + whitelisted in server_config.py, health reports tts_voice_loaded + tts_voice_id. piper-tts 1.4.x API: synthesize_wav(text, wave.Wave_write). Voice: en_US-lessac-medium at models/piper/. 159/159 tests pass.
