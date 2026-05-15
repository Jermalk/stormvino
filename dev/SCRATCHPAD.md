# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 56: TTS pipeline upgraded from piper-only to dual-engine: kokoro-onnx (af_kore, 24kHz) for English default, piper (pl_PL-gosia-medium, 22050Hz) for Polish. tts_pipeline.py holds multiple engines simultaneously keyed by model_dir. media_routes.py auto-routes via _PIPER_VOICE_RE (xx_XX- prefix → piper/). kokoro PyTorch package rejected due to catalogue.py shadowing; kokoro-onnx used instead. STT Polish confirmed working (language=pl, Whisper already multilingual). 159/159 tests pass.
