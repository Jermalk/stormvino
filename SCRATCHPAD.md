# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 13 (2026-05-06) summary

Session 13 was a short planning session. No code changed. Created `FUTURE_PLAN.md` documenting the four-phase local voice agent: Phase 1 STT (Whisper via OVModelForSpeechSeq2Seq, `/v1/audio/transcriptions`), Phase 2 TTS (Piper on CPU first, Kokoro-82M upgrade path, `/v1/audio/speech`), Phase 3 news scraper (feedparser + trafilatura, background refresh, `/v1/news/*`), Phase 4 Python voice client (sounddevice + VAD loop). VRAM budget confirmed: Whisper large-v3-turbo ~1.5 GB + qwen3-14b ~9 GB + KV 3 GB = ~14 GB, fits B60 on speed profile. Archived `CURRENT_PLAN.md` → `ARCHIVE_PLAN_2026-05-04.md` (historical only). No open blockers; user decides when to start Phase 1.
