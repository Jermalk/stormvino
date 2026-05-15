# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 60 (2026-05-15) completed the voice pipeline and launched the plugin PoC. VoiceButton.svelte gained VAD silence auto-detection (AnalyserNode RMS polling, 1.5s threshold), a TTS stop button (green pulsing ring, pauses currentAudio), a close button on the speech bubble, and an EN/PL/auto language toggle that sends the hint to Whisper so Polish isn't misidentified as Romanian. On the server side: TTS language auto-detection from Polish diacritics (`_auto_voice()`), `_date_prefix()` now injects local time + timezone (CEST/UTC+02:00) into every system prompt, VramBar loading shimmer fixed by removing the `loadingGb > 1.2` false trigger, and a WebSearchPlugin PoC wired into `plugin_runner.py` that injects SearxNG results as a synthetic system message before LLM inference. A 7-step integration plan (`dev/PLAN_plugin_infergate.md`) was written to unify the plugin and infergate classifiers; Step 1 (add `task_class_trigger` to BasePlugin) is the next action.
