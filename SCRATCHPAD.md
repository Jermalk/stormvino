# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 32: SDXL (sdxl-int8-ov, 3.5GB, GPU.1) + Whisper (whisper-large-v3-int8-ov, 1.57GB, GPU.1) integrated. Two bugs fixed: (1) tensor.data needed for OV Tensor → numpy conversion (Text2ImagePipeline output); (2) WhisperGenerationConfig(existing_cfg) invalid — use pipe.get_generation_config() and mutate directly. Both share GPU.1 with LLM/VLM — no contention policy yet. All tests green: 7/7 image_gen, 8/8 stt, 176/176 unit. basta-f1 query_decisions MCP tool still pending.
