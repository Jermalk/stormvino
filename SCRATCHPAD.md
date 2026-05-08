# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 21 (2026-05-08) summary

Observability Phase 1 complete: db.py (asyncpg + pgvector), inference_events / model_load_events / routing_centroids / system_snapshots tables live, /metrics/events + /metrics/summary endpoints, system snapshot loop. OV cache moved to ~/.cache/ov_b60. Per-model KV override added (model_kv_overrides in config.json). coder-14b (official) + coder-30b (official) wired to code task class.

## Model status — MUST resolve next session

### Bad models still on disk (self-converted, fail fresh OV 2026.1.0 compile)
- `qwen3-8b-int4-ov` — NFS tokenizer, assessor model, breaks on KV cache mismatch
- `qwen3-30b-a3b-int4-ov` — NFS tokenizer, general "best" model

### Root cause of the loop
- Downloads landed in -bak dirs; bad originals still active
- `qwen3-30b-a3b-int4-ov-bak` = OFFICIAL (HF: Qwen/Qwen3-30B-A3B-Instruct-2507) — just swap dirs
- `qwen3-8b-int4-ov-bak` = also SELF-CONVERTED — needs fresh download

### Fix commands (do first thing next session, before anything else)
```bash
# Step 1: swap 30B general — official is in -bak
mv /opt/ov_server/models/qwen3-30b-a3b-int4-ov /opt/ov_server/models/qwen3-30b-a3b-int4-ov-bad
mv /opt/ov_server/models/qwen3-30b-a3b-int4-ov-bak /opt/ov_server/models/qwen3-30b-a3b-int4-ov
rm -rf /opt/ov_server/models/qwen3-30b-a3b-int4-ov-bad

# Step 2: download official qwen3-8b (assessor)
# Search HF for: OpenVINO/Qwen3-8B-Instruct-int4-ov  or  openvino/qwen3-8b-instruct-int4-ov
# Download to /opt/ov_server/models/qwen3-8b-int4-ov-new, then swap once complete

# Step 3: after both models are official → restart server → assessor + 30B general will compile
```

### KV cache question (user raised, needs decision)
- 30B models get 2GB KV by model_kv_overrides — very tight (~2-4k effective context)
- Even with 1GB assessor KV, coder-30b barely fits (22.8GB/24.5GB)
- Worth discussing: remove 30B local entirely and rely on OVH for best-tier code/general?
- Alternatively accept limited context and benchmark if quality is acceptable

## Working models (confirmed good)
- qwen2.5-coder-14b-int4 — OFFICIAL, loads in ~2s from cache, 11.9GB VRAM
- qwen3-14b-int4-ov — OFFICIAL, loads OK
- qwen2.5-vl-7b-int4-ov — OFFICIAL (HF tokenizer)
- multilingual-e5-large-int8 — OFFICIAL
- qwen3-coder-30b-a3b-int4-ov — OFFICIAL, not yet tested (needs recompile, ~40 min first load)
