# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 30: InternVL2.5-26B abandoned (3 download failures at 4.6GB shard); switched to 8B (~4.7GB INT4, loads in 4s). Two fixes needed: trust_remote_code=True for VLM tokenizer; _vlm_content() flattens list content to string for simple jinja templates (InternVL). Image inference works (test 4+6 pass). Tests 3+7 fail due to hf-hub version conflict in test runner only — server unaffected. Dynamic KV sizing: compute_kv_cache_gb() in server_config.py reads config.json architecture. basta-f1 project created at /home/jerzy/basta-f1 with S2A/CMM design, Postgres schema, build order (query_decisions first).
