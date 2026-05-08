# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 20 (2026-05-08) summary

Phase 2 + Phase 3 complete in one session. Phase 2: routing wired into chat() (Step 2.4), routing decision with confidence/latency in /health + ov-monitor Last route row (Step 2.5), ThinkStreamHandler + usage chunk + StreamingToolCallHandler stub (Step 2.6). Phase 3: assessor LLMPipeline bootstrapped at startup (Step 3.1), routing prompt builder with per-(scope,profile) prefix-cacheable system block (Step 3.2), assessor wired into Stage 3 routing with pipe reuse when task model == assessor model (Step 3.3). Tests: 186/186. Next: Phase 4 Step 4.1 (task graph executor) OR live routing validation on running server.
