# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 52: VramBar loading animation made reliable — startup_loading flag added to app_state + /health, optimistic loadingOverride in App.svelte fires immediately on user action. Manual model selector added to ProfilesPanel with LOCAL/Remote optgroups, OVH pricing in PLN (IN/OUT), fetched from /v1/models with scope-reactive re-fetch. POST /admin/load-model added: AUTO re-applies profile, local LLM/VLM warm, OVH model ID sets in-memory routing override. OVH proxy fixed: strip `thinking` + `repetition_penalty` before forwarding. Both gpt-oss-120b and Qwen3-32B confirmed working via curl.
