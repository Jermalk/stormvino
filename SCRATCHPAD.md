# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over: Session 14 (2026-05-06) summary

Session 14 was a tooling and workflow design session. No ov_server code changed. Benchmarked t/s: qwen3-8b 23–31 t/s, qwen3-14b 14 t/s, coder-14b 11 t/s. Installed aider into ov_env; added `aider` and `george` aliases to ~/.bashrc (george defaults to qwen3-14b + --edit-format diff). Tested aider on GrainMesh — worked but introduced a `1.:0` SQL bug via whole-format truncation; reverted 4 commits. Designed Architect+George two-agent protocol (TASK_LEDGER.md in EternalGrain, ARCHITECT_MODE.md written). Planned MCP server (`george_mcp.py`) to expose george and ov_server as first-class Claude Code tools — deferred to next session. Server switched to document profile (12GB KV) during session; may need switching back to speed.
