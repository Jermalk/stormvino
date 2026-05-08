# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 25 wrapped cleanly. Docker.md created with full run/manage commands for Open WebUI (port 3000, host-gateway) and SearxNG (port 8080, ~/searxng-settings.yml mount). SearxNG 403 fixed by adding json to formats in settings.yml. Web search end-to-end confirmed working. Open WebUI web search uses RAG injection (not tool_calls); SearxNG must have json format enabled or Open WebUI raises HTTP 400.
