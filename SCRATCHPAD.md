# SCRATCHPAD.md — in-session working memory

> Cleared at start of every session. Carry-over summary written as first entry.
> Format: bullet points, max 5 lines per topic, no prose.

## Carried over:
Session 28: converted mistral-small-3.2-24b-int4-ov (text-only anthracite-core strip, INT4 sym group-128). Fixed load failure (AUTO → GPU.1 in config.json). Added Mistral tool support in prompt_builder.py: _is_mistral_template() detects [SYSTEM_PROMPT] template, _build_mistral_tool_prompt() injects [AVAILABLE_TOOLS]/[TOOL_RESULTS], parse_tool_calls() extended to handle function_name{json} format. Automated test autotest/test_web_search.py: 4/4 both models. Mistral stays at tier="balanced" (not selected in fast profile) pending real-traffic validation.
