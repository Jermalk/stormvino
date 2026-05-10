"""
Automated test for the web search / tool-call routing scenario.

Tests:
  1. Routing — keyword trigger routes to web_search task class
  2. Routing — has_tools trigger routes to web_search task class
  3. Tool call generation — model produces a valid <tool_call> or JSON tool_call
  4. Tool loop — model integrates tool result and produces final answer
  5. Model comparison — runs the tool loop against both local models

Usage:
  python3 autotest/test_web_search.py
  python3 autotest/test_web_search.py --model mistral-small-3.2-24b-int4-ov
  python3 autotest/test_web_search.py --model qwen3-14b-int4-ov
  python3 autotest/test_web_search.py --compare   # run both and compare
"""
import argparse
import json
import sys
import time
from typing import Any

import httpx

BASE = "http://localhost:11435"
TIMEOUT = 180.0

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for current information and news.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string",
                }
            },
            "required": ["query"],
        },
    },
}

SCRAPE_TOOL = {
    "type": "function",
    "function": {
        "name": "scrape_url",
        "description": "Fetch and return the text content of a web page.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to scrape",
                }
            },
            "required": ["url"],
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(path: str, payload: dict) -> dict:
    with httpx.Client(timeout=TIMEOUT) as c:
        r = c.post(f"{BASE}{path}", json=payload)
        r.raise_for_status()
        return r.json()


def _get(path: str) -> dict:
    with httpx.Client(timeout=10) as c:
        r = c.get(f"{BASE}{path}")
        r.raise_for_status()
        return r.json()


def _routing_decision() -> dict | None:
    return _get("/health").get("last_routing_decision")


def _check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def _chat(model: str, messages: list[dict], tools: list[dict] | None = None,
          max_tokens: int = 200) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
    return _post("/v1/chat/completions", payload)


def _extract_tool_calls(response: dict) -> list[dict]:
    choice = response["choices"][0]
    return choice.get("message", {}).get("tool_calls") or []


def _content(response: dict) -> str:
    return response["choices"][0].get("message", {}).get("content") or ""


def _finish(response: dict) -> str:
    return response["choices"][0].get("finish_reason", "")


# ---------------------------------------------------------------------------
# Test 1 — keyword routing
# ---------------------------------------------------------------------------

def test_keyword_routing() -> bool:
    print("\n[1] Keyword routing (no tools in payload)")
    resp = _chat(
        "Auto",
        [{"role": "user", "content": "search for the latest OpenVINO release notes"}],
        max_tokens=5,
    )
    decision = _routing_decision()
    tc = (decision or {}).get("task_class")
    strategy = (decision or {}).get("strategy")
    model = (decision or {}).get("model")

    ok = True
    ok &= _check("task_class=web_search", tc == "web_search", f"got '{tc}'")
    ok &= _check("strategy=rule", strategy == "rule", f"got '{strategy}'")
    print(f"    → routed to: {model}")
    return ok


# ---------------------------------------------------------------------------
# Test 2 — has_tools routing
# ---------------------------------------------------------------------------

def test_tools_routing() -> bool:
    print("\n[2] has_tools routing (tools in payload, Auto model)")
    resp = _chat(
        "Auto",
        [{"role": "user", "content": "What is the weather in Warsaw today?"}],
        tools=[SEARCH_TOOL],
        max_tokens=5,
    )
    decision = _routing_decision()
    tc = (decision or {}).get("task_class")
    strategy = (decision or {}).get("strategy")
    model = (decision or {}).get("model")

    ok = True
    ok &= _check("task_class=web_search", tc == "web_search", f"got '{tc}'")
    ok &= _check("strategy=rule (has_tools)", strategy == "rule", f"got '{strategy}'")
    print(f"    → routed to: {model}")
    return ok


# ---------------------------------------------------------------------------
# Test 3 — tool call generation
# ---------------------------------------------------------------------------

def test_tool_call_generation(model: str) -> bool:
    print(f"\n[3] Tool call generation ({model})")
    t0 = time.perf_counter()
    resp = _chat(
        model,
        [{"role": "user", "content": "What are the top Python packages released this week? Use the search tool."}],
        tools=[SEARCH_TOOL],
        max_tokens=150,
    )
    elapsed = time.perf_counter() - t0
    finish = _finish(resp)
    tool_calls = _extract_tool_calls(resp)
    content = _content(resp)

    ok = True
    has_tc = bool(tool_calls)
    ok &= _check("finish_reason=tool_calls OR tool_calls present",
                 has_tc or finish == "tool_calls",
                 f"finish={finish}, tool_calls={len(tool_calls)}")

    if tool_calls:
        tc = tool_calls[0]
        fn = tc.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        ok &= _check("tool name = search_web", name == "search_web", f"got '{name}'")
        ok &= _check("arguments.query present", "query" in args, f"args={args}")
        if "query" in args:
            print(f"    → query: {args['query']!r}")
    else:
        print(f"    → no tool_calls; content snippet: {content[:120]!r}")
        ok = False

    print(f"    → {elapsed:.1f}s")
    return ok


# ---------------------------------------------------------------------------
# Test 4 — full tool loop (search + scrape)
# ---------------------------------------------------------------------------

def test_tool_loop(model: str) -> bool:
    print(f"\n[4] Full tool loop ({model})")

    # Fake tool results
    fake_search_result = json.dumps({
        "results": [
            {"title": "OpenVINO 2026.1 Released", "url": "https://example.com/ov2026",
             "snippet": "Intel releases OpenVINO 2026.1 with improved INT4 support and GenAI API."},
            {"title": "Performance benchmarks", "url": "https://example.com/bench",
             "snippet": "New benchmarks show 30% improvement in LLM inference throughput."},
        ]
    })
    fake_scrape_result = (
        "OpenVINO 2026.1 release highlights: INT4 symmetric quantization now fully supported "
        "on Intel Arc GPUs. The GenAI pipeline API has been extended with new streaming options. "
        "Throughput improvements of up to 35% for 7B-14B parameter models."
    )

    messages: list[dict] = [
        {"role": "user",
         "content": "Find information about the latest OpenVINO release and summarise the key improvements."}
    ]

    ok = True
    t0 = time.perf_counter()

    # Turn 1: model should call search_web
    resp1 = _chat(model, messages, tools=[SEARCH_TOOL, SCRAPE_TOOL], max_tokens=150)
    tc1 = _extract_tool_calls(resp1)

    if not tc1:
        _check("turn 1: model issued tool call", False, f"content={_content(resp1)[:100]!r}")
        return False

    fn1 = tc1[0].get("function", {})
    _check("turn 1: called search_web or scrape_url",
           fn1.get("name") in ("search_web", "scrape_url"),
           f"called {fn1.get('name')!r}")
    print(f"    → turn 1 tool: {fn1.get('name')}({fn1.get('arguments','')[:60]})")

    # Append assistant turn + tool result
    messages.append({"role": "assistant", "content": None, "tool_calls": tc1})
    tool_result = fake_search_result if fn1.get("name") == "search_web" else fake_scrape_result
    messages.append({
        "role": "tool",
        "tool_call_id": tc1[0]["id"],
        "content": tool_result,
    })

    # Turn 2: model should produce final answer (or another tool call)
    resp2 = _chat(model, messages, tools=[SEARCH_TOOL, SCRAPE_TOOL], max_tokens=300)
    finish2 = _finish(resp2)
    content2 = _content(resp2)
    tc2 = _extract_tool_calls(resp2)

    if tc2:
        # Model called another tool — feed one more fake result
        fn2 = tc2[0].get("function", {})
        print(f"    → turn 2 tool: {fn2.get('name')}({fn2.get('arguments','')[:60]})")
        messages.append({"role": "assistant", "content": None, "tool_calls": tc2})
        messages.append({
            "role": "tool",
            "tool_call_id": tc2[0]["id"],
            "content": fake_scrape_result,
        })
        resp3 = _chat(model, messages, tools=[SEARCH_TOOL, SCRAPE_TOOL], max_tokens=300)
        content2 = _content(resp3)
        finish2 = _finish(resp3)

    elapsed = time.perf_counter() - t0
    ok &= _check("final answer produced", bool(content2.strip()), f"finish={finish2}")
    ok &= _check("answer mentions OpenVINO",
                 "openvino" in content2.lower() or "intel" in content2.lower(),
                 "")
    print(f"    → answer ({len(content2)} chars, {elapsed:.1f}s):")
    for line in content2.strip().splitlines()[:6]:
        print(f"       {line}")
    return ok


# ---------------------------------------------------------------------------
# Test 5 — compare models
# ---------------------------------------------------------------------------

def test_compare(models: list[str]) -> None:
    print(f"\n[5] Model comparison on tool call generation")
    results: dict[str, dict] = {}
    for model in models:
        t0 = time.perf_counter()
        try:
            resp = _chat(
                model,
                [{"role": "user",
                  "content": "Search for recent news about Mistral AI. Use the search tool."}],
                tools=[SEARCH_TOOL],
                max_tokens=150,
            )
            elapsed = time.perf_counter() - t0
            tc = _extract_tool_calls(resp)
            fn = tc[0].get("function", {}) if tc else {}
            try:
                args = json.loads(fn.get("arguments", "{}")) if fn else {}
            except json.JSONDecodeError:
                args = {}
            results[model] = {
                "tool_calls": len(tc),
                "name": fn.get("name", "—"),
                "query": args.get("query", "—"),
                "finish": _finish(resp),
                "tok": resp["usage"]["completion_tokens"],
                "elapsed": elapsed,
            }
        except Exception as exc:
            results[model] = {"error": str(exc)}

    print(f"\n  {'Model':40}  {'tool_calls':10}  {'fn_name':12}  {'tok':4}  {'t':6}  query")
    for m, r in results.items():
        if "error" in r:
            print(f"  {m:40}  ERROR: {r['error']}")
        else:
            print(f"  {m:40}  {r['tool_calls']:10}  {r['name']:12}  {r['tok']:4}  {r['elapsed']:5.1f}s  {str(r['query'])[:50]!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-14b-int4-ov",
                        help="Model to use for tool-call tests")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison between qwen3-14b and mistral")
    args = parser.parse_args()

    # Verify server is up
    try:
        health = _get("/health")
        print(f"Server: {health['status']} | profile: {health['active_profile']} | loaded: {health['loaded_models']}")
    except Exception as e:
        print(f"Server unreachable: {e}")
        sys.exit(1)

    passed = 0
    total = 0

    def run(fn, *a) -> bool:
        nonlocal passed, total
        total += 1
        ok = fn(*a)
        if ok:
            passed += 1
        return ok

    run(test_keyword_routing)
    run(test_tools_routing)
    run(test_tool_call_generation, args.model)
    run(test_tool_loop, args.model)

    if args.compare:
        test_compare(["qwen3-14b-int4-ov", "mistral-small-3.2-24b-int4-ov"])

    print(f"\n{'='*60}")
    print(f"Result: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
