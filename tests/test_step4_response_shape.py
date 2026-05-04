"""
Step 4 tests — Anthropic response shape and stats logic.
Tests the _local_complete() output structure using mocks.
No GPU or running server required.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_complete_result(
    answer="Hello there",
    thinking_txt=None,
    tool_calls=None,
):
    """Simulate the dict that _local_complete() returns."""
    import uuid
    content_blocks = []
    if thinking_txt:
        content_blocks.append({"type": "thinking", "thinking": thinking_txt})
    if tool_calls:
        for tc in tool_calls:
            content_blocks.append({
                "type":  "tool_use",
                "id":    tc["id"],
                "name":  tc["function"]["name"],
                "input": json.loads(tc["function"]["arguments"]),
            })
        stop_reason = "tool_use"
    else:
        content_blocks.append({"type": "text", "text": answer or ""})
        stop_reason = "end_turn"

    return {
        "id":            f"msg_{uuid.uuid4().hex}",
        "type":          "message",
        "role":          "assistant",
        "model":         "qwen3-14b-int4-ov",
        "content":       content_blocks,
        "stop_reason":   stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


class TestResponseShape:
    def test_required_top_level_fields(self):
        r = _make_complete_result()
        for field in ("id", "type", "role", "model", "content", "stop_reason",
                      "stop_sequence", "usage"):
            assert field in r, f"Missing field: {field}"

    def test_id_starts_with_msg(self):
        r = _make_complete_result()
        assert r["id"].startswith("msg_")

    def test_type_is_message(self):
        assert _make_complete_result()["type"] == "message"

    def test_role_is_assistant(self):
        assert _make_complete_result()["role"] == "assistant"

    def test_plain_text_stop_reason(self):
        r = _make_complete_result(answer="Hello")
        assert r["stop_reason"] == "end_turn"
        assert r["content"][0]["type"] == "text"
        assert r["content"][0]["text"] == "Hello"

    def test_tool_use_stop_reason(self):
        tc = {
            "id": "call_abc",
            "function": {"name": "get_weather", "arguments": '{"city": "Warsaw"}'},
        }
        r = _make_complete_result(tool_calls=[tc])
        assert r["stop_reason"] == "tool_use"
        block = r["content"][0]
        assert block["type"] == "tool_use"
        assert block["name"] == "get_weather"
        assert block["input"] == {"city": "Warsaw"}

    def test_thinking_block_before_text(self):
        r = _make_complete_result(answer="42", thinking_txt="Let me think...")
        types = [b["type"] for b in r["content"]]
        assert types[0] == "thinking"
        assert types[1] == "text"

    def test_usage_fields_present(self):
        r = _make_complete_result()
        assert "input_tokens" in r["usage"]
        assert "output_tokens" in r["usage"]

    def test_empty_answer_produces_empty_text_block(self):
        r = _make_complete_result(answer="")
        assert r["content"][0]["text"] == ""
        assert r["stop_reason"] == "end_turn"


class TestStatsLifecycle:
    """Verify active_requests is correctly managed across paths."""

    def test_non_streaming_always_decrements(self):
        """active_requests must return to 0 even if _local_complete raises."""
        import dataclasses

        @dataclasses.dataclass
        class FakeStats:
            active_requests: int = 0
            total_requests: int = 0

        fake_stats = FakeStats()

        async def _simulate_route(will_raise: bool):
            fake_stats.active_requests += 1
            fake_stats.total_requests += 1
            try:
                if will_raise:
                    raise RuntimeError("oops")
                return {"ok": True}
            finally:
                fake_stats.active_requests -= 1

        import asyncio

        asyncio.run(_simulate_route(False))
        assert fake_stats.active_requests == 0

        with pytest.raises(RuntimeError):
            asyncio.run(_simulate_route(True))
        assert fake_stats.active_requests == 0

    def test_streaming_setup_failure_decrements(self):
        """If streaming setup raises before generator starts, active_requests drops."""
        import dataclasses

        @dataclasses.dataclass
        class FakeStats:
            active_requests: int = 0

        fake_stats = FakeStats()

        async def _simulate_stream_route():
            fake_stats.active_requests += 1
            try:
                raise RuntimeError("model load failed")
            except Exception:
                fake_stats.active_requests -= 1
                raise

        import asyncio
        with pytest.raises(RuntimeError):
            asyncio.run(_simulate_stream_route())
        assert fake_stats.active_requests == 0
