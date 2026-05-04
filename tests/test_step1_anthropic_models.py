"""
Step 1 tests — Anthropic Pydantic models and conversion helpers.
All tests are pure-Python, no server or GPU required.
"""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from typing import Optional, Union, List

# Stub Message to avoid importing the full ov_server (which initialises GPU/models).
class _StubMessage(BaseModel):
    role: str
    content: Union[str, None] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

import anthropic_layer  # noqa: E402

from anthropic_layer import (
    AnthropicRequest,
    AnthropicContentPart,
    AnthropicSystemBlock,
    AnthropicThinking,
    _resolve_thinking,
)

# Patch Message import inside anthropic_layer so no GPU init happens.
anthropic_layer._stub_message_cls = _StubMessage


def _anthropic_to_messages(req):
    """Wrapper that patches the ov_server.Message import."""
    import anthropic_layer as al
    with patch.dict("sys.modules", {"ov_server": MagicMock(Message=_StubMessage)}):
        return al._anthropic_to_messages(req)


# ---------------------------------------------------------------------------
# Fixtures — realistic Claude Code payloads
# ---------------------------------------------------------------------------

SIMPLE_TEXT_PAYLOAD = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Hello, what is 2+2?"}
    ],
}

SYSTEM_STRING_PAYLOAD = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 512,
    "system": "You are a helpful assistant.",
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
}

SYSTEM_BLOCK_PAYLOAD = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 512,
    "system": [
        {"type": "text", "text": "You are a helpful assistant.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Be concise."}
    ],
    "messages": [
        {"role": "user", "content": "Hello"}
    ],
}

MULTIPART_USER_PAYLOAD = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 1024,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this?", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Describe it."}
            ]
        }
    ],
}

TOOL_RESULT_PAYLOAD = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Use the tool"},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_abc123",
                    "content": "42"
                }
            ]
        }
    ],
}

THINKING_DICT_PAYLOAD = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 2048,
    "thinking": {"type": "enabled", "budget_tokens": 1024},
    "messages": [{"role": "user", "content": "Solve this hard problem"}],
}

UNKNOWN_FIELDS_PAYLOAD = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 512,
    "messages": [{"role": "user", "content": "Hi"}],
    "metadata": {"user_id": "u_123"},   # unknown — must be silently dropped
    "betas": ["interleaved-thinking-2025-05-14"],  # unknown — must be silently dropped
}


# ---------------------------------------------------------------------------
# AnthropicRequest parsing
# ---------------------------------------------------------------------------

class TestAnthropicRequestParsing:
    def test_simple_text(self):
        req = AnthropicRequest(**SIMPLE_TEXT_PAYLOAD)
        assert req.model == "claude-sonnet-4-6"
        assert req.max_tokens == 1024
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"

    def test_system_string(self):
        req = AnthropicRequest(**SYSTEM_STRING_PAYLOAD)
        assert isinstance(req.system, str)
        assert "helpful" in req.system

    def test_system_block_list(self):
        req = AnthropicRequest(**SYSTEM_BLOCK_PAYLOAD)
        assert isinstance(req.system, list)
        assert len(req.system) == 2
        assert req.system[0].type == "text"
        assert req.system[0].cache_control is not None
        assert req.system[0].cache_control.type == "ephemeral"

    def test_multipart_user_content(self):
        req = AnthropicRequest(**MULTIPART_USER_PAYLOAD)
        parts = req.messages[0].content
        assert isinstance(parts, list)
        assert len(parts) == 2
        assert parts[0].cache_control is not None

    def test_unknown_fields_silently_dropped(self):
        req = AnthropicRequest(**UNKNOWN_FIELDS_PAYLOAD)
        assert not hasattr(req, "metadata")
        assert not hasattr(req, "betas")

    def test_thinking_dict_format(self):
        req = AnthropicRequest(**THINKING_DICT_PAYLOAD)
        assert isinstance(req.thinking, AnthropicThinking)
        assert req.thinking.type == "enabled"
        assert req.thinking.budget_tokens == 1024

    def test_defaults(self):
        req = AnthropicRequest(**SIMPLE_TEXT_PAYLOAD)
        assert req.stream is False
        assert req.temperature == 1.0
        assert req.tools is None
        assert req.thinking is None


# ---------------------------------------------------------------------------
# _resolve_thinking
# ---------------------------------------------------------------------------

class TestResolveThinking:
    def test_none_returns_true(self):
        assert _resolve_thinking(None) is True

    def test_bool_true(self):
        assert _resolve_thinking(True) is True

    def test_bool_false(self):
        assert _resolve_thinking(False) is False

    def test_thinking_obj_enabled(self):
        t = AnthropicThinking(type="enabled", budget_tokens=512)
        assert _resolve_thinking(t) is True

    def test_thinking_obj_disabled(self):
        t = AnthropicThinking(type="disabled", budget_tokens=0)
        assert _resolve_thinking(t) is False


# ---------------------------------------------------------------------------
# _anthropic_to_messages
# ---------------------------------------------------------------------------

class TestAnthropicToMessages:
    def test_simple_user(self):
        req = AnthropicRequest(**SIMPLE_TEXT_PAYLOAD)
        msgs = _anthropic_to_messages(req)
        assert msgs[-1].role == "user"
        assert "2+2" in msgs[-1].content

    def test_system_string_becomes_system_message(self):
        req = AnthropicRequest(**SYSTEM_STRING_PAYLOAD)
        msgs = _anthropic_to_messages(req)
        assert msgs[0].role == "system"
        assert "helpful" in msgs[0].content

    def test_system_blocks_concatenated(self):
        req = AnthropicRequest(**SYSTEM_BLOCK_PAYLOAD)
        msgs = _anthropic_to_messages(req)
        assert msgs[0].role == "system"
        assert "helpful" in msgs[0].content
        assert "concise" in msgs[0].content

    def test_multipart_text_joined(self):
        req = AnthropicRequest(**MULTIPART_USER_PAYLOAD)
        msgs = _anthropic_to_messages(req)
        user_msg = next(m for m in msgs if m.role == "user")
        assert "What is in this?" in user_msg.content
        assert "Describe it." in user_msg.content

    def test_tool_result_sets_tool_call_id(self):
        req = AnthropicRequest(**TOOL_RESULT_PAYLOAD)
        msgs = _anthropic_to_messages(req)
        tool_msgs = [m for m in msgs if m.tool_call_id is not None]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "toolu_abc123"
