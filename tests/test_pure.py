"""
Unit tests for pure functions in ov_server.py.
No GPU, no model loading, no network calls required.
conftest.py stubs out openvino_genai / transformers / optimum before import.
"""
import json
import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

import pytest

import ov_server
import catalogue
import router
import model_manager
import server_config

from ov_server import (
    ContentPart,
    Message,
    ThinkStreamHandler,
    _VALID_SCOPES,
    _limit_image_history,
    _pick_backend_name,
    _text_content,
    _extract_agent_json,
    decode_result,
    extract_thinking,
    parse_tool_calls,
)
from catalogue import (
    _build_catalogue,
    _catalogue_cache,
    _fetch_ovh_catalogue,
    _local_catalogue,
    _refresh_catalogue,
    _scope_includes,
    _tier_map_for_provider,
    _AUTO_ENTRY,
)
from router import (
    _compute_task_class_centroids,
    _detect_signal,
    _route_by_embedding,
    _select_model,
    complexity_score,
)
from server_config import (
    _discover_models,
    _discover_vlm_models,
    _load_config,
    _validate_config,
)
from model_manager import vram_free_gb
from prompt_builder import (
    format_thinking,
    has_images as _has_images,
)


# ──────────────────────────────────────────────────────────────────────────────
# extract_thinking
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractThinking:
    def test_no_think_block(self):
        thinking, answer = extract_thinking("Hello world")
        assert thinking is None
        assert answer == "Hello world"

    def test_closed_think_block(self):
        thinking, answer = extract_thinking("<think>I am reasoning</think>The answer is 42.")
        assert thinking == "I am reasoning"
        assert answer == "The answer is 42."

    def test_multiline_think_block(self):
        raw = "<think>\nstep 1\nstep 2\n</think>\nFinal answer."
        thinking, answer = extract_thinking(raw)
        assert "step 1" in thinking
        assert "step 2" in thinking
        assert answer == "Final answer."

    def test_empty_think_block(self):
        thinking, answer = extract_thinking("<think></think>Answer here.")
        assert thinking == ""
        assert answer == "Answer here."

    def test_unclosed_think_block(self):
        raw = "Preamble. <think>Partial reasoning..."
        thinking, answer = extract_thinking(raw)
        assert "Partial reasoning" in thinking
        # answer is the text before <think> or a fallback
        assert answer  # not empty

    def test_unclosed_no_preamble(self):
        raw = "<think>Only thinking, no answer"
        thinking, answer = extract_thinking(raw)
        assert "Only thinking" in thinking
        assert answer  # fallback message, not empty

    def test_think_stripped_from_answer(self):
        raw = "<think>reasoning</think>42"
        _, answer = extract_thinking(raw)
        assert "<think>" not in answer
        assert "</think>" not in answer


# ──────────────────────────────────────────────────────────────────────────────
# format_thinking
# ──────────────────────────────────────────────────────────────────────────────

class TestFormatThinking:
    def test_no_thinking_returns_answer(self):
        assert format_thinking(None, "Hello") == "Hello"

    def test_empty_thinking_returns_answer(self):
        assert format_thinking("", "Hello") == "Hello"

    def test_with_thinking_wraps_blockquote(self):
        result = format_thinking("I thought", "The answer")
        assert "I thought" in result
        assert "The answer" in result
        assert ">" in result  # markdown blockquote

    def test_multiline_thinking_uses_blockquote(self):
        result = format_thinking("line1\nline2", "Answer")
        assert "> line1" in result or "line1" in result
        assert "Answer" in result

    def test_separator_between_thinking_and_answer(self):
        result = format_thinking("thoughts", "answer")
        assert "---" in result


# ──────────────────────────────────────────────────────────────────────────────
# decode_result
# ──────────────────────────────────────────────────────────────────────────────

class TestDecodeResult:
    def test_plain_str(self):
        assert decode_result("hello") == "hello"

    def test_texts_attribute(self):
        obj = MagicMock()
        obj.texts = ["the answer"]
        assert decode_result(obj) == "the answer"

    def test_texts_empty(self):
        obj = MagicMock()
        obj.texts = []
        assert decode_result(obj) == ""

    def test_str_repr_list_wrapper(self):
        # Some OV builds return str(DecodedResults) as "['actual text']"
        class _FakeResult:
            def __str__(self):
                return "['actual text']"
        result = decode_result(_FakeResult())
        assert result == "actual text"

    def test_str_repr_plain(self):
        class _FakeResult:
            def __str__(self):
                return "plain text"
        result = decode_result(_FakeResult())
        assert result == "plain text"


# ──────────────────────────────────────────────────────────────────────────────
# parse_tool_calls
# ──────────────────────────────────────────────────────────────────────────────

class TestParseToolCalls:
    def test_no_tool_calls(self):
        tool_calls, remaining = parse_tool_calls("Just some text.")
        assert tool_calls is None
        assert remaining == "Just some text."

    def test_single_tool_call(self):
        raw = '<tool_call>{"name": "search", "arguments": {"q": "python"}}</tool_call>'
        tool_calls, remaining = parse_tool_calls(raw)
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "search"
        assert remaining == ""

    def test_tool_call_with_surrounding_text(self):
        raw = 'Preamble. <tool_call>{"name": "calc", "arguments": {}}</tool_call> Done.'
        tool_calls, remaining = parse_tool_calls(raw)
        assert tool_calls is not None
        assert tool_calls[0]["function"]["name"] == "calc"
        assert "Preamble" in remaining or "Done" in remaining

    def test_multiple_tool_calls(self):
        raw = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        )
        tool_calls, _ = parse_tool_calls(raw)
        assert tool_calls is not None
        assert len(tool_calls) == 2
        names = {tc["function"]["name"] for tc in tool_calls}
        assert names == {"a", "b"}

    def test_invalid_json_in_tool_call(self):
        raw = "<tool_call>not json at all</tool_call>"
        tool_calls, _ = parse_tool_calls(raw)
        # Invalid JSON → skipped, result may be None or empty list
        assert tool_calls is None or tool_calls == []

    def test_tool_call_id_present(self):
        raw = '<tool_call>{"name": "fn", "arguments": {"x": 1}}</tool_call>'
        tool_calls, _ = parse_tool_calls(raw)
        assert tool_calls[0]["id"].startswith("call_")
        assert tool_calls[0]["type"] == "function"

    def test_arguments_serialised_as_json_string(self):
        raw = '<tool_call>{"name": "fn", "arguments": {"k": "v"}}</tool_call>'
        tool_calls, _ = parse_tool_calls(raw)
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args == {"k": "v"}


# ──────────────────────────────────────────────────────────────────────────────
# _extract_agent_json
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractAgentJson:
    def test_clean_json(self):
        text = '{"name": "search", "arguments": {"q": "test"}}'
        result = _extract_agent_json(text)
        obj = json.loads(result)
        assert obj["name"] == "search"

    def test_json_embedded_in_prose(self):
        text = 'I will use this tool: {"name": "calc", "arguments": {}} to compute.'
        result = _extract_agent_json(text)
        obj = json.loads(result)
        assert obj["name"] == "calc"

    def test_no_json(self):
        assert _extract_agent_json("No JSON here at all.") == ""

    def test_json_without_name_key(self):
        text = '{"key": "value", "other": 1}'
        assert _extract_agent_json(text) == ""

    def test_json_without_arguments_key(self):
        text = '{"name": "fn"}'
        assert _extract_agent_json(text) == ""

    def test_returns_first_valid_object(self):
        text = (
            '{"name": "first", "arguments": {}} '
            '{"name": "second", "arguments": {}}'
        )
        result = _extract_agent_json(text)
        obj = json.loads(result)
        assert obj["name"] == "first"


# ──────────────────────────────────────────────────────────────────────────────
# _text_content
# ──────────────────────────────────────────────────────────────────────────────

class TestTextContent:
    def test_str_content(self):
        msg = Message(role="user", content="Hello")
        assert _text_content(msg) == "Hello"

    def test_none_content(self):
        msg = Message(role="user", content=None)
        assert _text_content(msg) == ""

    def test_list_with_text_part(self):
        part = ContentPart(type="text", text="Hi there")
        msg = Message(role="user", content=[part])
        assert _text_content(msg) == "Hi there"

    def test_list_with_image_only(self):
        part = ContentPart(type="image_url", image_url={"url": "data:..."})
        msg = Message(role="user", content=[part])
        assert _text_content(msg) == ""

    def test_list_with_text_and_image(self):
        parts = [
            ContentPart(type="text", text="Describe this:"),
            ContentPart(type="image_url", image_url={"url": "data:..."}),
        ]
        msg = Message(role="user", content=parts)
        assert _text_content(msg) == "Describe this:"

    def test_list_multiple_text_parts(self):
        parts = [
            ContentPart(type="text", text="Hello"),
            ContentPart(type="text", text="World"),
        ]
        msg = Message(role="user", content=parts)
        result = _text_content(msg)
        assert "Hello" in result
        assert "World" in result


# ──────────────────────────────────────────────────────────────────────────────
# _has_images
# ──────────────────────────────────────────────────────────────────────────────

class TestHasImages:
    def test_no_images(self):
        msgs = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        assert not _has_images(msgs)

    def test_one_image(self):
        part = ContentPart(type="image_url", image_url={"url": "data:..."})
        msgs = [Message(role="user", content=[part])]
        assert _has_images(msgs)

    def test_str_message_no_image(self):
        msgs = [Message(role="user", content="text")]
        assert not _has_images(msgs)

    def test_mixed_messages(self):
        plain = Message(role="user", content="text")
        with_img = Message(
            role="user",
            content=[ContentPart(type="image_url", image_url={"url": "x"})],
        )
        assert _has_images([plain, with_img])

    def test_list_without_image_url_type(self):
        part = ContentPart(type="text", text="no image here")
        msgs = [Message(role="user", content=[part])]
        assert not _has_images(msgs)


# ──────────────────────────────────────────────────────────────────────────────
# _limit_image_history
# ──────────────────────────────────────────────────────────────────────────────

def _img_msg(text: str = "") -> Message:
    parts = [ContentPart(type="image_url", image_url={"url": "data:img"})]
    if text:
        parts.insert(0, ContentPart(type="text", text=text))
    return Message(role="user", content=parts)


def _txt_msg(text: str) -> Message:
    return Message(role="user", content=text)


class TestLimitImageHistory:
    def test_no_images_unchanged(self):
        msgs = [_txt_msg("a"), _txt_msg("b")]
        with patch.object(ov_server, "VLM_MAX_IMAGE_TURNS", 1):
            result = _limit_image_history(msgs)
        assert result == msgs

    def test_single_image_turn_kept(self):
        msgs = [_img_msg("look")]
        with patch.object(ov_server, "VLM_MAX_IMAGE_TURNS", 1):
            result = _limit_image_history(msgs)
        # One turn, within limit — image preserved
        assert _has_images(result)

    def test_two_image_turns_keeps_last(self):
        msgs = [_img_msg("first"), _img_msg("second")]
        with patch.object(ov_server, "VLM_MAX_IMAGE_TURNS", 1):
            result = _limit_image_history(msgs)
        # Last turn keeps image, first turn loses it
        assert not _has_images([result[0]])
        assert _has_images([result[1]])

    def test_three_image_turns_keeps_last_two(self):
        msgs = [_img_msg("a"), _img_msg("b"), _img_msg("c")]
        with patch.object(ov_server, "VLM_MAX_IMAGE_TURNS", 2):
            result = _limit_image_history(msgs)
        # First dropped, last two kept
        assert not _has_images([result[0]])
        assert _has_images([result[1]])
        assert _has_images([result[2]])

    def test_zero_limit_keeps_all(self):
        msgs = [_img_msg("a"), _img_msg("b")]
        with patch.object(ov_server, "VLM_MAX_IMAGE_TURNS", 0):
            result = _limit_image_history(msgs)
        # 0 means no limit — all kept unchanged
        assert result == msgs

    def test_non_image_turns_untouched(self):
        msgs = [_txt_msg("text1"), _img_msg("img"), _txt_msg("text2")]
        with patch.object(ov_server, "VLM_MAX_IMAGE_TURNS", 1):
            result = _limit_image_history(msgs)
        assert result[0].content == "text1"
        assert result[2].content == "text2"


# ──────────────────────────────────────────────────────────────────────────────
# _pick_backend_name
# ──────────────────────────────────────────────────────────────────────────────

class TestPickBackendName:
    def _with_routing(self, routing: dict):
        with patch.dict(ov_server._cfg, {"routing": routing}):
            yield

    def test_default_local(self):
        routing = {"default": "local", "model_map": {}}
        with patch.dict(ov_server._cfg, {"routing": routing}):
            assert _pick_backend_name("any-model") == "local"

    def test_model_in_model_map(self):
        routing = {"default": "local", "model_map": {"gpt-4": "ovh"}}
        with patch.dict(ov_server._cfg, {"routing": routing}):
            assert _pick_backend_name("gpt-4") == "ovh"

    def test_unknown_model_uses_default(self):
        routing = {"default": "ovh", "model_map": {"specific": "local"}}
        with patch.dict(ov_server._cfg, {"routing": routing}):
            assert _pick_backend_name("other-model") == "ovh"

    def test_empty_routing_config(self):
        with patch.dict(ov_server._cfg, {"routing": {}}):
            assert _pick_backend_name("any") == "local"


# ──────────────────────────────────────────────────────────────────────────────
# vram_free_gb
# ──────────────────────────────────────────────────────────────────────────────

class TestVramFreeGb:
    def test_returns_none_when_total_unknown(self):
        with patch.object(model_manager, "_TOTAL_VRAM_GB", None):
            assert vram_free_gb() is None

    def test_free_equals_total_minus_allocated(self):
        with (
            patch.object(model_manager, "_TOTAL_VRAM_GB", 24.0),
            patch.object(model_manager, "_vram_allocated", {"model-a": 8.0, "model-b": 6.0}),
        ):
            assert abs(vram_free_gb() - 10.0) < 0.001

    def test_no_allocations(self):
        with (
            patch.object(model_manager, "_TOTAL_VRAM_GB", 16.0),
            patch.object(model_manager, "_vram_allocated", {}),
        ):
            assert abs(vram_free_gb() - 16.0) < 0.001


# ──────────────────────────────────────────────────────────────────────────────
# _discover_models / _discover_vlm_models
# ──────────────────────────────────────────────────────────────────────────────

class TestDiscoverModels:
    def test_missing_dir_returns_empty(self):
        result = _discover_models(Path("/nonexistent/path/xyz"))
        assert result == {}

    def test_empty_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _discover_models(Path(tmp))
        assert result == {}

    def test_valid_llm_dir_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "my-model"
            model_dir.mkdir()
            (model_dir / "openvino_model.xml").touch()
            (model_dir / "generation_config.json").touch()
            result = _discover_models(Path(tmp))
        assert "my-model" in result
        assert result["my-model"] == str(model_dir)

    def test_dir_missing_generation_config_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "incomplete"
            model_dir.mkdir()
            (model_dir / "openvino_model.xml").touch()
            # no generation_config.json
            result = _discover_models(Path(tmp))
        assert result == {}

    def test_multiple_models_all_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("alpha", "beta", "gamma"):
                d = Path(tmp) / name
                d.mkdir()
                (d / "openvino_model.xml").touch()
                (d / "generation_config.json").touch()
            result = _discover_models(Path(tmp))
        assert set(result) == {"alpha", "beta", "gamma"}


class TestDiscoverVlmModels:
    def test_vlm_dir_requires_language_model_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            vlm_dir = Path(tmp) / "vlm"
            vlm_dir.mkdir()
            (vlm_dir / "openvino_language_model.xml").touch()
            result = _discover_vlm_models(Path(tmp))
        assert "vlm" in result

    def test_llm_dir_not_treated_as_vlm(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm_dir = Path(tmp) / "llm"
            llm_dir.mkdir()
            (llm_dir / "openvino_model.xml").touch()
            (llm_dir / "generation_config.json").touch()
            result = _discover_vlm_models(Path(tmp))
        assert result == {}


# ──────────────────────────────────────────────────────────────────────────────
# _load_config
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_defaults_when_no_file(self):
        with patch.object(server_config, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        assert "models_dir" in cfg
        assert "device" in cfg
        assert "max_loaded_models" in cfg

    def test_overrides_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"device": "CPU", "max_loaded_models": 3}, f)
            fpath = Path(f.name)
        try:
            with patch.object(server_config, "_CONFIG_FILE", fpath):
                cfg = _load_config()
            assert cfg["device"] == "CPU"
            assert cfg["max_loaded_models"] == 3
        finally:
            fpath.unlink()

    def test_defaults_preserved_for_missing_keys(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"device": "CPU"}, f)
            fpath = Path(f.name)
        try:
            with patch.object(server_config, "_CONFIG_FILE", fpath):
                cfg = _load_config()
            assert "max_loaded_models" in cfg
        finally:
            fpath.unlink()

    def test_invalid_json_falls_back_to_defaults(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{ not valid json }")
            fpath = Path(f.name)
        try:
            with patch.object(server_config, "_CONFIG_FILE", fpath):
                cfg = _load_config()
            assert "device" in cfg
        finally:
            fpath.unlink()

    def test_new_routing_keys_present_in_defaults(self):
        with patch.object(server_config, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        for key in ("provider_scope", "active_profile", "providers",
                    "assessor", "router", "profiles", "task_classes"):
            assert key in cfg, f"Missing default for '{key}'"

    def test_provider_scope_default_is_local(self):
        with patch.object(server_config, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        assert cfg["provider_scope"] == "local"

    def test_active_profile_default_is_fast(self):
        with patch.object(server_config, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        assert cfg["active_profile"] == "fast"

    def test_assessor_block_has_model_and_kv(self):
        with patch.object(server_config, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        assert "model" in cfg["assessor"]
        assert "kv_cache_size_gb" in cfg["assessor"]

    def test_router_block_has_threshold_and_keywords(self):
        with patch.object(server_config, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        assert "embedding_threshold" in cfg["router"]
        assert "keywords" in cfg["router"]

    def test_default_profiles_shape(self):
        with patch.object(server_config, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        for name in ("fast", "precise", "laborious"):
            p = cfg["profiles"][name]
            assert "thinking" in p
            assert "max_new_tokens" in p
            assert "model_preference" in p
            assert "use_assessor" in p

    def test_task_classes_have_five_defaults(self):
        with patch.object(server_config, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        for cls in ("vision", "web_search", "document", "code", "general"):
            assert cls in cfg["task_classes"], f"Missing task class '{cls}'"

    def test_active_profile_overridden_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"active_profile": "precise"}, f)
            fpath = Path(f.name)
        try:
            with patch.object(server_config, "_CONFIG_FILE", fpath):
                cfg = _load_config()
            assert cfg["active_profile"] == "precise"
        finally:
            fpath.unlink()

    def test_assessor_model_overridden_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"assessor": {"model": "qwen3-8b-int4-ov", "kv_cache_size_gb": 2}}, f)
            fpath = Path(f.name)
        try:
            with patch.object(server_config, "_CONFIG_FILE", fpath):
                cfg = _load_config()
            assert cfg["assessor"]["model"] == "qwen3-8b-int4-ov"
        finally:
            fpath.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# _validate_config
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateConfig:
    def test_known_keys_produce_no_warnings(self, caplog):
        cfg = {
            "device": "AUTO", "provider_scope": "local",
            "profiles": {}, "task_classes": {},
        }
        with caplog.at_level(logging.WARNING, logger="ov_server"):
            _validate_config(cfg)
        assert "Unrecognised" not in caplog.text

    def test_unknown_key_produces_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ov_server"):
            _validate_config({"totally_unknown_key_xyz": "value"})
        assert "totally_unknown_key_xyz" in caplog.text

    def test_multiple_unknown_keys_all_warned(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ov_server"):
            _validate_config({"bad_a": 1, "bad_b": 2})
        assert "bad_a" in caplog.text
        assert "bad_b" in caplog.text

    def test_empty_config_no_warnings(self, caplog):
        with caplog.at_level(logging.WARNING, logger="ov_server"):
            _validate_config({})
        assert "Unrecognised" not in caplog.text

    def test_legacy_compat_keys_not_warned(self, caplog):
        cfg = {"default_model": "x", "agent_model": "y", "routing": {}}
        with caplog.at_level(logging.WARNING, logger="ov_server"):
            _validate_config(cfg)
        assert "Unrecognised" not in caplog.text


# ──────────────────────────────────────────────────────────────────────────────
# Catalogue helpers — _tier_map_for_provider, _local_catalogue, _build_catalogue
# ──────────────────────────────────────────────────────────────────────────────

_TASK_CLASSES_FIXTURE = {
    "general": {
        "description": "General",
        "models": [
            {"id": "small-llm", "provider": "loc", "tier": "fast"},
            {"id": "big-llm",   "provider": "loc", "tier": "best"},
            {"id": "cloud-llm", "provider": "ovh", "tier": "best"},
        ],
    },
    "code": {
        "description": "Code",
        "models": [
            {"id": "small-llm", "provider": "loc", "tier": "best"},  # promoted to best
        ],
    },
}


class TestScopeIncludes:
    def test_local_scope_excludes_ovh(self):
        assert not _scope_includes("local", "ovh")

    def test_local_plus_ovh_includes_ovh(self):
        assert _scope_includes("local+ovh", "ovh")

    def test_all_scope_includes_configured_provider(self):
        with patch.dict(ov_server._cfg, {"providers": {"ovh": {}}}):
            assert _scope_includes("all", "ovh")

    def test_all_scope_excludes_unconfigured_provider(self):
        with patch.dict(ov_server._cfg, {"providers": {}}):
            assert not _scope_includes("all", "ovh")

    def test_local_scope_excludes_any_remote(self):
        with patch.dict(ov_server._cfg, {"providers": {"ovh": {}}}):
            assert not _scope_includes("local", "ovh")


class TestTierMapForProvider:
    def test_loc_tier_map(self):
        with patch.dict(ov_server._cfg, {"task_classes": _TASK_CLASSES_FIXTURE}):
            m = _tier_map_for_provider("loc")
        # small-llm appears as "fast" in general and "best" in code → should be "best"
        assert m["small-llm"] == "best"
        assert m["big-llm"] == "best"
        assert "cloud-llm" not in m

    def test_ovh_tier_map(self):
        with patch.dict(ov_server._cfg, {"task_classes": _TASK_CLASSES_FIXTURE}):
            m = _tier_map_for_provider("ovh")
        assert m["cloud-llm"] == "best"
        assert "small-llm" not in m

    def test_unknown_provider_returns_empty(self):
        with patch.dict(ov_server._cfg, {"task_classes": _TASK_CLASSES_FIXTURE}):
            m = _tier_map_for_provider("ext")
        assert m == {}

    def test_empty_task_classes(self):
        with patch.dict(ov_server._cfg, {"task_classes": {}}):
            m = _tier_map_for_provider("loc")
        assert m == {}


class TestLocalCatalogue:
    def _run(self, available_models, available_vlm, loaded, loaded_vlm, task_classes):
        with (
            patch.object(catalogue, "AVAILABLE_MODELS", available_models),
            patch.object(catalogue, "AVAILABLE_VLM_MODELS", available_vlm),
            patch.object(model_manager, "loaded_models", loaded),
            patch.object(model_manager, "loaded_vlm_models", loaded_vlm),
            patch.dict(ov_server._cfg, {"task_classes": task_classes}),
        ):
            return _local_catalogue()

    def test_returns_list(self):
        result = self._run({"m1": "/p"}, {}, {}, {}, {})
        assert isinstance(result, list)

    def test_each_entry_has_required_fields(self):
        result = self._run({"m1": "/p"}, {}, {}, {}, {})
        entry = result[0]
        for field in ("id", "object", "provider", "tier", "context_length", "pricing", "loaded"):
            assert field in entry, f"Missing field '{field}'"

    def test_provider_is_loc(self):
        result = self._run({"m1": "/p"}, {}, {}, {}, {})
        assert all(e["provider"] == "loc" for e in result)

    def test_loaded_flag_reflects_state(self):
        result = self._run({"m1": "/p", "m2": "/q"}, {}, {"m1": object()}, {}, {})
        by_id = {e["id"]: e for e in result}
        assert by_id["m1"]["loaded"] is True
        assert by_id["m2"]["loaded"] is False

    def test_tier_from_task_classes(self):
        tc = {"code": {"models": [{"id": "m1", "provider": "loc", "tier": "best"}]}}
        result = self._run({"m1": "/p"}, {}, {}, {}, tc)
        assert result[0]["tier"] == "best"

    def test_tier_defaults_to_fast_when_not_in_task_classes(self):
        result = self._run({"unknown-model": "/p"}, {}, {}, {}, {})
        assert result[0]["tier"] == "fast"

    def test_vlm_models_included(self):
        result = self._run({}, {"vlm1": "/v"}, {}, {}, {})
        ids = [e["id"] for e in result]
        assert "vlm1" in ids

    def test_both_llm_and_vlm_present(self):
        result = self._run({"llm": "/l"}, {"vlm": "/v"}, {}, {}, {})
        ids = {e["id"] for e in result}
        assert {"llm", "vlm"} == ids

    def test_context_length_is_none(self):
        result = self._run({"m1": "/p"}, {}, {}, {}, {})
        assert result[0]["context_length"] is None

    def test_pricing_is_none(self):
        result = self._run({"m1": "/p"}, {}, {}, {}, {})
        assert result[0]["pricing"] is None


class TestBuildCatalogue:
    def _with_local(self, ids):
        entries = [{"id": i, "provider": "loc"} for i in ids]
        return patch.object(catalogue, "_local_catalogue", return_value=entries)

    def test_local_scope_returns_only_local(self):
        with self._with_local(["m1", "m2"]):
            catalogue._catalogue_cache.pop("ovh", None)
            result = _build_catalogue("local")
        assert all(e["provider"] == "loc" for e in result)
        assert len(result) == 3  # Auto entry + m1 + m2

    def test_local_scope_ignores_ovh_cache(self):
        catalogue._catalogue_cache["ovh"] = ([{"id": "cloud", "provider": "ovh"}], time.time())
        with self._with_local(["m1"]):
            result = _build_catalogue("local")
        catalogue._catalogue_cache.pop("ovh", None)
        assert len(result) == 2  # Auto entry + m1
        assert any(e["id"] == "m1" for e in result)

    def test_ovh_scope_includes_cached_ovh(self):
        catalogue._catalogue_cache["ovh"] = ([{"id": "cloud", "provider": "ovh"}], time.time())
        with self._with_local(["m1"]):
            result = _build_catalogue("local+ovh")
        catalogue._catalogue_cache.pop("ovh", None)
        ids = {e["id"] for e in result}
        assert ids == {"Auto", "m1", "cloud"}

    def test_all_scope_includes_cached_ovh(self):
        catalogue._catalogue_cache["ovh"] = ([{"id": "cloud", "provider": "ovh"}], time.time())
        # "all" resolves via config.providers — ensure ovh is present
        with (
            self._with_local(["m1"]),
            patch.dict(ov_server._cfg, {"providers": {"ovh": {}}}),
        ):
            result = _build_catalogue("all")
        catalogue._catalogue_cache.pop("ovh", None)
        ids = {e["id"] for e in result}
        assert "cloud" in ids

    def test_ovh_scope_with_empty_cache_returns_local_only(self):
        catalogue._catalogue_cache.pop("ovh", None)
        with self._with_local(["m1"]):
            result = _build_catalogue("local+ovh")
        assert len(result) == 2  # Auto entry + m1

    def test_empty_available_models_returns_empty(self):
        catalogue._catalogue_cache.pop("ovh", None)
        with patch.object(catalogue, "_local_catalogue", return_value=[]):
            result = _build_catalogue("local")
        assert result == [_AUTO_ENTRY]


# ──────────────────────────────────────────────────────────────────────────────
# _fetch_ovh_catalogue (async)
# ──────────────────────────────────────────────────────────────────────────────

pytestmark_anyio = pytest.mark.anyio


@pytest.mark.anyio
class TestFetchOvhCatalogue:
    def setup_method(self):
        catalogue._catalogue_cache.pop("ovh", None)

    def _mock_response(self, model_ids: list[str]):
        data = [{"id": mid, "object": "model"} for mid in model_ids]
        resp = MagicMock()
        resp.json.return_value = {"data": data}
        resp.raise_for_status = MagicMock()
        return resp

    async def test_fresh_fetch_populates_cache(self):
        spec = {"base_url": "https://fake.ovh/v1", "api_key_env": "", "catalogue_ttl_sec": 300}
        resp = self._mock_response(["ModelA", "ModelB"])
        with patch("catalogue.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__.return_value.get = AsyncMock(return_value=resp)
            result = await _fetch_ovh_catalogue(spec)
        assert len(result) == 2
        assert any(e["id"] == "ModelA" for e in result)
        assert "ovh" in catalogue._catalogue_cache

    async def test_provider_is_ovh(self):
        spec = {"base_url": "https://fake.ovh/v1", "api_key_env": "", "catalogue_ttl_sec": 300}
        resp = self._mock_response(["ModelA"])
        with patch("catalogue.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__.return_value.get = AsyncMock(return_value=resp)
            result = await _fetch_ovh_catalogue(spec)
        assert all(e["provider"] == "ovh" for e in result)

    async def test_loaded_is_always_false(self):
        spec = {"base_url": "https://fake.ovh/v1", "api_key_env": "", "catalogue_ttl_sec": 300}
        resp = self._mock_response(["ModelA"])
        with patch("catalogue.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__.return_value.get = AsyncMock(return_value=resp)
            result = await _fetch_ovh_catalogue(spec)
        assert all(e["loaded"] is False for e in result)

    async def test_cache_hit_skips_http(self):
        cached = [{"id": "cached-model", "provider": "ovh"}]
        catalogue._catalogue_cache["ovh"] = (cached, time.time())
        spec = {"base_url": "https://fake.ovh/v1", "api_key_env": "", "catalogue_ttl_sec": 300}
        with patch("catalogue.httpx.AsyncClient") as mock_cls:
            result = await _fetch_ovh_catalogue(spec)
            mock_cls.assert_not_called()
        assert result == cached
        catalogue._catalogue_cache.pop("ovh", None)

    async def test_expired_cache_triggers_refetch(self):
        old = [{"id": "old-model", "provider": "ovh"}]
        catalogue._catalogue_cache["ovh"] = (old, time.time() - 400)  # expired
        spec = {"base_url": "https://fake.ovh/v1", "api_key_env": "", "catalogue_ttl_sec": 300}
        resp = self._mock_response(["new-model"])
        with patch("catalogue.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__.return_value.get = AsyncMock(return_value=resp)
            result = await _fetch_ovh_catalogue(spec)
        assert any(e["id"] == "new-model" for e in result)
        catalogue._catalogue_cache.pop("ovh", None)

    async def test_network_error_returns_empty_when_no_cache(self):
        spec = {"base_url": "https://fake.ovh/v1", "api_key_env": "", "catalogue_ttl_sec": 300}
        with patch("catalogue.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("connection refused")
            )
            result = await _fetch_ovh_catalogue(spec)
        assert result == []

    async def test_network_error_returns_stale_cache(self):
        stale = [{"id": "stale", "provider": "ovh"}]
        catalogue._catalogue_cache["ovh"] = (stale, time.time() - 9999)  # very stale
        spec = {"base_url": "https://fake.ovh/v1", "api_key_env": "", "catalogue_ttl_sec": 300}
        with patch("catalogue.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("timeout")
            )
            result = await _fetch_ovh_catalogue(spec)
        assert result == stale
        catalogue._catalogue_cache.pop("ovh", None)

    async def test_tier_from_task_classes_applied(self):
        tc = {"code": {"models": [{"id": "ModelA", "provider": "ovh", "tier": "fast"}]}}
        spec = {"base_url": "https://fake.ovh/v1", "api_key_env": "", "catalogue_ttl_sec": 300}
        resp = self._mock_response(["ModelA"])
        with (
            patch.dict(ov_server._cfg, {"task_classes": tc}),
            patch("catalogue.httpx.AsyncClient") as mock_cls,
        ):
            mock_cls.return_value.__aenter__.return_value.get = AsyncMock(return_value=resp)
            result = await _fetch_ovh_catalogue(spec)
        assert result[0]["tier"] == "fast"
        catalogue._catalogue_cache.pop("ovh", None)

    async def test_unknown_model_defaults_to_best_tier(self):
        spec = {"base_url": "https://fake.ovh/v1", "api_key_env": "", "catalogue_ttl_sec": 300}
        resp = self._mock_response(["UnknownCloud"])
        with patch("catalogue.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__.return_value.get = AsyncMock(return_value=resp)
            result = await _fetch_ovh_catalogue(spec)
        assert result[0]["tier"] == "best"
        catalogue._catalogue_cache.pop("ovh", None)


# ──────────────────────────────────────────────────────────────────────────────
# GET /v1/models — list_models()
# ──────────────────────────────────────────────────────────────────────────────

_REQUIRED_MODEL_FIELDS = {"id", "object", "provider", "tier", "context_length", "pricing", "loaded"}

_LOCAL_CFG = {"provider_scope": "local", "task_classes": {}}
_EMPTY_MODELS = dict(
    AVAILABLE_MODELS={},
    AVAILABLE_VLM_MODELS={},
    loaded_models={},
    loaded_vlm_models={},
)


def _patch_models(**overrides):
    patches = {**_EMPTY_MODELS, **overrides}
    cat_keys = {"AVAILABLE_MODELS", "AVAILABLE_VLM_MODELS"}
    mm_keys = {"loaded_models", "loaded_vlm_models"}
    result = []
    for k, v in patches.items():
        if k in cat_keys:
            result.append(patch.object(catalogue, k, v))
        elif k in mm_keys:
            result.append(patch.object(model_manager, k, v))
    return result


@pytest.mark.anyio
class TestListModels:
    def setup_method(self):
        catalogue._catalogue_cache.pop("ovh", None)

    async def test_returns_object_list_shape(self):
        mock_refresh = AsyncMock()
        with (
            patch.object(catalogue, "_refresh_catalogue", mock_refresh),
            patch.object(catalogue, "AVAILABLE_MODELS", {"m1": "/p"}),
            patch.object(catalogue, "AVAILABLE_VLM_MODELS", {}),
            patch.object(model_manager, "loaded_models", {}),
            patch.object(model_manager, "loaded_vlm_models", {}),
            patch.dict(ov_server._cfg, _LOCAL_CFG),
        ):
            resp = await ov_server.list_models()
        assert resp["object"] == "list"
        assert isinstance(resp["data"], list)

    async def test_each_entry_has_required_fields(self):
        mock_refresh = AsyncMock()
        with (
            patch.object(catalogue, "_refresh_catalogue", mock_refresh),
            patch.object(catalogue, "AVAILABLE_MODELS", {"m1": "/p"}),
            patch.object(catalogue, "AVAILABLE_VLM_MODELS", {}),
            patch.object(model_manager, "loaded_models", {}),
            patch.object(model_manager, "loaded_vlm_models", {}),
            patch.dict(ov_server._cfg, _LOCAL_CFG),
        ):
            resp = await ov_server.list_models()
        assert resp["data"], "expected at least one entry"
        for entry in resp["data"]:
            assert _REQUIRED_MODEL_FIELDS <= entry.keys(), f"Missing fields in {entry}"

    async def test_scope_read_from_cfg(self):
        mock_refresh = AsyncMock()
        with (
            patch.object(catalogue, "_refresh_catalogue", mock_refresh),
            patch.object(catalogue, "AVAILABLE_MODELS", {}),
            patch.object(catalogue, "AVAILABLE_VLM_MODELS", {}),
            patch.object(model_manager, "loaded_models", {}),
            patch.object(model_manager, "loaded_vlm_models", {}),
            patch.dict(ov_server._cfg, {"provider_scope": "local+ovh", "task_classes": {}}),
        ):
            await ov_server.list_models()
        mock_refresh.assert_awaited_once_with("local+ovh")

    async def test_local_scope_excludes_ovh_cache(self):
        ovh_entry = {"id": "cloud-m", "object": "model", "provider": "ovh",
                     "tier": "best", "context_length": None, "pricing": None, "loaded": False}
        catalogue._catalogue_cache["ovh"] = ([ovh_entry], time.time())
        mock_refresh = AsyncMock()
        with (
            patch.object(catalogue, "_refresh_catalogue", mock_refresh),
            patch.object(catalogue, "AVAILABLE_MODELS", {}),
            patch.object(catalogue, "AVAILABLE_VLM_MODELS", {}),
            patch.object(model_manager, "loaded_models", {}),
            patch.object(model_manager, "loaded_vlm_models", {}),
            patch.dict(ov_server._cfg, _LOCAL_CFG),
        ):
            resp = await ov_server.list_models()
        assert not any(e["provider"] == "ovh" for e in resp["data"])

    async def test_local_plus_ovh_scope_includes_ovh_entries(self):
        ovh_entry = {"id": "cloud-m", "object": "model", "provider": "ovh",
                     "tier": "best", "context_length": None, "pricing": None, "loaded": False}
        catalogue._catalogue_cache["ovh"] = ([ovh_entry], time.time())
        mock_refresh = AsyncMock()
        with (
            patch.object(catalogue, "_refresh_catalogue", mock_refresh),
            patch.object(catalogue, "AVAILABLE_MODELS", {}),
            patch.object(catalogue, "AVAILABLE_VLM_MODELS", {}),
            patch.object(model_manager, "loaded_models", {}),
            patch.object(model_manager, "loaded_vlm_models", {}),
            patch.dict(ov_server._cfg, {"provider_scope": "local+ovh", "task_classes": {}}),
        ):
            resp = await ov_server.list_models()
        ids = [e["id"] for e in resp["data"]]
        assert "cloud-m" in ids
        assert any(e["provider"] == "ovh" for e in resp["data"])

    async def test_loaded_flag_reflects_loaded_models(self):
        mock_refresh = AsyncMock()
        with (
            patch.object(catalogue, "_refresh_catalogue", mock_refresh),
            patch.object(catalogue, "AVAILABLE_MODELS", {"hot": "/p", "cold": "/q"}),
            patch.object(catalogue, "AVAILABLE_VLM_MODELS", {}),
            patch.object(model_manager, "loaded_models", {"hot": object()}),
            patch.object(model_manager, "loaded_vlm_models", {}),
            patch.dict(ov_server._cfg, _LOCAL_CFG),
        ):
            resp = await ov_server.list_models()
        by_id = {e["id"]: e for e in resp["data"]}
        assert by_id["hot"]["loaded"] is True
        assert by_id["cold"]["loaded"] is False

    async def test_default_scope_is_local_when_cfg_missing(self):
        mock_refresh = AsyncMock()
        cfg_without_scope = {k: v for k, v in ov_server._cfg.items() if k != "provider_scope"}
        with (
            patch.object(catalogue, "_refresh_catalogue", mock_refresh),
            patch.object(catalogue, "AVAILABLE_MODELS", {}),
            patch.object(catalogue, "AVAILABLE_VLM_MODELS", {}),
            patch.object(model_manager, "loaded_models", {}),
            patch.object(model_manager, "loaded_vlm_models", {}),
            patch.object(ov_server, "_cfg", cfg_without_scope),
        ):
            await ov_server.list_models()
        mock_refresh.assert_awaited_once_with("local")


# ──────────────────────────────────────────────────────────────────────────────
# POST /admin/scope — set_scope()
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
class TestSetScope:
    def setup_method(self):
        catalogue._catalogue_cache.clear()
        ov_server._cfg["provider_scope"] = "local"

    def teardown_method(self):
        ov_server._cfg["provider_scope"] = "local"
        catalogue._catalogue_cache.clear()

    async def test_valid_scope_returns_200(self):
        for scope in _VALID_SCOPES:
            req = ov_server.ScopeRequest(scope=scope)
            resp = await ov_server.set_scope(req)
            assert resp.status_code == 200

    async def test_response_body_contains_new_scope(self):
        import json
        req = ov_server.ScopeRequest(scope="local+ovh")
        resp = await ov_server.set_scope(req)
        body = json.loads(resp.body)
        assert body["scope"] == "local+ovh"

    async def test_cfg_updated(self):
        req = ov_server.ScopeRequest(scope="local+ovh")
        await ov_server.set_scope(req)
        assert ov_server._cfg["provider_scope"] == "local+ovh"

    async def test_cache_cleared_on_scope_change(self):
        catalogue._catalogue_cache["ovh"] = ([], time.time())
        req = ov_server.ScopeRequest(scope="local+ovh")
        await ov_server.set_scope(req)
        assert "ovh" not in catalogue._catalogue_cache

    async def test_invalid_scope_raises_400(self):
        from fastapi import HTTPException
        req = ov_server.ScopeRequest(scope="bogus")
        with pytest.raises(HTTPException) as exc_info:
            await ov_server.set_scope(req)
        assert exc_info.value.status_code == 400

    async def test_invalid_scope_error_mentions_valid_values(self):
        from fastapi import HTTPException
        req = ov_server.ScopeRequest(scope="remote-only")
        with pytest.raises(HTTPException) as exc_info:
            await ov_server.set_scope(req)
        assert "local" in exc_info.value.detail

    async def test_all_three_valid_scopes_accepted(self):
        for scope in ("local", "local+ovh", "all"):
            req = ov_server.ScopeRequest(scope=scope)
            resp = await ov_server.set_scope(req)
            assert resp.status_code == 200
            assert ov_server._cfg["provider_scope"] == scope


# ──────────────────────────────────────────────────────────────────────────────
# _detect_signal()
# ──────────────────────────────────────────────────────────────────────────────

def _make_req(messages, tools=None):
    """Build a minimal ChatRequest for signal-detector tests."""
    return ov_server.ChatRequest(messages=messages, tools=tools)


def _user(text: str):
    return ov_server.Message(role="user", content=text)


def _assistant(text: str):
    return ov_server.Message(role="assistant", content=text)


def _image_msg():
    part = ov_server.ContentPart(
        type="image_url",
        image_url={"url": "data:image/png;base64,abc"},
    )
    return ov_server.Message(role="user", content=[part])


_ROUTER_CFG = {
    "long_context_tokens": 100,
    "keywords": {"web_search": ["search", "latest news"]},
}


class TestDetectSignal:
    def test_no_signal_returns_none(self):
        req = _make_req([_user("hello")])
        with patch.dict(ov_server._cfg, {"router": _ROUTER_CFG}):
            assert _detect_signal(req) is None

    def test_image_returns_vision(self):
        req = _make_req([_image_msg()])
        with patch.dict(ov_server._cfg, {"router": _ROUTER_CFG}):
            assert _detect_signal(req) == "vision"

    def test_tools_returns_web_search(self):
        req = _make_req([_user("what time is it")], tools=[{"type": "function"}])
        with patch.dict(ov_server._cfg, {"router": _ROUTER_CFG}):
            assert _detect_signal(req) == "web_search"

    def test_long_context_returns_document(self):
        long_text = "word " * 600          # ~600 words → ~150 tokens (char/4)
        req = _make_req([_user(long_text)])
        with patch.dict(ov_server._cfg, {"router": _ROUTER_CFG}):
            assert _detect_signal(req) == "document"

    def test_short_text_does_not_trigger_document(self):
        req = _make_req([_user("brief question")])
        with patch.dict(ov_server._cfg, {"router": _ROUTER_CFG}):
            assert _detect_signal(req) != "document"

    def test_keyword_match_returns_task_class(self):
        req = _make_req([_user("search for the latest news on AI")])
        with patch.dict(ov_server._cfg, {"router": _ROUTER_CFG}):
            assert _detect_signal(req) == "web_search"

    def test_keyword_case_insensitive(self):
        req = _make_req([_user("SEARCH for something")])
        with patch.dict(ov_server._cfg, {"router": _ROUTER_CFG}):
            assert _detect_signal(req) == "web_search"

    def test_keyword_checked_on_last_user_message_only(self):
        msgs = [_user("search for news"), _assistant("ok"), _user("thanks")]
        with patch.dict(ov_server._cfg, {"router": _ROUTER_CFG}):
            assert _detect_signal(_make_req(msgs)) is None

    def test_image_takes_priority_over_tools(self):
        req = _make_req([_image_msg()], tools=[{"type": "function"}])
        with patch.dict(ov_server._cfg, {"router": _ROUTER_CFG}):
            assert _detect_signal(req) == "vision"

    def test_tools_takes_priority_over_long_context(self):
        long_text = "word " * 600
        req = _make_req([_user(long_text)], tools=[{"type": "function"}])
        with patch.dict(ov_server._cfg, {"router": _ROUTER_CFG}):
            assert _detect_signal(req) == "web_search"

    def test_long_context_takes_priority_over_keyword(self):
        long_text = "search " * 600        # triggers both long_context and keyword
        req = _make_req([_user(long_text)])
        with patch.dict(ov_server._cfg, {"router": _ROUTER_CFG}):
            assert _detect_signal(req) == "document"

    def test_empty_keywords_list_no_match(self):
        cfg = {"long_context_tokens": 100, "keywords": {"web_search": []}}
        req = _make_req([_user("search the web")])
        with patch.dict(ov_server._cfg, {"router": cfg}):
            assert _detect_signal(req) is None

    def test_no_messages_returns_none(self):
        req = _make_req([])
        with patch.dict(ov_server._cfg, {"router": _ROUTER_CFG}):
            assert _detect_signal(req) is None

    def test_multi_turn_long_context_cumulative(self):
        # each message is short but combined they exceed threshold
        msgs = [_user("word " * 80), _assistant("word " * 80), _user("word " * 80)]
        with patch.dict(ov_server._cfg, {"router": {"long_context_tokens": 50, "keywords": {}}}):
            assert _detect_signal(_make_req(msgs)) == "document"


# ──────────────────────────────────────────────────────────────────────────────
# _compute_task_class_centroids / _route_by_embedding
# ──────────────────────────────────────────────────────────────────────────────

class _FakeHiddenState:
    """Mimics model output.last_hidden_state for test purposes."""
    def __init__(self, arr: np.ndarray):
        self._arr = arr          # shape [batch, hidden]

    def mean(self, dim):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self._arr


def _fake_model(vecs_by_text: dict[str, np.ndarray]):
    """Return a callable model that maps tokenised inputs → fake embeddings.
    vecs_by_text maps text → embedding vector. For batches, uses index order."""
    _calls = []

    class _Model:
        def __call__(self, **kwargs):
            idx = len(_calls)
            _calls.append(idx)
            # Return stacked array matching batch size derived from input_ids shape
            batch = list(vecs_by_text.values())
            arr = np.array(batch[:kwargs.get("input_ids", [[]] * 1).shape[0]
                                  if hasattr(kwargs.get("input_ids"), "shape") else 1])
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            return type("Out", (), {"last_hidden_state": _FakeHiddenState(arr)})()

    return _Model()


def _fake_tokenizer(texts_order: list[str] | None = None):
    """Minimal tokenizer mock — returns a dict with a fake input_ids tensor."""
    class _FakeTensor:
        def __init__(self, batch):
            self.shape = (batch, 10)

    class _Tok:
        def __call__(self, texts, **kwargs):
            batch = len(texts) if isinstance(texts, list) else 1
            return {"input_ids": _FakeTensor(batch)}

    return _Tok()


_TC_FIXTURE = {
    "general":    {"description": "General conversation", "models": []},
    "code":       {"description": "Code tasks", "models": []},
    "vision":     {"description": "Image understanding", "models": []},
}


class TestComputeTaskClassCentroids:
    def _run(self, task_classes, vecs):
        dim = len(next(iter(vecs.values())))
        call_idx = [0]

        class _Model:
            def __call__(self_, **kwargs):
                batch = kwargs["input_ids"].shape[0]
                arr = np.array([list(vecs.values())[call_idx[0] % len(vecs)]] * batch)
                call_idx[0] += batch
                return type("O", (), {"last_hidden_state": _FakeHiddenState(arr)})()

        with patch.dict(ov_server._cfg, {"task_classes": task_classes}):
            result = _compute_task_class_centroids(_Model(), _fake_tokenizer())
        return result

    def test_returns_dict_keyed_by_task_class(self):
        result = self._run(_TC_FIXTURE, {k: np.ones(4) for k in _TC_FIXTURE})
        assert set(result.keys()) == set(_TC_FIXTURE.keys())

    def test_centroid_is_l2_normalised(self):
        result = self._run(_TC_FIXTURE, {k: np.array([3.0, 4.0, 0.0, 0.0]) for k in _TC_FIXTURE})
        for name, vec in result.items():
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-5, f"{name} not normalised"

    def test_task_class_with_no_text_skipped(self):
        tc = {"empty": {"models": []}, "full": {"description": "Has text", "models": []}}
        result = self._run(tc, {"full": np.ones(4)})
        assert "empty" not in result
        assert "full" in result

    def test_examples_included_in_centroid(self):
        tc = {"code": {"description": "coding", "examples": ["write a function", "debug"], "models": []}}
        result = self._run(tc, {"code": np.ones(4)})
        assert "code" in result


class TestRouteByEmbedding:
    def setup_method(self):
        self._query_vec = np.array([1.0, 0.0, 0.0, 0.0])

        class _QModel:
            def __call__(self_, **kwargs):
                arr = np.array([self._query_vec])
                return type("O", (), {"last_hidden_state": _FakeHiddenState(arr)})()

        self._mock_model = _QModel()

    def _patches(self, embeddings):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(router, "_task_class_embeddings", embeddings))
        stack.enter_context(patch.object(model_manager, "emb_model", self._mock_model))
        stack.enter_context(patch.object(model_manager, "emb_tokenizer", _fake_tokenizer()))
        return stack

    def test_empty_embeddings_returns_general(self):
        with patch.object(router, "_task_class_embeddings", {}):
            cls, score, vec = _route_by_embedding("hello")
            assert cls == "general" and score == 0.0 and vec is None

    def test_none_embeddings_returns_general(self):
        with patch.object(router, "_task_class_embeddings", None):
            cls, score, vec = _route_by_embedding("hello")
            assert cls == "general" and score == 0.0 and vec is None

    def test_returns_class_with_highest_cosine(self):
        embeddings = {
            "general": np.array([0.0, 1.0, 0.0, 0.0]),
            "code":    np.array([1.0, 0.0, 0.0, 0.0]),
            "vision":  np.array([0.0, 0.0, 1.0, 0.0]),
        }
        self._query_vec = np.array([1.0, 0.0, 0.0, 0.0])
        with self._patches(embeddings):
            cls, score, vec = _route_by_embedding("some query")
        assert cls == "code"
        assert abs(score - 1.0) < 1e-5

    def test_score_is_cosine_similarity(self):
        v = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2)
        embeddings = {"general": np.array([1.0, 0.0, 0.0, 0.0])}
        self._query_vec = v
        with self._patches(embeddings):
            _, score, _ = _route_by_embedding("q")
        assert abs(score - 1 / np.sqrt(2)) < 1e-5

    def test_returns_tuple_of_str_and_float(self):
        embeddings = {"general": np.array([1.0, 0.0, 0.0, 0.0])}
        self._query_vec = np.array([1.0, 0.0, 0.0, 0.0])
        with self._patches(embeddings):
            result = _route_by_embedding("q")
        assert isinstance(result, tuple) and len(result) == 3
        assert isinstance(result[0], str)
        assert isinstance(result[1], float)
        assert isinstance(result[2], list)


# ──────────────────────────────────────────────────────────────────────────────
# complexity_score()
# ──────────────────────────────────────────────────────────────────────────────

class TestComplexityScore:
    def test_empty_messages_returns_zero(self):
        req = _make_req([])
        assert complexity_score(req) == 0.0

    def test_short_simple_question_is_low(self):
        req = _make_req([_user("What is the capital of France?")])
        assert complexity_score(req) < 0.3

    def test_long_text_raises_score(self):
        req = _make_req([_user("word " * 200)])
        assert complexity_score(req) >= 0.5

    def test_complexity_signal_raises_score(self):
        req = _make_req([_user("Please analyze and compare these two approaches in detail.")])
        assert complexity_score(req) > 0.0

    def test_multiple_signals_capped_at_04(self):
        text = "analyze compare evaluate critique summarize translate implement design"
        req = _make_req([_user(text)])
        score = complexity_score(req)
        assert score <= 1.0

    def test_simple_q_re_lowers_score(self):
        req_simple = _make_req([_user("What is Python?")])
        req_complex = _make_req([_user("analyze the Python ecosystem thoroughly")])
        assert complexity_score(req_simple) < complexity_score(req_complex)

    def test_many_user_turns_raises_score(self):
        msgs = [_user("q"), _assistant("a")] * 5 + [_user("final")]
        req = _make_req(msgs)
        assert complexity_score(req) > 0.0

    def test_score_clamped_0_to_1(self):
        req = _make_req([_user("analyze compare evaluate " * 20 + " " * 600)])
        score = complexity_score(req)
        assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# _select_model()
# ──────────────────────────────────────────────────────────────────────────────

_MODELS_FIXTURE = {
    "small-llm": "/models/small",
    "big-llm":   "/models/big",
}

_TC_SELECT_FIXTURE = {
    "general": {
        "models": [
            {"id": "small-llm", "provider": "loc", "tier": "fast"},
            {"id": "big-llm",   "provider": "loc", "tier": "best"},
            {"id": "cloud-llm", "provider": "ovh", "tier": "best"},
        ]
    },
    "vision": {
        "models": [
            {"id": "small-llm", "provider": "loc", "tier": "fast"},
        ]
    },
    "empty": {"models": []},
}

_PROFILE_FASTEST  = {"model_preference": "fastest"}
_PROFILE_BALANCED = {"model_preference": "balanced"}
_PROFILE_BEST     = {"model_preference": "best"}


def _run_select(task_class, profile, complexity=0.0, scope="local",
                available=None, available_vlm=None):
    avail = available if available is not None else _MODELS_FIXTURE
    with (
        patch.dict(ov_server._cfg, {
            "task_classes":  _TC_SELECT_FIXTURE,
            "provider_scope": scope,
        }),
        patch.object(router, "AVAILABLE_MODELS",     avail),
        patch.object(router, "AVAILABLE_VLM_MODELS", available_vlm or {}),
    ):
        return _select_model(task_class, profile, complexity)


class TestSelectModel:
    def test_fastest_picks_first_fast_loc_model(self):
        result = _run_select("general", _PROFILE_FASTEST)
        assert result["id"] == "small-llm"
        assert result["provider"] == "loc"

    def test_balanced_picks_last_loc_model(self):
        result = _run_select("general", _PROFILE_BALANCED)
        assert result["id"] == "big-llm"

    def test_best_picks_last_overall_in_local_scope(self):
        # local scope — cloud-llm filtered out, last loc model is big-llm
        result = _run_select("general", _PROFILE_BEST, scope="local")
        assert result["id"] == "big-llm"

    def test_best_with_ovh_scope_picks_cloud_model(self):
        result = _run_select("general", _PROFILE_BEST, scope="local+ovh")
        assert result["id"] == "cloud-llm"
        assert result["provider"] == "ovh"

    def test_balanced_high_complexity_promotes_to_best(self):
        result = _run_select("general", _PROFILE_BALANCED, complexity=0.8)
        # promoted to best — should pick big-llm (last loc) not cloud (local scope)
        assert result["id"] == "big-llm"

    def test_balanced_low_complexity_stays_balanced(self):
        result = _run_select("general", _PROFILE_BALANCED, complexity=0.3)
        assert result["id"] == "big-llm"   # last loc model

    def test_unavailable_local_model_skipped(self):
        # only big-llm available
        result = _run_select("general", _PROFILE_FASTEST, available={"big-llm": "/p"})
        # small-llm skipped → escalates to balanced → big-llm
        assert result["id"] == "big-llm"

    def test_empty_task_class_returns_fallback(self):
        result = _run_select("empty", _PROFILE_BEST)
        # no models → fallback (just check it doesn't crash and returns a dict)
        assert "id" in result
        assert "provider" in result

    def test_unknown_task_class_returns_fallback(self):
        result = _run_select("nonexistent", _PROFILE_BEST)
        assert "id" in result

    def test_result_shape(self):
        result = _run_select("general", _PROFILE_BALANCED)
        assert set(result.keys()) == {"id", "provider"}

    def test_local_scope_excludes_ovh_models(self):
        result = _run_select("general", _PROFILE_BEST, scope="local")
        assert result["provider"] != "ovh"

    def test_fastest_escalates_when_no_fast_model(self):
        # vision has only one fast model; if unavailable, escalates
        result = _run_select("vision", _PROFILE_FASTEST, available={})
        # no models available → fallback
        assert "id" in result


# ---------------------------------------------------------------------------
# ThinkStreamHandler
# ---------------------------------------------------------------------------

def _feed_all(handler, tokens):
    """Feed list of tokens; return all emitted delta dicts."""
    out = []
    for t in tokens:
        out.extend(handler.feed(t))
    out.extend(handler.flush())
    return out


class TestThinkStreamHandler:

    def test_no_think_block_passes_through(self):
        h = ThinkStreamHandler(strategy="suppress")
        tokens = ["Hello", " world", "!"]
        deltas = _feed_all(h, tokens)
        text = "".join(d["content"] for d in deltas)
        assert "Hello world!" in text

    def test_suppress_drops_think_block(self):
        h = ThinkStreamHandler(strategy="suppress")
        tokens = ["<think>", "reasoning here", "</think>", "Answer"]
        deltas = _feed_all(h, tokens)
        texts = [d.get("content", "") for d in deltas]
        full = "".join(texts)
        assert "reasoning here" not in full
        assert "Answer" in full

    def test_separate_field_emits_reasoning_content(self):
        h = ThinkStreamHandler(strategy="separate_field")
        tokens = ["<think>", "my reasoning", "</think>", "Answer"]
        deltas = _feed_all(h, tokens)
        reasoning = next((d.get("reasoning_content") for d in deltas if "reasoning_content" in d), None)
        assert reasoning is not None
        assert "my reasoning" in reasoning

    def test_content_after_think_emitted(self):
        h = ThinkStreamHandler(strategy="suppress")
        tokens = ["<think>skip</think>", "real answer"]
        deltas = _feed_all(h, tokens)
        texts = "".join(d.get("content", "") for d in deltas)
        assert "real answer" in texts
        assert "skip" not in texts

    def test_flush_emits_remaining_buffer(self):
        h = ThinkStreamHandler(strategy="suppress")
        h.feed("He")
        h.feed("llo")
        deltas = h.flush()
        assert any("llo" in d.get("content", "") for d in deltas)

    def test_no_think_block_separate_field_no_reasoning(self):
        h = ThinkStreamHandler(strategy="separate_field")
        deltas = _feed_all(h, ["Plain", " text"])
        assert all("reasoning_content" not in d for d in deltas)
