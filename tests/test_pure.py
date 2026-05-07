"""
Unit tests for pure functions in ov_server.py.
No GPU, no model loading, no network calls required.
conftest.py stubs out openvino_genai / transformers / optimum before import.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import logging
import pytest

import ov_server
from ov_server import (
    ContentPart,
    Message,
    _discover_models,
    _discover_vlm_models,
    _extract_agent_json,
    _has_images,
    _limit_image_history,
    _load_config,
    _pick_backend_name,
    _text_content,
    _validate_config,
    decode_result,
    extract_thinking,
    format_thinking,
    parse_tool_calls,
    vram_free_gb,
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
        with patch.object(ov_server, "_TOTAL_VRAM_GB", None):
            assert vram_free_gb() is None

    def test_free_equals_total_minus_allocated(self):
        with (
            patch.object(ov_server, "_TOTAL_VRAM_GB", 24.0),
            patch.object(ov_server, "_vram_allocated", {"model-a": 8.0, "model-b": 6.0}),
        ):
            assert abs(vram_free_gb() - 10.0) < 0.001

    def test_no_allocations(self):
        with (
            patch.object(ov_server, "_TOTAL_VRAM_GB", 16.0),
            patch.object(ov_server, "_vram_allocated", {}),
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
        with patch.object(ov_server, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        assert "models_dir" in cfg
        assert "device" in cfg
        assert "max_loaded_models" in cfg

    def test_overrides_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"device": "CPU", "max_loaded_models": 3}, f)
            fpath = Path(f.name)
        try:
            with patch.object(ov_server, "_CONFIG_FILE", fpath):
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
            with patch.object(ov_server, "_CONFIG_FILE", fpath):
                cfg = _load_config()
            assert "max_loaded_models" in cfg
        finally:
            fpath.unlink()

    def test_invalid_json_falls_back_to_defaults(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{ not valid json }")
            fpath = Path(f.name)
        try:
            with patch.object(ov_server, "_CONFIG_FILE", fpath):
                cfg = _load_config()
            assert "device" in cfg
        finally:
            fpath.unlink()

    def test_new_routing_keys_present_in_defaults(self):
        with patch.object(ov_server, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        for key in ("provider_scope", "active_profile", "providers",
                    "assessor", "router", "profiles", "task_classes"):
            assert key in cfg, f"Missing default for '{key}'"

    def test_provider_scope_default_is_local(self):
        with patch.object(ov_server, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        assert cfg["provider_scope"] == "local"

    def test_active_profile_default_is_fast(self):
        with patch.object(ov_server, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        assert cfg["active_profile"] == "fast"

    def test_assessor_block_has_model_and_kv(self):
        with patch.object(ov_server, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        assert "model" in cfg["assessor"]
        assert "kv_cache_size_gb" in cfg["assessor"]

    def test_router_block_has_threshold_and_keywords(self):
        with patch.object(ov_server, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        assert "embedding_threshold" in cfg["router"]
        assert "keywords" in cfg["router"]

    def test_default_profiles_shape(self):
        with patch.object(ov_server, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        for name in ("fast", "precise", "laborious"):
            p = cfg["profiles"][name]
            assert "thinking" in p
            assert "max_new_tokens" in p
            assert "model_preference" in p
            assert "use_assessor" in p

    def test_task_classes_have_five_defaults(self):
        with patch.object(ov_server, "_CONFIG_FILE", Path("/nonexistent/config.json")):
            cfg = _load_config()
        for cls in ("vision", "web_search", "document", "code", "general"):
            assert cls in cfg["task_classes"], f"Missing task class '{cls}'"

    def test_active_profile_overridden_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"active_profile": "precise"}, f)
            fpath = Path(f.name)
        try:
            with patch.object(ov_server, "_CONFIG_FILE", fpath):
                cfg = _load_config()
            assert cfg["active_profile"] == "precise"
        finally:
            fpath.unlink()

    def test_assessor_model_overridden_from_file(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"assessor": {"model": "qwen3-8b-int4-ov", "kv_cache_size_gb": 2}}, f)
            fpath = Path(f.name)
        try:
            with patch.object(ov_server, "_CONFIG_FILE", fpath):
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
