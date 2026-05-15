"""Prompt building, output parsing, and streaming handler utilities.

No server state, no OpenVINO runtime — pure text/model processing.
Imported by ov_server.py; keep this module free of circular dependencies.
"""
import json
import logging
import re
import uuid
from datetime import datetime as _datetime
from typing import Any, Protocol

from pydantic import BaseModel
from transformers import AutoTokenizer

log = logging.getLogger("ov_server")


def _date_prefix() -> str:
    now = _datetime.now().astimezone()
    tz_off = now.strftime("%z")                        # e.g. "+0200"
    utc_label = f"UTC{tz_off[:3]}:{tz_off[3:]}"       # e.g. "UTC+02:00"
    return (
        f"Today is {now:%Y-%m-%d}. "
        f"Current local time: {now:%H:%M} {now.strftime('%Z')} ({utc_label})."
    )


def _server_system_prefix(thinking: bool) -> str:
    """Server-controlled prefix prepended to every system message.

    Keeps server concerns (date, thinking directive) out of the client's text.
    /no_think is Qwen3-specific; only injected on the Qwen/default path.
    """
    parts = [_date_prefix()]
    if not thinking:
        parts.append("/no_think")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Pydantic message models — live here so both prompt builder and server share
# a single definition with no circular import.
# ---------------------------------------------------------------------------
class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: dict[str, str] | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentPart] | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    name: str | None = None


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------
def _text_content(msg: Message) -> str:
    if isinstance(msg.content, list):
        return " ".join(p.text for p in msg.content if p.type == "text" and p.text)
    return msg.content or ""


def has_images(messages: list[Message]) -> bool:
    return any(
        isinstance(m.content, list) and any(p.type == "image_url" for p in m.content)
        for m in messages
    )


# ---------------------------------------------------------------------------
# VLM prompt builder
# AutoTokenizer used instead of AutoProcessor to avoid the torchvision
# dependency pulled in by Qwen2.5-VL's video processor.
# ---------------------------------------------------------------------------
def _vlm_content(m: "Message", template_is_simple: bool) -> Any:
    """Convert message content for apply_chat_template.

    Simple templates (e.g. InternVL) do plain string concatenation and cannot
    handle list content — flatten to a string with <image> placeholders.
    Rich templates (Qwen2.5-VL) accept list of typed content dicts.
    """
    if not isinstance(m.content, list):
        return m.content or ""
    if template_is_simple:
        parts: list[str] = []
        for p in m.content:
            if p.type == "image_url":
                parts.append("<image>")
            elif p.type == "text" and p.text:
                parts.append(p.text)
        return "\n".join(parts)
    # Rich template — pass typed content blocks
    result: list[dict[str, Any]] = []
    for p in m.content:
        if p.type == "image_url":
            result.append({"type": "image"})
        elif p.type == "text" and p.text:
            result.append({"type": "text", "text": p.text})
    return result


def build_vlm_prompt(messages: list[Message], tokenizer: AutoTokenizer) -> str:
    chat_tmpl = getattr(tokenizer, "chat_template", "") or ""
    # Templates that do plain string concat cannot handle list content
    template_is_simple = ("message['content']" in chat_tmpl
                          or 'message["content"]' in chat_tmpl
                          or not chat_tmpl)
    prefix = _date_prefix()
    msg_dicts: list[dict[str, Any]] = []
    has_system = any(m.role == "system" for m in messages)
    if not has_system:
        msg_dicts.append({"role": "system", "content": f"{prefix}\nYou are a helpful assistant."})
    for m in messages:
        content = _vlm_content(m, template_is_simple)
        if m.role == "system":
            client_text = content if isinstance(content, str) else _text_content(m)
            content = f"{prefix}\n{client_text}"
        msg_dicts.append({"role": m.role, "content": content})
    return tokenizer.apply_chat_template(
        msg_dicts, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Shared msg_dict builder — common to DefaultAdapter and the no-tools path
# ---------------------------------------------------------------------------
def _build_msg_dicts(
    messages: list[Message],
    thinking: bool,
) -> list[dict[str, Any]]:
    prefix = _server_system_prefix(thinking)
    msg_dicts: list[dict[str, Any]] = []
    has_system = any(m.role == "system" for m in messages)
    if not has_system:
        msg_dicts.append({"role": "system", "content": f"{prefix}\nYou are a helpful assistant."})
    for m in messages:
        text = _text_content(m)
        d: dict[str, Any] = {"role": m.role, "content": text}
        if m.role == "system":
            d["content"] = f"{prefix}\n{text}"
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            d["tool_calls"] = m.tool_calls
        if m.name:
            d["name"] = m.name
        msg_dicts.append(d)
    return msg_dicts


# ---------------------------------------------------------------------------
# ModelFamilyAdapter — one implementation per model family.
#
# Responsibilities per family:
#   max_context_tokens  — input token ceiling; server returns 400 if exceeded
#   sampling_defaults   — base temperature/top_p/repetition_penalty merged
#                         with per-request overrides before generation
#   validate_messages   — structural checks specific to the family's template;
#                         raise ValueError with a clear message on violation
#   build_prompt        — inject tool schemas into the prompt string
#   parse_tool_calls    — extract OpenAI-format tool_calls from raw output
#
# Adding a new family: implement this Protocol, add a guard in get_adapter().
# ---------------------------------------------------------------------------
class ModelFamilyAdapter(Protocol):
    max_context_tokens: int
    sampling_defaults: dict[str, float]

    def validate_messages(self, messages: list[Message]) -> None: ...

    def build_prompt(
        self,
        messages: list[Message],
        tokenizer: AutoTokenizer,
        tools: list[dict[str, Any]],
        thinking: bool,
    ) -> str: ...

    def parse_tool_calls(self, text: str) -> tuple[list[dict] | None, str]: ...


# --- Compiled patterns (module-level, reused across calls) ---
_XML_CALL_RE = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
_MISTRAL_CALL_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)(\{.*\})\s*$', re.DOTALL)


class DefaultAdapter:
    """Native apply_chat_template tool support: Qwen3, Phi-4, Llama 3, Gemma.

    Injection:     tokenizer.apply_chat_template(tools=…) via Jinja template.
    Output format: <tool_call>{"name":…, "arguments":…}</tool_call>
    Sampling:      Qwen3 recommended defaults (temp=0.7, top_p=0.8, rep=1.1).
    """

    max_context_tokens: int = 32_768
    sampling_defaults: dict[str, float] = {
        "temperature": 0.7,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
    }

    def validate_messages(self, messages: list[Message]) -> None:
        if not messages:
            raise ValueError("messages list is empty")

    def build_prompt(
        self,
        messages: list[Message],
        tokenizer: AutoTokenizer,
        tools: list[dict[str, Any]],
        thinking: bool,
    ) -> str:
        return tokenizer.apply_chat_template(
            _build_msg_dicts(messages, thinking),
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

    def parse_tool_calls(self, text: str) -> tuple[list[dict] | None, str]:
        matches = _XML_CALL_RE.findall(text)
        if not matches:
            return None, text
        tool_calls = []
        for raw in matches:
            try:
                data = json.loads(raw)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": data["name"],
                        "arguments": json.dumps(data.get("arguments", {})),
                    },
                })
            except (json.JSONDecodeError, KeyError):
                log.warning(f"Failed to parse tool_call JSON: {raw[:100]}")
        remaining = _XML_CALL_RE.sub("", text).strip()
        return (tool_calls or None), remaining


class MistralAdapter:
    """Mistral anthracite-core [SYSTEM_PROMPT][INST] tokenizer.

    Injection:     [AVAILABLE_TOOLS]{json}[/AVAILABLE_TOOLS] in system block.
    Tool results:  [TOOL_RESULTS][{"content":…}][/TOOL_RESULTS]
    Output format: function_name{"key": "value"}
    Constraint:    system message must appear at position 0 (template raises
                   otherwise); validated before prompt building.
    """

    max_context_tokens: int = 32_768
    sampling_defaults: dict[str, float] = {
        "temperature": 0.7,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    }

    def validate_messages(self, messages: list[Message]) -> None:
        if not messages:
            raise ValueError("messages list is empty")
        system_indices = [i for i, m in enumerate(messages) if m.role == "system"]
        if len(system_indices) > 1:
            raise ValueError(
                f"Mistral template does not support multiple system messages "
                f"(found at positions {system_indices})"
            )
        if system_indices and system_indices[0] != 0:
            raise ValueError(
                f"Mistral template requires the system message at position 0 "
                f"(found at position {system_indices[0]})"
            )

    def build_prompt(
        self,
        messages: list[Message],
        tokenizer: AutoTokenizer,
        tools: list[dict[str, Any]],
        thinking: bool,  # Mistral has no thinking mode; kept for interface parity
    ) -> str:
        bos = tokenizer.bos_token or "<s>"
        eos = tokenizer.eos_token or "</s>"

        client_sys = "You are a helpful assistant."
        loop_messages = messages
        if messages and messages[0].role == "system":
            client_sys = _text_content(messages[0])
            loop_messages = messages[1:]

        sys_text = f"{_date_prefix()}\n{client_sys}"
        tools_block = (
            f"[AVAILABLE_TOOLS]{json.dumps(tools, ensure_ascii=False)}[/AVAILABLE_TOOLS]"
        )
        parts = [bos, f"[SYSTEM_PROMPT]{tools_block}\n{sys_text}[/SYSTEM_PROMPT]"]

        for m in loop_messages:
            if m.role == "user":
                parts.append(f"[INST]{_text_content(m)}[/INST]")
            elif m.role == "assistant":
                if m.tool_calls:
                    for tc in m.tool_calls:
                        fn = tc.get("function", {})
                        try:
                            args = json.loads(fn.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {}
                        parts.append(
                            f'{fn["name"]}{json.dumps(args, ensure_ascii=False)}{eos}'
                        )
                else:
                    text = _text_content(m)
                    if text:
                        parts.append(f"{text}{eos}")
            elif m.role == "tool":
                content = _text_content(m)
                parts.append(
                    f"[TOOL_RESULTS]"
                    f"[{json.dumps({'content': content}, ensure_ascii=False)}]"
                    f"[/TOOL_RESULTS]"
                )

        return "".join(parts)

    def parse_tool_calls(self, text: str) -> tuple[list[dict] | None, str]:
        hit = _MISTRAL_CALL_RE.match(text.strip())
        if not hit:
            return None, text.strip()
        fn_name, args_str = hit.group(1), hit.group(2)
        try:
            args = json.loads(args_str)
            return [{
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": fn_name,
                    "arguments": json.dumps(args),
                },
            }], ""
        except json.JSONDecodeError:
            log.warning(f"Failed to parse Mistral tool_call args: {args_str[:100]}")
            return None, text.strip()


class InternVLAdapter:
    """InternVL2.5 family (InternLM2 backbone, detected via <IMG_CONTEXT> token).

    max_context_tokens: 8192 — InternVL2.5 text budget; image tokens consume
    context at ~256 tokens per tile so the effective text ceiling is lower
    for multi-image inputs.

    Sampling: InternLM2 recommended defaults (higher temperature than Qwen).

    Tool calling: InternLM2 uses <|action_start|><|plugin|> format — not yet
    implemented. build_prompt/parse_tool_calls delegate to DefaultAdapter;
    update both when InternVL2.5-26B tool calling is validated on this server.
    """

    max_context_tokens: int = 8_192
    sampling_defaults: dict[str, float] = {
        "temperature": 0.9,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
    }

    def validate_messages(self, messages: list[Message]) -> None:
        if not messages:
            raise ValueError("messages list is empty")

    def build_prompt(
        self,
        messages: list[Message],
        tokenizer: AutoTokenizer,
        tools: list[dict[str, Any]],
        thinking: bool,
    ) -> str:
        return _DEFAULT_ADAPTER.build_prompt(messages, tokenizer, tools, thinking)

    def parse_tool_calls(self, text: str) -> tuple[list[dict] | None, str]:
        return _DEFAULT_ADAPTER.parse_tool_calls(text)


# Singletons — adapters are stateless
_DEFAULT_ADAPTER = DefaultAdapter()
_MISTRAL_ADAPTER = MistralAdapter()
_INTERNVL_ADAPTER = InternVLAdapter()


def get_adapter(tokenizer: AutoTokenizer) -> DefaultAdapter | MistralAdapter | InternVLAdapter:
    """Return the ModelFamilyAdapter for this tokenizer's model family.

    Detection order matters: check most-specific signatures first.
      [SYSTEM_PROMPT] → Mistral anthracite-core
      <IMG_CONTEXT>   → InternVL2.5 (InternLM2 backbone)
      default         → Qwen3, Phi-4, Llama 3, Gemma (native Jinja tool support)
    """
    tmpl = getattr(tokenizer, "chat_template", "") or ""
    if "[SYSTEM_PROMPT]" in tmpl:
        return _MISTRAL_ADAPTER
    special = str(getattr(tokenizer, "additional_special_tokens", []))
    if "<IMG_CONTEXT>" in special:
        return _INTERNVL_ADAPTER
    return _DEFAULT_ADAPTER


# ---------------------------------------------------------------------------
# Public prompt builder
# ---------------------------------------------------------------------------
def build_prompt(
    messages: list[Message],
    tokenizer: AutoTokenizer,
    tools: list[dict[str, Any]] | None = None,
    thinking: bool = True,
) -> str:
    if tools:
        return get_adapter(tokenizer).build_prompt(messages, tokenizer, tools, thinking)
    return tokenizer.apply_chat_template(
        _build_msg_dicts(messages, thinking),
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# AnythingLLM agent JSON extractor
# The model may wrap its tool-selection JSON in prose. Scans for the first
# valid {"name":..., "arguments":...} object and returns it clean.
# ---------------------------------------------------------------------------
_agent_json_decoder = json.JSONDecoder()


def _extract_agent_json(text: str) -> str:
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            break
        try:
            obj, _ = _agent_json_decoder.raw_decode(text, start)
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                return json.dumps(obj)
        except json.JSONDecodeError:
            pass
        pos = start + 1
    return ""


# ---------------------------------------------------------------------------
# Public tool-call parser
# Delegates to the correct adapter when the tokenizer is known.
# Falls back to trying both formats in order when called without a tokenizer.
# ---------------------------------------------------------------------------
def parse_tool_calls(
    text: str,
    tokenizer: AutoTokenizer | None = None,
) -> tuple[list[dict] | None, str]:
    if tokenizer is not None:
        return get_adapter(tokenizer).parse_tool_calls(text)
    result = _DEFAULT_ADAPTER.parse_tool_calls(text)
    if result[0]:
        return result
    return _MISTRAL_ADAPTER.parse_tool_calls(text)


# ---------------------------------------------------------------------------
# Raw output decoder — openvino_genai returns various result types
# ---------------------------------------------------------------------------
def decode_result(raw) -> str:
    if isinstance(raw, str):
        return raw
    if hasattr(raw, "texts"):
        texts = raw.texts
        return texts[0] if texts else ""
    text = str(raw)
    if text.startswith("['") and text.endswith("']"):
        return text[2:-2]
    return text


# ---------------------------------------------------------------------------
# Thinking block extraction and formatting
# ---------------------------------------------------------------------------
def extract_thinking(raw_text: str):
    think_match = re.search(r"<think>(.*?)</think>", raw_text, flags=re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        return thinking, answer

    # Unclosed <think> — model hit max_tokens mid-thought
    unclosed = re.search(r"<think>(.*)", raw_text, flags=re.DOTALL)
    if unclosed:
        thinking = unclosed.group(1).strip()
        log.warning(
            f"Unclosed <think> block — model likely hit max_tokens mid-thought "
            f"({len(thinking)} chars)"
        )
        answer = raw_text[:unclosed.start()].strip()
        if not answer:
            answer = "*(thinking was cut off by max_tokens limit)*"
        return thinking, answer

    return None, raw_text.strip()


def format_thinking(thinking: str | None, answer: str) -> str:
    if not thinking:
        return answer
    lines = thinking.replace("\n", "\n> ")
    return f"> 💭 **Thinking...**\n> {lines}\n\n---\n\n{answer}"


# ---------------------------------------------------------------------------
# Streaming think-block handler
# ---------------------------------------------------------------------------
class ThinkStreamHandler:
    """Buffer <think> blocks and emit content or reasoning_content deltas.

    strategy="suppress"       — fast profile: discard think block entirely.
    strategy="separate_field" — precise/laborious: emit reasoning_content
                                (Open WebUI compatible).
    """

    def __init__(self, strategy: str = "suppress") -> None:
        self.strategy = strategy
        self.buf = ""
        self.in_think = False
        self.think_acc = ""

    def feed(self, token: str) -> list[dict]:
        self.buf += token
        out: list[dict] = []
        if not self.in_think:
            if "<think>" in self.buf:
                before, _, rest = self.buf.partition("<think>")
                if before.strip():
                    out.append({"content": before})
                self.buf, self.in_think, self.think_acc = rest, True, ""
            else:
                # Keep last 7 chars buffered — long enough to detect "<think>"
                safe, self.buf = self.buf[:-7], self.buf[-7:]
                if safe:
                    out.append({"content": safe})
        else:
            if "</think>" in self.buf:
                think_raw, _, rest = self.buf.partition("</think>")
                self.think_acc += think_raw
                self.buf, self.in_think = rest, False
                if self.strategy == "separate_field":
                    out.append({"content": "", "reasoning_content": self.think_acc})
        return out

    def flush(self) -> list[dict]:
        return [{"content": self.buf}] if self.buf and not self.in_think else []
