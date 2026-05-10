"""Prompt building, output parsing, and streaming handler utilities.

No server state, no OpenVINO runtime — pure text/model processing.
Imported by ov_server.py; keep this module free of circular dependencies.
"""
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel
from transformers import AutoTokenizer

log = logging.getLogger("ov_server")


# ---------------------------------------------------------------------------
# Pydantic message models — live here so both prompt builder and server share
# a single definition with no circular import.
# ---------------------------------------------------------------------------
class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[ContentPart], None] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None  # assistant turns that invoked tools
    name: Optional[str] = None


# ---------------------------------------------------------------------------
# Content helpers
# ---------------------------------------------------------------------------
def _text_content(msg: Message) -> str:
    """Extract plain text from a message whose content may be str or a list of parts."""
    if isinstance(msg.content, list):
        return " ".join(p.text for p in msg.content if p.type == "text" and p.text)
    return msg.content or ""


def has_images(messages: List[Message]) -> bool:
    return any(
        isinstance(m.content, list) and any(p.type == "image_url" for p in m.content)
        for m in messages
    )


# ---------------------------------------------------------------------------
# VLM prompt builder
# AutoTokenizer used instead of AutoProcessor to avoid the torchvision
# dependency pulled in by Qwen2.5-VL's video processor.
# ---------------------------------------------------------------------------
def build_vlm_prompt(messages: List[Message], tokenizer: AutoTokenizer) -> str:
    msg_dicts: List[Dict[str, Any]] = []
    has_system = any(m.role == "system" for m in messages)
    if not has_system:
        msg_dicts.append({"role": "system", "content": "You are a helpful assistant."})
    for m in messages:
        if isinstance(m.content, list):
            content: Any = []
            for p in m.content:
                if p.type == "image_url":
                    content.append({"type": "image"})
                elif p.type == "text" and p.text:
                    content.append({"type": "text", "text": p.text})
        else:
            content = m.content or ""
        msg_dicts.append({"role": m.role, "content": content})
    return tokenizer.apply_chat_template(
        msg_dicts, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# LLM prompt builder — delegates tool schema injection to the tokenizer's
# Jinja template so Qwen3's native tool-call format is always correct.
# ---------------------------------------------------------------------------
def build_prompt(messages: List[Message], tokenizer: AutoTokenizer,
                 tools: Optional[List[Dict[str, Any]]] = None,
                 thinking: bool = True) -> str:
    suffix = " /no_think" if not thinking else ""
    msg_dicts: List[Dict[str, Any]] = []
    has_system = any(m.role == "system" for m in messages)
    if not has_system:
        msg_dicts.append({"role": "system", "content": f"You are a helpful assistant.{suffix}"})
    for m in messages:
        text = _text_content(m)
        d: Dict[str, Any] = {"role": m.role, "content": text}
        if m.role == "system" and not thinking and not text.endswith("/no_think"):
            d["content"] = text.rstrip() + suffix
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            d["tool_calls"] = m.tool_calls          # round-trip prior assistant tool calls
        if m.name:
            d["name"] = m.name
        msg_dicts.append(d)
    return tokenizer.apply_chat_template(
        msg_dicts,
        tools=tools,
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
        start = text.find('{', pos)
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
# OpenAI-format tool call parser — extracts <tool_call>…</tool_call> blocks
# ---------------------------------------------------------------------------
def parse_tool_calls(text: str):
    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None, text
    tool_calls = []
    for m in matches:
        try:
            data = json.loads(m)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": data["name"],
                    "arguments": json.dumps(data.get("arguments", {})),
                },
            })
        except (json.JSONDecodeError, KeyError):
            log.warning(f"Failed to parse tool_call JSON: {m[:100]}")
    remaining = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    return (tool_calls or None), remaining


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
    think_match = re.search(r'<think>(.*?)</think>', raw_text, flags=re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        return thinking, answer

    # Unclosed <think> — model hit max_tokens mid-thought
    unclosed = re.search(r'<think>(.*)', raw_text, flags=re.DOTALL)
    if unclosed:
        thinking = unclosed.group(1).strip()
        log.warning(f"Unclosed <think> block — model likely hit max_tokens mid-thought ({len(thinking)} chars)")
        answer = raw_text[:unclosed.start()].strip()
        if not answer:
            answer = "*(thinking was cut off by max_tokens limit)*"
        return thinking, answer

    return None, raw_text.strip()


def format_thinking(thinking: Optional[str], answer: str) -> str:
    if not thinking:
        return answer
    lines = thinking.replace('\n', '\n> ')
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
        """Return list of delta dicts to emit. May be empty while buffering."""
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
        """Drain any remaining buffer at end of stream."""
        return [{"content": self.buf}] if self.buf and not self.in_think else []
