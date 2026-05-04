"""
Anthropic API compatibility layer for ov_server.
New file — ov_server.py is not modified by this module.
Imported by ov_server.py to add /v1/messages routes.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import openvino_genai as ov_genai
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Pydantic models — Anthropic request format
# ---------------------------------------------------------------------------

class AnthropicCacheControl(BaseModel):
    type: str  # "ephemeral" — accepted and silently ignored (no local prompt caching)


class AnthropicContentPart(BaseModel):
    type: str
    text: Optional[str] = None
    cache_control: Optional[AnthropicCacheControl] = None
    # tool_use fields
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    # tool_result fields
    tool_use_id: Optional[str] = None
    content: Optional[Union[str, List["AnthropicContentPart"]]] = None

    model_config = ConfigDict(extra="ignore")


class AnthropicSystemBlock(BaseModel):
    type: str
    text: Optional[str] = None
    cache_control: Optional[AnthropicCacheControl] = None

    model_config = ConfigDict(extra="ignore")


class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[AnthropicContentPart]]

    model_config = ConfigDict(extra="ignore")


class AnthropicThinking(BaseModel):
    type: str           # "enabled" | "disabled"
    budget_tokens: int

    model_config = ConfigDict(extra="ignore")


class AnthropicRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[AnthropicSystemBlock]]] = None
    max_tokens: int = 1024
    temperature: float = 1.0
    stream: bool = False
    stop_sequences: Optional[List[str]] = None
    thinking: Optional[Union[bool, AnthropicThinking]] = None
    tools: Optional[List[Dict[str, Any]]] = None

    model_config = ConfigDict(extra="ignore")  # silently drop unknown fields (e.g. metadata, betas)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_thinking(param: Optional[Union[bool, AnthropicThinking]]) -> bool:
    """Resolve thinking flag from any format to bool.
    None → True (consistent with server default thinking=True)."""
    if param is None:
        return True
    if isinstance(param, bool):
        return param
    if isinstance(param, AnthropicThinking):
        return param.type == "enabled"
    return True


def _anthropic_to_messages(req: AnthropicRequest) -> list:
    """Convert AnthropicRequest to the internal Message list used by build_prompt().
    Returns plain dicts-compatible objects — actual Message model imported at call site."""
    from ov_server import Message  # imported here to avoid circular import at module level

    msgs: List[Message] = []

    if req.system:
        if isinstance(req.system, str):
            sys_text = req.system
        else:
            sys_text = " ".join(
                b.text for b in req.system if b.type == "text" and b.text
            )
        msgs.append(Message(role="system", content=sys_text))

    for m in req.messages:
        if isinstance(m.content, str):
            msgs.append(Message(role=m.role, content=m.content))
        else:
            text_parts = [p.text for p in m.content if p.type == "text" and p.text]
            tool_result = next(
                (p for p in m.content if p.type == "tool_result"), None
            )
            msgs.append(Message(
                role=m.role,
                content=" ".join(text_parts) if text_parts else "",
                tool_call_id=tool_result.tool_use_id if tool_result else None,
            ))

    return msgs


def _build_gen_config(req: AnthropicRequest) -> ov_genai.GenerationConfig:
    gc = ov_genai.GenerationConfig()
    gc.max_new_tokens = req.max_tokens
    gc.temperature = req.temperature
    gc.do_sample = req.temperature > 0
    if req.stop_sequences:
        gc.stop_strings = req.stop_sequences
    return gc
