"""
Routing logic: signal detection, embedding similarity, model selection.
Never import from ov_server.py. Imports: server_config, model_manager, catalogue, prompt_builder, db.
To add a new routing strategy: implement a _route_by_<name>() function and wire into chat() in ov_server.py.
"""
import asyncio
import logging
import re
from typing import Any

import model_manager
import catalogue
from prompt_builder import _text_content
from server_config import (
    _cfg,
    AVAILABLE_MODELS, AVAILABLE_VLM_MODELS,
    get_agent_model,
)

log = logging.getLogger("ov_server")

# ---------------------------------------------------------------------------
# Routing state
# ---------------------------------------------------------------------------
_last_routing_decision: dict | None = None
_routing_prompt_cache: dict[tuple[str, str], str] = {}  # keyed (scope, profile_name)

# ---------------------------------------------------------------------------
# Routing constants
# ---------------------------------------------------------------------------
COMPLEXITY_SIGNALS: tuple[str, ...] = (
    "analyze", "compare", "explain in detail", "evaluate", "critique",
    "summarize", "translate", "implement", "design", "architecture",
    "step by step", "in depth", "thoroughly", "comprehensive", "detailed",
)
SIMPLE_Q_RE = re.compile(
    r"^(what|who|when|where|how much|how many|is|are|was|were|can|does|do|did)"
    r"\b.{0,60}\??\s*$",
    re.IGNORECASE,
)

_CLOUD_DIRECTIVE_RE = re.compile(r'#(ovh|cloud)\b', re.IGNORECASE)
_TASK_DIRECTIVE_RE  = re.compile(r'#(code|document|general)\b', re.IGNORECASE)

_TASK_DIRECTIVE_MAP: dict[str, str] = {
    "code":     "code",
    "document": "document",
    "general":  "general",
}


def task_class_directive(messages: list) -> str | None:
    """Return task_class if the last user message contains #code, #document, or #general."""
    last_user = next((_text_content(m) for m in reversed(messages) if m.role == "user"), "")
    m = _TASK_DIRECTIVE_RE.search(last_user)
    return _TASK_DIRECTIVE_MAP.get(m.group(1).lower()) if m else None


async def _load_embedding_centroids() -> None:
    """Load embedding model into model_manager globals. Centroid computation handled by infergate."""
    await model_manager.get_embedding_model()
    log.info("[router] embedding model loaded")


# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------

def complexity_score(req: Any) -> float:
    """0.0 = simple, 1.0 = complex. Breaks ties within a preference tier."""
    last_user = next(
        (_text_content(m) for m in reversed(req.messages) if m.role == "user"),
        "",
    )
    words = last_user.split()
    score = 0.0
    if len(words) > 50:
        score += 0.3
    if len(words) > 150:
        score += 0.2
    hits = sum(1 for s in COMPLEXITY_SIGNALS if s in last_user.lower())
    score += min(hits * 0.15, 0.4)
    if sum(1 for m in req.messages if m.role == "user") > 4:
        score += 0.1
    if SIMPLE_Q_RE.match(last_user.strip()):
        score -= 0.3
    return max(0.0, min(1.0, score))


def _has_cloud_directive(messages: list) -> bool:
    """True if the last user message contains #ovh or #cloud."""
    last_user = next((_text_content(m) for m in reversed(messages) if m.role == "user"), "")
    return bool(_CLOUD_DIRECTIVE_RE.search(last_user))


def _fastest_from(pool: list[dict]) -> dict | None:
    return next((m for m in pool if m.get("tier") == "fast" and m.get("provider") == "loc"), None)


def _balanced_from(pool: list[dict]) -> dict | None:
    loc_balanced = [m for m in pool if m.get("provider") == "loc" and m.get("tier") == "balanced"]
    if loc_balanced:
        return loc_balanced[-1]
    loc = [m for m in pool if m.get("provider") == "loc"]
    return loc[-1] if loc else None


def _best_from(pool: list[dict]) -> dict | None:
    best = [m for m in pool if m.get("tier") == "best"]
    if best:
        return best[-1]
    return pool[-1] if pool else None


def _select_model(task_class: str, profile: dict, complexity: float = 0.0,
                  estimated_tokens: int = 0, scope_override: str | None = None,
                  pref_override: str | None = None) -> dict:
    """Return {id, provider} for the best available model given task class and profile.

    Preference escalation: fastest → balanced → best → any available.
    balanced + complexity > 0.65 promotes to best.
    scope_override: if set, replaces the global provider_scope for this call.
    pref_override: if set, replaces profile model_preference before escalation logic.
    Models not on disk (provider=loc, absent from AVAILABLE_MODELS) are skipped with a warning.
    Models whose max_context_tokens < estimated_tokens are skipped (context overflow guard).
    Falls back to AGENT_MODEL if nothing is available.
    """
    all_models: list[dict] = _cfg.get("task_classes", {}).get(task_class, {}).get("models", [])
    scope: str = scope_override or _cfg.get("provider_scope", "local")

    # Filter by scope, availability, and context limit
    available: list[dict] = []
    for m in all_models:
        if not catalogue._scope_includes(scope, m.get("provider", "loc")):
            continue
        if m.get("provider") == "loc" and m["id"] not in AVAILABLE_MODELS and m["id"] not in AVAILABLE_VLM_MODELS:
            log.warning(f"[router] '{m['id']}' not on disk — skipped (task_class='{task_class}')")
            continue
        limit = m.get("max_context_tokens", 0)
        if limit and estimated_tokens > limit:
            log.info(f"[router] '{m['id']}' context {limit}tk < prompt ~{estimated_tokens}tk — skipped")
            continue
        available.append(m)

    if not available:
        fallback_id = get_agent_model() or next(iter(AVAILABLE_MODELS), "")
        log.error(f"[router] no available models for task_class='{task_class}' scope='{scope}' — fallback '{fallback_id}'")
        return {"id": fallback_id, "provider": "loc"}

    # Effective preference: override wins, then profile, then complexity promotion
    pref = pref_override or profile.get("model_preference", "balanced")
    if not pref_override and pref == "balanced" and complexity > 0.65:
        pref = "best"

    def _pick(pool: list[dict]) -> dict | None:
        if pref == "fastest":
            return _fastest_from(pool) or _balanced_from(pool) or _best_from(pool)
        elif pref == "balanced":
            return _balanced_from(pool) or _best_from(pool)
        else:
            return _best_from(pool)

    # For "fastest" preference: prefer an already-loaded fast-tier model to avoid
    # eviction.  Restrict to fast-tier only — a loaded balanced/best model must
    # not be returned just because it happens to be in memory; that would
    # prevent the Fast profile from swapping back to qwen3-14b after Mistral
    # was loaded by Precise/Laborious.
    # For "balanced"/"best": always follow the tier hierarchy.
    if pref == "fastest":
        loaded_fast = [m for m in available
                       if m["id"] in model_manager.loaded_models and m.get("tier") == "fast"]
        chosen = _pick(loaded_fast) or _pick(available)
    else:
        chosen = _pick(available)

    if chosen is None:
        fallback_id = get_agent_model() or next(iter(AVAILABLE_MODELS), "")
        log.error(f"[router] model selection failed for task_class='{task_class}' — fallback '{fallback_id}'")
        return {"id": fallback_id, "provider": "loc"}

    return {"id": chosen["id"], "provider": chosen.get("provider", "loc")}
