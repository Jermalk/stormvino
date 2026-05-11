"""
Routing logic: signal detection, embedding similarity, model selection.
Never import from ov_server.py. Imports: server_config, model_manager, catalogue, prompt_builder, db.
To add a new routing strategy: implement a _route_by_<name>() function and wire into chat() in ov_server.py.
"""
import asyncio
import logging
import re
from typing import Any

import numpy as np

import db
import model_manager
import catalogue
from prompt_builder import _text_content, has_images
from server_config import (
    _cfg, _GIT_COMMIT,
    AVAILABLE_MODELS, AVAILABLE_VLM_MODELS,
    get_agent_model,
)

log = logging.getLogger("ov_server")

# ---------------------------------------------------------------------------
# Routing state
# ---------------------------------------------------------------------------
_task_class_embeddings: "dict[str, np.ndarray] | None" = None  # None=not loaded, {}=failed
_last_routing_decision: dict | None = None
_routing_prompt_cache: "dict[tuple[str, str], str]" = {}  # keyed (scope, profile_name)

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

_SIGNAL_ONLY_CLASSES: frozenset[str] = frozenset({"has_image", "has_tools"})

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


# ---------------------------------------------------------------------------
# Signal detector — fast-path routing (O(1) / O(n_keywords), always <1 ms)
# ---------------------------------------------------------------------------

def _detect_signal(req: Any) -> str | None:
    """Return task_class name if a fast-path signal fires, else None.

    Checked in priority order:
      0. hashtag directive (#code / #document / #general) → explicit override
      1. image content  → "vision"
      2. client tools   → "web_search"
      3. long context   → "document"
      4. keyword match  → task_class from router.keywords
    """
    # 0. explicit task-class directive — highest priority
    directive = task_class_directive(req.messages)
    if directive:
        return directive

    # 1. image
    if has_images(req.messages):
        return "vision"

    # 2. client-provided tools
    if req.tools:
        return "web_search"

    # 3. long context — char/4 token estimate across user+assistant only (exclude system
    #    prompt — AnythingLLM @agent system prompts are huge and would always trip this)
    router_cfg = _cfg.get("router", {})
    threshold = router_cfg.get("long_context_tokens", 4000)
    total_tokens = sum(len(_text_content(m)) for m in req.messages if m.role != "system") // 4
    if total_tokens > threshold:
        return "document"

    # 4. keyword match on last user message
    last_user_text = next(
        (_text_content(m) for m in reversed(req.messages) if m.role == "user"),
        "",
    )
    if last_user_text:
        text_lower = last_user_text.lower()
        for task_class, keywords in router_cfg.get("keywords", {}).items():
            if any(kw.lower() in text_lower for kw in keywords):
                return task_class

    return None


# ---------------------------------------------------------------------------
# Embedding similarity router — Stage 2 routing
# ---------------------------------------------------------------------------

def _compute_task_class_centroids(model: Any, tok: Any) -> "dict[str, np.ndarray]":
    """Compute L2-normalised centroid embedding for each task class.
    Uses description + optional 'examples' list from config.
    Skips task classes with binary signals (has_image, has_tools) — those are
    handled exclusively by _detect_signal() and must not appear as embedding targets."""
    centroids: dict[str, np.ndarray] = {}
    for name, cls_cfg in _cfg.get("task_classes", {}).items():
        if cls_cfg.get("signal") in _SIGNAL_ONLY_CLASSES:
            continue
        texts = []
        desc = cls_cfg.get("description", "")
        if desc:
            texts.append(desc)
        texts.extend(cls_cfg.get("examples", []))
        if not texts:
            continue
        inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        vecs = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[name] = centroid / max(norm, 1e-9)
    return centroids


async def _load_embedding_centroids() -> None:
    """Blocking startup step: load embedding model and compute task class centroids."""
    global _task_class_embeddings
    try:
        model, tok = await model_manager.get_embedding_model()
        loop = asyncio.get_running_loop()
        centroids = await loop.run_in_executor(None, _compute_task_class_centroids, model, tok)
        _task_class_embeddings = centroids
        log.info(f"[router] centroids ready for: {list(centroids.keys())}")
        cfg_classes = _cfg.get("task_classes", {})
        for tc, vec in centroids.items():
            examples = cfg_classes.get(tc, {}).get("examples", [])
            db.write_centroid_snapshot(
                commit=_GIT_COMMIT, task_class=tc,
                centroid=vec.tolist(), example_count=len(examples),
            )
    except Exception as exc:
        log.warning(f"[router] centroid computation failed ({exc}) — Stage 2 routing disabled")
        _task_class_embeddings = {}


def _route_by_embedding(query: str) -> "tuple[str, float, list[float] | None]":
    """Return (task_class, cosine_similarity, embedding_vector) for the best-matching task class.
    Returns ('general', 0.0, None) when embeddings are unavailable."""
    if not _task_class_embeddings:
        return ("general", 0.0, None)

    inputs = model_manager.emb_tokenizer(
        [query[:2048]],  # ~512-token char budget
        return_tensors="pt", padding=True, truncation=True, max_length=512,
    )
    outputs = model_manager.emb_model(**inputs)
    vec = outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
    norm = np.linalg.norm(vec)
    vec = vec / max(norm, 1e-9)

    best_class, best_score = "general", 0.0
    for task_class, centroid in _task_class_embeddings.items():
        score = float(np.dot(vec, centroid))
        if score > best_score:
            best_class, best_score = task_class, score

    min_conf: float = _cfg.get("router", {}).get("embedding_min_confidence", 0.72)
    if best_score < min_conf:
        best_class = "general"

    return (best_class, best_score, vec.tolist())


# ---------------------------------------------------------------------------
# Model selector — Stage 2/3 routing
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
