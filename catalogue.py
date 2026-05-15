"""
Owns the merged model catalogue: local discovery + remote provider entries with TTL caching.
Never import from ov_server.py or router.py.
Imports: server_config, model_manager.
To add a new provider: follow the _fetch_ovh_catalogue() pattern and wire into _build_catalogue().
"""
import logging
import os
import time

import httpx

import model_manager
from server_config import _cfg, AVAILABLE_MODELS, AVAILABLE_VLM_MODELS

log = logging.getLogger("ov_server")

# ---------------------------------------------------------------------------
# Model catalogue — merged local + remote model list with TTL caching.
#
# _build_catalogue(scope) is synchronous so it can be called from anywhere
# and tested without an event loop.  Remote providers are served from the
# cache; call _refresh_catalogue(scope) from an async context first to
# guarantee freshness.
# ---------------------------------------------------------------------------
_catalogue_cache: dict[str, tuple[list[dict], float]] = {}
# keyed by provider name ("ovh"), value is (entries, fetched_at_timestamp)


_TIER_RANK: dict[str, int] = {"fast": 1, "balanced": 2, "best": 3}

_VLM_KEYWORDS: tuple[str, ...] = ("vl-", "vl_", "-vl", "vision", "vlm", "multimodal", "pixtral", "internvl")


def _detect_model_type(mid: str) -> str:
    lower = mid.lower()
    return "vlm" if any(k in lower for k in _VLM_KEYWORDS) else "llm"


def _extract_ovh_pricing(raw: dict) -> dict | None:
    """Extract per-1M-token pricing from an OVH model entry and convert to PLN."""
    p = raw.get("pricing")
    if not isinstance(p, dict):
        return None
    inp_eur = p.get("input") or p.get("prompt")
    out_eur = p.get("output") or p.get("completion")
    if inp_eur is None and out_eur is None:
        return None
    rate = float(_cfg.get("eur_to_pln", 4.28))
    return {
        "input_pln":  round(float(inp_eur or 0) * rate, 2),
        "output_pln": round(float(out_eur or 0) * rate, 2),
    }


def _tier_map_for_provider(provider: str) -> dict[str, str]:
    """Return {model_id: tier} for the given provider, derived from task_classes.
    A model appearing in multiple classes gets the highest tier found (best > balanced > fast)."""
    result: dict[str, str] = {}
    for cls_cfg in _cfg.get("task_classes", {}).values():
        for m in cls_cfg.get("models", []):
            if m.get("provider") == provider:
                mid = m["id"]
                new_tier = m.get("tier", "fast")
                if _TIER_RANK.get(new_tier, 1) > _TIER_RANK.get(result.get(mid), 0):
                    result[mid] = new_tier
    return result


def _local_catalogue() -> list[dict]:
    """Catalogue entries for locally discovered LLM and VLM models."""
    tier_map = _tier_map_for_provider("loc")
    entries: list[dict] = []
    for mid in AVAILABLE_MODELS:
        entries.append({
            "id":             mid,
            "object":         "model",
            "model_type":     "llm",
            "provider":       "loc",
            "tier":           tier_map.get(mid, "fast"),
            "context_length": None,
            "pricing":        None,
            "loaded":         mid in model_manager.loaded_models,
        })
    for mid in AVAILABLE_VLM_MODELS:
        entries.append({
            "id":             mid,
            "object":         "model",
            "model_type":     "vlm",
            "provider":       "loc",
            "tier":           tier_map.get(mid, "fast"),
            "context_length": None,
            "pricing":        None,
            "loaded":         mid in model_manager.loaded_vlm_models,
        })
    return entries


async def _fetch_ovh_catalogue(spec: dict) -> list[dict]:
    """Fetch OVH /v1/models, update _catalogue_cache["ovh"], and return the result.
    On any error returns the existing cached entries (empty list if no prior fetch)."""
    ttl = spec.get("catalogue_ttl_sec", 300)
    cached_entries, fetched_at = _catalogue_cache.get("ovh", ([], 0.0))
    if cached_entries and (time.time() - fetched_at) < ttl:
        return cached_entries

    base_url = spec.get("base_url", "").rstrip("/")
    api_key  = os.environ.get(spec.get("api_key_env", ""), "")
    headers  = {"Authorization": f"Bearer {api_key}"}
    ovh_tier = _tier_map_for_provider("ovh")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{base_url}/models", headers=headers)
            resp.raise_for_status()
            raw: list[dict] = resp.json().get("data", [])

        entries: list[dict] = []
        for m in raw:
            mid = m.get("id", "")
            if not mid:
                continue
            entries.append({
                "id":             mid,
                "object":         "model",
                "model_type":     _detect_model_type(mid),
                "provider":       "ovh",
                "tier":           ovh_tier.get(mid, "best"),
                "context_length": m.get("context_length") or m.get("context_window"),
                "pricing":        _extract_ovh_pricing(m),
                "loaded":         False,
            })

        _catalogue_cache["ovh"] = (entries, time.time())
        log.info(f"[catalogue] OVH: fetched {len(entries)} models")
        return entries

    except Exception as exc:
        suffix = " — using cached result" if cached_entries else " — skipping provider"
        log.warning(f"[catalogue] OVH fetch failed: {exc}{suffix}")
        return cached_entries


def _scope_includes(scope: str, provider: str) -> bool:
    """True if *provider* is active under *scope*.
    scope values: "local" | "local+ovh" | "all".
    "all" activates every provider defined in config.providers."""
    if scope == "all":
        return provider in _cfg.get("providers", {})
    return provider in scope


_AUTO_ENTRY: dict = {
    "id":             "Auto",
    "object":         "model",
    "owned_by":       "ov-server",
    "provider":       "loc",
    "tier":           "auto",
    "context_length": None,
    "pricing":        None,
    "loaded":         True,
    "description":    "Automatic routing — server selects the best model for each request",
}


def _build_catalogue(scope: str) -> list[dict]:
    """Return merged model list for *scope*.
    Remote entries come from _catalogue_cache — call _refresh_catalogue() first
    if you need guaranteed-fresh data."""
    entries = [_AUTO_ENTRY] + _local_catalogue()
    if _scope_includes(scope, "ovh"):
        cached_entries, _ = _catalogue_cache.get("ovh", ([], 0.0))
        entries.extend(cached_entries)
    return entries


async def _refresh_catalogue(scope: str) -> None:
    """Trigger async refresh of remote catalogues whose TTL has expired."""
    providers = _cfg.get("providers", {})
    if _scope_includes(scope, "ovh") and "ovh" in providers:
        await _fetch_ovh_catalogue(providers["ovh"])
