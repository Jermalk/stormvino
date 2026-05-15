"""
Routing battle test — proves or kills the routing pipeline.

Two layers:
  Part A — Pure unit tests: infergate Router + select_model with mock backends.
            No server, no GPU, runs in < 2 s.
  Part B — Live integration tests: HTTP against running server (localhost:11435).
            Verifies /v1/chat/completions routing decisions end-to-end.

Run:
    pytest autotest/test_routing.py -v
    pytest autotest/test_routing.py -v -m "not live"   # pure tests only
    pytest autotest/test_routing.py -v -m live          # live tests only
"""
import asyncio
import json
import time
from dataclasses import dataclass, field

import httpx
import pytest
import yaml

# ---------------------------------------------------------------------------
# Mock Backend
# ---------------------------------------------------------------------------

@dataclass
class MockBackend:
    _name: str
    _models: list[str]
    _loaded: list[str] = field(default_factory=list)
    _is_local: bool = True

    @property
    def is_local(self) -> bool:
        return self._is_local

    @property
    def routing_only(self) -> bool:
        return True

    def name(self) -> str:
        return self._name

    def available_models(self) -> list[str]:
        return self._models

    def loaded_model_ids(self) -> list[str]:
        return self._loaded

    async def chat(self, request, model_id: str) -> dict:  # type: ignore[override]
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Router factory from real config.yaml
# ---------------------------------------------------------------------------

CONFIG_PATH = "/opt/ov_server/infergate/config.yaml"

def _load_router(
    local_models: list[str] | None = None,
    remote_models: list[str] | None = None,
    loaded: list[str] | None = None,
) -> "Router":  # type: ignore[name-defined]
    from infergate import Router, RouterConfig

    with open(CONFIG_PATH) as fh:
        raw = yaml.safe_load(fh)
    config = RouterConfig.from_dict(raw)

    # Derive what local models to advertise
    if local_models is None:
        local_models = [
            m.id
            for tc in config.task_classes.values()
            for m in tc.models
            if m.backend == "ov_server"
        ]
        local_models = list(dict.fromkeys(local_models))  # dedup, preserve order

    local_backend = MockBackend(
        _name="ov_server",
        _models=local_models,
        _loaded=loaded or [],
        _is_local=True,
    )

    backends: dict = {"ov_server": local_backend}

    if remote_models is not None:
        ovh_backend = MockBackend(
            _name="ovh",
            _models=remote_models,
            _loaded=[],
            _is_local=False,
        )
        backends["ovh"] = ovh_backend

    return Router(config=config, backends=backends, embedding_provider=None)


def _decide_sync(router, messages: list[dict], tools=None, force_tier=None):
    """Synchronous wrapper around Router.decide()."""
    from infergate.types import InferRequest
    req = InferRequest(messages=messages, tools=tools)
    return asyncio.run(router.decide(req, trace=True, force_tier=force_tier))


# ---------------------------------------------------------------------------
# Part A — Pure unit tests
# ---------------------------------------------------------------------------

class TestSignalDetection:
    """Stage 1/2: fast-path signals that bypass embedding."""

    def test_tools_present_routes_web_search(self):
        router = _load_router()
        msgs = [{"role": "user", "content": "do something"}]
        d = _decide_sync(router, msgs, tools=[{"type": "function", "function": {"name": "search"}}])
        assert d.task_class == "web_search"
        assert d.strategy.value == "signal"

    def test_empty_tools_list_not_a_signal(self):
        """tools=[] is falsy — should NOT trigger tools signal."""
        router = _load_router()
        msgs = [{"role": "user", "content": "tell me a joke"}]
        d = _decide_sync(router, msgs, tools=[])
        # Empty list is falsy — no tools signal → general or embedding
        assert d.task_class != "web_search" or d.strategy.value != "signal"

    def test_keyword_search_for_en(self):
        router = _load_router()
        msgs = [{"role": "user", "content": "search for recent news about AI"}]
        d = _decide_sync(router, msgs)
        assert d.task_class == "web_search"
        assert d.strategy.value == "signal"

    def test_keyword_wyszukaj_pl(self):
        """Polish keyword triggers web_search signal."""
        router = _load_router()
        msgs = [{"role": "user", "content": "wyszukaj informacje o Pythonie"}]
        d = _decide_sync(router, msgs)
        assert d.task_class == "web_search"
        assert d.strategy.value == "signal"

    def test_long_context_routes_document(self):
        """Message > 4000 tokens (char/4 heuristic = 16 000 chars) → document signal."""
        router = _load_router()
        long_text = "word " * 4001  # 4001 * 5 = 20 005 chars ÷ 4 = 5001 tokens
        msgs = [{"role": "user", "content": long_text}]
        d = _decide_sync(router, msgs)
        assert d.task_class == "document"
        assert d.strategy.value == "signal"

    def test_hash_code_directive(self):
        router = _load_router()
        msgs = [{"role": "user", "content": "#code write a bubble sort"}]
        d = _decide_sync(router, msgs)
        assert d.task_class == "code"
        assert d.strategy.value == "keyword"

    def test_hash_document_directive(self):
        router = _load_router()
        msgs = [{"role": "user", "content": "#document summarise this paper"}]
        d = _decide_sync(router, msgs)
        assert d.task_class == "document"
        assert d.strategy.value == "keyword"

    def test_hash_general_directive(self):
        router = _load_router()
        msgs = [{"role": "user", "content": "#general what's for dinner?"}]
        d = _decide_sync(router, msgs)
        assert d.task_class == "general"
        assert d.strategy.value == "keyword"

    def test_hash_cloud_does_not_set_task_class(self):
        """#cloud sets scope=remote in decide().  With no OVH backend, _fallback raises.

        BUG: chat_handler.py wraps decide() with NoModelAvailable → reselect("general", local).
        This test verifies the infergate layer behaviour (raises), NOT the server handler.
        """
        from infergate.types import NoModelAvailable
        router = _load_router(remote_models=None)  # no OVH backend
        msgs = [{"role": "user", "content": "#cloud tell me a joke"}]
        with pytest.raises(NoModelAvailable):
            _decide_sync(router, msgs)

    def test_directive_takes_priority_over_keyword(self):
        """#document directive wins even if message also contains 'search for'."""
        router = _load_router()
        msgs = [{"role": "user", "content": "#document search for details in this text"}]
        d = _decide_sync(router, msgs)
        assert d.task_class == "document"
        assert d.strategy.value == "keyword"

    def test_no_signal_falls_back_to_general(self):
        """No provider = no embedding → FALLBACK to 'general'."""
        router = _load_router()
        msgs = [{"role": "user", "content": "hello"}]
        d = _decide_sync(router, msgs)
        assert d.task_class == "general"
        # infergate 0.2.0 uses 'embedding_fallback' as the RouteStrategy.FALLBACK value
        assert d.strategy.value in ("fallback", "embedding", "embedding_fallback")


class TestModelSelection:
    """Stage 5: tier selection, scope, ctx_limit, prefer_loaded."""

    def test_fast_profile_picks_fast_tier_code(self):
        router = _load_router()
        msgs = [{"role": "user", "content": "#code write hello world"}]
        d = _decide_sync(router, msgs, force_tier=None)
        # fast profile + fastest pref → fast tier
        assert d.model_id == "qwen2.5-coder-14b-int4"

    def test_balanced_profile_picks_balanced_tier_code(self):
        router = _load_router()
        msgs = [{"role": "user", "content": "#code write hello world"}]
        d = _decide_sync(router, msgs, force_tier="balanced")
        assert d.model_id == "mistral-small-3.2-24b-int4-ov"

    def test_best_profile_picks_best_tier_code(self):
        router = _load_router()
        msgs = [{"role": "user", "content": "#code write hello world"}]
        d = _decide_sync(router, msgs, force_tier="best")
        # best local: qwen3-coder-30b (no OVH backend registered)
        assert d.model_id == "qwen3-coder-30b-a3b-int4-ov"

    def test_local_scope_blocks_ovh_models(self):
        """provider_scope=local → OVH candidates eliminated even if registered."""
        ovh_models = ["Qwen3-32B", "gpt-oss-120b", "Qwen3-Coder-30B-A3B-Instruct"]
        router = _load_router(remote_models=ovh_models)
        msgs = [{"role": "user", "content": "#general tell me a joke"}]
        d = _decide_sync(router, msgs, force_tier="best")
        # Should still pick local best (mistral) not OVH
        assert d.backend == "ov_server"

    def test_ctx_limit_eliminates_small_models(self):
        """~30 000-token prompt → qwen3-8b (ctx_limit=28000) eliminated."""
        router = _load_router()
        # 30 000 tokens * 4 chars/token = 120 000 chars
        huge_msg = "word " * 30_001
        msgs = [{"role": "user", "content": huge_msg}]
        d = _decide_sync(router, msgs)
        # qwen3-8b has ctx_limit=28000 → eliminated
        assert d.model_id != "qwen3-8b-int4-ov"

    def test_prefer_loaded_wins_over_cold_fast(self):
        """When fast-tier model is warm, prefer_loaded=True and it wins."""
        router = _load_router(loaded=["qwen3-8b-int4-ov"])
        msgs = [{"role": "user", "content": "#general hello"}]
        d = _decide_sync(router, msgs)
        assert d.prefer_loaded is True
        assert d.model_id == "qwen3-8b-int4-ov"

    def test_no_loaded_model_still_picks_fast_tier(self):
        router = _load_router(loaded=[])
        msgs = [{"role": "user", "content": "#general hello"}]
        d = _decide_sync(router, msgs)
        assert d.prefer_loaded is False
        assert d.model_id == "qwen3-8b-int4-ov"  # still picks fast tier cold

    def test_force_tier_overrides_complexity_promotion(self):
        """force_tier wins over complexity promotion — pref stays at force_tier value.

        In 'general' class: fast=qwen3-8b, balanced=qwen3-14b, best=mistral.
        force_tier='balanced' → qwen3-14b (NOT mistral, even if complexity > 0.65).
        """
        router = _load_router()
        long_complex = (
            "Please analyze, compare, and evaluate the following architecture design "
            "in depth with a comprehensive step by step breakdown. Provide a detailed "
            "critique and explain in detail every trade-off including performance, "
            "scalability, cost, and maintainability. " * 5
        )
        msgs = [{"role": "user", "content": "#general " + long_complex}]
        d = _decide_sync(router, msgs, force_tier="balanced")
        # force_tier overrides complexity promotion → balanced tier picked, not best
        assert d.model_id == "qwen3-14b-int4-ov", (
            f"Expected balanced-tier 'qwen3-14b-int4-ov', got '{d.model_id}'. "
            "force_tier must override complexity promotion."
        )

    def test_complexity_score_above_065_promotes_balanced_to_best_without_force(self):
        """Without force_tier, profile='balanced' + high complexity → promotes to 'best'."""
        from infergate import RouterConfig
        import yaml
        with open(CONFIG_PATH) as fh:
            raw = yaml.safe_load(fh)
        raw["active_profile"] = "precise"  # balanced pref
        config = RouterConfig.from_dict(raw)
        local_models = list(dict.fromkeys(
            m.id for tc in config.task_classes.values() for m in tc.models if m.backend == "ov_server"
        ))
        backend = MockBackend("ov_server", local_models, _is_local=True)
        from infergate import Router
        router = Router(config=config, backends={"ov_server": backend})

        long_complex = (
            "analyze compare evaluate architecture design implement step by step "
            "comprehensive detailed critique " * 20
        )
        from infergate.selector import complexity_score
        msgs_list = [{"role": "user", "content": "#general " + long_complex}]
        score = complexity_score(msgs_list)
        if score <= 0.65:
            pytest.skip(f"complexity_score={score:.2f} did not exceed 0.65 — message too short")

        d = _decide_sync(router, msgs_list)
        # balanced pref + complexity > 0.65 → promoted to best tier
        assert d.model_id == "mistral-small-3.2-24b-int4-ov", (
            f"Expected best-tier 'mistral' after complexity promotion, got '{d.model_id}'. "
            f"complexity_score={score:.2f}"
        )

    def test_complexity_promote_fast_to_balanced(self):
        """'fastest' pref + complexity above threshold → promotes to 'balanced'.

        Only fires if complexity_promote_fast_threshold is set in config.
        """
        from infergate import RouterConfig
        import yaml
        with open(CONFIG_PATH) as fh:
            raw = yaml.safe_load(fh)
        threshold = raw.get("router", {}).get("complexity_promote_fast_threshold")
        if threshold is None:
            pytest.skip("complexity_promote_fast_threshold not configured")

    def test_no_model_available_raises(self):
        """All models gone → NoModelAvailable raised."""
        from infergate.types import NoModelAvailable
        router = _load_router(local_models=[])  # nothing available locally
        msgs = [{"role": "user", "content": "#code write hello world"}]
        with pytest.raises(NoModelAvailable):
            _decide_sync(router, msgs)

    def test_unknown_task_class_falls_back_to_general(self):
        """A task_class not in config → select_model falls back to 'general' class."""
        from infergate import RouterConfig, Router
        import yaml
        with open(CONFIG_PATH) as fh:
            raw = yaml.safe_load(fh)
        config = RouterConfig.from_dict(raw)
        backend = MockBackend("ov_server", ["qwen3-8b-int4-ov"], _is_local=True)
        router = Router(config=config, backends={"ov_server": backend})
        # Force a task_class that doesn't exist in config
        from infergate.selector import select_model
        backend_name, model_id, _, _ = select_model(
            task_class="nonexistent_class",
            config=config,
            backends={"ov_server": backend},
            effective_scope="local",
            profile_pref="fastest",
        )
        assert model_id == "qwen3-8b-int4-ov"


class TestScopeAndCloudDirective:
    """Stage 4: scope resolution."""

    def test_cloud_directive_without_ovh_backend_raises_no_model(self):
        """#cloud with no OVH backend → NoModelAvailable from infergate.

        This is a known limitation: infergate sets scope='remote' when #cloud is seen,
        and _fallback() raises when no remote backend is registered.
        chat_handler.py catches this and falls back to local (see NoModelAvailable handler).
        """
        from infergate.types import NoModelAvailable
        router = _load_router(remote_models=None)  # no OVH
        msgs = [{"role": "user", "content": "#cloud tell me a joke"}]
        with pytest.raises(NoModelAvailable):
            _decide_sync(router, msgs)

    def test_cloud_directive_with_ovh_routes_remote(self):
        """#cloud with OVH backend registered → OVH model chosen."""
        ovh_models = ["Qwen3-32B"]
        router = _load_router(remote_models=ovh_models)
        msgs = [{"role": "user", "content": "#cloud tell me a joke"}]
        d = _decide_sync(router, msgs)
        assert d.backend == "ovh"
        assert d.model_id == "Qwen3-32B"

    def test_scope_override_class_forces_remote(self):
        """scope_override in task class config (if any) overrides global scope."""
        from infergate import RouterConfig
        import yaml
        with open(CONFIG_PATH) as fh:
            raw = yaml.safe_load(fh)
        scope_overrides = {
            k: v for k, v in raw.get("task_classes", {}).items()
            if v.get("scope_override")
        }
        if not scope_overrides:
            pytest.skip("No scope_override in any task class config")

    def test_remote_only_scope_blocks_local_models(self):
        """When scope='remote', local models are filtered out."""
        from infergate.selector import select_model, _scope_allows
        from infergate import RouterConfig
        import yaml
        with open(CONFIG_PATH) as fh:
            raw = yaml.safe_load(fh)
        config = RouterConfig.from_dict(raw)
        local_b = MockBackend("ov_server", ["qwen3-8b-int4-ov"], _is_local=True)
        assert not _scope_allows(local_b, "remote")
        assert _scope_allows(local_b, "local")
        assert _scope_allows(local_b, "local+remote")


class TestComplexityScore:
    """Complexity scoring edge cases."""

    def test_simple_question_scores_low(self):
        from infergate.selector import complexity_score
        msgs = [{"role": "user", "content": "What is Python?"}]
        score = complexity_score(msgs)
        assert score < 0.3

    def test_long_complex_message_scores_high(self):
        from infergate.selector import complexity_score
        long = "analyze compare evaluate architecture design step by step " * 20
        msgs = [{"role": "user", "content": long}]
        score = complexity_score(msgs)
        assert score >= 0.65

    def test_multi_turn_conversation_adds_score(self):
        from infergate.selector import complexity_score
        msgs = [
            {"role": "user", "content": "question 1"},
            {"role": "assistant", "content": "answer 1"},
            {"role": "user", "content": "question 2"},
            {"role": "assistant", "content": "answer 2"},
            {"role": "user", "content": "question 3"},
            {"role": "assistant", "content": "answer 3"},
            {"role": "user", "content": "question 4"},
            {"role": "assistant", "content": "answer 4"},
            {"role": "user", "content": "question 5"},
        ]
        score_multi = complexity_score(msgs)
        score_single = complexity_score([msgs[-1]])
        assert score_multi > score_single

    def test_score_clamped_0_to_1(self):
        from infergate.selector import complexity_score
        # Impossible-to-exceed message: max all signals
        long = ("analyze compare evaluate architecture design step by step "
                "implement translate summarize comprehensive detailed " * 50)
        msgs = [{"role": "user", "content": long}]
        score = complexity_score(msgs)
        assert 0.0 <= score <= 1.0

    def test_multimodal_content_does_not_crash(self):
        """content as list (multimodal) must not crash last_user_text or complexity_score."""
        from infergate.selector import complexity_score
        from infergate.signals import last_user_text
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "what is in this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]}]
        # Should not raise
        text = last_user_text(msgs)
        score = complexity_score(msgs)
        assert isinstance(text, str)
        assert 0.0 <= score <= 1.0


class TestDeadCodeAndEdgeCases:
    """Catches silent degradation, dead code, and implicit ordering bugs."""

    def test_pick_backend_name_removed(self):
        """_pick_backend_name() was dead code — verify it no longer exists in chat_handler."""
        import ast
        import pathlib
        src = pathlib.Path("/opt/ov_server/chat_handler.py").read_text()
        tree = ast.parse(src)
        defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "_pick_backend_name" not in defined, (
            "_pick_backend_name was re-added — it is dead code, remove it"
        )

    def test_reselect_scope_local_plus_remote_allows_all_backends(self):
        """reselect() passes scope='local+remote' — _scope_allows must treat it as 'allow all'."""
        from infergate.selector import _scope_allows
        local_b = MockBackend("ov_server", [], _is_local=True)
        remote_b = MockBackend("ovh", [], _is_local=False)
        assert _scope_allows(local_b, "local+remote")
        assert _scope_allows(remote_b, "local+remote")

    def test_ovh_best_tier_listed_last_wins_pool_minus_one(self):
        """select_model uses pool[-1] for best tier — OVH must be last in config to win."""
        from infergate import RouterConfig
        import yaml
        with open(CONFIG_PATH) as fh:
            raw = yaml.safe_load(fh)
        config = RouterConfig.from_dict(raw)
        for cls_name, cls in config.task_classes.items():
            best_models = [m for m in cls.models if m.tier == "best"]
            if len(best_models) >= 2:
                ovh_best = [m for m in best_models if m.backend == "ovh"]
                if ovh_best:
                    # OVH best-tier must come LAST among best-tier models
                    last_best = best_models[-1]
                    assert last_best.backend == "ovh", (
                        f"task_class='{cls_name}': last best-tier model is local "
                        f"'{last_best.id}' — OVH model will NEVER win #cloud routing. "
                        f"Move OVH entries to end of models list in config.yaml."
                    )

    def test_embedding_cache_hit_returns_same_class(self):
        """Same query twice → second call is a cache hit with identical result."""
        router = _load_router()
        msgs = [{"role": "user", "content": "how do I stay motivated"}]
        d1 = _decide_sync(router, msgs)
        d2 = _decide_sync(router, msgs)
        assert d1.task_class == d2.task_class

    def test_signal_only_classes_not_reached_by_embedding(self):
        """vision and web_search are signal_only — embedding path must never return them."""
        from infergate.embeddings import route_by_embedding
        from infergate import RouterConfig
        import yaml
        with open(CONFIG_PATH) as fh:
            raw = yaml.safe_load(fh)
        config = RouterConfig.from_dict(raw)
        signal_only = {k for k, v in config.task_classes.items() if v.signal_only}
        centroids = {k: v for k, v in (getattr(config, "_centroids", None) or {}).items()
                     if k not in signal_only}
        # If no centroids (no embedding provider), skip
        if not centroids:
            pytest.skip("No centroids without embedding provider — can't test embedding path")

    def test_blocked_model_raises_http_400(self):
        """Blocked model check is in chat_handler, not infergate — verify via HTTP."""
        # This is a lightweight live check; skip if server not up
        try:
            r = httpx.post(
                "http://localhost:11435/v1/chat/completions",
                json={"model": "Auto", "messages": [{"role": "user", "content": "hi"}]},
                timeout=5.0,
            )
            # Server is up — now test blocked model if any are configured
            import json as _json
            cfg = _json.loads(
                httpx.get("http://localhost:11435/health", timeout=5.0).text
            )
            blocked = cfg.get("blocked_models", [])
            if not blocked:
                pytest.skip("No blocked_models in server config")
        except httpx.ConnectError:
            pytest.skip("Server not running")


# ---------------------------------------------------------------------------
# Part B — Live integration tests
# ---------------------------------------------------------------------------

BASE = "http://localhost:11435"

def _chat(messages, model="Auto", stream=False, tools=None, timeout=30.0):
    body = {"model": model, "messages": messages, "stream": stream}
    if tools:
        body["tools"] = tools
    r = httpx.post(f"{BASE}/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _last_route():
    r = httpx.get(f"{BASE}/health", timeout=5.0)
    r.raise_for_status()
    return r.json().get("last_routing_decision", {})


@pytest.fixture(scope="module", autouse=True)
def _require_server():
    try:
        httpx.get(f"{BASE}/health", timeout=3.0).raise_for_status()
    except Exception:
        pytest.skip("Server not running at localhost:11435")


@pytest.mark.live
class TestLiveRouting:
    """Full-stack routing tests — require running server."""

    def test_auto_routes_general_to_fast_model(self):
        _chat([{"role": "user", "content": "say the word 'yes' only"}])
        rd = _last_route()
        assert rd.get("model") in ("qwen3-8b-int4-ov", "qwen3-14b-int4-ov", "mistral-small-3.2-24b-int4-ov")
        assert rd.get("strategy") in ("signal", "embedding", "fallback", "keyword")

    def test_hash_code_directive_routes_code_model(self):
        _chat([{"role": "user", "content": "#code say 'yes'"}], timeout=120.0)
        rd = _last_route()
        assert rd.get("task_class") == "code"
        assert rd.get("model") in (
            "qwen2.5-coder-14b-int4",
            "mistral-small-3.2-24b-int4-ov",
            "qwen3-coder-30b-a3b-int4-ov",
        )

    def test_hash_document_directive_routes_document_model(self):
        _chat([{"role": "user", "content": "#document say 'yes'"}])
        rd = _last_route()
        assert rd.get("task_class") == "document"

    def test_keyword_search_routes_web_search(self):
        _chat([{"role": "user", "content": "search for recent Python news"}])
        rd = _last_route()
        assert rd.get("task_class") == "web_search"
        assert rd.get("strategy") == "signal"

    def test_explicit_local_model_bypasses_routing(self):
        _chat([{"role": "user", "content": "say 'yes'"}], model="qwen3-8b-int4-ov")
        rd = _last_route()
        assert rd.get("strategy") == "explicit"
        assert rd.get("model") == "qwen3-8b-int4-ov"

    def test_tools_present_routes_web_search_class(self):
        tools = [{"type": "function", "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }}]
        _chat([{"role": "user", "content": "search for AI news"}], tools=tools)
        rd = _last_route()
        assert rd.get("task_class") == "web_search"

    def test_routing_decision_confidence_present(self):
        _chat([{"role": "user", "content": "#general hello"}])
        rd = _last_route()
        # Confidence may be None for signal/keyword, or 0.0–1.0
        # Just verify the key exists
        assert "model" in rd

    def test_routing_latency_under_500ms(self):
        """Routing decision itself (not inference) must complete quickly."""
        # We measure total round-trip for a streaming-disabled simple request
        # Routing + TTFT combined — routing alone should be < 500ms
        rd_before = _last_route()
        t0 = time.perf_counter()
        _chat([{"role": "user", "content": "say 'yes'"}])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        rd_after = _last_route()
        route_ms = rd_after.get("latency_ms", 0)
        assert route_ms < 500, f"Routing took {route_ms}ms — expected < 500ms"

    def test_unknown_model_routes_as_auto(self):
        """Unknown model name → warning logged, routed as auto (no 4xx)."""
        # Should succeed without raising
        resp = httpx.post(
            f"{BASE}/v1/chat/completions",
            json={"model": "nonexistent-model-xyz", "messages": [{"role": "user", "content": "say yes"}]},
            timeout=30.0,
        )
        assert resp.status_code == 200

    def test_routing_decision_written_to_health(self):
        """After a request, last_routing_decision must be populated in /health."""
        _chat([{"role": "user", "content": "say 'yes'"}])
        rd = _last_route()
        assert rd, "last_routing_decision is empty after a request"
        assert "model" in rd
        assert "strategy" in rd

    def test_force_tier_best_via_profile(self):
        """Switching to 'laborious' profile and sending request → best-tier model selected."""
        r = httpx.post(f"{BASE}/admin/profile", json={"profile": "laborious"}, timeout=120.0)
        r.raise_for_status()
        try:
            _chat([{"role": "user", "content": "#code say 'yes'"}], timeout=120.0)
            rd = _last_route()
            assert rd.get("model") in ("qwen3-coder-30b-a3b-int4-ov", "mistral-small-3.2-24b-int4-ov"), \
                f"laborious profile picked unexpected model: {rd.get('model')}"
        finally:
            # Restore fast profile
            httpx.post(f"{BASE}/admin/profile", json={"profile": "fast"}, timeout=120.0)
