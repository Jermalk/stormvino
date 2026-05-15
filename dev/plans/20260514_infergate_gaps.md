# infergate 0.1.1 — Gap Analysis & Lessons from ov_server PoC

**Date:** 2026-05-14
**infergate version reviewed:** 0.1.1
**Source of lessons:** ov_server — OpenVINO OpenAI-compatible inference server at /opt/ov_server

ov_server is the production PoC that drove infergate's design. Gaps below are things
ov_server discovered in live use that are not yet reflected in the library.

---

## Bug: prefer-loaded must be tier-restricted (selector.py)

**Severity: correctness bug**

infergate prioritises warm-VRAM models for the "fastest" profile without restricting
the candidate set to fast-tier models. ov_server learned this the hard way:

> A loaded balanced/best model must not be returned just because it is in memory.
> If a user ran a "precise" task that loaded Mistral-24B, the next "fast" request
> would keep using Mistral instead of swapping back to qwen3-8b.

**ov_server's correct logic (router.py):**
```python
if pref == "fastest":
    loaded_fast = [m for m in available
                   if m["id"] in loaded_models and m.get("tier") == "fast"]
    chosen = _pick(loaded_fast) or _pick(available)
else:
    chosen = _pick(available)
```

`prefer_loaded` only applies within the fast tier. A loaded balanced/best model
is invisible to the prefer-loaded optimisation.

**Fix:** In selector.py, when `prefer_loaded=True`, filter candidates to
`tier == "fast"` before applying the loaded-model preference.

---

## Bug: signal-only class skip is hardcoded by name (embeddings.py)

**Severity: fragile hardcoding**

infergate skips classes named literally `"vision"` and `"web_search"` from embedding
centroid computation. If a user names their VLM class `"image_tasks"` or `"multimodal"`,
it will be included as an embedding target and pollute cosine similarity.

**ov_server's approach:** each task class in config carries a `"signal"` field set to
a binary signal name (`"has_image"`, `"has_tools"`). The router skips any class whose
signal value is in `_SIGNAL_ONLY_CLASSES` — names are irrelevant.

**Fix:** Add `signal_only: bool = False` to `TaskClassConfig`. Users set it for any
task class that is triggered exclusively by a binary signal and must never appear as
an embedding target. Default false; no hardcoded names.

```yaml
task_classes:
  vision:
    signal_only: true          # never embed; triggered by has_image signal
    description: "..."
    models: [...]
  web_search:
    signal_only: true          # never embed; triggered by tools presence
    ...
```

---

## Design: tools presence → task class should be configurable (signals.py)

**Severity: generality**

`req.tools` present → hardcoded `"web_search"` task class. Works for ov_server's
n8n/AnythingLLM use case but breaks any agent that provides tools for non-search
purposes (file operations, calendar, code execution, database queries).

**Fix:** Add `tools_task_class: str = "web_search"` to `RouterSettings`. Defaults to
current behaviour; deployments override as needed.

```yaml
router:
  tools_task_class: agent   # or "web_search", "code", whatever fits
```

---

## Design: directive hashtags should derive from task class names (signals.py / router.py)

**Severity: extensibility**

Both ov_server and infergate hardcode `#code`, `#document`, `#general` as valid
directive hashtags. A user who adds a `"sql"` or `"creative"` task class cannot
use `#sql` without patching the library.

**Fix:** Derive allowed task-class directives automatically from the task class names
in config. Any `task_classes` key becomes a valid `#<key>` directive.

```python
# Instead of:
_TASK_DIRECTIVE_RE = re.compile(r'#(code|document|general)\b', re.IGNORECASE)

# Build at Router init time:
_task_names = "|".join(re.escape(k) for k in config.task_classes)
_TASK_DIRECTIVE_RE = re.compile(rf'#({_task_names})\b', re.IGNORECASE)
```

Scope directives (`#ovh`, `#cloud`) are infrastructure-level and can stay hardcoded
since they are not task class names.

---

## Safety: empty model pool should raise, not return empty strings (selector.py)

**Severity: silent failure**

When no model is reachable after scope + availability filtering, infergate's selector
returns `("", "", False)`. Empty strings propagate silently — callers that forget to
check will send a request with no model ID.

**Fix option A:** Raise a typed exception.
```python
class NoModelAvailable(Exception):
    def __init__(self, task_class: str, scope: str):
        super().__init__(f"No model available for task_class='{task_class}' scope='{scope}'")
```

**Fix option B:** Add an `error: str | None` field to `RouteDecision`. A decision with
`error` set means routing failed; caller decides whether to raise or fallback.

ov_server chose option B conceptually (falls back to agent model) but that couples
the router to deployment-specific fallback knowledge. Option A is cleaner for a library.

---

## Feature gap: VLM / LLM distinction

**Severity: roadmap item**

`Backend.available_models()` returns a flat list with no LLM/VLM distinction.
ov_server maintains separate `loaded_models` and `loaded_vlm_models` dicts because:
- VLMs consume different VRAM than LLMs
- VLMs need different prompt builders (multimodal content, image tokens)
- A VLM and an LLM can coexist in VRAM simultaneously (different memory slots)
- Routing a vision request to a flat "loaded" list risks selecting a text-only LLM

**Possible fix:** Add an optional `modality: "text" | "vision" | "any"` field to
`ModelDescriptor` (default `"text"`). The selector filters by modality when routing
a vision task class. `Backend.loaded_model_ids()` could optionally return
`dict[str, str]` (model_id → modality) instead of a flat list, but that's a breaking
change — worth planning for 0.2.x.

---

## Feature gap: complexity promotion is one-directional

**Severity: minor / intentional**

`balanced + complexity > 0.65 → best` is implemented.
`fastest + complexity > X → balanced` is not.

A highly complex request routed via the "fast" profile always gets the fast-tier model,
even if complexity warrants promotion. This may be intentional (latency contract) but
is not documented. Either document the decision explicitly in the docstring or add a
second threshold:

```yaml
router:
  complexity_promote_balanced_threshold: 0.65   # balanced → best
  complexity_promote_fast_threshold: null        # fast → balanced; null = disabled
```

---

## Feature gap: pref_override for emergency model forcing

ov_server's `_select_model` accepts `pref_override: str | None` so callers can
force a specific preference tier regardless of the active profile. Used by the
assessor to always use a specific model, and by admin endpoints to override routing
for debugging.

infergate has no equivalent. Consider adding `force_tier: str | None` to `InferRequest`
or as a separate argument to `Router.decide()`.

---

## Config namespace notes (for ov_server PoC migration)

When ov_server adopts infergate as a dependency, the friction points are:

| ov_server field | infergate field | Notes |
|---|---|---|
| `provider: "loc"` | `is_local=True` on Backend | Different abstraction |
| `provider: "ovh"` | `is_local=False` on Backend | Different abstraction |
| `max_context_tokens` | `ctx_limit` | Backward compat shim exists ✓ |
| `signal: "has_image"` | `signal_only: true` | Once gap #2 is fixed |
| task class `"vision"` | `signal_only: true` | Class name stays; signal drives skip |

The provider (`"loc"` / `"ovh"`) → backend (`is_local`) mapping is the largest
conceptual gap. ov_server's catalogue uses string provider names throughout; infergate
uses Backend objects with `is_local`. A thin adapter layer will be needed.

---

## Summary priority order

| # | Gap | Priority |
|---|---|---|
| 1 | prefer-loaded tier restriction | P0 — correctness bug |
| 2 | signal-only via config flag, not hardcoded names | P1 — fragile |
| 3 | configurable tools_task_class | P1 — generality |
| 4 | directive hashtags derived from config keys | P1 — extensibility |
| 5 | typed failure on empty model pool | P2 — safety |
| 6 | VLM/LLM modality distinction | P3 — roadmap |
| 7 | complexity promotion fast→balanced | P3 — minor |
| 8 | pref_override / force_tier | P3 — nice to have |
