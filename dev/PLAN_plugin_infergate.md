# PLAN — Plugin ↔ infergate integration
**Created:** 2026-05-15
**Status:** SCHEDULED (not started)
**Prerequisite commit:** bb41e90 (plugin architecture PoC)

---

## Problem

The plugin system and infergate both classify intent independently from the same
user text. `WebSearchPlugin` carries its own `_TRIGGERS` list; infergate carries
an identical list at `config.json → router.keywords.web_search`. When the user
says "search for X", two systems scan the same string and produce the same answer.

This means:
- Keyword lists must be maintained in two places
- Plugin firing is not guaranteed to match the model infergate picked for that task
- Adding a new intent requires updating both infergate config and a new plugin class

## Goal

infergate becomes the **single classifier**. Plugins become the **execution layer**
for infergate's decisions. The pattern:

```
infergate decides task_class='web_search'
               ↓
plugin_runner looks up plugin by task_class
               ↓
WebSearchPlugin.run() fetches context, injects before LLM
```

No plugin duplicates infergate's keyword/embedding detection.

---

## Steps (SBS — each verified before the next)

### Step 1 — Extend BasePlugin interface
**File:** `plugins/base.py`

Add an optional class attribute `task_class_trigger: str | None = None`.

When set, `plugin_runner` uses this as the primary match signal (task_class
equality) rather than calling `matches()`.

Keep `matches(text)` as a **fallback** path for plugins that don't map 1:1
to an infergate task class (e.g. a future CalculatorPlugin that fires on any
arithmetic expression regardless of task class).

```python
class BasePlugin(ABC):
    name: str
    task_class_trigger: str | None = None  # NEW

    def matches(self, text: str) -> str | None: ...  # fallback, still abstract
    async def run(self, query: str, cfg: dict) -> str: ...
```

**Verify:** `python3 -c "from plugins.base import BasePlugin; print('ok')"`

---

### Step 2 — Refactor WebSearchPlugin to use task_class_trigger
**File:** `plugins/web_search.py`

Set `task_class_trigger = "web_search"`.

Change `matches()` to be a pure fallback (keyword scan) kept for cases where
`task_class` is unavailable (e.g. direct API callers that bypass routing).

When `task_class_trigger` fires, the query is the full user text — strip trigger
prefix with a lightweight helper so SearxNG still gets clean keywords:

```python
def _strip_trigger(text: str) -> str:
    """Remove leading trigger phrase so SearxNG gets 'best pizza' not 'search for best pizza'."""
    lower = text.lower()
    for trigger in _TRIGGERS:
        if lower.startswith(trigger + " "):
            return text[len(trigger):].strip()
    return text
```

`run()` calls `_strip_trigger(query)` before querying SearxNG.

**Verify:** unit test — `p.matches("search for X")` still returns `"X"` via
keyword fallback when task_class not provided.

---

### Step 3 — Update plugin_runner.run() signature
**File:** `plugin_runner.py`

```python
async def run(
    text: str,
    cfg: dict,
    task_class: str | None = None,
) -> tuple[str, str]:
```

Match logic (in order):
1. If `plugin.task_class_trigger` is set **and** `task_class == plugin.task_class_trigger`
   → call `plugin.run(text, cfg)` directly (skip `matches()`)
2. Else if `task_class_trigger` is None → call `plugin.matches(text)` as before

This means a task_class match takes priority; keyword fallback still works when
`task_class` is None or doesn't match any plugin.

**Verify:** unit test — `plugin_runner.run("anything", cfg, task_class="web_search")`
fires `WebSearchPlugin.run()` without keyword scanning.

---

### Step 4 — Pass task_class from routing decision into plugin_runner
**File:** `chat_handler.py`

After the infergate routing block resolves `task_class` (currently stored in
`_route_task_class`), pass it to `plugin_runner.run()`:

```python
_plugin_name, _plugin_context = await plugin_runner.run(
    _last_user_text, _cfg, task_class=_route_task_class
)
```

`_route_task_class` is already available at the injection point — no structural
change needed.

**Verify:** `journalctl` shows `[plugin:web_search] matched via task_class`
(add a log flag to differentiate path used).

---

### Step 5 — Remove duplicate keyword list from WebSearchPlugin
**File:** `plugins/web_search.py`

Once Step 4 is proven working, the `_TRIGGERS` list in `web_search.py` is only
the fallback path. Decide:

- **Option A:** Keep `_TRIGGERS` as fallback for clients that bypass routing
  (direct POST with explicit model, OVH proxy path). Cost: list maintained but
  not primary.
- **Option B:** Remove `_TRIGGERS` entirely. Keyword-triggered web search only
  works via infergate routing. Simpler, one source of truth.

**Recommended:** Option A for PoC, document the fallback explicitly. Move to
Option B when infergate routing covers all clients.

**Verify:** send "search for X" with explicit `model=qwen3-8b-int4-ov` (bypasses
routing, `_route_task_class=None`) → confirm keyword fallback still fires.

---

### Step 6 — Tests
**File:** `autotest/test_plugins.py` (new)

```
TestBasePlugin
  - task_class_trigger=None → uses matches() path
  - task_class_trigger set + task_class matches → skips matches(), calls run()
  - task_class_trigger set + task_class mismatch → falls through to next plugin

TestWebSearchPlugin
  - matches() EN triggers: search for, look up, find on the web, find, browse for, google
  - matches() PL triggers: wyszukaj, znajdź, sprawdź, szukaj
  - _strip_trigger removes prefix correctly
  - task_class_trigger == "web_search"

TestPluginRunner
  - run(text, cfg, task_class="web_search") → fires via trigger, not keyword
  - run(text, cfg, task_class=None) → keyword fallback
  - run(text, cfg, task_class="general") → no match, returns ("", "")
  - disabled plugin (cfg plugins.web_search.enabled=false) → skipped
```

Target: all pass before commit.

---

### Step 7 — Log differentiation + DECISIONS.md entry
**File:** `plugin_runner.py`, `dev/DECISIONS.md`

Add log tag to distinguish the two paths:
```
[plugin:web_search] matched via task_class   ← new primary path
[plugin:web_search] matched via keyword      ← fallback path
```

Write DECISIONS.md entry for the integration decision.

---

## Files touched

| File | Change |
|---|---|
| `plugins/base.py` | Add `task_class_trigger` attribute |
| `plugins/web_search.py` | Set trigger, add `_strip_trigger()`, keep keyword fallback |
| `plugin_runner.py` | Accept `task_class`, primary match on trigger, fallback on keyword |
| `chat_handler.py` | Pass `_route_task_class` to `plugin_runner.run()` |
| `autotest/test_plugins.py` | New test file, ~40 tests |
| `dev/DECISIONS.md` | Decision entry |

**No changes to:** `config.json`, `infergate/`, `news_scraper.py`

---

## Definition of done

- [ ] All 6 steps verified individually
- [ ] `autotest/test_plugins.py` passes (all tests green)
- [ ] `/health` still ok after restart
- [ ] "search for X" via VoiceButton → server log shows `matched via task_class`
- [ ] "search for X" with explicit local model (bypass routing) → log shows `matched via keyword`
- [ ] DECISIONS.md entry written
- [ ] Session wrap committed

---

## Estimated scope
~2–3 hours. No schema changes, no new dependencies.
Pure refactor — external behaviour unchanged, internal coupling reduced.
