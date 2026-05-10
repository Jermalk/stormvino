# Local model + Aider planning logic
**Date:** 2026-05-10
**Context:** Planning session — arose from the ov_server.py module split discussion.

---

## The question

After creating CONVENTIONS.md and .aider.conf.yml for a hybrid Claude Code + Aider workflow,
the user asked: is a 14-20B local model (Qwen2.5-14b-coder, etc.) actually feasible as a
coding assistant here, or does it need significant guardrails to be useful?

---

## Feasibility scoring (Claude Code = 100)

| Dimension | Claude Code | Qwen3-30b-a3b via Aider | Qwen2.5-14b-coder via Aider |
|---|---|---|---|
| Single-file mechanical edit | 100 | 70 | 75 |
| Multi-file reasoning | 100 | 20 | 10 |
| Instruction constraint following | 100 | 55 | 50 |
| Async/threading edge cases | 100 | 25 | 20 |
| Session continuity (memory, progress) | 100 | 0 | 0 |
| Self-verification (run curl, read result) | 100 | 0 | 0 |
| Type annotation tasks | 100 | 65 | 75 |
| OpenVINO domain knowledge | 100 | 35 | 30 |
| Edit reliability (no failed apply) | 100 | 60 | 75 |
| Workflow friction (setup overhead) | 100 | 50 | 50 |
| **Weighted overall** | **100** | **~38** | **~37** |

---

## Critical insight: Qwen3-30b-a3b is not a 30B model in practice

Qwen3-30b-a3b is a Mixture-of-Experts model with **3B active parameters** per forward pass.
The "30B" refers to total parameter count; the "a3b" suffix means 3B are active.
For reasoning-heavy tasks it performs closer to a 7-8B dense model.
The 14B coder model is paradoxically competitive or slightly better on pure coding tasks
because coder fine-tuning compensates for smaller size.
Do not be misled by the 30B label when estimating task suitability.

---

## Where 14B drops the ball specifically

Two concrete failure modes for this project:

**1. Context overhead**
CONVENTIONS.md (~1500 tokens) + repo map (1024 tokens) + file + task = 3000-4000 tokens
before the model starts working. At 14B, instruction-following degrades near the edges
of a long prompt. Nuanced rules are the first to be forgotten.

**2. The stale-binding rule will not stick**
`import model_manager; model_manager.emb_model` vs `from model_manager import emb_model`
is the kind of nuanced constraint a 14B model ignores under pressure, defaulting to the
more common `from ... import` pattern. Mitigation: scope tasks so 14B never touches
router.py or catalogue.py (the only files where this matters).

---

## Guardrails needed for 14B if used

These are not in the config — they are operational practices:

1. **Always pass `--file <specific_file>`** — never let the model choose from the repo map.
2. **Reduce `map-tokens` to 512** — less context overhead; smaller map is better used than a larger one half-absorbed.
3. **Add `edit-format: whole`** — whole-function replacement is more reliable than diff-style for smaller models (fewer offset errors). Counterpoint: Aider already recorded a preference for diff format from prior george experience (see DECISIONS.md 2026-05-06).
4. **Phrase tasks as pattern-matching** — "same pattern as X at line N in this file" anchors the model to existing code.
5. **Never use 14B on async paths** — AsyncTokenStreamer, event loop capture, run_in_executor interaction. These require cross-file reasoning the model cannot do reliably.

---

## The narrow task category where local models are useful

Tasks where a 14B coder model reaches ~70-75% of Claude Code effectiveness:
- Add `Literal` type annotations to a single module
- Replace legacy `Optional`, `Dict`, `List` imports with modern `|` syntax
- Rename a function or variable within one file
- Add a new simple API endpoint following an exact existing template
- Write an isolated unit test for a pure function with no async or cross-module state

These tasks represent roughly 20% of actual work on this project.
The other 80% (async bugs, multi-file reasoning, routing diagnostics, architectural decisions)
requires Claude Code.

---

## The ROI question

The workflow overhead for Aider (session setup, file targeting, edit review, failed retries)
largely cancels the time savings for a single-developer project.
The parallel workflow argument (Claude on complex task, Aider on mechanical task simultaneously)
does not apply when there is one developer.

**Decision:** Keep CONVENTIONS.md and .aider.conf.yml — they document conventions in
machine-readable form and have value even when Aider is rarely launched.
Mark local model operation as **highly optional** in the split plan.
Do not invest time optimising the Aider workflow unless a specific high-volume
mechanical task (e.g. typing the entire codebase post-split) makes it worthwhile.

---

## Files created as a result of this planning

| File | Purpose |
|---|---|
| `coding_standards_python.json` | Python typing standards — risky techniques removed, 2nd-order marked |
| `CONVENTIONS.md` | Machine-facing coding conventions (module map, import rules, recipes) |
| `.aider.conf.yml` | Aider config — local ov_server backend; architect mode commented |
| `plans/20260510_PLAN_split.md` | Full module split plan including Step 7 (hybrid workflow artifacts) |
