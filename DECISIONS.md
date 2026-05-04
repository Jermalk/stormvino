# DECISIONS.md — architectural decisions log

> Append-only. Never delete entries. Read only when user explicitly asks about a past decision.
> Format: see CLAUDE.md § DECISIONS.md entry format.

---

### 2026-05-04 — Adopted session management framework
**Decision:** Merged LLM session management framework (re-entry protocol, KYE/SBS/AEC/OMK/YNC rules, context discipline, session-wrap) into project CLAUDE.md.
**Rationale:** Framework was proven in another project; centralises session discipline so Claude Code behaves consistently across restarts without re-explanation.
**Rejected alternative:** Keeping framework in separate tmp/ directory — too easy to miss on re-entry.
**Affects:** CLAUDE.md, PROGRESS.md, SCRATCHPAD.md, DECISIONS.md, CLAUDE-ref.md, CLAUDE-changes.md
