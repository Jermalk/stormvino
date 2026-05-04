# CLAUDE-changes.md — audit log of changes to CLAUDE.md

> Every change to CLAUDE.md is logged here with before/after and the failure mode it fixed.

---

## 2026-05-04 — Framework migration

**Change:** Merged session management framework into CLAUDE.md. Added: Re-entry protocol, Framework rules table (KYE/SBS/AEC/OMK/YNC), Context load discipline, PROGRESS.md NOW format, DECISIONS.md entry format, SCRATCHPAD.md discipline, session-wrap procedure. Added CLAUDE-ref.md pointer. Removed `qwen2.5-3b-int4` from model list (retired). Condensed Known Bugs table. Added Dev notes section.

**Issue fixed:** No consistent re-entry protocol — each session started cold with no structured handoff. Tool-Call Gap content was bloating CLAUDE.md; extracted to CLAUDE-ref.md.

**Pattern source:** Existing framework from another project (tmp/CLAUDE.md template).
