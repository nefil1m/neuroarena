# Workflow & Standards — Car AI Project

Status: living document — always current. Update this whenever the *process itself* changes (not project decisions — those go in OVERVIEW.md and phase docs).

## Purpose
This project spans many chat sessions with resets in between. This document is the entry point: read it first, every session, before touching requirements or code.

## Document map
1. `WORKFLOW.md` (this file) — how we work
2. `OVERVIEW.md` — the brief: high-level feature set, key decisions, open questions. DRAFT until stated otherwise.
3. `PHASES.md` — the phase list, each phase linking to `/phases/phase-N-*.md`
4. `/phases/phase-N-*.md` — one doc per phase; holds that phase's detailed requirements, then its implementation plan. **This is where actual build work happens** — and only once that phase's doc is FINALIZED.

## The rule: decide before building
No code gets written for a phase until that phase's doc status is FINALIZED. Exploration, options, and trade-offs happen in the doc first, in writing. This applies to the project as a whole too: `OVERVIEW.md` and `PHASES.md` both need to leave DRAFT before Phase 0 implementation starts.

## Document status labels
Every doc carries one status marker near the top:
- **DRAFT** — actively being explored, contents can change freely
- **FINALIZED** — decided; changing it requires a deliberate revision, not a drive-by edit mid-conversation about something else
- **SUPERSEDED** — replaced by a linked doc; kept for history, not deleted

## Session start checklist
1. Skim this file if anything about process feels unclear
2. Read `OVERVIEW.md` for the current state of decisions
3. Check `PHASES.md` to see which phase is active
4. Read that phase's doc in full before discussing or building anything in it

## Session end checklist
1. If any decision changed during the session — including inside a phase marked FINALIZED — update the relevant doc immediately, don't leave it to memory
2. Add a line to that doc's Revision History
3. If the change ripples into another doc (a Phase 3 decision invalidates something in `OVERVIEW.md`), update that doc too, same session
4. If scope grew (new phase, new feature, new open question), reflect it in `PHASES.md` / `OVERVIEW.md` as a DRAFT stub — don't let it live only in chat history

## How Claude should behave in this project (proactive teaching mode)
The user is learning as this project goes, and wants to see the landscape, not just get an answer. When a decision involves a technology or approach choice (e.g. "should I use Unity or Godot", "NEAT or RL", "SQLite or something else"):
1. Answer the direct question with a real recommendation and the reasoning behind it
2. Then explicitly name 1–3 other options that exist for that decision, briefly say what each would trade off, and ask whether the user wants to explore any of them before committing
3. Don't silently pick the "obviously right" answer and move past it — surfacing the alternatives is part of the point

This applies most heavily during requirements exploration. Once a phase is FINALIZED, re-opening a decision needs an explicit reason, not a casual "what if."

## Change propagation
Docs drifting out of sync with reality *during* a session is expected and fine — that's what exploration looks like. What's not fine is ending a session with that drift unresolved. Every doc touched by a decision made this session gets updated before the session is considered closed.

## Revision history
- 2026-09-09 — Initial workflow doc created
