# Agent Instructions

Any agent (or human) doing work in this repository MUST always start by reading:

1. `docs/WORKFLOW.md` — how this project works: document status labels, the
   "decide before building" rule, and session start/end checklists.
2. `docs/OVERVIEW.md` — the project brief: pitch, feature set, and key decisions.

These are the entry point for every session, before touching requirements or code — no
exceptions, even for small tasks. `WORKFLOW.md` itself then points to `PHASES.md` and the
per-phase docs under `docs/phases/`, which is where actual build work happens once a phase
is FINALIZED.

Do not start writing code, proposing designs, or making decisions until you've read both
files above and understand which phase (if any) is currently active and FINALIZED.
