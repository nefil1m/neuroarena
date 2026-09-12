# Phase 5 — Training Controls

Status: **DRAFT** — requirements seeded from the 2026-09-09 platform decisions pass (see `../OVERVIEW.md`); not yet a complete or finalized requirements set. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
Configurable training/run behaviour: population size, deviation limit, simulation speed, success criteria, episode/generation timeouts.

## Requirements
Carried over from the 2026-09-09 decisions pass. Dedicated exploration still needed before FINALIZED.

- **Config knobs** (per `../OVERVIEW.md`): population size; inheritance between runs; max deviation from track before death; adjustable simulation speed; headless vs rendered; per-episode time limit; per-generation time/step ceiling (the guarantee that a generation always ends — see Phase 0 and Phase 4); **track/map selection** — which stored track (by `track_id`, Phase 2) a run trains on.
- **Track/map selection is live-changeable**, per `../OVERVIEW.md`'s user-driven track lifecycle (training continues on a track until the user regenerates or switches). Switching appends a settings-history entry (Phase 6) like any other live config change and does **not** start a fresh model: a track's geometry doesn't change the observation/action shape Phase 0's compatibility check compares (rays still return the same 7 proximity floats), only the underlying raycast distances — unlike a sensor-layout change (Phase 3), which does.
- **Pluggable per-run success criteria** (e.g. finish line crossed / N checkpoints).
- **Each knob is classified** as either set-at-run-creation or live-changeable while a run is in progress. The dashboard (Phase 7) exposes the live-changeable subset.
- **Settings are changeable between runs and on resume.** A model saved under one settings set can be resumed under a different (e.g. harder) set and continue training from its saved state — see Phase 6. This is valid as long as the observation and action descriptors are unchanged; changing sensor layout (Phase 3) starts a fresh model rather than a resume.

## Open questions
- The exact partition of knobs into set-at-creation vs live-changeable.
- Semantics of changing certain knobs mid-run (e.g. population size between generations).
- A concrete definition of "inheritance between runs."
- Whether the per-episode limit and per-generation ceiling are set-at-creation or live-changeable, and their units (steps vs sim-time vs wall-clock).

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Stub created.
- 2026-09-09 — Seeded requirements from the platform decisions pass; status → DRAFT.
- 2026-09-10 — Added two timeout knobs to the config-knob set: a per-episode time limit and a per-generation time/step ceiling. Reason: nothing previously guaranteed a generation would terminate if an individual neither crashed nor finished.
- 2026-09-10 — Removed two knobs to match the platform scope cut (see `../OVERVIEW.md`): the batch-vs-sequential spawn-mode selector (the loop is batch only now) and the car-to-car collisions on/off toggle (genomes are evaluated in isolation). Track-boundary collision is unaffected — it is always on and is not a knob.
- 2026-09-12 — Added a track/map-selection config knob, following Phase 2's exploration pass which added map identity (`track_id`) and storage. Classified live-changeable, reusing the existing settings-history mechanism (Phase 6) rather than adding new machinery, and explicitly noted as not a fresh-model trigger since track geometry doesn't change the observation/action shape.
