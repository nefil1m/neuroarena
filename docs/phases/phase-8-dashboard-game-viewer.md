# Phase 8 — Dashboard: Game Viewer

Status: **DRAFT** — direction decided 2026-09-18 (see below), but requirements are not yet complete or finalized. Do not begin implementation until this doc reaches FINALIZED (see `../WORKFLOW.md`).

## Goal
Watch a training run on the real game: the existing `arcade` renderer, extended to show a whole generation's batch at once, opened and controlled from the dashboard's control panel.

## Requirements
Decided in the 2026-09-18 exploration pass. Remaining details are under Open questions.

- **The viewer is a native `arcade` window, not a browser view.** The dashboard stays a settings-and-metrics control panel; the browser does not draw the game. The panel gets an "open game window" / "close" control that spawns the viewer, plus the view settings below. (This revises the earlier framing of this phase as an in-browser live canvas — see Revision history and the rejected alternatives below.)
- **What the viewer shows: the whole batch, live.** Because Phase 4's concurrent-batch evaluation steps every genome of a generation in lockstep (each on its own isolated copy of the track, no interaction), all of them are running at the same instant. The viewer draws every still-active car on the track at once as non-interacting "ghosts"; a car that crashes or finishes drops out, so the field thins over the generation; the current best car is highlighted. No single-genome mode, no overlaid-trajectory mode, no montage.
- **The `arcade` renderer is extended, not duplicated.** Multi-car drawing and a camera with zoom and three selectable camera modes — **fit the whole track**, **follow the best car**, and **follow a chosen car** (all three decided 2026-09-18) — are added to the existing render layer. The same view is available to any other consumer of that layer.
- **The viewer is a separate process and a client of the dashboard API.** The trainer, dashboard backend and viewer talk only over that API (the platform rule in `../OVERVIEW.md`); the viewer reads car positions from the backend over its WebSocket and never touches the trainer directly. Positions are obtained the way Phase 7 obtains its progress snapshot: the backend samples the trainer's state on its own timer, decoupled from simulation ticks, with no hook inside the evaluation loop. The viewer's update rate is a design-time choice (it runs on the same machine as the backend).
- **View settings are controlled from the panel:** zoom, camera mode, which car to follow. The panel sets them through the backend, which forwards them to the running viewer.
- **Same-machine only, by nature.** A native window can only show on the machine running the trainer; the control panel itself remains usable remotely. Accepted.
- **Verified precondition:** the `arcade` game window opens and is playable on the project's WSL2 setup (confirmed 2026-09-18), so a window-based viewer is viable.
- **Independent phase** — reorderable relative to Phases 9 and 10, and other work may be inserted before it.

### Alternatives considered and rejected (2026-09-18)
- **Second renderer in the browser** (canvas/WebGL drawing the same tiles, kerbs, decor and car sprite from the same PNG assets): duplicates the game's drawing rules in TypeScript, with drift risk whenever the art rules change.
- **Stream the real game to the browser** (offscreen `arcade` render, JPEG/MJPEG frames): one renderer, but it depends on offscreen OpenGL working under WSL2 (a documented risk for this project), costs server CPU for rendering and encoding, and offers no zoom or follow control without more work. Not ruled out forever: the multi-car camera work built here is exactly what a later stream would render, so it can be added on top without redoing this phase.
- **Earlier same-day decisions, superseded:** a separate ~30 fps position stream to a browser canvas (moot without browser rendering); "what else goes on the canvas" (moot — the viewer draws what the game draws).

## Open questions
- How the car to follow is chosen from the panel (by rank? by id? click in the window?), and what the view does when the followed car crashes or finishes.
- How the viewer obtains the track and asset manifest: both processes share the data directory on one machine, so it could load the track by `track_id` itself; or the backend could serve it.
- The additive snapshot on the NEAT trainer that exposes every active car's pose (position, heading) — same "small, thread-safe, read-only" shape as Phase 7's progress snapshot; exact fields and the Phase 4 touchpoints.
- Viewer update rate and behaviour under high sim speeds (many steps between samples means the view shows a sample of the run).
- Behaviour between generations (show the last generation's final state briefly?) and when no run is active.
- Viewer lifecycle: what happens when the run ends, the backend exits, or the window is closed by hand; one viewer at a time or several.
- Whether the viewer window has its own keyboard shortcuts (zoom, camera) in addition to the panel controls.
- `arcade` runs its event loop on the process's main thread — the reason the viewer is its own process; confirm how the spawn/control channel and lifecycle are handled.

## Implementation plan
_Do not write this section until Requirements above is FINALIZED._

## Revision history
- 2026-09-09 — Created by splitting the original Phase 7 (Web Dashboard) into three phases; seeded requirements from the platform decisions pass.
- 2026-09-10 — Reframed for the platform scope cut (see `../OVERVIEW.md`): with genomes evaluated in isolation rather than in a shared simulation space, the live view is no longer "a running population" / swarm on one track. It now shows the currently-evaluating genome(s), optionally with this generation's completed trajectories overlaid. Added an open question on how to make that read as "watch the generation learn".
- 2026-09-18 — **Deliberate revision: from an in-browser live canvas to a native game-viewer window controlled from the panel.** First decided that the view shows the whole concurrent batch live (ghost cars, best highlighted), which resolved the earlier single-genome vs. overlay vs. montage question — Phase 4's 2026-09-15 concurrent-batch revision made that the natural view. Then, weighing how to get the game into the browser, rejected both a second browser-side renderer (duplicates the drawing rules) and server-side frame streaming (offscreen-GL risk under WSL2, encoding cost); chose instead to extend the `arcade` renderer with multi-car drawing and camera controls and open it as a separate viewer process from the panel, after confirming the `arcade` window works on the project's setup. Renamed from "Dashboard: Live Canvas" (file `phase-8-dashboard-live-canvas.md` → `phase-8-dashboard-game-viewer.md`); `../PHASES.md`, `../OVERVIEW.md` and `phase-7-dashboard-control-panel.md` updated to match. Camera modes settled the same day: fit-to-track, follow-best and follow-chosen-car, all three. Status stays DRAFT; open questions above remain.
