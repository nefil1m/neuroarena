"""Procedural closed-loop track generation: a self-avoiding random walk over the grid,
biased by `complexity`, backtracking out of dead ends, retried until a loop closes near
the requested size. See the Phase 2 doc, "Generation algorithm"."""

from __future__ import annotations

import math
import random

from neuroarena.sim.track import Facing, GridCell, TileKind, Track, tile_kind_for_open_edges

MIN_TRACK_SIZE = 4
"""Smallest closed loop expressible with 90°-only turns: a 2x2 block of curve tiles."""

SIZE_DRIFT = 0.15
"""Accepted fractional drift between requested and actual tile count (Phase 2 doc)."""

_CLOCKWISE = [Facing.N, Facing.E, Facing.S, Facing.W]
_TURN_RIGHT: dict[Facing, Facing] = {f: _CLOCKWISE[(i + 1) % 4] for i, f in enumerate(_CLOCKWISE)}
_TURN_LEFT: dict[Facing, Facing] = {f: _CLOCKWISE[(i - 1) % 4] for i, f in enumerate(_CLOCKWISE)}


class TrackGenerationError(RuntimeError):
    """Raised when no valid closed loop near the requested size could be generated."""


def generate_track(
    size: int,
    complexity: float,
    seed: int,
    *,
    max_attempts: int = 500,
    step_budget_factor: int = 25,
) -> Track:
    if size < MIN_TRACK_SIZE:
        raise ValueError(
            f"size must be >= {MIN_TRACK_SIZE} (a 90°-only closed loop's minimum); got {size}"
        )
    if not 0.0 <= complexity <= 1.0:
        raise ValueError(f"complexity must be in [0, 1]; got {complexity}")

    rng = random.Random(seed)
    min_close_length = max(MIN_TRACK_SIZE, math.ceil(size * (1 - SIZE_DRIFT)))
    max_length = math.floor(size * (1 + SIZE_DRIFT))
    step_budget = max(size * step_budget_factor, 200)

    # A 90°-only closed loop is a cycle on a grid graph, which is bipartite by (x+y) parity —
    # every cycle in a bipartite graph has even length, so a tile count is only reachable if
    # it's even. [min_close_length, max_length] contains an even integer unless it has
    # collapsed to a single odd value (any wider window straddles both parities); in that
    # case no amount of retrying can ever close a loop, so fail fast instead of burning the
    # full attempt budget on an impossible target.
    if min_close_length == max_length and min_close_length % 2 != 0:
        raise TrackGenerationError(
            f"no even tile count exists within the ±{SIZE_DRIFT:.0%} window "
            f"[{min_close_length}, {max_length}] for size={size} — a 90°-only closed loop "
            f"always has an even tile count; try size={size - 1} or size={size + 1}"
        )

    for _ in range(max_attempts):
        path = _walk_attempt(rng, min_close_length, max_length, complexity, step_budget)
        if path is not None:
            cells = _path_to_cells(path)
            return Track(cells=cells, start_cell=path[0], start_facing=_direction(path[0], path[1]))

    raise TrackGenerationError(
        f"failed to close a loop within {min_close_length}-{max_length} tiles "
        f"of size={size} after {max_attempts} attempts"
    )


def _walk_attempt(
    rng: random.Random,
    min_close_length: int,
    max_length: int,
    complexity: float,
    step_budget: int,
) -> list[GridCell] | None:
    """One self-avoiding random-walk attempt: returns an ordered cycle of cells (the last
    cell connects back to the first) or None if it dead-ended without closing in budget."""
    start: GridCell = (0, 0)
    heading = rng.choice(_CLOCKWISE)
    path: list[GridCell] = [start]
    visited: set[GridCell] = {start}
    choice_stack: list[list[Facing]] = [_ordered_choices(rng, heading, complexity)]
    steps = 0

    while steps < step_budget:
        steps += 1
        current = path[-1]
        choices = choice_stack[-1]

        if not choices:
            path.pop()
            choice_stack.pop()
            if not path:
                return None  # backtracked past the start — this attempt cannot close
            visited.discard(current)
            continue

        next_heading = choices.pop(0)
        dx, dy = next_heading.delta
        candidate = (current[0] + dx, current[1] + dy)

        if candidate == start:
            if len(path) >= min_close_length:
                return path
            continue  # too short to close yet — try the next queued heading instead
        if candidate in visited:
            continue
        # Feasibility prune: with only unit orthogonal steps, closing from `candidate`
        # needs at least its Manhattan distance back to `start` in remaining hops — one hop
        # per cell still appendable before `max_length` (max_length - len(path), counting
        # `candidate` itself), plus one final hop back to `start`. Without this, a
        # straight-biased (complexity near 0) or turn-biased (complexity near 1) walk
        # commits to running all the way to `max_length` before ever backtracking, which
        # made closing a loop combinatorially expensive (verified empirically: step counts
        # grew ~150x for a ~1.3x size increase at complexity=0.0). Pruning here forces the
        # backtrack the moment a candidate would strand the walk too far from `start` to
        # still close, which subsumes the old "at max_length" cap check (that check is the
        # special case where available_hops is already 0).
        available_hops = max_length - len(path)
        manhattan_to_start = abs(candidate[0] - start[0]) + abs(candidate[1] - start[1])
        if manhattan_to_start > available_hops:
            continue  # candidate can no longer close within max_length — dead end

        path.append(candidate)
        visited.add(candidate)
        choice_stack.append(_ordered_choices(rng, next_heading, complexity))

    return None  # exhausted the step budget without closing


def _ordered_choices(rng: random.Random, heading: Facing, complexity: float) -> list[Facing]:
    """Candidate next headings from `heading` — never a reversal, since {N,E,S,W} only
    connect via straight-through or 90°-turn TileKinds, never a U-turn. `complexity` biases
    whether a turn or the straight continuation is tried first."""
    turns = [_TURN_LEFT[heading], _TURN_RIGHT[heading]]
    rng.shuffle(turns)
    if rng.random() < complexity:
        return [*turns, heading]
    return [heading, *turns]


def _path_to_cells(path: list[GridCell]) -> dict[GridCell, TileKind]:
    cells: dict[GridCell, TileKind] = {}
    n = len(path)
    for i, cell in enumerate(path):
        prev_cell = path[i - 1]
        next_cell = path[(i + 1) % n]
        cells[cell] = tile_kind_for_open_edges(
            _direction(cell, prev_cell), _direction(cell, next_cell)
        )
    return cells


def _direction(frm: GridCell, to: GridCell) -> Facing:
    delta = (to[0] - frm[0], to[1] - frm[1])
    for facing in _CLOCKWISE:
        if facing.delta == delta:
            return facing
    raise AssertionError(f"{frm} -> {to} is not a unit grid step")
