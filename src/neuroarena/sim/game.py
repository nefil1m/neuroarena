"""Fixed-timestep game loop tying track, car physics, and collision together.

No episode/reset/terminated concept lives here — Phase 1 is a standalone playable game;
that's Phase 3's `Environment` concern (see `../../docs/WORKFLOW.md`, "stay within the
current phase").
"""

from __future__ import annotations

from dataclasses import dataclass

from neuroarena.sim.collision import resolve_collision
from neuroarena.sim.physics import CarState, PhysicsConstants, step_car
from neuroarena.sim.track import Track, boundary_segments

TICK_DT = 1 / 60
"""Fixed physics timestep (see the Phase 1 doc's "Tick rate" requirement) — never scaled by
display framerate or a future sim-speed setting."""


@dataclass(frozen=True)
class GameState:
    car: CarState
    track: Track


class Game:
    """Owns one track's collision geometry and the car driving on it."""

    def __init__(self, track: Track, constants: PhysicsConstants | None = None) -> None:
        self.track = track
        self.constants = constants if constants is not None else PhysicsConstants()
        self._boundary = boundary_segments(track)
        self.car = self._spawn_car()

    def _spawn_car(self) -> CarState:
        x, y = self.track.cell_center(self.track.start_cell)
        heading = self.track.start_facing.heading_radians
        return CarState(x=x, y=y, heading=heading, speed=0.0)

    def tick(self, steering: float, throttle: float) -> GameState:
        """Advance exactly one `TICK_DT` and return the resulting state."""
        previous = self.car
        stepped = step_car(previous, steering, throttle, TICK_DT, self.constants)
        self.car = resolve_collision(
            stepped, previous, self._boundary, self.constants.car_width / 2
        )
        return GameState(car=self.car, track=self.track)
