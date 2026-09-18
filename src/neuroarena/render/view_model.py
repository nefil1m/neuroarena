"""Pure logic for the viewer window: ranking, which car the camera follows, the camera
transform, and the overlay text. Imports neither `arcade` nor FastAPI, so it is unit-testable
without a display — the window only draws what this module decides. See
`../../../docs/phases/phase-8-dashboard-game-viewer.md`."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from neuroarena.render.view_settings import ViewSettings
from neuroarena.sim.track import Track

FOLLOW_BASE_SCALE = 0.5
"""World units -> pixels when following a car at zoom 1.0 (a 512-unit cell is 256 px)."""
FIT_MARGIN = 1.15
"""The whole-track view leaves ~15% breathing room around the track."""

Bounds = tuple[float, float, float, float]  # min_x, min_y, max_x, max_y


@dataclass(frozen=True)
class CarView:
    genome_id: int
    x: float
    y: float
    heading: float
    fitness: float


@dataclass(frozen=True)
class Camera:
    center_x: float
    center_y: float
    scale: float  # arcade's zoom factor: world units -> pixels


def rank_cars(cars: Sequence[CarView]) -> list[CarView]:
    """Best first: live fitness descending, ties broken by the lower genome id."""
    return sorted(cars, key=lambda car: (-car.fitness, car.genome_id))


def rank_of(cars: Sequence[CarView], car: CarView) -> int | None:
    for index, ranked in enumerate(rank_cars(cars), start=1):
        if ranked.genome_id == car.genome_id:
            return index
    return None


def track_bounds(track: Track) -> Bounds:
    half = track.cell_size / 2
    centers = [track.cell_center(cell) for cell in track.cells]
    xs = [center[0] for center in centers]
    ys = [center[1] for center in centers]
    return (min(xs) - half, min(ys) - half, max(xs) + half, max(ys) + half)


class ViewModel:
    """Holds the little state the follow logic needs (which car a follow-rank choice locked
    onto) and computes the camera. One instance per loaded track."""

    def __init__(self, bounds: Bounds, window_size: tuple[int, int]) -> None:
        self._bounds = bounds
        self._window_size = window_size
        self._followed_id: int | None = None
        self._last_mode: str | None = None
        self._applied_seq: int | None = None

    def followed_car(self, cars: Sequence[CarView], settings: ViewSettings) -> CarView | None:
        ranked = rank_cars(cars)
        mode = settings.camera_mode
        if not ranked or mode == "fit":
            self._followed_id = None
            self._last_mode = mode
            return None
        if mode == "follow_best":
            self._followed_id = ranked[0].genome_id
            self._last_mode = mode
            return ranked[0]
        # follow_rank: (re)resolve the rank to a car when the mode was just entered or a new
        # choice was made; otherwise keep following that same car.
        if mode != self._last_mode or settings.follow_seq != self._applied_seq:
            index = min(settings.follow_rank, len(ranked)) - 1
            self._followed_id = ranked[index].genome_id
            self._applied_seq = settings.follow_seq
        self._last_mode = mode
        followed = next((car for car in ranked if car.genome_id == self._followed_id), None)
        if followed is None:
            # The followed car crashed or finished: fall back to the best remaining car and
            # keep following that one, so the camera never sits on an empty spot.
            followed = ranked[0]
            self._followed_id = followed.genome_id
        return followed

    def camera(self, followed: CarView | None, settings: ViewSettings) -> Camera:
        if followed is not None:
            return Camera(followed.x, followed.y, FOLLOW_BASE_SCALE * settings.zoom)
        min_x, min_y, max_x, max_y = self._bounds
        width_px, height_px = self._window_size
        fit = min(width_px / (max_x - min_x), height_px / (max_y - min_y)) / FIT_MARGIN
        return Camera((min_x + max_x) / 2, (min_y + max_y) / 2, fit * settings.zoom)


def overlay_lines(
    *,
    connected: bool,
    run_state: str,
    generation: int | None,
    alive: int,
    population_size: int,
    speed: str,
    settings: ViewSettings,
    followed_rank: int | None,
) -> list[str]:
    if not connected:
        return ["Disconnected from the dashboard backend - retrying..."]
    if run_state == "idle" and alive == 0 and generation is None:
        return ["Waiting for a run..."]
    if run_state == "running" and generation is None:
        return ["Run starting..."]
    lines: list[str] = []
    if generation is not None:
        lines.append(f"Generation {generation}")
    lines.append(f"Cars: {alive} / {population_size}")
    lines.append(f"Speed: {speed}")
    if settings.camera_mode == "fit":
        camera = "whole track"
    elif settings.camera_mode == "follow_best":
        camera = "following the best car"
    else:
        camera = "following #?" if followed_rank is None else f"following #{followed_rank}"
    lines.append(f"Camera: {camera}")
    if run_state in ("completed", "stopped", "crashed"):
        lines.append(f"Run {run_state}")
    return lines
