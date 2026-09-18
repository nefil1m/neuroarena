"""View settings shared by the dashboard backend and the viewer process: the camera modes,
zoom bounds and the validated `ViewSettings` record. Pure — imports neither `arcade` nor
FastAPI — so both sides (and their tests) can use it. See
`../../../docs/phases/phase-8-dashboard-game-viewer.md`."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

CAMERA_MODES = ("fit", "follow_best", "follow_rank")
ZOOM_MIN = 0.25
ZOOM_MAX = 4.0
_PARTIAL_KEYS = frozenset({"zoom", "camera_mode", "follow_rank"})


@dataclass(frozen=True)
class ViewSettings:
    """`follow_rank` is 1-based and only matters in `follow_rank` mode. `follow_seq` counts
    how many times a follow-rank choice has been made, so choosing the same rank twice is
    still a new choice the viewer re-resolves (the choice is held on a *car* until it drops
    out — see `view_model.ViewModel`)."""

    zoom: float = 1.0
    camera_mode: str = "fit"
    follow_rank: int = 1
    follow_seq: int = 0

    def __post_init__(self) -> None:
        if self.camera_mode not in CAMERA_MODES:
            raise ValueError(f"camera_mode must be one of {CAMERA_MODES}, got {self.camera_mode!r}")
        if not ZOOM_MIN <= self.zoom <= ZOOM_MAX:
            raise ValueError(f"zoom must be within [{ZOOM_MIN}, {ZOOM_MAX}], got {self.zoom}")
        if self.follow_rank < 1:
            raise ValueError(f"follow_rank must be >= 1, got {self.follow_rank}")

    def updated(self, partial: Mapping[str, Any]) -> ViewSettings:
        unknown = set(partial) - _PARTIAL_KEYS
        if unknown:
            raise ValueError(f"unsupported view setting(s): {sorted(unknown)}")
        follow_seq = self.follow_seq + (1 if "follow_rank" in partial else 0)
        return ViewSettings(
            zoom=float(partial.get("zoom", self.zoom)),
            camera_mode=str(partial.get("camera_mode", self.camera_mode)),
            follow_rank=int(partial.get("follow_rank", self.follow_rank)),
            follow_seq=follow_seq,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ViewSettings:
        return cls(
            zoom=float(data["zoom"]),
            camera_mode=str(data["camera_mode"]),
            follow_rank=int(data["follow_rank"]),
            follow_seq=int(data["follow_seq"]),
        )
