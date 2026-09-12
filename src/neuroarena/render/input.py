"""Keyboard -> (steering, throttle) adapter. A human is just another policy — this is a
permanent feature, not a bootstrapping throwaway (see the Phase 1 doc)."""

from __future__ import annotations

import arcade

_STEER_LEFT = {arcade.key.LEFT, arcade.key.A}
_STEER_RIGHT = {arcade.key.RIGHT, arcade.key.D}
_THROTTLE_UP = {arcade.key.UP, arcade.key.W}
_THROTTLE_DOWN = {arcade.key.DOWN, arcade.key.S}


class KeyboardInput:
    """Digital-in, continuous-out: held keys map directly to `{-1, 0, 1}` per axis."""

    def __init__(self) -> None:
        self._keys: set[int] = set()

    def on_key_press(self, key: int) -> None:
        self._keys.add(key)

    def on_key_release(self, key: int) -> None:
        self._keys.discard(key)

    def poll(self) -> tuple[float, float]:
        steering = 0.0
        if self._keys & _STEER_LEFT:
            steering += 1.0
        if self._keys & _STEER_RIGHT:
            steering -= 1.0

        throttle = 0.0
        if self._keys & _THROTTLE_UP:
            throttle += 1.0
        if self._keys & _THROTTLE_DOWN:
            throttle -= 1.0

        return steering, throttle
