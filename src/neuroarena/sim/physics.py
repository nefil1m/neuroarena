"""Kinematic bicycle model car physics — no tire slip/drift (see Phase 1 doc)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsConstants:
    """World-unit physics constants. Starting defaults from the Phase 1 doc — explicitly
    tunable, not fixed law."""

    wheelbase: float = 180.0
    max_steering_angle_deg: float = 35.0
    max_speed: float = 750.0
    accel: float = 375.0
    brake_decel: float = 750.0
    max_reverse_speed: float = 300.0
    reverse_accel: float = 225.0
    # Car body geometry (world units) — see the Phase 1 doc's "World scale" section for how
    # these derive from the vendored art's drivable width, not from the physics itself.
    car_width: float = 148.0 / 3
    car_length: float = 100.0
    # Fraction of into-wall speed removed on a boundary hit (see collision.py). 1.0 = full
    # head-on stop; lower = "slippery" walls that don't trap the car. A hard collider either
    # way — this only tunes the speed loss on contact, not whether crossing is blocked.
    wall_friction: float = 0.15

    @property
    def max_steering_angle_rad(self) -> float:
        return math.radians(self.max_steering_angle_deg)


@dataclass(frozen=True)
class CarState:
    x: float
    y: float
    heading: float  # radians, 0 = +x axis, increasing counter-clockwise
    speed: float  # signed: positive = forward, negative = reverse


def step_car(
    state: CarState, steering: float, throttle: float, dt: float, constants: PhysicsConstants
) -> CarState:
    """Advance one fixed timestep. `steering`/`throttle` are clamped to [-1, 1]."""
    steering = max(-1.0, min(1.0, steering))
    throttle = max(-1.0, min(1.0, throttle))

    new_speed = _apply_throttle(state.speed, throttle, dt, constants)

    steering_angle = steering * constants.max_steering_angle_rad
    angular_velocity = (new_speed / constants.wheelbase) * math.tan(steering_angle)
    new_heading = state.heading + angular_velocity * dt
    new_x = state.x + new_speed * math.cos(state.heading) * dt
    new_y = state.y + new_speed * math.sin(state.heading) * dt

    return CarState(x=new_x, y=new_y, heading=new_heading, speed=new_speed)


def _apply_throttle(speed: float, throttle: float, dt: float, c: PhysicsConstants) -> float:
    """Brake-first-then-reverse: throttle opposing current motion brakes toward zero and
    never crosses it in the same step; only once stopped does continued opposing throttle
    build speed in the other direction."""
    if throttle == 0.0:
        return speed

    sign = 1.0 if throttle > 0 else -1.0
    magnitude = abs(throttle)
    accelerating = sign * speed >= 0  # same direction as current motion, or starting from rest

    if accelerating:
        rate = c.accel if sign > 0 else c.reverse_accel
        limit = c.max_speed if sign > 0 else -c.max_reverse_speed
        new_speed = speed + sign * rate * magnitude * dt
        return min(new_speed, limit) if sign > 0 else max(new_speed, limit)

    new_speed = speed + sign * c.brake_decel * magnitude * dt
    return min(new_speed, 0.0) if sign > 0 else max(new_speed, 0.0)
