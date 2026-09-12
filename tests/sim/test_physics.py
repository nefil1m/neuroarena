import math

from neuroarena.sim.physics import CarState, PhysicsConstants, step_car

DT = 1 / 60
C = PhysicsConstants()


def _run(state: CarState, steering: float, throttle: float, steps: int) -> CarState:
    for _ in range(steps):
        state = step_car(state, steering, throttle, DT, C)
    return state


def test_forward_accel_approaches_but_never_exceeds_max_speed() -> None:
    state = CarState(x=0, y=0, heading=0, speed=0)
    speeds = []
    for _ in range(600):  # 10s, comfortably past time-to-max at accel=250
        state = step_car(state, 0.0, 1.0, DT, C)
        speeds.append(state.speed)
        assert state.speed <= C.max_speed + 1e-9
    assert speeds == sorted(speeds)  # monotonically non-decreasing while below the cap
    assert math.isclose(state.speed, C.max_speed, rel_tol=1e-6)


def test_full_brake_from_cruise_decelerates_without_reversing_immediately() -> None:
    state = CarState(x=0, y=0, heading=0, speed=100.0)
    next_state = step_car(state, 0.0, -1.0, DT, C)
    assert next_state.speed == 100.0 - C.brake_decel * DT
    assert next_state.speed > 0


def test_brake_then_reverse_never_overshoots_zero_in_one_step() -> None:
    # A single large dt brake from low speed must clamp at exactly 0, not cross into reverse.
    state = CarState(x=0, y=0, heading=0, speed=1.0)
    next_state = step_car(state, 0.0, -1.0, dt=1.0, constants=C)
    assert next_state.speed == 0.0


def test_negative_throttle_reverses_only_once_stopped() -> None:
    state = CarState(x=0, y=0, heading=0, speed=50.0)
    # Brake to a stop.
    while state.speed > 0:
        state = step_car(state, 0.0, -1.0, DT, C)
    assert state.speed == 0.0
    # Now continued negative throttle should build reverse speed.
    state = step_car(state, 0.0, -1.0, DT, C)
    assert state.speed < 0.0
    assert state.speed == -C.reverse_accel * DT


def test_reverse_speed_capped_at_max_reverse_speed() -> None:
    state = CarState(x=0, y=0, heading=0, speed=0.0)
    for _ in range(600):
        state = step_car(state, 0.0, -1.0, DT, C)
        assert state.speed >= -C.max_reverse_speed - 1e-9
    assert math.isclose(state.speed, -C.max_reverse_speed, rel_tol=1e-6)


def test_max_steering_turn_radius_matches_wheelbase_geometry() -> None:
    # R = wheelbase / tan(max_steering_angle), independent of speed in this model.
    state = CarState(x=0, y=0, heading=0, speed=200.0)
    next_state = step_car(state, 1.0, 0.0, DT, C)
    angular_velocity = (next_state.heading - state.heading) / DT
    radius = state.speed / angular_velocity
    expected_radius = C.wheelbase / math.tan(C.max_steering_angle_rad)
    assert math.isclose(radius, expected_radius, rel_tol=1e-6)
    assert math.isclose(expected_radius, 257.1, abs_tol=1.0)  # the doc's sanity-check figure


def test_zero_input_at_rest_is_a_fixed_point() -> None:
    state = CarState(x=1.0, y=2.0, heading=0.3, speed=0.0)
    next_state = step_car(state, 0.0, 0.0, DT, C)
    assert next_state == state


def test_coasting_holds_speed_with_no_throttle() -> None:
    state = CarState(x=0, y=0, heading=0, speed=123.0)
    next_state = step_car(state, 0.0, 0.0, DT, C)
    assert next_state.speed == 123.0
