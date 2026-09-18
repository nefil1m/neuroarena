import pytest

from neuroarena.dashboard.pacing import DEFAULT_SPEED_PRESET, SPEED_PRESETS, Pacer
from neuroarena.sim.game import TICK_DT


class _FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _pacer(speed: float | None, fake: _FakeTime) -> Pacer:
    return Pacer(lambda: speed, clock=fake.clock, sleep=fake.sleep)


def test_presets_are_exactly_the_agreed_set_in_order() -> None:
    assert list(SPEED_PRESETS) == ["0.25x", "0.5x", "1x", "2x", "4x", "8x", "max"]
    assert SPEED_PRESETS["1x"] == 1.0
    assert SPEED_PRESETS["max"] is None
    assert DEFAULT_SPEED_PRESET == "max"


def test_max_speed_never_sleeps() -> None:
    fake = _FakeTime()
    pacer = _pacer(None, fake)
    for _ in range(5):
        pacer()
    assert fake.sleeps == []


def test_real_time_sleeps_one_tick_per_call() -> None:
    fake = _FakeTime()
    pacer = _pacer(1.0, fake)
    for _ in range(3):
        pacer()
    assert fake.sleeps == pytest.approx([TICK_DT] * 3)


def test_double_speed_sleeps_half_a_tick() -> None:
    fake = _FakeTime()
    pacer = _pacer(2.0, fake)
    pacer()
    assert fake.sleeps == pytest.approx([TICK_DT / 2])


def test_work_between_calls_is_subtracted_from_the_sleep() -> None:
    fake = _FakeTime()
    pacer = _pacer(1.0, fake)
    pacer()  # sleeps a full tick
    fake.now += 0.005  # 5 ms of "simulation work"
    pacer()
    assert fake.sleeps[1] == pytest.approx(TICK_DT - 0.005)


def test_a_speed_change_takes_effect_immediately() -> None:
    fake = _FakeTime()
    speed: list[float | None] = [1.0]
    pacer = Pacer(lambda: speed[0], clock=fake.clock, sleep=fake.sleep)
    pacer()
    speed[0] = 4.0
    pacer()
    assert fake.sleeps == pytest.approx([TICK_DT, TICK_DT / 4])


def test_falling_far_behind_does_not_cause_a_catch_up_burst() -> None:
    fake = _FakeTime()
    pacer = _pacer(1.0, fake)
    pacer()  # sleeps one tick
    fake.now += 1.0  # a very slow round
    pacer()  # behind by far more than the allowed lag: no sleep, deadline resets
    pacer()  # back to normal pacing, not a burst of zero-sleep calls
    assert len(fake.sleeps) == 2
    assert fake.sleeps[1] == pytest.approx(TICK_DT)
