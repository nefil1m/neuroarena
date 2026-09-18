import pytest

from neuroarena.render.view_model import (
    FIT_MARGIN,
    FOLLOW_BASE_SCALE,
    CarView,
    ViewModel,
    overlay_lines,
    rank_cars,
    rank_of,
    track_bounds,
)
from neuroarena.render.view_settings import ViewSettings
from neuroarena.sim.track import Facing, GridCell, TileKind, Track


def _rounded_rectangle() -> dict[GridCell, TileKind]:
    K = TileKind
    return {
        (0, 0): K.CURVE_NE,
        (1, 0): K.STRAIGHT_EW,
        (2, 0): K.STRAIGHT_EW,
        (3, 0): K.CURVE_NW,
        (3, 1): K.STRAIGHT_NS,
        (3, 2): K.CURVE_SW,
        (2, 2): K.STRAIGHT_EW,
        (1, 2): K.STRAIGHT_EW,
        (0, 2): K.CURVE_SE,
        (0, 1): K.STRAIGHT_NS,
    }


def _track() -> Track:
    return Track(cells=_rounded_rectangle(), start_cell=(1, 0), start_facing=Facing.E)


def _car(genome_id: int, fitness: float, x: float = 0.0, y: float = 0.0) -> CarView:
    return CarView(genome_id=genome_id, x=x, y=y, heading=0.0, fitness=fitness)


WINDOW = (1280, 720)


def _model() -> ViewModel:
    return ViewModel(track_bounds(_track()), WINDOW)


def test_rank_cars_orders_by_fitness_then_lower_genome_id() -> None:
    cars = [_car(5, 1.0), _car(2, 3.0), _car(9, 3.0), _car(1, 2.0)]
    assert [c.genome_id for c in rank_cars(cars)] == [2, 9, 1, 5]


def test_rank_of_is_one_based_and_none_for_an_unknown_car() -> None:
    cars = [_car(1, 1.0), _car(2, 5.0)]
    assert rank_of(cars, cars[1]) == 1
    assert rank_of(cars, cars[0]) == 2
    assert rank_of(cars, _car(99, 0.0)) is None


def test_track_bounds_cover_every_cell_edge() -> None:
    track = _track()
    half = track.cell_size / 2
    assert track_bounds(track) == (
        -half,
        -half,
        3 * track.cell_size + half,
        2 * track.cell_size + half,
    )


def test_fit_mode_follows_nobody() -> None:
    assert _model().followed_car([_car(1, 1.0)], ViewSettings()) is None


def test_follow_best_tracks_the_leader_as_it_changes() -> None:
    model = _model()
    settings = ViewSettings(camera_mode="follow_best")
    a, b = _car(1, 1.0), _car(2, 2.0)
    assert model.followed_car([a, b], settings) == b
    b_behind = _car(2, 0.5)
    assert model.followed_car([a, b_behind], settings) == a


def test_follow_rank_locks_onto_a_car_and_holds_it_when_ranks_change() -> None:
    model = _model()
    settings = ViewSettings(camera_mode="follow_rank", follow_rank=2, follow_seq=1)
    cars = [_car(1, 3.0), _car(2, 2.0), _car(3, 1.0)]
    assert model.followed_car(cars, settings) is cars[1]
    # car 2 is overtaken by car 3, but the choice is held on the car, not the rank
    overtaken = [_car(1, 3.0), _car(2, 0.5), _car(3, 1.0)]
    assert model.followed_car(overtaken, settings) == overtaken[1]


def test_follow_rank_falls_back_to_the_best_remaining_car_when_the_followed_one_drops_out() -> None:
    model = _model()
    settings = ViewSettings(camera_mode="follow_rank", follow_rank=2, follow_seq=1)
    model.followed_car([_car(1, 3.0), _car(2, 2.0), _car(3, 1.0)], settings)
    remaining = [_car(1, 3.0), _car(3, 1.0)]  # car 2 crashed
    assert model.followed_car(remaining, settings).genome_id == 1  # type: ignore[union-attr]
    # and it stays on that car afterwards, even if car 3 later out-scores it
    later = [_car(1, 3.0), _car(3, 9.0)]
    assert model.followed_car(later, settings).genome_id == 1  # type: ignore[union-attr]


def test_a_new_follow_choice_re_resolves_even_for_the_same_rank() -> None:
    model = _model()
    first = ViewSettings(camera_mode="follow_rank", follow_rank=1, follow_seq=1)
    cars = [_car(1, 3.0), _car(2, 2.0)]
    assert model.followed_car(cars, first).genome_id == 1  # type: ignore[union-attr]
    swapped = [_car(1, 1.0), _car(2, 2.0)]  # car 2 now leads, car 1 is still followed
    assert model.followed_car(swapped, first).genome_id == 1  # type: ignore[union-attr]
    again = ViewSettings(camera_mode="follow_rank", follow_rank=1, follow_seq=2)
    assert model.followed_car(swapped, again).genome_id == 2  # type: ignore[union-attr]


def test_follow_rank_larger_than_the_field_picks_the_last_car() -> None:
    model = _model()
    settings = ViewSettings(camera_mode="follow_rank", follow_rank=50, follow_seq=1)
    cars = [_car(1, 3.0), _car(2, 2.0)]
    assert model.followed_car(cars, settings).genome_id == 2  # type: ignore[union-attr]


def test_no_cars_means_nobody_to_follow() -> None:
    settings = ViewSettings(camera_mode="follow_best")
    assert _model().followed_car([], settings) is None


def test_fit_camera_centres_on_the_track_and_scales_to_the_window() -> None:
    track = _track()
    min_x, min_y, max_x, max_y = track_bounds(track)
    model = _model()
    camera = model.camera(None, ViewSettings())
    expected = min(WINDOW[0] / (max_x - min_x), WINDOW[1] / (max_y - min_y)) / FIT_MARGIN
    assert camera.center_x == pytest.approx((min_x + max_x) / 2)
    assert camera.center_y == pytest.approx((min_y + max_y) / 2)
    assert camera.scale == pytest.approx(expected)
    assert model.camera(None, ViewSettings(zoom=2.0)).scale == pytest.approx(expected * 2)


def test_follow_camera_centres_on_the_car_at_the_base_scale() -> None:
    car = _car(1, 1.0, x=123.0, y=-45.0)
    camera = _model().camera(car, ViewSettings(zoom=2.0))
    assert (camera.center_x, camera.center_y) == (123.0, -45.0)
    assert camera.scale == pytest.approx(FOLLOW_BASE_SCALE * 2.0)


def _lines(**overrides: object) -> list[str]:
    base: dict[str, object] = {
        "connected": True,
        "run_state": "running",
        "generation": 12,
        "alive": 87,
        "population_size": 150,
        "speed": "1x",
        "settings": ViewSettings(),
        "followed_rank": None,
    }
    base.update(overrides)
    return overlay_lines(**base)  # type: ignore[arg-type]


def test_overlay_shows_generation_cars_speed_and_camera() -> None:
    assert _lines() == [
        "Generation 12",
        "Cars: 87 / 150",
        "Speed: 1x",
        "Camera: whole track",
    ]


def test_overlay_describes_each_camera_mode() -> None:
    assert "Camera: following the best car" in _lines(
        settings=ViewSettings(camera_mode="follow_best")
    )
    following = ViewSettings(camera_mode="follow_rank", follow_rank=3)
    assert "Camera: following #3" in _lines(settings=following, followed_rank=3)
    assert "Camera: following #?" in _lines(settings=following, followed_rank=None)


def test_overlay_when_disconnected_idle_or_starting() -> None:
    assert _lines(connected=False) == ["Disconnected from the dashboard backend - retrying..."]
    assert _lines(run_state="idle", generation=None, alive=0) == ["Waiting for a run..."]
    assert _lines(run_state="running", generation=None, alive=0) == ["Run starting..."]


def test_overlay_marks_a_finished_run() -> None:
    assert _lines(run_state="stopped")[-1] == "Run stopped"
