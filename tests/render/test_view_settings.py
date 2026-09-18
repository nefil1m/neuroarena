import pytest

from neuroarena.render.view_settings import CAMERA_MODES, ZOOM_MAX, ZOOM_MIN, ViewSettings


def test_defaults() -> None:
    settings = ViewSettings()
    assert settings.zoom == 1.0
    assert settings.camera_mode == "fit"
    assert settings.follow_rank == 1
    assert settings.follow_seq == 0
    assert CAMERA_MODES == ("fit", "follow_best", "follow_rank")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"camera_mode": "orbit"},
        {"zoom": ZOOM_MIN - 0.01},
        {"zoom": ZOOM_MAX + 0.01},
        {"follow_rank": 0},
    ],
)
def test_invalid_settings_raise(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ViewSettings(**kwargs)  # type: ignore[arg-type]


def test_updated_merges_a_partial_and_leaves_the_original_alone() -> None:
    original = ViewSettings()
    changed = original.updated({"zoom": 2.0, "camera_mode": "follow_best"})
    assert changed == ViewSettings(zoom=2.0, camera_mode="follow_best")
    assert original == ViewSettings()


def test_setting_follow_rank_bumps_follow_seq_even_when_unchanged() -> None:
    first = ViewSettings().updated({"follow_rank": 3})
    assert (first.follow_rank, first.follow_seq) == (3, 1)
    again = first.updated({"follow_rank": 3})
    assert (again.follow_rank, again.follow_seq) == (3, 2)
    assert first.updated({"zoom": 2.0}).follow_seq == 1  # other keys do not bump it


def test_updated_rejects_unknown_keys_and_invalid_values() -> None:
    with pytest.raises(ValueError):
        ViewSettings().updated({"bogus": 1})
    with pytest.raises(ValueError):
        ViewSettings().updated({"camera_mode": "orbit"})


def test_round_trips_through_a_dict() -> None:
    settings = ViewSettings(zoom=1.5, camera_mode="follow_rank", follow_rank=4, follow_seq=7)
    assert ViewSettings.from_dict(settings.to_dict()) == settings
