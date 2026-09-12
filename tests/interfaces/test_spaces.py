from neuroarena.interfaces.spaces import Box, Discrete


def test_box_equal_when_all_fields_match():
    assert Box(-1.0, 1.0, (10,)) == Box(-1.0, 1.0, (10,))


def test_box_unequal_on_shape():
    assert Box(-1.0, 1.0, (10,)) != Box(-1.0, 1.0, (7,))


def test_box_unequal_on_bounds():
    assert Box(-1.0, 1.0, (10,)) != Box(0.0, 1.0, (10,))


def test_box_unequal_on_dtype():
    assert Box(-1.0, 1.0, (10,), "float32") != Box(-1.0, 1.0, (10,), "float64")


def test_box_is_hashable_and_hash_matches_equal_instances():
    assert hash(Box(-1.0, 1.0, (10,))) == hash(Box(-1.0, 1.0, (10,)))


def test_discrete_equality_and_inequality():
    assert Discrete(4) == Discrete(4)
    assert Discrete(4) != Discrete(2)


def test_box_never_equals_discrete():
    assert Box(0.0, 1.0, (1,)) != Discrete(1)  # type: ignore[comparison-overlap]


def test_box_repr_contains_type_and_shape():
    r = repr(Box(-1.0, 1.0, (10,)))
    assert "Box" in r and "10" in r
