"""Unit tests for the T5R6 verdict counting rule (pure, no data)."""


def rule(full, base, direction):
    n_ok = sum(1 for v in full if v == v)
    n_above = sum(1 for f, b in zip(full, base) if f == f and b == b and f > b)
    if n_ok == 8 and n_above == 8 and all(d > 0.5 for d in direction if d == d):
        return "T5R6_CONFIRMED"
    return "T5R6_MIXED" if n_above >= 5 else "T5R6_NOT_CONFIRMED"


def test_confirmed_all_eight():
    assert rule([0.5] * 8, [0.0] * 8, [0.7] * 8) == "T5R6_CONFIRMED"


def test_mixed_majority():
    assert rule([0.5] * 5 + [0.0] * 3, [0.0] * 8, [0.7] * 8) == "T5R6_MIXED"


def test_not_confirmed_minority():
    assert rule([0.5] * 4 + [0.0] * 4, [0.0] * 8, [0.7] * 8) == "T5R6_NOT_CONFIRMED"


def test_direction_failure_blocks_confirmed():
    assert rule([0.5] * 8, [0.0] * 8, [0.7] * 7 + [0.4]) == "T5R6_MIXED"
