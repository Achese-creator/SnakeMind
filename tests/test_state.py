
import numpy as np
import pytest

from snakemind.agents.q_learning.state import encode_state


def make_state(
    head=(5, 5),
    tail=(3, 5),
    direction=1,
    food=(7, 5),
    body=None,
    size=10,
):
    observation = np.zeros((3, size, size), dtype=np.int8)

    observation[0, head[1], head[0]] = 1

    if body is not None:
        for x, y in body:
            observation[1, y, x] = 1

    observation[2, food[1], food[0]] = 1

    info = {
        "head": head,
        "tail": tail,
        "direction": direction,
        "food": food,
    }

    return observation, info


def test_state_has_ten_binary_features():
    observation, info = make_state()

    state = encode_state(observation, info)

    assert len(state) == 10
    assert all(value in (0, 1) for value in state)


def test_food_ahead_is_detected():
    observation, info = make_state(
        direction=1,
        food=(7, 5),
    )

    state = encode_state(observation, info)

    assert state[3] == 1
    assert state[4] == 0


def test_diagonal_food_activates_two_direction_features():
    observation, info = make_state(
        direction=1,
        food=(7, 3),
    )

    state = encode_state(observation, info)

    assert state[3] == 1  # Forward
    assert state[6] == 1  # Left


def test_food_distance_is_one_hot():
    cases = [
        ((6, 5), (1, 0, 0)),  # Distance 1: near
        ((9, 5), (0, 1, 0)),  # Distance 4: medium
        ((9, 9), (0, 0, 1)),  # Distance 8: far
    ]

    for food, expected in cases:
        observation, info = make_state(food=food)
        state = encode_state(observation, info)

        assert state[7:10] == expected


def test_wall_danger_is_detected():
    observation, info = make_state(
        head=(9, 5),
        tail=(7, 5),
        direction=1,
        food=(5, 5),
    )

    state = encode_state(observation, info)

    assert state[0] == 1


def test_body_collision_is_detected():
    observation, info = make_state(
        head=(5, 5),
        tail=(3, 5),
        direction=1,
        food=(8, 8),
        body=[(6, 5), (6, 6), (5, 6)],
    )

    state = encode_state(observation, info)

    assert state[0] == 1


def test_vacating_tail_is_not_marked_as_dangerous():
    observation, info = make_state(
        head=(5, 5),
        tail=(6, 5),
        direction=1,
        food=(8, 8),
        body=[(5, 6), (6, 6), (6, 5)],
    )

    state = encode_state(observation, info)

    assert state[0] == 0


def test_missing_food_is_rejected():
    observation, info = make_state()

    info["food"] = None

    with pytest.raises(ValueError):
        encode_state(observation, info)


def test_invalid_direction_is_rejected():
    observation, info = make_state()

    info["direction"] = 4

    with pytest.raises(ValueError):
        encode_state(observation, info)
