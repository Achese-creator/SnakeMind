import numpy as np
import pytest
from collections import deque

from snakemind.env import SnakeEnv


def test_reset_returns_valid_observation():
    env = SnakeEnv(size=10)

    observation, info = env.reset(seed=42)

    assert observation.shape == (3, 10, 10)
    assert env.observation_space.contains(observation)
    assert info["score"] == 0
    assert info["length"] == 3
    assert info["steps"] == 0


def test_reset_is_reproducible():
    env = SnakeEnv(size=10)

    obs1, info1 = env.reset(seed=123)
    obs2, info2 = env.reset(seed=123)

    assert np.array_equal(obs1, obs2)
    assert info1["food"] == info2["food"]


def test_food_never_spawns_on_snake():
    env = SnakeEnv(size=10)
    env.reset(seed=42)

    assert env.food not in env.body


def test_eating_food_increases_length_and_score():
    env = SnakeEnv(size=10)
    env.reset(seed=42)

    initial_length = len(env.body)
    head_x, head_y = env.body[0]
    dx, dy = env.DIRECTIONS[env.direction]

    env.food = (head_x + dx, head_y + dy)

    _, reward, terminated, truncated, info = env.step(0)

    assert len(env.body) == initial_length + 1
    assert info["score"] == 1
    assert reward == pytest.approx(9.99)
    assert not terminated
    assert not truncated


def test_wall_collision_terminates_episode():
    env = SnakeEnv(size=6)
    env.reset(seed=42)

    env.step(0)
    env.step(0)

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert reward == -10.0
    assert not truncated
    assert info["termination_reason"] == "collision"


def test_invalid_action_is_rejected():
    env = SnakeEnv(size=10)
    env.reset(seed=42)

    with pytest.raises(ValueError):
        env.step(5)


def test_episode_cannot_continue_after_termination():
    env = SnakeEnv(size=6)
    env.reset(seed=42)

    env.step(0)
    env.step(0)
    env.step(0)

    with pytest.raises(RuntimeError):
        env.step(0)


def test_episode_limit_causes_truncation():
    env = SnakeEnv(size=10, max_episode_steps=1)
    env.reset(seed=42)

    _, _, terminated, truncated, info = env.step(0)

    assert not terminated
    assert truncated
    assert info["termination_reason"] == "time_limit"


def test_tail_position_can_be_entered_when_tail_moves():
    env = SnakeEnv(size=6)
    env.reset(seed=42)

    env.body = deque([
        (2, 2),
        (2, 1),
        (1, 1),
        (1, 2),
    ])
    env.direction = 2  # Down.
    env.food = (4, 4)

    _, _, terminated, truncated, info = env.step(1)

    assert not terminated
    assert not truncated
    assert info["head"] == (1, 2)
    assert info["length"] == 4


def test_observation_has_separate_head_body_and_food_channels():
    env = SnakeEnv(size=10)
    observation, _ = env.reset(seed=42)

    assert observation[0].sum() == 1
    assert observation[1].sum() == 2
    assert observation[2].sum() == 1


def test_rgb_rendering_returns_correct_shape():
    env = SnakeEnv(size=10, render_mode="rgb_array", cell_size=20)
    env.reset(seed=42)

    frame = env.render()

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (200, 200, 3)
    assert frame.dtype == np.uint8


def test_food_spawn_returns_none_when_board_is_full():
    env = SnakeEnv(size=4)
    env.reset(seed=42)

    env.body = deque(
        (x, y)
        for y in range(4)
        for x in range(4)
    )

    assert env._spawn_food() is None


def test_invalid_board_size_is_rejected():
    with pytest.raises(ValueError):
        SnakeEnv(size=3)


def test_filling_entire_board_terminates_successfully():
    env = SnakeEnv(size=4)
    env.reset(seed=42)

    # Construct a board with one free cell directly ahead of the head.
    env.body = deque(
        [(1, 0)] + [
            (x, y)
            for y in range(4)
            for x in range(4)
            if (x, y) not in {(1, 0), (2, 0)}
        ]
    )
    env.direction = 1
    env.food = (2, 0)

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert len(env.body) == 16
    assert info["score"] == 1
    assert info["termination_reason"] == "board_filled"
    assert reward == pytest.approx(109.99)
