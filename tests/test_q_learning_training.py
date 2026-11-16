import numpy as np
import pytest

from snakemind.agents.q_learning.training import train_agent
from snakemind.env import SnakeEnv


def make_env():
    return SnakeEnv(size=10, max_episode_steps=20)


def test_training_returns_agent_and_episode_history():
    agent, history = train_agent(
        env_factory=make_env,
        episodes=3,
        seed=42,
    )

    assert len(history) == 3
    assert len(agent.q_table) > 0
    assert agent.epsilon < 1.0

    required_fields = {
        "episode",
        "score",
        "survival_steps",
        "episode_return",
        "final_length",
        "terminated",
        "truncated",
        "termination_reason",
        "epsilon",
    }

    assert required_fields.issubset(history[0].keys())


def test_training_is_reproducible():
    first_agent, first_history = train_agent(
        env_factory=make_env,
        episodes=5,
        seed=123,
    )

    second_agent, second_history = train_agent(
        env_factory=make_env,
        episodes=5,
        seed=123,
    )

    assert first_history == second_history
    assert first_agent.epsilon == pytest.approx(second_agent.epsilon)
    assert first_agent.q_table.keys() == second_agent.q_table.keys()

    for state in first_agent.q_table:
        np.testing.assert_array_equal(
            first_agent.q_table[state],
            second_agent.q_table[state],
        )


def test_training_rejects_zero_episodes():
    with pytest.raises(ValueError):
        train_agent(
            env_factory=make_env,
            episodes=0,
        )
