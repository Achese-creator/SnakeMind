
import numpy as np
import pytest

from snakemind.agents.base import Agent
from snakemind.agents.q_learning.agent import QLearningAgent


def make_observation():
    observation = np.zeros((3, 10, 10), dtype=np.int8)
    observation[0, 5, 5] = 1
    observation[1, 5, 4] = 1
    observation[1, 5, 3] = 1
    observation[2, 5, 8] = 1

    info = {
        "head": (5, 5),
        "tail": (3, 5),
        "food": (8, 5),
        "direction": 1,
    }

    return observation, info


def test_agent_implements_common_interface():
    agent = QLearningAgent()
    assert isinstance(agent, Agent)


def test_agent_returns_valid_action():
    agent = QLearningAgent(seed=42)
    observation, info = make_observation()

    action = agent.act(observation, info)

    assert action in (0, 1, 2)


def test_seeded_agents_are_reproducible():
    observation, info = make_observation()

    first = QLearningAgent(seed=42)
    second = QLearningAgent(seed=42)

    actions_first = [
        first.act(observation, info) for _ in range(20)
    ]
    actions_second = [
        second.act(observation, info) for _ in range(20)
    ]

    assert actions_first == actions_second


def test_greedy_action_selects_highest_q_value():
    agent = QLearningAgent(epsilon=0.0)
    observation, info = make_observation()

    from snakemind.agents.q_learning.state import encode_state
    state = encode_state(observation, info)

    agent.q_table[state] = np.array([1.0, 5.0, 2.0])

    assert agent.act(observation, info) == 1


def test_greedy_tie_breaking_can_select_all_equal_actions():
    observation, info = make_observation()

    agent = QLearningAgent(epsilon=0.0, seed=42)

    from snakemind.agents.q_learning.state import encode_state
    state = encode_state(observation, info)

    agent.q_table[state] = np.array([0.0, 0.0, 0.0])

    actions = {
        agent.act(observation, info)
        for _ in range(300)
    }

    assert actions == {0, 1, 2}


def test_terminal_update_does_not_bootstrap():
    agent = QLearningAgent(alpha=0.5, gamma=0.9)

    state = (0,) * 10
    next_state = (1,) * 10

    agent.q_table[state] = np.array([2.0, 0.0, 0.0])
    agent.q_table[next_state] = np.array([100.0, 50.0, 25.0])

    agent.update(state, 0, -10.0, next_state, terminated=True)

    assert agent.q_table[state][0] == pytest.approx(-4.0)


def test_non_terminal_update_bootstraps():
    agent = QLearningAgent(alpha=0.5, gamma=0.9)

    state = (0,) * 10
    next_state = (1,) * 10

    agent.q_table[state] = np.zeros(3)
    agent.q_table[next_state] = np.array([10.0, 5.0, 2.0])

    agent.update(state, 0, 1.0, next_state, terminated=False)

    assert agent.q_table[state][0] == pytest.approx(5.0)


def test_epsilon_decay_respects_minimum():
    agent = QLearningAgent(
        epsilon=0.5,
        epsilon_decay=0.5,
        min_epsilon=0.1,
    )

    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.25)

    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.125)

    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.1)

    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.1)


def test_reset_preserves_learned_q_values():
    agent = QLearningAgent(seed=42)

    state = (0,) * 10
    agent.q_table[state] = np.array([1.0, 2.0, 3.0])

    agent.reset(seed=100)

    np.testing.assert_array_equal(
        agent.q_table[state],
        np.array([1.0, 2.0, 3.0]),
    )


def test_update_rejects_invalid_action():
    agent = QLearningAgent()

    with pytest.raises(ValueError):
        agent.update((0,) * 10, 3, 1.0, (1,) * 10, False)


def test_non_terminal_update_requires_next_state():
    agent = QLearningAgent()

    with pytest.raises(ValueError):
        agent.update((0,) * 10, 0, 1.0, None, False)
