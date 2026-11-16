import pytest
from snakemind.env import SnakeEnv
from snakemind.agents import Agent, RandomAgent


def test_random_agent_implements_agent_interface():
    agent = RandomAgent(seed=42)

    assert isinstance(agent, Agent)


def test_random_agent_returns_valid_actions():
    env = SnakeEnv(size=10)
    observation, info = env.reset(seed=42)

    agent = RandomAgent(seed=42)

    for _ in range(100):
        action = agent.act(observation, info)
        assert action in (0, 1, 2)


def test_random_agent_is_reproducible():
    env = SnakeEnv(size=10)
    observation, info = env.reset(seed=42)

    agent_a = RandomAgent()
    agent_b = RandomAgent()

    agent_a.reset(seed=123)
    agent_b.reset(seed=123)

    actions_a = [
        agent_a.act(observation, info)
        for _ in range(50)
    ]

    actions_b = [
        agent_b.act(observation, info)
        for _ in range(50)
    ]

    assert actions_a == actions_b


def test_random_agent_reset_reinitializes_rng():
    env = SnakeEnv(size=10)
    observation, info = env.reset(seed=42)

    agent = RandomAgent()

    agent.reset(seed=99)
    first_sequence = [
        agent.act(observation, info)
        for _ in range(20)
    ]

    agent.reset(seed=99)
    second_sequence = [
        agent.act(observation, info)
        for _ in range(20)
    ]

    assert first_sequence == second_sequence

def test_hamiltonian_cycle_visits_every_cell_once():
    from snakemind.agents import HamiltonianAgent

    size = 10
    cycle = HamiltonianAgent._build_cycle(size)

    assert len(cycle) == size * size
    assert len(set(cycle)) == size * size

    for index, (x, y) in enumerate(cycle):
        next_x, next_y = cycle[(index + 1) % len(cycle)]

        distance = abs(next_x - x) + abs(next_y - y)
        assert distance == 1


def test_hamiltonian_agent_starts_by_moving_straight():
    from snakemind.agents import HamiltonianAgent

    env = SnakeEnv(size=10)
    observation, info = env.reset(seed=42)

    agent = HamiltonianAgent()
    agent.reset(seed=42)

    assert agent.act(observation, info) == 0


def test_hamiltonian_agent_survives_initial_100_steps():
    from snakemind.agents import HamiltonianAgent

    env = SnakeEnv(size=10, max_episode_steps=2000)
    observation, info = env.reset(seed=42)

    agent = HamiltonianAgent()
    agent.reset(seed=42)

    for _ in range(100):
        action = agent.act(observation, info)

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        assert not terminated
        assert not truncated


def test_hamiltonian_agent_rejects_unsupported_board_size():
    from snakemind.agents import HamiltonianAgent

    env = SnakeEnv(size=9)
    observation, info = env.reset(seed=42)

    agent = HamiltonianAgent()

    with pytest.raises(ValueError):
        agent.act(observation, info)
