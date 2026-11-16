from collections.abc import Callable
from typing import Any

from snakemind.agents.q_learning.agent import QLearningAgent
from snakemind.agents.q_learning.state import encode_state


def train_agent(
    env_factory: Callable[[], Any],
    episodes: int = 1000,
    seed: int = 42,
    agent: QLearningAgent | None = None,
) -> tuple[QLearningAgent, list[dict]]:
    """
    Train a tabular Q-learning agent.

    Returns the trained agent and one metrics record per episode.
    The environment is reset with a reproducible seed each episode.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")

    if agent is None:
        agent = QLearningAgent(seed=seed)

    env = env_factory()
    history = []

    try:
        for episode in range(episodes):
            episode_seed = seed + episode

            # Use a separate seed stream for the agent's exploration.
            agent.reset(seed=seed + 1_000_000 + episode)

            observation, info = env.reset(seed=episode_seed)

            episode_return = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                state = encode_state(observation, info)

                action = agent.act(observation, info)

                next_observation, reward, terminated, truncated, next_info = (
                    env.step(action)
                )

                # True terminal states do not bootstrap.
                # Time-limit truncations do bootstrap from the final state.
                next_state = (
                    None
                    if terminated
                    else encode_state(next_observation, next_info)
                )

                agent.update(
                    state=state,
                    action=action,
                    reward=float(reward),
                    next_state=next_state,
                    terminated=terminated,
                )

                episode_return += float(reward)
                observation = next_observation
                info = next_info

            history.append(
                {
                    "episode": episode + 1,
                    "score": info["score"],
                    "survival_steps": info["steps"],
                    "episode_return": episode_return,
                    "final_length": info["length"],
                    "terminated": terminated,
                    "truncated": truncated,
                    "termination_reason": info["termination_reason"],
                    "epsilon": agent.epsilon,
                }
            )

            agent.decay_epsilon()

    finally:
        env.close()

    return agent, history
