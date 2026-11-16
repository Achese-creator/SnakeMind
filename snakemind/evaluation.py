import csv
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable

from gymnasium import Env

from snakemind.agents.base import Agent


def evaluate_agent(
    agent: Agent,
    env_factory: Callable[[], Env],
    seeds: list[int],
) -> list[dict]:
    """
    Evaluate an agent over independent episodes.

    The agent's learned parameters, if any, should remain unchanged
    during evaluation. Its reset() method should only reset
    episode-specific state.
    """
    if not seeds:
        raise ValueError("At least one evaluation seed is required.")

    results = []

    for seed in seeds:
        env = env_factory()

        try:
            agent.reset(seed=seed)
            observation, info = env.reset(seed=seed)

            episode_return = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.act(observation, info)

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                episode_return += float(reward)

            results.append({
                "seed": seed,
                "score": info["score"],
                "survival_steps": info["steps"],
                "episode_return": episode_return,
                "final_length": info["length"],
                "terminated": terminated,
                "truncated": truncated,
                "termination_reason": info["termination_reason"],
            })

        finally:
            env.close()

    return results


def summarize_results(results: list[dict]) -> dict:
    """Calculate descriptive statistics across evaluation episodes."""
    if not results:
        raise ValueError("Cannot summarize an empty result set.")

    scores = [row["score"] for row in results]
    survival = [row["survival_steps"] for row in results]
    returns = [row["episode_return"] for row in results]
    lengths = [row["final_length"] for row in results]

    completed = sum(
        row["termination_reason"] == "board_filled"
        for row in results
    )

    return {
        "episodes": len(results),
        "mean_score": mean(scores),
        "std_score": pstdev(scores),
        "mean_survival_steps": mean(survival),
        "std_survival_steps": pstdev(survival),
        "mean_episode_return": mean(returns),
        "std_episode_return": pstdev(returns),
        "mean_final_length": mean(lengths),
        "board_completion_rate": completed / len(results),
    }


def save_results_csv(
    results: list[dict],
    filepath: str | Path,
) -> None:
    """Save episode-level evaluation results to CSV."""
    if not results:
        raise ValueError("Cannot save an empty result set.")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
