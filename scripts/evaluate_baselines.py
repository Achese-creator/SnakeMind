from pathlib import Path

from snakemind.agents import RandomAgent, HamiltonianAgent
from snakemind.env import SnakeEnv
from snakemind.evaluation import (
    evaluate_agent,
    summarize_results,
    save_results_csv,
)


BOARD_SIZE = 10
MAX_EPISODE_STEPS = 10000
EVALUATION_SEEDS = list(range(100, 110))

OUTPUT_DIR = Path("results/baselines")


def run_evaluation(name, agent):
    results = evaluate_agent(
        agent=agent,
        env_factory=lambda: SnakeEnv(
            size=BOARD_SIZE,
            max_episode_steps=MAX_EPISODE_STEPS,
        ),
        seeds=EVALUATION_SEEDS,
    )

    summary = summarize_results(results)

    save_results_csv(
        results,
        OUTPUT_DIR / f"{name.lower()}_results.csv",
    )

    print(f"\n{'=' * 45}")
    print(name)
    print("=" * 45)

    for metric, value in summary.items():
        print(f"{metric}: {value}")

    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random_summary = run_evaluation(
        "RandomAgent",
        RandomAgent(),
    )

    hamiltonian_summary = run_evaluation(
        "HamiltonianAgent",
        HamiltonianAgent(),
    )

    print("\nEvaluation completed.")
    print(f"Results saved to: {OUTPUT_DIR}")

    return random_summary, hamiltonian_summary


if __name__ == "__main__":
    main()
