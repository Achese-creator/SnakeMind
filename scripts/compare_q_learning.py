from pathlib import Path

from snakemind.agents.random_agent import RandomAgent
from snakemind.agents.hamiltonian_agent import HamiltonianAgent
from snakemind.agents.q_learning.agent import QLearningAgent
from snakemind.agents.q_learning.training import train_agent
from snakemind.env import SnakeEnv
from snakemind.evaluation import (
    evaluate_agent,
    summarize_results,
    save_results_csv,
)

BOARD_SIZE = 10
MAX_STEPS = 10_000
TRAINING_EPISODES = 5_000
EVALUATION_SEEDS = list(range(100, 110))
TRAINING_SEED = 2026


def make_env():
    return SnakeEnv(
        size=BOARD_SIZE,
        max_episode_steps=MAX_STEPS,
    )


def main():
    print("=" * 55)
    print("SNAKEMIND: CONTROLLED AGENT COMPARISON")
    print("=" * 55)

    print(f"\nTraining Q-learning for {TRAINING_EPISODES} episodes...")

    q_agent, training_history = train_agent(
        env_factory=lambda: SnakeEnv(
            size=BOARD_SIZE,
            max_episode_steps=BOARD_SIZE * BOARD_SIZE * 4,
        ),
        episodes=TRAINING_EPISODES,
        seed=TRAINING_SEED,
        agent=QLearningAgent(
            epsilon_decay=0.999,
            seed=TRAINING_SEED,
        ),
    )

    # Disable exploration for policy evaluation.
    q_agent.epsilon = 0.0

    output_dir = Path("results/q_learning")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_path = output_dir / "training_metrics.csv"
    save_results_csv(training_history, training_path)

    print(f"Training completed. Visited states: {len(q_agent.q_table)}")
    print(f"Training metrics saved to: {training_path}")

    agents = {
        "RandomAgent": RandomAgent(seed=42),
        "HamiltonianAgent": HamiltonianAgent(),
        "QLearningAgent": q_agent,
    }

    all_results = []

    for name, agent in agents.items():
        print(f"\nEvaluating {name}...")

        results = evaluate_agent(
            agent=agent,
            env_factory=make_env,
            seeds=EVALUATION_SEEDS,
        )

        for record in results:
            record["agent"] = name

        summary = summarize_results(results)

        all_results.extend(results)

        print(f"\n{name}")
        for metric, value in summary.items():
            print(f"{metric}: {value}")

    output_path = output_dir / "agent_comparison.csv"
    save_results_csv(all_results, output_path)

    print("\n" + "=" * 55)
    print("COMPARISON COMPLETE")
    print(f"Results saved to: {output_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
