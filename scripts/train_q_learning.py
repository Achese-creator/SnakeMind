from pathlib import Path

from snakemind.agents.q_learning.training import train_agent
from snakemind.env import SnakeEnv
from snakemind.evaluation import save_results_csv


EPISODES = 5000
SEED = 2026
BOARD_SIZE = 10


def main():
    print("Starting SnakeMind Q-learning training...")
    print(f"Episodes: {EPISODES}")
    print(f"Board size: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"Random seed: {SEED}")

    agent, history = train_agent(
        env_factory=lambda: SnakeEnv(
            size=BOARD_SIZE,
            max_episode_steps=BOARD_SIZE * BOARD_SIZE * 4,
        ),
        episodes=EPISODES,
        seed=SEED,
    )

    output_dir = Path("results/q_learning")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "training_metrics.csv"
    save_results_csv(history, metrics_path)

    recent = history[-100:]
    average_score = sum(row["score"] for row in recent) / len(recent)
    average_steps = (
        sum(row["survival_steps"] for row in recent) / len(recent)
    )

    print("\nTraining completed.")
    print(f"Visited states: {len(agent.q_table)}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Average score (last 100 episodes): {average_score:.2f}")
    print(f"Average survival (last 100 episodes): {average_steps:.1f}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
