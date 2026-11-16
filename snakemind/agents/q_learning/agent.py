
import numpy as np

from snakemind.agents.base import Agent
from snakemind.agents.q_learning.state import encode_state


class QLearningAgent(Agent):
    """Tabular Q-learning agent for the Snake environment."""

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        seed: int | None = None,
    ):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1].")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1].")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1].")
        if not 0 <= min_epsilon <= 1:
            raise ValueError("min_epsilon must be in [0, 1].")
        if epsilon == 0.0:
            min_epsilon = 0.0
        elif min_epsilon > epsilon:
            raise ValueError("min_epsilon cannot exceed initial epsilon.")

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.rng = np.random.default_rng(seed)

        # Sparse Q-table: only visited states are stored.
        self.q_table: dict[tuple[int, ...], np.ndarray] = {}

    def reset(self, seed: int | None = None) -> None:
        """Reset episode randomness without discarding learned knowledge."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _get_q_values(self, state: tuple[int, ...]) -> np.ndarray:
        """Return the Q-values for a state, initializing unseen states."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3, dtype=np.float64)

        return self.q_table[state]

    def act(self, observation: np.ndarray, info: dict) -> int:
        """Choose an action using epsilon-greedy exploration."""
        state = encode_state(observation, info)

        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, 3))

        q_values = self._get_q_values(state)

        # Randomly break ties between equally valuable actions.
        # The agent's seeded RNG preserves reproducibility.
        best_value = np.max(q_values)
        best_actions = np.flatnonzero(q_values == best_value)

        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: tuple[int, ...],
        action: int,
        reward: float,
        next_state: tuple[int, ...] | None,
        terminated: bool,
    ) -> float:
        """Apply one Q-learning update and return the TD error."""
        if action not in (0, 1, 2):
            raise ValueError("action must be 0, 1 or 2.")

        if not np.isfinite(reward):
            raise ValueError("reward must be finite.")

        if not terminated and next_state is None:
            raise ValueError(
                "next_state is required for non-terminal transitions."
            )

        current_q = self._get_q_values(state)[action]

        if terminated:
            target = reward
        else:
            next_q_values = self._get_q_values(next_state)
            target = reward + self.gamma * float(np.max(next_q_values))

        td_error = target - current_q

        self.q_table[state][action] += self.alpha * td_error

        return float(td_error)

    def decay_epsilon(self) -> None:
        """Reduce exploration gradually, respecting the minimum."""
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )
