import numpy as np

from snakemind.agents.base import Agent


class RandomAgent(Agent):
    """Baseline agent that chooses uniformly random actions."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(
        self,
        observation: np.ndarray,
        info: dict,
    ) -> int:
        return int(self.rng.integers(0, 3))
