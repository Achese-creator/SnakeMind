from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    """Common interface for all SnakeMind agents."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        """Reset agent state and optionally initialize its RNG."""
        raise NotImplementedError

    @abstractmethod
    def act(
        self,
        observation: np.ndarray,
        info: dict,
    ) -> int:
        """
        Select an action.

        Returns:
            0: Straight
            1: Right
            2: Left
        """
        raise NotImplementedError
