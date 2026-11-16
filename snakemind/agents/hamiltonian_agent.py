import numpy as np

from snakemind.agents.base import Agent


class HamiltonianAgent(Agent):
    """
    Deterministic Snake agent that follows a Hamiltonian cycle.

    This implementation supports even square boards of size >= 6.
    It assumes the standard initial SnakeEnv.reset() configuration.

    Once aligned, following the cycle prevents self-collision until
    the board is filled, assuming the environment rules remain unchanged.
    """

    DIRECTIONS = (
        (0, -1),  # Up
        (1, 0),   # Right
        (0, 1),   # Down
        (-1, 0),  # Left
    )

    def __init__(self):
        self.cycle = None
        self.position_to_index = None

    def reset(self, seed: int | None = None) -> None:
        self.cycle = None
        self.position_to_index = None

    @staticmethod
    def _build_cycle(size: int) -> list[tuple[int, int]]:
        if size < 6 or size % 2 != 0:
            raise ValueError(
                "HamiltonianAgent requires an even board size >= 6."
            )

        cycle = []

        # Traverse the top row from left to right.
        for x in range(size):
            cycle.append((x, 0))

        # Serpentine traversal through the remaining rows,
        # excluding column zero.
        for y in range(1, size):
            if y % 2 == 1:
                columns = range(size - 1, 0, -1)
            else:
                columns = range(1, size)

            for x in columns:
                cycle.append((x, y))

        # Return upward through column zero.
        for y in range(size - 1, 0, -1):
            cycle.append((0, y))

        return cycle

    def _initialize_cycle(
        self,
        observation: np.ndarray,
        info: dict,
    ) -> None:
        size = observation.shape[1]

        if observation.shape != (3, size, size):
            raise ValueError("Expected a square three-channel observation.")

        head = info["head"]
        direction = info["direction"]

        dx, dy = self.DIRECTIONS[direction]

        # The standard environment starts with a straight snake of length 3.
        expected_body = [
            head,
            (head[0] - dx, head[1] - dy),
            (head[0] - 2 * dx, head[1] - 2 * dy),
        ]

        occupied = observation[0] | observation[1]

        for x, y in expected_body:
            if not (0 <= x < size and 0 <= y < size):
                raise ValueError("Initial snake body is outside the board.")

            if occupied[y, x] == 0:
                raise ValueError(
                    "HamiltonianAgent requires the standard initial "
                    "SnakeEnv.reset() state."
                )

        base_cycle = self._build_cycle(size)

        # Select the orientation whose previous two cells match
        # the snake's initial body order.
        for candidate in (base_cycle, list(reversed(base_cycle))):
            index_map = {
                position: index
                for index, position in enumerate(candidate)
            }

            head_index = index_map[head]

            previous_1 = candidate[
                (head_index - 1) % len(candidate)
            ]
            previous_2 = candidate[
                (head_index - 2) % len(candidate)
            ]

            if (
                previous_1 == expected_body[1]
                and previous_2 == expected_body[2]
            ):
                self.cycle = candidate
                self.position_to_index = index_map
                return

        raise ValueError(
            "Could not align the Hamiltonian cycle with the initial snake."
        )

    def act(
        self,
        observation: np.ndarray,
        info: dict,
    ) -> int:
        if self.cycle is None:
            self._initialize_cycle(observation, info)

        head = info["head"]
        current_direction = info["direction"]

        current_index = self.position_to_index[head]

        next_index = (current_index + 1) % len(self.cycle)
        next_position = self.cycle[next_index]

        dx = next_position[0] - head[0]
        dy = next_position[1] - head[1]

        desired_direction = self.DIRECTIONS.index((dx, dy))

        turn = (desired_direction - current_direction) % 4

        if turn == 0:
            return 0

        if turn == 1:
            return 1

        if turn == 3:
            return 2

        raise RuntimeError(
            "Hamiltonian cycle requested an immediate reversal."
        )
