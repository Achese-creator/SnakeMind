from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SnakeEnv(gym.Env):
    """
    Gymnasium-compatible Snake environment.

    Actions:
        0: Continue straight
        1: Turn right
        2: Turn left

    Observation:
        Three binary channels:
        0: Snake head
        1: Snake body (excluding head)
        2: Food

    Coordinates are represented as (x, y).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 8}

    # Direction order: up, right, down, left.
    DIRECTIONS = (
        (0, -1),
        (1, 0),
        (0, 1),
        (-1, 0),
    )

    def __init__(
        self,
        size: int = 10,
        max_episode_steps: int | None = None,
        render_mode: str | None = None,
        cell_size: int = 24,
    ):
        super().__init__()

        if size < 4:
            raise ValueError("Board size must be at least 4.")

        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive.")

        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'.")

        if cell_size < 4:
            raise ValueError("cell_size must be at least 4.")

        self.size = size
        self.max_episode_steps = (
            max_episode_steps
            if max_episode_steps is not None
            else size * size * 4
        )
        self.render_mode = render_mode
        self.cell_size = cell_size

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.MultiBinary(
            (3, size, size)
        )

        self.body = deque()
        self.direction = 1
        self.food = None
        self.score = 0
        self.steps = 0
        self._episode_done = False
        self.termination_reason = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        center = self.size // 2

        self.body = deque([
            (center, center),
            (center - 1, center),
            (center - 2, center),
        ])

        self.direction = 1
        self.score = 0
        self.steps = 0
        self._episode_done = False
        self.termination_reason = None

        self.food = self._spawn_food()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        if self._episode_done:
            raise RuntimeError(
                "Episode has ended. Call reset() before stepping again."
            )

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # Convert relative action into an absolute direction.
        if action == 1:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            self.direction = (self.direction - 1) % 4

        dx, dy = self.DIRECTIONS[self.direction]

        head_x, head_y = self.body[0]
        new_head = (head_x + dx, head_y + dy)

        self.steps += 1

        eating = new_head == self.food

        out_of_bounds = (
            new_head[0] < 0
            or new_head[0] >= self.size
            or new_head[1] < 0
            or new_head[1] >= self.size
        )

        # The tail moves away during a normal move.
        # Therefore, moving into the current tail position is legal
        # when the snake is not eating.
        occupied_body = (
            self.body if eating else list(self.body)[:-1]
        )

        collision = (
            out_of_bounds
            or new_head in occupied_body
        )

        if collision:
            self._episode_done = True
            self.termination_reason = "collision"

            return (
                self._get_observation(),
                -10.0,
                True,
                False,
                self._get_info(),
            )

        self.body.appendleft(new_head)

        reward = -0.01
        terminated = False
        truncated = False

        if eating:
            self.score += 1
            reward += 10.0
            self.food = self._spawn_food()

            if self.food is None:
                reward += 100.0
                terminated = True
                self.termination_reason = "board_filled"
        else:
            self.body.pop()

        # A completed board takes precedence over the time limit.
        if not terminated and self.steps >= self.max_episode_steps:
            truncated = True
            self.termination_reason = "time_limit"

        self._episode_done = terminated or truncated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def _spawn_food(self):
        free_cells = [
            (x, y)
            for y in range(self.size)
            for x in range(self.size)
            if (x, y) not in self.body
        ]

        if not free_cells:
            return None

        index = int(self.np_random.integers(len(free_cells)))
        return free_cells[index]

    def _get_observation(self):
        observation = np.zeros(
            (3, self.size, self.size),
            dtype=np.int8,
        )

        head_x, head_y = self.body[0]
        observation[0, head_y, head_x] = 1

        for x, y in list(self.body)[1:]:
            observation[1, y, x] = 1

        if self.food is not None:
            food_x, food_y = self.food
            observation[2, food_y, food_x] = 1

        return observation

    def _get_info(self):
        return {
            "score": self.score,
            "length": len(self.body),
            "steps": self.steps,
            "food": self.food,
            "head": self.body[0] if self.body else None,
            "tail": self.body[-1] if self.body else None,
            "direction": self.direction,
            "termination_reason": self.termination_reason,
        }

    def render(self):
        if self.render_mode is None:
            return None

        image_size = self.size * self.cell_size

        # Background and grid.
        frame = np.full(
            (image_size, image_size, 3),
            (28, 30, 36),
            dtype=np.uint8,
        )

        for y in range(self.size):
            for x in range(self.size):
                x0 = x * self.cell_size
                y0 = y * self.cell_size

                frame[
                    y0 + 1:y0 + self.cell_size - 1,
                    x0 + 1:x0 + self.cell_size - 1,
                ] = (40, 44, 52)

        # Snake body.
        for x, y in list(self.body)[1:]:
            x0 = x * self.cell_size
            y0 = y * self.cell_size

            frame[
                y0 + 2:y0 + self.cell_size - 2,
                x0 + 2:x0 + self.cell_size - 2,
            ] = (55, 170, 95)

        # Snake head.
        if self.body:
            x, y = self.body[0]
            x0 = x * self.cell_size
            y0 = y * self.cell_size

            frame[
                y0 + 2:y0 + self.cell_size - 2,
                x0 + 2:x0 + self.cell_size - 2,
            ] = (120, 235, 130)

        # Food.
        if self.food is not None:
            x, y = self.food
            x0 = x * self.cell_size
            y0 = y * self.cell_size

            frame[
                y0 + 4:y0 + self.cell_size - 4,
                x0 + 4:x0 + self.cell_size - 4,
            ] = (235, 75, 75)

        return frame

    def close(self):
        pass
