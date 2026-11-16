
import numpy as np


# Relative actions: straight, right, left.
RELATIVE_ACTION_TURNS = (0, 1, -1)

# Direction order matches SnakeEnv:
# up, right, down, left.
DIRECTIONS = (
    (0, -1),
    (1, 0),
    (0, 1),
    (-1, 0),
)


def encode_state(
    observation: np.ndarray,
    info: dict,
) -> tuple[int, ...]:
    """
    Convert a Snake observation into a compact 10-bit state.

    Feature order:
        0. Danger straight
        1. Danger right
        2. Danger left
        3. Food forward
        4. Food backward
        5. Food right
        6. Food left
        7. Food distance near (0-3)
        8. Food distance medium (4-7)
        9. Food distance far (8+)

    Returns:
        A tuple containing ten binary values.

    Note:
        This is a compact state abstraction. It does not represent
        the complete snake body configuration and therefore permits
        state aliasing.
    """
    observation = np.asarray(observation)

    if observation.ndim != 3 or observation.shape[0] != 3:
        raise ValueError(
            "Expected an observation with three spatial channels."
        )

    if observation.shape[1] != observation.shape[2]:
        raise ValueError("Expected a square board.")

    size = observation.shape[1]

    head = info.get("head")
    tail = info.get("tail")
    food = info.get("food")
    direction = info.get("direction")

    if head is None or tail is None or food is None:
        raise ValueError(
            "State encoding requires head, tail and food positions."
        )

    if not isinstance(direction, (int, np.integer)):
        raise ValueError("Direction must be an integer.")

    if not 0 <= direction < 4:
        raise ValueError("Direction must be between 0 and 3.")

    head_x, head_y = head
    tail_x, tail_y = tail
    food_x, food_y = food

    for name, position in (
        ("head", head),
        ("tail", tail),
        ("food", food),
    ):
        if (
            len(position) != 2
            or not all(
                isinstance(value, (int, np.integer))
                for value in position
            )
        ):
            raise ValueError(f"Invalid {name} coordinates.")

        x, y = position

        if not (0 <= x < size and 0 <= y < size):
            raise ValueError(f"{name} is outside the board.")

    if observation[0, head_y, head_x] != 1:
        raise ValueError("Head coordinates do not match observation.")

    if food_x == head_x and food_y == head_y:
        raise ValueError("Food cannot occupy the snake's head position.")

    # Extract immediate danger for each relative action.
    danger_features = []

    for turn in RELATIVE_ACTION_TURNS:
        next_direction = (direction + turn) % 4
        dx, dy = DIRECTIONS[next_direction]

        next_head = (head_x + dx, head_y + dy)
        next_x, next_y = next_head

        out_of_bounds = not (
            0 <= next_x < size
            and 0 <= next_y < size
        )

        eating = next_head == food

        occupied_body = observation[1].astype(bool)

        # The tail vacates its position during a normal move,
        # but remains occupied if the move results in eating.
        hits_body = False

        if not out_of_bounds:
            hits_body = bool(occupied_body[next_y, next_x])

            if next_head == tail and not eating:
                hits_body = False

        danger_features.append(
            int(out_of_bounds or hits_body)
        )

    # Convert food position into the snake's relative reference frame.
    forward_x, forward_y = DIRECTIONS[direction]

    # In screen coordinates, clockwise rotation is (-dy, dx).
    right_x, right_y = -forward_y, forward_x

    relative_x = food_x - head_x
    relative_y = food_y - head_y

    forward_projection = (
        relative_x * forward_x
        + relative_y * forward_y
    )

    right_projection = (
        relative_x * right_x
        + relative_y * right_y
    )

    food_features = [
        int(forward_projection > 0),
        int(forward_projection < 0),
        int(right_projection > 0),
        int(right_projection < 0),
    ]

    distance = abs(relative_x) + abs(relative_y)

    distance_features = [
        int(0 <= distance <= 3),
        int(4 <= distance <= 7),
        int(distance >= 8),
    ]

    state = tuple(
        danger_features
        + food_features
        + distance_features
    )

    return state
