"""
.. module:: utils
   :synopsis: Contains miscellaneous utility functions.
"""
import numpy as np
from math import floor, ceil


def _json_default(obj):
    """
    Converts numpy types to standard types for serialisation into JSON.
    :param obj: object to be converted
    :returns: object as a standard type
    :raises TypeError: if the object is not a numpy type
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError


def generate_maze(random, width=20, height=20, complexity=0.1, density=0.02):
    """
    Creates a randomly generated rectangular maze on a square grid.
    :param width: width of maze
    :param height: height of maze
    :param complexity: Controls the length of each maze segment (as a percentage)
    :param density: Controls the number of maze segments (as a percentage)
    :returns: Array containing maze layout (True = wall)
    """
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))  # number of components
    density = int(density * ((shape[0] // 2) * (shape[1] // 2)))  # size of components
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make aisles
    for i in range(density):
        x, y = (
            random.randint(0, shape[1] // 2) * 2,
            random.randint(0, shape[0] // 2) * 2,
        )  # pick a random position
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - 2))
            if x < shape[1] - 2:
                neighbours.append((y, x + 2))
            if y > 1:
                neighbours.append((y - 2, x))
            if y < shape[0] - 2:
                neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[random.randint(0, len(neighbours) - 1)]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
    return Z


def generate_random_walls(grid_size, random, complexity=0.1, density=0.02):
    """
    Creates a list of random walls .
    :param grid_size: dimension of gridworld
    :param complexity: Controls the length of each wall section (as a percentage)
    :param density: Controls the number of wall section (as a percentage)
    :returns: List of walls represented as edges in lattice
    :raises AssertError: if complexity and density values are invalid, must be in (0,1]
    """
    assert (
        complexity > 0 and complexity <= 1
    ), f"complexity value must be in the range (0,1] (currently {complexity})"
    assert (
        density > 0 and density <= 1
    ), f"density value must be in the range (0,1] (currently {density})"
    maze_size = 2 * grid_size + 1
    m = generate_maze(random, maze_size, maze_size, complexity, density)[1:-1, 1:-1]
    walls = []
    for ((x, y), v) in np.ndenumerate(m):
        if v and x % 2 + y % 2 == 1:  # xor of odd(x) and odd(y)
            walls.append(
                (
                    (int(floor(x / 2)), int(floor(y / 2))),
                    (int(ceil(x / 2)), int(ceil(y / 2))),
                )
            )
    return walls

