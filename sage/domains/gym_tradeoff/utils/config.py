"""
.. module:: config
   :synopsis: Contains config parameters for the SDRL taxi world.
"""

import numpy as np
from numpy.random import randint as rand
import matplotlib.pyplot as pyplot


MAX_EPISODE_LENGTH = 500
DISCRETE_ENVIRONMENT_STATES = 500
FIXED_GRID_SIZE = 5
LOCS = [(0, 0), (4, 0), (0, 4), (3, 4)]
MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]
MISSING_EDGES = [
    ((1, 0), (2, 0)),
    ((0, 3), (1, 3)),
    ((0, 4), (1, 4)),
    ((2, 3), (3, 3)),
    ((2, 4), (3, 4)),
]

OPEN = {
    "size": FIXED_GRID_SIZE,
    "wall_locations": [],
    "passenger_destinations": LOCS,
    "passenger_locations": LOCS,
    "delivery_limit": 1,
    "concurrent_passengers": 1,
    "timeout": MAX_EPISODE_LENGTH,
    "random_walls": False,
}

