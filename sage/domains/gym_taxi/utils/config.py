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

ORIGINAL = {
    "size": FIXED_GRID_SIZE,
    "wall_locations": MISSING_EDGES,
    "passenger_destinations": LOCS,
    "passenger_locations": LOCS,
    "delivery_limit": 1,
    "concurrent_passengers": 1,
    "timeout": MAX_EPISODE_LENGTH,
    "random_walls": False,
}


EXPANDED = {
    "size": 20,
    "delivery_limit": 1,
    "concurrent_passengers": 1,
    "timeout": MAX_EPISODE_LENGTH,
    "passenger_creation_probability": 1,
    "random_walls": True,
}

MULTI = {
    "size": 20,
    "delivery_limit": 100,
    "concurrent_passengers": 5,
    "timeout": MAX_EPISODE_LENGTH,
    "passenger_creation_probability": 0.05,
    "random_walls": True,
}

PREDICTABLE = {
    "size": 20,
    "delivery_limit": 100,
    "concurrent_passengers": 1,
    "timeout": 2000,
    "passenger_creation_probability": 0.04,
    "random_walls": True,
    "passenger_locations": [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ],
    "passenger_destinations": [
        (17, 17),
        (17, 18),
        (17, 19),
        (18, 17),
        (18, 18),
        (18, 19),
        (19, 17),
        (19, 18),
        (19, 19),
    ],
}


PREDICTABLE15 = {
    "size": 15,
    "delivery_limit": 100,
    "concurrent_passengers": 1,
    "timeout": 2000,
    "passenger_creation_probability": 0.06,
    "random_walls": True,
    "passenger_locations": [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ],
    "passenger_destinations": [
        (12, 12),
        (12, 13),
        (12, 14),
        (13, 12),
        (13, 13),
        (13, 14),
        (14, 12),
        (14, 13),
        (14, 14),
    ],
}


PREDICTABLE10 = {
    "size": 10,
    "delivery_limit": 100,
    "concurrent_passengers": 1,
    "timeout": 2000,
    "passenger_creation_probability": 0.08,
    "random_walls": True,
    "passenger_locations": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "passenger_destinations": [(8, 8), (8, 9), (9, 8), (9, 9)],
}


PREDICTABLE5 = {
    "size": 5,
    "delivery_limit": 100,
    "wall_locations": MISSING_EDGES,
    "concurrent_passengers": 1,
    "timeout": 1000,
    "passenger_creation_probability": 0.12,
    "random_walls": False,
    "passenger_locations": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "passenger_destinations": [(3, 3), (3, 4), (4, 3), (4, 4)],
}

FUEL = {
    "size": 20,
    "delivery_limit": 100,
    "concurrent_passengers": 5,
    "timeout": MAX_EPISODE_LENGTH,
    "passenger_creation_probability": 0.05,
    "random_walls": True,
    "fuel_use": 1,
}

CITY = {
    "size": 20,
    "delivery_limit": 100,
    "concurrent_passengers": 20,
    "timeout": 2000,
    "passenger_creation_probability": 0.1,
    "random_walls": True,
}
