"""
.. module:: tradeoff_env
   :synopsis: Provides gym environment wrappers for the underlying simulator.
"""

import sys
import re
from contextlib import closing
from six import StringIO
import numpy as np
import gym
import json
from scipy.spatial import distance
import networkx as nx
from ast import literal_eval as make_tuple

from gym import error, spaces, utils
from gym.utils import seeding
from sage.domains.gym_tradeoff.simulator.tradeoff_world import TradeoffWorldSimulator

from sage.domains.utils.spaces import JsonGraph, BinaryAction
from sage.domains.utils.representations import (
    json_to_graph
)
from matplotlib import pyplot as plt

# from simulator
ACTION_COUNT = 3

# gym environment specific
CHANNEL_COUNT = 4
OUTPUT_IMAGE_SIZE = 84

OBS_SPACES = {
    ("screen", "original"): (4, None),
    ("mixed", "predictable15"): (2, 6),
    ("one-hot", "original"): (1, 75),
}

# SCENARIOS = {
#     "original": ORIGINAL,
#     "predictable": PREDICTABLE,
# }


# CONVERTERS = {
#     "screen": json_to_screen,
#     "mixed": json_to_mixed,
#     "mlp": json_to_mlp,
#     "both": json_to_both,
#     "one-hot": json_to_one_hot,
# }

DIRECTIONS = {
    (0, -1): "move-up",
    (0, 1): "move-down",
    (-1, 0): "move-left",
    (1, 0): "move-right",
}


def _decode(i):
    """
    Turns discrete representation into form needed to print the ascii map.
    from https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py
    """
    out = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5
    return reversed(out)


def _construct_image(representation, scenario):
    channels, length = OBS_SPACES[(representation, scenario)]
    if channels is None:  # purely MLP input
        return spaces.Box(0, 4, (length,), dtype=np.float32)
    if length is None:  # purely image based input
        return spaces.Box(
            0, 1, (channels, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), dtype=np.uint8
        )
    else:  # mixed input
        screen = spaces.Box(
            0, 1, (channels, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), dtype=np.uint8
        )
        dense = spaces.Box(0, 4, (length,), dtype=np.float32)
        return spaces.Tuple((screen, dense))


class BaseTradeoffEnv(gym.Env):
    """
    Base class for all gym tradeoff environments
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, **kwargs):
        self.steps = 0
        self.lastaction = None
        self.seed()
        self.sim = self._init_simulator(**kwargs)
        self.action_space = spaces.Discrete(3)
        self.score = 0
        self.sim_kwargs = kwargs

    def _init_simulator(self, **kwargs):
        return TradeoffWorldSimulator(self.np_random, **kwargs)

    def _step(self, action):
        self.steps += 1
        obs, reward, done, info = self.sim.act(action)
        self.score += reward
        info["score"] = self.score
        if done:
            # print(f"completed, score of: {self.score}")
            self.score = 0

        if self.steps == self.sim.timeout:
            done = True
            info["bad_transition"] = True
            # print(f"timed out, score of: {self.score}")
            self.score = 0

        info['s_true'] = obs
        info['d_true'] = done
        return obs, reward, done, info

    def reset(self):
        self.sim = self._init_simulator(**self.sim_kwargs)
        self.steps = 0
        self.score = 0
        return self.sim._get_state_json()

    def close(self):
        return

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class JsonTradeoffEnv(BaseTradeoffEnv):
    """
    Gym taxi environment for 5x5 gridworld with discrete observations
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = JsonGraph(converter=json_to_graph,node_dimension=4,edge_dimension=1)

    def step(self, action):
        obs, reward, done, info = self._step(action)
        return obs, reward, done, info
