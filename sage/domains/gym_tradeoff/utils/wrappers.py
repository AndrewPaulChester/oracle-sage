import numpy as np

import gym
from gym import spaces
from gym_taxi.utils.representations import json_to_screen


# gym environment specific
CHANNEL_COUNT = 4
OUTPUT_IMAGE_SIZE = 84


class BoxWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(
            0, 1, (CHANNEL_COUNT, OUTPUT_IMAGE_SIZE, OUTPUT_IMAGE_SIZE), dtype=np.uint8
        )

    def observation(self, observation):
        return json_to_screen(observation)
