"""
.. module:: taxi_world
   :synopsis: Simulates the taxi world environment based on actions passed in.
"""

from enum import Enum
import argparse
import numpy as np
import networkx as nx
from sage.domains.gym_tradeoff.utils.representations import env_to_json
from sage.domains.gym_tradeoff.utils.config import MAX_EPISODE_LENGTH
from sage.domains.gym_tradeoff.utils.utils import generate_random_walls



class TradeoffWorldSimulator(object):
    def __init__(
        self,
        random,
        k=3,
        n=5,
        relative_gains=False,
    ):
        """
        Houses the game state and transition dynamics for the tradeoff world.

        :param size: size of gridworld
        :returns: this is a description of what is returned
        :raises keyError: raises an exception
        """
        self.random = random
        self.n = n
        self.k = k
        self.relative_gains=relative_gains
        self.done = False
        self.timeout = MAX_EPISODE_LENGTH
        self.valid_actions = [(n*(i+1))+1 for i in range(k)]

        self.create_scenario()

    def _get_state_json(self):
        return env_to_json(self)

    def act(self, action):
        """
        Advances the game state by one step
        :param action: action provided by the agent
        :returns: observation of the next state
        :raises assertionError: raises an exception if action is invalid
        """
        # action, param = action

        #conversion between row action and cell action
        if action in self.valid_actions:
            action = int(((action-1)/self.n)-1)
        else:
            return self._get_state_json(), -1, False, {}


      

        reward = self.get_reward(action,self.relative_gains)
        self.gains = np.zeros_like(self.gains)
        self.costs = np.zeros_like(self.costs)
        return self._get_state_json(), reward, True, {}

    def create_scenario(self):
        self.gains = self.random.randint(0,10,(self.k,self.n))
        self.costs = self.random.randint(0,10,(self.k,self.n))
        self.done = False



    def get_reward(self, action, relative):
        score = float(self.gains[action].sum()-self.costs[action].sum())
        if relative:
            best = float(abs(np.max(self.gains.sum(1)-self.costs.sum(1))))
            return score if best == 0 else score/best
        else:
            return score