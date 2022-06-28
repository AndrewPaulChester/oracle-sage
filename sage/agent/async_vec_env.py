from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Type, Union

import gym
import numpy as np

from sage.forks.stable_baselines3.stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from sage.forks.stable_baselines3.stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from sage.forks.stable_baselines3.stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv


class AsyncVecEnv(DummyVecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.mask = None
        self.metadata = env.metadata

    def step(self, actions: np.ndarray, mask: np.ndarray) -> VecEnvStepReturn:
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        self.step_async(actions, mask)
        return self.step_wait()

    def step_async(self, actions: np.ndarray, mask: np.ndarray) -> None:
        self.actions = actions
        self.mask = mask

    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            if self.mask[env_idx]:
                obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                    self.actions[env_idx]
                )
                if self.buf_dones[env_idx]:
                    # save final observation where user can get it, then reset
                    self.buf_infos[env_idx]["terminal_observation"] = obs
                    obs = self.envs[env_idx].reset()
                self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))
