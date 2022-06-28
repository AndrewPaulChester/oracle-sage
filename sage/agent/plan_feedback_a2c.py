from typing import Any, Dict, Optional, Type, Union, Tuple
from numpy.core.numeric import roll
import math

import gym
import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from sage.domains.utils.spaces import Autoregressive

from sage.forks.stable_baselines3.stable_baselines3.common import logger
from sage.forks.stable_baselines3.stable_baselines3.a2c.a2c import A2C
from sage.forks.stable_baselines3.stable_baselines3.common.policies import ActorCriticPolicy
from sage.forks.stable_baselines3.stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from sage.forks.stable_baselines3.stable_baselines3.common.utils import explained_variance


from sage.forks.stable_baselines3.stable_baselines3.common import logger
from sage.forks.stable_baselines3.stable_baselines3.common.base_class import BaseAlgorithm
from sage.forks.stable_baselines3.stable_baselines3.common.buffers import RolloutBuffer
from sage.forks.stable_baselines3.stable_baselines3.common.callbacks import BaseCallback
from sage.forks.stable_baselines3.stable_baselines3.common.policies import ActorCriticPolicy
from sage.forks.stable_baselines3.stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from sage.forks.stable_baselines3.stable_baselines3.common.utils import safe_mean
from sage.forks.stable_baselines3.stable_baselines3.common.vec_env import VecEnv

from sage.agent.epsilon_buffer import EpsilonRolloutBuffer
from sage.agent.feedback_a2c import Feedback_A2C


class PlanFeedback_A2C(Feedback_A2C):
    """
    Feedback Advantage Actor Critic (A2C)
    Modified A2C algorithm which has an additional path value loss designed to help train the path value function in feedback-sage.


    Code: This implementation is built off the stable baselines3 implementation of A2C
    """

    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: EpsilonRolloutBuffer, n_rollout_steps: int
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        self.policy._on_step(self._current_progress_remaining)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor
                actions, values, log_probs, explored, plans = self.policy.forward(self._last_obs)
            actions = actions.cpu().numpy()

            plan_lengths = np.array([len(p) for p in plans])
            max_plan_length = max(plan_lengths)
            env_finished = plan_lengths == 0
            plans_grid = -np.ones((len(plan_lengths),max_plan_length),dtype=np.int) #create a grid of plans
            for i,p in enumerate(plans):
                plans_grid[i,:plan_lengths[i]]=p
            plan_step=0
            plan_rewards = np.zeros(len(plan_lengths))

            while not env_finished.all() and plan_step<max_plan_length:
                env_actions = plans_grid[:,plan_step]
                env_finished = np.logical_or(env_finished,(env_actions == -1)) #if any plans have invalid actions, they are finished
                self.env_steps+=np.logical_not(env_finished).sum()
                new_obs, rewards, dones, infos = env.step(env_actions,np.logical_not(env_finished))
                plan_rewards += ((math.pow(self.gamma,plan_step))*rewards)*(1-env_finished) #aggregate reward for all continuing environments
                env_finished = np.logical_or(env_finished,dones)
                plan_step +=1

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if (isinstance(self.action_space, gym.spaces.Discrete) or
               isinstance(self.action_space, Autoregressive) ):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, plan_rewards, self._last_dones, values, log_probs, explored)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            # #Fixing episode timeout - only correct for num-steps = 1
            # true_obs = [[x['s_true']] for x in infos]
            # true_done = np.array([x['d_true'] for x in infos],dtype=np.bool)
            # Compute value for the last timestep
            #obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _, _, _ = self.policy.forward(new_obs)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        logger.record("time/env_steps", self.env_steps)

        callback.on_rollout_end()

        return True
