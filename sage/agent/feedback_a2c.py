from typing import Any, Dict, Optional, Type, Union, Tuple
from numpy.core.numeric import roll

import gym
import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

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
from sage.domains.utils.spaces import Autoregressive

class Feedback_A2C(A2C):
    """
    Feedback Advantage Actor Critic (A2C)
    Modified A2C algorithm which has an additional path value loss designed to help train the path value function in feedback-sage.


    Code: This implementation is built off the stable baselines3 implementation of A2C

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param normalize_advantage: Whether to normalize or not the advantage
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        policy_coef: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        pvf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        sample_entropy: bool = False,
        tis_heuristic: float = None,
        supported_action_spaces: Optional[Tuple[spaces.Space, ...]] = (
            spaces.Box,
            spaces.Discrete,
            spaces.MultiDiscrete,
            spaces.MultiBinary,
        )
    ):

        super(Feedback_A2C, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            policy_coef=policy_coef,
            max_grad_norm=max_grad_norm,
            rms_prop_eps=rms_prop_eps,
            use_rms_prop=use_rms_prop,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            normalize_advantage = normalize_advantage,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=_init_setup_model,
            sample_entropy=sample_entropy,
            supported_action_spaces=supported_action_spaces,
        )

        self.pvf_coef=pvf_coef
        self.tis_heuristic=tis_heuristic
        self.env_steps = 0

        if _init_setup_model: #overwrite base rollout buffer with explored version.
            self.rollout_buffer = EpsilonRolloutBuffer(
                self.n_steps,
                self.observation_space,
                self.action_space,
                self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
            )

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            if (isinstance(self.action_space, spaces.Discrete) or
               isinstance(self.action_space, Autoregressive) ):
                # Convert discrete action from float to long
                actions = actions.long().flatten()

            
            values, log_prob, entropy, path_values = self.policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage (not present in the original implementation)
            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob)

            if self.tis_heuristic is not None:
                probs = th.exp(log_prob.detach())
                policy_loss *= (probs*self.tis_heuristic).clamp(0,1) #clamp is the truncated in truncated importance sampling

            policy_loss = policy_loss.mean()


            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns*(1-rollout_data.explored), values*(1-rollout_data.explored))

            # Entropy loss favor exploration
            if entropy is None or self.sample_entropy:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            if self.pvf_coef == 0:
                loss = self.policy_coef*policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            else:
                path_values = path_values.flatten()
                path_value_loss = F.mse_loss(rollout_data.returns, path_values)
                loss = self.policy_coef*policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.pvf_coef * path_value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self._n_updates += 1
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/explained_variance", explained_var)
        logger.record("train/entropy_loss", entropy_loss.item())
        logger.record("train/entropy", entropy.item())
        logger.record("train/policy_loss", policy_loss.item())
        logger.record("train/value_loss", value_loss.item())
        if self.pvf_coef != 0:
            logger.record("train/path_value_loss", path_value_loss.item())
        if hasattr(self.policy, "log_std"):
            logger.record("train/std", th.exp(self.policy.log_std).mean().item())


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
                actions, values, log_probs, explored = self.policy.forward(self._last_obs)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_dones, values, log_probs, explored)
            self._last_obs = new_obs
            self._last_dones = dones

        with th.no_grad():
            #Fixing episode timeout - only correct for num-steps = 1
            true_obs = [[x['s_true']] for x in infos]
            true_done = np.array([x['d_true'] for x in infos],dtype=np.bool)
            # Compute value for the last timestep
            #obs_tensor = th.as_tensor(new_obs).to(self.device)
            _, values, _, _ = self.policy.forward(true_obs)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=true_done)
        logger.record("time/env_steps", self.num_timesteps)

        callback.on_rollout_end()

        return True
