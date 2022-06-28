
from functools import partial
import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
import numpy as np

from torch_geometric.data import  Batch

from torch_geometric.utils import  softmax

from sage.forks.stable_baselines3.stable_baselines3.common.policies import BasePolicy
from sage.forks.stable_baselines3.stable_baselines3.common.preprocessing import get_action_dim, is_image_space, preprocess_obs
from sage.forks.stable_baselines3.stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractor, NatureCNN, create_mlp
from sage.forks.stable_baselines3.stable_baselines3.common.type_aliases import Schedule
from sage.forks.stable_baselines3.stable_baselines3.common.utils import get_device, is_vectorized_observation
from sage.forks.stable_baselines3.stable_baselines3.common.vec_env import VecTransposeImage
from sage.forks.stable_baselines3.stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from sage.forks.stable_baselines3.stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

from sage.agent.graph_net import MultiMessagePassing
from sage.domains.utils.spaces import Autoregressive, BinaryAction, NodeAction

EMB_SIZE = 32


def get_start_indices(splits):
    splits = th.roll(splits, 1)
    splits[0] = 0

    start_indices = th.cumsum(splits, 0)
    return start_indices

def masked_segmented_softmax(energies, mask, batch_ind):
    energies = energies.clamp(-30,30) 
    energies[~mask.bool()] = -np.inf
    probs = softmax(energies, batch_ind) 

    return probs.flatten()

def segmented_sample(probs, splits):
    probs_split = th.split(probs, splits)
    samples = [th.multinomial(x, 1) for x in probs_split]
    
    return th.cat(samples)

def segmented_scatter_(dest, indices, start_indices, values):
    real_indices = start_indices + indices
    dest[real_indices] = values
    return dest

def segmented_gather(src, indices, start_indices):
    real_indices = start_indices + indices
    return src[real_indices]

def make_mask(batch):
    device = batch.mask.device
    data_splits = [sum(1 for _ in g) for _, g in itertools.groupby(batch.batch)]
    data_splits_tensor = th.tensor(data_splits,device=device)
    data_starts = get_start_indices(data_splits_tensor)

    # lst_lens = th.tensor([len(x.mask) for x in batch.to_data_list()], device=device)
    # mask_starts = data_starts.repeat_interleave(lst_lens)

    return batch.mask.flatten(), data_splits, data_starts

  # aux vars
 





class NodeExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space):
        super(NodeExtractor, self).__init__(observation_space, observation_space.node_dimension)
        self.embed_node = nn.Sequential( nn.Linear(self.features_dim, EMB_SIZE), nn.LeakyReLU() )   

    def forward(self, observations: Batch) -> Batch:
        embed_x = self.embed_node(observations.x)
        observations.x = embed_x
        return observations


class GNNExtractor(nn.Module):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(
        self, 
        edge_dim: int = 2,
        steps: int = 5,        
        activation_fn: Type[nn.Module] = nn.LeakyReLU(),
        device: Union[th.device, str] = "auto",
    ):
        super(GNNExtractor, self).__init__()
        self.gnn = MultiMessagePassing(edge_size=edge_dim,steps=steps,activation_fn=activation_fn)
        device = get_device(device)

    def forward(self, node_features: th.Tensor, global_features: th.Tensor, edge_features: th.Tensor, edge_indices: th.Tensor, batch_ind: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.gnn(node_features,global_features,edge_features,edge_indices,batch_ind,1)


class GNNPolicy(BasePolicy):
    """
    Policy class for GNN actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.LeakyReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = NodeExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(GNNPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]
            else:
                net_arch = []

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self.gnn_steps = features_extractor_kwargs.pop("gnn_steps",5)

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim
        self.edge_dim = self.observation_space.edge_dimension

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": sde_net_arch is not None,
            }

        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=dist_kwargs)

        self._build(lr_schedule)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                sde_net_arch=default_none_kwargs["sde_net_arch"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_gnn_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
  
        self.gnn_extractor = GNNExtractor(
            edge_dim=self.edge_dim,activation_fn=self.activation_fn, device=self.device,steps=self.gnn_steps
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_gnn_extractor()

        latent_dim_pi = EMB_SIZE

        # Separate features extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )

        if isinstance(self.action_space, gym.spaces.MultiDiscrete):
            self.action_net = nn.Linear(EMB_SIZE, 1)
            self.action_net2 = nn.Linear(EMB_SIZE, 1)
            self.sel_enc = nn.Sequential(nn.Linear(EMB_SIZE + 1, EMB_SIZE), nn.LeakyReLU() )
            self.a2 = GNNExtractor(self.edge_dim,steps=2)

        elif isinstance(self.action_space, NodeAction):
            self.action_net = nn.Linear(EMB_SIZE, 1)

        elif isinstance(self.action_space, Autoregressive):
            self.action_net = nn.Linear(EMB_SIZE, 1)
            self.action_net2 = self.action_dist.proba_distribution_net(latent_dim=EMB_SIZE)
            self.sel_enc = nn.Sequential(nn.Linear(EMB_SIZE + 1, EMB_SIZE), nn.LeakyReLU() )
            self.a2 = GNNExtractor(self.edge_dim,steps=2,activation_fn = self.activation_fn)

        elif isinstance(self.action_dist, CategoricalDistribution):
            self.action_net = nn.Linear(EMB_SIZE, 1)


        elif isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            latent_sde_dim = latent_dim_pi if self.sde_net_arch is None else latent_sde_dim
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, latent_sde_dim=latent_sde_dim, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        elif isinstance(self.action_dist, BernoulliDistribution):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(EMB_SIZE, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.gnn_extractor: np.sqrt(2), 
                self.action_net: 0.01,
                self.action_net2: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        batch = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(batch.global_features)
        actions, log_prob,_ = self._get_action_from_latent(batch,deterministic=deterministic)
        return actions, values, log_prob

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        batch,_ = self.extract_features(obs,self.device)

        latent_nodes, latent_global = self.gnn_extractor(batch.x, batch.global_features, batch.edge_attr, batch.edge_index,batch.batch)

        batch.x = latent_nodes
        batch.global_features = latent_global

        # Features for sde
        # latent_sde = latent_nodes
        # if self.sde_features_extractor is not None:
        #     latent_sde = self.sde_features_extractor(features)
        return batch

    def _get_action_from_latent(self, batch: Batch, latent_sde: Optional[th.Tensor] = None, deterministic: bool = False, eval_action=None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_nodes: Latent code for the individual nodes
        :param latent_global: Latent code for the whole network
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        #mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, MultiCategoricalDistribution):
            a1,pa1,data_starts,entropy1 = self._choose_node(self.action_net, batch)
            if eval_action is not None:
                a1 = eval_action[:,0].long()
            batch = self._propagate_choice(batch,a1,data_starts)
            a2,pa2,_,entropy2 = self._choose_node(self.action_net2, batch)
            if eval_action is not None:
                a2 = eval_action[:,1].long()
            
            a1_p = segmented_gather(pa1, a1, data_starts)
            a2_p = segmented_gather(pa2, a2, data_starts)
            tot_log_prob = th.log(a1_p * a2_p)

            # # convert the actions to tuples
            # a1 = a1.cpu().numpy()
            # a2 = a2.cpu().numpy()
            # a = list(zip(a1, a2))

            return th.stack((a1,a2),dim=1), tot_log_prob, entropy1*entropy2
        elif isinstance(self.action_dist, CategoricalDistribution):
            a1,pa1,data_starts,entropy = self._choose_node(self.action_net, batch)
            if eval_action is not None:
                a1 = eval_action.long()
           
    
            tot_log_prob = th.log(segmented_gather(pa1, a1, data_starts))

            # # convert the actions to tuples
            # a1 = a1.cpu().numpy()
            # a2 = a2.cpu().numpy()
            # a = list(zip(a1, a2))

            return a1, tot_log_prob,entropy
        else:
            raise ValueError("Invalid action distribution")

    def _choose_node(self, action_net, batch: Batch, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        x_a1 = action_net(batch.x).flatten()

        mask, data_splits, data_starts = make_mask(batch)
        # decode first action
        # x_a1, _ = self.a_1(x, xg, batch.edge_attr, batch.edge_index, batch_ind, batch.num_graphs)
        
        # if self.states:
        #     mask_a1, mask_starts_a1 = make_mask(tuple([list(range(1,n_obj))]*n_env))
        # else:
        #     mask_a1, mask_starts_a1 = make_mask(free_boxes)        # only the free boxes can be selected as a1

        p_a1 = masked_segmented_softmax(x_a1, mask, batch.batch)  
        a1 = segmented_sample(p_a1, data_splits) 

        n = a1.shape[0]
        masked_probs = p_a1[mask]
        log_probs = th.log(masked_probs)
        entropy = (-masked_probs*log_probs).sum()/n


        return a1,p_a1,data_starts, entropy


    def _propagate_choice(self, batch: Batch, choice: th.Tensor, data_starts: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        selected_ind = th.zeros(len(batch.x), 1, device=self.device)
        segmented_scatter_(selected_ind, choice, data_starts, 1.0)

        # decode second action
        x = th.cat((batch.x, selected_ind), dim=1)
        x = self.sel_enc(x) # 33 -> 32 
        x, xg = self.a2(x, batch.global_features, batch.edge_attr, batch.edge_index, batch.batch, batch.num_graphs)

        batch.x = x
        batch.global_features = xg

        # update mask for from action (depends on action specifics) TODO: generalise this or abstract out.
        batch.mask[:,0] = True # can always move to ground
        r = th.arange(len(choice),dtype=th.long,device=choice.device)
        batch.mask[r,choice]=False # can't move to self

        return batch
        

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        batch = self._get_latent(observation)
        actions, _, _ = self._get_action_from_latent(batch,deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        batch = self._get_latent(obs)
        _, log_prob, entropy = self._get_action_from_latent(batch,eval_action=actions)
        values = self.value_net(batch.global_features)
        return values, log_prob, entropy