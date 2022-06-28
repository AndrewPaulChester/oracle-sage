
from functools import partial
import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from copy import deepcopy

import gym
import torch as th
from torch import nn
import numpy as np

from torch_geometric.data import  Data,Batch

from torch_geometric.utils import  softmax

from sage.forks.stable_baselines3.stable_baselines3.common.policies import BasePolicy
from sage.forks.stable_baselines3.stable_baselines3.common.preprocessing import get_action_dim, is_image_space, preprocess_obs
from sage.forks.stable_baselines3.stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractor, NatureCNN, create_mlp
from sage.forks.stable_baselines3.stable_baselines3.common.type_aliases import Schedule
from sage.forks.stable_baselines3.stable_baselines3.common.utils import get_device, is_vectorized_observation
from sage.forks.stable_baselines3.stable_baselines3.common.vec_env import VecTransposeImage
from sage.forks.stable_baselines3.stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from sage.forks.stable_baselines3.stable_baselines3.common import logger
from sage.forks.stable_baselines3.stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from sage.forks.stable_baselines3.stable_baselines3.common.utils import get_linear_fn

from sage.agent.graph_net import MultiMessagePassing
from sage.domains.utils.spaces import BinaryAction
from sage.domains.gym_nle.utils.representations import encode_actions

from sage.agent.graph_policy import get_start_indices, masked_segmented_softmax,  segmented_scatter_, segmented_gather, make_mask, NodeExtractor, GNNExtractor, EMB_SIZE, GNNPolicy

def segmented_sample(probs, splits, num_samples):
    """
    Given n actors and k num_planning_action, select top k actions per actor.


    :return: k,n tensor of actions chosen.
    """
    probs_split = th.split(probs, splits)
    samples = [th.multinomial(x, num_samples) for x in probs_split]
    
    return th.transpose(th.stack(samples),0,1)

def segmented_mask(mask, splits, actions):
    """
    Given k,n actions, and n masks, determine which actions have which masked values.


    :return: k,n tensor of actions chosen.
    """
    masks = th.zeros_like(actions)
    mask_split = th.split(mask, splits)
    for i,(a,m) in enumerate(zip(actions.T,mask_split)):
        masks[:,i] = m[a]
    return masks


def project_actions(batch, actions):
    """
    Projects a batch of states and actions forward into the future. 

    :param batch: batch object representing current world states for all actors
    :param actions: tensor of num_planning_action actions per actor
    :return: list of batches? 
    """
    projections = np.zeros((actions.shape[0],batch.num_graphs),dtype=object)
    for j,state in enumerate(batch.to_data_list()):
        for i,a in enumerate(actions[:,j]):
            projections[i,j]=project_action(deepcopy(state),a)
    
    batches = [Batch.from_data_list(projections[i]) for i in range(projections.shape[0])]
    
    return batches

def project_action(state, action):
    """
    Uses a planning model of some kind to project a state and action forward into the future. 
    This needs to be either some general PDDL based planning system, or be custom built for each domain.

    This current implementation is designed only for tradeoff world v0

    :param state: current environment state as graph
    :param action: some goal 
    :return: projected environment state as graph
    """
    node_feats = state.x
    edge_index = state.edge_index

    to_collect = []
    next_node = action
    while next_node > 1:
        to_collect.append(next_node)
        edge_id = (edge_index[1,:] == next_node).nonzero(as_tuple=True)[0]
        next_node = edge_index[0,edge_id]

    if to_collect:
        ids = th.tensor(to_collect)
        pos = node_feats[:,2][ids].sum()
        neg = node_feats[:,3][ids].sum()
        node_feats[:,2:4][ids] = 0
        node_feats[0,2] = pos
        node_feats[0,3] = neg


    
    edge_index[1,0] = action

    return state #Data(x=node_feats,edge_index=edge_index,edge_attr=state.edge_attr) #the changes have happened in place so can just return original object.

def select_action(actions, projected_values):
    """
    Choose best action based on the projected values. 

    :param actions: tensor of num_planning_action actions per actor
    :param projected_values: projected values for each action
    :return: Best action per actor, and their corresponding values
    """
    #max over projected values to get indices, then take those elements from actions.
    values, indexes = th.max(projected_values,1)
    return (th.gather(actions,0,indexes.unsqueeze(0)).squeeze(), values)

class PathValueNet(nn.Module):
    def __init__(self,layer_norm):
        super(PathValueNet, self).__init__()
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln1 = nn.LayerNorm(EMB_SIZE)
            self.ln2 = nn.LayerNorm(EMB_SIZE)
        self.path_value_net = nn.Linear(EMB_SIZE*2, 1)
        
    def forward(self,s1,s2):
        if self.layer_norm:
            s1 = self.ln1(s1)
            s2 = self.ln1(s2)
        return self.path_value_net(th.cat((s1, s2),1))    

class GNNFeedbackPolicy(GNNPolicy):
    """
    Policy class for GNN actor-critic algorithms with feedback (has both policy and value prediction).

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
        num_planning_choices: int = 3,
        exploration_initial_eps: float = 0.5,
        exploration_final_eps: float = 0.01,
        exploration_fraction: float = 0.1,
        shared_gnn: bool = False,
        layer_norm: bool = False,
    ):

        self.num_planning_choices = num_planning_choices
        self.shared_gnn = shared_gnn

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )


        super(GNNFeedbackPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        
        self.path_value_net = PathValueNet(layer_norm)

        if self.shared_gnn:
            self.gnn_extractor2 = self.gnn_extractor
        else:            
            self.gnn_extractor2 = GNNExtractor(
                edge_dim=self.edge_dim,activation_fn=self.activation_fn, device=self.device,steps=self.gnn_steps
            )

    def _on_step(self, current_progress_remaining) -> None:
        """
        Update the exploration rate.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
  
        self.exploration_rate = self.exploration_schedule(current_progress_remaining)
        logger.record("rollout/exploration_rate", self.exploration_rate)
    
    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        batch, symbolic_batch = self._get_latent(obs)
        # Evaluate the values for the given observations
        values = self.value_net(batch.global_features)
        actions, log_prob, projected_values, _, explored = self._get_action_from_latent(batch, symbolic_batch, deterministic=deterministic) #This now needs to have access to the symbolic observations, as well as embedded ones
        return actions, values, log_prob, explored #now returning whether the action was an exploratory action.

    def _get_latent(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Get the latent code (i.e., activations of the last layer of each network)
        for the different networks.

        :param obs: Observation
        :return: Latent codes
            for the actor, the value function and for gSDE function
        """
        # Preprocess the observation if needed
        batch, symbolic_batch = self.extract_features(obs, self.device)

        latent_nodes, latent_global = self.gnn_extractor(batch.x, batch.global_features, batch.edge_attr, batch.edge_index, batch.batch) #this does a first pass of GNN.

        batch.x = latent_nodes
        batch.global_features = latent_global

        # Features for sde
        # latent_sde = latent_nodes
        # if self.sde_features_extractor is not None:
        #     latent_sde = self.sde_features_extractor(features)
        return batch, symbolic_batch

    def _get_action_from_latent(self, batch: Batch, symbolic_batch: Batch, latent_sde: Optional[th.Tensor] = None, deterministic: bool = False, eval_action=None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_nodes: Latent code for the individual nodes
        :param latent_global: Latent code for the whole network
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        #mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, MultiCategoricalDistribution):
            return self._get_edge_action(batch, symbolic_batch, eval_action)
           
        elif isinstance(self.action_dist, CategoricalDistribution):
            return self._get_node_action(batch, symbolic_batch, eval_action)
        else:
            raise ValueError("Invalid action distribution")

    def _get_edge_action(self, batch, eval_action):
            a1,pa1,data_starts = self._choose_node(self.action_net, batch)
            if eval_action is not None:
                a1 = eval_action[:,0].long()
            batch = self._propagate_choice(batch,a1,data_starts)
            a2,pa2,_ = self._choose_node(self.action_net2, batch)
            if eval_action is not None:
                a2 = eval_action[:,1].long()
            
            a1_p = segmented_gather(pa1, a1, data_starts)
            a2_p = segmented_gather(pa2, a2, data_starts)
            tot_log_prob = th.log(a1_p * a2_p)

            # # convert the actions to tuples
            # a1 = a1.cpu().numpy()
            # a2 = a2.cpu().numpy()
            # a = list(zip(a1, a2))

            return th.stack((a1,a2),dim=1), tot_log_prob

    def _get_node_action(self, batch, symbolic_batch, eval_action):
        a1,pa1,data_starts,entropy = self._choose_node(self.action_net, batch)
        k,n = a1.shape
        if eval_action is None:
            projected_values = []
            projected_batches = project_actions(symbolic_batch, a1)
                
            #actions can now be encoded, so no longer work as indices directly into the probability array, need to strip direction
            probs = segmented_gather(pa1, th.remainder(a1,1000), data_starts)
            probs1,index1 = probs.max(0)
            probs3,index3 = probs.min(0)
            probs2 = probs.sum(0)-probs1-probs3
            logger.record_mean("action_selection/prob_choice_1", probs1.mean().item())
            logger.record_mean("action_selection/prob_choice_2", probs2.mean().item())
            logger.record_mean("action_selection/prob_choice_3", probs3.mean().item())

            for projected_batch in projected_batches:
                projected_batch = self.features_extractor(projected_batch)
                _,projected_features = self.gnn_extractor2(projected_batch.x, projected_batch.global_features, projected_batch.edge_attr, projected_batch.edge_index,projected_batch.batch)
                projected_values.append(self.path_value_net(batch.global_features, projected_features))
            
            projected_values = th.cat(projected_values,1)
            selected_actions,selected_values = select_action(a1, projected_values)
            select_random = th.zeros_like(selected_values,device=self.device)
            
            chosen_probs = segmented_gather(pa1, th.remainder(selected_actions,1000), data_starts)
            chosen_index=(a1==selected_actions).max(0).indices
            choice1 = (chosen_index==index1).sum()/n
            choice3 = (chosen_index==index3).sum()/n
            choice2 = 1-choice1-choice3
            logger.record_mean("action_selection/choice_likelihood", (chosen_probs/probs1).mean().item())
            logger.record_mean("action_selection/choice_1_selected", choice1.item())
            logger.record_mean("action_selection/choice_2_selected", choice2.item())
            logger.record_mean("action_selection/choice_3_selected", choice3.item())


            if self.exploration_rate > 0: 
                k,n = a1.shape
                rand_index = th.randint(k,(n,),device=self.device)
                column_index = th.arange(n,device=self.device)
                random_actions = a1[rand_index,column_index]
                random_values = projected_values[column_index,rand_index]
                select_random = th.rand((n,),device=self.device) < self.exploration_rate
                selected_actions = th.where(select_random, random_actions, selected_actions) 
                selected_values = th.where(select_random, random_values, selected_values) 



        else:
            selected_actions = eval_action.long()
            projected_batches = project_actions(symbolic_batch, selected_actions.unsqueeze(0))
            projected_batch = self.features_extractor(projected_batches[0])
            _,projected_features = self.gnn_extractor2(projected_batch.x, projected_batch.global_features, projected_batch.edge_attr, projected_batch.edge_index,projected_batch.batch)
            selected_values = self.path_value_net(batch.global_features, projected_features)
            select_random = th.zeros_like(selected_values,device=self.device)
        

        tot_log_prob = th.log(segmented_gather(pa1, selected_actions, data_starts))

        # # convert the actions to tuples
        # a1 = a1.cpu().numpy()
        # a2 = a2.cpu().numpy()
        # a = list(zip(a1, a2))

        return selected_actions, tot_log_prob, selected_values, entropy, select_random

    def _choose_node(self, action_net, batch: Batch, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        x_a1 = action_net(batch.x).flatten()

        mask, data_splits, data_starts = make_mask(batch)

        p_a1 = masked_segmented_softmax(x_a1, mask, batch.batch)  
      
        a1 = segmented_sample(p_a1, data_splits,self.num_planning_choices) 
      
        n = a1.shape[1]
        masked_probs = p_a1[mask.bool()]
        log_probs = th.log(masked_probs)
        entropy = (-masked_probs*log_probs).sum()/n

        return a1,p_a1,data_starts,entropy

    def _choose_global(self, action_net, batch: Batch, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """

        xg = action_net(batch.global_features)
        dist = self.action_dist.proba_distribution(xg)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        prob = th.exp(log_probs)

        return actions,prob

    def _batch_clone(self, batch):
        new_batch = Batch()
        new_batch.x = batch.x.clone()
        new_batch.edge_attr = batch.edge_attr.clone()
        new_batch.batch = batch.batch.clone()
        new_batch.edge_index = batch.edge_index.clone()
        new_batch.global_features = batch.global_features.clone()
        new_batch.mask = batch.mask.clone()
        new_batch.ptr = batch.ptr.clone()
        return new_batch

    def _choose_hybrid(self, action_net, batch: Batch, latent_sde: Optional[th.Tensor] = None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :param latent_sde: Latent code for the gSDE exploration function
        :return: Action distribution
        """
        a1,pa1,data_starts,entropy = self._choose_node(self.action_net, batch)

        mask, data_splits, data_starts = make_mask(batch)
 

        mask_a1 = segmented_mask(mask,data_splits,a1)
        directions_needed = th.nonzero((mask_a1 == 8))
        a2 = th.zeros_like(a1)
        pa2 = th.ones((3,len(pa1)),dtype=pa1.dtype,device=pa1.device)
        for i in range (self.num_planning_choices): 
            if i in directions_needed[:,0]: #only do gnn step if there is an action in this choice set that needs propagating
                temp_batch = self._propagate_choice(self._batch_clone(batch),a1[i],data_starts)
                a2[i],probs = self._choose_global(self.action_net2, temp_batch)
                pa2[i][a1[i]+data_starts]=probs


        # adjust the probabilities according to the secondary action choice. 
        # We take the min in this dimension because:
        #    1. the probability of a hybrid action is always less than the original
        #    2. we only calculate the hybrid action probability if it is chosen
        #    3. If a hybrid action is necessary for that choice, then the original action probability is meaningless by itself.
        #    4. If that action is chosen, it will only be chosen with one value of a secondary action.
        p_a,_ = th.where((mask == 8),pa1.unsqueeze(0)*pa2,pa1).min(dim=0) 

        a = encode_actions(a1,a2,(mask_a1 == 8))

        """
        batch = self._propagate_choice(batch,a1,data_starts)
        a2,pa2,_ = self._choose_node(self.action_net2, batch)
        if eval_action is not None:
            a2 = eval_action[:,1].long()
        
        a1_p = segmented_gather(pa1, a1, data_starts)
        a2_p = segmented_gather(pa2, a2, data_starts)
        tot_log_prob = th.log(a1_p * a2_p)
        """
        # Trying to select a direction if required, and include in a1.
        # This process consists of a number of steps:
        #    1. determine which actions need a direction
        #    2. propagate those actions into their original graphs
        #    3. pass that propagated representation to the shared GNN to select the direction
        #    4. calculate the probabilities of the new direction
        #    5. combine choices and probabilities into a single set of actions.



        n = a1.shape[1]
        masked_probs = p_a[mask.bool()]
        log_probs = th.log(masked_probs)
        entropy = (-masked_probs*log_probs).sum()/n

        return a,p_a,data_starts,entropy

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
        x, xg = self.a2(x, batch.global_features, batch.edge_attr, batch.edge_index, batch.batch)

        batch.x = x
        batch.global_features = xg


        return batch
        

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        batch, symbolic_batch = self._get_latent(observation)
        actions, _, _, _, _ = self._get_action_from_latent(batch, symbolic_batch, deterministic=deterministic)
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
        batch, symbolic_batch = self._get_latent(obs)
        _, log_prob, projected_values, entropy, _  = self._get_action_from_latent(batch, symbolic_batch, eval_action=actions)
        values = self.value_net(batch.global_features)
        return values, log_prob, entropy, projected_values