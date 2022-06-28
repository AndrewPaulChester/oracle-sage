
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
from sage.domains.utils.spaces import Autoregressive, BinaryAction

from sage.agent.graph_policy import get_start_indices, masked_segmented_softmax,  segmented_scatter_, segmented_gather, make_mask, NodeExtractor, GNNExtractor, EMB_SIZE, GNNPolicy
from sage.agent.graph_feedback_policy import GNNFeedbackPolicy

def segmented_sample(probs, splits, num_samples):
    """
    Given n actors and k num_planning_action, select top k actions per actor.


    :return: k,n tensor of actions chosen.
    """
    probs_split = th.split(probs, splits)
    samples = [th.multinomial(x, num_samples) for x in probs_split]
    
    return th.transpose(th.stack(samples),0,1)


def project_actions(batch, actions, planner):
    """
    Projects a batch of states and actions forward into the future. 

    :param batch: batch object representing current world states for all actors
    :param actions: tensor of num_planning_action actions per actor
    :return: list of batches? 
    """
    projections = np.zeros((actions.shape[0],batch.num_graphs),dtype=object)
    plans = np.zeros((actions.shape[0],batch.num_graphs),dtype=object)
    for j,state in enumerate(batch.to_data_list()):
        for i,a in enumerate(actions[:,j]):
            projections[i,j],plans[i,j]=planner.plan(deepcopy(state),a.item())
    
    batches = [Batch.from_data_list(projections[i]) for i in range(projections.shape[0])]
    
    return batches, plans

def select_action(actions, projected_values, plans):
    """
    Choose best action based on the projected values. 

    :param actions: tensor of num_planning_action actions per actor
    :param projected_values: projected values for each action
    :return: Best action per actor, and their corresponding values
    """
    #max over projected values to get indices, then take those elements from actions.
    values, indexes = th.max(projected_values,1)
    return (th.gather(actions,0,indexes.unsqueeze(0)).squeeze(), values, np.take_along_axis(plans,indexes.unsqueeze(0).cpu().numpy(),0)[0])

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

class GNNPlanFeedbackPolicy(GNNFeedbackPolicy):
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
        actions, log_prob, projected_values, _, explored, plans = self._get_action_from_latent(batch, symbolic_batch, deterministic=deterministic) #This now needs to have access to the symbolic observations, as well as embedded ones
        return actions, values, log_prob, explored, plans #now returning whether the action was an exploratory action.

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
           
        elif isinstance(self.action_space, Autoregressive):
            a,pa,data_starts,entropy = self._choose_hybrid(self.action_net, batch)
        elif isinstance(self.action_dist, CategoricalDistribution):
            a,pa,data_starts,entropy = self._choose_node(self.action_net, batch)
            
        else:
            raise ValueError("Invalid action distribution")
        return self._choose_top_action(a,pa,data_starts,entropy,batch, symbolic_batch, eval_action)

    def _choose_top_action(self, a,pa,data_starts,entropy, batch, symbolic_batch, eval_action):
        
        k,n = a.shape
        select_random = th.zeros((n,),device=self.device)

        if eval_action is None:


            projected_values = []
            projected_batches, plans = project_actions(symbolic_batch, a,self.observation_space.planner)
                
            if self.num_planning_choices == 1:
                selected_actions = a[0]
                selected_plans = plans[0]
                selected_values = th.zeros((n,),device=self.device)
                logger.record_mean("action_selection/prob_choice_1", segmented_gather(pa, th.remainder(selected_actions,1000), data_starts).mean().item())

            else:

                #actions can now be encoded, so no longer work as indices directly into the probability array, need to strip direction
                probs = segmented_gather(pa, th.remainder(a,1000), data_starts)
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
                selected_actions,selected_values, selected_plans = select_action(a, projected_values, plans)

                chosen_probs = segmented_gather(pa, th.remainder(selected_actions,1000), data_starts)
                chosen_index=(a==selected_actions).max(0).indices
                choice1 = (chosen_index==index1).sum()/n
                choice3 = (chosen_index==index3).sum()/n
                choice2 = 1-choice1-choice3
                logger.record_mean("action_selection/choice_likelihood", (chosen_probs/probs1).mean().item())
                logger.record_mean("action_selection/choice_1_selected", choice1.item())
                logger.record_mean("action_selection/choice_2_selected", choice2.item())
                logger.record_mean("action_selection/choice_3_selected", choice3.item())

                if self.exploration_rate > 0: 
                    rand_index = th.randint(k,(n,),device=self.device)
                    column_index = th.arange(n,device=self.device)
                    
                    random_actions = a[rand_index,column_index]
                    random_values = projected_values[column_index,rand_index]
                    random_plans = plans[rand_index.cpu(),column_index.cpu()]
                    
                    select_random = th.rand((n,),device=self.device) < self.exploration_rate

                    selected_actions = th.where(select_random, random_actions, selected_actions) 
                    selected_values = th.where(select_random, random_values, selected_values) 
                    selected_plans = np.where(select_random.cpu(), random_plans, selected_plans) 


        else:
            selected_actions = eval_action.long()            
            projected_batches, _ = project_actions(symbolic_batch, selected_actions.unsqueeze(0),self.observation_space.planner)
            projected_batch = self.features_extractor(projected_batches[0])
            _,projected_features = self.gnn_extractor2(projected_batch.x, projected_batch.global_features, projected_batch.edge_attr, projected_batch.edge_index,projected_batch.batch)
            selected_values = self.path_value_net(batch.global_features, projected_features)
            selected_plans = None
        

        tot_log_prob = th.log(segmented_gather(pa, th.remainder(selected_actions,1000), data_starts))

        # # convert the actions to tuples
        # a1 = a1.cpu().numpy()
        # a2 = a2.cpu().numpy()
        # a = list(zip(a1, a2))

        return selected_actions, tot_log_prob, selected_values, entropy, select_random, selected_plans

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        batch, symbolic_batch = self._get_latent(observation)
        actions, _, _, _, _, _ = self._get_action_from_latent(batch, symbolic_batch, deterministic=deterministic)
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
        _, log_prob, projected_values, entropy, _, _  = self._get_action_from_latent(batch, symbolic_batch, eval_action=actions)
        values = self.value_net(batch.global_features)
        return values, log_prob, entropy, projected_values