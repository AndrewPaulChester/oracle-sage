"""
.. module:: representations
   :synopsis: Contains functions to convert between different representations of the taxi world.
   Functions are of the form x_to_y where x and y are represenation formats, from the list below:
   env - the actual TaxiWorld simulator class itself, holds more than just the current state.
   json - a general purpose JSON serialisation of the current state. All functions should convert to/from this as a common ground.
   image - a four channel image based encoding, designed for input to CNN
   discrete - a discretised encoding only valid for a 5x5 grid, designed for standard Q-agents and compatibility with gym
   pddl - planning domain file designed for input to planner, but lacks a goal.

"""


from math import floor, ceil
import json
import numpy as np
from torch_geometric.data import Data, Batch
import torch as th
from sage.domains.utils.representations import graph_to_json, EMB_SIZE

EMB_SIZE=32

def env_to_json(env):
    """
    Converts taxi world state from env to json representation

    :param env: taxi world state in env format
    :return: taxi world state in json format
    """
    return graph_to_json(*env_to_graph(env))
    
def env_to_graph(env):
    grid = np.arange(2,env.n*env.k+2,dtype=np.int).reshape(env.k,env.n)
    shift = np.roll(grid.copy(),1,-1)
    shift[:,0]=1 # starting room
    edge_index = np.stack((shift.flatten(),grid.flatten()))
    edge_feats = np.ones((edge_index.shape[1],1))
    
    edge_index = np.concatenate((np.array([[0],[1]]),edge_index),1) #player is in room 1
    edge_feats = np.concatenate((np.array([[0]]),edge_feats)) #


    player = np.array([[0,1,0,0]])
    start = np.array([[1,0,0,0]])
    rooms = np.stack((np.ones(env.n*env.k),np.zeros(env.n*env.k),env.gains.flatten(),env.costs.flatten())).T
    node_feats = np.concatenate((player,start,rooms),axis=0)
    
    mask = np.zeros((grid.shape),dtype=np.bool)
    mask[:,-1]=True
    mask = np.concatenate((np.zeros(2,dtype=np.bool),mask.flatten()),axis=0)

    global_feats = np.zeros(EMB_SIZE,dtype=np.float)

    return node_feats, edge_feats, edge_index, mask, global_feats
