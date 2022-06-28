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
import json
from torch_geometric.data import Data, Batch
import torch as th
import numpy as np

EMB_SIZE=32

def json_default(obj):
    """
    Converts numpy types to standard types for serialisation into JSON.
    :param obj: object to be converted
    :returns: object as a standard type
    :raises TypeError: if the object is not a numpy type
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError

def json_to_graph(js):
    """
    General purpose converter from json to graph format, requires correct json structure as input.

    :param js: list of json objects representing vector of environments
    :return: world state in batch (multi graph) format
    """
    envs = [json.loads(j[0]) for j in js] 
    data =  [Data(x = th.as_tensor(env["node_feats"]),edge_attr= th.as_tensor(env["edge_feats"]),edge_index = th.as_tensor(env["edge_index"])) for env in envs]
    for d,env in zip(data,envs):
        d.mask = th.as_tensor(env["mask"])#.unsqueeze(0)
        d.global_features = th.as_tensor(env["global_feats"]).unsqueeze(0) #th.zeros(1,EMB_SIZE)
    batch = Batch.from_data_list(data)
    return batch

def graph_to_json(node_feats, edge_feats, edge_index, mask, global_feats):
    """
    General purpose converter from graph to json format.

    """
    return json.dumps(
    {
        "node_feats": node_feats,
        "edge_feats": edge_feats,
        "edge_index": edge_index,
        "mask": mask,
        "global_feats": global_feats,
    },
    default=json_default,
)