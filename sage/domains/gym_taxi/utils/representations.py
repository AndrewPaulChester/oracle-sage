

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
from sage.domains.gym_taxi.utils.config import LOCS, PREDICTABLE5
from sage.domains.utils.representations import graph_to_json, EMB_SIZE
import networkx as nx
import cv2

def env_to_json(env):
    """
    Converts taxi world state from env to json representation

    :param env: taxi world state in env format
    :return: taxi world state in json format
    """
    return graph_to_json(*env_to_graph(env))


def find_new_node(simple_graph,old_node):
    return [x for x,y in simple_graph.nodes if y['old']==old_node ]

def nextwork_to_graph(network,mapping):
    simple_graph = nx.relabel_nodes(network,mapping)
    edges = nx.to_edgelist(simple_graph)
    node_feats = np.array([(1,0,0)]*len(simple_graph.nodes))
    edge_feats = np.array([(1,0,0,1)]*(len(edges)*2))
    start = [x for (x,_,_) in edges]
    end = [y for (_,y,_) in edges]
    edge_index = np.array([start+end,end+start])
    return node_feats,edge_feats,edge_index


def env_to_graph(env):
    node_feats = np.array([v['attr'] for _,v in sorted(env.graph.nodes.items())],dtype=np.float)
    edges = nx.to_edgelist(env.graph)
    edge_feats = np.array([v['attr'] for (_,_,v) in edges],dtype=np.float)
    edge_index = np.array([[x,y] for (x,y,_) in edges]).T
    
    #mask need to be different for SAGE vs SR-DRL. In SAGE, it's probably fine to let mask be true everywhere.
    #In SR-DRL, should only be: the taxis current position and any adjacent positions, any passengers in the taxi's location, and the taxi.
    #Conveniently, this is equal to the taxis location, and all nodes which are adjacent to it. 
    #Need to be a bit careful with this because this is masking a number of actions which can be taken in normal taxi: pickup when no passenger present, dropoff when no passenger in taxi, and move into walls.
    if env.planning == False:
        mask = np.zeros(len(node_feats),dtype=np.bool)
        mask[env.taxi.location]=True
        for x in env.graph[env.taxi.location]:
            mask[x]=True
    else:
        mask = np.ones(len(node_feats),dtype=np.bool)


    global_feats = np.zeros(EMB_SIZE,dtype=np.float)
    time_left = (env.timeout-env.time)/env.timeout
    global_feats[0] = time_left

    return node_feats, edge_feats, edge_index, mask, global_feats






def env_to_image(env):
    """
    Converts taxi world state from env to image representation

    :param env: taxi world state in env format
    :return: taxi world state in image format
    """

    channels = 4
    image_size = 2 * env.size - 1
    image = np.zeros((channels, image_size, image_size), dtype=np.uint8)
    fill_map(env, image[0], image_size)

    image[1][env.taxi.location] = 1
    for p in env.passengers.values():
        if p.location !=0:
            image[2][p.location] = 1
        image[3][p.destination] = 1

    final =  np.transpose(image, (0, 2, 1))  # swapping x&y

    return resize_image(final,84)

def fill_map(env, image, image_size):
    nodes = [(2 * x, 2 * y) for (x, y) in env.graph.nodes]
    edges = [(x1 + x2, y1 + y2) for ((x1, y1), (x2, y2)) in env.graph.edges]
    for n in nodes:
        image[n] = 1
    for n in edges:
        image[n] = 1
    # for the odd,odd coordinates, need to interpolate values.
    # Will be passable if at least 3 of it's neighbours are passable
    for x in range(1, image_size, 2):
        for y in range(1, image_size, 2):
            passable_neighbours = (
                image[x - 1][y] + image[x + 1][y] + image[x][y - 1] + image[x][y + 1]
            )
            image[x][y] = 1 if passable_neighbours > 2 else 0

def resize_image(img, size):
    """
    Modifies image dimensions . 
    :param img: taxi world state in image format
    :param size: size for converted 
    :return: taxi world state in image format of specified size
    """
    resized = cv2.resize(
        np.transpose(img, (1, 2, 0)), (size, size), interpolation=cv2.INTER_AREA
    )

    if len(resized.shape) == 2:
        return np.expand_dims(resized, axis=0)

    return np.transpose(resized, (2, 0, 1))



# def json_to_image(js):
#     """
#     Converts taxi world state from json to image representation

#     :param js: taxi world state in json format
#     :return: taxi world state in image format
#     """
#     env = json.loads(js)
#     channels = 5 if len(env["fuel_stations"]) > 0 else 4
#     image_size = 2 * env["n"] - 1
#     image = np.zeros((channels, image_size, image_size), dtype=np.uint8)
#     fill_map(env, image[0], image_size)

#     image[1][tuple(2 * i for i in env["taxi"]["location"])] = 1
#     for p in env["passenger"]:
#         if not p["in_taxi"]:
#             image[2][tuple(2 * i for i in p["location"])] = 1
#         image[3][tuple(2 * i for i in p["destination"])] = 1
#     for f in env["fuel_stations"]:
#         image[4][tuple(2 * i for i in f["location"])] = f["price"]

#     return np.transpose(image, (0, 2, 1))  # swapping x&y
