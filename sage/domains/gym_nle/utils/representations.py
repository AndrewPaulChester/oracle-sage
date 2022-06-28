

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
#from nle import nethack
import torch as th

ACTION_PICKUP = 9049
ACTION_DESCEND = 9016


def encode_actions(a1,a2,mask):
    """
    Given k,n actions, k,n directions, and a k,n mask, encode action and directions into a single value where needed.

    :return: k,n tensor of combined actions.
    """
    return th.where(mask, a1+1000*(a2+1), a1)


def encode_action(a1,a2):
    """
    Given k,n actions, k,n directions, and a k,n mask, encode action and directions into a single value where needed.

    :return: k,n tensor of combined actions.
    """
    a2 = 0 if a2 is None else a2
    return a1+1000*(a2+1)

def decode_action(action):
    if action < 1000: # if under 1000, then it's a plain node id.
        return action, None
    if action >= 9000: #if over 9000, then it's a direct action
        return None, action-9000 
    else: #if it's from 1000-8999, then its an encoded node id + direction.
        direction = int(action/1000)-1
        action = action % 1000 
        return action, direction


def env_to_json(env,glyph_db,planning = False):
    """
    Converts taxi world state from env to json representation

    :param env: taxi world state in env format
    :return: taxi world state in json format
    """
    return graph_to_json(*env_to_graph(env,glyph_db,planning))


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


def env_to_graph(env,glyph_db,planning = False):
    glyphs = env["glyphs"]
    uniques = np.unique(glyphs)
    for u in sorted(set(uniques).difference(glyph_db.KNOWN_GLYPHS)):
        print(f"tile ID: {u} not known")
        
    occupied = np.transpose(np.isin(glyphs,glyph_db.BLOCKING_GLYPHS,invert=True).nonzero())

    mapping = {tuple(c):i+1 for i,c in enumerate(occupied)}

    graph = nx.DiGraph()
    graph.add_nodes_from(mapping.values()) #create graph of terrain nodes.


    next_node = len(mapping)+1
    for ((x,y),id) in mapping.items():

        #add terrain edges
        possible_neighbours = [(x+1,y+1),(x+1,y),(x+1,y-1),(x,y-1)] #Only need to check half of adjacent squares because edges are added in both directions,(x-1,y-1),(x-1,y),(x-1,y+1),(x,y+1)]
        for x2,y2 in possible_neighbours:
            id2 = mapping.get((x2,y2),None)
            if id2 is not None:
                graph.add_edge(id,id2,attr=[1,0,0,0,x-x2,y-y2])
                graph.add_edge(id2,id,attr=[1,0,0,0,x2-x,y2-y])

        glyph = glyphs[x,y]
        #add terrain node attributes
        if glyph not in glyph_db.TERRAIN_GLYPHS:
            graph.nodes[id]['attr']=glyph_db.get_glyph_encoding(glyph_db.glyph_db.index[glyph_db.glyph_db['name'] =='floor'][0])

            if glyph in(329,337): #add player node - always id = 0 
                add_node = 0
                player_pos=id
            else:
                add_node=next_node
                next_node+=1
            graph.add_node(add_node,attr = glyph_db.get_glyph_encoding(glyph))
            graph.add_edge(id,add_node,attr=[0,-1,0,0,0,0])
            graph.add_edge(add_node,id,attr=[0,1,0,0,0,0])
        else:
            graph.nodes[id]['attr']=glyph_db.get_glyph_encoding(glyph)


    directional = []
    if glyph_db.tileset == "minimal":
        for item,chars in zip(env['inv_glyphs'],env['inv_strs']):
            if item == 5976:
                break

            if ((glyph_db.get_default_action(item) in [66,75]) #default action is inherently directional
                or (item in [2131])): #or item is directional (horn of cold/fire)
                directional.append(next_node)

            graph.add_node(next_node,attr = glyph_db.get_glyph_encoding(item))
            #Current assumption from minimal playtesting is that brackets in the inventory description mean they are equipped
            graph.add_edge(0,next_node,attr=[0,0,-1,0,0,0])
            graph.add_edge(next_node,0,attr=[0,0,1,0,0,0])
            
            next_node+=1




    node_feats = np.array([v['attr'] for _,v in sorted(graph.nodes.items())],dtype=np.float)
    edges = nx.to_edgelist(graph)
    edge_feats = np.array([v['attr'] for (_,_,v) in edges],dtype=np.float)
    edge_index = np.array([[x,y] for (x,y,_) in edges]).T

    if planning == False:
        mask = np.zeros(len(node_feats),dtype=np.bool)
        mask[player_pos]=False
        for x in graph[player_pos]:
            if x > 0:
                mask[x]=True
    else:
        if glyph_db.tileset == "minimal": #lava domain
            mask = np.ones(len(node_feats),dtype=np.uint32)
            mask[np.array(directional,dtype=np.int)]=0 #some items require directions
            mask[0]=0 #can't select self
        else: #movement only
            mask = np.ones(len(node_feats),dtype=np.bool)
            mask[player_pos]=False #can't do no-op
            mask[0]=False #can't select self
        mask[node_feats.sum(axis=1)==1]=0 # can't select unknown nodes.


    global_feats = np.zeros(EMB_SIZE,dtype=np.float)
    global_feats[0:25] = env['blstats'][:-1]

    return node_feats, edge_feats, edge_index, mask, global_feats



def convert_condition_mask(n):
    return np.array([x for x in bin(n)[2:].zfill(13)])