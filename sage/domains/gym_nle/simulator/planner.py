from os import remove
import numpy as np
import torch as th
import networkx as nx

from typing import List, NamedTuple

from sage.domains.gym_nle.utils.representations import decode_action, encode_action, ACTION_PICKUP, ACTION_DESCEND

class Player(NamedTuple):
    node: int
    location: int

class Item(NamedTuple):
    node: int
    location: int
class State(NamedTuple):
    graph: int
    player: Player
    items: List[Item]
    inventory: List[Item]

#[isTerrain,isCreature,isItem,isDownstairs,isPlayer]

class Planner():
    def __init__(self,glyphs):
        self.glyphs = glyphs


    def plan(self,graph,goal):
        goal, direction = decode_action(goal)
        state = self.graph_to_networkx(graph)
        if goal==state.player.node:
                projection = graph
                actions = []
                
        elif goal in [i.node for i in state.items]:
            projection, actions = pickup_item(graph,state,goal)
        elif goal in [i.node for i in state.inventory]: # do nothing with inventory items for now.
            projection = graph
            # item = self.glyphs.get_glyph_from_encoding(graph.x[goal])
            # action = self.glyphs['default_action'][item]
            actions = [encode_action(goal,direction)]
        else:
            projection, actions = move(graph,state,goal)

        if actions == []:
            actions = [state.player.location]
        
        return increment_timer(projection,actions)

    def graph_to_networkx(self,graph):
        nodes = graph.x.cpu().numpy()
        edges = graph.edge_index.cpu().numpy()
        edge_attr = graph.edge_attr.cpu().numpy()

        
        items=[]
        inventory = []
        G = nx.Graph()
        for i,encoding in enumerate(nodes):
            if encoding[0]:
                node_type = "location" 
            elif encoding[1]:
                glyph = self.glyphs.get_glyph_from_encoding(encoding)
                if glyph in (329,337):
                    node_type = "player" 
                    location = edges[1,np.logical_and(edges[0]==i,edge_attr[:,1]==1)]
                    player=Player(i,location.item())
                else:
                    node_type = "creature" 
            elif encoding[2]:
                node_type = "item" 
                location = edges[1,np.logical_and(edges[0]==i,edge_attr[:,1]==1)]
                if len(location): #if it is on the ground
                    item=Item(i,location.item())
                    items.append(item)
                else:
                    item=Item(i,0)
                    inventory.append(item)
            else:
                continue
                #raise ValueError("Invalid node is neither location, taxi or passenger.")
            G.add_node(i,type=node_type)

        orthogonal_edge_indices = (edge_attr[:,0]==1)&((edge_attr[:,4]==0)|(edge_attr[:,5]==0))
        diagonal_edge_indices = (edge_attr[:,0]==1)&(np.logical_not(orthogonal_edge_indices))
        G.add_edges_from(edges.T[orthogonal_edge_indices],cost=1)
        G.add_edges_from(edges.T[diagonal_edge_indices],cost=1.4)
    
        return State(G,player,items,inventory)

def move(graph,state,goal):
    actions = find_path_to(state,state.player.location,goal)
    if actions:
        move_player(graph,state.player.node,goal)
    return graph, actions

def find_path_to(state,start,end):
    try:
        path =  nx.shortest_path(state.graph,start,end,weight="cost")
    except nx.NetworkXNoPath:
        return []
    return [p for p in path[1:]] 

def move_player(graph,player,node):
    graph.edge_index[1,graph.edge_index[0]==player] = node
    player_edge_backwards_index = th.logical_and(graph.edge_index[1]==player,graph.edge_attr[:,1]==-1)
    graph.edge_index[0,player_edge_backwards_index] = node
    


def pickup_item(graph,state,goal):
    item = [i for i in state.items if i.node==goal][0]
    moves = find_path_to(state,state.player.location,item.location)
    move_player(graph,state.player.node,item.location)
    collect_item(graph,state.player.node,item.node)
    return  graph,moves + [ACTION_PICKUP]

def collect_item(graph,player,item):
    item_edge_forwards_index = graph.edge_index[0]==item
    graph.edge_index[1,item_edge_forwards_index] = player
    item_edge_backwards_index = th.logical_and(graph.edge_index[1]==item,graph.edge_attr[:,1]==-1)
    graph.edge_index[0,item_edge_backwards_index] = player
    graph.edge_attr[item_edge_forwards_index,[1,2]] = graph.edge_attr[item_edge_forwards_index,[2,1]]
    graph.edge_attr[item_edge_backwards_index,[1,2]] = graph.edge_attr[item_edge_backwards_index,[2,1]]

def increment_timer(projection,actions):
    projection.global_features[0,20] = projection.global_features[0,20] + len(actions) 
    return projection, actions